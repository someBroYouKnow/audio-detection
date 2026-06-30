# Zero to Audio-LLM Engineer: Beginner Learning Plan

This plan takes you from Python basics to a working local audio pipeline:

```text
WAV audio -> resample/mono -> denoise -> VAD timestamps -> ASR transcription -> WER/evaluation -> optional transcript repair -> system design notes
```

You do not need a formal machine learning background. Move module by module. Do not skip the small exercises, because they build the intuition needed for the final pipeline.

## Setup First

### Skills You Should Already Have

- Basic Python: functions, classes, lists, dictionaries, virtual environments.
- Basic terminal usage: running scripts, installing packages, reading errors.
- Comfortable editing `.py` and `.md` files.

### Create Your Project Environment

From this repository folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the beginner set:

```powershell
pip install numpy scipy soundfile librosa matplotlib noisereduce webrtcvad jiwer
```

Install ASR tools later in Module 3:

```powershell
pip install faster-whisper
```

Install transcript repair tools later in Module 4:

```powershell
pip install transformers torch accelerate sentencepiece
```

### Suggested Repository Structure

Create these folders as you progress:

```text
audio-detection/
  data/
    raw/
    processed/
  outputs/
    vad/
    transcripts/
    benchmarks/
  src/
    audio_io.py
    denoise.py
    vad.py
    asr.py
    metrics.py
    repair.py
    pipeline.py
  notebooks/
  reports/
```

### Your Practice Dataset

Record 10 to 20 short `.wav` files:

- 5 clean room recordings.
- 5 recordings with fan or AC noise.
- 5 recordings with background speech or street noise.
- 2 to 5 intentionally difficult recordings with pauses, coughs, static, or distance from microphone.

Keep each clip between 5 and 30 seconds. Write a matching text transcript for each file in `data/raw/transcripts.csv`.

Example:

```csv
file,reference_text
sample_001.wav,move the robot to shelf number four
sample_002.wav,stop the engine immediately
```

## Module 1: Digital Audio Fundamentals and Signal-Level VAD

**Time:** Weeks 1-2

**Goal:** Understand how computers represent sound and build a local voice activity detector that returns speech timestamps.

### What To Learn

- Sampling rate: how many audio samples are captured per second. Speech models commonly expect 16 kHz.
- Bit depth: how precise each sample is. WebRTC VAD expects 16-bit PCM bytes.
- Channels: mono has one channel, stereo has two. Most speech pipelines use mono.
- Frames: short windows of audio. WebRTC VAD accepts 10 ms, 20 ms, or 30 ms frames.
- RMS energy: a simple loudness measurement.
- Spectrogram: a time-frequency view of audio.
- Spectral gating: estimating noise and reducing frequencies that look like background noise.
- VAD: voice activity detection, the step that decides which frames contain speech.

### Recommended Links

- Digital audio basics: https://developer.mozilla.org/en-US/docs/Web/Media/Guides/Formats/Audio_concepts
- Nyquist-Shannon sampling theorem: https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem
- WebRTC VAD Python wrapper: https://github.com/wiseman/py-webrtcvad
- WebRTC source tree: https://webrtc.googlesource.com/src/
- Audacity noise reduction manual: https://manual.audacityteam.org/man/noise_reduction.html
- Librosa audio tutorial: https://librosa.org/doc/latest/tutorial.html

### Beginner Exercises

1. Load a `.wav` file with `soundfile`.
2. Print sample rate, shape, duration, min amplitude, and max amplitude.
3. Convert stereo to mono.
4. Resample audio to 16 kHz using `librosa.resample`.
5. Plot waveform and spectrogram with `matplotlib`.
6. Apply `noisereduce.reduce_noise`.
7. Split audio into 30 ms frames.
8. Run WebRTC VAD and produce timestamp ranges.

### Assignment: Local Clean-VAD Pipeline

Create `src/vad.py` that:

- Loads a `.wav`.
- Converts it to mono.
- Resamples it to 16 kHz.
- Denoises it.
- Converts it to 16-bit PCM.
- Runs WebRTC VAD.
- Merges nearby speech frames into clean segments.
- Saves timestamps to `outputs/vad/<audio_name>.json`.

Expected JSON:

```json
{
  "audio_file": "sample_001.wav",
  "sample_rate": 16000,
  "segments": [
    {"start": 0.30, "end": 2.10},
    {"start": 2.55, "end": 4.20}
  ]
}
```

### Starter Code

```python
import json
from pathlib import Path

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
import webrtcvad


TARGET_SR = 16000


def load_audio(path):
    data, sample_rate = sf.read(path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sample_rate != TARGET_SR:
        data = librosa.resample(data.astype(np.float32), orig_sr=sample_rate, target_sr=TARGET_SR)
        sample_rate = TARGET_SR
    return data.astype(np.float32), sample_rate


def float_to_pcm16(audio):
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16).tobytes()


def frame_bytes(pcm_bytes, sample_rate, frame_ms=30):
    bytes_per_sample = 2
    frame_size = int(sample_rate * frame_ms / 1000) * bytes_per_sample
    for offset in range(0, len(pcm_bytes) - frame_size + 1, frame_size):
        yield offset, pcm_bytes[offset:offset + frame_size]


def detect_speech(path, aggressiveness=3, frame_ms=30):
    audio, sample_rate = load_audio(path)
    cleaned = nr.reduce_noise(y=audio, sr=sample_rate, prop_decrease=0.85)
    pcm = float_to_pcm16(cleaned)
    vad = webrtcvad.Vad(aggressiveness)

    segments = []
    active_start = None
    bytes_per_second = sample_rate * 2

    for offset, frame in frame_bytes(pcm, sample_rate, frame_ms):
        start_time = offset / bytes_per_second
        end_time = start_time + frame_ms / 1000
        is_speech = vad.is_speech(frame, sample_rate)

        if is_speech and active_start is None:
            active_start = start_time
        elif not is_speech and active_start is not None:
            segments.append({"start": round(active_start, 3), "end": round(end_time, 3)})
            active_start = None

    if active_start is not None:
        segments.append({"start": round(active_start, 3), "end": round(len(audio) / sample_rate, 3)})

    return {
        "audio_file": Path(path).name,
        "sample_rate": sample_rate,
        "segments": segments,
    }


if __name__ == "__main__":
    result = detect_speech("data/raw/sample_001.wav")
    Path("outputs/vad").mkdir(parents=True, exist_ok=True)
    Path("outputs/vad/sample_001.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
```

### Done When

- You can explain why 16 kHz mono is used.
- You can identify speech and silence in a waveform.
- Your script returns reasonable speech timestamps for at least 5 local files.

## Module 2: Linear Algebra and Gradient Descent From Scratch

**Time:** Weeks 3-4

**Goal:** Understand the math engine behind machine learning before using deep learning frameworks.

### What To Learn

- Scalars, vectors, matrices, tensors.
- Dot product as similarity.
- Matrix multiplication as transforming data.
- Mean squared error.
- Partial derivatives.
- Chain rule.
- Gradient descent.
- Learning rate.
- Train/validation split.

### Recommended Links

- 3Blue1Brown Essence of Linear Algebra: https://www.3blue1brown.com/topics/linear-algebra
- 3Blue1Brown Neural Networks: https://www.3blue1brown.com/topics/neural-networks
- MIT Linear Algebra lectures: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- Khan Academy multivariable calculus: https://www.khanacademy.org/math/multivariable-calculus
- Deep Learning Book, optimization chapter: https://www.deeplearningbook.org/contents/optimization.html

### Beginner Exercises

1. Create vectors with NumPy.
2. Compute dot products manually and with `np.dot`.
3. Multiply a matrix by a vector.
4. Plot a line `y = wx + b`.
5. Compute mean squared error.
6. Change `w` and `b` manually and observe the loss.
7. Implement gradient descent for one feature.
8. Extend it to multiple features.

### Assignment: Matrix Optimizer From Scratch

Create `src/optimizer.py` that trains a linear regression model using only NumPy.

```python
import numpy as np


class NaiveOptimizer:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)

        for _ in range(self.epochs):
            y_predicted = np.dot(X, self.weights) + self.bias
            error = y_predicted - y

            loss = np.mean(error ** 2)
            self.loss_history.append(loss)

            dw = (2 / num_samples) * np.dot(X.T, error)
            db = (2 / num_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


if __name__ == "__main__":
    X = np.array([[1], [2], [3], [4]], dtype=np.float32)
    y = np.array([3, 5, 7, 9], dtype=np.float32)

    model = NaiveOptimizer(learning_rate=0.01, epochs=2000)
    model.fit(X, y)

    print(f"Weight: {model.weights[0]:.2f}")
    print(f"Bias: {model.bias:.2f}")
    print(f"Final loss: {model.loss_history[-1]:.6f}")
```

### Done When

- Your model learns approximately `weight = 2`, `bias = 1`.
- You can explain what happens when the learning rate is too small or too large.
- You understand why neural networks are trained by minimizing loss.

## Module 3: Deep Learning for Speech, Transformers, and Whisper

**Time:** Weeks 5-6

**Goal:** Move from signal processing to modern ASR and benchmark Whisper-style models.

### What To Learn

- Log-Mel spectrograms.
- Tokenization.
- Transformer encoder-decoder architecture.
- Attention at a high level.
- Beam search.
- Quantization: INT8 vs FP16.
- Real-time factor: processing time divided by audio duration.
- Word error rate.

### Recommended Links

- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- OpenAI Whisper paper: https://arxiv.org/abs/2212.04356
- OpenAI Whisper repository: https://github.com/openai/whisper
- Faster Whisper repository: https://github.com/SYSTRAN/faster-whisper
- Hugging Face audio course: https://huggingface.co/learn/audio-course/

### Beginner Exercises

1. Generate a Mel spectrogram from one audio file.
2. Install `faster-whisper`.
3. Transcribe one short audio file with `tiny`.
4. Transcribe the same file with `base`.
5. Compare speed and text quality.
6. Compute real-time factor.
7. Compute WER using `jiwer`.

### Assignment: ASR Benchmarking Script

Create `src/asr.py` that benchmarks `tiny` and `base` models.

```python
import json
import time
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel
from jiwer import wer


def audio_duration_seconds(path):
    info = sf.info(path)
    return info.frames / info.samplerate


def benchmark_asr(audio_file_path, reference_text=None):
    model_sizes = ["tiny", "base"]
    duration = audio_duration_seconds(audio_file_path)
    results = []

    for size in model_sizes:
        print(f"Benchmarking model: {size}")
        model = WhisperModel(size, device="cpu", compute_type="int8")

        started = time.perf_counter()
        segments, info = model.transcribe(audio_file_path, beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        elapsed = time.perf_counter() - started

        row = {
            "model": size,
            "language": info.language,
            "duration_seconds": round(duration, 3),
            "elapsed_seconds": round(elapsed, 3),
            "real_time_factor": round(elapsed / duration, 3),
            "text": text,
        }

        if reference_text:
            row["wer"] = round(wer(reference_text, text), 4)

        results.append(row)
        print(json.dumps(row, indent=2))

    return results


if __name__ == "__main__":
    output_dir = Path("outputs/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = benchmark_asr("data/raw/sample_001.wav", "move the robot to shelf number four")
    Path(output_dir / "sample_001_asr.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
```

### Done When

- You can explain the difference between audio duration and inference time.
- You can compare `tiny` and `base` using WER and real-time factor.
- You have benchmark JSON files for at least 5 recordings.

## Module 4: LLM Mechanics, Evaluation, and Transcript Repair

**Time:** Weeks 7-8

**Goal:** Understand text generation models and build a careful local transcript repair step.

### What To Learn

- Decoder-only LLMs.
- Self-attention with queries, keys, and values.
- Tokenization.
- Cross-entropy.
- Perplexity.
- WER and character error rate.
- Why transcript repair is risky: it can improve readability but may invent words.
- When to use repair: command-style domains with a known grammar are safer than open-ended conversations.

### Recommended Links

- Hugging Face NLP Course: https://huggingface.co/learn/nlp-course/
- Language Models are Few-Shot Learners: https://arxiv.org/abs/2005.14165
- Perplexity explanation: https://huggingface.co/docs/transformers/en/perplexity
- JiWER documentation: https://jitsi.github.io/jiwer/
- Transformers text generation docs: https://huggingface.co/docs/transformers/en/main_classes/text_generation

### Beginner Exercises

1. Tokenize text with a Hugging Face tokenizer.
2. Generate text locally with a very small model.
3. Compare original ASR text vs reference using WER.
4. Simulate corrupted transcripts.
5. Repair only commands from a fixed domain.
6. Recompute WER before and after repair.

### Assignment: Contextual WER Repair Engine

Create `src/repair.py`.

Important rule: keep the original ASR output and the repaired output. Never overwrite the raw transcription.

```python
from jiwer import wer
from transformers import pipeline


def build_repair_prompt(broken_transcript):
    return (
        "You repair short warehouse robot voice commands.\n"
        "Preserve the intended command. Do not add new facts.\n"
        "Return only the corrected command.\n\n"
        f"Broken command: {broken_transcript}\n"
        "Corrected command:"
    )


def repair_transcript(broken_transcript, model_name="google/flan-t5-small"):
    fixer = pipeline("text2text-generation", model=model_name)
    result = fixer(build_repair_prompt(broken_transcript), max_new_tokens=40)
    return result[0]["generated_text"].strip()


if __name__ == "__main__":
    reference = "move the robot to shelf number four and stop the engine immediately"
    corrupted = "move the robot to shelf number ... static ... and stop engine immediately"

    repaired = repair_transcript(corrupted)

    print("Corrupted:", corrupted)
    print("Repaired:", repaired)
    print("WER before:", wer(reference, corrupted))
    print("WER after:", wer(reference, repaired))
```

### Safer Alternative For Production

For warehouse commands, a grammar-based repair system may be safer than a free text LLM.

Example command schema:

```json
{
  "action": ["move", "stop", "pick", "drop", "scan"],
  "target": ["robot", "engine", "package", "shelf"],
  "location": ["shelf number one", "shelf number two", "dock A", "dock B"]
}
```

Use an LLM only as a fallback when confidence is low, and log all changes.

### Done When

- You can compute WER before and after repair.
- You understand that lower WER is useful, but preserving meaning is more important.
- Your script saves both raw and repaired transcripts.

## Module 5: Native Audio Models and Audio System Design

**Time:** Week 9

**Goal:** Understand where modern audio foundation models are heading and design a realistic low-latency warehouse voice pipeline.

### What To Learn

- Cascaded voice systems: ASR -> LLM -> TTS.
- Native audio models: models that process or generate audio more directly.
- Audio tokens.
- Speaker, style, emotion, and linguistic content.
- Disentanglement: separating content from style or speaker information.
- Edge vs server tradeoffs.
- Latency budgets.
- Privacy and offline operation.

### Recommended Links

- Voicebox paper: https://arxiv.org/abs/2306.15687
- Meta Voicebox overview: https://ai.meta.com/blog/voicebox-generative-ai-model-speech/
- Sarvam AI models page: https://www.sarvam.ai/
- Sarvam API docs: https://docs.sarvam.ai/
- SpeechBrain toolkit: https://speechbrain.github.io/
- ESPnet speech toolkit: https://espnet.github.io/espnet/

Note: The original pasted plan mentioned SarvamAI Bulbul as a native audio-text foundation model. Public materials describe Bulbul primarily as a text-to-speech voice model. Treat it as a useful speech technology reference, but do not assume it replaces ASR, LLM, and TTS stages unless official architecture documentation says so.

### Assignment: Warehouse Audio System Design

Create `reports/warehouse_audio_system_design.md`.

Your design must cover:

- 500 edge robots operating concurrently.
- Loud industrial noise.
- Multiple accents and dialects.
- P95 end-to-end latency below 800 ms.
- Offline fallback when the network is down.
- Privacy: raw audio should not be stored longer than necessary.

### Suggested Architecture

```text
Robot microphone
  -> local audio normalization
  -> local denoise
  -> local VAD
  -> local command keyword safety check
  -> ASR on edge for simple commands
  -> server ASR for difficult audio
  -> transcript confidence scoring
  -> optional grammar/LLM repair
  -> command parser
  -> robot action planner
  -> audit log
```

### Example Latency Budget

| Stage | Target P95 |
|---|---:|
| Audio capture window | 200 ms |
| Denoise + VAD | 40 ms |
| Edge ASR for command | 250 ms |
| Confidence + command parse | 40 ms |
| Optional repair fallback | 150 ms |
| Network and orchestration | 80 ms |
| Safety buffer | 40 ms |
| Total | 800 ms |

### Done When

- You can explain what should run on the robot and what can run on a server.
- You can defend your latency budget.
- You can explain how accent robustness should be evaluated with data, not guessed.

## Final Capstone: Build the Required Pipeline

**Time:** Week 10 or after Modules 1-5

Create `src/pipeline.py` that runs:

1. Load audio.
2. Denoise.
3. Detect speech segments.
4. Transcribe each segment or the full file.
5. Compute WER if reference text exists.
6. Optionally repair transcript.
7. Save all outputs.

### Final Output Format

Save one JSON file per audio input:

```json
{
  "audio_file": "sample_001.wav",
  "vad_segments": [
    {"start": 0.3, "end": 2.1}
  ],
  "asr": {
    "model": "base",
    "text": "move the robot to shelf number four",
    "elapsed_seconds": 1.25,
    "real_time_factor": 0.42
  },
  "evaluation": {
    "reference_text": "move the robot to shelf number four",
    "wer": 0.0
  },
  "repair": {
    "enabled": false,
    "text": null,
    "wer_after_repair": null
  }
}
```

### Capstone Checklist

- `src/vad.py` works on local audio.
- `src/asr.py` benchmarks `tiny` and `base`.
- `src/metrics.py` computes WER.
- `src/repair.py` keeps raw and repaired transcripts.
- `src/pipeline.py` connects the pieces.
- `outputs/` contains JSON results.
- `reports/warehouse_audio_system_design.md` explains the production design.

## Weekly Study Routine

Use this routine for every module:

1. Read or watch the beginner resource first.
2. Write notes in your own words.
3. Run the smallest possible code example.
4. Break it intentionally and read the error.
5. Run it on your own audio data.
6. Save results to `outputs/`.
7. Write a short module summary in `reports/`.

## What To Learn After This Roadmap

After completing the pipeline, continue with:

- PyTorch fundamentals: tensors, autograd, datasets, dataloaders.
- Fine-tuning small speech models.
- ONNX Runtime for optimized local inference.
- Streaming ASR.
- Speaker diarization.
- Wake word detection.
- Command grammar parsing.
- Edge deployment with Docker.
- Model monitoring and regression tests.

## Glossary

- **ASR:** Automatic speech recognition, converting speech to text.
- **VAD:** Voice activity detection, finding speech regions in audio.
- **PCM:** Pulse-code modulation, a common raw digital audio format.
- **WER:** Word error rate, a common ASR error metric.
- **RTF:** Real-time factor. `RTF = processing_time / audio_duration`.
- **P95 latency:** 95 percent of requests finish within this time.
- **Spectrogram:** A visual representation of frequencies over time.
- **Log-Mel spectrogram:** A spectrogram scaled closer to human hearing.
- **Transformer:** Neural network architecture based on attention.
- **Attention:** Mechanism that lets a model decide which parts of input matter.
- **Quantization:** Reducing numeric precision to make inference faster or smaller.
- **Disentanglement:** Separating mixed factors such as spoken content and speaker style.
