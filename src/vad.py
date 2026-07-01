import json
from pathlib import Path

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
import webrtcvad

from debug_utils import debug_value

TARGET_SR = 16000


def load_audio(path):
    data, sample_rate = sf.read(path)
    debug_value("raw_audio", data)
    debug_value("original_sample_rate", sample_rate)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sample_rate != TARGET_SR:
        data = librosa.resample(
            data.astype(np.float32), orig_sr=sample_rate, target_sr=TARGET_SR
        )
        debug_value("resampled_audio", data)
        sample_rate = TARGET_SR
    return data.astype(np.float32), sample_rate


def float_to_pcm16(audio):
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16).tobytes()


def frame_bytes(pcm_bytes, sample_rate, frame_ms=30):
    bytes_per_sample = 2
    frame_size = int(sample_rate * frame_ms / 1000) * bytes_per_sample
    for offset in range(0, len(pcm_bytes) - frame_size + 1, frame_size):
        yield offset, pcm_bytes[offset : offset + frame_size]


def detect_speech(path, aggressiveness=3, frame_ms=30):
    audio, sample_rate = load_audio(path)
    debug_value("audio", audio)
    debug_value("sample_rate", sample_rate)
    cleaned = nr.reduce_noise(y=audio, sr=sample_rate, prop_decrease=0.85)
    debug_value("cleaned_audio", cleaned)
    pcm = float_to_pcm16(cleaned)
    debug_value("pcm16_audio", pcm)
    vad = webrtcvad.Vad(aggressiveness)
    debug_value("vad", vad)

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
            segments.append(
                {"start": round(active_start, 3), "end": round(end_time, 3)}
            )
            active_start = None

    if active_start is not None:
        segments.append(
            {"start": round(active_start, 3), "end": round(len(audio) / sample_rate, 3)}
        )

    return {
        "audio_file": Path(path).name,
        "sample_rate": sample_rate,
        "segments": segments,
    }


if __name__ == "__main__":
    result = detect_speech("data/raw/sample_001.wav")
    Path("outputs/vad").mkdir(parents=True, exist_ok=True)
    Path("outputs/vad/sample_001.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
