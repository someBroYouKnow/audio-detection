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
    print("What load_audio found: ", data, sample_rate)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sample_rate != TARGET_SR:
        data = librosa.resample(
            data.astype(np.float32), orig_sr=sample_rate, target_sr=TARGET_SR
        )
        print("Printed data after librosa ", data)
        sample_rate = TARGET_SR
    return data.astype(np.float32), sample_rate


def float_to_pcm16(audio):
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16).tobytes()


def detect_speech(path, aggressiveness=3, frame_ms=30):
    audio, sample_rate = load_audio(path)
    print("return of detect_speech", audio, sample_rate)
    cleaned = nr.reduce_noise(y=audio, sr=sample_rate, prop_decrease=0.85)
    print("noise reduce: ", cleaned)
    pcm = float_to_pcm16(cleaned)
    print("pcm 16 : ", pcm)
    vad = webrtcvad.Vad(aggressiveness)
    print("vad output: ", vad)


if __name__ == "__main__":
    result = detect_speech("data/raw/sample_001.wav")
    Path("outputs/vad").mkdir(parents=True, exist_ok=True)
    Path("outputs/vad/sample_001.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
