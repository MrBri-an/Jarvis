"""Mic sanity test: lists input devices, records 5 s, prints levels, saves wav."""
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis.audio.mic import MicStream  # noqa: E402

import numpy as np  # noqa: E402

print("Input devices:")
print(MicStream.list_devices())

mic = MicStream()
mic.start()
print("\nRecording 5 seconds — say something...")
chunks = []
t_end = time.time() + 5
while time.time() < t_end:
    chunks += mic.read_frames()
    time.sleep(0.01)
mic.stop()

audio = np.concatenate(chunks).flatten() if chunks else np.zeros(1, np.int16)
peak = int(np.abs(audio.astype(np.int32)).max())
rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
out = Path(__file__).parent / "_mic_test.wav"
with wave.open(str(out), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(audio.astype(np.int16).tobytes())
print(f"\nsamples={len(audio)}  peak={peak}  rms={rms:.0f}  saved={out}")
print("PASS: audio captured" if peak > 500 else "WEAK/NO SIGNAL: check mic device/permissions")
