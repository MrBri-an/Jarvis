"""Wake word calibration: live mic level + wake scores while YOU speak.

Usage:  python scripts/calibrate.py [seconds] [device_index]
Say 'Hey Jarvis' 4-5 times during the window. At the end you get:
- your average speaking RMS (is the mic even hearing you?)
- the peak wake score the model gives your voice
- the 3 best utterances saved + transcribed via Groq (what the mic actually heard)
- a recommended [wakeword] threshold for config.toml
"""
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from jarvis.audio.mic import MicStream
from jarvis.audio.wake import WakeWordDetector

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0
DEVICE = int(sys.argv[2]) if len(sys.argv) > 2 else -1

print("Input devices:")
print(MicStream.list_devices())
if DEVICE >= 0:
    print(f"\nUsing device [{DEVICE}]")
else:
    print("\nUsing the system default mic (pass its index as 2nd arg to test another)")

mic = MicStream(DEVICE)
mic.start()
det = WakeWordDetector("hey_jarvis", 0.5)

print(f"\n>>> SAY 'HEY JARVIS' 4-5 TIMES IN THE NEXT {DURATION:.0f} SECONDS <<<\n")

peaks: list[tuple[float, float, np.ndarray]] = []  # (score, t, clip)
max_score = 0.0
max_rms = 0.0
t0 = time.monotonic()
ring: list[np.ndarray] = []
next_paint = 0.0

while time.monotonic() - t0 < DURATION:
    frames = mic.read_frames()
    for f in frames:
        ring.append(f)
        if len(ring) > 90:  # ~2.7 s rolling window
            ring.pop(0)
        det.feed(f)
        rms = float(np.sqrt(np.mean(f.astype(np.float64) ** 2)))
        max_rms = max(max_rms, rms)
        score = det.last_score
        if score > max_score:
            max_score = score
        if score > 0.12:
            clip = np.concatenate(ring).flatten()
            peaks.append((score, time.monotonic() - t0, clip))
            ring = []  # don't double-store the same utterance
    now = time.monotonic()
    if now > next_paint:
        live = det.last_score
        bar = "#" * int(min(live, 1.0) * 40)
        lvl = "|" * int(min(max_rms / 3000, 1.0) * 12)
        print(f"  t={now - t0:4.1f}s  mic_lvl{lvl:<13} wake={live:.3f} {bar:<42} peak={max_score:.3f}", end="\r")
        next_paint = now + 0.15
    time.sleep(0.004)

mic.stop()
print()

# keep only the best 3 peaks
peaks.sort(key=lambda p: -p[0])
best = peaks[:3]
out_dir = Path(__file__).parent
print(f"\nResults: max_mic_rms={max_rms:.0f}   peak_wake_score={max_score:.3f}   threshold_in_config=0.5")
for i, (score, t, clip) in enumerate(best):
    p = out_dir / f"_wakecal_{i}.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(clip.astype(np.int16).tobytes())
    print(f"  best#{i}: score={score:.3f} at t={t:.1f}s  ({len(clip)/16000:.1f}s clip -> {p.name})")

if max_rms < 300:
    print("\nDIAGNOSIS: your voice is barely reaching the mic (max RMS < 300).")
    print("  -> Check Windows Settings > System > Sound > Input: pick the right mic,")
    print("     set input volume to 100, disable 'audio enhancements'.")
    print("  -> Or try another device:  python scripts/calibrate.py 25 15   (e.g. the headset)")
elif max_score >= 0.5:
    print("\nDIAGNOSIS: wake model hears you well; the 0.5 threshold should work.")
    print("  If run.py still misses it, say the phrase more crisply: 'Hey. Jarvis.'")
elif max_score >= 0.25:
    rec = max(0.18, round(max_score * 0.55, 2))
    print(f"\nDIAGNOSIS: model hears you but weakly. RECOMMENDED: set threshold = {rec} in config.toml [wakeword]")
else:
    print("\nDIAGNOSIS: the model is not recognizing your pronunciation of the phrase.")
    print("  Try speaking slower and clearer, slightly louder. If scores stay < 0.2,")
    print("  we can train a custom wake model or switch activation to Enter-to-talk.")

# transcribe the best clips via Groq so we can see what the mic actually captured
try:
    from jarvis.config import get_config

    cfg = get_config()
    if cfg.groq_api_key and best:
        from jarvis.providers.stt import GroqSTT

        stt = GroqSTT(cfg.groq_api_key)
        print("\nWhat the microphone actually captured (Groq transcription):")
        for i, (score, t, clip) in enumerate(best):
            try:
                print(f"  best#{i} (score {score:.2f}): {stt.transcribe(clip.astype(np.int16))!r}")
            except Exception as e:
                print(f"  best#{i}: transcription failed: {e}")
except Exception:
    pass
