"""Wake word test: prints 'hey jarvis' detection scores live. Ctrl+C to stop.

Uses your config.toml device + threshold. Say 'Hey Jarvis' and watch the
scores; a '*** WAKE ***' line means run.py would trigger.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis.audio.mic import MicStream  # noqa: E402
from jarvis.audio.wake import WakeWordDetector  # noqa: E402
from jarvis.config import get_config  # noqa: E402

cfg = get_config()
threshold = float(cfg.wakeword.threshold)
det = WakeWordDetector(cfg.wakeword.name, threshold=threshold)
mic = MicStream(cfg.audio.input_device)
mic.start()
print(f"device={cfg.audio.input_device!r} threshold={threshold}")
print("Listening for 'Hey Jarvis' (live scores, Ctrl+C to stop)...")
hits = 0
try:
    while True:
        for frame in mic.read_frames():
            if det.feed(frame):
                hits += 1
                print(f"\n>>> WAKE #{hits} detected at {time.strftime('%H:%M:%S')}")
            elif det.last_score > 0.05:
                print(f"  score={det.last_score:.3f} (trigger at {threshold})", end="\r")
        time.sleep(0.01)
except KeyboardInterrupt:
    print(f"\nstopped. total hits: {hits}")
finally:
    mic.stop()
