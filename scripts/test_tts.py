"""TTS test: picks the best available engine and speaks a line. Measures first-audio time."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis.config import get_config  # noqa: E402
from jarvis.providers.tts import create_tts  # noqa: E402

config = get_config()
t0 = time.monotonic()
tts = create_tts(config)
print(f"engine selected: {tts.name} (init {time.monotonic() - t0:.1f}s)")

for line in ["JARVIS online. All systems nominal.",
             "I sorted seventeen files into four folders and moved the installers to the archive.",
             "The weather in Austin is eighty four degrees and mostly clear."]:
    t0 = time.monotonic()
    pcm, rate = tts.synthesize(line)
    from jarvis.audio.speaker import Speaker

    sp = Speaker()
    sp.play(pcm, rate)
    sp.wait_done()
    sp.stop_thread()
    print(f"  spoke {len(pcm)/rate:.1f}s audio in {time.monotonic() - t0:.2f}s ({tts.name})")
