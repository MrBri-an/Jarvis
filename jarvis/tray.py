"""System tray icon: runs the voice loop in the background.

python run.py --tray   → JARVIS lives in the tray; right-click for
Pause listening / Resume / Quit. Console output goes to jarvis_tray.log.
"""
from __future__ import annotations

import logging
import logging.handlers
import threading

from .config import ROOT


def _setup_logging() -> None:
    handler = logging.handlers.RotatingFileHandler(
        ROOT / "jarvis_tray.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8"
    )
    logging.basicConfig(handlers=[handler], level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")


def run_tray(config, no_tts: bool = False) -> None:
    from io import StringIO

    import pystray
    from PIL import Image, ImageDraw

    from .__main__ import run_voice

    _setup_logging()
    state = {"paused": False}

    # simple 64x64 icon: dark circle + cyan arc (a tiny "J")
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((4, 4, 60, 60), fill=(24, 26, 32, 255), outline=(0, 200, 255, 255), width=3)
    d.arc((14, 12, 50, 44), start=200, end=340, fill=(0, 200, 255, 255), width=4)
    d.line((32, 30, 32, 52), fill=(0, 200, 255, 255), width=4)

    import contextlib

    buf = StringIO()
    with contextlib.redirect_stdout(buf):  # tray mode has no console
        worker = threading.Thread(
            target=lambda: run_voice(config, no_wake=False, no_tts=no_tts),
            daemon=True, name="jarvis-voice",
        )

    def toggle_pause(icon, item):
        state["paused"] = not state["paused"]
        # wake detector is re-created by loop; pause acts via mic sleep
        import jarvis.__main__ as m

        if getattr(m, "_active_voiceio", None) is not None:
            m._active_voiceio.paused = state["paused"]
        return True

    def quit_app(icon, item):
        icon.stop()
        import os

        os._exit(0)

    worker.start()
    menu = pystray.Menu(
        pystray.MenuItem("Pause listening", toggle_pause, checked=lambda item: state["paused"]),
        pystray.MenuItem("Quit JARVIS", quit_app),
    )
    pystray.Icon("jarvis", img, "JARVIS", menu).run()
