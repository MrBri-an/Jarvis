"""JARVIS entry point.

Modes:
  python -m jarvis            voice mode: wake word -> talk -> spoken reply
  python -m jarvis --no-wake  voice mode without wake word (press Enter, then speak)
  python -m jarvis --text     typed input (no mic needed) — useful for testing tools
  python -m jarvis --no-tts   print replies instead of speaking them
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time

import numpy as np

from .audio.mic import MicStream
from .audio.speaker import Speaker
from .audio.vad import EndpointingState, SileroVAD
from .audio.wake import WakeWordDetector
from .brain import Brain
from .config import ROOT, get_config
from .memory import Memory
from .providers.stt import create_stt
from .providers.tts import create_tts
from .tools.file_tools import register_file_tools
from .tools.gui_tools import register_gui_tools
from .tools.registry import ToolRegistry
from .tools.system_tools import register_system_tools
from .tools.web_tools import register_web_tools

YES_WORDS = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "do it", "confirm",
             "affirmative", "go ahead", "please do", "proceed"}
NO_WORDS = {"no", "nope", "nah", "don't", "dont", "stop", "cancel", "never mind",
            "negative", "abort"}

# set by run_voice; used by the tray icon's pause toggle
_active_voiceio: VoiceIO | None = None


def _blip(speaker: Speaker, freq: int = 880, secs: float = 0.07) -> None:
    t = np.linspace(0, secs, int(16000 * secs), endpoint=False)
    tone = (np.sin(2 * np.pi * freq * t) * 12000).astype(np.int16)
    speaker.play(tone, 16000)


class VoiceIO:
    """Owns mic + VAD + wake word; provides listen/capture/speak/confirm."""

    def __init__(self, config, stt, tts):
        self.config = config
        self.stt = stt
        self.tts = tts
        self.speaker = Speaker(config.audio.output_device)
        self.mic = MicStream(config.audio.input_device)
        self.mic.start()
        self.vad = SileroVAD()
        self.wake: WakeWordDetector | None = None
        if config.wakeword.enabled:
            self.wake = WakeWordDetector(config.wakeword.name, config.wakeword.threshold)
        self.paused = False
        self.phase = "idle"          # idle | awake | listening | thinking (notch)
        self._manual_trigger = threading.Event()
        self._barge_frames: list[float] = []
        self._barge_active = False

    # ---- listening ----

    def listen_for_wake(self, timeout: float | None = None) -> bool:
        """Block until the wake word fires (or Enter is pressed)."""
        refractory_until = 0.0
        deadline = None if timeout is None else time.monotonic() + timeout
        last_printed = 0.0
        show_scores = getattr(self.config.general, "show_scores", True)
        while True:
            while self.paused:
                self.mic.read_frames()  # keep buffer drained while paused
                time.sleep(0.1)
            if self._manual_trigger.is_set():
                self._manual_trigger.clear()
                return True
            for frame in self.mic.read_frames():
                now = time.monotonic()
                if self.wake.feed(frame) and now > refractory_until:
                    refractory_until = now + self.config.wakeword.refractory_secs
                    return True
                score = self.wake.last_score
                if show_scores and score > 0.06 and score - last_printed > 0.03:
                    print(f"  hearing… wake score {score:.2f} "
                          f"(triggers at {self.config.wakeword.threshold})", end="\r", flush=True)
                    last_printed = score
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.01)

    def capture_utterance(self) -> np.ndarray:
        """Record until endpointing fires; returns 16 kHz int16 audio."""
        start_secs = self.config.vad.start_speech_secs
        end_secs = self.config.vad.end_silence_secs
        ep = EndpointingState(start_secs, end_secs)
        chunks: list[np.ndarray] = []
        speech_started = False
        t_max = time.monotonic() + self.config.vad.max_utterance_secs
        while time.monotonic() < t_max:
            for frame in self.mic.read_frames():
                chunks.append(frame)
                flat = frame.flatten()
                for i in range(0, len(flat) - 512 + 1, 512):
                    event = ep.feed(self.vad.prob(flat[i : i + 512]), 512 / 16000)
                    if event == "start":
                        speech_started = True
                    elif event == "end":
                        audio = np.concatenate(chunks)
                        min_samples = int(self.config.vad.min_utterance_secs * 16000)
                        return audio if len(audio) >= min_samples else np.zeros(0, np.int16)
            time.sleep(0.005)
        return np.concatenate(chunks) if chunks else np.zeros(0, np.int16)

    def transcribe(self, pcm: np.ndarray) -> str:
        t0 = time.monotonic()
        text = self.stt.transcribe(pcm)
        self._stt_secs = time.monotonic() - t0
        return text

    # ---- speaking + barge-in ----

    def _check_barge_in(self) -> bool:
        """Called while JARVIS speaks; user input cancels output.

        Two triggers (Windows-MCP takeover pattern):
        - physical: any mouse click or Esc — instant, most reliable
        - voice: strong sustained speech (0.75 VAD for half a second) —
          stricter than normal VAD so JARVIS's own TTS doesn't cancel itself
        """
        user32 = ctypes.windll.user32
        for vk in (0x01, 0x02, 0x1B):  # LMB, RMB, ESC
            if user32.GetAsyncKeyState(vk) & 0x8000:
                self.speaker.flush()
                self._barge_frames = []
                return True
        if not self.speaker.busy():
            self._barge_frames = []
            return False
        for frame in self.mic.read_frames():
            flat = frame.flatten()
            probs = self.vad.probs(flat[: len(flat) - (len(flat) % 512)])
            self._barge_frames.append(1.0 if any(p >= 0.75 for p in probs) else 0.0)
        if len(self._barge_frames) > 30:
            self._barge_frames = self._barge_frames[-30:]
        # ≥0.5s of strong speech within the last second
        if len(self._barge_frames) >= 30 and sum(self._barge_frames) >= 15:
            self.speaker.flush()
            self._barge_frames = []
            return True
        return False

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        pcm, rate = self.tts.synthesize(text)
        self.speaker.play(pcm, rate)
        while self.speaker.busy():
            self._check_barge_in()
            time.sleep(0.05)

    # ---- confirmation gate (spoken yes/no) ----

    def confirm_voice(self, question: str) -> bool:
        _blip(self.speaker, 660)
        self.speak(question)
        pcm = self.capture_utterance()
        if len(pcm) == 0:
            return False
        text = self.transcribe(pcm).lower()
        if any(w in text for w in YES_WORDS):
            return True
        if any(w in text for w in NO_WORDS):
            return False
        return False  # unclear = refuse

    def close(self) -> None:
        self.mic.stop()
        self.speaker.stop_thread()


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    register_system_tools(reg)
    register_web_tools(reg)
    register_file_tools(reg)
    register_gui_tools(reg)
    return reg


def _print_metrics(metrics: dict) -> None:
    if not metrics:
        return
    parts = [f"{k}={v}s" for k, v in metrics.items()]
    print(f"    [{', '.join(parts)}]")


def run_voice(config, no_wake: bool, no_tts: bool) -> None:
    global _active_voiceio
    from rich.console import Console

    console = Console()
    tts = None if no_tts else create_tts(config)
    stt = create_stt(config)
    mem = Memory(ROOT / "memory.db")
    registry = build_registry()
    brain = Brain(config, registry, tts, None, memory=mem)
    io = VoiceIO(config, stt, tts)
    _active_voiceio = io

    # status notch (safe to fail)
    notch = None
    if getattr(config.general, "notch", True):
        try:
            from .notch import NotchBar

            notch = NotchBar()
            notch.start()
        except Exception as e:
            console.print(f"[dim]notch disabled: {e}[/]")

    def _notch_monitor() -> None:
        last = None
        while notch is not None and notch.enabled:
            if io.phase in ("idle", "awake", "listening"):
                s = io.phase
            else:  # thinking: flip to speaking while TTS audio is playing
                s = "speaking" if (io.speaker.busy() and not no_tts) else "thinking"
            if s != last:
                notch.set_state(s)
                last = s
            time.sleep(0.15)

    threading.Thread(target=_notch_monitor, daemon=True, name="jarvis-notch-mon").start()

    # Enter-to-talk fallback: wake word misses shouldn't block the whole loop
    if config.wakeword.enabled and not no_wake:
        def _enter_trigger() -> None:
            while True:
                try:
                    input()
                except EOFError:
                    return
                io._manual_trigger.set()

        threading.Thread(target=_enter_trigger, daemon=True, name="jarvis-enter").start()
    brain.speaker = io.speaker
    brain.should_abort = lambda: io._check_barge_in()
    brain.ctx.confirm_fn = io.confirm_voice
    brain.on_delta = lambda d: console.print(d, end="")
    brain.on_tool = lambda n, a: console.print(f"\n[bold cyan]⚙ {n}[/] {a}")
    brain.on_tool_result = lambda n, r: console.print(f"\n[dim]  ↳ {r[:200]}[/]")

    mode = "say 'Hey Jarvis' (or press Enter)" if (config.wakeword.enabled and not no_wake) else "press Enter, then speak"
    resolved_mic = io.mic.input_device
    console.print(
        f"[bold green]JARVIS online.[/] {mode} — Ctrl+C to quit.\n"
        f"[dim]mic={resolved_mic!r} @ {io.mic._native_rate}Hz ({io.mic._channels}ch, "
        f"threshold={config.wakeword.threshold})[/]"
    )
    try:
        while True:
            io.phase = "idle"
            if config.wakeword.enabled and not no_wake:
                io.listen_for_wake()
            else:
                input()
            io.phase = "awake"
            _blip(io.speaker)
            io.speaker.wait_done()
            io.phase = "listening"
            pcm = io.capture_utterance()
            if len(pcm) < 16000 * 0.25:
                continue
            io.phase = "thinking"
            t0 = time.monotonic()
            text = io.transcribe(pcm)
            console.print(f"\n[bold yellow]You:[/] {text}")
            brain.metrics = {"stt": round(time.monotonic() - t0, 2)}
            try:
                reply = brain.respond(text)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                console.print(f"\n[bold red]error:[/] {e}")
                io.speaker.flush()
                continue
            io.speaker.wait_done()
            io.phase = "idle"
            console.print()  # newline after streamed deltas
            _print_metrics(brain.metrics)
            brain.metrics = {}
    except KeyboardInterrupt:
        console.print("\n[bold]Goodbye.[/]")
    finally:
        io.close()


def run_text(config, no_tts: bool) -> None:
    from rich.console import Console

    console = Console()
    tts = None if no_tts else create_tts(config)
    speaker = Speaker() if tts else None
    mem = Memory(ROOT / "memory.db")
    registry = build_registry()
    brain = Brain(config, registry, tts, speaker, memory=mem)
    brain.on_delta = lambda d: console.print(d, end="")
    brain.on_tool = lambda n, a: console.print(f"\n[bold cyan]⚙ {n}[/] {a}")
    brain.on_tool_result = lambda n, r: console.print(f"\n[dim]  ↳ {r[:200]}[/]")

    def confirm_text(q: str) -> bool:
        ans = console.input(f"[bold red]{q} [y/n][/] ")
        return ans.strip().lower() in ("y", "yes")

    brain.ctx.confirm_fn = confirm_text
    console.print("[bold green]JARVIS text mode.[/] Type messages, Ctrl+C to quit.")
    while True:
        try:
            user = console.input("\n[bold yellow]You:[/] ")
        except (EOFError, KeyboardInterrupt):
            break
        if not user.strip():
            continue
        try:
            brain.respond(user)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            console.print(f"[bold red]error:[/] {e}")
        console.print()
        _print_metrics(brain.metrics)
        brain.metrics = {}
        if speaker is not None:
            speaker.wait_done()


def main() -> None:
    ap = argparse.ArgumentParser(prog="jarvis")
    ap.add_argument("--text", action="store_true", help="typed input, no microphone")
    ap.add_argument("--no-wake", action="store_true", help="press Enter instead of wake word")
    ap.add_argument("--no-tts", action="store_true", help="print replies, don't speak")
    ap.add_argument("--tray", action="store_true", help="run in the system tray")
    ap.add_argument("--config", default=None, help="path to config.toml")
    args = ap.parse_args()

    config = get_config(args.config and __import__("pathlib").Path(args.config))

    if not config.has_cloud_key and config.llm.provider != "local":
        print("=" * 62)
        print("  No API key found. Get a FREE Groq key: https://console.groq.com/keys")
        print("  Then save it in jarvis/.env as:  GROQ_API_KEY=gsk_...")
        print("  (Voice listening still runs locally; the brain needs the key.)")
        print("=" * 62)

    if args.tray:
        from .tray import run_tray

        run_tray(config, args.no_tts)
        return

    if args.text:
        run_text(config, args.no_tts)
    else:
        try:
            run_voice(config, args.no_wake, args.no_tts)
        except Exception as e:
            print(f"[jarvis] voice mode failed: {e}", file=sys.stderr)
            print("Falling back to text mode. Fix audio/mic issues, or use --text.", file=sys.stderr)
            run_text(config, args.no_tts)


if __name__ == "__main__":
    main()
