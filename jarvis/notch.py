"""Status notch: a Dynamic-Island-style pill at the top of the screen.

Shows what JARVIS is doing: idle (dim), awake (green), thinking (cyan),
speaking (blue). Runs tkinter in a background thread and is safe to fail —
if it can't open, the assistant keeps running without it.

Later phases can make it interactive (click for transcript, waveform, etc.);
the rest of the app only talks to set_state().
"""
from __future__ import annotations

import queue
import threading

STATES = {
    "idle": ("JARVIS", "#6b7280", False),
    "awake": ("I'm listening…", "#30d158", True),
    "listening": ("Listening…", "#ffd60a", True),
    "thinking": ("Thinking…", "#00c7be", True),
    "speaking": ("Speaking…", "#0a84ff", True),
}

_PILL_W = 168
_PILL_H = 34
_BG = "#101218"
_BORDER = "#2a2d38"
_TEXT = "#e5e7eb"
_TRANSPARENCY_KEY = "#010101"


class NotchBar:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._q: queue.Queue = queue.Queue()
        self._root = None
        self._thread: threading.Thread | None = None

    # ---- public API (thread-safe) ----

    def start(self) -> None:
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="jarvis-notch")
        self._thread.start()

    def set_state(self, state: str) -> None:
        if not self.enabled:
            return
        self._q.put(state)

    def stop(self) -> None:
        if self._root is not None:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass

    # ---- tk thread ----

    def _run(self) -> None:
        try:
            import tkinter as tk

            root = tk.Tk()
            self._root = root
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            root.attributes("-transparentcolor", _TRANSPARENCY_KEY)
            root.configure(bg=_TRANSPARENCY_KEY)
            sw = root.winfo_screenwidth()
            x = (sw - _PILL_W) // 2
            root.geometry(f"{_PILL_W}x{_PILL_H}+{x}+8")

            canvas = tk.Canvas(
                root, width=_PILL_W, height=_PILL_H,
                bg=_TRANSPARENCY_KEY, highlightthickness=0,
            )
            canvas.pack()
            dot = canvas.create_oval(14, 12, 26, 24, fill="#6b7280", outline="")
            text = canvas.create_text(
                _PILL_W // 2 + 8, _PILL_H // 2, text="JARVIS",
                fill=_TEXT, font=("Segoe UI", 10, "bold"),
            )
            pill = self._rounded(canvas, 1, 1, _PILL_W - 2, _PILL_H - 2, 16)
            canvas.tag_lower(pill)

            state = {"name": "idle", "pulse_on": True}

            def apply(new_state: str) -> None:
                label, color, _ = STATES.get(new_state, STATES["idle"])
                state["name"] = new_state
                canvas.itemconfigure(text, text=label)
                canvas.itemconfigure(dot, fill=color)
                root.geometry(f"{_PILL_W}x{_PILL_H}+{(root.winfo_screenwidth() - _PILL_W) // 2}+8")

            def poll() -> None:
                changed = False
                while True:
                    try:
                        apply(self._q.get_nowait())
                        changed = True
                    except queue.Empty:
                        break
                _, color, pulses = STATES.get(state["name"], STATES["idle"])
                if pulses and not changed:
                    state["pulse_on"] = not state["pulse_on"]
                    shade = color if state["pulse_on"] else self._dim(color)
                    canvas.itemconfigure(dot, fill=shade)
                root.after(320 if pulses else 120, poll)

            poll()
            root.mainloop()
        except Exception as e:
            print(f"[jarvis] notch disabled ({e})")
            self.enabled = False

    @staticmethod
    def _rounded(canvas, x1, y1, x2, y2, r):  # noqa: ANN001
        import tkinter as tk

        points = [
            x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
            x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
        ]
        return canvas.create_polygon(points, smooth=True, fill=_BG, outline=_BORDER, width=1)

    @staticmethod
    def _dim(hex_color: str) -> str:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f"#{int(r * 0.35):02x}{int(g * 0.35):02x}{int(b * 0.35):02x}"
