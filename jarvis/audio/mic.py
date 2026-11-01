"""Microphone capture: WASAPI via sounddevice, 16 kHz mono int16.

Opens the stream at 16 kHz when the device allows it; otherwise opens at the
device default rate and resamples each block to 16 kHz (scipy resample_poly).
"""
from __future__ import annotations

import collections
import threading

import numpy as np
import sounddevice as sd

TARGET_RATE = 16000


class MicStream:
    """Continuous mic stream; consumers pull 16 kHz int16 numpy frames."""

    def __init__(self, input_device: int | str = -1, block_ms: int = 30):
        self.block_ms = block_ms
        self.input_device = input_device  # index, -1/None for default, or name substring
        self._buf: collections.deque[np.ndarray] = collections.deque(maxlen=200)
        self._lock = threading.Lock()
        self._native_rate = TARGET_RATE
        self._stream = self._open()

    def _open(self) -> sd.InputStream:
        device = self.input_device
        if isinstance(device, str):
            device = self._resolve_by_name(device)
            if device is None:
                print("[jarvis] configured mic not found; using system default")
        if device is None or (isinstance(device, int) and device < 0):
            device = None
        info = sd.query_devices(device or sd.default.device[0])
        # Mic arrays expose N raw elements; capture them all and downmix to
        # mono (channel 0 alone is often one unprocessed element).
        self._channels = max(1, min(4, int(info["max_input_channels"])))
        try:
            self._native_rate = TARGET_RATE
            return sd.InputStream(
                samplerate=TARGET_RATE,
                blocksize=int(TARGET_RATE * self.block_ms / 1000),
                channels=self._channels,
                dtype="int16",
                device=device,
                callback=self._callback,
            )
        except sd.PortAudioError:
            # Device can't do 16 kHz directly: open native, resample in callback.
            self._native_rate = int(info["default_samplerate"])
            return sd.InputStream(
                samplerate=self._native_rate,
                blocksize=int(self._native_rate * self.block_ms / 1000),
                channels=self._channels,
                dtype="int16",
                device=device,
                callback=self._callback,
            )

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            pass  # overflows happen under load; drop frame rather than stall
        with self._lock:
            if len(self._buf) == self._buf.maxlen:
                self._buf.popleft()
            if indata.shape[1] > 1:
                mono = indata.mean(axis=1).astype(np.int16).reshape(-1, 1)
            else:
                mono = indata
            self._buf.append(mono.copy())

    def read_frames(self) -> list[np.ndarray]:
        """Drain all pending frames (int16 mono @16 kHz, ~30 ms each)."""
        with self._lock:
            frames = list(self._buf)
            self._buf.clear()
        if self._native_rate == TARGET_RATE:
            return frames
        from scipy.signal import resample_poly
        import math

        g = math.gcd(self._native_rate, TARGET_RATE)
        return [resample_poly(f.flatten(), TARGET_RATE // g, self._native_rate // g).astype(np.int16) for f in frames]

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass

    @staticmethod
    def _resolve_by_name(name_substring: str) -> int | None:
        want = name_substring.lower()
        best = None
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and want in d["name"].lower():
                # the same array appears under several host APIs; MME measured
                # much better wake-word scores than WASAPI on this Realtek rig
                is_mme = "mme" in sd.query_hostapis(d["hostapi"])["name"].lower()
                key = (is_mme, d["max_input_channels"], d["default_samplerate"])
                if best is None or key > best[0]:
                    best = (key, i)
        return best[1] if best else None

    @staticmethod
    def list_devices() -> str:
        lines = []
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                lines.append(f"  [{i}] {d['name']}  (in={d['max_input_channels']}, default_sr={int(d['default_samplerate'])})")
        return "\n".join(lines)
