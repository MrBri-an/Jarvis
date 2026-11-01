"""Audio playback with a queue, rate switching, and barge-in flush."""
from __future__ import annotations

import queue
import threading
import time

import numpy as np
import sounddevice as sd


class Speaker:
    """Plays queued int16 numpy audio chunks (each tagged with its sample rate).

    - `play(chunk, rate)` enqueues and returns immediately.
    - `busy()` reports whether audio is queued or playing.
    - `flush()` drops everything queued and silences output immediately.
    """

    SENTINEL = object()

    def __init__(self, output_device: int = -1, volume: float = 1.0):
        self.output_device = None if output_device < 0 else int(output_device)
        self.volume = volume
        self._q: queue.Queue = queue.Queue()
        self._stream: sd.OutputStream | None = None
        self._rate = 0
        self._lock = threading.Lock()
        self._playing = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="jarvis-speaker")
        self._thread.start()

    def _ensure_stream(self, rate: int) -> sd.OutputStream:
        if self._stream is not None and self._rate == rate:
            return self._stream
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
        self._stream = sd.OutputStream(
            samplerate=rate, channels=1, dtype="int16", device=self.output_device
        )
        self._stream.start()
        self._rate = rate
        return self._stream

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is self.SENTINEL:
                return
            chunk, rate = item
            if chunk is None:
                self._drain_now()
                continue
            if self.volume != 1.0:
                chunk = (chunk.astype(np.float32) * self.volume).clip(-32768, 32767).astype(np.int16)
            try:
                with self._lock:
                    stream = self._ensure_stream(rate)
                    self._playing.set()
                    stream.write(chunk)
            except Exception:
                pass
            finally:
                if self._q.empty():
                    self._playing.clear()

    def _drain_now(self) -> None:
        """Signal used to append silence marker; actual flush via flush()."""
        pass

    def play(self, chunk: np.ndarray, rate: int) -> None:
        self._q.put((chunk, rate))

    def busy(self) -> bool:
        return self._playing.is_set() or not self._q.empty()

    def wait_done(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.busy():
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        return True

    def flush(self) -> None:
        """Immediate silence: drop queued chunks and close the playback stream."""
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.abort()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
                self._rate = 0
        self._playing.clear()

    def stop_thread(self) -> None:
        self._q.put(self.SENTINEL)
