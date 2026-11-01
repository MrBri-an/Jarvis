"""Silero VAD via onnxruntime (no torch dependency).

Supports both the v4 (h/c state) and v5 (state tensor) ONNX graphs.
"""
from __future__ import annotations

import pathlib

import numpy as np
import onnxruntime as ort
import requests

from ..config import MODELS_DIR

MODEL_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"


def ensure_vad_model() -> pathlib.Path:
    path = MODELS_DIR / "silero_vad.onnx"
    if not path.exists():
        print("[jarvis] downloading Silero VAD model (~2 MB)...")
        r = requests.get(MODEL_URL, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


class SileroVAD:
    CHUNK = 512  # 32 ms @ 16 kHz — required input size for the ONNX graph

    def __init__(self, threshold: float = 0.5):
        model_path = ensure_vad_model()
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self._input_names = {i.name for i in self.session.get_inputs()}
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((2, 1, 64), dtype=np.float32)   # v4 state
        self._state = np.zeros((2, 1, 128), dtype=np.float32)  # v5 state

    def prob(self, chunk_int16: np.ndarray) -> float:
        """Speech probability for a 512-sample int16 chunk."""
        x = chunk_int16.astype(np.float32) / 32768.0
        x = x.reshape(1, -1)
        if "state" in self._input_names:  # v5 graph
            feeds = {"input": x, "state": self._state, "sr": np.array(16000, dtype=np.int64)}
            out, self._state = self.session.run(None, feeds)
            return float(out[0, 0])
        # v4 graph
        feeds = {"input": x, "h": self._h, "c": self._h, "sr": np.array(16000, dtype=np.int64)}
        out, _, _ = self.session.run(None, feeds)
        return float(out[0, 0])

    def probs(self, samples_int16: np.ndarray) -> list[float]:
        """Slide over arbitrary-length audio in 512-sample chunks."""
        out = []
        flat = samples_int16.flatten()
        for i in range(0, len(flat) - self.CHUNK + 1, self.CHUNK):
            out.append(self.prob(flat[i : i + self.CHUNK]))
        return out


class EndpointingState:
    """Stateful speech endpointing over a stream of VAD probabilities."""

    def __init__(self, start_secs: float, end_secs: float):
        self.start_secs = start_secs
        self.end_secs = end_secs
        self.speech_started = False
        self._silence = 0.0
        self._speech = 0.0

    def feed(self, prob: float, chunk_secs: float) -> str | None:
        """Returns 'start', 'end', or None."""
        if prob >= 0.5:
            self._speech += chunk_secs
            self._silence = 0.0
            if not self.speech_started and self._speech >= self.start_secs:
                self.speech_started = True
                return "start"
        else:
            self._speech = max(0.0, self._speech - chunk_secs * 2)
            if self.speech_started:
                self._silence += chunk_secs
                if self._silence >= self.end_secs:
                    return "end"
        return None
