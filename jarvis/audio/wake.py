"""Wake word detection via openWakeWord ("hey jarvis" pretrained model)."""
from __future__ import annotations

import numpy as np

from ..config import MODELS_DIR

# openWakeWord consumes exactly 80 ms (1280 samples) of 16 kHz int16 audio.
FRAME_SAMPLES = 1280


class WakeWordDetector:
    def __init__(self, wake_name: str = "hey_jarvis", threshold: float = 0.5):
        import pathlib

        import openwakeword
        from openwakeword.model import Model

        # Ensure pretrained models are present (no-ops after first run).
        try:
            openwakeword.utils.download_models(
                model_names=[wake_name], target_directory=str(MODELS_DIR)
            )
        except Exception as e:
            print(f"[jarvis] wake model download skipped ({e}); using any existing copy")

        candidates = list(MODELS_DIR.glob(f"{wake_name}*.onnx"))
        if not candidates:
            pkg = pathlib.Path(openwakeword.__file__).parent
            candidates = list(pkg.rglob(f"{wake_name}*.onnx"))
        if not candidates:
            raise FileNotFoundError(
                f"wake word model '{wake_name}' not found in {MODELS_DIR} or openwakeword package"
            )
        self.model = Model(
            wakeword_models=[str(candidates[0])], inference_framework="onnx"
        )
        self.threshold = threshold
        self._buf = np.empty(0, dtype=np.int16)
        self.last_score = 0.0

    def feed(self, samples: np.ndarray) -> bool:
        """Feed int16 mono samples at 16 kHz. Returns True on a wake hit."""
        self._buf = np.concatenate([self._buf, samples.flatten().astype(np.int16)])
        hit = False
        while len(self._buf) >= FRAME_SAMPLES:
            frame = self._buf[:FRAME_SAMPLES]
            self._buf = self._buf[FRAME_SAMPLES:]
            scores = self.model.predict(frame)
            score = max(float(v) for v in scores.values()) if scores else 0.0
            self.last_score = score
            if score >= self.threshold:
                hit = True
                self.model.reset()
        return hit

    def reset(self) -> None:
        self._buf = np.empty(0, dtype=np.int16)
        self.model.reset()
