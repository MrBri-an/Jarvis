"""Text-to-speech providers with a common interface.

`create_tts(config)` picks the best available engine:
piper (fast local neural) -> kokoro (local neural) -> sapi (Windows built-in).
Cloud (Groq PlayAI) available via explicit config.
"""
from __future__ import annotations

import io
import time
import wave

import numpy as np

from ..config import MODELS_DIR

PIPER_VOICES = {
    "en_US-lessac-medium": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    ),
}


def _wav_bytes_to_pcm(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as w:
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
        sampwidth = w.getsampwidth()
    if sampwidth == 2:
        return np.frombuffer(raw, dtype=np.int16), rate
    if sampwidth == 4:
        return (np.frombuffer(raw, dtype=np.int32) // 65536).astype(np.int16), rate
    if sampwidth == 1:
        return ((np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128) * 256), rate
    raise ValueError(f"unsupported wav sample width {sampwidth}")


class TTSBase:
    name = "base"

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        raise NotImplementedError


class PiperTTS(TTSBase):
    name = "piper"

    def __init__(self, voice: str = "en_US-lessac-medium"):
        import requests  # local import: engine may be absent

        if voice not in PIPER_VOICES:
            voice = next(iter(PIPER_VOICES))
        onnx_url, cfg_url = PIPER_VOICES[voice]
        self.onnx_path = MODELS_DIR / f"{voice}.onnx"
        self.cfg_path = MODELS_DIR / f"{voice}.onnx.json"
        for url, path in ((onnx_url, self.onnx_path), (cfg_url, self.cfg_path)):
            if not path.exists():
                print(f"[jarvis] downloading piper voice '{voice}' (~60 MB, one time)...")
                r = requests.get(url, timeout=300)
                r.raise_for_status()
                path.write_bytes(r.content)
        from piper import PiperVoice

        self.voice = PiperVoice.load(str(self.onnx_path))
        self._rate = 22050

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        # piper >= 1.3: synthesize() yields AudioChunk; 1.2: synthesize_wav()
        if hasattr(self.voice, "synthesize") and not hasattr(self.voice, "synthesize_wav"):
            pcm_parts, rate = [], None
            for chunk in self.voice.synthesize(text):
                rate = chunk.sample_rate
                pcm_parts.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
            return np.concatenate(pcm_parts) if pcm_parts else np.zeros(0, np.int16), rate or self._rate
        import io as _io

        buf = _io.BytesIO()
        with wave.open(buf, "wb") as w:
            self.voice.synthesize_wav(text, w)
        return _wav_bytes_to_pcm(buf.getvalue())


class KokoroTTS(TTSBase):
    name = "kokoro"

    def __init__(self):
        import requests

        model_path = MODELS_DIR / "kokoro-v1.0.onnx"
        voices_path = MODELS_DIR / "voices-v1.0.bin"
        if not model_path.exists():
            print("[jarvis] downloading Kokoro TTS model (~310 MB, one time)...")
            r = requests.get("https://huggingface.co/hexgrad/kokoro/resolve/main/kokoro-v1.0.onnx", timeout=600)
            r.raise_for_status()
            model_path.write_bytes(r.content)
        if not voices_path.exists():
            r = requests.get("https://huggingface.co/hexgrad/kokoro/resolve/main/voices-v1.0.bin", timeout=120)
            r.raise_for_status()
            voices_path.write_bytes(r.content)
        from kokoro_onnx import Kokoro

        self.k = Kokoro(str(model_path), str(voices_path))

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        samples, rate = self.k.create(text, voice="af_heart", speed=1.0, lang="en-us")
        pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
        return (pcm * 32767).astype(np.int16), rate


class SapiTTS(TTSBase):
    name = "sapi"

    def __init__(self, voice: str = "", rate: int = 175, tmp_dir=None):
        import tempfile

        self.voice = voice
        self.rate = rate
        self.tmp = tmp_dir or MODELS_DIR

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        import os
        import tempfile
        import pyttsx3

        out = self.tmp / f"_tts_{os.getpid()}_{time.time_ns()}.wav"
        engine = pyttsx3.init()
        if self.voice:
            try:
                engine.setProperty("voice", self.voice)
            except Exception:
                pass
        engine.setProperty("rate", self.rate)
        engine.save_to_file(text, str(out))
        engine.runAndWait()
        try:
            data = out.read_bytes()
        finally:
            try:
                out.unlink()
            except OSError:
                pass
        return _wav_bytes_to_pcm(data)


class GroqTTS(TTSBase):
    name = "groq"

    def __init__(self, api_key: str, voice: str = "Celeste-PlayAI"):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self.voice = voice

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        resp = self.client.audio.speech.create(
            model="playai-tts", voice=self.voice, input=text,
            response_format="wav",
        )
        return _wav_bytes_to_pcm(resp.read())


def create_tts(config) -> TTSBase:
    pref = config.tts.provider
    order = {"auto": ["piper", "kokoro", "sapi"]}.get(pref, [pref])
    last_err: Exception | None = None
    for name in order:
        try:
            if name == "piper":
                return PiperTTS(config.tts.piper_voice)
            if name == "kokoro":
                return KokoroTTS()
            if name == "groq":
                return GroqTTS(config.groq_api_key)
            if name == "sapi":
                return SapiTTS(config.tts.sapi_voice, config.tts.rate)
        except Exception as e:  # engine missing or model download failed
            last_err = e
            print(f"[jarvis] TTS '{name}' unavailable: {e}")
    raise RuntimeError(f"no TTS engine available (last error: {last_err})")
