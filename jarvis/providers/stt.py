"""Speech-to-text providers: Groq (cloud, free tier) or faster-whisper (local)."""
from __future__ import annotations

import io
import wave

import numpy as np


def pcm_to_wav_bytes(pcm: np.ndarray, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.astype(np.int16).tobytes())
    return buf.getvalue()


class STTBase:
    def transcribe(self, pcm: np.ndarray, rate: int = 16000) -> str:
        raise NotImplementedError


class GroqSTT(STTBase):
    def __init__(self, api_key: str, model: str = "whisper-large-v3-turbo"):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self.model = model

    def transcribe(self, pcm: np.ndarray, rate: int = 16000) -> str:
        wav = pcm_to_wav_bytes(pcm, rate)
        result = self.client.audio.transcriptions.create(
            model=self.model,
            file=("utterance.wav", wav, "audio/wav"),
            temperature=0.0,
        )
        return (result.text or "").strip()


class LocalSTT(STTBase):
    """faster-whisper on CPU — offline fallback."""

    def __init__(self, model_size: str = "small", compute_type: str = "int8"):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_size, device="cpu", compute_type=compute_type, cpu_threads=4)

    def transcribe(self, pcm: np.ndarray, rate: int = 16000) -> str:
        audio = pcm.astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(audio, language="en", beam_size=1, vad_filter=False)
        return " ".join(s.text.strip() for s in segments).strip()


def create_stt(config) -> STTBase:
    if config.stt.provider == "groq" and config.groq_api_key:
        return GroqSTT(config.groq_api_key, config.stt.groq_model)
    if config.stt.provider == "groq":
        print("[jarvis] no GROQ_API_KEY — falling back to local STT (slower)")
    return LocalSTT(config.stt.local_model, config.stt.local_compute)
