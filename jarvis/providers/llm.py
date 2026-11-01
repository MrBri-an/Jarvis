"""Conversation brain provider: OpenAI-compatible chat with tool calling.

Works with Groq (free tier), OpenAI, or any OpenAI-compatible endpoint
(Ollama, LM Studio, llama.cpp server) via config.
"""
from __future__ import annotations

from typing import Callable

from openai import OpenAI

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LLMAborted(Exception):
    """Raised when the stream is cancelled (barge-in)."""


class ToolCall:
    def __init__(self) -> None:
        self.id = ""
        self.name = ""
        self.arguments = ""

    def complete(self) -> dict:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


class LLMClient:
    def __init__(self, api_key: str, model: str, base_url: str | None = None,
                 temperature: float = 0.4):
        # placeholder key keeps construction working when unset (tests, text mode);
        # real API calls then fail with a clear auth error
        self.client = OpenAI(api_key=api_key or "missing-key", base_url=base_url or GROQ_BASE_URL)
        self.model = model
        self.temperature = temperature

    @classmethod
    def from_config(cls, config) -> "LLMClient":
        llm = config.llm
        if llm.provider == "groq":
            return cls(config.groq_api_key, llm.model, GROQ_BASE_URL, llm.temperature)
        return cls(config.openai_api_key, llm.model, config.openai_base_url or None, llm.temperature)

    def stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        on_text_delta: Callable[[str], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
    ) -> tuple[str, list[dict]]:
        """Stream a chat completion.

        Calls on_text_delta for each content token as it arrives (for streaming
        TTS). Returns (full_text, tool_calls). Raises LLMAborted on barge-in.
        """
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools

        content_parts: list[str] = []
        pending: dict[int, ToolCall] = {}

        stream = self.client.chat.completions.create(stream=True, **kwargs)
        try:
            for chunk in stream:
                if should_abort is not None and should_abort():
                    raise LLMAborted()
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                if delta.content:
                    content_parts.append(delta.content)
                    if on_text_delta is not None:
                        on_text_delta(delta.content)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        slot = pending.setdefault(tc.index, ToolCall())
                        if tc.id:
                            slot.id = tc.id
                        if tc.function and tc.function.name:
                            slot.name = tc.function.name
                        if tc.function and tc.function.arguments:
                            slot.arguments += tc.function.arguments
        except LLMAborted:
            try:
                stream.close()
            except Exception:
                pass
            raise

        return "".join(content_parts), [p.complete() for p in pending.values()]
