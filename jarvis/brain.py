"""Conversation brain: LLM + tool loop + streaming TTS + barge-in.

Streams the LLM response and starts speaking each sentence as soon as it
completes, so first audio lands right after the first sentence instead of
after the whole answer. Tool calls execute between stream rounds; their
`speak` lines are queued immediately while the next round runs.
"""
from __future__ import annotations

import json
import re
import time
from typing import Callable

from .providers.llm import LLMAborted, LLMClient
from .tools.registry import ToolContext, ToolRegistry

SYSTEM_PROMPT = """You are JARVIS, a capable personal AI assistant living on the user's Windows PC.
You control the computer through the tools you are given. You do not invent tools that do not exist.

Style: you are being SPOKEN aloud through TTS. Answer in short, natural, conversational
sentences. Never use markdown, lists, emoji, or code blocks unless the user explicitly asks
for something to be written to a file. Keep small talk to 1-2 sentences.

Behavior rules:
- If the user asks something you can do with a tool, call the tool immediately and briefly.
  Do not narrate before calling a tool; the result will be spoken.
- After a tool succeeds, summarize the outcome in one short sentence unless the user asked
  for detail.
- If a tool fails or is declined, say so honestly and suggest the closest alternative.
- The wake phrase ("hey jarvis") may appear at the start of the user's message: ignore it.
- If you are unsure what app/file the user means, ask one short clarifying question.
- For anything destructive (deleting, closing with unsaved work, powering off) the system
  will ask the user to confirm before running; do not try to bypass confirmations."""


class Brain:
    def __init__(self, config, registry: ToolRegistry, tts, speaker, memory=None,
                 should_abort: Callable[[], bool] | None = None):
        self.config = config
        self.registry = registry
        self.tts = tts
        self.speaker = speaker
        self.memory = memory
        self.should_abort = should_abort or (lambda: False)
        self.llm = LLMClient.from_config(config)
        self.ctx = ToolContext(config)
        self.on_delta: Callable[[str], None] | None = None   # UI hook
        self.on_tool: Callable[[str, str], None] | None = None  # UI hook (name, args)
        self.on_tool_result: Callable[[str, str], None] | None = None
        self.metrics: dict[str, float] = {}
        self._sent_buf = ""
        self._interrupted = False

    # ---------- conversation ----------

    def _messages(self, user_text: str) -> list[dict]:
        msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if self.memory is not None:
            for role, content in self.memory.last_turns(self.config.llm.history_turns):
                msgs.append({"role": role, "content": content})
        msgs.append({"role": "user", "content": user_text})
        return msgs

    def respond(self, user_text: str) -> str:
        """Full turn: stream answer, run tools, speak. Returns final text."""
        self._interrupted = False
        self._sent_buf = ""
        messages = self._messages(user_text)
        schemas = self.registry.openai_schemas()
        final_text_parts: list[str] = []
        t_first_token = None
        t_start = time.monotonic()

        try:
            for _ in range(self.config.llm.max_tool_rounds):
                text, tool_calls = self.llm.stream_chat(
                    messages, schemas,
                    on_text_delta=self._on_text_delta,
                    should_abort=self._abort_check,
                )
                if text:
                    final_text_parts.append(text)
                if t_first_token is None and (text or tool_calls):
                    t_first_token = time.monotonic() - t_start

                if not tool_calls:
                    break

                messages.append({
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {"id": tc["id"], "type": "function",
                         "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                        for tc in tool_calls
                    ],
                })
                for tc in tool_calls:
                    if self.on_tool is not None:
                        self.on_tool(tc["name"], tc["arguments"])
                    t0 = time.monotonic()
                    result = self.registry.call(tc["name"], tc["arguments"], self.ctx)
                    if self.on_tool_result is not None:
                        self.on_tool_result(tc["name"], f"{result.status}: {result.text}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": f"[{result.status}] {result.text}"[:4000],
                    })
                    # speak the tool's canned line while the LLM thinks
                    if result.speak and not self._interrupted:
                        self._speak_text(result.speak)
                    self.metrics[f"tool:{tc['name']}"] = round(time.monotonic() - t0, 2)
            else:
                self._speak_text("I hit my tool-use limit for that request.")

            self._flush_sentence(force=True)
        except LLMAborted:
            self._interrupted = True
            self.speaker.flush()
            final_text_parts = ["(interrupted)"]
            if self._sent_buf:
                final_text_parts = [self._sent_buf, "(interrupted)"]
                self._sent_buf = ""

        if t_first_token is not None:
            self.metrics["llm_first_token"] = round(t_first_token, 2)

        final = " ".join(final_text_parts).strip()
        if self.memory is not None and not self._interrupted:
            self.memory.append("user", user_text)
            self.memory.append("assistant", final)
        return final

    # ---------- streaming TTS ----------

    def _on_text_delta(self, delta: str) -> None:
        if self.on_delta is not None:
            self.on_delta(delta)
        self._sent_buf += delta
        # flush complete sentences for TTS (punctuation followed by whitespace;
        # this avoids splitting decimals like 3.5)
        matches = list(re.finditer(r"[.!?…](?=\s)", self._sent_buf))
        if matches:
            m = matches[-1]
            sentence = self._sent_buf[: m.end()].strip()
            self._sent_buf = self._sent_buf[m.end():]
            if sentence:
                self._speak_text(sentence)

    def _flush_sentence(self, force: bool = False) -> None:
        buf = self._sent_buf.strip()
        self._sent_buf = ""
        if buf and (force or len(buf) > 200):
            self._speak_text(buf)

    def _speak_text(self, text: str) -> None:
        if self._interrupted or not text.strip():
            return
        t0 = time.monotonic()
        try:
            pcm, rate = self.tts.synthesize(text)
        except Exception as e:
            self.metrics["tts_error"] = str(e)[:200]
            return
        if self.metrics.get("tts_first_audio") is None:
            self.metrics["tts_first_audio"] = round(time.monotonic() - t0, 2)
        self.speaker.play(pcm, rate)

    def _abort_check(self) -> bool:
        if self.should_abort():
            self._interrupted = True
            return True
        return False
