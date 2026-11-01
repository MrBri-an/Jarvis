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

SYSTEM_PROMPT = """You are JARVIS, the user's personal AI assistant living on their Windows PC — highly knowledgeable, quick-witted, and genuinely helpful.

You can answer ANY question: science, history, technology, sports, movies, culture, math, code, health, finance, advice, opinions. Give real, substantive answers from your own knowledge — never deflect with "I'm just an assistant" or "I can only control your computer". You are also a great conversationalist: discuss ideas, debate, brainstorm, explain hard things simply, and joke when it fits.

Spoken style: your replies are read aloud by TTS. Always reply in English (or whichever language the user is speaking) — never switch languages on your own. Natural conversational sentences; no markdown, lists, emoji, or code fences unless the user asks you to write something to a file. Match depth to the question — one crisp sentence for "what time is it", several sentences when explaining or discussing. Lead with the answer, then the interesting context. Prefer concrete facts; when unsure, say what you know and flag the uncertainty honestly.

Tools: you control the PC through tools. When the user asks for an ACTION or live info (apps, files, system state, weather, web/Wikipedia lookups, screen control), call the right tool immediately without narrating first, then summarize the outcome briefly. For pure knowledge or conversation, just answer — no tool needed. If a tool fails or is declined, say so honestly and offer the closest alternative. Never invent tool results. The wake phrase "hey jarvis" may prefix the message; ignore it. Destructive actions are confirmation-gated by the system; never try to bypass confirmations."""

# Messages that are pure knowledge/chat skip the tool catalog entirely: the
# request prompt shrinks by ~1000+ tokens, cutting time-to-first-audio a lot.
_ACTION_WORDS = (
    "open ", "close ", "launch", "run ", "kill ", "terminate", "volume", "mute",
    "play ", "pause", "next track", "previous track", "skip ", "screenshot",
    "screen record", "clipboard", "copy this", "paste", "type ", "press ",
    "hotkey", "notify", "notification", "lock ", "sleep", "shut down",
    "shutdown", "restart", "battery", "cpu", "ram", "storage", "disk", "memory usage",
    "weather", "forecast", "search", "google", "look up", "find ", "file",
    "folder", "downloads", "documents", "organize", "tidy", "move ", "rename",
    "delete", "recycle", "my pc", "my computer", "this computer", "my laptop",
    "window", "app ", "chrome", "edge", "browser", "notepad", "spotify",
    "excel", "word", "vs code", "task manager", "click", "scroll", "tab ",
    "settings", "wifi", "bluetooth", "bluetooth", "brightness", "my screen",
)

_CONVO_MARKER = re.compile(
    r"\b(explain|why|who|what are|what is|what was|tell me about|your opinion|"
    r"do you think|joke|story|poem|translate|how does|how do|compare|"
    r"difference between|define|meaning of|help me understand|debate|"
    r"discuss|teach me|interesting fact|summarize|summarise|calculate|"
    r"solve|convert|write me|write a)\b",
    re.IGNORECASE,
)

# "explain deeply", "think hard", "in detail" … — noted for the model via the
# system prompt; the reasoning dial itself stayed at 'low' because Groq free
# tier's queue time, not the dial, dominates deep-turn latency.


def needs_tools(text: str) -> bool:
    """True unless the message is confidently pure knowledge/chat."""
    t = text.lower()
    if any(w in t for w in _ACTION_WORDS):
        return True
    if _CONVO_MARKER.search(t):
        return False
    return True  # unsure → keep tools available


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
        schemas = self.registry.openai_schemas() if needs_tools(user_text) else None
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
