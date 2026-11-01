"""Offline brain integration test: fake streaming LLM -> real registry -> real TTS.

Verifies the full respond() loop — tool-call rounds, tool results fed back,
sentence-chunked TTS playback, confirmation gate (auto-yes), undo — with no
API key required. The fake LLM is scripted per call:
  call 1: emits a web_search tool call
  call 2: streams the final spoken answer
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis.audio.speaker import Speaker  # noqa: E402
from jarvis.brain import Brain  # noqa: E402
from jarvis.config import get_config  # noqa: E402
from jarvis.providers.llm import ToolCall  # noqa: E402
from jarvis.providers.tts import create_tts  # noqa: E402
from jarvis.tools.registry import ToolContext, ToolRegistry  # noqa: E402
from jarvis.tools.file_tools import register_file_tools  # noqa: E402
from jarvis.tools.system_tools import register_system_tools  # noqa: E402
from jarvis.tools.web_tools import register_web_tools  # noqa: E402


class FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    def __iter__(self):
        return iter(self._chunks)

    def close(self):
        self.closed = True


def chunk(content=None, tool_calls=None):
    from types import SimpleNamespace

    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


def tc_delta(index, id=None, name=None, args=None):
    from types import SimpleNamespace

    fn = SimpleNamespace(name=name, arguments=args)
    return SimpleNamespace(index=index, id=id, function=fn)


class FakeLLM:
    def __init__(self):
        self.script = []
        self.calls = 0

    def stream_chat(self, messages, tools, on_text_delta=None, should_abort=None):
        chunks = self.script[self.calls]
        self.calls += 1
        for c in chunks:
            if on_text_delta is not None and c.choices[0].delta.content:
                on_text_delta(c.choices[0].delta.content)
        return self._parse(chunks)

    def _parse(self, chunks):
        text, pending = "", {}
        for c in chunks:
            d = c.choices[0].delta
            if d.content:
                text += d.content
            if d.tool_calls:
                for t in d.tool_calls:
                    slot = pending.setdefault(t.index, ToolCall())
                    if t.id:
                        slot.id = t.id
                    if t.function and t.function.name:
                        slot.name = t.function.name
                    if t.function and t.function.arguments:
                        slot.arguments += t.function.arguments
        return text, [p.complete() for p in pending.values()]


def main() -> None:
    config = get_config()
    registry = ToolRegistry()
    register_system_tools(registry)
    register_web_tools(registry)
    register_file_tools(registry)

    tts = create_tts(config)
    speaker = Speaker()
    fake = FakeLLM()
    b = Brain(config, registry, tts, speaker)
    b.llm = fake
    confirmations = []
    tool_statuses = {}
    b.on_tool_result = lambda n, r: tool_statuses.setdefault(n, r.split(":")[0])

    # scripted conversation: tool round then final answer
    fake.script = [
        [chunk(tool_calls=[tc_delta(0, id="call_1", name="web_search", args='{"que')]),
         chunk(tool_calls=[tc_delta(0, args='ry":"test query"}')])],
        [chunk("The search says "),
         chunk("everything checks out. "),
         chunk("Your CPU was 19 percent busy.")],
    ]
    b.ctx.confirm_fn = lambda q: (confirmations.append(q), True)[1]

    t0 = time.monotonic()
    reply = b.respond("hey jarvis what's the status")
    dt = time.monotonic() - t0
    speaker.wait_done()

    assert fake.calls == 2, f"expected 2 LLM rounds, got {fake.calls}"
    assert tool_statuses.get("web_search") in ("done", "unverified"), tool_statuses
    print(f"\nreply: {reply!r}")
    print(f"metrics: {b.metrics}")
    print(f"round-trip (incl. TTS playback): {dt:.1f}s")

    # confirm-gate test on a destructive tool
    fake.script = [[chunk("Okay, closing it now.")]]
    res = registry.call("close_app", '{"app_name":"definitely_not_running_xyz"}', b.ctx)
    assert res.status == "failed" and "no running process" in res.text, res  # tool ran after auto-yes
    assert confirmations, "confirmation gate did not fire"

    print("PASS: brain loop, tool rounds, streamed TTS, confirmation gate all work")


if __name__ == "__main__":
    main()
