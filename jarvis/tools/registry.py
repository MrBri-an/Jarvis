"""Typed tool registry: the LLM selects tools, it does not write them.

Every tool returns a ToolResult with an honest outcome status
(done / unverified / failed / cancelled), per the pattern from Mark-LVI,
alexberardi/jarvis, and Leon AI. Destructive tools declare confirm=True and
are gated behind a spoken yes/no via ToolContext.confirm_fn. Reversible tools
push an undo callable onto the context stack.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolResult:
    status: str = "done"          # done | unverified | failed | cancelled | started
    text: str = ""                # what the LLM sees / what we say
    speak: str | None = None      # override spoken phrasing
    data: Any = None


@dataclass
class Tool:
    name: str
    description: str
    params: dict                  # JSON-schema properties
    required: list[str]
    fn: Callable[..., ToolResult]
    confirm: bool = False
    confirm_question: str = ""
    tier: int = 0


class ToolContext:
    """Runtime services a tool may need; created by the brain per session."""

    def __init__(self, config):
        self.config = config
        self.confirm_fn: Callable[[str], bool] | None = None
        self.undo_stack: list[tuple[str, Callable[[], ToolResult]]] = []

    def request_confirmation(self, question: str) -> bool:
        if self.confirm_fn is None:
            return True  # text mode without confirmation UI: allow but note
        return self.confirm_fn(question)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool {tool.name}")
        self._tools[tool.name] = tool

    def tool(self, name: str, description: str, params: dict | None = None,
             required: list[str] | None = None, confirm: bool = False,
             confirm_question: str = "", tier: int = 0) -> Callable:
        """Decorator: @registry.tool("open_app", "Open an application", {...})"""
        params = params or {}
        required = required or []

        def deco(fn: Callable[..., ToolResult]) -> Callable:
            self.register(Tool(
                name=name, description=description, params=params,
                required=required, fn=fn, confirm=confirm,
                confirm_question=confirm_question, tier=tier,
            ))
            return fn

        return deco

    def names(self) -> list[str]:
        return list(self._tools)

    def openai_schemas(self) -> list[dict]:
        """Tools in OpenAI/Groq function-calling schema format.

        Compact on purpose: descriptions are clipped and parameter
        descriptions dropped — the schema rides along on EVERY tool-path
        request, so its size is time-to-first-audio.
        """
        out = []
        for t in self._tools.values():
            props = {}
            for k, v in t.params.items():
                props[k] = {"type": v.get("type", "string"), **({"enum": v["enum"]} if "enum" in v else {})}
            out.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description[:110],
                    "parameters": {"type": "object", "properties": props, "required": t.required},
                },
            })
        return out

    def call(self, name: str, arguments_json: str, ctx: ToolContext) -> ToolResult:
        t = self._tools.get(name)
        if t is None:
            return ToolResult("failed", f"unknown tool '{name}'")
        try:
            args = json.loads(arguments_json) if arguments_json.strip() else {}
        except json.JSONDecodeError as e:
            return ToolResult("failed", f"bad arguments JSON: {e}")
        missing = [k for k in t.required if k not in args]
        if missing:
            return ToolResult("failed", f"missing required arguments: {missing}")

        if t.confirm and ctx.config.tools.require_confirmation:
            q = t.confirm_question or f"Run {t.name} with {args}?"
            if not ctx.request_confirmation(q):
                return ToolResult("cancelled", "The user declined the action.")

        try:
            return t.fn(**args, _ctx=ctx) if _wants_ctx(t.fn) else t.fn(**args)
        except Exception as e:  # never let a tool crash the assistant
            return ToolResult("failed", f"{type(e).__name__}: {e}")

    def push_undo(self, ctx: ToolContext, label: str, undo_fn: Callable[[], ToolResult]) -> None:
        ctx.undo_stack.append((label, undo_fn))


def _wants_ctx(fn: Callable) -> bool:
    import inspect

    return "_ctx" in inspect.signature(fn).parameters
