"""Tier-1 GUI control: read and operate apps through the Windows UI Automation
accessibility tree (no vision model needed), with a screenshot+vision fallback
for canvas/Electron surfaces UIA can't see.

Snapshot model: snapshot_window() returns numbered elements; the ids are valid
until the next snapshot of that window (they index the cached element list).
Risky buttons (delete/send/pay/...) ask for confirmation before clicking.
"""
from __future__ import annotations

import json

from .registry import ToolContext, ToolRegistry, ToolResult

# element ids stay valid until the next snapshot replaces the cache
_SNAPSHOTS: dict[str, list] = {}
_LAST_WINDOW: str | None = None

INTERACTIVE_TYPES = {
    "Button", "MenuItem", "Hyperlink", "CheckBox", "RadioButton", "ComboBox",
    "Edit", "Document", "TabItem", "ListItem", "TreeItem", "SplitButton",
    "ToggleButton", "DataItem", "Slider", "Custom",
}
_RISKY_WORDS = ("delete", "remove", "send", "submit", "pay", "purchase", "buy",
                "format", "shutdown", "restart", "sign out", "erase", "empty")


def _desktop():
    from pywinauto import Desktop

    return Desktop(backend="uia")


def _find_window(title: str | None):
    wins = _desktop().windows()
    if not title:
        # foreground window
        import ctypes

        hwnd = ctypes.windll.user32.GetForegroundWindow()
        for w in wins:
            if w.handle == hwnd:
                return w
        return wins[0] if wins else None
    t = title.lower()
    for w in wins:
        if t in w.window_text().lower():
            return w
    return None


def _window_key(w) -> str:
    return f"{w.handle}"


def _snapshot(w, max_elements: int = 120) -> list[tuple[str, object]]:
    """Return [(label, wrapper)] for interactive elements via UIA control view.

    Uses wrapper descendants(): the element-info walker filters out
    cross-process children (Explorer, Chrome content), descendants() does not.
    """
    items: list[tuple[str, object]] = []
    try:
        desc = w.descendants()
    except Exception:
        return items
    for wrapper in desc:
        try:
            info = wrapper.element_info
            ctrl_type = info.control_type or ""
            name = (info.name or "").strip()
        except Exception:
            continue
        if ctrl_type in INTERACTIVE_TYPES and (name or ctrl_type == "Edit"):
            items.append((f"{ctrl_type}: {name[:80]}", wrapper))
            if len(items) >= max_elements:
                break
    return items


def _label_and_info(w):
    key = _window_key(w)
    _SNAPSHOTS[key] = _snapshot(w)
    global _LAST_WINDOW
    _LAST_WINDOW = key
    return key, _SNAPSHOTS[key]


def register_gui_tools(reg: ToolRegistry) -> None:
    reg.tool(
        "list_windows",
        "List open windows (title + number) so one can be focused or inspected.",
        {}, [],
    )(lambda _ctx=None: _list_windows())

    reg.tool(
        "focus_window",
        "Bring a window to the foreground by (part of its) title.",
        {"title": {"type": "string"}},
        ["title"],
    )(lambda title, _ctx=None: _focus(title))

    reg.tool(
        "snapshot_window",
        "Inspect a window's controls: returns numbered elements (buttons, fields, links) that click_element / set_text can operate on. Omit title for the focused window.",
        {"title": {"type": "string"}},
        [],
    )(lambda title=None, _ctx=None: _snapshot_tool(title))

    reg.tool(
        "click_element",
        "Click a numbered element from the last snapshot_window result.",
        {"element_id": {"type": "integer"}},
        ["element_id"],
    )(lambda element_id, _ctx=None: _click(element_id, _ctx))

    reg.tool(
        "set_text",
        "Type text into a numbered Edit element from the last snapshot (replaces existing text).",
        {"element_id": {"type": "integer"}, "text": {"type": "string"}},
        ["element_id", "text"],
    )(lambda element_id, text, _ctx=None: _set_text(element_id, text, _ctx))

    reg.tool(
        "vision_click",
        "Click anywhere on screen by describing the target in words (screenshot + vision model fallback for apps UIA can't read).",
        {"instruction": {"type": "string", "description": "e.g. 'the blue Send button at the bottom'"}},
        ["instruction"],
    )(_vision_click)


def _list_windows() -> ToolResult:
    wins = _desktop().windows()
    lines = [f"- {i}: {w.window_text()[:90]}" for i, w in enumerate(wins) if w.window_text()]
    if not lines:
        return ToolResult("failed", "no open windows found")
    return ToolResult("done", "open windows:\n" + "\n".join(lines[:25]))


def _focus(title: str) -> ToolResult:
    w = _find_window(title)
    if w is None:
        return ToolResult("failed", f"no window matching '{title}'")
    try:
        w.set_focus()
        return ToolResult("done", f"focused '{w.window_text()[:80]}'", speak="Focused")
    except Exception as e:
        return ToolResult("failed", f"could not focus: {e}")


def _snapshot_tool(title: str | None) -> ToolResult:
    w = _find_window(title)
    if w is None:
        return ToolResult("failed", f"no window matching '{title or "(focused)"}'")
    key, items = _label_and_info(w)
    if not items:
        return ToolResult(
            "unverified",
            f"window '{w.window_text()[:60]}' exposes no UIA elements; "
            "it may need the vision_click fallback",
        )
    lines = [f"window: {w.window_text()[:80]}"] + [
        f"  [{i}] {label}" for i, (label, _) in enumerate(items)
    ]
    return ToolResult("done", "\n".join(lines))


def _click(element_id: int, ctx: ToolContext) -> ToolResult:
    if _LAST_WINDOW is None or element_id >= len(_SNAPSHOTS.get(_LAST_WINDOW, [])):
        return ToolResult("failed", "no such element — run snapshot_window first")
    label, wrapper = _SNAPSHOTS[_LAST_WINDOW][element_id]
    risky = any(w in label.lower() for w in _RISKY_WORDS)
    if risky and not ctx.request_confirmation(f"That's a '{label}' action — proceed? yes or no?"):
        return ToolResult("cancelled", "user declined the click")
    try:
        rect = wrapper.rectangle()
        import pywinauto.mouse as mouse

        mouse.click(coords=(rect.left + rect.width() // 2, rect.top + rect.height() // 2))
        return ToolResult("done", f"clicked [{element_id}] {label}", speak="Done")
    except Exception as e:
        return ToolResult("failed", f"click failed: {e}")


def _set_text(element_id: int, text: str, ctx: ToolContext) -> ToolResult:
    if _LAST_WINDOW is None or element_id >= len(_SNAPSHOTS.get(_LAST_WINDOW, [])):
        return ToolResult("failed", "no such element — run snapshot_window first")
    label, wrapper = _SNAPSHOTS[_LAST_WINDOW][element_id]
    try:
        rect = wrapper.rectangle()
        import pywinauto.keyboard as kb
        import pywinauto.mouse as mouse

        mouse.click(coords=(rect.left + rect.width() // 2, rect.top + rect.height() // 2))
        kb.send_keys("^a{DELETE}")
        safe = text.replace("{", "{{").replace("}", "}}").replace("+", "{+}") \
                   .replace("^", "{^}").replace("%", "{%}").replace("~", "{~}") \
                   .replace("(", "{(}").replace(")", "{)}")
        kb.send_keys(safe, pause=0.005, with_spaces=True)
        return ToolResult("done", f"typed {len(text)} chars into [{element_id}] {label}", speak="Typed it")
    except Exception as e:
        return ToolResult("failed", f"set_text failed: {e}")


def _vision_click(instruction: str, _ctx: ToolContext | None = None) -> ToolResult:
    """Screenshot → Groq vision model → normalized coords → click."""
    import base64

    import mss

    from ..config import get_config

    config = get_config()
    if not config.groq_api_key:
        return ToolResult("failed", "vision fallback needs GROQ_API_KEY in .env")

    with mss.mss() as s:
        mon = s.monitors[1]
        img = s.grab(mon)
        from mss.tools import to_png

        path = config.screenshot_dir / "_vision.png"
        to_png(img.rgb, img.size, output=str(path))
    b64 = base64.b64encode(path.read_bytes()).decode()

    from openai import OpenAI

    client = OpenAI(api_key=config.groq_api_key, base_url="https://api.groq.com/openai/v1")
    prompt = (
        "You are a screen-grounding model. Locate this target on the screenshot: "
        f"'{instruction}'. Reply with ONLY a JSON object: "
        '{"x": <0-1000>, "y": <0-1000>} where coordinates are the click point '
        "scaled so the full image width=1000 and height=1000. No other text."
    )
    try:
        resp = client.chat.completions.create(
            model=config.llm.vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            temperature=0.0, max_tokens=100,
        )
        raw = resp.choices[0].message.content or ""
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        nx, ny = float(data["x"]) / 1000.0, float(data["y"]) / 1000.0
        import pywinauto.mouse as mouse

        abs_x = int(mon["left"] + nx * mon["width"])
        abs_y = int(mon["top"] + ny * mon["height"])
        mouse.click(coords=(abs_x, abs_y))
        return ToolResult("unverified",
                          f"clicked at screen ({abs_x},{abs_y}) for '{instruction}'; "
                          "take a screenshot or snapshot_window to verify",
                          speak="Clicked it")
    except Exception as e:
        return ToolResult("failed", f"vision click failed: {e}")
