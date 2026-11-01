"""Tier-0 system tools: apps, media, volume, screenshots, clipboard, info, URLs."""
from __future__ import annotations

import datetime as _dt
import ctypes
import os
import subprocess
import webbrowser

import psutil

from .registry import ToolContext, ToolRegistry, ToolResult

MEDIA_KEYS = {
    "play_pause": 0xB3,
    "next": 0xB0,
    "previous": 0xB1,
    "stop": 0xB2,
}


def _send_vk(vk: int) -> None:
    keybd = ctypes.windll.user32.keybd_event
    keybd(vk, 0, 0, 0)
    keybd(vk, 0, 2, 0)  # KEYEVENTF_KEYUP


def register_system_tools(reg: ToolRegistry) -> None:
    reg.tool(
        "open_app",
        "Open an application by name on this PC (e.g. 'chrome', 'spotify', 'word', 'notepad').",
        {"app_name": {"type": "string", "description": "App name, e.g. chrome, spotify, calculator"}},
        ["app_name"],
    )(lambda app_name, _ctx=None: _open_app(app_name))

    reg.tool(
        "close_app",
        "Close (terminate) an application's processes by name. DESTRUCTIVE: unsaved work is lost.",
        {"app_name": {"type": "string", "description": "App/process name, e.g. chrome, notepad"}},
        ["app_name"],
        confirm=True,
        confirm_question="Close that app? Unsaved work will be lost — yes or no?",
    )(lambda app_name, _ctx=None: _close_app(app_name))

    reg.tool(
        "list_running_apps",
        "List applications currently running.",
        {}, [],
    )(lambda _ctx=None: ToolResult("done", "running: " + ", ".join(_running_app_names())))

    reg.tool(
        "media_control",
        "Control media playback: play_pause, next, previous, or stop.",
        {"action": {"type": "string", "enum": list(MEDIA_KEYS)}},
        ["action"],
    )(lambda action, _ctx=None: _media(action))

    reg.tool(
        "set_volume",
        "Set speaker volume 0-100, or mute / unmute.",
        {"level": {"type": "integer", "description": "0-100"}, "mute": {"type": "boolean"}},
        [],
    )(lambda level=None, mute=None, _ctx=None: _volume(level, mute))

    reg.tool(
        "take_screenshot",
        "Capture the screen to a PNG file.",
        {}, [],
    )(lambda _ctx=None: _screenshot(_ctx))

    reg.tool(
        "clipboard",
        "Read or set the clipboard text.",
        {"text": {"type": "string", "description": "Text to put on clipboard; omit to read"}},
        [],
    )(lambda text=None, _ctx=None: _clipboard(text))

    reg.tool(
        "open_url",
        "Open a URL in the default browser.",
        {"url": {"type": "string", "description": "URL or domain, e.g. github.com"}},
        ["url"],
    )(lambda url, _ctx=None: _open_url(url))

    reg.tool(
        "press_keys",
        "Send a key or hotkey to the focused window, e.g. 'enter', 'ctrl+s', 'alt+tab', 'win+d', 'esc'.",
        {"keys": {"type": "string", "description": "Key combo like ctrl+shift+esc or a single key like f5"}},
        ["keys"],
    )(lambda keys, _ctx=None: _press_keys(keys))

    reg.tool(
        "type_text",
        "Type text into the currently focused window (as if typed on the keyboard).",
        {"text": {"type": "string"}},
        ["text"],
    )(lambda text, _ctx=None: _type_text(text))

    reg.tool(
        "system_info",
        "Get battery level, CPU load, RAM usage, disk space, or current time.",
        {}, [],
    )(lambda _ctx=None: ToolResult("done", _system_info()))

    reg.tool(
        "notify",
        "Show a Windows toast notification.",
        {"title": {"type": "string"}, "message": {"type": "string"}},
        ["title", "message"],
    )(lambda title, message, _ctx=None: _notify(title, message))

    reg.tool(
        "power_action",
        "Lock the PC or put it to sleep.",
        {"action": {"type": "string", "enum": ["lock", "sleep"]}},
        ["action"],
        confirm=True,
        confirm_question="Do it now — yes or no?",
    )(lambda action, _ctx=None: _power(action))

    reg.tool(
        "undo_last_action",
        "Undo the most recent undoable action (file moves/renames/copies done by JARVIS).",
        {}, [],
    )(lambda _ctx=None: _undo(_ctx))


# ---- implementations ----

def _proc_match(app_name: str) -> list[psutil.Process]:
    target = app_name.lower().removesuffix(".exe")
    hits = []
    for p in psutil.process_iter(["pid", "name"]):
        try:
            nm = (p.info["name"] or "").lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if nm and target in nm:
            hits.append(p)
    return hits


def _running_app_names() -> list[str]:
    seen = set()
    for p in psutil.process_iter(["name"]):
        try:
            nm = p.info["name"] or ""
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if nm.endswith(".exe") and not nm.lower().startswith(("svchost", "csrss", "dwm", "runtime", "system")):
            seen.add(nm[:-4])
    return sorted(seen)[:60]


def _open_app(app_name: str) -> ToolResult:
    name = app_name.strip()
    before = {p.pid for p in _proc_match(name)}
    # 1) direct launch (covers notepad, calc, chrome, explorer, ...)
    try:
        os.startfile(name)  # noqa: S606 — user-requested launch
        launched = True
    except OSError:
        launched = False
    # 2) installed-app lookup via Start menu (UWP + all installed apps)
    if not launched:
        try:
            script = (
                "Get-StartApps | Where-Object { $_.Name -like '*"
                + name.replace("'", "''") + "*' } | Select-Object -First 1 -ExpandProperty AppID"
            )
            appid = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True, text=True, timeout=15,
            ).stdout.strip()
            if appid:
                os.startfile("shell:AppsFolder\\" + appid)  # noqa: S606
                launched = True
        except Exception:
            pass
    # 3) Start-Process fallback
    if not launched:
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command", f"Start-Process '{name}'"],
                           capture_output=True, text=True, timeout=15)
            launched = True
        except Exception:
            pass
    if not launched:
        return ToolResult("failed", f"couldn't find an app called '{name}'")
    # verify: did a matching process appear?
    import time

    time.sleep(1.0)
    after = _proc_match(name)
    status = "unverified" if len(after) <= len(before) else "done"
    return ToolResult(
        status,
        f"launch attempted for '{name}'" + ("" if status == "done" else " (could not confirm it started)"),
        speak=f"Opening {name}" if status == "done" else f"I tried to open {name}, but I can't confirm it started",
    )


def _close_app(app_name: str) -> ToolResult:
    procs = _proc_match(app_name)
    procs = [p for p in procs if p.pid not in (0, 4, os.getpid())]
    if not procs:
        return ToolResult("failed", f"no running process matches '{app_name}'")
    killed = 0
    for p in procs:
        try:
            p.terminate()
            killed += 1
        except psutil.AccessDenied:
            try:
                p.kill()
                killed += 1
            except Exception:
                pass
    return ToolResult("done" if killed else "failed",
                      f"closed {killed} process(es) matching '{app_name}'",
                      speak=f"Closed {app_name}" if killed else f"Couldn't close {app_name}")


def _media(action: str) -> ToolResult:
    _send_vk(MEDIA_KEYS[action])
    return ToolResult("done", f"sent {action}", speak={"play_pause": "Toggled playback",
                     "next": "Next track", "previous": "Previous track", "stop": "Stopped"}.get(action, action))


def _volume(level=None, mute=None) -> ToolResult:
    from pycaw.pycaw import AudioUtilities

    dev = AudioUtilities.GetSpeakers()
    vol = dev.EndpointVolume  # IAudioEndpointVolume (pycaw >= 2026 exposes it directly)
    msgs = []
    if level is not None:
        lvl = max(0, min(100, int(level)))
        vol.SetMasterVolumeLevelScalar(lvl / 100.0, None)
        msgs.append(f"volume {lvl}%")
    if mute is not None:
        vol.SetMute(bool(mute), None)
        msgs.append("muted" if mute else "unmuted")
    return ToolResult("done", "; ".join(msgs) or f"volume is {int(vol.GetMasterVolumeLevelScalar() * 100)}%",
                      speak=(" and ".join(msgs)) or None)


def _screenshot(ctx: ToolContext) -> ToolResult:
    import mss
    import time as _t

    path = ctx.config.screenshot_dir / f"screenshot_{_t.strftime('%Y%m%d_%H%M%S')}.png"
    with mss.mss() as s:
        img = s.grab(s.monitors[1])
        mss.tools.to_png(img.rgb, img.size, output=str(path))
    return ToolResult("done", f"screenshot saved to {path}", speak="Screenshot saved")


def _clipboard(text=None) -> ToolResult:
    from pywinauto import clipboard as cb

    if text is None:
        try:
            data = cb.GetData()
        except Exception:
            data = ""
        return ToolResult("done", f"clipboard: {data!r}" if data else "clipboard is empty")
    cb.SetData(text)
    return ToolResult("done", "clipboard set", speak="Copied to clipboard")


def _open_url(url: str) -> ToolResult:
    if "://" not in url:
        url = "https://" + url
    webbrowser.open(url)
    return ToolResult("done", f"opened {url}", speak="Opening it now")


def _press_keys(keys: str) -> ToolResult:
    parts = [p.strip().lower() for p in keys.split("+") if p.strip()]
    if "win" in parts and len(parts) >= 1:
        _win_combo(parts)
        return ToolResult("done", f"sent keys {keys}", speak="Done")
    import pywinauto.keyboard as kb

    kb.send_keys(_escape_keys(parts), pause=0.03)
    return ToolResult("done", f"sent keys {keys}", speak="Done")


_WIN_VK = 0x5B  # left Windows key


def _win_combo(parts: list[str]) -> None:
    """Win+letter / Win+key combos via raw keybd_event."""
    keybd = ctypes.windll.user32.keybd_event
    other = [p for p in parts if p != "win"]
    letters = {"a": 0x41, "d": 0x44, "e": 0x45, "l": 0x4C, "r": 0x52,
               "s": 0x53, "tab": 0x09, "i": 0x49}
    keybd(_WIN_VK, 0, 0, 0)
    try:
        for p in other:
            vk = letters.get(p, ord(p[:1].upper()) if p[:1].isalpha() else 0)
            if vk:
                keybd(vk, 0, 0, 0)
                keybd(vk, 0, 2, 0)
    finally:
        keybd(_WIN_VK, 0, 2, 0)


_KEY_ALIASES = {
    "enter": "{ENTER}", "return": "{ENTER}", "tab": "{TAB}", "esc": "{ESC}",
    "escape": "{ESC}", "space": "{SPACE}", "backspace": "{BACKSPACE}",
    "delete": "{DELETE}", "del": "{DELETE}", "home": "{HOME}", "end": "{END}",
    "pageup": "{PGUP}", "pagedown": "{PGDN}", "up": "{UP}", "down": "{DOWN}",
    "left": "{LEFT}", "right": "{RIGHT}",
    "f1": "{F1}", "f2": "{F2}", "f3": "{F3}", "f4": "{F4}", "f5": "{F5}",
    "f6": "{F6}", "f7": "{F7}", "f8": "{F8}", "f9": "{F9}", "f10": "{F10}",
    "f11": "{F11}", "f12": "{F12}", "printscreen": "{PRTSC}",
}
_MODIFIER_PREFIX = {"ctrl": "^", "alt": "%", "shift": "+"}


def _escape_keys(parts: list[str]) -> str:
    """Translate ['ctrl','shift','esc'] into pywinauto send_keys syntax."""
    prefixes = ""
    key = ""
    for p in parts:
        if p in _MODIFIER_PREFIX:
            prefixes += _MODIFIER_PREFIX[p]
        else:
            key = _KEY_ALIASES.get(p) or _single_key(p)
    return prefixes + (key or "")


def _single_key(key: str) -> str:
    if len(key) == 1:
        return key
    return "{" + key.upper() + "}"


def _type_text(text: str) -> ToolResult:
    import pywinauto.keyboard as kb

    safe = text.replace("{", "{{").replace("}", "}}").replace("+", "{+}") \
               .replace("^", "{^}").replace("%", "{%}").replace("~", "{~}") \
               .replace("(", "{(}").replace(")", "{)}")
    kb.send_keys(safe, pause=0.005, with_spaces=True)
    return ToolResult("done", f"typed {len(text)} chars")


def _system_info() -> str:
    b = psutil.sensors_battery()
    vm = psutil.virtual_memory()
    lines = [
        f"time: {_dt.datetime.now().strftime('%A %Y-%m-%d %H:%M')}",
        f"CPU: {psutil.cpu_percent(interval=0.3):.0f}% busy",
        f"RAM: {vm.percent:.0f}% used ({vm.used // 2**30} / {vm.total // 2**30} GB)",
        f"Disk C: {psutil.disk_usage('C:\\').percent:.0f}% used",
    ]
    if b is not None:
        state = "charging" if b.power_plugged else "on battery"
        lines.append(f"Battery: {b.percent:.0f}% ({state})")
    return "\n".join(lines)


def _notify(title: str, message: str) -> ToolResult:
    from winotify import Notification

    Notification(app_id="JARVIS", title=title, msg=message).show()
    return ToolResult("done", "notification shown", speak="Notification sent")


def _power(action: str) -> ToolResult:
    if action == "lock":
        ctypes.windll.user32.LockWorkStation()
        return ToolResult("done", "PC locked", speak="Locking now")
    # sleep: SetSuspendState via rundll32 (0,1,0 = suspend, no force, disable wake false-ish)
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Application]::SetSuspendState('Suspend', $false, $false)"],
        capture_output=True, timeout=20,
    )
    return ToolResult("unverified", "sleep requested", speak="Going to sleep")


def _undo(ctx: ToolContext) -> ToolResult:
    if not ctx.undo_stack:
        return ToolResult("failed", "nothing to undo")
    label, fn = ctx.undo_stack.pop()
    res = fn()
    return ToolResult(res.status, f"undid: {label}. {res.text}", speak=f"Undid {label}")
