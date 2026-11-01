"""File/productivity tools (Tier 3) with guardrails.

Rules enforced here regardless of what the LLM asks for:
- All paths must resolve inside config.tools.allowed_dirs (default: home).
- Writes/moves/renames/deletes are confirm=True and push undo callables.
- run_powershell has a blocklist, a timeout, and captures output; it always
  requires confirmation. Treat it as a convenience, not a sandbox.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import time

from .registry import ToolContext, ToolRegistry, ToolResult

_BLOCKED_PS = (
    "remove-item", "rd ", "del ", "erase ", "format", "diskpart",
    "shutdown", "restart-computer", "stop-computer", "reg delete",
    "cipher /w", "vssadmin", "bcdedit", "set-executionpolicy",
)


def _safe_path(ctx: ToolContext, raw: str) -> pathlib.Path:
    p = pathlib.Path(os.path.expanduser(str(raw))).resolve()
    for root in ctx.config.allowed_dirs:
        try:
            p.relative_to(root.resolve())
            return p
        except ValueError:
            continue
    raise PermissionError(f"path '{raw}' is outside allowed directories {ctx.config.allowed_dirs}")


def register_file_tools(reg: ToolRegistry) -> None:
    reg.tool(
        "find_files",
        "Find files/folders by glob pattern (e.g. '*.pdf' in ~/Downloads). Returns up to 30 matches.",
        {"pattern": {"type": "string"}, "folder": {"type": "string", "description": "Defaults to home"}},
        ["pattern"],
    )(lambda pattern, folder=None, _ctx=None: _find(pattern, folder, _ctx))

    reg.tool(
        "read_file",
        "Read a text file (first ~4000 chars).",
        {"path": {"type": "string"}},
        ["path"],
    )(lambda path, _ctx=None: _read(path, _ctx))

    reg.tool(
        "write_file",
        "Create or overwrite a text file.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"],
        confirm=True, confirm_question="Write that file — yes or no?",
    )(lambda path, content, _ctx=None: _write(path, content, _ctx))

    reg.tool(
        "organize_folder",
        "Sort a folder's loose files into subfolders by type (Images, Documents, Audio, Video, Archives, Installers, Code, Other).",
        {"folder": {"type": "string", "description": "e.g. ~/Downloads"}},
        ["folder"],
        confirm=True, confirm_question="Sort those files into type folders — yes or no?",
    )(lambda folder, _ctx=None: _organize(folder, _ctx))

    reg.tool(
        "move_or_rename",
        "Move or rename a file/folder.",
        {"source": {"type": "string"}, "destination": {"type": "string"}},
        ["source", "destination"],
        confirm=True, confirm_question="Move or rename it — yes or no?",
    )(lambda source, destination, _ctx=None: _move(source, destination, _ctx))

    reg.tool(
        "delete_path",
        "Send a file or folder to the Recycle Bin.",
        {"path": {"type": "string"}},
        ["path"],
        confirm=True, confirm_question="Send it to the Recycle Bin — yes or no?",
    )(lambda path, _ctx=None: _delete(path, _ctx))

    reg.tool(
        "folder_stats",
        "Total size and file count of a folder.",
        {"folder": {"type": "string"}},
        ["folder"],
    )(lambda folder, _ctx=None: _stats(folder, _ctx))

    reg.tool(
        "run_powershell",
        "Run a short PowerShell command for bulk/system tasks nothing else covers (always asks first).",
        {"script": {"type": "string", "description": "PowerShell one-liner, no interactive prompts"},
         "reason": {"type": "string", "description": "Why this command is needed"}},
        ["script", "reason"],
        confirm=True, confirm_question="Run that PowerShell command — yes or no?",
    )(_run_powershell)


def _find(pattern: str, folder, ctx: ToolContext) -> ToolResult:
    base = _safe_path(ctx, folder or "~")
    hits = []
    for p in base.rglob(pattern):
        hits.append(p)
        if len(hits) >= 30:
            break
    if not hits:
        return ToolResult("failed", f"no matches for '{pattern}' under {base}")
    return ToolResult("done", "found:\n" + "\n".join(str(h) for h in hits))


def _read(path: str, ctx: ToolContext) -> ToolResult:
    p = _safe_path(ctx, path)
    text = p.read_text(encoding="utf-8", errors="replace")
    trimmed = text[:4000] + ("\n...[truncated]" if len(text) > 4000 else "")
    return ToolResult("done", f"{p}:\n{trimmed}")


def _write(path: str, content: str, ctx: ToolContext) -> ToolResult:
    p = _safe_path(ctx, path)
    old = p.read_bytes() if p.exists() else None
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

    def undo() -> ToolResult:
        if old is None:
            p.unlink(missing_ok=True)
        else:
            p.write_bytes(old)
        return ToolResult("done", f"reverted {p}")

    ctx.undo_stack.append((f"write {p}", undo))
    return ToolResult("done", f"wrote {p} ({len(content)} chars)", speak="File saved")


def _organize(folder: str, ctx: ToolContext) -> ToolResult:
    base = _safe_path(ctx, folder)
    if not base.is_dir():
        return ToolResult("failed", f"not a folder: {base}")
    groups = {
        "Images": {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".heic", ".svg"},
        "Documents": {".pdf", ".docx", ".doc", ".txt", ".md", ".xlsx", ".pptx", ".csv", ".rtf", ".odt"},
        "Audio": {".mp3", ".wav", ".flac", ".m4a", ".ogg"},
        "Video": {".mp4", ".mkv", ".mov", ".avi", ".webm"},
        "Archives": {".zip", ".rar", ".7z", ".tar", ".gz"},
        "Installers": {".exe", ".msi", ".msix"},
        "Code": {".py", ".js", ".ts", ".html", ".css", ".json", ".ipynb", ".sh", ".ps1", ".java", ".c", ".cpp"},
    }
    moved = []
    for f in sorted(base.iterdir()):
        if f.is_dir() or f.name.startswith("."):
            continue
        ext = f.suffix.lower()
        dest_name = next((g for g, exts in groups.items() if ext in exts), "Other")
        dest = base / dest_name
        dest.mkdir(exist_ok=True)
        target = dest / f.name
        if target.exists():
            target = dest / f"{f.stem}_{int(time.time())}{f.suffix}"
        shutil.move(str(f), str(target))
        moved.append((target, f))

    def undo() -> ToolResult:
        for target, original in moved:
            shutil.move(str(target), str(original))
        return ToolResult("done", f"restored {len(moved)} files")

    ctx.undo_stack.append((f"organize {base}", undo))
    return ToolResult("done" if moved else "failed",
                      f"sorted {len(moved)} files into type folders" if moved else "nothing to sort",
                      speak=f"Sorted {len(moved)} files into folders" if moved else "Nothing to organize")


def _move(source: str, destination: str, ctx: ToolContext) -> ToolResult:
    src = _safe_path(ctx, source)
    dst = _safe_path(ctx, destination)
    if dst.exists() and dst.is_dir():
        dst = dst / src.name
    shutil.move(str(src), str(dst))

    def undo() -> ToolResult:
        shutil.move(str(dst), str(src))
        return ToolResult("done", f"moved back {dst} -> {src}")

    ctx.undo_stack.append((f"move {src.name}", undo))
    return ToolResult("done", f"moved to {dst}", speak="Done")


def _delete(path: str, ctx: ToolContext) -> ToolResult:
    p = _safe_path(ctx, path)
    if not p.exists():
        return ToolResult("failed", f"not found: {p}")
    # Recycle Bin via PowerShell VisualBasic FileSystem (no extra deps)
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"Add-Type -AssemblyName Microsoft.VisualBasic; "
         f"[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory('{p.parent}', 'OnlyErrorDialogs', 'SendToRecycleBin')"
         if p.is_dir() else
         f"Add-Type -AssemblyName Microsoft.VisualBasic; "
         f"[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile('{p}', 'OnlyErrorDialogs', 'SendToRecycleBin')"],
        capture_output=True, text=True, timeout=30,
    )
    if p.exists():
        return ToolResult("failed", f"could not recycle {p}")
    return ToolResult("done", f"sent {p} to the Recycle Bin", speak="Moved to the recycle bin")


def _stats(folder: str, ctx: ToolContext) -> ToolResult:
    base = _safe_path(ctx, folder)
    total = count = 0
    for p in base.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
                count += 1
            except OSError:
                pass
    return ToolResult("done", f"{base}: {count} files, {total / 2**20:.1f} MB")


def _run_powershell(script: str, reason: str, ctx: ToolContext) -> ToolResult:
    low = script.lower()
    if any(b in low for b in _BLOCKED_PS):
        return ToolResult("failed",
                          "blocked: that command is on the denylist (format/delete/shutdown/registry). "
                          "Ask the user to do it manually.")
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=45,
        )
    except subprocess.TimeoutExpired:
        return ToolResult("failed", "command timed out after 45s")
    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()
    text = f"exit={r.returncode}" + (f"\n{out[:3000]}" if out else "") + (f"\nERR: {err[:1500]}" if err else "")
    status = "done" if r.returncode == 0 else "failed"
    return ToolResult(status, text, speak=None)
