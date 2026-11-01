# JARVIS — voice-controlled AI assistant for Windows

A local-first, free-tier voice assistant: say **"Hey Jarvis"**, speak a command,
and it acts on your PC — opens apps, sorts files, answers questions, reads your
screen, and talks back. Architecture reverse-engineered from the best open
source projects (isair/jarvis, alexberardi/jarvis, Mark-LVI, Leon AI,
OpenVoiceOS, Windows-MCP, Open Interpreter) — see `docs/` research summary in
the repo history.

```
mic → openWakeWord ("hey jarvis") → Silero VAD → Groq whisper (STT)
    → Groq llama-3.3-70b (brain + tool calling) → 24 tools → Piper TTS
```

## Setup

```powershell
pip install -r requirements.txt
copy .env.example .env        # paste your FREE key from https://console.groq.com/keys
python scripts\test_mic.py    # verify the microphone
python run.py                 # say "Hey Jarvis, what's my battery level?"
```

## Modes

| Command | What it does |
|---|---|
| `python run.py` | wake word mode (always listening for "hey jarvis") |
| `python run.py --no-wake` | press Enter, then speak |
| `python run.py --text` | type instead of talk (test tools without a mic) |
| `python run.py --text --no-tts` | silent, fastest dev loop |
| `python run.py --tray` | system tray + notch, no console |

While listening, the console shows live wake-word scores (`hearing… wake score 0.18`)
so you can see how close you are to triggering. **Pressing Enter also wakes it** —
the wake word never blocks you. A **status notch** at the top of the screen shows
idle / listening / thinking / speaking. Interrupt JARVIS by clicking the mouse,
pressing Esc, or just talking over it.

## What it can do (24 tools)

- **System**: open/close apps, volume, media keys, screenshots, clipboard,
  hotkeys, typing into focused windows, toast notifications, lock/sleep,
  battery/CPU/RAM/disk/time.
- **Web**: DuckDuckGo search, weather (Open-Meteo), open URLs.
- **Files**: find/read/write, organize-folder-by-type, move/rename, recycle-bin
  delete, folder stats, and guarded PowerShell for bulk jobs.
- All destructive actions ask **"yes or no?"** out loud first; file writes,
  moves, and folder-organizing are undoable (`"undo that"`).

## Safety model

- Confirmation gate: every `confirm=True` tool is spoken-confirmed before running.
- Path allowlist: file tools refuse anything outside `allowed_dirs` (default: home).
- PowerShell denylist + 45 s timeout; blocklisted verbs (format, shutdown,
  reg delete...) always refused.
- Undo stack for reversible actions.
- Tools report honest outcomes (`done / unverified / failed / cancelled`) — the
  assistant verifies, e.g., that a launched process actually appeared.

## Configuration (`config.toml`)

- `[stt]` / `[llm]` / `[tts]` — provider + model. Groq is the default brain;
  swap `provider` to point at OpenAI, Ollama, or any OpenAI-compatible server.
- `[wakeword] threshold` — raise it if it triggers on TV speech.
- `[vad] end_silence_secs` — lower to 0.3 for snappier replies.
- `[tts] provider` — `auto` picks piper → kokoro → Windows SAPI.

## Project layout

```
jarvis/
  audio/     mic.py (WASAPI capture) · wake.py (openWakeWord) · vad.py (Silero)
             speaker.py (queued playback + barge-in flush)
  providers/ stt.py · llm.py (streaming + tool calls) · tts.py (piper/kokoro/sapi)
  tools/     registry.py (typed tools, confirm gate, undo) · system/web/file tools
  brain.py   streaming LLM loop: speaks each sentence as it's generated
  __main__.py voice state machine + text mode
scripts/     test_mic / test_wake / test_tts — run these when something's off
```

## Roadmap

- [x] Voice loop with wake word, streaming TTS, barge-in
- [x] 30 system/web/file/GUI tools with confirmations + undo
- [x] GUI control (UIA snapshots → click/type; Chrome-class apps; vision fallback)
- [x] System tray icon (`python run.py --tray`) + autostart
      (`powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1`)
- [x] Live-verified: Groq gpt-oss-120b tool calling, whisper-large-v3-turbo STT,
      Piper TTS, wake word, UIA control, volume/media/apps, web search

## Known limits

- Explorer's inner file view is UIA-opaque on some setups — JARVIS falls back
  to screenshot+vision there. Chrome/Edge expose everything.
- Groq's free catalog rotates: `llama-3.3-70b-versatile` was removed in 2026;
  default is now `openai/gpt-oss-120b`. Check `python -c` models.list if a
  404 appears — it's a one-line config change.
- Groq currently offers no vision model on the free tier, so vision_click
  is best-effort (`qwen/qwen3.8-27b`) and may fail honestly.
