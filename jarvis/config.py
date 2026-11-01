"""Configuration loader: config.toml + .env secrets."""
from __future__ import annotations

import os
import pathlib
import tomllib
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
HOME = pathlib.Path.home()


def _expand(p: str) -> pathlib.Path:
    return pathlib.Path(os.path.expanduser(p))


def _ns(d: dict) -> SimpleNamespace:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _ns(v)
        else:
            out[k] = v
    return SimpleNamespace(**out)


def load_env() -> None:
    """Populate os.environ from ROOT/.env (does not override existing)."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


class Config:
    def __init__(self, path: pathlib.Path | None = None):
        load_env()
        cfg_path = path or ROOT / "config.toml"
        with open(cfg_path, "rb") as f:
            raw = tomllib.load(f)
        self.raw = raw
        c = _ns(raw)

        self.wakeword = c.wakeword
        self.vad = c.vad
        self.audio = c.audio
        self.llm = c.llm
        self.tools = c.tools

        self.stt = c.stt
        self.tts = c.tts
        self.general = c.general

        self.groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self.openai_base_url = os.environ.get("OPENAI_BASE_URL", "").strip()

        self.allowed_dirs = [_expand(p) for p in c.tools.allowed_dirs]
        self.screenshot_dir = _expand(c.tools.screenshot_dir)
        MODELS_DIR.mkdir(exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    @property
    def has_cloud_key(self) -> bool:
        if self.llm.provider == "groq":
            return bool(self.groq_api_key)
        if self.llm.provider == "gemini":
            return bool(self.gemini_api_key)
        return bool(self.openai_api_key)

    def api_key(self) -> str:
        if self.llm.provider == "groq":
            return self.groq_api_key
        if self.llm.provider == "gemini":
            return self.gemini_api_key
        return self.openai_api_key


_config: Config | None = None


def get_config(path: pathlib.Path | None = None) -> Config:
    global _config
    if _config is None or path is not None:
        _config = Config(path)
    return _config
