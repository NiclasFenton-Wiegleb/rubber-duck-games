"""
Prompt service.

Loads the system prompt from a local text file and caches it. The cache is
keyed by file path + modification time, so editing the prompt file is picked up
on the next request without restarting the server, while normal requests pay
only a cheap `stat()` call.

Config keys used (all optional, with fallbacks so the service never hard-fails):
    SYSTEM_PROMPT_PATH  → path to the system prompt text file.
"""

from __future__ import annotations

import threading
from pathlib import Path

# Where to look for the prompt file if config doesn't specify one:
# backend/prompts/system_prompt.txt (this file lives in backend/api/services/).
_DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "system_prompt.txt"

# Used if the file is missing/unreadable so the endpoint still works.
_FALLBACK_PROMPT = "You are a concise, helpful coding assistant. /no_think"

# path -> (mtime, content)
_CACHE: dict[str, tuple[float, str]] = {}
_LOCK = threading.Lock()


def _resolve_path(config) -> Path:
    raw = None
    try:
        raw = config.get("SYSTEM_PROMPT_PATH")
    except AttributeError:
        raw = config["SYSTEM_PROMPT_PATH"] if "SYSTEM_PROMPT_PATH" in config else None
    return Path(raw) if raw else _DEFAULT_PROMPT_PATH


def get_system_prompt(config) -> str:
    """Return the system prompt text, loaded from file and cached by mtime."""
    path = _resolve_path(config)
    key = str(path)

    try:
        mtime = path.stat().st_mtime
    except OSError:
        return _FALLBACK_PROMPT

    cached = _CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError:
            return _FALLBACK_PROMPT
        if not content:
            content = _FALLBACK_PROMPT
        _CACHE[key] = (mtime, content)
        return content
