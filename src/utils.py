"""
utils.py
--------
Small shared helpers used by app.py. Kept intentionally minimal.
"""

from __future__ import annotations

from datetime import datetime


def timestamp() -> str:
    """Short human-readable timestamp for history entries."""
    return datetime.now().strftime("%H:%M:%S")


def truncate(text: str, max_len: int = 60) -> str:
    """Truncate text for compact table display, adding an ellipsis if cut."""
    text = text or ""
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"
