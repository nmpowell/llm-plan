"""Plan discovery: alias resolution across user, environment and bundled dirs."""

from __future__ import annotations

from pathlib import Path

import llm


def user_plan_dir() -> Path:
    """Return the user's personal plans directory, creating it if needed."""
    path = llm.user_dir() / "plans"
    path.mkdir(parents=True, exist_ok=True)
    return path
