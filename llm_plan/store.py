"""Plan discovery: alias resolution across user, environment and bundled dirs.

Search order (first match wins):
  1. an explicit file path
  2. $LLM_PLAN_DIRS (colon-separated directories)
  3. <llm user dir>/plans/
  4. plans bundled with this package

An alias ``X`` matches ``plan_X.yaml``, ``X.yaml``, ``plan_X.yml`` or
``X.yml``; files starting with ``_`` are shared includes, not plans.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import llm
import yaml

from .models import PlanError

BUNDLED_PLAN_DIR = Path(__file__).parent / "plans"


@dataclass(frozen=True)
class PlanListing:
    alias: str
    path: Path
    name: str
    summary: str


def user_plan_dir() -> Path:
    """Return the user's personal plans directory, creating it if needed."""
    path = llm.user_dir() / "plans"
    path.mkdir(parents=True, exist_ok=True)
    return path


def search_dirs() -> list[Path]:
    """Plan directories in precedence order."""
    env = os.environ.get("LLM_PLAN_DIRS", "")
    dirs = [Path(d).expanduser() for d in env.split(":") if d.strip()]
    dirs.append(user_plan_dir())
    dirs.append(BUNDLED_PLAN_DIR)
    return [d for d in dirs if d.is_dir()]


def is_alias(value: str) -> bool:
    """True if ``value`` is a bare alias rather than a file path."""
    if not value:
        return False
    if "/" in value or "\\" in value or value.startswith("~"):
        return False
    return not value.endswith((".yaml", ".yml"))


def resolve_plan(value: str) -> Path:
    """Resolve an alias or path to a plan file."""
    if not is_alias(value):
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise PlanError(f"Plan file not found: {path}")
        return path

    candidates = [f"plan_{value}.yaml", f"{value}.yaml", f"plan_{value}.yml", f"{value}.yml"]
    for directory in search_dirs():
        for candidate in candidates:
            path = directory / candidate
            if path.is_file():
                return path.resolve()

    searched = ", ".join(str(d) for d in search_dirs())
    raise PlanError(
        f"Plan '{value}' not found: no {candidates[0]} or {candidates[1]} in "
        f"[{searched}]. Add one to {user_plan_dir()} or set LLM_PLAN_DIRS."
    )


def list_plans() -> list[PlanListing]:
    """Discover plans in every search location; earlier locations shadow later."""
    listings: dict[str, PlanListing] = {}
    for directory in search_dirs():
        for path in sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml")):
            if path.name.startswith("_"):
                continue
            alias = path.stem.removeprefix("plan_")
            if alias in listings:
                continue
            name = summary = None
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    name = data.get("name")
                    summary = data.get("summary")
            except (yaml.YAMLError, OSError, UnicodeDecodeError):
                pass
            listings[alias] = PlanListing(
                alias=alias,
                path=path.resolve(),
                name=str(name) if name else alias,
                summary=str(summary) if summary else "",
            )
    return sorted(listings.values(), key=lambda listing: listing.alias)
