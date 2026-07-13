"""Data structures for plans, stages and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class PlanError(Exception):
    """A plan could not be loaded, validated or executed."""


class PromptType(Enum):
    """How a single prompt string is interpreted."""

    NONE = "none"
    CLI = "cli"  # CLI-supplied instructions and/or files
    FILE = "file"  # read from a file path
    INLINE = "inline"  # literal text ("inline:" prefix)
    CHAIN = "chain"  # another stage's output text ("chain:" prefix)


@dataclass(frozen=True)
class FileRef:
    """A file included in a stage's prompt, with an optional section label."""

    path: Path
    label: str | None = None


@dataclass(frozen=True)
class AttachmentRef:
    """An attachment (local path or URL) passed to a stage's model."""

    uri: str
    label: str | None = None

    @property
    def is_url(self) -> bool:
        return self.uri.startswith(("http://", "https://"))


@dataclass(frozen=True)
class PromptSpec:
    """A stage's prompt specification, parsed and ready to compose.

    ``sections`` holds (label, text) pairs from inline/chain/CLI-instruction
    parts, in the order the plan listed them. ``requires_cli_content`` is set
    by a plain ``CLI``/``CLI:all`` token: the stage must receive *some* CLI
    input or the run is a user error.
    """

    prompt_files: list[FileRef] = field(default_factory=list)
    sections: list[tuple[str | None, str]] = field(default_factory=list)
    wants_cli_files: bool = False
    requires_cli_content: bool = False


@dataclass
class Stage:
    """One node in the plan DAG."""

    name: str
    summary: str
    type: str = "llm"  # "llm" | "python_script"

    # LLM stages
    model: str | None = None
    prompt: str | list | None = None
    prompt_label: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    # Script stages
    script: str | None = None
    script_args: list[str] = field(default_factory=list)
    python: str = "python3"
    timeout: int | None = None
    env: dict[str, str] = field(default_factory=dict)

    # DAG / chaining
    depends_on: list[str] = field(default_factory=list)
    produces: str | None = None

    # Scheduling
    exclusive: bool = False
    continue_on_failure: bool = False
    partial_dependencies: bool = False

    # Resolved during parsing (paths validated against the plan's directory)
    resolved_files: list[FileRef] = field(default_factory=list)
    resolved_attachments: list[AttachmentRef] = field(default_factory=list)
    resolved_script: Path | None = None


@dataclass(frozen=True)
class Plan:
    """A parsed plan: metadata plus stages in listed order."""

    path: Path
    stages: list[Stage]
    name: str | None = None
    summary: str | None = None
    version: str = "1.0"
    max_workers: int = 1

    @property
    def base_dir(self) -> Path:
        return self.path.parent


@dataclass
class StageResult:
    """Outcome of one executed stage.

    ``text`` is the LLM response text (or, for script stages, the content of
    the first produced file). ``files`` are the paths a script stage produced.
    ``response_id`` is the llm log ULID for finding the run in ``llm logs``.
    """

    name: str
    success: bool
    duration: float = 0.0
    text: str = ""
    files: list[Path] | None = None
    manifest: dict | None = None
    error: str | None = None
    response_id: str | None = None
