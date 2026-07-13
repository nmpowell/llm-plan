"""The ``python_script`` stage protocol, and helpers for script authors.

A script stage's contract with the plan runner:

* **Inputs**: the CLI instructions on ``$LLM_PLAN_INSTRUCTIONS`` (plus
  ``$LLM_PLAN_OUTPUT_DIR`` / ``$LLM_PLAN_RUN_ID`` / ``$LLM_PLAN_STAGE_NAME``),
  ``--plan-arg`` values and the stage's ``script_args``, then dependency
  output files as trailing positional argv.
* **Output**: printed to stdout - either one existing file path per line, or a
  single JSON manifest ``{"outputs": [{"path": ..., "label": ...}, ...],
  "metadata": {...}}``. Everything else (progress, logs) goes to stderr;
  exit 0 on success.

``${cli.*}``/``${run.*}`` tokens in ``script_args``, ``env`` values and
``--plan-arg`` values are substituted at execution time (see
``RUNTIME_VAR_SPEC``); in LLM-stage prompts they can never be substituted,
so plan loading rejects them.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from .models import PlanError, Stage
from .parser import prompt_texts

ENV_INSTRUCTIONS = "LLM_PLAN_INSTRUCTIONS"
ENV_OUTPUT_DIR = "LLM_PLAN_OUTPUT_DIR"
ENV_RUN_ID = "LLM_PLAN_RUN_ID"
ENV_STAGE_NAME = "LLM_PLAN_STAGE_NAME"

# Runtime variables available in script_args, env values and --plan-arg values.
RUNTIME_VAR_SPEC: dict[str, frozenset[str]] = {
    "cli": frozenset({"instructions", "run_id"}),
    "run": frozenset({"output_dir", "stage_name"}),
}

_RUNTIME_VAR_RE = re.compile(r"\$\{([^}]+)\}")


RESERVED_ENV_NAMES = frozenset(
    {ENV_INSTRUCTIONS, ENV_OUTPUT_DIR, ENV_RUN_ID, ENV_STAGE_NAME}
)


def validate_runtime_vars(stages: list[Stage]) -> None:
    """Reject unknown ``${...}`` tokens in script_args and env at load time.

    Runs after the load-time variable pass, so the only tokens left are the
    deferred runtime ones; this catches typos before any stage spends money.
    Also rejects ``env:`` entries that would override the runner-owned
    ``LLM_PLAN_*`` variables - overriding LLM_PLAN_OUTPUT_DIR would let two
    parallel stages share an output directory again.
    """
    for stage in stages:
        reserved = sorted(RESERVED_ENV_NAMES & set(stage.env))
        if reserved:
            raise PlanError(
                f"Stage '{stage.name}': env must not set the runner-owned "
                f"variable(s) {', '.join(reserved)}"
            )
        for value in list(stage.script_args) + list(stage.env.values()):
            if not isinstance(value, str):
                continue
            for token in _RUNTIME_VAR_RE.findall(value):
                parts = token.split(".")
                allowed = RUNTIME_VAR_SPEC.get(parts[0])
                if allowed is None or len(parts) != 2 or parts[1] not in allowed:
                    valid = ", ".join(
                        f"{ns}.{key}"
                        for ns, keys in RUNTIME_VAR_SPEC.items()
                        for key in sorted(keys)
                    )
                    raise PlanError(
                        f"Stage '{stage.name}': unknown runtime variable "
                        f"'${{{token}}}'. Valid runtime variables: {valid}"
                    )
        if stage.type == "python_script":
            continue
        for text in prompt_texts(stage.prompt):
            for token in _RUNTIME_VAR_RE.findall(text):
                if token.split(".")[0] in RUNTIME_VAR_SPEC:
                    raise PlanError(
                        f"Stage '{stage.name}': runtime variable '${{{token}}}' "
                        f"in a prompt would be sent to the model verbatim; "
                        f"runtime variables are only substituted in script_args, "
                        f"env values and --plan-arg values"
                    )


def substitute_runtime(value: str, runtime: dict) -> str:
    """Substitute runtime ``${...}`` tokens; unknown tokens are an error."""

    def replace(match: re.Match) -> str:
        token = match.group(1)
        cursor: Any = runtime
        for part in token.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                raise PlanError(f"Unknown runtime variable '${{{token}}}'")
            cursor = cursor[part]
        return str(cursor)

    return _RUNTIME_VAR_RE.sub(replace, value)


def parse_manifest(stdout: str) -> dict | None:
    """Parse a JSON manifest from script stdout; None for the bare-paths form.

    A JSON object with an ``outputs`` key is treated as a manifest and must be
    well-formed: ``outputs`` is a list of path strings or {path, ...} mappings.
    """
    text = stdout.strip()
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict) or "outputs" not in data:
        return None

    outputs = data["outputs"]
    if not isinstance(outputs, list):
        raise PlanError(
            f"Invalid script manifest: 'outputs' must be a list, "
            f"not {type(outputs).__name__}"
        )
    for spec in outputs:
        if isinstance(spec, str):
            continue
        if isinstance(spec, dict) and isinstance(spec.get("path"), str):
            continue
        raise PlanError(
            f"Invalid script manifest output entry {spec!r}: expected a path "
            f"string or a mapping with a 'path' string"
        )
    return data


def manifest_output_paths(manifest: dict) -> list[Path]:
    """Output file paths from a parsed manifest, in order."""
    paths = []
    for spec in manifest.get("outputs", []):
        if isinstance(spec, dict) and spec.get("path"):
            paths.append(Path(spec["path"]))
        elif isinstance(spec, str):
            paths.append(Path(spec))
    return paths


def manifest_label(manifest: dict | None, path: Path) -> str | None:
    """The label a manifest assigns to ``path``, if any."""
    if not manifest:
        return None
    for spec in manifest.get("outputs", []):
        if isinstance(spec, dict) and spec.get("label"):
            if Path(spec.get("path", "")) == Path(path):
                return str(spec["label"])
    return None


# --- Helpers for script authors ---------------------------------------------


def read_instructions() -> str:
    """The CLI instructions forwarded by the plan runner, or ''."""
    return os.environ.get(ENV_INSTRUCTIONS, "")


def output_dir() -> Path | None:
    """The stage's own scratch directory, if the plan runner forwarded one."""
    value = os.environ.get(ENV_OUTPUT_DIR)
    return Path(value) if value else None


def run_id() -> str | None:
    """The run id the plan runner forwarded, if any."""
    return os.environ.get(ENV_RUN_ID)


def emit_outputs(
    outputs: list[Any], *, metadata: dict | None = None, manifest: bool = True, stream=None
) -> None:
    """Print produced output paths to stdout for the plan runner.

    Each item is a path or a mapping with ``path`` and optional ``label``.
    With ``manifest=True`` a single JSON manifest is printed; otherwise the
    bare one-path-per-line form (labels and metadata are dropped).
    """
    out = stream or sys.stdout
    specs = []
    for item in outputs:
        if isinstance(item, dict):
            spec = {"path": str(item["path"])}
            if item.get("label"):
                spec["label"] = str(item["label"])
            if item.get("kind"):
                spec["kind"] = str(item["kind"])
            specs.append(spec)
        else:
            specs.append({"path": str(item)})
    if manifest:
        document: dict[str, Any] = {"outputs": specs}
        if metadata:
            document["metadata"] = metadata
        print(json.dumps(document), file=out)
    else:
        for spec in specs:
            print(spec["path"], file=out)
