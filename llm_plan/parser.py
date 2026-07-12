"""Plan YAML loading: extends, variable substitution, and the prompt grammar.

Chaining is in-memory: ``chain:`` prompts read completed-stage text from a
dict, and model strings pass straight through to ``llm.get_model()`` (which
resolves aliases itself) rather than through a local alias table.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import yaml

from .models import AttachmentRef, FileRef, Plan, PlanError, PromptSpec, PromptType, Stage

# Top-level plan keys that are structure, not variable namespaces.
RESERVED_KEYS = frozenset({"extends", "name", "summary", "version", "stages", "parallel_config"})

# Namespaces resolved at stage-execution time, not load time.
RUNTIME_NAMESPACES = frozenset({"cli", "run"})

_VARIABLE_RE = re.compile(r"\$\{([^}]+)\}")


def load_with_extends(plan_file: Path) -> dict:
    """Load a plan YAML, merging any ``extends:`` base file underneath it."""
    data = _load_yaml_mapping(plan_file)
    if not data:
        return {}

    if "extends" in data:
        base_path = plan_file.parent / data["extends"]
        if not base_path.exists():
            raise PlanError(
                f"Extended file not found: {base_path} (referenced from {plan_file})"
            )
        base = _load_yaml_mapping(base_path)
        data = deep_merge(base, data)
        del data["extends"]

    return data


def _load_yaml_mapping(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise PlanError(f"Invalid YAML in {path}: {exc}") from exc
    except OSError as exc:
        raise PlanError(f"Could not read {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise PlanError(f"{path} must contain a YAML mapping, not {type(data).__name__}")
    return data


def deep_merge(base: dict, override: dict) -> dict:
    """Merge ``override`` over ``base``; nested dicts merge recursively."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def substitute_variables(
    data: Any, variables: dict, defer: frozenset[str] = frozenset()
) -> Any:
    """Recursively substitute ``${namespace.key}`` tokens in a structure.

    Tokens whose first path segment is in ``defer`` are left verbatim for a
    later pass (used for runtime ``${cli.*}``/``${run.*}`` values).
    """
    if isinstance(data, str):

        def replace(match: re.Match) -> str:
            path = match.group(1)
            parts = path.split(".")
            if parts[0] in defer:
                return match.group(0)
            value = variables
            for part in parts:
                if not isinstance(value, dict) or part not in value:
                    raise PlanError(
                        f"Variable '${{{path}}}' not found. "
                        f"Available top-level keys: {sorted(variables)}"
                    )
                value = value[part]
            return str(value)

        return _VARIABLE_RE.sub(replace, data)

    if isinstance(data, dict):
        return {k: substitute_variables(v, variables, defer) for k, v in data.items()}
    if isinstance(data, list):
        return [substitute_variables(item, variables, defer) for item in data]
    return data


def parse_prompt_field(
    prompt: str | None, stage_name: str = "", base_dir: Path | None = None
) -> tuple[PromptType, str]:
    """Classify a single prompt string into (type, value)."""
    if not prompt:
        return (PromptType.NONE, "")

    upper = prompt.upper()
    if upper == "CLI":
        return (PromptType.CLI, "")
    if upper.startswith("CLI:"):
        suffix = prompt[4:].lower()
        if suffix in ("instructions", "files", "all"):
            return (PromptType.CLI, suffix)
        raise PlanError(
            f"Stage '{stage_name}': invalid CLI prompt '{prompt}'. "
            f"Valid forms: CLI, CLI:instructions, CLI:files, CLI:all"
        )
    if prompt.startswith("inline:"):
        return (PromptType.INLINE, prompt[len("inline:"):])
    if prompt.startswith("chain:"):
        target = prompt[len("chain:"):]
        if not target:
            raise PlanError(
                f"Stage '{stage_name}': 'chain:' must name a stage (e.g. 'chain:analyst')"
            )
        return (PromptType.CHAIN, target)

    # File path: resolve against the plan's directory first (portable plans),
    # then the working directory. Strings longer than a filename component can
    # make Path.is_file() raise OSError (ENAMETOOLONG) - treat those as inline.
    raw = Path(prompt)
    candidates = []
    if base_dir is not None and not raw.is_absolute():
        candidates.append(Path(base_dir) / raw)
    candidates.append(raw)
    try:
        for candidate in candidates:
            if candidate.is_file():
                return (PromptType.FILE, str(candidate))
    except OSError:
        return (PromptType.INLINE, prompt)

    if "/" in prompt or prompt.endswith((".md", ".txt", ".yaml", ".yml")):
        raise PlanError(
            f"Stage '{stage_name}': prompt '{prompt}' looks like a file path but "
            f"doesn't exist. For literal text, use the 'inline:' prefix."
        )
    warnings.warn(
        f"Prompt '{prompt}' interpreted as inline text; use an 'inline:' prefix "
        f"to make that explicit.",
        UserWarning,
        stacklevel=2,
    )
    return (PromptType.INLINE, prompt)


def prompt_has_cli(prompt: str | list | None) -> bool:
    """True if any element of the prompt field references CLI input."""
    if not prompt:
        return False
    items = prompt if isinstance(prompt, list) else [prompt]
    for item in items:
        text = item.get("prompt", "") if isinstance(item, dict) else item
        if isinstance(text, str) and (
            text.upper() == "CLI" or text.upper().startswith("CLI:")
        ):
            return True
    return False


def parse_prompt_list(
    prompt: str | list | None,
    stage_name: str,
    cli_instructions: str,
    completed_text: dict[str, str],
    base_dir: Path | None = None,
) -> PromptSpec:
    """Parse a stage's prompt spec (scalar or list) into a PromptSpec.

    ``completed_text`` maps finished-stage names to their response text, for
    ``chain:`` items. Called at stage-execution time.
    """
    if not prompt:
        return PromptSpec()

    items: list[tuple[str, str | None]] = []
    for entry in prompt if isinstance(prompt, list) else [prompt]:
        if isinstance(entry, str):
            items.append((entry, None))
        elif isinstance(entry, dict) and "prompt" in entry:
            items.append((entry["prompt"], entry.get("label")))
        elif isinstance(entry, dict) and "inline" in entry:
            # YAML trap: `prompt: inline: text` parses as a mapping, not the
            # string "inline:text".
            raise PlanError(
                f"Stage '{stage_name}': prompt item {entry!r} is a mapping. "
                f'Quote inline prompts as a string: "inline:..."'
            )
        else:
            raise PlanError(
                f"Stage '{stage_name}': invalid prompt item {entry!r}; expected a "
                f"string or a {{prompt, label}} mapping"
            )

    spec_files: list[FileRef] = []
    sections: list[tuple[str | None, str]] = []
    wants_cli_files = False
    requires_cli_content = False

    for text, label in items:
        ptype, value = parse_prompt_field(text, stage_name, base_dir)

        if ptype == PromptType.CLI:
            include_instructions = value in ("", "instructions", "all")
            if value in ("", "all"):
                wants_cli_files = True
                requires_cli_content = True
            elif value == "files":
                wants_cli_files = True
            if include_instructions and cli_instructions:
                sections.append((label, cli_instructions))

        elif ptype == PromptType.FILE:
            spec_files.append(FileRef(path=Path(value), label=label))

        elif ptype == PromptType.INLINE:
            sections.append((label, value))

        elif ptype == PromptType.CHAIN:
            if value not in completed_text:
                raise PlanError(
                    f"Stage '{stage_name}' references chain:'{value}' but that stage "
                    f"has not completed. Add it to depends_on. "
                    f"Available: {sorted(completed_text)}"
                )
            sections.append((label, completed_text[value]))

    return PromptSpec(
        prompt_files=spec_files,
        sections=sections,
        wants_cli_files=wants_cli_files,
        requires_cli_content=requires_cli_content,
    )


def parse_plan(plan_file: Path) -> Plan:
    """Parse a plan file into a Plan, validating stages and their paths.

    DAG validation (dependencies, cycles) lives in :mod:`llm_plan.dag`.
    """
    plan_file = Path(plan_file)
    data = load_with_extends(plan_file)
    if not data:
        raise PlanError(f"Empty plan file: {plan_file}")

    variables = {
        key: value
        for key, value in data.items()
        if key not in RESERVED_KEYS and isinstance(value, dict)
    }
    if variables:
        data = substitute_variables(data, variables, defer=RUNTIME_NAMESPACES)

    stages_data = data.get("stages") or []
    if not stages_data:
        raise PlanError(f"No stages defined in plan file: {plan_file}")

    stages = [
        _parse_stage(stage_data, number, plan_file)
        for number, stage_data in enumerate(stages_data, 1)
    ]

    parallel_config = data.get("parallel_config") or {}
    max_workers = parallel_config.get("max_workers", 1)
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise PlanError(
            f"parallel_config.max_workers must be a positive integer, "
            f"not {max_workers!r}"
        )
    return Plan(
        path=plan_file,
        stages=stages,
        name=data.get("name"),
        summary=data.get("summary"),
        version=str(data.get("version", "1.0")),
        max_workers=max_workers,
    )


def _parse_stage(stage_data: dict, number: int, plan_file: Path) -> Stage:
    def error(message: str) -> PlanError:
        return PlanError(f"Stage {number}{name_suffix}: {message}")

    name_suffix = ""
    if not isinstance(stage_data, dict):
        raise error(f"must be a mapping, not {type(stage_data).__name__}")
    name = stage_data.get("name")
    if not name:
        raise error("'name' is required")
    name_suffix = f" ({name})"

    summary = stage_data.get("summary")
    if not summary:
        raise error("'summary' is required")

    stage_type = stage_data.get("type", "llm")
    if stage_type not in ("llm", "python_script"):
        raise error(f"invalid type '{stage_type}'; must be 'llm' or 'python_script'")

    model = stage_data.get("model")
    script = stage_data.get("script")
    resolved_script = None
    if stage_type == "llm":
        if not model:
            raise error("'model' is required for llm stages")
    else:
        if not script:
            raise error("'script' is required for python_script stages")
        resolved_script = _resolve_path(script, plan_file)
        if not resolved_script.exists():
            raise error(f"script not found: {resolved_script}")

    script_args = stage_data.get("script_args", [])
    if not isinstance(script_args, list):
        raise error("'script_args' must be a list")

    timeout = stage_data.get("timeout")
    if timeout is not None and (
        isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0
    ):
        raise error("'timeout' must be a positive integer (seconds)")

    env = stage_data.get("env") or {}
    if not isinstance(env, dict):
        raise error("'env' must be a mapping of name to value")

    options = stage_data.get("options") or {}
    if not isinstance(options, dict):
        raise error("'options' must be a mapping")

    prompt = stage_data.get("prompt")
    if isinstance(prompt, str):
        # Fail fast on a missing scalar file prompt, before any stage runs.
        parse_prompt_field(prompt, name, base_dir=plan_file.parent)

    resolved_files = []
    for spec in stage_data.get("files", []):
        path = _resolve_path(spec["path"], plan_file)
        if not path.exists():
            raise error(f"file not found: {path}")
        resolved_files.append(FileRef(path=path, label=spec.get("label")))

    resolved_attachments = []
    for spec in stage_data.get("attachments", []):
        uri = spec["path"]
        if uri.startswith(("http://", "https://")):
            resolved_attachments.append(AttachmentRef(uri=uri, label=spec.get("label")))
        else:
            path = _resolve_path(uri, plan_file)
            if not path.exists():
                raise error(f"attachment not found: {path}")
            resolved_attachments.append(
                AttachmentRef(uri=str(path), label=spec.get("label"))
            )

    return Stage(
        name=name,
        summary=summary,
        type=stage_type,
        model=model,
        prompt=prompt,
        prompt_label=stage_data.get("prompt_label"),
        options=options,
        script=script,
        script_args=script_args,
        python=stage_data.get("python", "python3"),
        timeout=timeout,
        env=env,
        depends_on=stage_data.get("depends_on") or [],
        produces=stage_data.get("produces"),
        exclusive=stage_data.get("exclusive", False),
        continue_on_failure=stage_data.get("continue_on_failure", False),
        partial_dependencies=stage_data.get("partial_dependencies", False),
        resolved_files=resolved_files,
        resolved_attachments=resolved_attachments,
        resolved_script=resolved_script,
    )


def _resolve_path(value: str, plan_file: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = plan_file.parent / path
    return path.resolve()
