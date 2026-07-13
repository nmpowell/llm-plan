"""Plan YAML loading: extends, variable substitution, and the prompt grammar.

Chaining is in-memory: ``chain:`` prompts read completed-stage text from a
dict, and model strings pass straight through to ``llm.get_model()`` (which
resolves aliases itself) rather than through a local alias table.
"""

from __future__ import annotations

import difflib
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

# Every key a stage mapping may contain; anything else is a typo.
KNOWN_STAGE_KEYS = frozenset(
    {
        "name",
        "summary",
        "type",
        "model",
        "prompt",
        "prompt_label",
        "options",
        "script",
        "script_args",
        "python",
        "timeout",
        "env",
        "depends_on",
        "produces",
        "exclusive",
        "continue_on_failure",
        "partial_dependencies",
        "files",
        "attachments",
    }
)

_VARIABLE_RE = re.compile(r"\$\{([^}]+)\}")


def load_with_extends(plan_file: Path) -> dict:
    """Load a plan YAML, recursively merging its ``extends:`` chain underneath.

    A child's values override its parent's, which override the grandparent's,
    and so on up the chain. A circular chain is a PlanError.
    """
    return _load_with_extends(Path(plan_file), ancestors=())


def _load_with_extends(plan_file: Path, ancestors: tuple[Path, ...]) -> dict:
    resolved = plan_file.resolve()
    if resolved in ancestors:
        chain = " -> ".join(str(path) for path in (*ancestors, resolved))
        raise PlanError(f"Circular 'extends' chain: {chain}")

    data = _load_yaml_mapping(plan_file)
    if not data:
        return {}

    if "extends" in data:
        base_path = plan_file.parent / data["extends"]
        if not base_path.exists():
            raise PlanError(
                f"Extended file not found: {base_path} (referenced from {plan_file})"
            )
        base = _load_with_extends(base_path, (*ancestors, resolved))
        del data["extends"]
        data = deep_merge(base, data)

    return data


def _load_yaml_mapping(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise PlanError(f"Invalid YAML in {path}: {exc}") from exc
    except UnicodeError as exc:
        raise PlanError(f"{path} is not valid UTF-8: {exc}") from exc
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


def _prompt_texts(prompt: str | list | None) -> list[str]:
    """The text of each prompt item, skipping malformed entries."""
    if not prompt:
        return []
    texts = []
    for item in prompt if isinstance(prompt, list) else [prompt]:
        text = item.get("prompt", "") if isinstance(item, dict) else item
        if isinstance(text, str):
            texts.append(text)
    return texts


def prompt_has_cli(prompt: str | list | None) -> bool:
    """True if any element of the prompt field references CLI input."""
    return any(
        text.upper() == "CLI" or text.upper().startswith("CLI:")
        for text in _prompt_texts(prompt)
    )


def prompt_wants_cli_files(prompt: str | list | None) -> bool:
    """True if any CLI token in the prompt asks for the CLI seed files."""
    return any(
        text.upper() in ("CLI", "CLI:FILES", "CLI:ALL")
        for text in _prompt_texts(prompt)
    )


def prompt_chain_targets(prompt: str | list | None) -> list[str]:
    """Stage names referenced by ``chain:`` items in a prompt field."""
    return [
        text[len("chain:"):]
        for text in _prompt_texts(prompt)
        if text.startswith("chain:")
    ]


def parse_prompt_list(
    prompt: str | list | None,
    stage_name: str,
    cli_instruction_parts: list[tuple[str | None, str]],
    completed_text: dict[str, str],
    base_dir: Path | None = None,
) -> PromptSpec:
    """Parse a stage's prompt spec (scalar or list) into a PromptSpec.

    ``completed_text`` maps finished-stage names to their response text, for
    ``chain:`` items. ``cli_instruction_parts`` are ordered (heading, text)
    pairs from the command line, included wherever CLI instructions are
    requested; a part's heading (or, failing that, the prompt item's label)
    becomes its section heading. Called at stage-execution time.
    """
    if not prompt:
        return PromptSpec()

    items = _prompt_items(prompt, stage_name)

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
            if include_instructions:
                for heading, text in cli_instruction_parts:
                    sections.append((heading if heading else label, text))
            if value == "files":
                wants_cli_files = True

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


def _prompt_items(
    prompt: str | list, stage_name: str
) -> list[tuple[str, str | None]]:
    """Normalise a prompt field (scalar or list) into (text, label) pairs."""
    items: list[tuple[str, str | None]] = []
    for entry in prompt if isinstance(prompt, list) else [prompt]:
        if isinstance(entry, str):
            items.append((entry, None))
        elif isinstance(entry, dict) and "prompt" in entry:
            if not isinstance(entry["prompt"], str):
                raise PlanError(
                    f"Stage '{stage_name}': prompt item {entry!r} must have a "
                    f"string 'prompt'"
                )
            label = entry.get("label")
            if label is not None and not isinstance(label, str):
                raise PlanError(
                    f"Stage '{stage_name}': prompt label {label!r} must be a string"
                )
            items.append((entry["prompt"], label))
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
    return items


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
    data = substitute_variables(data, variables, defer=RUNTIME_NAMESPACES)

    stages_data = data.get("stages") or []
    if not stages_data:
        raise PlanError(f"No stages defined in plan file: {plan_file}")

    stages = [
        _parse_stage(stage_data, number, plan_file)
        for number, stage_data in enumerate(stages_data, 1)
    ]

    parallel_config = data.get("parallel_config")
    if parallel_config is None:
        parallel_config = {}
    if not isinstance(parallel_config, dict):
        raise PlanError(
            f"parallel_config must be a mapping, not {parallel_config!r}"
        )
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

    unknown_keys = sorted(set(stage_data) - KNOWN_STAGE_KEYS)
    if unknown_keys:
        raise error(_unknown_keys_message(unknown_keys))

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
        # No model means llm's configured default, like upstream templates.
        if model is not None and (not isinstance(model, str) or not model):
            raise error(f"'model' must be a non-empty string, not {model!r}")
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

    depends_on = stage_data.get("depends_on") or []
    if not isinstance(depends_on, list):
        raise error(f"'depends_on' must be a list of stage names, not {depends_on!r}")
    non_strings = [dep for dep in depends_on if not isinstance(dep, str)]
    if non_strings:
        raise error(f"'depends_on' entries must be strings; got {non_strings!r}")
    duplicate_deps = sorted({dep for dep in depends_on if depends_on.count(dep) > 1})
    if duplicate_deps:
        raise error(f"'depends_on' has duplicate entries: {duplicate_deps}")

    for flag in ("exclusive", "continue_on_failure", "partial_dependencies"):
        flag_value = stage_data.get(flag, False)
        if not isinstance(flag_value, bool):
            raise error(f"'{flag}' must be a boolean (true/false), not {flag_value!r}")

    prompt = stage_data.get("prompt")
    if prompt:
        # Fail fast on a malformed entry or missing prompt file, before any
        # stage runs (and spends money).
        for text, _label in _prompt_items(prompt, name):
            parse_prompt_field(text, name, base_dir=plan_file.parent)

    files_data = stage_data.get("files", [])
    if not isinstance(files_data, list):
        raise error("'files' must be a list of {path, label} mappings")
    resolved_files = []
    for spec in files_data:
        if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
            raise error(
                f"invalid files entry {spec!r}: expected a mapping with a 'path' string"
            )
        path = _resolve_path(spec["path"], plan_file)
        if not path.exists():
            raise error(f"file not found: {path}")
        resolved_files.append(FileRef(path=path, label=spec.get("label")))

    attachments_data = stage_data.get("attachments", [])
    if not isinstance(attachments_data, list):
        raise error("'attachments' must be a list of {path, label} mappings")
    resolved_attachments = []
    for spec in attachments_data:
        if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
            raise error(
                f"invalid attachments entry {spec!r}: expected a mapping with a "
                f"'path' string"
            )
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
        depends_on=depends_on,
        produces=stage_data.get("produces"),
        exclusive=stage_data.get("exclusive", False),
        continue_on_failure=stage_data.get("continue_on_failure", False),
        partial_dependencies=stage_data.get("partial_dependencies", False),
        resolved_files=resolved_files,
        resolved_attachments=resolved_attachments,
        resolved_script=resolved_script,
    )


def _unknown_keys_message(unknown_keys: list[str]) -> str:
    parts = []
    for key in unknown_keys:
        close = difflib.get_close_matches(key, KNOWN_STAGE_KEYS, n=1)
        suggestion = f" (did you mean '{close[0]}'?)" if close else ""
        parts.append(f"unknown key '{key}'{suggestion}")
    return f"{'; '.join(parts)}. Valid keys: {', '.join(sorted(KNOWN_STAGE_KEYS))}"


def _resolve_path(value: str, plan_file: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = plan_file.parent / path
    return path.resolve()
