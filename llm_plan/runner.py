"""Plan execution: prompt composition, model calls, chaining and scheduling.

Stage outputs are held in memory and threaded into dependent stages; nothing
is written to disk for LLM stages. Responses are handed to the ``on_response``
callback (the CLI uses it to log to llm's logs.db) on the coordinating thread.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path

import llm
import pydantic

from . import script_stage
from .dag import SEED, leaf_stages, resolve_dependencies, validate_plan
from .models import Plan, PlanError, Stage, StageResult
from .parser import parse_plan, parse_prompt_list, prompt_has_cli

DEFAULT_RETRIES = 2
DEFAULT_RETRY_DELAY = 5.0
SECTION_SEPARATOR = "\n\n---\n\n"


def load_plan(path) -> Plan:
    """Parse and fully validate a plan file."""
    plan = parse_plan(path)
    validate_plan(plan)
    script_stage.validate_runtime_vars(plan.stages)
    return plan


@dataclass
class CLIContext:
    """Seed input from the command line, routed to stages that ask for it."""

    instructions: str = ""
    fragments: list = field(default_factory=list)
    attachments: list = field(default_factory=list)
    model: str | None = None
    options: dict = field(default_factory=dict)
    plan_args: list = field(default_factory=list)

    @property
    def has_content(self) -> bool:
        return bool(self.instructions or self.fragments or self.attachments)


class PlanRunner:
    """Executes a plan's stages in dependency order."""

    def __init__(
        self,
        plan: Plan,
        cli: CLIContext | None = None,
        *,
        run_id: str | None = None,
        on_response=None,
        progress=None,
        retries: int = DEFAULT_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
    ):
        self.plan = plan
        self.cli = cli or CLIContext()
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.on_response = on_response
        self.progress = progress or (lambda message: None)
        self.retries = retries
        self.retry_delay = retry_delay
        self.results: dict[str, StageResult] = {}
        self._text: dict[str, str] = {}
        self._files: dict[str, list[Path]] = {}
        self._manifests: dict[str, dict] = {}

    def run(self) -> list[StageResult]:
        """Execute every stage; results in listed-stage order."""
        return self._run_sequential()

    def leaf_stages(self) -> list[str]:
        return leaf_stages(self.plan.stages)

    # -- scheduling -----------------------------------------------------------

    def _run_sequential(self) -> list[StageResult]:
        failed: set[str] = set()
        total = len(self.plan.stages)
        for index, stage in enumerate(self.plan.stages):
            deps = resolve_dependencies(stage, index, self.plan.stages)
            if self._skip_for_failed_deps(stage, deps, failed):
                continue
            self.progress(f"Stage {index + 1}/{total}: {stage.name} - {stage.summary}")
            result, response = self._run_stage(stage, index, deps, failed)
            self._finish(stage, result, response, failed)
        return [self.results[stage.name] for stage in self.plan.stages]

    def _skip_for_failed_deps(self, stage: Stage, deps: list[str], failed: set[str]) -> bool:
        failed_deps = [d for d in deps if d in failed]
        if not failed_deps or stage.partial_dependencies or stage.continue_on_failure:
            if failed_deps:
                self.progress(
                    f"  running {stage.name} with partial dependencies "
                    f"(missing: {', '.join(failed_deps)})"
                )
            return False
        result = StageResult(
            name=stage.name,
            success=False,
            error=f"skipped: dependencies failed: {failed_deps}",
        )
        self.progress(f"  skipping {stage.name}: dependencies failed: {failed_deps}")
        self.results[stage.name] = result
        failed.add(stage.name)
        return True

    def _finish(self, stage: Stage, result: StageResult, response, failed: set[str]) -> None:
        self.results[stage.name] = result
        if result.success:
            self._text[stage.name] = result.text
            if result.files:
                self._files[stage.name] = result.files
            if result.manifest:
                self._manifests[stage.name] = result.manifest
            self.progress(f"  ✓ {stage.name} ({result.duration:.1f}s)")
            if response is not None and self.on_response is not None:
                try:
                    self.on_response(stage, response)
                except Exception as exc:
                    self.progress(f"  warning: could not record response: {exc}")
        else:
            failed.add(stage.name)
            self.progress(f"  ✗ {stage.name}: {result.error}")

    # -- stage execution ------------------------------------------------------

    def _run_stage(self, stage: Stage, index: int, deps: list[str], failed: set[str]):
        """Execute one stage; returns (StageResult, llm response or None)."""
        if stage.type == "python_script":
            return self._run_script_stage(stage, deps, failed), None
        return self._run_llm_stage(stage, index, deps, failed)

    def _run_llm_stage(self, stage: Stage, index: int, deps: list[str], failed: set[str]):
        start = time.monotonic()
        try:
            prompt_text, fragments, attachments, model, options = self._prepare_llm(
                stage, index, deps, failed
            )
        except PlanError as exc:
            return StageResult(
                name=stage.name,
                success=False,
                duration=time.monotonic() - start,
                error=str(exc),
            ), None

        last_error = None
        for attempt in range(self.retries + 1):
            if attempt:
                self.progress(
                    f"  retrying {stage.name} (attempt {attempt + 1}/{self.retries + 1}) "
                    f"after: {last_error}"
                )
                time.sleep(self.retry_delay)
            try:
                # Options go as **kwargs: llm 0.31 has no options= parameter,
                # and 0.32 accepts both forms.
                response = model.prompt(
                    prompt_text or None,
                    fragments=fragments or None,
                    attachments=attachments or None,
                    stream=False,
                    **options,
                )
                text = response.text()
            except Exception as exc:
                last_error = str(exc) or type(exc).__name__
                continue
            return StageResult(
                name=stage.name,
                success=True,
                duration=time.monotonic() - start,
                text=text,
                response_id=getattr(response, "id", None),
            ), response

        return StageResult(
            name=stage.name,
            success=False,
            duration=time.monotonic() - start,
            error=last_error,
        ), None

    def _run_script_stage(self, stage: Stage, deps: list[str], failed: set[str]) -> StageResult:
        return StageResult(
            name=stage.name,
            success=False,
            error="python_script stages are not implemented yet",
        )

    # -- prompt preparation ---------------------------------------------------

    def _prepare_llm(self, stage: Stage, index: int, deps: list[str], failed: set[str]):
        spec = parse_prompt_list(
            stage.prompt, stage.name, self.cli.instructions, self._text, self.plan.base_dir
        )

        wants_cli_files = self._wants_cli_files(stage, index, deps, spec.wants_cli_files)
        if spec.requires_cli_content and not self.cli.has_content:
            raise PlanError(
                f"Stage '{stage.name}' uses prompt \"CLI\" but no input was provided. "
                f"Pass a prompt argument or pipe stdin, or use -f/--fragment or "
                f"-a/--attachment."
            )

        sections: list[tuple[str | None, str]] = []
        for ref in stage.resolved_files:
            sections.append((ref.label or ref.path.name, _read(ref.path, stage.name)))

        by_name = {s.name: s for s in self.plan.stages}
        for dep in deps:
            if dep == SEED or dep in failed:
                continue
            dep_stage = by_name.get(dep)
            label = dep_stage.produces if dep_stage and dep_stage.produces else f"Output from {dep}"
            if dep in self._files:
                files = self._files[dep]
                for position, path in enumerate(files, 1):
                    fallback = f"{label} ({position}/{len(files)})" if len(files) > 1 else label
                    file_label = script_stage.manifest_label(
                        self._manifests.get(dep), path
                    ) or fallback
                    sections.append((file_label, _read(path, stage.name)))
            elif dep in self._text:
                sections.append((label, self._text[dep]))

        prompt_files = list(spec.prompt_files)
        if stage.prompt_label and prompt_files:
            prompt_files[0] = replace(prompt_files[0], label=stage.prompt_label)
        for ref in prompt_files:
            sections.append((ref.label or "MAIN INSTRUCTIONS", _read(ref.path, stage.name)))

        sections.extend(spec.sections)

        prompt_text = SECTION_SEPARATOR.join(
            f"## {label}\n\n{text.strip()}" if label else text.strip()
            for label, text in sections
            if text and text.strip()
        )

        fragments = list(self.cli.fragments) if wants_cli_files else []
        attachments = [
            llm.Attachment(url=ref.uri) if ref.is_url else llm.Attachment(path=ref.uri)
            for ref in stage.resolved_attachments
        ]
        if wants_cli_files:
            attachments.extend(self.cli.attachments)

        if not prompt_text and not fragments and not attachments:
            raise PlanError(
                f"Stage '{stage.name}' composed an empty prompt. Check its prompt "
                f"spec, dependencies, and the CLI input it expects."
            )

        model_id = self.cli.model or stage.model
        try:
            model = llm.get_model(model_id)
        except llm.UnknownModelError as exc:
            raise PlanError(f"Stage '{stage.name}': {exc}") from exc

        options = self._merged_options(stage, model)
        return prompt_text, fragments, attachments, model, options

    def _wants_cli_files(
        self, stage: Stage, index: int, deps: list[str], prompt_wants_files: bool
    ) -> bool:
        if prompt_wants_files:
            return True
        if SEED in deps:
            return True
        opts_out = prompt_has_cli(stage.prompt) and not prompt_wants_files
        return index == 0 and not deps and not opts_out

    def _merged_options(self, stage: Stage, model) -> dict:
        stored = {}
        try:
            from llm.cli import get_model_options

            stored = get_model_options(model.model_id) or {}
        except ImportError:
            pass
        merged = {**stored, **stage.options, **self.cli.options}
        if merged:
            try:
                model.Options(**merged)
            except pydantic.ValidationError as exc:
                raise PlanError(
                    f"Stage '{stage.name}': invalid options for model "
                    f"'{model.model_id}': {exc}"
                ) from exc
        return merged


def _read(path: Path, stage_name: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PlanError(f"Stage '{stage_name}': could not read {path}: {exc}") from exc
