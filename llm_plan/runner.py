"""Plan execution: prompt composition, model calls, chaining and scheduling.

Stage outputs are held in memory and threaded into dependent stages; nothing
is written to disk for LLM stages. Responses are handed to the ``on_response``
callback (the CLI uses it to log to llm's logs.db) on the coordinating thread.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path

import llm
import pydantic

from . import script_stage
from .dag import SEED, leaf_stages, resolve_dependencies, topological_order, validate_plan
from .models import Plan, PlanError, Stage, StageResult
from .parser import (
    parse_plan,
    parse_prompt_list,
    prompt_chain_targets,
    prompt_has_cli,
    prompt_wants_cli_files,
)

DEFAULT_RETRIES = 2
DEFAULT_RETRY_DELAY = 5.0
DEFAULT_SCRIPT_TIMEOUT = 3600

# Deterministic model-call failures: a missing key or a validation error
# (llm raises ValueError/NotImplementedError/TypeError for unsupported
# attachments, tools and call shapes) never fixes itself between attempts.
_NON_RETRYABLE = (llm.NeedsKeyException, ValueError, NotImplementedError, TypeError)


def load_plan(path) -> Plan:
    """Parse and fully validate a plan file."""
    plan = parse_plan(path)
    validate_plan(plan)
    script_stage.validate_runtime_vars(plan.stages)
    return plan


def explain(plan: Plan, plan_args: list[str] | None = None) -> str:
    """A human-readable preview of the plan's DAG; executes nothing."""
    plan_args = list(plan_args or [])
    by_name = {stage.name: stage for stage in plan.stages}
    position = {stage.name: index for index, stage in enumerate(plan.stages)}
    lines = [f"Plan: {plan.name or 'Unnamed'}"]
    if plan.summary:
        lines.append(f"  {plan.summary}")
    lines.append(f"  Stages: {len(plan.stages)} (execution order)")
    lines.append("")
    for number, name in enumerate(topological_order(plan.stages), 1):
        stage = by_name[name]
        deps = resolve_dependencies(stage, position[name], plan.stages)
        lines.append(f"{number}. {name}  [{stage.type}]")
        lines.append(f"     {stage.summary}")
        if stage.type == "python_script":
            lines.append(f"     script: {stage.script}")
        else:
            lines.append(f"     model: {stage.model or '(llm default)'}")
        if deps:
            labelled = []
            for dep in deps:
                produces = getattr(by_name.get(dep), "produces", None)
                labelled.append(f"{dep} ({produces})" if produces else dep)
            lines.append(f"     inputs: {', '.join(labelled)}")
        else:
            lines.append("     inputs: CLI (prompt / -f / -a)")
        if stage.produces:
            lines.append(f"     produces: {stage.produces}")
        if stage.type == "python_script":
            command = [stage.python, str(stage.resolved_script or stage.script)]
            command += plan_args + [str(a) for a in stage.script_args]
            preview = shlex.join(command)
            if deps:
                preview += f"  <+ {len(deps)} dependency output file(s)>"
            lines.append(f"     command: {preview}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


@dataclass
class CLIContext:
    """Seed input from the command line, routed to stages that ask for it.

    ``files`` holds labelled context files (--cf); ``fragments`` holds llm
    fragments (-f). Both route to the same stages; files keep their labels.
    """

    instructions: str = ""
    instruction_parts: list = field(default_factory=list)  # ordered [(heading|None, text)]
    fragments: list = field(default_factory=list)
    files: list = field(default_factory=list)  # list[FileRef]
    attachments: list = field(default_factory=list)
    model: str | None = None
    options: dict = field(default_factory=dict)
    plan_args: list = field(default_factory=list)

    @property
    def all_instruction_parts(self) -> list[tuple[str | None, str]]:
        """-i/--ci parts in command-line order, then the prompt argument/stdin."""
        parts = list(self.instruction_parts)
        if self.instructions:
            parts.append((None, self.instructions))
        return parts

    @property
    def instructions_text(self) -> str:
        """The full instruction text, headings included - what scripts see."""
        return "\n\n".join(
            f"## {heading}\n\n{text}" if heading else text
            for heading, text in self.all_instruction_parts
        )

    @property
    def has_content(self) -> bool:
        return bool(
            self.all_instruction_parts or self.fragments or self.files or self.attachments
        )


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
        if retries < 0:
            raise PlanError(f"retries must be zero or more, not {retries}")
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
        self._scratch_dir: Path | None = None
        self._scratch_lock = threading.Lock()

    def run(self) -> list[StageResult]:
        """Execute every stage; results in listed-stage order."""
        self._warn_unrouted_cli_input()
        if self.plan.max_workers > 1 and len(self.plan.stages) > 1:
            self._run_parallel()
        else:
            self._run_sequential()
        return [self.results[stage.name] for stage in self.plan.stages]

    def leaf_stages(self) -> list[str]:
        return leaf_stages(self.plan.stages)

    def _warn_unrouted_cli_input(self) -> None:
        """Warn when -f/-a/--cf input was given but no stage will receive it."""
        if not (self.cli.files or self.cli.fragments or self.cli.attachments):
            return
        stages = self.plan.stages
        for index, stage in enumerate(stages):
            deps = resolve_dependencies(stage, index, stages)
            wants = self._wants_cli_files(
                stage, index, deps, prompt_wants_files=prompt_wants_cli_files(stage.prompt)
            )
            if wants:
                return
        self.progress(
            "  warning: -f/-a/--cf input provided but no stage requests CLI input "
            "(use prompt CLI, CLI:files or CLI:all, or depends_on: [seed])"
        )

    # -- scheduling -----------------------------------------------------------

    def _run_sequential(self) -> None:
        failed: set[str] = set()
        stages = self.plan.stages
        total = len(stages)
        for index, stage in enumerate(stages):
            deps = resolve_dependencies(stage, index, stages)
            if self._skip_for_failed_deps(stage, deps, failed):
                continue
            self.progress(f"Stage {index + 1}/{total}: {stage.name} - {stage.summary}")
            result, response = self._run_stage(stage, index, deps, failed)
            self._finish(stage, result, response, failed)
            if not result.success and not self._next_stage_tolerates_failure(index):
                # Abort rather than spend on later stages.
                raise PlanError(f"Stage '{stage.name}' failed: {result.error}")

    def _next_stage_tolerates_failure(self, index: int) -> bool:
        following = self.plan.stages[index + 1 : index + 2]
        return bool(following) and (
            following[0].partial_dependencies or following[0].continue_on_failure
        )

    def _run_parallel(self) -> None:
        """DAG scheduling over a thread pool.

        Workers only execute stages; results are collected, chained and
        logged on this coordinating thread. An exclusive stage waits for
        in-flight stages to drain and runs entirely alone.
        """
        from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

        llm.get_models()  # force llm's lazy plugin registry before threads race

        stages = self.plan.stages
        position = {stage.name: index for index, stage in enumerate(stages)}
        deps_of = {
            stage.name: resolve_dependencies(stage, index, stages)
            for index, stage in enumerate(stages)
        }
        dependents: dict[str, list[Stage]] = {stage.name: [] for stage in stages}
        in_degree: dict[str, int] = {}
        for stage in stages:
            real_deps = [d for d in deps_of[stage.name] if d != SEED]
            in_degree[stage.name] = len(real_deps)
            for dep in real_deps:
                dependents[dep].append(stage)

        failed: set[str] = set()
        ready = [stage for stage in stages if in_degree[stage.name] == 0]
        running: dict[Future, Stage] = {}
        total = len(stages)

        def release(stage: Stage) -> None:
            for dependent in dependents[stage.name]:
                in_degree[dependent.name] -= 1
                if in_degree[dependent.name] == 0:
                    ready.append(dependent)
            ready.sort(key=lambda s: position[s.name])

        executor = ThreadPoolExecutor(max_workers=self.plan.max_workers)
        interrupted = False
        try:
            while ready or running:
                while ready:
                    if running and any(s.exclusive for s in running.values()):
                        break
                    if ready[0].exclusive and running:
                        break  # drain in-flight stages before an exclusive one
                    stage = ready.pop(0)
                    deps = deps_of[stage.name]
                    if self._skip_for_failed_deps(stage, deps, failed):
                        release(stage)
                        continue
                    self.progress(
                        f"Stage {position[stage.name] + 1}/{total}: {stage.name} "
                        f"- {stage.summary}"
                    )
                    future = executor.submit(
                        self._run_stage, stage, position[stage.name], deps, failed
                    )
                    running[future] = stage
                    if stage.exclusive:
                        break

                if not running:
                    continue
                done, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for future in done:
                    stage = running.pop(future)
                    result, response = future.result()
                    self._finish(stage, result, response, failed)
                    release(stage)
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            # Ctrl-C must not block on in-flight stages: abandon them and
            # cancel anything queued. Every other exit drains the pool.
            executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

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
        """Execute one stage; returns (StageResult, llm response or None).

        Runs on worker threads in parallel mode, so it never raises -
        anything unexpected becomes a failed StageResult - except when
        ``LLM_RAISE_ERRORS`` is set (llm's debugging convention), which
        re-raises the original exception for the scheduler to surface.
        """
        try:
            if stage.type == "python_script":
                return self._run_script_stage(stage, index, deps, failed), None
            return self._run_llm_stage(stage, index, deps, failed)
        except Exception as exc:
            if os.environ.get("LLM_RAISE_ERRORS"):
                raise
            return StageResult(
                name=stage.name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            ), None

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
                # and 0.32 accepts both forms. Stream when the model can -
                # Anthropic's SDK rejects non-streaming requests it deems
                # long-running - but never force it on can_stream=False models.
                response = model.prompt(
                    prompt_text or None,
                    fragments=fragments or None,
                    attachments=attachments or None,
                    stream=bool(getattr(model, "can_stream", True)),
                    **options,
                )
                text = response.text()
            except Exception as exc:
                if os.environ.get("LLM_RAISE_ERRORS"):
                    raise  # llm's debugging convention: surface the original
                last_error = str(exc) or type(exc).__name__
                if isinstance(exc, _NON_RETRYABLE):
                    break  # deterministic failures never fix themselves mid-run
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

    def _run_script_stage(
        self, stage: Stage, index: int, deps: list[str], failed: set[str]
    ) -> StageResult:
        start = time.monotonic()

        def failure(error: str) -> StageResult:
            return StageResult(
                name=stage.name,
                success=False,
                duration=time.monotonic() - start,
                error=error,
            )

        scratch = self._stage_scratch(stage)
        runtime = {
            "cli": {"instructions": self.cli.instructions_text, "run_id": self.run_id},
            "run": {"output_dir": str(scratch), "stage_name": stage.name},
        }
        try:
            plan_args = [
                script_stage.substitute_runtime(str(a), runtime) for a in self.cli.plan_args
            ]
            args = [
                script_stage.substitute_runtime(str(a), runtime) for a in stage.script_args
            ]
            stage_env = {
                str(key): script_stage.substitute_runtime(str(value), runtime)
                for key, value in stage.env.items()
            }
        except PlanError as exc:
            return failure(str(exc))

        input_files = self._script_input_files(stage, index, deps, failed, scratch)
        command = [stage.python, str(stage.resolved_script)]
        command += plan_args + args + [str(path) for path in input_files]

        env = {
            **os.environ,
            script_stage.ENV_INSTRUCTIONS: self.cli.instructions_text,
            script_stage.ENV_OUTPUT_DIR: str(scratch),
            script_stage.ENV_RUN_ID: self.run_id,
            script_stage.ENV_STAGE_NAME: stage.name,
            **stage_env,
        }
        timeout = stage.timeout or DEFAULT_SCRIPT_TIMEOUT

        try:
            # Its own session so a timeout can kill the whole process group:
            # grandchildren would otherwise outlive the stage and hold the
            # inherited pipes open. Stdout is a path/manifest protocol, so
            # decode it as UTF-8 like the file reads below, not the locale.
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            return failure(f"could not run script: {exc}")

        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_group(process)
            _, stderr = process.communicate()
            error = f"script timeout: exceeded {timeout}s"
            if stderr.strip():
                error += f"\n{stderr.strip()[-2000:]}"
            return failure(error)

        if process.returncode != 0:
            error = f"script exited with code {process.returncode}"
            if stderr:
                error += f"\n{stderr.strip()[-2000:]}"
            return failure(error)

        try:
            manifest = script_stage.parse_manifest(stdout)
        except PlanError as exc:
            return failure(str(exc))
        if manifest is not None:
            files = script_stage.manifest_output_paths(manifest)
        else:
            files = [
                Path(line.strip())
                for line in stdout.splitlines()
                if line.strip()
            ]
        if not files:
            return failure(
                "script produced no output paths on stdout "
                "(print one path per line, or a JSON manifest)"
            )
        missing = [path for path in files if not path.is_file()]
        if missing:
            return failure(f"script output file does not exist: {missing[0]}")

        try:
            text = files[0].read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return failure(f"could not read script output {files[0]}: {exc}")

        return StageResult(
            name=stage.name,
            success=True,
            duration=time.monotonic() - start,
            text=text,
            files=files,
            manifest=manifest,
        )

    def _script_input_files(
        self, stage: Stage, index: int, deps: list[str], failed: set[str], scratch: Path
    ) -> list[Path]:
        """Input paths for a script, in a stable documented order.

        Dependency outputs first (LLM text materialised to disk), then the
        stage's own ``files:``, then routed CLI seed files - a file-backed
        fragment contributes its original path, any other fragment is written
        into the stage's scratch directory.
        """
        paths: list[Path] = []
        for dep in deps:
            if dep == SEED or dep in failed:
                continue
            if dep in self._files:
                paths.extend(self._files[dep])
            elif dep in self._text:
                target = scratch / f"{dep}.md"
                target.write_text(self._text[dep], encoding="utf-8")
                paths.append(target)

        paths.extend(ref.path for ref in stage.resolved_files)

        wants = self._wants_cli_files(
            stage, index, deps, prompt_wants_files=prompt_wants_cli_files(stage.prompt)
        )
        if wants:
            paths.extend(ref.path for ref in self.cli.files)
            for position, fragment in enumerate(self.cli.fragments, 1):
                source = getattr(fragment, "source", None)
                if source and Path(source).is_file():
                    paths.append(Path(source))
                else:
                    target = scratch / f"fragment-{position}.md"
                    target.write_text(str(fragment), encoding="utf-8")
                    paths.append(target)
        return paths

    def _scratch(self) -> Path:
        """The run's scratch root (kept for inspection); one subdirectory per stage."""
        with self._scratch_lock:
            if self._scratch_dir is None:
                self._scratch_dir = Path(
                    tempfile.mkdtemp(prefix=f"llm-plan-{self.run_id}-")
                )
                self.progress(f"  scratch directory: {self._scratch_dir}")
            return self._scratch_dir

    def _stage_scratch(self, stage: Stage) -> Path:
        """A stage's own scratch directory, for its inputs and outputs.

        Stages sharing one directory would collide: parallel stages that
        consume the same dependency truncate-and-rewrite the same
        materialised input, and scripts writing the same output filename
        overwrite each other.
        """
        path = self._scratch() / stage.name
        path.mkdir(exist_ok=True)
        return path

    # -- prompt preparation ---------------------------------------------------

    def _prepare_llm(self, stage: Stage, index: int, deps: list[str], failed: set[str]):
        spec = parse_prompt_list(
            stage.prompt,
            stage.name,
            self.cli.all_instruction_parts,
            self._text,
            self.plan.base_dir,
        )

        wants_cli_files = self._wants_cli_files(stage, index, deps, prompt_wants_files=spec.wants_cli_files)
        if spec.requires_cli_content and not self.cli.has_content:
            raise PlanError(
                f"Stage '{stage.name}' uses prompt \"CLI\" but no input was provided. "
                f"Pass a prompt argument or pipe stdin, or use -f/--fragment or "
                f"-a/--attachment."
            )

        # Composition format: file-backed sections are code-fenced under
        # "## <label>" headings with "---" separators; plain instruction text
        # lands under "Additional Instructions" when other content exists,
        # and stays raw when it is the whole prompt.
        fenced_sections: list[tuple[str, str]] = []
        for ref in stage.resolved_files:
            fenced_sections.append((ref.label or ref.path.name, _read(ref.path, stage.name)))
        if wants_cli_files:
            for ref in self.cli.files:
                fenced_sections.append(
                    (ref.label or ref.path.name, _read(ref.path, stage.name))
                )

        by_name = {s.name: s for s in self.plan.stages}
        # A dep consumed by a chain: item is placed by that item; an
        # auto-fenced copy here would send the same content twice.
        chain_consumed = set(prompt_chain_targets(stage.prompt))
        for dep in deps:
            if dep == SEED or dep in failed or dep in chain_consumed:
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
                    fenced_sections.append((file_label, _read(path, stage.name)))
            elif dep in self._text:
                fenced_sections.append((label, self._text[dep]))

        prompt_files = list(spec.prompt_files)
        if stage.prompt_label and prompt_files:
            prompt_files[0] = replace(prompt_files[0], label=stage.prompt_label)
        for ref in prompt_files:
            fenced_sections.append(
                (ref.label or "MAIN INSTRUCTIONS", _read(ref.path, stage.name))
            )

        rendered = [
            _render_section(label, body, fenced=True)
            for label, body in fenced_sections
            if body.strip()
        ]
        spec_parts = [(label, text) for label, text in spec.sections if text.strip()]

        if not rendered and not any(label for label, _ in spec_parts):
            # The whole prompt is plain instruction text: keep it raw. A
            # single part passes through untouched (piped code keeps its
            # indentation, like llm prompt); only joins are tidied.
            if len(spec_parts) == 1:
                prompt_text = spec_parts[0][1]
            else:
                prompt_text = "\n\n".join(text.strip() for _, text in spec_parts)
        else:
            # Each part keeps its listed position; label-less parts sit under
            # their own "Additional Instructions" heading.
            rendered += [
                _render_section(label or "Additional Instructions", text, fenced=False)
                for label, text in spec_parts
            ]
            prompt_text = "\n".join(rendered)

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
        self, stage: Stage, index: int, deps: list[str], *, prompt_wants_files: bool
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
                    f"'{model.model_id}': {_render_validation_error(exc)}"
                ) from exc
        return merged


def _render_validation_error(exc: pydantic.ValidationError) -> str:
    """Format option errors the way llm's own CLI renders them."""
    try:
        from llm.cli import render_errors
    except ImportError:
        return str(exc)
    return render_errors(exc.errors())


def _kill_process_group(process: subprocess.Popen) -> None:
    """Kill a timed-out script and everything it spawned.

    The script runs as its own session leader, so its pid is the group id.
    """
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:  # every group member already exited
        pass


def _render_section(label: str, body: str, *, fenced: bool) -> str:
    if fenced:
        return f"---\n\n## {label}\n\n```\n{body.strip()}\n```\n"
    return f"---\n\n## {label}\n\n{body.strip()}\n"


def _read(path: Path, stage_name: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise PlanError(f"Stage '{stage_name}': could not read {path}: {exc}") from exc
