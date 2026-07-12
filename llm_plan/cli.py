"""Click commands for llm-plan."""

from __future__ import annotations

import sys
from pathlib import Path

import click
import llm
from click_default_group import DefaultGroup

from .logs import log_response, logging_enabled, open_logs_db
from .models import FileRef, PlanError
from .runner import CLIContext, DEFAULT_RETRIES, PlanRunner, explain, load_plan
from .store import list_plans, resolve_plan, user_plan_dir


@click.group(cls=DefaultGroup, default="run", default_if_no_args=False)
def plan():
    """Run multi-stage LLM plans: DAGs of prompts and scripts.

    "llm plan PLAN ..." is shorthand for "llm plan run PLAN ...".
    """


class _InstructionOrderCommand(click.Command):
    """Record the command-line order of -i and --ci occurrences.

    Click hands each multiple= option's values over as one tuple, losing how
    -i and --ci interleaved; their relative order changes the composed prompt,
    so capture it from the raw arguments before normal parsing.
    """

    def parse_args(self, ctx, args):
        order = []
        skip = 0
        for token in args:
            if skip:
                skip -= 1
                continue
            if token in ("-i", "--instruction") or token.startswith("--instruction="):
                order.append("plain")
                skip = 0 if "=" in token else 1
            elif token in ("--ci", "--context-instruction"):
                order.append("headed")
                skip = 2
        ctx.meta["instruction_order"] = order
        return super().parse_args(ctx, args)


@plan.command(name="run", cls=_InstructionOrderCommand)
@click.argument("plan_ref")
@click.argument("prompt", required=False)
@click.option("instructions", "-i", "--instruction", multiple=True,
              help="Instruction text, before the prompt argument (repeatable)")
@click.option("headed_instructions", "--ci", "--context-instruction", multiple=True,
              type=(str, str),
              help="Headed instructions: --ci TEXT HEADING (repeatable)")
@click.option("fragments", "-f", "--fragment", multiple=True,
              help="Seed context: file path, URL, alias, hash or prefix:argument")
@click.option("context_files", "--cf", "--context-file", multiple=True,
              type=(click.Path(exists=True, dir_okay=False), str),
              help="Labelled seed file: --cf PATH LABEL (repeatable)")
@click.option("attachments", "-a", "--attachment", multiple=True,
              help="Seed attachment: file path or URL")
@click.option("model_id", "-m", "--model", help="Override the model for every LLM stage")
@click.option("options", "-o", "--option", type=(str, str), multiple=True,
              help="Model option key/value, applied to every LLM stage")
@click.option("plan_args", "--plan-arg", multiple=True,
              help="Extra argument forwarded to every script stage")
@click.option("do_explain", "--explain", is_flag=True,
              help="Print the plan's DAG and commands without executing")
@click.option("--retries", default=DEFAULT_RETRIES, show_default=True,
              type=click.IntRange(min=0),
              help="Retries per LLM stage after a failure")
@click.option("no_log", "-n", "--no-log", is_flag=True, help="Don't log to the database")
@click.option("force_log", "--log", is_flag=True, help="Log even if logging is off")
@click.option("database", "-d", "--database",
              type=click.Path(dir_okay=False, writable=True, allow_dash=False),
              help="Path to a log database to use instead of logs.db")
@click.option("quiet", "-q", "--quiet", is_flag=True, help="Suppress progress output")
def run_(plan_ref, prompt, instructions, headed_instructions, fragments,
         context_files, attachments, model_id, options, plan_args, do_explain,
         retries, no_log, force_log, database, quiet):
    """Execute a plan by alias or path.

    The final (leaf) stage text prints to stdout; progress goes to stderr.
    Every model response is logged to llm's logs.db - inspect a run with
    "llm logs".

    \b
    Examples:
      llm plan run synthesis_full "Review this design" -f notes.md
      cat notes.md | llm plan run ./my_plan.yaml -m claude-4.5-haiku
    """
    try:
        loaded = load_plan(resolve_plan(plan_ref))
    except PlanError as exc:
        raise click.ClickException(str(exc))

    if do_explain:
        click.echo(explain(loaded, list(plan_args)), nl=False)
        return

    db = open_logs_db(database)
    order = click.get_current_context().meta.get("instruction_order", [])
    context = CLIContext(
        instructions=_read_prompt(prompt),
        instruction_parts=_ordered_instruction_parts(order, instructions, headed_instructions),
        files=[FileRef(path=Path(path), label=label) for path, label in context_files],
        model=model_id,
        options=dict(options),
        plan_args=list(plan_args),
    )
    _resolve_seed_inputs(db, fragments, attachments, context)

    progress = None if quiet else (lambda message: click.echo(message, err=True))
    logged: list[tuple[str, str]] = []
    on_response = None
    if logging_enabled(force_log=force_log, no_log=no_log):

        def on_response(stage, response):
            log_response(db, response)
            logged.append((stage.name, response.id))

    runner = PlanRunner(
        loaded, context, on_response=on_response, progress=progress, retries=retries
    )
    try:
        runner.run()
    except PlanError as exc:
        _emit_tracking(runner.run_id, logged, quiet)
        raise click.ClickException(str(exc))
    _emit_tracking(runner.run_id, logged, quiet)

    leaves = runner.leaf_stages()
    succeeded = [runner.results[name] for name in leaves if runner.results[name].success]
    for index, result in enumerate(succeeded):
        if len(succeeded) > 1:
            click.echo(f"## {result.name}\n")
        click.echo(result.text)
        if index < len(succeeded) - 1:
            click.echo()

    failed = [name for name in leaves if not runner.results[name].success]
    if failed:
        details = "; ".join(f"{name}: {runner.results[name].error}" for name in failed)
        raise click.ClickException(f"Plan '{loaded.name or plan_ref}' failed - {details}")


def _emit_tracking(run_id: str, logged: list, quiet: bool) -> None:
    """Report the run and its stage → response-id mappings for llm logs."""
    if not logged or quiet:
        return
    click.echo(f"Run {run_id}:", err=True)
    for stage_name, response_id in logged:
        click.echo(f"  [{stage_name}] response {response_id}", err=True)
    click.echo(
        f"Logged {len(logged)} response(s); view with: llm logs -n {len(logged)}",
        err=True,
    )


def _ordered_instruction_parts(
    order: list, plain: tuple, headed: tuple
) -> list[tuple[str | None, str]]:
    """Interleave -i and --ci values back into their command-line order."""
    if order.count("plain") == len(plain) and order.count("headed") == len(headed):
        plain_values = iter(plain)
        headed_values = iter(headed)
        parts: list[tuple[str | None, str]] = []
        for kind in order:
            if kind == "plain":
                parts.append((None, next(plain_values)))
            else:
                text, heading = next(headed_values)
                parts.append((heading, text))
        return parts
    # The raw-argument scan disagreed with click's parse (unusual quoting):
    # keep every value, at the cost of -i/--ci interleaving.
    return [(None, text) for text in plain] + [(heading, text) for text, heading in headed]


def _read_prompt(argument: str | None) -> str:
    """Combine piped stdin and the prompt argument, stdin first (like llm)."""
    parts = []
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            parts.append(piped)
    if argument:
        parts.append(argument)
    return " ".join(parts)


def _resolve_seed_inputs(db, fragments, attachments, context: CLIContext) -> None:
    """Resolve -f/-a values into llm Fragment and Attachment objects."""
    from llm.cli import resolve_fragments

    try:
        resolved = resolve_fragments(db, fragments, allow_attachments=True)
    except Exception as exc:
        raise click.ClickException(str(exc))
    for item in resolved:
        if isinstance(item, llm.Attachment):
            context.attachments.append(item)
        else:
            context.fragments.append(item)

    for value in attachments:
        if "://" in value:
            context.attachments.append(llm.Attachment(url=value))
        else:
            path = click.Path(exists=True, dir_okay=False).convert(value, None, None)
            context.attachments.append(llm.Attachment(path=str(path)))


@plan.command(name="list")
@click.option("as_json", "--json", is_flag=True, help="Output as JSON")
def list_(as_json):
    """List available plans (bundled, user and $LLM_PLAN_DIRS)."""
    import json

    plans = list_plans()
    if as_json:
        click.echo(json.dumps(
            [{"alias": p.alias, "name": p.name, "summary": p.summary, "path": str(p.path)}
             for p in plans],
            indent=2,
        ))
        return
    if not plans:
        click.echo(f"No plans found. Add one to {user_plan_dir()} or set LLM_PLAN_DIRS.")
        return
    width = max(len(p.alias) for p in plans) + 2
    for p in plans:
        click.echo(f"{p.alias:<{width}}{p.summary}")


@plan.command(name="show")
@click.argument("plan_ref")
def show_(plan_ref):
    """Print a plan's YAML and its resolved path."""
    try:
        path = resolve_plan(plan_ref)
    except PlanError as exc:
        raise click.ClickException(str(exc))
    click.echo(f"# {path}", err=True)
    click.echo(path.read_text(encoding="utf-8"), nl=False)


@plan.command(name="path")
def path_():
    """Show the directory for your personal plans."""
    click.echo(user_plan_dir())
