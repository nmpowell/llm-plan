"""Click commands for llm-plan."""

from __future__ import annotations

import click

from .store import user_plan_dir


@click.group()
def plan():
    """Run multi-stage LLM plans: DAGs of prompts and scripts."""


@plan.command(name="path")
def path_():
    """Show the directory for your personal plans."""
    click.echo(user_plan_dir())
