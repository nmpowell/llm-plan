"""llm-plan: run multi-stage LLM workflows.

Adds an ``llm plan`` command group for executing plans: YAML-defined DAGs of
LLM prompts and Python scripts, with each stage's output chained into its
dependents. Models resolve through llm's own registry (so aliases work) and
responses are logged to llm's logs.db.
"""

import llm


@llm.hookimpl
def register_commands(cli):
    from .cli import plan

    cli.add_command(plan)
