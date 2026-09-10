"""Dependency resolution, validation and ordering for plan DAGs.

"seed" is a virtual node: it never executes, is always satisfied, and marks a
stage as wanting the CLI context (instructions, fragments, attachments).
"""

from __future__ import annotations

from collections import deque

from .models import Plan, PlanError, Stage
from .parser import prompt_chain_targets, prompt_has_cli

SEED = "seed"


def resolve_dependencies(stage: Stage, index: int, stages: list[Stage]) -> list[str]:
    """Effective dependencies: explicit, implicit-previous and ``chain:`` edges.

    A non-first stage with no explicit ``depends_on`` that does not use a CLI
    prompt depends on the stage listed before it. Stages named by ``chain:``
    prompt items are dependencies too - the chained text cannot exist until
    its stage has run.
    """
    if stage.depends_on:
        deps = list(stage.depends_on)
    elif index > 0 and not prompt_has_cli(stage.prompt):
        deps = [stages[index - 1].name]
    else:
        deps = []
    deps += [t for t in prompt_chain_targets(stage.prompt) if t not in deps]
    return deps


def validate_plan(plan: Plan) -> None:
    """Check stage names (unique, not the reserved 'seed'), dependency and
    ``chain:`` references, order and acyclicity."""
    stages = plan.stages
    names = [s.name for s in stages]
    if SEED in names:
        raise PlanError(
            f"Stage name '{SEED}' is reserved for the virtual seed node; "
            f"rename that stage."
        )
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise PlanError(f"Duplicate stage names: {duplicates}")

    known = set(names) | {SEED}
    for stage in stages:
        for dep in stage.depends_on:
            if dep not in known:
                raise PlanError(
                    f"Stage '{stage.name}' depends on unknown stage '{dep}'"
                )
        for target in prompt_chain_targets(stage.prompt):
            if target not in names:
                raise PlanError(
                    f"Stage '{stage.name}' references chain:'{target}' but no "
                    f"stage has that name"
                )

    seen = {SEED}
    for index, stage in enumerate(stages):
        for dep in resolve_dependencies(stage, index, stages):
            if dep not in seen:
                raise PlanError(
                    f"Stage '{stage.name}' depends on '{dep}' which appears later. "
                    f"Order stages so dependencies come before dependents."
                )
        seen.add(stage.name)

    # The order check above already rejects cycles among listed stages, but a
    # Kahn pass keeps the guarantee independent of that ordering rule.
    in_degree, dependents = _build_graph(stages)
    queue = deque(name for name, degree in in_degree.items() if degree == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for dependent in dependents[node]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
    if visited != len(in_degree):
        raise PlanError("Cycle detected in stage dependencies.")


def topological_order(stages: list[Stage]) -> list[str]:
    """Execution order for display; ties broken by listed order."""
    position = {stage.name: index for index, stage in enumerate(stages)}
    in_degree, dependents = _build_graph(stages)
    in_degree.pop(SEED, None)

    ready = sorted((n for n, d in in_degree.items() if d == 0), key=position.get)
    order = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for dependent in dependents[node]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                ready.append(dependent)
        ready.sort(key=position.get)
    return order


def leaf_stages(stages: list[Stage]) -> list[str]:
    """Stages nothing depends on - the plan's final outputs."""
    has_dependents = set()
    for index, stage in enumerate(stages):
        for dep in resolve_dependencies(stage, index, stages):
            if dep != SEED:
                has_dependents.add(dep)
    return [s.name for s in stages if s.name not in has_dependents]


def _build_graph(stages: list[Stage]) -> tuple[dict[str, int], dict[str, list[str]]]:
    """(in_degree, dependents) over stage names; seed edges are ignored."""
    in_degree = {stage.name: 0 for stage in stages}
    dependents: dict[str, list[str]] = {stage.name: [] for stage in stages}
    for index, stage in enumerate(stages):
        for dep in resolve_dependencies(stage, index, stages):
            if dep == SEED:
                continue
            dependents[dep].append(stage.name)
            in_degree[stage.name] += 1
    return in_degree, dependents
