# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`llm-plan` is a plugin for Simon Willison's [LLM](https://llm.datasette.io/) CLI, not a standalone tool — it registers the `llm plan` command group via llm's plugin entry point (`__init__.py`). A *plan* is a YAML DAG of *stages* (LLM prompts or Python scripts) whose outputs chain into their dependents. Supports llm >= 0.31, Python 3.10–3.14, POSIX only (script process management uses sessions/`killpg`; Windows is unsupported).

The README is the authoritative reference for user-facing behaviour: plan YAML fields, the prompt grammar (`CLI`, `inline:`, `chain:`, file paths), the script-stage contract, and CLI options. Keep it in sync when changing any of those.

## Commands

```bash
# Setup (a dev venv usually already exists at .venv/)
uv venv && uv pip install -e '.[test]'

# Tests — hermetic: fake in-process models, no network, no API keys
.venv/bin/python -m pytest                                  # full suite (~322 tests, fast)
.venv/bin/python -m pytest tests/test_runner.py             # one file
.venv/bin/python -m pytest tests/test_parser.py -k chain    # by keyword

# Lint (no project config; ruff defaults)
ruff check llm_plan tests
```

Exercising the real CLI (the plugin is installed editable into `.venv`):

```bash
.venv/bin/llm plan list
.venv/bin/llm plan run ./plan.yaml "question" --explain < /dev/null
```

`--explain` previews the resolved DAG without spending anything; `-m MODEL` overrides every LLM stage for a cheap end-to-end run. **Always redirect `< /dev/null`** unless piping input: `llm plan run` reads stdin whenever it is not a tty (like `llm prompt`), so it blocks under the Bash tool otherwise.

## Architecture

Execution flows through the modules in this order:

- `cli.py` — the `llm plan` click group (`DefaultGroup`, default subcommand `run`). Builds a `CLIContext` from args/stdin, wires the logging callback, prints leaf-stage output to stdout and progress/run-id to stderr.
- `store.py` — plan alias → path resolution: `$LLM_PLAN_DIRS`, then `<llm user dir>/plans/`, then the bundled `llm_plan/plans/`. Files starting with `_` are shared includes, hidden from `list`.
- `parser.py` — YAML loading: recursive `extends:` deep-merge, `${namespace.key}` variable substitution (any non-reserved top-level mapping is a variable table; `cli.*`/`run.*` are deferred to runtime), prompt-grammar classification, and per-stage field validation.
- `dag.py` — dependency semantics and validation. `seed` is a virtual always-satisfied node granting a stage the CLI context.
- `runner.py` — `PlanRunner`: sequential or thread-pool DAG scheduling, prompt composition, retries, script subprocess management, per-stage scratch dirs.
- `script_stage.py` — the `python_script` stage contract (argv/env/stdout-manifest protocol), runtime `${cli.*}`/`${run.*}` validation, and public helpers for script authors (`read_instructions()`, `emit_outputs()`, …).
- `logs.py` — logging to llm's `logs.db`. Deliberately the only module that touches llm's *logging* internals, so upstream changes land in one place.
- `models.py` — dataclasses (`Plan`, `Stage`, `StageResult`, `PromptSpec`) and `PlanError`.

### Invariants that span modules

- **Everything validates before anything runs.** `load_plan()` (runner.py) = `parse_plan` + `validate_plan` + `validate_runtime_vars`. Missing prompt files, unknown stage keys (with did-you-mean), malformed `depends_on`, undefined `${vars}`, forward/unknown `chain:` targets, and runtime vars in LLM prompts all fail at load — before any stage spends money. Preserve this property when adding stage fields.
- **All `llm.cli` imports must be lazy (function-level).** llm.cli imports this plugin while initialising, so a module-level `from llm.cli import ...` is a circular-import crash. See `cli._AttachmentType`, `logs.py`, `runner._merged_options`.
- **Dependency edges come from three places** (`dag.resolve_dependencies`): explicit `depends_on`; an implicit edge to the previously listed stage when a non-first stage has neither `depends_on` nor a `CLI` prompt; and `chain:` prompt items. A `chain:` dependency's text is placed only where the item puts it — it is excluded from the auto-fenced dependency sections so the same text is never sent (and billed) twice.
- **Chaining is in-memory** for LLM stages; nothing is written to disk. Script stages get each LLM dependency's text materialised as `<dep>.md` inside the stage's *own* scratch subdirectory — stages never share one (parallel stages sharing a directory previously raced).
- **Parallel mode: workers only execute.** Results are collected, chained, and logged on the coordinating thread. `exclusive` stages drain in-flight work and run alone. On any abort, a latched flag (`_scripts_aborted`) plus process-group SIGKILL prevents a mid-spawn script escaping, and the executor shuts down with `cancel_futures` so queued stages aren't billed.
- **Logging is a promise, not best-effort.** One `llm.Conversation` per run (id = the full-uuid run id) so `llm logs --cid RUN_ID` retrieves the whole run; a logging failure fails the command (exit 1) after printing leaf output. Upstream quirks handled in `logs.py`: `log_to_db` ignores `Conversation.name` (the name column is updated manually) and the finished response must be appended to `conversation.responses` explicitly.
- **llm 0.31 compatibility:** model options pass to `model.prompt()` as `**kwargs` (0.31 has no `options=`), and `stream=True` is set only on `can_stream` models (Anthropic's SDK rejects non-streaming long requests). Retries skip deterministic failures (`NeedsKeyException`, `ValueError`, `NotImplementedError`, `TypeError`); `LLM_RAISE_ERRORS=1` re-raises immediately (llm's debugging convention).
- Stage names become scratch directory names — that's why the parser rejects separators/dot components in `name`.

## Tests

Development here is red-first TDD: write the failing test, then the fix.

- `conftest.py` sets `LLM_LOAD_PLUGINS=llm-plan` *before anything imports llm* (llm reads it once at import time) and registers throwaway in-process models per test — `echo` (echoes its composed prompt, for asserting composition), `flaky`, `hold`, `nostream`, `needskey`, `pair`/`timing` (parallelism proofs), etc. An autouse fixture sandboxes `LLM_USER_PATH`, tempdirs, and provider key env vars.
- `pyproject.toml` promotes `ResourceWarning` (and its unraisable wrapper) to errors — leaked file handles or unclosed databases fail the suite.
- Pinned tests to know about: the composed-prompt golden test (`test_runner.py::test_composed_prompt_matches_the_pinned_shape_exactly`) pins the section/fencing format, and `test_cli.py` pins each bundled prompt by sha256 (`plans/prompts/synthesise.md`, `plans/prompts/expand_research.md`) — editing one requires updating the matching `PINNED_*_SHA256` in the same commit.
- CI (`.github/workflows/test.yml`): Python 3.10–3.14 matrix, plus an llm==0.31 floor job and an llm-prerelease job.

## Branches and releases

- All work lands on `dev` (local-only branch); `origin` carries only `main`. `main` moves **only** by squash-merging `dev` — the branches share no ancestor, so use `--allow-unrelated-histories`; never amend on `main`. Right after a release, `git diff main dev` should be empty.
- Publishing (PyPI trusted publishing via the `release` GitHub environment, which requires `main` + a `v*` tag): bump the version **on dev** so `pyproject.toml` exactly matches the intended tag, squash-merge to `main`, wait for green, tag, then `gh release create` **non-draft** — `publish.yml` triggers on `release: created`, which never fires when a draft is later published. v0.1.0 is live; PEP 440 treats 0.1 == 0.1.0, so the next version must be 0.1.1 or 0.2.0.
- Do not add `Co-Authored-By` trailers to commits in this repo.

## Deliberate non-features

These are decisions, not gaps — don't "fix" them unprompted: stage-level `system:` prompts, schema support, `--key` passthrough, fragments for stage `files:`, sequential mode aborting on first failure (legacy behaviour, golden-tested), and no run persistence/resume (a crashed run keeps only its logs.db entries and scratch dir).

## Gotchas

- `build/`, `dist/`, and `*.egg-info/` are stale build artifacts (gitignored). The real source is `llm_plan/`; ignore grep hits under `build/lib/`.
- The `-q` short flag is deliberately not used for `--quiet` (upstream llm uses `-q` for model queries).
