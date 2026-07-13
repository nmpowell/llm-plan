# llm-plan

[![PyPI](https://img.shields.io/pypi/v/llm-plan.svg)](https://pypi.org/project/llm-plan/)
[![Changelog](https://img.shields.io/github/v/release/nmpowell/llm-plan?include_prereleases&label=changelog)](https://github.com/nmpowell/llm-plan/releases)
[![Tests](https://github.com/nmpowell/llm-plan/actions/workflows/test.yml/badge.svg)](https://github.com/nmpowell/llm-plan/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/nmpowell/llm-plan/blob/main/LICENSE)

Run multi-stage LLM workflows with [LLM](https://llm.datasette.io/): a *plan*
is a YAML file describing a DAG of *stages* — each an LLM prompt or a Python
script — and each stage's output chains automatically into its dependents.

Plans orchestrate; LLM does everything else. Models resolve through LLM's own
registry (so plugin models and `llm aliases` work), options are validated per
model, keys come from `llm keys`, and every response is logged to `logs.db`
so `llm logs` can find it later.

## Installation

Install this plugin in the same environment as LLM:

```bash
llm install llm-plan
```

## Quick start

Run the bundled `synthesis_full` plan: four models analyse your question in
parallel, then a fifth reconciles their answers into a single response:

```bash
llm plan synthesis_full "What are the trade-offs of a monorepo?" -f notes.md
```

The final stage's text prints to stdout; progress goes to stderr. Inspect the
run afterwards with `llm logs`.

(The bundled plan uses Anthropic, Gemini and OpenAI models, so it expects the
`llm-anthropic` and `llm-gemini` plugins plus the relevant keys. Preview any
plan without spending anything using `--explain`, or dry-run the whole DAG on
a cheap model with `-m`.)

## Usage

`llm plan PLAN ...` is shorthand for `llm plan run PLAN ...`. `PLAN` is a
file path or an alias: the alias `research` finds `plan_research.yaml` or
`research.yaml` in (first match wins):

1. the directories in `$LLM_PLAN_DIRS` (colon-separated)
2. your personal plans directory — `llm plan path` prints it
3. the plans bundled with this package

```bash
llm plan run PLAN [PROMPT] [options]
llm plan list            # available plans and their summaries
llm plan show PLAN       # print a plan's YAML (its path goes to stderr)
llm plan path            # your personal plans directory
```

Options for `run`:

- `PROMPT` — the seed instructions. Piped stdin is prepended, so
  `cat notes.md | llm plan review "focus on risks"` works like `llm prompt`.
- `-i/--instruction TEXT` — repeatable instruction parts, composed before the
  positional prompt.
- `--ci/--context-instruction TEXT HEADING` — instructions under a `## HEADING`
  of their own (repeatable).
- `-f/--fragment SOURCE` — seed context for stages that ask for CLI input: a
  file path, URL, fragment alias/hash, or plugin source like
  `github:owner/repo`. Resolved exactly like `llm prompt -f`.
- `--cf/--context-file PATH LABEL` — a labelled seed file, included as a
  fenced `## LABEL` section (repeatable).
- `-a/--attachment PATH_OR_URL` — seed attachments (images etc.); accepts `-`
  for stdin and validates paths and URLs up front, exactly like `llm prompt -a`.
- `-m/--model MODEL` — override the model for **every** LLM stage (handy for
  a cheap end-to-end check: `-m haiku-4.5`). Honours `$LLM_MODEL`.
- `-o/--option KEY VALUE` — model option applied to every LLM stage. Merge
  order per stage: `llm models options` defaults < the stage's `options:` <
  `-o`.
- `--plan-arg VALUE` — extra argument forwarded to every script stage.
- `--explain` — print the resolved DAG and script commands; execute nothing.
- `--retries N` — retries per LLM stage after a failure (default 2).
- `-n/--no-log` / `--log` / `-d/--database PATH` — logging control, exactly
  as `llm prompt`.
- `--quiet` — suppress stage progress on stderr.

The exit code is non-zero unless every *leaf* stage (one nothing depends on)
succeeded. When several leaves succeed, each prints under a `## <stage>`
header.

## Writing plans

```yaml
name: "review"
summary: "Two analysts in parallel, then a synthesis"

parallel_config:
  max_workers: 4          # omit (or 1) for sequential execution, if possible

models:                   # any top-level mapping becomes a ${variable} table
  strong: claude-opus-4.8 # values pass straight to llm's model resolution

stages:
  - name: "analyst"
    summary: "Careful technical analysis"
    model: "${models.strong}"
    prompt: "CLI"                 # the CLI prompt, -f fragments and -a attachments
    produces: "Analysis"          # heading its output gets downstream

  - name: "skeptic"
    summary: "Hunt for weaknesses"
    model: "gpt-5.6"
    prompt:
      - "CLI"
      - "inline:Act as a sceptical reviewer. List the strongest objections."
    produces: "Objections"

  - name: "synthesis"
    summary: "Reconcile both views"
    model: "${models.strong}"
    depends_on: [analyst, skeptic]
    partial_dependencies: true    # run even if some dependencies failed
    prompt:
      - prompt: "CLI:instructions"
        label: "Original Question"
      - prompt: "prompts/synthesise.md"   # a file, relative to the plan
        label: "Synthesis Instructions"
```

Save it as `plan_review.yaml` in the directory `llm plan path` prints, then:

```bash
llm plan review "Should we adopt feature flags?" -f design.md
```

### Stage fields

| Field | Meaning |
|:---|:---|
| `name`, `summary` | Required. `summary` shows in progress output and `--explain` |
| `type` | `llm` (default) or `python_script` |
| `model` | Any model ID or alias LLM resolves; omit to use `llm models default` |
| `prompt` | Prompt spec — see grammar below |
| `prompt_label` | Relabels the first prompt-*file* section |
| `produces` | Heading this stage's output gets when chained downstream |
| `depends_on` | Stages whose outputs feed this one; `seed` grants the CLI context to a later stage |
| `files` | `[{path, label}]` — files included as labelled sections (paths relative to the plan) |
| `attachments` | `[{path, label}]` — file paths or URLs passed as model attachments |
| `options` | Model options for this stage (validated against the model) |
| `exclusive` | In parallel plans, run entirely alone |
| `continue_on_failure` | Run even though dependencies failed |
| `partial_dependencies` | Run with whichever dependencies succeeded |
| `script`, `script_args`, `python`, `timeout`, `env` | Script stages — see below |

A stage with no `depends_on` and no `CLI` prompt implicitly depends on the
previous listed stage, so simple pipelines need no wiring; a stage whose
prompt mentions `CLI` starts a fresh branch instead.

### The prompt grammar

`prompt` is a string, or a list of strings / `{prompt, label}` mappings,
composed in order:

- `CLI` — the command-line instructions plus `-f`/`-a` seed context.
  `CLI:instructions` and `CLI:files` take just that part; `CLI:all` is both,
  explicitly.
- `inline:Some literal text` — quote it (`"inline:..."`); YAML parses a bare
  `prompt: inline: x` as a mapping.
- `chain:stage_name` — a completed stage's output text, verbatim. The named
  stage automatically becomes a dependency, and its text appears only where
  the `chain:` item places it (no duplicate auto-fenced copy).
- anything else — a file path, resolved against the plan's directory.

Sections compose with `## <label>` headings separated by `---` rules:
stage `files:` first, then dependency outputs (headed by their `produces`),
then prompt files (default heading `MAIN INSTRUCTIONS`), then
inline/chain/CLI parts in listed order.

Plans are validated fully before anything runs — missing prompt files,
unknown stage keys (with a did-you-mean), malformed `depends_on`, undefined
`${variables}` and forward `chain:` references all fail at load, before any
stage spends money.

### Script stages

```yaml
  - name: "fetch_pr"
    type: "python_script"
    summary: "Fetch PR details from GitHub"
    script: "scripts/fetch_pr.py"     # relative to the plan file
    script_args: ["--quiet", "--stage=${run.stage_name}"]
    timeout: 600                      # seconds (default 3600)
    env: {GH_PAGER: ""}
    produces: "PR Details"
```

The contract:

- **argv**: `--plan-arg` values, then `script_args`, then each dependency
  output as a file path (an LLM dependency's text is first written to
  `<stage>.md` in the stage's scratch directory).
- **environment**: `LLM_PLAN_INSTRUCTIONS`, `LLM_PLAN_OUTPUT_DIR` (the
  stage's own scratch directory — stages never share one), `LLM_PLAN_RUN_ID`
  and `LLM_PLAN_STAGE_NAME`, plus the stage's `env:`.
- **stdout**: one produced-file path per line, or a JSON manifest
  `{"outputs": [{"path": "...", "label": "..."}, ...]}` — manifest labels
  become section headings when the files chain into an LLM stage. Everything
  else belongs on stderr. Exit 0 on success. A `timeout` kills the script's
  whole process group.
- `${cli.instructions}`, `${cli.run_id}`, `${run.output_dir}` and
  `${run.stage_name}` are substituted into `script_args`, `env:` values and
  `--plan-arg` values at execution time (never into LLM prompts, where they
  are rejected at load).

`llm_plan.script_stage` offers helpers for script authors
(`read_instructions()`, `output_dir()`, `emit_outputs()`).

### Shared definitions

`extends: "_defaults.yaml"` deep-merges another YAML file underneath the plan
(the plan wins), recursively — a base file may extend another. Every other
top-level mapping — `models:`, `prompts:`, `files:`, anything you like —
becomes a `${name.key}` variable table for the whole document. Files starting
with `_` are hidden from `llm plan list`.

## Logging and prior runs

Every stage's prompt and response is logged to LLM's `logs.db` under the same
rules as `llm prompt` (`-n/--no-log`, `--log`, the `llm logs off` sentinel),
and a run's responses share one conversation whose id is the run id printed
on stderr — so one command retrieves the whole run. A failure to log fails
the command: the log is a promise, not best-effort.

```bash
llm logs --cid RUN_ID          # every stage of one run
llm logs -r | llm plan review "critique this"   # feed a response back in
llm plan synthesis_full "..." > answer.md        # stdout is the final answer
```

Script stages write real files into per-stage scratch directories under a
per-run root (its path is printed on stderr and kept after the run).

## Development

```bash
cd llm-plan
python -m venv .venv && source .venv/bin/activate
pip install -e '.[test]'
python -m pytest
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
cd llm-plan
uv venv && source .venv/bin/activate
uv pip install -e '.[test]'
python -m pytest
```

The test suite uses fake in-process models — no network calls, no API keys.
Tested against llm 0.31 and 0.32, on macOS and Linux (script-stage process
management is POSIX-only; Windows is untested and unsupported).
