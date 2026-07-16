#!/usr/bin/env python3
"""Run Google's Gemini Deep Research agent as an llm-plan script stage.

Calls the Gemini Interactions API over plain REST (stdlib only — no
google-genai dependency, so the plan's default ``python3`` works):

    POST https://generativelanguage.googleapis.com/v1beta/interactions
    GET  https://generativelanguage.googleapis.com/v1beta/interactions/{id}

Deep Research is exclusively available through this API. The interaction is
created with ``background=True`` + ``store=True`` (required for background
runs) and polled until it reaches a terminal status. Research runs take
minutes to an hour and cost real money (roughly $1-3 per task on the
standard tier, $3-7 on max), so the interaction id is written to
``interaction_id.txt`` in the stage's scratch directory as soon as the
interaction is created: if the run is killed, the research keeps going
server-side and the finished report can still be fetched by hand.

Script-stage contract (see the llm-plan README): the research question is
the last argv path — the materialised output of the ``expand_prompt``
dependency — falling back to $LLM_PLAN_INSTRUCTIONS so the script also works
as a first stage. The report is written to $LLM_PLAN_OUTPUT_DIR/report.md
and announced with a JSON manifest on stdout; progress goes to stderr.

Needs a Gemini API key (https://aistudio.google.com/apikey): $GEMINI_API_KEY,
$LLM_GEMINI_KEY, or the key stored by `llm keys set gemini`.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API_BASE = "https://generativelanguage.googleapis.com/v1beta/interactions"

# Deep Research agent tiers:
#   deep-research-preview-04-2026      — standard (default): ~80 searches
#   deep-research-max-preview-04-2026  — max: ~160 searches, 2-3x the cost
# Precedence: --agent flag, then $GEMINI_DEEP_RESEARCH_AGENT, then this.
DEFAULT_AGENT = "deep-research-preview-04-2026"

DEFAULT_MAX_WAIT_MINUTES = 60.0  # the API's own research-duration maximum
POLL_INTERVAL_SECONDS = 15
HTTP_TIMEOUT_SECONDS = 120

# A background interaction keeps running server-side even when a status poll
# fails, so polling tolerates transient errors and only gives up after this
# many *consecutive* failures (or the --max-wait deadline).
MAX_CONSECUTIVE_POLL_FAILURES = 8

# Poll errors worth retrying, besides 5xx and connection failures. 400 is
# included because the poll endpoint intermittently returns a spurious 400
# ("invalid argument") on a bare GET of an in-progress interaction, typically
# right after a 500; a single bad poll must not abort a paid research run.
RETRYABLE_POLL_STATUSES = frozenset({400, 408, 409, 429})

SUCCESS_STATUS = "completed"
TERMINAL_FAILURE_STATUSES = frozenset(
    {"failed", "cancelled", "incomplete", "budget_exceeded"}
)


class ResearchError(Exception):
    """A failure with a message fit for the stage's error output."""


def resolve_api_key():
    """A Gemini API key from the environment, or the one stored by llm.

    Checked in order: $GEMINI_API_KEY (Google's convention), $LLM_GEMINI_KEY
    (llm-gemini's), then the "gemini" entry in llm's keys.json — located the
    way llm locates its user directory, since this script runs as a plain
    subprocess and cannot import llm.
    """
    for name in ("GEMINI_API_KEY", "LLM_GEMINI_KEY"):
        value = os.environ.get(name)
        if value:
            return value

    root = os.environ.get("LLM_USER_PATH")
    if not root:
        if sys.platform == "darwin":
            root = Path.home() / "Library" / "Application Support" / "io.datasette.llm"
        else:
            config = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
            root = Path(config) / "io.datasette.llm"
    try:
        keys = json.loads((Path(root) / "keys.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    key = keys.get("gemini")
    return key if isinstance(key, str) and key else None


def log(message):
    print(message, file=sys.stderr, flush=True)


def _request(method, url, api_key, body=None):
    """One JSON round-trip. Raises urllib.error.HTTPError / URLError."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def _is_transient_poll_error(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500 or exc.code in RETRYABLE_POLL_STATUSES
    # URLError, socket timeouts, connection resets.
    return isinstance(exc, (urllib.error.URLError, OSError))


def _interaction_error(interaction):
    error = interaction.get("error")
    if isinstance(error, dict):
        return error.get("message") or json.dumps(error)
    return str(error) if error else "no error detail provided"


def report_text(interaction):
    """The final report: ``output_text``, or the last step's text content."""
    text = interaction.get("output_text")
    if text:
        return text
    steps = interaction.get("steps") or []
    if not steps:
        return ""
    parts = [
        item["text"]
        for item in (steps[-1].get("content") or [])
        if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
    ]
    return "".join(parts)


def create_interaction(question, agent, api_key):
    body = {
        "input": question,
        "agent": agent,
        "background": True,
        # background=True requires store=True per the Interactions API docs.
        "store": True,
        # Only the text report is consumed, so turn the data-visualization
        # step off: its chart artifacts bloat the stored interaction.
        "agent_config": {"type": "deep-research", "visualization": "off"},
    }
    return _request("POST", API_BASE, api_key, body=body)


def poll_until_complete(interaction_id, api_key, max_wait_minutes):
    """Poll until a terminal status; return the completed interaction."""
    start = time.monotonic()
    deadline = start + max_wait_minutes * 60
    consecutive_failures = 0

    while True:
        if time.monotonic() > deadline:
            raise ResearchError(
                f"Deep Research did not complete within {max_wait_minutes:g} "
                f"minutes. It may still be running server-side. "
                f"Interaction ID: {interaction_id}"
            )

        try:
            interaction = _request(
                "GET", f"{API_BASE}/{interaction_id}", api_key
            )
        except Exception as exc:
            if not _is_transient_poll_error(exc):
                raise ResearchError(
                    f"Deep Research polling failed: {exc}. "
                    f"Interaction ID: {interaction_id}"
                ) from exc
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_POLL_FAILURES:
                raise ResearchError(
                    f"Deep Research polling failed {consecutive_failures} times "
                    f"in a row; giving up. The research may still be running "
                    f"server-side. Last error: {exc}. "
                    f"Interaction ID: {interaction_id}"
                ) from exc
            elapsed = time.monotonic() - start
            log(
                f"  [{elapsed:>5.0f}s] poll error "
                f"({consecutive_failures}/{MAX_CONSECUTIVE_POLL_FAILURES}), "
                f"retrying: {exc}"
            )
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        consecutive_failures = 0
        status = interaction.get("status")
        elapsed = time.monotonic() - start

        if status == SUCCESS_STATUS:
            log(f"Research completed in {elapsed:.0f}s")
            return interaction
        if status in TERMINAL_FAILURE_STATUSES:
            raise ResearchError(
                f"Deep Research ended with status {status!r}: "
                f"{_interaction_error(interaction)}. "
                f"Interaction ID: {interaction_id}"
            )

        log(f"  [{elapsed:>5.0f}s] status: {status}")
        time.sleep(POLL_INTERVAL_SECONDS)


def read_question(files):
    """The last argv path is the closest dependency's materialised output."""
    if files:
        return Path(files[-1]).read_text(encoding="utf-8")
    return os.environ.get("LLM_PLAN_INSTRUCTIONS", "")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", help=f"Deep Research agent (default {DEFAULT_AGENT})")
    parser.add_argument(
        "--max-wait",
        type=float,
        default=DEFAULT_MAX_WAIT_MINUTES,
        metavar="MINUTES",
        help="give up polling after this many minutes",
    )
    parser.add_argument("files", nargs="*", help="dependency output paths (from llm-plan)")
    args = parser.parse_args(argv)

    api_key = resolve_api_key()
    if not api_key:
        log(
            "No Gemini API key found: set GEMINI_API_KEY (or LLM_GEMINI_KEY), "
            "or store one with `llm keys set gemini`. Get a key from "
            "https://aistudio.google.com/apikey"
        )
        return 1

    question = read_question(args.files).strip()
    if not question:
        log("No research question: no dependency file and no CLI instructions.")
        return 1

    out_dir = Path(os.environ.get("LLM_PLAN_OUTPUT_DIR", "."))
    agent = args.agent or os.environ.get("GEMINI_DEEP_RESEARCH_AGENT") or DEFAULT_AGENT

    log(f"Starting Deep Research ({agent}): {question[:80]}...")
    try:
        interaction = create_interaction(question, agent, api_key)
    except (urllib.error.URLError, OSError) as exc:
        log(f"Could not start Deep Research: {exc}")
        return 1

    interaction_id = interaction.get("id", "")
    # Recorded immediately: the run can be killed but the research keeps
    # going server-side, and this id is what retrieves the finished report.
    (out_dir / "interaction_id.txt").write_text(interaction_id + "\n", encoding="utf-8")
    log(f"Interaction ID: {interaction_id}")
    log(f"Polling every {POLL_INTERVAL_SECONDS}s (max {args.max_wait:g}m)...")

    try:
        final = poll_until_complete(interaction_id, api_key, args.max_wait)
    except ResearchError as exc:
        log(str(exc))
        return 1

    report = report_text(final)
    if not report.strip():
        log(f"Completed interaction contained no report text. Interaction ID: {interaction_id}")
        return 1

    report_path = out_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {"outputs": [{"path": str(report_path), "label": "Gemini Deep Research Report"}]}
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
