"""Unit tests for the bundled Deep Research script (plans/scripts/deep_research.py).

The script is stdlib-only and calls the Gemini Interactions API over REST;
these tests replace its HTTP layer with fakes — no network, no real keys.
"""

import importlib.util
import json
import urllib.error
from pathlib import Path

import pytest

from llm_plan.store import BUNDLED_PLAN_DIR

SCRIPT_PATH = BUNDLED_PLAN_DIR / "scripts" / "deep_research.py"

INTERACTION_ID = "interactions/abc123"

COMPLETED = {
    "id": INTERACTION_ID,
    "status": "completed",
    "output_text": "# The Report\n\nFindings.",
}


@pytest.fixture
def script(monkeypatch):
    spec = importlib.util.spec_from_file_location("deep_research_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    return module


@pytest.fixture
def env(monkeypatch, tmp_path):
    """A script-stage environment: an output dir and an API key."""
    out_dir = tmp_path / "scratch"
    out_dir.mkdir()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PLAN_OUTPUT_DIR", str(out_dir))
    monkeypatch.delenv("LLM_PLAN_INSTRUCTIONS", raising=False)
    monkeypatch.delenv("GEMINI_DEEP_RESEARCH_AGENT", raising=False)
    return out_dir


def question_file(tmp_path, text="expanded research brief"):
    path = tmp_path / "expand_prompt.md"
    path.write_text(text, encoding="utf-8")
    return str(path)


def http_error(code):
    return urllib.error.HTTPError("https://api.example", code, "boom", None, None)


class FakeAPI:
    """Replaces the script's _request: POST records creates, GET replays polls.

    The poll list is consumed one response per GET; the final entry repeats
    forever. An Exception instance in the list is raised instead of returned.
    """

    def __init__(self, polls, created=None):
        self.polls = list(polls)
        self.created = created or {"id": INTERACTION_ID, "status": "in_progress"}
        self.creates = []
        self.api_keys = []
        self.gets = 0

    def __call__(self, method, url, api_key, body=None):
        self.api_keys.append(api_key)
        if method == "POST":
            self.creates.append(body)
            return self.created
        self.gets += 1
        result = self.polls.pop(0) if len(self.polls) > 1 else self.polls[0]
        if isinstance(result, Exception):
            raise result
        return result


def run(script, monkeypatch, api, argv):
    monkeypatch.setattr(script, "_request", api)
    return script.main(argv)


class TestHappyPath:
    def test_writes_report_emits_manifest_and_records_interaction_id(
        self, script, monkeypatch, env, tmp_path, capsys
    ):
        api = FakeAPI(polls=[{"status": "in_progress"}, COMPLETED])

        rc = run(script, monkeypatch, api, [question_file(tmp_path)])

        assert rc == 0
        manifest = json.loads(capsys.readouterr().out)
        [output] = manifest["outputs"]
        assert output["label"] == "Gemini Deep Research Report"
        report = Path(output["path"])
        assert report.parent == env
        assert report.read_text(encoding="utf-8") == "# The Report\n\nFindings."
        recorded = (env / "interaction_id.txt").read_text(encoding="utf-8")
        assert recorded.strip() == INTERACTION_ID

    def test_create_request_carries_the_question_and_agent_config(
        self, script, monkeypatch, env, tmp_path
    ):
        api = FakeAPI(polls=[COMPLETED])

        rc = run(
            script,
            monkeypatch,
            api,
            ["--agent", "deep-research-max-preview-04-2026", question_file(tmp_path)],
        )

        assert rc == 0
        [body] = api.creates
        assert body["input"] == "expanded research brief"
        assert body["agent"] == "deep-research-max-preview-04-2026"
        assert body["background"] is True
        assert body["store"] is True
        assert body["agent_config"] == {"type": "deep-research", "visualization": "off"}

    def test_agent_env_var_applies_when_no_flag_is_given(
        self, script, monkeypatch, env, tmp_path
    ):
        monkeypatch.setenv("GEMINI_DEEP_RESEARCH_AGENT", "deep-research-pro-preview-12-2025")
        api = FakeAPI(polls=[COMPLETED])

        rc = run(script, monkeypatch, api, [question_file(tmp_path)])

        assert rc == 0
        assert api.creates[0]["agent"] == "deep-research-pro-preview-12-2025"

    def test_question_falls_back_to_cli_instructions_without_dependency_files(
        self, script, monkeypatch, env
    ):
        monkeypatch.setenv("LLM_PLAN_INSTRUCTIONS", "the raw question")
        api = FakeAPI(polls=[COMPLETED])

        rc = run(script, monkeypatch, api, [])

        assert rc == 0
        assert api.creates[0]["input"] == "the raw question"

    def test_report_text_extracted_from_steps_when_output_text_is_missing(
        self, script, monkeypatch, env, tmp_path, capsys
    ):
        completed = {
            "id": INTERACTION_ID,
            "status": "completed",
            "steps": [
                {"content": [{"type": "text", "text": "ignored earlier step"}]},
                {
                    "content": [
                        {"type": "thought_summary", "content": {"text": "thinking"}},
                        {"type": "text", "text": "part one"},
                        {"type": "text", "text": " and two"},
                    ]
                },
            ],
        }
        api = FakeAPI(polls=[completed])

        rc = run(script, monkeypatch, api, [question_file(tmp_path)])

        assert rc == 0
        manifest = json.loads(capsys.readouterr().out)
        report = Path(manifest["outputs"][0]["path"])
        assert report.read_text(encoding="utf-8") == "part one and two"


class TestApiKeyResolution:
    def test_llm_gemini_key_env_var_is_accepted(
        self, script, monkeypatch, env, tmp_path
    ):
        monkeypatch.delenv("GEMINI_API_KEY")
        monkeypatch.setenv("LLM_GEMINI_KEY", "llm-env-key")
        api = FakeAPI(polls=[COMPLETED])

        rc = run(script, monkeypatch, api, [question_file(tmp_path)])

        assert rc == 0
        assert api.api_keys[0] == "llm-env-key"

    def test_a_key_stored_by_llm_keys_set_gemini_is_found(
        self, script, monkeypatch, env, tmp_path, user_dir
    ):
        monkeypatch.delenv("GEMINI_API_KEY")
        (user_dir / "keys.json").write_text(
            json.dumps({"gemini": "stored-key"}), encoding="utf-8"
        )
        api = FakeAPI(polls=[COMPLETED])

        rc = run(script, monkeypatch, api, [question_file(tmp_path)])

        assert rc == 0
        assert api.api_keys[0] == "stored-key"

    def test_the_env_var_beats_the_stored_key(
        self, script, monkeypatch, env, tmp_path, user_dir
    ):
        (user_dir / "keys.json").write_text(
            json.dumps({"gemini": "stored-key"}), encoding="utf-8"
        )
        api = FakeAPI(polls=[COMPLETED])

        rc = run(script, monkeypatch, api, [question_file(tmp_path)])

        assert rc == 0
        assert api.api_keys[0] == "test-key"

    def test_the_missing_key_error_names_every_option(
        self, script, monkeypatch, env, tmp_path, capsys
    ):
        monkeypatch.delenv("GEMINI_API_KEY")
        api = FakeAPI(polls=[COMPLETED])

        rc = run(script, monkeypatch, api, [question_file(tmp_path)])

        assert rc == 1
        err = capsys.readouterr().err
        assert "GEMINI_API_KEY" in err
        assert "llm keys set gemini" in err


class TestFailureModes:
    def test_missing_api_key_fails_fast_without_any_request(
        self, script, monkeypatch, env, tmp_path, capsys
    ):
        monkeypatch.delenv("GEMINI_API_KEY")
        api = FakeAPI(polls=[COMPLETED])

        rc = run(script, monkeypatch, api, [question_file(tmp_path)])

        assert rc == 1
        assert "GEMINI_API_KEY" in capsys.readouterr().err
        assert api.creates == []
        assert api.gets == 0

    def test_missing_question_fails_cleanly(self, script, monkeypatch, env, capsys):
        api = FakeAPI(polls=[COMPLETED])

        rc = run(script, monkeypatch, api, [])

        assert rc == 1
        assert "question" in capsys.readouterr().err.lower()
        assert api.creates == []

    def test_a_failed_create_is_a_clean_error(
        self, script, monkeypatch, env, tmp_path, capsys
    ):
        def refuse(method, url, api_key, body=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(script, "_request", refuse)

        rc = script.main([question_file(tmp_path)])

        assert rc == 1
        assert "connection refused" in capsys.readouterr().err

    def test_terminal_status_reports_status_detail_and_interaction_id(
        self, script, monkeypatch, env, tmp_path, capsys
    ):
        api = FakeAPI(
            polls=[{"status": "budget_exceeded", "error": {"message": "quota blown"}}]
        )

        rc = run(script, monkeypatch, api, [question_file(tmp_path)])

        assert rc == 1
        err = capsys.readouterr().err
        assert "budget_exceeded" in err
        assert "quota blown" in err
        assert INTERACTION_ID in err

    def test_transient_poll_errors_are_tolerated(
        self, script, monkeypatch, env, tmp_path
    ):
        api = FakeAPI(
            polls=[
                http_error(500),
                http_error(400),
                {"status": "in_progress"},
                http_error(429),
                COMPLETED,
            ]
        )

        rc = run(script, monkeypatch, api, [question_file(tmp_path)])

        assert rc == 0
        assert api.gets == 5

    def test_persistent_transient_errors_give_up_after_the_cap(
        self, script, monkeypatch, env, tmp_path, capsys
    ):
        api = FakeAPI(polls=[http_error(503)])

        rc = run(script, monkeypatch, api, [question_file(tmp_path)])

        assert rc == 1
        assert api.gets == script.MAX_CONSECUTIVE_POLL_FAILURES
        assert INTERACTION_ID in capsys.readouterr().err

    def test_a_non_transient_poll_error_aborts_immediately(
        self, script, monkeypatch, env, tmp_path, capsys
    ):
        api = FakeAPI(polls=[http_error(403)])

        rc = run(script, monkeypatch, api, [question_file(tmp_path)])

        assert rc == 1
        assert api.gets == 1
        assert INTERACTION_ID in capsys.readouterr().err

    def test_the_deadline_expiring_reports_the_interaction_id(
        self, script, monkeypatch, env, tmp_path, capsys
    ):
        api = FakeAPI(polls=[{"status": "in_progress"}])

        rc = run(script, monkeypatch, api, ["--max-wait", "0", question_file(tmp_path)])

        assert rc == 1
        assert INTERACTION_ID in capsys.readouterr().err

    def test_a_completed_interaction_with_no_text_fails(
        self, script, monkeypatch, env, tmp_path, capsys
    ):
        api = FakeAPI(polls=[{"id": INTERACTION_ID, "status": "completed"}])

        rc = run(script, monkeypatch, api, [question_file(tmp_path)])

        assert rc == 1
        assert "no report text" in capsys.readouterr().err.lower()
