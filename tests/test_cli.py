import json
import os
import textwrap

import pytest
import sqlite_utils
import yaml
from click.testing import CliRunner
from llm.cli import cli

not_as_root = pytest.mark.skipif(
    os.geteuid() == 0, reason="root ignores file permissions"
)


def write_plan(directory, stages, filename="plan.yaml", **top_level):
    data = {"name": "test", "summary": "a test plan", "stages": stages, **top_level}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def cli_stage(name="solo", model="echo", **kwargs):
    return {"name": name, "summary": f"{name} stage", "model": model, "prompt": "CLI", **kwargs}


PINNED_SYNTHESISE_SHA256 = "181893a6ee40d17fc049991fdded2ae5926a4d0980114ed12f30e902bf0e453a"


def invoke(*args, **kwargs):
    # Unexpected exceptions should fail loudly; click errors still produce
    # a normal Result because standalone mode converts them to SystemExit.
    kwargs.setdefault("catch_exceptions", False)
    return CliRunner().invoke(cli, list(args), **kwargs)


@pytest.fixture(autouse=True)
def no_ambient_model_env(monkeypatch):
    """-m reads $LLM_MODEL, so a developer's ambient value must not leak in."""
    monkeypatch.delenv("LLM_MODEL", raising=False)


@pytest.fixture
def logs_db(user_dir):
    db = sqlite_utils.Database(str(user_dir / "logs.db"))
    try:
        yield db
    finally:
        db.close()


class TestPlanPath:
    def test_prints_the_user_plans_directory(self, user_dir):
        result = invoke("plan", "path")

        assert result.exit_code == 0, result.output
        assert result.output.strip() == str(user_dir / "plans")


class TestPlanRun:
    def test_prints_the_leaf_stage_text_to_stdout(self, tmp_path):
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke("plan", "run", str(plan), "What is love?")

        assert result.exit_code == 0, result.stderr
        assert result.stdout == "ECHO[What is love?]\n"

    def test_run_is_the_default_subcommand(self, tmp_path):
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke("plan", str(plan), "hello")

        assert result.exit_code == 0, result.stderr
        assert result.stdout == "ECHO[hello]\n"

    def test_stdin_becomes_the_instructions(self, tmp_path):
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke("plan", "run", str(plan), input="piped question")

        assert result.exit_code == 0, result.stderr
        assert result.stdout == "ECHO[piped question]\n"

    def test_stdin_and_argument_combine_stdin_first(self, tmp_path):
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke("plan", "run", str(plan), "and the argument", input="from stdin")

        assert result.stdout == "ECHO[from stdin and the argument]\n"

    def test_piped_stdin_keeps_its_whitespace(self, tmp_path):
        # llm prompt does not strip stdin, so indented code survives piping
        # and the argument follows it after a single space.
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke(
            "plan", "run", str(plan), "explain this",
            input="def f():\n        return 1\n",
        )

        assert result.stdout == "ECHO[def f():\n        return 1\n explain this]\n"

    def test_aliases_resolve_from_the_user_plans_dir(self, user_dir, tmp_path):
        write_plan(user_dir / "plans", [cli_stage()], filename="plan_myalias.yaml")

        result = invoke("plan", "run", "myalias", "hi")

        assert result.exit_code == 0, result.stderr
        assert result.stdout == "ECHO[hi]\n"

    def test_file_fragments_reach_cli_stages(self, tmp_path):
        plan = write_plan(tmp_path, [cli_stage()])
        seed = tmp_path / "seed.md"
        seed.write_text("seed file content", encoding="utf-8")

        result = invoke("plan", "run", str(plan), "question", "-f", str(seed))

        assert "seed file content" in result.stdout
        assert "question" in result.stdout

    def test_missing_fragment_is_a_clean_error(self, tmp_path):
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke("plan", "run", str(plan), "hi", "-f", "no-such-fragment.md")

        assert result.exit_code == 1
        assert "no-such-fragment.md" in result.stderr
        assert "Traceback" not in result.stderr

    def test_a_bug_inside_fragment_resolution_propagates(self, tmp_path, monkeypatch):
        import llm.cli

        def broken_resolver(db, fragments, allow_attachments=False):
            raise TypeError("programming error")

        monkeypatch.setattr(llm.cli, "resolve_fragments", broken_resolver)
        plan = write_plan(tmp_path, [cli_stage()])

        with pytest.raises(TypeError, match="programming error"):
            invoke("plan", "run", str(plan), "hi", "-f", "anything")

    def test_an_upstream_rename_of_resolve_fragments_fails_intelligibly(
        self, tmp_path, monkeypatch
    ):
        from importlib.metadata import version

        monkeypatch.delattr("llm.cli.resolve_fragments")
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke("plan", "run", str(plan), "hi", "-f", "anything")

        assert result.exit_code == 1
        assert f"llm {version('llm')}" in result.stderr

    def test_repeatable_instructions_compose_in_order(self, tmp_path):
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke(
            "plan", "run", str(plan), "and the argument",
            "-i", "first instruction", "-i", "second instruction",
        )

        assert result.exit_code == 0, result.stderr
        assert result.stdout == (
            "ECHO[first instruction\n\nsecond instruction\n\nand the argument]\n"
        )

    def test_headed_instructions_via_ci_flag(self, tmp_path):
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke(
            "plan", "run", str(plan), "the question",
            "--ci", "look at concurrency", "Focus Areas",
        )

        assert result.exit_code == 0, result.stderr
        assert "## Focus Areas" in result.stdout
        assert "look at concurrency" in result.stdout

    def test_mixed_instructions_keep_their_command_line_order(self, tmp_path):
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke(
            "plan", "run", str(plan),
            "-i", "first plain",
            "--ci", "second headed", "Second Heading",
            "-i", "third plain",
        )

        assert result.exit_code == 0, result.stderr
        text = result.stdout
        assert (
            text.index("first plain")
            < text.index("## Second Heading")
            < text.index("second headed")
            < text.index("third plain")
        )

    @pytest.mark.parametrize(
        "arguments",
        [
            ["--ci=headed text", "The Heading", "-i", "plain text"],
            ["--ci", "headed text", "The Heading", "-iplain text"],
            ["--ci", "headed text", "The Heading", "-i", "plain text",
             "--plan-arg", "--ci"],
        ],
        ids=["equals-form", "attached-short-form", "flag-like-option-value"],
    )
    def test_click_option_forms_preserve_instruction_order(self, tmp_path, arguments):
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke("plan", "run", str(plan), *arguments)

        assert result.exit_code == 0, result.stderr
        assert result.stdout.index("headed text") < result.stdout.index("plain text")

    def test_labelled_context_files_become_fenced_sections(self, tmp_path):
        plan = write_plan(tmp_path, [cli_stage()])
        v1 = tmp_path / "v1.md"
        v1.write_text("first version", encoding="utf-8")

        result = invoke(
            "plan", "run", str(plan), "compare", "--cf", str(v1), "Version 1"
        )

        assert result.exit_code == 0, result.stderr
        assert "## Version 1" in result.stdout
        assert "```\nfirst version\n```" in result.stdout
        assert "compare" in result.stdout

    def test_model_option_overrides_all_stages(self, tmp_path):
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke("plan", "run", str(plan), "hi", "-m", "other")

        assert result.stdout == "OTHER[hi]\n"

    def test_model_env_var_is_used_when_m_is_absent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "other")
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke("plan", "run", str(plan), "hi")

        assert result.stdout == "OTHER[hi]\n"

    def test_m_option_beats_the_model_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "other")
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke("plan", "run", str(plan), "hi", "-m", "echo")

        assert result.stdout == "ECHO[hi]\n"

    def test_multiple_leaves_get_stage_headers(self, tmp_path):
        plan = write_plan(tmp_path, [cli_stage("a"), cli_stage("b")])

        result = invoke("plan", "run", str(plan), "hi")

        assert "## a" in result.stdout
        assert "## b" in result.stdout

    def test_failed_leaf_exits_non_zero_but_successful_leaves_still_print(
        self, tmp_path, fake_models
    ):
        fake_models.flaky.failures_left = 99
        plan = write_plan(
            tmp_path,
            [cli_stage("bad", model="flaky"), cli_stage("good")],
            parallel_config={"max_workers": 2},
        )

        result = invoke("plan", "run", str(plan), "hi", "--retries", "0")

        assert result.exit_code != 0
        assert "ECHO[hi]" in result.stdout
        assert "bad" in result.stderr

    def test_sequential_failure_aborts_with_a_clean_error(self, tmp_path, fake_models):
        fake_models.flaky.failures_left = 99
        plan = write_plan(
            tmp_path, [cli_stage("bad", model="flaky"), cli_stage("good")]
        )

        result = invoke("plan", "run", str(plan), "hi", "--retries", "0")

        assert result.exit_code != 0
        assert "bad" in result.stderr
        assert result.stdout == ""

    def test_progress_goes_to_stderr_and_quiet_silences_it(self, tmp_path):
        plan = write_plan(tmp_path, [cli_stage()])

        noisy = invoke("plan", "run", str(plan), "hi")
        quiet = invoke("plan", "run", str(plan), "hi", "--quiet")

        assert "solo" in noisy.stderr
        assert quiet.stderr == ""
        assert quiet.stdout == "ECHO[hi]\n"

    def test_quiet_has_no_short_flag(self, tmp_path):
        # Upstream llm reserves -q for model queries and log search, so the
        # plan commands must leave it free.
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke("plan", "run", str(plan), "hi", "-q")

        assert result.exit_code == 2
        assert "No such option" in result.stderr

    def test_missing_plan_is_a_clean_error(self):
        result = invoke("plan", "run", "no-such-plan")

        assert result.exit_code != 0
        assert "no-such-plan" in result.stderr

    def test_malformed_yaml_is_a_clean_error(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("stages: [unclosed", encoding="utf-8")

        result = invoke("plan", "run", str(bad), "hi")

        assert result.exit_code == 1
        assert "Error:" in result.stderr
        assert "Traceback" not in result.stderr

    def test_negative_retries_are_rejected(self, tmp_path):
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke("plan", "run", str(plan), "hi", "--retries", "-1")

        assert result.exit_code != 0
        assert "retries" in result.stderr.lower()


class TestAttachments:
    def test_attachment_from_stdin_reaches_cli_stages(
        self, tmp_path, fake_models, png_bytes
    ):
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke(
            "plan", "run", str(plan), "the question", "-a", "-", input=png_bytes
        )

        assert result.exit_code == 0, result.stderr
        (prompt,) = fake_models.echo.prompts
        (attachment,) = prompt.attachments
        assert attachment.content == png_bytes
        assert attachment.type == "image/png"
        assert prompt.prompt == "the question"

    def test_bad_attachment_url_fails_before_any_stage_runs(
        self, tmp_path, fake_models, monkeypatch
    ):
        import httpx

        def refuse_connection(url, **kwargs):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr("llm.cli.httpx.head", refuse_connection)
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke(
            "plan", "run", str(plan), "hi",
            "-a", "https://bad.example/x.png", "--retries", "0",
        )

        assert result.exit_code == 2
        assert fake_models.echo.prompts == []

    def test_missing_attachment_file_is_a_clean_usage_error(self, tmp_path):
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke("plan", "run", str(plan), "hi", "-a", "nope.png")

        assert result.exit_code == 2
        assert "does not exist" in result.stderr


class TestBundledPlans:
    def test_synthesis_full_has_the_expected_stage_structure(self):
        from llm_plan.runner import load_plan
        from llm_plan.store import resolve_plan

        plan = load_plan(resolve_plan("synthesis_full"))

        stages = {stage.name: stage for stage in plan.stages}
        assert [s.name for s in plan.stages] == [
            "opus", "sonnet", "gemini_pro", "gpt5", "synthesis",
        ]
        assert stages["gemini_pro"].produces == "Gemini Pro Analysis"
        assert all(
            stages[name].prompt_label == "Instructions"
            for name in ("opus", "sonnet", "gemini_pro", "gpt5")
        )
        assert stages["synthesis"].depends_on == ["opus", "sonnet", "gemini_pro", "gpt5"]
        assert stages["synthesis"].partial_dependencies is True
        assert stages["synthesis"].produces == "Final Synthesis"
        assert plan.max_workers == 7

    def test_the_bundled_synthesis_prompt_is_pinned(self):
        # Changing the bundled prompt changes what the synthesis_full alias
        # does for everyone, so any edit must be deliberate.
        import hashlib

        from llm_plan.store import BUNDLED_PLAN_DIR

        content = (BUNDLED_PLAN_DIR / "prompts" / "synthesise.md").read_bytes()

        assert hashlib.sha256(content).hexdigest() == PINNED_SYNTHESISE_SHA256

    def test_synthesis_full_runs_end_to_end_with_a_model_override(self, tmp_path):
        result = invoke("plan", "run", "synthesis_full", "the question", "-m", "echo")

        assert result.exit_code == 0, result.stderr
        assert result.stdout.startswith("ECHO[")
        assert "## Opus Analysis" in result.stdout
        assert "## GPT-5 Analysis" in result.stdout
        assert "Synthesis Instructions" in result.stdout
        assert "the question" in result.stdout

    def test_show_prints_the_bundled_yaml(self):
        result = invoke("plan", "show", "synthesis_full")

        assert result.exit_code == 0
        assert "name: \"synthesis_full\"" in result.stdout
        assert "plan_synthesis_full.yaml" in result.stderr


class TestPlanList:
    def test_lists_aliases_and_summaries_in_aligned_columns(self, user_dir):
        plans = user_dir / "plans"
        write_plan(plans, [cli_stage()], filename="plan_zz.yaml",
                   summary="the zz plan")
        write_plan(plans, [cli_stage()], filename="plan_a_much_longer_alias.yaml",
                   summary="the long plan")

        result = invoke("plan", "list")

        assert result.exit_code == 0, result.stderr
        lines = result.stdout.splitlines()
        assert "a_much_longer_alias  the long plan" in lines
        assert "zz                   the zz plan" in lines

    def test_json_output_has_alias_name_summary_and_path(self, user_dir):
        path = write_plan(user_dir / "plans", [cli_stage()],
                          filename="plan_review.yaml", summary="reviews things")

        result = invoke("plan", "list", "--json")

        assert result.exit_code == 0, result.stderr
        entries = {entry["alias"]: entry for entry in json.loads(result.stdout)}
        assert entries["review"] == {
            "alias": "review",
            "name": "test",
            "summary": "reviews things",
            "path": str(path.resolve()),
        }

    def test_no_plans_found_message_names_the_user_dir(
        self, user_dir, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            "llm_plan.store.BUNDLED_PLAN_DIR", tmp_path / "no-bundle"
        )

        result = invoke("plan", "list")

        assert result.exit_code == 0, result.stderr
        assert "No plans found" in result.stdout
        assert str(user_dir / "plans") in result.stdout


class TestScriptStages:
    def test_python_script_stage_runs_end_to_end(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AMBIENT_VAR", "ambient-value")
        script = tmp_path / "record.py"
        script.write_text(textwrap.dedent("""\
            import json, os, sys
            out = os.path.join(os.environ["LLM_PLAN_OUTPUT_DIR"], "record.json")
            with open(out, "w") as f:
                json.dump({
                    "argv": sys.argv[1:],
                    "stage_env": os.environ.get("CUSTOM_VAR"),
                    "inherited_env": os.environ.get("AMBIENT_VAR"),
                }, f)
            print(out)
        """), encoding="utf-8")
        plan = write_plan(tmp_path, [{
            "name": "worker", "summary": "records its inputs",
            "type": "python_script", "script": "record.py",
            "env": {"CUSTOM_VAR": "custom-value"},
        }])

        result = invoke(
            "plan", "run", str(plan), "--plan-arg", "--pr", "--plan-arg", "123"
        )

        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout) == {
            "argv": ["--pr", "123"],
            "stage_env": "custom-value",
            "inherited_env": "ambient-value",
        }


class TestPlanShow:
    @not_as_root
    def test_a_permission_denied_plan_file_is_a_clean_error(self, tmp_path):
        plan = write_plan(tmp_path, [cli_stage()])
        plan.chmod(0)

        result = invoke("plan", "show", str(plan))

        plan.chmod(0o644)
        assert result.exit_code == 1
        assert "Error:" in result.stderr
        assert "Traceback" not in result.stderr

    def test_a_non_utf8_plan_file_is_a_clean_error(self, tmp_path):
        binary = tmp_path / "binary.yaml"
        binary.write_bytes(b"\xff\xfe not utf-8")

        result = invoke("plan", "show", str(binary))

        assert result.exit_code == 1
        assert "Error:" in result.stderr
        assert "Traceback" not in result.stderr


class TestExplain:
    def test_explain_prints_the_dag_without_executing(self, tmp_path, fake_models):
        stages = [
            cli_stage("analyst", produces="Analysis"),
            {"name": "synthesis", "summary": "combine", "model": "other",
             "depends_on": ["analyst"], "prompt": "inline:Combine."},
        ]
        plan = write_plan(tmp_path, stages)

        result = invoke("plan", "run", str(plan), "--explain")

        assert result.exit_code == 0, result.stderr
        assert "1. analyst" in result.stdout
        assert "2. synthesis" in result.stdout
        assert "analyst (Analysis)" in result.stdout
        assert fake_models.echo.prompts == []


class TestLogging:
    def test_responses_are_logged_to_llms_database(self, tmp_path, logs_db):
        plan = write_plan(tmp_path, [cli_stage("a"), cli_stage("b")])

        result = invoke("plan", "run", str(plan), "hi")

        assert result.exit_code == 0, result.stderr
        rows = list(logs_db["responses"].rows)
        assert len(rows) == 2
        assert {row["model"] for row in rows} == {"echo"}
        assert rows[0]["response"] == "ECHO[hi]"

    def test_stderr_maps_each_stage_to_its_logged_response_id(self, tmp_path, logs_db):
        plan = write_plan(tmp_path, [cli_stage("a"), cli_stage("b")])

        result = invoke("plan", "run", str(plan), "hi")

        id_a, id_b = [
            row[0] for row in logs_db.execute("select id from responses order by rowid")
        ]
        assert f"[a] response {id_a}" in result.stderr
        assert f"[b] response {id_b}" in result.stderr
        assert "llm logs -n 2" in result.stderr

    def test_tracking_lines_carry_the_plan_run_id(self, tmp_path):
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke("plan", "run", str(plan), "hi")

        run_lines = [line for line in result.stderr.splitlines() if "Run " in line]
        assert run_lines, result.stderr
        run_id = run_lines[0].split("Run ")[1].split(":")[0]
        assert len(run_id) == 12

    def test_no_log_prints_no_tracking_lines(self, tmp_path):
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke("plan", "run", str(plan), "hi", "-n")

        assert result.exit_code == 0, result.stderr
        assert "response" not in result.stderr
        assert "llm logs" not in result.stderr

    def test_no_log_skips_the_database(self, tmp_path, logs_db):
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke("plan", "run", str(plan), "hi", "-n")

        assert result.exit_code == 0, result.stderr
        assert "responses" not in logs_db.table_names() or logs_db["responses"].count == 0

    def test_logs_off_sentinel_is_respected(self, tmp_path, user_dir, logs_db):
        (user_dir / "logs-off").touch()
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke("plan", "run", str(plan), "hi")

        assert result.exit_code == 0, result.stderr
        assert "responses" not in logs_db.table_names() or logs_db["responses"].count == 0

    def test_log_and_no_log_together_are_rejected(self, tmp_path, fake_models):
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke("plan", "run", str(plan), "hi", "--log", "--no-log")

        assert result.exit_code == 1
        assert "--log and --no-log are mutually exclusive" in result.stderr
        assert fake_models.echo.prompts == []

    def test_log_flag_overrides_the_sentinel(self, tmp_path, user_dir, logs_db):
        (user_dir / "logs-off").touch()
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke("plan", "run", str(plan), "hi", "--log")

        assert result.exit_code == 0, result.stderr
        assert logs_db["responses"].count == 1
