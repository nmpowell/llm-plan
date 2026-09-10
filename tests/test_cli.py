import gc
import json
import os
import textwrap

import pytest
import sqlite_utils
from click.testing import CliRunner
from llm.cli import cli
from conftest import write_plan

not_as_root = pytest.mark.skipif(
    os.geteuid() == 0, reason="root ignores file permissions"
)


def cli_stage(name="solo", model="echo", **kwargs):
    return {
        "name": name,
        "summary": f"{name} stage",
        "model": model,
        "prompt": "CLI",
        **kwargs,
    }


PINNED_SYNTHESISE_SHA256 = (
    "61425345ac41aae86be956a3910dc9ba9d6c81b5c05a6c7b7b7d796b13540f5b"
)

PINNED_EXPAND_RESEARCH_SHA256 = (
    "78523a9c605c8844876bc3db47dc56fbedcedbadf539c9c790b1c00762e22bc9"
)


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


def logged_responses(logs_db):
    """(response id, run id, model) per logged response, oldest first.

    Read from whichever table this llm version writes: ``turns`` (llm >= 0.32)
    or the legacy ``responses`` table (llm 0.31).
    """
    for table, run_column in (("turns", "thread_id"), ("responses", "conversation_id")):
        if table in logs_db.table_names() and logs_db[table].count:
            return [
                (row[0], row[1], row[2])
                for row in logs_db.execute(
                    f"select id, {run_column}, model from {table} order by rowid"
                )
            ]
    return []


def logged_run(logs_db, run_id):
    """The run's own row: ``threads`` (llm >= 0.32) or legacy ``conversations``."""
    for table in ("threads", "conversations"):
        if table in logs_db.table_names():
            rows = list(logs_db[table].rows_where("id = ?", [run_id]))
            if rows:
                return rows[0]
    raise AssertionError(f"no logged thread or conversation for run {run_id}")


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

        result = invoke(
            "plan", "run", str(plan), "and the argument", input="from stdin"
        )

        assert result.stdout == "ECHO[from stdin and the argument]\n"

    def test_piped_stdin_keeps_its_whitespace(self, tmp_path):
        # llm prompt does not strip stdin, so indented code survives piping
        # and the argument follows it after a single space.
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke(
            "plan",
            "run",
            str(plan),
            "explain this",
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
            "plan",
            "run",
            str(plan),
            "and the argument",
            "-i",
            "first instruction",
            "-i",
            "second instruction",
        )

        assert result.exit_code == 0, result.stderr
        assert result.stdout == (
            "ECHO[first instruction\n\nsecond instruction\n\nand the argument]\n"
        )

    def test_headed_instructions_via_ci_flag(self, tmp_path):
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke(
            "plan",
            "run",
            str(plan),
            "the question",
            "--ci",
            "look at concurrency",
            "Focus Areas",
        )

        assert result.exit_code == 0, result.stderr
        assert "## Focus Areas" in result.stdout
        assert "look at concurrency" in result.stdout

    def test_mixed_instructions_keep_their_command_line_order(self, tmp_path):
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke(
            "plan",
            "run",
            str(plan),
            "-i",
            "first plain",
            "--ci",
            "second headed",
            "Second Heading",
            "-i",
            "third plain",
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
            [
                "--ci",
                "headed text",
                "The Heading",
                "-i",
                "plain text",
                "--plan-arg",
                "--ci",
            ],
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
        import llm.cli

        http = getattr(llm.cli, "httpx2", None) or llm.cli.httpx

        def refuse_connection(url, **kwargs):
            raise http.ConnectError("connection refused")

        monkeypatch.setattr(http, "head", refuse_connection)
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke(
            "plan",
            "run",
            str(plan),
            "hi",
            "-a",
            "https://bad.example/x.png",
            "--retries",
            "0",
        )

        assert result.exit_code == 2
        assert fake_models.echo.prompts == []

    def test_missing_attachment_file_is_a_clean_usage_error(self, tmp_path):
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke("plan", "run", str(plan), "hi", "-a", "nope.png")

        assert result.exit_code == 2
        assert "does not exist" in result.stderr


class TestBundledPlans:
    def test_synthesis_has_the_expected_stage_structure(self):
        from llm_plan.runner import load_plan
        from llm_plan.store import resolve_plan

        plan = load_plan(resolve_plan("synthesis"))

        stages = {stage.name: stage for stage in plan.stages}
        assert [s.name for s in plan.stages] == [
            "claude_high",
            "claude_medium",
            "openai_high",
            "openai_medium",
            "gemini_high",
            "gemini_low",
            "synthesis",
        ]
        assert stages["gemini_high"].produces == "Gemini High Analysis"
        assert all(
            stages[name].prompt_label == "Instructions"
            for name in (
                "claude_high",
                "claude_medium",
                "openai_high",
                "openai_medium",
                "gemini_high",
                "gemini_low",
            )
        )
        assert stages["synthesis"].depends_on == [
            "claude_high",
            "claude_medium",
            "openai_high",
            "openai_medium",
            "gemini_high",
            "gemini_low",
        ]
        assert stages["synthesis"].partial_dependencies is True
        assert stages["synthesis"].produces == "Final Synthesis"
        assert plan.max_workers == 7

    def test_the_bundled_synthesis_prompt_is_pinned(self):
        # Changing the bundled prompt changes what the synthesis alias
        # does for everyone, so any edit must be deliberate.
        import hashlib

        from llm_plan.store import BUNDLED_PLAN_DIR

        content = (BUNDLED_PLAN_DIR / "prompts" / "synthesise.md").read_bytes()

        assert hashlib.sha256(content).hexdigest() == PINNED_SYNTHESISE_SHA256

    def test_synthesis_runs_end_to_end_with_a_model_override(self, tmp_path):
        result = invoke("plan", "run", "synthesis", "the question", "-m", "echo")

        assert result.exit_code == 0, result.stderr
        assert result.stdout.startswith("ECHO[")
        assert "## Claude High Analysis" in result.stdout
        assert "## OpenAI High Analysis" in result.stdout
        assert "Synthesis Instructions" in result.stdout
        assert "the question" in result.stdout

    def test_show_prints_the_bundled_yaml(self):
        result = invoke("plan", "show", "synthesis")

        assert result.exit_code == 0
        assert 'name: "synthesis"' in result.stdout
        assert "plan_synthesis.yaml" in result.stderr

    def test_deep_research_has_the_expected_stage_structure(self):
        from llm_plan.runner import load_plan
        from llm_plan.store import resolve_plan

        plan = load_plan(resolve_plan("deep_research"))

        assert [s.name for s in plan.stages] == ["expand_prompt", "research"]
        expand, research = plan.stages
        assert expand.type == "llm"
        assert expand.produces == "Expanded Deep Research Prompt"
        assert research.type == "python_script"
        assert research.depends_on == ["expand_prompt"]
        assert research.produces == "Gemini Deep Research Report"
        assert research.resolved_script is not None
        assert research.resolved_script.is_file()
        # The stage timeout must outlast the script's own --max-wait deadline
        # (60 minutes, the API maximum) so the script exits with a clean error
        # instead of being process-group-killed mid-poll.
        assert research.timeout is not None
        assert research.timeout > 3600
        assert "--max-wait" in research.script_args
        max_wait = research.script_args[research.script_args.index("--max-wait") + 1]
        assert float(max_wait) * 60 < research.timeout

    def test_the_bundled_expand_research_prompt_is_pinned(self):
        # Same rule as the synthesis prompt: editing the bundled prompt
        # changes what the deep_research alias does for everyone, so any
        # edit must be deliberate.
        import hashlib

        from llm_plan.store import BUNDLED_PLAN_DIR

        content = (BUNDLED_PLAN_DIR / "prompts" / "expand_research.md").read_bytes()

        assert hashlib.sha256(content).hexdigest() == PINNED_EXPAND_RESEARCH_SHA256

    def test_deep_research_explain_previews_the_dag_without_executing(self):
        result = invoke(
            "plan", "run", "deep_research", "a question", "--explain", "-m", "echo"
        )

        assert result.exit_code == 0, result.output
        assert "expand_prompt" in result.stdout
        assert "deep_research.py" in result.stdout

    def test_deep_research_script_fails_fast_without_an_api_key(self, tmp_path):
        # Hermetic end-to-end wiring check: the expand stage runs on the fake
        # echo model, then the real bundled script starts and must fail with
        # a clear message before touching the network (the user_dir fixture
        # guarantees GEMINI_API_KEY is unset).
        result = invoke("plan", "run", "deep_research", "a question", "-m", "echo")

        assert result.exit_code == 1
        assert "GEMINI_API_KEY" in result.stderr


class TestPlanList:
    def test_lists_aliases_and_summaries_in_aligned_columns(self, user_dir):
        plans = user_dir / "plans"
        write_plan(plans, [cli_stage()], filename="plan_zz.yaml", summary="the zz plan")
        write_plan(
            plans,
            [cli_stage()],
            filename="plan_a_much_longer_alias.yaml",
            summary="the long plan",
        )

        result = invoke("plan", "list")

        assert result.exit_code == 0, result.stderr
        lines = result.stdout.splitlines()
        assert "a_much_longer_alias  the long plan" in lines
        assert "zz                   the zz plan" in lines

    def test_json_output_has_alias_name_summary_and_path(self, user_dir):
        path = write_plan(
            user_dir / "plans",
            [cli_stage()],
            filename="plan_review.yaml",
            summary="reviews things",
        )

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
        monkeypatch.setattr("llm_plan.store.BUNDLED_PLAN_DIR", tmp_path / "no-bundle")

        result = invoke("plan", "list")

        assert result.exit_code == 0, result.stderr
        assert "No plans found" in result.stdout
        assert str(user_dir / "plans") in result.stdout


class TestScriptStages:
    def test_python_script_stage_runs_end_to_end(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AMBIENT_VAR", "ambient-value")
        script = tmp_path / "record.py"
        script.write_text(
            textwrap.dedent("""\
            import json, os, sys
            out = os.path.join(os.environ["LLM_PLAN_OUTPUT_DIR"], "record.json")
            with open(out, "w") as f:
                json.dump({
                    "argv": sys.argv[1:],
                    "stage_env": os.environ.get("CUSTOM_VAR"),
                    "inherited_env": os.environ.get("AMBIENT_VAR"),
                }, f)
            print(out)
        """),
            encoding="utf-8",
        )
        plan = write_plan(
            tmp_path,
            [
                {
                    "name": "worker",
                    "summary": "records its inputs",
                    "type": "python_script",
                    "script": "record.py",
                    "env": {"CUSTOM_VAR": "custom-value"},
                }
            ],
        )

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
            {
                "name": "synthesis",
                "summary": "combine",
                "model": "other",
                "depends_on": ["analyst"],
                "prompt": "inline:Combine.",
            },
        ]
        plan = write_plan(tmp_path, stages)

        result = invoke("plan", "run", str(plan), "--explain")

        assert result.exit_code == 0, result.stderr
        assert "1. analyst" in result.stdout
        assert "2. synthesis" in result.stdout
        assert "analyst (Analysis)" in result.stdout
        assert fake_models.echo.prompts == []


class TestLogging:
    # llm's own `logs` command opens its database without closing it. That
    # ResourceWarning is upstream's, so it is ignored for this test only, and
    # gc.collect() surfaces it here rather than in whichever test runs next.
    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    def test_responses_are_logged_to_llms_database(self, tmp_path, logs_db):
        plan = write_plan(tmp_path, [cli_stage("a"), cli_stage("b")])

        result = invoke("plan", "run", str(plan), "hi")

        assert result.exit_code == 0, result.stderr
        run_id = result.stderr.split("Run ")[1].split(":")[0]
        logged = logged_responses(logs_db)
        assert len(logged) == 2
        assert {model for _, _, model in logged} == {"echo"}
        logs = invoke("logs", "--cid", run_id)
        gc.collect()
        assert logs.exit_code == 0
        assert "ECHO[hi]" in logs.output
        assert "echo" in logs.output

    def test_stderr_maps_each_stage_to_its_logged_response_id(self, tmp_path, logs_db):
        plan = write_plan(tmp_path, [cli_stage("a"), cli_stage("b")])

        result = invoke("plan", "run", str(plan), "hi")

        (id_a, _, _), (id_b, _, _) = logged_responses(logs_db)
        assert f"[a] response {id_a}" in result.stderr
        assert f"[b] response {id_b}" in result.stderr
        assert "llm logs --cid" in result.stderr

    def test_tracking_lines_carry_the_plan_run_id(self, tmp_path):
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke("plan", "run", str(plan), "hi")

        run_lines = [line for line in result.stderr.splitlines() if "Run " in line]
        assert run_lines, result.stderr
        run_id = run_lines[0].split("Run ")[1].split(":")[0]
        assert len(run_id) == 32

    def test_a_runs_responses_share_one_conversation_named_for_the_plan(
        self, tmp_path, logs_db
    ):
        plan = write_plan(tmp_path, [cli_stage("a"), cli_stage("b")])

        result = invoke("plan", "run", str(plan), "hi")

        run_id = result.stderr.split("Run ")[1].split(":")[0]
        assert logged_run(logs_db, run_id)["name"] == "test"
        assert {rid for _, rid, _ in logged_responses(logs_db)} == {run_id}

    def test_a_logging_failure_fails_the_run_but_keeps_the_output(
        self, tmp_path, monkeypatch
    ):
        import llm_plan.cli

        def broken_log(db, response, conversation):
            raise RuntimeError("disk full")

        monkeypatch.setattr(llm_plan.cli, "log_response", broken_log)

        plan = write_plan(tmp_path, [cli_stage()])
        result = invoke("plan", "run", str(plan), "hi", "--quiet")

        assert result.exit_code != 0
        assert "Could not log response for stage 'solo'" in result.stderr
        assert "disk full" in result.stderr
        assert "ECHO[hi]" in result.output

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
        assert logged_responses(logs_db) == []

    def test_logs_off_sentinel_is_respected(self, tmp_path, user_dir, logs_db):
        (user_dir / "logs-off").touch()
        plan = write_plan(tmp_path, [cli_stage()])

        result = invoke("plan", "run", str(plan), "hi")

        assert result.exit_code == 0, result.stderr
        assert logged_responses(logs_db) == []

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
        assert len(logged_responses(logs_db)) == 1
