import sqlite_utils
import yaml
from click.testing import CliRunner
from llm.cli import cli


def write_plan(directory, stages, filename="plan.yaml", **top_level):
    data = {"name": "test", "summary": "a test plan", "stages": stages, **top_level}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def cli_stage(name="solo", model="echo", **kwargs):
    return {"name": name, "summary": f"{name} stage", "model": model, "prompt": "CLI", **kwargs}


def invoke(*args, **kwargs):
    return CliRunner().invoke(cli, list(args), **kwargs)


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
        quiet = invoke("plan", "run", str(plan), "hi", "-q")

        assert "solo" in noisy.stderr
        assert quiet.stderr == ""
        assert quiet.stdout == "ECHO[hi]\n"

    def test_missing_plan_is_a_clean_error(self):
        result = invoke("plan", "run", "no-such-plan")

        assert result.exit_code != 0
        assert "no-such-plan" in result.stderr


class TestBundledPlans:
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
    def test_responses_are_logged_to_llms_database(self, tmp_path, user_dir):
        plan = write_plan(tmp_path, [cli_stage("a"), cli_stage("b")])

        result = invoke("plan", "run", str(plan), "hi")

        assert result.exit_code == 0, result.stderr
        db = sqlite_utils.Database(str(user_dir / "logs.db"))
        rows = list(db["responses"].rows)
        assert len(rows) == 2
        assert {row["model"] for row in rows} == {"echo"}
        assert rows[0]["response"] == "ECHO[hi]"

    def test_no_log_skips_the_database(self, tmp_path, user_dir):
        plan = write_plan(tmp_path, [cli_stage()])

        invoke("plan", "run", str(plan), "hi", "-n")

        db = sqlite_utils.Database(str(user_dir / "logs.db"))
        assert "responses" not in db.table_names() or db["responses"].count == 0

    def test_logs_off_sentinel_is_respected(self, tmp_path, user_dir):
        (user_dir / "logs-off").touch()
        plan = write_plan(tmp_path, [cli_stage()])

        invoke("plan", "run", str(plan), "hi")

        db = sqlite_utils.Database(str(user_dir / "logs.db"))
        assert "responses" not in db.table_names() or db["responses"].count == 0

    def test_log_flag_overrides_the_sentinel(self, tmp_path, user_dir):
        (user_dir / "logs-off").touch()
        plan = write_plan(tmp_path, [cli_stage()])

        invoke("plan", "run", str(plan), "hi", "--log")

        db = sqlite_utils.Database(str(user_dir / "logs.db"))
        assert db["responses"].count == 1
