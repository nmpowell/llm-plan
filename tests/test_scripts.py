import json
import textwrap

import yaml

from llm_plan.models import PlanError
from llm_plan.runner import CLIContext, PlanRunner, load_plan

WRITER_SCRIPT = """
import json, os
out_dir = os.environ["LLM_PLAN_OUTPUT_DIR"]
first = os.path.join(out_dir, "one.md")
second = os.path.join(out_dir, "two.md")
open(first, "w").write("first file body")
open(second, "w").write("second file body")
print(json.dumps({"outputs": [
    {"path": first, "label": "Fetched Data"},
    {"path": second},
]}))
"""

RECORDER_SCRIPT = """
import json, os, sys
out = os.path.join(os.environ["LLM_PLAN_OUTPUT_DIR"], "record.json")
record = {
    "argv": sys.argv[1:],
    "instructions": os.environ.get("LLM_PLAN_INSTRUCTIONS"),
    "stage": os.environ.get("LLM_PLAN_STAGE_NAME"),
    "run_id": os.environ.get("LLM_PLAN_RUN_ID"),
    "custom": os.environ.get("CUSTOM_VAR"),
    "inputs": [open(path).read() for path in sys.argv[1:] if os.path.exists(path)],
}
open(out, "w").write(json.dumps(record))
print(out)
"""


def write_script(tmp_path, body, name="script.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def run_plan(tmp_path, stages, cli=None):
    """Run a plan; sequential-abort errors are tolerated so results stay inspectable."""
    data = {"name": "test", "summary": "script test plan", "stages": stages}
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(yaml.safe_dump(data), encoding="utf-8")
    runner = PlanRunner(load_plan(plan_file), cli or CLIContext(), retry_delay=0)
    try:
        runner.run()
    except PlanError:
        pass
    return runner, dict(runner.results)


def script_stage(tmp_path, body, name="worker", script_name="script.py", **kwargs):
    write_script(tmp_path, body, script_name)
    return {"name": name, "summary": f"{name} stage", "type": "python_script",
            "script": script_name, **kwargs}


class TestScriptExecution:
    def test_manifest_outputs_are_recorded_with_first_file_as_text(self, tmp_path):
        runner, results = run_plan(tmp_path, [script_stage(tmp_path, WRITER_SCRIPT)])

        result = results["worker"]
        assert result.success
        assert result.text == "first file body"
        assert [p.name for p in result.files] == ["one.md", "two.md"]
        assert result.manifest["outputs"][0]["label"] == "Fetched Data"

    def test_bare_path_lines_work_without_a_manifest(self, tmp_path):
        body = """
        import os
        out = os.path.join(os.environ["LLM_PLAN_OUTPUT_DIR"], "plain.md")
        open(out, "w").write("plain output")
        print(out)
        """
        runner, results = run_plan(tmp_path, [script_stage(tmp_path, body)])

        assert results["worker"].success
        assert results["worker"].text == "plain output"

    def test_declared_but_missing_output_fails_the_stage(self, tmp_path):
        body = 'print("/nonexistent/path/file.md")'
        runner, results = run_plan(tmp_path, [script_stage(tmp_path, body)])

        assert not results["worker"].success
        assert "file.md" in results["worker"].error

    def test_no_output_fails_the_stage(self, tmp_path):
        runner, results = run_plan(tmp_path, [script_stage(tmp_path, 'print()')])

        assert not results["worker"].success
        assert "no output" in results["worker"].error.lower()

    def test_nonzero_exit_fails_the_stage_with_stderr(self, tmp_path):
        body = """
        import sys
        print("diagnostic detail", file=sys.stderr)
        sys.exit(3)
        """
        runner, results = run_plan(tmp_path, [script_stage(tmp_path, body)])

        assert not results["worker"].success
        assert "3" in results["worker"].error
        assert "diagnostic detail" in results["worker"].error

    def test_timeout_fails_the_stage(self, tmp_path):
        body = "import time; time.sleep(10)"
        stage = script_stage(tmp_path, body, timeout=1)

        runner, results = run_plan(tmp_path, [stage])

        assert not results["worker"].success
        assert "timeout" in results["worker"].error.lower()


class TestScriptInputs:
    def test_env_plan_args_and_runtime_vars_reach_the_script(self, tmp_path):
        stage = script_stage(
            tmp_path,
            RECORDER_SCRIPT,
            script_args=["--stage=${run.stage_name}", "--fixed"],
            env={"CUSTOM_VAR": "custom-value"},
        )
        cli = CLIContext(instructions="the question", plan_args=["--pr", "123"])

        runner, results = run_plan(tmp_path, [stage], cli)

        record = json.loads(results["worker"].text)
        assert record["argv"] == ["--pr", "123", "--stage=worker", "--fixed"]
        assert record["instructions"] == "the question"
        assert record["stage"] == "worker"
        assert record["run_id"] == runner.run_id
        assert record["custom"] == "custom-value"

    def test_llm_dependency_text_arrives_as_a_file_argument(self, tmp_path):
        stages = [
            {"name": "thinker", "summary": "s", "model": "echo", "prompt": "CLI"},
            script_stage(tmp_path, RECORDER_SCRIPT, name="worker",
                         depends_on=["thinker"]),
        ]

        runner, results = run_plan(tmp_path, stages, CLIContext(instructions="hi"))

        record = json.loads(results["worker"].text)
        assert record["inputs"] == ["ECHO[hi]"]
        assert record["argv"][0].endswith("thinker.md")

    def test_script_files_flow_to_dependent_scripts(self, tmp_path):
        stages = [
            script_stage(tmp_path, WRITER_SCRIPT, name="producer",
                         script_name="producer.py"),
            script_stage(tmp_path, RECORDER_SCRIPT, name="consumer",
                         script_name="consumer.py", depends_on=["producer"]),
        ]

        runner, results = run_plan(tmp_path, stages)

        record = json.loads(results["consumer"].text)
        assert record["inputs"] == ["first file body", "second file body"]

    def test_stage_files_are_passed_after_dependency_outputs(self, tmp_path):
        (tmp_path / "reference.md").write_text("reference body", encoding="utf-8")
        stages = [
            {"name": "thinker", "summary": "s", "model": "echo", "prompt": "CLI"},
            script_stage(tmp_path, RECORDER_SCRIPT, name="worker",
                         depends_on=["thinker"],
                         files=[{"path": "reference.md", "label": "Reference"}]),
        ]

        runner, results = run_plan(tmp_path, stages, CLIContext(instructions="hi"))

        record = json.loads(results["worker"].text)
        assert record["inputs"] == ["ECHO[hi]", "reference body"]

    def test_first_stage_script_receives_seed_file_fragments(self, tmp_path):
        seed = tmp_path / "seed.md"
        seed.write_text("seed body", encoding="utf-8")
        fragment = type("Fragment", (str,), {"source": str(seed)})("seed body")
        stage = script_stage(tmp_path, RECORDER_SCRIPT)

        runner, results = run_plan(tmp_path, [stage], CLIContext(fragments=[fragment]))

        record = json.loads(results["worker"].text)
        assert record["argv"] == [str(seed)]
        assert record["inputs"] == ["seed body"]

    def test_seed_dependent_script_gets_pathless_fragments_materialised(self, tmp_path):
        stages = [
            {"name": "first", "summary": "s", "model": "echo", "prompt": "CLI"},
            script_stage(tmp_path, RECORDER_SCRIPT, name="worker",
                         depends_on=["seed"]),
        ]
        cli = CLIContext(instructions="hi", fragments=["fragment text with no path"])

        runner, results = run_plan(tmp_path, stages, cli)

        record = json.loads(results["worker"].text)
        assert record["inputs"] == ["fragment text with no path"]

    def test_unknown_runtime_variable_in_plan_args_fails_the_stage(self, tmp_path):
        stage = script_stage(tmp_path, RECORDER_SCRIPT)
        cli = CLIContext(plan_args=["${cli.bogus}"])

        runner, results = run_plan(tmp_path, [stage], cli)

        assert not results["worker"].success
        assert "cli.bogus" in results["worker"].error


class TestScriptChainingIntoLlm:
    def test_llm_stage_receives_labelled_sections_per_produced_file(self, tmp_path):
        stages = [
            script_stage(tmp_path, WRITER_SCRIPT, name="fetch", produces="Fetched"),
            {"name": "review", "summary": "s", "model": "echo",
             "depends_on": ["fetch"], "prompt": "inline:Review the data."},
        ]

        runner, results = run_plan(tmp_path, stages)

        text = results["review"].text
        assert "## Fetched Data" in text        # manifest label wins
        assert "## Fetched (2/2)" in text       # fallback label with index
        assert "first file body" in text and "second file body" in text

    def test_chain_prompt_from_a_script_uses_its_first_file(self, tmp_path):
        stages = [
            script_stage(tmp_path, WRITER_SCRIPT, name="fetch"),
            {"name": "review", "summary": "s", "model": "echo",
             "depends_on": ["fetch"],
             "prompt": [{"prompt": "chain:fetch", "label": "Data"}]},
        ]

        runner, results = run_plan(tmp_path, stages)

        assert "## Data" in results["review"].text
        assert "first file body" in results["review"].text
