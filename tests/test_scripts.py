import json
import os
import textwrap
import time

import llm
import pytest
import yaml

from llm_plan.runner import CLIContext

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


def script_stage(tmp_path, body, name="worker", script_name="script.py", **kwargs):
    write_script(tmp_path, body, script_name)
    return {"name": name, "summary": f"{name} stage", "type": "python_script",
            "script": script_name, **kwargs}


class TestScriptExecution:
    def test_manifest_outputs_are_recorded_with_first_file_as_text(self, tmp_path, run_plan):
        runner, results = run_plan([script_stage(tmp_path, WRITER_SCRIPT)])

        result = results["worker"]
        assert result.success
        assert result.text == "first file body"
        assert [p.name for p in result.files] == ["one.md", "two.md"]
        assert result.manifest["outputs"][0]["label"] == "Fetched Data"

    def test_bare_path_lines_work_without_a_manifest(self, tmp_path, run_plan):
        body = """
        import os
        out = os.path.join(os.environ["LLM_PLAN_OUTPUT_DIR"], "plain.md")
        open(out, "w").write("plain output")
        print(out)
        """
        runner, results = run_plan([script_stage(tmp_path, body)])

        assert results["worker"].success
        assert results["worker"].text == "plain output"

    def test_declared_but_missing_output_fails_the_stage(self, tmp_path, run_plan):
        body = 'print("/nonexistent/path/file.md")'
        runner, results = run_plan([script_stage(tmp_path, body)], expect_error=True)

        assert not results["worker"].success
        assert "file.md" in results["worker"].error

    def test_no_output_fails_the_stage(self, tmp_path, run_plan):
        runner, results = run_plan([script_stage(tmp_path, 'print()')], expect_error=True)

        assert not results["worker"].success
        assert "no output" in results["worker"].error.lower()

    def test_nonzero_exit_fails_the_stage_with_stderr(self, tmp_path, run_plan):
        body = """
        import sys
        print("diagnostic detail", file=sys.stderr)
        sys.exit(3)
        """
        runner, results = run_plan([script_stage(tmp_path, body)], expect_error=True)

        assert not results["worker"].success
        assert "3" in results["worker"].error
        assert "diagnostic detail" in results["worker"].error

    def test_timeout_fails_the_stage(self, tmp_path, run_plan):
        body = "import time; time.sleep(10)"
        stage = script_stage(tmp_path, body, timeout=1)

        runner, results = run_plan([stage], expect_error=True)

        assert not results["worker"].success
        assert "timeout" in results["worker"].error.lower()

    def test_timeout_kills_the_scripts_whole_process_group_promptly(self, tmp_path, run_plan):
        # The grandchild inherits the stdout pipe: unless the whole group is
        # killed it outlives the stage and can block the runner's pipe reads.
        body = """
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(8)"])
        open(sys.argv[1], "w").write(str(child.pid))
        time.sleep(8)
        """
        pid_file = tmp_path / "child.pid"
        stage = script_stage(tmp_path, body, timeout=1, script_args=[str(pid_file)])

        start = time.monotonic()
        runner, results = run_plan([stage], expect_error=True)
        elapsed = time.monotonic() - start

        assert "timeout" in results["worker"].error.lower()
        assert elapsed < 4, f"runner blocked for {elapsed:.1f}s after a 1s timeout"
        grandchild = int(pid_file.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(grandchild, 0)
            except ProcessLookupError:
                break
        else:
            pytest.fail(f"grandchild {grandchild} outlived the stage timeout")

    def test_timeout_error_includes_the_scripts_stderr_tail(self, tmp_path, run_plan):
        body = """
        import sys, time
        print("about to hang", file=sys.stderr, flush=True)
        time.sleep(8)
        """
        stage = script_stage(tmp_path, body, timeout=1)

        runner, results = run_plan([stage], expect_error=True)

        assert "timeout" in results["worker"].error.lower()
        assert "about to hang" in results["worker"].error

    def test_undecodable_stderr_still_reports_the_exit_code(self, tmp_path, run_plan):
        body = """
        import sys
        sys.stderr.buffer.write(b"diagnostic \\xff\\xfe tail\\n")
        sys.exit(3)
        """

        runner, results = run_plan([script_stage(tmp_path, body)], expect_error=True)

        assert "exited with code 3" in results["worker"].error
        assert "diagnostic" in results["worker"].error

    def test_non_ascii_output_paths_are_decoded_as_utf8(self, tmp_path, run_plan):
        body = """
        import os
        out = os.path.join(os.environ["LLM_PLAN_OUTPUT_DIR"], "r\\u00e9sum\\u00e9.md")
        open(out, "w", encoding="utf-8").write("accented filename body")
        print(out)
        """

        runner, results = run_plan([script_stage(tmp_path, body)])

        assert results["worker"].success, results["worker"].error
        assert results["worker"].text == "accented filename body"

    @pytest.mark.parametrize("stdout_line", [
        '{"outputs": null}',
        '{"outputs": {"path": "x"}}',
        '{"outputs": [123]}',
    ])
    def test_malformed_manifests_fail_the_stage_cleanly(self, tmp_path, stdout_line, run_plan):
        body = f"print('{stdout_line}')"

        runner, results = run_plan([script_stage(tmp_path, body)], expect_error=True)

        assert not results["worker"].success
        assert "manifest" in results["worker"].error.lower()

    def test_undecodable_output_file_fails_the_stage_cleanly(self, tmp_path, run_plan):
        body = """
        import os
        out = os.path.join(os.environ["LLM_PLAN_OUTPUT_DIR"], "binary.bin")
        open(out, "wb").write(b"\\xff\\xfe\\x00\\x01")
        print(out)
        """

        runner, results = run_plan([script_stage(tmp_path, body)], expect_error=True)

        assert not results["worker"].success
        assert "binary.bin" in results["worker"].error


class TestScriptSandbox:
    def test_scratch_directories_stay_inside_the_test_sandbox(self, tmp_path, run_plan):
        runner, results = run_plan([script_stage(tmp_path, WRITER_SCRIPT)])

        assert all(
            str(path).startswith(str(tmp_path)) for path in results["worker"].files
        ), results["worker"].files


# Writes its pid, signals readiness through a FIFO, then hangs.
HANG_AFTER_RENDEZVOUS = """
import os, sys, time
open(sys.argv[1], "w").write(str(os.getpid()))
open(sys.argv[2], "w").close()  # unblocks the test's FIFO reader
time.sleep(8)
"""


def assert_pid_dies(pid, within=2.0):
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.01)
    pytest.fail(f"script {pid} survived the abort")


class TestScriptAbort:
    def test_ctrl_c_kills_a_sequential_script(self, tmp_path, run_plan):
        import signal
        import threading

        pid_file = tmp_path / "script.pid"
        fifo = tmp_path / "ready.fifo"
        os.mkfifo(fifo)
        stage = script_stage(
            tmp_path, HANG_AFTER_RENDEZVOUS, script_args=[str(pid_file), str(fifo)]
        )

        def interrupt_when_ready():
            open(fifo).close()  # returns once the script opens its end
            os.kill(os.getpid(), signal.SIGINT)

        watcher = threading.Thread(target=interrupt_when_ready, daemon=True)
        watcher.start()
        with pytest.raises(KeyboardInterrupt):
            run_plan([stage])
        watcher.join(timeout=2)

        assert_pid_dies(int(pid_file.read_text(encoding="utf-8")))

    def test_a_script_spawned_during_an_abort_is_still_killed(
        self, tmp_path, monkeypatch
    ):
        # The abort can land between Popen() returning and the process being
        # registered; the registration must then kill it, not let it run on
        # for the stage timeout.
        import llm_plan.runner as runner_module
        from llm_plan.runner import CLIContext, PlanRunner, load_plan

        stage = script_stage(tmp_path, "import time; time.sleep(2)")
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            yaml.safe_dump({"name": "t", "summary": "s", "stages": [stage]}),
            encoding="utf-8",
        )
        runner = PlanRunner(load_plan(plan_file), CLIContext(), retry_delay=0)

        spawned = {}
        real_popen = runner_module.subprocess.Popen

        def popen_then_abort(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            spawned["process"] = process
            runner._abort_scripts()  # lands inside the registration window
            return process

        monkeypatch.setattr(runner_module.subprocess, "Popen", popen_then_abort)

        from llm_plan.models import PlanError

        with pytest.raises(PlanError):
            runner.run()

        assert not runner.results["worker"].success
        assert spawned["process"].poll() is not None, "script survived the abort"
        assert spawned["process"].poll() != 0, "script ran to completion"

    def test_an_aborting_parallel_run_kills_active_scripts(
        self, tmp_path, fake_models, run_plan
    ):
        import threading

        pid_file = tmp_path / "script.pid"
        fifo = tmp_path / "ready.fifo"
        os.mkfifo(fifo)
        stages = [
            script_stage(
                tmp_path, HANG_AFTER_RENDEZVOUS, script_args=[str(pid_file), str(fifo)]
            ),
            {"name": "trip", "summary": "s", "model": "interrupt", "prompt": "CLI"},
        ]

        def release_when_ready():
            open(fifo).close()
            fake_models.interrupt.gate.set()

        threading.Thread(target=release_when_ready, daemon=True).start()
        with pytest.raises(KeyboardInterrupt):
            run_plan(stages, CLIContext(instructions="go"), max_workers=2)

        assert_pid_dies(int(pid_file.read_text(encoding="utf-8")))


# Writes result.md, then blocks on a FIFO until the peer stage reaches the
# same point - a cross-process barrier proving the two stages overlapped
# while both had written the same filename.
RENDEZVOUS_WRITER = """
import os, sys
out = os.path.join(os.environ["LLM_PLAN_OUTPUT_DIR"], "result.md")
with open(out, "w", encoding="utf-8") as handle:
    handle.write("written by " + os.environ["LLM_PLAN_STAGE_NAME"])
fifo, mode = sys.argv[1], sys.argv[2]
with open(fifo, mode):
    pass
print(out)
"""


class TestScriptScratchIsolation:
    def test_overlapping_scripts_writing_the_same_filename_get_distinct_files(
        self, tmp_path, run_plan
    ):
        fifo = tmp_path / "rendezvous"
        os.mkfifo(fifo)
        stages = [
            script_stage(tmp_path, RENDEZVOUS_WRITER, name="left",
                         script_name="left.py", timeout=5,
                         depends_on=["seed"], script_args=[str(fifo), "w"]),
            script_stage(tmp_path, RENDEZVOUS_WRITER, name="right",
                         script_name="right.py", timeout=5,
                         depends_on=["seed"], script_args=[str(fifo), "r"]),
        ]

        runner, results = run_plan(stages, max_workers=4)

        left, right = results["left"], results["right"]
        assert left.success, left.error
        assert right.success, right.error
        assert left.files[0] != right.files[0]
        assert left.files[0].read_text(encoding="utf-8") == "written by left"
        assert right.files[0].read_text(encoding="utf-8") == "written by right"

    def test_stages_sharing_a_dependency_each_get_their_own_materialised_copy(
        self, tmp_path, run_plan
    ):
        stages = [
            {"name": "thinker", "summary": "s", "model": "echo", "prompt": "CLI"},
            script_stage(tmp_path, RECORDER_SCRIPT, name="left",
                         script_name="left.py", depends_on=["thinker"]),
            script_stage(tmp_path, RECORDER_SCRIPT, name="right",
                         script_name="right.py", depends_on=["thinker"]),
        ]

        runner, results = run_plan(stages, CLIContext(instructions="hi"))

        left = json.loads(results["left"].text)
        right = json.loads(results["right"].text)
        assert left["inputs"] == right["inputs"] == ["ECHO[hi]"]
        assert left["argv"][0] != right["argv"][0]


class TestScriptInputs:
    def test_env_plan_args_and_runtime_vars_reach_the_script(self, tmp_path, run_plan):
        stage = script_stage(
            tmp_path,
            RECORDER_SCRIPT,
            script_args=["--stage=${run.stage_name}", "--fixed"],
            env={"CUSTOM_VAR": "custom-value"},
        )
        cli = CLIContext(instructions="the question", plan_args=["--pr", "123"])

        runner, results = run_plan([stage], cli)

        record = json.loads(results["worker"].text)
        assert record["argv"] == ["--pr", "123", "--stage=worker", "--fixed"]
        assert record["instructions"] == "the question"
        assert record["stage"] == "worker"
        assert record["run_id"] == runner.run_id
        assert record["custom"] == "custom-value"

    def test_runtime_tokens_in_env_values_reach_the_script_substituted(self, tmp_path, run_plan):
        stage = script_stage(
            tmp_path,
            RECORDER_SCRIPT,
            env={"CUSTOM_VAR": "stage=${run.stage_name}"},
        )

        runner, results = run_plan([stage])

        record = json.loads(results["worker"].text)
        assert record["custom"] == "stage=worker"

    def test_llm_dependency_text_arrives_as_a_file_argument(self, tmp_path, run_plan):
        stages = [
            {"name": "thinker", "summary": "s", "model": "echo", "prompt": "CLI"},
            script_stage(tmp_path, RECORDER_SCRIPT, name="worker",
                         depends_on=["thinker"]),
        ]

        runner, results = run_plan(stages, CLIContext(instructions="hi"))

        record = json.loads(results["worker"].text)
        assert record["inputs"] == ["ECHO[hi]"]
        assert record["argv"][0].endswith("thinker.md")

    def test_script_files_flow_to_dependent_scripts(self, tmp_path, run_plan):
        stages = [
            script_stage(tmp_path, WRITER_SCRIPT, name="producer",
                         script_name="producer.py"),
            script_stage(tmp_path, RECORDER_SCRIPT, name="consumer",
                         script_name="consumer.py", depends_on=["producer"]),
        ]

        runner, results = run_plan(stages)

        record = json.loads(results["consumer"].text)
        assert record["inputs"] == ["first file body", "second file body"]

    def test_stage_files_are_passed_after_dependency_outputs(self, tmp_path, run_plan):
        (tmp_path / "reference.md").write_text("reference body", encoding="utf-8")
        stages = [
            {"name": "thinker", "summary": "s", "model": "echo", "prompt": "CLI"},
            script_stage(tmp_path, RECORDER_SCRIPT, name="worker",
                         depends_on=["thinker"],
                         files=[{"path": "reference.md", "label": "Reference"}]),
        ]

        runner, results = run_plan(stages, CLIContext(instructions="hi"))

        record = json.loads(results["worker"].text)
        assert record["inputs"] == ["ECHO[hi]", "reference body"]

    def test_first_stage_script_receives_seed_file_fragments(self, tmp_path, run_plan):
        seed = tmp_path / "seed.md"
        seed.write_text("seed body", encoding="utf-8")
        fragment = llm.Fragment("seed body", source=str(seed))
        stage = script_stage(tmp_path, RECORDER_SCRIPT)

        runner, results = run_plan([stage], CLIContext(fragments=[fragment]))

        record = json.loads(results["worker"].text)
        assert record["argv"] == [str(seed)]
        assert record["inputs"] == ["seed body"]

    @pytest.mark.parametrize("prompt, receives", [
        ("CLI", True),
        ("CLI:all", True),
        ("CLI:files", True),
        ("CLI:instructions", False),
    ])
    def test_script_cli_prompt_forms_control_seed_file_routing(
        self, tmp_path, prompt, receives
    , run_plan):
        stage = script_stage(tmp_path, RECORDER_SCRIPT, prompt=prompt)
        cli = CLIContext(instructions="hi", fragments=["seed fragment text"])

        runner, results = run_plan([stage], cli)

        record = json.loads(results["worker"].text)
        assert (record["inputs"] == ["seed fragment text"]) is receives

    def test_seed_dependent_script_gets_pathless_fragments_materialised(self, tmp_path, run_plan):
        stages = [
            {"name": "first", "summary": "s", "model": "echo", "prompt": "CLI"},
            script_stage(tmp_path, RECORDER_SCRIPT, name="worker",
                         depends_on=["seed"]),
        ]
        cli = CLIContext(instructions="hi", fragments=["fragment text with no path"])

        runner, results = run_plan(stages, cli)

        record = json.loads(results["worker"].text)
        assert record["inputs"] == ["fragment text with no path"]

    def test_headed_instructions_reach_scripts(self, tmp_path, run_plan):
        stage = script_stage(tmp_path, RECORDER_SCRIPT)
        cli = CLIContext(instruction_parts=[("Focus Areas", "headed body")])

        runner, results = run_plan([stage], cli)

        record = json.loads(results["worker"].text)
        assert record["instructions"] == "## Focus Areas\n\nheaded body"

    def test_unknown_runtime_variable_in_plan_args_fails_the_stage(self, tmp_path, run_plan):
        stage = script_stage(tmp_path, RECORDER_SCRIPT)
        cli = CLIContext(plan_args=["${cli.bogus}"])

        runner, results = run_plan([stage], cli, expect_error=True)

        assert not results["worker"].success
        assert "cli.bogus" in results["worker"].error


class TestScriptChainingIntoLlm:
    def test_llm_stage_receives_labelled_sections_per_produced_file(self, tmp_path, run_plan):
        stages = [
            script_stage(tmp_path, WRITER_SCRIPT, name="fetch", produces="Fetched"),
            {"name": "review", "summary": "s", "model": "echo",
             "depends_on": ["fetch"], "prompt": "inline:Review the data."},
        ]

        runner, results = run_plan(stages)

        text = results["review"].text
        assert "## Fetched Data" in text        # manifest label wins
        assert "## Fetched (2/2)" in text       # fallback label with index
        assert "first file body" in text and "second file body" in text

    def test_chain_prompt_from_a_script_uses_its_first_file(self, tmp_path, run_plan):
        stages = [
            script_stage(tmp_path, WRITER_SCRIPT, name="fetch"),
            {"name": "review", "summary": "s", "model": "echo",
             "depends_on": ["fetch"],
             "prompt": [{"prompt": "chain:fetch", "label": "Data"}]},
        ]

        runner, results = run_plan(stages)

        assert "## Data" in results["review"].text
        assert "first file body" in results["review"].text
