import os

# Keep the suite hermetic: llm loads every installed entry-point plugin unless
# told otherwise, and it reads this variable once at import time - so it must
# be assigned (not defaulted, or an exported value would defeat isolation)
# before anything imports llm.
os.environ["LLM_LOAD_PLUGINS"] = "llm-plan"

import base64
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Optional

import llm
import pytest
from llm.plugins import pm
from pydantic import Field

# A real 1x1 PNG: attachment type detection must work on every llm version.
PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture
def png_bytes():
    return PNG_1PX


@pytest.fixture
def run_plan(tmp_path):
    """Write a plan file, run it, and return (runner, results-by-stage-name).

    Unexpected errors propagate; a test exercising a failing plan passes
    ``expect_error=True`` and the fixture asserts the run raised PlanError.
    """
    import yaml

    from llm_plan.models import PlanError
    from llm_plan.runner import CLIContext, PlanRunner, load_plan

    def _run_plan(stages, cli=None, max_workers=1, expect_error=False, **runner_kwargs):
        data = {
            "name": "test",
            "summary": "a test plan",
            "parallel_config": {"max_workers": max_workers},
            "stages": stages,
        }
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(yaml.safe_dump(data), encoding="utf-8")
        runner = PlanRunner(load_plan(plan_file), cli or CLIContext(), retry_delay=0, **runner_kwargs)
        if expect_error:
            with pytest.raises(PlanError):
                runner.run()
        else:
            runner.run()
        return runner, dict(runner.results)

    return _run_plan


@pytest.fixture(autouse=True)
def user_dir(monkeypatch, tmp_path):
    """Isolate every test from the developer's real environment.

    Points llm's user directory and the runner's scratch directories into
    the test sandbox, and clears ambient variables that would change plan
    discovery or let a stray test reach a real provider.
    """
    llm_dir = tmp_path / "llm-user"
    llm_dir.mkdir()
    monkeypatch.setenv("LLM_USER_PATH", str(llm_dir))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    for name in ("LLM_PLAN_DIRS", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return llm_dir


class EchoModel(llm.Model):
    """Echoes its full prompt so tests can assert on composition."""

    model_id = "echo"
    can_stream = True
    attachment_types = {"image/png", "text/plain"}

    class Options(llm.Options):
        max_tokens: Optional[int] = Field(default=None)
        temperature: Optional[float] = Field(default=None)

    def __init__(self, model_id="echo"):
        self.model_id = model_id
        self.prompts = []
        self.stream_flags = []

    def execute(self, prompt, stream, response, conversation):
        self.prompts.append(prompt)
        self.stream_flags.append(stream)
        yield f"{self.model_id.upper()}[{prompt.prompt}]"


class FlakyModel(llm.Model):
    """Fails a set number of times, then echoes."""

    model_id = "flaky"
    can_stream = True

    def __init__(self):
        self.failures_left = 0
        self.calls = 0

    def execute(self, prompt, stream, response, conversation):
        self.calls += 1
        if self.failures_left > 0:
            self.failures_left -= 1
            raise RuntimeError("transient upstream error")
        yield "FLAKY-OK"


class NonStreamingModel(llm.Model):
    """Rejects streamed execution, like models that declare can_stream=False."""

    model_id = "nostream"
    can_stream = False

    def execute(self, prompt, stream, response, conversation):
        if stream:
            raise RuntimeError("non-stream model received stream=True")
        return ["NOSTREAM-OK"]


class PairModel(llm.Model):
    """Succeeds only when two executions overlap in time."""

    model_id = "pair"
    can_stream = True

    def __init__(self):
        self.barrier = threading.Barrier(2)

    def execute(self, prompt, stream, response, conversation):
        try:
            self.barrier.wait(timeout=5)
        except threading.BrokenBarrierError:
            raise RuntimeError("execution did not overlap with a second call")
        yield "PAIR-OK"


class TimingModel(llm.Model):
    """Records an (identifier, start, end) interval per execution."""

    model_id = "timing"
    can_stream = True

    def __init__(self):
        self.intervals = []
        self._lock = threading.Lock()

    def execute(self, prompt, stream, response, conversation):
        start = time.monotonic()
        time.sleep(0.05)
        end = time.monotonic()
        with self._lock:
            self.intervals.append((prompt.prompt, start, end))
        yield "TIMED"


@pytest.fixture(autouse=True)
def fake_models():
    """Register throwaway models with llm's plugin manager for each test."""
    models = SimpleNamespace(
        echo=EchoModel(),
        other=EchoModel(model_id="other"),
        flaky=FlakyModel(),
        nostream=NonStreamingModel(),
        pair=PairModel(),
        timing=TimingModel(),
    )

    class TestModelsPlugin:
        __name__ = "TestModelsPlugin"

        @llm.hookimpl
        def register_models(self, register):
            register(models.echo)
            register(models.other)
            register(models.flaky)
            register(models.nostream)
            register(models.pair)
            register(models.timing)

    pm.register(TestModelsPlugin(), name="llm-plan-test-models")
    try:
        yield models
    finally:
        pm.unregister(name="llm-plan-test-models")
