import threading
import time
from types import SimpleNamespace
from typing import Optional

import llm
import pytest
from llm.plugins import pm
from pydantic import Field


@pytest.fixture(autouse=True)
def user_dir(monkeypatch, tmp_path):
    """Isolate every test from the real llm user directory."""
    llm_dir = tmp_path / "llm-user"
    llm_dir.mkdir()
    monkeypatch.setenv("LLM_USER_PATH", str(llm_dir))
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
