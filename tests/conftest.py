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

    def execute(self, prompt, stream, response, conversation):
        self.prompts.append(prompt)
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


@pytest.fixture(autouse=True)
def fake_models():
    """Register throwaway models with llm's plugin manager for each test."""
    models = SimpleNamespace(
        echo=EchoModel(),
        other=EchoModel(model_id="other"),
        flaky=FlakyModel(),
    )

    class TestModelsPlugin:
        __name__ = "TestModelsPlugin"

        @llm.hookimpl
        def register_models(self, register):
            register(models.echo)
            register(models.other)
            register(models.flaky)

    pm.register(TestModelsPlugin(), name="llm-plan-test-models")
    try:
        yield models
    finally:
        pm.unregister(name="llm-plan-test-models")
