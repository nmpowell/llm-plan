import pytest


@pytest.fixture(autouse=True)
def user_dir(monkeypatch, tmp_path):
    """Isolate every test from the real llm user directory."""
    llm_dir = tmp_path / "llm-user"
    llm_dir.mkdir()
    monkeypatch.setenv("LLM_USER_PATH", str(llm_dir))
    return llm_dir
