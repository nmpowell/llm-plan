import pytest
import sqlite_utils

from llm_plan.logs import _name_run, open_logs_db


class TestOpenLogsDb:
    def test_closes_the_connection_when_migration_fails(self, tmp_path, monkeypatch):
        closed = []
        real_close = sqlite_utils.Database.close

        def spying_close(self):
            closed.append(True)
            real_close(self)

        def failing_migrate(db):
            raise RuntimeError("migration boom")

        monkeypatch.setattr(sqlite_utils.Database, "close", spying_close)
        monkeypatch.setattr("llm.migrations.migrate", failing_migrate)

        with pytest.raises(RuntimeError, match="migration boom"):
            open_logs_db(tmp_path / "logs.db")

        assert closed == [True]


@pytest.fixture
def db():
    db = sqlite_utils.Database(memory=True)
    try:
        yield db
    finally:
        db.close()


class TestNameRun:
    """The run's row lives in threads (llm >= 0.32) or conversations (0.31)."""

    def test_names_the_thread_when_llm_logged_one(self, db):
        db["threads"].insert({"id": "run1", "name": "from first prompt"}, pk="id")
        db["conversations"].insert({"id": "run1", "name": "legacy"}, pk="id")

        _name_run(db, "run1", "my plan")

        assert db["threads"].get("run1")["name"] == "my plan"
        assert db["conversations"].get("run1")["name"] == "legacy"

    def test_falls_back_to_the_legacy_conversations_table(self, db):
        db["conversations"].insert({"id": "run1", "name": "from prompt"}, pk="id")

        _name_run(db, "run1", "my plan")

        assert db["conversations"].get("run1")["name"] == "my plan"

    def test_raises_when_no_table_holds_the_run(self, db):
        db["threads"].insert({"id": "other", "name": "x"}, pk="id")

        with pytest.raises(RuntimeError, match="run1"):
            _name_run(db, "run1", "my plan")
