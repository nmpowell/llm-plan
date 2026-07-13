import pytest
import sqlite_utils

from llm_plan.logs import open_logs_db


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
