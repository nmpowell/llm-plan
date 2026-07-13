"""Log plan responses to llm's logs.db, exactly as ``llm prompt`` does.

This is the one module that touches llm's logging internals
(``logs_db_path``, ``migrate``, ``Response.log_to_db``) - they are not yet
part of llm's stable public API, so keep every use of them here.
"""

from __future__ import annotations

from pathlib import Path

import sqlite_utils


def open_logs_db(database: str | Path | None = None) -> sqlite_utils.Database:
    """Open (and migrate) the llm logs database."""
    from llm.cli import logs_db_path
    from llm.migrations import migrate

    path = Path(database) if database else logs_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite_utils.Database(path)
    try:
        migrate(db)
    except Exception:
        db.close()
        raise
    return db


def logging_enabled(*, force_log: bool = False, no_log: bool = False) -> bool:
    """Apply llm's logging rules: on unless logs are off or -n, --log forces."""
    from llm.cli import logs_on

    if no_log:
        return False
    return logs_on() or force_log


def log_response(db: sqlite_utils.Database, response) -> None:
    """Record one completed response, like the llm prompt command does."""
    response.log_to_db(db)
