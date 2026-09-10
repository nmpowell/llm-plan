"""Log plan responses to llm's logs.db, exactly as ``llm prompt`` does.

``Response.log_to_db`` is a documented API since llm 0.32, but the rest of
llm's logging (``logs_db_path``, ``logs_on``, ``migrate``, the table layout)
is internal; this module keeps every use of *those* in one place so an
upstream change to logging lands here. Other modules use other llm.cli internals for their
own concerns (fragment and attachment resolution, model options).
"""

from __future__ import annotations

from pathlib import Path

import click
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
    """Apply llm's logging rules: on unless logs are off or -n, --log forces.

    Asking to force and suppress at once is contradictory; like llm prompt,
    reject it rather than guessing.
    """
    from llm.cli import logs_on

    if force_log and no_log:
        raise click.ClickException("--log and --no-log are mutually exclusive")
    if no_log:
        return False
    return logs_on() or force_log


def log_response(db: sqlite_utils.Database, response, conversation) -> None:
    """Record one completed response under the run's shared conversation.

    Sharing one conversation per run is what lets ``llm logs --cid <run-id>``
    retrieve every stage of a run with llm's own tooling.
    """
    response.conversation = conversation
    # A finished response no longer appends itself while draining; log_to_db
    # expects it present in conversation.responses.
    if not any(existing is response for existing in conversation.responses):
        conversation.responses.append(response)
    response.log_to_db(db)
    if conversation.name:
        _name_run(db, conversation.id, conversation.name)


def _name_run(db: sqlite_utils.Database, run_id: str, name: str) -> None:
    """Set the run's name on the row ``log_to_db`` created for it.

    ``log_to_db`` derives the name from the first prompt and ignores
    ``Conversation.name``. llm >= 0.32 records the run as a ``threads`` row;
    llm 0.31 as a row in the legacy ``conversations`` table.
    """
    for table in ("threads", "conversations"):
        if table in db.table_names() and db[table].count_where("id = ?", [run_id]):
            db[table].update(run_id, {"name": name})
            return
    raise RuntimeError(f"no thread or conversation row was logged for run {run_id}")
