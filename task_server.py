"""MCP server for the AI-Powered Personal Task Manager.

Design notes
------------
* One SQLite database only.  The path is resolved from (in order):
  ``--db`` on the command line, the ``TASK_DB_PATH`` environment variable,
  then ``tasks.db`` next to this file.  It never depends on the process
  working directory, so the notebook, the Streamlit app and this server all
  share the same file.
* Every statement is parameterised - table/column names are never built from
  user input; the only dynamic SQL is a ``<column> = ?`` list built from a
  fixed whitelist of column names.
* Transactions: successful writes are committed, failures are rolled back and
  connections are always closed (the previous implementation leaked a
  connection per tool call because ``with sqlite3.Connection`` only
  commits/rolls back, it does not close).
* Every mutation returns a structured payload with ``success``, ``status``
  and ``affected`` plus the row read back from the database, so callers can
  verify the outcome instead of trusting that a SQL statement was attempted.
* Task resolution (ID or title) is deterministic Python code, never a
  guess by the model: ambiguous or conflicting targets refuse to act.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("task_server")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

CATEGORIES: tuple[str, ...] = (
    "Personal",
    "Work",
    "Study",
    "Finance",
    "Health",
    "Shopping",
    "Other",
)
PRIORITIES: tuple[str, ...] = ("Low", "Medium", "High")
STATUSES: tuple[str, ...] = ("Pending", "Completed")
DATE_FORMAT = "%Y-%m-%d"

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "tasks.db"

TASK_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS tasks (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT NOT NULL,
    Category TEXT NOT NULL DEFAULT 'Personal',
    Description TEXT NOT NULL DEFAULT '',
    Priority TEXT NOT NULL DEFAULT 'Medium'
        CHECK(Priority IN ('Low', 'Medium', 'High')),
    due_date TEXT NOT NULL DEFAULT '',
    Status TEXT NOT NULL DEFAULT 'Pending'
        CHECK(Status IN ('Pending', 'Completed')),
    Created_at TEXT NOT NULL,
    Updated_at TEXT NOT NULL,
    Completion_at TEXT
)
"""

HISTORY_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS task_history (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    function_name TEXT NOT NULL,
    event_time TEXT NOT NULL
)
"""

# Columns that must exist on the ``tasks`` table, with the definition used
# when migrating an older database.  Existing rows are never dropped.
TASK_COLUMNS: dict[str, str] = {
    "ID": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "Title": "TEXT NOT NULL DEFAULT ''",
    "Category": "TEXT NOT NULL DEFAULT 'Personal'",
    "Description": "TEXT NOT NULL DEFAULT ''",
    "Priority": "TEXT NOT NULL DEFAULT 'Medium'",
    "due_date": "TEXT NOT NULL DEFAULT ''",
    "Status": "TEXT NOT NULL DEFAULT 'Pending'",
    "Created_at": "TEXT NOT NULL DEFAULT ''",
    "Updated_at": "TEXT NOT NULL DEFAULT ''",
    "Completion_at": "TEXT",
}

# Columns a tool is allowed to write, keyed by the tool argument name.
UPDATABLE_COLUMNS: dict[str, str] = {
    "new_title": "Title",
    "category": "Category",
    "description": "Description",
    "priority": "Priority",
    "due_date": "due_date",
}

_CLI_DB_PATH: str | None = None
_MIGRATED: set[str] = set()

mcp = FastMCP("Task Manager")


# --------------------------------------------------------------------------- #
# Database plumbing
# --------------------------------------------------------------------------- #


def resolve_db_path() -> Path:
    """Return the stable path of the task database.

    Order of precedence: ``--db`` argument, ``TASK_DB_PATH`` environment
    variable, ``tasks.db`` next to this file.  The result never depends on
    the current working directory.
    """
    if _CLI_DB_PATH:
        candidate = Path(_CLI_DB_PATH)
    else:
        env_path = os.environ.get("TASK_DB_PATH")
        candidate = Path(env_path) if env_path else DEFAULT_DB_PATH
    return candidate.expanduser().resolve()


@contextmanager
def db_connection() -> Iterator[sqlite3.Connection]:
    """Open a connection, migrate on first use, commit/rollback and close.

    Unlike ``with sqlite3.connect(...)`` this actually closes the connection,
    so tool calls no longer leak file handles or hold stale locks.
    """
    path = resolve_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=15.0)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 15000")
        ensure_schema(connection, path)
        try:
            yield connection
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()
    finally:
        connection.close()


def ensure_schema(connection: sqlite3.Connection, path: Path | None = None) -> None:
    """Create missing tables/columns without touching existing rows."""
    key = str(path or resolve_db_path())
    if key in _MIGRATED:
        return
    connection.execute(TASK_TABLE_DDL)
    connection.execute(HISTORY_TABLE_DDL)

    existing = {
        str(row["name"]).lower()
        for row in connection.execute("PRAGMA table_info(tasks)")
    }
    for column, declaration in TASK_COLUMNS.items():
        if column.lower() in existing:
            continue
        try:
            connection.execute(
                f"ALTER TABLE tasks ADD COLUMN {column} {declaration}"
            )
        except sqlite3.OperationalError:  # pragma: no cover - concurrent migr.
            logger.warning("Could not add column %s (already migrated?)", column)
    connection.commit()
    _MIGRATED.add(key)


def init_db() -> Path:
    """Create/upgrade the schema and return the database path."""
    path = resolve_db_path()
    with db_connection() as connection:
        ensure_schema(connection, path)
    return path


def current_timestamp() -> str:
    """Local time with timezone offset, ISO-8601 to the second."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Small helpers: results, validation, logging
# --------------------------------------------------------------------------- #


def _result(success: bool, status: str, message: str = "", **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"success": bool(success), "status": status}
    if message:
        payload["message"] = message
    payload.update(extra)
    return payload


def _normalise_choice(value: Any, allowed: tuple[str, ...]) -> str | None:
    if not isinstance(value, str):
        return None
    wanted = value.strip().lower()
    for option in allowed:
        if option.lower() == wanted:
            return option
    return None


def _check_title(value: Any) -> tuple[str | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, "Title must be a non-empty string."
    return value.strip(), None


def _check_enum(value: Any, allowed: tuple[str, ...], field: str) -> tuple[str | None, str | None]:
    if value is None:
        return None, None  # caller decides what "unset" means
    option = _normalise_choice(value, allowed)
    if option is None:
        return None, f"Invalid {field} {value!r}. Allowed: {', '.join(allowed)}."
    return option, None


def _check_date(value: Any) -> tuple[str | None, str | None]:
    """Return a validated ``YYYY-MM-DD`` string (``''`` means no due date)."""
    if value is None:
        return None, None
    if not isinstance(value, str):
        return None, "Due date must be a string in YYYY-MM-DD format."
    text = value.strip()
    if not text:
        return "", None
    try:
        parsed = datetime.strptime(text, DATE_FORMAT)
    except ValueError:
        return None, f"Invalid date {value!r}. Use YYYY-MM-DD (for example 2026-10-25)."
    if parsed.strftime(DATE_FORMAT) != text:
        return None, f"Invalid date {value!r}. Use YYYY-MM-DD (for example 2026-10-25)."
    return text, None


def _check_description(value: Any) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    if not isinstance(value, str):
        return None, "Description must be a string."
    return value, None


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    data["Completion_at"] = data.get("Completion_at") or None
    return data


def _log_event(
    connection: sqlite3.Connection,
    task_id: int,
    event_type: str,
    old_value: Any,
    new_value: Any,
    function_name: str,
) -> None:
    """Best-effort audit row; never fails the mutation itself."""
    try:
        connection.execute(
            """INSERT INTO task_history
               (task_id, event_type, old_value, new_value, function_name, event_time)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                event_type,
                json.dumps(old_value, sort_keys=True) if old_value is not None else None,
                json.dumps(new_value, sort_keys=True) if new_value is not None else None,
                function_name,
                current_timestamp(),
            ),
        )
    except sqlite3.Error:  # pragma: no cover - audit must not break writes
        logger.warning("Could not write task_history row for task %s", task_id)


# --------------------------------------------------------------------------- #
# Deterministic task resolution
# --------------------------------------------------------------------------- #


def resolve_task_record(
    task_id: int | None = None, title: str | None = None
) -> dict[str, Any]:
    """Resolve ``task_id`` or ``title`` to exactly one row.

    Returns ``status`` of ``ok`` (``task`` set), ``ambiguous`` (``candidates``
    set), ``not_found`` (``suggestions`` may be set) or ``invalid_input``.
    When both an ID and a title are supplied they must agree, otherwise the
    call is refused with ``conflict`` - this is what guarantees a mutation can
    never land on the wrong record.
    """
    has_id = isinstance(task_id, int) and task_id > 0
    title_text = title.strip() if isinstance(title, str) else ""

    if not has_id and not title_text:
        return _result(
            False,
            "invalid_input",
            "Provide task_id or title so the target task can be identified.",
            affected=0,
        )

    with db_connection() as connection:
        if has_id:
            row = connection.execute(
                "SELECT * FROM tasks WHERE ID = ?", (task_id,)
            ).fetchone()
            if row is None:
                return _result(
                    False,
                    "not_found",
                    f"Task {task_id} does not exist.",
                    task_id=task_id,
                    affected=0,
                )
            if title_text and str(row["Title"]).strip().lower() != title_text.lower():
                return _result(
                    False,
                    "conflict",
                    f"Task {task_id} is {row['Title']!r} but title {title_text!r} was "
                    "given. Refusing to touch a different record - re-check the ID.",
                    task_id=task_id,
                    affected=0,
                )
            return _result(True, "ok", task=_row_to_dict(row), task_id=row["ID"])

        exact = connection.execute(
            "SELECT * FROM tasks WHERE lower(Title) = lower(?) ORDER BY ID",
            (title_text,),
        ).fetchall()
        if len(exact) == 1:
            task = _row_to_dict(exact[0])
            return _result(True, "ok", task=task, task_id=task["ID"])
        if len(exact) > 1:
            candidates = [_row_to_dict(row) for row in exact]
            return _result(
                False,
                "ambiguous",
                f"{len(exact)} tasks are named {title_text!r}. Ask the user which one "
                "(ID, title, status, due date) before changing or deleting anything.",
                candidates=candidates,
                task_id=None,
                affected=0,
            )

        partial = connection.execute(
            """SELECT * FROM tasks
               WHERE lower(Title) LIKE lower(?)
               ORDER BY ID LIMIT 5""",
            (f"%{title_text}%",),
        ).fetchall()
        suggestions = [_row_to_dict(row) for row in partial]
        if not suggestions:
            # Typo tolerance: offer near-miss titles instead of a bare "not found".
            all_rows = connection.execute("SELECT * FROM tasks").fetchall()
            close = difflib.get_close_matches(
                title_text.lower(),
                [str(row["Title"]).lower() for row in all_rows],
                n=5,
                cutoff=0.5,
            )
            if close:
                wanted = set(close)
                suggestions = [
                    _row_to_dict(row)
                    for row in all_rows
                    if str(row["Title"]).lower() in wanted
                ]
        message = f"No task named {title_text!r} was found."
        if suggestions:
            message += " Similar tasks: " + ", ".join(
                f"{item['ID']} {item['Title']!r}" for item in suggestions
            )
        return _result(
            False,
            "not_found",
            message,
            suggestions=suggestions,
            task_id=None,
            affected=0,
        )


# --------------------------------------------------------------------------- #
# Tool: create
# --------------------------------------------------------------------------- #


@mcp.tool()
def create_task(
    title: str,
    category: str | None = "Personal",
    description: str | None = "",
    priority: str | None = "Medium",
    due_date: str | None = "",
) -> dict[str, Any]:
    """Create a new task and return the row as stored in SQLite.

    Args:
        title: Required task name (non-empty).
        category: One of Personal, Work, Study, Finance, Health, Shopping, Other.
        description: Free text, may be empty.
        priority: Low, Medium or High (default Medium).
        due_date: YYYY-MM-DD or an empty string for "no due date".
    """
    if not isinstance(title, str):
        return _result(False, "invalid_input", "Title must be a string.", affected=0)

    title_text, error = _check_title(title)
    if error:
        return _result(False, "invalid_input", error, affected=0)

    category_value = "Personal" if category is None else category
    category_text, error = _check_enum(category_value, CATEGORIES, "category")
    if error:
        return _result(False, "invalid_input", error, affected=0)

    priority_value = "Medium" if priority is None else priority
    priority_text, error = _check_enum(priority_value, PRIORITIES, "priority")
    if error:
        return _result(False, "invalid_input", error, affected=0)

    date_text, error = _check_date(due_date)
    if error:
        return _result(False, "invalid_input", error, affected=0)

    description_text = "" if description is None else description
    if not isinstance(description_text, str):
        return _result(
            False, "invalid_input", "Description must be a string.", affected=0
        )

    now = current_timestamp()
    try:
        with db_connection() as connection:
            cursor = connection.execute(
                """INSERT INTO tasks
                   (Title, Category, Description, Priority, due_date, Status,
                    Created_at, Updated_at, Completion_at)
                   VALUES (?, ?, ?, ?, ?, 'Pending', ?, ?, NULL)""",
                (
                    title_text,
                    category_text,
                    description_text,
                    priority_text,
                    date_text or "",
                    now,
                    now,
                ),
            )
            task_id = int(cursor.lastrowid or 0)
            row = connection.execute(
                "SELECT * FROM tasks WHERE ID = ?", (task_id,)
            ).fetchone()
            task = _row_to_dict(row)
            if task is None:
                connection.rollback()
                return _result(
                    False,
                    "db_error",
                    "The task was inserted but could not be read back; the change "
                    "was rolled back.",
                    affected=0,
                )
            _log_event(connection, task_id, "create", None, task, "create_task")
    except sqlite3.Error as exc:
        logger.exception("create_task failed")
        return _result(
            False, "db_error", f"Database error while creating the task: {exc}", affected=0
        )

    return _result(
        True,
        "ok",
        f"Task {task_id} created.",
        task=task,
        task_id=task_id,
        affected=1,
    )


# --------------------------------------------------------------------------- #
# Tools: read / list / search
# --------------------------------------------------------------------------- #


@mcp.tool()
def list_tasks(
    status: str | None = None,
    category: str | None = None,
    priority: str | None = None,
) -> dict[str, Any]:
    """List tasks with optional filters.

    Leave every argument out to return ALL tasks (pending and completed).
    Only add the filters the user actually asked for - do not assume a status.

    Args:
        status: 'Pending' or 'Completed'. Omit for both.
        category: Personal, Work, Study, Finance, Health, Shopping or Other.
        priority: Low, Medium or High (matches regardless of status unless
            the user also asked for a status).
    """
    clauses: list[str] = ["1=1"]
    params: list[Any] = []
    applied: dict[str, str] = {}

    if status is not None and status != "":
        value, error = _check_enum(status, STATUSES, "status")
        if error:
            return _result(False, "invalid_input", error, tasks=[], count=0)
        clauses.append("Status = ?")
        params.append(value)
        applied["status"] = value

    if category is not None and category != "":
        value, error = _check_enum(category, CATEGORIES, "category")
        if error:
            return _result(False, "invalid_input", error, tasks=[], count=0)
        clauses.append("Category = ?")
        params.append(value)
        applied["category"] = value

    if priority is not None and priority != "":
        value, error = _check_enum(priority, PRIORITIES, "priority")
        if error:
            return _result(False, "invalid_input", error, tasks=[], count=0)
        clauses.append("Priority = ?")
        params.append(value)
        applied["priority"] = value

    query = f"SELECT * FROM tasks WHERE {' AND '.join(clauses)} ORDER BY ID DESC"

    try:
        with db_connection() as connection:
            rows = connection.execute(query, params).fetchall()
    except sqlite3.Error as exc:
        logger.exception("list_tasks failed")
        return _result(False, "db_error", f"Database error while listing tasks: {exc}")

    tasks = [_row_to_dict(row) for row in rows]
    counts = {status_name: 0 for status_name in STATUSES}
    for task in tasks:
        counts[task["Status"]] = counts.get(task["Status"], 0) + 1

    return _result(
        True,
        "ok",
        count=len(tasks),
        tasks=tasks,
        filters_applied=applied,
        counts_by_status=counts,
    )


@mcp.tool()
def get_task(task_id: int) -> dict[str, Any]:
    """Return the full stored record for one task by its ID."""
    if not isinstance(task_id, int) or isinstance(task_id, bool) or task_id <= 0:
        return _result(
            False, "invalid_input", "task_id must be a positive integer.", affected=0
        )
    with db_connection() as connection:
        row = connection.execute("SELECT * FROM tasks WHERE ID = ?", (task_id,)).fetchone()
    if row is None:
        return _result(
            False,
            "not_found",
            f"Task {task_id} does not exist.",
            task_id=task_id,
            affected=0,
        )
    return _result(True, "ok", task=_row_to_dict(row), task_id=task_id, affected=1)


@mcp.tool()
def find_tasks(title: str) -> dict[str, Any]:
    """Find tasks by title (case-insensitive).

    Tries an exact title match first, then a partial match. Returns
    ``ambiguous: true`` with ``candidates`` when several tasks share the
    title - ask the user which one they mean instead of guessing.
    """
    if not isinstance(title, str) or not title.strip():
        return _result(
            False,
            "invalid_input",
            "title must be a non-empty string.",
            tasks=[],
            count=0,
        )
    wanted = title.strip()
    with db_connection() as connection:
        exact = connection.execute(
            "SELECT * FROM tasks WHERE lower(Title) = lower(?) ORDER BY ID",
            (wanted,),
        ).fetchall()
        match_type = "exact"
        if not exact:
            exact = connection.execute(
                "SELECT * FROM tasks WHERE lower(Title) LIKE lower(?) ORDER BY ID LIMIT 10",
                (f"%{wanted}%",),
            ).fetchall()
            match_type = "partial"

    tasks = [_row_to_dict(row) for row in exact]
    if not tasks:
        return _result(
            False,
            "not_found",
            f"No task matching {wanted!r}.",
            tasks=[],
            count=0,
            ambiguous=False,
            match_type=match_type,
        )
    ambiguous = len(tasks) > 1
    message = None
    if ambiguous:
        message = (
            f"{len(tasks)} tasks match {wanted!r}. Ask the user which one "
            "(ID, title, status, due date) before acting."
        )
    return _result(
        True,
        "ok",
        message=message,
        count=len(tasks),
        tasks=tasks,
        ambiguous=ambiguous,
        match_type=match_type,
    )


# --------------------------------------------------------------------------- #
# Tools: mutate
# --------------------------------------------------------------------------- #


@mcp.tool()
def update_task(
    task_id: int | None = None,
    title: str | None = None,
    new_title: str | None = None,
    category: str | None = None,
    description: str | None = None,
    priority: str | None = None,
    due_date: str | None = None,
) -> dict[str, Any]:
    """Update the supplied fields of exactly one task; all others are preserved.

    Identify the task with ``task_id`` (preferred) or ``title`` (case-insensitive
    exact match). If both are given they must refer to the same task. An
    ambiguous or unknown target refuses to act.

    Args:
        task_id: ID of the task to change.
        title: Existing title of the task to change (use when no ID is known).
        new_title: New title for the task (renames it).
        category: New category (Personal/Work/Study/Finance/Health/Shopping/Other).
        description: New description; an empty string clears it.
        priority: New priority: Low, Medium or High.
        due_date: New due date YYYY-MM-DD; an empty string clears it.
        Omitted/None arguments are left unchanged.
    """
    # ---- validate every supplied field before touching the database ----
    updates: dict[str, Any] = {}

    if new_title is not None:
        text, error = _check_title(new_title)
        if error:
            return _result(False, "invalid_input", f"Invalid new_title: {error}", affected=0)
        updates["Title"] = text

    if category is not None:
        value, error = _check_enum(category, CATEGORIES, "category")
        if error:
            return _result(False, "invalid_input", error, affected=0)
        updates["Category"] = value

    if description is not None:
        value, error = _check_description(description)
        if error:
            return _result(False, "invalid_input", error, affected=0)
        updates["Description"] = value

    if priority is not None:
        value, error = _check_enum(priority, PRIORITIES, "priority")
        if error:
            return _result(False, "invalid_input", error, affected=0)
        updates["Priority"] = value

    if due_date is not None:
        value, error = _check_date(due_date)
        if error:
            return _result(False, "invalid_input", error, affected=0)
        updates["due_date"] = value or ""

    resolved = resolve_task_record(task_id, title)
    if not resolved["success"]:
        return resolved

    if not updates:
        return _result(
            False,
            "invalid_input",
            "No fields supplied to update. Omitted fields are left unchanged, so "
            "at least one of new_title, category, description, priority, due_date "
            "must be provided.",
            affected=0,
        )

    target = resolved["task"]
    now = current_timestamp()
    try:
        with db_connection() as connection:
            previous = {
                column: target[column] for column in updates
            }
            assignments = ", ".join(f"{column} = ?" for column in updates)
            connection.execute(
                f"UPDATE tasks SET {assignments}, Updated_at = ? WHERE ID = ?",
                (*updates.values(), now, target["ID"]),
            )
            affected = int(
                connection.execute("SELECT changes()").fetchone()[0]
            )
            if affected != 1:
                connection.rollback()
                return _result(
                    False,
                    "no_match",
                    f"Task {target['ID']} disappeared before the update was applied.",
                    task_id=target["ID"],
                    affected=0,
                )

            row = connection.execute(
                "SELECT * FROM tasks WHERE ID = ?", (target["ID"],)
            ).fetchone()
            task = _row_to_dict(row)
            mismatched = [
                column
                for column, value in updates.items()
                if task is None or task[column] != value
            ]
            if mismatched:
                connection.rollback()
                return _result(
                    False,
                    "db_error",
                    "Update verification failed for: "
                    + ", ".join(mismatched)
                    + ". The change was rolled back.",
                    task_id=target["ID"],
                    affected=0,
                )
            _log_event(
                connection,
                target["ID"],
                "update",
                previous,
                {column: task[column] for column in updates},
                "update_task",
            )
    except sqlite3.Error as exc:
        logger.exception("update_task failed")
        return _result(
            False, "db_error", f"Database error while updating the task: {exc}", affected=0
        )

    return _result(
        True,
        "ok",
        f"Task {target['ID']} updated ({', '.join(updates)}).",
        task=task,
        task_id=target["ID"],
        updated_fields=list(updates),
        affected=1,
    )


@mcp.tool()
def complete_task(task_id: int | None = None, title: str | None = None) -> dict[str, Any]:
    """Mark exactly one pending task as completed.

    Identify the task with ``task_id`` (preferred) or ``title``. Refuses to act
    on unknown or ambiguous targets, and reports ``already_completed`` instead
    of silently doing nothing.
    """
    resolved = resolve_task_record(task_id, title)
    if not resolved["success"]:
        return resolved

    target = resolved["task"]
    if target["Status"] == "Completed":
        return _result(
            False,
            "already_completed",
            f"Task {target['ID']} '{target['Title']}' is already completed "
            f"(completed at {target['Completion_at']}).",
            task=target,
            task_id=target["ID"],
            affected=0,
        )

    now = current_timestamp()
    try:
        with db_connection() as connection:
            connection.execute(
                """UPDATE tasks
                   SET Status = 'Completed', Completion_at = ?, Updated_at = ?
                   WHERE ID = ? AND Status = 'Pending'""",
                (now, now, target["ID"]),
            )
            affected = int(connection.execute("SELECT changes()").fetchone()[0])
            if affected != 1:
                connection.rollback()
                return _result(
                    False,
                    "no_match",
                    f"Task {target['ID']} was not pending, so nothing was changed.",
                    task_id=target["ID"],
                    affected=0,
                )
            row = connection.execute(
                "SELECT * FROM tasks WHERE ID = ?", (target["ID"],)
            ).fetchone()
            task = _row_to_dict(row)
            if task is None or task["Status"] != "Completed":
                connection.rollback()
                return _result(
                    False,
                    "db_error",
                    "Completion could not be verified; the change was rolled back.",
                    task_id=target["ID"],
                    affected=0,
                )
            _log_event(
                connection,
                target["ID"],
                "complete",
                {"Status": "Pending"},
                {"Status": "Completed", "Completion_at": task["Completion_at"]},
                "complete_task",
            )
    except sqlite3.Error as exc:
        logger.exception("complete_task failed")
        return _result(
            False,
            "db_error",
            f"Database error while completing the task: {exc}",
            affected=0,
        )

    return _result(
        True,
        "ok",
        f"Task {target['ID']} marked completed.",
        task=task,
        task_id=target["ID"],
        affected=1,
    )


@mcp.tool()
def delete_task(task_id: int | None = None, title: str | None = None) -> dict[str, Any]:
    """Permanently delete exactly one task.

    Identify the task with ``task_id`` (preferred) or ``title``. Ambiguous or
    unknown targets are refused, so the wrong record is never removed.
    """
    resolved = resolve_task_record(task_id, title)
    if not resolved["success"]:
        return resolved

    target = resolved["task"]
    try:
        with db_connection() as connection:
            connection.execute("DELETE FROM tasks WHERE ID = ?", (target["ID"],))
            affected = int(connection.execute("SELECT changes()").fetchone()[0])
            if affected != 1:
                connection.rollback()
                return _result(
                    False,
                    "no_match",
                    f"Task {target['ID']} was already gone; nothing was deleted.",
                    task_id=target["ID"],
                    affected=0,
                )
            leftover = connection.execute(
                "SELECT 1 FROM tasks WHERE ID = ?", (target["ID"],)
            ).fetchone()
            if leftover is not None:
                connection.rollback()
                return _result(
                    False,
                    "db_error",
                    "Delete verification failed (row still present); "
                    "the change was rolled back.",
                    task_id=target["ID"],
                    affected=0,
                )
            _log_event(
                connection,
                target["ID"],
                "delete",
                target,
                None,
                "delete_task",
            )
    except sqlite3.Error as exc:
        logger.exception("delete_task failed")
        return _result(
            False, "db_error", f"Database error while deleting the task: {exc}", affected=0
        )

    return _result(
        True,
        "ok",
        f"Task {target['ID']} '{target['Title']}' deleted.",
        task_id=target["ID"],
        title=target["Title"],
        task=target,
        affected=1,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    """Parse our own CLI flags, then serve MCP over stdio."""
    global _CLI_DB_PATH

    parser = argparse.ArgumentParser(prog="task_server.py")
    parser.add_argument(
        "--db",
        default=None,
        help="Path to the SQLite database (default: tasks.db next to this file, "
        "or $TASK_DB_PATH).",
    )
    args, _unknown = parser.parse_known_args(argv)
    if args.db:
        _CLI_DB_PATH = args.db

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    path = init_db()
    logger.info("Serving task MCP on stdio with database %s", path)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
