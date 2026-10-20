"""Database-layer tests for task_server (no LLM, no MCP transport).

Covers the CRUD matrix from the task brief: create, get by ID, get by title,
list all/status/priority/category, partial updates, completion, deletion,
missing and duplicate matches, invalid input and persistence.
"""

from __future__ import annotations

import sqlite3

import pytest

import task_server as ts


def db_rows():
    """Read the table straight from SQLite, bypassing the helper functions."""
    with sqlite3.connect(ts.resolve_db_path()) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute("SELECT * FROM tasks")]


# --------------------------------------------------------------------------- #
# 1-3. create / get by id / get by title
# --------------------------------------------------------------------------- #


def test_create_task_stores_defaults_and_returns_read_back(isolated_task_db):
    result = ts.create_task("Prepare assignment")

    assert result["success"] is True
    assert result["status"] == "ok"
    assert result["affected"] == 1
    task = result["task"]
    assert task["ID"] == result["task_id"]
    assert task["Title"] == "Prepare assignment"
    assert task["Category"] == "Personal"
    assert task["Priority"] == "Medium"
    assert task["Status"] == "Pending"
    assert task["due_date"] == ""
    assert task["Completion_at"] is None

    # the payload must match what a fresh read returns
    assert ts.get_task(task["ID"])["task"] == task


def test_create_task_stores_supplied_fields(seeded_db):
    task = ts.get_task(seeded_db["ids"][2])["task"]
    assert task["Title"] == "Python Assignment"
    assert task["Category"] == "Study"
    assert task["Priority"] == "Medium"
    assert task["due_date"] == "2026-10-25"


def test_get_task_by_id(isolated_task_db):
    created = ts.create_task("Finish Project Report", "Work")
    fetched = ts.get_task(created["task_id"])

    assert fetched["success"] is True
    assert fetched["task"]["Title"] == "Finish Project Report"
    assert fetched["task"]["Category"] == "Work"


def test_get_task_missing_id(seeded_db):
    result = ts.get_task(9999)

    assert result["success"] is False
    assert result["status"] == "not_found"
    assert result["affected"] == 0


def test_find_tasks_by_title_is_case_insensitive(seeded_db):
    result = ts.find_tasks(title="python assignment")

    assert result["success"] is True
    assert result["ambiguous"] is False
    assert result["match_type"] == "exact"
    assert result["count"] == 1
    assert result["tasks"][0]["Title"] == "Python Assignment"


def test_find_tasks_falls_back_to_partial_matches(seeded_db):
    result = ts.find_tasks(title="assignment")

    assert result["success"] is True
    assert result["match_type"] == "partial"
    titles = {task["Title"] for task in result["tasks"]}
    assert titles == {"Prepare assignment", "Python Assignment"}


def test_find_tasks_unknown_title(isolated_task_db):
    ts.create_task("Only task")
    result = ts.find_tasks(title="does not exist")

    assert result["success"] is False
    assert result["status"] == "not_found"
    assert result["tasks"] == []


# --------------------------------------------------------------------------- #
# 4-7. listing: all / status / priority / category
# --------------------------------------------------------------------------- #


def test_list_all_tasks_includes_completed(seeded_db):
    result = ts.list_tasks()

    assert result["success"] is True
    assert result["filters_applied"] == {}
    assert result["count"] == 4
    statuses = {task["Status"] for task in result["tasks"]}
    assert statuses == {"Pending", "Completed"}
    assert result["counts_by_status"] == {"Pending": 3, "Completed": 1}


def test_list_pending_only(seeded_db):
    result = ts.list_tasks(status="Pending")

    assert result["count"] == 3
    assert all(task["Status"] == "Pending" for task in result["tasks"])
    assert result["filters_applied"] == {"status": "Pending"}


def test_list_completed_only(seeded_db):
    result = ts.list_tasks(status="Completed")

    assert result["count"] == 1
    assert result["tasks"][0]["Title"] == "Prepare assignment"


def test_list_by_priority_includes_completed_tasks(seeded_db):
    """'Show all high-priority tasks' must not silently drop completed ones."""
    result = ts.list_tasks(priority="High")

    assert {task["Priority"] for task in result["tasks"]} == {"High"}
    assert {task["Status"] for task in result["tasks"]} == {"Pending", "Completed"}
    assert result["filters_applied"] == {"priority": "High"}


def test_list_priority_is_case_insensitive(seeded_db):
    result = ts.list_tasks(priority="high")
    assert result["success"] is True
    assert result["filters_applied"] == {"priority": "High"}


def test_list_by_category_includes_completed_tasks(seeded_db):
    result = ts.list_tasks(category="Study")

    assert result["count"] == 2
    assert all(task["Category"] == "Study" for task in result["tasks"])
    assert {task["Status"] for task in result["tasks"]} == {"Pending", "Completed"}


def test_list_combined_filters(seeded_db):
    result = ts.list_tasks(status="Pending", category="Study")

    assert result["count"] == 1
    assert result["tasks"][0]["Title"] == "Python Assignment"
    assert result["filters_applied"] == {"status": "Pending", "category": "Study"}


def test_list_rejects_unknown_status(seeded_db):
    result = ts.list_tasks(status="Done")

    assert result["success"] is False
    assert result["status"] == "invalid_input"
    assert result["tasks"] == []


# --------------------------------------------------------------------------- #
# 8-9, 17. partial updates
# --------------------------------------------------------------------------- #


def test_update_description_only(seeded_db):
    task_id = seeded_db["ids"][1]
    before = ts.get_task(task_id)["task"]

    result = ts.update_task(task_id=task_id, description="Enhance the UI of Model X")

    assert result["success"] is True
    assert result["updated_fields"] == ["Description"]
    after = ts.get_task(task_id)["task"]
    assert after["Description"] == "Enhance the UI of Model X"
    # every other field is untouched
    for column in ("Title", "Category", "Priority", "due_date", "Status", "Created_at"):
        assert after[column] == before[column]


def test_update_priority_and_due_date_together(seeded_db):
    task_id = seeded_db["ids"][2]

    result = ts.update_task(task_id=task_id, priority="Medium", due_date="2026-10-28")

    assert result["success"] is True
    assert set(result["updated_fields"]) == {"Priority", "due_date"}
    task = ts.get_task(task_id)["task"]
    assert task["Priority"] == "Medium"
    assert task["due_date"] == "2026-10-28"
    assert task["Description"] == ""  # preserved


def test_update_preserves_unspecified_fields(seeded_db):
    task_id = seeded_db["ids"][0]
    before = ts.get_task(task_id)["task"]

    ts.update_task(task_id=task_id, priority="Low")

    after = ts.get_task(task_id)["task"]
    assert after["Priority"] == "Low"
    assert after["Title"] == before["Title"]
    assert after["Category"] == before["Category"]
    assert after["Description"] == before["Description"]
    assert after["due_date"] == before["due_date"]
    assert after["Status"] == before["Status"]


def test_update_can_rename_and_clear_due_date(seeded_db):
    task_id = seeded_db["ids"][3]

    result = ts.update_task(task_id=task_id, new_title="Grocery shopping", due_date="")

    assert result["success"] is True
    task = ts.get_task(task_id)["task"]
    assert task["Title"] == "Grocery shopping"
    assert task["due_date"] == ""


def test_update_without_fields_is_rejected(seeded_db):
    result = ts.update_task(task_id=seeded_db["ids"][0])

    assert result["success"] is False
    assert result["status"] == "invalid_input"


def test_update_by_title_when_title_is_unique(seeded_db):
    result = ts.update_task(title="Python Assignment", priority="Low")

    assert result["success"] is True
    assert result["task_id"] == seeded_db["ids"][2]
    assert ts.get_task(seeded_db["ids"][2])["task"]["Priority"] == "Low"


def test_update_by_title_reports_duplicate_titles(isolated_task_db):
    first = ts.create_task("Report", "Work")["task_id"]
    second = ts.create_task("Report", "Personal")["task_id"]

    result = ts.update_task(title="Report", priority="High")

    assert result["success"] is False
    assert result["status"] == "ambiguous"
    assert {candidate["ID"] for candidate in result["candidates"]} == {first, second}
    # nothing changed
    assert ts.get_task(first)["task"]["Priority"] == "Medium"
    assert ts.get_task(second)["task"]["Priority"] == "Medium"


def test_update_missing_task(seeded_db):
    result = ts.update_task(task_id=9999, priority="Low")

    assert result["success"] is False
    assert result["status"] == "not_found"
    assert result["affected"] == 0


def test_update_refuses_when_id_and_title_disagree(seeded_db):
    result = ts.update_task(
        task_id=seeded_db["ids"][1], title="Python Assignment", priority="Low"
    )

    assert result["success"] is False
    assert result["status"] == "conflict"
    assert ts.get_task(seeded_db["ids"][1])["task"]["Priority"] == "High"


# --------------------------------------------------------------------------- #
# 10. completion
# --------------------------------------------------------------------------- #


def test_complete_task(seeded_db):
    task_id = seeded_db["ids"][1]

    result = ts.complete_task(task_id=task_id)

    assert result["success"] is True
    assert result["affected"] == 1
    task = ts.get_task(task_id)["task"]
    assert task["Status"] == "Completed"
    assert task["Completion_at"]


def test_complete_task_is_idempotent_but_explicit(seeded_db):
    task_id = seeded_db["ids"][1]
    ts.complete_task(task_id=task_id)

    result = ts.complete_task(task_id=task_id)

    assert result["success"] is False
    assert result["status"] == "already_completed"
    assert result["affected"] == 0


def test_complete_task_by_title(seeded_db):
    result = ts.complete_task(title="Buy groceries")

    assert result["success"] is True
    assert ts.get_task(seeded_db["ids"][3])["task"]["Status"] == "Completed"


def test_complete_missing_task(seeded_db):
    result = ts.complete_task(task_id=9999)

    assert result["success"] is False
    assert result["status"] == "not_found"


# --------------------------------------------------------------------------- #
# 11. deletion
# --------------------------------------------------------------------------- #


def test_delete_task(seeded_db):
    task_id = seeded_db["ids"][3]

    result = ts.delete_task(task_id=task_id)

    assert result["success"] is True
    assert result["affected"] == 1
    assert result["title"] == "Buy groceries"
    assert ts.get_task(task_id)["status"] == "not_found"
    assert len(db_rows()) == 3


def test_delete_task_with_ambiguous_title_does_not_delete_anything(isolated_task_db):
    first = ts.create_task("Duplicate", "Work")["task_id"]
    second = ts.create_task("Duplicate", "Personal")["task_id"]

    result = ts.delete_task(title="Duplicate")

    assert result["success"] is False
    assert result["status"] == "ambiguous"
    assert ts.get_task(first)["success"] is True
    assert ts.get_task(second)["success"] is True


def test_delete_missing_task(seeded_db):
    result = ts.delete_task(task_id=9999)

    assert result["success"] is False
    assert result["status"] == "not_found"
    assert result["affected"] == 0
    assert len(db_rows()) == 4


# --------------------------------------------------------------------------- #
# 12-14. invalid input
# --------------------------------------------------------------------------- #


def test_create_rejects_invalid_priority(seeded_db):
    result = ts.create_task("Something", priority="Urgent")

    assert result["success"] is False
    assert result["status"] == "invalid_input"
    assert len(db_rows()) == 4


def test_create_rejects_invalid_category(seeded_db):
    result = ts.create_task("Something", category="Errands")
    assert result["success"] is False
    assert result["status"] == "invalid_input"


@pytest.mark.parametrize("bad_date", ["25-10-2026", "2026-10-25T00:00:00", "tomorrow", "2026-13-01", "2026-02-30"])
def test_create_rejects_invalid_dates(seeded_db, bad_date):
    result = ts.create_task("Something", due_date=bad_date)

    assert result["success"] is False
    assert result["status"] == "invalid_input"
    assert len(db_rows()) == 4


def test_create_rejects_blank_title(seeded_db):
    result = ts.create_task("   ")
    assert result["success"] is False
    assert result["status"] == "invalid_input"


def test_priority_normalisation(seeded_db):
    result = ts.create_task("Something", priority="high")
    assert result["success"] is True
    assert result["task"]["Priority"] == "High"


def test_update_rejects_invalid_date_and_keeps_old_value(seeded_db):
    task_id = seeded_db["ids"][2]

    result = ts.update_task(task_id=task_id, due_date="not a date")

    assert result["success"] is False
    assert result["status"] == "invalid_input"
    assert ts.get_task(task_id)["task"]["due_date"] == "2026-10-25"


def test_resolve_requires_some_identifier(seeded_db):
    result = ts.resolve_task_record(None, "")
    assert result["success"] is False
    assert result["status"] == "invalid_input"


# --------------------------------------------------------------------------- #
# 15. persistence across connections
# --------------------------------------------------------------------------- #


def test_data_persists_across_connections(seeded_db):
    first = ts.create_task("Persisted task", "Health", "", "Low", "2026-11-01")

    with sqlite3.connect(ts.resolve_db_path()) as connection:
        row = connection.execute(
            "SELECT Title, Status FROM tasks WHERE ID = ?", (first["task_id"],)
        ).fetchone()

    assert row is not None
    assert row[0] == "Persisted task"
    assert row[1] == "Pending"
    assert ts.get_task(first["task_id"])["success"] is True


def test_single_database_file_is_used(isolated_task_db):
    ts.create_task("Only one db")
    directory = isolated_task_db.parent

    assert directory.joinpath("tasks.db").exists()
    stray = [
        path.name
        for path in directory.iterdir()
        if path.name.startswith("tasks") and not path.name.startswith("tasks.db")
    ]
    assert stray == []


def test_schema_migration_preserves_existing_rows(isolated_task_db):
    """An older database gains missing columns without losing data."""
    with sqlite3.connect(isolated_task_db) as connection:
        connection.execute(
            """CREATE TABLE tasks (
                   ID INTEGER PRIMARY KEY AUTOINCREMENT,
                   Title TEXT NOT NULL
               )"""
        )
        connection.execute("INSERT INTO tasks (Title) VALUES ('Legacy task')")
        connection.commit()

    ts._MIGRATED.discard(str(isolated_task_db.resolve()))
    ts.init_db()

    with sqlite3.connect(isolated_task_db) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM tasks WHERE ID = 1").fetchone()

    assert row["Title"] == "Legacy task"
    assert row["Status"] == "Pending"          # added by the migration
    assert row["Priority"] == "Medium"         # added by the migration
    assert row["due_date"] == ""


def test_history_records_mutations(seeded_db):
    task_id = seeded_db["ids"][1]
    ts.update_task(task_id=task_id, priority="Low")

    with sqlite3.connect(ts.resolve_db_path()) as connection:
        events = [
            row[0]
            for row in connection.execute(
                "SELECT event_type FROM task_history WHERE task_id = ? ORDER BY event_id",
                (task_id,),
            )
        ]

    assert "create" in events
    assert "update" in events


# --------------------------------------------------------------------------- #
# 16. duplicate / missing title resolution
# --------------------------------------------------------------------------- #


def test_duplicate_titles_resolve_as_ambiguous(isolated_task_db):
    ts.create_task("Pay bill", "Finance")
    ts.create_task("Pay bill", "Personal")

    resolved = ts.resolve_task_record(None, "pay bill")

    assert resolved["success"] is False
    assert resolved["status"] == "ambiguous"
    assert len(resolved["candidates"]) == 2


def test_missing_title_returns_suggestions(seeded_db):
    resolved = ts.resolve_task_record(None, "pyton asignmnt")

    assert resolved["success"] is False
    assert resolved["status"] == "not_found"
    assert resolved["suggestions"]


def test_unique_title_resolves_to_one_task(seeded_db):
    resolved = ts.resolve_task_record(None, "Buy groceries")

    assert resolved["success"] is True
    assert resolved["task"]["ID"] == seeded_db["ids"][3]


def test_database_path_is_independent_of_cwd(isolated_task_db, tmp_path, monkeypatch):
    elsewhere = tmp_path / "somewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert ts.resolve_db_path() == isolated_task_db.resolve()
