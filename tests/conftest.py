"""Shared fixtures.

Every test runs against a private temporary database.  ``TASK_DB_PATH`` is
autouse, so even a test that forgets an explicit path can never touch the real
``tasks.db``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SERVER_PATH = ROOT / "task_server.py"


@pytest.fixture(autouse=True)
def isolated_task_db(tmp_path, monkeypatch):
    """Point the task database at a throwaway file for the whole test."""
    import task_server

    db_file = tmp_path / "tasks.db"
    monkeypatch.setenv("TASK_DB_PATH", str(db_file))
    monkeypatch.setattr(task_server, "_CLI_DB_PATH", None)
    yield db_file


@pytest.fixture()
def seeded_db(isolated_task_db):
    """A database with a known mix of pending and completed tasks."""
    import task_server as ts

    ts.init_db()
    created = [
        ts.create_task(
            "Prepare assignment",
            "Study",
            "Revise chapter 4",
            "High",
            "2026-10-25",
        ),
        ts.create_task("Finish Project Report", "Work", "", "High", "2026-10-25"),
        ts.create_task("Python Assignment", "Study", "", "Medium", "2026-10-25"),
        ts.create_task("Buy groceries", "Shopping", "", "Low", ""),
    ]
    ids = [item["task_id"] for item in created]
    ts.complete_task(task_id=ids[0])
    return {"path": isolated_task_db, "ids": ids}
