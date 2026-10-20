"""Agent-level tests.

A deterministic scripted chat model drives the *real* LangGraph agent and the
*real* MCP stdio server against a temporary SQLite database, so the whole
message-conversion / tool-execution / final-answer path is covered without a
single paid Mistral API call.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_mistralai import ChatMistralAI  # noqa: F401  (import guard)
from langchain_mcp_adapters.client import MultiServerMCPClient

import app
import task_server as ts

ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "task_server.py"


class ScriptedModel(BaseChatModel):
    """Replays a fixed script of assistant messages; records what it saw."""

    script: list[Any]
    seen_messages: list[Any] = []
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self

    def _generate(
        self,
        messages: list[Any],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen_messages.append([message.type for message in messages])
        step = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        message = step(messages) if callable(step) else step
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def last_tool_payload(messages: list[Any]) -> dict[str, Any]:
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            return json.loads(app.message_text(message))
    raise AssertionError("no ToolMessage found in the scripted conversation")


def load_tools(db_path: Path) -> list[Any]:
    """Start the MCP server on stdio, pointed at the temporary database."""

    async def _load() -> list[Any]:
        client = MultiServerMCPClient(
            {
                "task_manager": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [str(SERVER_PATH), "--db", str(db_path)],
                    "cwd": str(ROOT),
                    "env": {**os.environ, "TASK_DB_PATH": str(db_path)},
                }
            }
        )
        return await asyncio.wait_for(client.get_tools(), timeout=60)

    return asyncio.run(_load())


def run_turn(model: ScriptedModel, history: list[dict[str, Any]], db_path: Path):
    tools = load_tools(db_path)
    return app.process_message(history, tools=tools, model=model)


# --------------------------------------------------------------------------- #
# Listing behaviour
# --------------------------------------------------------------------------- #


def test_show_all_tasks_returns_pending_and_completed(seeded_db):
    """'Show all tasks' must not silently add a pending-only filter."""
    db_path = seeded_db["path"]

    def build_answer(messages: list[Any]) -> AIMessage:
        payload = last_tool_payload(messages)
        lines = [
            f"- {task['ID']} {task['Title']} ({task['Status']})"
            for task in payload["tasks"]
        ]
        return AIMessage(content=f"{payload['count']} tasks:\n" + "\n".join(lines))

    model = ScriptedModel(
        script=[tool_call("list_tasks", {}, "call-1"), build_answer]
    )

    answer, notes = run_turn(
        model, [{"role": "user", "content": "Show me all my tasks."}], db_path
    )

    assert model.index == 2  # exactly two model calls: tool call + answer
    assert "Prepare assignment" in answer and "Completed" in answer
    assert "Python Assignment" in answer and "Pending" in answer
    assert notes == []
    # the database was not modified by a read-only request
    assert ts.list_tasks()["count"] == 4


def test_show_pending_tasks_filters_by_status(seeded_db):
    db_path = seeded_db["path"]

    def build_answer(messages: list[Any]) -> AIMessage:
        payload = last_tool_payload(messages)
        assert payload["filters_applied"] == {"status": "Pending"}
        titles = ", ".join(task["Title"] for task in payload["tasks"])
        return AIMessage(content=f"Pending: {titles}")

    model = ScriptedModel(
        script=[tool_call("list_tasks", {"status": "Pending"}, "call-1"), build_answer]
    )

    answer, notes = run_turn(
        model, [{"role": "user", "content": "Show all pending tasks."}], db_path
    )

    assert "Prepare assignment" not in answer  # completed -> excluded
    assert "Python Assignment" in answer
    assert notes == []


def test_medium_priority_listing_keeps_completed_tasks(seeded_db):
    db_path = seeded_db["path"]

    def build_answer(messages: list[Any]) -> AIMessage:
        payload = last_tool_payload(messages)
        assert payload["filters_applied"] == {"priority": "Medium"}
        return AIMessage(
            content=", ".join(
                f"{task['ID']} {task['Title']} {task['Status']}"
                for task in payload["tasks"]
            )
        )

    model = ScriptedModel(
        script=[tool_call("list_tasks", {"priority": "Medium"}, "call-1"), build_answer]
    )

    answer, _ = run_turn(
        model, [{"role": "user", "content": "Show all medium-priority tasks."}], db_path
    )

    assert "Python Assignment" in answer  # pending medium
    assert "Prepare assignment" not in answer  # that one is High


# --------------------------------------------------------------------------- #
# Mutations: the record that was identified is the record that changes
# --------------------------------------------------------------------------- #


def test_update_by_resolved_title_changes_only_that_task(seeded_db):
    db_path = seeded_db["path"]
    target = seeded_db["ids"][2]
    other = seeded_db["ids"][1]

    def next_step(messages: list[Any]) -> AIMessage:
        payload = last_tool_payload(messages)
        assert payload["ambiguous"] is False
        task_id = payload["tasks"][0]["ID"]
        assert task_id == target
        return tool_call(
            "update_task",
            {"task_id": task_id, "priority": "Medium", "due_date": "2026-10-28"},
            "call-2",
        )

    def final_step(messages: list[Any]) -> AIMessage:
        payload = last_tool_payload(messages)
        assert payload["success"] is True
        return AIMessage(
            content=f"Updated task {payload['task_id']} to Medium due "
            f"{payload['task']['due_date']}."
        )

    model = ScriptedModel(
        script=[
            tool_call("find_tasks", {"title": "Python Assignment"}, "call-1"),
            next_step,
            final_step,
        ]
    )

    answer, notes = run_turn(
        model,
        [{"role": "user", "content": "Change Python Assignment to Medium priority "
                                     "and move its due date to October 28."}],
        db_path,
    )

    updated = ts.get_task(target)["task"]
    assert updated["Priority"] == "Medium"
    assert updated["due_date"] == "2026-10-28"
    assert updated["Description"] == ""  # unspecified fields preserved
    assert updated["Title"] == "Python Assignment"
    assert ts.get_task(other)["task"]["Priority"] == "High"  # untouched
    assert "2026-10-28" in answer
    assert notes == []


def test_create_then_read_back_matches_database(seeded_db):
    db_path = seeded_db["path"]

    def next_step(messages: list[Any]) -> AIMessage:
        return tool_call(
            "create_task",
            {
                "title": "ModelX UI Enhancement",
                "category": "Work",
                "priority": "High",
                "due_date": "2026-10-25",
                "description": "Enhance the UI of Model X",
            },
            "call-2",
        )

    def final_step(messages: list[Any]) -> AIMessage:
        payload = last_tool_payload(messages)
        return AIMessage(content=f"Created task {payload['task_id']}.")

    model = ScriptedModel(
        script=[tool_call("list_tasks", {}, "call-1"), next_step, final_step]
    )

    answer, notes = run_turn(
        model,
        [{"role": "user", "content": "Add a high-priority Work task called "
                                     "ModelX UI Enhancement due October 25."}],
        db_path,
    )

    stored = [task for task in ts.list_tasks()["tasks"] if task["Title"] == "ModelX UI Enhancement"]
    assert len(stored) == 1
    assert stored[0]["Category"] == "Work"
    assert stored[0]["Description"] == "Enhance the UI of Model X"
    assert notes == []
    assert "Created task" in answer


def test_ambiguous_title_never_deletes_a_task(isolated_task_db):
    db_path = isolated_task_db
    ts.init_db()
    first = ts.create_task("ModelX UI Enhancement", "Work")["task_id"]
    second = ts.create_task("ModelX UI Enhancement", "Personal")["task_id"]

    def ask_user(messages: list[Any]) -> AIMessage:
        payload = last_tool_payload(messages)
        assert payload["ambiguous"] is True
        candidates = ", ".join(str(item["ID"]) for item in payload["tasks"])
        return AIMessage(
            content=f"Two tasks share that name (IDs {candidates}). Which one?"
        )

    model = ScriptedModel(
        script=[tool_call("find_tasks", {"title": "ModelX UI Enhancement"}, "call-1"), ask_user]
    )

    answer, notes = run_turn(
        model,
        [{"role": "user", "content": "Delete ModelX UI Enhancement."}],
        db_path,
    )

    assert "Which one?" in answer
    assert notes == []
    assert ts.get_task(first)["success"] is True
    assert ts.get_task(second)["success"] is True


def test_follow_up_answer_is_seen_with_full_context(seeded_db):
    """The model must receive the earlier turns when the user answers a question."""
    db_path = seeded_db["path"]
    answered_id = seeded_db["ids"][2]

    def apply_update(messages: list[Any]) -> AIMessage:
        payload = last_tool_payload(messages)
        assert payload["success"] is True
        return AIMessage(content=f"Task {payload['task_id']} updated.")

    model = ScriptedModel(
        script=[
            tool_call("find_tasks", {"title": "assignment"}, "call-1"),
            tool_call("update_task", {"task_id": answered_id, "priority": "High"}, "call-3"),
            apply_update,
        ]
    )

    history = [
        {"role": "user", "content": "Change an assignment to High priority."},
        {"role": "assistant", "content": "Which task, 1 or 2?"},
        {"role": "user", "content": str(answered_id)},
    ]

    answer, notes = run_turn(model, history, db_path)

    update_call_seen = model.seen_messages[1]  # the turn that issued update_task
    # the model saw the whole conversation: the original request, its own
    # earlier question, the numeric follow-up, and the latest tool result
    assert update_call_seen.count("human") == 2
    assert update_call_seen[-1] == "tool"
    assert "ai" in update_call_seen
    assert ts.get_task(answered_id)["task"]["Priority"] == "High"
    assert "updated." in answer
    assert notes == []


def test_tool_failure_is_reported_not_hidden(isolated_task_db):
    db_path = isolated_task_db
    ts.init_db()
    ts.create_task("ModelX UI Enhancement", "Work")
    ts.create_task("ModelX UI Enhancement", "Personal")

    def final_step(messages: list[Any]) -> AIMessage:
        payload = last_tool_payload(messages)
        assert payload["success"] is False
        assert payload["status"] == "ambiguous"
        return AIMessage(
            content="I found 2 tasks with that title; tell me which ID to delete."
        )

    model = ScriptedModel(
        script=[
            tool_call("delete_task", {"title": "ModelX UI Enhancement"}, "call-1"),
            final_step,
        ]
    )

    answer, notes = run_turn(
        model, [{"role": "user", "content": "Delete the ModelX task."}], db_path
    )

    assert "which ID" in answer
    assert notes == []
    assert ts.list_tasks()["count"] == 2  # nothing was deleted
