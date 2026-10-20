"""Unit tests for the app-level helpers (no Streamlit runtime, no API calls)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.errors import GraphRecursionError

import app
import task_server as ts


def tool_message(name: str, payload, call_id: str = "call-1") -> ToolMessage:
    return ToolMessage(
        content=json.dumps(payload), name=name, tool_call_id=call_id
    )


# --------------------------------------------------------------------------- #
# message_text / extract_final_answer
# --------------------------------------------------------------------------- #


def test_message_text_handles_string_and_blocks():
    assert app.message_text(HumanMessage(content="  hello ")) == "hello"
    blocked = AIMessage(content=[{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}])
    assert app.message_text(blocked) == "part one\npart two"


def test_extract_final_answer_returns_last_assistant_text():
    messages = [
        HumanMessage(content="Show all tasks"),
        AIMessage(
            content="",
            tool_calls=[{"name": "list_tasks", "args": {}, "id": "c1", "type": "tool_call"}],
        ),
        tool_message("list_tasks", {"count": 0, "tasks": []}),
        AIMessage(content="You have no tasks."),
    ]

    assert app.extract_final_answer(messages) == "You have no tasks."


def test_extract_final_answer_refuses_to_return_a_tool_call():
    messages = [
        HumanMessage(content="Show all tasks"),
        AIMessage(
            content="",
            tool_calls=[{"name": "list_tasks", "args": {}, "id": "c1", "type": "tool_call"}],
        ),
    ]

    with pytest.raises(RuntimeError, match="still calling tools"):
        app.extract_final_answer(messages)


def test_extract_final_answer_requires_an_assistant_message():
    with pytest.raises(RuntimeError, match="no assistant message"):
        app.extract_final_answer([HumanMessage(content="hi")])


def test_extract_final_answer_uses_content_blocks():
    messages = [
        AIMessage(content=[{"type": "text", "text": "blocked answer"}]),
    ]
    assert app.extract_final_answer(messages) == "blocked answer"


# --------------------------------------------------------------------------- #
# collect_tool_results
# --------------------------------------------------------------------------- #


def test_collect_tool_results_parses_json_payloads():
    messages = [
        HumanMessage(content="go"),
        AIMessage(
            content="",
            tool_calls=[{"name": "get_task", "args": {"task_id": 4}, "id": "c1", "type": "tool_call"}],
        ),
        tool_message("get_task", {"success": True, "task": {"ID": 4}}, call_id="c1"),
        tool_message("broken", "not json at all"),
    ]

    results = app.collect_tool_results(messages)

    assert results[0]["tool"] == "get_task"
    assert results[0]["args"] == {"task_id": 4}
    assert results[0]["result"]["success"] is True
    assert results[1]["result"] is None
    assert "not json at all" in results[1]["raw"]


# --------------------------------------------------------------------------- #
# verify_against_db
# --------------------------------------------------------------------------- #


class FakeTaskModule:
    def __init__(self, rows):
        self.rows = rows

    def get_task(self, task_id):
        if task_id in self.rows:
            return {"success": True, "task_id": task_id, "task": self.rows[task_id]}
        return {"success": False, "status": "not_found", "task_id": task_id}


def test_verify_accepts_a_correct_mutation():
    module = FakeTaskModule({4: {"ID": 4, "Title": "T", "Priority": "Medium"}})
    results = [
        {
            "tool": "update_task",
            "result": {
                "success": True,
                "task_id": 4,
                "task": {"ID": 4, "Title": "T", "Priority": "Medium"},
            },
        }
    ]

    assert app.verify_against_db(results, task_module=module) == []


def test_verify_flags_a_create_that_is_not_in_the_database():
    module = FakeTaskModule({})
    results = [
        {
            "tool": "create_task",
            "result": {"success": True, "task_id": 42, "task": {"ID": 42}},
        }
    ]

    notes = app.verify_against_db(results, task_module=module)

    assert notes and "create_task" in notes[0] and "42" in notes[0]


def test_verify_flags_a_delete_that_left_the_row_in_place():
    module = FakeTaskModule({7: {"ID": 7, "Title": "still here"}})
    results = [
        {"tool": "delete_task", "result": {"success": True, "task_id": 7}}
    ]

    notes = app.verify_against_db(results, task_module=module)

    assert notes and "still" in notes[0]


def test_verify_flags_a_mismatched_column():
    module = FakeTaskModule({5: {"ID": 5, "Priority": "High"}})
    results = [
        {
            "tool": "update_task",
            "result": {
                "success": True,
                "task_id": 5,
                "task": {"ID": 5, "Priority": "Medium"},
            },
        }
    ]

    notes = app.verify_against_db(results, task_module=module)

    assert notes and "Priority" in notes[0]


def test_verify_ignores_failures_and_read_only_tools():
    module = FakeTaskModule({})
    results = [
        {"tool": "update_task", "result": {"success": False, "message": "ambiguous"}},
        {"tool": "list_tasks", "result": {"success": True, "count": 3}},
        {"tool": "get_task", "result": {"success": True, "task": {"ID": 1}}},
    ]

    assert app.verify_against_db(results, task_module=module) == []


def test_verify_reports_success_without_task_id():
    results = [{"tool": "create_task", "result": {"success": True}}]

    notes = app.verify_against_db(results, task_module=FakeTaskModule({}))

    assert notes and "task_id" in notes[0]


def test_verify_against_the_real_database_after_real_calls(seeded_db):
    """End-to-end: tool payloads produced by task_server always verify clean."""
    created = ts.create_task("Verify me", "Work", "", "High", "2026-10-30")
    updated = ts.update_task(task_id=created["task_id"], priority="Low")
    completed = ts.complete_task(task_id=created["task_id"])
    listed = ts.list_tasks()

    results = [
        {"tool": "create_task", "result": created},
        {"tool": "update_task", "result": updated},
        {"tool": "complete_task", "result": completed},
        {"tool": "list_tasks", "result": listed},
    ]
    assert app.verify_against_db(results) == []

    deleted = ts.delete_task(task_id=created["task_id"])
    assert app.verify_against_db([{"tool": "delete_task", "result": deleted}]) == []


# --------------------------------------------------------------------------- #
# errors and secrets
# --------------------------------------------------------------------------- #


def test_describe_error_for_recursion_limit():
    message = app.describe_error(GraphRecursionError("recursion"), "abc12345")

    assert str(app.RECURSION_LIMIT) in message
    assert "abc12345" in message
    assert "smaller request" in message


def test_describe_error_for_timeout():
    message = app.describe_error(TimeoutError("too slow"), "abc12345")

    assert "timed out" in message
    assert "abc12345" in message


def test_describe_error_for_unexpected_exception_has_no_secrets(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "super-secret-key")
    message = app.describe_error(
        RuntimeError("boom with super-secret-key inside"), "abc12345"
    )

    assert "super-secret-key" not in message
    assert "RuntimeError" in message
    assert "abc12345" in message


def test_scrub_removes_the_api_key(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-secret-value")

    assert app.scrub("error at sk-secret-value end") == "error at ***REDACTED*** end"


# --------------------------------------------------------------------------- #
# system prompt
# --------------------------------------------------------------------------- #


def test_system_prompt_uses_todays_local_date():
    prompt = app.get_system_prompt()
    today = datetime.now().astimezone().date().isoformat()

    assert today in prompt
    assert "list_tasks() with NO arguments" in prompt
    assert "ambiguous" in prompt


# --------------------------------------------------------------------------- #
# process plumbing (regression tests for the real failures that were fixed)
# --------------------------------------------------------------------------- #


def test_mcp_connection_pins_the_database_path(isolated_task_db):
    """The stdio child must be told which database to open (see load_mcp_tools)."""
    config = app.build_mcp_connection()["task_manager"]

    assert config["transport"] == "stdio"
    assert config["args"][-2] == "--db"
    assert config["args"][-1] == str(isolated_task_db.resolve())


def test_run_coroutine_sync_reuses_a_single_event_loop():
    """Cached clients must not be re-bound to a fresh (and later closed) loop."""
    loops = []

    async def probe():
        loop = asyncio.get_running_loop()
        loops.append(loop)
        return loop

    first = app.run_coroutine_sync(probe())
    second = app.run_coroutine_sync(probe())

    assert first is second
    assert not first.is_closed()
