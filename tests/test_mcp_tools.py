"""MCP-layer tests.

These exercise the tools the agent actually sees, through FastMCP, including
argument validation (test 18 of the brief).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from mcp.server.fastmcp.exceptions import ToolError

import task_server as ts

EXPECTED_TOOLS = {
    "create_task",
    "list_tasks",
    "get_task",
    "find_tasks",
    "update_task",
    "complete_task",
    "delete_task",
}


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call a tool exactly like an MCP client would and decode the payload."""
    result = asyncio.run(ts.mcp.call_tool(name, arguments))
    blocks = result[0] if isinstance(result, tuple) else result
    texts = [block.text for block in blocks if getattr(block, "text", None)]
    assert texts, f"{name} returned no content"
    return json.loads(texts[0])


def test_all_expected_tools_are_exposed():
    tools = asyncio.run(ts.mcp.list_tools())
    names = {tool.name for tool in tools}

    assert EXPECTED_TOOLS <= names
    assert all(tool.description for tool in tools)


def test_tool_schemas_expose_the_expected_arguments():
    tools = {tool.name: tool for tool in asyncio.run(ts.mcp.list_tools())}

    assert set(tools["create_task"].inputSchema["properties"]) == {
        "title",
        "category",
        "description",
        "priority",
        "due_date",
    }
    assert set(tools["update_task"].inputSchema["properties"]) == {
        "task_id",
        "title",
        "new_title",
        "category",
        "description",
        "priority",
        "due_date",
    }
    assert set(tools["list_tasks"].inputSchema["properties"]) == {
        "status",
        "category",
        "priority",
    }
    # title is the only required argument anywhere
    assert tools["create_task"].inputSchema["required"] == ["title"]


def test_create_list_update_complete_delete_over_mcp(isolated_task_db):
    created = call_tool(
        "create_task",
        {
            "title": "ModelX UI Enhancement",
            "category": "Work",
            "priority": "Medium",
            "due_date": "2026-10-26",
        },
    )
    assert created["success"] is True
    task_id = created["task_id"]

    listed = call_tool("list_tasks", {})
    assert listed["count"] == 1
    assert listed["filters_applied"] == {}

    updated = call_tool(
        "update_task", {"task_id": task_id, "priority": "High", "due_date": "2026-10-28"}
    )
    assert updated["success"] is True
    assert ts.get_task(task_id)["task"]["due_date"] == "2026-10-28"

    completed = call_tool("complete_task", {"task_id": task_id})
    assert completed["success"] is True
    assert completed["task"]["Status"] == "Completed"

    deleted = call_tool("delete_task", {"task_id": task_id})
    assert deleted["success"] is True
    assert ts.get_task(task_id)["status"] == "not_found"


def test_find_and_get_over_mcp(seeded_db):
    found = call_tool("find_tasks", {"title": "Python Assignment"})
    assert found["success"] is True
    assert found["count"] == 1

    fetched = call_tool("get_task", {"task_id": found["tasks"][0]["ID"]})
    assert fetched["task"]["Title"] == "Python Assignment"


def test_list_never_silently_filters_by_status(seeded_db):
    """'Show all tasks' must return completed records too."""
    result = call_tool("list_tasks", {})

    assert result["count"] == 4
    assert result["counts_by_status"] == {"Pending": 3, "Completed": 1}


def test_empty_title_is_rejected(seeded_db):
    result = call_tool("create_task", {"title": ""})

    assert result["success"] is False
    assert result["status"] == "invalid_input"
    assert len(call_tool("list_tasks", {})["tasks"]) == 4


def test_wrong_argument_types_are_rejected(seeded_db):
    with pytest.raises(ToolError):
        call_tool("get_task", {"task_id": "not-an-int"})

    with pytest.raises(ToolError):
        call_tool("create_task", {"title": 123})

    with pytest.raises(ToolError):
        call_tool("list_tasks", {"status": 5})


def test_unknown_tool_is_rejected():
    with pytest.raises(ToolError):
        call_tool("drop_everything", {})


def test_mutations_never_target_another_record_over_mcp(seeded_db):
    other = seeded_db["ids"][2]

    result = call_tool(
        "update_task", {"task_id": seeded_db["ids"][1], "title": "Python Assignment"}
    )

    assert result["success"] is False
    assert result["status"] == "conflict"
    assert ts.get_task(other)["task"]["Priority"] == "Medium"
