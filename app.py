"""Streamlit front-end for the AI-Powered Personal Task Manager.

Responsibilities
----------------
* Owns the chat UI and the conversation history for the Streamlit session.
* Talks to Mistral through LangChain/LangGraph and lets the agent call the
  MCP tools exposed by ``task_server.py``.
* Keeps the MCP tool layer cached for the lifetime of the process instead of
  starting a new MCP handshake for every message.
* Extracts the *final assistant message* (never an intermediate tool call),
  verifies reported mutations against the real SQLite database, and surfaces
  actionable - but secret-free - errors while full tracebacks go to the log.

Everything interesting is a plain function so it can be unit tested without a
Streamlit runtime or a paid Mistral call; the UI lives in ``main()``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Sequence

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
    convert_to_messages,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mistralai import ChatMistralAI
from langgraph.errors import GraphRecursionError

try:  # langchain >= 1.0 (installed version: 1.4.x)
    from langchain.agents import create_agent
except ImportError:  # pragma: no cover - fallback for older installs
    from langgraph.prebuilt import create_react_agent as _create_react_agent

    def create_agent(model, tools=None, *, system_prompt=None, **kwargs):  # type: ignore[misc]
        return _create_react_agent(model, tools, prompt=system_prompt, **kwargs)

import task_server  # noqa: E402  (project module: shared DB path + verification)

BASE_DIR = Path(__file__).resolve().parent
SERVER_PATH = BASE_DIR / "task_server.py"
LOG_PATH = BASE_DIR / "app.log"

MODEL_NAME = os.environ.get("TASK_MANAGER_MODEL", "ministral-3b-2512")
RECURSION_LIMIT = int(os.environ.get("TASK_MANAGER_RECURSION_LIMIT", "30"))
TOOL_LOAD_TIMEOUT_SECONDS = float(os.environ.get("TASK_MANAGER_TOOL_TIMEOUT", "60"))
AGENT_TIMEOUT_SECONDS = float(os.environ.get("TASK_MANAGER_AGENT_TIMEOUT", "120"))
LLM_TIMEOUT_SECONDS = float(os.environ.get("TASK_MANAGER_LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.environ.get("TASK_MANAGER_LLM_RETRIES", "3"))

MUTATING_TOOLS = {"create_task", "update_task", "complete_task", "delete_task"}

load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("MISTRAL_API_KEY")

logger = logging.getLogger("task_manager")


# --------------------------------------------------------------------------- #
# Logging / secrets
# --------------------------------------------------------------------------- #


class _ScrubbingFilter(logging.Filter):
    """Strip the API key from anything that goes to a log handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = scrub(record.getMessage())
            record.args = ()
        except Exception:  # pragma: no cover - never break logging
            pass
        return True


def configure_logging() -> None:
    """Log to stderr *and* app.log so tracebacks survive past the terminal."""
    logger.setLevel(logging.INFO)
    if logger.handlers:  # already configured
        return
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    scrub_filter = _ScrubbingFilter()
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(scrub_filter)
    file_handler = RotatingFileHandler(
        LOG_PATH, maxBytes=1_000_000, backupCount=2, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(scrub_filter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False


def scrub(text: str) -> str:
    """Remove the API key from any text that may be logged or shown."""
    key = os.getenv("MISTRAL_API_KEY") or ""
    if key and key in text:
        text = text.replace(key, "***REDACTED***")
    return text


def describe_error(exc: BaseException, error_id: str) -> str:
    """User-facing error text: safe, specific, and traceable in the log."""
    if isinstance(exc, GraphRecursionError):
        return (
            f"⚠ The request needed more than {RECURSION_LIMIT} steps and was stopped "
            "before it finished. Please retry with a single, smaller request "
            f"(reference {error_id})."
        )
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return (
            f"⚠ The request timed out after {AGENT_TIMEOUT_SECONDS:g}s and was not "
            "retried automatically, so no partial change was repeated "
            f"(reference {error_id})."
        )
    return (
        f"⚠ Something went wrong while handling this request "
        f"({type(exc).__name__}). The full traceback is in {LOG_PATH.name} "
        f"with reference {error_id}."
    )


# --------------------------------------------------------------------------- #
# Message helpers (pure functions, unit tested)
# --------------------------------------------------------------------------- #


def message_text(message: BaseMessage) -> str:
    """Flatten a message's content (string or content blocks) to plain text."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def extract_final_answer(messages: Sequence[BaseMessage]) -> str:
    """Return the last *assistant* answer, never a tool call or tool output."""
    for message in reversed(list(messages)):
        if not isinstance(message, AIMessage):
            continue
        if message.tool_calls:
            names = ", ".join(
                str(call.get("name", "?")) for call in message.tool_calls
            )
            raise RuntimeError(
                "The agent stopped while it was still calling tools "
                f"({names}); there is no final answer to show."
            )
        text = message_text(message)
        if text:
            return text
    raise RuntimeError("The agent returned no assistant message.")


def _tool_call_args(message: ToolMessage) -> dict[str, Any]:
    """Tool call arguments, if the installed langchain version keeps them."""
    args = getattr(message, "tool_call_args", None)
    return dict(args) if isinstance(args, dict) else {}


def collect_tool_results(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """Extract every tool call + parsed JSON payload from a run."""
    collected: list[dict[str, Any]] = []
    args_by_id: dict[str, dict[str, Any]] = {}
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                call_id = str(call.get("id") or "")
                args = call.get("args")
                if call_id:
                    args_by_id[call_id] = (
                        dict(args) if isinstance(args, dict) else {}
                    )
            continue
        if not isinstance(message, ToolMessage):
            continue
        raw = message_text(message)
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            payload = None
        args = _tool_call_args(message) or args_by_id.get(
            str(message.tool_call_id), {}
        )
        collected.append(
            {
                "tool": message.name or "",
                "args": args,
                "result": payload if isinstance(payload, dict) else None,
                "raw": raw,
            }
        )
    return collected


def verify_against_db(
    tool_results: Sequence[dict[str, Any]],
    task_module: Any = task_server,
) -> list[str]:
    """Re-read the database and flag any mutation the tool layer got wrong.

    Mutations are replayed in order (a task created and then updated in the
    same run is compared at its final state) and the resulting expectations are
    checked against the real database. This is the deterministic backstop
    behind the model: a tool result that claims success must match what is
    actually stored.
    """
    problems: list[str] = []
    expected: dict[int, tuple[str, dict[str, Any]]] = {}
    deleted: dict[int, str] = {}

    for item in tool_results:
        name = item.get("tool", "")
        payload = item.get("result")
        if name not in MUTATING_TOOLS or not isinstance(payload, dict):
            continue
        if not payload.get("success"):
            logger.info("Tool %s reported a failure: %s", name, payload.get("message"))
            continue

        task_id = payload.get("task_id")
        if not isinstance(task_id, int):
            problems.append(f"{name} reported success without a task_id to verify.")
            continue

        if name == "delete_task":
            expected.pop(task_id, None)
            deleted[task_id] = name
            continue

        reported = payload.get("task")
        if isinstance(reported, dict):
            expected[task_id] = (name, reported)
            deleted.pop(task_id, None)
        else:
            problems.append(
                f"{name} reported success for task {task_id} without a record to verify."
            )

    for task_id, (name, reported) in expected.items():
        check = task_module.get_task(task_id)
        if not check.get("success"):
            problems.append(
                f"{name} reported task {task_id}, but it is not in the database."
            )
            continue
        stored = check.get("task") or {}
        for column, value in reported.items():
            if column in stored and stored[column] != value:
                problems.append(
                    f"{name} reported task {task_id} column {column}={value!r} but "
                    f"the database holds {stored[column]!r}."
                )

    for task_id, name in deleted.items():
        if task_module.get_task(task_id).get("success"):
            problems.append(
                f"{name} reported task {task_id} deleted, but it is still in the "
                "database."
            )
    return problems


# --------------------------------------------------------------------------- #
# Agent plumbing
# --------------------------------------------------------------------------- #


def get_system_prompt() -> str:
    """System prompt built from the *local* date at call time (never hardcoded)."""
    today = datetime.now().astimezone().date().isoformat()
    categories = ", ".join(task_server.CATEGORIES)
    return f"""You are an AI-powered personal task manager.
Today's date is {today} (local time).

The MCP tools are the only source of truth about tasks. Never invent task
records, IDs or tool results. Never claim success unless a tool returned
success=true; if a tool returns success=false, say so and quote its message.

LISTING - apply exactly the filter the user asked for, never add one:
- "show all tasks" / "show everything" -> list_tasks() with NO arguments
  (it returns pending AND completed tasks).
- "show my pending tasks" -> list_tasks(status="Pending").
- "show completed tasks" -> list_tasks(status="Completed").
- "show medium-priority tasks" -> list_tasks(priority="Medium") and NO status
  filter (completed tasks are included unless the user also said
  pending/completed).
- "show my Study tasks" -> list_tasks(category="Study") and NO status filter.
- Combine two filters only when the user asked for two dimensions, e.g.
  "pending Work tasks" -> list_tasks(status="Pending", category="Work").

CREATION - create_task(title, category, description, priority, due_date):
infer category ({categories}) and priority (Low/Medium/High) when they are
clear from the request; only ask when a required detail is genuinely missing
(usually just the title). Convert relative dates ("tomorrow", "next Monday")
into YYYY-MM-DD using today's date above.

TASK IDENTIFICATION - required before any update, completion, deletion or
"what is X?" lookup:
1. If the user gave a task ID, use it (get_task first when unsure).
2. Otherwise call find_tasks(title="...").
   - exactly one match -> act on that ID immediately.
   - ambiguous -> ask the user which one, listing ID, title, status and due
     date for each candidate, then stop.
   - not found -> say so and show any suggestions; never create a
     replacement task.
3. update_task/complete_task/delete_task accept task_id and/or title; when you
   pass both they must describe the same task.

UPDATES - send only the fields the user asked to change. Omitted fields are
preserved, so adding a description must not resend priority or due date.
Several changes go in one update_task call.

FOLLOW-UPS - use the whole conversation. If you previously asked for a task
ID and the user answers with a number, that number is the answer: continue
the original operation instead of asking again or starting over.

ERRORS AND SAFETY - never expose API keys. Never retry a create/update/delete
that failed with an uncertain result. Completing is not deleting.

RESPONSE - be concise. For listings include ID, title, category, priority,
due date and status for every task."""


class _EventLoopRunner:
    """Runs coroutines on one long-lived background event loop.

    Why not ``asyncio.run()`` per message?  The cached Mistral client (httpx)
    and the MCP adapter keep connections bound to the loop that first used
    them; creating and destroying a loop per message is exactly what produced
    the intermittent "Event loop is closed" failures.  One process-wide loop
    also works when the calling thread already has a loop of its own.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="agent-loop"
        )
        self._thread.start()

    def run(self, coro: Any, timeout: float | None = None) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout)
        except Exception:
            future.cancel()
            raise


_RUNNER: _EventLoopRunner | None = None


def _get_runner() -> _EventLoopRunner:
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = _EventLoopRunner()
    return _RUNNER


def run_coroutine_sync(coro: Any, timeout: float | None = None) -> Any:
    """Run a coroutine from synchronous Streamlit code, safely."""
    return _get_runner().run(coro, timeout=timeout)


def build_mcp_connection(server_path: Path | None = None) -> dict[str, Any]:
    """MCP stdio connection config with the database pinned via ``--db``.

    The MCP stdio transport only hands the child a filtered subset of the
    environment, so an env-only override would silently fall back to the
    default database - the database path must be passed on the command line.
    """
    path = server_path or SERVER_PATH
    return {
        "task_manager": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(path), "--db", str(task_server.resolve_db_path())],
            "cwd": str(path.parent or BASE_DIR),
        }
    }


async def load_mcp_tools(server_path: Path | None = None) -> list[Any]:
    """Start the MCP server on stdio and load its tools (with a timeout)."""
    client = MultiServerMCPClient(build_mcp_connection(server_path))
    return await asyncio.wait_for(
        client.get_tools(), timeout=TOOL_LOAD_TIMEOUT_SECONDS
    )


async def invoke_agent(agent: Any, messages: Sequence[BaseMessage]) -> dict[str, Any]:
    """Invoke the agent with a recursion limit and an overall timeout."""
    return await asyncio.wait_for(
        agent.ainvoke(
            {"messages": list(messages)},
            config={"recursion_limit": RECURSION_LIMIT},
        ),
        timeout=AGENT_TIMEOUT_SECONDS,
    )


def build_agent(tools: Sequence[Any], model: Any) -> Any:
    """Build the ReAct agent with the current date baked into the prompt."""
    return create_agent(model, tools=list(tools), system_prompt=get_system_prompt())


def load_tools_or_raise() -> list[Any]:
    try:
        tools = run_coroutine_sync(
            load_mcp_tools(), timeout=TOOL_LOAD_TIMEOUT_SECONDS + 30
        )
    except Exception as exc:  # noqa: BLE001 - converted into a user-facing error
        raise RuntimeError(
            f"The MCP task server at {SERVER_PATH.name} did not start or exposed "
            "no tools. Run `python task_server.py --help` in a terminal to see the "
            "server-side error."
        ) from exc
    if not tools:
        raise RuntimeError("The MCP server exposes no tools.")
    return tools


# Cached per Streamlit process: one MCP handshake instead of one per message.
@st.cache_resource(show_spinner=False)
def get_tools_cached() -> list[Any]:
    return load_tools_or_raise()


@st.cache_resource(show_spinner=False)
def get_model_cached() -> Any:
    return ChatMistralAI(
        model=MODEL_NAME,
        api_key=API_KEY,
        temperature=0,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )


def process_message(
    history: Sequence[dict[str, Any]],
    *,
    tools: Sequence[Any] | None = None,
    model: Any | None = None,
) -> tuple[str, list[str]]:
    """Run one conversation turn.

    ``history`` is the stored conversation (oldest first) including the latest
    user message. ``tools``/``model`` default to the cached production ones and
    exist so tests can inject a deterministic model and a temporary database.
    Returns ``(answer, audit_notes)``; raises on infrastructure failures so the
    caller can log and show a safe message.
    """
    if tools is None:
        tools = get_tools_cached()
    if model is None:
        model = get_model_cached()
    agent = build_agent(tools, model)

    lc_messages = convert_to_messages(list(history))
    result = invoke_agent_sync(agent, lc_messages)

    run_messages = result.get("messages", [])
    if not run_messages:
        raise RuntimeError("The agent returned no messages.")

    answer = extract_final_answer(run_messages)
    tool_results = collect_tool_results(run_messages)
    audit_notes = verify_against_db(tool_results)

    for item in tool_results:
        logger.info(
            "tool=%s args=%s success=%s",
            item["tool"],
            item["args"],
            (item.get("result") or {}).get("success"),
        )
    if audit_notes:
        logger.warning("Audit problems: %s", audit_notes)

    return answer, audit_notes


def invoke_agent_sync(agent: Any, messages: Sequence[BaseMessage]) -> dict[str, Any]:
    return run_coroutine_sync(
        invoke_agent(agent, messages), timeout=AGENT_TIMEOUT_SECONDS + 30
    )


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #


def render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        notes = message.get("notes") or []
        if notes:
            with st.expander("Database verification warnings"):
                for note in notes:
                    st.warning(note)


def main() -> None:
    st.set_page_config(
        page_title="AI Task Manager",
        page_icon="✅",
        layout="centered",
    )
    configure_logging()

    if not API_KEY:
        st.error("MISTRAL_API_KEY is missing from your .env file.")
        st.stop()
    if not SERVER_PATH.is_file():
        st.error(f"MCP server not found: {SERVER_PATH}")
        st.stop()

    st.title("AI-Powered Personal Task Manager")
    st.caption("Manage your tasks using natural language.")

    with st.sidebar:
        st.subheader("Task Manager")
        st.caption(f"Model: {MODEL_NAME}")
        st.caption(f"Today: {datetime.now().astimezone().date().isoformat()}")
        st.caption(f"Database: {task_server.resolve_db_path().name}")
        if st.button("Clear chat history", use_container_width=True):
            # Clears the conversation only - database records are untouched.
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        render_message(message)

    user_input = st.chat_input(
        "e.g. Show all tasks, or add a high-priority Work task due tomorrow"
    )
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Working on your tasks..."):
            error_id = uuid.uuid4().hex[:8]
            try:
                answer, notes = process_message(st.session_state.messages)
            except Exception as exc:  # noqa: BLE001 - boundary for the UI
                logger.exception("[%s] Agent execution failed", error_id)
                answer = describe_error(exc, error_id)
                notes = []
                st.caption(f"Error reference: {error_id}")
            answer = scrub(answer)
            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "notes": notes}
    )


def _is_running_in_streamlit() -> bool:
    """True when this file is being executed by ``streamlit run``."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:  # pragma: no cover - defensive
        return False


if __name__ == "__main__" or _is_running_in_streamlit():
    main()
