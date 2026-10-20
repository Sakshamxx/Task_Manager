# AI-Powered Personal Task Manager

A chat interface (Streamlit) that manages a SQLite task database through an
AI agent (Mistral + LangGraph) and an MCP tool server.

```
app.py          Streamlit UI + agent wiring + post-run database verification
task_server.py  MCP server (FastMCP) and the SQLite task layer
tests/          pytest suite (no paid API calls required)
tasks.db        your data (git-ignored)
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

echo "MISTRAL_API_KEY=your-key-here" > .env   # git-ignored
```

## Run the app

```bash
.venv/bin/streamlit run app.py
```

## Run the tests

```bash
.venv/bin/python -m pytest
```

The suite never calls the Mistral API and never touches `tasks.db`: every test
runs against a private temporary database (`tests/conftest.py` sets
`TASK_DB_PATH` automatically), and the agent-level tests drive the real
LangGraph agent + real MCP stdio server with a deterministic scripted model.

Useful extras:

```bash
.venv/bin/python -m pytest tests/test_task_store.py -q   # database layer only
.venv/bin/python -m pytest -k ambiguous -q               # one behaviour
.venv/bin/python task_server.py --db /tmp/scratch.db     # run the server by hand
```

## How it fits together

* `app.py` starts the MCP server once per process, builds a LangGraph ReAct
  agent (`langchain.agents.create_agent`) with the current local date in the
  system prompt, and runs every conversation turn on **one long-lived event
  loop** (a fresh loop per message is what caused the intermittent
  "Event loop is closed" failures).
* `task_server.py` owns all SQL. Every tool returns a structured payload
  (`success`, `status`, `affected`, plus the row read back from SQLite), so a
  "success" is always a verified database state, never an attempted statement.
* Task resolution (ID or title) is deterministic Python: ambiguous, missing or
  conflicting targets refuse to act, so a mutation can never land on the wrong
  record.
* After each turn `app.py` replays the tool results against the real database
  (`verify_against_db`) and shows a warning if anything disagrees.

## Configuration (environment variables, all optional)

| Variable | Default | Meaning |
| --- | --- | --- |
| `MISTRAL_API_KEY` | – | required, read from `.env` |
| `TASK_DB_PATH` | `tasks.db` next to `task_server.py` | database location (also `--db`) |
| `TASK_MANAGER_MODEL` | `ministral-3b-2512` | chat model |
| `TASK_MANAGER_RECURSION_LIMIT` | `30` | agent step limit |
| `TASK_MANAGER_AGENT_TIMEOUT` | `120` | seconds per request |
| `TASK_MANAGER_TOOL_TIMEOUT` | `60` | seconds for the MCP handshake |
| `TASK_MANAGER_LLM_TIMEOUT` | `60` | seconds per model call |
| `TASK_MANAGER_LLM_RETRIES` | `3` | retries of the (idempotent) model HTTP call |

Tool operations (create/update/delete) are **never** retried automatically.

## Debugging a failed request

The chat shows a short message with a reference ID; the full traceback is in
`app.log` (and the terminal) under the same reference ID. API keys are
scrubbed from logs.

## Note on the demo notebook

`AI_Task_Manager_Complete.ipynb` writes its own copies of `app.py` and
`task_server.py` when its cells are run, which will overwrite the maintained
implementation in this repository. Treat the notebook as a demo, not as the
source of truth.
