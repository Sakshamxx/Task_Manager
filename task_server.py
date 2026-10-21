
import os
import sqlite3
from datetime import datetime, date
from pathlib import Path
try:
    from mcp.server.fastmcp import FastMCP          # mcp 1.x
except ImportError:
    from mcp.server.mcpserver import MCPServer as FastMCP   # mcp 2.x

DEFAULT_DB = Path(__file__).resolve().with_name("tasks.db")
CATEGORIES = ["Personal", "Work", "Study", "Finance", "Health", "Shopping", "Other"]
PRIORITIES = ["Low", "Medium", "High"]
CLEAR = "CLEAR"

mcp = FastMCP("Task Manager")


def db_path():
    return Path(os.environ.get("TASK_DB_PATH", DEFAULT_DB))


def get_connection():
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("""
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
    """)
    return conn


def current_timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def check_date(value):
    """Return an error message, or None if value is a valid, non-past YYYY-MM-DD."""
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
        if parsed.strftime("%Y-%m-%d") != value:
            raise ValueError
    except ValueError:
        return "Use a valid YYYY-MM-DD date."
    if parsed.date() < date.today():
        return "Due date is in the past."
    return None


@mcp.tool()
def get_today() -> dict:
    """Return today's date (YYYY-MM-DD) and weekday. Use it to resolve 'today', 'tomorrow', 'next Friday'."""
    today = date.today()
    return {"success": True, "today": today.isoformat(), "weekday": today.strftime("%A")}


@mcp.tool()
def create_task(title: str, category: str = "Personal", description: str = "",
                priority: str = "Medium", due_date: str = "",
                allow_duplicate: bool = False) -> dict:
    """Create a task. due_date must be YYYY-MM-DD or empty. A pending task with the
    same title is refused unless allow_duplicate is true."""
    if not isinstance(title, str) or not title.strip():
        return {"success": False, "message": "Title cannot be empty."}
    if category not in CATEGORIES:
        return {"success": False, "message": f"Invalid category. Choose from {CATEGORIES}."}
    if priority not in PRIORITIES:
        return {"success": False, "message": "Invalid priority. Use Low, Medium or High."}
    if due_date:
        error = check_date(due_date)
        if error:
            return {"success": False, "message": error}
    now = current_timestamp()
    with get_connection() as conn:
        if not allow_duplicate:
            dup = conn.execute(
                "SELECT ID FROM tasks WHERE lower(Title) = lower(?) AND Status = 'Pending'",
                (title.strip(),)).fetchone()
            if dup:
                return {"success": False, "duplicate_of": dup["ID"],
                        "message": f"A pending task with this title already exists (ID {dup['ID']}). "
                                   "Ask the user whether to create it anyway."}
        cursor = conn.execute(
            """INSERT INTO tasks (Title, Category, Description, Priority, due_date,
               Status, Created_at, Updated_at, Completion_at)
               VALUES (?, ?, ?, ?, ?, 'Pending', ?, ?, NULL)""",
            (title.strip(), category, description, priority, due_date, now, now))
        task_id = cursor.lastrowid
    return {"success": True, "task_id": task_id, "title": title.strip(),
            "category": category, "priority": priority, "due_date": due_date,
            "status": "Pending", "created_at": now}


@mcp.tool()
def list_tasks(status: str = "", category: str = "") -> dict:
    """List tasks; optionally filter by status (Pending/Completed) and category."""
    query, params = "SELECT * FROM tasks WHERE 1=1", []
    if status:
        if status not in ["Pending", "Completed"]:
            return {"success": False, "message": "Status must be Pending or Completed."}
        query += " AND Status = ?"
        params.append(status)
    if category:
        query += " AND Category = ?"
        params.append(category)
    query += " ORDER BY ID DESC"
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return {"success": True, "count": len(rows), "tasks": [dict(r) for r in rows]}


@mcp.tool()
def find_tasks(keyword: str, status: str = "") -> dict:
    """Find tasks whose title contains the keyword (case-insensitive). Use this to
    identify a task by name before completing, updating, deleting or describing it.
    If count > 1 the request is ambiguous: ask the user which task they mean."""
    if not isinstance(keyword, str) or not keyword.strip():
        return {"success": False, "message": "Keyword cannot be empty."}
    query = "SELECT * FROM tasks WHERE lower(Title) LIKE ?"
    params = [f"%{keyword.strip().lower()}%"]
    if status:
        if status not in ["Pending", "Completed"]:
            return {"success": False, "message": "Status must be Pending or Completed."}
        query += " AND Status = ?"
        params.append(status)
    with get_connection() as conn:
        rows = conn.execute(query + " ORDER BY ID DESC", params).fetchall()
    return {"success": True, "count": len(rows), "tasks": [dict(r) for r in rows]}


@mcp.tool()
def get_task(task_id: int) -> dict:
    """Get full details (due date, status, etc.) for one task by ID."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE ID = ?", (task_id,)).fetchone()
    if row is None:
        return {"success": False, "message": f"Task {task_id} not found."}
    return {"success": True, "task": dict(row)}


@mcp.tool()
def update_task(task_id: int, title: str = "", category: str = "",
                description: str = "", priority: str = "", due_date: str = "") -> dict:
    """Update the supplied fields. Empty string = leave unchanged.
    Pass CLEAR as description or due_date to empty that field."""
    updates = {}
    if title:
        if not title.strip():
            return {"success": False, "message": "Title cannot be empty."}
        updates["Title"] = title.strip()
    if category:
        if category not in CATEGORIES:
            return {"success": False, "message": f"Invalid category. Choose from {CATEGORIES}."}
        updates["Category"] = category
    if description:
        updates["Description"] = "" if description == CLEAR else description
    if priority:
        if priority not in PRIORITIES:
            return {"success": False, "message": "Invalid priority. Use Low, Medium or High."}
        updates["Priority"] = priority
    if due_date:
        if due_date == CLEAR:
            updates["due_date"] = ""
        else:
            error = check_date(due_date)
            if error:
                return {"success": False, "message": error}
            updates["due_date"] = due_date
    if not updates:
        return {"success": False, "message": "No updates provided."}
    now = current_timestamp()
    with get_connection() as conn:
        if conn.execute("SELECT ID FROM tasks WHERE ID = ?", (task_id,)).fetchone() is None:
            return {"success": False, "message": f"Task {task_id} not found."}
        assignments = ", ".join(f"{field} = ?" for field in updates)
        conn.execute(f"UPDATE tasks SET {assignments}, Updated_at = ? WHERE ID = ?",
                     (*updates.values(), now, task_id))
    return {"success": True, "task_id": task_id, "updated_fields": list(updates),
            "message": "Task updated."}


@mcp.tool()
def complete_task(task_id: int) -> dict:
    """Mark a pending task as completed and store the completion time."""
    now = current_timestamp()
    with get_connection() as conn:
        row = conn.execute("SELECT Title, Status FROM tasks WHERE ID = ?", (task_id,)).fetchone()
        if row is None:
            return {"success": False, "message": f"Task {task_id} not found."}
        if row["Status"] == "Completed":
            return {"success": False, "message": "Task is already completed."}
        conn.execute("""UPDATE tasks SET Status = 'Completed', Completion_at = ?,
                     Updated_at = ? WHERE ID = ?""", (now, now, task_id))
    return {"success": True, "task_id": task_id, "title": row["Title"], "completion_at": now}


@mcp.tool()
def delete_task(task_id: int, confirm: bool = False) -> dict:
    """Permanently delete a task. Only call with confirm=true after the user
    explicitly asked to delete it."""
    if not confirm:
        return {"success": False, "message": "Deletion not confirmed. Ask the user to confirm."}
    with get_connection() as conn:
        row = conn.execute("SELECT Title FROM tasks WHERE ID = ?", (task_id,)).fetchone()
        if row is None:
            return {"success": False, "message": f"Task {task_id} not found."}
        conn.execute("DELETE FROM tasks WHERE ID = ?", (task_id,))
    return {"success": True, "task_id": task_id, "title": row["Title"], "message": "Task deleted."}


if __name__ == "__main__":
    mcp.run(transport="stdio")
