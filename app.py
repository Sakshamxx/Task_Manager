
import os, sys, asyncio, logging
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_mcp_adapters.client import MultiServerMCPClient

try:
    from langchain.agents import create_agent
    def build_agent(model, tools, prompt):
        return create_agent(model, tools, system_prompt=prompt)
except ImportError:
    from langgraph.prebuilt import create_react_agent
    def build_agent(model, tools, prompt):
        return create_react_agent(model, tools, prompt=prompt)

st.set_page_config(page_title="AI Task Manager", page_icon="✅", layout="centered")
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("MISTRAL_API_KEY")
SERVER_PATH = BASE_DIR / "task_server.py"
MODEL_NAME = "ministral-3b-2512"

if not API_KEY:
    st.error("MISTRAL_API_KEY is missing from your .env file.")
    st.stop()
if not SERVER_PATH.is_file():
    st.error(f"MCP server not found: {SERVER_PATH.name}")
    st.stop()


@st.cache_resource
def get_model():
    return ChatMistralAI(model=MODEL_NAME, api_key=API_KEY, temperature=0)


def system_prompt():
    today = datetime.now().astimezone().date()
    return (
        f"You are a personal task manager. Today is {today.isoformat()} ({today.strftime('%A')}).\n"
        "RULES:\n"
        "- Use tools for every task operation. Never invent tasks, IDs or results.\n"
        "- Never claim success unless a tool returned success=true.\n"
        "- Convert relative dates to YYYY-MM-DD; call get_today if unsure.\n"
        "- To act on a task named by the user, call find_tasks first. 0 matches: say it does not exist. "
        "More than one: list them with IDs and ask which one. Do not guess.\n"
        "- If the title is missing when creating a task, ask for it. Do not invent one.\n"
        "- If create_task reports a duplicate, tell the user and ask whether to create it anyway.\n"
        "- If you asked for a task ID and the user replies with a number, continue the original operation.\n"
        "- Delete only when explicitly asked, and pass confirm=true only then.\n"
        "- Be concise. For lists include ID, title, category, priority, due date and status."
    )


async def run_agent(messages):
    client = MultiServerMCPClient({"task_manager": {
        "transport": "stdio", "command": sys.executable, "args": [str(SERVER_PATH)]}})
    tools = await client.get_tools()
    agent = build_agent(get_model(), tools, system_prompt())
    result = await agent.ainvoke({"messages": messages}, config={"recursion_limit": 20})
    content = result["messages"][-1].content
    if isinstance(content, list):
        content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
    answer = content.strip()
    if not answer:
        raise RuntimeError("The agent returned an empty response.")
    return answer


st.title("AI-Powered Personal Task Manager")
st.caption("Manage your tasks using natural language.")

with st.sidebar:
    st.subheader("Task Manager")
    st.caption(f"Model: {MODEL_NAME}")
    st.caption(f"Today: {datetime.now().astimezone().date()}")
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("e.g. Add a high-priority Study task due tomorrow")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Working on your tasks..."):
            try:
                answer = asyncio.run(run_agent(st.session_state.messages))
            except Exception:
                logging.exception("Agent execution failed.")
                answer = "The request failed. Check the terminal logs for the underlying error."
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
