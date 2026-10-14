"""Follow-up Q&A node for refining an existing plan."""

from pathlib import Path

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from src.config import settings
from src.nodes.orchestrator import TOOLS
from src.state import AgentState

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "refine.md"
SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

_llm = ChatOpenAI(
    model=settings.model_name,
    temperature=settings.model_temperature,
    api_key=settings.openai_api_key,
).bind_tools(TOOLS, tool_choice="auto")


def refine_node(state: AgentState) -> AgentState:
    """Answer follow-up questions or adjust the current plan. Assumes a plan exists."""
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = _llm.invoke(messages)
    state["messages"].append(response)
    return state
