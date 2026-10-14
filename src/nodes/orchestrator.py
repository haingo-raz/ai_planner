"""Main LLM node: gathers the profile and drives tool calls."""

from pathlib import Path

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from src.config import settings
from src.state import AgentState
from src.tools.agent_tools import (
    build_plan,
    calculate_targets,
    filter_recipes,
    search_recipes,
)

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "orchestrator.md"
SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

TOOLS = [calculate_targets, search_recipes, filter_recipes, build_plan]

_llm = ChatOpenAI(
    model=settings.model_name,
    temperature=settings.model_temperature,
    api_key=settings.openai_api_key,
).bind_tools(TOOLS, tool_choice="auto")


def orchestrator_node(state: AgentState) -> AgentState:
    """Run one orchestrator step and append the AI response to the state."""
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = _llm.invoke(messages)
    return {"messages": [response]}
