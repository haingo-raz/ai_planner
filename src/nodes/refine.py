"""Follow-up Q&A node for refining an existing plan."""

from pathlib import Path

from langchain_core.messages import SystemMessage

from src.nodes.orchestrator import build_llm
from src.state import AgentState

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "refine.md"
SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


def refine_node(state: AgentState) -> AgentState:
    """Answer follow-up questions or adjust the current plan. Assumes a plan exists."""
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    llm = build_llm(state.get("openai_api_key"))
    response = llm.invoke(messages)
    state["messages"].append(response)
    return state
