"""LangGraph graph construction and routing."""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from src.nodes.orchestrator import TOOLS, orchestrator_node
from src.nodes.refine import refine_node
from src.state import AgentState


def intake_node(state: AgentState) -> AgentState:
    """Placeholder. Will collect/validate the user profile."""
    return state


def persist_node(state: AgentState) -> AgentState:
    """Placeholder. Will save the profile and plan to SQLite."""
    return state


def _last_ai_has_tool_calls(state: AgentState) -> bool:
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            return bool(message.tool_calls)
    return False


def route_from_orchestrator(state: AgentState) -> str:
    """Decide where to go after the orchestrator runs."""
    if _last_ai_has_tool_calls(state):
        return "tools"
    if state.get("done"):
        return "refine"
    if state.get("error") is not None:
        return "orchestrator"
    return END


def route_from_refine(state: AgentState) -> str:
    """After refine, persist if the user is satisfied, else keep refining."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            text = str(message.content).lower()
            if "done" in text or "looks good" in text:
                return "persist"
            break
    return "orchestrator"


def build_graph() -> CompiledStateGraph:
    """Build and compile the recipe agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("intake", intake_node)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("refine", refine_node)
    graph.add_node("persist", persist_node)

    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {"tools": "tools", "refine": "refine", "orchestrator": "orchestrator", END: END},
    )
    graph.add_edge("tools", "orchestrator")
    graph.add_conditional_edges(
        "refine",
        route_from_refine,
        {"persist": "persist", "orchestrator": "orchestrator"},
    )
    graph.add_edge("persist", END)

    return graph.compile()
