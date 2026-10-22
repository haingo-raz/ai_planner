"""LangGraph graph construction and routing."""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from src.nodes.orchestrator import TOOLS, orchestrator_node
from src.nodes.parse_tool_results import parse_tool_results_node
from src.nodes.persist import persist_node
from src.nodes.refine import refine_node
from src.state import AgentState, UserProfile


def _profile_to_message(profile: UserProfile) -> HumanMessage:
    """Render the structured profile as a message the orchestrator LLM can read."""
    allergies = ", ".join(profile.allergies) if profile.allergies else "none"
    dislikes = ", ".join(profile.dislikes) if profile.dislikes else "none"
    content = (
        f"My profile: {profile.age} years old, {profile.sex}, "
        f"{profile.height_cm} cm, {profile.weight_kg} kg. "
        f"Goal: {profile.goal}. Allergies: {allergies}. Dislikes: {dislikes}. "
        f"Please build me a {profile.days}-day plan with "
        f"{profile.meals_per_day} meal(s) per day."
    )
    return HumanMessage(content=content)


def intake_node(state: AgentState) -> AgentState:
    """Surface the structured profile to the LLM as a HumanMessage, once per run."""
    profile = state.get("profile")
    if profile is None or state.get("messages"):
        return state
    return {"messages": [_profile_to_message(profile)]}


def _last_ai_has_tool_calls(state: AgentState) -> bool:
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            return bool(message.tool_calls)
    return False


def _last_human_approves(state: AgentState) -> bool:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            text = str(message.content).lower()
            return any(
                phrase in text
                for phrase in [
                    "done",
                    "looks good",
                    "perfect",
                    "that works",
                    "save it",
                    "save the plan",
                ]
            )
    return False


def route_from_orchestrator(state: AgentState) -> str:
    """Decide where to go after the orchestrator runs."""
    if _last_ai_has_tool_calls(state):
        return "tools"
    if state.get("mode") == "chat" and _last_human_approves(state):
        return "persist"
    if state.get("done"):
        if state.get("mode") == "api":
            return "persist"
        return "refine"
    if state.get("error") is not None:
        return "orchestrator"
    return END


def route_from_refine(state: AgentState) -> str:
    """After refine, persist if the user is satisfied, else keep refining."""
    if _last_human_approves(state):
        return "persist"
    return "orchestrator"


def build_graph() -> CompiledStateGraph:
    """Build and compile the recipe agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("intake", intake_node)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("parse_tool_results", parse_tool_results_node)
    graph.add_node("refine", refine_node)
    graph.add_node("persist", persist_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "tools": "tools",
            "refine": "refine",
            "persist": "persist",
            "orchestrator": "orchestrator",
            END: END,
        },
    )
    graph.add_edge("tools", "parse_tool_results")
    graph.add_edge("parse_tool_results", "orchestrator")
    graph.add_conditional_edges(
        "refine",
        route_from_refine,
        {"persist": "persist", "orchestrator": "orchestrator"},
    )
    graph.add_edge("persist", END)

    return graph.compile()
