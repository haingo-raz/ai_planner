"""Main LLM node: gathers the profile and drives tool calls."""

from src.state import AgentState


def orchestrator_node(state: AgentState) -> dict:
    """Run one orchestrator step and return the state update."""
    pass
