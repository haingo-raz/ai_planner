"""Persist node: save the profile and plan to SQLite."""

from src.db import save_plan, save_profile
from src.state import AgentState


def persist_node(state: AgentState) -> AgentState:
    """Save the user's profile and plan, recording the profile id in state."""
    if state.get("profile") is not None:
        state["profile_id"] = save_profile(state["profile"])

    if state.get("plan") is not None and state.get("profile_id") is not None:
        save_plan(state["profile_id"], state["plan"])

    return state
