"""Lift structured tool artifacts from ToolMessages into AgentState."""

from langchain_core.messages import ToolMessage

from src.state import AgentState


def parse_tool_results_node(state: AgentState) -> AgentState:
    """Extract artifacts from the latest ToolMessages into structured state."""
    recent: list[ToolMessage] = []
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage):
            recent.append(message)
        else:
            break
    recent.reverse()  # restore call order

    updates: dict = {}
    recipes = list(state.get("recipes") or [])
    recipes_changed = False

    for message in recent:
        artifact = getattr(message, "artifact", None)
        if artifact is None:
            continue

        if message.name == "calculate_targets":
            updates["targets"] = artifact
        elif message.name == "search_recipes":
            # Accumulate across multiple searches, deduping by recipe id.
            seen = {r.id for r in recipes}
            for recipe in artifact:
                if recipe.id not in seen:
                    recipes.append(recipe)
                    seen.add(recipe.id)
            recipes_changed = True
        elif message.name == "filter_recipes":
            # The filtered set replaces the working list.
            recipes = list(artifact)
            recipes_changed = True
        elif message.name == "build_plan":
            updates["plan"] = artifact

    if recipes_changed:
        updates["recipes"] = recipes
    if "plan" in updates:
        updates["done"] = True

    return updates
