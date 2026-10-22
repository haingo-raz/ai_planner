"""Smoke-test driver: build the graph and run it against a hardcoded profile."""

from langchain_core.messages import HumanMessage

from src.config import settings
from src.graph import build_graph
from src.state import AgentState


def main() -> None:
    state: AgentState = {
        "messages": [
            HumanMessage(
                content="I'm 30yo male, 175cm, 75kg, goal is maintain, "
                "no allergies, 3 meals/day, plan 3 days"
            ),
            HumanMessage(content="looks good"),
        ],
        "profile": None,
        "targets": None,
        "recipes": [],
        "plan": None,
        "error": None,
        "done": False,
        "mode": "chat",
        "openai_api_key": None,
    }

    app = build_graph()
    result = app.invoke(state, config={"recursion_limit": settings.recursion_limit})

    print("error:", result.get("error"))
    print("messages:", result.get("messages"))
    print("targets:", result.get("targets"))
    print("recipes count:", len(result.get("recipes") or []))

    plan = result.get("plan")
    if plan is None:
        print("No plan was produced.")
        return

    for day, meals in plan.days.items():
        print(day)
        for slot, recipe in meals.items():
            print(f"  {slot}: {recipe.title} — {recipe.url}")


if __name__ == "__main__":
    main()
