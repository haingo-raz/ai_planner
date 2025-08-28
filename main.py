#!/usr/bin/env python3
import os
from typing import Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from rich import print as rprint
from rich.table import Table

# LangChain / LangGraph imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# ------------------------------
# Configuration
# ------------------------------

QUESTIONS: Dict[str, str] = {
    "name": "What's your name?",
    "age": "How old are you? (in years)",
    "height_cm": "What's your height in centimeters?",
    "weight_kg": "What's your weight in kilograms?",
    "liked_foods": "List a few foods you like (comma-separated).",
    "disliked_foods": "Any foods you dislike? (comma-separated)",
    "allergies": "Do you have any food allergies? (comma-separated, or 'none')",
    "special_needs": "Any special dietary needs or conditions (e.g., pregnancy, low-sodium)?",
    "meals_per_day": "How many times do you usually eat per day?",
    "meals_target": "Which meals should we plan for? (e.g., breakfast, lunch, dinner, snacks)",
    "plan_duration_days": "How long should the meal plan be? (number of days)",
}

QUESTION_KEYS: List[str] = list(QUESTIONS.keys())


class ChatState(TypedDict):
    """State that moves through the LangGraph."""
    idx: int
    answers: Dict[str, str]
    stage: Literal["ask", "store", "plan"]
    last_input: Optional[str]


def pretty_answers(answers: Dict[str, str]) -> None:
    table = Table(title="Collected Answers")
    table.add_column("Field", style="bold")
    table.add_column("Answer")
    for k in QUESTION_KEYS:
        v = answers.get(k, "")
        table.add_row(k, v)
    rprint(table)


def get_llm() -> ChatOpenAI:
    """
    Instantiate the LLM using OpenAI via langchain_openai.
    - Set OPENAI_API_KEY in your environment.
    - Optionally override MODEL_NAME & MODEL_TEMPERATURE.
    """
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    temperature = float(os.getenv("MODEL_TEMPERATURE", "0.4"))
    return ChatOpenAI(model=model, temperature=temperature)


# ------------------------------
# LangGraph nodes
# ------------------------------

def node_ask(state: ChatState) -> ChatState:
    """Ask the current question (CLI) and capture input."""
    idx = state.get("idx", 0)
    if idx >= len(QUESTION_KEYS):
        state["stage"] = "plan"
        return state

    key = QUESTION_KEYS[idx]
    prompt = QUESTIONS[key]

    rprint(f"\n[bold cyan]{prompt}[/bold cyan]")
    try:
        user_in = input("> ").strip()
    except EOFError:
        user_in = ""

    state["last_input"] = user_in
    state["stage"] = "store"
    return state


def node_store(state: ChatState) -> ChatState:
    """Store the last input under the current question key, then advance index."""
    idx = state.get("idx", 0)
    if idx < len(QUESTION_KEYS):
        key = QUESTION_KEYS[idx]
        ans = state.get("last_input") or ""
        state.setdefault("answers", {})[key] = ans
        state["idx"] = idx + 1

    if state["idx"] >= len(QUESTION_KEYS):
        state["stage"] = "plan"
    else:
        state["stage"] = "ask"
    return state


def node_plan(state: ChatState) -> ChatState:
    """Generate meal recommendations using the LLM based on collected answers."""
    answers = state.get("answers", {})

    # Ensure plan_duration_days is numeric – default 7
    try:
        duration = int(str(answers.get("plan_duration_days", "7")).strip() or 7)
    except ValueError:
        duration = 7

    sys = (
        "You are a nutrition-aware meal planner. Create practical, culturally flexible, "
        "ingredient-level meal recommendations that are budget-conscious and easy to prepare. "
        "Keep portions reasonable and include brief justifications (protein/veg/carb). "
        "Respect dislikes and allergies strictly; never suggest allergen ingredients."
    )

    def g(k: str) -> str:
        return str(answers.get(k, "")).strip()

    user = f"""
Please generate a {duration}-day meal plan.

Recipient name: {g('name')}
Age: {g('age')} years
Height: {g('height_cm')} cm
Weight: {g('weight_kg')} kg
Likes: {g('liked_foods')}
Dislikes: {g('disliked_foods')}
Allergies: {g('allergies')}
Special needs: {g('special_needs')}
Usual meals/day: {g('meals_per_day')}
Target meals to plan: {g('meals_target')}

Output format:
- A short overview of goals/constraints.
- A day-by-day plan. For each day, list the target meals (e.g., breakfast/lunch/dinner/snacks) with 1–2 options each.
- Keep the total output compact but clear (aim ~200–400 words per 3 days).
- Include a small shopping list per 2–3 days (grouped).
""".strip()

    prompt = ChatPromptTemplate.from_messages([
        ("system", sys),
        ("human", user),
    ])

    chain = prompt | get_llm()
    result = chain.invoke({})

    rprint("\n[bold green]Thanks! Here's your meal plan:[/bold green]\n")
    rprint(result.content)

    rprint("\n[dim]— — — — — — — — — — — — — — — —[/dim]")
    pretty_answers(answers)

    return state


# ------------------------------
# Build graph
# ------------------------------

def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("ask", node_ask)
    graph.add_node("store", node_store)
    graph.add_node("plan", node_plan)

    graph.set_entry_point("ask")

    # ask -> store
    graph.add_edge("ask", "store")

    # store -> ask or plan
    def next_step(s: ChatState):
        return "ask" if s.get("stage") == "ask" else "plan"

    graph.add_conditional_edges("store", next_step, {"ask": "ask", "plan": "plan"})

    # plan -> END
    graph.add_edge("plan", END)

    return graph.compile()


# ------------------------------
# Main
# ------------------------------

def main():
    load_dotenv()

    state: ChatState = {
        "idx": 0,
        "answers": {},
        "stage": "ask",
        "last_input": None,
    }

    app = build_graph()

    rprint("[bold]Welcome to the Meal Planner![/bold]")
    rprint("I will ask a few quick questions to personalize your plan.")

    while True:
        state = app.invoke(state, config={"recursion_limit": 1000})
        if state.get("stage") == "plan":
            app.invoke(state)  # run planning node
            break

    rprint("\n[bold]All set. Goodbye![/bold]")


if __name__ == "__main__":
    main()
