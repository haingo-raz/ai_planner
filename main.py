#!/usr/bin/env python3
import os
from typing import Dict, List, Literal, Optional, TypedDict
import json

from dotenv import load_dotenv
from rich import print as rprint
from rich.table import Table

# LangChain / LangGraph imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

VALIDATION_MODE = os.getenv("VALIDATION_MODE", "llm").lower()  # "none"|"simple"|"llm"

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

def llm_validate_answer(key: str, answer: str) -> Optional[Dict[str, str]]:
    """
    Use an LLM to semantically validate/normalize an answer.
    Returns a dict like:
      {"is_valid": true/false, "cleaned": "<normalized>", "message": "<optional re-ask>"}
    or None on any failure (so we can fall back).
    """
    try:
        model = get_llm()  # reuse your existing ChatOpenAI
        schema_hint = {
            "age": "an integer number of years",
            "height_cm": "a number in centimeters",
            "weight_kg": "a number in kilograms",
            "meals_per_day": "an integer count of meals per day",
            "plan_duration_days": "an integer number of days",
            "liked_foods": "a comma-separated list of foods",
            "disliked_foods": "a comma-separated list of foods",
            "allergies": "a comma-separated list of allergens or the word 'none'",
            "special_needs": "free text describing dietary needs",
            "meals_target": "a comma-separated list of meals (e.g., breakfast, lunch, dinner, snacks)",
            "name": "a human name (free text)",
        }.get(key, "free text")

        system = (
            "You evaluate a single user answer for a specific field and decide if it fits the expected type. "
            "If invalid, provide a short and polite re-ask message explaining the correct format. "
            "If valid, normalize it (trim spaces, consistent commas) but do not invent data. "
            "Respond as strict JSON only with keys: is_valid (bool), cleaned (string), message (string)."
        )
        user = f"""
            Field key: {key}
            Expected type: {schema_hint}
            User answer: {answer}
            Return JSON only.
        """

        resp = model.invoke([{"role": "system", "content": system},
                             {"role": "user", "content": user}])
        text = resp.content.strip()
        # Robust JSON extraction
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        data = json.loads(text[start:end+1])
        # Minimal sanity checks
        if not isinstance(data, dict):
            return None
        if "is_valid" not in data or "cleaned" not in data:
            return None
        return {
            "is_valid": bool(data.get("is_valid")),
            "cleaned": str(data.get("cleaned") or "").strip(),
            "message": str(data.get("message") or "").strip(),
        }
    except Exception:
        # Never let validation break the chat flow
        return None

def simple_validate_answer(key: str, answer: str) -> Optional[str]:
    """Fast local rules; return cleaned answer or None if invalid."""
    a = (answer or "").strip()
    if key in {"age", "height_cm", "weight_kg", "meals_per_day", "plan_duration_days"}:
        # Allow integers; you can expand to floats for height/weight if you like
        if a.isdigit():
            return a
        return None

    if key in {"liked_foods", "disliked_foods", "allergies", "meals_target"}:
        if a.lower() == "none" and key == "allergies":
            return "none"
        # Normalize comma lists
        parts = [p.strip() for p in a.split(",") if p.strip()]
        if parts:
            return ", ".join(parts)
        return None

    # Free-text fields
    return a or None


def validate_answer(key: str, answer: str) -> Optional[str]:
    """
    Unified validator:
      - if VALIDATION_MODE == 'llm', try LLM validation first, fallback to simple.
      - if 'simple', use local rules.
      - if 'none', accept anything (trimmed).
    """
    a = (answer or "").strip()
    if VALIDATION_MODE == "none":
        return a

    if VALIDATION_MODE == "llm":
        result = llm_validate_answer(key, a)
        if result is not None:
            if result.get("is_valid"):
                cleaned = result.get("cleaned") or a
                return cleaned.strip()
            else:
                # Show the model's gentle re-ask
                msg = result.get("message") or "Please try again with the expected format."
                rprint(f"[yellow]{msg}[/yellow]")
                return None
        # If LLM fails, fall through to simple rules

    return simple_validate_answer(key, a)

def node_store(state: ChatState) -> ChatState:
    idx = state.get("idx", 0)
    if idx < len(QUESTION_KEYS):
        key = QUESTION_KEYS[idx]
        raw_ans = state.get("last_input") or ""
        validated = validate_answer(key, raw_ans)
        if validated is None:
            # Re-ask same question with a hint for common types
            hints = {
                "age": "Enter a whole number (e.g., 30).",
                "height_cm": "Enter centimeters as a whole number (e.g., 175).",
                "weight_kg": "Enter kilograms as a whole number (e.g., 70).",
                "meals_per_day": "Enter a whole number (e.g., 3).",
                "plan_duration_days": "Enter a whole number (e.g., 7).",
                "liked_foods": "Provide a comma-separated list (e.g., eggs, oatmeal).",
                "disliked_foods": "Provide a comma-separated list (e.g., broccoli, tuna).",
                "allergies": "List allergens comma-separated or 'none'.",
                "meals_target": "Comma-separated meals (e.g., breakfast, dinner).",
            }
            hint = hints.get(key, "Please try again.")
            rprint(f"[red]That doesn't look valid.[/red] [dim]{hint}[/dim]")
            state["stage"] = "ask"
            return state

        state.setdefault("answers", {})[key] = validated
        state["idx"] = idx + 1

    state["stage"] = "plan" if state["idx"] >= len(QUESTION_KEYS) else "ask"
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
