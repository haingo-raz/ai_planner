#!/usr/bin/env python3
"""
Agentic Meal Planner Intake + Plan
- An agent decides which field to ask next, calls a console ask_user tool,
  validates the response, saves it to profile, and repeats until done.
- When complete, the agent calls the 'finish' tool and we produce a meal plan.
"""

import os
import json
from typing import Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from rich import print as rprint
from rich.table import Table

from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Config & Questions

QUESTIONS: Dict[str, str] = {
    "name": "What's your name?",
    "age": "How old are you? (in years)",
    "height_cm": "What's your height in centimeters?",
    "weight_kg": "What's your weight in kilograms?",
    "liked_foods": "List a few foods you like (comma-separated).",
    "disliked_foods": "Any foods you dislike? (comma-separated)",
    "allergies": "Do you have any food allergies? (comma-separated, or 'none')",
    "special_needs": "Any special dietary needs or conditions (e.g., pregnancy, low-sodium)?",
    "meals_to_plan": "Which meals would you like included in your plan? (e.g., breakfast, lunch, dinner, snacks)",
    "plan_duration_days": "How long should the meal plan be? (number of days)",
}

REQUIRED_FIELDS: List[str] = list(QUESTIONS.keys())

class AgentState(TypedDict):
    messages: List[object]                 # LangChain message objects (SystemMessage, AIMessage, ToolMessage)
    profile: Dict[str, str]                # gathered answers
    done: bool                             # finish signal
    last_field: Optional[str]              # for reference/UX


# Pretty printing helpers
def pretty_answers(answers: Dict[str, str]) -> None:
    table = Table(title="Collected Answers")
    table.add_column("Field", style="bold")
    table.add_column("Answer")
    for k in REQUIRED_FIELDS:
        v = answers.get(k, "")
        table.add_row(k, v)
    rprint(table)

# LLM factory
def llm() -> ChatOpenAI:
    model = os.getenv("MODEL_NAME")
    temperature = float(os.getenv("MODEL_TEMPERATURE", "0.4"))
    return ChatOpenAI(model=model, temperature=temperature)

# Validation
def llm_validate_answer(key: str, answer: str) -> Optional[Dict[str, str]]:
    """
    Semantic validation via LLM. Returns:
      {"is_valid": bool, "cleaned": str, "message": str} or None on failure.
    """
    try:
        model = llm()
        schema_hint = {
            "age": "an integer number of years",
            "height_cm": "a number in centimeters",
            "weight_kg": "a number in kilograms",
            "meals_to_plan": "a comma-separated list of meals (e.g., breakfast, lunch, dinner, snacks)",
            "plan_duration_days": "an integer number of days",
            "liked_foods": "a comma-separated list of foods",
            "disliked_foods": "a comma-separated list of foods, or 'none' / 'no' if they have no dislikes",
            "allergies": "a comma-separated list of allergens, or 'none' / 'no' if they have no allergies",
            "special_needs": "free text describing dietary needs, or 'none' / 'no' if they have none",
            "meals_to_plan": "a comma-separated list of meals (e.g., breakfast, lunch, dinner, snacks)",
            "name": "a human name (free text)",
        }.get(key, "free text")

        system = (
            "You validate a user answer for a specific field. Decide if it fits the expected type. "
            "If invalid, provide a short polite re-ask tip. If valid, normalize it (trim, commas) if necessary. "
            "Respond STRICT JSON with keys: is_valid (bool), cleaned (string), message (string)."
        )
        user = f"Field key: {key}\nExpected type: {schema_hint}\nUser answer: {answer}\nReturn JSON only."

        response = model.invoke([{"role": "system", "content": system},
                             {"role": "user", "content": user}])
        text = response.content.strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return None
        data = json.loads(text[start:end+1])
        if not isinstance(data, dict) or "is_valid" not in data or "cleaned" not in data:
            return None
        return {
            "is_valid": bool(data.get("is_valid")),
            "cleaned": str(data.get("cleaned") or "").strip(),
            "message": str(data.get("message") or "").strip(),
        }
    except Exception:
        return None


def validate_answer(key: str, answer: str) -> Optional[str]:
    a = (answer or "").strip()

    OPTIONAL_FIELDS = {"disliked_foods", "allergies", "special_needs"}

    if key in OPTIONAL_FIELDS:
        negative_answers = {"no", "none", "n/a", "na", "not applicable", "nothing", "nope", "nil"}
        if a.lower().strip() in negative_answers:
            return "none"

    if not a:
        rprint("[yellow]Answer cannot be empty.[/yellow]")
        return None

    if key in {"age", "height_cm", "weight_kg", "meals_to_plan", "plan_duration_days"}:
        digits = "".join(ch for ch in a if ch.isdigit())
        if digits:
            return digits
        rprint("[yellow]Please enter a valid whole number.[/yellow]")
        return None

    res = llm_validate_answer(key, a)
    if res is not None:
        if res.get("is_valid"):
            return (res.get("cleaned") or a).strip()
        else:
            msg = res.get("message") or "Please try again with the expected format."
            rprint(f"[yellow]{msg}[/yellow]")
            return None


# List of Tools
@tool("ask_user")
def ask_user(field_key: str, question: str) -> str:
    """Ask the human a question for a specific field. Returns JSON: {"field_key": "...", "answer": "..."}"""
    rprint(f"\n[bold cyan]{question}[/bold cyan]")
    try:
        user_in = input("> ").strip()
    except EOFError:
        user_in = ""
    return json.dumps({"field_key": field_key, "answer": user_in}, ensure_ascii=False)


@tool("finish")
def finish() -> str:
    """Signal that all required information has been collected."""
    return "FINISH"


TOOLS = [ask_user, finish]
tools_node = ToolNode(TOOLS)

# List of nodes
def agent_node(state: AgentState) -> AgentState:
    """LLM policy that decides which tool to call next."""
    # Force the assistant to call a tool each turn (no free-form replies)
    model = llm().bind_tools(TOOLS, tool_choice="required")

    # Build the system prompt reflecting current profile
    known_fields = {k: v for k, v in state["profile"].items() if v}
    missing_fields = [k for k in REQUIRED_FIELDS if not state["profile"].get(k)]
    system_text = (
        "You are an intake agent collecting dietary information to prepare a meal plan. "
        "Use the ask_user tool to ask exactly ONE question to gather the user's information one missing field at a time. "
        f"Allowed field keys: {REQUIRED_FIELDS}. "
        "If all required fields are filled, call the finish tool. "
        "Be friendly and specific. Prefer numeric clarifications for number fields. "
        "Avoid repeating the same or paraphrased questions if the answer is already present."
        f"\n\nKnown profile (JSON): {json.dumps(known_fields, ensure_ascii=False)}"
        f"\nMissing keys (ordered): {missing_fields}"
    )

    model_messages = [SystemMessage(content=system_text)]

    ai = model.invoke(model_messages)
    state["messages"].append(ai)  # record the model decision (will contain tool_calls)
    return state


def reduce_node(state: AgentState) -> AgentState:
    """
    After the tools run, inspect the last ToolMessage and update profile / flow.
    - ask_user → parse JSON {field_key, answer}, validate, save or signal invalid.
    - finish   → mark done=True
    """
    # Find the most recent ToolMessage
    tool_msg: Optional[ToolMessage] = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            tool_msg = msg
            break

    if tool_msg is None:
        # Nothing to reduce; loop back to the agent
        return state

    tool_name = getattr(tool_msg, "name", "")
    content = tool_msg.content or ""

    if tool_name == "finish":
        state["done"] = True
        return state

    if tool_name == "ask_user":
        try:
            data = json.loads(content)
        except Exception:
            data = {}
        field_key = str(data.get("field_key", "")).strip()
        raw_answer = str(data.get("answer", "")).strip()

        if field_key not in QUESTIONS:
            return state

        cleaned_answer = validate_answer(field_key, raw_answer)
        if cleaned_answer is None:
            hints = {
                "age": "Expect a whole number (e.g., 30).",
                "height_cm": "Enter centimeters as a whole number (e.g., 175).",
                "weight_kg": "Enter kilograms as a whole number (e.g., 70).",
                "meals_to_plan": "Comma-separated meals (e.g., breakfast, lunch, dinner).",
                "plan_duration_days": "Whole number (e.g., 7).",
                "liked_foods": "Comma-separated items (e.g., eggs, oatmeal).",
                "disliked_foods": "Comma-separated items (e.g., broccoli, tuna).",
                "allergies": "Comma-separated allergens or 'none'.",
            }
            hint = hints.get(field_key, "Please re-ask with a clearer example or format.")
            state["last_field"] = field_key
            return state

        state["profile"][field_key] = cleaned_answer
        state["last_field"] = field_key
        return state

    return state

# Planner (final meal plan generation)
def planner_node(state: AgentState) -> AgentState:
    """Generate meal recommendations using collected profile."""
    user_profile = state["profile"]

    # Coerce duration
    try:
        duration = int(str(user_profile.get("plan_duration_days", "7")).strip())
    except ValueError:
        duration = 7

    system_text = (
        "You are a nutrition-aware meal planner. Create practical, culturally flexible, "
        "ingredient-level meal recommendations that are budget-conscious and easy to prepare. "
        "Keep portions reasonable and include brief justifications (protein/veg/carb). "
        "Respect dislikes and allergies strictly; never suggest allergen ingredients."
    )
    def get_field(k: str) -> str:
        return str(user_profile.get(k, "")).strip()

    user_text = f"""
        Generate a {duration}-day meal plan.

        Recipient name: {get_field('name')}
        Age: {get_field('age')} years
        Height: {get_field('height_cm')} cm
        Weight: {get_field('weight_kg')} kg
        Likes: {get_field('liked_foods')}
        Dislikes: {get_field('disliked_foods')}
        Allergies: {get_field('allergies')}
        Special needs: {get_field('special_needs')}
        Meals to plan: {get_field('meals_to_plan')}

        Output format:
        - A short overview of goals/constraints.
        - A day-by-day plan. For each day, list the target meals (e.g., breakfast/lunch/dinner/snacks)
        - Keep the total output compact but clear.
        - Include a small compact shopping list.
        """.strip()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", user_text),
    ])

    result = (prompt | llm()).invoke({})

    rprint("\n[bold green]Thanks! Here's your meal plan:[/bold green]\n")
    rprint(result.content)

    rprint("\n[dim]— — — — — — — — — — — — — — — —[/dim]")
    pretty_answers(user_profile)

    return state

# Routing helpers
def route_from_agent(state: AgentState) -> str:
    """If the last AI message has tool calls, go to tools; otherwise stay with agent."""
    last = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
    if last and getattr(last, "tool_calls", None):
        return "tools"
    raise RuntimeError("Agent failed to invoke a tool despite tool_choice='required'.")

def route_from_reduce(state: AgentState) -> str:
    return "plan" if state.get("done") else "agent"

# Build graph
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("reduce", reduce_node)
    g.add_node("plan", planner_node)

    g.set_entry_point("agent")
    g.add_conditional_edges("agent", route_from_agent, ["tools"])
    g.add_edge("tools", "reduce")
    g.add_conditional_edges("reduce", route_from_reduce, {"agent": "agent", "plan": "plan"})
    g.add_edge("plan", END)
    return g.compile()

# Main
def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        rprint("[bold red]Error: OPENAI_API_KEY environment variable is not set.[/bold red]")
        return

    state: AgentState = {
        "messages": [],
        "profile": {k: "" for k in REQUIRED_FIELDS},
        "done": False,
        "last_field": None,
    }

    app = build_graph()
    graph = app.get_graph()
    png_bytes = graph.draw_mermaid_png()
    with open("agent_flow.png", "wb") as f:
        f.write(png_bytes)
        
    rprint("[bold]Welcome to the Agentic Meal Planner![/bold]")
    rprint("I'll ask a few questions (in any order) to personalize your plan.")

    state = app.invoke(state, config={"recursion_limit": 100})

    rprint("\n[bold]All set. Goodbye![/bold]")

if __name__ == "__main__":
    main()
