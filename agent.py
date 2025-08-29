#!/usr/bin/env python3
"""
Agentic Meal Planner Intake + Plan
- An agent decides which field to ask next, calls a console ask_user tool,
  validates the response, saves it to profile, and repeats until done.
- When complete, the agent calls the 'finish' tool and we produce a meal plan.
"""

import os
import json
from typing import Dict, List, Optional, TypedDict, Literal

from dotenv import load_dotenv
from rich import print as rprint
from rich.table import Table

from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.runnables.graph_mermaid import draw_mermaid_png

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# ------------------------------
# Config & Questions
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
    "meals_target": "What meals would you like included in your plan? (e.g., breakfast, lunch, dinner, snacks)",
    "plan_duration_days": "How long should the meal plan be? (number of days)",
}
FIELD_ORDER: List[str] = list(QUESTIONS.keys())

VALIDATION_MODE = os.getenv("VALIDATION_MODE", "simple").lower()  # "none"|"simple"|"llm"

# Types

class AgentState(TypedDict):
    messages: List[object]                 # LangChain message objects (SystemMessage, AIMessage, ToolMessage)
    profile: Dict[str, str]                # gathered answers
    done: bool                             # finish signal
    last_field: Optional[str]              # for reference/UX
    stage: Literal["agent", "tools", "reduce", "plan"]


# Pretty printing helpers

def pretty_answers(answers: Dict[str, str]) -> None:
    table = Table(title="Collected Answers")
    table.add_column("Field", style="bold")
    table.add_column("Answer")
    for k in FIELD_ORDER:
        v = answers.get(k, "")
        table.add_row(k, v)
    rprint(table)

# LLM factory

def llm() -> ChatOpenAI:
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
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
            "You validate a user answer for a specific field. Decide if it fits the expected type. "
            "If invalid, provide a short polite re-ask tip. If valid, normalize it (trim, commas). "
            "Respond STRICT JSON with keys: is_valid (bool), cleaned (string), message (string)."
        )
        user = f"Field key: {key}\nExpected type: {schema_hint}\nUser answer: {answer}\nReturn JSON only."

        resp = model.invoke([{"role": "system", "content": system},
                             {"role": "user", "content": user}])
        text = resp.content.strip()
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


def simple_validate_answer(key: str, answer: str) -> Optional[str]:
    """Fast local rules; return cleaned answer or None if invalid."""
    a = (answer or "").strip()
    if key in {"age", "height_cm", "weight_kg", "meals_per_day", "plan_duration_days"}:
        if a.isdigit():
            return a
        return None
    if key in {"liked_foods", "disliked_foods", "allergies", "meals_target"}:
        if a.lower() == "none" and key == "allergies":
            return "none"
        parts = [p.strip() for p in a.split(",") if p.strip()]
        if parts:
            return ", ".join(parts)
        return None
    return a or None


def validate_answer(key: str, answer: str) -> Optional[str]:
    """Unified validator according to VALIDATION_MODE."""
    a = (answer or "").strip()
    if VALIDATION_MODE == "none":
        return a
    if VALIDATION_MODE == "llm":
        res = llm_validate_answer(key, a)
        if res is not None:
            if res.get("is_valid"):
                return (res.get("cleaned") or a).strip()
            else:
                msg = res.get("message") or "Please try again with the expected format."
                rprint(f"[yellow]{msg}[/yellow]")
                return None
        # LLM failed → fall through
    return simple_validate_answer(key, a)


# Tools the agent can call

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
TOOLS_NODE = ToolNode(TOOLS)

# Agent node (policy)

def agent_node(state: AgentState) -> AgentState:
    """LLM policy that decides which tool to call next."""
    # Force the assistant to call a tool each turn (no free-form replies)
    model = llm().bind_tools(TOOLS, tool_choice="required")

    # Build the system prompt reflecting current profile
    known = {k: v for k, v in state["profile"].items() if v}
    missing = [k for k in FIELD_ORDER if not state["profile"].get(k)]
    system_text = (
        "You are an intake agent collecting dietary info to prepare a meal plan. "
        "Always use the ask_user tool to ask exactly ONE concise question for one missing field at a time. "
        f"Allowed field keys (use exactly): {FIELD_ORDER}. "
        "If all required fields are filled, call the finish tool. "
        "Be friendly and specific. Prefer numeric clarifications for number fields. "
        "Avoid repeating questions if the answer is already present."
        f"\n\nKnown profile (JSON): {json.dumps(known, ensure_ascii=False)}"
        f"\nMissing keys (ordered): {missing}"
    )

    # FILTER OUT ToolMessage objects before calling the model
    from langchain_core.messages import ToolMessage  # local import to avoid top-level circulars
    filtered_history = [m for m in state["messages"] if not isinstance(m, ToolMessage)]

    # trim very long histories
    MAX_CTX = 20
    if len(filtered_history) > MAX_CTX:
        filtered_history = filtered_history[-MAX_CTX:]

    model_messages = [SystemMessage(content=system_text), *filtered_history]

    ai = model.invoke(model_messages)
    state["messages"].append(ai)  # record the model decision (will contain tool_calls)
    return state

# Reduce node (apply tool results & validation)

def reduce_node(state: AgentState) -> AgentState:
    """
    After the tools run, inspect the last ToolMessage and update profile / flow.
    - ask_user → parse JSON {field_key, answer}, validate, save or signal invalid.
    - finish   → mark done=True
    Then append a short SystemMessage observation so the agent has fresh context.
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
        # Provide a short observation for the agent (not strictly needed now)
        state["messages"].append(SystemMessage(content="Observation: finish signal received."))
        return state

    if tool_name == "ask_user":
        try:
            data = json.loads(content)
        except Exception:
            data = {}
        field_key = str(data.get("field_key", "")).strip()
        raw_answer = str(data.get("answer", "")).strip()

        if field_key not in QUESTIONS:
            state["messages"].append(SystemMessage(
                content=f"Observation: invalid field_key '{field_key}'. Ask again with a valid key."
            ))
            return state

        cleaned = validate_answer(field_key, raw_answer)
        if cleaned is None:
            # Invalid — give the agent a hint to re-ask
            hints = {
                "age": "Expect a whole number (e.g., 30).",
                "height_cm": "Enter centimeters as a whole number (e.g., 175).",
                "weight_kg": "Enter kilograms as a whole number (e.g., 70).",
                "meals_per_day": "Whole number (e.g., 3).",
                "plan_duration_days": "Whole number (e.g., 7).",
                "liked_foods": "Comma-separated items (e.g., eggs, oatmeal).",
                "disliked_foods": "Comma-separated items (e.g., broccoli, tuna).",
                "allergies": "Comma-separated allergens or 'none'.",
                "meals_target": "Comma-separated meals (e.g., breakfast, dinner).",
            }
            hint = hints.get(field_key, "Please re-ask with a clearer example or format.")
            state["messages"].append(SystemMessage(
                content=f"Observation: answer for '{field_key}' invalid. Hint: {hint}"
            ))
            state["last_field"] = field_key
            return state

        # Valid — save
        state["profile"][field_key] = cleaned
        state["last_field"] = field_key
        state["messages"].append(SystemMessage(
            content=f"Observation: stored '{field_key}' = '{cleaned}'."
        ))
        return state

    # Unknown tool — ignore safely
    state["messages"].append(SystemMessage(content=f"Observation: unknown tool '{tool_name}'."))
    return state

# Planner (final meal plan generation)

def planner_node(state: AgentState) -> AgentState:
    """Generate meal recommendations using collected profile."""
    ans = state["profile"]

    # Coerce duration
    try:
        duration = int(str(ans.get("plan_duration_days", "7")).strip() or 7)
    except ValueError:
        duration = 7

    sys = (
        "You are a nutrition-aware meal planner. Create practical, culturally flexible, "
        "ingredient-level meal recommendations that are budget-conscious and easy to prepare. "
        "Keep portions reasonable and include brief justifications (protein/veg/carb). "
        "Respect dislikes and allergies strictly; never suggest allergen ingredients."
    )
    def g(k: str) -> str:
        return str(ans.get(k, "")).strip()

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
        - A day-by-day plan. For each day, list the target meals (e.g., breakfast/lunch/dinner/snacks)
        - Keep the total output compact but clear.
        - Include a small compact shopping list.
        """.strip()

    prompt = ChatPromptTemplate.from_messages([
        ("system", sys),
        ("human", user),
    ])

    result = (prompt | llm()).invoke({})

    rprint("\n[bold green]Thanks! Here's your meal plan:[/bold green]\n")
    rprint(result.content)

    rprint("\n[dim]— — — — — — — — — — — — — — — —[/dim]")
    pretty_answers(ans)

    return state

# Routing helpers

def route_from_agent(state: AgentState) -> str:
    """If the last AI message has tool calls, go to tools; otherwise stay with agent."""
    last = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
    if last and getattr(last, "tool_calls", None):
        return "tools"
    # Agent must always use tools; if it returned plain text, we'll loop and try again.
    return "agent"

def route_from_reduce(state: AgentState) -> str:
    return "plan" if state.get("done") else "agent"

# Build graph

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", TOOLS_NODE)
    g.add_node("reduce", reduce_node)
    g.add_node("plan", planner_node)

    g.set_entry_point("agent")
    g.add_conditional_edges("agent", route_from_agent, {"tools": "tools", "agent": "agent"})
    g.add_edge("tools", "reduce")
    g.add_conditional_edges("reduce", route_from_reduce, {"agent": "agent", "plan": "plan"})
    g.add_edge("plan", END)
    return g.compile()

# Main

def main():
    load_dotenv()

    # Initial conversation seed: instruct the agent explicitly.
    seed_system = SystemMessage(content=(
        "You will gather the user's dietary profile to create a meal plan. "
        "At each step, choose ONE missing field and call ask_user(field_key, question). "
        "After all fields are filled, call finish()."
    ))

    state: AgentState = {
        "messages": [seed_system],
        "profile": {k: "" for k in FIELD_ORDER},  # start empty; agent fills
        "done": False,
        "last_field": None,
        "stage": "agent",
    }

    app = build_graph()
    graph = app.get_graph()
    png_bytes = draw_mermaid_png(graph.draw_mermaid())
    with open("agent_flow.png", "wb") as f:
        f.write(png_bytes)
        rprint("[bold]Welcome to the Agentic Meal Planner![/bold]")
        rprint("I'll ask a few questions (in any order) to personalize your plan.")

    # Let the graph iterate until it reaches END
    while True:
        state = app.invoke(state, config={"recursion_limit": 1000})
        # We rely on the graph's END after planner_node runs.
        # To avoid an infinite loop if the agent refuses to use tools,
        # break if messages grow too large without progress.
        if state.get("done"):
            # Planner node will be triggered by routing; continue until END.
            pass
        # END is handled inside the graph; the loop exits when invoke stops making progress.
        # To be explicit, we can break after a plan is printed:
        # (But here we trust the END edge.)
        if not any(isinstance(m, AIMessage) for m in state["messages"][-3:]) and state.get("done"):
            break

    rprint("\n[bold]All set. Goodbye![/bold]")


if __name__ == "__main__":
    main()
