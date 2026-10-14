"""LangChain tool wrappers exposed to the orchestrator LLM."""

import json
from dataclasses import asdict

from langchain_core.tools import tool

from src.state import NutritionTargets, Recipe, UserProfile, WeeklyPlan
from src.tools.nutrition import calculate_targets as _calculate_targets
from src.tools.planner import build_plan as _build_plan
from src.tools.recipes import filter_recipes as _filter_recipes
from src.tools.recipes import search_recipes as _search_recipes


@tool("calculate_targets", response_format="content_and_artifact")
def calculate_targets(profile: UserProfile) -> tuple[str, NutritionTargets]:
    """Compute daily calorie and macro targets from the user's profile.

    Call this first, always. Pass the returned ``calories`` as ``max_calories``
    to ``search_recipes``.
    """
    targets = _calculate_targets(profile)
    return json.dumps(asdict(targets)), targets


@tool("search_recipes", response_format="content_and_artifact")
def search_recipes(
    query: str, diet: str, intolerances: str, max_calories: int, number: int
) -> tuple[str, list[Recipe]]:
    """Search Spoonacular for recipes matching a query and dietary constraints.

    Returns the matching recipes. Pass them on to ``filter_recipes``.
    """
    recipes = _search_recipes(query, diet, intolerances, max_calories, number)
    return json.dumps([asdict(r) for r in recipes]), recipes


@tool("filter_recipes", response_format="content_and_artifact")
def filter_recipes(
    recipes: list[Recipe], dislikes: list[str], allergies: list[str], already_used: list[int]
) -> tuple[str, list[Recipe]]:
    """Drop recipes that mention a dislike/allergy or were already used.

    Call this before ``build_plan``. Returns the surviving recipes.
    """
    kept = _filter_recipes(recipes, dislikes, allergies, already_used)
    return json.dumps([asdict(r) for r in kept]), kept


@tool("build_plan", response_format="content_and_artifact")
def build_plan(recipes: list[Recipe], meals_per_day: int, days: int) -> tuple[str, WeeklyPlan]:
    """Assemble the final weekly plan from filtered recipes. Call exactly once, last."""
    plan = _build_plan(recipes, meals_per_day, days)
    summary = (
        f"Built a {days}-day plan with {meals_per_day} meal(s) per day "
        f"from {len(recipes)} recipes."
    )
    return summary, plan
