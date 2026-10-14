"""Recipe search and filtering via the Spoonacular API."""

from typing import Any

import httpx

from src.config import settings
from src.state import Recipe


def _to_recipe(result: dict[str, Any]) -> Recipe:
    nutrients = result.get("nutrition", {}).get("nutrients", [])
    calorie_entry = next((n for n in nutrients if n["name"] == "Calories"), None)
    calories = round(calorie_entry["amount"]) if calorie_entry else 0
    return Recipe(
        id=result["id"],
        title=result["title"],
        url=result["sourceUrl"],
        calories=calories,
        ingredients=[],
    )


def search_recipes(
    query: str, diet: str, intolerances: str, max_calories: int, number: int
) -> list[Recipe]:
    """Search Spoonacular for recipes matching a query and dietary constraints."""
    response = httpx.get(
        f"{settings.spoonacular_base_url}/recipes/complexSearch",
        params={
            "query": query,
            "diet": diet,
            "intolerances": intolerances,
            "maxCalories": max_calories,
            "number": number,
            "addRecipeNutrition": "true",
            "addRecipeInformation": "true",
            "apiKey": settings.spoonacular_api_key,
        },
        timeout=settings.spoonacular_timeout,
    )
    if response.status_code != 200:
        raise ValueError(
            f"Spoonacular search failed with HTTP {response.status_code}: {response.text[:200]}"
        )
    return [_to_recipe(r) for r in response.json()["results"]]


def filter_recipes(
    recipes: list[Recipe], dislikes: list[str], allergies: list[str], already_used: list[int]
) -> list[Recipe]:
    """Drop recipes that mention a dislike/allergy or were already used."""
    banned = [w.lower() for w in dislikes + allergies if w.strip()]
    used = set(already_used)

    def is_allowed(recipe: Recipe) -> bool:
        if recipe.id in used:
            return False
        text = " ".join([recipe.title, *recipe.ingredients]).lower()
        return not any(word in text for word in banned)

    return [r for r in recipes if is_allowed(r)]
