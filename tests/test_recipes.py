"""Tests for search_recipes and filter_recipes."""

from typing import Any

import httpx
import pytest
from pytest_mock import MockerFixture

from src.state import Recipe
from src.tools.recipes import filter_recipes, search_recipes


@pytest.fixture
def recipes() -> list[Recipe]:
    return [
        Recipe(1, "Peanut Butter Oats", "https://x/1", 400, ["oats", "peanut butter", "milk"]),
        Recipe(2, "Grilled Chicken Salad", "https://x/2", 450, ["chicken", "lettuce", "tomato"]),
        Recipe(3, "Mushroom Risotto", "https://x/3", 600, ["rice", "Mushrooms", "parmesan"]),
        Recipe(4, "Salmon Bowl", "https://x/4", 550, ["salmon", "rice", "avocado"]),
        Recipe(5, "Veggie Stir Fry", "https://x/5", 350, ["broccoli", "carrot", "soy sauce"]),
    ]


def ids(recipes: list[Recipe]) -> list[int]:
    return [r.id for r in recipes]


def test_filter_dislikes_matches_ingredients_case_insensitive(recipes: list[Recipe]) -> None:
    result = filter_recipes(recipes, dislikes=["MUSHROOM"], allergies=[], already_used=[])
    assert ids(result) == [1, 2, 4, 5]


def test_filter_dislikes_matches_title(recipes: list[Recipe]) -> None:
    result = filter_recipes(recipes, dislikes=["salad"], allergies=[], already_used=[])
    assert ids(result) == [1, 3, 4, 5]


def test_filter_allergies(recipes: list[Recipe]) -> None:
    result = filter_recipes(recipes, dislikes=[], allergies=["peanut", "soy"], already_used=[])
    assert ids(result) == [2, 3, 4]


def test_filter_already_used(recipes: list[Recipe]) -> None:
    result = filter_recipes(recipes, dislikes=[], allergies=[], already_used=[2, 4])
    assert ids(result) == [1, 3, 5]


def test_filter_combined(recipes: list[Recipe]) -> None:
    result = filter_recipes(recipes, dislikes=["salmon"], allergies=["peanut"], already_used=[5])
    assert ids(result) == [2, 3]


def test_filter_no_constraints_returns_all(recipes: list[Recipe]) -> None:
    assert filter_recipes(recipes, [], [], []) == recipes


SPOONACULAR_RESPONSE: dict[str, Any] = {
    "results": [
        {
            "id": 715538,
            "title": "Bruschetta Style Pork & Pasta",
            "sourceUrl": "https://example.com/bruschetta-pork-pasta",
            "nutrition": {
                "nutrients": [
                    {"name": "Fat", "amount": 18.3, "unit": "g"},
                    {"name": "Calories", "amount": 521.6, "unit": "kcal"},
                ]
            },
        }
    ],
    "offset": 0,
    "number": 1,
    "totalResults": 1,
}


def test_search_recipes_maps_response(mocker: MockerFixture) -> None:
    request = httpx.Request("GET", "https://api.spoonacular.com/recipes/complexSearch")
    mock_get = mocker.patch(
        "src.tools.recipes.httpx.get",
        return_value=httpx.Response(200, json=SPOONACULAR_RESPONSE, request=request),
    )

    result = search_recipes("pasta", "", "gluten", max_calories=700, number=1)

    assert result == [
        Recipe(
            id=715538,
            title="Bruschetta Style Pork & Pasta",
            url="https://example.com/bruschetta-pork-pasta",
            calories=522,
            ingredients=[],
        )
    ]
    url = mock_get.call_args.args[0]
    params = mock_get.call_args.kwargs["params"]
    assert url == "https://api.spoonacular.com/recipes/complexSearch"
    assert params["query"] == "pasta"
    assert params["intolerances"] == "gluten"
    assert params["maxCalories"] == 700
    assert params["number"] == 1
    assert params["addRecipeNutrition"] == "true"
    assert params["addRecipeInformation"] == "true"
    assert params["apiKey"]


def test_search_recipes_raises_on_http_error(mocker: MockerFixture) -> None:
    request = httpx.Request("GET", "https://api.spoonacular.com/recipes/complexSearch")
    mocker.patch(
        "src.tools.recipes.httpx.get",
        return_value=httpx.Response(402, text="Daily quota exceeded", request=request),
    )
    with pytest.raises(ValueError, match="402"):
        search_recipes("pasta", "", "", max_calories=700, number=1)
