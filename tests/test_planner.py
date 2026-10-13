"""Tests for build_plan."""

import pytest

from src.state import Recipe
from src.tools.planner import build_plan


def make_recipes(n: int) -> list[Recipe]:
    return [Recipe(i, f"Recipe {i}", f"https://x/{i}", 500, []) for i in range(n)]


def test_structure_and_slot_names() -> None:
    plan = build_plan(make_recipes(10), meals_per_day=3, days=7)
    assert list(plan.days) == [f"Day {d}" for d in range(1, 8)]
    for meals in plan.days.values():
        assert list(meals) == ["breakfast", "lunch", "dinner"]


def test_round_robin_order() -> None:
    plan = build_plan(make_recipes(10), meals_per_day=2, days=2)
    assert [r.id for r in plan.days["Day 1"].values()] == [0, 1]
    assert [r.id for r in plan.days["Day 2"].values()] == [2, 3]


@pytest.mark.parametrize("n_recipes", [4, 5, 7])
def test_no_repeat_within_day_when_cycling(n_recipes: int) -> None:
    plan = build_plan(make_recipes(n_recipes), meals_per_day=4, days=7)
    for meals in plan.days.values():
        meal_ids = [r.id for r in meals.values()]
        assert len(set(meal_ids)) == len(meal_ids)


def test_cycles_across_days() -> None:
    plan = build_plan(make_recipes(3), meals_per_day=3, days=2)
    assert [r.id for r in plan.days["Day 2"].values()] == [0, 1, 2]


def test_too_few_recipes_raises() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        build_plan(make_recipes(2), meals_per_day=3, days=1)


@pytest.mark.parametrize("meals_per_day", [0, 5])
def test_invalid_meals_per_day_raises(meals_per_day: int) -> None:
    with pytest.raises(ValueError):
        build_plan(make_recipes(10), meals_per_day=meals_per_day, days=1)
