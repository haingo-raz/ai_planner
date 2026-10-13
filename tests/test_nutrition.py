"""Tests for calculate_targets."""

import pytest

from src.state import UserProfile
from src.tools.nutrition import calculate_targets


def make_profile(sex: str, age: int, weight_kg: int, height_cm: int, goal: str) -> UserProfile:
    return UserProfile(
        age=age,
        height_cm=height_cm,
        weight_kg=weight_kg,
        sex=sex,
        goal=goal,
        allergies=[],
        dislikes=[],
        meals_per_day=3,
        days=7,
    )


def test_male_maintain() -> None:
    # BMR 1698.75 * 1.55 = 2633
    targets = calculate_targets(make_profile("male", 30, 75, 175, "maintain"))
    assert targets.calories == 2633
    assert 2550 <= targets.calories <= 2700


def test_female_lose() -> None:
    # BMR 1345.25 * 1.55 - 500 = 1585
    targets = calculate_targets(make_profile("female", 25, 60, 165, "lose"))
    assert targets.calories == 1585
    assert 1500 <= targets.calories <= 1800


def test_male_gain() -> None:
    # BMR 1830 * 1.55 + 300 = 3136
    targets = calculate_targets(make_profile("male", 40, 90, 180, "gain"))
    assert targets.calories == 3136
    assert 3000 <= targets.calories <= 3400


@pytest.mark.parametrize(
    "profile",
    [
        make_profile("male", 30, 75, 175, "maintain"),
        make_profile("female", 25, 60, 165, "lose"),
        make_profile("male", 40, 90, 180, "gain"),
    ],
)
def test_macros_are_positive_ints(profile: UserProfile) -> None:
    targets = calculate_targets(profile)
    for value in (targets.calories, targets.protein_g, targets.carbs_g, targets.fat_g):
        assert isinstance(value, int)
        assert value > 0


def test_macros_derived_from_calories() -> None:
    targets = calculate_targets(make_profile("male", 30, 75, 175, "maintain"))
    assert targets.protein_g == round(2633 * 0.30 / 4)
    assert targets.carbs_g == round(2633 * 0.40 / 4)
    assert targets.fat_g == round(2633 * 0.30 / 9)


def test_case_insensitive_inputs() -> None:
    upper = calculate_targets(make_profile("Male", 30, 75, 175, "Maintain"))
    lower = calculate_targets(make_profile("male", 30, 75, 175, "maintain"))
    assert upper == lower


@pytest.mark.parametrize("sex, goal", [("other", "maintain"), ("male", "bulk")])
def test_invalid_inputs_raise(sex: str, goal: str) -> None:
    with pytest.raises(ValueError):
        calculate_targets(make_profile(sex, 30, 75, 175, goal))
