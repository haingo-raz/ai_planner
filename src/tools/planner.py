"""Meal plan assembly."""

from src.state import NutritionTargets, Recipe, UserProfile, WeeklyPlan


def build_plan(recipes: list[Recipe], profile: UserProfile, targets: NutritionTargets) -> WeeklyPlan:
    """Assign recipes to each day and meal slot to meet the nutrition targets."""
    pass
