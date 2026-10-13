"""Meal plan assembly."""

from src.state import Recipe, WeeklyPlan

MEAL_SLOTS = ["breakfast", "lunch", "dinner", "snack"]


def build_plan(recipes: list[Recipe], meals_per_day: int, days: int) -> WeeklyPlan:
    """Assign recipes round-robin to meal slots; no repeats within a day."""
    if not 1 <= meals_per_day <= len(MEAL_SLOTS):
        raise ValueError(f"meals_per_day must be between 1 and {len(MEAL_SLOTS)}")
    if len(recipes) < meals_per_day:
        raise ValueError(
            f"Need at least {meals_per_day} recipes to avoid repeats within a day, "
            f"got {len(recipes)}"
        )

    slots = MEAL_SLOTS[:meals_per_day]
    # Consecutive indices mod len(recipes) are distinct within a day because
    # len(recipes) >= meals_per_day; the list cycles across days.
    return WeeklyPlan(
        days={
            f"Day {day + 1}": {
                slot: recipes[(day * meals_per_day + i) % len(recipes)]
                for i, slot in enumerate(slots)
            }
            for day in range(days)
        }
    )
