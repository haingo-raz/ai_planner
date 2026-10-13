"""Nutrition target calculation (BMR/TDEE)."""

from src.state import NutritionTargets, UserProfile

ACTIVITY_FACTOR = 1.55  # moderately active
GOAL_ADJUSTMENT = {"lose": -500, "maintain": 0, "gain": 300}


def calculate_targets(profile: UserProfile) -> NutritionTargets:
    """Compute daily calorie and macro targets using Mifflin-St Jeor."""
    sex = profile.sex.lower()
    if sex not in ("male", "female"):
        raise ValueError(f"sex must be 'male' or 'female', got {profile.sex!r}")
    goal = profile.goal.lower()
    if goal not in GOAL_ADJUSTMENT:
        raise ValueError(f"goal must be one of {list(GOAL_ADJUSTMENT)}, got {profile.goal!r}")

    bmr = 10 * profile.weight_kg + 6.25 * profile.height_cm - 5 * profile.age
    bmr += 5 if sex == "male" else -161
    calories = round(bmr * ACTIVITY_FACTOR + GOAL_ADJUSTMENT[goal])

    return NutritionTargets(
        calories=calories,
        protein_g=round(calories * 0.30 / 4),
        carbs_g=round(calories * 0.40 / 4),
        fat_g=round(calories * 0.30 / 9),
    )
