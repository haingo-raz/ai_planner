"""Recipe search and filtering via the Spoonacular API."""

from src.state import Recipe, UserProfile


def search_recipes(query: str, max_calories: int | None = None, number: int = 10) -> list[Recipe]:
    """Search Spoonacular for recipes matching a query."""
    pass


def filter_recipes(recipes: list[Recipe], profile: UserProfile) -> list[Recipe]:
    """Remove recipes that conflict with the user's allergies or dislikes."""
    pass
