"""SQLite helpers to save and load user profiles and plans."""

from src.state import UserProfile, WeeklyPlan


def init_db() -> None:
    """Create database tables if they don't exist."""
    pass


def save_profile(user_id: str, profile: UserProfile) -> None:
    """Persist a user profile."""
    pass


def load_profile(user_id: str) -> UserProfile | None:
    """Load a user profile, or None if not found."""
    pass


def save_plan(user_id: str, plan: WeeklyPlan) -> None:
    """Persist a generated plan."""
    pass


def load_plans(user_id: str) -> list[WeeklyPlan]:
    """Load all past plans for a user."""
    pass
