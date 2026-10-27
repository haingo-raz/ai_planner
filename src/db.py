"""SQLite persistence for user profiles and plans via SQLModel."""

import json
from dataclasses import asdict
from datetime import datetime, timezone

from sqlalchemy import Engine
from sqlmodel import Field, Session, SQLModel, create_engine, select

from src.config import settings
from src.state import NutritionTargets, Recipe, UserProfile, WeeklyPlan


class UserProfileRecord(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    age: int
    height_cm: int
    weight_kg: int
    sex: str
    goal: str
    allergies: str  # JSON-serialised list
    dislikes: str  # JSON-serialised list
    meals_per_day: int
    days: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PlanRecord(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    profile_id: int = Field(foreign_key="userprofilerecord.id")
    plan_json: str  # full WeeklyPlan serialised as JSON
    targets_json: str  # NutritionTargets serialised as JSON
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


_engine: Engine | None = None


def get_engine() -> Engine:
    """Return the cached SQLite engine, creating tables on first call."""
    global _engine
    if _engine is None:
        _engine = create_engine(settings.database_url)
        SQLModel.metadata.create_all(_engine)
    return _engine


def _recipe_from_dict(data: dict) -> Recipe:
    """Rebuild a Recipe dataclass instance from a plain dict."""
    return Recipe(
        id=data["id"],
        title=data["title"],
        url=data["url"],
        calories=data["calories"],
        ingredients=list(data.get("ingredients", [])),
    )


def _plan_to_json(plan: WeeklyPlan) -> str:
    """Serialise a WeeklyPlan (with nested Recipe dataclasses) to a JSON string."""
    return json.dumps(asdict(plan))


def _plan_from_json(plan_json: str) -> WeeklyPlan:
    """Deserialise a WeeklyPlan, rebuilding nested Recipe instances."""
    raw = json.loads(plan_json)
    days = {
        day: {meal: _recipe_from_dict(recipe) for meal, recipe in meals.items()}
        for day, meals in raw["days"].items()
    }
    return WeeklyPlan(days=days)


def _targets_to_json(targets: NutritionTargets) -> str:
    """Serialise NutritionTargets to a JSON string."""
    return json.dumps(asdict(targets))


def _targets_from_json(targets_json: str) -> NutritionTargets:
    """Rebuild a NutritionTargets dataclass instance from JSON."""
    return NutritionTargets(**json.loads(targets_json))


def save_profile(profile: UserProfile) -> int:
    """Persist a user profile and return the new record id."""
    record = UserProfileRecord(
        age=profile.age,
        height_cm=profile.height_cm,
        weight_kg=profile.weight_kg,
        sex=profile.sex,
        goal=profile.goal,
        allergies=json.dumps(profile.allergies),
        dislikes=json.dumps(profile.dislikes),
        meals_per_day=profile.meals_per_day,
        days=profile.days,
    )
    with Session(get_engine()) as session:
        session.add(record)
        session.commit()
        session.refresh(record)
        return record.id


def load_profile(profile_id: int) -> UserProfile | None:
    """Load a user profile by id, or None if not found."""
    with Session(get_engine()) as session:
        record = session.get(UserProfileRecord, profile_id)
        if record is None:
            return None
        return UserProfile(
            age=record.age,
            height_cm=record.height_cm,
            weight_kg=record.weight_kg,
            sex=record.sex,
            goal=record.goal,
            allergies=json.loads(record.allergies),
            dislikes=json.loads(record.dislikes),
            meals_per_day=record.meals_per_day,
            days=record.days,
        )


def save_plan(profile_id: int, plan: WeeklyPlan, targets: NutritionTargets) -> int:
    """Persist a weekly plan and its targets for a profile; return the new record id."""
    record = PlanRecord(
        profile_id=profile_id,
        plan_json=_plan_to_json(plan),
        targets_json=_targets_to_json(targets),
    )
    with Session(get_engine()) as session:
        session.add(record)
        session.commit()
        session.refresh(record)
        return record.id


def load_plan(profile_id: int) -> tuple[WeeklyPlan, NutritionTargets] | None:
    """Load the most recent plan and targets for a profile, or None if none exists."""
    with Session(get_engine()) as session:
        statement = (
            select(PlanRecord)
            .where(PlanRecord.profile_id == profile_id)
            .order_by(PlanRecord.created_at.desc(), PlanRecord.id.desc())
        )
        record = session.exec(statement).first()
        if record is None:
            return None
        return _plan_from_json(record.plan_json), _targets_from_json(record.targets_json)
