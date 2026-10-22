"""Domain types and the LangGraph agent state."""

from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


@dataclass
class UserProfile:
    age: int
    height_cm: int
    weight_kg: int
    sex: str
    goal: str
    allergies: list[str]
    dislikes: list[str]
    meals_per_day: int
    days: int


@dataclass
class NutritionTargets:
    calories: int
    protein_g: int
    carbs_g: int
    fat_g: int


@dataclass
class Recipe:
    id: int
    title: str
    url: str
    calories: int
    ingredients: list[str]


@dataclass
class WeeklyPlan:
    # {"Day 1": {"breakfast": Recipe, "lunch": Recipe, ...}, ...}
    days: dict[str, dict[str, Recipe]]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    profile: UserProfile | None
    profile_id: int | None
    targets: NutritionTargets | None
    recipes: list[Recipe]
    plan: WeeklyPlan | None
    error: str | None
    done: bool
    mode: str  # "api" or "chat"
    openai_api_key: str | None
