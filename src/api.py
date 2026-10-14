"""FastAPI app exposing the agent over HTTP."""

from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import settings
from src.db import load_plan
from src.graph import build_graph
from src.state import AgentState, UserProfile, WeeklyPlan

VERSION = "0.1.0"

class PlanRequest(BaseModel):
    age: int
    height_cm: int
    weight_kg: int
    sex: str
    goal: str
    allergies: list[str] = Field(default_factory=list)
    dislikes: list[str] = Field(default_factory=list)
    meals_per_day: int = 3
    days: int = 7


class MealResponse(BaseModel):
    title: str
    url: str
    calories: int


class DayResponse(BaseModel):
    meals: dict[str, MealResponse]


class PlanResponse(BaseModel):
    days: dict[str, DayResponse]
    targets: dict
    profile_id: int | None = None


def _plan_to_response(
    plan: WeeklyPlan, targets: dict, profile_id: int | None = None
) -> PlanResponse:
    """Map the domain WeeklyPlan onto the HTTP response model."""
    return PlanResponse(
        days={
            day: DayResponse(
                meals={
                    slot: MealResponse(title=recipe.title, url=recipe.url, calories=recipe.calories)
                    for slot, recipe in meals.items()
                }
            )
            for day, meals in plan.days.items()
        },
        targets=targets,
        profile_id=profile_id,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build and cache the compiled graph once, for the app's lifetime."""
    app.state.graph = build_graph()
    yield


app = FastAPI(title="Recipe Discovery Agent", version=VERSION, lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": VERSION}


@app.post("/plan", response_model=PlanResponse)
def create_plan(request: PlanRequest) -> PlanResponse:
    profile = UserProfile(**request.model_dump())

    initial_state: AgentState = {
        "messages": [],
        "profile": profile,
        "profile_id": None,
        "targets": None,
        "recipes": [],
        "plan": None,
        "error": None,
        "done": False,
        "mode": "api",
    }

    result = app.state.graph.invoke(
        initial_state,
        config={"recursion_limit": settings.recursion_limit},
    )

    plan = result.get("plan")
    if plan is None:
        raise HTTPException(status_code=500, detail="failed to generate plan")

    targets = asdict(result["targets"]) if result.get("targets") is not None else {}
    return _plan_to_response(plan, targets, result.get("profile_id"))


@app.get("/plan/{profile_id}", response_model=PlanResponse)
def get_plan(profile_id: int) -> PlanResponse:
    plan = load_plan(profile_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="plan not found")
    return _plan_to_response(plan, {})


def main() -> None:
    """Run the API server."""
    import uvicorn

    uvicorn.run(app, host=settings.fastapi_host, port=settings.fastapi_port)
