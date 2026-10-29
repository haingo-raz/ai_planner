"""FastAPI app exposing the agent over HTTP."""

from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, messages_from_dict, messages_to_dict
from pydantic import BaseModel, Field

from src.config import settings
from src.db import load_plan, load_profile
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
    openai_api_key: str | None = None


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
    messages: list[dict] = Field(default_factory=list)


def _plan_to_response(
    plan: WeeklyPlan,
    targets: dict,
    profile_id: int | None = None,
    messages: list[dict] | None = None,
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
        messages=messages or [],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build and cache the compiled graph once, for the app's lifetime."""
    app.state.graph = build_graph()
    yield


app = FastAPI(title="Recipe Discovery Agent", version=VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": VERSION}


@app.post("/plan", response_model=PlanResponse)
def create_plan(request: PlanRequest) -> PlanResponse:
    profile = UserProfile(**request.model_dump(exclude={"openai_api_key"}))

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
        config={
            "recursion_limit": settings.recursion_limit,
            "configurable": {"openai_api_key": request.openai_api_key},
        },
    )

    plan = result.get("plan")
    if plan is None:
        raise HTTPException(status_code=500, detail="failed to generate plan")

    targets = asdict(result["targets"]) if result.get("targets") is not None else {}
    messages = messages_to_dict(result.get("messages") or [])
    return _plan_to_response(plan, targets, result.get("profile_id"), messages)


@app.get("/plan/{profile_id}", response_model=PlanResponse)
def get_plan(profile_id: int) -> PlanResponse:
    loaded = load_plan(profile_id)
    if loaded is None:
        raise HTTPException(status_code=404, detail="plan not found")
    plan, targets = loaded
    return _plan_to_response(plan, asdict(targets), profile_id)


class RefineRequest(BaseModel):
    message: str
    profile_id: int
    messages: list[dict] = Field(default_factory=list)
    openai_api_key: str | None = None


class RefineResponse(BaseModel):
    message: str
    plan: PlanResponse
    messages: list[dict]
    done: bool


def _last_ai_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return str(message.content)
    return ""


@app.post("/refine", response_model=RefineResponse)
def refine(request: RefineRequest) -> RefineResponse:
    profile = load_profile(request.profile_id)
    loaded = load_plan(request.profile_id)
    if profile is None or loaded is None:
        raise HTTPException(status_code=404, detail="profile or plan not found")

    plan, targets = loaded

    history = messages_from_dict(request.messages)
    history.append(HumanMessage(content=request.message))

    initial_state: AgentState = {
        "messages": history,
        "profile": profile,
        "profile_id": request.profile_id,
        "targets": targets,
        "recipes": [],
        "plan": plan,
        "error": None,
        "done": False,
        "mode": "chat",
    }

    result = app.state.graph.invoke(
        initial_state,
        config={
            "recursion_limit": settings.recursion_limit,
            "configurable": {"openai_api_key": request.openai_api_key},
        },
    )

    updated_plan = result.get("plan") or plan
    result_targets = asdict(result["targets"]) if result.get("targets") is not None else {}
    out_messages = messages_to_dict(result.get("messages") or [])

    persisted = result.get("profile_id") != request.profile_id
    done = bool(result.get("done")) or persisted

    return RefineResponse(
        message=_last_ai_text(result.get("messages") or []),
        plan=_plan_to_response(updated_plan, result_targets, request.profile_id),
        messages=out_messages,
        done=done,
    )


def main() -> None:
    """Run the API server."""
    import uvicorn

    uvicorn.run(app, host=settings.fastapi_host, port=settings.fastapi_port)
