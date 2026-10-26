# Recipe Discovery Agent

## What it does

Takes a user profile (age, height, weight, sex, goal, dietary restrictions) and computes
daily calorie and macro targets, then calls the Spoonacular API to find real recipes that
fit those targets and constraints. A LangGraph agent assembles the results into a weekly
meal plan and persists the profile and plan to SQLite, so they survive restarts and can be
retrieved or refined later.

## Quickstart (Docker)

1. `git clone <repo-url> && cd ai_planner`
2. `cp .env.example .env` — fill in `OPENAI_API_KEY` and `SPOONACULAR_API_KEY`
3. `docker compose up --build`
4. Open http://localhost:8501

## Local setup (without Docker)

Create and activate a virtual environment from the repository root.

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install backend dependencies, including backend development tools, and frontend dependencies, including frontend test tools.

```bash
pip install -e ".[dev]"
pip install -r frontend/requirements-dev.txt
```

Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY` and `SPOONACULAR_API_KEY`.

Run the backend and frontend in separate terminals, with the virtual environment activated in each.

```bash
# Terminal 1: FastAPI backend
uvicorn src.api:app --reload

# Terminal 2: Streamlit frontend
streamlit run frontend/app.py
```

## Tests

The dev dependencies include the test tooling, and the suite runs offline: `tests/conftest.py`
sets dummy API keys.

Install the development dependencies first:

```bash
pip install -e ".[dev]"
pip install -r frontend/requirements-dev.txt
```

Run the test suite from the repository root:

```bash
python -m pytest
```

## Environment variables

| Variable | Description | Required |
|---|---|---|
| `OPENAI_API_KEY` | API key for the OpenAI model that drives the agent | **Yes** |
| `SPOONACULAR_API_KEY` | API key for the Spoonacular recipe API | **Yes** |
| `MODEL_NAME` | OpenAI model to use | No (default `gpt-4o-mini`) |
| `MODEL_TEMPERATURE` | Sampling temperature for the model | No (default `0.3`) |
| `SPOONACULAR_BASE_URL` | Base URL for the Spoonacular API | No (default `https://api.spoonacular.com`) |
| `SPOONACULAR_TIMEOUT` | HTTP timeout (seconds) for Spoonacular requests | No (default `10.0`) |
| `LANGSMITH_TRACING` | Enable LangSmith tracing of nodes/tool calls | No (default `true`) |
| `LANGSMITH_API_KEY` | LangSmith API key (needed only when tracing is enabled) | No (default empty) |
| `LANGSMITH_PROJECT` | LangSmith project name traces are logged to | No (default `recipe-agent`) |
| `LANGSMITH_ENDPOINT` | LangSmith API endpoint | No (default `https://api.smith.langchain.com`) |
| `FASTAPI_HOST` | Host the API server binds to | No (default `0.0.0.0`) |
| `FASTAPI_PORT` | Port the API server binds to | No (default `8000`) |
| `DATABASE_URL` | SQLAlchemy URL for the SQLite database | No (default `sqlite:///./recipe_agent.db`) |
| `RECURSION_LIMIT` | Max LangGraph steps per invocation | No (default `50`) |
| `DEFAULT_PLAN_DAYS` | Default number of days in a plan | No (default `7`) |
| `API_BASE_URL` | Base URL the Streamlit frontend uses to reach the backend | No (default `http://localhost:8000`) |
| `PLAN_REQUEST_TIMEOUT` | Frontend HTTP timeout (seconds) for `POST /plan` | No (default `120.0`) |
| `REFINE_REQUEST_TIMEOUT` | Frontend HTTP timeout (seconds) for `POST /refine` | No (default `60.0`) |

## Architecture

### Stack at a glance

| Layer | Choice | Why |
|---|---|---|
| Frontend | Streamlit | Fastest path to a real UI; chat-friendly; no JS needed |
| Backend / Agent | FastAPI + LangGraph | FastAPI exposes the agent as an HTTP API; Streamlit calls it |
| LLM | OpenAI GPT-4o-mini | Cheap, fast, good tool-calling |
| Recipe data | Spoonacular API | Free tier (150 req/day), returns real URLs, nutrition, ingredients |
| Observability | LangSmith | Traces every node/tool call; 1 env var to enable |
| Storage | SQLite (local) | Persist user profile + past plans; zero infra |
| Config | pydantic-settings | Reads .env, validates types, no scattered os.getenv() calls |
| Packaging | pyproject.toml | Modern Python packaging; replaces requirements.txt |

### Graph flow

The agent is a LangGraph state machine. The **orchestrator** node is the main LLM loop: it
reads the conversation and calls tools (`calculate_targets`, `search_recipes`,
`filter_recipes`, `build_plan`). After each tool batch, **parse_tool_results** lifts the
structured artifacts returned by those tools into `AgentState` (targets, recipes, and the
finished plan), and flags the run done once a plan exists. When done, **persist** writes the
profile and plan (with its targets) to SQLite. In chat mode a **refine** node handles
follow-up questions, and an approval phrase from the user routes straight to persist; in API
mode the graph goes straight from a finished plan to persist and returns.
