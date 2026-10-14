"""Application settings"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # LLM
    openai_api_key: str
    model_name: str = "gpt-4o-mini"
    model_temperature: float = 0.3

    # Recipe API
    spoonacular_api_key: str
    spoonacular_base_url: str = "https://api.spoonacular.com"

    # Observability
    langsmith_tracing: bool = True
    langsmith_api_key: str = ""
    langsmith_project: str = "recipe-agent"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # App
    fastapi_host: str = "0.0.0.0"
    fastapi_port: int = 8000
    database_url: str = "sqlite:///./recipe_agent.db"
    recursion_limit: int = 50
    default_plan_days: int = 7


settings = Settings()

from src.tracing import configure_tracing

configure_tracing(settings)
