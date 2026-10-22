"""LangSmith tracing configuration."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from langsmith import Client

if TYPE_CHECKING:
    from src.config import Settings


SENSITIVE_KEYS = frozenset({"openai_api_key", "api_key"})
_REDACTED = "***REDACTED***"


def _redact(data: Any) -> Any:
    """Recursively replace values of sensitive keys with a placeholder."""
    if isinstance(data, dict):
        return {k: (_REDACTED if k in SENSITIVE_KEYS else _redact(v)) for k, v in data.items()}
    if isinstance(data, list):
        return [_redact(x) for x in data]
    if isinstance(data, tuple):
        return tuple(_redact(x) for x in data)
    return data


def configure_tracing(settings: Settings) -> None:
    """Export LangSmith settings and install a redacting client."""
    os.environ["LANGSMITH_TRACING"] = "true" if settings.langsmith_tracing else "false"
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key

    if settings.langsmith_tracing:
        _install_redacting_client()


def _install_redacting_client() -> None:
    """Patch LangChainTracer to use a Client that redacts sensitive fields."""
    from langchain_core.tracers import langchain as lc_tracers

    client = Client(hide_inputs=_redact, hide_outputs=_redact)
    original_init = lc_tracers.LangChainTracer.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("client", client)
        original_init(self, *args, **kwargs)

    lc_tracers.LangChainTracer.__init__ = patched_init  # type: ignore[method-assign]
