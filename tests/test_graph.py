"""Tests for graph routing and construction."""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from src.graph import (
    _last_ai_has_tool_calls,
    _last_human_approves,
    build_graph,
    route_from_orchestrator,
    route_from_refine,
)
from src.state import AgentState


def make_state(messages: list, **overrides) -> AgentState:
    state: AgentState = {
        "messages": messages,
        "profile": None,
        "profile_id": None,
        "plan_id": None,
        "targets": None,
        "recipes": [],
        "plan": None,
        "error": None,
        "done": False,
        "mode": "api",
        "openai_api_key": None,
    }
    state.update(overrides)
    return state


def _ai_with_tool_calls() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "search_recipes", "args": {}, "id": "call_1"}],
    )


def test_last_ai_has_tool_calls_true() -> None:
    assert _last_ai_has_tool_calls(make_state([_ai_with_tool_calls()])) is True


def test_last_ai_has_tool_calls_false_without_calls() -> None:
    assert _last_ai_has_tool_calls(make_state([AIMessage(content="hi")])) is False


def test_last_ai_has_tool_calls_false_when_no_ai_message() -> None:
    assert _last_ai_has_tool_calls(make_state([HumanMessage(content="hi")])) is False


def test_last_ai_has_tool_calls_uses_most_recent_ai() -> None:
    messages = [_ai_with_tool_calls(), AIMessage(content="final answer")]
    assert _last_ai_has_tool_calls(make_state(messages)) is False


def test_last_human_approves_detects_phrase_case_insensitive() -> None:
    assert _last_human_approves(make_state([HumanMessage(content="Looks GOOD, thanks")])) is True


def test_last_human_approves_false_for_plain_message() -> None:
    assert _last_human_approves(make_state([HumanMessage(content="swap the lunch")])) is False


def test_route_to_tools_when_ai_requests_them() -> None:
    assert route_from_orchestrator(make_state([_ai_with_tool_calls()])) == "tools"


def test_route_chat_approval_goes_to_persist() -> None:
    state = make_state([HumanMessage(content="save it")], mode="chat")
    assert route_from_orchestrator(state) == "persist"


def test_route_api_done_goes_to_persist() -> None:
    state = make_state([AIMessage(content="done")], mode="api", done=True)
    assert route_from_orchestrator(state) == "persist"


def test_route_chat_done_goes_to_refine() -> None:
    state = make_state([AIMessage(content="done")], mode="chat", done=True)
    assert route_from_orchestrator(state) == "refine"


def test_route_error_loops_back_to_orchestrator() -> None:
    state = make_state([AIMessage(content="oops")], error="boom")
    assert route_from_orchestrator(state) == "orchestrator"


def test_route_default_ends() -> None:
    assert route_from_orchestrator(make_state([AIMessage(content="hello")])) == END


def test_route_from_refine_persist_on_approval() -> None:
    assert route_from_refine(make_state([HumanMessage(content="perfect")])) == "persist"


def test_route_from_refine_loops_otherwise() -> None:
    assert route_from_refine(make_state([HumanMessage(content="change it")])) == "orchestrator"


def test_build_graph_compiles_with_expected_nodes() -> None:
    graph = build_graph()
    nodes = set(graph.get_graph().nodes)
    for name in ("orchestrator", "tools", "parse_tool_results", "refine", "persist"):
        assert name in nodes
