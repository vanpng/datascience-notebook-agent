"""Unit + integration tests for the agent pipeline."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agent.schemas import (
    AgentPlan,
    ExecutionResult,
    GeneratedCell,
    NotebookCell,
    PlanStep,
)
from src.agent.state import AgentState
from src.agent import nodes


# ── sandbox tests ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sandbox_success():
    from src.agent.sandbox import execute_code
    result = await execute_code("print('hello')")
    assert result.success
    assert "hello" in result.stdout


@pytest.mark.asyncio
async def test_sandbox_syntax_error():
    from src.agent.sandbox import execute_code
    result = await execute_code("def broken(:\n    pass")
    assert not result.success
    assert result.stderr


@pytest.mark.asyncio
async def test_sandbox_timeout():
    from src.agent.sandbox import execute_code
    result = await execute_code("import time; time.sleep(999)", timeout=2)
    assert not result.success
    assert result.timed_out


# ── node unit tests (mock LLM) ────────────────────────────────────────────────

def _base_state() -> AgentState:
    return {
        "session_id": "test-123",
        "notebook_cells": [],
        "dataframe_schemas": [],
        "user_query": "Print hello world",
        "session_context": None,
        "plan": None,
        "generated_cell": None,
        "execution_result": None,
        "debug_patch": None,
        "debug_attempts": 0,
        "max_debug_attempts": 3,
        "messages": [],
        "dry_run": False,
        "final_code": None,
        "agent_error": None,
        "status": "planning",
    }


@pytest.mark.asyncio
async def test_plan_node_parses_json():
    plan_json = (
        '{"reasoning":"simple","steps":[{"step":1,"description":"print",'
        '"imports_needed":[]}],"variables_needed":[],"variables_produced":[]}'
    )
    mock_response = MagicMock()
    mock_response.content = plan_json

    with patch.object(nodes, "_make_llm") as mock_llm_factory:
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm_factory.return_value = mock_llm

        updates = await nodes.plan_node(_base_state())

    assert isinstance(updates["plan"], AgentPlan)
    assert updates["plan"].steps[0].step == 1


@pytest.mark.asyncio
async def test_generate_node_parses_json():
    gen_json = '{"reasoning":"ok","imports":[],"code":"print(\'hello\')"}'
    mock_response = MagicMock()
    mock_response.content = gen_json

    state = _base_state()
    state["plan"] = AgentPlan(
        reasoning="test",
        steps=[PlanStep(step=1, description="print hello")],
    )

    with patch.object(nodes, "_make_llm") as mock_llm_factory:
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm_factory.return_value = mock_llm

        updates = await nodes.generate_node(state)

    assert isinstance(updates["generated_cell"], GeneratedCell)
    assert "print" in updates["generated_cell"].code


@pytest.mark.asyncio
async def test_execute_node_success():
    state = _base_state()
    state["generated_cell"] = GeneratedCell(
        reasoning="test", code="x = 1 + 1\nprint(x)"
    )

    updates = await nodes.execute_node(state)

    assert updates["execution_result"].success
    assert "2" in updates["execution_result"].stdout
    assert updates["status"] == "done"
    assert len(updates["notebook_cells"]) == 1


@pytest.mark.asyncio
async def test_execute_node_failure_increments_debug():
    state = _base_state()
    state["generated_cell"] = GeneratedCell(
        reasoning="test", code="raise ValueError('oops')"
    )

    updates = await nodes.execute_node(state)

    assert not updates["execution_result"].success
    assert updates["status"] == "debugging"


# ── graph integration test ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_graph_simple_query():
    """End-to-end: agent generates and executes a trivial cell without debug."""
    from src.agent.graph import agent_graph

    # build_context node LLM response
    ctx_json = '{"query_intent":"general","suggested_libraries":[]}'
    plan_json = (
        '{"reasoning":"print","steps":[{"step":1,"description":"print x",'
        '"imports_needed":[]}],"variables_needed":[],"variables_produced":["x"]}'
    )
    # Use a single-line code string to avoid JSON parse issues with literal newlines
    gen_json = '{"reasoning":"ok","code":"x = 42; print(x)"}'

    responses = [ctx_json, plan_json, gen_json]
    call_idx = {"n": 0}

    async def fake_ainvoke(messages, **_):
        r = MagicMock()
        r.content = responses[call_idx["n"] % len(responses)]
        call_idx["n"] += 1
        return r

    with patch.object(nodes, "_make_llm") as mock_llm_factory:
        mock_llm = AsyncMock()
        mock_llm.ainvoke = fake_ainvoke
        mock_llm_factory.return_value = mock_llm

        initial: AgentState = {
            "session_id": "int-test",
            "notebook_cells": [],
            "dataframe_schemas": [],
            "user_query": "Compute x = 42 and print it.",
            "session_context": None,
            "plan": None,
            "generated_cell": None,
            "execution_result": None,
            "debug_patch": None,
            "debug_attempts": 0,
            "max_debug_attempts": 3,
            "messages": [],
            "dry_run": False,
            "final_code": None,
            "agent_error": None,
            "status": "planning",
        }

        final = await agent_graph.ainvoke(initial)

    assert final["status"] == "done"
    assert "42" in final["execution_result"].stdout
    assert final["final_code"] is not None
