"""Tests for sandbox state persistence (prior_sources replay) and dry_run mode.

These tests cover two bugs that were fixed:
1. Each sandbox execution was isolated — variables from prior cells were lost.
2. The Jupyter magic ran generated code in the sandbox instead of inserting it
   for the user to run; dry_run mode fixes this.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agent.sandbox import _build_script, execute_code
from src.agent.schemas import (
    AgentPlan,
    ExecutionResult,
    GeneratedCell,
    NotebookCell,
    PlanStep,
)
from src.agent.state import AgentState
from src.agent import nodes


# ── _build_script unit tests ──────────────────────────────────────────────────

def test_build_script_no_prior():
    script = _build_script([], "print('hi')")
    assert script == "print('hi')"


def test_build_script_suppresses_prior_stdout():
    script = _build_script(["print('should be hidden')"], "print('visible')")
    # stdout redirect must appear before the prior cell
    assert "_sys.stdout = _io.StringIO()" in script
    # stdout must be restored before the new cell
    assert "_sys.stdout = _sys.__stdout__" in script
    prior_pos = script.index("should be hidden")
    restore_pos = script.index("_sys.stdout = _sys.__stdout__")
    new_cell_pos = script.index("visible")
    assert prior_pos < restore_pos < new_cell_pos


def test_build_script_skips_blank_prior_sources():
    script = _build_script(["", "   ", "x = 1"], "print(x)")
    # blank entries are filtered — only "x = 1" should appear as prior
    assert "x = 1" in script
    assert script.count("StringIO") > 0   # suppression still added


# ── sandbox execute_code with prior_sources ───────────────────────────────────

@pytest.mark.asyncio
async def test_prior_variable_available_in_new_cell():
    """Variable defined in a prior cell must be accessible in the new cell."""
    result = await execute_code(
        code="print(x * 2)",
        prior_sources=["x = 21"],
    )
    assert result.success, result.stderr
    assert "42" in result.stdout


@pytest.mark.asyncio
async def test_prior_stdout_not_leaked():
    """Prior cells' print output must not appear in the new cell's stdout."""
    result = await execute_code(
        code="print('new_output')",
        prior_sources=["print('old_output')"],
    )
    assert result.success, result.stderr
    assert "new_output" in result.stdout
    assert "old_output" not in result.stdout


@pytest.mark.asyncio
async def test_multiple_prior_cells_chain():
    """Variables accumulate across several prior cells."""
    result = await execute_code(
        code="print(a + b + c)",
        prior_sources=["a = 1", "b = 2", "c = 3"],
    )
    assert result.success, result.stderr
    assert "6" in result.stdout


@pytest.mark.asyncio
async def test_import_in_prior_cell_available():
    """An import in a prior cell must be usable in the new cell."""
    result = await execute_code(
        code="print(type(pd.DataFrame()))",
        prior_sources=["import pandas as pd"],
    )
    assert result.success, result.stderr
    assert "DataFrame" in result.stdout


@pytest.mark.asyncio
async def test_no_prior_sources_unchanged():
    """With no prior_sources the new cell runs normally (regression guard)."""
    result = await execute_code("print('solo')")
    assert result.success
    assert "solo" in result.stdout


# ── execute_node passes prior cells to sandbox ────────────────────────────────

def _make_state(cells: list[NotebookCell], code: str) -> AgentState:
    return {
        "session_id": "t",
        "notebook_cells": cells,
        "dataframe_schemas": [],
        "user_query": "test",
        "session_context": None,
        "plan": None,
        "generated_cell": GeneratedCell(reasoning="", code=code),
        "execution_result": None,
        "debug_patch": None,
        "debug_attempts": 0,
        "max_debug_attempts": 3,
        "messages": [],
        "dry_run": False,
        "final_code": None,
        "agent_error": None,
        "status": "executing",
    }


@pytest.mark.asyncio
async def test_execute_node_sees_prior_cell_variable():
    """execute_node must pass notebook_cells as prior_sources to the sandbox."""
    prior = NotebookCell(
        cell_id=0, source="df_size = 99", stdout="", stderr="", success=True
    )
    state = _make_state([prior], "print(df_size)")
    updates = await nodes.execute_node(state)
    assert updates["execution_result"].success, updates["execution_result"].stderr
    assert "99" in updates["execution_result"].stdout


@pytest.mark.asyncio
async def test_execute_node_failed_prior_cell_excluded():
    """Cells with success=False must not be replayed."""
    bad_prior = NotebookCell(
        cell_id=0, source="raise RuntimeError('boom')", stdout="", stderr="err", success=False
    )
    state = _make_state([bad_prior], "print('ok')")
    updates = await nodes.execute_node(state)
    assert updates["execution_result"].success
    assert "ok" in updates["execution_result"].stdout


# ── dry_run mode ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dry_run_skips_execution():
    """With dry_run=True the graph must return status='done' without executing."""
    from src.agent.graph import agent_graph

    build_ctx_json = '{"query_intent":"general","suggested_libraries":[]}'
    plan_json = (
        '{"reasoning":"p","steps":[{"step":1,"description":"d",'
        '"imports_needed":[]}],"variables_needed":[],"variables_produced":[]}'
    )
    gen_json = '{"reasoning":"r","code":"raise RuntimeError(\'should not run\')"}'

    responses = [build_ctx_json, plan_json, gen_json]
    call_idx = {"n": 0}

    async def fake_ainvoke(messages, **_):
        r = MagicMock()
        r.content = responses[call_idx["n"] % len(responses)]
        call_idx["n"] += 1
        return r

    with patch.object(nodes, "_make_llm") as mock_factory:
        mock_llm = AsyncMock()
        mock_llm.ainvoke = fake_ainvoke
        mock_factory.return_value = mock_llm

        initial: AgentState = {
            "session_id": "dry-test",
            "notebook_cells": [],
            "dataframe_schemas": [],
            "user_query": "test",
            "session_context": None,
            "plan": None,
            "generated_cell": None,
            "execution_result": None,
            "debug_patch": None,
            "debug_attempts": 0,
            "max_debug_attempts": 3,
            "messages": [],
            "dry_run": True,
            "final_code": None,
            "agent_error": None,
            "status": "planning",
        }

        final = await agent_graph.ainvoke(initial)

    assert final["status"] == "done"
    assert "should not run" in final["final_code"]
    # execution_result must be absent — no sandbox was used
    assert final.get("execution_result") is None


@pytest.mark.asyncio
async def test_dry_run_false_executes_code():
    """With dry_run=False the graph must execute the generated code normally."""
    from src.agent.graph import agent_graph

    build_ctx_json = '{"query_intent":"general","suggested_libraries":[]}'
    plan_json = (
        '{"reasoning":"p","steps":[{"step":1,"description":"d",'
        '"imports_needed":[]}],"variables_needed":[],"variables_produced":[]}'
    )
    gen_json = '{"reasoning":"r","code":"print(7 * 6)"}'

    responses = [build_ctx_json, plan_json, gen_json]
    call_idx = {"n": 0}

    async def fake_ainvoke(messages, **_):
        r = MagicMock()
        r.content = responses[call_idx["n"] % len(responses)]
        call_idx["n"] += 1
        return r

    with patch.object(nodes, "_make_llm") as mock_factory:
        mock_llm = AsyncMock()
        mock_llm.ainvoke = fake_ainvoke
        mock_factory.return_value = mock_llm

        initial: AgentState = {
            "session_id": "exec-test",
            "notebook_cells": [],
            "dataframe_schemas": [],
            "user_query": "print 42",
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
    assert final["execution_result"] is not None
    assert "42" in final["execution_result"].stdout


# ── multi-turn integration: second query can use first query's variables ───────

@pytest.mark.asyncio
async def test_multi_turn_variable_persistence():
    """Simulate two consecutive API queries sharing session state.

    First query defines `result = 100`.
    Second query prints `result` — must succeed without NameError.
    """
    # Seed the session with the first query's executed cell
    prior_cell = NotebookCell(
        cell_id=0,
        source="result = 100",
        stdout="",
        stderr="",
        success=True,
    )
    state = _make_state([prior_cell], "print(result + 1)")
    updates = await nodes.execute_node(state)
    assert updates["execution_result"].success, updates["execution_result"].stderr
    assert "101" in updates["execution_result"].stdout
