"""Tests that the agent does not re-emit imports that are already in the session.

The bug: SYSTEM_GENERATE said "all import statements at the top", which caused
every generated cell to repeat pandas/numpy/seaborn even when they were already
imported three cells ago.

The fix: the prompt now says "only import libraries NOT already listed under
`imports` in the Session Context".  These tests verify the prompt wording reaches
the LLM correctly, and that build_notebook_context exposes imported_libraries so
the model can act on the instruction.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent import nodes
from src.agent.schemas import (
    AgentPlan,
    NotebookCell,
    PlanStep,
    SessionContext,
    QueryIntent,
)


# ── build_notebook_context surfaces imported_libraries ────────────────────────

def test_context_lists_already_imported_libraries():
    """Session context must expose imported_libraries so the model can skip them."""
    from src.agent.prompts import build_notebook_context

    ctx_obj = SessionContext(
        has_data=True,
        query_intent=QueryIntent.visualization,
        available_variables=["df"],
        imported_libraries=["pandas", "numpy", "seaborn"],
        data_sources=["data.csv"],
        suggested_libraries=[],
    )
    ctx = build_notebook_context([], [], ctx_obj)
    assert "imports" in ctx
    assert "pandas" in ctx
    assert "numpy" in ctx
    assert "seaborn" in ctx


def test_context_omits_imports_section_when_empty():
    from src.agent.prompts import build_notebook_context

    ctx_obj = SessionContext(
        has_data=False,
        query_intent=QueryIntent.general,
        available_variables=[],
        imported_libraries=[],   # nothing imported yet
        data_sources=[],
        suggested_libraries=[],
    )
    ctx = build_notebook_context([], [], ctx_obj)
    # "imports" line should not appear when the list is empty
    assert "imports         :" not in ctx


# ── SYSTEM_GENERATE prompt wording ───────────────────────────────────────────

def test_system_generate_says_skip_already_imported():
    from src.agent.prompts import SYSTEM_GENERATE
    # The rule must tell the model not to repeat what is already imported
    assert "NOT already" in SYSTEM_GENERATE or "not already" in SYSTEM_GENERATE


def test_system_generate_no_longer_says_all_imports():
    """Old wording 'all import statements at the top' must be gone."""
    from src.agent.prompts import SYSTEM_GENERATE
    assert "all import statements" not in SYSTEM_GENERATE


# ── generate_node prompt contains imported_libraries ─────────────────────────

@pytest.mark.asyncio
async def test_generate_node_prompt_mentions_existing_imports():
    """The message sent to the LLM must list already-imported libraries."""
    captured_messages = []

    async def fake_ainvoke(messages, **_):
        captured_messages.extend(messages)
        r = MagicMock()
        r.content = '{"reasoning":"ok","code":"print(1)"}'
        return r

    state = {
        "session_id": "t",
        "notebook_cells": [
            NotebookCell(
                cell_id=0,
                source="import pandas as pd\nimport numpy as np\nimport seaborn as sns",
                stdout="",
                stderr="",
                success=True,
            )
        ],
        "dataframe_schemas": [],
        "user_query": "plot something",
        "session_context": SessionContext(
            has_data=True,
            query_intent=QueryIntent.visualization,
            available_variables=["df"],
            imported_libraries=["pandas", "numpy", "seaborn"],
            data_sources=[],
            suggested_libraries=["matplotlib"],
        ),
        "plan": AgentPlan(
            reasoning="test",
            steps=[PlanStep(step=1, description="plot")],
        ),
        "generated_cell": None,
        "execution_result": None,
        "debug_patch": None,
        "debug_attempts": 0,
        "max_debug_attempts": 3,
        "messages": [],
        "dry_run": False,
        "final_code": None,
        "agent_error": None,
        "status": "generating",
    }

    with patch.object(nodes, "_make_llm") as mock_factory:
        mock_llm = AsyncMock()
        mock_llm.ainvoke = fake_ainvoke
        mock_factory.return_value = mock_llm
        await nodes.generate_node(state)

    full_prompt = " ".join(m.content for m in captured_messages)
    # The LLM must see which libraries are already imported
    assert "pandas" in full_prompt
    assert "numpy" in full_prompt
    assert "seaborn" in full_prompt
    # It must also see the "NOT already" instruction
    assert "NOT already" in full_prompt or "not already" in full_prompt


# ── few-shot example demonstrates the correct behaviour ──────────────────────

def test_few_shot_generate_response_skips_already_imported_pandas():
    """The assistant example must NOT re-import pandas when it's in Session Context."""
    from src.agent.prompts import FEW_SHOT_GENERATE

    assistant_msg = next(m for m in FEW_SHOT_GENERATE if m["role"] == "assistant")
    # The user context shows pandas is imported; the response should only add matplotlib
    assert "import pandas" not in assistant_msg["content"]
    assert "import matplotlib" in assistant_msg["content"]


def test_few_shot_generate_user_context_shows_imported_libs():
    """The user message in the few-shot must list imported libraries."""
    from src.agent.prompts import FEW_SHOT_GENERATE

    user_msg = next(m for m in FEW_SHOT_GENERATE if m["role"] == "user")
    assert "imports" in user_msg["content"]
    assert "pandas" in user_msg["content"]
