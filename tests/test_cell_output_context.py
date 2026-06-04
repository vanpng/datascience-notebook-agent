"""Tests for cell output capture and its propagation into the LLM context.

Covers:
1. _StdoutTee — forwards to real stream AND captures
2. _pre_run_cell / _post_run_cell hooks — populate _cell_outputs correctly
3. _sync_cells — passes captured stdout when pushing cells to the session
4. build_notebook_context — renders stdout in the prompt string
5. End-to-end: generate_node sees prior-cell stdout when choosing column names
"""
from __future__ import annotations

import io
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── _StdoutTee ────────────────────────────────────────────────────────────────

def test_tee_writes_to_both_streams():
    from src.agent.magic import _StdoutTee
    real_buf = io.StringIO()
    cap_buf  = io.StringIO()
    tee = _StdoutTee(real_buf, cap_buf)
    tee.write("hello")
    assert real_buf.getvalue() == "hello"
    assert cap_buf.getvalue()  == "hello"


def test_tee_delegates_flush():
    from src.agent.magic import _StdoutTee
    flushed = []
    class _FakeStream:
        def write(self, s): return len(s)
        def flush(self): flushed.append(True)
    tee = _StdoutTee(_FakeStream(), io.StringIO())
    tee.flush()
    assert flushed == [True]


def test_tee_delegates_unknown_attrs():
    from src.agent.magic import _StdoutTee
    class _FakeStream:
        def write(self, s): return len(s)
        def flush(self): pass
        custom = "sentinel"
    tee = _StdoutTee(_FakeStream(), io.StringIO())
    assert tee.custom == "sentinel"


# ── pre/post_run_cell hooks ───────────────────────────────────────────────────

def test_pre_run_replaces_stdout_with_tee():
    from src.agent.magic import _StdoutTee, _pre_run_cell, _capture_state
    original = sys.stdout
    try:
        _pre_run_cell(None)
        assert isinstance(sys.stdout, _StdoutTee)
    finally:
        # restore even if assertion fails
        if isinstance(sys.stdout, _StdoutTee):
            sys.stdout = original
        _capture_state["real_stdout"] = None
        _capture_state["buf"] = None


def test_post_run_restores_stdout_and_stores_output():
    from src.agent.magic import (
        _StdoutTee, _pre_run_cell, _post_run_cell,
        _cell_outputs, _capture_state,
    )
    _cell_outputs.clear()
    original = sys.stdout
    try:
        _pre_run_cell(None)
        sys.stdout.write("captured line\n")

        result = MagicMock()
        result.execution_count = 5
        result.result = None
        _post_run_cell(result)

        assert sys.stdout is original
        assert _cell_outputs.get(5) == "captured line\n"
    finally:
        if isinstance(sys.stdout, _StdoutTee):
            sys.stdout = original
        _capture_state["real_stdout"] = None
        _capture_state["buf"] = None


def test_post_run_appends_expression_repr():
    from src.agent.magic import (
        _pre_run_cell, _post_run_cell,
        _cell_outputs, _capture_state, _StdoutTee,
    )
    _cell_outputs.clear()
    original = sys.stdout
    try:
        _pre_run_cell(None)
        # no print output, but cell had an expression result
        result = MagicMock()
        result.execution_count = 7
        result.result = [1, 2, 3]    # Out[7] = [1, 2, 3]
        _post_run_cell(result)

        stored = _cell_outputs.get(7, "")
        assert "[1, 2, 3]" in stored
    finally:
        if isinstance(sys.stdout, _StdoutTee):
            sys.stdout = original
        _capture_state["real_stdout"] = None
        _capture_state["buf"] = None


def test_post_run_combines_print_and_expression():
    from src.agent.magic import (
        _pre_run_cell, _post_run_cell,
        _cell_outputs, _capture_state, _StdoutTee,
    )
    _cell_outputs.clear()
    original = sys.stdout
    try:
        _pre_run_cell(None)
        sys.stdout.write("print output\n")

        result = MagicMock()
        result.execution_count = 9
        result.result = 42
        _post_run_cell(result)

        stored = _cell_outputs.get(9, "")
        assert "print output" in stored
        assert "42" in stored
    finally:
        if isinstance(sys.stdout, _StdoutTee):
            sys.stdout = original
        _capture_state["real_stdout"] = None
        _capture_state["buf"] = None


def test_pre_run_does_not_double_wrap():
    """Calling pre_run_cell twice must not nest Tees."""
    from src.agent.magic import (
        _pre_run_cell, _post_run_cell,
        _cell_outputs, _capture_state, _StdoutTee,
    )
    original = sys.stdout
    try:
        _pre_run_cell(None)
        first_tee = sys.stdout
        _pre_run_cell(None)          # second call — should be a no-op
        assert sys.stdout is first_tee
    finally:
        sys.stdout = original
        _capture_state["real_stdout"] = None
        _capture_state["buf"] = None


# ── _sync_cells passes stdout ─────────────────────────────────────────────────

def _make_mock_shell(history: list[str]) -> MagicMock:
    shell = MagicMock()
    shell.history_manager.input_hist_parsed = [""] + history  # index 0 is always empty
    return shell


def test_sync_cells_passes_captured_stdout(monkeypatch):
    from src.agent import magic as magic_mod

    magic_mod._cell_outputs.clear()
    magic_mod._cell_outputs[1] = "shape: (100, 5)"   # In [1] had this stdout

    posted = []

    class _FakeClient:
        session_id = "s1"
        def post(self, url, json=None):
            posted.append(json)
            r = MagicMock()
            r.raise_for_status = lambda: r
            return r

    shell = _make_mock_shell(["import pandas as pd\ndf = pd.read_csv('x.csv')"])
    instance = magic_mod.DSAgentMagic.__new__(magic_mod.DSAgentMagic)
    instance.shell = shell
    instance._client = _FakeClient()
    instance._session_id = "s1"
    instance._synced_up_to = 0

    instance._sync_cells()

    assert len(posted) == 1
    assert posted[0]["stdout"] == "shape: (100, 5)"


def test_sync_cells_empty_stdout_when_no_capture(monkeypatch):
    from src.agent import magic as magic_mod

    magic_mod._cell_outputs.clear()   # no captured output for any cell

    posted = []

    class _FakeClient:
        session_id = "s1"
        def post(self, url, json=None):
            posted.append(json)
            r = MagicMock()
            r.raise_for_status = lambda: r
            return r

    shell = _make_mock_shell(["x = 1"])
    instance = magic_mod.DSAgentMagic.__new__(magic_mod.DSAgentMagic)
    instance.shell = shell
    instance._client = _FakeClient()
    instance._session_id = "s1"
    instance._synced_up_to = 0

    instance._sync_cells()

    assert posted[0]["stdout"] == ""


# ── build_notebook_context renders stdout ────────────────────────────────────

def test_build_notebook_context_includes_stdout():
    from src.agent.prompts import build_notebook_context
    from src.agent.schemas import NotebookCell

    cells = [
        NotebookCell(
            cell_id=1,
            source="df = pd.read_csv('data.csv')\nprint(df.shape)",
            stdout="(300, 11)",
            stderr="",
            success=True,
        )
    ]
    ctx = build_notebook_context(cells, [], None)
    assert "[Cell 1] stdout:" in ctx
    assert "(300, 11)" in ctx


def test_build_notebook_context_stdout_truncated_at_1500():
    from src.agent.prompts import build_notebook_context
    from src.agent.schemas import NotebookCell

    # Use a string whose tail is uniquely identifiable: 1500 "a"s then 500 "Z"s.
    # After truncation at 1500 the "Z" section must not appear.
    long_output = "a" * 1500 + "Z" * 500
    cells = [
        NotebookCell(cell_id=0, source="pass", stdout=long_output, stderr="", success=True)
    ]
    ctx = build_notebook_context(cells, [], None)
    assert "…" in ctx
    assert "Z" not in ctx      # everything past position 1500 was cut


def test_build_notebook_context_no_stdout_when_empty():
    from src.agent.prompts import build_notebook_context
    from src.agent.schemas import NotebookCell

    cells = [NotebookCell(cell_id=0, source="x = 1", stdout="", stderr="", success=True)]
    ctx = build_notebook_context(cells, [], None)
    assert "stdout" not in ctx


def test_build_notebook_context_failed_cell_shows_stderr():
    from src.agent.prompts import build_notebook_context
    from src.agent.schemas import NotebookCell

    cells = [
        NotebookCell(
            cell_id=0,
            source="df.groupby('BadCol').mean()",
            stdout="",
            stderr="KeyError: 'BadCol'",
            success=False,
        )
    ]
    ctx = build_notebook_context(cells, [], None)
    assert "[Cell 0] stderr:" in ctx
    assert "KeyError" in ctx


# ── generate_node uses stdout from prior cell ─────────────────────────────────

@pytest.mark.asyncio
async def test_generate_node_prompt_contains_prior_stdout():
    """generate_node must include prior-cell stdout in the message it sends to the LLM."""
    from src.agent import nodes
    from src.agent.schemas import AgentPlan, NotebookCell, PlanStep

    captured_messages = []

    async def fake_ainvoke(messages, **_):
        captured_messages.extend(messages)
        r = MagicMock()
        r.content = '{"reasoning":"ok","code":"print(1)"}'
        return r

    prior_cell = NotebookCell(
        cell_id=0,
        source="df = pd.read_csv('aita.csv')\nprint(df.dtypes)",
        stdout=(
            "author_gender     object\n"
            "score              int64\n"
            "verdict           object\n"
            "dtype: object"
        ),
        stderr="",
        success=True,
    )

    state = {
        "session_id": "t",
        "notebook_cells": [prior_cell],
        "dataframe_schemas": [],
        "user_query": "plot verdict distribution",
        "session_context": None,
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

    # The LLM must have received the prior cell's stdout in its prompt
    full_prompt = " ".join(m.content for m in captured_messages)
    assert "author_gender" in full_prompt
    assert "object" in full_prompt
