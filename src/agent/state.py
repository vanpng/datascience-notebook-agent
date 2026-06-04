"""LangGraph agent state definition."""
from __future__ import annotations

from typing import Annotated, List, Optional

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from src.agent.schemas import (
    AgentPlan,
    DataFrameSchema,
    DebugPatch,
    ExecutionResult,
    GeneratedCell,
    NotebookCell,
    SessionContext,
)


class AgentState(TypedDict):
    # ── session context ───────────────────────────────────────────────────────
    session_id: str
    notebook_cells: List[NotebookCell]          # previously executed cells
    dataframe_schemas: List[DataFrameSchema]    # auto-extracted df metadata
    user_query: str
    session_context: Optional[SessionContext]   # built by build_context_node

    # ── per-turn working state ────────────────────────────────────────────────
    plan: Optional[AgentPlan]
    generated_cell: Optional[GeneratedCell]
    execution_result: Optional[ExecutionResult]
    debug_patch: Optional[DebugPatch]
    debug_attempts: int
    max_debug_attempts: int

    # ── message history (LangGraph managed) ──────────────────────────────────
    messages: Annotated[List[BaseMessage], add_messages]

    # ── execution mode ───────────────────────────────────────────────────────
    dry_run: bool                        # True → stop after generate (no sandbox)
    completion_mode: bool                # True → skip plan_node (code completion benchmarks)
    debug_mode: bool                     # True → skip to debug_node (fix a known failure)
    prompt_profile: str                  # "default" | "ds1000" | "dabstep" | "dscodebench" — selects prompt set

    # ── terminal state ────────────────────────────────────────────────────────
    final_code: Optional[str]           # accepted, executed cell source
    agent_error: Optional[str]          # fatal error message for the caller
    status: str                         # "planning" | "generating" | "executing"
                                        # | "debugging" | "done" | "failed"
