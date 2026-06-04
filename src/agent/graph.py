"""LangGraph workflow: Plan → Generate → Execute ↔ Debug (max N retries)."""
from __future__ import annotations

import os
from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    build_context_node,
    debug_node,
    execute_node,
    generate_node,
    plan_node,
)
from src.agent.state import AgentState


def _route_after_build_context(state: AgentState) -> Literal["plan", "generate", "debug"]:
    """Route after context is built.

    - debug_mode=True  → jump straight to debug_node to fix a known failure.
      The failed code and error are read from the last failed notebook cell.
    - completion_mode=True → skip plan_node, go straight to generate.
      Used for code-completion benchmarks (DS-1000) where plan_node would
      incorrectly produce 'load data first' boilerplate.
    - default → full plan → generate pipeline.
    """
    if state.get("debug_mode", False):
        return "debug"
    if state.get("completion_mode", False):
        return "generate"
    return "plan"


def _route_after_generate(state: AgentState) -> Literal["execute", "__end__"]:
    """Skip execution when caller requested dry_run (e.g. Jupyter magic)."""
    if state.get("dry_run", False):
        return "__end__"
    return "execute"


def _dry_run_finalise(state: AgentState) -> dict:
    """Promote generated code to final_code so the caller can surface it."""
    cell = state.get("generated_cell")
    return {
        "final_code": cell.code if cell else "",
        "status": "done",
    }


def _route_after_execute(
    state: AgentState,
) -> Literal["debug", "__end__", "failed"]:
    result = state.get("execution_result")
    attempts = state.get("debug_attempts", 0)
    max_attempts = state.get("max_debug_attempts", int(os.getenv("MAX_DEBUG_ATTEMPTS", "3")))

    if result and result.success:
        return "__end__"
    if attempts < max_attempts:
        return "debug"
    return "failed"


def _fail_node(state: AgentState) -> dict:
    result = state.get("execution_result")
    stderr = result.stderr if result else "unknown error"
    return {
        "agent_error": f"All debug attempts exhausted. Last error:\n{stderr}",
        "status": "failed",
    }


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("build_context",   build_context_node)
    g.add_node("plan",            plan_node)
    g.add_node("generate",        generate_node)
    g.add_node("dry_run_end",     _dry_run_finalise)
    g.add_node("execute",         execute_node)
    g.add_node("debug",           debug_node)
    g.add_node("failed",          _fail_node)

    g.add_edge(START,             "build_context")
    g.add_conditional_edges(
        "build_context",
        _route_after_build_context,
        {"plan": "plan", "generate": "generate", "debug": "debug"},
    )
    g.add_edge("plan",            "generate")
    g.add_conditional_edges(
        "generate",
        _route_after_generate,
        {
            "__end__":  "dry_run_end",
            "execute":  "execute",
        },
    )
    g.add_edge("dry_run_end", END)
    g.add_conditional_edges(
        "execute",
        _route_after_execute,
        {
            "__end__": END,
            "debug":   "debug",
            "failed":  "failed",
        },
    )
    g.add_edge("debug",   "execute")   # retry loop
    g.add_edge("failed",  END)

    return g


# Compile once at import time; callers just invoke `agent_graph`
agent_graph = build_graph().compile()
