"""FastAPI agent server."""
from __future__ import annotations

import os
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

load_dotenv()

from src.agent.graph import agent_graph
from src.agent.schemas import NotebookCell
from src.api.models import (
    AddCellRequest,
    CreateSessionRequest,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SessionResponse,
)

logger = logging.getLogger("agent_api")
logging.basicConfig(level=logging.INFO)

# In-memory session store (replace with Redis for multi-process deployments)
_sessions: Dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("DS Notebook Agent API starting up")
    yield
    logger.info("DS Notebook Agent API shutting down")


app = FastAPI(
    title="DS Notebook Agent",
    description="LangGraph agentic pipeline for data science notebook assistance",
    version="0.1.0",
    lifespan=lifespan,
)


# ── health ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


# ── sessions ──────────────────────────────────────────────────────────────────
@app.post("/sessions", response_model=SessionResponse)
async def create_session(req: CreateSessionRequest):
    sid = req.session_id or str(uuid.uuid4())
    if sid in _sessions:
        raise HTTPException(status_code=409, detail="Session already exists")

    _sessions[sid] = {
        "session_id": sid,
        "notebook_cells": req.notebook_cells,
        "dataframe_schemas": req.dataframe_schemas,
        "max_debug_attempts": req.max_debug_attempts,
        "status": "idle",
        "messages": [],
    }
    return _session_response(sid)


@app.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    _require_session(session_id)
    return _session_response(session_id)


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    _require_session(session_id)
    del _sessions[session_id]
    return JSONResponse({"deleted": session_id})


# ── cells (context management) ────────────────────────────────────────────────
@app.post("/sessions/{session_id}/cells", response_model=SessionResponse)
async def add_cell(session_id: str, req: AddCellRequest):
    """Manually push an already-executed cell into the session context."""
    _require_session(session_id)
    sess = _sessions[session_id]
    cells: list[NotebookCell] = sess["notebook_cells"]
    new_id = (cells[-1].cell_id + 1) if cells else 0
    cells.append(
        NotebookCell(
            cell_id=new_id,
            source=req.source,
            stdout=req.stdout,
            stderr=req.stderr,
            success=req.success,
        )
    )
    return _session_response(session_id)


# ── query ─────────────────────────────────────────────────────────────────────
@app.post("/sessions/{session_id}/query", response_model=QueryResponse)
async def run_query(session_id: str, req: QueryRequest):
    """Run the full Plan→Generate→Execute→Debug loop for one user query."""
    _require_session(session_id)
    sess = _sessions[session_id]

    initial_state = {
        **sess,
        "user_query": req.query,
        "dry_run": not req.execute,
        "completion_mode": req.completion_mode,
        "debug_mode": req.debug_mode,
        "prompt_profile": req.prompt_profile,
        "plan": None,
        "generated_cell": None,
        "execution_result": None,
        "debug_patch": None,
        "debug_attempts": 0,
        "final_code": None,
        "agent_error": None,
        "status": "planning",
    }

    try:
        final_state = await agent_graph.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("Agent graph error")
        return QueryResponse(
            session_id=session_id,
            status="failed",
            agent_error=str(exc),
        )

    # Persist updated cells back to session
    sess["notebook_cells"] = final_state.get("notebook_cells", sess["notebook_cells"])
    sess["dataframe_schemas"] = final_state.get("dataframe_schemas", sess["dataframe_schemas"])
    sess["status"] = final_state.get("status", "idle")
    sess["messages"] = final_state.get("messages", [])

    result = final_state.get("execution_result")
    # Return the last generated code even when execution failed, so callers
    # (e.g. the submission writer) can include it as a reasoning trace.
    generated_cell = final_state.get("generated_cell")
    final_code = final_state.get("final_code") or (
        generated_cell.code if generated_cell else None
    )
    return QueryResponse(
        session_id=session_id,
        status=final_state.get("status", "unknown"),
        final_code=final_code,
        stdout=result.stdout if result else None,
        stderr=result.stderr if result else None,
        agent_error=final_state.get("agent_error"),
        debug_attempts=final_state.get("debug_attempts", 0),
    )


# ── helpers ───────────────────────────────────────────────────────────────────
def _require_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")


def _session_response(session_id: str) -> SessionResponse:
    sess = _sessions[session_id]
    return SessionResponse(
        session_id=session_id,
        notebook_cells=sess["notebook_cells"],
        dataframe_schemas=sess["dataframe_schemas"],
        status=sess.get("status", "idle"),
    )


def main():
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("AGENT_API_HOST", "127.0.0.1"),
        port=int(os.getenv("AGENT_API_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
