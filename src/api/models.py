"""FastAPI request / response models."""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

from src.agent.schemas import DataFrameSchema, NotebookCell


class CreateSessionRequest(BaseModel):
    session_id: Optional[str] = None     # auto-generated if omitted
    notebook_cells: List[NotebookCell] = Field(default_factory=list)
    dataframe_schemas: List[DataFrameSchema] = Field(default_factory=list)
    max_debug_attempts: int = 3


class SessionResponse(BaseModel):
    session_id: str
    notebook_cells: List[NotebookCell]
    dataframe_schemas: List[DataFrameSchema]
    status: str


class QueryRequest(BaseModel):
    query: str
    stream: bool = False      # SSE streaming (not yet implemented; placeholder)
    execute: bool = True      # False → stop after generate, return code without running it
    completion_mode: bool = False  # True → skip plan_node (for code completion benchmarks like DS-1000)
    debug_mode: bool = False  # True → skip straight to debug_node to fix a known failure
    prompt_profile: str = "default"  # "default" | "ds1000" | "dabstep" | "dscodebench" — selects prompt set


class QueryResponse(BaseModel):
    session_id: str
    status: str              # "done" | "failed"
    final_code: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    agent_error: Optional[str] = None
    debug_attempts: int = 0


class AddCellRequest(BaseModel):
    source: str
    stdout: str = ""
    stderr: str = ""
    success: bool = True


class HealthResponse(BaseModel):
    status: str = "ok"
