"""Pydantic schemas for structured LLM outputs."""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    data_loading   = "data_loading"
    exploration    = "exploration"
    visualization  = "visualization"
    preprocessing  = "preprocessing"
    modeling       = "modeling"
    evaluation     = "evaluation"
    general        = "general"


class SessionContext(BaseModel):
    """Lightweight context snapshot built before the plan node runs."""
    has_data: bool = False
    query_intent: QueryIntent = QueryIntent.general
    available_variables: List[str] = []
    imported_libraries: List[str] = []
    data_sources: List[str] = []        # file paths / seaborn:<name> / url
    suggested_libraries: List[str] = []


class PlanStep(BaseModel):
    step: int
    description: str = Field(description="One-sentence description of what this step does")
    imports_needed: List[str] = Field(default_factory=list)


class AgentPlan(BaseModel):
    """Structured plan produced by the Plan node."""

    reasoning: str = Field(default="", description="Brief analysis of the user query and notebook state")
    steps: List[PlanStep] = Field(description="Ordered list of implementation steps")
    variables_needed: List[str] = Field(
        default_factory=list,
        description="Existing notebook variables this cell will consume",
    )
    variables_produced: List[str] = Field(
        default_factory=list,
        description="New variables or side-effects this cell will produce",
    )


class GeneratedCell(BaseModel):
    """A single executable notebook cell produced by the Generate node."""

    model_config = {"extra": "ignore"}

    reasoning: str = Field(default="", description="Concise explanation of implementation choices")
    code: str = Field(
        description="Complete, self-contained Python code for this notebook cell, "
        "including all import statements at the top. "
        "Must be valid Python 3.10+ with no ellipsis or placeholders."
    )


class DebugPatch(BaseModel):
    """Structured debug response — categorises the error and rewrites the cell."""

    error_category: str = Field(description="Short label for the error type, e.g. fix_logic, fix_syntax, fix_imports")
    root_cause: str = Field(description="One-sentence diagnosis of the error")
    code: str = Field(description="Corrected Python code for the notebook cell")


class DataFrameSchema(BaseModel):
    """Snapshot of a pandas DataFrame's schema inferred from notebook context."""

    name: str
    columns: List[str]
    dtypes: dict[str, str]   # column → dtype string
    shape: Optional[tuple[int, int]] = None
    sample_head: Optional[str] = None   # str(df.head(3))


class NotebookCell(BaseModel):
    """A single executed notebook cell stored in session context."""

    cell_id: int
    source: str
    stdout: str = ""
    stderr: str = ""
    success: bool = True


class ExecutionResult(BaseModel):
    stdout: str = ""
    stderr: str = ""
    success: bool
    timed_out: bool = False
    memory_exceeded: bool = False
