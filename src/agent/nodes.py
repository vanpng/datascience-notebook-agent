"""LangGraph node functions: plan, generate, execute, debug, extract_schemas."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agent import prompts as P
from src.agent.sandbox import execute_code
from src.agent.schemas import (
    AgentPlan,
    DataFrameSchema,
    DebugPatch,
    ExecutionResult,
    GeneratedCell,
    NotebookCell,
    QueryIntent,
    SessionContext,
)
from src.agent.state import AgentState

logger = logging.getLogger(__name__)


# ── LLM factory ──────────────────────────────────────────────────────────────

def _make_llm(temperature: float = 0.0) -> ChatOpenAI:
    import os, sys
    # Priority: AGENT_MODEL > MLX_MODEL (macOS) / VLLM_MODEL (Linux) > built-in default
    if os.getenv("AGENT_MODEL"):
        model = os.getenv("AGENT_MODEL")
    elif sys.platform == "darwin":
        model = os.getenv("MLX_MODEL", "mlx-community/Qwen3-4B-8bit")
    else:
        model = os.getenv("VLLM_MODEL", "Qwen/Qwen3-4B")
    return ChatOpenAI(
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1"),
        model=model,
        api_key="not-needed",
        temperature=temperature,
        max_tokens=4096,
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Strip markdown fences and parse the first JSON object in *text*."""
    # remove ```json … ``` fences
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "")
    # find first { … } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM output: {text[:200]}")
    return json.loads(match.group())


def _notebook_context(state: AgentState) -> str:
    return P.build_notebook_context(
        state.get("notebook_cells", []),
        state.get("dataframe_schemas", []),
        state.get("session_context"),
    )


# ── Node: build_context ───────────────────────────────────────────────────────

async def build_context_node(state: AgentState) -> dict:
    """
    Scan notebook history for deterministic facts (imports, variables, data sources,
    df schemas), then ask the LLM to infer query intent and suggested libraries.
    """
    cells: list[NotebookCell] = state.get("notebook_cells", [])
    query: str = state.get("user_query", "")

    schemas: list[DataFrameSchema] = list(state.get("dataframe_schemas", []))
    schema_names = {s.name for s in schemas}

    available_variables: list[str] = []
    imported_libraries: list[str] = []
    data_sources: list[str] = []

    for cell in cells:
        if not cell.success:
            continue
        src = cell.source

        for m in re.finditer(r"^(?:import\s+(\S+)|from\s+(\S+)\s+import)", src, re.MULTILINE):
            lib = (m.group(1) or m.group(2) or "").split(".")[0]
            if lib and lib not in imported_libraries:
                imported_libraries.append(lib)

        _skip = {"True", "False", "None", "self"}
        for m in re.finditer(r"^([A-Za-z_]\w*)\s*=", src, re.MULTILINE):
            var = m.group(1)
            if var not in _skip and var not in available_variables:
                available_variables.append(var)

        for m in re.finditer(
            r"""['"]([\w./\\-]+\.(?:csv|xlsx|xls|json|parquet|tsv|feather))['""]""", src
        ):
            path = m.group(1)
            if path not in data_sources:
                data_sources.append(path)

        for m in re.finditer(r"""sns\.load_dataset\(['"]([\w-]+)['"]\)""", src):
            entry = f"seaborn:{m.group(1)}"
            if entry not in data_sources:
                data_sources.append(entry)

        for m in re.finditer(r"(\w+)\s*=\s*(?:pd\.(read_\w+|DataFrame)|sns\.load_dataset)\(", src):
            varname = m.group(1)
            if varname in schema_names:
                continue
            schemas.append(DataFrameSchema(name=varname, columns=[], dtypes={}))
            schema_names.add(varname)

    has_data = bool(schemas) or bool(data_sources)

    # ── LLM inference for intent + suggested libraries ────────────────────────
    llm = _make_llm(temperature=0.0)
    summary_parts = [f'User query: "{query}"']
    if available_variables:
        summary_parts.append(f"Variables in scope: {', '.join(available_variables)}")
    if data_sources:
        summary_parts.append(f"Data sources: {', '.join(data_sources)}")
    if imported_libraries:
        summary_parts.append(f"Already imported: {', '.join(imported_libraries)}")
    if not has_data:
        summary_parts.append("No dataset is currently loaded.")

    llm_response = await llm.ainvoke([
        SystemMessage(content=P.SYSTEM_BUILD_CONTEXT),
        HumanMessage(content="\n".join(summary_parts)),
    ])

    intent = QueryIntent.general
    suggested_libraries: list[str] = []
    try:
        data = _extract_json(llm_response.content)
        intent = QueryIntent(data.get("query_intent", "general"))
        suggested_libraries = data.get("suggested_libraries", [])
    except Exception as exc:
        logger.warning("build_context LLM parse failed: %s", exc)

    ctx = SessionContext(
        has_data=has_data,
        query_intent=intent,
        available_variables=available_variables,
        imported_libraries=imported_libraries,
        data_sources=data_sources,
        suggested_libraries=suggested_libraries,
    )

    return {"dataframe_schemas": schemas, "session_context": ctx, "status": "planning"}


# ── Node: plan ────────────────────────────────────────────────────────────────

async def plan_node(state: AgentState) -> dict:
    llm = _make_llm(temperature=0.0)
    ctx = _notebook_context(state)
    profile = P.get_profile(state.get("prompt_profile"))

    messages = [
        SystemMessage(content=profile.system_plan),
        *[HumanMessage(content=m["content"]) if m["role"] == "user"
          else SystemMessage(content=m["content"])
          for m in profile.few_shot_plan],
        HumanMessage(content=f"Notebook context:\n{ctx}\n\nUser query: {state['user_query']}"),
    ]

    response = await llm.ainvoke(messages)
    raw = response.content

    try:
        data = _extract_json(raw)
        plan = AgentPlan(**data)
    except Exception as exc:
        logger.warning("Plan parse failed: %s — raw: %s", exc, raw[:300])
        plan = AgentPlan(
            reasoning="(parse error — proceeding with minimal plan)",
            steps=[],
        )

    return {
        "plan": plan,
        "status": "generating",
        "messages": [HumanMessage(content=state["user_query"])],
    }


# ── Node: generate ────────────────────────────────────────────────────────────

async def generate_node(state: AgentState) -> dict:
    llm = _make_llm(temperature=0)
    ctx = _notebook_context(state)
    profile = P.get_profile(state.get("prompt_profile"))

    if state.get("plan"):
        # Normal mode: follow the plan produced by plan_node
        plan_json = state["plan"].model_dump_json()
        human_content = f"Plan:\n{plan_json}\n\nNotebook context:\n{ctx}"
    else:
        # Completion mode (plan_node skipped): implement the user query directly.
        # The query already contains full task instructions (problem description,
        # setup code, and explicit constraints).
        human_content = (
            f"Task (implement directly — no plan phase):\n"
            f"{state.get('user_query', '')}\n\n"
            f"Notebook context:\n{ctx}"
        )

    messages = [
        SystemMessage(content=profile.system_generate),
        *[HumanMessage(content=m["content"]) if m["role"] == "user"
          else SystemMessage(content=m["content"])
          for m in profile.few_shot_generate],
        HumanMessage(content=human_content),
    ]

    response = await llm.ainvoke(messages)
    raw = response.content

    try:
        data = _extract_json(raw)
        cell = GeneratedCell(**data)
    except Exception as exc:
        logger.warning("Generate parse failed: %s", exc)
        # Fallback 1: extract the first ```python ... ``` block
        code_candidate = ""
        fence_match = re.search(r"```(?:python)?\s*\n?(.*?)```", raw, re.DOTALL)
        if fence_match:
            code_candidate = fence_match.group(1).strip()
        if not code_candidate:
            # Fallback 2: strip any JSON-looking text and take what remains.
            # Also strip <think>…</think> blocks that Qwen3 sometimes emits.
            text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
            code_candidate = re.sub(r"\{.*\}", "", text, flags=re.DOTALL).strip()
        cell = GeneratedCell(reasoning="(parse error)", code=code_candidate)

    return {"generated_cell": cell, "status": "executing"}


# ── Node: execute ─────────────────────────────────────────────────────────────

async def execute_node(state: AgentState) -> dict:
    cell = state.get("generated_cell")
    if not cell:
        return {
            "execution_result": ExecutionResult(
                stderr="No cell to execute", success=False
            ),
            "status": "debugging",
        }

    prior_sources = [c.source for c in state.get("notebook_cells", []) if c.success]
    result = await execute_code(cell.code, prior_sources=prior_sources)

    updates: dict = {"execution_result": result}

    if result.success:
        # Append to notebook history
        cells: list[NotebookCell] = list(state.get("notebook_cells", []))
        new_id = (cells[-1].cell_id + 1) if cells else 0
        cells.append(
            NotebookCell(
                cell_id=new_id,
                source=cell.code,
                stdout=result.stdout,
                stderr=result.stderr,
                success=True,
            )
        )
        updates["notebook_cells"] = cells
        updates["final_code"] = cell.code
        updates["status"] = "done"
    else:
        updates["status"] = "debugging"

    return updates


# ── Node: debug ───────────────────────────────────────────────────────────────

async def debug_node(state: AgentState) -> dict:
    llm = _make_llm(temperature=0.0)
    cell = state.get("generated_cell")
    result = state.get("execution_result")
    ctx = _notebook_context(state)
    profile = P.get_profile(state.get("prompt_profile"))

    # In debug_mode (routed here from build_context without a prior generate),
    # pull failed code and error from the last failed notebook cell.
    if cell is None:
        cells: list[NotebookCell] = state.get("notebook_cells", [])
        failed_cells = [c for c in cells if not c.success and (c.stderr or c.source)]
        if failed_cells:
            last_failed = failed_cells[-1]
            from src.agent.schemas import ExecutionResult as ER
            cell = type("_FakeCell", (), {"code": last_failed.source})()
            result = ER(stderr=last_failed.stderr or "", success=False)

    failed_code = cell.code if cell else "<no code>"
    stderr = result.stderr if result else "<no stderr>"

    messages = [
        SystemMessage(content=profile.system_debug),
        *[HumanMessage(content=m["content"]) if m["role"] == "user"
          else SystemMessage(content=m["content"])
          for m in profile.few_shot_debug],
        HumanMessage(
            content=(
                f"Failed code:\n{failed_code}\n\n"
                f"Stderr:\n{stderr}\n\n"
                f"Notebook context:\n{ctx}"
            )
        ),
    ]

    response = await llm.ainvoke(messages)
    raw = response.content

    try:
        data = _extract_json(raw)
        patch = DebugPatch(**data)
    except Exception as exc:
        logger.warning("Debug parse failed: %s", exc)
        patch = DebugPatch(
            error_category="fix_logic",
            root_cause="parse error",
            code=failed_code,
        )

    # Overwrite generated_cell.code with the patched version
    patched_cell = GeneratedCell(
        reasoning=f"[debug attempt {state.get('debug_attempts', 0) + 1}] {patch.root_cause}",
        code=patch.code,
    )

    return {
        "debug_patch": patch,
        "generated_cell": patched_cell,
        "debug_attempts": state.get("debug_attempts", 0) + 1,
        "status": "executing",
    }
