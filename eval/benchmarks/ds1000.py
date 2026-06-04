"""DS-1000 benchmark evaluator.

Paper : "DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation"
        (Lai et al., ICML 2023) — https://arxiv.org/abs/2211.11501
Data  : huggingface datasets "xlangai/DS-1000"  (split="test", 1000 samples)
Repo  : https://github.com/xlang-ai/DS-1000

HuggingFace schema (actual)
---------------------------
  prompt          : problem description
  reference_code  : correct solution
  code_context    : test harness — defines generate_test_case(), exec_test(),
                    and exec_context (a template string with [insert] placeholder)
  metadata        : {library, problem_id, …}

Evaluation
----------
  1. Extract exec_context from code_context.
  2. Replace [insert] with the generated solution.
  3. Execute via test_execution() defined in code_context.
  4. If no AssertionError → pass@1.

Agent adaptation
----------------
  - Seed a session cell with the exec_context prefix (everything before [insert]).
  - Query: "Write the missing Python code that sets result = <answer>."
  - Run returned code through the official test harness.

Known gaps (see docs/evaluation.md):
  - Agent may re-state context code instead of just filling the gap.
  - Temperature not configurable per-query yet.
"""
from __future__ import annotations

import logging
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from eval.harness import AgentEvaluator, Problem, ProblemResult
from eval.metrics import pass_at_k_bulk

logger = logging.getLogger(__name__)

LIBRARIES = ["Numpy", "Pandas", "Matplotlib", "Scipy", "Sklearn", "Tensorflow", "Pytorch"]


def _load_ds1000(libraries: list[str]) -> list[dict]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise RuntimeError("Install with: uv add datasets")

    ds = load_dataset("xlangai/DS-1000", split="test")
    rows = []
    for row in ds:
        lib = row["metadata"]["library"]
        if lib not in libraries:
            continue
        rows.append({
            "id": f"{lib}_{row['metadata']['problem_id']}",
            "lib": lib,
            "prompt": row["prompt"],
            "reference_code": row["reference_code"],
            "code_context": row["code_context"],
        })
    return rows


def _extract_exec_context(code_context: str) -> str:
    """Pull the exec_context template string out of code_context."""
    m = re.search(r'exec_context\s*=\s*r?"""(.*?)"""', code_context, re.DOTALL)
    if m:
        return m.group(1)
    # fallback: return empty so scoring gracefully fails
    return ""


def _run_test(solution: str, code_context: str, timeout: int = 30) -> tuple[bool, str]:
    """Execute the official DS-1000 test harness with solution inserted."""
    exec_ctx = _extract_exec_context(code_context)
    if not exec_ctx:
        return False, "Could not extract exec_context from code_context"

    script = code_context + f"\n\ntry:\n    test_execution({solution!r})\nexcept Exception as _e:\n    raise _e\n"
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode == 0, proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, f"TimeoutError: exceeded {timeout}s"
    finally:
        Path(path).unlink(missing_ok=True)


def _build_context_prefix(code_context: str) -> str:
    """Return the exec_context template up to (not including) [insert]."""
    exec_ctx = _extract_exec_context(code_context)
    if "[insert]" in exec_ctx:
        return exec_ctx.split("[insert]")[0].strip()
    return exec_ctx.strip()


def _clean_solution(code: str, prefix: str) -> str:
    """Post-process agent output before inserting into exec_context.

    Handles two mechanical failure modes:

    1. Function-body context (prefix ends with ``def f(...):``)
       The model generates a full ``def f(...): ... `` at column 0.
       Inserting it into exec_context creates an empty outer function body
       (IndentationError).  Fix: extract just the indented body lines.

    2. Leading imports already present in exec_context
       The model re-emits ``import pandas as pd`` etc.  These are harmless
       but stripping them makes the diff to [insert] cleaner for the harness.
    """
    # ── 0. Strip markdown code fences ────────────────────────────────────────
    # The model wraps output in ```python ... ``` — strip them before anything else.
    code = re.sub(r"^```(?:python)?\s*\n?", "", code.strip(), flags=re.MULTILINE)
    code = re.sub(r"\n?```\s*$", "", code.strip(), flags=re.MULTILINE)
    code = code.strip()

    lines = code.splitlines()

    # ── 1. Function-body extraction ───────────────────────────────────────────
    # If the prefix ends with a function header (def f(...):), the solution
    # should be only the indented body.  The model often wraps it in another
    # def, producing double-nesting.  Unwrap it.
    stripped_prefix = prefix.rstrip()
    if re.match(r"def \w+\(", stripped_prefix.splitlines()[-1] if stripped_prefix else ""):
        # Find where the model's function body starts (first indented block after
        # a matching def line), and extract just those body lines.
        func_header_pat = re.compile(r"^def \w+\(")
        in_body = False
        body_lines: list[str] = []
        for line in lines:
            if func_header_pat.match(line):
                in_body = True
                body_lines = []   # reset — take the last def's body
                continue
            if in_body:
                # Collect until we hit something at column 0 that isn't a continuation
                if line and not line[0].isspace() and not line.startswith("#"):
                    break
                body_lines.append(line)
        if body_lines:
            # Remove trailing empty lines
            while body_lines and not body_lines[-1].strip():
                body_lines.pop()
            return "\n".join(body_lines)

    # ── 2. Strip data-loading lines ───────────────────────────────────────────
    # The model often generates `df = pd.read_csv(...)` even when told not to.
    # Strip any line that loads data from a file — the test harness provides
    # `df` (and other variables) via `test_input`, so removing these lines
    # leaves only the actual computation, which is what we want to test.
    _DATA_LOAD_PATTERNS = (
        "pd.read_csv", "pd.read_excel", "pd.read_parquet",
        "pd.read_json", "pd.read_table", "pd.read_feather",
        "open(r'", 'open("', "load_csv", "loadtxt",
    )
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if any(p in stripped for p in _DATA_LOAD_PATTERNS):
            continue   # drop data-loading line
        cleaned_lines.append(line)
    lines = cleaned_lines

    # ── 3. Strip leading import-only preamble ────────────────────────────────
    # Remove leading lines that are just imports (already in exec_context).
    first_non_import = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("import ") and not stripped.startswith("from ") and not stripped.startswith("#"):
            first_non_import = i
            break
    return "\n".join(lines[first_non_import:]).strip()


def _infer_variables(prefix: str) -> str:
    """Return a human-readable summary of variables defined by the prefix.

    DS-1000 prefixes are short (< 10 lines) and follow a few patterns:
      df = test_input
      df, List = test_input
      def f(df):
    We parse them to produce a natural-language description for the seed-cell
    stdout so the agent's build_context_node knows what variables are in scope.
    """
    import re as _re
    lines = [l.strip() for l in prefix.splitlines() if l.strip()]
    vars_found = []
    for line in lines:
        # skip imports
        if line.startswith("import ") or line.startswith("from "):
            continue
        # df, List = test_input  OR  df = test_input
        m = _re.match(r"^([\w,\s]+)\s*=\s*test_input", line)
        if m:
            names = [n.strip() for n in m.group(1).split(",") if n.strip()]
            for nm in names:
                if nm == "df":
                    vars_found.append("df (pandas DataFrame)")
                elif nm.lower() in ("list", "lst"):
                    vars_found.append(f"{nm} (list)")
                else:
                    vars_found.append(nm)
        # def f(df):
        m2 = _re.match(r"^def\s+(\w+)\s*\((.*?)\)\s*:", line)
        if m2:
            params = [p.strip() for p in m2.group(2).split(",") if p.strip()]
            vars_found.append(f"function {m2.group(1)}({', '.join(params)}) — implement its body")
    if vars_found:
        return "Variables / context available:\n" + "\n".join(f"  {v}" for v in vars_found)
    return "Setup complete — pandas and numpy imported"


class DS1000Evaluator(AgentEvaluator):
    """Evaluates the agent on DS-1000.

    DS-1000 is a code-completion benchmark.  The generated code is evaluated
    by the official test harness (``_run_test``), NOT by our sandbox, so we
    run with ``execute=False`` by default to avoid spurious NameErrors and
    wasted debug attempts.  A synthetic seed cell is injected so that
    ``build_context_node`` knows what variables are in scope.

    Args:
        libraries:   Subset of DS-1000 libraries to evaluate.  None = all 7.
        k:           k for pass@k reporting.
        n_samples:   Times to call the agent per problem (for pass@k > 1).
    """

    benchmark_name = "ds1000"

    def __init__(
        self,
        libraries: Optional[list[str]] = None,
        k: int = 1,
        n_samples: int = 1,
        debug_rounds: int = 2,
        max_per_library: Optional[int] = None,
        **kwargs,
    ) -> None:
        # DS-1000 scoring is done via the official test harness subprocess, not our sandbox.
        # Force execute=False regardless of what the caller passes — running our sandbox
        # would cause NameErrors (test_input not defined) and burn debug retries.
        # Force completion_mode=True so the agent skips plan_node and goes straight to
        # code generation — plan_node incorrectly plans 'load data first' for every problem.
        kwargs["execute"] = False
        kwargs["completion_mode"] = True
        kwargs["prompt_profile"] = "ds1000"
        super().__init__(**kwargs)
        self.libraries = [l.capitalize() for l in (libraries or LIBRARIES)]
        self.k = k
        self.n_samples = n_samples
        self.debug_rounds = debug_rounds
        self.max_per_library = max_per_library

    def _run_one(self, problem: Problem) -> ProblemResult:
        """Override to add a harness-driven debug loop.

        After the first generation attempt, the code is tested with the official
        DS-1000 test harness.  If it fails, the error is pushed back into the
        agent session as a failed cell, and the agent is re-queried to fix it.
        This repeats up to ``self.debug_rounds`` times.
        """
        code_context = problem.metadata.get("code_context", "")
        prefix = _build_context_prefix(code_context)

        sid = self._create_session()
        t0 = time.perf_counter()
        last_code: Optional[str] = None
        debug_count = 0

        try:
            # ── seed the session ──────────────────────────────────────────────
            for cell in problem.seed_cells:
                self._seed_cell(
                    sid,
                    source=cell["source"],
                    stdout=cell.get("stdout", ""),
                    success=cell.get("success", True),
                )

            # ── first generation attempt ──────────────────────────────────────
            resp = self._query(sid, problem.query)
            last_code = resp.get("final_code") or ""

            # ── harness-driven debug loop ─────────────────────────────────────
            for attempt in range(self.debug_rounds):
                if not last_code:
                    break
                cleaned = _clean_solution(last_code, prefix)
                passed, stderr = _run_test(cleaned, code_context)
                if passed:
                    break
                if attempt >= self.debug_rounds - 1:
                    break   # exhausted debug rounds

                # Feed the real test error back into the session context as a
                # failed cell so the agent can see it and fix the code.
                err_summary = stderr[:600] if stderr else "AssertionError: wrong answer"
                self._seed_cell(
                    sid,
                    source=last_code,
                    stdout="",
                    stderr=err_summary,
                    success=False,
                )
                fix_query = (
                    f"The code above failed the test harness with this error:\n\n"
                    f"{err_summary}\n\n"
                    f"Diagnose the root cause and rewrite the solution from scratch. "
                    f"Store the final answer in `result`."
                )
                # Use debug_mode=True so the agent routes through debug_node
                # (SYSTEM_DEBUG with error-type routing) instead of generate_node.
                fix_resp = self._query(sid, fix_query, debug_mode=True)
                new_code = fix_resp.get("final_code") or ""
                if new_code and new_code != last_code:
                    last_code = new_code
                    debug_count += 1
                else:
                    break   # model gave up or repeated itself

            latency = time.perf_counter() - t0
            return ProblemResult(
                problem_id=problem.id,
                status="unknown",
                score=0.0,
                generated_code=last_code,
                stdout=resp.get("stdout"),
                stderr=resp.get("stderr"),
                agent_error=resp.get("agent_error"),
                debug_attempts=debug_count,
                latency_s=latency,
            )

        except Exception as exc:
            latency = time.perf_counter() - t0
            logger.warning("Problem %s raised %s", problem.id, exc)
            return ProblemResult(
                problem_id=problem.id,
                status="error",
                score=0.0,
                agent_error=str(exc),
                latency_s=latency,
            )
        finally:
            self._delete_session(sid)

    def load_problems(self) -> list[Problem]:
        raw = _load_ds1000(self.libraries)
        problems = []
        for r in raw:
            prefix = _build_context_prefix(r["code_context"])

            # Build a synthetic seed cell so build_context_node sees data in scope.
            # Source is a comment only (safe to replay in sandbox); stdout describes
            # what variables the prefix defines.  Because execute=False, the sandbox
            # never actually runs any code — only the LLM context matters.
            ctx_description = _infer_variables(prefix) if prefix else "pandas and numpy available"
            seed = {
                "source": "# DS-1000 problem — setup variables are provided by the test harness",
                "stdout": ctx_description,
                "success": True,
            }

            # The query embeds the prefix as a code block so the LLM sees exactly
            # what context it is completing.
            if prefix:
                query = (
                    f"CODE COMPLETION TASK — read carefully before writing anything.\n\n"
                    f"The variables below are ALREADY IN MEMORY (provided by a test harness).\n"
                    f"• DO NOT load any CSV, Excel, or other data files.\n"
                    f"• DO NOT restate or re-run the setup code.\n"
                    f"• ONLY write the missing lines that fill the `[insert]` placeholder.\n"
                    f"• Store the final answer in a variable named `result`.\n"
                    f"• No print statements needed.\n\n"
                    f"=== Setup (already executed — variables in scope) ===\n"
                    f"```python\n{prefix}\n# [insert your code here]\n```\n\n"
                    f"=== What to implement ===\n{r['prompt']}"
                )
            else:
                query = (
                    f"CODE COMPLETION TASK.\n"
                    f"Write the Python implementation that solves the task below.\n"
                    f"Store the answer in `result`.  No print statements.\n\n"
                    f"Task:\n{r['prompt']}"
                )

            problems.append(Problem(
                id=r["id"],
                query=query,
                seed_cells=[seed],
                metadata={
                    "lib": r["lib"],
                    "code_context": r["code_context"],
                    "reference_code": r["reference_code"],
                },
            ))
        # Optional per-library cap (e.g. max_per_library=10 → 70 problems across 7 libs)
        if self.max_per_library:
            from collections import defaultdict
            counts: dict[str, int] = defaultdict(int)
            capped = []
            for p in problems:
                lib = p.metadata.get("lib", "unknown")
                if counts[lib] < self.max_per_library:
                    capped.append(p)
                    counts[lib] += 1
            problems = capped

        logger.info("Loaded %d DS-1000 problems (%s)", len(problems), ", ".join(self.libraries))
        return problems

    # Patterns that indicate the agent generated a stub / refused to answer
    _STUB_PATTERNS = [
        "no steps to execute",
        "no data",
        "cannot be performed",
        "dataset is not available",
        "data is not available",
        "unable to proceed",
        "no operations can be performed",
    ]

    def _classify_failure(self, code: str) -> str:
        """Return a short label for why the generated code is likely wrong."""
        lower = code.lower()
        if not code.strip() or all(l.strip().startswith("#") or not l.strip() for l in code.splitlines()):
            return "empty_or_comments"
        if any(p in lower for p in self._STUB_PATTERNS):
            return "stub_no_data"
        if "pd.read_csv" in code or "pd.read_excel" in code or "read_parquet" in code:
            return "data_loading_hallucination"
        if "result" not in code and "return" not in code:
            return "no_result_assigned"
        return "logic_error"

    def score(self, problem: Problem, result: ProblemResult) -> float:
        # Propagate lib to result.metadata so _extra_metrics can aggregate by library.
        lib = problem.metadata.get("lib", "unknown")
        result.metadata["lib"] = lib
        if not result.generated_code:
            result.metadata["score_reason"] = "no_code"
            return 0.0

        code_context = problem.metadata.get("code_context", "")
        prefix = _build_context_prefix(code_context)

        # Clean the solution before testing (strips re-defined function wrappers,
        # leading imports, etc.)
        cleaned = _clean_solution(result.generated_code, prefix)
        result.metadata["cleaned_code"] = cleaned

        passed, stderr = _run_test(cleaned, code_context)
        result.metadata["test_stderr"] = stderr[:800] if stderr else ""

        if passed:
            result.metadata["score_reason"] = "pass"
            return 1.0

        # Also try the raw (uncleaned) code in case cleaning broke something
        if not passed:
            passed_raw, stderr_raw = _run_test(result.generated_code, code_context)
            if passed_raw:
                result.metadata["score_reason"] = "pass_raw"
                result.metadata["test_stderr"] = ""
                return 1.0

        result.metadata["score_reason"] = self._classify_failure(result.generated_code)
        return 0.0

    def _extra_metrics(self, results: list[ProblemResult]) -> dict:
        by_lib: dict[str, list[float]] = {}
        fail_reasons: dict[str, int] = {}
        for r in results:
            lib = r.metadata.get("lib", "unknown")
            by_lib.setdefault(lib, []).append(r.score)
            reason = r.metadata.get("score_reason", "unknown")
            if reason not in ("pass", "pass_raw"):
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
        metrics = {
            f"pass_rate_{lib}": sum(s >= 1.0 for s in scores) / len(scores)
            for lib, scores in by_lib.items()
        }
        # Add failure breakdown counts
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            metrics[f"fail_{reason}"] = count
        return metrics


# ── one-shot baseline (no agent loop) ──────────────────────────────────────────

_ONESHOT_SYSTEM = """\
You are an expert Python data-science programmer.
Complete the CODE COMPLETION task. Output ONLY the Python code that fills the \
`[insert]` placeholder — no explanations, no markdown fences, no prose.

Rules:
- The setup variables already exist in memory; do NOT reload data or restate setup.
- If the setup ends with `def f(...):`, output ONLY the indented body and `return` the answer.
- Otherwise store the final answer in a variable named `result`.
/no_think"""


class DS1000OneShotEvaluator(DS1000Evaluator):
    """One-shot baseline for DS-1000: a single, direct model call with NO agent.

    This deliberately bypasses the entire LangGraph pipeline — no build_context,
    no plan, no execute/sandbox, no self-debug retry loop. It sends one chat
    completion straight to the OpenAI-compatible model server (the same vLLM/mlx
    endpoint the agent uses) and tests the raw output with the official DS-1000
    harness.

    Purpose: the ablation baseline. Comparing this against ``DS1000Evaluator``
    isolates the contribution of the agentic loop (planning + execution-grounded
    self-correction) from the model's raw single-shot ability. The task framing
    (prompt + setup prefix + ``result`` contract) is held identical, so the only
    variable is the agent loop itself.
    """

    benchmark_name = "ds1000_oneshot"

    def __init__(
        self,
        *,
        model_url: Optional[str] = None,
        model_id: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        import os
        self.model_url = (
            model_url or os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
        ).rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_id = model_id or self._probe_model_id()
        logger.info("One-shot baseline → model server %s (model=%s)",
                    self.model_url, self.model_id)

    def _probe_model_id(self) -> str:
        import os
        if os.getenv("AGENT_MODEL"):
            return os.getenv("AGENT_MODEL")
        try:
            r = self._client.get(f"{self.model_url}/models", timeout=10)
            data = r.json().get("data", [])
            return data[0]["id"] if data else "local-model"
        except Exception:
            return "local-model"

    def _run_one(self, problem: Problem) -> ProblemResult:
        t0 = time.perf_counter()
        try:
            messages = [
                {"role": "system", "content": _ONESHOT_SYSTEM},
                {"role": "user", "content": problem.query},
            ]
            resp = self._client.post(
                f"{self.model_url}/chat/completions",
                json={
                    "model": self.model_id,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
            )
            resp.raise_for_status()
            content = (resp.json()["choices"][0]["message"]["content"] or "")
            # Strip <think>…</think> blocks Qwen3 sometimes emits in one-shot mode.
            content = re.sub(r"<think>.*?</think>", "", content,
                             flags=re.DOTALL | re.IGNORECASE).strip()
            latency = time.perf_counter() - t0
            return ProblemResult(
                problem_id=problem.id,
                status="unknown",
                score=0.0,
                generated_code=content,
                debug_attempts=0,
                latency_s=latency,
            )
        except Exception as exc:
            latency = time.perf_counter() - t0
            logger.warning("One-shot problem %s raised %s", problem.id, exc)
            return ProblemResult(
                problem_id=problem.id,
                status="error",
                score=0.0,
                agent_error=str(exc),
                latency_s=latency,
            )
