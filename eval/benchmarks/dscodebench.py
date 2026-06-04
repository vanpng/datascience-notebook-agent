"""DSCodeBench benchmark evaluator.

Paper : "DSCodeBench: A Realistic Benchmark for Data Science Code Generation"
        (Ouyang et al., 2025)
Repo  : https://github.com/ShuyinOuyang/DSCodeBench
Data  : benchmark/DSCodeBench.json  (1,000 problems, JSONL, ~5 MB)
        Official eval: benchmark_construction_evaluation/run_test.py

What the benchmark looks like
-----------------------------
1,000 function-implementation problems derived from real GitHub repos, spanning
10 libraries (numpy, pandas, scipy, sklearn, matplotlib, seaborn, tensorflow,
pytorch, keras, lightgbm). Each record::

  problem_id        : "numpy_0"
  library           : "numpy"
  code_problem       : NL task description, INCLUDING the function signature
  ground_truth_code : reference implementation (helper fns first, MAIN fn LAST)
  test_script       : defines test_case_input_generator(n) → list of input tuples

How scoring works (official, ported in ``_dscodebench_harness``)
----------------------------------------------------------------
The candidate solution and the ground truth are each combined with the test
script, RNG-seeded identically (seed 42), and run over the SAME ``n`` generated
inputs. Their output lists are compared element-by-element with a type-aware deep
comparison (ndarray/Tensor/DataFrame/estimator/keras-model/scipy-sparse/…). Plot
problems (ground truth writes ``output.png``) are rendered headless and compared
as RGB ndarrays. Each test case → 1/0. A problem is **solved** iff every test
case passes (strict pass@1, matching the upstream leaderboard convention; GPT-4o
≈ 0.392).

Agent adaptation
----------------
- DSCodeBench is open-ended code generation, not fill-in-the-blank, so we run
  with ``completion_mode=True`` (skip the planner) and a dedicated ``dscodebench``
  prompt profile that asks for a COMPLETE solution (imports + helpers + the exact
  signature, main function defined LAST, no data loading, no __main__ block).
- ``execute=False``: the official harness — not our sandbox — is the source of
  truth, so we never run the code in-session (it would NameError, the inputs come
  from the harness). A harness-driven debug loop feeds real test errors back to
  the agent, mirroring ``DS1000Evaluator``.

Known gaps
----------
- The official ``run_test.py`` imports torch/tf/keras/lightgbm unconditionally;
  our port guards them, so a problem whose own library is missing scores 0.
  Install lightgbm (and scikit-image for plot parity) for full coverage.
- keras problems can underscore under keras ≥ 3.10: ``model.trainable_weights``
  are ``keras.Variable`` (not ``tf.Variable``), which the upstream comparison
  doesn't special-case — it falls through to ``==`` and errors. This is faithful
  to upstream behaviour under newer keras, not a port bug.
- Default ``test_case_number`` is 50 (the upstream default is 200) to keep
  scoring tractable; raise it with ``--test-cases 200`` for paper-faithful runs.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from eval.harness import AgentEvaluator, Problem, ProblemResult

logger = logging.getLogger(__name__)

LIBRARIES = [
    "numpy", "pandas", "scipy", "sklearn", "matplotlib",
    "seaborn", "tensorflow", "pytorch", "keras", "lightgbm",
]

# Candidate locations for the benchmark JSON when --data-file is not given.
_DEFAULT_DATA_PATHS = [
    "data/DSCodeBench.json",
    ".cache/repos/DSCodeBench/benchmark/DSCodeBench.json",
    "DSCodeBench.json",
]


def _resolve_data_file(data_file: Optional[str | Path]) -> Path:
    if data_file:
        p = Path(data_file)
        if p.is_dir():
            p = p / "DSCodeBench.json"
        if not p.exists():
            raise FileNotFoundError(f"DSCodeBench data file not found: {p}")
        return p
    for cand in _DEFAULT_DATA_PATHS:
        p = Path(cand)
        if p.exists():
            return p
    raise FileNotFoundError(
        "DSCodeBench.json not found. Pass --data-file PATH, or download it:\n"
        "  git clone --depth 1 https://github.com/ShuyinOuyang/DSCodeBench\n"
        "  cp DSCodeBench/benchmark/DSCodeBench.json data/"
    )


def _load_dscodebench(data_file: Path, libraries: list[str]) -> list[dict]:
    rows = []
    with data_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("library") not in libraries:
                continue
            rows.append(r)
    return rows


def _strip_fences(code: str) -> str:
    """Strip <think> blocks and markdown code fences from raw model output."""
    code = re.sub(r"<think>.*?</think>", "", code, flags=re.DOTALL | re.IGNORECASE)
    code = code.strip()
    # If fenced, keep only the first fenced block (matches the official extract_code).
    m = re.search(r"```(?:[\w+-]*)\n(.*?)```", code, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Otherwise strip any stray leading/trailing fence markers.
    code = re.sub(r"^```(?:[\w+-]*)\s*\n?", "", code, flags=re.MULTILINE)
    code = re.sub(r"\n?```\s*$", "", code, flags=re.MULTILINE)
    return code.strip()


def _run_harness(
    ground_truth_code: str,
    solution_code: str,
    test_script: str,
    test_case_number: int,
    timeout: int,
) -> tuple[list[int], str]:
    """Score one solution via the official harness in an isolated subprocess.

    Returns (evaluation_result_list, error_string). The result list holds 1/0 per
    test case; an empty list with a non-empty error means the harness crashed
    (e.g. the solution failed to import/run on every input).
    """
    task = {
        "ground_truth_code": ground_truth_code,
        "solution_code": solution_code,
        "test_script": test_script,
        "test_case_number": test_case_number,
        "random_seed": 42,
    }
    task_f = res_f = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(task, f)
            task_f = f.name
        res_f = task_f + ".out"
        env = {**os.environ}
        # Make `eval.benchmarks._dscodebench_harness` importable from the repo root.
        root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        proc = subprocess.run(
            [sys.executable, "-m", "eval.benchmarks._dscodebench_harness", task_f, res_f],
            capture_output=True, text=True, timeout=timeout, env=env, cwd=root,
        )
        if res_f and Path(res_f).exists():
            out = json.loads(Path(res_f).read_text())
            return out.get("evaluation_result", []), out.get("error", "")
        return [], (proc.stderr or "harness produced no output")[-600:]
    except subprocess.TimeoutExpired:
        return [], f"TimeoutError: harness exceeded {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"
    finally:
        for p in (task_f, res_f):
            if p:
                Path(p).unlink(missing_ok=True)


def _build_query(code_problem: str) -> str:
    return (
        "CODE GENERATION TASK — implement the function(s) described below.\n\n"
        "Write a COMPLETE, self-contained Python solution:\n"
        "• Include every import the solution needs.\n"
        "• Define the EXACT function signature(s) given in the description.\n"
        "• Put any helper functions FIRST and the MAIN function (the one the task "
        "asks for) LAST — a test harness calls the last-defined function.\n"
        "• Do NOT read or load any data files: all inputs are passed as function "
        "arguments by the test harness.\n"
        "• Do NOT add a `__main__` block, example usage, prints, or your own tests.\n"
        "• For a plotting task, save the figure to the output path named in the "
        "signature (usually `output.png`) instead of calling plt.show().\n\n"
        "=== Problem ===\n"
        f"{code_problem}"
    )


class DSCodeBenchEvaluator(AgentEvaluator):
    """Evaluates the agent on DSCodeBench via the official test harness.

    Args:
        data_file:        Path to DSCodeBench.json (or a dir containing it).
                          Auto-resolved from common locations when omitted.
        libraries:        Subset of the 10 libraries to evaluate (None = all).
        test_case_number: Inputs generated per problem (upstream default 200;
                          we default 50 for tractability).
        debug_rounds:     Harness-driven self-debug attempts after the first try.
        harness_timeout:  Per-problem wall-clock cap for the scoring subprocess.
        max_per_library:  Optional cap of N problems per library.
    """

    benchmark_name = "dscodebench"

    def __init__(
        self,
        data_file: Optional[str | Path] = None,
        libraries: Optional[list[str]] = None,
        test_case_number: int = 50,
        debug_rounds: int = 2,
        harness_timeout: int = 600,
        max_per_library: Optional[int] = None,
        **kwargs,
    ) -> None:
        # Scoring is done by the official harness subprocess, not our sandbox:
        #   execute=False        → don't run code in-session (inputs come from harness)
        #   completion_mode=True → skip the planner (it would plan "load data")
        #   prompt_profile       → DSCodeBench code-generation prompts
        kwargs["execute"] = False
        kwargs["completion_mode"] = True
        kwargs["prompt_profile"] = "dscodebench"
        super().__init__(**kwargs)
        self.data_file = _resolve_data_file(data_file)
        self.libraries = [l.lower() for l in (libraries or LIBRARIES)]
        self.test_case_number = test_case_number
        # Quick pre-check during the debug loop uses fewer cases for speed.
        self.debug_check_cases = max(5, min(test_case_number, 10))
        self.debug_rounds = debug_rounds
        self.harness_timeout = harness_timeout
        self.max_per_library = max_per_library

    def load_problems(self) -> list[Problem]:
        raw = _load_dscodebench(self.data_file, self.libraries)
        problems = []
        for r in raw:
            problems.append(Problem(
                id=r["problem_id"],
                query=_build_query(r["code_problem"]),
                seed_cells=[],
                metadata={
                    "lib": r["library"],
                    "ground_truth_code": r["ground_truth_code"],
                    "test_script": r["test_script"],
                },
            ))
        if self.max_per_library:
            counts: dict[str, int] = defaultdict(int)
            capped = []
            for p in problems:
                lib = p.metadata.get("lib", "unknown")
                if counts[lib] < self.max_per_library:
                    capped.append(p)
                    counts[lib] += 1
            problems = capped
        logger.info("Loaded %d DSCodeBench problems (%s) from %s",
                    len(problems), ", ".join(self.libraries), self.data_file)
        return problems

    def _run_one(self, problem: Problem) -> ProblemResult:
        """Generate a solution, then run a harness-driven self-debug loop."""
        gt = problem.metadata.get("ground_truth_code", "")
        ts = problem.metadata.get("test_script", "")

        sid = self._create_session()
        t0 = time.perf_counter()
        last_code: Optional[str] = None
        debug_count = 0
        resp: dict = {}

        try:
            resp = self._query(sid, problem.query)
            last_code = resp.get("final_code") or ""

            for attempt in range(self.debug_rounds):
                if not last_code:
                    break
                cleaned = _strip_fences(last_code)
                results, err = _run_harness(
                    gt, cleaned, ts, self.debug_check_cases, self.harness_timeout)
                passed = bool(results) and all(x == 1 for x in results)
                if passed or attempt >= self.debug_rounds - 1:
                    break

                if err:
                    err_summary = err[:600]
                elif results:
                    n_ok = sum(results)
                    err_summary = (
                        f"The code ran but produced the WRONG output on "
                        f"{len(results) - n_ok}/{len(results)} test cases "
                        f"(outputs differ from the expected values)."
                    )
                else:
                    err_summary = "The code failed to run on the test inputs."

                self._seed_cell(sid, source=cleaned, stdout="",
                                stderr=err_summary, success=False)
                fix_query = (
                    f"Your solution failed the test harness:\n\n{err_summary}\n\n"
                    f"Diagnose the root cause and rewrite the COMPLETE solution. "
                    f"Keep the exact function signature, define the main function "
                    f"last, and do not load data or add a __main__ block."
                )
                fix_resp = self._query(sid, fix_query, debug_mode=True)
                new_code = fix_resp.get("final_code") or ""
                if new_code and new_code != last_code:
                    last_code = new_code
                    debug_count += 1
                else:
                    break

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
        except Exception as exc:  # noqa: BLE001
            latency = time.perf_counter() - t0
            logger.warning("Problem %s raised %s", problem.id, exc)
            return ProblemResult(
                problem_id=problem.id, status="error", score=0.0,
                agent_error=str(exc), latency_s=latency,
            )
        finally:
            self._delete_session(sid)

    def score(self, problem: Problem, result: ProblemResult) -> float:
        lib = problem.metadata.get("lib", "unknown")
        result.metadata["lib"] = lib
        if not result.generated_code:
            result.metadata["score_reason"] = "no_code"
            return 0.0

        gt = problem.metadata.get("ground_truth_code", "")
        ts = problem.metadata.get("test_script", "")
        cleaned = _strip_fences(result.generated_code)
        result.metadata["cleaned_code"] = cleaned

        results, err = _run_harness(
            gt, cleaned, ts, self.test_case_number, self.harness_timeout)
        n = len(results)
        n_ok = sum(results)
        result.metadata["n_test_cases"] = n
        result.metadata["n_passed"] = n_ok
        result.metadata["test_case_pass_rate"] = (n_ok / n) if n else 0.0
        result.metadata["test_stderr"] = err[:800] if err else ""

        if n > 0 and n_ok == n:
            result.metadata["score_reason"] = "pass"
            return 1.0
        if n == 0:
            result.metadata["score_reason"] = "harness_error" if err else "no_output"
        else:
            result.metadata["score_reason"] = "wrong_output"
        return 0.0

    def _extra_metrics(self, results: list[ProblemResult]) -> dict:
        by_lib: dict[str, list[float]] = {}
        fail_reasons: dict[str, int] = {}
        partial: list[float] = []
        for r in results:
            lib = r.metadata.get("lib", "unknown")
            by_lib.setdefault(lib, []).append(r.score)
            partial.append(r.metadata.get("test_case_pass_rate", 0.0))
            reason = r.metadata.get("score_reason", "unknown")
            if reason != "pass":
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
        metrics: dict = {
            f"pass_rate_{lib}": sum(s >= 1.0 for s in scores) / len(scores)
            for lib, scores in sorted(by_lib.items())
        }
        if partial:
            metrics["mean_test_case_pass_rate"] = sum(partial) / len(partial)
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            metrics[f"fail_{reason}"] = count
        return metrics


# ── one-shot baseline (no agent loop) ──────────────────────────────────────────

_ONESHOT_SYSTEM = """\
You are an expert Python data-science programmer.
Generate a COMPLETE, self-contained solution for the problem. Output ONLY Python \
code — no explanations, no prose.

Rules:
- Include every import the solution needs.
- Define the EXACT function signature(s) described.
- Put helper functions first and the MAIN function (the one asked for) LAST.
- Do NOT read or load data files — inputs are passed as function arguments.
- Do NOT add a __main__ block, example usage, prints, or tests.
- For a plotting task, save the figure to the output path in the signature.
/no_think"""


class DSCodeBenchOneShotEvaluator(DSCodeBenchEvaluator):
    """One-shot baseline for DSCodeBench: a single direct model call, NO agent.

    Mirrors ``DS1000OneShotEvaluator`` — it bypasses the LangGraph pipeline
    (no plan, no build_context, no self-debug loop) and sends one chat completion
    straight to the OpenAI-compatible model server, scoring the raw output with
    the official harness. The ablation baseline that isolates the agent loop's
    contribution; the task framing is held identical.
    """

    benchmark_name = "dscodebench_oneshot"

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
        self.model_url = (
            model_url or os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
        ).rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_id = model_id or self._probe_model_id()
        logger.info("One-shot baseline → model server %s (model=%s)",
                    self.model_url, self.model_id)

    def _probe_model_id(self) -> str:
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
            content = resp.json()["choices"][0]["message"]["content"] or ""
            latency = time.perf_counter() - t0
            return ProblemResult(
                problem_id=problem.id, status="unknown", score=0.0,
                generated_code=content, debug_attempts=0, latency_s=latency,
            )
        except Exception as exc:  # noqa: BLE001
            latency = time.perf_counter() - t0
            logger.warning("One-shot problem %s raised %s", problem.id, exc)
            return ProblemResult(
                problem_id=problem.id, status="error", score=0.0,
                agent_error=str(exc), latency_s=latency,
            )
