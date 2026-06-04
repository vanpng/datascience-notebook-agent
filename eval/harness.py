"""Base evaluation harness for the DS Notebook Agent.

Each benchmark evaluator subclasses AgentEvaluator and overrides:
  - load_problems()  → list[Problem]
  - score(problem, result) → float   (0.0 = fail, 1.0 = pass)

The harness handles session lifecycle, API calls, retries, and result logging.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

logger = logging.getLogger(__name__)


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class Problem:
    """A single benchmark problem."""
    id: str
    query: str                          # natural-language question sent to the agent
    seed_cells: list[dict] = field(default_factory=list)
    # Each seed cell: {"source": str, "stdout": str, "success": bool}
    metadata: dict[str, Any] = field(default_factory=dict)
    # Benchmark-specific data (expected answer, test code, data path, …)


@dataclass
class ProblemResult:
    problem_id: str
    status: str                          # "pass" | "fail" | "error" | "timeout"
    score: float                         # 0.0 – 1.0
    generated_code: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    agent_error: Optional[str] = None
    debug_attempts: int = 0
    latency_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    benchmark: str
    total: int
    passed: int
    failed: int
    errors: int
    pass_rate: float
    mean_latency_s: float
    results: list[ProblemResult] = field(default_factory=list)
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Benchmark : {self.benchmark}",
            f"Total     : {self.total}",
            f"Pass      : {self.passed}  ({self.pass_rate:.1%})",
            f"Fail      : {self.failed}",
            f"Error     : {self.errors}",
            f"Latency   : {self.mean_latency_s:.1f}s avg",
        ]
        for k, v in self.extra_metrics.items():
            lines.append(f"{k:<10}: {v}")
        return "\n".join(lines)


# ── base evaluator ────────────────────────────────────────────────────────────

class AgentEvaluator:
    """Connects to a running agent API and runs a benchmark against it.

    Args:
        api_url:    Base URL of the FastAPI agent server (default http://127.0.0.1:8000).
        execute:    Whether to run generated code in the sandbox (True) or return
                    code only (False).  Benchmarks that check execution results
                    must use execute=True.
        timeout:    HTTP timeout for each query (seconds).  Set higher than
                    SANDBOX_TIMEOUT to avoid spurious HTTP timeouts.
        max_problems: Cap the number of problems evaluated (useful for quick checks).
        results_dir: Directory to write per-run JSON results.
    """

    benchmark_name: str = "base"

    def __init__(
        self,
        api_url: str = "http://127.0.0.1:8000",
        execute: bool = True,
        completion_mode: bool = False,
        prompt_profile: str = "default",
        timeout: int = 120,
        max_problems: Optional[int] = None,
        results_dir: str | Path = "eval/results",
        concurrency: int = 1,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.execute = execute
        self.completion_mode = completion_mode
        self.prompt_profile = prompt_profile
        self.timeout = timeout
        self.max_problems = max_problems
        self.results_dir = Path(results_dir)
        self.concurrency = max(1, concurrency)
        self._client = httpx.Client(timeout=self.timeout)

    # ── abstract interface ────────────────────────────────────────────────────

    def load_problems(self) -> list[Problem]:
        """Return the list of problems to evaluate.  Override in subclass."""
        raise NotImplementedError

    def score(self, problem: Problem, result: ProblemResult) -> float:
        """Return a score in [0, 1] for a single problem result.  Override in subclass."""
        raise NotImplementedError

    # ── session helpers ───────────────────────────────────────────────────────

    def _create_session(self) -> str:
        sid = str(uuid.uuid4())
        self._client.post(f"{self.api_url}/sessions", json={"session_id": sid}).raise_for_status()
        return sid

    def _delete_session(self, sid: str) -> None:
        try:
            self._client.delete(f"{self.api_url}/sessions/{sid}")
        except Exception:
            pass

    def _seed_cell(self, sid: str, source: str, stdout: str = "", stderr: str = "", success: bool = True) -> None:
        self._client.post(
            f"{self.api_url}/sessions/{sid}/cells",
            json={"source": source, "stdout": stdout, "stderr": stderr, "success": success},
        ).raise_for_status()

    def _query(self, sid: str, query: str, debug_mode: bool = False) -> dict:
        return (
            self._client.post(
                f"{self.api_url}/sessions/{sid}/query",
                json={
                    "query": query,
                    "execute": self.execute,
                    "completion_mode": self.completion_mode,
                    "debug_mode": debug_mode,
                    "prompt_profile": self.prompt_profile,
                },
            )
            .raise_for_status()
            .json()
        )

    # ── main evaluation loop ──────────────────────────────────────────────────

    def run(self, problems: Optional[list[Problem]] = None) -> BenchmarkResult:
        """Run the full benchmark and return a BenchmarkResult."""
        if problems is None:
            problems = self.load_problems()
        if self.max_problems:
            problems = problems[: self.max_problems]

        results: list[ProblemResult] = []
        total = len(problems)

        def _run_and_score(args):
            i, problem = args
            pr = self._run_one(problem)
            pr.score = self.score(problem, pr)
            pr.status = "pass" if pr.score >= 1.0 else ("error" if pr.agent_error else "fail")
            return i, pr

        def _log_result(i: int, pr: ProblemResult, total: int) -> None:
            logger.info("[%d/%d] %s  → %s  (score=%.2f, %.1fs)",
                        i + 1, total, pr.problem_id, pr.status, pr.score, pr.latency_s)
            if pr.status != "pass":
                code = pr.metadata.get("cleaned_code") or pr.generated_code or ""
                if code:
                    preview = code[:300].replace("\n", "\n    │ ")
                    logger.info("    ┌─ code ─\n    │ %s", preview)
                err = pr.metadata.get("test_stderr") or pr.agent_error or ""
                if err:
                    logger.info("    └─ error: %s", err[:300])

        if self.concurrency <= 1:
            for i, problem in enumerate(problems):
                logger.info("[%d/%d] %s", i + 1, total, problem.id)
                _, pr = _run_and_score((i, problem))
                results.append(pr)
                _log_result(i, pr, total)
        else:
            logger.info("Running with concurrency=%d", self.concurrency)
            indexed_results: list[tuple[int, ProblemResult]] = []
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = {pool.submit(_run_and_score, (i, p)): i
                           for i, p in enumerate(problems)}
                for fut in as_completed(futures):
                    i, pr = fut.result()
                    indexed_results.append((i, pr))
                    _log_result(i, pr, total)
            results = [pr for _, pr in sorted(indexed_results)]

        passed  = sum(1 for r in results if r.status == "pass")
        errors  = sum(1 for r in results if r.status == "error")
        latencies = [r.latency_s for r in results]

        br = BenchmarkResult(
            benchmark=self.benchmark_name,
            total=len(results),
            passed=passed,
            failed=len(results) - passed - errors,
            errors=errors,
            pass_rate=passed / len(results) if results else 0.0,
            mean_latency_s=sum(latencies) / len(latencies) if latencies else 0.0,
            results=results,
            extra_metrics=self._extra_metrics(results),
        )

        self._save(br)
        return br

    def _run_one(self, problem: Problem) -> ProblemResult:
        sid = self._create_session()
        t0 = time.perf_counter()
        try:
            for cell in problem.seed_cells:
                self._seed_cell(
                    sid,
                    source=cell["source"],
                    stdout=cell.get("stdout", ""),
                    success=cell.get("success", True),
                )
            resp = self._query(sid, problem.query)
            latency = time.perf_counter() - t0
            return ProblemResult(
                problem_id=problem.id,
                status="unknown",
                score=0.0,
                generated_code=resp.get("final_code"),
                stdout=resp.get("stdout"),
                stderr=resp.get("stderr"),
                agent_error=resp.get("agent_error"),
                debug_attempts=resp.get("debug_attempts", 0),
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

    def _extra_metrics(self, results: list[ProblemResult]) -> dict[str, Any]:
        """Override to add benchmark-specific aggregate metrics."""
        return {}

    def _save(self, br: BenchmarkResult) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self.results_dir / f"{self.benchmark_name}_{ts}.json"
        data = asdict(br)
        path.write_text(json.dumps(data, indent=2))
        logger.info("Results saved → %s", path)
