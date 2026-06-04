"""DABStep benchmark evaluator.

Paper : "DABStep: Data Analysis Benchmark for Step-Wise Evaluation"
        (Cabannes et al., 2025) — https://arxiv.org/abs/2506.23719
HF    : https://huggingface.co/datasets/adyen/DABStep
Blog  : https://huggingface.co/blog/dabstep
Leaderboard: https://huggingface.co/spaces/adyen/DABstep

HuggingFace schema
------------------
  Splits  : "default" (450 tasks, answers withheld — leaderboard submission)
             "dev"    (10 tasks, answers provided — local eval)
  Columns : task_id, question, answer, guidelines, level

  Context files (data/context/ in the HF repo):
    payments.csv             — 138k payment records (21 columns)
    payments-readme.md       — column descriptions for payments.csv
    fees.json                — fee schedule per card scheme / ACI / account type
    merchant_data.json       — merchant metadata
    acquirer_countries.csv   — acquirer country ↔ region mapping
    merchant_category_codes.csv — MCC descriptions
    manual.md                — 337-line guide: fee calc, ACI codes, account types

  Download all via:
    huggingface_hub.snapshot_download("adyen/DABstep", repo_type="dataset")

Evaluation approach (matches official quickstart)
-------------------------------------------------
  1. Seed cell pre-loads CSV/JSON data into named variables.
  2. Markdown docs (manual.md, payments-readme.md) are embedded in the query
     as text so the agent has the domain knowledge without doing extra I/O.
  3. Query tells the agent exactly which variables are available and to print().
  4. completion_mode=True skips plan_node (which generates "explore data" plans
     instead of computation code for these analytical questions).
  5. execute=True so generated code runs; debug loop fires on errors.

Leaderboard submission
----------------------
  Use `--submit` flag in run_eval.py to produce runs/{RUN_ID}.jsonl:
    { "task_id": "5", "agent_answer": "NL", "reasoning_trace": "..." }
  Then upload to https://huggingface.co/spaces/adyen/DABstep

Known gaps:
  - Without data_dir all scores are 0.
  - Hard problems require multi-table joins with the fee schedule.
  - Qwen3-4B may compute wrong values on complex aggregations.
"""
from __future__ import annotations

import json as _json_mod
import logging
import time
from pathlib import Path
from typing import Optional

from eval.harness import AgentEvaluator, Problem, ProblemResult
from eval.metrics import numeric_match, string_match

logger = logging.getLogger(__name__)

# All 7 context files from the official quickstart (order matches the notebook)
_CONTEXT_FILES = [
    "acquirer_countries.csv",
    "payments-readme.md",
    "payments.csv",
    "merchant_category_codes.csv",
    "fees.json",
    "merchant_data.json",
    "manual.md",
]

# Files we pre-load as Python variables in the seed cell
_LOADABLE_FILES = {
    "payments.csv",
    "fees.json",
    "merchant_data.json",
    "acquirer_countries.csv",
    "merchant_category_codes.csv",
}

# Quickstart prompt template (verbatim from the official notebook)
_PROMPT_TEMPLATE = """\
You are an expert data analyst and you will answer factoid questions by \
loading and referencing the files/documents listed below.
You have these files available:
{context_files}
Don't forget to reference any documentation in the data dir before answering a question.

Here is the question you need to answer:
{question}

Here are the guidelines you must follow when answering the question above:
{guidelines}

--- DATA ALREADY LOADED ---
The following Python variables are already defined (do NOT reload them):
{vars_block}

Write Python code that computes the answer using the variables above.
The LAST line of your code MUST be a print() call that outputs ONLY the final answer.
Example: print(result)   or   print(42.5)   or   print("Visa")
No intermediate prints, no labels, no explanatory text."""


def _load_dabstep(split: str = "dev") -> list[dict]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise RuntimeError("Install with: uv add datasets")
    hf_split = "dev" if split in ("dev", "test") else "default"
    ds = load_dataset("adyen/DABstep", name="tasks", split=hf_split)
    return [dict(row) for row in ds]


def _schema_summary(data_dir: Path) -> str:
    """Return a compact schema description for pre-loaded variables."""
    import csv as _csv
    lines: list[str] = []

    def _csv_cols(p: Path) -> list[str]:
        try:
            with p.open(newline="") as f:
                return next(_csv.reader(f))
        except Exception:
            return []

    payments = data_dir / "payments.csv"
    if payments.exists():
        cols = _csv_cols(payments)
        size_kb = payments.stat().st_size // 1024
        lines.append(f"payments — DataFrame ({size_kb}KB, {len(cols)} cols): {', '.join(cols)}")

    for fname, vname in [
        ("acquirer_countries.csv", "acquirer_countries"),
        ("merchant_category_codes.csv", "merchant_category_codes"),
    ]:
        p = data_dir / fname
        if p.exists():
            cols = _csv_cols(p)
            lines.append(f"{vname} — DataFrame: {', '.join(cols)}")

    fees_json = data_dir / "fees.json"
    if fees_json.exists():
        try:
            data = _json_mod.loads(fees_json.read_text())
            if isinstance(data, list) and data and isinstance(data[0], dict):
                keys = list(data[0].keys())
                lines.append(f"fees — list[dict] ({len(data)} entries): {', '.join(keys)}")
        except Exception:
            lines.append("fees — JSON list loaded")

    merchant_json = data_dir / "merchant_data.json"
    if merchant_json.exists():
        try:
            data = _json_mod.loads(merchant_json.read_text())
            if isinstance(data, list) and data and isinstance(data[0], dict):
                keys = list(data[0].keys())
                lines.append(f"merchant_data — list[dict] ({len(data)} entries): {', '.join(keys)}")
        except Exception:
            lines.append("merchant_data — JSON list loaded")

    return "\n".join(f"  {l}" for l in lines)


def _context_files_block(data_dir: Path) -> str:
    """Return formatted list of absolute paths for all 7 context files."""
    lines = []
    for fname in _CONTEXT_FILES:
        p = data_dir / fname
        if p.exists():
            lines.append(f"  {p}")
    return "\n".join(lines) if lines else "  (no files found)"


def _build_seed_cell(data_dir: Optional[Path]) -> Optional[dict]:
    """Pre-load CSV/JSON data files. Markdown docs are embedded in the query instead."""
    if data_dir is None:
        return None

    lines = ["import pandas as pd", "import numpy as np", "import json"]

    payments = data_dir / "payments.csv"
    if payments.exists():
        lines.append(f"payments = pd.read_csv(r'{payments}')")

    fees_json = data_dir / "fees.json"
    if fees_json.exists():
        lines.append(f"with open(r'{fees_json}') as _f:")
        lines.append(f"    fees = json.load(_f)")

    merchant_json = data_dir / "merchant_data.json"
    if merchant_json.exists():
        lines.append(f"with open(r'{merchant_json}') as _f:")
        lines.append(f"    merchant_data = json.load(_f)")

    for p in sorted(data_dir.glob("*.csv")):
        if p.name in ("payments.csv",):
            continue
        vname = p.stem.replace("-", "_").replace(" ", "_")
        lines.append(f"{vname} = pd.read_csv(r'{p}')")

    schema = _schema_summary(data_dir)
    stdout = "Data loaded:\n" + schema
    return {"source": "\n".join(lines), "stdout": stdout, "success": True}


def _build_query(
    question: str,
    guidelines: str,
    data_dir: Optional[Path],
) -> str:
    """Build the agent query matching the official quickstart prompt structure."""
    if data_dir is None:
        context_files_str = "  (no data_dir configured)"
        vars_block = "  (no data loaded)"
    else:
        context_files_str = _context_files_block(data_dir)
        vars_block = _schema_summary(data_dir)

    return _PROMPT_TEMPLATE.format(
        context_files=context_files_str,
        question=question,
        guidelines=guidelines,
        vars_block=vars_block,
    )


class DABStepEvaluator(AgentEvaluator):
    """Evaluates the agent on DABStep and can produce leaderboard submission files.

    Local eval  : split="dev"  (10 tasks, ground truth available)
    Submission  : split="default" + run_submission() → runs/{RUN_ID}.jsonl

    Args:
        split:       "dev" for local eval; "default" for leaderboard submission.
        difficulty:  Filter to "easy", "hard", or None = all.
        data_dir:    Path to local directory with all DABStep context files.
        numeric_tol: Relative tolerance for numeric answer comparison.
    """

    benchmark_name = "dabstep"

    def __init__(
        self,
        split: str = "dev",
        difficulty: Optional[str] = None,
        data_dir: Optional[str | Path] = None,
        numeric_tol: float = 0.01,
        **kwargs,
    ) -> None:
        # completion_mode=True: skip plan_node (it generates "explore data" plans).
        # execute=True: run the code so the debug loop can fire on errors.
        kwargs.setdefault("execute", True)
        kwargs.setdefault("completion_mode", True)
        kwargs.setdefault("prompt_profile", "dabstep")
        super().__init__(**kwargs)
        self.split = split
        self.difficulty = difficulty
        self.data_dir = Path(data_dir) if data_dir else None
        self.numeric_tol = numeric_tol

        if split == "default":
            logger.warning(
                "DABStep 'default' split has no ground-truth answers. "
                "Use run_submission() to produce a leaderboard JSONL file."
            )

    def load_problems(self) -> list[Problem]:
        raw = _load_dabstep(self.split)

        if self.difficulty:
            raw = [t for t in raw if t.get("level") == self.difficulty]

        if self.data_dir is None:
            logger.warning(
                "No data_dir provided — scores will be 0.\n"
                "Download: huggingface_hub.snapshot_download('adyen/DABstep', repo_type='dataset')"
            )

        problems = []
        for task in raw:
            seed = _build_seed_cell(self.data_dir)
            query = _build_query(
                question=task["question"],
                guidelines=task.get("guidelines", ""),
                data_dir=self.data_dir,
            )
            problems.append(Problem(
                id=f"dabstep_{task['task_id']}",
                query=query,
                seed_cells=[seed] if seed else [],
                metadata={
                    "task_id": str(task["task_id"]),
                    "answer": task.get("answer", ""),
                    "level": task.get("level", ""),
                    "guidelines": task.get("guidelines", ""),
                    "has_data": self.data_dir is not None,
                },
            ))

        logger.info(
            "Loaded %d DABStep problems (split=%s, difficulty=%s, data_dir=%s)",
            len(problems), self.split, self.difficulty or "all",
            str(self.data_dir) if self.data_dir else "NONE",
        )
        return problems

    @staticmethod
    def _try_autoprint(code: str) -> Optional[str]:
        """Re-run code with print() on last bare expression to recover stdout."""
        import subprocess, sys, tempfile
        from pathlib import Path as _Path

        code_lines = [l for l in code.splitlines() if l.strip()]
        if not code_lines:
            return None
        last = code_lines[-1].strip()
        # Skip if already a print, comment, return, or a standalone assignment
        # (for assignments we try appending print(varname) instead)
        if last.startswith("print(") or last.startswith("#") or last.startswith("return"):
            return None

        import re as _re
        # If it's a bare assignment like `result = ...`, try printing the variable
        assign_m = _re.match(r"^([A-Za-z_]\w*)\s*=\s*(?!\s*=)", last)
        if assign_m:
            patched = "\n".join(code_lines) + f"\nprint({assign_m.group(1)})"
        elif "=" not in last:
            patched = "\n".join(code_lines[:-1]) + f"\nprint({last})"
        else:
            return None

        try:
            with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
                f.write(patched)
                path = f.name
            proc = subprocess.run(
                [sys.executable, path],
                capture_output=True, text=True, timeout=30,
            )
            out = proc.stdout.strip()
            return out if proc.returncode == 0 and out else None
        except Exception:
            return None
        finally:
            _Path(path).unlink(missing_ok=True)

    def score(self, problem: Problem, result: ProblemResult) -> float:
        expected = problem.metadata.get("answer", "")
        level = problem.metadata.get("level", "")
        result.metadata["level"] = level
        result.metadata["task_id"] = problem.metadata.get("task_id", "")

        if not expected:
            result.metadata["score_reason"] = "no_ground_truth"
            return 0.0
        if result.agent_error:
            result.metadata["score_reason"] = "agent_error"
            result.metadata["predicted"] = ""
            return 0.0

        stdout = result.stdout or ""

        # Recover answer if stdout is empty
        if not stdout.strip() and result.generated_code:
            recovered = self._try_autoprint(result.generated_code)
            if recovered:
                stdout = recovered
                result.metadata["score_note"] = "autoprint_recovery"

        if not stdout.strip():
            result.metadata["score_reason"] = "no_output"
            result.metadata["predicted"] = ""
            return 0.0

        # Last non-empty stdout line = predicted answer
        lines = [l.strip() for l in stdout.strip().splitlines() if l.strip()]
        predicted = lines[-1] if lines else ""
        result.metadata["predicted"] = predicted[:200]

        if numeric_match(predicted, expected, rel_tol=self.numeric_tol):
            result.metadata["score_reason"] = "numeric_match"
            return 1.0
        if string_match(predicted, expected):
            result.metadata["score_reason"] = "string_match"
            return 1.0

        # Contains-match fallback for short expected values
        expected_lower = str(expected).strip().lower()
        if expected_lower and len(expected_lower) <= 20 and expected_lower in predicted.lower():
            result.metadata["score_reason"] = "contains_match"
            return 1.0

        result.metadata["score_reason"] = "mismatch"
        result.metadata["expected"] = str(expected)[:200]
        return 0.0

    def _extra_metrics(self, results: list[ProblemResult]) -> dict:
        metrics: dict = {}
        for level in ("easy", "hard"):
            subset = [r for r in results if r.metadata.get("level") == level]
            if subset:
                metrics[f"pass_rate_{level}"] = (
                    sum(1 for r in subset if r.score >= 1.0) / len(subset)
                )
        fail_reasons: dict[str, int] = {}
        for r in results:
            reason = r.metadata.get("score_reason", "unknown")
            if reason not in ("numeric_match", "string_match", "contains_match"):
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            metrics[f"fail_{reason}"] = count
        return metrics

    # ── Leaderboard submission ─────────────────────────────────────────────────

    def run_submission(
        self,
        runs_dir: str | Path = "runs",
        run_id: Optional[int] = None,
    ) -> Path:
        """Run against the full benchmark and write a leaderboard submission JSONL.

        Matches the official quickstart submission format:
          { "task_id": "5", "agent_answer": "NL", "reasoning_trace": "..." }

        Upload the output file to:
          https://huggingface.co/spaces/adyen/DABstep

        Args:
            runs_dir: Directory to write the JSONL file.
            run_id:   Integer timestamp for the run (auto-set if None).

        Returns:
            Path to the JSONL submission file.
        """
        if self.split != "default":
            logger.warning(
                "run_submission() is for split='default' (450 tasks, leaderboard). "
                "Currently split=%s — switching to 'default'.", self.split
            )
            self.split = "default"

        run_id = run_id or int(time.time())
        runs_dir = Path(runs_dir)
        runs_dir.mkdir(parents=True, exist_ok=True)
        out_path = runs_dir / f"{run_id}.jsonl"

        problems = self.load_problems()
        if self.max_problems:
            problems = problems[: self.max_problems]

        entries: list[dict] = []

        for i, problem in enumerate(problems):
            logger.info("[%d/%d] %s", i + 1, len(problems), problem.id)
            pr = self._run_one(problem)

            # Extract the agent's answer from stdout (last non-empty line).
            # If the agent errored out, agent_answer is left blank — do NOT use
            # the error traceback as the answer (leaderboard would score it 0 anyway).
            stdout = pr.stdout or ""
            if not stdout.strip() and pr.generated_code and not pr.agent_error:
                recovered = self._try_autoprint(pr.generated_code)
                if recovered:
                    stdout = recovered
            lines = [l.strip() for l in stdout.strip().splitlines() if l.strip()]
            agent_answer = lines[-1] if lines else ""

            entries.append({
                "task_id": problem.metadata.get("task_id", problem.id),
                "agent_answer": str(agent_answer),
                "reasoning_trace": str(pr.generated_code or ""),
            })
            logger.info("  → answer=%r", agent_answer[:80] if agent_answer else "(empty)")

        # Write JSONL
        with out_path.open("w") as f:
            for entry in entries:
                f.write(_json_mod.dumps(entry) + "\n")

        logger.info("Submission written → %s  (%d tasks)", out_path, len(entries))
        logger.info("Upload to: https://huggingface.co/spaces/adyen/DABstep")
        return out_path


# ── one-shot baseline (no agent loop) ──────────────────────────────────────────

_ONESHOT_SYSTEM = """\
You are an expert data analyst and Python programmer.
You will be given a factoid question and a set of data variables that are ALREADY \
loaded in memory. Output ONLY Python code that computes the answer — no \
explanations, no markdown fences, no prose.

Rules:
- The listed variables (payments, fees, merchant_data, …) already exist; do NOT \
reload them or read any files.
- The LAST line of your code MUST be a print() call that outputs ONLY the final \
answer (no labels, no extra prints).
/no_think"""


class DABStepOneShotEvaluator(DABStepEvaluator):
    """One-shot baseline for DABStep: a single, direct model call with NO agent.

    Mirrors ``DS1000OneShotEvaluator`` — it bypasses the entire LangGraph pipeline
    (no plan, no build_context, no self-debug loop). It sends one chat completion
    straight to the OpenAI-compatible model server (the same vLLM endpoint the
    agent uses), then executes the returned code with the seed-cell data preloaded
    so the inherited :meth:`DABStepEvaluator.score` can read the printed answer
    from stdout.

    Purpose: the ablation baseline. Comparing this against ``DABStepEvaluator``
    isolates the contribution of the agentic loop (planning + execution-grounded
    self-correction) from the model's raw single-shot ability. The task framing
    (question + guidelines + loaded-variable contract) is held identical, so the
    only variable is the agent loop itself.
    """

    benchmark_name = "dabstep_oneshot"

    def __init__(
        self,
        *,
        model_url: Optional[str] = None,
        model_id: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        exec_timeout: int = 60,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        import os
        self.model_url = (
            model_url or os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
        ).rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.exec_timeout = exec_timeout
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

    @staticmethod
    def _strip_code(content: str) -> str:
        """Remove <think> blocks and markdown fences from raw model output."""
        import re as _re
        content = _re.sub(r"<think>.*?</think>", "", content,
                          flags=_re.DOTALL | _re.IGNORECASE)
        content = _re.sub(r"^```(?:python)?\s*\n?", "", content.strip(),
                          flags=_re.MULTILINE)
        content = _re.sub(r"\n?```\s*$", "", content.strip(), flags=_re.MULTILINE)
        return content.strip()

    def _exec_with_seed(self, problem: Problem, code: str) -> tuple[str, str]:
        """Run the generated code with the seed-cell data preloaded.

        Returns (stdout, stderr). The seed cell loads payments/fees/etc. as the
        named variables the prompt promised, so the model's code can reference
        them just as the agent's executed cells would.
        """
        import subprocess, sys, tempfile
        from pathlib import Path as _Path

        seed_src = "\n".join(c["source"] for c in problem.seed_cells if c.get("source"))
        script = (seed_src + "\n" + code) if seed_src else code
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
                f.write(script)
                path = f.name
            proc = subprocess.run(
                [sys.executable, path],
                capture_output=True, text=True, timeout=self.exec_timeout,
            )
            return proc.stdout.strip(), proc.stderr.strip()
        except subprocess.TimeoutExpired:
            return "", f"TimeoutError: exceeded {self.exec_timeout}s"
        except Exception as exc:  # pragma: no cover - defensive
            return "", str(exc)
        finally:
            if path:
                _Path(path).unlink(missing_ok=True)

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
            content = self._strip_code(resp.json()["choices"][0]["message"]["content"] or "")
            stdout, stderr = self._exec_with_seed(problem, content)
            latency = time.perf_counter() - t0
            return ProblemResult(
                problem_id=problem.id,
                status="unknown",
                score=0.0,
                generated_code=content,
                stdout=stdout,
                stderr=stderr,
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
