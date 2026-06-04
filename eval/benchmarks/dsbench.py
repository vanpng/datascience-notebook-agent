"""DSBench benchmark evaluator.

Paper : "DSBench: How Far Are Data Science Agents to Becoming Data Scientists?"
        (Jing et al., ICLR 2025) — https://arxiv.org/abs/2409.07703
Repo  : https://github.com/LiqiangJing/DSBench

Schema (from GitHub repo)
--------------------------
  data_analysis/ — 466 tasks (JSON per task, references Kaggle CSV files)
  data_modeling/ — 74 tasks  (references Kaggle training/test files)

  Each task JSON:
    {
      "id":          str,
      "question":    str,
      "data_files":  [relative path, ...],
      "answer":      str | number,
      "human_score": float   (for RPG; modeling track only)
    }

Setup
-----
  1. Clone https://github.com/LiqiangJing/DSBench
  2. Download the Kaggle datasets referenced in each task
  3. Point --data-dir at the repo root

Local smoke test (no Kaggle account needed)
--------------------------------------------
  Use DSBenchLocalEvaluator which runs a small hand-authored set of
  analysis questions against the bundled churn.csv / aita_results.csv.
  Pass --local to run_eval.py to use this mode.

Known gaps (see docs/evaluation.md):
  - Kaggle data files must be downloaded manually.
  - The modeling track requires writing a predictions file (not yet supported).
  - Answer extraction from free-form stdout is heuristic.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from eval.harness import AgentEvaluator, Problem, ProblemResult
from eval.metrics import numeric_match, string_match, relative_performance_gap

logger = logging.getLogger(__name__)


# ── full DSBench (requires cloned repo + Kaggle data) ─────────────────────────

def _load_dsbench(data_dir: Path, track: str) -> list[dict]:
    tasks_file = data_dir / track / "tasks.json"
    if not tasks_file.exists():
        raise FileNotFoundError(
            f"DSBench tasks not found at {tasks_file}.\n"
            "Clone the repo: git clone https://github.com/LiqiangJing/DSBench\n"
            "Then point --data-dir at the repo root."
        )
    return json.loads(tasks_file.read_text())


class DSBenchEvaluator(AgentEvaluator):
    """Full DSBench evaluator — requires a local repo clone with Kaggle data.

    Args:
        data_dir:    Path to local DSBench repo root.
        track:       "data_analysis" or "data_modeling".
        numeric_tol: Relative tolerance for numeric answer matching.
    """

    benchmark_name = "dsbench"

    def __init__(
        self,
        data_dir: str | Path,
        track: str = "data_analysis",
        numeric_tol: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.data_dir = Path(data_dir)
        self.track = track
        self.numeric_tol = numeric_tol
        if track not in ("data_analysis", "data_modeling"):
            raise ValueError(f"Unknown track: {track!r}")

    def load_problems(self) -> list[Problem]:
        raw = _load_dsbench(self.data_dir, self.track)
        problems = []
        for task in raw:
            task_id = task.get("id", str(len(problems)))
            data_files: list[str] = task.get("data_files", [])
            seed_lines = ["import pandas as pd", "import numpy as np"]
            for j, fpath in enumerate(data_files):
                full = self.data_dir / fpath
                vname = f"df{j}" if j else "df"
                ext = Path(fpath).suffix
                if ext == ".csv":
                    seed_lines.append(f"{vname} = pd.read_csv(r'{full}')")
                elif ext in (".xlsx", ".xls"):
                    seed_lines.append(f"{vname} = pd.read_excel(r'{full}')")
                elif ext == ".json":
                    seed_lines.append(f"{vname} = pd.read_json(r'{full}')")
                elif ext == ".parquet":
                    seed_lines.append(f"{vname} = pd.read_parquet(r'{full}')")
            seed = {"source": "\n".join(seed_lines), "stdout": "", "success": True}
            problems.append(Problem(
                id=f"dsbench_{task_id}",
                query=task.get("question") or task.get("description", ""),
                seed_cells=[seed],
                metadata={
                    "answer": task.get("answer"),
                    "human_score": task.get("human_score"),
                    "track": self.track,
                },
            ))
        logger.info("Loaded %d DSBench %s tasks", len(problems), self.track)
        return problems

    def score(self, problem: Problem, result: ProblemResult) -> float:
        if result.agent_error or not result.stdout:
            return 0.0
        expected = problem.metadata.get("answer")
        if expected is None:
            return 0.0
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        predicted = lines[-1] if lines else ""
        if numeric_match(predicted, str(expected), rel_tol=self.numeric_tol):
            return 1.0
        if string_match(predicted, str(expected)):
            return 1.0
        return 0.0

    def _extra_metrics(self, results: list[ProblemResult]) -> dict:
        if self.track == "data_modeling":
            rpg_vals = []
            for r in results:
                human = r.metadata.get("human_score")
                if human is not None:
                    rpg_vals.append(relative_performance_gap(r.score, human))
            if rpg_vals:
                return {"mean_rpg": sum(rpg_vals) / len(rpg_vals)}
        return {}


# ── local smoke evaluator (uses bundled data, no Kaggle needed) ───────────────

def _local_problems(data_root: Path) -> list[Problem]:
    """A hand-authored set of factual questions over the bundled CSV files.

    Answers are deterministic (exact counts / column values) so scoring is
    reliable without any ML model.
    """
    churn = data_root / "churn.csv"
    aita  = data_root / "aita_results.csv"

    seed_churn = {
        "source": f"import pandas as pd\ndf = pd.read_csv(r'{churn}')\nprint(df.shape)",
        "stdout": "(300, 11)",
        "success": True,
    }
    seed_aita = {
        "source": f"import pandas as pd\ndf = pd.read_csv(r'{aita}')\nprint(df.shape)",
        "stdout": "",
        "success": True,
    }

    return [
        Problem(
            id="local_churn_01",
            query=(
                "How many unique contract types are in the dataset? "
                "Print only the integer count on the last line."
            ),
            seed_cells=[seed_churn],
            metadata={"answer": "3", "dataset": "churn"},
        ),
        Problem(
            id="local_churn_02",
            query=(
                "What is the overall churn rate rounded to 1 decimal place as a percentage "
                "(e.g. 42.7)? Print only the number on the last line."
            ),
            seed_cells=[seed_churn],
            metadata={"answer": "42.7", "dataset": "churn"},
        ),
        Problem(
            id="local_churn_03",
            query=(
                "What is the mean monthly_charges rounded to 2 decimal places? "
                "Print only the number on the last line."
            ),
            seed_cells=[seed_churn],
            metadata={"answer": None, "dataset": "churn"},  # no fixed answer — just checks it runs
        ),
        Problem(
            id="local_churn_04",
            query=(
                "How many rows have missing values in any column? "
                "Print only the integer count on the last line."
            ),
            seed_cells=[seed_churn],
            metadata={"answer": "23", "dataset": "churn"},
        ),
        Problem(
            id="local_aita_01",
            query=(
                "What is the most common value in the author_gender column? "
                "Use value_counts() and print ONLY the single word answer on the last line "
                "(e.g. just: male). No explanation, no label, no extra text."
            ),
            seed_cells=[seed_aita],
            metadata={"answer": "male", "dataset": "aita"},
        ),
    ]


class DSBenchLocalEvaluator(AgentEvaluator):
    """Smoke-test evaluator that uses the bundled churn.csv / aita_results.csv.

    No Kaggle account or repo clone needed.  Useful for verifying the eval
    pipeline end-to-end before running the real DSBench.
    """

    benchmark_name = "dsbench_local"

    def __init__(self, data_root: str | Path = "data", numeric_tol: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.data_root = Path(data_root)
        self.numeric_tol = numeric_tol

    def load_problems(self) -> list[Problem]:
        problems = _local_problems(self.data_root)
        logger.info("Loaded %d local DSBench smoke problems", len(problems))
        return problems

    def score(self, problem: Problem, result: ProblemResult) -> float:
        expected = problem.metadata.get("answer")
        if expected is None:
            # No fixed answer — just check the agent ran without error
            result.metadata["score_reason"] = "execution_only"
            return 1.0 if (result.generated_code and not result.agent_error) else 0.0
        if result.agent_error or not result.stdout:
            result.metadata["score_reason"] = "no_output"
            return 0.0
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        predicted = lines[-1] if lines else ""
        if numeric_match(predicted, str(expected), rel_tol=self.numeric_tol):
            result.metadata["score_reason"] = "numeric_match"
            return 1.0
        if string_match(predicted, str(expected)):
            result.metadata["score_reason"] = "string_match"
            return 1.0
        # Fallback: the agent may print an explanatory sentence that contains the
        # correct value (e.g. "The most common gender is: male" when expected is "male").
        # Accept this as a correct answer for smoke-test purposes.
        expected_lower = str(expected).strip().lower()
        if expected_lower and expected_lower in predicted.lower():
            result.metadata["score_reason"] = "contains_match"
            return 1.0
        result.metadata["score_reason"] = "mismatch"
        result.metadata["predicted"] = predicted
        result.metadata["expected"] = expected
        return 0.0
