"""Evaluation metrics used across benchmarks.

References:
  pass@k  — Chen et al. 2021 (HumanEval), unbiased estimator
  RPG     — DSBench: Relative Performance Gap
"""
from __future__ import annotations

import math
import re
from typing import Sequence


# ── pass@k (unbiased) ─────────────────────────────────────────────────────────

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from Chen et al. 2021.

    Args:
        n: Total samples generated for this problem.
        c: Number of samples that pass.
        k: k in pass@k.

    Returns:
        Probability that at least one of k samples passes.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_at_k_bulk(results_per_problem: list[list[bool]], k: int) -> float:
    """Compute mean pass@k over a list of per-problem pass/fail lists."""
    if not results_per_problem:
        return 0.0
    scores = [
        pass_at_k(len(runs), sum(runs), k)
        for runs in results_per_problem
    ]
    return sum(scores) / len(scores)


# ── execution accuracy ────────────────────────────────────────────────────────

def execution_accuracy(passed: int, total: int) -> float:
    """Simple pass rate."""
    return passed / total if total > 0 else 0.0


# ── numeric result matching ───────────────────────────────────────────────────

def numeric_match(predicted: str, expected: str, rel_tol: float = 0.01) -> bool:
    """Check whether a numeric answer in stdout matches expected within tolerance.

    Extracts the last float-like token from each string and compares.
    Used for DSBench and DABStep where answers are numeric.
    """
    def _extract_last_number(s: str) -> float | None:
        tokens = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", s)
        if not tokens:
            return None
        try:
            return float(tokens[-1])
        except ValueError:
            return None

    p = _extract_last_number(str(predicted))
    e = _extract_last_number(str(expected))
    if p is None or e is None:
        return str(predicted).strip() == str(expected).strip()
    if e == 0:
        return abs(p) < 1e-9
    return abs(p - e) / abs(e) <= rel_tol


def string_match(predicted: str, expected: str, case_sensitive: bool = False) -> bool:
    """Exact string match after stripping whitespace."""
    p = predicted.strip()
    e = expected.strip()
    if not case_sensitive:
        p, e = p.lower(), e.lower()
    return p == e


# ── relative performance gap (DSBench) ───────────────────────────────────────

def relative_performance_gap(agent_score: float, human_score: float) -> float:
    """RPG = 1 - (agent_score / human_score).

    0.0 means agent matches human; 1.0 means agent scores 0.
    Negative values mean agent exceeds human.
    """
    if human_score == 0:
        return 0.0 if agent_score == 0 else float("inf")
    return 1.0 - (agent_score / human_score)


# ── aggregate report ──────────────────────────────────────────────────────────

def summarise(
    scores: Sequence[float],
    label: str = "score",
) -> dict[str, float]:
    """Return mean, median, and pass rate for a list of scores."""
    if not scores:
        return {}
    n = len(scores)
    total = sum(scores)
    sorted_s = sorted(scores)
    median = (sorted_s[n // 2] if n % 2 else (sorted_s[n // 2 - 1] + sorted_s[n // 2]) / 2)
    return {
        f"mean_{label}": total / n,
        f"median_{label}": median,
        f"pass_rate": sum(1 for s in scores if s >= 1.0) / n,
    }
