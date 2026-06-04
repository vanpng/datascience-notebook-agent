"""Curated dev/test evaluator (prompt-leakage-safe tuning set).

This evaluator runs the 30 hand-curated, benchmark-independent scenarios in
eval/devtest/scenarios.json against the live agent using the DEFAULT profile
(the normal plan → generate → execute → debug notebook workflow).

It reports two things:

  • pass_rate          — fraction of scenarios the agent answers correctly.
                         Answer scenarios: the expected value must appear EITHER
                         in the cell's own stdout OR in a notebook-faithful
                         re-execution of the generated cell (see below). Match is
                         numeric within 1% tolerance, or a word-boundary token /
                         substring match for labels.
                         Plot scenarios: the cell must execute successfully AND
                         the generated code must contain a real plotting call.

    Notebook-faithful re-execution
    ------------------------------
    The agent targets a Jupyter notebook, where the value of a cell's last
    top-level statement is auto-displayed — IPython's documented
    ``ast_node_interactivity = 'last_expr_or_assign'`` behaviour, which echoes a
    trailing expression OR a trailing assignment's target. The agent's sandbox,
    however, runs each cell as a plain subprocess script with NO auto-display, so
    a cell that computes the answer into a variable (very common, notebook-style)
    leaves stdout empty. To score the agent as it would behave in a real
    notebook, we re-run the generated cell over the scenario's setup data with
    that last-expr-or-assign echo applied, and accept a match from either the
    agent's own stdout or this re-execution. This is purely an evaluator fairness
    fix — the agent prompts are unchanged.

  • format_compliance  — fraction of returned code cells that satisfy the
                         agent's own output-format contract for the "code" field
                         (see src/agent/prompts.py::_JSON_OUTPUT_RULES and the
                         DEFAULT_SYSTEM_GENERATE "Code rules"):
                           - non-empty
                           - parses as valid Python (ast.parse)
                           - NO markdown fences / backticks
                           - NO ellipsis / placeholder / TODO / "your code here"
                         This is independent of correctness — a cell can be
                         format-compliant but produce the wrong answer, or vice
                         versa — so the two rates are reported separately.
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import re
from pathlib import Path
from typing import Any

import matplotlib  # force a headless backend so re-executed plotting cells never block
matplotlib.use("Agg")

from eval.harness import AgentEvaluator, Problem, ProblemResult
from eval.metrics import numeric_match

_DEFAULT_SCENARIOS = Path(__file__).resolve().parents[1] / "devtest" / "scenarios.json"

# Tokens that signal an incomplete / placeholder cell (forbidden by the prompt).
_PLACEHOLDER_PATTERNS = [
    re.compile(r"\bTODO\b", re.I),
    re.compile(r"\bFIXME\b", re.I),
    re.compile(r"your code here", re.I),
    re.compile(r"<\s*insert", re.I),
    re.compile(r"\.\.\.\s*#"),                 # `...  # placeholder`
    re.compile(r"^\s*\.\.\.\s*$", re.M),       # a bare ellipsis statement
    re.compile(r"\bpass\b\s*#\s*(?:placeholder|implement)", re.I),
]


def check_format_compliance(code: str | None) -> tuple[bool, list[str]]:
    """Return (compliant, issues) for a generated code-cell string.

    Validates the agent's output-format contract for the "code" field.
    """
    issues: list[str] = []
    if not code or not code.strip():
        return False, ["empty code"]

    if "```" in code:
        issues.append("contains markdown fence (```)")
    if "`" in code:
        issues.append("contains backtick")

    for pat in _PLACEHOLDER_PATTERNS:
        if pat.search(code):
            issues.append(f"placeholder/ellipsis match: {pat.pattern!r}")

    try:
        ast.parse(code)
    except SyntaxError as exc:
        issues.append(f"not valid Python: {exc.msg}")

    return (len(issues) == 0), issues


def _echo_last_node(mod: ast.Module) -> ast.Module:
    """Append a print() of the last top-level statement's value.

    Mimics IPython's ``last_expr_or_assign``: a trailing expression OR the target
    of a trailing assignment is echoed.
    """
    if not mod.body:
        return mod
    last = mod.body[-1]
    target: ast.expr | None = None
    if isinstance(last, ast.Expr):
        target = last.value
    elif isinstance(last, ast.Assign) and last.targets:
        t = last.targets[0]
        if isinstance(t, (ast.Name, ast.Subscript, ast.Attribute, ast.Tuple)):
            target = t
    elif isinstance(last, (ast.AnnAssign, ast.AugAssign)) and getattr(last, "target", None):
        target = last.target
    if target is not None:
        printer = ast.Expr(
            ast.Call(func=ast.Name(id="print", ctx=ast.Load()),
                     args=[ast.copy_location(target, last)], keywords=[])
        )
        mod.body.append(ast.copy_location(printer, last))
        ast.fix_missing_locations(mod)
    return mod


def notebook_reexecute(setup_code: str, cell_code: str) -> str:
    """Re-run the generated cell over the scenario setup with notebook semantics.

    Returns captured stdout (the cell's own prints plus the auto-echoed value of
    its last expression/assignment). Best-effort: any error yields whatever was
    printed before it. Isolated namespace per call; safe for the simple, loop-free
    compute cells in this set.
    """
    if not cell_code or not cell_code.strip():
        return ""
    try:
        mod = ast.parse(cell_code)
    except SyntaxError:
        return ""
    mod = _echo_last_node(mod)
    try:
        transformed = ast.unparse(mod)
    except Exception:
        transformed = cell_code
    buf = io.StringIO()
    ns: dict = {}
    try:
        with contextlib.redirect_stdout(buf):
            exec(setup_code, ns)
            exec(transformed, ns)
    except Exception:
        pass
    finally:
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass
    return buf.getvalue()


def _contains_match(expected: str, stdout: str) -> bool:
    """Label match: word-boundary for short tokens, case-insensitive substring otherwise."""
    exp = expected.strip()
    out = stdout or ""
    # Short / single-token labels (e.g. 'C', 'Eng') → require a word boundary so
    # a stray letter elsewhere in the output doesn't count as a hit.
    if len(exp) <= 3 and re.fullmatch(r"[A-Za-z]+", exp):
        return re.search(rf"\b{re.escape(exp)}\b", out) is not None
    return exp.lower() in out.lower()


class DevTestEvaluator(AgentEvaluator):
    """Runs the curated 30-scenario dev/test set against the agent."""

    benchmark_name = "devtest30"

    def __init__(self, *args, scenarios_path: str | Path = _DEFAULT_SCENARIOS, **kwargs):
        # This set is designed for the normal notebook workflow.
        kwargs.setdefault("prompt_profile", "default")
        kwargs.setdefault("completion_mode", False)
        super().__init__(*args, **kwargs)
        self.scenarios_path = Path(scenarios_path)

    def load_problems(self) -> list[Problem]:
        data = json.loads(self.scenarios_path.read_text())
        return [
            Problem(
                id=s["id"],
                query=s["query"],
                seed_cells=s["seed_cells"],
                metadata=s["metadata"],
            )
            for s in data["scenarios"]
        ]

    def score(self, problem: Problem, result: ProblemResult) -> float:
        meta = problem.metadata
        check = meta.get("check")
        stdout = result.stdout or ""

        # ── format compliance (independent of correctness) ──
        compliant, issues = check_format_compliance(result.generated_code)
        result.metadata["format_compliant"] = compliant
        result.metadata["format_issues"] = issues
        result.metadata["check"] = check
        result.metadata["intent"] = meta.get("intent")
        result.metadata["expected"] = meta.get("expected")

        # ── correctness ──
        if result.agent_error:
            return 0.0

        if check == "plot":
            code = result.generated_code or ""
            ran_ok = not (result.stderr or "").strip()
            has_plot = any(m in code for m in meta.get("plot_markers", []))
            return 1.0 if (ran_ok and has_plot) else 0.0

        expected = str(meta.get("expected", "")).strip()

        # Notebook-faithful re-execution: accept the answer from the agent's own
        # stdout OR from re-running the cell with last-expr-or-assign auto-echo.
        nb_stdout = notebook_reexecute(meta.get("setup_code", ""), result.generated_code or "")
        result.metadata["nb_stdout_tail"] = (nb_stdout or "").strip()[-120:]

        def _matches(out: str) -> bool:
            if check == "numeric":
                return numeric_match(out, expected, rel_tol=0.01)
            if check == "contains":
                return _contains_match(expected, out)
            return False

        return 1.0 if (_matches(stdout) or _matches(nb_stdout)) else 0.0

    def _extra_metrics(self, results: list[ProblemResult]) -> dict[str, Any]:
        n = len(results) or 1
        compliant = sum(1 for r in results if r.metadata.get("format_compliant"))

        # pass-rate broken down by intent
        by_intent: dict[str, list[float]] = {}
        for r in results:
            intent = r.metadata.get("intent", "unknown")
            by_intent.setdefault(intent, []).append(1.0 if r.status == "pass" else 0.0)

        metrics: dict[str, Any] = {
            "format_compliance_rate": round(compliant / n, 4),
            "format_compliant": compliant,
            "format_noncompliant": n - compliant,
        }
        for intent, scores in sorted(by_intent.items()):
            metrics[f"pass_{intent}"] = round(sum(scores) / len(scores), 4)
        return metrics
