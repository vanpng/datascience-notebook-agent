#!/usr/bin/env python
"""Convert a saved DABStep eval result JSON into a leaderboard submission JSONL.

The DABStep leaderboard (https://huggingface.co/spaces/adyen/DABstep) expects a
JSONL file where each line is:

    {"task_id": "1712", "agent_answer": "...", "reasoning_trace": "..."}

  - task_id        : bare task id (no "dabstep_" prefix)
  - agent_answer   : the final printed answer (last non-empty stdout line);
                     "" if the run produced no output or errored out — never an
                     error traceback (the leaderboard scores those 0 anyway).
  - reasoning_trace: the generated code.

This mirrors DABStepEvaluator.run_submission() but works offline from an already
saved eval/results/*.json file, so we don't have to re-run the model.

Usage:
    python scripts/make_dabstep_submission.py RESULTS.json [-o OUT.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _agent_answer(rec: dict) -> str:
    """Extract the final answer from a result record (last non-empty stdout line).

    If the agent errored out we deliberately return "" rather than using the
    traceback as the answer.
    """
    if rec.get("agent_error"):
        return ""
    stdout = (rec.get("stdout") or "").strip()
    if not stdout:
        return ""
    lines = [l.strip() for l in stdout.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def convert(results_path: Path, out_path: Path) -> int:
    data = json.loads(results_path.read_text())
    results = data.get("results", [])

    entries = []
    for rec in results:
        pid = rec.get("problem_id", "")
        # Prefer the explicit task_id in metadata; fall back to stripping the prefix.
        task_id = rec.get("metadata", {}).get("task_id") or pid
        if not task_id and pid.startswith("dabstep_"):
            task_id = pid[len("dabstep_"):]
        task_id = str(task_id).removeprefix("dabstep_")

        entries.append({
            "task_id": task_id,
            "agent_answer": _agent_answer(rec),
            "reasoning_trace": str(rec.get("generated_code") or ""),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return len(entries)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, help="Path to eval/results/*.json file")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="Output JSONL path (default: runs/<results-stem>.jsonl)")
    args = ap.parse_args()

    if not args.results.is_file():
        print(f"✗ Results file not found: {args.results}", file=sys.stderr)
        sys.exit(1)

    out = args.out or (Path("runs") / f"{args.results.stem}.jsonl")
    n = convert(args.results, out)
    n_answered = sum(
        1 for line in out.read_text().splitlines()
        if line.strip() and json.loads(line)["agent_answer"]
    )
    print(f"✓ Wrote {n} tasks → {out}  ({n_answered} with a non-empty answer)")


if __name__ == "__main__":
    main()
