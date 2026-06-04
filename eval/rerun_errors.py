#!/usr/bin/env python
"""Re-run only the server-errored DS-1000 problems and merge into a prior result.

The 2026-06-01 full agent run crashed the MLX server (Metal OOM) partway through,
leaving 275 problems with status="error" (all 'Connection error.'/'timed out' —
not genuine agent failures). This re-runs exactly those problems against a fresh
(cache-capped) server and merges them back into the original result file so we get
a clean full-1000 number without a 4.5h re-run.

Usage:
  uv run python eval/rerun_errors.py \
      --prior eval/results/mlx-community_Qwen3-4B-8bit/ds1000_20260601_152527.json
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from eval.benchmarks.ds1000 import DS1000Evaluator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True, help="Prior result JSON to repair")
    ap.add_argument("--api-url", default="http://127.0.0.1:8000")
    ap.add_argument("--results-dir",
                    default="eval/results/mlx-community_Qwen3-4B-8bit")
    args = ap.parse_args()

    prior_path = Path(args.prior)
    prior = json.loads(prior_path.read_text())
    prior_results = prior["results"]

    error_ids = {r["problem_id"] for r in prior_results if r["status"] == "error"}
    print(f"Prior file : {prior_path}")
    print(f"Total      : {len(prior_results)}")
    print(f"To re-run  : {len(error_ids)} errored problems")
    if not error_ids:
        print("Nothing to re-run.")
        return

    ev = DS1000Evaluator(
        api_url=args.api_url,
        execute=True,
        timeout=300,  # agent fires ~11 model calls/problem (debug loop + structured-output
                      # retries); worst-case problems need >120s. Interactive client uses 300s.
        results_dir=args.results_dir,
    )
    all_problems = ev.load_problems()
    subset = [p for p in all_problems if p.id in error_ids]
    print(f"Matched    : {len(subset)} problems in loader")
    missing = error_ids - {p.id for p in subset}
    if missing:
        print(f"WARNING: {len(missing)} errored ids not found in loader: "
              f"{sorted(missing)[:5]}…")

    # Run only the subset. (run() saves its own partial file too — harmless.)
    br = ev.run(subset)
    print(f"\nRe-run result: {br.passed}/{br.total} pass "
          f"({br.pass_rate:.1%}), {br.errors} still errored")

    # ── merge: replace errored entries by problem_id ──────────────────────────
    new_by_id = {r.problem_id: asdict(r) for r in br.results}
    merged = []
    replaced = 0
    for entry in prior_results:
        pid = entry["problem_id"]
        if pid in new_by_id:
            merged.append(new_by_id[pid])
            replaced += 1
        else:
            merged.append(entry)
    print(f"Replaced   : {replaced} entries")

    passed = sum(1 for r in merged if r["status"] == "pass")
    errors = sum(1 for r in merged if r["status"] == "error")
    total = len(merged)
    lat = [r.get("latency_s", 0.0) for r in merged]

    combined = dict(prior)
    combined["results"] = merged
    combined["total"] = total
    combined["passed"] = passed
    combined["failed"] = total - passed - errors
    combined["errors"] = errors
    combined["pass_rate"] = passed / total if total else 0.0
    combined["mean_latency_s"] = sum(lat) / len(lat) if lat else 0.0

    out_path = prior_path.with_name(prior_path.stem + "_merged.json")
    out_path.write_text(json.dumps(combined, indent=2))

    print("\n" + "=" * 50)
    print(f"MERGED full-1000 result → {out_path}")
    print(f"  pass@1 : {passed}/{total} = {combined['pass_rate']:.1%}")
    print(f"  failed : {combined['failed']}")
    print(f"  errors : {errors}  (should be ~0)")
    print("=" * 50)


if __name__ == "__main__":
    main()
