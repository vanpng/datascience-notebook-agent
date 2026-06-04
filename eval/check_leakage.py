#!/usr/bin/env python
"""CLI: check DS-1000 / DABStep contamination in the fine-tuning data.

Reconstructs the exact data the model was fine-tuned on (a seed-42 sample of
``jupyter-agent/jupyter-agent-dataset`` per ``fine_tuning/config.yaml``) and
measures n-gram containment of each benchmark problem statement / reference
solution / question against it.  No model server or agent API required.

Examples
--------
# Default: scan the full non_thinking split, check both benchmarks
python eval/check_leakage.py

# Scan both splits, also decode each row's raw original_notebook
python eval/check_leakage.py --splits non_thinking,thinking --include-notebook

# DS-1000 only, 13-grams, lower flag threshold, include keyword-filtered rows
python eval/check_leakage.py --benchmarks ds1000 --n 13 \
    --flag-threshold 0.6 --include-filtered

# Quick smoke test (cap docs scanned)
python eval/check_leakage.py --max-train-docs 500
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from eval.leakage import (
    LeakageDetector,
    iter_training_docs,
    load_dabstep_units,
    load_ds1000_units,
    summarize,
)

console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

_CFG_PATH = Path(__file__).resolve().parents[1] / "fine_tuning" / "config.yaml"


def _load_ft_cfg() -> dict:
    if _CFG_PATH.exists():
        return yaml.safe_load(_CFG_PATH.read_text()).get("data", {})
    return {}


def main() -> None:
    cfg = _load_ft_cfg()
    p = argparse.ArgumentParser(description="Benchmark contamination check for the fine-tuning data")
    p.add_argument("--benchmarks", default="ds1000,dabstep",
                   help="Comma-separated: ds1000,dabstep (default: both)")
    p.add_argument("--n", type=int, default=8, help="n-gram size in words (default: 8)")
    p.add_argument("--flag-threshold", type=float, default=0.8,
                   help="max_doc_containment at/above which a unit is flagged (default: 0.8)")
    p.add_argument("--splits", default="non_thinking",
                   help="Comma-separated dataset splits to scan (default: non_thinking; "
                        "the dataset also has 'thinking')")
    p.add_argument("--sample-size", type=int, default=None,
                   help="Scan a seeded random subset of this many rows per split "
                        "(default: scan ALL rows — the conservative check)")
    p.add_argument("--seed", type=int, default=int(cfg.get("seed", 42)),
                   help="RNG seed for --sample-size (default from fine_tuning/config.yaml)")
    p.add_argument("--include-filtered", action="store_true",
                   help="Do NOT drop rows matching benchmark keywords (scans them too)")
    p.add_argument("--include-notebook", action="store_true",
                   help="Also scan each row's raw original_notebook JSON (slower, broader)")
    p.add_argument("--max-train-docs", type=int, default=None,
                   help="Cap training docs scanned (smoke test)")
    p.add_argument("--dataset-id", default=cfg.get("dataset_id", "jupyter-agent/jupyter-agent-dataset"))
    # benchmark knobs
    p.add_argument("--ds1000-libraries", default=None, help="Comma-separated DS-1000 library subset")
    p.add_argument("--dabstep-splits", default="dev,default", help="DABStep splits to union (default: dev,default)")
    p.add_argument("--results-dir", default="eval/results")
    args = p.parse_args()

    benches = [b.strip().lower() for b in args.benchmarks.split(",") if b.strip()]

    # ── 1. load benchmark units ──────────────────────────────────────────────
    units = []
    if "ds1000" in benches:
        libs = args.ds1000_libraries.split(",") if args.ds1000_libraries else None
        units += load_ds1000_units(libs)
    if "dabstep" in benches:
        splits = [s.strip() for s in args.dabstep_splits.split(",") if s.strip()]
        units += load_dabstep_units(splits)
    if not units:
        console.print("[red]No benchmark units loaded — check --benchmarks[/red]")
        return

    # ── 2. build index + scan corpus ─────────────────────────────────────────
    det = LeakageDetector(n=args.n, flag_threshold=args.flag_threshold)
    det.build_index(units)

    scan_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    docs = iter_training_docs(
        dataset_id=args.dataset_id, splits=scan_splits,
        sample_size=args.sample_size, seed=args.seed,
        include_filtered=args.include_filtered, include_notebook=args.include_notebook,
        max_docs=args.max_train_docs,
    )
    t0 = time.perf_counter()
    det.scan(docs)
    findings = det.findings()
    report = summarize(findings, args.flag_threshold)
    report["meta"] = {
        "benchmarks": benches,
        "n_gram": args.n,
        "dataset_id": args.dataset_id,
        "splits": scan_splits,
        "sample_size": args.sample_size,
        "seed": args.seed if args.sample_size else None,
        "include_filtered": args.include_filtered,
        "include_notebook": args.include_notebook,
        "train_docs_scanned": det.n_docs,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }

    # ── 3. persist ───────────────────────────────────────────────────────────
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"leakage_{'-'.join(scan_splits)}_{ts}.json"
    out_path.write_text(json.dumps({"report": report, "findings": [vars(f) for f in findings]}, indent=2))

    # ── 4. print ─────────────────────────────────────────────────────────────
    console.rule("[bold]Benchmark leakage check[/bold]")
    console.print(f"Splits        : {', '.join(scan_splits)}  ({det.n_docs} docs scanned, n={args.n}-grams)")
    console.print(f"Benchmarks    : {', '.join(benches)}   units={report['total_units']}")
    verdict = ("[red]LEAKAGE DETECTED[/red]" if report["leakage_detected"]
               else "[green]No leakage above threshold[/green]")
    console.print(f"Flagged (≥{args.flag_threshold:.0%}) : {report['total_flagged']}   → {verdict}")

    table = Table("benchmark/field", "units", "mean", "max", "≥0.5", "≥0.8", "=1.0",
                  title=f"max single-doc containment (n={args.n})")
    for key, g in report["by_group"].items():
        table.add_row(key, str(g["units"]),
                      f"{g['mean_max_doc_containment']:.3f}", f"{g['max_max_doc_containment']:.3f}",
                      str(g["n_ge_0.5"]), str(g["n_ge_0.8"]), str(g["n_eq_1.0"]))
    console.print(table)

    if report["flagged"]:
        console.rule("[yellow]Flagged units (top 15)[/yellow]")
        for f in report["flagged"][:15]:
            console.print(
                f"  [bold]{f['uid']}[/bold]  max_doc={f['max_doc_containment']:.2f} "
                f"corpus={f['corpus_containment']:.2f}  doc={f['best_doc_id']}  "
                f"lcs={f['lcs_len']}chars"
            )
            if f.get("longest_common_substring"):
                console.print(f"    [dim]LCS:[/dim] {f['longest_common_substring'][:160]!r}")

    console.print(f"\n[green]Full report → {out_path}[/green]")


if __name__ == "__main__":
    main()
