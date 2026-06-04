#!/usr/bin/env python
"""CLI runner for benchmark evaluation.

Usage examples
--------------
# DS-1000, all libraries, pass@1
python eval/run_eval.py --benchmark ds1000

# DS-1000, Pandas + NumPy only, first 50 problems
python eval/run_eval.py --benchmark ds1000 --libraries Pandas,Numpy --max 50

# DSBench data_analysis track
python eval/run_eval.py --benchmark dsbench --data-dir /path/to/DSBench

# DABStep dev eval (10 tasks with ground truth)
python eval/run_eval.py --benchmark dabstep --data-dir /path/to/dabstep/context

# DABStep easy tier only
python eval/run_eval.py --benchmark dabstep --data-dir /path/to/context --difficulty easy

# DABStep leaderboard submission (450 tasks → runs/{RUN_ID}.jsonl)
python eval/run_eval.py --benchmark dabstep --data-dir /path/to/context --submit

# DSCodeBench, all 10 libraries (auto-resolves DSCodeBench.json)
python eval/run_eval.py --benchmark dscodebench

# DSCodeBench, numpy + pandas only, 200 test cases (paper-faithful), 5 per lib
python eval/run_eval.py --benchmark dscodebench --libraries numpy,pandas \
    --test-cases 200 --max-per-library 5 --data-file data/DSCodeBench.json

# DSCodeBench one-shot baseline (no agent loop)
python eval/run_eval.py --benchmark dscodebench --no-agent

# Check API is reachable before running
python eval/run_eval.py --check
"""
from __future__ import annotations

import argparse
import logging
import sys

import httpx
from rich.console import Console
from rich.table import Table

console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


_KNOWN_MODELS = {
    "base":      "mlx-community/Qwen3-4B-8bit",      # default (macOS/MLX)
    "base-linux": "Qwen/Qwen3-4B",                    # default (Linux/vLLM)
    "finetuned": "jupyter-agent/jupyter-agent-qwen3-4b-instruct",
}


def _check_api(api_url: str) -> bool:
    try:
        r = httpx.get(f"{api_url}/health", timeout=5)
        r.raise_for_status()
        console.print(f"[green]✓ Agent API reachable at {api_url}[/green]")
        return True
    except Exception as exc:
        console.print(f"[red]✗ Cannot reach agent API at {api_url}: {exc}[/red]")
        console.print("  Start the server with:  bash scripts/start.sh")
        return False


def _probe_server_model(vllm_url: str = "http://127.0.0.1:8001") -> str:
    """Return the model ID currently loaded in the inference server."""
    try:
        r = httpx.get(f"{vllm_url}/v1/models", timeout=5)
        data = r.json().get("data", [])
        return data[0]["id"] if data else "(unknown)"
    except Exception:
        return "(unknown)"


def _resolve_model(model_arg: str) -> str:
    """Expand a short alias (base, finetuned) to a full HuggingFace model ID."""
    return _KNOWN_MODELS.get(model_arg, model_arg)


def _print_result(br) -> None:
    console.rule(f"[bold]{br.benchmark}[/bold]")
    console.print(br.summary())

    if br.extra_metrics:
        table = Table("Metric", "Value", title="Per-category breakdown")
        for k, v in sorted(br.extra_metrics.items()):
            table.add_row(k, f"{v:.3f}" if isinstance(v, float) else str(v))
        console.print(table)

    # Top failures
    failures = [r for r in br.results if r.status != "pass"][:5]
    if failures:
        console.rule("[yellow]Sample failures[/yellow]")
        for r in failures:
            console.print(f"  [dim]{r.problem_id}[/dim]  agent_error={r.agent_error or '—'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the DS Notebook Agent on benchmarks")
    parser.add_argument("--benchmark",
                        choices=["ds1000", "dsbench", "dsbench_local", "dabstep", "dscodebench", "devtest"],
                        help="Which benchmark to run")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000",
                        help="Agent API base URL")
    parser.add_argument("--model", default=None, metavar="MODEL_ID",
                        help=(
                            "Model to evaluate. Aliases: base, finetuned. "
                            "Or any HuggingFace model ID. "
                            "If the running server uses a different model, instructions are printed. "
                            f"Known aliases: {', '.join(f'{k}={v}' for k,v in _KNOWN_MODELS.items())}"
                        ))
    parser.add_argument("--max", type=int, default=None, dest="max_problems",
                        help="Limit number of problems (for quick smoke tests)")
    parser.add_argument("--no-execute", action="store_true",
                        help="Dry-run: generate code but don't execute it (scores will be 0)")
    parser.add_argument("--no-agent", action="store_true",
                        help="DS-1000/DABStep/DSCodeBench: bypass the agent loop — "
                             "single direct model call (ablation baseline)")
    parser.add_argument("--results-dir", default="eval/results",
                        help="Directory to write JSON result files")
    parser.add_argument("--concurrency", type=int, default=1, metavar="N",
                        help="Number of problems to evaluate in parallel (default: 1)")
    parser.add_argument("--check", action="store_true",
                        help="Check API health and exit")

    # DS-1000 options
    parser.add_argument("--libraries", default=None,
                        help="DS-1000: comma-separated library list, e.g. Pandas,Numpy")
    parser.add_argument("--max-per-library", type=int, default=None, dest="max_per_library",
                        help="DS-1000: max problems per library (e.g. 10 → 70 total across 7 libs)")

    # DSCodeBench options
    parser.add_argument("--data-file", default=None,
                        help="DSCodeBench: path to DSCodeBench.json (or a dir containing it). "
                             "Auto-resolved from common locations if omitted.")
    parser.add_argument("--test-cases", type=int, default=50, dest="test_cases",
                        help="DSCodeBench: test-case inputs generated per problem "
                             "(upstream default 200; lower is faster)")

    # DSBench options
    parser.add_argument("--data-dir", default=None,
                        help="DSBench/DABStep: path to local dataset directory")
    parser.add_argument("--track", default="data_analysis",
                        choices=["data_analysis", "data_modeling"],
                        help="DSBench: which track to evaluate")

    # DABStep options
    parser.add_argument("--difficulty", default=None,
                        choices=["easy", "medium", "hard"],
                        help="DABStep: filter by difficulty level")
    parser.add_argument("--split", default="dev",
                        choices=["dev", "default"],
                        help="DABStep: 'dev' (10 tasks, ground truth) or 'default' (450 tasks, leaderboard)")
    parser.add_argument("--submit", action="store_true",
                        help="DABStep: run full benchmark (split=default) and write leaderboard JSONL to runs/")

    args = parser.parse_args()

    if not _check_api(args.api_url):
        sys.exit(1)

    # ── model check ───────────────────────────────────────────────────────────
    vllm_url = "http://127.0.0.1:8001"
    running_model = _probe_server_model(vllm_url)
    console.print(f"[dim]Model server : {running_model}[/dim]")

    if args.model:
        requested = _resolve_model(args.model)
        if running_model != requested and running_model != "(unknown)":
            console.print(
                f"\n[yellow]⚠  Requested model [bold]{requested}[/bold] "
                f"but server is running [bold]{running_model}[/bold].[/yellow]"
            )
            console.print("  Restart the server with the correct model:")
            console.print(f"  [bold]  AGENT_MODEL={requested} bash scripts/start.sh[/bold]")
            console.print("  or:")
            console.print(f"  [bold]  bash scripts/serve_model.sh --model {requested}[/bold]")
            console.print("\n  Continuing anyway — results will reflect the currently loaded model.\n")
        else:
            console.print(f"[green]✓ Model match: {requested}[/green]")

    if args.check:
        return

    if not args.benchmark:
        parser.error("--benchmark is required (unless using --check)")

    # Tag results_dir with model slug so runs from different models don't collide
    results_dir = args.results_dir
    if args.model:
        model_slug = _resolve_model(args.model).replace("/", "_")
        results_dir = f"{args.results_dir}/{model_slug}"

    common = dict(
        api_url=args.api_url,
        execute=not args.no_execute,
        max_problems=args.max_problems,
        results_dir=results_dir,
        concurrency=args.concurrency,
    )

    if args.benchmark == "devtest":
        from eval.benchmarks.devtest import DevTestEvaluator
        evaluator = DevTestEvaluator(**common)

    elif args.benchmark == "ds1000":
        libs = args.libraries.split(",") if args.libraries else None
        if args.no_agent:
            from eval.benchmarks.ds1000 import DS1000OneShotEvaluator
            evaluator = DS1000OneShotEvaluator(
                libraries=libs,
                max_per_library=args.max_per_library,
                **common,
            )
        else:
            from eval.benchmarks.ds1000 import DS1000Evaluator
            evaluator = DS1000Evaluator(
                libraries=libs,
                max_per_library=args.max_per_library,
                **common,
            )

    elif args.benchmark == "dsbench":
        if not args.data_dir:
            console.print("[red]--data-dir is required for dsbench (use dsbench_local for smoke tests)[/red]")
            sys.exit(1)
        from eval.benchmarks.dsbench import DSBenchEvaluator
        evaluator = DSBenchEvaluator(data_dir=args.data_dir, track=args.track, **common)

    elif args.benchmark == "dsbench_local":
        from eval.benchmarks.dsbench import DSBenchLocalEvaluator
        evaluator = DSBenchLocalEvaluator(data_root=args.data_dir or "data", **common)

    elif args.benchmark == "dscodebench":
        libs = args.libraries.split(",") if args.libraries else None
        # --data-file is preferred; fall back to --data-dir for convenience.
        data_file = args.data_file or args.data_dir
        dsc_common = dict(
            data_file=data_file,
            libraries=libs,
            test_case_number=args.test_cases,
            max_per_library=args.max_per_library,
            **common,
        )
        if args.no_agent:
            from eval.benchmarks.dscodebench import DSCodeBenchOneShotEvaluator
            evaluator = DSCodeBenchOneShotEvaluator(**dsc_common)
        else:
            from eval.benchmarks.dscodebench import DSCodeBenchEvaluator
            evaluator = DSCodeBenchEvaluator(**dsc_common)

    elif args.benchmark == "dabstep":
        submit = getattr(args, "submit", False)
        split = "default" if submit else getattr(args, "split", "dev")

        if args.no_agent:
            # One-shot baseline: a single direct model call, no agent loop.
            from eval.benchmarks.dabstep import DABStepOneShotEvaluator
            evaluator = DABStepOneShotEvaluator(
                split=split,
                difficulty=args.difficulty,
                data_dir=args.data_dir,
                **common,
            )
        else:
            from eval.benchmarks.dabstep import DABStepEvaluator
            evaluator = DABStepEvaluator(
                split=split,
                difficulty=args.difficulty,
                data_dir=args.data_dir,
                **common,
            )

        if submit:
            # Leaderboard submission path — produces runs/{RUN_ID}.jsonl
            console.print("[bold cyan]Running full DABStep benchmark for leaderboard submission…[/bold cyan]")
            console.print("[dim]This runs all 450 tasks and may take a long time.[/dim]")
            out_path = evaluator.run_submission(runs_dir="runs")
            console.print(f"\n[green]✓ Submission file written → {out_path}[/green]")
            console.print("[bold]Upload to:[/bold] https://huggingface.co/spaces/adyen/DABstep")
            return

    with console.status(f"[bold cyan]Running {args.benchmark}…[/bold cyan]"):
        br = evaluator.run()

    _print_result(br)


if __name__ == "__main__":
    main()
