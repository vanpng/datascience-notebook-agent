#!/usr/bin/env bash
# run_leakage_check.sh — check DS-1000 / DABStep contamination in the fine-tuning data.
#
# Scans jupyter-agent/jupyter-agent-dataset (the fine-tuning data source) and
# measures n-gram containment of every DS-1000 problem statement, DS-1000
# reference solution, and DABStep question against the row content (chat messages
# + tool-call code + question/answer). Pure data analysis — NO model server / API.
#
# Usage:
#   bash scripts/run_leakage_check.sh [extra args passed through to check_leakage.py]
#
# Examples:
#   bash scripts/run_leakage_check.sh                          # default: non_thinking split, both benchmarks
#   bash scripts/run_leakage_check.sh --splits non_thinking,thinking --include-notebook
#   bash scripts/run_leakage_check.sh --n 13 --flag-threshold 0.6
#   bash scripts/run_leakage_check.sh --benchmarks ds1000 --include-filtered

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

# Keep all caches inside the project directory (avoid home quota)
export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$ROOT/.venv"
export HF_HOME="$ROOT/.cache/huggingface"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$PIP_CACHE_DIR" "$ROOT/logs"
export PATH="$HOME/.local/bin:$PATH"

echo "▶ Running benchmark leakage check (args: $*)"
uv run python eval/check_leakage.py "$@"
