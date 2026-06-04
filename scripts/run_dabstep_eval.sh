#!/usr/bin/env bash
# run_dabstep_eval.sh — end-to-end evaluation runner for DABStep
#
# Starts the vLLM model server and FastAPI agent, resolves (downloading if
# needed) the DABStep context data directory, then runs BOTH the agent-workflow
# evaluator (DABStepEvaluator) and the one-shot baseline (DABStepOneShotEvaluator)
# against the DABStep "dev" split, then tears everything down.
#
# Usage:
#   bash scripts/run_dabstep_eval.sh [OPTIONS]
#
# Options:
#   --model        HuggingFace model ID or alias (base-linux | finetuned)
#                  Default: keeljimin/imt526-project-fine-tuning-checkpoint
#   --lora         Path to a local LoRA adapter directory (overrides .env)
#   --concurrency  Problems evaluated in parallel (default: 10)
#   --data-dir     DABStep context dir (auto-downloaded from HF if omitted)
#   --split        dev (10 tasks, ground truth) | default (450 tasks)  [default: dev]
#   --difficulty   easy | medium | hard  (filter; default: all)
#   --max          Max problems per run (smoke-test mode)
#   --no-agent     Skip the agent-workflow eval
#   --no-oneshot   Skip the one-shot baseline
#   --results-dir  Output directory for JSON results (default: eval/results)
#
# Example — full finetuned DABStep dev eval (agent + one-shot):
#   bash scripts/run_dabstep_eval.sh
#
# Example — quick smoke test on 3 easy tasks:
#   bash scripts/run_dabstep_eval.sh --max 3 --difficulty easy

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

# ── defaults ──────────────────────────────────────────────────────────────────
MODEL="keeljimin/imt526-project-fine-tuning-checkpoint"
LORA_OVERRIDE=""
CONCURRENCY=10
DATA_DIR=""
SPLIT="dev"
DIFFICULTY=""
MAX_PROBLEMS=""
RUN_AGENT=true
RUN_ONESHOT=true
RESULTS_DIR="eval/results"

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2";         shift 2 ;;
        --lora)        LORA_OVERRIDE="$2";  shift 2 ;;
        --concurrency) CONCURRENCY="$2";   shift 2 ;;
        --data-dir)    DATA_DIR="$2";      shift 2 ;;
        --split)       SPLIT="$2";         shift 2 ;;
        --difficulty)  DIFFICULTY="$2";    shift 2 ;;
        --max)         MAX_PROBLEMS="$2";  shift 2 ;;
        --no-agent)    RUN_AGENT=false;    shift ;;
        --no-oneshot)  RUN_ONESHOT=false;  shift ;;
        --results-dir) RESULTS_DIR="$2";   shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── environment setup ─────────────────────────────────────────────────────────
# Keep all caches and packages inside the project directory (avoid home quota)
export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$ROOT/.venv"
export HF_HOME="$ROOT/.cache/huggingface"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$PIP_CACHE_DIR" "$ROOT/logs"

# Load .env
if [[ -f "$ROOT/.env" ]]; then
    set -o allexport
    source "$ROOT/.env"
    set +o allexport
fi

# Apply LoRA override if provided
[[ -n "$LORA_OVERRIDE" ]] && export LORA_ADAPTER_PATH="$LORA_OVERRIDE"

# AGENT_MODEL is the model ID used in API requests — the LoRA module name when
# a LoRA adapter is loaded, otherwise the base model ID.
if [[ -n "${LORA_ADAPTER_PATH:-}" && -d "${LORA_ADAPTER_PATH:-}" ]]; then
    export AGENT_MODEL="ds-agent"
else
    export AGENT_MODEL="${VLLM_MODEL:-Qwen/Qwen3-4B}"
fi

# uv must be on PATH
export PATH="$HOME/.local/bin:$PATH"

VLLM_PORT="${VLLM_PORT:-8001}"
AGENT_PORT="${AGENT_API_PORT:-8000}"
AGENT_HOST="${AGENT_API_HOST:-127.0.0.1}"
VLLM_PID=""
API_PID=""

# ── resolve DABStep context data dir (download if needed) ───────────────────────
if [[ -z "$DATA_DIR" ]]; then
    echo "▶ Resolving DABStep context data dir (downloading if needed)…"
    DATA_DIR=$(uv run python - <<'PY'
import os
from huggingface_hub import snapshot_download
# Only pull the context files (CSV/JSON/markdown), not the full dataset.
path = snapshot_download(
    "adyen/DABstep",
    repo_type="dataset",
    allow_patterns=["data/context/*"],
)
ctx = os.path.join(path, "data", "context")
print(ctx if os.path.isdir(ctx) else path)
PY
)
    echo "  data dir → $DATA_DIR"
fi

if [[ ! -d "$DATA_DIR" ]]; then
    echo "  ✗ DABStep data dir not found: $DATA_DIR"
    echo "    Pass one explicitly with --data-dir, or check network access to HF."
    exit 1
fi

# ── cleanup on exit ───────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "▶ Shutting down services…"
    [[ -n "$VLLM_PID" ]] && kill "$VLLM_PID" 2>/dev/null || true
    [[ -n "$API_PID"  ]] && kill "$API_PID"  2>/dev/null || true
    wait 2>/dev/null || true
    echo "  Done."
}
trap cleanup EXIT INT TERM

# ── 1. start vLLM model server ────────────────────────────────────────────────
echo ""
echo "▶ [1/4] Starting vLLM model server…"
bash "$SCRIPT_DIR/serve_model.sh" > "$ROOT/logs/vllm.log" 2>&1 &
VLLM_PID=$!
echo "  PID $VLLM_PID  (logs → logs/vllm.log)"

echo "  Waiting for model server on port ${VLLM_PORT}…"
START_TS=$SECONDS
while true; do
    MODELS=$(curl -sf "http://localhost:${VLLM_PORT}/v1/models" 2>/dev/null \
        | python3 -c "import sys,json; print([m['id'] for m in json.load(sys.stdin)['data']])" 2>/dev/null || true)
    if [[ -n "$MODELS" ]]; then
        echo "  Models endpoint up: $MODELS"
        break
    fi
    elapsed=$(( SECONDS - START_TS ))
    if (( elapsed > 1800 )); then
        echo "  ✗ Model server did not start after 30 min. Check logs/vllm.log"
        exit 1
    fi
    (( elapsed % 60 == 0 && elapsed > 0 )) && echo "  still waiting ${elapsed}s…"
    sleep 10
done

# Wait for an actual inference request to succeed (model fully loaded)
echo "  Waiting for model weights to finish loading…"
START_TS=$SECONDS
PROBE_MODEL=$(curl -sf "http://localhost:${VLLM_PORT}/v1/models" 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "")
while true; do
    RESP=$(curl -sf --max-time 30 \
        "http://localhost:${VLLM_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${PROBE_MODEL:-Qwen/Qwen3-4B}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
        2>/dev/null || true)
    if echo "$RESP" | grep -q '"object"'; then
        echo "  ✓ Model ready ($(( SECONDS - START_TS ))s)"
        break
    fi
    elapsed=$(( SECONDS - START_TS ))
    if (( elapsed > 1800 )); then
        echo "  ✗ Model not ready after 30 min. Check logs/vllm.log"
        exit 1
    fi
    (( elapsed % 60 == 0 && elapsed > 0 )) && echo "  still loading ${elapsed}s…"
    sleep 10
done

# ── 2. start FastAPI agent server ─────────────────────────────────────────────
echo ""
echo "▶ [2/4] Starting FastAPI agent server…"
uv run uvicorn src.api.main:app \
    --host "$AGENT_HOST" \
    --port "$AGENT_PORT" \
    --log-level info \
    > "$ROOT/logs/api.log" 2>&1 &
API_PID=$!
echo "  PID $API_PID  (logs → logs/api.log)"

echo "  Waiting for agent API on port ${AGENT_PORT}…"
for i in $(seq 1 30); do
    if curl -sf "http://$AGENT_HOST:$AGENT_PORT/health" > /dev/null 2>&1; then
        echo "  ✓ Agent API ready at http://$AGENT_HOST:$AGENT_PORT"
        break
    fi
    sleep 2
    if [[ $i -eq 30 ]]; then
        echo "  ✗ Agent API failed to start. Check logs/api.log"
        exit 1
    fi
done

# ── build shared eval args ────────────────────────────────────────────────────
EVAL_ARGS=(
    "--benchmark"   "dabstep"
    "--model"       "$MODEL"
    "--data-dir"    "$DATA_DIR"
    "--split"       "$SPLIT"
    "--concurrency" "$CONCURRENCY"
    "--results-dir" "$RESULTS_DIR"
)
[[ -n "$DIFFICULTY"   ]] && EVAL_ARGS+=("--difficulty" "$DIFFICULTY")
[[ -n "$MAX_PROBLEMS" ]] && EVAL_ARGS+=("--max"        "$MAX_PROBLEMS")

# ── 3. run evaluations ────────────────────────────────────────────────────────
echo ""
echo "▶ [3/4] Launching DABStep evaluations (split=${SPLIT}, concurrency=${CONCURRENCY})…"

AGENT_EVAL_PID=""
ONESHOT_EVAL_PID=""

if $RUN_AGENT; then
    echo "  Starting agent-workflow eval…  (logs → logs/eval_dabstep_agent.log)"
    uv run python eval/run_eval.py "${EVAL_ARGS[@]}" \
        > "$ROOT/logs/eval_dabstep_agent.log" 2>&1 &
    AGENT_EVAL_PID=$!
    echo "  PID $AGENT_EVAL_PID"
fi

if $RUN_ONESHOT; then
    echo "  Starting one-shot baseline eval…  (logs → logs/eval_dabstep_oneshot.log)"
    uv run python eval/run_eval.py "${EVAL_ARGS[@]}" --no-agent \
        > "$ROOT/logs/eval_dabstep_oneshot.log" 2>&1 &
    ONESHOT_EVAL_PID=$!
    echo "  PID $ONESHOT_EVAL_PID"
fi

# ── 4. wait and report ────────────────────────────────────────────────────────
echo ""
echo "▶ [4/4] Waiting for evaluations to complete…"
echo "  Monitor progress:"
$RUN_AGENT   && echo "    tail -f logs/eval_dabstep_agent.log"
$RUN_ONESHOT && echo "    tail -f logs/eval_dabstep_oneshot.log"
echo ""

EXIT_CODE=0
if [[ -n "$AGENT_EVAL_PID" ]]; then
    wait "$AGENT_EVAL_PID" || { echo "  ✗ Agent eval exited with error"; EXIT_CODE=1; }
    echo "  ✓ Agent eval complete  →  $(ls -t "$RESULTS_DIR"/*/dabstep_2*.json "$RESULTS_DIR"/dabstep_2*.json 2>/dev/null | head -1)"
fi
if [[ -n "$ONESHOT_EVAL_PID" ]]; then
    wait "$ONESHOT_EVAL_PID" || { echo "  ✗ Oneshot eval exited with error"; EXIT_CODE=1; }
    echo "  ✓ Oneshot eval complete  →  $(ls -t "$RESULTS_DIR"/*/dabstep_oneshot_*.json "$RESULTS_DIR"/dabstep_oneshot_*.json 2>/dev/null | head -1)"
fi

echo ""
echo "▶ All done. Results in $RESULTS_DIR/"
exit "$EXIT_CODE"
