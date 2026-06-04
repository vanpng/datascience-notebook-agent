#!/usr/bin/env bash
# _devtest30_base.sh — run devtest30 SOLO on the BASE model Qwen/Qwen3-4B.
# Brings up a fresh vLLM (serving Qwen/Qwen3-4B) + FastAPI agent, runs the
# 30-scenario curated set with no other load, prints pass_rate +
# format_compliance_rate, then tears everything down.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

MODEL="Qwen/Qwen3-4B"
DEV_CONC=8

export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$ROOT/.venv"
export HF_HOME="$ROOT/.cache/huggingface"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$PIP_CACHE_DIR" "$ROOT/logs"

if [[ -f "$ROOT/.env" ]]; then
    set -o allexport; source "$ROOT/.env"; set +o allexport
fi
# Override .env: serve the BASE model, no LoRA.
export VLLM_MODEL="$MODEL"
export AGENT_MODEL="$MODEL"
export LORA_ADAPTER_PATH=""

VLLM_PORT="${VLLM_PORT:-8001}"
AGENT_PORT="${AGENT_API_PORT:-8000}"
AGENT_HOST="${AGENT_API_HOST:-127.0.0.1}"
VLLM_PID=""; API_PID=""

cleanup() {
    echo "▶ Shutting down services…"
    [[ -n "$VLLM_PID" ]] && kill "$VLLM_PID" 2>/dev/null || true
    [[ -n "$API_PID"  ]] && kill "$API_PID"  2>/dev/null || true
    wait 2>/dev/null || true
    echo "  Done."
}
trap cleanup EXIT INT TERM

# ── 1. vLLM (base model) ──────────────────────────────────────────────────────
echo "▶ [1/4] Starting vLLM model server (model: $MODEL)…"
VLLM_MODEL="$MODEL" bash "$SCRIPT_DIR/serve_model.sh" > "$ROOT/logs/vllm_base.log" 2>&1 &
VLLM_PID=$!
echo "  PID $VLLM_PID  (logs → logs/vllm_base.log)"

START_TS=$SECONDS; PROBE_MODEL=""
while true; do
    PROBE_MODEL=$(curl -sf "http://localhost:${VLLM_PORT}/v1/models" 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "")
    [[ -n "$PROBE_MODEL" ]] && { echo "  Models endpoint up: $PROBE_MODEL"; break; }
    (( SECONDS - START_TS > 1800 )) && { echo "  ✗ server start timeout"; exit 1; }
    sleep 10
done

echo "  Waiting for weights to finish loading…"
START_TS=$SECONDS
while true; do
    RESP=$(curl -sf --max-time 30 "http://localhost:${VLLM_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$PROBE_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" 2>/dev/null || true)
    echo "$RESP" | grep -q '"object"' && { echo "  ✓ Model ready ($(( SECONDS - START_TS ))s)"; break; }
    (( SECONDS - START_TS > 1800 )) && { echo "  ✗ weights load timeout"; exit 1; }
    sleep 10
done

# ── 2. FastAPI agent server ───────────────────────────────────────────────────
echo "▶ [2/4] Starting FastAPI agent server…"
uv run uvicorn src.api.main:app --host "$AGENT_HOST" --port "$AGENT_PORT" \
    --log-level info > "$ROOT/logs/api_base.log" 2>&1 &
API_PID=$!
echo "  PID $API_PID  (logs → logs/api_base.log)"
for i in $(seq 1 30); do
    curl -sf "http://$AGENT_HOST:$AGENT_PORT/health" >/dev/null 2>&1 && { echo "  ✓ Agent API ready"; break; }
    sleep 2
    [[ $i -eq 30 ]] && { echo "  ✗ API failed to start"; exit 1; }
done

# ── 3. run devtest30 solo ─────────────────────────────────────────────────────
echo "▶ [3/4] Running devtest30 SOLO on $MODEL (concurrency=$DEV_CONC)…"
uv run python eval/run_eval.py \
    --benchmark devtest --model "$MODEL" --concurrency "$DEV_CONC" \
    --results-dir eval/results > "$ROOT/logs/eval_devtest30_base.log" 2>&1
RC=$?
echo "  devtest30 finished (rc=$RC)"

# ── 4. summary ────────────────────────────────────────────────────────────────
echo "▶ [4/4] Summary…"
MODEL_SLUG="${MODEL//\//_}"
DEV_JSON="$(ls -t eval/results/${MODEL_SLUG}/devtest30_*.json 2>/dev/null | head -1)"
echo "=================== devtest30 (base Qwen/Qwen3-4B) ==================="
python3 - "$DEV_JSON" <<'PY'
import json, sys, os
f = sys.argv[1]
if f and os.path.isfile(f):
    d = json.load(open(f)); em = d.get("extra_metrics", {})
    print(f"FILE: {f}")
    print(f"total={d['total']} passed={d['passed']} failed={d['failed']} errors={d['errors']}")
    print(f"pass_rate={d['pass_rate']:.4f}  format_compliance_rate={em.get('format_compliance_rate')} "
          f"({em.get('format_compliant')}/{d['total']})")
else:
    print("✗ no devtest30 result json found")
PY
echo "====================================================================="
