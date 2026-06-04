#!/usr/bin/env bash
# start.sh — bring up vllm model server, FastAPI agent, and terminal UI
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$ROOT/.env"

# ── load .env ────────────────────────────────────────────────────────────────
if [[ -f "$ENV_FILE" ]]; then
    set -o allexport
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +o allexport
else
    echo "⚠  No .env found — copy .env.example to .env and configure it first."
    exit 1
fi

VLLM_PORT="${VLLM_PORT:-8001}"
AGENT_PORT="${AGENT_API_PORT:-8000}"
AGENT_HOST="${AGENT_API_HOST:-127.0.0.1}"

cleanup() {
    echo ""
    echo "▶ Shutting down…"
    [[ -n "${VLLM_PID:-}" ]] && kill "$VLLM_PID" 2>/dev/null || true
    [[ -n "${API_PID:-}" ]]  && kill "$API_PID"  2>/dev/null || true
    wait 2>/dev/null || true
    echo "  Done."
}
trap cleanup EXIT INT TERM

# ── 1. vllm model server ─────────────────────────────────────────────────────
echo "▶ [1/3] Starting vllm model server…"
bash "$SCRIPT_DIR/serve_model.sh" > "$ROOT/logs/vllm.log" 2>&1 &
VLLM_PID=$!
echo "  PID $VLLM_PID  (logs → logs/vllm.log)"

# Phase 1: wait for the HTTP server to bind and report its model ID.
# /v1/models is pre-populated on mlx_lm before weights finish loading, but it
# at least confirms the process started and gives us the model ID we need.
echo "  Waiting for model server on port ${VLLM_PORT} to start…"
PROBE_MODEL=""
for i in $(seq 1 30); do
    PROBE_MODEL="$(curl -sf "http://localhost:${VLLM_PORT}/v1/models" 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null \
        || echo '')"
    [[ -n "$PROBE_MODEL" ]] && break
    sleep 2
    if [[ $i -eq 30 ]]; then
        echo "  ✗ model server process did not start. Check logs/vllm.log"
        exit 1
    fi
done
echo "  Server up, probing model: $PROBE_MODEL"

# Phase 2: send a minimal inference request and wait for it to succeed.
# This is the only reliable signal that weights are fully loaded — /health and
# /v1/models both return 200 on mlx_lm while the model is still downloading.
echo "  Waiting for model to finish loading (first run downloads from HuggingFace)…"
MAX_WAIT=1800  # 30 min ceiling
START_TS=$SECONDS
LAST_REPORT=0
while true; do
    elapsed=$(( SECONDS - START_TS ))
    remaining=$(( MAX_WAIT - elapsed ))
    if [[ $remaining -le 0 ]]; then
        echo "  ✗ model not ready after ${MAX_WAIT}s. Check logs/vllm.log"
        exit 1
    fi
    if curl -sf --max-time "$remaining" \
            "http://localhost:${VLLM_PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$PROBE_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
            2>/dev/null | grep -q '"object"'; then
        echo "  ✓ model ready (${elapsed}s)"
        break
    fi
    elapsed=$(( SECONDS - START_TS ))
    if (( elapsed - LAST_REPORT >= 60 )); then
        echo "  still waiting ${elapsed}s… (check logs/vllm.log for download progress)"
        LAST_REPORT=$elapsed
    fi
    sleep 5
done

# ── 2. FastAPI agent server ───────────────────────────────────────────────────
echo "▶ [2/3] Starting FastAPI agent server…"
mkdir -p "$ROOT/logs"
cd "$ROOT"
uv run uvicorn src.api.main:app \
    --host "$AGENT_HOST" \
    --port "$AGENT_PORT" \
    --log-level info \
    > "$ROOT/logs/api.log" 2>&1 &
API_PID=$!
echo "  PID $API_PID  (logs → logs/api.log)"

echo "  Waiting for agent API on port ${AGENT_PORT}…"
for i in $(seq 1 20); do
    if curl -sf "http://$AGENT_HOST:$AGENT_PORT/health" > /dev/null 2>&1; then
        echo "  ✓ agent API ready at http://$AGENT_HOST:$AGENT_PORT"
        break
    fi
    sleep 1
    if [[ $i -eq 20 ]]; then
        echo "  ✗ agent API failed to start. Check logs/api.log"
        exit 1
    fi
done

# ── 3. Terminal UI ────────────────────────────────────────────────────────────
echo "▶ [3/3] Launching terminal UI…"
echo ""
uv run python -m src.ui.terminal \
    --api-url "http://$AGENT_HOST:$AGENT_PORT"
