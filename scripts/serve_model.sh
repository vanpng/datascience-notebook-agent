#!/usr/bin/env bash
# serve_model.sh — detect hardware and start the right inference backend
#
# Usage:
#   bash scripts/serve_model.sh                  # use defaults from .env
#   bash scripts/serve_model.sh --model <id>      # override model (any HF model ID)
#   AGENT_MODEL=<id> bash scripts/serve_model.sh  # same via env var
#
# Model override priority: --model flag > AGENT_MODEL env > MLX_MODEL/VLLM_MODEL > built-in default
set -euo pipefail

PORT="${VLLM_PORT:-8001}"

# ── parse --model flag ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            AGENT_MODEL="$2"
            shift 2
            ;;
        --model=*)
            AGENT_MODEL="${1#--model=}"
            shift
            ;;
        *)
            shift
            ;;
    esac
done
export AGENT_MODEL="${AGENT_MODEL:-}"

# ── hardware detection ────────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"
BACKEND=""
HW_DESC=""

if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    BACKEND="mlx"
    HW_DESC="$(sysctl -n hw.model 2>/dev/null || echo 'Apple Silicon')"

elif [[ "$OS" == "Linux" ]]; then
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
        BACKEND="vllm"
        HW_DESC="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
                   | head -1 \
                   | sed 's/,/ —/' \
                   || echo 'NVIDIA GPU')"
    elif command -v rocm-smi &>/dev/null && rocm-smi --showproductname &>/dev/null 2>&1; then
        BACKEND="vllm"
        HW_DESC="$(rocm-smi --showproductname 2>/dev/null \
                   | grep -i 'card\|gpu' | head -1 | tr -s ' ' \
                   || echo 'AMD GPU (ROCm)')"
    else
        echo "✗ No supported GPU found on Linux."
        echo "  Install nvidia-smi (CUDA) or rocm-smi (ROCm) and retry."
        exit 1
    fi

elif [[ "$OS" == "Darwin" && "$ARCH" == "x86_64" ]]; then
    echo "✗ macOS on Intel is not supported."
    echo "  Use a Linux+CUDA/ROCm machine or an Apple Silicon Mac."
    exit 1

else
    echo "✗ Unsupported platform: $OS $ARCH"
    exit 1
fi

echo "▶ Hardware : $HW_DESC"
echo "▶ Backend  : $BACKEND"

# ── launch ────────────────────────────────────────────────────────────────────
if [[ "$BACKEND" == "mlx" ]]; then
    # AGENT_MODEL overrides MLX_MODEL; fall back to built-in default
    MODEL="${AGENT_MODEL:-${MLX_MODEL:-mlx-community/Qwen3-4B-8bit}}"
    echo "▶ Model    : $MODEL"
    echo "▶ Starting mlx_lm server on port ${PORT}…"
    exec uv run mlx_lm.server --model "$MODEL" --port "$PORT"

elif [[ "$BACKEND" == "vllm" ]]; then
    # Load HPC modules and activate conda env with working CUDA/vLLM
    module purge
    module load gcc/11.5.0 cuda/12.4.0 conda 2>/dev/null || true
    CONDA_ENV="/gpfs/projects/imt526a/vanpham/env"
    # shellcheck disable=SC1091
    source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate "$CONDA_ENV" 2>/dev/null || true
    export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
    # Use Xformers backend to bypass FlashInfer JIT compilation
    export VLLM_ATTENTION_BACKEND=XFORMERS
    export HF_HOME="${HF_HOME:-/gpfs/projects/imt526a/group3/ds-notebook-agent/.cache/huggingface}"

    # AGENT_MODEL is for client requests; base model is always VLLM_MODEL
    MODEL="${VLLM_MODEL:-Qwen/Qwen3-4B}"
    GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
    MAX_LEN="${VLLM_MAX_LEN:-32768}"
    ADAPTER="${LORA_ADAPTER_PATH:-}"

    echo "▶ Model    : $MODEL"
    echo "▶ Starting vllm server on port ${PORT}…"

    ARGS=(
        serve "$MODEL"
        --port "$PORT"
        --max-model-len "$MAX_LEN"
        --gpu-memory-utilization "$GPU_UTIL"
        --enable-prefix-caching
        --trust-remote-code
    )

    if [[ -n "$ADAPTER" && -d "$ADAPTER" ]]; then
        echo "  ↳ LoRA adapter: $ADAPTER"
        ARGS+=(--lora-modules "ds-agent=$ADAPTER" --enable-lora)
    fi

    exec vllm "${ARGS[@]}"
fi
