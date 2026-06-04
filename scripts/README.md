# Scripts

## run_eval.sh — End-to-end DS-1000 Evaluation

Starts the vLLM model server and FastAPI agent, runs the **agent-workflow** and **one-shot baseline** evaluations against DS-1000 in parallel, then shuts everything down. All logs and results are written inside the project directory (never to home).

### Prerequisites

- NVIDIA GPU on the Tillicum HPC cluster
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Conda environment at `/gpfs/projects/imt526a/vanpham/env` (contains vLLM 0.19.1 + CUDA 12.4)
- Project dependencies synced:
  ```bash
  UV_CACHE_DIR=/gpfs/projects/imt526a/group3/ds-notebook-agent/.cache/uv \
  UV_PROJECT_ENVIRONMENT=/gpfs/projects/imt526a/group3/ds-notebook-agent/.venv \
  uv sync
  ```
- `.env` configured (copy `.env.example` → `.env` and fill in values)

### Quick Start

```bash
cd /gpfs/projects/imt526a/group3/ds-notebook-agent
bash scripts/run_eval.sh
```

This runs the full DS-1000 benchmark (1000 problems) using the finetuned checkpoint at `keeljimin/imt526-project-fine-tuning-checkpoint` with 16 concurrent workers. Results are written to `eval/results/`.

### Common Invocations

```bash
# Full eval with finetuned model (default)
bash scripts/run_eval.sh

# Base model only (no LoRA adapter)
bash scripts/run_eval.sh --model base-linux --lora ""

# Specific libraries only
bash scripts/run_eval.sh --libraries Pandas,Numpy

# Smoke test — 20 problems, Pandas only
bash scripts/run_eval.sh --max 20 --libraries Pandas

# Agent workflow only (skip one-shot baseline)
bash scripts/run_eval.sh --no-oneshot

# One-shot baseline only
bash scripts/run_eval.sh --no-agent

# Higher concurrency (if GPU has headroom)
bash scripts/run_eval.sh --concurrency 24

# Custom LoRA adapter
bash scripts/run_eval.sh --lora /path/to/your/checkpoint

# Write results to a custom directory
bash scripts/run_eval.sh --results-dir eval/results/my_run
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `keeljimin/imt526-project-fine-tuning-checkpoint` | HF model ID or alias (`base-linux`, `finetuned`) |
| `--lora` | from `.env` | Local path to a LoRA adapter directory |
| `--concurrency` | `16` | Problems evaluated in parallel |
| `--libraries` | all 7 | Comma-separated DS-1000 library subset |
| `--max` | all | Max problems to evaluate (smoke-test mode) |
| `--no-agent` | — | Skip agent-workflow evaluation |
| `--no-oneshot` | — | Skip one-shot baseline evaluation |
| `--results-dir` | `eval/results` | Where to write JSON result files |

### Monitoring Progress

```bash
# Agent workflow
tail -f logs/eval_agent.log

# One-shot baseline
tail -f logs/eval_oneshot.log

# Model server
tail -f logs/vllm.log

# Agent API
tail -f logs/api.log
```

### Output

Results are saved as JSON files under `eval/results/`:

```
eval/results/
└── keeljimin_imt526-project-fine-tuning-checkpoint/
    ├── ds1000_20260602_235000.json        # agent-workflow results
    └── ds1000_oneshot_20260602_235000.json  # one-shot baseline results
```

Each file contains overall pass rate, per-library breakdown, and per-problem details (generated code, test stderr, latency).

### Expected Runtime (H200 GPU, concurrency=16)

| Eval | Problems | Approx. time |
|------|----------|--------------|
| One-shot | 1000 | ~15 min |
| Agent workflow | 1000 | ~35 min |
| Both (parallel) | 1000 each | ~35–40 min total |


### Quick Start

```bash
cd /gpfs/projects/imt526a/group3/ds-notebook-agent

# Full eval (all 10 libraries, agent + one-shot)
bash scripts/run_dscodebench_eval.sh

# Smoke test — numpy + pandas, 5 problems each
bash scripts/run_dscodebench_eval.sh --libraries numpy,pandas --max-per-library 5

# Paper-faithful scoring (200 test cases per problem, like the upstream default)
bash scripts/run_dscodebench_eval.sh --test-cases 200

# One-shot baseline only
bash scripts/run_dscodebench_eval.sh --no-agent
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `keeljimin/imt526-project-fine-tuning-checkpoint` | HF model ID or alias (`base-linux`, `finetuned`) |
| `--lora` | from `.env` | Local path to a LoRA adapter directory |
| `--concurrency` | `16` | Problems evaluated in parallel |
| `--data-file` | auto | Path to `DSCodeBench.json` (downloaded to `data/` if omitted) |
| `--libraries` | all 10 | Comma-separated library subset (e.g. `numpy,pandas`) |
| `--test-cases` | `50` | Test-case inputs per problem (upstream uses `200`) |
| `--max` | all | Max problems to evaluate (smoke-test mode) |
| `--max-per-library` | — | Cap problems per library (e.g. `10` → ≤100 total) |
| `--no-agent` | — | Skip agent-workflow evaluation |
| `--no-oneshot` | — | Skip one-shot baseline evaluation |
| `--results-dir` | `eval/results` | Where to write JSON result files |

### Output

Results are saved under `eval/results/<model_slug>/`:

```
dscodebench_<timestamp>.json          # agent-workflow results
dscodebench_oneshot_<timestamp>.json  # one-shot baseline results
```

Each file contains the overall pass rate, per-library breakdown, `mean_test_case_pass_rate` (partial-credit signal), a failure-reason histogram, and per-problem details (generated code, harness stderr, latency).

> **Scoring deps:** lightgbm + scikit-image are in `pyproject.toml` for full coverage. Without lightgbm the 54 lightgbm problems score 0. Under keras ≥ 3.10 some keras problems underscore because `model.trainable_weights` are `keras.Variable` (not `tf.Variable`) — this matches upstream harness behaviour under newer keras, it is not a port bug. Run `uv sync` after pulling these changes.

---

## serve_model.sh

Starts the vLLM inference server with hardware auto-detection. On Linux, loads the required HPC modules (`gcc/11.5.0`, `cuda/12.4.0`) and activates the conda environment before launching vLLM.

```bash
# Default (reads VLLM_MODEL and LORA_ADAPTER_PATH from .env)
bash scripts/serve_model.sh

# Override model
bash scripts/serve_model.sh --model Qwen/Qwen3-4B
```

## start.sh

Interactive launcher — starts the model server, agent API, and terminal REPL UI together. Use this for interactive notebook sessions; use `run_eval.sh` for batch evaluation.

```bash
bash scripts/start.sh
```

## build_sandbox.sh

Builds the Apptainer container used for secure code execution in the sandbox.

```bash
bash scripts/build_sandbox.sh
```
