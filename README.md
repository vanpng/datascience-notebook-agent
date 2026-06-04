# DS Notebook Agent

A LangGraph agentic pipeline that helps data scientists write notebook code through natural language. The agent understands your current notebook state — loaded DataFrames, prior cell outputs, imported libraries — and generates, executes, and self-debugs Python cells in response to plain-English queries.

Supports **Qwen3-4B** (base or QLoRA fine-tuned) served via **mlx-lm** on Apple Silicon or **vllm** on Linux/CUDA.

---

## How it works

### Agent graph

Every query passes through a fixed five-node LangGraph pipeline:

```
User query
    │
    ▼
build_context   ← scans prior cells for imports/variables/data sources;
    │             calls LLM to classify intent + suggest libraries
    ▼
plan            ← LLM produces a structured AgentPlan (ordered steps,
    │             variables needed/produced)
    ▼
generate        ← LLM writes a complete Python cell; only adds imports
    │             for libraries not already in the session
    ▼
execute         ← subprocess sandbox replays all prior cells first,
    │             then runs the new cell (60 s / 2 GB cap)
    │
  success? ──yes──► append cell to session history → return to caller
    │
    no (≤ 3 retries)
    │
    ▼
debug           ← LLM reads the traceback + notebook context,
    │             diagnoses root cause, rewrites the cell
    └──────────────► execute (retry loop)
```

**`execute: false` (dry-run mode)** — the graph short-circuits after `generate` and returns the code without running it. Used by the Jupyter magic so the user executes the cell themselves in their kernel, with full access to all their variables.

### Session state

Each session keeps a list of `NotebookCell` objects — source, stdout, stderr, success flag. Every subsequent query:

- Replays prior successful cells in the sandbox so variables are in scope
- Includes prior cell outputs (stdout / expression reprs) in the LLM prompt so the model knows exact column names, shapes, and dtypes
- Skips re-importing libraries already present in the session

---

## User experience

### Option A — Jupyter notebook (recommended)

Load the magic extension once per kernel session, then type queries inline:

```python
# Cell 1 — load extension
%load_ext src.agent.magic
```

```python
# Cell 2 — your normal analysis code
import pandas as pd
df = pd.read_csv('../data/churn.csv')
print(df.shape)
print(df.dtypes)
```

```python
# Cell 3 — ask the agent
%agent plot churn rate by contract type as a bar chart
```

The agent sees Cell 2's source *and* its printed output (column names, dtypes), generates a cell, and inserts it directly below as an editable cell ready to run. You stay in control — review the code, tweak it if needed, then execute it in your kernel where `df` is already defined.

```python
# Additional magic commands
%agent_reset    # start a fresh agent session (kernel namespace unchanged)
%agent_status   # show current session ID and cells synced
```

**`notebooks/churn_analysis.ipynb`** — pre-built demo notebook using `churn.csv`. Run cells top-to-bottom then experiment with `%agent` queries.

**`notebooks/agent_interactive.ipynb`** — lower-level helper using plain `ask()` / `seed()` functions if you prefer not to use the magic extension.

### Option B — Terminal UI

```bash
bash scripts/start.sh
```

An interactive REPL with syntax-highlighted output. The agent executes generated code in a sandbox and shows stdout inline.

```
Query: my data is in data/aita_results.csv, load it and explore author_gender counts

✓ Done
╭── Generated Cell ──────────────────────────────────────────────╮
│  import pandas as pd                                           │
│  df = pd.read_csv('data/aita_results.csv')                    │
│  print(df['author_gender'].value_counts())                     │
╰────────────────────────────────────────────────────────────────╯
╭── stdout ──────────────────────────────────────────────────────╮
│  male                  33                                      │
│  female                31                                      │
│  unknown/non-binary    16                                      │
╰────────────────────────────────────────────────────────────────╯

Query: now show that as a bar chart
✓ Done   ← agent knows df and author_gender are already in scope
```

Terminal commands: `:history`, `:context`, `:clear`, `:quit`.

---

## Quick start

### 1. Install

```bash
# requires Python 3.10+ and uv
pip install uv
cd ds-notebook-agent
uv sync           # core deps
uv sync --extra dev   # adds pytest, jupyterlab, ruff, mypy
```

### 2. Configure

```bash
cp .env.example .env
# defaults work out of the box — edit only if using non-standard ports
```

Key settings in `.env`:

| Variable | Default | Description |
|---|---|---|
| `MLX_MODEL` | `mlx-community/Qwen3-4B-8bit` | Model for Apple Silicon |
| `VLLM_MODEL` | `Qwen/Qwen3-4B` | Model for Linux/CUDA |
| `VLLM_BASE_URL` | `http://localhost:8001/v1` | Model server URL |
| `AGENT_API_PORT` | `8000` | FastAPI server port |
| `SANDBOX_TIMEOUT` | `60` | Max seconds per cell execution |
| `MAX_DEBUG_ATTEMPTS` | `3` | Debug retries before giving up |
| `LORA_ADAPTER_PATH` | _(empty)_ | Path to fine-tuned LoRA adapter |

### 3. Start everything (terminal UI) (ARCHIVED - use Jupyter UI instead)

```bash
bash scripts/start.sh
```

Starts in order: model server → FastAPI agent API → terminal UI, and **waits for each to be ready** before starting the next (phase 1 = HTTP port bound, phase 2 = a real inference response, i.e. weights loaded). Ctrl-C shuts everything down cleanly. This is the easiest path — use it unless you need Jupyter.

### 4. Start for Jupyter (from scratch)

Three processes must be running: the **model server** (:8001), the **agent API** (:8000), and **JupyterLab**. Start them in this order, in three separate terminals, and wait for each to be ready before moving on — the API forwards every request to the model server, so if :8001 isn't up yet you'll get `Connection error.` in the notebook.

```bash
# ── Terminal 1 — model server (:8001) ───────────────────────────────
bash scripts/serve_model.sh
# Wait until it prints "Starting mlx_lm server on port 8001…" and stops logging.
# The weights load lazily on the FIRST request, so the very first %agent
# call (or the warm-up below) takes ~10–20 s — that's normal.
```

```bash
# ── Terminal 2 — agent API (:8000) ──────────────────────────────────
uv run uvicorn src.api.main:app --port 8000
# Wait for "Application startup complete."
```

```bash
# ── Terminal 3 — verify, then launch JupyterLab ─────────────────────
# (optional) confirm both servers are reachable before opening the notebook:
curl -s http://127.0.0.1:8001/v1/models | head -c 200   # model server
curl -s http://127.0.0.1:8000/health                     # → {"status":"ok"}

uv run jupyter lab notebooks/
```

> **Use the fine-tuned model instead of the base model.** Set `AGENT_MODEL` on **both** the model server and the API (they must match), e.g.:
> ```bash
> FT=$(pwd)/models/jupyter-agent-qwen3-4b-8bit
> AGENT_MODEL=$FT bash scripts/serve_model.sh                       # Terminal 1
> AGENT_MODEL=$FT uv run uvicorn src.api.main:app --port 8000       # Terminal 2
> ```

**Troubleshooting `Connection error.` in the notebook** — it means the API (:8000) couldn't reach the model server (:8001). Check that :8001 is actually listening (`lsof -nP -iTCP:8001 -sTCP:LISTEN`); if `serve_model.sh` exited, restart it and wait for the startup line before retrying the `%agent` cell.

---

## Repository layout

```
ds-notebook-agent/
│
├── .env.example                 # config template — copy to .env
├── pyproject.toml               # uv-managed deps (core + train + dev extras)
│
├── scripts/
│   ├── start.sh                 # one-command startup: model server + API + terminal UI
│   │                            #   phase 1: waits for HTTP server to bind
│   │                            #   phase 2: waits for a real inference response
│   │                            #            (the only reliable "weights loaded" signal)
│   ├── serve_model.sh           # hardware-aware model server launcher
│   │                            #   macOS arm64  → mlx_lm.server (8-bit quantised)
│   │                            #   Linux CUDA   → vllm serve
│   │                            #   Linux ROCm   → vllm serve
│   ├── run_***_eval.sh          # script to run evaluations
│
├── src/
│   ├── agent/
│   │   ├── state.py             # AgentState TypedDict — the single object passed
│   │   │                        #   between all LangGraph nodes
│   │   ├── schemas.py           # Pydantic I/O schemas:
│   │   │                        #   QueryIntent, SessionContext, AgentPlan,
│   │   │                        #   GeneratedCell, DebugPatch, NotebookCell,
│   │   │                        #   ExecutionResult, DataFrameSchema
│   │   ├── prompts.py           # system prompts + few-shot examples for each node;
│   │   │                        #   build_notebook_context() renders session state
│   │   │                        #   (prior cells, stdout, schemas) into a prompt string
│   │   ├── nodes.py             # async node functions:
│   │   │                        #   build_context_node — regex scan + LLM intent inference
│   │   │                        #   plan_node          — structured AgentPlan from LLM
│   │   │                        #   generate_node      — Python cell from plan + context
│   │   │                        #   execute_node       — runs sandbox, appends cell on success
│   │   │                        #   debug_node         — diagnoses stderr, rewrites cell
│   │   ├── graph.py             # compiled LangGraph graph; routing logic:
│   │   │                        #   dry_run=True  → build_context→plan→generate→END
│   │   │                        #   dry_run=False → ...→execute↔debug→END/failed
│   │   ├── sandbox.py           # code execution backend:
│   │   │                        #   replays all prior successful cells (stdout suppressed)
│   │   │                        #   then runs the new cell so variables are in scope;
│   │   │                        #   local: subprocess with memory limit
│   │   │                        #   cluster: Apptainer container
│   │   └── magic.py             # IPython/Jupyter magic extension (%agent, %agent_reset,
│   │                            #   %agent_status); registers pre/post_run_cell hooks
│   │                            #   to capture cell stdout and sync it to the session
│   ├── api/
│   │   ├── main.py              # FastAPI server; in-memory session store;
│   │   │                        #   POST /sessions/{id}/query accepts execute:bool
│   │   │                        #   (false → dry_run, inserts code without sandbox)
│   │   └── models.py            # API request/response Pydantic models
│   └── ui/
│       └── terminal.py          # Rich terminal UI; REPL with syntax highlighting,
│                                #   :history / :context / :clear commands
│
├── notebooks/
│   ├── churn_analysis.ipynb     # demo notebook: loads churn.csv, uses %agent
│   └── agent_interactive.ipynb  # helper notebook with ask() / seed() functions
│
├── data/
│   ├── churn.csv                # 300-row synthetic customer churn dataset
│   │                            #   (age, tenure, contract_type, monthly_charges, …)
│   └── aita_results.csv         # AITA Reddit post dataset with author_gender, verdict
│
├── examples/
│   └── churn_analysis.py        # CLI script: seeds context cells, runs 6 queries
│
├── sandbox/
│   └── Apptainer.def            # frozen Python 3.10 container definition (cluster)
│
├── fine_tuning/
│   ├── prepare_data.py          # formats jupyter-agent-dataset for QLoRA training
│   ├── train.py                 # SFTTrainer with QLoRA (4-bit, rank-16)
│   └── config.yaml              # hyperparameters: lr, batch size, epochs, etc.
│
└── tests/
    ├── test_agent.py            # node unit tests (mock LLM) + graph integration test
    ├── test_sandbox_state.py    # sandbox prior-cell replay + dry_run mode
    ├── test_cell_output_context.py  # stdout capture hooks, context rendering,
    │                            #   generate_node receives prior cell outputs
    └── test_import_dedup.py     # agent skips re-importing already-imported libraries
```

---

## FastAPI endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/sessions` | Create a new notebook session |
| `GET` | `/sessions/{id}` | Get session state (cells, schemas) |
| `DELETE` | `/sessions/{id}` | Delete session |
| `POST` | `/sessions/{id}/cells` | Push an already-executed cell into context |
| `POST` | `/sessions/{id}/query` | Run the agent loop for one query |

`POST /sessions/{id}/query` body:

```json
{
  "query": "plot churn rate by contract type",
  "execute": false
}
```

`execute: true` (default) runs the full generate → sandbox loop and returns stdout.  
`execute: false` returns generated code immediately without running it — used by the Jupyter magic.

Response:

```json
{
  "session_id": "...",
  "status": "done",
  "final_code": "import matplotlib.pyplot as plt\n...",
  "stdout": "...",
  "debug_attempts": 0
}
```

---

## Tests

```bash
uv run --extra dev python -m pytest -v
```

43 tests across four files covering: sandbox isolation, prior-cell state replay, dry-run routing, stdout capture hooks, context rendering, import deduplication, and end-to-end graph execution with mocked LLM.

---

## Fine-tuning

Trains a QLoRA adapter on the `jupyter-agent-dataset` to make the base Qwen3-4B model more reliably emit well-structured JSON responses.

```bash
# 1. Prepare dataset
uv run --extra train python fine_tuning/prepare_data.py

# 2. Train  (requires Linux + CUDA GPU)
uv run --extra train python fine_tuning/train.py
```

The adapter is saved to `checkpoints/qwen3-4b-ds-agent/final/`. Set `LORA_ADAPTER_PATH` in `.env` and `serve_model.sh` will load it automatically via `--lora-modules`.

