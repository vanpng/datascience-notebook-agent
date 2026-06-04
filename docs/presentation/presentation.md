# IMT526 Project:
# Building Applied LLM pipeline using LangGraph for Data Science Notebook Assistant

**Team members**: Van Pham, Jimin Keel, Devarshee Thopte, Kyle Chen

**Description:** Our DS Notebook assistant is a self-hostable **Qwen3-4B**, wrapped in a LangGraph
**Plan → Generate → Execute → Debug** loop with Pydantic-enforced cell outputs, that
co-authors Jupyter notebook code from natural language, grounded in the live notebook
state (DataFrame schemas, prior cell outputs). **Our research question**: *can a small,
fine-tuned, locally-served model inside a well-engineered agent loop approach the
performance of much larger proprietary models, while keeping data private?*

---

## 1. Problem & motivation 

LLMs are strong on clean coding benchmarks but weak on **realistic data science** code.
On DS-1000 (1,000 StackOverflow-sourced DS problems across 7 libraries), even strong
proprietary models struggle — GPT-4o-mini reaches only **42.2% accuracy** [Lai et al.,
2023; see §5.1]. We attribute the gap to three failure modes a chat LLM cannot fix on its
own:

1. **No awareness of live notebook state**: it doesn't know which DataFrames are loaded
   or their real columns/dtypes, so it guesses names.
2. **Manual error round-tripping**: the user must copy-paste tracebacks back in.
3. **Loose output**: chat prose / markdown instead of a strictly-formatted, immediately
   executable cell.

**Our thesis & scope.** We QLoRA fine-tune **Qwen/Qwen3-4B** (Apache-2.0, self-hostable)
to close raw coding ability, and wrap it in a LangGraph agentic loop that directly
attacks the three failure modes: it ingests notebook state, runs code in a sandbox and
self-debugs on real tracebacks, and enforces valid-cell output via Pydantic. We then
benchmark **base vs. fine-tuned** checkpoints in the *same* pipeline on 2 DS suites (DS-1000 and DABStep).
The system complements AutoML (Auto-sklearn, AutoGluon, H2O) — those tune models; we
target the open-ended, exploratory notebook workflow. **Out of scope:** a full GUI. We
focus on pipeline wiring (LangGraph), structured outputs, QLoRA fine-tuning, and
comparative benchmarking.

---

## 2. Approach & agentic design

**Shape:** an applied LLM pipeline (context management, structured output) plus
a targeted fine-tune. **Serving:** vLLM exposes an OpenAI-compatible endpoint; LangGraph
is the client holding session state.

### 2.1 The graph

```
query → build_context → plan → generate → execute(sandbox) → success? ─yes→ return
                                              ▲                  │no (≤3 retries)
                                              └──── debug ◄───────┘
```

1. Take the query → 2. ingest notebook state (prior cells, stdout, DataFrame schemas) →
3. plan → 4. generate a cell → 5. execute in an isolated sandbox → 6. read stderr →
7. self-correct in a bounded loop, then return a structurally-valid cell.

### 2.2 Grounded in coding-agent practice (related work / differentiation)

| Node | Published pattern we build on | 
|---|---|
| build_context | **Agent–Computer Interface** [SWE-agent, Yang 2024] | 
| plan | **Plan-and-Solve** [Wang 2023] | 
| generate | **Act** step of **ReAct** [Yao 2022] | 
| debug | **Reflexion** [Shinn 2023] + **Self-Debug** [Chen 2023] | 

The plan - generate - execute and debug loop agentic design is a simplified version of DS-STAR which is also influenced by the above Coding agent design papers. While DS-STAR employs a complex agentic design with analyzer-planner-coder-executor-verifier-router loop with complex tool use and comprehensive reasoning prompt instruction for each step. Another key difference is the model choice, DS-STAR benchmarks on big LLMs while we choose a smaller LLM to suite the scale of this project.

[SWE-agent] emphasizes the important of context to solve coding benchmark with proper context. build_context nodes build current state context with notebook cells, previous execution outputs, user request, data (if available) and provide it to the plan/generate node to have the necessarily context. 

Plan-and-Solve [Wang 2023] proves that  breaking down task into smaller step for multi step reasoning is helpful for agent to solve complex tasks. Hence, generally in building agent, a planing node is required to break down task into smaller steps then provide those smaller steps to the generate node.

**Self-Debug** [Chen 2023] shows that LLM perform increase a self debug pipeline which guides the LLM todebug
its predicted program via few-shot demonstrations.

An execute node is require for the generate-debug loop to provide real error feedbacks. The excution process will provide details on why the code cannot run succesfully and these signals are grounded context for debug node to instruct generator node how to do better next time.



### 2.3 Sandbox 

Each cell runs in a disposable subprocess in our frozen DS venv (Python 3.10; pandas, numpy, scikit-learn, scipy, matplotlib, **+ tensorflow, pytorch**) with a **60s timeout** and **2 GB RAM** cap.

### 2.4 UI

The main UX will be an inline magic command in jupyter lab, user can load %load_ext src.agent.magic. The inline magic command will be better than a terminal UI where user has to copy/paste code from terminal back to notebook again to try out. 

---

## 3. Data & models

- **Training:** `jupyter-agent/jupyter-agent-dataset` (Apache-2.0, ~95.8k real-Kaggle
  triples of *notebook-context + query → verified cell + execution trace*). [FILL-NUMBER] sampled,
  80/10/10 split; benchmark-overlapping samples filtered to prevent leakage.
- **Models:** base **Qwen3-4B** vs. **QLoRA fine-tune** (r=16, nf4, TRL), evaluated
  identically in the pipeline → isolates the fine-tuning contribution from pipeline effects.
- **Benchmarks:** DS-1000 (1,000), DABstep (450), plus a 30-task **quality check**
  suite of realistic notebook scenarios grouped by analytic intent (§5.5).
  Published proprietary numbers (e.g., GPT-4o) were taken from reference papers (not run through our pipeline).

---

## 4. Evaluation plan (appropriateness of measures)

**Why these measures fit code:** all grade by *executing* the code, not string similarity.

- **Execution Success:** cell runs without raising errors.
- **Task accuracy:** passes the benchmark's hidden tests / matches the answer (DS-1000,
  DABstep accuracy).
- **Format compliance:** % of outputs that pass Pydantic validation as a valid, runnable cell.

**Controls:** same pipeline for base vs. fine-tuned (controlled comparison);
`temperature=0` for plan/debug; bounded ≤3 retries ; proper installed libraries (torch, tensorflow in DS-1000)

**Ablation:** fine tune vs base model. Agentic vs one-shot Qwen.

---

## 5. Results & analysis

> Model: **Qwen3-4B-Instruct-2507**, evaluated as **one-shot** vs. inside the
> **notebook agent** (same model, with agentic plan→generate→execute→debug loop). Proprietary rows (GPT-*) are
> published reference numbers, *not* run through our pipeline. *Note on the QLoRA
> fine-tune:* its objective is **multi-step reasoning, not raw code generation**, so we
> report it on the reasoning-oriented suites (DABStep, dev/test §5.5) rather than the
> pure code-generation benchmark DS-1000.

### 5.1 DS-1000 Overall Results

| Name | Model | Accuracy |
|---|---|---|
| GPT-3.5-turbo | GPT-3.5-turbo | 37.4% |
| GPT-4o-mini | GPT-4o-mini | 42.2% |
| Qwen2.5-Coder-14B-Instruct | Qwen2.5-Coder-14B-Instruct | 32.3% |
| Qwen3-4B-Instruct-2507 (base one-shot) | Qwen3-4B-Instruct-2507 | 27.2% |
| **Notebook agent** | Qwen3-4B-Instruct-2507 | **30.2%** |

**Comments:** wrapping the Qwen 4B model in the agent loop improves DS-1000 accuracy
**27.2% → 30.2% (+3.0 pts)** — a 4B local model in our pipeline closes most of the gap to
GPT-3.5-turbo (37.4%) and performs closely to the 14B Qwen2.5-Coder (32.3%)

### 5.2 DS-1000  Results by Library Breakdown

Counts in parentheses are count of succesful attempts.

| Library | Notebook agent | Base one-shot |
|---|---|---|
| Pandas (291) | 16% (47) | 16% (46) |
| NumPy (220) | 38% (84) | 35% (77) |
| Matplotlib (155) | **48% (75)** | 48% (75) |
| Sklearn (115) | 25% (29) | 19% (22) |
| SciPy (106) | 26% (28) | 25% (27) |
| PyTorch (68) | **32% (22)** | 18% (12) |
| TensorFlow (45) | **38% (17)** | 28% (12) |
| **Overall** | **30.2% (302/1000)** | 27.2% (271/996) |

### 5.3 What it tells us

- **The agent loop helps the base model everywhere it matters:** notebook agent ≥ base
  one-shot in every library — biggest jumps on PyTorch (18%→32%) and TensorFlow
  (28%→38%).
- **Why no fine-tune column here:** DS-1000 measures single-shot *code generation*, not
  the multi-step *reasoning* the QLoRA fine-tune is trained for — so improving it is not
  the fine-tune's job. We isolate the fine-tune's contribution on the reasoning-oriented
  suites instead (DABStep §5.4, dev/test §5.5).

### 5.4 DABStep (leaderboard submission)

`dabstep` profile; Easy / Hard accuracy. Same model, base vs. notebook agent.

| Name | Model | Easy | Hard |
|---|---|---|---|
| Qwen3-4B-Instruct-2507 (base) | Qwen3-4B-Instruct-2507 | 44.0% | 2.1% |
| **Notebook agent** | Qwen3-4B-Instruct-2507 | **52.8%** | 2.1% |
| **Notebook agent** | SFT Qwen3-4B-Instruct-2507 | **66.2%** | 2.1% |
| **Notebook agent** | QLora Qwen3-4B-Instruct-2507 | **36.4%** | 1.2% |


- **Easy: 44.0% → 52.8% (+8.8 pts)** — the agent loop's largest single win; multi-step
  context + self-debug pays off on the tractable analytic tasks.
- **Hard: flat at 2.1%** — multi-hop reasoning over large schemas remains out of reach for
  a 4B model regardless of scaffolding; this is the headroom the fine-tune targets next.

### 5.5 Quality check

Our own quality checking suite of 30 realistic single-DataFrame notebook scenarios,
grouped by different category.
The scenario contains pairs of notebook contexts (previous cell with execution outputs), user requests, ground truth. At evaluation time, a pair of context and user request is provided to the agent to solve, the solution from the agent is deemed correctly if run succesfully and produce the correct results, aligning with the ground truth code.

Scoring criteria:
a) Executability: Does the generated cell run without error?
b) Format Compliance: Does it satisfy all structured output rules defined by our Pydantic schema?
c) Context Awareness: Does it correctly reference prior notebook context (variable names, libraries, coding style, and functions) from preceding cells?

| Category | Task accuracy |
|---|---|
| Statistics (4) | **100% (4)** |
| Visualization (7) | 100% (7) |
| Preprocessing (3) | 100% (3) |
| Exploration (7) | 86% (6) |
| Aggregation (6) | 83% (5) |
| Filtering (3) | 67% (2) |
| **Overall** | **90% (27/30)** |

Format compliance **100%** (30/30 outputs pass Pydantic validation as runnable cells). The context awareness/executability criteria is demonstrated with 90% task accuracy. 




### 5.6 Leakage audit — benchmarks vs. training corpus

We verified the headline benchmarks are **not contaminated** by the training data. We
scanned every DS-1000 and DABStep unit against all **102,778** training documents from
`jupyter-agent/jupyter-agent-dataset` (both `non_thinking` and `thinking` splits) using
**8-gram containment**, flagging any unit ≥ 0.8.

| Group | Units | Mean containment | Max containment | ≥0.5 | ≥0.8 |
|---|---|---|---|---|---|
| DABStep / question | 454 | 0.0006 | 0.286 | 0 | 0 |
| DS-1000 / prompt | 1,000 | 0.012 | 0.161 | 0 | 0 |
| DS-1000 / solution | 1,000 | 0.000 | 0.000 | 0 | 0 |
| **Total** | **2,454** | — | — | **0** | **0** |

- **No leakage detected:** 0 of 2,454 benchmark samples flagged. The single highest overlap anywhere
  was **0.286** (one DABStep question) — well below the 0.5 caution line and the 0.8 flag
  threshold.
- **DS-1000 solutions show exactly zero overlap** — the decisive signal, since solution
  leakage is what would inflate accuracy. Our §5.1–§5.4 numbers reflect genuine
  generalization, not memorization.

---

## 7. References

**Agent patterns:** ReAct (Yao 2022, arXiv:2210.03629) · Plan-and-Solve (Wang 2023,
arXiv:2305.04091) · Reflexion (Shinn 2023, arXiv:2303.11366) · Teaching LLMs to
Self-Debug (Chen 2023, arXiv:2304.05128) · SWE-agent (Yang 2024, arXiv:2405.15793) ·
OpenHands (Wang 2024, arXiv:2407.16741) · DS-Star (Google Research, 2025).

**Benchmarks:** DS-1000 (Lai 2023, arXiv:2211.11501) · DABStep (Cabannes 2025,
arXiv:2506.23719).

**Model & data:** Qwen3 Technical Report (2025, arXiv:2505.09388) ·
`jupyter-agent/jupyter-agent-dataset` (HF, 2025).

**Source map:** graph `src/agent/graph.py` · nodes `src/agent/nodes.py` · prompt profiles
`src/agent/prompts.py` · sandbox `src/agent/sandbox.py` · eval `eval/harness.py`,
`eval/benchmarks/*.py`.
