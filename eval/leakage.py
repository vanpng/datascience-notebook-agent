"""Benchmark contamination / data-leakage detector.

Question this answers
---------------------
"Did any DS-1000 or DABStep benchmark content leak into the data I used to
fine-tune Qwen3-4B?"  The model was fine-tuned on ``jupyter-agent/jupyter-agent-
dataset`` (see ``fine_tuning/config.yaml`` + ``fine_tuning/prepare_data.py``).
That prep step only de-duplicates against benchmarks with a *keyword* filter
(``"ds1000"``, ``"dabstep"``, …), which catches rows that literally name a
benchmark but **not** rows that contain the same problem statements or reference
solutions verbatim.  This module checks for that content-level overlap.

Dataset schema note
-------------------
The live dataset is keyed by ``messages`` (chat turns: system/user/assistant/
tool, with the run code in ``tool_calls[].function.arguments``) and split into
``thinking`` / ``non_thinking`` — it does NOT match the ``context``/``code_cell``
+ ``train`` schema named in the stale ``config.yaml``.  We therefore scan the
actual row content (see ``_row_text``).  Scanning a full split is conservative:
a benchmark item absent from the split was absent from any sample drawn from it.

Method — n-gram containment (GPT-3 / PaLM style)
------------------------------------------------
For each benchmark "unit" (a DS-1000 problem statement, a DS-1000 reference
solution, or a DABStep question) we compute its set of word ``n``-grams and
measure what fraction appear in the training corpus.  Two numbers are reported
per unit:

* ``corpus_containment``  — fraction of the unit's n-grams found *anywhere* in
  the corpus (union over all docs).  Sensitive but noisy: common idioms spread
  across many docs inflate it.
* ``max_doc_containment`` — the largest fraction found within a *single*
  training document.  This is the reliable leakage signal: a single training
  row reproducing most of a benchmark unit is what contamination looks like.

Scalability: we index the (small) benchmark side — n-gram hash → unit ids — and
stream the (large) training corpus past it exactly once.  Memory is bounded by
the benchmark, not the corpus, so it scales to the full 95.8k-row dataset.

For any unit flagged above ``flag_threshold`` we additionally compute the
longest common contiguous substring against its best-matching training doc
(``difflib``) as human-readable evidence.

This is a pure data-analysis pipeline: it needs no model server and no agent
API — only the ``datasets`` library and the benchmark caches.
"""
from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable, Iterator, Optional

logger = logging.getLogger(__name__)

# Mirrors fine_tuning/prepare_data.py — rows containing any of these are excluded
# from the fine-tuning set, so we exclude them here too when reconstructing the
# "data I trained on" (override with include_filtered=True to scan everything).
BENCHMARK_KEYWORDS = [
    "dscodebiench", "ds-1000", "ds1000", "dabstep", "dsbench",
]


def _is_benchmark_overlap(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in BENCHMARK_KEYWORDS)


# ── text normalisation & n-grams ────────────────────────────────────────────

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase and collapse all whitespace runs to single spaces.

    Collapsing whitespace makes the comparison robust to reformatting
    (indentation, blank lines, line wrapping) — the dominant source of
    cosmetic variation between a benchmark item and a copy of it in a notebook.
    Punctuation is kept, because in code it is meaningful (``df['a']`` vs ``df a``).
    """
    return _WS_RE.sub(" ", text.lower()).strip()


def _tokens(text: str) -> list[str]:
    return _normalize(text).split(" ") if text else []


def _ngram_hashes(tokens: list[str], n: int) -> set[int]:
    """Return the set of hashes of every contiguous ``n``-token window.

    For text shorter than ``n`` tokens the whole thing is treated as one gram so
    very short units (e.g. a one-line solution) still produce a signal — flagged
    as low-confidence by the caller via ``n_tokens``.
    Uses the builtin ``hash`` (fast; consistent within a single process run,
    which is all we need since hashes are never persisted).
    """
    if not tokens:
        return set()
    if len(tokens) < n:
        return {hash(" ".join(tokens))}
    return {hash(" ".join(tokens[i : i + n])) for i in range(len(tokens) - n + 1)}


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class BenchUnit:
    """One comparable piece of a benchmark."""
    uid: str                 # e.g. "ds1000:Pandas_3:solution"
    benchmark: str           # "ds1000" | "dabstep"
    field: str               # "prompt" | "solution" | "question"
    text: str
    tokens: list[str] = field(default_factory=list)
    grams: set[int] = field(default_factory=set)

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    @property
    def n_grams(self) -> int:
        return len(self.grams)


@dataclass
class UnitFinding:
    uid: str
    benchmark: str
    field: str
    n_tokens: int
    n_grams: int
    corpus_containment: float          # union over all docs
    max_doc_containment: float         # best single doc
    best_doc_id: Optional[str] = None
    best_doc_overlap: int = 0
    longest_common_substring: str = ""
    lcs_len: int = 0
    text_preview: str = ""
    best_doc_preview: str = ""


# ── benchmark unit loaders ────────────────────────────────────────────────────

def load_ds1000_units(libraries: Optional[list[str]] = None) -> list[BenchUnit]:
    """DS-1000 → two units per problem: the prompt and the reference solution."""
    from eval.benchmarks.ds1000 import LIBRARIES, _load_ds1000

    libs = [l.capitalize() for l in (libraries or LIBRARIES)]
    units: list[BenchUnit] = []
    for r in _load_ds1000(libs):
        pid = r["id"]
        if r.get("prompt"):
            units.append(BenchUnit(f"ds1000:{pid}:prompt", "ds1000", "prompt", r["prompt"]))
        if r.get("reference_code"):
            units.append(
                BenchUnit(f"ds1000:{pid}:solution", "ds1000", "solution", r["reference_code"])
            )
    logger.info("Loaded %d DS-1000 units (%d problems, %s)",
                len(units), len(units) // 2 if units else 0, ", ".join(libs))
    return units


def load_dabstep_units(splits: Iterable[str] = ("dev", "default")) -> list[BenchUnit]:
    """DABStep → one unit per unique question (across the requested splits).

    The ``default`` split (450 leaderboard tasks) and ``dev`` split (10 tasks
    with answers) share task_ids; we union by task_id.  Only the question text
    is checked — answers are too short to n-gram meaningfully.
    """
    from eval.benchmarks.dabstep import _load_dabstep

    seen: set[str] = set()
    units: list[BenchUnit] = []
    for split in splits:
        try:
            rows = _load_dabstep(split)
        except Exception as exc:  # noqa: BLE001 — a split may be unavailable
            logger.warning("DABStep split %r unavailable: %s", split, exc)
            continue
        for r in rows:
            tid = str(r.get("task_id", ""))
            q = r.get("question", "")
            if not q or tid in seen:
                continue
            seen.add(tid)
            units.append(BenchUnit(f"dabstep:{tid}:question", "dabstep", "question", q))
    logger.info("Loaded %d unique DABStep question units", len(units))
    return units


# ── training-corpus iterators ──────────────────────────────────────────────────

def _decode_notebook(nb_json: str) -> str:
    """Extract concatenated cell sources from an original_notebook JSON string."""
    import json
    try:
        nb = json.loads(nb_json)
        out = []
        for cell in nb.get("cells", []):
            src = cell.get("source", "")
            out.append("".join(src) if isinstance(src, list) else str(src))
        return "\n".join(out)
    except Exception:  # noqa: BLE001 — fall back to the raw string
        return nb_json


def _row_text(row: dict, include_notebook: bool = False) -> str:
    """Concatenate all training-relevant text from one jupyter-agent-dataset row.

    The fine-tuning content is the chat ``messages`` (system/user/assistant/tool
    turns).  The code the assistant runs is in each message's
    ``tool_calls[].function.arguments`` (a ``{"answer","code"}`` blob), so we
    pull those too — that is where a copied benchmark *solution* would surface.
    ``question``/``answer`` cover DABStep-style NL leakage.  ``original_notebook``
    (the raw source notebook) is included only when ``include_notebook`` is set.

    Also tolerates an older ``context``/``code_cell`` schema for forward-compat.
    """
    parts: list[str] = []
    msgs = row.get("messages")
    if msgs is not None:
        for m in msgs:
            if not hasattr(m, "get"):
                continue
            c = m.get("content")
            if c:
                parts.append(str(c))
            tcs = m.get("tool_calls")
            if tcs is not None:
                try:
                    for tc in tcs:
                        fn = tc.get("function") if hasattr(tc, "get") else None
                        if fn is not None and hasattr(fn, "get") and fn.get("arguments"):
                            parts.append(str(fn.get("arguments")))
                except TypeError:
                    pass
    for col in ("question", "answer", "context", "code_cell"):
        v = row.get(col)
        if v:
            parts.append(str(v))
    if include_notebook:
        nb = row.get("original_notebook")
        if nb:
            parts.append(_decode_notebook(str(nb)))
    return "\n".join(parts)


def iter_training_docs(
    *,
    dataset_id: str,
    splits: list[str],
    sample_size: Optional[int] = None,
    seed: int = 42,
    include_filtered: bool = False,
    include_notebook: bool = False,
    max_docs: Optional[int] = None,
) -> Iterator[tuple[str, str]]:
    """Yield ``(doc_id, text)`` for each row across the requested ``splits``.

    By default every row of each split is scanned (the conservative check: a
    benchmark item absent from the full split was absent from any sample of it).
    Pass ``sample_size`` to instead draw a seeded random subset per split.
    ``include_filtered=False`` drops rows matching benchmark keywords (mirroring
    ``prepare_data.py``); set it True to scan those too.
    """
    from datasets import load_dataset  # type: ignore

    yielded = filtered = 0
    for split in splits:
        logger.info("Loading %s [%s] …", dataset_id, split)
        ds = load_dataset(dataset_id, split=split)
        logger.info("  split %s: %d rows", split, len(ds))
        rows = ds
        if sample_size:
            idx = random.Random(seed).sample(range(len(ds)), min(sample_size, len(ds)))
            rows = ds.select(idx)
            logger.info("  sampled %d rows (seed=%d)", len(rows), seed)
        for i, row in enumerate(rows):
            text = _row_text(row, include_notebook=include_notebook)
            if not text.strip():
                continue
            if not include_filtered and _is_benchmark_overlap(text):
                filtered += 1
                continue
            yield (f"{split}:{i}", text)
            yielded += 1
            if max_docs and yielded >= max_docs:
                logger.info("Streamed %d docs (%d filtered by benchmark-keyword)",
                            yielded, filtered)
                return
    logger.info("Streamed %d docs (%d filtered by benchmark-keyword)", yielded, filtered)


# ── detector ────────────────────────────────────────────────────────────────

class LeakageDetector:
    """Benchmark-indexed n-gram containment scan over a training corpus."""

    def __init__(self, n: int = 8, flag_threshold: float = 0.8) -> None:
        self.n = n
        self.flag_threshold = flag_threshold
        self.units: list[BenchUnit] = []
        self._index: dict[int, list[int]] = {}
        self._index_keys: set[int] = set()
        # per-unit scan state
        self._covered: list[set[int]] = []
        self._best_overlap: list[int] = []
        self._best_doc: list[Optional[str]] = []
        self._best_doc_text: list[str] = []
        self.n_docs = 0

    # ── index ─────────────────────────────────────────────────────────────────
    def build_index(self, units: list[BenchUnit]) -> None:
        self.units = units
        self._index = {}
        for u_idx, u in enumerate(units):
            u.tokens = _tokens(u.text)
            u.grams = _ngram_hashes(u.tokens, self.n)
            for g in u.grams:
                self._index.setdefault(g, []).append(u_idx)
        self._index_keys = set(self._index)
        n_units = len(units)
        self._covered = [set() for _ in range(n_units)]
        self._best_overlap = [0] * n_units
        self._best_doc = [None] * n_units
        self._best_doc_text = [""] * n_units
        logger.info("Indexed %d benchmark units → %d distinct %d-grams",
                    n_units, len(self._index), self.n)

    # ── scan ──────────────────────────────────────────────────────────────────
    def scan(self, docs: Iterable[tuple[str, str]]) -> None:
        index = self._index
        keys = self._index_keys
        covered = self._covered
        n = self.n
        t0 = time.perf_counter()
        count = 0
        for doc_id, text in docs:
            count += 1
            doc_grams = _ngram_hashes(_tokens(text), n)
            matched = doc_grams & keys
            if matched:
                local: dict[int, int] = {}
                for g in matched:
                    for u in index[g]:
                        covered[u].add(g)
                        local[u] = local.get(u, 0) + 1
                for u, cnt in local.items():
                    if cnt > self._best_overlap[u]:
                        self._best_overlap[u] = cnt
                        self._best_doc[u] = doc_id
                        self._best_doc_text[u] = text[:4000]
            if count % 2000 == 0:
                logger.info("  scanned %d docs (%.0fs)…", count, time.perf_counter() - t0)
        self.n_docs = count
        logger.info("Scan complete: %d docs in %.1fs", count, time.perf_counter() - t0)

    # ── results ─────────────────────────────────────────────────────────────────
    def findings(self) -> list[UnitFinding]:
        out: list[UnitFinding] = []
        for i, u in enumerate(self.units):
            ng = max(1, u.n_grams)
            corpus_c = len(self._covered[i]) / ng
            max_doc_c = self._best_overlap[i] / ng
            f = UnitFinding(
                uid=u.uid, benchmark=u.benchmark, field=u.field,
                n_tokens=u.n_tokens, n_grams=u.n_grams,
                corpus_containment=round(corpus_c, 4),
                max_doc_containment=round(max_doc_c, 4),
                best_doc_id=self._best_doc[i],
                best_doc_overlap=self._best_overlap[i],
                text_preview=u.text[:300],
            )
            # Evidence for flagged units: longest common contiguous substring.
            if max_doc_c >= self.flag_threshold and self._best_doc_text[i]:
                a = _normalize(u.text)
                b = _normalize(self._best_doc_text[i])
                sm = SequenceMatcher(None, a, b, autojunk=False)
                m = sm.find_longest_match(0, len(a), 0, len(b))
                f.longest_common_substring = a[m.a : m.a + m.size][:400]
                f.lcs_len = m.size
                f.best_doc_preview = self._best_doc_text[i][:400]
            out.append(f)
        return out


# ── reporting ───────────────────────────────────────────────────────────────

def summarize(findings: list[UnitFinding], flag_threshold: float) -> dict:
    """Aggregate findings into a JSON-serialisable report."""
    by_group: dict[str, list[UnitFinding]] = {}
    for f in findings:
        by_group.setdefault(f"{f.benchmark}/{f.field}", []).append(f)

    groups = {}
    for key, fs in sorted(by_group.items()):
        md = [f.max_doc_containment for f in fs]
        groups[key] = {
            "units": len(fs),
            "mean_max_doc_containment": round(sum(md) / len(md), 4) if md else 0.0,
            "max_max_doc_containment": round(max(md), 4) if md else 0.0,
            "n_ge_0.5": sum(1 for x in md if x >= 0.5),
            "n_ge_0.8": sum(1 for x in md if x >= 0.8),
            "n_eq_1.0": sum(1 for x in md if x >= 0.999),
        }

    flagged = sorted(
        (f for f in findings if f.max_doc_containment >= flag_threshold),
        key=lambda f: -f.max_doc_containment,
    )
    return {
        "flag_threshold": flag_threshold,
        "total_units": len(findings),
        "total_flagged": len(flagged),
        "leakage_detected": bool(flagged),
        "by_group": groups,
        "flagged": [vars(f) for f in flagged],
    }
