"""
Prepare the jupyter-agent-dataset for fine-tuning.

Steps:
  1. Load jupyter-agent/jupyter-agent-dataset from Hugging Face.
  2. Randomly sample `sample_size` rows.
  3. Filter rows whose source overlaps with evaluation benchmarks.
  4. Format each row as a (system + user + assistant) chat template.
  5. Save train/val/test splits as JSONL under data/ft/.

Run: uv run --extra train python fine_tuning/prepare_data.py
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import yaml
from datasets import load_dataset, DatasetDict

CFG_PATH = Path(__file__).parent / "config.yaml"
OUT_DIR = Path("data/ft")

# Keywords that flag a sample as overlapping with an evaluation benchmark.
# Add benchmark-specific identifiers here as needed.
BENCHMARK_KEYWORDS = [
    "dscodebiench", "ds-1000", "ds1000", "dabstep", "dsbench",
]

CHAT_TEMPLATE = """\
<|im_start|>system
You are an expert data science assistant embedded in a Jupyter Notebook. \
Write a single, complete, immediately executable Python notebook cell.
<|im_end|>
<|im_start|>user
{context}
<|im_end|>
<|im_start|>assistant
{code_cell}
<|im_end|>"""


def is_benchmark_overlap(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in BENCHMARK_KEYWORDS)


def format_sample(row: dict, input_col: str, output_col: str) -> str | None:
    ctx = row.get(input_col, "")
    code = row.get(output_col, "")
    if not ctx or not code:
        return None
    if is_benchmark_overlap(ctx) or is_benchmark_overlap(code):
        return None
    return CHAT_TEMPLATE.format(context=ctx.strip(), code_cell=code.strip())


def main():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    sample_size = data_cfg["sample_size"]
    seed = data_cfg["seed"]
    input_col = data_cfg["input_col"]
    output_col = data_cfg["output_col"]

    print(f"Loading {data_cfg['dataset_id']} …")
    ds = load_dataset(data_cfg["dataset_id"], split="train")

    # Random sample
    indices = random.Random(seed).sample(range(len(ds)), min(sample_size, len(ds)))
    ds = ds.select(indices)

    formatted: list[str] = []
    skipped = 0
    for row in ds:
        text = format_sample(row, input_col, output_col)
        if text is None:
            skipped += 1
            continue
        formatted.append(text)

    print(f"Formatted {len(formatted)} samples ({skipped} skipped / benchmark overlap).")

    random.Random(seed).shuffle(formatted)
    n = len(formatted)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    splits = {
        "train":      formatted[:n_train],
        "validation": formatted[n_train : n_train + n_val],
        "test":       formatted[n_train + n_val :],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, samples in splits.items():
        out_path = OUT_DIR / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for s in samples:
                f.write(json.dumps({"text": s}) + "\n")
        print(f"  {split_name}: {len(samples)} samples → {out_path}")


if __name__ == "__main__":
    main()
