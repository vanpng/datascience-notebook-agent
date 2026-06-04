"""
QLoRA fine-tuning of Qwen3-4B on the prepared jupyter-agent-dataset.

Run (after prepare_data.py):
    uv run --extra train python fine_tuning/train.py

Expects data/ft/{train,validation}.jsonl to exist (produced by prepare_data.py).
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

CFG_PATH = Path(__file__).parent / "config.yaml"


def main():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    model_cfg   = cfg["model"]
    lora_cfg    = cfg["lora"]
    bnb_cfg     = cfg["bnb"]
    train_cfg   = cfg["training"]
    data_cfg    = cfg["data"]

    # ── Tokeniser ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        trust_remote_code=model_cfg["trust_remote_code"],
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 4-bit quantisation ────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bnb_cfg["load_in_4bit"],
        bnb_4bit_quant_type=bnb_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, bnb_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=bnb_cfg["bnb_4bit_use_double_quant"],
    )

    # ── Base model ────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=bnb_config,
        torch_dtype=getattr(torch, model_cfg["torch_dtype"]),
        device_map="auto",
        trust_remote_code=model_cfg["trust_remote_code"],
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds = load_dataset(
        "json",
        data_files={
            "train":      "data/ft/train.jsonl",
            "validation": "data/ft/validation.jsonl",
        },
    )

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        eval_steps=train_cfg["eval_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        report_to=train_cfg["report_to"],
        run_name=train_cfg["run_name"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        group_by_length=train_cfg["group_by_length"],
    )

    # Only train on the assistant turn (completion-only masking)
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=data_cfg["max_seq_length"],
        packing=False,
    )

    trainer.train()

    # Save final adapter
    final_dir = Path(train_cfg["output_dir"]) / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nFine-tuned adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
