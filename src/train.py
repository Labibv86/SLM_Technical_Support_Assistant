"""
QLoRA fine-tuning of Qwen2.5-1.5B-Instruct on Stack Overflow Python Q&A.

Run from project root:
    python src/train.py
"""

import os
import sys
import json
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    TRAIN_PATH, VAL_PATH, MAX_SEQ_LENGTH,
    BATCH_SIZE, GRADIENT_ACCUMULATION, LEARNING_RATE, NUM_EPOCHS,
    WARMUP_RATIO, LOGGING_STEPS, EVAL_STEPS, SAVE_STEPS, SEED,
    CHECKPOINT_DIR, TRAIN_LOG_PATH, PLOTS_DIR,
)
from model_loader import load_lora_model
from transformers import TrainerCallback



# ============================================================
# Prompt formatting
# ============================================================

PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a helpful Python programming assistant. "
    "Answer the user's question clearly with code where appropriate."
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
RESPONSE_SUFFIX = "<|im_end|>"


def build_user_text(example):
    if example["input"]:
        return f"{example['instruction']}\n\n{example['input']}"
    return example["instruction"]


# ============================================================
# Tokenization with assistant-only loss
# ============================================================

def make_tokenize_fn(tokenizer):
    def tokenize(example):
        user_text = build_user_text(example)
        prompt = PROMPT_TEMPLATE.format(user=user_text)
        answer = example["output"] + RESPONSE_SUFFIX
        full_text = prompt + answer

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full = tokenizer(full_text, add_special_tokens=False,
                         truncation=True, max_length=MAX_SEQ_LENGTH)

        input_ids = full["input_ids"]
        labels = list(input_ids)
        # Mask prompt tokens (loss = -100 means ignore)
        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": full["attention_mask"],
            "labels": labels,
        }
    return tokenize


# ============================================================
# Metrics callback -> collects step/loss for plotting later
# ============================================================

class LogHistoryCallback(TrainerCallback):
    def __init__(self):
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry = {"step": state.global_step}
        if "loss" in logs:
            entry["loss"] = logs["loss"]
        if "eval_loss" in logs:
            entry["eval_loss"] = logs["eval_loss"]
        if "learning_rate" in logs:
            entry["learning_rate"] = logs["learning_rate"]
        if len(entry) > 1:
            self.history.append(entry)


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading model + tokenizer ...")
    model, tokenizer = load_lora_model()
    tokenize_fn = make_tokenize_fn(tokenizer)

    print("Loading datasets ...")
    data = load_dataset("json", data_files={
        "train": TRAIN_PATH,
        "validation": VAL_PATH,
    })
    print(f"  train = {len(data['train'])}, val = {len(data['validation'])}")

    print("Tokenizing ...")
    tokenized = data.map(
        tokenize_fn,
        remove_columns=data["train"].column_names,
        desc="tokenize",
    )

    # ---- Sanity check: decode one sample and confirm masking ----
    sample = tokenized["train"][0]
    ids = torch.tensor(sample["input_ids"])
    labels = torch.tensor(sample["labels"])
    print("\n=== Sanity check (first training sample) ===")
    print("PROMPT+MASKED region (first 200 chars):")
    print(tokenizer.decode(ids[:50], skip_special_tokens=False)[:200])
    print("\nSUPERVISED region:")
    sup_ids = labels[labels != -100]
    print(tokenizer.decode(sup_ids, skip_special_tokens=False)[:300])
    print(f"\nsupervised tokens = {(labels != -100).sum().item()} / {len(labels)}")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_RATIO,   
        lr_scheduler_type="cosine",
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=False,
        report_to="none",
        seed=SEED,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    log_cb = LogHistoryCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        callbacks=[log_cb],
    )

    print("\nStarting training ...")
    trainer.train()

    print("\nSaving final adapter ...")
    final_dir = os.path.join(CHECKPOINT_DIR, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    with open(TRAIN_LOG_PATH, "w") as f:
        json.dump(log_cb.history, f, indent=2)
    print(f"Training log -> {TRAIN_LOG_PATH}")


if __name__ == "__main__":
    main()