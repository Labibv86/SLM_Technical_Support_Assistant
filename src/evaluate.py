"""
Evaluate base vs. fine-tuned Qwen2.5-1.5B-Instruct on held-out val set.

Run from project root:
    python src/evaluate.py

Outputs:
    outputs/plots/loss_curve.png
    outputs/plots/rouge_f1_comparison.png
    outputs/plots/length_vs_rouge.png
    outputs/eval_results.json
"""

import os
import sys
import json
import torch
from datasets import load_dataset
from tqdm import tqdm
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    VAL_PATH, MAX_SEQ_LENGTH, MAX_NEW_TOKENS, BATCH_SIZE,
    TRAIN_LOG_PATH, PLOTS_DIR, CHECKPOINT_DIR, OUTPUT_DIR, SEED,
)
from model_loader import load_tokenizer, load_base_model

from peft import PeftModel


# ============================================================
# Prompt (must match train.py exactly)
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


def build_user_text(example):
    if example["input"]:
        return f"{example['instruction']}\n\n{example['input']}"
    return example["instruction"]


def build_prompt(example):
    return PROMPT_TEMPLATE.format(user=build_user_text(example))


# ============================================================
# Metrics
# ============================================================

def token_f1(pred, ref):
    """SQuAD-style token-level F1 on whitespace tokens."""
    pred_toks = pred.lower().split()
    ref_toks = ref.lower().split()
    if not pred_toks or not ref_toks:
        return 0.0
    common = {}
    for t in pred_toks:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in ref_toks:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


def compute_rouge_l(scorer, pred, ref):
    return scorer.score(ref, pred)["rougeL"].fmeasure


# ============================================================
# Generation
# ============================================================

@torch.no_grad()
def generate_batch(model, tokenizer, prompts):
    """Greedy generation for a batch of prompts."""
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS,
    ).to(model.device)

    out = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,          # greedy
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # Strip input tokens
    gen = out[:, enc["input_ids"].shape[1]:]
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


@torch.no_grad()
def compute_perplexity(model, tokenizer, examples):
    """Perplexity of reference answers (loss on assistant tokens only)."""
    losses = []
    skipped = 0
    for ex in examples:
        prompt = build_prompt(ex)
        full = prompt + ex["output"] + "<|im_end|>"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        enc = tokenizer(full, return_tensors="pt", add_special_tokens=False,
                        truncation=True, max_length=MAX_SEQ_LENGTH).to(model.device)

        labels = enc["input_ids"].clone()
        mask_len = min(len(prompt_ids), labels.shape[1])
        labels[:, :mask_len] = -100

        # Skip if no supervised tokens survived truncation
        if (labels != -100).sum().item() == 0:
            skipped += 1
            continue

        out = model(**enc, labels=labels)
        if torch.isnan(out.loss) or torch.isinf(out.loss):
            skipped += 1
            continue
        losses.append(out.loss.item())

    if skipped:
        print(f"  (perplexity: skipped {skipped} samples with no supervised tokens)")
    if not losses:
        return float("nan")

    mean_loss = float(np.mean(losses))
    # Cap to avoid exp overflow
    mean_loss = min(mean_loss, 20.0)
    return float(np.exp(mean_loss))


# ============================================================
# Plots
# ============================================================

def plot_loss_curve():
    if not os.path.exists(TRAIN_LOG_PATH):
        print(f"  (skip) no train log at {TRAIN_LOG_PATH}")
        return
    with open(TRAIN_LOG_PATH) as f:
        history = json.load(f)
    steps   = [h["step"] for h in history if "loss" in h]
    train_l = [h["loss"]  for h in history if "loss" in h]
    eval_st = [h["step"]  for h in history if "eval_loss" in h]
    eval_l  = [h["eval_loss"] for h in history if "eval_loss" in h]

    plt.figure(figsize=(9, 5))
    plt.plot(steps, train_l, label="train loss", alpha=0.7)
    if eval_st:
        plt.plot(eval_st, eval_l, "o-", label="eval loss")
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.title("Training / Evaluation Loss")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "loss_curve.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


def plot_comparison(metrics):
    labels = ["ROUGE-L", "Token F1"]
    base_vals   = [metrics["base"]["rouge_l"],    metrics["tuned"]["rouge_l"]]
    tuned_vals  = [metrics["base"]["token_f1"],   metrics["tuned"]["token_f1"]]
    # Wait — this was a bug in my draft; fix:
    base_vals  = [metrics["base"]["rouge_l"],  metrics["base"]["token_f1"]]
    tuned_vals = [metrics["tuned"]["rouge_l"], metrics["tuned"]["token_f1"]]

    x = np.arange(len(labels))
    w = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - w/2, base_vals,  w, label="base",   color="#6aa3d5")
    plt.bar(x + w/2, tuned_vals, w, label="tuned",  color="#d56a6a")
    for i, (b, t) in enumerate(zip(base_vals, tuned_vals)):
        plt.text(i - w/2, b + 0.01, f"{b:.3f}", ha="center", fontsize=9)
        plt.text(i + w/2, t + 0.01, f"{t:.3f}", ha="center", fontsize=9)
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Base vs. Fine-tuned")
    plt.legend(); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "rouge_f1_comparison.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


def plot_length_vs_rouge(per_sample):
    plt.figure(figsize=(9, 5))
    for key, color, label in [("base", "#6aa3d5", "base"),
                              ("tuned", "#d56a6a", "tuned")]:
        xs = [s["ref_len"]  for s in per_sample[key]]
        ys = [s["rouge_l"]  for s in per_sample[key]]
        plt.scatter(xs, ys, s=18, alpha=0.6, label=label, color=color)
    plt.xlabel("Reference answer length (words)")
    plt.ylabel("ROUGE-L F1")
    plt.title("ROUGE-L vs. Reference Length")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "length_vs_rouge.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


# ============================================================
# Main
# ============================================================

def run_model(model, tokenizer, examples, tag):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    per_sample = []
    print(f"[{tag}] generating on {len(examples)} examples ...")
    for i in tqdm(range(0, len(examples), BATCH_SIZE)):
        batch = examples[i:i + BATCH_SIZE]
        prompts = [build_prompt(ex) for ex in batch]
        outputs = generate_batch(model, tokenizer, prompts)
        for ex, out in zip(batch, outputs):
            per_sample.append({
                "rouge_l":  compute_rouge_l(scorer, out, ex["output"]),
                "token_f1": token_f1(out, ex["output"]),
                "ref_len":  len(ex["output"].split()),
                "pred":     out,
            })

    print(f"[{tag}] computing perplexity ...")
    ppl = compute_perplexity(model, tokenizer, examples)

    agg = {
        "rouge_l":  float(np.mean([s["rouge_l"]  for s in per_sample])),
        "token_f1": float(np.mean([s["token_f1"] for s in per_sample])),
        "perplexity": ppl,
        "n": len(per_sample),
    }
    print(f"[{tag}] ROUGE-L={agg['rouge_l']:.4f}  "
          f"F1={agg['token_f1']:.4f}  PPL={ppl:.2f}")
    return agg, per_sample


def main():
    

    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    print("Loading tokenizer ...")
    tokenizer = load_tokenizer()

    print("Loading val set ...")
    ds = load_dataset("json", data_files={"val": VAL_PATH})["val"]
    examples = list(ds)
    print(f"  {len(examples)} examples")

    # ---- Base model ----
    print("\nLoading BASE model ...")
    base = load_base_model()
    base.eval()
    base_metrics, base_per = run_model(base, tokenizer, examples, "base")
    del base
    torch.cuda.empty_cache()

    # ---- Tuned model ----
    print("\nLoading FINE-TUNED model ...")
    tuned_base = load_base_model()
    tuned = PeftModel.from_pretrained(tuned_base,
                                      os.path.join(CHECKPOINT_DIR, "final"))
    tuned.eval()
    tuned_metrics, tuned_per = run_model(tuned, tokenizer, examples, "tuned")
    del tuned
    torch.cuda.empty_cache()

    # ---- Save aggregate ----
    metrics = {"base": base_metrics, "tuned": tuned_metrics}
    with open(os.path.join(OUTPUT_DIR, "eval_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- Plots ----
    print("\nGenerating plots ...")
    plot_loss_curve()
    plot_comparison(metrics)
    plot_length_vs_rouge({"base": base_per, "tuned": tuned_per})

    # ---- 3 side-by-side samples ----
    print("\n=== SIDE-BY-SIDE SAMPLES ===")
    # pick: worst base, best tuned gain, longest ref
    gains = [tuned_per[i]["rouge_l"] - base_per[i]["rouge_l"]
             for i in range(len(examples))]
    picks = [
        int(np.argmin([b["rouge_l"] for b in base_per])),   # worst base
        int(np.argmax(gains)),                              # biggest gain
        int(np.argmax([e["output"].count(" ") for e in examples])),  # longest
    ]
    for idx in picks:
        ex = examples[19]
        print("\n=== PROMPT for sample 19 (raw) ===")
        print(repr(build_prompt(ex)[:600]))
        print("\n" + "=" * 70)
        print(f"SAMPLE #{idx}  |  Q: {ex['instruction'][:80]}")
        print("=" * 70)
        print("\n--- REFERENCE (first 400 chars) ---")
        print(ex["output"][:400])
        print("\n--- BASE (first 400 chars) ---")
        print(base_per[idx]["pred"][:400])
        print("\n--- TUNED (first 400 chars) ---")
        print(tuned_per[idx]["pred"][:400])


if __name__ == "__main__":
    main()