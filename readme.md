# QLoRA Fine-Tuning of Qwen2.5-1.5B for Domain-Specific Python Q&A

A parameter-efficient fine-tuning (QLoRA) experiment on a small language model, trained on
Stack Overflow Python Q&A. The goal is not just to train a model, but to **rigorously measure
whether fine-tuning actually improves answer quality over the base model** — with a proper
held-out set, base-vs-tuned comparison, and three complementary metrics.


## Project Overview

I set out to answer a practical question: **can a 1.5B-parameter instruction-tuned model be
specialized on a narrow domain (Python programming Q&A) using only consumer hardware
(RTX 3060, 12GB VRAM)?**

The project delivers:

- A data pipeline that scrapes 4,000 high-scoring Python questions from Stack Overflow,
  pairs each with its accepted answer, and converts the raw HTML into clean Markdown
  (preserving code blocks as fenced blocks).
- A QLoRA fine-tuning pipeline using 4-bit NF4 quantization + LoRA adapters
  (r=8, alpha=16) on `Qwen/Qwen2.5-1.5B-Instruct`.
- A base-vs-tuned evaluation on a 20% held-out split, using **ROUGE-L**, **token-level F1**,
  and **perplexity** on the reference answers, plus per-sample scatter analysis.


## Motivation

The dominant story in LLM engineering is "bigger model = better". But for narrow domain
tasks — technical support, documentation QA, internal tooling — a small, well-tuned model
is often the right engineering choice: cheaper to host, faster to run, and easier to audit.

I wanted to build the full loop myself: from raw web data, through cleaning, through
parameter-efficient training, to a defensible evaluation. The last step is what most
portfolio projects skip, and it's the one that matters most.



## Data Pipeline

**Source**: Stack Exchange API (`/questions` endpoint, `tagged=python`, `min=5`).
**Volume**: 4,000 questions with accepted answers, ~100MB raw JSON.
**Quality filter**: only questions with `score >= 5` and an accepted answer.

The scraper uses a custom Stack Exchange filter to pull question and answer bodies in a
single request, with a 3-second delay between pages to respect rate limits. The raw output
is HTML (from Markdown source), which required careful cleaning:

- `<pre><code>` → fenced ```` ``` ```` blocks (language tag preserved when present)
- Inline `<code>` → backtick spans
- `<a href="...">` → `[text](url)`
- `<p>`, `<li>`, `<blockquote>`, `<h1..h4>`, `<b>/<i>` → Markdown equivalents
- HTML entity decoding (`&quot;` → `"`, etc.)

After cleaning, I deduplicated by title and filtered out answers shorter than 100 characters
(usually "see duplicate" links). The final corpus is split 80/20 into train and validation.

**Important detail**: only the **assistant tokens** are supervised during training — the
prompt is masked with `-100` in the label tensor. This is the correct way to fine-tune
instruct models; without it, the model learns to generate the system prompt and the user's
question verbatim, which wastes capacity and degrades quality.



## Model and Training

**Base model**: `Qwen/Qwen2.5-1.5B-Instruct` — a recent, strong small model with native
ChatML formatting and good instruction-following behavior.

**Quantization**: 4-bit NF4 with double quantization (`bitsandbytes`). This brings VRAM
usage down to ~2-3GB for the frozen base, leaving headroom for activations, gradients,
and optimizer states on a 12GB card.

**LoRA configuration**:

| Parameter | Value |
|---|---|
| Rank (`r`) | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Target modules | q, k, v, o, gate, up, down projections |
| Trainable params | ~6M (~0.4% of base) |

**Training**:

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Epochs | 2 |
| Batch size | 2 (×4 grad accum = effective 8) |
| Max sequence length | 1024 |
| LR schedule | cosine, 3% warmup |

**Engineering choices**:

- I initially tried `learning_rate = 2e-4, LORA_R = 16` — this produced repetition collapse
  ("importititititit", "The The") at inference. Lowering the LR to 1e-4 and reducing rank
  to 8 stabilized training. This kind of regression is not documented in the popular
  recipes, which mostly target 7B models.
- Tokenized lengths are capped at 1024. When the prompt (question + question body for
  short titles) plus answer exceeds this, the answer gets truncated and the sample is
  skipped in perplexity computation (guarded against `-100`-only label tensors).



## Evaluation

The **base model** and the **LoRA-adapted model** are both evaluated on the held-out 20%
split, with identical prompts and greedy decoding (`max_new_tokens=512`).

**Metrics**:

1. **ROUGE-L** (stemmed, F-measure). Standard for open-ended generation; captures longest
   common subsequence between generated and reference answer.
2. **Token-level F1** (SQuAD-style). Symmetric overlap measure; less forgiving than ROUGE-L
   for reordered content but stricter on content recall.
3. **Perplexity on reference answer** (assistant-token-only loss). Measures whether the
   tuned model assigns higher probability to the *actual* answer text — a proper
   likelihood check, complementary to ROUGE/F1.


## Why ROUGE-L in the 0.15–0.25 range is not a failure

Small-model fine-tuning on open-ended Q&A rarely produces high ROUGE-L against
single-reference Stack Overflow answers, for three reasons:

1. **Reference answers are one specific human answer.** A semantically identical answer
   phrased differently scores low on n-gram overlap. E.g., "use a dict comprehension"
   vs. "iterate over `.items()` and invert" — same advice, different tokens.
2. **Stack Overflow answers are long and meandering** (historical context, "per comment
   below...", external links). A concise correct answer scores low against them.
3. **The information-theoretic ceiling for single-reference ROUGE-L on open-ended
   generation is ~0.25–0.35**, even for much larger models.

What matters is the **delta between base and tuned**. If the tuned model wins on both
ROUGE-L and F1 and has lower perplexity on references, the fine-tuning worked — regardless
of the absolute number.


## Limitations and What I'd Do Differently

- **Validation set size**: 760 examples is enough for a directional signal, but not for
  tight confidence intervals. A proper setup would bootstrap the metric differences or
  hold out 2,000+ samples.
- **Single-reference ROUGE-L**: a BLEU-4 or BERTScore (semantic similarity) evaluation
  would complement ROUGE-L and provide a much more defensible number for open-ended
  generation. I'd add BERTScore if I extended the project.
- **No human evaluation**: the gold standard for QA quality is human preference. A 50-sample
  pairwise preference test between base and tuned outputs would strengthen the claim far
  more than any automatic metric.
- **Data contamination**: Stack Overflow answers include code that may have appeared in
  Qwen's pretraining data. A temporal split (train on old questions, test on questions
  posted after Qwen's cutoff) would be the clean way to avoid this — but the API doesn't
  easily support that at this scale.


