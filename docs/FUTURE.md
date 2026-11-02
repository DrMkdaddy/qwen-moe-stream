# From format-level MoE to a trained MoE

This repo stops at the format boundary: it makes a dense FFN *addressable* as
experts and proves the exact fanout. It does not create new capability. The
next levels are different projects with different compute budgets.

## Level 0 — format-level (this repo)

- Partition a dense FFN into experts.
- Select all experts for exactness, or top-k for a measured lossy trade.
- Stream expert blocks with mmap; peak memory is one expert per layer.

Useful for inference infrastructure, GGUF tooling, and memory-bounded serving.
Not a way to make a small model smarter.

## Level 1 — upcycled MoE

Start from a dense checkpoint, copy the FFN into `E` experts, add a router, and
continue training. This is the standard "sparse upcycling" route. The router
and experts learn to cooperate, so top-k stops being a lossy slice of a sum.

Rough recipe:

1. Initialize `E` experts from the dense FFN (or from slices plus a shared
   expert).
2. Train the router with a load-balancing auxiliary loss and a token-drop
   schedule; keep the dense path as a teacher for the first phase.
3. Fine-tune on the same data distribution, then distill.
4. Evaluate at `k/E` points, not just at `k=E`.

This needs GPU training even for a 1–4B model, but far less than pretraining.

## Level 2 — distilled MoE

The "combine Qwen and DeepSeek into a smaller MoE" idea is a teacher-student
project, not a weight merge.

### The tokenizer problem

See [`TOKENIZERS.md`](TOKENIZERS.md) for H-Net/BLT and why a tokenizer swap is
a new model, not a drop-in change.

Qwen and DeepSeek use different tokenizers. Logit distillation requires a
shared vocabulary (or a projection between vocabularies, which is expensive and
lossy). The practical options are:

- **Sequence-level distillation:** generate high-quality outputs/reasoning
  traces from both teachers, filter them, and train the student on its own
  tokenizer. Works across families; loses the teacher's full distribution.
- **Same-family logit distillation:** use a Qwen teacher for logit KD and a
  DeepSeek teacher for reasoning data/style. Simpler and cheaper.
- **Vocabulary projection:** only worth it if you have a large compute budget
  and a strong reason to keep both teachers' distributions.

### A realistic plan

1. **Pick the student shape first.** A 1.5–4B dense student upcycled to 8–16
   experts, or a 4B-active / 16–32B-total MoE. Active parameters drive latency;
   total parameters drive memory and streaming cost.
2. **Build the data.** 5–20B tokens of deduplicated text, code, math, and
   teacher-generated reasoning traces. Decontaminate against the eval sets.
3. **Distill dense first.** Train the student on teacher outputs and KL where
   the tokenizer allows. This is cheaper and gives a stronger initialization.
4. **Upcycle to MoE.** Copy/distill the dense FFN into experts, add a router,
   train with load balancing, then short continued training.
5. **Preference/instruction tuning.** SFT on curated teacher traces, then DPO
   or ORPO if the budget allows.
6. **Evaluate honestly.** Held-out perplexity, MMLU/GPQA, GSM8K/MATH,
   HumanEval, instruction-following, safety, and serving metrics (tok/s, VRAM,
   cold/warm expert I/O). Compare to the dense student and to same-size
   open models, not just to the teachers.

### Compute reality

Teacher inference is often the largest line item. Generating 10B tokens from a
frontier teacher is expensive even at API prices; a 30B-class teacher is more
practical. Training a 1–4B student on 10B tokens is tens to hundreds of
GPU-hours; an 8×H100 node for days to weeks is a reasonable order of magnitude,
depending on sequence length, MoE size, and how much data you generate.

### Licensing

Qwen3 is Apache-2.0. DeepSeek model licenses have use and redistribution terms;
teacher-generated data may inherit restrictions. Check the current license
before publishing weights or a large synthetic dataset. Do not redistribute
teacher outputs without reading the terms.

## The pragmatic next step

Before distilling anything, run an **existing** MoE model with expert streaming.
Qwen3-30B-A3B (30B total, 3B active) is the natural target for a 16 GB GPU:
the whole model does not fit, but a top-k expert stream does. That exercises
the real serving problem — router traces, expert cache hit rates, cold I/O,
prefetch — without a training budget. It also produces a much stronger
inference-infrastructure artifact than a small distillation run would.

## Resume framing

For platform/backend/inference-infra roles, the format-level and streaming work
is already the strongest evidence: it shows GGUF internals, memory scheduling,
and measurement discipline. A distillation project is a different job family
(ML training/research); it belongs on the resume only after it has a held-out
evaluation and a reproducible artifact.
