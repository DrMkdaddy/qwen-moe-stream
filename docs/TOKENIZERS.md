# Tokenizer-free alternatives, and what H-Net actually claims

Short version: **H-Net is not a better tokenizer. It is an architecture that
removes the tokenizer** by learning content-dependent chunk boundaries over raw
bytes, jointly with the rest of the model. It is promising, but it is not a
drop-in replacement for a pretrained Qwen checkpoint.

## The landscape

| approach | unit | fixed vocab? | notes |
|---|---|---|---|
| BPE / WordPiece / Unigram | subword | yes | what Qwen, Llama, DeepSeek use; fast, mature, biased toward frequent languages/domains |
| ByT5 / CANINE | byte or character | no | simple byte models; long sequences, weaker compute efficiency |
| MegaByte / MambaByte | byte patches | partly | patch-based byte modeling; architecture-specific |
| BLT (Byte Latent Transformer) | dynamic byte patches | no | entropy-based patching; byte-level with a learned patch boundary predictor |
| H-Net | learned chunks, hierarchical | no | dynamic chunking + encoder/main/decoder hierarchy; trained end-to-end |

## What H-Net is

Cartesia / Goomba Lab's H-Net ("Dynamic Chunking for End-to-End Hierarchical
Sequence Modeling", Hwang, Wang, Gu; arXiv:2507.07955) replaces the
tokenizer–LM–detokenizer pipeline with:

1. an encoder over bytes,
2. a learned, differentiable chunker that decides where boundaries go,
3. a main network that runs on the resulting chunks,
4. a dechunker/decoder back to bytes.

Multiple stages can be nested, so the model can build characters → words →
phrases. The boundaries are context-dependent and can change at inference time
("in-context tokenization").

## The numbers, without the Twitter compression

The 2025 paper reports that a compute- and data-matched one-stage byte H-Net
outperforms a strong BPE Transformer, and that iterating the hierarchy matches
a token-based Transformer of **twice its size**. It also reports large gains on
languages/code with weak tokenization heuristics and nearly 4x data efficiency
on DNA.

Cartesia's October 2026 scaling post reports, for its improved H-Net:

- one-stage H-Net: **1.28x** compute efficiency in-domain, ~**2x** on downstream
  evaluations and code;
- two-stage H-Net vs SmolLM3-3B: reaches the baseline's final average score
  after ~**1T tokens instead of ~11T**, at matched FLOPs/byte;
- the main network sees one position per **~7.6 bytes**, versus ~4.5 bytes per
  Llama-3 token — about **40% fewer positions**, not a universal 50% cut;
- the byte-level encoder/decoder stages consume **~43% of FLOPs**, so fewer main
  network positions does **not** mean total compute is halved.

So "cuts it in half" is directionally right for sequence positions/compression,
and the two-stage compute/data multiplier can be much larger than 2x — but the
outer networks eat a large share of the FLOPs. It is not a free 2x wall-clock
win, and it is not a tokenizer you can bolt onto an existing model.

## Why it is not a drop-in for Qwen

A pretrained Qwen model's embedding matrix and LM head are tied to its 151k BPE
vocabulary. Replacing the tokenizer means:

- new byte/chunk embeddings and a new output head,
- retraining or continued pretraining on a large corpus,
- re-tuning the context length, KV cache, and serving stack,
- new kernels/format support. llama.cpp/GGUF has no H-Net graph; H-Net
  checkpoints are PyTorch and depend on Mamba/CUDA components.

You cannot swap the tokenizer on `Qwen3-4B` and keep the weights. You would be
training a new model, optionally distilling the old one into it.

## What is practical today

- **Domain vocabulary expansion** on a pretrained model: add tokens for code,
  medical, legal, or non-English text, resize embeddings, continue training.
  Cheaper than a new architecture, but not free.
- **Use a model with a better tokenizer for the target language/domain** rather
  than forcing Qwen's tokenizer onto it.
- **Experiment with H-Net small checkpoints** (`goombalab/hnet` on GitHub;
  `hnet_1stage_L`, `hnet_2stage_L`, etc. on Hugging Face) to see the chunking
  behavior. Do not expect a GGUF/llama.cpp path.
- **Distill into a byte-level student only after** a same-tokenizer dense
  distillation works; teacher logits are token-level, so a byte-level student
  needs sequence-level distillation or vocabulary alignment.

## Relation to this repo

This project's MoE conversion is about FFN memory layout and expert streaming.
It is orthogonal to tokenization: the sequence length, attention cost, and KV
cache are unchanged. An H-Net-style model would shorten the main-network
sequence and change the serving problem, but it would also require a new model,
new kernels, and a new evaluation cycle. Keep the two projects separate on a
resume: one is inference infrastructure, the other is modeling research.
