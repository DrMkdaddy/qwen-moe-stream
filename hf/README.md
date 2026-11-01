---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: llama.cpp
tags:
  - gguf
  - llama.cpp
  - moe
  - mixture-of-experts
  - qwen3
  - inference
  - streaming
---

# Qwen3-4B-Instruct-2507 — exact 8-expert MoE reformat

This repository contains a **format-level conversion** of
[Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
from a dense GGUF to an 8-expert `qwen3moe` GGUF that llama.cpp can load.

It is **not a trained MoE**. The dense FFN is the sum of its hidden neurons:

```
FFN(x) = Σ_j W_down[:, j] · silu(W_gate[j] @ x) · (W_up[j] @ x)
```

The conversion partitions the 9,728 hidden neurons into 8 disjoint slices of
1,216 neurons. With all 8 experts selected and a uniform router, the result is
the same function. The exact model matches the source within F16 round-off.

## Files

| file | size | description |
|---|---:|---|
| `Qwen3-4B-MoE8-exact-f16.gguf` | 6.25 GB | exact full-fanout model; **use this one** |
| `Qwen3-4B-MoE8-ridge6-f16.gguf` | 6.25 GB | lossy top-6 variant, 75% of FFN bytes |
| `Qwen3-4B-MoE8-exact-f16.manifest.json` | 2.1 MB | tensor offsets, expert byte ranges, permutation hashes |

Checksums are in [`results/CHECKSUMS.txt`](https://github.com/DrMkdaddy/qwen-moe-stream/blob/main/results/CHECKSUMS.txt).

The non-FFN tensors (attention, embeddings, LM head) are copied bit-for-bit
from the source Q4_K_M file. The FFN experts are written F16 after
dequantization. The source is Q4_K_M, so the F16 conversion carries that
quantization's round-off.

## Run with llama.cpp

```bash
llama-cli -m Qwen3-4B-MoE8-exact-f16.gguf \
  -p "The capital of France is" -n 16 -ngl 0 -t 8 --temp 0
```

`llama.cpp` build b11050 or newer is required for the `qwen3moe` graph. The
exact model is a normal dense-equivalent FFN, so it is slower than the Q4
source: the FFN is F16 and all 8 experts run.

## Measurements

Held-out text, 512-token context, CPU, `llama-perplexity`:

| model | PPL | ratio vs dense |
|---|---:|---:|
| dense Qwen3-4B Q4_K_M | 17.6203 | 1.000× |
| MoE8 exact (all 8 experts) | **17.6157** | 0.9997× |
| MoE8 top-6 ridge router | 20.8920 | 1.186× |

Logit comparison, exact vs dense:

| metric | value |
|---|---:|
| mean KL divergence | 0.001350 |
| same top token | 96.47% |
| PPL ratio | 0.99972 |

Streaming, 4 tokens × 36 layers, real expert tensors, `madvise` after use:

| schedule | peak RSS | bytes touched |
|---|---:|---:|
| resident, no eviction | 5.93 GB | 21.5 GB |
| streamed, evict after use | **0.78 GB** | 21.5 GB |

The 0.78 GB figure is the executor alone, not a full serving stack. The exact
model's FFN is 5.38 GB; the file is 6.25 GB.

## What top-6 means

The top-6 file is a **lossy** point on the quality/IO curve. It reads 75% of
the FFN bytes and raises PPL from 17.62 to 20.89. The experts are slices of a
dense FFN, not experts trained to produce complete updates; selecting a subset
drops real terms. The exact file is the faithful one. See the project README
for the oracle ceiling and the negative result on gradient-trained routers.

## Calibration and provenance

The ridge router was fitted on 384 hidden states from public project READMEs
and specification documents, dumped with a llama.cpp eval callback. No model
weights were trained. No private user data was used. The conversion scripts
and full methodology live at:

- GitHub: https://github.com/DrMkdaddy/qwen-moe-stream
- Source model: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507

## Limitations

- The exact model is a reformat, not a new model, and not a speedup. All 8
  experts run; only resident memory is improved by streaming.
- The top-6 model is lossy and should not be presented as equivalent.
- Only the FFN is MoE-ified. Attention, embeddings, and the LM head stay dense.
- Measurements are from one CPU host with `-ngl 0`; GPU offload will change
  the memory and throughput numbers.
- F16 experts from a Q4_K_M source are larger than the source file.

## License

The conversion code is MIT. The weights are derived from
Qwen/Qwen3-4B-Instruct-2507 and are distributed under **Apache-2.0**.
