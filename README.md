# qwen-moe-stream

Reformat the local **Qwen3-4B-Instruct-2507** GGUF into an 8-expert
Mixture-of-Experts GGUF that llama.cpp can load, and stream the experts from
disk with bounded memory.

The conversion is exact by construction: the dense FFN is the sum of its hidden
neurons, so partitioning those neurons into 8 disjoint experts and selecting
all 8 reproduces the original function numerically. A router and top-k mode are
also included; top-k is a lossy codec and is measured as such.

Code: https://github.com/DrMkdaddy/qwen-moe-stream  
Weights: https://huggingface.co/StanzaAPI/qwen-moe-stream  
Model card source: [`hf/README.md`](hf/README.md); upload script: `scripts/publish_hf.sh`.

## Results

Source: `/home/noorm/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` (2.4 GB Q4_K_M).
Output: `/home/noorm/models/moe/Qwen3-4B-MoE8-exact-f16.gguf` (6.25 GB, F16 FFN,
all other tensors copied bit-for-bit).

Held-out perplexity, 512-token context, CPU (`llama-perplexity`):

| model | PPL | vs dense | FFN bytes / token |
|---|---:|---:|---:|
| dense Qwen3-4B (Q4_K_M) | **17.6203** | 1.000× | 0 (dense tensor) |
| MoE8 exact, all 8 experts | **17.6157** | 0.9997× | 5.379 GB |
| MoE8 top-6, ridge router | **20.8920** | 1.186× | 4.037 GB |

Logit-level comparison of exact MoE against the dense model on the same
held-out text (`--kl-divergence`):

| metric | value |
|---|---:|
| mean KL divergence | **0.001350** |
| same top token | **96.47%** |
| mean `ln(PPL_moe / PPL_dense)` | −0.000278 |
| PPL ratio | 0.99972 |

The small residual is F16 round-off plus the difference between Q4_K integer
dot products and dequantized F16 weights; the graph is mathematically the same.

Streaming, four tokens through all 36 layers (`src/stream_exec.py`, real expert
tensors, `madvise` after each expert):

| schedule | peak RSS | bytes touched |
|---|---:|---:|
| resident, no eviction | 5.93 GB | 21.5 GB |
| streamed, evict after use | **0.78 GB** | 21.5 GB |

The exact model's FFN alone is 5.38 GB and the whole file is 6.25 GB, so the
streaming path holds roughly one expert per layer instead of the model.

## Why top-k is lossy here

The experts are **slices of a dense FFN**, not experts trained to be complete
updates. `scripts/oracle_probe.py` gives the ceiling any router could reach for
this partition: at `k=6/8`, even an oracle that picks the best experts and fits
optimal weights leaves 0.16–0.39 relative L2 error in middle layers. That
compounds over 36 layers. The measured top-6 point is **20.8920 PPL at 75% of FFN bytes** (1.19×
the dense PPL); the per-layer FFN reconstruction error is 0.258. The top-k
files are useful as a quality/IO frontier and as a test of the streaming
layout; `--mode exact` is the faithful model.
`docs/DESIGN.md` develops this argument and the alternatives (trained experts,
shared expert, activation clustering).

## Layout

```
src/moe_convert.py       dense GGUF -> qwen3moe GGUF (+ stream manifest)
src/fit_ridge_router.py  closed-form log-energy router + per-layer scale
src/train_router.py      gradient router (negative result, kept for comparison)
src/stream_exec.py       mmap expert executor with byte and RSS accounting
src/dump_hidden.cpp      llama.cpp callback tool: dump ffn_inp per layer
scripts/oracle_probe.py  upper bound of top-k reconstruction
scripts/cluster_probe.py activation-clustering variant
scripts/router_diag.py   routing precision vs subset approximation
tests/test_expert_math.py exactness of the partition algebra
docs/DESIGN.md           the abstraction, the conservation law, the schedule
docs/FUTURE.md           format-level MoE -> upcycled MoE -> distilled MoE
```

Expert tensor shapes follow llama.cpp's `qwen3moe` convention:

```
blk.N.ffn_gate_inp.weight   [d_model, E]              f32 router
blk.N.ffn_gate_exps.weight  [d_model, d_expert, E]    f16
blk.N.ffn_up_exps.weight    [d_model, d_expert, E]    f16
blk.N.ffn_down_exps.weight  [d_expert, d_model, E]    f16
```

The `E` axis is outermost in the file, so each expert is one contiguous block
and can be paged in independently.

## Reproduce

```bash
cd /home/noorm/projects/qwen-moe-stream
SRC=/home/noorm/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
OUT=/home/noorm/models/moe

# 1. exact 8-expert conversion (about 40 s, 6.25 GB output)
python3 src/moe_convert.py --src $SRC --dst $OUT/Qwen3-4B-MoE8-exact-f16.gguf \
  --experts 8 --mode exact

# 2. dump calibration hidden states with the llama.cpp build in this repo
L=/home/noorm/llama.cpp/src-b11050
g++ -O2 -std=c++17 src/dump_hidden.cpp -I$L/include -I$L/ggml/include \
  -L$L/build/bin -lllama -lggml -lggml-base -lggml-cpu \
  -Wl,-rpath,$L/build/bin -o build_dump_hidden
./build_dump_hidden $SRC calib/corpus.txt calib/hidden 1024

# 3. fit the sparse router (top-6)
python3 src/fit_ridge_router.py --top-k 6 --out results/router_ridge_top6.npz
python3 src/moe_convert.py --src $SRC --dst $OUT/Qwen3-4B-MoE8-ridge6-f16.gguf \
  --experts 8 --mode router --router results/router_ridge_top6.npz --top-k 6

# 4. evaluate
llama-perplexity -m $OUT/Qwen3-4B-MoE8-exact-f16.gguf -f calib/heldout.txt \
  -c 512 --chunks 1 -ngl 0 -t 8

# 5. streaming memory test
python3 src/stream_exec.py --tokens 4 --layers 36 --mode exact --evict --skip-dense
```

`./build_dump_hidden` and the models are build artifacts and are not committed.

## Caveats

- The source is a Q4_K_M file. The FFN experts are written F16 after
  dequantization; non-FFN tensors are copied raw. Converting from an F16 source
  would remove the Q4 round-off in the equivalence test.
- Only the FFN is MoE-ified. Attention, embeddings, and the LM head stay dense,
  which is why the file is 6.25 GB rather than 5.4 GB.
- Streaming reads are logical expert bytes. On a warm page cache the kernel
  serves them from RAM; the RSS test uses `madvise` to make the schedule
  explicit. Cold NVMe throughput is a separate measurement.
- `top-k` is not presented as quality-preserving. The shipped ridge top-6 point is 20.8920 PPL (1.19× dense) at 75% of FFN bytes; the earlier gradient router was much worse.
