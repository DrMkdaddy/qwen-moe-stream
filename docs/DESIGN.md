# Dense → MoE → stream: the abstraction

## 1. A dense FFN is already an MoE with one expert

For a SwiGLU FFN with hidden width `F`:

```
FFN(x) = W_down · ( silu(W_gate x) ⊙ (W_up x) )
       = Σ_j W_down[:, j] · a_j(x)
```

Every hidden neuron `j` contributes a rank-1 vector `W_down[:, j] · a_j(x)`. The
dense model is the **sum over all F neurons**. Nothing about the computation
requires that sum to be evaluated as one matrix multiply.

Partition the neurons into `E` disjoint slices `S_e` of width `F/E`:

```
y_e(x) = W_down[:, S_e] · a_{S_e}(x)
FFN(x) = Σ_e y_e(x)
```

This is an MoE in the literal sense: there are `E` experts and a mixture. The
router is the constant uniform distribution `1/E`; all experts are selected.
That is the **exact** conversion. It is not an approximation, a distillation,
or a new model. It is the same function with a different memory schedule.

## 2. What top-k buys and what it costs

The usual MoE bargain is to select only `k < E` experts:

```
FFN(x) ≈ Σ_{e ∈ top-k} w_e(x) · y_e(x),   Σ w_e = 1
```

For a model **trained** as an MoE, each `y_e` is a complete expert that has
learned to produce a useful full-size update. For a **partitioned dense**
model, each `y_e` is only a partial sum. Dropping an expert drops a real piece
of the FFN output. There is no free lunch:

> Routing sparsity has to be paid for by approximation error, by training, or
> by redundant computation. A file format cannot create capacity.

The oracle probe in `scripts/oracle_probe.py` measures the ceiling of the
lossy conversion. Even choosing the best `k` experts by their true contribution
energy and fitting optimal per-token weights leaves a relative L2 error of
roughly 0.16–0.39 at `k=6` in the middle layers. The errors compound over 36 layers. A closed-form ridge router with a
calibrated per-layer scale gets 20.892 PPL at k=6 (1.19× dense) while reading
75% of the FFN bytes; the gradient-trained router was much worse (10²–10⁵ PPL).
The sparse mode is therefore shipped as an **explicit lossy mode**, not as a
replacement for the exact model.

The useful engineering result is that the same file, the same tensor names,
and the same runtime support both modes. Sparsity is a knob, not a fork.

## 3. Streaming is a schedule, not a model

A GGUF tensor with dimensions `{d_model, d_expert, E}` is laid out expert-major:
expert `e` occupies one contiguous `d_model × d_expert` block. That is the
entire streaming contract:

- parse the tensor index once (a few MB);
- `mmap` the file;
- for each layer, read only the `k` expert blocks selected by the router;
- optionally `madvise(MADV_DONTNEED)` those pages after the partial sum.

Peak resident memory is then `O(k · expert_bytes)` plus the KV cache and the
non-FFN weights, not `O(model_bytes)`. `src/stream_exec.py` implements this
against the real GGUF. On four tokens × 36 layers:

| schedule | peak RSS | bytes touched |
|---|---:|---:|
| all experts resident (no eviction) | 5.93 GB | 21.5 GB |
| all experts, evict after each expert | **0.78 GB** | 21.5 GB |

The model file is 6.25 GB. The exact model is fully streamable.

## 4. Why the router is trained the way it is

The router input is the post-RMSNorm hidden state `h`, exactly what llama.cpp
feeds to the MoE block. For each layer we fit a linear map `R` (the GGUF
`ffn_gate_inp.weight`) to predict the log energy of each expert's contribution
on calibration hidden states. The ridge solution is closed-form:

```
R = Tᵀ (H Hᵀ + λI)⁻¹ H
```

where `T` is the standardized log-energy target. The routing decision is the
top-k of `hRᵀ`. The gate weights are kept nearly uniform by baking a small
temperature into `R`, because partial sums are not trained experts and sharp
token-dependent convex weights amplify variance. A per-layer scalar `α`,
folded into the expert down projections, restores the output magnitude.

`src/fit_ridge_router.py` is the simple, stable version. `src/train_router.py`
is the gradient-trained variant; it is kept as a negative result: optimizing the
top-k reconstruction directly made routing worse, because hard selection gives
no useful gradient to the experts that were not selected.

## 5. What would make top-k actually good

The honest options are:

1. **Train the experts** (sparse upcycling or distillation) so each expert is a
   complete update rather than a slice. This is the standard route and needs
   GPU training.
2. **Keep k close to E.** Streaming saves `1 - k/E` of FFN bytes; the quality
   curve is steep. At `k=E` the conversion is exact.
3. **Use a shared expert** with a trained residual. The Qwen3-MoE graph in
   llama.cpp does not have a shared-expert branch; adding one is a runtime
   change, not a file-format change.
4. **Change the partition objective.** Activation-coherent clustering helps
   slightly (`scripts/cluster_probe.py`) but does not beat the information
   limit: a subset of a sum cannot reconstruct the sum without structure in
   the activations.

The conversion therefore exposes `--mode exact` as the default and
`--mode router` as a measured approximation.
