# Result artifacts

| file | what it is |
|---|---|
| `dense.logits` | dense-model logits on `calib/heldout.txt` (74 MB, `llama-perplexity --save-all-logits`) |
| `structure.txt` | GGUF architecture, expert count, tensor shapes |
| `stream_exact_nodense.json` | exact MoE, 4 tokens × 36 layers, evict each expert |
| `stream_exact_resident.json` | same, no eviction (pages accumulate) |
| `stream_exact.json` | exact MoE with dense reference comparison |
| `stream_ridge6.json` | shipped top-6 sparse run with dense reference comparison |
| `stream_gradient_top6.json` | earlier gradient-router top-6 run (negative result) |
| `router_ridge_top6.npz` | closed-form router, top-6 |
| `router_ridge_top7.npz` | closed-form router, top-7 |
| `router_*_top*.history.json` | per-layer alpha and calibration error |
| `greedy/` | greedy generation transcripts, dense vs exact |
| `ppl.json` | perplexity table for all models |
| `CHECKSUMS.txt` | SHA-256 of the published GGUF weights |

The `router_gradient_*` files and `stream_gradient_top6.json` are the
gradient-trained variant kept as a negative result (see `docs/DESIGN.md`).
The shipped sparse router is `router_ridge_top6.npz`.
