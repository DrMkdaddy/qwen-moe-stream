#!/usr/bin/env bash
# Reproduce the exact conversion, the sparse router, and the streaming test.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${SRC:-/home/noorm/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"
OUT="${OUT:-/home/noorm/models/moe}"
LLAMA="${LLAMA:-/home/noorm/llama.cpp/src-b11050}"
mkdir -p "$OUT"

cd "$ROOT"

# 1. exact MoE (all experts selected; the graph is the dense FFN)
python3 src/moe_convert.py --src "$SRC" --dst "$OUT/Qwen3-4B-MoE8-exact-f16.gguf" \
  --experts 8 --mode exact

# 2. calibration hidden states + sparse router
if [ ! -f build_dump_hidden ]; then
  g++ -O2 -std=c++17 src/dump_hidden.cpp \
    -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
    -L"$LLAMA/build/bin" -lllama -lggml -lggml-base -lggml-cpu \
    -Wl,-rpath,"$LLAMA/build/bin" -o build_dump_hidden
fi
cat /home/noorm/projects/local-inference-stack/README.md \
    /home/noorm/projects/PROJECT_SPEC.md > calib/corpus.txt
./build_dump_hidden "$SRC" calib/corpus.txt calib/hidden 1024

python3 src/fit_ridge_router.py --top-k 6 --out results/router_ridge_top6.npz
python3 src/moe_convert.py --src "$SRC" --dst "$OUT/Qwen3-4B-MoE8-ridge6-f16.gguf" \
  --experts 8 --mode router --router results/router_ridge_top6.npz --top-k 6

# 3. evidence
python3 tests/test_expert_math.py
python3 tests/test_manifest_offsets.py
python3 src/stream_exec.py --tokens 4 --layers 36 --mode exact --evict --skip-dense \
  --out results/stream_exact_nodense.json
python3 src/stream_exec.py --tokens 4 --layers 36 --mode exact --skip-dense \
  --out results/stream_exact_resident.json

"$LLAMA/../llama-b11050/llama-perplexity" -m "$OUT/Qwen3-4B-MoE8-exact-f16.gguf" \
  -f calib/heldout.txt -c 512 --chunks 1 -ngl 0 -t 8

echo "done"
