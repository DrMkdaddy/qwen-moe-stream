#!/usr/bin/env bash
# Publish the converted GGUF weights and the model card to Hugging Face.
#
# Requires an authenticated Hugging Face CLI:
#   pip install -U huggingface_hub
#   hf auth login
#
# Usage:
#   ./scripts/publish_hf.sh [repo-id]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${1:-DrMkdaddy/qwen-moe-stream}"
MODEL_DIR="${MODEL_DIR:-/home/noorm/models/moe}"

if command -v hf >/dev/null 2>&1; then
  HF=(hf)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF=(huggingface-cli)
else
  echo "huggingface_hub CLI not found. Run: pip install -U huggingface_hub" >&2
  exit 2
fi

"${HF[@]}" auth whoami >/dev/null

"${HF[@]}" repo create "$REPO" --repo-type model --exist-ok

"${HF[@]}" upload "$REPO" "$ROOT/hf/README.md" README.md --repo-type model
for name in \
  Qwen3-4B-MoE8-exact-f16.gguf \
  Qwen3-4B-MoE8-exact-f16.manifest.json \
  Qwen3-4B-MoE8-ridge6-f16.gguf \
  Qwen3-4B-MoE8-ridge6-f16.manifest.json; do
  "${HF[@]}" upload "$REPO" "$MODEL_DIR/$name" "$name" --repo-type model
done

echo "published https://huggingface.co/$REPO"
