#!/usr/bin/env python3
"""Reformat a dense Qwen3 GGUF into a streamable Qwen3-MoE GGUF.

The conversion is an exact graph rewrite when every expert is selected:

    ffn(x) = sum_j W2[:, j] * silu(W1[j] @ x) * (W3[j] @ x)
           = sum_e  (1/E) * (E * W2_e) * silu(W1_e x) * (W3_e x)

where the hidden neurons are partitioned into E disjoint expert slices and the
router logits are all zero (softmax -> uniform 1/E).  The E factor is folded
into each expert's down projection.  With top-k < E the same file becomes a
lossy, sparse MoE; the router weights then decide which slices are read.

The output is a normal llama.cpp `qwen3moe` GGUF.  Every expert is a contiguous
[d_model, d_expert] (or [d_expert, d_model]) block, so a memory-mapped reader
only pages in the experts selected by the router.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import gguf
except ModuleNotFoundError:  # use the llama.cpp checkout that built the runtime
    for candidate in (
        Path("/home/noorm/llama.cpp/src-b11050/gguf-py"),
        Path.home() / "llama.cpp" / "gguf-py",
    ):
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            break
    import gguf

from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter


SKIP_TENSOR_PREFIXES = (
    "blk.",
)


def is_ffn_matrix(name: str) -> bool:
    return name.endswith((".ffn_gate.weight", ".ffn_up.weight", ".ffn_down.weight"))


def dequant(t) -> np.ndarray:
    """Return a float32 array in PyTorch/HF orientation ([out, in]).

    gguf-py stores a tensor's bytes with ne0 fastest; ReaderTensor.data is
    reshaped to reversed(t.shape), i.e. rows are output dims.  That is the
    layout the conversion code and the expert slicers expect.
    """
    if t.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
        return t.data.astype(np.float32, copy=False)
    return gguf.quants.dequantize(t.data, t.tensor_type).reshape(
        tuple(int(x) for x in reversed(t.shape)))


def copy_metadata(reader: GGUFReader, writer: GGUFWriter, experts: int, top_k: int,
                  expert_ffn: int) -> None:
    src_arch = reader.fields["general.architecture"].contents()
    prefix = f"{src_arch}."
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        if key == "general.architecture":
            continue
        if key.startswith("quantize."):
            continue
        if key in ("general.file_type", "general.quantization_version"):
            continue
        if key.startswith(prefix):
            out_key = "qwen3moe." + key[len(prefix):]
        elif key.startswith("tokenizer.") or key.startswith("general."):
            out_key = key
        else:
            # Unknown architecture-specific key: keep it only if it is generic.
            out_key = key

        value = field.contents()
        types = field.types
        if isinstance(value, np.generic):
            value = value.item()
        if types[0] == gguf.GGUFValueType.ARRAY:
            if isinstance(value, list):
                value = [v.item() if isinstance(v, np.generic) else v for v in value]
            writer.add_key_value(out_key, value, types[0], types[1])
        else:
            writer.add_key_value(out_key, value, types[0])

    writer.add_file_type(int(gguf.LlamaFileType.MOSTLY_F16))
    writer.add_quantization_version(2)
    writer.add_key_value("qwen3moe.expert_count", experts, gguf.GGUFValueType.UINT32)
    writer.add_key_value("qwen3moe.expert_used_count", top_k, gguf.GGUFValueType.UINT32)
    writer.add_key_value("qwen3moe.expert_feed_forward_length", expert_ffn,
                         gguf.GGUFValueType.UINT32)

    # Provenance, visible in `llama-cli --verbose` / gguf-dump.
    writer.add_key_value("qwen-moe.source", src_arch, gguf.GGUFValueType.STRING)
    writer.add_key_value("qwen-moe.layout", "contiguous-expert-major",
                         gguf.GGUFValueType.STRING)


def importance_permutation(gate: np.ndarray, up: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Order neurons by an inexpensive weight-only importance proxy.

    The permutation is a no-op for the exact (all-expert) model.  For sparse
    routing it makes every expert carry a similar share of the FFN's weight
    energy instead of letting arbitrary contiguous blocks do so.
    """
    # gate/up: [d_ff, d_model]; down: [d_model, d_ff]
    gate_energy = np.mean(np.square(gate, dtype=np.float32), axis=1)
    up_energy = np.mean(np.square(up, dtype=np.float32), axis=1)
    down_energy = np.mean(np.square(down, dtype=np.float32), axis=0)
    importance = np.sqrt((gate_energy + up_energy) * down_energy + 1e-20)
    return np.argsort(-importance, kind="stable")


def build_experts(gate: np.ndarray, up: np.ndarray, down: np.ndarray, perm: np.ndarray,
                  experts: int, expert_ffn: int, down_scale: float) -> tuple[np.ndarray, ...]:
    d_model = gate.shape[1]
    g = np.empty((experts, expert_ffn, d_model), dtype=np.float16)
    u = np.empty_like(g)
    dn = np.empty((experts, d_model, expert_ffn), dtype=np.float16)
    for e in range(experts):
        idx = perm[e * expert_ffn:(e + 1) * expert_ffn]
        g[e] = gate[idx, :]
        u[e] = up[idx, :]
        dn[e] = down[:, idx] * down_scale
    return g, u, dn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--mode", choices=("exact", "router"), default="exact")
    ap.add_argument("--router", type=Path, default=None,
                    help="npz with per-layer arrays router.{layer} of shape [experts, d_model]")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--down-scale", type=float, default=None,
                    help="explicit down-projection scale; default E for exact, 1 for router")
    ap.add_argument("--manifest", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.time()
    reader = GGUFReader(args.src)
    arch = reader.fields["general.architecture"].contents()
    if arch != "qwen3":
        raise SystemExit(f"expected a dense qwen3 GGUF, got {arch!r}")

    d_model = int(reader.fields["qwen3.embedding_length"].contents())
    n_ff = int(reader.fields["qwen3.feed_forward_length"].contents())
    n_layer = int(reader.fields["qwen3.block_count"].contents())
    experts = args.experts
    if n_ff % experts:
        raise SystemExit(f"feed_forward_length {n_ff} is not divisible by {experts} experts")
    expert_ffn = n_ff // experts
    top_k = args.top_k if args.top_k is not None else (experts if args.mode == "exact" else max(1, experts // 4))
    if not 1 <= top_k <= experts:
        raise SystemExit("--top-k must be in [1, experts]")
    down_scale = args.down_scale
    if down_scale is None:
        down_scale = float(experts) if args.mode == "exact" else 1.0

    routers: dict[int, np.ndarray] = {}
    layer_scales: dict[int, float] = {}
    if args.mode == "router":
        if args.router is None:
            raise SystemExit("--mode router requires --router")
        z = np.load(args.router)
        for i in range(n_layer):
            key = f"router.{i}"
            if key not in z:
                raise SystemExit(f"router file is missing {key}")
            routers[i] = z[key].astype(np.float32)
            if routers[i].shape != (experts, d_model):
                raise SystemExit(f"{key} has shape {routers[i].shape}, expected {(experts, d_model)}")
            if f"scale.{i}" in z:
                layer_scales[i] = float(z[f"scale.{i}"][0])

    writer = GGUFWriter(args.dst, "qwen3moe", use_temp_file=True)
    copy_metadata(reader, writer, experts, top_k, expert_ffn)

    ffn_tensors = {t.name for t in reader.tensors if is_ffn_matrix(t.name)}
    expected = {
        f"blk.{i}.ffn_{kind}.weight"
        for i in range(n_layer) for kind in ("gate", "up", "down")
    }
    missing = expected - ffn_tensors
    if missing:
        raise SystemExit(f"source is missing FFN tensors, e.g. {sorted(missing)[:3]}")

    manifest = {
        "source": str(args.src),
        "output": str(args.dst),
        "architecture": "qwen3moe",
        "experts": experts,
        "top_k": top_k,
        "expert_ffn_length": expert_ffn,
        "feed_forward_length": n_ff,
        "d_model": d_model,
        "n_layer": n_layer,
        "mode": args.mode,
        "down_scale": down_scale,
        "tensors": [],
        "layers": [],
    }

    # Non-FFN tensors are copied bit-for-bit, including their quantization.
    for t in reader.tensors:
        if is_ffn_matrix(t.name):
            continue
        if t.name.endswith(".ffn_gate_inp.weight"):
            continue
        if t.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            writer.add_tensor(t.name, t.data)
        else:
            writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
        manifest["tensors"].append({
            "name": t.name, "nbytes": int(t.n_bytes), "type": t.tensor_type.name,
        })

    by_name = {t.name: t for t in reader.tensors}
    for layer in range(n_layer):
        gt = by_name[f"blk.{layer}.ffn_gate.weight"]
        ut = by_name[f"blk.{layer}.ffn_up.weight"]
        dt = by_name[f"blk.{layer}.ffn_down.weight"]
        gate = dequant(gt)          # [d_model, n_ff]
        up = dequant(ut)            # [d_model, n_ff]
        down = dequant(dt)          # [n_ff, d_model]

        perm = importance_permutation(gate, up, down)
        this_scale = layer_scales.get(layer, down_scale)
        g, u, dn = build_experts(gate, up, down, perm, experts, expert_ffn, this_scale)

        router = np.zeros((experts, d_model), dtype=np.float32)
        if layer in routers:
            router = routers[layer]

        writer.add_tensor(f"blk.{layer}.ffn_gate_inp.weight", router)
        writer.add_tensor(f"blk.{layer}.ffn_gate_exps.weight", g)
        writer.add_tensor(f"blk.{layer}.ffn_up_exps.weight", u)
        writer.add_tensor(f"blk.{layer}.ffn_down_exps.weight", dn)

        manifest["layers"].append({
            "layer": layer,
            "down_scale": float(this_scale),
            "permutation_sha256": __import__("hashlib").sha256(perm.astype(np.int32).tobytes()).hexdigest(),
            "permutation": perm.astype(np.int32).tolist(),
        })
        print(f"layer {layer + 1:02d}/{n_layer}: experts built "
              f"({(time.time() - t0):.1f}s)", flush=True)

        del gate, up, down, g, u, dn

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()

    # Locate each expert's byte range in the finished file.
    out = GGUFReader(args.dst)
    out_by_name = {t.name: t for t in out.tensors}
    expert_bytes = expert_ffn * d_model * 2  # f16
    streams = []
    for layer in range(n_layer):
        layer_stream = {"layer": layer}
        for kind, per_expert in (
            ("gate", expert_bytes),
            ("up", expert_bytes),
            ("down", expert_bytes),
        ):
            t = out_by_name[f"blk.{layer}.ffn_{kind}_exps.weight"]
            base = int(t.data_offset)
            layer_stream[kind] = [base + e * per_expert for e in range(experts)]
        router_t = out_by_name[f"blk.{layer}.ffn_gate_inp.weight"]
        layer_stream["router"] = int(router_t.data_offset)
        layer_stream["router_bytes"] = int(router_t.n_bytes)
        layer_stream["expert_bytes"] = expert_bytes
        streams.append(layer_stream)
    manifest["streams"] = streams
    manifest["model_bytes"] = args.dst.stat().st_size
    manifest["ffn_bytes_per_token_exact"] = n_layer * 3 * experts * expert_bytes
    manifest["ffn_bytes_per_token_topk"] = n_layer * (3 * top_k * expert_bytes + experts * d_model * 4)

    mp = args.manifest or args.dst.with_suffix(".manifest.json")
    mp.write_text(json.dumps(manifest))
    print(f"wrote {args.dst} ({args.dst.stat().st_size / 1e9:.2f} GB)")
    print(f"manifest {mp} ({mp.stat().st_size / 1e6:.1f} MB)")
    print(f"FFN bytes/token exact={manifest['ffn_bytes_per_token_exact']/1e6:.1f} MB "
          f"top{top_k}={manifest['ffn_bytes_per_token_topk']/1e6:.1f} MB")


if __name__ == "__main__":
    main()
