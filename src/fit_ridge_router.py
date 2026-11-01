#!/usr/bin/env python3
"""Closed-form router: ridge regression to expert log-energy + calibrated scale.

This is deliberately simpler than gradient training.  The diagnostic in
scripts/router_diag.py shows the energy ranking survives a linear map, while
the softmax weights are best kept nearly uniform (small temperature).  A
per-layer scalar re-calibrates the output magnitude.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from moe_convert import dequant, importance_permutation  # noqa: E402

sys.path.insert(0, "/home/noorm/llama.cpp/src-b11050/gguf-py")
from gguf import GGUFReader  # noqa: E402

MODEL = Path("/home/noorm/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
HIDDEN = ROOT / "calib" / "hidden"


def rms_norm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + eps) * w


def softmax(z, mask):
    z = np.where(mask, z, -1e30)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--hidden", type=Path, default=HIDDEN)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "router_ridge_top6.npz")
    args = ap.parse_args()

    r = GGUFReader(args.model)
    by = {t.name: t for t in r.tensors}
    eps = float(r.fields["qwen3.attention.layer_norm_rms_epsilon"].contents())
    n_layer = int(r.fields["qwen3.block_count"].contents())
    d = int(r.fields["qwen3.embedding_length"].contents())
    n_ff = int(r.fields["qwen3.feed_forward_length"].contents())
    E, K = args.experts, args.top_k
    ff = n_ff // E

    out, history = {}, []
    for layer in range(n_layer):
        H = np.fromfile(args.hidden / f"layer_{layer:02d}.f32", dtype=np.float32).reshape(-1, d)[:args.tokens]
        hn = rms_norm(H, by[f"blk.{layer}.ffn_norm.weight"].data.astype(np.float32), eps)
        gate = dequant(by[f"blk.{layer}.ffn_gate.weight"])
        up = dequant(by[f"blk.{layer}.ffn_up.weight"])
        down = dequant(by[f"blk.{layer}.ffn_down.weight"])
        perm = importance_permutation(gate, up, down)
        g = hn @ gate.T
        act = (g / (1.0 + np.exp(-g))) * (hn @ up.T)
        ys = np.stack([act[:, perm[e * ff:(e + 1) * ff]] @ down[:, perm[e * ff:(e + 1) * ff]].T
                       for e in range(E)], 1)
        y = ys.sum(1)

        en = np.mean(ys * ys, 2) + 1e-12
        t = np.log(en)
        t = (t - t.mean(1, keepdims=True)) / (t.std(1, keepdims=True) + 1e-6)

        N = len(hn)
        lam = 10.0
        Kmat = hn @ hn.T + lam * np.eye(N, dtype=np.float32)
        R = (np.linalg.solve(Kmat, t).T @ hn).astype(np.float32)
        z0 = hn @ R.T
        R *= 1.0 / (z0.std() + 1e-9)  # unit-scale logits before temperature
        z = hn @ R.T

        sel = np.argsort(-z, 1)[:, :K]
        mask = np.zeros_like(z, bool)
        np.put_along_axis(mask, sel, True, 1)
        w = softmax(args.temperature * z, mask)
        R *= args.temperature  # bake the temperature into the GGUF router matrix
        yh = np.einsum("ne,ned->nd", w, ys)
        alpha = float(np.sum(yh * y) / (np.sum(yh * yh) + 1e-12))
        rel = np.linalg.norm(alpha * yh - y, 1) / np.linalg.norm(y, 1)
        out[f"router.{layer}"] = R
        out[f"scale.{layer}"] = np.array([alpha], dtype=np.float32)
        history.append({"layer": layer, "alpha": alpha, "relative_l1": float(rel),
                        "z_std": float(z.std())})
        print(f"layer {layer:02d} alpha={alpha:.3f} relL1={rel:.4f}", flush=True)
        del gate, up, down, ys, y, act

    out["experts"] = np.array([E], dtype=np.int32)
    out["top_k"] = np.array([K], dtype=np.int32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    args.out.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
