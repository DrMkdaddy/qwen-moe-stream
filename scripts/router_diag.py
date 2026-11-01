#!/usr/bin/env python3
"""Diagnose routing vs subset approximation on held-out calibration tokens."""
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
E = 8
NTR = 384


def rms_norm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + eps) * w


def softmax(z, mask):
    z = np.where(mask, z, -1e30)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    r = GGUFReader(MODEL)
    by = {t.name: t for t in r.tensors}
    eps = float(r.fields["qwen3.attention.layer_norm_rms_epsilon"].contents())
    for layer in (0, 12, 24, 35):
        H = np.fromfile(HIDDEN / f"layer_{layer:02d}.f32", dtype=np.float32).reshape(-1, 2560)
        hn = rms_norm(H, by[f"blk.{layer}.ffn_norm.weight"].data.astype(np.float32), eps)
        gate = dequant(by[f"blk.{layer}.ffn_gate.weight"])
        up = dequant(by[f"blk.{layer}.ffn_up.weight"])
        down = dequant(by[f"blk.{layer}.ffn_down.weight"])
        perm = importance_permutation(gate, up, down)
        g = hn @ gate.T
        act = (g / (1.0 + np.exp(-g))) * (hn @ up.T)
        ys = np.stack([act[:, perm[e * 1216:(e + 1) * 1216]] @ down[:, perm[e * 1216:(e + 1) * 1216]].T for e in range(E)], 1)
        y = ys.sum(1)
        en = np.mean(ys * ys, 2) + 1e-12
        tr, te = slice(0, NTR), slice(NTR, None)

        t = np.log(en[tr])
        t = (t - t.mean(1, keepdims=True)) / (t.std(1, keepdims=True) + 1e-6)
        K = hn[tr] @ hn[tr].T + 10 * np.eye(NTR, dtype=np.float32)
        R = np.linalg.solve(K, t).T @ hn[tr]
        z = hn[te] @ R.T

        for k in (6, 7):
            oracle = np.argsort(-en[te], 1)[:, :k]
            rout = np.argsort(-z, 1)[:, :k]
            prec = np.mean([len(set(a) & set(b)) / k for a, b in zip(oracle, rout)])
            # router selection, softmax weights with temperature beta
            best = {}
            for beta in (0.05, 0.25, 1.0):
                sel = rout
                mask = np.zeros_like(z, bool)
                np.put_along_axis(mask, sel, True, 1)
                w = softmax(beta * z, mask)
                yh = np.einsum("ne,ned->nd", w, ys[te])
                alpha = np.sum(yh * y[te]) / np.sum(yh * yh)
                rel = np.linalg.norm(alpha * yh - y[te], 1) / np.linalg.norm(y[te], 1)
                best[beta] = (rel, alpha)
            # oracle selection, softmax(log energy)
            mask = np.zeros_like(z, bool)
            np.put_along_axis(mask, oracle, True, 1)
            w = softmax(np.log(en[te]), mask)
            yh = np.einsum("ne,ned->nd", w, ys[te])
            alpha = np.sum(yh * y[te]) / np.sum(yh * yh)
            rel_o = np.linalg.norm(alpha * yh - y[te], 1) / np.linalg.norm(y[te], 1)
            print(f"layer {layer} k={k} precision={prec:.3f} oracleLS_rel={rel_o:.4f} "
                  + " ".join(f"router(b={b})={best[b][0]:.4f}" for b in best))


if __name__ == "__main__":
    main()
