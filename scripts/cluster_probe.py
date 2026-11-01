#!/usr/bin/env python3
"""Does activation-coherent neuron clustering make top-k reconstruction viable?"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from moe_convert import dequant  # noqa: E402

sys.path.insert(0, "/home/noorm/llama.cpp/src-b11050/gguf-py")
from gguf import GGUFReader  # noqa: E402

MODEL = Path("/home/noorm/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
HIDDEN = ROOT / "calib" / "hidden"
E = 8
N = 256


def rms_norm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + eps) * w


def kmeans(X, k, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            if np.any(lab == j):
                C[j] = X[lab == j].mean(0)
    return lab


def main():
    r = GGUFReader(MODEL)
    by = {t.name: t for t in r.tensors}
    eps = float(r.fields["qwen3.attention.layer_norm_rms_epsilon"].contents())
    for layer in (0, 12, 24):
        H = np.fromfile(HIDDEN / f"layer_{layer:02d}.f32", dtype=np.float32).reshape(-1, 2560)[:N]
        hn = rms_norm(H, by[f"blk.{layer}.ffn_norm.weight"].data.astype(np.float32), eps)
        gate = dequant(by[f"blk.{layer}.ffn_gate.weight"])
        up = dequant(by[f"blk.{layer}.ffn_up.weight"])
        down = dequant(by[f"blk.{layer}.ffn_down.weight"])
        g = hn @ gate.T
        act = (g / (1.0 + np.exp(-g))) * (hn @ up.T)

        # cluster neurons on their (standardised) activation fingerprint
        X = act.T
        X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-6)
        lab = kmeans(X.astype(np.float32), E, iters=20)
        groups = [np.where(lab == e)[0] for e in range(E)]
        sizes = [len(x) for x in groups]
        ys = np.stack([act[:, g] @ down[:, g].T for g in groups], axis=1)
        y = ys.sum(1)
        energy = np.mean(ys * ys, 2)
        res = []
        for k in (2, 4, 6):
            sel = np.argsort(-energy, 1)[:, :k]
            err = 0.0
            for n in range(len(hn)):
                A = ys[n, sel[n]].T
                w, *_ = np.linalg.lstsq(A, y[n], rcond=None)
                err += np.linalg.norm(A @ w - y[n]) / (np.linalg.norm(y[n]) + 1e-9)
            res.append(err / len(hn))
        print(f"layer {layer}: sizes={sizes} oracleLS " +
              " ".join(f"k={k}:{v:.3f}" for k, v in zip((2, 4, 6), res)))


if __name__ == "__main__":
    main()
