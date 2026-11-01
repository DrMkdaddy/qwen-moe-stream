#!/usr/bin/env python3
"""Measure how much of the dense FFN a top-k neuron partition can reconstruct.

This is the feasibility probe behind the router design: if even the oracle
subset cannot reconstruct the FFN update, a sparse conversion is not worth
training.  It uses the calibration hidden states dumped by dump_hidden.cpp.
"""
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
TOKENS = 256


def rms_norm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + eps) * w


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def main():
    r = GGUFReader(MODEL)
    by = {t.name: t for t in r.tensors}
    eps = float(r.fields["qwen3.attention.layer_norm_rms_epsilon"].contents())
    rows = []
    for layer in [0, 12, 24, 35]:
        H = np.fromfile(HIDDEN / f"layer_{layer:02d}.f32", dtype=np.float32).reshape(-1, 2560)[:TOKENS]
        fn = by[f"blk.{layer}.ffn_norm.weight"].data.astype(np.float32)
        hn = rms_norm(H, fn, eps).astype(np.float32)
        gate = dequant(by[f"blk.{layer}.ffn_gate.weight"])
        up = dequant(by[f"blk.{layer}.ffn_up.weight"])
        down = dequant(by[f"blk.{layer}.ffn_down.weight"])
        perm = importance_permutation(gate, up, down)
        slices = [perm[e * 1216:(e + 1) * 1216] for e in range(E)]

        g = hn @ gate.T
        act = (g / (1.0 + np.exp(-g))) * (hn @ up.T)
        ys = np.stack([act[:, s] @ down[:, s].T for s in slices], axis=1)  # [N,E,d]
        y = ys.sum(axis=1)

        energy = np.mean(ys * ys, axis=2)  # [N,E]
        out = {}
        for k in (2, 4, 6, 7, 8):
            if k == 8:
                out[k] = 0.0
                continue
            sel = np.argsort(-energy, axis=1)[:, :k]
            # oracle least-squares weights on the selected experts, sums to 1
            yh = np.zeros_like(y)
            for n in range(len(hn)):
                A = ys[n, sel[n]].T  # [d,k]
                w, *_ = np.linalg.lstsq(A, y[n], rcond=None)
                yh[n] = A @ w
            rel = np.linalg.norm(yh - y, axis=1) / (np.linalg.norm(y, axis=1) + 1e-9)
            yf = np.zeros_like(y)
            for n in range(len(hn)):
                A = ys[n, sel[n]].T
                w, *_ = np.linalg.lstsq(A, y[n], rcond=None)
                yf[n] = A @ w
            # uniform weights, rescaled to keep the expected output magnitude
            yu = np.stack([ys[n, sel[n]].mean(axis=0) for n in range(len(hn))])
            relu = np.linalg.norm((E / k) * yu - y, axis=1) / (np.linalg.norm(y, axis=1) + 1e-9)
            # random subset, same rescaling, to show how much the selection helps
            rsel = np.stack([np.random.default_rng(n).choice(E, k, replace=False) for n in range(len(hn))])
            yr = np.stack([ys[n, rsel[n]].mean(axis=0) for n in range(len(hn))])
            relr = np.linalg.norm((E / k) * yr - y, axis=1) / (np.linalg.norm(y, axis=1) + 1e-9)
            # softmax of log energy
            logits = np.log(energy + 1e-9)
            z = np.where(np.isin(np.arange(E)[None, :], sel), logits, -1e9)
            w = softmax(z)
            yw = np.einsum("ne,ned->nd", w, ys)
            relw = np.linalg.norm(yw - y, axis=1) / (np.linalg.norm(y, axis=1) + 1e-9)
            out[k] = (float(rel.mean()), float(relu.mean()), float(relw.mean()), float(relr.mean()))
        rows.append((layer, out))
        print(f"layer {layer}: " + ", ".join(
            f"k={k} oracleLS={v[0]:.3f} uniform*E/k={v[1]:.3f} softE={v[2]:.3f} random*E/k={v[3]:.3f}"
            if isinstance(v, tuple) else f"k={k} exact={v:.3f}" for k, v in out.items()))

    print("\nA relative error near 1.0 means the selected neurons contribute almost")
    print("nothing useful; near 0.0 means the FFN update is reconstructed.  This is")
    print("the ceiling any router can reach for a fixed neuron partition.")


if __name__ == "__main__":
    main()
