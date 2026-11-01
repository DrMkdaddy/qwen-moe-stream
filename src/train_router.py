#!/usr/bin/env python3
"""Fit a linear top-k router for the partitioned Qwen FFN.

The router sees the same post-RMSNorm hidden state llama.cpp feeds to the MoE
block.  It is trained on two signals:

  1. ridge regression to the log energy of each expert's contribution, which
     gives a good ranking from the first step;
  2. the reconstruction error of the actual top-k softmax mixture, which
     calibrates the gate weights and a per-layer output scale.

Run after `dump_hidden` has written calib/hidden/layer_XX.f32.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from moe_convert import dequant, importance_permutation  # noqa: E402

sys.path.insert(0, "/home/noorm/llama.cpp/src-b11050/gguf-py")
from gguf import GGUFReader  # noqa: E402

DEFAULT_MODEL = Path("/home/noorm/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
DEFAULT_HIDDEN = ROOT / "calib" / "hidden"


def rms_norm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + eps) * w


def softmax_masked(z, mask):
    z = np.where(mask, z, -1e30)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--hidden", type=Path, default=DEFAULT_HIDDEN)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "router_top6.npz")
    args = ap.parse_args()

    reader = GGUFReader(args.model)
    by = {t.name: t for t in reader.tensors}
    eps = float(reader.fields["qwen3.attention.layer_norm_rms_epsilon"].contents())
    n_layer = int(reader.fields["qwen3.block_count"].contents())
    d_model = int(reader.fields["qwen3.embedding_length"].contents())
    n_ff = int(reader.fields["qwen3.feed_forward_length"].contents())
    E = args.experts
    K = args.top_k
    expert_ffn = n_ff // E

    out = {}
    history = []
    for layer in range(n_layer):
        H = np.fromfile(args.hidden / f"layer_{layer:02d}.f32", dtype=np.float32)
        H = H.reshape(-1, d_model)[:args.tokens]
        fn = by[f"blk.{layer}.ffn_norm.weight"].data.astype(np.float32)
        hn = rms_norm(H, fn, eps).astype(np.float32)

        gate = dequant(by[f"blk.{layer}.ffn_gate.weight"])
        up = dequant(by[f"blk.{layer}.ffn_up.weight"])
        down = dequant(by[f"blk.{layer}.ffn_down.weight"])
        perm = importance_permutation(gate, up, down)
        slices = [perm[e * expert_ffn:(e + 1) * expert_ffn] for e in range(E)]

        g = hn @ gate.T
        act = (g / (1.0 + np.exp(-g))) * (hn @ up.T)
        ys = np.stack([act[:, s] @ down[:, s].T for s in slices], axis=1)  # [N,E,d]
        y = ys.sum(axis=1)

        energy = np.mean(ys * ys, axis=2) + 1e-12
        target = np.log(energy)
        target = (target - target.mean(1, keepdims=True)) / (target.std(1, keepdims=True) + 1e-6)

        # ridge regression from hidden state to standardised log-energy
        lam = 10.0
        Kmat = hn @ hn.T + lam * np.eye(len(hn), dtype=np.float32)
        A = np.linalg.solve(Kmat, target)
        R = (A.T @ hn).astype(np.float32)
        R *= 0.5 / (R.std() + 1e-9)

        alpha = 1.0
        mR = np.zeros_like(R)
        vR = np.zeros_like(R)
        ma = 0.0
        va = 0.0
        best = None
        for step in range(1, args.steps + 1):
            z = hn @ R.T
            sel = np.argsort(-z, axis=1)[:, :K]
            mask = np.zeros_like(z, dtype=bool)
            np.put_along_axis(mask, sel, True, axis=1)
            p = softmax_masked(z, mask)

            yhat = alpha * np.einsum("ne,ned->nd", p, ys)
            diff = yhat - y
            mse = np.mean(diff * diff)
            rank = np.mean((z - target) ** 2)
            loss = mse + 0.02 * rank

            # dL/dyhat, dL/dp, dL/dz
            dy = 2.0 * diff / diff.size
            g_e = np.einsum("nd,ned->ne", dy, ys)          # [N,E]
            pg = np.sum(p * g_e, axis=1, keepdims=True)
            dz = p * (g_e - pg)
            dz = np.where(mask, dz, 0.0)
            dz += 0.04 * (z - target) / z.size

            dR = dz.T @ hn
            dalpha = np.sum(dy * yhat) / (alpha + 1e-12)

            # Adam
            mR = 0.9 * mR + 0.1 * dR
            vR = 0.999 * vR + 0.001 * dR * dR
            mRh = mR / (1 - 0.9 ** step)
            vRh = vR / (1 - 0.999 ** step)
            R -= args.lr * mRh / (np.sqrt(vRh) + 1e-8)
            ma = 0.9 * ma + 0.1 * dalpha
            va = 0.999 * va + 0.001 * dalpha * dalpha
            alpha -= args.lr * (ma / (1 - 0.9 ** step)) / (np.sqrt(va / (1 - 0.999 ** step)) + 1e-8)
            alpha = float(np.clip(alpha, 0.05, 20.0))

            if best is None or mse < best[0]:
                best = (float(mse), R.copy(), float(alpha))
            if step % 50 == 0 or step == 1:
                print(f"layer {layer:02d} step {step:3d} mse={mse:.5f} rank={rank:.3f} "
                      f"alpha={alpha:.3f}", flush=True)

        _, R, alpha = best
        out[f"router.{layer}"] = R
        out[f"scale.{layer}"] = np.array([alpha], dtype=np.float32)
        out[f"topk.{layer}"] = np.array([K], dtype=np.int32)
        history.append({"layer": layer, "mse": best[0], "alpha": alpha})
        print(f"layer {layer:02d} done mse={best[0]:.5f} alpha={alpha:.3f}", flush=True)
        del gate, up, down, ys, y, act

    out["experts"] = np.array([E], dtype=np.int32)
    out["top_k"] = np.array([K], dtype=np.int32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    import json
    args.out.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
