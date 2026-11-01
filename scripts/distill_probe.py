#!/usr/bin/env python3
"""Can a 1/8-width neuron slice be re-fitted to reproduce the full FFN?

For expert e we keep its gate/up slice A_e and refit only its down projection
D_e so that D_e A_e(x) approximates the *full* dense FFN output.  If this works,
one small expert is a distilled copy of the dense FFN and top-k becomes useful
without gradient training.
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
N_TRAIN = 384
N_TEST = 128


def rms_norm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + eps) * w


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
        y = act @ down.T

        tr = slice(0, N_TRAIN)
        te = slice(N_TRAIN, N_TRAIN + N_TEST)
        A_all = act[tr]
        y_tr = y[tr]
        y_te = y[te]
        errs_orig, errs_fit, errs_fit_ens = [], [], []
        for e in range(E):
            idx = perm[e * 1216:(e + 1) * 1216]
            A = A_all[:, idx]
            lam = 1.0
            G = A.T @ A + lam * np.eye(A.shape[1], dtype=np.float32)
            D = np.linalg.solve(G, A.T @ y_tr).T  # [d, 1216]
            Ate = act[te][:, idx]
            pred = Ate @ D.T
            errs_fit.append(np.linalg.norm(pred - y_te, axis=1) / (np.linalg.norm(y_te, axis=1) + 1e-9))
            errs_orig.append(np.linalg.norm(Ate @ down[:, idx].T - y_te, axis=1) / (np.linalg.norm(y_te, axis=1) + 1e-9))
        # ensemble = mean of all fitted experts
        preds = []
        for e in range(E):
            idx = perm[e * 1216:(e + 1) * 1216]
            Atr = act[tr][:, idx]
            G = Atr.T @ Atr + np.eye(Atr.shape[1], dtype=np.float32)
            D = np.linalg.solve(G, Atr.T @ y_tr).T
            preds.append(act[te][:, idx] @ D.T)
        ens = np.mean(preds, axis=0)
        rel_ens = np.linalg.norm(ens - y_te, axis=1) / (np.linalg.norm(y_te, axis=1) + 1e-9)
        print(f"layer {layer}: original slice err={np.mean(errs_orig):.3f} "
              f"refit err={np.mean(errs_fit):.3f} refit-ensemble err={np.mean(rel_ens):.3f}")


if __name__ == "__main__":
    main()
