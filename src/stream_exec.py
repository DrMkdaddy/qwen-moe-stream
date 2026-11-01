#!/usr/bin/env python3
"""A minimal, mmap-based streaming executor for the converted Qwen MoE FFN.

It executes the real expert tensors in the output GGUF, one expert at a time,
and can evict each expert's pages after use.  That makes the memory claim
measurable: peak RSS is O(one expert) instead of O(the whole model), even
though the file is 6.25 GB.

It also compares the streamed FFN output to the dense source FFN for the same
calibration hidden states, so correctness and approximation error are visible.
"""
from __future__ import annotations

import argparse
import json
import mmap
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/noorm/llama.cpp/src-b11050/gguf-py")
from gguf import GGUFReader, GGMLQuantizationType  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from moe_convert import dequant  # noqa: E402

PAGE = mmap.PAGESIZE


class Pack:
    def __init__(self, path: Path):
        self.path = path
        self.reader = GGUFReader(path)
        self.t = {t.name: t for t in self.reader.tensors}
        self.fd = open(path, "rb")
        self.mm = mmap.mmap(self.fd.fileno(), 0, access=mmap.ACCESS_READ)
        f = self.reader.fields
        arch = f["general.architecture"].contents()
        self.arch = arch
        self.d_model = int(f["qwen3moe.embedding_length"].contents())
        self.n_ff = int(f["qwen3moe.expert_feed_forward_length"].contents())
        self.experts = int(f["qwen3moe.expert_count"].contents())
        self.top_k = int(f["qwen3moe.expert_used_count"].contents())
        self.n_layer = int(f["qwen3moe.block_count"].contents())
        self.expert_bytes = self.n_ff * self.d_model * 2
        self.router_bytes = self.experts * self.d_model * 4

    def arr(self, name: str, dtype, shape, byte_off: int = 0, count: int | None = None):
        t = self.t[name]
        off = int(t.data_offset) + byte_off
        if count is None:
            count = int(t.n_elements)
        a = np.frombuffer(self.mm, dtype=dtype, count=count, offset=off)
        return a.reshape(shape)

    def f32(self, name: str):
        return self.arr(name, np.float32, tuple(reversed(self.t[name].shape)))

    def expert(self, layer: int, kind: str, e: int):
        name = f"blk.{layer}.ffn_{kind}_exps.weight"
        t = self.t[name]
        off = int(t.data_offset) + e * self.expert_bytes
        if kind in ("gate", "up"):
            shape = (self.n_ff, self.d_model)
        else:
            shape = (self.d_model, self.n_ff)
        return np.frombuffer(self.mm, dtype=np.float16, count=self.n_ff * self.d_model,
                             offset=off).reshape(shape)

    def evict(self, layer: int, kind: str, e: int):
        t = self.t[f"blk.{layer}.ffn_{kind}_exps.weight"]
        off = int(t.data_offset) + e * self.expert_bytes
        start = off - (off % PAGE)
        end = off + self.expert_bytes
        end = min(len(self.mm), end + (-end) % PAGE)
        self.mm.madvise(mmap.MADV_DONTNEED, start, end - start)

    def close(self):
        self.mm.close()
        self.fd.close()


def rms_norm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + eps) * w


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def act(gate, up, h):
    g = h @ gate.T
    return (g / (1.0 + np.exp(-g))) * (h @ up.T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moe", type=Path, default=Path("/home/noorm/models/moe/Qwen3-4B-MoE8-exact-f16.gguf"))
    ap.add_argument("--src", type=Path, default=Path("/home/noorm/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"))
    ap.add_argument("--hidden", type=Path, default=ROOT / "calib" / "hidden")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--mode", choices=("exact", "topk"), default="exact")
    ap.add_argument("--evict", action="store_true", help="madvise each expert out after use")
    ap.add_argument("--skip-dense", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pack = Pack(args.moe)
    src = GGUFReader(args.src)
    sby = {t.name: t for t in src.tensors}
    eps = float(src.fields["qwen3.attention.layer_norm_rms_epsilon"].contents())
    E = pack.experts
    K = pack.top_k if args.mode == "topk" else E

    errs, dense2, stream2 = [], 0.0, 0.0
    bytes_read = 0
    t0 = time.time()
    for layer in range(args.layers):
        H = np.fromfile(args.hidden / f"layer_{layer:02d}.f32", dtype=np.float32).reshape(-1, pack.d_model)[:args.tokens]
        fn = pack.f32(f"blk.{layer}.ffn_norm.weight")
        hn = rms_norm(H, fn, eps)
        router = pack.arr(f"blk.{layer}.ffn_gate_inp.weight", np.float32, (E, pack.d_model))
        z = hn @ router.T
        if args.mode == "topk":
            sel = np.argsort(-z, axis=1)[:, :K]
        else:
            sel = np.tile(np.arange(E), (len(hn), 1))
        bytes_read += len(hn) * (pack.router_bytes + K * 3 * pack.expert_bytes)

        y = np.zeros((len(hn), pack.d_model), dtype=np.float32)
        for row in range(len(hn)):
            h = hn[row:row + 1]
            chosen = sel[row]
            zz = z[row:row + 1, chosen]
            w = softmax(zz)
            acc = np.zeros((1, pack.d_model), dtype=np.float32)
            for j, e in enumerate(chosen):
                g = pack.expert(layer, "gate", int(e))
                u = pack.expert(layer, "up", int(e))
                d = pack.expert(layer, "down", int(e))
                a = act(g.astype(np.float32), u.astype(np.float32), h)
                acc += float(w[0, j]) * (a @ d.astype(np.float32).T)
                if args.evict:
                    pack.evict(layer, "gate", int(e))
                    pack.evict(layer, "up", int(e))
                    pack.evict(layer, "down", int(e))
            y[row] = acc[0]

        if not args.skip_dense:
            g = dequant(sby[f"blk.{layer}.ffn_gate.weight"])
            u = dequant(sby[f"blk.{layer}.ffn_up.weight"])
            d = dequant(sby[f"blk.{layer}.ffn_down.weight"])
            gg = hn @ g.T
            a = (gg / (1.0 + np.exp(-gg))) * (hn @ u.T)
            yd = a @ d.T
            err = np.linalg.norm(y - yd, axis=1) / (np.linalg.norm(yd, axis=1) + 1e-9)
            errs.append(float(err.mean()))
            dense2 += float(np.mean(yd * yd))
            stream2 += float(np.mean(y * y))
            del g, u, d, gg, a, yd
        print(f"layer {layer + 1:02d}/{args.layers}: {time.time() - t0:.1f}s "
              f"bytes={bytes_read / 1e6:.1f}MB rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f}GB",
              flush=True)

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    result = {
        "mode": args.mode,
        "evict": args.evict,
        "tokens": args.tokens,
        "layers": args.layers,
        "model_bytes": args.moe.stat().st_size,
        "bytes_touched": bytes_read,
        "peak_rss_gb": rss,
        "seconds": time.time() - t0,
        "mean_relative_error": float(np.mean(errs)) if errs else None,
        "energy_ratio": (stream2 / dense2) if dense2 else None,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
    # Numpy views still reference the mmap; the OS reclaims it at exit.


if __name__ == "__main__":
    main()
