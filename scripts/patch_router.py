#!/usr/bin/env python3
"""Scale the router tensors of a converted MoE GGUF in place (temperature sweep).

The router is stored as a separate F32 tensor per layer, so changing its scale
does not require rewriting the expert weights.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/noorm/llama.cpp/src-b11050/gguf-py")
from gguf import GGUFReader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--save", type=Path, default=None, help="save original routers to npz")
    ap.add_argument("--restore", type=Path, default=None, help="restore routers from npz")
    args = ap.parse_args()

    r = GGUFReader(args.model)
    routers = {t.name: t for t in r.tensors if t.name.endswith("ffn_gate_inp.weight")}
    if args.restore:
        z = np.load(args.restore)
        data = {k: z[k] for k in z.files}
    elif args.save:
        data = {n: np.array(t.data, dtype=np.float32) for n, t in routers.items()}
        np.savez_compressed(args.save, **data)
        print(f"saved {len(data)} routers to {args.save}")
        return
    else:
        data = {}

    if args.scale is None:
        raise SystemExit("need --scale or --save/--restore")

    with open(args.model, "r+b") as f:
        mm = np.memmap(f, dtype=np.uint8, mode="r+")
        for name, tensor in routers.items():
            offset = int(tensor.data_offset)
            count = int(tensor.n_elements)
            if args.restore:
                arr = data[name].astype(np.float32)
            else:
                arr = np.frombuffer(mm, dtype=np.float32, count=count, offset=offset).copy()
                arr *= args.scale
            payload = arr.astype(np.float32).tobytes()
            mm[offset:offset + len(payload)] = np.frombuffer(payload, dtype=np.uint8)
        mm.flush()
        del mm
    print(f"patched {len(routers)} routers, scale={args.scale}")


if __name__ == "__main__":
    main()
