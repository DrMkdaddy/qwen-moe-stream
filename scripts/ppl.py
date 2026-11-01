#!/usr/bin/env python3
"""Run llama-perplexity on a list of models and write a JSON table."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN = Path("/home/noorm/llama.cpp/llama-b11050/llama-perplexity")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--text", type=Path, default=ROOT / "calib" / "heldout.txt")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "ppl.json")
    args = ap.parse_args()

    rows = []
    for model in args.models:
        cmd = [str(args.bin), "-m", str(model), "-f", str(args.text),
               "-c", str(args.ctx), "--chunks", "1", "-ngl", "0", "-t", "8"]
        print("+", " ".join(cmd), flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = p.stdout + p.stderr
        m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", out)
        if not m:
            print(out[-2000:], file=sys.stderr)
            raise SystemExit(f"could not parse PPL for {model}")
        ppl, err = float(m.group(1)), float(m.group(2))
        rows.append({"model": model.name, "path": str(model), "ppl": ppl, "stderr": err})
        print(f"{model.name}: PPL={ppl:.4f} +/- {err:.4f}", flush=True)

    base = rows[0]["ppl"]
    for r in rows:
        r["ratio_vs_first"] = r["ppl"] / base
    args.out.write_text(json.dumps({"text": str(args.text), "ctx": args.ctx, "models": rows}, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
