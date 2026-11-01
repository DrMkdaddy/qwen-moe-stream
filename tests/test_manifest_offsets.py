#!/usr/bin/env python3
"""Check that the manifest's per-expert byte ranges are the real blocks."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/home/noorm/models/moe/Qwen3-4B-MoE8-exact-f16.gguf")
MANIFEST = MODEL.with_suffix(".manifest.json")

sys.path.insert(0, "/home/noorm/llama.cpp/src-b11050/gguf-py")
from gguf import GGUFReader  # noqa: E402


@unittest.skipUnless(MODEL.exists() and MANIFEST.exists(), "converted model not built")
class ManifestOffsetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = json.loads(MANIFEST.read_text())
        cls.r = GGUFReader(MODEL)
        cls.t = {x.name: x for x in cls.r.tensors}

    def test_expert_blocks_are_contiguous(self):
        eb = self.m["expert_ffn_length"] * self.m["d_model"] * 2
        for layer in self.m["layers"]:
            i = layer["layer"]
            for kind in ("gate", "up", "down"):
                t = self.t[f"blk.{i}.ffn_{kind}_exps.weight"]
                starts = [int(t.data_offset) + e * eb for e in range(self.m["experts"])]
                # offsets are exactly one expert apart
                self.assertEqual(starts, self.m["streams"][i][kind])
                self.assertEqual(int(t.n_bytes), eb * self.m["experts"])

    def test_router_is_small_and_present(self):
        for layer in self.m["layers"]:
            i = layer["layer"]
            t = self.t[f"blk.{i}.ffn_gate_inp.weight"]
            self.assertEqual(list(t.shape)[-1], self.m["experts"])
            self.assertEqual(t.tensor_type.name, "F32")

    def test_file_size_matches_manifest(self):
        self.assertEqual(MODEL.stat().st_size, self.m["model_bytes"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
