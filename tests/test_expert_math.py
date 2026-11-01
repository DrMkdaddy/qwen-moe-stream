#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from moe_convert import build_experts, importance_permutation  # noqa: E402


class ExpertMathTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.d, self.ff, self.e = 8, 16, 4
        self.ff_e = self.ff // self.e
        self.gate = rng.normal(0, 0.2, (self.ff, self.d)).astype(np.float32)
        self.up = rng.normal(0, 0.2, (self.ff, self.d)).astype(np.float32)
        self.down = rng.normal(0, 0.2, (self.d, self.ff)).astype(np.float32)
        self.perm = importance_permutation(self.gate, self.up, self.down)
        self.h = rng.normal(0, 1, (3, self.d)).astype(np.float32)

    def dense(self):
        g = self.h @ self.gate.T
        a = (g / (1 + np.exp(-g))) * (self.h @ self.up.T)
        return a, a @ self.down.T

    def test_partition_is_permutation(self):
        self.assertEqual(sorted(self.perm.tolist()), list(range(self.ff)))

    def test_full_fanout_is_dense_sum(self):
        g, u, d = build_experts(self.gate, self.up, self.down, self.perm,
                                self.e, self.ff_e, 1.0)
        a, y = self.dense()
        y_experts = np.zeros_like(y)
        for e in range(self.e):
            idx = self.perm[e * self.ff_e:(e + 1) * self.ff_e]
            ae = a[:, idx]
            y_experts += ae @ d[e].astype(np.float32).T
        np.testing.assert_allclose(y_experts, y, rtol=2e-3, atol=2e-3)

    def test_uniform_router_and_scaled_down_is_exact(self):
        g, u, d = build_experts(self.gate, self.up, self.down, self.perm,
                                self.e, self.ff_e, float(self.e))
        a, y = self.dense()
        y_moe = np.zeros_like(y)
        for e in range(self.e):
            idx = self.perm[e * self.ff_e:(e + 1) * self.ff_e]
            gate_e = g[e].astype(np.float32)
            up_e = u[e].astype(np.float32)
            ge = self.h @ gate_e.T
            ae = (ge / (1 + np.exp(-ge))) * (self.h @ up_e.T)
            y_moe += (1.0 / self.e) * (ae @ d[e].astype(np.float32).T)
        np.testing.assert_allclose(y_moe, y, rtol=3e-3, atol=3e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
