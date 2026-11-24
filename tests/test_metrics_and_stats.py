"""Unit tests for Dice, exact Euclidean HD95/ASSD, and paired statistical tests."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from evaluate import (
    binary_dice_pct,
    compute_hd95_assd,
    holm_bonferroni,
    optic_dice_pct,
    paired_wilcoxon_p,
    stratified_bootstrap_ci,
)


class TestMetricsAndStats(unittest.TestCase):
    def test_hd95_assd_bbox_matches_full_grid_2d_and_3d(self) -> None:
        rng = np.random.default_rng(123)
        for _ in range(6):
            a = np.zeros((96, 96), dtype=np.bool_)
            b = np.zeros((96, 96), dtype=np.bool_)
            r0, c0 = rng.integers(0, 60, size=2)
            r1, c1 = rng.integers(0, 60, size=2)
            a[r0 : r0 + 25, c0 : c0 + 20] = True
            b[r1 : r1 + 20, c1 : c1 + 25] = True
            h1, s1, _, _ = compute_hd95_assd(a, b, use_bbox=True)
            h2, s2, _, _ = compute_hd95_assd(a, b, use_bbox=False)
            self.assertAlmostEqual(h1, h2, places=9)
            self.assertAlmostEqual(s1, s2, places=9)

        a3 = np.zeros((10, 32, 32), dtype=np.bool_)
        b3 = np.zeros((10, 32, 32), dtype=np.bool_)
        a3[1:6, 4:18, 4:18] = True
        b3[2:7, 6:20, 5:19] = True
        h1, s1, _, _ = compute_hd95_assd(a3, b3, use_bbox=True)
        h2, s2, _, _ = compute_hd95_assd(a3, b3, use_bbox=False)
        self.assertAlmostEqual(h1, h2, places=9)
        self.assertAlmostEqual(s1, s2, places=9)

    def test_empty_mask_edge_cases(self) -> None:
        z = np.zeros((8, 16, 16), dtype=np.bool_)
        m = np.zeros((8, 16, 16), dtype=np.bool_)
        m[2:5, 4:10, 4:10] = True
        self.assertEqual(binary_dice_pct(z, z), 100.0)
        self.assertEqual(binary_dice_pct(z, m), 0.0)
        h, a, emp_fail, gt_emp = compute_hd95_assd(z, m)
        diag = math.sqrt(8**2 + 16**2 + 16**2)
        self.assertAlmostEqual(h, diag, places=9)
        self.assertAlmostEqual(a, diag, places=9)
        self.assertTrue(emp_fail)
        self.assertFalse(gt_emp)

    def test_optic_dice_and_stats(self) -> None:
        p = np.ones((2, 16, 16), dtype=np.bool_)
        gd = np.ones((16, 16), dtype=np.bool_)
        gc = np.zeros((16, 16), dtype=np.bool_)
        gc[4:12, 4:12] = True
        dd, cd = optic_dice_pct(p, gd, gc)
        self.assertAlmostEqual(dd, 100.0, places=4)
        self.assertGreater(cd, 30.0)

        a = [80.0, 82.0, 79.0, 85.0, 83.0, 81.0, 84.0, 78.0]
        b = [75.0, 76.0, 74.0, 79.0, 77.0, 75.0, 78.0, 73.0]
        p_val = paired_wilcoxon_p(a, b)
        self.assertLess(p_val, 0.02)
        adj = holm_bonferroni({"m1": p_val, "m2": 0.5})
        self.assertLess(adj["m1"], adj["m2"])
        pt, lo, hi = stratified_bootstrap_ci({"d1": a, "d2": b}, n_boot=500, seed=1)
        self.assertLessEqual(lo, pt)
        self.assertLessEqual(pt, hi)


if __name__ == "__main__":
    unittest.main()
