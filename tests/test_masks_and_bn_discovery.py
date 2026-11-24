"""Unit tests for anatomical masking, erosion replay, and BatchNorm partition discovery."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from models import ResUNet, SicTTAUNet, VPTTAResUNet
from tta import SpatialErosionReplay, detect_body_mask, discover_bn_partition


class TestMasksAndBNDiscovery(unittest.TestCase):
    def test_mri_padding_stripping(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(1, 64, 64)
        x[:, :6, :] = -2.0
        x[:, -6:, :] = -2.0
        body = detect_body_mask(x, strip_padding=True)
        self.assertEqual(tuple(body.shape), (64, 64))
        self.assertFalse(bool(body[0, 32]))
        self.assertFalse(bool(body[-1, 32]))
        self.assertTrue(bool(body[32, 32]))

    def test_rgb_fundus_rim_stripping(self) -> None:
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij")
        disc = (xx * xx + yy * yy) <= 0.75
        rgb = torch.zeros(3, 64, 64)
        rgb[0][disc] = 0.8
        rgb[1][disc] = 0.5
        rgb[2][disc] = 0.3
        body = detect_body_mask(rgb, strip_padding=True)
        self.assertFalse(bool(body[0, 0]))
        self.assertTrue(bool(body[32, 32]))

    def test_bn_partition_and_erosion_replay_across_backbones(self) -> None:
        specs = [
            (ResUNet(in_channels=1, num_classes=4), (1, 1, 64, 64)),
            (VPTTAResUNet(in_channels=3, num_classes=2, return_features=True), (1, 3, 64, 64)),
            (SicTTAUNet(in_channels=1, num_classes=4), (1, 1, 64, 64)),
        ]
        for model, shape in specs:
            dummy = torch.zeros(shape)
            sh, d_enc, d_dec = discover_bn_partition(model, dummy, max_stride=4)
            self.assertGreater(len(sh), 0)
            self.assertGreater(len(d_enc) + len(d_dec), 0)
            for nm, _, st in sh:
                self.assertLessEqual(st, 4)
                self.assertNotIn("in1", nm)

            replay = SpatialErosionReplay.trace_once(model, sh, dummy)
            body0 = torch.ones((shape[2], shape[3]), dtype=torch.bool)
            body0[0, :] = False
            masks = replay.replay(body0, torch.device("cpu"))
            self.assertEqual(len(masks), len(sh))


if __name__ == "__main__":
    unittest.main()
