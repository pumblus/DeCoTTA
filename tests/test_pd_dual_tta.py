"""Unit tests for `PrevalenceDeconfoundedDualTTA` invariants, serialization, and ablations."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from tta import (
    CANONICAL_PRESETS,
    PDDualConfig,
    PrevalenceDeconfoundedDualTTA,
    model_state_sha256,
    resolve_preset,
)


class _TinyEncoderDecoder(nn.Module):
    """Compact ResUNet-like test model with shallow (`stride <= 4`), deep encoder (`stride 8`), and decoder BNs."""

    def __init__(self, in_ch: int = 1, out_ch: int = 2) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, 8, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.c2 = nn.Conv2d(8, 12, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(12)
        self.c3 = nn.Conv2d(12, 16, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.up = nn.ConvTranspose2d(16, 12, 2, stride=2)
        self.bn_dec = nn.BatchNorm2d(24)
        self.head = nn.Conv2d(24, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.bn1(self.c1(x)))
        h2 = F.relu(self.bn2(self.c2(h1)))
        h3 = F.relu(self.bn3(self.c3(h2)))
        u = F.relu(self.bn_dec(torch.cat([self.up(h3), h2], dim=1)))
        u = F.interpolate(u, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.head(u)


def _make_trained_tiny_model(in_ch: int, out_ch: int, seed: int = 7) -> _TinyEncoderDecoder:
    torch.manual_seed(seed)
    m = _TinyEncoderDecoder(in_ch=in_ch, out_ch=out_ch)
    m.train()
    for _ in range(6):
        _ = m(torch.randn(4, in_ch, 64, 64))
    with torch.no_grad():
        m.head.bias.zero_()
        m.head.bias[0] = 0.2
    m.eval()
    return m


class TestPDDualTTA(unittest.TestCase):
    def test_zero_mutation_cached_pass0_and_summary_roundtrip(self) -> None:
        dev = torch.device("cpu")
        model = _make_trained_tiny_model(in_ch=1, out_ch=3, seed=11)
        sha0 = model_state_sha256(model)

        torch.manual_seed(12)
        src_imgs = [torch.randn(1, 64, 64) for _ in range(10)]
        tgt_imgs = [torch.randn(1, 64, 64) * 1.6 + 0.8 for _ in range(5)]

        adapter = PrevalenceDeconfoundedDualTTA.fit_from_source(
            model=model, train_imgs=src_imgs, device=dev, output_mode="multiclass_softmax", config=PDDualConfig()
        )
        self.assertEqual(model_state_sha256(model), sha0)

        with tempfile.TemporaryDirectory() as td:
            summary_pt = Path(td) / "source_summary.pt"
            adapter.save_source_summary(summary_pt)
            reloaded = PrevalenceDeconfoundedDualTTA.from_source_summary(model, summary_pt, dev)

        adapter.reset_stream()
        reloaded.reset_stream()
        for im in tgt_imgs:
            out_a, _ = adapter.predict_single(im, device=dev)
            out_b, _ = reloaded.predict_single(im, device=dev)
            self.assertTrue(torch.equal(out_a, out_b))
        self.assertEqual(model_state_sha256(model), sha0)

        adapter.reset_stream()
        replay_ad = adapter.with_config(PDDualConfig())
        replay_ad.reset_stream()
        for im in tgt_imgs:
            body0 = torch.ones((64, 64), dtype=torch.bool)
            im_nopad = im.clone()
            adapter_no_strip = adapter.with_config(PDDualConfig(strip_padding=False))
            break
        adapter.reset_stream()
        st_clone = PrevalenceDeconfoundedDualTTA.fit_from_source(
            model=model, train_imgs=src_imgs, device=dev, output_mode="multiclass_softmax", config=PDDualConfig()
        )
        from tta import detect_body_mask

        adapter.reset_stream()
        st_clone.reset_stream()
        for im in tgt_imgs:
            logits_direct, _ = adapter.predict_single(im, device=dev)
            b0 = detect_body_mask(im, strip_padding=True)
            lp0, ops, dg0 = st_clone.shallow_tta.forward_pass0(im.unsqueeze(0), b0, b0, dev)
            logits_cached, _ = st_clone.predict_from_cached_pass0(im, b0, lp0, ops, dg0)
            self.assertTrue(torch.equal(logits_direct, logits_cached))

    def test_multilabel_sigmoid_and_all_presets(self) -> None:
        dev = torch.device("cpu")
        model = _make_trained_tiny_model(in_ch=3, out_ch=2, seed=21)
        sha0 = model_state_sha256(model)
        torch.manual_seed(22)
        src_imgs = [torch.rand(3, 64, 64) for _ in range(8)]
        tgt = torch.rand(3, 64, 64) * 0.7 + 0.2

        base = PrevalenceDeconfoundedDualTTA.fit_from_source(
            model=model, train_imgs=src_imgs, device=dev, output_mode="multilabel_sigmoid", config=PDDualConfig()
        )
        for name in CANONICAL_PRESETS:
            cfg = resolve_preset(name)
            if name == "weight_only_sf":
                ad = PrevalenceDeconfoundedDualTTA.from_checkpoint_only(model, (3, 64, 64), dev, "multilabel_sigmoid", cfg)
            else:
                ad = base.with_config(cfg)
            ad.reset_stream()
            logits, diag = ad.predict_single(tgt, device=dev)
            self.assertEqual(tuple(logits.shape), (1, 2, 64, 64))
            self.assertTrue(torch.isfinite(logits).all())
            self.assertIn("deep_ran", diag)
        self.assertEqual(model_state_sha256(model), sha0)


if __name__ == "__main__":
    unittest.main()
