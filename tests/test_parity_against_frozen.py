"""Bitwise parity verification against the archived frozen `pd_dual_tta.py` implementation."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import unittest

import torch

_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from tests.test_pd_dual_tta import _make_trained_tiny_model
from tta import PDDualConfig, PrevalenceDeconfoundedDualTTA, resolve_preset


def _find_frozen_pair() -> tuple[Path | None, Path | None]:
    sh_env = os.environ.get("DECOTTA_FROZEN_SHALLOW_PY")
    v2_env = os.environ.get("DECOTTA_FROZEN_V2_PY")
    if not sh_env or not v2_env:
        return None, None
    sh, v2 = Path(sh_env), Path(v2_env)
    if sh.is_file() and v2.is_file():
        return sh, v2
    return None, None


_FROZEN_SHALLOW, _FROZEN_V2 = _find_frozen_pair()


def _load_frozen_v2_module():
    assert _FROZEN_SHALLOW is not None and _FROZEN_V2 is not None
    sh_dir = str(_FROZEN_SHALLOW.parent)
    if sh_dir not in sys.path:
        sys.path.insert(0, sh_dir)
    spec = importlib.util.spec_from_file_location("_frozen_pd_dual_tta", str(_FROZEN_V2))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_frozen_pd_dual_tta"] = mod
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(
    _FROZEN_SHALLOW is not None and _FROZEN_V2 is not None,
    "Set DECOTTA_FROZEN_SHALLOW_PY and DECOTTA_FROZEN_V2_PY to run frozen-reference parity check",
)
class TestBitwiseParityAgainstFrozen(unittest.TestCase):
    def test_exact_bitwise_parity_softmax_and_sigmoid(self) -> None:
        frozen = _load_frozen_v2_module()
        dev = torch.device("cpu")

        for mode, in_ch, out_ch in (("multiclass_softmax", 1, 4), ("multilabel_sigmoid", 3, 2)):
            model = _make_trained_tiny_model(in_ch=in_ch, out_ch=out_ch, seed=101 + in_ch)
            torch.manual_seed(202 + in_ch)
            src_imgs = [torch.randn(in_ch, 64, 64) for _ in range(12)]
            tgt_imgs = [
                torch.randn(in_ch, 64, 64) * 1.4 + 0.5,
                torch.zeros(in_ch, 64, 64),
                torch.randn(in_ch, 64, 64) * 0.3 - 0.8,
                torch.randn(in_ch, 64, 64) * 2.5 + 3.0,
                torch.randn(in_ch, 64, 64) * 2.6 + 3.1,
                torch.randn(in_ch, 64, 64) * 1.1 + 0.2,
            ]

            new_base = PrevalenceDeconfoundedDualTTA.fit_from_source(
                model=model, train_imgs=src_imgs, device=dev, output_mode=mode, config=PDDualConfig()
            )
            old_base = frozen.PrevalenceDeconfoundedDualTTA.fit_from_source(
                model=model, train_imgs=src_imgs, device=dev, output_mode=mode, config=frozen.PDDualConfig()
            )

            for preset in ("pd_dual_tta", "shallow_only", "no_rescue", "no_prev_match", "no_supp_only", "no_shift_only", "deep_inst_only"):
                cfg_new = resolve_preset(preset)
                cfg_old = frozen.PDDualConfig(**cfg_new.__dict__)
                ad_new = new_base.with_config(cfg_new)
                ad_old = frozen.PrevalenceDeconfoundedDualTTA(
                    model=model,
                    shallow_tta=old_base.shallow_tta,
                    deep_enc_bns=old_base.deep_enc_bns,
                    deep_dec_bns=old_base.deep_dec_bns,
                    source_bank=old_base.source_bank,
                    output_mode=mode,
                    device=dev,
                    config=cfg_old,
                )
                ad_new.reset_stream()
                ad_old.reset_stream()
                for im in tgt_imgs:
                    l_new, d_new = ad_new.predict_single(im, device=dev)
                    l_old, d_old = ad_old.predict_single(im, device=dev)
                    max_diff = float((l_new - l_old).abs().max().item())
                    self.assertEqual(max_diff, 0.0, msg=f"Mismatch on {mode}/{preset}: max_diff={max_diff}")
                    for k in ("rescued", "w_supp", "w_shift_mean", "lam_deep_mean", "deep_ran"):
                        self.assertAlmostEqual(d_new[k], d_old[k], places=7)


if __name__ == "__main__":
    unittest.main()
