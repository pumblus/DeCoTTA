"""Configuration and ablation presets for Prevalence-Deconfounded Dual-Regime TTA."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

DeepMode = Literal["hybrid", "inst"]
RescueMode = Literal["none", "enc_ema", "enc_inst", "enc_ema_fallback_inst"]


@dataclass(frozen=True)
class PDDualConfig:
    """Universal hyperparameter configuration (shared identically across all benchmarks)."""

    max_shallow_stride: int = 4
    strip_padding: bool = True
    exclude_fg: bool = True
    enable_shallow_stream: bool = True
    enforce_pad_bg: bool = False
    fg_fallback_cap: float = 0.35
    shallow_alpha_min: float = 0.05

    tau_pres: float = 0.002
    rescue_mode: RescueMode = "enc_inst"
    rescue_alpha: float = 0.45
    tau_rescue: float = 0.005

    enable_deep: bool = True
    use_prev_match: bool = True
    use_supp_gate: bool = True
    use_shift_gate: bool = True
    deep_mode: DeepMode = "hybrid"
    alpha_max: float = 0.75
    gamma_supp: float = 1.2
    supp_quantile: float = 0.90
    shift_kappa_st: float = 0.95
    shift_kappa_in: float = 1.00
    deep_ema_min: float = 0.15
    max_cal_samples: int = 600
    scale_clamp: float = 3.0


WEIGHT_ONLY_SF_CONFIG = PDDualConfig(
    enable_shallow_stream=True,
    tau_pres=0.002,
    rescue_mode="enc_inst",
    enable_deep=True,
    use_prev_match=False,
    use_supp_gate=False,
    use_shift_gate=False,
    deep_mode="hybrid",
    alpha_max=0.30,
)

CANONICAL_PRESETS: Dict[str, PDDualConfig] = {
    "pd_dual_tta": PDDualConfig(),
    "shallow_only": PDDualConfig(enable_deep=False),
    "no_rescue": PDDualConfig(rescue_mode="none"),
    "no_prev_match": PDDualConfig(use_prev_match=False),
    "no_supp_only": PDDualConfig(use_supp_gate=False, use_shift_gate=True),
    "no_shift_only": PDDualConfig(use_supp_gate=True, use_shift_gate=False),
    "no_supp_shift_gate": PDDualConfig(use_supp_gate=False, use_shift_gate=False),
    "deep_inst_only": PDDualConfig(deep_mode="inst"),
    "weight_only_sf": WEIGHT_ONLY_SF_CONFIG,
}

PRESET_ALIASES: Dict[str, str] = {
    "v2_full": "pd_dual_tta",
    "v1_shallow_only": "shallow_only",
    "abl_no_rescue": "no_rescue",
    "v2_no_prev_match": "no_prev_match",
    "abl_no_supp_only": "no_supp_only",
    "abl_no_shift_only": "no_shift_only",
    "v2_no_supp_shift_gate": "no_supp_shift_gate",
    "abl_deep_inst_only": "deep_inst_only",
    "abl_weight_only_sf": "weight_only_sf",
}


def resolve_preset(name: str) -> PDDualConfig:
    """Resolve a canonical or legacy preset name into an immutable `PDDualConfig`."""
    key = PRESET_ALIASES.get(name, name)
    if key not in CANONICAL_PRESETS:
        valid = sorted(list(CANONICAL_PRESETS.keys()) + list(PRESET_ALIASES.keys()))
        raise KeyError(f"Unknown preset {name!r}. Valid presets: {valid}")
    return CANONICAL_PRESETS[key]
