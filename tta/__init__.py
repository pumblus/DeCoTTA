"""Public interface for Prevalence-Deconfounded Dual-Regime TTA (`pd_dual_tta`)."""
from .baselines import CoTTAAdapter, PTBNAdapter, SARAdapter, SourceOnlyAdapter, TENTAdapter
from .config import CANONICAL_PRESETS, PDDualConfig, PRESET_ALIASES, WEIGHT_ONLY_SF_CONFIG, resolve_preset
from .deep_bank import DeepResidualStreamState, DeepSourceBank
from .masks import (
    OutputMode,
    SpatialErosionReplay,
    detect_body_mask,
    discover_bn_partition,
    discover_encoder_bns_up_to_stride,
    extract_logits,
    extract_prevalence_vector,
    model_state_sha256,
)
from .pd_dual_tta import PrevalenceDeconfoundedDualTTA
from .shallow_frontend import LayerStreamState, ShallowBackgroundStreamTTA, StreamLayerTable

__all__ = [
    "CANONICAL_PRESETS",
    "PRESET_ALIASES",
    "WEIGHT_ONLY_SF_CONFIG",
    "PDDualConfig",
    "resolve_preset",
    "OutputMode",
    "PrevalenceDeconfoundedDualTTA",
    "ShallowBackgroundStreamTTA",
    "DeepSourceBank",
    "DeepResidualStreamState",
    "StreamLayerTable",
    "LayerStreamState",
    "SpatialErosionReplay",
    "detect_body_mask",
    "discover_bn_partition",
    "discover_encoder_bns_up_to_stride",
    "extract_logits",
    "extract_prevalence_vector",
    "model_state_sha256",
    "SourceOnlyAdapter",
    "PTBNAdapter",
    "TENTAdapter",
    "SARAdapter",
    "CoTTAAdapter",
]
