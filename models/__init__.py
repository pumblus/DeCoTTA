"""Backbone factory and checkpoint loader across all supported segmentation architectures."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import torch
import torch.nn as nn

from .resunet import ResUNet
from .sictta_unet import SicTTAUNet
from .unet import UNet
from .vptta_resunet import VPTTAResUNet

DATASET_IO_SPECS: dict[str, Tuple[int, int]] = {
    "mms": (1, 4),
    "mms2d_paper": (1, 4),
    "rigaplus": (3, 3),
    "prostate": (1, 2),
    "optic": (3, 2),
}


def get_io_spec(dataset: str) -> Tuple[int, int]:
    if dataset not in DATASET_IO_SPECS:
        raise ValueError(f"Unknown dataset {dataset!r}. Supported: {sorted(DATASET_IO_SPECS)}")
    return DATASET_IO_SPECS[dataset]


def build_backbone(
    arch: str,
    in_channels: int,
    num_classes: int,
) -> nn.Module:
    if arch == "resunet":
        return ResUNet(in_channels=in_channels, num_classes=num_classes)
    if arch == "vptta_resunet":
        return VPTTAResUNet(in_channels=in_channels, num_classes=num_classes)
    if arch == "sictta_unet":
        return SicTTAUNet(in_channels=in_channels, num_classes=num_classes)
    if arch == "unet":
        return UNet(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"Unknown architecture {arch!r}.")


def load_model_checkpoint(
    arch: str,
    in_channels: int,
    num_classes: int,
    ckpt_path: Union[str, Path],
    device: torch.device,
) -> nn.Module:
    """Instantiate `arch`, load `ckpt_path` (`strict=True`), and set `eval()` mode."""
    p = Path(ckpt_path)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    model = build_backbone(arch, in_channels=in_channels, num_classes=num_classes).to(device)
    sd = torch.load(str(p), map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "model", "state_dict"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
            break
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


__all__ = [
    "DATASET_IO_SPECS",
    "ResUNet",
    "VPTTAResUNet",
    "SicTTAUNet",
    "UNet",
    "get_io_spec",
    "build_backbone",
    "load_model_checkpoint",
]
