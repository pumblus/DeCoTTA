"""Source-training augmentations and deterministic evaluation transforms."""
from __future__ import annotations

import random

import torch


def _hflip(img: torch.Tensor, mask: torch.Tensor):
    return torch.flip(img, dims=[-1]), torch.flip(mask, dims=[-1])


def _vflip(img: torch.Tensor, mask: torch.Tensor):
    return torch.flip(img, dims=[-2]), torch.flip(mask, dims=[-2])


def _rotate90(img: torch.Tensor, mask: torch.Tensor, k: int):
    return torch.rot90(img, k=k, dims=[-2, -1]), torch.rot90(mask, k=k, dims=[-2, -1])


class MRISourceTrainTransform:
    """Random flips, 90-degree rotations, and mild linear intensity jitter for MRI."""

    def __init__(self, p_flip: float = 0.5, p_rot: float = 0.5, intensity_scale: float = 0.1) -> None:
        self.p_flip = p_flip
        self.p_rot = p_rot
        self.intensity_scale = intensity_scale

    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        if random.random() < self.p_flip:
            img, mask = _hflip(img, mask)
        if random.random() < self.p_flip:
            img, mask = _vflip(img, mask)
        if random.random() < self.p_rot:
            img, mask = _rotate90(img, mask, k=random.choice([1, 2, 3]))
        if self.intensity_scale > 0:
            scale = 1.0 + (random.random() * 2.0 - 1.0) * self.intensity_scale
            shift = (random.random() * 2.0 - 1.0) * self.intensity_scale
            img = img * scale + shift
        return img, mask


class FundusSourceTrainTransform:
    """Random horizontal flip and mild per-channel color scaling for RGB fundus."""

    def __init__(self, p_flip: float = 0.5, color_jitter: float = 0.1) -> None:
        self.p_flip = p_flip
        self.color_jitter = color_jitter

    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        if random.random() < self.p_flip:
            img, mask = _hflip(img, mask)
        if self.color_jitter > 0:
            channel_scale = 1.0 + (torch.rand(3, 1, 1) * 2.0 - 1.0) * self.color_jitter
            img = (img * channel_scale).clamp(0.0, 1.0)
        return img, mask


class IdentityTransform:
    """Deterministic identity transform for validation and online TTA streams."""

    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        return img, mask


def build_transform(dataset: str, mode: str):
    if mode in ("eval", "tta", "val"):
        return IdentityTransform()
    if mode != "train":
        raise ValueError(f"Unsupported transform mode: {mode!r}")
    if dataset in ("mms", "prostate"):
        return MRISourceTrainTransform()
    if dataset == "rigaplus":
        return FundusSourceTrainTransform()
    raise ValueError(f"Unknown dataset: {dataset!r}")
