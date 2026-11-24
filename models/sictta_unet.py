"""Standalone SicTTA 2D U-Net compatible (`strict=True`) with official `mms2d_paper` checkpoints."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _UNetConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float) -> None:
        super().__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_conv(x)


class _UNetUpBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, up_mode: str, dropout_p: float) -> None:
        super().__init__()
        if up_mode == "upconv":
            self.up: nn.Module = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_chans, out_chans, kernel_size=1),
            )
        else:
            raise ValueError(f"Unsupported up_mode: {up_mode!r}")
        self.conv_block = _UNetConvBlock(in_chans, out_chans, dropout_p)

    @staticmethod
    def _centre_crop(layer: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        _, _, h, w = layer.size()
        dy = (h - target_size[0]) // 2
        dx = (w - target_size[1]) // 2
        return layer[:, :, dy : dy + target_size[0], dx : dx + target_size[1]]

    def forward(self, x: torch.Tensor, bridge: torch.Tensor) -> torch.Tensor:
        up = self.up(x)
        crop = self._centre_crop(bridge, (int(up.shape[2]), int(up.shape[3])))
        return self.conv_block(torch.cat([up, crop], dim=1))


class _SicTTAEncoder(nn.Module):
    def __init__(self, in_chns: int, ft_chns: Sequence[int], dropout_p: Sequence[float]) -> None:
        super().__init__()
        self.down_path = nn.ModuleList([
            _UNetConvBlock(in_chns, ft_chns[0], dropout_p[0]),
            _UNetConvBlock(ft_chns[0], ft_chns[1], dropout_p[0]),
            _UNetConvBlock(ft_chns[1], ft_chns[2], dropout_p[0]),
            _UNetConvBlock(ft_chns[2], ft_chns[3], dropout_p[0]),
            _UNetConvBlock(ft_chns[3], ft_chns[4], dropout_p[0]),
        ])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        blocks: List[torch.Tensor] = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)
        return blocks, x


class _SicTTADecoder(nn.Module):
    def __init__(
        self,
        n_classes: int,
        ft_chns: Sequence[int],
        dropout_p: Sequence[float],
        up_mode: str,
    ) -> None:
        super().__init__()
        self.up_path = nn.ModuleList([
            _UNetUpBlock(ft_chns[4], ft_chns[3], up_mode, dropout_p[1]),
            _UNetUpBlock(ft_chns[3], ft_chns[2], up_mode, dropout_p[0]),
            _UNetUpBlock(ft_chns[2], ft_chns[1], up_mode, dropout_p[0]),
            _UNetUpBlock(ft_chns[1], ft_chns[0], up_mode, dropout_p[0]),
        ])
        self.last = nn.Conv2d(ft_chns[0], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, blocks: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        return self.last(x), x


class SicTTAUNet(nn.Module):
    """Lightweight `[16, 32, 64, 128, 256]` 2D U-Net used on cross-backbone `mms2d_paper`."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        ft_chns: Sequence[int] = (16, 32, 64, 128, 256),
        dropout_p: Sequence[float] = (0.0, 0.0, 0.3, 0.4, 0.5),
        up_mode: str = "upsample",
    ) -> None:
        super().__init__()
        self.enc = _SicTTAEncoder(in_channels, ft_chns, dropout_p)
        self.dec1 = _SicTTADecoder(num_classes, ft_chns, dropout_p, up_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks, latent = self.enc(x)
        out, _ = self.dec1(latent, blocks)
        return out
