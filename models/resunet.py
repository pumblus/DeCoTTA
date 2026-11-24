"""ResNet-34 encoder + bilinear U-Net decoder (`ResUNet`)."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNetEncoder(nn.Module):
    """ResNet-34 feature extractor returning `[stem, layer1, layer2, layer3, layer4]`."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2)

    def _make_layer(self, inplanes: int, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(inplanes, planes, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x0, x1, x2, x3, x4]


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x)))


class ResUNet(nn.Module):
    """2D ResNet-34 U-Net used on M&Ms (`in_channels=1`) and RIGA+ (`in_channels=3`)."""

    def __init__(self, in_channels: int = 3, num_classes: int = 3) -> None:
        super().__init__()
        self.encoder = ResNetEncoder(in_channels=in_channels)
        self.dec4 = DecoderBlock(in_ch=512, skip_ch=256, out_ch=256)
        self.dec3 = DecoderBlock(in_ch=256, skip_ch=128, out_ch=128)
        self.dec2 = DecoderBlock(in_ch=128, skip_ch=64, out_ch=64)
        self.dec1 = DecoderBlock(in_ch=64, skip_ch=64, out_ch=32)
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.outc = nn.Conv2d(32, num_classes, kernel_size=1)
        self.feat_dim = 32

    def _encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        x = self.dec4(feats[4], feats[3])
        x = self.dec3(x, feats[2])
        x = self.dec2(x, feats[1])
        x = self.dec1(x, feats[0])
        return self.final_up(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.outc(self._encode_decode(x))
