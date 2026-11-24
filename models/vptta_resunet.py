"""Standalone VPTTA/OPTIC/Prostate `ResUnet-34` compatible (`strict=True`) with official checkpoints."""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class _VPTTAResNet34(nn.Module):
    """Exact attribute layout of official `VPTTA/OPTIC/networks/resnet.py`."""

    def __init__(self, ini: int = 3, num_classes: int = 1000) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(ini, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.in1 = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [_BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        sfs: List[torch.Tensor] = []
        x = self.relu(self.bn1(self.conv1(x)))
        sfs.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        sfs.append(x)
        x = self.layer2(x)
        sfs.append(x)
        x = self.layer3(x)
        sfs.append(x)
        x = self.layer4(x)
        sfs.append(x)
        return x, sfs


class _UnetBlock(nn.Module):
    def __init__(self, up_in: int, x_in: int, n_out: int) -> None:
        super().__init__()
        half = n_out // 2
        self.x_conv = nn.Conv2d(x_in, half, kernel_size=1)
        self.tr_conv = nn.ConvTranspose2d(up_in, half, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p: torch.Tensor, x_p: torch.Tensor) -> torch.Tensor:
        cat_p = torch.cat([self.tr_conv(up_p), self.x_conv(x_p)], dim=1)
        return self.bn(F.relu(cat_p))


class VPTTAResUNet(nn.Module):
    """Transposed-conv ResUnet-34 used on Prostate (`in_channels=1`) and OPTIC (`in_channels=3`)."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        return_features: bool = False,
    ) -> None:
        super().__init__()
        self.res = _VPTTAResNet34(ini=in_channels)
        self.num_classes = int(num_classes)
        self.return_features = bool(return_features)

        self.up1 = _UnetBlock(512, 256, 256)
        self.up2 = _UnetBlock(256, 128, 256)
        self.up3 = _UnetBlock(256, 64, 256)
        self.up4 = _UnetBlock(256, 64, 256)
        self.up5 = nn.ConvTranspose2d(256, 32, kernel_size=2, stride=2)
        self.bnout = nn.BatchNorm2d(32)
        self.seg_head = nn.Conv2d(32, self.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        x, sfs = self.res(x)
        x = F.relu(x)
        x = self.up1(x, sfs[3])
        x = self.up2(x, sfs[2])
        x = self.up3(x, sfs[1])
        x = self.up4(x, sfs[0])
        x = self.up5(x)
        logits = self.seg_head(F.relu(self.bnout(x)))
        if self.return_features:
            return logits, sfs
        return logits
