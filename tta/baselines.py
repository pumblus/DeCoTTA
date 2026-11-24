"""Reference single-sample (`batch_size=1`) online TTA baselines under a unified adapter seam."""
from __future__ import annotations

import copy
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .masks import OutputMode, extract_logits


def _pixel_entropy(logits: torch.Tensor, output_mode: OutputMode) -> torch.Tensor:
    if output_mode == "multiclass_softmax":
        probs = torch.softmax(logits, dim=1).clamp(1e-7, 1.0)
        return -(probs * probs.log()).sum(dim=1).mean()
    probs = torch.sigmoid(logits).clamp(1e-7, 1.0 - 1e-7)
    ent = -(probs * probs.log() + (1.0 - probs) * (1.0 - probs).log())
    return ent.mean()


def _configure_bn_test_stats(model: nn.Module, train_affine: bool = False) -> list[nn.Parameter]:
    model.eval()
    params: list[nn.Parameter] = []
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            if train_affine and m.affine:
                m.weight.requires_grad_(True)
                m.bias.requires_grad_(True)
                params.extend([m.weight, m.bias])
    return params


class SourceOnlyAdapter:
    """Frozen source model evaluation (`model.eval()`)."""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.model.eval()

    def reset_stream(self) -> None:
        self.model.eval()

    @torch.no_grad()
    def predict_single(self, img_chw: torch.Tensor, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        dev = self.device if device is None else device
        self.model.eval()
        return extract_logits(self.model(img_chw.unsqueeze(0).to(dev))), {}


class PTBNAdapter:
    """Prediction-Time BatchNorm (`batch_size=1` per-sample BN statistics, no backprop)."""

    def __init__(self, source_model: nn.Module, device: torch.device) -> None:
        self.model = copy.deepcopy(source_model).to(device)
        self.device = device
        _configure_bn_test_stats(self.model, train_affine=False)

    def reset_stream(self) -> None:
        pass

    @torch.no_grad()
    def predict_single(self, img_chw: torch.Tensor, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        dev = self.device if device is None else device
        return extract_logits(self.model(img_chw.unsqueeze(0).to(dev))), {}


class TENTAdapter:
    """Test-Time Entropy Minimization (Wang et al., ICLR 2021) at `batch_size=1`."""

    def __init__(
        self,
        source_model: nn.Module,
        device: torch.device,
        output_mode: OutputMode = "multiclass_softmax",
        lr: float = 1e-4,
    ) -> None:
        self._source_state = copy.deepcopy(source_model.state_dict())
        self.base_arch = copy.deepcopy(source_model).to(device)
        self.device = device
        self.output_mode: OutputMode = output_mode
        self.lr = float(lr)
        self.reset_stream()

    def reset_stream(self) -> None:
        self.model = copy.deepcopy(self.base_arch).to(self.device)
        self.model.load_state_dict(self._source_state, strict=True)
        params = _configure_bn_test_stats(self.model, train_affine=True)
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

    def predict_single(self, img_chw: torch.Tensor, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        dev = self.device if device is None else device
        x = img_chw.unsqueeze(0).to(dev)
        self.optimizer.zero_grad(set_to_none=True)
        logits = extract_logits(self.model(x))
        loss = _pixel_entropy(logits, self.output_mode)
        loss.backward()
        self.optimizer.step()
        return logits.detach(), {"entropy": float(loss.item())}


class SARAdapter:
    """Sharpness-Aware and Reliable Entropy Minimization (Niu et al., ICLR 2023) at `batch_size=1`."""

    def __init__(
        self,
        source_model: nn.Module,
        device: torch.device,
        output_mode: OutputMode = "multiclass_softmax",
        num_classes: int = 2,
        lr: float = 2.5e-4 / 16.0,
        rho: float = 0.05,
        reset_constant_em: float = 0.2,
    ) -> None:
        self._source_state = copy.deepcopy(source_model.state_dict())
        self.base_arch = copy.deepcopy(source_model).to(device)
        self.device = device
        self.output_mode: OutputMode = output_mode
        self.margin_e0 = 0.4 * math.log(max(2, int(num_classes)))
        self.lr = float(lr)
        self.rho = float(rho)
        self.reset_constant_em = float(reset_constant_em)
        self.reset_stream()

    def reset_stream(self) -> None:
        self.model = copy.deepcopy(self.base_arch).to(self.device)
        self.model.load_state_dict(self._source_state, strict=True)
        self.params = _configure_bn_test_stats(self.model, train_affine=True)
        self.optimizer = torch.optim.SGD(self.params, lr=self.lr, momentum=0.9)
        self.ema: Optional[float] = None

    def predict_single(self, img_chw: torch.Tensor, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        dev = self.device if device is None else device
        x = img_chw.unsqueeze(0).to(dev)
        self.optimizer.zero_grad(set_to_none=True)
        logits = extract_logits(self.model(x))
        ent = _pixel_entropy(logits, self.output_mode)
        if float(ent.item()) >= self.margin_e0:
            return logits.detach(), {"adapted": 0.0}

        ent.backward()
        grad_norm = torch.norm(torch.stack([p.grad.norm(p=2) for p in self.params if p.grad is not None]), p=2)
        scale = self.rho / (grad_norm + 1e-12)
        e_ws = []
        with torch.no_grad():
            for p in self.params:
                ew = (p.grad * scale) if p.grad is not None else torch.zeros_like(p)
                p.add_(ew)
                e_ws.append(ew)

        self.optimizer.zero_grad(set_to_none=True)
        logits2 = extract_logits(self.model(x))
        ent2 = _pixel_entropy(logits2, self.output_mode)
        val2 = float(ent2.item())
        if val2 < self.margin_e0:
            ent2.backward()
        with torch.no_grad():
            for p, ew in zip(self.params, e_ws):
                p.sub_(ew)
        if val2 < self.margin_e0:
            self.optimizer.step()
            self.ema = val2 if self.ema is None else (0.9 * self.ema + 0.1 * val2)
            if self.ema < self.reset_constant_em:
                self.reset_stream()
        return logits.detach(), {"adapted": 1.0}


class CoTTAAdapter:
    """Continual Test-Time Adaptation (Wang et al., CVPR 2022) at `batch_size=1`."""

    def __init__(
        self,
        source_model: nn.Module,
        device: torch.device,
        output_mode: OutputMode = "multiclass_softmax",
        lr: float = 1e-4,
        mt_alpha: float = 0.999,
        rst_m: float = 0.01,
    ) -> None:
        self._source_state = copy.deepcopy(source_model.state_dict())
        self.base_arch = copy.deepcopy(source_model).to(device)
        self.device = device
        self.output_mode: OutputMode = output_mode
        self.lr = float(lr)
        self.mt_alpha = float(mt_alpha)
        self.rst_m = float(rst_m)
        self.reset_stream()

    def reset_stream(self) -> None:
        self.model = copy.deepcopy(self.base_arch).to(self.device)
        self.model.load_state_dict(self._source_state, strict=True)
        _configure_bn_test_stats(self.model, train_affine=True)
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.teacher = copy.deepcopy(self.model).to(self.device)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def predict_single(self, img_chw: torch.Tensor, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        dev = self.device if device is None else device
        x = img_chw.unsqueeze(0).to(dev)
        with torch.no_grad():
            t0 = extract_logits(self.teacher(x))
            tf = torch.flip(extract_logits(self.teacher(torch.flip(x, dims=[3]))), dims=[3])
            t_logits = 0.5 * (t0 + tf)

        self.optimizer.zero_grad(set_to_none=True)
        s_logits = extract_logits(self.model(x))
        if self.output_mode == "multiclass_softmax":
            pt = torch.softmax(t_logits, dim=1).clamp(1e-7, 1.0)
            ps = torch.log_softmax(s_logits, dim=1)
            loss = -(pt * ps).sum(dim=1).mean()
        else:
            pt = torch.sigmoid(t_logits).clamp(1e-7, 1.0 - 1e-7)
            ps = torch.sigmoid(s_logits).clamp(1e-7, 1.0 - 1e-7)
            loss = -(pt * ps.log() + (1.0 - pt) * (1.0 - ps).log()).mean()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for pt_param, ps_param in zip(self.teacher.parameters(), self.model.parameters()):
                pt_param.mul_(self.mt_alpha).add_(ps_param, alpha=1.0 - self.mt_alpha)
            for nm, param in self.model.named_parameters():
                mask = (torch.rand_like(param) < self.rst_m).float()
                src_p = self._source_state[nm].to(dev)
                param.data.copy_(mask * src_p + (1.0 - mask) * param.data)

        return t_logits.detach(), {"loss": float(loss.item())}
