"""Anatomical support masking, BatchNorm partition discovery, and prevalence extraction."""
from __future__ import annotations

import hashlib
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS: float = 1e-8
OutputMode = Literal["multiclass_softmax", "multilabel_sigmoid"]


def extract_logits(output: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
    """Unwrap tuple/list model outputs (e.g. `(logits, skip_features)`) into a tensor."""
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def model_state_sha256(model: nn.Module) -> str:
    """Deterministic SHA-256 digest over all parameter and buffer tensors of `model`."""
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode("utf-8"))
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def _pair(val: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(val, int):
        return (val, val)
    return (int(val[0]), int(val[1]))


def _erode_op(
    m: torch.Tensor,
    k: Tuple[int, int],
    s: Tuple[int, int],
    p: Tuple[int, int],
) -> torch.Tensor:
    return 1.0 - F.max_pool2d(1.0 - m, kernel_size=k, stride=s, padding=p)


def _align_if_functional_pool(m: torch.Tensor, h_in: int, w_in: int) -> torch.Tensor:
    hm, wm = int(m.shape[2]), int(m.shape[3])
    if hm == h_in and wm == w_in:
        return m
    if hm > h_in and hm % h_in == 0 and wm % w_in == 0:
        sh, sw = hm // h_in, wm // w_in
        return _erode_op(m, (sh, sw), (sh, sw), (0, 0))
    return F.interpolate(m, size=(h_in, w_in), mode="nearest")


def detect_body_mask(x_chw: torch.Tensor, strip_padding: bool = True) -> torch.Tensor:
    """Separate anatomical support `Omega_body` from border-connected padding `Omega_pad`."""
    _, h, w = x_chw.shape
    if not strip_padding:
        return torch.ones((h, w), dtype=torch.bool)

    x_cpu = x_chw.detach().cpu().to(torch.float32)
    c = int(x_cpu.shape[0])
    xmin = x_cpu.amin(dim=(1, 2), keepdim=True)
    xmax = x_cpu.amax(dim=(1, 2), keepdim=True)
    dyn = xmax - xmin
    if float(dyn.max()) < 1e-7:
        return torch.ones((h, w), dtype=torch.bool)

    u = (x_cpu - xmin) / (dyn + EPS)
    u4_rep = F.pad(u.unsqueeze(0), (1, 1, 1, 1), mode="replicate")
    local_max = F.max_pool2d(u4_rep, kernel_size=3, stride=1, padding=0)[0]
    local_min = -F.max_pool2d(-u4_rep, kernel_size=3, stride=1, padding=0)[0]
    r3 = (local_max - local_min).amax(dim=0).numpy()

    flat_core = r3 <= 1e-4
    lbl_flat, n_flat = ndimage.label(flat_core)
    if n_flat > 0:
        b_lbls = (
            set(np.unique(lbl_flat[0, :]))
            | set(np.unique(lbl_flat[-1, :]))
            | set(np.unique(lbl_flat[:, 0]))
            | set(np.unique(lbl_flat[:, -1]))
        )
        b_lbls.discard(0)
        if b_lbls:
            flat_b = torch.from_numpy(np.isin(lbl_flat, list(b_lbls)).astype(np.float32))[None, None]
            pad_mask = F.max_pool2d(flat_b, kernel_size=3, stride=1, padding=1)[0, 0].numpy() > 0.5
        else:
            pad_mask = np.zeros((h, w), dtype=bool)
    else:
        pad_mask = np.zeros((h, w), dtype=bool)

    is_true_rgb = (c == 3) and float((x_cpu[0] - x_cpu[1]).abs().max()) > 1e-5
    if is_true_rgb:
        dark_rgb = u.amax(dim=0).numpy() <= 0.05
        lbl_d, n_d = ndimage.label(np.logical_or(pad_mask, dark_rgb))
        if n_d > 0:
            d_lbls = (
                set(np.unique(lbl_d[0, :]))
                | set(np.unique(lbl_d[-1, :]))
                | set(np.unique(lbl_d[:, 0]))
                | set(np.unique(lbl_d[:, -1]))
            )
            d_lbls.discard(0)
            if d_lbls:
                pad_mask = np.isin(lbl_d, list(d_lbls))

    body = ~pad_mask
    if is_true_rgb:
        body = ndimage.binary_fill_holes(body)
        body_t = torch.from_numpy(body.astype(np.float32))[None, None]
        body_er = -F.max_pool2d(-F.pad(body_t, (2, 2, 2, 2), value=0.0), kernel_size=5, stride=1, padding=0)
        body = body_er[0, 0].numpy() > 0.5

    if int(body.sum()) < int(0.15 * h * w):
        return torch.ones((h, w), dtype=torch.bool)
    return torch.from_numpy(body.astype(bool))


def discover_encoder_bns_up_to_stride(
    model: nn.Module,
    dummy_input: torch.Tensor,
    max_stride: int = 4,
) -> List[Tuple[str, nn.BatchNorm2d, int]]:
    """Discover executed encoder `BatchNorm2d` layers up to `max_stride` before upsampling."""
    h0 = int(dummy_input.shape[2])
    records: List[Tuple[str, nn.BatchNorm2d, int]] = []
    seen_ids = set()
    hooks = []

    for name, mod in model.named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            def _mk(nm: str, m: nn.BatchNorm2d):
                def _pre(_mod: nn.Module, inp: Tuple[torch.Tensor, ...]) -> None:
                    if id(m) not in seen_ids:
                        seen_ids.add(id(m))
                        records.append((nm, m, int(inp[0].shape[2])))
                return _pre
            hooks.append(mod.register_forward_pre_hook(_mk(name, mod)))

    try:
        model.eval()
        with torch.no_grad():
            model(dummy_input)
    finally:
        for hk in hooks:
            hk.remove()

    enc_bns: List[Tuple[str, nn.BatchNorm2d, int]] = []
    min_h_seen = h0
    for nm, mod, hl in records:
        if hl > min_h_seen:
            break
        min_h_seen = min(min_h_seen, hl)
        if "downsample" in nm:
            continue
        stride = max(1, round(h0 / float(hl)))
        if stride <= max_stride:
            enc_bns.append((nm, mod, stride))
    return enc_bns


def discover_bn_partition(
    model: nn.Module,
    dummy_input: torch.Tensor,
    max_stride: int = 4,
) -> Tuple[
    List[Tuple[str, nn.BatchNorm2d, int]],
    List[Tuple[str, nn.BatchNorm2d, int]],
    List[Tuple[str, nn.BatchNorm2d, int]],
]:
    """Partition executed `BatchNorm2d` layers into `(shallow_bns, deep_enc_bns, deep_dec_bns)`."""
    h0 = int(dummy_input.shape[2])
    records: List[Tuple[str, nn.BatchNorm2d, int]] = []
    seen_ids = set()
    hooks = []

    for name, mod in model.named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            def _mk(nm: str, m: nn.BatchNorm2d):
                def _pre(_mod: nn.Module, inp: Tuple[torch.Tensor, ...]) -> None:
                    if id(m) not in seen_ids:
                        seen_ids.add(id(m))
                        records.append((nm, m, int(inp[0].shape[2])))
                return _pre
            hooks.append(mod.register_forward_pre_hook(_mk(name, mod)))

    try:
        model.eval()
        with torch.no_grad():
            model(dummy_input)
    finally:
        for hk in hooks:
            hk.remove()

    shallow_bns = discover_encoder_bns_up_to_stride(model, dummy_input, max_stride=max_stride)
    shallow_ids = {id(m) for _, m, _ in shallow_bns}

    deep_enc: List[Tuple[str, nn.BatchNorm2d, int]] = []
    deep_dec: List[Tuple[str, nn.BatchNorm2d, int]] = []
    min_h_seen = h0
    in_decoder = False
    for nm, mod, hl in records:
        if hl > min_h_seen:
            in_decoder = True
        if not in_decoder:
            min_h_seen = min(min_h_seen, hl)
        stride = max(1, round(h0 / float(hl)))
        if id(mod) in shallow_ids:
            continue
        if in_decoder:
            deep_dec.append((nm, mod, stride))
        else:
            deep_enc.append((nm, mod, stride))

    return shallow_bns, deep_enc, deep_dec


class SpatialErosionReplay:
    """Traces shallow spatial conv/pool ops once to erode `Omega_body` without padding bleed."""

    def __init__(self, events: List[Tuple[object, ...]], n_layers: int) -> None:
        self.events = events
        self.n_layers = n_layers

    @classmethod
    def trace_once(
        cls,
        model: nn.Module,
        enc_bns: Sequence[Tuple[str, nn.BatchNorm2d, int]],
        dummy_input: torch.Tensor,
    ) -> "SpatialErosionReplay":
        target_ids = {id(bn): idx for idx, (_, bn, _) in enumerate(enc_bns)}
        seen_bn = set()
        events: List[Tuple[object, ...]] = []
        hooks = []

        for name, mod in model.named_modules():
            if "downsample" in name:
                continue
            if isinstance(mod, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
                def _mk_sp(m_op: nn.Module):
                    def _pre(_m: nn.Module, inp: Tuple[torch.Tensor, ...]) -> None:
                        if len(seen_bn) >= len(target_ids):
                            return
                        h_in, w_in = int(inp[0].shape[2]), int(inp[0].shape[3])
                        events.append(("sp", h_in, w_in, _pair(m_op.kernel_size), _pair(m_op.stride), _pair(m_op.padding)))
                    return _pre
                hooks.append(mod.register_forward_pre_hook(_mk_sp(mod)))
            elif id(mod) in target_ids:
                slot = target_ids[id(mod)]
                def _mk_bn(sl: int):
                    def _pre(_m: nn.Module, inp: Tuple[torch.Tensor, ...]) -> None:
                        if sl not in seen_bn:
                            seen_bn.add(sl)
                            events.append(("bn", sl, int(inp[0].shape[2]), int(inp[0].shape[3])))
                    return _pre
                hooks.append(mod.register_forward_pre_hook(_mk_bn(slot)))

        try:
            model.eval()
            with torch.no_grad():
                model(dummy_input)
        finally:
            for hk in hooks:
                hk.remove()
        return cls(events, len(enc_bns))

    def replay(self, body_mask_0: torch.Tensor, device: torch.device) -> List[torch.Tensor]:
        cur_m = body_mask_0.to(device=device, dtype=torch.float32)[None, None]
        out: List[Optional[torch.Tensor]] = [None] * self.n_layers
        for ev in self.events:
            if ev[0] == "sp":
                _, h_in, w_in, k, s, p = ev
                cur_m = _align_if_functional_pool(cur_m, int(h_in), int(w_in))
                if k != (1, 1) or s != (1, 1) or p != (0, 0):
                    cur_m = _erode_op(cur_m, k, s, p)
            else:
                _, sl, h_in, w_in = ev
                cur_m = _align_if_functional_pool(cur_m, int(h_in), int(w_in))
                out[int(sl)] = cur_m[0, 0] > 0.5
        return [m if m is not None else torch.ones((1, 1), dtype=torch.bool, device=device) for m in out]


def build_layer_bg_stat_mask(
    clean_mask_hw: torch.Tensor,
    body_mask_0: torch.Tensor,
    fg_mask_0: Optional[torch.Tensor],
    exclude_fg: bool = True,
    fg_fallback_cap: float = 0.35,
    dilate_cells: int = 1,
) -> Tuple[torch.Tensor, bool]:
    """Build eroded background statistics mask `[H_l, W_l]` with >35% hallucination fallback."""
    hl, wl = int(clean_mask_hw.shape[0]), int(clean_mask_hw.shape[1])
    n_clean = int(clean_mask_hw.sum())
    if n_clean < 16:
        clean_mask_hw = F.interpolate(body_mask_0.float()[None, None], size=(hl, wl), mode="nearest")[0, 0] > 0.5
        n_clean = int(clean_mask_hw.sum())

    if (not exclude_fg) or (fg_mask_0 is None):
        return clean_mask_hw, False

    n_body0 = max(1, int(body_mask_0.sum()))
    rho_fg = float((fg_mask_0 & body_mask_0).sum()) / float(n_body0)
    if rho_fg > fg_fallback_cap:
        return clean_mask_hw, False

    if int(fg_mask_0.sum()) == 0:
        return clean_mask_hw, True

    fg_l = F.adaptive_max_pool2d(fg_mask_0.float()[None, None], output_size=(hl, wl))
    if dilate_cells > 0:
        fg_l = F.max_pool2d(fg_l, kernel_size=2 * dilate_cells + 1, stride=1, padding=dilate_cells)
    cand = clean_mask_hw & (~(fg_l[0, 0] > 0.5))
    if int(cand.sum()) < max(64, int(0.45 * n_clean)):
        cand_nodil = clean_mask_hw & (~(F.adaptive_max_pool2d(fg_mask_0.float()[None, None], (hl, wl))[0, 0] > 0.5))
        if int(cand_nodil.sum()) >= max(64, int(0.45 * n_clean)):
            return cand_nodil, True
        return clean_mask_hw, False
    return cand, True


def pass0_foreground_mask(logits: torch.Tensor, output_mode: OutputMode) -> torch.Tensor:
    """Coarse binary foreground mask `[H, W]` from unadapted Pass-0 logits."""
    if output_mode == "multiclass_softmax":
        return logits.argmax(dim=1)[0] > 0
    if output_mode == "multilabel_sigmoid":
        return torch.sigmoid(logits[0]).amax(dim=0) > 0.5
    raise ValueError(f"Unsupported output_mode: {output_mode!r}")


def extract_prevalence_vector(
    logits: torch.Tensor,
    body_mask_dev: torch.Tensor,
    output_mode: OutputMode,
) -> Tuple[np.ndarray, float]:
    """Compute per-class foreground area fractions `pi_t in [0, 1]^K` and total `r_fg` inside `Omega_body`."""
    n_body = max(1.0, float(body_mask_dev.sum().item()))
    if output_mode == "multiclass_softmax":
        c_out = int(logits.shape[1])
        k_fg = max(1, c_out - 1)
        pred = logits.argmax(dim=1)[0]
        pi = np.zeros(k_fg, dtype=np.float64)
        for k in range(1, c_out):
            pi[k - 1] = float(((pred == k) & body_mask_dev).sum().item()) / n_body
        return pi, float(pi.sum())
    if output_mode == "multilabel_sigmoid":
        pbin = torch.sigmoid(logits[0]) >= 0.5
        k_fg = int(pbin.shape[0])
        pi = np.zeros(k_fg, dtype=np.float64)
        for k in range(k_fg):
            pi[k] = float((pbin[k] & body_mask_dev).sum().item()) / n_body
        r_fg = float((pbin.any(dim=0) & body_mask_dev).sum().item()) / n_body
        return pi, r_fg
    raise ValueError(f"Unsupported output_mode: {output_mode!r}")
