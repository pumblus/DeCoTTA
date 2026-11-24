"""Stage 1: Background-anchored shallow feature calibration (`stride <= 4`)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .masks import (
    EPS,
    OutputMode,
    SpatialErosionReplay,
    build_layer_bg_stat_mask,
    detect_body_mask,
    discover_encoder_bns_up_to_stride,
    extract_logits,
    pass0_foreground_mask,
)


@dataclass
class StreamLayerTable:
    """Source-domain reference moments and invariance quantiles for one shallow BN layer."""

    mu_s: np.ndarray
    sigma_s: np.ndarray
    log_sig_s: np.ndarray
    var_mu: np.ndarray
    var_log_sig: np.ndarray
    active: np.ndarray
    q_med_s: float
    q95_s: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "mu_s": self.mu_s.copy(),
            "sigma_s": self.sigma_s.copy(),
            "log_sig_s": self.log_sig_s.copy(),
            "var_mu": self.var_mu.copy(),
            "var_log_sig": self.var_log_sig.copy(),
            "active": self.active.copy(),
            "q_med_s": float(self.q_med_s),
            "q95_s": float(self.q95_s),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "StreamLayerTable":
        return cls(
            mu_s=np.asarray(d["mu_s"], dtype=np.float64),
            sigma_s=np.asarray(d["sigma_s"], dtype=np.float64),
            log_sig_s=np.asarray(d["log_sig_s"], dtype=np.float64),
            var_mu=np.asarray(d["var_mu"], dtype=np.float64),
            var_log_sig=np.asarray(d["var_log_sig"], dtype=np.float64),
            active=np.asarray(d["active"], dtype=bool),
            q_med_s=float(d["q_med_s"]),
            q95_s=float(d["q95_s"]),
        )


class LayerStreamState:
    """Running shallow shift ledger with invariance-radius shrinkage and 2-step jump auto-reset."""

    def __init__(self, num_channels: int, alpha_min: float = 0.05) -> None:
        self.c = num_channels
        self.alpha_min = float(alpha_min)
        self.reset()

    def reset(self) -> None:
        self.step: int = 0
        self.nu: float = 1.0
        self.bar_mu: np.ndarray = np.zeros(self.c, dtype=np.float64)
        self.bar_eta: np.ndarray = np.zeros(self.c, dtype=np.float64)
        self.jump_streak: int = 0
        self.pending_mu: Optional[np.ndarray] = None
        self.pending_eta: Optional[np.ndarray] = None

    def observe_and_estimate(
        self,
        d_mu: np.ndarray,
        d_eta: np.ndarray,
        valid: np.ndarray,
        tbl: StreamLayerTable,
        enable_stream: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, bool]:
        v_mask = valid & tbl.active
        if not np.any(v_mask):
            return np.zeros_like(d_mu), np.zeros_like(d_eta), 0.0, 0.0, False

        d_mu_clean = np.where(v_mask, d_mu, 0.0)
        d_eta_clean = np.where(v_mask, d_eta, 0.0)

        if not enable_stream:
            d2 = float(
                0.5 * np.mean(
                    (d_mu_clean[v_mask] ** 2) / tbl.var_mu[v_mask]
                    + (d_eta_clean[v_mask] ** 2) / tbl.var_log_sig[v_mask]
                )
            )
            lam_ind = max(0.0, 1.0 - tbl.q95_s / max(d2, EPS))
            return lam_ind * d_mu_clean, lam_ind * d_eta_clean, 0.0, lam_ind, False

        did_reset = False
        is_isolated_jump = False
        if self.step >= 3:
            denom_scale = 1.0 + self.nu
            d2_jump = float(
                0.5 * np.mean(
                    ((d_mu_clean[v_mask] - self.bar_mu[v_mask]) ** 2) / (denom_scale * tbl.var_mu[v_mask])
                    + ((d_eta_clean[v_mask] - self.bar_eta[v_mask]) ** 2) / (denom_scale * tbl.var_log_sig[v_mask])
                )
            )
            if d2_jump > 3.0 * max(tbl.q95_s, 1.0):
                self.jump_streak += 1
                if self.jump_streak >= 2 and self.pending_mu is not None and self.pending_eta is not None:
                    p_mu, p_eta = self.pending_mu.copy(), self.pending_eta.copy()
                    self.reset()
                    self.step = 2
                    self.nu = 0.5
                    self.bar_mu = 0.5 * (p_mu + d_mu_clean)
                    self.bar_eta = 0.5 * (p_eta + d_eta_clean)
                    did_reset = True
                else:
                    self.pending_mu = d_mu_clean.copy()
                    self.pending_eta = d_eta_clean.copy()
                    is_isolated_jump = True
            else:
                self.jump_streak = 0
                self.pending_mu = None
                self.pending_eta = None

        if (not did_reset) and (not is_isolated_jump):
            self.step += 1
            if self.step == 1:
                self.bar_mu = d_mu_clean.copy()
                self.bar_eta = d_eta_clean.copy()
                self.nu = 1.0
            else:
                alpha = max(self.alpha_min, 1.0 / float(self.step))
                self.bar_mu = (1.0 - alpha) * self.bar_mu + alpha * d_mu_clean
                self.bar_eta = (1.0 - alpha) * self.bar_eta + alpha * d_eta_clean
                self.nu = ((1.0 - alpha) ** 2) * self.nu + (alpha ** 2)

        act = tbl.active
        d2_bar = float(
            0.5 * np.mean(
                (self.bar_mu[act] ** 2) / tbl.var_mu[act]
                + (self.bar_eta[act] ** 2) / tbl.var_log_sig[act]
            )
        )
        null_inv = 1.5 * max(tbl.q_med_s, 0.25)
        lam_stream = max(0.0, 1.0 - null_inv / max(d2_bar, EPS))

        hat_stream_mu = lam_stream * self.bar_mu
        hat_stream_eta = lam_stream * self.bar_eta

        res_mu = d_mu_clean - hat_stream_mu
        res_eta = d_eta_clean - hat_stream_eta
        d2_res = float(
            0.5 * np.mean(
                (res_mu[v_mask] ** 2) / tbl.var_mu[v_mask]
                + (res_eta[v_mask] ** 2) / tbl.var_log_sig[v_mask]
            )
        )
        lam_indiv = max(0.0, 1.0 - tbl.q95_s / max(d2_res, EPS))

        hat_mu = np.where(v_mask, hat_stream_mu + lam_indiv * res_mu, 0.0)
        hat_eta = np.where(v_mask, hat_stream_eta + lam_indiv * res_eta, 0.0)
        return hat_mu, hat_eta, lam_stream, lam_indiv, did_reset


ShallowOp = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class ShallowBackgroundStreamTTA:
    """Four-step stream-stabilized background-anchored shallow feature calibration."""

    def __init__(
        self,
        model: nn.Module,
        enc_bns: List[Tuple[str, nn.BatchNorm2d, int]],
        erosion_replay: SpatialErosionReplay,
        tables_bg: List[StreamLayerTable],
        tables_body: List[StreamLayerTable],
        output_mode: OutputMode = "multiclass_softmax",
        strip_padding: bool = True,
        exclude_fg: bool = True,
        enable_stream: bool = True,
        enforce_pad_bg: bool = False,
        fg_fallback_cap: float = 0.35,
        alpha_min: float = 0.05,
    ) -> None:
        if output_mode not in ("multiclass_softmax", "multilabel_sigmoid"):
            raise ValueError(f"Unsupported output_mode: {output_mode!r}")
        self.model = model
        self.enc_bns = enc_bns
        self.erosion_replay = erosion_replay
        self.tables_bg = tables_bg
        self.tables_body = tables_body
        self.output_mode: OutputMode = output_mode
        self.strip_padding = bool(strip_padding)
        self.exclude_fg = bool(exclude_fg)
        self.enable_stream = bool(enable_stream)
        self.enforce_pad_bg = bool(enforce_pad_bg)
        self.fg_fallback_cap = float(fg_fallback_cap)
        self.alpha_min = float(alpha_min)
        self.layer_states = [LayerStreamState(len(t.mu_s), alpha_min=alpha_min) for t in tables_bg]

    def reset_stream(self) -> None:
        for st in self.layer_states:
            st.reset()

    def source_state_dict(self) -> Dict[str, object]:
        return {
            "tables_bg": [t.to_dict() for t in self.tables_bg],
            "tables_body": [t.to_dict() for t in self.tables_body],
            "erosion_events": list(self.erosion_replay.events),
            "n_layers": int(self.erosion_replay.n_layers),
            "output_mode": self.output_mode,
            "strip_padding": self.strip_padding,
            "exclude_fg": self.exclude_fg,
            "enable_stream": self.enable_stream,
            "enforce_pad_bg": self.enforce_pad_bg,
            "fg_fallback_cap": self.fg_fallback_cap,
            "alpha_min": self.alpha_min,
        }

    @classmethod
    def from_source_state_dict(
        cls,
        model: nn.Module,
        enc_bns: List[Tuple[str, nn.BatchNorm2d, int]],
        state: Dict[str, object],
    ) -> "ShallowBackgroundStreamTTA":
        replay = SpatialErosionReplay(list(state["erosion_events"]), int(state["n_layers"]))
        tables_bg = [StreamLayerTable.from_dict(d) for d in state["tables_bg"]]
        tables_body = [StreamLayerTable.from_dict(d) for d in state["tables_body"]]
        return cls(
            model=model,
            enc_bns=enc_bns,
            erosion_replay=replay,
            tables_bg=tables_bg,
            tables_body=tables_body,
            output_mode=state["output_mode"],
            strip_padding=bool(state["strip_padding"]),
            exclude_fg=bool(state["exclude_fg"]),
            enable_stream=bool(state["enable_stream"]),
            enforce_pad_bg=bool(state["enforce_pad_bg"]),
            fg_fallback_cap=float(state["fg_fallback_cap"]),
            alpha_min=float(state["alpha_min"]),
        )

    @classmethod
    def from_bn_running_stats(
        cls,
        model: nn.Module,
        shallow_bns: List[Tuple[str, nn.BatchNorm2d, int]],
        dummy_input: torch.Tensor,
        output_mode: OutputMode = "multiclass_softmax",
        strip_padding: bool = True,
        exclude_fg: bool = True,
        enable_stream: bool = True,
        enforce_pad_bg: bool = False,
        fg_fallback_cap: float = 0.35,
        alpha_min: float = 0.05,
    ) -> "ShallowBackgroundStreamTTA":
        """Construct a checkpoint-only shallow calibrator without source images."""
        erosion_replay = SpatialErosionReplay.trace_once(model, shallow_bns, dummy_input)
        tables: List[StreamLayerTable] = []
        for _, bn, _ in shallow_bns:
            assert bn.running_mean is not None and bn.running_var is not None
            rm = bn.running_mean.detach().cpu().numpy().astype(np.float64)
            rv = np.maximum(bn.running_var.detach().cpu().numpy().astype(np.float64), 0.0)
            ep = float(bn.eps)
            sig_s = np.sqrt(rv + ep)
            act = sig_s > 1e-4
            if not np.any(act):
                act = np.ones_like(act, dtype=bool)
            tables.append(
                StreamLayerTable(
                    mu_s=rm,
                    sigma_s=sig_s,
                    log_sig_s=np.log(sig_s + 1e-6),
                    var_mu=rv + ep,
                    var_log_sig=np.ones_like(rm, dtype=np.float64),
                    active=act,
                    q_med_s=0.5,
                    q95_s=2.0,
                )
            )
        return cls(
            model=model,
            enc_bns=shallow_bns,
            erosion_replay=erosion_replay,
            tables_bg=tables,
            tables_body=tables,
            output_mode=output_mode,
            strip_padding=strip_padding,
            exclude_fg=exclude_fg,
            enable_stream=enable_stream,
            enforce_pad_bg=enforce_pad_bg,
            fg_fallback_cap=fg_fallback_cap,
            alpha_min=alpha_min,
        )

    @classmethod
    def fit_from_source(
        cls,
        model: nn.Module,
        train_imgs: Sequence[torch.Tensor],
        device: torch.device,
        output_mode: OutputMode = "multiclass_softmax",
        cached_train_fg_masks: Optional[Sequence[torch.Tensor]] = None,
        max_stride: int = 4,
        strip_padding: bool = True,
        exclude_fg: bool = True,
        enable_stream: bool = True,
        enforce_pad_bg: bool = False,
        fg_fallback_cap: float = 0.35,
        alpha_min: float = 0.05,
    ) -> "ShallowBackgroundStreamTTA":
        """Calibrate shallow BN reference tables strictly on source-domain images."""
        if len(train_imgs) < 2:
            raise ValueError(f"fit_from_source requires >= 2 source images, got {len(train_imgs)}")
        if output_mode not in ("multiclass_softmax", "multilabel_sigmoid"):
            raise ValueError(f"Unsupported output_mode: {output_mode!r}")

        model.eval()
        sample0 = train_imgs[0].unsqueeze(0).to(device)
        enc_bns = discover_encoder_bns_up_to_stride(model, sample0, max_stride=max_stride)
        if not enc_bns:
            raise ValueError(f"No encoder BatchNorm2d layers found up to max_stride={max_stride}")
        erosion_replay = SpatialErosionReplay.trace_once(model, enc_bns, sample0)
        n_layers = len(enc_bns)
        c_dims = [int(bn.num_features) for _, bn, _ in enc_bns]

        n_tr = len(train_imgs)
        body_masks: List[torch.Tensor] = []
        fg_masks: List[torch.Tensor] = []

        for i in range(n_tr):
            im = train_imgs[i]
            body_masks.append(detect_body_mask(im, strip_padding=strip_padding))
            if cached_train_fg_masks is not None:
                fg = cached_train_fg_masks[i].detach().cpu().bool()
            else:
                with torch.no_grad():
                    logits0 = extract_logits(model(im.unsqueeze(0).to(device)))
                    fg = pass0_foreground_mask(logits0, output_mode).detach().cpu()
            fg_masks.append(fg)

        mu_bg = [np.zeros((n_tr, c), dtype=np.float64) for c in c_dims]
        ls_bg = [np.zeros((n_tr, c), dtype=np.float64) for c in c_dims]
        mu_bd = [np.zeros((n_tr, c), dtype=np.float64) for c in c_dims]
        ls_bd = [np.zeros((n_tr, c), dtype=np.float64) for c in c_dims]

        cur_idx = 0
        cur_cleans: List[torch.Tensor] = []

        def _mk_collect(slot: int):
            def _pre(_mod: nn.Module, inp: Tuple[torch.Tensor, ...]) -> None:
                z = inp[0].detach()
                cm = cur_cleans[slot]
                bm0 = body_masks[cur_idx].to(z.device)
                fg0 = fg_masks[cur_idx].to(z.device)
                m_bg, _ = build_layer_bg_stat_mask(
                    cm, bm0, fg0, exclude_fg=exclude_fg, fg_fallback_cap=fg_fallback_cap
                )
                m_bd, _ = build_layer_bg_stat_mask(cm, bm0, None, exclude_fg=False)
                for m_mask, mu_arr, ls_arr in ((m_bg, mu_bg[slot], ls_bg[slot]), (m_bd, mu_bd[slot], ls_bd[slot])):
                    vals = z[0][:, m_mask].double()
                    if vals.shape[1] < 2:
                        vals = z[0].reshape(z.shape[1], -1).double()
                    mu_arr[cur_idx] = vals.mean(dim=1).cpu().numpy()
                    ls_arr[cur_idx] = torch.log(vals.std(dim=1, unbiased=False) + 1e-6).cpu().numpy()
            return _pre

        hooks = [bn.register_forward_pre_hook(_mk_collect(s)) for s, (_, bn, _) in enumerate(enc_bns)]
        try:
            with torch.no_grad():
                for i in range(n_tr):
                    cur_idx = i
                    cur_cleans = erosion_replay.replay(body_masks[i], device)
                    model(train_imgs[i].unsqueeze(0).to(device))
        finally:
            for hk in hooks:
                hk.remove()

        def _make_tables(mu_list: List[np.ndarray], ls_list: List[np.ndarray]) -> List[StreamLayerTable]:
            out: List[StreamLayerTable] = []
            for s in range(n_layers):
                m_tr, l_tr = mu_list[s], ls_list[s]
                mu_s = np.mean(m_tr, axis=0)
                ls_s = np.mean(l_tr, axis=0)
                sig_s = np.exp(ls_s)
                v_mu = np.var(m_tr, axis=0, ddof=1) + 1e-6
                v_ls = np.var(l_tr, axis=0, ddof=1) + 1e-6
                act = (sig_s > 1e-4) & (np.var(m_tr, axis=0, ddof=1) > 1e-8)
                if not np.any(act):
                    act = np.ones_like(act, dtype=bool)
                d2 = 0.5 * np.mean(
                    ((m_tr[:, act] - mu_s[None, act]) ** 2) / v_mu[None, act]
                    + ((l_tr[:, act] - ls_s[None, act]) ** 2) / v_ls[None, act],
                    axis=1,
                )
                out.append(
                    StreamLayerTable(
                        mu_s=mu_s,
                        sigma_s=sig_s,
                        log_sig_s=ls_s,
                        var_mu=v_mu,
                        var_log_sig=v_ls,
                        active=act,
                        q_med_s=float(np.median(d2)),
                        q95_s=float(np.quantile(d2, 0.95)),
                    )
                )
            return out

        return cls(
            model=model,
            enc_bns=enc_bns,
            erosion_replay=erosion_replay,
            tables_bg=_make_tables(mu_bg, ls_bg),
            tables_body=_make_tables(mu_bd, ls_bd),
            output_mode=output_mode,
            strip_padding=strip_padding,
            exclude_fg=exclude_fg,
            enable_stream=enable_stream,
            enforce_pad_bg=enforce_pad_bg,
            fg_fallback_cap=fg_fallback_cap,
            alpha_min=alpha_min,
        )

    def forward_pass0(
        self,
        x_dev: torch.Tensor,
        body0: torch.Tensor,
        body0_dev: torch.Tensor,
        device: torch.device,
        cached_pass0_fg_hw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Optional[ShallowOp]], Dict[str, float]]:
        """Execute Stage-1 shallow calibration forward pass and record per-layer affine ops."""
        if self.exclude_fg:
            if cached_pass0_fg_hw is not None:
                fg0: Optional[torch.Tensor] = cached_pass0_fg_hw.to(device).bool()
            else:
                with torch.no_grad():
                    logits0 = extract_logits(self.model(x_dev))
                    fg0 = pass0_foreground_mask(logits0, self.output_mode)
        else:
            fg0 = None

        clean_masks = self.erosion_replay.replay(body0, device)
        shallow_ops: List[Optional[ShallowOp]] = [None] * len(self.enc_bns)
        apply_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        body0_f = body0_dev.float()[None, None]

        lam_st_list: List[float] = []
        lam_in_list: List[float] = []
        used_fg_list: List[bool] = []
        reset_cnt = 0

        def _get_apply_mask(hl: int, wl: int) -> torch.Tensor:
            m = apply_cache.get((hl, wl))
            if m is None:
                if self.strip_padding:
                    m = (F.interpolate(body0_f, size=(hl, wl), mode="nearest")[0, 0] > 0.5)[None, None]
                else:
                    m = torch.ones((1, 1, hl, wl), dtype=torch.bool, device=device)
                apply_cache[(hl, wl)] = m
            return m

        def _mk_adapt(slot: int):
            def _pre(_mod: nn.Module, inp: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor]:
                nonlocal reset_cnt
                z = inp[0]
                hl, wl = int(z.shape[2]), int(z.shape[3])
                cm = clean_masks[slot].to(device)
                m_stat, used_fg = build_layer_bg_stat_mask(
                    cm, body0_dev, fg0, exclude_fg=self.exclude_fg, fg_fallback_cap=self.fg_fallback_cap
                )
                used_fg_list.append(bool(used_fg))
                tbl = self.tables_bg[slot] if (self.exclude_fg and used_fg) else self.tables_body[slot]

                vals = z[0][:, m_stat].double()
                if vals.shape[1] < 4:
                    lam_st_list.append(0.0)
                    lam_in_list.append(0.0)
                    return (z,)

                mu_cur = vals.mean(dim=1).detach().cpu().numpy()
                sd_cur = vals.std(dim=1, unbiased=False).detach().cpu().numpy()
                ls_cur = np.log(sd_cur + 1e-6)
                valid_ch = sd_cur >= (0.05 * tbl.sigma_s)

                hat_mu, hat_eta, l_st, l_in, did_rst = self.layer_states[slot].observe_and_estimate(
                    mu_cur - tbl.mu_s,
                    ls_cur - tbl.log_sig_s,
                    valid_ch,
                    tbl,
                    enable_stream=self.enable_stream,
                )
                lam_st_list.append(l_st)
                lam_in_list.append(l_in)
                if did_rst:
                    reset_cnt += 1

                if l_st == 0.0 and l_in == 0.0:
                    return (z,)

                mu_base_t = torch.from_numpy((tbl.mu_s + hat_mu).astype(np.float32)).to(device)[None, :, None, None]
                mu_ref_t = torch.from_numpy(tbl.mu_s.astype(np.float32)).to(device)[None, :, None, None]
                gain_np = np.clip(np.exp(-hat_eta), 1.0 / 3.0, 3.0).astype(np.float32)
                gain_t = torch.from_numpy(gain_np).to(device)[None, :, None, None]
                shallow_ops[slot] = (mu_base_t, mu_ref_t, gain_t)

                z_rescaled = (z - mu_base_t) * gain_t + mu_ref_t
                return (torch.where(_get_apply_mask(hl, wl), z_rescaled, z),)
            return _pre

        hooks = [bn.register_forward_pre_hook(_mk_adapt(s)) for s, (_, bn, _) in enumerate(self.enc_bns)]
        try:
            with torch.no_grad():
                logits = extract_logits(self.model(x_dev))
        finally:
            for hk in hooks:
                hk.remove()

        if self.strip_padding and self.enforce_pad_bg:
            pad_hw = ~body0_dev
            if bool(pad_hw.any()):
                logits = logits.clone()
                if self.output_mode == "multiclass_softmax":
                    logits[0, 0, pad_hw] = logits[0, :, pad_hw].max(dim=0).values + 10.0
                else:
                    logits[0, :, pad_hw] = -20.0

        diag = {
            "body_frac": float(body0.float().mean().item()),
            "used_fg_frac": float(np.mean(used_fg_list)) if used_fg_list else 0.0,
            "lam_stream_l00": float(lam_st_list[0]) if lam_st_list else 0.0,
            "lam_stream_max": float(max(lam_st_list)) if lam_st_list else 0.0,
            "lam_stream_mean": float(np.mean(lam_st_list)) if lam_st_list else 0.0,
            "lam_indiv_max": float(max(lam_in_list)) if lam_in_list else 0.0,
            "lam_indiv_mean": float(np.mean(lam_in_list)) if lam_in_list else 0.0,
            "resets": float(reset_cnt),
        }
        return logits, shallow_ops, diag

    def build_replay_hooks(
        self,
        shallow_ops: Sequence[Optional[ShallowOp]],
        body0_dev: torch.Tensor,
        device: torch.device,
    ) -> List[torch.utils.hooks.RemovableHandle]:
        """Attach read-only replay pre-hooks for `shallow_ops` without advancing stream state."""
        body_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        body0_f = body0_dev.float()[None, None]

        def _get_m(hl: int, wl: int) -> torch.Tensor:
            m = body_cache.get((hl, wl))
            if m is None:
                if self.strip_padding:
                    m = F.interpolate(body0_f, size=(hl, wl), mode="nearest") > 0.5
                else:
                    m = torch.ones((1, 1, hl, wl), dtype=torch.bool, device=device)
                body_cache[(hl, wl)] = m
            return m

        hooks = []
        for s, (_, bn, _) in enumerate(self.enc_bns):
            op = shallow_ops[s]
            if op is None:
                continue
            mb, mr, gn = op

            def _mk(mb_t: torch.Tensor, mr_t: torch.Tensor, gn_t: torch.Tensor):
                def _pre(_mod: nn.Module, inp: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor]:
                    z = inp[0]
                    m_apply = _get_m(int(z.shape[2]), int(z.shape[3]))
                    return (torch.where(m_apply, (z - mb_t) * gn_t + mr_t, z),)
                return _pre

            hooks.append(bn.register_forward_pre_hook(_mk(mb, mr, gn)))
        return hooks

    def predict_single(
        self,
        img_chw: torch.Tensor,
        device: torch.device,
        cached_pass0_fg_hw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Run standalone Stage-1 shallow calibration on one `[C, H, W]` image tensor."""
        self.model.eval()
        body0 = detect_body_mask(img_chw, strip_padding=self.strip_padding)
        x_dev = img_chw.unsqueeze(0).to(device)
        body0_dev = body0.to(device)
        logits, _, diag = self.forward_pass0(x_dev, body0, body0_dev, device, cached_pass0_fg_hw)
        return logits, diag
