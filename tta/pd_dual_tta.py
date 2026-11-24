"""Prevalence-Deconfounded Dual-Regime Test-Time Adaptation (`PrevalenceDeconfoundedDualTTA`).

Unified backprop-free single-sample (`batch_size=1`) online TTA operator for medical image
segmentation under coupled appearance shift, organ-prevalence shift, and empty organ slices:
  - Stage 1: Background-anchored shallow feature calibration (`stride <= 4`).
  - Stage 2: Anatomical presence gating & conservative encoder-only dim-foreground rescue.
  - Stage 3: Prevalence-matched source deep anchoring with support & residual-shift shrinkage.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .config import PDDualConfig, WEIGHT_ONLY_SF_CONFIG
from .deep_bank import DeepResidualStreamState, DeepSourceBank
from .masks import (
    EPS,
    OutputMode,
    detect_body_mask,
    discover_bn_partition,
    extract_logits,
    extract_prevalence_vector,
)
from .shallow_frontend import ShallowBackgroundStreamTTA, ShallowOp


class PrevalenceDeconfoundedDualTTA:
    """Unified 3-stage prevalence-deconfounded shallow + deep online TTA operator."""

    def __init__(
        self,
        model: nn.Module,
        shallow_tta: ShallowBackgroundStreamTTA,
        deep_enc_bns: List[Tuple[str, nn.BatchNorm2d, int]],
        deep_dec_bns: List[Tuple[str, nn.BatchNorm2d, int]],
        source_bank: DeepSourceBank,
        output_mode: OutputMode,
        device: torch.device,
        config: PDDualConfig = PDDualConfig(),
        input_shape_chw: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        self.model = model
        self.shallow_tta = shallow_tta
        self.shallow_bns = shallow_tta.enc_bns
        self.deep_enc_bns = deep_enc_bns
        self.deep_dec_bns = deep_dec_bns
        self.deep_bns = list(deep_enc_bns) + list(deep_dec_bns)
        self.n_deep_enc = len(deep_enc_bns)
        self.source_bank = source_bank
        self.output_mode: OutputMode = output_mode
        self.device = device
        self.config = config
        self.input_shape_chw = input_shape_chw

        c_dims = [int(bn.num_features) for _, bn, _ in self.deep_bns]
        self.deep_state = DeepResidualStreamState(c_dims, device=device, ema_min=config.deep_ema_min)

    def reset_stream(self) -> None:
        """Reset both shallow and deep online stream ledgers."""
        self.shallow_tta.reset_stream()
        self.deep_state.reset()

    def with_config(self, config: PDDualConfig) -> "PrevalenceDeconfoundedDualTTA":
        """Create a fresh stream instance sharing calibrated source tables under `config`."""
        return PrevalenceDeconfoundedDualTTA(
            model=self.model,
            shallow_tta=self.shallow_tta,
            deep_enc_bns=self.deep_enc_bns,
            deep_dec_bns=self.deep_dec_bns,
            source_bank=self.source_bank,
            output_mode=self.output_mode,
            device=self.device,
            config=config,
            input_shape_chw=self.input_shape_chw,
        )

    @classmethod
    def fit_from_source(
        cls,
        model: nn.Module,
        train_imgs: Sequence[torch.Tensor],
        device: torch.device,
        output_mode: OutputMode = "multiclass_softmax",
        config: PDDualConfig = PDDualConfig(),
        cached_shallow_tta: Optional[ShallowBackgroundStreamTTA] = None,
    ) -> "PrevalenceDeconfoundedDualTTA":
        """Calibrate shallow tables and prevalence-indexed deep bank strictly on source images."""
        if len(train_imgs) < 2:
            raise ValueError(f"fit_from_source requires >= 2 source images, got {len(train_imgs)}")
        if output_mode not in ("multiclass_softmax", "multilabel_sigmoid"):
            raise ValueError(f"Unsupported output_mode: {output_mode!r}")

        model.eval()
        s0 = train_imgs[0]
        input_shape_chw = (int(s0.shape[0]), int(s0.shape[1]), int(s0.shape[2]))
        sample0 = s0.unsqueeze(0).to(device)
        shallow_bns, deep_enc_bns, deep_dec_bns = discover_bn_partition(
            model, sample0, max_stride=config.max_shallow_stride
        )
        deep_bns = list(deep_enc_bns) + list(deep_dec_bns)
        if not deep_bns:
            raise RuntimeError("No deep BatchNorm2d layers discovered outside shallow stride partition.")

        shallow_tta = cached_shallow_tta or ShallowBackgroundStreamTTA.fit_from_source(
            model=model,
            train_imgs=train_imgs,
            device=device,
            output_mode=output_mode,
            max_stride=config.max_shallow_stride,
            strip_padding=config.strip_padding,
            exclude_fg=config.exclude_fg,
            enable_stream=config.enable_shallow_stream,
            enforce_pad_bg=config.enforce_pad_bg,
            fg_fallback_cap=config.fg_fallback_cap,
            alpha_min=config.shallow_alpha_min,
        )

        n_total = len(train_imgs)
        if n_total > config.max_cal_samples:
            cal_idxs = np.linspace(0, n_total - 1, config.max_cal_samples, dtype=int).tolist()
        else:
            cal_idxs = list(range(n_total))
        n_cal = len(cal_idxs)

        c_dims = [int(bn.num_features) for _, bn, _ in deep_bns]
        bn_eps = [float(bn.eps) for _, bn, _ in deep_bns]
        mu_all = [np.zeros((n_cal, c), dtype=np.float32) for c in c_dims]
        var_all = [np.zeros((n_cal, c), dtype=np.float32) for c in c_dims]
        cur_slot_idx = 0

        def _mk_collect(slot: int):
            def _pre(_mod: nn.Module, inp: Tuple[torch.Tensor, ...]) -> None:
                z = inp[0].detach()
                mu_all[slot][cur_slot_idx] = z.mean(dim=(0, 2, 3)).cpu().numpy()
                var_all[slot][cur_slot_idx] = z.var(dim=(0, 2, 3), unbiased=False).cpu().numpy()
            return _pre

        hooks = [bn.register_forward_pre_hook(_mk_collect(s)) for s, (_, bn, _) in enumerate(deep_bns)]
        pi_list: List[np.ndarray] = []
        r_fg_list: List[float] = []
        try:
            with torch.no_grad():
                for ci, img_idx in enumerate(cal_idxs):
                    cur_slot_idx = ci
                    im = train_imgs[img_idx]
                    bm = detect_body_mask(im, strip_padding=config.strip_padding).to(device)
                    logits = extract_logits(model(im.unsqueeze(0).to(device)))
                    pi_vec, r_fg = extract_prevalence_vector(logits, bm, output_mode)
                    pi_list.append(pi_vec)
                    r_fg_list.append(r_fg)
        finally:
            for hk in hooks:
                hk.remove()

        source_bank = DeepSourceBank.build(
            mu_all=mu_all,
            var_all=var_all,
            bn_eps=bn_eps,
            pi_arr=np.stack(pi_list, axis=0),
            r_fg_arr=np.asarray(r_fg_list, dtype=np.float64),
            config=config,
            device=device,
        )

        return cls(
            model=model,
            shallow_tta=shallow_tta,
            deep_enc_bns=deep_enc_bns,
            deep_dec_bns=deep_dec_bns,
            source_bank=source_bank,
            output_mode=output_mode,
            device=device,
            config=config,
            input_shape_chw=input_shape_chw,
        )

    @classmethod
    def from_checkpoint_only(
        cls,
        model: nn.Module,
        input_shape_chw: Tuple[int, int, int],
        device: torch.device,
        output_mode: OutputMode = "multiclass_softmax",
        config: PDDualConfig = WEIGHT_ONLY_SF_CONFIG,
    ) -> "PrevalenceDeconfoundedDualTTA":
        """Build a strict weight-only operator using solely frozen `.pth` BN running stats."""
        if output_mode not in ("multiclass_softmax", "multilabel_sigmoid"):
            raise ValueError(f"Unsupported output_mode: {output_mode!r}")

        model.eval()
        c_in, h_in, w_in = int(input_shape_chw[0]), int(input_shape_chw[1]), int(input_shape_chw[2])
        dummy = torch.zeros((1, c_in, h_in, w_in), dtype=torch.float32, device=device)

        shallow_bns, deep_enc_bns, deep_dec_bns = discover_bn_partition(
            model, dummy, max_stride=config.max_shallow_stride
        )
        deep_bns = list(deep_enc_bns) + list(deep_dec_bns)
        if not shallow_bns or not deep_bns:
            raise RuntimeError("Failed to discover shallow and deep BatchNorm2d partitions.")

        with torch.no_grad():
            c_out = int(extract_logits(model(dummy)).shape[1])
        k_fg = max(1, c_out - 1) if output_mode == "multiclass_softmax" else c_out

        shallow_tta = ShallowBackgroundStreamTTA.from_bn_running_stats(
            model=model,
            shallow_bns=shallow_bns,
            dummy_input=dummy,
            output_mode=output_mode,
            strip_padding=config.strip_padding,
            exclude_fg=config.exclude_fg,
            enable_stream=config.enable_shallow_stream,
            enforce_pad_bg=config.enforce_pad_bg,
            fg_fallback_cap=config.fg_fallback_cap,
            alpha_min=config.shallow_alpha_min,
        )
        source_bank = DeepSourceBank.from_bn_running_stats(deep_bns, k_fg=k_fg, device=device)
        return cls(
            model=model,
            shallow_tta=shallow_tta,
            deep_enc_bns=deep_enc_bns,
            deep_dec_bns=deep_dec_bns,
            source_bank=source_bank,
            output_mode=output_mode,
            device=device,
            config=config,
            input_shape_chw=(c_in, h_in, w_in),
        )

    def save_source_summary(self, path: Union[str, Path]) -> Path:
        """Save anonymous channel-moment summary tables to `.pt` (contains zero patient images)."""
        if self.input_shape_chw is None:
            raise ValueError("input_shape_chw is required to serialize a standalone source summary.")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "input_shape_chw": tuple(self.input_shape_chw),
            "output_mode": self.output_mode,
            "config": asdict(self.config),
            "shallow": self.shallow_tta.source_state_dict(),
            "deep_bank": self.source_bank.to_state_dict(),
        }
        torch.save(payload, str(out))
        return out

    @classmethod
    def from_source_summary(
        cls,
        model: nn.Module,
        path: Union[str, Path],
        device: torch.device,
        config: Optional[PDDualConfig] = None,
    ) -> "PrevalenceDeconfoundedDualTTA":
        """Instantiate directly from a pre-exported `.pt` source summary (zero source images needed)."""
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        cfg = config if config is not None else PDDualConfig(**payload["config"])
        c_in, h_in, w_in = payload["input_shape_chw"]
        model.eval()
        dummy = torch.zeros((1, int(c_in), int(h_in), int(w_in)), dtype=torch.float32, device=device)
        shallow_bns, deep_enc_bns, deep_dec_bns = discover_bn_partition(
            model, dummy, max_stride=cfg.max_shallow_stride
        )
        shallow_tta = ShallowBackgroundStreamTTA.from_source_state_dict(
            model=model, enc_bns=shallow_bns, state=payload["shallow"]
        )
        source_bank = DeepSourceBank.from_state_dict(payload["deep_bank"], device=device)
        return cls(
            model=model,
            shallow_tta=shallow_tta,
            deep_enc_bns=deep_enc_bns,
            deep_dec_bns=deep_dec_bns,
            source_bank=source_bank,
            output_mode=payload["output_mode"],
            device=device,
            config=cfg,
            input_shape_chw=(int(c_in), int(h_in), int(w_in)),
        )

    def _run_enc_rescue_probe(
        self,
        x_dev: torch.Tensor,
        body0_dev: torch.Tensor,
        shallow_ops: Sequence[Optional[ShallowOp]],
    ) -> Tuple[np.ndarray, float]:
        """Conservative encoder-only deep probe (decoder strictly `eval()`) for dim foreground."""
        cfg = self.config
        mode = cfg.rescue_mode
        if mode == "none":
            return np.zeros(self.source_bank.k_fg, dtype=np.float64), 0.0

        use_ema = (mode in ("enc_ema", "enc_ema_fallback_inst")) and (self.deep_state.n_raw_pres >= 1)
        if mode == "enc_ema" and not use_ema:
            return np.zeros(self.source_bank.k_fg, dtype=np.float64), 0.0

        alpha_r = cfg.rescue_alpha if use_ema else (0.6 * cfg.rescue_alpha if mode == "enc_ema_fallback_inst" else cfg.rescue_alpha)
        clamp_hi = cfg.scale_clamp
        clamp_lo = 1.0 / clamp_hi

        hooks = self.shallow_tta.build_replay_hooks(shallow_ops, body0_dev, self.device)
        bank = self.source_bank
        for l_idx in range(self.n_deep_enc):
            _, bn, _ = self.deep_enc_bns[l_idx]
            sm, sv, ep = bn.running_mean, bn.running_var, float(bn.eps)
            act = bank.active_list[l_idx]
            u_mu = bank.uncond_mu_list[l_idx]

            if use_ema:
                d_mu_ema = self.deep_state.enc_raw_d_mu[l_idx]
                d_eta_ema = self.deep_state.enc_raw_d_eta[l_idx]

                def _mk_ema(um: torch.Tensor, dmu: torch.Tensor, deta: torch.Tensor, a_r: float, ac: torch.Tensor):
                    shift = torch.where(ac, a_r * dmu, torch.zeros_like(dmu)).view(1, -1, 1, 1)
                    gain = torch.where(
                        ac, torch.clamp(torch.exp(-a_r * deta), clamp_lo, clamp_hi), torch.ones_like(deta)
                    ).view(1, -1, 1, 1)
                    base = um.view(1, -1, 1, 1) + shift
                    um_4d = um.view(1, -1, 1, 1)

                    def _pre(_mod: nn.Module, inp: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor]:
                        return ((inp[0] - base) * gain + um_4d,)
                    return _pre

                hooks.append(bn.register_forward_pre_hook(_mk_ema(u_mu, d_mu_ema, d_eta_ema, alpha_r, act)))
            else:
                def _mk_inst(s_m: torch.Tensor, s_v: torch.Tensor, e_p: float, a_r: float):
                    sm_4d = s_m.view(1, -1, 1, 1)
                    sv_ep = s_v + e_p

                    def _pre(_mod: nn.Module, inp: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor]:
                        z = inp[0]
                        mc = z.mean(dim=(0, 2, 3))
                        vc = z.var(dim=(0, 2, 3), unbiased=False)
                        mb = (1.0 - a_r) * s_m + a_r * mc
                        vb = (1.0 - a_r) * s_v + a_r * vc + a_r * (1.0 - a_r) * (mc - s_m) ** 2
                        sc = torch.clamp(torch.sqrt(sv_ep / (vb + e_p)), clamp_lo, clamp_hi)
                        return ((z - mb.view(1, -1, 1, 1)) * sc.view(1, -1, 1, 1) + sm_4d,)
                    return _pre

                hooks.append(bn.register_forward_pre_hook(_mk_inst(sm, sv, ep, alpha_r)))

        try:
            with torch.no_grad():
                logits_r = extract_logits(self.model(x_dev))
        finally:
            for hk in hooks:
                hk.remove()

        return extract_prevalence_vector(logits_r, body0_dev, self.output_mode)

    def _run_deep_pass1(
        self,
        x_dev: torch.Tensor,
        body0_dev: torch.Tensor,
        shallow_ops: Sequence[Optional[ShallowOp]],
        mu_hat_list: Sequence[torch.Tensor],
        var_hat_list: Sequence[torch.Tensor],
        w_supp: float,
        update_raw_enc_ema: bool,
    ) -> Tuple[torch.Tensor, float, float]:
        """Run Stage-3 prevalence-deconfounded deep Pass-1 without mutating model weights/buffers."""
        cfg = self.config
        bank = self.source_bank
        state = self.deep_state
        clamp_hi = cfg.scale_clamp
        clamp_lo = 1.0 / clamp_hi

        do_stream_upd = (cfg.deep_mode == "hybrid") and (w_supp > 0.1)
        if do_stream_upd:
            state.n_pres += 1
            state.n_eff += w_supp
            beta_stream = max(cfg.deep_ema_min, w_supp / max(state.n_eff, 1.0))
        else:
            beta_stream = 0.0

        if update_raw_enc_ema:
            state.n_raw_pres += 1
            beta_raw = max(cfg.deep_ema_min, 1.0 / float(state.n_raw_pres))
        else:
            beta_raw = 0.0

        w_shift_vals: List[torch.Tensor] = []
        lam_eff_vals: List[torch.Tensor] = []

        hooks = self.shallow_tta.build_replay_hooks(shallow_ops, body0_dev, self.device)

        for l_idx, (_, bn, _) in enumerate(self.deep_bns):
            ep = float(bn.eps)
            mu_s = mu_hat_list[l_idx]
            var_s = var_hat_list[l_idx]
            sd_s = torch.sqrt(var_s + ep)
            tau_mu_v = bank.tau_mu_var_list[l_idx]
            tau_eta_v = bank.tau_eta_var_list[l_idx]
            act_l = bank.active_list[l_idx]
            q_med_l = bank.q_med_list[l_idx]
            q90_l = bank.q90_list[l_idx]
            is_enc = l_idx < self.n_deep_enc
            u_mu_l = bank.uncond_mu_list[l_idx]
            u_sd_l = torch.sqrt(bank.uncond_var_list[l_idx] + ep)

            def _mk_deep(
                slot: int,
                e_p: float,
                m_s: torch.Tensor,
                v_s: torch.Tensor,
                s_s: torch.Tensor,
                tmv: torch.Tensor,
                tev: torch.Tensor,
                act: torch.Tensor,
                qm: float,
                q90: float,
                enc_flag: bool,
                um: torch.Tensor,
                usd: torch.Tensor,
            ):
                v_s_ep = v_s + e_p
                alpha_w = cfg.alpha_max * w_supp

                def _pre(_mod: nn.Module, inp: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor]:
                    z = inp[0]
                    mu_cur = z.mean(dim=(0, 2, 3))
                    var_cur = z.var(dim=(0, 2, 3), unbiased=False)
                    sd_cur = torch.sqrt(var_cur + e_p)

                    if update_raw_enc_ema and enc_flag:
                        raw_dmu = mu_cur - um
                        raw_deta = torch.log(sd_cur) - torch.log(usd)
                        if state.n_raw_pres == 1:
                            state.enc_raw_d_mu[slot] = raw_dmu.detach()
                            state.enc_raw_d_eta[slot] = raw_deta.detach()
                        else:
                            state.enc_raw_d_mu[slot] = (1.0 - beta_raw) * state.enc_raw_d_mu[slot] + beta_raw * raw_dmu.detach()
                            state.enc_raw_d_eta[slot] = (1.0 - beta_raw) * state.enc_raw_d_eta[slot] + beta_raw * raw_deta.detach()

                    valid = act & (sd_cur >= 0.05 * s_s)
                    valid_f = valid.float()
                    n_val = valid_f.sum().clamp_min(1.0)

                    d_mu = torch.where(valid, mu_cur - m_s, torch.zeros_like(mu_cur))
                    d_eta = torch.where(valid, torch.log(sd_cur) - torch.log(s_s), torch.zeros_like(sd_cur))

                    if cfg.deep_mode == "inst":
                        if cfg.use_shift_gate:
                            d2 = 0.5 * ((d_mu * d_mu) / tmv + (d_eta * d_eta) / tev)
                            d2_mean = (d2 * valid_f).sum() / n_val
                            w_sh = torch.clamp(1.0 - (cfg.shift_kappa_in * q90) / d2_mean.clamp_min(EPS), min=0.0)
                        else:
                            w_sh = torch.ones((), device=z.device, dtype=torch.float32)

                        lam = alpha_w * w_sh
                        w_shift_vals.append(w_sh.detach())
                        lam_eff_vals.append(lam.detach())

                        var_mix = (1.0 - lam) * v_s + lam * var_cur + lam * (1.0 - lam) * (d_mu * d_mu)
                        gain = torch.where(
                            valid,
                            torch.clamp(torch.sqrt(v_s_ep / (var_mix + e_p)), clamp_lo, clamp_hi),
                            torch.ones_like(v_s),
                        )
                        offset = m_s * (1.0 - gain) - (lam * d_mu) * gain
                        return (z * gain.view(1, -1, 1, 1) + offset.view(1, -1, 1, 1),)

                    if do_stream_upd:
                        if state.n_pres == 1:
                            state.bar_d_mu[slot] = d_mu.detach()
                            state.bar_d_eta[slot] = d_eta.detach()
                        else:
                            state.bar_d_mu[slot] = (1.0 - beta_stream) * state.bar_d_mu[slot] + beta_stream * d_mu.detach()
                            state.bar_d_eta[slot] = (1.0 - beta_stream) * state.bar_d_eta[slot] + beta_stream * d_eta.detach()

                    bar_mu = state.bar_d_mu[slot]
                    bar_eta = state.bar_d_eta[slot]
                    act_f = act.float()
                    n_act = act_f.sum().clamp_min(1.0)

                    if cfg.use_shift_gate:
                        d2_bar = 0.5 * ((bar_mu * bar_mu) / tmv + (bar_eta * bar_eta) / tev)
                        d2_bar_mean = (d2_bar * act_f).sum() / n_act
                        w_st = torch.clamp(1.0 - (cfg.shift_kappa_st * qm) / d2_bar_mean.clamp_min(EPS), min=0.0)
                    else:
                        w_st = torch.ones((), device=z.device, dtype=torch.float32)

                    hat_st_mu = w_st * bar_mu
                    hat_st_eta = w_st * bar_eta
                    res_mu = d_mu - hat_st_mu
                    res_eta = d_eta - hat_st_eta

                    if cfg.use_shift_gate:
                        d2_res = 0.5 * ((res_mu * res_mu) / tmv + (res_eta * res_eta) / tev)
                        d2_res_mean = (d2_res * valid_f).sum() / n_val
                        w_in = torch.clamp(1.0 - (cfg.shift_kappa_in * q90) / d2_res_mean.clamp_min(EPS), min=0.0)
                    else:
                        w_in = torch.ones((), device=z.device, dtype=torch.float32)

                    w_comb = torch.clamp(w_st + (1.0 - w_st) * w_in, 0.0, 1.0)
                    w_shift_vals.append(w_comb.detach())
                    lam_eff_vals.append((alpha_w * w_comb).detach())

                    hat_dmu = torch.where(valid, alpha_w * (hat_st_mu + w_in * res_mu), torch.zeros_like(d_mu))
                    hat_deta = torch.where(valid, alpha_w * (hat_st_eta + w_in * res_eta), torch.zeros_like(d_eta))

                    gain = torch.clamp(torch.exp(-hat_deta), clamp_lo, clamp_hi)
                    offset = m_s * (1.0 - gain) - hat_dmu * gain
                    return (z * gain.view(1, -1, 1, 1) + offset.view(1, -1, 1, 1),)

                return _pre

            hooks.append(
                bn.register_forward_pre_hook(
                    _mk_deep(
                        l_idx, ep, mu_s, var_s, sd_s, tau_mu_v, tau_eta_v,
                        act_l, q_med_l, q90_l, is_enc, u_mu_l, u_sd_l
                    )
                )
            )

        try:
            with torch.no_grad():
                logits_p1 = extract_logits(self.model(x_dev))
        finally:
            for hk in hooks:
                hk.remove()

        mean_w_shift = float(torch.stack(w_shift_vals).mean().item()) if w_shift_vals else 0.0
        mean_lam = float(torch.stack(lam_eff_vals).mean().item()) if lam_eff_vals else 0.0
        return logits_p1, mean_w_shift, mean_lam

    def _run_stage2_and_3(
        self,
        x_dev: torch.Tensor,
        body0_dev: torch.Tensor,
        logits_p0: torch.Tensor,
        shallow_ops: Sequence[Optional[ShallowOp]],
        diag: Dict[str, float],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.config
        if not cfg.enable_deep:
            diag.update({
                "p0_r_fg": 0.0,
                "rescued": 0.0,
                "w_supp": 0.0,
                "mean_d_m": 0.0,
                "w_shift_mean": 0.0,
                "lam_deep_mean": 0.0,
                "deep_ran": 0.0,
            })
            return logits_p0, diag

        pi_t, r_fg_p0 = extract_prevalence_vector(logits_p0, body0_dev, self.output_mode)
        rescued = False
        is_present = r_fg_p0 >= cfg.tau_pres

        if (not is_present) and (cfg.rescue_mode != "none"):
            pi_r, r_fg_r = self._run_enc_rescue_probe(x_dev, body0_dev, shallow_ops)
            if r_fg_r >= cfg.tau_rescue:
                pi_t = pi_r
                is_present = True
                rescued = True

        if not is_present:
            diag.update({
                "p0_r_fg": float(r_fg_p0),
                "rescued": 0.0,
                "w_supp": 0.0,
                "mean_d_m": 0.0,
                "w_shift_mean": 0.0,
                "lam_deep_mean": 0.0,
                "deep_ran": 0.0,
            })
            return logits_p0, diag

        mu_hat_list, var_hat_list, w_supp, mean_d_m = self.source_bank.query(
            pi_t,
            use_prev_match=cfg.use_prev_match,
            use_supp_gate=cfg.use_supp_gate,
            gamma_supp=cfg.gamma_supp,
            device=self.device,
        )
        if (cfg.alpha_max * w_supp) < 1e-3:
            diag.update({
                "p0_r_fg": float(r_fg_p0),
                "rescued": 1.0 if rescued else 0.0,
                "w_supp": float(w_supp),
                "mean_d_m": float(mean_d_m),
                "w_shift_mean": 0.0,
                "lam_deep_mean": 0.0,
                "deep_ran": 0.0,
            })
            return logits_p0, diag

        logits_p1, w_shift_mean, lam_deep_mean = self._run_deep_pass1(
            x_dev=x_dev,
            body0_dev=body0_dev,
            shallow_ops=shallow_ops,
            mu_hat_list=mu_hat_list,
            var_hat_list=var_hat_list,
            w_supp=w_supp,
            update_raw_enc_ema=(not rescued) and (w_supp > 0.2),
        )

        diag.update({
            "p0_r_fg": float(r_fg_p0),
            "rescued": 1.0 if rescued else 0.0,
            "w_supp": float(w_supp),
            "mean_d_m": float(mean_d_m),
            "w_shift_mean": float(w_shift_mean),
            "lam_deep_mean": float(lam_deep_mean),
            "deep_ran": 1.0,
        })
        return logits_p1, diag

    def predict_single(
        self,
        img_chw: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Run backprop-free `batch_size=1` adapted inference on `[C, H, W]` without mutating `model`."""
        dev = self.device if device is None else device
        self.model.eval()
        body0 = detect_body_mask(img_chw, strip_padding=self.config.strip_padding)
        x_dev = img_chw.unsqueeze(0).to(dev)
        body0_dev = body0.to(dev)
        logits_p0, shallow_ops, diag = self.shallow_tta.forward_pass0(x_dev, body0, body0_dev, dev)
        return self._run_stage2_and_3(x_dev, body0_dev, logits_p0, shallow_ops, diag)

    def predict_from_cached_pass0(
        self,
        img_chw: torch.Tensor,
        body0: torch.Tensor,
        logits_p0: torch.Tensor,
        shallow_ops: Sequence[Optional[ShallowOp]],
        base_diag: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Replay Stages 2 & 3 from cached Stage-1 outputs (bitwise identical to `predict_single`)."""
        self.model.eval()
        x_dev = img_chw.unsqueeze(0).to(self.device)
        body0_dev = body0.to(self.device)
        diag = dict(base_diag) if base_diag is not None else {}
        return self._run_stage2_and_3(x_dev, body0_dev, logits_p0.to(self.device), shallow_ops, diag)
