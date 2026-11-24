"""Stage 3 source calibration bank and running residual stream state."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import PDDualConfig


@dataclass
class DeepSourceBank:
    """Prevalence-indexed source deep BatchNorm moment bank with LOO variance & support thresholds."""

    k_fg: int
    psi_scaled: np.ndarray
    psi_scale: np.ndarray
    m_neighbors: int
    d_sq: float
    h_bw: float
    source_empty_frac: float

    mu_bank_cat: torch.Tensor
    var_bank_cat: torch.Tensor
    uncond_mu_list: List[torch.Tensor]
    uncond_var_list: List[torch.Tensor]
    layer_slices: List[Tuple[int, int]]
    tau_mu_var_list: List[torch.Tensor]
    tau_eta_var_list: List[torch.Tensor]
    active_list: List[torch.Tensor]
    q_med_list: List[float]
    q90_list: List[float]

    @classmethod
    def build(
        cls,
        mu_all: List[np.ndarray],
        var_all: List[np.ndarray],
        bn_eps: List[float],
        pi_arr: np.ndarray,
        r_fg_arr: np.ndarray,
        config: PDDualConfig,
        device: torch.device,
    ) -> "DeepSourceBank":
        """Build LOO-calibrated prevalence-indexed deep bank strictly from source statistics."""
        n_cal, k_fg = pi_arr.shape
        n_deep = len(mu_all)

        uncond_mu_list = [
            torch.from_numpy(np.mean(mu_all[l], axis=0).astype(np.float32)).to(device)
            for l in range(n_deep)
        ]
        uncond_var_list = [
            torch.from_numpy(np.mean(var_all[l], axis=0).astype(np.float32)).to(device)
            for l in range(n_deep)
        ]

        fg_mask = r_fg_arr >= config.tau_pres
        source_empty_frac = float(np.mean(~fg_mask))
        fg_idxs = np.where(fg_mask)[0]
        if len(fg_idxs) < 4:
            fg_idxs = np.arange(n_cal)
        n_fg = len(fg_idxs)

        psi_fg = np.sqrt(np.clip(pi_arr[fg_idxs], 0.0, 1.0))
        q75 = np.percentile(psi_fg, 75.0, axis=0)
        q25 = np.percentile(psi_fg, 25.0, axis=0)
        iqr_std = (q75 - q25) / 1.349
        emp_std = np.std(psi_fg, axis=0)
        psi_scale = np.maximum(np.maximum(iqr_std, emp_std), 0.02)
        psi_scaled = psi_fg / psi_scale[None, :]

        m_neighbors = min(24, max(6, int(0.20 * n_fg)))
        m_neighbors = max(1, min(m_neighbors, n_fg - 1))

        diff = psi_scaled[:, None, :] - psi_scaled[None, :, :]
        dist_mat = np.sqrt(np.mean(diff * diff, axis=2))
        np.fill_diagonal(dist_mat, np.inf)

        nn_idx = np.argpartition(dist_mat, kth=m_neighbors - 1, axis=1)[:, :m_neighbors]
        nn_dists = np.take_along_axis(dist_mat, nn_idx, axis=1)
        loo_mean_d = np.mean(nn_dists, axis=1)

        d_sq = max(float(np.quantile(loo_mean_d, config.supp_quantile)), 0.15)
        h_bw = max(float(np.median(loo_mean_d)), 0.10)

        logits_w = -0.5 * ((nn_dists / h_bw) ** 2)
        logits_w -= np.max(logits_w, axis=1, keepdims=True)
        w_loo = np.exp(logits_w)
        w_loo /= np.sum(w_loo, axis=1, keepdims=True)

        layer_slices: List[Tuple[int, int]] = []
        mu_fg_blocks: List[np.ndarray] = []
        var_fg_blocks: List[np.ndarray] = []
        tau_mu_var_list: List[torch.Tensor] = []
        tau_eta_var_list: List[torch.Tensor] = []
        active_list: List[torch.Tensor] = []
        q_med_list: List[float] = []
        q90_list: List[float] = []

        offset = 0
        for l in range(n_deep):
            mu_l = mu_all[l][fg_idxs].astype(np.float64)
            var_l = var_all[l][fg_idxs].astype(np.float64)
            c_l = mu_l.shape[1]
            layer_slices.append((offset, offset + c_l))
            offset += c_l
            mu_fg_blocks.append(mu_l.astype(np.float32))
            var_fg_blocks.append(var_l.astype(np.float32))

            mu_hat_loo = np.einsum("nm,nmc->nc", w_loo, mu_l[nn_idx])
            var_hat_loo = np.einsum("nm,nmc->nc", w_loo, var_l[nn_idx])

            ep = bn_eps[l]
            eta_l = 0.5 * np.log(np.maximum(var_l, 0.0) + ep)
            eta_hat_loo = 0.5 * np.log(np.maximum(var_hat_loo, 0.0) + ep)

            res_mu = mu_l - mu_hat_loo
            res_eta = eta_l - eta_hat_loo

            tau_mu_v = np.mean(res_mu ** 2, axis=0) + 1e-5
            tau_eta_v = np.mean(res_eta ** 2, axis=0) + 1e-5

            mean_sd = np.sqrt(np.mean(np.maximum(var_l, 0.0), axis=0) + ep)
            act = (mean_sd > 1e-3) & (np.var(mu_l, axis=0) > 1e-7)
            if not np.any(act):
                act = np.ones(c_l, dtype=bool)

            d2_loo = 0.5 * np.mean(
                (res_mu[:, act] ** 2) / tau_mu_v[None, act]
                + (res_eta[:, act] ** 2) / tau_eta_v[None, act],
                axis=1,
            )
            q_med = max(float(np.median(d2_loo)), 0.25)
            q90 = max(float(np.quantile(d2_loo, 0.90)), q_med)

            tau_mu_var_list.append(torch.from_numpy(tau_mu_v.astype(np.float32)).to(device))
            tau_eta_var_list.append(torch.from_numpy(tau_eta_v.astype(np.float32)).to(device))
            active_list.append(torch.from_numpy(act).to(device))
            q_med_list.append(q_med)
            q90_list.append(q90)

        return cls(
            k_fg=k_fg,
            psi_scaled=psi_scaled,
            psi_scale=psi_scale,
            m_neighbors=m_neighbors,
            d_sq=d_sq,
            h_bw=h_bw,
            source_empty_frac=source_empty_frac,
            mu_bank_cat=torch.from_numpy(np.concatenate(mu_fg_blocks, axis=1)).to(device),
            var_bank_cat=torch.from_numpy(np.concatenate(var_fg_blocks, axis=1)).to(device),
            uncond_mu_list=uncond_mu_list,
            uncond_var_list=uncond_var_list,
            layer_slices=layer_slices,
            tau_mu_var_list=tau_mu_var_list,
            tau_eta_var_list=tau_eta_var_list,
            active_list=active_list,
            q_med_list=q_med_list,
            q90_list=q90_list,
        )

    @classmethod
    def from_bn_running_stats(
        cls,
        deep_bns: Sequence[Tuple[str, nn.BatchNorm2d, int]],
        k_fg: int,
        device: torch.device,
    ) -> "DeepSourceBank":
        """Construct a checkpoint-only deep bank using frozen BN `running_mean`/`running_var`."""
        uncond_mu_list: List[torch.Tensor] = []
        uncond_var_list: List[torch.Tensor] = []
        layer_slices: List[Tuple[int, int]] = []
        tau_mu_var_list: List[torch.Tensor] = []
        tau_eta_var_list: List[torch.Tensor] = []
        active_list: List[torch.Tensor] = []
        q_med_list: List[float] = []
        q90_list: List[float] = []

        offset = 0
        for _, bn, _ in deep_bns:
            assert bn.running_mean is not None and bn.running_var is not None
            rm_t = bn.running_mean.detach().clone().to(device=device, dtype=torch.float32)
            rv_t = bn.running_var.detach().clone().to(device=device, dtype=torch.float32).clamp_min(0.0)
            ep = float(bn.eps)
            c_l = int(rm_t.shape[0])
            layer_slices.append((offset, offset + c_l))
            offset += c_l

            uncond_mu_list.append(rm_t)
            uncond_var_list.append(rv_t)
            tau_mu_var_list.append((rv_t + ep).clone())
            tau_eta_var_list.append(torch.ones(c_l, device=device, dtype=torch.float32))
            act_t = torch.sqrt(rv_t + ep) > 1e-3
            if not bool(act_t.any().item()):
                act_t = torch.ones(c_l, device=device, dtype=torch.bool)
            active_list.append(act_t)
            q_med_list.append(0.5)
            q90_list.append(2.0)

        return cls(
            k_fg=k_fg,
            psi_scaled=np.zeros((2, k_fg), dtype=np.float64),
            psi_scale=np.ones(k_fg, dtype=np.float64),
            m_neighbors=1,
            d_sq=1.0,
            h_bw=1.0,
            source_empty_frac=0.0,
            mu_bank_cat=torch.zeros((2, offset), device=device, dtype=torch.float32),
            var_bank_cat=torch.ones((2, offset), device=device, dtype=torch.float32),
            uncond_mu_list=uncond_mu_list,
            uncond_var_list=uncond_var_list,
            layer_slices=layer_slices,
            tau_mu_var_list=tau_mu_var_list,
            tau_eta_var_list=tau_eta_var_list,
            active_list=active_list,
            q_med_list=q_med_list,
            q90_list=q90_list,
        )

    def query(
        self,
        pi_vec: np.ndarray,
        use_prev_match: bool,
        use_supp_gate: bool,
        gamma_supp: float,
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], float, float]:
        """Query prevalence-matched source deep moments `(mu_hat_list, var_hat_list, w_supp, mean_d_m)`."""
        psi_t = np.sqrt(np.clip(pi_vec, 0.0, 1.0)) / self.psi_scale
        diff = self.psi_scaled - psi_t[None, :]
        dists = np.sqrt(np.mean(diff * diff, axis=1))

        m = self.m_neighbors
        top_idx = np.argpartition(dists, kth=m - 1)[:m]
        top_d = dists[top_idx]
        mean_d = float(np.mean(top_d))

        if use_supp_gate:
            excess = max(0.0, (mean_d / self.d_sq) ** 2 - 1.0)
            w_supp = float(np.exp(-gamma_supp * excess))
        else:
            w_supp = 1.0

        if not use_prev_match:
            return self.uncond_mu_list, self.uncond_var_list, w_supp, mean_d

        lw = -0.5 * ((top_d / self.h_bw) ** 2)
        lw -= float(np.max(lw))
        w_np = np.exp(lw)
        w_np /= float(np.sum(w_np))

        idx_t = torch.from_numpy(top_idx.astype(np.int64)).to(device)
        w_t = torch.from_numpy(w_np.astype(np.float32)).to(device)

        mu_cat = w_t @ self.mu_bank_cat.index_select(0, idx_t)
        var_cat = w_t @ self.var_bank_cat.index_select(0, idx_t)

        mu_hat_list = [mu_cat[s:e] for (s, e) in self.layer_slices]
        var_hat_list = [var_cat[s:e] for (s, e) in self.layer_slices]
        return mu_hat_list, var_hat_list, w_supp, mean_d

    def to_state_dict(self) -> Dict[str, object]:
        return {
            "k_fg": int(self.k_fg),
            "psi_scaled": self.psi_scaled.copy(),
            "psi_scale": self.psi_scale.copy(),
            "m_neighbors": int(self.m_neighbors),
            "d_sq": float(self.d_sq),
            "h_bw": float(self.h_bw),
            "source_empty_frac": float(self.source_empty_frac),
            "mu_bank_cat": self.mu_bank_cat.detach().cpu(),
            "var_bank_cat": self.var_bank_cat.detach().cpu(),
            "uncond_mu_list": [t.detach().cpu() for t in self.uncond_mu_list],
            "uncond_var_list": [t.detach().cpu() for t in self.uncond_var_list],
            "layer_slices": list(self.layer_slices),
            "tau_mu_var_list": [t.detach().cpu() for t in self.tau_mu_var_list],
            "tau_eta_var_list": [t.detach().cpu() for t in self.tau_eta_var_list],
            "active_list": [t.detach().cpu() for t in self.active_list],
            "q_med_list": [float(x) for x in self.q_med_list],
            "q90_list": [float(x) for x in self.q90_list],
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, object], device: torch.device) -> "DeepSourceBank":
        return cls(
            k_fg=int(state["k_fg"]),
            psi_scaled=np.asarray(state["psi_scaled"], dtype=np.float64),
            psi_scale=np.asarray(state["psi_scale"], dtype=np.float64),
            m_neighbors=int(state["m_neighbors"]),
            d_sq=float(state["d_sq"]),
            h_bw=float(state["h_bw"]),
            source_empty_frac=float(state["source_empty_frac"]),
            mu_bank_cat=state["mu_bank_cat"].to(device),
            var_bank_cat=state["var_bank_cat"].to(device),
            uncond_mu_list=[t.to(device) for t in state["uncond_mu_list"]],
            uncond_var_list=[t.to(device) for t in state["uncond_var_list"]],
            layer_slices=[tuple(s) for s in state["layer_slices"]],
            tau_mu_var_list=[t.to(device) for t in state["tau_mu_var_list"]],
            tau_eta_var_list=[t.to(device) for t in state["tau_eta_var_list"]],
            active_list=[t.to(device) for t in state["active_list"]],
            q_med_list=[float(x) for x in state["q_med_list"]],
            q90_list=[float(x) for x in state["q90_list"]],
        )


class DeepResidualStreamState:
    """Running stream EMA of prevalence-deconfounded deep residuals on device."""

    def __init__(self, layer_dims: Sequence[int], device: torch.device, ema_min: float = 0.15) -> None:
        self.layer_dims = [int(c) for c in layer_dims]
        self.device = device
        self.ema_min = float(ema_min)
        self.reset()

    def reset(self) -> None:
        self.n_eff: float = 0.0
        self.n_pres: int = 0
        self.bar_d_mu: List[torch.Tensor] = [
            torch.zeros(c, device=self.device, dtype=torch.float32) for c in self.layer_dims
        ]
        self.bar_d_eta: List[torch.Tensor] = [
            torch.zeros(c, device=self.device, dtype=torch.float32) for c in self.layer_dims
        ]
        self.enc_raw_d_mu: List[torch.Tensor] = [
            torch.zeros(c, device=self.device, dtype=torch.float32) for c in self.layer_dims
        ]
        self.enc_raw_d_eta: List[torch.Tensor] = [
            torch.zeros(c, device=self.device, dtype=torch.float32) for c in self.layer_dims
        ]
        self.n_raw_pres: int = 0
