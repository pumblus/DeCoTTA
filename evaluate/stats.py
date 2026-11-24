"""Paired statistical hypothesis testing and stratified bootstrap confidence intervals."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon


def paired_wilcoxon_p(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided paired Wilcoxon signed-rank p-value."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    diff = arr_a - arr_b
    if np.all(np.abs(diff) < 1e-12):
        return 1.0
    stat = wilcoxon(arr_a, arr_b, zero_method="wilcox", alternative="two-sided")
    return float(stat.pvalue)


def holm_bonferroni(p_values: Dict[str, float]) -> Dict[str, float]:
    """Holm step-down family-wise error correction over `{name: raw_p}`."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: Dict[str, float] = {}
    running_max = 0.0
    for rank_idx, (name, raw_p) in enumerate(items):
        adj = min(1.0, float(raw_p) * float(m - rank_idx))
        running_max = max(running_max, adj)
        adjusted[name] = running_max
    return adjusted


def stratified_bootstrap_ci(
    domain_scores: Dict[str, Sequence[float]],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute `(point_mean, ci_low, ci_high)` resampling units within each target domain."""
    rng = np.random.default_rng(seed)
    dom_arrays = [np.asarray(vals, dtype=np.float64) for vals in domain_scores.values() if len(vals) > 0]
    if not dom_arrays:
        return 0.0, 0.0, 0.0
    point = float(np.mean([float(np.mean(arr)) for arr in dom_arrays]))
    boot_means = np.zeros(n_boot, dtype=np.float64)
    for arr in dom_arrays:
        n = len(arr)
        idx = rng.integers(0, n, size=(n_boot, n))
        boot_means += arr[idx].mean(axis=1)
    boot_means /= float(len(dom_arrays))
    lo = float(np.quantile(boot_means, 0.5 * alpha))
    hi = float(np.quantile(boot_means, 1.0 - 0.5 * alpha))
    return point, lo, hi
