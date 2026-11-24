"""Segmentation accuracy, exact Euclidean surface distance, and safety metrics."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import math
import os
from pathlib import Path
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

_STRUCT_2D = generate_binary_structure(2, 1)
_STRUCT_3D = generate_binary_structure(3, 1)

PROSTATE_COLS = [
    "call_idx", "domain", "volume_id", "z_idx", "image",
    "slice_dice", "tp_px", "pred_px", "gt_px", "has_fg", "fp_empty_slice",
    "lam_stream_max", "lam_indiv_max", "resets",
    "w_supp", "w_shift_mean", "lam_deep_mean", "rescued",
]

CARDIAC_COLS = [
    "call_idx", "domain", "domain_idx", "image",
    "lv_dice", "myo_dice", "rv_dice", "mean_dice",
    "gt_lv_px", "gt_myo_px", "gt_rv_px", "gt_fg_px", "gt_fg_frac",
    "pred_lv_px", "pred_myo_px", "pred_rv_px", "pred_fg_px", "pred_fg_frac",
    "has_lv", "has_myo", "has_rv", "fp_lv_absent", "fp_myo_absent", "fp_rv_absent",
]

FUNDUS_COLS = [
    "call_idx", "domain", "domain_idx", "image",
    "ring_dice", "cup_dice", "whole_disc_dice",
    "mean_ring_cup_dice", "mean_whole_disc_cup_dice",
    "gt_disc_px", "pred_disc_px", "gt_cup_px", "pred_cup_px",
    "gt_fg_frac", "pred_fg_frac",
]

OPTIC_COLS = [
    "call_idx", "image", "disc_dice", "cup_dice", "mean_dice",
    "lam_stream_max", "lam_indiv_max", "resets",
    "w_supp", "w_shift_mean", "lam_deep_mean",
]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv_atomic(path: Path, cols: Sequence[str], rows: Sequence[Dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols))
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.6f}" if isinstance(r[c], float) else r[c]) for c in cols})
    os.replace(tmp, path)
    return file_sha256(path)


def binary_dice_pct_from_counts(tp: int, pred_px: int, gt_px: int) -> float:
    if pred_px == 0 and gt_px == 0:
        return 100.0
    if pred_px == 0 or gt_px == 0:
        return 0.0
    return 100.0 * (2.0 * tp) / float(pred_px + gt_px)


def binary_dice_pct(pred: np.ndarray, gt: np.ndarray) -> float:
    p = int(np.count_nonzero(pred))
    g = int(np.count_nonzero(gt))
    if p == 0 and g == 0:
        return 100.0
    if p == 0 or g == 0:
        return 0.0
    return 100.0 * (2.0 * int(np.count_nonzero(pred & gt))) / float(p + g)


def compute_hd95_assd(
    pred: np.ndarray,
    gt: np.ndarray,
    use_bbox: bool = True,
) -> Tuple[float, float, bool, bool]:
    """Compute exact 1-connected Euclidean HD95 and ASSD in pixels/voxels.

    Returns `(hd95, assd, empty_pred_failure, gt_is_empty)`. When exactly one mask
    is empty, assigns the image/volume Euclidean diagonal penalty.
    """
    pred_b = np.asarray(pred, dtype=np.bool_)
    gt_b = np.asarray(gt, dtype=np.bool_)
    if pred_b.shape != gt_b.shape:
        raise ValueError(f"Shape mismatch: {pred_b.shape} vs {gt_b.shape}")

    p_any, g_any = bool(pred_b.any()), bool(gt_b.any())
    if not p_any and not g_any:
        return 0.0, 0.0, False, True
    if not p_any or not g_any:
        diag = math.sqrt(sum(float(s) * float(s) for s in pred_b.shape))
        return diag, diag, (not p_any and g_any), (not g_any)

    ndim = pred_b.ndim
    if ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D mask, got ndim={ndim}")
    struct = _STRUCT_2D if ndim == 2 else _STRUCT_3D

    if use_bbox:
        union = pred_b | gt_b
        slices = []
        for axis in range(ndim):
            other = tuple(i for i in range(ndim) if i != axis)
            nz = np.flatnonzero(union.any(axis=other))
            slices.append(slice(max(0, int(nz[0]) - 2), min(pred_b.shape[axis], int(nz[-1]) + 3)))
        p_sub, g_sub = pred_b[tuple(slices)], gt_b[tuple(slices)]
    else:
        p_sub, g_sub = pred_b, gt_b

    s_pred = p_sub ^ binary_erosion(p_sub, structure=struct, border_value=0)
    s_gt = g_sub ^ binary_erosion(g_sub, structure=struct, border_value=0)

    d_p2g = distance_transform_edt(~s_gt)[s_pred]
    d_g2p = distance_transform_edt(~s_pred)[s_gt]

    hd95 = float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)))
    assd = float((d_p2g.sum() + d_g2p.sum()) / float(d_p2g.size + d_g2p.size))
    return hd95, assd, False, False


def cardiac_slice_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, object]:
    total_px = float(gt.shape[0] * gt.shape[1])
    out: Dict[str, object] = {}
    dices = []
    gt_fg, pred_fg = 0, 0
    for cid, cname in ((1, "lv"), (2, "myo"), (3, "rv")):
        pb, gb = (pred == cid), (gt == cid)
        p_px, g_px = int(pb.sum()), int(gb.sum())
        d = binary_dice_pct(pb, gb)
        dices.append(d)
        gt_fg += g_px
        pred_fg += p_px
        out[f"{cname}_dice"] = d
        out[f"gt_{cname}_px"] = g_px
        out[f"pred_{cname}_px"] = p_px
        out[f"has_{cname}"] = 1 if g_px > 0 else 0
        out[f"fp_{cname}_absent"] = (1 if p_px > 0 else 0) if g_px == 0 else -1
    out["mean_dice"] = sum(dices) / 3.0
    out["gt_fg_px"] = gt_fg
    out["pred_fg_px"] = pred_fg
    out["gt_fg_frac"] = gt_fg / total_px
    out["pred_fg_frac"] = pred_fg / total_px
    return out


def fundus_slice_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, object]:
    total_px = float(gt.shape[0] * gt.shape[1])
    ring_dice = binary_dice_pct(pred == 1, gt == 1)
    cup_dice = binary_dice_pct(pred == 2, gt == 2)
    whole_disc_dice = binary_dice_pct(pred >= 1, gt >= 1)
    gt_disc_px = int((gt >= 1).sum())
    pred_disc_px = int((pred >= 1).sum())
    return {
        "ring_dice": ring_dice,
        "cup_dice": cup_dice,
        "whole_disc_dice": whole_disc_dice,
        "mean_ring_cup_dice": 0.5 * (ring_dice + cup_dice),
        "mean_whole_disc_cup_dice": 0.5 * (whole_disc_dice + cup_dice),
        "gt_disc_px": gt_disc_px,
        "pred_disc_px": pred_disc_px,
        "gt_cup_px": int((gt == 2).sum()),
        "pred_cup_px": int((pred == 2).sum()),
        "gt_fg_frac": gt_disc_px / total_px,
        "pred_fg_frac": pred_disc_px / total_px,
    }


def optic_dice_pct(pred_bin_2hw: np.ndarray, gt_disc: np.ndarray, gt_cup: np.ndarray) -> Tuple[float, float]:
    """Official OPTIC smooth-1e-6 Whole-Disc and Optic-Cup Dice in `[0, 100]`."""
    smooth = 1e-6
    pd_m, pc_m = pred_bin_2hw[0], pred_bin_2hw[1]
    dd = (2.0 * float(np.logical_and(pd_m, gt_disc).sum()) + smooth) / (float(pd_m.sum()) + float(gt_disc.sum()) + smooth) * 100.0
    cd = (2.0 * float(np.logical_and(pc_m, gt_cup).sum()) + smooth) / (float(pc_m.sum()) + float(gt_cup.sum()) + smooth) * 100.0
    return dd, cd


def summarize_prostate(rows: Sequence[Dict[str, object]], target_sites: Sequence[str]) -> Dict[str, object]:
    per_center: Dict[str, Dict[str, object]] = {}
    all_empty_gt0, all_empty_ge10, all_empty_n = 0, 0, 0

    for site in target_sites:
        s_rows = [r for r in rows if r["domain"] == site]
        vol_agg: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
        present_dices: List[float] = []
        empty_n, empty_gt0, empty_ge10 = 0, 0, 0

        for r in s_rows:
            vid = str(r["volume_id"])
            tp, pp, gp = int(r["tp_px"]), int(r["pred_px"]), int(r["gt_px"])
            vol_agg[vid][0] += tp
            vol_agg[vid][1] += pp
            vol_agg[vid][2] += gp
            if gp > 0:
                present_dices.append(float(r["slice_dice"]))
            else:
                empty_n += 1
                if pp > 0:
                    empty_gt0 += 1
                if pp >= 10:
                    empty_ge10 += 1

        vol_dices = [binary_dice_pct_from_counts(v[0], v[1], v[2]) for v in vol_agg.values()]
        all_empty_n += empty_n
        all_empty_gt0 += empty_gt0
        all_empty_ge10 += empty_ge10

        per_center[site] = {
            "num_volumes": len(vol_dices),
            "num_slices": len(s_rows),
            "num_present_slices": len(present_dices),
            "num_empty_slices": empty_n,
            "volume_3d_dice": round(float(np.mean(vol_dices)), 6),
            "slice_2d_present_only_dice": round(float(np.mean(present_dices)) if present_dices else 0.0, 6),
            "empty_slice_fpr_gt0_pct": round(100.0 * empty_gt0 / float(max(1, empty_n)), 4),
            "empty_slice_fpr_ge10_pct": round(100.0 * empty_ge10 / float(max(1, empty_n)), 4),
        }

    mean_3d = float(np.mean([float(per_center[s]["volume_3d_dice"]) for s in target_sites]))
    mean_2d_pres = float(np.mean([float(per_center[s]["slice_2d_present_only_dice"]) for s in target_sites]))
    return {
        "mean_5center_3d_dice": round(mean_3d, 6),
        "mean_5center_2d_present_only_dice": round(mean_2d_pres, 6),
        "total_empty_slices": all_empty_n,
        "overall_empty_slice_fpr_gt0_pct": round(100.0 * all_empty_gt0 / float(max(1, all_empty_n)), 4),
        "overall_empty_slice_fpr_ge10_pct": round(100.0 * all_empty_ge10 / float(max(1, all_empty_n)), 4),
        "per_center": per_center,
    }


def _parse_cardiac_stem(name: str) -> Tuple[str, str]:
    b = name.replace(".nii.gz", "").replace(".npy", "")
    m = re.match(r"^(.+?)_(ED|ES|\d+)_z(\d+)$", b)
    if not m:
        raise ValueError(f"Cannot parse cardiac slice identifier: {name}")
    return m.group(1), m.group(2)


def summarize_cardiac(rows: Sequence[Dict[str, object]], domains: Sequence[str]) -> Dict[str, object]:
    vol_tp: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(lambda: {"lv": 0.0, "myo": 0.0, "rv": 0.0})
    vol_den: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(lambda: {"lv": 0.0, "myo": 0.0, "rv": 0.0})

    for r in rows:
        dom = str(r["domain"])
        pid, ph = _parse_cardiac_stem(str(r["image"]))
        vk = (dom, pid, ph)
        for c in ("lv", "myo", "rv"):
            p_px, g_px = int(r[f"pred_{c}_px"]), int(r[f"gt_{c}_px"])
            d_val = float(r[f"{c}_dice"])
            tp = round(d_val / 100.0 * (p_px + g_px) / 2.0) if (p_px > 0 and g_px > 0) else 0.0
            vol_tp[vk][c] += tp
            vol_den[vk][c] += float(p_px + g_px)

    pat_scores_by_dom: Dict[str, Dict[str, List[float]]] = {d: defaultdict(list) for d in domains}
    for (dom, pid, _ph), tps in vol_tp.items():
        dens = vol_den[(dom, pid, _ph)]
        c_scores = [100.0 if dens[c] == 0 else (200.0 * tps[c] / dens[c]) for c in ("lv", "myo", "rv")]
        pat_scores_by_dom[dom][pid].append(sum(c_scores) / 3.0)

    dom_3d, dom_2d_pres, dom_2d_all = {}, {}, {}
    for d in domains:
        p_means = [float(np.mean(vs)) for vs in pat_scores_by_dom[d].values()]
        dom_3d[d] = round(float(np.mean(p_means)), 4)
        sub = [r for r in rows if r["domain"] == d]
        dom_2d_all[d] = round(float(np.mean([float(r["mean_dice"]) for r in sub])), 4)
        pres_vals = []
        for r in sub:
            present_cs = [float(r[f"{c}_dice"]) for c in ("lv", "myo", "rv") if int(r[f"gt_{c}_px"]) > 0]
            if present_cs:
                pres_vals.append(float(np.mean(present_cs)))
        dom_2d_pres[d] = round(float(np.mean(pres_vals)), 4)

    absent_rv = [r for r in rows if int(r["gt_rv_px"]) == 0]
    n_abs = max(1, len(absent_rv))
    fpr_gt0 = 100.0 * sum(1 for r in absent_rv if int(r["pred_rv_px"]) > 0) / float(n_abs)
    fpr_ge10 = 100.0 * sum(1 for r in absent_rv if int(r["pred_rv_px"]) >= 10) / float(n_abs)
    return {
        "n_rows": len(rows),
        "patient_3d_dom_eq": round(float(np.mean([dom_3d[d] for d in domains])), 4),
        "per_domain_3d": dom_3d,
        "slice_2d_present_only_dom_eq": round(float(np.mean([dom_2d_pres[d] for d in domains])), 4),
        "slice_2d_all_dom_eq": round(float(np.mean([dom_2d_all[d] for d in domains])), 4),
        "slice_2d_pooled": round(float(np.mean([float(r["mean_dice"]) for r in rows])), 4),
        "per_domain_2d_all": dom_2d_all,
        "absent_rv_n": len(absent_rv),
        "absent_rv_fpr_gt0_pct": round(fpr_gt0, 2),
        "absent_rv_fpr_ge10_pct": round(fpr_ge10, 2),
    }


def summarize_fundus(rows: Sequence[Dict[str, object]], sites: Sequence[str]) -> Dict[str, object]:
    per_dom = {}
    for s in sites:
        sub = [r for r in rows if r["domain"] == s]
        gt_d = float(sum(int(r["gt_disc_px"]) for r in sub))
        pr_d = float(sum(int(r["pred_disc_px"]) for r in sub))
        per_dom[s] = {
            "n": len(sub),
            "whole_disc": round(float(np.mean([float(r["whole_disc_dice"]) for r in sub])), 4),
            "cup": round(float(np.mean([float(r["cup_dice"]) for r in sub])), 4),
            "wd_cup_mean": round(float(np.mean([float(r["mean_whole_disc_cup_dice"]) for r in sub])), 4),
            "disc_area_ratio": round(pr_d / max(1.0, gt_d), 4),
        }
    return {
        "n_rows": len(rows),
        "dom_eq_wd_cup": round(float(np.mean([per_dom[s]["wd_cup_mean"] for s in sites])), 4),
        "per_domain": per_dom,
    }
