#!/usr/bin/env python3
"""Benchmark stream-order robustness, continual un-reset streams, and isolated GPU latency/VRAM."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List

import numpy as np
import torch

_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from data import (
    DEFAULT_PROCESSED_ROOT,
    PROSTATE_SOURCE_SITE,
    PROSTATE_TARGET_SITES,
    load_prostate_manifest_by_site,
)
from evaluate.metrics import binary_dice_pct_from_counts, summarize_prostate
from evaluate.run_benchmark import DEFAULT_CKPT_DIR, set_seed
from models import load_model_checkpoint
from tta import (
    CoTTAAdapter,
    PDDualConfig,
    PTBNAdapter,
    PrevalenceDeconfoundedDualTTA,
    SARAdapter,
    SourceOnlyAdapter,
    TENTAdapter,
)

SEEDS = (42, 123, 2024, 7, 2025)


def run_prostate_continual_and_shuffle(ckpt_dir: Path, processed_root: Path, device: torch.device) -> Dict[str, object]:
    set_seed(42)
    model = load_model_checkpoint("vptta_resunet", 1, 2, ckpt_dir / "source_prostate_runmc_seed42.pth", device)
    p_root = processed_root / "prostate"
    by_site = load_prostate_manifest_by_site(processed_root)
    src_imgs = [torch.from_numpy(np.load(p_root / r["image"]).astype(np.float32)) for r in by_site[PROSTATE_SOURCE_SITE]]
    site_targets = {
        site: [
            (r, torch.from_numpy(np.load(p_root / r["image"]).astype(np.float32)), np.load(p_root / r["mask"]).astype(np.int64) > 0)
            for r in by_site[site]
        ]
        for site in PROSTATE_TARGET_SITES
    }

    adapter = PrevalenceDeconfoundedDualTTA.fit_from_source(model, src_imgs, device, "multiclass_softmax", PDDualConfig())

    adapter.reset_stream()
    cont_rows: List[Dict[str, object]] = []
    idx = 0
    for site in PROSTATE_TARGET_SITES:
        for r, im, gt_b in site_targets[site]:
            logits, dg = adapter.predict_single(im, device=device)
            pb = logits.argmax(dim=1)[0].detach().cpu().numpy() > 0
            pp, gp = int(pb.sum()), int(gt_b.sum())
            tp = int(np.logical_and(pb, gt_b).sum()) if (pp > 0 and gp > 0) else 0
            cont_rows.append({
                "call_idx": idx, "domain": site, "volume_id": r["case_id"], "z_idx": int(r["z"]),
                "image": r["image"], "slice_dice": binary_dice_pct_from_counts(tp, pp, gp),
                "tp_px": tp, "pred_px": pp, "gt_px": gp, "has_fg": int(gp > 0),
                "fp_empty_slice": (int(pp > 0) if gp == 0 else -1),
                "resets": int(dg.get("resets", 0.0)),
            })
            idx += 1
    cont_summary = summarize_prostate(cont_rows, PROSTATE_TARGET_SITES)

    shuffle_scores = []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        rows_s: List[Dict[str, object]] = []
        for site in PROSTATE_TARGET_SITES:
            adapter.reset_stream()
            perm = rng.permutation(len(site_targets[site]))
            for pidx in perm:
                r, im, gt_b = site_targets[site][int(pidx)]
                logits, _ = adapter.predict_single(im, device=device)
                pb = logits.argmax(dim=1)[0].detach().cpu().numpy() > 0
                pp, gp = int(pb.sum()), int(gt_b.sum())
                tp = int(np.logical_and(pb, gt_b).sum()) if (pp > 0 and gp > 0) else 0
                rows_s.append({
                    "domain": site, "volume_id": r["case_id"],
                    "slice_dice": binary_dice_pct_from_counts(tp, pp, gp),
                    "tp_px": tp, "pred_px": pp, "gt_px": gp,
                })
        shuffle_scores.append(float(summarize_prostate(rows_s, PROSTATE_TARGET_SITES)["mean_5center_3d_dice"]))

    return {
        "continual_unreset": cont_summary,
        "slice_shuffle_5seeds": {
            "seeds": list(SEEDS),
            "per_seed_3d_dice": shuffle_scores,
            "mean": round(float(np.mean(shuffle_scores)), 4),
            "std": round(float(np.std(shuffle_scores, ddof=1)), 4),
        },
    }


def run_latency_vram_benchmark(ckpt_dir: Path, processed_root: Path, device: torch.device, n_warmup: int = 10, n_timed: int = 100) -> Dict[str, object]:
    set_seed(42)
    model = load_model_checkpoint("vptta_resunet", 1, 2, ckpt_dir / "source_prostate_runmc_seed42.pth", device)
    p_root = processed_root / "prostate"
    by_site = load_prostate_manifest_by_site(processed_root)
    src_imgs = [torch.from_numpy(np.load(p_root / r["image"]).astype(np.float32)) for r in by_site[PROSTATE_SOURCE_SITE]]
    bmc_imgs = [torch.from_numpy(np.load(p_root / r["image"]).astype(np.float32)) for r in by_site["BMC"][: (n_warmup + n_timed)]]

    pd_adapter = PrevalenceDeconfoundedDualTTA.fit_from_source(model, src_imgs, device, "multiclass_softmax", PDDualConfig())
    methods = {
        "SourceOnly": SourceOnlyAdapter(model, device),
        "PTBN": PTBNAdapter(model, device),
        "TENT": TENTAdapter(model, device),
        "SAR": SARAdapter(model, device, num_classes=2),
        "CoTTA": CoTTAAdapter(model, device),
        "DeCoTTA": pd_adapter,
    }

    report: Dict[str, object] = {}
    for name, ad in methods.items():
        ad.reset_stream()
        for im in bmc_imgs[:n_warmup]:
            ad.predict_single(im, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        times_ms = []
        for im in bmc_imgs[n_warmup : n_warmup + n_timed]:
            t0 = time.perf_counter()
            ad.predict_single(im, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        vram_mib = (torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == "cuda" else 0.0
        report[name] = {
            "ms_mean": round(float(np.mean(times_ms)), 2),
            "ms_std": round(float(np.std(times_ms, ddof=1)), 2),
            "fps": round(1000.0 / float(np.mean(times_ms)), 1),
            "peak_vram_mib": round(float(vram_mib), 1),
        }
    return report


def main() -> None:
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["latency", "robustness", "both"], default="latency")
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument("--ckpt_dir", type=Path, default=DEFAULT_CKPT_DIR)
    ap.add_argument("--processed_root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    payload: Dict[str, object] = {}
    if args.mode in ("latency", "both"):
        payload["latency_vram"] = run_latency_vram_benchmark(args.ckpt_dir, args.processed_root, dev)
    if args.mode in ("robustness", "both"):
        payload["prostate_robustness"] = run_prostate_continual_and_shuffle(args.ckpt_dir, args.processed_root, dev)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, args.out_json)
    print(f"Saved -> {args.out_json}")


if __name__ == "__main__":
    main()
