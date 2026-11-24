"""Unified CLI evaluation runner for `PrevalenceDeconfoundedDualTTA` across all 5 benchmarks."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from data import (
    DEFAULT_PROCESSED_ROOT,
    DEFAULT_RAW_ROOT,
    OPTIC_DOMAINS,
    PROSTATE_SOURCE_SITE,
    PROSTATE_TARGET_SITES,
    build_dataset,
    build_transform,
    load_mms2d_paper_splits,
    load_optic_domain,
    load_prostate_manifest_by_site,
)
from evaluate.metrics import (
    CARDIAC_COLS,
    FUNDUS_COLS,
    OPTIC_COLS,
    PROSTATE_COLS,
    binary_dice_pct_from_counts,
    cardiac_slice_metrics,
    file_sha256,
    fundus_slice_metrics,
    optic_dice_pct,
    summarize_cardiac,
    summarize_fundus,
    summarize_prostate,
    write_csv_atomic,
)
from models import load_model_checkpoint
from tta import PDDualConfig, PrevalenceDeconfoundedDualTTA, model_state_sha256, resolve_preset

DEFAULT_CKPT_DIR = Path(os.environ.get("PD_TTA_CKPT_DIR", str(_CODE_ROOT / "checkpoints")))


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_preset_adapter(
    preset: str,
    cfg: PDDualConfig,
    base_adapter: PrevalenceDeconfoundedDualTTA,
    model: torch.nn.Module,
    input_shape_chw: Tuple[int, int, int],
    output_mode: str,
    device: torch.device,
) -> PrevalenceDeconfoundedDualTTA:
    if preset in ("weight_only_sf", "abl_weight_only_sf"):
        return PrevalenceDeconfoundedDualTTA.from_checkpoint_only(
            model=model, input_shape_chw=input_shape_chw, device=device, output_mode=output_mode, config=cfg
        )
    return base_adapter.with_config(cfg)


def evaluate_prostate(
    presets: Sequence[str],
    out_dir: Path,
    device: torch.device,
    ckpt_dir: Path = DEFAULT_CKPT_DIR,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
) -> Dict[str, object]:
    set_seed(42)
    ckpt = ckpt_dir / "source_prostate_runmc_seed42.pth"
    model = load_model_checkpoint("vptta_resunet", 1, 2, ckpt, device)
    sha_ref = model_state_sha256(model)

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

    base = PrevalenceDeconfoundedDualTTA.fit_from_source(model, src_imgs, device, "multiclass_softmax", PDDualConfig())
    assert model_state_sha256(model) == sha_ref

    results: Dict[str, object] = {}
    for pname in presets:
        cfg = resolve_preset(pname)
        t0 = time.perf_counter()
        adapter = _build_preset_adapter(pname, cfg, base, model, (1, 384, 384), "multiclass_softmax", device)
        rows: List[Dict[str, object]] = []
        call_idx = 0
        for site in PROSTATE_TARGET_SITES:
            adapter.reset_stream()
            for r, img_t, gt_b in site_targets[site]:
                logits, dg = adapter.predict_single(img_t, device=device)
                pb = logits.argmax(dim=1)[0].detach().cpu().numpy() > 0
                pp, gp = int(pb.sum()), int(gt_b.sum())
                tp = int(np.logical_and(pb, gt_b).sum()) if (pp > 0 and gp > 0) else 0
                rows.append({
                    "call_idx": call_idx,
                    "domain": site,
                    "volume_id": r["case_id"],
                    "z_idx": int(r["z"]),
                    "image": r["image"],
                    "slice_dice": binary_dice_pct_from_counts(tp, pp, gp),
                    "tp_px": tp,
                    "pred_px": pp,
                    "gt_px": gp,
                    "has_fg": 1 if gp > 0 else 0,
                    "fp_empty_slice": (1 if pp > 0 else 0) if gp == 0 else -1,
                    "lam_stream_max": float(dg.get("lam_stream_max", 0.0)),
                    "lam_indiv_max": float(dg.get("lam_indiv_max", 0.0)),
                    "resets": int(dg.get("resets", 0.0)),
                    "w_supp": float(dg.get("w_supp", 0.0)),
                    "w_shift_mean": float(dg.get("w_shift_mean", 0.0)),
                    "lam_deep_mean": float(dg.get("lam_deep_mean", 0.0)),
                    "rescued": int(dg.get("rescued", 0.0)),
                })
                call_idx += 1
            assert model_state_sha256(model) == sha_ref

        csv_path = out_dir / "csv" / "prostate" / f"per_slice_{pname}.csv"
        c_sha = write_csv_atomic(csv_path, PROSTATE_COLS, rows)
        summ = summarize_prostate(rows, PROSTATE_TARGET_SITES)
        summ.update({"config": asdict(cfg), "csv_path": str(csv_path), "csv_sha256": c_sha, "elapsed_s": round(time.perf_counter() - t0, 2)})
        results[pname] = summ
        print(f"[Prostate | {pname:20s}] 3D={summ['mean_5center_3d_dice']:6.2f}% | Empty-FPR={summ['overall_empty_slice_fpr_gt0_pct']:5.2f}% ({summ['elapsed_s']:.1f}s)")
    return {"suite": "prostate", "model_state_sha256": sha_ref, "results": results}


def evaluate_rigaplus(
    presets: Sequence[str],
    out_dir: Path,
    device: torch.device,
    ckpt_dir: Path = DEFAULT_CKPT_DIR,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
) -> Dict[str, object]:
    set_seed(42)
    ckpt = ckpt_dir / "source_rigaplus_resunet_seed42.pth"
    model = load_model_checkpoint("resunet", 3, 3, ckpt, device)
    sha_ref = model_state_sha256(model)

    ds_tr = build_dataset("rigaplus", "source_train", "Magrabia", transform=build_transform("rigaplus", "tta"), seed=42, processed_root=processed_root)
    src_imgs = [ds_tr[i][0].float() for i in range(len(ds_tr))]
    sites = ("BinRushed", "MESSIDOR")
    site_targets = {
        s: [(ds[i][0].float(), ds[i][1].numpy().astype(np.int64), os.path.basename(ds.items[i].image_path))
            for ds in (build_dataset("rigaplus", "target", s, seed=42, processed_root=processed_root),)
            for i in range(len(ds))]
        for s in sites
    }

    base = PrevalenceDeconfoundedDualTTA.fit_from_source(model, src_imgs, device, "multiclass_softmax", PDDualConfig())
    assert model_state_sha256(model) == sha_ref

    results: Dict[str, object] = {}
    for pname in presets:
        cfg = resolve_preset(pname)
        t0 = time.perf_counter()
        adapter = _build_preset_adapter(pname, cfg, base, model, (3, 512, 512), "multiclass_softmax", device)
        rows: List[Dict[str, object]] = []
        call_idx = 0
        for s in sites:
            adapter.reset_stream()
            for d_idx, (img_t, gt_np, nm) in enumerate(site_targets[s]):
                logits, _ = adapter.predict_single(img_t, device=device)
                pred_np = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)
                met = fundus_slice_metrics(pred_np, gt_np)
                met.update({"call_idx": call_idx, "domain": s, "domain_idx": d_idx, "image": nm})
                rows.append(met)
                call_idx += 1
            assert model_state_sha256(model) == sha_ref

        csv_path = out_dir / "csv" / "rigaplus" / f"per_image_{pname}.csv"
        c_sha = write_csv_atomic(csv_path, FUNDUS_COLS, rows)
        summ = summarize_fundus(rows, sites)
        summ.update({"config": asdict(cfg), "csv_path": str(csv_path), "csv_sha256": c_sha, "elapsed_s": round(time.perf_counter() - t0, 2)})
        results[pname] = summ
        print(f"[RIGA+    | {pname:20s}] Dice={summ['dom_eq_wd_cup']:6.2f}% ({summ['elapsed_s']:.1f}s)")
    return {"suite": "rigaplus", "model_state_sha256": sha_ref, "results": results}


def _evaluate_cardiac_common(
    suite_name: str,
    model: torch.nn.Module,
    src_imgs: List[torch.Tensor],
    targets: Dict[str, List[Tuple[torch.Tensor, np.ndarray, str]]],
    domains: Tuple[str, ...],
    input_shape_chw: Tuple[int, int, int],
    presets: Sequence[str],
    out_dir: Path,
    device: torch.device,
) -> Dict[str, object]:
    sha_ref = model_state_sha256(model)
    base = PrevalenceDeconfoundedDualTTA.fit_from_source(model, src_imgs, device, "multiclass_softmax", PDDualConfig())
    assert model_state_sha256(model) == sha_ref

    results: Dict[str, object] = {}
    for pname in presets:
        cfg = resolve_preset(pname)
        t0 = time.perf_counter()
        adapter = _build_preset_adapter(pname, cfg, base, model, input_shape_chw, "multiclass_softmax", device)
        rows: List[Dict[str, object]] = []
        call_idx = 0
        for d in domains:
            adapter.reset_stream()
            for d_idx, (img_t, gt_np, nm) in enumerate(targets[d]):
                logits, _ = adapter.predict_single(img_t, device=device)
                pred_np = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)
                met = cardiac_slice_metrics(pred_np, gt_np)
                met.update({"call_idx": call_idx, "domain": d, "domain_idx": d_idx, "image": nm})
                rows.append(met)
                call_idx += 1
            assert model_state_sha256(model) == sha_ref

        csv_path = out_dir / "csv" / suite_name / f"per_slice_{pname}.csv"
        c_sha = write_csv_atomic(csv_path, CARDIAC_COLS, rows)
        summ = summarize_cardiac(rows, domains)
        summ.update({"config": asdict(cfg), "csv_path": str(csv_path), "csv_sha256": c_sha, "elapsed_s": round(time.perf_counter() - t0, 2)})
        results[pname] = summ
        print(f"[{suite_name:12s} | {pname:20s}] 3D={summ['patient_3d_dom_eq']:6.2f}% | AbsRV-FPR={summ['absent_rv_fpr_gt0_pct']:5.2f}% ({summ['elapsed_s']:.1f}s)")
    return {"suite": suite_name, "model_state_sha256": sha_ref, "results": results}


def evaluate_mms(
    presets: Sequence[str],
    out_dir: Path,
    device: torch.device,
    ckpt_dir: Path = DEFAULT_CKPT_DIR,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
) -> Dict[str, object]:
    set_seed(42)
    model = load_model_checkpoint("resunet", 1, 4, ckpt_dir / "source_mms_resunet_seed42.pth", device)
    ds_tr = build_dataset("mms", "source_train", "B", transform=build_transform("mms", "tta"), seed=42, processed_root=processed_root)
    src_imgs = [ds_tr[i][0].float() for i in range(len(ds_tr))]
    domains = ("A", "C", "D")
    targets = {
        d: [(ds[i][0].float(), ds[i][1].numpy().astype(np.int64), os.path.basename(ds.items[i].image_path))
            for ds in (build_dataset("mms", "target", d, seed=42, processed_root=processed_root),)
            for i in range(len(ds))]
        for d in domains
    }
    return _evaluate_cardiac_common("mms", model, src_imgs, targets, domains, (1, 256, 256), presets, out_dir, device)


def evaluate_mms2d_paper(
    presets: Sequence[str],
    out_dir: Path,
    device: torch.device,
    ckpt_dir: Path = DEFAULT_CKPT_DIR,
    raw_root: Path = DEFAULT_RAW_ROOT,
) -> Dict[str, object]:
    set_seed(1)
    model = load_model_checkpoint("sictta_unet", 1, 4, ckpt_dir / "source_mms2d_unet_A.pth", device)
    src_imgs, targets = load_mms2d_paper_splits(raw_root / "mms2d_paper")
    return _evaluate_cardiac_common("mms2d_paper", model, src_imgs, targets, ("B", "C", "D"), (1, 320, 320), presets, out_dir, device)


def evaluate_optic(
    presets: Sequence[str],
    out_dir: Path,
    device: torch.device,
    ckpt_dir: Path = DEFAULT_CKPT_DIR,
    raw_root: Path = DEFAULT_RAW_ROOT,
    sources: Sequence[str] = OPTIC_DOMAINS,
) -> Dict[str, object]:
    domain_cache = {d: load_optic_domain(d, optic_root=raw_root / "optic") for d in OPTIC_DOMAINS}
    preset_streams: Dict[str, Dict[str, Dict[str, object]]] = {p: {} for p in presets}
    src_shas: Dict[str, str] = {}

    for src in sources:
        set_seed(42)
        ckpt = ckpt_dir / "optic" / src / "last-Res_Unet.pth"
        model = load_model_checkpoint("vptta_resunet", 3, 2, ckpt, device)
        sha_ref = model_state_sha256(model)
        src_shas[src] = sha_ref

        sd_src = domain_cache[src]
        src_imgs = sd_src["imgs"][: int(sd_src["n_train"])]
        base = PrevalenceDeconfoundedDualTTA.fit_from_source(model, src_imgs, device, "multilabel_sigmoid", PDDualConfig())
        assert model_state_sha256(model) == sha_ref

        for pname in presets:
            cfg = resolve_preset(pname)
            adapter = _build_preset_adapter(pname, cfg, base, model, (3, 512, 512), "multilabel_sigmoid", device)
            for tgt in OPTIC_DOMAINS:
                if tgt == src:
                    continue
                adapter.reset_stream()
                td = domain_cache[tgt]
                rows: List[Dict[str, object]] = []
                dds, cds = [], []
                for idx, (nm, im, gd, gc) in enumerate(zip(td["names"], td["imgs"], td["gt_discs"], td["gt_cups"])):
                    logits, dg = adapter.predict_single(im, device=device)
                    pbin = (torch.sigmoid(logits[0]) >= 0.5).detach().cpu().numpy().astype(np.bool_)
                    dd, cd = optic_dice_pct(pbin, gd, gc)
                    dds.append(dd)
                    cds.append(cd)
                    rows.append({
                        "call_idx": idx,
                        "image": nm,
                        "disc_dice": dd,
                        "cup_dice": cd,
                        "mean_dice": 0.5 * (dd + cd),
                        "lam_stream_max": float(dg.get("lam_stream_max", 0.0)),
                        "lam_indiv_max": float(dg.get("lam_indiv_max", 0.0)),
                        "resets": int(dg.get("resets", 0.0)),
                        "w_supp": float(dg.get("w_supp", 0.0)),
                        "w_shift_mean": float(dg.get("w_shift_mean", 0.0)),
                        "lam_deep_mean": float(dg.get("lam_deep_mean", 0.0)),
                    })
                assert model_state_sha256(model) == sha_ref
                csv_path = out_dir / "csv" / "optic" / pname / f"per_image_{src}_to_{tgt}.csv"
                c_sha = write_csv_atomic(csv_path, OPTIC_COLS, rows)
                dm, cm = float(np.mean(dds)), float(np.mean(cds))
                preset_streams[pname][f"{src}_to_{tgt}"] = {
                    "n_images": len(rows),
                    "disc_dice": round(dm, 4),
                    "cup_dice": round(cm, 4),
                    "mean_dice": round(0.5 * (dm + cm), 4),
                    "csv_path": str(csv_path),
                    "csv_sha256": c_sha,
                }

    results = {}
    for pname in presets:
        st = preset_streams[pname]
        mean_all = float(np.mean([v["mean_dice"] for v in st.values()]))
        results[pname] = {"config": asdict(resolve_preset(pname)), "mean_20stream_dice": round(mean_all, 4), "streams": st}
        print(f"[OPTIC    | {pname:20s}] Dice={mean_all:6.2f}% ({len(st)} streams)")
    return {"suite": "optic", "source_model_sha256s": src_shas, "results": results}


def main() -> None:
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description="Evaluate DeCoTTA and decoupled ablations across 5 medical segmentation benchmarks.")
    ap.add_argument("--suite", required=True, choices=["prostate", "rigaplus", "mms", "mms2d_paper", "optic", "all"])
    ap.add_argument("--presets", nargs="+", default=["pd_dual_tta"])
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--ckpt_dir", type=Path, default=DEFAULT_CKPT_DIR)
    ap.add_argument("--processed_root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    ap.add_argument("--raw_root", type=Path, default=DEFAULT_RAW_ROOT)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, object] = {}

    suites = ("prostate", "rigaplus", "mms", "mms2d_paper", "optic") if args.suite == "all" else (args.suite,)
    for s in suites:
        if s == "prostate":
            summary[s] = evaluate_prostate(args.presets, args.out_dir, dev, args.ckpt_dir, args.processed_root)
        elif s == "rigaplus":
            summary[s] = evaluate_rigaplus(args.presets, args.out_dir, dev, args.ckpt_dir, args.processed_root)
        elif s == "mms":
            summary[s] = evaluate_mms(args.presets, args.out_dir, dev, args.ckpt_dir, args.processed_root)
        elif s == "mms2d_paper":
            summary[s] = evaluate_mms2d_paper(args.presets, args.out_dir, dev, args.ckpt_dir, args.raw_root)
        elif s == "optic":
            summary[s] = evaluate_optic(args.presets, args.out_dir, dev, args.ckpt_dir, args.raw_root)

    out_json = args.out_dir / f"summary_{args.suite}.json"
    tmp = out_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, out_json)
    print(f"Wrote summary JSON to {out_json}")


if __name__ == "__main__":
    main()
