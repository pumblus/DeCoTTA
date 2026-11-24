#!/usr/bin/env python3
"""Export and inspect privacy-preserving `.pt` source-domain moment summaries (zero patient pixels)."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from data import (
    DEFAULT_PROCESSED_ROOT,
    DEFAULT_RAW_ROOT,
    PROSTATE_SOURCE_SITE,
    build_dataset,
    build_transform,
    load_mms2d_paper_splits,
    load_optic_domain,
    load_prostate_manifest_by_site,
)
from evaluate.run_benchmark import DEFAULT_CKPT_DIR, set_seed
from models import load_model_checkpoint
from tta import PDDualConfig, PrevalenceDeconfoundedDualTTA


def main() -> None:
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description="Export a standalone `.pt` source-domain moment summary.")
    ap.add_argument("--suite", required=True, choices=["prostate", "rigaplus", "mms", "mms2d_paper", "optic"])
    ap.add_argument("--optic_source", default="Drishti_GS")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ckpt_dir", type=Path, default=DEFAULT_CKPT_DIR)
    ap.add_argument("--processed_root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    ap.add_argument("--raw_root", type=Path, default=DEFAULT_RAW_ROOT)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(1 if args.suite == "mms2d_paper" else 42)
    dev = torch.device(args.device)
    if args.suite == "prostate":
        model = load_model_checkpoint("vptta_resunet", 1, 2, args.ckpt_dir / "source_prostate_runmc_seed42.pth", dev)
        by_site = load_prostate_manifest_by_site(args.processed_root)
        p_root = args.processed_root / "prostate"
        src_imgs = [torch.from_numpy(np.load(p_root / r["image"]).astype(np.float32)) for r in by_site[PROSTATE_SOURCE_SITE]]
        mode = "multiclass_softmax"
    elif args.suite == "rigaplus":
        model = load_model_checkpoint("resunet", 3, 3, args.ckpt_dir / "source_rigaplus_resunet_seed42.pth", dev)
        ds = build_dataset(
            "rigaplus",
            "source_train",
            "Magrabia",
            transform=build_transform("rigaplus", "tta"),
            seed=42,
            processed_root=args.processed_root,
        )
        src_imgs = [ds[i][0].float() for i in range(len(ds))]
        mode = "multiclass_softmax"
    elif args.suite == "mms":
        model = load_model_checkpoint("resunet", 1, 4, args.ckpt_dir / "source_mms_resunet_seed42.pth", dev)
        ds = build_dataset(
            "mms",
            "source_train",
            "B",
            transform=build_transform("mms", "tta"),
            seed=42,
            processed_root=args.processed_root,
        )
        src_imgs = [ds[i][0].float() for i in range(len(ds))]
        mode = "multiclass_softmax"
    elif args.suite == "mms2d_paper":
        model = load_model_checkpoint("sictta_unet", 1, 4, args.ckpt_dir / "source_mms2d_unet_A.pth", dev)
        src_imgs, _ = load_mms2d_paper_splits(args.raw_root / "mms2d_paper")
        mode = "multiclass_softmax"
    else:
        ckpt = args.ckpt_dir / "optic" / args.optic_source / "last-Res_Unet.pth"
        model = load_model_checkpoint("vptta_resunet", 3, 2, ckpt, dev)
        d = load_optic_domain(args.optic_source, optic_root=args.raw_root / "optic")
        src_imgs = d["imgs"][: int(d["n_train"])]
        mode = "multilabel_sigmoid"

    adapter = PrevalenceDeconfoundedDualTTA.fit_from_source(model, src_imgs, dev, mode, PDDualConfig())
    saved = adapter.save_source_summary(args.output)
    mb = saved.stat().st_size / (1024.0 * 1024.0)
    print(f"Exported {args.suite} source summary -> {saved} ({mb:.2f} MB, 0 raw source images)")


if __name__ == "__main__":
    main()
