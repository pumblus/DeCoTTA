"""Preprocess M&Ms cardiac MRI volumes into 2D `(1, 256, 256)` slices under `data/processed/mms/`."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

SPLITS = [
    ("Training/Labeled", "train_labeled"),
    ("Training/Unlabeled", "train_unlabeled"),
    ("Validation", "validation"),
    ("Testing", "testing"),
]
TARGET_SIZE = 256


def zscore_slice(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)


def center_crop_or_pad_2d(arr: np.ndarray, target_h: int, target_w: int, pad_value=0) -> np.ndarray:
    h, w = arr.shape[-2:]
    if h > target_h:
        s = (h - target_h) // 2
        arr = arr[s : s + target_h, :]
    if w > target_w:
        s = (w - target_w) // 2
        arr = arr[:, s : s + target_w]
    h, w = arr.shape[-2:]
    if h < target_h or w < target_w:
        pad_h, pad_w = target_h - h, target_w - w
        top, left = pad_h // 2, pad_w // 2
        arr = np.pad(arr, ((top, pad_h - top), (left, pad_w - left)), mode="constant", constant_values=pad_value)
    return arr


def load_csv(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    return {r["External code"]: (str(r["Vendor"]), int(r["ED"]), int(r["ES"])) for _, r in df.iterrows()}


def process_patient(
    pat_dir: Path,
    patient_id: str,
    info: tuple,
    out_split_dir: Path,
    manifest_rows: list,
    split_label: str,
) -> int:
    vendor, ed, es = info
    sa = pat_dir / f"{patient_id}_sa.nii.gz"
    gt = pat_dir / f"{patient_id}_sa_gt.nii.gz"
    if not sa.exists() or not gt.exists():
        return 0

    img = nib.load(str(sa)).get_fdata()
    seg = nib.load(str(gt)).get_fdata().astype(np.int64)
    n_saved = 0
    out_vendor_dir = out_split_dir / vendor
    out_vendor_dir.mkdir(parents=True, exist_ok=True)

    for phase_name, t_idx in (("ED", ed), ("ES", es)):
        if t_idx >= img.shape[3]:
            continue
        img_t, seg_t = img[..., t_idx], seg[..., t_idx]
        for z in range(img_t.shape[2]):
            seg_slice = seg_t[..., z]
            if seg_slice.sum() == 0:
                continue
            img_slice = center_crop_or_pad_2d(zscore_slice(img_t[..., z].astype(np.float32)), TARGET_SIZE, TARGET_SIZE, 0.0)
            seg_slice = center_crop_or_pad_2d(seg_slice.astype(np.int64), TARGET_SIZE, TARGET_SIZE, 0)
            tag = f"{patient_id}_{phase_name}_z{z:02d}"
            img_path = out_vendor_dir / f"{tag}.npy"
            mask_path = out_vendor_dir / f"{tag}_gt.npy"
            np.save(img_path, img_slice[np.newaxis, ...].astype(np.float32))
            np.save(mask_path, seg_slice.astype(np.int64))
            manifest_rows.append({
                "patient": patient_id,
                "vendor": vendor,
                "phase": phase_name,
                "z": z,
                "split": split_label,
                "image": str(img_path.relative_to(out_split_dir.parent)),
                "mask": str(mask_path.relative_to(out_split_dir.parent)),
            })
            n_saved += 1
    return n_saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    args = ap.parse_args()

    csv_files = list(args.raw_dir.glob("*Dataset_information*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"M&Ms Dataset_information CSV not found under {args.raw_dir}")
    info_map = load_csv(csv_files[0])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list = []

    for split_subdir, split_label in SPLITS:
        in_dir = args.raw_dir / split_subdir
        if not in_dir.exists():
            continue
        out_split_dir = args.output_dir / split_label
        for pat_dir in sorted(in_dir.iterdir()):
            if pat_dir.is_dir() and pat_dir.name in info_map:
                process_patient(pat_dir, pat_dir.name, info_map[pat_dir.name], out_split_dir, manifest, split_label)

    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["patient", "vendor", "phase", "z", "split", "image", "mask"])
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Wrote {len(manifest)} slices to {manifest_path}")


if __name__ == "__main__":
    main()
