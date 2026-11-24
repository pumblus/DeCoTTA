"""Preprocess 6-center 3D Prostate MRI volumes into `(1, 384, 384)` axial slices."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.ndimage

SITES = ("RUNMC", "BMC", "I2CVB", "UCL", "BIDMC", "HK")
TARGET_HW = (384, 384)
MANIFEST_COLUMNS = [
    "site", "case_id", "split", "all_order_idx", "z", "num_z",
    "has_fg", "fg_pixels", "spacing_z", "spacing_y", "spacing_x",
    "raw_nonzero_labels", "image", "mask",
]


def _read_site_csv(csv_path: Path) -> list[tuple[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, skipinitialspace=True)
        return [(r["image"].strip(), r["mask"].strip()) for r in reader]


def _extract_case_id(rel_path: str) -> str:
    name = Path(rel_path).name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def _load_nifti_zyx(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    nii = nib.load(str(path))
    arr = np.asarray(nii.dataobj)
    if arr.ndim == 4:
        arr = arr[..., 0]
    vol = np.transpose(arr, (2, 1, 0)).astype(np.float32)
    zooms = nii.header.get_zooms()[:3]
    return vol, (float(zooms[2]), float(zooms[1]), float(zooms[0]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []

    for site in SITES:
        all_rows = _read_site_csv(args.raw_dir / f"{site}_all.csv")
        train_set = set(_read_site_csv(args.raw_dir / f"{site}_train.csv"))
        for split_name in ("train", "test"):
            (args.output_dir / site / split_name).mkdir(parents=True, exist_ok=True)

        for all_idx, (img_rel, seg_rel) in enumerate(all_rows):
            split_name = "train" if (img_rel, seg_rel) in train_set else "test"
            case_id = _extract_case_id(img_rel)
            img_vol, (sp_z, sp_y, sp_x) = _load_nifti_zyx(args.raw_dir / img_rel)
            seg_vol, _ = _load_nifti_zyx(args.raw_dir / seg_rel)

            nonzero = sorted(int(v) for v in set(np.rint(np.unique(seg_vol)).astype(np.int64).tolist()) if v > 0)
            z_dim, h_dim, w_dim = img_vol.shape
            if (h_dim, w_dim) != TARGET_HW:
                zf = (1.0, TARGET_HW[0] / float(h_dim), TARGET_HW[1] / float(w_dim))
                img_vol = scipy.ndimage.zoom(img_vol, zf, order=3).astype(np.float32)
                seg_vol = scipy.ndimage.zoom(seg_vol, zf, order=0)

            mask_vol = (np.rint(seg_vol).astype(np.int64) >= 1).astype(np.int64)
            p5, p95 = np.percentile(img_vol, 5), np.percentile(img_vol, 95)
            img_vol = np.clip(img_vol, p5, p95)
            img_vol = ((img_vol - img_vol.mean()) / max(float(img_vol.std()), 1e-8)).astype(np.float32)

            for z_idx in range(z_dim):
                img_rel_out = f"{site}/{split_name}/{case_id}_z{z_idx:03d}.npy"
                msk_rel_out = f"{site}/{split_name}/{case_id}_z{z_idx:03d}_gt.npy"
                np.save(args.output_dir / img_rel_out, img_vol[z_idx][np.newaxis, ...])
                np.save(args.output_dir / msk_rel_out, mask_vol[z_idx])
                fg_px = int(mask_vol[z_idx].sum())
                manifest_rows.append({
                    "site": site,
                    "case_id": case_id,
                    "split": split_name,
                    "all_order_idx": all_idx,
                    "z": z_idx,
                    "num_z": z_dim,
                    "has_fg": 1 if fg_px > 0 else 0,
                    "fg_pixels": fg_px,
                    "spacing_z": sp_z,
                    "spacing_y": sp_y,
                    "spacing_x": sp_x,
                    "raw_nonzero_labels": ",".join(str(v) for v in nonzero),
                    "image": img_rel_out,
                    "mask": msk_rel_out,
                })

    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Wrote {len(manifest_rows)} slices to {manifest_path}")


if __name__ == "__main__":
    main()
