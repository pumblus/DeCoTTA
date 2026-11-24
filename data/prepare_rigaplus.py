"""Preprocess RIGA+ fundus images into `(3, 512, 512)` `.npy` tensors under `data/processed/rigaplus/`."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

TARGET_SIZE = (512, 512)
SITES = ["BinRushed", "Magrabia", "MESSIDOR_Base1", "MESSIDOR_Base2", "MESSIDOR_Base3"]
SPLITS = ["train", "test"]


def reencode_mask(label_arr: np.ndarray) -> np.ndarray:
    """Encode RIGA+ annotation `{0: bg, 255: disc, 128: cup}` into `{0, 1, 2}`."""
    mask = np.zeros_like(label_arr, dtype=np.int64)
    mask[label_arr == 255] = 1
    mask[label_arr == 128] = 2
    return mask


def to_annotator1(mask_path: str) -> str:
    p = Path(mask_path)
    if p.stem.split("-")[-1].isdigit():
        return mask_path
    return str(p.with_name(f"{p.stem}-1{p.suffix}"))


def process_pair(
    raw_dir: Path,
    image_rel: str,
    mask_rel: str,
    out_dir: Path,
    manifest_rows: list,
    site: str,
    split: str,
) -> bool:
    img_path = raw_dir / image_rel
    mask_path = raw_dir / to_annotator1(mask_rel)
    if not img_path.exists() or not mask_path.exists():
        return False

    img = Image.open(img_path).convert("RGB").resize(TARGET_SIZE, Image.BILINEAR)
    label = Image.open(mask_path).convert("L").resize(TARGET_SIZE, Image.NEAREST)
    img_np = np.array(img).transpose(2, 0, 1).astype(np.float32)
    img_np = (img_np - img_np.min()) / max(float(img_np.max() - img_np.min()), 1e-8)
    mask_np = reencode_mask(np.array(label))

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_rel).stem.replace("/", "_")
    sub = Path(image_rel).parent.name
    tag = f"{sub}__{stem}"
    img_out = out_dir / f"{tag}.npy"
    mask_out = out_dir / f"{tag}_gt.npy"
    np.save(img_out, img_np)
    np.save(mask_out, mask_np)
    manifest_rows.append({
        "site": site,
        "split": split,
        "subfolder": sub,
        "stem": stem,
        "image": str(img_out.relative_to(out_dir.parent.parent)),
        "mask": str(mask_out.relative_to(out_dir.parent.parent)),
    })
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list = []
    for site in SITES:
        for split in SPLITS:
            csv_path = args.raw_dir / f"{site}_{split}.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            out_dir = args.output_dir / site / split
            for _, row in df.iterrows():
                process_pair(args.raw_dir, row["image"], row["mask"], out_dir, manifest, site, split)

    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["site", "split", "subfolder", "stem", "image", "mask"])
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Wrote {len(manifest)} images to {manifest_path}")


if __name__ == "__main__":
    main()
