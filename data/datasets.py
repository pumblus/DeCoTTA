"""Unified dataset and stream loaders for M&Ms, RIGA+, Prostate, OPTIC, and `mms2d_paper`."""
from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROCESSED_ROOT = Path(os.environ.get("PD_TTA_PROCESSED_DIR", str(REPO_ROOT / "data" / "processed")))
DEFAULT_RAW_ROOT = Path(os.environ.get("PD_TTA_RAW_DIR", str(REPO_ROOT / "data" / "raw")))

PROSTATE_SOURCE_SITE = "RUNMC"
PROSTATE_TARGET_SITES: Tuple[str, ...] = ("BMC", "I2CVB", "UCL", "BIDMC", "HK")
OPTIC_DOMAINS: Tuple[str, ...] = ("RIM_ONE_r3", "REFUGE", "ORIGA", "REFUGE_Valid", "Drishti_GS")
RIGAPLUS_SITE_GROUPS = {"MESSIDOR": ["MESSIDOR_Base1", "MESSIDOR_Base2", "MESSIDOR_Base3"]}


@dataclass
class SliceItem:
    image_path: str
    mask_path: str


class NpySliceDataset(Dataset):
    """Dataset over `(image.npy, mask.npy)` slice pairs."""

    def __init__(self, items: List[SliceItem], transform: Optional[Callable] = None) -> None:
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        it = self.items[idx]
        img_t = torch.from_numpy(np.load(it.image_path).astype(np.float32))
        mask_t = torch.from_numpy(np.load(it.mask_path).astype(np.int64))
        if self.transform is not None:
            img_t, mask_t = self.transform(img_t, mask_t)
        return img_t, mask_t


def _load_manifest(dataset: str, processed_root: Path = DEFAULT_PROCESSED_ROOT) -> pd.DataFrame:
    base = processed_root / dataset
    csv_path = base / "manifest.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {csv_path}. Run `python data/prepare_{dataset}.py` first.")
    df = pd.read_csv(csv_path)
    df["_image_abs"] = df["image"].apply(lambda r: str(base / r))
    df["_mask_abs"] = df["mask"].apply(lambda r: str(base / r))
    return df


def mms_source_items(source_vendor: str = "B", val_fraction: float = 0.1, seed: int = 42, processed_root: Path = DEFAULT_PROCESSED_ROOT):
    df = _load_manifest("mms", processed_root)
    df = df[(df["vendor"] == source_vendor) & (df["split"] == "train_labeled")]
    if df.empty:
        raise ValueError(f"No M&Ms train_labeled slices for vendor {source_vendor!r}")
    patients = sorted(df["patient"].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)
    val_pts = set(patients[: max(1, int(len(patients) * val_fraction))])
    train_items = [SliceItem(r["_image_abs"], r["_mask_abs"]) for _, r in df[~df["patient"].isin(val_pts)].iterrows()]
    val_items = [SliceItem(r["_image_abs"], r["_mask_abs"]) for _, r in df[df["patient"].isin(val_pts)].iterrows()]
    return train_items, val_items


def mms_target_items(target_vendor: str, processed_root: Path = DEFAULT_PROCESSED_ROOT) -> List[SliceItem]:
    df = _load_manifest("mms", processed_root)
    df = df[df["vendor"] == target_vendor].sort_values(["patient", "phase", "z"])
    if df.empty:
        raise ValueError(f"No M&Ms slices for vendor {target_vendor!r}")
    return [SliceItem(r["_image_abs"], r["_mask_abs"]) for _, r in df.iterrows()]


def rigaplus_source_items(source_site: str = "Magrabia", val_fraction: float = 0.1, seed: int = 42, processed_root: Path = DEFAULT_PROCESSED_ROOT):
    sites = RIGAPLUS_SITE_GROUPS.get(source_site, [source_site])
    df = _load_manifest("rigaplus", processed_root)
    df = df[df["site"].isin(sites) & (df["split"] == "train")].sort_values(["site", "subfolder", "stem"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No RIGA+ train split for site {source_site!r}")
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    val_idx = set(idx[: max(1, int(len(idx) * val_fraction))].tolist())
    train_items, val_items = [], []
    for i, (_, r) in enumerate(df.iterrows()):
        (val_items if i in val_idx else train_items).append(SliceItem(r["_image_abs"], r["_mask_abs"]))
    return train_items, val_items


def rigaplus_target_items(target_site: str, processed_root: Path = DEFAULT_PROCESSED_ROOT) -> List[SliceItem]:
    sites = RIGAPLUS_SITE_GROUPS.get(target_site, [target_site])
    df = _load_manifest("rigaplus", processed_root)
    df = df[df["site"].isin(sites) & (df["split"] == "test")].sort_values(["site", "subfolder", "stem"])
    if df.empty:
        raise ValueError(f"No RIGA+ test split for site {target_site!r}")
    return [SliceItem(r["_image_abs"], r["_mask_abs"]) for _, r in df.iterrows()]


def prostate_source_items(source_site: str = PROSTATE_SOURCE_SITE, val_fraction: float = 0.1, seed: int = 42, processed_root: Path = DEFAULT_PROCESSED_ROOT):
    df = _load_manifest("prostate", processed_root)
    df = df[df["site"] == source_site].sort_values(["all_order_idx", "z"])
    if df.empty:
        raise ValueError(f"No Prostate slices for site {source_site!r}")
    cases = sorted(df["case_id"].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(cases)
    val_cases = set(cases[: max(1, int(len(cases) * val_fraction))])
    train_items = [SliceItem(r["_image_abs"], r["_mask_abs"]) for _, r in df[~df["case_id"].isin(val_cases)].iterrows()]
    val_items = [SliceItem(r["_image_abs"], r["_mask_abs"]) for _, r in df[df["case_id"].isin(val_cases)].iterrows()]
    return train_items, val_items


def load_prostate_manifest_by_site(processed_root: Path = DEFAULT_PROCESSED_ROOT) -> Dict[str, List[Dict[str, str]]]:
    prostate_dir = processed_root / "prostate"
    manifest_path = prostate_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prostate manifest not found: {manifest_path}")
    by_site: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with manifest_path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_site[row["site"]].append(row)
    for site in by_site:
        by_site[site].sort(key=lambda r: (int(r["all_order_idx"]), int(r["z"])))
    return by_site


def load_optic_domain(
    domain: str,
    optic_root: Optional[Path] = None,
    image_size: int = 512,
) -> Dict[str, object]:
    """Load an OPTIC domain (`n_train`, `names`, `imgs`, `gt_discs`, `gt_cups`) matching official protocol."""
    root = Path(optic_root) if optic_root is not None else (DEFAULT_RAW_ROOT / "optic")
    if not root.is_dir():
        raise FileNotFoundError(f"OPTIC raw directory not found: {root}")

    train_csvs = [f"{domain}.csv"] if domain == "REFUGE_Valid" else [f"{domain}_train.csv"]
    all_csvs = [f"{domain}.csv"] if domain == "REFUGE_Valid" else [f"{domain}_train.csv", f"{domain}_test.csv"]

    def _read_lists(csv_names: List[str]) -> Tuple[List[str], List[str]]:
        im_l, lb_l = [], []
        for cn in csv_names:
            df = pd.read_csv(root / cn)
            im_l.extend(df["image"].tolist())
            lb_l.extend(df["mask"].tolist())
        return im_l, lb_l

    list_tr, _ = _read_lists(train_csvs)
    list_all, labels_all = _read_lists(all_csvs)
    target_sz = (image_size, image_size)

    imgs: List[torch.Tensor] = []
    gt_discs: List[np.ndarray] = []
    gt_cups: List[np.ndarray] = []
    for img_rel, lbl_rel in zip(list_all, labels_all):
        if lbl_rel.endswith("tif"):
            lbl_rel = lbl_rel.replace(".tif", "-1.tif")
        img_pil = Image.open(root / img_rel).resize(target_sz)
        lbl_pil = Image.open(root / lbl_rel).convert("L").resize(target_sz, resample=Image.NEAREST)

        arr = np.array(img_pil).transpose(2, 0, 1).astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        lbl = np.array(lbl_pil)
        mask = np.zeros_like(lbl)
        mask[lbl < 255] = 1
        mask[lbl == 0] = 2

        imgs.append(torch.from_numpy(arr).float().contiguous())
        gt_discs.append(((mask == 1) | (mask == 2)).astype(np.bool_))
        gt_cups.append((mask == 2).astype(np.bool_))

    return {
        "n_train": len(list_tr),
        "names": list_all,
        "imgs": imgs,
        "gt_discs": gt_discs,
        "gt_cups": gt_cups,
    }


def _load_2d_nifti_slice(path: str) -> np.ndarray:
    arr = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    if arr.ndim == 2:
        return arr.T
    if arr.ndim == 3:
        return np.transpose(arr, (2, 1, 0))[0]
    raise ValueError(f"Unexpected NIfTI slice shape {arr.shape} at {path}")


def load_mms2d_paper_splits(mms2d_root: Optional[Path] = None) -> Tuple[List[torch.Tensor], Dict[str, List[Tuple[torch.Tensor, np.ndarray, str]]]]:
    """Load `mms2d_paper` source (`A/train.csv`) and target (`B, C, D` `all.csv`) NIfTI slices."""
    root = Path(mms2d_root) if mms2d_root is not None else (DEFAULT_RAW_ROOT / "mms2d_paper")
    if not root.is_dir():
        raise FileNotFoundError(f"mms2d_paper directory not found: {root}")

    df_src = pd.read_csv(root / "A" / "train.csv")
    src_imgs = [torch.from_numpy(_load_2d_nifti_slice(str(r.iloc[0]))[None, ...]).float() for _, r in df_src.iterrows()]

    targets: Dict[str, List[Tuple[torch.Tensor, np.ndarray, str]]] = {}
    for dom in ("B", "C", "D"):
        df_dom = pd.read_csv(root / dom / "all.csv")
        dom_list: List[Tuple[torch.Tensor, np.ndarray, str]] = []
        for _, r in df_dom.iterrows():
            img_p = str(r.iloc[0])
            lbl_p = str(r["label"])
            im_t = torch.from_numpy(_load_2d_nifti_slice(img_p)[None, ...]).float()
            gt_np = np.rint(_load_2d_nifti_slice(lbl_p)).astype(np.int64)
            dom_list.append((im_t, gt_np, os.path.basename(img_p)))
        targets[dom] = dom_list
    return src_imgs, targets


def build_dataset(
    dataset: str,
    role: str,
    vendor_or_site: str,
    transform: Optional[Callable] = None,
    seed: int = 42,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
) -> NpySliceDataset:
    if dataset == "mms":
        if role == "source_train":
            items, _ = mms_source_items(vendor_or_site, seed=seed, processed_root=processed_root)
        elif role == "source_val":
            _, items = mms_source_items(vendor_or_site, seed=seed, processed_root=processed_root)
        elif role == "target":
            items = mms_target_items(vendor_or_site, processed_root=processed_root)
        else:
            raise ValueError(f"Unknown role: {role!r}")
    elif dataset == "rigaplus":
        if role == "source_train":
            items, _ = rigaplus_source_items(vendor_or_site, seed=seed, processed_root=processed_root)
        elif role == "source_val":
            _, items = rigaplus_source_items(vendor_or_site, seed=seed, processed_root=processed_root)
        elif role == "target":
            items = rigaplus_target_items(vendor_or_site, processed_root=processed_root)
        else:
            raise ValueError(f"Unknown role: {role!r}")
    elif dataset == "prostate":
        if role == "source_train":
            items, _ = prostate_source_items(vendor_or_site, seed=seed, processed_root=processed_root)
        elif role == "source_val":
            _, items = prostate_source_items(vendor_or_site, seed=seed, processed_root=processed_root)
        else:
            raise ValueError(f"Unknown role for prostate: {role!r}")
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    return NpySliceDataset(items, transform=transform)
