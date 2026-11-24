"""Supervised source-domain pretraining (strictly on source domain training split)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from data import build_dataset, build_transform
from models import build_backbone, get_io_spec
from tta.masks import extract_logits


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    inter = (probs * onehot).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
    dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
    return 1.0 - (dice[1:].mean() if num_classes > 1 else dice.mean())


def combined_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    return 0.5 * F.cross_entropy(logits, target.long()) + 0.5 * soft_dice_loss(logits, target, num_classes)


@torch.no_grad()
def evaluate_foreground_dice(model: nn.Module, loader: DataLoader, num_classes: int, device: torch.device) -> float:
    model.eval()
    scores = []
    for x, y in loader:
        pred = extract_logits(model(x.to(device))).argmax(dim=1).cpu()
        for b in range(pred.shape[0]):
            cls_d = []
            for c in range(1, num_classes):
                pc, gc = (pred[b] == c), (y[b] == c)
                ps, gs = int(pc.sum()), int(gc.sum())
                if ps == 0 and gs == 0:
                    cls_d.append(1.0)
                elif ps == 0 or gs == 0:
                    cls_d.append(0.0)
                else:
                    cls_d.append(2.0 * float((pc & gc).sum()) / float(ps + gs))
            scores.append(float(np.mean(cls_d)))
    return float(np.mean(scores)) if scores else 0.0


def build_source_loaders(dataset: str, source_domain: str, batch_size: int, seed: int) -> Tuple[DataLoader, DataLoader]:
    train_ds = build_dataset(dataset, "source_train", source_domain, transform=build_transform(dataset, "train"), seed=seed)
    val_ds = build_dataset(dataset, "source_val", source_domain, transform=build_transform(dataset, "eval"), seed=seed)
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, generator=gen, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def main() -> None:
    ap = argparse.ArgumentParser(description="Supervised source-domain segmentation pretraining.")
    ap.add_argument("--dataset", required=True, choices=["mms", "rigaplus", "prostate"])
    ap.add_argument("--backbone", default="resunet", choices=["resunet", "vptta_resunet", "unet"])
    ap.add_argument("--source_domain", required=True, help="'B' for M&Ms, 'Magrabia' for RIGA+, 'RUNMC' for Prostate")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--log_dir", type=Path, default=Path("logs"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch, num_classes = get_io_spec(args.dataset)
    model = build_backbone(args.backbone, in_channels=in_ch, num_classes=num_classes).to(device)
    train_loader, val_loader = build_source_loaders(args.dataset, args.source_domain, args.batch_size, args.seed)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, nesterov=True, weight_decay=3e-5)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.epochs, power=0.9)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"source_pretrain_{args.dataset}_{args.backbone}_seed{args.seed}.jsonl"

    best_val = -1.0
    with log_path.open("w", encoding="utf-8") as log_fh:
        for epoch in range(args.epochs):
            t0 = time.perf_counter()
            model.train()
            losses = []
            for x, y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = extract_logits(model(x.to(device)))
                loss = combined_loss(logits, y.to(device), num_classes)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            scheduler.step()

            rec: Dict[str, object] = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "elapsed_s": round(time.perf_counter() - t0, 2),
            }
            if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
                val_d = evaluate_foreground_dice(model, val_loader, num_classes, device)
                rec["val_dice"] = val_d
                if val_d > best_val:
                    best_val = val_d
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "backbone": args.backbone,
                            "dataset": args.dataset,
                            "source_domain": args.source_domain,
                            "seed": args.seed,
                            "epoch": epoch + 1,
                            "val_dice": best_val,
                        },
                        args.output,
                    )
            log_fh.write(json.dumps(rec) + "\n")
            log_fh.flush()


if __name__ == "__main__":
    main()
