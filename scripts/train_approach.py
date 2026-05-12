#!/usr/bin/env python3
"""
Train the ApproachEstimator CNN.

Uses inverse-frequency weighted sampling so that each ratio bin (0.0–0.1,
0.1–0.2, …, 0.9–1.0) is drawn roughly equally often, counteracting the
heavy right-skew of the raw data (~28 % in [0.9,1.0) vs ~2 % in [0.0,0.1)).

Usage::

    python scripts/train_approach.py
    python scripts/train_approach.py --crops dataset/crops --epochs 50
    python scripts/train_approach.py --no-weighted   # disable weighted sampling
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure project root is on sys.path so `from src.xxx` works
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.vision.approach_estimator import ApproachEstimator

NUM_BINS = 20  # 0.05-wide bins for weight computation


class CropDataset(Dataset):
    """Dataset of 64x64 crops with approach_ratio labels encoded in filenames."""

    def __init__(self, image_paths: List[Path], augment: bool = False):
        self.image_paths = image_paths
        self.augment = augment
        # Pre-parse labels for fast access (used by sampler weight computation)
        self.labels = np.array([
            float(p.stem.rsplit("_", 1)[-1]) for p in image_paths
        ], dtype=np.float32)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        import cv2
        path = self.image_paths[idx]
        ratio = self.labels[idx]

        img = cv2.imread(str(path))
        if img is None:
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            ratio = 0.0

        if img.shape[:2] != (64, 64):
            img = cv2.resize(img, (64, 64))

        # Data augmentation for training
        if self.augment:
            # Random horizontal flip
            if random.random() < 0.5:
                img = img[:, ::-1, :].copy()
            # Random brightness jitter ±15 %
            if random.random() < 0.5:
                factor = 0.85 + random.random() * 0.30
                img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        tensor = torch.from_numpy(img[:, :, ::-1].copy()).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        label = torch.tensor([ratio], dtype=torch.float32)
        return tensor, label


def _build_weighted_sampler(
    labels: np.ndarray,
    num_bins: int = NUM_BINS,
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler with inverse-frequency weights per bin.

    Each bin [k/num_bins, (k+1)/num_bins) gets weight = 1 / count(bin),
    so bins with fewer samples are drawn more often.
    """
    bin_indices = np.clip((labels * num_bins).astype(int), 0, num_bins - 1)

    # Count per bin
    bin_counts = np.bincount(bin_indices, minlength=num_bins).astype(np.float64)
    bin_counts = np.maximum(bin_counts, 1.0)  # avoid division by zero

    # Weight per sample = 1 / count(its bin)
    sample_weights = 1.0 / bin_counts[bin_indices]

    # Print distribution info
    print(f"\n  Bin distribution (before weighting):")
    for i in range(num_bins):
        lo = i / num_bins
        hi = (i + 1) / num_bins
        cnt = int(bin_counts[i])
        w = 1.0 / bin_counts[i]
        bar = "#" * min(50, int(cnt / bin_counts.max() * 50))
        print(f"    [{lo:.2f}, {hi:.2f})  n={cnt:6d}  w={w:.6f}  {bar}")

    effective_ratio = sample_weights.max() / sample_weights.min()
    print(f"  Max/min weight ratio: {effective_ratio:.1f}x")
    print(f"  After weighting, each bin is drawn ~equally.\n")

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(labels),
        replacement=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Train ApproachEstimator")
    parser.add_argument("--crops", default="dataset/crops")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", default="runs/approach/best.pth")
    parser.add_argument("--no-weighted", action="store_true",
                        help="Disable weighted sampling (use uniform)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    crops_dir = Path(args.crops)
    all_crops = sorted(crops_dir.glob("crop_*.jpg"))
    if not all_crops:
        print(f"ERROR: No crops found in {crops_dir}")
        return

    print(f"Found {len(all_crops)} crops")

    random.seed(42)
    random.shuffle(all_crops)
    split_idx = int(len(all_crops) * (1 - args.val_ratio))
    train_paths = all_crops[:split_idx]
    val_paths = all_crops[split_idx:]
    print(f"Train: {len(train_paths)} | Val: {len(val_paths)}")

    use_augment = not args.no_augment
    train_dataset = CropDataset(train_paths, augment=use_augment)
    val_dataset = CropDataset(val_paths, augment=False)

    # Build weighted sampler for training
    use_weighted = not args.no_weighted
    train_sampler = None
    if use_weighted:
        print("Building weighted sampler...")
        train_sampler = _build_weighted_sampler(train_dataset.labels)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch,
        shuffle=(train_sampler is None),  # shuffle only if no sampler
        sampler=train_sampler,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    model = ApproachEstimator().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Weighted sampling: {use_weighted}")
    print(f"Data augmentation: {use_augment}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_mae = float("inf")

    for epoch in range(args.epochs):
        # ── Train ──
        model.train()
        train_loss, train_n = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            train_n += imgs.size(0)
        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss, val_n = 0.0, 0
        # Per-bucket MAE for monitoring
        bucket_err = [[] for _ in range(5)]  # [0,.2) [.2,.4) [.4,.6) [.6,.8) [.8,1]
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs)
                val_loss += criterion(preds, labels).item() * imgs.size(0)
                val_n += imgs.size(0)
                # Per-bucket
                errs = (preds - labels).abs().cpu().numpy().flatten()
                labs = labels.cpu().numpy().flatten()
                for e, l in zip(errs, labs):
                    b = min(4, int(l * 5))
                    bucket_err[b].append(e)

        avg_val = val_loss / max(1, val_n)

        # Overall MAE
        all_errs = []
        for be in bucket_err:
            all_errs.extend(be)
        val_mae = np.mean(all_errs) if all_errs else 0.0

        # Per-bucket MAE string
        bucket_str = "  ".join(
            f"{np.mean(be):.3f}" if be else "  n/a"
            for be in bucket_err
        )

        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            torch.save(model.state_dict(), output_path)

        print(
            f"[{epoch+1:3d}/{args.epochs}] "
            f"train={train_loss/max(1,train_n):.5f} "
            f"val={avg_val:.5f} "
            f"mae={val_mae:.4f} "
            f"[{bucket_str}]"
            f"{' *' if is_best else ''}"
        )

    print(f"\nBest val MAE: {best_val_mae:.4f} → {output_path}")
    print(f"Bucket labels: [0,.2) [.2,.4) [.4,.6) [.6,.8) [.8,1]")


if __name__ == "__main__":
    main()
