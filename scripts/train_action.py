#!/usr/bin/env python3
"""
Train the ActionModel (GRU behavioral cloning) from sequence data.

Expects .npz files in dataset/sequences/ with keys:
    - states: (T, STATE_DIM) float32
    - actions: (T, ACTION_DIM) float32

Usage::

    python scripts/train_action.py
    python scripts/train_action.py --sequences dataset/sequences --epochs 60
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
from torch.utils.data import DataLoader, Dataset

from src.action.model import ActionModel
from src.action.state import STATE_DIM, ACTION_DIM


class SequenceDataset(Dataset):
    """
    Dataset of (state, action) sequences.

    Each .npz contains one full replay as a time series.
    We chunk them into fixed-length windows for batched training.
    """

    def __init__(self, npz_paths: List[Path], seq_len: int = 64):
        self.seq_len = seq_len
        self.chunks: List[Tuple[np.ndarray, np.ndarray]] = []

        for path in npz_paths:
            data = np.load(path)
            states = data["states"]   # (T, STATE_DIM)
            actions = data["actions"]  # (T, ACTION_DIM)

            # Chunk into windows with 50% overlap
            T = len(states)
            step = max(1, seq_len // 2)
            for start in range(0, T - seq_len + 1, step):
                s = states[start:start + seq_len]
                a = actions[start:start + seq_len]
                self.chunks.append((s, a))

            # Include tail if long enough
            if T >= seq_len and (T - seq_len) % step != 0:
                self.chunks.append((states[-seq_len:], actions[-seq_len:]))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        s, a = self.chunks[idx]
        return torch.from_numpy(s), torch.from_numpy(a)


def main():
    parser = argparse.ArgumentParser(description="Train ActionModel (behavioral cloning)")
    parser.add_argument("--sequences", default="dataset/sequences")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=64,
                        help="Sequence window length (default: 64 frames = ~2s at 30fps)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", default="runs/action/best.pth")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load sequences
    seq_dir = Path(args.sequences)
    npz_files = sorted(seq_dir.glob("seq_*.npz"))
    if not npz_files:
        print(f"ERROR: No sequence files found in {seq_dir}")
        return

    print(f"Found {len(npz_files)} sequence files")

    # Train/val split at file level
    random.seed(42)
    random.shuffle(npz_files)
    split_idx = int(len(npz_files) * (1 - args.val_ratio))
    train_files = npz_files[:split_idx]
    val_files = npz_files[split_idx:]

    print(f"Loading train sequences ({len(train_files)} files)...")
    train_ds = SequenceDataset(train_files, seq_len=args.seq_len)
    print(f"Loading val sequences ({len(val_files)} files)...")
    val_ds = SequenceDataset(val_files, seq_len=args.seq_len)

    print(f"Train chunks: {len(train_ds)} | Val chunks: {len(val_ds)}")

    if len(train_ds) == 0:
        print("ERROR: No training chunks (sequences too short?)")
        return

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Model
    model = ActionModel(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=args.hidden,
        num_layers=args.layers,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss: MSE for cursor movement + BCE for keys
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # ── Train ──
        model.train()
        train_total, train_move, train_key, train_n = 0.0, 0.0, 0.0, 0

        for states, actions in train_loader:
            states = states.to(device)   # (B, T, STATE_DIM)
            actions = actions.to(device)  # (B, T, ACTION_DIM)

            preds, _ = model(states)  # (B, T, ACTION_DIM)

            # Movement loss (first 2 dims: dx_norm, dy_norm)
            loss_move = mse_loss(preds[:, :, :2], actions[:, :, :2])

            # Key loss (last 2 dims: key_z, key_x as binary targets)
            loss_key = bce_loss(preds[:, :, 2:4], actions[:, :, 2:4])

            # Combined (weight keys slightly lower to avoid collapse)
            loss = loss_move + 0.5 * loss_key

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            bs = states.size(0)
            train_total += loss.item() * bs
            train_move += loss_move.item() * bs
            train_key += loss_key.item() * bs
            train_n += bs

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_total, val_move, val_key, val_n = 0.0, 0.0, 0.0, 0
        val_move_mae = 0.0
        val_key_acc = 0.0

        with torch.no_grad():
            for states, actions in val_loader:
                states = states.to(device)
                actions = actions.to(device)

                preds, _ = model(states)

                loss_move = mse_loss(preds[:, :, :2], actions[:, :, :2])
                loss_key = bce_loss(preds[:, :, 2:4], actions[:, :, 2:4])
                loss = loss_move + 0.5 * loss_key

                bs = states.size(0)
                val_total += loss.item() * bs
                val_move += loss_move.item() * bs
                val_key += loss_key.item() * bs
                val_n += bs

                # Movement MAE (unnormalised: multiply by 50 for osu!px)
                move_mae = (preds[:, :, :2] - actions[:, :, :2]).abs().mean().item() * 50.0
                val_move_mae += move_mae * bs

                # Key accuracy
                key_preds = (preds[:, :, 2:4].sigmoid() > 0.5).float()
                key_acc = (key_preds == actions[:, :, 2:4]).float().mean().item()
                val_key_acc += key_acc * bs

        avg_val = val_total / max(1, val_n)
        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            torch.save(model.state_dict(), output_path)

        print(
            f"[{epoch+1:3d}/{args.epochs}] "
            f"train={train_total/max(1,train_n):.4f} "
            f"(move={train_move/max(1,train_n):.4f} key={train_key/max(1,train_n):.4f}) | "
            f"val={avg_val:.4f} "
            f"move_mae={val_move_mae/max(1,val_n):.1f}px "
            f"key_acc={val_key_acc/max(1,val_n):.3f}"
            f"{'  *' if is_best else ''}"
        )

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Saved: {output_path}")
    print(f"\nMove MAE indicates average cursor error in osu!px per frame")
    print(f"Key accuracy is binary classification accuracy for Z/X presses")


if __name__ == "__main__":
    main()
