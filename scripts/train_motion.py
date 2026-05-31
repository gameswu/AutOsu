#!/usr/bin/env python3
"""
Train the neural motion policy.

Supervised regression of the human cursor *velocity* (osu!px/ms) toward the
navigation goal, on the dataset built by ``scripts/build_motion_dataset.py``::

    target = velocity / max_speed                      # in [-1, 1]
    loss   = MSE(net(features), target)

The net is the same tiny MLP the runtime uses
(:func:`src.control.motion_net.make_motion_net`, ``tanh`` output). The saved
checkpoint is a plain ``state_dict`` loadable by
:class:`src.control.motion_net.MotionPolicy`.

IMPORTANT: train with the **same** ``--max-speed`` baked into the dataset and
used by the runtime ``MotionPolicy`` (default 4.0 osu!px/ms), so the [-1, 1]
network output maps back to the same speed scale.

Usage::

    python scripts/train_motion.py --dataset runs/motion/dataset.npz \
        --output runs/motion/motion_net.pt --epochs 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src.xxx` works
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser(description="Train the neural motion policy.")
    ap.add_argument("--dataset", "-d", default="runs/motion/dataset.npz",
                    help="dataset .npz from build_motion_dataset.py")
    ap.add_argument("--output", "-o", default="runs/motion/motion_net.pt",
                    help="output checkpoint (state_dict)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-speed", type=float, default=None,
                    help="override; defaults to the value stored in the dataset")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.control.motion_net import make_motion_net, FEATURE_DIM

    data = np.load(args.dataset)
    feats = data["features"].astype("float32")
    vel = data["velocity"].astype("float32")
    if feats.shape[1] != FEATURE_DIM:
        print(f"ERROR: dataset feature dim {feats.shape[1]} != FEATURE_DIM "
              f"{FEATURE_DIM}", file=sys.stderr)
        sys.exit(1)
    max_speed = float(args.max_speed
                      if args.max_speed is not None
                      else data["max_speed"])
    targets = np.clip(vel / max_speed, -1.0, 1.0).astype("float32")

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}  samples={len(feats)}  "
          f"max_speed={max_speed} osu!px/ms")

    # Split
    n = len(feats)
    idx = np.random.default_rng(0).permutation(n)
    n_val = int(n * args.val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    def _loader(ix, shuffle):
        ds = TensorDataset(torch.from_numpy(feats[ix]),
                           torch.from_numpy(targets[ix]))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    tr_loader = _loader(tr_idx, True)
    val_loader = _loader(val_idx, False) if n_val > 0 else None

    net = make_motion_net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        net.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= max(1, len(tr_idx))

        val_loss = tr_loss
        if val_loader is not None:
            net.eval()
            vl = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    vl += loss_fn(net(xb), yb).item() * len(xb)
            val_loss = vl / max(1, n_val)

        # RMS velocity error for an interpretable number.
        rms = (val_loss ** 0.5) * max_speed
        print(f"  epoch {epoch:3d}  train={tr_loss:.5f}  "
              f"val={val_loss:.5f}  (~{rms:.3f} osu!px/ms RMS)")

        if val_loss <= best_val:
            best_val = val_loss
            torch.save(net.state_dict(), out)

    print(f"\nBest val MSE {best_val:.5f}  ->  {out}")
    print("Set `motion_net_path` (and matching max_speed_osu) in your config "
          "to enable the motion policy.")


if __name__ == "__main__":
    main()
