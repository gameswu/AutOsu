#!/usr/bin/env python3
"""
Train the attention-based trajectory model.

Supervised regression of the **full human cursor velocity** (osu!px/ms) on the
dataset built by ``scripts/build_motion_dataset.py``::

    loss = MSE(model(cursor_features, target_features, target_mask), velocity)

The model is :func:`src.control.motion_net.make_trajectory_model`.  Output is
raw (vx, vy) — no tanh normalisation; the velocity range is learned from data.

The saved checkpoint is a plain ``state_dict`` loadable by
:class:`src.control.motion_net.TrajectoryPolicy`.

Usage::

    python scripts/train_motion.py -d runs/motion/dataset.npz \
        -o runs/motion/trajectory.pt --epochs 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser(description="Train the trajectory model.")
    ap.add_argument("--config", "-c", default=None,
                    help="Path to config YAML")
    ap.add_argument("--dataset", "-d", default="runs/motion/dataset.npz",
                    help="dataset .npz from build_motion_dataset.py")
    ap.add_argument("--output", "-o", default="runs/motion/trajectory.pt",
                    help="output checkpoint (state_dict)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import yaml
    cfg: dict = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
            sys.exit(1)
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    mt = cfg.get("motion_training", {}) or {}

    # CLI > config > built-in defaults.
    dataset_path = (args.dataset if args.dataset != "runs/motion/dataset.npz"
                    else mt.get("dataset", "runs/motion/dataset.npz"))
    output_path = (args.output if args.output != "runs/motion/trajectory.pt"
                   else mt.get("output", "runs/motion/trajectory.pt"))
    epochs = args.epochs if args.epochs != 60 else mt.get("epochs", 60)
    batch_size = (args.batch_size if args.batch_size != 4096
                  else mt.get("batch_size", 4096))
    lr = args.lr if args.lr != 1e-3 else mt.get("lr", 1e-3)
    val_frac = args.val_frac if args.val_frac != 0.1 else mt.get("val_frac", 0.1)

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.control.motion_net import make_trajectory_model, CURSOR_DIM, TARGET_DIM

    data = np.load(dataset_path)
    cursor_feats = data["cursor_features"].astype("float32")
    target_feats = data["target_features"].astype("float32")
    target_masks = data["target_masks"]
    velocities = data["velocities"].astype("float32")

    if cursor_feats.shape[1] != CURSOR_DIM:
        print(f"ERROR: cursor_dim {cursor_feats.shape[1]} != CURSOR_DIM {CURSOR_DIM}",
              file=sys.stderr)
        sys.exit(1)
    if target_feats.shape[2] != TARGET_DIM:
        print(f"ERROR: target_dim {target_feats.shape[2]} != TARGET_DIM {TARGET_DIM}",
              file=sys.stderr)
        sys.exit(1)

    device = args.device or cfg.get("device", "cuda")
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"

    n = len(cursor_feats)
    vel_rms = float(np.sqrt(np.mean(velocities ** 2)))
    print(f"[train] device={device}  samples={n}  vel_rms={vel_rms:.4f} osu!px/ms")

    # Train / val split.
    idx = np.random.default_rng(0).permutation(n)
    n_val = int(n * val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    def _loader(ix, shuffle):
        ds = TensorDataset(
            torch.from_numpy(cursor_feats[ix]),
            torch.from_numpy(target_feats[ix]),
            torch.from_numpy(target_masks[ix]),
            torch.from_numpy(velocities[ix]),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    tr_loader = _loader(tr_idx, True)
    val_loader = _loader(val_idx, False) if n_val > 0 else None

    net = make_trajectory_model().to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[train] model params: {n_params:,}")

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        net.train()
        tr_loss = 0.0
        for cb, tb, mb, vb in tr_loader:
            cb = cb.to(device)
            tb = tb.to(device)
            mb = mb.to(device)
            vb = vb.to(device)
            opt.zero_grad()
            pred = net(cb, tb, mb)
            loss = loss_fn(pred, vb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(cb)
        tr_loss /= max(1, len(tr_idx))

        val_loss = tr_loss
        if val_loader is not None:
            net.eval()
            vl = 0.0
            with torch.no_grad():
                for cb, tb, mb, vb in val_loader:
                    cb = cb.to(device)
                    tb = tb.to(device)
                    mb = mb.to(device)
                    vb = vb.to(device)
                    vl += loss_fn(net(cb, tb, mb), vb).item() * len(cb)
            val_loss = vl / max(1, n_val)

        rms = val_loss ** 0.5
        print(f"  epoch {epoch:3d}  train={tr_loss:.6f}  "
              f"val={val_loss:.6f}  (~{rms:.4f} osu!px/ms RMS)")

        if val_loss <= best_val:
            best_val = val_loss
            torch.save(net.state_dict(), out)

    print(f"\nBest val MSE {best_val:.6f}  ({best_val**0.5:.4f} RMS)  ->  {out}")
    print("Set `motion_net_path` in your config to enable the learned trajectory model.")


if __name__ == "__main__":
    main()
