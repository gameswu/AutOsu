#!/usr/bin/env python3
"""
Train the attention-based motion model (multi-task).

The model (:func:`src.control.motion_net.make_trajectory_model`) emits, per
frame, residual weights ``w`` for the approach movement primitive and a velocity
``v`` for the slide/spin phases.  The loss is phase-routed by ``is_approach``::

    approach:  MSE( primitive(bnd, w)(path_tau), path_xy )   # cloned human path
    slide/spin: MSE( v, velocities )                          # human velocity

The minimum-jerk boundary guarantees on-target arrival; training only shapes the
residual style (approach) and the slide/spin velocity.  Output is raw — no tanh.

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
    from src.control.motion_net import (
        make_trajectory_model, primitive_path_torch,
        CURSOR_DIM, TARGET_DIM, N_BASIS,
    )

    data = np.load(dataset_path)
    cursor_feats = data["cursor_features"].astype("float32")
    target_feats = data["target_features"].astype("float32")
    target_masks = data["target_masks"]
    is_approach = data["is_approach"].astype("bool")
    bnd = data["bnd"].astype("float32")
    path_tau = data["path_tau"].astype("float32")
    path_xy = data["path_xy"].astype("float32")
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

    # Sanity: detect NaN / Inf in loaded data.
    for name, arr in [("cursor_features", cursor_feats),
                      ("target_features", target_feats),
                      ("bnd", bnd), ("path_xy", path_xy),
                      ("velocities", velocities)]:
        bad = ~np.isfinite(arr)
        if bad.any():
            n_bad = int(bad.any(axis=tuple(range(1, arr.ndim))).sum())
            print(f"WARNING: {name} has {bad.sum()} non-finite values "
                  f"in {n_bad} samples — replacing with 0")
            arr[bad] = 0.0

    # Check target mask: samples where no target is valid.
    no_target = ~target_masks.any(axis=-1)
    n_no_tgt = int(no_target.sum())
    if n_no_tgt > 0:
        print(f"[train] {n_no_tgt}/{n} samples have empty target masks "
              f"({100*n_no_tgt/n:.1f}%) — model uses idle_embed for these")

    n_ap = int(is_approach.sum())
    vel_rms = float(np.sqrt(np.mean(velocities[~is_approach] ** 2))) if n_ap < n else 0.0
    print(f"[train] device={device}  samples={n}  approach={n_ap}  "
          f"slide/spin={n - n_ap}  vel_rms={vel_rms:.4f} osu!px/ms")

    # Train / val split.
    idx = np.random.default_rng(0).permutation(n)
    n_val = int(n * val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    def _loader(ix, shuffle):
        ds = TensorDataset(
            torch.from_numpy(cursor_feats[ix]),
            torch.from_numpy(target_feats[ix]),
            torch.from_numpy(target_masks[ix]),
            torch.from_numpy(is_approach[ix]),
            torch.from_numpy(bnd[ix]),
            torch.from_numpy(path_tau[ix]),
            torch.from_numpy(path_xy[ix]),
            torch.from_numpy(velocities[ix]),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    tr_loader = _loader(tr_idx, True)
    val_loader = _loader(val_idx, False) if n_val > 0 else None

    net = make_trajectory_model().to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[train] model params: {n_params:,}")

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    K = N_BASIS

    def _batch_loss(cb, tb, mb, ab, bb, ptb, pxb, vb):
        """Phase-routed loss: approach path-MSE + slide/spin velocity-MSE."""
        out = net(cb, tb, mb)                 # (B, 2*K + 2)
        w = out[:, :2 * K]
        vel = out[:, 2 * K:]
        zero = out.sum() * 0.0
        # Approach: clone the human path via the primitive.
        if ab.any():
            x0 = bb[ab, 0:2]
            v0 = bb[ab, 2:4]
            g = bb[ab, 4:6]
            vg = bb[ab, 6:8]
            tau0 = bb[ab, 8:9]
            pred_path = primitive_path_torch(x0, v0, g, vg, tau0, w[ab], ptb[ab])
            loss_a = ((pred_path - pxb[ab]) ** 2).mean()
        else:
            loss_a = zero
        # Slide / spin: regress the human velocity.
        nb = ~ab
        loss_v = ((vel[nb] - vb[nb]) ** 2).mean() if nb.any() else zero
        return loss_a + loss_v, loss_a, loss_v

    best_val = float("inf")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        net.train()
        tr_loss = 0.0
        for batch in tr_loader:
            batch = [x.to(device) for x in batch]
            opt.zero_grad()
            loss, _, _ = _batch_loss(*batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            opt.step()
            tr_loss += loss.item() * len(batch[0])
        tr_loss /= max(1, len(tr_idx))

        val_loss = tr_loss
        val_a = val_v = 0.0
        if val_loader is not None:
            net.eval()
            vl = va = vv = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = [x.to(device) for x in batch]
                    loss, la, lv = _batch_loss(*batch)
                    nbt = len(batch[0])
                    vl += loss.item() * nbt
                    va += float(la) * nbt
                    vv += float(lv) * nbt
            val_loss = vl / max(1, n_val)
            val_a = va / max(1, n_val)
            val_v = vv / max(1, n_val)

        print(f"  epoch {epoch:3d}  train={tr_loss:.6f}  val={val_loss:.6f}  "
              f"(approach_path={val_a:.4f} px²  slide/spin_v={val_v:.6f})")

        if val_loss <= best_val:
            best_val = val_loss
            torch.save(net.state_dict(), out)

    print(f"\nBest val loss {best_val:.6f}  ->  {out}")
    print("Set `motion_net_path` in your config to enable the learned model.")


if __name__ == "__main__":
    main()
