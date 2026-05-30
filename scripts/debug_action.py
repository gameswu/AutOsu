#!/usr/bin/env python3
"""
Action model diagnostic — inspect state vectors, model outputs, and
training data statistics side-by-side.

Three modes:
  live    — capture osu! window, run YOLO + action model, print per-frame
            diagnostics (no input injected)
  replay  — load a training .npz sequence, feed it frame-by-frame through
            the model, compare predicted vs ground-truth actions
  stats   — scan all training sequences and print summary statistics
            (dx/dy range, time_delta distribution, key ratios, etc.)

Usage::

    python scripts/debug_action.py stats
    python scripts/debug_action.py stats --sequences dataset/sequences
    python scripts/debug_action.py replay --file dataset/sequences/seq_0000_00.npz
    python scripts/debug_action.py live
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import yaml


# ── stats mode ──────────────────────────────────────────────────────────

def cmd_stats(args):
    """Print summary statistics of the training data."""
    seq_dir = Path(args.sequences)
    npz_files = sorted(seq_dir.glob("seq_*.npz"))
    if not npz_files:
        print(f"No sequence files in {seq_dir}")
        return

    all_states, all_actions = [], []
    for f in npz_files:
        d = np.load(f)
        all_states.append(d["states"])
        all_actions.append(d["actions"])

    states = np.concatenate(all_states)   # (N, 133)
    actions = np.concatenate(all_actions)  # (N, 4)

    N = len(states)
    print(f"=== Training data: {len(npz_files)} files, {N} frames ===\n")

    # Action stats (actions are already normalised: dx/50, dy/50, key_z, key_x)
    dx_norm = actions[:, 0]
    dy_norm = actions[:, 1]
    key_z = actions[:, 2]
    key_x = actions[:, 3]

    dx_osu = dx_norm * 50.0
    dy_osu = dy_norm * 50.0

    print("── Actions ──")
    print(f"  dx (osu!px)  mean={dx_osu.mean():+.2f}  std={dx_osu.std():.2f}  "
          f"min={dx_osu.min():.1f}  max={dx_osu.max():.1f}  "
          f"|dx|>100: {(np.abs(dx_osu)>100).sum()}/{N}")
    print(f"  dy (osu!px)  mean={dy_osu.mean():+.2f}  std={dy_osu.std():.2f}  "
          f"min={dy_osu.min():.1f}  max={dy_osu.max():.1f}  "
          f"|dy|>100: {(np.abs(dy_osu)>100).sum()}/{N}")
    print(f"  key_z        mean={key_z.mean():.3f}  "
          f"pressed={key_z.sum():.0f}/{N} ({key_z.mean()*100:.1f}%)")
    print(f"  key_x        mean={key_x.mean():.3f}  "
          f"pressed={key_x.sum():.0f}/{N} ({key_x.mean()*100:.1f}%)")

    # State: last 5 dims are [cursor_x_n, cursor_y_n, vx_n, vy_n, dt_n]
    cursor_x_n = states[:, -5]
    cursor_y_n = states[:, -4]
    vx_n = states[:, -3]
    vy_n = states[:, -2]
    dt_n = states[:, -1]

    cursor_x = cursor_x_n * 512
    cursor_y = cursor_y_n * 384
    vx = vx_n * 10
    vy = vy_n * 10
    dt_ms = dt_n * 50

    print("\n── State (cursor + timing) ──")
    print(f"  cursor_x     mean={cursor_x.mean():.1f}  std={cursor_x.std():.1f}  "
          f"range=[{cursor_x.min():.1f}, {cursor_x.max():.1f}]")
    print(f"  cursor_y     mean={cursor_y.mean():.1f}  std={cursor_y.std():.1f}  "
          f"range=[{cursor_y.min():.1f}, {cursor_y.max():.1f}]")
    print(f"  vx (px/ms)   mean={vx.mean():.4f}  std={vx.std():.4f}")
    print(f"  vy (px/ms)   mean={vy.mean():.4f}  std={vy.std():.4f}")
    print(f"  dt_ms        mean={dt_ms.mean():.1f}  std={dt_ms.std():.1f}  "
          f"range=[{dt_ms.min():.1f}, {dt_ms.max():.1f}]")
    if dt_ms.mean() > 0:
        print(f"               ≈ {1000.0/dt_ms.mean():.1f} fps effective training rate")

    # Object slot occupancy
    obj_block = states[:, :-5].reshape(N, 16, 8)  # (N, 16, 8)
    slot_nonzero = (obj_block.sum(axis=2) != 0).sum(axis=1)  # per frame
    print(f"\n── Objects per frame ──")
    print(f"  mean={slot_nonzero.mean():.1f}  max={slot_nonzero.max()}  "
          f"zero-obj frames={int((slot_nonzero==0).sum())}/{N}")

    print()


# ── replay mode ─────────────────────────────────────────────────────────

def cmd_replay(args):
    """Feed a training sequence through the model, compare predictions."""
    import torch
    from src.action.model import ActionModelInference
    from src.action.state import ACTION_DIM

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Action model not found: {model_path}")
        return

    npz_path = Path(args.file)
    if not npz_path.exists():
        print(f"Sequence file not found: {npz_path}")
        return

    data = np.load(npz_path)
    states = data["states"]   # (T, 133)
    actions = data["actions"]  # (T, 4)
    T = len(states)
    print(f"Sequence: {npz_path.name}  length={T}")

    model = ActionModelInference(model_path, device=args.device)
    model.reset()

    print(f"\n{'frame':>6} │ {'gt_dx':>7} {'gt_dy':>7} {'gt_z':>5} {'gt_x':>5} │ "
          f"{'pred_dx':>8} {'pred_dy':>8} {'pred_z':>6} {'pred_x':>6} │ "
          f"{'err_dx':>7} {'err_dy':>7}")
    print("─" * 100)

    errs_dx, errs_dy = [], []
    for t in range(min(T, args.frames)):
        raw = model.predict(states[t])  # [dx_n, dy_n, key_z_prob, key_x_prob]

        pred_dx = raw[0] * 50.0
        pred_dy = raw[1] * 50.0
        pred_z = raw[2]
        pred_x = raw[3]

        gt_dx = actions[t, 0] * 50.0
        gt_dy = actions[t, 1] * 50.0
        gt_z = actions[t, 2]
        gt_x = actions[t, 3]

        err_dx = pred_dx - gt_dx
        err_dy = pred_dy - gt_dy
        errs_dx.append(err_dx)
        errs_dy.append(err_dy)

        if t < 20 or t % 50 == 0 or abs(err_dx) > 30 or abs(err_dy) > 30:
            print(f"{t:6d} │ {gt_dx:+7.1f} {gt_dy:+7.1f} {gt_z:5.2f} {gt_x:5.2f} │ "
                  f"{pred_dx:+8.1f} {pred_dy:+8.1f} {pred_z:6.3f} {pred_x:6.3f} │ "
                  f"{err_dx:+7.1f} {err_dy:+7.1f}")

    errs_dx = np.array(errs_dx)
    errs_dy = np.array(errs_dy)
    n = len(errs_dx)
    print(f"\n── Summary ({n} frames) ──")
    print(f"  MAE dx={np.abs(errs_dx).mean():.1f}  dy={np.abs(errs_dy).mean():.1f}  "
          f"combined={np.sqrt(errs_dx**2 + errs_dy**2).mean():.1f} osu!px")
    print(f"  Max |err| dx={np.abs(errs_dx).max():.1f}  dy={np.abs(errs_dy).max():.1f}")


# ── live mode ───────────────────────────────────────────────────────────

def cmd_live(args):
    """Capture live, run model, print diagnostics (no injection)."""
    import torch

    cfg_path = Path(args.config)
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    device = args.device or cfg.get("device", "cuda:0")
    model_w = cfg.get("model_input_w", 640)
    model_h = cfg.get("model_input_h", 384)

    # Window
    from src.runtime.window import get_playfield_mapping
    mapping = get_playfield_mapping(model_input_w=model_w, model_input_h=model_h)
    print(f"[Live] Window: {mapping.client_w}x{mapping.client_h}  "
          f"playfield: ({mapping.playfield_x},{mapping.playfield_y})")

    # Capture
    from src.runtime.capture import ScreenCapture
    capture = ScreenCapture(
        region=mapping.capture_region,
        target_size=(model_w, model_h),
        use_dxcam=cfg.get("use_dxcam", True),
    )

    # Detector
    from src.vision.detector import Detector
    det_path = Path(cfg.get("detector_path", "runs/detect/train/weights/best.pt"))
    detector = Detector(det_path, device=device,
                        conf_threshold=cfg.get("conf_threshold", 0.3),
                        iou_threshold=cfg.get("iou_threshold", 0.45),
                        imgsz=(model_h, model_w))
    detector.load()

    # Approach estimator (geometric CV — no model)
    from src.vision.approach_geometry import GeometricApproachEstimator
    approach_estimator = GeometricApproachEstimator()
    print("[Live] Approach estimator: geometric CV (no model)")

    # Action model
    from src.action.model import ActionModelInference
    from src.action.state import GameStateVector, ObjectState, ActionVector
    act_path = Path(cfg.get("action_model_path", "runs/action/best.pth"))
    action_model = None
    if act_path.exists():
        action_model = ActionModelInference(act_path, device=device)
        action_model.reset()
        print("[Live] Action model loaded")
    else:
        print("[Live] No action model — detection diagnostics only")

    # Cursor reader
    try:
        import ctypes, ctypes.wintypes as wt
        _u32 = ctypes.windll.user32
        def get_cursor():
            pt = wt.POINT()
            _u32.GetCursorPos(ctypes.byref(pt))
            return pt.x, pt.y
    except Exception:
        def get_cursor():
            return 0, 0

    # Init cursor from actual position
    sx, sy = get_cursor()
    prev_cx, prev_cy = mapping.screen_to_osu(sx, sy)
    prev_cx = max(0, min(512, prev_cx))
    prev_cy = max(0, min(384, prev_cy))
    prev_prev_cx, prev_prev_cy = prev_cx, prev_cy
    prev_time_ms = 0.0

    print(f"[Live] Initial cursor: screen({sx},{sy}) -> osu({prev_cx:.0f},{prev_cy:.0f})")
    print(f"\nPress Ctrl+C to stop.\n")

    hdr = (f"{'frame':>5} │ {'dets':>4} │ "
           f"{'cur_x':>6} {'cur_y':>6} {'vx':>7} {'vy':>7} {'dt_ms':>6} │ "
           f"{'pred_dx':>8} {'pred_dy':>8} {'p(z)':>5} {'p(x)':>5} │ "
           f"{'new_x':>6} {'new_y':>6} │ {'ar_top':>6}")
    print(hdr)
    print("─" * len(hdr))

    frame_idx = 0
    try:
        while True:
            t0 = time.perf_counter()
            now_ms = t0 * 1000.0

            frame = capture.grab()
            if frame is None:
                time.sleep(0.001)
                continue

            dets = detector.detect(frame, timestamp_ms=now_ms)

            # Estimate approach ratios (same as pipeline.py)
            actionable = dets.actionable_objects
            if actionable:
                approach_estimator.estimate(frame, actionable)

            # Read actual cursor
            sx, sy = get_cursor()
            cur_ox, cur_oy = mapping.screen_to_osu(sx, sy)
            cur_ox = max(0, min(512, cur_ox))
            cur_oy = max(0, min(384, cur_oy))

            dt = now_ms - prev_time_ms if prev_time_ms > 0 else 16.0

            # Build state
            objects = []
            for d in dets.detections:
                ox, oy = mapping.model_to_osu(d.cx, d.cy)
                objects.append(ObjectState(
                    class_id=int(d.cls), x=ox, y=oy,
                    approach_ratio=d.approach_ratio))

            if dt > 0 and prev_time_ms > 0:
                vx = (cur_ox - prev_prev_cx) / dt
                vy = (cur_oy - prev_prev_cy) / dt
            else:
                vx, vy = 0.0, 0.0

            state = GameStateVector(
                objects=objects,
                cursor_x=cur_ox, cursor_y=cur_oy,
                cursor_vx=vx, cursor_vy=vy,
                time_delta_ms=dt,
            )
            state_np = state.to_numpy()

            # Run model
            pred_dx, pred_dy, pred_z, pred_x = 0.0, 0.0, 0.0, 0.0
            new_x, new_y = cur_ox, cur_oy
            if action_model is not None:
                raw = action_model.predict(state_np)
                act = ActionVector.from_numpy(raw)
                pred_dx, pred_dy = act.dx, act.dy
                pred_z, pred_x = act.key_z, act.key_x
                new_x = max(0, min(512, cur_ox + pred_dx))
                new_y = max(0, min(384, cur_oy + pred_dy))

            # Print (first 30 frames, then every 30th)
            if frame_idx < 30 or frame_idx % 30 == 0:
                # Show approach_ratio of the highest-ratio object (slot 0 in state)
                ar_top = 0.0
                if objects:
                    ar_top = max(o.approach_ratio for o in objects)
                print(f"{frame_idx:5d} │ {len(dets.detections):4d} │ "
                      f"{cur_ox:6.0f} {cur_oy:6.0f} {vx:+7.3f} {vy:+7.3f} {dt:6.1f} │ "
                      f"{pred_dx:+8.1f} {pred_dy:+8.1f} {pred_z:5.2f} {pred_x:5.2f} │ "
                      f"{new_x:6.0f} {new_y:6.0f} │ {ar_top:6.3f}")

                # Flag anomalies
                warnings = []
                if abs(pred_dx) > 100:
                    warnings.append(f"dx={pred_dx:+.0f} HUGE")
                if abs(pred_dy) > 100:
                    warnings.append(f"dy={pred_dy:+.0f} HUGE")
                if dt > 200:
                    warnings.append(f"dt={dt:.0f}ms SLOW")
                if warnings:
                    print(f"       ⚠ {', '.join(warnings)}")

            prev_prev_cx, prev_prev_cy = cur_ox, cur_oy
            prev_time_ms = now_ms
            frame_idx += 1

            # Rate limit ~60fps
            elapsed = time.perf_counter() - t0
            sleep = 1.0 / 60 - elapsed
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        pass

    capture.stop()
    print(f"\n[Live] {frame_idx} frames captured.")


# ── approach-geo mode ─────────────────────────────────────────────────────

def cmd_approach_geo(args):
    """
    Validate the geometric approach estimator against ground-truth timing
    ratios, by re-rendering (beatmap, replay) pairs with no background and
    measuring each object's approach ring directly from the rendered frame.
    """
    import sys
    # scripts/ dir on path so we can reuse the render helpers
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_dataset import open_replay_frames, _generate_labels, _obj_kind
    from src.data.replay_parser import find_replay_pairs
    from src.data.renderer import PlayfieldTransform
    from src.vision.approach_geometry import GeometricApproachEstimator
    from src.vision.detector import Detection, ObjClass

    pairs = find_replay_pairs(args.data)
    if not pairs:
        print(f"No (beatmap, replay) pairs found in {args.data}")
        return

    W, H = args.width, args.height
    tf = PlayfieldTransform(W, H)
    estimator = GeometricApproachEstimator(temporal=not args.no_temporal)
    mode = "per-frame only" if args.no_temporal else "temporal linear-fit"
    print(f"Estimator mode: {mode}")

    def gt_ratio_by_pos(cx, cy, visible, t_ms, time_preempt):
        """Ground-truth timing ratio of the visible object closest to (cx, cy)."""
        best, best_d = None, 1e18
        for obj in visible:
            if _obj_kind(obj) not in ("circle", "slider"):
                continue
            rx, ry = tf.osu_to_render(obj["x"], obj["y"])
            d = (rx - cx) ** 2 + (ry - cy) ** 2
            if d < best_d:
                best_d, best = d, obj
        if best is None or time_preempt <= 0:
            return None
        dt = t_ms - (best["time"] - time_preempt)
        return min(1.0, max(0.0, dt / time_preempt))

    gts, preds = [], []
    frames_done = 0
    print(f"Rendering up to {args.frames} frames (no background)…")
    for osu_path, osr_paths in pairs:
        for osr_path in osr_paths:
            try:
                _, frame_iter = open_replay_frames(
                    osu_path, osr_path, args.skin, W, H, args.fps,
                )
            except Exception as e:
                print(f"  skip {osr_path.name}: {e}")
                continue
            estimator.reset()  # tracks must not carry across replays
            for fd in frame_iter:
                frame, visible = fd["frame"], fd["visible"]
                if visible:
                    labels = _generate_labels(
                        visible, fd["t_ms"], fd["time_preempt"],
                        fd["radius_osu"], tf, W, H,
                    )
                    # Build lightweight detections for actionable classes
                    dets, dets_gt = [], []
                    for line in labels:
                        parts = line.split()
                        cls_id = int(parts[0])
                        if cls_id not in (0, 1):  # hitcircle, slider_head
                            continue
                        cx = float(parts[1]) * W
                        cy = float(parts[2]) * H
                        bw = float(parts[3]) * W
                        bh = float(parts[4]) * H
                        gt = gt_ratio_by_pos(
                            cx, cy, visible, fd["t_ms"], fd["time_preempt"],
                        )
                        if gt is None:
                            continue
                        dets.append(Detection(
                            cls=ObjClass(cls_id), confidence=1.0,
                            cx=cx, cy=cy, w=bw, h=bh,
                        ))
                        dets_gt.append(gt)
                    if dets:
                        estimator.estimate(frame, dets, t_ms=fd["t_ms"])
                        for d, gt in zip(dets, dets_gt):
                            gts.append(gt)
                            preds.append(d.approach_ratio)
                frames_done += 1
                if frames_done % 200 == 0:
                    print(f"  …{frames_done} frames, {len(gts)} measurements")
                if frames_done >= args.frames:
                    break
            if frames_done >= args.frames:
                break
        if frames_done >= args.frames:
            break

    if not gts:
        print("No measurements collected.")
        return

    gts = np.array(gts)
    preds = np.array(preds)
    errs = np.abs(preds - gts)
    n = len(errs)

    print(f"\n=== Geometric Approach Validation ({n} measurements) ===")
    print(f"  GT range:    [{gts.min():.4f}, {gts.max():.4f}]")
    print(f"  Pred range:  [{preds.min():.4f}, {preds.max():.4f}]")
    print(f"  MAE:         {errs.mean():.4f}")
    print(f"  Median err:  {np.median(errs):.4f}")
    print(f"  Max err:     {errs.max():.4f}")

    buckets = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    print(f"\n  Bucket MAE (by GT ratio):")
    for lo, hi in buckets:
        mask = (gts >= lo) & (gts < hi)
        if mask.sum() > 0:
            print(f"    [{lo:.1f}, {hi:.1f}):  n={mask.sum():5d}  "
                  f"MAE={errs[mask].mean():.4f}  "
                  f"pred_mean={preds[mask].mean():.4f}")


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Action model diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd")

    p_stats = sub.add_parser("stats", help="Training data statistics")
    p_stats.add_argument("--sequences", default="dataset/sequences")

    p_replay = sub.add_parser("replay", help="Compare model vs ground truth on training data")
    p_replay.add_argument("--file", required=True, help=".npz sequence file")
    p_replay.add_argument("--model", default="runs/action/best.pth")
    p_replay.add_argument("--device", default="cuda:0")
    p_replay.add_argument("--frames", type=int, default=500)

    p_live = sub.add_parser("live", help="Live capture diagnostics (no injection)")
    p_live.add_argument("--config", "-c", default="configs/default.yaml")
    p_live.add_argument("--device", default=None)

    p_geo = sub.add_parser("approach-geo",
                           help="Validate geometric approach estimator vs timing GT (re-renders)")
    p_geo.add_argument("--data", "-d", required=True,
                       help="raw_data dir (beatmaps/ + replays/)")
    p_geo.add_argument("--skin", "-s", required=True, help="osu! skin directory")
    p_geo.add_argument("--width", type=int, default=640)
    p_geo.add_argument("--height", type=int, default=384)
    p_geo.add_argument("--fps", type=int, default=30)
    p_geo.add_argument("--frames", type=int, default=1000,
                       help="Max frames to render across pairs")
    p_geo.add_argument("--no-temporal", action="store_true",
                       help="Disable temporal linear-fit (pure per-frame geometry)")

    args = parser.parse_args()
    if args.cmd == "stats":
        cmd_stats(args)
    elif args.cmd == "replay":
        cmd_replay(args)
    elif args.cmd == "live":
        cmd_live(args)
    elif args.cmd == "approach-geo":
        cmd_approach_geo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
