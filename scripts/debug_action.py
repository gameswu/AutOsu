#!/usr/bin/env python3
"""
Controller / vision diagnostics.

Two modes:
  live          — capture osu! window, run YOLO + box approach estimator +
                  deterministic controller, print per-frame diagnostics
                  (no input injected)
  approach-geo  — validate the *geometric* approach estimator (CV fallback)
                  against ground-truth timing by re-rendering replays

Usage::

    python scripts/debug_action.py live
    python scripts/debug_action.py approach-geo -d raw_data -s path/to/skin
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


# ── live mode ───────────────────────────────────────────────────────────

def cmd_live(args):
    """Capture live, run controller, print diagnostics (no injection)."""
    cfg = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            print(f"[Live] config not found: {cfg_path}")
            sys.exit(1)
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

    # Approach estimator (primary: YOLO ring boxes)
    from src.vision.approach_from_boxes import BoxApproachEstimator
    approach_estimator = BoxApproachEstimator()
    print("[Live] Approach estimator: YOLO box (approach_circle class)")

    # Deterministic controller
    from src.control import Controller
    controller = Controller(
        hit_window=cfg.get("hit_window", 0.90),
        tap_hold_ms=cfg.get("tap_hold_ms", 40.0),
        tap_refractory_ms=cfg.get("tap_refractory_ms", 70.0),
        slide_grace_ms=cfg.get("slide_grace_ms", 90.0),
        spin_grace_ms=cfg.get("spin_grace_ms", 120.0),
        motion_net_path=cfg.get("motion_net_path", None),
        device=device,
    )

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
    cur_ox, cur_oy = mapping.screen_to_osu(sx, sy)
    cur_ox = max(0, min(512, cur_ox))
    cur_oy = max(0, min(384, cur_oy))
    prev_time_ms = 0.0

    print(f"[Live] Initial cursor: screen({sx},{sy}) -> osu({cur_ox:.0f},{cur_oy:.0f})")
    print(f"\nPress Ctrl+C to stop.\n")

    hdr = (f"{'frame':>5} │ {'dets':>4} {'ring':>4} {'ball':>4} │ "
           f"{'cur_x':>6} {'cur_y':>6} {'dt_ms':>6} │ "
           f"{'state':>8} {'tgt_x':>6} {'tgt_y':>6} {'Z':>2} {'X':>2} │ "
           f"{'ar_top':>6}")
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

            # Estimate approach ratios from ring boxes (same as pipeline.py)
            approach_estimator.estimate(dets, t_ms=now_ms)

            # Read actual cursor
            sx, sy = get_cursor()
            cur_ox, cur_oy = mapping.screen_to_osu(sx, sy)
            cur_ox = max(0, min(512, cur_ox))
            cur_oy = max(0, min(384, cur_oy))

            dt = now_ms - prev_time_ms if prev_time_ms > 0 else 16.0

            # Run controller (does not inject — diagnostics only)
            out = controller.update(
                dets, (cur_ox, cur_oy), now_ms, mapping.model_to_osu,
            )

            if frame_idx < 30 or frame_idx % 30 == 0:
                ar_top = 0.0
                actionable = dets.actionable_objects
                if actionable:
                    ar_top = max(d.approach_ratio for d in actionable)
                n_ring = len(dets.approach_circles)
                n_ball = len(dets.slider_balls)
                print(f"{frame_idx:5d} │ {len(dets.detections):4d} {n_ring:4d} "
                      f"{n_ball:4d} │ {cur_ox:6.0f} {cur_oy:6.0f} {dt:6.1f} │ "
                      f"{controller.reference._state:>8} {out.x:6.0f} {out.y:6.0f} "
                      f"{'Z' if out.key_z else '·':>2} {'X' if out.key_x else '·':>2} │ "
                      f"{ar_top:6.3f}")
                if dt > 200:
                    print(f"       ⚠ dt={dt:.0f}ms SLOW")

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
        description="Controller / vision diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd")

    p_live = sub.add_parser("live", help="Live capture + controller diagnostics (no injection)")
    p_live.add_argument("--config", "-c", default=None,
                        help="Path to config YAML")
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
    if args.cmd == "live":
        cmd_live(args)
    elif args.cmd == "approach-geo":
        cmd_approach_geo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
