#!/usr/bin/env python3
"""
Offline "fake inference" demo — watch the controller play without launching osu!.

Instead of capturing the live game, this reuses the data-synthesis renderer to
draw a real beatmap frame-by-frame, then runs the *actual* runtime stack on each
frame::

     rendered frame  ->  YOLO detect  ->  approach ratio (from ring boxes)
                    ->  Controller (trajectory model + keys)  ->  (cursor x/y, Z, X)

The controller's commanded cursor + key presses are overlaid on the frame and
written to an annotated MP4. The replay's human cursor (drawn by the renderer)
is kept as a ground-truth reference.

Usage (run via the Windows uv, where torch/CUDA live)::

    uv run python scripts/demo_inference.py \
        --data raw_data \
        --skin "C:/Users/ASUS/AppData/Local/osu!/Skins/WhiteCat - Selyu v2.3" \
        --output demo.mp4 --frames 1200 --fps 30 --scale 2
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

# Project + lib on path (mirrors generate_dataset.py)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_LIB_DIR = str(Path(__file__).resolve().parent.parent / "lib")
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)
# scripts/ on path for the shared render generator
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np


# Colours (BGR)
COL_BOX = (0, 220, 220)       # yellow detection boxes
COL_RING = (200, 160, 0)      # approach-circle boxes
COL_BALL = (255, 0, 200)      # slider-ball boxes
COL_MODEL = (60, 60, 255)     # red controller cursor
COL_GT = (80, 230, 80)        # green human (replay) cursor
COL_HUD = (255, 255, 255)
COL_KEY_ON = (60, 220, 60)
COL_KEY_OFF = (90, 90, 90)

_CLS_SHORT = {0: "C", 1: "SH", 2: "BALL", 3: "SP", 4: "AC", 5: "BODY"}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", "-c", default=None,
                   help="Path to config YAML (shares controller params with runtime)")
    p.add_argument("--data", "-d", default="raw_data",
                   help="raw_data dir (beatmaps/ + replays/)")
    p.add_argument("--index", type=int, default=0,
                   help="Which matched (beatmap, replay) pair to use")
    p.add_argument("--skin", "-s", required=True, help="osu! skin directory")
    p.add_argument("--detector", default="runs/detect/train/weights/best.pt")
    p.add_argument("--output", "-o", default="demo.mp4")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--fps", type=int, default=30,
                   help="Render/inference fps")
    p.add_argument("--frames", type=int, default=1200, help="Max frames to render")
    p.add_argument("--scale", type=int, default=2, help="Output upscale factor")
    p.add_argument("--conf", type=float, default=0.3, help="Detector confidence")
    p.add_argument("--iou", type=float, default=0.45)
    args = p.parse_args()

    from generate_dataset import open_replay_frames
    from src.data.replay_parser import find_replay_pairs
    from src.data.renderer import PlayfieldTransform
    from src.vision.detector import Detector, ObjClass
    from src.vision.approach_from_boxes import BoxApproachEstimator
    from src.control import Controller

    # Load config
    import yaml
    cfg: dict = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
            sys.exit(1)
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    device = args.device or cfg.get("device", "cuda:0")

    W, H, S = args.width, args.height, max(1, args.scale)
    tf = PlayfieldTransform(W, H)
    to_osu = tf.render_to_osu

    # ── Pick a beatmap/replay pair ────────────────────────────────────────
    pairs = find_replay_pairs(args.data)
    if not pairs:
        print(f"ERROR: no (beatmap, replay) pairs in {args.data}", file=sys.stderr)
        sys.exit(1)
    if args.index >= len(pairs):
        print(f"ERROR: --index {args.index} out of range ({len(pairs)} pairs)",
              file=sys.stderr)
        sys.exit(1)
    osu_path, osr_paths = pairs[args.index]
    osr_path = osr_paths[0]
    print(f"Beatmap: {osu_path.parent.name}/{osu_path.name}")
    print(f"Replay:  {osr_path.name}")

    # ── Load models ───────────────────────────────────────────────────────
    det_path = Path(args.detector or cfg.get("detector_path",
                    "runs/detect/train/weights/best.pt"))
    if not det_path.exists():
        print(f"ERROR: detector not found: {det_path}", file=sys.stderr)
        sys.exit(1)
    detector = Detector(det_path, device=device,
                        conf_threshold=args.conf,
                        iou_threshold=args.iou, imgsz=(H, W))
    detector.load()
    print(f"Detector: {det_path.name}")

    approach = BoxApproachEstimator()
    approach.reset()
    controller = Controller(
        hit_window=cfg.get("hit_window", 0.90),
        tap_hold_ms=cfg.get("tap_hold_ms", 40.0),
        tap_refractory_ms=cfg.get("tap_refractory_ms", 70.0),
        slide_grace_ms=cfg.get("slide_grace_ms", 90.0),
        spin_grace_ms=cfg.get("spin_grace_ms", 120.0),
        path_noise=cfg.get("path_noise", 0.18),
        noise_smooth=cfg.get("noise_smooth", 0.90),
        spin_radius=cfg.get("spin_radius", 60.0),
        spin_speed=cfg.get("spin_speed", 0.025),
    )

    # ── Video writer ──────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps), (W * S, H * S),
    )
    if not writer.isOpened():
        print(f"ERROR: could not open video writer for {out_path}", file=sys.stderr)
        sys.exit(1)

    # ── Inference state ───────────────────────────────────────────────────
    cur_x, cur_y = 256.0, 192.0          # controller cursor (osu px)
    initialised = False
    trail = deque(maxlen=18)             # controller cursor trail (canvas px)

    n_z = n_x = 0
    written = 0

    _, frame_iter = open_replay_frames(
        osu_path, osr_path, args.skin, W, H, args.fps,
    )

    for fd in frame_iter:
        frame = fd["frame"]
        t_ms = fd["t_ms"]
        gt_cx, gt_cy = fd["cursor_x"], fd["cursor_y"]

        if not initialised:
            cur_x, cur_y = gt_cx, gt_cy   # fair start point
            controller.reset(cursor=(cur_x, cur_y))
            initialised = True

        # 1) Detect  2) approach ratio from ring boxes  3) controller
        dets = detector.detect(frame, timestamp_ms=t_ms)
        approach.estimate(dets, t_ms=t_ms)
        out = controller.update(dets, (cur_x, cur_y), t_ms, to_osu)

        cur_x = min(512.0, max(0.0, out.x))
        cur_y = min(384.0, max(0.0, out.y))
        if out.key_z:
            n_z += 1
        if out.key_x:
            n_x += 1

        # 4) Draw
        canvas = cv2.resize(frame, (W * S, H * S), interpolation=cv2.INTER_NEAREST)
        _draw_overlay(canvas, S, tf, dets, cur_x, cur_y, gt_cx, gt_cy,
                      trail, out, fd["frame_idx"], t_ms)
        writer.write(canvas)

        written += 1
        if written % 100 == 0:
            print(f"  ...{written} frames")
        if written >= args.frames:
            break

    writer.release()
    print(f"\nWrote {written} frames -> {out_path.resolve()}")
    if written:
        print(f"Key presses: Z {n_z} ({100*n_z/written:.0f}%)  "
              f"X {n_x} ({100*n_x/written:.0f}%)")


def _draw_overlay(canvas, S, tf, dets, cur_x, cur_y, gt_cx, gt_cy,
                  trail, out, frame_idx, t_ms):
    """Annotate one (already-upscaled) canvas frame in place."""
    from src.vision.detector import ObjClass

    for d in dets.detections:
        x1, y1 = int(d.x1 * S), int(d.y1 * S)
        x2, y2 = int(d.x2 * S), int(d.y2 * S)
        if d.cls == ObjClass.APPROACH_CIRCLE:
            col = COL_RING
        elif d.cls == ObjClass.SLIDER_BALL:
            col = COL_BALL
        else:
            col = COL_BOX
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 1)
        tag = _CLS_SHORT.get(int(d.cls), "?")
        if d.cls in (ObjClass.HITCIRCLE, ObjClass.SLIDER_HEAD):
            tag += f" {d.approach_ratio:.2f}"
        cv2.putText(canvas, tag, (x1, max(8, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)

    # Human (replay) cursor reference — green crosshair
    gx, gy = tf.osu_to_render(gt_cx, gt_cy)
    gx, gy = int(gx * S), int(gy * S)
    cv2.drawMarker(canvas, (gx, gy), COL_GT, cv2.MARKER_CROSS, 14, 2)

    # Controller cursor — red filled dot + white ring + trail
    mx, my = tf.osu_to_render(cur_x, cur_y)
    mx, my = int(mx * S), int(my * S)
    trail.append((mx, my))
    for i in range(1, len(trail)):
        cv2.line(canvas, trail[i - 1], trail[i], COL_MODEL, 1, cv2.LINE_AA)
    cv2.circle(canvas, (mx, my), 6, COL_MODEL, -1, cv2.LINE_AA)
    cv2.circle(canvas, (mx, my), 8, (255, 255, 255), 1, cv2.LINE_AA)

    # HUD bar
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    hud = f"f{frame_idx}  t={t_ms:.0f}ms  dets={len(dets.detections)}"
    cv2.putText(canvas, hud, (6, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                COL_HUD, 1, cv2.LINE_AA)
    info = f"target=({out.x:.0f},{out.y:.0f})  Z={int(out.key_z)} X={int(out.key_x)}"
    cv2.putText(canvas, info, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                COL_HUD, 1, cv2.LINE_AA)

    # Z / X key lamps (top-right)
    for i, (lbl, on) in enumerate((("Z", out.key_z), ("X", out.key_x))):
        bx = canvas.shape[1] - 60 + i * 28
        cv2.rectangle(canvas, (bx, 6), (bx + 22, 28),
                      COL_KEY_ON if on else COL_KEY_OFF, -1)
        cv2.putText(canvas, lbl, (bx + 5, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0) if on else COL_HUD, 1, cv2.LINE_AA)

    # Legend (bottom)
    yb = canvas.shape[0] - 8
    cv2.putText(canvas, "green=human  red=controller", (6, yb),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_HUD, 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()
