#!/usr/bin/env python3
"""
Offline "fake inference" demo — watch the model play without launching osu!.

Instead of capturing the live game, this reuses the data-synthesis renderer to
draw a real beatmap frame-by-frame, then runs the *actual* runtime stack on each
frame:

    rendered frame  →  YOLO detect  →  geometric approach ratio
                    →  GameStateVector  →  GRU action model  →  (dx, dy, z, x)

The model's predicted cursor + key presses are overlaid on the frame and written
to an annotated MP4. The replay's human cursor (drawn by the renderer) is kept as
a ground-truth reference.

Two modes:
  closed-loop (default): the model drives its own cursor autoregressively — this
      is the true autonomous behaviour (and shows behavioural-cloning drift).
  --teacher-forcing:     the model is fed the human cursor each step and we plot
      its predicted next position — isolates per-step accuracy from drift.

Usage (run via the Windows uv, where torch/CUDA live):
    uv run python scripts/demo_inference.py \
        --data raw_data \
        --skin "C:/Users/ASUS/AppData/Local/osu!/Skins/WhiteCat - Selyu v2.3" \
        --output demo.mp4 --frames 1200 --fps 15 --scale 2
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
COL_MODEL = (60, 60, 255)     # red model cursor
COL_GT = (80, 230, 80)        # green human (replay) cursor
COL_HUD = (255, 255, 255)
COL_KEY_ON = (60, 220, 60)
COL_KEY_OFF = (90, 90, 90)

_CLS_SHORT = {0: "C", 1: "SH", 2: "SB", 3: "SE", 4: "SP"}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", "-d", default="raw_data",
                   help="raw_data dir (beatmaps/ + replays/)")
    p.add_argument("--index", type=int, default=0,
                   help="Which matched (beatmap, replay) pair to use")
    p.add_argument("--skin", "-s", required=True, help="osu! skin directory")
    p.add_argument("--detector", default="runs/detect/train/weights/best.pt")
    p.add_argument("--action-model", default="runs/action/best.pth")
    p.add_argument("--output", "-o", default="demo.mp4")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--fps", type=int, default=15,
                   help="Render/inference fps — should match training fps")
    p.add_argument("--frames", type=int, default=1200, help="Max frames to render")
    p.add_argument("--scale", type=int, default=2, help="Output upscale factor")
    p.add_argument("--conf", type=float, default=0.3, help="Detector confidence")
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--key-threshold", type=float, default=0.5)
    p.add_argument("--teacher-forcing", action="store_true",
                   help="Feed the human cursor each step (open-loop) instead of "
                        "letting the model drive itself")
    args = p.parse_args()

    from generate_dataset import open_replay_frames
    from src.data.replay_parser import find_replay_pairs
    from src.data.renderer import PlayfieldTransform
    from src.vision.detector import Detector, ObjClass
    from src.vision.approach_geometry import GeometricApproachEstimator
    from src.action.model import ActionModelInference
    from src.action.state import GameStateVector, ObjectState, ActionVector

    W, H, S = args.width, args.height, max(1, args.scale)
    tf = PlayfieldTransform(W, H)

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
    det_path = Path(args.detector)
    if not det_path.exists():
        print(f"ERROR: detector not found: {det_path}", file=sys.stderr)
        sys.exit(1)
    detector = Detector(det_path, device=args.device, conf_threshold=args.conf,
                        iou_threshold=args.iou, imgsz=(H, W))
    detector.load()
    print(f"Detector: {det_path.name}")

    approach = GeometricApproachEstimator()
    approach.reset()

    action_path = Path(args.action_model)
    action_model = None
    if action_path.exists():
        action_model = ActionModelInference(action_path, device=args.device)
        action_model.reset()
        print(f"Action model: {action_path.name}")
    else:
        print("WARNING: action model not found — overlaying detections only")

    mode = "teacher-forcing (open-loop)" if args.teacher_forcing else "closed-loop"
    print(f"Mode: {mode}")

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
    cur_x, cur_y = 256.0, 192.0          # model cursor (osu px)
    prev_x, prev_y = cur_x, cur_y         # for velocity
    prev_t = None
    initialised = False
    trail = deque(maxlen=18)              # model cursor trail (canvas px)

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
            # Start the model cursor where the human starts (fair start point)
            cur_x, cur_y = gt_cx, gt_cy
            prev_x, prev_y = gt_cx, gt_cy
            initialised = True

        # 1) Detect
        dets = detector.detect(frame, timestamp_ms=t_ms)
        actionable = dets.actionable_objects
        # 2) Approach ratio (temporal geometric)
        if actionable:
            approach.estimate(frame, actionable, t_ms=t_ms)

        # 3) Build state + 4) run action model
        dt = (t_ms - prev_t) if prev_t is not None else (1000.0 / args.fps)
        dt = max(1e-3, dt)
        pred = None
        if action_model is not None:
            src_x = gt_cx if args.teacher_forcing else cur_x
            src_y = gt_cy if args.teacher_forcing else cur_y
            vx = (cur_x - prev_x) / dt
            vy = (cur_y - prev_y) / dt
            objects = []
            for d in dets.detections:
                ox, oy = tf.render_to_osu(d.cx, d.cy)
                objects.append(ObjectState(class_id=int(d.cls), x=ox, y=oy,
                                           approach_ratio=d.approach_ratio))
            state = GameStateVector(
                objects=objects, cursor_x=src_x, cursor_y=src_y,
                cursor_vx=vx, cursor_vy=vy, time_delta_ms=dt,
            )
            action = ActionVector.from_numpy(action_model.predict(state.to_numpy()))

            # Advance the model cursor
            base_x = gt_cx if args.teacher_forcing else cur_x
            base_y = gt_cy if args.teacher_forcing else cur_y
            new_x = min(512.0, max(0.0, base_x + action.dx))
            new_y = min(384.0, max(0.0, base_y + action.dy))
            prev_x, prev_y = cur_x, cur_y
            cur_x, cur_y = new_x, new_y
            pred = action
            if action.key_z > args.key_threshold:
                n_z += 1
            if action.key_x > args.key_threshold:
                n_x += 1

        prev_t = t_ms

        # 5) Draw
        canvas = cv2.resize(frame, (W * S, H * S), interpolation=cv2.INTER_NEAREST)
        _draw_overlay(canvas, S, tf, dets, actionable, ObjClass,
                      cur_x, cur_y, gt_cx, gt_cy, trail, pred,
                      args.key_threshold, fd["frame_idx"], t_ms, mode)
        writer.write(canvas)

        written += 1
        if written % 100 == 0:
            print(f"  …{written} frames")
        if written >= args.frames:
            break

    writer.release()
    print(f"\nWrote {written} frames → {out_path.resolve()}")
    if action_model is not None and written:
        print(f"Key presses: Z {n_z} ({100*n_z/written:.0f}%)  "
              f"X {n_x} ({100*n_x/written:.0f}%)")


def _draw_overlay(canvas, S, tf, dets, actionable, ObjClass,
                  cur_x, cur_y, gt_cx, gt_cy, trail, pred,
                  key_thr, frame_idx, t_ms, mode):
    """Annotate one (already-upscaled) canvas frame in place."""
    # Detection boxes + approach ratio
    actionable_ids = {id(d) for d in actionable}
    for d in dets.detections:
        x1, y1 = int(d.x1 * S), int(d.y1 * S)
        x2, y2 = int(d.x2 * S), int(d.y2 * S)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COL_BOX, 1)
        tag = _CLS_SHORT.get(int(d.cls), "?")
        if id(d) in actionable_ids:
            tag += f" {d.approach_ratio:.2f}"
        cv2.putText(canvas, tag, (x1, max(8, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL_BOX, 1, cv2.LINE_AA)

    # Human (replay) cursor reference — green crosshair
    gx, gy = tf.osu_to_render(gt_cx, gt_cy)
    gx, gy = int(gx * S), int(gy * S)
    cv2.drawMarker(canvas, (gx, gy), COL_GT, cv2.MARKER_CROSS, 14, 2)

    # Model cursor — red filled dot + white ring + trail
    mx, my = tf.osu_to_render(cur_x, cur_y)
    mx, my = int(mx * S), int(my * S)
    trail.append((mx, my))
    for i in range(1, len(trail)):
        cv2.line(canvas, trail[i - 1], trail[i], COL_MODEL, 1, cv2.LINE_AA)
    cv2.circle(canvas, (mx, my), 6, COL_MODEL, -1, cv2.LINE_AA)
    cv2.circle(canvas, (mx, my), 8, (255, 255, 255), 1, cv2.LINE_AA)

    # HUD bar
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    hud = f"f{frame_idx}  t={t_ms:.0f}ms  {mode}  dets={len(dets.detections)}"
    cv2.putText(canvas, hud, (6, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                COL_HUD, 1, cv2.LINE_AA)

    if pred is not None:
        info = (f"dx={pred.dx:+.1f} dy={pred.dy:+.1f}  "
                f"p(z)={pred.key_z:.2f} p(x)={pred.key_x:.2f}")
        cv2.putText(canvas, info, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    COL_HUD, 1, cv2.LINE_AA)

        # Z / X key lamps (top-right)
        for i, (lbl, prob) in enumerate((("Z", pred.key_z), ("X", pred.key_x))):
            on = prob > key_thr
            bx = canvas.shape[1] - 60 + i * 28
            cv2.rectangle(canvas, (bx, 6), (bx + 22, 28),
                          COL_KEY_ON if on else COL_KEY_OFF, -1)
            cv2.putText(canvas, lbl, (bx + 5, 23), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0) if on else COL_HUD, 1, cv2.LINE_AA)

    # Legend (bottom)
    yb = canvas.shape[0] - 8
    cv2.putText(canvas, "green=human  red=model", (6, yb),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_HUD, 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()
