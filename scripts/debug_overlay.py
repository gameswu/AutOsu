#!/usr/bin/env python3
"""
Debug overlay — visual diagnosis of detection + coordinate conversion.

Captures the osu! window, runs YOLO, and overlays:
  - Bounding boxes + class labels + confidence
  - Converted osu! coordinates for each detection
  - Current cursor position (screen → osu!)
  - Playfield boundary rectangle
  - FPS + inference time

Press Q or Esc to quit.

Usage::

    python scripts/debug_overlay.py
    python scripts/debug_overlay.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np
import yaml

# ── Class colours (BGR) ─────────────────────────────────────────────────
_COLOURS = {
    0: (0, 255, 255),   # hitcircle   — yellow
    1: (0, 200, 0),     # slider_head — green
    2: (200, 100, 0),   # slider_body — teal
    3: (0, 100, 200),   # slider_end  — orange
    4: (255, 0, 255),   # spinner     — magenta
}
_CLASS_NAMES = {
    0: "circle",
    1: "s_head",
    2: "s_body",
    3: "s_end",
    4: "spinner",
}


def main():
    parser = argparse.ArgumentParser(description="AutOsu debug overlay")
    parser.add_argument("--config", "-c", default="configs/default.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    device = args.device or cfg.get("device", "cuda:0")
    detector_path = Path(cfg.get("detector_path", "runs/detect/train/weights/best.pt"))
    conf_threshold = cfg.get("conf_threshold", 0.3)
    iou_threshold = cfg.get("iou_threshold", 0.45)
    model_w = cfg.get("model_input_w", 640)
    model_h = cfg.get("model_input_h", 384)

    if not detector_path.exists():
        print(f"[ERROR] Detector model not found: {detector_path}")
        sys.exit(1)

    # ── Window mapping ──────────────────────────────────────────────────
    from src.runtime.window import get_playfield_mapping
    mapping = get_playfield_mapping(model_input_w=model_w, model_input_h=model_h)
    print(f"[Debug] Window client: ({mapping.client_x},{mapping.client_y}) "
          f"{mapping.client_w}x{mapping.client_h}")
    print(f"[Debug] Playfield screen origin: ({mapping.playfield_x},{mapping.playfield_y}) "
          f"{mapping.playfield_w}x{mapping.playfield_h}")
    print(f"[Debug] Scale: {mapping.scale:.4f}")

    # Verify coordinate roundtrip
    for name, ox, oy in [("center", 256, 192), ("origin", 0, 0), ("corner", 512, 384)]:
        mx, my = mapping.osu_to_model(ox, oy)
        ox2, oy2 = mapping.model_to_osu(mx, my)
        err = ((ox2 - ox) ** 2 + (oy2 - oy) ** 2) ** 0.5
        print(f"  roundtrip {name}: osu({ox},{oy}) -> model({mx:.1f},{my:.1f}) "
              f"-> osu({ox2:.1f},{oy2:.1f})  err={err:.4f}")

    # Playfield corners in model-pixel coords (for drawing the boundary)
    pf_tl = mapping.osu_to_model(0, 0)
    pf_br = mapping.osu_to_model(512, 384)

    # ── Screen capture ──────────────────────────────────────────────────
    from src.runtime.capture import ScreenCapture
    capture = ScreenCapture(
        region=mapping.capture_region,
        target_size=(model_w, model_h),
        use_dxcam=cfg.get("use_dxcam", True),
    )
    print(f"[Debug] Capture backend: {capture.backend}")

    # ── YOLO detector ───────────────────────────────────────────────────
    from src.vision.detector import Detector
    detector = Detector(
        detector_path, device=device,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        imgsz=(model_h, model_w),
    )
    print("[Debug] Loading YOLO model...")
    detector.load()
    print("[Debug] YOLO loaded. Press Q/Esc to quit.\n")

    # ── Cursor reader (works on Windows only) ───────────────────────────
    try:
        import ctypes
        import ctypes.wintypes as wt
        _user32 = ctypes.windll.user32

        def get_cursor_screen():
            pt = wt.POINT()
            _user32.GetCursorPos(ctypes.byref(pt))
            return pt.x, pt.y
    except Exception:
        def get_cursor_screen():
            return 0, 0

    # ── Main loop ───────────────────────────────────────────────────────
    frame_times = []
    target_interval = 1.0 / 60  # 60 fps display

    while True:
        t0 = time.perf_counter()

        frame = capture.grab()
        if frame is None:
            time.sleep(0.001)
            continue

        # Detect
        dets = detector.detect(frame, timestamp_ms=t0 * 1000)

        # Make a copy for drawing
        vis = frame.copy()

        # Draw playfield boundary
        cv2.rectangle(
            vis,
            (int(pf_tl[0]), int(pf_tl[1])),
            (int(pf_br[0]), int(pf_br[1])),
            (100, 100, 100), 1,
        )

        # Draw detections
        for det in dets.detections:
            colour = _COLOURS.get(int(det.cls), (255, 255, 255))
            name = _CLASS_NAMES.get(int(det.cls), "?")

            # Bounding box
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)

            # Centre cross
            cx_i, cy_i = int(det.cx), int(det.cy)
            cv2.drawMarker(vis, (cx_i, cy_i), colour,
                           cv2.MARKER_CROSS, 10, 1)

            # Convert to osu! coordinates
            osu_x, osu_y = mapping.model_to_osu(det.cx, det.cy)

            # Label: class conf  osu(x,y)
            label = f"{name} {det.confidence:.2f}  osu({osu_x:.0f},{osu_y:.0f})"
            cv2.putText(vis, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, colour, 1,
                        cv2.LINE_AA)

        # Draw current cursor position
        csx, csy = get_cursor_screen()
        # Convert screen cursor → model pixel (to draw on the frame)
        cursor_model_x = (csx - mapping.client_x) * model_w / mapping.client_w
        cursor_model_y = (csy - mapping.client_y) * model_h / mapping.client_h
        cursor_osu_x, cursor_osu_y = mapping.screen_to_osu(csx, csy)

        cmx_i, cmy_i = int(cursor_model_x), int(cursor_model_y)
        if 0 <= cmx_i < model_w and 0 <= cmy_i < model_h:
            cv2.drawMarker(vis, (cmx_i, cmy_i), (0, 0, 255),
                           cv2.MARKER_TILTED_CROSS, 14, 2)
            cv2.putText(vis, f"cursor osu({cursor_osu_x:.0f},{cursor_osu_y:.0f})",
                        (cmx_i + 10, cmy_i - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1,
                        cv2.LINE_AA)

        # FPS / timing info
        elapsed = time.perf_counter() - t0
        frame_times.append(elapsed)
        if len(frame_times) > 60:
            frame_times = frame_times[-60:]
        avg = sum(frame_times) / len(frame_times)
        fps = 1.0 / avg if avg > 0 else 0

        info_lines = [
            f"FPS: {fps:.0f}  YOLO: {dets.inference_ms:.1f}ms  "
            f"dets: {len(dets.detections)}",
            f"cursor screen({csx},{csy})  osu({cursor_osu_x:.0f},{cursor_osu_y:.0f})",
            f"playfield origin screen({mapping.playfield_x},{mapping.playfield_y})  "
            f"scale={mapping.scale:.3f}",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(vis, line, (6, 16 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                        cv2.LINE_AA)

        cv2.imshow("AutOsu Debug", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # Q or Esc
            break

        # Rate limit
        sleep_time = target_interval - (time.perf_counter() - t0)
        if sleep_time > 0:
            time.sleep(sleep_time)

    capture.stop()
    cv2.destroyAllWindows()
    print("[Debug] Done.")


if __name__ == "__main__":
    main()
