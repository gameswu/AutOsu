#!/usr/bin/env python3
"""AutOsu runtime.

Usage::

    python scripts/run.py play -c myconfig.yaml
    python scripts/run.py observe -c myconfig.yaml
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import yaml


def _load_config(path: Optional[str]) -> dict:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[AutOsu] config not found: {p}")
        sys.exit(1)
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


from typing import Optional


def main():
    parser = argparse.ArgumentParser(description="AutOsu — vision-based osu! AI player")
    parser.add_argument("mode", choices=["play", "observe"], default="observe", nargs="?")
    parser.add_argument("--config", "-c", default=None,
                        help="Path to config YAML (no default; uses built-in defaults if omitted)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-tensorrt", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.config:
        print(f"[AutOsu] Config: {args.config}")

    from src.runtime.pipeline import PipelineConfig, GamePipeline

    pipeline_cfg = PipelineConfig(
        detector_path=cfg.get("detector_path", "runs/detect/train/weights/best.pt"),
        device=args.device or cfg.get("device", "cuda:0"),
        conf_threshold=cfg.get("conf_threshold", 0.3),
        iou_threshold=cfg.get("iou_threshold", 0.45),
        model_input_w=cfg.get("model_input_w", 640),
        model_input_h=cfg.get("model_input_h", 384),
        use_dxcam=cfg.get("use_dxcam", True),
        target_fps=cfg.get("target_fps", 120),
        hit_window=cfg.get("hit_window", 0.90),
        tap_hold_ms=cfg.get("tap_hold_ms", 40.0),
        spin_radius_osu=cfg.get("spin_radius_osu", 60.0),
        slide_follow_radius_osu=cfg.get("slide_follow_radius_osu", 120.0),
        slide_grace_ms=cfg.get("slide_grace_ms", 90.0),
        spin_grace_ms=cfg.get("spin_grace_ms", 120.0),
        max_speed_osu_pms=cfg.get("max_speed_osu_pms", 3.0),
        max_accel_osu_pms2=cfg.get("max_accel_osu_pms2", 0.20),
        seek_tau_ms=cfg.get("seek_tau_ms", 45.0),
        aim_cut_fraction=cfg.get("aim_cut_fraction", 0.65),
        lookahead_n=cfg.get("lookahead_n", 3),
        motion_net_path=cfg.get("motion_net_path", None),
        max_residual_osu_pms=cfg.get("max_residual_osu_pms", 1.5),
        style_scale=cfg.get("style_scale", 1.0),
        use_tensorrt=False if args.no_tensorrt else cfg.get("use_tensorrt", False),
    )

    if args.mode == "play":
        from src.runtime.injector import InputInjector
        injector = InputInjector(polling_rate_hz=1000)
        injector.start()
        print("[AutOsu] Mode: PLAY")
    else:
        from src.runtime.injector import MockInjector
        injector = MockInjector()
        print("[AutOsu] Mode: OBSERVE")

    pipeline = GamePipeline(pipeline_cfg)
    try:
        pipeline.initialize(injector=injector)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Init failed: {e}")
        sys.exit(1)

    def signal_handler(sig, frame):
        print("\n[AutOsu] Shutting down...")
        pipeline.stop()
        if args.mode == "play":
            injector.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    pipeline.start()
    print(f"[AutOsu] Running @ {pipeline_cfg.target_fps} FPS  (Ctrl+C to stop)\n")

    try:
        while True:
            time.sleep(1.0)
            print(f"\r  FPS: {pipeline.fps:.0f}    ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        if args.mode == "play":
            injector.stop()
        print("\n[AutOsu] Done.")


if __name__ == "__main__":
    main()
