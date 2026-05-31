#!/usr/bin/env python3
"""
AutOsu runtime — run the AI player on a live osu! window.

Modes:
    play    — full control (capture → detect → approach → controller → inject)
    observe — detection only, no input injection (for debugging/overlay)

Usage::

    python scripts/run.py play
    python scripts/run.py observe
    python scripts/run.py play --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so `from src.xxx` works
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import yaml


def main():
    parser = argparse.ArgumentParser(
        description="AutOsu — vision-based osu! AI player",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=["play", "observe"], default="observe", nargs="?",
                        help="Run mode: 'play' (inject inputs) or 'observe' (display only)")
    parser.add_argument("--config", "-c", default="configs/default.yaml",
                        help="Path to config YAML (default: configs/default.yaml)")
    parser.add_argument("--device", default=None,
                        help="Override CUDA device (e.g. 'cuda:0' or 'cpu')")
    parser.add_argument("--no-tensorrt", action="store_true",
                        help="Disable TensorRT even if available")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        print(f"[AutOsu] Config: {config_path}")
    else:
        cfg = {}
        print(f"[AutOsu] No config file found, using defaults")

    # Build PipelineConfig
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
        motion_net_path=cfg.get("motion_net_path", None),
        max_speed_osu_pms=cfg.get("max_speed_osu_pms", 4.0),
        speed_scale=cfg.get("speed_scale", 1.0),
        use_tensorrt=False if args.no_tensorrt else cfg.get("use_tensorrt", False),
    )

    # Setup injector based on mode
    if args.mode == "play":
        from src.runtime.injector import InputInjector
        injector = InputInjector(polling_rate_hz=1000)
        injector.start()
        print("[AutOsu] Mode: PLAY (input injection active)")
    else:
        from src.runtime.injector import MockInjector
        injector = MockInjector()
        print("[AutOsu] Mode: OBSERVE (no input injection)")

    # Initialize pipeline
    pipeline = GamePipeline(pipeline_cfg)
    try:
        pipeline.initialize(injector=injector)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Make sure you have trained the detector first:")
        print("  python scripts/train_detector.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Initialization failed: {e}")
        print("Make sure osu! is running and visible on screen.")
        sys.exit(1)

    # Graceful shutdown
    def signal_handler(sig, frame):
        print("\n[AutOsu] Shutting down...")
        pipeline.stop()
        if args.mode == "play":
            injector.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Start
    pipeline.start()
    print(f"[AutOsu] Running at target {pipeline_cfg.target_fps} FPS")
    print("[AutOsu] Press Ctrl+C to stop\n")

    # Main loop (display stats)
    try:
        while True:
            time.sleep(1.0)
            fps = pipeline.fps
            print(f"\r  FPS: {fps:.0f}    ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        if args.mode == "play":
            injector.stop()
        print("\n[AutOsu] Done.")


if __name__ == "__main__":
    main()
