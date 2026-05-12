#!/usr/bin/env python3
"""
Train YOLOv8n for osu! object detection.

Usage::

    python scripts/train_detector.py
    python scripts/train_detector.py --epochs 150 --batch 16
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
    parser = argparse.ArgumentParser(description="Train YOLOv8n on osu! dataset")
    parser.add_argument("--data", default="dataset/data.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="train")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram_gb:.1f} GB")
        if vram_gb < 7 and args.batch > 16:
            args.batch = 16
            print(f"Reduced batch to {args.batch} for GPU memory")

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=args.name,
            exist_ok=True,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            weight_decay=0.0005,
            warmup_epochs=3,
            hsv_h=0.015,
            hsv_s=0.3,
            hsv_v=0.3,
            degrees=0.0,
            translate=0.05,
            scale=0.2,
            flipud=0.0,
            fliplr=0.0,
            mosaic=0.5,
            mixup=0.1,
            patience=20,
            save_period=10,
            val=True,
            plots=True,
            verbose=True,
            rect=True,
        )

    if not args.no_export:
        print("\nExporting to ONNX...")
        best_path = Path(args.project) / args.name / "weights" / "best.pt"
        if best_path.exists():
            export_model = YOLO(str(best_path))
            export_model.export(format="onnx", imgsz=[384, 640], simplify=True, dynamic=False)
            print(f"ONNX exported: {best_path.with_suffix('.onnx')}")


if __name__ == "__main__":
    main()
