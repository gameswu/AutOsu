"""
Object detection for osu! using YOLOv8.

Provides real-time detection of hitcircles, slider heads/bodies/ends,
and spinners from captured frames.

Supports:
- PyTorch inference (training/debugging)
- ONNX inference (optimised)
- TensorRT inference (production, lowest latency)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


class ObjClass(IntEnum):
    """Detection class IDs matching the training labels."""
    HITCIRCLE = 0
    SLIDER_HEAD = 1
    SLIDER_BODY = 2
    SLIDER_END = 3
    SPINNER = 4


@dataclass
class Detection:
    """A single detected object."""
    cls: ObjClass
    confidence: float
    # Bounding box in model-input pixel coordinates (640x384)
    cx: float          # centre x
    cy: float          # centre y
    w: float           # width
    h: float           # height
    # Converted osu! coordinates (set by pipeline after detection)
    osu_x: float = 0.0
    osu_y: float = 0.0
    # Approach ratio (set by ApproachEstimator, 0=just appeared, 1=hit time)
    approach_ratio: float = 0.0

    @property
    def x1(self) -> float:
        return self.cx - self.w / 2

    @property
    def y1(self) -> float:
        return self.cy - self.h / 2

    @property
    def x2(self) -> float:
        return self.cx + self.w / 2

    @property
    def y2(self) -> float:
        return self.cy + self.h / 2


@dataclass
class FrameDetections:
    """All detections in a single frame."""
    detections: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0  # time taken for inference
    timestamp_ms: float = 0.0  # frame capture timestamp

    @property
    def hitcircles(self) -> List[Detection]:
        return [d for d in self.detections if d.cls == ObjClass.HITCIRCLE]

    @property
    def slider_heads(self) -> List[Detection]:
        return [d for d in self.detections if d.cls == ObjClass.SLIDER_HEAD]

    @property
    def slider_bodies(self) -> List[Detection]:
        return [d for d in self.detections if d.cls == ObjClass.SLIDER_BODY]

    @property
    def slider_ends(self) -> List[Detection]:
        return [d for d in self.detections if d.cls == ObjClass.SLIDER_END]

    @property
    def spinners(self) -> List[Detection]:
        return [d for d in self.detections if d.cls == ObjClass.SPINNER]

    @property
    def actionable_objects(self) -> List[Detection]:
        """Objects the player needs to interact with (circles + slider heads)."""
        return [d for d in self.detections
                if d.cls in (ObjClass.HITCIRCLE, ObjClass.SLIDER_HEAD)]


class Detector:
    """
    YOLO-based object detector for osu!.

    Wraps Ultralytics YOLO model for inference.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda:0",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: Tuple[int, int] = (384, 640),  # (H, W)
    ):
        """
        Args:
            model_path: Path to .pt, .onnx, or .engine model file
            device: CUDA device or 'cpu'
            conf_threshold: Minimum confidence to keep a detection
            iou_threshold: NMS IoU threshold
            imgsz: Model input size (height, width)
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self._model = None
        self._model_path = Path(model_path)

    def load(self):
        """Load the model (lazy loading to avoid import overhead)."""
        from ultralytics import YOLO
        self._model = YOLO(str(self._model_path))
        # Warm up with a dummy inference
        dummy = np.zeros((self.imgsz[0], self.imgsz[1], 3), dtype=np.uint8)
        self._model.predict(
            dummy, device=self.device, verbose=False,
            imgsz=self.imgsz, conf=self.conf_threshold,
        )

    def detect(self, frame: np.ndarray, timestamp_ms: float = 0.0) -> FrameDetections:
        """
        Run detection on a single frame.

        Args:
            frame: BGR image of shape (384, 640, 3)
            timestamp_ms: Optional timestamp for tracking

        Returns:
            FrameDetections with all detected objects
        """
        if self._model is None:
            self.load()

        import time
        t0 = time.perf_counter()

        results = self._model.predict(
            frame,
            device=self.device,
            verbose=False,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
        )

        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000.0

        detections: List[Detection] = []

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                # boxes.xyxy: (N, 4), boxes.cls: (N,), boxes.conf: (N,)
                xyxy = boxes.xyxy.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    w = x2 - x1
                    h = y2 - y1

                    detections.append(Detection(
                        cls=ObjClass(cls_ids[i]),
                        confidence=float(confs[i]),
                        cx=cx, cy=cy, w=w, h=h,
                    ))

        return FrameDetections(
            detections=detections,
            inference_ms=inference_ms,
            timestamp_ms=timestamp_ms,
        )


class DetectorTRT:
    """
    TensorRT-optimised detector for production use.

    Provides ~2-3ms inference on RTX 3070 at 640x384 FP16.
    Build the engine with::

        trtexec --onnx=best.onnx --saveEngine=best.engine --fp16
    """

    def __init__(
        self,
        engine_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        self.engine_path = Path(engine_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._loaded = False
        # TRT specifics will be initialized in load()
        self._context = None
        self._bindings = None

    def load(self):
        """Load TensorRT engine. Requires tensorrt package."""
        # TensorRT inference via Ultralytics engine format
        from ultralytics import YOLO
        self._model = YOLO(str(self.engine_path))
        self._loaded = True
        # Warmup
        dummy = np.zeros((384, 640, 3), dtype=np.uint8)
        self._model.predict(dummy, verbose=False)

    def detect(self, frame: np.ndarray, timestamp_ms: float = 0.0) -> FrameDetections:
        """Run TensorRT inference."""
        if not self._loaded:
            self.load()

        import time
        t0 = time.perf_counter()

        results = self._model.predict(
            frame, verbose=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
        )

        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000.0

        detections: List[Detection] = []

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    w = x2 - x1
                    h = y2 - y1

                    detections.append(Detection(
                        cls=ObjClass(cls_ids[i]),
                        confidence=float(confs[i]),
                        cx=cx, cy=cy, w=w, h=h,
                    ))

        return FrameDetections(
            detections=detections,
            inference_ms=inference_ms,
            timestamp_ms=timestamp_ms,
        )
