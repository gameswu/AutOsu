"""Real-time pipeline: capture -> detect -> approach -> controller -> inject."""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.vision.detector import Detection, Detector, FrameDetections, ObjClass
from src.control import Controller, ControlOutput


@dataclass
class PipelineConfig:
    # Models
    detector_path: str = "runs/detect/best.pt"
    device: str = "cuda:0"

    # Detection
    conf_threshold: float = 0.3
    iou_threshold: float = 0.45
    model_input_w: int = 640
    model_input_h: int = 384

    # Capture
    use_dxcam: bool = True

    # Runtime
    target_fps: int = 120
    use_tensorrt: bool = False

    # Controller (key-press timing — rule-based, kept)
    hit_window: float = 0.90
    tap_hold_ms: float = 40.0
    tap_refractory_ms: float = 70.0
    slide_grace_ms: float = 90.0
    spin_grace_ms: float = 120.0

    # Trajectory model (learned cursor motion)
    motion_net_path: Optional[str] = None


class GamePipeline:
    """Main pipeline: capture -> detect -> approach -> controller -> inject."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._mapping = None
        self._detector = None
        self._approach_estimator = None
        self._controller = None
        self._capture = None
        self._injector = None
        self._prev_cursor_x = 256.0
        self._prev_cursor_y = 192.0
        self._fps = 0.0
        self._key_z_pressed = False
        self._key_x_pressed = False
        self._latest_detections: Optional[FrameDetections] = None

    def initialize(self, injector=None):
        from src.runtime.window import get_playfield_mapping
        from src.runtime.capture import ScreenCapture

        self._mapping = get_playfield_mapping(
            model_input_w=self.config.model_input_w,
            model_input_h=self.config.model_input_h,
        )
        print(f"[Pipeline] window: {self._mapping.client_w}x{self._mapping.client_h}")

        detector_path = Path(self.config.detector_path)
        if not detector_path.exists():
            raise FileNotFoundError(f"Detector not found: {detector_path}")

        self._detector = Detector(
            detector_path,
            device=self.config.device,
            conf_threshold=self.config.conf_threshold,
            iou_threshold=self.config.iou_threshold,
            imgsz=(self.config.model_input_h, self.config.model_input_w),
        )
        self._detector.load()
        print(f"[Pipeline] Detector: {detector_path.name}")

        from src.vision.approach_from_boxes import BoxApproachEstimator
        self._approach_estimator = BoxApproachEstimator()
        self._approach_estimator.reset()

        self._controller = Controller(
            hit_window=self.config.hit_window,
            tap_hold_ms=self.config.tap_hold_ms,
            tap_refractory_ms=self.config.tap_refractory_ms,
            slide_grace_ms=self.config.slide_grace_ms,
            spin_grace_ms=self.config.spin_grace_ms,
            motion_net_path=self.config.motion_net_path,
            device=self.config.device,
        )
        mode = "learned" if self._controller.motion_net_active else "inactive"
        print(f"[Pipeline] Controller: trajectory model {mode}")

        region = self._mapping.capture_region
        self._capture = ScreenCapture(
            region=region,
            target_size=(self.config.model_input_w, self.config.model_input_h),
            use_dxcam=self.config.use_dxcam,
        )
        print(f"[Pipeline] Capture: {self._capture.backend}")

        self._injector = injector
        if self._injector:
            sx, sy = self._injector.get_cursor_pos()
            ox, oy = self._mapping.screen_to_osu(sx, sy)
            ox = max(0, min(512, ox))
            oy = max(0, min(384, oy))
            self._prev_cursor_x = ox
            self._prev_cursor_y = oy
            self._controller.reset(cursor=(ox, oy))

    def start(self):
        if self._running:
            return
        self._running = True
        if self._controller:
            self._controller.reset(cursor=(self._prev_cursor_x, self._prev_cursor_y))
        if self._approach_estimator:
            self._approach_estimator.reset()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._injector:
            if self._key_z_pressed:
                self._injector.key_up(0x5A)
                self._key_z_pressed = False
            if self._key_x_pressed:
                self._injector.key_up(0x58)
                self._key_x_pressed = False

    @property
    def fps(self) -> float:
        return self._fps

    def _loop(self):
        target_interval = 1.0 / self.config.target_fps
        frame_times: List[float] = []

        while self._running:
            t0 = time.perf_counter()
            now_ms = t0 * 1000.0

            frame = self._capture.grab()
            if frame is None:
                time.sleep(0.001)
                continue

            detections = self._detector.detect(frame, timestamp_ms=now_ms)
            self._latest_detections = detections

            if self._approach_estimator:
                self._approach_estimator.estimate(detections, t_ms=now_ms)

            if self._controller and self._injector:
                out = self._controller.update(
                    detections,
                    (self._prev_cursor_x, self._prev_cursor_y),
                    now_ms,
                    self._mapping.model_to_osu,
                )
                self._execute_output(out)

            elapsed = time.perf_counter() - t0
            frame_times.append(elapsed)
            if len(frame_times) > 60:
                frame_times = frame_times[-60:]
            avg = sum(frame_times) / len(frame_times)
            self._fps = 1.0 / avg if avg > 0 else 0

            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _execute_output(self, out: ControlOutput):
        new_x = max(0, min(512, out.x))
        new_y = max(0, min(384, out.y))
        sx, sy = self._mapping.osu_to_screen(new_x, new_y)
        self._injector.move_to_immediate(int(sx), int(sy))

        actual_sx, actual_sy = self._injector.get_cursor_pos()
        actual_ox, actual_oy = self._mapping.screen_to_osu(actual_sx, actual_sy)
        self._prev_cursor_x = max(0, min(512, actual_ox))
        self._prev_cursor_y = max(0, min(384, actual_oy))

        if out.key_z and not self._key_z_pressed:
            self._injector.key_down(0x5A)
            self._key_z_pressed = True
        elif not out.key_z and self._key_z_pressed:
            self._injector.key_up(0x5A)
            self._key_z_pressed = False

        if out.key_x and not self._key_x_pressed:
            self._injector.key_down(0x58)
            self._key_x_pressed = True
        elif not out.key_x and self._key_x_pressed:
            self._injector.key_up(0x58)
            self._key_x_pressed = False
