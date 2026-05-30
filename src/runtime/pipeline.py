"""
Real-time pipeline: capture -> detect -> estimate approach -> controller -> inject.

Runs at ~120Hz, orchestrating all components. The action is produced by a
deterministic, vision-only :class:`Controller` (no learned policy).
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from src.vision.detector import Detection, Detector, FrameDetections, ObjClass
from src.control import Controller, ControlOutput


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
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

    # Controller (deterministic player) tuning
    hit_window: float = 0.90
    hit_radius_osu: float = 80.0
    tap_hold_ms: float = 40.0
    spin_radius_osu: float = 60.0
    motion_jitter: float = 1.2


class GamePipeline:
    """
    Main pipeline: capture → detect → build state → action model → inject.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Components (initialised in .initialize())
        self._mapping = None
        self._detector = None
        self._approach_estimator = None
        self._controller = None
        self._capture = None
        self._injector = None

        # State tracking
        self._prev_cursor_x = 256.0
        self._prev_cursor_y = 192.0
        self._fps = 0.0

        # Key state tracking (only send events on transitions)
        self._key_z_pressed = False
        self._key_x_pressed = False

        self._latest_detections: Optional[FrameDetections] = None

    def initialize(self, injector=None):
        """
        Load all models and initialize capture.

        Args:
            injector: InputInjector or MockInjector instance
        """
        from src.runtime.window import get_playfield_mapping
        from src.runtime.capture import ScreenCapture

        # Window detection
        self._mapping = get_playfield_mapping(
            model_input_w=self.config.model_input_w,
            model_input_h=self.config.model_input_h,
        )
        print(f"[Pipeline] osu! window: {self._mapping.client_w}x{self._mapping.client_h}")

        # YOLO detector
        detector_path = Path(self.config.detector_path)
        if not detector_path.exists():
            raise FileNotFoundError(f"Detector model not found: {detector_path}")

        self._detector = Detector(
            detector_path,
            device=self.config.device,
            conf_threshold=self.config.conf_threshold,
            iou_threshold=self.config.iou_threshold,
            imgsz=(self.config.model_input_h, self.config.model_input_w),
        )
        self._detector.load()
        print(f"[Pipeline] Detector loaded: {detector_path.name}")

        # Approach estimator: ratio read from the detector's approach_circle
        # boxes (de-overlaps cleanly), with online temporal smoothing.
        from src.vision.approach_from_boxes import BoxApproachEstimator
        self._approach_estimator = BoxApproachEstimator()
        self._approach_estimator.reset()
        print(f"[Pipeline] Approach estimator: YOLO approach_circle boxes + temporal fit")

        # Deterministic vision-only controller (replaces the action model)
        self._controller = Controller(
            hit_window=self.config.hit_window,
            hit_radius_osu=self.config.hit_radius_osu,
            tap_hold_ms=self.config.tap_hold_ms,
            spin_radius_osu=self.config.spin_radius_osu,
            jitter=self.config.motion_jitter,
        )
        print(f"[Pipeline] Controller: deterministic approach/slide/spin state machine")

        # Screen capture
        region = self._mapping.capture_region
        self._capture = ScreenCapture(
            region=region,
            target_size=(self.config.model_input_w, self.config.model_input_h),
            use_dxcam=self.config.use_dxcam,
        )
        print(f"[Pipeline] Capture: {self._capture.backend}")

        # Injector
        self._injector = injector

        # Read actual cursor position at startup (instead of guessing center)
        if self._injector:
            sx, sy = self._injector.get_cursor_pos()
            ox, oy = self._mapping.screen_to_osu(sx, sy)
            ox = max(0, min(512, ox))
            oy = max(0, min(384, oy))
            self._prev_cursor_x = ox
            self._prev_cursor_y = oy
            self._controller.reset(cursor=(ox, oy))
            print(f"[Pipeline] Initial cursor: screen({sx},{sy}) -> osu({ox:.0f},{oy:.0f})")

    def start(self):
        """Start the pipeline loop."""
        if self._running:
            return
        self._running = True
        if self._controller:
            self._controller.reset(cursor=(self._prev_cursor_x, self._prev_cursor_y))
        if self._approach_estimator:
            self._approach_estimator.reset()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[Pipeline] Started")

    def stop(self):
        """Stop the pipeline and release any held keys."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        # Release held keys
        if self._injector:
            if self._key_z_pressed:
                self._injector.key_up(0x5A)
                self._key_z_pressed = False
            if self._key_x_pressed:
                self._injector.key_up(0x58)
                self._key_x_pressed = False
        print("[Pipeline] Stopped")

    @property
    def fps(self) -> float:
        return self._fps

    def _loop(self):
        target_interval = 1.0 / self.config.target_fps
        frame_times: List[float] = []

        while self._running:
            t0 = time.perf_counter()
            now_ms = t0 * 1000.0

            # 1. Capture frame
            frame = self._capture.grab()
            if frame is None:
                time.sleep(0.001)
                continue

            # 2. Detect objects (every frame for responsive detection)
            detections = self._detector.detect(frame, timestamp_ms=now_ms)
            self._latest_detections = detections

            # 3. Estimate approach ratios
            if self._approach_estimator:
                self._estimate_approach(frame, detections)

            # 4. Run the deterministic controller every captured frame.
            if self._controller and self._injector:
                out = self._controller.update(
                    detections,
                    (self._prev_cursor_x, self._prev_cursor_y),
                    now_ms,
                    self._mapping.model_to_osu,
                )
                self._execute_output(out)

            # FPS
            elapsed = time.perf_counter() - t0
            frame_times.append(elapsed)
            if len(frame_times) > 60:
                frame_times = frame_times[-60:]
            avg = sum(frame_times) / len(frame_times)
            self._fps = 1.0 / avg if avg > 0 else 0

            # Rate limit
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _estimate_approach(self, frame: np.ndarray, detections: FrameDetections):
        """Set approach_ratio on actionable objects from approach_circle boxes."""
        # The box estimator pairs each hitcircle/slider-head with its detected
        # approach ring (de-overlaps cleanly), with temporal smoothing keyed by
        # the frame timestamp. The frame itself is unused (kept for signature
        # parity / optional CV fallback).
        self._approach_estimator.estimate(detections, t_ms=detections.timestamp_ms)

    def _execute_output(self, out: ControlOutput):
        """Execute a controller output (absolute target + key state)."""
        # Clamp target to playfield and move.
        new_x = max(0, min(512, out.x))
        new_y = max(0, min(384, out.y))
        sx, sy = self._mapping.osu_to_screen(new_x, new_y)
        self._injector.move_to_immediate(int(sx), int(sy))

        # Read actual cursor position to correct for drift / window moves.
        actual_sx, actual_sy = self._injector.get_cursor_pos()
        actual_ox, actual_oy = self._mapping.screen_to_osu(actual_sx, actual_sy)
        self._prev_cursor_x = max(0, min(512, actual_ox))
        self._prev_cursor_y = max(0, min(384, actual_oy))

        # Key presses — only send events on state transitions.
        if out.key_z and not self._key_z_pressed:
            self._injector.key_down(0x5A)  # Z
            self._key_z_pressed = True
        elif not out.key_z and self._key_z_pressed:
            self._injector.key_up(0x5A)
            self._key_z_pressed = False

        if out.key_x and not self._key_x_pressed:
            self._injector.key_down(0x58)  # X
            self._key_x_pressed = True
        elif not out.key_x and self._key_x_pressed:
            self._injector.key_up(0x58)
            self._key_x_pressed = False
