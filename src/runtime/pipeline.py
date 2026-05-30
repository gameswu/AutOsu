"""
Real-time pipeline: capture → detect → estimate approach → action model → inject.

Runs at ~120Hz, orchestrating all components.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from src.vision.detector import Detection, Detector, FrameDetections, ObjClass
from src.action.state import GameStateVector, ObjectState, ActionVector


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    # Models
    detector_path: str = "runs/detect/best.pt"
    action_model_path: str = "runs/action/best.pth"
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
    action_fps: int = 30       # must match training FPS (data.fps in config)
    key_threshold: float = 0.5  # probability threshold for key presses
    use_tensorrt: bool = False


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
        self._action_model = None
        self._capture = None
        self._injector = None

        # State tracking
        self._prev_cursor_x = 256.0
        self._prev_cursor_y = 192.0
        self._prev_prev_cursor_x = 256.0  # for velocity calculation
        self._prev_prev_cursor_y = 192.0
        self._prev_time_ms = 0.0
        self._fps = 0.0

        # Key state tracking (only send events on transitions)
        self._key_z_pressed = False
        self._key_x_pressed = False

        # Action model timing (throttle to training FPS)
        self._action_interval_ms = 1000.0 / config.action_fps
        self._last_action_ms = 0.0
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

        # Approach estimator (geometric CV — no model file needed)
        from src.vision.approach_geometry import GeometricApproachEstimator
        self._approach_estimator = GeometricApproachEstimator()
        self._approach_estimator.reset()
        print(f"[Pipeline] Approach estimator: geometric CV + temporal fit (no model)")

        # Action model (optional - can run detection-only)
        action_path = Path(self.config.action_model_path)
        if action_path.exists():
            from src.action.model import ActionModelInference
            self._action_model = ActionModelInference(
                action_path, device=self.config.device
            )
            print(f"[Pipeline] Action model loaded")
        else:
            print(f"[Pipeline] Action model not found, running detection-only")

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
            self._prev_prev_cursor_x = ox
            self._prev_prev_cursor_y = oy
            print(f"[Pipeline] Initial cursor: screen({sx},{sy}) -> osu({ox:.0f},{oy:.0f})")

        print(f"[Pipeline] Action model FPS: {self.config.action_fps} "
              f"(interval: {self._action_interval_ms:.1f}ms)")

    def start(self):
        """Start the pipeline loop."""
        if self._running:
            return
        self._running = True
        if self._action_model:
            self._action_model.reset()
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

            # 4. Run action model at training FPS (not capture FPS)
            action_due = (now_ms - self._last_action_ms) >= self._action_interval_ms
            if action_due and self._action_model and self._injector:
                state = self._build_state(detections, now_ms)
                action_arr = self._action_model.predict(state.to_numpy())
                action = ActionVector.from_numpy(action_arr)
                self._execute_action(action, now_ms)
                self._prev_time_ms = now_ms
                self._last_action_ms = now_ms

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
        """Estimate approach_ratio for actionable objects via geometric CV."""
        actionable = detections.actionable_objects
        if not actionable:
            return
        # Sets det.approach_ratio in-place by measuring the approach-circle
        # radius directly from the frame (no neural network, no crops), with
        # temporal linear-fit filtering keyed by the frame timestamp.
        self._approach_estimator.estimate(
            frame, actionable, t_ms=detections.timestamp_ms
        )

    def _build_state(self, detections: FrameDetections, now_ms: float) -> GameStateVector:
        """Convert detections to a GameStateVector."""
        dt = now_ms - self._prev_time_ms if self._prev_time_ms > 0 else 16.0

        objects = []
        for det in detections.detections:
            ox, oy = self._mapping.model_to_osu(det.cx, det.cy)
            objects.append(ObjectState(
                class_id=int(det.cls),
                x=ox, y=oy,
                approach_ratio=det.approach_ratio,
            ))

        # Cursor velocity from previous movement (osu!px per ms)
        if dt > 0 and self._prev_time_ms > 0:
            vx = (self._prev_cursor_x - self._prev_prev_cursor_x) / dt
            vy = (self._prev_cursor_y - self._prev_prev_cursor_y) / dt
        else:
            vx = 0.0
            vy = 0.0

        return GameStateVector(
            objects=objects,
            cursor_x=self._prev_cursor_x,
            cursor_y=self._prev_cursor_y,
            cursor_vx=vx,
            cursor_vy=vy,
            time_delta_ms=dt,
        )

    def _execute_action(self, action: ActionVector, now_ms: float):
        """Execute an action through the injector."""
        # Save previous position for velocity calculation
        self._prev_prev_cursor_x = self._prev_cursor_x
        self._prev_prev_cursor_y = self._prev_cursor_y

        # Update cursor position
        new_x = self._prev_cursor_x + action.dx
        new_y = self._prev_cursor_y + action.dy

        # Clamp to playfield
        new_x = max(0, min(512, new_x))
        new_y = max(0, min(384, new_y))

        # Convert to screen coordinates and move
        sx, sy = self._mapping.osu_to_screen(new_x, new_y)
        self._injector.move_to_immediate(int(sx), int(sy))

        # Read actual cursor position to correct for drift
        actual_sx, actual_sy = self._injector.get_cursor_pos()
        actual_ox, actual_oy = self._mapping.screen_to_osu(actual_sx, actual_sy)
        # Clamp to playfield (cursor may be outside if window moved)
        self._prev_cursor_x = max(0, min(512, actual_ox))
        self._prev_cursor_y = max(0, min(384, actual_oy))

        # Key presses — only send events on state transitions
        threshold = self.config.key_threshold

        z_should_press = action.key_z > threshold
        if z_should_press and not self._key_z_pressed:
            self._injector.key_down(0x5A)  # Z
            self._key_z_pressed = True
        elif not z_should_press and self._key_z_pressed:
            self._injector.key_up(0x5A)
            self._key_z_pressed = False

        x_should_press = action.key_x > threshold
        if x_should_press and not self._key_x_pressed:
            self._injector.key_down(0x58)  # X
            self._key_x_pressed = True
        elif not x_should_press and self._key_x_pressed:
            self._injector.key_up(0x58)
            self._key_x_pressed = False
