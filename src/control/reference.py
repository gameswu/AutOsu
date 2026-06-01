"""Navigation-goal + key-state generator (vision-only).

Emits per frame:
  - target list  (all visible targets the trajectory model should consider)
  - key state    (tap / hold / release)
  - phase        (idle / approach / slide / spin)

Cursor motion is handled entirely by the learned TrajectoryModel.  This module
only determines *which keys to press* and *what targets exist*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from src.vision.detector import FrameDetections
from src.control.tracker import TimingTracker
from src.control.planner import (
    Scene, SceneObject, build_scene, select_target,
)

Vec = Tuple[float, float]

VK_Z = 0x5A
VK_X = 0x58

PHASE_IDLE = "idle"
PHASE_APPROACH = "approach"
PHASE_SLIDE = "slide"
PHASE_SPIN = "spin"


@dataclass
class TargetInfo:
    """One target visible to the trajectory model."""
    x: float
    y: float
    approach_ratio: float
    kind: str        # "circle" | "slider" | "spinner"
    is_active: bool  # True for slider ball visible / spinner active


@dataclass
class Reference:
    """One frame of key state + target list for the trajectory model."""
    key_z: bool = False
    key_x: bool = False
    phase: str = PHASE_IDLE
    targets: List[TargetInfo] = field(default_factory=list)


def build_targets(scene: Scene) -> List[TargetInfo]:
    """Convert scene objects to a flat target list for the model."""
    targets: List[TargetInfo] = []
    for o in scene.objects:
        if o.head_visible:
            targets.append(TargetInfo(o.x, o.y, o.approach_ratio, o.kind, False))
        if o.has_ball and o.ball_x is not None:
            is_dup = (o.head_visible
                      and abs(o.x - o.ball_x) < 2
                      and abs(o.y - o.ball_y) < 2)
            if not is_dup:
                targets.append(TargetInfo(o.ball_x, o.ball_y, 1.0, "slider", True))
    if scene.spinner is not None:
        targets.append(TargetInfo(
            scene.spinner.x, scene.spinner.y, 1.0, "spinner", True))
    return targets


class ReferenceController:
    """Vision-only key-state + target-list generator.

    Phase FSM determines when to tap / hold / release keys.
    Target list is passed through to the trajectory model unchanged.
    """

    def __init__(
        self,
        hit_window: float = 0.90,
        tap_hold_ms: float = 40.0,
        tap_refractory_ms: float = 70.0,
        slide_grace_ms: float = 90.0,
        spin_grace_ms: float = 120.0,
    ):
        self.hit_window = hit_window
        self.tap_hold_ms = tap_hold_ms
        self.tap_refractory_ms = tap_refractory_ms
        self.slide_grace_ms = slide_grace_ms
        self.spin_grace_ms = spin_grace_ms

        self.timing = TimingTracker()

        self._state = PHASE_APPROACH
        self._next_key = VK_Z
        self._held_key: Optional[int] = None
        self._tap_key: Optional[int] = None
        self._tap_until = 0.0
        self._last_tap_t = -1e9
        self._spin_last_seen_t = -1e9
        self._slide_last_seen_t = -1e9
        self._last_t: Optional[float] = None

    def reset(self, cursor: Optional[Vec] = None) -> None:
        self.timing.reset()
        self._state = PHASE_APPROACH
        self._next_key = VK_Z
        self._held_key = None
        self._tap_key = None
        self._tap_until = 0.0
        self._last_tap_t = -1e9
        self._spin_last_seen_t = -1e9
        self._slide_last_seen_t = -1e9
        self._last_t = None

    # ── entry points ──────────────────────────────────────────────

    def update(
        self,
        detections: FrameDetections,
        cursor: Vec,
        t_ms: float,
        to_osu: Callable[[float, float], Tuple[float, float]],
    ) -> Reference:
        scene = build_scene(detections, to_osu)
        return self.step(scene, cursor, t_ms)

    def step(self, scene: Scene, cursor: Vec, t_ms: float) -> Reference:
        self._last_t = t_ms

        self.timing.update(
            [{"x": o.x, "y": o.y, "ratio": o.approach_ratio}
             for o in scene.actionables],
            t_ms,
        )

        targets = build_targets(scene)

        # Spinner takes priority.
        if scene.spinner is not None:
            self._spin_last_seen_t = t_ms
            return self._handle_spin(targets, t_ms)

        if self._state == PHASE_SPIN:
            if (t_ms - self._spin_last_seen_t) <= self.spin_grace_ms:
                return self._handle_spin(targets, t_ms)
            self._state = PHASE_APPROACH
            self._held_key = None

        out = self._handle_slide(scene, targets, t_ms)
        if out is not None:
            return out

        return self._handle_approach(scene, targets, t_ms)

    # ── approach ──────────────────────────────────────────────────

    def _handle_approach(
        self, scene: Scene, targets: List[TargetInfo], t_ms: float,
    ) -> Reference:
        target = select_target(scene)
        if target is None:
            return self._finish(targets, t_ms, PHASE_IDLE)

        ready = (target.approach_ratio >= self.hit_window
                 and (t_ms - self._last_tap_t) >= self.tap_refractory_ms)
        if ready:
            key = self._take_key()
            self._last_tap_t = t_ms
            if target.kind == "slider":
                self._held_key = key
                self._state = PHASE_SLIDE
                self._slide_last_seen_t = t_ms
            else:
                self._tap_key = key
                self._tap_until = t_ms + self.tap_hold_ms

        return self._finish(targets, t_ms, PHASE_APPROACH)

    # ── slide ─────────────────────────────────────────────────────

    def _handle_slide(
        self, scene: Scene, targets: List[TargetInfo], t_ms: float,
    ) -> Optional[Reference]:
        has_ball = any(o.has_ball for o in scene.objects if o.kind == "slider")
        has_body = getattr(scene, "has_slider_body", False)

        if has_ball or has_body:
            if self._held_key is None:
                self._held_key = self._take_key()
            self._state = PHASE_SLIDE
            self._slide_last_seen_t = t_ms
            return self._finish(targets, t_ms, PHASE_SLIDE)

        if self._state == PHASE_SLIDE:
            if (t_ms - self._slide_last_seen_t) <= self.slide_grace_ms:
                return self._finish(targets, t_ms, PHASE_SLIDE)
            self._held_key = None
            self._state = PHASE_APPROACH
        return None

    # ── spin ──────────────────────────────────────────────────────

    def _handle_spin(
        self, targets: List[TargetInfo], t_ms: float,
    ) -> Reference:
        if self._state != PHASE_SPIN or self._held_key is None:
            self._held_key = self._take_key()
        self._state = PHASE_SPIN
        return self._finish(targets, t_ms, PHASE_SPIN)

    # ── internals ─────────────────────────────────────────────────

    def _take_key(self) -> int:
        key = self._next_key
        self._next_key = VK_X if key == VK_Z else VK_Z
        return key

    def _finish(
        self,
        targets: List[TargetInfo],
        t_ms: float,
        phase: str = PHASE_IDLE,
    ) -> Reference:
        if self._tap_key is not None and t_ms >= self._tap_until:
            self._tap_key = None
        z = (self._held_key == VK_Z) or (self._tap_key == VK_Z)
        x = (self._held_key == VK_X) or (self._tap_key == VK_X)
        return Reference(key_z=z, key_x=x, phase=phase, targets=targets)
