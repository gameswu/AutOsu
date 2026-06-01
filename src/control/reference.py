"""Navigation-goal + key-state generator (vision-only).

Emits per frame:
  - navigation goal (aim-point-offset toward the *next* target, not dead centre)
  - key state (tap / hold / release)

The actual cursor motion is handled by the controller (seek + optional style).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from src.vision.detector import FrameDetections
from src.control.tracker import TimingTracker
from src.control.planner import (
    Scene, build_scene, select_target, select_targets, estimate_hit_radius,
)

Vec = Tuple[float, float]

VK_Z = 0x5A
VK_X = 0x58

PHASE_IDLE = "idle"
PHASE_APPROACH = "approach"
PHASE_SLIDE = "slide"
PHASE_SPIN = "spin"


@dataclass
class Reference:
    """One frame of navigation goal + key state."""
    x: float
    y: float
    key_z: bool = False
    key_x: bool = False
    phase: str = PHASE_IDLE
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    approach_ratio: float = 0.0
    time_to_hit_ms: float = 0.0


class ReferenceController:
    """Vision-only navigation-goal + key-state generator."""

    def __init__(
        self,
        hit_window: float = 0.90,
        tap_hold_ms: float = 40.0,
        tap_refractory_ms: float = 70.0,
        spin_radius_osu: float = 60.0,
        spin_speed: float = 0.025,
        slide_follow_radius_osu: float = 120.0,
        slide_grace_ms: float = 90.0,
        spin_grace_ms: float = 120.0,
        aim_cut_fraction: float = 0.65,
        lookahead_n: int = 3,
    ):
        self.hit_window = hit_window
        self.tap_hold_ms = tap_hold_ms
        self.tap_refractory_ms = tap_refractory_ms
        self.spin_radius_osu = spin_radius_osu
        self.spin_speed = spin_speed
        self.slide_follow_radius_osu = slide_follow_radius_osu
        self.slide_grace_ms = slide_grace_ms
        self.spin_grace_ms = spin_grace_ms
        self.aim_cut_fraction = aim_cut_fraction
        self.lookahead_n = lookahead_n

        self.timing = TimingTracker()

        self._state = PHASE_APPROACH
        self._next_key = VK_Z
        self._held_key: Optional[int] = None
        self._tap_key: Optional[int] = None
        self._tap_until = 0.0
        self._last_tap_t = -1e9
        self._spin_angle = 0.0
        self._spin_center: Optional[Vec] = None
        self._spin_last_seen_t = -1e9
        self._slide_pos: Optional[Vec] = None
        self._slide_vel: Vec = (0.0, 0.0)
        self._slide_last_seen_t = -1e9
        self._last_t: Optional[float] = None
        self._cursor: Vec = (256.0, 192.0)
        self._hit_radius_ema: float = 36.0  # running estimate of hit-circle radius

    def reset(self, cursor: Optional[Vec] = None) -> None:
        self.timing.reset()
        self._state = PHASE_APPROACH
        self._next_key = VK_Z
        self._held_key = None
        self._tap_key = None
        self._tap_until = 0.0
        self._last_tap_t = -1e9
        self._spin_angle = 0.0
        self._spin_center = None
        self._spin_last_seen_t = -1e9
        self._slide_pos = None
        self._slide_vel = (0.0, 0.0)
        self._slide_last_seen_t = -1e9
        self._last_t = None
        self._hit_radius_ema = 36.0
        if cursor is not None:
            self._cursor = cursor

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
        self._cursor = cursor
        dt = (t_ms - self._last_t) if self._last_t is not None else 16.0
        self._last_t = t_ms

        self.timing.update(
            [{"x": o.x, "y": o.y, "ratio": o.approach_ratio}
             for o in scene.actionables],
            t_ms,
        )

        # Spinner takes over.
        if scene.spinner is not None:
            self._spin_center = (scene.spinner.x, scene.spinner.y)
            self._spin_last_seen_t = t_ms
            return self._do_spin(self._spin_center, dt, t_ms)

        if self._state == PHASE_SPIN:
            if (self._spin_center is not None
                    and (t_ms - self._spin_last_seen_t) <= self.spin_grace_ms):
                return self._do_spin(self._spin_center, dt, t_ms)
            self._state = PHASE_APPROACH
            self._held_key = None
            self._spin_center = None

        out = self._do_slide(scene, t_ms)
        if out is not None:
            return out

        return self._do_approach(scene, t_ms)

    # ── approach with aim-point offset ────────────────────────────

    def _do_approach(self, scene: Scene, t_ms: float) -> Reference:
        targets = select_targets(scene, self.lookahead_n)
        if not targets:
            return self._finish(self._cursor, t_ms, phase=PHASE_IDLE)

        target = targets[0]
        next_target = targets[1] if len(targets) > 1 else None

        # Update hit-radius estimate from detection boxes.
        r_est = estimate_hit_radius(targets, fallback=self._hit_radius_ema)
        self._hit_radius_ema += 0.15 * (r_est - self._hit_radius_ema)

        # Aim-point: shift toward next target within the hit circle.
        goal = self._aim_offset((target.x, target.y), next_target,
                                self._hit_radius_ema)

        tth = self.timing.time_to_hit_ms(target.approach_ratio)

        ready = (target.approach_ratio >= self.hit_window
                 and (t_ms - self._last_tap_t) >= self.tap_refractory_ms)
        if ready:
            key = self._take_key()
            self._last_tap_t = t_ms
            if target.kind == "slider":
                self._held_key = key
                self._state = PHASE_SLIDE
                self._slide_pos = (target.x, target.y)
                self._slide_vel = (0.0, 0.0)
                self._slide_last_seen_t = t_ms
            else:
                self._tap_key = key
                self._tap_until = t_ms + self.tap_hold_ms

        return self._finish(goal, t_ms, phase=PHASE_APPROACH,
                            target=(target.x, target.y),
                            ratio=target.approach_ratio, tth=tth)

    def _aim_offset(self, center: Vec, next_obj, hit_radius: float) -> Vec:
        """Shift aim toward next_obj within the hit circle of center."""
        if next_obj is None or hit_radius < 1e-3:
            return center
        dx = next_obj.x - center[0]
        dy = next_obj.y - center[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 1e-3:
            return center
        offset = min(hit_radius * self.aim_cut_fraction, dist * 0.5)
        nx, ny = dx / dist, dy / dist
        return (center[0] + nx * offset, center[1] + ny * offset)

    # ── slide ─────────────────────────────────────────────────────

    def _slide_predict(self, t_ms: float) -> Optional[Vec]:
        if self._slide_pos is None:
            return None
        elapsed = max(0.0, t_ms - self._slide_last_seen_t)
        px = self._slide_pos[0] + self._slide_vel[0] * elapsed
        py = self._slide_pos[1] + self._slide_vel[1] * elapsed
        return (max(0.0, min(512.0, px)), max(0.0, min(384.0, py)))

    def _do_slide(self, scene: Scene, t_ms: float) -> Optional[Reference]:
        predicted = self._slide_predict(t_ms) if self._state == PHASE_SLIDE else None
        ref_pt: Vec = predicted or self._slide_pos or self._cursor
        ball: Optional[Vec] = None
        best_d = self.slide_follow_radius_osu
        for o in scene.objects:
            if o.kind != "slider":
                continue
            if o.has_ball and o.ball_x is not None:
                d = _dist(ref_pt, (o.ball_x, o.ball_y))
                if d < best_d:
                    best_d, ball = d, (o.ball_x, o.ball_y)

        if ball is not None:
            if self._held_key is None:
                self._held_key = self._take_key()
            if self._slide_pos is not None:
                dt = t_ms - self._slide_last_seen_t
                if 1e-3 < dt <= self.slide_grace_ms + 50.0:
                    ivx = (ball[0] - self._slide_pos[0]) / dt
                    ivy = (ball[1] - self._slide_pos[1]) / dt
                    a = 0.5
                    self._slide_vel = (a * ivx + (1 - a) * self._slide_vel[0],
                                       a * ivy + (1 - a) * self._slide_vel[1])
            self._state = PHASE_SLIDE
            self._slide_pos = ball
            self._slide_last_seen_t = t_ms
            return self._finish(ball, t_ms, phase=PHASE_SLIDE, target=ball,
                                ratio=1.0)

        if self._state == PHASE_SLIDE:
            elapsed = t_ms - self._slide_last_seen_t
            if self._slide_pos is not None and elapsed <= self.slide_grace_ms:
                pred = predicted or self._slide_pos
                return self._finish(pred, t_ms, phase=PHASE_SLIDE,
                                    target=pred, ratio=1.0)
            self._held_key = None
            self._state = PHASE_APPROACH
            self._slide_pos = None
            self._slide_vel = (0.0, 0.0)
        return None

    # ── spin ──────────────────────────────────────────────────────

    def _do_spin(self, center: Vec, dt: float, t_ms: float) -> Reference:
        if self._state != PHASE_SPIN or self._held_key is None:
            self._held_key = self._take_key()
        self._state = PHASE_SPIN
        self._spin_angle += self.spin_speed * dt
        cx, cy = center
        target = (cx + self.spin_radius_osu * math.cos(self._spin_angle),
                  cy + self.spin_radius_osu * math.sin(self._spin_angle))
        return self._finish(target, t_ms, phase=PHASE_SPIN, target=target,
                            ratio=1.0)

    # ── internals ─────────────────────────────────────────────────

    def _take_key(self) -> int:
        key = self._next_key
        self._next_key = VK_X if key == VK_Z else VK_Z
        return key

    def _finish(
        self,
        goal: Vec,
        t_ms: float,
        phase: str = PHASE_IDLE,
        target: Optional[Vec] = None,
        ratio: float = 0.0,
        tth: float = 0.0,
    ) -> Reference:
        if self._tap_key is not None and t_ms >= self._tap_until:
            self._tap_key = None
        z = (self._held_key == VK_Z) or (self._tap_key == VK_Z)
        x = (self._held_key == VK_X) or (self._tap_key == VK_X)
        tx, ty = (target if target is not None else (None, None))
        return Reference(x=goal[0], y=goal[1], key_z=z, key_x=x, phase=phase,
                          target_x=tx, target_y=ty, approach_ratio=ratio,
                          time_to_hit_ms=tth)


def _dist(a: Vec, b: Vec) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
