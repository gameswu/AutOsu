"""
Deterministic *navigation* reference for the neural-motion controller.

This layer no longer moves the cursor. Every frame, purely from the current
vision scene, it emits two things:

* the **navigation goal** — the point the cursor should be heading to right now
  (the most imminent circle / slider head, the live slider ball, or the spin
  orbit point), in osu! coordinates, and
* the **key state** — the hard constraint (tap a circle, hold a slider / spin),
  decided from the approach ring.

The actual cursor motion toward that goal is produced by the
:mod:`src.control.motion_net` layer (a deterministic seek plus an optional
learned style residual). There is deliberately **no hand-coded motion model
here** — no min-jerk, jitter, overshoot, fixed reach time, or fixed dwell.

Two entry points:

* :meth:`ReferenceController.update` — build a :class:`Scene` from raw
  detections and step (the runtime path).
* :meth:`ReferenceController.step` — advance one frame from an already-built
  :class:`Scene` (reused offline by ``scripts/build_motion_dataset.py``).

Each step returns a :class:`Reference` carrying the navigation goal, key state,
phase, and the timing features the policy consumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from src.vision.detector import FrameDetections
from src.control.tracker import TimingTracker
from src.control.planner import Scene, SceneObject, build_scene, select_target

Vec = Tuple[float, float]

# Key virtual codes (osu! default: Z = K1, X = K2)
VK_Z = 0x5A
VK_X = 0x58

# Phase labels (also used as policy feature one-hot order)
PHASE_IDLE = "idle"
PHASE_APPROACH = "approach"
PHASE_SLIDE = "slide"
PHASE_SPIN = "spin"


@dataclass
class Reference:
    """One frame of navigation goal + key state + policy features."""
    x: float                     # navigation goal, osu! coords (mirror of target)
    y: float
    key_z: bool = False
    key_x: bool = False
    phase: str = PHASE_IDLE      # idle | approach | slide | spin
    # Active goal this frame (circle/head, ball, or spin orbit point), osu!.
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    approach_ratio: float = 0.0  # of the active target (1.0 during slide/spin)
    time_to_hit_ms: float = 0.0  # estimated ms to the tap (0 outside approach)


class ReferenceController:
    """Vision-only navigation-goal + key-state generator."""

    def __init__(
        self,
        hit_window: float = 0.90,      # tap once ratio >= this
        tap_hold_ms: float = 40.0,     # how long a tap key stays down
        tap_refractory_ms: float = 70.0,  # min gap between taps (anti double-hit)
        spin_radius_osu: float = 60.0,
        spin_speed: float = 0.025,     # rad per ms (~4 rev/s)
        slide_follow_radius_osu: float = 120.0,  # match a ball within this of the slide point
        seed: Optional[int] = None,    # unused (kept for call-site compatibility)
    ):
        self.hit_window = hit_window
        self.tap_hold_ms = tap_hold_ms
        self.tap_refractory_ms = tap_refractory_ms
        self.spin_radius_osu = spin_radius_osu
        self.spin_speed = spin_speed
        self.slide_follow_radius_osu = slide_follow_radius_osu

        self.timing = TimingTracker()

        self._state = PHASE_APPROACH    # approach | slide | spin
        self._next_key = VK_Z           # alternation
        self._held_key: Optional[int] = None   # slider / spin hold
        self._tap_key: Optional[int] = None
        self._tap_until = 0.0
        self._last_tap_t = -1e9
        self._spin_angle = 0.0
        self._slide_pos: Optional[Vec] = None
        self._last_t: Optional[float] = None
        self._cursor: Vec = (256.0, 192.0)

    def reset(self, cursor: Optional[Vec] = None) -> None:
        self.timing.reset()
        self._state = PHASE_APPROACH
        self._next_key = VK_Z
        self._held_key = None
        self._tap_key = None
        self._tap_until = 0.0
        self._last_tap_t = -1e9
        self._spin_angle = 0.0
        self._slide_pos = None
        self._last_t = None
        if cursor is not None:
            self._cursor = cursor

    # ── entry points ──────────────────────────────────────────────────────

    def update(
        self,
        detections: FrameDetections,
        cursor: Vec,
        t_ms: float,
        to_osu: Callable[[float, float], Tuple[float, float]],
    ) -> Reference:
        """Runtime path: build a scene from detections, then step."""
        scene = build_scene(detections, to_osu)
        return self.step(scene, cursor, t_ms)

    def step(self, scene: Scene, cursor: Vec, t_ms: float) -> Reference:
        """Advance one frame from an already-built scene."""
        self._cursor = cursor
        dt = (t_ms - self._last_t) if self._last_t is not None else 16.0
        self._last_t = t_ms

        self.timing.update(
            [{"x": o.x, "y": o.y, "ratio": o.approach_ratio}
             for o in scene.actionables],
            t_ms,
        )

        # Spinner takes over whenever present.
        if scene.spinner is not None:
            return self._do_spin(scene.spinner, dt, t_ms)

        if self._state == PHASE_SPIN:
            self._state = PHASE_APPROACH  # spinner gone — release the spin key
            self._held_key = None

        # Follow a live slider ball whenever one is present near the slide point,
        # regardless of internal state (salvages a missed head-tap).
        out = self._do_slide(scene, t_ms)
        if out is not None:
            return out

        return self._do_approach(scene, t_ms)

    # ── regimes ───────────────────────────────────────────────────────────

    def _do_approach(self, scene: Scene, t_ms: float) -> Reference:
        target = select_target(scene)
        if target is None:
            # Nothing to do — hold (the policy is skipped on idle).
            return self._finish(self._cursor, t_ms, phase=PHASE_IDLE)

        goal: Vec = (target.x, target.y)
        tth = self.timing.time_to_hit_ms(target.approach_ratio)

        # Tap purely on the vision ring; the policy owns where the cursor is.
        ready = (target.approach_ratio >= self.hit_window
                 and (t_ms - self._last_tap_t) >= self.tap_refractory_ms)
        if ready:
            key = self._take_key()
            self._last_tap_t = t_ms
            if target.kind == "slider":
                # Hold the key and begin following the slider.
                self._held_key = key
                self._state = PHASE_SLIDE
                self._slide_pos = goal
            else:
                self._tap_key = key
                self._tap_until = t_ms + self.tap_hold_ms

        return self._finish(goal, t_ms, phase=PHASE_APPROACH, target=goal,
                            ratio=target.approach_ratio, tth=tth)

    def _do_slide(self, scene: Scene, t_ms: float) -> Optional[Reference]:
        # Find a followable ball within the continuity radius of the slide point.
        # The distance ceiling stops us latching onto the next slider's ball.
        ref_pt: Vec = self._slide_pos or self._cursor
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
            # Follow the ball. If we reached it without a registered head-tap
            # (missed tap, orphan ball), grab a key so the ticks still count.
            if self._held_key is None:
                self._held_key = self._take_key()
            self._state = PHASE_SLIDE
            self._slide_pos = ball
            return self._finish(ball, t_ms, phase=PHASE_SLIDE, target=ball,
                                ratio=1.0)

        # No ball near the slide point. If the ball is genuinely gone the slider
        # has ended — release immediately and hand off to approach (no dwell). A
        # transient occlusion is recovered next frame when the ball reappears.
        if self._state == PHASE_SLIDE:
            self._held_key = None
            self._state = PHASE_APPROACH
            self._slide_pos = None
        return None

    def _do_spin(self, spinner: SceneObject, dt: float, t_ms: float) -> Reference:
        # Spinners only count rotations while a key is held; grab one on entry.
        if self._state != PHASE_SPIN or self._held_key is None:
            self._held_key = self._take_key()
        self._state = PHASE_SPIN
        self._spin_angle += self.spin_speed * dt
        cx, cy = spinner.x, spinner.y
        target = (cx + self.spin_radius_osu * math.cos(self._spin_angle),
                  cy + self.spin_radius_osu * math.sin(self._spin_angle))
        return self._finish(target, t_ms, phase=PHASE_SPIN, target=target,
                            ratio=1.0)

    # ── key bookkeeping ───────────────────────────────────────────────────

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
        # Expire a finished tap.
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
