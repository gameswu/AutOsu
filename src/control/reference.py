"""
Deterministic reference generator for the CPRP controller.

This is the *constraint-satisfying* layer of the Constraint-Projected Residual
Policy (CPRP)::

    cursor(t) = reference(t) + gate(phase) * residual(t)

``reference(t)`` is produced here, every frame, purely from the current vision
scene. It is the same explicit geometry the old monolithic controller used —
min-jerk reach to the most imminent circle / slider head, reactive slider-ball
following, and circular spinner sweep — and it guarantees the hard constraints
(pass through the hit object, stay on the ball / spin circle) on its own. A
learned residual (:mod:`src.control.motion_net`) is layered on top by
:class:`src.control.controller.Controller`; when no weights are present the
reference is emitted unchanged (min-jerk fallback).

The controller exposes two entry points:

* :meth:`ReferenceController.update` — build a :class:`Scene` from raw
  detections and step (the runtime path).
* :meth:`ReferenceController.step` — advance one frame from an already-built
  :class:`Scene` (reused offline by ``scripts/build_motion_dataset.py`` to
  reconstruct the reference from beatmap ground truth).

Each step returns a :class:`Reference` carrying the reference cursor target, the
key state, the current phase, and the reference-relative target features the
residual net consumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from src.vision.detector import FrameDetections
from src.control.tracker import TimingTracker
from src.control.motion import HumanMotion, MotionProfile
from src.control.planner import Scene, SceneObject, build_scene, select_target

Vec = Tuple[float, float]

# Key virtual codes (osu! default: Z = K1, X = K2)
VK_Z = 0x5A
VK_X = 0x58

# Phase labels (also used as residual-net feature one-hot order)
PHASE_IDLE = "idle"
PHASE_APPROACH = "approach"
PHASE_SLIDE = "slide"
PHASE_SPIN = "spin"


@dataclass
class Reference:
    """One frame of deterministic reference output + residual-net features."""
    x: float                     # reference cursor target, osu! coords
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
    """Deterministic approach / slide / spin reference generator."""

    def __init__(
        self,
        hit_window: float = 0.90,      # tap once ratio >= this
        hit_radius_osu: float = 80.0,  # cursor must be within this of target to tap
        tap_hold_ms: float = 40.0,     # how long a tap key stays down
        tap_refractory_ms: float = 70.0,  # min gap between taps (anti double-hit)
        spin_radius_osu: float = 60.0,
        spin_speed: float = 0.025,     # rad per ms (~4 rev/s)
        slide_lost_ms: float = 120.0,  # release slider key after ball lost this long
        slide_follow_radius_osu: float = 120.0,  # only follow a ball within this of the slide point
        motion_profile: Optional[MotionProfile] = None,
        seed: Optional[int] = None,
    ):
        self.hit_window = hit_window
        self.hit_radius_osu = hit_radius_osu
        self.tap_hold_ms = tap_hold_ms
        self.tap_refractory_ms = tap_refractory_ms
        self.spin_radius_osu = spin_radius_osu
        self.spin_speed = spin_speed
        self.slide_lost_ms = slide_lost_ms
        self.slide_follow_radius_osu = slide_follow_radius_osu

        prof = motion_profile or MotionProfile()
        self.tap_lead_ms = prof.tap_lead_ms
        self.timing = TimingTracker()
        self.motion = HumanMotion(
            jitter=prof.jitter,
            follow_alpha=prof.follow_alpha,
            overshoot=prof.overshoot,
            seed=seed,
        )

        self._state = PHASE_APPROACH    # approach | slide | spin
        self._next_key = VK_Z           # alternation
        self._held_key: Optional[int] = None   # slider hold
        self._tap_key: Optional[int] = None
        self._tap_until = 0.0
        self._last_tap_t = -1e9
        self._spin_angle = 0.0
        self._slide_last_seen = 0.0
        self._slide_pos: Optional[Vec] = None
        self._last_t: Optional[float] = None
        self._cursor: Vec = (256.0, 192.0)

    def reset(self, cursor: Optional[Vec] = None) -> None:
        self.timing.reset()
        self.motion.reset()
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

        # Follow an active slider ball whenever one is present near the slide
        # point, regardless of internal state. This salvages a missed head-tap
        # (head already faded, orphan ball visible) instead of ignoring it.
        out = self._do_slide(scene, t_ms)
        if out is not None:
            return out

        return self._do_approach(scene, t_ms)

    # ── regimes ───────────────────────────────────────────────────────────

    def _do_approach(self, scene: Scene, t_ms: float) -> Reference:
        target = select_target(scene)
        if target is None:
            return self._finish(self._cursor, t_ms, phase=PHASE_IDLE)

        goal: Vec = (target.x, target.y)
        tth = self.timing.time_to_hit_ms(target.approach_ratio)
        pos = self.motion.plan_point(self._cursor, goal, tth, t_ms)

        # Fire when the approach ring has (nearly) collapsed OR we are within the
        # baked human tap-lead window of the exact hit moment.
        timed = (target.approach_ratio >= self.hit_window
                 or (self.tap_lead_ms > 0.0 and tth <= self.tap_lead_ms))
        ready = (timed
                 and _dist(self._cursor, goal) <= self.hit_radius_osu
                 and (t_ms - self._last_tap_t) >= self.tap_refractory_ms)

        if ready:
            key = self._take_key()
            self._last_tap_t = t_ms
            if target.kind == "slider":
                # Hold the key and begin following the slider.
                self._held_key = key
                self._state = PHASE_SLIDE
                self._slide_last_seen = t_ms
                self._slide_pos = goal
            else:
                # Circle: brief tap.
                self._tap_key = key
                self._tap_until = t_ms + self.tap_hold_ms

        return self._finish(pos, t_ms, phase=PHASE_APPROACH, target=goal,
                            ratio=target.approach_ratio, tth=tth)

    def _do_slide(self, scene: Scene, t_ms: float) -> Optional[Reference]:
        # Find a followable ball within the continuity radius of the current
        # slide point. The distance ceiling is essential: without it we would
        # latch onto *any* visible slider ball (the next slider's, or a stray
        # detection) and never release the slide, skipping the following taps.
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
            # Best case: follow the detected ball / follow-circle. If we reached
            # the ball without a registered head-tap (missed tap, orphan ball),
            # grab a key now so the remaining slider ticks still count.
            if self._held_key is None:
                self._held_key = self._take_key()
            self._state = PHASE_SLIDE
            self._slide_last_seen = t_ms
            self._slide_pos = ball
            pos = self.motion.follow(self._cursor, ball)
            return self._finish(pos, t_ms, phase=PHASE_SLIDE, target=ball,
                                ratio=1.0)

        # No ball near the slide point this frame. Only hold through the grace
        # window if we are genuinely mid-slide; otherwise this is not a slide
        # and approach should take over.
        if self._state != PHASE_SLIDE:
            return None

        # Do NOT jump to the slider end — that skips the path. Hold the current
        # position with the key down for a short grace window, covering
        # transient ball occlusion / missed detections. Once that grace lapses
        # with no ball, treat the slider as ended.
        if (t_ms - self._slide_last_seen) <= self.slide_lost_ms:
            hold = self._slide_pos or self._cursor
            return self._finish(hold, t_ms, phase=PHASE_SLIDE, target=hold,
                                ratio=1.0)

        # Ball lost beyond grace — the slider has ended.
        self._held_key = None
        self._state = PHASE_APPROACH
        self._slide_pos = None
        return None

    def _do_spin(self, spinner: SceneObject, dt: float, t_ms: float) -> Reference:
        # Spinners only count rotations while a key is held (osu! ignores cursor
        # motion when no key is down), so grab a key on entry and hold it for
        # the whole spin.
        if self._state != PHASE_SPIN or self._held_key is None:
            self._held_key = self._take_key()
        self._state = PHASE_SPIN
        self._spin_angle += self.spin_speed * dt
        cx, cy = spinner.x, spinner.y
        target = (cx + self.spin_radius_osu * math.cos(self._spin_angle),
                  cy + self.spin_radius_osu * math.sin(self._spin_angle))
        pos = self.motion.follow(self._cursor, target, alpha=0.6)
        return self._finish(pos, t_ms, phase=PHASE_SPIN, target=target,
                            ratio=1.0)

    # ── key bookkeeping ───────────────────────────────────────────────────

    def _take_key(self) -> int:
        key = self._next_key
        self._next_key = VK_X if key == VK_Z else VK_Z
        return key

    def _finish(
        self,
        pos: Vec,
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
        return Reference(x=pos[0], y=pos[1], key_z=z, key_x=x, phase=phase,
                         target_x=tx, target_y=ty, approach_ratio=ratio,
                         time_to_hit_ms=tth)


def _dist(a: Vec, b: Vec) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
