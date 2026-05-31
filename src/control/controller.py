"""
Deterministic, vision-only osu! controller.

Replaces the behavioural-cloning action model. Each frame it consumes the
detector output (with approach ratios already set), decides what the player
should do, and emits a cursor target plus key state. It runs a small state
machine over three regimes:

* **approach / tap** — drive a min-jerk trajectory to the most imminent circle
  or slider head, arriving exactly at the hit moment, then tap (alternating
  Z / X).
* **slide** — once a slider head is hit, hold the key and reactively follow the
  detected slider ball (falling back to the slider end) until the slider ends.
* **spin** — while a spinner is on screen, sweep the cursor in a circle around
  its centre.

Geometry comes purely from vision (no beatmap parsing); timing comes from the
online :class:`TimingTracker`; motion smoothness from :class:`HumanMotion`.
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


@dataclass
class ControlOutput:
    x: float            # commanded cursor position, osu! coords
    y: float
    key_z: bool = False
    key_x: bool = False


class Controller:
    def __init__(
        self,
        hit_window: float = 0.90,      # tap once ratio >= this
        hit_radius_osu: float = 80.0,  # cursor must be within this of target to tap
        tap_hold_ms: float = 40.0,     # how long a tap key stays down
        tap_refractory_ms: float = 70.0,  # min gap between taps (anti double-hit)
        spin_radius_osu: float = 60.0,
        spin_speed: float = 0.025,     # rad per ms (~4 rev/s)
        slide_lost_ms: float = 120.0,  # release slider key after ball lost this long
        jitter: float = 1.2,
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

        # Motion profile (baked offline from real replays) tunes *how* the
        # cursor moves and *when* taps fire, without changing vision logic.
        prof = motion_profile or MotionProfile(jitter=jitter)
        self.tap_lead_ms = prof.tap_lead_ms
        self.timing = TimingTracker()
        self.motion = HumanMotion(
            jitter=prof.jitter,
            follow_alpha=prof.follow_alpha,
            overshoot=prof.overshoot,
            seed=seed,
        )

        self._state = "approach"        # approach | slide | spin
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
        self._state = "approach"
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

    # ── main entry ────────────────────────────────────────────────────────

    def update(
        self,
        detections: FrameDetections,
        cursor: Vec,
        t_ms: float,
        to_osu: Callable[[float, float], Tuple[float, float]],
    ) -> ControlOutput:
        self._cursor = cursor
        dt = (t_ms - self._last_t) if self._last_t is not None else 16.0
        self._last_t = t_ms

        scene = build_scene(detections, to_osu)
        self.timing.update(
            [{"x": o.x, "y": o.y, "ratio": o.approach_ratio}
             for o in scene.actionables],
            t_ms,
        )

        # Spinner takes over whenever present.
        if scene.spinner is not None:
            return self._do_spin(scene.spinner, dt, t_ms)

        if self._state == "spin":
            self._state = "approach"  # spinner gone — release the spin key
            self._held_key = None

        if self._state == "slide":
            out = self._do_slide(scene, t_ms)
            if out is not None:
                return out
            # slide finished -> fall through to approach

        return self._do_approach(scene, t_ms)

    # ── regimes ───────────────────────────────────────────────────────────

    def _do_approach(self, scene: Scene, t_ms: float) -> ControlOutput:
        target = select_target(scene)
        if target is None:
            return self._finish(self._cursor, t_ms)

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
                self._state = "slide"
                self._slide_last_seen = t_ms
                self._slide_pos = goal
            else:
                # Circle: brief tap.
                self._tap_key = key
                self._tap_until = t_ms + self.tap_hold_ms

        return self._finish(pos, t_ms)

    def _do_slide(self, scene: Scene, t_ms: float) -> Optional[ControlOutput]:
        # Find the active slider (one carrying a detected ball, else nearest).
        ball: Optional[Vec] = None
        best_d = 1e9
        for o in scene.objects:
            if o.kind != "slider":
                continue
            if o.has_ball and o.ball_x is not None:
                d = _dist(self._slide_pos or (o.ball_x, o.ball_y),
                          (o.ball_x, o.ball_y))
                if d < best_d:
                    best_d, ball = d, (o.ball_x, o.ball_y)

        if ball is not None:
            # Best case: follow the detected ball / follow-circle.
            self._slide_last_seen = t_ms
            self._slide_pos = ball
            pos = self.motion.follow(self._cursor, ball)
            return self._finish(pos, t_ms)

        # No ball this frame. Do NOT jump to the slider end — that skips the
        # path. Instead hold the current position with the key down for as long
        # as the slider is plausibly still active, i.e. while its body is still
        # on screen (plus a short grace for transient ball occlusion).
        if scene.slider_bodies:
            self._slide_last_seen = t_ms
            return self._finish(self._slide_pos or self._cursor, t_ms)

        if (t_ms - self._slide_last_seen) <= self.slide_lost_ms:
            return self._finish(self._slide_pos or self._cursor, t_ms)

        # Body gone and ball lost beyond grace — the slider has ended.
        self._held_key = None
        self._state = "approach"
        self._slide_pos = None
        return None

    def _do_spin(self, spinner: SceneObject, dt: float, t_ms: float) -> ControlOutput:
        # Spinners only count rotations while a key is held (osu! ignores cursor
        # motion when no key is down), so grab a key on entry and hold it for
        # the whole spin.
        if self._state != "spin" or self._held_key is None:
            self._held_key = self._take_key()
        self._state = "spin"
        self._spin_angle += self.spin_speed * dt
        cx, cy = spinner.x, spinner.y
        target = (cx + self.spin_radius_osu * math.cos(self._spin_angle),
                  cy + self.spin_radius_osu * math.sin(self._spin_angle))
        pos = self.motion.follow(self._cursor, target, alpha=0.6)
        return self._finish(pos, t_ms)

    # ── key bookkeeping ───────────────────────────────────────────────────

    def _take_key(self) -> int:
        key = self._next_key
        self._next_key = VK_X if key == VK_Z else VK_Z
        return key

    def _finish(self, pos: Vec, t_ms: float) -> ControlOutput:
        # Expire a finished tap.
        if self._tap_key is not None and t_ms >= self._tap_until:
            self._tap_key = None

        z = (self._held_key == VK_Z) or (self._tap_key == VK_Z)
        x = (self._held_key == VK_X) or (self._tap_key == VK_X)
        return ControlOutput(x=pos[0], y=pos[1], key_z=z, key_x=x)


def _dist(a: Vec, b: Vec) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
