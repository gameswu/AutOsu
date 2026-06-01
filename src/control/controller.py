"""Vision-only scripted motion controller.

Cursor motion is fully rule-based (no learned model).  Each frame the reference
FSM supplies the phase, the visible targets and the key state; this controller
turns that into an absolute cursor position:

    approach:  timed arrival — step toward the primary target so the cursor
               lands on it exactly when its approach ring closes
               (``step = distance · dt / time_to_hit``), plus a smoothly varying
               random perpendicular perturbation that decays to zero on arrival
               so aim accuracy is preserved while the path looks human.
    slide:     snap onto the visible slider ball (follow-circle tracking).
    spin:      orbit the spinner centre at a fixed radius / angular speed.

All commands are absolute positions; the pipeline feeds the realised cursor back
as next frame's input, which makes the timed-arrival step self-correcting.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from src.vision.detector import FrameDetections
from src.control.reference import (
    ReferenceController,
    PHASE_IDLE, PHASE_APPROACH, PHASE_SLIDE, PHASE_SPIN,
    VK_Z, VK_X,  # noqa: F401
)

Vec = Tuple[float, float]


@dataclass
class ControlOutput:
    x: float
    y: float
    key_z: bool = False
    key_x: bool = False


class Controller:
    """Scripted cursor controller driven by the reference FSM."""

    def __init__(
        self,
        hit_window: float = 0.90,
        tap_hold_ms: float = 40.0,
        tap_refractory_ms: float = 70.0,
        slide_grace_ms: float = 90.0,
        spin_grace_ms: float = 120.0,
        # ── scripted-motion knobs ──────────────────────────────────────────
        path_noise: float = 0.18,        # perpendicular wander, frac of distance
        noise_smooth: float = 0.90,      # 0=jittery, →1=slow smooth wander
        spin_radius: float = 60.0,       # osu!px orbit radius during a spinner
        spin_speed: float = 0.025,       # rad/ms angular speed during a spinner
        seed: Optional[int] = None,
    ):
        self.path_noise = path_noise
        self.noise_smooth = noise_smooth
        self.spin_radius = spin_radius
        self.spin_speed = spin_speed
        self._rng = random.Random(seed)

        self.reference = ReferenceController(
            hit_window=hit_window,
            tap_hold_ms=tap_hold_ms,
            tap_refractory_ms=tap_refractory_ms,
            slide_grace_ms=slide_grace_ms,
            spin_grace_ms=spin_grace_ms,
        )

        self._last_t: Optional[float] = None
        self._pert = 0.0                      # filtered perpendicular wander
        self._spin_angle: Optional[float] = None

    # Scripted motion is always available.
    @property
    def active(self) -> bool:
        return True

    def reset(self, cursor: Optional[Vec] = None) -> None:
        self.reference.reset(cursor)
        self._last_t = None
        self._pert = 0.0
        self._spin_angle = None

    def update(
        self,
        detections: FrameDetections,
        cursor: Vec,
        t_ms: float,
        to_osu: Callable[[float, float], Tuple[float, float]],
    ) -> ControlOutput:
        dt_ms = (t_ms - self._last_t) if self._last_t is not None else 16.0
        if dt_ms <= 1e-3:
            dt_ms = 16.0
        self._last_t = t_ms

        ref = self.reference.update(detections, cursor, t_ms, to_osu)

        if ref.phase != PHASE_SPIN:
            self._spin_angle = None

        if ref.phase == PHASE_APPROACH:
            x, y = self._approach(ref, cursor, dt_ms)
        elif ref.phase == PHASE_SLIDE:
            x, y = self._slide(ref, cursor)
        elif ref.phase == PHASE_SPIN:
            x, y = self._spin(ref, cursor, dt_ms)
        else:  # PHASE_IDLE
            x, y = cursor

        return ControlOutput(x=x, y=y, key_z=ref.key_z, key_x=ref.key_x)

    # ── approach: timed arrival + decaying random perturbation ────────────────

    def _approach(self, ref, cursor: Vec, dt_ms: float) -> Vec:
        approach_targets = [t for t in ref.targets if not t.is_active]
        if not approach_targets:
            self._pert = 0.0
            return cursor

        primary = max(approach_targets, key=lambda t: t.approach_ratio)
        gx, gy = primary.x, primary.y
        dx, dy = gx - cursor[0], gy - cursor[1]
        d = math.hypot(dx, dy)
        if d < 1e-3:
            return (gx, gy)

        # Timed arrival: cover a fraction dt/tth of the remaining distance so the
        # cursor reaches the target right as the ring closes.  Self-correcting
        # because the realised cursor is fed back each frame.
        tth = self.reference.timing.time_to_hit_ms(primary.approach_ratio)
        tth = max(tth, dt_ms)
        frac = min(1.0, dt_ms / tth)
        nx, ny = dx / d, dy / d
        px = cursor[0] + dx * frac
        py = cursor[1] + dy * frac

        # Random perpendicular wander (filtered random walk) scaled by remaining
        # distance and (1 - ratio): it vanishes as the ring closes, so arrival
        # accuracy is never compromised.
        self._pert = (self.noise_smooth * self._pert
                      + (1.0 - self.noise_smooth) * self._rng.uniform(-1.0, 1.0))
        wander = self.path_noise * self._pert * d * (1.0 - primary.approach_ratio)
        px += -ny * wander
        py += nx * wander
        return (px, py)

    # ── slide: track the visible slider ball ──────────────────────────────────

    def _slide(self, ref, cursor: Vec) -> Vec:
        ball = next((t for t in ref.targets
                     if t.is_active and t.kind == "slider"), None)
        if ball is not None:
            return (ball.x, ball.y)
        return cursor  # body-only grace window: hold position

    # ── spin: orbit the spinner centre ────────────────────────────────────────

    def _spin(self, ref, cursor: Vec, dt_ms: float) -> Vec:
        center = next((t for t in ref.targets if t.kind == "spinner"), None)
        cx, cy = (center.x, center.y) if center is not None else (256.0, 192.0)
        if self._spin_angle is None:
            self._spin_angle = math.atan2(cursor[1] - cy, cursor[0] - cx)
        self._spin_angle += self.spin_speed * dt_ms
        return (cx + self.spin_radius * math.cos(self._spin_angle),
                cy + self.spin_radius * math.sin(self._spin_angle))
