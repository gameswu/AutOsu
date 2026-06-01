"""Vision-only motion controller.

Per-frame composition::

    targets, keys, phase = reference(scene)
    w, v = motion_model(cursor, velocity, phase, targets)

    approach:  MPC replan — build the minimum-jerk + residual primitive from the
               current cursor state to the goal over the remaining ratio
               interval [τ_now, 1] and command the next position on it.
    slide:     v = arrival_safeguard(v, cursor, slider_ball, dt)   # reach in 1 frame
    spin:      v = spin_tangential(v, cursor, spinner_centre)      # orbit, no drift

Approach commands an **absolute position** (the primitive guarantees on-target
arrival); slide/spin integrate ``cursor += v·dt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from src.vision.detector import FrameDetections
from src.control.reference import (
    ReferenceController,
    PHASE_IDLE, PHASE_APPROACH, PHASE_SLIDE, PHASE_SPIN,
    VK_Z, VK_X,  # noqa: F401
)
from src.control.motion_net import (
    TrajectoryPolicy,
    eval_primitive,
    arrival_safeguard,
    spin_tangential,
)

Vec = Tuple[float, float]

# Robust τ-space velocity estimation (numeric-stability constants, not tuning):
# a single-frame Δτ ≈ dt/preempt is tiny and noisy, so dividing Δx by it blows
# up.  We instead take the secant over a short window of frames, giving a stable
# denominator.  The window is cleared on a segment change (ratio drop).
_VEL_WINDOW = 6        # frames pooled for the τ-velocity secant
_MIN_DTAU = 1e-2       # minimum ratio span before a velocity is trusted
_RATIO_RESET = 0.05    # ratio drop that signals a new object recycled the slot


@dataclass
class ControlOutput:
    x: float
    y: float
    key_z: bool = False
    key_x: bool = False


class Controller:
    def __init__(
        self,
        hit_window: float = 0.90,
        tap_hold_ms: float = 40.0,
        tap_refractory_ms: float = 70.0,
        slide_grace_ms: float = 90.0,
        spin_grace_ms: float = 120.0,
        motion_net_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.policy = TrajectoryPolicy(motion_net_path, device=device)

        self.reference = ReferenceController(
            hit_window=hit_window,
            tap_hold_ms=tap_hold_ms,
            tap_refractory_ms=tap_refractory_ms,
            slide_grace_ms=slide_grace_ms,
            spin_grace_ms=spin_grace_ms,
        )

        self._vel: Vec = (0.0, 0.0)
        self._last_t: Optional[float] = None
        # Short window of (tau, x, y) for the primary target, for a robust
        # τ-space velocity.  Cleared on segment change.
        self._vwin: list = []

    @property
    def motion_net_active(self) -> bool:
        return self.policy.active

    def reset(self, cursor: Optional[Vec] = None) -> None:
        self.reference.reset(cursor)
        self._last_t = None
        self._vel = (0.0, 0.0)
        self._vwin = []

    def update(
        self,
        detections: FrameDetections,
        cursor: Vec,
        t_ms: float,
        to_osu: Callable[[float, float], Tuple[float, float]],
    ) -> ControlOutput:
        dt_ms = (t_ms - self._last_t) if self._last_t is not None else 16.0
        self._last_t = t_ms

        ref = self.reference.update(detections, cursor, t_ms, to_osu)

        if ref.phase == PHASE_IDLE or not self.policy.active:
            self._vel = (0.0, 0.0)
            self._vwin = []
            return ControlOutput(x=cursor[0], y=cursor[1],
                                 key_z=ref.key_z, key_x=ref.key_x)

        tth_fn = self.reference.timing.time_to_hit_ms
        w, vel = self.policy.predict(
            cursor, self._vel, ref.phase, ref.targets, tth_fn,
        )

        if ref.phase == PHASE_APPROACH:
            out = self._approach(ref, cursor, dt_ms, w)
            if out is not None:
                return out
            # No approachable target this frame: hold.
            self._vel = (0.0, 0.0)
            self._vwin = []
            return ControlOutput(x=cursor[0], y=cursor[1],
                                 key_z=ref.key_z, key_x=ref.key_x)

        # ── slide / spin: velocity policy + zero-parameter geometry ──
        self._vwin = []
        vx, vy = vel
        if ref.phase == PHASE_SLIDE:
            ball = next((t for t in ref.targets
                         if t.is_active and t.kind == "slider"), None)
            if ball is not None:
                vx, vy = arrival_safeguard(
                    (vx, vy), cursor, (ball.x, ball.y), dt_ms,
                )
        elif ref.phase == PHASE_SPIN:
            center = next((t for t in ref.targets if t.kind == "spinner"), None)
            if center is not None:
                vx, vy = spin_tangential((vx, vy), cursor, (center.x, center.y))

        self._vel = (vx, vy)
        return ControlOutput(x=cursor[0] + vx * dt_ms, y=cursor[1] + vy * dt_ms,
                             key_z=ref.key_z, key_x=ref.key_x)

    # ── approach: per-frame MPC replan on the movement primitive ──────────────

    def _approach(self, ref, cursor: Vec, dt_ms: float, w) -> Optional[ControlOutput]:
        approach_targets = [t for t in ref.targets if not t.is_active]
        if not approach_targets:
            return None

        primary = max(approach_targets, key=lambda t: t.approach_ratio)
        tau0 = primary.approach_ratio

        # τ-space velocity v₀ = Δx / Δτ.  A single-frame Δτ is tiny and noisy
        # (division blows up → darting), so take the secant over a short window
        # of frames.  A ratio drop means the slot was recycled by a new object
        # → clear the window.
        if self._vwin and (tau0 - self._vwin[-1][0]) < -_RATIO_RESET:
            self._vwin = []
        self._vwin.append((tau0, cursor[0], cursor[1]))
        if len(self._vwin) > _VEL_WINDOW:
            self._vwin = self._vwin[-_VEL_WINDOW:]

        tau_f, xf, yf = self._vwin[0]
        tau_b, xb, yb = self._vwin[-1]
        span = tau_b - tau_f
        if len(self._vwin) >= 2 and span > _MIN_DTAU:
            v0 = ((xb - xf) / span, (yb - yf) / span)
            dtau_frame = span / (len(self._vwin) - 1)
        else:
            v0 = (0.0, 0.0)
            dtau_frame = 0.0

        # Flow aim: goal velocity points toward the next object (the visible
        # target with the next-highest ratio) at the incoming τ-speed.
        vg = (0.0, 0.0)
        others = [t for t in approach_targets if t is not primary]
        if others:
            nxt = max(others, key=lambda t: t.approach_ratio)
            dx, dy = nxt.x - primary.x, nxt.y - primary.y
            d = (dx * dx + dy * dy) ** 0.5
            if d > 1e-6:
                sp = (v0[0] * v0[0] + v0[1] * v0[1]) ** 0.5
                vg = (sp * dx / d, sp * dy / d)

        # Execute one MPC step: command the position one frame's progress ahead.
        tau_q = min(1.0, tau0 + max(dtau_frame, 0.0))
        cmd = eval_primitive(cursor, v0, (primary.x, primary.y), vg, tau0, w, tau_q)

        self._vel = ((cmd[0] - cursor[0]) / dt_ms, (cmd[1] - cursor[1]) / dt_ms)
        return ControlOutput(x=cmd[0], y=cmd[1],
                             key_z=ref.key_z, key_x=ref.key_x)
