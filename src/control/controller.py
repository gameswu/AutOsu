"""
Vision-only motion controller.

Three layers, composed every frame::

    goal, keys = ReferenceController(scene)              # deterministic geometry
    v_ref      = seek_velocity(goal, cursor)             # deterministic, converges
    v_style    = MotionPolicy.residual(goal-features)    # learned, bounded (optional)
    v          = accel_limit(v_prev, v_ref + gate*v_style)  # kinematic smoothing
    cursor(t)  = cursor(t-1) + v * dt

* :class:`~src.control.reference.ReferenceController` produces, purely from
  vision, the navigation goal (circle / slider-head / slider-ball / spin point)
  and the key state (the hard constraints — taps and holds).
* :func:`~src.control.motion_net.seek_velocity` is a deterministic proportional
  seek that always points at the goal and decays on arrival, so the cursor is
  *guaranteed to converge* (no behavioural-cloning drift).
* :class:`~src.control.motion_net.MotionPolicy` adds an **optional**, bounded
  style residual for human-like motion. ``gate = 1 - approach_ratio`` fades it
  to ~0 at the tap instant and during slider/spinner contact, so accuracy is
  never sacrificed for style. With no trained weights the cursor runs on the
  pure seek (which already plays accurately).

Keys come straight from the reference (the motion layer only moves the cursor,
it never decides taps), so timing is owned by the deterministic layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from src.vision.detector import FrameDetections
from src.control.reference import ReferenceController, PHASE_IDLE, VK_Z, VK_X  # noqa: F401
from src.control.motion_net import (
    MotionPolicy,
    seek_velocity,
    limit_velocity_change,
    MAX_SPEED_OSU_PMS,
    MAX_ACCEL_OSU_PMS2,
    MAX_RESIDUAL_OSU_PMS,
    SEEK_TAU_MS,
)

Vec = Tuple[float, float]


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
        tap_hold_ms: float = 40.0,     # how long a tap key stays down
        tap_refractory_ms: float = 70.0,  # min gap between taps (anti double-hit)
        spin_radius_osu: float = 60.0,
        spin_speed: float = 0.025,     # rad per ms (~4 rev/s)
        slide_follow_radius_osu: float = 120.0,  # match a ball within this of the slide point
        max_speed_osu_pms: float = MAX_SPEED_OSU_PMS,  # deterministic seek speed cap
        max_accel_osu_pms2: float = MAX_ACCEL_OSU_PMS2,  # kinematic turn/launch smoothing
        seek_tau_ms: float = SEEK_TAU_MS,   # proportional seek time constant
        motion_net_path: Optional[str] = None,   # OPTIONAL learned style residual
        max_residual_osu_pms: float = MAX_RESIDUAL_OSU_PMS,  # style residual cap
        style_scale: float = 1.0,      # global style-residual gain
        slide_grace_ms: float = 90.0,  # keep holding a slider through brief ball dropouts
        spin_grace_ms: float = 120.0,  # keep spinning through brief spinner dropouts
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        self.max_speed = float(max_speed_osu_pms)
        self.max_accel = float(max_accel_osu_pms2)
        self.seek_tau_ms = float(seek_tau_ms)

        # Learned style residual (optional — inactive without weights).
        self.policy = MotionPolicy(
            motion_net_path,
            max_residual_osu_pms=max_residual_osu_pms,
            scale=style_scale,
            device=device,
        )

        self.reference = ReferenceController(
            hit_window=hit_window,
            tap_hold_ms=tap_hold_ms,
            tap_refractory_ms=tap_refractory_ms,
            spin_radius_osu=spin_radius_osu,
            spin_speed=spin_speed,
            slide_follow_radius_osu=slide_follow_radius_osu,
            slide_grace_ms=slide_grace_ms,
            spin_grace_ms=spin_grace_ms,
            seed=seed,
        )

        self._prev_cursor: Vec = (256.0, 192.0)
        self._vel: Vec = (0.0, 0.0)    # current cursor velocity, osu!px/ms
        self._last_t: Optional[float] = None

    @property
    def motion_net_active(self) -> bool:
        return self.policy.active

    def reset(self, cursor: Optional[Vec] = None) -> None:
        self.reference.reset(cursor)
        self._last_t = None
        self._vel = (0.0, 0.0)
        if cursor is not None:
            self._prev_cursor = cursor

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

        if ref.phase == PHASE_IDLE:
            # Nothing to navigate to — hold position and bleed off velocity.
            cmd_x, cmd_y = cursor
            self._vel = (0.0, 0.0)
        else:
            goal = (ref.x, ref.y)
            # Deterministic seek guarantees convergence to the goal.
            vx, vy = seek_velocity(goal, cursor, self.max_speed, self.seek_tau_ms)
            # Optional learned style residual, faded out near the tap / during
            # contact (ratio -> 1) so accuracy is preserved.
            if self.policy.active:
                rx, ry = self.policy.residual(ref, cursor, self._prev_cursor, dt_ms)
                gate = max(0.0, 1.0 - ref.approach_ratio)
                vx += gate * rx
                vy += gate * ry
            # Kinematic smoothing: cap the per-frame velocity change so the
            # cursor ramps up instead of snapping to top speed and sweeps
            # through corners instead of cutting them. Then enforce the speed
            # cap on the resulting velocity.
            max_dv = self.max_accel * max(dt_ms, 1e-3)
            vx, vy = limit_velocity_change(self._vel, (vx, vy), max_dv)
            speed = (vx * vx + vy * vy) ** 0.5
            if speed > self.max_speed and speed > 1e-9:
                s = self.max_speed / speed
                vx *= s
                vy *= s
            self._vel = (vx, vy)
            cmd_x = cursor[0] + vx * dt_ms
            cmd_y = cursor[1] + vy * dt_ms

        self._prev_cursor = cursor
        return ControlOutput(x=cmd_x, y=cmd_y,
                             key_z=ref.key_z, key_x=ref.key_x)
