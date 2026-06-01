"""Vision-only motion controller.

Composition per frame::

    goal, keys = reference(scene)                 # navigation + keys
    v_ref      = seek_velocity(goal, cursor)      # deterministic, converges
    v_style    = policy.residual(features)         # learned, bounded (optional)
    v          = accel_limit(v_prev, v_ref + gate*v_style)
    cursor(t)  = cursor(t-1) + v * dt
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
        spin_radius_osu: float = 60.0,
        spin_speed: float = 0.025,
        slide_follow_radius_osu: float = 120.0,
        slide_grace_ms: float = 90.0,
        spin_grace_ms: float = 120.0,
        max_speed_osu_pms: float = MAX_SPEED_OSU_PMS,
        max_accel_osu_pms2: float = MAX_ACCEL_OSU_PMS2,
        seek_tau_ms: float = SEEK_TAU_MS,
        motion_net_path: Optional[str] = None,
        max_residual_osu_pms: float = MAX_RESIDUAL_OSU_PMS,
        style_scale: float = 1.0,
        aim_cut_fraction: float = 0.65,
        lookahead_n: int = 3,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        self.max_speed = float(max_speed_osu_pms)
        self.max_accel = float(max_accel_osu_pms2)
        self.seek_tau_ms = float(seek_tau_ms)

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
            aim_cut_fraction=aim_cut_fraction,
            lookahead_n=lookahead_n,
            seed=seed,
        )

        self._prev_cursor: Vec = (256.0, 192.0)
        self._vel: Vec = (0.0, 0.0)
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
            cmd_x, cmd_y = cursor
            self._vel = (0.0, 0.0)
        else:
            goal = (ref.x, ref.y)
            vx, vy = seek_velocity(goal, cursor, self.max_speed, self.seek_tau_ms)

            if self.policy.active:
                rx, ry = self.policy.residual(ref, cursor, self._prev_cursor, dt_ms)
                gate = max(0.0, 1.0 - ref.approach_ratio)
                vx += gate * rx
                vy += gate * ry

            # Kinematic smoothing + speed cap.
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
