"""Vision-only motion controller.

Per-frame composition::

    targets, keys, phase = reference(scene)
    v = trajectory_model(cursor, velocity, phase, targets)
    v = arrival_safeguard(v, cursor, primary_target, tth)   # approach only
    cursor(t) = cursor(t-1) + v * dt
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from src.vision.detector import FrameDetections
from src.control.reference import ReferenceController, PHASE_IDLE, PHASE_APPROACH, VK_Z, VK_X  # noqa: F401
from src.control.motion_net import (
    TrajectoryPolicy,
    arrival_safeguard,
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

    @property
    def motion_net_active(self) -> bool:
        return self.policy.active

    def reset(self, cursor: Optional[Vec] = None) -> None:
        self.reference.reset(cursor)
        self._last_t = None
        self._vel = (0.0, 0.0)

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
            return ControlOutput(x=cursor[0], y=cursor[1],
                                 key_z=ref.key_z, key_x=ref.key_x)

        tth_fn = self.reference.timing.time_to_hit_ms
        vx, vy = self.policy.predict(
            cursor, self._vel, ref.phase, ref.targets, tth_fn,
        )

        # Arrival safeguard: approach phase only, on the most urgent target.
        if ref.phase == PHASE_APPROACH and ref.targets:
            approach_targets = [t for t in ref.targets if not t.is_active]
            if approach_targets:
                primary = max(approach_targets, key=lambda t: t.approach_ratio)
                tth = tth_fn(primary.approach_ratio)
                vx, vy = arrival_safeguard(
                    (vx, vy), cursor, (primary.x, primary.y), tth,
                )

        self._vel = (vx, vy)
        cmd_x = cursor[0] + vx * dt_ms
        cmd_y = cursor[1] + vy * dt_ms

        return ControlOutput(x=cmd_x, y=cmd_y,
                             key_z=ref.key_z, key_x=ref.key_x)
