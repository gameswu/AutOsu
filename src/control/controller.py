"""
Vision-only motion controller.

Two layers, composed every frame::

    goal, keys      = ReferenceController(scene)        # deterministic geometry
    cursor(t)       = cursor(t-1) + MotionPolicy.velocity(goal) * dt   # learned

* :class:`~src.control.reference.ReferenceController` produces, purely from
  vision, the navigation goal (circle / slider-head / slider-ball / spin point)
  and the key state (the hard constraints — taps and holds).
* :class:`~src.control.motion_net.MotionPolicy` produces the actual cursor
  motion toward that goal. The policy is **mandatory** (trained weights are
  required); there is no hand-coded motion fallback.

Keys come straight from the reference (the policy only moves the cursor, it
never decides taps), so timing is owned by the deterministic layer while all
human-like motion is owned by the learned policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from src.vision.detector import FrameDetections
from src.control.reference import ReferenceController, PHASE_IDLE, VK_Z, VK_X  # noqa: F401
from src.control.motion_net import MotionPolicy, MAX_SPEED_OSU_PMS

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
        motion_net_path: Optional[str] = None,   # REQUIRED learned policy weights
        max_speed_osu_pms: float = MAX_SPEED_OSU_PMS,  # cap on commanded speed
        speed_scale: float = 1.0,      # global speed gain
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        # Learned motion policy (mandatory — raises if weights are missing).
        self.policy = MotionPolicy(
            motion_net_path,
            max_speed_osu_pms=max_speed_osu_pms,
            scale=speed_scale,
            device=device,
        )

        self.reference = ReferenceController(
            hit_window=hit_window,
            tap_hold_ms=tap_hold_ms,
            tap_refractory_ms=tap_refractory_ms,
            spin_radius_osu=spin_radius_osu,
            spin_speed=spin_speed,
            slide_follow_radius_osu=slide_follow_radius_osu,
            seed=seed,
        )

        self._prev_cursor: Vec = (256.0, 192.0)
        self._last_t: Optional[float] = None

    @property
    def motion_net_active(self) -> bool:
        return self.policy.active

    def reset(self, cursor: Optional[Vec] = None) -> None:
        self.reference.reset(cursor)
        self._last_t = None
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
            # Nothing to navigate to — hold position.
            cmd_x, cmd_y = cursor
        else:
            vx, vy = self.policy.velocity(ref, cursor, self._prev_cursor, dt_ms)
            cmd_x = cursor[0] + vx * dt_ms
            cmd_y = cursor[1] + vy * dt_ms

        self._prev_cursor = cursor
        return ControlOutput(x=cmd_x, y=cmd_y,
                             key_z=ref.key_z, key_x=ref.key_x)
