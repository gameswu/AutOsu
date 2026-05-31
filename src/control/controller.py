"""
CPRP controller — Constraint-Projected Residual Policy.

The runtime player composes two layers::

    cursor(t) = reference(t) + gate(phase) * residual(t)

* :class:`~src.control.reference.ReferenceController` produces a deterministic,
  constraint-satisfying reference (min-jerk reach + tap timing, reactive slider
  follow, spinner sweep) and the key state, purely from vision.
* :class:`~src.control.motion_net.ResidualPolicy` adds a small, bounded,
  phase-gated *learned* human offset on top of the reference cursor. With no
  trained weights it is inactive and the reference passes through unchanged.

Keys come straight from the reference (the residual only nudges the cursor, it
never decides taps), so accuracy and timing are owned by the deterministic
layer while the learned layer only shapes *how* the cursor travels.

This replaces the previous monolithic deterministic controller; that logic now
lives in :mod:`src.control.reference` as the reference layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from src.vision.detector import FrameDetections
from src.control.motion import MotionProfile
from src.control.reference import ReferenceController, VK_Z, VK_X  # noqa: F401
from src.control.motion_net import ResidualPolicy

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
        hit_radius_osu: float = 80.0,  # cursor must be within this of target to tap
        tap_hold_ms: float = 40.0,     # how long a tap key stays down
        tap_refractory_ms: float = 70.0,  # min gap between taps (anti double-hit)
        spin_radius_osu: float = 60.0,
        spin_speed: float = 0.025,     # rad per ms (~4 rev/s)
        slide_lost_ms: float = 120.0,  # release slider key after ball lost this long
        slide_follow_radius_osu: float = 120.0,  # only follow a ball within this of the slide point
        jitter: float = 1.2,
        motion_profile: Optional[MotionProfile] = None,
        motion_net_path: Optional[str] = None,   # learned residual weights
        max_residual_osu: float = 20.0,          # cap on the learned offset
        residual_scale: float = 1.0,             # global residual gain
        seed: Optional[int] = None,
    ):
        prof = motion_profile or MotionProfile(jitter=jitter)

        # Learned residual layer (inactive -> pure deterministic reference).
        self.residual = ResidualPolicy(
            motion_net_path,
            max_residual_osu=max_residual_osu,
            scale=residual_scale,
        )

        # When the net supplies human deviation, the reference should be a
        # *clean* constraint-satisfying path: drop the hand-made jitter /
        # overshoot so the two layers don't double-count style.
        if self.residual.active:
            ref_profile = MotionProfile(
                jitter=0.0, overshoot=0.0,
                follow_alpha=prof.follow_alpha, tap_lead_ms=prof.tap_lead_ms,
            )
        else:
            ref_profile = prof

        self.reference = ReferenceController(
            hit_window=hit_window,
            hit_radius_osu=hit_radius_osu,
            tap_hold_ms=tap_hold_ms,
            tap_refractory_ms=tap_refractory_ms,
            spin_radius_osu=spin_radius_osu,
            spin_speed=spin_speed,
            slide_lost_ms=slide_lost_ms,
            slide_follow_radius_osu=slide_follow_radius_osu,
            motion_profile=ref_profile,
            seed=seed,
        )

        self._prev_cursor: Vec = (256.0, 192.0)

    @property
    def motion_net_active(self) -> bool:
        return self.residual.active

    def reset(self, cursor: Optional[Vec] = None) -> None:
        self.reference.reset(cursor)
        if cursor is not None:
            self._prev_cursor = cursor

    def update(
        self,
        detections: FrameDetections,
        cursor: Vec,
        t_ms: float,
        to_osu: Callable[[float, float], Tuple[float, float]],
    ) -> ControlOutput:
        ref = self.reference.update(detections, cursor, t_ms, to_osu)
        dx, dy = self.residual.residual(ref, cursor, self._prev_cursor)
        self._prev_cursor = cursor
        return ControlOutput(
            x=ref.x + dx, y=ref.y + dy,
            key_z=ref.key_z, key_x=ref.key_x,
        )
