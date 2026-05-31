"""Vision-only motion controller package.

A deterministic navigation-goal generator + a deterministic goal-seek, with an
optional learned style residual on top::

    goal, keys = reference(scene)                          # deterministic geometry
    v_ref      = seek_velocity(goal, cursor)               # deterministic, converges
    v_style    = policy.residual(goal-features)            # learned, bounded (optional)
    cursor(t)  = cursor(t-1) + (v_ref + gate * v_style) * dt

* :mod:`src.control.tracker`    — online approach-preempt / time-to-hit estimate
* :mod:`src.control.planner`    — scene construction + target selection
* :mod:`src.control.motion`     — offline human-motion profile (analysis only)
* :mod:`src.control.reference`  — navigation goal + key-state generator
* :mod:`src.control.motion_net` — deterministic seek + optional learned style residual
* :mod:`src.control.controller` — composition (reference + seek + style)
"""

from src.control.controller import Controller, ControlOutput
from src.control.reference import ReferenceController, Reference
from src.control.motion import MotionProfile
from src.control.motion_net import (
    MotionPolicy,
    MotionPolicyError,
    seek_velocity,
    limit_velocity_change,
    build_features,
    FEATURE_DIM,
    MAX_SPEED_OSU_PMS,
    MAX_ACCEL_OSU_PMS2,
    MAX_RESIDUAL_OSU_PMS,
    SEEK_TAU_MS,
)

__all__ = [
    "Controller",
    "ControlOutput",
    "ReferenceController",
    "Reference",
    "MotionProfile",
    "MotionPolicy",
    "MotionPolicyError",
    "seek_velocity",
    "limit_velocity_change",
    "build_features",
    "FEATURE_DIM",
    "MAX_SPEED_OSU_PMS",
    "MAX_ACCEL_OSU_PMS2",
    "MAX_RESIDUAL_OSU_PMS",
    "SEEK_TAU_MS",
]
