"""Vision-only motion controller package.

A deterministic navigation-goal generator plus a mandatory learned motion
policy::

    goal, keys = reference(scene)                       # deterministic geometry
    cursor(t)  = cursor(t-1) + policy.velocity(goal) * dt   # learned motion

* :mod:`src.control.tracker`    — online approach-preempt / time-to-hit estimate
* :mod:`src.control.planner`    — scene construction + target selection
* :mod:`src.control.motion`     — offline human-motion profile (analysis only)
* :mod:`src.control.reference`  — navigation goal + key-state generator
* :mod:`src.control.motion_net` — mandatory learned motion policy
* :mod:`src.control.controller` — composition (reference + policy)
"""

from src.control.controller import Controller, ControlOutput
from src.control.reference import ReferenceController, Reference
from src.control.motion import MotionProfile
from src.control.motion_net import (
    MotionPolicy,
    MotionPolicyError,
    build_features,
    FEATURE_DIM,
    MAX_SPEED_OSU_PMS,
)

__all__ = [
    "Controller",
    "ControlOutput",
    "ReferenceController",
    "Reference",
    "MotionProfile",
    "MotionPolicy",
    "MotionPolicyError",
    "build_features",
    "FEATURE_DIM",
    "MAX_SPEED_OSU_PMS",
]
