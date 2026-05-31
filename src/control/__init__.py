"""Vision-only CPRP controller package.

Constraint-Projected Residual Policy: a deterministic, constraint-satisfying
reference plus an optional learned, bounded, phase-gated residual::

    cursor(t) = reference(t) + gate(phase) * residual(t)

* :mod:`src.control.tracker`    — online approach-preempt / time-to-hit estimate
* :mod:`src.control.planner`    — scene construction + target selection
* :mod:`src.control.motion`     — human-like (min-jerk / damped) motion + profile
* :mod:`src.control.reference`  — deterministic approach / slide / spin reference
* :mod:`src.control.motion_net` — learned residual policy (fallback = zero)
* :mod:`src.control.controller` — CPRP composition (reference + residual)
"""

from src.control.controller import Controller, ControlOutput
from src.control.reference import ReferenceController, Reference
from src.control.motion import HumanMotion, MotionProfile
from src.control.motion_net import ResidualPolicy, build_features, FEATURE_DIM

__all__ = [
    "Controller",
    "ControlOutput",
    "ReferenceController",
    "Reference",
    "HumanMotion",
    "MotionProfile",
    "ResidualPolicy",
    "build_features",
    "FEATURE_DIM",
]
