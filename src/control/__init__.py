"""Vision-only motion controller package."""

from src.control.controller import Controller, ControlOutput
from src.control.reference import (
    ReferenceController, Reference, TargetInfo, build_targets,
    PHASE_IDLE, PHASE_APPROACH, PHASE_SLIDE, PHASE_SPIN,
)
from src.control.motion_net import (
    TrajectoryPolicy,
    TrajectoryPolicyError,
    build_cursor_features,
    build_target_features,
    arrival_safeguard,
    CURSOR_DIM,
    TARGET_DIM,
    MAX_TARGETS,
)

__all__ = [
    "Controller",
    "ControlOutput",
    "ReferenceController",
    "Reference",
    "TargetInfo",
    "build_targets",
    "TrajectoryPolicy",
    "TrajectoryPolicyError",
    "build_cursor_features",
    "build_target_features",
    "arrival_safeguard",
    "CURSOR_DIM",
    "TARGET_DIM",
    "MAX_TARGETS",
    "PHASE_IDLE",
    "PHASE_APPROACH",
    "PHASE_SLIDE",
    "PHASE_SPIN",
]
