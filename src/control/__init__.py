"""Vision-only motion controller package."""

from src.control.controller import Controller, ControlOutput
from src.control.reference import ReferenceController, Reference
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
