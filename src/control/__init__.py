"""Vision-only motion controller package."""

from src.control.controller import Controller, ControlOutput
from src.control.reference import (
    ReferenceController, Reference, TargetInfo, build_targets,
    PHASE_IDLE, PHASE_APPROACH, PHASE_SLIDE, PHASE_SPIN,
)

__all__ = [
    "Controller",
    "ControlOutput",
    "ReferenceController",
    "Reference",
    "TargetInfo",
    "build_targets",
    "PHASE_IDLE",
    "PHASE_APPROACH",
    "PHASE_SLIDE",
    "PHASE_SPIN",
]
