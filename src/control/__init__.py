"""Deterministic vision-only controller package.

Replaces the behavioural-cloning action model with an explicit
track -> plan -> move pipeline:

* :mod:`src.control.tracker`   — online approach-preempt / time-to-hit estimate
* :mod:`src.control.planner`   — scene construction + target selection
* :mod:`src.control.motion`    — human-like (min-jerk / damped) motion
* :mod:`src.control.controller`— approach / slide / spin state machine
"""

from src.control.controller import Controller, ControlOutput
from src.control.motion import HumanMotion, MotionProfile

__all__ = ["Controller", "ControlOutput", "HumanMotion", "MotionProfile"]
