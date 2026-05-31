"""
Offline human-motion profile.

The runtime controller's cursor motion is produced entirely by the learned
:mod:`src.control.motion_net` policy — there is no hand-coded motion model.
This module only holds :class:`MotionProfile`, a small container of human-motion
statistics extracted offline from real replays (``scripts/analyze_motion.py``)
for analysis / bookkeeping. It is not used to move the cursor at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MotionProfile:
    """Human-motion statistics extracted offline from real replays."""
    jitter: float = 1.2          # osu!px tremor amplitude
    overshoot: float = 0.0       # fractional overshoot past target near arrival
    follow_alpha: float = 0.45   # slider-follow low-pass gain (higher = tighter)
    tap_lead_ms: float = 0.0     # humans tap this many ms before the exact hit

    @classmethod
    def load(cls, path) -> "MotionProfile":
        """Load a profile from a YAML file, ignoring unknown keys."""
        import yaml
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        fields = {"jitter", "overshoot", "follow_alpha", "tap_lead_ms"}
        return cls(**{k: float(v) for k, v in data.items() if k in fields})
