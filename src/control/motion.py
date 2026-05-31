"""
Human-like motion generation.

Two regimes:

* :meth:`HumanMotion.plan_point` — move to a target that must be reached at a
  specific time (a circle / slider head). Uses a **minimum-jerk** velocity
  profile (the canonical model of human point-to-point reaching: smooth
  ease-in / ease-out, zero velocity at both ends) and refines the arrival time
  each frame as the timing estimate improves.

* :meth:`HumanMotion.follow` — reactively track a moving target (slider ball,
  spinner sweep) with a critically-damped low-pass, no fixed arrival time.

A small distance-tapered jitter is added so motion looks organic without
hurting accuracy near the target.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

Vec = Tuple[float, float]


def _dist(a: Vec, b: Vec) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _lerp(a: Vec, b: Vec, t: float) -> Vec:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def min_jerk(p: float) -> float:
    """Minimum-jerk easing: 6p^5 - 15p^4 + 10p^3 on [0, 1]."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return p * p * p * (10.0 + p * (-15.0 + 6.0 * p))


class HumanMotion:
    def __init__(
        self,
        jitter: float = 1.2,        # osu!px peak jitter amplitude (mid-move)
        retarget_dist: float = 40.0,  # osu!px; target jump that restarts a move
        follow_alpha: float = 0.45,   # low-pass gain for reactive following
        overshoot: float = 0.0,       # fractional overshoot past target near arrival
        seed: Optional[int] = None,
    ):
        self.jitter = jitter
        self.retarget_dist = retarget_dist
        self.follow_alpha = follow_alpha
        self.overshoot = overshoot
        self._rng = random.Random(seed)

        # Active point-move state
        self._start: Optional[Vec] = None
        self._start_t: float = 0.0
        self._target: Optional[Vec] = None

    def reset(self) -> None:
        self._start = None
        self._target = None

    # ── point-to-point (timed arrival) ────────────────────────────────────

    def plan_point(self, cur: Vec, target: Vec, time_to_hit_ms: float,
                   t_ms: float) -> Vec:
        """Next cursor position on a min-jerk path arriving at the hit time."""
        if (self._target is None or self._start is None
                or _dist(self._target, target) > self.retarget_dist):
            # New movement: anchor the start at the current cursor.
            self._start = cur
            self._start_t = t_ms
        self._target = target

        arrival_t = t_ms + max(0.0, time_to_hit_ms)
        total = arrival_t - self._start_t
        if total <= 1e-3:
            p = 1.0
        else:
            p = (t_ms - self._start_t) / total
        p = max(0.0, min(1.0, p))
        e = min_jerk(p)
        # Human-like overshoot: ease slightly past the target before arrival,
        # converging exactly back to it at p=1 (sin term is 0 at both ends, the
        # extra *p biases the bump toward the end of the move).
        if self.overshoot > 0.0:
            e = e + self.overshoot * math.sin(math.pi * p) * p
        pos = _lerp(self._start, target, e)
        # Jitter tapers to zero as we settle on the target (accuracy at the hit).
        return self._jittered(pos, taper=(1.0 - min(1.0, e)))

    # ── reactive following (no fixed arrival) ─────────────────────────────

    def follow(self, cur: Vec, target: Vec, alpha: Optional[float] = None) -> Vec:
        """Critically-damped tracking of a moving target (slider ball / spin)."""
        self._target = None  # invalidate any point-move so the next one restarts
        a = self.follow_alpha if alpha is None else alpha
        pos = _lerp(cur, target, a)
        return self._jittered(pos, taper=0.3)

    # ── helpers ───────────────────────────────────────────────────────────

    def _jittered(self, pos: Vec, taper: float) -> Vec:
        if self.jitter <= 0.0 or taper <= 0.0:
            return pos
        amp = self.jitter * taper
        return (pos[0] + self._rng.gauss(0.0, amp) * 0.5,
                pos[1] + self._rng.gauss(0.0, amp) * 0.5)


# ── motion profile (baked from osr analysis, applied at runtime) ───────────


@dataclass
class MotionProfile:
    """Human-motion parameters extracted offline from real replays
    (`scripts/analyze_motion.py`) and baked into the deterministic controller.
    Runtime stays purely vision-driven — this only tunes *how* the cursor moves.
    """
    jitter: float = 1.2          # osu!px tremor amplitude
    overshoot: float = 0.0       # fractional overshoot past target near arrival
    follow_alpha: float = 0.45   # slider-follow low-pass gain (higher = tighter)
    tap_lead_ms: float = 0.0     # tap this many ms before the exact hit moment

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
