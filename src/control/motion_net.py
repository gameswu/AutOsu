"""
Neural motion policy — the (mandatory) learned cursor driver.

The deterministic :class:`~src.control.reference.ReferenceController` no longer
*moves* the cursor. It only produces, every frame and purely from vision, the
**navigation goal** (the circle / slider-head / slider-ball / spin-orbit point)
and the **key state** (the hard constraints). All cursor motion — how fast and
along what curve the cursor travels toward that goal — is produced by this
learned policy::

    cursor(t) = cursor(t-1) + velocity(features(t)) * dt

There is **no hand-coded motion simulation** (no min-jerk, jitter, overshoot,
dwell or fixed reach times) and **no deterministic fallback**: the policy
requires trained weights. The network regresses the *human cursor velocity*
(osu!px per ms) conditioned on goal-relative features, so it is resampling-rate
independent — the same weights drive the cursor correctly at any runtime FPS.

The features are goal-relative (phase one-hot, time-to-hit, the goal vector in
the cursor frame, recent velocity) so the policy is largely self-correcting:
even if the cursor drifts, the goal vector pulls it back. Train offline with
``scripts/build_motion_dataset.py`` + ``scripts/train_motion.py``; ``torch`` is
imported lazily so this module imports fine on machines without it.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.control.reference import (
    Reference,
    PHASE_APPROACH,
    PHASE_SLIDE,
    PHASE_SPIN,
)

Vec = Tuple[float, float]

# Feature layout (keep in sync with build_features / the training dataset):
#   [0:3] phase one-hot   (approach, slide, spin)   -> idle = all zero
#   [3]   approach_ratio  (0 outside approach)
#   [4]   time_to_hit normalised  (clamp tth / TTH_NORM_MS)
#   [5:7] goal vector in cursor frame (/ POS_NORM)
#   [7]   distance to goal (/ POS_NORM)
#   [8:10] recent velocity, osu!px/ms (/ VEL_NORM_PMS)
FEATURE_DIM = 10

_POS_NORM = 256.0        # osu!px half-playfield, normalises offsets to ~[-1, 1]
_VEL_NORM_PMS = 1.5      # osu!px/ms, a brisk human flick
_TTH_NORM_MS = 500.0     # ms; approach features saturate past half a second

# Default cap on the commanded speed (osu!px per ms). The tanh output maps to
# [-MAX_SPEED, MAX_SPEED]; MUST match the value baked into the training dataset.
MAX_SPEED_OSU_PMS = 4.0

# Hidden width of the policy MLP.
HIDDEN_DIM = 64


def build_features(ref: Reference, cursor: Vec, prev_cursor: Vec,
                   dt_ms: float) -> List[float]:
    """Goal-relative feature vector (length :data:`FEATURE_DIM`).

    Identical at runtime and during offline dataset construction so the policy
    sees exactly what it was trained on. ``dt_ms`` is the time since the
    previous frame, used to express velocity in osu!px/ms (FPS-independent).
    """
    approach = 1.0 if ref.phase == PHASE_APPROACH else 0.0
    slide = 1.0 if ref.phase == PHASE_SLIDE else 0.0
    spin = 1.0 if ref.phase == PHASE_SPIN else 0.0

    if ref.target_x is not None and ref.target_y is not None:
        tvx = (ref.target_x - cursor[0]) / _POS_NORM
        tvy = (ref.target_y - cursor[1]) / _POS_NORM
        tdist = (tvx * tvx + tvy * tvy) ** 0.5
    else:
        tvx = tvy = tdist = 0.0

    dt = dt_ms if dt_ms > 1e-3 else 16.0
    vx = ((cursor[0] - prev_cursor[0]) / dt) / _VEL_NORM_PMS
    vy = ((cursor[1] - prev_cursor[1]) / dt) / _VEL_NORM_PMS

    tth = max(0.0, min(1.0, ref.time_to_hit_ms / _TTH_NORM_MS))
    ratio = approach * ref.approach_ratio

    return [approach, slide, spin, ratio, tth, tvx, tvy, tdist, vx, vy]


def make_motion_net(in_dim: int = FEATURE_DIM, hidden: int = HIDDEN_DIM):
    """Build the policy MLP (lazy ``torch`` import).

    Two hidden layers, ``tanh`` output in [-1, 1] (scaled to osu!px/ms by the
    caller). Returns an ``nn.Module``.
    """
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 2),
        nn.Tanh(),
    )


class MotionPolicyError(RuntimeError):
    """Raised when the mandatory motion-policy weights cannot be loaded."""


class MotionPolicy:
    """Runtime wrapper: load weights, predict a bounded cursor velocity.

    Weights are **mandatory** — construction raises :class:`MotionPolicyError`
    if the path is missing / unreadable or ``torch`` is unavailable. There is no
    deterministic motion fallback by design.
    """

    def __init__(
        self,
        path: Optional[str],
        max_speed_osu_pms: float = MAX_SPEED_OSU_PMS,
        scale: float = 1.0,
        device: str = "cpu",
    ):
        self.max_speed = float(max_speed_osu_pms)
        self.scale = float(scale)
        self.device = device
        self._net = None
        self._torch = None
        self.load(path)

    @property
    def active(self) -> bool:
        return self._net is not None

    def load(self, path: Optional[str]) -> None:
        from pathlib import Path
        if not path:
            raise MotionPolicyError(
                "motion_net_path is required: a trained motion policy must be "
                "provided (train one with scripts/train_motion.py). There is no "
                "deterministic motion fallback.")
        p = Path(path)
        if not p.exists():
            raise MotionPolicyError(f"motion policy weights not found at {p}")
        try:
            import torch
            net = make_motion_net()
            state = torch.load(str(p), map_location=self.device)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            net.load_state_dict(state)
            net.eval()
            net.to(self.device)
            self._torch = torch
            self._net = net
            print(f"[MotionPolicy] loaded motion net: {p} "
                  f"(max_speed={self.max_speed} osu!px/ms)")
        except MotionPolicyError:
            raise
        except Exception as e:  # torch missing / bad checkpoint
            raise MotionPolicyError(
                f"failed to load motion policy {p}: {e}") from e

    def velocity(self, ref: Reference, cursor: Vec, prev_cursor: Vec,
                 dt_ms: float) -> Vec:
        """Commanded cursor velocity (osu!px/ms) for this frame."""
        feats = build_features(ref, cursor, prev_cursor, dt_ms)
        torch = self._torch
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            out = self._net(x)[0]
            vx = float(out[0]) * self.max_speed * self.scale
            vy = float(out[1]) * self.max_speed * self.scale
        return (vx, vy)
