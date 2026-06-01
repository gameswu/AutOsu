"""Motion layer — deterministic seek + optional learned style residual.

Composition::

    v_ref   = seek_velocity(goal, cursor)
    v_style = MotionPolicy.residual(features)     (learned, bounded, optional)
    v       = accel_limit(v_prev, v_ref + gate * v_style)
    cursor += v * dt

torch is imported lazily.
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

# Feature layout (keep in sync with build_features / dataset):
#   [0:3]  phase one-hot  (approach, slide, spin)
#   [3]    approach_ratio
#   [4]    tth normalised
#   [5:7]  goal vector / POS_NORM
#   [7]    distance / POS_NORM
#   [8:10] velocity / VEL_NORM
FEATURE_DIM = 10

_POS_NORM = 256.0
_VEL_NORM_PMS = 1.5
_TTH_NORM_MS = 500.0

SEEK_TAU_MS = 45.0
MAX_SPEED_OSU_PMS = 3.0
MAX_ACCEL_OSU_PMS2 = 0.20
MAX_RESIDUAL_OSU_PMS = 1.5
HIDDEN_DIM = 64


def seek_velocity(goal: Vec, cursor: Vec,
                  max_speed: float = MAX_SPEED_OSU_PMS,
                  tau_ms: float = SEEK_TAU_MS) -> Vec:
    """Proportional seek toward *goal* (osu!px/ms), capped at *max_speed*."""
    dx = goal[0] - cursor[0]
    dy = goal[1] - cursor[1]
    tau = tau_ms if tau_ms > 1e-3 else SEEK_TAU_MS
    vx, vy = dx / tau, dy / tau
    speed = (vx * vx + vy * vy) ** 0.5
    if speed > max_speed and speed > 1e-9:
        s = max_speed / speed
        vx *= s
        vy *= s
    return (vx, vy)


def limit_velocity_change(v_prev: Vec, v_target: Vec, max_dv: float) -> Vec:
    """Slew-limit velocity change to *max_dv* (= max_accel * dt)."""
    dvx = v_target[0] - v_prev[0]
    dvy = v_target[1] - v_prev[1]
    mag = (dvx * dvx + dvy * dvy) ** 0.5
    if mag > max_dv and mag > 1e-9:
        s = max_dv / mag
        dvx *= s
        dvy *= s
    return (v_prev[0] + dvx, v_prev[1] + dvy)


def build_features(ref: Reference, cursor: Vec, prev_cursor: Vec,
                   dt_ms: float) -> List[float]:
    """Goal-relative feature vector (length FEATURE_DIM)."""
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
    """Build the policy MLP (lazy torch import). tanh output in [-1, 1]."""
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
    """Weights path given but cannot be loaded."""


class MotionPolicy:
    """Optional learned style residual.

    path=None/empty -> inactive (pure seek).
    path given but bad -> MotionPolicyError.
    """

    def __init__(self, path: Optional[str],
                 max_residual_osu_pms: float = MAX_RESIDUAL_OSU_PMS,
                 scale: float = 1.0, device: str = "cpu"):
        self.max_residual = float(max_residual_osu_pms)
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
            self._net = None
            self._torch = None
            print("[MotionPolicy] no weights -> deterministic seek only")
            return
        p = Path(path)
        if not p.exists():
            raise MotionPolicyError(f"weights not found: {p}")
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
            print(f"[MotionPolicy] loaded: {p}")
        except MotionPolicyError:
            raise
        except Exception as e:
            raise MotionPolicyError(f"failed to load {p}: {e}") from e

    def residual(self, ref: Reference, cursor: Vec, prev_cursor: Vec,
                 dt_ms: float) -> Vec:
        if self._net is None:
            return (0.0, 0.0)
        feats = build_features(ref, cursor, prev_cursor, dt_ms)
        torch = self._torch
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            out = self._net(x)[0]
            rx = float(out[0]) * self.max_residual * self.scale
            ry = float(out[1]) * self.max_residual * self.scale
        return (rx, ry)
