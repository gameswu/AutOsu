"""
Motion layer — deterministic goal-seeking velocity + learned style residual.

The deterministic :class:`~src.control.reference.ReferenceController` no longer
*moves* the cursor. It only produces, every frame and purely from vision, the
**navigation goal** (the circle / slider-head / slider-ball / spin-orbit point)
and the **key state** (the hard constraints). The cursor motion is composed
here, in *velocity* space (so it is resampling-rate independent)::

    v_ref(t)   = seek_velocity(goal, cursor)              # guarantees convergence
    v_style(t) = MotionPolicy.residual(features(t))       # learned, bounded
    cursor(t)  = cursor(t-1) + (v_ref + gate * v_style) * dt

* **v_ref** — a simple proportional seek toward the goal, capped at a max speed.
  It always points at the goal and shrinks as the cursor arrives, so the cursor
  is *mathematically guaranteed* to converge (no covariate-shift drift). This is
  reactive/self-correcting, not a pre-baked trajectory.
* **v_style** — a small ``tanh``-bounded MLP that adds human-like deviation
  (curved approach, micro-tremor, hesitation) on top of the seek. It is
  **optional**: with no trained weights the cursor runs on the pure seek, which
  already plays accurately. ``gate = 1 - approach_ratio`` fades the style to ~0
  at the tap instant and during slider/spinner contact (ratio = 1), so accuracy
  is never sacrificed for style.

The features are goal-relative (phase one-hot, time-to-hit, the goal vector in
the cursor frame, recent velocity). Train offline with
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

# Deterministic seek: proportional time constant and the speed cap. v_ref =
# clamp((goal - cursor) / SEEK_TAU_MS, max_speed). Smaller tau = snappier.
SEEK_TAU_MS = 45.0
MAX_SPEED_OSU_PMS = 4.0          # cap on the deterministic seek speed (osu!px/ms)

# Learned style residual cap (osu!px/ms). The tanh output maps to
# [-MAX_RESIDUAL, MAX_RESIDUAL]; MUST match the value baked into the dataset.
MAX_RESIDUAL_OSU_PMS = 1.5

# Hidden width of the policy MLP.
HIDDEN_DIM = 64


def seek_velocity(goal: Vec, cursor: Vec,
                  max_speed: float = MAX_SPEED_OSU_PMS,
                  tau_ms: float = SEEK_TAU_MS) -> Vec:
    """Deterministic proportional seek toward ``goal`` (osu!px/ms).

    Always points at the goal and decays as the cursor arrives, so integrating
    ``cursor += seek_velocity(...) * dt`` is guaranteed to converge (critically
    damped, no overshoot). The magnitude is capped at ``max_speed``.
    """
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
    """Raised when a *provided* motion-policy weights path cannot be loaded."""


class MotionPolicy:
    """Optional learned style residual on top of the deterministic seek.

    Loads a trained MLP and predicts a bounded cursor-velocity *residual*
    (osu!px/ms) that the controller adds — gated — to the deterministic
    :func:`seek_velocity`. The policy is **optional**:

    * ``path`` is ``None`` / empty  -> inactive (no error); the controller runs
      on the pure deterministic seek, which already converges accurately.
    * ``path`` is given but missing / unreadable, or ``torch`` is unavailable
      -> :class:`MotionPolicyError` (a real misconfiguration, surfaced loudly).
    """

    def __init__(
        self,
        path: Optional[str],
        max_residual_osu_pms: float = MAX_RESIDUAL_OSU_PMS,
        scale: float = 1.0,
        device: str = "cpu",
    ):
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
            # Optional: no weights -> pure deterministic seek.
            self._net = None
            self._torch = None
            print("[MotionPolicy] no weights -> deterministic seek only "
                  "(no learned style)")
            return
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
            print(f"[MotionPolicy] loaded style residual: {p} "
                  f"(max_residual={self.max_residual} osu!px/ms)")
        except MotionPolicyError:
            raise
        except Exception as e:  # torch missing / bad checkpoint
            raise MotionPolicyError(
                f"failed to load motion policy {p}: {e}") from e

    def residual(self, ref: Reference, cursor: Vec, prev_cursor: Vec,
                 dt_ms: float) -> Vec:
        """Bounded style-residual velocity (osu!px/ms). (0, 0) if inactive."""
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
