"""
Learned residual layer for the CPRP controller.

The deterministic :class:`~src.control.reference.ReferenceController` already
satisfies every hard constraint (pass through the hit object, ride the slider
ball, stay on the spin circle). On top of it we add a small, **bounded**,
**phase-gated** learned offset so the cursor moves the way a human does without
ever sacrificing accuracy::

    cursor(t) = reference(t) + gate(phase) * residual(features(t))

* ``features`` are *reference-relative* (phase one-hot, time-to-hit, the target
  vector in the cursor frame, recent velocity) so the policy never has to learn
  the absolute geometry — only the human deviation from it.
* ``residual`` is a tiny MLP whose output is squashed with ``tanh`` and scaled
  to at most ``max_residual_osu`` osu!px, so it can never throw the cursor off
  the object.
* ``gate`` collapses the residual to ~0 at the precise tap / contact instants
  (it scales with ``1 - approach_ratio`` while approaching), so style is never
  traded for hits.

Training is fully offline / supervised (regress ``human - reference`` from
replays; see ``scripts/build_motion_dataset.py`` + ``scripts/train_motion.py``).
At runtime the policy is vision-only and, when **no weights are present**, it
returns a zero residual so the controller falls back to the pure deterministic
reference. ``torch`` is imported lazily so this module (and the rest of the
control package) import fine on machines without it.
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
#   [5:7] target vector in cursor frame (/ POS_NORM)
#   [7]   distance to target (/ POS_NORM)
#   [8:10] recent velocity (/ VEL_NORM)
FEATURE_DIM = 10

_POS_NORM = 256.0      # osu!px half-playfield, normalises offsets to ~[-1, 1]
_VEL_NORM = 40.0       # osu!px/frame, a brisk flick
_TTH_NORM_MS = 500.0   # ms; approach features saturate past half a second

# Hidden width of the residual MLP.
HIDDEN_DIM = 64


def build_features(ref: Reference, cursor: Vec, prev_cursor: Vec) -> List[float]:
    """Reference-relative feature vector (length :data:`FEATURE_DIM`).

    Identical at runtime and during offline dataset construction so the policy
    sees exactly what it was trained on.
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

    vx = (cursor[0] - prev_cursor[0]) / _VEL_NORM
    vy = (cursor[1] - prev_cursor[1]) / _VEL_NORM

    tth = max(0.0, min(1.0, ref.time_to_hit_ms / _TTH_NORM_MS))
    ratio = approach * ref.approach_ratio

    return [approach, slide, spin, ratio, tth, tvx, tvy, tdist, vx, vy]


def phase_gate(ref: Reference) -> float:
    """Scalar in [0, 1] shrinking the residual where accuracy matters most."""
    if ref.phase == PHASE_APPROACH:
        # Vanishes as the ring collapses onto the hit moment.
        return max(0.0, min(1.0, 1.0 - ref.approach_ratio))
    if ref.phase == PHASE_SLIDE:
        return 0.5
    if ref.phase == PHASE_SPIN:
        return 0.3
    return 0.0


def make_motion_net(in_dim: int = FEATURE_DIM, hidden: int = HIDDEN_DIM):
    """Build the residual MLP (lazy ``torch`` import).

    A deliberately small net: two hidden layers, ``tanh`` output in [-1, 1]
    (scaled to osu!px by the caller). Returns an ``nn.Module``.
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


class ResidualPolicy:
    """Runtime wrapper: load weights, predict a bounded, gated residual.

    When no weights load (path is ``None`` / missing, or ``torch`` is
    unavailable) the policy is *inactive* and returns a zero residual, so the
    controller emits the pure deterministic reference.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        max_residual_osu: float = 20.0,
        scale: float = 1.0,
        device: str = "cpu",
    ):
        self.max_residual_osu = float(max_residual_osu)
        self.scale = float(scale)
        self.device = device
        self._net = None
        self._torch = None
        if path:
            self.load(path)

    @property
    def active(self) -> bool:
        return self._net is not None

    def load(self, path: str) -> bool:
        """Load a trained residual net; return True on success."""
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            print(f"[ResidualPolicy] no weights at {p} — using deterministic reference")
            return False
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
            print(f"[ResidualPolicy] loaded motion net: {p}")
            return True
        except Exception as e:  # torch missing / bad checkpoint -> safe fallback
            print(f"[ResidualPolicy] failed to load {p} ({e}) — using deterministic reference")
            self._net = None
            return False

    def residual(self, ref: Reference, cursor: Vec, prev_cursor: Vec) -> Vec:
        """Gated, bounded (dx, dy) offset in osu!px (zero when inactive)."""
        if self._net is None:
            return (0.0, 0.0)
        gate = phase_gate(ref)
        if gate <= 0.0:
            return (0.0, 0.0)
        feats = build_features(ref, cursor, prev_cursor)
        torch = self._torch
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(0)
            out = self._net(x)[0]
            dx = float(out[0]) * self.max_residual_osu
            dy = float(out[1]) * self.max_residual_osu
        g = gate * self.scale
        return (dx * g, dy * g)
