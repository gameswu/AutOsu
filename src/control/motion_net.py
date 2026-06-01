"""Attention-based motion model — learned human-like cursor motion.

Architecture::

    cursor_state → cursor_encoder ──┐
    targets[]    → target_encoder   │  cross_attention → output_mlp → (w, v)
                   (shared)        ─┘

The model receives the current cursor state (position, velocity, phase) and a
variable-length set of visible targets.  Cross-attention lets the cursor
"query" the targets to decide which to prioritise.

The output is multi-task:

  * ``w`` — residual weights for the **approach movement primitive**.  The
    executed approach path is a minimum-jerk (quintic) boundary term — which
    alone guarantees on-time, on-target arrival — plus a learned residual
    ``Σⱼ wⱼ·sin(jπτ)`` that vanishes at both endpoints, so it shapes the
    *style* of the path without ever breaking the arrival guarantee.  The phase
    parameter τ is the observed approach_ratio (fps- and preempt-independent).
  * ``v`` — raw velocity (osu!px/ms) for the **slide / spin** phases, projected
    by zero-parameter geometric constraints (``arrival_safeguard`` onto the
    slider ball, ``spin_tangential`` about the spinner centre).

No tanh, no speed cap; magnitudes are learned from human replay data.

torch is imported lazily.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

Vec = Tuple[float, float]

# ── normalisation constants (numerical stability, not kinematic constraints) ──
_POS_NORM = 256.0       # half playfield width
_VEL_NORM = 1.5          # typical osu! cursor speed, osu!px/ms
_TTH_NORM = 500.0        # typical time-to-hit, ms

# ── architecture constants ────────────────────────────────────────────────────
CURSOR_DIM = 8           # cx, cy, vx, vy, phase_one_hot(4)
TARGET_DIM = 8           # dx, dy, tth, ratio, circle, slider, spinner, active
EMBED_DIM = 64
N_HEADS = 2
MLP_HIDDEN = 128
MAX_TARGETS = 16         # padding width for datasets
N_BASIS = 4              # residual basis functions per axis: sin(jπτ), j=1..N_BASIS
OUTPUT_DIM = 2 * N_BASIS + 2   # [residual weights (x,y) | slide/spin velocity (x,y)]


# ── feature builders ──────────────────────────────────────────────────────────

def build_cursor_features(cursor: Vec, velocity: Vec, phase: str) -> List[float]:
    """Cursor state vector (length CURSOR_DIM)."""
    from src.control.reference import PHASE_IDLE, PHASE_APPROACH, PHASE_SLIDE, PHASE_SPIN
    return [
        cursor[0] / _POS_NORM,
        cursor[1] / _POS_NORM,
        velocity[0] / _VEL_NORM,
        velocity[1] / _VEL_NORM,
        1.0 if phase == PHASE_IDLE else 0.0,
        1.0 if phase == PHASE_APPROACH else 0.0,
        1.0 if phase == PHASE_SLIDE else 0.0,
        1.0 if phase == PHASE_SPIN else 0.0,
    ]


def build_target_features(
    cursor: Vec,
    targets: list,
    tth_from_ratio: Optional[Callable[[float], float]] = None,
) -> Tuple[List[List[float]], List[bool]]:
    """Per-target feature vectors and validity mask.

    *tth_from_ratio*: ``approach_ratio -> tth_ms``.  At runtime pass
    ``timing_tracker.time_to_hit_ms``; for offline dataset building pass
    ``lambda r: (1 - r) * preempt_ms``.
    """
    feats: List[List[float]] = []
    mask: List[bool] = []
    for t in targets:
        dx = (t.x - cursor[0]) / _POS_NORM
        dy = (t.y - cursor[1]) / _POS_NORM
        tth = tth_from_ratio(t.approach_ratio) / _TTH_NORM if tth_from_ratio else 0.0
        feats.append([
            dx, dy, tth, t.approach_ratio,
            1.0 if t.kind == "circle" else 0.0,
            1.0 if t.kind == "slider" else 0.0,
            1.0 if t.kind == "spinner" else 0.0,
            1.0 if t.is_active else 0.0,
        ])
        mask.append(True)
    return feats, mask


# ── model factory ─────────────────────────────────────────────────────────────

def make_trajectory_model(
    cursor_dim: int = CURSOR_DIM,
    target_dim: int = TARGET_DIM,
    embed_dim: int = EMBED_DIM,
    n_heads: int = N_HEADS,
    mlp_hidden: int = MLP_HIDDEN,
):
    """Build the attention-based trajectory model (lazy torch import)."""
    import torch
    import torch.nn as nn

    class TrajectoryModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cursor_encoder = nn.Sequential(
                nn.Linear(cursor_dim, embed_dim),
                nn.ReLU(),
            )
            self.target_encoder = nn.Sequential(
                nn.Linear(target_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.idle_embed = nn.Parameter(torch.zeros(embed_dim))
            self.cross_attn = nn.MultiheadAttention(
                embed_dim, n_heads, batch_first=True,
            )
            self.output_mlp = nn.Sequential(
                nn.Linear(embed_dim + cursor_dim, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, OUTPUT_DIM),
            )

        def forward(self, cursor_features, target_features, target_mask):
            """
            Args:
                cursor_features: (B, cursor_dim)
                target_features: (B, N, target_dim) — zero-padded
                target_mask:     (B, N) — True = valid target
            Returns:
                out: (B, OUTPUT_DIM) — [residual weights (2*N_BASIS) | velocity (2)]
            """
            B = cursor_features.shape[0]
            cursor_embed = self.cursor_encoder(cursor_features)

            has_targets = target_mask.any(dim=-1)  # (B,)
            if not has_targets.any():
                attn_out = self.idle_embed.unsqueeze(0).expand(B, -1)
            else:
                target_embeds = self.target_encoder(target_features)
                query = cursor_embed.unsqueeze(1)
                # Unmask first key for no-target rows to prevent
                # softmax(all -inf) = NaN in MHA.
                no_tgt = ~has_targets
                if no_tgt.any():
                    safe_mask = target_mask.clone()
                    safe_mask[no_tgt, 0] = True
                    key_padding_mask = ~safe_mask
                else:
                    key_padding_mask = ~target_mask
                attn_out, _ = self.cross_attn(
                    query, target_embeds, target_embeds,
                    key_padding_mask=key_padding_mask,
                )
                attn_out = attn_out.squeeze(1)
                if no_tgt.any():
                    attn_out[no_tgt] = self.idle_embed

            combined = torch.cat([cursor_features, attn_out], dim=-1)
            return self.output_mlp(combined)

    return TrajectoryModel()


# ── runtime policy wrapper ────────────────────────────────────────────────────

class TrajectoryPolicyError(RuntimeError):
    """Weights path given but cannot be loaded."""


class TrajectoryPolicy:
    """Runtime wrapper — loads weights, provides ``predict(…) -> (vx, vy)``."""

    def __init__(self, path: Optional[str], device: str = "cpu"):
        self.device = device
        self._net = None
        self._torch = None
        self.load(path)

    @property
    def active(self) -> bool:
        return self._net is not None

    def load(self, path: Optional[str]) -> None:
        from pathlib import Path as _Path
        if not path:
            self._net = None
            self._torch = None
            print("[TrajectoryPolicy] no weights — model inactive")
            return
        p = _Path(path)
        if not p.exists():
            raise TrajectoryPolicyError(f"weights not found: {p}")
        try:
            import torch
            net = make_trajectory_model()
            state = torch.load(str(p), map_location=self.device, weights_only=True)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            net.load_state_dict(state)
            net.eval()
            net.to(self.device)
            self._torch = torch
            self._net = net
            print(f"[TrajectoryPolicy] loaded: {p}")
        except TrajectoryPolicyError:
            raise
        except Exception as e:
            raise TrajectoryPolicyError(f"failed to load {p}: {e}") from e

    def predict(
        self,
        cursor: Vec,
        velocity: Vec,
        phase: str,
        targets: list,
        tth_from_ratio: Optional[Callable[[float], float]] = None,
    ) -> Tuple[List[float], Vec]:
        """Return ``(residual_weights, velocity)``.

        ``residual_weights`` (length ``2*N_BASIS``) shapes the approach
        movement primitive; ``velocity`` (osu!px/ms) drives the slide/spin
        phases.  When no weights are loaded, returns zeros.
        """
        if self._net is None:
            return ([0.0] * (2 * N_BASIS), (0.0, 0.0))
        torch = self._torch
        cf = build_cursor_features(cursor, velocity, phase)
        tf, tm = build_target_features(cursor, targets, tth_from_ratio)

        with torch.no_grad():
            c = torch.tensor([cf], dtype=torch.float32, device=self.device)
            if tf:
                t = torch.tensor([tf], dtype=torch.float32, device=self.device)
                m = torch.tensor([tm], dtype=torch.bool, device=self.device)
            else:
                t = torch.zeros(1, 1, TARGET_DIM, dtype=torch.float32,
                                device=self.device)
                m = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
            out = self._net(c, t, m)[0]
            w = [float(x) for x in out[:2 * N_BASIS]]
            vel = (float(out[2 * N_BASIS]), float(out[2 * N_BASIS + 1]))
            return (w, vel)


# ── movement primitive (minimum-jerk boundary + endpoint-vanishing residual) ──
#
# The approach path over the remaining interval [τ₀, 1] is::
#
#     s(τ) = B(u; x₀,v₀,g,v_g) + Σⱼ wⱼ·sin(jπu),   u = (τ-τ₀)/(1-τ₀)
#
# B is the minimum-jerk quintic with boundary conditions s(τ₀)=x₀, s'(τ₀)=v₀,
# s(1)=g, s'(1)=v_g and the natural free-acceleration conditions s'''=0 at both
# ends.  Velocities are in τ-space (dx/dτ), so the path shape is independent of
# fps and of the preempt time.  The residual basis sin(jπu) vanishes at u=0 and
# u=1, so s(1)=g exactly — on-target arrival is guaranteed by construction,
# whatever the network outputs.  Zero hand-tuned parameters.


def _axis_primitive(x0: float, v0: float, g: float, vg: float,
                    L: float, w_axis: Sequence[float], u: float) -> float:
    """One-axis primitive position at normalised phase ``u`` ∈ [0, 1].

    ``L = 1 - τ₀`` reparametrises the remaining interval; τ-space velocities
    ``v0``/``vg`` are scaled to u-space by ``L``.
    """
    V0 = v0 * L
    Vg = vg * L
    c5 = (g - x0) - 0.5 * (V0 + Vg)
    c4 = -2.5 * c5
    c2 = (g - x0 - V0) + 1.5 * c5
    p = x0 + V0 * u + c2 * u * u + c4 * u ** 4 + c5 * u ** 5
    r = 0.0
    for k, wk in enumerate(w_axis):
        r += wk * math.sin((k + 1) * math.pi * u)
    return p + r


def eval_primitive(x0: Vec, v0: Vec, g: Vec, vg: Vec,
                   tau0: float, w: Sequence[float], tau: float) -> Vec:
    """Evaluate the approach primitive at phase ``tau`` (absolute position).

    *x0, v0*: current cursor position and τ-space velocity (dx/dτ).
    *g, vg*:  goal position and τ-space flow velocity (toward next object).
    *tau0*:   current approach_ratio; *tau*: query ratio (≥ tau0).
    *w*:      residual weights, length ``2*N_BASIS`` (x-axis then y-axis).
    """
    L = 1.0 - tau0
    if L < 1e-6:
        return (g[0], g[1])
    u = (tau - tau0) / L
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    wx = w[:N_BASIS]
    wy = w[N_BASIS:2 * N_BASIS]
    return (
        _axis_primitive(x0[0], v0[0], g[0], vg[0], L, wx, u),
        _axis_primitive(x0[1], v0[1], g[1], vg[1], L, wy, u),
    )


def primitive_path_torch(x0, v0, g, vg, tau0, w, taus):
    """Differentiable batched primitive for training.

    Shapes: x0/v0/g/vg (B,2); tau0 (B,1); w (B, 2*N_BASIS); taus (B,P).
    Returns positions (B, P, 2).
    """
    import torch
    L = (1.0 - tau0).clamp_min(1e-6)             # (B,1)
    u = ((taus - tau0) / L).clamp(0.0, 1.0)      # (B,P)
    K = N_BASIS
    k = torch.arange(1, K + 1, device=u.device, dtype=u.dtype)
    basis = torch.sin(u.unsqueeze(-1) * k.view(1, 1, K) * math.pi)  # (B,P,K)
    outs = []
    for a in range(2):
        x0a = x0[:, a:a + 1]
        V0 = v0[:, a:a + 1] * L
        ga = g[:, a:a + 1]
        Vg = vg[:, a:a + 1] * L
        c5 = (ga - x0a) - 0.5 * (V0 + Vg)
        c4 = -2.5 * c5
        c2 = (ga - x0a - V0) + 1.5 * c5
        p = x0a + V0 * u + c2 * u ** 2 + c4 * u ** 4 + c5 * u ** 5   # (B,P)
        wa = w[:, a * K:(a + 1) * K]                                 # (B,K)
        r = (basis * wa.unsqueeze(1)).sum(-1)                        # (B,P)
        outs.append(p + r)
    return torch.stack(outs, dim=-1)             # (B,P,2)


# ── arrival safeguard ─────────────────────────────────────────────────────────

def arrival_safeguard(v: Vec, cursor: Vec, target: Vec, tth_ms: float) -> Vec:
    """Ensure velocity makes sufficient progress toward *target* for on-time
    arrival.

    Only meaningful during approach phase.  If the model's velocity already
    projects enough speed toward the target, returns it unchanged.  Otherwise,
    adds the minimum radial correction.

    This is NOT a kinematic model — it is ``distance / time = speed``, the
    irreducible physics constraint.  No tau, no damping profile.
    """
    if tth_ms <= 1e-3:
        return v
    dx = target[0] - cursor[0]
    dy = target[1] - cursor[1]
    d = (dx * dx + dy * dy) ** 0.5
    if d < 1e-3:
        return v
    nx, ny = dx / d, dy / d
    needed = d / tth_ms
    v_toward = v[0] * nx + v[1] * ny
    if v_toward >= needed:
        return v
    correction = needed - v_toward
    return (v[0] + correction * nx, v[1] + correction * ny)


def spin_tangential(v: Vec, cursor: Vec, center: Vec) -> Vec:
    """Project velocity onto the tangent of the circle about *center*.

    Drops the radial component, keeping only the tangential (rotational) part.
    The model's learned angular speed is preserved; radial drift — the spiral
    collapse/blow-up that pure-model inference accumulates during a spinner — is
    removed.  The orbit radius stays fixed at whatever it was on spin entry.

    Zero parameters: pure geometry from the observed cursor/center.
    """
    rx = cursor[0] - center[0]
    ry = cursor[1] - center[1]
    r = (rx * rx + ry * ry) ** 0.5
    if r < 1e-3:
        return v          # at the centre: no defined tangent, let model act
    nx, ny = rx / r, ry / r
    v_radial = v[0] * nx + v[1] * ny
    return (v[0] - v_radial * nx, v[1] - v_radial * ny)
