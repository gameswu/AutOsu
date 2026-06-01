"""Attention-based trajectory model — learned human-like cursor motion.

Architecture::

    cursor_state → cursor_encoder ──┐
    targets[]    → target_encoder   │  cross_attention → output_mlp → (vx, vy)
                   (shared)        ─┘

The model receives the current cursor state (position, velocity, phase) and a
variable-length set of visible targets.  Cross-attention lets the cursor
"query" the targets to decide which to prioritise.  Output is raw velocity in
osu!px/ms — no tanh, no speed cap; the magnitude range is learned from human
replay data.

An arrival safeguard (not part of the network) ensures the cursor makes
sufficient progress toward the primary target during the approach phase.
This is the irreducible constraint ``distance / time = speed`` — not a
kinematic model.

torch is imported lazily.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

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
                nn.Linear(mlp_hidden, 2),
            )

        def forward(self, cursor_features, target_features, target_mask):
            """
            Args:
                cursor_features: (B, cursor_dim)
                target_features: (B, N, target_dim) — zero-padded
                target_mask:     (B, N) — True = valid target
            Returns:
                velocity: (B, 2) — osu!px/ms, unbounded
            """
            B = cursor_features.shape[0]
            cursor_embed = self.cursor_encoder(cursor_features)

            has_targets = target_mask.any(dim=-1)  # (B,)
            if not has_targets.any():
                attn_out = self.idle_embed.unsqueeze(0).expand(B, -1)
            else:
                target_embeds = self.target_encoder(target_features)
                query = cursor_embed.unsqueeze(1)
                key_padding_mask = ~target_mask
                attn_out, _ = self.cross_attn(
                    query, target_embeds, target_embeds,
                    key_padding_mask=key_padding_mask,
                )
                attn_out = attn_out.squeeze(1)
                no_tgt = ~has_targets
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
    ) -> Vec:
        if self._net is None:
            return (0.0, 0.0)
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
            return (float(out[0]), float(out[1]))


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
