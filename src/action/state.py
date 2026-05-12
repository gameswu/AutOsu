"""
Structured game state representation.

This is the interface between the vision system and the action model.
Both the real-time pipeline and the offline training data builder produce
this same format, ensuring consistency.

State vector per frame:
    - objects: list of visible objects sorted by approach_ratio descending
    - cursor: current cursor position + velocity
    - time_delta_ms: time since previous frame

Action vector per frame:
    - dx, dy: cursor displacement in osu! coordinates
    - key_z, key_x: key press probability / state
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


# Maximum visible objects kept in state (zero-padded if fewer)
MAX_OBJECTS = 16

# Per-object features: (class_onehot[5], x_norm, y_norm, approach_ratio) = 8
OBJECT_FEATURE_DIM = 8

# Cursor features: (x_norm, y_norm, vx_norm, vy_norm) = 4
CURSOR_FEATURE_DIM = 4

# Total: MAX_OBJECTS * 8 + 4 + 1 = 133
STATE_DIM = MAX_OBJECTS * OBJECT_FEATURE_DIM + CURSOR_FEATURE_DIM + 1

# Action: (dx_norm, dy_norm, key_z, key_x) = 4
ACTION_DIM = 4


@dataclass
class ObjectState:
    """A single detected object."""
    class_id: int       # 0=hitcircle, 1=slider_head, 2=slider_body, 3=slider_end, 4=spinner
    x: float            # osu! x (0-512)
    y: float            # osu! y (0-384)
    approach_ratio: float  # 0.0 (just appeared) → 1.0 (hit time)

    def to_feature(self) -> np.ndarray:
        feat = np.zeros(OBJECT_FEATURE_DIM, dtype=np.float32)
        if 0 <= self.class_id < 5:
            feat[self.class_id] = 1.0
        feat[5] = self.x / 512.0
        feat[6] = self.y / 384.0
        feat[7] = self.approach_ratio
        return feat


@dataclass
class GameStateVector:
    """Complete game state for one frame — input to the action model."""
    objects: List[ObjectState] = field(default_factory=list)
    cursor_x: float = 256.0
    cursor_y: float = 192.0
    cursor_vx: float = 0.0   # osu!px per ms
    cursor_vy: float = 0.0
    time_delta_ms: float = 0.0

    def to_numpy(self) -> np.ndarray:
        """Flatten to (STATE_DIM,) numpy vector."""
        sorted_objs = sorted(
            self.objects, key=lambda o: o.approach_ratio, reverse=True
        )[:MAX_OBJECTS]

        obj_feats = np.zeros((MAX_OBJECTS, OBJECT_FEATURE_DIM), dtype=np.float32)
        for i, obj in enumerate(sorted_objs):
            obj_feats[i] = obj.to_feature()

        cursor_feat = np.array([
            self.cursor_x / 512.0,
            self.cursor_y / 384.0,
            self.cursor_vx / 10.0,
            self.cursor_vy / 10.0,
        ], dtype=np.float32)

        time_feat = np.array([self.time_delta_ms / 50.0], dtype=np.float32)

        return np.concatenate([obj_feats.flatten(), cursor_feat, time_feat])


@dataclass
class ActionVector:
    """Action output for one timestep."""
    dx: float = 0.0     # cursor delta x (osu!px)
    dy: float = 0.0     # cursor delta y (osu!px)
    key_z: float = 0.0  # 0 or 1
    key_x: float = 0.0  # 0 or 1

    def to_numpy(self) -> np.ndarray:
        return np.array([
            self.dx / 50.0,    # normalise (typical max displacement ~50px per frame)
            self.dy / 50.0,
            self.key_z,
            self.key_x,
        ], dtype=np.float32)

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "ActionVector":
        return cls(
            dx=float(arr[0]) * 50.0,
            dy=float(arr[1]) * 50.0,
            key_z=float(arr[2]),
            key_x=float(arr[3]),
        )
