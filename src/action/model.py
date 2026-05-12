"""
Action model — GRU-based policy network for behavioral cloning.

Takes a sequence of GameStateVectors and predicts ActionVectors.
Trained on (state, action) pairs extracted from .osr replays
aligned with .osu beatmap ground truth.

Architecture:
    state (133-dim) → Linear(133, 256) → GRU(256, hidden=256, 2 layers)
    → Linear(256, 4) → (dx, dy, key_z_logit, key_x_logit)

At runtime:
    - dx, dy are used directly as cursor displacement
    - key_z, key_x are passed through sigmoid and thresholded at 0.5
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.action.state import STATE_DIM, ACTION_DIM


class ActionModel(nn.Module):
    """
    GRU-based action prediction model.

    Input: (batch, seq_len, STATE_DIM)
    Output: (batch, seq_len, ACTION_DIM)
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(
        self,
        states: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            states: (batch, seq_len, state_dim)
            hidden: optional GRU hidden state (num_layers, batch, hidden_dim)

        Returns:
            actions: (batch, seq_len, action_dim)
                     [:, :, 0:2] = cursor dx/dy (normalised)
                     [:, :, 2:4] = key logits (apply sigmoid for probability)
            hidden: updated GRU hidden state
        """
        x = self.input_proj(states)       # (B, T, hidden)
        x, hidden = self.gru(x, hidden)   # (B, T, hidden)
        actions = self.output_head(x)     # (B, T, action_dim)
        return actions, hidden


class ActionModelInference:
    """
    Stateful inference wrapper for the action model.

    Maintains GRU hidden state across calls for real-time use.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model = ActionModel()
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self._hidden: Optional[torch.Tensor] = None

    def reset(self):
        """Reset hidden state (call at start of each beatmap)."""
        self._hidden = None

    @torch.no_grad()
    def predict(self, state_vector: np.ndarray) -> np.ndarray:
        """
        Predict action from a single state vector.

        Args:
            state_vector: (STATE_DIM,) numpy array

        Returns:
            (ACTION_DIM,) numpy array: [dx_norm, dy_norm, key_z_prob, key_x_prob]
        """
        # (1, 1, STATE_DIM)
        x = torch.from_numpy(state_vector).float().unsqueeze(0).unsqueeze(0)
        x = x.to(self.device)

        actions, self._hidden = self.model(x, self._hidden)
        action = actions[0, 0].cpu().numpy()  # (ACTION_DIM,)

        # Apply sigmoid to key logits
        action[2] = 1.0 / (1.0 + np.exp(-action[2]))
        action[3] = 1.0 / (1.0 + np.exp(-action[3]))

        return action
