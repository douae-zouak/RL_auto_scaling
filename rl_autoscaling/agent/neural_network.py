"""
Dueling Double DQN Neural Network
===================================

Architecture
------------
    Input  (flat obs: T×F = 64 dims)
       │
       ▼
    FC(256) + ReLU
       │
       ▼
    FC(256) + ReLU
       │
    ┌──┴──────────────────┐
    │                     │
    ▼                     ▼
 Value stream         Advantage stream
 FC(128) + ReLU       FC(128) + ReLU
    │                     │
    ▼                     ▼
   V(s) ∈ ℝ¹          A(s,a) ∈ ℝ^|A|
    └─────────┬──────────┘
              │
              ▼
    Q(s,a) = V(s) + A(s,a) − mean_a A(s,a)

Weight initialisation: orthogonal (gain=√2) for feature layers,
                       smaller gain for output layers.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    """
    Dueling network architecture for Q-value estimation.

    Parameters
    ----------
    obs_dim    : flattened observation dimension (T × feature_dim)
    n_actions  : size of discrete action space
    hidden_dim : width of shared FC layers (default 256)
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        half           = hidden_dim // 2       # 128

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(obs_dim,    hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

        # Value stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, half), nn.ReLU(),
            nn.Linear(half,       1),             # scalar V(s)
        )

        # Advantage stream A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, half), nn.ReLU(),
            nn.Linear(half,       n_actions),     # per-action A(s,a)
        )

        self._init_weights()

    # weight initialisation

    def _init_weights(self):
        for m in self.modules():
            if not isinstance(m, nn.Linear):
                continue
            # Orthogonal init — better for RL than default Kaiming uniform
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

        # Slightly smaller gain on output layers for stable early training
        for stream in (self.value_stream, self.advantage_stream):
            last = [l for l in stream if isinstance(l, nn.Linear)][-1]
            nn.init.orthogonal_(last.weight, gain=0.01)

    # forward pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, obs_dim) float tensor

        Returns
        -------
        q : (batch, n_actions) Q-value tensor
        """
        phi = self.features(x)                            # (B, hidden)
        V   = self.value_stream(phi)                      # (B, 1)
        A   = self.advantage_stream(phi)                  # (B, |A|)

        # Dueling combination — subtract mean to keep identifiability
        Q = V + A - A.mean(dim=-1, keepdim=True)          # (B, |A|)
        return Q

    # convenience

# dire à PyTorch : “Ne calcule aucun gradient ici”
# Sans no_grad() : PyTorch construit le computation graph et consomme la mémoire inutilement
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Greedy action selection (no grad, eval mode)."""
        self.eval()
        q = self.forward(x)
        self.train()
        return q

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
