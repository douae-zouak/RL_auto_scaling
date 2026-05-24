"""
DQN Agent — Dueling Double DQN with Prioritized Experience Replay
=================================================================

Key algorithmic choices
-----------------------
• Double DQN        : online net selects action, target net evaluates it
                      →  r + γ · Q_target(s', argmax_a Q_online(s', a))
• Dueling network   : V(s) + A(s,a) – mean(A)  →  better value estimation
• PER buffer        : samples high-TD-error transitions more often
• Soft target update: θ_target ← τ·θ_online + (1–τ)·θ_target  (τ = 0.005)
• Epsilon-greedy    : ε decays exponentially from 1.0 → 0.01
• Huber loss        : less sensitive to outlier TD errors than MSE
• Gradient clipping : max-norm = 10 for training stability
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Optional

from .neural_network import DuelingDQN
from .per_buffer import PrioritizedReplayBuffer


class DQNAgent:
    """
    Full Dueling Double DQN agent with PER.

    Parameters
    ----------
    config : Config dataclass containing agent / env / training sub-configs
    """

    def __init__(self, config):
        self.config = config
        ac = config.agent
        ec = config.env                                        

        # device
        self.device = (torch.device("cuda")
                       if torch.cuda.is_available()
                       else torch.device("cpu"))

        # networks
        obs_dim = config.obs_flat_dim      # 8 × 8 = 64
        self.online_net = DuelingDQN(obs_dim, ec.n_actions, ac.hidden_dim).to(self.device)
        self.target_net = DuelingDQN(obs_dim, ec.n_actions, ac.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()             # target never receives gradients

        # optimiser
        self.optimizer = optim.Adam(
            self.online_net.parameters(), lr=ac.learning_rate, eps=1e-5
        )

        # replay buffer
        self.buffer = PrioritizedReplayBuffer(config)

        # hyper-parameters 
        self.gamma         = ac.gamma
        self.tau           = ac.tau
        self.batch_size    = ac.batch_size
        self.grad_clip     = ac.gradient_clip
        self.train_freq    = ac.train_freq
        self.eps_start     = ac.epsilon_start
        self.eps_end       = ac.epsilon_end
        self.eps_decay     = ac.epsilon_decay   # exponential decay constant

        # counters 
        self.total_steps      = 0
        self.gradient_updates = 0
        self._epsilon         = self.eps_start

        # diagnostics
        self._last_loss     = 0.0
        self._last_td_mean  = 0.0
        self._last_q_mean   = 0.0

        print(f"[DQNAgent] device={self.device}")
        print(f"[DQNAgent] obs_dim={obs_dim}  n_actions={ec.n_actions}")
        print(f"[DQNAgent] parameters={self.online_net.count_parameters():,}")

    # exploration 

    @property
    def epsilon(self) -> float:
        """Current ε value (exponential schedule)."""
        self._epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -self.total_steps / max(1, self.eps_decay)
        )
        return self._epsilon

    def select_action(self, obs: np.ndarray, training: bool = True) -> int:
        """
        ε-greedy policy during training; greedy otherwise.

        Parameters
        ----------
        obs      : flat observation array (obs_flat_dim,)
        training : whether to apply ε-greedy exploration
        """
        if training and np.random.random() < self.epsilon:
            return int(np.random.randint(self.config.env.n_actions))

        # (batch_size, features)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online_net.predict(obs_t)
        return int(q.argmax(dim=-1).item())

    # memory

    def store(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ):
        self.buffer.add(state, action, reward, next_state, done)
        self.total_steps += 1

    # learning

    def update(self) -> Dict[str, float]:
        """
        Perform one gradient update if the buffer has enough data.

        Returns a dict of diagnostics (empty dict if skipped).
        """
        if len(self.buffer) < self.batch_size:
            return {}

        # sample 
        (states, actions, rewards, next_states,
         dones, tree_indices, is_weights) = self.buffer.sample(self.batch_size)

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)
        is_weights  = torch.FloatTensor(is_weights).to(self.device)

        # Double DQN target 
        with torch.no_grad():
            # Online net selects best next action
            next_actions = self.online_net(next_states).argmax(dim=1)
            # Target net evaluates that action
            next_q = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        # Current Q values
        current_q = self.online_net(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # TD errors for PER update
        td_errors = (target_q - current_q).detach().cpu().numpy()

        # Huber loss with IS weights
        elementwise_loss = F.huber_loss(current_q, target_q, reduction="none")
        loss = (is_weights * elementwise_loss).mean()

        # Optimise 
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Update priorities
        self.buffer.update_priorities(tree_indices, td_errors)

        # Soft target update 
        self._soft_update_target()

        self.gradient_updates += 1
        self._last_loss    = float(loss.item())
        self._last_td_mean = float(np.abs(td_errors).mean())
        self._last_q_mean  = float(current_q.mean().item())

        return {
            "loss":             self._last_loss,
            "td_error_mean":    self._last_td_mean,
            "q_mean":           self._last_q_mean,
            "epsilon":          self.epsilon,
            "per_beta":         self.buffer.beta,
            "gradient_updates": self.gradient_updates,
        }

    # target network 

    def _soft_update_target(self):
        """θ_target ← τ·θ_online + (1−τ)·θ_target"""
        for t_p, o_p in zip(self.target_net.parameters(),
                             self.online_net.parameters()):
            t_p.data.copy_(self.tau * o_p.data + (1.0 - self.tau) * t_p.data)

    # persistence 

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "online_net":      self.online_net.state_dict(),
                "target_net":      self.target_net.state_dict(),
                "optimizer":       self.optimizer.state_dict(),
                "total_steps":     self.total_steps,
                "gradient_updates": self.gradient_updates,
                "epsilon":         self._epsilon,
                "per_beta":        self.buffer.beta,
            },
            path,
        )
        print(f"[DQNAgent] Checkpoint saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps      = ckpt["total_steps"]
        self.gradient_updates = ckpt.get("gradient_updates", 0)
        self._epsilon         = ckpt["epsilon"]
        self.buffer.beta      = ckpt.get("per_beta", self.buffer.beta)
        print(f"[DQNAgent] Checkpoint loaded ← {path}  "
              f"(step={self.total_steps}, ε={self._epsilon:.4f})")

    # diagnostics
    
    @property
    def diagnostics(self) -> Dict[str, float]:
        return {
            "loss":          self._last_loss,
            "td_error_mean": self._last_td_mean,
            "q_mean":        self._last_q_mean,
            "epsilon":       self.epsilon,
            "buffer_size":   len(self.buffer),
            "grad_updates":  self.gradient_updates,
        }
