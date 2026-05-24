"""
Prioritized Experience Replay (PER)
=====================================

Implements a SumTree-backed priority buffer as described in
"Prioritized Experience Replay" (Schaul et al., 2015).

Key properties
--------------
• O(log N) insertion and sampling via binary SumTree
• Priority:  p_i = |δ_i| + ε   (TD-error magnitude + floor)
• Sampling:  P(i) = p_i^α / Σ p_j^α
• IS weights: w_i = (N · P(i))^{-β} / max_j w_j
• β annealed from β_0 → 1.0 over training (bias correction)
• max-priority initialisation for fresh transitions
"""

import numpy as np
from typing import Tuple, List

# Binary SumTree

class SumTree:
    """
    A binary tree where each leaf holds a transition priority and each
    internal node holds the sum of its children's values.

    Layout (1-indexed tree of size 2*cap - 1):
        internal nodes: indices 0 … cap-2
        leaf nodes:     indices cap-1 … 2*cap-2
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        # We store data as object array to avoid dtype issues with transitions
        self.data: np.ndarray = np.empty(capacity, dtype=object)
        self._write_ptr = 0          # circular write pointer
        self.n_entries  = 0          # actual number of stored transitions

    # tree operations

    def _propagate(self, leaf_idx: int, delta: float):
        """Propagate a priority change up from *leaf_idx* to the root."""
        parent = (leaf_idx - 1) >> 1 
        self.tree[parent] += delta
        if parent:
            self._propagate(parent, delta)

    def update(self, tree_idx: int, priority: float):
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, delta)

    def add(self, priority: float, data) -> int:
        """Insert a new transition with *priority* and return its tree index."""
        tree_idx  = self._write_ptr + self.capacity - 1
        self.data[self._write_ptr] = data
        self.update(tree_idx, priority)

        self._write_ptr = (self._write_ptr + 1) % self.capacity
        self.n_entries  = min(self.n_entries + 1, self.capacity)
        return tree_idx

    def _retrieve(self, node: int, target: float) -> int:
        """Walk down the tree to find the leaf whose cumulative sum ≥ target."""
        left  = 2 * node + 1
        right = left + 1
        if left >= len(self.tree):          # reached a leaf
            return node
        if target <= self.tree[left]:
            return self._retrieve(left, target)
        return self._retrieve(right, target - self.tree[left])

    def get(self, target: float) -> Tuple[int, float, object]:
        """
        Sample a transition proportional to priority.

        Returns
        -------
        tree_idx  : index in self.tree (needed for priority updates)
        priority  : p_i (raw priority value)
        data      : stored transition tuple
        """
        tree_idx = self._retrieve(0, target)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, self.tree[tree_idx], self.data[data_idx]

    @property
    def total(self) -> float:
        return float(self.tree[0])


# Prioritized Replay Buffer

class PrioritizedReplayBuffer:
    """
    Experience replay buffer with priority sampling and IS-weight correction.

    Transitions are stored as tuples:
        (state, action, reward, next_state, done)
    All numpy arrays.
    """

    def __init__(self, config):
        ac = config.agent
        self.capacity  = ac.buffer_size
        self.alpha     = ac.per_alpha          # priority exponent
        self.beta      = ac.per_beta_start     # IS correction (annealed)
        self.beta_end  = ac.per_beta_end
        self.epsilon   = ac.per_epsilon        # priority floor

        beta_steps     = ac.per_beta_steps
        self._beta_inc = (self.beta_end - self.beta) / max(1, beta_steps)

        self.tree         = SumTree(self.capacity)
        self._max_priority = 1.0               # initialise new transitions at max

    # public API

    def add(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ):
        """Store a transition with maximum current priority."""
        p = self._max_priority ** self.alpha
        self.tree.add(p, (state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, List[int], np.ndarray]:
        """
        Sample *batch_size* transitions proportional to priority.

        Returns
        -------
        states, actions, rewards, next_states, dones : arrays
        tree_indices : for priority updates after TD-error computation
        is_weights   : importance-sampling weights, normalised to [0,1]
        """
        total     = self.tree.total + 1e-8
        segment   = total / batch_size 

        indices, priorities, transitions = [], [], []
        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, prio, data = self.tree.get(np.clip(s, 0.0, total))
            indices.append(idx)
            priorities.append(max(prio, 1e-8))
            transitions.append(data)

        # Importance-sampling weights
        n      = self.tree.n_entries
        probs  = np.array(priorities, dtype=np.float64) / total
        raw_w  = (n * probs) ** (-self.beta)
        is_w   = (raw_w / raw_w.max()).astype(np.float32)

        # β annealing
        self.beta = min(self.beta_end, self.beta + self._beta_inc)

        # Unpack transitions
        states      = np.array([t[0] for t in transitions], dtype=np.float32)
        actions     = np.array([t[1] for t in transitions], dtype=np.int64)
        rewards     = np.array([t[2] for t in transitions], dtype=np.float32)
        next_states = np.array([t[3] for t in transitions], dtype=np.float32)
        dones       = np.array([t[4] for t in transitions], dtype=np.float32)

        return states, actions, rewards, next_states, dones, indices, is_w

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities using fresh TD-errors after a learning step."""
        for idx, td_err in zip(indices, td_errors):
            p = (abs(float(td_err)) + self.epsilon) ** self.alpha
            self._max_priority = max(self._max_priority, p)
            self.tree.update(idx, p)

    def __len__(self) -> int:
        return self.tree.n_entries

    @property
    def is_ready(self) -> bool:
        return len(self) > 0
