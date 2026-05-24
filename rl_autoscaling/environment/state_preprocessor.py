"""
State Preprocessor
==================
Transforms raw cluster metrics into a rich, normalised observation tensor:

  1. Normalise raw metrics → [0, 1]
  2. Compute derived features:
       • Δ CPU  (rate of change)
       • Δ Memory
       • Utilisation ratio  ((cpu + mem) / 2)
  3. Build a sliding window of the last T=8 augmented states →
       shape (T, state_dim + derived_dim) = (8, 8)
     which is then flattened to (64,) before entering the neural network.
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple


class StatePreprocessor:
    # Min-max bounds for each raw metric (used for normalisation)
    _BOUNDS: Dict[str, Tuple[float, float]] = {
        "cpu_utilization":    (0.0,   1.0),
        "memory_utilization": (0.0,   1.0),
        "queue_length":       (0.0, 200.0),
        "avg_latency":        (0.0, 600.0),   # seconds
        "n_active_nodes":     (1.0,  20.0),
    }
    _METRIC_KEYS = list(_BOUNDS.keys())

    def __init__(self, config):
        self.state_dim   = config.env.state_dim       # 5 raw metrics
        self.derived_dim = config.env.derived_dim      # 3 derived features
        self.window_size = config.env.window_size      # T = 8

        self.feature_dim = self.state_dim + self.derived_dim   # 8
        self._history: deque = deque(maxlen=self.window_size)
        self._prev_normalised: Optional[np.ndarray] = None

    # public API

    def reset(self):
        """Call at the start of every episode."""
        self._history.clear()
        self._prev_normalised = None

    def process(self, metrics: Dict) -> np.ndarray:
        """
        Ingest a raw metrics dict, update internal state, and return the
        flattened sliding-window observation of shape (window_size * feature_dim,).
        """
        normalised  = self._normalise(metrics)
        derived     = self._derived_features(normalised)
        augmented   = np.concatenate([normalised, derived]).astype(np.float32)

        self._prev_normalised = normalised
        self._history.append(augmented)

        return self._build_window()

    # normalisation

    def _normalise(self, metrics: Dict) -> np.ndarray:
        out = np.empty(self.state_dim, dtype=np.float64)
        for i, key in enumerate(self._METRIC_KEYS):
            lo, hi = self._BOUNDS[key]
            val = float(metrics.get(key, 0.0))
            out[i] = np.clip((val - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        return out.astype(np.float32)

    # derived features

    def _derived_features(self, normalised: np.ndarray) -> np.ndarray:
        if self._prev_normalised is None:
            delta_cpu = 0.0
            delta_mem = 0.0
        else:
            delta_cpu = float(normalised[0] - self._prev_normalised[0])
            delta_mem = float(normalised[1] - self._prev_normalised[1])

        # Utilisation ratio: mean of normalised CPU + Memory
        util_ratio = float((normalised[0] + normalised[1]) / 2.0)

        return np.array([delta_cpu, delta_mem, util_ratio], dtype=np.float32)

    # sliding window

    def _build_window(self) -> np.ndarray:
        """
        Returns a (window_size * feature_dim,) float32 array.
        Earlier time-steps are zero-padded when history is short.
        """
        zero = np.zeros(self.feature_dim, dtype=np.float32)
        window = [zero] * (self.window_size - len(self._history))
        window += list(self._history)          # oldest … newest
        stacked = np.stack(window, axis=0)     # (T, feature_dim)
        return stacked.flatten()               # (T * feature_dim,)

    # shape helpers

    @property
    def obs_shape(self) -> Tuple[int, int]:
        return (self.window_size, self.feature_dim)

    @property
    def flat_obs_dim(self) -> int:
        return self.window_size * self.feature_dim
