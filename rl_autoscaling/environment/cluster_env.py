"""
ClusterAutoscalingEnv — Custom Gymnasium Environment
=====================================================

Action space (Discrete 6)
--------------------------
  0  idle          — no scaling operation
  1  scale_up_1    — add 1 worker node
  2  scale_up_2    — add 2 worker nodes
  3  scale_down_1  — remove 1 worker node
  4  scale_down_2  — remove 2 worker nodes
  5  migrate       — redistribute workload across active nodes

Observation space (Box [0,1]^{T×F} flattened)
-------------------------------------------------
  T = 8 time-steps  ×  F = 8 features per step  →  64-dimensional vector
  F = [cpu_util, mem_util, queue_norm, lat_norm, nodes_norm,
       Δcpu, Δmem, util_ratio]

Reward
------
  Multi-objective: -cost - λ·SLA + throughput + efficiency - stability
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Optional, Tuple, Any

from .cluster_simulator import ClusterSimulator
from .state_preprocessor import StatePreprocessor
from .reward_function import RewardFunction


class ClusterAutoscalingEnv(gym.Env):
    """
    Gym-compatible environment wrapping the full cluster simulation stack.
    Suitable for off-policy RL agents (DQN, SAC, …).
    """

    metadata = {"render_modes": ["human"]}

    #  action catalogue 
    _ACTION_MAP: Dict[int, Tuple[str, int]] = {
        0: ("idle",       0),
        1: ("scale_up",   1),
        2: ("scale_up",   2),
        3: ("scale_down", 1),
        4: ("scale_down", 2),
        5: ("migrate",    0),
    }
    
    ACTION_NAMES = [
        "idle", "scale_up_1", "scale_up_2",
        "scale_down_1", "scale_down_2", "migrate",
    ]

    def __init__(self, config, render_mode: Optional[str] = None):
        super().__init__()
        self.config      = config
        self.render_mode = render_mode

        # Sub-systems
        self.simulator   = ClusterSimulator(config)
        self.preprocessor = StatePreprocessor(config)
        self.reward_fn   = RewardFunction(config)

        # Spaces
        flat_dim = self.preprocessor.flat_obs_dim   # 64
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(flat_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(config.env.n_actions)

        # Episode bookkeeping
        self._max_steps       = config.env.max_steps
        self._dt              = config.env.dt_seconds
        self._current_step    = 0
        self._prev_completed  = 0
        self._episode_reward  = 0.0
        self._last_metrics: Dict = {}

    #  Gymnasium API

    def reset(
        self,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self._current_step   = 0
        self._prev_completed = 0
        self._episode_reward = 0.0

        raw_metrics = self.simulator.reset()
        self.preprocessor.reset()

        obs = self.preprocessor.process(raw_metrics)
        self._last_metrics = raw_metrics

        info = {"metrics": raw_metrics, "step": 0}
        return obs.astype(np.float32), info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.action_space.contains(action), f"Invalid action {action}"

        self._current_step += 1
        action_type, intensity = self._ACTION_MAP[action]

        # 1. Execute action on simulator
        success, action_msg = self._execute(action_type, intensity)

        # 2. Advance simulation one timestep
        raw_metrics = self.simulator.step(self._dt)

        # 3. Build observation
        obs = self.preprocessor.process(raw_metrics).astype(np.float32)

        # 4. Compute reward
        reward, reward_info = self.reward_fn.compute(
            raw_metrics, action, self._prev_completed
        )

        self._prev_completed  = raw_metrics["completed_jobs"]
        self._episode_reward += reward
        self._last_metrics    = raw_metrics

        # 5. Termination / truncation
        terminated = False
        truncated  = self._current_step >= self._max_steps

        info = {
            "metrics":       raw_metrics,
            "reward_info":   reward_info,
            "action_name":   self.ACTION_NAMES[action],
            "action_success": success,
            "action_msg":    action_msg,
            "step":          self._current_step,
            "episode_reward": self._episode_reward,
        }
        return obs, reward, terminated, truncated, info

    # action execution

    def _execute(self, action_type: str, intensity: int) -> Tuple[bool, str]:
        if action_type == "idle":
            return True, "idle"
        elif action_type == "scale_up":
            return self.simulator.scale_up(intensity)
        elif action_type == "scale_down":
            return self.simulator.scale_down(intensity)
        elif action_type == "migrate":
            return self.simulator.migrate()
        return False, "unknown_action"

    # rendering 
    
    def render(self):
        if self.render_mode != "human":
            return
        m = self._last_metrics
        bar_cpu = "█" * int(m.get("cpu_utilization", 0) * 20)
        bar_mem = "█" * int(m.get("memory_utilization", 0) * 20)
        print(
            f"Step {self._current_step:>5} | "
            f"Nodes:{m.get('n_active_nodes', 0):>3} | "
            f"CPU [{bar_cpu:<20}] {m.get('cpu_utilization', 0):.1%} | "
            f"Mem [{bar_mem:<20}] {m.get('memory_utilization', 0):.1%} | "
            f"Queue:{m.get('queue_length', 0):>4} | "
            f"Done:{m.get('completed_jobs', 0):>6} | "
            f"EpRew:{self._episode_reward:>+8.2f}"
        )

    # properties

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def last_metrics(self) -> Dict:
        return self._last_metrics
