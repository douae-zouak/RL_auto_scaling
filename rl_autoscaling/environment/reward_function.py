"""
Multi-Objective Reward Function
================================

  R = -cost  −  λ·SLA_violation  +  throughput_bonus  +  efficiency_bonus  −  stability_penalty

Where:
  • cost              ∝ number of active nodes  (infrastructure spend)
  • SLA_violation     = latency excess + queue excess (0 if within SLA)
  • throughput_bonus  = normalised count of newly completed jobs
  • efficiency_bonus  = incentive for 50–80 % CPU utilisation sweet-spot
  • stability_penalty = penalty for unnecessary scaling churn
"""

import numpy as np
from typing import Dict, Tuple


class RewardFunction:
    """Stateless (except previous-completed counter) multi-objective reward."""

    # Action indices that trigger a scaling operation
    _SCALING_ACTIONS = {1, 2, 3, 4}      # scale_up_1/2, scale_down_1/2
    _MIGRATE_ACTION   = 5

    def __init__(self, config):
        rc = config.reward
        ec = config.env
        cc = config.cluster

        self._cost_w        = rc.cost_weight
        self._sla_w         = rc.sla_penalty_weight
        self._tp_w          = rc.throughput_weight
        self._stab_w        = rc.stability_weight
        self._eff_bonus     = rc.efficiency_bonus
        self._max_nodes     = cc.max_nodes
        self._lat_thresh    = ec.sla_latency_threshold
        self._queue_thresh  = ec.sla_queue_threshold

    # normalisation helpers 

    @staticmethod
    def _norm_latency(latency_s: float) -> float:
        return float(np.clip(latency_s / 600.0, 0.0, 1.0))

    @staticmethod
    def _norm_queue(queue_len: int) -> float:
        return float(np.clip(queue_len / 200.0, 0.0, 1.0))

    # main API 

    def compute(
        self,
        metrics: Dict,
        action: int,
        prev_completed: int,
    ) -> Tuple[float, Dict]:
        """
        Parameters
        ----------
        metrics         : current cluster metrics dict
        action          : integer action taken (0-5)
        prev_completed  : completed job count from the *previous* step

        Returns
        -------
        reward  : scalar float clipped to [−10, 10]
        info    : dict with individual reward components (for logging)
        """
        n_nodes    = int(metrics["n_active_nodes"])
        cpu_util   = float(metrics["cpu_utilization"])
        queue_len  = int(metrics["queue_length"])
        avg_lat    = float(metrics["avg_latency"])
        completed  = int(metrics["completed_jobs"])

        # 1. Infrastructure cost 
        cost = self._cost_w * (n_nodes / self._max_nodes)

        # 2. SLA violations 
        lat_norm   = self._norm_latency(avg_lat)
        queue_norm = self._norm_queue(queue_len)

        lat_viol   = max(0.0, lat_norm   - self._lat_thresh)
        queue_viol = max(0.0, queue_norm - self._queue_thresh)
        sla_viol   = lat_viol + queue_viol

        # 3. Throughput
        new_jobs = max(0, completed - prev_completed)
        throughput = self._tp_w * min(new_jobs / 15.0, 1.0)

# garder les serveurs dans une “zone d’utilisation optimale”
        # 4. Efficiency bonus
        if 0.50 <= cpu_util <= 0.80:
            eff_bonus = self._eff_bonus                # sweet-spot
        elif cpu_util > 0.90:
            eff_bonus = -self._eff_bonus * 1.5         # overloaded
        elif cpu_util < 0.20 and n_nodes > 1:
            eff_bonus = -self._eff_bonus * 0.5         # over-provisioned
        else:
            eff_bonus = 0.0

        # 5. Stability penalty 
        if action in self._SCALING_ACTIONS:
            stab_penalty = self._stab_w * 0.1
        elif action == self._MIGRATE_ACTION:
            stab_penalty = self._stab_w * 0.05
        else:
            stab_penalty = 0.0

        # Combined reward
        reward = (
            -cost
            - self._sla_w   * sla_viol
            + throughput
            + eff_bonus
            - stab_penalty
        )
        reward = float(np.clip(reward, -10.0, 10.0))

        info = {
            "reward_total":      reward,
            "cost":              cost,
            "sla_violation":     sla_viol,
            "lat_violation":     lat_viol,
            "queue_violation":   queue_viol,
            "throughput_bonus":  throughput,
            "efficiency_bonus":  eff_bonus,
            "stability_penalty": stab_penalty,
            "new_jobs_done":     new_jobs,
        }
        return reward, info
