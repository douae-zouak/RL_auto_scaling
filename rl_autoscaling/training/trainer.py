"""
Training Pipeline
=================

Full off-policy RL training loop:

  ┌─ episode loop ──────────────────────────────────────┐
  │  reset env                                          │
  │  ┌─ step loop ───────────────────────────────────┐  │
  │  │  select action (ε-greedy)                     │  │
  │  │  step env  → obs', reward, done, info         │  │
  │  │  store transition in PER buffer               │  │
  │  │  every train_freq steps:                      │  │
  │  │      sample mini-batch                        │  │
  │  │      compute Double-DQN target                │  │
  │  │      Huber loss + IS weights                  │  │
  │  │      gradient clip + Adam step                │  │
  │  │      soft-update target network               │  │
  │  │      update PER priorities                    │  │
  │  │  every eval_freq steps:                       │  │
  │  │      greedy evaluation (N episodes)           │  │
  │  │      log + save best checkpoint               │  │
  │  │  every save_freq steps:                       │  │
  │  │      save checkpoint                          │  │
  │  └───────────────────────────────────────────────┘  │
  └─────────────────────────────────────────────────────┘
"""

import os
import time
import random
import numpy as np
import torch

from ..environment.cluster_env import ClusterAutoscalingEnv
from ..agent.dqn_agent import DQNAgent
from ..monitoring.metrics_logger import MetricsLogger
from .evaluator import Evaluator


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """
    Orchestrates the full training run.

    Parameters
    ----------
    config : Config dataclass
    """

    def __init__(self, config):
        self.config = config
        tc = config.training

        set_seed(tc.seed)

        self.env       = ClusterAutoscalingEnv(config, render_mode="human")
        self.agent     = DQNAgent(config)
        self.evaluator = Evaluator(config)
        self.logger    = MetricsLogger(config)

        self.total_steps   = tc.total_steps
        self.warmup_steps  = tc.warmup_steps
        self.train_freq    = config.agent.train_freq
        self.eval_freq     = tc.eval_freq
        self.save_freq     = tc.save_freq
        self.log_freq      = tc.log_freq
        self.ckpt_dir      = tc.checkpoint_dir

        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Best-model tracking (by eval reward)
        self._best_eval_reward = -float("inf")

        # Episode counters
        self._ep            = 0
        self._ep_reward     = 0.0
        self._ep_steps      = 0
        self._ep_sla_sum    = 0.0
        self._ep_tp_start   = 0

        # Step counters (global across episodes)
        self._global_step   = 0

    # public API

    def train(self):
        """Run the full training loop."""
        print(f"\n{'═'*70}")
        print("  Dueling Double DQN  |  Cluster Autoscaling  |  Training Start")
        print(f"{'═'*70}\n")

        obs, info = self.env.reset()
        self._reset_episode_accumulators(info)

        t0 = time.time()

        while self._global_step < self.total_steps:

            # select and execute action 
            action = self._select_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # store transition
            self.agent.store(obs, action, float(reward), next_obs, done)
            obs = next_obs

            # accumulate episode metrics
            self._ep_reward += reward
            self._ep_steps  += 1
            self._ep_sla_sum += info["reward_info"].get("sla_violation", 0.0)

            # learn 
            agent_info: dict = {}
            if (self._global_step >= self.warmup_steps
                    and self._global_step % self.train_freq == 0):
                agent_info = self.agent.update()

            # step logging 
            if self._global_step % self.log_freq == 0:
                metrics = info["metrics"]
                step_data = {
                    "episode":       self._ep,
                    "reward":        reward,
                    "episode_reward": self._ep_reward,
                    "sla_violation": info["reward_info"].get("sla_violation", 0.0),
                    "latency":       metrics.get("avg_latency", 0.0),
                    "throughput":    metrics.get("completed_jobs", 0),
                    "n_nodes":       metrics.get("n_active_nodes", 0),
                    "cpu_util":      metrics.get("cpu_utilization", 0.0),
                    "queue_len":     metrics.get("queue_length", 0),
                }
                step_data.update(agent_info)
                self.logger.log_step(self._global_step, **step_data)

                if self._global_step % (self.log_freq * 10) == 0:
                    self.logger.print_progress(self._global_step, agent_info)

            # periodic evaluation 
            if (self._global_step > self.warmup_steps
                    and self._global_step % self.eval_freq == 0):
                self._run_evaluation()

            # periodic checkpoint 
            if (self._global_step > 0
                    and self._global_step % self.save_freq == 0):
                ckpt_path = os.path.join(
                    self.ckpt_dir, f"ckpt_step_{self._global_step}.pt"
                )
                self.agent.save(ckpt_path)

            # episode end 
            if done:
                self._ep += 1
                self.logger.log_episode_end(self._ep, self._ep_reward)
                obs, info = self.env.reset()
                self._reset_episode_accumulators(info)

            self._global_step += 1

        # final save 
        self.agent.save(os.path.join(self.ckpt_dir, "final.pt"))
        elapsed = time.time() - t0
        print(f"\n[Trainer] Training complete in {elapsed/60:.1f} min  "
              f"({self.total_steps:,} steps)")
        self.logger.close()

    # internals

    def _select_action(self, obs: np.ndarray) -> int:
        """Random during warmup, ε-greedy thereafter."""
        if self._global_step < self.warmup_steps:
            return int(self.env.action_space.sample())
        return self.agent.select_action(obs, training=True)

    def _run_evaluation(self):
        results = self.evaluator.evaluate(self.agent)
        self.logger.log_eval(self._global_step, results)
        self.logger.print_eval(self._global_step, results)

        mean_reward = results["eval/ep_reward_mean"]
        if mean_reward > self._best_eval_reward:
            self._best_eval_reward = mean_reward
            best_path = os.path.join(self.ckpt_dir, "best.pt")
            self.agent.save(best_path)
            print(f"  ★ New best model  (reward={mean_reward:.3f}) → {best_path}")

    def _reset_episode_accumulators(self, info):
        self._ep_reward   = 0.0
        self._ep_steps    = 0
        self._ep_sla_sum  = 0.0
        metrics = info.get("metrics", {})
        self._ep_tp_start = metrics.get("completed_jobs", 0)
