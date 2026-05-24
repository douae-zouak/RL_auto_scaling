"""
Evaluator
=========
Runs the agent in greedy (ε=0) mode for a fixed number of episodes
and computes aggregate performance statistics used for model selection
and early-stopping decisions.

Metrics reported
----------------
  • mean / std episode reward
  • mean SLA violation rate
  • mean throughput (jobs completed / step)
  • mean node count
  • mean CPU utilisation
  • mean latency (s)
  • mean queue length
"""

import numpy as np
from typing import Dict, List

from ..environment.cluster_env import ClusterAutoscalingEnv


class Evaluator:
    """
    Parameters
    ----------
    config : Config dataclass
    """

    def __init__(self, config):
        self.config       = config
        self.n_episodes   = config.training.eval_episodes
        self.env          = ClusterAutoscalingEnv(config)

    # public API 
    
    def evaluate(self, agent, n_episodes: int = None) -> Dict[str, float]:
        """
        Run *n_episodes* greedy episodes and return aggregated metrics.

        Parameters
        ----------
        agent      : DQNAgent instance
        n_episodes : override default episode count (optional)
        """
        n_ep = n_episodes or self.n_episodes
        episode_rewards:   List[float] = []
        sla_violations:    List[float] = []
        throughputs:       List[float] = []
        node_counts:       List[float] = []
        cpu_utils:         List[float] = []
        latencies:         List[float] = []
        queue_lens:        List[float] = []

        for ep in range(n_ep):
            obs, info = self.env.reset()
            ep_reward = 0.0
            ep_sla    = 0.0
            ep_tp     = []
            ep_nodes  = []
            ep_cpu    = []
            ep_lat    = []
            ep_queue  = []
            done = trunc = False

            while not (done or trunc):
                action = agent.select_action(obs, training=False)  # greedy
                obs, reward, done, trunc, info = self.env.step(action)

                ep_reward += reward
                metrics    = info["metrics"]
                rew_info   = info["reward_info"]

                ep_sla   += rew_info.get("sla_violation", 0.0)
                ep_tp.append(metrics.get("completed_jobs", 0))
                ep_nodes.append(metrics.get("n_active_nodes", 0))
                ep_cpu.append(metrics.get("cpu_utilization", 0.0))
                ep_lat.append(metrics.get("avg_latency", 0.0))
                ep_queue.append(metrics.get("queue_length", 0))

            ep_steps = max(1, len(ep_nodes))
            episode_rewards.append(ep_reward)
            sla_violations.append(ep_sla / ep_steps)
            # throughput = completed jobs per step
            throughputs.append(ep_tp[-1] / ep_steps if ep_tp else 0.0)
            node_counts.append(float(np.mean(ep_nodes)))
            cpu_utils.append(float(np.mean(ep_cpu)))
            latencies.append(float(np.mean(ep_lat)))
            queue_lens.append(float(np.mean(ep_queue)))

        results = {
            "eval/ep_reward_mean":   float(np.mean(episode_rewards)),
            "eval/ep_reward_std":    float(np.std(episode_rewards)),
            "eval/sla_violation":    float(np.mean(sla_violations)),
            "eval/throughput":       float(np.mean(throughputs)),
            "eval/n_nodes_mean":     float(np.mean(node_counts)),
            "eval/cpu_util_mean":    float(np.mean(cpu_utils)),
            "eval/latency_mean":     float(np.mean(latencies)),
            "eval/queue_len_mean":   float(np.mean(queue_lens)),
            "eval/n_episodes":       n_ep,
        }
        return results
