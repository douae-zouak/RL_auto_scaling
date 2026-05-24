"""
Metrics Logger
==============
Records training and evaluation metrics to:
  • Console (colourised tabular output)
  • CSV file  (logs/training_metrics.csv)
  • JSON file (logs/eval_results.json)

Provides lightweight Prometheus-style metric aggregation
(running min / max / mean / std over a window).
"""

import os
import csv
import json
import time
import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


# ANSI colour helpers
_RESET = "\033[0m"
_BOLD  = "\033[1m"
_GREEN = "\033[92m"
_CYAN  = "\033[96m"
_YELLOW= "\033[93m"
_RED   = "\033[91m"
_BLUE  = "\033[94m"


def _colour(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}"


# Running statistics 

class RunningStats:
    """Maintains a rolling window of scalar values."""

    def __init__(self, window: int = 100):
        self._buf = deque(maxlen=window)

    def push(self, v: float):
        self._buf.append(float(v))

    @property
    def mean(self) -> float:
        return float(np.mean(self._buf)) if self._buf else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self._buf)) if len(self._buf) > 1 else 0.0

    @property
    def min(self) -> float:
        return float(np.min(self._buf)) if self._buf else 0.0

    @property
    def max(self) -> float:
        return float(np.max(self._buf)) if self._buf else 0.0

    def summary(self) -> Dict[str, float]:
        return dict(mean=self.mean, std=self.std, min=self.min, max=self.max)


#  Main logger

class MetricsLogger:
    """
    Centralised metrics recorder for the RL training pipeline.

    Usage
    -----
    logger = MetricsLogger(config)
    logger.log_step(step, reward=0.5, sla_violation=0.1, ...)
    logger.log_eval(step, eval_results)
    logger.close()
    """

    # Columns written to the CSV (order preserved)
    _TRAIN_FIELDS = [
        "step", "wall_time", "episode", "reward", "episode_reward",
        "sla_violation", "latency", "throughput", "n_nodes",
        "cpu_util", "queue_len", "loss", "td_error", "q_mean",
        "epsilon", "per_beta", "buffer_size", "grad_updates",
    ]

    def __init__(self, config, log_dir: Optional[str] = None):
        self.log_dir = log_dir or config.training.log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self._start_time = time.time()
        self._episode    = 0
        self._episode_reward_buf = RunningStats(100)

        # Running stats keyed by metric name
        self._stats: Dict[str, RunningStats] = defaultdict(
            lambda: RunningStats(100)
        )

        # CSV writer
        csv_path = os.path.join(self.log_dir, "training_metrics.csv")
        self._csv_file   = open(csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=self._TRAIN_FIELDS, extrasaction="ignore"
        )
        self._csv_writer.writeheader()

        # Eval results list → JSON on close
        self._eval_results: List[Dict] = []
        self._eval_json_path = os.path.join(self.log_dir, "eval_results.json")

        # Training log file
        logging.basicConfig(
            filename=os.path.join(self.log_dir, "training.log"),
            level=logging.INFO,
            format="%(asctime)s  %(levelname)s  %(message)s",
        )
        print(f"[MetricsLogger] Logging to {self.log_dir}/")

    #  training step logging 
    def log_step(self, step: int, **kwargs):
        """Record metrics for one environment step."""
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                self._stats[k].push(float(v))

        row = {"step": step, "wall_time": round(time.time() - self._start_time, 2)}
        row.update(kwargs)
        self._csv_writer.writerow(row)

    def log_episode_end(self, episode: int, episode_reward: float):
        self._episode = episode
        self._episode_reward_buf.push(episode_reward)
        self._stats["episode_reward"].push(episode_reward)

    #  evaluation logging 

    def log_eval(self, step: int, results: Dict[str, Any]):
        """Record a periodic evaluation result."""
        record = {"step": step, "wall_time": time.time() - self._start_time}
        record.update(results)
        self._eval_results.append(record)
        # Persist immediately so we don't lose data on crash
        with open(self._eval_json_path, "w") as f:
            json.dump(self._eval_results, f, indent=2)

    # console printing 

    def print_progress(self, step: int, agent_info: Dict):
        """Print a rich one-liner progress update to stdout."""
        ep_rew    = self._stats["episode_reward"].mean
        reward    = self._stats["reward"].mean
        sla       = self._stats["sla_violation"].mean
        latency   = self._stats["latency"].mean
        nodes     = self._stats["n_nodes"].mean
        cpu       = self._stats["cpu_util"].mean
        eps       = agent_info.get("epsilon", 0)
        loss      = agent_info.get("loss", 0)
        buf       = agent_info.get("buffer_size", 0)

        # Colour-code reward trend
        rew_col = _GREEN if reward > 0 else _RED

        print(
            f"{_BOLD}[{step:>7}]{_RESET} "
            f"EpRew={_colour(f'{ep_rew:>+7.2f}', rew_col)}  "
            f"Rew={_colour(f'{reward:>+6.3f}', rew_col)}  "
            f"SLA={_colour(f'{sla:.3f}', _YELLOW if sla > 0.1 else _GREEN)}  "
            f"Lat={latency:>7.1f}s  "
            f"Nodes={_colour(f'{nodes:>4.1f}', _CYAN)}  "
            f"CPU={cpu:.1%}  "
            f"ε={_colour(f'{eps:.4f}', _BLUE)}  "
            f"Loss={loss:.4f}  "
            f"Buf={buf:>6}"
        )

    def print_eval(self, step: int, results: Dict):
        line = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in results.items())
        print(_colour(f"\n{'─'*70}", _CYAN))
        print(_colour(f"  EVAL @ step {step:,}  |  {line}", _BOLD + _CYAN))
        print(_colour(f"{'─'*70}\n", _CYAN))

    # cleanup 

    def close(self):
        self._csv_file.flush()
        self._csv_file.close()
        with open(self._eval_json_path, "w") as f:
            json.dump(self._eval_results, f, indent=2)
        print(f"[MetricsLogger] Closed.  Logs saved to {self.log_dir}/")

    # convenience accessors 

    def get_summary(self) -> Dict[str, Dict]:
        return {k: v.summary() for k, v in self._stats.items()}
