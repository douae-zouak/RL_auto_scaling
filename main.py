"""
main.py — Entry Point
======================

Usage
-----
  # Full training run (default config)
  python main.py --mode train

  # Quick smoke-test (5 episodes, no learning)
  python main.py --mode demo

  # Evaluate a saved checkpoint
  python main.py --mode eval --checkpoint checkpoints/best.pt

  # Short training with custom config overrides
  python main.py --mode train --total_steps 50000 --seed 123
"""

import argparse
import sys
import os

# Ensure the project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_autoscaling.utils.config import get_default_config
from rl_autoscaling.training.trainer import Trainer
from rl_autoscaling.training.evaluator import Evaluator
from rl_autoscaling.agent.dqn_agent import DQNAgent
from rl_autoscaling.environment.cluster_env import ClusterAutoscalingEnv


# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deep RL Cluster Autoscaling — Dueling Double DQN + PER"
    )
    p.add_argument("--mode", choices=["train", "eval"], default="train",
                   help="Execution mode")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to checkpoint (.pt) for eval / resume")
    # Training overrides
    p.add_argument("--total_steps",  type=int, default=None)
    p.add_argument("--eval_freq",    type=int, default=None)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--seed",         type=int, default=None)
    # Agent overrides
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--batch_size",   type=int,   default=None)
    p.add_argument("--hidden_dim",   type=int,   default=None)
    # Logging
    p.add_argument("--log_dir",      type=str, default=None)
    p.add_argument("--ckpt_dir",     type=str, default=None)
    return p.parse_args()


# Config builder

def build_config(args: argparse.Namespace):
    cfg = get_default_config()

    if args.total_steps  is not None: cfg.training.total_steps  = args.total_steps
    if args.eval_freq    is not None: cfg.training.eval_freq    = args.eval_freq
    if args.warmup_steps is not None: cfg.training.warmup_steps = args.warmup_steps
    if args.seed         is not None: cfg.training.seed         = args.seed
    if args.lr           is not None: cfg.agent.learning_rate   = args.lr
    if args.batch_size   is not None: cfg.agent.batch_size      = args.batch_size
    if args.hidden_dim   is not None: cfg.agent.hidden_dim      = args.hidden_dim
    if args.log_dir      is not None: cfg.training.log_dir      = args.log_dir
    if args.ckpt_dir     is not None: cfg.training.checkpoint_dir = args.ckpt_dir

    return cfg


# Modes

def run_train(cfg):
    trainer = Trainer(cfg)
    trainer.train()


def run_eval(cfg, checkpoint_path: str):
    if not checkpoint_path:
        raise ValueError("--checkpoint is required for eval mode")

    agent = DQNAgent(cfg)
    agent.load(checkpoint_path)

    evaluator = Evaluator(cfg)
    results = evaluator.evaluate(agent, n_episodes=10)

    print("\nEvaluation Results")
    for k, v in results.items():
        unit = "s" if "latency" in k else ""
        print(f"  {k:<35} {v:.4f}{unit}")


# Entrypoint

def main():
    args = parse_args()
    cfg  = build_config(args)

    _banner(cfg, args.mode)

    if args.mode == "train":
        run_train(cfg)
    elif args.mode == "eval":
        run_eval(cfg, args.checkpoint)


def _banner(cfg, mode: str):
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║       Deep RL Cluster Autoscaling  —  Dueling Double DQN + PER      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Mode         : {mode:<53}║
║  Steps        : {cfg.training.total_steps:<53,}║
║  Obs dim      : {cfg.obs_flat_dim:<53}║
║  Actions      : {cfg.env.n_actions:<53}║
║  Window (T)   : {cfg.env.window_size:<53}║
║  Buffer       : {cfg.agent.buffer_size:<53,}║
║  Batch        : {cfg.agent.batch_size:<53}║
║  LR           : {cfg.agent.learning_rate:<53}║
║  γ            : {cfg.agent.gamma:<53}║
║  τ (soft upd) : {cfg.agent.tau:<53}║
║  Max nodes    : {cfg.cluster.max_nodes:<53}║
╚══════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
