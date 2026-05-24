"""
Configuration module — all hyperparameters in one place.
Centralizes Cluster, Environment, Agent, Training, and Reward settings.
"""
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# Cluster infrastructure config
# ─────────────────────────────────────────────
@dataclass
class ClusterConfig:
    min_nodes: int = 1
    max_nodes: int = 20
    initial_nodes: int = 3
    vcpus_per_node: int = 8        # 8 → 160 total vCPUs at max
    memory_per_node_gb: int = 32   # 32 → 640 GB total at max
    cooldown_seconds: float = 30.0 # minimum seconds between scaling ops
    scaling_delay_seconds: float = 10.0


# ─────────────────────────────────────────────
# Gym environment config
# ─────────────────────────────────────────────
@dataclass
class EnvConfig:
    max_steps: int = 1000
    state_dim: int = 5    
    derived_dim: int = 3           # delta_cpu, delta_mem, util_ratio
    window_size: int = 8           # T = 8  temporal sliding window
    n_actions: int = 6             # idle / scale_up×2 / scale_down×2 / migrate
    sla_latency_threshold: float = 0.5   # normalised latency SLA bound
    sla_queue_threshold: float = 0.6     # normalised queue SLA bound
    dt_seconds: float = 10.0             # simulation timestep


# ─────────────────────────────────────────────
# Agent / DQN config
# ─────────────────────────────────────────────
@dataclass
class AgentConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005             # soft-update coefficient
    batch_size: int = 64
    buffer_size: int = 100_000
    train_freq: int = 4            # update every N env steps
    hidden_dim: int = 256

    # Epsilon-greedy exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 50_000    # exponential decay steps

    # Gradient clipping
    gradient_clip: float = 10.0
    # empêche ton réseau de sur-réagir aux pics de reward en limitant la norme du gradient

    # Prioritized Experience Replay
    per_alpha: float = 0.6         # priority exponent
    per_beta_start: float = 0.4    # IS correction start
    per_beta_end: float = 1.0
    per_beta_steps: int = 200_000  # β annealing horizon
    per_epsilon: float = 1e-6      # priority floor


# ─────────────────────────────────────────────
# Training pipeline config
# ─────────────────────────────────────────────
@dataclass
class TrainingConfig:
    total_steps: int = 300_000
    eval_freq: int = 10_000        # eval every N steps
    eval_episodes: int = 5
    save_freq: int = 50_000
    log_freq: int = 500
    warmup_steps: int = 5_000      # random exploration before learning
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    seed: int = 42


# ─────────────────────────────────────────────
# Multi-objective reward config
# ─────────────────────────────────────────────
@dataclass
class RewardConfig:
    cost_weight: float = 0.15          # infrastructure cost penalty
    sla_penalty_weight: float = 2.5    # λ SLA violation penalty
    throughput_weight: float = 1.0     # jobs completed bonus
    stability_weight: float = 0.3      # penalise thrashing
    efficiency_bonus: float = 0.25     # bonus for 50–80% utilisation


# Master config (aggregates all sub-configs)
@dataclass
class Config:
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @property
    def obs_flat_dim(self) -> int:
        """Flattened observation dimension sent to the neural network."""
        return self.env.window_size * (self.env.state_dim + self.env.derived_dim)


def get_default_config() -> Config:
    return Config()
