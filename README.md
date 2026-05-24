# 🤖 Deep RL Cluster Autoscaling

> **Dueling Double DQN + Prioritized Experience Replay** for intelligent Kubernetes-style cluster resource management.

---

## 🏗️ Architecture

```
rl_autoscaling/
├── environment/
│   ├── cluster_env.py          # ClusterAutoscalingEnv (Gymnasium)
│   ├── cluster_simulator.py    # Distributed cluster backend
│   ├── state_preprocessor.py  # Normalisation + sliding window T=8
│   └── reward_function.py     # Multi-objective reward
├── agent/
│   ├── neural_network.py      # Dueling DQN architecture
│   ├── per_buffer.py          # SumTree + Prioritized Replay Buffer
│   └── dqn_agent.py           # Double DQN agent + soft target update
├── training/
│   ├── trainer.py             # Full training pipeline
│   └── evaluator.py           # Periodic greedy evaluation
├── monitoring/
│   └── metrics_logger.py      # CSV / JSON / console logging
└── utils/
    └── config.py              # All hyperparameters (dataclasses)
main.py                        # CLI entrypoint
run_viz.py                     # Visualisation runner script
requirements.txt
visualization/
└── datacenter_viz.py          # Interactive Pygame dashboard
```


---

## 🎯 Problem

Traditional cluster autoscalers (HPA, VPA) react **reactively** to instantaneous metrics using hand-tuned thresholds. This system trains a **RL agent** to learn a proactive scaling policy that optimises:

| Objective | Signal |
|-----------|--------|
| 💰 Infrastructure cost | Proportional to active node count |
| ⚡ Performance (latency) | Average job completion time |
| 📈 Throughput | Jobs completed per timestep |
| 📋 SLA compliance | Latency + queue-length thresholds |

---

## 📊 State Space

Each observation = **sliding window** of the last **T = 8** cluster states.

| Feature | Description | Range |
|---------|-------------|-------|
| `cpu_utilization` | Avg CPU across active nodes | [0, 1] |
| `memory_utilization` | Avg RAM across active nodes | [0, 1] |
| `queue_length` | Pending jobs (normalised) | [0, 1] |
| `avg_latency` | Mean job completion time (normalised) | [0, 1] |
| `n_active_nodes` | Cluster size (normalised) | [0, 1] |
| `Δ cpu` | Rate of CPU change vs previous step | [−1, 1] |
| `Δ mem` | Rate of memory change | [−1, 1] |
| `util_ratio` | (cpu + mem) / 2 | [0, 1] |

**Flattened dimension**: 8 timesteps × 8 features = **64**

---

## ⚙️ Action Space

| ID | Action | Effect |
|----|--------|--------|
| 0 | `idle` | No scaling operation |
| 1 | `scale_up_1` | +1 worker node |
| 2 | `scale_up_2` | +2 worker nodes |
| 3 | `scale_down_1` | −1 worker node (drain + migrate) |
| 4 | `scale_down_2` | −2 worker nodes |
| 5 | `migrate` | Redistribute load, reduce hotspots |

**Safety constraints**: min 1 node · max 20 nodes · 30 s cooldown

---

## 🧠 Agent: Dueling Double DQN + PER

```
Input (64,)
   │
FC(256) → ReLU
   │
FC(256) → ReLU
   │
   ┌──────────────────────┐
   │                      │
FC(128) → ReLU      FC(128) → ReLU
   │                      │
  V(s) ∈ ℝ¹         A(s,a) ∈ ℝ⁶
   │                      │
   └────── Q = V + A − mean(A) ──┘
```

| Hyperparameter | Value |
|---------------|-------|
| Learning rate | 3e-4 (Adam) |
| Discount γ | 0.99 |
| Soft update τ | 0.005 |
| Batch size | 64 |
| Buffer size | 100 000 |
| PER α | 0.6 |
| PER β₀ → 1.0 | annealed over 200 000 steps |
| ε | 1.0 → 0.01 (exponential) |
| Gradient clip | norm ≤ 10 |
| Loss | Huber |

---

## 📈 Reward Function

```
R = −cost − λ·SLA_violation + throughput_bonus + efficiency_bonus − stability_penalty
```

| Component | Formula |
|-----------|---------|
| `cost` | 0.15 × nodes / max_nodes |
| `SLA_violation` | λ=2.5 × (latency_excess + queue_excess) |
| `throughput_bonus` | min(new_jobs / 15, 1.0) |
| `efficiency_bonus` | +0.25 if 50%≤CPU≤80%, −0.375 if CPU>90% |
| `stability_penalty` | 0.03 per scaling action |

---

## 🚀 Quick Start

```bash
# Install dependencies (includes pygame for visualization)
pip install -r requirements.txt

# Run the interactive Pygame dashboard
python run_viz.py

# Demo (random agent, 200 steps, CLI/visual output)
python main.py --mode demo

# Full training (300 000 steps, ~15 min on CPU)
python main.py --mode train

# Short experiment
python main.py --mode train --total_steps 50000 --warmup_steps 2000

# Evaluate best checkpoint
python main.py --mode eval --checkpoint checkpoints/best.pt
```

### 🎮 Controls for Interactive Visualization (`run_viz.py`)
When running the interactive simulation window, the following keyboard controls are active:
* **`1` - `5`** : Switch load scenarios in real time:
  * `1` : **Normal** (Standard harmonic Poisson workload)
  * `2` : **Spike** (Periodic intense workload bursts)
  * `3` : **Overload** (Constant extremely high workload)
  * `4` : **Attack** (DDoS style traffic surge followed by calm)
  * `5` : **Recovery** (Peak traffic followed by gradual return to normal)
* **`SPACE`** : Pause / Resume the simulation
* **`+` / `-`** : Accelerate or decelerate simulation speed (from `0.25x` up to `8.0x`)
* **`R`** : Reset the simulation and clear metrics
* **`ESC`** : Quit the interactive interface


---

## 📁 Outputs

| Path | Content |
|------|---------|
| `logs/training_metrics.csv` | Per-step reward, SLA, latency, throughput, Q-values, ε |
| `logs/eval_results.json` | Greedy evaluation snapshots |
| `logs/training.log` | Timestamped training log |
| `checkpoints/best.pt` | Best model (by eval reward) |
| `checkpoints/final.pt` | End-of-training checkpoint |
| `checkpoints/ckpt_step_N.pt` | Periodic saves |

---

## 🔬 Cluster Simulation

The backend simulates a realistic distributed compute infrastructure:

- **Compute layer**: configurable vCPUs (8–160) and RAM (32–640 GB) across 1–20 nodes
- **Scheduler**: Priority-FIFO with best-fit node selection to minimize fragmentation
- **Workload**: Multi-harmonic Poisson arrivals (hourly + daily cycles), random burst events (5% probability, ×2–5 intensity), mixed short/long job durations
- **Metrics**: CPU util, memory util, queue depth, avg latency — all evolving realistically
- **Scaling**: Cooldown-gated (30 s), drain-and-migrate on scale-down, automatic reuse of inactive nodes on scale-up

---

## 🔧 Extending the System

### Custom workload
Override `ClusterSimulator._generate_arrivals()` with your own arrival process.

### New actions
Add entries to `ClusterAutoscalingEnv._ACTION_MAP` and update `n_actions` in `EnvConfig`.

### LSTM temporal encoder
Replace the flat FC feature extractor in `DuelingDQN.features` with an LSTM or Transformer accepting shape `(T, F)`.

### Multi-agent / multi-cluster
Wrap multiple `ClusterSimulator` instances in a `VecEnv`-style parallel environment.
