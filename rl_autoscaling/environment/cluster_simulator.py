"""
Cluster Simulation Backend
==========================
Simulates a realistic distributed compute cluster with:
  • Multiple worker nodes (compute layer)
  • FIFO + priority job scheduler (scheduling layer)
  • Stochastic Poisson job arrivals with burst patterns
  • Prometheus-inspired metric generation (monitoring layer)
  • Cooldown-gated scaling operations with rollback support
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Data models

@dataclass
class Job:
    job_id: int
    arrival_time: float
    execution_time: float       # seconds
    priority: int = 1           # 1 (low) → 3 (high)
    cpu_requirement: float = 1.0
    memory_requirement_gb: float = 2.0
    start_time: Optional[float] = None


@dataclass
class WorkerNode:
    node_id: int
    vcpus: int
    memory_gb: int
    cpu_used: float = 0.0
    memory_used_gb: float = 0.0
    active: bool = True

    @property
    def cpu_utilization(self) -> float:
        return min(1.0, self.cpu_used / self.vcpus) if self.vcpus > 0 else 0.0

    @property
    def memory_utilization(self) -> float:
        return min(1.0, self.memory_used_gb / self.memory_gb) if self.memory_gb > 0 else 0.0

    @property
    def available_cpus(self) -> float:
        return max(0.0, self.vcpus - self.cpu_used)

    @property
    def available_memory_gb(self) -> float:
        return max(0.0, self.memory_gb - self.memory_used_gb)

    @property
    def is_idle(self) -> bool:
        return self.cpu_used < 0.05 * self.vcpus


@dataclass
class RunningJob:
    job: Job
    node_id: int
    start_time: float
    end_time: float


# Cluster simulator

class ClusterSimulator:
    """
    Realistic distributed cluster simulation.

    Workload model
    --------------
    • Base arrival rate follows a multi-harmonic sinusoid (hourly + daily cycle)
    • Gaussian noise added for realism
    • Random bursts (5 % probability per step, ×2–5 intensity)
    • Job execution times drawn from a mixed exponential distribution

    Scheduling
    ----------
    • Priority-FIFO: jobs sorted by (priority DESC, arrival_time ASC)
    • Best-fit node selection to minimise fragmentation
    """

    def __init__(self, config):
        self.config = config
        self.cc = config.cluster
        self._rng = np.random.default_rng()
        self.reset()

    # lifecycle

    def reset(self):
        self.nodes: List[WorkerNode] = []
        self.job_queue: deque = deque()      # pending jobs
        self.running_jobs: List[RunningJob] = []
        self.completed_jobs: int = 0
        self.total_latency: float = 0.0      # sum of queue-wait + execution
        self.current_time: float = 0.0
        self.last_scaling_time: float = -1e9
        self.job_counter: int = 0
        self.step_counter: int = 0
        self._anomaly_count: int = 0

        # Pre-allocate initial nodes
        for _ in range(self.cc.initial_nodes):
            self._create_node()

        logger.debug("ClusterSimulator reset — %d nodes", len(self.nodes))
        return self._compute_metrics()

    # internal node management

    def _create_node(self) -> Optional[WorkerNode]:
        if len(self.nodes) >= self.cc.max_nodes:
            return None
        node = WorkerNode(
            node_id=len(self.nodes),
            vcpus=self.cc.vcpus_per_node,
            memory_gb=self.cc.memory_per_node_gb,
        )
        self.nodes.append(node)
        return node

    def _deactivate_node(self, node: WorkerNode):
        """Drain a node: migrate its jobs back to the queue."""
        node.active = False
        # Evict running jobs from this node back to queue front
        evicted = [rj for rj in self.running_jobs if rj.node_id == node.node_id]
        for rj in evicted:
            rj.job.start_time = None          # reset start
            self.job_queue.appendleft(rj.job)  # priority re-queue
        self.running_jobs = [rj for rj in self.running_jobs
                             if rj.node_id != node.node_id]
        node.cpu_used = 0.0
        node.memory_used_gb = 0.0

    # workload generation

    def _generate_arrivals(self, dt: float) -> List[Job]:
        """Stochastic Poisson arrivals with multi-harmonic intensity."""
        t_hours = self.current_time / 3600.0

        # Multi-harmonic rate: hourly + daily patterns
        # nombre moyen d’arrivées par unité de temps.
        rate = (
            3.0
            + 2.0 * np.sin(2 * np.pi * t_hours)          # hourly cycle
            + 1.0 * np.sin(2 * np.pi * t_hours / 24.0)   # daily cycle
            + self._rng.normal(0, 0.3)                     # noise
        )
        rate = max(0.2, rate)

        # Random burst events
        if self._rng.random() < 0.05:
            rate *= self._rng.uniform(2.0, 5.0)

        n_arrivals = self._rng.poisson(rate * dt / 10.0)  # per-step scaling
        
        jobs = []
        for _ in range(int(n_arrivals)):
            # Mixed execution times: 70% short (≈15 s), 30% long (≈120 s)
            if self._rng.random() < 0.7:
                exec_time = self._rng.exponential(15.0)
            else:
                exec_time = self._rng.exponential(120.0)

            job = Job(
                job_id=self.job_counter,
                arrival_time=self.current_time,
                execution_time=max(1.0, exec_time),
                priority=int(self._rng.choice([1, 2, 3], p=[0.65, 0.25, 0.10])),
                cpu_requirement=float(self._rng.uniform(0.5, 2.5)),
                memory_requirement_gb=float(self._rng.uniform(1.0, 6.0)),
            )
            self.job_counter += 1
            jobs.append(job)
        return jobs

    # scheduling

    def _schedule_jobs(self):
        """Priority-FIFO + best-fit scheduling."""
        if not self.job_queue:
            return

        active_nodes = [n for n in self.nodes if n.active]
        if not active_nodes:
            return

        # Sort queue: priority DESC, arrival_time ASC
        sorted_queue = sorted(self.job_queue,
                              key=lambda j: (-j.priority, j.arrival_time))
        scheduled_ids = set()

        for job in sorted_queue:
            # Best-fit: choose node with smallest sufficient free CPU
            candidates = [
                n for n in active_nodes
                if n.available_cpus >= job.cpu_requirement
                and n.available_memory_gb >= job.memory_requirement_gb
            ]
            if not candidates:
                continue

            node = min(candidates, key=lambda n: n.available_cpus)
            node.cpu_used += job.cpu_requirement
            node.memory_used_gb += job.memory_requirement_gb
            end_t = self.current_time + job.execution_time
            self.running_jobs.append(
                RunningJob(job=job, node_id=node.node_id,
                           start_time=self.current_time, end_time=end_t)
            )
            scheduled_ids.add(job.job_id)

        # Remove scheduled jobs from queue
        self.job_queue = deque(
            j for j in self.job_queue if j.job_id not in scheduled_ids
        )

    # job completion

    def _complete_jobs(self) -> int:
        """Retire finished jobs and free node resources."""
        done, still_running = [], []
        for rj in self.running_jobs:
            if self.current_time >= rj.end_time:
                done.append(rj)
            else:
                still_running.append(rj)

        for rj in done:
            node = next((n for n in self.nodes if n.node_id == rj.node_id), None)
            if node:
                node.cpu_used = max(0.0, node.cpu_used - rj.job.cpu_requirement)
                node.memory_used_gb = max(0.0, node.memory_used_gb
                                          - rj.job.memory_requirement_gb)
            latency = self.current_time - rj.job.arrival_time
            self.total_latency += latency
            self.completed_jobs += 1

        self.running_jobs = still_running
        return len(done)

    # metrics

    def _compute_metrics(self) -> Dict:
        active = [n for n in self.nodes if n.active]
        n = len(active)

        if n == 0:
            return dict(cpu_utilization=0.0, memory_utilization=0.0,
                        queue_length=int(len(self.job_queue)),
                        avg_latency=0.0, n_active_nodes=0,
                        completed_jobs=self.completed_jobs)

        avg_cpu = float(np.mean([nd.cpu_utilization for nd in active]))
        avg_mem = float(np.mean([nd.memory_utilization for nd in active]))
        avg_lat = (self.total_latency / self.completed_jobs
                   if self.completed_jobs > 0 else 0.0)

        return dict(
            cpu_utilization=min(1.0, avg_cpu),
            memory_utilization=min(1.0, avg_mem),
            queue_length=int(len(self.job_queue)),
            avg_latency=float(avg_lat),
            n_active_nodes=int(n),
            completed_jobs=int(self.completed_jobs),
        )

    # public API

    def step(self, dt: float = 10.0) -> Dict:
        """Advance the simulation by *dt* seconds and return updated metrics."""
        self.current_time += dt
        self.step_counter += 1

        # 1. New job arrivals
        for job in self._generate_arrivals(dt):
            self.job_queue.append(job)

        # 2. Complete finished jobs first (frees resources for scheduler)
        self._complete_jobs()

        # 3. Schedule queued jobs
        self._schedule_jobs()

        return self._compute_metrics()

    # scaling actions

    @property
    def can_scale(self) -> bool:
        return (self.current_time - self.last_scaling_time) >= self.cc.cooldown_seconds

    def scale_up(self, n: int = 1) -> Tuple[bool, str]:
        if not self.can_scale:
            return False, "cooldown"
        
        active_count = sum(1 for nd in self.nodes if nd.active)
        if active_count >= self.cc.max_nodes:
            return False, "max_nodes_reached"

        added = 0
        for _ in range(n):
            if sum(1 for nd in self.nodes if nd.active) >= self.cc.max_nodes:
                break 
            # Reuse inactive nodes first, then create new ones
            inactive = next((nd for nd in self.nodes if not nd.active), None)
            if inactive:
                inactive.active = True
                inactive.cpu_used = 0.0
                inactive.memory_used_gb = 0.0
            else:
                self._create_node()
            added += 1

        if added:
            self.last_scaling_time = self.current_time
            logger.info("scale_up +%d nodes (total active=%d)",
                        added, sum(1 for nd in self.nodes if nd.active))
        return added > 0, f"added_{added}_nodes"

    def scale_down(self, n: int = 1) -> Tuple[bool, str]:
        if not self.can_scale:
            return False, "cooldown"
        active = [nd for nd in self.nodes if nd.active]
        if len(active) <= self.cc.min_nodes:
            return False, "min_nodes_reached"

        # Remove least-utilized nodes first
        targets = sorted(active, key=lambda nd: nd.cpu_utilization)
        removed = 0
        for nd in targets:
            if sum(1 for x in self.nodes if x.active) <= self.cc.min_nodes:
                break
            self._deactivate_node(nd)
            removed += 1
            if removed >= n:
                break

        if removed:
            self.last_scaling_time = self.current_time
            logger.info("scale_down -%d nodes (total active=%d)",
                        removed, sum(1 for nd in self.nodes if nd.active))
        return removed > 0, f"removed_{removed}_nodes"

    def migrate(self) -> Tuple[bool, str]:
        """Redistribute workload to reduce hotspots."""
        active = [nd for nd in self.nodes if nd.active]
        if len(active) < 2:
            return False, "not_enough_nodes"

        total_cpu = sum(nd.cpu_used for nd in active)
        total_mem = sum(nd.memory_used_gb for nd in active)
        target_cpu = total_cpu / len(active)
        target_mem = total_mem / len(active)

        # Smooth redistribution with small random perturbation
        raw = [self._rng.uniform(0.88, 1.12) for _ in active]
        scale = len(active) / sum(raw)
        factors = [r * scale for r in raw]

        for nd, factor in zip(active, factors):
            nd.cpu_used = float(np.clip(
                target_cpu * factor, 0.0, nd.vcpus))
            nd.memory_used_gb = float(np.clip(
            target_mem * factor, 0.0, nd.memory_gb))

        self.last_scaling_time = self.current_time
        return True, "migrated"

    # helpers

    def get_node_stats(self) -> List[Dict]:
        return [
            dict(node_id=nd.node_id, 
                 active=nd.active,
                 cpu_util=nd.cpu_utilization,
                 mem_util=nd.memory_utilization)
            for nd in self.nodes
        ]
