"""
Experiment configuration using Python dataclasses.

Usage:
    from configs.config import DQNConfig, QLearningConfig, EnvConfig

    cfg = DQNConfig()               # defaults
    cfg = DQNConfig(lr=5e-4)        # override one field
    cfg = DQNConfig(**yaml_dict)    # from loaded YAML
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvConfig:
    """Snake environment settings."""
    grid_size:    int   = 10
    step_penalty: float = -0.01
    max_steps:    int   = 1000
    render_mode:  str   = "none"


@dataclass
class TrainConfig:
    """Training loop settings (shared by all agents)."""
    n_episodes:    int   = 2_000
    eval_interval: int   = 100       # evaluate every N episodes
    eval_episodes: int   = 10        # episodes per evaluation run
    log_interval:  int   = 50        # print stats every N episodes
    save_interval: int   = 500       # save checkpoint every N episodes
    experiment_dir: str  = "experiments/run_001"
    seed:           Optional[int] = 42


@dataclass
class DQNConfig:
    """DQN-specific hyperparameters."""
    hidden_dim:         int   = 128
    lr:                 float = 1e-3
    gamma:              float = 0.99
    epsilon_start:      float = 1.0
    epsilon_end:        float = 0.01
    epsilon_decay:      float = 0.995
    buffer_capacity:    int   = 50_000
    batch_size:         int   = 64
    target_update_freq: int   = 500
    device:             str   = "cpu"

    # Nested configs
    env:   EnvConfig   = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class QLearningConfig:
    """Q-Learning-specific hyperparameters."""
    lr:            float = 0.1
    gamma:         float = 0.9
    epsilon_start: float = 1.0
    epsilon_end:   float = 0.01
    epsilon_decay: float = 0.995

    env:   EnvConfig   = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class RobotNavEnvConfig:
    """Robot Navigation environment settings."""
    world_size:     float = 12.0
    n_obstacles:    int   = 8
    obstacle_speed: float = 0.8
    dt:             float = 0.15
    max_steps:      int   = 400
    step_penalty:   float = -0.01
    render_mode:    str   = "none"


@dataclass
class DWAConfig:
    """Dynamic Window Approach — classical, non-learning planner."""
    n_obstacles:   int   = 8
    dt:            float = 0.15
    horizon_steps: int   = 8
    weights: tuple[float, float, float] = (1.2, 1.0, 0.6)  # heading, clearance, speed

    env: RobotNavEnvConfig = field(default_factory=RobotNavEnvConfig)


@dataclass
class RobotNavDQNConfig:
    """DQN hyperparameters, tuned defaults for RobotNavEnv."""
    hidden_dim:         int   = 128
    lr:                 float = 5e-4
    gamma:              float = 0.98
    epsilon_start:      float = 1.0
    epsilon_end:        float = 0.02
    epsilon_decay:      float = 0.998
    buffer_capacity:    int   = 100_000
    batch_size:         int   = 64
    target_update_freq: int   = 500
    device:             str   = "cpu"

    env:   RobotNavEnvConfig = field(default_factory=RobotNavEnvConfig)
    train: TrainConfig       = field(default_factory=lambda: TrainConfig(
        n_episodes=3_000, experiment_dir="experiments/robot_nav_dqn_run"
    ))
