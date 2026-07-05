from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EnvConfig:
    grid_size: int = 10
    step_penalty: float = -0.01
    max_steps: int = 1000
    render_mode: str = 'none'

@dataclass
class TrainConfig:
    n_episodes: int = 2000
    eval_interval: int = 100
    eval_episodes: int = 10
    log_interval: int = 50
    save_interval: int = 500
    experiment_dir: str = 'experiments/run_001'
    seed: Optional[int] = 42

@dataclass
class DQNConfig:
    hidden_dim: int = 128
    lr: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_capacity: int = 50000
    batch_size: int = 64
    target_update_freq: int = 500
    device: str = 'cpu'
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

@dataclass
class QLearningConfig:
    lr: float = 0.1
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

@dataclass
class RobotNavEnvConfig:
    world_size: float = 12.0
    n_obstacles: int = 8
    obstacle_speed: float = 0.8
    dt: float = 0.15
    max_steps: int = 400
    step_penalty: float = -0.01
    render_mode: str = 'none'

@dataclass
class DWAConfig:
    n_obstacles: int = 8
    dt: float = 0.15
    horizon_steps: int = 8
    weights: tuple[float, float, float] = (1.2, 1.0, 0.6)
    env: RobotNavEnvConfig = field(default_factory=RobotNavEnvConfig)

@dataclass
class RobotNavDQNConfig:
    hidden_dim: int = 128
    lr: float = 0.0005
    gamma: float = 0.98
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay: float = 0.998
    buffer_capacity: int = 100000
    batch_size: int = 64
    target_update_freq: int = 500
    device: str = 'cpu'
    env: RobotNavEnvConfig = field(default_factory=RobotNavEnvConfig)
    train: TrainConfig = field(default_factory=lambda: TrainConfig(n_episodes=3000, experiment_dir='experiments/robot_nav_dqn_run'))
