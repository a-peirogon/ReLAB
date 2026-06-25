"""
run_train.py — Entry point for training an RL agent on Snake.

Usage:
    python run_train.py                         # DQN with defaults
    python run_train.py --agent qlearning       # Tabular Q-learning
    python run_train.py --episodes 5000         # Custom episode count
    python run_train.py --grid 15               # Larger grid
    python run_train.py --exp experiments/exp1  # Custom experiment dir
    python run_train.py --seed 0                # Fix seed
"""

import argparse
import random
import sys

from agents.dqn_agent import DQNAgent
from agents.qlearning_agent import QLearningAgent
from configs.config import DQNConfig, EnvConfig, QLearningConfig, TrainConfig
from environments.snake_env import SnakeEnv
from training.trainer import Trainer


def set_seed(seed: int):
    """Fija semillas para reproducibilidad."""
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agent on Snake")
    parser.add_argument("--agent", choices=["dqn", "qlearning"], default="dqn")
    parser.add_argument("--episodes", type=int,   default=None)
    parser.add_argument("--grid",     type=int,   default=None)
    parser.add_argument("--lr",       type=float, default=None)
    parser.add_argument("--gamma",    type=float, default=None)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--exp",      type=str,   default=None,
                        help="Experiment output directory")
    parser.add_argument("--device",   type=str,   default="cpu",
                        help="PyTorch device (cpu / cuda) — DQN only")
    return parser.parse_args()


def build_dqn(args) -> tuple[SnakeEnv, DQNAgent, DQNConfig]:
    cfg = DQNConfig(
        lr=args.lr     or 1e-3,
        gamma=args.gamma or 0.99,
        device=args.device,
        env=EnvConfig(grid_size=args.grid or 10),
        train=TrainConfig(
            n_episodes=args.episodes or 2_000,
            seed=args.seed,
            experiment_dir=args.exp or "experiments/dqn_run",
        ),
    )
    env   = SnakeEnv(**cfg.env.__dict__)
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_dim=cfg.hidden_dim,
        lr=cfg.lr,
        gamma=cfg.gamma,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay=cfg.epsilon_decay,
        buffer_capacity=cfg.buffer_capacity,
        batch_size=cfg.batch_size,
        target_update_freq=cfg.target_update_freq,
        device=cfg.device,
    )
    return env, agent, cfg


def build_qlearning(args) -> tuple[SnakeEnv, QLearningAgent, QLearningConfig]:
    cfg = QLearningConfig(
        lr=args.lr     or 0.1,
        gamma=args.gamma or 0.9,
        env=EnvConfig(grid_size=args.grid or 10),
        train=TrainConfig(
            n_episodes=args.episodes or 2_000,
            seed=args.seed,
            experiment_dir=args.exp or "experiments/qlearning_run",
        ),
    )
    env = SnakeEnv(**cfg.env.__dict__)
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = QLearningAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        lr=cfg.lr,
        gamma=cfg.gamma,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay=cfg.epsilon_decay,
    )
    return env, agent, cfg


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.agent == "dqn":
        env, agent, cfg = build_dqn(args)
    elif args.agent == "qlearning":
        env, agent, cfg = build_qlearning(args)
    else:
        print(f"Unknown agent: {args.agent}")
        sys.exit(1)

    trainer = Trainer(env, agent, cfg)
    trainer.train()


if __name__ == "__main__":
    main()
