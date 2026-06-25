"""
Trainer — decoupled training loop.

Responsibilities:
  - Episode loop
  - Env ↔ Agent interaction
  - Reward / stats tracking
  - Periodic evaluation
  - Checkpointing
  - CSV logging

Design note: Trainer has NO knowledge of agent internals.
It only calls select_action(), update(), on_episode_end(), save(), load().
This makes it compatible with ANY BaseAgent subclass.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Union

from agents.base_agent import BaseAgent
from configs.config import DQNConfig, QLearningConfig
from environments.snake_env import SnakeEnv
from utils.reproducibility import set_seed
from utils.stats import CSVLogger, EpisodeStats, MovingAverage

TrainerConfig = Union[DQNConfig, QLearningConfig]


class Trainer:
    """
    Environment-agnostic training loop for episodic RL.

    Parameters
    ----------
    env   : SnakeEnv (or any Gymnasium env)
    agent : BaseAgent subclass
    cfg   : DQNConfig or QLearningConfig (train sub-config used)
    """

    def __init__(self, env: SnakeEnv, agent: BaseAgent, cfg: TrainerConfig):
        self.env   = env
        self.agent = agent
        self.cfg   = cfg
        self.train_cfg = cfg.train

        # Logging
        os.makedirs(self.train_cfg.experiment_dir, exist_ok=True)
        self.csv_logger = CSVLogger(
            path=os.path.join(self.train_cfg.experiment_dir, "train_log.csv"),
            fields=["episode", "reward", "score", "steps", "epsilon", "loss_mean", "ma100_reward"],
        )

        # Trackers
        self.reward_ma = MovingAverage(window=100)
        self._best_eval_reward: float = float("-inf")

    # ── Main train loop ────────────────────────────────────────────────────────

    def train(self) -> None:
        set_seed(self.train_cfg.seed)
        print(f"\n{'='*60}")
        print(f"  Training  |  {self.agent.__class__.__name__}")
        print(f"  Episodes  |  {self.train_cfg.n_episodes}")
        print(f"  Dir       |  {self.train_cfg.experiment_dir}")
        print(f"{'='*60}\n")

        for episode in range(1, self.train_cfg.n_episodes + 1):
            ep_stats, losses = self._run_episode(train=True)
            self.reward_ma.push(ep_stats.total_reward)

            # Epsilon (only DQN/QLearning have it)
            epsilon = getattr(self.agent, "epsilon", 0.0)
            loss_mean = sum(losses) / len(losses) if losses else 0.0

            # Log to CSV
            self.csv_logger.log(
                episode=episode,
                reward=round(ep_stats.total_reward, 4),
                score=ep_stats.score,
                steps=ep_stats.steps,
                epsilon=round(epsilon, 4),
                loss_mean=round(loss_mean, 6),
                ma100_reward=round(self.reward_ma.mean, 4),
            )

            # Console print
            if episode % self.train_cfg.log_interval == 0:
                self._print_progress(episode, ep_stats, epsilon, loss_mean)

            # Periodic evaluation
            if episode % self.train_cfg.eval_interval == 0:
                eval_reward = self.evaluate(n_episodes=self.train_cfg.eval_episodes, verbose=False)
                print(f"  [Eval ep {episode:>5}]  mean_reward={eval_reward:.2f}")
                if eval_reward > self._best_eval_reward:
                    self._best_eval_reward = eval_reward
                    best_path = os.path.join(self.train_cfg.experiment_dir, "checkpoint_best")
                    self._save(best_path)
                    print(f"  ✓ New best eval reward → saved checkpoint_best")

            # Periodic checkpoint
            if episode % self.train_cfg.save_interval == 0:
                ckpt_path = os.path.join(
                    self.train_cfg.experiment_dir, f"checkpoint_ep{episode}"
                )
                self._save(ckpt_path)

            # Episode end hook (e.g. epsilon decay)
            self.agent.on_episode_end()

        # Final checkpoint
        final_path = os.path.join(self.train_cfg.experiment_dir, "checkpoint_final")
        self._save(final_path)
        print(f"\nTraining complete. Final checkpoint → {final_path}")

    # ── Single episode ─────────────────────────────────────────────────────────

    def _run_episode(self, train: bool) -> tuple[EpisodeStats, list[float]]:
        obs, info = self.env.reset()
        stats = EpisodeStats()
        losses: list[float] = []

        while True:
            action = self.agent.select_action(obs, explore=train)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            stats.update(reward, info)

            if train:
                loss = self.agent.update(obs, action, reward, next_obs, done)
                if loss is not None:
                    losses.append(loss)

            obs = next_obs
            if done:
                break

        return stats, losses

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def evaluate(self, n_episodes: int = 10, verbose: bool = True) -> float:
        """Run n_episodes in greedy mode; return mean episode reward."""
        total_rewards = []
        for _ in range(n_episodes):
            stats, _ = self._run_episode(train=False)
            total_rewards.append(stats.total_reward)
        mean = sum(total_rewards) / len(total_rewards)
        if verbose:
            print(f"Eval ({n_episodes} eps): mean_reward={mean:.2f}  "
                  f"min={min(total_rewards):.2f}  max={max(total_rewards):.2f}")
        return mean

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Extension handled by agent
        ext = ".pkl" if "QLearning" in self.agent.__class__.__name__ else ".pt"
        self.agent.save(path + ext)

    def _print_progress(
        self,
        episode: int,
        stats: EpisodeStats,
        epsilon: float,
        loss_mean: float,
    ) -> None:
        print(
            f"  ep {episode:>5} | "
            f"reward {stats.total_reward:>7.2f} | "
            f"score {stats.score:>3} | "
            f"steps {stats.steps:>4} | "
            f"ε {epsilon:.3f} | "
            f"MA-100 {self.reward_ma.mean:>7.2f} | "
            f"loss {loss_mean:.5f}"
        )
