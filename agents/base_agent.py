"""
Base agent interface.

All agents must implement:
  - select_action(obs)  →  action (int)
  - update(...)         →  loss or None
  - save(path)
  - load(path)

Design note: Agents are environment-agnostic.
They only interact with raw numpy arrays and scalar rewards.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseAgent(ABC):
    """Abstract base class for all RL agents."""

    def __init__(self, obs_dim: int, n_actions: int):
        self.obs_dim   = obs_dim
        self.n_actions = n_actions

    @abstractmethod
    def select_action(self, obs: np.ndarray, explore: bool = True) -> int:
        """
        Select an action given the current observation.

        Parameters
        ----------
        obs : np.ndarray  shape (obs_dim,)
        explore : bool
            If False, act greedily (evaluation mode).
        """
        ...

    @abstractmethod
    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """
        Update the agent from a single transition.

        Returns a scalar loss (or None if no gradient step taken this call).
        """
        ...

    def save(self, path: str) -> None:
        """Persist agent state to `path`."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement save()")

    def load(self, path: str) -> None:
        """Restore agent state from `path`."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement load()")

    def on_episode_end(self) -> None:
        """Optional hook called at the end of each episode (e.g. epsilon decay)."""
        pass
