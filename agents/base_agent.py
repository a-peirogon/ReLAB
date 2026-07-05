from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class BaseAgent(ABC):

    def __init__(self, obs_dim: int, n_actions: int):
        self.obs_dim = obs_dim
        self.n_actions = n_actions

    @abstractmethod
    def select_action(self, obs: np.ndarray, explore: bool=True) -> int:
        ...

    @abstractmethod
    def update(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> Optional[float]:
        ...

    def save(self, path: str) -> None:
        raise NotImplementedError(f'{self.__class__.__name__} does not implement save()')

    def load(self, path: str) -> None:
        raise NotImplementedError(f'{self.__class__.__name__} does not implement load()')

    def on_episode_end(self) -> None:
        pass
