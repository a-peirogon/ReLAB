from __future__ import annotations
import pickle
from collections import defaultdict
from typing import Optional
import numpy as np
from agents.base_agent import BaseAgent

class QLearningAgent(BaseAgent):

    def __init__(self, obs_dim: int=11, n_actions: int=4, lr: float=0.1, gamma: float=0.9, epsilon_start: float=1.0, epsilon_end: float=0.01, epsilon_decay: float=0.995):
        super().__init__(obs_dim, n_actions)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float64))

    def select_action(self, obs: np.ndarray, explore: bool=True) -> int:
        if explore and np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        state = self._obs_to_key(obs)
        return int(np.argmax(self.q_table[state]))

    def update(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> float:
        s = self._obs_to_key(obs)
        s_ = self._obs_to_key(next_obs)
        q_current = self.q_table[s][action]
        q_target = reward if done else reward + self.gamma * np.max(self.q_table[s_])
        td_error = q_target - q_current
        self.q_table[s][action] += self.lr * td_error
        return abs(td_error)

    def on_episode_end(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        state = {'q_table': dict(self.q_table), 'epsilon': self.epsilon, 'lr': self.lr, 'gamma': self.gamma}
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions), state['q_table'])
        self.epsilon = state['epsilon']
        self.lr = state['lr']
        self.gamma = state['gamma']

    @staticmethod
    def _obs_to_key(obs: np.ndarray) -> tuple:
        return tuple(obs.astype(int).tolist())

    @property
    def table_size(self) -> int:
        return len(self.q_table)
