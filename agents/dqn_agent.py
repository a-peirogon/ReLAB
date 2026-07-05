from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agents.base_agent import BaseAgent

@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool

class ReplayBuffer:

    def __init__(self, capacity: int):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DQNAgent(BaseAgent):

    def __init__(self, obs_dim: int=11, n_actions: int=4, hidden_dim: int=128, lr: float=0.001, gamma: float=0.99, epsilon_start: float=1.0, epsilon_end: float=0.01, epsilon_decay: float=0.995, buffer_capacity: int=50000, batch_size: int=64, target_update_freq: int=500, device: str='cpu'):
        super().__init__(obs_dim, n_actions)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        self.policy_net = MLP(obs_dim, hidden_dim, n_actions).to(self.device)
        self.target_net = MLP(obs_dim, hidden_dim, n_actions).to(self.device)
        self._sync_target()
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self._grad_steps = 0

    def select_action(self, obs: np.ndarray, explore: bool=True) -> int:
        if explore and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        self.policy_net.eval()
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q_values = self.policy_net(obs_t)
        self.policy_net.train()
        return int(q_values.argmax(dim=1).item())

    def update(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> Optional[float]:
        self.replay_buffer.push(obs, action, reward, next_obs, done)
        if len(self.replay_buffer) < self.batch_size:
            return None
        return self._gradient_step()

    def _gradient_step(self) -> float:
        batch = self.replay_buffer.sample(self.batch_size)
        obs_b = torch.FloatTensor(np.array([t.obs for t in batch])).to(self.device)
        action_b = torch.LongTensor(np.array([t.action for t in batch])).to(self.device)
        reward_b = torch.FloatTensor(np.array([t.reward for t in batch])).to(self.device)
        next_obs_b = torch.FloatTensor(np.array([t.next_obs for t in batch])).to(self.device)
        done_b = torch.FloatTensor(np.array([t.done for t in batch])).to(self.device)
        q_values = self.policy_net(obs_b).gather(1, action_b.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q = self.target_net(next_obs_b).max(dim=1).values
            q_targets = reward_b + self.gamma * max_next_q * (1.0 - done_b)
        loss = self.loss_fn(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self._grad_steps += 1
        if self._grad_steps % self.target_update_freq == 0:
            self._sync_target()
        return loss.item()

    def on_episode_end(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        torch.save({'policy_state_dict': self.policy_net.state_dict(), 'target_state_dict': self.target_net.state_dict(), 'optimizer_state': self.optimizer.state_dict(), 'epsilon': self.epsilon, 'grad_steps': self._grad_steps}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy_state_dict'])
        self.target_net.load_state_dict(ckpt['target_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.epsilon = ckpt['epsilon']
        self._grad_steps = ckpt['grad_steps']

    def _sync_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())
