from __future__ import annotations
import random
from collections import deque
from typing import Optional
import gymnasium as gym
import numpy as np
from gymnasium import spaces
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
DIRECTION_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
TURN_RIGHT = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}
TURN_LEFT = {UP: LEFT, LEFT: DOWN, DOWN: RIGHT, RIGHT: UP}

class SnakeEnv(gym.Env):
    metadata = {'render_modes': ['ansi', 'none']}

    def __init__(self, grid_size: int=10, step_penalty: float=-0.01, max_steps: int=1000, render_mode: str='none', seed: Optional[int]=None):
        super().__init__()
        self.grid_size = grid_size
        self.step_penalty = step_penalty
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self._np_rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)
        self.snake: deque[tuple[int, int]] = deque()
        self.direction: int = RIGHT
        self.food: tuple[int, int] = (0, 0)
        self.score: int = 0
        self._step_count: int = 0

    def reset(self, *, seed: Optional[int]=None, options: Optional[dict]=None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)
            self._py_rng = random.Random(seed)
        mid = self.grid_size // 2
        self.snake = deque([(mid, mid), (mid, mid - 1), (mid, mid - 2)])
        self.direction = RIGHT
        self.score = 0
        self._step_count = 0
        self._place_food()
        obs = self._get_obs()
        info = self._get_info()
        return (obs, info)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f'Invalid action: {action}'
        if not self._is_opposite(action, self.direction):
            self.direction = action
        dr, dc = DIRECTION_DELTAS[self.direction]
        head_r, head_c = self.snake[0]
        new_head = (head_r + dr, head_c + dc)
        terminated = False
        reward = self.step_penalty
        if self._is_collision(new_head):
            terminated = True
            reward = -1.0
            obs = self._get_obs()
            return (obs, reward, terminated, False, self._get_info())
        self.snake.appendleft(new_head)
        if new_head == self.food:
            self.score += 1
            reward = 1.0
            self._place_food()
        else:
            self.snake.pop()
        self._step_count += 1
        truncated = self._step_count >= self.max_steps
        obs = self._get_obs()
        return (obs, reward, terminated, truncated, self._get_info())

    def render(self) -> Optional[str]:
        if self.render_mode == 'ansi':
            return self._render_ansi()
        return None

    def close(self):
        pass

    def _get_obs(self) -> np.ndarray:
        head_r, head_c = self.snake[0]
        food_r, food_c = self.food
        straight_cell = self._cell_ahead(self.direction)
        right_cell = self._cell_ahead(TURN_RIGHT[self.direction])
        left_cell = self._cell_ahead(TURN_LEFT[self.direction])
        obs = np.array([float(self._is_collision(straight_cell)), float(self._is_collision(right_cell)), float(self._is_collision(left_cell)), float(self.direction == UP), float(self.direction == DOWN), float(self.direction == LEFT), float(self.direction == RIGHT), float(food_r < head_r), float(food_r > head_r), float(food_c < head_c), float(food_c > head_c)], dtype=np.float32)
        return obs

    def _place_food(self) -> None:
        body_set = set(self.snake)
        empty_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if (r, c) not in body_set]
        if not empty_cells:
            return
        idx = int(self._np_rng.integers(0, len(empty_cells)))
        self.food = empty_cells[idx]

    def _is_collision(self, cell: tuple[int, int]) -> bool:
        r, c = cell
        if r < 0 or r >= self.grid_size or c < 0 or (c >= self.grid_size):
            return True
        if cell in list(self.snake)[1:]:
            return True
        return False

    def _cell_ahead(self, direction: int) -> tuple[int, int]:
        dr, dc = DIRECTION_DELTAS[direction]
        head_r, head_c = self.snake[0]
        return (head_r + dr, head_c + dc)

    @staticmethod
    def _is_opposite(a: int, b: int) -> bool:
        opposites = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
        return opposites[a] == b

    def _get_info(self) -> dict:
        return {'score': self.score, 'snake_length': len(self.snake), 'steps': self._step_count}

    def _render_ansi(self) -> str:
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        fr, fc = self.food
        grid[fr][fc] = 'F'
        for i, (r, c) in enumerate(self.snake):
            grid[r][c] = 'H' if i == 0 else 'o'
        border = '+' + '-' * self.grid_size + '+'
        rows = [border]
        for row in grid:
            rows.append('|' + ''.join(row) + '|')
        rows.append(border)
        rows.append(f'Score: {self.score}  Steps: {self._step_count}')
        return '\n'.join(rows)
