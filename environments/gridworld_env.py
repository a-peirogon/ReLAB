"""
GridWorld / Maze Environment — the tabular-RL fundamentals environment.

Unlike Snake or RobotNav, the observation here is deliberately *minimal*:
just the agent's integer (row, col) position. No shaping, no relative
geometry — the point of this environment is to watch a Q-table fill in
from scratch, cell by cell, as value propagates backward from the goal
through the Bellman update. That only works if state == position, so a
tabular agent has exactly one row per grid cell.

Three built-in layouts (`layout=`):
    "empty" — open size x size grid, no walls. Sanity-check baseline.
    "maze"  — a fixed corridor maze. Introduces walls / dead ends.
    "cliff" — the classic Sutton & Barto "Cliff Walking" gridworld:
              a row of pits between start and goal. Textbook example of
              on-policy (SARSA) vs off-policy (Q-learning) risk-taking,
              though this env only ships Q-learning for now.

Design note: PURE — no RL logic. Raw state (`agent_pos`, `walls`, `pits`,
`goal`) is exposed as public attributes for the renderer, same convention
as SnakeEnv / RobotNavEnv.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# row-delta, col-delta for each action
_ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 0=up 1=down 2=left 3=right
ACTION_NAMES = ["Up", "Down", "Left", "Right"]

MAZE_LAYOUT = [
    "############",
    "#S....#....#",
    "#.####.##.##",
    "#.#..#....##",
    "#.#.##.###.#",
    "#...#......#",
    "###.#.####.#",
    "#...#.#..#.#",
    "#.###.#.##.#",
    "#.....#...G#",
    "############",
]

CLIFF_LAYOUT = [
    "............",
    "............",
    "............",
    "SPPPPPPPPPPG",
]


class GridWorldEnv(gym.Env):
    """
    Discrete grid navigation task — the "hello world" of tabular RL.

    Parameters
    ----------
    layout : {"empty", "maze", "cliff"}
    size : int
        Only used when layout == "empty" (grid is size x size).
    step_penalty : float
        Reward applied every non-terminal step.
    max_steps : int
    """

    metadata = {"render_modes": ["ansi", "none"]}

    def __init__(
        self,
        layout: str = "maze",
        size: int = 10,
        step_penalty: float = -0.01,
        goal_reward: float = 1.0,
        pit_reward: float = -1.0,
        max_steps: int = 200,
        render_mode: str = "none",
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.layout_name  = layout
        self.step_penalty = step_penalty
        self.goal_reward   = goal_reward
        self.pit_reward     = pit_reward
        self.max_steps       = max_steps
        self.render_mode     = render_mode

        rows = self._build_layout(layout, size)
        self.n_rows = len(rows)
        self.n_cols = len(rows[0])

        self.walls: set[tuple[int, int]] = set()
        self.pits:  set[tuple[int, int]] = set()
        self.start: tuple[int, int] = (0, 0)
        self.goal:  tuple[int, int] = (self.n_rows - 1, self.n_cols - 1)

        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                if ch == "#":
                    self.walls.add((r, c))
                elif ch == "P":
                    self.pits.add((r, c))
                elif ch == "S":
                    self.start = (r, c)
                elif ch == "G":
                    self.goal = (r, c)

        self.observation_space = spaces.Box(
            low=0.0,
            high=float(max(self.n_rows, self.n_cols)),
            shape=(2,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        self._np_rng = np.random.default_rng(seed)

        self.agent_pos: tuple[int, int] = self.start
        self._step_count = 0
        self.reached_goal = False
        self.fell_in_pit  = False

    # ── Layout construction ──────────────────────────────────────────────────

    @staticmethod
    def _build_layout(layout: str, size: int) -> list[str]:
        if layout == "maze":
            return MAZE_LAYOUT
        if layout == "cliff":
            return CLIFF_LAYOUT
        if layout == "empty":
            rows = ["." * size for _ in range(size)]
            rows[0]  = "S" + rows[0][1:]
            rows[-1] = rows[-1][:-1] + "G"
            return rows
        raise ValueError(f"Unknown layout: {layout!r} (choose empty/maze/cliff)")

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)

        self.agent_pos     = self.start
        self._step_count    = 0
        self.reached_goal   = False
        self.fell_in_pit    = False

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        dr, dc = _ACTIONS[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc

        if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols and (nr, nc) not in self.walls:
            self.agent_pos = (nr, nc)

        self._step_count += 1
        reward = self.step_penalty
        terminated = False

        if self.agent_pos == self.goal:
            terminated = True
            self.reached_goal = True
            reward = self.goal_reward
        elif self.agent_pos in self.pits:
            terminated = True
            self.fell_in_pit = True
            reward = self.pit_reward

        truncated = self._step_count >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> Optional[str]:
        if self.render_mode == "ansi":
            return self._render_ansi()
        return None

    def close(self):
        pass

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        return np.array(self.agent_pos, dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            "score": int(self.reached_goal),
            "steps": self._step_count,
            "fell_in_pit": self.fell_in_pit,
        }

    def _render_ansi(self) -> str:
        grid = [["." for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        for (r, c) in self.walls:
            grid[r][c] = "#"
        for (r, c) in self.pits:
            grid[r][c] = "P"
        gr, gc = self.goal
        grid[gr][gc] = "G"
        ar, ac = self.agent_pos
        grid[ar][ac] = "A"
        lines = ["".join(row) for row in grid]
        lines.append(f"step {self._step_count}  pos={self.agent_pos}")
        return "\n".join(lines)
