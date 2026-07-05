"""
Robot Navigation Environment — Gymnasium-compatible, continuous 2D world.

A differential-drive (unicycle) robot must reach a goal point while
avoiding circular obstacles that move with constant velocity and bounce
off the arena walls. Inspired by classic local-planning demos such as
the Dynamic Window Approach (DWA).

Observation vector (5 + 5*N_OBSTACLES floats), all EGOCENTRIC
(relative to the robot's current pose and heading — this is what lets
a model-based planner like DWA use the same vector as the RL agents):

    [0] v_norm        current linear velocity / V_MAX        in [0, 1]
    [1] omega_norm    current angular velocity / OMEGA_MAX    in [-1, 1]
    [2] goal_dist_norm   distance to goal / world diagonal    in [0, 1]
    [3] sin(goal_bearing)   goal angle relative to heading
    [4] cos(goal_bearing)
    for each of N_OBSTACLES obstacles (sorted by distance, nearest first):
    [.] obs_dist_norm       distance / world diagonal
    [.] sin(obs_bearing)    obstacle angle relative to heading
    [.] cos(obs_bearing)
    [.] rel_vx_norm         obstacle velocity relative to robot, robot frame
    [.] rel_vy_norm

Actions (Discrete 9): combinations of {decelerate, hold, accelerate} on
linear velocity x {turn left, straight, turn right} on angular velocity.

Design note: This class is intentionally PURE — no RL logic, no agent
references. Raw simulation state (pose, obstacles, goal) is exposed as
public attributes for renderers, exactly like SnakeEnv exposes
`snake`/`food`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ── Physical constants ─────────────────────────────────────────────────────────
V_MAX     = 1.5      # m/s, max linear velocity
V_MIN     = 0.0       # no reverse
OMEGA_MAX = 2.5       # rad/s, max yaw rate
ACCEL     = 1.5        # m/s^2 applied per action
ALPHA     = 3.0        # rad/s^2 applied per action

ROBOT_RADIUS = 0.3
GOAL_RADIUS  = 0.4

# Discrete action → (dv_sign, domega_sign)
_ACTIONS = [(dv, dw) for dv in (-1, 0, 1) for dw in (-1, 0, 1)]  # 9 actions


@dataclass
class Obstacle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float


class RobotNavEnv(gym.Env):
    """
    Continuous 2D navigation task: reach the goal, avoid moving obstacles.

    Parameters
    ----------
    world_size : float
        Side length of the square arena (meters).
    n_obstacles : int
        Number of moving circular obstacles.
    dt : float
        Simulation timestep (seconds).
    max_steps : int
        Truncation horizon.
    step_penalty : float
        Small constant penalty applied every step.
    seed : Optional[int]
    """

    metadata = {"render_modes": ["ansi", "none"]}

    N_OBSTACLES_DEFAULT = 8

    def __init__(
        self,
        world_size: float = 12.0,
        n_obstacles: int = N_OBSTACLES_DEFAULT,
        obstacle_speed: float = 0.8,
        dt: float = 0.15,
        max_steps: int = 400,
        step_penalty: float = -0.01,
        render_mode: str = "none",
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.world_size     = world_size
        self.n_obstacles     = n_obstacles
        self.obstacle_speed = obstacle_speed
        self.dt             = dt
        self.max_steps       = max_steps
        self.step_penalty   = step_penalty
        self.render_mode     = render_mode

        self._world_diag = math.hypot(world_size, world_size)

        obs_dim = 5 + 5 * n_obstacles
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(_ACTIONS))

        self._np_rng = np.random.default_rng(seed)

        # State (set in reset)
        self.x = self.y = self.theta = self.v = self.omega = 0.0
        self.goal: tuple[float, float] = (0.0, 0.0)
        self.obstacles: list[Obstacle] = []
        self._step_count = 0
        self._prev_goal_dist = 0.0
        self.collided = False
        self.reached_goal = False

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)

        margin = self.world_size * 0.1
        self.x = float(self._np_rng.uniform(margin, self.world_size - margin))
        self.y = float(self._np_rng.uniform(margin, self.world_size - margin))
        self.theta = float(self._np_rng.uniform(-math.pi, math.pi))
        self.v = 0.0
        self.omega = 0.0

        self.goal = self._sample_far_point(self.x, self.y, min_dist=self.world_size * 0.5)
        self.obstacles = [self._sample_obstacle() for _ in range(self.n_obstacles)]

        self._step_count = 0
        self.collided = False
        self.reached_goal = False
        self._prev_goal_dist = self._dist((self.x, self.y), self.goal)

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        dv_sign, dw_sign = _ACTIONS[action]
        self.v     = float(np.clip(self.v + dv_sign * ACCEL * self.dt, V_MIN, V_MAX))
        self.omega = float(np.clip(self.omega + dw_sign * ALPHA * self.dt, -OMEGA_MAX, OMEGA_MAX))

        self.theta += self.omega * self.dt
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        self.x += self.v * math.cos(self.theta) * self.dt
        self.y += self.v * math.sin(self.theta) * self.dt

        self._step_obstacles()

        self._step_count += 1
        reward = self.step_penalty
        terminated = False

        goal_dist = self._dist((self.x, self.y), self.goal)
        reward += (self._prev_goal_dist - goal_dist) * 0.5  # progress shaping
        self._prev_goal_dist = goal_dist

        if self._hits_wall() or self._hits_obstacle():
            terminated = True
            self.collided = True
            reward = -1.0
        elif goal_dist < GOAL_RADIUS:
            terminated = True
            self.reached_goal = True
            reward = 1.0

        truncated = self._step_count >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> Optional[str]:
        if self.render_mode == "ansi":
            return self._render_ansi()
        return None

    def close(self):
        pass

    # ── Observation builder ────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        vals = [
            self.v / V_MAX,
            self.omega / OMEGA_MAX,
        ]
        gdist, gbearing = self._relative(self.goal)
        vals += [
            min(gdist / self._world_diag, 1.0),
            math.sin(gbearing),
            math.cos(gbearing),
        ]

        obstacles_sorted = sorted(
            self.obstacles, key=lambda o: self._dist((self.x, self.y), (o.x, o.y))
        )
        cos_t, sin_t = math.cos(self.theta), math.sin(self.theta)
        for obst in obstacles_sorted:
            odist, obearing = self._relative((obst.x, obst.y))
            # obstacle velocity relative to robot, expressed in robot frame
            rvx_world, rvy_world = obst.vx - self.v * cos_t, obst.vy - self.v * sin_t
            rvx_body =  rvx_world * cos_t + rvy_world * sin_t
            rvy_body = -rvx_world * sin_t + rvy_world * cos_t
            vals += [
                min(odist / self._world_diag, 1.0),
                math.sin(obearing),
                math.cos(obearing),
                np.clip(rvx_body / (V_MAX + self.obstacle_speed), -1.0, 1.0),
                np.clip(rvy_body / (V_MAX + self.obstacle_speed), -1.0, 1.0),
            ]

        return np.array(vals, dtype=np.float32)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _relative(self, point: tuple[float, float]) -> tuple[float, float]:
        """Return (distance, bearing) of `point` relative to robot pose."""
        dx, dy = point[0] - self.x, point[1] - self.y
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx) - self.theta
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        return dist, bearing

    def _step_obstacles(self) -> None:
        for o in self.obstacles:
            o.x += o.vx * self.dt
            o.y += o.vy * self.dt
            if o.x - o.radius < 0 or o.x + o.radius > self.world_size:
                o.vx *= -1
                o.x = float(np.clip(o.x, o.radius, self.world_size - o.radius))
            if o.y - o.radius < 0 or o.y + o.radius > self.world_size:
                o.vy *= -1
                o.y = float(np.clip(o.y, o.radius, self.world_size - o.radius))

    def _hits_wall(self) -> bool:
        return not (
            ROBOT_RADIUS <= self.x <= self.world_size - ROBOT_RADIUS
            and ROBOT_RADIUS <= self.y <= self.world_size - ROBOT_RADIUS
        )

    def _hits_obstacle(self) -> bool:
        for o in self.obstacles:
            if self._dist((self.x, self.y), (o.x, o.y)) < ROBOT_RADIUS + o.radius:
                return True
        return False

    def _sample_far_point(self, x: float, y: float, min_dist: float) -> tuple[float, float]:
        margin = self.world_size * 0.05
        for _ in range(50):
            px = float(self._np_rng.uniform(margin, self.world_size - margin))
            py = float(self._np_rng.uniform(margin, self.world_size - margin))
            if self._dist((x, y), (px, py)) >= min_dist:
                return (px, py)
        return (px, py)  # fallback: last sample

    def _sample_obstacle(self) -> Obstacle:
        margin = 0.6
        x = float(self._np_rng.uniform(margin, self.world_size - margin))
        y = float(self._np_rng.uniform(margin, self.world_size - margin))
        angle = float(self._np_rng.uniform(-math.pi, math.pi))
        speed = self.obstacle_speed * float(self._np_rng.uniform(0.5, 1.0))
        return Obstacle(
            x=x, y=y,
            vx=speed * math.cos(angle), vy=speed * math.sin(angle),
            radius=float(self._np_rng.uniform(0.25, 0.5)),
        )

    @staticmethod
    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _get_info(self) -> dict:
        return {
            "score": int(self.reached_goal),
            "steps": self._step_count,
            "collided": self.collided,
        }

    # ── ASCII renderer ─────────────────────────────────────────────────────────

    def _render_ansi(self, cols: int = 40, rows: int = 20) -> str:
        grid = [["." for _ in range(cols)] for _ in range(rows)]

        def to_cell(px, py):
            c = int(px / self.world_size * (cols - 1))
            r = int((1 - py / self.world_size) * (rows - 1))
            return max(0, min(rows - 1, r)), max(0, min(cols - 1, c))

        gr, gc = to_cell(*self.goal)
        grid[gr][gc] = "G"
        for o in self.obstacles:
            r, c = to_cell(o.x, o.y)
            grid[r][c] = "o"
        rr, rc = to_cell(self.x, self.y)
        grid[rr][rc] = "R"

        border = "+" + "-" * cols + "+"
        lines = [border] + ["|" + "".join(row) + "|" for row in grid] + [border]
        lines.append(f"step {self._step_count}  v={self.v:.2f}  omega={self.omega:.2f}")
        return "\n".join(lines)
