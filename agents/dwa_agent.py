from __future__ import annotations
import math
from typing import Optional
import numpy as np
from agents.base_agent import BaseAgent
from environments.robot_nav_env import ACCEL, ALPHA, OMEGA_MAX, ROBOT_RADIUS, V_MAX, _ACTIONS
ASSUMED_OBSTACLE_RADIUS = 0.4

class DWAAgent(BaseAgent):

    def __init__(self, obs_dim: Optional[int]=None, n_actions: int=9, n_obstacles: int=8, dt: float=0.15, horizon_steps: int=8, world_size: float=12.0, obstacle_speed: float=0.8, weights: tuple[float, float, float]=(1.2, 1.0, 0.6), **_ignored):
        obs_dim = obs_dim or 5 + 5 * n_obstacles
        super().__init__(obs_dim, n_actions)
        self.n_obstacles = n_obstacles
        self.dt = dt
        self.horizon_steps = horizon_steps
        self.w_heading, self.w_clearance, self.w_speed = weights
        self._world_diag = math.hypot(world_size, world_size)
        self._vel_scale = V_MAX + obstacle_speed
        self.last_candidates: list[dict] = []
        self.last_best_idx: int = 0

    def select_action(self, obs: np.ndarray, explore: bool=True) -> int:
        v0 = float(obs[0]) * V_MAX
        omega0 = float(obs[1]) * OMEGA_MAX
        goal_local = self._decode_goal(obs)
        obstacles_local = self._decode_obstacles(obs)
        candidates = []
        best_idx, best_score = (0, -math.inf)
        for idx, (dv_sign, dw_sign) in enumerate(_ACTIONS):
            v = float(np.clip(v0 + dv_sign * ACCEL * self.dt, 0.0, V_MAX))
            omega = float(np.clip(omega0 + dw_sign * ALPHA * self.dt, -OMEGA_MAX, OMEGA_MAX))
            traj = self._rollout(v, omega)
            score = self._score_trajectory(traj, v, goal_local, obstacles_local)
            candidates.append({'action': idx, 'v': v, 'omega': omega, 'traj': traj, 'score': score})
            if score > best_score:
                best_score, best_idx = (score, idx)
        self.last_candidates = candidates
        self.last_best_idx = best_idx
        return best_idx

    def update(self, obs, action, reward, next_obs, done) -> Optional[float]:
        return None

    def on_episode_end(self) -> None:
        pass

    def _decode_goal(self, obs: np.ndarray) -> tuple[float, float]:
        gdist = float(obs[2]) * self._world_diag
        gsin, gcos = (float(obs[3]), float(obs[4]))
        return (gdist * gcos, gdist * gsin)

    def _decode_obstacles(self, obs: np.ndarray) -> list[dict]:
        obstacles = []
        base = 5
        for i in range(self.n_obstacles):
            o = obs[base + i * 5:base + i * 5 + 5]
            odist_n, osin, ocos, rvx_n, rvy_n = [float(v) for v in o]
            odist = odist_n * self._world_diag
            obstacles.append({'x': odist * ocos, 'y': odist * osin, 'vx': rvx_n * self._vel_scale, 'vy': rvy_n * self._vel_scale})
        return obstacles

    def _rollout(self, v: float, omega: float) -> list[tuple[float, float, float]]:
        x, y, theta = (0.0, 0.0, 0.0)
        pts = [(x, y, theta)]
        for _ in range(self.horizon_steps):
            theta += omega * self.dt
            x += v * math.cos(theta) * self.dt
            y += v * math.sin(theta) * self.dt
            pts.append((x, y, theta))
        return pts

    def _score_trajectory(self, traj: list[tuple[float, float, float]], v: float, goal_local: tuple[float, float], obstacles_local: list[dict]) -> float:
        end_x, end_y, end_theta = traj[-1]
        goal_bearing = math.atan2(goal_local[1], goal_local[0])
        heading_err = abs(math.atan2(math.sin(goal_bearing - end_theta), math.cos(goal_bearing - end_theta)))
        heading_score = 1.0 - heading_err / math.pi
        min_clearance = math.inf
        for t_idx, (px, py, _) in enumerate(traj):
            t = t_idx * self.dt
            for o in obstacles_local:
                ox, oy = (o['x'] + o['vx'] * t, o['y'] + o['vy'] * t)
                d = math.hypot(px - ox, py - oy) - (ROBOT_RADIUS + ASSUMED_OBSTACLE_RADIUS)
                min_clearance = min(min_clearance, d)
        if min_clearance is math.inf:
            min_clearance = 5.0
        if min_clearance < 0:
            return -1000000.0
        clearance_score = min(min_clearance, 2.0) / 2.0
        speed_score = v / V_MAX
        return self.w_heading * heading_score + self.w_clearance * clearance_score + self.w_speed * speed_score
