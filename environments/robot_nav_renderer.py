"""
Pygame renderer for RobotNavEnv.

Visual language matches PygameRenderer (Snake): black background, neon
green robot/UI, red goal-danger accents — but the star feature here is
drawing DWA's candidate trajectory fan (all 9 simulated arcs, colored by
score, best one highlighted) directly on top of the world. Watching the
fan sweep and collapse onto the safe path each step is the whole point
of having a visual lab.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from environments.robot_nav_env import ROBOT_RADIUS

try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False

BLACK      = (0,   0,   0)
GREEN_HI   = (0,   255, 65)
GREEN_MID  = (0,   180, 40)
GREEN_DIM  = (0,   100, 20)
GREEN_DARK = (0,   40,  10)
RED_HI     = (255, 60,  60)
YELLOW     = (255, 220, 60)
BLUE_OBST  = (60,  140, 255)
FONT_MONO  = "Courier New"


class RobotNavRenderer:

    BOARD_PAD = 20
    INFO_H    = 90

    def __init__(
        self,
        world_size: float = 12.0,
        px_per_meter: int = 45,
        fps: int = 20,
    ):
        if not _HAS_PYGAME:
            raise ImportError("pygame requerido: pip install pygame")

        self.world_size   = world_size
        self.px_per_meter = px_per_meter
        self.fps           = fps

        board_px      = int(world_size * px_per_meter)
        self.board_px = board_px
        total_w = board_px + self.BOARD_PAD * 2
        total_h = board_px + self.BOARD_PAD * 2 + self.INFO_H

        pygame.init()
        pygame.display.set_caption("Robot Navigation — DWA vs RL")
        self.screen  = pygame.display.set_mode((total_w, total_h))
        self.clock   = pygame.time.Clock()
        self.font_sm = pygame.font.SysFont(FONT_MONO, 12)
        self.font_lg = pygame.font.SysFont(FONT_MONO, 20, bold=True)

    # ── Coordinate transform (world meters → screen pixels) ─────────────────────

    def _to_px(self, x: float, y: float) -> tuple[int, int]:
        px = self.BOARD_PAD + int(x * self.px_per_meter)
        py = self.BOARD_PAD + self.board_px - int(y * self.px_per_meter)  # flip y
        return px, py

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        robot_pose: tuple[float, float, float],   # x, y, theta
        goal: tuple[float, float],
        obstacles: list,                            # objects with .x .y .radius
        episode: int = 0,
        step_count: int = 0,
        agent_name: str = "",
        score: int = 0,
        candidates: Optional[list[dict]] = None,     # DWAAgent.last_candidates
        best_idx: int = 0,
    ) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                return False

        self.screen.fill(BLACK)
        self._draw_border()
        self._draw_goal(goal)
        self._draw_obstacles(obstacles)
        if candidates:
            self._draw_candidates(robot_pose, candidates, best_idx)
        self._draw_robot(robot_pose)
        self._draw_info(episode, step_count, agent_name, score, robot_pose)

        pygame.display.flip()
        return True

    def tick(self) -> None:
        self.clock.tick(self.fps)

    def close(self) -> None:
        pygame.quit()

    # ── Drawing helpers ──────────────────────────────────────────────────────────

    def _draw_border(self) -> None:
        pygame.draw.rect(
            self.screen, GREEN_MID,
            pygame.Rect(self.BOARD_PAD - 1, self.BOARD_PAD - 1, self.board_px + 2, self.board_px + 2),
            1,
        )

    def _draw_goal(self, goal: tuple[float, float]) -> None:
        gx, gy = self._to_px(*goal)
        pygame.draw.circle(self.screen, GREEN_HI, (gx, gy), 10, 2)
        pygame.draw.circle(self.screen, GREEN_HI, (gx, gy), 3)

    def _draw_obstacles(self, obstacles: list) -> None:
        for o in obstacles:
            ox, oy = self._to_px(o.x, o.y)
            r_px = int(o.radius * self.px_per_meter)
            pygame.draw.circle(self.screen, BLUE_OBST, (ox, oy), r_px)
            # velocity direction tick
            end_x = ox + int(o.vx * 0.4 * self.px_per_meter)
            end_y = oy - int(o.vy * 0.4 * self.px_per_meter)
            pygame.draw.line(self.screen, (150, 200, 255), (ox, oy), (end_x, end_y), 1)

    def _draw_robot(self, pose: tuple[float, float, float]) -> None:
        x, y, theta = pose
        cx, cy = self._to_px(x, y)
        r_px = int(ROBOT_RADIUS * self.px_per_meter)

        pygame.draw.circle(self.screen, GREEN_HI, (cx, cy), r_px)
        pygame.draw.circle(self.screen, BLACK, (cx, cy), r_px, 1)

        # Heading tick: a short line from center to edge, in the direction faced.
        tip = (cx + r_px * math.cos(theta), cy - r_px * math.sin(theta))
        pygame.draw.line(self.screen, BLACK, (cx, cy), tip, 2)

    def _draw_candidates(
        self,
        robot_pose: tuple[float, float, float],
        candidates: list[dict],
        best_idx: int,
    ) -> None:
        rx, ry, rtheta = robot_pose
        cos_t, sin_t = math.cos(rtheta), math.sin(rtheta)

        scores = [c["score"] for c in candidates if c["score"] > -1e5]
        lo = min(scores) if scores else 0.0
        hi = max(scores) if scores else 1.0
        span = max(hi - lo, 1e-6)

        for c in candidates:
            pts_world = []
            for (lx, ly, _) in c["traj"]:
                # rotate local (robot-frame) point into world frame, offset by robot pose
                wx = rx + lx * cos_t - ly * sin_t
                wy = ry + lx * sin_t + ly * cos_t
                pts_world.append(self._to_px(wx, wy))

            is_best = c["action"] == best_idx
            if c["score"] <= -1e5:
                color = (90, 20, 20)      # collision-course candidate: dim red
            else:
                t = (c["score"] - lo) / span
                color = YELLOW if is_best else (0, int(60 + 120 * t), int(20 * t))
            width = 3 if is_best else 1
            if len(pts_world) > 1:
                pygame.draw.lines(self.screen, color, False, pts_world, width)

    def _draw_info(
        self,
        episode: int,
        step_count: int,
        agent_name: str,
        score: int,
        robot_pose: tuple[float, float, float],
    ) -> None:
        y0 = self.BOARD_PAD * 2 + self.board_px
        _, _, theta = robot_pose
        lines = [
            f"agent: {agent_name:<10}  ep: {episode:<5}  step: {step_count:<5}  reached: {score}",
            f"theta: {math.degrees(theta):>6.1f} deg",
        ]
        for i, line in enumerate(lines):
            txt = self.font_sm.render(line, True, GREEN_DIM)
            self.screen.blit(txt, (self.BOARD_PAD, y0 + 10 + i * 18))
