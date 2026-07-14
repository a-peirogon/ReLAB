"""
Pygame renderer for GridWorldEnv.

The whole point of this environment is watching the Q-table fill in.
Each cell is colored by its max Q-value (dark = unvisited/zero, red =
negative, green = positive, brighter = higher magnitude) and, once a
cell has been visited, a small arrow shows the current greedy action —
so you can watch the policy arrows reorganize themselves, episode by
episode, as value backs up from the goal.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False

BLACK      = (0,   0,   0)
GRID_LINE  = (30,  30,  30)
WALL       = (70,  70,  70)
GOAL_COLOR = (0,   255, 65)
PIT_COLOR  = (200, 30,  30)
AGENT_COLOR = (255, 220, 60)
TEXT_COLOR  = (0,   200, 60)

# arrow deltas matching GridWorldEnv._ACTIONS order: up, down, left, right
_ARROW_DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]


class GridWorldRenderer:

    INFO_H = 70

    def __init__(self, n_rows: int, n_cols: int, cell_size: int = 48, fps: int = 20):
        if not _HAS_PYGAME:
            raise ImportError("pygame requerido: pip install pygame")

        self.n_rows, self.n_cols = n_rows, n_cols
        self.cell_size = cell_size
        self.fps = fps

        w = n_cols * cell_size
        h = n_rows * cell_size + self.INFO_H

        pygame.init()
        pygame.display.set_caption("GridWorld — Q-table live")
        self.screen  = pygame.display.set_mode((w, h))
        self.clock   = pygame.time.Clock()
        self.font_sm = pygame.font.SysFont("Courier New", 13)
        self.font_lg = pygame.font.SysFont("Courier New", 18, bold=True)

    def _cell_rect(self, r: int, c: int) -> "pygame.Rect":
        return pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        agent_pos: tuple[int, int],
        goal: tuple[int, int],
        walls: set,
        pits: set,
        q_table: dict,
        n_actions: int = 4,
        episode: int = 0,
        step_count: int = 0,
        epsilon: float = 0.0,
        episodes_solved: int = 0,
    ) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                return False

        self.screen.fill(BLACK)

        # ── Q-value heatmap + greedy-action arrows ──
        q_values_seen = [np.max(q) for s, q in q_table.items() if s not in walls]
        vmax = max([abs(v) for v in q_values_seen] + [1e-6])

        for r in range(self.n_rows):
            for c in range(self.n_cols):
                rect = self._cell_rect(r, c)
                if (r, c) in walls:
                    pygame.draw.rect(self.screen, WALL, rect)
                elif (r, c) in pits:
                    pygame.draw.rect(self.screen, PIT_COLOR, rect)
                elif (r, c) == goal:
                    pygame.draw.rect(self.screen, (0, 60, 20), rect)
                    pygame.draw.rect(self.screen, GOAL_COLOR, rect.inflate(-10, -10), 2)
                else:
                    q = q_table.get((r, c), None)
                    if q is not None and np.any(q != 0):
                        best_a = int(np.argmax(q))
                        val = float(q[best_a])
                        t = min(abs(val) / vmax, 1.0)
                        color = (int(60 * t), int(150 * t), int(30 * t)) if val >= 0 else (int(150 * t), int(20 * t), int(20 * t))
                        pygame.draw.rect(self.screen, color, rect)
                        self._draw_arrow(rect, best_a)
                    else:
                        pygame.draw.rect(self.screen, (12, 12, 12), rect)

                pygame.draw.rect(self.screen, GRID_LINE, rect, 1)

        # ── Agent ──
        ar, ac = agent_pos
        cx = ac * self.cell_size + self.cell_size // 2
        cy = ar * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, AGENT_COLOR, (cx, cy), self.cell_size // 3)

        self._draw_info(episode, step_count, epsilon, episodes_solved, len(q_table))

        pygame.display.flip()
        return True

    def tick(self) -> None:
        self.clock.tick(self.fps)

    def close(self) -> None:
        pygame.quit()

    # ── Drawing helpers ──────────────────────────────────────────────────────────

    def _draw_arrow(self, rect: "pygame.Rect", action: int) -> None:
        dx, dy = _ARROW_DIRS[action]
        cx, cy = rect.center
        size = self.cell_size * 0.28
        tip   = (cx + dx * size,        cy + dy * size)
        left  = (cx - dy * size * 0.35 - dx * size * 0.4, cy + dx * size * 0.35 - dy * size * 0.4)
        right = (cx + dy * size * 0.35 - dx * size * 0.4, cy - dx * size * 0.35 - dy * size * 0.4)
        pygame.draw.polygon(self.screen, BLACK, [tip, left, right])

    def _draw_info(
        self, episode: int, step_count: int, epsilon: float, episodes_solved: int, table_size: int,
    ) -> None:
        y0 = self.n_rows * self.cell_size
        pygame.draw.line(self.screen, GRID_LINE, (0, y0), (self.n_cols * self.cell_size, y0), 1)
        line1 = f"episode {episode:<6} step {step_count:<4} epsilon {epsilon:.3f}"
        line2 = f"solved: {episodes_solved:<6} states visited: {table_size}"
        self.screen.blit(self.font_sm.render(line1, True, TEXT_COLOR), (10, y0 + 10))
        self.screen.blit(self.font_sm.render(line2, True, TEXT_COLOR), (10, y0 + 32))
