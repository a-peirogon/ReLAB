from __future__ import annotations
from typing import Optional
import numpy as np
try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False
BLACK = (0, 0, 0)
GREEN_HI = (0, 255, 65)
GREEN_MID = (0, 180, 40)
GREEN_DIM = (0, 100, 20)
GREEN_DARK = (0, 40, 10)
RED_HI = (200, 30, 30)
RED_MID = (255, 60, 60)
FONT_MONO = 'Courier New'

class PygameRenderer:
    NN_W = 300
    BOARD_PAD = 12
    MAX_VISIBLE = 16

    def __init__(self, grid_size: int=18, cell_size: int=20, nn_layers: list[int]=None, fps: int=10):
        if not _HAS_PYGAME:
            raise ImportError('pygame requerido: pip install pygame')
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.nn_layers = nn_layers or [11, 16, 16, 4]
        self.fps = fps
        board_px = grid_size * cell_size
        self.BOARD_W = board_px + self.BOARD_PAD * 2
        total_w = self.NN_W + self.BOARD_W
        total_h = max(520, board_px + self.BOARD_PAD * 2 + 60)
        self._total_h = total_h
        pygame.init()
        pygame.display.set_caption('Snake')
        self.screen = pygame.display.set_mode((total_w, total_h))
        self.clock = pygame.time.Clock()
        self.font_sm = pygame.font.SysFont(FONT_MONO, 11)
        self.font_md = pygame.font.SysFont(FONT_MONO, 14)
        self.font_lg = pygame.font.SysFont(FONT_MONO, 22, bold=True)
        self._vis_layers = self._compute_visible_layers()
        self._node_positions = self._compute_node_positions(total_h)
        self.surf_nn = pygame.Surface((self.NN_W, total_h))
        self.surf_board = pygame.Surface((self.BOARD_W, total_h))
        self._output_labels = ['Up', 'Down', 'Left', 'Right']

    def render(self, snake: list[tuple[int, int]], food: tuple[int, int], obs: np.ndarray, score: int, activations: Optional[list[np.ndarray]]=None, episode: int=0, step_count: int=0, epsilon: float=0.0) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                return False
        self._draw_nn_panel(obs, activations)
        self._draw_board(snake, food, score, episode, step_count, epsilon)
        self.screen.blit(self.surf_nn, (0, 0))
        self.screen.blit(self.surf_board, (self.NN_W, 0))
        pygame.display.flip()
        return True

    def tick(self) -> None:
        self.clock.tick(self.fps)

    def close(self) -> None:
        pygame.quit()

    def _compute_visible_layers(self) -> list[int]:
        n = len(self.nn_layers)
        return [size if li == 0 or li == n - 1 else min(size, self.MAX_VISIBLE) for li, size in enumerate(self.nn_layers)]

    def _sample_acts(self, acts: list[np.ndarray]) -> list[np.ndarray]:
        sampled = []
        for li, a in enumerate(acts):
            n_vis = self._vis_layers[li] if li < len(self._vis_layers) else len(a)
            if len(a) <= n_vis:
                sampled.append(a)
            else:
                idx = np.linspace(0, len(a) - 1, n_vis, dtype=int)
                sampled.append(a[idx])
        return sampled

    def _compute_node_positions(self, total_h: int) -> list[list[tuple[int, int]]]:
        pad_t, pad_b = (30, 30)
        usable = total_h - pad_t - pad_b
        n_layers = len(self._vis_layers)
        positions = []
        for li, n_nodes in enumerate(self._vis_layers):
            x = int(20 + li * (self.NN_W - 40) / max(n_layers - 1, 1))
            layer_pos = []
            for ni in range(n_nodes):
                y = total_h // 2 if n_nodes == 1 else int(pad_t + ni * usable / (n_nodes - 1))
                layer_pos.append((x, y))
            positions.append(layer_pos)
        return positions

    def _draw_nn_panel(self, obs: np.ndarray, activations: Optional[list[np.ndarray]]) -> None:
        s = self.surf_nn
        s.fill(BLACK)
        if activations is None:
            acts_raw = [obs] + [np.zeros(self.nn_layers[li + 1]) for li in range(len(self.nn_layers) - 1)]
        else:
            acts_raw = activations
        while len(acts_raw) < len(self._vis_layers):
            acts_raw.append(np.zeros(self._vis_layers[len(acts_raw)]))
        acts = self._sample_acts(acts_raw)
        n_layers = len(self._vis_layers)
        for li in range(n_layers - 1):
            src_pos = self._node_positions[li]
            dst_pos = self._node_positions[li + 1]
            src_acts = acts[li] if li < len(acts) else []
            for si, (sx, sy) in enumerate(src_pos):
                a = float(abs(src_acts[si])) if si < len(src_acts) else 0.0
                brightness = min(255, int(30 + a * 130))
                line_col = (0, brightness, int(brightness * 0.25))
                for dx, dy in dst_pos:
                    pygame.draw.line(s, line_col, (sx, sy), (dx, dy), 1)
        n_out = n_layers - 1
        best_out = int(np.argmax(acts[-1])) if len(acts[-1]) == 4 else -1
        for li, layer_pos in enumerate(self._node_positions):
            layer_acts = acts[li] if li < len(acts) else []
            is_output = li == n_out
            max_a = float(np.max(np.abs(layer_acts))) + 1e-08
            for ni, (x, y) in enumerate(layer_pos):
                a = float(abs(layer_acts[ni])) / max_a if ni < len(layer_acts) else 0.0
                radius = 7 if is_output else 5
                if a > 0.15:
                    fill = min(255, int(60 + a * 195))
                    pygame.draw.circle(s, (0, fill, int(fill * 0.25)), (x, y), radius)
                highlight = is_output and ni == best_out
                border = GREEN_HI if a > 0.5 or highlight else GREEN_DIM
                pygame.draw.circle(s, border, (x, y), radius, 1)
                if is_output and ni < len(self._output_labels):
                    prefix = '>' if highlight else ' '
                    col = GREEN_HI if highlight else GREEN_DIM
                    s.blit(self.font_sm.render(f'{prefix}{self._output_labels[ni]}', True, col), (x + radius + 4, y - 6))
        for li in range(1, n_layers - 1):
            real, vis = (self.nn_layers[li], self._vis_layers[li])
            if real > vis:
                x = self._node_positions[li][0][0]
                lbl = self.font_sm.render(f'{real}n', True, GREEN_DIM)
                s.blit(lbl, (x - lbl.get_width() // 2, self._total_h - 22))
        pygame.draw.line(s, GREEN_DIM, (self.NN_W - 1, 0), (self.NN_W - 1, self._total_h), 1)

    def _draw_board(self, snake: list[tuple[int, int]], food: tuple[int, int], score: int, episode: int, step_count: int, epsilon: float) -> None:
        s = self.surf_board
        cs = self.cell_size
        ox = self.BOARD_PAD
        oy = self.BOARD_PAD + 40
        s.fill(BLACK)
        stxt = self.font_lg.render(f'SCORE:{score:03d}', True, GREEN_HI)
        s.blit(stxt, (self.BOARD_W // 2 - stxt.get_width() // 2, 8))
        pygame.draw.rect(s, GREEN_MID, pygame.Rect(ox - 1, oy - 1, self.grid_size * cs + 2, self.grid_size * cs + 2), 1)
        for i in range(self.grid_size + 1):
            pygame.draw.line(s, GREEN_DARK, (ox + i * cs, oy), (ox + i * cs, oy + self.grid_size * cs), 1)
            pygame.draw.line(s, GREEN_DARK, (ox, oy + i * cs), (ox + self.grid_size * cs, oy + i * cs), 1)
        fr, fc = food
        pygame.draw.rect(s, RED_HI, pygame.Rect(ox + fc * cs + 1, oy + fr * cs + 1, cs - 2, cs - 2))
        pygame.draw.rect(s, RED_MID, pygame.Rect(ox + fc * cs + 3, oy + fr * cs + 3, 4, 4))
        for i, (r, c) in enumerate(snake):
            fade = max(0.25, 1.0 - i / max(len(snake), 1) * 0.6)
            col = GREEN_HI if i == 0 else (0, int(160 * fade + 20), int(20 * fade))
            rect = pygame.Rect(ox + c * cs + 1, oy + r * cs + 1, cs - 2, cs - 2)
            pygame.draw.rect(s, col, rect)
            if i > 0:
                pygame.draw.rect(s, BLACK, rect, 1)
        info = self.font_sm.render(f'ep:{episode}  step:{step_count}  ε:{epsilon:.3f}', True, GREEN_DIM)
        s.blit(info, (ox, oy + self.grid_size * cs + 6))
