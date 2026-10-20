"""Interactive roundabout viewer with live sliders.

Run with ``python -m roundabout_sim``. Drag the sliders to change the traffic
while it runs; every agent is retuned immediately.

Keys: Space pause/resume, R restart with a new random seed, Esc quit.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass

import numpy as np
import pygame
from highway_env.envs.common.graphics import EnvViewer

from roundabout_sim.env import CRASH_COLOR, FAST_COLOR, MID_COLOR, SLOW_COLOR, RoundaboutTrafficEnv

VIEW_SIZE = 720  # [px] square simulation view
PANEL_WIDTH = 380  # [px] control panel on the right
VIEW_SCALING = 4.2  # [px/m]
FPS = 15  # frames per second; one frame = one simulation step at sim speed 1x
HISTORY_SECONDS = 300  # [s] span of the live charts

# Dark UI surface and text tokens.
SURFACE = (26, 26, 25)
SURFACE_RAISED = (40, 40, 38)
TEXT_PRIMARY = (240, 239, 236)
TEXT_SECONDARY = (195, 194, 183)
TEXT_MUTED = (140, 139, 132)
GRID = (60, 60, 57)
ACCENT = (57, 135, 229)  # series / slider fill


@dataclass
class Slider:
    label: str
    key: str  # env config key, or "sim_speed" for the app's own setting
    lo: float
    hi: float
    step: float
    value: float
    fmt: str
    hint: str = ""
    rect: pygame.Rect | None = None

    def set_from_x(self, x: int) -> None:
        t = np.clip((x - self.rect.x) / self.rect.w, 0.0, 1.0)
        raw = self.lo + t * (self.hi - self.lo)
        self.value = float(np.clip(round(raw / self.step) * self.step, self.lo, self.hi))

    @property
    def config_value(self):
        return int(round(self.value)) if self.step >= 1 else self.value


class RoundaboutApp:
    def __init__(self, seed: int | None = None):
        pygame.init()
        pygame.display.set_caption("Roundabout traffic model")
        self.screen = pygame.display.set_mode((VIEW_SIZE + PANEL_WIDTH, VIEW_SIZE))
        self.font = pygame.font.SysFont("segoeui,arial", 15)
        self.font_small = pygame.font.SysFont("segoeui,arial", 13)
        self.font_title = pygame.font.SysFont("segoeui,arial", 20, bold=True)
        self.clock = pygame.time.Clock()

        self.env = RoundaboutTrafficEnv()
        defaults = self.env.config
        self.sliders = [
            Slider("Cars in system", "vehicles_count", 2, 80, 1,
                   defaults["vehicles_count"], "{:.0f}", "density"),
            Slider("Mean desired speed", "mean_speed", 4, 20, 0.5,
                   defaults["mean_speed"], "{:.1f} m/s"),
            Slider("Speed std. deviation", "speed_std", 0, 6, 0.1,
                   defaults["speed_std"], "{:.1f} m/s", "driver variance"),
            Slider("Roundabout speed factor", "roundabout_speed_factor", 0.3, 1.0, 0.05,
                   defaults["roundabout_speed_factor"], "{:.2f}"),
            Slider("Simulation speed", "sim_speed", 1, 10, 1, 1, "{:.0f}x"),
        ]
        self.dragging: Slider | None = None
        self.paused = False
        self.seed = seed
        self.restart(seed)
        viewer_config = dict(self.env.config, screen_width=VIEW_SIZE, screen_height=VIEW_SIZE,
                             scaling=VIEW_SCALING, offscreen_rendering=True)
        self.viewer = EnvViewer(self.env, config=viewer_config)
        self.viewer.enabled = True  # highway-env disables drawing under SDL's dummy driver

    # ------------------------------------------------------------ control
    def restart(self, seed: int | None) -> None:
        self.seed = seed if seed is not None else int(np.random.default_rng().integers(1_000_000))
        self.apply_sliders()
        self.stats, _ = self.env.reset(seed=self.seed)
        self.history: deque[tuple[float, float, float]] = deque()

    def apply_sliders(self) -> None:
        self.env.configure({s.key: s.config_value for s in self.sliders if s.key != "sim_speed"})

    @property
    def sim_speed(self) -> int:
        return int(self.sliders[-1].value)

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if event.key == pygame.K_r:
                    self.restart(None)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for s in self.sliders:
                    if s.rect and s.rect.inflate(16, 24).collidepoint(event.pos):
                        self.dragging = s
                        s.set_from_x(event.pos[0])
                        self.apply_sliders()
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging = None
            if event.type == pygame.MOUSEMOTION and self.dragging:
                self.dragging.set_from_x(event.pos[0])
                self.apply_sliders()
        return True

    def run(self) -> None:
        running = True
        while running:
            running = self.handle_events()
            if not self.paused:
                for _ in range(self.sim_speed):
                    self.stats, *_ = self.env.step()
                    self.record()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()

    def record(self) -> None:
        t = self.stats["time"]
        if self.history and t - self.history[-1][0] < 1.0:
            return  # sample once per simulated second
        self.history.append((t, self.stats["throughput_per_min"], self.stats["mean_speed"]))
        while self.history and self.history[0][0] < t - HISTORY_SECONDS:
            self.history.popleft()

    # ------------------------------------------------------------ drawing
    def draw(self) -> None:
        self.viewer.display()
        self.screen.blit(self.viewer.sim_surface, (0, 0))
        if self.paused:
            self.text("PAUSED  (space to resume)", (16, 12), self.font, TEXT_PRIMARY)
        self.draw_panel()
        pygame.display.flip()

    def text(self, s, pos, font=None, color=TEXT_PRIMARY, align="left") -> None:
        surf = (font or self.font).render(s, True, color)
        x, y = pos
        if align == "right":
            x -= surf.get_width()
        self.screen.blit(surf, (x, y))

    def draw_panel(self) -> None:
        x0, pad = VIEW_SIZE, 20
        left, right = x0 + pad, x0 + PANEL_WIDTH - pad
        pygame.draw.rect(self.screen, SURFACE, (x0, 0, PANEL_WIDTH, VIEW_SIZE))
        self.text("Roundabout traffic", (left, 14), self.font_title)
        self.text(f"seed {self.seed}   ·   R restart   ·   Space pause", (left, 42),
                  self.font_small, TEXT_MUTED)

        y = 74
        for s in self.sliders:
            label = s.label + (f"  ({s.hint})" if s.hint else "")
            self.text(label, (left, y), self.font, TEXT_SECONDARY)
            self.text(s.fmt.format(s.value), (right, y), self.font, TEXT_PRIMARY, "right")
            s.rect = pygame.Rect(left, y + 26, right - left, 6)
            t = (s.value - s.lo) / (s.hi - s.lo)
            pygame.draw.rect(self.screen, GRID, s.rect, border_radius=3)
            filled = s.rect.copy()
            filled.w = max(int(s.rect.w * t), 1)
            pygame.draw.rect(self.screen, ACCENT, filled, border_radius=3)
            pygame.draw.circle(self.screen, TEXT_PRIMARY, (s.rect.x + int(s.rect.w * t), s.rect.centery), 8)
            y += 52

        st = self.stats
        rows = [
            ("Simulated time", f"{st['time']:.0f} s"),
            ("Cars in system", f"{st['vehicles']}"),
            ("Ring density", f"{st['ring_density']:.0f} veh/km  ({st['ring_vehicles']} cars)"),
            ("Queued at entries", f"{st['queued']}"),
            ("Mean speed", f"{st['mean_speed']:.1f} m/s"),
            ("Throughput", f"{st['throughput_per_min']:.1f} cars/min"),
            ("Mean trip time", f"{st['mean_travel_time']:.0f} s"),
            ("Crashes", f"{st['crashes']}"),
        ]
        y += 4
        for name, value in rows:
            self.text(name, (left, y), self.font_small, TEXT_SECONDARY)
            self.text(value, (right, y), self.font_small, TEXT_PRIMARY, "right")
            y += 19

        y += 10
        h = 62
        span = f"last {HISTORY_SECONDS // 60} min"
        self.sparkline(f"Throughput (cars/min), {span}", 1, pygame.Rect(left, y, right - left, h))
        self.sparkline(f"Mean speed (m/s), {span}", 2, pygame.Rect(left, y + h + 26, right - left, h))
        self.legend(VIEW_SIZE - 32)

    def sparkline(self, title: str, column: int, rect: pygame.Rect) -> None:
        self.text(title, (rect.x, rect.y - 2), self.font_small, TEXT_SECONDARY)
        plot = pygame.Rect(rect.x, rect.y + 18, rect.w, rect.h - 18)
        pygame.draw.rect(self.screen, SURFACE_RAISED, plot)
        if len(self.history) < 2:
            return
        data = np.array(self.history)
        t, v = data[:, 0], data[:, column]
        top = max(float(v.max()) * 1.15, 1.0)
        t_end = t[-1]
        xs = plot.right - (t_end - t) / HISTORY_SECONDS * plot.w
        ys = plot.bottom - v / top * plot.h
        pygame.draw.line(self.screen, GRID, (plot.x, plot.bottom - 1), (plot.right, plot.bottom - 1))
        pygame.draw.lines(self.screen, ACCENT, False, list(zip(xs, ys)), 2)
        pygame.draw.circle(self.screen, ACCENT, (int(xs[-1]), int(ys[-1])), 4)
        self.text(f"{v[-1]:.1f}", (plot.right, rect.y - 2), self.font_small, TEXT_PRIMARY, "right")
        self.text(f"max {top / 1.15:.0f}", (plot.x + 4, plot.y + 2), self.font_small, TEXT_MUTED)

    def legend(self, y: int) -> None:
        x = VIEW_SIZE + 20
        self.text("Car colour = driver's desired speed:", (x, y - 20), self.font_small, TEXT_MUTED)
        for color, label in ((SLOW_COLOR, "slow"), (MID_COLOR, "average"),
                             (FAST_COLOR, "fast"), (CRASH_COLOR, "crashed")):
            pygame.draw.rect(self.screen, tuple(int(c) for c in color), (x, y + 3, 16, 9), border_radius=2)
            self.text(label, (x + 22, y - 1), self.font_small, TEXT_SECONDARY)
            x += 88


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: random)")
    args = parser.parse_args()
    RoundaboutApp(seed=args.seed).run()


if __name__ == "__main__":
    main()
