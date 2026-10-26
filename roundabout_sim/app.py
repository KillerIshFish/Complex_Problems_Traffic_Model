"""Interactive viewer for the single-lane closed-ring traffic model.

Run ``python -m roundabout_sim`` (``--help`` lists the model settings you can
change, e.g. ``--density 80 --driver-noise 0.5``). The left view draws the ring
at true scale; the panel on the right changes parameters while the model
runs; the live charts (the last 10 minutes) track speed, congestion over
time, stop-and-go, flux and how the cars congregate, and the space-time
diagram shows phantom jams as stripes running backwards against the traffic.

Mouse: click a car to make it brake, drag sliders, hover anything (charts,
readouts, controls) for help, scroll the parameter list with the wheel.
Keys: Space pause, R reset (new seed), B brake a random car, +/- sim speed,
C cycle car colours, S save a screenshot to results/, D save the chart data
(CSV, plus the settings, the live changes and an equivalent experiments
command), Esc quit.
"""

# Viewer setup: n_lanes=1, n_arms=0, car_width=1.8 m, dt=0.1 s. Hidden
# randomized settings: car_length 1-8 m; exit_prob 0-1, entry_gap 1.5-6 s,
# turn_speed 2-15 m/s; lane_change_rate 0-1 /s, lane_change_motive
# advantage/random, politeness 0-1; reaction_time 0-2 s, time_headway 0.5-3 s,
# min_gap 0.5-5 m, lateral_accel 1-10 m/s2. Entrance/exit settings have no
# effect on the closed ring.
# Other settings are randomized once per launch and stay fixed on reset;
# max_accel=1.2 m/s2, comfort_decel=2 m/s2 and reaction_distance=25 m are fixed.

from __future__ import annotations

import argparse
import ctypes
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import pygame

from roundabout_sim import metrics, palette
from roundabout_sim.config import PARAM_BY_NAME, PARAMS, ParamSpec, SimConfig, coerce
from roundabout_sim.engine import MAX_DECEL, MAX_QUEUE, RESUME_FRACTION, SLOW_FRACTION, TWO_PI, RingSim

FPS_CAP = 60
SAMPLE_INTERVAL = 0.5  # [s] of simulated time between chart samples
SPACETIME_REFRESH = 0.1  # [s] of wall time between rebuilds of the space-time image
HISTORY = 600.0  # [s] of chart history kept (10 minutes)
MIN_SPAN = 60.0  # [s] shortest time axis on the charts
MIN_WINDOW = (1100, 700)
RATE_WARMUP = 10.0  # [s] rolling rates (over metrics.RATE_WINDOW) are too noisy to chart before this
ROLLING = ("detector_flux", "stops_per_car_min")  # chart series that are rolling rates
SPACETIME_SHARE = 0.34  # of the chart column's height
FONT_FACE = "segoeui,helveticaneue,arial"
WINDOW = f"{metrics.RATE_WINDOW:g} s"
UI = palette.UI
EMPTY = UI["surface"]  # space-time bins without cars
HIDDEN_PARAMS = frozenset({
    "n_lanes", "n_arms", "car_length", "car_width", "exit_prob", "entry_gap", "turn_speed",
    "lane_change_rate", "lane_change_motive", "politeness", "reaction_time", "time_headway", "min_gap",
    "lateral_accel", "max_accel", "comfort_decel", "reaction_distance",
})
FIXED_PARAMS = {"max_accel": 1.2, "comfort_decel": 2.0, "reaction_distance": 25.0}
CONTROL_PARAMS = tuple(
    replace(p, hi={"radius": 60, "speed_std": 2}.get(p.name, p.hi))
    for p in PARAMS if p.name not in HIDDEN_PARAMS
)

# Viewer settings use the same spec type as model parameters, so they get the
# same widgets; group "View" keeps them out of the model's tabs.
VIEW_PARAMS = {p.name: p for p in (
    ParamSpec("steps_per_frame", "Sim speed, steps per frame", 1, 20, 1, "", "View",
              "Simulation steps (of dt each) per drawn frame. Keys + and -."),
    ParamSpec("colour_mode", "Car colour", 0, 0, 0, "", "View",
              "speed: own speed / free speed; desired: driver's desired speed (slow to fast); "
              "lane: which lane. Key C.", choices=("speed", "desired", "lane")),
    ParamSpec("spacetime_mode", "Space-time colour", 0, 0, 0, "", "View",
              "speed: mean speed at each angle / the cars' mean free speed; occupancy: share of lane "
              f"covered in each of the {metrics.SECTORS} sectors.", choices=("speed", "occupancy")),
)}

HISTORY_TEXT = f"the last {HISTORY / 60:g} min"
BUTTONS = (
    ("pause", "Pause", "Pause or resume the simulation (Space)."),
    ("step", "Step", "Advance one frame of simulation, also while paused."),
    ("reset", "Reset", "Restart with a new random seed (R)."),
    ("brake", "Brake car", "Make a random driver brake hard for 3 s to seed a jam (B). Clicking a car does the same."),
    ("clear", "Clear charts", "Forget the chart history and start recording afresh."),
    ("screenshot", "Save PNG", "Save a screenshot into the results folder (S)."),
    ("csv", "Save CSV", f"Save the chart data ({HISTORY_TEXT}, a row every {SAMPLE_INTERVAL:g} s) as CSV into the "
                        "results folder, with a settings file listing the settings, the live parameter changes "
                        "and the equivalent experiments command (D)."),
)
BUTTON_ROWS = (("pause", "step", "reset", "brake"), ("clear", "screenshot", "csv"))

DEFAULT_HINT = ("Click a car to make it brake.  Space pause  ·  R reset  ·  B brake  ·  "
                "+/- speed  ·  C colours  ·  S screenshot  ·  D data (CSV)  ·  Esc quit")

def control_help(spec: ParamSpec) -> str:
    """Help shown for a slider or choice."""
    note = " Changing it rebuilds the roundabout and clears the charts." if spec.reset else ""
    return f"{spec.label}: {spec.help}{note}"


@dataclass(frozen=True)
class Series:
    key: str
    label: str
    scale: float = 1.0  # display factor (100 for percentages)
    reference: bool = False  # faint dashed reference for the series before it (no palette slot)


@dataclass(frozen=True)
class ChartSpec:
    title: str
    series: tuple[Series, ...]
    fmt: str
    min_top: float  # smallest y-axis maximum, so a flat line is not blown up
    help: str
    extra: str | None = None  # extra readout in the header: "jams" or "bunching"
    max_top: float | None = None  # largest y-axis maximum while the data stay below it (100 %)


ONE_SIDED = 2.0  # order counts as one-sided above this multiple of the random baseline (~4 % by chance)
CLUSTERING_HELP = (f"Spread of the car counts over 6-{metrics.SECTORS} sectors (about 4+ cars each) beyond random "
                   "scatter, relative to the mean (0 = even or random, about 1 = half the ring empty). Often 0 "
                   "in dense flow.")
ORDER_HELP = ("Order: 0 = cars balanced around the ring, 1 = all on one side. Randomly placed cars give about "
              f"√(π/4N) (the dashed 'random' line), so only order well above it (over {ONE_SIDED:g}×) is real "
              "one-sidedness; the arrow in the view then points to where the cars bunch.")
SPACING_HELP = ("Gini coefficient of the road each car has (gap ahead + its length): 0 = shared evenly, about "
                "0.35 = random gaps, towards 1 = packed queues or platoons with empty road between. Shows "
                "uneven starts and platoons; not a jam detector.")
STOPS_HELP = (f"Stop-and-go, a measure of instability: stops per car per minute over the last {WINDOW} (a car "
              f"dropping below {SLOW_FRACTION:.0%} of its free speed after being above {RESUME_FRACTION:.0%}). "
              "About 0 in smooth flow and also in a standstill, where nobody moves to be stopped; 1–2 in "
              "stop-and-go waves. Read it together with Congestion.")
JAMS_HELP = ("Jams (queues of 2+ jammed cars in a lane) by what the front car is doing now: following the car "
             "ahead, merging out to its exit, slowing to exit, just entered, or other (braked by you or slotted "
             "in). A wave set off at an exit travels back with ordinary followers at its front, so following "
             "is only an upper bound on natural (phantom) jams; to see what the arms cause, compare with "
             "Exit chance = 0.")
EXTRA_HELP = {"jams": JAMS_HELP, "bunching": ORDER_HELP}  # hover help of the header readouts

CHARTS = (
    ChartSpec("Mean speed [m/s]", (Series("mean_speed", "actual"), Series("mean_free_speed", "desired")),
              "{:.1f}", 2.0,
              "Mean speed of the cars on the ring, and the mean speed they want to go (their free speed, "
              "capped by cornering on inner lanes). Jams open a gap between the two."),
    ChartSpec("Congestion over time [%]", (Series("jam_fraction", "jammed cars", 100.0),
                                           Series("speed_deficit", "speed deficit", 100.0)), "{:.0f}", 10.0,
              f"Jammed cars, the headline congestion measure: share of cars crawling below {SLOW_FRACTION:.0%} "
              "of their free speed, in queues and stop-and-go waves alike; it rises steadily with density. "
              "Speed deficit: 1 − total speed / total free speed (0 = free flow, 100 = standstill).",
              max_top=100.0),
    ChartSpec("Stop-and-go (instability) [stops/car/min]", (Series("stops_per_car_min", "stop-and-go"),),
              "{:.2f}", 1.0, f"{STOPS_HELP} Hover the jams readout for what holds each jam up.", extra="jams"),
    ChartSpec("Flux [veh/s]", (Series("flux", "whole ring"), Series("detector_flux", f"detector, {WINDOW}")),
              "{:.2f}", 0.5,
              "Cars per second passing a cross-section of the ring (all lanes). Whole ring: from every "
              f"car's speed (sum of v / lane length). Detector: cars crossing the line, over {WINDOW}."),
    ChartSpec("Congregation [0–1]", (Series("spacing_gini", "spacing (Gini)"), Series("order_param", "order"),
                                     Series("order_baseline", "random", reference=True)), "{:.2f}", 0.5,
              "Do the cars congregate? Spacing (Gini): 0 = even, about 0.35 = random gaps, towards 1 = packed "
              "queues or platoons. Order: 0 = balanced around the ring, 1 = all on one side; dashed = what "
              "randomly placed cars give, so only order well above it is real one-sidedness (the arrow in "
              "the view then points there).", extra="bunching"),
)
SPACETIME_HELP = ("Space-time diagram: time runs right, angle around the ring up (driving direction). "
                  "Cars trace rising lines; jams are stripes sloping down as they travel backwards. "
                  "Speed: mean speed / the cars' mean free speed. Ticks mark the arms; red rings mark "
                  "braked cars.")
VIEW_HELP = ("Click a car to make it brake hard for 3 s. The inner band shades each sector by how much "
             "lane it has covered; the arrow points to where the cars bunch when they are clearly more "
             "one-sided than random (longer = more one-sided).")
STATUS_LINES = (6, 10)  # fewest / most lines of the help box: it grows to fit the longest help text

STAT_HELP = {
    "Sim time": "Simulated time since the last reset or rebuild.",
    "Cars on ring": "Cars on the ring now / the number the density setting asks for.",
    "Queued to enter": f"Cars waiting at the entrances (at most {MAX_QUEUE} each; further replacements for "
                       "leavers are slotted straight into the ring).",
    "Density": "Cars on the ring per km of lane.",
    "Area covered": "Share of the road surface (all lanes) covered by cars.",
    "Mean speed": "Mean speed of the cars on the ring.",
    "Mean free speed": "Mean speed the drivers on the ring want to go: their desired speed, capped by "
                       "comfortable cornering on their lane.",
    "Ring flux": "Cars per second passing a cross-section of the ring, from every car's speed "
                 "(sum of v / lane length over all lanes).",
    "Detector flux": f"Cars per second crossing the detector line in the view, over the last {WINDOW}.",
    "Stop-and-go": STOPS_HELP,
    "Jammed cars": f"Share of cars crawling below {SLOW_FRACTION:.0%} of their free speed: the headline "
                   "congestion measure, rising steadily with density.",
    "Jams": "Separate queues of 2+ jammed cars in a lane, now. Hover the jams readout of the stop-and-go "
            "chart for what holds them up.",
    "Following share of jams": f"Share of the jams seen over the last {WINDOW} whose front car is just following "
                               "the car ahead. Only an upper bound on natural (phantom) jams: a wave set off at "
                               "an exit travels back with ordinary followers at its front. Compare with Exit "
                               "chance = 0 to see what the entrances and exits cause.",
    "Speed deficit": "1 − total speed / total free speed of the cars on the ring (0 = free flow, "
                     "100 % = standstill).",
    "Harsh braking (episodes/min)": f"Harsh-braking episodes per minute over the last {WINDOW}: the driver model "
                                    f"asking for more than {MAX_DECEL:g} m/s² (an over-reaction after a late or "
                                    "delayed response), or the no-overlap constraint stopping a car harder. "
                                    "Grows with shorter reaction distance, longer reaction time and density.",
    "Spacing (Gini)": SPACING_HELP,
    "Clustering": CLUSTERING_HELP,
    "Order": f"Order parameter now, and what randomly placed cars would give. {ORDER_HELP.removeprefix('Order: ')}",
    "Exits": f"Cars leaving the ring per minute, over the last {WINDOW}.",
    "Lane changes": f"Lane changes per minute, over the last {WINDOW}.",
    "Missed exits": "Leavers that reached their exit on an inner lane and had to go round again "
                    "(since the reset).",
    "Mean trip time": f"Mean time on the ring of the cars that left over the last {WINDOW}.",
    "Lane": "Mean speed and number of cars in this lane (lane 1 is the inner lane).",
}


def help_texts() -> list[str]:
    """Every fixed help text the status box can show (it is sized to fit the longest)."""
    texts = [help_text for _, _, help_text in BUTTONS]
    texts += [control_help(spec) for spec in (*CONTROL_PARAMS, VIEW_PARAMS["colour_mode"])]
    texts += [spec.help for spec in CHARTS] + list(EXTRA_HELP.values())
    texts += [f"{'Lane 8 outer' if label == 'Lane' else label}: {text}" for label, text in STAT_HELP.items()]
    return texts + [SPACETIME_HELP, VIEW_HELP, DEFAULT_HINT]


@dataclass(frozen=True)
class Stat:
    """One readout in the stats card; ``short`` is a (label, value) pair for narrow windows."""
    label: str
    value: str
    swatch: tuple | None = None
    short: tuple[str, str] | None = None
    key: str | None = None  # STAT_HELP entry, if not the label

    def text(self, short: bool) -> tuple[str, str]:
        return self.short if short and self.short else (self.label, self.value)

    def help_text(self) -> str:
        return f"{self.label}: {STAT_HELP[self.key or self.label]}"


def mix(a, b, t: float) -> tuple[int, int, int]:
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def one_sided(order: float, baseline: float) -> bool:
    """True when the order parameter is clearly above what randomly placed cars give."""
    base = baseline if np.isfinite(baseline) else 0.0
    return bool(order > max(0.05, ONE_SIDED * base))


def value_text(value) -> str:
    """A setting as the experiments CLI takes it (``--set name=value``)."""
    if isinstance(value, float):
        return format(value, ".10g")
    return str(value)


def occupancy_rgb(frac) -> np.ndarray:
    """Sequential ramp for the dark surface: 0 = darkest step (recedes), 1 = lightest."""
    return palette.sequential_rgb(frac, dark=True)


def finite_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) index ranges of the runs of True in a boolean array."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], valid.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def speed_colours(speed: np.ndarray, free: float) -> np.ndarray:
    """uint8 RGB per bin of a speed profile; empty (NaN) bins get the empty colour."""
    empty = np.isnan(speed)
    rgb = palette.speed_rgb(np.where(empty, 0.0, speed) / max(free, 0.1))
    return np.where(empty[:, None], np.array(EMPTY), rgb).astype(np.uint8)


def nice_step(span: float, target: int) -> float:
    raw = max(span, 1e-9) / max(target, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    return next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag * 1.0001)


def anchor_out(a: float) -> str:
    """Rect anchor that puts a label on the outward side of a point at world angle ``a``."""
    c, s = math.cos(a), math.sin(a)
    v = "bottom" if s > 0.38 else "top" if s < -0.38 else ""
    h = "left" if c > 0.38 else "right" if c < -0.38 else ""
    if v and h:
        return v + h
    return "mid" + (v or h) if (v or h) else "center"


def decimals(step: float) -> int:
    return 0 if step >= 1 else len(f"{step:g}".split(".")[1])


def format_value(spec: ParamSpec, value) -> str:
    if spec.choices:
        return str(value)
    return f"{value:.{decimals(spec.step)}f}"


# ---------------------------------------------------------------- helpers
class Fonts:
    """System fonts at one UI scale, with a cache of rendered strings."""

    def __init__(self, scale: float):
        self.scale = scale
        px = lambda size: max(10, round(size * scale))  # noqa: E731
        self.fonts = {
            "title": pygame.font.SysFont(FONT_FACE, px(18), bold=True),
            "base": pygame.font.SysFont(FONT_FACE, px(14)),
            "small": pygame.font.SysFont(FONT_FACE, px(12)),
        }
        self._cache: dict[tuple, pygame.Surface] = {}

    def render(self, text: str, font: str = "base", colour=UI["text"]) -> pygame.Surface:
        key = (text, font, colour)
        surf = self._cache.get(key)
        if surf is None:
            if len(self._cache) > 4000:
                self._cache.clear()
            surf = self._cache[key] = self.fonts[font].render(text, True, colour)
        return surf

    def blit(self, target: pygame.Surface, text: str, pos, font: str = "base", colour=UI["text"],
             anchor: str = "topleft") -> pygame.Rect:
        surf = self.render(text, font, colour)
        rect = surf.get_rect(**{anchor: (round(pos[0]), round(pos[1]))})
        target.blit(surf, rect)
        return rect

    def width(self, text: str, font: str = "base") -> int:
        return self.fonts[font].size(text)[0]

    def line(self, font: str = "base") -> int:
        return self.fonts[font].get_linesize()

    def wrap(self, text: str, width: int, font: str = "small") -> list[str]:
        lines, current = [], ""
        for word in text.split(" "):
            trial = f"{current} {word}".strip()
            if current and self.width(trial, font) > width:
                lines.append(current)
                current = word
            else:
                current = trial
            while len(current) > 1 and self.width(current, font) > width:  # a word longer than a line (a path)
                n = len(current) - 1
                while n > 1 and self.width(current[:n], font) > width:
                    n -= 1
                lines.append(current[:n])
                current = current[n:]
        return lines + [current] if current else lines


@dataclass
class Widget:
    kind: str  # "button", "slider" or "choice"
    name: str  # action, parameter or group name
    rect: pygame.Rect  # hover area
    spec: ParamSpec | None = None
    help: str = ""
    track: pygame.Rect | None = None  # slider track
    options: list[pygame.Rect] = field(default_factory=list)  # choice buttons
    scrolls: bool = False  # lives in the scrollable parameter list


@dataclass
class Drag:
    widget: Widget
    start: float
    value: float


class History:
    """Chart series pulled incrementally from a :class:`metrics.Recorder`."""

    KEYS = ("time", "mean_speed", "mean_free_speed", "flux", "detector_flux", "stops_per_car_min",
            "jam_fraction", "speed_deficit", "n_jams", "jams_following", "jams_merge", "jams_exit",
            "jams_entry", "jams_other", "clustering", "order_param", "order_angle", "order_baseline",
            "spacing_gini")

    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.clear()

    def clear(self) -> None:
        self.cols = {k: deque(maxlen=self.maxlen) for k in self.KEYS}
        self.profile: deque[np.ndarray] = deque(maxlen=self.maxlen)  # mean speed per angle bin
        self.speed_rgb: deque[np.ndarray] = deque(maxlen=self.maxlen)  # angle bins top (360°) to bottom
        self.occupancy: deque[np.ndarray] = deque(maxlen=self.maxlen)
        self.version = 0
        self._arrays: dict[str, np.ndarray] = {}
        self._free = 1.0

    def __len__(self) -> int:
        return len(self.cols["time"])

    def pull(self, rec: metrics.Recorder) -> None:
        times = self.cols["time"]
        last = times[-1] if times else -np.inf
        new = 0
        for row in reversed(rec.rows):
            if row["time"] <= last + 1e-9:
                break
            new += 1
        for j in range(new, 0, -1):
            row = rec.rows[-j]
            for key in self.KEYS:
                self.cols[key].append(row[key])
            # Space-time speeds are shown relative to the cars' mean free speed
            # (NaN on an empty ring: keep the last one, nothing is drawn anyway).
            if np.isfinite(row["mean_free_speed"]) and row["mean_free_speed"] > 0:
                self._free = row["mean_free_speed"]
            self.profile.append(rec.spacetime[-j])
            self.speed_rgb.append(speed_colours(rec.spacetime[-j], self._free)[::-1])
            self.occupancy.append(rec.occupancy[-j])
        if new:
            self.version += 1
            self._arrays = {}

    def array(self, key: str) -> np.ndarray:
        if key not in self._arrays:
            self._arrays[key] = np.fromiter(self.cols[key], dtype=float, count=len(self.cols[key]))
        return self._arrays[key]


# ------------------------------------------------------------ ring view
class RingView:
    """The roundabout at true scale: world/screen transform, cached road, cars on top."""

    def __init__(self, sim: RingSim, rect: pygame.Rect, fonts: Fonts):
        cfg = sim.cfg
        self.rect = rect
        self.lw = cfg.lane_width
        self.r_in = float(sim.radii[0] - self.lw / 2)
        self.r_edge = float(sim.r_out + self.lw / 2)
        self.stub = float(np.clip(0.22 * self.r_edge, 12.0, 40.0)) if sim.n_arms else 0.0
        # Room outside the road for the queue counts and the detector label.
        margin = max(2.6 * fonts.line("small"), fonts.width("detector", "small") + 10)
        self.k = max((rect.w / 2 - margin) / (self.r_edge + self.stub), 0.05)  # [px/m]
        self.cx, self.cy = rect.centerx, rect.centery
        # An arm's exit and entry lanes sit this far either side of its axis.
        self.arm_offset = float(sim.r_out * np.sin(sim.entry_angle[0] - sim.arm_angle[0])) if sim.n_arms else 0.0
        self.layer = self._draw_road(sim, fonts)
        self._sector_geometry()

    def to_world(self, px: float, py: float) -> tuple[float, float]:
        return (px - self.cx) / self.k, -(py - self.cy) / self.k

    def point(self, r, lat, a, local: bool = False):
        """Screen point at radius ``r`` along angle ``a``, ``lat`` metres to its anticlockwise side."""
        x = r * np.cos(a) - lat * np.sin(a)
        y = r * np.sin(a) + lat * np.cos(a)
        ox, oy = (self.rect.w / 2, self.rect.h / 2) if local else (self.cx, self.cy)
        return ox + x * self.k, oy - y * self.k

    def _draw_road(self, sim: RingSim, fonts: Fonts) -> pygame.Surface:
        surf = pygame.Surface(self.rect.size)
        surf.fill(UI["background"])
        k, lw = self.k, self.lw
        pt = lambda r, lat, a: self.point(r, lat, a, local=True)  # noqa: E731
        edge = UI["road_edge"]
        dash = mix(edge, UI["road"], 0.4)
        half_w = self.arm_offset + lw / 2
        r_end = self.r_edge + self.stub
        for a in sim.arm_angle:
            pygame.draw.polygon(surf, UI["road"], [pt(sim.r_out, -half_w, a), pt(r_end, -half_w, a),
                                                   pt(r_end, half_w, a), pt(sim.r_out, half_w, a)])
        centre = (self.rect.w / 2, self.rect.h / 2)
        lane_fill = (UI["road"], mix(UI["road"], UI["background"], 0.3))
        for lane in range(sim.n_lanes - 1, -1, -1):  # outer to inner, alternating surfaces
            pygame.draw.circle(surf, lane_fill[(sim.n_lanes - 1 - lane) % 2], centre, (sim.radii[lane] + lw / 2) * k)
        pygame.draw.circle(surf, UI["background"], centre, self.r_in * k)

        for lane in range(sim.n_lanes - 1):
            r = sim.radii[lane] + lw / 2
            n = max(12, int(TWO_PI * r * k / 14))  # a dash every 14 px
            for a in np.arange(n) * TWO_PI / n:
                pygame.draw.line(surf, dash, pt(r, 0, a), pt(r, 0, a + 0.5 * TWO_PI / n))

        gap = math.asin(min(half_w / self.r_edge, 1.0)) if sim.n_arms else 0.0
        for run in self._edge_runs(sim.arm_angle, gap):
            pygame.draw.aalines(surf, edge, False, run)
        n = max(90, int(TWO_PI * self.r_in * k / 3))
        a = np.arange(n) * TWO_PI / n
        pygame.draw.aalines(surf, edge, True, list(zip(*pt(self.r_in, 0, a))))

        r0 = math.sqrt(max(self.r_edge ** 2 - half_w ** 2, 0.0))
        size = min(0.35 * lw, 0.15 * self.stub)
        for a in sim.arm_angle:
            for lat in (-half_w, half_w):
                pygame.draw.aaline(surf, edge, pt(r0, lat, a), pt(r_end, lat, a))
            pygame.draw.line(surf, dash, pt(self.r_edge, 0, a), pt(r_end, 0, a))
            if size * k >= 2.5:  # direction arrows: out on the exit lane, in on the entry lane
                r = r_end - 0.3 * self.stub
                for lat, sign in ((-self.arm_offset, 1), (self.arm_offset, -1)):
                    pygame.draw.polygon(surf, dash, [pt(r + sign * size, lat, a), pt(r - sign * size, lat - size, a),
                                                     pt(r - sign * size, lat + size, a)])

        det = sim.detector_angle
        pygame.draw.line(surf, UI["text2"], pt(self.r_in, 0, det), pt(self.r_edge, 0, det), 2)
        fonts.blit(surf, "detector", pt(self.r_edge + 4 / k, 0, det), "small", UI["muted"], anchor_out(det))
        inner_px = self.r_in * k
        if inner_px > 6 * fonts.line("small"):  # angle labels on the island, to read the space-time axis
            r = (inner_px - 0.16 * inner_px - 6) / k
            for deg in (0, 90, 180, 270):
                a = math.radians(deg)
                fonts.blit(surf, f"{deg}°", pt(r, 0, a), "small", UI["muted"], anchor_out(a + math.pi))
        return surf

    def _edge_runs(self, arms: np.ndarray, gap: float) -> list[list[tuple[float, float]]]:
        """Outer road edge as polylines, broken where the arms join."""
        n = max(180, int(TWO_PI * self.r_edge * self.k / 3))
        a = np.arange(n) * TWO_PI / n
        keep = np.ones(n, dtype=bool)
        for arm in arms:
            keep &= np.abs((a - arm + np.pi) % TWO_PI - np.pi) > gap
        x, y = self.point(self.r_edge, 0, a, local=True)
        if keep.all():
            return [list(zip(x, y)) + [(x[0], y[0])]]
        start = int(np.flatnonzero(~keep)[0])
        order = np.roll(np.arange(n), -start)
        runs, current = [], []
        for i in order:
            if keep[i]:
                current.append((x[i], y[i]))
            elif len(current) > 1:
                runs.append(current)
                current = []
            else:
                current = []
        if len(current) > 1:
            runs.append(current)
        return runs

    def _sector_geometry(self) -> None:
        """Polygons of the 24 occupancy sectors, a thin band just inside the inner edge."""
        inner_px = self.r_in * self.k
        self.band_outer = inner_px - max(2.0, 0.03 * inner_px)
        self.band_inner = self.band_outer - float(np.clip(0.05 * inner_px, 3.0, 7.0))
        n = metrics.SECTORS
        spacer = 1.0 / max(self.band_outer, 1.0)  # ~1 px gap between sectors
        self.sectors = []
        for j in range(n):
            a = np.linspace(TWO_PI * j / n + spacer, TWO_PI * (j + 1) / n - spacer, 6)
            outer = zip(self.cx + self.band_outer * np.cos(a), self.cy - self.band_outer * np.sin(a))
            inner = zip(self.cx + self.band_inner * np.cos(a[::-1]), self.cy - self.band_inner * np.sin(a[::-1]))
            self.sectors.append(list(outer) + list(inner))

    def car_polygons(self, x, y, heading, length: float, width: float, pad_px: float = 0.0) -> np.ndarray:
        """(n, 4, 2) screen corners of oriented rectangles centred on world points."""
        k = self.k
        sx, sy = self.cx + x * k, self.cy - y * k
        c, s = np.cos(heading), np.sin(heading)
        hl = max(length * k, 2.0) / 2 + pad_px
        hw = max(width * k, 1.2) / 2 + pad_px
        fx, fy = c * hl, -s * hl  # forward, screen y points down
        nx, ny = -s * hw, -c * hw  # left of forward
        corners = [(sx + fx + nx, sy + fy + ny), (sx + fx - nx, sy + fy - ny),
                   (sx - fx - nx, sy - fy - ny), (sx - fx + nx, sy - fy + ny)]
        return np.stack([np.stack(p, axis=-1) for p in corners], axis=1)

    def draw(self, screen: pygame.Surface, sim: RingSim, snap: dict, colours: np.ndarray,
             fonts: Fonts, row: dict, hovered: int | None) -> None:
        cfg = sim.cfg
        screen.blit(self.layer, self.rect)
        self._draw_queues(screen, sim, fonts)

        jam_cover = cfg.car_length / (cfg.car_length + cfg.min_gap)
        occ = occupancy_rgb(metrics.sector_occupancy(sim) / jam_cover).astype(int).tolist()
        for poly, colour in zip(self.sectors, occ):
            pygame.draw.polygon(screen, colour, poly)
        self._draw_order_arrow(screen, row)

        if snap["slot"].size:
            self._draw_cars(screen, snap, colours, cfg)
            braking = snap["braking"]
            if braking.any():
                rings = self.car_polygons(snap["x"][braking], snap["y"][braking], snap["heading"][braking],
                                          cfg.car_length, cfg.car_width, pad_px=2.0)
                for poly in rings.tolist():
                    pygame.draw.polygon(screen, palette.CRITICAL, poly, 2)
            if hovered is not None:
                i = np.flatnonzero(snap["slot"] == hovered)
                if i.size:
                    poly = self.car_polygons(snap["x"][i], snap["y"][i], snap["heading"][i],
                                             cfg.car_length, cfg.car_width, pad_px=1.5)[0]
                    pygame.draw.polygon(screen, UI["text"], poly.tolist(), 1)

    def _draw_cars(self, screen: pygame.Surface, snap: dict, colours: np.ndarray, cfg: SimConfig) -> None:
        """One filled, oriented rectangle per car (at least 2 x 1.2 px, so tiny cars stay visible).

        Cheap enough for the biggest rings: ~2000 cars take about 1.5 ms, less
        than a numpy pixel splat of the same cars measured.
        """
        polys = self.car_polygons(snap["x"], snap["y"], snap["heading"], cfg.car_length, cfg.car_width)
        for poly, colour in zip(polys.tolist(), colours.tolist()):
            pygame.draw.polygon(screen, colour, poly)

    def _draw_queues(self, screen: pygame.Surface, sim: RingSim, fonts: Fonts) -> None:
        cfg = sim.cfg
        pitch = cfg.car_length + cfg.min_gap
        fit = max(int((self.stub - 1.0) // pitch), 0)
        xs, ys, hs = [], [], []
        for a, q in zip(sim.arm_angle, sim.queue):
            shown = min(int(q), fit)
            r = self.r_edge + 1.0 + cfg.car_length / 2 + pitch * np.arange(shown)
            xs.append(r * np.cos(a) - self.arm_offset * np.sin(a))
            ys.append(r * np.sin(a) + self.arm_offset * np.cos(a))
            hs.append(np.full(shown, a + np.pi))
            if q > fit:
                fonts.blit(screen, str(int(q)), self.point(self.r_edge + self.stub + 3 / self.k, self.arm_offset, a),
                           "small", UI["text2"], anchor_out(a))
        if xs and sum(len(x) for x in xs):
            polys = self.car_polygons(np.concatenate(xs), np.concatenate(ys), np.concatenate(hs),
                                      cfg.car_length, cfg.car_width)
            for poly in polys.tolist():
                pygame.draw.polygon(screen, UI["muted"], poly)

    def _draw_order_arrow(self, screen: pygame.Surface, row: dict) -> None:
        order = row.get("order_param", 0.0)
        room = self.band_inner - 6
        if not one_sided(order, row.get("order_baseline", np.nan)) or room < 10:
            return  # no more one-sided than randomly placed cars: no direction worth pointing at
        length = min(1.0, 2.0 * order) * room  # proportional, full length at order 0.5
        a = row["order_angle"]
        ux, uy = math.cos(a), -math.sin(a)
        tip = (self.cx + ux * length, self.cy + uy * length)
        head = min(8.0, 0.4 * length)
        base = (tip[0] - ux * head, tip[1] - uy * head)
        pygame.draw.line(screen, UI["text2"], (self.cx, self.cy), base, 2)
        pygame.draw.polygon(screen, UI["text2"], [tip, (base[0] - uy * head * 0.6, base[1] + ux * head * 0.6),
                                                  (base[0] + uy * head * 0.6, base[1] - ux * head * 0.6)])
        pygame.draw.circle(screen, UI["text2"], (self.cx, self.cy), 2)


# ------------------------------------------------------------------ app
class App:
    """Pygame window: ring view, stats, live charts and the control panel."""

    def __init__(self, config: SimConfig | None = None, size: tuple[int, int] | None = None,
                 results_dir: str | Path = "results"):
        _dpi_aware()
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        pygame.init()
        pygame.display.set_caption("Roundabout traffic model")
        self.screen = pygame.display.set_mode(size or _initial_size(), pygame.RESIZABLE)
        self.sim = RingSim(config if config is not None else randomized_config())
        self.rec = metrics.Recorder(self.sim, SAMPLE_INTERVAL, maxlen=int(HISTORY / SAMPLE_INTERVAL))
        self.hist = History(self.rec.maxlen)
        self.results_dir = Path(results_dir)
        self.clock = pygame.time.Clock()
        self.paused = False
        self.steps_per_frame = 1
        self.colour_mode = "speed"
        self.spacetime_mode = "speed"
        self.scroll = 0
        self.drag: Drag | None = None
        self.pressed: tuple[Widget, int] | None = None  # button or choice option under a held click
        self.mouse = (-1, -1)
        self.markers: deque[tuple[float, str]] = deque()  # (time, "Density 50→80")
        self.brakes: deque[tuple[float, float]] = deque()  # (time, angle) of perturbed cars
        self.message = ("", 0.0)
        self.work_ms: deque[float] = deque(maxlen=240)  # update + draw time per frame
        self.fonts: Fonts | None = None
        self.ring: RingView | None = None
        self.stat_hits: list[tuple[pygame.Rect, str]] = []  # (hover area, help) of each readout
        self.chart_hits: list[tuple[pygame.Rect, str]] = []  # (hover area, help) of chart header readouts
        self._status_fit: tuple[tuple, int] = ((), STATUS_LINES[0])  # (font scale, width), lines
        self._gradients: dict[tuple, pygame.Surface] = {}
        self._st_cache: tuple | None = None
        self._generation = self.sim.generation  # bumped by every reset / rebuild
        self.rec.update()
        self.hist.pull(self.rec)
        self.layout()

    # ------------------------------------------------------------ layout
    def layout(self) -> None:
        W, H = self.screen.get_size()
        s = round(float(np.clip(min(W / 1500, H / 850), 0.85, 2.0)) * 20) / 20
        if self.fonts is None or self.fonts.scale != s:
            self.fonts = Fonts(s)
            self._gradients = {}
        self.s = s
        self._layout_size = (W, H)
        m = self.gap = round(10 * s)
        panel_w = int(np.clip(310 * s, 290, 0.32 * W))
        self.panel_rect = pygame.Rect(W - panel_w, 0, panel_w, H)
        area = pygame.Rect(m, m, W - panel_w - 2 * m, H - 2 * m)
        stats_h = self._stats_height()
        side = int(max(200, min(area.h - stats_h - m, area.w * 0.56)))
        self.view_rect = pygame.Rect(area.x, area.y, side, side)
        self.stats_rect = pygame.Rect(area.x, area.y + side + m, side, area.bottom - area.y - side - m)
        self.charts_rect = pygame.Rect(area.x + side + m, area.y, area.right - area.x - side - m, area.h)
        self.ring = None
        self._st_cache = None
        self._layout_charts()
        self._layout_panel()

    def _latest_row(self) -> dict:
        return self.rec.rows[-1] if self.rec.rows else metrics.snapshot(self.sim)

    def _stats_height(self) -> int:
        """Height the stats card needs for two legend lines and its columns of readouts."""
        f = self.fonts
        rows = max(len(col) for col in self._stat_columns(self._latest_row()))
        return 2 * round(8 * self.s) + 2 * f.line("small") + round(6 * self.s) + rows * (f.line("small") + 1)

    def _layout_charts(self) -> None:
        """Line charts stacked (two columns on very wide windows) above the space-time diagram.

        All charts share one time axis, so only the bottom row labels it; the
        others keep its tick marks. Plot areas get equal heights.
        """
        f, s, r, m = self.fonts, self.s, self.charts_rect, self.gap
        pad = round(8 * s)
        st_h = int(r.h * SPACETIME_SHARE)
        cols = 2 if r.w >= 900 * s else 1
        rows = math.ceil(len(CHARTS) / cols)
        card_w = (r.w - (cols - 1) * m) / cols
        y_label = f.width("000.0", "small") + round(6 * s)
        x_band = f.line("small") + 2
        tick_band = round(4 * s)
        cells = []  # (row, col, columns spanned): an odd last chart spans the row
        for i in range(len(CHARTS)):
            row, col = divmod(i, cols)
            cells.append((row, col, cols if i == len(CHARTS) - 1 and len(CHARTS) % cols else 1))
        widths = [round(span * card_w + (span - 1) * m) for _, _, span in cells]
        two = [self._header_width(spec) > w - 2 * pad for spec, w in zip(CHARTS, widths)]
        header = [f.line("base") * (1 + any(t for (rw, _, _), t in zip(cells, two) if rw == row)) + round(6 * s)
                  for row in range(rows)]
        fixed = rows * 2 * pad + sum(header) + x_band + (rows - 1) * (tick_band + m)
        plot_h = max((r.h - st_h - m - fixed) / rows, 10)
        self.chart_cards = []
        y = r.y
        for row in range(rows):
            last = row == rows - 1
            band = x_band if last else tick_band
            # The last row takes up the rounding, so it ends exactly a gap above the space-time card.
            card_h = r.bottom - st_h - m - y if last else round(2 * pad + header[row] + plot_h + band)
            for i, (rw, col, span) in enumerate(cells):
                if rw != row:
                    continue
                card = pygame.Rect(round(r.x + col * (card_w + m)), y, widths[i], card_h)
                plot = pygame.Rect(card.x + pad + y_label, card.y + pad + header[row],
                                   card.w - 2 * pad - y_label, max(card_h - 2 * pad - header[row] - band, 10))
                self.chart_cards.append((CHARTS[i], card, plot, two[i], last))
            y += card_h + m
        card = pygame.Rect(r.x, r.bottom - st_h, r.w, st_h)
        y_label = f.width("360°", "small") + f.line("small") + round(8 * s)
        header = f.line("base") + round(8 * s)
        plot = pygame.Rect(card.x + pad + y_label, card.y + pad + header,
                           card.w - 2 * pad - y_label - round(6 * s), card.h - 2 * pad - header - x_band)
        self.st_card, self.st_plot = card, plot
        # Speed / occupancy toggle in the space-time header.
        spec = VIEW_PARAMS["spacetime_mode"]
        opt_w = max(f.width(c, "small") for c in spec.choices) + round(14 * s)
        opt_h = f.line("small") + round(2 * s)
        x = card.x + pad + f.width("Space-time diagram", "base") + round(12 * s)
        y = card.y + pad + (f.line("base") - opt_h) // 2
        options = [pygame.Rect(x + i * opt_w, y, opt_w, opt_h) for i in range(len(spec.choices))]
        self.chart_widgets = [Widget("choice", spec.name, options[0].union(options[-1]), spec,
                                     spec.help, options=options)]

    def _header_width(self, spec: ChartSpec) -> int:
        """Width a one-line header needs (title, legend or value, shortest extra readout)."""
        f = self.fonts
        width = f.width(spec.title) + round(16 * self.s)
        if len(spec.series) > 1:
            width += sum(f.width(f"{s.label} 00.00", "small") + round(37 * self.s) for s in spec.series)
        else:
            width += f.width("00.00")
        if spec.extra:
            sample = dict(n_jams=88, jams_following=88, jams_merge=8, jams_exit=8, jams_entry=8, jams_other=8,
                          order_param=0.5, order_angle=math.radians(355), order_baseline=0.05)
            width += f.width(self._extra_texts(spec.extra, sample)[-1], "small") + round(12 * self.s)
        return width

    def _layout_panel(self) -> None:
        f, s, r = self.fonts, self.s, self.panel_rect
        p = round(14 * s)
        x, w = r.x + p, r.w - 2 * p
        y = r.y + p + f.line("title") + f.line("small") + round(10 * s)
        self.panel_widgets: list[Widget] = []
        gap = round(6 * s)
        bh = f.line("base") + round(8 * s)
        helps = {name: help_text for name, _, help_text in BUTTONS}
        for row, names in enumerate(BUTTON_ROWS):  # each row shares out the full width
            bw = (w - (len(names) - 1) * gap) / len(names)
            for col, name in enumerate(names):
                left, right = round(x + col * (bw + gap)), round(x + col * (bw + gap) + bw)
                rect = pygame.Rect(left, y + row * (bh + gap), right - left, bh)
                self.panel_widgets.append(Widget("button", name, rect, help=helps[name]))
        y += len(BUTTON_ROWS) * (bh + gap) - gap + round(14 * s)
        for spec in (VIEW_PARAMS["colour_mode"],):
            y = self._add_control(spec, x, y, w, scrolls=False)
        y += round(4 * s)
        y += round(10 * s)
        self.status_lines = self._status_lines(w)
        status_h = self.status_lines * f.line("small") + round(12 * s)
        self.status_rect = pygame.Rect(x, r.bottom - p - status_h, w, status_h)
        self.content_rect = pygame.Rect(r.x, y, r.w, self.status_rect.y - round(8 * s) - y)
        self.content_h = sum(self._control_height(spec) for spec in CONTROL_PARAMS)
        self.scroll = int(np.clip(self.scroll, 0, max(0, self.content_h - self.content_rect.h)))
        y -= self.scroll
        for spec in CONTROL_PARAMS:
            y = self._add_control(spec, x, y, w, scrolls=True)

    def _status_lines(self, width: int) -> int:
        """Lines the help box needs so that the longest help text fits ``width`` (cached per
        font scale and width; within STATUS_LINES)."""
        key = (self.fonts.scale, width)
        if self._status_fit[0] != key:
            need = max(len(self.fonts.wrap(text, width)) for text in help_texts())
            self._status_fit = (key, int(np.clip(need, *STATUS_LINES)))
        return self._status_fit[1]

    def _control_height(self, spec: ParamSpec) -> int:
        f, s = self.fonts, self.s
        body = f.line("small") + round(6 * s) if spec.choices else round(18 * s)
        return f.line("base") + body + round(10 * s)

    def _add_control(self, spec: ParamSpec, x: int, y: int, w: int, scrolls: bool) -> int:
        f, s = self.fonts, self.s
        h = self._control_height(spec)
        rect = pygame.Rect(x, y, w, h)
        top = y + f.line("base") + round(2 * s)
        if spec.choices:
            n = len(spec.choices)
            oh = f.line("small") + round(6 * s)
            options = [pygame.Rect(round(x + i * w / n), top, round(w / n) - 2, oh) for i in range(n)]
            widget = Widget("choice", spec.name, rect, spec, spec.help, options=options, scrolls=scrolls)
        else:
            knob = round(7 * s)
            track = pygame.Rect(x + knob, top + round(9 * s) - 2, w - 2 * knob, 4)
            widget = Widget("slider", spec.name, rect, spec, spec.help, track=track, scrolls=scrolls)
        self.panel_widgets.append(widget)
        return y + h

    @property
    def widgets(self) -> list[Widget]:
        return self.panel_widgets + self.chart_widgets

    # ------------------------------------------------------------ state
    def value(self, name: str):
        if self.drag and self.drag.widget.name == name:
            return self.drag.value
        if name in VIEW_PARAMS:
            return getattr(self, name)
        return getattr(self.sim.cfg, name)

    def apply(self, name: str, value) -> None:
        if name in VIEW_PARAMS:
            setattr(self, name, int(value) if name == "steps_per_frame" else value)
            return
        value = coerce(name, value)
        if value != getattr(self.sim.cfg, name) and self.sim.set(**{name: value}):
            self._clear_history()

    def _mark(self, spec: ParamSpec, old, new) -> None:
        if spec.group != "View" and old != new:
            self.markers.append((self.sim.time, f"{spec.label} {format_value(spec, old)}→{format_value(spec, new)}"))

    def _clear_history(self) -> None:
        """Start the charts afresh (needed after a rebuild: the time restarts at 0)."""
        self._generation = self.sim.generation
        self.rec.clear()
        self.hist.clear()
        self.markers.clear()
        self.brakes.clear()
        self.ring = None
        self._st_cache = None
        self.rec.update()
        self.hist.pull(self.rec)
        if self.stats_rect.h < self._stats_height():  # more lanes need more stat rows
            self.layout()

    def _follow_rebuilds(self) -> None:
        """Clear the charts and the cached road whenever the simulation was reset or rebuilt,
        whatever did it (the recorder clears itself on the same signal)."""
        if self.sim.generation != self._generation:
            self._clear_history()

    def advance(self, steps: int) -> None:
        self._follow_rebuilds()
        for _ in range(steps):
            self.sim.step()
            self.rec.update()
        self.hist.pull(self.rec)

    def action(self, name: str) -> None:
        if name == "pause":
            self.paused = not self.paused
        elif name == "step":
            self.advance(self.steps_per_frame)
        elif name == "reset":
            self.sim.reset(seed=int(np.random.default_rng().integers(2**31)))
            self._clear_history()
        elif name == "brake":
            self._record_brake(self.sim.brake_random())
        elif name == "clear":
            self._clear_history()
        elif name == "screenshot":
            self.screenshot()
        elif name == "csv":
            self.save_data()

    def _record_brake(self, slot: int | None) -> None:
        """Remember where and when a car was made to brake, for the space-time diagram."""
        if slot is not None:
            self.brakes.append((self.sim.time, float(self.sim.theta[slot])))

    def _new_paths(self, *suffixes: str) -> list[Path]:
        """results/roundabout_<timestamp><suffix> for each suffix, none of them taken yet
        (saving twice in the same second adds -1, -2, ...)."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        stem = time.strftime("roundabout_%Y%m%d_%H%M%S")
        n = 0
        while True:
            paths = [self.results_dir / f"{stem}{'-' + str(n) if n else ''}{suffix}" for suffix in suffixes]
            if not any(p.exists() for p in paths):
                return paths
            n += 1

    def screenshot(self) -> Path:
        path, = self._new_paths(".png")
        pygame.image.save(self.screen, str(path))
        self.message = (f"Saved {path}", time.monotonic() + 4.0)
        return path

    def save_data(self) -> tuple[Path, Path]:
        """The recorded chart data as CSV, and a text file with the settings, the live parameter
        changes and the experiments command that reruns the final settings from the start."""
        from roundabout_sim import experiments  # only needed here; it pulls in the CLI

        csv_path, txt_path = self._new_paths(".csv", "_settings.txt")
        frame = self.rec.frame()
        renames = getattr(experiments, "measured_names", None)  # density -> measured_density, ...
        if callable(renames):
            frame = frame.rename(columns=renames(frame.columns))
        frame.to_csv(csv_path, index=False, float_format="%.10g")
        txt_path.write_text(self._settings_text(csv_path, frame, experiments), encoding="utf-8")
        self.message = (f"Saved {csv_path} ({HISTORY_TEXT} of chart data, a row every {SAMPLE_INTERVAL:g} s) "
                        f"and {txt_path.name} (settings, live changes, equivalent command).",
                        time.monotonic() + 6.0)
        return csv_path, txt_path

    def _settings_text(self, csv_path: Path, frame, experiments) -> str:
        sim = self.sim
        settings = getattr(experiments, "settings", None)
        values = settings(sim) if callable(settings) else {**asdict(sim.cfg), "seed": sim.seed}
        t = frame["time"] if "time" in frame else []
        span = f"t = {t.iloc[0]:.1f}-{t.iloc[-1]:.1f} s" if len(t) else "no samples"
        lines = ["Roundabout traffic model: data saved from the viewer",
                 f"saved {time.strftime('%Y-%m-%d %H:%M:%S')}",
                 "",
                 f"{csv_path.name}: {len(frame)} rows, one every {SAMPLE_INTERVAL:g} s of simulated time, {span}",
                 f"  (the charts keep {HISTORY_TEXT}; older samples are not in the file)."]
        measured = [str(c) for c in frame.columns if str(c).startswith("measured_")]
        if measured:
            lines += ["  Measurements named like a setting carry a measured_ prefix: " + ", ".join(measured),
                      "  (measured_density is the density actually on the ring, density the setting)."]
        lines += ["",
                  "main columns",
                  f"  jam_fraction        share of cars below {SLOW_FRACTION:.0%} of their free speed: the headline "
                  "congestion measure",
                  "  speed_deficit       1 - total speed / total free speed (0 = free flow, 1 = standstill)",
                  f"  stops_per_car_min   stop-and-go, stops per car per minute over {WINDOW}: ~0 in smooth flow",
                  "                      and also in a standstill, so it measures instability, not how jammed",
                  "  n_jams, jams_*      jams now, by what their front car is doing; jams_following is only an upper",
                  "                      bound on natural (phantom) jams: compare with a run at exit_prob = 0",
                  f"  emergency_per_min   harsh-braking episodes per minute over {WINDOW}",
                  "  spacing_gini        road each car has: 0 = even, ~0.35 = random gaps, towards 1 = packed queues",
                  "  order_param         0 = balanced around the ring, 1 = all on one side; order_baseline is what",
                  "                      randomly placed cars give (only well above it is real one-sidedness)",
                  "  flux, occupancy     cars per second past a point (all lanes); share of the road covered",
                  "",
                  f"settings at the time of saving (seed {sim.seed})"]
        for name, value in values.items():
            if name == "seed":
                continue
            spec = PARAM_BY_NAME.get(name)
            what = f"{spec.label}{', ' + spec.unit if spec.unit else ''}" if spec else ""
            lines.append(f"  {f'{name} = {value_text(value)}':<32}{what}".rstrip())
        lines += ["", "live parameter changes since the last rebuild or chart clear (time, change)"]
        lines += [f"  {when:8.1f} s  {label}" for when, label in self.markers] or ["  none"]
        if self.brakes:
            lines += ["", "cars you made brake (time, angle)"]
            lines += [f"  {when:8.1f} s  {math.degrees(angle):.0f}°" for when, angle in self.brakes]
        defaults = {f.name: getattr(SimConfig(), f.name) for f in fields(SimConfig) if f.name != "seed"}
        changed = [f"--set {name}={value_text(value)}" for name, value in values.items()
                   if name in defaults and value != defaults[name]]
        command = " ".join(["python -m roundabout_sim.experiments run", f"--seed {sim.seed}",
                            *([f"--duration {value_text(round(sim.time, 6))}"] if sim.time > 0 else []), *changed])
        lines += ["", "equivalent command: the final settings from t = 0 (settings not listed are at their",
                  "defaults; live changes and braked cars are not replayed)", f"  {command}", ""]
        return "\n".join(lines)

    # ------------------------------------------------------------ events
    def handle(self, e: pygame.event.Event) -> bool:
        """Process one event; False means quit."""
        if e.type == pygame.QUIT:
            return False
        if e.type == pygame.VIDEORESIZE:
            # Never smaller than MIN_WINDOW (unless the screen itself is smaller).
            desk = (pygame.display.get_desktop_sizes() or [MIN_WINDOW])[0]
            size = (max(e.size[0], min(MIN_WINDOW[0], desk[0])), max(e.size[1], min(MIN_WINDOW[1], desk[1])))
            if self.screen.get_size() != size:  # clamped, injected, or a driver that did not resize
                self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
            self.layout()
        elif e.type == pygame.KEYDOWN:
            return self._key(e.key)
        elif e.type == pygame.MOUSEMOTION:
            self.mouse = e.pos
            if self.drag:
                self._drag_to(e.pos[0])
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            self.mouse = e.pos
            self._press(e.pos)
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            self.mouse = e.pos
            self._release(e.pos)
        elif e.type == pygame.MOUSEWHEEL and self.content_rect.collidepoint(self.mouse):
            self.scroll -= int(e.y * 40 * self.s)
            self._layout_panel()
        return True

    def _key(self, key: int) -> bool:
        if key == pygame.K_ESCAPE:
            return False
        if key == pygame.K_SPACE:
            self.action("pause")
        elif key == pygame.K_r:
            self.action("reset")
        elif key == pygame.K_b:
            self.action("brake")
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.steps_per_frame = min(self.steps_per_frame + 1, int(VIEW_PARAMS["steps_per_frame"].hi))
        elif key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
            self.steps_per_frame = max(self.steps_per_frame - 1, int(VIEW_PARAMS["steps_per_frame"].lo))
        elif key == pygame.K_c:
            modes = VIEW_PARAMS["colour_mode"].choices
            self.colour_mode = modes[(modes.index(self.colour_mode) + 1) % len(modes)]
        elif key == pygame.K_s:
            self.screenshot()
        elif key == pygame.K_d:
            self.save_data()
        return True

    def _visible(self, w: Widget, pos) -> bool:
        return not w.scrolls or self.content_rect.collidepoint(pos)

    def widget_at(self, pos) -> Widget | None:
        return next((w for w in self.widgets if w.rect.collidepoint(pos) and self._visible(w, pos)), None)

    def _press(self, pos) -> None:
        w = self.widget_at(pos)
        if w is None:
            slot = self._car_at(pos)
            if slot is not None:
                self.sim.brake(slot)
                self._record_brake(slot)
            return
        if w.kind == "slider":
            if w.track.inflate(2 * round(8 * self.s), round(20 * self.s)).collidepoint(pos):
                start = self.value(w.name)
                self.drag = Drag(w, start, start)
                self._drag_to(pos[0])
        elif w.kind == "choice":
            hit = next((i for i, o in enumerate(w.options) if o.collidepoint(pos)), None)
            if hit is not None:
                self.pressed = (w, hit)
        else:
            self.pressed = (w, 0)

    def _drag_to(self, x: int) -> None:
        d = self.drag
        spec, track = d.widget.spec, d.widget.track
        t = float(np.clip((x - track.x) / track.w, 0.0, 1.0))
        value = spec.lo + round(t * (spec.hi - spec.lo) / spec.step) * spec.step
        value = round(float(np.clip(value, spec.lo, spec.hi)), 6)
        if value == d.value:
            return
        d.value = value
        if not spec.reset:  # rebuilding parameters wait for the release
            self.apply(spec.name, value)

    def _release(self, pos) -> None:
        if self.drag:
            d, self.drag = self.drag, None
            spec = d.widget.spec
            if spec.reset:
                self.apply(spec.name, d.value)
            else:
                self._mark(spec, d.start, d.value)
            return
        if self.pressed is None:
            return
        w, i = self.pressed
        self.pressed = None
        if w.kind == "button" and w.rect.collidepoint(pos):
            self.action(w.name)
        elif w.kind == "choice" and w.options[i].collidepoint(pos):
            old, new = self.value(w.name), w.spec.choices[i]
            self.apply(w.name, new)
            if not w.spec.reset:
                self._mark(w.spec, old, new)

    def _car_at(self, pos) -> int | None:
        ring = self.ring
        if ring is None or not self.view_rect.collidepoint(pos):
            return None
        x, y = ring.to_world(*pos)
        return self.sim.car_near(x, y, max_dist=max(self.sim.cfg.car_length, 8.0 / ring.k))

    # ------------------------------------------------------------ loop
    def run(self) -> None:
        running = True
        while running:
            start = time.perf_counter()
            for event in pygame.event.get():
                running = self.handle(event) and running
            if not running:
                break
            if self.screen.get_size() != self._layout_size:
                self.layout()
            if not self.paused:
                self.advance(self.steps_per_frame)
            self.draw()
            pygame.display.flip()
            self.work_ms.append(1000.0 * (time.perf_counter() - start))
            self.clock.tick(FPS_CAP)
        pygame.quit()

    # ------------------------------------------------------------ drawing
    def draw(self) -> None:
        self._follow_rebuilds()
        self.screen.fill(UI["surface"])
        row = self._latest_row()
        self._draw_view(row)
        self._draw_stats(row)
        self._draw_charts()
        self._draw_panel()

    def _card(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, UI["background"], rect, border_radius=round(6 * self.s))

    def _car_colours(self, snap: dict) -> np.ndarray:
        if self.colour_mode == "lane":
            return np.array(palette.SERIES)[snap["lane"] % len(palette.SERIES)]
        if self.colour_mode == "desired":
            # Temperament z sets the desired speed; with no spread every driver is the same.
            spread = 1.0 if self.sim.cfg.speed_std > 0 else 0.0
            return palette.speed_rgb(0.5 + spread * snap["z"] / 4.0).astype(int)
        return palette.speed_rgb(snap["v"] / np.maximum(snap["v0"], 0.1)).astype(int)

    def _draw_view(self, row: dict) -> None:
        sim, f = self.sim, self.fonts
        if self.ring is None:
            self.ring = RingView(sim, self.view_rect, f)
        snap = sim.cars()
        hovered = self._car_at(self.mouse) if not self.drag else None
        self.screen.set_clip(self.view_rect)
        self.ring.draw(self.screen, sim, snap, self._car_colours(snap), f, row, hovered)
        pad = round(10 * self.s)
        x, y = self.view_rect.x + pad, self.view_rect.y + pad
        f.blit(self.screen, f"{sim.n_lanes} lane{'s' if sim.n_lanes > 1 else ''}  ·  closed ring",
               (x, y), "small", UI["muted"])
        if self.paused:
            f.blit(self.screen, "PAUSED", (self.view_rect.right - pad, y), "base", UI["text"], "topright")
        if hovered is not None:
            self._car_tooltip(hovered)
        self.screen.set_clip(None)

    def _car_tooltip(self, slot: int) -> None:
        sim, f, s = self.sim, self.fonts, self.s
        v0 = float(sim.desired_speed(np.array([slot]))[0])
        lines = [f"car {sim.uid[slot]}  ·  lane {sim.lane[slot] + 1}",
                 f"{sim.v[slot]:.1f} m/s of {v0:.1f} desired"]
        if np.isfinite(sim.exit_at[slot]):
            moving_out = bool(sim.must_move_out(np.array([slot]))[0])
            lines.append(f"exits in {(sim.exit_at[slot] - sim.travel[slot]) * sim.r_out:.0f} m"
                         + ("  ·  moving out" if moving_out else ""))
        if sim.brake_until[slot] > sim.time:
            lines.append("braking")
        pad = round(6 * s)
        w = max(f.width(t, "small") for t in lines) + 2 * pad
        h = len(lines) * f.line("small") + 2 * pad
        box = pygame.Rect(self.mouse[0] + 14, self.mouse[1] + 14, w, h).clamp(self.view_rect)
        pygame.draw.rect(self.screen, UI["raised"], box, border_radius=4)
        pygame.draw.rect(self.screen, UI["grid"], box, 1, border_radius=4)
        for i, text in enumerate(lines):
            f.blit(self.screen, text, (box.x + pad, box.y + pad + i * f.line("small")), "small",
                   UI["text"] if i == 0 else UI["text2"])

    def _gradient(self, kind: str, w: int, h: int) -> pygame.Surface:
        key = (kind, w, h)
        if key not in self._gradients:
            t = np.linspace(0.0, 1.0, max(w, 2))
            rgb = palette.speed_rgb(t) if kind == "speed" else occupancy_rgb(t)
            self._gradients[key] = pygame.surfarray.make_surface(
                np.repeat(rgb[:, None, :], h, axis=1).astype(np.uint8))
        return self._gradients[key]

    def _legend(self) -> list[list[tuple]]:
        """Legend items for the view, each a list of parts drawn left to right."""
        T, T2 = UI["text2"], UI["muted"]
        gw = round(56 * self.s)
        if self.colour_mode == "speed":
            car = [("text", "Car speed / free speed:", T), ("text", "stopped", T2), ("grad", "speed", gw),
                   ("text", "free", T2)]
        elif self.colour_mode == "desired":
            car = [("text", "Driver's desired speed:", T), ("text", "slow", T2), ("grad", "speed", gw),
                   ("text", "fast", T2)]
        else:
            car = [("text", "Lane:", T)]
            for lane in range(self.sim.n_lanes):
                car += [("box", palette.SERIES[lane % len(palette.SERIES)]), ("text", str(lane + 1), T2)]
            car.append(("text", "(1 = inner)", T2))
        cfg = self.sim.cfg
        jam = 100 * cfg.car_length / (cfg.car_length + cfg.min_gap)
        return [car, [("ring", palette.CRITICAL), ("text", "braking", T)],
                [("text", "Inner band, lane covered:", T), ("text", "0", T2), ("grad", "seq", gw),
                 ("text", f"{jam:.0f}% (jam)", T2)],
                [("arrow", UI["text2"]), ("text", "bunching", T)]]

    def _parts_width(self, parts: list[tuple]) -> int:
        f, sp = self.fonts, round(5 * self.s)
        sizes = {"grad": lambda p: p[2], "text": lambda p: f.width(p[1], "small")}
        return sum(sizes.get(p[0], lambda p: round(12 * self.s))(p) + sp for p in parts) - sp

    def _draw_parts(self, parts: list[tuple], x: int, y: int) -> None:
        f, s, sp = self.fonts, self.s, round(5 * self.s)
        lh = f.line("small")
        cy = y + lh // 2
        for part in parts:
            kind = part[0]
            if kind == "text":
                x = f.blit(self.screen, part[1], (x, y), "small", part[2]).right
            elif kind == "grad":
                self.screen.blit(self._gradient(part[1], part[2], round(8 * s)), (x, cy - round(4 * s)))
                x += part[2]
            else:
                box = pygame.Rect(x, cy - round(5 * s), round(12 * s), round(10 * s))
                if kind in ("box", "empty"):
                    pygame.draw.rect(self.screen, part[1], box, border_radius=2)
                    if kind == "empty":
                        pygame.draw.rect(self.screen, UI["grid"], box, 1, border_radius=2)
                elif kind == "ring":
                    pygame.draw.rect(self.screen, part[1], box, 2, border_radius=2)
                else:  # arrow
                    pygame.draw.line(self.screen, part[1], (box.x, cy), (box.right - 4, cy), 2)
                    pygame.draw.polygon(self.screen, part[1], [(box.right, cy), (box.right - 5, cy - 4),
                                                               (box.right - 5, cy + 4)])
                x = box.right
            x += sp

    def _following_share(self) -> float:
        """Share of the jams sampled over the last RATE_WINDOW whose front car was just following
        (an upper bound on natural jams, see JAMS_HELP)."""
        h = self.hist
        t = h.array("time")
        if t.size == 0:
            return float("nan")
        recent = t >= t[-1] - metrics.RATE_WINDOW
        total = h.array("n_jams")[recent].sum()
        return float(h.array("jams_following")[recent].sum() / total) if total else float("nan")

    def _stat_columns(self, row: dict) -> list[list[Stat]]:
        """Readouts in three columns: traffic, congestion and space, events and lanes."""
        sim = self.sim
        v, free, trip = row["mean_speed"], row["mean_free_speed"], row["mean_trip_time"]
        following = self._following_share()
        share = f"{100 * following:.0f} %" if np.isfinite(following) else "–"
        order, base = row["order_param"], row.get("order_baseline", np.nan)
        base_text = f"{base:.2f}" if np.isfinite(base) else "–"
        traffic = [
            Stat("Sim time", f"{row['time']:.0f} s"),
            Stat("Cars on ring", f"{row['cars']} / {sim.target_cars()}"),
            Stat("Queued to enter", f"{row['queued']}"),
            Stat("Density", f"{row['density']:.1f} veh/km/lane"),
            Stat("Area covered", f"{100 * row['occupancy']:.1f} %"),
            Stat("Mean speed", f"{v:.1f} m/s · {3.6 * v:.0f} km/h"),
            Stat("Mean free speed", f"{free:.1f} m/s" if np.isfinite(free) else "–"),
            Stat("Ring flux", f"{row['flux']:.2f} veh/s"),
            Stat("Detector flux", f"{row['detector_flux']:.2f} veh/s"),
        ]
        congestion = [
            Stat("Jammed cars", f"{100 * row['jam_fraction']:.1f} %"),
            Stat("Speed deficit", f"{100 * row['speed_deficit']:.1f} %"),
            Stat("Stop-and-go", f"{row['stops_per_car_min']:.2f} stops/car/min",
                 short=("Stop-and-go", f"{row['stops_per_car_min']:.2f} /car/min")),
            Stat("Jams", f"{row['n_jams']}"),
            Stat("Following share of jams", share, short=("Following share", share)),
            Stat("Harsh braking (episodes/min)", f"{row['emergency_per_min']:.1f}",
                 short=("Harsh braking", f"{row['emergency_per_min']:.1f} /min")),
            Stat("Spacing (Gini)", f"{row.get('spacing_gini', np.nan):.2f}"),
            Stat("Clustering", f"{row['clustering']:.2f}"),
            Stat("Order", f"{order:.2f} · random {base_text}", short=("Order", f"{order:.2f} / {base_text}")),
        ]
        events = [
            Stat("Exits", f"{row['exits_per_min']:.1f} /min"),
            Stat("Lane changes", f"{row['lane_changes_per_min']:.1f} /min"),
            Stat("Missed exits", f"{row['missed_exits']}"),
            Stat("Mean trip time", f"{trip:.0f} s" if np.isfinite(trip) else "–"),
        ]
        for lane in range(sim.n_lanes):
            speed, cars = row.get(f"lane{lane}_speed", np.nan), row.get(f"lane{lane}_cars", 0)
            where = " inner" if lane == 0 and sim.n_lanes > 1 else " outer" if lane == sim.outer and lane else ""
            events.append(Stat(f"Lane {lane + 1}{where}",
                               f"{speed:.1f} m/s · {cars} cars" if np.isfinite(speed) else f"{cars} cars",
                               swatch=palette.SERIES[lane % len(palette.SERIES)], key="Lane"))
        return [traffic, congestion, events]

    def _draw_stats(self, row: dict) -> None:
        f, s, r = self.fonts, self.s, self.stats_rect
        self.stat_hits = []
        if r.h < 3 * f.line("small"):
            return
        self._card(r)
        pad = round(10 * s)
        x0, y = r.x + pad, r.y + round(8 * s)
        width = r.w - 2 * pad
        x, lh = x0, f.line("small")
        for parts in self._legend():  # flows onto a second line if needed
            w = self._parts_width(parts)
            if x > x0 and x + w > x0 + width:
                x, y = x0, y + lh
            self._draw_parts(parts, x, y)
            x += w + round(16 * s)
        y += lh + round(6 * s)
        pygame.draw.line(self.screen, UI["grid"], (x0, y - round(3 * s)), (x0 + width, y - round(3 * s)))
        row_h = lh + 1
        rows = max(1, (r.bottom - round(6 * s) - y) // row_h)
        columns = self._stat_columns(row)
        if max(len(c) for c in columns) > rows:  # too short for the grouping: flow into more columns
            flat = [st for c in columns for st in c]
            columns = [flat[i:i + rows] for i in range(0, len(flat), rows)]
        gap, inner, sw = round(20 * s), round(8 * s), round(8 * s)
        swatch_w = sw + round(4 * s)

        def need(col: list[Stat], short: bool) -> int:
            return max(swatch_w * (st.swatch is not None) + f.width(st.text(short)[0], "small") + inner
                       + f.width(st.text(short)[1], "small") for st in col)

        for short in (False, True):  # full labels if they fit, else the short ones
            needs = [need(c, short) for c in columns]
            slack = width - sum(needs) - gap * (len(columns) - 1)
            if slack >= 0:
                break
        if slack >= 0:  # columns sized to their content, spare room shared out
            col_ws = [n + slack / len(columns) for n in needs]
        else:
            col_ws = [(width - gap * (len(columns) - 1)) / len(columns)] * len(columns)
        cx0 = float(x0)
        for col, col_w in zip(columns, col_ws):
            right = round(cx0 + col_w)
            for k, st in enumerate(col):
                label, value = st.text(short)
                cx, cy = round(cx0), y + k * row_h
                self.stat_hits.append((pygame.Rect(cx, cy, right - cx, row_h), st.help_text()))
                if st.swatch:
                    pygame.draw.rect(self.screen, st.swatch, (cx, cy + (lh - sw) // 2, sw, sw), border_radius=2)
                    cx += swatch_w
                value_rect = f.blit(self.screen, value, (right, cy), "small", UI["text"], "topright")
                self.screen.set_clip(pygame.Rect(cx, cy, max(value_rect.x - inner // 2 - cx, 0), row_h))
                f.blit(self.screen, label, (cx, cy), "small", UI["text2"])
                self.screen.set_clip(None)
            cx0 += col_w + gap

    # ------------------------------------------------------------ charts
    def _time_axis(self) -> tuple[float, float]:
        t = self.hist.array("time")
        if t.size == 0:
            return self.sim.time, MIN_SPAN
        return float(t[-1]), float(np.clip(t[-1] - t[0], MIN_SPAN, HISTORY))

    def _draw_charts(self) -> None:
        t_end, span = self._time_axis()
        self.chart_hits = []
        for i, (spec, card, plot, two_lines, x_labels) in enumerate(self.chart_cards):
            self._line_chart(spec, card, plot, two_lines, x_labels, t_end, span, label_markers=i == 0)
        self._spacetime(t_end, span)

    def _x_ticks(self, plot: pygame.Rect, card: pygame.Rect, t_end: float, span: float,
                 labels: bool = True) -> None:
        """Time ticks below a plot; ``labels`` False keeps just the marks (the axis is shared)."""
        f = self.fonts
        step = next((st for st in (5, 10, 15, 30, 60, 120, 300)
                     if st / span * plot.w >= f.width("0000 s", "small") * 1.6), 600)
        last_right = -1
        for tick in np.arange(math.ceil((t_end - span) / step) * step, t_end + 1e-9, step):
            x = plot.right - (t_end - tick) / span * plot.w
            text = f"{tick:.0f} s"
            w = f.width(text, "small")
            if x - w / 2 < max(last_right + 4, plot.x - 2) or x + w / 2 > card.right:
                continue
            if labels:
                last_right = f.blit(self.screen, text, (x, plot.bottom + 1), "small", UI["muted"], "midtop").right
            else:
                last_right = x + w / 2
            pygame.draw.line(self.screen, UI["grid"], (x, plot.bottom), (x, plot.bottom + 2))

    @staticmethod
    def _time_x(plot: pygame.Rect, when: float, t_end: float, span: float) -> float:
        return plot.right - (t_end - when) / span * plot.w

    def _markers(self, plot: pygame.Rect, t_end: float, span: float) -> None:
        """Parameter-change markers as vertical lines (drawn under the data)."""
        for when, _ in self.markers:
            if when >= t_end - span:
                x = self._time_x(plot, when, t_end, span)
                pygame.draw.line(self.screen, UI["muted"], (x, plot.y), (x, plot.bottom - 1))

    def _marker_labels(self, plot: pygame.Rect, t_end: float, span: float) -> list[tuple[pygame.Rect, str]]:
        """Where the marker labels go. Newest first, each beside its own line (right of it, else
        left), on the first row (of two on tall plots) where it stays inside the plot, overlaps no
        label already placed and covers no newer marker's line (so every label reads as its own
        line's). A label without room is left out; hovering its line shows it."""
        f = self.fonts
        lh = f.line("small")
        rows = 2 if plot.h >= 4 * lh else 1
        taken: list[list[pygame.Rect]] = [[] for _ in range(rows)]
        placed, newer = [], []  # newer: x of the lines already seen
        for when, label in reversed(self.markers):
            if when < t_end - span:
                continue
            x = round(self._time_x(plot, when, t_end, span))
            w = f.width(label, "small")
            spot = next((box for row in range(rows) for box in (
                pygame.Rect(x + 3, plot.y + 2 + row * (lh + 2), w, lh),
                pygame.Rect(x - 3 - w, plot.y + 2 + row * (lh + 2), w, lh))
                if plot.contains(box)
                and not any(box.inflate(8, 0).colliderect(other) for other in taken[row])
                and not any(box.x - 2 <= nx <= box.right + 2 for nx in newer)), None)
            if spot is not None:
                taken[(spot.y - plot.y - 2) // (lh + 2)].append(spot)
                placed.append((spot, label))
            newer.append(x)
        return placed

    def _draw_marker_labels(self, plot: pygame.Rect, t_end: float, span: float) -> None:
        """Marker labels over the data, on a raised box so no series runs through the text."""
        for box, label in self._marker_labels(plot, t_end, span):
            pygame.draw.rect(self.screen, UI["raised"], box.inflate(4, 0))
            self.screen.blit(self.fonts.render(label, "small", UI["text2"]), box)

    def _marker_near(self, plot: pygame.Rect, x: float, t_end: float, span: float) -> str | None:
        """Label of the marker line nearest to screen column ``x`` (within a few px), newest on ties."""
        best = None
        for when, label in self.markers:
            if when < t_end - span:
                continue
            d = abs(self._time_x(plot, when, t_end, span) - x)
            if d <= max(6, round(10 * self.s)) and (best is None or d <= best[0]):
                best = (d, label)
        return best[1] if best else None

    def _hover_index(self, plot: pygame.Rect, t: np.ndarray, t_end: float, span: float) -> int | None:
        if t.size == 0 or not plot.collidepoint(self.mouse) or self.drag:
            return None
        th = t_end - (plot.right - self.mouse[0]) / plot.w * span
        i = int(np.clip(np.searchsorted(t, th), 0, t.size - 1))
        if i > 0 and abs(t[i - 1] - th) < abs(t[i] - th):
            i -= 1
        return i if abs(t[i] - th) <= 2 * SAMPLE_INTERVAL + span / plot.w else None

    @staticmethod
    def _extra_texts(kind: str, v) -> list[str]:
        """Header readouts for sample values ``v``, longest first: the longest that fits is shown."""
        if kind == "jams":
            n, fol = int(v["n_jams"]), int(v["jams_following"])
            if n == 0:
                return ["no jams"]
            merge, leave, enter, other = (int(v[k]) for k in ("jams_merge", "jams_exit", "jams_entry", "jams_other"))
            return [f"jams {n} — following {fol} · merge {merge} · exit {leave} · entry {enter} · other {other}",
                    f"jams {n} — following {fol} · arms {merge + leave + enter} · other {other}",
                    f"jams {n} · following {fol}", f"jams {n}"]
        order, base = float(v["order_param"]), float(v.get("order_baseline", np.nan))
        if not one_sided(order, base):  # same test as the arrow in the view
            return ["no more one-sided than random", "not one-sided"]
        angle = f"bunched at {math.degrees(v['order_angle']):.0f}°"
        return ([f"{angle} · {order / base:.1f}× random"] if np.isfinite(base) and base > 0 else []) + [angle]

    def _extra(self, spec: ChartSpec, i: int, room: int) -> str:
        """The longest header readout for sample ``i`` that fits in ``room`` px ("" if none)."""
        values = {key: self.hist.array(key)[i] for key in self.hist.KEYS}
        return next((t for t in self._extra_texts(spec.extra, values) if self.fonts.width(t, "small") <= room), "")

    @staticmethod
    def _series_colours(spec: ChartSpec) -> list[tuple]:
        """Palette colours in order for the series; a reference takes a faint tint of the series
        before it (it is that series' baseline) instead of a palette slot."""
        colours, k = [], 0
        for sr in spec.series:
            if sr.reference:
                colours.append(mix(colours[-1] if colours else UI["muted"], UI["background"], 0.4))
            else:
                colours.append(palette.SERIES[k])
                k += 1
        return colours

    def _dashed(self, xs: np.ndarray, ys: np.ndarray, colour, plot: pygame.Rect) -> None:
        """A faint dashed line through the finite points, resampled per pixel column so the
        dashes stay even however dense the samples are."""
        ok = np.isfinite(ys)
        if ok.sum() < 2:
            return
        px = np.arange(max(plot.x, math.ceil(xs[ok][0])), min(plot.right, math.floor(xs[ok][-1])) + 1)
        if px.size < 2:
            return
        py = np.interp(px, xs[ok], ys[ok])
        dash = max(3, round(5 * self.s))
        width = 1 if self.s < 1.3 else 2
        for a, b in finite_runs(((px - plot.x) // dash) % 2 == 0):
            if b - a > 1:
                pygame.draw.lines(self.screen, colour, False, np.column_stack((px[a:b], py[a:b])).tolist(), width)

    def _swatch(self, colour, x: int, cy: int, reference: bool) -> None:
        """Legend key: a short thick line, or a faint dashed one for a reference."""
        s = self.s
        if not reference:
            pygame.draw.line(self.screen, colour, (x, cy), (x + round(14 * s), cy), 3)
            return
        dash = round(14 * s) // 3
        for start in (0, 2 * dash):
            pygame.draw.line(self.screen, colour, (x + start, cy), (x + start + dash, cy), 2)

    def _line_chart(self, spec: ChartSpec, card: pygame.Rect, plot: pygame.Rect, two_lines: bool,
                    x_labels: bool, t_end: float, span: float, label_markers: bool) -> None:
        f, s, scr = self.fonts, self.s, self.screen
        self._card(card)
        t = self.hist.array("time")
        colours = self._series_colours(spec)
        data = [self.hist.array(sr.key) * sr.scale for sr in spec.series]
        data = [np.where(t < RATE_WARMUP, np.nan, d) if sr.key in ROLLING else d
                for sr, d in zip(spec.series, data)]
        peak = max((float(np.nanmax(d)) for d in data if np.isfinite(d).any()), default=0.0)
        need = max(peak * 1.05, spec.min_top)
        if spec.max_top is not None and peak <= spec.max_top:
            need = min(need, spec.max_top)
        # As many intervals (up to 3) as leave the tick labels apart (digits are ~half a line tall).
        target = int(np.clip(plot.h // (0.9 * f.line("small")), 1, 3))
        step = nice_step(need, target)
        top = step * math.ceil(need / step - 1e-9)
        ydec = max(0, decimals(step))
        for tick in np.arange(0, top + step / 2, step):
            y = plot.bottom - tick / top * plot.h
            pygame.draw.line(scr, UI["grid"], (plot.x, y), (plot.right, y))
            f.blit(scr, f"{tick:.{ydec}f}", (plot.x - 4, y), "small", UI["muted"], "midright")
        self._x_ticks(plot, card, t_end, span, labels=x_labels)
        self._markers(plot, t_end, span)
        hover = self._hover_index(plot, t, t_end, span)
        xs = plot.right - (t_end - t) / span * plot.w
        scr.set_clip(plot.inflate(2, 4))
        for sr, colour, d in sorted(zip(spec.series, colours, data), key=lambda c: not c[0].reference):
            ys = plot.bottom - d / top * plot.h  # references first, so the data draw over them
            if sr.reference:
                self._dashed(xs, ys, colour, plot)
                continue
            for a, b in finite_runs(np.isfinite(ys)):  # gaps where a value is undefined (e.g. empty ring)
                if b - a > 1:
                    pygame.draw.lines(scr, colour, False, np.column_stack((xs[a:b], ys[a:b])).tolist(), 2)
                else:
                    pygame.draw.circle(scr, colour, (xs[a], ys[a]), 1)
            i = t.size - 1 if hover is None else hover
            if t.size and np.isfinite(ys[i]):
                pygame.draw.circle(scr, UI["background"], (xs[i], ys[i]), 5)
                pygame.draw.circle(scr, colour, (xs[i], ys[i]), 3)
        scr.set_clip(None)
        if label_markers:
            self._draw_marker_labels(plot, t_end, span)
        if hover is not None:
            x = xs[hover]
            pygame.draw.line(scr, UI["text2"], (x, plot.y), (x, plot.bottom))
            text = f"{t[hover]:.0f} s"
            marker = self._marker_near(plot, self.mouse[0], t_end, span)
            if marker:
                text += f"  ·  {marker}"
            label = f.render(text, "small", UI["text"])
            if x_labels:
                box = label.get_rect(midtop=(x, plot.bottom + 1)).clamp(card)
            else:  # no axis band below: inside the plot, along its bottom
                box = label.get_rect(midbottom=(x, plot.bottom - 1)).clamp(plot)
            pygame.draw.rect(scr, UI["raised"], box.inflate(6, 0))
            scr.blit(label, box)

        # Header: title, then the value (one series) or a legend with values (on a second line
        # if it does not fit), and the extra readout on the right of whichever line has more room.
        i = (t.size - 1) if hover is None else hover
        x, y = card.x + round(8 * s), card.y + round(8 * s)
        right = card.right - round(8 * s)
        x = f.blit(scr, spec.title, (x, y), "base", UI["text"]).right + round(14 * s)
        lines = [(x, y)]  # where each header line's free room starts
        if len(spec.series) == 1:
            if t.size:
                value = spec.fmt.format(data[0][i]) if np.isfinite(data[0][i]) else "–"
                lines = [(f.blit(scr, value, (x, y), "base", UI["text"]).right, y)]
        else:
            lx, ly = (card.x + round(8 * s), y + f.line("base")) if two_lines else (x, y)
            cy = ly + f.line("base") // 2
            for colour, sr, d in zip(colours, spec.series, data):
                self._swatch(colour, lx, cy, sr.reference)
                lx = f.blit(scr, sr.label, (lx + round(18 * s), cy), "small", UI["text2"], "midleft").right
                if t.size:
                    value = spec.fmt.format(d[i]) if np.isfinite(d[i]) else "–"
                    lx = f.blit(scr, value, (lx + round(5 * s), cy), "small", UI["text"], "midleft").right
                lx += round(14 * s)
            lines = [(x, y), (lx - round(14 * s), ly)] if two_lines else [(lx - round(14 * s), y)]
        if spec.extra and t.size:
            start, ly = max(lines, key=lambda line: right - line[0])
            text = self._extra(spec, i, right - start - round(16 * s))
            if text:
                rect = f.blit(scr, text, (right, ly + f.line("base") // 2), "small", UI["text2"], "midright")
                self.chart_hits.append((rect.inflate(6, 4), EXTRA_HELP[spec.extra]))

    def _spacetime_surface(self, plot: pygame.Rect, span: float) -> pygame.Surface | None:
        h = self.hist
        key = (h.version, self.spacetime_mode, plot.size)
        now = time.monotonic()
        cached = self._st_cache
        # At high sim speeds a sample arrives every frame; rebuilding ~10 times a second is plenty.
        if cached and (cached[0] == key or (cached[0][1:] == key[1:] and now - cached[2] < SPACETIME_REFRESH)):
            return cached[1]
        surf = None
        if len(h):
            if self.spacetime_mode == "speed":
                rgb = np.stack(h.speed_rgb)
            else:
                cfg = self.sim.cfg
                jam_cover = cfg.car_length / (cfg.car_length + cfg.min_gap)
                rgb = occupancy_rgb(np.stack(h.occupancy)[:, ::-1] / jam_cover).astype(np.uint8)
            img = pygame.surfarray.make_surface(rgb)  # x = sample, y = angle bin from the top
            width = int(np.clip(round(len(h) * SAMPLE_INTERVAL / span * plot.w), 1, plot.w))
            surf = pygame.transform.smoothscale(img, (width, plot.h))
        self._st_cache = (key, surf, now)
        return surf

    def _spacetime(self, t_end: float, span: float) -> None:
        f, s, scr = self.fonts, self.s, self.screen
        card, plot = self.st_card, self.st_plot
        self._card(card)
        scr.fill(EMPTY, plot)
        surf = self._spacetime_surface(plot, span)
        if surf:
            scr.blit(surf, (plot.right - surf.get_width(), plot.y))
        for deg in (0, 90, 180, 270, 360):
            y = plot.bottom - deg / 360 * plot.h
            f.blit(scr, f"{deg}°", (plot.x - 6, y), "small", UI["muted"], "midright")
            pygame.draw.line(scr, UI["grid"], (plot.x - 4, y), (plot.x - 1, y))
        for a in self.sim.arm_angle:  # arms as faint ticks on both edges
            y = plot.bottom - a / TWO_PI * plot.h
            pygame.draw.line(scr, UI["road_edge"], (plot.x, y), (plot.x + 5, y), 2)
            pygame.draw.line(scr, UI["road_edge"], (plot.right - 5, y), (plot.right, y), 2)
        axis = pygame.transform.rotate(f.render("angle, driving direction →", "small", UI["muted"]), 90)
        if axis.get_height() > plot.h:
            axis = pygame.transform.rotate(f.render("angle", "small", UI["muted"]), 90)
        scr.blit(axis, axis.get_rect(midleft=(card.x + round(6 * s), plot.centery)))
        self._x_ticks(plot, card, t_end, span)
        self._markers(plot, t_end, span)
        for when, angle in self.brakes:
            if when >= t_end - span:
                pos = (plot.right - (t_end - when) / span * plot.w, plot.bottom - angle / TWO_PI * plot.h)
                pygame.draw.circle(scr, UI["background"], pos, round(5 * s) + 2, 2)  # halo to lift it off the data
                pygame.draw.circle(scr, palette.CRITICAL, pos, round(5 * s), 2)

        x, y = card.x + round(8 * s), card.y + round(8 * s)
        f.blit(scr, "Space-time diagram", (x, y), "base", UI["text"])
        self._draw_choice(self.chart_widgets[0])
        readout = self._spacetime_hover(plot, t_end, span)
        right = card.right - round(8 * s)
        cy = y + f.line("base") // 2
        if readout:
            f.blit(scr, readout, (right, cy), "small", UI["text"], "midright")
        else:
            cfg = self.sim.cfg
            if self.spacetime_mode == "speed":
                parts = [("text", "stopped", UI["muted"]), ("grad", "speed", round(56 * s)),
                         ("text", "free", UI["muted"]), ("empty", EMPTY), ("text", "no cars", UI["muted"])]
            else:
                jam = 100 * cfg.car_length / (cfg.car_length + cfg.min_gap)
                parts = [("text", "0", UI["muted"]), ("grad", "seq", round(56 * s)),
                         ("text", f"{jam:.0f}% covered", UI["muted"])]
            w = self._parts_width(parts)
            if right - w > self.chart_widgets[0].rect.right + round(10 * s):
                self._draw_parts(parts, right - w, cy - f.line("small") // 2)

    def _spacetime_hover(self, plot: pygame.Rect, t_end: float, span: float) -> str:
        """Crosshair under the mouse; returns the readout for that time and angle (or "")."""
        h = self.hist
        i = self._hover_index(plot, h.array("time"), t_end, span)
        if i is None:
            return ""
        x, y = self.mouse
        pygame.draw.line(self.screen, UI["text2"], (x, plot.y), (x, plot.bottom))
        pygame.draw.line(self.screen, UI["text2"], (plot.x, y), (plot.right, y))
        frac = float(np.clip((plot.bottom - y) / plot.h, 0.0, 0.9999))
        if self.spacetime_mode == "speed":
            v = h.profile[i][int(frac * len(h.profile[i]))]
            value = f"{v:.1f} m/s" if np.isfinite(v) else "no cars"
        else:
            value = f"{100 * h.occupancy[i][int(frac * len(h.occupancy[i]))]:.0f}% covered"
        return f"{h.array('time')[i]:.0f} s  ·  {360 * frac:.0f}°  ·  {value}"

    # ------------------------------------------------------------ panel
    def _draw_panel(self) -> None:
        f, s, scr, r = self.fonts, self.s, self.screen, self.panel_rect
        pygame.draw.rect(scr, UI["background"], r)
        pygame.draw.line(scr, UI["grid"], r.topleft, r.bottomleft)
        p = round(14 * s)
        x, y = r.x + p, r.y + p
        f.blit(scr, "Roundabout traffic", (x, y), "title", UI["text"])
        fps = self.clock.get_fps()
        speed = fps * self.steps_per_frame * self.sim.cfg.dt
        info = f"seed {self.sim.seed}  ·  {fps:.0f} FPS"
        if fps > 0 and not self.paused:
            info += f"  ·  {speed:.0f}× real time"
        f.blit(scr, info, (x, y + f.line("title")), "small", UI["muted"])
        for w in self.panel_widgets:
            if w.scrolls:
                continue
            if w.kind == "button":
                self._draw_button(w)
            else:
                self._draw_control(w)
        scr.set_clip(self.content_rect)
        for w in self.panel_widgets:
            if w.scrolls and w.rect.colliderect(self.content_rect):
                self._draw_control(w)
        scr.set_clip(None)
        overflow = self.content_h - self.content_rect.h
        if overflow > 0:  # scroll bar
            c = self.content_rect
            bar_h = max(20, c.h * c.h // self.content_h)
            bar_y = c.y + (c.h - bar_h) * self.scroll // overflow
            pygame.draw.rect(scr, UI["grid"], (c.right - round(6 * s), bar_y, round(3 * s), bar_h), border_radius=2)
        self._draw_status()

    def _draw_button(self, w: Widget) -> None:
        f, scr = self.fonts, self.screen
        hot = w.rect.collidepoint(self.mouse)
        active = w.name == "pause" and self.paused
        fill = UI["accent"] if active else UI["grid"] if hot else UI["raised"]
        pygame.draw.rect(scr, fill, w.rect, border_radius=round(4 * self.s))
        label = next(lbl for name, lbl, _ in BUTTONS if name == w.name)
        if w.name == "pause" and self.paused:
            label = "Play"
        f.blit(scr, label, w.rect.center, "small", UI["text"], "center")

    def _draw_control(self, w: Widget) -> None:
        f, s, scr, spec = self.fonts, self.s, self.screen, w.spec
        x, y = w.rect.topleft
        label = f.blit(scr, spec.label, (x, y), "base", UI["text2"])
        if spec.reset:
            f.blit(scr, "rebuilds", (label.right + round(6 * s), label.centery), "small", UI["muted"], "midleft")
        value = self.value(spec.name)
        if spec.choices:
            self._draw_choice(w)
            return
        unit = f" {spec.unit}" if spec.unit else ""
        f.blit(scr, f"{format_value(spec, value)}{unit}", (w.rect.right, y), "base", UI["text"], "topright")
        track = w.track
        frac = float(np.clip((value - spec.lo) / (spec.hi - spec.lo), 0.0, 1.0))
        knob_x = track.x + frac * track.w
        pygame.draw.rect(scr, UI["grid"], track, border_radius=2)
        pygame.draw.rect(scr, UI["accent"], (track.x, track.y, max(int(frac * track.w), 1), track.h), border_radius=2)
        hot = (self.drag and self.drag.widget is w) or w.rect.collidepoint(self.mouse)
        pygame.draw.circle(scr, UI["text"], (knob_x, track.centery), round((8 if hot else 6.5) * s))

    def _draw_choice(self, w: Widget) -> None:
        f, scr = self.fonts, self.screen
        current = self.value(w.name)
        for option, rect in zip(w.spec.choices, w.options):
            selected = option == current
            hot = rect.collidepoint(self.mouse)
            fill = UI["accent"] if selected else UI["grid"] if hot else UI["raised"]
            pygame.draw.rect(scr, fill, rect, border_radius=round(4 * self.s))
            f.blit(scr, option, rect.center, "small", UI["text"] if selected else UI["text2"], "center")

    def _status_text(self) -> str:
        w = self.drag.widget if self.drag else self.widget_at(self.mouse)
        if w is not None:
            if w.spec is None:
                return w.help
            return control_help(w.spec)
        for rect, help_text in self.chart_hits:  # a header readout has its own help
            if rect.collidepoint(self.mouse):
                return help_text
        for spec, card, *_ in self.chart_cards:
            if card.collidepoint(self.mouse):
                return spec.help
        if self.st_card.collidepoint(self.mouse):
            return SPACETIME_HELP
        if self.view_rect.collidepoint(self.mouse):
            return VIEW_HELP
        for rect, help_text in self.stat_hits:
            if rect.collidepoint(self.mouse):
                return help_text
        text, until = self.message
        return text if time.monotonic() < until else DEFAULT_HINT

    def _draw_status(self) -> None:
        f, r = self.fonts, self.status_rect
        pygame.draw.line(self.screen, UI["grid"], r.topleft, r.topright)
        y = r.y + round(6 * self.s)
        lines = f.wrap(self._status_text(), r.w)
        n = self.status_lines
        if len(lines) > n:  # only a long message (a saved file's path) can overflow
            lines = lines[:n - 1] + [lines[n - 1] + " …"]
        for line in lines:
            f.blit(self.screen, line, (r.x, y), "small", UI["text2"])
            y += f.line("small")


# ------------------------------------------------------------------ main
def _dpi_aware() -> None:
    """Render at native resolution on high-DPI Windows displays instead of being upscaled blurrily."""
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass


def _initial_size() -> tuple[int, int]:
    """About 90 % of the desktop, but at least ``MIN_WINDOW`` where the screen allows."""
    sizes = pygame.display.get_desktop_sizes()
    w, h = sizes[0] if sizes else (MIN_WINDOW[0], MIN_WINDOW[1])
    return (int(max(0.9 * w, min(MIN_WINDOW[0], w))), int(max(0.9 * h, min(MIN_WINDOW[1], h))))


def _window_size(text: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, e.g. 1600x900") from None
    return w, h


def randomized_config(seed: int | None = None, *, density: float | None = None,
                      driver_noise: float | None = None) -> SimConfig:
    """Choose model settings once; hidden road topology and car dimensions stay fixed."""
    rng = random.Random(seed)
    changes = {"n_lanes": 1, "n_arms": 0}
    for spec in PARAMS:
        if spec.name in {"n_lanes", "n_arms", "density", "driver_noise", *FIXED_PARAMS}:
            continue
        if spec.choices:
            changes[spec.name] = rng.choice(spec.choices)
        else:
            hi = {"radius": 60, "speed_std": 2}.get(spec.name, spec.hi)
            steps = round((hi - spec.lo) / spec.step)
            changes[spec.name] = round(spec.lo + rng.randrange(steps + 1) * spec.step, 6)
    if density is not None:
        changes["density"] = density
    if driver_noise is not None:
        changes["driver_noise"] = driver_noise
    return SimConfig(seed=seed).updated(**changes)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m roundabout_sim", description="Interactive single-lane closed-ring traffic model.",
        epilog=f"Keys: Space pause, R reset (new seed), B brake a random car, +/- sim speed, C car colours, "
               f"S save a screenshot, D save the chart data ({HISTORY_TEXT}) as CSV with a settings file "
               "(settings, live changes, equivalent experiments command), Esc quit. Files go to results/. "
               "Click a car to make it brake; hover anything for help.")
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: random)")
    parser.add_argument("--size", type=_window_size, default=None, metavar="WxH",
                        help="window size in pixels (default: 90%% of the screen)")
    defaults = SimConfig()
    for spec in CONTROL_PARAMS:
        flags = [f"--{spec.name}"] + ([f"--{spec.name.replace('_', '-')}"] if "_" in spec.name else [])
        default = getattr(defaults, spec.name)
        text = f"{spec.label}: {spec.help} (default {default})".replace("%", "%%")
        if spec.choices:
            parser.add_argument(*flags, dest=spec.name, choices=spec.choices, help=text)
        else:
            unit = f" {spec.unit}" if spec.unit else ""
            parser.add_argument(*flags, dest=spec.name, type=float, metavar=f"{spec.lo:g}-{spec.hi:g}{unit}",
                                help=text)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> SimConfig:
    changes = {spec.name: coerce(spec.name, getattr(args, spec.name))
               for spec in CONTROL_PARAMS if getattr(args, spec.name) is not None}
    return randomized_config(args.seed).updated(**changes)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    App(config_from_args(args), size=args.size).run()
