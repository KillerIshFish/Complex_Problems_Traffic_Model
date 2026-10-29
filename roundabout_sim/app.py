"""Interactive viewer for the roundabout traffic model.

Run ``python -m roundabout_sim`` (``--help`` lists one flag per model
parameter, e.g. ``--density 80 --n_lanes 4``). The left view draws the ring
at true scale; the panel on the right changes parameters while the model
runs; the live charts (the last 10 minutes) track speed, congestion over
time, stop-and-go, flux and how the cars congregate (in open traffic mode
also the cars on the ring, the queues and the entry delay), and the
space-time diagram shows phantom jams as stripes running backwards against
the traffic.

Mouse: click a car to make it brake, drag sliders, hover anything (charts,
readouts, controls) for help, scroll the parameter list with the wheel.
Keys: Space pause, R reset (new seed), B brake a random car, +/- sim speed,
C cycle car colours, S save a screenshot to results/, D save the chart data
(CSV, plus the settings, the live changes and an equivalent experiments
command), F5 save the state (results/state_<time>.npz), F9 carry on from the
most recent one, X random settings (Shift+X: only the tab shown), V open the
sweep viewer with the current settings, Esc quit.

Presets: the Presets tab starts real-world settings (:mod:`roundabout_sim.presets`:
ring-road experiments, design-guide roundabouts, the Place Charles de Gaulle)
and shows each one's description, notes and sources; ``--preset NAME`` and
``--random SEED`` start with a preset or random settings, ``--list-presets``
lists the presets.

Sessions: closing the window (or Esc) saves the simulation, its charts and the
view (tab, scroll, sim speed, colour modes, paused) to results/session/, and
the next launch carries on there. ``--fresh`` ignores the saved session,
``--no-autosave`` does not save one, ``--state PATH`` carries on from a saved
state file, and model settings given as flags (``--preset`` and ``--random``
too) start a new simulation with them (the view settings still restore). See :mod:`roundabout_sim.state` for
the file format.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import pygame

from roundabout_sim import cli, metrics, palette, presets, state
from roundabout_sim.config import (NEEDS_NONZERO, PARAM_BY_NAME, PARAMS, ParamSpec, SimConfig, coerce,
                                   inactive_reason, same_value)
from roundabout_sim.engine import (AUTO_KINDS, HUMAN, KIND_NAMES, MAX_DECEL, MAX_QUEUE, RESUME_FRACTION,
                                   SLOW_FRACTION, SUBJECT_ARM, TRUCK_ACCEL_FACTOR, TRUCK_HEADWAY_FACTOR,
                                   TRUCK_SPEED_FACTOR, TRUCK_WIDTH, TWO_PI, RingSim, StateError)
from roundabout_sim.textfmt import (DISPLAY_DIGITS, decimals, format_number, plural,  # noqa: F401 (also app's API)
                                    value_text)
from roundabout_sim.ui import (FONT_FACE, MIN_WINDOW, STATUS_LINES, Fonts, Messages, clamp_resize,  # noqa: F401
                               dpi_aware, draw_tab, ellipsize, finite_runs, nice_step, open_window, row_rects,
                               status_lines, tick_text, wrap_lines)

FPS_CAP = 60
SAMPLE_INTERVAL = 0.5  # [s] of simulated time between chart samples
SPACETIME_REFRESH = 0.1  # [s] of wall time between rebuilds of the space-time image
HISTORY = 600.0  # [s] of chart history kept (10 minutes)
MIN_SPAN = 60.0  # [s] shortest time axis on the charts
RATE_WARMUP = 10.0  # [s] rolling rates (over metrics.RATE_WINDOW) are too noisy to chart before this
ROLLING = ("detector_flux", "stops_per_car_min")  # chart series that are rolling rates
SPACETIME_SHARE = 0.34  # of the chart column's height
QUEUE_MIN_PX = 4.0  # [px] of stub per queued car below which a queue cannot be counted by eye (gets a tag)
BEND_POINTS = 5  # points along each side of a truck drawn bent along its lane (22 m on a 20 m ring: ~0.2 m off)
# Saved sessions: <results>/session/ holds the full state and a readable summary (settings and view).
SESSION_DIR, SESSION_STATE, SESSION_FILE = "session", "state.npz", "session.json"
SESSION_FORMAT = "roundabout_sim session"
STATE_FILES = "state_*.npz"  # states saved with F5, in the results folder (F9 loads the most recent)
STATE_NAME = re.compile(r"state_(\d{8}_\d{6})(?:-(\d+))?\.npz")  # their names: timestamp, counter
SWEEP_LOG = "sweep_viewer.log"  # in the results folder: what a sweep viewer started with V writes to stderr
CONFIG_FIELDS = tuple(f.name for f in fields(SimConfig) if f.name != "seed")
WINDOW = f"{metrics.RATE_WINDOW:g} s"
UI = palette.UI
EMPTY = UI["surface"]  # space-time bins without cars
GROUPS = tuple(dict.fromkeys(p.group for p in PARAMS))
PRESETS_TAB = "Presets"  # the tab of the real-world presets and random settings (no parameter group)
TABS = (PRESETS_TAB, *GROUPS)

# Viewer settings use the same spec type as model parameters, so they get the
# same widgets; group "View" keeps them out of the model's tabs.
VIEW_PARAMS = {p.name: p for p in (
    ParamSpec("steps_per_frame", "Sim speed, steps per frame", 1, 20, 1, "", "View",
              "Simulation steps (of dt each) per drawn frame. Keys + and -."),
    ParamSpec("colour_mode", "Car colour", 0, 0, 0, "", "View",
              "speed: own speed / free speed; desired: driver's desired speed (slow to fast); "
              "lane: which lane; type: car or truck; kind: human driver or automated (acc, cav, robotaxi). Key C.",
              choices=("speed", "desired", "lane", "type", "kind")),
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
    ("save_state", "Save state", "Save the whole simulation with its charts and markers into the results folder "
                                 "(state_<date>_<time>.npz), to carry on from later: Load state or F9 picks the "
                                 "most recent (F5)."),
    ("load_state", "Load state", "Carry on from the most recently saved state in the results folder (state_*.npz): "
                                 "its simulation, charts and markers; the view stays as it is (F9). A damaged or "
                                 "incompatible file restarts with the settings it holds."),
    ("sweep", "Sweep viewer", "Open the sweep viewer in a window of its own, with the current settings as the base "
                              "of its new sweeps (saved as sweep_base_<date>_<time>.json in the results folder), to "
                              "run and browse sweeps of 1-3 settings (V)."),
    # On the Presets tab:
    ("apply_preset", "Apply preset", "Start the preset chosen above: the model defaults with its settings (taken from "
                                     "a published experiment, a design guide or traffic counts), keeping the seed. "
                                     "Rebuilds the roundabout and clears the charts; the tabs then change it as usual. "
                                     "On the command line: --preset NAME."),
    ("randomize", "Randomize all (X)", "Give every setting a plausible random value (the AVs and Coop tabs only with "
                                       "Shift+X there): real-world ranges, linked as on real roundabouts (the radius "
                                       "follows the lanes, the demand the capacity, ...). "
                                       "Each press draws a fresh seed, shown in the help box and kept with the "
                                       "settings: --random SEED starts with the same ones (X). Shift+X randomizes only "
                                       "the settings of the tab shown."),
)
BUTTON_ROWS = (("pause", "step", "reset", "brake"), ("clear", "screenshot", "csv"),
               ("save_state", "load_state", "sweep"))
# The button at the end of each parameter tab (Shift+X).
RANDOM_TAB_HELP = ("Give the {group} settings plausible random values, linked to the current settings of the other "
                   "tabs, with a fresh seed shown in the help box (Shift+X). Randomize all (X) draws every setting "
                   "but those of the AVs and Coop tabs.")
PRESET_HELP = ("Preset {i} of {n}: {title}. {description}{source} Click the title or the arrows for another; Apply "
               "preset starts it (details below).")
PRESET_TEXT_HELP = ("{title}: what it is, its settings (by parameter name, as --set takes them), the notes (caveats, "
                    "and what check runs of this model give) and the sources. The wheel scrolls the list.")
ORIGIN_HELP = ("Where the running settings come from: the preset applied or the random settings drawn (with their "
               "seed), and which settings were changed since. Saved with the settings (D) and the session.")

DEFAULT_HINT = ("Click a car to make it brake.  Space pause  ·  R reset  ·  B brake  ·  "
                "+/- speed  ·  C colours  ·  S screenshot  ·  D data (CSV)  ·  F5 save state  ·  "
                "F9 load state  ·  X random settings (Shift+X: this tab)  ·  V sweep viewer  ·  Esc quit")
AUTOSAVE_HINT = f"{DEFAULT_HINT} (saves the session, which the next launch resumes)"

# Tooltips of the parameter tabs (groups from config.PARAMS; others get a generic one).
TAB_HELP = {
    "Road": "Show the road parameters: the layout (changing it rebuilds the roundabout) and the road condition.",
    "Traffic": "Show the traffic parameters: density, exits and entering.",
    "Demand": "Show the demand parameters: a closed ring of cars or open arrivals at the entrances, how many "
              "arrive and when, how long the queues can grow, and the capacity test (a saturated test entrance).",
    "Vehicles": "Show the vehicle parameters: the share and length of trucks (changing them rebuilds the "
                "roundabout).",
    "Drivers": "Show the driver parameters: speeds, reactions, following and noise, and how much drivers differ.",
    "Lane chg": "Show the lane-changing parameters.",
    "Start": "Show the initial-state parameters: how the cars are spaced at the start.",
    "AVs": "Show the automated-vehicle parameters: their share of the new vehicles (changing it rebuilds the "
           "roundabout), their kind and the robotaxis' time gap.",
    "Coop": "Show the cooperation parameters: how many companies the automated vehicles belong to.",
    PRESETS_TAB: "Show the presets: settings of real roundabouts and ring-road experiments, with their sources and "
                 "notes, and random settings.",
}


def tab_help(group: str) -> str:
    return TAB_HELP.get(group, f"Show the {group.lower()} parameters.")


TYPE_COLOURS = (palette.SERIES[0], palette.SERIES[1])  # car, truck ("type" car colours)
# "kind" car colours, by engine kind code: human drivers grey (palette.MID, the speed scale's middle: they recede),
# each automated kind a categorical colour of its own (never those of "type", so the modes do not read alike).
KIND_COLOURS = (palette.MID, palette.SERIES[3], palette.SERIES[2], palette.SERIES[4], palette.SERIES[6])
TRUCK_TEXT = (f"Trucks are drawn to scale, {TRUCK_WIDTH:g} m wide; they accelerate at {TRUCK_ACCEL_FACTOR:g}× a car's "
              f"rate, want {TRUCK_SPEED_FACTOR:g}× their driver's desired speed and keep a "
              f"{TRUCK_HEADWAY_FACTOR:g}× longer time gap.")


def control_help(spec: ParamSpec) -> str:
    """Help shown for a slider or choice."""
    note = ""
    if spec.reset:
        other = NEEDS_NONZERO.get(spec.name)
        when = f" (while {PARAM_BY_NAME[other[0]].label} is above 0)" if other else ""
        note = f" Changing it rebuilds the roundabout and clears the charts{when}."
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
SPACING_HELP = ("Gini coefficient of the road each car has (gap ahead + the mean vehicle length, so a mix of cars "
                "and trucks alone does not raise it): 0 = shared evenly, about "
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
LOS_TEXT = metrics.los_text()
ENTRY_DELAY_HELP = (f"Mean queue wait of the cars that entered over the last {WINDOW} (joining the queue to "
                    "entering the ring; not the full HCM control delay), with an HCM-style level of service: "
                    f"{LOS_TEXT}, and in open mode F whenever an entrance got more arrivals than it could let in "
                    f"over the last {metrics.VC_WINDOW / 60:g} min (v/c > 1).")
DELAY_HELP = (f"Entry delay: {ENTRY_DELAY_HELP} v/c: demand / capacity of the busiest entrance (its arrivals "
              "over its entry rate while it had a queue). Longest queue: cars at the longest one now.")
EXTRA_HELP = {"jams": JAMS_HELP, "bunching": ORDER_HELP, "delay": DELAY_HELP}  # hover help of the header readouts

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
              "the view then points there).", extra="bunching", max_top=1.0),
)
# Extra charts in open mode, where the number of cars varies.
OPEN_CHARTS = (
    ChartSpec("Cars and queues [veh]", (Series("cars", "on ring"), Series("queued", "queued")), "{:.0f}", 10.0,
              "Open mode: cars on the ring and cars waiting at all the entrances. Arrivals add cars and exits "
              "take them away, so the number on the ring drifts with the demand; the queues grow when the "
              "entries cannot keep up. The header gives the entry delay and its level of service (hover it).",
              extra="delay"),
)
SPACETIME_HELP = ("Space-time diagram: time runs right, angle around the ring up (driving direction). "
                  "Cars trace rising lines; jams are stripes sloping down as they travel backwards. "
                  "Speed: mean speed / the cars' mean free speed. Ticks mark the arms; red rings mark "
                  "braked cars.")
VIEW_HELP = ("Click a car to make it brake hard for 3 s; hover one for its details. The inner band shades each "
             "sector by how much lane it has covered; the arrow points to where the cars bunch when they are "
             "clearly more one-sided than random (longer = more one-sided). Queued cars wait on the entry stubs "
             "(hover a queue's count for more). Trucks are drawn longer and wider. In open mode the header "
             "gives the demand per entrance in veh/h.")
QUEUE_TAG_HELP = ("Queue: cars waiting at this entrance. The stub shows only as many as fit (on a big ring they "
                  "are too small to count), so longer queues get this count. Red and marked full: the queue has "
                  "reached Max queue and turns further arrivals away (Blocked). Marked test: the capacity test's "
                  "entrance, its queue kept full whatever Max queue.")

STAT_HELP = {
    "Sim time": "Simulated time since the last reset or rebuild.",
    "Cars on ring": "Cars on the ring now / the number the density setting asks for.",
    "Cars on ring (open)": "Cars on the ring now. In open mode the number varies: cars arrive at random "
                           "(Entrance chance) and leave for good (Exit chance); Density set only the start.",
    "Queued to enter": f"Cars waiting at the entrances (at most {MAX_QUEUE} each; further replacements for "
                       "leavers are slotted straight into the ring).",
    "Queued to enter (open)": "Cars waiting at all the entrances now, and (max) at the longest queue (at most Max "
                              "queue each; arrivals that find it full are turned away as blocked). The capacity "
                              "test's standing queue is left out.",
    "Density": "Cars on the ring per km of lane.",
    "Area covered": "Share of the road surface (all lanes) covered by vehicles.",
    "Trucks": f"Trucks on the ring now, their share of the vehicles and their mean speed. {TRUCK_TEXT}",
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
                                    f"asking for more than the braking limit ({MAX_DECEL:g} m/s² on a dry road, "
                                    "less on wet, snowy or icy ones; an over-reaction after a late or delayed "
                                    "response), or the no-overlap constraint stopping a car harder. Grows with "
                                    "shorter reaction distance, longer reaction time, density and a slippery road.",
    "Spacing (Gini)": SPACING_HELP,
    "Clustering": CLUSTERING_HELP,
    "Order": f"Order parameter now, and what randomly placed cars would give. {ORDER_HELP.removeprefix('Order: ')}",
    "Exits": f"Cars leaving the ring per minute, over the last {WINDOW}.",
    "Lane changes": f"Lane changes per minute, over the last {WINDOW}.",
    "Missed exits": "Leavers that reached their exit on an inner lane and had to go round again "
                    "(since the reset).",
    "Mean trip time": f"Mean time on the ring of the cars that left over the last {WINDOW}.",
    "Entry delay": ENTRY_DELAY_HELP,
    "Arrivals": f"Cars arriving at the entrances per minute, over the last {WINDOW}: at random, blocked ones "
                "included. Their mean is the Entrance chance × 60 per entrance, shaped by the demand profile.",
    "Entries": f"Cars entering the ring from the queues per minute, over the last {WINDOW}. While it stays below "
               "Arrivals − Blocked the queues are growing (the capacity test's entries come on top: they are not "
               "arrivals).",
    "Blocked": f"Arrivals per minute, over the last {WINDOW}, that found their queue full (Max queue) and were "
               "turned away: spill-back onto the approach roads.",
    "Lane": "Mean speed and number of cars in this lane (lane 1 is the inner lane).",
}


def preset_help(name: str) -> str:
    """Help of the preset selector while it shows preset ``name``."""
    p, names = presets.get(name), presets.names()
    return PRESET_HELP.format(i=names.index(p.name) + 1, n=len(names), title=p.title, description=p.description,
                              source=f" Source: {p.cite}." if p.cite else "")


def help_texts() -> list[str]:
    """Every fixed help text the status box can show (it is sized to fit the longest)."""
    texts = [help_text for _, _, help_text in BUTTONS] + [tab_help(group) for group in TABS]
    texts += [RANDOM_TAB_HELP.format(group=group) for group in GROUPS] + [ORIGIN_HELP]
    texts += [preset_help(name) for name in presets.names()]
    texts += [PRESET_TEXT_HELP.format(title=p.title) for p in presets.PRESETS.values()]
    texts += [control_help(spec) for spec in (*PARAMS, *VIEW_PARAMS.values())]
    texts += [spec.help for spec in (*CHARTS, *OPEN_CHARTS)] + list(EXTRA_HELP.values())
    texts += [f"{'Lane 8 outer' if label == 'Lane' else label}: {text}" for label, text in STAT_HELP.items()]
    return texts + [SPACETIME_HELP, VIEW_HELP, QUEUE_TAG_HELP, DEFAULT_HINT, AUTOSAVE_HINT]


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


DIM_ACCENT = mix(UI["accent"], UI["raised"], 0.6)  # selection of a control that does nothing right now


def one_sided(order: float, baseline: float) -> bool:
    """True when the order parameter is clearly above what randomly placed cars give."""
    base = baseline if np.isfinite(baseline) else 0.0
    return bool(order > max(0.05, ONE_SIDED * base))


# Where the running settings came from (the viewer keeps it with the session and the saved settings): a
# preset, {"preset": name, "values": its settings}, or random settings, {"random": seed (the latest),
# "groups": the tabs drawn or None for all, "seeds": for some tabs, each one's seed (draws of other tabs
# on top of each other add up), "values": the values drawn, "given": for all tabs, the settings given with
# --random rather than drawn, "base": for some tabs, the preset or full random draw they were drawn on, if
# any}. "values" (with the base's) tells which settings were changed since.
def preset_origin(name: str) -> dict:
    cfg = presets.apply(name)
    return {"preset": presets.get(name).name, "values": {n: getattr(cfg, n) for n in CONFIG_FIELDS}}


def random_origin(seed: int, groups, values: dict, base: dict | None = None, seeds: dict | None = None,
                  given=()) -> dict:
    """Random settings drawn with ``seed``: of every tab (``groups`` None; ``given``: the settings given
    with --random rather than drawn, which ``values`` holds too) or of the tabs ``groups`` (``seeds``:
    each tab's own seed where earlier draws are merged in, else ``seed``; ``base``: the preset or full
    random draw they were drawn on)."""
    full = groups is None
    return {"random": int(seed), "groups": None if full else list(groups),
            "seeds": None if full else {g: int((seeds or {}).get(g, seed)) for g in groups},
            "values": dict(values), "given": list(given) if full else [], "base": None if full else base}


def opt_in_values(cfg: SimConfig) -> dict:
    """``cfg``'s settings of the tabs a draw of every tab leaves alone (presets.OPT_IN_GROUPS: AVs, Coop). An
    origin of all tabs holds them too, so that a change to one counts and one not at its default is named as
    given (--random SEED starts with them only with their flags)."""
    return {p.name: getattr(cfg, p.name) for p in PARAMS if p.group in presets.OPT_IN_GROUPS}


def _whole(value) -> bool:
    """A seed as JSON gives it back: a whole number >= 0."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def valid_origin(origin, nested: bool = False) -> dict | None:
    """An origin as read back from a saved state (any version), or None if it is not a valid one."""
    if not isinstance(origin, dict):
        return None
    if isinstance(origin.get("preset"), str):
        return preset_origin(origin["preset"]) if origin["preset"] in presets.PRESETS else None
    seed, groups, raw = origin.get("random"), origin.get("groups"), origin.get("values")
    if not _whole(seed):
        return None
    if groups is not None and not (isinstance(groups, list) and groups and all(g in GROUPS for g in groups)):
        return None
    base = None if nested else valid_origin(origin.get("base"), nested=True)  # (one level deep)
    seeds = origin.get("seeds") if isinstance(origin.get("seeds"), dict) else {}
    seeds = {g: s for g, s in seeds.items() if g in (groups or ()) and _whole(s)}
    given = origin.get("given") if isinstance(origin.get("given"), list) else []
    given = [n for n in given if isinstance(n, str) and n in CONFIG_FIELDS]
    values = {}
    for name, value in (raw.items() if isinstance(raw, dict) else ()):
        if name not in CONFIG_FIELDS:
            continue  # a setting this version does not know: left out
        try:
            value = coerce(name, value)  # (JSON keeps an infinite density as "inf")
            SimConfig().updated(**{name: value})
        except (TypeError, ValueError, OverflowError):
            continue
        values[name] = value
    return random_origin(seed, groups, values, base if base is None or base.get("groups") is None else None,
                         seeds, [n for n in given if n in values])


def occupancy_rgb(frac) -> np.ndarray:
    """Sequential ramp for the dark surface: 0 = darkest step (recedes), 1 = lightest."""
    return palette.sequential_rgb(frac, dark=True)


def jam_cover(sim: RingSim) -> float:
    """Share of a lane its vehicles cover at jam density (vehicles of the mean length): the top of
    the occupancy scale."""
    return sim.mean_length / (sim.mean_length + sim.cfg.min_gap)


def speed_colours(speed: np.ndarray, free: float) -> np.ndarray:
    """uint8 RGB per bin of a speed profile; empty (NaN) bins get the empty colour."""
    empty = np.isnan(speed)
    rgb = palette.speed_rgb(np.where(empty, 0.0, speed) / max(free, 0.1))
    return np.where(empty[:, None], np.array(EMPTY), rgb).astype(np.uint8)


def anchor_out(a: float) -> str:
    """Rect anchor that puts a label on the outward side of a point at world angle ``a``."""
    c, s = math.cos(a), math.sin(a)
    v = "bottom" if s > 0.38 else "top" if s < -0.38 else ""
    h = "left" if c > 0.38 else "right" if c < -0.38 else ""
    if v and h:
        return v + h
    return "mid" + (v or h) if (v or h) else "center"


def format_value(spec: ParamSpec, value) -> str:
    if spec.choices:
        return str(value)
    return format_number(value, spec.step)


# ---------------------------------------------------------------- helpers
@dataclass
class Widget:
    kind: str  # "button", "slider", "choice", "tab", "cycler" (◀ preset ▶), "text" or "info" (drawn live)
    name: str  # action, parameter or group name
    rect: pygame.Rect  # hover area
    spec: ParamSpec | None = None
    help: str = ""
    track: pygame.Rect | None = None  # slider track
    options: list[pygame.Rect] = field(default_factory=list)  # choice buttons; a cycler's arrows
    scrolls: bool = False  # lives in the scrollable parameter list
    label: str = ""  # a button's caption
    lines: list[tuple[int, str, tuple]] = field(default_factory=list)  # a text block: (y offset, text, colour)


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
            "spacing_gini", "cars", "queued", "longest_queue", "mean_entry_delay", "entry_vc")

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
        margin = max(2.6 * fonts.line("small"), fonts.width("detector", "small") + 10,
                     fonts.width("888 full", "small") + 14 if sim.open_demand else 0)
        self.k = max((rect.w / 2 - margin) / (self.r_edge + self.stub), 0.05)  # [px/m]
        self.cx, self.cy = rect.centerx, rect.centery
        # An arm's exit and entry lanes sit this far either side of its axis.
        self.arm_offset = float(sim.r_out * np.sin(sim.entry_angle[0] - sim.arm_angle[0])) if sim.n_arms else 0.0
        self.layer = self._draw_road(sim, fonts)
        self._sector_geometry()
        self.tag_rects: list[pygame.Rect] = []  # queue count tags as last drawn (hover help)

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
        half_w = self.arm_half_width(sim)
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
        # Its label beside the road, or past the arms' ends where the arms beside it would cover it.
        label = fonts.render("detector", "small", UI["muted"])
        for r in (self.r_edge + 4 / k, r_end + 4 / k):
            x, y = pt(r, 0, det)
            box = label.get_rect(**{anchor_out(det): (round(x), round(y))})
            if not self._on_arms(box.inflate(8, 4), sim, half_w):  # (clear of them by a few pixels)
                break
        surf.blit(label, box)
        self.detector_box = box  # (in the layer's coordinates)
        inner_px = self.r_in * k
        if inner_px > 6 * fonts.line("small"):  # angle labels on the island, to read the space-time axis
            r = (inner_px - 0.16 * inner_px - 6) / k
            for deg in (0, 90, 180, 270):
                a = math.radians(deg)
                fonts.blit(surf, f"{deg}°", pt(r, 0, a), "small", UI["muted"], anchor_out(a + math.pi))
        return surf

    def arm_half_width(self, sim: RingSim) -> float:
        """[m] Half the width of an arm as drawn: its exit and entry lanes, narrowed where more arms than
        fit crowd a small ring (12 arms on a 10 m ring), so that neighbouring arms meet at the ring's edge
        at most, never overlap."""
        half_w = self.arm_offset + self.lw / 2
        return min(half_w, self.r_edge * math.sin(math.pi / sim.n_arms)) if sim.n_arms > 1 else half_w

    def _on_arms(self, box: pygame.Rect, sim: RingSim, half_w: float) -> bool:
        """Whether the road layer's ``box`` (px) overlaps an arm outside the ring (sampled on a grid)."""
        if not sim.n_arms:
            return False
        gx, gy = np.meshgrid(np.linspace(box.left, box.right, 7), np.linspace(box.top, box.bottom, 3))
        wx, wy = (gx.ravel() - self.rect.w / 2) / self.k, -(gy.ravel() - self.rect.h / 2) / self.k
        a = np.asarray(sim.arm_angle)[:, None]
        along, lat = wx * np.cos(a) + wy * np.sin(a), -wx * np.sin(a) + wy * np.cos(a)
        return bool(((along >= self.r_in) & (along <= self.r_edge + self.stub) & (np.abs(lat) <= half_w)).any())

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

    def car_polygons(self, x, y, heading, length, width, pad_px: float = 0.0) -> np.ndarray:
        """(n, 4, 2) screen corners of oriented rectangles centred on world points (``length``
        and ``width`` [m] one for all or one per rectangle)."""
        k = self.k
        sx, sy = self.cx + x * k, self.cy - y * k
        c, s = np.cos(heading), np.sin(heading)
        hl = np.maximum(np.asarray(length) * k, 2.0) / 2 + pad_px
        hw = np.maximum(np.asarray(width) * k, 1.2) / 2 + pad_px
        fx, fy = c * hl, -s * hl  # forward, screen y points down
        nx, ny = -s * hw, -c * hw  # left of forward
        corners = [(sx + fx + nx, sy + fy + ny), (sx + fx - nx, sy + fy - ny),
                   (sx - fx - nx, sy - fy - ny), (sx - fx + nx, sy - fy + ny)]
        return np.stack([np.stack(p, axis=-1) for p in corners], axis=1)

    def ring_polygons(self, snap: dict, sel=None, pad_px: float = 0.0) -> list:
        """Screen polygons (lists of points) of the vehicles ``sel`` (a mask or indices; default
        all) of a :meth:`RingSim.cars` snapshot.

        Cars are :meth:`car_polygons` rectangles. Trucks bend along their lane
        (BEND_POINTS along each side): a long truck drawn straight on a tight
        ring would stick out of its lane over the next one or the verge.
        """
        sel = slice(None) if sel is None else sel
        x, y, heading = snap["x"][sel], snap["y"][sel], snap["heading"][sel]
        length, width, truck = snap["length"][sel], snap["width"][sel], snap["truck"][sel]
        polys = self.car_polygons(x, y, heading, length, width, pad_px).tolist()
        if truck.any():
            k = self.k
            r, theta = np.hypot(x[truck], y[truck]), snap["theta"][sel][truck]
            tilt = (theta + np.pi / 2 - heading[truck] + np.pi) % TWO_PI - np.pi  # outwards during a lane change
            hl = (np.maximum(length[truck] * k, 2.0) / 2 + pad_px) / k  # [m] as car_polygons, at least 2 x 1.2 px
            hw = (np.maximum(width[truck] * k, 1.2) / 2 + pad_px) / k
            along = np.linspace(-1.0, 1.0, BEND_POINTS) * hl[:, None]
            # Round the outline: the left side back to front, then the right side front to back.
            s = np.concatenate((along, along[:, ::-1]), axis=1)
            q = np.concatenate((np.repeat(hw[:, None], BEND_POINTS, 1), np.repeat(-hw[:, None], BEND_POINTS, 1)), 1)
            c, sn = np.cos(tilt)[:, None], np.sin(tilt)[:, None]
            arc = s * c + q * sn  # [m] along the lane from the centre
            out = s * sn - q * c  # [m] outwards from the lane's centreline (left of the heading = inwards)
            rr, a = r[:, None] + out, theta[:, None] + arc / r[:, None]
            bent = np.stack((self.cx + rr * np.cos(a) * k, self.cy - rr * np.sin(a) * k), axis=-1)
            for i, poly in zip(np.flatnonzero(truck), bent.tolist()):
                polys[i] = poly
        return polys

    def draw(self, screen: pygame.Surface, sim: RingSim, snap: dict, colours: np.ndarray,
             fonts: Fonts, row: dict, hovered: int | None) -> None:
        screen.blit(self.layer, self.rect)
        self._draw_queues(screen, sim, fonts)

        occ = occupancy_rgb(metrics.sector_occupancy(sim) / jam_cover(sim)).astype(int).tolist()
        for poly, colour in zip(self.sectors, occ):
            pygame.draw.polygon(screen, colour, poly)
        self._draw_order_arrow(screen, row)

        if snap["slot"].size:
            self._draw_cars(screen, snap, colours)
            braking = snap["braking"]
            if braking.any():
                for poly in self.ring_polygons(snap, braking, pad_px=2.0):
                    pygame.draw.polygon(screen, palette.CRITICAL, poly, 2)
            if hovered is not None:
                i = np.flatnonzero(snap["slot"] == hovered)
                if i.size:
                    pygame.draw.polygon(screen, UI["text"], self.ring_polygons(snap, i, pad_px=1.5)[0], 1)

    def _draw_cars(self, screen: pygame.Surface, snap: dict, colours: np.ndarray) -> None:
        """One filled, oriented rectangle per vehicle at its own length and width (at least
        2 x 1.2 px, so tiny cars stay visible), trucks bent along their lane.

        Cheap enough for the biggest rings: ~2000 cars take about 1.5 ms, less
        than a numpy pixel splat of the same cars measured.
        """
        for poly, colour in zip(self.ring_polygons(snap), colours.tolist()):
            pygame.draw.polygon(screen, colour, poly)

    def _draw_queues(self, screen: pygame.Surface, sim: RingSim, fonts: Fonts) -> None:
        """Queued vehicles on each entry stub, trucks at their length. A queue longer than the stub,
        or any queue when the cars are too small to count (big rings), gets its count at the stub's
        end on a raised tag (red and marked "full" once it is at the limit and turns arrivals away;
        marked "test" at the capacity test's entrance); ``tag_rects`` keeps the tags for their hover help."""
        cfg = sim.cfg
        countable = (cfg.car_length + cfg.min_gap) * self.k >= QUEUE_MIN_PX
        xs, ys, hs, kinds, tags = [], [], [], [], []
        for arm, (a, q) in enumerate(zip(sim.arm_angle, sim.queue)):
            q = int(q)
            test = sim.subject and arm == SUBJECT_ARM  # the capacity test's queue: kept full, never "full"
            # Oldest first from the road outwards; (a queue edited by hand may lack the types: cars).
            trucks = np.zeros(q, dtype=bool)
            types = list(sim.queue_trucks[arm])[:q]
            trucks[:len(types)] = types
            length = np.where(trucks, cfg.truck_length, cfg.car_length)
            ends = np.cumsum(length + cfg.min_gap)  # [m] past the first one's front, each one's end + its gap
            shown = int(np.sum(ends <= self.stub - 1.0))
            r = self.r_edge + 1.0 + ends[:shown] - cfg.min_gap - length[:shown] / 2
            xs.append(r * np.cos(a) - self.arm_offset * np.sin(a))
            ys.append(r * np.sin(a) + self.arm_offset * np.cos(a))
            hs.append(np.full(shown, a + np.pi))
            kinds.append(trucks[:shown])
            if q > shown or (q > 0 and not countable):
                tags.append((a, q, test))
        if xs and sum(len(x) for x in xs):
            trucks = np.concatenate(kinds)
            polys = self.car_polygons(np.concatenate(xs), np.concatenate(ys), np.concatenate(hs),
                                      np.where(trucks, cfg.truck_length, cfg.car_length),
                                      np.where(trucks, TRUCK_WIDTH, cfg.car_width))
            for poly in polys.tolist():
                pygame.draw.polygon(screen, UI["text2"], poly)  # light enough to see on the grey stub
        self.tag_rects = []
        for a, q, test in tags:
            full = sim.open_demand and q >= sim.queue_limit and not test
            colour = palette.CRITICAL if full else UI["text"]
            text = fonts.render(f"{q} full" if full else f"{q} test" if test else str(q), "small", colour)
            pos = self.point(self.r_edge + self.stub + 4 / self.k, self.arm_offset, a)
            box = text.get_rect(**{anchor_out(a): (round(pos[0]), round(pos[1]))})
            pygame.draw.rect(screen, UI["raised"], box.inflate(6, 2), border_radius=3)
            if full:
                pygame.draw.rect(screen, colour, box.inflate(6, 2), 1, border_radius=3)
            screen.blit(text, box)
            self.tag_rects.append(box.inflate(8, 4))

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


def _saved_pairs(items, kind: type) -> list[tuple[float, object]]:
    """(time, value) pairs as a saved state keeps them (chart markers, braked cars): the
    well-formed ones of ``items`` with each value as ``kind``; anything else is skipped."""
    pairs = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, list) and len(item) == 2 and isinstance(item[0], (int, float)) \
                and isinstance(item[1], (str if kind is str else (int, float))):
            pairs.append((float(item[0]), kind(item[1])))
    return pairs


# ------------------------------------------------------------------ app
class App(Messages):
    """Pygame window: ring view, stats, live charts and the control panel."""

    def __init__(self, config: SimConfig | None = None, size: tuple[int, int] | None = None,
                 results_dir: str | Path = "results", autosave: bool = False, resume: str | Path | None = None,
                 restore_view: bool = False, origin: dict | None = None):
        """``autosave``: on quit, save the session (the simulation, its charts and the view) to
        <results_dir>/session/. ``resume``: "session" carries on from that saved session, a path
        from that saved state file; ``config`` is then only the fallback when there is nothing to
        resume (a file that cannot be restored gives its valid settings on top of it).
        ``restore_view``: also restore the saved session's view settings (tab, scroll, sim speed,
        colour modes, paused). The defaults keep no session at all. ``origin``: where ``config``
        comes from (:func:`preset_origin`, :func:`random_origin`; the command line's --preset /
        --random), for a new simulation (a resumed one has its own)."""
        dpi_aware()
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        pygame.init()
        pygame.display.set_caption("Roundabout traffic model")
        self.screen = open_window(size)
        self.results_dir = Path(results_dir)
        self.autosave = autosave
        self.clock = pygame.time.Clock()
        self.paused = False
        self.steps_per_frame = 1
        self.colour_mode = "speed"
        self.spacetime_mode = "speed"
        self.group = GROUPS[0]
        self.scroll = 0
        self.drag: Drag | None = None
        self.pressed: tuple[Widget, int] | None = None  # button or choice option under a held click
        self.mouse = (-1, -1)
        self.markers: deque[tuple[float, str]] = deque()  # (time, "Density 50→80")
        self.brakes: deque[tuple[float, float]] = deque()  # (time, angle) of perturbed cars
        self._init_messages()
        self.work_ms: deque[float] = deque(maxlen=240)  # update + draw time per frame
        self.fonts: Fonts | None = None
        self.ring: RingView | None = None
        self.stat_hits: list[tuple[pygame.Rect, str]] = []  # (hover area, help) of each readout
        self.chart_hits: list[tuple[pygame.Rect, str]] = []  # (hover area, help) of chart header readouts
        self._status_fit: tuple[tuple, int] = ((), STATUS_LINES[0])  # (font scale, width), lines
        self._gradients: dict[tuple, pygame.Surface] = {}
        self._st_cache: tuple | None = None
        self.origin: dict | None = None  # where the running settings come from (preset_origin / random_origin)
        self.preset_choice = "default"  # the preset the Presets tab shows
        self._found: tuple[SimConfig | None, str | None] = (None, None)  # (settings, the preset they are)
        self._text_cache: tuple | None = None  # a preset's text wrapped for the Presets tab
        self.sweep_viewers: list[subprocess.Popen] = []  # sweep viewers started with V, still open
        session = self._read_session() if restore_view or resume == "session" else {}
        sim, saved, note = None, None, ""
        if resume is not None:
            sim, saved, config, note = self._resume(resume, session, config)
        note += self._install(sim or RingSim(config), saved, origin=None if sim else valid_origin(origin))
        if not note and self.origin is not None:  # --preset / --random: say what is running
            more = ("its description, notes and sources are on the Presets tab" if "preset" in self.origin else
                    "X draws others, Shift+X only those of the tab shown")
            note = f"Started from {self.origin_text()}: {more}."
        if restore_view:  # the view saved with the state, then the session's on top (each invalid value is skipped)
            for view in (saved.app.get("view") if saved else None, session.get("view")):
                self._apply_view(view)
        if note:
            self._say(note, 10.0, pin=True)  # shown even with the mouse resting on the view
        self.layout()

    def _install(self, sim: RingSim, saved: state.SavedState | None = None, origin: dict | None = None) -> str:
        """Run ``sim`` with fresh charts, or with the charts, markers, braked cars and the settings'
        origin saved in ``saved`` (else ``origin``). Returns a note for the status line if those
        could not be restored."""
        self.sim = sim
        self.rec = metrics.Recorder(sim, SAMPLE_INTERVAL, maxlen=int(HISTORY / SAMPLE_INTERVAL))
        self.hist = History(self.rec.maxlen)
        self.markers.clear()
        self.brakes.clear()
        self._generation = sim.generation  # bumped by every reset / rebuild
        self.origin = origin
        note, free = "", None
        if saved is not None:
            note = self._restore_charts(saved.part("rec"))
            extras = saved.app
            self.markers.extend(_saved_pairs(extras.get("markers"), str))
            self.brakes.extend(_saved_pairs(extras.get("brakes"), float))
            free = extras.get("history_free")
            self.origin = valid_origin(extras.get("origin"))
        self.preset_choice = (self.origin or {}).get("preset") or self.found_preset() or self.preset_choice
        self.rec.update()
        self.hist.pull(self.rec)
        if isinstance(free, (int, float)) and math.isfinite(free) and free > 0:
            self.hist._free = float(free)  # the space-time colours' reference speed, as when it was saved
        return note

    def _restore_charts(self, rec_state: dict | None) -> str:
        """Fill the new, empty recorder and History with saved chart samples. Returns a note for the
        status line if they could not be restored (the charts then start afresh)."""
        if rec_state is None:
            return ""
        try:
            # The viewer reads every metric of a sample by name (charts, stats card) and draws one
            # sample per SAMPLE_INTERVAL, so the saved samples must have both. (The metrics are those
            # of a snapshot of a copy: taking one refreshes the simulation's lane index, which would
            # change how the resumed run carries on.)
            metric_names = metrics.snapshot(RingSim.from_state(self.sim.state_dict())).keys()
            self.rec.load_state(rec_state, interval=SAMPLE_INTERVAL, numbers=metric_names)
            self.rec.update()
            self.hist.pull(self.rec)
        except Exception as e:  # whatever else is wrong with them, the simulation still carries on
            self.rec.clear()
            self.hist.clear()
            reason = str(e) if isinstance(e, StateError) else f"{type(e).__name__}: {e}"
            return f" Its charts could not be restored ({reason})."
        return ""

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

    @property
    def charts(self) -> tuple[ChartSpec, ...]:
        """The line charts for the current traffic mode (open mode adds cars and queues)."""
        return CHARTS + OPEN_CHARTS if self.sim.open_demand else CHARTS

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
        charts = self._laid_out = self.charts
        pad = round(8 * s)
        cols = 2 if r.w >= 900 * s else 1
        rows = math.ceil(len(charts) / cols)
        # Beyond five rows of line charts (open mode) the space-time card gives up some of its share.
        st_h = int(r.h * SPACETIME_SHARE * min(1.0, 5 / rows))
        card_w = (r.w - (cols - 1) * m) / cols
        y_label = f.width("000.0", "small") + round(6 * s)
        x_band = f.line("small") + 2
        tick_band = round(4 * s)
        cells = []  # (row, col, columns spanned): an odd last chart spans the row
        for i in range(len(charts)):
            row, col = divmod(i, cols)
            cells.append((row, col, cols if i == len(charts) - 1 and len(charts) % cols else 1))
        widths = [round(span * card_w + (span - 1) * m) for _, _, span in cells]
        two = [self._header_width(spec) > w - 2 * pad for spec, w in zip(charts, widths)]
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
                self.chart_cards.append((charts[i], card, plot, two[i], last))
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
                          order_param=0.5, order_angle=math.radians(355), order_baseline=0.05,
                          mean_entry_delay=88.8, longest_queue=188, entry_vc=8.88)
            width += f.width(self._extra_texts(spec.extra, sample)[-1], "small") + round(12 * self.s)
        return width

    def _button_h(self) -> int:
        return self.fonts.line("base") + round(8 * self.s)

    def _button(self, name: str, rect: pygame.Rect, scrolls: bool = False, label: str = "",
                help_text: str = "") -> Widget:
        """A button widget: ``name`` from BUTTONS (its caption and help) unless both are given."""
        _, caption, text = next((b for b in BUTTONS if b[0] == name), (name, name, ""))
        widget = Widget("button", name, rect, help=help_text or text, scrolls=scrolls, label=label or caption)
        self.panel_widgets.append(widget)
        return widget

    def _button_row(self, names: list[str], x: int, y: int, w: int, scrolls: bool = False) -> None:
        """Buttons sharing out the width ``w`` in one row."""
        for name, rect in zip(names, row_rects(x, y, w, self._button_h(), len(names), round(6 * self.s))):
            self._button(name, rect, scrolls)

    def _layout_panel(self) -> None:
        f, s, r = self.fonts, self.s, self.panel_rect
        p = round(14 * s)
        x, w = r.x + p, r.w - 2 * p
        y = r.y + p + f.line("title") + f.line("small") + round(10 * s)
        self.panel_widgets: list[Widget] = []
        gap = round(6 * s)
        bh = self._button_h()
        for row, names in enumerate(BUTTON_ROWS):  # each row shares out the full width
            self._button_row(names, x, y + row * (bh + gap), w)
        y += len(BUTTON_ROWS) * (bh + gap) - gap + round(14 * s)
        for spec in (VIEW_PARAMS["steps_per_frame"], VIEW_PARAMS["colour_mode"]):
            y = self._add_control(spec, x, y, w, scrolls=False)
        y += round(4 * s)
        y = self._layout_tabs(x, y, w) + round(10 * s)
        self.status_lines = self._status_lines(w)
        status_h = self.status_lines * f.line("small") + round(12 * s)
        self.status_rect = pygame.Rect(x, r.bottom - p - status_h, w, status_h)
        self.content_rect = pygame.Rect(r.x, y, r.w, self.status_rect.y - round(8 * s) - y)
        # The tab's list: rows of (height, place(x, y, w)), laid out once the scroll is known.
        rows = self._preset_rows(w) if self.group == PRESETS_TAB else self._group_rows()
        self.content_h = sum(h for h, _ in rows)
        self.scroll = int(np.clip(self.scroll, 0, max(0, self.content_h - self.content_rect.h)))
        y -= self.scroll
        for h, place in rows:
            place(x, y, w)
            y += h

    def _group_rows(self) -> list[tuple[int, Callable]]:
        """A parameter tab: its controls, then a button that randomizes them (Shift+X)."""
        rows = [(self._control_height(spec), lambda x, y, w, spec=spec: self._add_control(spec, x, y, w, True))
                for spec in PARAMS if spec.group == self.group]
        pad = round(6 * self.s)

        def randomize(x, y, w):
            self._button("randomize_tab", pygame.Rect(x, y + pad, w, self._button_h()), scrolls=True,
                         label=f"Randomize {self.group} (Shift+X)",
                         help_text=RANDOM_TAB_HELP.format(group=self.group))
        return rows + [(self._button_h() + 3 * pad, randomize)]

    def _preset_rows(self, w: int) -> list[tuple[int, Callable]]:
        """The Presets tab: the preset selector (◀ title ▶), which preset it is, Apply preset and
        Randomize all, where the running settings come from, then the chosen preset's description,
        settings, notes and sources (presets.describe)."""
        f, s = self.fonts, self.s
        bh, small, pad = self._button_h(), f.line("small"), round(6 * s)
        name = self.preset_choice

        def cycler(x, y, w):
            arrows = [pygame.Rect(x, y, bh, bh), pygame.Rect(x + w - bh, y, bh, bh)]
            self.panel_widgets.append(Widget("cycler", "preset", pygame.Rect(x, y, w, bh), help=preset_help(name),
                                             options=arrows, scrolls=True))

        def info(which: str, height: int, help_text: str):
            return lambda x, y, w: self.panel_widgets.append(
                Widget("info", which, pygame.Rect(x, y, w, height), help=help_text, scrolls=True))

        def text(x, y, w):
            self.panel_widgets.append(Widget("text", "preset_text", pygame.Rect(x, y, w, block_h), scrolls=True,
                                             help=PRESET_TEXT_HELP.format(title=presets.get(name).title),
                                             lines=lines))

        lines, block_h = self._preset_text(name, w)
        origin_h = len(f.wrap(self._origin_line(), w)) * small  # all of it: the changed settings come last
        return [(bh + round(2 * s), cycler),
                (small + 2 * pad, info("caption", small, preset_help(name))),
                (bh + 2 * pad, lambda x, y, w: self._button_row(["apply_preset", "randomize"], x, y, w, True)),
                (origin_h + 2 * pad, info("origin", origin_h, ORIGIN_HELP)),
                (block_h + 2 * pad, text)]

    def _origin_line(self) -> str:
        """The Presets tab's line saying where the running settings come from."""
        origin = self.origin_text() or ("the model defaults" if self.found_preset() == "default" else
                                        "settings of your own (no preset or random settings)")
        return f"Running: {origin}."

    def _caption(self, width: int) -> str:
        """The line under the preset selector: which preset of how many, its --preset name and whether it
        is running (changed since), shortened by leaving out the less important parts to fit ``width``."""
        names = presets.names()
        running = (self._origin() or {}).get("preset") == self.preset_choice
        status = "running, changed" if running and self.changed_since() else "running" if running else ""
        which = f"{names.index(self.preset_choice) + 1} of {len(names)}"
        texts = [f"{which}  ·  --preset {self.preset_choice}", f"{which}  ·  {self.preset_choice}", which]
        texts = [t + (f"  ·  {status}" if status else "") for t in texts]
        return next((t for t in texts if self.fonts.width(t, "small") <= width), texts[-1])

    def _preset_text(self, name: str, w: int) -> tuple[list[tuple[int, str, tuple]], int]:
        """Preset ``name``'s :func:`presets.describe` text wrapped to ``w`` px: (y offset, line, colour)
        per line, and the block's height. Cached per preset, width and font scale."""
        key = (name, w, self.fonts.scale)
        if self._text_cache and self._text_cache[0] == key:
            return self._text_cache[1]
        f, lh, gap = self.fonts, self.fonts.line("small"), round(5 * self.s)
        out, y = [], 0
        paragraphs = presets.describe(name).splitlines()[1:]  # (the title is on the selector)
        for i, paragraph in enumerate(paragraphs):
            colour = UI["text"] if i == 0 else UI["muted"] if paragraph.startswith(("Sources", "- ")) else UI["text2"]
            if i and not paragraph.startswith("- "):
                y += gap
            for line in f.wrap(paragraph, w):
                out.append((y, line, colour))
                y += lh
        self._text_cache = (key, (out, y))
        return out, y

    def _layout_tabs(self, x: int, y: int, w: int) -> int:
        """The Presets tab and one tab per parameter group, in rows that share out the width: equal
        widths while every label fits, else widths sized to the labels; tabs that still do not fit
        wrap onto another row, as evenly shared between the rows as the labels allow. Returns the y
        just below the tabs."""
        f, s = self.fonts, self.s
        th = f.line("base") + round(8 * s)
        need = [f.width(group, "small") + round(10 * s) for group in TABS]
        rows: list[list[int]] = [[]]
        for i, width in enumerate(need):
            if rows[-1] and sum(need[j] for j in rows[-1]) + width > w:
                rows.append([])
            rows[-1].append(i)
        if len(rows) > 1:  # the same number of rows, filled evenly (first rows one tab more), if that fits
            per, more = divmod(len(TABS), len(rows))
            cuts = np.cumsum([0] + [per + (r < more) for r in range(len(rows))])
            even = [list(range(a, b)) for a, b in zip(cuts[:-1], cuts[1:])]
            if all(sum(need[i] for i in row) <= w for row in even):
                rows = even
        for row in rows:
            if max(need[i] for i in row) * len(row) <= w:
                widths = [w / len(row)] * len(row)
            else:
                spare = (w - sum(need[i] for i in row)) / len(row)
                widths = [need[i] + spare for i in row]
            left = float(x)
            for i, width in zip(row, widths):
                rect = pygame.Rect(round(left), y, round(left + width) - round(left), th)
                self.panel_widgets.append(Widget("tab", TABS[i], rect, help=tab_help(TABS[i])))
                left += width
            y += th
        return y

    def _status_lines(self, width: int) -> int:
        """Lines the help box needs so that the longest help text fits ``width`` (see ui.status_lines)."""
        self._status_fit = status_lines(self.fonts, width, help_texts, self._status_fit)
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

    def _marker(self, spec: ParamSpec, old, new) -> tuple[float, str] | None:
        """The chart marker for changing ``spec`` from ``old`` to ``new`` now (None: it gets none)."""
        if spec.group != "View" and old != new:
            return self.sim.time, f"{spec.label} {format_value(spec, old)}→{format_value(spec, new)}"
        return None

    def _mark(self, spec: ParamSpec, old, new) -> None:
        marker = self._marker(spec, old, new)
        if marker is not None:
            self.markers.append(marker)

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
        # More lanes (or open mode) need more stat rows; a new traffic mode another set of charts.
        if self.stats_rect.h < self._stats_height() or self._laid_out != self.charts:
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
        elif name == "save_state":
            self.save_state()
        elif name == "load_state":
            self.load_state()
        elif name == "sweep":
            self.open_sweep_viewer()
        elif name == "apply_preset":
            self.apply_preset()
        elif name == "randomize":
            self.randomize()
        elif name == "randomize_tab":
            self.randomize_tab()

    def _record_brake(self, slot: int | None) -> None:
        """Remember where and when a car was made to brake, for the space-time diagram."""
        if slot is not None:
            self.brakes.append((self.sim.time, float(self.sim.theta[slot])))

    def _new_paths(self, *suffixes: str, stem: str = "roundabout", folder: Path | None = None) -> list[Path]:
        """results/<stem>_<timestamp><suffix> (or in ``folder``) for each suffix, none of them taken
        yet (state.new_paths)."""
        return state.new_paths(self.results_dir if folder is None else folder, stem, *suffixes)

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
                  "  cars, queued        cars on the ring; waiting at all entrances (longest_queue: at the longest)",
                  "  arrivals_per_min    cars arriving at the entrances, entries_per_min entering the ring and",
                  f"                      blocked_per_min turned away by a full queue (open mode), over {WINDOW}",
                  f"  mean_entry_delay    queue wait of the cars that entered over {WINDOW} [s]; only the queue",
                  "                      wait, not the full HCM control delay",
                  f"  entry_vc            open mode: demand / capacity of the busiest entrance over the last "
                  f"{metrics.VC_WINDOW:g} s;",
                  "                      above 1 it gets more arrivals than it can let in (level of service F)",
                  "  throughput_per_h    cars leaving the ring per hour; demand_per_min what open mode asks for",
                  "  trucks              trucks on the ring; truck_speed, car_speed: mean speed of the trucks and",
                  "                      of the other vehicles (empty without them)",
                  "",
                  f"settings at the time of saving (seed {sim.seed})"]
        origin = self.origin_text()
        if origin:
            lines.append(f"  from {origin}")
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
        base, reference = self._command_base()
        changed = [f"--set {name}={value_text(value)}" for name, value in values.items()
                   if name in reference and not same_value(value, reference[name])]
        command = " ".join(["python -m roundabout_sim.experiments run", *base, f"--seed {sim.seed}",
                            *([f"--duration {value_text(round(sim.time, 6))}"] if sim.time > 0 else []), *changed])
        unlisted = {"--preset": "as the preset sets them", "--random": "as --random draws them"}.get(
            base[0] if base else "", "at their defaults")
        lines += ["", f"equivalent command: the final settings from t = 0 (settings not listed are {unlisted};",
                  "live changes and braked cars are not replayed)", f"  {command}", ""]
        return "\n".join(lines)

    def _command_base(self) -> tuple[list[str], dict]:
        """The experiments flags that start from the running settings' origin (--preset NAME, or
        --random SEED when it and --set give exactly the running settings, as experiments links the
        random draws to settings given with it) and the settings they give; else none and the defaults."""
        defaults = {n: getattr(SimConfig(), n) for n in CONFIG_FIELDS}
        origin = self._origin()
        if origin is not None and origin.get("base") is not None:  # some tabs drawn on a preset or a full draw
            origin = origin["base"]
        if origin is not None and "preset" in origin:
            return ["--preset", origin["preset"]], dict(origin["values"])
        if origin is not None and origin["groups"] is None:
            seed, current = origin["random"], {n: getattr(self.sim.cfg, n) for n in CONFIG_FIELDS}
            drawn = {**defaults, **presets.random_changes(seed)}
            given = {n: v for n, v in current.items() if not same_value(v, drawn[n])}
            again = {**defaults, **presets.random_changes(seed, fixed=given), **given}
            if all(same_value(again[n], current[n]) for n in CONFIG_FIELDS):
                return ["--random", str(seed)], drawn
        return [], defaults

    # ------------------------------------------------------- saved states
    def _say(self, text: str, seconds: float = 6.0, pin: bool = False) -> None:
        """Show ``text`` in the help box for a while; ``pin``: ahead of the hover help (see ui.Messages)."""
        self.message = (text, time.monotonic() + seconds)
        if pin:
            self._pin()

    @property
    def session_dir(self) -> Path:
        return self.results_dir / SESSION_DIR

    def view_settings(self) -> dict:
        """The view settings a session keeps (restored by ``restore_view``)."""
        return {"group": self.group, "scroll": int(self.scroll), "steps_per_frame": int(self.steps_per_frame),
                "colour_mode": self.colour_mode, "spacetime_mode": self.spacetime_mode, "paused": bool(self.paused)}

    def _apply_view(self, view) -> None:
        """Restore saved view settings; each one missing or invalid keeps its current value."""
        if not isinstance(view, dict):
            return
        if view.get("group") in TABS:
            self.group = view["group"]
        whole = {name: value for name, value in view.items() if isinstance(value, int) and not isinstance(value, bool)}
        if whole.get("scroll", -1) >= 0:
            self.scroll = whole["scroll"]  # (the layout clamps it to the list)
        speed = VIEW_PARAMS["steps_per_frame"]
        if speed.lo <= whole.get("steps_per_frame", 0) <= speed.hi:
            self.steps_per_frame = whole["steps_per_frame"]
        for name in ("colour_mode", "spacetime_mode"):
            if view.get(name) in VIEW_PARAMS[name].choices:
                setattr(self, name, view[name])
        if isinstance(view.get("paused"), bool):
            self.paused = view["paused"]

    def _saved_extras(self) -> dict:
        """What a saved state keeps for the viewer besides the simulation and its chart samples."""
        markers = list(self.markers)
        d = self.drag
        if d is not None and not d.widget.spec.reset:  # a live change still being dragged: as if let go now
            markers += [m for m in (self._marker(d.widget.spec, d.start, d.value),) if m is not None]
        return {"markers": [[when, label] for when, label in markers],
                "brakes": [[when, angle] for when, angle in self.brakes],
                "history_free": float(self.hist._free), "view": self.view_settings(), "origin": self.origin}

    def _read_session(self) -> dict:
        """The saved session's summary (settings and view), {} if there is none or it cannot be read."""
        try:
            summary = state.read_json(self.session_dir / SESSION_FILE)
        except StateError:
            return {}
        return summary if summary.get("format") == SESSION_FORMAT else {}

    def _resume(self, resume: str | Path, session: dict, config: SimConfig | None
                ) -> tuple[RingSim | None, state.SavedState | None, SimConfig | None, str]:
        """(simulation, saved state, settings for a new simulation, status note) for ``resume``: the
        saved simulation, or none and the valid settings of a file that cannot be restored (else
        ``config``). Never raises for a missing, damaged or incompatible file."""
        own = resume == "session"
        path = self.session_dir / SESSION_STATE if own else Path(resume)
        what = "the saved session" if own else str(path)
        if own and not path.exists() and not session:
            return None, None, config, ""  # nothing saved yet
        try:
            saved = state.load(path)
            sim = saved.simulation()
        except Exception as e:  # whatever is wrong with the file, the viewer starts
            reason = str(e) if isinstance(e, StateError) else f"{type(e).__name__}: {e}"
            settings = (session.get("settings") if own else None) or state.saved_settings(path)
            kept = self._set_aside(path) if own and self.autosave and path.exists() else ""
            if not isinstance(settings, dict):
                which = "default" if config in (None, SimConfig()) else "given"
                return (None, None, config,
                        f"Could not resume {what}: {reason}. Started with the {which} settings.{kept}")
            config, skipped = state.valid_settings(settings, config)
            left = f" except {', '.join(skipped)}" if skipped else ""
            return None, None, config, \
                f"Could not resume {what}: {reason}. Started afresh with its settings{left}.{kept}"
        how = " Start with --fresh to ignore it." if own else ""
        return sim, saved, config, f"Resumed {what} at t = {sim.time:.1f} s (saved {saved.saved}).{how}"

    def _set_aside(self, path: Path) -> str:
        """Move a saved session that could not be resumed (one from a newer version, say, or read
        while another program held it) out of the way of the autosave on quit, which would replace
        it: to session/not_resumed_<date>_<time>.npz. Returns a note saying where it went; if it
        cannot be moved, this run saves no session, so the old one stays as it is."""
        try:
            target, = self._new_paths(".npz", stem="not_resumed", folder=path.parent)
            os.replace(path, target)
        except OSError:
            self.autosave = False
            return " Quitting will not replace that session."
        return f" That session is kept as {target}."

    def save_session(self) -> Path | None:
        """Save the session to <results_dir>/session/ (on quit): the full state (simulation, charts,
        markers, view) in state.npz and the settings and view, readable, in session.json. Returns the
        folder, or None if it could not be written (the window is closing: the reason goes to stderr)."""
        folder = self.session_dir
        try:
            folder.mkdir(parents=True, exist_ok=True)
            saved = state.save(folder / SESSION_STATE, self.sim, self.rec, self._saved_extras())
            state.write_json(folder / SESSION_FILE, {
                "format": SESSION_FORMAT, "version": state.FORMAT_VERSION, "saved": saved.saved,
                "state": SESSION_STATE, "time": self.sim.time, "settings": state.settings_of(self.sim),
                "origin": self.origin_text() or None, "view": self.view_settings()})
        except (OSError, TypeError, ValueError) as e:  # (the window still closes)
            print(f"Could not save the session to {folder}: {e}", file=sys.stderr)
            return None
        return folder

    def save_state(self) -> Path | None:
        """Save the simulation, its charts and markers as results/state_<date>_<time>.npz (F5)."""
        try:
            path, = self._new_paths(".npz", stem="state")
            state.save(path, self.sim, self.rec, self._saved_extras())
        except (OSError, TypeError, ValueError) as e:  # (the viewer carries on)
            self._say(f"Could not save the state: {e}")
            return None
        self._say(f"Saved state {path} (simulation, charts and markers): F9 carries on from the most recent.")
        return path

    def latest_state(self) -> Path | None:
        """The most recently saved state_*.npz in the results folder, if any."""
        def saved_at(path: Path) -> tuple[int, str, int, str]:
            # Saves within one tick of a coarse file clock (FAT: 2 s) have the same time: then the
            # name's timestamp and counter (state_<date>_<time>-1, -2, ...) tell which came last.
            name = STATE_NAME.fullmatch(path.name)
            order = (name[1], int(name[2] or 0), path.name) if name else ("", 0, path.name)
            try:
                return path.stat().st_mtime_ns, *order
            except OSError:
                return -1, *order
        return max((p for p in self.results_dir.glob(STATE_FILES) if p.is_file()), key=saved_at, default=None)

    def load_state(self, path: str | Path | None = None) -> bool:
        """Carry on from a saved state file (default: the most recent in the results folder; F9): its
        simulation, charts, markers and braked cars, keeping the view. A file that cannot be restored
        restarts with the valid settings it holds, or changes nothing. True if it was restored."""
        path = Path(path) if path is not None else self.latest_state()
        if path is None:
            self._say(f"No saved state ({STATE_FILES}) in {self.results_dir}: F5 or Save state saves one.")
            return False
        self.drag = self.pressed = None
        try:
            saved = state.load(path)
            sim = saved.simulation()
        except Exception as e:  # whatever is wrong with the file, the viewer keeps running
            reason = str(e) if isinstance(e, StateError) else f"{type(e).__name__}: {e}"
            settings = state.saved_settings(path)
            if settings is None:
                self._say(f"Could not load {path.name}: {reason}. Nothing changed.", 10.0)
                return False
            config, skipped = state.valid_settings(settings)
            self._install(RingSim(config))
            self.layout()
            left = f" except {', '.join(skipped)}" if skipped else ""
            self._say(f"Could not load {path.name}: {reason}. Started afresh with its settings{left}.", 10.0)
            return False
        note = self._install(sim, saved)
        self.layout()
        self._say(f"Loaded {path.name} at t = {sim.time:.1f} s (saved {saved.saved}).{note}", 8.0)
        return True

    # ------------------------------------------- presets and random settings
    def found_preset(self) -> str | None:
        """The preset the running settings are (whatever the seed; presets.find), None if none is,
        cached per settings object (every change makes a new one)."""
        cfg = self.sim.cfg
        if self._found[0] is not cfg:
            self._found = (cfg, presets.find(cfg))
        return self._found[1]

    def _origin(self) -> dict | None:
        """``origin``, or the preset the settings are (other than the defaults) when none is known."""
        if self.origin is not None:
            return self.origin
        found = self.found_preset()
        return preset_origin(found) if found not in (None, "default") else None

    def changed_since(self) -> list[str]:
        """Labels of the settings changed since the preset or random settings were applied."""
        origin = self._origin() or {}
        values = {**(origin.get("base") or {}).get("values", {}), **origin.get("values", {})}
        cfg = self.sim.cfg
        return [PARAM_BY_NAME[n].label if n in PARAM_BY_NAME else n
                for n, v in values.items() if not same_value(getattr(cfg, n), v)]

    @staticmethod
    def _named(labels: list[str]) -> str:
        """Setting labels in a sentence: the first three, then how many more."""
        return ", ".join(labels[:3]) + (f" and {len(labels) - 3} more" if len(labels) > 3 else "")

    @staticmethod
    def _origin_what(origin: dict, short: bool) -> str:
        """An origin in words: "preset etoile (Place Charles de Gaulle)", "random settings, seed 4821",
        "random settings, seed 40 (given: Density)", "preset etoile (...) with random Drivers settings,
        seed 77", "random Road settings, seed 5, and Drivers settings, seed 9" (short: the title, "random,
        seed 4821", "random, seed 40, Density given", "Place Charles de Gaulle, random Drivers",
        "random Road, Drivers, seeds 5, 9")."""
        if "preset" in origin:
            title = presets.get(origin["preset"]).title
            return title if short else f"preset {origin['preset']} ({title})"
        seed = origin["random"]
        if origin["groups"] is None:
            given = App._named([PARAM_BY_NAME[n].label if n in PARAM_BY_NAME else n
                                for n in origin.get("given") or ()])
            if short:
                return f"random, seed {seed}" + (f", {given} given" if given else "")
            return f"random settings, seed {seed}" + (f" (given: {given})" if given else "")
        draws: dict[int, list[str]] = {}  # seed -> the tabs drawn with it, in the order drawn
        for group in origin["groups"]:
            draws.setdefault((origin.get("seeds") or {}).get(group, seed), []).append(group)
        base = origin.get("base")
        if short:
            tabs = ", ".join(origin["groups"])
            if base is not None:
                return f"{App._origin_what(base, True)}, random {tabs}"
            return f"random {tabs}, seed{'s' if len(draws) > 1 else ''} {', '.join(map(str, draws))}"
        parts = [f"{', '.join(tabs)} settings, seed {s}" for s, tabs in draws.items()]
        drawn = "random " + (parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])}, and {parts[-1]}")
        return f"{App._origin_what(base, False)} with {drawn}" if base is not None else drawn

    def origin_text(self, short: bool = False) -> str:
        """Where the running settings come from, e.g. "preset etoile (Place Charles de Gaulle)" or
        "random settings, seed 4821", with the settings changed since; "" if they are one's own (or
        the defaults). ``short``: for the view's header ("Place Charles de Gaulle (changed)")."""
        origin = self._origin()
        if origin is None:
            return ""
        what = self._origin_what(origin, short)
        changed = self.changed_since()
        if short or not changed:
            return what + (" (changed)" if changed else "")
        return f"{what}, then {self._named(changed)} changed"

    def cycle_preset(self, step: int) -> None:
        """Show the next (``step`` 1) or previous (-1) preset in the Presets tab (not applied yet)."""
        names = presets.names()
        self.preset_choice = names[(names.index(self.preset_choice) + step) % len(names)]
        self._layout_panel()

    def apply_preset(self, name: str | None = None) -> None:
        """Start preset ``name`` (default: the one the Presets tab shows): the defaults with its settings
        and the running seed (presets.apply), always rebuilt, with a chart marker and a status message."""
        p = presets.get(name or self.preset_choice)
        cfg = presets.apply(p.name, self.sim.cfg)
        self.drag = self.pressed = None
        changes = {n: getattr(cfg, n) for n in CONFIG_FIELDS if getattr(cfg, n) != getattr(self.sim.cfg, n)}
        self.sim.reset(seed=self.sim.seed, **changes)  # a fresh start even when no setting changes
        self._clear_history()
        self.origin, self.preset_choice = preset_origin(p.name), p.name
        self.markers.append((self.sim.time, f"Preset {p.title}"))
        self._layout_panel()
        self._say(f"Preset {p.title} ({p.name}): rebuilt with its settings, seed {self.sim.seed}. Its notes and "
                  "sources are on the Presets tab; D saves the settings with its name.", 8.0)

    @staticmethod
    def drawn_origin(seed: int, groups: list[str] | None, changes: dict, previous: dict | None) -> dict:
        """The origin after drawing ``changes`` for the tabs ``groups`` (None: all) with ``seed`` while
        the settings come from ``previous``. A draw of some tabs keeps a preset or a full draw as its
        base, and joins earlier draws of other tabs, so that each one's seed stays named (a tab drawn
        again replaces its earlier draw)."""
        if groups is None or previous is None:
            return random_origin(seed, groups, changes)
        if "preset" in previous or previous["groups"] is None:
            return random_origin(seed, groups, changes, previous)
        kept = [g for g in previous["groups"] if g not in groups]
        values = {n: v for n, v in previous["values"].items()
                  if n not in PARAM_BY_NAME or PARAM_BY_NAME[n].group not in groups}
        seeds = {**{g: (previous.get("seeds") or {}).get(g, previous["random"]) for g in kept},
                 **{g: seed for g in groups}}
        return random_origin(seed, kept + list(groups), {**values, **changes}, previous.get("base"), seeds)

    def randomize(self, groups: list[str] | None = None) -> int:
        """Plausible random values (presets.random_changes) for every setting, or for those of the tabs
        ``groups`` only, linked to the current settings of the others, drawn with a fresh seed
        (presets.new_seed) that the status message shows and the settings keep. Rebuilds when a
        rebuilding setting changes (keeping the simulation's seed), else changes the drivers live;
        either way a chart marker names it. Returns the seed."""
        seed = presets.new_seed()
        changes = presets.random_changes(seed, groups, base=self.sim.cfg)
        self.drag = self.pressed = None
        previous = self._origin()
        changed = [n for n, v in changes.items() if not same_value(getattr(self.sim.cfg, n), v)]
        rebuilt = self.sim.set(**changes)
        if rebuilt:
            self._clear_history()
        what = "Random settings" if groups is None else f"Random {', '.join(groups)} settings"
        self.markers.append((self.sim.time, f"{what}, seed {seed}"))
        if groups is None:  # the AVs and Coop tabs are kept, not drawn: given with --random, as their flags would be
            kept = opt_in_values(self.sim.cfg)
            given = [n for n, v in kept.items() if not same_value(v, getattr(SimConfig(), n))]
            self.origin = random_origin(seed, None, {**changes, **kept}, given=given)
        else:
            self.origin = self.drawn_origin(seed, groups, changes, previous)
        self._layout_panel()
        how = "rebuilt" if rebuilt else "changed live"
        again = (f"python -m roundabout_sim --random {seed}"
                 f"{''.join(f' --{n} {value_text(kept[n])}' for n in given)} starts with them" if groups is None else
                 "linked to the settings of the other tabs")
        self._say(f"{what}, seed {seed}: {len(changed)} of {len(changes)} changed, {how} ({again}). "
                  f"{'X' if groups is None else 'Shift+X'} again draws others.", 8.0)
        return seed

    def randomize_tab(self) -> int | None:
        """Randomize the settings of the tab shown (Shift+X); the Presets tab has none."""
        if self.group == PRESETS_TAB:
            self._say("Shift+X randomizes the settings of the tab shown: choose a parameter tab first (X or "
                      "Randomize all draws every setting).")
            return None
        return self.randomize([self.group])

    # ------------------------------------------------------- sweep viewer
    def open_sweep_viewer(self) -> subprocess.Popen | None:
        """Start the sweep viewer (V) in a process of its own, not waited for, with the current settings
        as the base of its new sweeps: a flat {parameter: value, "seed": ...} JSON file in the results
        folder (sweep_base_<date>_<time>.json). What it writes to stderr goes to SWEEP_LOG."""
        try:
            path, = self._new_paths(".json", stem="sweep_base")
            state.write_json(path, state.settings_of(self.sim))
        except (OSError, TypeError, ValueError) as e:
            self._say(f"Could not save the settings for the sweep viewer: {e}")
            return None
        argv = [sys.executable, "-m", "roundabout_sim.sweep_viewer", "--base-settings", str(path.resolve()),
                "--results", str(self.results_dir.resolve())]
        log = self.results_dir / SWEEP_LOG
        try:
            with open(log, "a", encoding="utf-8") as err:
                err.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')}: {cli.shell_line(argv)}\n")
                err.flush()
                # (No console window of its own on Windows; elsewhere it outlives a viewer stopped with Ctrl+C.)
                proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err,
                                        **cli.child_process({"PYGAME_HIDE_SUPPORT_PROMPT": "1"}))
        except OSError as e:
            self._say(f"Could not start the sweep viewer: {e}")
            return None
        self.sweep_viewers.append(proc)
        self._say(f"Opening the sweep viewer: its new sweeps start from these settings (seed {self.sim.seed}), "
                  f"saved as {path.name}. Set up a sweep on its New sweep tab.", 8.0)
        return proc

    def _check_sweep_viewers(self) -> None:
        """Say so when a sweep viewer started with V stopped with an error (its messages are in SWEEP_LOG)."""
        for proc in list(self.sweep_viewers):
            code = proc.poll()
            if code is None:
                continue
            self.sweep_viewers.remove(proc)
            if code:
                self._say(f"The sweep viewer stopped with an error (exit code {code}); what it printed is in "
                          f"{self.results_dir / SWEEP_LOG}.", 10.0, pin=True)

    # ------------------------------------------------------------ events
    def handle(self, e: pygame.event.Event) -> bool:
        """Process one event; False means quit."""
        if e.type == pygame.QUIT:
            return False
        if e.type == pygame.VIDEORESIZE:
            self.screen = clamp_resize(self.screen, e.size)
            self.layout()
        elif e.type == pygame.KEYDOWN:  # a message the key gives shows wherever the mouse rests
            return self._by_key(lambda: self._key(e.key, getattr(e, "mod", 0)))
        elif e.type == pygame.MOUSEMOTION:
            self._motion(e.pos, self.widget_at(e.pos))
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

    def _key(self, key: int, mod: int = 0) -> bool:
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
        elif key == pygame.K_F5:
            self.save_state()
        elif key == pygame.K_F9:
            self.load_state()
        elif key == pygame.K_x and mod & pygame.KMOD_SHIFT:
            self.randomize_tab()
        elif key == pygame.K_x:
            self.randomize()
        elif key == pygame.K_v:
            self.open_sweep_viewer()
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
        if w.kind == "tab":
            if w.name != self.group:
                self.group, self.scroll = w.name, 0
                self._layout_panel()
        elif w.kind == "slider":
            if w.track.inflate(2 * round(8 * self.s), round(20 * self.s)).collidepoint(pos):
                start = self.value(w.name)
                self.drag = Drag(w, start, start)
                self._drag_to(pos[0])
        elif w.kind in ("choice", "cycler"):
            hit = next((i for i, o in enumerate(w.options) if o.collidepoint(pos)), None)
            if hit is None and w.kind == "cycler":
                hit = 1  # the title itself steps forward (as in the sweep viewer)
            if hit is not None:
                self.pressed = (w, hit)
        elif w.kind == "button":
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
            self._clicked_on(w)  # a message it gives shows while the mouse stays on it
            self.action(w.name)
        elif w.kind == "cycler" and w.rect.collidepoint(pos):  # (let go on it: the arrow or title pressed)
            self._clicked_on(w)
            self.cycle_preset(1 if i else -1)
        elif w.kind == "choice" and w.options[i].collidepoint(pos):
            self._clicked_on(w)
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
        if self.autosave:  # quit with Esc or by closing the window: the next launch carries on here
            self.save_session()
        pygame.quit()

    # ------------------------------------------------------------ drawing
    def draw(self) -> None:
        self._follow_rebuilds()
        self._check_sweep_viewers()
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
        if self.colour_mode == "type":
            return np.array(TYPE_COLOURS)[snap["truck"].astype(int)]
        if self.colour_mode == "kind":
            return np.array(KIND_COLOURS)[snap["kind"].astype(int)]
        if self.colour_mode == "desired":
            # Temperament z sets the desired speed; with no spread every driver is the same.
            spread = 1.0 if self.sim.cfg.speed_std > 0 else 0.0
            return palette.speed_rgb(0.5 + spread * snap["z"] / 4.0).astype(int)
        return palette.speed_rgb(snap["v"] / np.maximum(snap["v0"], 0.1)).astype(int)

    def _draw_view(self, row: dict) -> None:
        sim, f = self.sim, self.fonts
        cfg = sim.cfg
        if self.ring is None:
            self.ring = RingView(sim, self.view_rect, f)
        snap = sim.cars()
        hovered = self._car_at(self.mouse) if not self.drag else None
        self.screen.set_clip(self.view_rect)
        self.ring.draw(self.screen, sim, snap, self._car_colours(snap), f, row, hovered)
        pad = round(10 * self.s)
        x, y = self.view_rect.x + pad, self.view_rect.y + pad
        road = (f"{sim.n_lanes} lane{'s' if sim.n_lanes > 1 else ''}  ·  "
                f"{sim.n_arms or 'no'} entrance{'s' if sim.n_arms != 1 else ''}/exit"
                f"{'s' if sim.n_arms != 1 else ''}  ·  radius {cfg.radius:g} m")
        mix = "".join([f"  ·  {100 * cfg.truck_share:g} % trucks"] * (cfg.truck_share > 0)
                      + [f"  ·  {cfg.road_condition} road"] * (cfg.road_condition != "dry"))
        room = self.view_rect.w - 2 * pad - (f.width("PAUSED", "base") + pad if self.paused else 0)
        origin = self.origin_text(short=True)  # the preset's title, or the random settings' seed
        firsts = dict.fromkeys((f"{origin}  ·  " if origin else "", ""))
        header = next((first + road + m + d for first in firsts for m in dict.fromkeys((mix, ""))
                       for d in self._demand_texts() if f.width(first + road + m + d, "small") <= room), road)
        f.blit(self.screen, header, (x, y), "small", UI["muted"])
        if self.paused:
            f.blit(self.screen, "PAUSED", (self.view_rect.right - pad, y), "base", UI["text"], "topright")
        if hovered is not None:
            self._car_tooltip(hovered)
        self.screen.set_clip(None)

    def _demand_texts(self) -> list[str]:
        """The open-mode demand for the view header, longest first (the longest that fits is shown)."""
        sim, cfg = self.sim, self.sim.cfg
        if not sim.open_demand:
            return [""]
        if not sim.n_arms:
            return ["  ·  open", ""]
        rate, peak, uneven = 3600 * cfg.entry_rate, cfg.demand_profile == "peak", cfg.demand_imbalance > 0
        # The capacity test's entrance gets no arrivals (its queue is kept full): the others' demand, and a note.
        other, test, short_test = ("other ", "  ·  capacity test at the bottom entrance", "  ·  capacity test") \
            if sim.subject else ("", "", "")
        if not (peak or uneven):
            texts = [f"  ·  open: {rate:.0f} veh/h at each {other}entrance{test}",
                     f"  ·  open: {rate:.0f} veh/h each{short_test}", f"  ·  open{short_test}"]
        else:
            main = rate * float(sim.demand_weight.max())  # the main road's mean
            detail = ", ".join([f"main road {main:.0f}"] * uneven + ["peak profile"] * peak)
            short = ", ".join([f"main {main:.0f}"] * uneven + ["peak"] * peak)
            texts = [f"  ·  open: {rate:.0f} veh/h per {other}entrance on average ({detail}){test}",
                     f"  ·  open: {rate:.0f} veh/h mean ({short}){short_test}",
                     f"  ·  open: {rate:.0f} veh/h mean{short_test}"]
        return list(dict.fromkeys(texts + ["  ·  open", ""]))

    def _car_tooltip(self, slot: int) -> None:
        sim, f, s = self.sim, self.fonts, self.s
        v0 = float(sim.desired_speed(np.array([slot]))[0])
        kind = "truck" if sim.truck[slot] else "car"
        lines = [f"{kind} {sim.uid[slot]}  ·  lane {sim.lane[slot] + 1}  ·  {sim.length[slot]:g} m long",
                 f"{sim.v[slot]:.1f} m/s of {v0:.1f} desired",
                 f"time gap {sim.headway[slot]:.2f} s  ·  reaction {sim.delay[slot] * sim.cfg.dt:.1f} s"]
        if sim.kind[slot] != HUMAN or sim.cfg.auto_share > 0:  # (who drives, once there can be automated ones)
            company = int(sim.company[slot])
            lines.append("human driver" if sim.kind[slot] == HUMAN else
                         f"automated: {KIND_NAMES[sim.kind[slot]]}  ·  standstill gap {sim.s0[slot]:.2f} m"
                         + (f"  ·  company {company + 1} of {sim.cfg.companies}" if company >= 0 else ""))
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
        elif self.colour_mode == "type":
            car = [("text", "Vehicle:", T), ("box", TYPE_COLOURS[0]), ("text", "car", T2),
                   ("box", TYPE_COLOURS[1]), ("text", "truck", T2)]
        elif self.colour_mode == "kind":
            car = [("text", "Kind:", T)]
            for code in self._kinds_shown():
                car += [("box", KIND_COLOURS[code]), ("text", KIND_NAMES[code], T2)]
        else:
            car = [("text", "Lane:", T)]
            for lane in range(self.sim.n_lanes):
                car += [("box", palette.SERIES[lane % len(palette.SERIES)]), ("text", str(lane + 1), T2)]
            car.append(("text", "(1 = inner)", T2))
        jam = 100 * jam_cover(self.sim)
        return [car, [("ring", palette.CRITICAL), ("text", "braking", T)],
                [("text", "Inner band, lane covered:", T), ("text", "0", T2), ("grad", "seq", gw),
                 ("text", f"{jam:.0f}% (jam)", T2)],
                [("arrow", UI["text2"]), ("text", "bunching", T)]]

    def _kinds_shown(self) -> list[int]:
        """The kinds the "kind" legend names: human drivers, the run's automated kind (while there can be
        automated vehicles) and any other kind on the ring, by code."""
        sim = self.sim
        codes = {HUMAN, *np.unique(sim.kind[sim.alive]).tolist()}
        if sim.cfg.auto_share > 0:
            codes.add(AUTO_KINDS[sim.cfg.auto_kind])
        return sorted(codes)

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
        order_text = f"{order:.2f}" if np.isfinite(order) else "–"

        def percent(value: float) -> str:  # (NaN on an empty ring: no car to average over)
            return f"{100 * value:.1f} %" if np.isfinite(value) else "–"

        def fixed(value: float) -> str:
            return f"{value:.2f}" if np.isfinite(value) else "–"

        open_demand = sim.open_demand
        if open_demand:  # the number of cars varies; Density set only the start
            cars = Stat("Cars on ring", f"{row['cars']}", key="Cars on ring (open)")
            queued = Stat("Queued to enter", f"{row['queued']} · max {row['longest_queue']}",
                          short=("Queued", f"{row['queued']} · max {row['longest_queue']}"),
                          key="Queued to enter (open)")
        else:
            cars = Stat("Cars on ring", f"{row['cars']} / {sim.target_cars()}")
            queued = Stat("Queued to enter", f"{row['queued']}")
        trucks = []
        if sim.cfg.truck_share > 0:  # only with trucks, so the card keeps its size without them
            n_trucks, truck_v = row["trucks"], row["truck_speed"]
            pct = 100 * n_trucks / max(row["cars"], 1)
            truck_speed = f"{truck_v:.1f} m/s" if np.isfinite(truck_v) else "–"
            trucks = [Stat("Trucks", f"{n_trucks} ({pct:.0f} %) · {truck_speed}",
                           short=("Trucks", f"{n_trucks} · {truck_speed}"))]
        traffic = [
            Stat("Sim time", f"{row['time']:.0f} s"),
            cars,
            queued,
            *trucks,
            Stat("Density", f"{row['density']:.1f} veh/km/lane"),
            Stat("Area covered", f"{100 * row['occupancy']:.1f} %"),
            Stat("Mean speed", f"{v:.1f} m/s · {3.6 * v:.0f} km/h" if np.isfinite(v) else "–"),
            Stat("Mean free speed", f"{free:.1f} m/s" if np.isfinite(free) else "–"),
            Stat("Ring flux", f"{row['flux']:.2f} veh/s"),
            Stat("Detector flux", f"{row['detector_flux']:.2f} veh/s"),
        ]
        congestion = [
            Stat("Jammed cars", percent(row["jam_fraction"])),
            Stat("Speed deficit", percent(row["speed_deficit"])),
            Stat("Stop-and-go", f"{row['stops_per_car_min']:.2f} stops/car/min",
                 short=("Stop-and-go", f"{row['stops_per_car_min']:.2f} /car/min")),
            Stat("Jams", f"{row['n_jams']}"),
            Stat("Following share of jams", share, short=("Following share", share)),
            Stat("Harsh braking (episodes/min)", f"{row['emergency_per_min']:.1f}",
                 short=("Harsh braking", f"{row['emergency_per_min']:.1f} /min")),
            Stat("Spacing (Gini)", fixed(row.get("spacing_gini", np.nan))),
            Stat("Clustering", fixed(row["clustering"])),
            Stat("Order", f"{order_text} · random {base_text}", short=("Order", f"{order_text} / {base_text}")),
        ]
        delay = row["mean_entry_delay"]
        delay_text = f"{delay:.1f} s · LOS {metrics.level_of_service(delay, row['entry_vc'])}" \
            if np.isfinite(delay) else "–"
        entry_delay = [Stat("Entry delay", delay_text)]
        entrances = [  # open mode: what happens at the entrances comes first
            Stat("Arrivals", f"{row['arrivals_per_min']:.1f} /min"),
            Stat("Entries", f"{row['entries_per_min']:.1f} /min"),
            Stat("Blocked", f"{row['blocked_per_min']:.1f} /min"),
            *entry_delay,
        ] if open_demand else []
        events = [
            *entrances,
            Stat("Exits", f"{row['exits_per_min']:.1f} /min"),
            Stat("Lane changes", f"{row['lane_changes_per_min']:.1f} /min"),
            Stat("Missed exits", f"{row['missed_exits']}"),
            Stat("Mean trip time", f"{trip:.0f} s" if np.isfinite(trip) else "–"),
            *([] if open_demand else entry_delay),
        ]
        for lane in range(sim.n_lanes):
            speed, cars = row.get(f"lane{lane}_speed", np.nan), row.get(f"lane{lane}_cars", 0)
            where = " inner" if lane == 0 and sim.n_lanes > 1 else " outer" if lane == sim.outer and lane else ""
            events.append(Stat(f"Lane {lane + 1}{where}",
                               f"{speed:.1f} m/s · {plural(cars, 'car')}" if np.isfinite(speed) else
                               plural(cars, "car"),
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
        if slack < 0 and len(columns) > 1 and width - sum(needs) >= round(8 * s) * (len(columns) - 1):
            gap, slack = (width - sum(needs)) / (len(columns) - 1), 0  # narrower gaps rather than clipped text
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
        if kind == "delay":
            delay, longest, vc = float(v["mean_entry_delay"]), int(v["longest_queue"]), float(v["entry_vc"])
            if not np.isfinite(delay):
                return [f"no entries in {WINDOW} · longest queue {longest}", "no entries"]
            los = metrics.level_of_service(delay, vc)
            ratio = [f"entry delay {delay:.1f} s · LOS {los} · v/c {vc:.2f} · longest queue {longest}"] \
                if np.isfinite(vc) else []
            return ratio + [f"entry delay {delay:.1f} s · LOS {los} · longest queue {longest}",
                            f"entry delay {delay:.1f} s · LOS {los}", f"delay {delay:.1f} s · LOS {los}",
                            f"{delay:.1f} s · LOS {los}", f"LOS {los}"]
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
        for tick in np.arange(0, top + step / 2, step):
            y = plot.bottom - tick / top * plot.h
            pygame.draw.line(scr, UI["grid"], (plot.x, y), (plot.right, y))
            f.blit(scr, tick_text(tick, step), (plot.x - 4, y), "small", UI["muted"], "midright")
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
                rgb = occupancy_rgb(np.stack(h.occupancy)[:, ::-1] / jam_cover(self.sim)).astype(np.uint8)
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
            if self.spacetime_mode == "speed":
                parts = [("text", "stopped", UI["muted"]), ("grad", "speed", round(56 * s)),
                         ("text", "free", UI["muted"]), ("empty", EMPTY), ("text", "no cars", UI["muted"])]
            else:
                jam = 100 * jam_cover(self.sim)
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
            if not w.scrolls:
                self._draw_widget(w)
        scr.set_clip(self.content_rect)
        for w in self.panel_widgets:
            if w.scrolls and w.rect.colliderect(self.content_rect):
                self._draw_widget(w)
        scr.set_clip(None)
        overflow = self.content_h - self.content_rect.h
        if overflow > 0:  # scroll bar
            c = self.content_rect
            bar_h = max(20, c.h * c.h // self.content_h)
            bar_y = c.y + (c.h - bar_h) * self.scroll // overflow
            pygame.draw.rect(scr, UI["grid"], (c.right - round(6 * s), bar_y, round(3 * s), bar_h), border_radius=2)
        self._draw_status()

    def _draw_widget(self, w: Widget) -> None:
        draw = {"button": self._draw_button, "tab": self._draw_tab, "cycler": self._draw_cycler,
                "info": self._draw_info, "text": self._draw_text}.get(w.kind, self._draw_control)
        draw(w)

    def _draw_button(self, w: Widget) -> None:
        f, scr = self.fonts, self.screen
        hot = w.rect.collidepoint(self.mouse) and self._visible(w, self.mouse)
        active = w.name == "pause" and self.paused
        fill = UI["accent"] if active else UI["grid"] if hot else UI["raised"]
        pygame.draw.rect(scr, fill, w.rect, border_radius=round(4 * self.s))
        label = "Play" if active else w.label
        f.blit(scr, ellipsize(f, label, w.rect.w - round(6 * self.s)), w.rect.center, "small", UI["text"], "center")

    def _draw_cycler(self, w: Widget) -> None:
        """The preset selector: ◀ arrow, the preset's title (a click on it steps forward), ▶ arrow."""
        f, s, scr = self.fonts, self.s, self.screen
        radius = round(4 * s)
        pygame.draw.rect(scr, UI["raised"], w.rect, border_radius=radius)
        for i, box in enumerate(w.options):
            hot = box.collidepoint(self.mouse) and self._visible(w, self.mouse)
            pygame.draw.rect(scr, UI["grid"] if hot else UI["raised"], box, border_radius=radius)
            cx, cy, half = box.centerx, box.centery, round(5 * s)
            tip = cx + (half if i else -half)
            back = cx - (half if i else -half)
            pygame.draw.polygon(scr, UI["text"] if hot else UI["text2"],
                                [(tip, cy), (back, cy - half), (back, cy + half)])
        room = w.rect.w - 2 * w.options[0].w - round(8 * s)
        title = ellipsize(f, presets.get(self.preset_choice).title, room, "base")
        f.blit(scr, title, w.rect.center, "base", UI["text"], "center")

    def _draw_info(self, w: Widget) -> None:
        """Live text of the Presets tab: which preset the selector shows ("caption"), or where the
        running settings come from ("origin")."""
        f, scr = self.fonts, self.screen
        if w.name == "caption":
            text = ellipsize(f, self._caption(w.rect.w), w.rect.w)
            f.blit(scr, text, (w.rect.centerx, w.rect.y), "small", UI["muted"], "midtop")
            return
        lh = f.line("small")  # (laid out to hold every line; cut only if it changed since)
        for i, line in enumerate(wrap_lines(f, self._origin_line(), w.rect.w, max(1, w.rect.h // lh))):
            f.blit(scr, line, (w.rect.x, w.rect.y + i * lh), "small", UI["text2"])

    def _draw_text(self, w: Widget) -> None:
        """A text block (the chosen preset's description, notes and sources), only its visible lines."""
        f, c, lh = self.fonts, self.content_rect, self.fonts.line("small")
        for dy, line, colour in w.lines:
            y = w.rect.y + dy
            if c.y - lh < y < c.bottom:
                f.blit(self.screen, line, (w.rect.x, y), "small", colour)

    def _draw_tab(self, w: Widget) -> None:
        draw_tab(self.screen, self.fonts, w.rect, w.name, w.name == self.group, w.rect.collidepoint(self.mouse))

    def inactive(self, spec: ParamSpec) -> str:
        """Why a control has no effect right now ("" if it has one), e.g. "open mode only" (config.inactive_reason)."""
        return inactive_reason(spec.name, self.sim.cfg)

    def _draw_control(self, w: Widget) -> None:
        """Label, tags ("rebuilds"; "open mode only" when it does nothing now: then drawn dimmed,
        though it can still be set), value and slider or choice buttons."""
        f, s, scr, spec = self.fonts, self.s, self.screen, w.spec
        x, y = w.rect.topleft
        idle = self.inactive(spec)
        label = f.blit(scr, spec.label, (x, y), "base", UI["muted"] if idle else UI["text2"])
        right = label.right
        rebuilds = spec.reset and not (idle and spec.name in NEEDS_NONZERO)  # (it is only kept meanwhile)
        for tag in ("rebuilds" if rebuilds else "", idle):
            if tag:
                right = f.blit(scr, tag, (right + round(6 * s), label.centery), "small", UI["muted"], "midleft").right
        value = self.value(spec.name)
        if spec.choices:
            self._draw_choice(w, dimmed=bool(idle))
            return
        unit = f" {spec.unit}" if spec.unit else ""
        f.blit(scr, f"{format_value(spec, value)}{unit}", (w.rect.right, y), "base",
               UI["text2"] if idle else UI["text"], "topright")
        track = w.track
        frac = float(np.clip((value - spec.lo) / (spec.hi - spec.lo), 0.0, 1.0))
        knob_x = track.x + frac * track.w
        pygame.draw.rect(scr, UI["grid"], track, border_radius=2)
        pygame.draw.rect(scr, DIM_ACCENT if idle else UI["accent"],
                         (track.x, track.y, max(int(frac * track.w), 1), track.h), border_radius=2)
        hot = (self.drag and self.drag.widget is w) or w.rect.collidepoint(self.mouse)
        pygame.draw.circle(scr, UI["text2"] if idle else UI["text"], (knob_x, track.centery),
                           round((8 if hot else 6.5) * s))

    def _draw_choice(self, w: Widget, dimmed: bool = False) -> None:
        f, scr = self.fonts, self.screen
        current = self.value(w.name)
        for option, rect in zip(w.spec.choices, w.options):
            selected = option == current
            hot = rect.collidepoint(self.mouse)
            fill = (DIM_ACCENT if dimmed else UI["accent"]) if selected else UI["grid"] if hot else UI["raised"]
            pygame.draw.rect(scr, fill, rect, border_radius=round(4 * self.s))
            f.blit(scr, option, rect.center, "small", UI["text"] if selected else UI["text2"], "center")

    def _status_text(self) -> str:
        w = self.drag.widget if self.drag else self.widget_at(self.mouse)
        text, until = self.message
        live = None if self.drag else self._live_message(w)
        if live is not None:  # from a key or a click, or one that came in by itself (see ui.Messages)
            return live
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
            if self.ring is not None and any(r.collidepoint(self.mouse) for r in self.ring.tag_rects):
                return QUEUE_TAG_HELP
            return VIEW_HELP
        for rect, help_text in self.stat_hits:
            if rect.collidepoint(self.mouse):
                return help_text
        return text if time.monotonic() < until else AUTOSAVE_HINT if self.autosave else DEFAULT_HINT

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
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m roundabout_sim", description="Interactive multi-lane roundabout traffic model.",
        epilog=f"Keys: Space pause, R reset (new seed), B brake a random car, +/- sim speed, C car colours, "
               f"S save a screenshot, D save the chart data ({HISTORY_TEXT}) as CSV with a settings file "
               "(settings, live changes, equivalent experiments command), F5 save the state "
               "(state_<date>_<time>.npz: simulation, charts, markers), F9 carry on from the most recent one, "
               "X random settings (Shift+X: only those of the tab shown; the seed shows in the help box), "
               "V open the sweep viewer with the current settings as its base, Esc quit. Files go to results/. "
               "Click a car to make it brake; hover anything for help; the Presets tab starts real-world "
               f"presets. Closing the viewer saves the session (simulation, charts and view) to "
               f"results/{SESSION_DIR}/ and the next launch carries on there; model settings given as flags "
               "(--preset and --random too) start a new simulation instead (the view settings still restore). "
               "Settings: the defaults, then --preset or --random, then the parameter flags.")
    parser.add_argument("--seed", type=cli.seed_arg, default=None, help="random seed (default: random)")
    cli.add_base_flags(parser, "parameter flags", seed_note=" (X in the viewer shows the seed it drew)")
    parser.add_argument("--size", type=cli.window_size, default=None, metavar="WxH",
                        help="window size in pixels (default: 90%% of the screen; at least 1100x700 where the "
                             "screen allows, also when given)")
    parser.add_argument("--fresh", action="store_true",
                        help=f"ignore the session saved in results/{SESSION_DIR}/ (simulation and view settings); "
                             "quitting replaces it unless --no-autosave")
    parser.add_argument("--no-autosave", "--no_autosave", dest="no_autosave", action="store_true",
                        help=f"do not save the session on quit (results/{SESSION_DIR}/ keeps the last one)")
    parser.add_argument("--state", metavar="PATH", default=None,
                        help="carry on from this saved state file (a results/state_*.npz saved with F5, or "
                             f"results/{SESSION_DIR}/{SESSION_STATE}); cannot be combined with model settings")
    defaults = SimConfig()
    for spec in PARAMS:
        flags = [f"--{spec.name}"] + ([f"--{spec.name.replace('_', '-')}"] if "_" in spec.name else [])
        default = getattr(defaults, spec.name)
        text = f"{spec.label}: {spec.help} (default {default})".replace("%", "%%")
        if spec.choices:
            parser.add_argument(*flags, dest=spec.name, choices=spec.choices, help=text)
        else:
            unit = f" {spec.unit}" if spec.unit else ""
            parser.add_argument(*flags, dest=spec.name, type=float, metavar=f"{spec.lo:g}-{spec.hi:g}{unit}",
                                help=text)
    args = parser.parse_args(argv)
    if args.preset is not None:
        try:
            args.preset = presets.get(args.preset).name
        except presets.UnknownPresetError as e:  # (lists the names)
            parser.error(f"--preset: {e}")
    given = explicit_settings(args)
    if args.state and given:
        parser.error(f"--state carries on from a saved simulation, so it takes no model settings ({', '.join(given)})")
    try:
        flags = _flag_settings(args)
    except ValueError as e:  # e.g. --n_lanes 2.5: an error, not rounded (as experiments takes it)
        parser.error(str(e))
    try:
        config_from_args(args)
    except ValueError as e:  # e.g. --density -5: say so before a window opens
        parser.error(f"invalid settings: {e}")
    for name, value in flags.items():
        if note := cli.range_note(name, value):
            print(note)
    return args


def explicit_settings(args: argparse.Namespace) -> list[str]:
    """The model settings given on the command line, as their flags (e.g. ["--seed", "--density"]).
    Any of them makes the viewer start a new simulation instead of resuming the saved session
    (whose view settings still restore); --preset and --random, which pick whole sets of settings,
    count too."""
    given = ["--seed"] * (getattr(args, "seed", None) is not None)
    given += [flag for flag in ("--preset", "--random") if getattr(args, flag[2:], None) is not None]
    return given + [f"--{spec.name}" for spec in PARAMS if getattr(args, spec.name, None) is not None]


def _flag_settings(args: argparse.Namespace) -> dict:
    """The parameters given with their own flags (--density 80, ...), read as the experiments CLI reads
    them (cli.setting_value: --n_lanes 2.5 is a ValueError, not 2 lanes)."""
    return {spec.name: cli.setting_value(spec.name, getattr(args, spec.name))
            for spec in PARAMS if getattr(args, spec.name, None) is not None}


def config_from_args(args: argparse.Namespace) -> SimConfig:
    """The settings the command line asks for (see :func:`explicit_settings` for whether it asks for any):
    the defaults, then --preset or --random, then the parameter flags (as experiments takes them). With
    --random the flags take part in the links: ``--random 40 --traffic_mode open`` draws an open-mode
    start density."""
    changes, base = _flag_settings(args), SimConfig(seed=args.seed)
    if getattr(args, "preset", None) is not None:
        base = presets.apply(args.preset, base)
    elif getattr(args, "random", None) is not None:
        base = base.updated(**presets.random_changes(args.random, fixed=changes))
    return base.updated(**changes)


def origin_from_args(args: argparse.Namespace) -> dict | None:
    """Where the command line's settings come from: --preset or --random (see :func:`preset_origin`).
    The parameter flags given with --random are named as given (they are not drawn), and changing
    them later counts as a change, as with --preset."""
    if getattr(args, "preset", None) is not None:
        return preset_origin(args.preset)
    if getattr(args, "random", None) is not None:
        changes = _flag_settings(args)
        # (--random leaves the AVs and Coop tabs alone: their values are the defaults or the flags given)
        drawn = {**presets.random_changes(args.random, fixed=changes), **opt_in_values(config_from_args(args))}
        return random_origin(args.random, None, drawn, given=[n for n in drawn if n in changes])
    return None


def session_options(args: argparse.Namespace) -> dict:
    """App's ``resume`` and ``restore_view`` for the command line: model settings start a new
    simulation, else --state carries on from that file, else the saved session resumes (--fresh:
    nothing of it, not even the view settings)."""
    if explicit_settings(args):
        return {"resume": None, "restore_view": not args.fresh}
    if args.state:
        return {"resume": Path(args.state), "restore_view": not args.fresh}
    return {"resume": None if args.fresh else "session", "restore_view": not args.fresh}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    App(config_from_args(args), size=args.size, autosave=not args.no_autosave, origin=origin_from_args(args),
        **session_options(args)).run()
