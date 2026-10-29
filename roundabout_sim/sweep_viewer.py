"""Sweep viewer: the results of a parameter sweep, and new sweeps run from the window.

Run ``python -m roundabout_sim.sweep_viewer [CSV]`` (without a CSV it opens the
most recent sweep in results/). It reads the CSV that ``python -m
roundabout_sim.experiments sweep`` writes (one row per run: every setting, the
seed, then the measured columns), also ones written before the sweep viewer
existed, and the optional ``PREFIX_meta.json`` beside it. It works out which
settings were varied and draws any measured column against them:

* lines: the metric against the X setting, mean across the runs at each point
  with a min-max band, one line per value of an optional series setting; every
  other varied setting is fixed to one of its values (a slice) or averaged over;
* heatmap: X x Y cells coloured by the mean, with a colour bar;
* scatter (the view of random-sample designs): every run as a dot coloured by
  a second setting, with the mean at each X value (in a random sample, the
  means of equal-width bins of X).

Without a meta file the varied settings, the design and the axes come from the
CSV itself: experiments writes the swept settings first, the X just before the
seed column and the series before it.

The New sweep tab sets up 1-3 settings with their ranges, the seeds and the run
lengths, and runs ``experiments sweep`` in the background (progress, ETA,
cancel); the finished CSV loads by itself. ``--base-settings FILE`` (a JSON
``{parameter: value}`` object, as the main viewer writes it) gives the settings
every new run starts from.

Mouse: hover anything for help, the chart for exact values; scroll the lists
with the wheel. Keys: Up/Down metric, Left/Right X setting, Tab series (Shift+Tab
back), H heatmap, S save PNG, R reload, O files, N new sweep, Esc quit.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pygame
import pygame.gfxdraw

from roundabout_sim import cli, palette, state
from roundabout_sim.config import (NEEDS_NONZERO, NEEDS_WEATHER, PARAM_BY_NAME, PARAMS, TEST_FIXED, SimConfig,
                                   axis_label, changed_settings, coerce, idle_note, inactive_reason)
from roundabout_sim.experiments import range_values
from roundabout_sim.metrics import LOS_LIMITS
from roundabout_sim.textfmt import base_text, format_number, plural, significant, value_text
from roundabout_sim.ui import (STATUS_LINES, Fonts, Messages, clamp_resize, dpi_aware, draw_tab, ellipsize,
                               finite_runs, nice_step, open_window, row_rects, status_lines, tick_text, wrap_lines)

FPS_CAP = 30
UI = palette.UI
EMPTY = UI["surface"]  # heatmap cells without runs
ROOT = cli.ROOT  # the folder holding the package (cwd of a sweep run)
SETTINGS = tuple(f.name for f in fields(SimConfig) if f.name != "seed")  # CSV columns that are settings
SEED = "seed"
RUN = "run"  # stands in for the varied setting of a CSV in which nothing varies and there is no seed column
TEXT_METRICS = ("entry_los",)  # text even when every value is empty (pandas would read those as numbers)
NO_TEXT = ("", "-", "–")  # a text column's "no value" (experiments writes "-" for no level of service)
SAMPLE_DISTINCT = 12  # a numeric setting with more distinct values, mostly unique and unevenly spaced, is sampled
LAYOUT_KEYS = 2  # experiments without a meta file writes at most --series and --x before the seed column
MAX_SERIES = len(palette.SERIES)  # one palette colour per series line, never cycled
MAX_POINTS = (41, MAX_SERIES, 12)  # most values of the 1st (x), 2nd (series) and 3rd setting of a new sweep
MAX_VARIABLES = len(MAX_POINTS)
SEED_CHOICES = tuple(range(1, 11))
WARMUP_CHOICES = (0.0, 10.0, 30.0, 60.0, 120.0, 180.0, 300.0, 600.0)  # [s]
MEASURE_CHOICES = (10.0, 30.0, 60.0, 120.0, 180.0, 300.0, 600.0, 900.0, 1200.0)  # [s]
PROGRESS = re.compile(r"^\[\s*(\d+)/(\d+)\]")  # experiments sweep: "[  3/12] ..." after each run
TOTAL = re.compile(r"^(\d+) runs\b")  # its first line: "12 runs (3 x 2 series x 2 seeds) ..."
LEFT = re.compile(r"~(\d+(?:\.\d+)?) s left")
VIEWS = ("lines", "heatmap", "scatter")
PAGES = (("chart", "Chart"), ("files", "Files"), ("new", "New sweep"))
DESIGN_TEXT = {"grid": "full grid", "plan": "plan", "sample": "random sample", "single": "no setting varied"}

# ------------------------------------------------------------------ help texts
DEFAULT_HINT = ("Hover the chart for exact values.  Up/Down metric  ·  Left/Right X  ·  Tab series  ·  "
                "H heatmap  ·  S save PNG  ·  R reload  ·  O files  ·  N new sweep  ·  Esc quit")
FILES_HINT = ("Click a sweep CSV to load it; the wheel scrolls the list.  O back to the chart  ·  N new sweep  ·  "
              "S save PNG  ·  R reload  ·  Esc quit (the chart keys work on the chart page)")
NEW_HINT = ("Pick 1-3 settings on the left, set their ranges and points, then Run sweep.  N back to the chart  ·  "
            "O files  ·  S save PNG  ·  Esc quit (the chart keys work on the chart page)")
VIEW_HELP = {
    "lines": "Mean of the runs at each point (the seeds, pooled with any setting averaged over) with a band from "
             "the lowest to the highest run; one line per value of the series setting. Hover for exact values.",
    "heatmap": "Each cell: the mean of the runs at that X and Y (the seeds, pooled with any setting averaged over), "
               "coloured on the bar at the right; dark = low, light = high, empty = no runs. Hover a cell for "
               "its mean, min, max and runs.",
    "scatter": "Every run as a dot: x = the chosen setting, y = the metric, colour = the colour setting (bar or "
               "legend). The light line joins the mean at each x value (a random sample: of equal-width bins of "
               "x); a ring marks a mean of one run (its dot). Hover a dot for its run, elsewhere for its value or "
               "bin.",
}
VIEW_CHOICE_HELP = ("View. lines: the metric against the X setting, mean with a min-max band, one line per series "
                    "value; heatmap: X × Y cells coloured by the mean (two varied settings; key H); scatter: every "
                    "run as a dot coloured by a second setting, with the means (the view of random samples).")
SECOND_HELP = {
    "lines": "Series: a second varied setting drawn as one line per value (palette colours, legend); up to "
             f"{MAX_SERIES} values. Key Tab.",
    "heatmap": "Y axis of the heatmap: the second varied setting. Key Tab.",
    "scatter": "Colour of the dots: another varied setting (colour bar, or a legend for a choice). Key Tab.",
}
X_HELP = "Setting on the x axis (keys Left/Right)."
SLICE_HELP = ("{label}: fix it to one of its values (a slice through the sweep) or average over all of them (every "
              "run pooled). The arrows step through average, then each value.")
BUTTON_HELP = {
    "save": "Save the chart as a PNG into the results folder (S).",
    "reload": "Read the CSV again, e.g. after a sweep wrote it anew (R).",
    "refresh": "Look for sweep CSVs in the results folder again.",
    "newest": "Load the most recent sweep CSV in the results folder.",
    "run": "Run the sweep set up here in the background (python -m roundabout_sim.experiments sweep ...); the "
           "chart opens on it when it is done.",
    "cancel": "Stop the running sweep and its workers: nothing of it is kept, unless its CSV was already complete "
              "(cancelled while drawing its figures); that CSV is then loaded, without its PNG.",
}
TAB_HELP = {
    "chart": "The chart of the loaded sweep: choose the view, the settings on the axes and the metric (O, N switch "
             "pages).",
    "files": "Sweep CSVs in the results folder, newest first: click one to load it (O).",
    "new": "Set up and run a new sweep of 1-3 settings from the base settings (N).",
}
FORM_HELP = ("Pick 1-3 settings on the left. The 1st is the x axis, the 2nd draws one line per value (up to "
             f"{MAX_SERIES}), the 3rd can be fixed or averaged over in the chart. Every run starts from the base "
             "settings.")
RANGE_HELP = "Range swept, within the slider range of the main viewer. Drag either end."
POINTS_HELP = ("Number of evenly spaced values from the start to the end of the range (rounded to the slider step). "
               "At most {cap} for the {role}.")
TOGGLE_HELP = "Include this choice in the sweep (at least one)."
REMOVE_HELP = "Take this setting out of the sweep."
STEPPER_HELP = {
    "seeds": "Runs per point, each with its own seed (base seed + k); the band shows their spread.",
    "warmup": "Simulated warm-up of each run before measuring.",
    "measure": "Simulated measurement window of each run; the metrics are its averages (onset times count from "
               "the start).",
}
FILE_HELP = "{name}: modified {when}. Click to load it."
CONTEXT_HELP = ("What the chart holds: the value or average of each other varied setting, the runs per point, the "
                "run lengths, the base settings (a preset or random settings) and every setting held fixed that "
                "differs from the defaults, all of them in the box by the mouse.")
HEADER_HELP = ("The sweep loaded: its file, its runs and design, and the settings it varies (all of them in the box "
               "by the mouse).")
EMPTY_HINT = "No sweep loaded: open one on the Files tab, or set one up and run it on the New sweep tab."
BASE_HELP = ("Base settings of every run, from the main viewer's Sweep viewer button (V) or --base-settings: the "
             "model defaults except the settings listed on the right under Base settings, which the command passes "
             "as --set. A setting swept takes the swept values instead (it is left out of that list).")
PICK_HINT = "Click a setting on the left to sweep it: up to three, in order (x axis, series, 3rd setting)."
SUMMARY_HELP = ("Size of the sweep set up here: the points of the settings multiplied, times the seeds, gives the "
                "runs; each run simulates the warm-up and then the measured window (more runs take longer).")
COMMAND_HELP = ("The command Run sweep starts, to copy into a terminal: python -m roundabout_sim.experiments sweep "
                "with the settings, the seeds, the run lengths, the base settings as --set and the output name "
                "(the date and time when it starts).")
JOB_HELP = {  # the progress readout, by the state of the sweep; paths and the command go last (cut first)
    "running": "The sweep running in the background: runs done of all runs, seconds since it started and its own "
               "estimate of the time left. When it is done its CSV loads by itself. Cancel (New sweep tab) or "
               "quitting stops it. Output: {out}.csv",
    "failed": "The last sweep failed: what went wrong first (the exception, or the command-line error), then the "
              "end of what it printed before. Command: {command}",
    "cancelled": "The last sweep was cancelled before its CSV was written, so nothing was kept (experiments "
                 "writes its CSV once every run is done; anything it had begun to write was removed).",
    "saved": "The last sweep was cancelled while drawing its figures: its CSV, with every run, was already written "
             "and was loaded on the Chart tab; there is no PNG. Output: {out}.csv",
    "done": "The last sweep finished and its CSV was loaded on the Chart tab. Output: {out}.csv",
}


# ------------------------------------------------------------------ settings
def setting_label(name: str) -> str:
    """Label of a setting with its unit, as on the chart axes (config.axis_label; also the seed and the
    run number)."""
    return {SEED: "Seed", RUN: "Run"}.get(name) or axis_label(name)


def setting_text(name: str, value) -> str:
    """A setting's value as shown, by the main viewer's rule (textfmt.format_number): with its slider step's
    decimals, and more for a value off the step (113.5, a preset's 18.25 m radius, a random sample's
    37.1235), up to 6 significant digits, so that such values are told apart exactly."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if math.isnan(v):
            return "–"
        spec = PARAM_BY_NAME.get(name)
        return format_number(v, spec.step if spec and not spec.choices and spec.step else 1.0)
    return str(value)


def setting_phrase(name: str, value) -> str:
    """'Exit chance = 0.20', 'Ring radius = 80 m'."""
    spec = PARAM_BY_NAME.get(name)
    label = spec.label if spec else setting_label(name)
    unit = f" {spec.unit}" if spec and spec.unit and not spec.choices else ""
    return f"{label} = {setting_text(name, value)}{unit}"


def fmt_num(value, digits: int = 3) -> str:
    """A measured value with ``digits`` significant digits (no exponent notation; at most 4 decimals)."""
    return significant(value, digits, max_places=4, missing="–")


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _numeric(col: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)


def _numeric_axis(col: pd.Series) -> bool:
    """A setting drawn on a numeric axis: numbers, all finite. An infinite value (Density or Reaction
    distance = inf, both valid settings) has no place on a scale, so such a setting is shown like a
    choice: its values evenly spaced and labelled, ∞ last."""
    return _numeric(col) and bool(np.isfinite(col.dropna().to_numpy(float)).all())


# ------------------------------------------------------------------ metrics
@dataclass(frozen=True)
class Metric:
    """How to show a measured column: label, scale (100 for %) and axis hints from plots.sweep_panels."""
    column: str
    label: str
    note: str = ""
    help: str = ""
    scale: float = 1.0
    from_zero: bool | None = None  # None: from zero unless some value is negative
    min_top: float = 0.0
    ref: float | None = None
    empty: str = "no run has a value"
    headline: bool = False  # one of the experiments' sweep charts (listed first)
    los: bool = False  # dashed lines at the level-of-service limits (an entry delay in s)
    ref_column: str | None = None  # a faint dashed reference per series, the mean of this column
    lines: str = ""  # what the reference lines are: with the note where they are drawn (the lines view)

    def note_for(self, view: str) -> str:
        """The note under the title: in the lines view, which draws the reference lines, also what they are."""
        return "; ".join(part for part in (self.note, self.lines if view == "lines" else "") if part)


# Columns that one sweep chart draws together, each shown on its own here: the name of each.
SPLIT_LABELS = {"stop_go_onset": "Stop-and-go onset", "congestion_onset": "Congestion onset"}
# Readable names (with units) of the measured columns that have no sweep chart of their own, or whose chart
# a sweep of closed-mode runs leaves out; anything else goes by its column name.
COLUMN_LABELS = {
    "cars": "Cars on the ring", "queued": "Cars queued to enter",
    "measured_density": "Density, measured (veh/km/lane)", "mean_free_speed": "Mean desired speed (m/s)",
    "measured_speed_std": "Speed spread (std, m/s)", "speed_deficit": "Speed deficit (0-1)",
    "trucks": "Trucks on the ring", "truck_speed": "Truck speed (m/s)", "car_speed": "Car speed (m/s)",
    "detector_flux": "Detector flux (cars/s)", "occupancy": "Area covered (0-1)",
    "jams_following": "Jams, front car following (count)", "jams_merge": "Jams, front car moving out (count)",
    "jams_exit": "Jams, front car exiting (count)", "jams_entry": "Jams, front car just entered (count)",
    "jams_other": "Jams, other front cars (count)", "order_baseline": "Order of random placement (0-1)",
    "exits_per_min": "Exits (per minute)", "missed_per_min": "Missed exits (per minute)",
    "mean_trip_time": "Mean trip time (s)", "wave_speed_kmh": "Jam wave speed (km/h)",
    "demand_per_min": "Demand (arrivals asked for, per minute)", "arrivals_per_min": "Arrivals (per minute)",
    "entries_per_min": "Entries (per minute)", "blocked_per_min": "Blocked arrivals (per minute)",
    "entry_vc": "Entry v/c (busiest entrance)", "entry_los": "Entry LOS",
    "mean_entry_delay": "Mean entry delay (s, queue wait)", "throughput_per_h": "Throughput (veh/h)",
    "longest_queue": "Longest queue, mean over time (cars)", "peak_queue": "Longest queue, longest seen (cars)",
    "speed_sd_pooled": "Speed SD, pooled over cars and time (m/s)",
    **{f"lane{k}_speed_sd": f"Speed SD on lane {k} (m/s, 0 = innermost)" for k in range(5)},
    "subject_cap_h": "Capacity test: capacity (veh/h)",
    "subject_cap15_h": "Capacity test: capacity, busiest 15 min x 4 (veh/h)",
    "subject_cap_pch": "Capacity test: capacity (pc/h)",
    "subject_vc_h": "Capacity test: conflicting flow, all lanes (veh/h)",
    "subject_vc_pch": "Capacity test: conflicting flow, all lanes (pc/h)",
    "subject_vc_outer_h": "Capacity test: conflicting flow, outer lane (veh/h)",
    "subject_queue_empty_share": "Capacity test: share with an empty queue (0-1)",
    "subject_tf": "Capacity test: follow-up headway (s)",
    "subject_tc_raff": "Capacity test: Raff critical gap (s)",
    "subject_tc_raff_count": "Capacity test: Raff critical gap, by counts (s)",
}


def column_label(column: str) -> str:
    """A measured column's readable name ("Entry LOS"), else the column name itself."""
    return COLUMN_LABELS.get(column, column)
IDENTIFIER = re.compile(r"\b([a-z][a-z0-9]*_)([a-z0-9_]*[a-z0-9])((?:\s*/\s*_[a-z0-9_]+)*)")


@lru_cache(maxsize=1)
def definitions() -> dict[str, str]:
    """experiments.DEFINITIONS split into column -> text ("cars, queued  ..." gives both names the text).
    Columns that an entry's text names ("truck_speed / car_speed: ...", "jams_following / _merge /
    ...", "peak_queue = ...") get that entry's text too, saying which entry it is."""
    try:
        from roundabout_sim import experiments
        text = experiments.DEFINITIONS
    except Exception:  # noqa: BLE001 - help text only; the viewer works without it
        return {}
    entries: dict[str, list[str]] = {}
    current: list[str] = []
    for line in text.splitlines()[1:]:
        m = re.match(r"^  (\S+(?:, \S+)*)\s{2,}(\S.*)$", line)
        if m:
            current = [n.strip() for n in m.group(1).split(",")]
            for name in current:
                entries[name] = [m.group(2).strip()]
        elif current and line.startswith("   ") and line.strip():
            for name in current:
                entries[name].append(line.strip())
    # (On screen as the main viewer writes them: "A ≤ 10", not the console's "A <= 10".)
    found = {name: re.sub(r"\s+", " ", " ".join(parts)).replace(" <= ", " ≤ ") for name, parts in entries.items()}
    mentioned: dict[str, str] = {}
    for lead, body in found.items():
        for m in IDENTIFIER.finditer(body):
            prefix = m.group(1)
            names = [prefix + m.group(2)] + [prefix + s.strip(" /_") for s in m.group(3).split("/") if s.strip()]
            for name in names:  # settings (exit_prob, max_queue ...) are mentioned too, but are never metrics
                if name not in found and name not in SETTINGS and name not in mentioned:
                    mentioned[name] = f"(defined with {lead}) {body}"
    return {**mentioned, **found}


@lru_cache(maxsize=16)
def _panels(x: str, open_demand: bool = False) -> dict:
    """Column -> (sweep chart, 0 for its own column or 1 for the second one it draws dashed); with
    ``open_demand`` (a sweep of open-mode runs) the entrance charts too, as in its sweep figure."""
    try:
        from roundabout_sim import experiments, plots
        panels = plots.sweep_panels(x, experiments.STOP_GO_ONSET, experiments.CONGESTION_ONSET,
                                    open_demand=open_demand)
    except Exception:  # noqa: BLE001 - labels only; columns fall back to their names
        return {}
    out = {}
    for p in panels:
        out[p.column] = (p, 0)
        if p.second:
            out.setdefault(p.second, (p, 1))
    return out


def metric_info(column: str, x: str = "", open_demand: bool = False) -> Metric:
    """Label, note and help of a measured column: the experiments' sweep chart of it if there is one
    (its title and note; the entrance charts for a sweep of open-mode runs), the experiments
    definitions as help, else the column name. A chart that draws two columns (the two onsets, the
    mean and the longest queue) gives each a label of its own, with its threshold or style."""
    panel, which = _panels(x, open_demand).get(column, (None, 0))
    definition = definitions().get(column, "")
    if panel is not None:
        label, note, empty = panel.title, panel.note, panel.empty
        if panel.second:  # one of two columns: name it, and say which threshold it is about
            style = panel.styles[which]
            unit = panel.title[panel.title.index(" ("):] if " (" in panel.title else ""
            name = SPLIT_LABELS.get(column, f"{panel.title.split(' (')[0]}, {style}")
            empty = f"no run crossed {style}" if column in SPLIT_LABELS else panel.empty
            # (the label names the style, "Longest queue, longest seen"; an onset's threshold goes in the note)
            label, note = f"{name}{unit}", f"{style}; {panel.note}" if column in SPLIT_LABELS else panel.note
        text = definition or note
        help_text = text if text.startswith(label) else f"{label}: {text}"  # some definitions start with it
        if which:  # the second column of a chart: its reference lines are the first one's
            return Metric(column, label, note, help_text, panel.scale, panel.from_zero, panel.min_top, None, empty,
                          headline=True)
        lines = "; ".join(part for part in (panel.lines, panel.ref_column and f"dashed = {panel.ref_label}") if part)
        return Metric(column, label, note, help_text, panel.scale, panel.from_zero, panel.min_top, panel.ref,
                      empty, headline=True, los=panel.los, ref_column=panel.ref_column, lines=lines)
    label = column_label(column)
    name = f"{label} [{column}]" if label != column else column  # the help says which CSV column it is
    help_text = f"{name}: {definition}" if definition else f"{name}: a measured column of the sweep CSV."
    return Metric(column, label, "", help_text)


def known_metrics() -> list[str]:
    """Every column the experiments describe or chart (sizes the help box before any CSV is loaded)."""
    return list(dict.fromkeys([*_panels("", True), *definitions()]))


# ------------------------------------------------------------------ sweep data
@dataclass(frozen=True)
class Variable:
    """A varied setting and the distinct values the runs have (numbers ascending, choices in their order)."""
    name: str
    values: tuple
    numeric: bool

    @property
    def label(self) -> str:
        return setting_label(self.name)

    def text(self, value) -> str:
        return setting_text(self.name, value)


def distinct(col: pd.Series, name: str) -> tuple:
    values = col.dropna().unique().tolist()
    if _numeric(col):
        return tuple(sorted(values))
    spec = PARAM_BY_NAME.get(name)
    order = list(spec.choices) if spec and spec.choices else []
    return tuple(sorted(values, key=lambda v: (order.index(v) if v in order else len(order), values.index(v))))


def looks_sampled(col: pd.Series) -> bool:
    """True for a numeric setting drawn at random: many distinct values, mostly unique, unevenly spaced.
    ``col`` holds one row per design point (the seeds' repeats dropped)."""
    values = np.sort(pd.to_numeric(col, errors="coerce").dropna().unique())
    if values.size <= SAMPLE_DISTINCT or values.size < 0.5 * col.notna().sum():
        return False
    steps = np.diff(values)
    return not np.allclose(steps, steps.mean(), rtol=1e-3, atol=1e-9)


def on_step(name: str, values) -> bool:
    """True when every value lies on the setting's slider step (as typed or set on a slider)."""
    spec = PARAM_BY_NAME.get(name)
    if spec is None or spec.choices or not spec.step:
        return True
    v = np.asarray(values, dtype=float)
    return any(bool(np.all(np.abs(k - np.round(k)) < 1e-6)) for k in (v / spec.step, (v - spec.lo) / spec.step))


def detect_design(frame: pd.DataFrame, variables: list[Variable]) -> str:
    """"grid" (every combination of the values), "sample" (random values) or "plan" (some combinations).

    Judged on the design points (the seeds' repeats dropped). A full grid is a grid however unevenly its
    values are spaced (a hand-typed list, the New sweep form's values snapped to the slider step); only
    with one setting does a random sample also look like one, and there its values lie off the step."""
    if not variables:
        return "single"
    points = frame[[v.name for v in variables]].dropna().drop_duplicates()
    sampled = [v for v in variables if v.numeric and looks_sampled(points[v.name])]
    if len(points) == math.prod(len(v.values) for v in variables):
        one = variables[0]
        return "sample" if len(variables) == 1 and sampled and not on_step(one.name, one.values) else "grid"
    return "sample" if sampled else "plan"


@dataclass
class Sweep:
    """A loaded sweep CSV with its columns classified."""
    frame: pd.DataFrame
    variables: list[Variable]  # varied settings (or the seed / run when nothing varies)
    fixed: dict  # settings with one value in every run
    metrics: list[str]  # numeric measured columns, headline charts first
    text_metrics: list[str]  # text measured columns (entry_los)
    design: str
    meta: dict = field(default_factory=dict)
    path: Path | None = None
    settings: list[str] = field(default_factory=list)
    ordered: bool = False  # the meta file listed the varied settings (in their --vary order)
    layout: list[str] = field(default_factory=list)  # no meta file: the CSV's --x, then --series (see classify)
    open_demand: bool = False  # some runs are in open traffic mode: the entrance charts are headline metrics

    def var(self, name: str) -> Variable:
        return next(v for v in self.variables if v.name == name)

    @property
    def runs(self) -> int:
        return len(self.frame)

    @property
    def varied(self) -> list[Variable]:
        """The varied settings proper (not the seed / run stand-in)."""
        return [v for v in self.variables if v.name not in (SEED, RUN) or v.name in self.settings]


def classify(frame: pd.DataFrame, meta: dict | None = None) -> Sweep:
    """Settings (named like SimConfig fields), the seed and the measured rest; the varied settings
    from the meta file if it names them, else those with more than one value.

    experiments writes the swept settings first and the seed right after them (``--series``,
    ``--x``, ``seed``, ...), so a column before the seed that varies is a setting even when it is no
    SimConfig field (one from another version), and without a meta file the column just before the
    seed is the X and the one before it the series (``layout``)."""
    meta = meta if isinstance(meta, dict) else {}
    columns = [str(c) for c in frame.columns]
    frame = frame.set_axis(columns, axis=1)
    named = [v.get("name") for v in meta.get("varied", []) if isinstance(v, dict)] \
        if isinstance(meta.get("varied"), list) else []
    named = [n for n in named if isinstance(n, str) and n in columns and n != SEED]
    lead = columns[:columns.index(SEED)] if SEED in columns else []
    swept = [c for c in lead if c not in SETTINGS and frame[c].nunique(dropna=True) > 1]
    settings = [c for c in columns if c in SETTINGS or c in named or c in swept]
    measured = [c for c in columns if c not in settings and c != SEED]
    text = [c for c in measured if c in TEXT_METRICS or not _numeric(frame[c])]
    numeric = [c for c in measured if c not in text]
    open_demand = "traffic_mode" in frame and bool((frame["traffic_mode"].astype(str) == "open").any())
    headline = [c for c in _panels("", open_demand) if c in numeric]
    metrics = headline + [c for c in numeric if c not in headline]
    varied = list(dict.fromkeys(named)) if named else [c for c in settings if frame[c].nunique(dropna=True) > 1]
    variables = [Variable(n, distinct(frame[n], n), _numeric_axis(frame[n])) for n in varied]
    fixed = {c: frame[c].dropna().iloc[0] for c in settings if c not in varied and frame[c].notna().any()}
    fixed = {c: (v.item() if hasattr(v, "item") else v) for c, v in fixed.items()}
    design = meta.get("design") if meta.get("design") in ("grid", "plan", "sample") else \
        detect_design(frame, variables)
    if not variables:  # nothing varies: one point per seed (or per row)
        if SEED in frame and frame[SEED].nunique() > 0:
            variables = [Variable(SEED, distinct(frame[SEED], SEED), _numeric(frame[SEED]))]
        else:
            frame = frame.assign(**{RUN: np.arange(1, len(frame) + 1)})
            variables = [Variable(RUN, tuple(range(1, len(frame) + 1)), True)]
        design = "single"
    layout = []
    if not named and 0 < len(lead) <= LAYOUT_KEYS and all(c in settings for c in lead):
        layout = [c for c in reversed(lead) if c in varied]
    return Sweep(frame, variables, fixed, metrics, text, design, meta, settings=settings, ordered=bool(named),
                 layout=layout, open_demand=open_demand)


def axis_order(sweep: Sweep) -> list[str]:
    """The varied settings in the order they go on the axes: as the meta file lists them; else, in a
    CSV laid out as experiments writes it, its --x and then its --series; the others (or, in any
    other CSV, all of them) by the numeric one with the most values first and then by fewest values."""
    names = [v.name for v in sweep.variables]
    if sweep.ordered:
        return names
    first = [n for n in sweep.layout if n in names]
    pos = {name: i for i, name in enumerate(names)}
    rest = [v for v in sweep.variables if v.name not in first]
    if not first and rest:
        x = max(rest, key=lambda v: (v.numeric, len(v.values), pos[v.name]))
        first, rest = [x.name], [v for v in rest if v is not x]
    return first + [v.name for v in sorted(rest, key=lambda v: (len(v.values), pos[v.name]))]


def meta_path(csv: str | Path) -> Path:
    csv = Path(csv)
    return csv.with_name(f"{csv.stem}_meta.json")


def read_meta(csv: str | Path) -> dict:
    """PREFIX_meta.json beside PREFIX.csv ({} if there is none or it cannot be read)."""
    path = meta_path(csv)
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def load_sweep(path: str | Path) -> Sweep:
    """Read and classify a sweep CSV; ValueError if it is not one."""
    path = Path(path)
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as err:
        raise ValueError(f"cannot read {path.name}: {err}") from None
    if frame.empty:
        raise ValueError(f"{path.name} has no runs")
    sweep = classify(frame, read_meta(path))
    if not sweep.settings:
        raise ValueError(f"{path.name} has no setting columns (not a sweep CSV)")
    if not sweep.metrics:
        raise ValueError(f"{path.name} has no numeric measured columns (not a sweep CSV)")
    sweep.path = path
    return sweep


def looks_like_sweep(path: Path) -> bool:
    """A CSV whose header has the seed, some settings and the headline metrics (not a run's time series)."""
    if path.name.endswith(("_timeseries.csv", "_summary.csv")):
        return False
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            header = fh.readline()
    except OSError:
        return False
    names = {h.strip().strip('"') for h in header.split(",")}
    return SEED in names and "time" not in names and len(names & set(SETTINGS)) >= 3 and \
        bool(names & {"flux", "jam_fraction"})


def find_sweeps(results_dir: str | Path) -> list[Path]:
    """Sweep CSVs in ``results_dir`` and its subfolders one level down, newest first."""
    root = Path(results_dir)
    if not root.is_dir():
        return []
    paths = [p for p in (*root.glob("*.csv"), *root.glob("*/*.csv")) if p.is_file() and looks_like_sweep(p)]
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


# ------------------------------------------------------------------ aggregation
def matches(col: pd.Series, value) -> np.ndarray:
    if _numeric(col) and _is_number(value):
        return np.isclose(col.to_numpy(float), float(value), rtol=1e-12, atol=1e-12)
    return (col.astype(str) == str(value)).to_numpy()


def select(frame: pd.DataFrame, slices: dict) -> pd.DataFrame:
    """The runs with each setting in ``slices`` at its value (None = all of them: averaged over)."""
    keep = np.ones(len(frame), dtype=bool)
    for name, value in slices.items():
        if value is not None:
            keep &= matches(frame[name], value)
    return frame[keep]


def codes(col: pd.Series, values: tuple) -> np.ndarray:
    """Index of each row's value in ``values`` (-1 if it is not there)."""
    if _numeric(col):
        ref = np.asarray(values, dtype=float)
        x = col.to_numpy(float)
        hit = np.isclose(x[:, None], ref[None, :], rtol=1e-12, atol=1e-12) if ref.size else np.zeros((x.size, 0), bool)
        return np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    return pd.Index([str(v) for v in values]).get_indexer(col.astype(str))


@dataclass
class Stats:
    """Per cell (rows = Y or series values, columns = X values): mean, min, max of the finite
    values, how many there are (n) and how many runs (runs, values or not)."""
    mean: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    n: np.ndarray
    runs: np.ndarray


def grid_stats(frame: pd.DataFrame, metric: str, x: Variable, y: Variable | None = None) -> Stats:
    """Mean / min / max of ``metric`` over the runs at each X (and Y) value, pooling everything else."""
    vals = pd.to_numeric(frame[metric], errors="coerce").to_numpy(float) if metric in frame else \
        np.full(len(frame), np.nan)
    xi = codes(frame[x.name], x.values)
    yi = codes(frame[y.name], y.values) if y is not None else np.zeros(len(frame), dtype=int)
    shape = (len(y.values) if y is not None else 1, len(x.values))
    inside = (xi >= 0) & (yi >= 0)
    runs = np.zeros(shape, dtype=int)
    np.add.at(runs, (yi[inside], xi[inside]), 1)
    ok = inside & np.isfinite(vals)
    n = np.zeros(shape, dtype=int)
    total = np.zeros(shape)
    lo = np.full(shape, np.inf)
    hi = np.full(shape, -np.inf)
    cells = (yi[ok], xi[ok])
    np.add.at(n, cells, 1)
    np.add.at(total, cells, vals[ok])
    np.minimum.at(lo, cells, vals[ok])
    np.maximum.at(hi, cells, vals[ok])
    empty = n == 0
    mean = np.where(empty, np.nan, total / np.maximum(n, 1))
    return Stats(mean, np.where(empty, np.nan, lo), np.where(empty, np.nan, hi), n, runs)


def binned_means(x: np.ndarray, y: np.ndarray, bins: int | None = None):
    """Equal-width bins of ``x``: (centres, means, mins, maxs, counts, edges), as the sample figure bins
    them (plots.binned_stats)."""
    from roundabout_sim.plots import binned_stats

    return binned_stats(x, y, bins)


def text_value(value) -> str | None:
    """A text column's value, or None for no value (empty, or the "-" experiments writes)."""
    if not isinstance(value, str) or value.strip() in NO_TEXT:
        return None
    return value.strip()


def text_counts(frame: pd.DataFrame, column: str) -> str:
    """'A ×2, B ×1': how often each value of a text column occurs ("" if none)."""
    if column not in frame:
        return ""
    counts = frame[column].dropna().astype(str).str.strip().value_counts()
    counts = counts[~counts.index.isin(NO_TEXT)]
    return ", ".join(f"{k} ×{v}" for k, v in sorted(counts.items()))


# ------------------------------------------------------------------ sweep command
def range_spec(values: list) -> str:
    """start:stop:step when the experiments CLI turns it back into exactly these values, else a comma list."""
    if len(values) >= 3 and all(_is_number(v) for v in values):
        start, stop = float(values[0]), float(values[-1])
        step = value_text(round((stop - start) / (len(values) - 1), 10))
        if float(step) > 0:
            parsed = range_values(start, stop, float(step))  # what experiments makes of it
            if len(parsed) == len(values) and all(abs(a - float(b)) <= 1e-9 * max(1.0, abs(float(b)))
                                                  for a, b in zip(parsed, values)):
                return f"{value_text(values[0])}:{value_text(values[-1])}:{step}"
    return ",".join(value_text(v) for v in values)


def sweep_args(variables: list[tuple[str, list]], *, seeds: int, warmup: float, measure: float, out: str | Path,
               base: dict | None = None, seed: int = 0) -> list[str]:
    """``experiments`` arguments of a sweep: one or two settings as --x and --series (at most
    MAX_SERIES values), as the sweep has always taken them; three as one --vary each, in order (so
    their order never depends on how --x / --series mix with --vary); ``base`` settings that differ
    from the defaults as --set."""
    if not 1 <= len(variables) <= MAX_VARIABLES:
        raise ValueError(f"a sweep needs 1-{MAX_VARIABLES} settings, got {len(variables)}")
    names = [name for name, _ in variables]
    if len(set(names)) != len(names):
        raise ValueError("each setting can be swept only once")
    args = ["sweep"]
    for i, (name, values) in enumerate(variables):
        if name not in SETTINGS:
            raise ValueError(f"unknown setting {name!r}")
        if not values:
            raise ValueError(f"{name}: no values")
        if i == 1 and len(values) > MAX_SERIES:
            raise ValueError(f"{name}: at most {MAX_SERIES} values for the 2nd setting (one line each)")
        vary = len(variables) > 2
        spec = ",".join(value_text(v) for v in values) if i == 1 and not vary else range_spec(list(values))
        args += ["--vary" if vary else ("--x", "--series")[i], f"{name}={spec}"]
    args += ["--seeds", str(int(seeds)), "--seed", str(int(seed)), "--warmup", value_text(float(warmup)),
             "--measure", value_text(float(measure)), "--out", str(out)]
    for name, value in changed_settings(base or {}).items():
        if name not in names:
            args += ["--set", f"{name}={value_text(value)}"]
    return args


def sweep_command(variables: list[tuple[str, list]], *, python: str | None = None, **kwargs) -> list[str]:
    """The full command line (unbuffered, so the progress arrives as it happens)."""
    return [python or sys.executable, "-u", "-m", "roundabout_sim.experiments", *sweep_args(variables, **kwargs)]


def display_command(argv: list[str]) -> str:
    """The command as one line to paste into this platform's shell."""
    return cli.shell_line(argv)


def cancelled_when(job) -> str:
    """'after 3 of 12 runs', or 'while saving its results' once every run had finished."""
    if job.total and job.done >= job.total:
        return "while saving its results"
    return f"after {job.done} of {job.total or '?'} runs"


def parse_progress(line: str) -> tuple[int, int] | None:
    """(runs done, runs in all) from an ``experiments sweep`` progress line, else None."""
    m = PROGRESS.match(line)
    return (int(m.group(1)), int(m.group(2))) if m else None


class SweepRun:
    """A sweep subprocess whose output is read on threads, so the window never waits for it."""

    def __init__(self, argv: list[str], out: str | Path, cwd: str | Path | None = None):
        self.argv, self.out = list(argv), Path(out)
        self.started = time.monotonic()
        self.finished: float | None = None
        self.done, self.total = 0, None
        self.left: float | None = None  # [s] the sweep's own estimate from its last progress line
        self.last = ""  # last line it printed
        self.output: deque[str] = deque(maxlen=60)
        self.errors: deque[str] = deque(maxlen=60)
        self.cancelled = False
        self.state = "running"  # then "done", "failed" or "cancelled"
        self.handled = False  # the viewer has dealt with the outcome
        self.saved = False  # cancelled, but only once its CSV was written (and the viewer loaded it)
        self.proc = subprocess.Popen(self.argv, cwd=str(cwd or ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1,
                                     **cli.child_process({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}))
        self._readers = [threading.Thread(target=self._read, args=(stream, sink), daemon=True)
                         for stream, sink in ((self.proc.stdout, self._out), (self.proc.stderr, self.errors.append))]
        for reader in self._readers:
            reader.start()

    @staticmethod
    def _read(stream, sink) -> None:
        for line in iter(stream.readline, ""):
            sink(line.rstrip("\r\n"))
        stream.close()

    def _out(self, line: str) -> None:
        self.output.append(line)
        if line.strip():
            self.last = line
        progress = parse_progress(line)
        if progress:
            self.done, self.total = progress
            m = LEFT.search(line)
            self.left = float(m.group(1)) if m else None
        elif self.total is None and (m := TOTAL.match(line)):
            self.total = int(m.group(1))

    def poll(self) -> str:
        if self.state == "running" and self.proc.poll() is not None:
            for reader in self._readers:  # the pipes close with the process: read what is left
                reader.join(timeout=2.0)
            self.finished = time.monotonic()
            self.state = "cancelled" if self.cancelled else "done" if self.proc.returncode == 0 else "failed"
        return self.state

    def cancel(self) -> None:
        """Stop the sweep and its worker processes."""
        if self.proc.poll() is not None:
            return
        self.cancelled = True
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)], capture_output=True,
                               timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                os.killpg(self.proc.pid, 15)
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)

    def remove_partial(self) -> list[Path]:
        """After a cancel before its CSV was written: remove what experiments left of this sweep (a meta
        file, half-written temporary files; their names start with the sweep's own fresh prefix), so
        that nothing of it stays in the results folder. Returns the files removed."""
        out = Path(self.out)
        stem = glob.escape(out.name)
        found = [Path(f"{out}_meta.json"), *out.parent.glob(f".{stem}.csv.*.tmp"),
                 *out.parent.glob(f".{stem}_meta.json.*.tmp")]
        removed = []
        for path in found:
            try:
                path.unlink()
                removed.append(path)
            except OSError:
                pass
        return removed

    @property
    def elapsed(self) -> float:
        return (self.finished or time.monotonic()) - self.started

    def fraction(self) -> float:
        return self.done / self.total if self.total else 0.0

    def eta(self) -> float | None:
        """[s] left: the sweep's own estimate, else from the pace so far (None before the first run)."""
        if self.left is not None:
            return self.left
        if self.done and self.total:
            return self.elapsed / self.done * (self.total - self.done)
        return None

    def _tail(self) -> list[str]:
        return [line for line in (self.errors or self.output) if line.strip()]

    def error_line(self) -> str:
        """What went wrong in one line: the last unindented line of stderr (or stdout), i.e. the
        exception at the end of a traceback or argparse's "error: ..." line."""
        tail = self._tail()
        last = next((line for line in reversed(tail) if not line[:1].isspace()), tail[-1] if tail else "")
        return last.strip() or f"exit code {self.proc.returncode}"

    def error_text(self, lines: int = 6) -> str:
        """The error line first, then the last lines before it (the end of the traceback)."""
        head = self.error_line()
        before = [line.strip() for line in self._tail() if line.strip() != head][-lines:]
        return f"{head}  ·  {' '.join(before)}" if before else head


# ------------------------------------------------------------------ new-sweep form
@dataclass
class FormVar:
    """One setting of a new sweep: a range and a number of points, or a subset of the choices."""
    name: str
    lo: float = 0.0
    hi: float = 0.0
    points: int = 5
    chosen: list = field(default_factory=list)

    @property
    def spec(self):
        return PARAM_BY_NAME[self.name]

    @classmethod
    def new(cls, name: str) -> "FormVar":
        spec = PARAM_BY_NAME[name]
        if spec.choices:
            return cls(name, chosen=list(spec.choices))
        var = cls(name, float(spec.lo), float(spec.hi))
        var.points = min(5, var.max_points())
        return var

    def max_points(self, cap: int | None = None) -> int:
        spec = self.spec
        n = len(spec.choices) if spec.choices else int(round((self.hi - self.lo) / spec.step)) + 1
        return max(1, min(n, cap or n))

    def values(self) -> list:
        spec = self.spec
        if spec.choices:
            return [c for c in spec.choices if c in self.chosen]
        n = max(1, min(self.points, self.max_points()))
        raw = [self.lo] if n == 1 else np.linspace(self.lo, self.hi, n).tolist()
        snapped = [round(self.lo + round((v - self.lo) / spec.step) * spec.step, 10) for v in raw]
        return list(dict.fromkeys(coerce(self.name, v) for v in snapped))


# What to change when a setting picked for a new sweep would do nothing in any of its runs.
IDLE_FIX = {"open mode only": "pick Traffic mode too (with open), or start from open-mode base settings",
            "no entrances": "pick Arms too, or start from base settings with entrances",
            "no trucks": "pick Truck share too, or start from base settings with trucks",
            "dry road": "pick Road condition too, or start from base settings off a dry road",
            "capacity test": "pick Capacity test too (with off), or start from base settings without it"}


def param_help(spec, chosen: bool, idle: str = "") -> str:
    """Help of a setting in the New sweep list: its range, its help, whether it would do anything in the
    sweep set up (``idle``: why not, from config.inactive_reason) and what a click does."""
    rng = "/".join(spec.choices) if spec.choices else f"{spec.lo:g}–{spec.hi:g}{' ' + spec.unit if spec.unit else ''}"
    note = f" Here it would change nothing ({idle}): {IDLE_FIX.get(idle, 'change the base settings')}." if idle else ""
    return f"{spec.label} ({rng}): {spec.help}{note} Click to {'remove it from' if chosen else 'add it to'} the sweep."


def idle_text(form_idle: dict[str, str]) -> str:
    """The New sweep form's warning for the picked settings that would change nothing in any run
    (``form_idle``: label -> reason), one sentence per reason ("" if every one acts)."""
    reasons: dict[str, list[str]] = {}
    for label, reason in form_idle.items():
        if reason:
            reasons.setdefault(reason, []).append(label)
    return " ".join(f"{idle_note(labels, reason).replace('every run here', 'every run of this sweep')} "
                    f"({IDLE_FIX.get(reason, 'change the base settings')})." for reason, labels in reasons.items())


def load_base(source) -> tuple[dict, int | None, list[str]]:
    """Base settings of new sweeps from a JSON file or dict ``{parameter: value}`` (plus an optional
    seed): (the settings, the seed, the names that are no settings and were left out). Bad values,
    and an object in which no name is a setting (a typo'd or nested file), raise ValueError."""
    raw = json.loads(Path(source).read_text(encoding="utf-8")) if isinstance(source, (str, Path)) else source
    if not isinstance(raw, dict):
        raise ValueError("expected a JSON object {parameter: value}")
    changes, seed, ignored = {}, None, []
    for name, value in raw.items():
        if name == SEED:
            # As SimConfig and --seed take it: a whole number >= 0 (not 1.5 cut to 1, nor true for 1).
            if value is not None and (not _is_number(value) or not math.isfinite(value)
                                      or not float(value).is_integer() or value < 0):
                raise ValueError(f"seed must be a whole number >= 0, got {value!r}")
            seed = None if value is None else int(value)
        elif name in SETTINGS:
            changes[name] = coerce(name, value)
        else:
            ignored.append(str(name))
    if ignored and not changes and seed is None:
        raise ValueError(f"no setting in it ({', '.join(ignored[:5])}{' ...' if len(ignored) > 5 else ''} are "
                         "not parameter names; expected {parameter: value})")
    SimConfig().updated(**changes)  # validates the values
    return changes, seed, ignored


def help_texts() -> list[str]:
    """Every help text the help box can show (with short stand-ins where a path or a list of settings
    goes at the end), to size the box once for all of them."""
    texts = [DEFAULT_HINT, FILES_HINT, NEW_HINT, EMPTY_HINT, CONTEXT_HELP, HEADER_HELP, VIEW_CHOICE_HELP, X_HELP,
             FORM_HELP, RANGE_HELP,
             TOGGLE_HELP, REMOVE_HELP, SUMMARY_HELP, COMMAND_HELP,
             *(text.format(out="results/sweep", command="python") for text in JOB_HELP.values()),
             BASE_HELP, FILE_HELP.format(name="folder/sweep_20260101_120000.csv", when="2026-01-01 12:00"),
             *VIEW_HELP.values(), *SECOND_HELP.values(), *BUTTON_HELP.values(), *TAB_HELP.values(),
             *STEPPER_HELP.values()]
    texts += [POINTS_HELP.format(cap=cap, role=role) for cap, role in
              zip(MAX_POINTS, ("x axis (1st setting)", "series (one line per value)", "3rd setting"))]
    for spec in PARAMS:
        idle = f"{spec.applies_in} mode only" if spec.applies_in else NEEDS_NONZERO.get(spec.name, ("", ""))[1] \
            or ("dry road" if spec.name in NEEDS_WEATHER else "")
        texts += [param_help(spec, True), param_help(spec, False, idle), f"{setting_label(spec.name)}: {spec.help}",
                  SLICE_HELP.format(label=setting_label(spec.name))]
    texts.append(idle_text({setting_label(n): "open mode only" for n in ("entry_rate", "demand_imbalance",
                                                                         "demand_profile")}))
    texts += [param_help(PARAM_BY_NAME[name], False, "capacity test") for name in TEST_FIXED]
    texts += [metric_info(name, x, True).help for name in known_metrics() for x in ("", "density")]
    return texts


# ------------------------------------------------------------------ widgets
@dataclass
class Widget:
    kind: str  # button, tab, choice, cycler, metric, file, heading, param, range, toggle, remove, info
    name: str
    rect: pygame.Rect
    help: str = ""
    region: str | None = None  # the scroll region it lives in (clipped to it)
    options: list[pygame.Rect] = field(default_factory=list)  # choice/toggle buttons; cycler arrows
    label: str = ""
    data: object = None


def heat_rgb(t) -> np.ndarray:
    """Heatmap and colour-bar ramp on the dark surface: dark = low, light = high."""
    return palette.sequential_rgb(t, dark=True)


def dot_rgb(t) -> np.ndarray:
    """Scatter dots: the same ramp without its darkest steps, so every dot stands off the surface."""
    return palette.sequential_rgb(0.25 + 0.75 * np.clip(np.nan_to_num(np.asarray(t, dtype=float)), 0, 1), dark=True)


def dashed_line(surface: pygame.Surface, colour, points: list, width: int, dash: tuple[int, int]) -> None:
    """A polyline through ``points`` drawn as (on, off) px dashes, continued round its corners."""
    on, off = max(1, dash[0]), max(1, dash[1])
    phase = 0.0  # along the dash pattern
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        length = math.hypot(x1 - x0, y1 - y0)
        at = 0.0
        while at < length:
            inside = phase % (on + off)
            run = (on - inside) if inside < on else (on + off - inside)
            end = min(length, at + run)
            if inside < on:
                t0, t1 = at / length, end / length
                pygame.draw.line(surface, colour, (x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0),
                                 (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1), width)
            phase += end - at
            at = end


def _ticks(lo: float, hi: float, target: int) -> tuple[np.ndarray, float]:
    """Round ticks from ``lo`` to ``hi`` (none for a span that is not a positive finite number)."""
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        return np.zeros(0), 1.0
    step = nice_step(hi - lo, target)
    first = math.ceil(lo / step - 1e-9) * step
    return np.arange(first, hi + step * 1e-6, step), step


# ------------------------------------------------------------------ viewer
class SweepViewer(Messages):
    """Pygame window: the chart (or the new-sweep form) on the left, the controls on the right."""

    def __init__(self, csv: str | Path | None = None, size: tuple[int, int] | None = None,
                 results_dir: str | Path = "results", base_settings: str | Path | dict | None = None):
        dpi_aware()
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        pygame.init()
        pygame.display.set_caption("Roundabout sweep viewer")
        self.screen = open_window(size)
        pygame.key.set_repeat(350, 45)
        self.results_dir = Path(results_dir)
        self.clock = pygame.time.Clock()
        self.mouse = (-1, -1)
        self._init_messages()
        self.page = "chart"
        self.data: Sweep | None = None
        self.metric: str | None = None
        self.view = "lines"
        self.x: str | None = None
        self.series: str | None = None  # lines: one line per value
        self.y: str | None = None  # heatmap rows
        self.colour: str | None = None  # scatter dots
        self.slices: dict[str, object] = {}  # other varied settings: a value, or None = averaged over
        self.scroll = {"panel": 0, "files": 0, "params": 0}
        self.content_h = {"panel": 0, "files": 0, "params": 0}
        self.regions: dict[str, pygame.Rect] = {}
        self.files: list[Path] = []
        self.form: list[FormVar] = []
        self.seeds, self.warmup, self.measure = 3, 120.0, 300.0  # the experiments sweep defaults
        self.job: SweepRun | None = None
        self.drag: tuple[Widget, int] | None = None  # range slider and which end (0 = from, 1 = to)
        self.pressed: tuple[Widget, int] | None = None
        self.fonts: Fonts | None = None
        self.tooltip: list[str] = []  # exact values under the mouse, as last drawn
        self.plot_rect = pygame.Rect(0, 0, 0, 0)
        self.context_rect = pygame.Rect(0, 0, 0, 0)  # the chart's context lines (slices, runs, fixed settings)
        self.header_rect = pygame.Rect(0, 0, 0, 0)  # the panel's lines naming the sweep
        self._reveal_metric = False  # scroll the metric list to the metric when the chart page shows next
        self._cache: dict = {}
        self._gradients: dict = {}
        self._status_fit: tuple = ((), STATUS_LINES[0])
        self.base, self.base_seed, self.base_note = {}, None, "the defaults"
        problem = ""
        if base_settings is not None:
            where = Path(base_settings).name if isinstance(base_settings, (str, Path)) else "the main viewer"
            try:
                self.base, self.base_seed, ignored = load_base(base_settings)
                changed = len(changed_settings(self.base))
                self.base_note = f"{changed} changed from the defaults ({where})" if changed else \
                    f"the defaults ({where})"
                if ignored:
                    problem = (f"Base settings {where}: left out {', '.join(ignored)} (not parameter names); "
                               f"new sweeps start from {self.base_note}.")
            except (OSError, ValueError, KeyError, TypeError) as err:
                problem = f"Could not read the base settings {where}: {err}. New sweeps start from the defaults."
        self.layout()
        self.refresh_files()
        path = csv if csv is not None else (self.files[0] if self.files else None)
        if path is not None:
            self.load(path)
        if self.data is None:
            self.page = "new"
            if csv is None:
                self._say(f"No sweep CSV in {self.results_dir}: set one up here and run it.", 8.0)
            self._pin()  # why there is no chart (or why the CSV given could not load), wherever the mouse rests
        if problem:  # after the load message, so that it is the one shown
            self._say(problem, 10.0, pin=True)
        self.layout()

    # ------------------------------------------------------------ state
    def _say(self, text: str, seconds: float = 5.0, pin: bool = False) -> None:
        """Show ``text`` in the help box for a while; ``pin``: ahead of the hover help (see ui.Messages)."""
        self.message = (" ".join(str(text).split()), time.monotonic() + seconds)
        if pin:
            self._pin()

    def load(self, path: str | Path) -> bool:
        """Load a sweep CSV; the metric stays when the new one has it, and reloading the same file
        keeps the view, the axes and the slices."""
        try:
            data = load_sweep(path)
        except ValueError as err:
            self._say(f"Could not load {path}: {err}", 10.0)
            return False
        same = (self.data is not None and self.data.path is not None
                and Path(self.data.path).resolve() == Path(path).resolve())
        keep = (self.view, self.x, self.series, self.y, self.colour, dict(self.slices))
        self.data, self._cache = data, {}
        order = axis_order(data)
        self.x = order[0]
        others = order[1:]
        self.series = next((n for n in others if len(data.var(n).values) <= MAX_SERIES), None)
        self.y = others[0] if others else None
        self.colour = others[0] if others else None
        self.slices = {}
        self.view = "scatter" if data.design == "sample" else "lines"
        if same and keep[1] in order:  # a reload: keep what still fits
            view, self.x, series, y, colour, slices = keep
            self.series = series if series is None or (series in order and series != self.x and
                                                       len(data.var(series).values) <= MAX_SERIES) else None
            self.y = y if y in order and y != self.x else self.y if self.y != self.x else None
            self.colour = colour if colour is None or (colour in order and colour != self.x) else None
            self.slices = {k: v for k, v in slices.items() if k in order and (v is None or any(
                matches(pd.Series([v]), c)[0] for c in data.var(k).values))}
            self.view = view if view in self.views() else self.view
        def has_values(metric: str) -> bool:
            return bool(np.isfinite(pd.to_numeric(data.frame[metric], errors="coerce").to_numpy(float)).any())

        # Another file keeps the metric only where it has values (entry_vc of an open-mode sweep has
        # none in a closed-mode one); else flux, or the first metric that has some.
        if self.metric not in data.metrics or (not same and not has_values(self.metric)):
            filled = [m for m in data.metrics if has_values(m)] or data.metrics
            self.metric = "flux" if "flux" in filled else filled[0]
        if not same:
            self.scroll["panel"] = 0
        self._say(f"Loaded {data.path.name} ({plural(data.runs, 'run')}, {DESIGN_TEXT[data.design]}"
                  f"{', no meta file' if not data.meta else ''}) from {data.path.parent}")
        self._reveal_metric = True  # the metric's row in sight, wherever it is in the list (now or on the chart page)
        if self.fonts is not None:
            self.layout()
            self._reveal()
        return True

    def reload(self) -> None:
        if self.data is not None and self.data.path is not None:
            self.load(self.data.path)

    def refresh_files(self) -> None:
        self.files = find_sweeps(self.results_dir)
        if self.fonts is not None:
            self._layout_panel()

    def views(self) -> list[str]:
        """Views that make sense for the loaded sweep."""
        d = self.data
        if d is None:
            return []
        if len(d.variables) < 2:  # one setting (a random sample of one too): its runs at each value
            return ["lines", "scatter"]
        if d.design == "sample":
            return ["scatter"]
        return ["lines", "heatmap", "scatter"]

    def set_view(self, view: str) -> None:
        if view not in self.views():
            why = "a random sample of several settings has no grid to draw lines or cells on" if self.data and \
                self.data.design == "sample" and len(self.data.variables) > 1 else "it needs two varied settings"
            self._say(f"No {view} view for this sweep: {why}.")
            return
        self.view = view
        if view == "heatmap" and self.y in (None, self.x):
            self.y = next(v.name for v in self.data.variables if v.name != self.x)
        self._layout_panel()

    def toggle_heatmap(self) -> None:
        self.set_view("lines" if self.view == "heatmap" else "heatmap")

    def second_options(self) -> list[str | None]:
        """Choices for the second setting of the current view (None = none)."""
        d = self.data
        others = [v for v in d.variables if v.name != self.x] if d else []
        if self.view == "lines":
            return [None] + [v.name for v in others if len(v.values) <= MAX_SERIES]
        if self.view == "heatmap":
            return [v.name for v in others]
        return [None] + [v.name for v in others]

    @property
    def second(self) -> str | None:
        return {"lines": self.series, "heatmap": self.y, "scatter": self.colour}[self.view]

    @second.setter
    def second(self, name: str | None) -> None:
        setattr(self, {"lines": "series", "heatmap": "y", "scatter": "colour"}[self.view], name)

    def slice_vars(self) -> list[Variable]:
        """Varied settings neither on an axis nor the series: each is sliced or averaged over."""
        if self.data is None or self.view == "scatter":
            return []
        return [v for v in self.data.variables if v.name not in (self.x, self.second)]

    def current_slices(self) -> dict:
        return {v.name: self.slices.get(v.name) for v in self.slice_vars()}

    def step_metric(self, d: int) -> None:
        if self.data is None:
            return
        metrics = self.data.metrics
        self.set_metric(metrics[(metrics.index(self.metric) + d) % len(metrics)])
        self._scroll_to_metric()

    def set_metric(self, name: str) -> None:
        self.metric = name
        self._layout_panel()

    def step_x(self, d: int) -> None:
        if self.data is None:
            return
        names = [v.name for v in self.data.variables]
        self.set_x(names[(names.index(self.x) + d) % len(names)])

    def set_x(self, name: str) -> None:
        old, self.x = self.x, name
        for attr in ("series", "y", "colour"):  # the old X takes the place of a second setting it displaces
            if getattr(self, attr) == name:
                setattr(self, attr, old)
        if self.series is not None and len(self.data.var(self.series).values) > MAX_SERIES:
            self.series = None
        self._layout_panel()

    def step_second(self, d: int) -> None:
        if self.data is None:
            return
        options = self.second_options()
        if not options:
            return
        i = options.index(self.second) if self.second in options else 0
        self.second = options[(i + d) % len(options)]
        self._layout_panel()

    def step_slice(self, name: str, d: int) -> None:
        options = [None, *self.data.var(name).values]
        current = self.slices.get(name)
        i = next((k for k, v in enumerate(options) if (v is None and current is None) or
                  (v is not None and current is not None and v == current)), 0)
        self.slices[name] = options[(i + d) % len(options)]
        self._layout_panel()

    def show_page(self, page: str) -> None:
        if page == "files" and self.page != "files":
            self.refresh_files()
        self.page = page
        self.layout()
        self._reveal()

    # ------------------------------------------------------------ new sweep
    def role(self, i: int) -> str:
        return ("x axis", "series", "3rd setting")[i]

    def toggle_param(self, name: str) -> None:
        names = [v.name for v in self.form]
        if name in names:
            self.remove_var(names.index(name))
            return
        if len(self.form) >= MAX_VARIABLES:
            self._say(f"A sweep takes at most {MAX_VARIABLES} settings: remove one first.")
            return
        self.form.append(FormVar.new(name))
        self._cap_points()
        self.layout()

    def remove_var(self, i: int) -> None:
        del self.form[i]
        self._cap_points()
        self.layout()

    def _cap_points(self) -> None:
        """Keep each setting within the points its role allows (the 2nd draws a line per value)."""
        for i, var in enumerate(self.form):
            cap = var.max_points(MAX_POINTS[i])
            if var.spec.choices:
                var.chosen = [c for c in var.spec.choices if c in var.chosen][:cap] or [var.spec.choices[0]]
            else:
                var.points = int(np.clip(var.points, min(2, cap), cap))

    def idle_reason(self, name: str) -> str:
        """Why the setting ``name`` would change nothing in any run of the sweep set up here ("" if it acts
        in some): from the base settings (config.inactive_reason), unless another picked setting makes it
        act (Traffic mode with open for the open-mode settings, Arms above 0, Truck share above 0)."""
        try:
            base = SimConfig().updated(**{k: v for k, v in self.base.items() if k in SETTINGS})
        except (ValueError, TypeError):
            return ""
        reason = inactive_reason(name, base)
        if not reason:
            return ""
        for var in self.form:
            if var.name == name:
                continue
            for value in var.values():
                try:
                    if not inactive_reason(name, base.updated(**{var.name: value})):
                        return ""
                except (ValueError, TypeError):
                    continue
        return reason

    def form_variables(self) -> list[tuple[str, list]]:
        return [(var.name, var.values()) for var in self.form]

    def _new_prefix(self) -> Path:
        """results/sweep_<date>_<time>: no file the sweep writes is there yet (state.new_paths)."""
        csv, *_ = state.new_paths(self.results_dir.resolve(), "sweep", ".csv", "_meta.json", ".png")
        return csv.with_suffix("")

    def command_preview(self) -> str:
        """The command the Run button starts, as one would type it."""
        if not self.form:
            return "Pick a setting on the left to see the command."
        try:
            args = sweep_args(self.form_variables(), seeds=self.seeds, warmup=self.warmup, measure=self.measure,
                              out=self.results_dir / "sweep_<date>_<time>", base=self.base, seed=self.base_seed or 0)
        except ValueError as err:
            return f"Cannot run: {err}"
        return display_command(["python", "-m", "roundabout_sim.experiments", *args])

    def running(self) -> bool:
        return self.job is not None and self.job.state == "running"

    def start_sweep(self, argv: list[str] | None = None, out: str | Path | None = None) -> bool:
        """Run the sweep set up in the form (or ``argv``, writing ``out``.csv) without blocking the window."""
        if self.running():
            self._say("A sweep is already running: cancel it first.")
            return False
        if argv is None and not self.form:
            self._say("Pick at least one setting to sweep.")
            return False
        try:
            out = Path(out) if out is not None else self._new_prefix()
            if argv is None:
                argv = sweep_command(self.form_variables(), seeds=self.seeds, warmup=self.warmup,
                                     measure=self.measure, out=out, base=self.base, seed=self.base_seed or 0)
        except (ValueError, OSError) as err:
            self._say(f"Cannot run the sweep: {err}")
            return False
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            self.job = SweepRun(argv, out)
        except OSError as err:
            self._say(f"Could not start the sweep: {err}", 10.0)
            return False
        self._say(f"Sweep started; it writes {out.name}.csv into {out.parent} when it is done.", 6.0)
        self.layout()
        return True

    def cancel_sweep(self) -> None:
        if self.running():
            self.job.cancel()
            self.update()

    def update(self) -> None:
        """Follow a running sweep; load its CSV when it has finished."""
        job = self.job
        if job is None or job.handled or job.poll() == "running":
            return
        job.handled = True
        if job.state == "done":
            csv = Path(f"{job.out}.csv")
            if csv.exists() and self.load(csv):
                self.page = "chart"
                self._say(f"Sweep finished in {job.elapsed:.0f} s: loaded {csv.name} ({plural(self.data.runs, 'run')}) "
                          f"from {csv.parent}", 8.0, pin=True)
            else:
                job.state = "failed"
                why = "it is not a sweep CSV" if csv.exists() else "it was not written"
                job.errors.append(f"The sweep finished, but {csv.name} could not be loaded: {why}.")
                self._say(job.errors[-1], 15.0, pin=True)
        elif job.state == "cancelled":
            # experiments writes the CSV (atomically) once every run is done, then draws its figures: a
            # sweep cancelled while drawing has its whole CSV, so load it.
            csv = Path(f"{job.out}.csv")
            if not csv.exists():
                job.remove_partial()  # (a meta file or a half-written CSV, if it was stopped while saving)
                self._say(f"Sweep cancelled {cancelled_when(job)} (nothing was kept).")
            elif self.load(csv):  # (else load says why it could not)
                job.saved, self.page = True, "chart"
                self._say(f"Sweep cancelled while drawing its figures: all {plural(self.data.runs, 'run')} were "
                          f"already saved to {csv.name} (no PNG), loaded from {csv.parent}", 8.0, pin=True)
        elif job.state == "failed":  # the error line only: the traceback is on the New sweep tab
            self._say(f"Sweep failed (exit {job.proc.returncode}): {job.error_line()}", 15.0, pin=True)
        self.refresh_files()
        self.layout()
        self._reveal()  # (a sweep that finished on another page opens the chart on its CSV)

    # ------------------------------------------------------------ files
    def save_png(self) -> Path | None:
        """The chart card (the whole window on the other pages) as a PNG in the results folder,
        drawn without the hover readout; None (and a message) if it cannot be written."""
        stem = Path(self.data.path).stem[:40] if self.data is not None and self.data.path else "sweep"
        what = f"_{self.view}_{self.metric}" if self.page == "chart" and self.data is not None else f"_{self.page}"
        mouse, self.mouse = self.mouse, (-1, -1)
        self.draw()
        self.mouse = mouse
        area = self.main_rect if self.page == "chart" and self.data is not None else self.screen.get_rect()
        try:
            path, = state.new_paths(self.results_dir, f"sweepview_{stem}{what}", ".png")
            pygame.image.save(self.screen.subsurface(area).copy(), str(path))
        except (pygame.error, OSError) as err:
            self._say(f"Could not save the PNG into {self.results_dir}: {err}", 10.0)
            return None
        self._say(f"Saved {path.name} into {path.parent}")
        return path

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
        self.main_rect = pygame.Rect(m, m, W - panel_w - 2 * m, H - 2 * m)
        self._layout_panel()

    def _row_h(self) -> int:
        return self.fonts.line("base") + round(8 * self.s)

    def _layout_panel(self) -> None:
        f, s, r = self.fonts, self.s, self.panel_rect
        p = round(14 * s)
        x, w = r.x + p, r.w - 2 * p
        self.widgets: list[Widget] = []
        y = r.y + p + f.line("title") + 2 * f.line("small") + round(8 * s)
        if self.job is not None and self.page != "new":  # the running sweep's progress, on every page
            self.widgets.append(Widget("readout", "progress", pygame.Rect(x, y, w, self._progress_h()),
                                       self._progress_help()))
            y += f.line("small") + round(10 * s)
        th = self._row_h()
        for i, (page, label) in enumerate(PAGES):
            left, right = round(x + i * w / len(PAGES)), round(x + (i + 1) * w / len(PAGES))
            rect = pygame.Rect(left, y, right - left, th)
            self.widgets.append(Widget("tab", page, rect, TAB_HELP[page], label=label))
        y += th + round(10 * s)
        self.status_lines = self._status_lines(w)
        status_h = self.status_lines * f.line("small") + round(12 * s)
        self.status_rect = pygame.Rect(x, r.bottom - p - status_h, w, status_h)
        bottom = self.status_rect.y - round(8 * s)
        if self.page == "chart":
            self._layout_chart_controls(x, y, w, bottom)
        elif self.page == "files":
            self._layout_files(x, y, w, bottom)
        else:
            self._layout_new_controls(x, y, w)
            self._layout_form()

    def _buttons(self, names: list[tuple[str, str]], x: int, y: int, w: int) -> int:
        """Buttons sharing out the width ``w`` in one row; returns the y below them."""
        bh = self._row_h()
        for (name, label), rect in zip(names, row_rects(x, y, w, bh, len(names), round(6 * self.s))):
            self.widgets.append(Widget("button", name, rect, BUTTON_HELP[name], label=label))
        return y + bh + round(10 * self.s)

    def _cycler(self, name: str, caption: str, text: str, x: int, y: int, w: int, help_text: str,
                enabled: bool = True, region: str | None = None) -> int:
        """A caption line (if any), then ◀ value ▶ (click the arrows or the value)."""
        f, s = self.fonts, self.s
        top = y + (f.line("base") + round(2 * s) if caption else 0)
        bh = f.line("small") + round(8 * s)
        rect = pygame.Rect(x, y, w, top + bh - y)
        arrows = [pygame.Rect(x, top, bh + round(4 * s), bh), pygame.Rect(x + w - bh - round(4 * s), top,
                                                                            bh + round(4 * s), bh)]
        self.widgets.append(Widget("cycler", name, rect, help_text, region, arrows, caption,
                                   data={"text": text, "enabled": enabled, "row": pygame.Rect(x, top, w, bh)}))
        return rect.bottom + round(8 * s)

    def _layout_chart_controls(self, x: int, y: int, w: int, bottom: int) -> None:
        f, s, d = self.fonts, self.s, self.data
        y = self._buttons([("save", "Save PNG"), ("reload", "Reload")], x, y, w)
        if d is None:
            self.regions = {}
            return
        top = y + f.line("base") + round(2 * s)
        oh = f.line("small") + round(8 * s)
        options = [pygame.Rect(round(x + i * w / len(VIEWS)), top, round(w / len(VIEWS)) - 2, oh)
                   for i in range(len(VIEWS))]
        self.widgets.append(Widget("choice", "view", pygame.Rect(x, y, w, top + oh - y), VIEW_CHOICE_HELP,
                                   options=options, label="View", data=list(VIEWS)))
        y = top + oh + round(10 * s)
        names = [v.name for v in d.variables]
        y = self._cycler("x", "X axis", d.var(self.x).label, x, y, w, X_HELP, enabled=len(names) > 1)
        caption = {"lines": "Series (colour)", "heatmap": "Y axis", "scatter": "Colour"}[self.view]
        second = self.second
        text = d.var(second).label if second else ("none" if self.second_options() else "none (one setting)")
        y = self._cycler("second", caption, text, x, y, w, SECOND_HELP[self.view],
                         enabled=len(self.second_options()) > 1)
        # Scrolling: the slice controls and the metric list.
        region = pygame.Rect(x, y, w, max(bottom - y, 10))
        self.regions = {"panel": region}
        cy = y - self.scroll["panel"]
        start = cy
        lh = f.line("small") + round(6 * s)
        slices = self.slice_vars()
        if slices:
            self.widgets.append(Widget("heading", "slices", pygame.Rect(x, cy, w, lh), region="panel",
                                       label="Other varied settings"))
            cy += lh
            for v in slices:
                value = self.slices.get(v.name)
                if value is None:
                    shown = ", ".join(v.text(c) for c in v.values) if len(v.values) <= 4 else \
                        f"{len(v.values)} values"
                    text = f"average over {shown}"
                else:
                    text = f"= {v.text(value)}"
                cy = self._cycler(f"slice:{v.name}", v.label, text, x, cy, w,
                                  SLICE_HELP.format(label=v.label), region="panel")
        self.widgets.append(Widget("heading", "metrics", pygame.Rect(x, cy, w, lh), region="panel",
                                   label=f"Metric ({len(d.metrics)} columns; Up/Down)"))
        cy += lh
        row = f.line("small") + round(6 * s)
        for name in d.metrics:
            info = metric_info(name, self.x, d.open_demand)
            self.widgets.append(Widget("metric", name, pygame.Rect(x, cy, w, row), info.help, "panel",
                                       label=info.label))
            cy += row
        if d.text_metrics:
            note = f"Text columns ({', '.join(map(column_label, d.text_metrics))}) are counted in the hover readout."
            for line in f.wrap(note, w - round(8 * s)):
                self.widgets.append(Widget("heading", "text", pygame.Rect(x, cy, w, f.line("small")),
                                           region="panel", label=line))
                cy += f.line("small")
        self._fit_scroll("panel", cy - start)

    def _fit_scroll(self, region: str, height: int) -> None:
        """Clamp a region's scroll offset; lay out again if it moved."""
        self.content_h[region] = height
        limit = max(0, height - self.regions[region].h)
        clamped = int(np.clip(self.scroll[region], 0, limit))
        if clamped != self.scroll[region]:
            self.scroll[region] = clamped
            self._layout_panel()

    def _reveal(self) -> None:
        """Bring the metric's row into sight if a CSV loaded since the chart page last showed (its
        list exists only there)."""
        if self._reveal_metric and self.page == "chart":
            self._reveal_metric = False
            self._scroll_to_metric()

    def _scroll_to_metric(self) -> None:
        w = next((w for w in self.widgets if w.kind == "metric" and w.name == self.metric), None)
        region = self.regions.get("panel")
        if w is None or region is None:
            return
        if w.rect.top < region.top:
            self.scroll["panel"] -= region.top - w.rect.top
        elif w.rect.bottom > region.bottom:
            self.scroll["panel"] += w.rect.bottom - region.bottom
        else:
            return
        self._layout_panel()

    def _layout_files(self, x: int, y: int, w: int, bottom: int) -> None:
        f, s = self.fonts, self.s
        y = self._buttons([("refresh", "Refresh"), ("newest", "Load newest")], x, y, w)
        folder = f"Folder: {self.results_dir.resolve()}"
        self.folder_lines = wrap_lines(f, folder, w, 2)
        y += len(self.folder_lines) * f.line("small") + round(8 * s)
        region = pygame.Rect(x, y, w, max(bottom - y, 10))
        self.regions = {"files": region}
        row = 2 * f.line("small") + round(8 * s)
        cy = y - self.scroll["files"]
        for path in self.files:
            try:
                when = time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))
            except OSError:
                continue
            where = path.name if path.parent == self.results_dir else f"{path.parent.name}/{path.name}"
            self.widgets.append(Widget("file", str(path), pygame.Rect(x, cy, w, row - round(2 * s)),
                                       FILE_HELP.format(name=where, when=when), "files", label=where, data=when))
            cy += row
        self._fit_scroll("files", cy - (y - self.scroll["files"]))

    def _layout_new_controls(self, x: int, y: int, w: int) -> None:
        running = self.running()
        y = self._buttons([("run", "Run sweep"), ("cancel", "Cancel")], x, y, w)
        for w_ in self.widgets[-2:]:
            w_.data = {"disabled": running if w_.name == "run" else not running}
        y = self._cycler("seeds", "Seeds per point", str(self.seeds), x, y, w, STEPPER_HELP["seeds"])
        y = self._cycler("warmup", "Warm-up", f"{self.warmup:g} s", x, y, w, STEPPER_HELP["warmup"])
        y = self._cycler("measure", "Measured", f"{self.measure:g} s", x, y, w, STEPPER_HELP["measure"])
        self.regions = {}
        f, s = self.fonts, self.s
        if self.form:
            counts = [len(v) for _, v in self.form_variables()]
            points = math.prod(counts)
            product = f"{' × '.join(map(str, counts))} = " if len(counts) > 1 else ""  # (one setting: its points)
            what = (f"{product}{plural(points, 'point')} × {plural(self.seeds, 'seed')} = "
                    f"{plural(points * self.seeds, 'run')}, each {self.warmup:g} s warm-up + {self.measure:g} s "
                    "measured")
        else:
            what = "No settings picked yet."
        self.summary_lines = wrap_lines(f, what, w, 3)
        self.widgets.append(Widget("readout", "summary",
                                   pygame.Rect(x, y, w, len(self.summary_lines) * f.line("small")), SUMMARY_HELP))
        self.new_text_y = y
        y += len(self.summary_lines) * f.line("small") + round(10 * s)  # the progress and its detail below
        bottom = self.status_rect.y - round(8 * s)
        if self.job is not None and bottom > y:
            self.widgets.append(Widget("readout", "job", pygame.Rect(x, y, w, bottom - y), self._progress_help()))

    def _progress_h(self) -> int:
        """Height of the progress bar with its line of text."""
        return max(3, round(5 * self.s)) + round(3 * self.s) + self.fonts.line("small")

    def _progress_help(self) -> str:
        job = self.job
        return JOB_HELP["saved" if job.saved else job.state].format(out=job.out, command=display_command(job.argv))

    def _layout_form(self) -> None:
        """New sweep: the parameter list on the left of the card, the chosen settings on the right."""
        f, s, card = self.fonts, self.s, self.main_rect
        pad = round(14 * s)
        x, y = card.x + pad, card.y + pad
        self.form_intro = wrap_lines(f, FORM_HELP, card.w - 2 * pad, 3)
        y += f.line("title") + len(self.form_intro) * f.line("small") + round(6 * s)
        base = f"Base settings: {self.base_note}"
        if self.base_seed is not None:
            base += f", seed {self.base_seed}"
        self.widgets.append(Widget("info", "base", pygame.Rect(x, y, card.w - 2 * pad, f.line("small")),
                                   BASE_HELP, label=base))
        y += f.line("small") + round(10 * s)
        left_w = int(np.clip(0.36 * card.w, 190 * s, 320 * s))
        region = pygame.Rect(x, y, left_w, card.bottom - pad - y)
        self.regions["params"] = region
        row = f.line("small") + round(6 * s)
        groups = list(dict.fromkeys(spec.group for spec in PARAMS))
        height = (len(groups) + len(PARAMS)) * row + (len(groups) - 1) * round(4 * s)
        self.content_h["params"] = height
        self.scroll["params"] = int(np.clip(self.scroll["params"], 0, max(0, height - region.h)))
        cy = y - self.scroll["params"]
        start = cy
        chosen = [v.name for v in self.form]
        self.idle = {spec.name: self.idle_reason(spec.name) for spec in PARAMS}
        group = None
        for spec in PARAMS:
            if spec.group != group:
                group = spec.group
                if cy != start:
                    cy += round(4 * s)
                self.widgets.append(Widget("heading", f"group:{group}", pygame.Rect(x, cy, left_w, row),
                                           region="params", label=group))
                cy += row
            self.widgets.append(Widget("param", spec.name, pygame.Rect(x, cy, left_w - round(8 * s), row),
                                       param_help(spec, spec.name in chosen, self.idle[spec.name]), "params",
                                       label=spec.label, data=self.idle[spec.name]))
            cy += row
        # The chosen settings (or how to choose them).
        rx = region.right + round(20 * s)
        rw = card.right - pad - rx
        self.form_right = pygame.Rect(rx, y, rw, card.bottom - pad - y)
        self.pick_lines = [] if self.form else wrap_lines(f, PICK_HINT, rw, 3, "base")
        cy = y + len(self.pick_lines) * f.line("base") + (round(10 * s) if self.pick_lines else 0)
        for i, var in enumerate(self.form):
            cy = self._layout_var(i, var, rx, cy, rw) + round(10 * s)
        self.form_blocks_bottom = cy
        # A picked setting that would change nothing in any run: say so under the blocks.
        warning = idle_text({setting_label(v.name): self.idle.get(v.name, "") for v in self.form})
        self.idle_lines = (cy, wrap_lines(f, warning, rw, 3) if warning else [])
        if warning:
            self.widgets.append(Widget("readout", "idle", pygame.Rect(rx, cy, rw, len(self.idle_lines[1]) * f.line(
                "small")), warning))
            cy += len(self.idle_lines[1]) * f.line("small") + round(4 * s)
        # Below them every base setting that is not swept, then the command at the bottom of the column:
        # each one whole while they fit, else the base settings keep at least as many lines as the
        # command keeps (cut with an ellipsis).
        lh, head, gap = f.line("small"), f.line("small") + round(2 * s), round(10 * s)
        swept = {var.name for var in self.form}
        base = ", ".join(setting_phrase(k, v) for k, v in changed_settings(self.base).items() if k not in swept)
        base_all = f.wrap(base, rw) if base else []
        command = self.command_preview()
        command_all = f.wrap(command, rw)
        top = cy + round(6 * s)
        room = max(0, (self.form_right.bottom - top - head - (head + gap if base_all else 0)) // lh)
        command_n = min(len(command_all), max(room - len(base_all), min(room // 2, 3) if base_all else room))
        base_n = min(len(base_all), room - command_n)
        base_lines = wrap_lines(f, base, rw, base_n) if base_n else []
        command_lines = wrap_lines(f, command, rw, command_n) if command_n else []
        self.base_lines = (top, base_lines)
        if base_lines:
            self.widgets.append(Widget("readout", "base_list", pygame.Rect(rx, top, rw, head + len(base_lines) * lh),
                                       BASE_HELP))
        bottom = self.form_right.bottom - head - len(command_lines) * lh
        top = max(top + (head + len(base_lines) * lh + gap if base_lines else 0), bottom)
        self.command_lines = (top, command_lines)
        self.widgets.append(Widget("readout", "command",
                                   pygame.Rect(rx, top, rw, head + len(command_lines) * lh), COMMAND_HELP))

    def _layout_var(self, i: int, var: FormVar, x: int, y: int, w: int) -> int:
        f, s = self.fonts, self.s
        spec = var.spec
        head = f.line("base") + round(4 * s)
        size = f.line("small") + round(6 * s)
        self.widgets.append(Widget("remove", str(i), pygame.Rect(x + w - size, y + (head - size) // 2, size, size),
                                   REMOVE_HELP))
        block = Widget("block", str(i), pygame.Rect(x, y, w, head), spec.help, label=setting_label(var.name),
                       data=self.role(i))
        self.widgets.append(block)
        y += head + round(4 * s)
        cap = MAX_POINTS[i]
        if spec.choices:
            n = len(spec.choices)
            oh = f.line("small") + round(8 * s)
            options = [pygame.Rect(round(x + k * w / n), y, round(w / n) - 2, oh) for k in range(n)]
            self.widgets.append(Widget("toggle", str(i), pygame.Rect(x, y, w, oh), TOGGLE_HELP, options=options))
            return y + oh + round(4 * s)
        line = f.line("small")
        knob = round(7 * s)
        track = pygame.Rect(x + knob, y + line + round(8 * s), w - 2 * knob, 4)
        self.widgets.append(Widget("range", str(i), pygame.Rect(x, y, w, line + round(18 * s)), RANGE_HELP,
                                   options=[track]))
        y += line + round(20 * s)
        role = {0: "x axis (1st setting)", 1: "series (one line per value)", 2: "3rd setting"}[i]
        return self._cycler(f"points:{i}", "", str(var.points), x, y, min(w // 2, round(200 * s)),
                            POINTS_HELP.format(cap=cap, role=role), enabled=var.max_points(cap) > 2) - round(4 * s)

    def _status_lines(self, width: int) -> int:
        """Lines the help box needs so that the longest help text fits ``width`` (see ui.status_lines:
        the box keeps its height whatever is loaded)."""
        self._status_fit = status_lines(self.fonts, width, help_texts, self._status_fit)
        return self._status_fit[1]

    # ------------------------------------------------------------ events
    def handle(self, e: pygame.event.Event) -> bool:
        """Process one event; False means quit."""
        if e.type == pygame.QUIT:
            return False
        if e.type == pygame.VIDEORESIZE:
            self.screen = clamp_resize(self.screen, e.size)
            self.layout()
        elif e.type == pygame.KEYDOWN:  # a message the key gives shows wherever the mouse rests
            return self._by_key(lambda: self._key(e))
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
        elif e.type == pygame.MOUSEWHEEL:
            for name, rect in self.regions.items():
                if rect.collidepoint(self.mouse):
                    self.scroll[name] -= int(e.y * 40 * self.s)
                    self._layout_panel()
                    break
        return True

    def _key(self, e: pygame.event.Event) -> bool:
        k = e.key
        chart = self.page == "chart"  # the chart keys act only where the chart shows
        if k == pygame.K_ESCAPE:
            return False
        if k == pygame.K_s:
            self.save_png()
        elif k == pygame.K_h:
            if self.data is not None and chart:
                self.toggle_heatmap()
        elif k in (pygame.K_UP, pygame.K_DOWN):
            if chart:
                self.step_metric(-1 if k == pygame.K_UP else 1)
        elif k in (pygame.K_LEFT, pygame.K_RIGHT):
            if chart:
                self.step_x(-1 if k == pygame.K_LEFT else 1)
        elif k == pygame.K_TAB:
            if chart:
                self.step_second(-1 if getattr(e, "mod", 0) & pygame.KMOD_SHIFT else 1)
        elif k in (pygame.K_r, pygame.K_F5):
            self.reload()
        elif k == pygame.K_o:
            self.show_page("chart" if self.page == "files" else "files")
        elif k == pygame.K_n:
            self.show_page("chart" if self.page == "new" else "new")
        return True

    def _visible(self, w: Widget, pos) -> bool:
        return w.region is None or (w.region in self.regions and self.regions[w.region].collidepoint(pos))

    def widget_at(self, pos) -> Widget | None:
        return next((w for w in self.widgets if w.kind not in ("heading", "block") and w.rect.collidepoint(pos)
                     and self._visible(w, pos)), None)

    def _press(self, pos) -> None:
        w = self.widget_at(pos)
        if w is None:
            return
        if w.kind == "range":
            track = w.options[0]
            if not track.inflate(2 * round(10 * self.s), round(24 * self.s)).collidepoint(pos):
                return
            var = self.form[int(w.name)]
            spec = var.spec
            ends = [track.x + (v - spec.lo) / (spec.hi - spec.lo) * track.w for v in (var.lo, var.hi)]
            end = int(abs(pos[0] - ends[1]) < abs(pos[0] - ends[0]) or (ends[0] == ends[1] and pos[0] > ends[1]))
            self.drag = (w, end)
            self._drag_to(pos[0])
            return
        if w.kind in ("choice", "toggle", "cycler"):
            hit = next((i for i, o in enumerate(w.options) if o.collidepoint(pos)), None)
            if hit is None and w.kind == "cycler" and w.data["row"].collidepoint(pos):
                hit = 1  # the value itself steps forward
            if hit is not None:
                self.pressed = (w, hit)
            return
        self.pressed = (w, 0)

    def _drag_to(self, px: int) -> None:
        w, end = self.drag
        var = self.form[int(w.name)]
        spec, track = var.spec, w.options[0]
        t = float(np.clip((px - track.x) / track.w, 0.0, 1.0))
        value = round(float(np.clip(spec.lo + round(t * (spec.hi - spec.lo) / spec.step) * spec.step,
                                    spec.lo, spec.hi)), 6)
        if end == 0:
            var.lo = min(value, round(var.hi - spec.step, 6))
        else:
            var.hi = max(value, round(var.lo + spec.step, 6))
        var.lo, var.hi = max(var.lo, float(spec.lo)), min(var.hi, float(spec.hi))
        self._cap_points()
        self._layout_panel()

    def _release(self, pos) -> None:
        if self.drag:
            self.drag = None
            return
        if self.pressed is None:
            return
        w, i = self.pressed
        self.pressed = None
        current = next((c for c in self.widgets if c.kind == w.kind and c.name == w.name), w)
        target = current.options[i] if current.kind in ("choice", "toggle") else current.rect
        if not target.collidepoint(pos):
            return
        self._clicked_on(w)  # a message the click gives shows while the mouse stays on the control
        if w.kind == "button":
            self.action(w.name)
        elif w.kind == "tab":
            self.show_page(w.name)
        elif w.kind == "choice":
            self.set_view(VIEWS[i])
        elif w.kind == "cycler":
            if current.data.get("enabled", True):
                self.cycle(w.name, -1 if i == 0 else 1)
        elif w.kind == "metric":
            self.set_metric(w.name)
        elif w.kind == "file":
            if self.load(w.name):
                self.layout()
        elif w.kind == "param":
            self.toggle_param(w.name)
        elif w.kind == "remove":
            self.remove_var(int(w.name))
        elif w.kind == "toggle":
            self._toggle_choice(int(w.name), i)

    def _toggle_choice(self, k: int, i: int) -> None:
        var = self.form[k]
        choice = var.spec.choices[i]
        if choice in var.chosen:
            if len(var.chosen) > 1:
                var.chosen.remove(choice)
        else:
            var.chosen.append(choice)
        self._cap_points()
        self._layout_panel()

    def cycle(self, name: str, d: int) -> None:
        if name == "x":
            self.step_x(d)
        elif name == "second":
            self.step_second(d)
        elif name.startswith("slice:"):
            self.step_slice(name.split(":", 1)[1], d)
        elif name.startswith("points:"):
            i = int(name.split(":", 1)[1])
            var = self.form[i]
            cap = var.max_points(MAX_POINTS[i])
            low = min(2, cap)
            var.points = low + (var.points - low + d) % (cap - low + 1)
            self._layout_panel()
        elif name in ("seeds", "warmup", "measure"):
            choices = {"seeds": SEED_CHOICES, "warmup": WARMUP_CHOICES, "measure": MEASURE_CHOICES}[name]
            value = getattr(self, name)
            i = int(np.argmin([abs(c - value) for c in choices]))
            setattr(self, name, choices[(i + d) % len(choices)])
            self._layout_panel()

    def action(self, name: str) -> None:
        if name == "save":
            self.save_png()
        elif name == "reload":
            self.reload()
        elif name == "refresh":
            self.refresh_files()
            self._say(f"{len(self.files)} sweep CSV(s) in {self.results_dir}.")
        elif name == "newest":
            self.refresh_files()
            if self.files:
                self.load(self.files[0])
            else:
                self._say(f"No sweep CSV in {self.results_dir}.")
        elif name == "run":
            self.start_sweep()
        elif name == "cancel":
            self.cancel_sweep()

    # ------------------------------------------------------------ loop
    def run(self) -> None:
        try:
            running = True
            while running:
                for event in pygame.event.get():
                    running = self.handle(event) and running
                if not running:
                    break
                if self.screen.get_size() != self._layout_size:
                    self.layout()
                self.update()
                self.draw()
                pygame.display.flip()
                self.clock.tick(FPS_CAP)
        finally:  # whatever ends the loop, a running sweep is not left behind
            self.close()
            pygame.quit()

    def close(self) -> None:
        """Stop a sweep that is still running (it would only be orphaned)."""
        if self.running():
            self.job.cancel()
            self.job.poll()

    # ------------------------------------------------------------ drawing
    def draw(self) -> None:
        self.screen.fill(UI["surface"])
        self.tooltip = []
        if self.page == "new":
            self._draw_form()
        elif self.data is None:
            self._card(self.main_rect)
            self.fonts.blit(self.screen, EMPTY_HINT, self.main_rect.center, "base", UI["text2"], "center")
        else:
            self._draw_chart()
        self._draw_panel()
        self._draw_tooltip()

    def _card(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, UI["background"], rect, border_radius=round(6 * self.s))

    def _gradient(self, kind: str, w: int, h: int) -> pygame.Surface:
        """Vertical colour bar, high values at the top."""
        key = (kind, w, h)
        if key not in self._gradients:
            t = np.linspace(1.0, 0.0, max(h, 2))
            rgb = (heat_rgb if kind == "heat" else dot_rgb)(t)
            self._gradients[key] = pygame.surfarray.make_surface(
                np.repeat(rgb[None, :, :], max(w, 1), axis=0).astype(np.uint8))
        return self._gradients[key]

    def _stats(self, metric: str, x: str, y: str | None, slices: dict) -> Stats:
        key = ("grid", metric, x, y, tuple(sorted((k, str(v)) for k, v in slices.items())))
        if key not in self._cache:
            d = self.data
            self._cache[key] = grid_stats(select(d.frame, slices), metric, d.var(x), d.var(y) if y else None)
        return self._cache[key]

    def _context(self, stats: Stats | None) -> str:
        """Second header line: slices, averages, runs per point, run lengths and the fixed settings."""
        d = self.data
        parts = []
        for v in self.slice_vars():
            value = self.slices.get(v.name)
            if value is None:
                shown = ", ".join(v.text(c) for c in v.values) if len(v.values) <= 4 else f"{len(v.values)} values"
                parts.append(f"averaged over {v.label} ({shown})")
            else:
                parts.append(setting_phrase(v.name, value))
        if self.view == "scatter":
            parts.append(f"{plural(d.runs, 'run')} ({DESIGN_TEXT[d.design]})")
        elif stats is not None and stats.runs.size:
            k = stats.runs[stats.runs > 0]
            if k.size:
                parts.append(f"{plural(k.min(), 'run')} per point" if k.min() == k.max() else
                             f"{k.min()}-{k.max()} runs per point")
        meta = d.meta
        if _is_number(meta.get("warmup")) and _is_number(meta.get("measure")):
            parts.append(f"{meta['warmup']:g} s warm-up + {meta['measure']:g} s measured")
        if base := base_text(meta.get("preset"), meta.get("random")):  # as experiments names the base
            parts.append(base)
        changed = [setting_phrase(name, value) for name, value in changed_settings(d.fixed).items()]
        parts.append(f"fixed: {', '.join(changed)}" if changed else "other settings at their defaults")
        return "  ·  ".join(parts)

    def _draw_chart(self) -> None:
        f, s, scr, d = self.fonts, self.s, self.screen, self.data
        card = self.main_rect
        self._card(card)
        pad = round(12 * s)
        x0, y = card.x + pad, card.y + pad
        width = card.w - 2 * pad
        info = metric_info(self.metric, self.x, d.open_demand)
        x_var = d.var(self.x)
        title = f"{info.label} vs {x_var.label.split(' (')[0].lower()}" if self.view != "heatmap" else \
            f"{info.label}: {x_var.label.split(' (')[0].lower()} × {d.var(self.y).label.split(' (')[0].lower()}"
        f.blit(scr, ellipsize(f, title, width, "title"), (x0, y), "title", UI["text"])
        y += f.line("title")
        stats = None
        if self.view != "scatter":
            stats = self._stats(self.metric, self.x, self.y if self.view == "heatmap" else self.series,
                                self.current_slices())
        if info.note_for(self.view):
            for line in wrap_lines(f, info.note_for(self.view), width, 2):
                f.blit(scr, line, (x0, y), "small", UI["muted"])
                y += f.line("small")
        context, top = self._context(stats), y
        for line in wrap_lines(f, context, width, 2):
            f.blit(scr, line, (x0, y), "small", UI["text2"])
            y += f.line("small")
        self.context_rect = pygame.Rect(x0, top, width, y - top)
        if self.context_rect.collidepoint(self.mouse):  # (cut with an ellipsis when long: here all of it)
            self._info_tooltip([d.path.name if d.path else "sweep", *context.split("  ·  ")])
        y += round(6 * s)
        if self.view == "lines":
            self._draw_lines(card, y, info, stats)
        elif self.view == "heatmap":
            self._draw_heatmap(card, y, info, stats)
        else:
            self._draw_scatter(card, y, info)

    def _info_tooltip(self, parts: list[str]) -> None:
        """A box by the mouse with ``parts`` (the first one as its heading), each wrapped to a readable width
        between the items of a list ("Ring radius = 67.25 m, Density = ..."), not inside one."""
        f, width = self.fonts, min(self.main_rect.w - round(24 * self.s), round(560 * self.s))
        lines = []
        for part in parts:
            line = ""
            for item in part.split(", "):
                if line and f.width(f"{line}, {item},", "small") > width:
                    lines.append(f"{line},")
                    line = item
                else:
                    line = f"{line}, {item}" if line else item
            lines += f.wrap(line, width)  # (an item longer than a line, wrapped within it)
        self.tooltip = lines

    # ------------------------------------------------------------ chart pieces
    def _legend(self, items: list[tuple[tuple, str]], x: int, y: int, width: int, caption: str) -> int:
        """Swatch + text items after a caption, flowing onto more lines; returns the y below."""
        f, s, scr = self.fonts, self.s, self.screen
        lh = f.line("small")
        cx = f.blit(scr, caption, (x, y), "small", UI["text2"]).right + round(10 * s)
        for colour, text in items:
            need = round(18 * s) + f.width(text, "small") + round(14 * s)
            if cx > x and cx + need > x + width:
                cx, y = x, y + lh + round(2 * s)
            cy = y + lh // 2
            pygame.draw.line(scr, colour, (cx, cy), (cx + round(14 * s), cy), 3)
            cx = f.blit(scr, text, (cx + round(18 * s), y), "small", UI["text"]).right + round(14 * s)
        return y + lh + round(6 * s)

    def _y_axis(self, plot: pygame.Rect, values: np.ndarray, info: Metric) -> tuple[float, float]:
        """Grid lines and tick labels of a value axis; returns its (bottom, top)."""
        f, scr = self.fonts, self.screen
        finite = values[np.isfinite(values)]
        if finite.size:
            vmin, vmax = float(finite.min()), float(finite.max())
        else:
            vmin, vmax = 0.0, 1.0
        from_zero = info.from_zero if info.from_zero is not None else vmin >= 0
        lo = min(0.0, vmin) if from_zero else vmin
        hi = max(vmax, info.min_top if from_zero else vmax)
        if info.ref is not None:
            lo, hi = min(lo, info.ref), max(hi, info.ref)
        if hi - lo < 1e-12:
            spread = abs(hi) * 0.1 or 1.0
            hi += spread
            lo = lo if from_zero and lo == 0 else lo - spread
        target = int(np.clip(plot.h // (2.4 * f.line("small")), 2, 8))
        step = nice_step(hi - lo, target)
        lo, hi = math.floor(lo / step + 1e-9) * step, math.ceil(hi / step - 1e-9) * step
        ticks = np.arange(lo, hi + step * 1e-6, step)
        for tick in ticks:
            py = plot.bottom - (tick - lo) / (hi - lo) * plot.h
            if plot.y - 1 <= py <= plot.bottom + 1:
                pygame.draw.line(scr, UI["grid"], (plot.x, py), (plot.right, py))
                f.blit(scr, tick_text(tick, step), (plot.x - 5, py), "small", UI["muted"], "midright")
        return lo, hi

    def _x_positions(self, var: Variable, plot: pygame.Rect, values: list | None = None):
        """Screen x of each value (categories evenly spaced) and the (lo, hi) of a numeric axis."""
        vals = list(var.values if values is None else values)
        if var.numeric:
            lo, hi = float(min(var.values)), float(max(var.values))
            if hi <= lo:
                lo, hi = lo - 1.0, hi + 1.0
            margin = 0.04 * (hi - lo)
            lo, hi = lo - margin, hi + margin
            return np.array([plot.x + (float(v) - lo) / (hi - lo) * plot.w for v in vals]), (lo, hi)
        n = len(var.values)
        return np.array([plot.x + (var.values.index(v) + 0.5) / n * plot.w for v in vals]), None

    def _x_axis(self, var: Variable, plot: pygame.Rect, card: pygame.Rect, positions: np.ndarray,
                span: tuple | None) -> None:
        """Tick labels below the plot: at the values when they are few and fit, else round numbers;
        and the setting's label below them."""
        f, scr = self.fonts, self.screen
        labels = [(px, var.text(v)) for px, v in zip(positions, var.values)]
        fits = all(b[0] - a[0] > (f.width(a[1], "small") + f.width(b[1], "small")) / 2 + 6
                   for a, b in zip(labels, labels[1:]))
        if var.numeric and span is not None and (len(labels) > 12 or not fits):
            lo, hi = span
            target = max(2, plot.w // (f.width("0000.0", "small") * 2))
            ticks, step = _ticks(lo, hi, target)
            labels = [(plot.x + (t - lo) / (hi - lo) * plot.w, tick_text(t, step)) for t in ticks]
        last = -1e9
        for px, text in labels:
            w = f.width(text, "small")
            if px - w / 2 < last + 6 or px + w / 2 > card.right - 4:
                continue
            pygame.draw.line(scr, UI["grid"], (px, plot.bottom), (px, plot.bottom + 3))
            last = f.blit(scr, text, (px, plot.bottom + 4), "small", UI["muted"], "midtop").right
        f.blit(scr, var.label, (plot.centerx, plot.bottom + 4 + f.line("small")), "small", UI["text2"], "midtop")

    def _plot_rect(self, card: pygame.Rect, top: int, left: int, right: int) -> pygame.Rect:
        f, s = self.fonts, self.s
        pad = round(12 * s)
        bottom = card.bottom - pad - 2 * f.line("small") - round(6 * s)
        rect = pygame.Rect(card.x + pad + left, top, card.right - pad - right - (card.x + pad + left),
                           max(bottom - top, 20))
        self.plot_rect = rect
        return rect

    def _draw_lines(self, card: pygame.Rect, top: int, info: Metric, st: Stats) -> None:
        f, s, scr, d = self.fonts, self.s, self.screen, self.data
        x_var = d.var(self.x)
        series = d.var(self.series) if self.series else None
        pad = round(12 * s)
        if series is not None:
            items = [f"{series.text(v)}" for v in series.values]
            colours = palette.SERIES[:len(items)]
            top = self._legend(list(zip(colours, items)), card.x + pad, top, card.w - 2 * pad,
                               f"{series.label}:")
        y_label = f.width("-0000.00", "small") + round(8 * s)
        right = f.width("F", "small") + round(10 * s) if info.los else round(6 * s)  # room for the LOS letters
        plot = self._plot_rect(card, top + round(4 * s), y_label, right)
        scale = info.scale
        mean, lo_v, hi_v = st.mean * scale, st.lo * scale, st.hi * scale
        refs = None  # the reference column's mean per series (order_baseline: random placement)
        if info.ref_column and info.ref_column in d.frame:
            refs = self._stats(info.ref_column, self.x, self.series, self.current_slices()).mean * scale
        shown = [lo_v.ravel(), hi_v.ravel()] + ([refs.ravel()] if refs is not None else [])
        bottom, upper = self._y_axis(plot, np.concatenate(shown), info)
        xs, span = self._x_positions(x_var, plot)
        self._x_axis(x_var, plot, card, xs, span)

        def py(v):
            return plot.bottom - (v - bottom) / (upper - bottom) * plot.h

        dash = (round(5 * s), round(5 * s))
        if info.ref is not None:
            ry = py(info.ref * 1.0)
            dashed_line(scr, UI["muted"], [(plot.x, ry), (plot.right, ry)], 1, dash)
        if info.los:
            self._los_lines(plot, bottom, upper, py, dash)
        if not np.isfinite(mean).any():
            f.blit(scr, info.empty, plot.center, "base", UI["muted"], "center")
        colours = palette.SERIES if series is not None else [UI["accent"]]
        band = pygame.Surface(plot.size, pygame.SRCALPHA)
        scr.set_clip(plot.inflate(8, 8))
        for k in range(mean.shape[0]):
            colour = colours[k % len(colours)]
            ok = np.isfinite(mean[k])
            for a, b in finite_runs(ok):
                if b - a > 1:
                    pts = [(xs[j] - plot.x, py(hi_v[k, j]) - plot.y) for j in range(a, b)] + \
                          [(xs[j] - plot.x, py(lo_v[k, j]) - plot.y) for j in range(b - 1, a - 1, -1)]
                    pygame.draw.polygon(band, (*colour, 40), pts)
            for j in np.flatnonzero(ok):  # whiskers: min-max of every point
                if hi_v[k, j] > lo_v[k, j]:
                    pygame.draw.line(band, (*colour, 110), (xs[j] - plot.x, py(lo_v[k, j]) - plot.y),
                                     (xs[j] - plot.x, py(hi_v[k, j]) - plot.y), 1)
        scr.blit(band, plot)
        width = 2 if s < 1.3 else 3
        for k in range(refs.shape[0] if refs is not None else 0):  # faint, behind the lines
            colour = tuple(int(0.55 * c + 0.45 * b) for c, b in zip(colours[k % len(colours)], UI["surface"]))
            for a, b in finite_runs(np.isfinite(refs[k])):
                if b - a > 1:
                    dashed_line(scr, colour, [(xs[j], py(refs[k, j])) for j in range(a, b)], max(1, width - 1),
                                (round(7 * s), round(5 * s)))
        hover = self._nearest_x(xs, plot)
        for k in range(mean.shape[0]):
            colour = colours[k % len(colours)]
            ok = np.isfinite(mean[k])
            for a, b in finite_runs(ok):
                if b - a > 1:
                    pygame.draw.lines(scr, colour, False, [(xs[j], py(mean[k, j])) for j in range(a, b)], width)
            for j in np.flatnonzero(ok):
                r = round((5 if j == hover else 4) * s)
                pygame.draw.circle(scr, UI["background"], (xs[j], py(mean[k, j])), r + 2)
                pygame.draw.circle(scr, colour, (xs[j], py(mean[k, j])), r)
        scr.set_clip(None)
        if hover is not None:
            pygame.draw.line(scr, UI["text2"], (xs[hover], plot.y), (xs[hover], plot.bottom))
            value = x_var.values[hover]
            at = {**self.current_slices(), x_var.name: value}
            lines = [setting_phrase(x_var.name, value)]
            for k in range(mean.shape[0]):
                text = self._cell_text(st, k, hover, scale)
                if series is not None:  # each line's own text counts (entry_los), not pooled over the lines
                    counts = self._text_counts(select(d.frame, {**at, series.name: series.values[k]}))
                    text = f"{series.text(series.values[k])}: {text}" + (f"  ·  {counts}" if counts else "")
                lines.append(text)
            lines += self._slice_lines()
            if series is None:
                lines += [f"{column_label(c)}: {t}" for c in d.text_metrics
                          if (t := text_counts(select(d.frame, at), c))]
            self.tooltip = lines

    def _los_lines(self, plot: pygame.Rect, bottom: float, upper: float, py, dash: tuple[int, int]) -> None:
        """The level-of-service limits inside the value range as dashed lines, and each band's letter just
        right of the plot where the band is tall enough to hold it (as in the experiments figure)."""
        f, scr = self.fonts, self.screen
        edges = [0.0, *(v for _, v in LOS_LIMITS), math.inf]
        for v in edges[1:-1]:
            if bottom < v < upper:
                dashed_line(scr, UI["muted"], [(plot.x, py(v)), (plot.right, py(v))], 1, (dash[0], 2 * dash[1]))
        for letter, lo, hi in zip([k for k, _ in LOS_LIMITS] + ["F"], edges[:-1], edges[1:]):
            lo, hi = max(lo, bottom), min(hi, upper)
            if hi > lo and py(lo) - py(hi) >= f.line("small"):
                f.blit(scr, letter, (plot.right + round(5 * self.s), (py(lo) + py(hi)) / 2), "small", UI["muted"],
                       "midleft")

    def _text_counts(self, frame: pd.DataFrame) -> str:
        """'entry_los A ×1, F ×1': the text columns' counts in ``frame``, on one line."""
        return "  ·  ".join(f"{column_label(c)} {t}" for c in self.data.text_metrics if (t := text_counts(frame, c)))

    @staticmethod
    def _cell_text(st: Stats, k: int, j: int, scale: float) -> str:
        runs, n = int(st.runs[k, j]), int(st.n[k, j])
        if runs == 0:
            return "no runs"
        if n == 0:
            return f"no value in {plural(runs, 'run')}"
        of = f"{n} of {runs} runs" if n != runs else plural(runs, "run")
        if n == 1:
            return f"{fmt_num(st.mean[k, j] * scale)} ({of})"
        return (f"mean {fmt_num(st.mean[k, j] * scale)}, min {fmt_num(st.lo[k, j] * scale)}, "
                f"max {fmt_num(st.hi[k, j] * scale)} ({of})")

    def _slice_lines(self) -> list[str]:
        lines = []
        for v in self.slice_vars():
            value = self.slices.get(v.name)
            lines.append(f"averaged over {v.label}" if value is None else setting_phrase(v.name, value))
        return lines

    def _nearest_x(self, xs: np.ndarray, plot: pygame.Rect) -> int | None:
        if not plot.collidepoint(self.mouse) or xs.size == 0:
            return None
        return int(np.argmin(np.abs(xs - self.mouse[0])))

    def _colour_bar(self, rect: pygame.Rect, lo: float, hi: float, kind: str, caption: str) -> None:
        """A vertical colour bar with tick labels on its right and its caption above. The ticks
        point into the bar: one outside it, next to the label, would read as a minus sign."""
        f, s, scr = self.fonts, self.s, self.screen
        scr.blit(self._gradient(kind, rect.w, rect.h), rect)
        pygame.draw.rect(scr, UI["grid"], rect, 1)
        if hi <= lo:
            hi = lo + 1.0
        target = int(np.clip(rect.h // (2.4 * f.line("small")), 2, 8))
        ticks, step = _ticks(lo, hi, target)
        for tick in ticks:
            y = rect.bottom - (tick - lo) / (hi - lo) * rect.h
            pygame.draw.line(scr, UI["surface"], (rect.right - 1 - round(4 * s), y), (rect.right - 2, y))
            f.blit(scr, tick_text(tick, step), (rect.right + round(6 * s), y), "small", UI["muted"], "midleft")
        f.blit(scr, caption, (rect.right, rect.y - 4), "small", UI["text2"], "bottomright")

    def _draw_heatmap(self, card: pygame.Rect, top: int, info: Metric, st: Stats) -> None:
        f, s, scr, d = self.fonts, self.s, self.screen, self.data
        x_var, y_var = d.var(self.x), d.var(self.y)
        scale = info.scale
        mean = st.mean * scale
        lh = f.line("small")
        y_labels = [y_var.text(v) for v in y_var.values]
        label_w = min(max(f.width(t, "small") for t in y_labels), card.w // 5)
        left = lh + round(6 * s) + label_w + round(8 * s)
        bar_w = round(14 * s)
        right = bar_w + round(12 * s) + f.width("-0000.0", "small") + round(6 * s)
        plot = self._plot_rect(card, top + lh + round(6 * s), left, right)
        finite = mean[np.isfinite(mean)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        if hi <= lo:
            lo, hi = lo - 0.5 * (abs(lo) or 1.0), hi + 0.5 * (abs(hi) or 1.0)
        nx, ny = len(x_var.values), len(y_var.values)
        cw, ch = plot.w / nx, plot.h / ny
        t = (mean - lo) / (hi - lo)
        rgb = heat_rgb(t).astype(int)
        # Values in the cells: in all of them when the widest fits the narrowest cell, else in none.
        widest = max((f.width(fmt_num(v), "small") for v in finite), default=0)
        labels = bool(finite.size) and math.floor(cw) - 1 >= widest + 6 and math.floor(ch) - 1 >= lh + 2
        hover = None
        for j in range(ny):
            for i in range(nx):
                cell = pygame.Rect(round(plot.x + i * cw), round(plot.bottom - (j + 1) * ch),
                                   round(plot.x + (i + 1) * cw) - round(plot.x + i * cw),
                                   round(plot.bottom - j * ch) - round(plot.bottom - (j + 1) * ch))
                inner = cell.inflate(-1 if cw > 4 else 0, -1 if ch > 4 else 0)
                if np.isfinite(mean[j, i]):
                    colour = tuple(rgb[j, i])
                    pygame.draw.rect(scr, colour, inner)
                    if labels:
                        light = 0.2126 * colour[0] + 0.7152 * colour[1] + 0.0722 * colour[2] > 140
                        f.blit(scr, fmt_num(mean[j, i]), inner.center, "small", UI["surface"] if light else UI["text"],
                               "center")
                else:
                    pygame.draw.rect(scr, EMPTY, inner)
                if cell.collidepoint(self.mouse):
                    hover = (j, i, cell)
        if hover is not None:
            pygame.draw.rect(scr, UI["text"], hover[2], 2)
        # Axes: X values under the columns, Y values beside the rows (every k-th one when they do not all
        # fit, the same k throughout), the Y label turned up the left edge.
        x_labels = [x_var.text(v) for v in x_var.values]
        every = max(1, math.ceil((max(f.width(t, "small") for t in x_labels) + 6) / cw))
        for i in range(0, nx, every):
            f.blit(scr, x_labels[i], (plot.x + (i + 0.5) * cw, plot.bottom + 4), "small", UI["muted"], "midtop")
        f.blit(scr, x_var.label, (plot.centerx, plot.bottom + 4 + lh), "small", UI["text2"], "midtop")
        every = max(1, math.ceil((lh + 2) / ch))
        for j in range(0, ny, every):
            f.blit(scr, ellipsize(f, y_labels[j], label_w), (plot.x - 6, plot.bottom - (j + 0.5) * ch), "small",
                   UI["muted"], "midright")
        axis = pygame.transform.rotate(f.render(ellipsize(f, y_var.label, plot.h), "small", UI["text2"]), 90)
        scr.blit(axis, axis.get_rect(midleft=(card.x + round(10 * s), plot.centery)))
        bar = pygame.Rect(plot.right + round(10 * s), plot.y, bar_w, plot.h)
        self._colour_bar(bar, lo, hi, "heat", ellipsize(f, info.label, card.right - card.x - 40))
        if not finite.size:
            f.blit(scr, info.empty, plot.center, "base", UI["muted"], "center")
        if hover is not None:
            j, i, _ = hover
            lines = [setting_phrase(x_var.name, x_var.values[i]), setting_phrase(y_var.name, y_var.values[j]),
                     self._cell_text(st, j, i, scale), *self._slice_lines()]
            frame = select(d.frame, {**self.current_slices(), x_var.name: x_var.values[i],
                                     y_var.name: y_var.values[j]})
            lines += [f"{column_label(c)}: {t}" for c in d.text_metrics if (t := text_counts(frame, c))]
            self.tooltip = lines

    def _scatter_data(self, metric: str, x: str, colour: str | None):
        key = ("scatter", metric, x, colour)
        if key not in self._cache:
            d = self.data
            frame = d.frame
            ys = pd.to_numeric(frame[metric], errors="coerce").to_numpy(float)
            x_var = d.var(x)
            if x_var.numeric:
                xs = frame[x].to_numpy(float)
            else:  # categories at their index, spread a little so the dots do not hide each other
                jitter = np.random.default_rng(0).uniform(-0.28, 0.28, len(frame))
                xs = codes(frame[x], x_var.values).astype(float) + jitter
            cs = None
            if colour is not None:
                c_var = d.var(colour)
                cs = frame[colour].to_numpy(float) if c_var.numeric else codes(frame[colour], c_var.values)
            self._cache[key] = (xs, ys, cs)
        return self._cache[key]

    def _draw_scatter(self, card: pygame.Rect, top: int, info: Metric) -> None:
        f, s, scr, d = self.fonts, self.s, self.screen, self.data
        x_var = d.var(self.x)
        c_var = d.var(self.colour) if self.colour else None
        xs, ys, cs = self._scatter_data(self.metric, self.x, self.colour)
        scale = info.scale
        ys = ys * scale
        pad = round(12 * s)
        bar = c_var is not None and c_var.numeric
        if c_var is not None and not c_var.numeric:
            items = [(palette.SERIES[k % MAX_SERIES], c_var.text(v)) for k, v in enumerate(c_var.values)]
            top = self._legend(items, card.x + pad, top, card.w - 2 * pad, f"{c_var.label}:")
        y_label = f.width("-0000.00", "small") + round(8 * s)
        bar_w = round(14 * s)
        right = bar_w + round(12 * s) + f.width("-0000.0", "small") + round(6 * s) if bar else round(6 * s)
        plot = self._plot_rect(card, top + (f.line("small") + round(6 * s) if bar else round(4 * s)), y_label, right)
        bottom, upper = self._y_axis(plot, ys, info)
        if x_var.numeric:
            positions, span = self._x_positions(x_var, plot)
            lo_x, hi_x = span

            def px(v):
                return plot.x + (v - lo_x) / (hi_x - lo_x) * plot.w
        else:
            positions, span = self._x_positions(x_var, plot)
            n = len(x_var.values)

            def px(v):
                return plot.x + (v + 0.5) / n * plot.w
        self._x_axis(x_var, plot, card, positions, span)

        def py(v):
            return plot.bottom - (v - bottom) / (upper - bottom) * plot.h

        ok = np.isfinite(xs) & np.isfinite(ys)
        if bar:
            finite_c = cs[np.isfinite(cs)]
            c_lo, c_hi = (float(finite_c.min()), float(finite_c.max())) if finite_c.size else (0.0, 1.0)
            if c_hi <= c_lo:
                c_lo, c_hi = c_lo - 0.5, c_hi + 0.5
            rgb = dot_rgb((cs - c_lo) / (c_hi - c_lo)).astype(int)
        elif c_var is not None:
            rgb = np.array(palette.SERIES)[np.clip(cs, 0, MAX_SERIES - 1) % MAX_SERIES]
        else:
            rgb = np.tile(np.array(UI["accent"]), (len(xs), 1))
        sx, sy = px(xs), py(ys)
        self._dots = (sx, sy, ok)  # screen positions of the runs, for hovering
        r = max(2, round((3 if ok.sum() < 1500 else 2) * s))
        scr.set_clip(plot.inflate(2 * r, 2 * r))
        for i in np.flatnonzero(ok):
            cx, cy, colour = int(round(sx[i])), int(round(sy[i])), tuple(int(c) for c in rgb[i])
            pygame.gfxdraw.filled_circle(scr, cx, cy, r, colour)
            pygame.gfxdraw.aacircle(scr, cx, cy, r, colour)
        # The means: of equal-width bins of a sampled x, else at each value (each category).
        from roundabout_sim.plots import BIN_ABOVE  # (the sample figure's rule, so the two agree)

        binned = x_var.numeric and len(x_var.values) > BIN_ABOVE.get(d.design, len(x_var.values))
        if binned:
            centres, means, mins, maxs, counts, edges = binned_means(xs, ys)
        else:
            st = self._stats(self.metric, self.x, None, {})
            means, mins, maxs, counts = st.mean[0] * scale, st.lo[0] * scale, st.hi[0] * scale, st.n[0]
            if x_var.numeric:  # hover areas reach halfway to the neighbouring values
                centres = np.asarray(x_var.values, dtype=float)
                edges = np.concatenate([[lo_x], (centres[1:] + centres[:-1]) / 2, [hi_x]])
            else:
                centres = np.arange(len(x_var.values), dtype=float)
                edges = np.arange(len(x_var.values) + 1) - 0.5
        good = np.isfinite(means)
        line_pts = [(px(c), py(m)) for c, m in zip(centres[good], means[good])]
        if len(line_pts) > 1:
            pygame.draw.lines(scr, UI["background"], False, line_pts, 5 if s < 1.3 else 6)
            pygame.draw.lines(scr, UI["text"], False, line_pts, 2 if s < 1.3 else 3)
        for p, k in zip(line_pts, np.asarray(counts)[good]):
            if k == 1:  # the mean is that run: a ring round its dot, drawn again over the line, keeps its colour
                pygame.draw.circle(scr, UI["text"], p, r + round(3 * s), max(1, round(1.5 * s)))
                for i in np.flatnonzero(ok & (np.hypot(sx - p[0], sy - p[1]) <= r + 1)):
                    colour = tuple(int(c) for c in rgb[i])
                    pygame.gfxdraw.filled_circle(scr, int(round(sx[i])), int(round(sy[i])), r, colour)
                    pygame.gfxdraw.aacircle(scr, int(round(sx[i])), int(round(sy[i])), r, colour)
                continue
            pygame.draw.circle(scr, UI["background"], p, round(5 * s) + 1)
            pygame.draw.circle(scr, UI["text"], p, round(4 * s))
        scr.set_clip(None)
        if bar:
            self._colour_bar(pygame.Rect(plot.right + round(10 * s), plot.y, bar_w, plot.h), c_lo, c_hi, "dots",
                             ellipsize(f, c_var.label, card.w // 2))
        if not ok.any():
            f.blit(scr, info.empty, plot.center, "base", UI["muted"], "center")
            return
        if not plot.collidepoint(self.mouse):
            return
        dist = np.hypot(sx - self.mouse[0], sy - self.mouse[1])
        dist[~ok] = np.inf
        i = int(np.argmin(dist))
        if dist[i] <= max(8.0, 3 * r):
            pygame.draw.circle(scr, UI["text"], (sx[i], sy[i]), r + 3, 2)
            row = d.frame.iloc[i]
            lines = [f"{info.label}: {fmt_num(ys[i])}", setting_phrase(x_var.name, row[x_var.name])]
            if c_var is not None:
                lines.append(setting_phrase(c_var.name, row[c_var.name]))
            shown = (x_var.name, c_var.name if c_var is not None else None)
            lines += [setting_phrase(v.name, row[v.name]) for v in d.varied if v.name not in shown]
            if SEED in d.frame:
                lines.append(f"seed {row[SEED]}  ·  run {i + 1} of {d.runs}")
            lines += [f"{column_label(c)}: {t}" for c in d.text_metrics if (t := text_value(row[c]))]
            self.tooltip = lines
            return
        # Else the bin under the mouse.
        bins = np.array([px(e) for e in edges])
        k = int(np.clip(np.searchsorted(bins, self.mouse[0], side="right") - 1, 0, len(centres) - 1)) \
            if len(centres) else None
        if k is None:
            return
        pygame.draw.line(scr, UI["grid"], (bins[k], plot.y), (bins[k], plot.bottom))
        pygame.draw.line(scr, UI["grid"], (bins[min(k + 1, len(bins) - 1)], plot.y),
                         (bins[min(k + 1, len(bins) - 1)], plot.bottom))
        if binned:
            head = f"{x_var.label}: {fmt_num(edges[k])}–{fmt_num(edges[k + 1])} (bin {k + 1} of {len(centres)})"
        else:
            head = setting_phrase(x_var.name, x_var.values[k])
        if counts[k]:
            body = (f"mean {fmt_num(means[k])}, min {fmt_num(mins[k])}, max {fmt_num(maxs[k])} "
                    f"({plural(counts[k], 'run')})")
        else:
            body = "no runs with a value in this bin" if binned else "no runs with a value here"
        self.tooltip = [head, body]

    # ------------------------------------------------------------ form
    def _draw_form(self) -> None:
        f, s, scr, card = self.fonts, self.s, self.screen, self.main_rect
        self._card(card)
        pad = round(14 * s)
        x, y = card.x + pad, card.y + pad
        f.blit(scr, "New sweep", (x, y), "title", UI["text"])
        y += f.line("title")
        for line in self.form_intro:
            f.blit(scr, line, (x, y), "small", UI["muted"])
            y += f.line("small")
        region = self.regions["params"]
        scr.set_clip(region)
        chosen = [v.name for v in self.form]
        for w in self.widgets:
            if w.region != "params" or not w.rect.colliderect(region):
                continue
            if w.kind == "heading":
                f.blit(scr, w.label, (w.rect.x, w.rect.centery), "small", UI["muted"], "midleft")
                continue
            hot = w.rect.collidepoint(self.mouse) and region.collidepoint(self.mouse)
            if w.name in chosen:
                pygame.draw.rect(scr, UI["accent"], w.rect, border_radius=round(4 * s))
            elif hot:
                pygame.draw.rect(scr, UI["grid"], w.rect, border_radius=round(4 * s))
            text_x = w.rect.x + round(8 * s)
            if w.name in chosen:
                badge = str(chosen.index(w.name) + 1)
                f.blit(scr, badge, (w.rect.right - round(8 * s), w.rect.centery), "small", UI["text"], "midright")
            room = w.rect.w - round(28 * s)
            idle = w.data  # why it would change nothing here ("" if it acts)
            tag_w = f.width(idle, "small") + round(8 * s) if idle else 0
            if idle and f.width(w.label, "small") + tag_w <= room:  # the reason too, where it fits whole
                f.blit(scr, idle, (text_x + room, w.rect.centery), "small",
                       UI["text2"] if w.name in chosen else UI["muted"], "midright")
            colour = UI["text"] if w.name in chosen else UI["muted"] if idle else UI["text2"]
            f.blit(scr, ellipsize(f, w.label, room), (text_x, w.rect.centery), "small", colour, "midleft")
        scr.set_clip(None)
        self._scroll_bar("params")
        right = self.form_right
        for i, line in enumerate(self.pick_lines):  # (none picked yet)
            f.blit(scr, line, (right.x, right.y + i * f.line("base")), "base", UI["muted"])
        for w in self.widgets:
            if w.region is not None or w.kind not in ("block", "remove", "toggle", "range", "cycler", "info"):
                continue
            if w.kind == "info":
                f.blit(scr, ellipsize(f, w.label, w.rect.w), w.rect.topleft, "small",
                       UI["text"] if w.rect.collidepoint(self.mouse) else UI["text2"])
            elif w.kind == "block":
                self._draw_block(w)
            elif w.kind == "remove":
                self._draw_remove(w)
            elif w.kind == "toggle":
                self._draw_toggle(w)
            elif w.kind == "range":
                self._draw_range(w)
            elif w.kind == "cycler" and w.name.startswith("points:"):
                self._draw_cycler(w)
                var = self.form[int(w.name.split(":")[1])]
                values = ", ".join(setting_text(var.name, v) for v in var.values())
                row = w.data["row"]
                f.blit(scr, "points", (row.right + round(8 * s), row.centery), "small", UI["text2"], "midleft")
                vx = row.right + round(8 * s) + f.width("points", "small") + round(8 * s)
                f.blit(scr, ellipsize(f, values, right.right - vx), (vx, row.centery), "small", UI["muted"], "midleft")
        # A warning, the base settings not swept, then the command at the bottom of the right column (laid
        # out with the form).
        lh = f.line("small")
        cy, lines = self.idle_lines
        for line in lines:
            f.blit(scr, line, (right.x, cy), "small", palette.CRITICAL)
            cy += lh
        cy, lines = self.base_lines
        if lines:
            f.blit(scr, "Base settings (the rest at their defaults)", (right.x, cy), "small", UI["text2"])
            cy += lh + round(2 * s)
            for line in lines:
                f.blit(scr, line, (right.x, cy), "small", UI["muted"])
                cy += lh
        cy, lines = self.command_lines
        f.blit(scr, "Command", (right.x, cy), "small", UI["text2"])
        cy += lh + round(2 * s)
        for line in lines:
            f.blit(scr, line, (right.x, cy), "small", UI["muted"])
            cy += lh

    def _draw_block(self, w: Widget) -> None:
        f, s, scr = self.fonts, self.s, self.screen
        i = int(w.name)
        badge = pygame.Rect(w.rect.x, w.rect.y + (w.rect.h - f.line("small")) // 2 - 1, f.line("small") + 2,
                            f.line("small") + 2)
        pygame.draw.rect(scr, UI["accent"], badge, border_radius=round(4 * s))
        f.blit(scr, str(i + 1), badge.center, "small", UI["text"], "center")
        role = f"  ·  {w.data}"
        x = badge.right + round(8 * s)
        room = w.rect.right - x - f.line("small") - round(16 * s)
        label = ellipsize(f, w.label, room - f.width(role, "small"), "base")
        x = f.blit(scr, label, (x, w.rect.centery), "base", UI["text"], "midleft").right
        f.blit(scr, role, (x, w.rect.centery), "small", UI["muted"], "midleft")

    def _draw_remove(self, w: Widget) -> None:
        scr, r = self.screen, w.rect
        hot = r.collidepoint(self.mouse)
        pygame.draw.rect(scr, UI["grid"] if hot else UI["raised"], r, border_radius=round(4 * self.s))
        k = r.w // 4
        colour = UI["text"] if hot else UI["text2"]
        pygame.draw.line(scr, colour, (r.x + k, r.y + k), (r.right - k - 1, r.bottom - k - 1), 2)
        pygame.draw.line(scr, colour, (r.x + k, r.bottom - k - 1), (r.right - k - 1, r.y + k), 2)

    def _draw_toggle(self, w: Widget) -> None:
        f, scr = self.fonts, self.screen
        var = self.form[int(w.name)]
        for choice, rect in zip(var.spec.choices, w.options):
            on = choice in var.chosen
            hot = rect.collidepoint(self.mouse)
            fill = UI["accent"] if on else UI["grid"] if hot else UI["raised"]
            pygame.draw.rect(scr, fill, rect, border_radius=round(4 * self.s))
            f.blit(scr, ellipsize(f, choice, rect.w - 4), rect.center, "small", UI["text"] if on else UI["text2"],
                   "center")

    def _draw_range(self, w: Widget) -> None:
        f, s, scr = self.fonts, self.s, self.screen
        var = self.form[int(w.name)]
        spec, track = var.spec, w.options[0]
        unit = f" {spec.unit}" if spec.unit else ""
        text = f"from {setting_text(var.name, var.lo)} to {setting_text(var.name, var.hi)}{unit}"
        f.blit(scr, text, (w.rect.x, w.rect.y), "small", UI["text2"])
        f.blit(scr, f"slider range {spec.lo:g}–{spec.hi:g}", (w.rect.right, w.rect.y), "small", UI["muted"],
               "topright")
        ends = [track.x + (v - spec.lo) / (spec.hi - spec.lo) * track.w for v in (var.lo, var.hi)]
        pygame.draw.rect(scr, UI["grid"], track, border_radius=2)
        pygame.draw.rect(scr, UI["accent"], (ends[0], track.y, max(ends[1] - ends[0], 1), track.h), border_radius=2)
        hot = w.rect.collidepoint(self.mouse) or (self.drag is not None and self.drag[0].name == w.name)
        for k, e in enumerate(ends):
            big = hot and (self.drag is None or self.drag[1] == k)
            pygame.draw.circle(scr, UI["text"], (e, track.centery), round((8 if big else 6.5) * s))

    # ------------------------------------------------------------ panel
    def _draw_panel(self) -> None:
        f, s, scr, r = self.fonts, self.s, self.screen, self.panel_rect
        pygame.draw.rect(scr, UI["background"], r)
        pygame.draw.line(scr, UI["grid"], r.topleft, r.bottomleft)
        p = round(14 * s)
        x, y = r.x + p, r.y + p
        w = r.w - 2 * p
        f.blit(scr, "Sweep viewer", (x, y), "title", UI["text"])
        y += f.line("title")
        d = self.data
        if d is not None:
            name = d.path.name if d.path else "sweep"
            varied = ", ".join(v.label.split(" (")[0] for v in d.varied)
            line2 = "  ·  ".join(filter(None, (plural(d.runs, "run"), DESIGN_TEXT[d.design], varied)))
        else:
            name, line2 = "no sweep loaded", f"results: {self.results_dir}"
        f.blit(scr, ellipsize(f, name, w), (x, y), "small", UI["text2"])
        f.blit(scr, ellipsize(f, line2, w), (x, y + f.line("small")), "small", UI["muted"])
        self.header_rect = pygame.Rect(x, y, w, 2 * f.line("small")) if d is not None else pygame.Rect(0, 0, 0, 0)
        if self.header_rect.collidepoint(self.mouse):  # (cut with an ellipsis when long: here all of it)
            self._info_tooltip([name, f"{plural(d.runs, 'run')}, {DESIGN_TEXT[d.design]}",
                                *([f"Varied: {varied}"] if varied else [])])
        y += 2 * f.line("small") + round(8 * s)
        if self.job is not None and self.page != "new":
            self._draw_progress(x, y, w)
        for wd in self.widgets:
            if wd.region is not None:
                continue
            if wd.kind == "button":
                self._draw_button(wd)
            elif wd.kind == "tab":
                self._draw_tab(wd)
            elif wd.kind == "choice":
                self._draw_view_choice(wd)
            elif wd.kind == "cycler" and not wd.name.startswith("points:"):
                self._draw_cycler(wd)
        if self.page == "chart" and "panel" in self.regions:
            region = self.regions["panel"]
            scr.set_clip(region)
            for wd in self.widgets:
                if wd.region == "panel" and wd.rect.colliderect(region):
                    self._draw_region_widget(wd)
            scr.set_clip(None)
            self._scroll_bar("panel")
        elif self.page == "files":
            fy = self.regions["files"].y - len(self.folder_lines) * f.line("small") - round(8 * s)
            for line in self.folder_lines:
                f.blit(scr, line, (x, fy), "small", UI["muted"])
                fy += f.line("small")
            region = self.regions["files"]
            if not self.files:
                f.blit(scr, "No sweep CSVs here yet.", (x, region.y), "small", UI["muted"])
            scr.set_clip(region)
            for wd in self.widgets:
                if wd.region == "files" and wd.rect.colliderect(region):
                    self._draw_file(wd)
            scr.set_clip(None)
            self._scroll_bar("files")
        elif self.page == "new":
            self._draw_new_text(x, self.new_text_y, w)
        self._draw_status()

    def _draw_region_widget(self, wd: Widget) -> None:
        f, s, scr = self.fonts, self.s, self.screen
        if wd.kind == "heading":
            f.blit(scr, wd.label, (wd.rect.x, wd.rect.centery), "small", UI["muted"], "midleft")
        elif wd.kind == "cycler":
            self._draw_cycler(wd)
        elif wd.kind == "metric":
            selected = wd.name == self.metric
            hot = wd.rect.collidepoint(self.mouse) and self.regions["panel"].collidepoint(self.mouse)
            if selected or hot:
                pygame.draw.rect(scr, UI["accent"] if selected else UI["grid"], wd.rect, border_radius=round(4 * s))
            text = ellipsize(f, wd.label, wd.rect.w - round(16 * s))
            f.blit(scr, text, (wd.rect.x + round(8 * s), wd.rect.centery), "small",
                   UI["text"] if selected else UI["text2"], "midleft")

    def _draw_file(self, wd: Widget) -> None:
        f, s, scr = self.fonts, self.s, self.screen
        loaded = self.data is not None and self.data.path is not None and \
            Path(self.data.path).resolve() == Path(wd.name).resolve()
        hot = wd.rect.collidepoint(self.mouse) and self.regions["files"].collidepoint(self.mouse)
        if loaded or hot:
            pygame.draw.rect(scr, UI["accent"] if loaded else UI["grid"], wd.rect, border_radius=round(4 * s))
        x, y = wd.rect.x + round(8 * s), wd.rect.y + round(3 * s)
        f.blit(scr, ellipsize(f, wd.label, wd.rect.w - round(16 * s)), (x, y), "small",
               UI["text"] if loaded else UI["text2"])
        f.blit(scr, wd.data + ("  ·  loaded" if loaded else ""), (x, y + f.line("small")), "small",
               UI["text2"] if loaded else UI["muted"])

    def _scroll_bar(self, region: str) -> None:
        rect, height = self.regions.get(region), self.content_h.get(region, 0)
        if rect is None or height <= rect.h:
            return
        bar_h = max(20, rect.h * rect.h // height)
        bar_y = rect.y + (rect.h - bar_h) * self.scroll[region] // max(height - rect.h, 1)
        pygame.draw.rect(self.screen, UI["grid"], (rect.right - round(3 * self.s), bar_y, round(3 * self.s), bar_h),
                         border_radius=2)

    def _draw_button(self, w: Widget) -> None:
        f, scr = self.fonts, self.screen
        disabled = isinstance(w.data, dict) and w.data.get("disabled")
        hot = w.rect.collidepoint(self.mouse) and not disabled
        fill = UI["grid"] if hot else UI["raised"]
        pygame.draw.rect(scr, fill, w.rect, border_radius=round(4 * self.s))
        f.blit(scr, w.label, w.rect.center, "small", UI["muted"] if disabled else UI["text"], "center")

    def _draw_tab(self, w: Widget) -> None:
        draw_tab(self.screen, self.fonts, w.rect, w.label, w.name == self.page, w.rect.collidepoint(self.mouse))

    def _draw_view_choice(self, w: Widget) -> None:
        f, s, scr = self.fonts, self.s, self.screen
        f.blit(scr, w.label, w.rect.topleft, "base", UI["text2"])
        available = self.views()
        for view, rect in zip(VIEWS, w.options):
            selected = view == self.view
            hot = rect.collidepoint(self.mouse)
            fill = UI["accent"] if selected else UI["grid"] if hot and view in available else UI["raised"]
            pygame.draw.rect(scr, fill, rect, border_radius=round(4 * s))
            colour = UI["text"] if selected else UI["text2"] if view in available else UI["muted"]
            f.blit(scr, view, rect.center, "small", colour, "center")

    def _draw_cycler(self, w: Widget) -> None:
        f, s, scr = self.fonts, self.s, self.screen
        if w.label:
            f.blit(scr, ellipsize(f, w.label, w.rect.w, "base"), w.rect.topleft, "base", UI["text2"])
        row = w.data["row"]
        enabled = w.data.get("enabled", True)
        pygame.draw.rect(scr, UI["raised"], row, border_radius=round(4 * s))
        for k, arrow in enumerate(w.options):
            hot = arrow.collidepoint(self.mouse) and enabled
            if hot:
                pygame.draw.rect(scr, UI["grid"], arrow, border_radius=round(4 * s))
            cx, cy, a = arrow.centerx, arrow.centery, max(3, round(4 * s))
            pts = [(cx - a, cy), (cx + a // 2 + 1, cy - a), (cx + a // 2 + 1, cy + a)] if k == 0 else \
                [(cx + a, cy), (cx - a // 2 - 1, cy - a), (cx - a // 2 - 1, cy + a)]
            pygame.draw.polygon(scr, UI["text"] if enabled else UI["grid"], pts)
        room = row.w - w.options[0].w - w.options[1].w - round(8 * s)
        f.blit(scr, ellipsize(f, w.data["text"], room), row.center, "small",
               UI["text"] if enabled else UI["muted"], "center")

    def _draw_progress(self, x: int, y: int, w: int) -> int:
        """Progress bar of the sweep (running or finished) and its text; returns the y below."""
        f, s, scr, job = self.fonts, self.s, self.screen, self.job
        bar = pygame.Rect(x, y, w, max(3, round(5 * s)))
        pygame.draw.rect(scr, UI["grid"], bar, border_radius=2)
        state = job.state
        fill = palette.CRITICAL if state == "failed" else UI["accent"]
        done = 1.0 if state == "done" or job.saved else job.fraction()
        if done > 0:
            pygame.draw.rect(scr, fill, (bar.x, bar.y, max(round(done * bar.w), 2), bar.h), border_radius=2)
        y = bar.bottom + round(3 * s)
        if state == "running":
            eta = job.eta()
            text = (f"Sweep running: {job.done} / {job.total or '?'} runs  ·  {job.elapsed:.0f} s"
                    + (f"  ·  ~{eta:.0f} s left" if eta is not None else ""))
        elif state == "done":
            text = f"Sweep finished: {plural(job.total or job.done, 'run')} in {job.elapsed:.0f} s"
        elif state == "cancelled" and job.saved:
            text = f"Sweep cancelled while drawing: all {plural(job.total or job.done, 'run')} saved"
        elif state == "cancelled":
            text = f"Sweep cancelled {cancelled_when(job)}"
        elif job.proc.returncode:
            text = f"Sweep failed (exit {job.proc.returncode})"
        else:
            text = "Sweep finished, but its CSV could not be loaded"
        f.blit(scr, ellipsize(f, text, w), (x, y), "small", palette.CRITICAL if state == "failed" else UI["text2"])
        return y + f.line("small")

    def _draw_new_text(self, x: int, y: int, w: int) -> None:
        """New sweep, in the panel: the size of the sweep, then the progress or the outcome."""
        f, s, scr = self.fonts, self.s, self.screen
        lh = f.line("small")
        bottom = self.status_rect.y - round(8 * s)
        for line in self.summary_lines:  # laid out with the panel
            f.blit(scr, line, (x, y), "small", UI["text2"])
            y += lh
        y += round(10 * s)
        job = self.job
        if job is None:
            return
        y = self._draw_progress(x, y, w) + round(4 * s)
        if job.state == "running" and job.last:
            detail = job.last
        elif job.state == "failed":
            detail = job.error_text()
        elif job.state == "done":
            detail = f"Wrote {job.out}.csv"
        elif job.saved:
            detail = f"Wrote {job.out}.csv (no PNG)"
        else:
            detail = ""
        colour = palette.CRITICAL if job.state == "failed" else UI["muted"]
        for line in wrap_lines(f, detail, w, max(1, (bottom - y) // lh)):
            if y + lh > bottom:
                break
            f.blit(scr, line, (x, y), "small", colour)
            y += lh

    def _status_text(self) -> str:
        w = self.widget_at(self.mouse)
        live = None if self.drag else self._live_message(w)
        if live is not None:  # from a key or a click, or one that came in by itself (see ui.Messages)
            return live
        if w is not None:
            return w.help
        for wd in self.widgets:  # a setting's block header: its help
            if wd.kind == "block" and wd.rect.collidepoint(self.mouse):
                return f"{wd.label}: {wd.help}"
        if self.data is not None and self.header_rect.collidepoint(self.mouse):
            return HEADER_HELP
        if self.page == "chart" and self.data is not None:
            if self.context_rect.collidepoint(self.mouse):
                return CONTEXT_HELP
            if self.plot_rect.collidepoint(self.mouse):
                return VIEW_HELP[self.view]
            if self.main_rect.collidepoint(self.mouse):
                return metric_info(self.metric, self.x, self.data.open_demand).help
        text, until = self.message
        if time.monotonic() < until:
            return text
        if self.page == "chart":
            return DEFAULT_HINT if self.data is not None else EMPTY_HINT
        return FILES_HINT if self.page == "files" else NEW_HINT

    def _draw_status(self) -> None:
        f, r = self.fonts, self.status_rect
        pygame.draw.line(self.screen, UI["grid"], r.topleft, r.topright)
        y = r.y + round(6 * self.s)
        for line in wrap_lines(f, self._status_text(), r.w, self.status_lines):
            f.blit(self.screen, line, (r.x, y), "small", UI["text2"])
            y += f.line("small")

    def _draw_tooltip(self) -> None:
        """The exact values under the mouse, in a box beside it."""
        if not self.tooltip:
            return
        f, s, scr = self.fonts, self.s, self.screen
        pad = round(6 * s)
        # In the chart card, or anywhere in the window for the panel's lines naming the sweep.
        area = self.main_rect if self.main_rect.collidepoint(self.mouse) else scr.get_rect()
        lines = [ellipsize(f, t, self.main_rect.w - 4 * pad) for t in self.tooltip]
        w = max(f.width(t, "small") for t in lines) + 2 * pad
        h = len(lines) * f.line("small") + 2 * pad
        box = pygame.Rect(self.mouse[0] + 16, self.mouse[1] + 16, w, h)
        if box.right > area.right:
            box.right = self.mouse[0] - 12
        if box.bottom > area.bottom:
            box.bottom = self.mouse[1] - 12
        box = box.clamp(area)
        pygame.draw.rect(scr, UI["raised"], box, border_radius=4)
        pygame.draw.rect(scr, UI["grid"], box, 1, border_radius=4)
        for i, text in enumerate(lines):
            f.blit(scr, text, (box.x + pad, box.y + pad + i * f.line("small")), "small",
                   UI["text"] if i == 0 else UI["text2"])


# ------------------------------------------------------------------ main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m roundabout_sim.sweep_viewer",
        description="Browse the CSV of a parameter sweep (python -m roundabout_sim.experiments sweep), or set up "
                    "and run a new one.",
        epilog="Keys: Up/Down metric, Left/Right X setting, Tab series (Shift+Tab back), H heatmap, S save PNG, "
               "R reload, O files, N new sweep, Esc quit. Hover the chart for exact values.")
    parser.add_argument("csv", nargs="?", default=None,
                        help="sweep CSV to open (default: the most recent sweep CSV in the results folder)")
    parser.add_argument("--base-settings", dest="base_settings", default=None, metavar="JSON",
                        help="JSON file {parameter: value} that new sweeps start from (default: the defaults)")
    parser.add_argument("--results", default="results", metavar="DIR",
                        help="results folder: sweep CSVs listed and new sweeps and PNGs written there "
                             "(default results)")
    parser.add_argument("--size", type=cli.window_size, default=None, metavar="WxH",
                        help="window size in pixels (default: 90%% of the screen)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    SweepViewer(args.csv, size=args.size, results_dir=args.results, base_settings=args.base_settings).run()


if __name__ == "__main__":
    main()
