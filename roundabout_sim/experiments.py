"""Headless experiments: one run with a dashboard figure, or a parameter sweep.

Examples::

    # one run; the space-time diagram shows jams as stripes sloping downwards
    python -m roundabout_sim.experiments run --set n_lanes=1 --set n_arms=0 --set density=80

    # how much of the jamming do the entrances/exits cause? (repeats the run with exit_prob = 0)
    python -m roundabout_sim.experiments run --baseline

    # fundamental diagram for 1-3 lanes (3 seeds per point, in parallel)
    python -m roundabout_sim.experiments sweep --x density=10:150:10 --series n_lanes=1,2,3

    # natural (phantom) jams against those the arms set off
    python -m roundabout_sim.experiments sweep --x density=20:120:20 --series exit_prob=0,0.1

    # three or more settings (full grid): the first two on the figure, averaged over the rest
    python -m roundabout_sim.experiments sweep --vary density=20:100:20 --vary n_lanes=1,2 --vary road_condition=dry,wet

    # one run per row of a CSV (columns = parameter names, optional seed), from a preset
    python -m roundabout_sim.experiments sweep --plan plan.csv --preset us_single_lane

    # a random design: 40 points, Latin hypercube over the ranges (or the plausible ranges of every parameter)
    python -m roundabout_sim.experiments sweep --sample 40 --vary density=10:120 --vary reaction_time=0.2:1.2

    # browse the results
    python -m roundabout_sim.experiments view

Any model parameter can be set with ``--set name=value`` (repeatable) or its
own flag (``--density 70``); ``--help`` on each command lists them all.
``--preset NAME`` or ``--random SEED`` give the base settings first
(``--list-presets`` lists the presets): defaults < preset / random < --set and
the parameter flags < the values a sweep varies. The random draw links some
settings to others (the radius to the lanes, the density to the traffic mode,
...), and settings given with --set or a flag take part in those links.

Outputs
  run    PREFIX_timeseries.csv  one row per sample (every --interval s)
         PREFIX_spacetime.npz   speed per angular bin and sector occupancy over time
         PREFIX_summary.csv     one row: every setting + the metrics over the second half
                                (+ baseline_* and arm_jam_share with --baseline, + preset
                                or random with --preset / --random)
         PREFIX.png             dashboard
  sweep  PREFIX.csv             one row per run: the varied settings (the x just before
                                the seed), the seed, every other setting, then the metrics
                                averaged over the measurement window
         PREFIX_meta.json       the design: grid / plan / sample, the varied settings with
                                their values or ranges, the fixed ones, preset, seeds, times
         PREFIX.png             metrics against the x (mean across seeds, band = min-max);
                                plans and samples: every run against each varied setting
         PREFIX_heatmap.png     2+ grid settings: x by series cells of the headline metrics

CSV columns: settings are named exactly like the parameters (density, n_lanes,
reaction_time, ...). A measurement that shares a parameter's name carries a
``measured_`` prefix: ``measured_density`` is the density actually on the ring,
``density`` the setting; ``measured_speed_std`` is the spread of the cars'
current speeds, ``speed_std`` the spread of their desired speeds. Every other
column is measured.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from roundabout_sim import cli, state
from roundabout_sim.config import (PARAM_BY_NAME, PARAMS, SimConfig, axis_label, changed_settings, idle_note,
                                   inactive_reason)
from roundabout_sim.engine import (MAX_DECEL, PEAK_PERIOD, SLOW_FRACTION, TWO_PI, RingSim, braking_limits,
                                   mean_demand_factor)
from roundabout_sim.metrics import LOS_LIMITS, RATE_WINDOW, Recorder, entry_vc, level_of_service, los_text
from roundabout_sim.palette import SERIES_HEX
from roundabout_sim.textfmt import plain as _plain
from roundabout_sim.textfmt import short_value as _value_text
from roundabout_sim.textfmt import and_join, base_text, plural, significant, table_number

CONFIG_NAMES = tuple(f.name for f in fields(SimConfig) if f.name != "seed")
MAX_SERIES = len(SERIES_HEX)  # values of the 2nd swept setting: one palette colour each in the figure
META_VERSION = 1  # format of PREFIX_meta.json (the sweep viewer reads it)
PROGRESS_SETTINGS = 4  # varied settings named in a progress line of a plan or a sample (then "+N more")
SCATTER_SETTINGS = 8  # most settings in the figure of a plan or a sample (those correlating most strongly)
PRINTED_CORRELATIONS = 12  # most settings in the printed rank-correlation table of a sample


# Onset of stop-and-go = "the first jam": the stop-and-go index (stops per car
# per minute over RATE_WINDOW) first exceeding this, i.e. a quarter of the cars
# forced to a near-stop within 30 s. Measured: smooth flow stays below ~0.4 (its
# 90th percentile, even with merging traffic in nearly empty roundabouts),
# developed stop-and-go runs at 0.8-2 for small and 2000-car rings alike.
STOP_GO_ONSET = 0.5
# Onset of congestion: the RATE_WINDOW mean of jam_fraction first exceeding this.
CONGESTION_ONSET = 0.3
JAM_PRESENT = 0.02  # below this mean jam fraction no wave speed (or arm share) is reported
JAM_ORIGINS = ("jams_following", "jams_merge", "jams_exit", "jams_entry", "jams_other")  # metrics.jam_origins
WAVE_MAX_LAG = 20.0  # [s] longest lag used to follow a speed pattern
WAVE_MIN_CORR = 0.3  # stop following once the pattern decorrelates below this
WAVE_SMOOTH = 3.0  # [bins] spatial smoothing that hides single cars, keeps jams
PATTERN_SPEED_CAP = 20.0  # [m/s] fastest a pattern may move between two samples
# run --baseline: these summary columns of the same run with exit_prob = 0 go into
# PREFIX_summary.csv with a "baseline_" prefix.
BASELINE_COLUMNS = ("jam_fraction", "stops_per_car_min", "n_jams", "flux", "measured_mean_speed")
INIT_SPACING_NOTE = ("Initial spacing shapes the start; near the jam threshold it can lock in jams, elsewhere it "
                     "fades in 1-2 min (use --warmup 0 to see the transient)")

DEFINITIONS = f"""\
measured columns (CSV) and how to read them
  flux                 cars per second past a point, all lanes (from the speeds)
  detector_flux        cars per second actually crossing the detector line
  measured_density     density actually on the ring [veh/km/lane], mean over the
                       window (density is the setting; in open mode the number
                       of cars on the ring varies)
  measured_mean_speed  mean speed of the cars on the ring [m/s] (mean_speed is
                       the setting: the mean desired speed). Moments with no car
                       on the ring (open mode) count for none of the per-car
                       measures (speeds, jammed share, spacing, order)
  mean_free_speed      mean desired speed of the cars on the ring [m/s]: what
                       the drivers want, with the road condition and the
                       trucks' lower speed applied
  measured_speed_std   spread (standard deviation) of the cars' current speeds
                       [m/s] (speed_std is the setting: the spread of their
                       desired speeds)
  jam_fraction         CONGESTION (the headline): share of cars crawling below
                       {100 * SLOW_FRACTION:g} % of their free speed. Rises steadily with density.
  speed_deficit        1 - total speed / total desired speed of the cars on the
                       ring: 0 = everyone at their desired speed, 1 = standstill
  congestion_onset     time to congestion [s] from the start: the first time the
                       {RATE_WINDOW:g} s mean of jam_fraction exceeds {100 * CONGESTION_ONSET:g} %. Counted from
                       t = 0, so a start that is already jammed scores ~0.
                       Empty = never.
  stops_per_car_min    INSTABILITY, the companion: stop-and-go index = how often a
                       moving car is forced to (nearly) stop, per car per minute.
                       ~0 when smooth and also in a standstill (nobody is moving
                       to be stopped), ~1-2 in stop-and-go waves. It says how
                       unstable the flow is, not how congested.
  stop_go_onset        time to the first stop-and-go jam [s] from the start: the
                       first time the stop-and-go index, counted over a full
                       {RATE_WINDOW:g} s window, exceeds {STOP_GO_ONSET:g} (a quarter of the cars forced
                       to stop within {RATE_WINDOW:g} s). Smooth flow stays below ~0.4,
                       stop-and-go runs at 0.8-2. A packed start or a ring that
                       only crawls never counts (use congestion_onset there).
                       Empty = never.
  n_jams               runs of 2+ jammed cars in a lane; jams_following / _merge /
                       _exit / _entry / _other split them by what the front car is
                       doing NOW (other = braked by you or just slotted in)
  following_jam_share  jams_following / all jams: share of jams whose front car
                       is just following. Only an UPPER BOUND on natural (phantom)
                       jams: a wave set off at an exit keeps travelling backwards
                       with ordinary followers at its front. The honest test of
                       what entrances/exits cause is the same run at exit_prob 0:
                       run --baseline, or sweep --series exit_prob=0,0.1.
  arm_jam_share        (run --baseline) share of the jamming caused by the
                       entrances/exits: 1 - jam_fraction(same run, same seed,
                       exit_prob 0) / jam_fraction, over the second half, clipped
                       to 0-1. Empty if the run is (almost) unjammed (< {100 * JAM_PRESENT:g} %
                       jammed), has no arms or already has exit_prob 0. The
                       baseline run's own numbers are the baseline_* columns.
  spacing_gini         how unevenly the road is shared (Gini of each car's road
                       length: bumper gap + the mean vehicle length, so trucks
                       alone do not raise it): 0 = even, ~0.35 = random gaps,
                       towards 1 = packed queues or platoons. Good for initial
                       spacing and platoons, not a jam detector.
  occupancy            share of the road surface (all lanes) covered by the
                       vehicles, with their real lengths and widths
  clustering           spread of the cars over 6-24 sectors beyond random scatter:
                       0 = even or random, ~1 = half the ring empty (often 0 in
                       dense flow)
  order_param          0 = balanced around the ring, 1 = all cars on one side
  order_baseline       sqrt(pi / 4N): what N randomly placed cars give for
                       order_param; only values well above it are one-sided
  emergency_per_min    Harsh braking (episodes/min): the driver model asking for
                       more than the braking limit (an over-reaction after a
                       late or delayed response), or the no-overlap constraint
                       stopping a car harder. The limit is {MAX_DECEL:g} m/s2 on a dry
                       road, {braking_limits()} m/s2 wet / snow / ice. 0 with
                       the default drivers on a ring without exits; a few per
                       minute from merging at entrances/exits; many more on snow
                       and ice, where comfortable braking is close to the limit.
  lane_changes_per_min  lane changes per minute, whole ring
  exits_per_min        cars leaving the ring at the exits, per minute
  missed_per_min       leavers that missed their exit, per minute (not on the
                       outer lane in time, or stuck waiting to move over): they
                       go round again
  mean_trip_time       time on the ring [s] from entering to leaving of the cars
                       that left (each sample: those of the last {RATE_WINDOW:g} s; mean
                       over the window). Empty if nobody left
  trucks               trucks on the ring (mean); truck_speed / car_speed: mean
                       speed of the trucks and of the other vehicles (empty if
                       there are none)
  wave_speed           speed of jam patterns along the ring [m/s]; < 0 = moving
                       backwards against the traffic, as stop-and-go waves do
  wave_speed_kmh       the same jam wave speed in km/h
  cars, queued         cars on the ring and waiting at all entrances (means). In
                       open mode (traffic_mode=open) the number on the ring varies
  longest_queue        cars waiting at the longest queue (mean); peak_queue = the
                       longest queue seen
  arrivals_per_min     cars arriving at the entrances. Open mode: every random
                       arrival, blocked ones included; closed mode: the leavers'
                       replacements that join a queue (the rest are slotted in)
  entries_per_min      cars entering the ring from the queues
  blocked_per_min      open mode: arrivals that found their queue full
                       (max_queue) and were turned away - spill-back
  demand_per_min       open mode: arrivals the demand settings ask for (mean over
                       the window). The peak profile averages exactly the
                       entrance chance only over whole {PEAK_PERIOD / 60:g}-minute cycles
  mean_entry_delay     queue wait [s] of the cars that entered: from joining the
                       queue to entering the ring. Only the queue wait, not the
                       HCM control delay. Its level of service is entry_los:
                       {los_text("<=")},
                       and F whenever entry_vc > 1
  entry_vc             open mode: demand / capacity (v/c) of the busiest
                       entrance = its arrivals per second (blocked ones
                       included) / its entry rate while it had a queue. Above 1
                       it gets more arrivals than it can let in: its queue grows
                       or turns them away
  entry_los            HCM-style level of service of the entry delay (text):
                       {', '.join(f'{k} <= {v:g}' for k, v in LOS_LIMITS)}, F above (s), and
                       F whenever entry_vc > 1, as the HCM rates an entry lane
                       (with max_queue turning arrivals away, the wait of the
                       cars that got in no longer shows it)
  throughput_per_h     cars leaving the ring per hour"""

# Recorder columns averaged over a measurement window.
MEAN_COLUMNS = ("cars", "queued", "longest_queue", "density", "mean_speed", "mean_free_speed", "speed_std",
                "flux", "jam_fraction", "speed_deficit", "n_jams", *JAM_ORIGINS, "occupancy", "spacing_gini",
                "clustering", "order_param", "order_baseline", "mean_trip_time", "demand_per_min",
                "trucks", "truck_speed", "car_speed")
# Columns of the sweep's printed table (means across seeds); OPEN_COLUMNS are added when a run is in open
# mode, TRUCK_COLUMNS when a run has trucks.
SUMMARY_COLUMNS = ("measured_density", "measured_mean_speed", "flux", "jam_fraction", "congestion_onset",
                   "stops_per_car_min", "stop_go_onset", "n_jams", "following_jam_share", "spacing_gini",
                   "order_param", "emergency_per_min", "wave_speed", "mean_entry_delay")
OPEN_COLUMNS = ("cars", "queued", "demand_per_min", "blocked_per_min", "entry_vc", "throughput_per_h")
TRUCK_COLUMNS = ("trucks", "truck_speed", "car_speed")


# ----------------------------------------------------------------- analysis
def stop_go_index(frame: pd.DataFrame, window: float = RATE_WINDOW) -> np.ndarray:
    """``stops_per_car_min`` counted over a full ``window`` from the very start.

    metrics.snapshot divides the stops of the last ``window`` s by
    ``min(t, window)``, so in the first window a single early stop already
    looks like a wave. Dividing by the whole window instead (no stops before
    t = 0) changes nothing once ``t >= window`` and can only understate the
    rate before, so a crossing in the first window is a real one.
    """
    t = frame["time"].to_numpy(float)
    return frame["stops_per_car_min"].to_numpy(float) * np.clip(t / window, 0.0, 1.0)


def stop_go_onset(frame: pd.DataFrame, threshold: float = STOP_GO_ONSET) -> float:
    """Time of the first jam: first sample whose :func:`stop_go_index` exceeds ``threshold`` (NaN if never)."""
    hit = stop_go_index(frame) > threshold
    return round(float(frame["time"].to_numpy()[hit.argmax()]), 6) if hit.any() else float("nan")


def rolling_jam_fraction(frame: pd.DataFrame, window: float = RATE_WINDOW) -> np.ndarray:
    """Trailing ``window``-second mean of ``jam_fraction`` at every sample.

    Counted from t = 0: in the first window the mean covers only the samples
    so far, so a start that is already jammed shows up at once.
    """
    t = frame["time"].to_numpy(float)
    jam = np.nan_to_num(frame["jam_fraction"].to_numpy(float))
    total = np.concatenate(([0.0], np.cumsum(jam)))
    first = np.searchsorted(t, t - window + 1e-9, side="left")  # samples in (t - window, t]
    return (total[1:] - total[first]) / np.maximum(np.arange(1, t.size + 1) - first, 1)


def congestion_onset(frame: pd.DataFrame, threshold: float = CONGESTION_ONSET) -> float:
    """Time of the onset of congestion: first sample whose :func:`rolling_jam_fraction` exceeds ``threshold``.

    NaN if never; ~0 for a start that is already jammed.
    """
    hit = rolling_jam_fraction(frame) > threshold
    return round(float(frame["time"].to_numpy()[hit.argmax()]), 6) if hit.any() else float("nan")


def following_share(frame: pd.DataFrame) -> float:
    """Share of the jams whose front car is just following (NaN if no jams).

    An upper bound on natural jams: a wave set off at an exit keeps travelling
    backwards with ordinary followers at its front. Compare with exit_prob = 0
    (:func:`arm_share`) for what the entrances/exits really cause.
    """
    attributed = float(frame[list(JAM_ORIGINS)].to_numpy().sum())
    return float(frame["jams_following"].sum()) / attributed if attributed > 0 else float("nan")


def arm_share(jam_fraction: float, baseline_jam_fraction: float) -> float:
    """Share of the jamming caused by entrances/exits: 1 - jammed(exit_prob = 0) / jammed(run), in [0, 1].

    NaN when the run is (almost) unjammed (below JAM_PRESENT) or either value is missing.
    """
    if not (np.isfinite(jam_fraction) and np.isfinite(baseline_jam_fraction) and jam_fraction >= JAM_PRESENT):
        return float("nan")
    return float(np.clip(1.0 - baseline_jam_fraction / jam_fraction, 0.0, 1.0))


def rank_correlations(df: pd.DataFrame, names: list[str], metrics: list[str]) -> pd.DataFrame:
    """Spearman rank correlation of each setting in ``names`` with each metric, the strongest first.

    Rows: the settings, ordered by their largest absolute correlation with any
    of the metrics; columns: the metrics. A choice is ranked in the order its
    parameter lists the choices. NaN where either side does not vary. Only
    the runs with both values count, ranked among themselves (a metric is
    missing, e.g. the entry delay when no car entered).
    """
    table = {}
    for name in names:
        x = df[name]
        if not pd.api.types.is_numeric_dtype(x):
            spec = PARAM_BY_NAME.get(name)
            order = list(spec.choices) if spec and spec.choices else list(dict.fromkeys(x))
            x = x.map({v: i for i, v in enumerate(order)})
        x = x.astype(float)
        row = {}
        for metric in metrics:
            y = df[metric].astype(float) if metric in df else pd.Series(np.nan, index=df.index)
            both = x.notna() & y.notna()
            rx, ry = x[both].rank(), y[both].rank()
            varies = rx.nunique() > 1 and ry.nunique() > 1
            row[metric] = float(np.corrcoef(rx, ry)[0, 1]) if varies else float("nan")
        table[name] = row
    frame = pd.DataFrame.from_dict(table, orient="index", columns=list(metrics))
    strength = frame.abs().max(axis=1).fillna(0.0)
    return frame.loc[strength.sort_values(ascending=False, kind="stable").index]


def wave_speed(times: np.ndarray, speeds: np.ndarray, radius: float, t_from: float = 0.0) -> float:
    """Speed [m/s along ``radius``] at which speed patterns travel round the ring.

    The speed profiles (``speeds[time, angle bin]``) are gap-filled, smoothed
    and stripped of their time-averaged shape, so the slow-downs that are
    always at the same place (entries, exits) drop out and only moving jams
    remain. The angular shift that best aligns profiles ``k`` samples apart is
    then followed lag by lag; its slope against the lag is the wave speed.
    Negative = the pattern moves against the traffic, as stop-and-go waves do.
    """
    keep = (times >= t_from) & ~np.all(np.isnan(speeds), axis=1)
    t, s = times[keep], speeds[keep]
    if t.size < 8:
        return float("nan")
    bins = s.shape[1]
    x = np.arange(bins)
    field = np.array([np.interp(x, x[ok], row[ok], period=bins) for row, ok in zip(s, ~np.isnan(s))])
    spectrum = np.fft.rfft(field, axis=1) * np.exp(-2 * (np.pi * np.fft.rfftfreq(bins) * WAVE_SMOOTH) ** 2)
    spectrum -= spectrum.mean(axis=0)  # remove the stationary profile
    spectrum[:, 0] = 0.0  # and each frame's mean speed
    power = (np.fft.irfft(spectrum, n=bins, axis=1) ** 2).sum(axis=1)
    if power.sum() <= 1e-9 * t.size:
        return float("nan")

    step = float(np.median(np.diff(t)))
    bin_arc = TWO_PI * radius / bins
    reach = int(np.ceil(PATTERN_SPEED_CAP * step / bin_arc)) + 1
    lags, shifts, shift = [], [], 0.0
    for k in range(1, min(int(WAVE_MAX_LAG / step), t.size // 2) + 1):
        corr = np.fft.irfft((np.conj(spectrum[:-k]) * spectrum[k:]).sum(axis=0), n=bins)
        corr /= np.sqrt(power[:-k].sum() * power[k:].sum())
        near = int(round(shift)) + np.arange(-reach, reach + 1)
        j = int(near[np.argmax(corr[near % bins])])
        if corr[j % bins] < WAVE_MIN_CORR or abs(j - int(round(shift))) >= reach:
            break  # decorrelated, or the peak hit the edge of the search window: pattern lost
        y0, y1, y2 = corr[(j - 1) % bins], corr[j % bins], corr[(j + 1) % bins]
        curvature = y0 - 2 * y1 + y2
        shift = j + (0.5 * (y0 - y2) / curvature if curvature < 0 else 0.0)  # sub-bin peak
        lags.append(k * step)
        shifts.append(shift)
    if len(lags) < 4:
        return float("nan")
    lags_a, shifts_a = np.array(lags), np.array(shifts)
    return float(lags_a @ shifts_a / (lags_a @ lags_a) * bin_arc)


def _counters(sim: RingSim) -> dict[str, float | np.ndarray]:
    return {"exits": sim.n_exited, "missed": sim.n_missed, "lane_changes": sim.n_lane_changes,
            "emergency": sim.n_emergency, "stops": sim.n_stops, "arrivals": sim.n_arrived,
            "entries": sim.n_entered, "blocked": sim.n_blocked, "entry_wait": sim.entry_wait_total,
            # per entrance, for entry_vc
            "arm_arrived": sim.arm_arrived.copy(), "arm_entered": sim.arm_entered.copy(),
            "queue_busy": sim.queue_busy.copy()}


def measured_names(names) -> dict[str, str]:
    """Renames that give measurements sharing a parameter's name a ``measured_`` prefix."""
    return {k: f"measured_{k}" for k in names if k in CONFIG_NAMES}


def settings(sim: RingSim) -> dict:
    """Every parameter of ``sim`` (named like the CLI ``--set`` names) and its seed."""
    return {**{name: getattr(sim.cfg, name) for name in CONFIG_NAMES}, "seed": sim.seed}


def summarise(sim: RingSim, rec: Recorder, t0: float, start: dict[str, int]) -> dict:
    """Metrics from ``t0`` to now (``start`` = :func:`_counters` at ``t0``).

    Sampled metrics are averaged over the window; event rates come from the
    engine's counters, so they count exactly the events inside the window
    (``mean_entry_delay`` too: the queue wait of exactly the cars that entered
    in it; and ``entry_vc``). ``stop_go_onset`` and ``congestion_onset`` are
    measured from the start of the run. ``entry_los`` is the one text column.
    Metrics that share a parameter's name get a ``measured_`` prefix.
    """
    frame = rec.frame()
    window = frame[frame["time"] >= t0]
    span = max(sim.time - t0, 1e-9)
    mean = {name: float(window[name].mean()) for name in MEAN_COLUMNS}
    now = _counters(sim)

    def per_min(key: str) -> float:
        return 60.0 * (now[key] - start[key]) / span

    # Rolling rates sampled at least one window after t0 count only events inside the window.
    settled = window[window["time"] >= t0 + RATE_WINDOW] if span > 2 * RATE_WINDOW else window
    times, speeds = rec.spacetime_array()
    jams = mean["jam_fraction"] >= JAM_PRESENT
    wave = wave_speed(times, speeds, float(sim.radii.mean()), t0) if jams else float("nan")
    entered = now["entries"] - start["entries"]
    delay = (now["entry_wait"] - start["entry_wait"]) / entered if entered > 0 else float("nan")
    vc = entry_vc(*(now[k] - start[k] for k in ("arm_arrived", "arm_entered", "queue_busy")), span) \
        if sim.open_demand else float("nan")  # closed mode has no demand of its own
    row = {
        **{k: mean[k] for k in ("cars", "queued", "density", "mean_speed", "mean_free_speed", "speed_std",
                                "flux", "trucks", "truck_speed", "car_speed")},
        "detector_flux": float(settled["detector_flux"].mean()),
        "jam_fraction": mean["jam_fraction"],
        "congestion_onset": congestion_onset(frame),
        "stops_per_car_min": per_min("stops") / mean["cars"] if mean["cars"] > 0 else float("nan"),
        "stop_go_onset": stop_go_onset(frame),
        **{k: mean[k] for k in ("speed_deficit", "n_jams", *JAM_ORIGINS)},
        "following_jam_share": following_share(window),
        **{k: mean[k] for k in ("occupancy", "spacing_gini", "clustering", "order_param", "order_baseline")},
        "emergency_per_min": per_min("emergency"),
        "lane_changes_per_min": per_min("lane_changes"),
        "exits_per_min": per_min("exits"),
        "missed_per_min": per_min("missed"),
        "mean_trip_time": mean["mean_trip_time"],
        "wave_speed": wave,
        "wave_speed_kmh": 3.6 * wave,
        "longest_queue": mean["longest_queue"],
        "peak_queue": float(window["longest_queue"].max()),
        "demand_per_min": mean["demand_per_min"],
        "arrivals_per_min": per_min("arrivals"),
        "entries_per_min": per_min("entries"),
        "blocked_per_min": per_min("blocked"),
        "mean_entry_delay": delay,
        "entry_vc": vc,
        "entry_los": level_of_service(delay, vc),
        "throughput_per_h": 60.0 * per_min("exits"),
    }
    # Measurements named like a parameter (density, mean_speed, ...) must not overwrite its setting.
    renames = measured_names(row)
    return {renames.get(k, k): v for k, v in row.items()}


# --------------------------------------------------------------- simulation
def advance(sim: RingSim, rec: Recorder, seconds: float, on_sample: Callable[[], None] | None = None) -> None:
    """Step ``sim`` for ``seconds``, recording; ``on_sample`` runs after each new sample."""
    for _ in range(int(round(seconds / sim.cfg.dt))):
        sim.step()
        before = len(rec.rows)
        rec.update()
        if on_sample is not None and len(rec.rows) > before:
            on_sample()


def window_sampled(start: float, measure: float, dt: float, interval: float) -> bool:
    """Whether a run's measured window, ``measure`` s from ``start`` s, holds a sample: the steps of
    :func:`advance` (the warm-up, then the window) and the sample times of Recorder.update, which
    samples on the first step and then once every ``interval`` s, worked out without simulating."""
    t, due, last = 0.0, 0.0, -1.0
    for _ in range(int(round(start / dt)) + int(round(measure / dt))):
        t += dt  # as RingSim.step adds up its time
        if interval <= dt:  # every step is sampled
            last = t
        elif t + 1e-9 >= due:
            last = t
            while due <= t + 1e-9:
                due += interval
    return last >= start


def _check_sampled(start: float, measure: float, interval: float, configs, parser: argparse.ArgumentParser,
                   window: str, longer: str) -> None:
    """A parser error, before anything runs, when the measured window would hold no sample (its metrics all
    NaN, or no samples at all)."""
    for dt in sorted({cfg.dt for cfg in configs}):
        if not window_sampled(start, measure, dt, interval):
            parser.error(f"no sample in {window} ({start:g}-{start + measure:g} s at a time step of {dt:g} s, "
                         f"sampling every --interval {interval:g} s): make {longer} longer or --interval shorter")


def free_speed(sim: RingSim) -> float:
    """Free speed of an average driver on the middle lane, on this road (top of the speed colour scale)."""
    return float(min(sim.eff.mean_speed, np.sqrt(sim.eff.lateral_accel * sim.radii.mean())))


def _sweep_job(job: tuple[dict, int, float, float, float]) -> dict:
    """One sweep run (top level so worker processes can import it)."""
    changes, seed, warmup, measure, interval = job
    sim = RingSim(SimConfig().updated(**changes, seed=seed))
    rec = Recorder(sim, interval)
    advance(sim, rec, warmup)
    start = _counters(sim)
    advance(sim, rec, measure)
    return {**settings(sim), **summarise(sim, rec, warmup, start)}


# ------------------------------------------------------------- CLI parsing
def _assignment(text: str) -> tuple[str, str]:
    name, sep, value = text.partition("=")
    name = name.strip().replace("-", "_")
    if not sep or not value.strip():
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {text!r}")
    if name not in CONFIG_NAMES:
        raise argparse.ArgumentTypeError(f"unknown parameter {name!r}; choose from: {', '.join(CONFIG_NAMES)}")
    return name, value.strip()


_coerce = cli.setting_value  # (strict: lanes=2.5 is an error, not rounded)


def range_values(start: float, stop: float, step: float) -> list[float]:
    """The values of a ``start:stop:step`` range (stop included, rounded to 10 decimals)."""
    count = int(np.floor((stop - start) / step + 1e-9)) + 1
    return [round(start + i * step, 10) for i in range(count)]


def _values(name: str, spec: str) -> list:
    """``start:stop:step`` (stop included) or a comma list, converted to the parameter's type."""
    if ":" in spec:
        _no_choice_range(name)
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"{name}: range must be start:stop:step, got {spec!r}")
        try:
            start, stop, step = (float(p) for p in parts)
        except ValueError:
            raise ValueError(f"{name}: bad range {spec!r}: start, stop and step must be numbers") from None
        if not np.isfinite([start, stop, step]).all():
            also = f" ({name}=inf goes in a comma list, e.g. {name}=40,80,inf)" \
                if np.isinf([start, stop]).any() and _takes_inf(name) else ""
            raise ValueError(f"{name}: a range needs a finite start, stop and step, got {spec!r}{also}")
        if step <= 0 or stop < start:
            raise ValueError(f"{name}: need step > 0 and stop >= start, got {spec!r}")
        raw = range_values(start, stop, step)
    else:
        raw = [v.strip() for v in spec.split(",") if v.strip()]
    values = list(dict.fromkeys(_coerce(name, v) for v in raw))
    if not values:
        raise ValueError(f"{name}: no values in {spec!r}")
    return values


def _sample_spec(name: str, spec: str) -> tuple[float, float] | list:
    """A ``--sample`` variable: ``start:stop[:step]`` gives the range (lo, hi) drawn from, a comma list the
    values dealt out evenly. The step of a range is not used: the values lie on the slider step, one in each
    of the N strata, or, when the range holds no more slider values than N, those values dealt out evenly."""
    if ":" not in spec:
        return _values(name, spec)
    _no_choice_range(name)
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"{name}: range must be start:stop or start:stop:step, got {spec!r}")
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError:
        raise ValueError(f"{name}: bad range {spec!r}") from None
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"{name}: a range needs a finite start and stop, got {spec!r}")
    if hi < lo:
        raise ValueError(f"{name}: need stop >= start, got {spec!r}")
    return lo, hi


def _takes_inf(name: str) -> bool:
    """Whether ``name`` = inf is a valid setting (Density, Reaction distance)."""
    try:
        SimConfig().updated(**{name: float("inf")})
    except (TypeError, ValueError):
        return False
    return True


def _no_choice_range(name: str) -> None:
    """A range for a choice (road_condition=dry:ice) is an error that names the parameter."""
    spec = PARAM_BY_NAME.get(name)
    if spec and spec.choices:
        raise ValueError(f"{name}: a choice takes a comma list of values, not a range (e.g. "
                         f"{name}={','.join(spec.choices[:2])})")


def _differ(cfg: SimConfig, names=CONFIG_NAMES) -> dict:
    """The settings of ``cfg`` that differ from the defaults."""
    return changed_settings({n: getattr(cfg, n) for n in names})


@dataclass
class Base:
    """The settings every run starts from: defaults < preset / random < --set and the parameter flags."""

    changes: dict  # every setting that differs from the defaults (preset or random, then the explicit ones)
    explicit: dict  # --set and the parameter flags only
    preset: str | None = None  # the preset's name
    random: int | None = None  # the seed of --random

    @property
    def cfg(self) -> SimConfig:
        return SimConfig().updated(**self.changes)

    def text(self) -> str:
        """'preset etoile', 'random settings (seed 3)' or '' (as the sweep viewer names it, from the meta file)."""
        return base_text(self.preset, self.random)


def _base(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Base:
    """--preset or --random, then --set and the parameter flags (parser errors for unknown or invalid ones).

    With --random the explicit settings take part in the links of the draw (presets.random_changes'
    ``fixed``): ``--random 40 --traffic-mode open`` draws an open-mode start density, and a given lane
    count a radius for it, rather than putting them on top of a draw made for other settings."""
    explicit = _overrides(args, parser)
    first, preset, random_seed = {}, None, None
    if args.preset is not None or args.random is not None:
        from roundabout_sim import presets

        try:
            if args.preset is not None:
                preset = presets.get(args.preset).name
                first = _differ(presets.apply(preset))
            else:
                random_seed = args.random
                first = presets.random_changes(random_seed, fixed=explicit)
        except (KeyError, ValueError) as err:  # an unknown preset lists the names
            parser.error(f"--{'preset' if args.preset is not None else 'random'}: {err}")
    base = Base({**first, **explicit}, explicit, preset, random_seed)
    try:
        base.cfg  # noqa: B018 - validates the combination
    except ValueError as err:
        parser.error(f"invalid settings: {err}")
    return base


@dataclass
class Design:
    """The design points of a sweep, in run order: each point's varied settings and its base seed."""

    kind: str  # "grid", "plan" or "sample"
    names: list[str]  # the varied settings: x first, then the series, then the rest
    points: list[dict]
    seeds: list[int]
    varied: list[dict]  # meta.json: {"name", "values"} or {"name", "range"} per varied setting, in names order
    shape: str  # how the run count comes about, for the first line printed
    sources: list[str] | None = None  # where each point comes from, for error messages (a plan: file and line)

    @property
    def keys(self) -> list[str]:
        """The varied settings as the CSV's first columns: the last one first, the x just before the seed."""
        return self.names[::-1]

    def location(self, point: int) -> str:
        """Where point number ``point`` is, for an error message: 'plan p.csv, line 3', 'point 2: density=40',
        'n_lanes=2 density=40'."""
        if self.sources:
            return self.sources[point]
        return _where(self, point).strip().rstrip(":")


def _variables(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[tuple[str, str]]:
    """--x, --series and the --vary flags in that order, as (name, spec); each name only once."""
    if args.series and not args.x:
        parser.error("--series needs --x (or give both as --vary: the first is the x, the second the series)")
    given = [v for v in (args.x, args.series) if v] + list(args.vary)
    names = [name for name, _ in given]
    twice = next((n for n in names if names.count(n) > 1), None)
    if twice:
        parser.error(f"--x, --series and --vary must be different parameters: {twice} is given twice")
    return given


def _grid(given: list[tuple[str, str]], args: argparse.Namespace, parser: argparse.ArgumentParser) -> Design:
    """Every combination of the values (full factorial), the x varying fastest."""
    try:
        variables = [(name, _values(name, spec)) for name, spec in given]
    except ValueError as err:
        parser.error(str(err))
    if len(variables) > 1 and len(variables[1][1]) > MAX_SERIES:
        flag = "--series" if args.series else "--vary"
        parser.error(f"at most {MAX_SERIES} {flag} values for {variables[1][0]}, the 2nd variable (one colour each "
                     "in the figure); vary it 3rd or later to take more")
    names = [name for name, _ in variables]
    keys = names[::-1]
    points = [dict(zip(keys, combo)) for combo in itertools.product(*(values for _, values in variables[::-1]))]
    sizes = [len(values) for _, values in variables]
    if len(variables) <= 2:  # as the sweep has always put it
        shape = f"{sizes[0]} x {sizes[1] if len(sizes) > 1 else 1} series"
    else:
        shape = " x ".join(f"{n} {name}" for name, n in zip(names, sizes))
    return Design("grid", names, points, [args.seed] * len(points),
                  [{"name": name, "values": values} for name, values in variables], shape)


def _read_plan(path: str, base: Base, args: argparse.Namespace, parser: argparse.ArgumentParser) -> Design:
    """One design point per row of a CSV: columns named like the parameters, plus an optional seed column
    (that row's base seed). An empty cell takes the base setting, and so do the cells missing at the end
    of a short row (as some editors write rows with empty trailing cells): a note names those rows.

    Read with the csv module rather than pandas, which would rename a column
    given twice ('density.1') or a blank one left by a trailing comma
    ('Unnamed: 2'): the header is checked as written, a column with neither a
    name nor a value is dropped, and errors give the file's line numbers.
    """
    try:
        with open(path, newline="", encoding="utf-8-sig") as file:
            reader = csv.reader(file, skipinitialspace=True)
            rows = [(reader.line_num, [cell.strip() for cell in row]) for row in reader]
    except (OSError, UnicodeDecodeError, csv.Error) as err:
        parser.error(f"cannot read the plan {path}: {err}")
    rows = [(line, row) for line, row in rows if any(row) or len(row) > 1]  # skip blank lines
    if not rows:
        parser.error(f"plan {path} is empty: it needs a header row of parameter names")
    (_, header), body = rows[0], rows[1:]
    given = {line: len(row) for line, row in body}  # cells per row as written (before the padding)
    width = max(len(row) for _, row in rows)
    header += [""] * (width - len(header))
    body = [(line, row + [""] * (width - len(row))) for line, row in body]
    keep = []
    for j, name in enumerate(header):
        if name:
            keep.append(j)
        elif any(row[j] for _, row in body):
            line = next(line for line, row in body if row[j])
            parser.error(f"plan {path}, line {line}: column {j + 1} has values but no name in the header")
    columns = [header[j].replace("-", "_") for j in keep]
    unknown = [c for c in columns if c not in CONFIG_NAMES and c != "seed"]
    if unknown:
        parser.error(f"plan {path}: unknown column(s) {', '.join(map(repr, unknown))}; the columns must be "
                     f"parameter names or seed. Valid names: seed, {', '.join(CONFIG_NAMES)}")
    twice = next((c for c in columns if columns.count(c) > 1), None)
    if twice:
        parser.error(f"plan {path}: column {twice!r} is given twice")
    if not body:
        parser.error(f"plan {path} has no rows")
    named = max(keep) + 1 if keep else 0  # the cells a row needs to reach the last named column
    short = [line for line, _ in body if given[line] < named]
    if short:
        missing = [columns[k] for k, j in enumerate(keep) if j >= min(given[line] for line in short)]
        where = ", ".join(map(str, short[:8])) + (" ..." if len(short) > 8 else "")
        print(f"note: plan {path}, line{'s' * (len(short) > 1)} {where}: fewer cells than the header; the missing "
              f"ones ({', '.join(missing)}) take the base setting, as empty cells do")
    points, seeds, lines = [], [], []
    for line, row in body:
        point, seed = {}, args.seed
        for name, text in zip(columns, (row[j] for j in keep)):
            if not text:
                continue  # the base setting
            try:
                if name == "seed":
                    seed = float(text)
                    if not seed.is_integer() or seed < 0:
                        raise ValueError("a seed must be a whole number >= 0")
                    seed = int(seed)
                else:
                    point[name] = _coerce(name, text)
            except ValueError as err:
                parser.error(f"plan {path}, line {line}: {name}={text!r}: {err}")
        points.append(point)
        seeds.append(seed)
        lines.append(line)
    # Varied: the columns that take more than one value (with empty cells at the base setting).
    cfg = base.cfg
    names = [n for n in columns if n != "seed" and len({_key(p.get(n, getattr(cfg, n))) for p in points}) > 1]
    for n in names:  # an empty cell is the base setting: name it, so the CSV and the meta file agree
        for p in points:
            p.setdefault(n, getattr(cfg, n))
    shape = f"{len(points)} plan row{'s' * (len(points) != 1)}"
    return Design("plan", names, points, seeds,
                  [{"name": n, "values": _distinct([p[n] for p in points])} for n in names], shape,
                  [f"plan {path}, line {line}" for line in lines])


def plan_repeats(design: Design, jobs: list[tuple], configs: list[SimConfig], seeds: int) -> list[str]:
    """Plan rows that run one simulation more than once, the same settings with the same seed (each row
    runs its seed + k for k < ``seeds``): 'lines 2 and 3 with seeds 0 and 1', one per set of rows."""
    rows: dict[tuple, list[int]] = {}
    for i, (job, cfg) in enumerate(zip(jobs, configs)):
        key = (tuple(_key(getattr(cfg, n)) for n in CONFIG_NAMES), job[1])
        rows.setdefault(key, []).append(i // seeds)
    repeats: dict[tuple, list[int]] = {}  # the rows -> the seeds they share
    for (_, seed), found in rows.items():
        found = tuple(dict.fromkeys(found))
        if len(found) > 1:
            repeats.setdefault(found, []).append(seed)
    return [f"lines {and_join(design.location(r).rsplit('line ', 1)[-1] for r in found)} with "
            f"seed{'s' * (len(shared) > 1)} {and_join(shared)}" for found, shared in repeats.items()]


def _sample_groups(args: argparse.Namespace) -> list[str] | None:
    """--sample-groups as the tabs name them, in tab order ("drivers , ROAD" -> ["Road", "Drivers"]), or
    None for every group; ValueError for an unknown one."""
    from roundabout_sim import presets

    if not args.sample_groups:
        return None
    return presets.group_names([g.strip() for g in args.sample_groups.split(",") if g.strip()])


def _sample(given: list[tuple[str, str]], base: Base, args: argparse.Namespace,
            parser: argparse.ArgumentParser) -> Design:
    """``--sample N`` points: a Latin hypercube over the --vary ranges, else over the plausible ranges
    (presets.RANDOM_RANGES) of every parameter, or of the --sample-groups ones, not set explicitly.

    Without --vary every plausible range is cut into N strata, each sampled
    once, and each list of choices (lanes, entrances, road condition, ...) is
    dealt out evenly: the design covers the ranges evenly, so the weights and
    links of ``presets.random_changes`` (e.g. the radius drawn for the lane
    count) do not apply.
    """
    from roundabout_sim import presets

    n = args.sample
    if n < 1:
        parser.error("--sample needs N >= 1")
    if given and args.sample_groups:
        parser.error("--sample-groups works only without --vary (then the --vary ranges are sampled)")
    try:
        if given:
            specs = {name: _sample_spec(name, spec) for name, spec in given}
        else:
            specs = presets.plausible_ranges(_sample_groups(args))
    except ValueError as err:
        parser.error(f"--sample: {err}")
    if not given:
        # Settings given with --set or their flags stay fixed: they are not drawn.
        kept = [name for name in specs if name in base.explicit]
        if kept:
            print(f"note: {', '.join(kept)} keep{'s' * (len(kept) == 1)} the value set with --set / the flags; "
                  "the sample draws the other settings")
        specs = {name: s for name, s in specs.items() if name not in base.explicit}
        from_base = [name for name in base.changes if name not in base.explicit]  # the preset's or --random's
        if base.text() and from_base and all(name in specs for name in from_base):
            flag = "--preset" if base.preset else "--random"
            parser.error(f"{flag} would do nothing here: --sample draws every setting it gives. Leave it out, or "
                         "draw only some settings with --sample-groups")
    try:
        # Lists go in as text, so that two numbers are two values and not a range.
        points = presets.latin_hypercube(n, {name: list(map(str, s)) if isinstance(s, list) else s
                                             for name, s in specs.items()}, args.seed)
    except ValueError as err:
        parser.error(f"--sample: {err}")
    if given:
        names = [name for name, _ in given]
    else:  # those that came out with more than one value (all of them, unless N is tiny)
        names = [p.name for p in PARAMS if p.name in specs and len({_key(pt[p.name]) for pt in points}) > 1]
    varied = []
    for name in names:
        s = specs[name]
        if isinstance(s, tuple):
            varied.append({"name": name, "range": [float(s[0]), float(s[1])]})
        else:  # the values given (--vary), or the plausible ones drawn, in the listed order
            drawn = {_key(pt[name]) for pt in points}
            varied.append({"name": name, "values": s if given else [v for v in s if _key(v) in drawn]})
    return Design("sample", names, points, [args.seed] * n, varied, f"{n} sampled points")


def _key(value):
    """A hashable stand-in for a setting value (numbers rounded, so 0.1 and 0.1000000001 are one value)."""
    return round(float(value), 9) if isinstance(value, (int, float)) and not isinstance(value, bool) else value


def _distinct(values: list) -> list:
    """The different values: numbers ascending, text in order of appearance."""
    unique = list(dict.fromkeys(values))
    return sorted(unique) if all(isinstance(v, (int, float)) for v in unique) else unique


def _add_param_flags(parser: argparse.ArgumentParser) -> None:
    """--set plus one flag per model parameter, generated from config.PARAMS."""
    parser.add_argument("--set", action="append", type=_assignment, default=[], metavar="NAME=VALUE",
                        help="set any model parameter (repeatable), e.g. --set n_lanes=2")
    group = parser.add_argument_group("model parameters (shorthand for --set)")
    defaults = SimConfig()
    for spec in PARAMS:
        flags = list(dict.fromkeys([f"--{spec.name.replace('_', '-')}", f"--{spec.name}"]))
        unit = f" [{_plain(spec.unit)}]" if spec.unit else ""
        if spec.choices:
            group.add_argument(*flags, dest=spec.name, choices=spec.choices, default=None,
                               help=f"{spec.label} (default {getattr(defaults, spec.name)}). "
                                    f"{spec.help}".replace("%", "%%"))
        else:
            group.add_argument(*flags, dest=spec.name, default=None, metavar="X",
                               help=f"{spec.label}{unit}, {spec.lo:g}-{spec.hi:g} "
                                    f"(default {getattr(defaults, spec.name):g}). {spec.help}".replace("%", "%%"))


def _overrides(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    raw = {spec.name: getattr(args, spec.name) for spec in PARAMS if getattr(args, spec.name) is not None}
    raw.update(dict(args.set))
    try:
        changes = {name: _coerce(name, value) for name, value in raw.items()}
    except ValueError as err:
        parser.error(str(err))
    for name, value in changes.items():
        if note := cli.range_note(name, value):
            print(note)
    return changes


def _road(cfg: SimConfig) -> str:
    return f"{cfg.n_lanes} lane{'s' * (cfg.n_lanes != 1)}, " + \
        (f"{cfg.n_arms} entrances/exits" if cfg.n_arms else "closed ring (no entrances/exits)")


def _describe(cfg: SimConfig) -> str:
    road = f", {cfg.road_condition} road" if cfg.road_condition != "dry" else ""
    trucks = f", {100 * cfg.truck_share:g} % trucks of {cfg.truck_length:g} m" if cfg.truck_share > 0 else ""
    reaction = f" ± {cfg.reaction_std:g}" if cfg.reaction_std > 0 else ""
    headway = f" ± {cfg.headway_std:g}" if cfg.headway_std > 0 else ""
    return (f"{_road(cfg)}, radius {cfg.radius:g} m{road}, car {cfg.car_length:g} m{trucks} | "
            f"density {cfg.density:g} veh/km/lane | "
            f"desired speed {cfg.mean_speed:g} ± {cfg.speed_std:g} m/s, reaction {cfg.reaction_time:g}{reaction} s / "
            f"{cfg.reaction_distance:g} m, headway {cfg.time_headway:g}{headway} s, noise {cfg.driver_noise:g} m/s² | "
            f"lane changes {cfg.lane_change_rate:g}/s ({cfg.lane_change_motive}) | start {cfg.init_spacing}"
            + (f" | {_demand(cfg)}" if cfg.traffic_mode == "open" else ""))


def _demand(cfg: SimConfig) -> str:
    """Open-mode demand in words (the density above is then only the start)."""
    shape = "peak profile" if cfg.demand_profile == "peak" else "constant"
    imbalance = f", imbalance {cfg.demand_imbalance:g}" if cfg.demand_imbalance else ""
    return (f"open demand {cfg.entry_rate:g}/s per entrance ({3600 * cfg.entry_rate:.0f} veh/h, {shape}{imbalance}), "
            f"max queue {cfg.max_queue}, entry gap {cfg.entry_gap:g} s, follow-up {cfg.follow_up:g} s")


def _open(cfg: SimConfig) -> bool:
    return cfg.traffic_mode == "open"


def _peak_note(t0: float, t1: float) -> None:
    """Warn when the peak profile does not average the entrance chance over the measured window."""
    factor = mean_demand_factor("peak", t0, t1)
    if abs(factor - 1.0) > 0.01:
        print(f"note: the peak demand profile averages {factor:.2f} x the entrance chance over the measured "
              f"{t0:g}-{t1:g} s, not 1 (only over whole {PEAK_PERIOD / 60:g}-minute cycles); demand_per_min "
              "gives the demand actually offered")


def _label(name: str, value) -> str:
    spec = PARAM_BY_NAME.get(name)
    label = spec.label if spec else name
    return f"{label} = {_value_text(value)}{' ' + spec.unit if spec and spec.unit else ''}"


_axis_label = axis_label


def _fmt(value, digits: int = 2) -> str:
    return "-" if value is None or (isinstance(value, float) and not np.isfinite(value)) else f"{value:.{digits}f}"


def _pct(value, digits: int = 1) -> str:
    return _fmt(100 * value if isinstance(value, float) and np.isfinite(value) else value, digits)


# ---------------------------------------------------------------- commands
def _simulate(cfg: SimConfig, duration: float, interval: float, late_from: float,
              keep_speeds: bool = False) -> tuple[RingSim, Recorder, dict, list[np.ndarray]]:
    """One run of ``duration`` s, summarised from ``late_from``; optionally every car's speed after it."""
    sim = RingSim(cfg)
    rec = Recorder(sim, interval)
    advance(sim, rec, late_from)
    start = _counters(sim)
    car_speeds: list[np.ndarray] = []
    advance(sim, rec, duration - late_from,
            (lambda: car_speeds.append(sim.v[sim.alive].copy())) if keep_speeds else None)
    return sim, rec, summarise(sim, rec, late_from, start), car_speeds


def cmd_run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from roundabout_sim.plots import dashboard

    setup = _base(args, parser)
    if args.duration <= 0 or args.interval <= 0:
        parser.error("need --duration > 0 and --interval > 0")
    cfg = SimConfig().updated(**setup.changes, seed=args.seed)
    late_from = round(args.duration / 2, 6)
    _check_sampled(late_from, args.duration - late_from, args.interval, [cfg], parser,
                   "the second half of the run, which the summary covers,", "--duration")
    origin = f", {setup.text()}" if setup.text() else ""
    print(f"Simulating {args.duration:g} s (seed {cfg.seed}{origin}): {_plain(_describe(cfg))}")
    if _open(cfg) and cfg.demand_profile == "peak":
        _peak_note(late_from, args.duration)
    t_wall = time.time()
    sim, rec, summary, car_speeds = _simulate(cfg, args.duration, args.interval, late_from, keep_speeds=True)

    base_frame, base, skipped = None, {}, ""  # skipped: why there is no baseline, for the summary
    if args.baseline:
        if cfg.n_arms == 0:
            why, skipped = "a closed ring has no entrances/exits, so this run is its own baseline", \
                "nothing to switch off"
        elif _open(cfg):
            why, skipped = "in open mode exit_prob = 0 would fill the ring with arrivals that never leave", \
                "not meaningful in open mode"
        elif cfg.exit_prob == 0:
            why, skipped = "exit_prob is already 0, so this run is its own baseline", "nothing to switch off"
        if skipped:
            print(f"note: --baseline skipped: {why}")
            base = {f"baseline_{k}": float("nan") for k in BASELINE_COLUMNS}
        else:
            print("Baseline: the same run (seed and every other setting) with exit_prob = 0")
            _, base_rec, base_summary, _ = _simulate(cfg.updated(exit_prob=0.0), args.duration, args.interval,
                                                     late_from)
            base_frame = base_rec.frame()
            base = {f"baseline_{k}": base_summary[k] for k in BASELINE_COLUMNS}
        base["arm_jam_share"] = arm_share(summary["jam_fraction"], base["baseline_jam_fraction"])
    print(f"done in {time.time() - t_wall:.1f} s wall time")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = rec.frame()
    frame.rename(columns=measured_names(frame.columns)).to_csv(f"{out}_timeseries.csv", index=False)
    origin = {k: v for k, v in (("preset", setup.preset), ("random", setup.random)) if v is not None}
    pd.DataFrame([{**settings(sim), **origin, **summary, **base}]).to_csv(f"{out}_summary.csv", index=False)
    times, speeds = rec.spacetime_array()
    occupancy = rec.occupancy_array()
    np.savez_compressed(
        f"{out}_spacetime.npz", times=times, speeds=speeds,
        bin_edges_deg=np.linspace(0, 360, speeds.shape[1] + 1), occupancy=occupancy,
        sector_edges_deg=np.linspace(0, 360, occupancy.shape[1] + 1),
        arm_deg=np.degrees(sim.arm_angle), exit_deg=np.degrees(sim.exit_angle),
        entry_deg=np.degrees(sim.entry_angle), detector_deg=np.degrees(sim.detector_angle),
        radii=sim.radii, v_free=free_speed(sim))
    fig = dashboard(
        frame, times, speeds, occupancy,
        np.concatenate(car_speeds) if car_speeds else np.zeros(0),
        title=f"Roundabout run: {_road(sim.cfg)}, {args.duration:g} s",
        subtitle=f"{_describe(sim.cfg)} | seed {sim.seed}" + (f" | {setup.text()}" if setup.text() else ""),
        n_lanes=sim.n_lanes, arm_deg=np.degrees(sim.arm_angle),
        detector_deg=float(np.degrees(sim.detector_angle)), v_free=free_speed(sim),
        late_from=late_from, stop_index=stop_go_index(frame), onset=summary["stop_go_onset"],
        onset_threshold=STOP_GO_ONSET, following_share=summary["following_jam_share"],
        wave_speed=summary["wave_speed"], rate_window=RATE_WINDOW,
        congestion_onset=summary["congestion_onset"], congestion_threshold=CONGESTION_ONSET,
        baseline=base_frame, arm_share=base.get("arm_jam_share", float("nan")), open_demand=_open(sim.cfg),
        max_queue=sim.cfg.max_queue, entry_delay=summary["mean_entry_delay"], entry_los=summary["entry_los"])
    fig.savefig(f"{out}.png")

    rows = [
        ("Cars on the ring", _fmt(summary["cars"], 0), ""),
        ("Actual density", _fmt(summary["measured_density"], 1), "veh/km/lane"),
        ("Cars queued at entries", _fmt(summary["queued"], 1), ""),
        ("Mean speed", _fmt(summary["measured_mean_speed"]), "m/s"),
        ("Mean desired speed", _fmt(summary["mean_free_speed"]), "m/s (what the drivers on the ring want)"),
        *([("Trucks on the ring", _fmt(summary["trucks"], 1),
            f"mean ({_pct(summary['trucks'] / summary['cars'] if summary['cars'] else float('nan'), 0)} % of the "
            f"vehicles; {100 * cfg.truck_share:g} % of the new ones are trucks)"),
           ("Mean speed, cars / trucks", f"{_fmt(summary['car_speed'])} / {_fmt(summary['truck_speed'])}", "m/s")]
          if cfg.truck_share > 0 else []),
        ("Flux (whole ring)", _fmt(summary["flux"], 3), "cars/s"),
        ("Flux (detector)", _fmt(summary["detector_flux"], 3), "cars/s"),
        ("Jammed cars (congestion)", _pct(summary["jam_fraction"]),
         f"% (below {100 * SLOW_FRACTION:g} % of free speed)"),
        ("Time to congestion", _fmt(summary["congestion_onset"], 1),
         f"s from the start ({RATE_WINDOW:g} s mean of jammed > {100 * CONGESTION_ONSET:g} %; - = never)"),
        ("Speed deficit", _pct(summary["speed_deficit"]), "%"),
        ("Stop-and-go (instability)", _fmt(summary["stops_per_car_min"]),
         "stops per car per min (~0 smooth or standstill, 1-2 waves)"),
        ("Time to first stop-and-go", _fmt(summary["stop_go_onset"], 1),
         f"s from the start (stop-and-go index > {STOP_GO_ONSET:g}; - = never)"),
        ("Number of jams", _fmt(summary["n_jams"], 1), "mean count, by what the front car is doing now:"),
        ("  just following", _fmt(summary["jams_following"], 2), ""),
        ("  merging out to exit", _fmt(summary["jams_merge"], 2), ""),
        ("  slowing to exit", _fmt(summary["jams_exit"], 2), ""),
        ("  just entered", _fmt(summary["jams_entry"], 2), ""),
        ("  braked by you / slotted in", _fmt(summary["jams_other"], 2), ""),
        ("Front car following", _pct(summary["following_jam_share"], 0),
         "% of jams (upper bound on natural jams" + (")" if _open(cfg) else "; see --baseline)")),
        ("Spacing Gini", _fmt(summary["spacing_gini"]), "0 = even, ~0.35 = random gaps, -> 1 = packed queues"),
        ("Clustering", _fmt(summary["clustering"]), "0 = even or random, ~1 = half the ring empty"),
        ("Order parameter", _fmt(summary["order_param"]),
         f"0 = balanced, 1 = all on one side (random placement gives {_fmt(summary['order_baseline'])})"),
        ("Harsh braking", _fmt(summary["emergency_per_min"], 1), "episodes/min"),
        ("Lane changes", _fmt(summary["lane_changes_per_min"], 1), "per min"),
        ("Exits", _fmt(summary["exits_per_min"], 1), "per min"),
        ("Missed exits", _fmt(summary["missed_per_min"], 1), "per min"),
        ("Jam wave speed", f"{_fmt(summary['wave_speed'])} m/s = {_fmt(summary['wave_speed_kmh'], 1)}",
         "km/h (negative = backwards)"),
        *([("Demand", _fmt(summary["demand_per_min"], 1), "arrivals per min the demand settings ask for")]
          if _open(cfg) else []),
        ("Arrivals at the entrances", _fmt(summary["arrivals_per_min"], 1),
         "per min (blocked ones included)" if _open(cfg) else "per min (leavers' replacements joining a queue)"),
        ("Entries", _fmt(summary["entries_per_min"], 1), "per min"),
        *([("Blocked arrivals", _fmt(summary["blocked_per_min"], 1),
            f"per min (found their queue full: {cfg.max_queue} cars)"),
           ("Demand / capacity (v/c)", _fmt(summary["entry_vc"]),
            "busiest entrance; above 1 it gets more arrivals than it can let in")] if _open(cfg) else []),
        ("Mean entry delay", _fmt(summary["mean_entry_delay"], 1),
         f"s queue wait, LOS {summary['entry_los']}"
         + (" because v/c > 1" if summary["entry_los"] != level_of_service(summary["mean_entry_delay"]) else "")
         + " (HCM-style; not full control delay)"),
        ("Longest queue", _fmt(summary["longest_queue"], 1),
         f"cars on average (at most {_fmt(summary['peak_queue'], 0)})"),
        ("Throughput", _fmt(summary["throughput_per_h"], 0), "veh/h leaving the ring"),
    ]
    if args.baseline and base_frame is None:
        rows.append(("Baseline: same run, exit_prob 0", "skipped", f"({skipped}; columns left empty)"))
    elif args.baseline:
        rows += [
            ("Baseline: same run, exit_prob 0", "", ""),
            ("  Jammed cars", _pct(base["baseline_jam_fraction"]), "%"),
            ("  Stop-and-go", _fmt(base["baseline_stops_per_car_min"]), "stops per car per min"),
            ("  Number of jams", _fmt(base["baseline_n_jams"], 1), "mean count"),
            ("  Flux (whole ring)", _fmt(base["baseline_flux"], 3), "cars/s"),
            ("  Mean speed", _fmt(base["baseline_measured_mean_speed"]), "m/s"),
            ("Jamming due to entrances/exits", _pct(base["arm_jam_share"], 0),
             f"% = 1 - jammed(exit_prob 0) / jammed(this run); - = this run under {100 * JAM_PRESENT:g} % jammed"),
        ]
    print(f"\nSummary over the second half of the run ({late_from:g}-{args.duration:g} s):")
    width = max(len(r[0]) for r in rows)
    for name, value, unit in rows:
        print(f"  {name:<{width}}  {value:>10}  {unit}")
    print(f"\nwrote {out}_timeseries.csv, {out}_summary.csv, {out}_spacetime.npz and {out}.png")


IDLE_FIX = {"open mode only": "add --set traffic_mode=open, or vary traffic_mode too",
            "no entrances": "set n_arms above 0", "no trucks": "set truck_share above 0"}


def idle_notes(names, configs: list[SimConfig]) -> list[str]:
    """Notes on the varied settings that have no effect in any of the runs ``configs`` (e.g. an
    Entrance chance swept in closed mode, where every run is the same), one per reason."""
    reasons: dict[str, list[str]] = {}
    for name in names:
        why = {inactive_reason(name, cfg) for cfg in configs} if name in PARAM_BY_NAME else {""}
        if len(why) == 1 and (reason := why.pop()):
            reasons.setdefault(reason, []).append(name)
    return [f"{idle_note(group, reason)} ({IDLE_FIX.get(reason, 'change the base settings')})"
            for reason, group in reasons.items()]


def cmd_sweep(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from roundabout_sim.plots import BIN_ABOVE, heatmap_figure, headline_panels, sample_figure, sweep_figure

    setup = _base(args, parser)
    if args.seeds < 1 or args.warmup < 0 or args.measure <= 0 or args.interval <= 0:
        parser.error("need --seeds >= 1, --warmup >= 0, --measure > 0 and --interval > 0")
    given = _variables(args, parser)
    if args.plan is not None and (given or args.sample is not None):
        parser.error("--plan takes the varied settings from its columns: no --x, --series, --vary or --sample "
                     "with it")
    if args.sample_groups and args.sample is None:
        parser.error("--sample-groups needs --sample")
    if args.plan is not None:
        if Path(args.plan).resolve() == Path(f"{args.out}.csv").resolve():
            parser.error(f"--out {args.out} would write its CSV over the plan {args.plan}: choose another --out")
        design = _read_plan(args.plan, setup, args, parser)
    elif args.sample is not None:
        design = _sample(given, setup, args, parser)
    elif given:
        design = _grid(given, args, parser)
    else:
        parser.error("need --x (or --vary NAME=SPEC), --plan FILE or --sample N")
    for i, point in enumerate(design.points):  # every run's settings are valid before anything runs
        try:
            SimConfig().updated(**{**setup.changes, **point})
        except ValueError as err:
            parser.error(f"invalid settings at {design.location(i)}: {err}")
    if "init_spacing" in design.names:
        print(f"note: {INIT_SPACING_NOTE}")

    jobs = [({**setup.changes, **point}, seed + k, args.warmup, args.measure, args.interval)
            for point, seed in zip(design.points, design.seeds) for k in range(args.seeds)]
    configs = [SimConfig().updated(**job[0]) for job in jobs]
    _check_sampled(args.warmup, args.measure, args.interval, configs, parser, "the measured window", "--measure")
    workers = max(1, min(args.workers, len(jobs)))
    print(f"{len(jobs)} runs ({design.shape} x {args.seeds} seeds) on {workers} worker(s); {args.warmup:g} s "
          f"warm-up + {args.measure:g} s measured each")
    if setup.text():
        given_too = [f"{k}={_value_text(v)}" for k, v in setup.explicit.items() if k not in design.names]
        then = f", then {', '.join(given_too)}" if given_too else ""
        print(f"Base settings: {setup.text()}{then}")
    if any(_open(cfg) and cfg.demand_profile == "peak" for cfg in configs):
        _peak_note(args.warmup, args.warmup + args.measure)
    for note in idle_notes(design.names, configs):
        print(f"note: {note}")
    if design.kind == "plan" and (repeats := plan_repeats(design, jobs, configs, args.seeds)):
        shown = "; ".join(repeats[:5]) + (f"; and {len(repeats) - 5} more" if len(repeats) > 5 else "")
        print(f"note: plan {args.plan}: rows with the same settings and seed repeat the same run: {shown}. Give "
              "each row its own seed (a seed column), or use --seeds for repeats")

    results: list[dict] = [{}] * len(jobs)
    t_wall = time.time()

    def report(done: int, job: int, r: dict) -> None:
        elapsed = time.time() - t_wall
        eta = elapsed / done * (len(jobs) - done)
        demand = (f", {_fmt(r['cars'], 0)} cars + {_fmt(r['queued'], 0)} queued, entry delay "
                  f"{_fmt(r['mean_entry_delay'], 1)} s, blocked {_fmt(r['blocked_per_min'], 1)}/min"
                  if r["traffic_mode"] == "open" else "")
        print(f"[{done:3d}/{len(jobs)}] {_where(design, job // args.seeds, r)}seed={r['seed']}: "
              f"flux {_fmt(r['flux'], 3)} cars/s, speed {_fmt(r['measured_mean_speed'])} m/s, "
              f"jammed {_pct(r['jam_fraction'], 0)} % (from {_fmt(r['congestion_onset'], 0)} s), "
              f"stop-and-go {_fmt(r['stops_per_car_min'])}/car/min (from {_fmt(r['stop_go_onset'], 0)} s), "
              f"front car following {_pct(r['following_jam_share'], 0)} %{demand} "
              f"({elapsed:.0f} s, ~{eta:.0f} s left)", flush=True)

    if workers == 1:
        for i, job in enumerate(jobs):
            results[i] = _sweep_job(job)
            report(i + 1, i, results[i])
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_sweep_job, job): i for i, job in enumerate(jobs)}
            for done, future in enumerate(as_completed(futures), 1):
                i = futures[future]
                results[i] = future.result()
                report(done, i, results[i])

    # Rows in run order: the x varies fastest, then the series and so on, the seeds within each point.
    # Columns: the varied settings (the x just before the seed), the seed, the other settings, the measurements.
    keys = design.keys
    df = pd.DataFrame(results)
    first = keys + ["seed"] + [n for n in CONFIG_NAMES if n not in keys]
    df = df[first + [c for c in df.columns if c not in first]].reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = json.dumps(_meta(design, setup, args), indent=1, default=_json_value) + "\n"
    # The CSV first: a sweep stopped in between leaves the CSV without its meta file, which the sweep
    # viewer reads all the same (never a meta file without its CSV).
    state.write_atomic(f"{out}.csv", lambda fh: df.to_csv(fh, index=False, encoding="utf-8"))
    state.write_atomic(f"{out}_meta.json", lambda fh: fh.write(meta.encode("utf-8")))
    # A heatmap of an earlier sweep with this prefix would no longer match the CSV (a grid writes a new one).
    Path(f"{out}_heatmap.png").unlink(missing_ok=True)

    open_runs = bool((df["traffic_mode"] == "open").any())
    truck_runs = bool((df["truck_share"] > 0).any())
    panels = headline_panels(open_runs, STOP_GO_ONSET)
    subtitle = _held_text(df, design, setup)
    times = f"{args.warmup:g} s warm-up + {args.measure:g} s measured"
    written = [f"{out}.csv", f"{out}_meta.json", f"{out}.png"]
    if design.kind == "grid":
        x_name, series_name = design.names[0], (design.names[1] if len(design.names) > 1 else None)
        rest = design.varied[2:]
        over = " and ".join(f"{_axis_label(v['name']).split(' (')[0]} ({', '.join(map(_value_text, v['values']))})"
                            for v in rest)
        per_point = len(df) // max(1, len(df.groupby(design.names[:2], sort=False)))
        if not rest:
            band = "seeds"  # as the sweep has always put it
        elif per_point == 1:
            band = f"the single run at each point (at {over})"
        else:
            band = (f"the {per_point} runs at each point: averaged over {over}"
                    + (", every value pooled with the seeds" if args.seeds > 1 else ""))
        fig = sweep_figure(
            df, x_name, series_name, x_label=_axis_label(x_name), series_label=_value_text,
            title=f"Roundabout sweep: metrics vs {_axis_label(x_name).split(' (')[0].lower()}",
            subtitle=subtitle + f"\n{args.seeds} seed(s) per point, {times}; line = mean, band = min-max across "
                     + band,
            legend_title=_axis_label(series_name) if series_name else "", onset_threshold=STOP_GO_ONSET,
            congestion_threshold=CONGESTION_ONSET, open_demand=open_runs, value_text=_value_text)
        fig.savefig(f"{out}.png")
        if series_name:
            if per_point == 1:
                cell = "the single run at that pair of values" + (f" (at {over})" if rest else "")
            else:
                pooled = [f"over {over}"] * bool(rest) + ["the seeds"] * (args.seeds > 1)
                cell = (f"the mean of the {per_point} runs at that pair of values ("
                        + ("averaged " if rest else "") + " and ".join(pooled) + ")")
            heat = heatmap_figure(
                df, x_name, series_name, panels, x_label=_axis_label(x_name), y_label=_axis_label(series_name),
                value_text=_value_text,
                title=f"Roundabout sweep: {_axis_label(x_name).split(' (')[0].lower()} by "
                      f"{_axis_label(series_name).split(' (')[0].lower()}",
                subtitle=subtitle + f"\n{args.seeds} seed(s) per point, {times}; each cell = {cell}; light = low, "
                                    "dark = high, blank = no value")
            heat.savefig(f"{out}_heatmap.png")
            written.append(f"{out}_heatmap.png")
    else:
        shown, nothing = design.names, not design.names  # nothing varied: the runs against their number
        if len(shown) > SCATTER_SETTINGS:
            shown = list(rank_correlations(df, design.names, [p.column for p in panels]).index[:SCATTER_SETTINGS])
        if design.kind == "plan":
            how = f"{len(design.points)} plan rows"
        elif given:
            how = f"{len(design.points)} random points (Latin hypercube over the given ranges)"
        else:
            groups = _sample_groups(args)
            drawn = f"the {and_join(groups)} settings" if groups else "every setting"
            how = f"{len(design.points)} random points (Latin hypercube over the plausible ranges of {drawn})"
        kind = "random sample" if design.kind == "sample" else "plan"
        if nothing:
            fig = sample_figure(
                df.assign(run=np.arange(1, len(df) + 1)), ["run"], panels, means=False,
                param_label=lambda _: f"Run (in order: the {'rows' if design.kind == 'plan' else 'points'}, each "
                                      "with its seeds)",
                title=f"Roundabout {kind}: every run (nothing varied)",
                subtitle=subtitle + f"\n{how} x {args.seeds} seed(s), {times}. Nothing varies: every "
                                    f"{'row' if design.kind == 'plan' else 'point'} has the same settings, so each "
                                    "dot is one run, against its number")
        else:
            fig = sample_figure(
                df, shown, panels, param_label=_axis_label, value_text=_value_text,
                distinct_max=BIN_ABOVE[design.kind],
                title=f"Roundabout {kind}: metrics against each varied setting",
                subtitle=subtitle + f"\n{how} x {args.seeds} seed(s), {times}. Each dot = one run; line = mean of "
                                    "the runs in equal-width bins of the setting (per value when it takes few)"
                         + (". The other settings vary at the same time, so a trend is the setting's effect "
                            "averaged over them" if len(design.names) > 1 else "")
                         + (f"; showing the {len(shown)} of {len(design.names)} varied settings with the strongest "
                            "rank correlation with these metrics (all are in the CSV)"
                            if len(shown) < len(design.names) else ""))
        fig.savefig(f"{out}.png")

    if design.kind == "sample":
        _print_sample(df, design, [p.column for p in panels])
    else:
        columns = list(SUMMARY_COLUMNS) + list(OPEN_COLUMNS * open_runs) + list(TRUCK_COLUMNS * truck_runs)
        _print_means(df, keys, columns, open_runs, truck_runs)
        _print_arm_shares(df, keys)
    print(f"\nwrote {', '.join(written[:-1])} and {written[-1]} ({time.time() - t_wall:.0f} s)")


def _held_text(df: pd.DataFrame, design: Design, setup: Base) -> str:
    """The figures' first subtitle line: the base (preset or random) and the settings held, named."""
    open_runs = (df["traffic_mode"] == "open").any()
    demand = ("traffic_mode", "entry_rate", "demand_profile", "max_queue") if open_runs else ()
    vehicles = ("truck_share", "truck_length") if (df["truck_share"] > 0).any() else ()
    spread = tuple(n for n in ("reaction_std", "headway_std") if (df[n] > 0).any())
    road = ("road_condition",) if (df["road_condition"] != "dry").any() else ()
    held = [n for n in ("n_lanes", "n_arms", "density", "speed_std", "reaction_time", "reaction_distance",
                        "lane_change_rate", "driver_noise", *demand, *vehicles, *spread, *road)
            if n not in design.names]
    held += [n for n in _meta_fixed(design, setup) if n not in held]  # the others that differ from the defaults
    cfg = SimConfig().updated(**{**setup.changes, **(design.points[0] if design.points else {})})
    text = ", ".join(_label(n, getattr(cfg, n)) for n in held)
    return f"{setup.text()[:1].upper()}{setup.text()[1:]}: {text}" if setup.text() else text


def _where(design: Design, point: int, row: dict | None = None) -> str:
    """Where a run is in the design, for its progress line ('n_lanes=2 density=40 ', 'row 3: ...')."""
    values = row if row is not None else design.points[point]
    if design.kind == "grid":
        return "".join(f"{n}={values[n]} " for n in design.keys)
    shown = design.names[:PROGRESS_SETTINGS]
    more = f"+{len(design.names) - len(shown)} more " if len(design.names) > len(shown) else ""
    label = "row" if design.kind == "plan" else "point"
    return f"{label} {point + 1}: " + "".join(f"{n}={_compact(values[n])} " for n in shown) + more


def _compact(value) -> str:
    """A setting's value in a progress line: numbers with at most 4 significant digits."""
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def _meta_fixed(design: Design, setup: Base) -> dict:
    """The settings held in every run that differ from the defaults."""
    first = {**setup.changes, **(design.points[0] if design.points else {})}
    return {n: v for n, v in _differ(SimConfig().updated(**first)).items() if n not in design.names}


def _meta(design: Design, setup: Base, args: argparse.Namespace) -> dict:
    """PREFIX_meta.json: what the sweep viewer needs to know about the design (format version META_VERSION)."""
    return {"version": META_VERSION, "design": design.kind, "varied": design.varied,
            "fixed": _meta_fixed(design, setup), "preset": setup.preset, "random": setup.random,
            "sample_groups": _sample_groups(args) if design.kind == "sample" else None,
            "seeds": args.seeds, "base_seed": args.seed, "warmup": args.warmup, "measure": args.measure,
            "interval": args.interval, "command": _command(getattr(args, "argv", None))}


def _command(argv: list[str] | None) -> str:
    """The command line that ran, as one line to paste into this platform's shell."""
    line = cli.shell_line(list(sys.argv[1:] if argv is None else argv))
    return f"python -m roundabout_sim.experiments {line}".rstrip()


def _json_value(value):
    """numpy numbers as plain ones in JSON."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def _print_means(df: pd.DataFrame, keys: list[str], columns: list[str], open_runs: bool, truck_runs: bool) -> None:
    """The means across seeds of every point (grid or plan): one row per combination of the varied settings."""
    table = df.groupby(keys, sort=False)[columns].mean() if keys else df[columns].mean().to_frame("all runs").T
    table[["jam_fraction", "following_jam_share"]] *= 100
    # The letter from the means across seeds (entry_los per run is in the CSV).
    vc = table["entry_vc"] if "entry_vc" in table else pd.Series(np.nan, index=table.index)
    table.insert(table.columns.get_loc("mean_entry_delay") + 1, "LOS",
                 [level_of_service(d, v) for d, v in zip(table["mean_entry_delay"], vc)])
    table = table.rename(columns={"measured_density": "dens_meas", "measured_mean_speed": "speed",
                                  "jam_fraction": "jam_%", "congestion_onset": "congested_s",
                                  "stops_per_car_min": "stops/car/min", "stop_go_onset": "stop_go_s",
                                  "following_jam_share": "following_%", "spacing_gini": "gini",
                                  "order_param": "order", "emergency_per_min": "harsh/min",
                                  "wave_speed": "wave_m/s", "mean_entry_delay": "delay_s",
                                  "demand_per_min": "demand/min", "blocked_per_min": "blocked/min",
                                  "entry_vc": "v/c", "throughput_per_h": "veh/h", "truck_speed": "truck_m/s",
                                  "car_speed": "car_m/s"})
    print("\nMeans across seeds:")
    print(table.to_string(float_format=table_number, na_rep="-"))
    print(f"(jam_% = cars below {100 * SLOW_FRACTION:g} % of free speed, the congestion level; congested_s = first "
          f"time its {RATE_WINDOW:g} s mean > {100 * CONGESTION_ONSET:g} %;\n"
          f" stops/car/min = stop-and-go index (instability: ~0 when smooth and in a standstill); stop_go_s = "
          f"first time it > {STOP_GO_ONSET:g};\n onset times are means over the runs that got there; "
          f"following_% = jams whose front car is just following,\n an upper bound on natural jams - compare "
          f"--series exit_prob=0,0.1 or run --baseline; harsh/min = Harsh braking (episodes/min);\n"
          f" delay_s = queue wait of the cars that entered, with its HCM-style level of service (LOS)"
          + ("; cars, queued = on the ring and waiting; demand/min = arrivals asked for;\n blocked/min = arrivals "
             "turned away by a full queue; v/c = demand / capacity of the busiest entrance (LOS F above 1); "
             "veh/h = cars leaving per hour" if open_runs else "")
          + (";\n trucks = trucks on the ring; truck_m/s, car_m/s = mean speed of the trucks and of the other "
             "vehicles" if truck_runs else "") + ")")


def _print_sample(df: pd.DataFrame, design: Design, metrics: list[str]) -> None:
    """A random sample in brief: the spread of the headline metrics and their rank correlation with each
    varied setting (a table of every point would be as long as the CSV)."""
    print(f"\nHeadline metrics over the {len(df)} runs: mean (min - max)")
    width = max(len(m) for m in metrics)
    for m in metrics:
        values = df[m].astype(float)
        print(f"  {m:<{width}}  {significant(values.mean())}  ({significant(values.min())} - "
              f"{significant(values.max())})")
    if not design.names:
        return
    table = rank_correlations(df, design.names, metrics)
    shown = table.head(PRINTED_CORRELATIONS)
    more = f"; {len(table) - len(shown)} weaker ones left out" if len(table) > len(shown) else ""
    print(f"\nRank correlation (Spearman) of each varied setting with the metrics, strongest first{more}:")
    print(shown.to_string(float_format=lambda v: f"{v:+.2f}", na_rep="-"))
    print("(+1 / -1 = the metric rises / falls steadily with the setting, ~0 = no steady trend"
          + ("; the other settings vary at the same time" if len(design.names) > 1 else "")
          + ";\n choices are ranked in their listed order)")


def _arm_shares(df: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, int] | None:
    """:func:`arm_share_table`, which of its cells had a run, and how many points were left out.

    A point (a value of the other swept parameters) is kept when it has an
    exit_prob = 0 run to compare with and at least one run above 0; a plan
    need not have both, nor every exit_prob at every point.
    """
    if "exit_prob" not in keys or not (df["exit_prob"] == 0).any():
        return None
    jammed = df.groupby(keys, sort=False)["jam_fraction"].mean()
    other = [k for k in keys if k != "exit_prob"]
    wide = jammed.unstack("exit_prob") if other else jammed.to_frame().T
    runs = wide[[p for p in wide.columns if p > 0]]
    keep = wide[0.0].notna() & runs.notna().any(axis=1)
    runs = runs[keep].dropna(axis=1, how="all")
    if runs.empty:
        return None
    base = wide.loc[keep, 0.0]
    shares = {f"exit_prob={p:g}": [arm_share(run, b) for run, b in zip(runs[p], base)] for p in runs.columns}
    index = runs.index if other else ["all"]
    ran = pd.DataFrame(runs.notna().to_numpy(), index=index, columns=list(shares))
    return pd.DataFrame(shares, index=index), ran, int((~keep).sum())


def arm_share_table(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame | None:
    """Share of the jamming caused by entrances/exits, from a sweep that includes exit_prob = 0.

    Rows: the other swept parameter (if any), at the values with an exit_prob = 0
    run and one above 0; columns: each exit_prob > 0 that ran there. NaN where
    that run is (almost) unjammed, or (in a plan) did not run at that point.
    None if exit_prob is not swept, 0 is not among its values or nothing has a
    run at 0 to compare with.
    """
    found = _arm_shares(df, keys)
    return None if found is None else found[0]


def _print_arm_shares(df: pd.DataFrame, keys: list[str]) -> None:
    """The arm-share table for the closed-mode runs of a sweep (open-mode runs have no baseline)."""
    if "exit_prob" not in keys or not (df["exit_prob"] == 0).any():
        return
    is_open = df["traffic_mode"] == "open"
    if is_open.any():
        which = "in open mode" if is_open.all() else "for the open-mode runs"
        print(f"\n(no share of the jamming due to entrances/exits {which}: at exit_prob 0 the arrivals never "
              "leave, so that run is no baseline)")
    closed = df[~is_open]
    found = _arm_shares(closed, keys)
    if found is None:
        return
    if (closed["n_arms"] == 0).all():  # from the runs: --set, a plan column or a single --vary value alike
        print("\n(exit_prob does nothing on a closed ring: n_arms = 0)")
        return
    table, ran, left_out = found
    blank = not ran.to_numpy().all()
    print(f"\nJamming due to entrances/exits (%) = 1 - jammed(exit_prob 0) / jammed, from the means across "
          f"seeds; - = under {100 * JAM_PRESENT:g} % jammed" + ("; blank = no run there" if blank else "") + ":")
    cells = (100 * table).map(lambda v: f"{v:.0f}" if np.isfinite(v) else "-").where(ran, "")
    print(cells.to_string())
    if left_out:
        print(f"({plural(left_out, 'point')} left out: no exit_prob 0 run to compare with, or no run above 0)")


def cmd_view(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Open the sweep viewer (pygame) on a sweep CSV."""
    from roundabout_sim import sweep_viewer

    argv = [args.csv] if args.csv else []
    if args.results is not None:
        argv += ["--results", args.results]
    if args.size is not None:
        argv += ["--size", "x".join(map(str, args.size))]
    sweep_viewer.main(argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m roundabout_sim.experiments",
                                     description=f"{__doc__}\n{DEFINITIONS}",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    parser.add_argument("--list-presets", action=cli.ListPresetsAction, help=cli.LIST_PRESETS_HELP)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="one simulation -> time series, space-time data and a dashboard figure",
                         description="One simulation -> PREFIX_timeseries.csv, PREFIX_summary.csv, "
                                     "PREFIX_spacetime.npz and a dashboard PREFIX.png. The summary covers "
                                     "the second half of the run. With --baseline the run is repeated with "
                                     "exit_prob = 0 to measure what the entrances/exits cause. Settings: "
                                     "the defaults, then --preset or --random, then --set and the parameter "
                                     "flags.",
                         epilog=DEFINITIONS, formatter_class=argparse.RawDescriptionHelpFormatter,
                         allow_abbrev=False)
    run.add_argument("--seed", type=cli.seed_arg, default=0, help="random seed, a whole number >= 0 (default 0)")
    run.add_argument("--duration", type=float, default=600.0, help="simulated seconds (default 600)")
    run.add_argument("--interval", type=float, default=0.5, help="sampling interval in s (default 0.5)")
    run.add_argument("--out", default="results/run", help="output prefix (default results/run)")
    run.add_argument("--baseline", action="store_true",
                     help="also repeat the run with exit_prob = 0 (same seed and every other setting) and "
                          "report how much of the jamming the entrances/exits cause (baseline_* and "
                          "arm_jam_share in the summary; dashed line in the dashboard)")
    cli.add_base_flags(run, "--set and the parameter flags")
    _add_param_flags(run)
    run.set_defaults(handler=cmd_run, command_parser=run)

    sweep = sub.add_parser("sweep", help="parameter sweep in parallel (grid, plan or random sample) -> CSV, "
                                         "meta file and figures",
                           description="Parameter sweep -> PREFIX.csv (one row per run), PREFIX_meta.json (the "
                                       "design) and PREFIX.png (+ PREFIX_heatmap.png for 2+ grid settings). "
                                       "Each run warms up, then the metrics are averaged over the "
                                       "measurement window (onset times: from the start). Designs: a full "
                                       "grid of --x, --series and any number of --vary settings (in that "
                                       "order: the 1st is the x of the figure, the 2nd its series, the rest "
                                       "averaged over); --plan FILE.csv (one point per row); or --sample N "
                                       "random points. Every point runs --seeds times. Settings: the "
                                       "defaults, then --preset or --random, then --set and the parameter "
                                       "flags, then the varied values. Sweeping exit_prob with 0 among its "
                                       "values also prints how much of the jamming the entrances/exits cause.",
                           epilog=DEFINITIONS, formatter_class=argparse.RawDescriptionHelpFormatter,
                           allow_abbrev=False)
    sweep.add_argument("--x", type=_assignment, metavar="NAME=SPEC",
                       help="parameter on the x axis (the 1st varied); SPEC = start:stop:step (stop included) or "
                            "a comma list")
    sweep.add_argument("--series", type=_assignment, metavar="NAME=V1,V2,...",
                       help="parameter drawn as one line per value (the 2nd varied; up to 8 values)")
    sweep.add_argument("--vary", action="append", type=_assignment, default=[], metavar="NAME=SPEC",
                       help="one more varied parameter (repeatable; a full grid of all the values). Without --x "
                            "the first --vary is the x and the second the series. With --sample, SPEC = "
                            "start:stop is the range drawn from and a comma list the values dealt out")
    sweep.add_argument("--plan", metavar="FILE.csv",
                       help="one design point per row: columns named like the parameters, plus an optional "
                            "seed column (the row's base seed); an empty cell keeps the base setting")
    sweep.add_argument("--sample", type=int, metavar="N",
                       help="N random design points (seeded by --seed), a Latin hypercube: each --vary range is "
                            "cut into N strata, each sampled once on the slider step (a range holding no more "
                            "slider values than N, and a comma list, are dealt out evenly). Without --vary: over "
                            "the plausible ranges (presets.RANDOM_RANGES) of every parameter (or of "
                            "--sample-groups) not given with --set or its flag")
    sweep.add_argument("--sample-groups", dest="sample_groups", metavar="G1,G2",
                       help="--sample without --vary: draw only the parameters of these groups (the viewer's "
                            f"tabs: {', '.join(dict.fromkeys(p.group for p in PARAMS))})")
    sweep.add_argument("--seeds", type=int, default=3, help="repetitions per point (default 3)")
    sweep.add_argument("--seed", type=cli.seed_arg, default=0,
                       help="base seed, a whole number >= 0; run k of each point uses base + k")
    sweep.add_argument("--warmup", type=float, default=120.0, help="simulated warm-up in s (default 120)")
    sweep.add_argument("--measure", type=float, default=300.0, help="simulated measurement in s (default 300)")
    sweep.add_argument("--interval", type=float, default=0.5, help="sampling interval in s (default 0.5)")
    sweep.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                       help="parallel processes (default: CPU count - 1)")
    sweep.add_argument("--out", default="results/sweep", help="output prefix (default results/sweep)")
    cli.add_base_flags(sweep, "--set and the parameter flags")
    _add_param_flags(sweep)
    sweep.set_defaults(handler=cmd_sweep, command_parser=sweep)

    view = sub.add_parser("view", help="open the sweep viewer on a sweep CSV",
                          description="Open the sweep viewer (python -m roundabout_sim.sweep_viewer): the "
                                      "charts of a sweep CSV, and new sweeps run from the window.",
                          allow_abbrev=False)
    view.add_argument("csv", nargs="?", default=None,
                      help="sweep CSV to open (default: the most recent sweep CSV in the results folder)")
    view.add_argument("--results", default=None, metavar="DIR",
                      help="results folder: sweep CSVs listed, new sweeps and PNGs written there (default results)")
    view.add_argument("--size", type=cli.window_size, default=None, metavar="WxH",
                      help="window size in pixels (default: 90%% of the screen)")
    view.set_defaults(handler=cmd_view, command_parser=view)
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = build_parser().parse_args(argv)
    args.argv = argv  # for the command line in the sweep's meta file
    args.handler(args, args.command_parser)


if __name__ == "__main__":
    main()
