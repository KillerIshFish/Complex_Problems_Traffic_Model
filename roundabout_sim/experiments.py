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

    # entry capacity against conflicting flow (the capacity test; FHWA-OR-RD-22-11's design: 300 s + 7,200 s, 5
    # seeds), then the HCM form fitted to it
    python -m roundabout_sim.experiments sweep --x entry_rate=0:0.15:0.01 --preset us_single_lane \\
        --set subject_arm=on --seeds 5 --warmup 300 --measure 7200 --out results/cap
    python -m roundabout_sim.experiments capacity results/cap.csv

    # every metric of each run against the same seed's run at road_condition=dry, with 95 % CIs
    python -m roundabout_sim.experiments compare results/sweep.csv --ref road_condition=dry

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
  capacity
         SWEEP_capacity.csv     one row per series: the fit c = A exp(-B vc) with 95 % bootstrap
                                CIs, its ratio to the HCM base curve, tf and Raff tc, fA, fB
                                (fitted and from the headways), the gain at vc 0/400/800/1200
                                and d ln(gain)/d vc against the reference series
         SWEEP_capacity.png     capacity against conflicting flow (runs, fits, HCM curves) and
                                the gain over the reference series
  compare
         SWEEP_compare.csv      per point and metric: each seed's run against the same seed's
                                run at the reference value, mean difference and ratio, 95 % CI

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
import math
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
from roundabout_sim.engine import (LOG_HORIZON, MAX_DECEL, PEAK_PERIOD, SLOW_FRACTION, SUBJECT_ARM, SUBJECT_QUEUE,
                                   TWO_PI, RingSim, braking_limits, mean_demand_factor)
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
# experiments capacity. FHWA-OR-RD-22-11 Table 1.7 (= Table 7.4): the HCM 7 CAV factors (share, fA, fB) for a
# single-lane entry against one circulating lane and against two (0 % is 1.00 / 1.00).
CAV_FACTORS = {1: ((0.2, 1.05, 0.99), (0.4, 1.12, 0.97), (0.6, 1.22, 0.94), (0.8, 1.29, 0.90), (1.0, 1.35, 0.85)),
               2: ((0.2, 1.03, 0.99), (0.4, 1.08, 0.96), (0.6, 1.18, 0.92), (0.8, 1.28, 0.89), (1.0, 1.38, 0.85))}
# (gain at vc 0, d ln(gain) / d vc [per pc/h]) at 100 % CAVs, single-lane entry against one circulating lane.
# HCM 7: gain = fA exp((1 - fB) B vc), so (1 - 0.85) x 1.02e-3 = +1.53e-4 (derived); FHWA made those factors
# from Raff tc and tf by Siegloch (FHWA-OR-RD-22-11 Eqs. 7-7 to 7-12: capacity_table's *_hw columns). Its own
# simulated capacities (Sec. 7.3.1, p. 114: +36 % at vc 0 rising to +116 % at 1,200 pc/h) rise more steeply:
# ln(2.16 / 1.36) / 1200 = +3.9e-4 (derived), the like-for-like check of the fitted gains. Jiang et al. 2022
# (J. Transp. Eng. A 148(5) 04022014), 1.40 at vc 0 falling to 1.26 at 1,200 veh/h: ln(1.26 / 1.40) / 1200 =
# -8.8e-5 (derived); *assumption*: a secondary value, the paper not read, unverified and at odds with Sec. 7.3.1.
FHWA_SIM_GAIN = (1.36, math.log(2.16 / 1.36) / 1200.0)
JIANG_GAIN = (1.40, math.log(1.26 / 1.40) / 1200.0)
GAIN_VC = (0.0, 400.0, 800.0, 1200.0)  # [veh/h] conflicting flows the gains and HCM ratios are reported at
V1_TOLERANCE = 0.10  # validated: the reference series' capacity within +-10 % of the HCM base (*assumption*)
BOOTSTRAP = 2000  # resamples of the seeds for the capacity CIs (percentile intervals; seeded, reproducible)
# Percentile bootstrap intervals over fewer seeds than this are too narrow: a 95 % interval of a mean covers
# 0.76 of the time with 3 seeds, 0.84 with 5 and 0.90 with 10 (derived: Monte Carlo, normal data, 2,000 trials).
FEW_SEEDS = 10
# The capacity columns experiments capacity can fit, each with its conflicting flow.
CAP_COLUMNS = {"subject_cap_h": "subject_vc_h", "subject_cap15_h": "subject_vc_h",
               "subject_cap_pch": "subject_vc_pch"}
CAP15 = 900.0  # [s] subject_cap15_h: 4 x the most entries in one 15-minute interval (FHWA-OR-RD-22-11 p. 111)
# experiments compare: Student t 0.975 quantiles for 1-30 degrees of freedom (any t table); beyond, the
# Cornish-Fisher series of Abramowitz & Stegun 26.7.5 (error < 1e-4 there).
T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
        2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)
COMPARE_METRICS = ("flux", "measured_mean_speed", "jam_fraction", "stops_per_car_min", "speed_sd_pooled",
                   "emergency_per_min", "mean_entry_delay", "throughput_per_h", "subject_cap_h")  # the default

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
  automated            automated vehicles on the ring (mean): those of the
                       auto_kind (acc, cav or robotaxi), auto_share of the new
                       vehicles; human_speed / auto_speed: mean speed of the
                       human-driven and of the automated vehicles (empty if
                       there are none). A truck counts by its kind
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
  throughput_per_h     cars leaving the ring per hour
  speed_sd_pooled      spread of the speeds over the cars and the window [m/s]:
                       sqrt(mean of measured_speed_std^2 + variance of
                       measured_mean_speed) over the samples, Stern et al.
                       2018's speed SD (3.31 m/s in waves, 0.64 with their
                       wave-smoothing car). Higher = more stop-and-go
  lane0_speed_sd       the same on each lane (0 = innermost; lane1_speed_sd,
                       lane2_speed_sd, lane3_speed_sd, lane4_speed_sd), empty
                       for a lane the run does not have
  subject_cap_h        CAPACITY TEST (subject_arm on, open mode; empty
                       otherwise): entries per hour from the bottom entrance,
                       whose queue is kept full: its capacity [veh/h], against
                       its conflicting flow subject_vc_h. experiments capacity
                       fits c = A exp(-B vc) to these (HCM form). In these runs
                       queued, longest_queue, peak_queue, mean_entry_delay and
                       entry_los leave the test entrance's standing queue out
  subject_cap15_h      the capacity as FHWA-OR-RD-22-11 p. 111 measures it: 4 x
                       the most entries in one of the consecutive 15-minute
                       intervals of the window [veh/h] (empty for a window
                       under 15 min); above subject_cap_h, the more so the
                       lower the capacity
  subject_cap_pch      the same as subject_cap_h in passenger cars per hour (a
                       truck counts as 2: the HCM 6 E_T)
  subject_vc_h         conflicting flow at the test entrance [veh/h]: the
                       circulating vehicles passing its entry point, every
                       lane. Every other car passes it once, so the Entrance
                       chance of the other entrances sets it
  subject_vc_pch       the same in passenger cars per hour (a truck counts as
                       2), the flow to set against subject_cap_pch
  subject_vc_outer_h   the same, outer lane only (the lane entering cars merge
                       into; = subject_vc_h on a one-lane ring)
  subject_queue_empty_share  share of the window the test entrance had no car
                       waiting: must be 0 (else it was not saturated)
  subject_tf           follow-up headway [s]: mean time between consecutive
                       entries from the test entrance with no outer-lane car
                       passing in between (FHWA-OR-RD-22-11's measure; the
                       model gives follow_up + one time step)
  subject_tc_raff      critical gap [s] by Raff's method on the gaps between
                       outer-lane cars passing the test entry point (accepted:
                       a car entered in it; rejected: every other, as the
                       queue is never empty): the gap length where the share
                       of the accepted gaps shorter than it equals the share
                       of the rejected gaps longer than it, the crossing of
                       their cumulative distributions (FHWA-OR-RD-22-11 p.
                       117, its Table 7.1 estimator). Not the model's
                       entry_gap, which is a lag to the car behind, nor HCM's
                       Siegloch tc
  subject_tc_raff_count  the same with counts for shares: as many accepted gaps
                       shorter as rejected gaps longer (Raff 1950, as
                       FHWA-OR-RD-22-11 p. 110 words it). Short rejected gaps
                       (a platoon's) lower subject_tc_raff, not this"""

# Recorder columns averaged over a measurement window.
MEAN_COLUMNS = ("cars", "queued", "longest_queue", "density", "mean_speed", "mean_free_speed", "speed_std",
                "flux", "jam_fraction", "speed_deficit", "n_jams", *JAM_ORIGINS, "occupancy", "spacing_gini",
                "clustering", "order_param", "order_baseline", "mean_trip_time", "demand_per_min",
                "trucks", "truck_speed", "car_speed", "automated", "human_speed", "auto_speed")
# Columns of the sweep's printed table (means across seeds); OPEN_COLUMNS are added when a run is in open
# mode, TRUCK_COLUMNS when a run has trucks.
SUMMARY_COLUMNS = ("measured_density", "measured_mean_speed", "flux", "jam_fraction", "congestion_onset",
                   "stops_per_car_min", "stop_go_onset", "n_jams", "following_jam_share", "spacing_gini",
                   "order_param", "emergency_per_min", "wave_speed", "mean_entry_delay")
OPEN_COLUMNS = ("cars", "queued", "demand_per_min", "blocked_per_min", "entry_vc", "throughput_per_h")
TRUCK_COLUMNS = ("trucks", "truck_speed", "car_speed")
SUBJECT_PRINTED = ("subject_vc_h", "subject_cap_h", "subject_tf", "subject_tc_raff")  # printed with capacity tests
# The capacity test's columns, in every summary (NaN without the test), so sweep CSVs stay rectangular.
SUBJECT_COLUMNS = ("subject_cap_h", "subject_cap15_h", "subject_cap_pch", "subject_vc_h", "subject_vc_pch",
                   "subject_vc_outer_h", "subject_queue_empty_share", "subject_tf", "subject_tc_raff",
                   "subject_tc_raff_count")
LANE_SD_LANES = 5  # lane{k}_speed_sd columns for k below this (the slider's most lanes), NaN for lanes a run lacks


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


def pooled_sd(std, mean) -> float:
    """sqrt(mean(std^2) + var(mean)) over the samples that have both (``std``, ``mean``: each sample's spread
    and mean of the speeds): the spread over the vehicles and the time, as Stern et al. 2018 report it (the law
    of total variance with every sample weighted alike). NaN if no sample has both."""
    s, m = np.asarray(std, dtype=float), np.asarray(mean, dtype=float)
    ok = np.isfinite(s) & np.isfinite(m)
    return float(np.sqrt(np.mean(s[ok] ** 2) + np.var(m[ok]))) if ok.any() else float("nan")


def gap_acceptance(entries, passes) -> tuple[np.ndarray, np.ndarray]:
    """(accepted, rejected) gaps [s] at a saturated entrance: the headways between consecutive circulating
    vehicles passing its entry point (``passes``: times, in order), accepted when at least one queued car
    entered in it (``entries``: times; a gap runs from one passing up to the next, and an entry in the same
    step as a passing comes after it, as the engine moves cars before it lets any in). With a queue that is
    never empty every gap was offered: FHWA-OR-RD-22-11 p. 110 counts as rejected "any headway between
    circulating vehicles not taken by an entering vehicle waiting at the approach"."""
    p, e = np.asarray(passes, dtype=float), np.asarray(entries, dtype=float)
    if p.size < 2:
        return np.zeros(0), np.zeros(0)
    gaps = np.diff(p)
    k = np.searchsorted(p, e, side="right") - 1  # the gap each entry falls in (-1: before the first passing)
    taken = np.zeros(gaps.size, dtype=bool)
    taken[k[(k >= 0) & (k < gaps.size)]] = True
    return gaps[taken], gaps[~taken]


def raff_critical_gap(accepted, rejected, shares: bool = False) -> float:
    """Raff's (1950) critical gap [s]: the gap length t at which the number of accepted gaps shorter than t
    equals the number of rejected gaps longer than t, as FHWA-OR-RD-22-11 p. 110 words it; with ``shares``,
    the share of the accepted gaps shorter than t equals the share of the rejected gaps longer than t: the
    crossing of the cumulative distribution of the accepted gaps and the reversed one of the rejected gaps,
    as FHWA computed Table 7.1 (p. 117; :func:`raff_cdf_critical_gap`).

    Exactly: D(t) = #{accepted < t} - #{rejected > t} (shares: each count
    over its total) never falls as t grows, and changes only at the gap
    lengths; the estimate is the midpoint of the interval of t where D = 0,
    from inf{t: D(t) >= 0} to inf{t: D(t) > 0} (where D jumps past 0 at one
    length, that length). So drivers who accept every gap above g and none
    below give the midpoint of the longest rejected and the shortest accepted
    gap either way. The two differ where the gaps are lopsided: more rejected
    gaps shorter than t (a platoon's) lower the shares' estimate and leave the
    counts' alone (FHWA's explanation of the CAV drop in Table 7.1, p. 117).
    NaN without an accepted or a rejected gap.
    """
    a, r = np.sort(np.asarray(accepted, dtype=float)), np.sort(np.asarray(rejected, dtype=float))
    if a.size == 0 or r.size == 0:
        return float("nan")
    u = np.unique(np.concatenate((a, r)))
    # D just above each length (on the open interval to the next one): below all of them it is < 0, above all
    # of them > 0, and at a length itself it is no more than just above it. (Shares compared as whole numbers:
    # #a/na - #r/nr has the sign of #a nr - #r na.)
    shorter, longer = np.searchsorted(a, u, side="right"), r.size - np.searchsorted(r, u, side="right")
    above = shorter * r.size - longer * a.size if shares else shorter - longer
    return float(0.5 * (u[np.argmax(above >= 0)] + u[np.argmax(above > 0)]))


def raff_cdf_critical_gap(accepted, rejected) -> float:
    """Raff's critical gap [s] from the cumulative distributions (FHWA-OR-RD-22-11 p. 117, Table 7.1): where
    the share of the accepted gaps shorter than t equals the share of the rejected gaps longer than t
    (:func:`raff_critical_gap` with shares)."""
    return raff_critical_gap(accepted, rejected, shares=True)


def follow_up_headway(entries, passes) -> float:
    """Mean headway [s] of consecutive entries from a saturated entrance with no circulating vehicle passing
    its entry point between them (``entries``, ``passes``: times, in order): the follow-up headway as
    FHWA-OR-RD-22-11 p. 110 measures it ("consecutive queued vehicles entering ... within the same gap").
    NaN with no such pair."""
    e, p = np.asarray(entries, dtype=float), np.asarray(passes, dtype=float)
    if e.size < 2:
        return float("nan")
    gap = np.searchsorted(p, e, side="right")  # passings up to each entry (in its step: before it)
    same = gap[1:] == gap[:-1]
    return float(np.diff(e)[same].mean()) if same.any() else float("nan")


class SubjectTap:
    """The capacity test's events from now on: the subject entrance's entry times and the times outer-lane
    vehicles passed its entry point, copied from the engine's logs (which keep only LOG_HORIZON s) as they
    come, so the gap statistics of :func:`summarise` cover a measured window of any length. :func:`advance`
    polls it after every step; read-only on the simulation."""

    def __init__(self, sim: RingSim):
        self.sim = sim
        self.entries: list[float] = []
        self.passes: list[float] = []
        self._upto = sim.time  # events logged at this time or later are new
        self.start = sim.time  # it holds every event from this time on

    @classmethod
    def from_logs(cls, sim: RingSim, since: float) -> SubjectTap:
        """The events the engine's logs still hold from ``since`` on (at most the last LOG_HORIZON s)."""
        tap = cls(sim)
        tap._upto, tap.start = since, max(since, sim.time - LOG_HORIZON)
        tap.poll()
        return tap

    def poll(self) -> None:
        sim, upto = self.sim, self._upto
        self.entries += [t for t, arm, _ in _logged_since(sim.entry_log, upto) if arm == SUBJECT_ARM]
        self.passes += [t for t, arm, lane, *_ in _logged_since(sim.conflict_log, upto)
                        if arm == SUBJECT_ARM and lane == sim.outer]
        self._upto = sim.time  # (every event of the steps so far was logged at an earlier time)


def _logged_since(log, since: float) -> list:
    """Entries of a time-ordered engine log at ``since`` or later, oldest first (read-only)."""
    recent = []
    for entry in reversed(log):
        if entry[0] < since:
            break
        recent.append(entry)
    return recent[::-1]


def peak_quarter(entries, t0: float, t1: float, start: float) -> float:
    """[veh/h] 4 x the most ``entries`` (times) in one of the consecutive whole CAP15-s intervals from ``t0`` up
    to ``t1``: FHWA-OR-RD-22-11 p. 111's capacity (the busiest of eight 15-minute counts in its 2 h, x 4). NaN
    without a whole interval, or if the entries are known only from ``start`` on, after ``t0``."""
    k = int(math.floor((t1 - t0) / CAP15 + 1e-6))  # (the clock is a sum of steps: 899.99999 s is 15 min)
    if k < 1 or start > t0 + 1e-6:
        return float("nan")
    e = np.asarray(entries, dtype=float)
    which = np.floor((e[(e >= t0) & (e < t0 + k * CAP15)] - t0) / CAP15).astype(int)
    return 3600.0 / CAP15 * float(np.bincount(np.minimum(which, k - 1), minlength=k).max())


def subject_columns(sim: RingSim, now: dict, start: dict, t0: float, tap: SubjectTap | None) -> dict:
    """The capacity test's summary columns (SUBJECT_COLUMNS) from ``t0`` to now (:func:`_counters` then,
    ``start``, and ``now``), with the gap statistics and the 15-minute counts from ``tap``; all NaN without
    the test."""
    if not sim.subject:
        return dict.fromkeys(SUBJECT_COLUMNS, float("nan"))
    from roundabout_sim.presets import TRUCK_PCE

    def count(key: str) -> float:
        return float(now[key][SUBJECT_ARM] - start[key][SUBJECT_ARM])

    span = max(sim.time - t0, 1e-9)
    per_h = 3600.0 / span
    entries, passes = (tap.entries, tap.passes) if tap is not None else ([], [])
    accepted, rejected = gap_acceptance(entries, passes)
    return {
        "subject_cap_h": per_h * count("arm_entered"),
        "subject_cap15_h": peak_quarter(entries, t0, sim.time, tap.start) if tap is not None else float("nan"),
        "subject_cap_pch": per_h * (count("arm_entered") + (TRUCK_PCE - 1.0) * count("arm_entered_trucks")),
        "subject_vc_h": per_h * count("arm_passed"),
        "subject_vc_pch": per_h * (count("arm_passed") + (TRUCK_PCE - 1.0) * count("arm_passed_trucks")),
        "subject_vc_outer_h": per_h * count("arm_passed_outer"),
        # (the time with a queue is summed per step like the clock: 1 - busy / span is float noise when full)
        "subject_queue_empty_share": round(max(0.0, 1.0 - count("queue_busy") / span), 9),
        "subject_tf": follow_up_headway(entries, passes),
        "subject_tc_raff": raff_cdf_critical_gap(accepted, rejected),
        "subject_tc_raff_count": raff_critical_gap(accepted, rejected),
    }


def _counters(sim: RingSim) -> dict[str, float | np.ndarray]:
    return {"exits": sim.n_exited, "missed": sim.n_missed, "lane_changes": sim.n_lane_changes,
            "emergency": sim.n_emergency, "stops": sim.n_stops, "arrivals": sim.n_arrived,
            "entries": sim.n_entered, "blocked": sim.n_blocked, "entry_wait": sim.entry_wait_total,
            # per entrance, for entry_vc and the capacity test
            "arm_arrived": sim.arm_arrived.copy(), "arm_entered": sim.arm_entered.copy(),
            "queue_busy": sim.queue_busy.copy(), "arm_passed": sim.arm_passed.copy(),
            "arm_passed_outer": sim.arm_passed_outer.copy(), "arm_passed_trucks": sim.arm_passed_trucks.copy(),
            "arm_entered_trucks": sim.arm_entered_trucks.copy(), "arm_entry_wait": sim.arm_entry_wait.copy()}


def measured_names(names) -> dict[str, str]:
    """Renames that give measurements sharing a parameter's name a ``measured_`` prefix."""
    return {k: f"measured_{k}" for k in names if k in CONFIG_NAMES}


def settings(sim: RingSim) -> dict:
    """Every parameter of ``sim`` (named like the CLI ``--set`` names) and its seed."""
    return {**{name: getattr(sim.cfg, name) for name in CONFIG_NAMES}, "seed": sim.seed}


def summarise(sim: RingSim, rec: Recorder, t0: float, start: dict[str, int], tap: SubjectTap | None = None) -> dict:
    """Metrics from ``t0`` to now (``start`` = :func:`_counters` at ``t0``).

    Sampled metrics are averaged over the window; event rates come from the
    engine's counters, so they count exactly the events inside the window
    (``mean_entry_delay`` too: the queue wait of exactly the cars that entered
    in it, those of the capacity test's entrance left out; and ``entry_vc``;
    and the capacity test's flows). ``stop_go_onset``
    and ``congestion_onset`` are measured from the start of the run.
    ``entry_los`` is the one text column. Metrics that share a parameter's
    name get a ``measured_`` prefix. The capacity test's gap statistics come
    from ``tap`` (a :class:`SubjectTap` started at ``t0``), else from what the
    engine's logs still hold (the last LOG_HORIZON s of the window).
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
    wait = now["entry_wait"] - start["entry_wait"]
    if sim.subject:  # the capacity test's standing queue is an instrument: its entrants' wait is left out
        others = np.arange(sim.n_arms) != SUBJECT_ARM
        entered = int((now["arm_entered"] - start["arm_entered"])[others].sum())
        wait = float((now["arm_entry_wait"] - start["arm_entry_wait"])[others].sum())
    delay = wait / entered if entered > 0 else float("nan")
    vc = entry_vc(*(now[k] - start[k] for k in ("arm_arrived", "arm_entered", "queue_busy")), span) \
        if sim.open_demand else float("nan")  # closed mode has no demand of its own
    row = {
        **{k: mean[k] for k in ("cars", "queued", "density", "mean_speed", "mean_free_speed", "speed_std",
                                "flux", "trucks", "truck_speed", "car_speed", "automated", "human_speed",
                                "auto_speed")},
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
        "speed_sd_pooled": pooled_sd(window["speed_std"], window["mean_speed"]),
        **{f"lane{k}_speed_sd": pooled_sd(window[f"lane{k}_speed_std"], window[f"lane{k}_speed"])
           if f"lane{k}_speed_std" in window and f"lane{k}_speed" in window else float("nan")
           for k in range(LANE_SD_LANES)},
        **subject_columns(sim, now, start, t0,
                          tap if tap is not None or not sim.subject else SubjectTap.from_logs(sim, t0)),
    }
    # Measurements named like a parameter (density, mean_speed, ...) must not overwrite its setting.
    renames = measured_names(row)
    return {renames.get(k, k): v for k, v in row.items()}


# --------------------------------------------------------------- simulation
def advance(sim: RingSim, rec: Recorder, seconds: float, on_sample: Callable[[], None] | None = None,
            tap: SubjectTap | None = None) -> None:
    """Step ``sim`` for ``seconds``, recording; ``on_sample`` runs after each new sample, and ``tap`` (the
    capacity test's events) is polled after every step."""
    for _ in range(int(round(seconds / sim.cfg.dt))):
        sim.step()
        if tap is not None:
            tap.poll()
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
    tap = SubjectTap(sim) if sim.subject else None
    advance(sim, rec, measure, tap=tap)
    return {**settings(sim), **summarise(sim, rec, warmup, start, tap)}


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
    None for the default groups (presets.DEFAULT_GROUPS: every group but AVs and Coop); ValueError for an
    unknown one."""
    from roundabout_sim import presets

    if not args.sample_groups:
        return None
    return presets.group_names([g.strip() for g in args.sample_groups.split(",") if g.strip()])


def _sample(given: list[tuple[str, str]], base: Base, args: argparse.Namespace,
            parser: argparse.ArgumentParser) -> Design:
    """``--sample N`` points: a Latin hypercube over the --vary ranges, else over the plausible ranges
    (presets.RANDOM_RANGES) of every parameter (but the AVs and Coop tabs', which only --sample-groups
    draws: presets.OPT_IN_GROUPS), or of the --sample-groups ones, not set explicitly.

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
    if cfg.auto_share > 0:
        brands = f" of {cfg.companies} companies" if cfg.companies > 1 else ""
        gap = f" (time gap x {cfg.robotaxi_headway:g})" if cfg.auto_kind == "robotaxi" else ""
        trucks += f", {100 * cfg.auto_share:g} % automated: {cfg.auto_kind}{gap}{brands}"
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


def _g(value) -> str:
    """Three significant figures, "-" for a missing value."""
    return "-" if value is None or not np.isfinite(value) else f"{value:.3g}"


# ---------------------------------------------------------------- commands
def _simulate(cfg: SimConfig, duration: float, interval: float, late_from: float,
              keep_speeds: bool = False) -> tuple[RingSim, Recorder, dict, list[np.ndarray]]:
    """One run of ``duration`` s, summarised from ``late_from``; optionally every car's speed after it."""
    sim = RingSim(cfg)
    rec = Recorder(sim, interval)
    advance(sim, rec, late_from)
    start = _counters(sim)
    tap = SubjectTap(sim) if sim.subject else None
    car_speeds: list[np.ndarray] = []
    advance(sim, rec, duration - late_from,
            (lambda: car_speeds.append(sim.v[sim.alive].copy())) if keep_speeds else None, tap)
    return sim, rec, summarise(sim, rec, late_from, start, tap), car_speeds


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
        *([("Automated on the ring", _fmt(summary["automated"], 1),
            f"mean ({_pct(summary['automated'] / summary['cars'] if summary['cars'] else float('nan'), 0)} % of the "
            f"vehicles; {100 * cfg.auto_share:g} % of the new ones are {cfg.auto_kind})"),
           ("Mean speed, human / automated", f"{_fmt(summary['human_speed'])} / {_fmt(summary['auto_speed'])}",
            "m/s")] if cfg.auto_share > 0 else []),
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
        *([("Capacity test entrance", _fmt(summary["subject_cap_h"], 0),
            f"veh/h ({_fmt(summary['subject_cap_pch'], 0)} pc/h) against {_fmt(summary['subject_vc_h'], 0)} veh/h "
            f"circulating; follow-up {_fmt(summary['subject_tf'])} s, Raff critical gap "
            f"{_fmt(summary['subject_tc_raff'])} s")] if sim.subject else []),
        ("Speed spread (pooled)", _fmt(summary["speed_sd_pooled"]), "m/s over the cars and the time (Stern 2018's SD)"),
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
            "no entrances": "set n_arms above 0", "no trucks": "set truck_share above 0",
            "dry road": "set road_condition to wet, snow or ice, or vary it too",
            "capacity test": "set subject_arm=off: the capacity test fixes every route",
            "no automated vehicles": "set auto_share above 0",
            "no robotaxis": "set auto_kind=robotaxi and auto_share above 0, or vary them too"}


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
        if np.isfinite(r["subject_cap_h"]):
            demand += f", test entrance {_fmt(r['subject_cap_h'], 0)} veh/h at vc {_fmt(r['subject_vc_h'], 0)}"
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
    subject_runs = bool(np.isfinite(df["subject_cap_h"].astype(float)).any())
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
            drawn = f"the {and_join(groups)} settings" if groups else "every setting but the AVs and Coop ones"
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
        columns = list(SUMMARY_COLUMNS) + list(OPEN_COLUMNS * open_runs) + list(TRUCK_COLUMNS * truck_runs) \
            + list(SUBJECT_PRINTED * subject_runs)
        _print_means(df, keys, columns, open_runs, truck_runs, subject_runs)
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


def _print_means(df: pd.DataFrame, keys: list[str], columns: list[str], open_runs: bool, truck_runs: bool,
                 subject_runs: bool = False) -> None:
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
                                  "car_speed": "car_m/s", "subject_vc_h": "test_vc", "subject_cap_h": "test_cap",
                                  "subject_tf": "test_tf_s", "subject_tc_raff": "test_tc_s"})
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
             "vehicles" if truck_runs else "")
          + (f";\n test_vc, test_cap = conflicting flow and capacity of the capacity test's entrance (veh/h; queue "
             f"kept at {SUBJECT_QUEUE}); test_tf_s, test_tc_s = its follow-up headway and Raff critical gap (s)"
             if subject_runs else "") + ")")


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


# ------------------------------------------------------- capacity and compare
def fit_capacity(vc, cap, weights=None) -> tuple:
    """(A, B) of the HCM form c = A exp(-B vc): least squares of ln c on ``vc``, each point weighted by
    ``weights`` (default 1; an array (..., n) gives one fit per row, e.g. bootstrap counts). Points need
    c > 0. NaN where the points have fewer than two distinct conflicting flows."""
    x, y = np.asarray(vc, dtype=float), np.log(np.asarray(cap, dtype=float))
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    total = w.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        xm, ym = (w * x).sum(axis=-1) / total, (w * y).sum(axis=-1) / total
        dx = x - np.expand_dims(xm, -1)
        sxx = (w * dx * dx).sum(axis=-1)
        slope = np.where(sxx > 1e-9 * np.maximum(total, 1.0), (w * dx * (y - np.expand_dims(ym, -1))).sum(axis=-1)
                         / np.where(sxx > 0, sxx, 1.0), np.nan)
    return np.exp(ym - slope * xm), -slope


def _interval(samples) -> tuple[float, float]:
    """The 95 % percentile interval of bootstrap ``samples`` (NaN replicates left out; NaN, NaN if all are)."""
    samples = np.asarray(samples, dtype=float)
    ok = samples[np.isfinite(samples)]
    return (float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))) if ok.size else (np.nan, np.nan)


def t975(df: int) -> float:
    """The Student t 0.975 quantile for ``df`` >= 1 degrees of freedom (T975, then A&S 26.7.5)."""
    if df <= len(T975):
        return T975[df - 1]
    z = 1.959963985
    return (z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
            + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3)
            + (79 * z**9 + 776 * z**7 + 1482 * z**5 - 1920 * z**3 - 945 * z) / (92160 * df**4))


def mean_ci(values) -> tuple[float, float, float]:
    """(mean, low, high): the mean of the finite ``values`` and its 95 % t interval (NaN bounds for one value,
    all NaN for none)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(v.mean())
    if v.size == 1:
        return mean, np.nan, np.nan
    half = t975(v.size - 1) * float(v.std(ddof=1)) / math.sqrt(v.size)
    return mean, mean - half, mean + half


def _design_columns(df: pd.DataFrame, leave_out: tuple[str, ...] = ()) -> list[str]:
    """The settings that differ between the runs of a sweep CSV (the design's varied ones), but ``leave_out``."""
    return [c for c in CONFIG_NAMES if c in df and c not in leave_out and df[c].nunique(dropna=False) > 1]


def _matches(column: pd.Series, value) -> pd.Series:
    """Rows of a CSV column equal to a setting's ``value`` (numbers within rounding, as ``_key``)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and pd.api.types.is_numeric_dtype(column):
        return pd.Series(np.isclose(column.to_numpy(float), float(value), rtol=1e-9, atol=1e-12), index=column.index)
    return column.astype(str) == str(value)


def _series_values(column: pd.Series, name: str) -> list:
    """A --by column's distinct values: numbers ascending, a choice in its listed order, else as they appear."""
    values = column.dropna().unique().tolist()
    if pd.api.types.is_numeric_dtype(column):
        return sorted(values)
    spec = PARAM_BY_NAME.get(name)
    order = list(spec.choices) if spec and spec.choices else []
    return sorted(values, key=lambda v: (order.index(v) if v in order else len(order), values.index(v)))


def validated_range(vc, cap, groups, hcm: tuple[float, float]) -> tuple[float, float] | None:
    """The conflicting flows where the reference series follows the HCM base ``hcm`` = (A, B): the longest run
    of consecutive design points (runs grouped by ``groups``, one label per run; in order of their mean
    conflicting flow) whose mean capacity is within V1_TOLERANCE of the base at their mean conflicting flow, as
    (lowest, highest) conflicting flow of those points' runs (so a fit over it takes them whole); None if no
    point is within it."""
    frame = pd.DataFrame({"vc": np.asarray(vc, float), "cap": np.asarray(cap, float), "g": groups})
    points = frame.groupby("g", sort=False).agg(vc=("vc", "mean"), cap=("cap", "mean"), lo=("vc", "min"),
                                                 hi=("vc", "max")).sort_values("vc")
    ok = (np.abs(points["cap"] / (hcm[0] * np.exp(-hcm[1] * points["vc"])) - 1.0) <= V1_TOLERANCE).to_numpy()
    best, run = None, None
    for i, good in enumerate(ok):
        run = (run if run is not None else i) if good else None
        if run is not None and (best is None or i - run > best[1] - best[0]):
            best = (run, i)
    if best is None:
        return None
    return float(points["lo"].iloc[best[0]:best[1] + 1].min()), float(points["hi"].iloc[best[0]:best[1] + 1].max())


def _weighted_mean(values, weights) -> np.ndarray:
    """Mean of the finite ``values`` (n,) with ``weights`` (..., n): one per row; NaN where no weight is left."""
    v = np.asarray(values, dtype=float)
    w = np.where(np.isfinite(v), np.asarray(weights, dtype=float), 0.0)
    total = w.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (w * np.where(np.isfinite(v), v, 0.0)).sum(axis=-1) / total, np.nan)


def _column(part: pd.DataFrame, name: str) -> np.ndarray:
    """A numeric column as floats, all NaN if the CSV lacks it (a sweep of an older version)."""
    return part[name].to_numpy(float) if name in part else np.full(len(part), np.nan)


def capacity_analysis(df: pd.DataFrame, by: str | None = None, ref=None, fit="validated", boot: int = BOOTSTRAP,
                      seed: int = 0, validated: tuple[float, float] | None = None, cap: str | None = None) -> dict:
    """The capacity test's fits from a sweep CSV, per series (the values of ``by``; one series without it).

    ``cap`` is the capacity column, fitted against its conflicting flow
    (CAP_COLUMNS); default subject_cap_pch against subject_vc_pch when a run
    has trucks (the HCM curves are in pc/h), else subject_cap_h. The fit is
    c = A exp(-B vc) by least squares of ln c, over the runs with capacity > 0
    in the fitted range: with ``fit`` = "validated" (default) the validated
    range (``validated`` if given, else validated_range of the reference
    series) when it spans some flows, else every run; (lo, hi) that range;
    None every run; one range for every series. Each run is weighted by its
    capacity, i.e. by its entry count (a sweep's runs share one window), as
    Var(ln c) is about 1/N for N entries (*assumption*: entries counted like a
    renewal process): a collapsed run with a handful of entries no longer
    weighs as much as one with hundreds. 95 % CIs come from ``boot``
    resamples of the seeds, drawn alike for every series (paired: the series
    share their seeds' random traffic). Each series also gets the means of its
    fitted runs' subject_tf and subject_tc_raff (FHWA-OR-RD-22-11 averages them
    over the conflicting flows, Sec. 7.3.2), with the same resamples.

    Returns {"series": [dict per series], "ref": its index, "lanes": the
    circulating lanes, "hcm": the HCM base (A, B) (presets.HCM6_CAPACITY),
    "validated": the validated range or None, "fit": the fitted range or None
    (every run), "fit_from": "validated range", "given" or "every run",
    "cap"/"vc": the columns, "units", "trucks": whether a run has trucks,
    "boot": the resamples}. ValueError if the CSV has no capacity-test run, or
    runs on 3+ circulating lanes (no HCM base curve).
    """
    from roundabout_sim.presets import HCM6_CAPACITY

    if any(c not in df for c in ("subject_vc_h", "subject_cap_h", "seed")):
        raise ValueError("it has no capacity-test columns (subject_vc_h, subject_cap_h): not a sweep of this version")
    tested = df[np.isfinite(df["subject_vc_h"].astype(float)) & np.isfinite(df["subject_cap_h"].astype(float))]
    if tested.empty:
        raise ValueError("it has no capacity-test runs: sweep with --set subject_arm=on (open mode)")
    trucks = bool("truck_share" in tested and (tested["truck_share"].astype(float) > 0).any())
    if cap is None:
        cap = "subject_cap_pch" if trucks and {"subject_cap_pch", "subject_vc_pch"} <= set(df) else "subject_cap_h"
    if cap not in CAP_COLUMNS:
        raise ValueError(f"--cap {cap}: choose from {', '.join(CAP_COLUMNS)}")
    vc_name = CAP_COLUMNS[cap]
    if cap not in df or vc_name not in df:
        raise ValueError(f"it has no {cap} / {vc_name} column: a sweep of an older version")
    runs = tested[np.isfinite(tested[vc_name].astype(float)) & np.isfinite(tested[cap].astype(float))]
    if runs.empty:
        raise ValueError(f"its {cap} is empty in every run" + (": the measured windows are shorter than 15 minutes "
                                                                f"(sweep with --measure {CAP15:g} or more)"
                                                                if cap == "subject_cap15_h" else ""))
    lanes = int(runs["n_lanes"].max()) if "n_lanes" in runs else 1
    if lanes >= 3:
        raise ValueError("runs on 3 or more circulating lanes: the HCM 6/7 roundabout model has no base curve for "
                         "them (1 entry lane against 1 or 2 circulating lanes)")
    hcm = HCM6_CAPACITY[1 if lanes <= 1 else 2]
    if by is not None and by not in runs:
        raise ValueError(f"--by {by}: no such column")
    values = _series_values(runs[by], by) if by is not None else [None]
    if ref is None:
        ref = values[0]
    elif not any(_matches(pd.Series([v]), ref).iloc[0] for v in values):
        raise ValueError(f"--ref {ref}: {by} takes {', '.join(map(_value_text, values))} here")
    ref_index = next(i for i, v in enumerate(values) if v is None or _matches(pd.Series([v]), ref).iloc[0])
    seeds = np.sort(runs["seed"].unique())
    rng = np.random.default_rng(seed)
    counts = np.zeros((boot, seeds.size))
    np.add.at(counts, (np.arange(boot)[:, None], rng.integers(seeds.size, size=(boot, seeds.size))), 1.0)
    design = _design_columns(runs, ("seed", by) if by else ("seed",))
    name = _axis_label(by).split(" (")[0] if by is not None else ""
    out = []
    for value in values:
        part = runs if value is None else runs[_matches(runs[by], value)]
        vc = part[vc_name].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            outer = _column(part, "subject_vc_outer_h") / _column(part, "subject_vc_h")
        out.append({"value": value, "label": "all runs" if value is None else f"{name} = {_value_text(value)}",
                    "vc": vc, "cap": part[cap].to_numpy(float), "seed_index": np.searchsorted(seeds, part["seed"]),
                    "tf_runs": _column(part, "subject_tf"), "tc_runs": _column(part, "subject_tc_raff"),
                    "outer_runs": np.where(vc > 0, outer, np.nan), "runs": len(part),
                    "seeds": int(part["seed"].nunique()),
                    "groups": part[design].astype(str).agg("|".join, axis=1).to_numpy() if design
                    else np.zeros(len(part))})
    r = out[ref_index]
    if validated is None:
        validated = validated_range(r["vc"], r["cap"], r["groups"], hcm)
    if fit == "validated":
        span = validated is not None and validated[1] > validated[0]
        fit, fit_from = (validated, "validated range") if span else (None, "every run")
    else:
        fit, fit_from = (None, "every run") if fit is None else (tuple(fit), "given")
    for s in out:
        vc, c = s["vc"], s["cap"]
        use = c > 0
        if fit is not None:
            use &= (vc >= fit[0]) & (vc <= fit[1])
        weights = counts[:, s["seed_index"][use]]
        a, b = fit_capacity(vc[use], c[use], c[use])
        boot_a, boot_b = fit_capacity(vc[use], c[use], weights * c[use]) if use.any() \
            else (np.full(boot, np.nan),) * 2
        s.update(used=use, A=float(a), B=float(b), boot_A=np.asarray(boot_a), boot_B=np.asarray(boot_b),
                 tf=float(_weighted_mean(s["tf_runs"][use], np.ones(use.sum()))),
                 tc=float(_weighted_mean(s["tc_runs"][use], np.ones(use.sum()))),
                 boot_tf=_weighted_mean(s["tf_runs"][use], weights), boot_tc=_weighted_mean(s["tc_runs"][use], weights),
                 outer=float(np.nanmean(s["outer_runs"][use])) if np.isfinite(s["outer_runs"][use]).any()
                 else np.nan)
    return {"series": out, "ref": ref_index, "lanes": lanes, "hcm": hcm, "boot": boot, "validated": validated,
            "fit": fit, "fit_from": fit_from, "cap": cap, "vc": vc_name, "trucks": trucks,
            "units": "pc/h" if cap == "subject_cap_pch" else "veh/h"}


def _fitted_flows(s: dict) -> tuple[float, float]:
    """(lowest, highest) conflicting flow of a series' fitted runs (NaN, NaN without any)."""
    vc = s["vc"][s["used"]]
    return (float(vc.min()), float(vc.max())) if vc.size else (np.nan, np.nan)


def capacity_table(result: dict, by: str | None) -> pd.DataFrame:
    """One row per series of :func:`capacity_analysis`: the fit and its CIs; the ratio to the HCM base at
    GAIN_VC; the mean tf and Raff tc of the fitted runs with A and B by Siegloch as HCM 6 and FHWA-OR-RD-22-11
    Eqs. 7-7 to 7-9 have them (A_hw = 3600 / tf, B_hw = (tc - tf / 2) / 3600); and against the reference series
    fA, fB, the gains at GAIN_VC and d ln(gain)/d vc from the fits (comparable to FHWA Sec. 7.3.1's simulated
    capacities), and fA_hw = tf_ref / tf, fB_hw = (tc - tf/2) / (tc_ref - tf_ref/2) from the headways (how
    FHWA made the Table 1.7 factors, Eqs. 7-10 to 7-12), each with its 95 % CI. A ratio or gain at a flow
    outside the series' fitted conflicting flows (or the reference's) is NaN: an extrapolation, not a result."""
    series, ref = result["series"], result["series"][result["ref"]]
    a0, b0 = result["hcm"]
    slope_hcm = (1.0 - CAV_FACTORS[1 if result["lanes"] <= 1 else 2][-1][2]) * b0
    validated = result["validated"] or (np.nan, np.nan)
    fit = result["fit"] or (np.nan, np.nan)
    ref_lo, ref_hi = _fitted_flows(ref)
    rows = []
    for s in series:
        lo, hi = _fitted_flows(s)
        row = {**({by: s["value"]} if by else {}), "runs": s["runs"], "seeds": s["seeds"],
               "fitted_runs": int(s["used"].sum()), "vc_min": lo, "vc_max": hi}
        for name in ("A", "B", "tf", "tc"):
            row[name] = s[name]
            row[f"{name}_lo"], row[f"{name}_hi"] = _interval(s[f"boot_{name}"])
        with np.errstate(invalid="ignore", divide="ignore"):
            row["A_hw"], row["B_hw"] = 3600.0 / s["tf"], (s["tc"] - s["tf"] / 2.0) / 3600.0
        for vc in GAIN_VC:
            row[f"hcm_ratio_{vc:.0f}"] = s["A"] * math.exp(-s["B"] * vc) / (a0 * math.exp(-b0 * vc)) \
                if lo <= vc <= hi else np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            pairs = {"fA": (s["A"] / ref["A"], s["boot_A"] / ref["boot_A"]),
                     "fB": (s["B"] / ref["B"], s["boot_B"] / ref["boot_B"]),
                     **{f"gain_{vc:.0f}": (s["A"] / ref["A"] * math.exp(-(s["B"] - ref["B"]) * vc),
                                           s["boot_A"] / ref["boot_A"] * np.exp(-(s["boot_B"] - ref["boot_B"]) * vc))
                        if max(lo, ref_lo) <= vc <= min(hi, ref_hi) else (np.nan, np.full(1, np.nan))
                        for vc in GAIN_VC},
                     "slope": (ref["B"] - s["B"], ref["boot_B"] - s["boot_B"]),
                     "fA_hw": (ref["tf"] / s["tf"], ref["boot_tf"] / s["boot_tf"]),
                     "fB_hw": ((s["tc"] - s["tf"] / 2) / (ref["tc"] - ref["tf"] / 2),
                               (s["boot_tc"] - s["boot_tf"] / 2) / (ref["boot_tc"] - ref["boot_tf"] / 2))}
        for name, (value, samples) in pairs.items():
            row[name] = float(value)
            row[f"{name}_lo"], row[f"{name}_hi"] = _interval(samples)
        row.update(slope_hcm7=slope_hcm, slope_fhwa_sim=FHWA_SIM_GAIN[1], slope_jiang=JIANG_GAIN[1],
                   outer_share=s["outer"], validated_lo=validated[0], validated_hi=validated[1], fit_lo=fit[0],
                   fit_hi=fit[1], units=result["units"])
        rows.append(row)
    return pd.DataFrame(rows)


def _range_arg(name: str, text: str | None, parser: argparse.ArgumentParser) -> tuple[float, float] | None:
    """A LO:HI option as (lo, hi) (None if not given); a parser error if it is not two numbers, lo <= hi."""
    if text is None:
        return None
    try:
        lo, hi = (float(v) for v in text.split(":"))
    except ValueError:
        parser.error(f"{name} takes LO:HI (two numbers), got {text!r}")
    if not lo <= hi:
        parser.error(f"{name}: need LO <= HI, got {text!r}")
    return lo, hi


def _read_sweep(path: str, parser: argparse.ArgumentParser) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as err:
        parser.error(f"cannot read {path}: {err}")
    if df.empty:
        parser.error(f"{path} has no runs")
    return df


def _gain_curve(s: dict, r: dict, grid: np.ndarray, k: int) -> dict:
    """A series' fitted gain over the reference ``r`` on ``grid`` with its 95 % bootstrap band, for
    capacity_figure; "range" = the flows both fitted (the band covers only those)."""
    lo, hi = max(_fitted_flows(s)[0], _fitted_flows(r)[0]), min(_fitted_flows(s)[1], _fitted_flows(r)[1])
    inside = (grid >= lo) & (grid <= hi)
    band = np.full((grid.size, 2), np.nan)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        boot = (s["boot_A"] / r["boot_A"])[:, None] * np.exp(-(s["boot_B"] - r["boot_B"])[:, None] * grid[inside])
        band[inside] = np.array([_interval(column) for column in boot.T]).reshape(-1, 2)
        mean = s["A"] / r["A"] * np.exp(-(s["B"] - r["B"]) * grid)
    return {"label": s["label"], "k": k, "vc": grid, "mean": mean, "lo": band[:, 0], "hi": band[:, 1],
            "range": (lo, hi)}


def cmd_capacity(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from roundabout_sim.plots import capacity_figure

    df = _read_sweep(args.csv, parser)
    if args.bootstrap < 1:
        parser.error("need --bootstrap >= 1")
    ref = None
    if args.ref is not None:
        if args.by is None:
            parser.error("--ref needs --by (the column whose values are the series)")
        numeric = args.by in df and pd.api.types.is_numeric_dtype(df[args.by])
        try:  # (as the column holds it: a setting's type, else a number for a numeric column)
            ref = _coerce(args.by, args.ref) if args.by in CONFIG_NAMES else float(args.ref) if numeric else args.ref
        except ValueError as err:
            parser.error(f"--ref: {err}")
        if not isinstance(ref, str) and args.by in df and not numeric:
            ref = str(ref)
    given = _range_arg("--validated", args.validated, parser)
    try:
        result = capacity_analysis(df, args.by, ref, _range_arg("--fit", args.fit, parser) or "validated",
                                   args.bootstrap, validated=given, cap=args.cap)
    except ValueError as err:
        parser.error(f"{args.csv}: {err}")
    validated, fit, units = result["validated"], result["fit"], result["units"]
    table = capacity_table(result, args.by)
    out = Path(args.out) if args.out else Path(args.csv).with_suffix("").with_name(Path(args.csv).stem + "_capacity")
    out.parent.mkdir(parents=True, exist_ok=True)
    state.write_atomic(f"{out}.csv", lambda fh: table.to_csv(fh, index=False, encoding="utf-8"))

    series, r = result["series"], result["series"][result["ref"]]
    a0, b0 = result["hcm"]
    factors = CAV_FACTORS[1 if result["lanes"] <= 1 else 2]
    lanes = "1 circulating lane" if result["lanes"] <= 1 else "2 circulating lanes"
    hcm_label = f"HCM 6/7 base {a0:g} exp(-{b0:.3g} vc) (1 entry lane, {lanes}; pc/h)"
    slope_hcm = float(table["slope_hcm7"].iloc[0])
    references = [(f"HCM 7 (Table 1.7 factors), 100 % CAVs: {factors[-1][1]:g} exp({slope_hcm:+.2g} vc)",
                   factors[-1][1], slope_hcm),
                  ("FHWA 2022 Sec. 7.3.1 simulated, 100 % CAVs: 1.36 -> 2.16 at 1,200", *FHWA_SIM_GAIN),
                  ("Jiang 2022, 100 % CAVs: 1.40 -> 1.26 at 1,200 (unverified)", *JIANG_GAIN)]
    grid = np.linspace(0.0, max(1200.0, 1.05 * max(float(np.nanmax(s["vc"])) for s in series)), 120)
    gains = [_gain_curve(s, r, grid, k) for k, s in enumerate(series) if k != result["ref"]]
    seeds = int(r["seeds"])
    fitted = {"validated range": f"in the validated range {fit[0]:.0f}-{fit[1]:.0f} {units}" if fit else "",
              "given": f"with vc {fit[0]:g}-{fit[1]:g} {units} (--fit)" if fit else "",
              "every run": "of every flow"}[result["fit_from"]]
    subtitle = (f"{Path(args.csv).name}: {sum(s['runs'] for s in series)} capacity-test runs, {seeds} seed(s)"
                + (f", {len(series)} series by {args.by} (reference {r['label']})" if args.by else "")
                + f"; {result['cap']} against {result['vc']}"
                + f"\nfits ln c = ln A - B vc over the runs {fitted} with entries, each weighted by its entries; 95 % "
                  f"CIs from {args.bootstrap} resamples of the seeds; {hcm_label}")
    fig = capacity_figure([{"label": s["label"], "vc": s["vc"], "cap": s["cap"], "A": s["A"], "B": s["B"],
                            "range": _fitted_flows(s)} for s in series], hcm=(a0, b0), hcm_label=hcm_label,
                          factors=factors, validated=validated, gains=gains, references=references,
                          ref_label=r["label"], units=units,
                          title="Roundabout capacity test: entry capacity against conflicting flow",
                          subtitle=subtitle)
    fig.savefig(f"{out}.png")

    print(subtitle)
    where = (f"validated range (the reference series within {100 * V1_TOLERANCE:g} % of the HCM base): "
             f"{validated[0]:.0f}-{validated[1]:.0f} {units}" if validated else
             f"no validated range: no design point of the reference series within {100 * V1_TOLERANCE:g} % of the "
             "HCM base")
    print(f"{where}{' (given with --validated)' if given else ''}")
    print("fitted: " + {"validated range": "the runs in the validated range (--fit LO:HI for others)",
                        "given": "the runs in the --fit range",
                        "every run": "every run (no validated range spanning some flows; --fit LO:HI to choose)"}
          [result["fit_from"]])
    if validated and fit != validated:
        beyond = sum(int((s["used"] & ((s["vc"] < validated[0]) | (s["vc"] > validated[1]))).sum()) for s in series)
        if beyond:
            print(f"warning: {plural(beyond, 'fitted run')} outside the validated range, where the model does not "
                  "follow the HCM base")
    if seeds < FEW_SEEDS:
        print(f"warning: {plural(seeds, 'seed')}: percentile bootstrap intervals from fewer than {FEW_SEEDS} seeds "
              "are too narrow (a 95 % interval covers about 76 % of the time with 3 seeds, 84 % with 5, 90 % with "
              "10); sweep with more --seeds")
    if result["trucks"] and units != "pc/h":
        print(f"note: some runs have trucks, and {result['cap']} counts vehicles while the HCM curves are in pc/h "
              "(--cap subject_cap_pch fits pc/h against pc/h)")
    at = " / ".join(f"{vc:.0f}" for vc in GAIN_VC)
    for (_, row), s in zip(table.iterrows(), series):
        print(f"\n{s['label']}: {int(row['runs'])} runs ({int(row['fitted_runs'])} fitted), {int(row['seeds'])} "
              f"seeds, fitted vc {_fmt(row['vc_min'], 0)}-{_fmt(row['vc_max'], 0)} {units}")
        print(f"  A = {_fmt(row['A'], 0)} {units} [{_fmt(row['A_lo'], 0)}, {_fmt(row['A_hi'], 0)}]   B = "
              f"{_g(row['B'])} [{_g(row['B_lo'])}, {_g(row['B_hi'])}] per {units}")
        print("  capacity / HCM base: " + " / ".join(_fmt(row[f'hcm_ratio_{vc:.0f}']) for vc in GAIN_VC)
              + f" at vc {at} {units} (- = outside the fitted flows)")
        print(f"  headways: follow-up tf {_fmt(row['tf'])} s [{_fmt(row['tf_lo'])}, {_fmt(row['tf_hi'])}], Raff "
              f"critical gap tc {_fmt(row['tc'])} s [{_fmt(row['tc_lo'])}, {_fmt(row['tc_hi'])}] -> by Siegloch "
              f"A = {_fmt(row['A_hw'], 0)}, B = {_g(row['B_hw'])} (FHWA-OR-RD-22-11 Eqs. 7-7 to 7-9)")
        if result["lanes"] > 1:
            print(f"  outer lane: {_pct(row['outer_share'], 0)} % of the conflicting flow")
        if args.by and s is not r:
            print(f"  fits:     fA = {_fmt(row['fA'], 3)} [{_fmt(row['fA_lo'], 3)}, {_fmt(row['fA_hi'], 3)}]   fB = "
                  f"{_fmt(row['fB'], 3)} [{_fmt(row['fB_lo'], 3)}, {_fmt(row['fB_hi'], 3)}]")
            print(f"  headways: fA = {_fmt(row['fA_hw'], 3)} [{_fmt(row['fA_hw_lo'], 3)}, {_fmt(row['fA_hw_hi'], 3)}]"
                  f"   fB = {_fmt(row['fB_hw'], 3)} [{_fmt(row['fB_hw_lo'], 3)}, {_fmt(row['fB_hw_hi'], 3)}] (as "
                  "FHWA made the Table 1.7 factors)")
            print("  gain " + " / ".join(f"{_fmt(row[f'gain_{vc:.0f}'], 3)} [{_fmt(row[f'gain_{vc:.0f}_lo'], 3)}, "
                                         f"{_fmt(row[f'gain_{vc:.0f}_hi'], 3)}]" for vc in GAIN_VC) + f" at vc {at}")
            print(f"  d ln(gain)/d vc = {row['slope']:+.2e} [{row['slope_lo']:+.2e}, {row['slope_hi']:+.2e}] per "
                  f"{units} (100 % CAVs: FHWA Sec. 7.3.1 simulated {FHWA_SIM_GAIN[1]:+.2e}, HCM 7 factors "
                  f"{slope_hcm:+.2e}, Jiang 2022 {JIANG_GAIN[1]:+.2e} unverified)")
    print(f"\nwrote {out}.csv and {out}.png")


def compare_table(df: pd.DataFrame, name: str, value, metrics: list[str]) -> pd.DataFrame:
    """Each run against the run at ``name`` = ``value`` with the same seed and the same other settings: per
    point (the other varied settings and each other value of ``name``) and metric, the pairs, the means, the
    mean difference and the mean ratio of the pairs with their 95 % t intervals. Ratios leave out pairs whose
    reference value is 0."""
    others = [c for c in CONFIG_NAMES if c in df and c != name]
    is_ref = _matches(df[name], value)
    key = others + ["seed"]
    pairs = df[~is_ref].merge(df[is_ref][key + metrics], on=key, suffixes=("", "__ref"))
    varied = _design_columns(df, (name, "seed"))
    rows = []
    for point, group in pairs.groupby(varied + [name], sort=False, dropna=False):
        point = point if isinstance(point, tuple) else (point,)
        for metric in metrics:
            x, x0 = group[metric].to_numpy(float), group[f"{metric}__ref"].to_numpy(float)
            both = np.isfinite(x) & np.isfinite(x0)
            diff, diff_lo, diff_hi = mean_ci(x[both] - x0[both])
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio, ratio_lo, ratio_hi = mean_ci((x / x0)[both & (x0 != 0)])
            rows.append({**dict(zip(varied + [name], point)), "metric": metric, "pairs": int(both.sum()),
                         "ref_mean": float(np.mean(x0[both])) if both.any() else np.nan,
                         "mean": float(np.mean(x[both])) if both.any() else np.nan,
                         "diff": diff, "diff_lo": diff_lo, "diff_hi": diff_hi,
                         "ratio": ratio, "ratio_lo": ratio_lo, "ratio_hi": ratio_hi})
    return pd.DataFrame(rows)


def cmd_compare(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    df = _read_sweep(args.csv, parser)
    name, text = args.ref
    if name not in df:
        parser.error(f"--ref {name}: {args.csv} has no such column")
    try:
        value = _coerce(name, text)
    except ValueError as err:
        parser.error(f"--ref: {err}")
    if "seed" not in df:
        parser.error(f"{args.csv} has no seed column: nothing to pair the runs by")
    if not _matches(df[name], value).any():
        parser.error(f"--ref {name}={text}: no run has it; {name} takes "
                     f"{', '.join(map(_value_text, _series_values(df[name], name)))} here")
    if args.metrics:
        metrics = [m.strip() for item in args.metrics for m in item.split(",") if m.strip()]
        unknown = [m for m in metrics if m not in df]
        if unknown:
            parser.error(f"--metrics: no column {', '.join(unknown)} in {args.csv}")
    else:
        metrics = [m for m in COMPARE_METRICS if m in df and np.isfinite(df[m].astype(float)).any()]
    numeric = [m for m in metrics if pd.api.types.is_numeric_dtype(df[m])]
    if len(numeric) < len(metrics):
        parser.error(f"--metrics: {', '.join(m for m in metrics if m not in numeric)} is not a numeric column")
    table = compare_table(df, name, value, metrics)
    if table.empty:
        parser.error(f"no run pairs with the --ref run (same seed and other settings) in {args.csv}")
    out = Path(args.out) if args.out else Path(args.csv).with_name(Path(args.csv).stem + "_compare")
    out.parent.mkdir(parents=True, exist_ok=True)
    state.write_atomic(f"{out}.csv", lambda fh: table.to_csv(fh, index=False, encoding="utf-8"))
    varied = [c for c in table.columns if c in CONFIG_NAMES and c != name]
    print(f"{args.csv}: each run against the same seed's run at {name} = {_value_text(value)} (same other "
          "settings); mean of the pairs and its 95 % t interval")
    for point, group in table.groupby(varied + [name], sort=False, dropna=False):
        point = point if isinstance(point, tuple) else (point,)
        print("\n" + ", ".join(f"{k} = {_value_text(v)}" for k, v in zip(varied + [name], point))
              + f" ({int(group['pairs'].max())} pairs)")
        width = max(len(m) for m in group["metric"])
        for _, row in group.iterrows():
            print(f"  {row['metric']:<{width}}  {significant(row['ref_mean'])} -> {significant(row['mean'])}:  "
                  f"diff {significant(row['diff'])} [{significant(row['diff_lo'])}, {significant(row['diff_hi'])}]"
                  f"  ratio {_fmt(row['ratio'], 3)} [{_fmt(row['ratio_lo'], 3)}, {_fmt(row['ratio_hi'], 3)}]")
    print(f"\nwrote {out}.csv")


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
                            "the plausible ranges (presets.RANDOM_RANGES) of every parameter but the AVs and "
                            "Coop tabs' (or of --sample-groups) not given with --set or its flag")
    sweep.add_argument("--sample-groups", dest="sample_groups", metavar="G1,G2",
                       help="--sample without --vary: draw only the parameters of these groups (the viewer's "
                            f"tabs: {', '.join(dict.fromkeys(p.group for p in PARAMS))}; by default all but AVs "
                            "and Coop, which are drawn only when named)")
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

    cap = sub.add_parser("capacity", help="entry capacity against conflicting flow from a capacity-test sweep -> "
                                          "fits, factors, CSV and figure",
                         description="Entry capacity against conflicting flow from a sweep of capacity-test runs "
                                     "(subject_arm=on, open mode; vary entry_rate for the conflicting flow): per "
                                     "series, c = A exp(-B vc) fitted to the runs' (subject_vc_h, subject_cap_h) in "
                                     "the validated range (each run weighted by its entries), with 95 % CIs from a "
                                     "bootstrap over the seeds (paired across the series); the ratio to the HCM 6/7 "
                                     "base curve; the mean follow-up headway and Raff critical gap, with A and B "
                                     "from them by Siegloch; and against the reference series fA = A/A_ref and fB = "
                                     "B/B_ref (from the fits and from the headways), the gain at vc 0, 400, 800 and "
                                     "1200 veh/h and d ln(gain)/d vc (at 100 % CAVs: FHWA-OR-RD-22-11's simulations "
                                     "+3.9e-4, the HCM 7 factors +1.5e-4) -> PREFIX.csv and PREFIX.png (with the HCM "
                                     "7 CAV factor curves of FHWA-OR-RD-22-11 Table 1.7, and the flows outside the "
                                     "validated range shaded).",
                         allow_abbrev=False)
    cap.add_argument("csv", help="sweep CSV with capacity-test runs")
    cap.add_argument("--by", metavar="NAME", help="column whose values are the series (e.g. road_condition); "
                                                  "default: all runs as one series")
    cap.add_argument("--ref", metavar="VALUE", help="the --by value the factors and gains are against (default: its "
                                                    "first value: the lowest number, or the first choice)")
    cap.add_argument("--fit", metavar="LO:HI", help="fit only the runs with a conflicting flow in this range [veh/h] "
                                                    "(default: the validated range, else every run with entries)")
    cap.add_argument("--validated", metavar="LO:HI",
                     help="conflicting flows [veh/h] to leave unshaded as validated (default: where the design points "
                          f"of the reference series are within {100 * V1_TOLERANCE:g} %% of the HCM base curve)")
    cap.add_argument("--cap", choices=tuple(CAP_COLUMNS), default=None,
                     help="the capacity column to fit, against its conflicting flow: subject_cap_h (entries per hour), "
                          "subject_cap15_h (4 x the busiest 15 minutes, as FHWA-OR-RD-22-11; needs --measure 900 or "
                          "more) or subject_cap_pch (pc/h, against subject_vc_pch); default subject_cap_pch when a run "
                          "has trucks, else subject_cap_h")
    cap.add_argument("--bootstrap", type=int, default=BOOTSTRAP,
                     help=f"resamples of the seeds for the CIs (default {BOOTSTRAP})")
    cap.add_argument("--out", help="output prefix (default: the CSV's, with _capacity)")
    cap.set_defaults(handler=cmd_capacity, command_parser=cap)

    compare = sub.add_parser("compare", help="each run against the same seed's run at a reference value -> paired "
                                             "differences and ratios with 95 %% CIs",
                             description="Pairs every run of a sweep CSV with the run at --ref NAME=VALUE that has the "
                                         "same seed and the same other settings, and gives per point and metric the "
                                         "mean difference and the mean ratio of the pairs with their 95 % t intervals "
                                         "(common random numbers: the pairs share their random traffic) -> "
                                         "PREFIX.csv and a printed table.",
                             allow_abbrev=False)
    compare.add_argument("csv", help="sweep CSV")
    compare.add_argument("--ref", type=_assignment, required=True, metavar="NAME=VALUE",
                         help="the reference grid value, e.g. road_condition=dry")
    compare.add_argument("--metrics", nargs="+", metavar="COLUMN",
                         help="metric columns to compare (default: " + ", ".join(COMPARE_METRICS) + ", those with "
                              "values)")
    compare.add_argument("--out", help="output prefix (default: the CSV's, with _compare)")
    compare.set_defaults(handler=cmd_compare, command_parser=compare)
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = build_parser().parse_args(argv)
    args.argv = argv  # for the command line in the sweep's meta file
    args.handler(args, args.command_parser)


if __name__ == "__main__":
    main()
