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

Any model parameter can be set with ``--set name=value`` (repeatable) or its
own flag (``--density 70``); ``--help`` on each command lists them all.

Outputs
  run    PREFIX_timeseries.csv  one row per sample (every --interval s)
         PREFIX_spacetime.npz   speed per angular bin and sector occupancy over time
         PREFIX_summary.csv     one row: every setting + the metrics over the second half
                                (+ baseline_* and arm_jam_share with --baseline)
         PREFIX.png             dashboard
  sweep  PREFIX.csv             one row per run: every setting + the metrics
                                averaged over the measurement window
         PREFIX.png             metrics against --x (mean across seeds, band = min-max)

CSV columns: settings are named exactly like the parameters (density, n_lanes,
reaction_time, ...). A measurement that shares a parameter's name carries a
``measured_`` prefix: ``measured_density`` is the density actually on the ring,
``density`` the setting; ``measured_speed_std`` is the spread of the cars'
current speeds, ``speed_std`` the spread of their desired speeds. Every other
column is measured.
"""

from __future__ import annotations

import argparse
import itertools
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from roundabout_sim.config import PARAM_BY_NAME, PARAMS, SimConfig, coerce
from roundabout_sim.engine import MAX_DECEL, SLOW_FRACTION, TWO_PI, RingSim
from roundabout_sim.metrics import RATE_WINDOW, Recorder

CONFIG_NAMES = tuple(f.name for f in fields(SimConfig) if f.name != "seed")
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
  jam_fraction         CONGESTION (the headline): share of cars crawling below
                       {100 * SLOW_FRACTION:g} % of their free speed. Rises steadily with density.
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
                       length): 0 = even, ~0.35 = random gaps, towards 1 = packed
                       queues or platoons. Good for initial spacing and platoons,
                       not a jam detector.
  clustering           spread of the cars over 6-24 sectors beyond random scatter:
                       0 = even or random, ~1 = half the ring empty (often 0 in
                       dense flow)
  order_param          0 = balanced around the ring, 1 = all cars on one side
  order_baseline       sqrt(pi / 4N): what N randomly placed cars give for
                       order_param; only values well above it are one-sided
  emergency_per_min    Harsh braking (episodes/min): the driver model asking for
                       more than {MAX_DECEL:g} m/s2 (an over-reaction after a late or delayed
                       response), or the no-overlap constraint stopping a car
                       harder. 0 with the default drivers on a ring without exits;
                       a few per minute from merging at entrances/exits.
  wave_speed           speed of jam patterns along the ring [m/s]; < 0 = moving
                       backwards against the traffic, as stop-and-go waves do"""

# Recorder columns averaged over a measurement window.
MEAN_COLUMNS = ("cars", "queued", "density", "mean_speed", "mean_free_speed", "speed_std", "flux",
                "jam_fraction", "speed_deficit", "n_jams", *JAM_ORIGINS, "occupancy", "spacing_gini",
                "clustering", "order_param", "order_baseline", "mean_trip_time")
# Columns of the sweep's printed table (means across seeds).
SUMMARY_COLUMNS = ("measured_density", "measured_mean_speed", "flux", "jam_fraction", "congestion_onset",
                   "stops_per_car_min", "stop_go_onset", "n_jams", "following_jam_share", "spacing_gini",
                   "order_param", "emergency_per_min", "wave_speed")


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


def _counters(sim: RingSim) -> dict[str, int]:
    return {"exits": sim.n_exited, "missed": sim.n_missed, "lane_changes": sim.n_lane_changes,
            "emergency": sim.n_emergency, "stops": sim.n_stops}


def measured_names(names) -> dict[str, str]:
    """Renames that give measurements sharing a parameter's name a ``measured_`` prefix."""
    return {k: f"measured_{k}" for k in names if k in CONFIG_NAMES}


def settings(sim: RingSim) -> dict:
    """Every parameter of ``sim`` (named like the CLI ``--set`` names) and its seed."""
    return {**{name: getattr(sim.cfg, name) for name in CONFIG_NAMES}, "seed": sim.seed}


def summarise(sim: RingSim, rec: Recorder, t0: float, start: dict[str, int]) -> dict:
    """Metrics from ``t0`` to now (``start`` = :func:`_counters` at ``t0``).

    Sampled metrics are averaged over the window; event rates come from the
    engine's counters, so they count exactly the events inside the window.
    ``stop_go_onset`` and ``congestion_onset`` are measured from the start of
    the run. Metrics that share a parameter's name get a ``measured_`` prefix.
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
    row = {
        **{k: mean[k] for k in ("cars", "queued", "density", "mean_speed", "mean_free_speed", "speed_std",
                                "flux")},
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


def free_speed(sim: RingSim) -> float:
    """Free speed of an average driver on the middle lane (top of the speed colour scale)."""
    return float(min(sim.cfg.mean_speed, np.sqrt(sim.cfg.lateral_accel * sim.radii.mean())))


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


def _coerce(name: str, value: str):
    """config.coerce, with the parameter named in the error message."""
    try:
        return coerce(name, value)
    except ValueError as err:
        raise ValueError(f"bad value {value!r} for {name}: {err}") from None


def _values(name: str, spec: str) -> list:
    """``start:stop:step`` (stop included) or a comma list, converted to the parameter's type."""
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"{name}: range must be start:stop:step, got {spec!r}")
        start, stop, step = (float(p) for p in parts)
        if step <= 0 or stop < start:
            raise ValueError(f"{name}: need step > 0 and stop >= start, got {spec!r}")
        count = int(np.floor((stop - start) / step + 1e-9)) + 1
        raw = [round(start + i * step, 10) for i in range(count)]
    else:
        raw = [v.strip() for v in spec.split(",") if v.strip()]
    values = list(dict.fromkeys(_coerce(name, v) for v in raw))
    if not values:
        raise ValueError(f"{name}: no values in {spec!r}")
    return values


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
        spec = PARAM_BY_NAME.get(name)
        if spec and not spec.choices and not spec.lo <= value <= spec.hi:
            print(f"note: {name}={value:g} is outside the usual range {spec.lo:g}-{spec.hi:g}")
    return changes


def _road(cfg: SimConfig) -> str:
    return f"{cfg.n_lanes} lane{'s' * (cfg.n_lanes != 1)}, " + \
        (f"{cfg.n_arms} entrances/exits" if cfg.n_arms else "closed ring (no entrances/exits)")


def _describe(cfg: SimConfig) -> str:
    return (f"{_road(cfg)}, radius {cfg.radius:g} m, car {cfg.car_length:g} m | density {cfg.density:g} veh/km/lane | "
            f"desired speed {cfg.mean_speed:g} ± {cfg.speed_std:g} m/s, reaction {cfg.reaction_time:g} s / "
            f"{cfg.reaction_distance:g} m, headway {cfg.time_headway:g} s, noise {cfg.driver_noise:g} m/s² | "
            f"lane changes {cfg.lane_change_rate:g}/s ({cfg.lane_change_motive}) | start {cfg.init_spacing}")


def _plain(text: str) -> str:
    """ASCII version for the console (Windows terminals often mangle the rest)."""
    return text.replace("±", "+/-").replace("²", "2")


def _value_text(value) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def _label(name: str, value) -> str:
    spec = PARAM_BY_NAME.get(name)
    label = spec.label if spec else name
    return f"{label} = {_value_text(value)}{' ' + spec.unit if spec and spec.unit else ''}"


def _axis_label(name: str) -> str:
    spec = PARAM_BY_NAME.get(name)
    return f"{spec.label} ({spec.unit})" if spec and spec.unit else (spec.label if spec else name)


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

    changes = _overrides(args, parser)
    if args.duration <= 0 or args.interval <= 0:
        parser.error("need --duration > 0 and --interval > 0")
    cfg = SimConfig().updated(**changes, seed=args.seed)
    late_from = round(args.duration / 2, 6)
    print(f"Simulating {args.duration:g} s (seed {cfg.seed}): {_plain(_describe(cfg))}")
    t_wall = time.time()
    sim, rec, summary, car_speeds = _simulate(cfg, args.duration, args.interval, late_from, keep_speeds=True)

    base_frame, base = None, {}
    if args.baseline:
        if cfg.n_arms == 0 or cfg.exit_prob == 0:
            print("note: --baseline skipped: " + ("a closed ring has no entrances/exits" if cfg.n_arms == 0
                                                  else "exit_prob is already 0") + ", so this run is its own baseline")
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
    pd.DataFrame([{**settings(sim), **summary, **base}]).to_csv(f"{out}_summary.csv", index=False)
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
        subtitle=f"{_describe(sim.cfg)} | seed {sim.seed}",
        n_lanes=sim.n_lanes, arm_deg=np.degrees(sim.arm_angle),
        detector_deg=float(np.degrees(sim.detector_angle)), v_free=free_speed(sim),
        late_from=late_from, stop_index=stop_go_index(frame), onset=summary["stop_go_onset"],
        onset_threshold=STOP_GO_ONSET, following_share=summary["following_jam_share"],
        wave_speed=summary["wave_speed"], rate_window=RATE_WINDOW,
        congestion_onset=summary["congestion_onset"], congestion_threshold=CONGESTION_ONSET,
        baseline=base_frame, arm_share=base.get("arm_jam_share", float("nan")))
    fig.savefig(f"{out}.png")

    rows = [
        ("Cars on the ring", _fmt(summary["cars"], 0), ""),
        ("Actual density", _fmt(summary["measured_density"], 1), "veh/km/lane"),
        ("Cars queued at entries", _fmt(summary["queued"], 1), ""),
        ("Mean speed", _fmt(summary["measured_mean_speed"]), "m/s"),
        ("Mean desired speed", _fmt(summary["mean_free_speed"]), "m/s (what the drivers on the ring want)"),
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
         "% of jams (upper bound on natural jams; see --baseline)"),
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
    ]
    if args.baseline and base_frame is None:
        rows.append(("Baseline: same run, exit_prob 0", "skipped", "(nothing to switch off; columns left empty)"))
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


def cmd_sweep(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from roundabout_sim.plots import sweep_figure

    fixed = _overrides(args, parser)
    if args.seeds < 1 or args.warmup < 0 or args.measure <= 0 or args.interval <= 0:
        parser.error("need --seeds >= 1, --warmup >= 0, --measure > 0 and --interval > 0")
    x_name, x_spec = args.x
    try:
        xs = _values(x_name, x_spec)
        series_name, series_values = (args.series[0], _values(*args.series)) if args.series else (None, [None])
    except ValueError as err:
        parser.error(str(err))
    if series_name == x_name:
        parser.error("--x and --series must be different parameters")
    if len(series_values) > 8:
        parser.error("at most 8 --series values (one colour each)")
    if "init_spacing" in (x_name, series_name):
        print(f"note: {INIT_SPACING_NOTE}")

    jobs = []
    for s_value, x_value, k in itertools.product(series_values, xs, range(args.seeds)):
        changes = {**fixed, x_name: x_value}
        if series_name:
            changes[series_name] = s_value
        jobs.append((changes, args.seed + k, args.warmup, args.measure, args.interval))
    workers = max(1, min(args.workers, len(jobs)))
    print(f"{len(jobs)} runs ({len(xs)} x {len(series_values)} series x {args.seeds} seeds) on {workers} "
          f"worker(s); {args.warmup:g} s warm-up + {args.measure:g} s measured each")

    rows, t_wall = [], time.time()

    def report(i: int, r: dict) -> None:
        elapsed = time.time() - t_wall
        eta = elapsed / i * (len(jobs) - i)
        series = f"{series_name}={r[series_name]} " if series_name else ""
        print(f"[{i:3d}/{len(jobs)}] {series}{x_name}={r[x_name]} seed={r['seed']}: "
              f"flux {_fmt(r['flux'], 3)} cars/s, speed {_fmt(r['measured_mean_speed'])} m/s, "
              f"jammed {_pct(r['jam_fraction'], 0)} % (from {_fmt(r['congestion_onset'], 0)} s), "
              f"stop-and-go {_fmt(r['stops_per_car_min'])}/car/min (from {_fmt(r['stop_go_onset'], 0)} s), "
              f"front car following {_pct(r['following_jam_share'], 0)} % "
              f"({elapsed:.0f} s, ~{eta:.0f} s left)", flush=True)

    if workers == 1:
        for i, job in enumerate(jobs, 1):
            rows.append(_sweep_job(job))
            report(i, rows[-1])
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_sweep_job, job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                rows.append(future.result())
                report(i, rows[-1])

    keys = [k for k in (series_name, x_name) if k]
    given = {x_name: {v: i for i, v in enumerate(xs)}, series_name: {v: i for i, v in enumerate(series_values)}}
    df = pd.DataFrame(rows)
    df = df.sort_values(keys + ["seed"], key=lambda c: c.map(given[c.name]) if c.name in keys else c)
    # Columns: the swept settings and the seed, the other settings, then the measurements.
    first = keys + ["seed"] + [n for n in CONFIG_NAMES if n not in keys]
    df = df[first + [c for c in df.columns if c not in first]].reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{out}.csv", index=False)

    base = SimConfig().updated(**fixed)
    held = [n for n in ("n_lanes", "n_arms", "density", "speed_std", "reaction_time", "reaction_distance",
                        "lane_change_rate", "driver_noise") if n not in (x_name, series_name)]
    subtitle = (", ".join(_label(n, getattr(base, n)) for n in held)
                + f"\n{args.seeds} seed(s) per point, {args.warmup:g} s warm-up + {args.measure:g} s measured; "
                  "line = mean, band = min-max across seeds")
    fig = sweep_figure(df, x_name, series_name, x_label=_axis_label(x_name), series_label=_value_text,
                       title=f"Roundabout sweep: metrics vs {_axis_label(x_name).split(' (')[0].lower()}",
                       subtitle=subtitle, legend_title=_axis_label(series_name) if series_name else "",
                       onset_threshold=STOP_GO_ONSET, congestion_threshold=CONGESTION_ONSET)
    fig.savefig(f"{out}.png")

    table = df.groupby(keys, sort=False)[list(SUMMARY_COLUMNS)].mean()
    table[["jam_fraction", "following_jam_share"]] *= 100
    table = table.rename(columns={"measured_density": "dens_meas", "measured_mean_speed": "speed",
                                  "jam_fraction": "jam_%", "congestion_onset": "congested_s",
                                  "stops_per_car_min": "stops/car/min", "stop_go_onset": "stop_go_s",
                                  "following_jam_share": "following_%", "spacing_gini": "gini",
                                  "order_param": "order", "emergency_per_min": "harsh/min",
                                  "wave_speed": "wave_m/s"})
    print("\nMeans across seeds:")
    print(table.to_string(float_format=lambda v: f"{v:.3g}", na_rep="-"))
    print(f"(jam_% = cars below {100 * SLOW_FRACTION:g} % of free speed, the congestion level; congested_s = first "
          f"time its {RATE_WINDOW:g} s mean > {100 * CONGESTION_ONSET:g} %;\n"
          f" stops/car/min = stop-and-go index (instability: ~0 when smooth and in a standstill); stop_go_s = "
          f"first time it > {STOP_GO_ONSET:g};\n onset times are means over the runs that got there; "
          f"following_% = jams whose front car is just following,\n an upper bound on natural jams - compare "
          f"--series exit_prob=0,0.1 or run --baseline; harsh/min = Harsh braking (episodes/min))")
    _print_arm_shares(df, keys, fixed)
    print(f"\nwrote {out}.csv and {out}.png ({time.time() - t_wall:.0f} s)")


def arm_share_table(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame | None:
    """Share of the jamming caused by entrances/exits, from a sweep that includes exit_prob = 0.

    Rows: the other swept parameter (if any); columns: each exit_prob > 0.
    None if exit_prob is not swept or 0 is not among its values.
    """
    if "exit_prob" not in keys or not (df["exit_prob"] == 0).any():
        return None
    jammed = df.groupby(keys, sort=False)["jam_fraction"].mean()
    other = [k for k in keys if k != "exit_prob"]
    wide = jammed.unstack("exit_prob") if other else jammed.to_frame().T
    base = wide[0.0]
    shares = {f"exit_prob={p:g}": [arm_share(run, b) for run, b in zip(wide[p], base)]
              for p in wide.columns if p > 0}
    return pd.DataFrame(shares, index=wide.index if other else ["all"])


def _print_arm_shares(df: pd.DataFrame, keys: list[str], fixed: dict) -> None:
    table = arm_share_table(df, keys)
    if table is None:
        return
    if fixed.get("n_arms", SimConfig().n_arms) == 0 and "n_arms" not in keys:
        print("\n(exit_prob does nothing on a closed ring: n_arms = 0)")
        return
    print(f"\nJamming due to entrances/exits (%) = 1 - jammed(exit_prob 0) / jammed, from the means across "
          f"seeds; - = under {100 * JAM_PRESENT:g} % jammed:")
    print((100 * table).to_string(float_format=lambda v: f"{v:.0f}", na_rep="-"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m roundabout_sim.experiments",
                                     description=f"{__doc__}\n{DEFINITIONS}",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="one simulation -> time series, space-time data and a dashboard figure",
                         description="One simulation -> PREFIX_timeseries.csv, PREFIX_summary.csv, "
                                     "PREFIX_spacetime.npz and a dashboard PREFIX.png. The summary covers "
                                     "the second half of the run. With --baseline the run is repeated with "
                                     "exit_prob = 0 to measure what the entrances/exits cause.",
                         epilog=DEFINITIONS, formatter_class=argparse.RawDescriptionHelpFormatter,
                         allow_abbrev=False)
    run.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    run.add_argument("--duration", type=float, default=600.0, help="simulated seconds (default 600)")
    run.add_argument("--interval", type=float, default=0.5, help="sampling interval in s (default 0.5)")
    run.add_argument("--out", default="results/run", help="output prefix (default results/run)")
    run.add_argument("--baseline", action="store_true",
                     help="also repeat the run with exit_prob = 0 (same seed and every other setting) and "
                          "report how much of the jamming the entrances/exits cause (baseline_* and "
                          "arm_jam_share in the summary; dashed line in the dashboard)")
    _add_param_flags(run)
    run.set_defaults(handler=cmd_run, command_parser=run)

    sweep = sub.add_parser("sweep", help="parameter sweep in parallel -> CSV and small-multiple figure",
                           description="Parameter sweep -> PREFIX.csv (one row per run) and PREFIX.png. "
                                       "Each run warms up, then the metrics are averaged over the "
                                       "measurement window (onset times: from the start). Sweeping "
                                       "exit_prob with 0 among its values also prints how much of the "
                                       "jamming the entrances/exits cause.",
                           epilog=DEFINITIONS, formatter_class=argparse.RawDescriptionHelpFormatter,
                           allow_abbrev=False)
    sweep.add_argument("--x", required=True, type=_assignment, metavar="NAME=SPEC",
                       help="parameter on the x axis; SPEC = start:stop:step (stop included) or a comma list")
    sweep.add_argument("--series", type=_assignment, metavar="NAME=V1,V2,...",
                       help="parameter drawn as one line per value (up to 8)")
    sweep.add_argument("--seeds", type=int, default=3, help="repetitions per point (default 3)")
    sweep.add_argument("--seed", type=int, default=0, help="base seed; run k of each point uses base + k")
    sweep.add_argument("--warmup", type=float, default=120.0, help="simulated warm-up in s (default 120)")
    sweep.add_argument("--measure", type=float, default=300.0, help="simulated measurement in s (default 300)")
    sweep.add_argument("--interval", type=float, default=0.5, help="sampling interval in s (default 0.5)")
    sweep.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                       help="parallel processes (default: CPU count - 1)")
    sweep.add_argument("--out", default="results/sweep", help="output prefix (default results/sweep)")
    _add_param_flags(sweep)
    sweep.set_defaults(handler=cmd_sweep, command_parser=sweep)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args, args.command_parser)


if __name__ == "__main__":
    main()
