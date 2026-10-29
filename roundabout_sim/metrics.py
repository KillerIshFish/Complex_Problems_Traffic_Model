"""Measurements of a running :class:`~roundabout_sim.engine.RingSim`.

Flow
  * ``mean_speed`` - average speed of the cars on the ring [m/s].
  * ``flux`` - cars per second passing a cross-section of the ring, summed
    over lanes, from the instantaneous speeds (sum of v / lane length).
  * ``detector_flux`` - cars per second actually crossing the detector line
    (between two arms), averaged over the last ``window`` seconds.

Congestion ("how jammed is it?")
  * ``jam_fraction`` - share of cars crawling below 30 % of their free speed
    (dense queues and stop-and-go waves alike); rises steadily with density.
    The headline "congestion over time" measure.
  * ``stops_per_car_min`` - stop-and-go: how often a moving car is forced to
    (nearly) stop, per car per minute. ~0 in smooth flow however dense, and
    also in a standstill (nobody is moving to be stopped): it measures
    instability, not how congested the ring is.
  * ``speed_deficit`` - 1 - (total speed / total free speed): 0 = everyone at
    their desired speed, 1 = standstill.
  * ``n_jams`` - number of distinct jams: runs of 2+ consecutive jammed cars
    in a lane (or a whole lane jammed round); ``jams_following / _merge /
    _exit / _entry / _other`` split them by what the front car is doing right
    now (see :func:`jam_origins`). A wave set off by an exit keeps travelling
    backwards with ordinary followers at its front, so "following" is an upper
    bound on natural jams; the honest test of what the arms cause is to compare
    with the same run at exit_prob = 0.
  * ``emergency_per_min`` - harsh-braking episodes per minute: the driver
    model asking for more than the braking limit (MAX_DECEL on a dry road,
    less on wet, snowy or icy ones; over-reaction after a late or delayed
    response), or the no-overlap constraint stopping a car harder.

Vehicles
  * ``trucks`` - trucks on the ring; ``truck_speed`` / ``car_speed`` - mean
    speed of the trucks and of the other vehicles (NaN if there are none).

Lanes
  * ``lane{k}_cars``, ``lane{k}_speed``, ``lane{k}_speed_std`` - cars on lane
    k (0 = innermost), their mean speed and its spread (NaN on an empty lane).

Space ("do cars congregate on one side?")
  * ``occupancy`` - share of the road surface covered by vehicles (their real
    lengths and widths).
  * ``clustering`` - spread of the car count over equal sectors of the ring
    (6-24, about 4+ cars each), beyond random scatter, relative to the mean:
    0 = even or random, ~1 = half the ring empty. Comparable across densities.
  * ``order_param`` - |mean of e^(i*theta)| over all cars: 0 = balanced around
    the ring, 1 = all on one side; ``order_angle`` is where they bunch up.
    ``order_baseline`` = sqrt(pi / 4N) is what N randomly placed cars give,
    so only values well above it mean real one-sidedness.
  * ``spacing_gini`` - Gini coefficient of the space each car has (gap to the
    car ahead + the mean vehicle length, so trucks alone do not raise it):
    0 = road shared evenly, ~0.35 = random gaps, towards 1 = packed queues
    with empty road between (platoons or jams).

Entrances ("how long do arriving cars wait?")
  * ``queued`` / ``longest_queue`` - cars waiting at all entrances / at the
    longest queue.
  * ``arrivals_per_min`` - cars arriving at the entrances (open mode: every
    random arrival, blocked ones included; closed mode: the leavers'
    replacements that join a queue). ``demand_per_min`` is what the demand
    settings ask for right now (open mode; NaN in closed mode).
  * ``entries_per_min`` - cars entering the ring from the queues.
  * ``blocked_per_min`` - open mode: arrivals that found their queue full
    (``max_queue``) and were turned away, i.e. spill-back.
  * ``mean_entry_delay`` - mean queue wait of the cars that entered in the
    window, from joining the queue to entering the ring (NaN if none
    entered). :func:`level_of_service` turns it into an HCM-style letter. It
    is only the wait in the queue, not the HCM control delay (which also
    counts slowing down and speeding up).
  * ``entry_vc`` - open mode: demand / capacity (v/c) of the busiest
    entrance (see :func:`entry_vc`); above 1 it gets more arrivals than it
    can let in, and the level of service is F whatever the delay.
  * ``throughput_per_h`` - cars leaving the ring per hour.

In the capacity test (``subject_arm`` on) the test entrance's standing queue
is an instrument, not traffic: ``queued``, ``longest_queue`` and
``mean_entry_delay`` leave it out (``entries_per_min`` counts its entries).

Every arrival ends up entered, still queued, blocked or given up (a queued
car dropped when Max queue or Density is lowered live: ``RingSim.n_gave_up``).

:class:`Recorder` samples these at a fixed interval, together with a
space-time diagram (mean speed per angular bin over time) and per-sector
occupancy, for the live charts and the experiment plots. Its samples are
saved and restored with the simulation (``Recorder.state_dict`` /
``load_state``; files: :mod:`roundabout_sim.state`); samples saved before a
key was added get it filled in (ADDED_KEYS).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Iterable

import numpy as np
import pandas as pd

from roundabout_sim.engine import LOG_HORIZON, SLOW_FRACTION, SUBJECT_ARM, TRUCK_WIDTH, TWO_PI, RingSim, StateError

SECTORS = 24  # sectors for the clustering index and the occupancy ring
SPACETIME_BINS = 180  # 2-degree bins for the space-time diagram
# How a saved Recorder marks each value of its rows (0: the row has no such key).
MISSING, FLOAT, INT, BOOL, TEXT, NONE = range(6)
# A saved Recorder's clock: the shortest sampling interval [s] it may have (the viewer samples every
# 0.5 s), and how many intervals its next sample may be overdue (a recorder not updated for a while).
MIN_INTERVAL = 1e-3
MAX_CATCH_UP = 1_000_000
# Snapshot keys added after the first saved states, with the value the rows saved before them get:
# Recorder.load_state fills a key only when no saved row has it (one that some rows lack is damaged). A key
# with "{k}" stands for one per lane, filled for the lanes the rows have (their lane{k}_cars). Viewer code
# reads these keys with row.get, so a row made elsewhere without them still draws.
ADDED_KEYS: dict[str, float] = {"lane{k}_speed_std": float("nan")}


RATE_WINDOW = 30.0  # [s] default window for rolling rates
# [s] shortest window for the rolling demand / capacity ratio: over 30 s it would hang on a few
# cars per entrance (HCM analyses use 15-minute periods).
VC_WINDOW = 300.0

# HCM-style level of service from the entry delay: upper limit [s] of each letter, F above the last
# (the HCM 6th edition's roundabout thresholds). Applied here to the queue wait only, not to the
# full HCM control delay, so it reads somewhat better than an HCM analysis of the same traffic.
# As in the HCM's table for an entry lane, demand above capacity (v/c > 1) is F whatever the delay:
# with Max queue turning arrivals away, the wait of the cars that got in no longer shows it.
LOS_LIMITS = (("A", 10.0), ("B", 15.0), ("C", 25.0), ("D", 35.0), ("E", 50.0))
LOS_VC_TEXT = "F whenever an entrance gets more arrivals than it can let in (v/c > 1)"


def los_text(le: str = "≤") -> str:
    """The level-of-service limits as the viewers and figures state them: 'A ≤ 10, B ≤ 15, C ≤ 25, D ≤ 35,
    E ≤ 50 s, F above' (``le="<="`` for the console)."""
    return ", ".join(f"{letter} {le} {limit:g}" for letter, limit in LOS_LIMITS) + " s, F above"


def level_of_service(delay: float, vc: float | None = None) -> str:
    """HCM-style level of service letter (A-F) for an entry delay [s]; "-" if there is none (NaN).

    F whenever the demand / capacity ratio ``vc`` (:func:`entry_vc`) is above 1.
    """
    if delay is None or not np.isfinite(delay):
        return "-"
    if vc is not None and np.isfinite(vc) and vc > 1.0:
        return "F"
    return next((letter for letter, limit in LOS_LIMITS if delay <= limit), "F")


def entry_vc(arrived, entered, busy, span: float) -> float:
    """Demand / capacity (v/c) of the busiest entrance from per-entrance counts over ``span`` s.

    ``arrived``/``entered`` are each entrance's arrivals (blocked ones
    included) and entries, ``busy`` the time [s] it had cars waiting. Its
    capacity is estimated as its entry rate while it had a queue (a server's
    service rate in queueing terms), so v/c = arrivals per second / (entries
    / busy). Below capacity this is about the share of the time it had a
    queue; above 1 arrivals outrun what it lets in. Entrances that let no car
    in are left out; NaN if none did.
    """
    arrived, entered, busy = (np.asarray(a, dtype=float) for a in (arrived, entered, busy))
    ok = entered > 0
    if span <= 0 or not ok.any():
        return float("nan")
    return float(np.max(arrived[ok] * busy[ok] / (entered[ok] * span)))


def _rolling_entry_vc(sim: RingSim, window: float) -> float:
    """:func:`entry_vc` over about the last ``window`` s, at least VC_WINDOW (open mode; NaN in
    closed mode, whose arrivals are only the leavers' replacements that found room in a queue)."""
    if not sim.open_demand or not sim.entrance_log:
        return float("nan")
    cutoff = sim.time - min(max(window, VC_WINDOW), LOG_HORIZON) + 1e-9
    ref = next((entry for entry in reversed(sim.entrance_log) if entry[0] <= cutoff), sim.entrance_log[0])
    when, arrived, entered, busy = ref
    return entry_vc(sim.arm_arrived - arrived, sim.arm_entered - entered, sim.queue_busy - busy, sim.time - when)


def _since(log: deque, cutoff: float) -> list:
    """Entries of a time-ordered engine log at or after ``cutoff``, oldest first (read-only)."""
    recent = []
    for entry in reversed(log):
        if entry[0] < cutoff:
            break
        recent.append(entry)
    return recent[::-1]


def _count_since(log: deque, cutoff: float, count_at: int | None = None) -> int:
    """Events in a time-ordered engine log at or after ``cutoff`` (read-only).

    Entries are times or tuples starting with the time; ``count_at`` names the
    tuple field holding how many events the entry stands for.
    """
    total = 0
    for entry in reversed(log):
        when = entry[0] if isinstance(entry, tuple) else entry
        if when < cutoff:
            break
        total += entry[count_at] if count_at is not None else 1
    return total


def _rolling_rate(log: deque, now: float, window: float, count_at: int | None = None) -> float:
    """Events per second over the last ``window`` seconds."""
    span = min(now, window)
    return _count_since(log, now - window, count_at) / span if span > 0 else 0.0


def jammed(sim: RingSim, idx: np.ndarray, v_free: np.ndarray | None = None) -> np.ndarray:
    v_free = sim.desired_speed(idx) if v_free is None else v_free
    return sim.v[idx] < SLOW_FRACTION * v_free


def count_jams(sim: RingSim, idx: np.ndarray, slow: np.ndarray | None = None) -> int:
    """Runs of at least two consecutive jammed cars in the same lane (around the ring).

    ``slow`` is :func:`jammed` for ``idx`` if already known.
    """
    slow = jammed(sim, idx) if slow is None else slow
    order = np.argsort(sim.theta[idx], kind="stable")
    order = order[np.argsort(sim.lane[idx][order].astype(np.int16), kind="stable")]
    lanes, slow = sim.lane[idx][order], slow[order]
    size = np.bincount(lanes, minlength=sim.n_lanes)
    start = (np.cumsum(size) - size)[lanes]
    end = start + size[lanes] - 1
    i = np.arange(lanes.size)
    before = slow[np.where(i > start, i - 1, end)]  # neighbours round each lane's ring
    after = slow[np.where(i < end, i + 1, start)]
    runs = int((slow & ~before & after).sum())
    all_slow = (size >= 2) & (np.bincount(lanes, weights=slow, minlength=sim.n_lanes) == size)
    return runs + int(all_slow.sum())  # a lane that is one closed jam has no start


def jam_heads(sim: RingSim, idx: np.ndarray, slow: np.ndarray) -> np.ndarray:
    """Slots of the front (downstream) car of each jam counted by :func:`count_jams`.

    For a lane that is one closed jam all the way round, the front car is the
    one with the most room ahead of it (bumper to bumper, whatever the lengths).
    """
    order = np.argsort(sim.theta[idx], kind="stable")
    order = order[np.argsort(sim.lane[idx][order].astype(np.int16), kind="stable")]
    lanes, s = sim.lane[idx][order], slow[order]
    size = np.bincount(lanes, minlength=sim.n_lanes)
    start = (np.cumsum(size) - size)[lanes]
    end = start + size[lanes] - 1
    i = np.arange(lanes.size)
    before = s[np.where(i > start, i - 1, end)]
    nxt = np.where(i < end, i + 1, start)
    heads = [idx[order][s & before & ~s[nxt]]]
    all_slow = (size >= 2) & (np.bincount(lanes, weights=s, minlength=sim.n_lanes) == size)
    for lane in np.flatnonzero(all_slow):
        in_lane = idx[order][lanes == lane]
        th = sim.theta[in_lane]
        room = (np.roll(th, -1) - th) % TWO_PI
        if not sim.uniform_length:  # the bumper gap: less half of each one's length
            length = sim.length[in_lane]
            room = room * sim.radii[lane] - 0.5 * (length + np.roll(length, -1))
        heads.append(in_lane[[int(np.argmax(room))]])
    return np.concatenate(heads)


ORIGIN_MEMORY = 8.0  # [s] a car still counts as "just entered" / "just inserted" this long


def jam_origins(sim: RingSim, heads: np.ndarray) -> dict[str, int]:
    """What the front car of each jam is doing right now.

    other: braked by the user, or just slotted in by the density control;
    merge: a leaver waiting to move out to its exit; exit: slowing to leave;
    entry: came in from an entrance in the last few seconds; following: none
    of these - the front car is just following. The counts add up to
    ``n_jams``. This says what is holding each jam up now, not what started
    it: a wave set off at an exit keeps travelling backwards with ordinary
    followers at its front. To measure what the arms cause, compare with the
    same run at exit_prob = 0.
    """
    t = sim.time
    other = (sim.brake_until[heads] > t) | (t - sim.inserted_at[heads] < ORIGIN_MEMORY)
    merge = ~other & sim.must_move_out(heads)
    to_exit = (sim.exit_at[heads] - sim.travel[heads]) * sim.r_out
    exiting = ~other & ~merge & (sim.lane[heads] == sim.outer) & (to_exit < 60.0)
    entry = ~other & ~merge & ~exiting & (t - sim.entered_at[heads] < ORIGIN_MEMORY)
    following = ~(other | merge | exiting | entry)
    return {"jams_following": int(following.sum()), "jams_merge": int(merge.sum()),
            "jams_exit": int(exiting.sum()), "jams_entry": int(entry.sum()),
            "jams_other": int(other.sum())}


def spacing_gini(sim: RingSim) -> float:
    """Gini coefficient of the road length each car has: its bumper gap ahead plus the mean vehicle
    length (a car's without trucks). One length for all, so evenly spaced traffic scores 0 however
    cars and trucks mix: the mix of lengths alone is no congregation."""
    _, gap = sim.bumper_gaps()
    if gap.size < 2:
        return 0.0
    pitch = np.sort(gap + sim.mean_length)
    n = pitch.size
    return float(2 * np.sum(np.arange(1, n + 1) * pitch) / (n * pitch.sum()) - (n + 1) / n)


def sector_counts(sim: RingSim, idx: np.ndarray, sectors: int = SECTORS) -> np.ndarray:
    bins = (sim.theta[idx] / TWO_PI * sectors).astype(int) % sectors
    return np.bincount(bins, minlength=sectors)


def sector_occupancy(sim: RingSim, sectors: int = SECTORS) -> np.ndarray:
    """Fraction of each sector's lane length covered by vehicles (all lanes together; by centre)."""
    idx = np.flatnonzero(sim.alive)
    sector_len = sim.total_lane_length / sectors
    if sim.uniform_length:  # all cars: their count times a car's length
        return sector_counts(sim, idx, sectors) * sim.cfg.car_length / sector_len
    bins = (sim.theta[idx] / TWO_PI * sectors).astype(int) % sectors
    return np.bincount(bins, weights=sim.length[idx], minlength=sectors) / sector_len


def covered_area(sim: RingSim, idx: np.ndarray) -> float:
    """[m2] Road surface the vehicles ``idx`` cover (length x width; trucks are wider)."""
    cfg = sim.cfg
    if sim.uniform_length:
        return idx.size * cfg.car_length * cfg.car_width
    return float(np.sum(sim.length[idx] * np.where(sim.truck[idx], TRUCK_WIDTH, cfg.car_width)))


def speed_profile(sim: RingSim, bins: int = SPACETIME_BINS, lane: int | None = None) -> np.ndarray:
    """Mean speed per angular bin (NaN where the bin is empty)."""
    idx = np.flatnonzero(sim.alive)
    if lane is not None:
        idx = idx[sim.lane[idx] == lane]
    b = (sim.theta[idx] / TWO_PI * bins).astype(int) % bins
    n = np.bincount(b, minlength=bins)
    s = np.bincount(b, weights=sim.v[idx], minlength=bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def snapshot(sim: RingSim, window: float = RATE_WINDOW) -> dict:
    """All scalar metrics at the current instant (rates over the last ``window`` s,
    at most the engine's LOG_HORIZON)."""
    window = min(window, LOG_HORIZON)
    idx = np.flatnonzero(sim.alive)
    n = idx.size
    cfg = sim.cfg
    t = sim.time
    test = getattr(sim, "subject", False)  # the capacity test's queue is left out of the queue measures
    queue = np.delete(sim.queue, SUBJECT_ARM) if test else sim.queue
    row = {
        "time": t,
        "cars": n,
        "queued": int(queue.sum()),
        "longest_queue": int(queue.max()) if queue.size else 0,
        "density": n / sim.total_lane_length * 1000.0,  # veh/km/lane
    }
    if n == 0:
        # An empty ring (open mode) has no car to average over: the per-car measures are NaN, so
        # time means skip the empty moments and charts show a gap instead of a dive to 0.
        nan = float("nan")
        row.update(mean_speed=nan, mean_free_speed=nan, speed_std=nan, flux=0.0,
                   jam_fraction=nan, speed_deficit=nan, n_jams=0, jams_following=0, jams_merge=0,
                   jams_exit=0, jams_entry=0, jams_other=0, occupancy=0.0, clustering=nan,
                   order_param=nan, order_angle=nan, order_baseline=nan, spacing_gini=nan,
                   trucks=0, truck_speed=nan, car_speed=nan)
    else:
        v = sim.v[idx]
        v_free = sim.desired_speed(idx)
        slow = jammed(sim, idx, v_free)
        # Fewer, wider sectors on small rings, so a sector holds a few cars on average.
        counts = sector_counts(sim, idx, int(np.clip(n // 4, 6, SECTORS)))
        mean_count = counts.mean()
        phase = np.exp(1j * sim.theta[idx]).mean()
        road_area = sim.total_lane_length * cfg.lane_width
        truck = sim.truck[idx]
        n_trucks = int(truck.sum())
        row.update(
            mean_speed=float(v.mean()),
            mean_free_speed=float(v_free.mean()),
            speed_std=float(v.std()),
            flux=float(np.sum(v / sim.lane_len[sim.lane[idx]])),
            jam_fraction=float(slow.mean()),
            speed_deficit=float(1.0 - v.sum() / v_free.sum()),
            n_jams=count_jams(sim, idx, slow),
            **jam_origins(sim, jam_heads(sim, idx, slow)),
            occupancy=float(covered_area(sim, idx) / road_area),
            # Spread of the sector counts beyond random (Poisson) scatter, relative
            # to the mean: 0 = even or random, ~1 = half the ring empty.
            clustering=float(np.sqrt(max(counts.var() - mean_count, 0.0)) / mean_count)
            if mean_count > 0 else 0.0,
            order_param=float(abs(phase)),
            order_angle=float(np.angle(phase) % TWO_PI),
            order_baseline=float(np.sqrt(np.pi / (4 * n))),
            spacing_gini=spacing_gini(sim),
            trucks=n_trucks,
            truck_speed=float(v[truck].mean()) if n_trucks else float("nan"),
            car_speed=float(v[~truck].mean()) if n_trucks < n else float("nan"),
        )
    span = min(t, window)
    stops = _count_since(sim.stop_log, t - window, count_at=1)
    row.update(
        detector_flux=_rolling_rate(sim.detector_log, t, window),
        exits_per_min=60.0 * _rolling_rate(sim.exit_log, t, window),
        lane_changes_per_min=60.0 * _rolling_rate(sim.lane_change_log, t, window),
        emergency_per_min=60.0 * _rolling_rate(sim.emergency_log, t, window, count_at=1),
        # Stop-and-go: how often a moving car is forced to (nearly) stop, per car
        # per minute. Unlike jam_fraction it stays ~0 in dense but smooth flow and in a standstill.
        stops_per_car_min=60.0 * stops / span / n if span > 0 and n else 0.0,
        missed_exits=sim.n_missed,
    )
    trips = [trip for when, _, trip in sim.exit_log if when >= t - window]
    row["mean_trip_time"] = float(np.mean(trips)) if trips else float("nan")
    waits = [wait for _, arm, wait in _since(sim.entry_log, t - window) if not (test and arm == SUBJECT_ARM)]
    row.update(
        arrivals_per_min=60.0 * _rolling_rate(sim.arrival_log, t, window),
        demand_per_min=60.0 * float(sim.arrival_rates().sum()) if sim.open_demand else float("nan"),
        entries_per_min=60.0 * _rolling_rate(sim.entry_log, t, window),
        blocked_per_min=60.0 * _rolling_rate(sim.blocked_log, t, window),
        mean_entry_delay=float(np.mean(waits)) if waits else float("nan"),
        entry_vc=_rolling_entry_vc(sim, window),
        throughput_per_h=3600.0 * _rolling_rate(sim.exit_log, t, window),
    )
    for lane in range(sim.n_lanes):  # (also when the ring is empty: every lane key, NaN speeds)
        in_lane = idx[sim.lane[idx] == lane]
        row[f"lane{lane}_cars"] = in_lane.size
        row[f"lane{lane}_speed"] = float(sim.v[in_lane].mean()) if in_lane.size else float("nan")
        row[f"lane{lane}_speed_std"] = float(sim.v[in_lane].std()) if in_lane.size else float("nan")
    return row


def added_keys(keys: Iterable[str]) -> dict[str, float]:
    """The ADDED_KEYS that rows with these ``keys`` lack, with their fill: each "{k}" key for every lane k the
    rows have (lane{k}_cars)."""
    keys = set(keys)
    lanes = sorted(int(key[4:-5]) for key in keys if key.startswith("lane") and key.endswith("_cars")
                   and key[4:-5].isdigit())
    wanted = {}
    for key, value in ADDED_KEYS.items():
        for name in ([key.format(k=k) for k in lanes] if "{k}" in key else [key]):
            wanted[name] = value
    return {name: value for name, value in wanted.items() if name not in keys}


class Recorder:
    """Samples :func:`snapshot`, the speed profile and sector occupancy every ``interval`` s.

    ``maxlen`` bounds memory for long interactive sessions (None = keep all).
    Call :meth:`update` after every simulation step; the recorder starts over
    by itself when the simulation is reset or rebuilt.
    """

    def __init__(self, sim: RingSim, interval: float = 0.5, maxlen: int | None = None,
                 bins: int = SPACETIME_BINS, sectors: int = SECTORS):
        self.sim = sim
        self.interval = interval
        self.bins = bins
        self.sectors = sectors
        self.maxlen = maxlen
        self.clear()

    def clear(self) -> None:
        self.rows: deque[dict] = deque(maxlen=self.maxlen)
        self.spacetime: deque[np.ndarray] = deque(maxlen=self.maxlen)
        self.occupancy: deque[np.ndarray] = deque(maxlen=self.maxlen)
        self._next = self.sim.time
        self._generation = self.sim.generation

    def update(self) -> None:
        if self.sim.generation != self._generation:  # reset / rebuilt: time restarted
            self.clear()
        if self.sim.time + 1e-9 >= self._next:
            self.record()
            while self._next <= self.sim.time + 1e-9:
                self._next += self.interval

    def record(self) -> None:
        self.rows.append(snapshot(self.sim))
        self.spacetime.append(speed_profile(self.sim, self.bins))
        self.occupancy.append(sector_occupancy(self.sim, self.sectors))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.rows))

    def spacetime_array(self) -> tuple[np.ndarray, np.ndarray]:
        """(times, speeds[time, angle bin]) for a space-time diagram."""
        times = np.array([r["time"] for r in self.rows])
        return times, (np.array(self.spacetime) if self.spacetime else np.zeros((0, self.bins)))

    def occupancy_array(self) -> np.ndarray:
        return np.array(self.occupancy) if self.occupancy else np.zeros((0, self.sectors))

    # ------------------------------------------------------------ saved state
    def state_dict(self) -> dict:
        """The samples and the sampling clock as ``{"meta": JSON-serialisable dict, "arrays": {...}}``
        (see :meth:`load_state`).

        Rows become one float array (a column per key, in the order the keys
        first appear) and a code per value: missing (a row without that key),
        float (NaN included), whole number, true/false, text or None. Text
        values go into the meta.
        """
        keys = list(dict.fromkeys(key for row in self.rows for key in row))
        col = {key: j for j, key in enumerate(keys)}
        values = np.full((len(self.rows), len(keys)), np.nan)
        kinds = np.zeros(values.shape, dtype=np.int8)
        texts = []
        for i, row in enumerate(self.rows):
            for key, value in row.items():
                j = col[key]
                if isinstance(value, (bool, np.bool_)):
                    kinds[i, j], values[i, j] = BOOL, float(value)
                elif isinstance(value, (int, np.integer)):
                    kinds[i, j], values[i, j] = INT, float(value)
                elif isinstance(value, (float, np.floating)):
                    kinds[i, j], values[i, j] = FLOAT, value
                elif value is None:
                    kinds[i, j] = NONE
                else:
                    kinds[i, j] = TEXT
                    texts.append([i, j, str(value)])
        meta = {"interval": self.interval, "maxlen": self.maxlen, "bins": self.bins, "sectors": self.sectors,
                "next": self._next, "generation": self._generation, "keys": keys, "texts": texts}
        arrays = {"values": values, "kinds": kinds,
                  "spacetime": np.array(self.spacetime).reshape(len(self.spacetime), self.bins),
                  "occupancy": np.array(self.occupancy).reshape(len(self.occupancy), self.sectors)}
        return {"meta": meta, "arrays": arrays}

    def load_state(self, state: dict, interval: float | None = None, numbers: Iterable[str] = ()) -> None:
        """Carry on from a :meth:`state_dict`: the same rows (keys, values and their Python types), the
        space-time and occupancy samples and the sampling clock, interval included. Keeps this
        recorder's ``maxlen`` (the newest samples if there are more).

        Every row must hold a finite "time", and each of ``numbers`` as a whole or decimal number
        (NaN allowed, not infinite): a program that reads rows by name, as the viewer does, passes
        the keys of :func:`snapshot`. Keys added since the samples were saved (ADDED_KEYS) that no
        saved row has are filled in first. ``interval`` (if given) is the sampling interval the
        samples must have been taken at. Raises :class:`StateError` (and changes nothing) if
        ``state`` is damaged, its clock does not fit this recorder's simulation, or it was recorded
        with other bins, sectors or interval.
        """
        try:
            meta, arrays = state["meta"], state["arrays"]
            if (meta["bins"], meta["sectors"]) != (self.bins, self.sectors):
                raise StateError(f"charts of {meta['bins']} angle bins and {meta['sectors']} sectors; "
                                 f"these have {self.bins} and {self.sectors}")
            if interval is not None and not math.isclose(float(meta["interval"]), interval):
                raise StateError(f"charts sampled every {float(meta['interval']):g} s; these sample every {interval:g} s")
            keys = [str(key) for key in meta["keys"]]
            values = np.asarray(arrays["values"], dtype=float)
            kinds = np.asarray(arrays["kinds"])
            n = values.shape[0] if values.ndim == 2 else -1
            spacetime, occupancy = np.asarray(arrays["spacetime"], dtype=float), np.asarray(arrays["occupancy"], dtype=float)
            if values.shape != (n, len(keys)) or kinds.shape != values.shape or kinds.dtype.kind not in "iu" \
                    or spacetime.shape != (n, self.bins) or occupancy.shape != (n, self.sectors):
                raise StateError("the chart samples do not fit together")
            fills = added_keys(keys)  # keys no saved row has yet: as floats, the same in every row
            if fills:
                keys = keys + list(fills)
                values = np.hstack((values, np.tile(np.array(list(fills.values()), dtype=float), (n, 1))))
                kinds = np.hstack((kinds, np.full((n, len(fills)), FLOAT, dtype=kinds.dtype)))
            for key in dict.fromkeys(("time", *numbers)):  # (in order, once each)
                j = keys.index(key) if key in keys else None
                if n and (j is None or not np.isin(kinds[:, j], (FLOAT, INT)).all()):
                    raise StateError(f"not every chart sample holds {key} as a number")
                if n and (np.isinf(values[:, j]).any() or key == "time" and np.isnan(values[:, j]).any()):
                    raise StateError(f"the chart samples hold an infinite or undefined {key}")
            texts = {(int(i), int(j)): str(text) for i, j, text in meta["texts"]}
            columns = [(key, values[:, j].tolist(), kinds[:, j].tolist()) for j, key in enumerate(keys)]
            rows = [{} for _ in range(n)]
            for j, (key, column, codes) in enumerate(columns):
                for i, (value, code) in enumerate(zip(column, codes)):
                    if code == FLOAT:
                        rows[i][key] = value
                    elif code == INT:
                        rows[i][key] = int(value)
                    elif code == BOOL:
                        rows[i][key] = bool(value)
                    elif code == TEXT:
                        rows[i][key] = texts[i, j]
                    elif code == NONE:
                        rows[i][key] = None
                    elif code != MISSING:
                        raise StateError(f"unknown value code {code} in the chart samples")
            saved_interval, upcoming, generation = float(meta["interval"]), float(meta["next"]), int(meta["generation"])
        except StateError:
            raise
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as e:
            raise StateError(f"the chart samples are damaged ({e})") from e
        if not (math.isfinite(saved_interval) and saved_interval >= MIN_INTERVAL):
            raise StateError(f"the chart samples' sampling interval ({saved_interval:g} s) is out of range")
        # A recorder kept up to date (update after every step) has its next sample due within an
        # interval of the time; allow it to lag a while, but not so far that catching up never ends.
        # (Another generation than the simulation's: the next update starts the charts afresh anyway.)
        t = self.sim.time
        if not math.isfinite(upcoming) or generation == self.sim.generation and not \
                t - MAX_CATCH_UP * saved_interval <= upcoming <= t + saved_interval + 1e-6:
            raise StateError(f"the chart samples' clock (next sample at {upcoming:g} s) does not fit "
                             f"the simulation's time ({t:g} s)")
        self.rows = deque(rows, maxlen=self.maxlen)
        self.spacetime = deque((profile.copy() for profile in spacetime), maxlen=self.maxlen)
        self.occupancy = deque((sectors.copy() for sectors in occupancy), maxlen=self.maxlen)
        self.interval, self._next, self._generation = saved_interval, upcoming, generation
