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
    model asking for more than MAX_DECEL (over-reaction after a late or
    delayed response), or the no-overlap constraint stopping a car harder.

Space ("do cars congregate on one side?")
  * ``occupancy`` - share of the road surface covered by cars.
  * ``clustering`` - spread of the car count over equal sectors of the ring
    (6-24, about 4+ cars each), beyond random scatter, relative to the mean:
    0 = even or random, ~1 = half the ring empty. Comparable across densities.
  * ``order_param`` - |mean of e^(i*theta)| over all cars: 0 = balanced around
    the ring, 1 = all on one side; ``order_angle`` is where they bunch up.
    ``order_baseline`` = sqrt(pi / 4N) is what N randomly placed cars give,
    so only values well above it mean real one-sidedness.
  * ``spacing_gini`` - Gini coefficient of the space each car has (gap to the
    car ahead + its length): 0 = road shared evenly, ~0.35 = random gaps,
    towards 1 = packed queues with empty road between (platoons or jams).

:class:`Recorder` samples these at a fixed interval, together with a
space-time diagram (mean speed per angular bin over time) and per-sector
occupancy, for the live charts and the experiment plots.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from roundabout_sim.engine import LOG_HORIZON, SLOW_FRACTION, TWO_PI, RingSim

SECTORS = 24  # sectors for the clustering index and the occupancy ring
SPACETIME_BINS = 180  # 2-degree bins for the space-time diagram


RATE_WINDOW = 30.0  # [s] default window for rolling rates


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
    one with the most room ahead of it.
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
        in_lane = lanes == lane
        th = sim.theta[idx[order][in_lane]]
        room = (np.roll(th, -1) - th) % TWO_PI
        heads.append(idx[order][in_lane][[int(np.argmax(room))]])
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
    """Gini coefficient of the road length each car has (bumper gap ahead + car length)."""
    _, gap = sim.bumper_gaps()
    if gap.size < 2:
        return 0.0
    pitch = np.sort(gap + sim.cfg.car_length)
    n = pitch.size
    return float(2 * np.sum(np.arange(1, n + 1) * pitch) / (n * pitch.sum()) - (n + 1) / n)


def sector_counts(sim: RingSim, idx: np.ndarray, sectors: int = SECTORS) -> np.ndarray:
    bins = (sim.theta[idx] / TWO_PI * sectors).astype(int) % sectors
    return np.bincount(bins, minlength=sectors)


def sector_occupancy(sim: RingSim, sectors: int = SECTORS) -> np.ndarray:
    """Fraction of each sector's lane length covered by cars (all lanes together)."""
    idx = np.flatnonzero(sim.alive)
    counts = sector_counts(sim, idx, sectors)
    sector_len = sim.total_lane_length / sectors
    return counts * sim.cfg.car_length / sector_len


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
    row = {
        "time": t,
        "cars": n,
        "queued": int(sim.queue.sum()),
        "density": n / sim.total_lane_length * 1000.0,  # veh/km/lane
    }
    if n == 0:
        row.update(mean_speed=0.0, mean_free_speed=float("nan"), speed_std=0.0, flux=0.0,
                   jam_fraction=0.0, speed_deficit=0.0, n_jams=0, jams_following=0, jams_merge=0,
                   jams_exit=0, jams_entry=0, jams_other=0, occupancy=0.0, clustering=0.0,
                   order_param=0.0, order_angle=0.0, order_baseline=float("nan"), spacing_gini=0.0)
    else:
        v = sim.v[idx]
        v_free = sim.desired_speed(idx)
        slow = jammed(sim, idx, v_free)
        # Fewer, wider sectors on small rings, so a sector holds a few cars on average.
        counts = sector_counts(sim, idx, int(np.clip(n // 4, 6, SECTORS)))
        mean_count = counts.mean()
        phase = np.exp(1j * sim.theta[idx]).mean()
        road_area = sim.total_lane_length * cfg.lane_width
        row.update(
            mean_speed=float(v.mean()),
            mean_free_speed=float(v_free.mean()),
            speed_std=float(v.std()),
            flux=float(np.sum(v / sim.lane_len[sim.lane[idx]])),
            jam_fraction=float(slow.mean()),
            speed_deficit=float(1.0 - v.sum() / v_free.sum()),
            n_jams=count_jams(sim, idx, slow),
            **jam_origins(sim, jam_heads(sim, idx, slow)),
            occupancy=float(n * cfg.car_length * cfg.car_width / road_area),
            # Spread of the sector counts beyond random (Poisson) scatter, relative
            # to the mean: 0 = even or random, ~1 = half the ring empty.
            clustering=float(np.sqrt(max(counts.var() - mean_count, 0.0)) / mean_count)
            if mean_count > 0 else 0.0,
            order_param=float(abs(phase)),
            order_angle=float(np.angle(phase) % TWO_PI),
            order_baseline=float(np.sqrt(np.pi / (4 * n))),
            spacing_gini=spacing_gini(sim),
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
    for lane in range(sim.n_lanes):
        in_lane = idx[sim.lane[idx] == lane]
        row[f"lane{lane}_cars"] = in_lane.size
        row[f"lane{lane}_speed"] = float(sim.v[in_lane].mean()) if in_lane.size else float("nan")
    return row


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
