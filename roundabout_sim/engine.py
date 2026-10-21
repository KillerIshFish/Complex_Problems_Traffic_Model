"""Vectorised multi-lane roundabout (ring road) traffic engine.

Cars drive anticlockwise on concentric circular lanes. A car's state is its
lane, its angle ``theta`` on the ring (radians, increasing in the driving
direction) and its speed. Every step:

1. **Car following** - each driver's acceleration comes from the Intelligent
   Driver Model (IDM) towards the cars ahead in its lane. Drivers react to
   every car within ``reaction_distance`` (multi-anticipation: seeing further
   ahead damps waves; too short and they notice the car in front too late),
   act on what they perceived ``reaction_time`` ago, and add a little random
   noise. A hard constraint stops anyone driving into the car ahead.
2. **Exits** - a car whose route ends at an exit leaves if it reached it on
   the outer lane; otherwise it misses the exit and goes round again. Every
   leaver is replaced by a car queued at a random entrance.
3. **Lane changes** - leavers move outwards in time for their exit; other
   drivers consider a change ``lane_change_rate`` times per second and make it
   if it is safe (and, in "advantage" mode, faster - the MOBIL model).
4. **Entries** - queued cars merge into the outer lane when the gap to
   circulating traffic is at least ``entry_gap`` seconds. At most
   ``MAX_QUEUE`` cars wait per entrance; beyond that a leaver's replacement
   is slotted straight into the ring, so the ring never drains into queues.

Density is controlled as cars on the ring per km of lane: raising it inserts
cars into the largest gaps, lowering it removes random cars.
"""

from __future__ import annotations

import heapq
from collections import deque

import numpy as np

from roundabout_sim.config import PARAM_BY_NAME, SimConfig

TWO_PI = 2.0 * np.pi
LC_ZONE = 50.0  # [m] of ring a leaver needs per lane it must cross to reach the outer lane
LC_COOLDOWN = 2.0  # [s] minimum time between two lane changes of one driver
LC_THRESHOLD = 0.2  # [m/s2] acceleration gain needed for an "advantage" lane change
LC_DURATION = 1.5  # [s] how long a lane change takes on screen (rendering only)
NOISE_TAU = 2.0  # [s] correlation time of the acceleration noise
SAFE_MARGIN = 0.3  # [m] bumper gap enforced by the no-overlap constraint
MAX_REACTION = 2.0  # [s] longest supported reaction time
ENTRY_COOLDOWN = 1.0  # [s] between two cars entering from the same arm
PERTURB_DECEL = 4.0  # [m/s2] braking of a car the user tells to brake
MAX_DECEL = 9.0  # [m/s2] physical braking limit; needing more counts as harsh braking
SLOW_FRACTION = 0.3  # a car counts as jammed below this fraction of its free speed
RESUME_FRACTION = 0.6  # ...and as moving again above this (hysteresis for counting stops)
MAX_QUEUE = 2  # cars waiting per entrance
LOG_HORIZON = 600.0  # [s] how long event logs are kept for rolling rates
EMERGENCY_HOLD = 2.0  # [s] an emergency episode ends after this long without one
LEAVER_PATIENCE = 5.0  # [s] a leaver held at a standstill by merging gives up its exit
SMALL_FILL = 3  # deficits up to this many cars are slotted into random gaps
ANTICIPATION = 5  # most cars ahead a driver reacts to (within the reaction distance)


class RingSim:
    """Agent-based traffic on a multi-lane roundabout.

    Per-car state lives in fixed-capacity numpy arrays indexed by slot; only
    slots with ``alive[i]`` are cars on the ring. Use :meth:`cars` for a
    snapshot of the live cars instead of reading the arrays directly.
    """

    def __init__(self, config: SimConfig | None = None, **overrides):
        self.cfg = (config or SimConfig()).updated(**overrides)
        self.reset()

    # ------------------------------------------------------------- set-up
    def reset(self, seed: int | None = None, **changes) -> None:
        """Rebuild the roundabout and place the initial cars."""
        if changes:
            self.cfg = self.cfg.updated(**changes)
        cfg = self.cfg
        if seed is None:
            seed = cfg.seed if cfg.seed is not None else int(np.random.SeedSequence().entropy % 2**31)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.time = 0.0
        self.steps = 0
        self.generation = getattr(self, "generation", -1) + 1  # lets observers notice rebuilds

        self.n_lanes = int(cfg.n_lanes)
        self.outer = self.n_lanes - 1
        self.radii = cfg.radius + cfg.lane_width * np.arange(self.n_lanes)
        self.lane_len = TWO_PI * self.radii
        self.r_out = float(self.radii[-1])
        n = int(cfg.n_arms)
        self.n_arms = n
        if n:
            self.arm_angle = (-np.pi / 2 + TWO_PI * np.arange(n) / n) % TWO_PI
            half = min(0.35 * TWO_PI / n, 1.2 * cfg.lane_width / self.r_out)
            self.exit_angle = (self.arm_angle - half) % TWO_PI  # exit comes first...
            self.entry_angle = (self.arm_angle + half) % TWO_PI  # ...then the entry
            self.detector_angle = float((-np.pi / 2 + np.pi / n) % TWO_PI)  # between arms
        else:
            self.arm_angle = self.exit_angle = self.entry_angle = np.zeros(0)
            self.detector_angle = 0.0
        self.queue = np.zeros(n, dtype=int)  # cars waiting at each entrance
        self.entry_cool = np.zeros(n)

        self.n_entered = self.n_exited = self.n_missed = 0
        self.n_lane_changes = self.n_emergency = self.n_stops = 0
        # Event logs for rolling rates, oldest first, pruned after LOG_HORIZON.
        self.detector_log: deque[tuple[float, int]] = deque()  # (time, lane) crossings
        self.exit_log: deque[tuple[float, int, float]] = deque()  # (time, arm, trip time)
        self.lane_change_log: deque[float] = deque()  # time
        self.emergency_log: deque[tuple[float, int]] = deque()  # (time, new emergencies)
        self.stop_log: deque[tuple[float, int]] = deque()  # (time, cars that stopped)
        self._next_uid = 0
        self._next_fill = 0.0
        self._next_prune = 0.0
        self._lanes: list[tuple[np.ndarray, np.ndarray]] = [(np.zeros(0, dtype=np.int64), np.zeros(0))] * self.n_lanes

        self._hist_len = int(round(MAX_REACTION / cfg.dt)) + 1
        self._hist_ptr = 0
        self._alloc(max(64, 2 * self.capacity()))
        self._place_initial(self.target_cars())

    def set(self, **changes) -> bool:
        """Change parameters live. Returns True if the change rebuilt the roundabout."""
        if not changes:
            return False
        old, old_seed = self.cfg, self.seed
        new_cfg = old.updated(**changes)  # validates names and values
        # Geometry/numerics fields without a slider (lane_width, dt, ...) also rebuild.
        rebuild = any((PARAM_BY_NAME[k].reset if k in PARAM_BY_NAME else k != "seed")
                      and getattr(old, k) != v for k, v in changes.items())
        self.cfg = new_cfg
        if rebuild:
            try:
                self.reset(seed=new_cfg.seed if "seed" in changes else self.seed)
            except Exception:
                self.cfg = old
                self.reset(seed=old_seed)
                raise
            return True
        if "seed" in changes and new_cfg.seed is not None:  # applies from now on
            self.seed = int(new_cfg.seed)
            self.rng = np.random.default_rng(self.seed)
        alive = np.flatnonzero(self.alive)
        if new_cfg.exit_prob != old.exit_prob:
            self._assign_routes(alive)  # geometric routes are memoryless
        if any(getattr(new_cfg, k) != getattr(old, k) for k in
               ("mean_speed", "speed_std", "lateral_accel", "turn_speed", "exit_prob", "comfort_decel")):
            # New free speeds: re-baseline who counts as moving, so the change
            # itself is not logged as a wave of stops.
            self.moving[alive] &= self.v[alive] >= SLOW_FRACTION * self.target_speed(alive)
        return False

    def _alloc(self, cap: int) -> None:
        self.cap = cap
        self.alive = np.zeros(cap, dtype=bool)
        self.lane = np.zeros(cap, dtype=np.int64)
        self.theta = np.zeros(cap)  # angle on the ring in [0, 2pi)
        self.travel = np.zeros(cap)  # unwrapped angle driven since the car appeared
        self.v = np.zeros(cap)
        self.z = np.zeros(cap)  # driver temperament ~ N(0, 1): desired speed = mean + z * std
        self.exit_at = np.full(cap, np.inf)  # value of ``travel`` at which the car's exit is
        self.born = np.zeros(cap)
        self.lat = np.zeros(cap)  # drawn lateral position in lanes (animates lane changes)
        self.lc_cool = np.zeros(cap)
        self.noise = np.zeros(cap)
        self.brake_until = np.full(cap, -np.inf)
        self.uid = np.full(cap, -1, dtype=np.int64)
        self.moving = np.zeros(cap, dtype=bool)  # above RESUME_FRACTION since its last stop
        self.emergency_until = np.full(cap, -np.inf)  # end of the current emergency episode
        self.stuck = np.zeros(cap)  # [s] a leaver has been held at a standstill by merging
        self.entered_at = np.full(cap, -np.inf)  # time the car came in from an entrance
        self.inserted_at = np.full(cap, -np.inf)  # time it was slotted in by the density control
        self.acc_hist = np.zeros((self._hist_len, cap))

    def _grow(self, need: int) -> None:
        old = {name: getattr(self, name) for name in (
            "alive", "lane", "theta", "travel", "v", "z", "exit_at", "born", "lat",
            "lc_cool", "noise", "brake_until", "uid", "moving", "emergency_until", "stuck",
            "entered_at", "inserted_at")}
        hist, old_cap = self.acc_hist, self.cap
        self._alloc(max(2 * old_cap, old_cap + need))
        for name, arr in old.items():
            getattr(self, name)[:old_cap] = arr
        self.acc_hist[:, :old_cap] = hist

    # ----------------------------------------------------------- geometry
    @property
    def total_lane_length(self) -> float:
        return float(self.lane_len.sum())

    @property
    def jam_density(self) -> float:
        """Densest packing [veh/km/lane]: bumper to bumper at the standstill gap."""
        return 1000.0 / (self.cfg.car_length + self.cfg.min_gap)

    def capacity(self) -> int:
        pitch = self.cfg.car_length + self.cfg.min_gap
        return int(np.floor(self.lane_len / pitch).sum())

    def target_cars(self) -> int:
        density = min(self.cfg.density, self.jam_density)
        return min(int(round(density * self.total_lane_length / 1000.0)), self.capacity())

    def lane_radius(self, lat) -> np.ndarray:
        return self.cfg.radius + self.cfg.lane_width * np.asarray(lat)

    # ------------------------------------------------------- driver model
    def desired_speed(self, idx, lane=None) -> np.ndarray:
        """Free-road desired speed, capped by comfortable cornering on the car's lane
        (or on ``lane`` if given).

        Floored at half the mean: a huge spread would otherwise create drivers
        who barely move, i.e. moving bottlenecks rather than phantom jams.
        """
        cfg = self.cfg
        v0 = np.maximum(0.5 * cfg.mean_speed, cfg.mean_speed + self.z[idx] * cfg.speed_std)
        lanes = self.lane[idx] if lane is None else lane
        return np.minimum(v0, np.sqrt(cfg.lateral_accel * self.radii)[lanes])

    def target_speed(self, idx) -> np.ndarray:
        """Desired speed, also slowing down comfortably to ``turn_speed`` before an exit.

        Only cars already on the outer lane slow for their exit: a leaver still
        on an inner lane keeps pace with traffic so it can merge outwards.
        """
        cfg = self.cfg
        v = self.desired_speed(idx)
        to_exit = (self.exit_at[idx] - self.travel[idx]) * self.r_out
        approach = np.sqrt(cfg.turn_speed**2 + 2 * cfg.comfort_decel * np.maximum(to_exit - 5.0, 0.0))
        slowing = np.isfinite(to_exit) & (self.lane[idx] == self.outer)
        return np.where(slowing, np.minimum(v, approach), v)

    def _free_acc(self, v, v0) -> np.ndarray:
        """IDM free-road acceleration towards ``v0``. Above the desired speed it
        brakes gently (improved IDM) instead of the plain IDM's hard (v/v0)^4."""
        cfg = self.cfg
        a, b = cfg.max_accel, cfg.comfort_decel
        v = np.asarray(v, dtype=float)
        v0 = np.maximum(np.asarray(v0, dtype=float), 0.1)
        free = a * (1 - (v / v0) ** 4)
        over = v > v0
        if over.any():
            free[over] = -b * (1 - (v0[over] / v[over]) ** (4 * a / b))
        return free

    def _interaction(self, v, gap, v_lead) -> np.ndarray:
        """IDM braking towards a leader ``gap`` metres (bumper to bumper) ahead."""
        cfg = self.cfg
        a, b = cfg.max_accel, cfg.comfort_decel
        s_star = cfg.min_gap + np.maximum(0.0, v * cfg.time_headway + v * (v - v_lead) / (2 * np.sqrt(a * b)))
        return a * (s_star / np.maximum(gap, 0.1)) ** 2

    def _idm(self, v, v0, gap, v_lead) -> np.ndarray:
        """IDM acceleration towards one leader. ``gap`` is bumper to bumper;
        leaders beyond the reaction distance (or ``inf``) are ignored."""
        v = np.asarray(v, dtype=float)
        seen = np.isfinite(gap) & (gap <= self.cfg.reaction_distance)
        return self._free_acc(v, v0) - np.where(seen, self._interaction(v, np.where(seen, gap, 1.0), v_lead), 0.0)

    def _car_following(self, idx, v, v0, ld, g, lead, gap) -> np.ndarray:
        """Car-following acceleration with multi-anticipation (as in Treiber's
        Human Driver Model): a driver reacts to every car ahead in its lane
        within the reaction distance (up to ANTICIPATION cars), not just the
        next one. The j-th car ahead is judged by the average gap per car to it
        and weighted 1/j^2; the weighted mean of these IDM braking terms
        replaces the single-leader term. In even traffic this changes nothing,
        but drivers who see further brake earlier and more gently for a slowdown
        a few cars ahead, which damps stop-and-go waves. Arrays are aligned with
        ``idx``; ``ld``/``g`` are each car's leader slot and bumper gap, and
        ``lead``/``gap`` the full-size arrays from :meth:`_index_lanes`.
        """
        cfg = self.cfg
        seen = np.isfinite(g) & (g <= cfg.reaction_distance)
        v_lead = np.where(ld >= 0, self.v[np.maximum(ld, 0)], v)
        braking = np.where(seen, self._interaction(v, np.where(seen, g, 1.0), v_lead), 0.0)
        weight = seen.astype(float)
        cars_ahead = np.bincount(self.lane[idx], minlength=self.n_lanes)[self.lane[idx]] - 1
        car, gaps = np.maximum(ld, 0), g.copy()
        for j in range(2, ANTICIPATION + 1):
            gaps = gaps + gap[car]  # summed bumper gaps to the j-th car ahead
            car = lead[car]
            sees = seen & (j <= cars_ahead) & (gaps + (j - 1) * cfg.car_length <= cfg.reaction_distance)
            if not sees.any():
                break
            w = 1.0 / j**2
            braking += np.where(sees, w * self._interaction(v, gaps / j, self.v[car]), 0.0)
            weight += np.where(sees, w, 0.0)
        return self._free_acc(v, v0) - np.where(weight > 0, braking / np.maximum(weight, 1e-12), 0.0)

    def equilibrium_speed(self, gap, v0) -> np.ndarray:
        """Steady-state IDM speed for a bumper gap (used to start cars smoothly)."""
        cfg = self.cfg
        gap = np.asarray(gap, dtype=float)
        v0 = np.maximum(np.broadcast_to(np.asarray(v0, dtype=float), gap.shape), 0.1)
        lo, hi = np.zeros_like(gap), v0.copy()
        seen = gap <= cfg.reaction_distance
        for _ in range(40):  # bisection on the IDM steady state (acceleration = 0)
            mid = 0.5 * (lo + hi)
            f = 1 - (mid / v0) ** 4 \
                - np.where(seen, ((cfg.min_gap + mid * cfg.time_headway) / np.maximum(gap, 0.1)) ** 2, 0)
            lo, hi = np.where(f > 0, mid, lo), np.where(f > 0, hi, mid)
        return np.where(gap > cfg.min_gap, lo, 0.0)

    # ------------------------------------------------------ neighbourhood
    def _index_lanes(self, idx):
        """Sort live cars by angle within each lane.

        Returns full-size arrays ``lead`` (slot of the car ahead, -1 if none)
        and ``gap`` (bumper gap to it [m]); a car alone in its lane follows
        itself round the ring. Also caches each lane's sorted slots/angles.
        """
        lead = np.full(self.cap, -1, dtype=np.int64)
        gap = np.full(self.cap, np.inf)
        # Stable sorts by angle, then by lane: grouped by lane, by angle within it.
        order = idx[np.argsort(self.theta[idx], kind="stable")]
        order = order[np.argsort(self.lane[order].astype(np.int16), kind="stable")]  # radix sort
        lanes, th = self.lane[order], self.theta[order]
        bounds = np.searchsorted(lanes, np.arange(self.n_lanes + 1))
        filled = bounds[1:] > bounds[:-1]
        nxt_pos = np.arange(1, order.size + 1)
        nxt_pos[bounds[1:][filled] - 1] = bounds[:-1][filled]  # the last car in a lane follows the first
        nxt = order[nxt_pos]
        dth = th[nxt_pos] - th
        dth[dth < 0] += TWO_PI  # == % TWO_PI on (-2pi, 2pi), but 3x cheaper
        dth[nxt == order] = TWO_PI
        lead[order] = nxt
        gap[order] = dth * self.radii[lanes] - self.cfg.car_length
        self._lanes = [(order[a:b], th[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]
        return lead, gap

    def bumper_gaps(self) -> tuple[np.ndarray, np.ndarray]:
        """(slots of the live cars, bumper gap from each to the car ahead in its lane [m])."""
        idx = np.flatnonzero(self.alive)
        _, gap = self._index_lanes(idx)
        return idx, gap[idx]

    def _neighbours_in(self, lane: int, angles: np.ndarray):
        """Leader/follower slots and bumper gaps for positions ``angles`` in ``lane``."""
        order, th = self._lanes[lane]
        n = angles.size
        if order.size == 0:
            return (np.full(n, -1), np.full(n, np.inf), np.full(n, -1), np.full(n, np.inf))
        pos = np.searchsorted(th, angles)
        ld = order[pos % order.size]
        fl = order[(pos - 1) % order.size]
        r, L = self.radii[lane], self.cfg.car_length
        front = ((self.theta[ld] - angles) % TWO_PI) * r - L
        back = ((angles - self.theta[fl]) % TWO_PI) * r - L
        return ld, front, fl, back

    def _neighbours(self, lanes: np.ndarray, angles: np.ndarray):
        """:meth:`_neighbours_in` for positions spread over several lanes."""
        n = angles.size
        out = (np.full(n, -1, dtype=np.int64), np.full(n, np.inf), np.full(n, -1, dtype=np.int64), np.full(n, np.inf))
        for lane in range(self.n_lanes):
            m = lanes == lane
            if m.any():
                for arr, part in zip(out, self._neighbours_in(lane, angles[m])):
                    arr[m] = part
        return out

    # ----------------------------------------------------------- spawning
    def _spawn(self, lanes, angles, speeds, lat=None) -> np.ndarray:
        k = len(angles)
        free = np.flatnonzero(~self.alive)
        if free.size < k:
            self._grow(k - free.size)
            free = np.flatnonzero(~self.alive)
        i = free[:k]
        self.alive[i] = True
        self.lane[i] = lanes
        self.theta[i] = np.asarray(angles) % TWO_PI
        self.travel[i] = 0.0
        self.v[i] = speeds
        self.z[i] = np.clip(self.rng.standard_normal(k), -2.5, 2.5)
        self.born[i] = self.time
        self.lat[i] = lanes if lat is None else lat
        self.lc_cool[i] = self.rng.uniform(0, LC_COOLDOWN, k)
        self.noise[i] = 0.0
        self.brake_until[i] = -np.inf
        self.uid[i] = np.arange(self._next_uid, self._next_uid + k)
        self._next_uid += k
        self.moving[i] = False
        self.emergency_until[i] = -np.inf
        self.stuck[i] = 0.0
        self.entered_at[i] = -np.inf
        self.inserted_at[i] = -np.inf
        self.acc_hist[:, i] = 0.0
        self._assign_routes(i)
        return i

    def _assign_routes(self, i) -> None:
        """Draw each car's exit: it leaves at each exit it passes with chance ``exit_prob``."""
        self.exit_at[i] = np.inf
        p = self.cfg.exit_prob
        if self.n_arms == 0 or p <= 0 or len(i) == 0:
            return
        k = self.rng.geometric(min(p, 1.0), size=len(i))  # exits passed, >= 1
        ahead = (self.exit_angle[None, :] - self.theta[i][:, None]) % TWO_PI
        ahead = np.where(ahead < 1e-6, ahead + TWO_PI, ahead)
        ahead.sort(axis=1)
        laps, nth = np.divmod(k - 1, self.n_arms)
        self.exit_at[i] = self.travel[i] + ahead[np.arange(len(i)), nth] + TWO_PI * laps

    def _place_initial(self, n: int) -> None:
        if n <= 0:
            return
        cfg = self.cfg
        pitch = cfg.car_length + cfg.min_gap
        share = n * self.lane_len / self.lane_len.sum()
        counts = np.floor(share).astype(int)
        counts[np.argsort(counts - share)[: n - counts.sum()]] += 1
        counts = np.minimum(counts, np.floor(self.lane_len / pitch).astype(int))
        # A clustered start is one packed block across all lanes, so it starts
        # at the same angle in every lane.
        start = self.rng.uniform(0, TWO_PI) if cfg.init_spacing == "clustered" else 0.0
        for lane, k in enumerate(counts):
            if k == 0:
                continue
            C, r = self.lane_len[lane], self.radii[lane]
            if cfg.init_spacing == "random":
                offset = self.rng.uniform(0, C)
                gaps = pitch + (C - k * pitch) * self.rng.dirichlet(np.ones(k))
                s = offset + np.concatenate(([0.0], np.cumsum(gaps[:-1])))
            elif cfg.init_spacing == "clustered":
                s = start * r + min(pitch * 1.05, C / k) * np.arange(k)
            else:  # uniform
                s = self.rng.uniform(0, C) + C * np.arange(k) / k
            self._spawn(np.full(k, lane), s / r, np.zeros(k))
        idx = np.flatnonzero(self.alive)
        _, gap = self._index_lanes(idx)
        self.v[idx] = self.equilibrium_speed(gap[idx], self.desired_speed(idx))

    # ------------------------------------------------------- user actions
    def brake(self, slot: int, duration: float = 3.0) -> None:
        """Make one driver brake hard for ``duration`` seconds (a perturbation)."""
        if 0 <= slot < self.cap and self.alive[slot]:
            self.brake_until[slot] = self.time + duration

    def brake_random(self, duration: float = 3.0) -> int | None:
        idx = np.flatnonzero(self.alive)
        if idx.size == 0:
            return None
        slot = int(self.rng.choice(idx))
        self.brake(slot, duration)
        return slot

    def car_near(self, x: float, y: float, max_dist: float = 4.0) -> int | None:
        """Slot of the live car closest to world point (x, y), if within ``max_dist``."""
        idx = np.flatnonzero(self.alive)
        if idx.size == 0:
            return None
        cx, cy = self.positions(idx)
        d = np.hypot(cx - x, cy - y)
        k = int(np.argmin(d))
        return int(idx[k]) if d[k] <= max_dist else None

    # -------------------------------------------------------------- views
    def positions(self, idx):
        r = self.lane_radius(self.lat[idx])
        return r * np.cos(self.theta[idx]), r * np.sin(self.theta[idx])

    def cars(self) -> dict[str, np.ndarray]:
        """Snapshot of the live cars (all arrays aligned)."""
        idx = np.flatnonzero(self.alive)
        x, y = self.positions(idx)
        to_exit = (self.exit_at[idx] - self.travel[idx]) * self.r_out
        # Angle cars across while a lane change is being animated (outwards = clockwise tilt).
        drift = np.clip(self.lane[idx] - self.lat[idx], -1, 1) * self.cfg.lane_width / LC_DURATION
        tilt = np.arctan2(np.where(np.abs(self.lane[idx] - self.lat[idx]) > 1e-6, drift, 0.0),
                          np.maximum(self.v[idx], 1.0))
        return {
            "slot": idx,
            "uid": self.uid[idx],
            "lane": self.lane[idx],
            "lat": self.lat[idx],
            "theta": self.theta[idx],
            "x": x,
            "y": y,
            "heading": self.theta[idx] + np.pi / 2 - tilt,
            "v": self.v[idx],
            "v0": self.desired_speed(idx),
            "z": self.z[idx],
            "leaving": to_exit < 0.5 * TWO_PI * self.r_out,
            "braking": self.brake_until[idx] > self.time,
        }

    # ------------------------------------------------------------ dynamics
    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            self._step()

    def _step(self) -> None:
        dt = self.cfg.dt
        idx = np.flatnonzero(self.alive)
        if idx.size:
            self._move(idx)
            self._exits()
            self._lane_changes(np.flatnonzero(self.alive))
        self._entries()
        # Last, so that after every step the population matches the density
        # setting (density changes, and leavers that found their queue full).
        self._sync_population()
        # Whole arrays: cheaper than masking, and _spawn resets free slots.
        self.lc_cool -= dt
        self.entry_cool -= dt
        self.lat += np.clip(self.lane - self.lat, -dt / LC_DURATION, dt / LC_DURATION)
        self.time += dt
        self.steps += 1
        if self.time >= self._next_prune:
            self._prune_logs()

    def _prune_logs(self) -> None:
        """Drop events older than LOG_HORIZON (once per simulated second)."""
        self._next_prune = self.time + 1.0
        cutoff = self.time - LOG_HORIZON
        for log in (self.detector_log, self.exit_log, self.emergency_log, self.stop_log):
            while log and log[0][0] < cutoff:
                log.popleft()
        while self.lane_change_log and self.lane_change_log[0] < cutoff:
            self.lane_change_log.popleft()

    def _move(self, idx) -> None:
        cfg, dt = self.cfg, self.cfg.dt
        lead, gap = self._index_lanes(idx)
        v = self.v[idx]
        ld = lead[idx]
        g = gap[idx]
        v0 = self.target_speed(idx)
        acc_now = self._car_following(idx, v, v0, ld, g, lead, gap)
        self._make_room_to_leave(idx, v, v0, acc_now)
        # The driver model asking for more than the brakes can give: harsh
        # braking, typically over-reaction after a late or delayed response.
        over_braking = acc_now < -MAX_DECEL

        # Reaction time: act on the acceleration decided ``delay`` steps ago.
        delay = int(round(min(cfg.reaction_time, MAX_REACTION) / dt))
        self._hist_ptr = (self._hist_ptr + 1) % self._hist_len
        self.acc_hist[self._hist_ptr, idx] = acc_now
        acc = self.acc_hist[(self._hist_ptr - delay) % self._hist_len, idx].copy()

        braking = self.brake_until[idx] > self.time
        if cfg.driver_noise > 0:
            f = np.exp(-dt / NOISE_TAU)
            self.noise[idx] = f * self.noise[idx] + cfg.driver_noise * np.sqrt(1 - f * f) * \
                self.rng.standard_normal(idx.size)
            acc += self.noise[idx]
        acc = np.where(braking, -PERTURB_DECEL, np.maximum(acc, -MAX_DECEL))  # physical braking limit

        v_new = v + acc * dt
        stops = v_new < 0
        ds = np.where(stops, v * v / (2 * np.maximum(-acc, 1e-9)), v * dt + 0.5 * acc * dt * dt)
        ds = np.maximum(ds, 0.0)
        v_new = np.maximum(v_new, 0.0)

        # No overlaps: nobody may end the step closer than SAFE_MARGIN behind
        # where its leader ends the step. Iterate until nothing shrinks: moves
        # only decrease and are bounded by 0, so this settles chains of any length.
        planned = ds
        pos = np.empty(self.cap, dtype=np.int64)
        pos[idx] = np.arange(idx.size)
        ahead = pos[ld]  # every live car has a leader (itself if alone in its lane)
        for _ in range(idx.size):
            limited = np.minimum(ds, np.maximum(g + ds[ahead] - SAFE_MARGIN, 0.0))
            if np.array_equal(limited, ds):
                break
            ds = limited
        clamped = ds < planned - 1e-3
        v_new = np.where(clamped, np.minimum(v_new, ds / dt), v_new)

        # Harsh braking ("emergency"): the driver model asks for more than
        # MAX_DECEL, or the no-overlap constraint has to stop the car harder
        # than that. Counted per episode, which ends after EMERGENCY_HOLD s.
        emergency = over_braking | ((v - v_new) / dt > MAX_DECEL + 1e-6)
        onset = int((emergency & (self.emergency_until[idx] < self.time)).sum())
        self.emergency_until[idx[emergency]] = self.time + EMERGENCY_HOLD
        if onset:
            self.n_emergency += onset
            self.emergency_log.append((self.time, onset))

        # A stop: dropping below SLOW_FRACTION of the target speed after having
        # been above RESUME_FRACTION (so crawling along in dense flow, or
        # slowing to turn off at an exit, is not a stop).
        moving = self.moving[idx]
        stopped = moving & (v_new < SLOW_FRACTION * v0)
        self.moving[idx] = ~stopped & (moving | (v_new > RESUME_FRACTION * v0))
        n_stopped = int(stopped.sum())
        if n_stopped:
            self.n_stops += n_stopped
            self.stop_log.append((self.time, n_stopped))

        dtheta = ds / self.radii[self.lane[idx]]
        old = self.theta[idx]
        new = old + dtheta
        new[new >= TWO_PI] -= TWO_PI  # == % TWO_PI: a step is far shorter than a lap
        self.theta[idx] = new
        self.travel[idx] += dtheta
        self.v[idx] = v_new

        det = self.detector_angle
        before, after = old - det, new - det
        before[before < 0] += TWO_PI
        after[after < 0] += TWO_PI
        crossed = (dtheta > 0) & (before > after)
        for lane in self.lane[idx[crossed]]:
            self.detector_log.append((self.time, int(lane)))

    def must_move_out(self, idx) -> np.ndarray:
        """Leavers not yet on the outer lane that are inside their lane-change zone."""
        to_go = self.outer - self.lane[idx]
        to_exit = (self.exit_at[idx] - self.travel[idx]) * self.r_out
        return (to_go > 0) & (to_exit < to_go * LC_ZONE + 20.0)

    def _make_room_to_leave(self, idx, v, v0, acc) -> None:
        """Cooperative merging for leavers that must move outwards.

        Driving alongside the target lane's traffic at its speed, a leaver
        would never find a gap. So it follows the car just ahead in that lane
        (braking at most 1.5x comfortably), and the car just behind it in that
        lane eases off to let it in (at most comfortable braking). Updates
        ``acc`` (aligned with ``idx``) in place.

        In dense traffic leavers and yielding cars can end up holding each
        other at a standstill; a leaver held there by merging (its own lane
        would let it go) for LEAVER_PATIENCE seconds gives up its exit and
        goes round again, which breaks the deadlock.
        """
        if self.n_lanes < 2:
            return
        must = self.must_move_out(idx)
        if not must.any():
            self.stuck[idx] = 0.0
            return
        own = acc.copy()
        m = np.flatnonzero(must)
        nl, front, nf, back = self._neighbours(self.lane[idx[m]] + 1, self.theta[idx[m]])
        v_front = np.where(nl >= 0, self.v[np.maximum(nl, 0)], v[m])
        follow = self._idm(v[m], v0[m], front, v_front)
        acc[m] = np.minimum(acc[m], np.maximum(follow, -1.5 * self.cfg.comfort_decel))
        has = nf >= 0
        if has.any():
            pos = np.empty(self.cap, dtype=np.int64)
            pos[idx] = np.arange(idx.size)
            f = pos[nf[has]]
            yield_acc = self._idm(v[f], v0[f], back[has], v[m][has])
            np.minimum.at(acc, f, np.maximum(yield_acc, -self.cfg.comfort_decel))

        held = must & (v < 0.5) & (own > 0) & (acc <= 0)
        self.stuck[idx] = np.where(held, self.stuck[idx] + self.cfg.dt, 0.0)
        give_up = idx[self.stuck[idx] > LEAVER_PATIENCE]
        if give_up.size:
            self.exit_at[give_up] += TWO_PI
            self.n_missed += give_up.size
            self.stuck[give_up] = 0.0

    def _exits(self) -> None:
        if self.n_arms == 0:
            return
        idx = np.flatnonzero(self.alive)
        due = idx[self.travel[idx] >= self.exit_at[idx]]
        if due.size == 0:
            return
        on_outer = self.lane[due] == self.outer
        missed = due[~on_outer]
        self.exit_at[missed] += TWO_PI  # not on the outer lane in time: go round again
        self.n_missed += missed.size
        leaving = due[on_outer]
        if leaving.size == 0:
            return
        arms = np.argmin(np.abs(((self.theta[leaving][:, None] - self.exit_angle[None, :]) + np.pi)
                                % TWO_PI - np.pi), axis=1)
        for slot, arm in zip(leaving, arms):
            self.exit_log.append((self.time, int(arm), self.time - self.born[slot]))
        self.alive[leaving] = False
        self.n_exited += leaving.size
        # Each leaver is replaced at a random entrance; if that queue is full the
        # replacement is slotted into the ring instead (by _sync_population).
        for arm in self.rng.integers(self.n_arms, size=leaving.size):
            if self.queue[arm] < MAX_QUEUE:
                self.queue[arm] += 1

    def _lane_changes(self, idx) -> None:
        cfg = self.cfg
        if self.n_lanes < 2 or idx.size == 0:
            return
        ready = idx[(self.lc_cool[idx] <= 0) & (self.brake_until[idx] <= self.time)]
        if ready.size == 0:
            return
        to_exit = (self.exit_at[ready] - self.travel[ready]) * self.r_out
        mandatory = self.must_move_out(ready)
        considering = mandatory | (self.rng.random(ready.size) < cfg.lane_change_rate * cfg.dt)
        cars = ready[considering]
        if cars.size == 0:
            return
        mandatory, to_exit = mandatory[considering], to_exit[considering]

        lead, gap = self._index_lanes(idx)
        ld = lead[idx]
        follower = np.full(self.cap, -1, dtype=np.int64)
        others = ld != idx
        follower[ld[others]] = idx[others]

        # Every (car, direction) option at once: outwards for cars[k] at k, inwards at n + k.
        n = cars.size
        target = np.concatenate((self.lane[cars] + 1, self.lane[cars] - 1))
        allowed = (target >= 0) & (target < self.n_lanes)
        # Move inwards only when the exit is far enough away.
        allowed[n:] &= ~mandatory & (to_exit > (self.outer - target[n:]) * LC_ZONE + 60.0)
        opt = np.flatnonzero(allowed)
        if opt.size == 0:
            return
        k = opt % n
        c = cars[k]
        nl, front, nf, back = self._neighbours(target[opt], self.theta[c])
        has_f = nf >= 0
        f = np.maximum(nf, 0)
        # The car that was following us gets our old leader.
        o = follower[c]
        has_o = o >= 0
        os_ = np.maximum(o, 0)
        # Desired speeds and current accelerations of everyone involved.
        need = np.concatenate((c, f, os_))
        v0 = np.zeros(self.cap)
        v0[need] = self.target_speed(need)
        acc = np.zeros(self.cap)
        acc[need] = self._idm(self.v[need], v0[need], gap[need], self.v[lead[need]])

        vc = self.v[c]
        v_front = np.where(nl >= 0, self.v[np.maximum(nl, 0)], vc)
        # Judge the target lane with the speed this driver could do there
        # (its cornering cap differs from the current lane's).
        a_self = self._idm(vc, self.desired_speed(c, target[opt]), front, v_front)
        a_f_new = np.where(has_f, self._idm(self.v[f], v0[f], back, vc), 0.0)
        a_f_old = np.where(has_f, acc[f], 0.0)
        a_o_new = np.where(has_o, self._idm(self.v[os_], v0[os_], gap[os_] + cfg.car_length + gap[c],
                                            self.v[lead[c]]), 0.0)
        a_o_old = np.where(has_o, acc[os_], 0.0)
        forced = mandatory[k] & (opt < n)  # a leaver must move out
        b_safe = cfg.comfort_decel * np.where(mandatory[k], 2.5, 1.5)
        min_room = max(0.5 * cfg.min_gap, SAFE_MARGIN + 0.2)
        safe = (front >= min_room) & (back >= min_room) & (a_f_new >= -b_safe) & (a_self >= -b_safe)
        gain = (a_self - acc[c]) + cfg.politeness * ((a_f_new - a_f_old) + (a_o_new - a_o_old))
        if cfg.lane_change_motive == "random":
            wants = np.ones(opt.size, dtype=bool)
            s = np.empty(opt.size)
            # Same draw order as one draw per (direction, target lane) group.
            s[np.lexsort((target[opt], opt >= n))] = self.rng.random(opt.size)
        else:
            wants = gain > LC_THRESHOLD
            s = gain
        ok = safe & (wants | forced)
        score = np.full(2 * n, -np.inf)
        score[opt] = np.where(ok, np.where(forced, np.inf, s), -np.inf)
        new_lead = np.full(2 * n, -1, dtype=np.int64)
        new_foll = np.full(2 * n, -1, dtype=np.int64)
        new_lead[opt], new_foll[opt] = nl, nf

        t_out, s_out, l_out, f_out = target[:n], score[:n], new_lead[:n], new_foll[:n]
        t_in, s_in, l_in, f_in = target[n:], score[n:], new_lead[n:], new_foll[n:]
        go_out = s_out >= s_in
        best = np.where(go_out, s_out, s_in)
        chosen = np.flatnonzero(best > -np.inf)
        if chosen.size == 0:
            return
        involved: set[int] = set()
        for k in self.rng.permutation(chosen):
            c = int(cars[k])
            nl = int(l_out[k] if go_out[k] else l_in[k])
            nf = int(f_out[k] if go_out[k] else f_in[k])
            lane = int(t_out[k] if go_out[k] else t_in[k])
            group = {c, nl, nf, int(follower[c])} - {-1}
            if nl < 0:  # empty target lane: nothing links two entrants, so allow one per step
                group.add(-10 - lane)
            if group & involved:
                continue  # a neighbour is already changing lane this step
            involved |= group
            self.lane[c] = lane
            self.lc_cool[c] = LC_COOLDOWN
            self.n_lane_changes += 1
            self.lane_change_log.append(self.time)

    def _entries(self) -> None:
        if self.n_arms == 0 or not self.queue.any():
            return
        ready = np.flatnonzero((self.queue > 0) & (self.entry_cool <= 0))
        if ready.size == 0:
            return
        # Only the outer lane's index is needed (the others are rebuilt before their next use).
        cars = np.flatnonzero(self.alive & (self.lane == self.outer))
        order = cars[np.argsort(self.theta[cars], kind="stable")]
        self._lanes[self.outer] = (order, self.theta[order])
        arms = self.rng.permutation(ready)
        while arms.size:
            # Check the remaining arms together; after an entry, re-check the ones after it.
            ok, v_in = self._entry_ok(self.entry_angle[arms])
            hits = np.flatnonzero(ok)
            if hits.size == 0:
                break
            k = hits[0]
            arm, angle = arms[k], self.entry_angle[arms[k]]
            slot = self._spawn(np.array([self.outer]), np.array([angle]), v_in[k:k + 1],
                               lat=np.array([self.outer + 1.0]))
            self.entered_at[slot] = self.time
            self.queue[arm] -= 1
            self.entry_cool[arm] = ENTRY_COOLDOWN
            self.n_entered += 1
            # Keep the outer lane's sorted index current for the next arm.
            order, th = self._lanes[self.outer]
            pos = np.searchsorted(th, angle)
            self._lanes[self.outer] = (np.insert(order, pos, slot[0]), np.insert(th, pos, angle % TWO_PI))
            arms = arms[k + 1:]

    def _entry_ok(self, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Whether a car may enter the outer lane at each of ``angles``, and at what speed."""
        cfg = self.cfg
        v_in = np.full(angles.size, min(cfg.turn_speed, cfg.mean_speed))
        nl, front, nf, back = self._neighbours_in(self.outer, angles)
        if nl[0] < 0:  # empty outer lane
            return np.ones(angles.size, dtype=bool), v_in
        vf = self.v[nf]
        follower_ok = self._idm(vf, self.desired_speed(nf), back, v_in) >= -cfg.comfort_decel
        ok = ((front >= cfg.min_gap + 1.0) & (back >= cfg.min_gap + 1.0)
              & (back >= cfg.entry_gap * vf) & follower_ok)
        return ok, np.where(front < 15.0, np.minimum(v_in, self.v[nl]), v_in)

    def _sync_population(self) -> None:
        """Match the number of cars to the density setting."""
        target = self.target_cars()
        n_ring = int(self.alive.sum())
        surplus = n_ring + int(self.queue.sum()) - target
        if surplus > 0:
            for arm in np.argsort(-self.queue):
                take = min(self.queue[arm], surplus)
                self.queue[arm] -= take
                surplus -= take
            if surplus > 0:
                victims = self.rng.choice(np.flatnonzero(self.alive), size=min(surplus, n_ring), replace=False)
                self.alive[victims] = False
        elif surplus < 0 and self.time >= self._next_fill:
            got = self._insert_into_gaps(-surplus)
            if got < -surplus and not self._respace_lanes(-surplus - got) and got == 0:
                self._next_fill = self.time + 1.0  # ring is full; retry in a second

    def _insert_into_gaps(self, k: int) -> int:
        """Insert up to ``k`` cars into gaps; returns how many fitted.

        A few cars (leavers' replacements whose queue was full) go into random
        gaps that are big enough for the car behind not to slow down, so they
        do not systematically fill the emptiest stretch of road that a "do cars
        congregate?" measure looks for. Bulk density raises, and small ones
        when too few such gaps exist (dense traffic), go into the largest gaps.
        """
        cfg = self.cfg
        # A gap must leave the new car and its follower the lane-change safety room.
        min_room = max(0.5 * cfg.min_gap, SAFE_MARGIN + 0.2)
        fits = 2 * (cfg.car_length + min_room)
        idx = np.flatnonzero(self.alive)
        _, gap = self._index_lanes(idx)
        # Gaps: (-arc length, lane, angle of the car behind the gap, its slot).
        # Slot -2: a car inserted here; -3: an empty lane.
        heap: list[tuple[float, int, float, int]] = []
        for lane, (order, th) in enumerate(self._lanes):
            if order.size == 0:
                heap.append((-self.lane_len[lane], lane, float(self.rng.uniform(0, TWO_PI)), -3))
            else:
                heap.extend(zip(-(gap[order] + cfg.car_length), [lane] * order.size,
                                th.tolist(), order.tolist()))
        small = k <= SMALL_FILL
        if small:
            # Only gaps where the car behind need not slow down for the new car.
            cand = [entry for entry in heap if -entry[0] >= fits and not self._near_exit(entry)]
            behind = np.array([entry[3] for entry in cand], dtype=np.int64)
            keep = behind < 0
            if (~keep).any():  # one vectorised check for all the cars behind
                room = np.array([-0.5 * entry[0] - cfg.car_length for entry in cand])[~keep]
                keep[~keep] = self.equilibrium_speed(room, self.desired_speed(behind[~keep])) >= self.v[behind[~keep]]
            ok = [entry for entry, fine in zip(cand, keep) if fine]
            if len(ok) >= k:
                heap = [ok[i] for i in self.rng.choice(len(ok), size=k, replace=False)]
        heapq.heapify(heap)
        lanes, angles, followers, arcs = [], [], [], []
        for _ in range(k):
            if not heap:
                break
            neg_arc, lane, start, foll = heapq.heappop(heap)
            arc = -neg_arc
            if arc < fits:
                break
            lanes.append(lane)
            if foll == -3:  # empty lane: the first car's gap is the whole lane
                angles.append(start)
                followers.append(-2)
                arcs.append(arc)
                heapq.heappush(heap, (neg_arc, lane, start, -2))
                continue
            angle = (start + 0.5 * arc / self.radii[lane]) % TWO_PI
            angles.append(angle)
            followers.append(foll)
            arcs.append(0.5 * arc)
            heapq.heappush(heap, (-0.5 * arc, lane, start, foll))
            heapq.heappush(heap, (-0.5 * arc, lane, angle, -2))
        if not lanes:
            return 0
        slots = self._spawn(np.array(lanes), np.array(angles), np.zeros(len(lanes)))
        self.inserted_at[slots] = self.time
        eq = self.equilibrium_speed(np.array(arcs) - cfg.car_length, self.desired_speed(slots))
        foll = np.array(followers)
        v_f = np.where(foll >= 0, self.v[np.maximum(foll, 0)], eq)
        self.v[slots] = np.minimum(eq, v_f)
        # The car now directly behind each new car (its old follower, or an
        # earlier new car in a gap split twice) slows to suit its new gap, so
        # the insertion does not set off a man-made braking wave.
        idx = np.flatnonzero(self.alive)
        lead, gap = self._index_lanes(idx)
        behind = idx[np.isin(lead[idx], slots) & (lead[idx] != idx)]
        # A replacement slotted in now and then: the car behind brakes physically
        # (the driver model does the rest). A bulk density raise from the slider
        # teleports many cars at once: settle their followers at once instead
        # of setting off a man-made braking wave.
        self._cap_speed(behind, gap[behind], max_cut=MAX_DECEL * cfg.dt if small else np.inf)
        return len(lanes)

    def _near_exit(self, gap_entry: tuple[float, int, float, int]) -> bool:
        """Whether a car slotted into this gap (outer lane) would sit just before an exit."""
        neg_arc, lane, start, _ = gap_entry
        if lane != self.outer or self.n_arms == 0:
            return False
        mid = start + 0.5 * -neg_arc / self.radii[lane]
        before = (self.exit_angle - mid) % TWO_PI * self.radii[lane]
        return bool((before < 15.0).any())

    def _cap_speed(self, slots, gap, max_cut: float = np.inf) -> None:
        """Slow ``slots`` to the steady-state speed for ``gap`` (by at most
        ``max_cut`` m/s; the driver model brakes the rest), without logging stops.

        Slowed cars are marked as touched by the density control, so a jam they
        head is counted as "other", not "following".
        """
        eq = self.equilibrium_speed(gap, self.desired_speed(slots))
        if np.isfinite(max_cut):
            eq = np.maximum(eq, self.v[slots] - max_cut)
        slowed = slots[self.v[slots] > eq]
        self.v[slots] = np.minimum(self.v[slots], eq)
        self.acc_hist[:, slowed] = np.minimum(self.acc_hist[:, slowed], 0.0)
        self.moving[slowed] &= self.v[slowed] >= SLOW_FRACTION * self.target_speed(slowed)
        self.inserted_at[slowed] = self.time

    def _respace_lanes(self, k: int) -> bool:
        """Fallback for near-jam densities, where no single gap can take a car.

        Gives up to ``k`` extra cars to the lanes with spare room (one more each
        round, to the lane whose cars would keep the most space), then spaces
        each changed lane's cars evenly, keeping their order. Returns False if
        every lane is full.
        """
        cfg = self.cfg
        pitch = cfg.car_length + cfg.min_gap
        idx = np.flatnonzero(self.alive)
        self._index_lanes(idx)
        count = np.array([order.size for order, _ in self._lanes])
        room = np.floor(self.lane_len / pitch).astype(int)
        extra = np.zeros(self.n_lanes, dtype=int)
        for _ in range(k):
            # At most one new car per existing gap in a lane.
            spare = (count + extra < room) & ((count == 0) | (extra < count))
            if not spare.any():
                break
            extra[np.argmax(np.where(spare, self.lane_len / (count + extra + 1), -1.0))] += 1
        if not extra.any():
            return False
        for lane in np.flatnonzero(extra):
            order, th = self._lanes[lane]
            n_new = order.size + extra[lane]
            step = TWO_PI / n_new
            if order.size:
                # New cars go into the largest gaps, one per gap.
                gaps = np.diff(np.append(th, th[0] + TWO_PI))
                take = np.zeros(order.size, dtype=bool)
                take[np.argsort(-gaps)[:extra[lane]]] = True
                new = self._spawn(np.full(extra[lane], lane), th[take], np.zeros(extra[lane]))
                sequence = np.empty(n_new, dtype=np.int64)
                at = np.arange(order.size) + np.cumsum(np.concatenate(([0], take[:-1])))
                sequence[at] = order
                sequence[at[take] + 1] = new
                old = self.theta[sequence].copy()
                self.theta[sequence] = (th[0] + step * np.arange(n_new)) % TWO_PI
                # Keep each car's distance to its exit tied to its new position.
                self.travel[sequence] += (self.theta[sequence] - old + np.pi) % TWO_PI - np.pi
            else:
                new = sequence = self._spawn(np.full(n_new, lane), np.zeros(n_new), np.zeros(n_new))
                self.theta[sequence] = (self.rng.uniform(0, TWO_PI) + step * np.arange(n_new)) % TWO_PI
                self._assign_routes(new)
            self.inserted_at[new] = self.time
            self.v[new] = np.inf  # capped to the steady-state speed below
            self._cap_speed(sequence, np.full(n_new, step * self.radii[lane] - cfg.car_length))
        return True
