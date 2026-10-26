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
   the outer lane; otherwise it misses the exit and goes round again. In
   closed mode every leaver is replaced by a car queued at a random entrance.
3. **Lane changes** - leavers move outwards in time for their exit; other
   drivers consider a change ``lane_change_rate`` times per second and make it
   if it is safe (and, in "advantage" mode, faster - the MOBIL model).
4. **Arrivals** (open mode) - cars arrive at each entrance at random
   (Poisson, ``entry_rate`` per second, shaped by ``demand_imbalance`` and
   ``demand_profile``) and join its queue; arrivals that find ``max_queue``
   cars waiting are turned away (blocked).
5. **Entries** - queued cars merge into the outer lane when the gap to
   circulating traffic is at least ``entry_gap`` seconds and more than
   ``follow_up`` seconds after the previous car from that entrance (on the
   first step after it: 1 s means 1.1 s at the default time step). In closed
   mode at most ``MAX_QUEUE`` cars wait per entrance; beyond that a leaver's
   replacement is slotted straight into the ring, so the ring never drains
   into queues. Each queue keeps its cars' arrival times (oldest first), so
   every entry knows how long the car waited.

In closed mode density is held as cars on the ring per km of lane: raising it
inserts cars into the largest gaps, lowering it removes random cars. In open
mode it only sets the cars at the start (and once more whenever it is changed
live); after that arrivals and exits decide how many cars are on the ring.

Vehicles and drivers differ: a ``truck_share`` of the vehicles are trucks
(``truck_length`` long, slower to accelerate, slower and keeping a longer time
gap, see TRUCK_*), and each driver gets their own reaction time and time gap
around the settings (``reaction_std``, ``headway_std``). A vehicle's ``theta``
is its centre, so the bumper gap to the leader is the distance between their
centres less half of each one's length. The ``road_condition`` scales the
settings the dynamics use (desired speed, acceleration, braking, time gap,
entry gap, follow-up, cornering and the braking limit; ROAD_CONDITIONS), all
through one set of effective values, ``RingSim.eff`` (dry: the settings
themselves).
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import asdict, dataclass, fields

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
PERTURB_DECEL = 4.0  # [m/s2] braking of a car the user tells to brake
MAX_DECEL = 9.0  # [m/s2] physical braking limit; needing more counts as harsh braking
SLOW_FRACTION = 0.3  # a car counts as jammed below this fraction of its free speed
RESUME_FRACTION = 0.6  # ...and as moving again above this (hysteresis for counting stops)
MAX_QUEUE = 2  # cars waiting per entrance in closed mode (open mode: cfg.max_queue)
PEAK_PERIOD = 600.0  # [s] one rise and fall of the "peak" demand profile
PEAK_SWING = 0.5  # the "peak" profile runs from (1 - this) to (1 + this) times the mean demand
LOG_HORIZON = 600.0  # [s] how long event logs are kept for rolling rates
EMERGENCY_HOLD = 2.0  # [s] an emergency episode ends after this long without one
LEAVER_PATIENCE = 5.0  # [s] a leaver held at a standstill by merging gives up its exit
SMALL_FILL = 3  # deficits up to this many cars are slotted into random gaps
ANTICIPATION = 5  # most cars ahead a driver reacts to (within the reaction distance)
HEADWAY_FLOOR = 0.5  # nobody keeps less than this share of the Following time gap (cf. desired speeds)
TEMPERAMENT_CLIP = 2.5  # per-driver draws are standard normals clipped to +- this

# Per-car arrays, one value per slot (RingSim._alloc); the acceleration history acc_hist holds one row
# per step of history on top.
CAR_ARRAYS = ("alive", "lane", "theta", "travel", "v", "z", "exit_at", "born", "lat", "lc_cool", "noise",
              "brake_until", "uid", "moving", "emergency_until", "stuck", "entered_at", "inserted_at", "truck",
              "length", "z_react", "z_gap", "accel", "headway", "delay")
# Saved states (RingSim.state_dict): the layout version, and what they hold besides the per-car arrays.
STATE_VERSION = 1
ARM_ARRAYS = {"queue": int, "entry_cool": np.int64, "arm_arrived": np.int64, "arm_entered": np.int64,
              "queue_busy": float}  # one value per entrance, and its dtype
COUNTERS = ("n_entered", "n_exited", "n_missed", "n_lane_changes", "n_emergency", "n_stops", "n_arrived",
            "n_blocked", "n_gave_up")  # whole numbers (entry_wait_total is the one float total)
EVENT_LOGS = ("detector_log", "exit_log", "lane_change_log", "emergency_log", "stop_log", "arrival_log",
              "entry_log", "blocked_log", "gave_up_log")  # tuples starting with the time; lane_change_log: times
GENERATORS = ("rng", "rng_demand", "rng_drivers")  # every random stream

# Heavy vehicles (trucks): multiples of a car driver's setting. Truck drivers keep a car's standstill gap
# and comfortable braking (the sources disagree: x1-2 and x0.67-1) and a car's entry gap and follow-up
# time (field data give x1.7-1.9; here a truck's length and slow start already make it need more room).
# max acceleration: IDM a 0.5 vs 1.0 m/s2 (Treiber & Kesting 2013, errata 2017), 0.7 vs 1.4 (EIDM, Kesting 2010)
TRUCK_ACCEL_FACTOR = 0.5  # and trucks measured at 0.59 m/s2 from a stop (Yang 2016)
TRUCK_SPEED_FACTOR = 0.7  # desired speed: 80 vs 120 km/h (Treiber & Kesting; MOBIL, Kesting 2007), 85 vs 120 (EIDM)
# time gap: 2.0 vs 1.5 s (EIDM) to 1.8 vs 1.0 s (Treiber & Kesting)
TRUCK_HEADWAY_FACTOR = 1.5  # and truck time gaps 1.7-2.0x a car's on roundabouts (Macioszek 2019)
TRUCK_WIDTH = 2.55  # [m] EU maximum (Directive 96/53/EC Annex I); US WB-62/67 trucks are 2.59 m

# Road surface: multiple of each setting the dynamics use ("max_decel" scales MAX_DECEL; the speed and time-gap
# factors scale each driver's own value, so the spreads between drivers too). A synthesis of
# weather studies, mostly on freeways and signalised streets (Hranac et al. 2006 FHWA-HOP-07-073; Rakha
# et al. 2009 FHWA-JPO-09-066 and 2010 FHWA-JPO-11-020; Martin/Perrin et al. 2000 MPC-01-120; Wallman &
# Astrom 2001 VTI 911A); ice is the least certain. The standstill gap (jam density) and reaction time are
# not scaled: the studies find no weather effect on jam density, and the reaction-time evidence is thin.
ROAD_SCALED = ("mean_speed", "max_accel", "comfort_decel", "time_headway", "entry_gap", "follow_up",
               "lateral_accel", "max_decel")
ROAD_CONDITIONS: dict[str, dict[str, float]] = {
    "dry": dict.fromkeys(ROAD_SCALED, 1.0),
    "wet": dict(
        mean_speed=0.92,  # free speed -6 to -9 % in rain (FHWA 2006), -10 % on arterials (Perrin 2000)
        max_accel=0.95,  # assumption: wet/dry peak adhesion 0.90 (Rakha 2010 Tab. 4.1), start-up barely slower (Perrin)
        comfort_decel=0.88,  # approach deceleration x0.876 in 1.6 cm/h rain (Rakha 2009 Eq. 5.2)
        time_headway=1.12,  # capacity -10 to -11 % in rain (FHWA 2006); SHRP 2 heavy-rain factor 0.89
        entry_gap=1.10,  # accepted gaps +10 % at four roundabouts in rain (Lee 2018); 1.104 (Rakha 2010 Tab. 3.5)
        follow_up=1.05,  # queue discharge headways +2 to +6 % in rain (Sadek 2004, Perrin 2000)
        lateral_accel=0.90,  # wet/dry friction 0.9 (Rakha 2010)
        max_decel=0.80,  # friction wet 0.7-0.8 vs dry 0.8-1.0 (VTI 2001): 7.2 m/s2
    ),
    "snow": dict(
        mean_speed=0.75,  # arterial speeds -25 to -40 % on snowy or slushy roads (FHWA RWM, Perrin 2000)
        max_accel=0.70,  # assumption: start-up lost time +23 % (Perrin 2000); traction on packed snow 1.2-1.8 m/s2
        comfort_decel=0.70,  # comfortable braking 0.27 g -> 0.20 g on snow (x0.74, via Perrin 2000); x0.56 (Sun 2026)
        time_headway=1.25,  # discharge headways +13 to +26 % in snow and slush (Perrin 2000); SHRP 2 heavy snow 0.80
        entry_gap=1.20,  # critical gap x1.19-1.20 with snow on a snowy surface (Rakha 2010 Tab. 3.5)
        follow_up=1.20,  # discharge headways +13 to +26 % in snow (Perrin 2000)
        lateral_accel=0.50,  # 2 m/s2: about the packed-snow friction limit, 0.2-0.3 g (VTI 2001); Kageyama 2013
        max_decel=0.30,  # packed-snow friction 0.20-0.30 (VTI 2001): 2.7 m/s2
    ),
    "ice": dict(
        mean_speed=0.70,  # free speed -28 % on an icy test track (Rakha 2010 Tab. 2.3)
        max_accel=0.50,  # assumption: traction limit mu 0.1 x 0.6 x g = 0.59 m/s2, half the default 1.2
        comfort_decel=0.45,  # assumption: below snow (Perrin 2000, qualitative) and the 1.35 m/s2 limit
        time_headway=1.60,  # assumption between snow (1.25) and x1.87 at capacity on an icy track (Rakha 2010)
        entry_gap=1.12,  # critical gaps x1.11-1.12 on an icy surface (Rakha 2010 Tab. 3.5)
        follow_up=1.30,  # assumption: above snow (no field data)
        lateral_accel=0.30,  # 1.2 m/s2: about the ice friction limit, 0.1-0.15 g (VTI 2001, Rakha 2010)
        max_decel=0.15,  # black ice 0.15-0.30, packed snow or ice 0.10-0.25 (VTI 2001, Rakha 2010): 1.35 m/s2
    ),
}


@dataclass(frozen=True)
class Effective:
    """The settings the dynamics use: ``SimConfig``'s, scaled by the road condition (dry: unchanged).

    The factors act on each driver's own desired speed and time gap, so they
    scale the spreads between drivers too.
    """
    mean_speed: float  # [m/s]
    speed_std: float  # [m/s]
    max_accel: float  # [m/s2]
    comfort_decel: float  # [m/s2]
    time_headway: float  # [s]
    headway_std: float  # [s]
    entry_gap: float  # [s]
    follow_up: float  # [s]
    lateral_accel: float  # [m/s2]
    max_decel: float  # [m/s2] physical braking limit (MAX_DECEL on a dry road)


def effective(cfg: SimConfig) -> Effective:
    """The settings ``cfg`` drives with on its road condition (one multiplier each, ROAD_CONDITIONS)."""
    k = ROAD_CONDITIONS[cfg.road_condition]
    return Effective(**{name: getattr(cfg, name) * k[name] for name in ROAD_SCALED if name != "max_decel"},
                     speed_std=cfg.speed_std * k["mean_speed"], headway_std=cfg.headway_std * k["time_headway"],
                     max_decel=MAX_DECEL * k["max_decel"])


def braking_limits(conditions=("wet", "snow", "ice")) -> str:
    """The physical braking limit [m/s2] on each of the road ``conditions``, e.g. "7.2 / 2.7 / 1.35"."""
    return " / ".join(f"{MAX_DECEL * ROAD_CONDITIONS[c]['max_decel']:.3g}" for c in conditions)


def mean_demand_factor(profile: str, t0: float, t1: float) -> float:
    """Mean of the demand profile (a multiple of the entrance chance) from ``t0`` to ``t1`` s.

    1 for "constant". For "peak" it is 1 only over whole PEAK_PERIODs (or a
    window placed symmetrically about a peak or trough): a sweep measuring
    120-420 s, for example, catches the rise to the peak and averages 1.30.
    """
    if profile != "peak" or t1 <= t0:
        return 1.0
    w = TWO_PI / PEAK_PERIOD
    return float(1.0 - PEAK_SWING * (np.sin(w * t1) - np.sin(w * t0)) / (w * (t1 - t0)))


# ---------------------------------------------------------------- saved states
class StateError(ValueError):
    """A saved state that cannot be restored: incomplete, damaged or from another version (the
    message says what is wrong)."""


# Types the fields of an event log's entries can have, by the name a saved state records them under.
FIELD_TYPES = {"float": float, "int": int, "bool": bool, "float64": np.float64, "int64": np.int64,
               "bool_": np.bool_}
_FIELD_NAMES = {kind: name for name, kind in FIELD_TYPES.items()}


def _pack_log(log: deque) -> tuple[np.ndarray, list[str]]:
    """An event log as an array (a row per entry, a column per field; 1-D for plain times) and the
    type of each field (as in its first entry), so :func:`_unpack_log` gives back equal entries of
    the same Python types. Whole numbers are exact in the float array up to 2**53."""
    if not log:
        return np.zeros(0), []
    first = log[0]
    types = [_FIELD_NAMES.get(type(value), "float") for value in (first if isinstance(first, tuple) else (first,))]
    return np.array(list(log), dtype=float), types


def _unpack_log(name: str, values: np.ndarray, types: list[str], plain: bool) -> deque:
    """The event log :func:`_pack_log` packed: tuples of ``types``, or plain values if ``plain``."""
    if values.shape[:1] == (0,):
        return deque()
    width = 1 if plain else (values.shape[1] if values.ndim == 2 else -1)
    if values.ndim != (1 if plain else 2) or len(types) != width or not set(types) <= set(FIELD_TYPES):
        raise StateError(f"{name} is not a valid event log")
    columns = []
    for column, kind in zip([values] if plain else values.T, types):
        if kind not in ("float", "float64") and not (np.isfinite(column).all() and (column == np.round(column)).all()):
            raise StateError(f"{name} holds a fraction where a whole number belongs")
        typed = column.astype(FIELD_TYPES[kind])
        columns.append(typed.tolist() if FIELD_TYPES[kind] in (float, int, bool) else list(typed))
    return deque(columns[0] if plain else zip(*columns))


def _generator(state) -> np.random.Generator:
    """A random generator set to a saved bit-generator ``state`` (numpy's PCG64 family)."""
    name = state.get("bit_generator") if isinstance(state, dict) else None
    if name not in ("PCG64", "PCG64DXSM"):
        raise StateError(f"unsupported random generator {name!r}")
    generator = np.random.Generator(getattr(np.random, name)())
    generator.bit_generator.state = state
    return generator


def _array(arrays: dict, name: str, shape: tuple | None, dtype) -> np.ndarray:
    """A copy of saved array ``name`` as ``dtype``, checking its shape (None: any) and kind of number."""
    if name not in arrays:
        raise StateError(f"the state lacks the array {name}")
    arr = np.asarray(arrays[name])
    if shape is not None and arr.shape != tuple(shape):
        raise StateError(f"{name} has shape {arr.shape}, expected {tuple(shape)}")
    if not np.can_cast(arr.dtype, dtype, casting="same_kind"):
        raise StateError(f"{name} holds {arr.dtype}, expected {np.dtype(dtype)}")
    return arr.astype(dtype, copy=True)


def _split(values: np.ndarray, counts: np.ndarray, name: str) -> list[np.ndarray]:
    """``values`` cut into consecutive parts of ``counts`` elements (one per entrance or lane)."""
    if values.ndim != 1 or (counts < 0).any() or counts.sum() != values.size:
        raise StateError(f"{name} does not match its counts")
    bounds = np.concatenate(([0], np.cumsum(counts)))
    return [values[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


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
        # Open-mode arrivals draw from a stream of their own, so the demand settings never change a
        # closed-mode run and the same seed gives the same arrivals. What the arrivals cause (entries,
        # new drivers) still draws from self.rng, so different demand shifts the rest of an open run.
        self.rng_demand = np.random.default_rng([self.seed, 1])
        # Vehicle types and drivers' own reaction times and time gaps: a stream of their own too, so
        # with every vehicle a car and every driver alike the run is the same as without them.
        self.rng_drivers = np.random.default_rng([self.seed, 2])
        self._update_effective()
        self.time = 0.0
        self.steps = 0
        self.generation = getattr(self, "generation", -1) + 1  # lets observers notice rebuilds

        self._geometry()
        n = self.n_arms
        self.queue = np.zeros(n, dtype=int)  # cars waiting at each entrance
        # Arrival time of every queued car, oldest first: len(queue_times[arm]) == queue[arm] always.
        self.queue_times: list[deque[float]] = [deque() for _ in range(n)]
        # Whether each queued vehicle is a truck (drawn when it joins), in the same order.
        self.queue_trucks: list[deque[bool]] = [deque() for _ in range(n)]
        self.entry_cool = np.zeros(n, dtype=np.int64)  # steps until each entrance may let its next car in
        self.demand_weight = self._demand_weights()

        self.n_entered = self.n_exited = self.n_missed = 0
        self.n_lane_changes = self.n_emergency = self.n_stops = 0
        # Every arrival ends up entered, still queued, blocked or given up:
        # n_arrived == n_entered + queue.sum() + n_blocked + n_gave_up.
        self.n_arrived = self.n_blocked = 0
        self.n_gave_up = 0  # queued cars dropped when Max queue (open) or Density (closed) was lowered live
        self.entry_wait_total = 0.0  # [s] queue wait summed over every car that entered
        # Per entrance, for its demand / capacity ratio (metrics.entry_vc).
        self.arm_arrived = np.zeros(n, dtype=np.int64)  # arrivals (blocked ones included)
        self.arm_entered = np.zeros(n, dtype=np.int64)  # entries
        self.queue_busy = np.zeros(n)  # [s] time it has had cars waiting to enter
        # Event logs for rolling rates, oldest first, pruned after LOG_HORIZON.
        self.detector_log: deque[tuple[float, int]] = deque()  # (time, lane) crossings
        self.exit_log: deque[tuple[float, int, float]] = deque()  # (time, arm, trip time)
        self.lane_change_log: deque[float] = deque()  # time
        self.emergency_log: deque[tuple[float, int]] = deque()  # (time, new emergencies)
        self.stop_log: deque[tuple[float, int]] = deque()  # (time, cars that stopped)
        self.arrival_log: deque[tuple[float, int]] = deque()  # (time, arm) of each arrival
        self.entry_log: deque[tuple[float, int, float]] = deque()  # (time, arm, queue wait)
        self.blocked_log: deque[tuple[float, int]] = deque()  # (time, arm) arrivals turned away
        self.gave_up_log: deque[tuple[float, int]] = deque()  # (time, arm) queued cars that gave up
        # Once a second: (time, arm_arrived, arm_entered, queue_busy) copies, for rolling ratios.
        self.entrance_log: deque[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = deque(
            [(0.0, self.arm_arrived.copy(), self.arm_entered.copy(), self.queue_busy.copy())])
        self._next_uid = 0
        self._next_fill = 0.0
        # Whether each vehicle the density control drew but could not place yet is a truck (oldest
        # first): the next fill places them first, so a truck that fits no gap waits for one.
        self._pending: list[bool] = []
        self._next_prune = 0.0
        self._lanes: list[tuple[np.ndarray, np.ndarray]] = [(np.zeros(0, dtype=np.int64), np.zeros(0))] * self.n_lanes

        self._hist_len = int(round(MAX_REACTION / cfg.dt)) + 1
        self._hist_ptr = 0
        self._alloc(max(64, 2 * self.capacity()))
        self._place_initial(self.target_cars())
        if self.open_demand and int(self.alive.sum()) < self.target_cars():
            # Vehicles that fitted no lane of their own (a heavy truck mix, or a start at jam
            # density): closed mode tops the ring up on its first step, open mode places them
            # now, as a live Density raise does, so Density sets the cars at the start.
            self._sync_ring()

    def _geometry(self) -> None:
        """Lanes, arms and the detector line from the settings (nothing random)."""
        cfg = self.cfg
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

    def set(self, **changes) -> bool:
        """Change parameters live. Returns True if the change rebuilt the roundabout."""
        if not changes:
            return False
        old, old_seed = self.cfg, self.seed
        new_cfg = old.updated(**changes)  # validates names and values
        # Truck length matters only once there are trucks: without any, before and after, it is just
        # kept for later (nothing on the ring depends on it).
        idle = {"truck_length"} if old.truck_share == 0 and new_cfg.truck_share == 0 else set()
        # Geometry/numerics fields without a slider (lane_width, dt, ...) also rebuild.
        rebuild = any((PARAM_BY_NAME[k].reset if k in PARAM_BY_NAME else k != "seed")
                      and getattr(old, k) != v for k, v in changes.items() if k not in idle)
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
            self.rng_demand = np.random.default_rng([self.seed, 1])
            self.rng_drivers = np.random.default_rng([self.seed, 2])
        alive = np.flatnonzero(self.alive)
        # Every driver keeps their temperaments; their own time gap, acceleration and reaction
        # delay follow the new settings and road condition at once.
        self._update_effective()
        self._set_driving(alive)
        if new_cfg.exit_prob != old.exit_prob:
            self._assign_routes(alive)  # geometric routes are memoryless
        if any(getattr(new_cfg, k) != getattr(old, k) for k in
               ("mean_speed", "speed_std", "lateral_accel", "turn_speed", "exit_prob", "comfort_decel",
                "road_condition")):
            # New free speeds: re-baseline who counts as moving, so the change
            # itself is not logged as a wave of stops.
            self.moving[alive] &= self.v[alive] >= SLOW_FRACTION * self.target_speed(alive)
        if new_cfg.demand_imbalance != old.demand_imbalance:
            self.demand_weight = self._demand_weights()
        if self.open_demand:
            if new_cfg.max_queue < old.max_queue:  # shorter queues: their latest arrivals give up
                for arm in range(self.n_arms):
                    self._drop_queued(arm, max(int(self.queue[arm]) - new_cfg.max_queue, 0))
            if new_cfg.density != old.density:
                self._sync_ring()  # Density sets the cars on the ring once, at this moment
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
        # Vehicle and driver: drawn at spawn from rng_drivers.
        self.truck = np.zeros(cap, dtype=bool)
        self.length = np.zeros(cap)  # [m] car_length or truck_length
        self.z_react = np.zeros(cap)  # temperaments ~ N(0, 1): own reaction time = setting + z * spread
        self.z_gap = np.zeros(cap)  # ...and own time gap
        # Derived from those and the effective settings (_set_driving), so the step need not.
        self.accel = np.zeros(cap)  # [m/s2] max acceleration
        self.headway = np.zeros(cap)  # [s] IDM time gap
        self.delay = np.zeros(cap, dtype=np.int64)  # [steps] reaction delay

    def _grow(self, need: int) -> None:
        old = {name: getattr(self, name) for name in CAR_ARRAYS}
        hist, old_cap = self.acc_hist, self.cap
        self._alloc(max(2 * old_cap, old_cap + need))
        for name, arr in old.items():
            getattr(self, name)[:old_cap] = arr
        self.acc_hist[:, :old_cap] = hist

    # -------------------------------------------------------- saved state
    def state_dict(self) -> dict:
        """Everything an exact resume needs: ``{"meta": a JSON-serialisable dict, "arrays": {name: array}}``.

        :meth:`from_state` turns it back into a simulation that carries on bit
        for bit as this one would. It holds the settings, seed, clock and
        generation; every per-car array (CAR_ARRAYS), the acceleration history
        and its pointer; the queues with each queued car's arrival time and
        vehicle type; the entrance cooldowns, counters, event logs and
        once-a-second entrance totals; the uid counter, the fill and prune
        timers and the vehicles the density control still has to place; the
        lane index cache; and the bit-generator state of every random stream
        (GENERATORS). What follows from the settings alone (the geometry,
        ``eff``, the demand weights) is worked out again. Arrays are copies.
        """
        arrays = {name: getattr(self, name).copy() for name in (*CAR_ARRAYS, "acc_hist", *ARM_ARRAYS)}
        logs = {}
        for name in EVENT_LOGS:
            arrays[name], logs[name] = _pack_log(getattr(self, name))
        # Each queue's arrival times (and vehicle types), all entrances in one array, and how many each has.
        arrays["queue_times"] = np.array([t for fifo in self.queue_times for t in fifo], dtype=float)
        arrays["queue_times_count"] = np.array([len(fifo) for fifo in self.queue_times], dtype=np.int64)
        arrays["queue_trucks"] = np.array([truck for fifo in self.queue_trucks for truck in fifo], dtype=bool)
        arrays["queue_trucks_count"] = np.array([len(fifo) for fifo in self.queue_trucks], dtype=np.int64)
        k = len(self.entrance_log)
        arrays["entrance_log_time"] = np.array([entry[0] for entry in self.entrance_log], dtype=float)
        for i, (part, dtype) in enumerate((("arrived", np.int64), ("entered", np.int64), ("busy", float)), start=1):
            arrays[f"entrance_log_{part}"] = np.array([entry[i] for entry in self.entrance_log],
                                                      dtype=dtype).reshape(k, self.n_arms)
        arrays["lanes_order"] = np.concatenate([order for order, _ in self._lanes]).astype(np.int64)
        arrays["lanes_theta"] = np.concatenate([th for _, th in self._lanes]).astype(float)
        arrays["lanes_count"] = np.array([order.size for order, _ in self._lanes], dtype=np.int64)
        meta = {
            "version": STATE_VERSION,
            "config": {name: value.item() if isinstance(value, np.generic) else value
                       for name, value in asdict(self.cfg).items()},
            "seed": self.seed, "time": self.time, "steps": self.steps, "generation": self.generation,
            "cap": self.cap, "hist_ptr": self._hist_ptr, "next_uid": self._next_uid,
            "next_fill": self._next_fill, "next_prune": self._next_prune,
            "pending": [bool(truck) for truck in self._pending],
            "counters": {name: int(getattr(self, name)) for name in COUNTERS},
            "entry_wait_total": float(self.entry_wait_total),
            "logs": logs,
            "generators": {name: getattr(self, name).bit_generator.state for name in GENERATORS},
        }
        return {"meta": meta, "arrays": arrays}

    @classmethod
    def from_state(cls, state: dict) -> RingSim:
        """The simulation a :meth:`state_dict` was taken from, carrying on bit for bit as it would have.

        Raises :class:`StateError` saying what is wrong if ``state`` is
        incomplete, damaged or from another version (e.g. settings this
        version does not know, or arrays of the wrong shape).
        """
        try:
            return cls._from_state(state["meta"], state["arrays"])
        except StateError:
            raise
        except KeyError as e:
            raise StateError(f"the state lacks {e.args[0]!r}") from e
        except (TypeError, ValueError, AttributeError, IndexError, OverflowError) as e:
            raise StateError(f"the state does not fit together ({e})") from e

    @classmethod
    def _from_state(cls, meta: dict, arrays: dict) -> RingSim:
        version = meta.get("version")
        if version != STATE_VERSION:
            raise StateError(f"simulation state version {version!r}; this version reads {STATE_VERSION}")
        config, names = dict(meta["config"]), {f.name for f in fields(SimConfig)}
        unknown, missing = sorted(set(config) - names), sorted(names - set(config))
        if unknown or missing:
            raise StateError("settings of another version (" + "; ".join(
                [f"unknown: {', '.join(unknown)}"] * bool(unknown) + [f"missing: {', '.join(missing)}"] * bool(missing))
                + ")")
        self = cls.__new__(cls)
        self.cfg = cfg = SimConfig(**config)  # validates the values
        self.seed = int(meta["seed"])
        for name in GENERATORS:
            setattr(self, name, _generator(meta["generators"][name]))
        self._update_effective()
        self.time, self.steps, self.generation = float(meta["time"]), int(meta["steps"]), int(meta["generation"])
        if not (np.isfinite(self.time) and self.time >= 0 and self.steps >= 0 and self.generation >= 0):
            raise StateError("the clock (time, steps or generation) is out of range")
        self._geometry()
        n = self.n_arms
        for name, dtype in ARM_ARRAYS.items():
            setattr(self, name, _array(arrays, name, (n,), dtype))
        self.demand_weight = self._demand_weights()
        for name, dtype in (("queue_times", float), ("queue_trucks", bool)):
            parts = _split(_array(arrays, name, None, dtype), _array(arrays, f"{name}_count", (n,), np.int64), name)
            setattr(self, name, [deque(part.tolist()) for part in parts])
        if (self.queue < 0).any() or any(len(fifo) != q for fifo, q in zip(self.queue_times, self.queue)) \
                or any(len(fifo) > q for fifo, q in zip(self.queue_trucks, self.queue)):
            raise StateError("the queues do not match their arrival times and vehicle types")
        for name in COUNTERS:
            setattr(self, name, int(meta["counters"][name]))
        self.entry_wait_total = float(meta["entry_wait_total"])
        for name in EVENT_LOGS:
            setattr(self, name, _unpack_log(name, _array(arrays, name, None, float), list(meta["logs"][name]),
                                            plain=name == "lane_change_log"))
        when = _array(arrays, "entrance_log_time", None, float)
        k = when.shape[0] if when.ndim == 1 else -1
        parts = [_array(arrays, f"entrance_log_{part}", (k, n), dtype)
                 for part, dtype in (("arrived", np.int64), ("entered", np.int64), ("busy", float))]
        self.entrance_log = deque((t, *(row.copy() for row in rows)) for t, *rows in zip(when.tolist(), *parts))
        self._next_uid, self._next_fill = int(meta["next_uid"]), float(meta["next_fill"])
        self._pending = list(meta["pending"])
        if not all(isinstance(truck, bool) for truck in self._pending):
            raise StateError("pending holds something other than true/false")
        self._next_prune = float(meta["next_prune"])
        if not (np.isfinite(self._next_fill) and np.isfinite(self._next_prune)):
            raise StateError("the fill or prune timer is not a finite time")
        self._hist_len = int(round(MAX_REACTION / cfg.dt)) + 1
        self._hist_ptr = int(meta["hist_ptr"])
        cap = int(meta["cap"])
        if cap < 1 or np.shape(arrays.get("alive")) != (cap,) or not 0 <= self._hist_ptr < self._hist_len:
            raise StateError("the car arrays' size or history pointer is out of range")  # (before allocating)
        self._alloc(cap)
        for name in CAR_ARRAYS:
            getattr(self, name)[:] = _array(arrays, name, (cap,), getattr(self, name).dtype)
        self.acc_hist[:] = _array(arrays, "acc_hist", (self._hist_len, cap), float)
        counts = _array(arrays, "lanes_count", (self.n_lanes,), np.int64)
        order = _array(arrays, "lanes_order", None, np.int64)
        if order.ndim != 1 or ((order < 0) | (order >= cap)).any():
            raise StateError("the lane index holds slots that do not exist")
        self._lanes = list(zip(_split(order, counts, "lanes_order"),
                               _split(_array(arrays, "lanes_theta", order.shape, float), counts, "lanes_theta")))
        lanes = self.lane[self.alive]
        if ((lanes < 0) | (lanes >= self.n_lanes)).any():
            raise StateError("a car is on a lane the ring does not have")
        return self

    # ----------------------------------------------------------- geometry
    @property
    def total_lane_length(self) -> float:
        return float(self.lane_len.sum())

    @property
    def mean_length(self) -> float:
        """[m] Expected vehicle length for the truck share (car_length when there are no trucks)."""
        cfg = self.cfg
        return cfg.car_length + cfg.truck_share * (cfg.truck_length - cfg.car_length)

    @property
    def jam_density(self) -> float:
        """Densest packing [veh/km/lane]: bumper to bumper at the standstill gap (vehicles of the mean length)."""
        return 1000.0 / (self.mean_length + self.cfg.min_gap)

    def capacity(self) -> int:
        """Most vehicles the lanes hold at jam density (of the mean length: a random mix with more
        trucks than average can need a few places more)."""
        pitch = self.mean_length + self.cfg.min_gap
        return int(np.floor(self.lane_len / pitch).sum())

    def target_cars(self) -> int:
        density = min(self.cfg.density, self.jam_density)
        return min(int(round(density * self.total_lane_length / 1000.0)), self.capacity())

    def lane_radius(self, lat) -> np.ndarray:
        return self.cfg.radius + self.cfg.lane_width * np.asarray(lat)

    # ------------------------------------------------------------- demand
    @property
    def open_demand(self) -> bool:
        """True in open mode: cars arrive at random and leavers are not replaced."""
        return self.cfg.traffic_mode == "open"

    @property
    def queue_limit(self) -> int:
        """Most cars that can wait at one entrance in the current mode."""
        return int(self.cfg.max_queue) if self.open_demand else MAX_QUEUE

    def _demand_weights(self) -> np.ndarray:
        """Each entrance's share of the demand, with mean 1 over the entrances.

        A main road runs along the axis through arm 0 (the bottom one) and, with
        an even number of arms, the opposite arm n/2: an arm's weight is
        1 + demand_imbalance * cos(2 * its angle from arm 0), so arms on the axis
        get the most and arms at right angles to it the least (none at
        imbalance 1). With one or two arms every arm is on the axis.
        """
        n = self.n_arms
        if n == 0:
            return np.zeros(0)
        w = 1.0 + self.cfg.demand_imbalance * np.cos(2 * TWO_PI * np.arange(n) / n)
        return w / w.mean()

    def demand_factor(self, t=None):
        """The demand profile at time ``t`` (default: now) as a multiple of the
        entrance chance; its mean over each whole PEAK_PERIOD is exactly 1
        (over other windows see :func:`mean_demand_factor`).

        "peak" is 1 - PEAK_SWING * cos(2 pi t / PEAK_PERIOD): half the demand at
        t = 0, 10, 20... min, 1.5 times it at 5, 15, 25... min.
        """
        t = self.time if t is None else t
        if self.cfg.demand_profile == "peak":
            return 1.0 - PEAK_SWING * np.cos(TWO_PI * np.asarray(t, dtype=float) / PEAK_PERIOD)
        return np.ones_like(np.asarray(t, dtype=float)) if np.ndim(t) else 1.0

    def arrival_rates(self) -> np.ndarray:
        """[1/s] Mean arrival rate at each entrance right now (open mode)."""
        return self.cfg.entry_rate * float(self.demand_factor()) * self.demand_weight

    # ------------------------------------------------- vehicles and drivers
    def _update_effective(self) -> None:
        """The effective settings for the road condition, and whether vehicles and drivers differ.

        ``uniform_length``: every vehicle is a car (no trucks), so lengths are
        the one ``car_length``. ``alike``: also every driver keeps the same
        time gap and reaction time. Then the dynamics use plain numbers, which
        keeps the arithmetic, and the speed, of a model without trucks or
        driver spread; otherwise they use each vehicle's own values.
        """
        cfg = self.cfg
        self.eff = effective(cfg)
        self.uniform_length = cfg.truck_share == 0
        self.alike = self.uniform_length and cfg.headway_std == 0 and cfg.reaction_std == 0

    def _draw_trucks(self, k: int) -> np.ndarray:
        """Whether each of ``k`` new vehicles is a truck (chance ``truck_share``, from rng_drivers)."""
        return self.rng_drivers.random(k) < self.cfg.truck_share

    def _fill_types(self, k: int) -> np.ndarray:
        """Whether each of the next ``k`` vehicles the density control adds is a truck: first the
        ones it could not place before (oldest first), then new draws."""
        waiting = self._pending[:k]
        return np.concatenate((np.array(waiting, dtype=bool), self._draw_trucks(k - len(waiting))))

    def _set_driving(self, i) -> None:
        """Each driver's own time gap, max acceleration and reaction delay (in steps), from their
        temperaments, their vehicle and the effective settings.

        Time gap: Following time gap + z * spread, at least HEADWAY_FLOOR of the
        setting, times TRUCK_HEADWAY_FACTOR in a truck. Reaction time: Reaction
        time + z * spread within 0-MAX_REACTION, rounded to whole steps.
        """
        cfg, eff = self.cfg, self.eff
        own_gap = np.maximum(HEADWAY_FLOOR * eff.time_headway, eff.time_headway + self.z_gap[i] * eff.headway_std)
        truck = self.truck[i]
        self.headway[i] = np.where(truck, TRUCK_HEADWAY_FACTOR * own_gap, own_gap)
        self.accel[i] = np.where(truck, TRUCK_ACCEL_FACTOR * eff.max_accel, eff.max_accel)
        reaction = np.clip(cfg.reaction_time + self.z_react[i] * cfg.reaction_std, 0.0, MAX_REACTION)
        self.delay[i] = np.rint(reaction / cfg.dt).astype(np.int64)

    def _driving(self, slots):
        """(max acceleration, time gap) of the drivers in ``slots`` for the IDM: plain numbers
        while every driver is alike, else arrays aligned with ``slots``."""
        if self.alike:
            return self.eff.max_accel, self.eff.time_headway
        return self.accel[slots], self.headway[slots]

    def _pair_length(self, behind, ahead):
        """[m] Distance between the centres of two vehicles bumper to bumper: half of each one's length."""
        if self.uniform_length:
            return self.cfg.car_length
        return 0.5 * (self.length[behind] + self.length[ahead])

    def _extra_length(self, slots) -> np.ndarray:
        """[m] How much longer than a car each vehicle is (0 for a car, exactly)."""
        return np.where(self.truck[slots], self.cfg.truck_length - self.cfg.car_length, 0.0)

    # ------------------------------------------------------- driver model
    def desired_speed(self, idx, lane=None) -> np.ndarray:
        """Free-road desired speed, capped by comfortable cornering on the car's lane
        (or on ``lane`` if given). Trucks want TRUCK_SPEED_FACTOR of their driver's.

        Floored at half the mean: a huge spread would otherwise create drivers
        who barely move, i.e. moving bottlenecks rather than phantom jams.
        """
        eff = self.eff
        v0 = np.maximum(0.5 * eff.mean_speed, eff.mean_speed + self.z[idx] * eff.speed_std)
        if not self.uniform_length:
            v0 = np.where(self.truck[idx], TRUCK_SPEED_FACTOR * v0, v0)
        lanes = self.lane[idx] if lane is None else lane
        return np.minimum(v0, np.sqrt(eff.lateral_accel * self.radii)[lanes])

    def target_speed(self, idx) -> np.ndarray:
        """Desired speed, also slowing down comfortably to ``turn_speed`` before an exit.

        Only cars already on the outer lane slow for their exit: a leaver still
        on an inner lane keeps pace with traffic so it can merge outwards.
        """
        cfg = self.cfg
        v = self.desired_speed(idx)
        to_exit = (self.exit_at[idx] - self.travel[idx]) * self.r_out
        approach = np.sqrt(cfg.turn_speed**2 + 2 * self.eff.comfort_decel * np.maximum(to_exit - 5.0, 0.0))
        slowing = np.isfinite(to_exit) & (self.lane[idx] == self.outer)
        return np.where(slowing, np.minimum(v, approach), v)

    def _free_acc(self, v, v0, a) -> np.ndarray:
        """IDM free-road acceleration towards ``v0`` for max acceleration ``a`` (a number, or
        per car). Above the desired speed it brakes gently (improved IDM) instead of the
        plain IDM's hard (v/v0)^4."""
        b = self.eff.comfort_decel
        v = np.asarray(v, dtype=float)
        v0 = np.maximum(np.asarray(v0, dtype=float), 0.1)
        free = a * (1 - (v / v0) ** 4)
        over = v > v0
        if over.any():
            a_over = a[over] if np.ndim(a) else a
            free[over] = -b * (1 - (v0[over] / v[over]) ** (4 * a_over / b))
        return free

    def _interaction(self, v, gap, v_lead, a, T) -> np.ndarray:
        """IDM braking towards a leader ``gap`` metres (bumper to bumper) ahead, for drivers
        with max acceleration ``a`` and time gap ``T`` (numbers, or aligned with ``v``)."""
        cfg = self.cfg
        b = self.eff.comfort_decel
        s_star = cfg.min_gap + np.maximum(0.0, v * T + v * (v - v_lead) / (2 * np.sqrt(a * b)))
        return a * (s_star / np.maximum(gap, 0.1)) ** 2

    def _idm(self, v, v0, gap, v_lead, slots) -> np.ndarray:
        """IDM acceleration towards one leader for the drivers in ``slots`` (aligned with
        ``v``). ``gap`` is bumper to bumper; leaders beyond the reaction distance (or
        ``inf``) are ignored."""
        a, T = self._driving(slots)
        v = np.asarray(v, dtype=float)
        seen = np.isfinite(gap) & (gap <= self.cfg.reaction_distance)
        return self._free_acc(v, v0, a) - np.where(seen, self._interaction(v, np.where(seen, gap, 1.0), v_lead,
                                                                           a, T), 0.0)

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
        Each driver uses their own acceleration and time gap; the distance to
        the j-th car ahead counts the real lengths of the cars in between.
        """
        cfg = self.cfg
        a, T = self._driving(idx)
        seen = np.isfinite(g) & (g <= cfg.reaction_distance)
        v_lead = np.where(ld >= 0, self.v[np.maximum(ld, 0)], v)
        braking = np.where(seen, self._interaction(v, np.where(seen, g, 1.0), v_lead, a, T), 0.0)
        weight = seen.astype(float)
        cars_ahead = np.bincount(self.lane[idx], minlength=self.n_lanes)[self.lane[idx]] - 1
        car, gaps = np.maximum(ld, 0), g.copy()
        between = 0.0  # [m] summed lengths of the cars passed so far (vehicles of mixed lengths)
        for j in range(2, ANTICIPATION + 1):
            gaps = gaps + gap[car]  # summed bumper gaps to the j-th car ahead
            if not self.uniform_length:
                between = between + self.length[car]
            car = lead[car]
            reach = gaps + ((j - 1) * cfg.car_length if self.uniform_length else between)
            sees = seen & (j <= cars_ahead) & (reach <= cfg.reaction_distance)
            if not sees.any():
                break
            w = 1.0 / j**2
            braking += np.where(sees, w * self._interaction(v, gaps / j, self.v[car], a, T), 0.0)
            weight += np.where(sees, w, 0.0)
        return self._free_acc(v, v0, a) - np.where(weight > 0, braking / np.maximum(weight, 1e-12), 0.0)

    def equilibrium_speed(self, gap, v0, slots=None) -> np.ndarray:
        """Steady-state IDM speed for a bumper gap (used to start cars smoothly), for the
        drivers in ``slots`` (aligned with ``gap``; default: the effective Following time gap)."""
        cfg = self.cfg
        T = self.eff.time_headway if slots is None else self._driving(slots)[1]
        gap = np.asarray(gap, dtype=float)
        v0 = np.maximum(np.broadcast_to(np.asarray(v0, dtype=float), gap.shape), 0.1)
        lo, hi = np.zeros_like(gap), v0.copy()
        seen = gap <= cfg.reaction_distance
        for _ in range(40):  # bisection on the IDM steady state (acceleration = 0)
            mid = 0.5 * (lo + hi)
            f = 1 - (mid / v0) ** 4 \
                - np.where(seen, ((cfg.min_gap + mid * T) / np.maximum(gap, 0.1)) ** 2, 0)
            lo, hi = np.where(f > 0, mid, lo), np.where(f > 0, hi, mid)
        return np.where(gap > cfg.min_gap, lo, 0.0)

    # ------------------------------------------------------ neighbourhood
    def _index_lanes(self, idx):
        """Sort live cars by angle within each lane.

        Returns full-size arrays ``lead`` (slot of the car ahead, -1 if none)
        and ``gap`` (bumper gap to it [m]: the distance between their centres
        less half of each one's length); a car alone in its lane follows
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
        gap[order] = dth * self.radii[lanes] - self._pair_length(order, nxt)
        self._lanes = [(order[a:b], th[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]
        return lead, gap

    def bumper_gaps(self) -> tuple[np.ndarray, np.ndarray]:
        """(slots of the live cars, bumper gap from each to the car ahead in its lane [m])."""
        idx = np.flatnonzero(self.alive)
        _, gap = self._index_lanes(idx)
        return idx, gap[idx]

    def _neighbours_in(self, lane: int, angles: np.ndarray, length=None):
        """Leader/follower slots and bumper gaps for vehicles ``length`` m long (a number or
        aligned with ``angles``; default a car) centred at ``angles`` in ``lane``."""
        order, th = self._lanes[lane]
        n = angles.size
        if order.size == 0:
            return (np.full(n, -1), np.full(n, np.inf), np.full(n, -1), np.full(n, np.inf))
        pos = np.searchsorted(th, angles)
        ld = order[pos % order.size]
        fl = order[(pos - 1) % order.size]
        r = self.radii[lane]
        if self.uniform_length:
            front_len = back_len = self.cfg.car_length
        else:
            own = self.cfg.car_length if length is None else length
            front_len, back_len = 0.5 * (own + self.length[ld]), 0.5 * (own + self.length[fl])
        front = ((self.theta[ld] - angles) % TWO_PI) * r - front_len
        back = ((angles - self.theta[fl]) % TWO_PI) * r - back_len
        return ld, front, fl, back

    def _neighbours(self, lanes: np.ndarray, angles: np.ndarray, length=None):
        """:meth:`_neighbours_in` for positions spread over several lanes (``length`` a number
        or aligned with ``angles``)."""
        n = angles.size
        out = (np.full(n, -1, dtype=np.int64), np.full(n, np.inf), np.full(n, -1, dtype=np.int64), np.full(n, np.inf))
        for lane in range(self.n_lanes):
            m = lanes == lane
            if m.any():
                part_len = length[m] if np.ndim(length) else length
                for arr, part in zip(out, self._neighbours_in(lane, angles[m], part_len)):
                    arr[m] = part
        return out

    # ----------------------------------------------------------- spawning
    def _spawn(self, lanes, angles, speeds, lat=None, trucks=None) -> np.ndarray:
        """New vehicles centred at ``angles``; ``trucks`` says which are trucks (default: drawn).
        Their drivers' temperaments come from rng_drivers (nothing extra from the main stream)."""
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
        self.truck[i] = self._draw_trucks(k) if trucks is None else trucks
        self.length[i] = np.where(self.truck[i], self.cfg.truck_length, self.cfg.car_length)
        self.z_react[i], self.z_gap[i] = np.clip(self.rng_drivers.standard_normal((2, k)),
                                                 -TEMPERAMENT_CLIP, TEMPERAMENT_CLIP)
        self._set_driving(i)
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
        """Lay out ``n`` vehicles by lane length in the init_spacing pattern, at their steady speeds.

        Positions are vehicle centres. Trucks are drawn first and each pattern
        makes room for their extra length (``extra``, 0 for a car): spacings
        grow by half the extra length of the two vehicles they join, so the
        bumper gaps follow the pattern whatever the mix.
        """
        if n <= 0:
            return
        cfg = self.cfg
        pitch = cfg.car_length + cfg.min_gap  # of a car
        share = n * self.lane_len / self.lane_len.sum()
        counts = np.floor(share).astype(int)
        counts[np.argsort(counts - share)[: n - counts.sum()]] += 1
        counts = np.minimum(counts, np.floor(self.lane_len / (self.mean_length + cfg.min_gap)).astype(int))
        # A clustered start is one packed block across all lanes, so it starts
        # at the same angle in every lane.
        start = self.rng.uniform(0, TWO_PI) if cfg.init_spacing == "clustered" else 0.0
        for lane, k in enumerate(counts):
            if k == 0:
                continue
            C, r = self.lane_len[lane], self.radii[lane]
            trucks = self._draw_trucks(k)
            extra = np.where(trucks, cfg.truck_length - cfg.car_length, 0.0)
            while not self.uniform_length and k and k * pitch + extra[:k].sum() > C:
                k -= 1  # more trucks than this lane's share holds: the last ones do not fit here
            if k == 0:
                continue
            trucks, extra = trucks[:k], extra[:k]
            pair = 0.5 * (extra + np.roll(extra, -1))  # half the extra length of each vehicle and the next
            if cfg.init_spacing == "random":
                offset = self.rng.uniform(0, C)
                gaps = pitch + (C - k * pitch - extra.sum()) * self.rng.dirichlet(np.ones(k)) + pair
                s = offset + np.concatenate(([0.0], np.cumsum(gaps[:-1])))
            elif cfg.init_spacing == "clustered":
                s = start * r + min(pitch * 1.05, (C - extra.sum()) / k) * np.arange(k) \
                    + np.concatenate(([0.0], np.cumsum(pair[:-1])))
            else:  # uniform
                s = self.rng.uniform(0, C) + C * np.arange(k) / k + self._even_shift(extra)
            self._spawn(np.full(k, lane), s / r, np.zeros(k), trucks=trucks)
        idx = np.flatnonzero(self.alive)
        _, gap = self._index_lanes(idx)
        self.v[idx] = self.equilibrium_speed(gap[idx], self.desired_speed(idx), idx)

    @staticmethod
    def _even_shift(extra: np.ndarray) -> np.ndarray:
        """[m] Shifts that turn evenly spaced centres into even bumper gaps for vehicles ``extra`` m
        longer than a car, in driving order round a lane: each moves on by half the extra length of
        every pair behind it, less the mean. All 0 (exactly) when every vehicle is a car."""
        pair = 0.5 * (extra + np.roll(extra, -1))
        return np.concatenate(([0.0], np.cumsum(pair[:-1] - extra.mean())))

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
        """Slot of the live car closest to world point (x, y), if within ``max_dist``.

        A vehicle longer than a car counts as near all along its body: the
        distance along its heading is shortened by half its extra length.
        """
        idx = np.flatnonzero(self.alive)
        if idx.size == 0:
            return None
        cx, cy = self.positions(idx)
        dx, dy = x - cx, y - cy
        th = self.theta[idx]
        along = dx * -np.sin(th) + dy * np.cos(th)  # along the driving direction (anticlockwise)
        across = dx * np.cos(th) + dy * np.sin(th)
        d = np.hypot(np.maximum(np.abs(along) - 0.5 * self._extra_length(idx), 0.0), across)
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
        if not self.uniform_length:  # a truck turns no further than to put its front half a lane across from its back
            cap = np.where(self.truck[idx], np.arctan(0.5 * self.cfg.lane_width / self.length[idx]), np.pi / 2)
            tilt = np.clip(tilt, -cap, cap)
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
            "truck": self.truck[idx],
            "length": self.length[idx],
            "width": np.where(self.truck[idx], TRUCK_WIDTH, self.cfg.car_width),
        }

    # ------------------------------------------------------------ dynamics
    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            self._step()

    def _step(self) -> None:
        dt = self.cfg.dt
        open_demand = self.open_demand
        idx = np.flatnonzero(self.alive)
        if idx.size:
            self._move(idx)
            self._exits()
            self._lane_changes(np.flatnonzero(self.alive))
        if open_demand:
            self._arrivals()
        self._entries()
        if not open_demand:
            # Last, so that after every step the population matches the density
            # setting (density changes, and leavers that found their queue full).
            self._sync_population()
        # Whole arrays: cheaper than masking, and _spawn resets free slots.
        self.lc_cool -= dt
        self.entry_cool -= 1
        self.lat += np.clip(self.lane - self.lat, -dt / LC_DURATION, dt / LC_DURATION)
        self.time += dt
        self.steps += 1
        if self.time >= self._next_prune:
            self._prune_logs()

    def _prune_logs(self) -> None:
        """Drop events older than LOG_HORIZON and log the entrances' counters (once per simulated second)."""
        self._next_prune = self.time + 1.0
        cutoff = self.time - LOG_HORIZON
        if self.n_arms:
            self.entrance_log.append((self.time, self.arm_arrived.copy(), self.arm_entered.copy(),
                                      self.queue_busy.copy()))
        for log in (self.detector_log, self.exit_log, self.emergency_log, self.stop_log, self.arrival_log,
                    self.entry_log, self.blocked_log, self.gave_up_log, self.entrance_log):
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
        # The driver model asking for more than the brakes can give (on this
        # road): harsh braking, typically over-reaction after a late or delayed response.
        max_decel = self.eff.max_decel
        over_braking = acc_now < -max_decel

        # Reaction time: act on the acceleration decided ``delay`` steps ago (each driver their own).
        if self.alike:
            delay = int(round(min(cfg.reaction_time, MAX_REACTION) / dt))
        else:
            delay = self.delay[idx]
        self._hist_ptr = (self._hist_ptr + 1) % self._hist_len
        self.acc_hist[self._hist_ptr, idx] = acc_now
        acc = self.acc_hist[(self._hist_ptr - delay) % self._hist_len, idx].copy()

        braking = self.brake_until[idx] > self.time
        if cfg.driver_noise > 0:
            f = np.exp(-dt / NOISE_TAU)
            self.noise[idx] = f * self.noise[idx] + cfg.driver_noise * np.sqrt(1 - f * f) * \
                self.rng.standard_normal(idx.size)
            acc += self.noise[idx]
        # Physical braking limit; a car told to brake brakes hard, but no harder than the road allows.
        acc = np.where(braking, -min(PERTURB_DECEL, max_decel), np.maximum(acc, -max_decel))

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
        # the braking limit, or the no-overlap constraint has to stop the car
        # harder than that. Counted per episode, which ends after EMERGENCY_HOLD s.
        emergency = over_braking | ((v - v_new) / dt > max_decel + 1e-6)
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
        b = self.eff.comfort_decel
        leavers = idx[m]
        nl, front, nf, back = self._neighbours(self.lane[leavers] + 1, self.theta[leavers],
                                               None if self.uniform_length else self.length[leavers])
        v_front = np.where(nl >= 0, self.v[np.maximum(nl, 0)], v[m])
        follow = self._idm(v[m], v0[m], front, v_front, leavers)
        acc[m] = np.minimum(acc[m], np.maximum(follow, -1.5 * b))
        has = nf >= 0
        if has.any():
            pos = np.empty(self.cap, dtype=np.int64)
            pos[idx] = np.arange(idx.size)
            f = pos[nf[has]]
            yield_acc = self._idm(v[f], v0[f], back[has], v[m][has], idx[f])
            np.minimum.at(acc, f, np.maximum(yield_acc, -b))

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
        if self.open_demand:
            return  # leavers are gone for good; new cars come from _arrivals
        # Each leaver is replaced at a random entrance; if that queue is full the
        # replacement is slotted into the ring instead (by _sync_population).
        for arm in self.rng.integers(self.n_arms, size=leaving.size):
            if self.queue[arm] < MAX_QUEUE:
                self.queue[arm] += 1
                self.queue_times[arm].append(self.time)
                self.queue_trucks[arm].append(bool(self._draw_trucks(1)[0]))
                self.n_arrived += 1
                self.arm_arrived[arm] += 1
                self.arrival_log.append((self.time, int(arm)))

    def _arrivals(self) -> None:
        """Open mode: cars arrive at each entrance at random (Poisson, from the demand
        stream) and join its queue; those that find it full are turned away (blocked)."""
        if self.n_arms == 0 or self.cfg.entry_rate <= 0:
            return
        arrived = self.rng_demand.poisson(self.arrival_rates() * self.cfg.dt)
        if not arrived.any():
            return
        limit, now = int(self.cfg.max_queue), self.time
        for arm in np.flatnonzero(arrived).tolist():
            k = int(arrived[arm])
            join = min(k, max(limit - int(self.queue[arm]), 0))
            self.queue[arm] += join
            self.queue_times[arm].extend([now] * join)
            self.queue_trucks[arm].extend(self._draw_trucks(join).tolist())
            self.arrival_log.extend([(now, arm)] * k)
            if k > join:
                self.n_blocked += k - join
                self.blocked_log.extend([(now, arm)] * (k - join))
        self.n_arrived += int(arrived.sum())
        self.arm_arrived += arrived

    def _drop_queued(self, arm: int, k: int) -> None:
        """Take the ``k`` latest arrivals out of an entrance's queue: they give up
        (counted in n_gave_up and gave_up_log, so every arrival stays accounted for)."""
        k = int(k)
        if k > 0:
            self.queue[arm] -= k
            for _ in range(k):
                self.queue_times[arm].pop()
            trucks = self.queue_trucks[arm]
            while len(trucks) > self.queue[arm]:  # (a queue edited by hand may hold fewer types)
                trucks.pop()
            self.n_gave_up += k
            self.gave_up_log.extend([(self.time, int(arm))] * k)

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
        own_length = cfg.car_length if self.uniform_length else self.length[c]
        nl, front, nf, back = self._neighbours(target[opt], self.theta[c], own_length)
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
        acc[need] = self._idm(self.v[need], v0[need], gap[need], self.v[lead[need]], need)

        vc = self.v[c]
        v_front = np.where(nl >= 0, self.v[np.maximum(nl, 0)], vc)
        # Judge the target lane with the speed this driver could do there
        # (its cornering cap differs from the current lane's).
        a_self = self._idm(vc, self.desired_speed(c, target[opt]), front, v_front, c)
        a_f_new = np.where(has_f, self._idm(self.v[f], v0[f], back, vc, f), 0.0)
        a_f_old = np.where(has_f, acc[f], 0.0)
        # Once we are gone, the old follower's gap reaches over our length to our old leader.
        a_o_new = np.where(has_o, self._idm(self.v[os_], v0[os_], gap[os_] + own_length + gap[c],
                                            self.v[lead[c]], os_), 0.0)
        a_o_old = np.where(has_o, acc[os_], 0.0)
        forced = mandatory[k] & (opt < n)  # a leaver must move out
        # Braking a change may force on anyone: more for a leaver, never beyond what the road allows.
        b_safe = np.minimum(self.eff.comfort_decel * np.where(mandatory[k], 2.5, 1.5), self.eff.max_decel)
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

    def follow_up_steps(self) -> int:
        """Steps an entrance waits after letting a car in: the next car may go on the
        first step after ``follow_up`` (scaled by the road condition) has passed (1 s takes 11
        steps of 0.1 s)."""
        return int(np.floor(self.eff.follow_up / self.cfg.dt + 1e-9)) + 1

    def _entries(self) -> None:
        """Queued cars merge in where the gap allows, at most one per entrance per follow-up time."""
        if self.n_arms == 0 or not self.queue.any():
            return
        self.queue_busy += self.cfg.dt * (self.queue > 0)  # entrances with cars waiting this step
        ready = np.flatnonzero((self.queue > 0) & (self.entry_cool <= 0))
        if ready.size == 0:
            return
        # Only the outer lane's index is needed (the others are rebuilt before their next use).
        cars = np.flatnonzero(self.alive & (self.lane == self.outer))
        order = cars[np.argsort(self.theta[cars], kind="stable")]
        self._lanes[self.outer] = (order, self.theta[order])
        follow_up = self.follow_up_steps()
        arms = self.rng.permutation(ready)
        while arms.size:
            # Check the remaining arms together; after an entry, re-check the ones after it.
            ok, v_in = self._entry_ok(self.entry_angle[arms], self._head_trucks(arms))
            hits = np.flatnonzero(ok)
            if hits.size == 0:
                break
            k = hits[0]
            arm, angle = arms[k], self.entry_angle[arms[k]]
            trucks = self.queue_trucks[arm]
            truck = trucks.popleft() if trucks else False  # (a queue edited by hand may lack the types)
            slot = self._spawn(np.array([self.outer]), np.array([angle]), v_in[k:k + 1],
                               lat=np.array([self.outer + 1.0]), trucks=np.array([truck]))
            self.entered_at[slot] = self.time
            self.queue[arm] -= 1
            wait = self.time - self.queue_times[arm].popleft()  # first come, first served
            self.entry_wait_total += wait
            self.entry_log.append((self.time, int(arm), wait))
            self.entry_cool[arm] = follow_up
            self.n_entered += 1
            self.arm_entered[arm] += 1
            # Keep the outer lane's sorted index current for the next arm.
            order, th = self._lanes[self.outer]
            pos = np.searchsorted(th, angle)
            self._lanes[self.outer] = (np.insert(order, pos, slot[0]), np.insert(th, pos, angle % TWO_PI))
            arms = arms[k + 1:]

    def _head_trucks(self, arms) -> np.ndarray | None:
        """Whether the vehicle at the head of each of these entrances' queues is a truck
        (None when every vehicle is a car)."""
        if self.uniform_length:
            return None
        return np.array([bool(self.queue_trucks[arm][0]) if self.queue_trucks[arm] else False for arm in arms])

    def _entry_ok(self, angles: np.ndarray, trucks: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Whether a vehicle may enter the outer lane at each of ``angles`` (``trucks``: which are
        trucks; None = all cars), and at what speed."""
        cfg, eff = self.cfg, self.eff
        if trucks is None:
            length, v_in = cfg.car_length, np.full(angles.size, min(cfg.turn_speed, eff.mean_speed))
        else:
            length = np.where(trucks, cfg.truck_length, cfg.car_length)
            v_in = np.minimum(cfg.turn_speed, eff.mean_speed * np.where(trucks, TRUCK_SPEED_FACTOR, 1.0))
        nl, front, nf, back = self._neighbours_in(self.outer, angles, length)
        if nl[0] < 0:  # empty outer lane: room for anything shorter than the lane
            return np.broadcast_to(length + cfg.min_gap <= self.lane_len[self.outer], angles.shape).copy(), v_in
        vf = self.v[nf]
        follower_ok = self._idm(vf, self.desired_speed(nf), back, v_in, nf) >= -eff.comfort_decel
        ok = ((front >= cfg.min_gap + 1.0) & (back >= cfg.min_gap + 1.0)
              & (back >= eff.entry_gap * vf) & follower_ok)
        return ok, np.where(front < 15.0, np.minimum(v_in, self.v[nl]), v_in)

    def _sync_population(self) -> None:
        """Closed mode: match the number of cars (on the ring and queued) to the density setting."""
        target = self.target_cars()
        n_ring = int(self.alive.sum())
        surplus = n_ring + int(self.queue.sum()) - target
        if surplus >= 0:
            self._pending.clear()  # nobody missing: nothing left to place
        if surplus > 0:
            for arm in np.argsort(-self.queue):
                take = min(self.queue[arm], surplus)
                self._drop_queued(arm, take)
                surplus -= take
            if surplus > 0:
                victims = self.rng.choice(np.flatnonzero(self.alive), size=min(surplus, n_ring), replace=False)
                self.alive[victims] = False
        elif surplus < 0 and self.time >= self._next_fill:
            got, full = self._insert_into_gaps(-surplus)
            if got < -surplus:
                # Respacing is for a ring where no gap takes even a car (near jam density). A truck
                # that fits no gap yet waits for one instead: it is the next vehicle placed.
                if not full or (not self._respace_lanes(-surplus - got) and got == 0):
                    self._next_fill = self.time + 1.0  # retry in a second

    def _sync_ring(self) -> None:
        """Open mode, when Density is changed live (or the start left vehicles out): bring
        the cars on the ring to the density setting at once (queues untouched); after
        that arrivals and exits decide again. Trucks that fit no gap are placed by respacing the lanes (a
        bulk raise teleports cars anyway), so the vehicle mix holds; only vehicles
        that fit no lane at all (a full ring) are left out."""
        target = self.target_cars()
        surplus = int(self.alive.sum()) - target
        if surplus > 0:
            self.alive[self.rng.choice(np.flatnonzero(self.alive), size=surplus, replace=False)] = False
        while surplus < 0:  # every round adds at least one car, or stops
            got, _ = self._insert_into_gaps(-surplus)
            # Nothing fitted a gap: no gap takes a car, or only trucks are left to place.
            if got == 0 and not self._respace_lanes(-surplus):
                break
            surplus = int(self.alive.sum()) - target
        self._pending.clear()  # (once only: open mode keeps no vehicles waiting to be placed)

    def _insert_into_gaps(self, k: int) -> tuple[int, bool]:
        """Insert up to ``k`` vehicles into gaps; returns how many fitted, and whether
        the ring ran out of gaps that take a car.

        A few cars (leavers' replacements whose queue was full) go into random
        gaps that are big enough for the car behind not to slow down, so they
        do not systematically fill the emptiest stretch of road that a "do cars
        congregate?" measure looks for. Bulk density raises, and small ones
        when too few such gaps exist (dense traffic), go into the largest gaps.

        Gaps are ranked by their bumper gap plus a car's length (the distance
        between the centres of two cars). A new vehicle goes halfway along the
        bumper gap, leaving the same room ahead and behind; trucks need their
        extra length on top. A truck that fits no gap waits for one: it stays
        first in line for the next fill (``_pending``, as do vehicles left over
        when no gap takes a car), and the vehicles after it may still go.
        """
        cfg = self.cfg
        # A gap must leave the new car and its follower the lane-change safety room.
        min_room = max(0.5 * cfg.min_gap, SAFE_MARGIN + 0.2)
        fits = 2 * (cfg.car_length + min_room)
        trucks = self._fill_types(k)
        extra = np.where(trucks, cfg.truck_length - cfg.car_length, 0.0)  # [m] longer than a car
        idx = np.flatnonzero(self.alive)
        _, gap = self._index_lanes(idx)
        # Gaps: (-(bumper gap + car length), lane, angle of the vehicle behind the gap, its slot,
        # its extra length). Slot -2: a vehicle inserted here; -3: an empty lane.
        heap: list[tuple[float, int, float, int, float]] = []
        for lane, (order, th) in enumerate(self._lanes):
            if order.size == 0:
                heap.append((-self.lane_len[lane], lane, float(self.rng.uniform(0, TWO_PI)), -3, 0.0))
            else:
                heap.extend(zip(-(gap[order] + cfg.car_length), [lane] * order.size,
                                th.tolist(), order.tolist(), self._extra_length(order).tolist()))
        small = k <= SMALL_FILL
        if small:
            # Only gaps where the car behind need not slow down for the new vehicles (the longest of
            # them that fits anywhere: a truck that fits no gap waits, and need not hold up the cars).
            longest = 0.0
            if extra.any():
                largest = -min(entry[0] for entry in heap)
                longest = float(max((e for e in extra if fits + e <= largest), default=0.0))
            cand = [entry for entry in heap if -entry[0] >= fits + longest and not self._near_exit(entry)]
            behind = np.array([entry[3] for entry in cand], dtype=np.int64)
            keep = behind < 0
            if (~keep).any():  # one vectorised check for all the cars behind
                room = np.array([-0.5 * (entry[0] + longest) - cfg.car_length for entry in cand])[~keep]
                keep[~keep] = self.equilibrium_speed(room, self.desired_speed(behind[~keep]),
                                                     behind[~keep]) >= self.v[behind[~keep]]
            ok = [entry for entry, fine in zip(cand, keep) if fine]
            if len(ok) >= k:
                heap = [ok[i] for i in self.rng.choice(len(ok), size=k, replace=False)]
        heapq.heapify(heap)
        placed = np.zeros(k, dtype=bool)
        full = False  # no gap left that takes a car
        lanes, angles, followers, arcs = [], [], [], []
        for j, e in enumerate(extra):
            if not heap or -heap[0][0] < fits:
                full = True
                break
            if -heap[0][0] < fits + e:
                continue  # a truck that fits not even the largest gap waits for one
            neg_arc, lane, start, foll, foll_extra = heapq.heappop(heap)
            arc = -neg_arc
            placed[j] = True
            lanes.append(lane)
            if foll == -3:  # empty lane: the first vehicle's gap is the whole lane
                angles.append(start)
                followers.append(-2)
                arcs.append(arc - e)
                heapq.heappush(heap, (neg_arc + e, lane, start, -2, e))
                continue
            # Halfway along the bumper gap: (gap + follower's length) / 2 from the follower's centre.
            angle = (start + 0.5 * (arc + foll_extra) / self.radii[lane]) % TWO_PI
            angles.append(angle)
            followers.append(foll)
            arcs.append(0.5 * (arc - e))
            heapq.heappush(heap, (-0.5 * (arc - e), lane, start, foll, foll_extra))
            heapq.heappush(heap, (-0.5 * (arc - e), lane, angle, -2, e))
        self._pending = trucks[~placed].tolist()
        if not lanes:
            return 0, full
        slots = self._spawn(np.array(lanes), np.array(angles), np.zeros(len(lanes)), trucks=trucks[placed])
        self.inserted_at[slots] = self.time
        # arcs: bumper gap ahead of each new vehicle + a car's length.
        eq = self.equilibrium_speed(np.array(arcs) - cfg.car_length, self.desired_speed(slots), slots)
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
        self._cap_speed(behind, gap[behind], max_cut=self.eff.max_decel * cfg.dt if small else np.inf)
        return len(lanes), full

    def _near_exit(self, gap_entry: tuple[float, int, float, int, float]) -> bool:
        """Whether a car slotted into this gap (outer lane) would sit just before an exit."""
        neg_arc, lane, start, _, foll_extra = gap_entry
        if lane != self.outer or self.n_arms == 0:
            return False
        mid = start + 0.5 * (-neg_arc + foll_extra) / self.radii[lane]
        before = (self.exit_angle - mid) % TWO_PI * self.radii[lane]
        return bool((before < 15.0).any())

    def _cap_speed(self, slots, gap, max_cut: float = np.inf) -> None:
        """Slow ``slots`` to the steady-state speed for ``gap`` (by at most
        ``max_cut`` m/s; the driver model brakes the rest), without logging stops.

        Slowed cars are marked as touched by the density control, so a jam they
        head is counted as "other", not "following".
        """
        eq = self.equilibrium_speed(gap, self.desired_speed(slots), slots)
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
        every lane is full. With trucks (the vehicles still waiting to be placed
        first) a lane has room for one more vehicle while its lengths and
        standstill gaps add up to no more than the lane, a truck no lane has
        room for keeps waiting, and the spacing evens out the bumper gaps.
        """
        cfg = self.cfg
        pitch = cfg.car_length + cfg.min_gap
        trucks = self._fill_types(k)
        idx = np.flatnonzero(self.alive)
        self._index_lanes(idx)
        count = np.array([order.size for order, _ in self._lanes])
        room = np.floor(self.lane_len / pitch).astype(int)
        used = np.array([self.length[order].sum() for order, _ in self._lanes]) + count * cfg.min_gap
        added = np.zeros(self.n_lanes, dtype=int)
        given: list[list[bool]] = [[] for _ in range(self.n_lanes)]  # truck or not, per new vehicle
        placed = np.zeros(k, dtype=bool)
        for j, truck in enumerate(trucks):
            # At most one new car per existing gap in a lane.
            if self.uniform_length:
                free = count + added < room
            else:
                free = used + (cfg.truck_length if truck else cfg.car_length) + cfg.min_gap <= self.lane_len
            spare = free & ((count == 0) | (added < count))
            if not spare.any():
                if truck:
                    continue  # no room for a truck: it waits, a car may still fit
                break
            lane = np.argmax(np.where(spare, self.lane_len / (count + added + 1), -1.0))
            added[lane] += 1
            used[lane] += (cfg.truck_length if truck else cfg.car_length) + cfg.min_gap
            given[lane].append(bool(truck))
            placed[j] = True
        self._pending = trucks[~placed].tolist()
        if not added.any():
            return False
        for lane in np.flatnonzero(added):
            order, th = self._lanes[lane]
            n_new = order.size + added[lane]
            step = TWO_PI / n_new
            new_trucks = np.array(given[lane])
            if order.size:
                # New vehicles go into the largest gaps (bumper to bumper), one per gap.
                gaps = np.diff(np.append(th, th[0] + TWO_PI))
                if not self.uniform_length:
                    length = self.length[order]
                    gaps = gaps * self.radii[lane] - 0.5 * (length + np.roll(length, -1))
                take = np.zeros(order.size, dtype=bool)
                take[np.argsort(-gaps)[:added[lane]]] = True
                new = self._spawn(np.full(added[lane], lane), th[take], np.zeros(added[lane]), trucks=new_trucks)
                sequence = np.empty(n_new, dtype=np.int64)
                at = np.arange(order.size) + np.cumsum(np.concatenate(([0], take[:-1])))
                sequence[at] = order
                sequence[at[take] + 1] = new
                old = self.theta[sequence].copy()
                extra = self._extra_length(sequence)
                self.theta[sequence] = (th[0] + step * np.arange(n_new)
                                        + self._even_shift(extra) / self.radii[lane]) % TWO_PI
                # Keep each car's distance to its exit tied to its new position.
                self.travel[sequence] += (self.theta[sequence] - old + np.pi) % TWO_PI - np.pi
            else:
                new = sequence = self._spawn(np.full(n_new, lane), np.zeros(n_new), np.zeros(n_new), trucks=new_trucks)
                extra = self._extra_length(sequence)
                self.theta[sequence] = (self.rng.uniform(0, TWO_PI) + step * np.arange(n_new)
                                        + self._even_shift(extra) / self.radii[lane]) % TWO_PI
                self._assign_routes(new)
            self.inserted_at[new] = self.time
            self.v[new] = np.inf  # capped to the steady-state speed below
            even_gap = step * self.radii[lane] - cfg.car_length - extra.mean()  # every bumper gap
            self._cap_speed(sequence, np.full(n_new, even_gap))
        return True
