"""Model parameters: defaults, ranges, units and descriptions.

``SimConfig`` holds the values; ``PARAMS`` describes each one (label, range,
unit, group, help text, whether changing it needs a reset, and whether it only
has an effect in one traffic mode). The viewer's
sliders and the experiments CLI are both generated from ``PARAMS``, so this
file is the single place to add or retune a parameter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from functools import lru_cache
from numbers import Integral, Real


@dataclass
class SimConfig:
    # --- Road (changing these rebuilds the roundabout) -----------------------
    n_lanes: int = 3
    n_arms: int = 4  # entry/exit pairs, evenly spaced; 0 = closed ring
    radius: float = 80.0  # [m] centreline radius of the innermost lane
    lane_width: float = 3.5  # [m]
    car_length: float = 4.0  # [m]
    car_width: float = 1.8  # [m]
    road_condition: str = "dry"  # "dry", "wet", "snow" or "ice" (live: scales the driving, see engine)
    # --- Traffic --------------------------------------------------------------
    density: float = 50.0  # [veh/km/lane] cars on the ring per km of lane (open mode: at the start)
    exit_prob: float = 0.1  # chance a car leaves at each exit it passes
    entry_gap: float = 2.0  # [s] smallest time gap an entering car accepts (critical gap)
    follow_up: float = 1.0  # [s] time between two cars entering from one entrance (next step after it)
    turn_speed: float = 6.0  # [m/s] speed when entering / leaving the ring
    # --- Demand (open mode: cars arrive at random instead of replacing leavers) ---
    traffic_mode: str = "closed"  # "closed" (density held) or "open" (arrivals)
    entry_rate: float = 0.1  # [1/s per entrance] mean arrival rate in open mode
    demand_profile: str = "constant"  # "constant" or "peak" (rises and falls, same mean over each cycle)
    demand_imbalance: float = 0.0  # 0 = equal entrances, 1 = all on the main road
    max_queue: int = 50  # cars that can wait per entrance in open mode
    # --- Vehicles (changing these rebuilds the roundabout) --------------------
    truck_share: float = 0.0  # share of the vehicles that are trucks
    truck_length: float = 12.0  # [m]
    # --- Drivers --------------------------------------------------------------
    mean_speed: float = 12.0  # [m/s] mean desired speed
    speed_std: float = 1.5  # [m/s] spread of desired speeds across drivers
    reaction_time: float = 0.4  # [s] delay between perceiving and acting
    reaction_std: float = 0.0  # [s] spread of reaction times across drivers
    reaction_distance: float = 25.0  # [m] how far ahead drivers react to cars
    time_headway: float = 1.2  # [s] desired time gap to the car in front
    headway_std: float = 0.0  # [s] spread of desired time gaps across drivers
    min_gap: float = 2.0  # [m] bumper-to-bumper gap kept when stopped
    max_accel: float = 1.2  # [m/s2]
    comfort_decel: float = 2.0  # [m/s2]
    driver_noise: float = 0.35  # [m/s2] std. dev. of random acceleration noise
    lateral_accel: float = 4.0  # [m/s2] comfortable cornering -> per-lane speed cap
    # --- Lane changing --------------------------------------------------------
    lane_change_rate: float = 0.2  # [1/s] how often a driver considers a change
    lane_change_motive: str = "advantage"  # "advantage" (MOBIL) or "random"
    politeness: float = 0.3  # MOBIL weight on the gain/loss of other drivers
    # --- Initial state --------------------------------------------------------
    init_spacing: str = "uniform"  # "uniform", "random" or "clustered"
    # --- Numerics -------------------------------------------------------------
    dt: float = 0.1  # [s] time step
    seed: int | None = None

    def __post_init__(self) -> None:
        """Reject values the model cannot run with (looser than the slider ranges,
        which the experiments CLI may exceed on purpose)."""
        problems = []
        unbounded = ("density", "reaction_distance")  # inf is meaningful: jam density / see everything
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and (
                    math.isnan(value) or (math.isinf(value) and f.name not in unbounded)):
                problems.append(f"{f.name} must be a finite number")
        if problems:
            raise ValueError("; ".join(problems))
        positive = ("radius", "lane_width", "car_length", "car_width", "truck_length", "dt", "mean_speed",
                    "time_headway", "min_gap", "max_accel", "comfort_decel", "lateral_accel",
                    "turn_speed", "entry_gap", "follow_up", "reaction_distance")
        problems = [f"{name} must be > 0" for name in positive if not getattr(self, name) > 0]
        for name, low in (("n_lanes", 1), ("n_arms", 0), ("max_queue", 1)):
            value = getattr(self, name)
            if not (float(value).is_integer() and value >= low):
                problems.append(f"{name} must be a whole number >= {low}")
        if self.seed is not None and not (isinstance(self.seed, Integral) and not isinstance(self.seed, bool)
                                          and self.seed >= 0):
            problems.append("seed must be None or a whole number >= 0")
        for name in ("density", "speed_std", "reaction_time", "reaction_std", "headway_std", "driver_noise",
                     "lane_change_rate", "politeness", "entry_rate"):
            if not getattr(self, name) >= 0:
                problems.append(f"{name} must be >= 0")
        for name in ("exit_prob", "demand_imbalance", "truck_share"):
            if not 0 <= getattr(self, name) <= 1:
                problems.append(f"{name} must be between 0 and 1")
        for spec in PARAMS:
            if spec.choices and getattr(self, spec.name) not in spec.choices:
                problems.append(f"{spec.name} must be one of {', '.join(spec.choices)}")
        if problems:
            raise ValueError("; ".join(problems))

    def updated(self, **changes) -> "SimConfig":
        """Copy with ``changes`` applied (unknown names raise)."""
        unknown = set(changes) - {f.name for f in fields(self)}
        if unknown:
            raise KeyError(f"unknown parameter(s): {', '.join(sorted(unknown))}")
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values.update(changes)
        return SimConfig(**values)


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    lo: float
    hi: float
    step: float
    unit: str
    group: str
    help: str
    reset: bool = False  # True: changing it rebuilds the roundabout
    choices: tuple[str, ...] | None = None  # set for categorical parameters
    applies_in: str | None = None  # the only traffic_mode it has an effect in (None = every mode)


PARAMS: tuple[ParamSpec, ...] = (
    # Road
    ParamSpec("n_lanes", "Lanes", 1, 5, 1, "", "Road",
              "Number of concentric lanes on the ring.", reset=True),
    # Up to 12 arms for the Place Charles de Gaulle (12 avenues, presets.py "etoile"); was 8.
    ParamSpec("n_arms", "Entrances / exits", 0, 12, 1, "", "Road",
              "Evenly spaced entry/exit pairs on the outer lane (0 = closed ring; up to 12, as at the "
              "Place Charles de Gaulle in Paris).", reset=True),
    # Down to 10 m for compact and single-lane roundabouts (US single-lane 16-21 m, UK compact from 12 m),
    # in 1 m steps: fine enough for real layouts, and one step per pixel or two of the slider track (finer
    # steps could not be hit by dragging). Presets keep their published radius between the steps (36.6,
    # 18.25 m), and the viewers show such values exactly (textfmt.format_number). Was 20-300 m in 5 m steps.
    ParamSpec("radius", "Ring radius", 10, 300, 1, "m", "Road",
              "Centreline radius of the innermost lane. Real single-lane roundabouts are about 12-21 m "
              "(28-46 m across), two-lane ones 18-28 m, the ring-road jam experiments 37-50 m. Presets keep "
              "their published radius, also between the slider's 1 m steps.", reset=True),
    ParamSpec("car_length", "Car length", 1.0, 8.0, 0.5, "m", "Road",
              "Space each car takes up along the lane.", reset=True),
    ParamSpec("road_condition", "Road condition", 0, 0, 0, "", "Road",
              "Road surface. wet, snow and ice lower the desired speed, acceleration, comfortable braking, "
              "cornering limit and the braking limit (9 m/s2 dry, 7.2 wet, 2.7 snow, 1.35 ice), and lengthen "
              "the following time gap, entry gap and follow-up time (factors from weather studies; ice is "
              "the least certain). The other sliders keep showing the dry values. Applies at once.",
              choices=("dry", "wet", "snow", "ice")),
    # Traffic
    ParamSpec("density", "Density", 1, 300, 1, "veh/km/lane", "Traffic",
              "Cars on the ring per km of lane (capped at jam density). On a closed ring with "
              "the default drivers, stop-and-go waves appear from about 75-80; entrances and "
              "exits, shorter reaction distance, longer reaction time or more noise bring them "
              "on earlier. In open mode it sets only the cars at the start (moved live: once, then)."),
    ParamSpec("exit_prob", "Exit chance", 0.0, 1.0, 0.05, "", "Traffic",
              "Chance a car leaves at each exit it passes. In closed mode each leaver is replaced by a "
              "car at a random entrance (slotted into the ring if that queue is full); in open mode "
              "leavers are gone for good."),
    ParamSpec("entry_gap", "Entry gap", 1.5, 6.0, 0.1, "s", "Traffic",
              "Smallest time gap to circulating traffic an entering car accepts (the critical gap)."),
    ParamSpec("follow_up", "Follow-up time", 1.0, 5.0, 0.1, "s", "Traffic",
              "Shortest time between two cars entering from the same entrance (the follow-up headway; "
              "real roundabouts show about 2.5-3.2 s). The next car may go on the first time step after "
              "it has passed, so 1 s gives 1.1 s at the 0.1 s step. Longer = a queue empties more slowly."),
    ParamSpec("turn_speed", "Turn speed", 2.0, 15.0, 0.5, "m/s", "Traffic",
              "Speed at which cars enter and leave the ring."),
    # Demand
    ParamSpec("traffic_mode", "Traffic mode", 0, 0, 0, "", "Demand",
              "closed: the number of cars is held at Density (each leaver is replaced at a random "
              "entrance). open: cars arrive at the entrances at random (Entrance chance), queue, enter "
              "and leave for good, so the number on the ring varies; Density sets only the start.",
              reset=True, choices=("closed", "open")),
    ParamSpec("entry_rate", "Entrance chance", 0.0, 0.5, 0.01, "/s", "Demand",
              "Open mode: mean arrivals per second at each entrance (at random, Poisson). 0.1 /s = "
              "360 veh/h per entrance; 0.5 /s = 1800 veh/h, more than one entry lane can take.",
              applies_in="open"),
    ParamSpec("demand_profile", "Demand profile", 0, 0, 0, "", "Demand",
              "Open mode. constant: the Entrance chance all the time; peak: it rises smoothly from half "
              "to 1.5 times its value and back every 10 minutes (a rush hour in miniature), with the "
              "same mean over each whole 10 minutes (not over shorter windows).", choices=("constant", "peak"),
              applies_in="open"),
    ParamSpec("demand_imbalance", "Demand imbalance", 0.0, 1.0, 0.05, "", "Demand",
              "Open mode: 0 = every entrance gets the same demand; higher puts more on a main road "
              "through the bottom entrance (and the top one), less on the side roads (at 1 none at "
              "right angles to it). The mean per entrance stays the Entrance chance.",
              applies_in="open"),
    ParamSpec("max_queue", "Max queue", 1, 200, 1, "cars", "Demand",
              "Open mode: cars that can wait at each entrance. Arrivals that find the queue full are "
              "turned away and counted as blocked (spill-back onto the approach road).",
              applies_in="open"),
    # Vehicles
    ParamSpec("truck_share", "Truck share", 0.0, 0.3, 0.01, "", "Vehicles",
              "Share of the vehicles that are trucks (each new vehicle is one by chance). Trucks are Truck "
              "length long and 2.55 m wide, accelerate half as briskly, want 0.7 x a driver's desired speed "
              "and keep a 1.5 x longer time gap. Real shares: about 1-5 % in towns, 5-15 % on main roads.",
              reset=True),
    ParamSpec("truck_length", "Truck length", 6.0, 23.0, 0.5, "m", "Vehicles",
              "Length of a truck: 9 m single unit, 12 m rigid lorry (the EU limit), 16.5 m articulated, "
              "18.75 m road train, 22.4 m US WB-67. Capacity and jam density use the mean vehicle length.",
              reset=True),
    # Drivers
    ParamSpec("mean_speed", "Mean desired speed", 3.0, 30.0, 0.5, "m/s", "Drivers",
              "Average speed drivers want to go on a free road."),
    ParamSpec("speed_std", "Speed variance (std)", 0.0, 6.0, 0.1, "m/s", "Drivers",
              "Spread of desired speeds between drivers (nobody wants less than half the mean)."),
    ParamSpec("reaction_time", "Reaction time", 0.0, 2.0, 0.1, "s", "Drivers",
              "Delay between a driver perceiving the traffic and acting on it."),
    ParamSpec("reaction_std", "Reaction time spread (std)", 0.0, 0.5, 0.05, "s", "Drivers",
              "How much reaction times differ between drivers. Each driver's own is drawn once around Reaction "
              "time (kept within 0-2 s, rounded to the time step); changed live, every driver keeps their "
              "place in the spread. Real drivers differ by about 0.2 s."),
    ParamSpec("reaction_distance", "Reaction distance", 10, 100, 5, "m", "Drivers",
              "How far ahead drivers pay attention. They react to every car within this "
              "distance (up to 5), so looking further ahead lets them ease off early for a "
              "slowdown a few cars ahead, which damps stop-and-go waves. Very short (below "
              "about speed x following gap) and they notice the car in front too late and "
              "brake violently."),
    ParamSpec("time_headway", "Following time gap", 0.5, 3.0, 0.1, "s", "Drivers",
              "Desired time gap to the car in front."),
    ParamSpec("headway_std", "Time gap spread (std)", 0.0, 0.8, 0.05, "s", "Drivers",
              "How much the following time gap differs between drivers. Each driver's own is drawn once "
              "around Following time gap (at least half of it; trucks keep 1.5 x theirs); changed live, every "
              "driver keeps their place in the spread. Observed: about 0.3 s."),
    ParamSpec("min_gap", "Standstill gap", 0.5, 5.0, 0.25, "m", "Drivers",
              "Bumper-to-bumper gap kept in a stopped queue."),
    ParamSpec("max_accel", "Max acceleration", 0.3, 4.0, 0.1, "m/s²", "Drivers",
              "How briskly drivers speed up."),
    ParamSpec("comfort_decel", "Comfortable braking", 0.5, 5.0, 0.1, "m/s²", "Drivers",
              "Braking drivers are happy to use."),
    ParamSpec("driver_noise", "Driver noise", 0.0, 1.5, 0.05, "m/s²", "Drivers",
              "Random fluctuations in each driver's acceleration."),
    ParamSpec("lateral_accel", "Cornering limit", 1.0, 10.0, 0.5, "m/s²", "Drivers",
              "Comfortable sideways acceleration; caps speed on tight (inner) lanes."),
    # Lane changing
    ParamSpec("lane_change_rate", "Lane-change chance", 0.0, 1.0, 0.05, "/s", "Lane chg",
              "How often (per second) a driver considers changing lane."),
    ParamSpec("lane_change_motive", "Lane-change motive", 0, 0, 0, "", "Lane chg",
              "advantage: only change when it is faster (MOBIL); random: any safe gap.",
              choices=("advantage", "random")),
    ParamSpec("politeness", "Politeness", 0.0, 1.0, 0.05, "", "Lane chg",
              "How much drivers weigh the braking they force on others when changing lane."),
    # Initial state
    ParamSpec("init_spacing", "Initial spacing", 0, 0, 0, "", "Start",
              "How the starting gaps are laid out (their average is set by Density): uniform = "
              "evenly spaced, random = random gaps, clustered = one packed queue across all lanes. "
              "Near the jam threshold an uneven start can lock in jams an even start never forms; "
              "elsewhere its effect fades within a minute or two.",
              reset=True, choices=("uniform", "random", "clustered")),
)

PARAM_BY_NAME = {p.name: p for p in PARAMS}


@lru_cache(maxsize=1)
def _defaults() -> dict:
    """Every SimConfig field's default (built once: coerce runs for every value of a random draw or a plan)."""
    return dict(vars(SimConfig()))


def coerce(name: str, value):
    """Convert a CLI/UI value to the type ``SimConfig`` expects for ``name``."""
    if name not in _defaults():
        raise AttributeError(f"'SimConfig' object has no attribute {name!r}")
    default = _defaults()[name]
    if isinstance(default, bool):
        return str(value).lower() in ("1", "true", "yes", "on")
    if isinstance(default, int) and not isinstance(default, bool):
        return int(round(float(value)))
    if isinstance(default, float):
        return float(value)
    if default is None:  # seed
        return None if value in (None, "", "none", "None") else int(value)
    spec = PARAM_BY_NAME.get(name)
    if spec and spec.choices and value not in spec.choices:
        raise ValueError(f"{name} must be one of {', '.join(spec.choices)}")
    return value


def same_value(a, b) -> bool:
    """Whether two values of a setting are the same: numbers within rounding (numpy's too, not
    bools), anything else exactly. The one test of "differs from the default" for the viewers and the
    experiments."""
    def number(v) -> bool:
        return isinstance(v, Real) and not isinstance(v, bool) and type(v).__name__ != "bool_"

    if number(a) and number(b):
        return a == b or math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def changed_settings(values: dict) -> dict:
    """The entries of ``values`` (setting -> value) that differ from the defaults (:func:`same_value`);
    names that are no settings, and the seed, are left out. What a sweep's base, fixed settings and
    meta file list."""
    defaults = _defaults()
    return {n: v for n, v in values.items() if n in defaults and n != "seed" and not same_value(v, defaults[n])}


def axis_label(name: str) -> str:
    """'Ring radius (m)', 'Lanes': a setting's label with its unit, as on the chart axes (the name
    itself for a column that is no setting)."""
    spec = PARAM_BY_NAME.get(name)
    if spec is None:
        return name
    return f"{spec.label} ({spec.unit})" if spec.unit else spec.label


# Settings that act only while another setting is above 0: (that setting, the reason shown meanwhile).
# Without entrances (n_arms = 0: the ring-road experiments) nobody enters, queues or leaves.
NEEDS_NONZERO = {"truck_length": ("truck_share", "no trucks"),
                 **{name: ("n_arms", "no entrances") for name in ("exit_prob", "entry_gap", "follow_up", "turn_speed",
                                                                  "entry_rate", "demand_profile", "demand_imbalance",
                                                                  "max_queue")}}
IDLE_WHY = {  # (reason, one setting or several) -> what it says about every run of a sweep
    "open mode only": ("acts only in open mode", "act only in open mode", "every run here is closed"),
    "no entrances": ("does nothing without entrances", "do nothing without entrances",
                     "every run here has none (n_arms = 0)"),
    "no trucks": ("does nothing without trucks", "do nothing without trucks", "every run here has truck_share = 0"),
}


def inactive_reason(name: str, cfg: SimConfig) -> str:
    """Why the setting ``name`` has no effect with the settings ``cfg`` ("" if it has one): "open mode
    only" (``applies_in``), "no entrances" or "no trucks" (NEEDS_NONZERO). The main viewer tags its
    controls with it; the sweep tools warn of a varied setting that does nothing in every run."""
    spec = PARAM_BY_NAME.get(name)
    if spec is not None and spec.applies_in and cfg.traffic_mode != spec.applies_in:
        return f"{spec.applies_in} mode only"
    if name in NEEDS_NONZERO:
        other, reason = NEEDS_NONZERO[name]
        if not getattr(cfg, other) > 0:
            return reason
    return ""


def idle_note(names: list[str], reason: str) -> str:
    """'entry_rate acts only in open mode, and every run here is closed: its values change nothing'
    (``names`` as the caller shows them; ``reason`` from :func:`inactive_reason`)."""
    one, several, runs = IDLE_WHY.get(reason, (f"does nothing here ({reason})", f"do nothing here ({reason})",
                                               "in every run"))
    who = ", ".join(names[:-1]) + f" and {names[-1]}" if len(names) > 1 else names[0]
    return f"{who} {one if len(names) == 1 else several}, and {runs}: {'its' if len(names) == 1 else 'their'} " \
           "values change nothing"
