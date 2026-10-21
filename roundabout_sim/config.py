"""Model parameters: defaults, ranges, units and descriptions.

``SimConfig`` holds the values; ``PARAMS`` describes each one (label, range,
unit, group, help text, whether changing it needs a reset). The viewer's
sliders and the experiments CLI are both generated from ``PARAMS``, so this
file is the single place to add or retune a parameter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from numbers import Integral


@dataclass
class SimConfig:
    # --- Road (changing these rebuilds the roundabout) -----------------------
    n_lanes: int = 3
    n_arms: int = 4  # entry/exit pairs, evenly spaced; 0 = closed ring
    radius: float = 80.0  # [m] centreline radius of the innermost lane
    lane_width: float = 3.5  # [m]
    car_length: float = 4.0  # [m]
    car_width: float = 1.8  # [m]
    # --- Traffic --------------------------------------------------------------
    density: float = 50.0  # [veh/km/lane] cars on the ring per km of lane
    exit_prob: float = 0.1  # chance a car leaves at each exit it passes
    entry_gap: float = 2.0  # [s] smallest time gap an entering car accepts
    turn_speed: float = 6.0  # [m/s] speed when entering / leaving the ring
    # --- Drivers --------------------------------------------------------------
    mean_speed: float = 12.0  # [m/s] mean desired speed
    speed_std: float = 1.5  # [m/s] spread of desired speeds across drivers
    reaction_time: float = 0.4  # [s] delay between perceiving and acting
    reaction_distance: float = 25.0  # [m] how far ahead drivers react to cars
    time_headway: float = 1.2  # [s] desired time gap to the car in front
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
        positive = ("radius", "lane_width", "car_length", "car_width", "dt", "mean_speed",
                    "time_headway", "min_gap", "max_accel", "comfort_decel", "lateral_accel",
                    "turn_speed", "entry_gap", "reaction_distance")
        problems = [f"{name} must be > 0" for name in positive if not getattr(self, name) > 0]
        for name, low in (("n_lanes", 1), ("n_arms", 0)):
            value = getattr(self, name)
            if not (float(value).is_integer() and value >= low):
                problems.append(f"{name} must be a whole number >= {low}")
        if self.seed is not None and not (isinstance(self.seed, Integral) and not isinstance(self.seed, bool)
                                          and self.seed >= 0):
            problems.append("seed must be None or a whole number >= 0")
        for name in ("density", "speed_std", "reaction_time", "driver_noise", "lane_change_rate",
                     "politeness"):
            if not getattr(self, name) >= 0:
                problems.append(f"{name} must be >= 0")
        if not 0 <= self.exit_prob <= 1:
            problems.append("exit_prob must be between 0 and 1")
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


PARAMS: tuple[ParamSpec, ...] = (
    # Road
    ParamSpec("n_lanes", "Lanes", 1, 5, 1, "", "Road",
              "Number of concentric lanes on the ring.", reset=True),
    ParamSpec("n_arms", "Entrances / exits", 0, 8, 1, "", "Road",
              "Evenly spaced entry/exit pairs on the outer lane (0 = closed ring).", reset=True),
    ParamSpec("radius", "Ring radius", 20, 300, 5, "m", "Road",
              "Centreline radius of the innermost lane.", reset=True),
    ParamSpec("car_length", "Car length", 1.0, 8.0, 0.5, "m", "Road",
              "Space each car takes up along the lane.", reset=True),
    # Traffic
    ParamSpec("density", "Density", 1, 300, 1, "veh/km/lane", "Traffic",
              "Cars on the ring per km of lane (capped at jam density). On a closed ring with "
              "the default drivers, stop-and-go waves appear from about 75-80; entrances and "
              "exits, shorter reaction distance, longer reaction time or more noise bring them "
              "on earlier."),
    ParamSpec("exit_prob", "Exit chance", 0.0, 1.0, 0.05, "", "Traffic",
              "Chance a car leaves at each exit it passes. Each leaver is replaced by a car at a random "
              "entrance (or slotted into the ring if that entrance already has a queue)."),
    ParamSpec("entry_gap", "Entry gap", 1.5, 6.0, 0.1, "s", "Traffic",
              "Smallest time gap to circulating traffic an entering car accepts."),
    ParamSpec("turn_speed", "Turn speed", 2.0, 15.0, 0.5, "m/s", "Traffic",
              "Speed at which cars enter and leave the ring."),
    # Drivers
    ParamSpec("mean_speed", "Mean desired speed", 3.0, 30.0, 0.5, "m/s", "Drivers",
              "Average speed drivers want to go on a free road."),
    ParamSpec("speed_std", "Speed variance (std)", 0.0, 6.0, 0.1, "m/s", "Drivers",
              "Spread of desired speeds between drivers (nobody wants less than half the mean)."),
    ParamSpec("reaction_time", "Reaction time", 0.0, 2.0, 0.1, "s", "Drivers",
              "Delay between a driver perceiving the traffic and acting on it."),
    ParamSpec("reaction_distance", "Reaction distance", 10, 100, 5, "m", "Drivers",
              "How far ahead drivers pay attention. They react to every car within this "
              "distance (up to 5), so looking further ahead lets them ease off early for a "
              "slowdown a few cars ahead, which damps stop-and-go waves. Very short (below "
              "about speed x following gap) and they notice the car in front too late and "
              "brake violently."),
    ParamSpec("time_headway", "Following time gap", 0.5, 3.0, 0.1, "s", "Drivers",
              "Desired time gap to the car in front."),
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


def coerce(name: str, value):
    """Convert a CLI/UI value to the type ``SimConfig`` expects for ``name``."""
    default = getattr(SimConfig(), name)
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
