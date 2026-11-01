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
    weather_scale: float = 1.0  # strength of the road condition's effect on driving (1 = ROAD_CONDITIONS as given)
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
    subject_arm: str = "off"  # "on": entrance 0 is a saturated capacity test (open mode; see engine)
    subject_exiting: float = 0.0  # capacity test: share of the other entrances' entrants that leave at its exit
    # --- Vehicles (the share and length of trucks rebuild the roundabout) -----
    truck_share: float = 0.0  # share of the vehicles that are trucks
    truck_length: float = 12.0  # [m]
    truck_follow_up: float = 1.0  # follow-up behind an entering truck as a multiple of the Follow-up time
    # --- Drivers --------------------------------------------------------------
    mean_speed: float = 12.0  # [m/s] mean desired speed
    speed_std: float = 1.5  # [m/s] spread of desired speeds across drivers
    reaction_time: float = 0.4  # [s] delay between perceiving and acting
    reaction_std: float = 0.0  # [s] spread of reaction times across drivers
    reaction_distance: float = 25.0  # [m] how far ahead drivers react to cars
    anticipation: int = 5  # most cars ahead a driver reacts to (engine.ANTICIPATION)
    time_headway: float = 1.2  # [s] desired time gap to the car in front
    headway_std: float = 0.0  # [s] spread of desired time gaps across drivers
    min_gap: float = 2.0  # [m] bumper-to-bumper gap kept when stopped
    max_accel: float = 1.2  # [m/s2]
    comfort_decel: float = 2.0  # [m/s2]
    driver_noise: float = 0.35  # [m/s2] std. dev. of random acceleration noise
    lateral_accel: float = 4.0  # [m/s2] comfortable cornering -> per-lane speed cap
    # (their adaptation to recognisable AVs; see engine RingSim._entry_check, _lane_changes, _car_following)
    exploit_gap: float = 1.0  # drivers' gap ahead of a recognisable AV as a multiple of their usual (< 1 exploits)
    follow_adapt: float = 0.0  # how far drivers behind a recognisable AV close up and start sooner (Wang 2023)
    # --- Lane changing --------------------------------------------------------
    lane_change_rate: float = 0.2  # [1/s] how often a driver considers a change
    lane_change_motive: str = "advantage"  # "advantage" (MOBIL) or "random"
    politeness: float = 0.3  # MOBIL weight on the gain/loss of other drivers
    # --- Initial state --------------------------------------------------------
    init_spacing: str = "uniform"  # "uniform", "random" or "clustered"
    # --- Automated vehicles (changing the share or kind rebuilds the roundabout; see engine KIND_NAMES) ----
    auto_share: float = 0.0  # share of the new vehicles that are automated, all of the kind auto_kind
    auto_kind: str = "cav"  # "cav", "robotaxi" or "acc": the one automated kind of a run
    robotaxi_headway: float = 1.3  # robotaxis' time gap as a multiple of a driver's (Hu et al. 2023; live)
    acc_headway: float = 1.1  # [s] ACC cars' time gap (the PATH ACC law, Milanes & Shladover 2014; live)
    acc_delay: float = 0.5  # [s] ACC cars' dead time (Gunter et al. 2019 Table II; live)
    cav_headway: float = 0.7  # [s] a CAV's time gap behind a cooperating CAV (PATH CACC, FHWA-OR-RD-22-11; live)
    platoon_max: int = 8  # most CAVs in one CACC string (FHWA-OR-RD-22-11 Sec. 6.2.2.1; live)
    # --- Cooperation ----------------------------------------------------------
    companies: int = 1  # companies (brands) the automated vehicles belong to, evenly at random
    coop_class: str = "B"  # cooperation class within a company: "none", "A", "B", "C" or "D" (SAE J3216; live)
    coop_between: str = "same"  # most cooperation between companies: "same" (= coop_class) or a class (live)
    brand_spread: float = 0.0  # spread of the companies' ACC time gaps and ACC dead time (live)
    # --- Entries by kind (live; see engine RingSim._entry_check, _follow_up_slack) ----------------------------
    cav_entry_gap: float = 1.0  # a CAV's critical gap as a multiple of Entry gap (FHWA-OR-RD-22-11 Sec. 7.3.2)
    cav_follow_up: float = 0.74  # a CAV's follow-up behind a cooperating CAV, x the realised follow-up (FHWA Table 7.1)
    robotaxi_entry: float = 1.3  # a robotaxi's critical gap and follow-up as multiples of the drivers' (assumption)
    av_weather_scale: float = 1.0  # strength of the road condition's effect on driverless vehicles (1 = as on drivers)
    av_handover: float = 0.0  # [s] on snow or ice a driverless queue head waits this long at the yield line (0 = off)
    # --- Wave smoothers (automated cars converted from drivers; all live; see engine RingSim._law_smoother) ---
    smoothers: int = 0  # how many (a count, not a share)
    smoother_law: str = "followerstopper"  # "followerstopper" (Stern et al. 2018), "pi" (PI with saturation) or "quiet"
    smoother_speed: float = 7.5  # [m/s] FollowerStopper's set speed U (Stern 2018 Exp. A's best)
    smoother_start: float = 0.0  # [s] when their controllers switch on (each drives as its driver until then)
    smoother_lane: int = -1  # their lane: -1 = round-robin from the innermost, else that lane (0 = innermost)
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
                    "turn_speed", "entry_gap", "follow_up", "reaction_distance", "robotaxi_headway", "acc_headway",
                    "cav_headway", "smoother_speed", "cav_entry_gap", "cav_follow_up", "robotaxi_entry",
                    "truck_follow_up", "exploit_gap")
        problems = [f"{name} must be > 0" for name in positive if not getattr(self, name) > 0]
        for name, low in (("n_lanes", 1), ("n_arms", 0), ("max_queue", 1), ("anticipation", 1), ("companies", 1),
                          ("platoon_max", 1), ("smoothers", 0), ("smoother_lane", -1)):
            value = getattr(self, name)
            if not (float(value).is_integer() and value >= low):
                problems.append(f"{name} must be a whole number >= {low}")
        if self.companies > 127:  # (each vehicle's company is an int8: 0 to companies - 1)
            problems.append("companies must be at most 127")
        if self.platoon_max > 126:  # (a CAV's string position is an int8: up to platoon_max + 1)
            problems.append("platoon_max must be at most 126")
        if not 0 <= self.brand_spread < 1:  # (the lowest brand factor, 1 - brand_spread, must stay above 0)
            problems.append("brand_spread must be at least 0 and below 1")
        if self.seed is not None and not (isinstance(self.seed, Integral) and not isinstance(self.seed, bool)
                                          and self.seed >= 0):
            problems.append("seed must be None or a whole number >= 0")
        for name in ("density", "speed_std", "reaction_time", "reaction_std", "headway_std", "driver_noise",
                     "lane_change_rate", "politeness", "entry_rate", "weather_scale", "acc_delay", "smoother_start",
                     "av_weather_scale", "av_handover"):
            if not getattr(self, name) >= 0:
                problems.append(f"{name} must be >= 0")
        for name in ("exit_prob", "demand_imbalance", "truck_share", "auto_share", "subject_exiting", "follow_adapt"):
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
    # The behavioural factors only, as 1 + scale x (factor - 1) (engine.road_factors), so 0.5-2 brackets the
    # studies' spread and the snow loss Munehiro 2017 measured (-300 veh/h, TRB 17-00960). Physics stays in full.
    ParamSpec("weather_scale", "Weather effect", 0.0, 2.0, 0.05, "", "Road",
              "How strongly the road condition changes driving: each wet, snow or ice factor on desired speed, "
              "acceleration, comfortable braking, time gap, entry gap and follow-up becomes 1 + this x (factor "
              "- 1): 1 = the studies, 0 = as if dry, 2 = twice the change. The grip limits (cornering, braking) "
              "apply in full."),
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
    ParamSpec("subject_arm", "Capacity test", 0, 0, 0, "", "Demand",
              "Open mode. on: the bottom entrance is a capacity test: its queue is kept full whatever Max queue "
              "(not counted as arrivals), and every car leaves at the first exit after it (with 2+ lanes the "
              "first far enough on; Exit chance is ignored), so its circulating flow is the other entrances' "
              "entry flow (Entrance chance). Experiments report its capacity (subject_* columns).",
              reset=True, choices=("off", "on"), applies_in="open"),
    # FHWA-OR-RD-22-11 Sec. 7.3.4 (Table 7.3) added exiting flows of 10, 30 and 50 % of the conflicting flow; with a
    # share s of the feeding entrants leaving, the exiting flow is s / (1 - s) of the conflicting one (*derived*), so s
    # = 0.09, 0.23, 0.33. With uniform destinations on 4 arms (no U-turns), half the flow reaching an arm's exit leaves
    # there (*derived*): 0.5, the slider's end. Each n-th entrant leaves (a Bresenham count: engine _route_feeding). On
    # rings of three or more lanes a few of them change lane inward, miss that exit and go round, passing the entry
    # point (*derived*: 600 s at 0.5, 0 of 109 / 96 on us_single_lane / us_two_lane, 6 of 109 on the 3-lane default, 13
    # of 194 on the 5-lane etoile): use the capacity test on one or two circulating lanes, as validated.
    ParamSpec("subject_exiting", "Test exiting share", 0.0, 0.5, 0.05, "", "Demand",
              "Capacity test: this share of the other entrances' entrants leaves at the test entrance's own exit, "
              "before its entry point (every n-th, no chance): an exiting stream drivers cannot tell from the "
              "conflicting flow, CAVs at Cooperation B can. FHWA-OR-RD-22-11 added 10-50 % of the conflicting flow "
              "(0.09-0.33 here).",
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
    # HCM 7 counts a heavy vehicle as E_T = 2.0 passenger cars at roundabouts (Exhibit 22-11, f_HV Eq. 22-10, as the
    # design review cites them; not checked against the HCM text). The model's entering truck costs about one car.
    # *derived* (us_single_lane capacity test, 10 % trucks, 10 seeds, 300 + 1,800 s; capacity against the HCM's
    # c_cars(vc in pc/h) / 1.1): 2.0 gives E_T 2 with no conflicting flow (ratio 1.000) and within 5 % up to vc 610
    # veh/h (1.079 at 814); 2.4 within 5 % over the whole validated 0-820 veh/h (0.966-1.038, mean 1.002). The model's
    # truck costs less the busier the circle (its extra wait overlaps the wait for the next gap); the HCM's E_T is one
    # constant. 1.0 = the model before (bit-identical).
    ParamSpec("truck_follow_up", "Truck follow-up", 1.0, 3.0, 0.1, "", "Vehicles",
              "The queued vehicle behind an entering truck waits this many times the follow-up (2.6 s for a Follow-up "
              "time of 2.5 s) while the truck clears the yield line. 1 = as behind a car: a truck then costs about one "
              "car, not the HCM's 2 passenger cars (E_T 2.0): 2 gives that with no circulating traffic, 2.4 over "
              "0-820 veh/h."),
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
              "distance (up to Anticipation), so looking further ahead lets them ease off early for a "
              "slowdown a few cars ahead, which damps stop-and-go waves. Very short (below "
              "about speed x following gap) and they notice the car in front too late and "
              "brake violently."),
    ParamSpec("anticipation", "Anticipation", 1, 5, 1, "cars", "Drivers",
              "Most cars ahead a driver reacts to (within Reaction distance), as in Treiber's Human Driver "
              "Model: the j-th car ahead counts with weight 1/j^2. 1 = only the car in front (the plain IDM); "
              "5, the default, lets drivers ease off early for a slowdown a few cars ahead, which damps "
              "stop-and-go waves. Drivers anticipate 1-7 vehicles (Treiber, Kesting & Helbing 2006)."),
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
    # Drivers ahead of a recognisable AV (engine RingSim._entry_check, _lane_changes; on the Drivers tab, never drawn
    # at random: presets.RANDOM_RANGES). Alhuraish, Wang & Zhou 2026
    # (arXiv 2605.01485, a preprint; Waymo Open Motion Dataset, highway and arterial cut-ins): drivers cutting in ahead
    # of a Waymo accepted a median gap of 7.58 m against 9.57 m ahead of a driver, 0.79 (*derived*; speed-matched 7.51 /
    # 9.25 = 0.81); its transfer to a yield line is an *assumption*. Reddy et al. 2025 (EJTIR 25(2), Model 2, Table 5
    # and Sec. 4.3): in a simulator drivers accepted gaps in front of recognisable AVs less readily than in front of
    # drivers (at the 90 % level): above 1. Range 0.5-1.5: *assumption*.
    ParamSpec("exploit_gap", "Gap ahead of AVs", 0.5, 1.5, 0.05, "", "Drivers",
              "Drivers entering or changing lane in front of a recognisable AV (CAV, robotaxi, smoother) accept this "
              "many times their usual gap and may force up to 1 / this times the usual braking on it. Below 1 they "
              "exploit cautious AVs (0.79: cut-ins ahead of Waymos, Alhuraish 2026), above 1 they are wary (Reddy "
              "2025)."),
    # Drivers behind a recognisable AV (engine FOLLOW_*): Wang et al. 2023 (Waymo Open Dataset, signalised
    # intersections; via Reddy et al. 2025 p. 3, Wang's paper not read): standstill gap 1.73 m behind an AV against
    # 2.77 m behind a driver, start-up reaction 0.49 against 1.04 s. Hu et al. 2023 corroborate the gap (Sec. III-C,
    # Table IV: jam spacing 6.73 m behind an AV against 7.88 m) and find no general change in response time (Sec.
    # III-D, Table VII: 1.84 against 1.80 s, not significant), so only a start from standstill is quicker. 0 = none, 1
    # = the measured change. Carrying signalised-intersection queues over to following on a roundabout, and the start-up
    # reaction over to the entry follow-up, is an *assumption*.
    ParamSpec("follow_adapt", "Following AVs", 0.0, 1.0, 0.05, "", "Drivers",
              "How far drivers behind a recognisable AV adapt as in Waymo data (Wang 2023): at 1 a standstill gap of "
              "1.73 against 2.77 m, a start from standstill 0.49 against 1.04 s after it moves off, and a follow-up "
              "0.55 s shorter behind an AV entering just before. 0 = as behind any car."),
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
    # Automated vehicles (engine: kinds, the monotone coupling of their draws, RingSim._automated_command)
    ParamSpec("auto_share", "Automated share", 0.0, 1.0, 0.05, "", "AVs",
              "Share of the new vehicles that are automated, all of the AV kind. One software: they "
              "follow only the car in front, without noise or spread (robotaxis keep the drivers' time-gap spread); "
              "a safety override brakes as the IDM would beyond comfortable braking. They enter by their kind's "
              "rules (Entries tab), change lane as drivers. At the same seed a higher share automates the same "
              "vehicles and more.", reset=True),
    ParamSpec("auto_kind", "AV kind", 0, 0, 0, "", "AVs",
              "Kind of the automated vehicles. robotaxi: driverless, by the IDM at a longer time gap and a 1.94 m "
              "longer standstill gap (Hu et al. 2023, Waymo data). acc: driver-supervised cruise control, the PATH "
              "ACC law with a dead time. cav: connected and automated, in CACC strings behind cooperating CAVs, else "
              "by the ACC law.", reset=True, choices=("cav", "robotaxi", "acc")),
    # 1.3 = Waymo's median time headway over the human one, 3.49 / 2.69 s (Hu et al. 2023 Sec. III-B, Fig. 5:
    # lognormal mu 1.25 vs 0.99; *derived*); its mean ratio is 1.26. 1.0-1.6: as close as drivers up to a more
    # cautious fleet (*assumption*; Hu's AV jam spacing is 1.27 x the human one, 10.0 vs 7.88 m, Sec. III-C).
    ParamSpec("robotaxi_headway", "Robotaxi time gap", 1.0, 1.6, 0.05, "", "AVs",
              "Robotaxis' time gap as a multiple of a driver's Following time gap (a truck body keeps 1.5 x on top). "
              "1.3: Waymo's median time headway, 3.49 s against 2.69 s for drivers (Hu et al. 2023, Waymo Open "
              "Dataset); 1 = as close as drivers. Applies at once."),
    # 0.5-2.4 s: Gunter et al. 2019 Table II fits 0.54-2.05 s over production ACC settings (with the flat 2 m margin
    # every slider value is locally stable at every brand factor: engine ACC_DEAD_MAX); openACC's shortest settings
    # 0.9-1.5 s (the openACC database, Makridis et al. 2021, arXiv 2004.06342 Sec. 4.1). 1.1 s: the gap Milanes &
    # Shladover 2014 fitted the law at (Table 1), and the setting 50.4 % of PATH drivers chose (Liu et al. 2018 p. 21).
    ParamSpec("acc_headway", "ACC time gap", 0.5, 2.4, 0.05, "s", "AVs",
              "Time gap T of the ACC cars' PATH law (Milanes & Shladover 2014): a = 0.23 (gap - 2 m - T v) + 0.07 "
              "(leader's v - v), at most 0.4 (target speed - v). 1.1 s: the gap it was fitted at; production cars' "
              "settings span 0.54-2.05 s, the shortest 0.9-1.5 s. Times the brand factor. Fitted at 15-34 m/s: "
              "roundabouts are an extrapolation."),
    # Gunter et al. 2019 Table II: fitted dead times 0.15-0.99 s, median 0.47 s at the shortest settings (derived), but
    # with gains k1 of 0.012-0.071 s^-2; with the PATH law's 0.23 a lone follower goes unstable from 0.8 s at short
    # time gaps (engine ACC_DEAD_MAX, *derived*), so the slider stops at the engine's cap, 0.6 s.
    ParamSpec("acc_delay", "ACC dead time", 0.0, 0.6, 0.1, "s", "AVs",
              "Dead time of the ACC cars: they act on what they measured this long ago (production cars 0.15-0.99 "
              "s, median 0.47 s: Gunter et al. 2019, whose longer ones go with much smaller gains). At most 0.6 s, "
              "brand factor included: with this law's gains a longer one makes a car oscillate behind a steady "
              "leader. Rounded to the time step."),
    # 0.7 s intra-platoon, 1.5 s between platoons in FHWA-OR-RD-22-11's roundabout runs (p. 108); Liu et al. 2018
    # p. 22: 0.6 s 57 %, 0.7 s 24 %, 0.9 s 7 %, 1.1 s 12 % (the PATH drivers' choices).
    ParamSpec("cav_headway", "CACC time gap", 0.5, 1.5, 0.05, "s", "AVs",
              "Time gap of a CAV behind a cooperating CAV, by the PATH CACC law of the HCM 7 CAV factors "
              "(FHWA-OR-RD-22-11: 0.7 s; drivers chose 0.6-1.1 s; strings amplify below 0.67 s). The CAV one past Max "
              "string length keeps 1.5 s and leads a new string. Behind any other vehicle a CAV follows by the ACC "
              "law at its own 1.1, 1.6 or 2.2 s."),
    # Cooperation
    ParamSpec("companies", "Companies", 1, 4, 1, "", "Coop",
              "Companies (brands) the automated vehicles belong to: each new one joins one at random, all equally "
              "likely. With ACC cars or CAVs, Brand spread sets how their settings differ and Between companies how "
              "far CAVs of different companies cooperate; otherwise the company only labels them (hover a car).",
              reset=True),
    # The SAE J3216 classes of cooperative driving automation (*assumption* for what each does here: the capability
    # table in engine COOP_CLASSES). B by default: the HCM 7 CAV basis had CACC and V2V route-conflict anticipation
    # (FHWA-OR-RD-22-11 Sec. 7.2.2).
    ParamSpec("coop_class", "Cooperation", 0, 0, 0, "", "Coop",
              "How far CAVs of one company cooperate (SAE J3216). none: no V2X, the ACC law. A status-sharing: CACC "
              "strings, a shorter follow-up behind a CAV entrant. B intent-sharing (the HCM 7 basis): + route-conflict "
              "anticipation at entries. C agreement-seeking, D prescriptive (every pair, any company): gap creation, "
              "not modelled yet (as B).",
              choices=("none", "A", "B", "C", "D")),
    ParamSpec("coop_between", "Between companies", 0, 0, 0, "", "Coop",
              "Most cooperation between CAVs of different companies: same = as within a company (Cooperation); a "
              "class caps it (none: CAVs of different companies never form CACC strings). Acts only with two or "
              "more companies, and not at Cooperation D (prescribed for every pair).",
              choices=("same", "none", "A", "B", "C", "D")),
    # 0.6 around a mean of 1.3 s gives 0.52-2.08 s, Gunter et al. 2019 Table II's 0.54-2.05 s (derived).
    ParamSpec("brand_spread", "Brand spread", 0.0, 0.6, 0.05, "", "Coop",
              "How much the companies' ACC settings differ: company k of n multiplies its ACC time gaps (a CAV's "
              "ACC-mode gap too, not CACC gaps) and ACC dead time by 1 + spread (2k/(n-1) - 1), evenly around 1 (0.6 "
              "around 1.3 s spans production ACC time gaps, 0.54-2.05 s: Gunter et al. 2019). 0 = alike."),
    # 8: FHWA-OR-RD-22-11 Sec. 6.2.2.1 (and its signal runs) with the same CACC; 10: Liu et al. 2018 p. 24 and the
    # FHWA freeway runs; its roundabout chapter (Ch. 7) states none. (On the Coop tab: strings are status-sharing
    # cooperation, class A; the AVs tab would overflow at 1280 x 720.)
    ParamSpec("platoon_max", "Max string length", 1, 20, 1, "cars", "Coop",
              "Most CAVs in one CACC string: the one past it keeps the 1.5 s gap between strings and leads a new "
              "string (FHWA-OR-RD-22-11 uses 8 with the same CACC, Liu et al. 2018 10; its roundabout runs state "
              "none). 1 = no strings: every CAV behind a CAV keeps 1.5 s."),
    # Entries by kind (engine RingSim._entry_check, _follow_up_slack). FHWA-OR-RD-22-11 Sec. 7.3.2 (p. 117, Fig. 7.7):
    # at higher CAV shares the accepted gaps stay as they were, the rejected ones shrink (CACC strings' 0.7 s gaps), so
    # the Raff critical headway falls (Table 7.1: 4.08 -> 3.48 s) with no change in gap acceptance: 1.0. Guerrieri et
    # al. 2024 calibrate HCM 7 with tc 4.2 against 4.98 s at 100 % (4.2 / 4.98 = 0.84, *derived*; 0.85 on the slider's
    # steps; the model's entry gap is a lag, not tc, so the factor carries over only roughly). Range 0.6-1.2:
    # *assumption*.
    ParamSpec("cav_entry_gap", "CAV entry gap", 0.6, 1.2, 0.05, "", "Entries",
              "A CAV's critical gap as a multiple of Entry gap. 1: in the runs behind the HCM 7 CAV factors CAVs "
              "accepted the gaps drivers accept (their lower critical headway, 3.48 against 4.08 s, came from short "
              "gaps rejected in CACC strings: FHWA-OR-RD-22-11 Sec. 7.3.2); 0.85 gives Guerrieri et al. 2024's 4.2 "
              "against 4.98 s."),
    # FHWA-OR-RD-22-11 Table 7.1 (single-lane entry): follow-up headway 2.62 s at 0 % to 1.93 s at 100 % CAVs (x0.737;
    # 1 / fA = 1 / 1.35 = 0.741, Table 1.7), from waiting CAVs entering one gap as a CACC group (p. 117). On the
    # realised spacing (engine follow_up_steps: a Follow-up time of 2.5 s gives 2.6 s), so x0.74 gives 1.9 s. The slider
    # stops at 0.7 (1.8 s, ring speed as at 0.74): below it the realised follow-up stays near 1.74 s and the entering
    # string slows the circle (6.5 -> 5.0 m/s at 0.5-0.67; *derived*: us_single_lane capacity test, 100 % CAVs at
    # Cooperation B, no conflicting flow, seeds 0-3). --set may still go below.
    ParamSpec("cav_follow_up", "CAV follow-up", 0.7, 1.0, 0.01, "", "Entries",
              "Follow-up of a CAV behind a CAV it cooperates with (Cooperation A or more) that entered just before it "
              "and is still directly ahead, x the drivers' realised follow-up (2.6 s for 2.5 s). 0.74: "
              "FHWA-OR-RD-22-11 measured 2.62 s at 0 % and 1.93 s at 100 % CAVs (waiting CAVs entering one gap as a "
              "CACC group). 1 = as drivers."),
    # *assumption*: no source measures robotaxi gap acceptance; 1.3 as its time gap (Hu et al. 2023 Sec. III-B, Fig. 5:
    # median time headway 3.49 against 2.69 s). 0.8-1.6: assertive to cautious fleets. Below 1 a head entering before
    # the drivers' follow-up has passed also needs its own steady-state gap ahead (engine _entry_check), so 0.8-0.9
    # realise about 2.4 s at us_single_lane's speeds, not 2.1-2.3 s (*derived*).
    ParamSpec("robotaxi_entry", "Robotaxi entry", 0.8, 1.6, 0.05, "", "Entries",
              "A robotaxi's critical gap and follow-up as multiples of the drivers' (Entry gap; the realised "
              "follow-up). 1.3 by assumption, as its time gap (Waymo keeps 1.3 x a driver's: Hu et al. 2023); no "
              "source measures robotaxi gap acceptance. Below 1 an assertive fleet."),
    # The weather policy of the driverless kinds (engine av_road_factors; *assumption*: no source gives how a CAV or a
    # robotaxi drives on wet, snowy or icy roads): each behavioural factor of the drivers' (road_factors, after Weather
    # effect) f becomes 1 + this x (f - 1) for them, so 0 = AVs that ignore the road (unchanged gaps and speeds), 1 = as
    # the drivers, 2 = twice the drivers' caution; the grip limits (cornering, braking) apply in full whatever the kind.
    ParamSpec("av_weather_scale", "AV weather effect", 0.0, 2.0, 0.05, "", "Entries",
              "How strongly a wet, snowy or icy road changes how driverless vehicles (CAVs, robotaxis) drive and "
              "enter: each of the drivers' factors on desired speed, acceleration, comfortable braking, time gap, "
              "entry gap and follow-up becomes 1 + this x (factor - 1). 1 = as the drivers, 0 = as if dry, 2 = twice "
              "as cautious."),
    # *assumption*: a driverless vehicle hands the decision to enter on snow or ice to remote assistance (no source
    # gives how long that takes; 0-120 s brackets a confirmation within a minute or two). Engine RingSim._handed_over.
    ParamSpec("av_handover", "AV handover", 0, 120, 5, "s", "Entries",
              "On snow or ice a driverless vehicle (CAV, robotaxi) at the head of a queue waits this long at the yield "
              "line before it may enter, for remote assistance to confirm it (an assumption: no source gives a time). "
              "Then it enters by its usual rules. 0 = off."),
    # Wave smoothers (engine RingSim._law_smoother, _sync_smoothers). Stern et al. 2018 (TR-C 89, arXiv 1705.01693):
    # one automated car among 21 (Sec. 2.1) cut the speed SD 3.31 -> 0.64 m/s (-80.8 %, Table 3, Exp. A).
    ParamSpec("smoothers", "Smoothers", 0, 20, 1, "cars", "Smoothers",
              "Automated cars that damp stop-and-go waves: one among 21 cut the speed spread by 81 % (Stern et al. "
              "2018). Picked without chance: the human-driven cars (not trucks) nearest evenly spaced points round the "
              "ring. They never leave, change lane or get removed. Changed live, the nearest are converted or handed "
              "back."),
    # Stern 2018 Sec. 3.1 (FollowerStopper, Eqs. 1-2) and Sec. 3.2 (PI with saturation, Eqs. 3-5); the gap cap
    # max(120 m, 6 s x v) is Lichtle et al. 2024's gap-closing threshold (IEEE ITSC 2023, arXiv 2401.09666, Sec.
    # III-A.2), where their wrapper accelerates at 1.5 m/s2; here the driver's IDM takes over (*model*).
    ParamSpec("smoother_law", "Smoother law", 0, 0, 0, "", "Smoothers",
              "followerstopper: Smoother speed unless the gap is short, then the speed of the car ahead or a stop "
              "(Stern et al. 2018). pi: the mean of its own speed over 38 s, saturated near the car ahead. quiet: a "
              "placebo, its driver's own model without noise. Beyond Lichtle et al. 2024's gap cap, max(120 m, 6 s "
              "x speed): the driver's model.",
              choices=("followerstopper", "pi", "quiet")),
    # Stern 2018 Sec. 4.2 (Exp. A): U = 6.5, 7.0, 7.5, 8.0, then 7.5 m/s; 7.5 best ('matches the average traffic speed
    # almost precisely'), 6.5 damped the wave only partly, 8.0 brought it back.
    ParamSpec("smoother_speed", "Smoother speed", 2.0, 15.0, 0.25, "m/s", "Smoothers",
              "FollowerStopper's set speed U, which it never exceeds. Stern et al. 2018 settled the waves best at 7.5 "
              "m/s, near the ring's mean speed; 6.5 damped them only partly and 8.0 brought them back. Never above the "
              "lane's comfortable cornering speed."),
    # Stern 2018 switched the controller on once the waves had formed (Exp. A: waves from 79 s, control from 126 s,
    # Tables 1 and 3).
    ParamSpec("smoother_start", "Smoother start", 0, 600, 10, "s", "Smoothers",
              "When the smoothers' controllers switch on. Until then each drives as its human driver does, noise and "
              "all (Stern et al. 2018 let the waves form first, then switched the car's controller on). 0 = from the "
              "start."),
    ParamSpec("smoother_lane", "Smoother lane", -1, 4, 1, "", "Smoothers",
              "Lane the smoothers are picked in: -1 = spread round the lanes, the innermost first; 0 = the innermost "
              "lane, 1 the next, and so on (beyond the ring's lanes: the outermost). Changed live, they are picked "
              "again."),
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


def range_text(spec: ParamSpec, dash: str = "-") -> str:
    """'0.6-2.4': a numeric setting's slider range as the help texts show it (``dash`` between the ends); with a
    negative end, which a dash would garble ('-1-4'), '-1 to 4'."""
    return f"{spec.lo:g} to {spec.hi:g}" if spec.lo < 0 else f"{spec.lo:g}{dash}{spec.hi:g}"


def axis_label(name: str) -> str:
    """'Ring radius (m)', 'Lanes': a setting's label with its unit, as on the chart axes (the name
    itself for a column that is no setting)."""
    spec = PARAM_BY_NAME.get(name)
    if spec is None:
        return name
    return f"{spec.label} ({spec.unit})" if spec.unit else spec.label


# Settings that act only while another setting is above 0: (that setting, the reason shown meanwhile).
# Without entrances (n_arms = 0: the ring-road experiments) nobody enters, queues or leaves.
NEEDS_NONZERO = {"truck_length": ("truck_share", "no trucks"), "truck_follow_up": ("truck_share", "no trucks"),
                 **{name: ("n_arms", "no entrances") for name in ("exit_prob", "entry_gap", "follow_up", "turn_speed",
                                                                  "entry_rate", "demand_profile", "demand_imbalance",
                                                                  "max_queue", "subject_arm", "subject_exiting",
                                                                  "cav_entry_gap", "cav_follow_up", "robotaxi_entry",
                                                                  "av_handover")},
                 **{name: ("auto_share", "no automated vehicles") for name in ("auto_kind", "companies")},
                 # (the driverless kinds' factors scale the drivers' after the Weather effect: all 1 at 0)
                 "av_weather_scale": ("weather_scale", "no weather effect"),
                 **{name: ("smoothers", "no smoothers") for name in ("smoother_law", "smoother_speed", "smoother_start",
                                                                     "smoother_lane")}}
# Settings that act only on automated vehicles of one kind: (the kind, the reason shown while there are none, for
# want of automated vehicles (auto_share = 0) or of that kind: one reason, so a sweep of either still warns).
NEEDS_KIND = {**{name: ("robotaxi", "no robotaxis") for name in ("robotaxi_headway", "robotaxi_entry")},
              **{name: ("acc", "no ACC cars") for name in ("acc_headway", "acc_delay")},
              **{name: ("cav", "no CAVs") for name in ("cav_headway", "platoon_max", "coop_class", "coop_between",
                                                       "cav_entry_gap", "cav_follow_up")}}
# Settings that act only at entrances besides another reason (NEEDS_NONZERO names that one): n_arms above 0.
NEEDS_ENTRANCES = ("truck_follow_up",)
# Settings that act only on the kinds with ACC or CACC settings (the brand factors scale those, not a robotaxi's).
NEEDS_CONTROLLER = ("brand_spread",)
# Settings that act only while CAVs of one company cooperate (CACC strings and the CAV follow-up need class A or
# more: coop_class; the class between companies is at most the class within one).
NEEDS_COOP = ("cav_headway", "platoon_max", "coop_between", "cav_follow_up")
# Settings that act only below Cooperation D (D binds every CAV pair, whatever the company: engine COOP_PRESCRIPTIVE).
NEEDS_BELOW_D = ("coop_between",)
# Settings that act only while a string can hold two CAVs (platoon_max 1: every CAV behind a CAV keeps 1.5 s).
NEEDS_STRINGS = ("cav_headway",)
# Settings that act only between two or more companies.
NEEDS_COMPANIES = ("coop_between", "brand_spread")


# The highest cooperation class with behaviour of its own so far (engine COOP_CLASSES): C and D act as B until gap
# creation is modelled, when this rises with it.
COOP_ACTING = "B"


def companies_differ(cfg: SimConfig) -> bool:
    """Whether the company a vehicle joins changes how it drives under ``cfg``: the ACC cars' or CAVs' settings
    differ by brand (brand_spread above 0), or CAVs of different companies cooperate less than CAVs of one
    company, counting only the classes up to COOP_ACTING (C and D act as B so far: so coop_between none or A where
    coop_class is B or more; D is never capped, engine COOP_PRESCRIPTIVE). Otherwise Companies only labels them."""
    if not (cfg.auto_share > 0 and cfg.auto_kind in ("acc", "cav")):
        return False
    if cfg.brand_spread > 0:
        return True
    rank = PARAM_BY_NAME["coop_class"].choices.index  # ("none", "A", "B", "C", "D")
    within = min(rank(cfg.coop_class), rank(COOP_ACTING))
    uncapped = cfg.coop_between == "same" or cfg.coop_class == "D"
    between = min(rank(cfg.coop_class if uncapped else cfg.coop_between), within)
    return cfg.auto_kind == "cav" and between < within
# Settings that act only on smoothers of the FollowerStopper law (PI sets its speed from its own mean; quiet is its
# driver's model).
NEEDS_FS = ("smoother_speed",)
# Settings that act only on a ring of two or more lanes.
NEEDS_LANES = ("smoother_lane",)
# Settings that act only off a dry road: the weather effect scales the wet, snow and ice factors (dry are all 1).
NEEDS_WEATHER = ("weather_scale", "av_weather_scale")
# Settings that act only on snow or ice (the driverless vehicles' handover at the yield line).
NEEDS_SNOW = ("av_handover",)
SNOW_ROADS = ("snow", "ice")
# Settings that act only on driverless vehicles (engine DRIVERLESS: CAVs and robotaxis; an ACC car has a driver).
NEEDS_DRIVERLESS = ("av_weather_scale", "av_handover")
DRIVERLESS_KINDS = ("cav", "robotaxi")
# Settings that act only on drivers meeting a recognisable AV (engine RECOGNISABLE: CAVs, robotaxis and smoothers; an
# ACC car looks like any car).
NEEDS_RECOGNISABLE = ("exploit_gap", "follow_adapt")
# Settings that act only where drivers merge in front of another vehicle: at an entrance or by changing lane.
NEEDS_MERGES = ("exploit_gap",)
# Settings the capacity test (subject_arm on, in open mode with entrances) overrides: it fixes every route.
TEST_FIXED = ("exit_prob",)
# Settings that act only in the capacity test (subject_arm on).
NEEDS_TEST = ("subject_exiting",)
IDLE_WHY = {  # (reason, one setting or several) -> what it says about every run of a sweep
    "open mode only": ("acts only in open mode", "act only in open mode", "every run here is closed"),
    "no entrances": ("does nothing without entrances", "do nothing without entrances",
                     "every run here has none (n_arms = 0)"),
    "no trucks": ("does nothing without trucks", "do nothing without trucks", "every run here has truck_share = 0"),
    "dry road": ("does nothing on a dry road", "do nothing on a dry road", "every run here is on a dry road"),
    "capacity test": ("does nothing in the capacity test (it fixes every route)",
                      "do nothing in the capacity test (it fixes every route)", "every run here runs it"),
    "no automated vehicles": ("does nothing without automated vehicles", "do nothing without automated vehicles",
                              "every run here has auto_share = 0"),
    "no robotaxis": ("acts only on robotaxis", "act only on robotaxis",
                     "no run here has any (auto_kind = robotaxi with auto_share above 0)"),
    "no ACC cars": ("acts only on ACC cars", "act only on ACC cars",
                    "no run here has any (auto_kind = acc with auto_share above 0)"),
    "no CAVs": ("acts only on CAVs", "act only on CAVs", "no run here has any (auto_kind = cav with auto_share above 0)"),
    "no ACC cars or CAVs": ("acts only on ACC cars and CAVs", "act only on ACC cars and CAVs",
                            "no run here has any (auto_kind = acc or cav with auto_share above 0)"),
    "no cooperation": ("does nothing while CAVs do not cooperate", "do nothing while CAVs do not cooperate",
                       "every run here has coop_class = none"),
    "no strings": ("does nothing without CACC strings", "do nothing without CACC strings",
                   "every run here has platoon_max = 1"),
    "one company": ("acts only between companies", "act only between companies", "every run here has companies = 1"),
    "companies alike": ("does nothing while the companies drive alike", "do nothing while the companies drive alike",
                        "no run here has brand_spread above 0 (with ACC cars or CAVs) or coop_between = none (with "
                        "cooperating CAVs)"),
    "no smoothers": ("does nothing without smoothers", "do nothing without smoothers",
                     "every run here has smoothers = 0"),
    "no FollowerStopper": ("acts only on FollowerStopper smoothers", "act only on FollowerStopper smoothers",
                           "every run here has smoother_law = pi or quiet"),
    "one lane": ("does nothing on a single-lane ring", "do nothing on a single-lane ring",
                 "every run here has n_lanes = 1"),
    "class D": ("does nothing at Cooperation D (prescribed for every CAV pair, whatever the company)",
                "do nothing at Cooperation D (prescribed for every CAV pair, whatever the company)",
                "every run here has coop_class = D"),
    "no capacity test": ("acts only in the capacity test", "act only in the capacity test",
                         "every run here has subject_arm = off"),
    "no driverless vehicles": ("acts only on driverless vehicles (CAVs, robotaxis)",
                               "act only on driverless vehicles (CAVs, robotaxis)",
                               "no run here has any (auto_kind = cav or robotaxi with auto_share above 0)"),
    "no recognisable AVs": ("acts only on drivers meeting recognisable AVs (CAVs, robotaxis, smoothers)",
                            "act only on drivers meeting recognisable AVs (CAVs, robotaxis, smoothers)",
                            "no run here has any (auto_kind = cav or robotaxi with auto_share above 0, or smoothers)"),
    "no merges": ("does nothing without entrances or a second lane", "do nothing without entrances or a second lane",
                  "every run here is a single-lane ring without entrances"),
    "no snow or ice": ("acts only on snow or ice", "act only on snow or ice", "no run here is on snow or ice"),
    "no weather effect": ("does nothing without a weather effect", "do nothing without a weather effect",
                          "every run here has weather_scale = 0"),
}


def inactive_reason(name: str, cfg: SimConfig) -> str:
    """Why the setting ``name`` has no effect with the settings ``cfg`` ("" if it has one): "open mode
    only" (``applies_in``), "no entrances", "no trucks", "no automated vehicles" or "no weather effect"
    (NEEDS_NONZERO), "no robotaxis", "no ACC cars" or "no CAVs" (NEEDS_KIND: no automated vehicle of the kind,
    auto_share 0 included; "no entrances" first for NEEDS_ENTRANCES), "no ACC cars or CAVs" (NEEDS_CONTROLLER), "no
    cooperation" (NEEDS_COOP: coop_class none), "no strings" (NEEDS_STRINGS: platoon_max 1), "one company"
    (NEEDS_COMPANIES), "class D" (NEEDS_BELOW_D), "companies alike" (companies, while :func:`companies_differ`
    is false), "no smoothers" (NEEDS_NONZERO), "no driverless vehicles" (NEEDS_DRIVERLESS), "no recognisable AVs"
    (NEEDS_RECOGNISABLE), "no merges" (NEEDS_MERGES: no entrance and one lane), "no FollowerStopper" (NEEDS_FS), "one
    lane" (NEEDS_LANES), "dry road" (NEEDS_WEATHER), "no snow or ice" (NEEDS_SNOW), "capacity test" (TEST_FIXED) or
    "no capacity test" (NEEDS_TEST); what it reads:
    :func:`inactive_inputs`. The main viewer tags its controls with it; the sweep tools warn of a varied setting
    that does nothing in every run."""
    spec = PARAM_BY_NAME.get(name)
    if spec is not None and spec.applies_in and cfg.traffic_mode != spec.applies_in:
        return f"{spec.applies_in} mode only"
    if name in NEEDS_NONZERO:
        other, reason = NEEDS_NONZERO[name]
        if not getattr(cfg, other) > 0:
            return reason
    if name in NEEDS_ENTRANCES and not cfg.n_arms > 0:
        return "no entrances"
    if name in NEEDS_KIND and not (cfg.auto_share > 0 and cfg.auto_kind == NEEDS_KIND[name][0]):
        return NEEDS_KIND[name][1]
    if name in NEEDS_CONTROLLER and not (cfg.auto_share > 0 and cfg.auto_kind in ("acc", "cav")):
        return "no ACC cars or CAVs"
    if name in NEEDS_DRIVERLESS and not (cfg.auto_share > 0 and cfg.auto_kind in DRIVERLESS_KINDS):
        return "no driverless vehicles"
    if name in NEEDS_RECOGNISABLE and not (cfg.auto_share > 0 and cfg.auto_kind in DRIVERLESS_KINDS
                                           or cfg.smoothers > 0):  # (the recognisable kinds: the driverless ones too)
        return "no recognisable AVs"
    if name in NEEDS_MERGES and not (cfg.n_arms > 0 or cfg.n_lanes > 1):
        return "no merges"
    if name in NEEDS_COOP and cfg.coop_class == "none":
        return "no cooperation"
    if name in NEEDS_STRINGS and cfg.platoon_max <= 1:
        return "no strings"
    if name in NEEDS_COMPANIES and cfg.companies <= 1:
        return "one company"
    if name in NEEDS_BELOW_D and cfg.coop_class == "D":
        return "class D"
    if name == "companies" and not companies_differ(cfg):
        return "companies alike"
    if name in NEEDS_FS and cfg.smoother_law != "followerstopper":
        return "no FollowerStopper"
    if name in NEEDS_LANES and cfg.n_lanes <= 1:
        return "one lane"
    if name in NEEDS_WEATHER and cfg.road_condition == "dry":
        return "dry road"
    if name in NEEDS_SNOW and cfg.road_condition not in SNOW_ROADS:
        return "no snow or ice"
    if name in TEST_FIXED and cfg.subject_arm == "on" and cfg.traffic_mode == "open" and cfg.n_arms > 0:
        return "capacity test"
    if name in NEEDS_TEST and cfg.subject_arm != "on":
        return "no capacity test"
    return ""


def inactive_inputs(name: str) -> set[str]:
    """The settings other than ``name`` that :func:`inactive_reason` reads for it (kept in step with it): those
    that must vary, together, for ``name`` to act in a sweep whose base settings leave it idle."""
    spec = PARAM_BY_NAME.get(name)
    inputs = {"traffic_mode"} if spec is not None and spec.applies_in else set()
    if name in NEEDS_NONZERO:
        inputs.add(NEEDS_NONZERO[name][0])
    if name in NEEDS_ENTRANCES:
        inputs.add("n_arms")
    if name in NEEDS_KIND or name in NEEDS_CONTROLLER or name in NEEDS_DRIVERLESS:
        inputs |= {"auto_share", "auto_kind"}
    if name in NEEDS_RECOGNISABLE:
        inputs |= {"auto_share", "auto_kind", "smoothers"}
    if name in NEEDS_MERGES:
        inputs |= {"n_arms", "n_lanes"}
    if name in NEEDS_COOP or name in NEEDS_BELOW_D:
        inputs.add("coop_class")
    if name in NEEDS_STRINGS:
        inputs.add("platoon_max")
    if name in NEEDS_COMPANIES:
        inputs.add("companies")
    if name == "companies":
        inputs |= {"auto_share", "auto_kind", "brand_spread", "coop_class", "coop_between"}
    if name in NEEDS_FS:
        inputs.add("smoother_law")
    if name in NEEDS_LANES:
        inputs.add("n_lanes")
    if name in NEEDS_WEATHER or name in NEEDS_SNOW:
        inputs.add("road_condition")
    if name in TEST_FIXED:
        inputs |= {"subject_arm", "traffic_mode", "n_arms"}
    if name in NEEDS_TEST:
        inputs.add("subject_arm")
    return inputs - {name}


def idle_note(names: list[str], reason: str) -> str:
    """'entry_rate acts only in open mode, and every run here is closed: its values change nothing'
    (``names`` as the caller shows them; ``reason`` from :func:`inactive_reason`)."""
    one, several, runs = IDLE_WHY.get(reason, (f"does nothing here ({reason})", f"do nothing here ({reason})",
                                               "in every run"))
    who = ", ".join(names[:-1]) + f" and {names[-1]}" if len(names) > 1 else names[0]
    return f"{who} {one if len(names) == 1 else several}, and {runs}: {'its' if len(names) == 1 else 'their'} " \
           "values change nothing"
