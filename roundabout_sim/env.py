"""Agent-based roundabout traffic model built on highway-env.

Every car is an independent agent (car-following via the Intelligent Driver
Model, plus roundabout right-of-way rules). There is no RL-controlled ego vehicle:
the environment keeps a target number of cars in the system by spawning new
agents at the four entry roads and removing them once they leave via an exit.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from highway_env import utils
from highway_env.envs.roundabout_env import RoundaboutGenericEnv
from highway_env.road.road import Road

from roundabout_sim.agents import APPROACH_SEGMENTS, RING_NODES, on_ring

# Road segments where agents enter the model, and the exit roads they leave by.
ENTRY_LANES = [("ser", "ses"), ("eer", "ees"), ("ner", "nes"), ("wer", "wes")]
EXIT_LANES = {("sxs", "sxr"), ("exs", "exr"), ("nxs", "nxr"), ("wxs", "wxr")}
DESTINATIONS = ["sxr", "exr", "nxr", "wxr"]

# Lanes used to scatter the initial population (entry roads and the ring).
INITIAL_LANES = ENTRY_LANES + [
    ("se", "ex"), ("ex", "ee"), ("ee", "nx"), ("nx", "ne"),
    ("ne", "wx"), ("wx", "we"), ("we", "sx"), ("sx", "se"),
]

# Diverging driver colours: orange (slow) <- grey (average) -> blue (fast).
SLOW_COLOR = np.array([235, 104, 52])
MID_COLOR = np.array([200, 199, 192])
FAST_COLOR = np.array([42, 120, 214])
CRASH_COLOR = (208, 59, 59)


class RoundaboutRoad(Road):
    """Road with two fixes to highway-env's leader search.

    * Ring traffic ignores cars on the approach roads. Leaders are matched
      with a lateral margin, so a car waiting at a yield line looks like a
      stopped car in the ring lane. Ring cars instead respond to entering cars
      explicitly (see ``YieldingIDMVehicle._make_room_acceleration``).
    * Near the end of a segment IDMVehicle also queries the *next* segment,
      but measures the distance along its current lane. On a circular lane
      that projection wraps around and turns cars far down an exit road into
      phantom leaders a few metres ahead. The connected-lane search on the
      current lane already covers the next segment, so such queries are
      skipped. (Same-segment queries, i.e. MOBIL lane changes, still run.)
    """

    def neighbour_vehicles(self, vehicle, lane_index=None):
        lane_index = lane_index or vehicle.lane_index
        if lane_index and vehicle.lane_index and lane_index[:2] != vehicle.lane_index[:2]:
            return None, None
        if not lane_index or lane_index[0] not in RING_NODES or lane_index[1] not in RING_NODES:
            return super().neighbour_vehicles(vehicle, lane_index)
        everyone = self.vehicles
        self.vehicles = [v for v in everyone if v.lane_index[:2] not in APPROACH_SEGMENTS]
        try:
            return super().neighbour_vehicles(vehicle, lane_index)
        finally:
            self.vehicles = everyone


class RoundaboutTrafficEnv(RoundaboutGenericEnv):
    """Roundabout populated only by autonomous IDM agents.

    Live-tunable parameters (read from ``self.config`` every step):
      * ``vehicles_count``  - target number of cars in the system
      * ``mean_speed``      - mean desired speed of drivers [m/s]
      * ``speed_std``       - std. dev. of desired speed across drivers [m/s]
      * ``roundabout_speed_factor`` - fraction of desired speed kept on the ring
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        utils.update_config(
            config,
            {
                "other_vehicles_type": "roundabout_sim.agents.YieldingIDMVehicle",
                # Single-lane ring. highway-env's multi-lane ring has no lane
                # discipline: inner-lane cars cut across the outer lane to exit.
                "roundabout_lanes": 1,
                # Let cars see leaders on the next lane segment (off by default
                # in highway-env, which causes rear-end crashes at boundaries).
                "neighbour_vehicles_connected_lanes": True,
                "vehicles_count": 20,
                "mean_speed": 12.0,
                "speed_std": 2.0,
                # Drivers slow to this fraction of their desired speed on the
                # roundabout's curves (keeps the spread between drivers).
                "roundabout_speed_factor": 0.6,
                "spawn_clearance": 15.0,  # [m] free road needed at an entry to spawn
                "crash_removal_time": 3.0,  # [s] crashed cars are cleared after this
                "throughput_window": 60.0,  # [s] rolling window for exits/minute
                "simulation_frequency": 15,
                "policy_frequency": 15,  # one road update per step() call
                "duration": float("inf"),
                "screen_width": 800,
                "screen_height": 800,
                "scaling": 2.8,
                "centering_position": [0.5, 0.5],
                "offscreen_rendering": True,
            },
        )
        return config

    # ------------------------------------------------------------------ setup
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        gym.Env.reset(self, seed=seed)
        if options and "config" in options:
            self.configure(options["config"])
        self.time = 0.0
        self.steps = 0
        self.total_spawned = 0
        self.total_exited = 0
        self.total_crashes = 0
        self._exit_times: deque[float] = deque()
        self._travel_times: deque[float] = deque(maxlen=200)
        self.trips: list[tuple[float, float]] = []  # (exit time, travel time) per car
        self._reset()
        return self.stats(), {}

    def _make_road(self) -> None:
        super()._make_road()
        road = self.road
        self.road = RoundaboutRoad(
            network=road.network,
            np_random=road.np_random,
            record_history=road.record_history,
            neighbour_vehicles_connected_lanes=road.neighbour_vehicles_connected_lanes,
        )

    def _make_vehicles(self) -> None:
        self.controlled_vehicles = []  # no ego: every car is an autonomous agent
        lanes = INITIAL_LANES
        placed: list[np.ndarray] = []
        for _ in range(self.config["vehicles_count"]):
            for _ in range(20):
                start, end = lanes[self.np_random.integers(len(lanes))]
                index = self.np_random.integers(len(self.road.network.graph[start][end]))
                lane = self.road.network.get_lane((start, end, index))
                s = self.np_random.uniform(5.0, max(5.0, lane.length - 5.0))
                pos = lane.position(s, 0.0)
                if all(np.linalg.norm(pos - p) > 10.0 for p in placed):
                    # Start from rest: cars this close at mismatched speeds collide.
                    self._add_agent((start, end, index), s, initial_speed=0.0)
                    placed.append(pos)
                    break

    def _add_agent(
        self, lane_index: tuple, longitudinal: float, initial_speed: float | None = None
    ) -> None:
        vehicle_cls = utils.class_from_path(self.config["other_vehicles_type"])
        # Each driver keeps a fixed "temperament" z ~ N(0, 1); desired speed is
        # mean + z * std, so moving the sliders retunes every agent live.
        z = float(np.clip(self.np_random.normal(), -2.5, 2.5))
        speed = self._desired_speed(z)
        start_speed = speed if initial_speed is None else min(speed, initial_speed)
        vehicle = vehicle_cls.make_on_lane(
            self.road, lane_index, longitudinal=longitudinal, speed=start_speed
        )
        vehicle.driver_z = z
        vehicle.spawn_time = self.time
        self._retune(vehicle)
        vehicle.target_speed = speed
        vehicle.color = self._color_for(z)
        vehicle.plan_route_to(DESTINATIONS[self.np_random.integers(len(DESTINATIONS))])
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        self.total_spawned += 1

    def _desired_speed(self, z: float) -> float:
        return float(max(1.0, self.config["mean_speed"] + z * self.config["speed_std"]))

    def _retune(self, vehicle) -> None:
        """Apply the current speed settings to a driver."""
        vehicle.desired_speed = self._desired_speed(vehicle.driver_z)
        vehicle.ring_speed = vehicle.desired_speed * self.config["roundabout_speed_factor"]

    @staticmethod
    def _color_for(z: float) -> tuple[int, int, int]:
        t = float(np.clip(z, -2.0, 2.0)) / 2.0
        pole = FAST_COLOR if t > 0 else SLOW_COLOR
        return tuple(int(c) for c in (1 - abs(t)) * MID_COLOR + abs(t) * pole)

    # ------------------------------------------------------------- dynamics
    def step(self, action=None):
        """Advance the traffic model by 1 / policy_frequency seconds.

        ``action`` is ignored (there is no ego vehicle); it is kept so the
        class still satisfies the gymnasium API.
        """
        dt = 1.0 / self.config["simulation_frequency"]
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        for _ in range(frames):
            for v in self.road.vehicles:
                self._retune(v)
            self.road.act()
            self.road.step(dt)
            self.time += dt
            self.steps += 1
            self._handle_crashes()
            self._remove_exited()
            self._spawn_inflow()
        return self.stats(), 0.0, False, self.time >= self.config["duration"], {}

    def _handle_crashes(self) -> None:
        for v in self.road.vehicles:
            if v.crashed and not hasattr(v, "crash_time"):
                v.crash_time = self.time
                v.color = CRASH_COLOR
                self.total_crashes += 1
        limit = self.config["crash_removal_time"]
        self.road.vehicles = [
            v for v in self.road.vehicles
            if not (v.crashed and self.time - v.crash_time > limit)
        ]

    def _remove_exited(self) -> None:
        keep = []
        for v in self.road.vehicles:
            lane = self.road.network.get_lane(v.lane_index)
            on_exit = v.lane_index[:2] in EXIT_LANES
            done = on_exit and lane.local_coordinates(v.position)[0] > lane.length - 5.0
            if done or np.linalg.norm(v.position) > 300.0:
                self.total_exited += 1
                self._exit_times.append(self.time)
                self._travel_times.append(self.time - v.spawn_time)
                self.trips.append((self.time, self.time - v.spawn_time))
            else:
                keep.append(v)
        self.road.vehicles = keep

    def _spawn_inflow(self) -> None:
        """Top the system back up to ``vehicles_count`` from random entries."""
        deficit = self.config["vehicles_count"] - len(self.road.vehicles)
        if deficit <= 0:
            return
        clearance = self.config["spawn_clearance"]
        for lane_id in self.np_random.permutation(len(ENTRY_LANES)):
            if deficit <= 0:
                break
            lane_index = (*ENTRY_LANES[lane_id], 0)
            lane = self.road.network.get_lane(lane_index)
            spawn_point = lane.position(0.0, 0.0)
            if all(np.linalg.norm(v.position - spawn_point) > clearance
                   for v in self.road.vehicles):
                # Don't launch a fast car into the back of a queue: match the
                # speed of any car still near the start of this entry road.
                ahead = [v.speed for v in self.road.vehicles
                         if v.lane_index[:2] == ENTRY_LANES[lane_id]
                         and lane.local_coordinates(v.position)[0] < 4 * clearance]
                self._add_agent(lane_index, 0.0, initial_speed=min(ahead, default=None))
                deficit -= 1

    # -------------------------------------------------------------- metrics
    def stats(self) -> dict:
        window = self.config["throughput_window"]
        while self._exit_times and self._exit_times[0] < self.time - window:
            self._exit_times.popleft()
        speeds = [v.speed for v in self.road.vehicles if not v.crashed]
        ring = sum(on_ring(v) for v in self.road.vehicles)
        queued = sum(v.lane_index[:2] in APPROACH_SEGMENTS and v.speed < 2.0
                     for v in self.road.vehicles)
        circumference = 2 * np.pi * self.config["roundabout_radius"] / 1000.0  # [km]
        elapsed = min(self.time, window)
        return {
            "time": self.time,
            "vehicles": len(self.road.vehicles),
            "ring_vehicles": ring,
            "ring_density": ring / circumference,  # [veh/km]
            "queued": queued,  # cars (nearly) stopped on the approach roads
            "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
            "throughput_per_min": len(self._exit_times) * 60.0 / elapsed if elapsed > 0 else 0.0,
            "mean_travel_time": float(np.mean(self._travel_times)) if self._travel_times else 0.0,
            "spawned": self.total_spawned,
            "exited": self.total_exited,
            "crashes": self.total_crashes,
        }

    def _info(self, obs, action=None) -> dict:
        return {}
