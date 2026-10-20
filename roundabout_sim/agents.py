"""Driver agents for the roundabout model."""

from __future__ import annotations

import numpy as np
from highway_env.vehicle.behavior import IDMVehicle

# Ring nodes, their angle on the circle [deg], and for each entry the exit
# immediately upstream of it (cars taking that exit never reach the entry).
RING_NODES = {"se", "ex", "ee", "nx", "ne", "wx", "we", "sx"}
ALPHA = 24
ENTRY_PHASE = {"se": 90 - ALPHA, "ee": -ALPHA, "ne": -90 - ALPHA, "we": 180 - ALPHA}
UPSTREAM_EXIT = {"se": "sx", "ee": "ex", "ne": "nx", "we": "wx"}
# Approach lanes leading to each entry: straight road, then the curved merge lane.
APPROACH = {
    "se": (("ser", "ses"), ("ses", "se")),
    "ee": (("eer", "ees"), ("ees", "ee")),
    "ne": (("ner", "nes"), ("nes", "ne")),
    "we": (("wer", "wes"), ("wes", "we")),
}
APPROACH_SEGMENTS = {segment for pair in APPROACH.values() for segment in pair}
MERGE_SEGMENTS = {merge for _, merge in APPROACH.values()}
EXIT_CURVES = {("sx", "sxs"), ("ex", "exs"), ("nx", "nxs"), ("wx", "wxs")}


def on_ring(vehicle) -> bool:
    return vehicle.lane_index[0] in RING_NODES and vehicle.lane_index[1] in RING_NODES


def arc_upstream_of(position: np.ndarray, entry: str) -> tuple[float, float]:
    """Signed arc length [m] from ``position`` to an entry's merge point.

    Positive: the position is upstream (the merge point is ahead of it on the
    ring); negative: it is already past the merge point. Also returns the
    radius used, so callers can sanity-check they are on the ring.
    """
    radius = float(np.linalg.norm(position))
    angle = (np.arctan2(position[1], position[0]) - np.deg2rad(ENTRY_PHASE[entry])) % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle * radius, radius


class YieldingIDMVehicle(IDMVehicle):
    """IDM/MOBIL driver with roundabout right-of-way.

    highway-env's stock IDMVehicle has no notion of right-of-way, so cars
    merge blindly into the ring. This agent adds two rules:

    * **Entering cars give way.** While approaching an entry a car treats the
      yield line as a stopped obstacle unless every circulating car is at
      least ``CRITICAL_GAP`` seconds and ``MIN_GAP_DISTANCE`` metres away from
      the merge point. Once it can no longer stop comfortably, or has crossed
      the line, it commits and enters.
    * **Zipper merge.** An entering car is projected onto the ring at its
      merge point minus the distance it still has to drive. Whichever of the
      pair (entering car, ring car) is virtually behind follows the other
      with IDM. Cars waiting at the line are ignored by ring traffic.

    Drivers have a free-road ``desired_speed`` and slow to ``ring_speed`` on
    the curved roundabout lanes (both set by the environment).
    """

    CRITICAL_GAP = 3.0  # [s] smallest time gap to an oncoming ring car accepted
    # [m] smallest distance gap accepted. Without it a slow or queued ring car
    # just upstream looks "infinitely far" in time and gets cut off.
    MIN_GAP_DISTANCE = 15.0
    LOOKAHEAD = 50.0  # [m] start evaluating gaps this far before the yield line
    # [m] where the car's centre stops before the merge point. The merge lane
    # meets the ring at the ring's centreline, so a waiting car must stop
    # this far back for its front corners to clear the circulating lane.
    STOP_OFFSET = 7.0
    MERGE_ZONE = 15.0  # [m] ring cars this close past the merge point block it
    MIN_STANDSTILL_GAP = 2.0  # [m] bumper gap kept to virtual leaders
    APPROACH_DECEL = 1.5  # [m/s2] gentle braking from desired to ring speed

    committed = False  # passed the point of no return at the current entry
    entering = False  # approaching an entry and not yielding (ring cars make room)
    desired_speed = 12.0  # [m/s] free-road speed
    ring_speed = 8.0  # [m/s] speed on the roundabout's curved lanes

    def act(self, action: dict | str = None):
        if self.crashed:
            return
        entry, distance = self._distance_to_yield_line()
        self.target_speed = self._situational_speed()
        super().act(action)
        yield_acc = self._yield_acceleration(entry, distance) if entry else None
        self.entering = entry is not None and distance <= self.LOOKAHEAD and yield_acc is None
        if entry is None:
            self.committed = False
        zipper_acc = (self._follow_ring_acceleration(entry, distance) if self.entering
                      else self._make_room_acceleration())
        for acc in (yield_acc, zipper_acc):
            if acc is not None:
                self.action["acceleration"] = float(
                    np.clip(min(self.action["acceleration"], acc), -self.ACC_MAX, self.ACC_MAX)
                )

    def step(self, dt: float):
        super().step(dt)
        # IDM braking can overshoot below zero; reversing destabilises the
        # steering controller and sends cars off their lane. Drivers stop.
        if not self.crashed and self.speed < 0:
            self.speed = 0.0

    def _situational_speed(self) -> float:
        segment = self.lane_index[:2]
        if on_ring(self) or segment in MERGE_SEGMENTS or segment in EXIT_CURVES:
            return self.ring_speed
        if segment in APPROACH_SEGMENTS:  # straight approach: brake gently ahead of the curve
            lane = self.road.network.get_lane(self.lane_index)
            to_curve = max(lane.length - lane.local_coordinates(self.position)[0], 0.0)
            return min(self.desired_speed,
                       float(np.sqrt(self.ring_speed**2 + 2 * self.APPROACH_DECEL * to_curve)))
        return self.desired_speed

    # ----------------------------------------------------- entering the ring
    def _yield_acceleration(self, entry: str, distance: float) -> float | None:
        if self.committed or distance > self.LOOKAHEAD:
            return None
        if not self._conflict_at(entry):
            return None
        gap = max(distance - self.STOP_OFFSET, 0.1)
        braking_distance = self.speed**2 / (2 * -self.COMFORT_ACC_MIN)
        # Too late to stop comfortably, or already over the line (stopping
        # here would block the ring car we are yielding to): go.
        if gap < braking_distance or distance <= self.STOP_OFFSET:
            self.committed = True
            return None
        return self._idm_to_obstacle(gap, leader_speed=0.0, standstill_gap=1.0)

    def _distance_to_yield_line(self) -> tuple[str | None, float]:
        """Return (entry node, distance along road to its merge point), if approaching one."""
        segment = self.lane_index[:2]
        for entry, (straight, merge) in APPROACH.items():
            if segment not in (straight, merge):
                continue
            net = self.road.network
            lane = net.get_lane(self.lane_index)
            remaining = lane.length - lane.local_coordinates(self.position)[0]
            if segment == straight:
                remaining += net.get_lane((*merge, 0)).length
            return entry, remaining
        return None, 0.0

    def _conflict_at(self, entry: str) -> bool:
        exit_node = UPSTREAM_EXIT[entry]
        for other in self.road.vehicles:
            if other is self or not on_ring(other):
                continue
            if any(r[0] == exit_node and r[1] == exit_node + "s" for r in (other.route or [])):
                continue  # leaving before it reaches us
            arc, _ = arc_upstream_of(other.position, entry)
            if -self.MERGE_ZONE < arc < 0:
                return True
            if 0 <= arc < self.MIN_GAP_DISTANCE:
                return True
            if arc >= 0 and arc / max(other.speed, 0.1) < self.CRITICAL_GAP:
                return True
        return False

    def _follow_ring_acceleration(self, entry: str, distance: float) -> float | None:
        """IDM response to ring cars that are virtually ahead of this entering car."""
        exit_node = UPSTREAM_EXIT[entry]
        acc = None
        for other in self.road.vehicles:
            if other is self or not on_ring(other):
                continue
            if any(r[0] == exit_node and r[1] == exit_node + "s" for r in (other.route or [])):
                continue
            arc, _ = arc_upstream_of(other.position, entry)
            # Past the merge point, ordinary car-following takes over once we
            # are on the ring; only cars still upstream need projecting.
            ahead = distance - arc
            if arc < 0 or ahead <= 0:
                continue
            gap = max(ahead - self.LENGTH, 0.1)
            a = self._idm_to_obstacle(gap, other.speed, self.MIN_STANDSTILL_GAP)
            acc = a if acc is None else min(acc, a)
        return acc

    # ---------------------------------------------------- circulating cars
    def _make_room_acceleration(self) -> float | None:
        """IDM response to entering cars, projected onto the ring as virtual leaders."""
        if not on_ring(self):
            return None
        # Only the car nearest the merge point on each approach is projected:
        # cars queued behind it are not about to enter, and projecting them
        # would stop the ring for phantom leaders (and can gridlock it).
        first: dict[str, tuple[float, YieldingIDMVehicle]] = {}
        for other in self.road.vehicles:
            if other is self or not isinstance(other, YieldingIDMVehicle):
                continue
            entry, remaining = other._distance_to_yield_line()
            if entry is not None and (entry not in first or remaining < first[entry][0]):
                first[entry] = (remaining, other)
        acc = None
        for entry, (remaining, other) in first.items():
            if not other.entering:
                continue
            ahead, _ = arc_upstream_of(self.position, entry)
            # Centre-to-centre distance to the virtual leader. Negative means
            # the entering car is virtually behind us: we go first.
            distance = ahead - remaining
            if ahead <= 0 or distance <= 0:
                continue
            gap = max(distance - self.LENGTH, 0.1)
            a = self._idm_to_obstacle(gap, other.speed, self.MIN_STANDSTILL_GAP)
            acc = a if acc is None else min(acc, a)
        return acc

    def _idm_to_obstacle(self, gap: float, leader_speed: float, standstill_gap: float) -> float:
        """IDM acceleration towards a leader ``gap`` metres (bumper to bumper) ahead."""
        # DELTA is randomised (non-integer), so keep the speed non-negative.
        speed = max(self.speed, 0.0)
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        s_star = (standstill_gap + speed * self.TIME_WANTED
                  + speed * (speed - leader_speed) / (2 * np.sqrt(ab)))
        v0 = max(self.target_speed, 0.1)
        return self.COMFORT_ACC_MAX * (1 - (speed / v0) ** self.DELTA - (max(s_star, 0) / gap) ** 2)
