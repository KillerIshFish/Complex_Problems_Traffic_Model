"""Heavy vehicles (trucks), driver heterogeneity (each driver's own reaction time and time gap) and
road conditions: invariants with vehicles of mixed lengths, capacity, the dynamics factors,
validation, live changes, metrics, and their viewer and experiments integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pygame
import pytest

from roundabout_sim import app as viewer
from roundabout_sim import experiments
from roundabout_sim.config import PARAM_BY_NAME, SimConfig, coerce
from roundabout_sim.engine import (HEADWAY_FLOOR, MAX_DECEL, MAX_REACTION, PERTURB_DECEL, ROAD_CONDITIONS,
                                   ROAD_SCALED, SAFE_MARGIN, TRUCK_ACCEL_FACTOR, TRUCK_HEADWAY_FACTOR,
                                   TRUCK_SPEED_FACTOR, TRUCK_WIDTH, TWO_PI, RingSim, effective)
from roundabout_sim.metrics import (Recorder, jam_heads, jammed, sector_occupancy, snapshot,
                                    spacing_gini)
from tests.helpers import assert_valid, bumper_gaps, click, frames, rendered_texts, widget

NEW_PARAMS = ("truck_share", "truck_length", "reaction_std", "headway_std", "road_condition")
TRUCKS = dict(truck_share=0.2)
STILL = dict(n_lanes=1, n_arms=0, speed_std=0.0, driver_noise=0.0)  # one lane, no exits, no randomness


# ------------------------------------------------------------------- config
def test_new_parameters_are_documented_and_validated():
    cfg = SimConfig()
    assert (cfg.truck_share, cfg.truck_length, cfg.reaction_std, cfg.headway_std, cfg.road_condition) == (
        0.0, 12.0, 0.0, 0.0, "dry")
    groups = {name: PARAM_BY_NAME[name].group for name in NEW_PARAMS}
    assert groups == {"truck_share": "Vehicles", "truck_length": "Vehicles", "reaction_std": "Drivers",
                      "headway_std": "Drivers", "road_condition": "Road"}
    for name in NEW_PARAMS:
        spec = PARAM_BY_NAME[name]
        assert spec.help and spec.label and spec.applies_in is None
        assert spec.reset == (name in ("truck_share", "truck_length"))  # a vehicle's length is set at its start
    assert PARAM_BY_NAME["road_condition"].choices == tuple(ROAD_CONDITIONS) == ("dry", "wet", "snow", "ice")
    assert PARAM_BY_NAME["truck_share"].hi == 0.3 and PARAM_BY_NAME["truck_length"].lo <= 9.0
    assert coerce("road_condition", "ice") == "ice" and coerce("truck_share", "0.15") == 0.15
    with pytest.raises(ValueError):
        coerce("road_condition", "slush")


@pytest.mark.parametrize("bad", [dict(truck_share=-0.1), dict(truck_share=1.5), dict(truck_share=float("nan")),
                                 dict(truck_length=0.0), dict(truck_length=-4.0), dict(reaction_std=-0.1),
                                 dict(headway_std=-1.0), dict(headway_std=float("inf")),
                                 dict(road_condition="slush")])
def test_invalid_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        SimConfig(**bad)
    sim = RingSim(seed=0)
    before = sim.cfg
    with pytest.raises(ValueError):
        sim.set(**bad)
    assert sim.cfg == before
    sim.step(5)


def test_help_texts_quote_the_engine_factors():
    road = PARAM_BY_NAME["road_condition"].help
    for condition in ("wet", "snow", "ice"):
        assert f"{MAX_DECEL * ROAD_CONDITIONS[condition]['max_decel']:.3g}" in road
    assert f"{MAX_DECEL:g} m/s2" in road
    trucks = PARAM_BY_NAME["truck_share"].help
    assert f"{TRUCK_SPEED_FACTOR:g} x" in trucks and f"{TRUCK_HEADWAY_FACTOR:g} x" in trucks
    assert f"{TRUCK_WIDTH:g} m wide" in trucks and TRUCK_ACCEL_FACTOR == 0.5 and "half" in trucks
    assert f"{HEADWAY_FLOOR:g}" == "0.5" and "half" in PARAM_BY_NAME["headway_std"].help


# ----------------------------------------------------------- road conditions
def test_dry_is_exactly_the_settings():
    assert ROAD_CONDITIONS["dry"] == dict.fromkeys(ROAD_SCALED, 1.0)
    cfg = SimConfig(mean_speed=11.3, max_accel=1.7, comfort_decel=2.3, time_headway=0.9, entry_gap=2.6,
                    follow_up=2.7, lateral_accel=3.1)
    eff = effective(cfg)
    for name in ROAD_SCALED:
        expected = MAX_DECEL if name == "max_decel" else getattr(cfg, name)
        assert getattr(eff, name) == expected  # bit for bit: x * 1.0
    sim = RingSim(seed=1)
    assert sim.alike and sim.uniform_length
    np.testing.assert_array_equal(sim.headway[sim.alive], sim.cfg.time_headway)
    np.testing.assert_array_equal(sim.accel[sim.alive], sim.cfg.max_accel)
    assert (sim.delay[sim.alive] == 4).all() and not sim.truck.any()


def test_road_conditions_get_worse_from_dry_to_ice():
    order = ("dry", "wet", "snow", "ice")
    for name in ("mean_speed", "max_accel", "comfort_decel", "lateral_accel", "max_decel"):
        values = [ROAD_CONDITIONS[c][name] for c in order]
        assert values == sorted(values, reverse=True) and values[-1] < 1.0, name
    for name in ("time_headway", "follow_up"):
        values = [ROAD_CONDITIONS[c][name] for c in order]
        assert values == sorted(values) and values[-1] > 1.0, name
    assert all(ROAD_CONDITIONS[c]["entry_gap"] > 1.0 for c in order[1:])
    for condition in order:  # drivers can always brake comfortably within the road's limit
        eff = effective(SimConfig(road_condition=condition))
        assert eff.comfort_decel < eff.max_decel
    assert effective(SimConfig(road_condition="ice")).max_decel == pytest.approx(1.35)


@pytest.mark.parametrize("condition", ["wet", "snow", "ice"])
def test_road_condition_scales_the_dynamics(condition):
    k = ROAD_CONDITIONS[condition]
    sim = RingSim(seed=2, road_condition=condition, follow_up=2.0, **STILL)
    cfg, eff = sim.cfg, sim.eff
    assert eff.mean_speed == pytest.approx(k["mean_speed"] * cfg.mean_speed)
    live = np.flatnonzero(sim.alive)
    np.testing.assert_allclose(sim.headway[live], k["time_headway"] * cfg.time_headway)
    np.testing.assert_allclose(sim.accel[live], k["max_accel"] * cfg.max_accel)
    cap = np.sqrt(k["lateral_accel"] * cfg.lateral_accel * sim.radii[0])
    np.testing.assert_allclose(sim.desired_speed(live), min(eff.mean_speed, cap))
    assert sim.follow_up_steps() == int(np.floor(2.0 * k["follow_up"] / cfg.dt + 1e-9)) + 1
    # A car told to brake brakes no harder than the road allows.
    slot = int(live[np.argmax(sim.v[live])])
    sim.brake(slot, duration=1.0)
    v0 = sim.v[slot]
    sim.step()
    assert v0 - sim.v[slot] == pytest.approx(min(PERTURB_DECEL, eff.max_decel) * cfg.dt, abs=1e-9)


def entry_accepted(condition: str, back_time: float, v_behind: float = 3.0) -> bool:
    """Whether a car may enter one lane where the only vehicle passes the entrance ``back_time`` s
    (at ``v_behind``) after it would."""
    sim = RingSim(seed=0, n_lanes=1, n_arms=1, density=1.0, road_condition=condition, speed_std=0.0)
    sim.alive[:] = False
    angle = sim.entry_angle[0]
    behind = angle - (back_time * v_behind + sim.cfg.car_length) / sim.radii[0]
    slot = sim._spawn(np.array([0]), np.array([behind]), np.array([v_behind]))
    sim._lanes[0] = (slot, sim.theta[slot])  # the outer lane's index, as _entries keeps it
    ok, _ = sim._entry_ok(np.array([angle]))
    return bool(ok[0])


@pytest.mark.parametrize("condition", ["wet", "snow", "ice"])
def test_entries_need_the_longer_entry_gap_of_the_road(condition):
    """A gap between the dry and the slippery critical gap lets a car in on a dry road only."""
    dry, slippery = SimConfig().entry_gap, effective(SimConfig(road_condition=condition)).entry_gap
    assert slippery == pytest.approx(ROAD_CONDITIONS[condition]["entry_gap"] * dry) and slippery > dry
    between = 0.5 * (dry + slippery)
    assert entry_accepted("dry", between) and not entry_accepted(condition, between)
    assert entry_accepted(condition, 1.02 * slippery)


def test_comfortable_braking_follows_the_road():
    """Car following, gentle braking above the desired speed and slowing for an exit all use the
    road's comfortable braking (snow: 0.7 x the setting)."""
    sim = RingSim(seed=0, road_condition="snow", **STILL)
    cfg, eff = sim.cfg, sim.eff
    b = eff.comfort_decel
    assert b == pytest.approx(ROAD_CONDITIONS["snow"]["comfort_decel"] * cfg.comfort_decel) and b < cfg.comfort_decel
    a, T = eff.max_accel, eff.time_headway
    v, gap, v_lead = np.array([10.0]), np.array([20.0]), np.array([5.0])
    s_star = cfg.min_gap + v * T + v * (v - v_lead) / (2 * np.sqrt(a * b))
    np.testing.assert_allclose(sim._interaction(v, gap, v_lead, a, T), a * (s_star / gap) ** 2)
    v, v0 = np.array([12.0]), np.array([8.0])
    np.testing.assert_allclose(sim._free_acc(v, v0, a), -b * (1 - (v0 / v) ** (4 * a / b)))
    slot = np.flatnonzero(sim.alive)[:1]
    sim.exit_at[slot] = sim.travel[slot] + 12.0 / sim.r_out  # 12 m before its exit, on the outer lane
    approach = np.sqrt(cfg.turn_speed ** 2 + 2 * b * (12.0 - 5.0))
    assert approach < sim.desired_speed(slot)[0]  # (so the approach decides)
    np.testing.assert_allclose(sim.target_speed(slot), approach)


def test_harsh_braking_counts_against_the_road_braking_limit(monkeypatch):
    """Asking for 2 m/s2 of braking is harsh on ice (limit 1.35 m/s2), not on a dry road (9)."""
    episodes = {}
    for condition in ("dry", "ice"):
        sim = RingSim(seed=0, road_condition=condition, **STILL)
        sim.v[sim.alive] = 5.0
        monkeypatch.setattr(sim, "_car_following", lambda idx, *a: np.full(idx.size, -2.0))
        sim.step(3)
        episodes[condition] = sim.n_emergency
    assert episodes == {"dry": 0, "ice": int(sim.alive.sum())}  # one episode per driver on ice


@pytest.mark.parametrize("braking, changes", [(8.5, True), (10.5, False)])
def test_a_forced_lane_change_never_asks_more_than_the_braking_limit(braking, changes):
    """A leaver that must get out may make the driver behind it brake at up to 2.5 x Comfortable
    braking, but never harder than the road allows: at 5 m/s2 the 9 m/s2 dry limit decides."""
    sim = RingSim(seed=0, n_lanes=2, density=1.0, comfort_decel=5.0, speed_std=0.0, driver_noise=0.0)
    sim.alive[:] = False
    leaver = sim._spawn(np.array([0]), np.array([1.0]), np.array([2.0]))
    sim.exit_at[leaver] = sim.travel[leaver] + 40.0 / sim.r_out  # 40 m to go: it must move out now
    sim.lc_cool[leaver] = 0.0
    assert sim.must_move_out(leaver).all()
    behind = sim._spawn(np.array([1]), np.array([1.0]), np.array([10.0]))
    sim.exit_at[behind], sim.lc_cool[behind] = np.inf, 10.0
    v0 = sim.target_speed(behind)
    lo, hi = sim.cfg.min_gap, sim.cfg.reaction_distance  # the gap behind at which it would brake this hard
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if sim._idm(np.array([10.0]), v0, np.array([mid]), np.array([2.0]), behind)[0] < -braking:
            lo = mid
        else:
            hi = mid
    sim.theta[behind] = (1.0 - (hi + sim.cfg.car_length) / sim.radii[1]) % TWO_PI
    sim._lane_changes(np.flatnonzero(sim.alive))
    assert bool(sim.lane[leaver][0] == 1) == changes


def test_lone_car_reaches_the_free_speed_of_the_road():
    for condition in ("dry", "snow"):
        sim = RingSim(seed=0, radius=200.0, road_condition=condition, **STILL)
        sim.reset(density=1000.0 / sim.total_lane_length)  # exactly one car
        sim.v[sim.alive] = 0.0
        sim.step(900)
        assert sim.v[sim.alive][0] == pytest.approx(sim.cfg.mean_speed * ROAD_CONDITIONS[condition]["mean_speed"],
                                                    rel=0.02)


def test_road_condition_changes_live_and_back():
    sim = RingSim(seed=3, headway_std=0.3)
    sim.step(100)
    uids, gen = sim.cars()["uid"].copy(), sim.generation
    dry = sim.headway.copy()
    assert sim.set(road_condition="ice") is False
    assert sim.generation == gen and sim.eff == effective(sim.cfg) and sim.eff.max_decel == pytest.approx(1.35)
    np.testing.assert_array_equal(sim.cars()["uid"], uids)  # the same drivers carry on
    live = sim.alive
    np.testing.assert_allclose(sim.headway[live], dry[live] * ROAD_CONDITIONS["ice"]["time_headway"])
    sim.step(100)
    assert_valid(sim)
    sim.set(road_condition="dry")
    assert sim.eff == effective(sim.cfg)
    sim.step(50)
    assert_valid(sim)


def test_snow_and_ice_slow_the_traffic():
    speeds = {}
    for condition in ("dry", "wet", "snow", "ice"):
        sim = RingSim(seed=1, road_condition=condition, density=30.0, exit_prob=0.0)
        sim.step(600)
        speeds[condition] = snapshot(sim)["mean_speed"]
        assert_valid(sim)
    assert speeds["dry"] > speeds["wet"] > speeds["snow"] > speeds["ice"] > 0


# --------------------------------------------------------- separate streams
def test_vehicles_and_drivers_draw_from_their_own_stream():
    """Trucks and driver spreads draw only from rng_drivers: the main stream is where it would be."""
    base = RingSim(seed=4)
    for change in (dict(truck_share=0.2), dict(reaction_std=0.3, headway_std=0.4),
                   dict(truck_share=0.1, road_condition="wet")):
        other = RingSim(seed=4, **change)
        assert int(other.alive.sum()) == int(base.alive.sum())
        assert other.rng.bit_generator.state == base.rng.bit_generator.state
    # The same seed draws the same vehicles and temperaments, whatever the spreads.
    a, b = RingSim(seed=4, truck_share=0.2), RingSim(seed=4, truck_share=0.2, reaction_std=0.3, headway_std=0.4)
    np.testing.assert_array_equal(a.truck, b.truck)
    np.testing.assert_array_equal(a.z_gap, b.z_gap)
    np.testing.assert_array_equal(a.z_react, b.z_react)


def test_same_seed_same_mixed_run():
    a, b = (RingSim(seed=8, truck_share=0.2, reaction_std=0.2, headway_std=0.3, road_condition="wet",
                    traffic_mode="open", entry_rate=0.3) for _ in range(2))
    a.step(400)
    b.step(400)
    for name in ("theta", "v", "truck", "headway", "delay"):
        np.testing.assert_array_equal(getattr(a, name)[a.alive], getattr(b, name)[b.alive])
    assert [list(q) for q in a.queue_trucks] == [list(q) for q in b.queue_trucks]


def test_seed_change_reseeds_the_driver_stream():
    a, b = RingSim(seed=1), RingSim(seed=2)
    a.set(seed=9)
    b.set(seed=9)
    assert a.rng_drivers.random(5).tolist() == b.rng_drivers.random(5).tolist()


# ------------------------------------------------------------- no overlaps
OVERLAP_CASES = {
    "share-0.1": dict(truck_share=0.1),
    "share-0.3-long": dict(truck_share=0.3, truck_length=20.0),
    "small-radius": dict(truck_share=0.3, truck_length=20.0, radius=20.0, n_lanes=1, density=60.0),
    "small-3-lanes": dict(truck_share=0.25, truck_length=16.5, radius=20.0, exit_prob=0.4),
    "five-lanes": dict(truck_share=0.2, n_lanes=5, density=90.0, exit_prob=0.3, lane_change_rate=0.6),
    "random-start": dict(truck_share=0.3, init_spacing="random", density=80.0),
    "clustered-start": dict(truck_share=0.3, init_spacing="clustered", density=100.0, truck_length=18.75),
    "open-entries": dict(truck_share=0.3, traffic_mode="open", entry_rate=0.4, exit_prob=0.3, truck_length=16.5),
    "near-jam": dict(truck_share=0.2, n_lanes=2, density=1000.0, exit_prob=0.3),
    "drivers-ice": dict(truck_share=0.15, reaction_std=0.5, headway_std=0.8, road_condition="ice",
                        driver_noise=1.0),
}


@pytest.mark.parametrize("config", OVERLAP_CASES.values(), ids=OVERLAP_CASES.keys())
def test_no_overlaps_with_trucks(config):
    sim = RingSim(seed=11, **config)
    assert_valid(sim)
    for k in range(8):
        sim.step(40)
        assert_valid(sim)
        if k == 3:  # a live density raise: insertion and respacing
            sim.set(density=min(4 * sim.cfg.density, 1000.0))
            assert_valid(sim)
        if k == 5:
            sim.set(density=sim.cfg.density / 4)
            assert_valid(sim)
    assert sim.truck.any()  # (types stay in freed slots: some vehicle on the way was a truck)


def test_insertion_and_respacing_make_room_for_trucks():
    sim = RingSim(seed=5, truck_share=0.3, truck_length=20.0, n_lanes=2, n_arms=0, density=20.0)
    for count in (3, 30):  # a few into random gaps, then many into the largest
        before = int(sim.alive.sum())
        got, _ = sim._insert_into_gaps(count)
        assert int(sim.alive.sum()) == before + got and got > 0
        assert_valid(sim)
    assert sim._respace_lanes(10)
    assert_valid(sim)
    while sim._respace_lanes(20):  # fill the lanes to the brim
        assert_valid(sim)
    for lane in range(sim.n_lanes):  # full: lengths and standstill gaps add up to at most the lane
        in_lane = sim.alive & (sim.lane == lane)
        assert sim.length[in_lane].sum() + in_lane.sum() * sim.cfg.min_gap <= sim.lane_len[lane] + 1e-9
    sim.step(100)
    assert_valid(sim)


# ------------------------------------------------- geometry and population
def test_capacity_and_jam_density_use_the_mean_length():
    sim = RingSim(seed=0, truck_share=0.2, truck_length=12.0, car_length=4.0, min_gap=2.0)
    assert sim.mean_length == pytest.approx(5.6)
    assert sim.jam_density == pytest.approx(1000.0 / 7.6)
    assert sim.capacity() == int(np.floor(sim.lane_len / 7.6).sum())
    assert sim.target_cars() == round(sim.cfg.density * sim.total_lane_length / 1000)
    # Density still counts vehicles, trucks or not.
    row = snapshot(sim)
    assert row["density"] == pytest.approx(row["cars"] / sim.total_lane_length * 1000.0)
    no_trucks = RingSim(seed=0)
    assert no_trucks.mean_length == no_trucks.cfg.car_length and no_trucks.jam_density == 1000.0 / 6.0


def test_truck_share_lengths_and_widths():
    sim = RingSim(seed=1, truck_share=0.2, truck_length=16.5, n_lanes=5, radius=200.0, density=100.0)
    live = sim.alive
    share = sim.truck[live].mean()
    n = int(live.sum())
    assert share == pytest.approx(0.2, abs=4 * np.sqrt(0.2 * 0.8 / n))
    cars = sim.cars()
    np.testing.assert_array_equal(cars["length"], np.where(cars["truck"], 16.5, sim.cfg.car_length))
    np.testing.assert_array_equal(cars["width"], np.where(cars["truck"], TRUCK_WIDTH, sim.cfg.car_width))


@pytest.mark.parametrize("spacing", ["uniform", "random", "clustered"])
def test_initial_layout_keeps_the_pattern_with_trucks(spacing):
    sim = RingSim(seed=8, truck_share=0.3, n_lanes=3, n_arms=0, density=40.0, init_spacing=spacing)
    cfg = sim.cfg
    for lane in range(3):
        in_lane = sim.alive & (sim.lane == lane)
        slots = np.flatnonzero(in_lane)
        order = slots[np.argsort(sim.theta[slots])]
        th, length = sim.theta[order], sim.length[order]
        gaps = np.diff(np.append(th, th[0] + TWO_PI)) * sim.radii[lane] - 0.5 * (length + np.roll(length, -1))
        assert gaps.min() >= cfg.min_gap - 1e-6
        if spacing == "uniform":  # even bumper gaps however the lengths mix
            assert np.ptp(gaps) < 1e-6
        elif spacing == "clustered":  # one packed queue: all but the gap behind the last car are tight
            packed = gaps <= 1.05 * (cfg.car_length + cfg.min_gap) - cfg.car_length + 1e-6
            assert packed.sum() == gaps.size - 1


def test_a_truck_that_fits_no_gap_waits_for_one():
    """The density control keeps a truck that fits no gap waiting, first in line for the next fill,
    instead of respacing a whole lane: that is only for a ring where no gap takes even a car."""
    sim = RingSim(seed=0, n_lanes=1, n_arms=0, truck_share=0.01, truck_length=22.0, density=60.0)
    cfg = sim.cfg
    _, gaps = sim.bumper_gaps()
    fits = 2 * (cfg.car_length + max(0.5 * cfg.min_gap, SAFE_MARGIN + 0.2))  # what a car needs
    assert fits <= gaps.max() + cfg.car_length < fits + cfg.truck_length - cfg.car_length  # a car fits, a truck not
    n = int(sim.alive.sum())
    sim._pending = [True, False]
    assert sim._insert_into_gaps(2) == (1, False)  # the car behind the truck goes...
    assert sim._pending == [True] and int(sim.alive.sum()) == n + 1  # ...the truck waits, first in line
    respaced = []
    respace = sim._respace_lanes
    sim._respace_lanes = lambda k: respaced.append(k) or respace(k)
    assert sim.set(density=(n + 2) / sim.total_lane_length * 1000.0) is False  # one vehicle short
    sim._sync_population()
    assert respaced == [] and sim._pending == [True] and int(sim.alive.sum()) == n + 1
    assert sim._next_fill == pytest.approx(sim.time + 1.0)  # tried again in a second
    sim.set(density=1000.0)  # cars fill the gaps until none takes one: then the lanes are respaced
    sim.step(15)  # (once that second has passed)
    assert respaced and int(sim.alive.sum()) > n + 1
    assert_valid(sim)


def test_dense_traffic_with_long_trucks_keeps_its_lanes():
    """At 0.8 x jam density a truck slotted in for a leaver used to make the density control respace a
    whole lane whenever it fitted no gap; now it waits a few seconds for one."""
    sim = RingSim(seed=1, truck_share=0.3, truck_length=22.0)
    sim.reset(density=0.8 * sim.jam_density)
    respaced, waited = [], 0
    respace = sim._respace_lanes
    sim._respace_lanes = lambda k: respaced.append(k) or respace(k)
    for _ in range(600):
        sim.step()
        waited += any(sim._pending)
    assert respaced == [] and waited > 0
    assert sim.truck[sim.alive].mean() == pytest.approx(0.3, abs=0.1)  # no trucks lost on the way
    assert_valid(sim)


def stopped_lane(kinds, gaps_ahead, circumference=94.0):
    """One lane of stopped vehicles (True: a 22 m truck) with these bumper gaps ahead of each but the
    last (whose gap is what is left of the lane)."""
    sim = RingSim(seed=1, n_lanes=1, n_arms=0, density=1.0, radius=circumference / TWO_PI, truck_share=0.3,
                  truck_length=22.0)
    sim.alive[:] = False
    length = np.where(kinds, 22.0, sim.cfg.car_length)
    s = np.concatenate(([0.0], np.cumsum(np.array(gaps_ahead) + 0.5 * (length[:-1] + length[1:]))))
    slots = sim._spawn(np.zeros(len(kinds), dtype=int), s / sim.radii[0], np.zeros(len(kinds)),
                       trucks=np.array(kinds))
    return sim, slots


def test_the_most_bumper_room_heads_a_closed_jam_and_takes_the_new_car():
    """Room is measured bumper to bumper: a truck 25 m behind another truck has only 3 m of it."""
    sim, slots = stopped_lane([False, False, True, True, True], [3.0, 8.0, 3.0, 3.0])
    idx, gap = sim.bumper_gaps()
    roomiest = idx[np.argmax(gap)]
    assert roomiest == slots[1] and gap.max() == pytest.approx(8.0)
    slow = jammed(sim, idx)
    assert slow.all()
    np.testing.assert_array_equal(jam_heads(sim, idx, slow), [roomiest])
    # Respacing puts a new car into the roomiest gap, so the others move the least.
    sim._pending = [False]
    before = sim.theta[slots].copy()
    assert sim._respace_lanes(1)
    new = np.setdiff1d(np.flatnonzero(sim.alive), slots)
    lead, _ = sim._index_lanes(np.flatnonzero(sim.alive))
    assert lead[slots[1]] == new[0] and lead[new[0]] == slots[2]
    moved = np.abs((sim.theta[slots] - before + np.pi) % TWO_PI - np.pi) * sim.radii[0]
    assert moved.max() < 1.5  # (into the 3 m gap between two trucks: 6 m)
    assert_valid(sim)


def test_a_lane_full_of_trucks_by_chance_drops_the_last():
    sim = RingSim(seed=0, truck_share=0.3, truck_length=23.0, n_lanes=1, n_arms=0, radius=20.0,
                  density=1000.0, min_gap=0.5)
    assert_valid(sim)
    assert int(sim.alive.sum()) <= sim.target_cars()


# ---------------------------------------------------------------- dynamics
def test_trucks_accelerate_drive_and_follow_like_trucks():
    sim = RingSim(seed=3, truck_share=0.5, **STILL)
    live = np.flatnonzero(sim.alive)
    truck, car = live[sim.truck[live]], live[~sim.truck[live]]
    assert truck.size and car.size
    cfg = sim.cfg
    np.testing.assert_allclose(sim.accel[truck], TRUCK_ACCEL_FACTOR * cfg.max_accel)
    np.testing.assert_allclose(sim.accel[car], cfg.max_accel)
    np.testing.assert_allclose(sim.headway[truck], TRUCK_HEADWAY_FACTOR * cfg.time_headway)
    np.testing.assert_allclose(sim.headway[car], cfg.time_headway)
    np.testing.assert_allclose(sim.desired_speed(truck), TRUCK_SPEED_FACTOR * cfg.mean_speed)
    np.testing.assert_allclose(sim.desired_speed(car), cfg.mean_speed)
    # A truck needs a longer gap than a car for the same speed.
    v_car, v_truck = sim.equilibrium_speed(np.full(2, 20.0), cfg.mean_speed, np.array([car[0], truck[0]]))
    assert v_truck < v_car
    # From a standstill on a free road a truck pulls away at half a car's acceleration.
    a = sim._idm(np.zeros(2), np.full(2, 10.0), np.full(2, np.inf), np.zeros(2), np.array([car[0], truck[0]]))
    np.testing.assert_allclose(a, [cfg.max_accel, TRUCK_ACCEL_FACTOR * cfg.max_accel])


def test_lone_truck_reaches_its_lower_free_speed():
    sim = RingSim(seed=0, radius=200.0, truck_share=1.0, reaction_time=0.0, **STILL)
    sim.reset(density=1000.0 / sim.total_lane_length)  # exactly one vehicle: a truck
    slot = np.flatnonzero(sim.alive)
    assert sim.truck[slot].all()
    sim.v[slot] = 0.0
    sim.step(10)
    assert sim.v[slot][0] == pytest.approx(TRUCK_ACCEL_FACTOR * sim.cfg.max_accel * 1.0, rel=0.05)
    sim.step(1500)
    assert sim.v[slot][0] == pytest.approx(TRUCK_SPEED_FACTOR * sim.cfg.mean_speed, rel=0.02)


def test_a_driver_with_a_longer_time_gap_brakes_earlier():
    sim = RingSim(seed=0, headway_std=0.5, **STILL)
    live = np.flatnonzero(sim.alive)
    calm, keen = live[np.argmax(sim.headway[live])], live[np.argmin(sim.headway[live])]
    assert sim.headway[calm] > sim.headway[keen] + 0.5
    same = np.full(2, 8.0)  # same speed, desired speed, gap and leader
    acc = sim._idm(same, np.full(2, 12.0), np.full(2, 15.0), same, np.array([calm, keen]))
    assert acc[0] < acc[1]
    v = sim.equilibrium_speed(np.full(2, 15.0), 12.0, np.array([calm, keen]))
    assert v[0] < v[1]


def test_drivers_with_a_longer_time_gap_settle_further_back():
    sim = RingSim(seed=2, density=40.0, reaction_time=0.0, reaction_distance=100.0, headway_std=0.5, **STILL)
    sim.step(2500)
    idx, gaps = sim.bumper_gaps()
    T = sim.headway[idx]
    assert np.ptp(T) > 0.5 and np.ptp(sim.v[idx]) < 0.1  # settled: one speed, each driver their own gap
    ranks = lambda a: np.argsort(np.argsort(a))  # noqa: E731
    assert np.corrcoef(ranks(T), ranks(gaps))[0, 1] > 0.9


def test_each_driver_reacts_after_their_own_delay(monkeypatch):
    sim = RingSim(seed=6, radius=150.0, density=15.0, reaction_time=0.6, reaction_std=0.4, **STILL)
    live = np.flatnonzero(sim.alive)
    assert np.unique(sim.delay[live]).size >= 3  # drivers really differ
    sim.v[live] = 5.0
    start = 20  # every driver decides to brake at this step
    monkeypatch.setattr(sim, "_car_following", lambda idx, *a: np.where(sim.steps >= start, -1.0, 0.0)
                        * np.ones(idx.size))
    speeds = []
    for _ in range(start + 25):
        sim.step()
        speeds.append(sim.v[live].copy())
    first_drop = np.argmax(np.array(speeds) < 5.0 - 1e-9, axis=0)
    np.testing.assert_array_equal(first_drop, start + sim.delay[live])


def test_per_driver_values_stay_in_their_ranges():
    sim = RingSim(seed=1, n_lanes=5, radius=200.0, density=100.0, reaction_std=5.0, headway_std=5.0)
    live = sim.alive
    assert sim.delay[live].min() == 0 and sim.delay[live].max() == round(MAX_REACTION / sim.cfg.dt)
    assert sim.headway[live].min() == pytest.approx(HEADWAY_FLOOR * sim.cfg.time_headway)
    sim.step(50)
    assert_valid(sim)


def test_spreads_match_the_settings():
    sim = RingSim(seed=1, n_lanes=5, radius=300.0, car_length=2.0, min_gap=1.0, density=200.0,
                  reaction_std=0.2, headway_std=0.3)
    live = sim.alive
    assert live.sum() > 1800
    reaction = sim.delay[live] * sim.cfg.dt
    assert reaction.mean() == pytest.approx(sim.cfg.reaction_time, abs=0.02)
    assert reaction.std() == pytest.approx(0.2, rel=0.15)  # (rounded to the 0.1 s step)
    assert sim.headway[live].mean() == pytest.approx(sim.cfg.time_headway, abs=0.03)
    assert sim.headway[live].std() == pytest.approx(0.3, rel=0.1)  # (floored at half the setting)


def test_spreads_and_settings_change_every_driver_live():
    sim = RingSim(seed=3, truck_share=0.2)
    sim.step(50)
    gen = sim.generation
    for change in (dict(headway_std=0.4), dict(reaction_std=0.3), dict(time_headway=1.6), dict(reaction_time=0.8),
                   dict(max_accel=2.0), dict(headway_std=0.0, reaction_std=0.0)):
        live, uids = sim.alive.copy(), sim.uid.copy()
        z_gap, z_react = sim.z_gap.copy(), sim.z_react.copy()
        assert sim.set(**change) is False
        np.testing.assert_array_equal(sim.uid[live], uids[live])  # the same drivers...
        np.testing.assert_array_equal(sim.z_gap[live], z_gap[live])  # ...keep their temperaments
        cfg = sim.cfg
        truck = sim.truck[live]
        own = np.maximum(HEADWAY_FLOOR * cfg.time_headway, cfg.time_headway + z_gap[live] * cfg.headway_std)
        np.testing.assert_allclose(sim.headway[live], np.where(truck, TRUCK_HEADWAY_FACTOR, 1.0) * own)
        reaction = np.clip(cfg.reaction_time + z_react[live] * cfg.reaction_std, 0, MAX_REACTION)
        np.testing.assert_array_equal(sim.delay[live], np.rint(reaction / cfg.dt))
        np.testing.assert_allclose(sim.accel[live], np.where(truck, TRUCK_ACCEL_FACTOR, 1.0) * cfg.max_accel)
        sim.step(20)
        assert_valid(sim)
    assert sim.generation == gen and not sim.alike  # (trucks still differ)
    plain = RingSim(seed=3, headway_std=0.4)
    assert not plain.alike
    plain.set(headway_std=0.0)
    assert plain.alike
    np.testing.assert_array_equal(plain.headway[plain.alive], plain.cfg.time_headway)


def test_truck_share_and_length_rebuild():
    sim = RingSim(seed=3)
    sim.step(20)
    # With no trucks, before and after, Truck length is only kept for later: the run goes on.
    assert sim.set(truck_length=15.0) is False and sim.time == pytest.approx(2.0)
    assert sim.cfg.truck_length == 15.0 and sim.uniform_length and sim.mean_length == sim.cfg.car_length
    assert sim.set(truck_share=0.2) is True and sim.time == 0.0
    np.testing.assert_array_equal(sim.length[sim.alive & sim.truck], 15.0)
    assert sim.truck[sim.alive].any() and not sim.uniform_length
    assert sim.set(truck_length=18.0) is True
    np.testing.assert_array_equal(sim.length[sim.alive & sim.truck], 18.0)
    assert sim.set(truck_share=0.0) is True and not sim.truck[sim.alive].any() and sim.uniform_length


# ------------------------------------------------------------- the queues
def test_open_mode_queues_carry_trucks_in_order():
    sim = RingSim(seed=7, truck_share=0.3, traffic_mode="open", entry_rate=0.4, exit_prob=0.3)
    entered_trucks = []
    spawn = sim._spawn

    def recording_spawn(*args, trucks=None, **kw):
        slots = spawn(*args, trucks=trucks, **kw)
        if kw.get("lat") is not None:  # an entry
            entered_trucks.append(bool(sim.truck[slots[0]]))
        return slots

    sim._spawn = recording_spawn
    for _ in range(1500):
        sim.step()
        assert [len(q) for q in sim.queue_trucks] == sim.queue.tolist()
    n = len(entered_trucks)
    assert n == sim.n_entered > 50
    assert np.mean(entered_trucks) == pytest.approx(0.3, abs=4 * np.sqrt(0.21 / n))
    assert sim.n_arrived == sim.n_entered + int(sim.queue.sum()) + sim.n_blocked + sim.n_gave_up
    sim.set(max_queue=2)  # cut queues: the types of the latest arrivals go with them
    assert [len(q) for q in sim.queue_trucks] == sim.queue.tolist()
    assert_valid(sim)


def test_closed_mode_replacements_get_a_type():
    sim = RingSim(seed=4, truck_share=0.3, exit_prob=0.5, density=70.0)
    for _ in range(30):
        sim.step(20)
        assert_valid(sim)
    assert sim.n_entered > 10 and sim.truck[sim.alive].any()


def test_a_queued_truck_waits_for_a_gap_it_fits():
    sim = RingSim(seed=0, n_lanes=1, truck_share=0.5, truck_length=20.0)
    order = np.flatnonzero(sim.alive & (sim.lane == sim.outer))
    sim._lanes[sim.outer] = (order[np.argsort(sim.theta[order])], np.sort(sim.theta[order]))
    angle = np.array([sim.entry_angle[0]])
    ok_car, _ = sim._entry_ok(angle, np.array([False]))
    ok_truck, _ = sim._entry_ok(angle, np.array([True]))
    assert not (ok_truck & ~ok_car).any()  # wherever a truck may go, a car may too
    _, front_car, _, back_car = sim._neighbours_in(sim.outer, angle, sim.cfg.car_length)
    _, front_truck, _, back_truck = sim._neighbours_in(sim.outer, angle, 20.0)
    assert front_truck == pytest.approx(front_car - 8.0) and back_truck == pytest.approx(back_car - 8.0)


# ------------------------------------------------------------------ metrics
def test_metrics_use_the_real_lengths_and_widths():
    sim = RingSim(seed=2, truck_share=0.3, n_lanes=2, density=60.0)
    sim.step(100)
    cfg = sim.cfg
    idx = np.flatnonzero(sim.alive)
    row = snapshot(sim)
    width = np.where(sim.truck[idx], TRUCK_WIDTH, cfg.car_width)
    area = np.sum(sim.length[idx] * width)
    assert row["occupancy"] == pytest.approx(area / (sim.total_lane_length * cfg.lane_width))
    assert sector_occupancy(sim, sectors=12).mean() == pytest.approx(sim.length[idx].sum() / sim.total_lane_length)
    slots, gaps = sim.bumper_gaps()
    road = np.sort(gaps + sim.mean_length)  # one length for all: the mix of lengths is no congregation
    n = road.size
    gini = 2 * np.sum(np.arange(1, n + 1) * road) / (n * road.sum()) - (n + 1) / n
    assert spacing_gini(sim) == pytest.approx(gini)
    np.testing.assert_allclose(np.sort(gaps), np.sort(bumper_gaps(sim)))


@pytest.mark.parametrize("density", [50.0, 1000.0])
def test_spacing_gini_reads_even_bumper_gaps_as_even_with_trucks(density):
    """An even start, or a jam bumper to bumper, scores about 0 whatever the mix of lengths (it used
    to read a jam with long trucks as random gaps, ~0.3)."""
    mixed = RingSim(seed=1, n_arms=0, density=density, truck_share=0.3, truck_length=22.0)
    assert mixed.truck[mixed.alive].any()
    assert spacing_gini(mixed) < 0.1
    assert spacing_gini(RingSim(seed=1, n_arms=0, density=density)) < 0.02


def test_truck_counts_and_speeds_in_the_rows():
    sim = RingSim(seed=5, truck_share=0.25, density=15.0, exit_prob=0.0)  # free flow: everyone at their own speed
    rec = Recorder(sim, interval=0.5)
    for _ in range(300):
        sim.step()
        rec.update()
    row = rec.rows[-1]
    live = sim.alive
    assert row["trucks"] == int(sim.truck[live].sum()) > 0
    assert row["truck_speed"] == pytest.approx(sim.v[live & sim.truck].mean())
    assert row["car_speed"] == pytest.approx(sim.v[live & ~sim.truck].mean())
    assert row["truck_speed"] < row["car_speed"]  # trucks want 0.7 x
    assert all(isinstance(v, (int, float, np.integer, np.floating)) for v in row.values())  # numeric rows
    plain = snapshot(RingSim(seed=5))
    assert plain["trucks"] == 0 and np.isnan(plain["truck_speed"]) and plain["car_speed"] == plain["mean_speed"]
    empty = snapshot(RingSim(seed=5, density=0.0))
    assert empty["trucks"] == 0 and np.isnan(empty["truck_speed"]) and np.isnan(empty["car_speed"])


def test_clicking_anywhere_on_a_truck_finds_it():
    sim = RingSim(seed=1, truck_share=1.0, truck_length=20.0, density=5.0, n_lanes=1)
    slot = int(np.flatnonzero(sim.alive)[0])
    th = sim.theta[slot]
    x, y = sim.positions(np.array([slot]))
    along = 9.0  # metres towards its front: outside a car's 4 m reach, inside the truck
    px, py = x[0] - along * np.sin(th), y[0] + along * np.cos(th)
    assert sim.car_near(px, py) == slot
    assert sim.car_near(x[0] - 14.0 * np.sin(th), y[0] + 14.0 * np.cos(th), max_dist=1.0) is None


# ------------------------------------------------------------------- viewer
@pytest.mark.parametrize("size", [(1100, 700), (1920, 1080)])
def test_viewer_draws_trucks_and_colours_by_type(make_app, size, monkeypatch):
    app = make_app(size=size, truck_share=0.2, road_condition="snow", headway_std=0.3, reaction_std=0.2)
    app.steps_per_frame = 5
    frames(app, 10)
    app.colour_mode = "type"
    texts = rendered_texts(app, monkeypatch)
    assert {"Vehicle:", "car", "truck"} <= set(texts)  # the legend
    header = next(t for t in texts if t.startswith("3 lanes"))
    assert "20 % trucks" in header and "snow road" in header
    snap = app.sim.cars()
    colours = app._car_colours(snap)
    assert (colours[snap["truck"]] == viewer.TYPE_COLOURS[1]).all()
    assert (colours[~snap["truck"]] == viewer.TYPE_COLOURS[0]).all()
    labels = {text.split(":")[0] for _, text in app.stat_hits}
    assert "Trucks" in labels
    stats = {st.label: st for col in app._stat_columns(app._latest_row()) for st in col}
    assert stats["Trucks"].value.startswith(f"{app._latest_row()['trucks']} (")
    # Eight tabs (Presets and the seven parameter groups) on two balanced rows, every label fitting its tab.
    tabs = [w for w in app.panel_widgets if w.kind == "tab"]
    assert [w.name for w in tabs] == list(viewer.TABS) == ["Presets", *viewer.GROUPS] and "Vehicles" in viewer.GROUPS
    rows = sorted({w.rect.y for w in tabs})
    per_row = [sum(w.rect.y == y for w in tabs) for y in rows]
    assert len(rows) == 2 and max(per_row) - min(per_row) <= 1
    for w in tabs:
        assert app.panel_rect.contains(w.rect) and app.fonts.width(w.name, "small") < w.rect.w
    for text in viewer.help_texts():  # the help box still fits the longest help
        assert len(app.fonts.wrap(text, app.status_rect.w)) <= app.status_lines, text[:60]
    for mode in ("speed", "desired", "lane"):
        app.colour_mode = mode
        frames(app, 1)


def test_hovering_a_truck_shows_its_details(make_app, monkeypatch):
    app = make_app(truck_share=0.3, headway_std=0.3, reaction_std=0.3, density=20.0)
    frames(app, 3)
    app.paused = True
    snap = app.sim.cars()
    i = int(np.flatnonzero(snap["truck"])[0])
    ring = app.ring
    app.mouse = (round(ring.cx + snap["x"][i] * ring.k), round(ring.cy - snap["y"][i] * ring.k))
    assert app._car_at(app.mouse) == snap["slot"][i]
    texts = rendered_texts(app, monkeypatch)
    slot = int(snap["slot"][i])
    assert any(t.startswith(f"truck {app.sim.uid[slot]}") and f"{app.sim.cfg.truck_length:g} m long" in t
               for t in texts)
    assert any(t.startswith(f"time gap {app.sim.headway[slot]:.2f} s") and "reaction" in t for t in texts)


def test_vehicle_controls_and_colour_key(make_app, monkeypatch):
    app = make_app()
    frames(app, 2)
    click(app, widget(app, "tab", "Vehicles").rect.center)
    assert app.group == "Vehicles"
    assert app.inactive(widget(app, "slider", "truck_length").spec) == "no trucks"
    assert app.inactive(widget(app, "slider", "truck_share").spec) == ""
    # Without trucks Truck length is only kept for later: no "rebuilds" tag, and no rebuild.
    assert rendered_texts(app, monkeypatch).count("rebuilds") == 1  # (Truck share's)
    assert "while Truck share is above 0" in viewer.control_help(PARAM_BY_NAME["truck_length"])
    app.apply("truck_length", 15.0)
    assert app.sim.cfg.truck_length == 15.0 and app.sim.generation == 0 and app.sim.time > 0
    share = widget(app, "slider", "truck_share")
    end = (share.track.right, share.track.centery)
    for kind in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION, pygame.MOUSEBUTTONUP):
        extra = dict(rel=(0, 0), buttons=(0, 0, 0)) if kind == pygame.MOUSEMOTION else dict(button=1)
        pos = share.track.midleft if kind == pygame.MOUSEBUTTONDOWN else end
        app.handle(pygame.event.Event(kind, pos=pos, **extra))
    assert app.sim.cfg.truck_share == pytest.approx(0.3) and app.sim.generation == 1  # rebuilt on release
    assert app.inactive(widget(app, "slider", "truck_length").spec) == ""
    assert rendered_texts(app, monkeypatch).count("rebuilds") == 2
    frames(app, 3)
    click(app, widget(app, "tab", "Road").rect.center)
    road = widget(app, "choice", "road_condition")
    click(app, road.options[road.spec.choices.index("ice")].center)
    assert app.sim.cfg.road_condition == "ice" and app.sim.generation == 1  # live
    assert app.markers[-1][1] == "Road condition dry→ice"
    frames(app, 3)
    seen = []
    for _ in range(4):  # C cycles the four car colours
        app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c, mod=0, unicode="", scancode=0))
        seen.append(app.colour_mode)
    assert seen == ["desired", "lane", "type", "speed"]


@pytest.mark.parametrize("size", [(1100, 700), (1920, 1080)])
def test_open_mode_queues_draw_trucks_on_the_stubs(make_app, size):
    app = make_app(size=size, truck_share=0.3, traffic_mode="open", entry_rate=0.5, max_queue=12, follow_up=5.0)
    app.steps_per_frame = 20
    frames(app, 12)
    assert any(any(q) for q in app.sim.queue_trucks)  # trucks waiting
    sim = app.sim
    sim.queue[0] = 3  # a queue edited by hand, without types: drawn as cars, no crash
    sim.queue_times[0] = type(sim.queue_times[0])([0.0] * 3)
    sim.queue_trucks[0].clear()
    app.draw()
    app.sim.step(30)
    app.draw()


def filled_polygons(monkeypatch, draw) -> list[np.ndarray]:
    """The point lists pygame.draw.polygon fills (outlines left out) while ``draw()`` runs."""
    shapes = []
    polygon = pygame.draw.polygon

    def record(surface, colour, points, width=0):
        if width == 0:
            shapes.append(np.array(points, dtype=float))
        return polygon(surface, colour, points, width)

    monkeypatch.setattr(pygame.draw, "polygon", record)
    draw()
    monkeypatch.setattr(pygame.draw, "polygon", polygon)
    return shapes


def side(points: np.ndarray) -> float:
    """[px] Length of a polyline."""
    return float(np.hypot(*np.diff(points, axis=0).T).sum())


def test_viewer_draws_trucks_at_their_length_and_width(make_app, monkeypatch):
    app = make_app(radius=150.0, density=10.0, truck_share=0.3, truck_length=16.5)
    frames(app, 3)
    ring, sim, cfg, k = app.ring, app.sim, app.sim.cfg, app.ring.k
    snap = sim.cars()
    shapes = filled_polygons(monkeypatch, lambda: ring._draw_cars(app.screen, snap, app._car_colours(snap)))
    assert len(shapes) == snap["slot"].size
    steady = np.abs(snap["lat"] - snap["lane"]) < 1e-6  # (not tilted by a lane change)
    assert (steady & snap["truck"]).any() and (steady & ~snap["truck"]).any()
    n = viewer.BEND_POINTS
    for poly, truck in zip(np.array(shapes, dtype=object)[steady], snap["truck"][steady]):
        if truck:  # bent along the lane: two sides of BEND_POINTS points each
            left, right = poly[:n], poly[n:][::-1]
            assert 0.5 * (side(left) + side(right)) == pytest.approx(cfg.truck_length * k, rel=1e-3)
            assert np.hypot(*(left[0] - right[0])) == pytest.approx(TRUCK_WIDTH * k, rel=1e-6)
        else:
            assert len(poly) == 4
            assert np.hypot(*(poly[1] - poly[2])) == pytest.approx(cfg.car_length * k)
            assert np.hypot(*(poly[0] - poly[1])) == pytest.approx(cfg.car_width * k)
    # The entrance queues: a truck at the head, then a car, at their own sizes.
    for arm in range(sim.n_arms):
        sim.queue[arm] = 0
        sim.queue_times[arm].clear()
        sim.queue_trucks[arm].clear()
    sim.queue[0] = 2
    sim.queue_times[0].extend([0.0, 0.0])
    sim.queue_trucks[0].extend([True, False])
    queued = filled_polygons(monkeypatch, lambda: ring._draw_queues(app.screen, sim, app.fonts))
    assert len(queued) == 2
    for poly, (length, width) in zip(queued, [(cfg.truck_length, TRUCK_WIDTH), (cfg.car_length, cfg.car_width)]):
        assert np.hypot(*(poly[1] - poly[2])) == pytest.approx(length * k)
        assert np.hypot(*(poly[0] - poly[1])) == pytest.approx(width * k)


@pytest.mark.parametrize("size", [(1100, 700), (1920, 1080)])
def test_long_trucks_on_a_tight_ring_keep_to_their_lane(make_app, size):
    """Drawn straight, a 22 m truck on a 20 m ring would stick out of its lane by almost 3 m; bent
    along it, it keeps to it, and changing lanes it stays on the two lanes involved."""
    app = make_app(size=size, radius=20.0, density=40.0, truck_share=0.3, truck_length=22.0)
    frames(app, 3)
    ring, sim = app.ring, app.sim
    lw = sim.cfg.lane_width

    def radii(snap, i) -> np.ndarray:
        """[m] Distance of each point of vehicle i's outline from the ring's centre."""
        poly = np.array(ring.ring_polygons(snap, [i])[0])
        return np.hypot(poly[:, 0] - ring.cx, poly[:, 1] - ring.cy) / ring.k

    snap = sim.cars()
    trucks = np.flatnonzero(snap["truck"] & (np.abs(snap["lat"] - snap["lane"]) < 1e-6))
    assert trucks.size
    for i in trucks:
        r = sim.radii[snap["lane"][i]]
        assert np.abs(radii(snap, i) - r).max() == pytest.approx(0.5 * TRUCK_WIDTH, abs=1e-6)
    slot = snap["slot"][trucks[0]]
    sim.lane[slot], sim.lat[slot], sim.v[slot] = 1, 0.5, 0.0  # halfway from lane 0 to 1, standing (steepest)
    snap = sim.cars()
    rho = radii(snap, int(np.flatnonzero(snap["slot"] == slot)[0]))
    assert rho.min() >= sim.radii[0] - 0.5 * lw and rho.max() <= sim.radii[1] + 0.5 * lw
    app.draw()


def test_big_ring_with_trucks_and_driver_spread_draws(make_app):
    app = make_app(n_lanes=5, radius=300, car_length=2, min_gap=1, density=200, truck_share=0.2, reaction_std=0.2,
                   headway_std=0.3)
    assert int(app.sim.alive.sum()) > 1400
    for mode in ("speed", "type"):
        app.colour_mode = mode
        frames(app, 2)
    assert_valid(app.sim)


def test_viewer_command_line_takes_the_new_flags():
    args = viewer.parse_args(["--truck-share", "0.1", "--truck_length", "16.5", "--reaction-std", "0.2",
                              "--headway-std", "0.3", "--road-condition", "wet"])
    cfg = viewer.config_from_args(args)
    assert (cfg.truck_share, cfg.truck_length, cfg.reaction_std, cfg.headway_std, cfg.road_condition) == (
        0.1, 16.5, 0.2, 0.3, "wet")


# -------------------------------------------------------------- experiments
def test_experiments_run_with_trucks(tmp_path, capsys, no_figures):
    out = tmp_path / "trucks"
    experiments.main(["run", "--duration", "10", "--seed", "1", "--truck-share", "0.2", "--road-condition", "wet",
                      "--headway-std", "0.3", "--density", "15", "--out", str(out)])
    printed = capsys.readouterr().out
    for text in ("Trucks on the ring", "Mean speed, cars / trucks", "20 % trucks of 12 m", "wet road",
                 "headway 1.2 +/- 0.3 s"):
        assert text in printed, text
    summary = pd.read_csv(f"{out}_summary.csv").iloc[0]
    assert summary["truck_share"] == 0.2 and summary["road_condition"] == "wet" and summary["headway_std"] == 0.3
    assert summary["trucks"] > 0 and summary["truck_speed"] < summary["car_speed"]
    ts = pd.read_csv(f"{out}_timeseries.csv")
    assert {"trucks", "truck_speed", "car_speed"} <= set(ts.columns)


@pytest.mark.parametrize("command", ["run", "sweep"])
def test_command_line_help_stays_plain_ascii(command, capsys):
    """Windows consoles often mangle anything else, so the help texts the flags print are ASCII."""
    with pytest.raises(SystemExit):
        experiments.main([command, "--help"])
    printed = capsys.readouterr().out
    assert "--truck-share" in printed and printed.isascii()


def test_sweep_over_the_truck_share(tmp_path, capsys, no_figures):
    out = tmp_path / "sweep"
    experiments.main(["sweep", "--x", "truck_share=0,0.2", "--series", "road_condition=dry,ice", "--seeds", "1",
                      "--warmup", "0", "--measure", "8", "--workers", "1", "--out", str(out)])
    printed = capsys.readouterr().out
    assert "trucks" in printed and "truck_m/s" in printed
    df = pd.read_csv(f"{out}.csv")
    assert list(df["truck_share"]) == [0.0, 0.2, 0.0, 0.2] and list(df["road_condition"]) == ["dry"] * 2 + ["ice"] * 2
    assert (df.loc[df["truck_share"] == 0, "trucks"] == 0).all() and (df.loc[df["truck_share"] > 0, "trucks"] > 0).all()
    assert df["measured_mean_speed"].iloc[2] < df["measured_mean_speed"].iloc[0]  # ice is slower
    assert "braking limit" in experiments.DEFINITIONS and "1.35" in experiments.DEFINITIONS
