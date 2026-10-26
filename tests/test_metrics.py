"""Measurements: flux, congestion, spatial distribution, speed profiles and the Recorder."""

from __future__ import annotations

import numpy as np
import pytest

from roundabout_sim.engine import TWO_PI, RingSim
from roundabout_sim.metrics import Recorder, count_jams, sector_occupancy, snapshot, speed_profile


def ring_with(n_cars: int, **overrides) -> RingSim:
    """Closed single-lane ring holding exactly ``n_cars`` cars."""
    sim = RingSim(seed=0, n_lanes=1, n_arms=0, radius=40.0, **overrides)
    sim.reset(density=n_cars * 1000.0 / sim.total_lane_length)
    assert int(sim.alive.sum()) == n_cars
    return sim


def sorted_slots(sim: RingSim) -> np.ndarray:
    slots = np.flatnonzero(sim.alive)
    return slots[np.argsort(sim.theta[slots])]


def averaged(sim: RingSim, keys: tuple[str, ...], samples: int, every: int = 10) -> dict[str, float]:
    rows = []
    for _ in range(samples):
        sim.step(every)
        rows.append(snapshot(sim))
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


# ----------------------------------------------------------------------- flow
def test_detector_flux_matches_ring_flux_on_closed_ring():
    sim = RingSim(seed=2, n_lanes=2, n_arms=0, radius=40.0, density=40.0)
    sim.step(100)
    rows = []
    for _ in range(240):
        sim.step(10)
        rows.append(snapshot(sim, window=30.0))
    settled = rows[30:]  # once the detector's 30 s window is full
    detector = np.mean([r["detector_flux"] for r in settled])
    ring = np.mean([r["flux"] for r in settled])
    assert ring > 0
    assert detector == pytest.approx(ring, rel=0.15)


def test_flux_and_mean_speed_formulas():
    sim = RingSim(seed=1, n_lanes=3, n_arms=0, density=50.0)
    sim.step(50)
    cars = sim.cars()
    radius = sim.cfg.radius + sim.cfg.lane_width * cars["lane"]
    row = snapshot(sim)
    assert row["flux"] == pytest.approx(np.sum(cars["v"] / (TWO_PI * radius)))
    assert row["mean_speed"] == pytest.approx(cars["v"].mean())
    assert row["cars"] == cars["slot"].size
    assert row["density"] == pytest.approx(cars["slot"].size / sim.total_lane_length * 1000.0)
    assert sum(row[f"lane{l}_cars"] for l in range(3)) == row["cars"]
    for lane in range(3):
        assert row[f"lane{lane}_speed"] == pytest.approx(cars["v"][cars["lane"] == lane].mean())


# ----------------------------------------------------------------- congestion
@pytest.mark.parametrize("n_arms", [0, 4])
def test_free_flow_has_no_jams(n_arms):
    sim = RingSim(seed=4, n_arms=n_arms, density=10.0)
    stats = averaged(sim, ("jam_fraction", "n_jams", "speed_deficit"), samples=60)
    assert stats["jam_fraction"] < 0.02
    assert stats["n_jams"] < 0.2
    assert stats["speed_deficit"] < 0.2


def test_dense_slow_reacting_traffic_jams():
    sim = RingSim(seed=0, n_arms=0, density=100.0, reaction_time=1.0)
    sim.step(600)
    rows = [snapshot(sim)]
    for _ in range(10):
        sim.step(10)
        rows.append(snapshot(sim))
    assert np.mean([r["jam_fraction"] for r in rows]) > 0.1
    assert min(r["n_jams"] for r in rows) >= 1
    assert np.mean([r["speed_deficit"] for r in rows]) > 0.3


@pytest.mark.parametrize("pattern, jams", [
    ("JJFFJJJFJF", 2),
    ("JFFFFFFFJJ", 1),  # a jam that wraps past angle 0 is one jam
    ("JJJJJJJJJJ", 1),
    ("FFFFFFFFFF", 0),
    ("JFJFJFJFJF", 0),  # isolated slow cars are not jams
])
def test_count_jams(pattern, jams):
    sim = ring_with(len(pattern))
    slots = sorted_slots(sim)
    slow = np.array([c == "J" for c in pattern])
    sim.v[slots] = np.where(slow, 0.0, sim.desired_speed(slots))
    assert count_jams(sim, slots) == jams
    row = snapshot(sim)
    assert row["n_jams"] == jams
    assert row["jam_fraction"] == pytest.approx(slow.mean())


# --------------------------------------------------------------------- space
@pytest.mark.parametrize("n_lanes", [1, 4])
@pytest.mark.parametrize("car_length, car_width", [(2.0, 1.2), (4.0, 1.8)])
def test_occupancy_is_car_area_over_road_area(n_lanes, car_length, car_width):
    sim = RingSim(seed=3, n_lanes=n_lanes, n_arms=0, density=60.0, car_length=car_length,
                  car_width=car_width)
    sim.step(20)
    cfg = sim.cfg
    n = int(sim.alive.sum())
    lane_lengths = TWO_PI * (cfg.radius + cfg.lane_width * np.arange(n_lanes))
    road_area = lane_lengths.sum() * cfg.lane_width
    assert snapshot(sim)["occupancy"] == pytest.approx(n * car_length * car_width / road_area)
    sectors = sector_occupancy(sim, sectors=12)
    assert sectors.shape == (12,)
    assert sectors.mean() == pytest.approx(n * car_length / lane_lengths.sum())


@pytest.mark.parametrize("seed", range(4))
def test_clustering_ranks_initial_spacings(seed):
    # Random scatter is subtracted: even and random starts both read ~0, a packed block high.
    clustering = {s: snapshot(RingSim(seed=seed, n_arms=0, init_spacing=s))["clustering"]
                  for s in ("uniform", "random", "clustered")}
    assert clustering["uniform"] < 0.1
    assert clustering["random"] < 0.4
    assert clustering["clustered"] > max(0.5, clustering["random"])


@pytest.mark.parametrize("n_cars", [120, 240, 480])
def test_clustering_does_not_grow_with_car_count(n_cars):
    # Half the ring empty reads ~1 at any car count (minus a small-count bias).
    sim = RingSim(seed=0, n_lanes=1, n_arms=0, radius=300.0, car_length=2.0, min_gap=1.0)
    sim.reset(density=n_cars * 1000.0 / sim.total_lane_length)
    slots = sorted_slots(sim)
    assert slots.size == n_cars
    sim.theta[slots] = np.linspace(0, np.pi, slots.size, endpoint=False)
    assert snapshot(sim)["clustering"] == pytest.approx(1.0, abs=0.12)


def test_order_param_uniform_vs_clustered_single_lane():
    uniform = snapshot(RingSim(seed=0, n_lanes=1, n_arms=0, init_spacing="uniform"))
    clustered = snapshot(RingSim(seed=0, n_lanes=1, n_arms=0, init_spacing="clustered"))
    assert uniform["order_param"] < 0.05
    assert clustered["order_param"] > 0.5


@pytest.mark.parametrize("n_lanes", [3, 5])
def test_uniform_start_is_balanced_on_every_lane_count(n_lanes):
    for seed in range(5):
        row = snapshot(RingSim(seed=seed, n_lanes=n_lanes, n_arms=0, init_spacing="uniform"))
        assert row["order_param"] < 0.05


@pytest.mark.parametrize("n_lanes", [3, 5])
def test_clustered_start_is_one_sided_on_every_lane_count(n_lanes):
    order = [snapshot(RingSim(seed=seed, n_lanes=n_lanes, n_arms=0, init_spacing="clustered"))["order_param"]
             for seed in range(10)]
    assert min(order) > 0.5


@pytest.mark.parametrize("centre", [1.0, 6.2])
def test_order_angle_points_at_the_bunch(centre):
    sim = ring_with(12)
    slots = sorted_slots(sim)
    sim.theta[slots] = (centre + np.linspace(-0.2, 0.2, slots.size)) % TWO_PI
    row = snapshot(sim)
    assert row["order_param"] > 0.95
    assert abs((row["order_angle"] - centre + np.pi) % TWO_PI - np.pi) < 1e-6


# ------------------------------------------------------------- speed profile
def test_speed_profile_means_and_empty_bins():
    sim = RingSim(seed=1, n_lanes=2, n_arms=0, density=10.0)
    slots = np.flatnonzero(sim.alive)
    assert slots.size >= 4
    width = TWO_PI / 12
    sim.lane[slots] = 0
    sim.lane[slots[1]] = 1
    sim.theta[slots] = 7.5 * width
    sim.theta[slots[:2]] = [2.3 * width, 2.7 * width]
    sim.v[slots] = 10.0
    sim.v[slots[:2]] = [3.0, 5.0]

    profile = speed_profile(sim, bins=12)
    assert profile.shape == (12,)
    assert profile[2] == pytest.approx(4.0)
    assert profile[7] == pytest.approx(10.0)
    empty = np.ones(12, dtype=bool)
    empty[[2, 7]] = False
    assert np.isnan(profile[empty]).all()

    lane0, lane1 = speed_profile(sim, bins=12, lane=0), speed_profile(sim, bins=12, lane=1)
    assert lane0[2] == pytest.approx(3.0) and lane0[7] == pytest.approx(10.0)
    assert lane1[2] == pytest.approx(5.0)
    assert np.isnan(np.delete(lane1, 2)).all()


# ------------------------------------------------------------------- Recorder
@pytest.mark.parametrize("interval", [0.5, 0.25, 1.0])
def test_recorder_samples_at_interval(interval):
    sim = RingSim(seed=0, n_lanes=1, n_arms=0, density=30.0)
    rec = Recorder(sim, interval=interval)
    rec.update()
    rec.update()  # same instant: no second sample
    for _ in range(100):
        sim.step()
        rec.update()
    times = rec.frame()["time"].to_numpy()
    due = interval * np.arange(times.size)
    assert times.size == int(round(10.0 / interval)) + 1
    # Each sample is taken at the first step at or after it is due.
    assert (times >= due - 1e-9).all() and (times < due + sim.cfg.dt - 1e-9).all()


def test_recorder_respects_maxlen():
    sim = RingSim(seed=0, n_lanes=1, n_arms=0, density=30.0)
    rec = Recorder(sim, interval=0.1, maxlen=5, bins=36, sectors=8)
    for _ in range(20):
        sim.step()
        rec.update()
    frame = rec.frame()
    assert len(frame) == 5
    assert frame["time"].iloc[-1] == pytest.approx(sim.time)
    times, speeds = rec.spacetime_array()
    assert times.shape == (5,) and speeds.shape == (5, 36)
    assert rec.occupancy_array().shape == (5, 8)


def test_recorder_clear_restarts_after_reset():
    sim = RingSim(seed=0, n_lanes=1, n_arms=0, density=30.0)
    rec = Recorder(sim, interval=0.5)
    for _ in range(30):
        sim.step()
        rec.update()
    sim.reset()
    rec.clear()
    assert len(rec.frame()) == 0 and rec.spacetime_array()[1].shape == (0, rec.bins)
    assert rec.occupancy_array().shape == (0, rec.sectors)
    rec.update()
    assert rec.frame()["time"].tolist() == [0.0]


def test_recorder_holds_the_requested_graphs():
    """Average velocity, flux, coverage / bunching and congestion over time."""
    sim = RingSim(seed=0)
    rec = Recorder(sim, interval=1.0)
    for _ in range(30):
        sim.step()
        rec.update()
    frame = rec.frame()
    for column in ("time", "mean_speed", "flux", "detector_flux", "occupancy", "clustering",
                   "order_param", "order_angle", "jam_fraction", "speed_deficit", "n_jams"):
        assert column in frame.columns, column
    assert frame[["mean_speed", "flux", "occupancy", "jam_fraction"]].notna().all().all()


def test_snapshot_of_empty_ring():
    row = snapshot(RingSim(seed=0, density=0.0))
    assert row["cars"] == 0
    for key in ("flux", "occupancy", "n_jams", "trucks"):
        assert row[key] == 0.0, key
    # No car to average over: the per-car measures are NaN (time means skip them), not a speed of 0.
    for key in ("mean_speed", "speed_std", "jam_fraction", "speed_deficit", "clustering", "order_param",
                "spacing_gini", "mean_trip_time"):
        assert np.isnan(row[key]), key


def test_trip_time_and_exit_rate_with_arms():
    sim = RingSim(seed=1, exit_prob=0.5)
    sim.step(400)
    row = snapshot(sim)
    assert sim.n_exited > 0
    assert row["exits_per_min"] > 0
    assert 0 < row["mean_trip_time"] < sim.time
