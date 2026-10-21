"""Regression tests for the issues found in the physics/robustness review."""

from __future__ import annotations

import numpy as np
import pytest

from roundabout_sim import engine
from roundabout_sim.engine import RingSim
from roundabout_sim.metrics import Recorder, snapshot


def min_bumper_gap(sim: RingSim) -> float:
    worst = np.inf
    for lane in range(sim.n_lanes):
        slots = np.flatnonzero(sim.alive & (sim.lane == lane))
        if slots.size < 2:
            continue
        th = np.sort(sim.theta[slots])
        arcs = np.diff(np.append(th, th[0] + engine.TWO_PI)) * sim.radii[lane]
        worst = min(worst, float(arcs.min() - sim.cfg.car_length))
    return worst


def test_rolling_rates_do_not_consume_engine_logs():
    sim = RingSim(seed=1, n_lanes=2, density=60)
    sim.step(1800)
    logs = (len(sim.detector_log), len(sim.exit_log), len(sim.lane_change_log))
    wide = snapshot(sim, window=60.0)
    snapshot(sim, window=5.0)
    assert (len(sim.detector_log), len(sim.exit_log), len(sim.lane_change_log)) == logs
    again = snapshot(sim, window=60.0)
    for key in ("detector_flux", "exits_per_min", "lane_changes_per_min", "mean_trip_time"):
        assert again[key] == pytest.approx(wide[key], nan_ok=True)


def test_logs_are_pruned_to_the_horizon():
    sim = RingSim(seed=0, n_lanes=1, n_arms=0, density=40)
    sim.step(int((engine.LOG_HORIZON + 120) / sim.cfg.dt))
    assert sim.detector_log[0][0] >= sim.time - engine.LOG_HORIZON - 1.0


def test_recorder_restarts_after_rebuild():
    sim = RingSim(seed=0)
    rec = Recorder(sim, interval=0.5)
    for _ in range(100):
        sim.step()
        rec.update()
    assert sim.set(n_lanes=2) is True
    for _ in range(100):
        sim.step()
        rec.update()
    times = [row["time"] for row in rec.rows]
    assert times[0] == pytest.approx(0.1) and len(times) == 20
    assert "lane2_cars" not in rec.rows[-1]


def test_raising_exit_chance_live_starts_exits():
    sim = RingSim(seed=0, exit_prob=0.0)
    sim.step(100)
    sim.set(exit_prob=0.5)
    sim.step(1500)
    assert sim.n_exited > 20


def test_unchanged_exit_chance_does_not_reroute():
    sim = RingSim(seed=0)
    sim.step(50)
    before = sim.exit_at.copy()
    sim.set(exit_prob=sim.cfg.exit_prob)
    np.testing.assert_array_equal(sim.exit_at, before)


def test_long_clamped_chain_never_overlaps():
    # A stopped car with six fast cars bunched behind it, all acting on stale
    # (zero) accelerations: the no-overlap constraint must hold for every car.
    sim = RingSim(seed=0, n_lanes=1, n_arms=0, radius=40.0, reaction_time=2.0, driver_noise=0.0,
                  density=7 * 1000.0 / (engine.TWO_PI * 40.0))  # target: exactly 7 cars
    sim.alive[:] = False
    pitch = sim.cfg.car_length + 0.35
    sim._spawn(np.zeros(7, dtype=int), np.arange(7) * pitch / sim.radii[0], np.r_[15.0 * np.ones(6), 0.0])
    sim.step()
    assert int(sim.alive.sum()) == 7
    assert min_bumper_gap(sim) >= engine.SAFE_MARGIN - 1e-9


def test_two_cars_never_change_into_the_same_empty_lane():
    for seed in range(10):
        sim = RingSim(seed=seed, n_lanes=3, n_arms=0, density=1, lane_change_motive="random",
                      lane_change_rate=1.0)
        sim.alive[:] = False
        a, b = sim._spawn(np.array([0, 2]), np.array([1.0, 1.0]), np.array([10.0, 10.0]))
        sim.lc_cool[[a, b]] = 0.0
        sim.cfg = sim.cfg.updated(lane_change_rate=1.0 / sim.cfg.dt)  # both consider a change now
        sim._lane_changes(np.flatnonzero(sim.alive))
        assert not (sim.lane[a] == 1 and sim.lane[b] == 1)


def test_emergencies_count_episodes_not_steps():
    sim = RingSim(seed=0, n_lanes=1, n_arms=0, density=60, reaction_time=2.0)
    sim.step(3000)
    total = sum(count for _, count in sim.emergency_log)
    assert total == sim.n_emergency
    assert sim.n_emergency < 0.2 * 3000 * int(sim.alive.sum())


def test_braking_never_exceeds_the_physical_limit_from_driving():
    sim = RingSim(seed=0, n_lanes=1, n_arms=0, density=50, reaction_distance=20, reaction_time=0.0,
                  driver_noise=0.0)
    for _ in range(600):
        v_before = sim.v.copy()
        sim.step()
        live = sim.alive
        decel = (v_before[live] - sim.v[live]) / sim.cfg.dt
        flagged = sim.emergency_until[live] > sim.time  # every over-limit stop is an emergency
        assert np.all(decel[~flagged] <= engine.MAX_DECEL + 1e-6)


def test_stop_and_go_index_separates_smooth_dense_flow_from_waves():
    smooth = RingSim(seed=0, n_lanes=1, n_arms=0, density=130, driver_noise=0.0, reaction_time=0.0,
                     speed_std=0.0)
    waves = RingSim(seed=0, n_lanes=1, n_arms=0, density=80, reaction_time=0.8)
    for sim in (smooth, waves):
        sim.step(3000)
    assert snapshot(smooth, window=120)["stops_per_car_min"] < 0.05
    assert snapshot(smooth)["jam_fraction"] > 0.9  # crawling, but not stop-and-go
    assert snapshot(waves, window=120)["stops_per_car_min"] > 0.3


def test_queues_are_capped_and_ring_keeps_its_density():
    sim = RingSim(seed=2, density=110, exit_prob=0.3)
    for _ in range(30):
        sim.step(100)
        assert np.all(sim.queue <= engine.MAX_QUEUE)
        assert int(sim.alive.sum()) + int(sim.queue.sum()) == sim.target_cars()
    assert int(sim.alive.sum()) >= sim.target_cars() - engine.MAX_QUEUE * sim.n_arms


def test_jam_origins_add_up_to_jam_heads():
    sim = RingSim(seed=3, density=80, reaction_time=0.8)
    sim.step(2000)
    row = snapshot(sim)
    parts = sum(row[k] for k in ("jams_following", "jams_merge", "jams_exit", "jams_entry", "jams_other"))
    assert parts == row["n_jams"]
    assert row["jams_following"] >= 1


def test_closed_ring_phantom_jams_are_car_following():
    sim = RingSim(seed=0, n_arms=0, density=80, reaction_time=0.8)
    sim.step(2000)
    row = snapshot(sim)
    assert row["n_jams"] >= 1
    assert row["jams_merge"] == row["jams_exit"] == row["jams_entry"] == row["jams_other"] == 0


def test_inner_lanes_judged_with_their_own_speed_cap():
    sim = RingSim(seed=0, radius=20, n_arms=0, lateral_accel=4.0, mean_speed=20, speed_std=0)
    slots = np.flatnonzero(sim.alive)[:3]
    caps = [sim.desired_speed(slots, lane)[0] for lane in range(3)]
    assert caps[0] < caps[1] < caps[2]


def test_clustered_start_is_one_block_across_lanes():
    sim = RingSim(seed=4, n_lanes=4, n_arms=0, init_spacing="clustered", density=30)
    row = snapshot(sim)
    assert row["order_param"] > 0.5
    spread = []
    for lane in range(4):
        th = sim.theta[sim.alive & (sim.lane == lane)]
        spread.append(np.angle(np.exp(1j * th).mean()))
    assert np.ptp(np.unwrap(spread)) < 0.3


def test_heading_tilts_during_a_lane_change():
    sim = RingSim(seed=0, n_lanes=2, n_arms=0, density=10)
    slot = int(np.flatnonzero(sim.alive & (sim.lane == 0))[0])
    sim.lane[slot] = 1  # mid lane change outwards: lat is still 0
    cars = sim.cars()
    k = int(np.flatnonzero(cars["slot"] == slot)[0])
    assert cars["heading"][k] < cars["theta"][k] + np.pi / 2


# ------------------------------------------------------------ second review
def test_merging_leavers_cannot_gridlock_the_ring():
    sim = RingSim(seed=0, n_lanes=5, n_arms=8, exit_prob=1.0, density=140)
    sim.step(3000)
    exited = sim.n_exited
    sim.step(1200)
    assert sim.n_exited - exited > 20  # deadlocked, this was 0
    assert snapshot(sim)["mean_speed"] > 0.2


def test_density_raise_slows_the_cars_behind_new_ones():
    sim = RingSim(seed=0, n_lanes=1, n_arms=0, density=20, reaction_time=0.6)
    sim.step(600)
    before = sim.n_emergency
    sim.set(density=100)
    sim.step(50)
    assert sim.n_emergency - before <= 5
    assert min_bumper_gap(sim) >= engine.SAFE_MARGIN - 1e-9


def test_respaced_cars_still_exit_at_their_exit():
    sim = RingSim(seed=0, n_lanes=1, n_arms=4, radius=40.0, density=60, exit_prob=0.5)
    sim.step(100)
    distances = []
    exits = sim._exits

    def recording_exits():  # where each leaver is when it leaves
        due = sim.alive & (sim.travel >= sim.exit_at) & (sim.lane == sim.outer)
        for slot in np.flatnonzero(due):
            off = np.abs((sim.theta[slot] - sim.exit_angle + np.pi) % engine.TWO_PI - np.pi).min()
            distances.append(off * sim.r_out)
        exits()

    sim._exits = recording_exits
    assert sim._respace_lanes(3)
    sim.step(3000)
    assert len(distances) > 20 and np.percentile(distances, 95) < 1.5


def test_slowing_for_an_exit_is_not_a_stop():
    sim = RingSim(seed=0, density=10, exit_prob=0.3, turn_speed=2.0)
    sim.step(3000)
    row = snapshot(sim, window=120)
    assert row["n_jams"] == 0
    assert row["stops_per_car_min"] < 0.1


def test_changing_free_speed_live_is_not_a_wave_of_stops():
    sim = RingSim(seed=0, n_lanes=1, n_arms=0, density=20, mean_speed=5.0)
    sim.step(600)
    stops = sim.n_stops
    sim.set(mean_speed=20.0)
    sim.step(10)
    assert sim.n_stops - stops <= 1


def test_population_matches_target_every_step_near_jam():
    sim = RingSim(seed=0, n_lanes=2, n_arms=4, density=100)
    sim.step(50)
    sim.set(density=0.99 * sim.jam_density)
    for _ in range(20):
        sim.step()
        assert int(sim.alive.sum()) + int(sim.queue.sum()) == sim.target_cars()


@pytest.mark.parametrize("bad", [dict(n_lanes=0), dict(dt=0.0), dict(n_arms=-1), dict(density=-5.0),
                                 dict(init_spacing="bogus"), dict(exit_prob=1.5)])
def test_invalid_values_are_rejected_and_leave_the_sim_intact(bad):
    sim = RingSim(seed=0)
    before = sim.cfg
    with pytest.raises(ValueError):
        sim.set(**bad)
    assert sim.cfg == before
    sim.step(10)


def test_tiny_standstill_gap_fills_to_jam_without_crashing():
    sim = RingSim(seed=0, n_lanes=3, n_arms=0, min_gap=0.1, density=115)
    sim.set(density=300)
    sim.step(30)
    assert int(sim.alive.sum()) == sim.target_cars()


def test_seed_change_applies_on_rebuild():
    sim = RingSim(seed=1)
    sim.set(seed=99, n_lanes=2)
    assert sim.seed == 99


def test_colour_ramps_accept_nan():
    from roundabout_sim import palette
    for f in (palette.sequential_rgb, palette.speed_rgb):
        out = f(np.array([np.nan, 0.0, 1.0]))
        assert np.isfinite(out).all() and out.shape == (3, 3)


# ------------------------------------------------------------- third review
def _jammed_share(sim: RingSim, steps: int = 2400, after: int = 1200) -> float:
    rows = []
    for k in range(steps):
        sim.step()
        if k >= after and k % 10 == 0:
            rows.append(snapshot(sim)["jam_fraction"])
    return float(np.mean(rows))


def test_longer_reaction_distance_damps_phantom_jams():
    short = _jammed_share(RingSim(seed=0, n_lanes=1, n_arms=0, density=80, reaction_distance=20))
    long = _jammed_share(RingSim(seed=0, n_lanes=1, n_arms=0, density=80, reaction_distance=60))
    assert short > 0.15 and long < 0.75 * short


def test_anticipation_changes_nothing_in_even_flow():
    a = RingSim(seed=0, n_lanes=1, n_arms=0, density=60, driver_noise=0.0, speed_std=0.0, reaction_distance=20)
    b = RingSim(seed=0, n_lanes=1, n_arms=0, density=60, driver_noise=0.0, speed_std=0.0, reaction_distance=100)
    a.step(300)
    b.step(300)
    np.testing.assert_allclose(a.v[a.alive], b.v[b.alive], atol=1e-6)


def test_noise_cannot_brake_past_the_physical_limit():
    sim = RingSim(seed=0, n_lanes=1, n_arms=0, density=60, driver_noise=1.5, reaction_time=0.0)
    for _ in range(600):
        v_before = sim.v.copy()
        sim.step()
        live = sim.alive
        decel = (v_before[live] - sim.v[live]) / sim.cfg.dt
        flagged = sim.emergency_until[live] > sim.time
        assert np.all(decel[~flagged] <= engine.MAX_DECEL + 1e-6)


def test_failed_rebuild_keeps_the_old_seed():
    sim = RingSim(seed=5)
    with pytest.raises(ValueError):
        sim.set(seed=-1, n_lanes=2)
    assert sim.seed == 5 and sim.cfg.seed is None or sim.cfg.seed == 5
    assert sim.set(n_lanes=2) is True


def test_seed_set_alone_is_used_from_then_on():
    sim = RingSim(seed=1)
    sim.set(seed=99)
    assert sim.seed == 99
    sim.set(n_lanes=2)
    assert sim.seed == 99


@pytest.mark.parametrize("bad", [dict(driver_noise=float("inf")), dict(n_lanes=float("nan")),
                                 dict(n_lanes=2.5), dict(dt=float("inf"))])
def test_non_finite_and_fractional_values_are_rejected(bad):
    sim = RingSim(seed=0)
    with pytest.raises(ValueError):
        sim.set(**bad)


def test_unbounded_density_and_reaction_distance_are_allowed():
    sim = RingSim(seed=0, n_lanes=1, n_arms=0, density=float("inf"), reaction_distance=float("inf"))
    sim.step(20)
    assert int(sim.alive.sum()) == sim.capacity()


def test_replacements_slotted_in_do_not_knock_the_car_behind():
    sim = RingSim(seed=0, density=50, exit_prob=0.3)
    sim.step(300)
    worst = 0.0
    for _ in range(1500):
        v_before, uid_before = sim.v.copy(), np.where(sim.alive, sim.uid, -1)
        sim.step()
        same = sim.alive & (sim.uid == uid_before)  # slots can be reused within a step
        cut = (v_before[same] - sim.v[same]) / sim.cfg.dt
        harsh = sim.emergency_until[same] > sim.time
        worst = max(worst, float(cut[~harsh].max()))
    assert worst <= engine.MAX_DECEL + 1e-6


def test_live_exit_chance_change_logs_no_false_stops():
    sim = RingSim(seed=0, n_lanes=2, density=40, exit_prob=0.5, turn_speed=2.0)
    sim.step(600)
    stops = sim.n_stops
    sim.set(exit_prob=0.0)
    sim.step()
    assert sim.n_stops - stops <= 3


def test_order_baseline_and_spacing_gini():
    uniform = snapshot(RingSim(seed=0, n_lanes=1, n_arms=0, density=40, init_spacing="uniform"))
    random_ = snapshot(RingSim(seed=0, n_lanes=1, n_arms=0, density=40, init_spacing="random"))
    clustered = snapshot(RingSim(seed=0, n_lanes=1, n_arms=0, density=40, init_spacing="clustered"))
    assert uniform["order_baseline"] == pytest.approx(np.sqrt(np.pi / (4 * uniform["cars"])))
    assert uniform["spacing_gini"] < 0.05 < random_["spacing_gini"] < clustered["spacing_gini"]
    assert clustered["order_param"] > 3 * clustered["order_baseline"]
