"""RingSim behaviour: invariants, population control, determinism, driver model, exits,
perturbations and lane changes."""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from roundabout_sim.engine import LC_COOLDOWN, TWO_PI, RingSim

# Per-car arrays that must never hold NaN for a live car (``exit_at`` may be +inf).
STATE_ARRAYS = ("theta", "travel", "v", "z", "exit_at", "born", "lat", "lc_cool", "noise")


def lane_gaps(sim: RingSim) -> dict[int, np.ndarray]:
    """Bumper-to-bumper gaps [m] to the car ahead, per lane, from the configured geometry."""
    cfg = sim.cfg
    cars = sim.cars()
    gaps = {}
    for lane in range(cfg.n_lanes):
        th = np.sort(cars["theta"][cars["lane"] == lane])
        if th.size:
            r = cfg.radius + cfg.lane_width * lane
            gaps[lane] = np.diff(th, append=th[0] + TWO_PI) * r - cfg.car_length
    return gaps


def assert_valid(sim: RingSim) -> None:
    cfg = sim.cfg
    cars = sim.cars()
    slots = cars["slot"]
    for name in STATE_ARRAYS:
        assert not np.isnan(getattr(sim, name)[slots]).any(), f"NaN in {name}"
    assert not np.isnan(sim.acc_hist[:, slots]).any(), "NaN in acceleration history"
    assert np.isfinite(cars["v"]).all() and (cars["v"] >= 0).all()
    assert ((cars["theta"] >= 0) & (cars["theta"] < TWO_PI)).all()
    assert ((cars["lane"] >= 0) & (cars["lane"] < cfg.n_lanes)).all()
    assert ((cars["lat"] >= 0) & (cars["lat"] <= cfg.n_lanes)).all()
    assert np.unique(cars["uid"]).size == cars["uid"].size
    for lane, gaps in lane_gaps(sim).items():
        assert gaps.min() >= -1e-6, f"cars overlap in lane {lane} by {-gaps.min():.3f} m"


def assert_population(sim: RingSim) -> None:
    """Cars on the ring plus cars queued at the entrances match the density setting."""
    on_ring, queued = int(sim.alive.sum()), int(sim.queue.sum())
    assert on_ring + queued == sim.target_cars()
    if sim.n_arms == 0:
        assert queued == 0


def same_state(a: RingSim, b: RingSim) -> bool:
    ca, cb = a.cars(), b.cars()
    return all(ca[k].shape == cb[k].shape and np.array_equal(ca[k], cb[k]) for k in ca) \
        and np.array_equal(a.queue, b.queue)


def density_for(level: str, sim: RingSim) -> float:
    return {"low": 10.0, "medium": 60.0, "near_jam": 0.95 * sim.jam_density}[level]


# ------------------------------------------------------------ invariant matrix
LANES, ARMS, LEVELS = (1, 2, 3, 4, 5), (0, 1, 4, 8), ("low", "medium", "near_jam")
SPACINGS, MOTIVES, REACTIONS, NOISES = ("uniform", "random", "clustered"), ("advantage", "random"), (0.0, 1.5), (0.0, 1.0)
GEOMETRY = list(itertools.product(LANES, ARMS, LEVELS))
DRIVERS = list(itertools.product(SPACINGS, MOTIVES, REACTIONS, NOISES))
# Walking the driver settings at a stride coprime to their count puts every pair
# of values of any two factors in some case (checked below) with 60 cases
# instead of the full 1440-case product.
MATRIX = [g + DRIVERS[(7 * i) % len(DRIVERS)] for i, g in enumerate(GEOMETRY)]
MATRIX_IDS = [f"L{c[0]}-A{c[1]}-{c[2]}-{c[3]}-{c[4]}-rt{c[5]}-n{c[6]}" for c in MATRIX]


def test_matrix_covers_every_pair_of_settings():
    levels = (LANES, ARMS, LEVELS, SPACINGS, MOTIVES, REACTIONS, NOISES)
    for a, b in itertools.combinations(range(len(levels)), 2):
        seen = {(case[a], case[b]) for case in MATRIX}
        assert seen == set(itertools.product(levels[a], levels[b]))


@pytest.mark.parametrize("n_lanes, n_arms, level, spacing, motive, reaction, noise", MATRIX, ids=MATRIX_IDS)
def test_invariants_hold(n_lanes, n_arms, level, spacing, motive, reaction, noise):
    sim = RingSim(seed=11, n_lanes=n_lanes, n_arms=n_arms, radius=30.0, init_spacing=spacing,
                  lane_change_motive=motive, reaction_time=reaction, driver_noise=noise,
                  lane_change_rate=0.5, exit_prob=0.2)
    sim.reset(density=density_for(level, sim))
    assert_valid(sim)
    assert_population(sim)
    for _ in range(6):
        sim.step(50)
        assert_valid(sim)
        assert_population(sim)


# ------------------------------------------------------------------ geometry
@pytest.mark.parametrize("n_lanes", [1, 3, 5])
def test_lane_geometry(n_lanes):
    sim = RingSim(seed=0, n_lanes=n_lanes, radius=50.0, lane_width=3.0)
    assert sim.n_lanes == n_lanes
    np.testing.assert_allclose(sim.radii, 50.0 + 3.0 * np.arange(n_lanes))
    np.testing.assert_allclose(sim.lane_len, TWO_PI * sim.radii)
    assert sim.total_lane_length == pytest.approx(sim.lane_len.sum())


@pytest.mark.parametrize("n_arms", [1, 2, 3, 4, 6, 8])
def test_arms_evenly_spaced_exit_before_entry(n_arms):
    sim = RingSim(seed=0, n_arms=n_arms)
    assert sim.n_arms == n_arms and sim.queue.shape == (n_arms,)
    spacing = np.diff(np.sort(sim.arm_angle), append=np.sort(sim.arm_angle)[0] + TWO_PI)
    np.testing.assert_allclose(spacing, TWO_PI / n_arms)
    # Driving anticlockwise a car passes an arm's exit just before its entry.
    lead = (sim.entry_angle - sim.exit_angle) % TWO_PI
    assert ((lead > 0) & (lead < TWO_PI / n_arms)).all()
    # The detector sits strictly between arms.
    assert np.abs((sim.arm_angle - sim.detector_angle + np.pi) % TWO_PI - np.pi).min() > 1e-3


def test_closed_ring_has_no_arms():
    sim = RingSim(seed=0, n_arms=0)
    assert sim.n_arms == 0 and sim.arm_angle.size == 0 and sim.queue.size == 0


# ---------------------------------------------------------------- population
@pytest.mark.parametrize("n_lanes", [1, 3, 5])
@pytest.mark.parametrize("car_length", [2.0, 4.0, 8.0])
@pytest.mark.parametrize("density", [1.0, 50.0, 160.0, 1000.0])
def test_target_never_exceeds_capacity(n_lanes, car_length, density):
    sim = RingSim(seed=0, n_lanes=n_lanes, car_length=car_length, density=density, radius=25.0)
    pitch = car_length + sim.cfg.min_gap
    assert sim.capacity() == int(np.floor(sim.lane_len / pitch).sum())
    assert 0 <= sim.target_cars() <= sim.capacity()
    if density <= sim.jam_density:
        assert sim.target_cars() == min(round(density * sim.total_lane_length / 1000), sim.capacity())


@pytest.mark.parametrize("n_lanes", [1, 3, 5])
@pytest.mark.parametrize("density", [5.0, 60.0, 1000.0])
def test_closed_ring_keeps_exactly_target(n_lanes, density):
    sim = RingSim(seed=2, n_lanes=n_lanes, n_arms=0, density=density, radius=40.0)
    for _ in range(4):
        assert int(sim.alive.sum()) == sim.target_cars()
        sim.step(50)


def test_ring_plus_queue_matches_target_with_arms():
    sim = RingSim(seed=4, n_arms=4, exit_prob=0.5, density=80.0)
    for _ in range(300):
        sim.step()
        assert_population(sim)
    assert sim.n_exited > 0 and sim.n_entered > 0
    # Leavers are replaced through the queues, or slotted into the ring once a queue is full.
    assert sim.n_exited - sim.n_entered >= int(sim.queue.sum())


@pytest.mark.parametrize("n_arms", [0, 4])
def test_raising_density_inserts_cars(n_arms):
    sim = RingSim(seed=5, n_arms=n_arms, density=20.0)
    before = int(sim.alive.sum())
    assert sim.set(density=60.0) is False
    sim.step(10)
    assert_population(sim)
    assert int(sim.alive.sum()) > before
    assert_valid(sim)


def test_raising_density_towards_jam_reaches_target():
    sim = RingSim(seed=0, n_lanes=1, n_arms=0, radius=40.0, density=50.0, speed_std=0.0,
                  driver_noise=0.0, reaction_time=0.0)
    sim.set(density=0.8 * sim.jam_density)
    sim.step(600)
    assert int(sim.alive.sum()) == sim.target_cars()


@pytest.mark.parametrize("n_arms", [0, 4])
def test_lowering_density_removes_cars(n_arms):
    sim = RingSim(seed=6, n_arms=n_arms, density=80.0)
    sim.step(20)
    sim.set(density=20.0)
    sim.step()
    assert_population(sim)
    assert int(sim.alive.sum()) <= sim.target_cars()


def test_empty_ring_runs():
    sim = RingSim(seed=0, density=0.0)
    sim.step(20)
    assert sim.cars()["slot"].size == 0
    assert sim.brake_random() is None
    assert sim.car_near(0.0, 0.0) is None


def test_unknown_override_raises():
    with pytest.raises(KeyError):
        RingSim(not_a_param=1)


# --------------------------------------------------------------- determinism
def test_same_seed_same_run():
    a, b = RingSim(seed=123), RingSim(seed=123)
    a.step(300)
    b.step(300)
    assert same_state(a, b)
    assert (a.n_exited, a.n_entered, a.n_lane_changes, a.n_emergency) == \
        (b.n_exited, b.n_entered, b.n_lane_changes, b.n_emergency)


def test_different_seeds_differ():
    a, b = RingSim(seed=1), RingSim(seed=2)
    a.step(100)
    b.step(100)
    assert not same_state(a, b)


def test_reset_replays_the_run():
    a, b = RingSim(seed=9), RingSim(seed=9)
    a.step(150)
    a.reset()
    assert a.time == 0.0
    a.step(100)
    b.step(100)
    assert same_state(a, b)


def test_unseeded_run_can_be_replayed_from_its_seed():
    a = RingSim()
    b = RingSim(seed=a.seed)
    a.step(100)
    b.step(100)
    assert same_state(a, b)


# ----------------------------------------------------------------------- set()
@pytest.mark.parametrize("change", [{"n_lanes": 2}, {"radius": 50.0}, {"n_arms": 6},
                                    {"car_length": 2.5}, {"init_spacing": "clustered"}])
def test_set_rebuild_param_rebuilds(change):
    sim = RingSim(seed=3)
    sim.step(50)
    assert sim.set(**change) is True
    assert sim.time == 0.0
    cfg = sim.cfg
    for name, value in change.items():
        assert getattr(cfg, name) == value
    assert sim.n_lanes == cfg.n_lanes and sim.n_arms == cfg.n_arms
    np.testing.assert_allclose(sim.radii, cfg.radius + cfg.lane_width * np.arange(cfg.n_lanes))
    assert_population(sim)
    assert_valid(sim)


@pytest.mark.parametrize("change", [{"mean_speed": 8.0}, {"reaction_distance": 30.0},
                                    {"speed_std": 3.0}, {"lane_change_rate": 0.9},
                                    {"reaction_time": 1.2}, {"lane_change_motive": "random"}])
def test_set_live_param_keeps_cars(change):
    sim = RingSim(seed=3)
    sim.step(50)
    uids, t = sim.cars()["uid"].copy(), sim.time
    assert sim.set(**change) is False
    assert sim.time == t
    np.testing.assert_array_equal(sim.cars()["uid"], uids)
    for name, value in change.items():
        assert getattr(sim.cfg, name) == value
    sim.step(20)
    assert_valid(sim)


def test_set_same_value_does_not_rebuild():
    sim = RingSim(seed=3)
    sim.step(10)
    assert sim.set(n_lanes=sim.cfg.n_lanes, radius=sim.cfg.radius) is False
    assert sim.set() is False
    assert sim.time > 0


def test_set_unknown_name_raises():
    with pytest.raises(KeyError):
        RingSim(seed=0).set(lanes=2)


def test_set_lane_width_rebuilds_geometry():
    sim = RingSim(seed=0, n_lanes=3)
    sim.set(lane_width=5.0)
    np.testing.assert_allclose(sim.radii, sim.cfg.radius + 5.0 * np.arange(3))


# --------------------------------------------------------------- driver model
@pytest.mark.parametrize("mean_speed, radius, lateral_accel", [(12.0, 80.0, 4.0), (25.0, 30.0, 2.0)])
def test_lone_car_reaches_free_speed(mean_speed, radius, lateral_accel):
    sim = RingSim(seed=0, n_lanes=1, n_arms=0, radius=radius, mean_speed=mean_speed, speed_std=0.0,
                  driver_noise=0.0, lateral_accel=lateral_accel)
    sim.reset(density=1000.0 / sim.total_lane_length)  # exactly one car
    slot = sim.cars()["slot"]
    assert slot.size == 1
    sim.v[slot] = 0.0
    expected = min(mean_speed, np.sqrt(lateral_accel * radius))
    top = 0.0
    for _ in range(60):
        sim.step(10)
        top = max(top, float(sim.v[slot][0]))
    assert sim.v[slot][0] == pytest.approx(expected, rel=0.02)
    assert top <= expected * 1.001


def test_desired_speed_spread_matches_speed_std():
    sim = RingSim(seed=1, n_lanes=5, n_arms=0, radius=200.0, density=100.0, speed_std=3.0)
    v0 = sim.cars()["v0"]
    assert v0.size > 500
    assert v0.mean() == pytest.approx(sim.cfg.mean_speed, abs=0.3)
    assert v0.std() == pytest.approx(3.0, rel=0.1)
    sim.set(speed_std=0.0)
    np.testing.assert_allclose(sim.cars()["v0"], sim.cfg.mean_speed)


def test_inner_lanes_capped_by_cornering():
    sim = RingSim(seed=1, n_lanes=3, n_arms=0, radius=20.0, lateral_accel=2.0, mean_speed=20.0,
                  speed_std=0.0)
    cars = sim.cars()
    cap = np.sqrt(2.0 * sim.radii[cars["lane"]])
    np.testing.assert_allclose(cars["v0"], np.minimum(20.0, cap))


def test_reaction_distance_limits_who_drivers_notice():
    sim = RingSim(seed=0, reaction_distance=20.0)
    v0 = sim.cfg.mean_speed
    beyond, within = sim.equilibrium_speed(np.array([21.0, 19.0]), v0)
    assert beyond == pytest.approx(v0, rel=1e-6)
    assert within < 0.9 * v0
    gaps = np.linspace(0.0, 19.0, 40)
    speeds = sim.equilibrium_speed(gaps, v0)
    assert (np.diff(speeds) >= -1e-9).all()
    assert (speeds[gaps <= sim.cfg.min_gap] == 0).all()


@pytest.mark.parametrize("spacing", ["uniform", "random", "clustered"])
def test_initial_spacing_patterns(spacing):
    sim = RingSim(seed=8, n_lanes=3, n_arms=0, density=40.0, init_spacing=spacing)
    cfg = sim.cfg
    for gaps in lane_gaps(sim).values():
        assert gaps.min() >= cfg.min_gap - 1e-6
        if spacing == "uniform":
            assert np.ptp(gaps) < 1e-6
        elif spacing == "random":
            assert gaps.std() > 1.0
        else:  # one packed queue: all but the gap behind the last car are tight
            packed = gaps <= 1.05 * (cfg.car_length + cfg.min_gap) - cfg.car_length + 1e-6
            assert packed.sum() == gaps.size - 1


def test_slow_reactions_turn_smooth_flow_into_stop_and_go():
    """The model's core phenomenon: at the same density, a reaction delay alone creates
    phantom jams from small fluctuations; instant reactions damp them."""
    def settle(reaction_time: float) -> tuple[float, float]:
        sim = RingSim(seed=0, n_lanes=1, n_arms=0, density=45.0, speed_std=0.0, driver_noise=0.25,
                      reaction_time=reaction_time)
        sim.step(600)
        jammed, spread = [], []
        for _ in range(20):
            sim.step(10)
            v, v0 = sim.cars()["v"], sim.cars()["v0"]
            jammed.append(np.mean(v < 0.3 * v0))
            spread.append(v.std())
        return float(np.mean(jammed)), float(np.mean(spread))

    quick, slow = settle(0.0), settle(1.5)
    assert quick[0] < 0.01
    assert slow[0] > 0.1
    assert slow[1] > 3 * quick[1]


# --------------------------------------------------------------------- exits
@pytest.mark.parametrize("n_lanes", [1, 3])
def test_exits_only_from_outer_lane(n_lanes):
    sim = RingSim(seed=3, n_lanes=n_lanes, n_arms=4, exit_prob=1.0, density=40.0)
    departed = []
    for _ in range(400):
        before = sim.cars()
        lane_of = dict(zip(before["uid"].tolist(), before["lane"].tolist()))
        exited = sim.n_exited
        sim.step()
        gone = set(lane_of) - set(sim.cars()["uid"].tolist())
        assert len(gone) == sim.n_exited - exited
        departed += [lane_of[u] for u in gone]
    assert departed and set(departed) == {n_lanes - 1}
    assert len(sim.exit_log) == sim.n_exited
    assert all(0 <= arm < sim.n_arms and trip > 0 for _, arm, trip in sim.exit_log)


def test_more_exits_with_higher_exit_prob():
    exits = []
    for p in (0.1, 1.0):
        sim = RingSim(seed=3, exit_prob=p)
        sim.step(400)
        exits.append(sim.n_exited)
    assert 0 < exits[0] < exits[1]


def test_no_exits_when_exit_prob_zero():
    sim = RingSim(seed=3, n_arms=4, exit_prob=0.0)
    uids = sim.cars()["uid"].copy()
    sim.step(300)
    assert sim.n_exited == sim.n_missed == sim.n_entered == 0
    assert not sim.queue.any()
    np.testing.assert_array_equal(np.sort(sim.cars()["uid"]), np.sort(uids))


# ------------------------------------------------------------- perturbations
def test_brake_stops_the_car_then_it_recovers():
    sim = RingSim(seed=2, n_lanes=1, n_arms=0, density=30.0)
    sim.step(50)
    cars = sim.cars()
    slot = int(cars["slot"][np.argmax(cars["v"])])
    assert sim.v[slot] > 5.0
    sim.brake(slot, duration=6.0)
    sim.step(40)
    assert sim.v[slot] < 0.1
    cars = sim.cars()
    assert cars["braking"][cars["slot"] == slot].all()
    assert cars["braking"].sum() == 1
    sim.step(150)
    assert sim.v[slot] > 2.0
    assert not sim.cars()["braking"].any()


def test_brake_ignores_invalid_slots():
    sim = RingSim(seed=2, n_arms=0)
    dead = int(np.flatnonzero(~sim.alive)[0])
    for slot in (-1, sim.cap + 5, dead):
        sim.brake(slot)
    assert not sim.cars()["braking"].any()
    sim.step(5)


def test_car_near_finds_each_car_at_its_own_position():
    sim = RingSim(seed=2)
    sim.step(30)
    cars = sim.cars()
    for i in range(0, cars["slot"].size, 7):
        assert sim.car_near(float(cars["x"][i]), float(cars["y"][i])) == cars["slot"][i]
    assert sim.car_near(0.0, 0.0) is None  # the island in the middle


def test_brake_random_picks_a_live_car():
    sim = RingSim(seed=2)
    slot = sim.brake_random()
    assert slot is not None and sim.alive[slot]
    cars = sim.cars()
    np.testing.assert_array_equal(cars["slot"][cars["braking"]], [slot])


# -------------------------------------------------------------- lane changes
@pytest.mark.parametrize("motive", ["advantage", "random"])
def test_lane_changes_happen_one_lane_at_a_time(motive):
    sim = RingSim(seed=5, n_lanes=3, n_arms=0, density=40.0, speed_std=3.0,
                  lane_change_rate=0.5, lane_change_motive=motive)
    lane_of = dict(zip(sim.cars()["uid"].tolist(), sim.cars()["lane"].tolist()))
    last_change: dict[int, float] = {}
    for _ in range(300):
        sim.step()
        cars = sim.cars()
        for uid, lane in zip(cars["uid"].tolist(), cars["lane"].tolist()):
            if lane != lane_of[uid]:
                assert abs(lane - lane_of[uid]) == 1
                if uid in last_change:
                    assert sim.time - last_change[uid] >= LC_COOLDOWN - 1e-6
                last_change[uid] = sim.time
            lane_of[uid] = lane
    assert sim.n_lane_changes > 0
    assert len(sim.lane_change_log) == sim.n_lane_changes


def test_lane_change_rate_controls_how_often():
    counts = []
    for rate in (0.05, 1.0):
        sim = RingSim(seed=5, n_lanes=3, n_arms=0, density=40.0, lane_change_rate=rate,
                      lane_change_motive="random")
        sim.step(300)
        counts.append(sim.n_lane_changes)
    assert counts[0] < counts[1]


@pytest.mark.parametrize("n_lanes, rate, exit_prob", [(3, 0.0, 0.0), (1, 1.0, 0.5)])
def test_no_lane_changes_without_a_reason(n_lanes, rate, exit_prob):
    sim = RingSim(seed=5, n_lanes=n_lanes, n_arms=4, lane_change_rate=rate, exit_prob=exit_prob)
    lanes = dict(zip(sim.cars()["uid"].tolist(), sim.cars()["lane"].tolist()))
    sim.step(300)
    assert sim.n_lane_changes == 0
    cars = sim.cars()
    assert all(lanes[u] == l for u, l in zip(cars["uid"].tolist(), cars["lane"].tolist()) if u in lanes)


# --------------------------------------------------------------- performance
def test_thousand_cars_step_quickly():
    sim = RingSim(seed=1, n_lanes=5, radius=200.0, density=160.0)
    assert int(sim.alive.sum()) >= 1000
    start = time.perf_counter()
    sim.step(50)
    assert time.perf_counter() - start < 10.0
    assert_valid(sim)
