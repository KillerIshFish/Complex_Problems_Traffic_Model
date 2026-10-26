"""Open demand: random arrivals, queues, follow-up headway, demand imbalance and profile,
entry-delay metrics, and their viewer and experiments integration (traffic_mode="open")."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from roundabout_sim import app as viewer
from roundabout_sim import experiments
from roundabout_sim.config import PARAM_BY_NAME, SimConfig, coerce
from roundabout_sim.engine import MAX_QUEUE, PEAK_PERIOD, PEAK_SWING, RingSim, mean_demand_factor
from roundabout_sim.metrics import (LOS_LIMITS, VC_WINDOW, Recorder, entry_vc, level_of_service,
                                    snapshot)
from tests.helpers import assert_valid, click, frames, rendered_texts, widget

OPEN = dict(traffic_mode="open")
NEW_PARAMS = ("traffic_mode", "entry_rate", "demand_profile", "demand_imbalance", "max_queue", "follow_up")
NEW_COLUMNS = {"arrivals_per_min", "entries_per_min", "blocked_per_min", "mean_entry_delay", "longest_queue",
               "throughput_per_h"}


def assert_queues_consistent(sim: RingSim) -> None:
    """Every queued car has an arrival time (oldest first) and no queue is over its limit."""
    assert [len(times) for times in sim.queue_times] == sim.queue.tolist()
    for times in sim.queue_times:
        assert list(times) == sorted(times) and all(t <= sim.time for t in times)
    assert (sim.queue >= 0).all() and (sim.queue <= sim.queue_limit).all()


def assert_conserved(sim: RingSim) -> None:
    """Every arrival is accounted for: entered, still queued, blocked or given up."""
    assert sim.n_arrived == sim.n_entered + int(sim.queue.sum()) + sim.n_blocked + sim.n_gave_up
    assert len(sim.gave_up_log) == sim.n_gave_up or sim.time > 600  # (the log is pruned after 600 s)
    assert sim.arm_arrived.sum() == sim.n_arrived and sim.arm_entered.sum() == sim.n_entered


def entry_times(sim: RingSim, arm: int) -> np.ndarray:
    return np.array([when for when, a, _ in sim.entry_log if a == arm])


# ------------------------------------------------------------------- config
def test_new_parameters_are_documented_and_validated():
    cfg = SimConfig()
    assert (cfg.traffic_mode, cfg.entry_rate, cfg.demand_profile, cfg.demand_imbalance, cfg.max_queue,
            cfg.follow_up) == ("closed", 0.1, "constant", 0.0, 50, 1.0)
    assert PARAM_BY_NAME["traffic_mode"].reset and PARAM_BY_NAME["traffic_mode"].choices == ("closed", "open")
    for name in NEW_PARAMS:
        spec = PARAM_BY_NAME[name]
        assert spec.help and spec.label and spec.group in ("Traffic", "Demand")
        assert spec.reset == (name == "traffic_mode")
    # Parameters that only act in open mode say so (and the viewer marks them in closed mode).
    assert {n for n in NEW_PARAMS if PARAM_BY_NAME[n].applies_in == "open"} == {
        "entry_rate", "demand_profile", "demand_imbalance", "max_queue"}
    assert "veh/h" in PARAM_BY_NAME["entry_rate"].help and PARAM_BY_NAME["entry_rate"].label == "Entrance chance"
    assert coerce("max_queue", "12.0") == 12 and coerce("traffic_mode", "open") == "open"
    with pytest.raises(ValueError):
        coerce("demand_profile", "rush")


@pytest.mark.parametrize("bad", [dict(entry_rate=-0.1), dict(entry_rate=float("nan")), dict(max_queue=0),
                                 dict(max_queue=2.5), dict(follow_up=0.0), dict(demand_imbalance=1.5),
                                 dict(demand_imbalance=-0.1), dict(traffic_mode="bogus"),
                                 dict(demand_profile="rush")])
def test_invalid_demand_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        SimConfig(**bad)
    sim = RingSim(seed=0)
    before = sim.cfg
    with pytest.raises(ValueError):
        sim.set(**bad)
    assert sim.cfg == before


# ------------------------------------------------------------------ arrivals
def test_arrival_rate_matches_the_entrance_chance():
    sim = RingSim(seed=3, n_lanes=1, radius=40.0, density=10.0, entry_rate=0.3, exit_prob=0.5, max_queue=10**6,
                  **OPEN)
    sim.step(3000)  # 300 s
    expected = 0.3 * sim.n_arms * sim.time
    assert sim.n_arrived == pytest.approx(expected, abs=4 * np.sqrt(expected))  # Poisson: 4 sigma
    assert sim.n_blocked == 0 and len(sim.arrival_log) == sim.n_arrived
    assert sim.n_arrived == sim.n_entered + int(sim.queue.sum())  # nothing lost without blocking
    assert_queues_consistent(sim)


@pytest.mark.parametrize("n_arms", range(1, 9))
@pytest.mark.parametrize("imbalance", [0.0, 0.5, 1.0])
def test_demand_weights_have_mean_one_and_favour_the_main_road(n_arms, imbalance):
    w = RingSim(seed=0, n_arms=n_arms, demand_imbalance=imbalance, density=5.0).demand_weight
    assert w.shape == (n_arms,) and w.mean() == pytest.approx(1.0) and (w >= -1e-12).all()
    if imbalance == 0 or n_arms <= 2:
        np.testing.assert_allclose(w, 1.0)  # with one or two arms every arm is on the main road
    else:
        assert w[0] == pytest.approx(w.max()) and w[0] > 1.0
        if n_arms % 2 == 0:
            assert w[n_arms // 2] == pytest.approx(w[0])
    if imbalance == 1.0 and n_arms % 4 == 0:
        assert w[n_arms // 4] == pytest.approx(0.0, abs=1e-12)  # side road at right angles: no demand


def test_imbalanced_arrivals_follow_the_weights():
    sim = RingSim(seed=1, n_lanes=1, radius=40.0, density=10.0, entry_rate=0.4, demand_imbalance=1.0,
                  exit_prob=0.5, max_queue=10**6, **OPEN)
    sim.step(2000)
    per_arm = np.bincount([arm for _, arm in sim.arrival_log], minlength=4)
    assert per_arm[1] == per_arm[3] == 0  # the side roads at right angles get nothing
    expected = 0.4 * 2 * sim.time  # the main road carries all of it: twice the mean per arm
    assert per_arm[0] == pytest.approx(expected, abs=4 * np.sqrt(expected))
    assert per_arm[0] + per_arm[2] == sim.n_arrived


def test_peak_profile_rises_and_falls_with_mean_one():
    sim = RingSim(seed=0, demand_profile="peak", **OPEN)
    t = np.linspace(0, PEAK_PERIOD, 6000, endpoint=False)
    f = sim.demand_factor(t)
    assert f.mean() == pytest.approx(1.0, abs=1e-9)
    assert f.min() == pytest.approx(1 - PEAK_SWING) and f.max() == pytest.approx(1 + PEAK_SWING, abs=1e-6)
    assert sim.demand_factor(0.0) == pytest.approx(1 - PEAK_SWING)  # starts low...
    assert sim.demand_factor(PEAK_PERIOD / 2) == pytest.approx(1 + PEAK_SWING)  # ...peaks half a period on
    assert np.all(np.abs(np.diff(f)) < 1e-3)  # smooth
    np.testing.assert_allclose(sim.arrival_rates(), sim.cfg.entry_rate * (1 - PEAK_SWING))
    flat = RingSim(seed=0, **OPEN)
    assert flat.demand_factor() == 1.0 and np.all(flat.demand_factor(t) == 1.0)


def test_mean_demand_factor_over_a_window():
    sim = RingSim(seed=0, demand_profile="peak", **OPEN)
    for t0, t1 in ((120.0, 420.0), (300.0, 600.0), (0.0, 1200.0), (50.0, 100.0), (37.0, 637.0)):
        grid = np.linspace(t0, t1, 200001)
        assert mean_demand_factor("peak", t0, t1) == pytest.approx(sim.demand_factor(grid).mean(), abs=1e-4)
    assert mean_demand_factor("peak", 120.0, 420.0) == pytest.approx(1.3027, abs=1e-4)  # the sweep's default window
    assert mean_demand_factor("peak", 300.0, 600.0) == pytest.approx(1.0)  # run's default: the second half
    assert mean_demand_factor("constant", 120.0, 420.0) == 1.0


def test_peak_profile_changes_the_arrival_rate_over_the_cycle():
    sim = RingSim(seed=2, n_lanes=1, radius=40.0, density=10.0, entry_rate=0.5, demand_profile="peak",
                  exit_prob=0.5, max_queue=10**6, **OPEN)
    sim.step(int(PEAK_PERIOD / sim.cfg.dt))
    times = np.array([t for t, _ in sim.arrival_log])
    low = ((times < PEAK_PERIOD / 8) | (times >= 7 * PEAK_PERIOD / 8)).sum()  # around the troughs
    high = ((times >= 3 * PEAK_PERIOD / 8) & (times < 5 * PEAK_PERIOD / 8)).sum()  # around the peak
    assert high > 2 * low
    expected = 0.5 * sim.n_arms * PEAK_PERIOD  # the mean over a cycle is the entrance chance
    assert sim.n_arrived == pytest.approx(expected, abs=4 * np.sqrt(expected))


# ------------------------------------------------------------ car numbers
def test_car_count_varies_in_open_mode_and_is_held_in_closed_mode():
    ring = dict(n_lanes=1, radius=60.0)  # (a small ring: quick)
    closed, open_ = RingSim(seed=4, exit_prob=0.3, **ring), RingSim(seed=4, exit_prob=0.3, entry_rate=0.25, **ring,
                                                                    **OPEN)
    assert int(open_.alive.sum()) == open_.target_cars()  # Density sets the start
    counts = []
    for _ in range(30):
        closed.step(50)
        open_.step(50)
        assert int(closed.alive.sum()) + int(closed.queue.sum()) == closed.target_cars()
        counts.append(int(open_.alive.sum()))
        assert_queues_consistent(open_)
        assert_queues_consistent(closed)
    assert len(set(counts)) > 5 and open_.n_exited > 0 and open_.n_entered > 0
    assert open_.n_exited - open_.n_entered == open_.target_cars() - counts[-1]  # leavers are not replaced


def test_entrance_chance_does_nothing_in_closed_mode():
    """The demand stream is separate: closed runs are identical whatever the open-mode settings."""
    a = RingSim(seed=5, exit_prob=0.3)
    b = RingSim(seed=5, exit_prob=0.3, entry_rate=0.5, demand_profile="peak", demand_imbalance=1.0, max_queue=3)
    a.step(400)
    b.step(400)
    np.testing.assert_array_equal(a.theta[a.alive], b.theta[b.alive])
    np.testing.assert_array_equal(a.queue, b.queue)


def test_open_mode_without_arrivals_or_exits_matches_the_closed_ring():
    """Open mode adds no draws to the main random stream: with nothing arriving or leaving it is
    the same run as closed mode."""
    closed = RingSim(seed=6, exit_prob=0.0)
    open_ = RingSim(seed=6, exit_prob=0.0, entry_rate=0.0, **OPEN)
    closed.step(300)
    open_.step(300)
    np.testing.assert_array_equal(closed.theta[closed.alive], open_.theta[open_.alive])
    np.testing.assert_array_equal(closed.v[closed.alive], open_.v[open_.alive])


def test_same_seed_same_open_run():
    a, b = (RingSim(seed=8, entry_rate=0.3, demand_profile="peak", **OPEN) for _ in range(2))
    a.step(500)
    b.step(500)
    np.testing.assert_array_equal(a.theta[a.alive], b.theta[b.alive])
    assert (a.n_arrived, a.n_entered, a.n_blocked) == (b.n_arrived, b.n_entered, b.n_blocked)
    assert [list(q) for q in a.queue_times] == [list(q) for q in b.queue_times]


# ------------------------------------------------------------------- queues
def test_full_queues_turn_arrivals_away():
    sim = RingSim(seed=1, entry_rate=0.5, max_queue=3, follow_up=5.0, **OPEN)
    for _ in range(40):
        sim.step(25)
        assert_queues_consistent(sim)
    assert sim.n_blocked > 0 and (sim.queue == 3).any()
    assert sim.n_arrived == sim.n_entered + int(sim.queue.sum()) + sim.n_blocked
    assert_conserved(sim)
    assert len(sim.blocked_log) == sim.n_blocked
    row = snapshot(sim)
    assert row["blocked_per_min"] > 0 and row["longest_queue"] == 3
    assert row["queued"] == int(sim.queue.sum())


def test_queue_waits_are_first_come_first_served():
    sim = RingSim(seed=2, entry_rate=0.3, follow_up=3.0, max_queue=10**6, **OPEN)
    sim.step(2000)
    assert sim.n_entered > 20
    for arm in range(sim.n_arms):
        arrived = [when for when, a in sim.arrival_log if a == arm]
        entries = [(when, wait) for when, a, wait in sim.entry_log if a == arm]
        for k, (when, wait) in enumerate(entries):  # the k-th car in is the k-th to arrive
            assert wait == when - arrived[k] and wait >= 0
        assert list(sim.queue_times[arm]) == arrived[len(entries):]  # the rest are still waiting
    waits = [wait for *_, wait in sim.entry_log]
    assert sim.entry_wait_total == pytest.approx(sum(waits))


@pytest.mark.parametrize("follow_up", [1.0, 3.0])
def test_follow_up_time_spaces_entries_from_one_entrance(follow_up):
    sim = RingSim(seed=3, n_lanes=1, density=10.0, entry_rate=0.5, follow_up=follow_up, max_queue=10**6,
                  exit_prob=0.5, **OPEN)
    sim.step(1500)
    gaps = np.concatenate([np.diff(entry_times(sim, arm)) for arm in range(sim.n_arms)])
    assert gaps.size > 20 and gaps.min() >= follow_up - 1e-6
    if follow_up >= 2.0:  # (at 1 s the room the previous entrant needs binds first: about 1.2 s)
        assert np.quantile(gaps, 0.2) < follow_up + 0.5  # a queue empties at about one car per follow-up


@pytest.mark.parametrize("follow_up, dt", [(1.0, 0.1), (1.2, 0.1), (2.5, 0.1), (3.0, 0.1), (4.7, 0.1), (5.0, 0.1),
                                           (2.5, 0.05), (1.25, 0.1)])
def test_follow_up_is_one_step_past_the_setting_for_every_value(follow_up, dt):
    """The next car goes on the first step after follow_up has passed, for every value alike
    (a float countdown used to give follow_up + dt for some values and exactly follow_up for others)."""
    sim = RingSim(seed=1, n_lanes=1, density=10.0, entry_rate=2.0, follow_up=follow_up, dt=dt, max_queue=10**6,
                  exit_prob=0.5, **OPEN)
    expected = (np.floor(follow_up / dt + 1e-9) + 1) * dt
    assert sim.follow_up_steps() * dt == pytest.approx(expected)
    sim.step(int(round(120 / dt)))
    gaps = np.concatenate([np.diff(entry_times(sim, arm)) for arm in range(sim.n_arms)])
    assert gaps.size > 20 and gaps.min() == pytest.approx(expected, abs=1e-6)  # saturated queues reach it


def test_every_arrival_is_accounted_for_after_live_cuts():
    """Cars dropped from a queue when Max queue (open) or Density (closed) is lowered live give up:
    counted, so arrived = entered + queued + blocked + given up still holds."""
    sim = RingSim(seed=7, entry_rate=0.4, max_queue=40, n_lanes=1, radius=40.0, **OPEN)
    sim.step(3000)
    assert (sim.queue == 40).all() and sim.n_gave_up == 0
    assert_conserved(sim)
    sim.set(max_queue=5)
    assert (sim.queue == 5).all() and sim.n_gave_up == 140 and len(sim.gave_up_log) == 140
    assert {arm for _, arm in sim.gave_up_log} == {0, 1, 2, 3}
    assert_conserved(sim)
    sim.step(500)
    assert_conserved(sim)

    closed = RingSim(seed=3, exit_prob=0.4, density=70)
    closed.step(400)
    assert closed.queue.sum() > 0
    assert_conserved(closed)
    closed.set(density=10.0)  # the density control drops the queued cars first
    closed.step()
    assert closed.n_gave_up > 0 and closed.queue.sum() == 0 and closed.n_blocked == 0
    assert_conserved(closed)


def test_closed_mode_queues_keep_their_arrival_times():
    sim = RingSim(seed=4, exit_prob=0.4, density=70)
    for _ in range(20):
        sim.step(20)
        assert_queues_consistent(sim)
    assert sim.queue_limit == MAX_QUEUE and sim.n_arrived > 0 and sim.n_blocked == 0
    sim.set(density=20.0)  # the density control drops queued cars too
    sim.step()
    assert_queues_consistent(sim)
    assert all(wait >= 0 for *_, wait in sim.entry_log)
    assert np.isfinite(snapshot(sim, window=60)["mean_entry_delay"])


def test_open_mode_on_a_closed_ring():
    sim = RingSim(seed=0, n_arms=0, entry_rate=0.5, **OPEN)
    n = int(sim.alive.sum())
    sim.step(200)
    assert int(sim.alive.sum()) == n and sim.n_arrived == sim.n_entered == 0
    assert sim.demand_weight.size == 0 and sim.arrival_rates().size == 0
    row = snapshot(sim)
    assert row["arrivals_per_min"] == 0 and np.isnan(row["mean_entry_delay"]) and row["longest_queue"] == 0
    sim.set(density=80.0)  # a live density change sets the ring once
    assert int(sim.alive.sum()) == sim.target_cars() > n
    sim.step(50)
    assert int(sim.alive.sum()) == sim.target_cars()


# ------------------------------------------------------------- live changes
def test_every_new_parameter_can_be_set_live():
    sim = RingSim(seed=5, entry_rate=0.4, follow_up=2.0, **OPEN)
    sim.step(600)
    gen = sim.generation
    for change in (dict(entry_rate=0.2), dict(demand_profile="peak"), dict(demand_imbalance=0.8),
                   dict(follow_up=3.0), dict(max_queue=100)):
        assert sim.set(**change) is False
        sim.step(50)
        assert_queues_consistent(sim)
    assert sim.generation == gen
    np.testing.assert_allclose(sim.arrival_rates(), 0.2 * sim.demand_factor() * sim.demand_weight)
    assert sim.demand_weight[0] > 1.5  # recomputed for the new imbalance

    sim.queue[0] = 0  # make one queue long, then shorten the limit below it
    sim.queue_times[0].clear()
    for t in range(20):
        sim.queue[0] += 1
        sim.queue_times[0].append(float(t))
    assert sim.set(max_queue=5) is False
    assert sim.queue[0] == 5 and list(sim.queue_times[0]) == [0.0, 1.0, 2.0, 3.0, 4.0]  # the latest give up
    assert_queues_consistent(sim)

    entered = sim.n_entered
    while sim.n_entered == entered and sim.time < 600:
        sim.step()
    arm = sim.entry_log[-1][1]
    # The live follow_up of 3 s is in force: 31 steps of 0.1 s, one of them already counted down.
    assert sim.follow_up_steps() == 31 and sim.entry_cool[arm] == 30

    assert sim.set(traffic_mode="closed") is True  # rebuilds
    assert sim.generation == gen + 1 and sim.time == 0.0 and not sim.open_demand
    assert sim.set(traffic_mode="open") is True and sim.open_demand


def test_live_density_change_in_open_mode_happens_once():
    sim = RingSim(seed=6, entry_rate=0.0, exit_prob=0.5, **OPEN)
    sim.step(100)
    assert sim.set(density=80.0) is False
    assert int(sim.alive.sum()) == sim.target_cars()  # set at once...
    sim.step(600)
    assert int(sim.alive.sum()) < sim.target_cars()  # ...then leavers are not replaced
    sim.set(density=10.0)
    assert int(sim.alive.sum()) == sim.target_cars()


def test_live_density_raise_in_open_mode_keeps_the_trucks():
    """Trucks that fit no gap are placed by respacing the lanes: open mode never retries, so
    leaving them out would lower the truck share for good."""
    shares = []
    for seed in range(8):
        sim = RingSim(seed=seed, entry_rate=0.0, exit_prob=0.0, truck_share=0.3, truck_length=12.0, n_lanes=2,
                      radius=40.0, density=5.0, **OPEN)
        sim.step(20)
        sim.set(density=float(round(0.8 * sim.jam_density)))
        assert int(sim.alive.sum()) == sim.target_cars() and not sim._pending  # nobody left out
        shares.append(sim.truck[sim.alive].mean())
        for _ in range(25):
            sim.step(4)
            idx = np.flatnonzero(sim.alive)
            _, gap = sim._index_lanes(idx)
            assert (gap[idx] > 0).all() and (sim.v[idx] >= 0).all()
    assert np.mean(shares) > 0.28  # the 0.3 share (it was about 0.23 with the trucks left out)


def test_seed_change_reseeds_the_demand_stream():
    a, b = RingSim(seed=1, entry_rate=0.4, **OPEN), RingSim(seed=2, entry_rate=0.4, **OPEN)
    a.set(seed=9)
    b.set(seed=9)
    assert a.rng_demand.poisson(5.0, 20).tolist() == b.rng_demand.poisson(5.0, 20).tolist()


# ------------------------------------------------------------------ metrics
@pytest.mark.parametrize("delay, letter", [(0.0, "A"), (10.0, "A"), (10.01, "B"), (15.0, "B"), (25.0, "C"),
                                           (30.0, "D"), (35.0, "D"), (49.9, "E"), (50.0, "E"), (50.1, "F"),
                                           (500.0, "F"), (float("nan"), "-"), (None, "-")])
def test_level_of_service_thresholds(delay, letter):
    assert level_of_service(delay) == letter


def test_los_limits_are_the_hcm_ones():
    assert LOS_LIMITS == (("A", 10.0), ("B", 15.0), ("C", 25.0), ("D", 35.0), ("E", 50.0))


@pytest.mark.parametrize("delay, vc, letter", [(16.2, 6.3, "F"), (2.0, 1.01, "F"), (16.2, 1.0, "C"), (16.2, 0.4, "C"),
                                               (16.2, float("nan"), "C"), (16.2, None, "C"), (60.0, 0.5, "F"),
                                               (float("nan"), 3.0, "-")])
def test_level_of_service_is_f_above_capacity(delay, vc, letter):
    assert level_of_service(delay, vc) == letter


def test_entry_vc_is_demand_over_the_entry_rate_while_queued():
    # 10 arrivals in 60 s, 5 let in during 30 s with a queue: 1/6 per s against 1/6 per s.
    assert entry_vc([10, 2], [5, 2], [30.0, 2.0], 60.0) == pytest.approx(1.0)
    assert entry_vc([10, 30], [5, 10], [30.0, 50.0], 60.0) == pytest.approx(2.5)  # the busiest entrance counts
    assert entry_vc([4, 3], [0, 3], [60.0, 6.0], 60.0) == pytest.approx(0.1)  # one that let nobody in is left out
    assert np.isnan(entry_vc([4], [0], [10.0], 60.0)) and np.isnan(entry_vc([], [], [], 60.0))


def test_oversaturated_entrances_read_f_however_short_the_wait():
    """A short Max queue caps the wait of the cars that get in: the level of service must still be F."""
    sim, rec, summary, _ = experiments._simulate(SimConfig(seed=0, entry_rate=0.5, max_queue=2, **OPEN), 200.0,
                                                 0.5, 100.0)
    assert summary["blocked_per_min"] > 2 * summary["entries_per_min"]  # most arrivals turned away
    assert level_of_service(summary["mean_entry_delay"]) in "ABCD"  # the wait alone looks fine...
    assert summary["entry_vc"] > 2 and summary["entry_los"] == "F"  # ...but demand is far over capacity
    row = rec.rows[-1]
    assert row["entry_vc"] > 2 and level_of_service(row["mean_entry_delay"], row["entry_vc"]) == "F"
    light = experiments._simulate(SimConfig(seed=0, entry_rate=0.05, **OPEN), 200.0, 0.5, 100.0)[2]
    assert light["entry_vc"] < 0.5 and light["entry_los"] == "A"
    closed = experiments._simulate(SimConfig(seed=0), 60.0, 0.5, 30.0)[2]
    assert np.isnan(closed["entry_vc"]) and closed["entry_los"] in "ABCDEF-"  # closed mode: by the delay only


def test_rolling_entry_vc_covers_at_least_five_minutes():
    sim = RingSim(seed=2, entry_rate=0.2, n_lanes=1, radius=40.0, **OPEN)  # (a small ring: a quicker 400 s)
    sim.step(4000)
    t = sim.time
    when, arrived, entered, busy = next(e for e in reversed(sim.entrance_log) if e[0] <= t - VC_WINDOW + 1e-9)
    assert t - VC_WINDOW - 1.0 <= when <= t - VC_WINDOW
    expected = entry_vc(sim.arm_arrived - arrived, sim.arm_entered - entered, sim.queue_busy - busy, t - when)
    assert snapshot(sim)["entry_vc"] == pytest.approx(expected) and np.isfinite(expected)
    assert (sim.queue_busy <= t + 1e-9).all() and (sim.queue_busy > 0).all()
    assert np.isnan(snapshot(RingSim(seed=0))["entry_vc"])  # closed mode: no demand of its own


def test_an_empty_ring_does_not_count_as_standing_cars():
    """Open mode at low demand empties the ring now and then: those moments have no mean speed,
    so the summary's mean speed is that of the cars on the ring (with no trucks, the car speed)."""
    sim, rec, summary, _ = experiments._simulate(
        SimConfig(seed=0, entry_rate=0.02, exit_prob=0.5, radius=20.0, density=10.0, **OPEN), 200.0, 0.5, 20.0)
    late = rec.frame().query("time >= 20")
    empty = late["cars"] == 0
    assert 0.1 < empty.mean() < 0.9  # the ring empties now and then
    assert late.loc[empty, ["mean_speed", "jam_fraction", "order_param"]].isna().all().all()
    assert summary["measured_mean_speed"] == pytest.approx(late.loc[~empty, "mean_speed"].mean())
    assert summary["measured_mean_speed"] == pytest.approx(summary["car_speed"])
    assert summary["measured_mean_speed"] > 5.0  # not dragged towards 0 by the empty moments


def test_snapshot_and_recorder_carry_the_entrance_metrics():
    sim = RingSim(seed=7, entry_rate=0.3, **OPEN)
    rec = Recorder(sim, interval=0.5)
    for _ in range(1200):
        sim.step()
        rec.update()
    row = rec.rows[-1]
    assert NEW_COLUMNS | {"demand_per_min"} <= set(row)
    assert all(isinstance(v, (int, float, np.integer, np.floating)) for v in row.values())  # numeric rows
    t = sim.time
    recent = [e for e in sim.entry_log if e[0] >= t - 30.0]
    assert row["entries_per_min"] == pytest.approx(2.0 * len(recent))
    assert row["mean_entry_delay"] == pytest.approx(np.mean([w for *_, w in recent]))
    assert row["arrivals_per_min"] == pytest.approx(2.0 * sum(1 for e in sim.arrival_log if e[0] >= t - 30.0))
    assert row["throughput_per_h"] == pytest.approx(60.0 * row["exits_per_min"])
    assert row["demand_per_min"] == pytest.approx(60 * 0.3 * sim.n_arms)
    assert np.isnan(snapshot(RingSim(seed=0))["demand_per_min"])  # closed mode: no set demand


# ------------------------------------------------------------------- viewer
def test_viewer_switches_to_open_mode_with_its_chart_and_readouts(make_app):
    app = make_app()
    frames(app, 3)
    assert [spec for spec, *_ in app.chart_cards] == list(viewer.CHARTS)
    click(app, widget(app, "tab", "Demand").rect.center)
    assert app.group == "Demand"
    rate = widget(app, "slider", "entry_rate")
    assert app.inactive(rate.spec) == "open mode only"  # marked as doing nothing in closed mode
    mode = widget(app, "choice", "traffic_mode")
    click(app, mode.options[mode.spec.choices.index("open")].center)
    assert app.sim.open_demand and app.sim.generation == 1
    assert app.inactive(widget(app, "slider", "entry_rate").spec) == ""
    app.steps_per_frame = 10
    frames(app, 40)
    assert [spec for spec, *_ in app.chart_cards] == [*viewer.CHARTS, *viewer.OPEN_CHARTS]
    labels = {text.split(":")[0] for _, text in app.stat_hits}
    assert {"Arrivals", "Entries", "Blocked", "Entry delay", "Cars on ring", "Queued to enter"} <= labels
    stats = {st.label: st for col in app._stat_columns(app._latest_row()) for st in col}
    assert "varies" in stats["Cars on ring"].help_text() and "LOS" in stats["Entry delay"].value
    for spec, card, plot, *_ in app.chart_cards:  # every chart, the new one too, has its hover help
        click(app, plot.center)
        app.draw()
        assert app._status_text() == spec.help
    app.paused = True
    app.draw()
    assert viewer.DELAY_HELP in {text for _, text in app.chart_hits}
    rect = next(rect for rect, text in app.chart_hits if text == viewer.DELAY_HELP)
    app.mouse = (rect.right - 4, rect.centery)
    assert app._status_text() == viewer.DELAY_HELP
    for spec, card, plot, *_ in app.chart_cards:
        assert app.charts_rect.contains(card) and card.contains(plot) and plot.h >= 20
    # Back to closed mode: the extra chart goes again.
    click(app, mode.options[0].center)
    frames(app, 2)
    assert not app.sim.open_demand and len(app.chart_cards) == len(viewer.CHARTS)


def test_entrance_controls_are_marked_on_a_ring_without_entrances(make_app, monkeypatch):
    """With n_arms = 0 (the ring-road experiments) nobody enters or leaves: the entrance controls
    say so, as the demand settings say "open mode only" in closed mode."""
    app = make_app(n_arms=0)
    frames(app, 2)
    for name in ("exit_prob", "entry_gap", "follow_up", "turn_speed"):
        assert app.inactive(viewer.PARAM_BY_NAME[name]) == "no entrances", name
    assert app.inactive(viewer.PARAM_BY_NAME["entry_rate"]) == "open mode only"  # (that comes first)
    assert app.inactive(viewer.PARAM_BY_NAME["density"]) == ""
    click(app, widget(app, "tab", "Traffic").rect.center)
    rendered = []
    blit = app.fonts.blit
    monkeypatch.setattr(app.fonts, "blit", lambda scr, text, *a, **k: rendered.append(text) or blit(scr, text, *a, **k))
    app.draw()
    assert rendered.count("no entrances") >= 3
    app.apply("traffic_mode", "open")
    assert app.inactive(viewer.PARAM_BY_NAME["entry_rate"]) == "no entrances"
    app.apply("n_arms", 4.0)
    assert app.inactive(viewer.PARAM_BY_NAME["entry_rate"]) == app.inactive(viewer.PARAM_BY_NAME["exit_prob"]) == ""


def test_viewer_reads_out_an_empty_ring_without_nan(make_app):
    app = make_app(entry_rate=0.0, exit_prob=1.0, density=5.0, radius=20.0, **OPEN)
    app.steps_per_frame = 20
    while app.sim.alive.any() and app.sim.time < 120:
        frames(app, 1)
    assert not app.sim.alive.any()
    frames(app, 2)  # the charts take the NaN samples as a gap
    stats = {st.label: st.value for col in app._stat_columns(app._latest_row()) for st in col}
    assert not any("nan" in value for value in stats.values()), stats
    assert stats["Mean speed"] == stats["Jammed cars"] == stats["Clustering"] == "–"


@pytest.mark.parametrize("size", [(1100, 700), (1920, 1080)])
def test_open_mode_viewer_fits_and_draws_long_queues(make_app, size, monkeypatch):
    app = make_app(size=size, entry_rate=0.5, max_queue=12, follow_up=5.0, **OPEN)
    app.steps_per_frame = 20
    frames(app, 15)
    assert (app.sim.queue == 12).any()  # full queues, longer than the stubs
    rendered = []
    render = app.fonts.render
    monkeypatch.setattr(app.fonts, "render", lambda text, *a, **k: rendered.append(text) or render(text, *a, **k))
    app.draw()
    assert "12 full" in rendered  # long queues are labelled with their length
    screen = app.screen.get_rect()
    for r in (app.view_rect, app.stats_rect, app.charts_rect, app.panel_rect):
        assert screen.contains(r)
    tabs = [w.rect for w in app.panel_widgets if w.kind == "tab"]
    assert len(tabs) == len(viewer.TABS) and all(app.panel_rect.contains(t) for t in tabs)
    for w in (w for w in app.panel_widgets if w.kind == "tab"):  # every tab label fits its tab
        assert app.fonts.width(w.name, "small") < w.rect.w
    for text in viewer.help_texts():  # the help box still fits the longest help
        assert len(app.fonts.wrap(text, app.status_rect.w)) <= app.status_lines, text[:60]


@pytest.mark.parametrize("size", [(1100, 700), (1920, 1080)])
def test_view_header_gives_the_mean_demand_when_entrances_differ(make_app, size, monkeypatch):
    app = make_app(size=size, entry_rate=0.2, demand_imbalance=1.0, **OPEN)
    frames(app, 2)
    header = next(t for t in rendered_texts(app, monkeypatch) if t.startswith("3 lanes"))
    assert "each" not in header and ("720 veh/h mean" in header or "720 veh/h per entrance on average" in header)
    assert app.fonts.width(header, "small") <= app.view_rect.w - 2 * round(10 * app.s)
    if size[0] >= 1920:
        assert header.endswith("open: 720 veh/h per entrance on average (main road 1440)")
    app.sim.set(demand_imbalance=0.0, demand_profile="peak")
    assert "720 veh/h per entrance on average (peak profile)" in app._demand_texts()[0]
    app.sim.set(demand_profile="constant")
    header = next(t for t in rendered_texts(app, monkeypatch) if t.startswith("3 lanes"))
    assert header.endswith("720 veh/h at each entrance") or header.endswith("720 veh/h each")
    app.paused = True  # the header makes room for PAUSED
    header = next(t for t in rendered_texts(app, monkeypatch) if t.startswith("3 lanes"))
    room = app.view_rect.w - 2 * round(10 * app.s) - app.fonts.width("PAUSED", "base") - round(10 * app.s)
    assert app.fonts.width(header, "small") <= room


def test_queue_tags_have_hover_help(make_app):
    app = make_app(size=(1100, 700), entry_rate=0.5, max_queue=12, follow_up=5.0, **OPEN)
    app.steps_per_frame = 20
    frames(app, 15)
    assert (app.sim.queue == 12).any() and app.ring.tag_rects
    for rect in app.ring.tag_rects:
        app.mouse = rect.center
        assert app._status_text() == viewer.QUEUE_TAG_HELP
    app.mouse = app.view_rect.center
    assert app._status_text() == viewer.VIEW_HELP
    assert viewer.QUEUE_TAG_HELP in viewer.help_texts() and "Max queue" in viewer.QUEUE_TAG_HELP
    assert "header" in viewer.VIEW_HELP and "queue" in viewer.VIEW_HELP.lower()


@pytest.mark.parametrize("size", [(1100, 700), (1920, 1080)])
def test_short_queues_get_a_count_when_the_cars_are_too_small_to_see(make_app, size, monkeypatch):
    app = make_app(size=size, n_lanes=5, radius=300.0, car_length=2.0, min_gap=1.0, n_arms=8, **OPEN)
    frames(app, 1)
    pitch_px = (app.sim.cfg.car_length + app.sim.cfg.min_gap) * app.ring.k
    assert pitch_px < viewer.QUEUE_MIN_PX  # each queued car would be a speck
    sim = app.sim
    for arm, q in enumerate([0, 3, 0, 13, 1, 0, 0, 7]):  # short queues that fit on the stub
        sim.queue[arm] = q
        sim.queue_times[arm] = type(sim.queue_times[arm])([0.0] * q)
    rendered = rendered_texts(app, monkeypatch)
    assert {"1", "3", "7", "13"} <= set(rendered) and len(app.ring.tag_rects) == 4
    for rect in app.ring.tag_rects:
        assert app.view_rect.contains(rect)
    # On the default ring the cars are big enough to count: a queue that fits gets no tag.
    small = make_app(size=size, **OPEN)
    frames(small, 1)
    small.sim.queue[:] = [1, 0, 0, 0]
    small.sim.queue_times[0].append(0.0)
    small.draw()
    assert (small.sim.cfg.car_length + small.sim.cfg.min_gap) * small.ring.k >= viewer.QUEUE_MIN_PX
    assert small.ring.tag_rects == []


def test_viewer_command_line_takes_the_demand_flags():
    args = viewer.parse_args(["--traffic-mode", "open", "--entry-rate", "0.25", "--max_queue", "20",
                              "--demand-profile", "peak", "--follow-up", "2.8"])
    cfg = viewer.config_from_args(args)
    assert (cfg.traffic_mode, cfg.entry_rate, cfg.max_queue, cfg.demand_profile, cfg.follow_up) == (
        "open", 0.25, 20, "peak", 2.8)
    assert isinstance(cfg.max_queue, int)


# -------------------------------------------------------------- experiments
def test_experiments_run_in_open_mode_writes_the_entrance_columns(tmp_path, capsys, no_figures):
    out = tmp_path / "open"
    experiments.main(["run", "--duration", "30", "--seed", "2", "--traffic-mode", "open", "--entry-rate", "0.3",
                      "--baseline", "--out", str(out)])
    printed = capsys.readouterr().out
    assert "--baseline skipped: in open mode" in printed
    for text in ("Arrivals at the entrances", "Blocked arrivals", "Mean entry delay", "LOS", "Throughput",
                 "open demand 0.3/s per entrance (1080 veh/h", "Demand / capacity (v/c)",
                 "skipped  (not meaningful in open mode; columns left empty)"):
        assert text in printed, text
    # Nothing contradicts the skipped baseline.
    assert "nothing to switch off" not in printed and "see --baseline" not in printed
    summary = pd.read_csv(f"{out}_summary.csv").iloc[0]
    assert NEW_COLUMNS | {"peak_queue", "demand_per_min", "entry_vc", "entry_los"} <= set(summary.index)
    assert summary["entry_los"] in tuple("ABCDEF") and summary["entry_los"] == level_of_service(
        summary["mean_entry_delay"], summary["entry_vc"])
    assert summary["demand_per_min"] == pytest.approx(60 * 0.3 * 4)
    assert summary["traffic_mode"] == "open" and summary["entry_rate"] == 0.3
    assert summary["arrivals_per_min"] > 0 and summary["entries_per_min"] > 0
    assert summary["throughput_per_h"] == pytest.approx(60 * summary["exits_per_min"])
    assert summary["peak_queue"] >= summary["longest_queue"]
    ts = pd.read_csv(f"{out}_timeseries.csv")
    assert NEW_COLUMNS | {"demand_per_min"} <= set(ts.columns)
    assert ts["cars"].nunique() > 1  # the number on the ring varies


def test_summary_entry_delay_counts_exactly_the_cars_entering_in_the_window():
    sim = RingSim(seed=3, entry_rate=0.3, **OPEN)
    rec = Recorder(sim, 0.5)
    experiments.advance(sim, rec, 30.0)
    start, done = experiments._counters(sim), len(sim.entry_log)
    experiments.advance(sim, rec, 30.0)
    row = experiments.summarise(sim, rec, 30.0, start)
    waits = [w for *_, w in list(sim.entry_log)[done:]]
    assert row["mean_entry_delay"] == pytest.approx(np.mean(waits))
    assert row["entries_per_min"] == pytest.approx(len(waits) * 2.0)


def test_open_mode_sweep_prints_delay_and_level_of_service(tmp_path, capsys, no_figures):
    out = tmp_path / "sweep"
    experiments.main(["sweep", "--x", "entry_rate=0.1,0.3", "--set", "traffic_mode=open", "--seeds", "1",
                      "--warmup", "5", "--measure", "20", "--workers", "1", "--out", str(out)])
    printed = capsys.readouterr().out
    assert "delay_s" in printed and "LOS" in printed and "blocked/min" in printed and "queued" in printed
    assert "v/c" in printed and "demand/min" in printed
    df = pd.read_csv(f"{out}.csv")
    assert list(df["entry_rate"]) == [0.1, 0.3] and (df["traffic_mode"] == "open").all()
    assert NEW_COLUMNS | {"demand_per_min", "entry_vc", "entry_los"} <= set(df.columns)
    assert df["arrivals_per_min"].iloc[1] > df["arrivals_per_min"].iloc[0]
    assert list(df["entry_los"]) == [level_of_service(d, v) for d, v in zip(df["mean_entry_delay"], df["entry_vc"])]


def test_closed_run_summary_keeps_its_baseline_wording(tmp_path, capsys, no_figures):
    experiments.main(["run", "--duration", "10", "--exit-prob", "0", "--baseline", "--out", str(tmp_path / "c")])
    printed = capsys.readouterr().out
    assert "--baseline skipped: exit_prob is already 0, so this run is its own baseline" in printed
    assert "(nothing to switch off; columns left empty)" in printed and "see --baseline" in printed
    summary = pd.read_csv(tmp_path / "c_summary.csv").iloc[0]
    assert np.isnan(summary["entry_vc"]) and np.isnan(summary["demand_per_min"])  # closed mode


def test_mixed_closed_and_open_sweep_keeps_the_closed_arm_shares(tmp_path, capsys, no_figures):
    experiments.main(["sweep", "--x", "exit_prob=0,0.2", "--series", "traffic_mode=closed,open", "--seeds", "1",
                      "--warmup", "5", "--measure", "20", "--workers", "1", "--out", str(tmp_path / "mixed")])
    printed = capsys.readouterr().out
    assert "for the open-mode runs" in printed  # the open rows are left out, and why
    shares = printed.split("Jamming due to entrances/exits (%)")[1]
    assert "closed" in shares and "open " not in shares.split("wrote")[0]


def test_peak_profile_warns_when_the_window_is_not_whole_cycles(tmp_path, capsys, no_figures):
    experiments._peak_note(120.0, 420.0)
    assert "averages 1.30 x the entrance chance over the measured 120-420 s" in capsys.readouterr().out
    experiments._peak_note(300.0, 600.0)  # the default run's second half: exactly 1
    assert capsys.readouterr().out == ""
    out = tmp_path / "peak"
    experiments.main(["run", "--duration", "100", "--traffic-mode", "open", "--demand-profile", "peak",
                      "--out", str(out)])
    assert "note: the peak demand profile averages 0.65 x" in capsys.readouterr().out
    summary = pd.read_csv(f"{out}_summary.csv").iloc[0]
    expected = 60 * 0.1 * 4 * mean_demand_factor("peak", 50.0, 100.0)  # the demand actually offered
    assert summary["demand_per_min"] == pytest.approx(expected, rel=0.02)


def test_open_mode_sweep_over_exit_chance_reports_no_arm_share(tmp_path, capsys, no_figures):
    experiments.main(["sweep", "--x", "exit_prob=0,0.1", "--set", "traffic_mode=open", "--seeds", "1",
                      "--warmup", "0", "--measure", "5", "--workers", "1", "--out", str(tmp_path / "exits")])
    printed = capsys.readouterr().out
    assert "no share of the jamming due to entrances/exits in open mode" in printed
    assert "Jamming due to entrances/exits (%)" not in printed


@pytest.mark.parametrize("config", [dict(n_lanes=5, n_arms=12, truck_share=0.3, truck_length=23.0, radius=20.0,
                                         density=60.0),
                                    dict(n_lanes=2, n_arms=4, truck_share=0.3, truck_length=23.0, radius=20.0,
                                         density=50.0),
                                    dict(n_lanes=3, radius=10.5, density=200.0)])
def test_open_start_places_every_vehicle_density_asks_for(config):
    """Density sets the cars on the ring at the start of an open-mode run also when the vehicles do not
    all fit the lanes' shares (a heavy truck mix, a start at jam density): the start places the rest as
    a live Density raise does, instead of leaving them out (closed mode tops up on its first step)."""
    for seed in range(8):
        sim = RingSim(SimConfig(traffic_mode="open", seed=seed, **config))
        assert int(sim.alive.sum()) == sim.target_cars(), seed
        assert not sim._pending
        assert_valid(sim)
        sim.step(30)
        assert_valid(sim)
