"""The capacity test (S0): the conflict-point log and counts (observer only, every pass counted once, a car
placed on an entry point not), the saturated subject entrance (queue never empty, no arrivals counted, every
car leaving at the first exit after it, so its conflicting flow is the other entrances' entry flow), and the
summary columns it feeds (subject_*, speed_sd_pooled, lane{k}_speed_sd) in runs, sweep CSVs and the sweep
viewer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from roundabout_sim import experiments, presets
from roundabout_sim import sweep_viewer as sv
from roundabout_sim.config import PARAM_BY_NAME, SimConfig
from roundabout_sim.engine import LOG_HORIZON, SUBJECT_ARM, SUBJECT_QUEUE, TWO_PI, RingSim
from roundabout_sim.metrics import Recorder, snapshot
from tests.helpers import assert_valid

SUBJECT = dict(traffic_mode="open", subject_arm="on")


def us_single(seed: int = 0, **changes) -> SimConfig:
    return presets.apply("us_single_lane", SimConfig(seed=seed)).updated(**changes)


def expected_passes(old: np.ndarray, new: np.ndarray, points: np.ndarray) -> list[tuple[int, int]]:
    """(car index, arm) of every entry point a car's step from ``old`` to ``new`` (angles in [0, 2pi)) reached,
    worked out as the detector line is (independently of the engine's sorted search)."""
    hits = []
    for arm, e in enumerate(points):
        before, after = (old - e) % TWO_PI, (new - e) % TWO_PI
        hits += [(int(i), arm) for i in np.flatnonzero((new != old) & (before > after))]
    return hits


# ------------------------------------------------------------------ conflict points
def test_hand_placed_cars_pass_the_entry_points_they_reach():
    sim = RingSim(SimConfig(seed=0, n_lanes=2, n_arms=4, traffic_mode="open", entry_rate=0.0, density=1,
                            driver_noise=0.0, lane_change_rate=0.0))
    sim.alive[:] = False
    e = sim.entry_angle
    assert list(sim._entry_order) == [1, 2, 3, 0]  # arm 0 is the bottom one: its entry point comes last
    lanes = np.array([1, 0, 0, 0, 1])  # (cars 2 and 4 in different lanes: they would touch)
    angles = np.array([e[2] - 1e-3, e[0] - 1e-3, e[1], e[3] + 0.5, (e[1] - 2e-3) % TWO_PI])
    slots = sim._spawn(lanes, angles, np.full(5, 8.0))
    sim.exit_at[slots] = np.inf
    old = sim.theta[slots].copy()
    sim.step()
    passed = [(sim.uid[slots].tolist().index(uid), arm, lane) for _, arm, lane, uid, _ in sim.conflict_log]
    # just before arm 2 (outer lane) and arm 0 (inner lane): passed; on arm 1's point or far off: not
    assert sorted(passed) == [(0, 2, 1), (1, 0, 0), (4, 1, 1)]
    assert sorted((i, a) for i, a, _ in passed) == sorted(expected_passes(old, sim.theta[slots], e))
    assert sim.arm_passed.tolist() == [1, 1, 1, 0] and sim.arm_passed_outer.tolist() == [0, 1, 1, 0]
    assert all(t == 0.0 for t, *_ in sim.conflict_log)


def test_every_pass_is_logged_once_as_the_detector_counts_them():
    """Over a long open run on three lanes, the log holds exactly the passes the detector formula finds,
    step by step, and the per-entrance counts match the log."""
    sim = RingSim(SimConfig(seed=1, n_lanes=3, n_arms=5, traffic_mode="open", entry_rate=0.15, mean_speed=14.0))
    want = []
    for _ in range(1500):
        idx = np.flatnonzero(sim.alive)
        uid, old, t = sim.uid[idx].copy(), sim.theta[idx].copy(), sim.time
        sim.step()
        alive = np.isin(uid, sim.uid[sim.alive])
        pos = {u: s for s, u in zip(np.flatnonzero(sim.alive), sim.uid[sim.alive])}
        new = np.array([sim.theta[pos[u]] if a else o for u, o, a in zip(uid, old, alive)])
        # (a leaver gone this step reached no entry point: its exit comes first)
        want += [(t, arm, int(uid[i])) for i, arm in expected_passes(old, new, sim.entry_angle) if alive[i]]
    got = [(t, arm, uid) for t, arm, _, uid, _ in sim.conflict_log]
    assert sorted(got) == sorted(want) and len(got) > 300
    assert sim.arm_passed.sum() == len(got) and np.bincount([a for _, a, _ in got], minlength=5).tolist() == \
        sim.arm_passed.tolist()
    assert sim.arm_passed_outer.sum() == sum(lane == sim.outer for _, _, lane, *_ in sim.conflict_log)


def test_the_conflict_log_is_an_observer_and_is_pruned():
    sim = RingSim(us_single(2, entry_rate=0.12))
    rngs = [g.bit_generator.state for g in (sim.rng, sim.rng_demand, sim.rng_drivers)]
    lanes = [order.copy() for order, _ in sim._lanes]
    sim._log_conflicts(np.flatnonzero(sim.alive), sim.theta[sim.alive] - 0.3, np.full(int(sim.alive.sum()), 0.3))
    assert [g.bit_generator.state for g in (sim.rng, sim.rng_demand, sim.rng_drivers)] == rngs
    assert all(np.array_equal(a, order) for a, (order, _) in zip(lanes, sim._lanes))
    sim.step(int(700 / sim.cfg.dt))
    assert sim.conflict_log and sim.conflict_log[0][0] >= sim.time - LOG_HORIZON - 1.0
    assert sim.arm_passed.sum() > len(sim.conflict_log)  # the counts go back further than the log


# ------------------------------------------------------------------ the subject entrance
@pytest.fixture(scope="module")
def subject_run():
    """us_single_lane with the capacity test at a demand that loads the ring (about 700 veh/h at arm 0), stepped
    with every arm-0 entrant's uid and the smallest subject queue seen."""
    sim = RingSim(us_single(3, entry_rate=0.07, truck_share=0.1, **SUBJECT))
    entrants, smallest = set(), SUBJECT_QUEUE
    for _ in range(6000):
        t = sim.time
        sim.step()
        smallest = min(smallest, int(sim.queue[SUBJECT_ARM]))
        new = np.flatnonzero(sim.alive & (sim.entered_at == t) & (sim.theta == sim.entry_angle[SUBJECT_ARM]))
        entrants |= set(sim.uid[new].tolist())
    return sim, entrants, smallest


def test_the_subject_queue_never_empties_and_counts_no_arrivals(subject_run):
    sim, entrants, smallest = subject_run
    assert sim.subject and smallest >= SUBJECT_QUEUE - 1
    assert sim.arm_arrived[SUBJECT_ARM] == 0 and all(arm != SUBJECT_ARM for _, arm in sim.arrival_log)
    assert all(arm != SUBJECT_ARM for _, arm in sim.blocked_log)
    assert sim.arrival_rates()[SUBJECT_ARM] == 0.0 and (sim.arrival_rates()[1:] > 0).all()
    assert snapshot(sim)["demand_per_min"] == pytest.approx(60 * sim.cfg.entry_rate * sim.demand_weight[1:].sum())
    assert sim.n_topped_up == sim.arm_entered[SUBJECT_ARM] + int(sim.queue[SUBJECT_ARM])
    assert sim.n_arrived + sim.n_topped_up == sim.n_entered + int(sim.queue.sum()) + sim.n_blocked + sim.n_gave_up
    assert len(entrants) == sim.arm_entered[SUBJECT_ARM] > 40
    assert 0 < sim.arm_entered_trucks[SUBJECT_ARM] < sim.arm_entered[SUBJECT_ARM]
    assert_valid(sim)


def test_every_car_leaves_at_the_first_exit_after_the_subject_entrance(subject_run):
    sim, entrants, _ = subject_run
    assert {arm for _, arm, _ in sim.exit_log} == {(SUBJECT_ARM + 1) % sim.n_arms}
    at_subject = [uid for _, arm, _, uid, _ in sim.conflict_log if arm == SUBJECT_ARM]
    assert not set(at_subject) & entrants  # the subject's own entrants never pass its entry point
    assert max(np.unique(at_subject, return_counts=True)[1]) == 1  # every other car passes it once
    # so its conflicting flow is the other entrances' entry flow (less the cars still on their way)
    others = int(sim.arm_entered[1:].sum())
    assert others - int(sim.alive.sum()) <= sim.arm_passed[SUBJECT_ARM] <= others + 10
    # entrance 1 sees none (everyone leaves at its exit, before its entry point), entrance 3 the most others
    assert sim.arm_passed[1] == 0 < sim.arm_passed[2] < sim.arm_passed[3]


def test_routes_are_set_without_drawing():
    sim = RingSim(us_single(4, entry_rate=0.1, **SUBJECT))
    states = [g.bit_generator.state for g in (sim.rng, sim.rng_demand, sim.rng_drivers)]
    idx = np.flatnonzero(sim.alive)
    before = sim.exit_at[idx].copy()
    sim._assign_routes(idx)
    np.testing.assert_array_equal(sim.exit_at[idx], before)
    assert [g.bit_generator.state for g in (sim.rng, sim.rng_demand, sim.rng_drivers)] == states
    at = sim.theta[idx] + sim.exit_at[idx] - sim.travel[idx]  # where each one leaves: entrance 1's exit
    np.testing.assert_allclose((at - sim.exit_angle[1] + np.pi) % TWO_PI - np.pi, 0.0, atol=1e-9)
    assert (sim.exit_at[idx] - sim.travel[idx] <= TWO_PI).all()


def test_the_test_acts_in_open_mode_only_and_rebuilds():
    spec = PARAM_BY_NAME["subject_arm"]
    assert spec.reset and spec.applies_in == "open" and spec.choices == ("off", "on") and spec.group == "Demand"
    closed = [RingSim(SimConfig(seed=5, subject_arm=arm)) for arm in ("off", "on")]
    for sim in closed:
        sim.step(600)
    assert not closed[1].subject and closed[0].n_exited == closed[1].n_exited
    np.testing.assert_array_equal(closed[0].theta, closed[1].theta)
    assert not RingSim(SimConfig(seed=5, n_arms=0, **SUBJECT)).subject  # nothing to test without entrances
    sim = RingSim(us_single(5, entry_rate=0.1))
    assert sim.set(subject_arm="on") and sim.subject and sim.time == 0.0
    sim.step(100)
    assert sim.queue[SUBJECT_ARM] >= SUBJECT_QUEUE - 1
    assert not sim.set(max_queue=3)  # (a live Max queue cut leaves the test's queue full)
    assert sim.queue[SUBJECT_ARM] >= SUBJECT_QUEUE - 1 and sim.queue[1:].max() <= 3


def test_conflicting_flow_spans_the_capacity_range():
    """vc at the subject entrance follows the Entrance chance of the others, from 0 up past 1,000 veh/h
    (the full probe, 0 to about 1,400 veh/h at 0.2 /s, is in the stage report)."""
    vcs = []
    for rate in (0.0, 0.05, 0.12):
        sim = RingSim(us_single(6, entry_rate=rate, **SUBJECT))
        sim.step(3000)
        before = sim.arm_passed[SUBJECT_ARM]
        sim.step(3000)
        vcs.append((sim.arm_passed[SUBJECT_ARM] - before) * 3600 / 300.0)
    assert vcs[0] == 0 and 300 < vcs[1] < 700 and vcs[2] > 1000


# ------------------------------------------------------------------ summary columns
def test_pooled_speed_spread():
    assert experiments.pooled_sd([1.0, 1.0], [2.0, 4.0]) == pytest.approx(np.sqrt(2.0))
    assert experiments.pooled_sd([3.0, np.nan], [5.0, 9.0]) == pytest.approx(3.0)
    assert np.isnan(experiments.pooled_sd([np.nan], [np.nan]))


def test_every_run_has_the_columns_nan_where_they_do_not_apply():
    row = experiments._sweep_job(({"n_lanes": 2}, 0, 10.0, 20.0, 0.5))
    assert all(np.isnan(row[c]) for c in experiments.SUBJECT_COLUMNS)
    assert np.isfinite(row["speed_sd_pooled"]) and np.isfinite(row["lane0_speed_sd"])
    assert np.isfinite(row["lane1_speed_sd"]) and all(np.isnan(row[f"lane{k}_speed_sd"]) for k in (2, 3, 4))
    assert not set(row) - set(experiments.CONFIG_NAMES) - {"seed"} & set(experiments.CONFIG_NAMES)


def test_pooled_spreads_follow_the_samples():
    sim = RingSim(SimConfig(seed=7, n_lanes=2, density=60))
    rec = Recorder(sim, 0.5)
    experiments.advance(sim, rec, 10.0)
    start = experiments._counters(sim)
    experiments.advance(sim, rec, 20.0)
    row = experiments.summarise(sim, rec, 10.0, start)
    window = rec.frame()[rec.frame()["time"] >= 10.0]
    std, mean = window["speed_std"].to_numpy(), window["mean_speed"].to_numpy()
    assert row["speed_sd_pooled"] == pytest.approx(np.sqrt(np.mean(std**2) + np.var(mean)))
    lane = window["lane1_speed_std"].to_numpy(), window["lane1_speed"].to_numpy()
    assert row["lane1_speed_sd"] == pytest.approx(np.sqrt(np.nanmean(lane[0] ** 2) + np.nanvar(lane[1])))


def test_the_subject_columns_count_the_window_exactly():
    sim = RingSim(us_single(8, entry_rate=0.06, truck_share=0.2, **SUBJECT))
    rec = Recorder(sim, 0.5)
    experiments.advance(sim, rec, 60.0)
    start, entries = experiments._counters(sim), len(sim.entry_log)
    tap = experiments.SubjectTap(sim)
    experiments.advance(sim, rec, 300.0, tap=tap)
    row = experiments.summarise(sim, rec, 60.0, start, tap)
    span = sim.time - 60.0
    mine = [e for e in list(sim.entry_log)[entries:] if e[1] == SUBJECT_ARM]
    assert row["subject_cap_h"] == pytest.approx(3600 * len(mine) / span)
    trucks = sim.arm_entered_trucks[0] - start["arm_entered_trucks"][0]
    assert 0 < trucks and row["subject_cap_pch"] == pytest.approx(3600 * (len(mine) + trucks) / span)
    assert row["subject_vc_h"] == pytest.approx(3600 * (sim.arm_passed[0] - start["arm_passed"][0]) / span)
    assert row["subject_vc_outer_h"] == row["subject_vc_h"]  # one lane
    assert row["subject_queue_empty_share"] == 0.0
    assert tap.entries == [t for t, *_ in mine]
    # the tap and what the logs still hold agree for a window shorter than they keep
    again = experiments.summarise(sim, rec, 60.0, start)
    assert (again["subject_tf"], again["subject_tc_raff"]) == (row["subject_tf"], row["subject_tc_raff"])
    assert 2.6 <= row["subject_tf"] < 2.8 and 3.0 < row["subject_tc_raff"] < 7.0  # follow_up 2.5 + one step


def test_the_follow_up_headway_is_the_engines_at_zero_conflicting_flow():
    row = experiments._sweep_job((experiments._differ(us_single(9, entry_rate=0.0, **SUBJECT)), 9, 30.0, 120.0,
                                  0.5))
    assert row["subject_vc_h"] == 0 and np.isnan(row["subject_tc_raff"])  # no gap was ever rejected
    assert row["subject_tf"] == pytest.approx(2.6)  # entries every follow_up_steps() x dt
    assert row["subject_cap_h"] == pytest.approx(3600 / 2.6, rel=0.02)


def test_the_tap_covers_windows_longer_than_the_logs():
    sim = RingSim(us_single(10, entry_rate=0.05, **SUBJECT))
    rec = Recorder(sim, 1.0)
    tap = experiments.SubjectTap(sim)
    experiments.advance(sim, rec, LOG_HORIZON + 100.0, tap=tap)
    assert len(tap.entries) == sim.arm_entered[SUBJECT_ARM]
    assert len(tap.passes) == sim.arm_passed_outer[SUBJECT_ARM] > len([1 for _, a, *_ in sim.conflict_log if a == 0])


def test_capacity_test_sweep_csv_columns_are_described(tmp_path, capsys, no_figures):
    out = tmp_path / "cap"
    experiments.main(["sweep", "--x", "entry_rate=0,0.08", "--preset", "us_single_lane", "--set", "subject_arm=on",
                      "--seeds", "1", "--warmup", "20", "--measure", "60", "--workers", "1", "--out", str(out)])
    printed = capsys.readouterr().out
    assert "test_cap" in printed and "test entrance" in printed
    df = pd.read_csv(f"{out}.csv")
    new = [*experiments.SUBJECT_COLUMNS, "speed_sd_pooled", *(f"lane{k}_speed_sd" for k in range(5))]
    assert set(new) <= set(df.columns) and (df["subject_queue_empty_share"] == 0).all()
    assert df["subject_vc_h"].iloc[0] == 0 and df["subject_vc_h"].iloc[1] > 300
    data = sv.load_sweep(f"{out}.csv")
    assert set(new) <= set(data.metrics)
    for name in new:  # the sweep viewer's help for each comes from experiments.DEFINITIONS
        assert name in sv.definitions(), name
        assert sv.metric_info(name).help.split(": ", 1)[1].strip(), name
    assert "(defined with lane0_speed_sd)" in sv.definitions()["lane3_speed_sd"]


def test_a_single_run_reports_the_capacity_test(tmp_path, capsys, no_figures):
    out = tmp_path / "run"
    experiments.main(["run", "--preset", "us_single_lane", "--set", "subject_arm=on", "--set", "entry_rate=0.05",
                      "--duration", "120", "--out", str(out)])
    printed = capsys.readouterr().out
    assert "Capacity test entrance" in printed and "Raff critical gap" in printed and "Speed spread" in printed
    summary = pd.read_csv(f"{out}_summary.csv").iloc[0]
    assert summary["subject_arm"] == "on" and summary["subject_cap_h"] > 0 and summary["subject_vc_h"] > 0
    assert summary["subject_queue_empty_share"] == 0
