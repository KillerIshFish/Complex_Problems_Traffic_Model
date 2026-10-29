"""Stage S0 after review: whole-number floats for anticipation; Raff's critical gap from the cumulative
distributions (FHWA-OR-RD-22-11 Table 7.1) beside the count version; the 15-minute capacity, pc/h conflicting
flow and the entry delay without the test entrance's standing queue; the capacity test's routes on 2+ lanes;
its idle rules (Exit chance, no entrances, closed mode); the random entrance chance on the drawn weather effect;
experiments capacity fitting the validated range weighted by entries, leaving extrapolations out, giving the
headway-based factors and refusing 3+ lanes; the viewers' labels and texts for the test."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from roundabout_sim import app as viewer
from roundabout_sim import engine, experiments, presets
from roundabout_sim import sweep_viewer as sv
from roundabout_sim.config import SimConfig, idle_note, inactive_reason
from roundabout_sim.engine import LC_ZONE, SUBJECT_ARM, TWO_PI, RingSim
from roundabout_sim.metrics import Recorder, snapshot
from roundabout_sim.presets import HCM6_CAPACITY
from tests.helpers import rendered_texts

SUBJECT = dict(traffic_mode="open", subject_arm="on")


def preset(name: str, seed: int = 0, **changes) -> SimConfig:
    return presets.apply(name, SimConfig(seed=seed)).updated(**changes)


# ------------------------------------------------------------------ anticipation
def test_anticipation_as_a_whole_number_float_runs_as_the_integer():
    runs = [RingSim(SimConfig(seed=1, anticipation=a)) for a in (3, 3.0, np.float64(3))]
    for sim in runs:
        sim.step(200)
    for sim in runs[1:]:
        np.testing.assert_array_equal(sim.theta, runs[0].theta)
    sim = RingSim(SimConfig(seed=1))
    assert not sim.set(anticipation=2.0)
    sim.step(20)  # (it used to raise on the next step)


# ------------------------------------------------------------------ Raff by shares and by counts
def test_short_rejected_gaps_lower_the_cdf_critical_gap_only():
    """FHWA-OR-RD-22-11 p. 117: CACC platoons add short rejected gaps, which shift the rejected gaps'
    distribution and lower the Raff tc of Table 7.1; counts of rejected gaps longer than t do not change."""
    rng = np.random.default_rng(0)
    accepted, rejected = rng.normal(5.0, 1.0, 300), rng.normal(3.0, 0.8, 300)
    count, cdf = experiments.raff_critical_gap(accepted, rejected), experiments.raff_cdf_critical_gap(accepted, rejected)
    assert count == pytest.approx(cdf, abs=0.05) and 3.6 < cdf < 4.2  # equal numbers: nearly the same
    platoons = np.concatenate((rejected, np.full(300, 0.7)))
    assert experiments.raff_critical_gap(accepted, platoons) == count
    assert experiments.raff_cdf_critical_gap(accepted, platoons) < cdf - 0.05
    # a sharp threshold gives the midpoint either way; equal-sized sets give the counts' answer
    assert experiments.raff_cdf_critical_gap([3.1, 3.5, 4, 5], [1, 2, 2.9]) == pytest.approx(3.0)
    assert experiments.raff_cdf_critical_gap([2, 4, 5], [1, 3, 6]) == pytest.approx(3.5)
    assert math.isnan(experiments.raff_cdf_critical_gap([], [1.0]))


def test_cdf_raff_balances_the_shares_at_its_estimate():
    accepted, rejected = [2.0, 4.0, 5.0, 6.0], [1.0, 1.5, 3.0, 4.5, 7.0, 0.5, 0.6, 0.7]
    t = experiments.raff_cdf_critical_gap(accepted, rejected)
    assert np.mean(np.array(accepted) < t) == pytest.approx(np.mean(np.array(rejected) > t))


# ------------------------------------------------------------------ the new columns
@pytest.fixture(scope="module")
def truck_run():
    """us_single_lane capacity test with trucks, 60 s warm-up and a 960 s window (one whole 15 minutes)."""
    sim = RingSim(preset("us_single_lane", 8, entry_rate=0.06, truck_share=0.2, **SUBJECT))
    rec = Recorder(sim, 1.0)
    experiments.advance(sim, rec, 60.0)
    start, entries = experiments._counters(sim), len(sim.entry_log)
    tap = experiments.SubjectTap(sim)
    experiments.advance(sim, rec, 960.0, tap=tap)
    return sim, rec, start, tap, experiments.summarise(sim, rec, 60.0, start, tap)


def test_the_gap_columns_are_the_cdf_and_count_versions(truck_run):
    sim, rec, start, tap, row = truck_run
    accepted, rejected = experiments.gap_acceptance(tap.entries, tap.passes)
    assert row["subject_tc_raff"] == experiments.raff_cdf_critical_gap(accepted, rejected)
    assert row["subject_tc_raff_count"] == experiments.raff_critical_gap(accepted, rejected)
    assert 3.0 < row["subject_tc_raff"] < 6.0


def test_the_fifteen_minute_capacity(truck_run):
    sim, rec, start, tap, row = truck_run
    t0 = 60.0
    first = [t for t in tap.entries if t0 <= t < t0 + 900.0]
    assert row["subject_cap15_h"] == pytest.approx(4.0 * len(first))
    assert experiments.peak_quarter([0.0, 10.0, 950.0], 0.0, 1800.0, 0.0) == 8.0  # max(2, 1) x 4
    assert math.isnan(experiments.peak_quarter([1.0], 0.0, 899.0, 0.0))  # no whole 15 minutes
    assert math.isnan(experiments.peak_quarter([1.0], 0.0, 1800.0, 5.0))  # entries known only from 5 s
    # what the logs still hold covers no whole 15 minutes
    assert math.isnan(experiments.summarise(sim, rec, t0, start)["subject_cap15_h"])


def test_the_conflicting_flow_in_passenger_cars(truck_run):
    sim, rec, start, tap, row = truck_run
    span = sim.time - 60.0
    trucks = sim.arm_passed_trucks[SUBJECT_ARM] - start["arm_passed_trucks"][SUBJECT_ARM]
    assert 0 < trucks < sim.arm_passed[SUBJECT_ARM] - start["arm_passed"][SUBJECT_ARM]
    assert row["subject_vc_pch"] == pytest.approx(row["subject_vc_h"] + 3600.0 * trucks / span)
    cars = experiments._sweep_job((experiments._differ(preset("us_single_lane", 9, entry_rate=0.05, **SUBJECT)), 9,
                                   30.0, 120.0, 0.5))
    assert cars["subject_vc_pch"] == cars["subject_vc_h"] > 0  # no trucks: the same


def test_the_entry_delay_and_queues_leave_the_test_queue_out():
    sim = RingSim(preset("us_single_lane", 11, entry_rate=0.08, **SUBJECT))
    rec = Recorder(sim, 0.5)
    experiments.advance(sim, rec, 60.0)
    start, before = experiments._counters(sim), len(sim.entry_log)
    experiments.advance(sim, rec, 240.0)
    row = experiments.summarise(sim, rec, 60.0, start)
    waits = [w for _, arm, w in list(sim.entry_log)[before:] if arm != SUBJECT_ARM]
    assert waits and row["mean_entry_delay"] == pytest.approx(np.mean(waits))
    now = snapshot(sim)
    assert now["queued"] == int(sim.queue[1:].sum()) and now["longest_queue"] == int(sim.queue[1:].max())
    idle = experiments._sweep_job((experiments._differ(preset("us_single_lane", 12, entry_rate=0.0, **SUBJECT)), 12,
                                   30.0, 60.0, 0.5))
    assert math.isnan(idle["mean_entry_delay"]) and idle["queued"] == 0 and idle["peak_queue"] == 0
    # without the test, the totals as before
    plain = RingSim(preset("us_single_lane", 11, entry_rate=0.08))
    plain.step(600)
    assert plain.arm_entry_wait.sum() == pytest.approx(plain.entry_wait_total)


# ------------------------------------------------------------------ routes on 2+ lanes
def test_two_lane_routes_leave_room_to_move_out():
    sim = RingSim(preset("us_two_lane", 0, entry_rate=0.08, **SUBJECT))
    arc = ((sim.exit_angle - sim.entry_angle[SUBJECT_ARM]) % TWO_PI) * sim.r_out
    e = sim._subject_exit
    assert arc[e] >= (sim.n_lanes - 1) * LC_ZONE + 20.0 and e != (SUBJECT_ARM + 1) % sim.n_arms
    assert arc[e] == min(a for a in arc if a >= (sim.n_lanes - 1) * LC_ZONE + 20.0)
    entrants = set()
    for _ in range(6000):
        t = sim.time
        sim.step()
        new = np.flatnonzero(sim.alive & (sim.entered_at == t) & (sim.theta == sim.entry_angle[SUBJECT_ARM]))
        entrants |= set(sim.uid[new].tolist())
    assert {arm for _, arm, _ in sim.exit_log} == {e} and sim.n_missed == 0
    at_subject = [uid for _, arm, _, uid in sim.conflict_log if arm == SUBJECT_ARM]
    assert not set(at_subject) & entrants and max(np.unique(at_subject, return_counts=True)[1]) == 1
    # lane use at the conflict point is the drivers' own, not all on the outer lane
    assert 0 < sim.arm_passed_outer[SUBJECT_ARM] < sim.arm_passed[SUBJECT_ARM]
    # one lane: the next exit, as before
    assert RingSim(preset("us_single_lane", 0, **SUBJECT))._subject_exit == (SUBJECT_ARM + 1) % 4


def test_a_live_exit_chance_change_leaves_the_tests_routes_alone():
    sim = RingSim(preset("us_two_lane", 1, entry_rate=0.08, **SUBJECT))
    sim.step(300)
    before = sim.exit_at.copy()
    states = [g.bit_generator.state for g in (sim.rng, sim.rng_demand, sim.rng_drivers)]
    assert not sim.set(exit_prob=0.6)
    np.testing.assert_array_equal(sim.exit_at, before)
    assert [g.bit_generator.state for g in (sim.rng, sim.rng_demand, sim.rng_drivers)] == states


# ------------------------------------------------------------------ idle rules
def test_what_the_capacity_test_leaves_idle():
    on = preset("us_single_lane", **SUBJECT)
    assert inactive_reason("exit_prob", on) == "capacity test"
    assert inactive_reason("exit_prob", on.updated(subject_arm="off")) == ""
    assert inactive_reason("exit_prob", on.updated(traffic_mode="closed")) == ""
    assert inactive_reason("subject_arm", on.updated(n_arms=0)) == "no entrances"
    assert "capacity test" in idle_note(["exit_prob"], "capacity test")
    notes = experiments.idle_notes(["exit_prob"], [on, on.updated(exit_prob=0.5)])
    assert notes and "subject_arm=off" in notes[0]
    assert "Capacity test" in sv.IDLE_FIX["capacity test"]


def test_switching_the_test_where_it_cannot_act_keeps_it_for_later():
    sim = RingSim(SimConfig(seed=2))  # closed
    sim.step(50)
    generation = sim.generation
    assert not sim.set(subject_arm="on") and sim.generation == generation and sim.time > 0
    assert sim.cfg.subject_arm == "on" and not sim.subject
    assert sim.set(traffic_mode="open") and sim.subject  # it acts once it can
    no_arms = RingSim(SimConfig(seed=3, traffic_mode="open", n_arms=0))
    assert not no_arms.set(subject_arm="on") and not no_arms.subject
    assert RingSim(preset("us_single_lane", 4, traffic_mode="open")).set(subject_arm="on")  # where it acts: rebuild


# ------------------------------------------------------------------ random entrance chance
def test_the_random_entrance_chance_follows_the_drawn_weather_effect():
    a, b = HCM6_CAPACITY[1]
    base = dict(vc=0.5, n_lanes=1, n_arms=4, exit_prob=0.5, imbalance=0.0)
    for road in ("dry", "snow"):
        assert presets._hcm_entry_rate(**base, road_condition=road, weather_scale=1.0) == \
            presets._hcm_entry_rate(**base, road_condition=road)
    dry = [presets._hcm_entry_rate(**base, road_condition="dry", weather_scale=s) for s in (0.5, 2.0)]
    snow = [presets._hcm_entry_rate(**base, road_condition="snow", weather_scale=s) for s in (0.5, 1.0, 2.0)]
    assert dry[0] == dry[1] and snow[0] > snow[1] > snow[2]
    fixed = dict(traffic_mode="open", road_condition="snow", n_arms=4, n_lanes=1)
    rates = [presets.random_changes(5, fixed={**fixed, "weather_scale": s})["entry_rate"] for s in (0.5, 2.0)]
    assert rates[0] > rates[1]
    drawn = presets.random_changes(5, fixed={k: v for k, v in fixed.items() if k != "road_condition"})
    assert set(drawn) >= {"weather_scale", "entry_rate"}


# ------------------------------------------------------------------ experiments capacity
def collapse_csv(seeds=range(6), tf=(2.6, 2.0), tc=(4.1, 3.6), lanes=1, trucks=0.0) -> pd.DataFrame:
    """Capacity-test runs on the HCM base up to vc 600 and collapsing beyond (to a handful of entries, as the
    model's ring does), two series (follow_up 2.5 = the base, 2.0 = x 1.2 and B x 0.9), with per-run tf and tc."""
    a, b = HCM6_CAPACITY[1]
    rng = np.random.default_rng(2)
    rows = []
    for seed in seeds:
        for rate in np.arange(0.0, 0.17, 0.02):
            vc = 3600 * 3 * rate * (1 + 0.03 * rng.standard_normal())
            for (fu, fa, fb), f, c in zip(((2.5, 1.0, 1.0), (2.0, 1.2, 0.9)), tf, tc):
                cap = fa * a * math.exp(-fb * b * vc) * math.exp(0.02 * rng.standard_normal())
                if vc > 700:
                    cap *= math.exp(-(vc - 700) / 80.0)  # the collapse
                rows.append({"entry_rate": round(rate, 2), "follow_up": fu, "seed": seed, "n_lanes": lanes,
                             "truck_share": trucks, "subject_vc_h": vc, "subject_vc_pch": vc * 1.1,
                             "subject_cap_h": cap, "subject_cap_pch": cap * 1.1, "subject_vc_outer_h": vc,
                             "subject_tf": f + 0.01 * rng.standard_normal(),
                             "subject_tc_raff": c + 0.01 * rng.standard_normal() if vc > 0 else np.nan})
    return pd.DataFrame(rows)


def test_the_default_fit_is_the_validated_range_weighted_by_entries():
    a, b = HCM6_CAPACITY[1]
    df = collapse_csv()
    result = experiments.capacity_analysis(df, "follow_up", 2.5, boot=200)
    base = result["series"][result["ref"]]
    assert base["value"] == 2.5 and result["fit_from"] == "validated range" and result["fit"] == result["validated"]
    assert 600 < result["validated"][1] < 760 and result["validated"][0] == 0.0
    assert base["A"] == pytest.approx(a, rel=0.03) and base["B"] == pytest.approx(b, rel=0.1)
    assert (base["vc"][base["used"]] <= result["fit"][1]).all() and not base["used"].all()
    everything = experiments.capacity_analysis(df, "follow_up", 2.5, fit=None, boot=200)["series"][1]
    unweighted = experiments.fit_capacity(everything["vc"], everything["cap"])
    # over the collapse, weighting by entries keeps A near the measured vc-0 capacity; the plain fit does not
    assert abs(everything["A"] / a - 1) < abs(unweighted[0] / a - 1)


def test_ratios_and_gains_outside_the_fitted_flows_are_left_out():
    df = collapse_csv()
    result = experiments.capacity_analysis(df, "follow_up", 2.5, boot=200)
    table = experiments.capacity_table(result, "follow_up")
    for _, row in table.iterrows():
        assert np.isfinite(row["hcm_ratio_0"]) and np.isfinite(row["hcm_ratio_400"])
        assert np.isnan(row["hcm_ratio_800"]) and np.isnan(row["hcm_ratio_1200"])
    other = table.set_index("follow_up").loc[2.0]
    assert np.isfinite(other["gain_400"]) and np.isnan(other["gain_1200"]) and np.isnan(other["gain_1200_lo"])
    assert other["fA"] == pytest.approx(1.2, rel=0.03)


def test_the_headway_factors_follow_siegloch():
    df = collapse_csv()
    result = experiments.capacity_analysis(df, "follow_up", 2.5, boot=200)
    row = experiments.capacity_table(result, "follow_up").set_index("follow_up").loc[2.0]
    assert row["tf"] == pytest.approx(2.0, abs=0.01) and row["tc"] == pytest.approx(3.6, abs=0.01)
    assert row["A_hw"] == pytest.approx(3600 / row["tf"]) and row["B_hw"] == pytest.approx((row["tc"] - row["tf"] / 2)
                                                                                           / 3600)
    assert row["fA_hw"] == pytest.approx(2.6 / 2.0, rel=0.01)
    assert row["fB_hw"] == pytest.approx((3.6 - 1.0) / (4.1 - 1.3), rel=0.01)
    assert row["fA_hw_lo"] <= row["fA_hw"] <= row["fA_hw_hi"]
    assert row["slope_fhwa_sim"] == pytest.approx(math.log(2.16 / 1.36) / 1200)


def test_capacity_picks_pc_per_hour_with_trucks_and_refuses_three_lanes():
    trucks = experiments.capacity_analysis(collapse_csv(trucks=0.1), boot=50)
    assert (trucks["cap"], trucks["vc"], trucks["units"]) == ("subject_cap_pch", "subject_vc_pch", "pc/h")
    assert experiments.capacity_analysis(collapse_csv(trucks=0.1), boot=50, cap="subject_cap_h")["units"] == "veh/h"
    assert experiments.capacity_analysis(collapse_csv(), boot=50)["cap"] == "subject_cap_h"
    with pytest.raises(ValueError, match="3 or more circulating lanes"):
        experiments.capacity_analysis(collapse_csv(lanes=3), boot=50)
    two = experiments.capacity_analysis(collapse_csv(lanes=2), boot=50)
    assert two["hcm"] == HCM6_CAPACITY[2]


def test_capacity_command_prints_the_fitted_range_and_warnings(tmp_path, capsys):
    path = tmp_path / "c.csv"
    collapse_csv(seeds=range(3)).to_csv(path, index=False)
    experiments.main(["capacity", str(path), "--by", "follow_up", "--ref", "2.5", "--bootstrap", "100"])
    printed = capsys.readouterr().out
    assert "fitted: the runs in the validated range" in printed and "Follow-up time = 2" in printed
    assert "warning: 3 seeds" in printed and "27 runs (" in printed and "27.0 runs" not in printed
    assert "headways: fA = " in printed and "FHWA Sec. 7.3.1 simulated +3.86e-04" in printed
    assert "(- = outside the fitted flows)" in printed
    experiments.main(["capacity", str(path), "--fit", "0:1500", "--bootstrap", "50", "--out", str(tmp_path / "all")])
    assert "outside the validated range" in capsys.readouterr().out
    with pytest.raises(SystemExit):  # (the CSV has no subject_cap15_h)
        experiments.main(["capacity", str(path), "--cap", "subject_cap15_h"])
    assert "subject_cap15_h" in capsys.readouterr().err


# ------------------------------------------------------------------ viewers and help
def test_the_sweep_viewer_names_the_new_columns():
    new = [*experiments.SUBJECT_COLUMNS, "speed_sd_pooled", *(f"lane{k}_speed_sd" for k in range(5))]
    for name in new:
        assert sv.column_label(name) != name, name
        assert name in sv.definitions(), name


def test_the_view_header_and_queue_tag_show_the_test(new_app, monkeypatch):
    sim = RingSim(preset("us_single_lane", 5, entry_rate=0.1, demand_imbalance=0.0, **SUBJECT))
    texts = viewer.App._demand_texts(SimpleNamespace(sim=sim))
    assert "at each other entrance" in texts[0] and "capacity test" in texts[0] and texts[-1] == ""
    plain = viewer.App._demand_texts(SimpleNamespace(sim=RingSim(preset("us_single_lane", 5, entry_rate=0.1,
                                                                        demand_imbalance=0.0))))
    assert plain == [f"  ·  open: {3600 * 0.1:.0f} veh/h at each entrance", f"  ·  open: {3600 * 0.1:.0f} veh/h each",
                     "  ·  open", ""]
    monkeypatch.setattr(engine, "SUBJECT_QUEUE", 60)  # longer than any stub: its count shows on a tag
    app = new_app(preset("us_single_lane", 5, entry_rate=0.1, max_queue=3, **SUBJECT))
    app.sim.step(50)
    shown = rendered_texts(app, monkeypatch)
    assert any(t.endswith(" test") for t in shown) and not any(t.startswith("6") and "full" in t for t in shown)
    assert "capacity test" in viewer.TAB_HELP["Demand"] and "Marked test" in viewer.QUEUE_TAG_HELP


def test_the_help_example_keeps_its_line_break():
    assert "--preset us_single_lane \\\n        --set subject_arm=on" in experiments.__doc__
