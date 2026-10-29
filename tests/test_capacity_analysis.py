"""The capacity test's statistics and the analysis commands (S0): gap acceptance and Raff's critical gap on
hand-built and synthetic gaps, the follow-up headway, the HCM-form fit recovering a known A and B, the
paired bootstrap over seeds, the validated range, ``experiments capacity`` and ``experiments compare``
(pairing by seed, t intervals), with their files and errors."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from roundabout_sim import experiments
from roundabout_sim.presets import HCM6_CAPACITY
from tests.helpers import assert_png


# ------------------------------------------------------------------ gaps and headways
def test_gaps_are_accepted_when_a_car_entered_in_them():
    passes = [0.0, 2.0, 5.0, 6.0, 10.0]
    # an entry in the same step as a passing belongs to the gap that passing opens; before the first passing
    # and after the last one there is no whole gap
    entries = [-1.0, 2.0, 3.0, 9.0, 11.0]
    accepted, rejected = experiments.gap_acceptance(entries, passes)
    assert accepted.tolist() == [3.0, 4.0] and rejected.tolist() == [2.0, 1.0]
    assert [a.size for a in experiments.gap_acceptance([1.0], [5.0])] == [0, 0]


@pytest.mark.parametrize("accepted, rejected, tc", [
    ([2, 4, 5], [1, 3, 6], 3.5),  # D(t) = #accepted < t - #rejected > t is 0 on [3, 4]
    ([3.1, 3.5, 4, 5], [1, 2, 2.9], 3.0),  # a clean threshold: midway from the longest rejected to the shortest
    # accepted
    ([1, 3], [2, 4], 2.5),  # 0 on [2, 3]
    ([1, 2, 3], [0.5], 0.75),  # 0 on [0.5, 1]
    ([5], [1, 2, 3, 6], 5.5),  # 0 on [5, 6]
])
def test_raff_on_hand_counted_gaps(accepted, rejected, tc):
    assert experiments.raff_critical_gap(accepted, rejected) == pytest.approx(tc)
    t = experiments.raff_critical_gap(accepted, rejected)
    shorter = sum(a < t for a in accepted)
    longer = sum(r > t for r in rejected)
    assert shorter == longer  # Raff's definition holds at the estimate


def test_raff_where_the_counts_jump_past_each_other():
    # D is -1 below 2 (one rejected gap longer) and +2 above it (two accepted shorter): the estimate is 2
    assert experiments.raff_critical_gap([2.0, 2.0], [2.0]) == pytest.approx(2.0)


def test_raff_recovers_a_critical_gap_from_synthetic_lags():
    rng = np.random.default_rng(0)
    gaps = rng.uniform(0.0, 10.0, 20000)
    # a sharp threshold: everyone takes a gap of 4 s or more
    sharp = gaps >= 4.0
    assert experiments.raff_critical_gap(gaps[sharp], gaps[~sharp]) == pytest.approx(4.0, abs=0.01)
    # drivers who differ: acceptance a logistic around 4 s (symmetric, so Raff's count balance lands on it)
    take = rng.random(gaps.size) < 1.0 / (1.0 + np.exp(-(gaps - 4.0) / 0.4))
    assert experiments.raff_critical_gap(gaps[take], gaps[~take]) == pytest.approx(4.0, abs=0.15)
    assert math.isnan(experiments.raff_critical_gap(gaps, []))
    assert math.isnan(experiments.raff_critical_gap([], gaps))


def test_follow_up_headway_counts_only_entries_within_one_gap():
    entries = [1.0, 3.6, 6.2, 10.0, 12.6]
    assert experiments.follow_up_headway(entries, [7.0]) == pytest.approx(2.6)  # (6.2, 10) spans the passing
    assert math.isnan(experiments.follow_up_headway([1.0, 3.6], [3.6]))  # the car passed just before the entry
    assert math.isnan(experiments.follow_up_headway([1.0], []))
    assert experiments.follow_up_headway([1.0, 3.0, 6.0], []) == pytest.approx(2.5)


# ------------------------------------------------------------------ fits
def capacity_csv(seeds=range(6), noise: float = 0.05, factor: tuple[float, float] = (1.2, 0.9), same_noise=False):
    """A sweep CSV of capacity-test runs: two series (road_condition dry and wet) on HCM-form curves, dry the
    HCM base, wet A x factor[0] and B x factor[1], lognormal noise per run (the same per seed with
    ``same_noise``), its spread growing as 1 / sqrt(entries) of the base curve, as counted entries' do."""
    a, b = HCM6_CAPACITY[1]
    rng = np.random.default_rng(1)
    rows = []
    for seed in seeds:
        for rate in np.arange(0.0, 0.13, 0.02):
            vc = 3600 * 3 * rate * (1 + 0.05 * rng.standard_normal())
            shared = rng.standard_normal()
            for road, (fa, fb) in (("dry", (1.0, 1.0)), ("wet", factor)):
                eps = shared if same_noise else rng.standard_normal()
                rows.append({"entry_rate": round(rate, 2), "road_condition": road, "seed": seed, "n_lanes": 1,
                             "subject_vc_h": vc,
                             "subject_cap_h": fa * a * math.exp(-fb * b * vc) * math.exp(noise * eps
                                                                                        * math.exp(b * vc / 2))})
    return pd.DataFrame(rows)


def test_the_hcm_form_fit_recovers_a_known_curve():
    a, b = HCM6_CAPACITY[1]
    vc = np.linspace(0, 1200, 13)
    fit_a, fit_b = experiments.fit_capacity(vc, a * np.exp(-b * vc))
    assert fit_a == pytest.approx(a, rel=1e-12) and fit_b == pytest.approx(b, rel=1e-12)
    # weights per point, one fit per row (as the bootstrap uses them)
    many_a, many_b = experiments.fit_capacity(vc, a * np.exp(-b * vc), np.ones((3, vc.size)))
    np.testing.assert_allclose(many_a, a) and np.testing.assert_allclose(many_b, b)
    assert all(math.isnan(x) for x in experiments.fit_capacity([400, 400], [500, 520]))  # one flow: no slope


def test_capacity_analysis_recovers_the_factors_with_paired_intervals():
    df = capacity_csv(seeds=range(8))
    result = experiments.capacity_analysis(df, "road_condition", None, boot=500)
    dry, wet = result["series"]
    a, b = HCM6_CAPACITY[1]
    assert result["ref"] == 0 and result["lanes"] == 1 and result["hcm"] == (a, b)
    assert dry["A"] == pytest.approx(a, rel=0.05) and dry["B"] == pytest.approx(b, rel=0.1)
    table = experiments.capacity_table(result, "road_condition")
    row = table.iloc[1]
    assert row["fA"] == pytest.approx(1.2, rel=0.05) and row["fB"] == pytest.approx(0.9, rel=0.1)
    for name, true in (("fA", 1.2), ("fB", 0.9), ("slope", 0.1 * b), ("gain_1200", 1.2 * math.exp(0.1 * b * 1200))):
        assert row[f"{name}_lo"] <= true <= row[f"{name}_hi"], name
    assert row["slope"] == pytest.approx(dry["B"] - wet["B"])
    assert row["slope_hcm7"] == pytest.approx(0.15 * b) and row["slope_jiang"] == pytest.approx(-8.78e-5, rel=0.01)
    assert table.iloc[0]["fA"] == 1.0 and table.iloc[0]["gain_800"] == 1.0
    assert table.iloc[0]["hcm_ratio_0"] == pytest.approx(dry["A"] / a)


def test_the_bootstrap_resamples_the_seeds_alike_for_every_series():
    """Wet = dry x 1.1 exactly, run by run (the same random traffic per seed): paired resampling of the seeds
    gives fA exactly 1.1 in every resample, while each series' own A varies."""
    df = capacity_csv(seeds=range(6), factor=(1.1, 1.0), same_noise=True)
    result = experiments.capacity_analysis(df, "road_condition", "dry", boot=300)
    dry, wet = result["series"]
    np.testing.assert_allclose(wet["boot_A"] / dry["boot_A"], 1.1, rtol=1e-9)
    assert np.ptp(dry["boot_A"]) > 1.0
    row = experiments.capacity_table(result, "road_condition").iloc[1]
    assert row["fA_lo"] == pytest.approx(1.1) and row["fA_hi"] == pytest.approx(1.1)
    again = experiments.capacity_analysis(df, "road_condition", "dry", boot=300)
    np.testing.assert_array_equal(again["series"][0]["boot_A"], dry["boot_A"])  # seeded: reproducible


def test_the_validated_range_is_where_the_base_follows_hcm():
    a, b = HCM6_CAPACITY[1]
    vc = np.array([0, 200, 400, 600, 800, 1000, 1200], dtype=float)
    cap = a * np.exp(-b * vc) * np.array([1.0, 1.05, 0.95, 0.92, 0.7, 0.5, 0.3])
    assert experiments.validated_range(vc, cap, np.arange(7), (a, b)) == (0.0, 600.0)
    cap[0] *= 0.5  # the longest run of points within 10 % wins
    assert experiments.validated_range(vc, cap, np.arange(7), (a, b)) == (200.0, 600.0)
    assert experiments.validated_range(vc, cap * 0.1, np.arange(7), (a, b)) is None


def test_capacity_command_writes_the_table_and_the_figure(tmp_path, capsys):
    path = tmp_path / "cap.csv"
    capacity_csv(seeds=range(4)).to_csv(path, index=False)
    experiments.main(["capacity", str(path), "--by", "road_condition", "--bootstrap", "200"])
    printed = capsys.readouterr().out
    assert "Road condition = wet" in printed and "fA = " in printed and "d ln(gain)/d vc" in printed
    assert "validated range" in printed and "HCM 6/7 base 1380" in printed
    table = pd.read_csv(tmp_path / "cap_capacity.csv")
    assert list(table["road_condition"]) == ["dry", "wet"] and {"A_lo", "B_hi", "gain_400", "slope_lo"} <= set(table)
    assert_png(tmp_path / "cap_capacity.png")
    # one series, a fit range and a given validated range
    experiments.main(["capacity", str(path), "--fit", "0:900", "--validated", "0:700", "--bootstrap", "50", "--out",
                      str(tmp_path / "one")])
    one = pd.read_csv(tmp_path / "one.csv")
    assert len(one) == 1 and one["vc_max"].iloc[0] <= 900 and one["validated_hi"].iloc[0] == 700
    assert "(given with --validated)" in capsys.readouterr().out


@pytest.mark.parametrize("argv, message", [
    (["--ref", "wet"], "--ref needs --by"),
    (["--by", "road_condition", "--ref", "ice"], "road_condition takes dry, wet"),
    (["--by", "no_such"], "no such column"),
    (["--fit", "900"], "LO:HI"),
])
def test_capacity_command_errors(tmp_path, capsys, argv, message):
    path = tmp_path / "cap.csv"
    capacity_csv(seeds=range(2)).to_csv(path, index=False)
    with pytest.raises(SystemExit):
        experiments.main(["capacity", str(path), *argv])
    assert message in capsys.readouterr().err


def test_capacity_command_needs_capacity_test_runs(tmp_path, capsys):
    path = tmp_path / "plain.csv"
    pd.DataFrame({"density": [10, 20], "seed": 0, "subject_vc_h": np.nan, "subject_cap_h": np.nan}).to_csv(path)
    with pytest.raises(SystemExit):
        experiments.main(["capacity", str(path)])
    assert "subject_arm=on" in capsys.readouterr().err


# ------------------------------------------------------------------ compare
def test_t_quantiles():
    assert experiments.t975(1) == 12.706 and experiments.t975(30) == 2.042
    assert experiments.t975(31) == pytest.approx(2.0395, abs=2e-4)
    assert experiments.t975(120) == pytest.approx(1.9799, abs=2e-4)
    assert experiments.t975(10**6) == pytest.approx(1.95997, abs=1e-4)
    mean, lo, hi = experiments.mean_ci([1, 2, 3, 4, 5])
    assert mean == 3 and hi - mean == pytest.approx(2.776 * math.sqrt(2.5) / math.sqrt(5))
    assert experiments.mean_ci([2.0])[0] == 2.0 and math.isnan(experiments.mean_ci([2.0])[1])
    assert all(math.isnan(x) for x in experiments.mean_ci([]))


def compare_csv() -> pd.DataFrame:
    """Runs of weather_scale 1 (the reference) and 2 at two densities, five seeds, shuffled: the metric is
    100 + 10 x seed + (weather_scale - 1) x density, so each seed's pair differs by exactly the density."""
    rows = [{"density": d, "weather_scale": w, "seed": s, "road_condition": "snow", "flux": 100 + 10 * s + (w - 1) * d,
             "jam_fraction": 0.1 * w}
            for d in (10.0, 20.0) for w in (1.0, 2.0) for s in range(5)]
    return pd.DataFrame(rows).sample(frac=1.0, random_state=3).reset_index(drop=True)


def test_compare_pairs_each_run_with_the_same_seeds_reference():
    table = experiments.compare_table(compare_csv(), "weather_scale", 1.0, ["flux", "jam_fraction"])
    flux = table[table["metric"] == "flux"].set_index("density")
    assert list(flux.index) in ([10.0, 20.0], [20.0, 10.0]) and (flux["pairs"] == 5).all()
    for d in (10.0, 20.0):
        row = flux.loc[d]
        assert [row["diff"], row["diff_lo"], row["diff_hi"]] == pytest.approx([d, d, d])
        ratios = [(100 + 10 * s + d) / (100 + 10 * s) for s in range(5)]
        assert row["ratio"] == pytest.approx(np.mean(ratios)) and row["ratio_lo"] < row["ratio"] < row["ratio_hi"]
        assert row["ref_mean"] == pytest.approx(120.0)
    jam = table[table["metric"] == "jam_fraction"]
    assert np.allclose(jam["ratio"], 2.0) and np.allclose(jam["diff"], 0.1)


def test_compare_command(tmp_path, capsys):
    path = tmp_path / "s.csv"
    compare_csv().to_csv(path, index=False)
    experiments.main(["compare", str(path), "--ref", "weather_scale=1", "--metrics", "flux,jam_fraction"])
    printed = capsys.readouterr().out
    assert "density = 10, weather_scale = 2 (5 pairs)" in printed and "diff 10.0 [10.0, 10.0]" in printed
    table = pd.read_csv(tmp_path / "s_compare.csv")
    assert set(table["metric"]) == {"flux", "jam_fraction"} and len(table) == 4
    experiments.main(["compare", str(path), "--ref", "weather_scale=1", "--out", str(tmp_path / "default")])
    assert set(pd.read_csv(tmp_path / "default.csv")["metric"]) == {"flux", "jam_fraction"}  # those it has
    for argv, message in ((["--ref", "weather_scale=3"], "no run has it"), (["--ref", "n_lanes=1"], "no such column"),
                          (["--ref", "weather_scale=1", "--metrics", "nope"], "no column nope")):
        with pytest.raises(SystemExit):
            experiments.main(["compare", str(path), *argv])
        assert message in capsys.readouterr().err
