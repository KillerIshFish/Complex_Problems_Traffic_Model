"""The experiments CLI end to end: tiny runs and sweeps into a temporary folder, and bad arguments."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # plots.py draws on Agg canvases anyway; never open a GUI backend

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from roundabout_sim import experiments  # noqa: E402
from roundabout_sim.metrics import SECTORS, SPACETIME_BINS, Recorder  # noqa: E402
from tests.helpers import assert_png  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Metrics every time series must carry (measurements named like a parameter get "measured_").
TIMESERIES_COLUMNS = {
    "time", "cars", "queued", "measured_density", "measured_mean_speed", "mean_free_speed", "flux",
    "detector_flux", "stops_per_car_min", "jam_fraction", "speed_deficit", "n_jams", "jams_following",
    "jams_merge", "jams_exit", "jams_entry", "jams_other", "occupancy", "clustering", "order_param", "order_angle",
    "order_baseline", "spacing_gini", "emergency_per_min", "exits_per_min", "lane_changes_per_min", "mean_trip_time",
}
MEASURED_COLUMNS = {
    "measured_density", "measured_mean_speed", "mean_free_speed", "flux", "detector_flux",
    "stops_per_car_min", "stop_go_onset", "jam_fraction", "congestion_onset", "n_jams", "jams_following",
    "following_jam_share", "occupancy", "spacing_gini", "clustering", "order_param", "order_baseline",
    "emergency_per_min", "wave_speed",
}
BASELINE_COLUMNS = {f"baseline_{name}" for name in experiments.BASELINE_COLUMNS} | {"arm_jam_share"}


@pytest.mark.parametrize("flags, expect", [
    ([], dict(n_lanes=3, n_arms=4, density=50.0)),
    (["--n-lanes", "2", "--set", "n_arms=0", "--density", "70", "--init_spacing", "random"],
     dict(n_lanes=2, n_arms=0, density=70.0, init_spacing="random")),
], ids=["defaults", "closed-ring-flags"])
def test_run_writes_timeseries_spacetime_summary_and_dashboard(tmp_path, capsys, flags, expect):
    out = tmp_path / "sub" / "run"  # the output folder is created
    experiments.main(["run", "--duration", "20", "--seed", "1", "--out", str(out), *flags])
    printed = capsys.readouterr().out
    assert "Jammed cars (congestion)" in printed and "Stop-and-go (instability)" in printed
    assert "Harsh braking" in printed and "wrote" in printed
    assert "mergency" not in printed and "atural share" not in printed and "Baseline" not in printed

    ts = pd.read_csv(f"{out}_timeseries.csv")
    assert TIMESERIES_COLUMNS <= set(ts.columns)
    assert not {"density", "mean_speed", "speed_std"} & set(ts.columns)  # settings are not columns here
    lanes = expect["n_lanes"]
    assert f"lane{lanes - 1}_cars" in ts and f"lane{lanes}_cars" not in ts
    assert len(ts) == 41  # every 0.5 s from the first step to 20 s
    assert ts["time"].is_monotonic_increasing and ts["time"].iloc[-1] == pytest.approx(20.0)
    for col in ("stops_per_car_min", "flux", "clustering", "order_param", "jam_fraction", "spacing_gini",
                "order_baseline"):
        assert np.isfinite(ts[col]).all() and (ts[col] >= 0).all(), col
    for col in ("order_param", "jam_fraction", "spacing_gini", "order_baseline"):
        assert (ts[col] <= 1).all(), col
    assert ts["measured_density"].mean() == pytest.approx(expect["density"], rel=0.1)

    summary = pd.read_csv(f"{out}_summary.csv")
    assert len(summary) == 1
    row = summary.iloc[0]
    assert MEASURED_COLUMNS <= set(summary.columns)
    assert not BASELINE_COLUMNS & set(summary.columns)  # only with --baseline
    assert "natural_jam_share" not in summary.columns  # renamed: it is only an upper bound on natural jams
    assert row["seed"] == 1
    for name, value in expect.items():
        assert row[name] == value, name
    assert row["stops_per_car_min"] >= 0
    assert np.isnan(row["following_jam_share"]) or 0 <= row["following_jam_share"] <= 1
    assert np.isnan(row["congestion_onset"]) or 0 <= row["congestion_onset"] <= 20
    assert 0 <= row["spacing_gini"] < 1 and 0 < row["order_baseline"] < 1

    with np.load(f"{out}_spacetime.npz") as npz:
        times, speeds, occupancy = npz["times"], npz["speeds"], npz["occupancy"]
        assert speeds.shape == (len(ts), SPACETIME_BINS) and occupancy.shape == (len(ts), SECTORS)
        np.testing.assert_allclose(times, ts["time"])
        assert len(npz["bin_edges_deg"]) == SPACETIME_BINS + 1 and len(npz["sector_edges_deg"]) == SECTORS + 1
        assert len(npz["arm_deg"]) == expect["n_arms"] and len(npz["radii"]) == lanes
        assert float(npz["v_free"]) > 0
    assert np.nanmax(speeds) > 0 and np.all((occupancy >= 0) & (occupancy <= 1))

    assert_png(Path(f"{out}.png"))


def test_run_baseline_repeats_the_same_run_without_exits(tmp_path, capsys):
    out = tmp_path / "with_baseline"
    experiments.main(["run", "--duration", "60", "--seed", "3", "--baseline", "--out", str(out)])
    printed = capsys.readouterr().out
    assert "Baseline: the same run" in printed and "Jamming due to entrances/exits" in printed
    row = pd.read_csv(f"{out}_summary.csv").iloc[0]
    assert BASELINE_COLUMNS <= set(row.index)
    assert row["exit_prob"] == 0.1  # the settings recorded are the run's, not the baseline's
    assert row["jam_fraction"] >= experiments.JAM_PRESENT  # the defaults jam, so there is a share to report
    assert 0 <= row["arm_jam_share"] <= 1
    assert row["arm_jam_share"] == pytest.approx(
        experiments.arm_share(row["jam_fraction"], row["baseline_jam_fraction"]))

    # The baseline is exactly the run with exit_prob = 0: same seed, every other setting and duration.
    experiments.main(["run", "--duration", "60", "--seed", "3", "--set", "exit_prob=0",
                      "--out", str(tmp_path / "no_exits")])
    capsys.readouterr()
    plain = pd.read_csv(tmp_path / "no_exits_summary.csv").iloc[0]
    for name in experiments.BASELINE_COLUMNS:
        assert row[f"baseline_{name}"] == pytest.approx(plain[name], nan_ok=True), name
    assert_png(Path(f"{out}.png"))


@pytest.mark.parametrize("setting", ["n_arms=0", "exit_prob=0"])
def test_run_baseline_is_skipped_when_there_is_nothing_to_switch_off(tmp_path, capsys, setting):
    out = tmp_path / "skipped"
    experiments.main(["run", "--duration", "5", "--baseline", "--set", setting, "--out", str(out)])
    printed = capsys.readouterr().out
    assert "--baseline skipped" in printed and "Baseline: the same run" not in printed
    row = pd.read_csv(f"{out}_summary.csv").iloc[0]
    assert BASELINE_COLUMNS <= set(row.index)
    assert all(np.isnan(row[name]) for name in BASELINE_COLUMNS)
    assert_png(Path(f"{out}.png"))


def test_arm_share_is_the_jamming_the_exits_add():
    assert experiments.arm_share(0.4, 0.1) == pytest.approx(0.75)
    assert experiments.arm_share(0.2, 0.0) == 1.0
    assert experiments.arm_share(0.2, 0.3) == 0.0  # clipped: the run without exits jammed more
    assert np.isnan(experiments.arm_share(0.0, 0.0))  # no jamming to explain
    assert np.isnan(experiments.arm_share(experiments.JAM_PRESENT / 2, 0.0))  # (almost) no jamming
    assert np.isnan(experiments.arm_share(0.3, float("nan")))


def test_arm_share_table_compares_each_exit_chance_with_zero():
    df = pd.DataFrame({"exit_prob": [0.0, 0.0, 0.1, 0.1, 0.2, 0.2], "density": [40.0, 80.0] * 3,
                       "jam_fraction": [0.0, 0.2, 0.1, 0.4, 0.0, 0.5]})
    table = experiments.arm_share_table(df, ["exit_prob", "density"])
    assert list(table.columns) == ["exit_prob=0.1", "exit_prob=0.2"]
    assert list(table.index) == [40.0, 80.0]
    np.testing.assert_allclose(table.loc[80.0], [0.5, 0.6])
    assert table.loc[40.0, "exit_prob=0.1"] == 1.0 and np.isnan(table.loc[40.0, "exit_prob=0.2"])
    # exit_prob as the only swept parameter, seeds averaged first
    alone = experiments.arm_share_table(pd.DataFrame({"exit_prob": [0.0, 0.0, 0.1, 0.1],
                                                      "jam_fraction": [0.1, 0.1, 0.3, 0.5]}), ["exit_prob"])
    assert alone.loc["all", "exit_prob=0.1"] == pytest.approx(0.75)
    assert experiments.arm_share_table(df[df["exit_prob"] > 0], ["exit_prob", "density"]) is None  # no 0
    assert experiments.arm_share_table(df, ["density"]) is None  # exit_prob not swept


def test_arm_share_table_leaves_out_points_without_a_baseline(capsys):
    """A plan need not pair every exit_prob > 0 run with an exit_prob = 0 run: such points are left out,
    a cell without a run is blank (not '-', which means under 2 % jammed), and with nothing to compare
    nothing is printed (no empty pandas frame)."""
    df = pd.DataFrame({"exit_prob": [0.0, 0.1, 0.0, 0.2, 0.0, 0.3],
                       "density": [80.0, 80.0, 90.0, 100.0, 70.0, 70.0],
                       "jam_fraction": [0.1, 0.4, 0.2, 0.7, 0.0, 0.01], "traffic_mode": "closed", "n_arms": 4})
    table = experiments.arm_share_table(df, ["exit_prob", "density"])
    assert list(table.index) == [80.0, 70.0]  # 90: no run above 0; 100: no run at 0
    assert list(table.columns) == ["exit_prob=0.1", "exit_prob=0.3"]  # 0.2 ran only at 100
    assert table.loc[80.0, "exit_prob=0.1"] == pytest.approx(0.75)
    experiments._print_arm_shares(df, ["exit_prob", "density"])
    printed = capsys.readouterr().out
    assert "- = under 2 % jammed; blank = no run there:" in printed
    assert "(2 points left out: no exit_prob 0 run to compare with, or no run above 0)" in printed
    rows = {line.split()[0]: line.split()[1:] for line in printed.splitlines() if line[:2] in ("80", "70")}
    assert rows == {"80.0": ["75"], "70.0": ["-"]}  # the blank cells print as nothing
    # Nothing to compare: exit_prob 0 only (a single --vary value), or the 0 runs elsewhere (open, a plan).
    for frame, keys in ((df[df["exit_prob"] == 0], ["exit_prob", "density"]),
                        (df.assign(traffic_mode=["closed", "open", "open", "open", "open", "open"]),
                         ["exit_prob", "traffic_mode"])):
        assert experiments.arm_share_table(frame[frame["traffic_mode"] == "closed"], keys) is None
        experiments._print_arm_shares(frame, keys)
        printed = capsys.readouterr().out
        assert "Empty DataFrame" not in printed and "Jamming due to" not in printed


def _jam_frame(times, jam) -> pd.DataFrame:
    return pd.DataFrame({"time": times, "jam_fraction": jam})


def test_congestion_onset_is_when_the_30_s_mean_of_jammed_cars_passes_30_percent():
    t = np.arange(0.5, 200.0, 0.5)
    # Everyone jammed from 50 s: the trailing 30 s (60 samples) mean passes 0.3 with the 19th jammed sample.
    assert experiments.congestion_onset(_jam_frame(t, np.where(t >= 50, 1.0, 0.0))) == pytest.approx(59.0)
    # A start that is already jammed scores ~0 (the mean is taken over the samples so far).
    assert experiments.congestion_onset(_jam_frame(t, np.full(t.size, 0.8))) == pytest.approx(0.5)
    # Never: steady 25 % jammed, or a 5 s burst that the 30 s mean smooths away.
    assert np.isnan(experiments.congestion_onset(_jam_frame(t, np.full(t.size, 0.25))))
    burst = np.where((t >= 100) & (t < 105), 1.0, 0.0)
    assert np.isnan(experiments.congestion_onset(_jam_frame(t, burst)))
    rolled = experiments.rolling_jam_fraction(_jam_frame(t, burst))
    assert rolled.max() == pytest.approx(10 / 60) and rolled[0] == 0.0


def test_sweep_figure_draws_every_chart_even_for_columns_it_lacks():
    from roundabout_sim.plots import sweep_figure

    df = pd.DataFrame({"density": [20.0, 40.0], "seed": [0, 0], "flux": [0.5, 0.9]})  # e.g. an older CSV
    fig = sweep_figure(df, "density", None, x_label="Density", series_label=str, title="t", subtitle="s")
    assert len(fig.axes) == 14  # header + 13 charts


def test_sweep_writes_one_row_per_run_and_a_figure(tmp_path, capsys):
    out = tmp_path / "sweep"
    experiments.main(["sweep", "--x", "density=30,60", "--seeds", "1", "--warmup", "5", "--measure", "10",
                      "--workers", "1", "--out", str(out)])
    printed = capsys.readouterr().out
    assert "2 runs" in printed and "Means across seeds" in printed

    df = pd.read_csv(f"{out}.csv")
    assert len(df) == 2
    assert list(df.columns[:2]) == ["density", "seed"]  # swept setting first, then the seed
    assert MEASURED_COLUMNS <= set(df.columns)
    assert list(df["density"]) == [30.0, 60.0] and list(df["seed"]) == [0, 0]
    assert (df["n_lanes"] == 3).all()  # the settings held fixed are recorded too
    np.testing.assert_allclose(df["measured_density"], df["density"], rtol=0.1)
    assert (df["stops_per_car_min"] >= 0).all()
    assert_png(Path(f"{out}.png"))


def test_sweep_with_series_and_fixed_settings(tmp_path, capsys):
    out = tmp_path / "series"
    experiments.main(["sweep", "--x", "reaction_time=0.2:0.8:0.6", "--series", "n_lanes=2,1", "--set", "n_arms=0",
                      "--density", "40", "--seeds", "1", "--warmup", "5", "--measure", "10", "--workers", "1",
                      "--out", str(out)])
    capsys.readouterr()
    df = pd.read_csv(f"{out}.csv")
    assert list(df.columns[:3]) == ["n_lanes", "reaction_time", "seed"]
    # Rows in the order the values were given (series 2 before 1), x ascending within each.
    assert list(zip(df["n_lanes"], df["reaction_time"])) == [(2, 0.2), (2, 0.8), (1, 0.2), (1, 0.8)]
    assert (df["n_arms"] == 0).all() and (df["density"] == 40).all()
    assert (df["exits_per_min"] == 0).all()  # closed ring: nobody leaves
    assert_png(Path(f"{out}.png"))


def test_sweep_over_initial_spacing_and_exit_chance_prints_the_notes(tmp_path, capsys):
    out = tmp_path / "start"
    experiments.main(["sweep", "--x", "init_spacing=uniform,clustered", "--series", "exit_prob=0,0.1",
                      "--seeds", "1", "--warmup", "0", "--measure", "5", "--workers", "1", "--out", str(out)])
    printed = capsys.readouterr().out
    assert experiments.INIT_SPACING_NOTE in printed
    assert "Jamming due to entrances/exits" in printed and "exit_prob=0.1" in printed
    df = pd.read_csv(f"{out}.csv")
    assert list(df["init_spacing"]) == ["uniform", "clustered"] * 2
    assert (df.loc[df["init_spacing"] == "clustered", "spacing_gini"]
            > df.loc[df["init_spacing"] == "uniform", "spacing_gini"].max()).all()  # a packed queue is uneven
    assert_png(Path(f"{out}.png"))


def test_parallel_sweep_matches_the_serial_one(tmp_path, capsys, no_figures):
    base = ["sweep", "--x", "density=30,60", "--seeds", "1", "--warmup", "2", "--measure", "5"]
    experiments.main([*base, "--workers", "2", "--out", str(tmp_path / "parallel")])
    experiments.main([*base, "--workers", "1", "--out", str(tmp_path / "serial")])
    assert "on 2 worker(s)" in capsys.readouterr().out
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "parallel.csv"), pd.read_csv(tmp_path / "serial.csv"))


@pytest.mark.parametrize("argv, message", [
    ([], "required"),
    (["fly"], "invalid choice"),
    (["run", "--duration", "0"], "--duration > 0"),
    (["run", "--interval", "-1"], "--interval > 0"),
    (["run", "--set", "bogus=1"], "unknown parameter"),
    (["run", "--set", "n_lanes"], "NAME=VALUE"),
    (["run", "--set", "n_lanes="], "NAME=VALUE"),
    (["run", "--density", "lots"], "bad value 'lots' for density"),
    (["run", "--set", "init_spacing=diagonal"], "init_spacing must be one of"),
    (["run", "--init-spacing", "diagonal"], "invalid choice"),
    (["run", "--dur", "5"], "unrecognized arguments"),  # no abbreviations
    (["sweep"], "--x"),
    (["sweep", "--x", "density"], "NAME=VALUE"),
    (["sweep", "--x", "density=10:5:1"], "stop >= start"),
    (["sweep", "--x", "density=10:50:0"], "step > 0"),
    (["sweep", "--x", "density=1:2"], "start:stop:step"),
    (["sweep", "--x", "density=,"], "no values"),
    (["sweep", "--x", "density=a,b"], "bad value"),
    (["sweep", "--x", "density=10,20", "--series", "density=1,2"], "must be different"),
    (["sweep", "--x", "density=10,20", "--series", "n_lanes=1,2,3,4,5,6,7,8,9"], "at most 8"),
    (["sweep", "--x", "density=10,20", "--seeds", "0"], "--seeds >= 1"),
    (["sweep", "--x", "density=10,20", "--measure", "0"], "--measure > 0"),
    (["sweep", "--x", "density=10,20", "--warmup", "-5"], "--warmup >= 0"),
])
def test_bad_arguments_exit_cleanly(tmp_path, capsys, argv, message):
    if argv and argv[0] in ("run", "sweep"):
        argv = [*argv, "--out", str(tmp_path / "out")]
    with pytest.raises(SystemExit) as exc:
        experiments.main(argv)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "error:" in err and message in err
    assert not any(tmp_path.iterdir())  # failed before simulating or writing anything


def test_help_lists_the_measured_columns(capsys):
    with pytest.raises(SystemExit) as exc:
        experiments.main(["run", "--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "stops_per_car_min" in text and "--reaction-distance" in text and "--set NAME=VALUE" in text
    for name in ("following_jam_share", "UPPER BOUND", "congestion_onset", "spacing_gini", "order_baseline",
                 "arm_jam_share", "Harsh braking (episodes/min)", "--baseline"):
        assert name in text, name
    assert "natural_jam_share" not in text and "mergency braking" not in text


def test_values_outside_the_slider_range_are_allowed_with_a_note(capsys):
    parser = experiments.build_parser()
    args = parser.parse_args(["run", "--set", "density=400", "--n-lanes", "2"])
    assert experiments._overrides(args, args.command_parser) == {"density": 400.0, "n_lanes": 2}
    assert "outside the usual range" in capsys.readouterr().out


def test_command_line_entry_point_reports_errors_without_a_traceback(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, "-m", "roundabout_sim.experiments", "run", "--duration", "-1",
         "--out", str(tmp_path / "out")],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 2
    assert "error:" in result.stderr and "Traceback" not in result.stderr
    assert not any(tmp_path.iterdir())


def _sampled_by_recorder(start: float, measure: float, dt: float, interval: float) -> bool:
    """Whether a Recorder stepped as experiments.advance steps it samples at or after ``start``."""
    class Clock:
        time, generation = 0.0, 0

    rec = Recorder.__new__(Recorder)
    rec.sim, rec.interval, rec._next, rec._generation, times = Clock(), interval, 0.0, 0, []
    rec.record = lambda: times.append(rec.sim.time)
    for _ in range(int(round(start / dt)) + int(round(measure / dt))):
        rec.sim.time += dt
        rec.update()
    return any(t >= start for t in times)


@pytest.mark.parametrize("dt", [0.1, 0.05, 0.25])
def test_window_sampled_matches_the_recorder(dt):
    cases = [(start, measure, interval) for start in (0.0, 0.3, 1.0, 3.3, 10.0)
             for measure in (0.1, 0.25, 0.5, 2.0, 5.0) for interval in (0.05, 0.3, 0.5, 2.0, 10.0)]
    for start, measure, interval in cases:
        assert experiments.window_sampled(start, measure, dt, interval) == \
            _sampled_by_recorder(start, measure, dt, interval), (start, measure, interval)


def test_a_window_without_a_sample_is_refused_before_anything_runs(tmp_path, capsys):
    """A measured window that falls between two samples would give a summary of nothing (NaN metrics):
    the parser says so instead, for a sweep and for the second half of a run."""
    for argv, why in ((["sweep", "--x", "density=20,30", "--warmup", "1", "--measure", "0.2", "--interval", "5"],
                       "no sample in the measured window (1-1.2 s at a time step of 0.1 s, sampling every "
                       "--interval 5 s): make --measure longer or --interval shorter"),
                      (["run", "--duration", "2", "--interval", "5"],
                       "no sample in the second half of the run, which the summary covers,")):
        with pytest.raises(SystemExit):
            experiments.main([*argv, "--out", str(tmp_path / "x")])
        assert why in capsys.readouterr().err
    assert not list(tmp_path.iterdir())  # nothing ran, nothing was written
