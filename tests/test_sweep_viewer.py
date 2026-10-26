"""Tests of the sweep viewer: reading and classifying sweep CSVs, the aggregation behind every
view, the sweep command and its subprocess, and the pygame window driven headlessly (SDL's dummy
video driver) through every mode at the minimum, a medium and a large window size.

The CSVs are made up in the tests (the values follow simple formulas, so the expected means are
known); one test runs a real two-run sweep through the window.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pygame
import pytest

from roundabout_sim import sweep_viewer as sv
from roundabout_sim import ui
from roundabout_sim.config import SimConfig
from tests.helpers import click, down, key, move, up, wheel, widget

SIZES = ((1100, 700), (1280, 720), (1920, 1080))
SIZE = (1280, 720)
EXIT = SimConfig().exit_prob  # exit_prob of the runs that do not vary it
PLACEHOLDER = "a measured column of the sweep CSV."  # the help of a column nobody describes


# ------------------------------------------------------------------ CSVs
def default_settings() -> dict:
    cfg = SimConfig()
    return {f.name: getattr(cfg, f.name) for f in fields(SimConfig) if f.name != "seed"}


def flux_of(density: float, n_lanes: int = 1, exit_prob: float = 0.0, seed: int = 0) -> float:
    """A made-up metric with a known value for every run (the seed shifts it by 0.01 per seed)."""
    return 0.01 * density * n_lanes * (1 - exit_prob) + 0.01 * seed


def grid_frame(axes: dict, seeds: int = 2, settings: dict | None = None) -> pd.DataFrame:
    """One row per combination of ``axes`` values and seed: every setting, the seed, then metrics."""
    rows = []
    names = list(axes)
    for values in itertools.product(*axes.values(), range(seeds)):
        point, seed = dict(zip(names, values[:-1])), values[-1]
        row = {**default_settings(), **(settings or {}), **point, "seed": seed}
        flux = flux_of(row["density"], row["n_lanes"], row["exit_prob"], seed)
        rows.append({**row, "flux": flux, "jam_fraction": min(row["density"] / 200, 1.0),
                     "congestion_onset": np.nan if row["density"] < 60 else 30.0 + seed,
                     "mean_entry_delay": 2.0 + seed, "entry_los": "A" if seed == 0 else "B"})
    frame = pd.DataFrame(rows)
    return frame[names + ["seed"] + [c for c in frame.columns if c not in names and c != "seed"]]


def write(frame: pd.DataFrame, path: Path, meta: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    if meta is not None:
        (path.parent / f"{path.stem}_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def sample_frame(n: int = 120, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        density, reaction = rng.uniform(10, 150), rng.uniform(0.1, 1.5)
        motive = ("advantage", "random")[i % 2]
        rows.append({**default_settings(), "density": density, "reaction_time": reaction,
                     "lane_change_motive": motive, "seed": i, "flux": 0.01 * density + reaction,
                     "jam_fraction": density / 150, "entry_los": "A"})
    return pd.DataFrame(rows)


@pytest.fixture
def csvs(tmp_path) -> dict[str, Path]:
    """A results folder with a 1-setting, a 2-setting, a 3-setting and a random-sample sweep, an
    old-format CSV and files that are not sweeps."""
    res = tmp_path / "results"
    paths = {
        "one": write(grid_frame({"density": [20.0, 40.0, 60.0, 80.0]}), res / "one.csv"),
        "two": write(grid_frame({"n_lanes": [1, 2], "density": [20.0, 50.0, 80.0]}), res / "two.csv"),
        "three": write(grid_frame({"density": [20.0, 50.0, 80.0, 110.0], "n_lanes": [1, 2, 3],
                                   "exit_prob": [0.0, 0.1]}, seeds=2), res / "three.csv"),
        "sample": write(sample_frame(), res / "sample.csv"),
    }
    old = grid_frame({"lane_change_motive": ["random", "advantage"], "reaction_time": [0.2, 0.6, 1.0]})
    old = old.drop(columns=[c for c in ("follow_up", "traffic_mode", "entry_rate", "demand_profile",
                                        "demand_imbalance", "max_queue", "entry_los", "mean_entry_delay")
                            if c in old] + [c for c in old.columns if c.startswith("truck") or "road" in c])
    paths["old"] = write(old, res / "old.csv")
    write(pd.DataFrame({"time": [0.0, 0.5], "flux": [0.1, 0.2]}), res / "run_timeseries.csv")
    write(grid_frame({"density": [20.0]}, seeds=1), res / "run_summary.csv")
    write(pd.DataFrame({"a": [1, 2]}), res / "notes.csv")
    now = time.time()
    for k, name in enumerate(("old", "one", "two", "three", "sample")):  # "sample" is the newest
        os.utime(paths[name], (now + k, now + k))
    return paths


# ------------------------------------------------------------------ viewer helpers
@pytest.fixture
def make_viewer(tmp_path):
    """Factory for headless viewers writing into tmp_path/results; pygame is shut down afterwards
    and a sweep left running is stopped."""
    made = []

    def make(csv=None, size=SIZE, **kwargs) -> sv.SweepViewer:
        kwargs.setdefault("results_dir", tmp_path / "results")
        viewer = sv.SweepViewer(csv, size=size, **kwargs)
        made.append(viewer)
        return viewer

    yield make
    for viewer in made:
        viewer.close()
    pygame.quit()


def frame(viewer: sv.SweepViewer, *events: pygame.event.Event) -> None:
    """One pass of ``SweepViewer.run``'s loop body."""
    for event in events:
        assert viewer.handle(event), f"{event} asked to quit"
    if viewer.screen.get_size() != viewer._layout_size:
        viewer.layout()
    viewer.update()
    viewer.draw()
    pygame.display.flip()


def param_pos(viewer: sv.SweepViewer, name: str) -> tuple[int, int]:
    """Centre of a setting in the new-sweep list, scrolled into view first."""
    region = viewer.regions["params"]
    w = widget(viewer, "param", name)
    if not region.contains(w.rect):
        viewer.scroll["params"] += w.rect.y - region.y
        viewer._layout_panel()
        w = widget(viewer, "param", name)
    assert region.contains(w.rect)
    return w.rect.center


def arrow(viewer: sv.SweepViewer, name: str, side: int = 1) -> tuple[int, int]:
    """Centre of a cycler's left (0) or right (1) arrow."""
    return widget(viewer, "cycler", name).options[side].center


def wait_for(viewer: sv.SweepViewer, done, timeout: float = 60.0) -> None:
    end = time.monotonic() + timeout
    while not done() and time.monotonic() < end:
        frame(viewer)
        time.sleep(0.05)
    assert done(), "timed out"


def slow_job(lines: int = 2, total: int = 9) -> list[str]:
    """A stand-in sweep: prints ``lines`` progress lines like experiments sweep, then waits."""
    code = (f"print('{total} runs (3 x 3 series x 1 seeds) on 1 worker(s)', flush=True)\n"
            f"for i in range(1, {lines + 1}):\n"
            f"    print(f'[{{i:3d}}/{total}] density=20 seed=0: flux 0.2 cars/s ({{i}} s, ~{{{total} - i}} s left)',"
            " flush=True)\n"
            "import time; time.sleep(60)\n")
    return [sys.executable, "-c", code]


# ================================================================== loading
def test_classify_two_setting_grid(csvs):
    data = sv.load_sweep(csvs["two"])
    assert [v.name for v in data.variables] == ["n_lanes", "density"]
    assert data.var("n_lanes").values == (1, 2) and data.var("density").values == (20.0, 50.0, 80.0)
    assert data.var("density").numeric
    assert data.design == "grid" and data.runs == 12
    assert data.fixed["n_arms"] == 4 and data.fixed["lane_change_motive"] == "advantage"
    assert "seed" not in data.metrics and "seed" not in data.fixed
    assert data.metrics[:2] == ["flux", "jam_fraction"]  # the experiments' headline charts first
    assert {"congestion_onset", "mean_entry_delay"} <= set(data.metrics)
    assert data.text_metrics == ["entry_los"]
    assert not set(data.metrics) & set(data.settings)
    # Without a meta file the x is the numeric setting with the most values, the rest are series.
    assert sv.axis_order(data) == ["density", "n_lanes"]


def test_old_format_csv_and_text_settings(csvs):
    data = sv.load_sweep(csvs["old"])
    assert "follow_up" not in data.frame and not data.meta
    motive = data.var("lane_change_motive")
    assert not motive.numeric and motive.values == ("advantage", "random")  # the choices' order, not the file's
    assert data.design == "grid" and data.text_metrics == []
    assert sv.axis_order(data)[0] == "reaction_time"
    stats = sv.grid_stats(data.frame, "flux", motive)
    assert stats.mean.shape == (1, 2) and np.all(stats.runs == 6)


def test_text_metric_stays_text_when_empty(tmp_path):
    frame = grid_frame({"density": [20.0, 40.0]})
    frame["entry_los"] = np.nan
    data = sv.load_sweep(write(frame, tmp_path / "s.csv"))
    assert data.text_metrics == ["entry_los"] and "entry_los" not in data.metrics


def test_meta_file_gives_design_and_order(tmp_path):
    frame = grid_frame({"n_lanes": [1, 2], "density": [20.0, 50.0, 80.0]})
    meta = {"version": 1, "design": "grid", "varied": [{"name": "n_lanes", "values": [1, 2]},
                                                       {"name": "density", "values": [20, 50, 80]}],
            "fixed": {}, "preset": "fhwa_single", "seeds": 2, "base_seed": 0, "warmup": 60.0, "measure": 120.0,
            "interval": 0.5, "command": "python -m roundabout_sim.experiments sweep ..."}
    data = sv.load_sweep(write(frame, tmp_path / "s.csv", meta))
    assert data.ordered and sv.axis_order(data) == ["n_lanes", "density"]  # as the meta file lists them
    assert data.meta["preset"] == "fhwa_single"
    # A meta file that is not JSON is ignored.
    (tmp_path / "s_meta.json").write_text("{not json", encoding="utf-8")
    assert sv.load_sweep(tmp_path / "s.csv").meta == {}


def test_not_a_sweep_and_the_file_list(csvs, tmp_path):
    res = tmp_path / "results"
    for name in ("run_timeseries.csv", "notes.csv"):
        with pytest.raises(ValueError):
            sv.load_sweep(res / name)
    with pytest.raises(ValueError):
        sv.load_sweep(res / "missing.csv")
    found = sv.find_sweeps(res)
    assert [p.name for p in found] == ["sample.csv", "three.csv", "two.csv", "one.csv", "old.csv"]
    assert sv.find_sweeps(tmp_path / "nowhere") == []


def test_design_detection(tmp_path):
    regular = grid_frame({"density": [float(d) for d in range(10, 310, 10)]}, seeds=1)  # 30 even values
    assert sv.classify(regular).design == "grid"
    assert sv.classify(sample_frame()).design == "sample"
    plan = grid_frame({"density": [20.0, 50.0], "n_lanes": [1, 2]}).iloc[:-2]  # one combination missing
    assert sv.classify(plan).design == "plan"
    single = grid_frame({"density": [50.0]}, seeds=3)
    data = sv.classify(single)
    assert data.design == "single" and [v.name for v in data.variables] == ["seed"]
    no_seed = sv.classify(single.drop(columns="seed"))
    assert [v.name for v in no_seed.variables] == [sv.RUN] and no_seed.var(sv.RUN).values == (1, 2, 3)


def test_uneven_grids_are_grids_and_samples_with_seeds_are_samples():
    """A full grid is a grid however unevenly spaced and with any number of seeds (the New sweep form
    snaps its values to the slider step: 1, 26, 51, 76, 101, 126, 151, 175, ...); a random sample is
    a sample with one seed or several."""
    for points in (13, 25, 41):
        density = sv.FormVar.new("density")
        density.points = points
        values = density.values()
        assert len(set(np.diff(values))) > 1  # unevenly spaced
        for seeds in (1, 2, 3):
            assert sv.classify(grid_frame({"density": values}, seeds=seeds)).design == "grid", (points, seeds)
    typed = [5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0, 200.0]  # a comma list
    for seeds in (1, 2):
        assert sv.classify(grid_frame({"density": typed}, seeds=seeds)).design == "grid"
        assert sv.classify(grid_frame({"n_lanes": [1, 2], "density": typed}, seeds=seeds)).design == "grid"
    # 20 random points over three settings, repeated for 3 seeds: still a sample.
    rng = np.random.default_rng(1)
    points = [{"density": rng.uniform(10, 120), "exit_prob": rng.uniform(0, 0.4), "politeness": rng.uniform(0, 1)}
              for _ in range(20)]
    rows = [{**default_settings(), **p, "seed": k, "flux": 0.01 * p["density"]} for k in range(3) for p in points]
    assert sv.classify(pd.DataFrame(rows)).design == "sample"
    # One setting drawn at random (values off the slider step): a sample, which still has a lines view.
    one = pd.DataFrame([{**default_settings(), "density": d, "seed": 0, "flux": 0.01 * d}
                        for d in rng.uniform(10, 150, 20)])
    assert sv.classify(one).design == "sample"


def test_axis_order_follows_the_csv_layout(tmp_path):
    """Without a meta file, experiments' layout (--series, --x, seed, ...) gives the axes, even when
    the series has more values than the x; a varying column there that is no setting today is one."""
    swapped = grid_frame({"n_lanes": [1, 2, 3], "exit_prob": [0.0, 0.2]}, seeds=1)
    data = sv.classify(swapped)
    assert data.layout == ["exit_prob", "n_lanes"] and sv.axis_order(data) == ["exit_prob", "n_lanes"]
    future = swapped.copy()
    future.insert(0, "future_setting", [0.1, 0.2] * 3)
    data = sv.classify(future)
    assert "future_setting" in data.settings and "future_setting" not in data.metrics
    assert {v.name for v in data.variables} == {"future_setting", "n_lanes", "exit_prob"}
    # Other layouts: the numeric setting with the most values is the x.
    shuffled = swapped[["seed"] + [c for c in swapped.columns if c != "seed"]]
    assert sv.axis_order(sv.classify(shuffled)) == ["n_lanes", "exit_prob"]


# ================================================================== aggregation
def test_mean_min_max_across_seeds(csvs):
    data = sv.load_sweep(csvs["two"])
    x, series = data.var("density"), data.var("n_lanes")
    st = sv.grid_stats(data.frame, "flux", x, series)
    assert st.mean.shape == (2, 3)
    for k, lanes in enumerate(series.values):
        for j, density in enumerate(x.values):
            runs = [flux_of(density, lanes, EXIT, seed) for seed in range(2)]
            assert st.mean[k, j] == pytest.approx(np.mean(runs))
            assert st.lo[k, j] == pytest.approx(min(runs)) and st.hi[k, j] == pytest.approx(max(runs))
            assert st.n[k, j] == 2 and st.runs[k, j] == 2
    # NaN values (never reached) are left out of the mean but counted as runs.
    onset = sv.grid_stats(data.frame, "congestion_onset", x)
    assert np.isnan(onset.mean[0, 0]) and onset.n[0, 0] == 0 and onset.runs[0, 0] == 4
    assert onset.mean[0, 2] == pytest.approx(30.5) and onset.n[0, 2] == 4


def test_slicing_versus_averaging(csvs):
    data = sv.load_sweep(csvs["three"])
    df, x = data.frame, data.var("density")
    sliced = sv.grid_stats(sv.select(df, {"exit_prob": 0.1, "n_lanes": 2}), "flux", x)
    for j, density in enumerate(x.values):
        expect = [flux_of(density, 2, 0.1, seed) for seed in range(2)]
        assert sliced.mean[0, j] == pytest.approx(np.mean(expect)) and sliced.runs[0, j] == 2
    averaged = sv.grid_stats(sv.select(df, {"exit_prob": None, "n_lanes": None}), "flux", x)
    for j, density in enumerate(x.values):
        pooled = [flux_of(density, n, e, seed) for n in (1, 2, 3) for e in (0.0, 0.1) for seed in range(2)]
        assert averaged.mean[0, j] == pytest.approx(np.mean(pooled)) and averaged.runs[0, j] == 12
        assert averaged.lo[0, j] == pytest.approx(min(pooled)) and averaged.hi[0, j] == pytest.approx(max(pooled))
    # A categorical slice works the same way.
    old = sv.load_sweep(csvs["old"])
    random_only = sv.select(old.frame, {"lane_change_motive": "random"})
    assert set(random_only["lane_change_motive"]) == {"random"} and len(random_only) == 6


def test_heatmap_matrix(csvs):
    data = sv.load_sweep(csvs["three"])
    x, y = data.var("density"), data.var("n_lanes")
    st = sv.grid_stats(sv.select(data.frame, {"exit_prob": 0.0}), "flux", x, y)
    assert st.mean.shape == (3, 4)  # rows = Y values, columns = X values
    for i, lanes in enumerate(y.values):
        for j, density in enumerate(x.values):
            assert st.mean[i, j] == pytest.approx(np.mean([flux_of(density, lanes, 0.0, s) for s in range(2)]))
    # A plan without some combinations leaves those cells empty.
    plan = data.frame[~((data.frame["density"] == 110.0) & (data.frame["n_lanes"] == 3))]
    st = sv.grid_stats(plan, "flux", x, y)
    assert np.isnan(st.mean[2, 3]) and st.runs[2, 3] == 0 and st.runs[0, 0] == 4


def test_binned_means():
    x = np.arange(10, dtype=float)
    y = 2 * x
    centres, means, mins, maxs, counts, edges = sv.binned_means(x, y, bins=5)
    assert np.allclose(edges, np.linspace(0, 9, 6))
    assert counts.tolist() == [2, 2, 2, 2, 2]
    assert np.allclose(means, [1, 5, 9, 13, 17]) and np.allclose(mins, [0, 4, 8, 12, 16])
    assert np.allclose(maxs, [2, 6, 10, 14, 18])
    assert np.allclose(centres, [0.5, 2.5, 4.5, 6.5, 8.5])  # the mean x of each bin's runs, not its middle
    # NaNs are dropped; an empty bin is NaN; the default is about sqrt(n) bins.
    xs = np.array([0.0, 0.1, 0.2, 9.8, 10.0, np.nan])
    _, means, _, _, counts, _ = sv.binned_means(xs, np.array([1.0, 2.0, 3.0, 4.0, np.nan, 5.0]), bins=4)
    assert counts.tolist() == [3, 0, 0, 1] and np.isnan(means[1]) and means[0] == pytest.approx(2.0)
    assert len(sv.binned_means(np.arange(100.0), np.arange(100.0))[0]) == 10
    one = sv.binned_means(np.full(3, 5.0), np.array([1.0, 2.0, 6.0]))
    assert one[1].tolist() == [3.0] and one[4].tolist() == [3]


def test_scatter_bins_as_the_sample_figure_does(make_viewer, tmp_path, monkeypatch):
    """One binning (plots.binned_stats) and one rule for when to bin (plots.BIN_ABOVE): the viewer's
    scatter of a random sample draws the means the sample figure draws."""
    from roundabout_sim import plots

    binned = []
    real = sv.binned_means
    monkeypatch.setattr(sv, "binned_means", lambda *a, **k: binned.append(a) or real(*a, **k))

    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 10, 40), rng.uniform(0, 1, 40)
    centres, means, _, _, counts, _ = sv.binned_means(x, y)
    fig_centres, fig_means = plots._bin_means(x, y, plots.BIN_ABOVE["sample"])
    assert np.allclose(centres[counts > 0], fig_centres) and np.allclose(means[counts > 0], fig_means)
    # A sample with 8 distinct values (more than the figure's 6): binned in the viewer too.
    frame_ = sample_frame(n=8)
    viewer = make_viewer(write(frame_, tmp_path / "results" / "s8.csv", {"design": "sample", "varied": [
        {"name": "density"}, {"name": "reaction_time"}, {"name": "lane_change_motive"}]}))
    assert viewer.data.design == "sample" and viewer.view == "scatter"
    viewer.set_x("density")
    binned.clear()
    frame(viewer, move(viewer.plot_rect.center))
    assert len(viewer.data.var("density").values) == 8 > plots.BIN_ABOVE["sample"] and binned  # binned, as in the PNG


def test_text_counts(csvs):
    frame = sv.load_sweep(csvs["two"]).frame
    assert sv.text_counts(frame, "entry_los") == "A ×6, B ×6"
    assert sv.text_counts(frame, "missing") == ""
    # The "-" experiments writes for no level of service (closed mode) is no value.
    closed = pd.DataFrame({"entry_los": ["-", "A", "-", " ", np.nan, "–"]})
    assert sv.text_counts(closed, "entry_los") == "A ×1"
    assert sv.text_value("-") is None and sv.text_value(np.nan) is None and sv.text_value(" B ") == "B"


# ================================================================== the sweep command
def test_command_for_one_two_three_settings(tmp_path):
    out = tmp_path / "sweep"
    common = dict(seeds=2, warmup=60, measure=120.0, out=out)
    one = sv.sweep_command([("density", [20.0, 40.0, 60.0, 80.0, 100.0])], **common)
    assert one[:5] == [sys.executable, "-u", "-m", "roundabout_sim.experiments", "sweep"]
    assert one[5:7] == ["--x", "density=20:100:20"]
    assert "--series" not in one and "--vary" not in one
    for flag, value in (("--seeds", "2"), ("--seed", "0"), ("--warmup", "60"), ("--measure", "120"),
                        ("--out", str(out))):
        assert one[one.index(flag) + 1] == value
    two = sv.sweep_command([("density", [20.0, 50.0]), ("n_lanes", [1, 2, 3])], **common)
    assert two[5:9] == ["--x", "density=20,50", "--series", "n_lanes=1,2,3"]
    three = sv.sweep_command([("density", [20.0, 40.0, 60.0]), ("n_lanes", [1, 2]),
                              ("lane_change_motive", ["advantage", "random"])], python="py", **common)
    assert three[0] == "py"
    assert three[5:11] == ["--vary", "density=20:60:20", "--vary", "n_lanes=1,2",
                           "--vary", "lane_change_motive=advantage,random"]  # one --vary each, in order
    assert "--x" not in three and "--series" not in three
    # The experiments CLI takes the commands as they are (the 3-setting one: test_three_setting_command_parses).
    from roundabout_sim import experiments
    parser = experiments.build_parser()
    parsed = parser.parse_args(two[4:])
    assert parsed.x == ("density", "20,50") and parsed.series == ("n_lanes", "1,2,3") and parsed.seeds == 2
    assert parser.parse_args(one[4:]).x == ("density", "20:100:20")


def test_three_setting_command_parses(tmp_path):
    """The 3-setting command against the experiments parser: one --vary each, in order."""
    from roundabout_sim import experiments
    parser = experiments.build_parser()
    three = sv.sweep_args([("density", [20.0, 40.0, 60.0]), ("n_lanes", [1, 2]),
                           ("lane_change_motive", ["advantage", "random"])], seeds=1, warmup=0, measure=10,
                          out=tmp_path / "s")
    vary = parser.parse_args(three).vary
    assert [name for name, _ in vary] == ["density", "n_lanes", "lane_change_motive"]


def test_command_base_settings_and_limits(tmp_path):
    base = {"exit_prob": 0.2, "radius": SimConfig().radius, "density": 70.0, "init_spacing": "random"}
    argv = sv.sweep_args([("density", [20.0, 40.0])], seeds=1, warmup=0, measure=10, out=tmp_path / "s",
                         base=base, seed=7)
    sets = [argv[i + 1] for i, a in enumerate(argv) if a == "--set"]
    assert sets == ["exit_prob=0.2", "init_spacing=random"]  # defaults and the swept setting are left out
    assert argv[argv.index("--seed") + 1] == "7"
    with pytest.raises(ValueError, match="at most 8"):
        sv.sweep_args([("density", [20.0]), ("radius", list(range(20, 65, 5)))], seeds=1, warmup=0, measure=10,
                      out="s")
    with pytest.raises(ValueError):
        sv.sweep_args([("density", [1.0])] * 2, seeds=1, warmup=0, measure=10, out="s")
    with pytest.raises(ValueError):
        sv.sweep_args([(n, [1]) for n in ("n_lanes", "n_arms", "density", "radius")], seeds=1, warmup=0,
                      measure=10, out="s")
    with pytest.raises(ValueError):
        sv.sweep_args([], seeds=1, warmup=0, measure=10, out="s")


def test_range_spec_round_trips():
    assert sv.range_spec([0.1, 0.2, 0.3]) == "0.1:0.3:0.1"
    assert sv.range_spec([1, 2, 3, 4]) == "1:4:1"
    assert sv.range_spec([1, 2]) == "1,2"
    assert sv.range_spec([1, 76, 151, 225, 300]) == "1,76,151,225,300"
    assert sv.range_spec(["advantage", "random"]) == "advantage,random"
    from roundabout_sim import experiments
    parse = experiments._values  # how experiments reads the spec back
    for name, values in (("exit_prob", [0.0, 0.05, 0.1, 0.15]), ("density", [10.0, 37.5, 65.0, 92.5]),
                         ("n_lanes", [1, 3, 5]), ("reaction_time", [0.1, 0.4, 0.7, 1.0, 1.3])):
        assert parse(name, sv.range_spec(values)) == pytest.approx(values)


def test_command_quoting(tmp_path):
    out = tmp_path / "my results" / "sweep 1"
    argv = sv.sweep_command([("density", [20.0, 40.0])], seeds=1, warmup=0, measure=10, out=out)
    assert argv[argv.index("--out") + 1] == str(out)  # one argument, however many spaces
    text = sv.display_command(argv)
    assert (f'"{out}"' if sys.platform == "win32" else f"'{out}'") in text


def test_form_values():
    density = sv.FormVar.new("density")
    density.lo, density.hi, density.points = 20.0, 100.0, 5
    assert density.values() == [20.0, 40.0, 60.0, 80.0, 100.0]
    density.points = 4
    assert density.values() == [20.0, 47.0, 73.0, 100.0]  # rounded to the slider step
    lanes = sv.FormVar.new("n_lanes")
    assert (lanes.lo, lanes.hi) == (1.0, 5.0) and lanes.max_points() == 5
    lanes.points = 3
    assert lanes.values() == [1, 3, 5] and all(isinstance(v, int) for v in lanes.values())
    lanes.points = 99
    assert lanes.values() == [1, 2, 3, 4, 5]
    motive = sv.FormVar.new("lane_change_motive")
    motive.chosen = ["random", "advantage"]
    assert motive.values() == ["advantage", "random"]


def test_progress_parsing():
    assert sv.parse_progress("[  3/12] n_lanes=1 density=60.0 seed=0: flux 0.437 (1 s, ~2 s left)") == (3, 12)
    assert sv.parse_progress("[120/120] x") == (120, 120)
    assert sv.parse_progress("12 runs (3 x 2 series x 2 seeds) on 12 worker(s)") is None
    assert sv.parse_progress("Means across seeds:") is None


# ================================================================== the subprocess
def test_sweep_run_progress_and_success(tmp_path):
    code = ("print('3 runs (3 x 1 series x 1 seeds) on 1 worker(s)', flush=True)\n"
            "for i in range(1, 4):\n"
            "    print(f'[{i:3d}/3] density=20 seed=0: flux 0.2 ({i} s, ~{3 - i} s left)', flush=True)\n")
    job = sv.SweepRun([sys.executable, "-c", code], tmp_path / "s")
    end = time.monotonic() + 30
    while job.poll() == "running" and time.monotonic() < end:
        time.sleep(0.05)
    assert job.state == "done" and (job.done, job.total) == (3, 3)
    assert job.fraction() == 1.0 and job.eta() == 0.0 and job.last.startswith("[  3/3]")


def test_sweep_run_failure_shows_stderr(tmp_path):
    code = ("import sys; sys.stderr.write('usage: experiments sweep [-h] ...\\n"
            "experiments sweep: error: unrecognized arguments: --vary x\\n'); sys.exit(2)")
    job = sv.SweepRun([sys.executable, "-c", code], tmp_path / "s")
    job.proc.wait(timeout=30)
    assert job.poll() == "failed" and job.proc.returncode == 2
    assert job.error_line() == "experiments sweep: error: unrecognized arguments: --vary x"
    assert "unrecognized arguments" in job.error_text()


def test_error_line_is_the_exception_not_the_traceback(tmp_path):
    """A traceback's long frames come first in the output, the exception last: the error line is the
    exception, and error_text puts it before the frames."""
    deep = "C:/a/very/long/install/path/" * 4
    code = ("import sys\n"
            "sys.stderr.write('Traceback (most recent call last):\\n')\n"
            f"for k in range(8): sys.stderr.write('  File \"{deep}engine.py\", line %d, in step\\n' % k)\n"
            "sys.stderr.write('    raise ValueError(\"the ring cannot hold that many cars\")\\n')\n"
            "sys.stderr.write('ValueError: the ring cannot hold that many cars\\n')\n"
            "sys.exit(1)\n")
    job = sv.SweepRun([sys.executable, "-c", code], tmp_path / "s")
    job.proc.wait(timeout=30)
    assert job.poll() == "failed"
    assert job.error_line() == "ValueError: the ring cannot hold that many cars"
    assert job.error_text().startswith("ValueError: the ring cannot hold that many cars  ·  ")
    assert "engine.py" in job.error_text()


def test_sweep_run_cancel(tmp_path):
    job = sv.SweepRun(slow_job(), tmp_path / "s")
    end = time.monotonic() + 30
    while job.done < 2 and time.monotonic() < end:
        time.sleep(0.05)
    assert (job.done, job.total) == (2, 9) and job.eta() == 7.0
    started = time.monotonic()
    job.cancel()
    assert time.monotonic() - started < 15
    assert job.poll() == "cancelled" and job.proc.poll() is not None


def tree_job(beat: Path) -> list[str]:
    """A stand-in sweep with a worker process (as experiments' process pool has) that keeps writing
    a heartbeat file; the sweep itself only waits."""
    worker = f"import time\nwhile True:\n    open({str(beat)!r}, 'w').write(str(time.time()))\n    time.sleep(0.05)\n"
    code = (f"import subprocess, sys, time\nsubprocess.Popen([sys.executable, '-c', {worker!r}])\n"
            "print('4 runs (4 x 1 series x 1 seeds) on 2 worker(s)', flush=True)\ntime.sleep(60)\n")
    return [sys.executable, "-c", code]


def beating(beat: Path, seconds: float = 0.6) -> bool:
    """True if the heartbeat file changes within ``seconds``."""
    before = beat.read_text() if beat.exists() else ""
    time.sleep(seconds)
    return beat.exists() and beat.read_text() != before


def test_cancel_stops_the_worker_processes_too(tmp_path):
    """Cancel kills the sweep's whole process tree, not only the process it started (works on every
    platform: the worker's heartbeat stops)."""
    beat = tmp_path / "beat.txt"
    job = sv.SweepRun(tree_job(beat), tmp_path / "s")
    end = time.monotonic() + 30
    while not beat.exists() and time.monotonic() < end:
        time.sleep(0.05)
    assert beating(beat)
    job.cancel()
    time.sleep(0.3)  # a write under way when it was killed
    assert not beating(beat) and job.poll() == "cancelled"


def test_quitting_stops_a_running_sweep(make_viewer, csvs, tmp_path):
    """Esc in the window's own loop: run() returns and the sweep, workers and all, is stopped."""
    viewer = make_viewer(csvs["two"])
    beat = tmp_path / "beat.txt"
    assert viewer.start_sweep(tree_job(beat), out=tmp_path / "results" / "quit")
    end = time.monotonic() + 30
    while not beat.exists() and time.monotonic() < end:
        time.sleep(0.05)
    job = viewer.job
    pygame.event.post(key(pygame.K_ESCAPE))
    started = time.monotonic()
    viewer.run()
    assert time.monotonic() - started < 20
    assert job.proc.poll() is not None and job.state == "cancelled"
    assert not beating(beat)


def test_finished_sweep_without_its_csv_is_a_failure(make_viewer, csvs, tmp_path):
    viewer = make_viewer(csvs["two"])
    viewer.show_page("new")
    assert viewer.start_sweep([sys.executable, "-c", "print('1 runs')"], out=tmp_path / "results" / "nocsv")
    wait_for(viewer, lambda: viewer.job.handled, 30)
    assert viewer.job.state == "failed" and viewer.job.proc.returncode == 0
    assert "nocsv.csv could not be loaded: it was not written" in viewer.message[0]
    assert viewer.data.path == csvs["two"]
    frame(viewer)


def test_base_settings_file(tmp_path, make_viewer, csvs):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"exit_prob": 0.3, "n_lanes": 2.0, "seed": 5, "colour_mode": "lane"}),
                    encoding="utf-8")
    changes, seed, ignored = sv.load_base(base)
    assert changes == {"exit_prob": 0.3, "n_lanes": 2} and seed == 5 and ignored == ["colour_mode"]
    viewer = make_viewer(csvs["two"], base_settings=base)
    assert "left out colour_mode" in viewer.message[0]  # names that are not settings are reported
    viewer.toggle_param("density")
    preview = viewer.command_preview()
    assert "--set exit_prob=0.3" in preview and "--set n_lanes=2" in preview and "--seed 5" in preview
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"exit_prob": 7}), encoding="utf-8")
    viewer = make_viewer(csvs["two"], base_settings=bad)
    assert viewer.base == {} and "Could not read the base settings" in viewer.message[0]
    # A typo is reported; a file in which no name is a setting (nested, say) is an error, not the defaults.
    assert sv.load_base({"densty": 80, "n_lanes": 2}) == ({"n_lanes": 2}, None, ["densty"])
    with pytest.raises(ValueError, match="no setting"):
        sv.load_base({"settings": {"density": 80}})
    assert sv.load_base({}) == ({}, None, [])
    nested = tmp_path / "nested.json"
    nested.write_text(json.dumps({"settings": {"density": 80}}), encoding="utf-8")
    viewer = make_viewer(csvs["two"], base_settings=nested)
    assert viewer.base == {} and "Could not read the base settings nested.json: no setting" in viewer.message[0]


# ================================================================== the window
def test_opens_the_newest_sweep_or_the_new_sweep_page(make_viewer, csvs, tmp_path):
    viewer = make_viewer()
    assert viewer.data.path.name == "sample.csv" and viewer.view == "scatter" and viewer.page == "chart"
    empty = make_viewer(results_dir=tmp_path / "empty")
    assert empty.data is None and empty.page == "new" and "No sweep CSV" in empty.message[0]
    frame(empty)
    missing = make_viewer(tmp_path / "nope.csv")
    assert missing.data is None and "Could not load" in missing.message[0]


def test_defaults_per_design(make_viewer, csvs):
    viewer = make_viewer(csvs["two"])
    assert (viewer.view, viewer.x, viewer.series, viewer.metric) == ("lines", "density", "n_lanes", "flux")
    assert viewer.views() == ["lines", "heatmap", "scatter"]
    viewer.load(csvs["one"])
    assert viewer.views() == ["lines", "scatter"] and viewer.series is None and viewer.slice_vars() == []
    viewer.load(csvs["three"])
    assert viewer.x == "density" and viewer.series == "exit_prob"
    assert [v.name for v in viewer.slice_vars()] == ["n_lanes"] and viewer.current_slices() == {"n_lanes": None}


def test_keys(make_viewer, csvs):
    viewer = make_viewer(csvs["three"])
    metrics = viewer.data.metrics
    frame(viewer, key(pygame.K_DOWN))
    assert viewer.metric == metrics[1]
    frame(viewer, key(pygame.K_UP), key(pygame.K_UP))
    assert viewer.metric == metrics[-1]  # wraps round
    frame(viewer, key(pygame.K_RIGHT))
    assert viewer.x == "n_lanes" and viewer.series == "exit_prob"
    frame(viewer, key(pygame.K_RIGHT))
    assert viewer.x == "exit_prob" and viewer.series == "n_lanes"  # the old X takes the place it displaced
    frame(viewer, key(pygame.K_LEFT), key(pygame.K_LEFT))
    assert viewer.x == "density"
    before = viewer.series
    frame(viewer, key(pygame.K_TAB))
    assert viewer.series != before
    frame(viewer, key(pygame.K_TAB, pygame.KMOD_SHIFT))
    assert viewer.series == before
    frame(viewer, key(pygame.K_h))
    assert viewer.view == "heatmap" and viewer.y not in (None, viewer.x)
    frame(viewer, key(pygame.K_h))
    assert viewer.view == "lines"
    frame(viewer, key(pygame.K_o))
    assert viewer.page == "files"
    frame(viewer, key(pygame.K_n))
    assert viewer.page == "new"
    frame(viewer, key(pygame.K_n))
    assert viewer.page == "chart"
    assert not viewer.handle(key(pygame.K_ESCAPE))
    assert not viewer.handle(pygame.event.Event(pygame.QUIT))
    # One setting: no heatmap, and a message says why.
    viewer.load(csvs["one"])
    frame(viewer, key(pygame.K_h))
    assert viewer.view == "lines" and "needs two varied settings" in viewer.message[0]


def test_infinite_setting_values_are_shown_as_choices(make_viewer, tmp_path):
    """Density and Reaction distance = inf are valid: such a sweep loads, the setting's values sit
    evenly spaced with ∞ last, and every view draws (it crashed on the round ticks of an inf span)."""
    frame_ = grid_frame({"reaction_distance": [10.0, 25.0, float("inf")], "n_lanes": [1, 2]})
    path = write(frame_, tmp_path / "results" / "inf.csv")
    assert not sv.load_sweep(path).var("reaction_distance").numeric
    viewer = make_viewer(path)
    viewer.set_x("reaction_distance")
    for view in ("lines", "scatter", "heatmap"):
        viewer.set_view(view)
        frame(viewer, move(viewer.plot_rect.center))
    viewer.set_view("lines")
    positions, span = viewer._x_positions(viewer.data.var("reaction_distance"), viewer.plot_rect)
    assert span is None and np.isfinite(positions).all() and np.all(np.diff(positions) > 0)
    assert viewer.data.var("reaction_distance").text(float("inf")) == "∞"
    viewer.set_x("n_lanes")
    viewer.second = "reaction_distance"  # an infinite series value draws too
    frame(viewer, move(viewer.plot_rect.center))
    assert sv._ticks(0.0, float("inf"), 5)[0].size == 0  # no round ticks for an infinite span


def test_reload_keeps_the_view(make_viewer, csvs):
    viewer = make_viewer(csvs["three"])
    viewer.set_view("heatmap")
    viewer.set_x("n_lanes")
    assert [v.name for v in viewer.slice_vars()] == ["density"]
    viewer.step_slice("density", 2)
    viewer.set_metric("jam_fraction")
    frame(viewer, key(pygame.K_r))
    assert (viewer.view, viewer.x, viewer.y, viewer.metric) == ("heatmap", "n_lanes", "exit_prob", "jam_fraction")
    assert viewer.slices == {"density": 50.0}
    viewer.load(viewer.data.path.parent / "two.csv")  # another file: its own defaults
    assert (viewer.view, viewer.x, viewer.slices) == ("lines", "density", {})


def test_another_file_keeps_the_metric_only_where_it_has_values(make_viewer, csvs, tmp_path):
    """Loading another CSV (a finished sweep, say) keeps the metric when that file has values of it,
    else falls back to flux; the metric's row is scrolled into sight either way."""
    rich = grid_frame({"density": [20.0, 40.0, 60.0]})
    for k in range(30):  # many measured columns, so the list scrolls
        rich[f"zz_extra_{k:02d}"] = rich["flux"] + k
    rich["zz_open_only"] = rich["flux"]
    path = write(rich, tmp_path / "results" / "rich.csv")
    poor = rich.copy()
    poor["zz_open_only"] = np.nan  # a column without values (entry_vc of a closed-mode sweep)
    poor_path = write(poor, tmp_path / "results" / "poor.csv")
    viewer = make_viewer(path)
    viewer.set_metric("zz_extra_29")
    viewer.load(poor_path)  # has values: kept, and in sight
    region = viewer.regions["panel"]
    row = widget(viewer, "metric", "zz_extra_29")
    assert viewer.metric == "zz_extra_29" and region.contains(row.rect)
    viewer.load(path)
    viewer.set_metric("zz_open_only")
    viewer.load(poor_path)  # no value there: flux instead
    assert viewer.metric == "flux" and region.contains(widget(viewer, "metric", "flux").rect)
    viewer.set_metric("zz_open_only")
    viewer.reload()  # the same file: the choice stays
    assert viewer.metric == "zz_open_only"


def test_mouse_controls(make_viewer, csvs):
    viewer = make_viewer(csvs["three"])
    frame(viewer)
    click(viewer, widget(viewer, "metric", "jam_fraction").rect.center)
    assert viewer.metric == "jam_fraction"
    click(viewer, arrow(viewer, "slice:n_lanes"))
    assert viewer.slices["n_lanes"] == 1
    click(viewer, arrow(viewer, "slice:n_lanes"))
    assert viewer.slices["n_lanes"] == 2
    click(viewer, arrow(viewer, "slice:n_lanes", 0))
    click(viewer, arrow(viewer, "slice:n_lanes", 0))
    assert viewer.slices["n_lanes"] is None  # back to the average
    click(viewer, widget(viewer, "cycler", "x").data["row"].center)  # the value itself steps forward
    assert viewer.x == "n_lanes"
    view = widget(viewer, "choice", "view")
    click(viewer, view.options[sv.VIEWS.index("heatmap")].center)
    assert viewer.view == "heatmap"
    click(viewer, view.options[sv.VIEWS.index("scatter")].center)
    assert viewer.view == "scatter" and viewer.slice_vars() == []
    # The tabs.
    click(viewer, widget(viewer, "tab", "files").rect.center)
    assert viewer.page == "files"
    click(viewer, widget(viewer, "file", str(csvs["two"])).rect.center)
    assert viewer.data.path == csvs["two"]
    click(viewer, widget(viewer, "button", "newest").rect.center)
    assert viewer.data.path.name == "sample.csv"
    click(viewer, widget(viewer, "tab", "chart").rect.center)
    assert viewer.page == "chart"


def test_metric_list_scrolls(make_viewer, tmp_path):
    frame_ = grid_frame({"density": [20.0, 40.0], "n_lanes": [1, 2], "exit_prob": [0.0, 0.1],
                         "radius": [40.0, 80.0]})
    for k in range(40):
        frame_[f"extra_{k}"] = k
    viewer = make_viewer(write(frame_, tmp_path / "results" / "many.csv"), size=(1100, 700))
    frame(viewer)
    region = viewer.regions["panel"]
    assert viewer.content_h["panel"] > region.h
    frame(viewer, move(region.center), wheel(-3))
    assert viewer.scroll["panel"] > 0
    for _ in range(30):
        frame(viewer, wheel(-5))
    assert viewer.scroll["panel"] == viewer.content_h["panel"] - region.h  # clamped at the end
    viewer.scroll["panel"] = 0
    viewer.set_metric(viewer.data.metrics[0])
    frame(viewer, key(pygame.K_UP))  # wraps to the last metric, which is scrolled into view
    last = widget(viewer, "metric", viewer.data.metrics[-1])
    assert viewer.scroll["panel"] > 0 and region.contains(last.rect)


def test_hover_shows_exact_values(make_viewer, csvs):
    viewer = make_viewer(csvs["two"])
    frame(viewer)
    plot = viewer.plot_rect
    frame(viewer, move((plot.x + 3, plot.centery)))  # nearest point: the first density
    tip = viewer.tooltip
    assert tip[0] == "Density = 20 veh/km/lane"
    expect = np.mean([flux_of(20.0, 1, EXIT, s) for s in range(2)])
    assert any(line.startswith("1: mean") and sv.fmt_num(expect) in line and "(2 runs)" in line for line in tip)
    # The text column is counted per line (each line has one run of each seed), not pooled over them.
    assert [line.split("  ·  ")[-1] for line in tip[1:3]] == ["Entry LOS A ×1, B ×1"] * 2
    assert not any(line.startswith("Entry LOS") for line in tip)
    assert viewer._status_text() == sv.VIEW_HELP["lines"]
    viewer.set_view("heatmap")
    frame(viewer)
    plot = viewer.plot_rect
    frame(viewer, move((plot.x + 3, plot.bottom - 3)))  # bottom-left cell: first X, first Y
    assert viewer.tooltip[:2] == ["Density = 20 veh/km/lane", "Lanes = 1"]
    assert "(2 runs)" in viewer.tooltip[2]
    viewer.load(viewer.data.path.parent / "one.csv")  # no series: the pooled count on its own line
    frame(viewer)
    plot = viewer.plot_rect
    frame(viewer, move((plot.x + 3, plot.centery)))
    assert viewer.tooltip[-1] == "Entry LOS: A ×1, B ×1"
    viewer.load(viewer.data.path.parent / "sample.csv")
    frame(viewer)
    plot = viewer.plot_rect
    frame(viewer, move((plot.right - 2, plot.bottom - 2)))  # no dot down there: the bin under the mouse
    assert "bin" in viewer.tooltip[0] and "runs)" in viewer.tooltip[1] and "mean" in viewer.tooltip[1]


def test_scatter_of_a_grid_has_the_means_at_the_values(make_viewer, csvs):
    """A grid's scatter joins the mean at each X value (no bins that mix values or sit between them)."""
    viewer = make_viewer(csvs["two"])
    viewer.set_view("scatter")
    frame(viewer)
    sx, sy, ok = viewer._dots
    density = viewer.data.frame["density"].to_numpy()
    at20, at50 = sx[density == 20.0][0], sx[density == 50.0][0]
    frame(viewer, move((at20 + 0.3 * (at50 - at20), viewer.plot_rect.centery)))  # between the dot columns
    expect = np.mean([flux_of(20.0, n, EXIT, s) for n in (1, 2) for s in range(2)])
    assert viewer.tooltip[0] == "Density = 20 veh/km/lane"
    assert viewer.tooltip[1].startswith(f"mean {sv.fmt_num(expect)}") and viewer.tooltip[1].endswith("(4 runs)")


def test_scatter_hover_on_a_dot(make_viewer, csvs):
    viewer = make_viewer(csvs["sample"])
    frame(viewer)
    sx, sy, ok = viewer._dots
    d = np.hypot(sx[:, None] - sx[None, :], sy[:, None] - sy[None, :])
    np.fill_diagonal(d, np.inf)
    i = int(np.argmax(np.where(ok, d.min(axis=1), -1)))  # the run whose dot is farthest from any other
    frame(viewer, move((sx[i], sy[i])))
    run = viewer.data.frame.iloc[i]
    assert viewer.tooltip[0] == f"{sv.metric_info('flux').label}: {sv.fmt_num(run['flux'])}"
    assert viewer.tooltip[1] == sv.setting_phrase(viewer.x, run[viewer.x])
    assert viewer.tooltip[2] == sv.setting_phrase(viewer.colour, run[viewer.colour])
    # Every other varied setting of the run too (a random sample's run has its own value of each).
    other = next(v.name for v in viewer.data.varied if v.name not in (viewer.x, viewer.colour))
    assert viewer.tooltip[3] == sv.setting_phrase(other, run[other])
    assert f"run {i + 1} of {viewer.data.runs}" in viewer.tooltip[4] and viewer.tooltip[-1] == "Entry LOS: A"


def test_save_png(make_viewer, csvs, tmp_path):
    viewer = make_viewer(csvs["two"])
    frame(viewer, key(pygame.K_s))
    saved = list((tmp_path / "results").glob("sweepview_two_lines_flux_*.png"))
    assert len(saved) == 1 and "Saved" in viewer.message[0]
    assert pygame.image.load(str(saved[0])).get_size() == viewer.main_rect.size
    click(viewer, widget(viewer, "button", "save").rect.center)
    assert len(list((tmp_path / "results").glob("sweepview_two_*.png"))) == 2  # a second name, not overwritten
    viewer.show_page("new")
    path = viewer.save_png()
    assert pygame.image.load(str(path)).get_size() == viewer.screen.get_size()


def test_clicks_and_keys_answer_in_the_help_box(make_viewer, csvs, monkeypatch):
    """What a click or a key did shows in the help box although the mouse rests on the control or
    the chart (whose hover help would otherwise come first)."""
    from roundabout_sim import ui

    viewer = make_viewer(csvs["one"])
    frame(viewer)
    save = widget(viewer, "button", "save").rect.center
    click(viewer, save)
    frame(viewer)
    assert viewer.message[0].startswith("Saved") and viewer._status_text() == viewer.message[0]
    frame(viewer, move((save[0] + 1, save[1])))  # still on the button
    assert viewer._status_text() == viewer.message[0]
    frame(viewer, move(viewer.plot_rect.center))  # off it: the chart's help again
    assert viewer._status_text() == sv.VIEW_HELP["lines"]
    heatmap = widget(viewer, "choice", "view").options[sv.VIEWS.index("heatmap")].center
    click(viewer, heatmap)  # refused on a one-setting sweep: it says why
    assert viewer.view == "lines" and viewer._status_text() == viewer.message[0]
    assert "needs two varied settings" in viewer.message[0]
    frame(viewer, move(viewer.plot_rect.center), key(pygame.K_s))  # a key, the mouse on the chart
    assert viewer.message[0].startswith("Saved") and viewer._status_text() == viewer.message[0]
    monkeypatch.setattr(ui, "PIN_HOLD", 0.0)
    frame(viewer, move((viewer.plot_rect.centerx + 2, viewer.plot_rect.centery)))
    assert viewer._status_text() == sv.VIEW_HELP["lines"]  # once the mouse moves, the hover help is back
    viewer.show_page("new")
    frame(viewer)
    run = widget(viewer, "button", "run").rect.center
    click(viewer, run)  # nothing picked: it says so
    assert viewer._status_text() == viewer.message[0] == "Pick at least one setting to sweep."
    viewer.show_page("files")
    frame(viewer)
    click(viewer, widget(viewer, "button", "refresh").rect.center)
    assert "sweep CSV(s) in" in viewer._status_text()


def test_a_failed_save_is_reported_not_raised(make_viewer, csvs, tmp_path):
    """An unwritable results folder (here: a file in its place) gives a message; the window goes on,
    and so does starting a sweep that cannot write there."""
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("x", encoding="utf-8")
    viewer = make_viewer(csvs["two"], results_dir=blocker)
    frame(viewer, key(pygame.K_s))
    assert viewer.save_png() is None and "Could not save the PNG" in viewer.message[0]
    assert not viewer.start_sweep(slow_job(), out=blocker / "sub" / "s")
    assert viewer.job is None and "Could not start the sweep" in viewer.message[0]
    # A long CSV name is cut in the PNG's name.
    long = write(grid_frame({"density": [20.0, 40.0]}), tmp_path / "results" / f"{'x' * 120}.csv")
    viewer.load(long)
    viewer.results_dir = tmp_path / "results"
    assert len(viewer.save_png().name) < 100


def test_chart_keys_act_only_on_the_chart_page(make_viewer, csvs):
    viewer = make_viewer(csvs["three"])
    before = (viewer.metric, viewer.x, viewer.series, viewer.view)
    for page, hint in (("new", sv.NEW_HINT), ("files", sv.FILES_HINT)):
        viewer.show_page(page)
        viewer.message = ("", 0.0)
        frame(viewer, key(pygame.K_DOWN), key(pygame.K_RIGHT), key(pygame.K_TAB), key(pygame.K_h), move((-1, -1)))
        assert (viewer.metric, viewer.x, viewer.series, viewer.view) == before
        assert viewer._status_text() == hint  # the idle hint of that page
    viewer.show_page("chart")
    frame(viewer, key(pygame.K_DOWN), move((-1, -1)))
    assert viewer.metric != before[0] and viewer._status_text() == sv.DEFAULT_HINT


def test_help_box_fits_every_help_text_and_keeps_its_height(make_viewer, csvs, tmp_path):
    for size in SIZES:
        viewer = make_viewer(results_dir=tmp_path / "empty", size=size)
        width, lines = viewer.status_rect.w, viewer.status_lines
        assert sv.STATUS_LINES[0] <= lines <= sv.STATUS_LINES[1]
        cut = [t[:50] for t in sv.help_texts() if len(viewer.fonts.wrap(t, width)) > lines]
        assert not cut, f"{size}: help cut in the help box: {cut}"
        for path in (csvs["sample"], csvs["three"], csvs["one"]):  # the same height whatever is loaded
            viewer.load(path)
            assert viewer.status_lines == lines


def test_readouts_have_help(make_viewer, csvs, tmp_path):
    viewer = make_viewer(csvs["two"])
    assert viewer.start_sweep(slow_job(), out=tmp_path / "results" / "dummy")
    wait_for(viewer, lambda: viewer.job.done == 2, 30)
    progress = widget(viewer, "readout", "progress")  # on the chart page too
    frame(viewer, move(progress.rect.center))
    assert viewer._status_text().startswith("The sweep running in the background") and "dummy.csv" in \
        viewer._status_text()
    viewer.show_page("new")
    frame(viewer, move(widget(viewer, "readout", "job").rect.center))
    assert viewer._status_text().startswith("The sweep running in the background")
    viewer.cancel_sweep()
    assert widget(viewer, "readout", "job").help == sv.JOB_HELP["cancelled"]


def test_one_setting_sample_keeps_the_lines_view(make_viewer, tmp_path):
    rng = np.random.default_rng(2)
    rows = [{**default_settings(), "density": d, "seed": 0, "flux": 0.01 * d, "jam_fraction": d / 150}
            for d in rng.uniform(10, 150, 20)]
    viewer = make_viewer(write(pd.DataFrame(rows), tmp_path / "results" / "one_sample.csv"))
    assert viewer.data.design == "sample" and viewer.view == "scatter"
    assert viewer.views() == ["lines", "scatter"]
    viewer.set_view("lines")
    assert viewer.view == "lines"
    frame(viewer)


def test_new_sweep_form(make_viewer, csvs):
    viewer = make_viewer(csvs["two"])
    frame(viewer, key(pygame.K_n))
    click(viewer, widget(viewer, "button", "run").rect.center)
    assert viewer.job is None and "Pick at least one" in viewer.message[0]
    for name in ("density", "n_lanes", "lane_change_motive"):
        click(viewer, param_pos(viewer, name))
    assert [v.name for v in viewer.form] == ["density", "n_lanes", "lane_change_motive"]
    click(viewer, param_pos(viewer, "radius"))
    assert len(viewer.form) == 3 and "at most 3" in viewer.message[0]
    # Drag the start of the density range up; the points follow the step.
    rng = widget(viewer, "range", "0")
    track = rng.options[0]
    for event in (move(track.midleft), down(track.midleft), move((track.x + track.w // 2, track.centery)),
                  up((track.x + track.w // 2, track.centery))):
        assert viewer.handle(event)
    density = viewer.form[0]
    assert 140 <= density.lo <= 160 and density.hi == 300
    click(viewer, arrow(viewer, "points:0"))
    assert density.points == 6
    click(viewer, arrow(viewer, "points:1", 0))
    assert viewer.form[1].points == 4
    toggle = widget(viewer, "toggle", "2")
    click(viewer, toggle.options[1].center)
    assert viewer.form[2].values() == ["advantage"]
    click(viewer, toggle.options[0].center)  # the last choice cannot be switched off
    assert viewer.form[2].values() == ["advantage"]
    for name, value in (("seeds", 4), ("warmup", 180.0), ("measure", 600.0)):
        click(viewer, arrow(viewer, name))
        assert getattr(viewer, name) == value
    preview = viewer.command_preview()
    assert "--vary density=" in preview and "--vary n_lanes=" in preview and "--vary lane_change_motive=advantage" \
        in preview and "--seeds 4" in preview
    assert widget(viewer, "readout", "command").help == sv.COMMAND_HELP
    assert widget(viewer, "readout", "summary").help == sv.SUMMARY_HELP
    click(viewer, widget(viewer, "remove", "0").rect.center)
    assert [v.name for v in viewer.form] == ["n_lanes", "lane_change_motive"]
    frame(viewer, move(param_pos(viewer, "density")))
    assert viewer._status_text().startswith("Density (1–300 veh/km/lane)")
    # The parameter list scrolls.
    viewer.scroll["params"] = 0
    frame(viewer, move(viewer.regions["params"].center), wheel(-10))
    assert viewer.scroll["params"] > 0


def test_new_sweep_form_warns_of_settings_that_change_nothing(make_viewer, csvs, monkeypatch):
    """With closed-mode base settings (the defaults) the open-mode settings are marked in the list,
    and picking one warns that every run would be the same, until Traffic mode is picked with open."""
    viewer = make_viewer(csvs["two"])
    frame(viewer, key(pygame.K_n))
    assert widget(viewer, "param", "entry_rate").data == "open mode only"
    assert widget(viewer, "param", "density").data == ""
    assert "Here it would change nothing (open mode only)" in widget(viewer, "param", "entry_rate").help
    rendered = []
    blit = viewer.fonts.blit
    monkeypatch.setattr(viewer.fonts, "blit", lambda scr, text, *a, **k: rendered.append(text) or blit(scr, text, *a,
                                                                                                         **k))
    click(viewer, param_pos(viewer, "entry_rate"))
    frame(viewer)
    warning = " ".join(viewer.idle_lines[1])
    assert warning.startswith("Entrance chance (/s) acts only in open mode, and every run of this sweep is closed")
    assert "open mode only" in rendered and any(line in rendered for line in viewer.idle_lines[1])
    click(viewer, param_pos(viewer, "traffic_mode"))  # closed and open: it acts in the open runs
    frame(viewer)
    assert viewer.idle_lines[1] == [] and widget(viewer, "param", "entry_rate").data == ""
    click(viewer, widget(viewer, "toggle", "1").options[1].center)  # open switched off: the warning is back
    assert viewer.form[1].values() == ["closed"] and viewer.idle_lines[1]
    # Open-mode base settings (the main viewer in open mode): nothing to warn of.
    viewer.base = {"traffic_mode": "open"}
    viewer.layout()
    assert viewer.idle_reason("entry_rate") == "" and viewer.idle_reason("truck_length") == "no trucks"


def test_new_sweep_summary_counts_the_runs(make_viewer, csvs):
    viewer = make_viewer(csvs["two"])
    viewer.show_page("new")
    viewer.toggle_param("density")
    viewer.form[0].points = 6
    viewer.seeds = 2
    viewer.layout()
    summary = " ".join(viewer.summary_lines)
    assert summary.startswith("6 points × 2 seeds = 12 runs, each")  # one setting: no "6 = 6 points"
    viewer.toggle_param("n_lanes")
    viewer.layout()
    n = len(viewer.form[1].values())
    assert " ".join(viewer.summary_lines).startswith(f"6 × {n} = {6 * n} points × 2 seeds = {12 * n} runs")
    viewer.seeds = 1
    viewer.layout()
    assert f"× 1 seed = {6 * n} runs" in " ".join(viewer.summary_lines)


def test_series_setting_capped_at_eight_values(make_viewer, csvs):
    viewer = make_viewer(csvs["two"])
    viewer.show_page("new")
    viewer.toggle_param("n_lanes")
    viewer.toggle_param("radius")  # 2nd: one line per value
    radius = viewer.form[1]
    assert radius.max_points(sv.MAX_POINTS[1]) == 8
    for _ in range(20):
        viewer.cycle("points:1", 1)
        assert radius.points <= 8 and len(radius.values()) <= 8
    viewer.remove_var(0)  # now the x axis: more points allowed
    for _ in range(10):
        viewer.cycle("points:0", 1)
    assert radius.points > 8


def test_cancel_a_running_sweep(make_viewer, csvs, tmp_path):
    viewer = make_viewer(csvs["two"])
    viewer.show_page("new")
    assert viewer.start_sweep(slow_job(), out=tmp_path / "results" / "dummy")
    wait_for(viewer, lambda: viewer.job.done == 2, 30)
    frame(viewer)
    assert widget(viewer, "button", "run").data["disabled"] and not widget(viewer, "button", "cancel").data["disabled"]
    assert not viewer.start_sweep(slow_job())  # one at a time
    click(viewer, widget(viewer, "button", "cancel").rect.center)
    assert viewer.job.state == "cancelled" and viewer.job.handled
    assert "cancelled after 2 of 9" in viewer.message[0]
    frame(viewer)


def test_cancelling_while_saving_leaves_nothing_behind(make_viewer, csvs, tmp_path):
    """Cancelled after its last run but before its CSV was renamed into place: what experiments had begun
    to write (a meta file, a temporary CSV) is removed, and the message says when it stopped."""
    viewer = make_viewer(csvs["two"])
    viewer.show_page("new")
    out = tmp_path / "results" / "sweep_20260101_120000"
    assert viewer.start_sweep(slow_job(lines=3, total=3), out=out)
    wait_for(viewer, lambda: viewer.job.done == 3, 30)
    left = [Path(f"{out}_meta.json"), out.parent / f".{out.name}.csv.k3j_x.tmp",
            out.parent / f".{out.name}_meta.json.q1.tmp"]
    other = out.parent / ".another_sweep.csv.zz.tmp"  # not this sweep's: kept
    for path in [*left, other]:
        path.write_text("", encoding="utf-8")
    viewer.cancel_sweep()
    assert viewer.job.state == "cancelled" and not any(path.exists() for path in left) and other.exists()
    assert viewer.message[0] == "Sweep cancelled while saving its results (nothing was kept)."
    frame(viewer)


def test_cancelling_once_the_csv_is_written_loads_it(make_viewer, csvs, tmp_path):
    """experiments sweep writes its CSV when every run is done and only then draws its figures: a
    cancel in that phase leaves the whole CSV, which the viewer loads instead of saying nothing was
    written."""
    viewer = make_viewer(csvs["two"])
    viewer.show_page("new")
    out = tmp_path / "results" / "drawing"
    assert viewer.start_sweep(slow_job(lines=9, total=9), out=out)
    wait_for(viewer, lambda: viewer.job.done == 9, 30)
    Path(f"{out}.csv").write_bytes(csvs["one"].read_bytes())  # as experiments has written it by then
    viewer.cancel_sweep()
    job = viewer.job
    assert job.state == "cancelled" and job.saved and viewer.data.path == Path(f"{out}.csv")
    assert viewer.page == "chart" and "all 8 runs were already saved to drawing.csv (no PNG)" in viewer.message[0]
    assert widget(viewer, "readout", "progress").help == sv.JOB_HELP["saved"].format(out=out)
    frame(viewer)
    viewer.show_page("new")
    frame(viewer)


def test_new_sweep_page_lists_every_base_setting(make_viewer, tmp_path):
    """A random base (V after X in the main viewer) differs from the defaults in most settings: the page
    lists every one that is not swept, and the whole command, at every size with up to three settings."""
    from roundabout_sim import presets, state
    from roundabout_sim.engine import RingSim

    base = tmp_path / "base.json"
    state.write_json(base, state.settings_of(RingSim(SimConfig(seed=3).updated(**presets.random_changes(1234)))))
    for size in SIZES:
        viewer = make_viewer(size=size, results_dir=tmp_path / "results", base_settings=base)
        viewer.show_page("new")
        changed = sv.changed_settings(viewer.base)
        assert len(changed) > 20
        for picks in ([], ["density"], ["density", "radius", "exit_prob"], ["entry_rate", "follow_up", "road_condition"]):
            viewer.form = []
            for name in picks:
                viewer.toggle_param(name)
            frame(viewer)
            shown = " ".join(viewer.base_lines[1])
            assert all((sv.setting_phrase(k, v) in shown) != (k in picks) for k, v in changed.items()), (size, picks)
            top, lines = viewer.command_lines
            assert lines == viewer.fonts.wrap(viewer.command_preview(), viewer.form_right.w), (size, picks)
            assert top + (len(lines) + 1) * viewer.fonts.line("small") <= viewer.main_rect.bottom
            listed = widget(viewer, "readout", "base_list")
            assert listed.rect.bottom <= widget(viewer, "readout", "command").rect.top
            assert listed.help == sv.BASE_HELP
        pygame.quit()


def test_failed_sweep_is_reported(make_viewer, csvs, tmp_path):
    viewer = make_viewer(csvs["two"])
    viewer.show_page("new")
    code = "import sys; sys.stderr.write('Traceback ...\\nValueError: density: bad value\\n'); sys.exit(1)"
    viewer.start_sweep([sys.executable, "-c", code], out=tmp_path / "results" / "bad")
    wait_for(viewer, lambda: viewer.job.handled, 30)
    assert viewer.job.state == "failed" and "Sweep failed (exit 1)" in viewer.message[0]
    assert "ValueError: density: bad value" in viewer.message[0]
    assert viewer.page == "new" and viewer.data.path == csvs["two"]  # the chart is left as it was


def test_runs_a_real_sweep_and_loads_it(make_viewer, csvs, tmp_path):
    """A real (tiny) experiments sweep started from the form, followed to the end and loaded."""
    viewer = make_viewer(csvs["two"])
    viewer.show_page("new")
    viewer.toggle_param("density")
    density = viewer.form[0]
    density.lo, density.hi, density.points = 20.0, 40.0, 2
    viewer.seeds, viewer.warmup, viewer.measure = 1, 0.0, 10.0
    click(viewer, widget(viewer, "button", "run").rect.center)
    job = viewer.job
    assert job is not None and job.argv[job.argv.index("--x") + 1] == "density=20,40"
    wait_for(viewer, lambda: job.handled, 180)
    assert job.state == "done", job.error_text()
    assert (job.done, job.total) == (2, 2)
    assert viewer.page == "chart" and viewer.data.path == Path(f"{job.out}.csv")
    assert [v.name for v in viewer.data.variables] == ["density"] and viewer.data.runs == 2
    assert "Sweep finished" in viewer.message[0]
    # Every measured column of a real sweep CSV has a definition as its help.
    undescribed = [m for m in viewer.data.metrics if sv.metric_info(m, "density").help.endswith(PLACEHOLDER)]
    assert not undescribed, f"metric columns without help: {undescribed}"


@pytest.mark.parametrize("size", SIZES)
def test_every_mode_draws(make_viewer, csvs, tmp_path, size):
    """Each view, page and state at the minimum, a medium and a large window: drawn without errors,
    with the controls inside the window and not overlapping."""
    viewer = make_viewer(csvs["one"], size=size)
    W, H = size

    def check(name: str) -> None:
        for pos in ((-1, -1), viewer.plot_rect.center, (viewer.plot_rect.x + 2, viewer.plot_rect.y + 2)):
            frame(viewer, move(pos))
        screen = viewer.screen.get_rect()
        fixed = [w for w in viewer.widgets if w.region is None and w.kind not in ("heading", "block")]
        for w in fixed:
            assert screen.contains(w.rect), f"{name}: {w.kind} {w.name} outside the window"
        for a, b in itertools.combinations(fixed, 2):
            assert not a.rect.colliderect(b.rect) or a.kind == "remove" or b.kind == "remove", \
                f"{name}: {a.kind} {a.name} overlaps {b.kind} {b.name}"
        for region in viewer.regions.values():
            assert screen.contains(region) and region.h > 0
        assert viewer.status_rect.bottom <= H and viewer.panel_rect.right == W

    check("one setting, lines")
    viewer.set_view("scatter")
    check("one setting, scatter")
    viewer.load(csvs["two"])
    check("multi-series")
    viewer.toggle_heatmap()
    check("heatmap")
    viewer.load(csvs["three"])
    check("three settings, averaged")
    viewer.step_slice("n_lanes", 1)
    viewer.set_metric("congestion_onset")
    check("three settings, slice")
    viewer.toggle_heatmap()
    check("three settings, heatmap with a slice")
    viewer.load(csvs["old"])
    check("old CSV, categorical x")
    viewer.load(csvs["sample"])
    check("scatter, numeric colour")
    viewer.step_second(1)
    check("scatter, categorical colour")
    viewer.step_second(1)
    check("scatter, no colour")
    viewer.show_page("files")
    check("files")
    viewer.show_page("new")
    check("empty form")
    for name in ("density", "n_lanes", "lane_change_motive"):
        viewer.toggle_param(name)
    check("form")
    viewer.start_sweep(slow_job(), out=tmp_path / "results" / "dummy")
    wait_for(viewer, lambda: viewer.job.done == 2, 30)
    check("running")
    viewer.show_page("chart")
    check("chart while running")
    viewer.cancel_sweep()
    check("cancelled")
    frame(viewer, pygame.event.Event(pygame.VIDEORESIZE, size=(800, 500), w=800, h=500))
    desk = (pygame.display.get_desktop_sizes() or [ui.MIN_WINDOW])[0]  # the minimum, where the screen allows
    assert viewer.screen.get_size() == (max(800, min(ui.MIN_WINDOW[0], desk[0])),
                                        max(500, min(ui.MIN_WINDOW[1], desk[1])))


# ================================================================== texts and CLI
def test_metric_labels_and_help():
    flux = sv.metric_info("flux")
    assert flux.headline and flux.label.startswith("Flux") and flux.help.startswith(flux.label)
    assert sv.metric_info("flux", "density").label.startswith("Fundamental diagram")
    jam = sv.metric_info("jam_fraction")
    assert jam.scale == 100.0 and "CONGESTION" in jam.help  # the experiments definition as help
    cars = sv.metric_info("queued")  # defined together with "cars"
    assert cars.label == "Cars queued to enter" and "waiting at all entrances" in cars.help
    assert cars.help.startswith("Cars queued to enter [queued]: ")  # which CSV column it is
    for column in ("arrivals_per_min", "entry_vc", "entries_per_min", "cars", "truck_speed", "demand_per_min"):
        assert sv.metric_info(column, "", open_demand=True).label != column, column  # readable names
    other = sv.metric_info("extra_column")
    assert other.label == "extra_column" and not other.headline
    # The sweep chart draws both onsets; here each is shown alone, so each says which threshold it is.
    stop_go, congestion = sv.metric_info("stop_go_onset"), sv.metric_info("congestion_onset")
    assert stop_go.label == "Stop-and-go onset (s from the start)" and "either" not in stop_go.empty
    assert congestion.label == "Congestion onset (s from the start)" and congestion.headline
    assert "stop-and-go index" in stop_go.empty and "jammed" in congestion.empty
    # Columns named inside another definition get that one; those without one, the viewer's own.
    assert sv.metric_info("truck_speed").help.startswith("Truck speed (m/s) [truck_speed]: (defined with trucks)")
    assert "(defined with n_jams)" in sv.metric_info("jams_merge").help
    assert "longest queue seen" in sv.metric_info("peak_queue").help
    assert "exit_prob" not in sv.definitions()  # a setting named in a definition is no metric
    for name in ("measured_density", "mean_free_speed", "measured_speed_std", "speed_deficit", "occupancy",
                 "exits_per_min", "missed_per_min", "mean_trip_time", "wave_speed_kmh"):  # experiments describes them
        assert name in sv.definitions() and not sv.metric_info(name).help.endswith(PLACEHOLDER), name
    # A sweep of open-mode runs titles the entrance columns as its figure does; closed mode by name.
    delay = sv.metric_info("mean_entry_delay", "", open_demand=True)
    assert delay.headline and delay.label == "Mean entry delay (s, queue wait)"
    assert not sv.metric_info("mean_entry_delay").headline
    peak = sv.metric_info("peak_queue", "", open_demand=True)
    assert peak.label == "Longest queue, longest seen (cars)" and "crossed" not in peak.empty
    assert not sv.metric_info("emergency_per_min").help.startswith("Harsh braking (episodes/min): Harsh")


def test_chart_notes_name_only_lines_the_view_draws():
    """The sweep figure's notes name its reference lines; here they show only in the lines view,
    which draws them (the LOS limits, the random-placement level, the stop-and-go onset)."""
    for column in [*sv._panels("", True), *sv._panels("density", True)]:
        info = sv.metric_info(column, "", open_demand=True)
        for view in sv.VIEWS:
            note = info.note_for(view)
            assert not any(style in note for style in ("(solid)", "(dashed)", "(dotted)")), (column, note)
            if view != "lines":
                assert not any(word in note for word in ("line =", "lines =", "dashed")), (column, view, note)
        drawn = info.ref is not None or info.los or info.ref_column  # (a zero line needs no words)
        assert not info.lines or drawn, column
    assert sv.metric_info("mean_entry_delay", "", True).los
    order = sv.metric_info("order_param", "", True)
    assert order.ref_column == "order_baseline" and "dashed = random placement" in order.note_for("lines")
    assert sv.metric_info("peak_queue", "", True).note == sv.metric_info("longest_queue", "", True).note


def test_lines_view_draws_the_los_limits_and_the_random_level(make_viewer, tmp_path, monkeypatch):
    frame_ = grid_frame({"entry_rate": [0.05, 0.2, 0.4]}, settings={"traffic_mode": "open"})
    frame_["mean_entry_delay"] = frame_["entry_rate"] * 100  # 5, 20 and 40 s: bands A, C and E
    frame_["order_param"], frame_["order_baseline"] = 0.3, 0.12
    viewer = make_viewer(write(frame_, tmp_path / "results" / "open.csv"))
    texts, dashes = [], []
    blit = viewer.fonts.blit
    monkeypatch.setattr(viewer.fonts, "blit", lambda scr, text, *a, **k: texts.append(text) or blit(scr, text, *a, **k))
    monkeypatch.setattr(sv, "dashed_line", lambda surface, colour, points, *a: dashes.append(points))
    viewer.set_metric("mean_entry_delay")
    frame(viewer)
    assert {"A", "B", "C", "D"} <= set(texts) and "lines = each limit" in " ".join(texts)
    assert len(dashes) == 4  # the limits at 10, 15, 25 and 35 s, inside the 0-40 s range
    texts.clear(), dashes.clear()
    viewer.set_metric("order_param")
    frame(viewer)
    assert any(abs(y - dashes[0][0][1]) < 1e-9 for _, y in dashes[0])  # a level line at the random level
    viewer.set_view("scatter")
    texts.clear()
    frame(viewer)
    assert "dashed" not in " ".join(texts)


def test_value_formatting():
    assert sv.setting_text("density", 20.0) == "20"
    assert sv.setting_text("exit_prob", 0.1) == "0.10"
    assert sv.setting_text("density", 37.123456) == "37.1235"  # off the step: enough digits to tell apart
    assert [sv.setting_text("density", v) for v in (113.5, 128.5, 143.5)] == ["113.5", "128.5", "143.5"]
    assert sv.setting_text("n_lanes", np.int64(2)) == "2"
    assert sv.setting_text("lane_change_motive", "random") == "random"
    assert sv.setting_phrase("radius", 80.0) == "Ring radius = 80 m"
    assert sv.setting_label("density") == "Density (veh/km/lane)" and sv.setting_label("seed") == "Seed"
    assert sv.fmt_num(0.43412) == "0.434" and sv.fmt_num(12.345) == "12.3" and sv.fmt_num(np.nan) == "–"
    assert sv.tick_text(7.5, 2.5) == "7.5" and sv.tick_text(20.0, 20.0) == "20" and sv.tick_text(-1e-17, 0.1) == "0.0"


def test_parse_args():
    args = sv.parse_args(["s.csv", "--base-settings", "b.json", "--results", "out", "--size", "1100x700"])
    assert (args.csv, args.base_settings, args.results, args.size) == ("s.csv", "b.json", "out", (1100, 700))
    args = sv.parse_args([])
    assert args.csv is None and args.base_settings is None and args.results == "results"
    with pytest.raises(SystemExit):
        sv.parse_args(["--size", "big"])


# ------------------------------------------------------------------ the second review's fixes
FIXED_MANY = {"n_arms": 12, "radius": 67.25, "n_lanes": 5, "exit_prob": 0.17, "entry_gap": 1.5, "follow_up": 1.9,
              "traffic_mode": "open", "demand_imbalance": 0.6, "mean_speed": 8.5, "reaction_time": 0.6,
              "time_headway": 0.9, "min_gap": 1.5, "driver_noise": 0.3}


@pytest.mark.parametrize("base, text", [({"random": 8}, "random settings (seed 8)"),
                                        ({"preset": "etoile"}, "preset etoile"), ({}, None)])
def test_context_names_the_base_and_its_hover_box_holds_every_fixed_setting(make_viewer, tmp_path, base, text):
    """The chart's context line names where the base settings came from (as experiments prints it), and
    the box by the mouse holds the whole context, which the line cuts with an ellipsis at 1100x700."""
    meta = {"version": 1, "design": "grid", "varied": [{"name": "density", "values": [20.0, 40.0]}],
            "preset": None, "random": None, "warmup": 2.0, "measure": 4.0, **base}
    path = write(grid_frame({"density": [20.0, 40.0]}, settings=FIXED_MANY), tmp_path / "results" / "b.csv", meta)
    viewer = make_viewer(path, size=(1100, 700))
    frame(viewer)  # (lays the header lines out)
    context = viewer._context(None)
    assert (text in context) if text else ("random settings" not in context and "preset" not in context)
    frame(viewer, move(viewer.context_rect.center))
    assert viewer._status_text() == sv.CONTEXT_HELP
    shown = " ".join(viewer.tooltip)
    for name, value in FIXED_MANY.items():
        assert sv.setting_phrase(name, value) in shown, name  # all of them, none cut off
    assert viewer.tooltip[0] == "b.csv" and (text is None or text in viewer.tooltip)
    assert all(viewer.fonts.width(line, "small") <= viewer.main_rect.w for line in viewer.tooltip)
    frame(viewer, move(viewer.header_rect.center))  # the panel's lines naming the sweep: in full too
    assert viewer._status_text() == sv.HEADER_HELP and viewer.tooltip[:2] == ["b.csv", "4 runs, full grid"]
    assert viewer.tooltip[2] == "Varied: Density"


def test_a_csv_in_which_nothing_varies_is_labelled_so_and_one_run_is_one(make_viewer, tmp_path, monkeypatch):
    res = tmp_path / "results"
    same = write(grid_frame({"density": [40.0]}, seeds=3), res / "plan6.csv")  # three seeds of one setting
    viewer = make_viewer(same)
    assert viewer.data.design == "single" and viewer.x == "seed"
    assert viewer.message[0].startswith("Loaded plan6.csv (3 runs, no setting varied, no meta file)")
    rendered = []
    render = viewer.fonts.render
    monkeypatch.setattr(viewer.fonts, "render", lambda t, *a, **k: rendered.append(t) or render(t, *a, **k))
    viewer.draw()
    monkeypatch.undo()
    assert "3 runs  ·  no setting varied" in rendered  # (not "one setting · nothing varied")
    viewer.load(write(grid_frame({"density": [40.0]}, seeds=1), res / "r_summary.csv"))
    assert viewer.message[0].startswith("Loaded r_summary.csv (1 run, no setting varied")
    assert "nothing is written" not in sv.BUTTON_HELP["cancel"] and "loaded" in sv.BUTTON_HELP["cancel"]


@pytest.mark.parametrize("seed", [-3, 1.5, True, "4", float("nan")])
def test_base_settings_with_an_invalid_seed_are_refused_at_start(make_viewer, tmp_path, seed):
    """A hand-written --base-settings file is checked as SimConfig and --seed check a seed (a whole number
    >= 0), when the viewer starts, not by a failed sweep after Run."""
    path = tmp_path / "base.json"
    path.write_text(json.dumps({"density": 40, "seed": seed}), encoding="utf-8")
    with pytest.raises(ValueError, match="seed must be a whole number >= 0"):
        sv.load_base(path)
    viewer = make_viewer(None, base_settings=path)
    assert viewer.base_seed is None and viewer.base == {}
    assert viewer.message[0].startswith("Could not read the base settings base.json: seed must be")
    assert sv.load_base({"seed": 4.0, "density": 40}) == ({"density": 40.0}, 4, [])


def test_metric_row_comes_into_sight_after_a_load_on_another_page(make_viewer, tmp_path):
    """The list of metrics exists only on the chart page: a CSV loaded from the Files tab (or a sweep that
    finished on the New sweep tab) scrolls it to the metric when the chart shows again."""
    res = tmp_path / "results"
    many = grid_frame({"density": [20.0, 40.0]}).assign(**{f"m{k:02d}": float(k) for k in range(40)})
    first, second = write(many, res / "first.csv"), write(many, res / "second.csv")
    viewer = make_viewer(first, size=(1100, 700))
    viewer.set_metric(viewer.data.metrics[-1])
    viewer._scroll_to_metric()

    def row_in_sight() -> bool:
        w = widget(viewer, "metric", viewer.metric)
        return viewer.regions["panel"].contains(w.rect)

    assert row_in_sight() and viewer.scroll["panel"] > 0
    frame(viewer, key(pygame.K_o))
    assert viewer.page == "files"
    click(viewer, widget(viewer, "file", str(second)).rect.center)
    frame(viewer, key(pygame.K_o))
    assert viewer.page == "chart" and viewer.metric == "m39" and row_in_sight()
    viewer.scroll["panel"] = 0  # scrolled back by hand: a plain switch of pages leaves it there
    viewer._layout_panel()
    frame(viewer, key(pygame.K_o), key(pygame.K_o))
    assert viewer.scroll["panel"] == 0


def test_scatter_keeps_the_colour_of_a_run_that_is_its_own_mean(make_viewer, tmp_path):
    """Where an x value has a single run, its mean is that run: the mean is marked with a ring round the
    dot instead of a disc over it, so the dot keeps the colour of its colour setting."""
    rows = [{**default_settings(), "n_lanes": n, "n_arms": arms, "seed": 0, "flux": 0.1 * n}
            for n, arms in ((1, 3), (2, 6), (3, 4), (4, 5), (4, 3))]
    viewer = make_viewer(write(pd.DataFrame(rows), tmp_path / "results" / "plan.csv"))
    viewer.set_view("scatter")
    viewer.mouse = (-1, -1)
    viewer.draw()
    sx, sy, ok = viewer._dots
    text = tuple(sv.UI["text"])
    for i in np.flatnonzero(ok):
        pixel = tuple(viewer.screen.get_at((int(round(sx[i])), int(round(sy[i]))))[:3])
        assert pixel != text, f"run {i}'s dot is hidden under its mean"


def test_window_size_below_the_minimum_is_raised(make_viewer, csvs, capsys):
    viewer = make_viewer(csvs["two"], size=(400, 300))
    assert viewer.screen.get_size() == ui.clamp_size((400, 300))
    assert viewer.screen.get_size()[1] >= ui.MIN_WINDOW[1]
    assert "below the smallest the layout supports" in capsys.readouterr().err


def test_entry_delay_help_states_every_level_of_service_rule():
    """The sweep chart's note and the metric's help give the limits as the main viewer does (≤) and the
    rule its own hover readout applies: F whenever an entrance gets more arrivals than it can let in."""
    info = sv.metric_info("mean_entry_delay", "entry_rate", open_demand=True)
    for text in (info.note, info.help):
        assert "A ≤ 10" in text and "E ≤ 50 s, F above" in text and "<=" not in text
        assert "v/c > 1" in text or "entry_vc > 1" in text


def test_the_sweep_viewer_needs_nothing_of_the_main_viewer():
    """Its shared pieces come from ui and textfmt: starting it does not import the main viewer."""
    code = "import sys, roundabout_sim.sweep_viewer; print('roundabout_sim.app' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                         env={**os.environ, "PYTHONPATH": str(Path(sv.__file__).resolve().parents[1])})
    assert out.stdout.strip().splitlines()[-1] == "False", out.stderr
