"""Sweeps of many settings: --vary grids, --plan files, --sample designs, preset / random base settings,
the meta file, the figures and ``experiments view``. Tiny runs (a few simulated seconds) into tmp folders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.text import Annotation

from roundabout_sim import experiments, plots, presets
from roundabout_sim import sweep_viewer as sv
from roundabout_sim.config import PARAMS, SimConfig
from tests.helpers import PNG_MAGIC, assert_png, strata

META_KEYS = {"version", "design", "varied", "fixed", "preset", "random", "sample_groups", "seeds", "base_seed",
             "warmup", "measure", "interval", "command"}
TINY = ["--warmup", "1", "--measure", "3", "--workers", "1"]


def sweep(tmp_path: Path, capsys, *argv: str, name: str = "s") -> tuple[pd.DataFrame, dict, str]:
    out = tmp_path / name
    experiments.main(["sweep", *argv, "--out", str(out)])
    printed = capsys.readouterr().out
    meta = json.loads(Path(f"{out}_meta.json").read_text(encoding="utf-8"))
    return pd.read_csv(f"{out}.csv"), meta, printed


def progress_lines(printed: str) -> list[str]:
    return [line for line in printed.splitlines() if sv.parse_progress(line)]


# ================================================================== grids of many settings
@pytest.fixture(scope="module")
def three(tmp_path_factory):
    """A real 3-setting grid with its figures (shared: the figures take a few seconds)."""
    out = tmp_path_factory.mktemp("three") / "grid"
    import contextlib
    import io
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        experiments.main(["sweep", "--vary", "density=20,40", "--vary", "n_lanes=1,2", "--vary", "exit_prob=0,0.1",
                          "--seeds", "2", *TINY, "--out", str(out)])
    return out, buffer.getvalue()


def test_three_settings_make_a_full_grid(three):
    out, printed = three
    df = pd.read_csv(f"{out}.csv")
    assert len(df) == 2 * 2 * 2 * 2  # every combination x the seeds
    # The varied settings first, the last one first, so that the x is just before the seed (as with --x).
    assert list(df.columns[:4]) == ["exit_prob", "n_lanes", "density", "seed"]
    assert list(df.columns[4:]) [:len(experiments.CONFIG_NAMES) - 3] == \
        [n for n in experiments.CONFIG_NAMES if n not in ("exit_prob", "n_lanes", "density")]
    # Rows in run order: the x varies fastest, the seeds within each point.
    first = df[["exit_prob", "n_lanes", "density", "seed"]].head(5).to_numpy().tolist()
    assert first == [[0.0, 1, 20.0, 0], [0.0, 1, 20.0, 1], [0.0, 1, 40.0, 0], [0.0, 1, 40.0, 1], [0.0, 2, 20.0, 0]]
    assert printed.splitlines()[0].startswith("16 runs (2 density x 2 n_lanes x 2 exit_prob x 2 seeds) on 1 worker")
    # Progress lines the sweep viewer parses, naming every varied setting.
    lines = progress_lines(printed)
    assert len(lines) == 16 and sv.parse_progress(lines[-1]) == (16, 16)
    assert lines[0].startswith("[  1/16] exit_prob=0.0 n_lanes=1 density=20.0 seed=0: flux ")
    assert sv.TOTAL.match(printed.splitlines()[0])
    # The means table groups by all three; the arm-share table still works with exit_prob one of three.
    assert "Means across seeds" in printed and "exit_prob n_lanes density" in printed
    assert "Jamming due to entrances/exits" in printed and "exit_prob=0.1" in printed


def test_meta_file_follows_the_sweep_viewer_contract(three):
    out, _ = three
    meta = json.loads(Path(f"{out}_meta.json").read_text(encoding="utf-8"))
    assert set(meta) == META_KEYS
    assert meta["version"] == 1 and meta["design"] == "grid"
    assert meta["varied"] == [{"name": "density", "values": [20.0, 40.0]}, {"name": "n_lanes", "values": [1, 2]},
                              {"name": "exit_prob", "values": [0.0, 0.1]}]
    assert meta["fixed"] == {} and meta["preset"] is None and meta["random"] is None
    assert (meta["seeds"], meta["base_seed"], meta["warmup"], meta["measure"], meta["interval"]) == (2, 0, 1, 3, 0.5)
    assert meta["command"].startswith("python -m roundabout_sim.experiments sweep --vary density=20,40 --vary ")
    assert not list(Path(out).parent.glob(".*tmp"))  # written atomically: no temporary file left behind


def test_three_setting_figures_are_written(three):
    out, printed = three
    assert_png(Path(f"{out}.png"))
    assert_png(Path(f"{out}_heatmap.png"))
    assert f"{out}_heatmap.png" in printed  # named in the "wrote ..." line


def test_sweep_viewer_reads_the_grid(three):
    out, _ = three
    data = sv.load_sweep(f"{out}.csv")
    assert data.design == "grid" and data.ordered
    assert [v.name for v in data.variables] == ["density", "n_lanes", "exit_prob"]
    assert sv.axis_order(data) == ["density", "n_lanes", "exit_prob"]
    assert data.runs == 16 and "seed" not in [v.name for v in data.variables]
    # Every measured column of the CSV has a definition (experiments.DEFINITIONS) or a sweep chart.
    described = set(sv.definitions()) | set(sv._panels(""))
    assert not [m for m in data.metrics + data.text_metrics if m not in described]


def test_x_and_series_sweep_is_as_it_always_was(tmp_path, capsys, no_figures):
    """--x / --series: the same rows, values and column order as the runs themselves give (what the sweep has
    always written), and the same as the equivalent --vary sweep."""
    df, meta, printed = sweep(tmp_path, capsys, "--x", "reaction_time=0.2,0.8", "--series", "n_lanes=2,1",
                              "--set", "n_arms=0", "--seeds", "1", *TINY)
    assert printed.splitlines()[0] == ("4 runs (2 x 2 series x 1 seeds) on 1 worker(s); 1 s warm-up + 3 s "
                                       "measured each")
    assert progress_lines(printed)[0].startswith("[  1/4] n_lanes=2 reaction_time=0.2 seed=0: flux ")
    rows = [experiments._sweep_job(({"n_arms": 0, "reaction_time": x, "n_lanes": s}, 0, 1.0, 3.0, 0.5))
            for s in (2, 1) for x in (0.2, 0.8)]
    first = ["n_lanes", "reaction_time", "seed"] + [n for n in experiments.CONFIG_NAMES
                                                     if n not in ("n_lanes", "reaction_time")]
    expected = pd.DataFrame(rows)
    expected = expected[first + [c for c in expected.columns if c not in first]]
    expected.to_csv(tmp_path / "expected.csv", index=False)
    pd.testing.assert_frame_equal(df, pd.read_csv(tmp_path / "expected.csv"))
    assert meta["varied"] == [{"name": "reaction_time", "values": [0.2, 0.8]}, {"name": "n_lanes", "values": [2, 1]}]
    assert meta["fixed"] == {"n_arms": 0}
    # The figure: x and series as given; a heatmap of the two as well.
    names = [c.name for c in no_figures]
    assert names == ["sweep_figure", "heatmap_figure"]
    assert no_figures[0].args[1:3] == ("reaction_time", "n_lanes")
    assert "band = min-max across seeds" in no_figures[0].kwargs["subtitle"]
    assert "each cell = the single run at that pair of values;" in no_figures[1].kwargs["subtitle"]  # 1 seed
    # --vary in the same order is the same sweep.
    same, _, _ = sweep(tmp_path, capsys, "--vary", "reaction_time=0.2,0.8", "--vary", "n_lanes=2,1", "--set",
                       "n_arms=0", "--seeds", "1", *TINY, name="vary")
    pd.testing.assert_frame_equal(df, same)


def test_three_settings_figure_averages_over_the_rest(tmp_path, capsys, no_figures):
    sweep(tmp_path, capsys, "--x", "density=20,40", "--vary", "road_condition=dry,wet", "--vary",
          "lane_change_motive=advantage,random", "--seeds", "1", "--warmup", "0", "--measure", "1", "--workers", "1")
    main, heat = no_figures
    assert main.args[1:3] == ("density", "road_condition")  # x and series: the first two
    subtitle = main.kwargs["subtitle"]
    assert "averaged over Lane-change motive (advantage, random)" in subtitle and "2 runs at each point" in subtitle
    assert heat.name == "heatmap_figure" and "averaged over" in heat.kwargs["subtitle"]
    assert [p.column for p in heat.args[3]] == list(plots.HEADLINE)


def test_series_cap_applies_to_the_second_setting_only(tmp_path, capsys, no_figures):
    with pytest.raises(SystemExit):
        experiments.main(["sweep", "--vary", "density=10,20", "--vary", "radius=10:100:10", "--out",
                          str(tmp_path / "x")])
    assert "at most 8 --vary values for radius" in capsys.readouterr().err
    # Nine values are fine as the x or as a third setting.
    df, _, _ = sweep(tmp_path, capsys, "--vary", "radius=20:100:10", "--vary", "n_lanes=1",
                     "--vary", "exit_prob=0:0.8:0.1", "--seeds", "1", "--warmup", "0", "--measure", "0.2",
                     "--workers", "1")
    assert len(df) == 9 * 1 * 9


def test_arm_share_table_with_exit_prob_one_of_three_settings():
    rows = [dict(road_condition=r, exit_prob=p, density=d, jam_fraction=j)
            for r, p, d, j in [("dry", 0.0, 40.0, 0.1), ("dry", 0.2, 40.0, 0.4), ("dry", 0.0, 80.0, 0.0),
                               ("dry", 0.2, 80.0, 0.0), ("wet", 0.0, 40.0, 0.2), ("wet", 0.2, 40.0, 0.5),
                               ("wet", 0.0, 80.0, 0.3), ("wet", 0.2, 80.0, 0.6)]]
    table = experiments.arm_share_table(pd.DataFrame(rows), ["road_condition", "exit_prob", "density"])
    assert list(table.columns) == ["exit_prob=0.2"]
    assert table.index.names == ["road_condition", "density"]
    assert table.loc[("dry", 40.0), "exit_prob=0.2"] == pytest.approx(0.75)
    assert table.loc[("wet", 80.0), "exit_prob=0.2"] == pytest.approx(0.5)
    assert np.isnan(table.loc[("dry", 80.0), "exit_prob=0.2"])  # nothing jammed


# ================================================================== plans
def write_plan(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_plan_runs_each_row_with_its_seeds(tmp_path, capsys, no_figures):
    plan = write_plan(tmp_path / "plan.csv", " density , n-lanes ,seed\n20,1,\n40,,5\n,3,\n")
    df, meta, printed = sweep(tmp_path, capsys, "--plan", str(plan), "--set", "n_lanes=2", "--density", "55",
                              "--seed", "10", "--seeds", "2", *TINY, name="out")
    assert printed.splitlines()[0].startswith("6 runs (3 plan rows x 2 seeds)")
    assert progress_lines(printed)[0].startswith("[  1/6] row 1: density=20 n_lanes=1 seed=10: ")
    assert list(df.columns[:3]) == ["n_lanes", "density", "seed"]  # the x (first plan column) just before the seed
    # Empty cells take the base (--set / flags); the seed column is that row's base seed (else --seed).
    assert df[["density", "n_lanes", "seed"]].to_numpy().tolist() == [
        [20.0, 1, 10], [20.0, 1, 11], [40.0, 2, 5], [40.0, 2, 6], [55.0, 3, 10], [55.0, 3, 11]]
    assert meta["design"] == "plan"
    assert meta["varied"] == [{"name": "density", "values": [20.0, 40.0, 55.0]},
                              {"name": "n_lanes", "values": [1, 2, 3]}]
    assert meta["fixed"] == {} and meta["seeds"] == 2 and meta["base_seed"] == 10
    assert [c.name for c in no_figures] == ["sample_figure"]
    assert no_figures[0].args[1] == ["density", "n_lanes"] and "3 plan rows" in no_figures[0].kwargs["subtitle"]
    assert "Means across seeds" in printed
    data = sv.load_sweep(tmp_path / "out.csv")
    assert data.design == "plan" and [v.name for v in data.variables] == ["density", "n_lanes"]


def test_plan_column_with_one_value_is_fixed(tmp_path, capsys, no_figures):
    plan = write_plan(tmp_path / "plan.csv", "road_condition,density\nwet,20\nwet,30\n")
    df, meta, _ = sweep(tmp_path, capsys, "--plan", str(plan), "--seeds", "1", *TINY, name="out")
    assert meta["varied"] == [{"name": "density", "values": [20.0, 30.0]}]
    assert meta["fixed"] == {"road_condition": "wet"} and (df["road_condition"] == "wet").all()


@pytest.mark.parametrize("text, message", [
    ("density,bogus\n20,1\n", "unknown column(s) 'bogus'"),
    ("density\n", "has no rows"),
    ("density,seed\n20,1.5\n", "line 2: seed='1.5'"),
    ("density,n_lanes\n20,1\nlots,2\n", "line 3: density='lots'"),
    ("init_spacing\ndiagonal\n", "init_spacing must be one of"),
    ("radius\n-5\n", "plan.csv, line 2: radius must be > 0"),
    ("density,exit_prob\n20,0.1\n\n30,7\n", "plan.csv, line 4: exit_prob"),  # the file's line, blank ones counted
    ("density,exit_prob\nnan,0.1\n", "invalid settings at plan"),
    ("density,n_lanes\n20,2.5\n", "line 2: n_lanes='2.5': bad value '2.5' for n_lanes: must be a whole number"),
    ("density,density\n20,30\n", "column 'density' is given twice"),
    ("density,\n20,5\n", "line 2: column 2 has values but no name"),
    ("", "is empty"),
])
def test_bad_plans_are_reported_before_anything_runs(tmp_path, capsys, text, message):
    plan = write_plan(tmp_path / "plan.csv", text)
    with pytest.raises(SystemExit) as exc:
        experiments.main(["sweep", "--plan", str(plan), "--out", str(tmp_path / "out" / "s")])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert message in err and "::" not in err
    if "unknown column" in message:
        assert "Valid names: seed, n_lanes, n_arms" in err  # lists what is valid
    assert not (tmp_path / "out").exists()


def test_plan_header_as_written(tmp_path, capsys, no_figures):
    """A trailing comma (as spreadsheets write) is no column; a byte-order mark is no part of the first name."""
    plan = write_plan(tmp_path / "plan.csv", "﻿density,n_lanes,\n20,1,\n30,2,\n")
    df, meta, _ = sweep(tmp_path, capsys, "--plan", str(plan), "--seeds", "1", *TINY, name="out")
    assert [v["name"] for v in meta["varied"]] == ["density", "n_lanes"]
    assert df[["density", "n_lanes"]].to_numpy().tolist() == [[20.0, 1], [30.0, 2]]


def test_short_plan_rows_take_the_base_setting_with_a_note(tmp_path, capsys, no_figures):
    plan = write_plan(tmp_path / "plan.csv", "density,n_lanes\n30,1\n60\n")
    df, meta, printed = sweep(tmp_path, capsys, "--plan", str(plan), "--seeds", "1", *TINY, name="out")
    assert (f"note: plan {plan}, line 3: fewer cells than the header; the missing ones (n_lanes) take the base "
            "setting, as empty cells do") in printed
    assert df["n_lanes"].tolist() == [1, SimConfig().n_lanes]  # as an empty cell
    plan = write_plan(tmp_path / "whole.csv", "density,n_lanes,\n30,1,\n60,2,\n")  # trailing commas: whole rows
    _, _, printed = sweep(tmp_path, capsys, "--plan", str(plan), "--seeds", "1", *TINY, name="full")
    assert "fewer cells" not in printed


def test_closed_ring_note_from_the_runs(tmp_path, capsys, no_figures):
    """exit_prob on a closed ring: the note, whether n_arms = 0 comes from --set or from a plan column."""
    plan = write_plan(tmp_path / "plan.csv", "density,exit_prob,n_arms\n60,0,0\n60,0.1,0\n")
    _, _, printed = sweep(tmp_path, capsys, "--plan", str(plan), "--seeds", "1", "--warmup", "0", "--measure",
                          "1", "--workers", "1", name="out")
    assert "(exit_prob does nothing on a closed ring: n_arms = 0)" in printed
    assert "Jamming due to entrances/exits" not in printed


def test_a_setting_that_does_nothing_in_every_run_is_noted(tmp_path, capsys, no_figures):
    """Entrance chance swept in closed mode gives identical runs: the sweep says so before it runs."""
    short = ["--seeds", "1", "--warmup", "0", "--measure", "1", "--workers", "1"]
    _, _, printed = sweep(tmp_path, capsys, "--x", "entry_rate=0.05,0.3", "--series", "demand_imbalance=0,1",
                          *short)
    assert ("note: entry_rate and demand_imbalance act only in open mode, and every run here is closed: their "
            "values change nothing (add --set traffic_mode=open, or vary traffic_mode too)") in printed
    _, _, printed = sweep(tmp_path, capsys, "--x", "entry_rate=0.05,0.3", "--set", "traffic_mode=open", *short)
    assert "change nothing" not in printed
    # Varying what they depend on makes them act in some runs: no note.
    assert experiments.idle_notes(["entry_rate"], [SimConfig(), SimConfig(traffic_mode="open")]) == []
    assert experiments.idle_notes(["exit_prob", "density"], [SimConfig(n_arms=0)]) == [
        "exit_prob does nothing without entrances, and every run here has none (n_arms = 0): its values change "
        "nothing (set n_arms above 0)"]
    assert experiments.idle_notes(["truck_length", "seed"], [SimConfig()])[0].startswith("truck_length does nothing "
                                                                                          "without trucks")


def test_the_csv_is_written_before_its_meta_file(tmp_path, capsys, no_figures, monkeypatch):
    """Through the one atomic writer (state.write_atomic), the CSV first: a sweep stopped in between
    leaves a CSV the viewer reads, never a meta file without its CSV."""
    from roundabout_sim import state

    written = []
    write = state.write_atomic
    monkeypatch.setattr(state, "write_atomic", lambda path, fn: written.append(Path(path).name) or write(path, fn))
    df, meta, _ = sweep(tmp_path, capsys, "--x", "density=20,30", "--seeds", "1", "--warmup", "0", "--measure", "1",
                        "--workers", "1", name="order")
    assert written == ["order.csv", "order_meta.json"] and len(df) == 2 and meta["design"] == "grid"
    assert not list(tmp_path.glob(".*.tmp"))


def test_plan_is_never_overwritten(tmp_path, capsys):
    plan = write_plan(tmp_path / "p.csv", "density\n20\n")
    with pytest.raises(SystemExit):
        experiments.main(["sweep", "--plan", str(plan), "--out", str(tmp_path / "p")])
    assert "over the plan" in capsys.readouterr().err and plan.read_text() == "density\n20\n"


# ================================================================== random samples
def test_sample_is_a_latin_hypercube_over_the_ranges(tmp_path, capsys, no_figures):
    args = ["--sample", "8", "--vary", "density=10:90", "--vary", "lane_change_motive=advantage,random",
            "--vary", "n_lanes=1,3", "--seeds", "1", "--warmup", "0", "--measure", "0.5", "--workers", "1"]
    df, meta, printed = sweep(tmp_path, capsys, *args, "--seed", "4")
    assert len(df) == 8 and printed.splitlines()[0].startswith("8 runs (8 sampled points x 1 seeds)")
    assert list(df.columns[:4]) == ["n_lanes", "lane_change_motive", "density", "seed"]
    # One density in each eighth of 10-90 (Latin hypercube strata); the choices dealt out evenly.
    strata = np.floor((df["density"] - 10) / 80 * 8).clip(0, 7).astype(int)
    assert sorted(strata) == list(range(8))
    assert df["lane_change_motive"].value_counts().to_dict() == {"advantage": 4, "random": 4}
    assert sorted(df["n_lanes"]) == [1] * 4 + [3] * 4  # "1,3" is two values, not the range 1-3
    assert (df["seed"] == 4).all()
    assert meta["design"] == "sample" and meta["base_seed"] == 4
    assert meta["varied"] == [{"name": "density", "range": [10.0, 90.0]},
                              {"name": "lane_change_motive", "values": ["advantage", "random"]},
                              {"name": "n_lanes", "values": [1, 3]}]
    assert "Rank correlation (Spearman)" in printed and "Means across seeds" not in printed
    assert progress_lines(printed)[0].startswith("[  1/8] point 1: density=")
    # Reproducible by --seed, different for another seed.
    again, _, _ = sweep(tmp_path, capsys, *args, "--seed", "4", name="again")
    other, _, _ = sweep(tmp_path, capsys, *args, "--seed", "5", name="other")
    pd.testing.assert_frame_equal(df[["n_lanes", "lane_change_motive", "density"]],
                                  again[["n_lanes", "lane_change_motive", "density"]])
    assert list(other["density"]) != list(df["density"])
    data = sv.load_sweep(tmp_path / "s.csv")
    assert data.design == "sample" and [v.name for v in data.variables] == ["density", "lane_change_motive",
                                                                            "n_lanes"]


def design(*argv: str) -> experiments.Design:
    """The design a sweep command line gives, without running it."""
    args = experiments.build_parser().parse_args(["sweep", *argv])
    parser = args.command_parser
    setup = experiments._base(args, parser)
    return experiments._sample(experiments._variables(args, parser), setup, args, parser)


def test_sample_without_vary_is_a_latin_hypercube_over_the_plausible_ranges(tmp_path, capsys, no_figures):
    df, meta, printed = sweep(tmp_path, capsys, "--sample", "8", "--sample-groups", "Drivers,lane chg",
                              "--set", "reaction_time=0.5", "--seed", "7", "--seeds", "1", "--warmup", "0",
                              "--measure", "0.5", "--workers", "1")
    ranges = presets.plausible_ranges(["Drivers", "Lane chg"])
    names = [v["name"] for v in meta["varied"]]
    assert names == [n for n in ranges if n != "reaction_time"]  # those groups, but not the --set one
    assert (df["reaction_time"] == 0.5).all() and "reaction_time keeps the value set with --set" in printed
    assert (df["n_lanes"] == SimConfig().n_lanes).all()  # other groups at the base
    assert list(df.columns[:len(names)]) == names[::-1] and df.columns[len(names)] == "seed"
    # Exactly presets.latin_hypercube over those plausible ranges, seeded by --seed.
    del ranges["reaction_time"]
    expected = presets.latin_hypercube(8, {k: list(map(str, v)) if isinstance(v, list) else v
                                           for k, v in ranges.items()}, 7)
    for name in names:
        assert list(df[name]) == [p[name] for p in expected], name
    for entry in meta["varied"]:  # numbers as their plausible range, choices as the values drawn
        name, spec = entry["name"], ranges[entry["name"]]
        if isinstance(spec, tuple):
            assert entry["range"] == list(spec)
            lo, hi = spec
            if (hi - lo) / next(p.step for p in PARAMS if p.name == name) + 1 > 8:  # 8+ slider values
                assert strata(df[name], lo, hi, 8) == list(range(8)), name  # one in each eighth
        else:
            assert entry["values"] == spec
    assert df["lane_change_motive"].value_counts().to_dict() == {"advantage": 4, "random": 4}
    # Every setting (no groups): the lanes and road conditions dealt out evenly, the density stratified.
    df, meta, _ = sweep(tmp_path, capsys, "--sample", "8", "--seeds", "1", "--warmup", "0", "--measure", "0.2",
                        "--workers", "1", name="all")
    groups = {p.name: p.group for p in PARAMS}
    assert len({groups[v["name"]] for v in meta["varied"]}) == len(set(groups.values()))
    assert sorted(df["n_lanes"].value_counts()) == [1, 1, 2, 2, 2]
    assert sorted(df["road_condition"].value_counts()) == [2, 2, 2, 2]
    assert strata(df["density"], 5, 110, 8) == list(range(8))
    assert {"name": "density", "range": [5.0, 110.0]} in meta["varied"]


def test_samples_with_neighbouring_seeds_share_no_points():
    names = [p.name for p in PARAMS]
    for argv in (["--sample", "10"], ["--sample", "10", "--sample-groups", "Drivers"]):
        one, two = (design(*argv, "--seed", s) for s in ("1", "2"))
        shared = {tuple(p[n] for n in names if n in p) for p in one.points} & \
            {tuple(p[n] for n in names if n in p) for p in two.points}
        assert not shared, argv
        assert one.seeds == [1] * 10  # each point runs from the base seed (run k: base + k)
    again = design("--sample", "10", "--seed", "1")
    assert again.points == design("--sample", "10", "--seed", "1").points


def test_sample_redrawing_every_preset_setting_is_an_error(capsys):
    for flag in (["--preset", "etoile"], ["--random", "4"]):
        with pytest.raises(SystemExit) as exc:
            design("--sample", "3", *flag)
        assert exc.value.code == 2
        assert f"{flag[0]} would do nothing here" in capsys.readouterr().err
    # Drawing only some groups keeps the preset's other settings.
    args = experiments.build_parser().parse_args(["sweep", "--sample", "3", "--preset", "etoile",
                                                  "--sample-groups", "Drivers"])
    setup = experiments._base(args, args.command_parser)
    sample = experiments._sample([], setup, args, args.command_parser)
    groups = {p.name: p.group for p in PARAMS}
    assert {groups[n] for n in sample.names} == {"Drivers"}
    assert SimConfig().updated(**{**setup.changes, **sample.points[0]}).n_lanes == presets.apply("etoile").n_lanes


@pytest.mark.parametrize("argv, message", [
    (["--sample", "0"], "N >= 1"),
    (["--sample", "3", "--sample-groups", "bogus"], "unknown group(s) 'bogus'"),
    (["--sample-groups", "Road", "--x", "density=1,2"], "--sample-groups needs --sample"),
    (["--sample", "3", "--vary", "density=1:5", "--sample-groups", "Road"], "only without --vary"),
    (["--sample", "3", "--vary", "road_condition=dry:ice"], "a choice takes a comma list"),
    (["--sample", "3", "--vary", "density=9:5"], "stop >= start"),
    (["--plan", "p.csv", "--x", "density=1,2"], "--plan takes the varied settings"),
    (["--series", "n_lanes=1,2"], "--series needs --x"),
    (["--x", "density=1,2", "--vary", "density=3"], "must be different"),
    (["--preset", "nowhere", "--x", "density=1,2"], "unknown preset 'nowhere'"),
    (["--preset", "etoile", "--random", "3", "--x", "density=1,2"], "not allowed with"),
    (["--random", "-1", "--x", "density=1,2"], "seed must be"),
    (["--x", "density=20", "--seed", "-1"], "argument --seed: seed must be a whole number >= 0"),
    (["--sample", "3", "--preset", "etoile"], "--preset would do nothing here"),
    # A whole-number setting takes whole numbers (it used to round 2.5 to 2 silently).
    (["--x", "n_lanes=1:3:0.5"], "bad value '1.5' for n_lanes: must be a whole number"),
    (["--set", "n_lanes=2.5", "--x", "density=20"], "must be a whole number"),
    (["--sample", "4", "--vary", "n_lanes=1.5,2.5"], "must be a whole number"),
])
def test_bad_design_arguments_exit_cleanly(tmp_path, capsys, argv, message):
    with pytest.raises(SystemExit) as exc:
        experiments.main(["sweep", *argv, "--out", str(tmp_path / "out")])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def test_run_rejects_a_negative_seed(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        experiments.main(["run", "--seed", "-1", "--duration", "1", "--out", str(tmp_path / "r")])
    assert exc.value.code == 2 and "seed must be a whole number >= 0" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def test_rank_correlations_rank_the_complete_pairs_only():
    """Spearman over the runs that have both values (the entry delay is missing when no car entered)."""
    df = pd.DataFrame({"density": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "delay": [np.nan, 3.0, np.nan, 1.0, 2.0, np.nan],
                       "flux": [1.0, 2.0, 3.0, 4.0, 5.0, 7.0]})
    table = experiments.rank_correlations(df, ["density"], ["delay", "flux"])
    # pairs (2, 3), (4, 1), (5, 2): ranks (1, 3), (2, 1), (3, 2) -> -0.5 (ranks over all six rows gave -0.65)
    assert table.loc["density", "delay"] == pytest.approx(-0.5)
    assert table.loc["density", "flux"] == pytest.approx(1.0)


def test_sample_figure_draws_a_row_per_setting():
    df = pd.DataFrame({"density": [10.0, 20.0, 30.0, 40.0], "road_condition": ["dry", "wet", "dry", "snow"],
                       "n_lanes": [1, 2, 3, 2], "seed": 0, "flux": [0.1, 0.3, 0.2, 0.4],
                       "jam_fraction": [0.0, 0.1, np.nan, 0.3], "stops_per_car_min": [0.0, 0.5, 1.0, 2.0]})
    fig = plots.sample_figure(df, ["density", "road_condition", "n_lanes"], plots.headline_panels(), title="t",
                              subtitle="s", param_label=str)
    assert len(fig.axes) == 1 + 3 * 3  # header + settings x metrics
    road = fig.axes[1 + 3]
    assert [t.get_text() for t in road.get_xticklabels()] == ["dry", "wet", "snow"]  # in the choices' order


# ================================================================== base settings
def test_preset_and_set_and_varied_values_in_order(tmp_path, capsys, no_figures):
    df, meta, printed = sweep(tmp_path, capsys, "--preset", "Sugiyama 2008 ring", "--set", "radius=40",
                              "--density", "10", "--x", "density=60,80", "--seeds", "1", *TINY)
    preset = presets.apply("sugiyama2008")
    assert list(df["density"]) == [60.0, 80.0]  # varied beats --set
    assert (df["radius"] == 40).all()  # --set beats the preset
    assert (df["n_lanes"] == preset.n_lanes).all() and (df["n_arms"] == preset.n_arms).all()  # the preset
    assert (df["mean_speed"] == preset.mean_speed).all()
    assert meta["preset"] == "sugiyama2008" and meta["random"] is None
    assert meta["fixed"]["radius"] == 40 and meta["fixed"]["n_arms"] == preset.n_arms and "density" not in meta["fixed"]
    assert "Base settings: preset sugiyama2008, then radius=40\n" in printed  # density is varied anyway
    assert no_figures[0].kwargs["subtitle"].startswith("Preset sugiyama2008: ")


def test_random_base_settings(tmp_path, capsys, no_figures):
    df, meta, _ = sweep(tmp_path, capsys, "--random", "3", "--set", "n_arms=2", "--x", "exit_prob=0.1,0.3",
                        "--seeds", "1", *TINY)
    drawn = presets.random_changes(3, fixed={"n_arms": 2})  # the explicit settings take part in the links
    assert (df["n_arms"] == 2).all() and list(df["exit_prob"]) == [0.1, 0.3]
    for name in ("radius", "mean_speed", "reaction_time", "road_condition"):
        assert (df[name] == drawn[name]).all(), name
    assert meta["random"] == 3 and meta["preset"] is None
    assert "Random settings (seed 3)" in no_figures[0].kwargs["subtitle"]


@pytest.mark.parametrize("argv", [["--list-presets"], ["run", "--list-presets"], ["sweep", "--list-presets"]])
def test_list_presets(capsys, argv):
    with pytest.raises(SystemExit) as exc:
        experiments.main(argv)
    assert exc.value.code in (0, None)
    assert capsys.readouterr().out == presets.list_text()


@pytest.fixture
def no_dashboard(monkeypatch):
    """Skip drawing the run dashboard; records its keyword arguments."""
    calls = []

    class Fake:
        def savefig(self, path):
            Path(path).write_bytes(PNG_MAGIC)

    def fake(*args, **kwargs):
        calls.append(kwargs)
        return Fake()
    monkeypatch.setattr(plots, "dashboard", fake)
    return calls


def test_run_from_a_preset_or_random_settings(tmp_path, capsys, no_dashboard):
    out = tmp_path / "run"
    experiments.main(["run", "--preset", "sugiyama2008", "--set", "density=90", "--duration", "4", "--out",
                      str(out)])
    assert "preset sugiyama2008" in capsys.readouterr().out
    assert no_dashboard[-1]["subtitle"].endswith(" | preset sugiyama2008")  # the dashboard names it
    row = pd.read_csv(f"{out}_summary.csv").iloc[0]
    preset = presets.apply("sugiyama2008")
    assert row["preset"] == "sugiyama2008" and "random" not in row.index
    assert row["density"] == 90 and row["radius"] == preset.radius and row["n_lanes"] == preset.n_lanes
    experiments.main(["run", "--random", "5", "--duration", "4", "--out", str(out)])
    capsys.readouterr()
    assert no_dashboard[-1]["subtitle"].endswith(" | random settings (seed 5)")
    row = pd.read_csv(f"{out}_summary.csv").iloc[0]
    drawn = presets.random_changes(5)
    assert row["random"] == 5 and "preset" not in row.index
    assert row["radius"] == pytest.approx(drawn["radius"]) and row["n_lanes"] == drawn["n_lanes"]
    # Without either, the summary has no such columns (as before).
    experiments.main(["run", "--duration", "2", "--out", str(out)])
    capsys.readouterr()
    assert not {"preset", "random"} & set(pd.read_csv(f"{out}_summary.csv").columns)


# ================================================================== figures
def test_open_mode_sweep_adds_the_entrance_charts(tmp_path, capsys, no_figures):
    """The figures of an open-mode sweep get the entrance charts (drawn in test_figure_layouts; the three
    fixture writes real PNGs)."""
    df, _, printed = sweep(tmp_path, capsys, "--x", "entry_rate=0.1,0.3", "--series", "follow_up=1,3", "--set",
                           "traffic_mode=open", "--seeds", "1", "--warmup", "0", "--measure", "4", "--workers", "1")
    assert (df["traffic_mode"] == "open").all() and "blocked/min" in printed
    figure, heat = no_figures
    assert (figure.name, heat.name) == ("sweep_figure", "heatmap_figure") and figure.kwargs["open_demand"]
    assert [p.column for p in heat.args[3]] == [p.column for p in plots.headline_panels(True)]
    assert (tmp_path / "s.png").exists() and (tmp_path / "s_heatmap.png").exists()


def test_figure_layouts():
    df = pd.DataFrame({"entry_rate": [0.1, 0.2, 0.1, 0.2], "follow_up": [1.0, 1.0, 3.0, 3.0], "seed": 0,
                       "flux": [0.2, 0.3, 0.2, 0.25], "mean_entry_delay": [2.0, 9.0, np.nan, 20.0]})
    fig = plots.sweep_figure(df, "entry_rate", "follow_up", x_label="x", series_label=str, title="t", subtitle="s",
                             open_demand=True)
    assert len(fig.axes) == 1 + 17  # header + 13 charts + the open-mode row of 4
    titles = [ax.get_title(loc="left") for ax in fig.axes]
    assert "Mean entry delay (s, queue wait)" in titles and "Blocked arrivals (per minute)" in titles
    heat = plots.heatmap_figure(df, "entry_rate", "follow_up", plots.headline_panels(True), x_label="x",
                                y_label="y", title="t", subtitle="s")
    assert sum(ax.get_title(loc="left") != "" for ax in heat.axes) == 5  # one heatmap per headline metric (+ colour bars)
    emergency = next(p for p in plots.sweep_panels("density", 0.5) if p.column == "emergency_per_min")
    assert plots.braking_limits() in emergency.note  # the braking limit on every road


def test_numbers_keep_three_digits_without_an_exponent():
    """Heatmap cells round first ("100" next to "100", not "100.0"); the printed table writes 1650 veh/h,
    not 1.65e+03."""
    from roundabout_sim import textfmt

    cells = {99.96: "100", 100.0: "100", 9.996: "10.0", 0.99996: "1.00", -99.96: "-100", 999.6: "1000",
             0.43412: "0.434", 0.0: "0", 1234.5: "1234"}
    for value, text in cells.items():
        assert plots._cell_text(value) == text, value
    assert textfmt.significant(float("nan")) == "-" and textfmt.significant(None, missing="–") == "–"
    for value, text in {1650.0: "1650", 999.6: "1000", 999.4: "999", 2.31e3: "2310", 0.000123: "0.000123",
                        12.345: "12.3"}.items():
        assert textfmt.table_number(value) == text, value
    df = pd.DataFrame({"x": [1.0, 2.0], "seed": 0, "throughput_per_h": [1650.0, 2310.0], "flux": [1.2, 1.3]})
    table = df.groupby("x")[["throughput_per_h", "flux"]].mean().to_string(float_format=textfmt.table_number)
    assert "e+0" not in table and "1650" in table


def test_subtitles_wrap_between_settings_not_inside_one():
    items = ["Ring radius = 18.25 m", "Mean desired speed = 11 m/s", "Reaction time spread (std) = 0.2 s",
             "Max queue = 50 cars", "Demand profile = constant"]
    text = ", ".join(items) + " | seed 0 | preset sugiyama2008\nsecond part"
    for width in (6, 8, 10, 14):
        lines = plots._wrap(text, width)
        assert lines[-1] == "second part" and "\xa0" not in "".join(lines)
        for item in items + ["preset sugiyama2008"]:  # each whole on one line
            assert any(item in line for line in lines), (width, item, lines)
    assert plots._wrap("x" * 40 + " y" * 30, 4)  # an item longer than a line still wraps


def test_sample_figure_written_for_a_random_sample(tmp_path, capsys):
    (tmp_path / "s_heatmap.png").write_bytes(PNG_MAGIC)  # from an earlier grid with this prefix
    sweep(tmp_path, capsys, "--sample", "6", "--vary", "density=10:80", "--vary", "road_condition=dry,snow",
          "--seeds", "1", "--warmup", "0", "--measure", "2", "--workers", "1")
    assert_png(tmp_path / "s.png")
    assert not (tmp_path / "s_heatmap.png").exists()  # it no longer matched the CSV


def test_subtitles_say_what_was_pooled(tmp_path, capsys, no_figures):
    _, _, printed = sweep(tmp_path, capsys, "--sample", "4", "--vary", "density=10:80", "--seeds", "1",
                          "--warmup", "0", "--measure", "0.5", "--workers", "1")
    assert "vary at the same time" not in no_figures[-1].kwargs["subtitle"]  # nothing else varies
    assert "vary at the same time" not in printed
    sweep(tmp_path, capsys, "--sample", "4", "--vary", "density=10:80", "--vary", "n_lanes=1,2", "--seeds", "1",
          "--warmup", "0", "--measure", "0.5", "--workers", "1")
    assert "The other settings vary at the same time" in no_figures[-1].kwargs["subtitle"]
    sweep(tmp_path, capsys, "--vary", "density=20,40", "--vary", "n_lanes=1,2", "--vary", "exit_prob=0.1",
          "--seeds", "1", "--warmup", "0", "--measure", "0.5", "--workers", "1")
    line, heat = no_figures[-2].kwargs["subtitle"], no_figures[-1].kwargs["subtitle"]
    assert "the single run at each point (at Exit chance (0.1))" in line and "1 runs" not in line
    assert "each cell = the single run at that pair of values (at Exit chance (0.1))" in heat


def figure_texts(fig) -> list:
    return [t for ax in fig.axes for t in ax.texts]


def test_heatmap_cell_values_never_overlap():
    """13 x values by 2: the values shrink to fit their cells (they used to run into each other)."""
    x = np.arange(10.0, 140.0, 10.0)
    df = pd.DataFrame({"density": np.tile(x, 2), "n_lanes": np.repeat([2, 3], x.size),
                       "flux": np.linspace(0.3501, 1.1, 2 * x.size), "jam_fraction": 0.5,
                       "stops_per_car_min": np.linspace(0.07, 0.6, 2 * x.size)})
    for size in (13, 3):
        part = df[df["density"] <= x[size - 1]]
        fig = plots.heatmap_figure(part, "density", "n_lanes", plots.headline_panels(), x_label="x", y_label="y",
                                   title="t", subtitle="s")
        fig.draw_without_rendering()
        renderer = fig.canvas.get_renderer()
        charts = 0
        for ax in fig.axes[1:]:
            cells = [t for t in ax.texts if t.get_visible() and not isinstance(t, Annotation)]
            if not cells:
                continue
            charts += 1
            box = ax.get_window_extent(renderer)
            extents = sorted((t.get_window_extent(renderer) for t in cells), key=lambda e: (round(e.y0), e.x0))
            for a, b in zip(extents, extents[1:]):
                assert round(a.y0) != round(b.y0) or a.x1 <= b.x0, (size, a, b)  # neighbours apart
            assert all(box.x0 <= e.x0 and e.x1 <= box.x1 for e in extents)  # inside the chart
            # Roomy cells: the largest size; narrow ones: smaller, the same in every chart.
            assert {t.get_fontsize() for t in cells} == {plots.CELL_SIZES[0] if size == 3 else plots.CELL_SIZES[1]}
        assert charts == 3  # every heatmap still shows its values
        if size == 13:
            assert any(t.get_visible() for t in figure_texts(fig) if t.get_text() == "0.350")


def test_heatmap_keeps_the_order_of_the_grid():
    df = pd.DataFrame({"road_condition": ["snow", "dry", "ice", "wet"] * 2, "n_lanes": [2] * 4 + [1] * 4,
                       "flux": 0.5, "jam_fraction": 0.1, "stops_per_car_min": 0.2})
    heat = plots.heatmap_figure(df, "road_condition", "n_lanes", plots.headline_panels(), x_label="x",
                                y_label="y", title="t", subtitle="s", value_text=str)
    ax = heat.axes[1]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["snow", "dry", "ice", "wet"]  # as sweep_figure has it
    assert [t.get_text() for t in ax.get_yticklabels()] == ["1", "2"]  # numbers ascending


def test_entry_delay_charts_show_the_level_of_service_limits():
    df = pd.DataFrame({"entry_rate": [0.1, 0.2, 0.3], "seed": 0, "flux": 0.3, "mean_entry_delay": [5.0, 20.0, 60.0]})
    fig = plots.sweep_figure(df, "entry_rate", None, x_label="x", series_label=str, title="t", subtitle="s",
                             open_demand=True)
    ax = next(a for a in fig.axes if a.get_title(loc="left") == "Mean entry delay (s, queue wait)")
    levels = sorted(line.get_ydata()[0] for line in ax.lines if line.get_linestyle() == ":")
    assert levels == [v for _, v in plots.LOS_LIMITS]
    letters = {t.get_text() for t in ax.texts}
    assert {"A", "F"} <= letters  # the bands tall enough to hold their letter
    assert ax.get_ylim()[1] == pytest.approx(1.08 * 60)  # the lines do not raise the top


def test_open_mode_dashboard_shows_the_entrances(tmp_path, capsys, monkeypatch):
    figures = []
    real = plots.dashboard

    def dashboard(*args, **kwargs):  # the real figure, not rasterised to a PNG (the test reads its axes)
        figures.append(real(*args, **kwargs))
        figures[-1].savefig = lambda path, **_: Path(path).write_bytes(PNG_MAGIC)
        return figures[-1]

    monkeypatch.setattr(plots, "dashboard", dashboard)
    out = tmp_path / "run"
    experiments.main(["run", "--traffic-mode", "open", "--truck-share", "0.3", "--entry-rate", "0.3",
                      "--duration", "40", "--out", str(out)])
    capsys.readouterr()
    fig = figures[0]
    by_title = {ax.get_title(loc="left"): ax for ax in fig.axes}
    queues = by_title["Cars on the ring and at the entrances (count)"]
    assert [t.get_text() for t in queues.get_legend().get_texts()][:3] == ["On the ring", "Queued (all entrances)",
                                                                            "Longest queue"]
    delay = by_title["Entry delay (s, queue wait)"]
    summary = pd.read_csv(f"{out}_summary.csv").iloc[0]
    assert f"LOS {summary['entry_los']}" in delay.get_title(loc="right")
    # No --baseline advice in open mode (run refuses it there), and no density control slotting cars in.
    origins = by_title["Jams by what their front car is doing (count)"]
    texts = " ".join(t.get_text() for t in origins.texts)
    assert "--baseline" not in texts and "density control" not in " ".join(
        t.get_text() for t in origins.get_legend().get_texts())
    # The histogram's "mean desired speed" is the measured one (trucks drive slower), as the summary gives it.
    hist = by_title["Speed distribution (% of car samples)"]
    assert hist.lines[0].get_xdata()[0] == pytest.approx(summary["mean_free_speed"])
    assert summary["mean_free_speed"] < experiments.free_speed(experiments.RingSim(
        SimConfig().updated(traffic_mode="open", truck_share=0.3, entry_rate=0.3))) - 0.3


def test_dashboard_draws_max_queue_only_near_the_longest_queue():
    """The Max queue line belongs to each entrance's queue: not drawn where only the cars on the ring
    (or all the queues together) pass it; the "no car entered" note sits clear of the LOS lines."""
    from matplotlib.figure import Figure

    t = np.arange(0.0, 60.0, 0.5)
    frame = pd.DataFrame({"time": t, "cars": np.linspace(80, 0, t.size), "queued": 30.0, "longest_queue": 4.0,
                          "mean_entry_delay": np.nan})

    def max_queue_lines(df):
        ax = Figure().subplots()
        plots._queues_panel(ax, df, 30.0, 50)
        return [line for line in ax.lines if np.allclose(line.get_ydata(), 50)]

    assert max_queue_lines(frame) == []  # 80 cars on the ring and 30 queued in all, but no queue near 50
    assert len(max_queue_lines(frame.assign(longest_queue=np.linspace(0, 45, t.size)))) == 1
    ax = Figure().subplots()
    plots._entry_delay_panel(ax, frame, 30.0, 30.0, float("nan"), "-")
    note = next(text for text in ax.texts if text.get_text() == "no car entered")
    limits = [v for _, v in plots.LOS_LIMITS]
    assert note.get_position()[1] == pytest.approx(0.5 * (limits[0] + limits[1]))  # between the 10 and 15 s lines


# ================================================================== experiments view
def test_view_opens_the_sweep_viewer(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(sv, "main", lambda argv=None: calls.append(argv))
    experiments.main(["view"])
    experiments.main(["view", "results/s.csv", "--results", "elsewhere", "--size", "1100X700"])
    assert calls == [[], ["results/s.csv", "--results", "elsewhere", "--size", "1100x700"]]
    with pytest.raises(SystemExit) as exc:
        experiments.main(["view", "--size", "huge"])
    assert exc.value.code == 2 and "WIDTHxHEIGHT" in capsys.readouterr().err


def test_definitions_cover_the_sweep_columns():
    text = experiments.DEFINITIONS
    for name in ("measured_density", "mean_free_speed", "measured_speed_std", "speed_deficit", "occupancy",
                 "exits_per_min", "missed_per_min", "mean_trip_time", "wave_speed_kmh", "lane_changes_per_min",
                 "measured_mean_speed"):
        assert f"\n  {name} " in text, name
        assert name in sv.definitions(), name


# ------------------------------------------------------------------ the second review's fixes
def test_infinite_x_values_are_drawn_like_a_choice_in_every_figure():
    """Density = inf is a valid setting: the line, heatmap and plan/sample figures place its values evenly
    with ∞ last (as the sweep viewer does), instead of dropping the inf runs from a numeric axis."""
    assert plots._ordered(pd.Series([60.0, np.inf, 30.0, 60.0])) == ([30.0, 60.0, np.inf], True)
    assert plots._ordered(pd.Series([60.0, 30.0])) == ([30.0, 60.0], False)
    df = pd.DataFrame({"density": [np.inf, 60.0, np.inf, 60.0], "n_lanes": [1, 1, 2, 2], "seed": 0,
                       "flux": [0.0, 0.4, 0.0, 0.8], "jam_fraction": [1.0, 0.0, 1.0, 0.0],
                       "stops_per_car_min": [0.0, 0.1, 0.0, 0.2]})
    fig = plots.sweep_figure(df, "density", "n_lanes", x_label="Density", series_label=str, title="t",
                             subtitle="s", value_text=lambda v: f"{v:g}")
    hero = fig.axes[1]  # (axes[0] is the header)
    for line in hero.lines[:2]:  # one per lanes value, both points each: 60, then inf at position 1
        np.testing.assert_array_equal(line.get_xdata(), [0, 1])
    assert [t.get_text() for t in hero.get_xticklabels()] == ["60", "∞"]
    sample = plots.sample_figure(df, ["density"], plots.headline_panels(), param_label=str, title="t",
                                 subtitle="s", value_text=lambda v: f"{v:g}")
    ax = sample.axes[1]
    assert len(ax.collections[0].get_offsets()) == 4  # every run, the inf ones too
    assert [t.get_text() for t in ax.get_xticklabels()] == ["60", "∞"]


def test_plan_in_which_nothing_varies_says_so_and_repeated_runs_are_noted(tmp_path, capsys, no_figures):
    plan = write_plan(tmp_path / "plan.csv", "density\n30\n30\n")
    _, meta, printed = sweep(tmp_path, capsys, "--plan", str(plan), "--seeds", "2", "--warmup", "0",
                             "--measure", "0.5", "--workers", "1", name="out")
    assert meta["varied"] == []
    assert ("rows with the same settings and seed repeat the same run: lines 2 and 3 with seeds 0 and 1. Give "
            "each row its own seed") in printed
    fig = no_figures[0]
    assert fig.name == "sample_figure" and fig.args[1] == ["run"] and fig.kwargs["means"] is False
    assert list(fig.args[0]["run"]) == [1, 2, 3, 4]
    assert "nothing varied" in fig.kwargs["title"] and "every row has the same settings" in fig.kwargs["subtitle"]
    seeded = write_plan(tmp_path / "seeded.csv", "density,seed\n30,1\n30,2\n")  # one seed each: no repeat
    _, _, printed = sweep(tmp_path, capsys, "--plan", str(seeded), "--seeds", "1", "--warmup", "0",
                          "--measure", "0.5", "--workers", "1", name="seeded_out")
    assert "repeat the same run" not in printed


def test_sample_groups_are_named_as_the_tabs_name_them(tmp_path, capsys, no_figures):
    _, meta, _ = sweep(tmp_path, capsys, "--sample", "3", "--sample-groups", "drivers , ROAD", "--seeds", "1",
                       "--warmup", "0", "--measure", "0.5", "--workers", "1")
    assert meta["sample_groups"] == ["Road", "Drivers"]
    assert "plausible ranges of the Road and Drivers settings)" in no_figures[0].kwargs["subtitle"]
    assert presets.group_names("lane CHG") == ["Lane chg"] and presets.group_names(None) == list(presets.GROUPS)
