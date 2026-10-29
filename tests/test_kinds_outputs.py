"""What the kinds show (stage S1): the per-kind snapshot keys (automated count, human and automated mean speed;
defined on an empty ring too), the matching summary columns with their DEFINITIONS (named unlike any setting),
the run command's printout, and the viewer: the "kind" car colours with their key, the hover details of an
automated car, the AVs and Coop tabs fitting the panel, and App.apply taking every value of the new settings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from roundabout_sim import app as viewer
from roundabout_sim import experiments, metrics
from roundabout_sim import sweep_viewer as sv
from roundabout_sim.config import PARAM_BY_NAME, PARAMS, SimConfig
from roundabout_sim.engine import AUTO_KINDS, CAV, HUMAN, KIND_NAMES, ROBOTAXI, RingSim
from tests.helpers import frames, rendered_texts, show_tab, widget
from tests.test_app import assert_layout_fits

NEW = ("auto_share", "auto_kind", "robotaxi_headway", "companies")
PER_KIND = ("automated", "human_speed", "auto_speed")


# ------------------------------------------------------------------ measurements
def test_snapshot_counts_the_automated_vehicles_and_their_speeds():
    sim = RingSim(SimConfig(seed=1, auto_share=0.4, auto_kind="robotaxi", truck_share=0.1))
    sim.step(300)
    row = metrics.snapshot(sim)
    live = sim.alive
    robot, human = live & (sim.kind == ROBOTAXI), live & (sim.kind == HUMAN)
    assert row["automated"] == int(robot.sum()) > 0 and type(row["automated"]) is int
    assert row["human_speed"] == pytest.approx(sim.v[human].mean())
    assert row["auto_speed"] == pytest.approx(sim.v[robot].mean())
    plain = metrics.snapshot(RingSim(SimConfig(seed=1)))
    assert plain["automated"] == 0 and plain["human_speed"] == plain["mean_speed"] and np.isnan(plain["auto_speed"])
    every = metrics.snapshot(RingSim(SimConfig(seed=1, auto_share=1.0)))
    assert every["automated"] == every["cars"] and np.isnan(every["human_speed"])


def test_an_empty_ring_defines_the_per_kind_keys():
    empty = RingSim(SimConfig(seed=2, traffic_mode="open", entry_rate=0.0, density=1, auto_share=0.5))
    empty.alive[:] = False
    row = metrics.snapshot(empty)
    assert row["automated"] == 0 and np.isnan(row["human_speed"]) and np.isnan(row["auto_speed"])
    assert set(PER_KIND) <= set(metrics.ADDED_KEYS)


def test_summary_columns_are_defined_and_named_unlike_any_setting(tmp_path, capsys, no_figures):
    out = tmp_path / "s"
    experiments.main(["sweep", "--x", "auto_share=0,0.5", "--set", "auto_kind=robotaxi", "--seeds", "1",
                      "--warmup", "10", "--measure", "30", "--workers", "1", "--out", str(out)])
    df = pd.read_csv(f"{out}.csv")
    assert set(PER_KIND) <= set(df.columns) and not set(PER_KIND) & set(experiments.CONFIG_NAMES)
    off, on = df.iloc[0], df.iloc[1]
    assert off["automated"] == 0 and np.isnan(off["auto_speed"]) and off["human_speed"] > 0
    assert 0 < on["automated"] < on["cars"] and on["auto_speed"] > 0
    for name in PER_KIND:  # experiments.DEFINITIONS explains each (the sweep viewer's help)
        assert name in sv.definitions(), name
    assert "\n  automated " in experiments.DEFINITIONS
    measured = [c for c in df.columns if c not in experiments.CONFIG_NAMES and c != "seed"]
    assert not set(measured) & {p.name for p in PARAMS}  # no measured column shares a setting's name
    capsys.readouterr()


@pytest.mark.parametrize("share", [0.0, 0.5])
def test_the_run_printout_shows_the_automated_vehicles_when_there_are_some(tmp_path, capsys, no_figures, share):
    out = tmp_path / "run"
    experiments.main(["run", "--set", f"auto_share={share}", "--set", "auto_kind=cav", "--duration", "40",
                      "--out", str(out)])
    printed = capsys.readouterr().out
    assert ("Automated on the ring" in printed) == (share > 0)
    assert ("Mean speed, human / automated" in printed) == (share > 0)
    if share:
        assert "50 % automated: cav" in printed and "of the new ones are cav" in printed
    summary = pd.read_csv(f"{out}_summary.csv").iloc[0]
    assert (summary["automated"] > 0) == (share > 0)


# ------------------------------------------------------------------ the viewer
@pytest.mark.parametrize("size", [(1100, 700), (1920, 1080)])
def test_the_kind_colours_and_their_key(make_app, monkeypatch, size):
    app = make_app(size=size, auto_share=0.5, auto_kind="robotaxi", truck_share=0.2)
    frames(app, 5)
    app.colour_mode = "kind"
    texts = rendered_texts(app, monkeypatch)
    assert {"Kind:", "human", "robotaxi"} <= set(texts) and "cav" not in texts
    snap = app.sim.cars()
    colours = app._car_colours(snap)
    robot = snap["kind"] == ROBOTAXI
    assert robot.any() and (~robot).any()
    assert (colours[robot] == viewer.KIND_COLOURS[ROBOTAXI]).all()
    assert (colours[~robot] == viewer.KIND_COLOURS[HUMAN]).all()
    assert len(set(viewer.KIND_COLOURS)) == len(KIND_NAMES)  # every kind its own colour
    assert not set(viewer.KIND_COLOURS) & set(viewer.TYPE_COLOURS)
    assert_layout_fits(app)
    for mode in viewer.VIEW_PARAMS["colour_mode"].choices:
        app.colour_mode = mode
        frames(app, 1)


def test_the_kind_key_without_automated_vehicles(make_app, monkeypatch):
    app = make_app()
    frames(app, 2)
    app.colour_mode = "kind"
    texts = rendered_texts(app, monkeypatch)
    assert "Kind:" in texts and "human" in texts and not {"cav", "robotaxi", "acc"} & set(texts)
    assert app._kinds_shown() == [HUMAN]
    assert make_app(auto_share=0.2)._kinds_shown() == [HUMAN, CAV]


def test_hovering_an_automated_car_shows_its_kind_and_company(make_app, monkeypatch):
    app = make_app(auto_share=1.0, auto_kind="robotaxi", companies=3, density=20.0)
    frames(app, 3)
    app.paused = True
    snap = app.sim.cars()
    i = 0
    ring = app.ring
    app.mouse = (round(ring.cx + snap["x"][i] * ring.k), round(ring.cy - snap["y"][i] * ring.k))
    slot = int(snap["slot"][i])
    assert app._car_at(app.mouse) == slot
    texts = rendered_texts(app, monkeypatch)
    company = int(app.sim.company[slot]) + 1
    assert any(t.startswith("automated: robotaxi") and f"company {company} of 3" in t and "standstill gap" in t
               for t in texts), texts
    human = make_app(auto_share=0.0, density=20.0)
    frames(human, 2)
    human.paused = True
    snap = human.sim.cars()
    ring = human.ring
    human.mouse = (round(ring.cx + snap["x"][0] * ring.k), round(ring.cy - snap["y"][0] * ring.k))
    texts = rendered_texts(human, monkeypatch)
    assert not any(t.startswith(("automated", "human driver")) for t in texts)  # (no kinds possible: as before)


@pytest.mark.parametrize("size", [(1100, 700), (1920, 1080)])
def test_the_avs_and_coop_tabs_fit_and_apply_every_value(make_app, size):
    app = make_app(size=size)
    frames(app, 2)
    tabs = [w for w in app.panel_widgets if w.kind == "tab"]
    assert [w.name for w in tabs] == list(viewer.TABS) and {"AVs", "Coop"} <= set(viewer.GROUPS)
    for w in tabs:
        assert app.panel_rect.contains(w.rect) and app.fonts.width(w.name, "small") < w.rect.w
    for group in ("AVs", "Coop"):
        assert viewer.tab_help(group) == viewer.TAB_HELP[group]
        show_tab(app, group)
        frames(app, 1)
        for spec in PARAMS:
            if spec.group == group:
                assert widget(app, "choice" if spec.choices else "slider", spec.name).rect.w > 0
    for text in viewer.help_texts():  # the help box still fits the longest help
        assert len(app.fonts.wrap(text, app.status_rect.w)) <= app.status_lines, text[:60]
    for name in NEW:  # App.apply takes every value of the new settings, at once or with a rebuild
        spec = PARAM_BY_NAME[name]
        for value in (spec.choices or (spec.lo, spec.hi, spec.lo + spec.step)):
            app.apply(name, value)
            frames(app, 1)
            assert getattr(app.sim.cfg, name) == value
    assert app.inactive(PARAM_BY_NAME["robotaxi_headway"]) in ("", "no robotaxis", "no automated vehicles")
    app.apply("auto_share", 0.3)
    app.apply("auto_kind", "robotaxi")
    assert app.inactive(PARAM_BY_NAME["robotaxi_headway"]) == ""
    assert app.sim.cfg.auto_share == 0.3 and (app.sim.kind[app.sim.alive] == AUTO_KINDS["robotaxi"]).any()
    frames(app, 1)  # (a rebuild redraws the road on the next frame)
    assert_layout_fits(app)


def test_the_new_settings_are_saved_and_restored_with_the_session(new_app, tmp_path):
    a = new_app(SimConfig(seed=3, auto_share=0.4, auto_kind="acc", companies=2, robotaxi_headway=1.5),
                (1100, 700))
    frames(a, 20)
    path = a.save_state()
    b = new_app(SimConfig(), (1100, 700), resume=Path(path))
    assert b.message[0].startswith("Resumed") and b.sim.cfg == a.sim.cfg
    np.testing.assert_array_equal(b.sim.kind, a.sim.kind)
    np.testing.assert_array_equal(b.sim.company, a.sim.company)
