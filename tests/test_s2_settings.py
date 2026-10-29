"""The ACC/CACC and cooperation settings (AVs: acc_headway, acc_delay, cav_headway, platoon_max; Coop: coop_class,
coop_between, brand_spread): defaults, ranges and validation; why each does nothing where it does nothing (and
companies' real rule now: it acts through brand_spread and coop_between with ACC cars or CAVs); live changes; at
any value they leave runs without their kind alone (bit for bit); random draws only when their tabs are named;
the viewer takes every value, shows a CAV's string, and its help texts fit."""

from __future__ import annotations

import numpy as np
import pytest

from roundabout_sim import app as viewer
from roundabout_sim import experiments, presets
from roundabout_sim import sweep_viewer as sv
from roundabout_sim.config import (IDLE_WHY, PARAM_BY_NAME, SimConfig, companies_differ, inactive_inputs,
                                   inactive_reason)
from roundabout_sim.engine import CAV, RingSim
from tests.helpers import frames, rendered_texts

NEW = {"acc_headway": 1.1, "acc_delay": 0.5, "cav_headway": 0.7, "platoon_max": 8, "coop_class": "B",
       "coop_between": "same", "brand_spread": 0.0}
OTHER = {"acc_headway": 1.8, "acc_delay": 0.9, "cav_headway": 1.2, "platoon_max": 3, "coop_class": "none",
         "coop_between": "none", "brand_spread": 0.5}


def test_defaults_tabs_and_ranges():
    cfg = SimConfig()
    assert {name: getattr(cfg, name) for name in NEW} == NEW
    # (platoon_max on the Coop tab: strings are status-sharing cooperation, and the AVs tab fits at 1280 x 720)
    assert [PARAM_BY_NAME[n].group for n in NEW] == ["AVs"] * 3 + ["Coop"] * 4
    assert PARAM_BY_NAME["coop_class"].choices == ("none", "A", "B", "C", "D")
    assert PARAM_BY_NAME["coop_between"].choices == ("same", "none", "A", "B", "C", "D")
    for name in NEW:
        spec = PARAM_BY_NAME[name]
        assert not spec.reset, name  # every one applies at once
        assert len(spec.help) <= 340, name  # (the help boxes add the range, the reason and what a click does)
        if not spec.choices:
            assert spec.lo <= NEW[name] <= spec.hi
    assert (PARAM_BY_NAME["acc_delay"].lo, PARAM_BY_NAME["acc_delay"].step) == (0.0, 0.1)


@pytest.mark.parametrize("bad", [dict(platoon_max=0), dict(platoon_max=2.5), dict(platoon_max=200),
                                 dict(brand_spread=-0.1), dict(brand_spread=1.0), dict(coop_class="E"),
                                 dict(coop_between="all"), dict(acc_headway=0.0), dict(cav_headway=-0.5),
                                 dict(acc_delay=-0.1), dict(acc_delay=float("nan"))])
def test_invalid_values_are_refused(bad):
    with pytest.raises(ValueError):
        SimConfig(**bad)


# ------------------------------------------------------------------ why they do nothing
@pytest.mark.parametrize("name, config, reason", [
    ("acc_headway", dict(), "no ACC cars"),
    ("acc_delay", dict(auto_share=0.5, auto_kind="cav"), "no ACC cars"),
    ("acc_headway", dict(auto_share=0.5, auto_kind="acc"), ""),
    ("cav_headway", dict(auto_share=0.5, auto_kind="acc"), "no CAVs"),
    ("cav_headway", dict(auto_share=0.5), ""),
    ("cav_headway", dict(auto_share=0.5, coop_class="none"), "no cooperation"),
    ("cav_headway", dict(auto_share=0.5, platoon_max=1), "no strings"),
    ("platoon_max", dict(auto_share=0.5, platoon_max=1), ""),
    ("platoon_max", dict(auto_share=0.5, coop_class="none"), "no cooperation"),
    ("coop_class", dict(auto_share=0.5, auto_kind="robotaxi"), "no CAVs"),
    ("coop_class", dict(auto_share=0.5, coop_class="none"), ""),
    ("coop_between", dict(auto_share=0.5), "one company"),
    ("coop_between", dict(auto_share=0.5, companies=2), ""),
    ("coop_between", dict(auto_share=0.5, companies=2, coop_class="none"), "no cooperation"),
    ("brand_spread", dict(auto_share=0.5, auto_kind="robotaxi", companies=2), "no ACC cars or CAVs"),
    ("brand_spread", dict(auto_share=0.5, auto_kind="acc"), "one company"),
    ("brand_spread", dict(auto_share=0.5, auto_kind="acc", companies=3), ""),
    ("companies", dict(), "no automated vehicles"),
    ("companies", dict(auto_share=0.5, auto_kind="robotaxi", brand_spread=0.3), "companies alike"),
    ("companies", dict(auto_share=0.5, auto_kind="acc"), "companies alike"),
    ("companies", dict(auto_share=0.5, auto_kind="acc", brand_spread=0.3), ""),
    ("companies", dict(auto_share=0.5, coop_between="none"), ""),
    ("companies", dict(auto_share=0.5, coop_between="A"), "companies alike"),  # (class A or more: CACC either way)
    ("companies", dict(auto_share=0.5, coop_between="none", coop_class="none"), "companies alike"),
])
def test_each_setting_says_why_it_does_nothing(name, config, reason):
    assert inactive_reason(name, SimConfig(**config)) == reason
    if name == "companies" and config.get("auto_share"):
        assert companies_differ(SimConfig(**config)) == (reason == "")


def test_every_reason_has_its_fix_and_the_inputs_are_named():
    assert {"no ACC cars", "no CAVs", "no ACC cars or CAVs", "no cooperation", "no strings", "one company",
            "companies alike"} <= set(IDLE_WHY)
    assert set(IDLE_WHY) <= set(experiments.IDLE_FIX) and set(IDLE_WHY) <= set(sv.IDLE_FIX)
    assert inactive_inputs("acc_headway") == {"auto_share", "auto_kind"}
    assert inactive_inputs("cav_headway") == {"auto_share", "auto_kind", "coop_class", "platoon_max"}
    assert inactive_inputs("coop_between") == {"auto_share", "auto_kind", "coop_class", "companies"}
    assert inactive_inputs("brand_spread") == {"auto_share", "auto_kind", "companies"}
    assert inactive_inputs("companies") == {"auto_share", "auto_kind", "brand_spread", "coop_class", "coop_between"}
    notes = experiments.idle_notes(["cav_headway"], [SimConfig(auto_share=0.5, auto_kind="acc", cav_headway=x)
                                                     for x in (0.6, 0.9)])
    assert len(notes) == 1 and "acts only on CAVs" in notes[0] and "auto_kind=cav" in notes[0]


# ------------------------------------------------------------------ bit-identity and live changes
def run(cfg: SimConfig, steps: int = 300) -> tuple:
    sim = RingSim(cfg)
    sim.step(steps)
    live = np.flatnonzero(sim.alive)
    return sim.uid[live].tolist(), sim.theta[live].tolist(), sim.v[live].tolist(), sim.n_emergency


@pytest.mark.parametrize("name", sorted(NEW))
def test_at_any_value_a_new_setting_leaves_runs_without_its_vehicles_alone(name):
    """Human drivers only (any value), and robotaxis (which none of them drive), run bit for bit as at the
    defaults; so the defaults themselves change nothing for them."""
    for base in (SimConfig(seed=3, truck_share=0.1, headway_std=0.3),
                 SimConfig(seed=3, auto_share=0.5, auto_kind="robotaxi", companies=3)):
        assert run(base.updated(**{name: OTHER[name]})) == run(base), name


def test_the_settings_apply_live():
    sim = RingSim(SimConfig(seed=5, auto_share=1.0, auto_kind="cav", companies=2))
    sim.step(100)
    gen = sim.generation
    assert (sim.string_pos[sim.alive] >= 2).any()
    for change in (dict(cav_headway=0.9), dict(platoon_max=4), dict(coop_between="none"), dict(brand_spread=0.4),
                   dict(coop_class="none"), dict(acc_headway=1.5, acc_delay=0.3)):
        assert not sim.set(**change), change
    sim.step()
    assert sim.generation == gen and (sim.string_pos[sim.alive] <= 1).all()  # (no cooperation: no strings)
    assert not sim.set(coop_class="A")
    sim.step()
    assert (sim.string_pos[sim.alive] >= 2).any()


# ------------------------------------------------------------------ random draws
def test_random_draws_only_when_their_tabs_are_named():
    names = set(NEW)
    for seed in range(10):
        assert not set(presets.random_changes(seed)) & names
    assert not set(presets.plausible_ranges()) & names
    for seed in range(60):
        drawn = presets.random_changes(seed, ["AVs", "Coop"])
        assert names <= set(drawn)
        cfg = SimConfig(**drawn)
        assert 0.9 <= cfg.acc_headway <= 1.5 and 0.2 <= cfg.acc_delay <= 0.6 and 0.6 <= cfg.cav_headway <= 1.1
        assert cfg.platoon_max in (5, 8, 10) and 0.0 <= cfg.brand_spread <= 0.6
    for name in names:
        assert presets.RANDOM_RANGES[name].basis.strip()


# ------------------------------------------------------------------ the viewer
def test_the_viewer_takes_every_value_and_shows_a_cavs_string(make_app, monkeypatch):
    app = make_app(auto_share=0.8, auto_kind="cav", companies=2, density=60.0)
    frames(app, 30)
    for name in NEW:
        spec = PARAM_BY_NAME[name]
        values = spec.choices or np.linspace(spec.lo, spec.hi, 4).tolist()
        for value in values:
            app.apply(name, value)
        app.apply(name, NEW[name])
    frames(app, 5)
    app.paused = True
    sim, ring = app.sim, app.ring
    snap = sim.cars()
    i = int(np.flatnonzero((snap["kind"] == CAV) & (sim.string_pos[snap["slot"]] >= 2)
                           & (sim.string_pos[snap["slot"]] <= sim.cfg.platoon_max))[0])
    app.mouse = (round(ring.cx + snap["x"][i] * ring.k), round(ring.cy - snap["y"][i] * ring.k))
    slot = int(snap["slot"][i])
    assert app._car_at(app.mouse) == slot
    texts = rendered_texts(app, monkeypatch)
    assert f"CACC: string member {sim.string_pos[slot]} of at most {sim.cfg.platoon_max}" in texts, texts
    assert "ACC" in viewer.TAB_HELP["AVs"] and "companies" in viewer.TAB_HELP["Coop"]
