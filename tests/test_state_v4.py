"""Saved states of version 4 (the ACC/CACC controllers, cooperation, the safety layer and the surrogate safety
counts; the wave smoothers' state: tests/test_smoothers.py): states of versions 1-3 load with every registered
addition filled (the new settings at their defaults, zero counters and pair counts, no string position, no episode
under way, no smoother), carry on bit for bit and save as version
4; an ACC car or CAV of a version-3 state (which drove a driver's IDM) takes its controller's time gap and dead
time; a version-4 state lacking any of it is damaged; runs with CACC strings, brands and episodes under way resume
exactly; chart rows saved before get the in_cacc key; a version-3 file resumes in the viewer with its charts."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from roundabout_sim import metrics, state
from roundabout_sim.config import SimConfig
from roundabout_sim.engine import (ACC, ADDED, CAR_ARRAYS, CAV, COUNTERS, PAIR_ARRAYS, PAIR_TYPES, STATE_VERSION,
                                   RingSim, StateError)
from roundabout_sim.metrics import Recorder
from tests.helpers import frames
from tests.test_state import assert_exact_resume
from tests.test_state_migration import as_version_1, trajectory
from tests.test_state_v3 import as_version_2

V4_SETTINGS = {"acc_headway", "acc_delay", "cav_headway", "platoon_max", "coop_class", "coop_between", "brand_spread",
               "smoothers", "smoother_law", "smoother_speed", "smoother_start", "smoother_lane"}  # (+ the smoothers')
V4_NAMES = V4_SETTINGS | {"string_pos", "aeb_until", "ttc_until", "n_aeb", "n_emergency_auto", "n_auto_steps",
                          "n_human_steps", "n_cutins_auto", "pair_follow_steps", "pair_ttc_steps", "pair_ttc_episodes",
                          "v_cmd", "smoother_on", "speed_count", "speed_hist"}
OPEN = dict(traffic_mode="open", entry_rate=0.25, n_lanes=2, truck_share=0.1, follow_up=2.5, min_gap=1.5)


def as_version_3(sim_state: dict) -> dict:
    """``sim_state`` (RingSim.state_dict) as S1 saved it: version 3, without what version 4 added."""
    meta, arrays = copy.deepcopy(sim_state["meta"]), dict(sim_state["arrays"])
    meta["version"] = 3
    for name in V4_NAMES:
        meta["config"].pop(name, None)
        meta["counters"].pop(name, None)
        arrays.pop(name, None)
    return {"meta": json.loads(json.dumps(meta)), "arrays": arrays}


def assert_v4_fills(old: RingSim) -> None:
    assert all(getattr(old.cfg, name) == ADDED[name] for name in V4_SETTINGS)
    assert (old.string_pos == 0).all() and np.isneginf(old.aeb_until).all() and np.isneginf(old.ttc_until).all()
    assert (old.v_cmd == 0).all() and not old.smoother_on.any() and (old.speed_count == 0).all()
    assert old.speed_hist.shape == (0, old.cap) and not old.smoothing
    assert all(getattr(old, name) == 0 for name in V4_NAMES & set(COUNTERS))
    for name in PAIR_ARRAYS:
        np.testing.assert_array_equal(getattr(old, name), np.zeros(len(PAIR_TYPES), dtype=np.int64))


# ------------------------------------------------------------------ the registry
def test_version_4_registers_the_controllers_and_the_safety_counts():
    assert STATE_VERSION == 4 and state.FORMAT_VERSION == 1
    assert V4_NAMES <= set(ADDED)
    assert {"string_pos", "aeb_until", "ttc_until"} <= set(CAR_ARRAYS) and set(PAIR_ARRAYS) <= set(ADDED)
    defaults = SimConfig()
    assert all(ADDED[name] == getattr(defaults, name) for name in V4_SETTINGS)
    assert ADDED["coop_class"] == "B" and ADDED["coop_between"] == "same" and ADDED["platoon_max"] == 8


# ------------------------------------------------------------------ older states
@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize("config", [dict(seed=4, **OPEN), dict(seed=6, auto_share=0.5, auto_kind="robotaxi")],
                         ids=["human", "robotaxi"])
def test_an_older_state_loads_with_the_fills_and_carries_on_exactly(version, config):
    if version < 3 and config.get("auto_share"):
        pytest.skip("versions 1 and 2 had no automated vehicles")
    sim = RingSim(SimConfig(**config))
    sim.step(500)
    st = sim.state_dict()
    old = RingSim.from_state({1: as_version_1, 2: as_version_2, 3: as_version_3}[version](st))
    assert_v4_fills(old)
    assert old.cfg == sim.cfg and trajectory(old) == trajectory(sim)
    for s in (sim, old):
        s.step(400)
    assert trajectory(old) == trajectory(sim)  # (the new counts are observers: they start from 0 there)
    assert 0 < old.pair_follow_steps.sum() < sim.pair_follow_steps.sum()
    again = old.state_dict()
    assert again["meta"]["version"] == STATE_VERSION
    assert trajectory(RingSim.from_state(again)) == trajectory(old)


@pytest.mark.parametrize("kind", ["acc", "cav"])
def test_acc_cars_and_cavs_of_a_version_3_state_take_their_controllers_values(kind):
    """In version 3 they drove a driver's IDM (a driver's mean time gap, Reaction time): loaded, they get the
    controllers' time gap, margin and dead time, as _set_driving gives them."""
    sim = RingSim(SimConfig(seed=7, auto_share=0.6, auto_kind=kind, reaction_time=0.8, density=40))
    sim.step(100)
    st = as_version_3(sim.state_dict())
    code = ACC if kind == "acc" else CAV
    slots = np.flatnonzero(sim.alive & (sim.kind == code))
    st["arrays"]["headway"][slots] = sim.eff.time_headway  # as S1 left them
    st["arrays"]["delay"][slots] = 8
    old = RingSim.from_state(st)
    np.testing.assert_array_equal(old.delay[slots], 5 if kind == "acc" else 0)
    expect = np.where(old.truck[slots], 1.5, 1.0) * (1.1 if kind == "acc" else old.acc_gap[slots])
    np.testing.assert_allclose(old.headway[slots], expect)
    old.step(50)  # and drives on
    assert np.isfinite(old.v[old.alive]).all()


@pytest.mark.parametrize("name", sorted(V4_NAMES))
def test_a_version_4_state_lacking_a_name_it_added_is_damaged(name):
    sim = RingSim(SimConfig(seed=8, auto_share=0.3, **OPEN))
    sim.step(100)
    st = sim.state_dict()
    meta, arrays = json.loads(json.dumps(st["meta"])), dict(st["arrays"])
    meta["config"].pop(name, None)
    meta["counters"].pop(name, None)
    arrays.pop(name, None)
    with pytest.raises(StateError):
        RingSim.from_state({"meta": meta, "arrays": arrays})


def test_a_pair_array_of_the_wrong_shape_is_damaged():
    sim = RingSim(SimConfig(seed=8))
    sim.step(20)
    st = sim.state_dict()
    st["arrays"]["pair_ttc_steps"] = np.zeros(3, dtype=np.int64)
    with pytest.raises(StateError, match="pair_ttc_steps has shape"):
        RingSim.from_state(st)


# ------------------------------------------------------------------ runs with the controllers
@pytest.mark.parametrize("config", [
    dict(auto_share=0.7, auto_kind="cav", companies=3, brand_spread=0.4, coop_between="none", platoon_max=3),
    dict(auto_share=1.0, auto_kind="cav", n_arms=0, n_lanes=1, density=70),  # (a loop with its cuts)
    dict(auto_share=0.5, auto_kind="acc", companies=2, brand_spread=0.3, acc_delay=0.7, density=70, **OPEN),
], ids=["cav-brands", "cav-loop", "acc-open"])
def test_a_run_with_controllers_resumes_exactly(config, tmp_path):
    sim = RingSim(SimConfig(seed=10, **config))
    sim.step(600)
    sim.brake_random()
    sim.step(5)
    assert sim.pair_follow_steps.sum() > 0
    if config["auto_kind"] == "cav":
        assert (sim.string_pos[sim.alive] >= 2).any()
    assert_exact_resume(sim, tmp_path)


def test_chart_rows_saved_before_get_the_in_cacc_key():
    sim = RingSim(SimConfig(seed=12, auto_share=0.5, auto_kind="cav"))
    rec = Recorder(sim)
    for _ in range(30):
        sim.step()
        rec.update()
    st = rec.state_dict()
    keep = [j for j, k in enumerate(st["meta"]["keys"]) if k != "in_cacc"]
    st["meta"]["keys"] = [st["meta"]["keys"][j] for j in keep]
    st["arrays"] = {**st["arrays"], "values": st["arrays"]["values"][:, keep], "kinds": st["arrays"]["kinds"][:, keep]}
    restored = Recorder(sim)
    restored.load_state(st, numbers=metrics.snapshot(sim).keys())
    assert all(row["in_cacc"] == 0.0 for row in restored.rows) and len(restored.rows) == len(rec.rows)


def test_a_version_3_file_resumes_in_the_viewer_with_its_charts(new_app, tmp_path):
    a = new_app(SimConfig(seed=14, auto_share=0.4, auto_kind="robotaxi", **OPEN), (1100, 700))
    frames(a, 40)
    saved = state.load(a.save_state())
    v3 = as_version_3(saved.part("sim"))
    rows = copy.deepcopy(saved.part("rec"))  # chart rows as S1 saved them: without in_cacc
    keep = [j for j, k in enumerate(rows["meta"]["keys"]) if k != "in_cacc"]
    rows["meta"]["keys"] = [rows["meta"]["keys"][j] for j in keep]
    rows["arrays"] = {**rows["arrays"], "values": rows["arrays"]["values"][:, keep],
                      "kinds": rows["arrays"]["kinds"][:, keep]}
    meta = {**saved.meta, "sim": v3["meta"], "rec": rows["meta"],
            "settings": {k: v for k, v in saved.meta["settings"].items() if k not in V4_NAMES}}
    old = tmp_path / "v3_state.npz"
    np.savez(old, meta_json=np.array(json.dumps(meta)), **{f"sim.{k}": v for k, v in v3["arrays"].items()},
             **{f"rec.{k}": v for k, v in rows["arrays"].items()})
    b = new_app(SimConfig(), (1100, 700), resume=old)
    assert b.message[0].startswith(f"Resumed {old}") and "could not be restored" not in b.message[0]
    assert len(b.rec.rows) == len(a.rec.rows) and trajectory(b.sim) == trajectory(a.sim)
    assert b.rec.rows[0]["in_cacc"] == 0.0
    frames(b, 10)
    assert b.rec.rows[-1]["in_cacc"] == 0
