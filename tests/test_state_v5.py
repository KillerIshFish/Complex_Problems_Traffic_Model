"""Saved states of version 5 (the entries by kind: their settings, each entrance's last entrant and the capacity
test's exiting count; the safety layer's counts renamed n_override / override_until, engine.RENAMED): states of
versions 1-4 load with every registered addition filled (no automated entrant known, a count of 0) and a version-4
state's n_aeb / aeb_until under their new names, carry on bit for bit and save as version 5; a version-5 state lacking
any of it, or holding an old name in place of a new one, is damaged; runs using the new mechanisms resume exactly; a
version-4 file resumes in the viewer with its charts."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from roundabout_sim import presets, state
from roundabout_sim.config import SimConfig
from roundabout_sim.engine import (ACC, ACC_S0, ADDED, ARM_ARRAYS, COUNTERS, HUMAN, RENAMED, STATE_VERSION, RingSim,
                                   StateError)
from tests.helpers import frames
from tests.test_state import assert_exact_resume
from tests.test_state_migration import as_version_1, trajectory
from tests.test_state_v3 import as_version_2
from tests.test_state_v4 import OPEN, as_version_3

V5_SETTINGS = {"cav_entry_gap", "cav_follow_up", "robotaxi_entry", "truck_follow_up", "subject_exiting",
               "exploit_gap", "follow_adapt", "av_weather_scale", "av_handover"}
# (+ the drivers' adaptation and the AV weather policy, and the entries' conflicts and delays: tests/test_conflicts.py)
V5_NAMES = V5_SETTINGS | {"last_kind", "last_company", "last_uid", "n_feeding", "last_entry", "pet_slot", "pet_uid",
                          "pet_kind", "pet_clear", "waited", "merge_lags", "merge_lag_finite", "merge_lag_total",
                          "merge_closed", "forced_law", "forced_override", "pet_count", "pet_total", "pet_low",
                          "class_entries", "class_wait", "class_exits", "class_travel", "class_control", "merge_log",
                          "merge_pending", "delay_used"}


def as_version_4(sim_state: dict) -> dict:
    """``sim_state`` (RingSim.state_dict) as S2 saved it: version 4, without what version 5 added, the safety
    layer's counts under their old names."""
    meta, arrays = copy.deepcopy(sim_state["meta"]), dict(sim_state["arrays"])
    meta["version"] = 4
    for name in V5_NAMES:
        meta["config"].pop(name, None)
        meta["counters"].pop(name, None)
        meta["logs"].pop(name, None)
        arrays.pop(name, None)
    for new, (old, _) in RENAMED.items():
        if new in meta["counters"]:
            meta["counters"][old] = meta["counters"].pop(new)
        if new in arrays:
            arrays[old] = arrays.pop(new)
    return {"meta": json.loads(json.dumps(meta)), "arrays": arrays}


def assert_v5_fills(old: RingSim) -> None:
    assert all(getattr(old.cfg, name) == ADDED[name] for name in V5_SETTINGS)
    assert (old.last_kind == HUMAN).all() and (old.last_company == -1).all() and (old.last_uid == -1).all()
    assert old.n_feeding == 0


# ------------------------------------------------------------------ the registry
def test_version_5_registers_the_entries_and_renames_the_safety_counts():
    assert STATE_VERSION == 5 and state.FORMAT_VERSION == 1
    assert V5_NAMES <= set(ADDED) and {"last_kind", "last_company", "last_uid"} <= set(ARM_ARRAYS)
    assert "n_feeding" in COUNTERS
    defaults = SimConfig()
    assert all(ADDED[name] == getattr(defaults, name) for name in V5_SETTINGS)
    assert RENAMED == {"n_override": ("n_aeb", 5), "override_until": ("aeb_until", 5)}
    assert ADDED["n_override"] == 0 and np.isneginf(ADDED["override_until"])


# ------------------------------------------------------------------ older states
@pytest.mark.parametrize("version", [1, 2, 3, 4])
@pytest.mark.parametrize("config", [dict(seed=4, **OPEN), dict(seed=6, auto_share=0.5, auto_kind="cav", **OPEN)],
                         ids=["human", "cav"])
def test_an_older_state_loads_with_the_fills_and_carries_on_exactly(version, config):
    if version < 3 and config.get("auto_share"):
        pytest.skip("versions 1 and 2 had no automated vehicles")
    sim = RingSim(SimConfig(**config))
    sim.step(500)
    st = sim.state_dict()
    old = RingSim.from_state({1: as_version_1, 2: as_version_2, 3: as_version_3, 4: as_version_4}[version](st))
    assert_v5_fills(old)
    assert old.cfg == sim.cfg and trajectory(old) == trajectory(sim)
    # (an older state knows no last entrants, which a CAV's follow-up reads: the run it carries on is the one whose
    # entrances forgot theirs, each learning its own again at its next entry)
    sim.last_kind[:], sim.last_company[:], sim.last_uid[:] = HUMAN, -1, -1
    for s in (sim, old):
        s.step(400)
    assert trajectory(old) == trajectory(sim)
    again = old.state_dict()
    assert again["meta"]["version"] == STATE_VERSION
    assert trajectory(RingSim.from_state(again)) == trajectory(old)


def test_a_version_4_state_keeps_its_safety_counts_under_the_new_names():
    sim = RingSim(SimConfig(seed=2, auto_share=1.0, auto_kind="acc", n_lanes=1, n_arms=0, density=70))
    sim.step(3000)
    assert sim.n_override > 0 and np.isfinite(sim.override_until).any()
    v4 = as_version_4(sim.state_dict())
    assert v4["meta"]["counters"]["n_aeb"] == sim.n_override and "n_override" not in v4["meta"]["counters"]
    old = RingSim.from_state(v4)
    assert old.n_override == sim.n_override
    np.testing.assert_array_equal(old.override_until, sim.override_until)
    assert "n_aeb" in v4["meta"]["counters"] and "aeb_until" in v4["arrays"]  # (the caller's state is left alone)


def test_a_version_4_state_gives_its_acc_cars_the_flat_margin_and_no_time_gap_floor():
    """Version 4 set each ACC car's margin every step to Xiao's at its speed (75/v - 5 m above 10.8 m/s) and floored
    its ACC time gap at 0.7 s; nothing refreshes either now, so an older state's ACC cars get ACC_S0 and the time gap
    _set_driving gives (ACC time gap x brand: 0.44 s at brand factor 0.4) at once, while a version-5 state keeps what
    it holds."""
    sim = RingSim(SimConfig(seed=3, auto_share=1.0, auto_kind="acc", companies=2, brand_spread=0.6, n_lanes=1,
                            n_arms=0, density=40, mean_speed=14.0))
    sim.step(200)
    acc = np.flatnonzero(sim.alive & (sim.kind == ACC))
    assert acc.size > 10 and (sim.s0[acc] == ACC_S0).all() and (sim.headway[acc] < 0.7).any()
    st = sim.state_dict()
    arrays = dict(st["arrays"])  # (the values version 4 held: a margin at 14 m/s, the floor)
    arrays["s0"], arrays["headway"] = arrays["s0"].copy(), arrays["headway"].copy()
    arrays["s0"][acc] = 75.0 / 14.0 - 5.0
    arrays["headway"][acc] = np.maximum(arrays["headway"][acc], 0.7)
    saved = {"meta": st["meta"], "arrays": arrays}
    old = RingSim.from_state(as_version_4(saved))
    assert (old.s0[acc] == ACC_S0).all()
    np.testing.assert_array_equal(old.headway[acc], sim.headway[acc])
    assert trajectory(old) == trajectory(sim)
    kept = RingSim.from_state(saved)
    assert (kept.s0[acc] == 75.0 / 14.0 - 5.0).all() and (kept.headway[acc] >= 0.7).all()


@pytest.mark.parametrize("name", sorted(V5_NAMES | set(RENAMED)))
def test_a_version_5_state_lacking_a_name_it_holds_is_damaged(name):
    sim = RingSim(SimConfig(seed=8, auto_share=0.3, **OPEN))
    sim.step(100)
    st = sim.state_dict()
    meta, arrays = json.loads(json.dumps(st["meta"])), dict(st["arrays"])
    meta["config"].pop(name, None)
    meta["counters"].pop(name, None)
    arrays.pop(name, None)
    with pytest.raises(StateError):
        RingSim.from_state({"meta": meta, "arrays": arrays})


def test_a_version_5_state_under_the_old_names_is_damaged():
    sim = RingSim(SimConfig(seed=8, auto_share=0.3, **OPEN))
    sim.step(100)
    st = sim.state_dict()
    meta, arrays = json.loads(json.dumps(st["meta"])), dict(st["arrays"])
    meta["counters"]["n_aeb"] = meta["counters"].pop("n_override")
    arrays["aeb_until"] = arrays.pop("override_until")
    with pytest.raises(StateError):
        RingSim.from_state({"meta": meta, "arrays": arrays})


# ------------------------------------------------------------------ runs with the new mechanisms
@pytest.mark.parametrize("config", [
    dict(auto_share=1.0, auto_kind="cav", subject_arm="on", subject_exiting=0.3, entry_rate=0.05),
    dict(auto_share=0.6, auto_kind="robotaxi", truck_share=0.2, truck_follow_up=2.2, entry_rate=0.12),
    dict(auto_share=0.5, auto_kind="cav", companies=2, coop_between="A", cav_follow_up=0.6, entry_rate=0.15),
], ids=["cav-test-exiting", "robotaxi-trucks", "cav-companies"])
def test_a_run_with_the_entry_rules_resumes_exactly(config, tmp_path):
    sim = RingSim(presets.apply("us_single_lane", SimConfig(seed=11)).updated(**config))
    sim.step(1500)
    assert (sim.last_uid >= 0).any()
    if config.get("subject_exiting"):
        assert sim.n_feeding > 0
    assert_exact_resume(sim, tmp_path, more=300)


def test_a_version_4_file_resumes_in_the_viewer_with_its_charts(new_app, tmp_path):
    a = new_app(SimConfig(seed=14, auto_share=0.4, auto_kind="cav", **OPEN), (1100, 700))
    frames(a, 40)
    saved = state.load(a.save_state())
    v4 = as_version_4(saved.part("sim"))
    rows = saved.part("rec")
    meta = {**saved.meta, "sim": v4["meta"], "rec": rows["meta"],
            "settings": {k: v for k, v in saved.meta["settings"].items() if k not in V5_NAMES}}
    old = tmp_path / "v4_state.npz"
    np.savez(old, meta_json=np.array(json.dumps(meta)), **{f"sim.{k}": v for k, v in v4["arrays"].items()},
             **{f"rec.{k}": v for k, v in rows["arrays"].items()})
    b = new_app(SimConfig(), (1100, 700), resume=old)
    assert b.message[0].startswith(f"Resumed {old}") and "could not be restored" not in b.message[0]
    assert len(b.rec.rows) == len(a.rec.rows) and trajectory(b.sim) == trajectory(a.sim)
    frames(b, 10)
    assert b.sim.time > a.sim.time
