"""Saved states across versions (SPEC 0.2): the registry engine.ADDED fills what a state of an older version
lacks (and nothing else), a state of the current version that lacks a registered name is still damaged,
versions outside 1-STATE_VERSION are refused; metrics.ADDED_KEYS fills chart keys no saved sample has; the
viewer resumes an old state with its charts; and the two states saved on main (results/, when present)."""

from __future__ import annotations

import copy
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from roundabout_sim import engine, metrics, state
from roundabout_sim.config import SimConfig
from roundabout_sim.engine import (ADDED, ARM_ARRAYS, CAR_ARRAYS, COUNTERS, EVENT_LOGS, FLOAT_TOTALS, GENERATORS,
                                   STATE_VERSION, RingSim, StateError)
from roundabout_sim.metrics import FLOAT, Recorder
from tests.helpers import frames

ROOT = Path(__file__).resolve().parents[1]
MAIN_STATES = [ROOT / "results" / "state_20261025_214756.npz", ROOT / "results" / "session" / "state.npz"]
OPEN = dict(traffic_mode="open", entry_rate=0.2, n_lanes=2, truck_share=0.1)
CONFIG_FIELDS = {f.name for f in fields(SimConfig)}


def as_version_1(sim_state: dict) -> dict:
    """``sim_state`` (RingSim.state_dict) as main saved it: version 1, without anything ADDED since."""
    meta, arrays = copy.deepcopy(sim_state["meta"]), dict(sim_state["arrays"])
    meta["version"] = 1
    for name in ADDED:
        meta["config"].pop(name, None)
        meta["counters"].pop(name, None)
        meta["logs"].pop(name, None)
        meta["generators"].pop(name, None)
        meta.pop(name, None)
        arrays.pop(name, None)
    return {"meta": json.loads(json.dumps(meta)), "arrays": arrays}


def old_rows(rec_state: dict) -> dict:
    """A Recorder.state_dict as main saved it: without the ADDED_KEYS columns."""
    meta, arrays = copy.deepcopy(rec_state["meta"]), dict(rec_state["arrays"])
    keep = [j for j, key in enumerate(meta["keys"]) if not key.endswith("_speed_std") or key == "speed_std"]
    meta["keys"] = [meta["keys"][j] for j in keep]
    arrays["values"], arrays["kinds"] = arrays["values"][:, keep], arrays["kinds"][:, keep]
    assert not meta["texts"]
    return {"meta": meta, "arrays": arrays}


def trajectory(sim: RingSim) -> tuple:
    live = np.flatnonzero(sim.alive)
    return (sim.time, sim.uid[live].tolist(), sim.theta[live].tolist(), sim.v[live].tolist(), sim.queue.tolist(),
            [getattr(sim, n) for n in COUNTERS if n not in ADDED])


# ------------------------------------------------------------------ the registry
def test_the_registry_names_only_saved_things_with_matching_defaults():
    defaults = SimConfig()
    saved = CONFIG_FIELDS | set(ARM_ARRAYS) | set(COUNTERS) | set(FLOAT_TOTALS) | set(EVENT_LOGS) | set(GENERATORS)
    assert set(ADDED) <= saved and not set(ADDED) & set(CAR_ARRAYS)  # (per-car arrays are never filled)
    for name, fill in ADDED.items():
        if name in CONFIG_FIELDS:
            assert fill == getattr(defaults, name) and type(fill) is type(getattr(defaults, name)), name
    assert {"weather_scale", "subject_arm", "anticipation", "arm_passed", "arm_passed_outer", "arm_entered_trucks",
            "n_topped_up", "conflict_log"} <= set(ADDED)
    assert STATE_VERSION == 2 and state.FORMAT_VERSION == 1
    assert ADDED["anticipation"] == engine.ANTICIPATION == SimConfig().anticipation


@pytest.mark.parametrize("config", [dict(seed=4, **OPEN), dict(seed=5, n_arms=0, n_lanes=1, density=80)],
                         ids=["open", "closed-ring"])
def test_a_version_1_state_loads_and_carries_on_exactly(config):
    sim = RingSim(SimConfig(**config))
    sim.step(400)
    old = RingSim.from_state(as_version_1(sim.state_dict()))
    # What was added since takes its fill; everything else is the saved state.
    assert old.cfg == sim.cfg and old.n_topped_up == 0 and not old.conflict_log
    for name in ("arm_passed", "arm_passed_outer", "arm_entered_trucks"):
        np.testing.assert_array_equal(getattr(old, name), np.zeros(sim.n_arms, dtype=np.int64))
        assert getattr(old, name).dtype == ARM_ARRAYS[name]
    assert trajectory(old) == trajectory(sim)
    # The new measurements are observers: the old state carries on bit for bit as the original does.
    for s in (sim, old):
        s.step(300)
    assert trajectory(old) == trajectory(sim)
    if sim.n_arms:
        assert old.conflict_log and old.arm_passed.sum() == len(old.conflict_log) < sim.arm_passed.sum()
    # It saves in the current layout again, which loads like any other.
    again = old.state_dict()
    assert again["meta"]["version"] == STATE_VERSION
    assert trajectory(RingSim.from_state(again)) == trajectory(old)


def test_a_version_1_file_loads_through_the_state_module(tmp_path):
    sim = RingSim(SimConfig(seed=6, **OPEN))
    sim.step(200)
    saved = state.pack(sim, Recorder(sim))
    v1 = as_version_1({"meta": saved.meta["sim"], "arrays": {k[4:]: v for k, v in saved.arrays.items()
                                                             if k.startswith("sim.")}})
    meta = {**saved.meta, "sim": v1["meta"],
            "settings": {k: v for k, v in saved.meta["settings"].items() if k not in ADDED}}
    arrays = {**{k: v for k, v in saved.arrays.items() if not k.startswith("sim.")},
              **{f"sim.{k}": v for k, v in v1["arrays"].items()}}
    path = tmp_path / "v1.npz"
    np.savez(path, meta_json=np.array(json.dumps(meta)), **arrays)
    loaded = state.load(path).simulation()
    assert trajectory(loaded) == trajectory(sim) and loaded.cfg == sim.cfg


def test_a_missing_generator_of_an_old_state_comes_from_the_seed(monkeypatch):
    """A generator added later (S1's rng_kinds, say) is created from [seed, k] for a state saved before it."""
    sim = RingSim(SimConfig(seed=7, **OPEN))
    sim.step(50)
    current = sim.state_dict()
    v1 = as_version_1(current)
    monkeypatch.setattr(engine, "GENERATORS", (*GENERATORS, "rng_later"))
    monkeypatch.setitem(engine.ADDED, "rng_later", 9)
    old = RingSim.from_state(v1)
    assert old.rng_later.bit_generator.state == np.random.default_rng([sim.seed, 9]).bit_generator.state
    with pytest.raises(StateError, match="rng_later"):  # this version's states must have it
        RingSim.from_state(current)


@pytest.mark.parametrize("name", ["weather_scale", "arm_passed", "n_topped_up", "conflict_log"])
def test_a_current_state_lacking_a_registered_name_is_damaged(name):
    sim = RingSim(SimConfig(seed=8, **OPEN))
    sim.step(50)
    st = sim.state_dict()
    meta, arrays = json.loads(json.dumps(st["meta"])), dict(st["arrays"])
    meta["config"].pop(name, None)
    meta["counters"].pop(name, None)
    meta["logs"].pop(name, None)
    arrays.pop(name, None)
    with pytest.raises(StateError):
        RingSim.from_state({"meta": meta, "arrays": arrays})


def test_an_old_state_lacking_anything_else_is_still_damaged():
    sim = RingSim(SimConfig(seed=9, **OPEN))
    sim.step(50)
    v1 = as_version_1(sim.state_dict())
    del v1["meta"]["config"]["politeness"]
    with pytest.raises(StateError, match="missing: politeness"):
        RingSim.from_state(v1)
    v1 = as_version_1(sim.state_dict())
    del v1["arrays"]["arm_entered"]
    with pytest.raises(StateError, match="arm_entered"):
        RingSim.from_state(v1)


@pytest.mark.parametrize("version", [0, STATE_VERSION + 1, 99, "2", 1.0, True, None])
def test_versions_outside_the_readable_range_are_refused(version):
    sim = RingSim(SimConfig(seed=10))
    st = sim.state_dict()
    st["meta"]["version"] = version
    with pytest.raises(StateError, match=f"simulation state version {version!r}"):
        RingSim.from_state(st)


# ------------------------------------------------------------------ chart samples
def test_chart_keys_added_since_are_filled_for_the_lanes_the_rows_have():
    sim = RingSim(SimConfig(seed=11, n_lanes=3))
    rec = Recorder(sim)
    for _ in range(40):
        sim.step()
        rec.update()
    names = metrics.snapshot(sim).keys()
    assert {"lane0_speed_std", "lane2_speed_std"} <= set(names) and "lane3_speed_std" not in names
    restored = Recorder(sim)
    restored.load_state(old_rows(rec.state_dict()), numbers=names)
    assert len(restored.rows) == len(rec.rows)
    for row, original in zip(restored.rows, rec.rows):
        assert set(row) == set(original)  # lanes 0-2 filled, no lane3 key made up
        assert all(np.isnan(row[f"lane{k}_speed_std"]) for k in range(3))
        assert all(row[k] == original[k] or np.isnan(row[k]) and np.isnan(original[k])
                   for k in original if not k.startswith("lane") or "std" not in k)
    assert metrics.added_keys(names) == {}
    assert metrics.added_keys(["time", "lane0_cars", "lane1_cars"]).keys() == {"lane0_speed_std", "lane1_speed_std"}


def test_a_key_some_rows_have_and_others_lack_is_still_damaged():
    sim = RingSim(SimConfig(seed=12, n_lanes=2))
    rec = Recorder(sim)
    for _ in range(20):
        sim.step()
        rec.update()
    st = rec.state_dict()
    j = st["meta"]["keys"].index("lane1_speed_std")
    st["arrays"]["kinds"] = st["arrays"]["kinds"].copy()
    st["arrays"]["kinds"][0, j] = metrics.MISSING  # one sample without it
    with pytest.raises(StateError, match="lane1_speed_std"):
        Recorder(sim).load_state(st, numbers=metrics.snapshot(sim).keys())
    st["arrays"]["kinds"][0, j] = FLOAT
    Recorder(sim).load_state(st, numbers=metrics.snapshot(sim).keys())


def test_an_empty_ring_snapshot_has_every_key(make_app):
    """snapshot's n == 0 branch defines every key a ring with cars gives (open mode empties the ring)."""
    empty = RingSim(SimConfig(seed=13, traffic_mode="open", entry_rate=0.0, density=1, n_lanes=2, radius=20))
    empty.alive[:] = False
    full = RingSim(SimConfig(seed=13, n_lanes=2, radius=20))
    assert set(metrics.snapshot(empty)) == set(metrics.snapshot(full))
    assert np.isnan(metrics.snapshot(empty)["lane1_speed_std"])


def test_the_viewer_resumes_an_old_state_with_its_charts(new_app, tmp_path):
    a = new_app(SimConfig(seed=14, **OPEN), (1100, 700))
    frames(a, 40)
    path = a.save_state()
    saved = state.load(path)
    sim_part, rec_part = saved.part("sim"), saved.part("rec")
    v1, rows = as_version_1(sim_part), old_rows(rec_part)
    meta = {**saved.meta, "sim": v1["meta"], "rec": rows["meta"],
            "settings": {k: v for k, v in saved.meta["settings"].items() if k not in ADDED}}
    arrays = {"meta_json": np.array(json.dumps(meta)), **{f"sim.{k}": v for k, v in v1["arrays"].items()},
              **{f"rec.{k}": v for k, v in rows["arrays"].items()}}
    old = tmp_path / "old_state.npz"
    np.savez(old, **arrays)
    b = new_app(SimConfig(), (1100, 700), resume=old)
    assert b.message[0].startswith(f"Resumed {old}") and "could not be restored" not in b.message[0]
    assert len(b.rec.rows) == len(a.rec.rows) and trajectory(b.sim) == trajectory(a.sim)
    frames(b, 10)  # draws its charts and stats from rows without the new keys, then with them
    assert np.isfinite(b.rec.rows[-1]["lane1_speed_std"]) or b.rec.rows[-1]["lane1_cars"] == 0


@pytest.mark.parametrize("path", MAIN_STATES, ids=lambda p: p.name)
def test_the_states_saved_on_main_resume_with_their_charts(path):
    if not path.exists():
        pytest.skip(f"{path} is not here (results/ is not in the repository)")
    saved = state.load(path)
    assert saved.meta["sim"]["version"] == 1
    sim = saved.simulation()
    assert all(getattr(sim.cfg, n) == ADDED[n] for n in ADDED if n in CONFIG_FIELDS)
    rec = Recorder(sim, 0.5)
    rec.load_state(saved.part("rec"), interval=0.5, numbers=metrics.snapshot(RingSim.from_state(sim.state_dict())))
    assert len(rec.rows) == saved.arrays["rec.values"].shape[0]
    sim.step(20)
