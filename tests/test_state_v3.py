"""Saved states of version 3 (stage S1: vehicle kinds): states of versions 1 and 2 load with every vehicle a
human driver (kind, company, fallback gap, a_now and the queue fields filled; each car's standstill gap worked
out from the state's own settings, not a constant), version 2's 4-field conflict_log widened by the kind, the
kinds' stream from [seed, 3]; they carry on bit for bit; a version-3 state lacking any of it is damaged; runs
with automated vehicles resume exactly; the chart rows saved before get the per-kind keys."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from roundabout_sim import metrics, presets, state
from roundabout_sim.config import SimConfig
from roundabout_sim.engine import (ADDED, CAR_ARRAYS, HUMAN, LOG_WIDENED, QUEUE_FIELDS, STATE_VERSION, FromState,
                                   RingSim, StateError)
from roundabout_sim.metrics import Recorder
from tests.helpers import frames
from tests.test_state import assert_exact_resume
from tests.test_state_migration import as_version_1, trajectory

# What version 2 (S0) added; everything else in ADDED came with version 3.
V2_NAMES = {"weather_scale", "subject_arm", "anticipation", "arm_passed", "arm_passed_outer", "arm_passed_trucks",
            "arm_entered_trucks", "arm_entry_wait", "n_topped_up", "conflict_log"}
V3_NAMES = {"auto_share", "auto_kind", "robotaxi_headway", "companies", "rng_kinds", "rng_kinds_test", "kind",
            "company", "acc_gap", "a_now", "s0", "queue_kinds", "queue_companies", "queue_acc_gaps"}
LATER = set(ADDED) - V2_NAMES - V3_NAMES  # (version 4 on: the ACC/CACC controllers, test_state_v4)
OPEN = dict(traffic_mode="open", entry_rate=0.25, n_lanes=2, truck_share=0.1, follow_up=2.5, min_gap=1.5)


def as_version_2(sim_state: dict) -> dict:
    """``sim_state`` (RingSim.state_dict) as S0 saved it: version 2, without what version 3 added, and its
    conflict_log entries of 4 fields (time, arm, lane, uid)."""
    meta, arrays = copy.deepcopy(sim_state["meta"]), dict(sim_state["arrays"])
    meta["version"] = 2
    for name in V3_NAMES | LATER:
        meta["config"].pop(name, None)
        meta["generators"].pop(name, None)
        meta["counters"].pop(name, None)
        arrays.pop(name, None)
        arrays.pop(f"{name}_count", None)
    log = arrays["conflict_log"]
    arrays["conflict_log"] = log[:, :4] if log.ndim == 2 else log
    meta["logs"]["conflict_log"] = meta["logs"]["conflict_log"][:4]
    return {"meta": json.loads(json.dumps(meta)), "arrays": arrays}


def assert_human_fills(old: RingSim) -> None:
    """Every vehicle of an older state is a human driver, with the standstill gap of the state's settings."""
    assert (old.kind == HUMAN).all() and (old.company == -1).all() and np.isnan(old.acc_gap).all()
    assert (old.a_now == 0).all() and (old.s0 == old.cfg.min_gap).all()
    for name in ("queue_kinds", "queue_companies", "queue_acc_gaps"):
        assert [len(fifo) for fifo in getattr(old, name)] == old.queue.tolist()
    assert all(set(fifo) <= {HUMAN} for fifo in old.queue_kinds)
    assert all(set(fifo) <= {-1} for fifo in old.queue_companies)
    assert old.rng_kinds.bit_generator.state == np.random.default_rng([old.seed, 3]).bit_generator.state
    assert old.rng_kinds_test.bit_generator.state == np.random.default_rng([old.seed, 3, 1]).bit_generator.state
    assert old.cfg.auto_share == 0.0 and old.cfg.companies == 1 and not old.automated


# ------------------------------------------------------------------ the registry
def test_version_3_registers_the_kinds():
    assert STATE_VERSION >= 3 and state.FORMAT_VERSION == 1  # (version 4 added the controllers: LATER)
    assert V3_NAMES <= set(ADDED) and not LATER & V3_NAMES
    assert ADDED["rng_kinds"] == 3 and ADDED["rng_kinds_test"] == (3, 1)
    assert set(V3_NAMES) & set(CAR_ARRAYS) == {"kind", "company", "acc_gap", "a_now", "s0"}
    assert isinstance(ADDED["s0"], FromState) and ADDED["kind"] == HUMAN and ADDED["company"] == -1
    assert set(QUEUE_FIELDS) - {"queue_trucks"} <= V3_NAMES and "queue_trucks" not in ADDED
    assert LOG_WIDENED == {"conflict_log": (5, (HUMAN,))}


# ------------------------------------------------------------------ older states
@pytest.mark.parametrize("version", [1, 2])
def test_an_older_state_loads_with_human_drivers_and_carries_on_exactly(version):
    sim = RingSim(SimConfig(seed=4, **OPEN))
    sim.step(500)
    assert sim.queue.sum() > 0 and sim.conflict_log
    old = RingSim.from_state((as_version_1 if version == 1 else as_version_2)(sim.state_dict()))
    assert_human_fills(old)
    assert old.cfg == sim.cfg and trajectory(old) == trajectory(sim)
    if version == 2:  # widened by the kind: the same log as the running simulation's, which is all human
        assert list(old.conflict_log) == list(sim.conflict_log) and all(len(e) == 5 for e in old.conflict_log)
        assert all(type(e[4]) is int and e[4] == HUMAN for e in old.conflict_log)
    else:
        assert not old.conflict_log
    for s in (sim, old):
        s.step(400)
    assert trajectory(old) == trajectory(sim)
    np.testing.assert_array_equal(old.s0[old.alive], sim.s0[sim.alive])
    again = old.state_dict()
    assert again["meta"]["version"] == STATE_VERSION
    assert trajectory(RingSim.from_state(again)) == trajectory(old)


@pytest.mark.parametrize("min_gap", [1.0, 3.25])
def test_the_standstill_gaps_of_an_older_state_follow_its_settings(min_gap):
    """s0's fill is each car's Standstill gap from the state's own settings (FromState), not a constant
    default (2 m would be wrong here)."""
    sim = RingSim(SimConfig(seed=5, min_gap=min_gap, density=40))
    sim.step(50)
    st = as_version_2(sim.state_dict())
    assert "s0" not in st["arrays"]
    old = RingSim.from_state(st)
    assert (old.s0 == min_gap).all() and old.s0.shape == (old.cap,)
    assert trajectory(old) == trajectory(sim)


def test_a_version_2_file_resumes_in_the_viewer_with_its_charts(new_app, tmp_path):
    a = new_app(SimConfig(seed=14, **OPEN), (1100, 700))
    frames(a, 40)
    saved = state.load(a.save_state())
    v2 = as_version_2(saved.part("sim"))
    rows = copy.deepcopy(saved.part("rec"))  # chart rows as S0 saved them: without the per-kind keys
    keep = [j for j, k in enumerate(rows["meta"]["keys"]) if k not in ("automated", "human_speed", "auto_speed")]
    rows["meta"]["keys"] = [rows["meta"]["keys"][j] for j in keep]
    rows["arrays"] = {**rows["arrays"], "values": rows["arrays"]["values"][:, keep],
                      "kinds": rows["arrays"]["kinds"][:, keep]}
    meta = {**saved.meta, "sim": v2["meta"], "rec": rows["meta"],
            "settings": {k: v for k, v in saved.meta["settings"].items() if k not in V3_NAMES | LATER}}
    old = tmp_path / "v2_state.npz"
    np.savez(old, meta_json=np.array(json.dumps(meta)), **{f"sim.{k}": v for k, v in v2["arrays"].items()},
             **{f"rec.{k}": v for k, v in rows["arrays"].items()})
    b = new_app(SimConfig(), (1100, 700), resume=old)
    assert b.message[0].startswith(f"Resumed {old}") and "could not be restored" not in b.message[0]
    assert len(b.rec.rows) == len(a.rec.rows) and trajectory(b.sim) == trajectory(a.sim)
    first = b.rec.rows[0]
    assert first["automated"] == 0.0 and np.isnan(first["human_speed"]) and np.isnan(first["auto_speed"])
    b.colour_mode = "kind"
    frames(b, 10)
    assert b.rec.rows[-1]["automated"] == 0 and b.rec.rows[-1]["human_speed"] == b.rec.rows[-1]["mean_speed"]


# ------------------------------------------------------------------ damaged states
@pytest.mark.parametrize("name", ["kind", "s0", "a_now", "queue_kinds", "auto_share", "companies", "rng_kinds",
                                  "rng_kinds_test"])
def test_a_version_3_state_lacking_a_name_it_added_is_damaged(name):
    sim = RingSim(SimConfig(seed=8, auto_share=0.3, **OPEN))
    sim.step(100)
    st = sim.state_dict()
    meta, arrays = json.loads(json.dumps(st["meta"])), dict(st["arrays"])
    meta["config"].pop(name, None)
    meta["generators"].pop(name, None)
    arrays.pop(name, None)
    with pytest.raises(StateError):
        RingSim.from_state({"meta": meta, "arrays": arrays})


def test_a_conflict_log_of_the_wrong_width_is_damaged():
    sim = RingSim(SimConfig(seed=9, **OPEN))
    sim.step(200)
    st = sim.state_dict()
    narrow = {"meta": json.loads(json.dumps(st["meta"])), "arrays": dict(st["arrays"])}
    narrow["arrays"]["conflict_log"] = st["arrays"]["conflict_log"][:, :4]
    narrow["meta"]["logs"]["conflict_log"] = narrow["meta"]["logs"]["conflict_log"][:4]
    with pytest.raises(StateError, match="conflict_log holds entries of 4 fields, expected 5"):
        RingSim.from_state(narrow)  # a version-3 state must hold the kind
    older = as_version_2(st)
    older["arrays"]["conflict_log"] = older["arrays"]["conflict_log"][:, :3]
    older["meta"]["logs"]["conflict_log"] = older["meta"]["logs"]["conflict_log"][:3]
    with pytest.raises(StateError, match="conflict_log holds entries of 3 fields"):
        RingSim.from_state(older)


# ------------------------------------------------------------------ automated runs
@pytest.mark.parametrize("config", [
    dict(auto_share=0.5, auto_kind="robotaxi", companies=3, max_queue=6, **OPEN),
    dict(auto_share=0.4, auto_kind="cav", exit_prob=0.4, truck_share=0.2, headway_std=0.3, reaction_std=0.2),
    dict(auto_share=1.0, auto_kind="acc", n_arms=0, n_lanes=1, density=80),
], ids=["open-robotaxi", "closed-cav", "ring-acc"])
def test_a_run_with_automated_vehicles_resumes_exactly(config, tmp_path):
    sim = RingSim(SimConfig(seed=10, **config))
    sim.step(400)
    sim.brake_random()
    assert (sim.kind[sim.alive] != HUMAN).any()
    assert_exact_resume(sim, tmp_path)


def test_the_capacity_test_resumes_exactly_with_its_queue_kinds(tmp_path):
    cfg = presets.apply("us_single_lane", SimConfig(seed=11)).updated(subject_arm="on", entry_rate=0.08,
                                                                       auto_share=0.5, auto_kind="robotaxi")
    sim = RingSim(cfg)
    sim.step(600)
    assert len(sim.queue_kinds[0]) == 10 and any(k != HUMAN for k in sim.queue_kinds[0])
    assert any(kind != HUMAN for *_, kind in sim.conflict_log)
    assert_exact_resume(sim, tmp_path)


def test_chart_rows_saved_before_the_kinds_get_the_per_kind_keys():
    sim = RingSim(SimConfig(seed=12, auto_share=0.3))
    rec = Recorder(sim)
    for _ in range(30):
        sim.step()
        rec.update()
    st = rec.state_dict()
    keep = [j for j, k in enumerate(st["meta"]["keys"]) if k not in ("automated", "human_speed", "auto_speed")]
    st["meta"]["keys"] = [st["meta"]["keys"][j] for j in keep]
    st["arrays"] = {**st["arrays"], "values": st["arrays"]["values"][:, keep], "kinds": st["arrays"]["kinds"][:, keep]}
    restored = Recorder(sim)
    restored.load_state(st, numbers=metrics.snapshot(sim).keys())
    assert all(row["automated"] == 0.0 and np.isnan(row["human_speed"]) for row in restored.rows)
    assert metrics.ADDED_KEYS["automated"] == 0.0
