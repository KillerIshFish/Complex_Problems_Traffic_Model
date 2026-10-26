"""Saved states: exact resume of the engine (in memory and through a file), the chart history, the
file format and its failure modes, and the viewer's sessions (autosave on quit, resume, --fresh,
--no-autosave, --state, F5 / F9)."""

from __future__ import annotations

import json
import os
import zipfile
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import pygame
import pytest

from roundabout_sim import app as viewer
from roundabout_sim import metrics, state
from roundabout_sim.config import SimConfig
from roundabout_sim.engine import CAR_ARRAYS, EVENT_LOGS, GENERATORS, RingSim, StateError
from roundabout_sim.metrics import Recorder
from tests.helpers import click, frames, key, widget

SIZE = (1100, 700)  # the smallest window the viewer supports


# ------------------------------------------------------------------ helpers
def assert_same(x, y, where: str = "sim") -> None:
    """Equal all the way down, with the same types (arrays: dtype, shape and values; NaN equals NaN)."""
    assert type(x) is type(y), f"{where}: {type(x).__name__} vs {type(y).__name__}"
    if isinstance(x, RingSim):
        assert vars(x).keys() == vars(y).keys(), f"{where}: {set(vars(x)) ^ set(vars(y))}"
        for name in vars(x):
            assert_same(vars(x)[name], vars(y)[name], f"{where}.{name}")
    elif isinstance(x, np.ndarray):
        assert x.dtype == y.dtype and x.shape == y.shape, f"{where}: {x.dtype}{x.shape} vs {y.dtype}{y.shape}"
        assert np.array_equal(x, y, equal_nan=x.dtype.kind in "fc"), where
    elif isinstance(x, (list, tuple, deque)):
        assert len(x) == len(y), f"{where}: {len(x)} vs {len(y)} entries"
        for i, (a, b) in enumerate(zip(x, y)):
            assert_same(a, b, f"{where}[{i}]")
    elif isinstance(x, dict):
        assert list(x) == list(y), f"{where}: keys differ"
        for k in x:
            assert_same(x[k], y[k], f"{where}[{k!r}]")
    elif isinstance(x, np.random.Generator):
        assert x.bit_generator.state == y.bit_generator.state, where
    elif isinstance(x, float) and np.isnan(x):
        assert np.isnan(y), where
    else:
        assert x == y, f"{where}: {x!r} vs {y!r}"


def through_file(sim: RingSim, path: Path) -> RingSim:
    state.save(path, sim)
    return state.load(path).simulation()


def assert_exact_resume(sim: RingSim, tmp_path: Path, more: int = 100) -> None:
    """Copies made in memory and through a file are the simulation exactly, and stay so for ``more`` steps."""
    copies = [RingSim.from_state(sim.state_dict()), through_file(sim, tmp_path / "resume.npz")]
    for copy in copies:
        assert_same(sim, copy)
    for s in (sim, *copies):
        s.step(more)
    for copy in copies:
        assert_same(sim, copy)


def write_state(path: Path, meta: dict, arrays: dict) -> None:
    """A state file with the given (possibly broken) contents."""
    np.savez(path, meta_json=np.array(json.dumps(meta)), **arrays)


def saved_contents(tmp_path: Path, **config) -> tuple[dict, dict]:
    """(meta, arrays) of a freshly saved small simulation, to break in tests."""
    sim = RingSim(SimConfig(seed=1, **config))
    sim.step(30)
    path = tmp_path / "good.npz"
    state.save(path, sim, Recorder(sim))
    saved = state.load(path)
    return saved.meta, dict(saved.arrays)


# ------------------------------------------------------------ exact resume
RESUME_CASES = {
    "defaults": (dict(), 150),
    "open-high-demand": (dict(traffic_mode="open", entry_rate=0.5, follow_up=3.0, entry_gap=3.5, max_queue=40,
                              demand_profile="peak", demand_imbalance=0.6), 600),
    "open-spill-back": (dict(traffic_mode="open", entry_rate=0.8, follow_up=3.0, max_queue=3, truck_share=0.15), 300),
    "trucks-spreads-snow": (dict(truck_share=0.25, reaction_std=0.3, headway_std=0.4, road_condition="snow"), 300),
    "big-5-lane-ring": (dict(n_lanes=5, radius=300, car_length=2, min_gap=1, density=200), 60),
    "dense-long-trucks": (dict(n_lanes=1, truck_share=0.3, truck_length=20, density=110), 200),
    "closed-ring": (dict(n_arms=0, n_lanes=1, density=80, lane_change_motive="random"), 200),
    "empty-ring-open": (dict(n_lanes=1, radius=20, density=1, traffic_mode="open", entry_rate=0.2), 200),
}


@pytest.mark.parametrize("config, steps", RESUME_CASES.values(), ids=RESUME_CASES.keys())
def test_resume_is_exact(config, steps, tmp_path):
    sim = RingSim(SimConfig(seed=11, **config))
    sim.step(steps)
    sim.brake_random()
    if config.get("max_queue", 50) <= 5:
        assert sim.n_blocked > 0 and sim.blocked_log and sim.queue.max() == 3  # full queues turn arrivals away
    elif config.get("entry_rate", 0) >= 0.5:
        assert sim.queue.max() >= 10  # long queues to carry over
    assert_exact_resume(sim, tmp_path)


def test_resume_is_exact_after_the_arrays_grew(tmp_path):
    sim = RingSim(SimConfig(seed=12))
    sim.step(50)
    cap = sim.cap
    sim._grow(1)
    sim.step(50)
    assert sim.cap >= 2 * cap
    assert_exact_resume(sim, tmp_path)


def test_resume_is_exact_in_the_middle_of_a_lane_change(tmp_path):
    sim = RingSim(SimConfig(seed=13, lane_change_rate=1.0))
    for _ in range(600):
        sim.step()
        drifting = sim.alive & (np.abs(sim.lane - sim.lat) > 1e-6)
        if drifting.any():
            break
    assert drifting.any()
    copy = through_file(sim, tmp_path / "mid.npz")
    np.testing.assert_array_equal(copy.lat, sim.lat)  # still drawn between the lanes
    assert_exact_resume(sim, tmp_path)


def test_resume_is_exact_after_live_changes(tmp_path):
    sim = RingSim(SimConfig(seed=14))
    sim.step(100)
    assert not sim.set(density=70, exit_prob=0.3, reaction_std=0.2, headway_std=0.3, road_condition="wet")
    sim.step(50)
    assert not sim.set(seed=99)  # re-seeds every stream from now on
    sim.step(30)
    assert_exact_resume(sim, tmp_path)

    opened = RingSim(SimConfig(seed=15, traffic_mode="open", entry_rate=0.5, follow_up=3.0))
    opened.step(400)
    assert not opened.set(max_queue=3, density=30, demand_imbalance=0.8)  # queued cars give up, the ring thins
    assert opened.n_gave_up > 0
    opened.step(20)
    assert_exact_resume(opened, tmp_path)


def test_every_attribute_is_saved_or_derived():
    sim = RingSim(SimConfig(seed=16, truck_share=0.1))
    sim.step(30)
    restored = RingSim.from_state(sim.state_dict())
    assert set(vars(restored)) == set(vars(sim))
    # CAR_ARRAYS names every per-car array (_alloc), so growing and saving keep them all.
    per_car = {name for name, value in vars(sim).items() if isinstance(value, np.ndarray) and value.shape == (sim.cap,)}
    assert per_car == set(CAR_ARRAYS)
    assert {name for name, value in vars(sim).items() if isinstance(value, deque)} == {*EVENT_LOGS, "entrance_log"}
    assert {name for name, value in vars(sim).items() if isinstance(value, np.random.Generator)} == set(GENERATORS)


def test_state_dict_is_json_and_copies():
    sim = RingSim(SimConfig(seed=17, traffic_mode="open", entry_rate=0.3, truck_share=0.2))
    sim.step(200)
    saved = sim.state_dict()
    assert json.loads(json.dumps(saved["meta"])) == saved["meta"]  # plain JSON, exactly
    assert all(isinstance(arr, np.ndarray) and arr.dtype != object for arr in saved["arrays"].values())
    before = sim.v.copy()
    saved["arrays"]["v"][:] = -1.0  # the state does not share memory with the simulation
    np.testing.assert_array_equal(sim.v, before)
    # Logs keep their Python types: whole numbers int, times float, trip times numpy floats.
    copy = RingSim.from_state(sim.state_dict())
    for name in EVENT_LOGS:
        for a, b in zip(getattr(sim, name), getattr(copy, name)):
            assert [type(v) for v in np.atleast_1d(a)] == [type(v) for v in np.atleast_1d(b)] if not isinstance(a, tuple) \
                else [type(v) for v in a] == [type(v) for v in b]
    assert sim.exit_log and type(copy.exit_log[0][1]) is int and type(copy.exit_log[0][2]) is np.float64
    assert all(type(t) is float for fifo in copy.queue_times for t in fifo)
    assert all(type(truck) is bool for fifo in copy.queue_trucks for truck in fifo)


def test_restoring_leaves_the_original_alone(tmp_path):
    sim = RingSim(SimConfig(seed=18))
    sim.step(40)
    twin = RingSim.from_state(sim.state_dict())
    RingSim.from_state(sim.state_dict()).step(100)  # another copy running on draws nothing from these
    sim.step(10)
    twin.step(10)
    assert_same(sim, twin)


# ------------------------------------------------------------ the file
def test_file_round_trip_without_pickle(tmp_path):
    sim = RingSim(SimConfig(seed=19, truck_share=0.1))
    sim.step(80)
    rec = Recorder(sim)
    for _ in range(40):
        sim.step()
        rec.update()
    path = tmp_path / "state.npz"
    saved = state.save(path, sim, rec, app={"note": "hello", "markers": [[1.5, "Density 50→60"]]})
    assert saved.path == path and not list(tmp_path.glob(".*.tmp"))
    with np.load(path, allow_pickle=False) as data:
        assert "meta_json" in data.files and data["meta_json"].dtype.kind == "U"
        assert {"sim.theta", "sim.acc_hist", "rec.values", "rec.spacetime"} <= set(data.files)
    loaded = state.load(path)
    assert loaded.meta == json.loads(json.dumps(saved.meta))
    assert loaded.meta["version"] == state.FORMAT_VERSION and loaded.app["note"] == "hello"
    assert loaded.meta["settings"] == {**{k: v for k, v in vars(sim.cfg).items()}, "seed": sim.seed}
    for name, arr in saved.arrays.items():
        assert_same(arr, loaded.arrays[name], name)
    assert_same(sim, loaded.simulation())
    assert state.saved_settings(path)["truck_share"] == 0.1


def test_missing_and_damaged_files_raise_a_readable_error(tmp_path):
    with pytest.raises(StateError, match="no saved state"):
        state.load(tmp_path / "missing.npz")
    junk = tmp_path / "junk.npz"
    junk.write_bytes(b"this is not a zip archive")
    with pytest.raises(StateError, match="not a readable state file"):
        state.load(junk)
    good = tmp_path / "good.npz"
    state.save(good, RingSim(SimConfig(seed=1)))
    cut = tmp_path / "cut.npz"
    cut.write_bytes(good.read_bytes()[: good.stat().st_size // 2])  # a save cut short
    with pytest.raises(StateError):
        state.load(cut)
    plain = tmp_path / "plain.npz"
    np.savez(plain, a=np.zeros(3))
    with pytest.raises(StateError, match="no meta_json"):
        state.load(plain)
    pickled = tmp_path / "pickled.npz"
    np.savez(pickled, meta_json=np.array("{}"), sneaky=np.array([{"x": 1}], dtype=object))
    with pytest.raises(StateError, match="not a readable state file"):  # never unpickled
        state.load(pickled)
    for path in (tmp_path / "missing.npz", junk, cut, plain, pickled):
        assert state.saved_settings(path) is None


def test_odd_files_raise_a_state_error_with_a_plain_reason(tmp_path):
    """Whatever the damage, a StateError (never numpy's advice to unpickle, never another exception)."""
    good = tmp_path / "good.npz"
    state.save(good, RingSim(SimConfig(seed=1)))
    odd = {"junk": b"\x00\x13garbage" * 50, "zeros": bytes(4096), "empty": b"", "text": b"density = 80\n",
           "cut": good.read_bytes()[:200]}
    for name, data in odd.items():
        (tmp_path / f"{name}.npz").write_bytes(data)
    with zipfile.ZipFile(tmp_path / "raw-member.npz", "w") as z:  # meta_json.npy holding no array at all
        z.writestr("meta_json.npy", b"not an array")
    np.savez(tmp_path / "deep.npz", meta_json=np.array("[" * 100_000))  # too deeply nested to parse
    np.savez(tmp_path / "pickled.npz", meta_json=np.array("{}"), sneaky=np.array([{"x": 1}], dtype=object))
    for path in sorted(tmp_path.glob("*.npz")):
        if path != good:
            with pytest.raises(StateError) as exc:
                state.load(path)
            assert "allow_pickle" not in str(exc.value) and "unsafely" not in str(exc.value), path.name
            assert state.saved_settings(path) is None, path.name
    for name in ("junk", "zeros", "empty", "text"):
        with pytest.raises(StateError, match=r"not a readable state file \(not an \.npz archive\)"):
            state.load(tmp_path / f"{name}.npz")
    with pytest.raises(StateError, match="holds Python objects, which are never loaded"):
        state.load(tmp_path / "pickled.npz")


@pytest.mark.parametrize("damage", ["time-nan", "time-negative", "steps", "generation", "timer"])
def test_a_damaged_clock_raises_state_error(tmp_path, damage):
    meta, arrays = saved_contents(tmp_path)
    meta = json.loads(json.dumps(meta))
    sim_meta = meta["sim"]
    if damage == "time-nan":
        sim_meta["time"] = float("nan")
    elif damage == "time-negative":
        sim_meta["time"] = -1.0
    elif damage in ("steps", "generation"):
        sim_meta[damage] = -5
    else:
        sim_meta["next_fill"] = float("inf")
    write_state(tmp_path / "bad.npz", meta, arrays)
    with pytest.raises(StateError, match="clock|timer"):
        state.load(tmp_path / "bad.npz").simulation()


def test_incompatible_files_raise_but_keep_their_settings(tmp_path):
    meta, arrays = saved_contents(tmp_path, density=70)

    old = dict(meta, version=0)
    write_state(tmp_path / "old.npz", old, arrays)
    with pytest.raises(StateError, match="format version 0"):
        state.load(tmp_path / "old.npz")
    assert state.saved_settings(tmp_path / "old.npz")["density"] == 70

    other = json.loads(json.dumps(meta))
    other["format"] = "something else"
    write_state(tmp_path / "other.npz", other, arrays)
    with pytest.raises(StateError, match="not a saved state of this model"):
        state.load(tmp_path / "other.npz")

    newer = json.loads(json.dumps(meta))
    newer["sim"]["config"]["warp_drive"] = 3
    newer["settings"]["warp_drive"] = 3
    del newer["sim"]["config"]["politeness"]
    write_state(tmp_path / "newer.npz", newer, arrays)
    with pytest.raises(StateError, match="unknown: warp_drive; missing: politeness"):
        state.load(tmp_path / "newer.npz").simulation()
    cfg, skipped = state.valid_settings(state.saved_settings(tmp_path / "newer.npz"))
    assert skipped == ["warp_drive"] and cfg.density == 70 and cfg.seed == 1

    engine = json.loads(json.dumps(meta))
    engine["sim"]["version"] = 99
    write_state(tmp_path / "engine.npz", engine, arrays)
    with pytest.raises(StateError, match="simulation state version 99"):
        state.load(tmp_path / "engine.npz").simulation()


@pytest.mark.parametrize("damage", ["shape", "missing", "dtype", "lane", "queue", "log", "log-fraction", "rng",
                                    "value", "slots", "cap"])
def test_damaged_contents_raise_state_error(tmp_path, damage):
    meta, arrays = saved_contents(tmp_path, traffic_mode="open", entry_rate=0.3)
    meta = json.loads(json.dumps(meta))
    if damage == "shape":
        arrays["sim.theta"] = arrays["sim.theta"][:-1]
    elif damage == "missing":
        del arrays["sim.v"]
    elif damage == "dtype":
        arrays["sim.lane"] = arrays["sim.lane"] + 0.5
    elif damage == "lane":
        lane = arrays["sim.lane"].copy()
        lane[np.flatnonzero(arrays["sim.alive"])[0]] = 7
        arrays["sim.lane"] = lane
    elif damage == "queue":
        arrays["sim.queue"] = arrays["sim.queue"] + 1  # more cars than arrival times
    elif damage == "log":  # one field type for (time, arm) entries
        assert arrays["sim.arrival_log"].shape[0] > 0
        meta["sim"]["logs"]["arrival_log"] = ["float"]
    elif damage == "log-fraction":  # an arm number that is not whole
        log = arrays["sim.arrival_log"].copy()
        log[0, 1] += 0.5
        arrays["sim.arrival_log"] = log
    elif damage == "rng":
        meta["sim"]["generators"]["rng"] = {"bit_generator": "Mystery"}
    elif damage == "value":
        meta["sim"]["config"]["density"] = -5
    elif damage == "cap":  # refused before allocating that many slots
        meta["sim"]["cap"] = 10**12
    else:
        arrays["sim.lanes_order"] = arrays["sim.lanes_order"] + 10_000
    write_state(tmp_path / "bad.npz", meta, arrays)
    with pytest.raises(StateError):
        state.load(tmp_path / "bad.npz").simulation()


def test_valid_settings_keep_what_they_can():
    cfg, skipped = state.valid_settings({"density": 80, "n_lanes": "two", "warp": 1, "seed": 5,
                                         "road_condition": "lava", "max_queue": 20.0, "radius": -3})
    assert (cfg.density, cfg.seed, cfg.max_queue, cfg.n_lanes, cfg.radius) == (80.0, 5, 20, 3, 80.0)
    assert type(cfg.max_queue) is int and skipped == ["n_lanes", "warp", "road_condition", "radius"]
    base = SimConfig(n_arms=2)
    assert state.valid_settings({}, base) == (base, [])


def test_writes_are_atomic(tmp_path, monkeypatch):
    path = tmp_path / "state.npz"
    first = RingSim(SimConfig(seed=20))
    state.save(path, first)
    second = RingSim(SimConfig(seed=21))
    second.step(10)

    def disk_full(fh, **arrays):
        fh.write(b"half a file")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", disk_full)
    with pytest.raises(OSError, match="disk full"):
        state.save(path, second)
    monkeypatch.undo()
    assert_same(first, state.load(path).simulation())  # the old file, whole
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.npz"]  # no temporary file left

    state.write_json(tmp_path / "s.json", {"a": 1})
    monkeypatch.setattr(state.json, "dumps", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(ValueError):
        state.write_json(tmp_path / "s.json", {"a": 2})
    monkeypatch.undo()
    assert state.read_json(tmp_path / "s.json") == {"a": 1}
    with pytest.raises(StateError):
        state.read_json(tmp_path / "nothing.json")


# ---------------------------------------------------------------- Recorder
def test_recorder_round_trip_keeps_rows_types_and_clock(tmp_path):
    sim = RingSim(SimConfig(seed=22, traffic_mode="open", entry_rate=0.2))
    rec = Recorder(sim, interval=0.5)
    for _ in range(60):
        sim.step()
        rec.update()
    rec.rows[0]["extra"] = "text"  # rows may differ in keys and hold any plain value
    rec.rows[1]["flag"] = True
    rec.rows[2]["nothing"] = None
    rec.rows[3]["queued"] = np.int64(4)
    path = tmp_path / "rec.npz"
    state.save(path, sim, rec)
    loaded = state.load(path)
    sim2 = loaded.simulation()
    rec2 = Recorder(sim2, interval=2.0)
    rec2.load_state(loaded.part("rec"))
    assert (rec2.interval, rec2._next, rec2._generation) == (rec.interval, rec._next, rec._generation)
    assert len(rec2.rows) == len(rec.rows)
    for a, b in zip(rec.rows, rec2.rows):
        assert list(a) == list(b)
        for k in a:
            assert a[k] == b[k] or (isinstance(a[k], float) and np.isnan(a[k]) and np.isnan(b[k])), k
            assert type(b[k]) is (int if isinstance(a[k], np.integer) else type(a[k])), k
    pd.testing.assert_frame_equal(rec.frame(), rec2.frame())
    np.testing.assert_array_equal(rec.spacetime_array()[1], rec2.spacetime_array()[1])
    np.testing.assert_array_equal(rec.occupancy_array(), rec2.occupancy_array())
    for s, r in ((sim, rec), (sim2, rec2)):  # both carry on sampling alike
        for _ in range(30):
            s.step()
            r.update()
    pd.testing.assert_frame_equal(rec.frame(), rec2.frame())


def test_recorder_restore_keeps_its_own_length_and_checks_the_bins():
    sim = RingSim(SimConfig(seed=23))
    rec = Recorder(sim, interval=0.1)
    for _ in range(20):
        sim.step()
        rec.update()
    short = Recorder(sim, maxlen=5)
    short.load_state(rec.state_dict())
    assert [r["time"] for r in short.rows] == [r["time"] for r in list(rec.rows)[-5:]]  # the newest
    with pytest.raises(StateError, match="angle bins"):
        Recorder(sim, bins=90).load_state(rec.state_dict())
    broken = rec.state_dict()
    broken["arrays"]["kinds"] = broken["arrays"]["kinds"][:-1]
    rows = list(short.rows)
    with pytest.raises(StateError):
        short.load_state(broken)
    assert list(short.rows) == rows  # unchanged


def test_recorder_restore_checks_the_clock_interval_and_metrics():
    sim = RingSim(SimConfig(seed=24))
    rec = Recorder(sim, interval=0.5)
    for _ in range(30):
        sim.step()
        rec.update()
    good = rec.state_dict()

    def damaged(**meta) -> dict:
        state_ = rec.state_dict()
        state_["meta"].update(meta)
        return state_

    # Clocks that would hang the next update (an interval too small to move it on, or a next sample
    # due ages ago) or stop the sampling for good (next sample never due, or NaN).
    for meta in (dict(interval=1e-300), dict(interval=float("nan")), dict(interval=0.0), dict(next=-1e300),
                 dict(next=float("nan")), dict(next=1e300), dict(next=sim.time + 5.0)):
        with pytest.raises(StateError, match="interval|clock"):
            Recorder(sim).load_state(damaged(**meta))
    lagging = Recorder(sim)
    lagging.load_state(damaged(next=sim.time - 10.0))  # not updated for a while: it catches up
    lagging.update()
    assert lagging._next > sim.time and len(lagging.rows) == len(rec.rows) + 1
    Recorder(sim).load_state(damaged(next=1e300, generation=sim.generation + 1))  # another run: starts afresh anyway

    with pytest.raises(StateError, match="sampled every 0.5 s; these sample every 2 s"):
        Recorder(sim).load_state(good, interval=2.0)
    Recorder(sim).load_state(good, interval=0.5)

    keys = list(good["meta"]["keys"])
    with pytest.raises(StateError, match="not every chart sample holds warp as a number"):
        Recorder(sim).load_state(good, numbers=["mean_speed", "warp"])
    for column, code, value, problem in (("mean_speed", metrics.TEXT, 0.0, "as a number"),
                                         ("mean_speed", metrics.MISSING, np.nan, "as a number"),
                                         ("mean_speed", metrics.FLOAT, np.inf, "infinite"),
                                         ("time", metrics.FLOAT, np.nan, "undefined")):
        broken = rec.state_dict()
        broken["arrays"]["kinds"][2, keys.index(column)] = code
        broken["arrays"]["values"][2, keys.index(column)] = value
        if code == metrics.TEXT:
            broken["meta"]["texts"] = [[2, keys.index(column), "fast"]]
        with pytest.raises(StateError, match=problem):
            Recorder(sim).load_state(broken, numbers=["mean_speed"])
    Recorder(sim).load_state(good, numbers=metrics.snapshot(sim).keys())  # every metric there


# ------------------------------------------------------------------ viewer
@pytest.fixture
def open_app(new_app):
    """Factory for viewers on SIZE with results in tmp_path/results; pygame is shut down afterwards."""
    return lambda config=None, **options: new_app(config, SIZE, **options)


def quit_by_closing(app: viewer.App) -> None:
    """Close the window: App.run handles the QUIT event (and autosaves) and shuts pygame down."""
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    app.run()


def assert_same_charts(a: viewer.App, b: viewer.App) -> None:
    pd.testing.assert_frame_equal(a.rec.frame(), b.rec.frame())
    assert (a.rec._next, a.rec._generation) == (b.rec._next, b.rec._generation)
    for name in viewer.History.KEYS:
        np.testing.assert_array_equal(a.hist.array(name), b.hist.array(name))
    for name in ("profile", "speed_rgb", "occupancy"):
        assert_same(list(getattr(a.hist, name)), list(getattr(b.hist, name)), name)
    assert a.hist._free == b.hist._free
    assert list(a.markers) == list(b.markers) and list(a.brakes) == list(b.brakes)


def test_quitting_saves_the_session_and_the_next_launch_resumes_it(open_app, tmp_path):
    a = open_app(autosave=True)
    frames(a, 30)
    a.handle(key(pygame.K_b))  # a braked car (ringed in the space-time diagram)
    w = widget(a, "slider", "density") if a.group == "Traffic" else None
    click(a, widget(a, "tab", "Traffic").rect.center)
    w = widget(a, "slider", "density")
    spec = w.spec
    target = (round(w.track.x + (70 - spec.lo) / (spec.hi - spec.lo) * w.track.w), w.track.centery)
    click(a, target)  # a live change: a chart marker
    frames(a, 30)
    click(a, widget(a, "tab", "Drivers").rect.center)
    a.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=a.content_rect.center, rel=(0, 0), buttons=(0, 0, 0)))
    a.handle(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-2, flipped=False, precise_x=0.0, precise_y=-2.0))
    for k in (pygame.K_PLUS, pygame.K_PLUS, pygame.K_c, pygame.K_SPACE):
        a.handle(key(k))
    a.spacetime_mode = "occupancy"
    assert a.markers and a.brakes and a.scroll > 0 and a.paused
    view = a.view_settings()
    quit_by_closing(a)

    folder = tmp_path / "results" / "session"
    summary = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert summary["view"] == view == dict(group="Drivers", scroll=a.scroll, steps_per_frame=3, colour_mode="desired",
                                           spacetime_mode="occupancy", paused=True)
    assert summary["settings"]["density"] == a.sim.cfg.density and summary["settings"]["seed"] == 7
    assert summary["time"] == a.sim.time and summary["state"] == "state.npz"

    b = open_app(SimConfig(), autosave=True, resume="session", restore_view=True)
    assert_same(a.sim, b.sim)
    assert_same_charts(a, b)
    assert b.view_settings() == view
    assert b.message[0].startswith("Resumed the saved session at t = ") and "--fresh" in b.message[0]
    frames(b, 2)  # draws as resumed (paused: nothing moves)
    assert b.sim.time == a.sim.time
    for app in (a, b):  # and both carry on alike
        app.advance(40)
    assert_same(a.sim, b.sim)
    assert_same_charts(a, b)


def test_esc_saves_the_session_too(open_app):
    a = open_app(autosave=True)
    frames(a, 10)
    pygame.event.clear()
    pygame.event.post(key(pygame.K_ESCAPE))
    a.run()  # Esc ends the loop like closing the window
    b = open_app(SimConfig(), resume="session", restore_view=True)
    assert_same(a.sim, b.sim)
    assert_same_charts(a, b)


def test_damaged_extras_or_charts_still_resume_the_simulation(open_app, tmp_path):
    a = open_app(SimConfig(seed=8, truck_share=0.2))
    frames(a, 20)
    extras = {**a._saved_extras(), "markers": [[1.0, "Density 50→60"], ["x", "bad"], [2.0], 5, [3.0, 4]],
              "brakes": "junk", "history_free": "fast"}
    path = tmp_path / "extras.npz"
    state.save(path, a.sim, a.rec, extras)
    b = open_app(resume=path)
    assert_same(a.sim, b.sim)
    assert list(b.markers) == [(1.0, "Density 50→60")] and not b.brakes  # the well-formed ones only
    pd.testing.assert_frame_equal(a.rec.frame(), b.rec.frame())
    assert b.message[0].startswith("Resumed") and "could not" not in b.message[0]

    meta, arrays = state.load(path).meta, dict(state.load(path).arrays)
    arrays["rec.kinds"] = arrays["rec.kinds"][:-1]  # chart samples that do not fit together
    write_state(tmp_path / "charts.npz", meta, arrays)
    c = open_app(resume=tmp_path / "charts.npz")
    assert_same(a.sim, c.sim)  # the simulation still carries on; its charts start afresh
    assert "Its charts could not be restored" in c.message[0] and len(c.rec.rows) == 1
    frames(c, 3)


def drop_chart_metric(meta: dict, arrays: dict, key: str) -> None:
    """Remove one metric from a state's chart samples (as if saved before that metric existed)."""
    keys = meta["rec"]["keys"]
    j = keys.index(key)
    del keys[j]
    for name in ("rec.values", "rec.kinds"):
        arrays[name] = np.delete(arrays[name], j, axis=1)
    meta["rec"]["texts"] = [[i, k - (k > j), text] for i, k, text in meta["rec"]["texts"] if k != j]


CHART_PROBLEMS = {"no-entry-vc": "not every chart sample holds entry_vc as a number",  # (a chart)
                  "no-trip-time": "not every chart sample holds mean_trip_time as a number",  # (the stats card)
                  "text-time": "not every chart sample holds time as a number",
                  "sampled-every-2-s": "charts sampled every 2 s; these sample every 0.5 s"}


@pytest.mark.parametrize("problem", CHART_PROBLEMS)
def test_charts_the_viewer_cannot_draw_start_afresh(open_app, tmp_path, problem):
    path = tmp_path / "charts.npz"
    if problem == "sampled-every-2-s":  # saved from a script, whose recorder samples less often
        sim = RingSim(SimConfig(seed=9))
        rec = Recorder(sim, interval=2.0)
        for _ in range(300):
            sim.step()
            rec.update()
        state.save(path, sim, rec)
    else:
        a = open_app(SimConfig(seed=9, traffic_mode="open", entry_rate=0.3))
        frames(a, 20)
        sim = a.sim
        state.save(path, sim, a.rec, a._saved_extras())
        loaded = state.load(path)
        meta, arrays = loaded.meta, dict(loaded.arrays)
        if problem == "text-time":
            j = meta["rec"]["keys"].index("time")
            arrays["rec.kinds"] = arrays["rec.kinds"].copy()
            arrays["rec.kinds"][3, j] = metrics.TEXT
            meta["rec"]["texts"] = [[3, j, "soon"]]
        else:
            drop_chart_metric(meta, arrays, "entry_vc" if problem == "no-entry-vc" else "mean_trip_time")
        write_state(path, meta, arrays)
    twin = RingSim.from_state(sim.state_dict())

    b = open_app(resume=path)  # --state PATH (and a session resumed on launch)
    c = open_app()
    assert c.load_state(path)  # F9
    for app in (b, c):
        assert app.message[0].startswith(("Resumed", "Loaded"))
        assert f"Its charts could not be restored ({CHART_PROBLEMS[problem]})." in app.message[0]
        assert app.rec.interval == viewer.SAMPLE_INTERVAL and len(app.rec.rows) == len(app.hist) == 1  # afresh
        frames(app, 3)  # draws and carries on
    twin.step(3)
    assert_same(twin, b.sim)  # the simulation itself resumed exactly


def test_a_session_that_cannot_be_resumed_is_kept_not_replaced(open_app, tmp_path, monkeypatch):
    a = open_app(SimConfig(seed=6, density=60), autosave=True)
    frames(a, 30)
    quit_by_closing(a)
    folder = tmp_path / "results" / "session"
    saved = state.load(folder / "state.npz")
    meta, arrays = dict(saved.meta, version=state.FORMAT_VERSION + 1), dict(saved.arrays)

    write_state(folder / "state.npz", meta, arrays)  # saved by a newer version of the model
    newer = (folder / "state.npz").read_bytes()
    b = open_app(SimConfig(), autosave=True, resume="session", restore_view=True)
    kept, = folder.glob("not_resumed_*.npz")  # moved out of the way of the autosave on quit
    assert kept.read_bytes() == newer and not (folder / "state.npz").exists()
    assert f"format version {state.FORMAT_VERSION + 1}" in b.message[0]
    assert b.message[0].endswith(f"Started afresh with its settings. That session is kept as {kept}.")
    assert b.sim.cfg.density == 60 and b.autosave
    frames(b, 5)
    quit_by_closing(b)
    assert state.load(folder / "state.npz").simulation().time == pytest.approx(0.5)  # this run's session
    assert kept.read_bytes() == newer  # and the one it could not resume, as it was
    with pytest.raises(StateError, match="format version"):
        state.load(kept)

    write_state(folder / "state.npz", meta, arrays)
    newer = (folder / "state.npz").read_bytes()
    c = open_app(SimConfig(), autosave=False, resume="session")  # --no-autosave: nothing would replace it
    assert c.message[0].endswith("Started afresh with its settings.") and list(folder.glob("not_resumed_*")) == [kept]

    def in_use(src, dst):
        raise PermissionError(13, "The file is in use")

    monkeypatch.setattr(viewer.os, "replace", in_use)  # another program holds it: it cannot be moved
    d = open_app(SimConfig(), autosave=True, resume="session")
    monkeypatch.undo()
    assert d.message[0].endswith("Quitting will not replace that session.") and not d.autosave
    frames(d, 5)
    quit_by_closing(d)
    assert (folder / "state.npz").read_bytes() == newer and list(folder.glob("not_resumed_*")) == [kept]


def test_saving_in_the_middle_of_a_live_drag_keeps_its_marker(open_app, tmp_path):
    a = open_app(autosave=True)
    frames(a, 10)
    click(a, widget(a, "tab", "Traffic").rect.center)
    w = widget(a, "slider", "density")
    spec, start = w.spec, a.sim.cfg.density
    assert not spec.reset  # applied while dragging, marked when let go

    def x(value):
        return round(w.track.x + (value - spec.lo) / (spec.hi - spec.lo) * w.track.w)

    a.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(x(start), w.track.centery), button=1))
    a.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=(x(start + 40), w.track.centery), rel=(0, 0),
                                buttons=(1, 0, 0)))
    frames(a, 5)
    assert a.drag and a.sim.cfg.density == a.drag.value != start and not a.markers
    label = f"{spec.label} {viewer.format_value(spec, start)}→{viewer.format_value(spec, a.drag.value)}"
    path = a.save_state()  # F5 with the mouse button still down
    assert [m[1] for m in state.load(path).app["markers"]] == [label] and not a.markers and a.drag
    quit_by_closing(a)  # the window closed before the button came up
    b = open_app(SimConfig(), resume="session")
    assert list(b.markers) == [(a.sim.time, label)]  # (so Save CSV lists the change too)


def test_session_summary_is_standard_json_with_infinite_settings(open_app, tmp_path):
    inf = float("inf")
    a = open_app(SimConfig(seed=2, density=inf, reaction_distance=inf), autosave=True)
    frames(a, 5)
    quit_by_closing(a)
    text = (tmp_path / "results" / "session" / "session.json").read_text(encoding="utf-8")

    def nonstandard(name):
        raise ValueError(f"not standard JSON: {name}")

    summary = json.loads(text, parse_constant=nonstandard)  # what any other JSON reader accepts
    assert summary["settings"]["density"] == summary["settings"]["reaction_distance"] == "inf"
    cfg, skipped = state.valid_settings(summary["settings"])  # read back as the numbers
    assert cfg.density == cfg.reaction_distance == inf and not skipped
    b = open_app(resume="session")
    assert b.sim.cfg.density == inf and b.sim.time == a.sim.time
    state.write_json(tmp_path / "x.json", {"a": [float("nan"), {"b": -inf}], "c": (1.5, inf)})
    assert state.read_json(tmp_path / "x.json") == {"a": ["nan", {"b": "-inf"}], "c": [1.5, "inf"]}


def test_the_most_recent_state_wins_ties_by_its_save_counter(open_app, tmp_path):
    """Saves in the same second (state_<date>_<time>-1, -2, ...) on a file system whose clock
    gives them the same time (FAT: 2 s): the later name wins."""
    app = open_app()
    folder = tmp_path / "results"
    folder.mkdir()
    names = ["state_20261025_120000.npz", "state_20261025_120000-1.npz", "state_20261025_120000-9.npz",
             "state_20261025_120000-10.npz", "state_20261025_115959-11.npz", "state_copy.npz"]
    for name in names:
        (folder / name).write_bytes(b"x")
        os.utime(folder / name, ns=(10**18, 10**18))
    assert app.latest_state().name == "state_20261025_120000-10.npz"
    os.utime(folder / names[0], ns=(10**18 + 10**9,) * 2)  # a later file time still decides first
    assert app.latest_state().name == names[0]


def test_viewer_without_session_options_saves_and_resumes_nothing(open_app, tmp_path):
    a = open_app()
    frames(a, 5)
    quit_by_closing(a)
    assert not (tmp_path / "results").exists()  # tests and scripts keep no session by default
    b = open_app(resume="session", restore_view=True)  # nothing saved yet: a first launch
    assert b.sim.time == 0 and b.message == ("", 0.0)


def test_command_line_session_flags():
    def options(*argv):
        args = viewer.parse_args(list(argv))
        return viewer.session_options(args), not args.no_autosave

    assert options() == ({"resume": "session", "restore_view": True}, True)
    assert options("--fresh") == ({"resume": None, "restore_view": False}, True)
    assert options("--no-autosave") == ({"resume": "session", "restore_view": True}, False)
    assert options("--state", "x.npz") == ({"resume": Path("x.npz"), "restore_view": True}, True)
    assert options("--state", "x.npz", "--fresh") == ({"resume": Path("x.npz"), "restore_view": False}, True)
    # Model settings start a new simulation; the view settings still restore (unless --fresh).
    assert options("--density", "80") == ({"resume": None, "restore_view": True}, True)
    assert options("--seed", "3", "--fresh") == ({"resume": None, "restore_view": False}, True)
    assert viewer.explicit_settings(viewer.parse_args(["--seed", "3", "--n-lanes", "2", "--size", "1100x700"])) \
        == ["--seed", "--n_lanes"]
    assert viewer.explicit_settings(viewer.parse_args(["--fresh", "--no-autosave", "--size", "1100x700"])) == []
    with pytest.raises(SystemExit) as exc:
        viewer.parse_args(["--state", "x.npz", "--density", "80"])
    assert exc.value.code == 2


def test_help_lists_the_keys_and_session_flags(capsys):
    with pytest.raises(SystemExit):
        viewer.parse_args(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    for word in ("F5 save the state", "F9", "--fresh", "--no-autosave", "--state PATH", "results/session/"):
        assert word in text, word
    assert "F5 save state" in viewer.DEFAULT_HINT and "F9 load state" in viewer.DEFAULT_HINT


def test_main_resumes_after_quitting_and_explicit_settings_start_anew(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # main() keeps its files in ./results
    apps = []
    run = viewer.App.run

    def run_a_little(self):
        apps.append(self)
        self.advance(25)
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        run(self)

    monkeypatch.setattr(viewer.App, "run", run_a_little)
    viewer.main(["--size", "1100x700", "--seed", "4", "--density", "60"])
    assert (tmp_path / "results" / "session" / "state.npz").is_file()
    viewer.main(["--size", "1100x700"])  # resumes: 2.5 s, then 2.5 s more
    assert apps[1].sim.time == pytest.approx(2 * apps[0].sim.time / 2 + 2.5)
    assert apps[1].sim.cfg.density == 60 and apps[1].sim.seed == 4
    viewer.main(["--size", "1100x700", "--density", "40"])  # a new simulation with these settings
    assert apps[2].sim.cfg.density == 40 and apps[2].sim.time == pytest.approx(2.5)
    viewer.main(["--size", "1100x700", "--fresh", "--no-autosave"])
    assert apps[3].sim.cfg == SimConfig() and apps[3].sim.time == pytest.approx(2.5)
    viewer.main(["--size", "1100x700"])  # --no-autosave kept the session of the third run
    assert apps[4].sim.cfg.density == 40 and apps[4].sim.time == pytest.approx(5.0)
    pygame.quit()


def test_explicit_settings_restore_only_the_view(open_app):
    a = open_app(autosave=True)
    a.group, a.colour_mode, a.steps_per_frame = "Demand", "lane", 5
    frames(a, 10)
    quit_by_closing(a)
    b = open_app(SimConfig(density=80), resume=None, restore_view=True)
    assert b.sim.time == 0 and b.sim.cfg.density == 80
    assert (b.group, b.colour_mode, b.steps_per_frame) == ("Demand", "lane", 5)
    c = open_app(resume=None, restore_view=False)  # --fresh
    assert (c.group, c.colour_mode, c.steps_per_frame) == (viewer.GROUPS[0], "speed", 1)


def test_resume_from_a_given_state_file(open_app, tmp_path):
    a = open_app(SimConfig(seed=3, traffic_mode="open", entry_rate=0.3))
    frames(a, 40)
    path = a.save_state()
    b = open_app(SimConfig(), resume=path)
    assert_same(a.sim, b.sim)
    assert_same_charts(a, b)
    assert b.message[0].startswith(f"Resumed {path} at t = 4.0 s")
    missing = open_app(SimConfig(), resume=tmp_path / "nowhere.npz")
    assert missing.sim.time == 0 and missing.sim.cfg == SimConfig()
    assert "Could not resume" in missing.message[0] and "default settings" in missing.message[0]


@pytest.mark.parametrize("problem", ["corrupt-state", "missing-state", "unknown-setting", "corrupt-summary",
                                     "nothing-readable", "bad-view"])
def test_a_damaged_session_falls_back_without_crashing(open_app, tmp_path, problem):
    a = open_app(SimConfig(seed=5, density=70, n_lanes=2), autosave=True)
    a.group, a.colour_mode = "Drivers", "lane"
    frames(a, 20)
    quit_by_closing(a)
    folder = tmp_path / "results" / "session"
    summary = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    if problem in ("corrupt-state", "nothing-readable"):
        (folder / "state.npz").write_bytes(b"garbage")
    if problem == "missing-state":
        (folder / "state.npz").unlink()
    if problem == "unknown-setting":
        (folder / "state.npz").unlink()
        summary["settings"].update(warp_drive=9, road_condition="lava")
    if problem == "bad-view":
        summary["view"] = dict(group="Nonsense", scroll=-4, steps_per_frame=99, colour_mode=3, paused="yes")
    (folder / "session.json").write_text(json.dumps(summary), encoding="utf-8")
    if problem in ("corrupt-summary", "nothing-readable"):
        (folder / "session.json").write_text("{not json", encoding="utf-8")

    b = open_app(SimConfig(), autosave=True, resume="session", restore_view=True)
    frames(b, 3)  # runs and draws
    text = b.message[0]
    if problem in ("corrupt-summary", "bad-view"):  # the state itself is fine
        assert b.sim.time == pytest.approx(a.sim.time + 0.3) and text.startswith("Resumed")
        expected_view = ("Drivers", "lane") if problem == "bad-view" else (viewer.GROUPS[0], "speed")
        if problem == "corrupt-summary":  # the state file's own copy of the view
            expected_view = ("Drivers", "lane")
        assert (b.group, b.colour_mode) == expected_view and b.steps_per_frame == 1 and not b.paused
        return
    assert b.sim.time == pytest.approx(0.3) and text.startswith("Could not resume the saved session: ")
    kept = sorted(folder.glob("not_resumed_*.npz"))
    if problem in ("corrupt-state", "nothing-readable"):  # moved aside: quitting would replace it
        assert [p.read_bytes() for p in kept] == [b"garbage"] and not (folder / "state.npz").exists()
        assert "pickle" not in text and "(not an .npz archive)" in text  # never numpy's "load it unsafely"
        text = text.removesuffix(f" That session is kept as {kept[0]}.")
    else:
        assert not kept
    if problem == "nothing-readable":
        assert b.sim.cfg == SimConfig() and text.endswith("Started with the default settings.")
        return
    assert (b.sim.cfg.density, b.sim.cfg.n_lanes, b.sim.seed) == (70, 2, 5)
    assert (b.group, b.colour_mode) == ("Drivers", "lane")
    if problem == "unknown-setting":
        assert text.endswith("Started afresh with its settings except road_condition, warp_drive.")
    else:
        assert text.endswith("Started afresh with its settings.")


def test_saving_where_it_cannot_write_does_not_crash(open_app, tmp_path, capsys):
    (tmp_path / "results").write_text("a file where the folder should be")
    app = open_app(autosave=True)
    frames(app, 3)
    app.handle(key(pygame.K_F5))
    assert app.message[0].startswith("Could not save the state")
    assert app.save_session() is None
    assert "Could not save the session" in capsys.readouterr().err
    quit_by_closing(app)  # quitting still works


def test_f5_saves_and_f9_carries_on_from_the_most_recent(open_app, tmp_path):
    app = open_app()
    frames(app, 3)
    app.handle(key(pygame.K_F9))
    assert app.message[0].startswith("No saved state (state_*.npz)")
    frames(app, 10)
    app.handle(key(pygame.K_F5))
    first = Path(app.message[0].split()[2])
    assert first.parent == tmp_path / "results" and first.name.startswith("state_") and first.is_file()
    frames(app, 20)
    click(app, widget(app, "button", "save_state").rect.center)  # the same second: a new name
    second = Path(app.message[0].split()[2])
    assert second != first and second.is_file() and app.latest_state() == second
    twin = RingSim.from_state(app.sim.state_dict())
    rows = app.rec.frame()
    brakes = list(app.brakes)
    frames(app, 25)
    app.handle(key(pygame.K_b))
    app.group, app.colour_mode = "Demand", "lane"

    app.handle(key(pygame.K_F9))  # back to the second save, charts and all; the view stays
    assert app.message[0].startswith(f"Loaded {second.name} at t = 3.3 s")
    assert_same(twin, app.sim)
    pd.testing.assert_frame_equal(app.rec.frame(), rows)
    assert len(app.hist) == len(rows) and list(app.brakes) == brakes
    assert (app.group, app.colour_mode) == ("Demand", "lane")
    frames(app, 5)
    twin.step(5)
    assert_same(twin, app.sim)

    click(app, widget(app, "button", "load_state").rect.center)
    assert app.sim.time == pytest.approx(3.3)

    damaged = tmp_path / "results" / "state_99999999_999999.npz"  # the newest file, unreadable
    damaged.write_bytes(b"garbage")
    os.utime(damaged, ns=(second.stat().st_mtime_ns + 10**9,) * 2)
    t = app.sim.time
    app.handle(key(pygame.K_F9))
    assert app.message[0].startswith(f"Could not load {damaged.name}") and app.message[0].endswith("Nothing changed.")
    assert app.sim.time == t

    meta, arrays = state.load(first).meta, dict(state.load(first).arrays)
    meta["sim"]["version"] = 0  # incompatible, but its settings are fine
    write_state(damaged, meta, arrays)
    app.handle(key(pygame.K_F9))
    assert "simulation state version 0" in app.message[0] and "Started afresh with its settings" in app.message[0]
    assert app.sim.time == 0 and app.sim.seed == 7
    frames(app, 3)


def test_state_buttons_fit_legibly_at_the_smallest_window(open_app):
    app = open_app()
    frames(app, 2)
    for name, label in (("save_state", "Save state"), ("load_state", "Load state")):
        w = widget(app, "button", name)
        assert app.panel_rect.contains(w.rect) and w.rect.bottom < app.content_rect.y
        assert app.fonts.width(label, "small") <= w.rect.w - 12  # room either side of the label
    for text in (viewer.DEFAULT_HINT, viewer.AUTOSAVE_HINT, *(h for _, _, h in viewer.BUTTONS)):
        assert len(app.fonts.wrap(text, app.status_rect.w)) <= app.status_lines, text[:40]


def test_new_paths_never_reuse_a_name(tmp_path):
    """Timestamped names for files saved in the same second get -1, -2, ...; every suffix is free."""
    folder = tmp_path / "out" / "deeper"
    first = state.new_paths(folder, "sweep", ".csv", "_meta.json")
    assert folder.is_dir() and first[0].name.startswith("sweep_") and first[1].name.endswith("_meta.json")
    first[1].write_text("{}")  # only the meta file of the first one exists
    second = state.new_paths(folder, "sweep", ".csv", "_meta.json")
    assert second[0].stem.endswith("-1") or second[0].stem[-15:] != first[0].stem[-15:]  # (a second later)
    assert not any(p.exists() for p in second)
