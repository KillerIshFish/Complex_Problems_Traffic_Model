"""The conflicts and delays at the entries (SPEC 3 'Conflict measurements', 6 S3, 12 item 6): merge_log (each entry
FORCED_WINDOW after it: entrant, lag vehicle, accepted lag = bumper gap / lag speed, inf for a standing lag, NaN for
none, platoon, forced yield, the lag's safety override acting); forced yields counted per pair type and law against
override; true PET at the entry points from positions (the entrant's rear clearing the point to the next vehicle's
front reaching it, up to PET_MAX); entry delay, travel time and control delay per entrant class, the capacity test's
standing queue left out; the summary columns and their definitions; observers that change nothing; saved states."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from roundabout_sim import experiments, presets
from roundabout_sim import sweep_viewer as sv
from roundabout_sim.config import PARAMS, SimConfig
from roundabout_sim.engine import (ACC, ADDED, CAV, CLASS_ARRAYS, ENTRANT_CLASSES, EVENT_LOGS, FORCED_WINDOW, HUMAN,
                                   MERGE_ARRAYS, PAIR_TYPES, PET_CRITICAL, PET_MAX, ROBOTAXI, SUBJECT_ARM, TWO_PI,
                                   RingSim)
from tests.test_entries import entry_scene
from tests.test_state import assert_exact_resume
from tests.test_state_v5 import as_version_4

OPEN = dict(traffic_mode="open", entry_rate=0.2)
WINDOW = int(round(FORCED_WINDOW / 0.1))  # [steps]
HH, HA, AH, AA = range(4)  # pair types, the lag vehicle (follower) first


class Admit(RingSim):
    """Lets every ready queue head in whatever the gaps (to provoke forced yields)."""

    def _entry_check(self, angles, trucks=None, s0=None, arms=None, **kw):
        checked = super()._entry_check(angles, trucks, s0, arms, **kw)
        return (np.ones_like(checked[0]), *checked[1:])


def scene(lag_kind: int, lag_gap: float, lag_v: float, cls=RingSim, head=HUMAN, **config) -> tuple[RingSim, int]:
    """entry_scene with a ``head`` queued at entrance 0, a lag vehicle of ``lag_kind`` ``lag_gap`` m (bumper) behind the
    entry point at ``lag_v`` m/s and a driver 60 m ahead; as ``cls``. Returns (sim, the lag's slot)."""
    kind = {ROBOTAXI: "robotaxi", ACC: "acc"}.get(lag_kind, "cav")
    sim, slots = entry_scene({0: (head, 0 if head != HUMAN else -1, 1.6)},
                             [(lag_kind, -(lag_gap + 4.0), lag_v), (HUMAN, 60.0, 6.0)], auto_kind=kind,
                             **config)
    sim.__class__ = cls
    return sim, int(slots[0])


# ------------------------------------------------------------------ the registry
def test_the_logs_and_counts_are_registered():
    assert {"merge_log", "merge_pending"} <= set(EVENT_LOGS) and ADDED["merge_log"] == ADDED["merge_pending"] == ()
    assert set(MERGE_ARRAYS) | set(CLASS_ARRAYS) <= set(ADDED) and ENTRANT_CLASSES == ("human", "coop", "other")
    assert (FORCED_WINDOW, PET_CRITICAL, PET_MAX) == (3.0, 1.5, 10.0)
    sim = RingSim(SimConfig(seed=1, **OPEN))
    sim.step(300)
    assert all(len(entry) == 3 for entry in sim.entry_log)  # (entry_log stays a 3-tuple)
    assert sim.merge_log and all(len(entry) == 10 for entry in sim.merge_log)


# ------------------------------------------------------------------ merge_log and the accepted lag
def test_an_entry_is_logged_with_its_lag_once_its_window_closes():
    """A driver enters 18 m (bumper) ahead of a robotaxi doing 6 m/s: accepted lag 18 / 6 = 3 s, pair AH (a recognisable
    lag behind a driver entrant); merge_log gets it FORCED_WINDOW s later, with its fields as Python values."""
    sim, lag = scene(ROBOTAXI, 18.0, 6.0)
    sim._entries()
    assert sim.n_entered == 1 and len(sim.merge_pending) == 1 and not sim.merge_log
    assert sim.merge_lags.tolist() == [0, 0, 1, 0] and sim.merge_lag_total[AH] == pytest.approx(3.0)
    entrant = int(np.flatnonzero(sim.alive & (sim.entered_at == 0.0))[0])
    sim.step(WINDOW - 1)
    assert len(sim.merge_pending) == 1 and not sim.merge_log
    sim.step(1)
    assert not sim.merge_pending and len(sim.merge_log) == 1
    t, arm, uid, kind, lag_uid, lag_kind, accepted, platoon, forced, acted = sim.merge_log[0]
    assert (t, arm, uid, kind, lag_uid, lag_kind) == (0.0, 0, int(sim.uid[entrant]), HUMAN, int(sim.uid[lag]),
                                                      ROBOTAXI)
    assert accepted == pytest.approx(3.0) and (platoon, forced, acted) == (False, False, False)
    assert [type(x) for x in sim.merge_log[0]] == [float, int, int, int, int, int, float, bool, bool, bool]
    assert sim.merge_closed.tolist() == [0, 0, 1, 0]


def test_the_accepted_lag_of_a_standing_lag_is_inf_and_without_one_nan():
    sim, lag = scene(HUMAN, 14.0, 0.0)
    sim._entries()
    assert sim.merge_pending[0][8] == math.inf and sim.merge_lags[HH] == 1 and sim.merge_lag_finite[HH] == 0
    empty, _ = entry_scene({0: (HUMAN, -1, np.nan)}, [])
    empty._entries()
    assert empty.n_entered == 1 and math.isnan(empty.merge_pending[0][8]) and empty.merge_pending[0][5] == -1
    empty.step(WINDOW)
    assert empty.merge_log[0][4:6] == (-1, -1) and empty.merge_closed.sum() == 0


# ------------------------------------------------------------------ forced yields
@pytest.mark.parametrize("lag_kind, pair, override", [
    (HUMAN, HH, False),  # a driver brakes by itself (the law)
    (ACC, HH, True),  # an ACC car looks like any car (H) and its safety override brakes it
    (CAV, AH, True),
])
def test_a_lag_forced_to_brake_hard_is_a_forced_yield(lag_kind, pair, override):
    """Let in 8 m (bumper) ahead of a lag vehicle doing 10 m/s, the entrant at 6 m/s forces it to brake harder than
    comfortably (2 m/s2) within FORCED_WINDOW s: a forced yield, by its law (or driver) or its safety override."""
    sim, lag = scene(lag_kind, 8.0, 10.0, Admit)
    sim._entries()
    sim.step(WINDOW)
    *_, forced, acted = sim.merge_log[0]
    assert forced and acted == override
    assert sim.merge_closed[pair] == 1
    assert (sim.forced_override[pair], sim.forced_law[pair]) == ((1, 0) if override else (0, 1))


def test_a_lag_far_behind_is_no_forced_yield():
    sim, lag = scene(HUMAN, 40.0, 6.0)
    sim._entries()
    sim.step(WINDOW)
    assert sim.merge_log[0][8] is False and sim.forced_law.sum() == sim.forced_override.sum() == 0


# ------------------------------------------------------------------ PET
class Track(RingSim):
    """Records, around every step, the positions an independent PET recount needs."""

    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            t0 = self.time
            before = (self.travel.copy(), self.theta.copy(), self.alive.copy(), self.lane.copy())
            super().step()
            self.track.append((t0, before, (self.travel.copy(), self.theta.copy(), self.alive.copy())))


def pet_recount(sim: Track, entrant: int, lag: int, arm: int = 0) -> float:
    """PET from the tracked positions: when the entrant's rear (its centre half a length on) passed the entry point and
    when the lag's front reached it, each interpolated within its step."""
    r, half_e, half_l = sim.r_out, 0.5 * sim.length[entrant], 0.5 * sim.length[lag]
    p = sim.entry_angle[arm]
    clear = reach = None
    for t0, (tr0, th0, _, _), (tr1, th1, _) in sim.track:
        a0, a1 = tr0[entrant] * r - half_e, tr1[entrant] * r - half_e
        if clear is None and a1 >= 0 > a0:
            clear = t0 + (-a0) / (a1 - a0) * sim.cfg.dt
        d0 = ((p - th0[lag]) % TWO_PI) * r - half_l  # [m] its front short of the point
        d1 = ((p - th1[lag]) % TWO_PI) * r - half_l
        if clear is not None and reach is None and d0 > 0 and (d1 <= 0 or d1 > d0):
            reach = t0 + d0 / (d0 - (d1 if d1 <= 0 else d1 - TWO_PI * r)) * sim.cfg.dt
    return reach - clear


@pytest.mark.parametrize("lag_kind, lag_gap, pair", [(HUMAN, 20.0, HH), (ROBOTAXI, 22.0, AH)])
def test_pet_is_the_entrants_rear_clearing_the_point_to_the_lags_front_reaching_it(lag_kind, lag_gap, pair):
    sim, lag = scene(lag_kind, lag_gap, 8.0, Track)
    sim.track = []
    sim._entries()
    entrant = int(np.flatnonzero(sim.alive & (sim.entered_at == 0.0))[0])
    sim.step(150)
    assert sim.pet_count.tolist() == [int(k == pair) for k in range(4)]
    pet = pet_recount(sim, entrant, lag)
    assert 0 < pet < PET_MAX and sim.pet_total[pair] == pytest.approx(pet, abs=1e-9)
    assert sim.pet_low[pair] == int(pet < PET_CRITICAL) and sim.pet_slot[0] == -1


def test_a_pet_above_pet_max_is_no_conflict():
    sim, lag = scene(HUMAN, 120.0, 8.0)
    sim._entries()
    sim.step(400)
    assert sim.pet_count.sum() == 0 and sim.pet_slot[0] == -1  # (closed when the lag arrived, ~15 s on)


def test_a_group_entering_one_gap_has_one_pet():
    """Two CAVs enter one gap (the second by the CAV follow-up behind the first): the lag vehicle's PET is with the
    second, the entrant directly ahead of it."""
    sim, lag = scene(HUMAN, 40.0, 6.0, head=CAV, follow_up=2.5)
    sim.queue[0] = 2
    for name, value in (("queue_times", 0.0), ("queue_trucks", False), ("queue_kinds", CAV), ("queue_companies", 0),
                        ("queue_acc_gaps", 1.6)):
        getattr(sim, name)[0].append(value)
    first = None
    for _ in range(300):
        sim.step()
        if first is None and sim.n_entered:
            first = int(sim.last_uid[0])
    assert sim.n_entered == 2 and sim.pet_count.sum() == 1
    assert [entry[7] for entry in sim.merge_log] == [False, True]  # (the second: a platoon entry)
    assert sim.merge_log[1][2] != first


# ------------------------------------------------------------------ delays by entrant class
class TripLog(RingSim):
    """Records each timed leaver's queue wait, time on the ring, angle driven and free speed (as _time_trips is
    called)."""

    def _time_trips(self, slots) -> None:
        for s in slots:
            free = float(np.minimum(self.desired_speed(np.array([s]))[0],
                                    np.sqrt(self.eff.lateral_accel * self.r_out)))
            self.trips.append((int(self.kind[s]), float(self.waited[s]), self.time - float(self.entered_at[s]),
                               float(self.travel[s]) * self.r_out / free))
        super()._time_trips(slots)


@pytest.mark.parametrize("coop, coop_class_index", [("B", 1), ("none", 2)])
def test_the_delays_by_entrant_class(coop, coop_class_index):
    """Open mode, 50 % CAVs: every entry counted in its class (drivers 0; CAVs 1 if they cooperate, else 2), with its
    queue wait; each timed exit's travel time = wait + time on the ring, its control delay = travel time - the angle
    driven at its free speed on the outer lane."""
    sim = TripLog(SimConfig(seed=2, auto_share=0.5, auto_kind="cav", coop_class=coop, **OPEN))
    sim.trips = []
    sim.step(2500)
    assert sim.class_entries.sum() == sim.n_entered and sim.class_entries[0] > 20
    assert sim.class_entries[coop_class_index] > 20 and sim.class_entries[3 - coop_class_index] == 0
    assert sim.class_wait.sum() == pytest.approx(sim.entry_wait_total, rel=1e-12)
    assert sim.trips and sim.class_exits.sum() == len(sim.trips)
    for c in range(3):
        mine = [trip for trip in sim.trips if (0 if trip[0] == HUMAN else coop_class_index) == c]
        assert sim.class_exits[c] == len(mine)
        assert sim.class_travel[c] == pytest.approx(sum(w + ring for _, w, ring, _ in mine), rel=1e-12)
        assert sim.class_control[c] == pytest.approx(sum(w + ring - free for _, w, ring, free in mine), rel=1e-12)
    assert (sim.class_control <= sim.class_travel).all()


def test_the_capacity_tests_standing_queue_is_left_out_of_the_delays():
    sim = RingSim(presets.apply("us_single_lane", SimConfig(seed=3)).updated(subject_arm="on", entry_rate=0.06,
                                                                              auto_share=0.5))
    sim.step(3000)
    assert sim.class_entries.sum() == sim.arm_entered[1:].sum() < sim.n_entered
    entrants = sim.alive & np.isfinite(sim.entered_at)
    assert np.isnan(sim.waited[entrants]).any() and np.isfinite(sim.waited[entrants]).any()  # (the test's: no wait)
    assert sim.pet_count.sum() > 0 and sim.merge_closed.sum() > 0.5 * sim.n_entered  # (its conflicts do count)


# ------------------------------------------------------------------ observers
class Blind(RingSim):
    """The entries' observers switched off."""

    def _observe_entries(self, idx, old, dtheta):
        pass

    def _record_entry(self, *a):
        pass

    def _time_trips(self, slots):
        pass


@pytest.mark.parametrize("config", [dict(**OPEN), dict(auto_share=0.5, auto_kind="cav", **OPEN),
                                    dict(auto_share=0.5, auto_kind="robotaxi", exploit_gap=0.8, follow_adapt=1.0,
                                         **OPEN)], ids=["human", "cav", "robotaxi-adapting"])
def test_the_entry_observers_change_nothing(config):
    real, blind = RingSim(SimConfig(seed=6, **config)), Blind(SimConfig(seed=6, **config))
    real.step(1200)
    blind.step(1200)
    for name in ("theta", "v", "lane", "alive", "uid", "queue"):
        np.testing.assert_array_equal(getattr(real, name), getattr(blind, name))
    for name in ("rng", "rng_demand", "rng_drivers", "rng_kinds"):
        assert getattr(real, name).bit_generator.state == getattr(blind, name).bit_generator.state
    assert real.merge_closed.sum() > 0 and real.pet_count.sum() > 0 and blind.merge_closed.sum() == 0


# ------------------------------------------------------------------ summary columns
def counts(**values) -> dict:
    base = {name: np.zeros(len(PAIR_TYPES)) for name in MERGE_ARRAYS}
    base.update({name: np.zeros(len(ENTRANT_CLASSES)) for name in CLASS_ARRAYS})
    base.update({k: np.asarray(v, dtype=float) for k, v in values.items()})
    return base


def test_the_entry_columns_from_the_counts():
    now = counts(merge_lag_total=[4, 2, 6, 0], merge_lag_finite=[2, 1, 2, 0], forced_law=[1, 0, 1, 0],
                 forced_override=[0, 0, 1, 0], merge_closed=[4, 2, 2, 0], pet_count=[2, 0, 1, 0],
                 pet_total=[6, 0, 2, 0], pet_low=[1, 0, 1, 0], class_entries=[2, 1, 0], class_wait=[10, 2, 0],
                 class_exits=[1, 0, 0], class_travel=[30, 0, 0], class_control=[12, 0, 0])
    row = experiments.entry_columns(now, counts())
    assert list(row) == list(experiments.ENTRY_COLUMNS)
    want = {"accepted_lag_s": 12 / 5, "accepted_lag_h_s": 2.0, "accepted_lag_a_s": 3.0,
            "forced_yield_share": 3 / 8, "forced_yield_h_share": 1 / 6, "forced_yield_a_share": 1.0,
            "forced_yield_law_share": 2 / 8, "forced_yield_override_share": 1 / 8,
            "pet_mean_s": 8 / 3, "pet_low_share": 2 / 3, "pet_h_mean_s": 3.0, "pet_a_mean_s": 2.0,
            "pet_h_low_share": 0.5, "pet_a_low_share": 1.0,
            "entry_delay_human_s": 5.0, "entry_delay_coop_s": 2.0, "travel_time_human_s": 30.0,
            "control_delay_human_s": 12.0}
    for name, value in want.items():
        assert row[name] == pytest.approx(value), name
    assert all(math.isnan(row[name]) for name in ("entry_delay_other_s", "travel_time_coop_s", "control_delay_other_s"))
    assert all(math.isnan(v) for v in experiments.entry_columns(counts(), counts()).values())


def test_the_entry_columns_are_defined_and_named_unlike_any_setting(tmp_path, capsys, no_figures):
    out = tmp_path / "e"
    experiments.main(["sweep", "--x", "auto_share=0,0.5", "--set", "traffic_mode=open", "--set", "entry_rate=0.2",
                      "--seeds", "1", "--warmup", "20", "--measure", "120", "--workers", "1", "--out", str(out)])
    df = pd.read_csv(f"{out}.csv")
    cols = experiments.ENTRY_COLUMNS
    assert set(cols) <= set(df.columns) and not set(cols) & {p.name for p in PARAMS}
    for name in cols:  # experiments.DEFINITIONS explains each (the sweep viewer's help), with a readable label
        assert name in sv.definitions(), name
        assert sv.column_label(name) != name, name
    off, on = df.iloc[0], df.iloc[1]
    assert off["entry_delay_human_s"] >= 0 and np.isnan(off["entry_delay_coop_s"]) and np.isnan(off["accepted_lag_a_s"])
    assert on["entry_delay_coop_s"] >= 0 and on["accepted_lag_a_s"] > 0 and 0 <= on["forced_yield_share"] <= 1
    assert on["forced_yield_law_share"] + on["forced_yield_override_share"] == pytest.approx(on["forced_yield_share"])
    capsys.readouterr()


def test_the_run_printout_shows_the_entry_measures_with_automated_vehicles(tmp_path, capsys, no_figures):
    experiments.main(["run", "--set", "auto_share=0.5", "--set", "traffic_mode=open", "--set", "entry_rate=0.2",
                      "--set", "exploit_gap=0.8", "--set", "follow_adapt=0.5", "--duration", "60",
                      "--out", str(tmp_path / "run")])
    printed = capsys.readouterr().out
    for text in ("Entry delay, human / coop CAV / other", "Forced yields, H / A lag vehicle", "PET at the entries",
                 "drivers: gap ahead of AVs x 0.8, following AVs 0.5"):
        assert text in printed, text
    experiments.main(["run", "--duration", "20", "--out", str(tmp_path / "plain")])
    assert "PET at the entries" not in capsys.readouterr().out


# ------------------------------------------------------------------ saved states
def test_a_run_with_entries_under_way_resumes_exactly(tmp_path):
    sim = RingSim(SimConfig(seed=9, auto_share=0.5, auto_kind="robotaxi", road_condition="snow", av_handover=5.0,
                            av_weather_scale=1.5, exploit_gap=0.8, follow_adapt=1.0, **OPEN))
    for _ in range(3000):
        sim.step()
        if sim.merge_pending and (sim.pet_slot >= 0).any() and sim.merge_log and sim.time > 100:
            break
    assert sim.merge_pending and (sim.pet_slot >= 0).any() and np.isfinite(sim.waited).any()
    assert_exact_resume(sim, tmp_path, more=200)


def test_an_older_state_knows_no_entries_under_way():
    sim = RingSim(SimConfig(seed=4, auto_share=0.5, **OPEN))
    sim.step(600)
    old = RingSim.from_state(as_version_4(sim.state_dict()))
    assert not old.merge_log and not old.merge_pending and (old.pet_slot == -1).all() and np.isnan(old.waited).all()
    assert all((getattr(old, name) == 0).all() for name in (*MERGE_ARRAYS, *CLASS_ARRAYS))
    assert np.isneginf(old.last_entry).all() and np.isnan(old.pet_clear).all()
    fills = (old.cfg.exploit_gap, old.cfg.follow_adapt, old.cfg.av_weather_scale, old.cfg.av_handover)
    assert fills == (1.0, 0.0, 1.0, 0.0)
    again = json.loads(json.dumps(old.state_dict()["meta"]))
    assert again["version"] == 5 and "merge_log" in again["logs"]
