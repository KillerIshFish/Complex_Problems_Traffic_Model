"""The automated vehicles' measurements (SPEC 6): time-to-collision episodes (TTC <= 1.5 s, SSAM; per-car hold) and
time exposed (TET, Minderhoud & Bovy 2001) by follower-leader pair type HH/HA/AH/AA (FOLLOWER letter first, as Hu
et al. 2023; H = a human driver or an ACC car, A = a recognisable AV), each per hour of following exposure (a
leader within the reaction distance); harsh braking of human-driven and automated vehicles per vehicle-hour; the
safety layer's interventions per automated vehicle-hour; cut-ins ahead of recognisable AVs; the share of the
automated vehicles' time in CACC. Exact counters, observers only (they change nothing), summary columns with
DEFINITIONS named unlike any setting, the in_cacc snapshot key and the run printout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from roundabout_sim import experiments, metrics
from roundabout_sim import sweep_viewer as sv
from roundabout_sim.config import PARAMS, SimConfig
from roundabout_sim.engine import (ACC, CAV, HUMAN, PAIR_ARRAYS, PAIR_TYPES, ROBOTAXI, SEEN_AS_AV, TTC_CRITICAL,
                                   RingSim)
from tests.test_controllers import lane_of

HH, HA, AH, AA = range(4)


def observe(sim) -> None:
    """One call of the observer on the state as _move sees it at the start of a step."""
    idx = np.flatnonzero(sim.alive)
    lead, gap = sim._index_lanes(idx)
    sim._observe_following(idx, sim.v[idx], lead[idx], gap[idx])


# ------------------------------------------------------------------ TTC, TET, exposure
def test_ttc_episodes_tet_and_following_exposure_on_hand_built_pairs():
    """Front to back: a leader at 4 m/s; 8 m behind it a car at 10 m/s (TTC 8 / 6 = 1.33 s <= 1.5: low); 10 m
    behind that one at 12 m/s (TTC 5 s: following, not low); 30 m behind that one at 20 m/s (beyond the 25 m
    reaction distance: not following, though closing at TTC 3.75 s). The leader follows nobody near."""
    assert TTC_CRITICAL == 1.5 and PAIR_TYPES == ("HH", "HA", "AH", "AA")
    sim, slots = lane_of([HUMAN] * 4, [0.0, 8.0, 10.0, 30.0], [4.0, 10.0, 12.0, 20.0], auto_kind="acc",
                         auto_share=0.0)
    assert not sim.automated and sim.cfg.reaction_distance == 25.0
    observe(sim)
    assert sim.pair_follow_steps.tolist() == [2, 0, 0, 0] and sim.pair_ttc_steps.tolist() == [1, 0, 0, 0]
    assert sim.pair_ttc_episodes.tolist() == [1, 0, 0, 0] and sim.n_human_steps == 4 and sim.n_auto_steps == 0
    assert sim.ttc_until[slots[1]] == sim.time + 2.0 and np.isneginf(sim.ttc_until[slots[[0, 2, 3]]]).all()
    observe(sim)  # the same moment again: one more step below, the episode still under way
    assert sim.pair_ttc_steps.tolist() == [2, 0, 0, 0] and sim.pair_ttc_episodes.tolist() == [1, 0, 0, 0]
    sim.time += 2.1  # after the hold: a new episode
    observe(sim)
    assert sim.pair_ttc_episodes.tolist() == [2, 0, 0, 0] and sim.pair_follow_steps.tolist() == [6, 0, 0, 0]
    # equal speeds at any gap, or opening: never low
    calm, _ = lane_of([HUMAN] * 3, [0.0, 2.0, 3.0], [8.0, 8.0, 7.0], auto_kind="acc", auto_share=0.0)
    observe(calm)
    assert calm.pair_follow_steps[HH] == 2 and calm.pair_ttc_steps.sum() == 0


def test_pair_types_name_the_follower_first():
    """H = human driver or ACC car (looks like any car), A = recognisable AV (cav, robotaxi, smoother); the
    follower's letter first: a robotaxi behind a human is AH."""
    assert SEEN_AS_AV.tolist() == [False, False, True, True, True]  # human, acc, cav, robotaxi, smoother
    sim, slots = lane_of([HUMAN, ROBOTAXI, HUMAN, ROBOTAXI, ROBOTAXI], [0.0, 10.0, 10.0, 10.0, 10.0], [6.0] * 5,
                         auto_kind="robotaxi")
    observe(sim)
    assert sim.pair_follow_steps.tolist() == [0, 1, 2, 1]  # (H behind R: HA; R behind H twice: AH; R behind R)
    assert sim.n_auto_steps == 3 and sim.n_human_steps == 2
    mixed, _ = lane_of([ACC, CAV, ACC], [0.0, 10.0, 10.0], [6.0] * 3, auto_kind="cav")
    observe(mixed)
    assert mixed.pair_follow_steps.tolist() == [0, 1, 1, 0]  # CAV behind ACC: AH; ACC behind CAV: HA
    assert mixed.n_auto_steps == 3 and mixed.n_human_steps == 0  # (an ACC car is automated, seen as H)
    closing, cs = lane_of([HUMAN, ROBOTAXI, HUMAN], [0.0, 5.0, 5.0], [4.0, 9.0, 9.5], auto_kind="robotaxi")
    observe(closing)  # robotaxi behind the human closes at TTC 1 s: AH; the human behind it at TTC 10 s
    assert closing.pair_ttc_steps.tolist() == [0, 0, 1, 0] and closing.pair_ttc_episodes.tolist() == [0, 0, 1, 0]


def test_the_safety_columns_from_the_exact_counts():
    """safety_columns: rates over vehicle-hours of exposure (steps x dt / 3600), NaN over none."""
    sim = RingSim(SimConfig(seed=1, auto_share=0.5))
    start = {k: (np.zeros(4, dtype=np.int64) if k.startswith("pair") else 0) for k in
             ("auto_steps", "human_steps", "aeb", "emergency", "emergency_auto", "cutins_auto", "pair_follow",
              "pair_ttc", "pair_episodes")}
    now = dict(auto_steps=36000, human_steps=72000, aeb=5, emergency=7, emergency_auto=3, cutins_auto=4,
               pair_follow=np.array([36000, 18000, 0, 3600]), pair_ttc=np.array([100, 50, 0, 36]),
               pair_episodes=np.array([4, 2, 0, 1]))
    row = experiments.safety_columns(sim, now, start, 1800.0, {"in_cacc": 3.0, "automated": 12.0})
    assert list(row) == list(experiments.SAFETY_COLUMNS)
    expect = dict(cacc_share=0.25, aeb_per_veh_h=5.0, harsh_human_per_veh_h=2.0, harsh_auto_per_veh_h=3.0,
                  ttc_per_h=7 / 1.6, ttc_hh_per_h=4.0, ttc_ha_per_h=4.0, ttc_aa_per_h=10.0, tet_s_per_h=18.6 / 1.6,
                  tet_hh_s_per_h=10.0, tet_ha_s_per_h=10.0, tet_aa_s_per_h=36.0, follow_hh_h=1.0, follow_ha_h=0.5,
                  follow_ah_h=0.0, follow_aa_h=0.1, cutins_ahead_auto_per_h=8.0)
    for name, value in expect.items():
        assert row[name] == pytest.approx(value), name
    assert np.isnan(row["ttc_ah_per_h"]) and np.isnan(row["tet_ah_s_per_h"])  # no AH following at all
    none = experiments.safety_columns(sim, start, start, 1800.0, {"in_cacc": 0.0, "automated": 0.0})
    assert all(np.isnan(none[k]) for k in ("cacc_share", "aeb_per_veh_h", "harsh_auto_per_veh_h", "ttc_per_h"))
    assert none["cutins_ahead_auto_per_h"] == 0.0


# ------------------------------------------------------------------ counters in runs
def test_exposure_and_harsh_braking_by_group_in_runs():
    """The vehicle-steps on the ring add up to the cars at the start of each step; harsh-braking episodes split
    into the automated vehicles' (n_emergency_auto) and the rest; a human-only run counts every pair HH."""
    for share, kind in ((0.0, "cav"), (0.5, "acc"), (1.0, "acc")):
        sim = RingSim(SimConfig(seed=2, n_lanes=1, n_arms=0, density=70, auto_share=share, auto_kind=kind))
        cars = 0
        for _ in range(2500):
            cars += int(sim.alive.sum())
            sim.step()
        assert sim.n_auto_steps + sim.n_human_steps == cars
        assert 0 <= sim.n_emergency_auto <= sim.n_emergency
        if share == 0:
            assert sim.n_auto_steps == 0 and sim.n_emergency_auto == 0 and sim.n_aeb == 0
            assert sim.pair_follow_steps[1:].sum() == 0 and sim.pair_follow_steps[0] > 0
        if share == 1:
            assert sim.n_human_steps == 0 and sim.n_emergency_auto == sim.n_emergency
            assert sim.pair_follow_steps[1:].sum() == 0  # (ACC cars look like any car: HH)


class CutInRecount(RingSim):
    """Recounts the lane changes whose new follower is a recognisable AV within the reaction distance behind
    (bumper to bumper) from the lanes after each step's changes (independently of the engine's own count);
    ``farther``: those with the AV beyond it, which are no cut-ins."""

    recount = farther = 0

    def _lane_changes(self, idx):
        before = self.lane.copy()
        super()._lane_changes(idx)
        for c in np.flatnonzero(self.alive & (self.lane != before)):
            mates = np.flatnonzero(self.alive & (self.lane == self.lane[c]))
            behind = mates[mates != c]
            if behind.size:
                back = (self.theta[c] - self.theta[behind]) % (2 * np.pi)
                f = behind[np.argmin(back)]
                bumper = back.min() * self.radii[self.lane[c]] - 0.5 * (self.length[c] + self.length[f])
                if SEEN_AS_AV[self.kind[f]]:
                    near = bumper <= self.cfg.reaction_distance
                    self.recount += bool(near)
                    self.farther += not near


def test_cut_ins_ahead_of_recognisable_avs():
    sim = CutInRecount(SimConfig(seed=3, n_lanes=3, density=40, exit_prob=0.3, auto_share=0.4, auto_kind="robotaxi"))
    sim.step(2000)
    assert sim.n_cutins_auto == sim.recount > 0 and sim.n_cutins_auto < sim.n_lane_changes
    assert sim.farther > 0  # (changes far ahead of an AV happen, and are not counted)
    acc = CutInRecount(SimConfig(seed=3, n_lanes=3, density=40, exit_prob=0.3, auto_share=0.4, auto_kind="acc"))
    acc.step(500)
    assert acc.n_cutins_auto == acc.recount == 0 and acc.n_lane_changes > 0  # (ACC cars are not recognisable)


@pytest.mark.parametrize("config", [dict(), dict(auto_share=0.5, auto_kind="robotaxi"),
                                    dict(auto_share=0.5, auto_kind="cav", companies=2),
                                    dict(auto_share=0.5, auto_kind="acc", traffic_mode="open", entry_rate=0.2)],
                         ids=["human", "robotaxi", "cav", "acc-open"])
def test_the_observers_change_nothing(config):
    """The surrogate safety counts draw nothing and touch nothing the dynamics read: runs without them (the
    observer a no-op) are the same bit for bit, every random stream included."""
    cfg = SimConfig(seed=4, density=60, **config)
    real, quiet = RingSim(cfg), RingSim(cfg)
    quiet._observe_following = lambda *a: None
    real.step(800)
    quiet.step(800)
    for name in ("theta", "v", "lane", "alive", "uid"):
        np.testing.assert_array_equal(getattr(real, name), getattr(quiet, name))
    for name in ("rng", "rng_demand", "rng_drivers", "rng_kinds"):
        assert getattr(real, name).bit_generator.state == getattr(quiet, name).bit_generator.state
    assert real.pair_follow_steps.sum() > 0 and quiet.pair_follow_steps.sum() == 0
    assert set(PAIR_ARRAYS) == {"pair_follow_steps", "pair_ttc_steps", "pair_ttc_episodes"}


# ------------------------------------------------------------------ snapshot and summaries
def test_the_in_cacc_snapshot_key():
    sim = RingSim(SimConfig(seed=5, auto_share=0.6, auto_kind="cav"))
    sim.step(300)
    row = metrics.snapshot(sim)
    assert row["in_cacc"] == int((sim.string_pos[sim.alive] >= 2).sum()) > 0 and type(row["in_cacc"]) is int
    assert metrics.snapshot(RingSim(SimConfig(seed=5)))["in_cacc"] == 0
    empty = RingSim(SimConfig(seed=2, traffic_mode="open", entry_rate=0.0, density=1, auto_share=0.5))
    empty.alive[:] = False
    assert metrics.snapshot(empty)["in_cacc"] == 0 and metrics.ADDED_KEYS["in_cacc"] == 0.0


def test_the_safety_columns_are_defined_and_named_unlike_any_setting(tmp_path, capsys, no_figures):
    out = tmp_path / "s"
    experiments.main(["sweep", "--x", "auto_share=0,0.6", "--set", "auto_kind=cav", "--set", "density=70",
                      "--seeds", "1", "--warmup", "20", "--measure", "60", "--workers", "1", "--out", str(out)])
    df = pd.read_csv(f"{out}.csv")
    cols = experiments.SAFETY_COLUMNS
    assert set(cols) <= set(df.columns) and not set(cols) & {p.name for p in PARAMS}
    for name in cols:  # experiments.DEFINITIONS explains each (the sweep viewer's help), with a readable label
        assert name in sv.definitions(), name
        assert sv.column_label(name) != name, name
    off, on = df.iloc[0], df.iloc[1]
    assert np.isnan(off["cacc_share"]) and np.isnan(off["aeb_per_veh_h"]) and np.isnan(off["harsh_auto_per_veh_h"])
    assert off["follow_hh_h"] > 0 and off["follow_aa_h"] == 0 and np.isnan(off["ttc_aa_per_h"])
    assert 0 < on["cacc_share"] <= 1 and on["follow_aa_h"] > 0 and on["follow_ah_h"] > 0
    assert on["ttc_per_h"] >= 0 and on["tet_s_per_h"] >= 0
    capsys.readouterr()


@pytest.mark.parametrize("kind", ["cav", "acc"])
def test_the_run_printout_shows_the_safety_measures_with_automated_vehicles(tmp_path, capsys, no_figures, kind):
    experiments.main(["run", "--set", "auto_share=0.5", "--set", f"auto_kind={kind}", "--duration", "40",
                      "--out", str(tmp_path / "run")])
    printed = capsys.readouterr().out
    assert "Safety layer (AEB)" in printed and "TTC <= 1.5 s, HH / HA / AH / AA" in printed
    assert ("Following by CACC" in printed) == (kind == "cav")
    assert ("CACC 0.7 s, strings of <= 8, cooperation B" in printed) == (kind == "cav")
    assert ("time gap 1.1 s, dead time 0.5 s" in printed) == (kind == "acc")
    experiments.main(["run", "--duration", "20", "--out", str(tmp_path / "plain")])
    assert "Safety layer" not in capsys.readouterr().out
