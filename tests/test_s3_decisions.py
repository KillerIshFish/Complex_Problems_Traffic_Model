"""The orchestrator's decisions on S2's open questions (SPEC 12 items 1-4, 7): flat Xiao margins with no ACC time-gap
floor (the dead-time cap kept); the safety override after the delay buffer, without the ACC dead time, renamed from
"AEB" (counter, columns, DEFINITIONS, labels, printout); brand_spread on the ACC time gaps and the ACC dead time only;
coop_class D binding every CAV pair whatever coop_between; the density control's insertions judging a CAV follower by
the time gap and margin it keeps behind the inserted vehicle (_gaps_behind)."""

from __future__ import annotations

import numpy as np
import pytest

from roundabout_sim import engine, experiments
from roundabout_sim import sweep_viewer as sv
from roundabout_sim.engine import ACC, ACC_S0, CACC_S0, CAV, CAV_PLATOON_GAP, COOP_CLASSES, COOP_PRESCRIPTIVE, HUMAN
from tests.test_controllers import controller_state, lane_of


# ------------------------------------------------------------------ flat margins, no floor
def test_the_margins_do_not_change_with_speed_and_no_floor_is_left():
    assert not hasattr(engine, "ACC_GAP_MIN") and not hasattr(engine, "CACC_S0_SLOPE")
    sim, slots = lane_of([HUMAN, CAV, CAV, ACC], [0.0, 30.0, 8.0, 30.0], [14.0, 12.5, 11.0, 13.0], auto_kind="cav")
    controller_state(sim)
    assert sim.s0[slots[1]] == ACC_S0 and sim.s0[slots[2]] == CACC_S0  # (above 10.8 and 10 m/s: as at standstill)
    acc, a = lane_of([HUMAN, ACC], [0.0, 30.0], [14.0, 12.5], auto_kind="acc", acc_headway=0.5, companies=2,
                     brand_spread=0.4, firms=[0, 0])
    assert acc.headway[a[1]] == pytest.approx(0.5 * 0.6) and acc.s0[a[1]] == ACC_S0  # (0.3 s: no floor)


# ------------------------------------------------------------------ the safety override
def test_the_override_acts_on_an_acc_car_without_its_dead_time():
    """An ACC car (dead time 0.5 s) closing on a slow car: in the first step the IDM with its own T and s0 asks for
    more than comfortable braking, the car brakes as that IDM asks at once (a_now = a_idm), while its own law's
    command still waits in the delay buffer."""
    sim, slots = lane_of([HUMAN, ACC], [0.0, 9.0], [2.0, 9.0], auto_kind="acc")
    assert sim.delay[slots[1]] == 5
    idx = np.flatnonzero(sim.alive)
    lead, gap = sim._index_lanes(idx)
    k = int(np.flatnonzero(idx == slots[1])[0])
    v = sim.v[idx]
    a_idm = sim._idm(v[k:k + 1], sim.target_speed(idx[k:k + 1]), gap[idx][k:k + 1], np.array([2.0]), idx[k:k + 1])
    assert a_idm[0] < -sim.eff.comfort_decel
    sim.step()
    assert sim.a_now[slots[1]] == pytest.approx(max(a_idm[0], -sim.eff.max_decel)) and sim.n_override == 1


def test_the_override_is_named_so_everywhere():
    assert "override_per_veh_h" in experiments.SAFETY_COLUMNS and "aeb_per_veh_h" not in experiments.SAFETY_COLUMNS
    assert "override_per_veh_h   SAFETY OVERRIDE" in experiments.DEFINITIONS and "aeb" not in experiments.DEFINITIONS
    assert sv.column_label("override_per_veh_h") == "Safety overrides (per automated veh-h)"
    assert "n_override" in engine.COUNTERS and "override_until" in engine.CAR_ARRAYS
    assert engine.RENAMED == {"n_override": ("n_aeb", 5), "override_until": ("aeb_until", 5)}
    assert not {"n_aeb", "aeb_until"} & (set(engine.COUNTERS) | set(engine.CAR_ARRAYS) | set(engine.ADDED))


# ------------------------------------------------------------------ brands
def test_the_brand_factor_scales_acc_gaps_and_dead_time_but_not_cacc_gaps():
    """Company 0 of 2 at brand_spread 0.4: factor 0.6 on an ACC car's time gap and dead time and on a CAV's
    ACC-fallback gap; a CAV's CACC time gap and the gap between strings keep their settings (x the road only)."""
    sim, s = lane_of([HUMAN, CAV, CAV, CAV], [0.0, 30.0, 8.0, 8.0], [6.0] * 4, firms=[0, 0, 0, 0], companies=2,
                     brand_spread=0.4, fallback=2.2, platoon_max=2, road_condition="wet")
    controller_state(sim)
    wet = engine.road_factors("wet")["time_headway"]
    assert sim.string_pos[s].tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(sim.headway[s[1:]], [2.2 * 0.6 * wet, 0.7 * wet, CAV_PLATOON_GAP * wet])
    acc, a = lane_of([HUMAN, ACC], [0.0, 30.0], [6.0, 6.0], auto_kind="acc", firms=[0, 0], companies=2,
                     brand_spread=0.4, acc_delay=0.5)
    assert acc.headway[a[1]] == pytest.approx(1.1 * 0.6) and acc.delay[a[1]] == 3


# ------------------------------------------------------------------ class D
def test_class_d_binds_every_cav_pair_whatever_the_company():
    """coop_between caps the class between companies below D; at D (prescribed by the infrastructure, SAE J3256)
    every CAV pair cooperates at D: strings form across companies even with coop_between none."""
    for between, cls in (("none", COOP_PRESCRIPTIVE), ("A", COOP_PRESCRIPTIVE), ("same", COOP_PRESCRIPTIVE)):
        sim, _ = lane_of([CAV, CAV], [0.0, 8.0], [6.0, 6.0], coop_class="D", coop_between=between, companies=2)
        assert sim._coop_class(CAV, np.array([0]), CAV, np.array([1]))[0] == cls, between
    sim, _ = lane_of([CAV, CAV], [0.0, 8.0], [6.0, 6.0], coop_class="C", coop_between="none", companies=2)
    assert sim._coop_class(CAV, np.array([0]), CAV, np.array([1]))[0] == 0
    assert COOP_CLASSES[COOP_PRESCRIPTIVE] == "D"
    sim, s = lane_of([HUMAN, CAV, CAV], [0.0, 30.0, 8.0], [6.0] * 3, firms=[0, 0, 1], companies=2, coop_class="D",
                     coop_between="none")
    controller_state(sim)
    assert sim.string_pos[s].tolist() == [0, 1, 2]


# ------------------------------------------------------------------ insertions
def test_an_insertion_slows_a_cav_follower_to_the_gap_it_keeps_behind_the_new_vehicle():
    """A CAV in CACC (0.7 s, 1.25 m) behind a CAV gets a driver slotted in ahead of it: it now leads its string at
    its 2.2 s ACC-fallback gap with the 2 m margin, and the density control caps its speed (by at most max_decel x
    dt, a small fill) towards the IDM steady state for THAT gap; judged by its present 0.7 s it was left alone and
    its law braked hard next step (S2's remaining CAV harsh braking at high shares)."""
    sim, s = lane_of([HUMAN, CAV, CAV], [0.0, 30.0, 40.0], [6.0] * 3, fallback=2.2, radius=16.0)  # (40 m: the
    # largest gap, none slow-free: the vehicle goes there, halfway)
    controller_state(sim)
    follower = s[2]
    assert sim.string_pos[follower] == 2 and sim.headway[follower] == 0.7
    sim._draw_kinds = lambda k, rng=None: (np.zeros(k, np.int8), np.full(k, -1, np.int8), np.full(k, np.nan))
    v_before = float(sim.v[follower])
    got, _ = sim._insert_into_gaps(1)
    assert got == 1
    idx = np.flatnonzero(sim.alive)
    lead, gap = sim._index_lanes(idx)
    new = int(lead[follower])
    assert sim.kind[new] == HUMAN and sim.inserted_at[new] == sim.time
    T, s0, pos = sim._gaps_behind(np.array([follower]), np.array([new]))
    assert (T[0], s0[0], pos[0]) == (2.2, ACC_S0, 1)
    eq = sim.equilibrium_speed(gap[[follower]], sim.desired_speed(np.array([follower])), np.array([follower]), T, s0)
    present = sim.equilibrium_speed(gap[[follower]], sim.desired_speed(np.array([follower])), np.array([follower]),
                                    np.array([0.7]), np.array([CACC_S0]))
    assert eq[0] < present[0]
    assert sim.v[follower] == pytest.approx(max(eq[0], v_before - sim.eff.max_decel * sim.cfg.dt))
    assert sim.v[follower] < v_before
