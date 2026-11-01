"""Entries by kind (SPEC 3, 11, 12): each queue head's critical gap (entry_gap x cav_entry_gap for a CAV, x
robotaxi_entry for a robotaxi; a driver's or an ACC car's driver's entry_gap); the room an automated head needs ahead
(its own IDM at its entry speed brakes no harder than comfortably); a lag vehicle judged by the time gap and margin it
would keep behind the entrant; the follow-up rule (a head of factor f is ready when entry_cool <= follow_up_steps() -
round(f x follow_up_steps()): cav_follow_up behind a cooperating CAV entrant still directly ahead, robotaxi_entry for
a robotaxi; f = 1 the human rule) with its realised spacings; route-conflict anticipation (class B: a CAV head ignores
a cooperating lag CAV that leaves before the entry point and judges the next vehicle behind) and the automated leaver
that then follows its exit, not the entrant; truck_follow_up behind an entering truck; subject_exiting's Bresenham
count; every new setting at its default changes nothing, and each acts only where it can (idle reasons)."""

from __future__ import annotations

import numpy as np
import pytest

from roundabout_sim import presets
from roundabout_sim.config import IDLE_WHY, PARAM_BY_NAME, SimConfig, inactive_inputs, inactive_reason
from roundabout_sim.engine import (ACC, ACC_S0, CACC_S0, CAV, HUMAN, ROBOTAXI, ROBOTAXI_S0_EXTRA, SUBJECT_ARM, TWO_PI,
                                   RingSim)
from tests.test_state_migration import trajectory

NEW = {"cav_entry_gap": 1.0, "cav_follow_up": 0.74, "robotaxi_entry": 1.3, "truck_follow_up": 1.0,
       "subject_exiting": 0.0}


def entry_scene(heads: dict, vehicles: list, arm: int = 0, **config) -> tuple[RingSim, np.ndarray]:
    """A one-lane roundabout (r 30 m, 4 arms, open mode, no arrivals, no noise) holding only ``vehicles``: (kind,
    centre metres from ``arm``'s entry point along the lane, + ahead, speed[, company[, ACC-fallback gap]]), none of
    them leaving; each arm in ``heads`` has one queued vehicle, (kind, company, ACC-fallback gap). Returns (sim, the
    vehicles' slots)."""
    base = dict(seed=1, n_lanes=1, n_arms=4, radius=30.0, traffic_mode="open", entry_rate=0.0, density=1,
                driver_noise=0.0, speed_std=0.0, auto_share=1.0, auto_kind="cav")
    sim = RingSim(SimConfig(**{**base, **config}))
    sim.alive[:] = False
    for name in ("queue_times", "queue_trucks", "queue_kinds", "queue_companies", "queue_acc_gaps"):
        for fifo in getattr(sim, name):
            fifo.clear()
    sim.queue[:] = 0
    r = sim.radii[0]
    kinds = np.array([v[0] for v in vehicles], dtype=np.int8)
    angles = (sim.entry_angle[arm] + np.array([v[1] for v in vehicles], dtype=float) / r) % TWO_PI
    company = np.array([v[3] if len(v) > 3 else (0 if v[0] != HUMAN else -1) for v in vehicles], dtype=np.int8)
    gap = np.array([v[4] if len(v) > 4 else 1.6 for v in vehicles])
    slots = sim._spawn(np.zeros(len(vehicles), dtype=int), angles, np.array([v[2] for v in vehicles], dtype=float),
                       kinds=(kinds, company, np.where(kinds == CAV, gap, np.nan)))
    sim.exit_at[slots] = np.inf
    for a, (kind, firm, fallback) in heads.items():
        sim.queue[a] = 1
        sim.queue_times[a].append(0.0)
        sim.queue_trucks[a].append(False)
        sim.queue_kinds[a].append(int(kind))
        sim.queue_companies[a].append(int(firm))
        sim.queue_acc_gaps[a].append(float(fallback))
    sim._index_lanes(np.flatnonzero(sim.alive))
    return sim, slots


def check(sim: RingSim, arm: int = 0) -> tuple[bool, float, int]:
    """(may the head of ``arm`` enter, at what speed, the slot it would follow) as _entries asks it."""
    arms = np.array([arm])
    ok, v_in, ahead = sim._entry_check(sim.entry_angle[arms], sim._head_trucks(arms), sim._head_s0(arms), arms)
    return bool(ok[0]), float(v_in[0]), int(ahead[0])


# ------------------------------------------------------------------ settings
def test_the_new_settings_defaults_ranges_and_tabs():
    cfg = SimConfig()
    assert {name: getattr(cfg, name) for name in NEW} == NEW
    groups = {name: PARAM_BY_NAME[name].group for name in NEW}
    assert groups == {"cav_entry_gap": "Entries", "cav_follow_up": "Entries", "robotaxi_entry": "Entries",
                      "truck_follow_up": "Vehicles", "subject_exiting": "Demand"}
    assert (PARAM_BY_NAME["robotaxi_entry"].lo, PARAM_BY_NAME["robotaxi_entry"].hi) == (0.8, 1.6)
    assert (PARAM_BY_NAME["truck_follow_up"].lo, PARAM_BY_NAME["truck_follow_up"].hi) == (1.0, 3.0)
    assert (PARAM_BY_NAME["subject_exiting"].lo, PARAM_BY_NAME["subject_exiting"].hi) == (0.0, 0.5)
    for name in NEW:
        spec = PARAM_BY_NAME[name]
        assert not spec.reset and len(spec.help) <= 340 and spec.lo <= NEW[name] <= spec.hi, name  # (+ range, reason)
        assert presets.RANDOM_RANGES[name].basis.strip(), name
    assert "Entries" in presets.OPT_IN_GROUPS
    for seed in range(10):  # (the default draws leave the Entries tab alone; the capacity test's stream is never on)
        changes = presets.random_changes(seed)
        assert not {"cav_entry_gap", "cav_follow_up", "robotaxi_entry"} & set(changes)
        assert changes.get("subject_exiting", 0.0) == 0.0
        # (truck_follow_up, on the default Vehicles tab, is never drawn either: designs with trucks stay as they were)
        assert changes.get("truck_follow_up", 1.0) == 1.0 and presets.random_changes(seed, ["Vehicles"])[
            "truck_follow_up"] == 1.0
    assert presets.RANDOM_RANGES["truck_follow_up"].choices == (1.0,)
    ranges = presets.plausible_ranges(["Vehicles"])
    design = presets.latin_hypercube(8, ranges, 3)
    assert all(point["truck_follow_up"] == 1.0 for point in design)
    assert [{k: v for k, v in point.items() if k != "truck_follow_up"} for point in design] == \
        presets.latin_hypercube(8, {k: v for k, v in ranges.items() if k != "truck_follow_up"}, 3)
    for bad in (dict(cav_entry_gap=0.0), dict(robotaxi_entry=-1.0), dict(truck_follow_up=0.0),
                dict(subject_exiting=1.5), dict(cav_follow_up=0.0)):
        with pytest.raises(ValueError):
            SimConfig(**bad)


@pytest.mark.parametrize("name, config, reason", [
    ("cav_entry_gap", dict(auto_share=0.5), ""),
    ("cav_entry_gap", dict(auto_share=0.5, n_arms=0), "no entrances"),
    ("cav_entry_gap", dict(auto_share=0.5, auto_kind="robotaxi"), "no CAVs"),
    ("cav_follow_up", dict(auto_share=0.5, coop_class="none"), "no cooperation"),
    ("cav_follow_up", dict(auto_share=0.5, coop_class="A"), ""),
    ("robotaxi_entry", dict(auto_share=0.5, auto_kind="robotaxi"), ""),
    ("robotaxi_entry", dict(), "no robotaxis"),
    ("truck_follow_up", dict(), "no trucks"),
    ("truck_follow_up", dict(truck_share=0.1, n_arms=0), "no entrances"),
    ("truck_follow_up", dict(truck_share=0.1), ""),
    ("subject_exiting", dict(), "open mode only"),
    ("subject_exiting", dict(traffic_mode="open"), "no capacity test"),
    ("subject_exiting", dict(traffic_mode="open", subject_arm="on"), ""),
    ("coop_between", dict(auto_share=0.5, companies=2, coop_class="D"), "class D"),
])
def test_each_new_setting_says_why_it_does_nothing(name, config, reason):
    assert inactive_reason(name, SimConfig(**config)) == reason
    assert reason == "" or reason in IDLE_WHY or reason.endswith("mode only")


def test_inactive_inputs_name_what_the_reasons_read():
    assert inactive_inputs("truck_follow_up") == {"truck_share", "n_arms"}
    assert inactive_inputs("subject_exiting") == {"traffic_mode", "n_arms", "subject_arm"}
    assert inactive_inputs("cav_follow_up") == {"n_arms", "auto_share", "auto_kind", "coop_class"}
    assert inactive_inputs("robotaxi_entry") == {"n_arms", "auto_share", "auto_kind"}
    assert "coop_class" in inactive_inputs("coop_between")


@pytest.mark.parametrize("config", [
    dict(traffic_mode="open", entry_rate=0.2, truck_share=0.1, n_lanes=2),
    dict(auto_share=0.5, auto_kind="acc", traffic_mode="open", entry_rate=0.2),
], ids=["human", "acc"])
def test_the_entry_settings_change_nothing_where_they_cannot_act(config):
    """Human drivers and ACC cars' drivers enter by the human rule: the CAV and robotaxi entry settings leave their
    runs bit for bit as they were; so does subject_exiting outside the capacity test."""
    base = RingSim(SimConfig(seed=3, **config))
    other = RingSim(SimConfig(seed=3, cav_entry_gap=0.7, cav_follow_up=0.5, robotaxi_entry=1.6, subject_exiting=0.4,
                              **config))
    base.step(1500)
    other.step(1500)
    assert trajectory(other) == trajectory(base) and base.n_entered > 50


# ------------------------------------------------------------------ the critical gap by kind
@pytest.mark.parametrize("head, factor, ok", [
    (HUMAN, {}, True), (ACC, {}, True), (CAV, {}, True), (ROBOTAXI, {}, False),  # lag at 2.2 s: 2.0 <= 2.2 < 2.6
    (ROBOTAXI, dict(robotaxi_entry=1.0), True), (CAV, dict(cav_entry_gap=1.2), False),
])
def test_each_heads_critical_gap(head, factor, ok):
    """entry_gap 2 s, a lag vehicle 2.2 s behind (bumper gap / its speed): a driver, an ACC car's driver and a CAV
    (cav_entry_gap 1) accept it, a robotaxi (x 1.3) does not, nor a CAV at x 1.2."""
    v = 6.0
    sim, _ = entry_scene({0: (head, 0, 1.6)}, [(HUMAN, -(2.2 * v + 4.0), v), (HUMAN, 60.0, v)],
                         auto_kind={ACC: "acc", ROBOTAXI: "robotaxi"}.get(head, "cav"), **factor)
    assert check(sim)[0] == ok


def test_a_human_head_in_an_automated_run_keeps_the_human_rule_exactly():
    """The same lag at 1.98 s and 2.02 s (entry_gap 2 s): rejected and accepted, as before automated vehicles."""
    v = 6.0
    for t, ok in ((1.98, False), (2.02, True)):
        sim, _ = entry_scene({0: (HUMAN, -1, np.nan)}, [(HUMAN, -(t * v + 4.0), v), (HUMAN, 60.0, v)])
        assert check(sim)[0] == ok, t


# ------------------------------------------------------------------ the room an automated head needs ahead
def test_an_automated_head_needs_room_its_own_idm_accepts():
    """A leader 8 m ahead (bumper) at the entry speed: a driver needs only its standstill gap + 1 m, but a CAV that
    leads its string (ACC mode, its 2.2 s fallback gap, 2 m margin) would brake at -3.2 m/s2 there (IDM, *derived*),
    more than comfortably: it waits; at 14 m (-0.3 m/s2) it goes. Behind a CAV it cooperates with it joins the string
    (0.7 s, 1.25 m): 8 m suffice."""
    v = 6.0
    a, b, v0 = 1.2, 2.0, min(12.0, np.sqrt(4.0 * 30.0))
    for front, ok in ((8.0, False), (14.0, True)):
        s_star = ACC_S0 + 2.2 * v
        idm = a * (1 - (v / v0) ** 4) - a * (s_star / front) ** 2
        assert (idm >= -b) == ok
        for head, accept in ((HUMAN, True), (CAV, ok)):
            sim, _ = entry_scene({0: (head, 0 if head == CAV else -1, 2.2)}, [(HUMAN, front + 4.0, v),
                                                                             (HUMAN, -80.0, v)])
            assert check(sim)[0] == accept, (front, head)
    sim, slots = entry_scene({0: (CAV, 0, 2.2)}, [(CAV, 12.0, v), (HUMAN, -80.0, v)])
    ok, _, ahead = check(sim)
    assert ok and ahead == slots[0]
    T, s0, pos = sim._entrant_gaps(np.array([CAV]), np.array([0]), np.array([2.2]), np.array([False]), slots[:1])
    assert (T[0], s0[0], pos[0]) == (0.7, CACC_S0, 2)


def test_the_entrants_own_time_gap_and_margin_by_kind():
    sim, slots = entry_scene({}, [(HUMAN, 20.0, 6.0)], auto_kind="robotaxi", time_headway=1.0, robotaxi_headway=1.3)
    kinds = np.array([HUMAN, ROBOTAXI, ROBOTAXI])
    T, s0, pos = sim._entrant_gaps(kinds, np.array([-1, 0, 0]), np.full(3, np.nan), np.array([False, False, True]),
                                   np.repeat(slots[:1], 3))
    np.testing.assert_allclose(T, [1.0, 1.3, 1.95])  # (a truck x 1.5)
    np.testing.assert_allclose(s0, [2.0, 2.0 + ROBOTAXI_S0_EXTRA, 2.0 + ROBOTAXI_S0_EXTRA])
    acc, a_slots = entry_scene({}, [(HUMAN, 20.0, 6.0)], auto_kind="acc", acc_headway=1.4)
    T, s0, _ = acc._entrant_gaps(np.array([ACC]), np.array([0]), np.array([np.nan]), np.array([False]), a_slots)
    assert (T[0], s0[0]) == (1.4, ACC_S0)


def test_a_cav_lag_is_judged_by_the_gap_it_would_keep_behind_the_entrant():
    """A CAV lag 13 m behind (bumper) at 8 m/s, now in CACC at 0.7 s behind a cooperating CAV far ahead: behind a
    human entrant (6 m/s) it would lead its string at its 2.2 s fallback gap and its IDM would brake at -3.5 m/s2, so
    the entrant is refused, although its present 0.7 s (-0.2 m/s2) would let it in; behind a cooperating CAV
    entrant it stays in CACC and the entrant goes."""
    v = 8.0
    sim, slots = entry_scene({0: (HUMAN, -1, np.nan)}, [(CAV, -(13.0 + 4.0), v, 0, 2.2), (CAV, 30.0, v, 0, 2.2),
                                                        (HUMAN, 80.0, v)], entry_gap=1.5)  # (a driver: no loop)
    idx = np.flatnonzero(sim.alive)
    lead, _ = sim._index_lanes(idx)
    sim._controller_gaps(idx, sim.v[idx], lead[idx])
    assert sim.headway[slots[0]] == 0.7  # (its present gap: a string member)
    T, s0 = sim._lag_gaps(slots[:1], np.array([HUMAN]), np.array([-1]), np.array([0]))
    assert (T[0], s0[0]) == (2.2, ACC_S0)
    T, s0 = sim._lag_gaps(slots[:1], np.array([CAV]), np.array([0]), np.array([2]))
    assert (T[0], s0[0]) == (0.7, CACC_S0)
    assert not check(sim)[0]
    sim.queue_kinds[0][0], sim.queue_companies[0][0], sim.queue_acc_gaps[0][0] = CAV, 0, 2.2
    assert check(sim)[0]


# ------------------------------------------------------------------ the follow-up rule
def test_the_follow_up_slack_formula():
    """follow_up 2.5 s at dt 0.1 s: follow_up_steps() 26 (2.6 s realised); a CAV behind a cooperating CAV entrant
    26 - round(0.74 x 26) = 7 steps early (19 steps: 1.9 s), a robotaxi 26 - round(1.3 x 26) = -8 (34 steps: 3.4 s),
    a CAV behind a human or a non-cooperating CAV and any human 0 (the human rule); f = 1 gives 0 exactly."""
    sim, _ = entry_scene({0: (CAV, 0, 1.6), 1: (ROBOTAXI, 0, np.nan), 2: (CAV, 0, 1.6), 3: (HUMAN, -1, np.nan)},
                         [(HUMAN, 60.0, 6.0)], follow_up=2.5, companies=2, coop_between="none")
    assert sim.follow_up_steps() == 26
    sim.last_kind[:] = [CAV, HUMAN, CAV, CAV]
    sim.last_company[:] = [0, -1, 1, 0]
    slack, behind = sim._follow_up_slack(26)
    assert slack.tolist() == [7, -8, 0, 0] and behind.tolist() == [True, False, False, False]
    assert not sim.set(cav_follow_up=1.0, robotaxi_entry=1.0)
    assert sim._follow_up_slack(26)[0].tolist() == [0, 0, 0, 0]


def spacings(kind: str, steps: int = 1500, **config) -> np.ndarray:
    """[s] Times between consecutive entries of the capacity test's entrance (us_single_lane, no conflicting flow,
    every vehicle of ``kind``)."""
    cfg = presets.apply("us_single_lane", SimConfig(seed=4)).updated(
        subject_arm="on", entry_rate=0.0, auto_share=0.0 if kind == "human" else 1.0,
        auto_kind="cav" if kind == "human" else kind, **config)
    sim = RingSim(cfg)
    sim.step(steps)
    times = np.array([t for t, arm, _ in sim.entry_log if arm == SUBJECT_ARM])
    return np.round(np.diff(times), 6)


def test_the_realised_follow_up_spacings():
    """With no conflicting flow the test entrance discharges at the follow-up: humans 2.6 s (follow_up 2.5 s + a
    step), CAVs entering behind a cooperating CAV 0.74 x 2.6 = 1.9 s (FHWA-OR-RD-22-11 Table 7.1: 1.93 s at 100 %),
    as humans at cav_follow_up 1 or without cooperation; robotaxis 1.3 x 2.6 = 3.4 s."""
    assert np.median(spacings("human")) == pytest.approx(2.6)
    cav = spacings("cav")
    assert np.median(cav) == pytest.approx(1.9) and (cav >= 1.9 - 1e-6).all()
    assert np.median(spacings("cav", cav_follow_up=1.0)) == pytest.approx(2.6)
    assert np.median(spacings("cav", coop_class="none")) == pytest.approx(2.6)
    assert np.median(spacings("robotaxi")) == pytest.approx(3.4)


def test_an_early_head_needs_its_own_steady_state_gap_unless_it_joins_a_cacc_string():
    """A robotaxi head ready before the drivers' follow-up has passed (entry_cool above 0: robotaxi_entry below 1)
    also needs s0 + T x its entry speed ahead (a leader at 4 m/s: 3.94 + 1.95 x 4 = 11.74 m): 1 m less is refused
    while early and accepted once entry_cool reaches 0 (the room rule alone); 1 m more goes at once. A CAV joining
    its cooperating last entrant's CACC string is taken in by the string's law: early or not alike."""
    v = 4.0
    for extra, early_ok in ((-1.0, False), (1.0, True)):
        sim, slots = entry_scene({0: (ROBOTAXI, 0, np.nan)}, [(HUMAN, 30.0, v), (HUMAN, -80.0, v)],
                                 auto_kind="robotaxi", robotaxi_entry=0.8, time_headway=1.5)
        T, s0, pos = sim._entrant_gaps(np.array([ROBOTAXI]), np.array([0]), np.array([np.nan]), np.array([False]),
                                       slots[:1])
        assert (T[0], s0[0], pos[0]) == (pytest.approx(1.95), pytest.approx(2.0 + ROBOTAXI_S0_EXTRA), 0)
        steady = s0[0] + T[0] * v
        r = sim.radii[0]
        sim.theta[slots[0]] = (sim.entry_angle[0] + (steady + extra + 4.0) / r) % TWO_PI  # (bumper gap steady + extra)
        sim._index_lanes(np.flatnonzero(sim.alive))
        sim.entry_cool[0] = 0
        ok, v_in, ahead = check(sim)
        assert ok and v_in == v and ahead == slots[0], extra
        sim.entry_cool[0] = 5
        assert check(sim)[0] == early_ok, extra
    for cool in (0, 5):
        sim, slots = entry_scene({0: (CAV, 0, 1.6)}, [(CAV, 3.5 + 4.0, v), (HUMAN, -80.0, v)])
        sim.entry_cool[0] = cool
        ok, _, ahead = check(sim)
        assert ok and ahead == slots[0], cool  # (3.5 m: inside the CACC steady state, 1.25 + 0.7 x 4 = 4.05 m)


def test_an_assertive_robotaxi_fleet_enters_faster_without_slowing_the_circle():
    """Capacity test, no conflicting flow: at robotaxi_entry 0.9 robotaxis enter more often than at 1 (2.6 s) and keep
    the circle's speed; without the steady-state rule their platoon locked at 3.55 m/s, entering every 3.1 s."""
    def run(factor):
        cfg = presets.apply("us_single_lane", SimConfig(seed=11)).updated(
            subject_arm="on", entry_rate=0.0, auto_share=1.0, auto_kind="robotaxi", robotaxi_entry=factor)
        sim = RingSim(cfg)
        sim.step(3000)
        times = np.array([t for t, arm, _ in sim.entry_log if arm == SUBJECT_ARM and t > 150.0])
        return np.median(np.diff(times)), float(sim.v[sim.alive].mean())

    fast, speed = run(0.9)
    plain, plain_speed = run(1.0)
    assert plain == pytest.approx(2.6) and fast < 2.5 and speed > 6.0 and plain_speed > 6.0


def test_the_cav_follow_up_needs_its_last_entrant_directly_ahead():
    """A CAV head ready 5 steps early (slack 7) behind its cooperating CAV entrant enters only while that one is the
    vehicle directly ahead of the entry point; behind another cooperating CAV (its last entrant further on) it waits
    for the human rule, although it could join that one's string."""
    v = 6.0
    for last, ok in ((0, True), (1, False)):  # (the last entrant: the CAV directly ahead, or the one beyond it)
        sim, slots = entry_scene({0: (CAV, 0, 1.6)}, [(CAV, 12.0, v), (CAV, 40.0, v), (HUMAN, -80.0, v)],
                                 follow_up=2.5)
        sim.last_kind[0], sim.last_company[0], sim.last_uid[0] = CAV, 0, sim.uid[slots[last]]
        sim.entry_cool[0] = 5
        assert check(sim)[0]  # (the gaps would do)
        n = sim.n_entered
        sim._entries()
        assert (sim.n_entered == n + 1) == ok, last
        if not ok:
            sim.entry_cool[0] = 0  # (the human rule: whoever is ahead)
            sim._entries()
            assert sim.n_entered == n + 1


# ------------------------------------------------------------------ route-conflict anticipation
@pytest.mark.parametrize("head, coop, leaves, ok", [
    (CAV, "B", True, True),  # a cooperating lag CAV leaving before the entry point: judged by the next behind it
    (CAV, "A", True, False),  # (status-sharing only: no route known)
    (CAV, "B", False, False),  # (it does not leave: a conflict)
    (HUMAN, "B", True, False),  # (a driver cannot know)
])
def test_route_conflict_anticipation(head, coop, leaves, ok):
    v = 7.0
    sim, slots = entry_scene({0: (head, 0 if head == CAV else -1, 1.6)}, [(CAV, -10.0, v), (CAV, -60.0, v)],
                             coop_class=coop)
    if leaves:  # (its route ends at this entrance's exit, which comes before the entry point)
        lag = slots[0]
        sim.exit_at[lag] = sim.travel[lag] + (sim.exit_angle[0] - sim.theta[lag]) % TWO_PI
    assert check(sim)[0] == ok


@pytest.mark.parametrize("lag_truck, head_truck, ok", [
    (False, False, True), (True, False, True), (False, True, True),  # (room behind the entrant: 4.4 / 0.4 / 0.4 m)
    (True, True, False),  # (a truck lag behind a truck entrant: the entrant would stand on it, -3.6 m)
])
def test_route_conflict_anticipation_skips_a_lag_only_if_it_turns_off_behind_the_entrant(lag_truck, head_truck, ok):
    """A cooperating lag CAV whose centre is 0.5 m before its exit (the exit-to-entry arc, 8.4 m, before the entry
    point) is ignored only if its distance to its exit is at most the bumper gap to the entrant's rear less
    SAFE_MARGIN: cars and a car with a truck leave it room, two trucks (12 m) do not, so a truck entrant then judges
    the truck lag itself (bumper gap -3.1 m: refused) instead of spawning on it."""
    def scene():
        sim, slots = entry_scene({0: (CAV, 0, 1.6)}, [(CAV, -(arc + 0.5), 7.0), (CAV, -80.0, 7.0)], truck_share=0.1)
        for s, truck in zip(slots, (lag_truck, False)):  # (the scene draws its trucks: set them)
            sim.truck[s], sim.length[s] = truck, sim.cfg.truck_length if truck else sim.cfg.car_length
        sim._set_driving(slots)
        sim.queue_trucks[0][0] = head_truck
        lag = slots[0]
        sim.exit_at[lag] = sim.travel[lag] + (sim.exit_angle[0] - sim.theta[lag]) % TWO_PI
        sim._index_lanes(np.flatnonzero(sim.alive))
        return sim, slots

    probe, _ = entry_scene({}, [(HUMAN, 60.0, 7.0)])
    arc = ((probe.entry_angle[0] - probe.exit_angle[0]) % TWO_PI) * probe.radii[0]
    assert arc == pytest.approx(8.4)
    sim, slots = scene()
    arms = np.array([0])
    _, _, _, lag, back = sim._entry_check(sim.entry_angle[arms], sim._head_trucks(arms), sim._head_s0(arms), arms,
                                          lags=True)
    assert check(sim)[0] == ok
    assert lag[0] == (slots[1] if ok else slots[0])  # (the vehicle judged: the next one behind, or the lag itself)
    if not ok:
        assert back[0] == pytest.approx(arc + 0.5 - sim.cfg.truck_length)


def test_route_conflict_anticipation_needs_cooperation_between_the_companies():
    sim, slots = entry_scene({0: (CAV, 0, 1.6)}, [(CAV, -10.0, 7.0, 1), (CAV, -60.0, 7.0, 1)], companies=2,
                             coop_between="A")
    lag = slots[0]
    sim.exit_at[lag] = sim.travel[lag] + (sim.exit_angle[0] - sim.theta[lag]) % TWO_PI
    assert not check(sim)[0]
    assert sim.set(coop_between="B") is False
    assert check(sim)[0]


class Following(RingSim):
    """The car following as before the S3 rule: every leaver follows its leader, even one beyond its exit."""

    def _leaving_first(self, idx, g):
        return g


def test_an_automated_leaver_follows_its_exit_not_a_vehicle_beyond_it():
    """_leaving_first: an automated vehicle whose exit comes before its leader's rear (a CAV 6 m behind a slow CAV
    that is past its exit, 5.6 m from the exit) follows its exit approach (gap inf for its car following and law) and
    turns off without braking for it; following that leader it calls on the safety override. A human driver in its
    place still follows the leader (the model as it was)."""
    def scene(kind, cls=RingSim):
        sim, slots = entry_scene({}, [(CAV, -4.0, 3.0), (kind, -14.0, 7.0), (HUMAN, -70.0, 7.0)])
        sim.__class__ = cls
        leaver = slots[1]
        sim.exit_at[leaver] = sim.travel[leaver] + (sim.exit_angle[0] - sim.theta[leaver]) % TWO_PI
        return sim, slots

    sim, slots = scene(CAV)
    idx = np.flatnonzero(sim.alive)
    lead, gap = sim._index_lanes(idx)
    k = int(np.flatnonzero(idx == slots[1])[0])
    assert lead[slots[1]] == slots[0] and gap[slots[1]] == pytest.approx(6.0)
    assert np.isinf(sim._leaving_first(idx, gap[idx])[k])
    human, _ = scene(HUMAN)
    assert human._leaving_first(idx, gap[idx])[k] == gap[slots[1]]
    sim.step(30)
    assert not sim.alive[slots[1]] and sim.n_override == 0 and sim.n_emergency == 0
    before, _ = scene(CAV, Following)
    before.step(30)
    assert before.n_override > 0


# ------------------------------------------------------------------ trucks and the capacity test's exiting stream
class EntryLog(RingSim):
    """Records (time, entrance, truck) of every entry (_entries spawns its entrants with ``lat``)."""

    def _spawn(self, lanes, angles, speeds, lat=None, trucks=None, *, kinds=None):
        slots = super()._spawn(lanes, angles, speeds, lat, trucks, kinds=kinds)
        if lat is not None:
            arm = int(np.argmin(np.abs((self.entry_angle - self.theta[slots[0]] + np.pi) % TWO_PI - np.pi)))
            self.entries.append((self.time, arm, bool(self.truck[slots[0]])))
        return slots


def test_the_queue_waits_truck_follow_up_times_the_follow_up_behind_an_entering_truck():
    """After a truck enters, its entrance's next vehicle may go round(truck_follow_up x follow_up_steps()) steps
    later (2.5 s at dt 0.1: 26 steps, 2.6 s; x2 -> 52 steps, 5.2 s); 1 = as behind a car. In the capacity test with no
    conflicting flow the realised spacings are 5.2 s behind a truck and 2.6 s behind a car."""
    for tff, cool in ((2.0, 52), (1.0, 26), (1.5, 39)):
        sim, _ = entry_scene({0: (HUMAN, -1, np.nan)}, [(HUMAN, -80.0, 6.0)], follow_up=2.5, truck_share=0.1,
                             truck_follow_up=tff, auto_share=0.0)
        sim.queue_trucks[0][0] = True
        sim._entries()
        assert (sim.n_entered, int(sim.arm_entered_trucks[0]), int(sim.entry_cool[0])) == (1, 1, cool), tff
    cfg = presets.apply("us_single_lane", SimConfig(seed=5)).updated(subject_arm="on", entry_rate=0.0,
                                                                      truck_share=0.2, truck_follow_up=2.0)
    sim = EntryLog(cfg)
    sim.entries = []
    sim.step(3000)
    test = [(t, truck) for t, arm, truck in sim.entries if arm == SUBJECT_ARM]
    after = {True: [], False: []}
    for (t0, truck), (t1, _) in zip(test[:-1], test[1:]):
        after[truck].append(round(t1 - t0, 6))
    assert len(after[True]) >= 5 and min(after[True]) == pytest.approx(5.2) and min(after[False]) == pytest.approx(2.6)


def test_subject_exiting_is_a_bresenham_count_of_the_feeding_entrants():
    """Entrant n of the feeding entrances leaves at the test entrance's own exit iff floor((n + 1) s) > floor(n s):
    exactly s of them, evenly spread, no draws; they never pass its entry point."""
    sim = RingSim(presets.apply("us_single_lane", SimConfig(seed=6)).updated(subject_arm="on", entry_rate=0.0))
    s = 0.25
    sim.set(subject_exiting=s)
    slot = sim._spawn(np.zeros(1, dtype=int), sim.entry_angle[[2]], np.array([6.0]))[0]
    routed = []
    for n in range(12):
        sim.exit_at[slot] = -1.0
        sim._route_feeding(slot)
        routed.append(sim.exit_at[slot] != -1.0)
    assert routed == [np.floor((n + 1) * s) > np.floor(n * s) for n in range(12)] and sum(routed) == 3
    assert sim.n_feeding == 12
    ahead = sim.exit_at[slot] - sim.travel[slot]
    assert routed[-1] and ahead == pytest.approx((sim.exit_angle[0] - sim.entry_angle[2]) % TWO_PI)
    # a run: the routed share leaves at the test entrance's own exit and never passes its entry point; the others
    # (and the cars placed at the start, routed past it) pass it once (those still on the ring: not yet)
    for s in (0.5, 0.0):
        run = RingSim(presets.apply("us_single_lane", SimConfig(seed=7)).updated(
            subject_arm="on", entry_rate=0.06, subject_exiting=s))
        start = int(run.alive.sum())
        run.step(6000)
        fed = int(run.arm_entered[1:].sum())
        routed = int(np.floor(fed * s))  # (the count's exact share)
        exits_at_0 = sum(1 for _, arm, _ in run.exit_log if arm == SUBJECT_ARM)
        passed, now = int(run.arm_passed[SUBJECT_ARM]), int(run.alive.sum())
        assert run.n_feeding == fed > 50
        assert routed - now <= exits_at_0 <= routed
        assert fed - routed - now <= passed <= fed - routed + start
