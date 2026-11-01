"""Fixes after the S2 review, as the S3 decisions left them: the ACC law's guards (the dead time capped at
ACC_DEAD_MAX, so a lone follower stays locally stable; the time-gap floor went with Xiao's speed-dependent margin,
now flat) and its behaviour at 8 and 12 m/s; MOBIL judging a CAV by the time gap and margin it would
keep behind its new leader (_gaps_behind); CACC string cuts kept while a vehicle joins or leaves the chain ahead
(merging when both strings fit in one), which spares CAVs harsh braking at high shares; a live switch of smoother
law engaging the new law from the car's speed;
the observers (a smoother counts as its driver's car until Smoother start; low TTC at any distance; cut-ins only
within the reaction distance; pair_type, follower first); the cooperation class needing two CAVs; companies_differ
by class rank; and the printouts, help texts, labels and notes that changed."""

from __future__ import annotations

import numpy as np

from roundabout_sim import cli, config, engine, experiments
from roundabout_sim import sweep_viewer as sv
from roundabout_sim.config import PARAM_BY_NAME, SimConfig, companies_differ, range_text
from roundabout_sim.engine import (ACC, ACC_DEAD_MAX, ACC_K1, ACC_K2, CAV,
                                   CAV_PLATOON_GAP, COOP_CACC, COOP_INTENT, HUMAN, PAIR_TYPES, ROBOTAXI, SMOOTHER,
                                   TTC_CRITICAL, RingSim, pair_type)
from tests.helpers import frames, show_tab
from tests.test_controllers import controller_state, lane_of, platoon
from tests.test_safety_measures import observe
from tests.test_smoothers import STERN, smoothers_of


# ------------------------------------------------------------------ the ACC law's guards
def acc_radius(T_eff: float, dead: float, dt: float = 0.1) -> float:
    """Spectral radius of the linearised loop of an ACC follower behind a steady leader as the engine steps it: the
    whole command ACC_K1 gap - (ACC_K1 T_eff + ACC_K2) v delayed by ``dead`` s in the buffer, v' = v + a dt, gap' =
    gap - v dt - a dt^2 / 2 (T_eff: the time gap plus the margin's slope, Xiao 2017 Eq. 6)."""
    d = int(np.rint(dead / dt))
    n = 2 + d
    row = np.zeros(n)
    row[0], row[1] = ACC_K1, -(ACC_K1 * T_eff + ACC_K2)
    a = row if d == 0 else np.eye(n)[n - 1]  # (the command applied: computed d steps ago)
    M = np.zeros((n, n))
    M[0] = -dt * np.eye(n)[1] - 0.5 * dt * dt * a
    M[0, 0] += 1
    M[1] = np.eye(n)[1] + dt * a
    if d:
        M[2] = row
        for j in range(1, d):
            M[2 + j, 1 + j] = 1.0
    return float(max(abs(np.linalg.eigvals(M))))


def test_the_acc_dead_time_is_capped_and_no_time_gap_floor_is_left():
    """Changed on purpose (flat margins, the S2 decision): dead time = ACC dead time x brand, at most ACC_DEAD_MAX
    (also beyond the slider, --set acc_delay=1); the ACC law's time gap (ACC time gap, a CAV's fallback gap) x brand x
    road with no floor (ACC_GAP_MIN went with Xiao's speed term); a CACC time gap takes no brand factor."""
    assert ACC_DEAD_MAX == 0.6 and not hasattr(engine, "ACC_GAP_MIN")
    assert PARAM_BY_NAME["acc_delay"].hi == ACC_DEAD_MAX and PARAM_BY_NAME["acc_headway"].lo == 0.5
    long = RingSim(SimConfig(seed=1, auto_share=1.0, auto_kind="acc", acc_delay=1.0, density=30))
    assert (long.delay[long.alive] == 6).all()
    brands = SimConfig(seed=1, auto_share=1.0, auto_kind="acc", companies=2, brand_spread=0.6, density=40,
                       acc_headway=1.1, acc_delay=0.5)
    sim = RingSim(brands)
    live = np.flatnonzero(sim.alive & ~sim.truck)
    fast, slow = live[sim.company[live] == 0], live[sim.company[live] == 1]  # brand factors 0.4 and 1.6
    assert np.allclose(sim.headway[fast], 1.1 * 0.4) and np.allclose(sim.headway[slow], 1.1 * 1.6)  # (0.44 s: kept)
    assert (sim.delay[fast] == 2).all() and (sim.delay[slow] == 6).all()  # (0.8 s: capped)
    cav = RingSim(brands.updated(auto_kind="cav", cav_headway=0.5))
    c = np.flatnonzero(cav.alive & ~cav.truck & (cav.company == 0))
    np.testing.assert_allclose(cav.headway[c], 0.4 * cav.acc_gap[c])
    sim, slots = lane_of([CAV, CAV], [0.0, 3.0], [5.0, 5.0], firms=[0, 0], companies=2, brand_spread=0.6,
                         cav_headway=0.5)
    controller_state(sim)
    assert sim.string_pos[slots[1]] == 2 and sim.headway[slots[1]] == 0.5  # CACC: no brand factor, no floor


def test_the_delayed_acc_law_is_locally_stable_up_to_the_cap():
    """*derived* (ACC_DEAD_MAX, flat margin): up to 0.6 s of dead time a lone ACC follower settles behind a steady
    leader at every slider time gap (0.5-2.4 s) and brand factor (0.4-1.6), with no floor; without the cap the
    shortest time gaps of a slow brand oscillate (0.5 s x 1.6 with 0.6 s x 1.6)."""
    for brand in np.arange(0.4, 1.61, 0.1):
        for T in (0.5, 0.7, 1.1, 1.8, 2.4):
            for dead in np.arange(0.0, 0.61, 0.1):
                assert acc_radius(T * brand, min(dead * brand, ACC_DEAD_MAX)) < 1, (brand, T, dead)
    assert acc_radius(0.5 * 1.6, 0.6 * 1.6) > 1  # (why the cap)
    assert acc_radius(0.6, 0.8) > 1 and acc_radius(1.1, 1.2) > 1 and acc_radius(1.1, 1.1) < 1


def test_one_acc_car_settles_behind_a_steady_leader_at_12_ms_even_set_to_a_long_dead_time():
    """At 12 m/s, set to a dead time of 1.0 s, a lone follower gets the 0.6 s cap and settles from a 0.5 m offset."""
    v0 = 12.0
    gap = float(RingSim.acc_margin(v0)) + 1.1 * v0 + 0.5
    sim, slots = lane_of([ACC, ACC], [0.0, gap], [v0, v0], auto_kind="acc", mean_speed=25.0, acc_delay=1.0)
    assert sim.delay[slots[1]] == 6
    lead, pos, vs = slots[0], 0.0, []
    while sim.time < 250.0:
        sim.step()
        pos += v0 * sim.cfg.dt
        sim.theta[lead], sim.v[lead] = (pos / sim.radii[0]) % (2 * np.pi), v0
        vs.append(sim.v[slots[1]])
    swings = [np.ptp(vs[k:k + 500]) for k in range(0, 2500, 500)]  # (each 50 s)
    assert swings[0] > 0.1 and (np.diff(swings) < 0).all() and swings[-1] < 0.01 and sim.n_override == 0, swings


def test_with_flat_margins_an_acc_string_amplifies_alike_at_12_and_8_ms_and_cacc_damps():
    """Changed on purpose (flat margins): at 12 m/s, once Xiao's band, the ACC margin no longer falls with speed, so
    the first follower of an ACC platoon amplifies a leader's dip as at 8 m/s (the law alone, with and without its
    dead time; it amplified 0.08-0.11 more there before); a CACC string damps the dip at 12 m/s."""
    for cfg in (dict(acc_delay=0.0), dict()):
        flat, _ = platoon("acc", v0=8.0, law_only=True, **cfg)
        band, _ = platoon("acc", v0=12.0, law_only=True, **cfg)
        assert abs(band[1] / band[0] - flat[1] / flat[0]) < 0.03 and flat[1] / flat[0] > 1.1, (cfg, flat, band)
    dips, sim = platoon("cav", v0=12.0)
    assert (np.diff(dips) < 0).all() and dips[-1] < 0.98 * dips[0] and sim.n_override == 0


# ------------------------------------------------------------------ MOBIL by the gaps after the change
def test_gaps_behind_gives_the_time_gap_margin_and_string_position_after_a_change():
    """A CAV behind a cooperating CAV joins its string (the next position; past platoon_max the one too many at
    CAV_PLATOON_GAP, then 2 again) at the CACC time gap with the CACC margin; behind anything else, or a CAV of
    a company it does not cooperate with, it leads at its fallback gap with the ACC margin. Others keep theirs."""
    sim, s = lane_of([HUMAN, CAV, CAV, CAV, HUMAN], [0.0, 10.0, 6.0, 6.0, 10.0], [6.0] * 5, fallback=1.6,
                     platoon_max=3)
    controller_state(sim)
    assert sim.string_pos[s].tolist() == [0, 1, 2, 3, 0]
    T, s0, pos = sim._gaps_behind(s[[3, 1, 2, 1, 4, 1]], np.array([s[0], s[3], s[1], s[2], s[3], -1]))
    assert pos.tolist() == [1, 4, 2, 3, 0, 1]  # (behind member 3 = platoon_max: the one too many)
    np.testing.assert_allclose(T, [1.6, CAV_PLATOON_GAP, 0.7, 0.7, sim.headway[s[4]], 1.6])
    np.testing.assert_allclose(s0, [2.0, 1.25, 1.25, 1.25, sim.s0[s[4]], 2.0])  # (the CACC margin, flat)
    T, _, pos = sim._gaps_behind(s[[1, 1]], s[[2, 2]], leader_pos=np.array([4, 2]))  # (prospective positions)
    assert pos.tolist() == [2, 3] and T.tolist() == [0.7, 0.7]
    split, t = lane_of([HUMAN, CAV, CAV], [0.0, 10.0, 6.0], [6.0] * 3, firms=[0, 0, 1], companies=2,
                       coop_between="none", fallback=2.2)
    controller_state(split)
    T, s0, pos = split._gaps_behind(t[[2]], t[[1]])
    assert pos.tolist() == [1] and T.tolist() == [2.2] and s0.tolist() == [2.0]
    for kind in ("acc", "robotaxi"):  # (no CAVs: every car keeps its own, as before)
        other, o = lane_of([HUMAN, ACC if kind == "acc" else ROBOTAXI], [0.0, 10.0], [6.0] * 2, auto_kind=kind)
        assert other._gaps_behind(o[[1]], o[[0]]) == (None, None, None)


class Present(RingSim):
    """MOBIL as before: every car judged by its present time gap and margin."""

    def _gaps_behind(self, cars, leaders, leader_pos=None):
        return None, None, None


def test_mobil_by_the_gaps_after_the_change_spares_cavs_harsh_braking():
    """A CACC member judged at its 0.7 s gap changed lanes (or let a driver cut in ahead) and was then demoted to
    its 1.1-2.2 s fallback gap, which the safety layer's IDM enforced by braking hard: most harsh braking of CAVs
    on the default roundabout came from it (seeds 1-3, 200 s, half CAVs: 35 onsets before, 5 now)."""
    before = now = 0
    for seed in (1, 2, 3):
        old, new = Present(seed=seed, auto_kind="cav", auto_share=0.5), RingSim(seed=seed, auto_kind="cav", auto_share=0.5)
        old.step(2000)
        new.step(2000)
        before += old.n_emergency_auto
        now += new.n_emergency_auto
    assert 3 * now < before, (now, before)


# ------------------------------------------------------------------ string cuts kept
def insert_cav(sim, ahead, behind, speed=6.0) -> int:
    """A cooperating CAV (company 0, fallback 1.6 s) spawned midway between the vehicles in slots ``ahead`` and
    ``behind`` of a lane_of lane (a vehicle joining the chain there). Returns its slot."""
    mid = 0.5 * (sim.theta[ahead] + sim.theta[behind])
    new = sim._spawn(np.zeros(1, dtype=int), np.array([mid]), np.array([speed]),
                     kinds=(np.array([CAV], dtype=np.int8), np.array([0], dtype=np.int8), np.array([1.6])))
    sim.exit_at[new] = np.inf
    return int(new[0])


def test_a_cut_keeps_its_head_when_a_vehicle_ahead_leaves_the_chain():
    """A vehicle leaving the chain ahead of a cut (an exit, a lane change) re-counts the cars up to the cut only:
    the CAV at platoon_max + 1 keeps leading its string, so no member behind it changes its time gap. Counted
    afresh from the chain's leader, the cut moved one car back and the member at 0.7 s behind it had to hold 1.5 s
    at once."""
    n = 12
    sim, s = lane_of([HUMAN] + [CAV] * n, [0.0] + [14.0] * n, [6.0] * (n + 1), fallback=1.1)
    controller_state(sim)
    assert sim.string_pos[s].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4]
    sim.alive[s[3]] = False  # (member 3 leaves the lane)
    controller_state(sim)
    rest = np.delete(s, 3)
    assert sim.string_pos[rest].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 9, 2, 3, 4]
    assert sim.headway[s[9]] == CAV_PLATOON_GAP and sim.headway[s[10]] == 0.7
    afresh, _ = lane_of([HUMAN] + [CAV] * n, [0.0] + [14.0] * n, [6.0] * (n + 1), fallback=1.1)
    afresh.alive[s[3]] = False  # (the same lane with no cut to keep: the count from the chain's leader)
    controller_state(afresh)
    assert afresh.string_pos[rest].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3]
    # The cut merges into the string ahead once both fit in one (the cars ahead up to its leader + its own <= 8).
    sim.alive[s[[4, 5]]] = False  # (5 ahead + 4 of its own: still cut)
    controller_state(sim)
    assert sim.string_pos[s[[1, 2, 6, 7, 8, 9, 10, 11, 12]]].tolist() == [1, 2, 3, 4, 5, 9, 2, 3, 4]
    sim.alive[s[6]] = False  # (4 + 4: one string)
    controller_state(sim)
    assert sim.string_pos[s[[1, 2, 7, 8, 9, 10, 11, 12]]].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert sim.headway[s[9]] == 0.7


def test_a_string_growing_past_platoon_max_is_still_cut_and_merges_go_one_at_a_time():
    """A vehicle joining a full string ahead of a cut makes the member it pushes to platoon_max + 1 the new cut, and
    the old cut, now behind a string of one, merges into it (only those two change their time gaps). Of two cuts
    that could merge at once, the one behind waits a step (one merge per chain of strings per step); a loop of
    CAVs keeps its head, which never merges into its own loop."""
    n = 12
    sim, s = lane_of([HUMAN] + [CAV] * n, [0.0] + [14.0] * n, [6.0] * (n + 1), fallback=1.1)
    controller_state(sim)
    new = insert_cav(sim, s[2], s[3])
    controller_state(sim)
    order = np.insert(s, 3, new)
    assert sim.string_pos[order].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
    assert sim.headway[s[8]] == CAV_PLATOON_GAP and sim.headway[s[9]] == 0.7
    two, t = lane_of([HUMAN] + [CAV] * 6, [0.0] + [14.0] * 6, [6.0] * 7)
    two.string_pos[t] = [0, 1, 2, 9, 2, 9, 2]  # (two cuts, as left by earlier joins)
    controller_state(two)
    assert two.string_pos[t].tolist() == [0, 1, 2, 3, 4, 9, 2]
    controller_state(two)
    assert two.string_pos[t].tolist() == [0, 1, 2, 3, 4, 5, 6]
    for k in (10, 3):
        loop, u = lane_of([CAV] * k, [0.0] + [8.0] * (k - 1), [6.0] * k)
        controller_state(loop)
        first = loop.string_pos[u].tolist()
        controller_state(loop)
        assert loop.string_pos[u].tolist() == first == ([9, 2, 3, 4, 5, 6, 7, 8, 9, 2] if k == 10 else [9, 2, 3])


def test_mobil_judges_a_cut_at_the_gap_it_keeps():
    """_gaps_behind: a CAV at platoon_max + 1 behind a cooperating CAV keeps leading its string at CAV_PLATOON_GAP
    (whether it would merge is left out: the longer gap); a member joins after its new leader."""
    n = 12
    sim, s = lane_of([HUMAN] + [CAV] * n, [0.0] + [14.0] * n, [6.0] * (n + 1), fallback=1.1)
    controller_state(sim)
    T, _, pos = sim._gaps_behind(s[[9, 10]], s[[3, 3]])
    assert pos.tolist() == [9, 4] and T.tolist() == [CAV_PLATOON_GAP, 0.7]  # (behind member 3)


class Afresh(RingSim):
    """String positions counted afresh from each chain's leader every step (no cut kept), as before."""

    def _string_positions(self, idx, ld, kind):
        self.string_pos[idx] = 0
        return super()._string_positions(idx, ld, kind)


def test_keeping_the_cuts_spares_cavs_harsh_braking_at_high_shares():
    """With every position counted afresh, each vehicle joining or leaving a chain moved every cut behind it, and
    the member just behind a cut braked hard to open 0.7 s to 1.5 s: most harsh braking of CAVs at high shares
    (100 % CAVs on two lanes at density 80, seeds 1-2, 200 s: 30 onsets counted afresh, 2 with the cuts kept)."""
    before = now = 0
    for seed in (1, 2):
        cfg = SimConfig(seed=seed, auto_kind="cav", auto_share=1.0, n_lanes=2, density=80)
        old, new = Afresh(cfg), RingSim(cfg)
        old.step(2000)
        new.step(2000)
        before += old.n_emergency_auto
        now += new.n_emergency_auto
    assert 4 * now < before, (now, before)


# ------------------------------------------------------------------ smoothers
def test_a_live_switch_of_law_engages_the_new_law_from_the_cars_speed(monkeypatch):
    """FollowerStopper -> PI live: PI's first update starts from the car's speed (smoother_on cleared), not from
    FollowerStopper's last command; a change of anything else keeps the command."""
    sim = RingSim(SimConfig(**{**STERN, "seed": 101, "smoothers": 1}))
    sim.step(600)
    slot = smoothers_of(sim)[0]
    assert sim.smoother_on[slot] and not sim.set(smoother_speed=7.0) and sim.smoother_on[slot]
    assert not sim.set(smoother_law="pi") and not sim.smoother_on.any()
    seen = []
    pi = RingSim.pi_saturation
    monkeypatch.setattr(sim, "pi_saturation", lambda dx, v, vl, U, last: seen.append((v.copy(), last.copy()))
                        or pi(dx, v, vl, U, last))
    sim.step()
    assert len(seen) == 1 and seen[0][1].tolist() == seen[0][0].tolist() and sim.smoother_on[slot]


def test_a_smoother_counts_as_its_drivers_car_until_its_start():
    """Before Smoother start its driver drives it: human exposure, H in the pair types, no automated harsh braking;
    from the start on it is an automated vehicle seen as an AV (A)."""
    sim = RingSim(SimConfig(**{**STERN, "seed": 102, "smoothers": 1, "smoother_start": 30.0}))
    sim.step(299)
    assert sim.n_auto_steps == 0 and sim.n_emergency_auto == 0 and sim.pair_follow_steps[1:].sum() == 0
    sim.step(301)
    assert 299 <= sim.n_auto_steps <= 301 and sim.pair_follow_steps[1] > 0 and sim.pair_follow_steps[2] > 0
    assert sim.n_auto_steps + sim.n_human_steps == 600 * int(sim.alive.sum())


# ------------------------------------------------------------------ observers
def test_low_ttc_counts_at_any_distance():
    """With a 10 m reaction distance, a car 12 m behind a slower leader closing at 10 m/s (TTC 1.2 s) is a near
    conflict all the same: a low-TTC step, an episode and a step of following exposure."""
    sim, s = lane_of([HUMAN] * 3, [0.0, 12.0, 30.0], [4.0, 14.0, 14.0], auto_kind="acc", auto_share=0.0,
                     reaction_distance=10.0)
    observe(sim)
    assert sim.pair_ttc_steps.tolist() == [1, 0, 0, 0] and sim.pair_ttc_episodes.tolist() == [1, 0, 0, 0]
    assert sim.pair_follow_steps.tolist() == [1, 0, 0, 0]  # (the third, 30 m behind at equal speed: neither)
    assert TTC_CRITICAL == 1.5


def test_pair_type_names_the_follower_first():
    assert PAIR_TYPES[pair_type(CAV, HUMAN)] == "AH" and PAIR_TYPES[pair_type(HUMAN, ROBOTAXI)] == "HA"
    assert PAIR_TYPES[pair_type(ACC, SMOOTHER)] == "HA" and PAIR_TYPES[pair_type(ROBOTAXI, CAV)] == "AA"
    np.testing.assert_array_equal(pair_type(np.array([HUMAN, CAV, ACC]), np.array([HUMAN, HUMAN, CAV])), [0, 2, 1])


# ------------------------------------------------------------------ cooperation
def test_the_cooperation_class_needs_two_cavs():
    """A human driver's company is -1 like another's: the class tests the kinds (CONNECTED = CAVs only)."""
    sim = RingSim(SimConfig(seed=1, auto_share=0.5, auto_kind="cav", companies=2, coop_class="D"))
    assert sim._coop_class(HUMAN, -1, HUMAN, -1) == 0 and sim._coop_class(CAV, 0, HUMAN, -1) == 0
    assert sim._coop_class(CAV, 0, CAV, 0) == 4 and sim._coop_class(ACC, 0, ACC, 0) == 0
    assert (COOP_CACC, COOP_INTENT) == (1, 2)
    np.testing.assert_array_equal(sim._coop_class(np.array([CAV, HUMAN]), np.array([0, -1]), CAV, np.array([1, -1])),
                                  [4, 0])


def test_companies_differ_by_the_classes_that_act(monkeypatch):
    """Companies matter where CAVs of different companies cooperate less than within one, counting classes up to
    COOP_ACTING (B since route-conflict anticipation: C and D act as B), and never at D (prescribed for every pair,
    whatever coop_between: changed on purpose, the S2 decision); once a higher class acts, (C, B) differs too."""
    assert config.COOP_ACTING == "B"
    cav = SimConfig(auto_share=0.5, auto_kind="cav", companies=2)
    cases = {("B", "A"): True, ("C", "B"): False, ("D", "A"): False, ("D", "none"): False, ("A", "none"): True,
             ("B", "same"): False, ("none", "none"): False, ("C", "A"): True}
    for (within, between), differ in cases.items():
        assert companies_differ(cav.updated(coop_class=within, coop_between=between)) == differ, (within, between)
    assert companies_differ(cav.updated(auto_kind="acc", brand_spread=0.2))
    monkeypatch.setattr(config, "COOP_ACTING", "D")
    assert companies_differ(cav.updated(coop_class="C", coop_between="B"))
    assert not companies_differ(cav.updated(coop_class="D", coop_between="B"))
    assert not companies_differ(cav.updated(coop_class="B", coop_between="same"))


# ------------------------------------------------------------------ texts and printouts
def test_a_smoother_run_prints_the_safety_measures_and_the_start_note(tmp_path, capsys, no_figures):
    experiments.main(["run", "--preset", "stern2018", "--set", "smoothers=1", "--set", "smoother_start=30",
                      "--duration", "40", "--out", str(tmp_path / "sm")])
    printed = capsys.readouterr().out
    assert "Safety override" in printed and f"TTC <= {TTC_CRITICAL:g} s, HH / HA / AH / AA" in printed
    assert "Automated on the ring" not in printed  # (the auto_share kinds' rows)
    assert "smoothers switch on (smoother_start 30 s) after the measured window starts at 20 s" in printed
    experiments.main(["sweep", "--preset", "stern2018", "--x", "smoother_start=0,100", "--set", "smoothers=1",
                      "--seeds", "1", "--warmup", "10", "--measure", "10", "--workers", "1", "--out",
                      str(tmp_path / "sw")])
    printed = capsys.readouterr().out
    assert "1 of the 2 runs' smoothers switch on (smoother_start 100 s)" in printed and "--warmup of 100 s" in printed


def test_help_texts_labels_and_ranges():
    sample = next(a for a in experiments.build_parser()._subparsers._group_actions[0].choices["sweep"]._actions
                  if a.dest == "sample")
    assert "AVs, Coop, Entries and Smoothers tabs'" in sample.help
    assert range_text(PARAM_BY_NAME["smoother_lane"]) == "-1 to 4" and range_text(PARAM_BY_NAME["n_lanes"]) == "1-5"
    assert cli.range_note("smoother_lane", 6) == "note: smoother_lane=6 is outside the usual range -1 to 4"
    assert sv.column_label("ttc_ah_per_h") == f"TTC ≤ {TTC_CRITICAL:g} s episodes, AH (per h following)"
    assert "enter by their kind's rules (Entries tab)" in PARAM_BY_NAME["auto_share"].help
    assert PARAM_BY_NAME["platoon_max"].group == "Coop"


def test_the_avs_tab_fits_at_1280_by_720(make_app):
    app = make_app(auto_share=0.5, auto_kind="cav")
    frames(app, 2)
    for tab in ("AVs", "Coop", "Entries", "Smoothers"):
        show_tab(app, tab)
        frames(app, 1)
        assert app.content_h <= app.content_rect.h, tab
