"""The ACC and CACC controllers (SPEC 2, 4): the PATH ACC law (Milanes & Shladover 2014 Eq. 5) with its speed mode
(Xiao et al. 2017 Eq. 2), the flat spacing margins (Xiao 2017 Eqs. 6-7 at low speed), PATH CACC (FHWA-OR-RD-22-11
Eqs. 7-3 to 7-6) behind a cooperating CAV and the ACC law at the fallback gap otherwise, against hand-computed
values; the controllers' settings, dead time and brand factors; CACC gating by cooperation class, company and
coop_between; string positions with the platoon_max cut (the CAV that would be member platoon_max + 1 keeps 1.5 s)
and loops; the clip and the safety override with its counter; the CACC law at time steps longer than its update
period; and the
controllers' behaviour in a platoon (an ACC platoon amplifies a leader's speed dip, a CACC string attenuates it)."""

from __future__ import annotations

import numpy as np
import pytest

from roundabout_sim.config import SimConfig
from roundabout_sim.engine import (ACC, ACC_DEAD_MAX, ACC_K1, ACC_K2, ACC_S0, ACC_SPEED_GAIN, CACC_GAIN_GAP,
                                   CACC_KD, CACC_KP, CACC_PERIOD, CACC_S0, CAV, CAV_PLATOON_GAP, COOP_CLASSES, HUMAN,
                                   ROBOTAXI, TRUCK_HEADWAY_FACTOR, RingSim, _chain_depth, road_factors)

LANE = dict(n_lanes=1, n_arms=0, traffic_mode="open", density=1, driver_noise=0.0, speed_std=0.0)


def lane_of(kinds, gaps, speeds, firms=None, fallback=1.6, radius=1000.0, auto_kind="cav", **config) -> tuple:
    """A single-lane ring (no entrances; open mode, so no density control) holding only these vehicles, in
    driving order from the front: ``gaps[i]`` is vehicle i's bumper gap to vehicle i - 1 (the first's is
    ignored), ``speeds`` their speeds, ``firms`` their companies (default 0 for automated vehicles), every CAV's
    ACC-fallback gap ``fallback``. Returns (sim, slots in that order)."""
    sim = RingSim(SimConfig(**{"seed": 1, "auto_share": 1.0, "auto_kind": auto_kind, "radius": radius, **LANE,
                               **config}))
    sim.alive[:] = False
    kinds = np.asarray(kinds, dtype=np.int8)
    n = kinds.size
    companies = np.where(kinds == HUMAN, -1, 0 if firms is None else np.asarray(firms)).astype(np.int8)
    s = -np.concatenate(([0.0], np.cumsum(np.asarray(gaps, dtype=float)[1:] + sim.cfg.car_length)))
    slots = sim._spawn(np.zeros(n, dtype=int), (s / sim.radii[0]) % (2 * np.pi), np.asarray(speeds, dtype=float),
                       kinds=(kinds, companies, np.where(kinds == CAV, fallback, np.nan)))
    sim.exit_at[slots] = np.inf
    return sim, slots


def controller_state(sim):
    """(idx, v, lead slots, gaps) as _move sees them, after _controller_gaps has set the margins and time gaps."""
    idx = np.flatnonzero(sim.alive)
    lead, gap = sim._index_lanes(idx)
    v = sim.v[idx]
    sim._controller_gaps(idx, v, lead[idx])
    return idx, v, lead[idx], gap[idx]


# ------------------------------------------------------------------ the spacing margins
def test_the_margins_are_flat_at_xiaos_low_speed_values():
    """Changed on purpose (the S2 decision on Xiao's margins): ACC 2.0 m (Xiao 2017 Eq. 6's d0 = 7 m for v <= 10.8 m/s
    less the 5 m vehicle length) and CACC 1.25 m (Eq. 7 at v = 0) at every speed: taken at the car's own speed the
    full margins acted as time-gap terms (the ACC one made 3.9 % of the reachable settings locally unstable, the CACC
    one a 0.7 s string amplify below 10 m/s)."""
    speeds = np.array([0.0, 5.0, 10.8, 10.8001, 12.0, 15.0, 20.0, 30.0])
    np.testing.assert_array_equal(RingSim.acc_margin(speeds), np.full(speeds.size, 2.0))
    np.testing.assert_array_equal(RingSim.cacc_margin(speeds), np.full(speeds.size, 1.25))
    assert ACC_S0 == 2.0 and CACC_S0 == 1.25


# ------------------------------------------------------------------ the laws
def test_the_acc_law_and_its_speed_mode_against_hand_values():
    """a = min(0.23 (gap - s0 - T v) + 0.07 (v_lead - v), 0.4 (v0 - v)): the gap mode close behind a leader, the
    speed mode far behind it (the switch the sources make at 100-120 m, here where the two cross)."""
    assert (ACC_K1, ACC_K2, ACC_SPEED_GAIN) == (0.23, 0.07, 0.4)
    sim, slots = lane_of([ACC, ACC], [0.0, 15.0], [7.0, 8.0], auto_kind="acc")
    idx, v, ld, g = controller_state(sim)
    follower = int(np.flatnonzero(idx == slots[1])[0])
    assert sim.s0[slots[1]] == 2.0 and sim.headway[slots[1]] == 1.1  # (8 m/s: the margin's 2 m; ACC time gap)
    one = np.array([follower])
    gap_mode = 0.23 * (15.0 - 2.0 - 1.1 * 8.0) + 0.07 * (7.0 - 8.0)  # 0.896
    a = sim._law_acc(idx[one], v[one], np.array([12.0]), ld[one], np.array([15.0]), np.zeros(1))
    assert a[0] == pytest.approx(gap_mode) and gap_mode == pytest.approx(0.896)
    far = sim._law_acc(idx[one], v[one], np.array([12.0]), ld[one], np.array([40.0]), np.zeros(1))
    assert far[0] == pytest.approx(0.4 * (12.0 - 8.0))  # the speed mode is the smaller far behind
    # the crossing: 0.23 (g - 10.8) - 0.07 = 1.6 at g = 18.06 m
    for g_x, mode in ((18.0, "gap"), (18.2, "speed")):
        got = sim._law_acc(idx[one], v[one], np.array([12.0]), ld[one], np.array([g_x]), np.zeros(1))[0]
        assert got == pytest.approx(0.23 * (g_x - 10.8) - 0.07 if mode == "gap" else 1.6), g_x


def test_the_cacc_law_against_hand_values():
    """e = gap - s0 - Tc v, edot = (v_lead - v) - Tc a_now, a = (0.45 e + 0.0125 edot) / 0.1 = 4.5 e + 0.125 edot,
    for a CAV directly behind a cooperating CAV (string position 2), with the CACC margin (flat 1.25 m)."""
    assert (CACC_KP, CACC_KD, CACC_PERIOD) == (0.45, 0.0125, 0.1)
    sim, slots = lane_of([HUMAN, CAV, CAV], [0.0, 20.0, 5.9], [6.0, 6.5, 6.0])
    sim.a_now[slots[2]] = 0.2
    idx, v, ld, g = controller_state(sim)
    k = np.flatnonzero(idx == slots[2])
    assert sim.string_pos[slots].tolist() == [0, 1, 2]  # (behind a human: a string leader, then a member)
    assert sim.headway[slots[2]] == 0.7 and sim.s0[slots[2]] == 1.25
    e = 5.9 - 1.25 - 0.7 * 6.0  # 0.45
    edot = (6.5 - 6.0) - 0.7 * 0.2  # 0.36
    a = sim._law_cav(idx[k], v[k], np.array([12.0]), ld[k], g[k], np.zeros(1))
    assert a[0] == pytest.approx((0.45 * e + 0.0125 * edot) / 0.1) and a[0] == pytest.approx(2.07)
    assert sim._law_cav(idx[k], v[k], np.array([6.5]), ld[k], g[k], np.zeros(1))[0] == pytest.approx(0.2)  # speed


def test_a_cav_behind_anything_else_follows_by_the_acc_law_at_its_fallback_gap_without_dead_time():
    sim, slots = lane_of([HUMAN, CAV], [0.0, 15.0], [7.0, 8.0], fallback=2.2)
    idx, v, ld, g = controller_state(sim)
    cav = slots[1]
    assert sim.string_pos[cav] == 1 and sim.headway[cav] == 2.2 and sim.s0[cav] == 2.0 and sim.delay[cav] == 0
    k = np.flatnonzero(idx == cav)
    a = sim._law_cav(idx[k], v[k], np.array([12.0]), ld[k], g[k], np.zeros(1))
    assert a[0] == pytest.approx(0.23 * (15.0 - 2.0 - 2.2 * 8.0) + 0.07 * (7.0 - 8.0))
    assert sim.string_pos[slots[0]] == 0  # (a human driver is in no string)


# ------------------------------------------------------------------ settings, dead time, brands
def test_the_controllers_settings_dead_time_and_brand_factors():
    """ACC car: T = ACC time gap x brand x road factor, no floor since the flat margin (a truck x 1.5 on top), dead
    time ACC dead time x brand, at most ACC_DEAD_MAX, in the delay buffer, no spread; company k of n gets 1 + spread
    (2k/(n-1) - 1). A CAV: its ACC-fallback gap the same way, no dead time. Changed live, they follow at once."""
    cfg = SimConfig(seed=3, auto_share=1.0, auto_kind="acc", companies=3, brand_spread=0.5, truck_share=0.3,
                    acc_delay=0.6, acc_headway=1.2, density=40, headway_std=0.3, reaction_std=0.2, speed_std=1.5)
    sim = RingSim(cfg)
    live = np.flatnonzero(sim.alive)
    k = sim.company[live].astype(float)
    brand = 1.0 + 0.5 * (2.0 * k / 2 - 1.0)
    assert set(np.round(brand, 12)) == {0.5, 1.0, 1.5}
    np.testing.assert_array_equal(sim._brand(sim.company[live]), brand)
    gap = 1.2 * brand * 1.0  # (brand 0.5: 0.6 s; the floor of 0.7 s went with Xiao's speed term)
    assert ACC_DEAD_MAX == 0.6 and set(np.round(gap, 12)) == {0.6, 1.2, 1.8}
    np.testing.assert_array_equal(sim.headway[live], np.where(sim.truck[live], TRUCK_HEADWAY_FACTOR * gap, gap))
    np.testing.assert_array_equal(sim.delay[live], np.rint(np.minimum(0.6 * brand, ACC_DEAD_MAX) / 0.1).astype(int))
    assert set(sim.delay[live]) == {3, 6} and (sim.s0[live] == ACC_S0).all()  # (brand 1.5: 0.9 s, kept at 0.6 s)
    assert not sim.set(acc_headway=1.5, acc_delay=0.2, road_condition="wet")  # live, with the road's factor
    gap = 1.5 * brand * road_factors("wet")["time_headway"]
    np.testing.assert_allclose(sim.headway[live], np.where(sim.truck[live], TRUCK_HEADWAY_FACTOR * gap, gap))
    np.testing.assert_array_equal(sim.delay[live], np.rint(0.2 * brand / 0.1).astype(int))
    # one company or no spread: the plain number 1 (exact)
    assert RingSim(cfg.updated(companies=1))._brand(np.array([0])) == 1.0
    assert RingSim(cfg.updated(brand_spread=0.0))._brand(np.array([2])) == 1.0
    two = RingSim(cfg.updated(companies=2, brand_spread=0.3))
    np.testing.assert_allclose(two._brand(np.array([0, 1, -1])), [0.7, 1.3, 1.0])
    cav = RingSim(cfg.updated(auto_kind="cav"))
    c = np.flatnonzero(cav.alive)
    fb = cav.acc_gap[c] * cav._brand(cav.company[c])
    np.testing.assert_array_equal(cav.headway[c], np.where(cav.truck[c], TRUCK_HEADWAY_FACTOR * fb, fb))
    assert (cav.delay[c] == 0).all() and set(np.unique(cav.acc_gap[c])) <= {1.1, 1.6, 2.2}


def test_no_spread_between_controllers_but_a_robotaxi_keeps_the_drivers_time_gap_spread():
    base = dict(seed=4, density=40, headway_std=0.3, reaction_std=0.2, speed_std=1.5, auto_share=1.0)
    acc = RingSim(SimConfig(**base, auto_kind="acc"))
    cars = acc.alive & ~acc.truck
    assert np.unique(acc.headway[cars]).size == 1 and np.unique(acc.delay[acc.alive]).size == 1
    robot = RingSim(SimConfig(**base, auto_kind="robotaxi"))
    live = robot.alive
    assert np.unique(robot.headway[live]).size > 1 and np.unique(robot.delay[live]).size == 1
    assert (robot.z[live] == 0).all() and (robot.z_react[live] == 0).all() and (robot.z_gap[live] != 0).all()


# ------------------------------------------------------------------ strings
def test_string_positions_and_the_platoon_max_cut():
    """Behind a human, 12 cooperating CAVs: the first leads by the ACC law (position 1), members 2-8 follow at
    the CACC time gap, the CAV that would be member 9 keeps CAV_PLATOON_GAP under the CACC law and leads the next
    string (2, 3, 4 behind it)."""
    n = 12
    sim, slots = lane_of([HUMAN] + [CAV] * n, [0.0] + [8.0] * n, [6.0] * (n + 1), fallback=1.1)
    controller_state(sim)
    assert sim.string_pos[slots].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4]
    assert sim.cfg.platoon_max == 8 and CAV_PLATOON_GAP == 1.5
    T = sim.headway[slots]
    assert T[1] == 1.1 and (T[2:9] == 0.7).all() and T[9] == 1.5 and (T[10:] == 0.7).all()
    s0 = sim.s0[slots]
    assert s0[1] == 2.0 and (s0[2:] == 1.25).all()  # ACC margin for the leader, CACC behind
    # platoon_max 1: no strings, every CAV behind a CAV keeps the 1.5 s gap
    assert not sim.set(platoon_max=1)
    controller_state(sim)
    assert sim.string_pos[slots].tolist() == [0, 1] + [2] * (n - 1) and (sim.headway[slots[2:]] == 1.5).all()
    assert not sim.set(platoon_max=4, cav_headway=0.9)
    controller_state(sim)
    assert sim.string_pos[slots].tolist() == [0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4]
    assert (sim.headway[slots[[2, 3, 4, 6]]] == 0.9).all() and sim.headway[slots[5]] == 1.5


def test_a_lane_of_cavs_only_is_a_loop_led_by_its_lowest_uid():
    sim, slots = lane_of([CAV] * 10, [0.0] + [8.0] * 9, [6.0] * 10)
    controller_state(sim)
    order = np.argsort(sim.uid[slots])  # spawned in driving order: the first has the lowest uid
    assert order[0] == 0
    assert sim.string_pos[slots].tolist() == [9, 2, 3, 4, 5, 6, 7, 8, 9, 2]
    assert sim.headway[slots[0]] == 1.5 and sim.headway[slots[8]] == 1.5 and sim.headway[slots[1]] == 0.7
    short, s2 = lane_of([CAV] * 3, [0.0, 8.0, 8.0], [6.0] * 3)  # a loop shorter than platoon_max: led all the same
    controller_state(short)
    assert short.string_pos[s2].tolist() == [9, 2, 3]


def test_pointer_jumping_gives_exact_depths_and_finds_loops():
    # chains 0 <- 1 <- 2 <- 3 (0 heads), 4 <- 5, and a loop 6 -> 7 -> 8 -> 6 (each joined to the one ahead)
    lead = np.array([0, 0, 1, 2, 4, 4, 8, 6, 7])
    joined = np.array([False, True, True, True, False, True, True, True, True])
    depth, loop, low = _chain_depth(joined, lead, np.array([10, 11, 12, 13, 14, 15, 26, 25, 27]))
    assert depth[:6].tolist() == [0, 1, 2, 3, 0, 1] and loop.tolist() == [False] * 6 + [True] * 3
    assert low[6:].tolist() == [25, 25, 25]
    long = np.arange(-1, 999) % 1000  # a chain of 1000 (element 0 heads)
    j = np.ones(1000, dtype=bool)
    j[0] = False
    d, loop, _ = _chain_depth(j, long)
    assert d.tolist() == list(range(1000)) and not loop.any()


@pytest.mark.parametrize("coop_class, coop_between, companies, joined", [
    ("B", "same", (0, 0), True),  # within a company: class B >= A
    ("A", "same", (0, 1), True),  # same: as within
    ("B", "none", (0, 0), True),  # (the cap acts only between companies)
    ("B", "none", (0, 1), False),  # capped to none between them
    ("B", "A", (0, 1), True),
    ("none", "B", (0, 1), False),  # never above the class within a company
    ("none", "same", (0, 0), False),  # no V2X: ACC mode
])
def test_cacc_needs_class_a_or_more_between_the_pair(coop_class, coop_between, companies, joined):
    sim, slots = lane_of([HUMAN, CAV, CAV], [0.0, 10.0, 8.0], [6.0] * 3, firms=[0, *companies],
                         coop_class=coop_class, coop_between=coop_between, companies=2)
    controller_state(sim)
    assert sim.string_pos[slots[2]] == (2 if joined else 1)
    level = sim._coop_class(CAV, np.array([companies[0]]), CAV, np.array([companies[1]]))[0]
    assert (level >= COOP_CLASSES.index("A")) == joined


# ------------------------------------------------------------------ the clip and the safety override
def test_the_clip_then_the_safety_override_after_the_dead_time_with_its_episodes():
    """Changed on purpose (the S2 decision on the safety layer): _automated_limits only clips to [-max_decel, accel]
    before over_braking and the delay buffer; the override a = min(a, a_idm) where a_idm < -comfortable braking acts
    after the buffer (_safety_override: no ACC dead time), its demand beyond the braking limit counts as harsh
    braking, and an activation (a step where it lowers the command it acts on) is counted once per episode
    (per-car hold of 2 s) as n_override (it was n_aeb)."""
    sim, slots = lane_of([ACC] * 6, [0.0] + [30.0] * 5, [5.0] * 6, auto_kind="acc")
    idx = slots[[1, 2, 3, 4, 5]]
    b, md, a_max = sim.eff.comfort_decel, sim.eff.max_decel, sim.eff.max_accel
    assert (b, md, a_max) == (2.0, 9.0, 1.2)
    law = np.array([1.0, -12.0, 0.5, -5.0, 3.0])
    a_idm = np.array([-3.0, -1.0, -1.5, -10.0, 0.4])
    acc = law.copy()
    pending = sim._automated_limits(idx, acc, a_idm)
    np.testing.assert_array_equal(acc, [1.0, -9.0, 0.5, -5.0, 1.2])  # (the clip only)
    np.testing.assert_array_equal(pending[0], np.arange(5))
    np.testing.assert_array_equal(pending[1], a_idm)
    over = np.zeros(5, dtype=bool)
    sim._safety_override(idx, acc, pending, over)
    np.testing.assert_array_equal(acc, [-3.0, -9.0, 0.5, -10.0, 1.2])
    assert over.tolist() == [False, False, False, True, False]  # (-10 beyond the 9 m/s2 limit: harsh)
    assert sim.n_override == 2 and (sim.override_until[idx[[0, 3]]] == sim.time + 2.0).all()
    acc = law.copy()
    sim._safety_override(idx, acc, sim._automated_limits(idx, acc, a_idm), over)  # still under way: no new episode
    assert sim.n_override == 2
    sim.time += 2.5
    acc = law.copy()
    sim._safety_override(idx, acc, sim._automated_limits(idx, acc, a_idm), over)
    assert sim.n_override == 4
    # after an ACC car's dead time the override acts on the delayed command: one asking for less braking than the
    # IDM is lowered (counted), one already braking harder is left alone
    acc = np.array([-1.0, -4.0])
    sim.time += 2.5
    sim._safety_override(idx[:2], acc, (np.arange(2), np.array([-3.0, -3.0])), np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(acc, [-3.0, -4.0])
    assert sim.n_override == 5
    # a robotaxi's law is its IDM: it gets the override in its decision (never lowering it), nothing after the buffer
    robot, rs = lane_of([ROBOTAXI] * 3, [0.0, 30.0, 30.0], [5.0] * 3, auto_kind="robotaxi")
    a = np.array([-3.0, -10.0])
    acc = a.copy()
    assert robot._automated_limits(rs[1:], acc, a) is None
    np.testing.assert_array_equal(acc, a)
    assert robot.n_override == 0
    human = np.array([-20.0])
    plain, ps = lane_of([HUMAN, HUMAN], [0.0, 10.0], [5.0, 5.0], auto_kind="acc")
    assert plain._automated_limits(ps[1:], human, np.array([-20.0])) is None
    assert human[0] == -20.0  # (human drivers: never touched)


def test_the_safety_layer_acts_in_a_run_and_robotaxis_pass_it_unchanged_on_a_dry_road():
    """A dense ring of ACC cars (string-unstable) calls on the safety override; robotaxi runs are the same with the
    layer reduced to the clip (on a dry road their commands lie within the limits it keeps). Changed on purpose (the
    override after the dead time, the S2 decision): the ring's harsh braking, the override delayed by the ACC dead
    time before, is gone."""
    acc = RingSim(SimConfig(seed=2, n_lanes=1, n_arms=0, density=70, auto_share=1.0, auto_kind="acc"))
    acc.step(3000)
    assert acc.n_override > 0 and acc.n_emergency_auto == acc.n_emergency == 0
    cfg = SimConfig(seed=5, n_lanes=3, density=60, exit_prob=0.3, auto_share=0.5, auto_kind="robotaxi")
    real, clipped = RingSim(cfg), RingSim(cfg)
    clipped._automated_limits = lambda idx, acc_, a_idm: np.copyto(
        acc_, np.where(clipped.kind[idx] != HUMAN, np.minimum(np.maximum(acc_, -clipped.eff.max_decel),
                                                              clipped.accel[idx]), acc_))
    real.step(600)
    clipped.step(600)
    np.testing.assert_array_equal(real.theta, clipped.theta)
    np.testing.assert_array_equal(real.v, clipped.v)
    assert real.n_override == 0


# ------------------------------------------------------------------ dt and the CACC update period
def spectral_radius(dt: float, Tc: float, divisor: float) -> float:
    """The linearised loop of a CACC follower behind a steady leader as the engine steps it (state: gap, speed,
    last acceleration; a = (kp e + kd edot) / divisor, v' = v + a dt, gap' = gap - v dt - a dt^2 / 2)."""
    K, D = CACC_KP / divisor, CACC_KD / divisor
    a_s, a_v, a_ap = K, -K * Tc - D, -D * Tc
    M = np.array([[1 - 0.5 * dt * dt * a_s, -dt - 0.5 * dt * dt * a_v, -0.5 * dt * dt * a_ap],
                  [dt * a_s, 1 + dt * a_v, dt * a_ap], [a_s, a_v, a_ap]])
    return float(max(abs(np.linalg.eigvals(M))))


def test_the_cacc_law_at_steps_longer_than_its_update_period():
    """The discretised law a = (kp e + kd edot) / CACC_PERIOD is stable up to dt = CACC_PERIOD (the default
    step); dividing by CACC_PERIOD at a longer step goes unstable (a 1.5 s gap from dt 0.25 s), so the engine
    divides by max(CACC_PERIOD, dt): one update per step with the published gains, stable to dt 1 s."""
    assert SimConfig().dt == CACC_PERIOD and CACC_GAIN_GAP == 2.0

    def guarded(dt, Tc):  # the engine's divisor (_law_cav)
        return spectral_radius(dt, Tc, max(CACC_PERIOD, dt) * max(1.0, Tc / CACC_GAIN_GAP))

    for Tc in (0.7, 1.5, 2.4):
        assert spectral_radius(0.1, Tc, 0.1) < 1 and spectral_radius(0.05, Tc, 0.1) < 1
        for dt in (0.25, 0.5, 1.0):
            assert guarded(dt, Tc) < 1, (Tc, dt)
    assert spectral_radius(0.25, 1.5, 0.1) > 1 and spectral_radius(1.0, 0.7, 0.1) > 1
    # Beyond 2.84 s the published law fails even at its own period (a truck of a slow brand at the 1.5 s string
    # gap: 1.5 x 1.5 x 1.6 = 3.6 s); with its gains scaled by CACC_GAIN_GAP / Tc it holds.
    assert spectral_radius(0.1, 2.8, 0.1) < 1 < spectral_radius(0.1, 2.9, 0.1)
    assert all(guarded(dt, Tc) < 1 for dt in (0.05, 0.1, 0.2, 0.5, 1.0) for Tc in (0.5, 2.9, 3.6, 5.8, 8.0))
    # In the engine at dt 0.5 s: a CACC follower at the 1.5 s gap (platoon_max 1) settles behind a steady leader.
    sim, slots = lane_of([CAV, CAV], [0.0, 6.0], [10.0, 10.0], dt=0.5, platoon_max=1, mean_speed=12.0)
    lead = slots[0]
    pos = 0.0
    for _ in range(240):
        sim.step()
        pos += 10.0 * 0.5
        sim.theta[lead], sim.v[lead] = (pos / sim.radii[0]) % (2 * np.pi), 10.0
    idx, gap = sim.bumper_gaps()
    follower = int(np.flatnonzero(idx == slots[1])[0])
    assert sim.string_pos[slots[1]] == 2 and sim.headway[slots[1]] == 1.5
    assert gap[follower] == pytest.approx(1.25 + 1.5 * 10.0, abs=0.05) and sim.v[slots[1]] == pytest.approx(10.0, abs=1e-3)


# ------------------------------------------------------------------ behaviour in a platoon
def platoon(kind: str, v0: float = 20.0, depth: float = 2.0, period: float = 12.0, n: int = 8, law_only=False,
            **config) -> tuple[np.ndarray, RingSim]:
    """A leader whose speed dips smoothly by ``depth`` m/s over ``period`` s, with ``n`` followers of ``kind`` at
    their law's equilibrium gap at ``v0`` (desired speed 25 m/s, so the speed mode stays off), alone on a 1 km
    radius ring. Returns each vehicle's dip (v0 - its lowest speed), leader first, and the simulation.
    ``law_only``: the clip only, no safety override (the controller alone)."""
    T = config.get("acc_headway", 1.1) if kind == "acc" else config.get("cav_headway", 0.7)
    s0 = RingSim.acc_margin(v0) if kind == "acc" else RingSim.cacc_margin(v0)
    gap = float(s0 + T * v0)
    sim, slots = lane_of([ACC if kind == "acc" else CAV] * (n + 1), [0.0] + [gap] * n, [v0] * (n + 1),
                         auto_kind=kind, mean_speed=25.0, platoon_max=20, **config)
    if law_only:
        sim._automated_limits = lambda idx, acc, a_idm: np.copyto(
            acc, np.minimum(np.maximum(acc, -sim.eff.max_decel), sim.accel[idx]))
    lead, pos, low = slots[0], 0.0, np.full(n + 1, v0)
    for _ in range(900):
        t = sim.time
        v_lead = v0 - (depth * 0.5 * (1 - np.cos(2 * np.pi * (t - 5.0) / period)) if 5.0 <= t <= 5.0 + period else 0.0)
        sim.step()
        pos += v_lead * sim.cfg.dt
        sim.theta[lead], sim.v[lead] = (pos / sim.radii[0]) % (2 * np.pi), v_lead
        low = np.minimum(low, sim.v[slots])
    return v0 - low, sim


def test_an_acc_platoon_amplifies_a_leaders_speed_dip():
    """Milanes & Shladover 2014 (Table 3, Sec. 5): the production ACC law is string-unstable. A 2 m/s dip of the
    leader over 12 s (~0.5 rad/s, near the law's peak gain) at 20 m/s, inside the calibration range, grows down a
    platoon of 8 ACC cars, by the law alone (0.5 s dead time or none) and with the safety override on top."""
    for config in (dict(), dict(acc_delay=0.0)):
        dips, sim = platoon("acc", law_only=True, **config)
        assert dips[-1] > 1.8 * dips[0], (config, dips)
        assert (np.diff(dips[2:]) > 0).all(), dips  # (growing car after car once under way)
    dips, sim = platoon("acc")
    assert dips[-1] > 1.8 * dips[0] and sim.n_override > 0  # (the safety override steps in, the dip still grows)


def test_a_cacc_string_attenuates_a_leaders_speed_dip():
    """PATH CACC (string members at 0.7 s) damps the same dip car after car, without the safety override; faster
    dips more (the law is a low-pass filter, Milanes & Shladover 2014 Table 2)."""
    for depth, period in ((2.0, 12.0), (4.0, 10.0), (2.0, 6.0)):
        dips, sim = platoon("cav", depth=depth, period=period)
        assert (np.diff(dips) < 0).all() and dips[-1] < 0.98 * dips[0], (depth, period, dips)
        # (alone in the lane the platoon is a loop: its leader, the lowest uid, leads it at position 21)
        assert sim.n_override == 0 and sorted(sim.string_pos[sim.alive].tolist()) == [2, 3, 4, 5, 6, 7, 8, 9, 21]


def test_below_10_ms_the_flat_cacc_margin_keeps_the_string_damping():
    """Changed on purpose (flat margins): Xiao 2017 Eq. 7's margin fell 0.125 m per m/s below 10 m/s, so a 0.7 s string
    regulated (Tc - 0.125 s) v + 1.25 m and amplified a dip mildly at 8 m/s (x1.065 over 8 cars). Flat, it damps
    there too (string-stable iff Tc >= sqrt(2 CACC_PERIOD / kp) = 0.667 s, *derived*); at Tc 0.6 s it amplifies."""
    dips, _ = platoon("cav", v0=8.0)
    assert (np.diff(dips) < 0).all() and dips[-1] < dips[0], dips
    dips, _ = platoon("cav", v0=8.0, cav_headway=0.6)
    assert dips[-1] > dips[0], dips
