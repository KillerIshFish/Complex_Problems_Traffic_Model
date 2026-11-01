"""Drivers' adaptation to recognisable AVs (SPEC 3, 12): exploit_gap (a driver or an ACC car's driver entering in front
of a recognisable lag vehicle accepts entry_gap x exploit_gap and asks it for up to comfortable braking / exploit_gap;
changing lane in front of a recognisable new follower, x the room it leaves behind and / the braking it may force on
it) and follow_adapt (behind a recognisable AV a driver keeps s0 x (1 - 0.375 a), starts from standstill after a
reaction delay x (1 - 0.53 a), and enters round(a x 0.55 / dt) steps early behind an AV entrant still directly
ahead); their settings, idle reasons, and that they act only on drivers meeting recognisable AVs."""

from __future__ import annotations

import numpy as np
import pytest

from roundabout_sim import presets
from roundabout_sim.config import IDLE_WHY, PARAM_BY_NAME, SimConfig, inactive_inputs, inactive_reason
from roundabout_sim.engine import (ACC, CAV, FOLLOW_S0_CUT, FOLLOW_START_CUT, FOLLOW_UP_CUT, HUMAN, ROBOTAXI,
                                   SMOOTHER, SUBJECT_ARM, TWO_PI, RingSim)
from tests.test_entries import entry_scene, check
from tests.test_state_migration import trajectory

NEW = {"exploit_gap": 1.0, "follow_adapt": 0.0}


# ------------------------------------------------------------------ settings
def test_the_adaptation_settings_defaults_ranges_and_tab():
    cfg = SimConfig()
    assert {name: getattr(cfg, name) for name in NEW} == NEW
    for name in NEW:
        spec = PARAM_BY_NAME[name]
        assert spec.group == "Drivers" and not spec.reset and len(spec.help) <= 340, name
        assert spec.lo <= NEW[name] <= spec.hi and presets.RANDOM_RANGES[name].basis.strip()
    assert (PARAM_BY_NAME["exploit_gap"].lo, PARAM_BY_NAME["exploit_gap"].hi) == (0.5, 1.5)
    assert (PARAM_BY_NAME["follow_adapt"].lo, PARAM_BY_NAME["follow_adapt"].hi) == (0.0, 1.0)
    # (on the Drivers tab but never drawn: random settings and designs of the drivers stay what they were)
    assert all(presets.RANDOM_RANGES[name].choices == (NEW[name],) for name in NEW)
    for seed in range(10):
        assert {name: presets.random_changes(seed).get(name, NEW[name]) for name in NEW} == NEW
        assert {name: presets.random_changes(seed, ["Drivers"])[name] for name in NEW} == NEW
    design = presets.latin_hypercube(8, presets.plausible_ranges(), 3)
    assert all(point[name] == NEW[name] for point in design for name in NEW)
    without = presets.latin_hypercube(8, {k: v for k, v in presets.plausible_ranges().items() if k not in NEW}, 3)
    assert [{k: v for k, v in point.items() if k not in NEW} for point in design] == without
    for bad in (dict(exploit_gap=0.0), dict(follow_adapt=-0.1), dict(follow_adapt=1.5)):
        with pytest.raises(ValueError):
            SimConfig(**bad)
    # the derived factors: 1 - 1.73 / 2.77 and 1 - 0.49 / 1.04 (Wang et al. 2023 via Reddy et al. 2025), 1.04 - 0.49 s
    assert FOLLOW_S0_CUT == pytest.approx(1 - 1.73 / 2.77, abs=1e-3)
    assert FOLLOW_START_CUT == pytest.approx(1 - 0.49 / 1.04, abs=2e-3) and FOLLOW_UP_CUT == pytest.approx(1.04 - 0.49)


@pytest.mark.parametrize("name, config, reason", [
    ("exploit_gap", dict(), "no recognisable AVs"),
    ("exploit_gap", dict(auto_share=0.5, auto_kind="acc"), "no recognisable AVs"),  # (an ACC car looks like any car)
    ("exploit_gap", dict(auto_share=0.5, auto_kind="robotaxi"), ""),
    ("exploit_gap", dict(smoothers=2), ""),
    ("exploit_gap", dict(smoothers=2, n_arms=0, n_lanes=1), "no merges"),
    ("exploit_gap", dict(smoothers=2, n_arms=0, n_lanes=2), ""),
    ("follow_adapt", dict(auto_share=0.5, auto_kind="acc"), "no recognisable AVs"),
    ("follow_adapt", dict(auto_share=0.5, auto_kind="cav", n_arms=0, n_lanes=1), ""),
])
def test_each_adaptation_setting_says_why_it_does_nothing(name, config, reason):
    assert inactive_reason(name, SimConfig(**config)) == reason
    assert reason == "" or reason in IDLE_WHY
    assert inactive_inputs("exploit_gap") == {"auto_share", "auto_kind", "smoothers", "n_arms", "n_lanes"}
    assert inactive_inputs("follow_adapt") == {"auto_share", "auto_kind", "smoothers"}


@pytest.mark.parametrize("config", [
    dict(auto_share=0.5, auto_kind="acc", traffic_mode="open", entry_rate=0.2),
    dict(traffic_mode="open", entry_rate=0.2, truck_share=0.1),
], ids=["acc", "human"])
def test_without_recognisable_avs_the_adaptation_changes_nothing(config):
    base = RingSim(SimConfig(seed=3, **config))
    other = RingSim(SimConfig(seed=3, exploit_gap=0.6, follow_adapt=1.0, **config))
    assert not (other.exploiting or other.adapting)
    base.step(1500)
    other.step(1500)
    assert trajectory(other) == trajectory(base) and base.n_entered > 50


# ------------------------------------------------------------------ exploit_gap at the entries
@pytest.mark.parametrize("lag_kind, v, lag_s, exploit, ok", [
    (ROBOTAXI, 9.0, 1.8, 1.0, False),  # 1.8 s < entry_gap 2 s
    (ROBOTAXI, 9.0, 1.8, 0.8, True),  # 1.8 >= 2 x 0.8, and its IDM (-2.44 m/s2) >= -2 / 0.8
    (ROBOTAXI, 10.0, 2.1, 1.0, False),  # the gap would do, but its IDM (-2.24 m/s2) < -2: the AV's longer T vetoes
    (ROBOTAXI, 10.0, 2.1, 0.8, True),  # ...which the exploit's braking threshold, -2 / 0.8, lets through
    (ROBOTAXI, 6.0, 2.1, 1.2, False),  # wary: 2.1 s < 2 x 1.2
    (ROBOTAXI, 6.0, 2.1, 1.0, True),
    (HUMAN, 9.0, 1.8, 0.6, False),  # a driver lag: no exploit
    (ACC, 9.0, 1.8, 0.6, False),  # an ACC lag looks like any car
])
def test_a_driver_exploits_or_is_wary_of_a_recognisable_lag_vehicle(lag_kind, v, lag_s, exploit, ok):
    """A driver head (entry_gap 2 s, comfortable braking 2 m/s2) with the lag vehicle lag_s seconds behind at v m/s
    (a robotaxi: T 1.56 s, s0 3.94 m; its IDM at the entrant's 6 m/s computed by the engine's own check). The
    entrant's own exploit applies only in front of a recognisable lag."""
    kind = {ROBOTAXI: "robotaxi", ACC: "acc"}.get(lag_kind, "robotaxi")
    sim, _ = entry_scene({0: (HUMAN, -1, np.nan)}, [(lag_kind, -(lag_s * v + 4.0), v), (HUMAN, 60.0, 6.0)],
                         auto_kind=kind, exploit_gap=exploit, radius=60.0)
    assert check(sim)[0] == ok


def test_exploiting_never_makes_a_driver_stricter():
    """On ice with av_weather_scale 0 a robotaxi lag's comfortable braking (2 m/s2, a dry road's) exceeds the road's
    braking limit (1.35 m/s2): the exploited threshold, capped at the limit, must not be stricter than the plain -2
    m/s2, so a lag 2.3 s back at 8 m/s that a driver accepts at exploit_gap 1 is accepted at 0.95 and 0.8 too."""
    for exploit in (1.0, 0.95, 0.8):
        sim, _ = entry_scene({0: (HUMAN, -1, np.nan)}, [(ROBOTAXI, -(2.3 * 8.0 + 4.0), 8.0), (HUMAN, 60.0, 6.0)],
                             auto_kind="robotaxi", exploit_gap=exploit, radius=60.0, road_condition="ice",
                             av_weather_scale=0.0)
        lag = np.flatnonzero(sim.alive)[0]
        assert float(np.atleast_1d(sim._comfort(np.array([lag])))[0]) > sim.eff.max_decel
        assert check(sim)[0], exploit


def test_an_automated_head_does_not_exploit():
    for head, kind in ((CAV, "cav"), (ROBOTAXI, "robotaxi")):
        base, _ = entry_scene({0: (head, 0, 1.6)}, [(ROBOTAXI, -(1.8 * 9.0 + 4.0), 9.0), (HUMAN, 60.0, 6.0)],
                              auto_kind=kind, radius=60.0)
        other, _ = entry_scene({0: (head, 0, 1.6)}, [(ROBOTAXI, -(1.8 * 9.0 + 4.0), 9.0), (HUMAN, 60.0, 6.0)],
                               auto_kind=kind, radius=60.0, exploit_gap=0.6)
        assert check(base) == check(other) and not check(other)[0]


def test_an_acc_cars_driver_exploits_as_a_driver_does():
    """An ACC car's driver enters as a driver: in an ACC run with a smoother (recognisable from Smoother start on)
    as its lag vehicle it exploits it; before Smoother start the smoother counts as its driver's car."""
    for start, ok in ((0.0, True), (100.0, False)):
        sim, _ = entry_scene({0: (ACC, 0, np.nan)}, [(HUMAN, -(1.8 * 9.0 + 4.0), 9.0), (HUMAN, 60.0, 6.0)],
                             auto_kind="acc", radius=60.0, smoothers=1, smoother_start=start, exploit_gap=0.8)
        lag = np.flatnonzero(sim.alive)[0]
        sim.kind[lag] = SMOOTHER
        assert sim.exploiting and check(sim)[0] == ok, start
        sim.set(exploit_gap=1.0)
        assert not check(sim)[0]


# ------------------------------------------------------------------ exploit_gap at lane changes
def lane_change_scene(follower_kind: int, back: float, exploit: float) -> tuple[RingSim, int]:
    """Two lanes (r 30 m, 4 arms, no noise): a driver on the inner lane at 5 m/s that must move out (its exit 30 m on),
    and on the outer lane a single vehicle (kind ``follower_kind``) ``back`` m behind it (bumper) at 9 m/s. Returns
    (sim, the driver's slot)."""
    cfg = SimConfig(seed=2, n_lanes=2, n_arms=4, radius=30.0, traffic_mode="open", entry_rate=0.0, density=1,
                    driver_noise=0.0, speed_std=0.0, auto_share=1.0, auto_kind="robotaxi", exploit_gap=exploit)
    sim = RingSim(cfg)
    sim.alive[:] = False
    for fifo in (*sim.queue_times, *sim.queue_trucks, *sim.queue_kinds, *sim.queue_companies, *sim.queue_acc_gaps):
        fifo.clear()
    sim.queue[:] = 0
    r_in, r_out = sim.radii
    th = 1.0
    slots = sim._spawn(np.array([0, 1]), np.array([th, th - (back + 4.0) / r_out]), np.array([5.0, 9.0]),
                       kinds=(np.array([HUMAN, follower_kind], dtype=np.int8), np.array([-1, 0], dtype=np.int8),
                              np.full(2, np.nan)))
    sim.exit_at[slots[0]] = sim.travel[slots[0]] + 30.0 / r_out
    sim.exit_at[slots[1]] = np.inf
    sim.lc_cool[slots] = 0.0
    return sim, int(slots[0])


def braking_of(sim: RingSim, slot: int, back: float) -> float:
    """[m/s2] The new follower's IDM behind the driver after the change (MOBIL's a_f_new)."""
    f = np.flatnonzero(sim.alive & (sim.lane == 1))
    return float(sim._idm(sim.v[f], sim.target_speed(f), np.array([back]), sim.v[[slot]], f)[0])


def test_a_driver_changing_lane_in_front_of_a_recognisable_follower_exploits_it():
    """A mandatory change (a leaver): safe while the new follower need brake no harder than b_safe = 2.5 x 2 = 5 m/s2.
    With its follower asked for between 5 and 5 / 0.8 m/s2 (each follower kind at its own such gap) the driver changes
    in front of a robotaxi only with exploit_gap 0.8, never in front of a driver."""
    backs = np.linspace(4.0, 20.0, 321)
    for kind in (ROBOTAXI, HUMAN):
        probe, slot = lane_change_scene(kind, 10.0, 1.0)
        need = np.array([braking_of(probe, slot, b) for b in backs])
        band = backs[(need < -5.0) & (need >= -5.0 / 0.8)]
        assert band.size, kind
        back = float(band[band.size // 2])
        for exploit, moved in ((1.0, False), (0.8, kind == ROBOTAXI)):
            sim, slot = lane_change_scene(kind, back, exploit)
            sim._lane_changes(np.flatnonzero(sim.alive))
            assert (sim.lane[slot] == 1) == moved, (kind, exploit)


def test_the_room_left_behind_a_driver_shrinks_in_front_of_a_recognisable_follower():
    """Both standing, the robotaxi follower 1.85 m behind: its IDM asks 1.2 (1 - (3.94 / 1.85)^2) = -4.24 m/s2, within
    b_safe 5, but the room it needs, 0.5 x 3.94 = 1.97 m, is more than 1.85 m: no change; at exploit_gap 0.8 the
    room is 1.58 m (never below SAFE_MARGIN) and the driver changes lane."""
    for exploit, moved in ((1.0, False), (0.8, True)):
        sim, slot = lane_change_scene(ROBOTAXI, 1.85, exploit)
        sim.v[sim.alive] = 0.0
        assert braking_of(sim, slot, 1.85) == pytest.approx(1.2 * (1 - (3.94 / 1.85) ** 2))
        sim._lane_changes(np.flatnonzero(sim.alive))
        assert (sim.lane[slot] == 1) == moved, exploit


# ------------------------------------------------------------------ follow_adapt: car following
def ring_pair(leader_kind: int, gap: float, follow_adapt: float, **config) -> tuple[RingSim, int, int]:
    """One lane, no arms, no noise: a leader of ``leader_kind`` and a driver ``gap`` m (bumper) behind it, both
    standing. Returns (sim, leader slot, follower slot)."""
    cfg = SimConfig(**{**dict(seed=1, n_lanes=1, n_arms=0, radius=40.0, traffic_mode="open", density=1,
                              driver_noise=0.0, speed_std=0.0, auto_share=0.5, auto_kind="robotaxi",
                              follow_adapt=follow_adapt), **config})
    sim = RingSim(cfg)
    sim.alive[:] = False
    th = 1.0
    slots = sim._spawn(np.zeros(2, dtype=int), np.array([th, th - (gap + 4.0) / sim.radii[0]]), np.zeros(2),
                       kinds=(np.array([leader_kind, HUMAN], dtype=np.int8), np.array([0, -1], dtype=np.int8),
                              np.full(2, np.nan)))
    return sim, int(slots[0]), int(slots[1])


def test_a_driver_behind_a_recognisable_av_keeps_a_shorter_standstill_gap():
    """The IDM term of a driver behind a robotaxi uses s0 x (1 - 0.375 a) (2 m -> 1.25 m at a = 1); behind a driver its
    own 2 m. Hand-computed from the IDM."""
    for leader, adapt, s0 in ((ROBOTAXI, 1.0, 1.25), (ROBOTAXI, 0.5, 2.0 * (1 - 0.1875)), (HUMAN, 1.0, 2.0),
                              (ROBOTAXI, 0.0, 2.0)):
        sim, lead, me = ring_pair(leader, 3.0, adapt)
        sim.v[me] = 3.0
        idx = np.flatnonzero(sim.alive)
        lead_of, gap = sim._index_lanes(idx)
        k = int(np.flatnonzero(idx == me)[0])
        acc = sim._car_following(idx, sim.v[idx], sim.target_speed(idx), lead_of[idx], gap[idx], lead_of, gap)
        a, b, T, v, v0 = 1.2, 2.0, 1.2, 3.0, float(sim.target_speed(np.array([me]))[0])
        s_star = s0 + v * T + v * v / (2 * np.sqrt(a * b))
        assert acc[k] == pytest.approx(a * (1 - (v / v0) ** 4) - a * (s_star / 3.0) ** 2, rel=1e-12), (leader, adapt)


def test_a_driver_closes_up_behind_a_standing_av():
    """A driver rolling up to a standing robotaxi stops 1.25 m behind it at follow_adapt 1 (2 m at 0, and behind a
    driver)."""
    stops = {}
    for leader, adapt in ((ROBOTAXI, 1.0), (ROBOTAXI, 0.0), (HUMAN, 1.0)):
        sim, lead, me = ring_pair(leader, 20.0, adapt)
        sim.v[me] = 4.0
        for _ in range(600):
            sim.v[lead] = 0.0  # (the leader stays put)
            sim.brake_until[lead] = sim.time + 1.0
            sim.step()
        stops[(leader, adapt)] = float(((sim.theta[lead] - sim.theta[me]) % TWO_PI) * sim.radii[0] - 4.0)
    assert stops[(ROBOTAXI, 1.0)] == pytest.approx(1.25, abs=0.05)
    assert stops[(ROBOTAXI, 0.0)] == pytest.approx(2.0, abs=0.05)
    assert stops[(HUMAN, 1.0)] == pytest.approx(2.0, abs=0.05)


def test_a_driver_starts_sooner_from_standstill_behind_an_av():
    """Reaction time 1 s (10 steps): behind a robotaxi moving off a driver at a stand reacts after round(10 x (1 -
    0.53)) = 5 steps at follow_adapt 1; behind a driver, or at 0, after 10. Only from standstill (below 0.5 m/s)."""
    starts = {}
    for leader, adapt in ((ROBOTAXI, 1.0), (ROBOTAXI, 0.0), (HUMAN, 1.0)):
        s0 = 2.0 * (1 - 0.375 * adapt) if leader == ROBOTAXI else 2.0
        sim, lead, me = ring_pair(leader, s0, adapt, reaction_time=1.0)
        moved = None
        for k in range(200):
            sim.step()
            if moved is None and sim.v[me] > 0.02:
                moved = k
        starts[(leader, adapt)] = moved
    assert starts[(ROBOTAXI, 0.0)] - starts[(ROBOTAXI, 1.0)] == 5
    assert starts[(HUMAN, 1.0)] == starts[(ROBOTAXI, 0.0)]


def test_a_driver_under_way_lets_its_delay_grow_back_a_step_at_a_time():
    """Reaction time 1 s, follow_adapt 1: past FOLLOW_START_SPEED the driver's delay grows back from 5 to 10 steps one
    step per step (delay_used), holding its latest decision meanwhile, so its acceleration never drops while it gathers
    speed (switched back at once it fell 0.77 -> 0.56 m/s2 and repeated five older decisions) and it keeps a 5-step
    head start to 3 m/s; settled, the delay is its own again."""
    reach = {}
    for adapt in (1.0, 0.0):
        sim, lead, me = ring_pair(ROBOTAXI, 2.0 * (1 - 0.375 * adapt), adapt, reaction_time=1.0)
        speeds, accels, used = [], [], []
        for _ in range(150):
            sim.step()
            speeds.append(float(sim.v[me]))
            accels.append(float(sim.a_now[me]))
            used.append(int(sim.delay_used[me]))
        speeds, accels, used = np.array(speeds), np.array(accels), np.array(used)
        moving = np.flatnonzero((speeds > 0.02) & (speeds < 3.0))
        assert (np.diff(accels[moving]) >= 0.0).all(), adapt
        reach[adapt] = int(np.argmax(speeds >= 3.0))
        assert used[-1] == sim.delay[me] == 10
        if adapt:
            assert set(used.tolist()) == set(range(5, 11)) and (np.diff(used) <= 1).all()
    assert reach[0.0] - reach[1.0] == 5


def test_the_adaptation_is_a_drivers_only():
    """Automated followers keep their own gaps behind recognisable AVs; a lag driver judged behind a recognisable
    entrant uses its adapted standstill gap (_lag_gaps)."""
    sim, lead, me = ring_pair(ROBOTAXI, 3.0, 1.0)
    idx = np.flatnonzero(sim.alive)
    lead_of, _ = sim._index_lanes(idx)
    behind = sim._behind_av(idx, lead_of[idx])
    assert behind.tolist() == [False, True]  # (the robotaxi follows the driver round the ring: no adaptation)
    T, s0 = sim._lag_gaps(np.array([me, me]), np.array([ROBOTAXI, HUMAN]), np.array([0, -1]), np.array([0, 0]))
    assert s0.tolist() == [1.25, 2.0]


# ------------------------------------------------------------------ follow_adapt: the entry follow-up
def test_the_follow_up_slack_of_a_driver_behind_an_av_entrant():
    """entry_cool <= round(follow_adapt x 0.55 / dt): 6 steps at 1 (5.5 rounds to 6), 3 at 0.5 (2.75); behind a CAV
    or robotaxi entrant (recognisable), not an ACC car or a driver; an ACC car's driver too; behind True (only while
    that entrant is still directly ahead)."""
    sim, _ = entry_scene({0: (HUMAN, -1, np.nan), 1: (HUMAN, -1, np.nan), 2: (ACC, 0, np.nan), 3: (HUMAN, -1, np.nan)},
                         [(HUMAN, 60.0, 6.0)], follow_up=2.5, follow_adapt=1.0, auto_kind="robotaxi")
    sim.last_kind[:] = [ROBOTAXI, ACC, CAV, HUMAN]
    slack, behind = sim._follow_up_slack(sim.follow_up_steps())
    assert slack.tolist() == [6, 0, 6, 0] and behind.tolist() == [True, False, True, False]
    sim.set(follow_adapt=0.5)
    assert sim._follow_up_slack(sim.follow_up_steps())[0].tolist() == [3, 0, 3, 0]
    sim.set(follow_adapt=0.0)
    assert sim._follow_up_slack(sim.follow_up_steps())[0].tolist() == [0, 0, 0, 0]


class KindLog(RingSim):
    """Records (time, entrance, kind) of every entry (_entries spawns its entrants with ``lat``)."""

    def _spawn(self, lanes, angles, speeds, lat=None, trucks=None, *, kinds=None):
        slots = super()._spawn(lanes, angles, speeds, lat, trucks, kinds=kinds)
        if lat is not None:
            arm = int(np.argmin(np.abs((self.entry_angle - self.theta[slots[0]] + np.pi) % TWO_PI - np.pi)))
            self.entries.append((self.time, arm, int(self.kind[slots[0]])))
        return slots


def test_the_realised_follow_up_of_a_driver_behind_an_av_entrant():
    """Capacity test with no conflicting flow, 50 % robotaxis (robotaxi_entry 1 so only the drivers differ): a driver
    entering right behind a robotaxi goes 2.6 - 0.6 = 2.0 s after it at follow_adapt 1, 2.6 s behind a driver."""
    cfg = presets.apply("us_single_lane", SimConfig(seed=4)).updated(
        subject_arm="on", entry_rate=0.0, auto_share=0.5, auto_kind="robotaxi", robotaxi_entry=1.0, follow_adapt=1.0)
    sim = KindLog(cfg)
    sim.entries = []
    sim.step(2500)
    test = [(t, kind) for t, arm, kind in sim.entries if arm == SUBJECT_ARM]
    after = {}
    for (t0, k0), (t1, k1) in zip(test[:-1], test[1:]):
        after.setdefault((k0, k1), []).append(round(t1 - t0, 6))
    assert min(after[(ROBOTAXI, HUMAN)]) == pytest.approx(2.0) and np.median(after[(ROBOTAXI, HUMAN)]) == \
        pytest.approx(2.0)
    assert min(after[(HUMAN, HUMAN)]) == pytest.approx(2.6) and min(after[(ROBOTAXI, ROBOTAXI)]) == pytest.approx(2.6)
