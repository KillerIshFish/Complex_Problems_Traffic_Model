"""The wave smoothers (SPEC 2 'smoother'; Stern et al. 2018): the settings (Smoothers tab), why each does nothing
where it does nothing, and random draws only when the tab is named; FollowerStopper's boundaries and branches (the
paper's third branch), the speed tracking's time constants, PI with saturation's 38 s mean, saturation and update,
and the gap cap, against hand-computed values; the selection (no draws: the drivers' cars nearest evenly spaced
angles, never trucks) and live conversion and hand-back; smoothers never leave, change lane or get removed by the
density control; before Smoother start a smoother drives exactly as its driver (and then without a reaction delay,
through the clip and the safety layer); saved states with smoothers resume exactly; with smoothers 0 their other
settings change nothing; the viewer; and one FollowerStopper damping the waves of the calibrated stern2018 ring."""

from __future__ import annotations

import numpy as np
import pytest

from roundabout_sim import app as viewer
from roundabout_sim import experiments, presets
from roundabout_sim import sweep_viewer as sv
from roundabout_sim.config import IDLE_WHY, PARAM_BY_NAME, SimConfig, inactive_inputs, inactive_reason
from roundabout_sim.engine import (FS_DECEL, FS_DX0, HUMAN, PI_WINDOW, SMOOTHER, SMOOTHER_TAU_ACCEL,
                                   SMOOTHER_TAU_BRAKE, TWO_PI, RingSim, StateError)
from tests.helpers import frames, rendered_texts, show_tab, widget
from tests.test_state import assert_exact_resume

NEW = {"smoothers": 0, "smoother_law": "followerstopper", "smoother_speed": 7.5, "smoother_start": 0.0,
       "smoother_lane": -1}
OTHER = {"smoother_law": "pi", "smoother_speed": 6.0, "smoother_start": 50.0, "smoother_lane": 1}
STERN = {k: v for k, v in vars(presets.apply("stern2018")).items() if k != "seed"}  # the C1-calibrated ring


def smoothers_of(sim: RingSim) -> np.ndarray:
    return np.flatnonzero(sim.alive & (sim.kind == SMOOTHER))


def lane_with(gaps, speeds, smoother: int, **config) -> tuple:
    """A single-lane ring (no entrances; open mode: no density control) holding only these cars, in driving order
    from the front (``gaps[i]``: car i's bumper gap to car i - 1, the first's ignored), car ``smoother`` a smoother.
    Returns (sim, slots in that order)."""
    sim = RingSim(SimConfig(**{"seed": 1, "n_lanes": 1, "n_arms": 0, "traffic_mode": "open", "density": 1,
                               "radius": 1000.0, "driver_noise": 0.0, "speed_std": 0.0, "smoothers": 1, **config}))
    sim.alive[:] = False
    n = len(speeds)
    s = -np.concatenate(([0.0], np.cumsum(np.asarray(gaps, dtype=float)[1:] + sim.cfg.car_length)))
    slots = sim._spawn(np.zeros(n, dtype=int), (s / sim.radii[0]) % TWO_PI, np.asarray(speeds, dtype=float))
    sim.kind[slots[smoother]] = SMOOTHER
    sim.exit_at[slots] = np.inf
    return sim, slots


def law_inputs(sim: RingSim, slot: int) -> tuple:
    """(slots, v, v0, ld, g, a_idm) for one car, as _move hands them to _law_smoother."""
    idx = np.flatnonzero(sim.alive)
    lead, gap = sim._index_lanes(idx)
    i = np.flatnonzero(idx == slot)
    v, v0 = sim.v[idx], sim.target_speed(idx)
    a = sim._car_following(idx, v, v0, lead[idx], gap[idx], lead, gap)
    return idx[i], v[i], v0[i], lead[idx][i], gap[idx][i], a[i]


# ------------------------------------------------------------------ settings
def test_defaults_tab_ranges_and_validation():
    cfg = SimConfig()
    assert {name: getattr(cfg, name) for name in NEW} == NEW
    assert [PARAM_BY_NAME[n].group for n in NEW] == ["Smoothers"] * 5
    assert PARAM_BY_NAME["smoother_law"].choices == ("followerstopper", "pi", "quiet")
    for name in NEW:
        spec = PARAM_BY_NAME[name]
        assert not spec.reset, name  # every one applies at once
        assert len(spec.help) <= 340, name
        if not spec.choices:
            assert spec.lo <= NEW[name] <= spec.hi
    assert "Smoothers" in presets.OPT_IN_GROUPS and viewer.TAB_HELP["Smoothers"].startswith("Show the ")
    for bad in (dict(smoothers=-1), dict(smoothers=1.5), dict(smoother_speed=0.0), dict(smoother_start=-1.0),
                dict(smoother_start=float("inf")), dict(smoother_lane=-2), dict(smoother_law="pid")):
        with pytest.raises(ValueError):
            SimConfig(**bad)


@pytest.mark.parametrize("name, config, reason", [
    ("smoothers", dict(), ""),
    ("smoother_law", dict(), "no smoothers"),
    ("smoother_start", dict(smoothers=1), ""),
    ("smoother_speed", dict(smoothers=1), ""),
    ("smoother_speed", dict(smoothers=1, smoother_law="pi"), "no FollowerStopper"),
    ("smoother_speed", dict(smoother_law="quiet"), "no smoothers"),
    ("smoother_lane", dict(smoothers=2), ""),
    ("smoother_lane", dict(smoothers=2, n_lanes=1), "one lane"),
    ("smoother_lane", dict(n_lanes=1), "no smoothers"),
])
def test_each_setting_says_why_it_does_nothing(name, config, reason):
    assert inactive_reason(name, SimConfig(**config)) == reason


def test_every_reason_has_its_fix_and_the_inputs_are_named():
    assert {"no smoothers", "no FollowerStopper", "one lane"} <= set(IDLE_WHY)
    assert set(IDLE_WHY) <= set(experiments.IDLE_FIX) and set(IDLE_WHY) <= set(sv.IDLE_FIX)
    assert inactive_inputs("smoother_law") == inactive_inputs("smoother_start") == {"smoothers"}
    assert inactive_inputs("smoother_speed") == {"smoothers", "smoother_law"}
    assert inactive_inputs("smoother_lane") == {"smoothers", "n_lanes"}
    assert inactive_inputs("smoothers") == set()
    notes = experiments.idle_notes(["smoother_speed"], [SimConfig(smoothers=1, smoother_law="pi", smoother_speed=x)
                                                        for x in (6.5, 7.5)])
    assert len(notes) == 1 and "FollowerStopper" in notes[0] and "smoother_law=followerstopper" in notes[0]
    assert "2 smoothers (PI with saturation from 120 s) round the lanes" in experiments._describe(
        SimConfig(smoothers=2, smoother_law="pi", smoother_start=120))


def test_random_draws_only_when_the_tab_is_named():
    names = set(NEW)
    for seed in range(10):
        assert not set(presets.random_changes(seed)) & names
    assert not set(presets.plausible_ranges()) & names
    for seed in range(40):
        drawn = presets.random_changes(seed, ["Smoothers"])
        assert set(drawn) == names
        cfg = SimConfig(**drawn)
        assert cfg.smoothers in (1, 2, 3, 4) and type(cfg.smoothers) is int and 6.5 <= cfg.smoother_speed <= 8.0
        assert 0 <= cfg.smoother_start <= 300 and cfg.smoother_lane in (-1, 0, 1)
    assert all(presets.RANDOM_RANGES[name].basis.strip() for name in names)


# ------------------------------------------------------------------ the laws against hand values
def test_follower_stopper_boundaries_and_branches():
    """Stern 2018 Eqs. 1-2: at dv = -3 m/s the boundaries are 7.5, 9.75 and 15 m (the paper's worked example); the
    command is 0, then vbar (dx - dx1)/(dx2 - dx1), then vbar + (U - vbar)(dx - dx2)/(dx3 - dx2), then U."""
    assert FS_DX0 == (4.5, 5.25, 6.0) and FS_DECEL == (1.5, 1.0, 0.5)
    U, v, lead = 7.5, 8.0, 5.0  # closing at 3 m/s: dx1, dx2, dx3 = 7.5, 9.75, 15 m; vbar = 5
    dx = np.array([7.0, 7.5, 8.5, 9.75, 12.0, 15.0, 20.0])
    np.testing.assert_allclose(RingSim.follower_stopper(dx, v, lead, U),
                               [0.0, 0.0, 5 * 1.0 / 2.25, 5.0, 5 + 2.5 * 2.25 / 5.25, 7.5, 7.5], rtol=1e-12)
    # not closing (dv >= 0): the boundaries are dx0 themselves; the third branch ramps from vbar (not from v: the
    # paper's, not Flow's U - v)
    np.testing.assert_allclose(RingSim.follower_stopper(np.array([4.0, 4.875, 5.625, 6.5]), 3.0, 6.0, U),
                               [0.0, 3.0, 6.0 + 1.5 * 0.5, 7.5], rtol=1e-12)
    # vbar = min(max(v_lead, 0), U): a leader faster than U counts as U
    assert RingSim.follower_stopper(np.array([4.875]), 9.0, 9.0, U)[0] == pytest.approx(0.5 * U)


def test_tracking_time_constants():
    """a = (v_cmd - v) / tau: 0.73 s accelerating (and down to -0.25 m/s), 0.36 s braking at or below -0.25 m/s
    (Stern's rise times 1.6 / 0.8 s / 2.2); tau at least the step."""
    assert (SMOOTHER_TAU_ACCEL, SMOOTHER_TAU_BRAKE) == (0.73, 0.36)
    dv = np.array([1.0, 0.0, -0.2, -0.25, -2.0])
    np.testing.assert_allclose(RingSim.smoother_tracking(dv, 0.1),
                               [1 / 0.73, 0.0, -0.2 / 0.73, -0.25 / 0.36, -2 / 0.36])
    np.testing.assert_allclose(RingSim.smoother_tracking(dv, 0.5),
                               [1 / 0.73, 0.0, -0.2 / 0.73, -0.25 / 0.5, -2 / 0.5])


def test_pi_with_saturation_against_hand_values():
    """Stern 2018 Eqs. 3-5: v_target = U + clip((dx - 7)/23, 0, 1); alpha = clip((dx - max(2 v, 4))/2, 0, 1); beta =
    1 - alpha/2; v_cmd <- beta (alpha v_target + (1 - alpha) v_lead) + (1 - beta) v_cmd."""
    dx = np.array([20.0, 11.0, 8.0, 40.0])
    v, lead, last = np.array([5.0, 5.0, 5.0, 1.0]), np.array([6.0, 6.0, 6.0, 6.0]), np.array([5.0, 5.0, 5.0, 4.0])
    U = 5.5
    t20, t11 = 5.5 + 13 / 23, 5.5 + 4 / 23
    expected = [0.5 * t20 + 0.5 * 5.0,  # alpha 1 (dx_s = 10 m), beta 1/2
                0.75 * (0.5 * t11 + 0.5 * 6.0) + 0.25 * 5.0,  # alpha 1/2, beta 3/4
                6.0,  # alpha 0: the leader's speed
                0.5 * 6.5 + 0.5 * 4.0]  # dx_s = 4 m at 1 m/s; v_target saturates at U + 1
    np.testing.assert_allclose(RingSim.pi_saturation(dx, v, lead, U, last), expected, rtol=1e-12)


def test_pi_tracks_the_mean_of_its_own_speed_over_the_last_38_s(monkeypatch):
    """U is the mean of the smoother's own speed over the last PI_WINDOW s (exactly: 380 steps), from its pick
    on until then; the command it keeps is the update of its last one, capped at the lane's cornering speed."""
    sim = RingSim(SimConfig(seed=3, **{**STERN, "smoothers": 1, "smoother_law": "pi", "smoother_start": 20.0}))
    s = int(smoothers_of(sim)[0])
    rows = int(round(PI_WINDOW / sim.cfg.dt))
    assert sim.speed_hist.shape == (rows, sim.cap) == (380, sim.cap)
    seen, calls = [], []
    real = sim.pi_saturation
    monkeypatch.setattr(sim, "pi_saturation", lambda dx, v, lead, U, last: calls.append(
        (U.copy(), last.copy(), real(dx, v, lead, U, last))) or real(dx, v, lead, U, last))
    for _ in range(600):
        seen.append(sim.v[s])
        before, was_on, n = sim.v_cmd[s], sim.smoother_on[s], len(calls)
        sim.step()
        if len(calls) == n:  # (before Smoother start: no PI yet, only the history)
            assert sim.time <= 20.1 + 1e-9 and not sim.smoother_on[s]
            continue
        U, last, out = calls[-1]
        assert U[0] == pytest.approx(np.mean(seen[-rows:]), rel=1e-12)
        assert last[0] == (before if was_on else seen[-1])  # (it engages from its speed)
        assert sim.v_cmd[s] == min(out[0], np.sqrt(sim.eff.lateral_accel * sim.radii[0]))
    assert 399 <= len(calls) <= 400 and sim.speed_count[s] == rows
    # a smoother picked later holds only its own speeds since
    sim.set(smoothers=2)
    new = int(np.setdiff1d(smoothers_of(sim), [s])[0])
    assert sim.speed_count[new] == 0
    sim.step(5)
    assert sim.speed_count[new] == 5 and calls[-1][0][list(smoothers_of(sim)).index(new)] == pytest.approx(
        np.mean(sim.speed_hist[(sim.steps - 1 - np.arange(5)) % rows, new]), rel=1e-12)


def test_the_gap_cap_hands_over_to_the_drivers_idm():
    """Beyond max(120 m, 6 s x v) to the car ahead the smoother follows its driver's IDM (Lichtle's cap)."""
    for gap, v, capped in ((150.0, 5.0, True), (110.0, 5.0, False), (140.0, 25.0, False), (160.0, 25.0, True)):
        sim, slots = lane_with([0.0, gap], [5.0, v], smoother=1)
        args = law_inputs(sim, slots[1])
        a = sim._law_smoother(*args)
        law = RingSim.smoother_tracking(RingSim.follower_stopper(args[4], args[1], args[1] * 0 + 5.0, 7.5) - args[1],
                                        sim.cfg.dt)
        np.testing.assert_allclose(a, args[5] if capped else law, rtol=1e-12)


def test_a_tracking_smoother_acts_without_a_reaction_delay_and_passes_the_clip():
    """From Smoother start its command acts in the same step (no reaction delay, no noise), clipped to the braking
    limit; before it, its driver drives (reaction delay, noise, no clip)."""
    sim, slots = lane_with([0.0, 30.0], [8.0, 8.0], smoother=1, reaction_time=1.0, driver_noise=0.3)
    sim.step()
    v_before = sim.v[slots[1]]
    slot, v, v0, ld, g, a_idm = law_inputs(sim, slots[1])
    a = np.clip(RingSim.smoother_tracking(RingSim.follower_stopper(g, v, sim.v[ld], 7.5) - v, 0.1), -9.0,
                sim.accel[slot])
    sim.step()
    assert sim.a_now[slots[1]] == pytest.approx(a[0], rel=1e-9)
    assert sim.v[slots[1]] == pytest.approx(v_before + 0.1 * a[0])
    # the clip and the safety layer: a smoother's -20 m/s2 becomes the braking limit, once it tracks
    acc = np.array([-20.0])
    sim._automated_limits(slots[1:], acc, np.array([-1.0]))
    assert acc[0] == -sim.eff.max_decel
    late, ls = lane_with([0.0, 30.0], [8.0, 8.0], smoother=1, smoother_start=100.0)
    acc = np.array([-20.0])
    late._automated_limits(ls[1:], acc, np.array([-1.0]))
    assert acc[0] == -20.0  # (its driver's: untouched, as any driver's)


def test_quiet_is_its_drivers_model_without_noise():
    """The placebo: _car_following gives it its driver's multi-anticipation, _law_smoother its driver's command;
    the noise is masked; FollowerStopper and PI smoothers see only the car in front."""
    sim = RingSim(SimConfig(seed=4, n_lanes=1, n_arms=0, density=60, smoothers=1, smoother_law="quiet"))
    kinds = np.array([HUMAN, SMOOTHER])
    assert sim._anticipates(kinds).tolist() == [True, True] and sim._driven(kinds).tolist() == [True, False]
    s = smoothers_of(sim)
    args = law_inputs(sim, s[0])
    np.testing.assert_array_equal(sim._law_smoother(*args), args[5])
    sim.set(smoother_law="followerstopper")
    assert sim._anticipates(kinds).tolist() == [True, False]


# ------------------------------------------------------------------ selection, live changes
def test_selection_is_deterministic_and_takes_the_drivers_nearest_evenly_spaced_angles():
    cfg = SimConfig(seed=5, truck_share=0.3, smoothers=5)  # 3 lanes: round-robin from the innermost
    sim, plain = RingSim(cfg), RingSim(cfg.updated(smoothers=0))
    for name in ("rng", "rng_demand", "rng_drivers", "rng_kinds", "rng_kinds_test"):  # (no draws)
        assert getattr(sim, name).bit_generator.state == getattr(plain, name).bit_generator.state, name
    np.testing.assert_array_equal(sim.theta, plain.theta)
    chosen = set()
    for k in range(5):  # an independent pick: lane k mod 3, angle 2 pi k / 5, a driver's car (no truck)
        cand = np.flatnonzero(plain.alive & (plain.lane == k % 3) & ~plain.truck)
        cand = cand[[c not in chosen for c in cand]]
        chosen.add(int(cand[np.argmin(np.abs((plain.theta[cand] - TWO_PI * k / 5 + np.pi) % TWO_PI - np.pi))]))
    assert set(smoothers_of(sim).tolist()) == chosen and not sim.truck[smoothers_of(sim)].any()
    s = smoothers_of(sim)
    assert (sim.company[s] == -1).all() and np.isinf(sim.exit_at[s]).all()
    for name in ("z", "z_gap", "z_react", "headway", "delay", "s0", "accel"):  # (its driver stays: _set_driving)
        np.testing.assert_array_equal(getattr(sim, name)[s], getattr(plain, name)[s], err_msg=name)
    np.testing.assert_array_equal(sim.desired_speed(s), plain.desired_speed(s))  # (its own free speed: metrics)
    one = RingSim(cfg.updated(smoothers=3, smoother_lane=7))  # a lane beyond the ring's: the outermost
    assert (one.lane[smoothers_of(one)] == 2).all() and smoothers_of(one).size == 3


def test_a_live_change_converts_or_hands_back_the_nearest():
    sim = RingSim(SimConfig(seed=6, smoothers=2, density=40, exit_prob=0.2))
    sim.step(300)
    first = smoothers_of(sim)
    states = [g.bit_generator.state for g in (sim.rng, sim.rng_drivers, sim.rng_kinds)]
    gen = sim.generation
    assert not sim.set(smoothers=5)
    assert sim.generation == gen and set(first) <= set(smoothers_of(sim)) and smoothers_of(sim).size == 5
    assert [g.bit_generator.state for g in (sim.rng, sim.rng_drivers, sim.rng_kinds)] == states  # (no draws)
    sim.step(50)
    five = smoothers_of(sim)
    ref = sim.theta[five[np.argmin(sim.uid[five])]]  # (the angle the places are spaced from)
    inner = five[sim.lane[five] == 0]
    kept = int(inner[np.argmin(np.abs((sim.theta[inner] - ref + np.pi) % TWO_PI - np.pi))])
    assert not sim.set(smoothers=1)  # one place, in the innermost lane: the smoother there nearest it stays
    assert smoothers_of(sim).tolist() == [kept]
    back = np.setdiff1d(five, [kept])
    assert (sim.kind[back] == HUMAN).all() and np.isfinite(sim.exit_at[back]).all()  # (routes of their own again)
    assert not sim.set(smoother_lane=2)  # picked again, in the outer lane
    assert sim.lane[smoothers_of(sim)].tolist() == [2] and sim.kind[kept] == HUMAN
    assert not sim.set(smoothers=0)
    assert smoothers_of(sim).size == 0 and not sim.automated and sim.speed_hist.shape == (0, sim.cap)
    sim.step(20)


# ------------------------------------------------------------------ never leave, change lane or get removed
def test_smoothers_never_leave_change_lane_or_get_removed():
    sim = RingSim(SimConfig(seed=7, smoothers=4, exit_prob=0.5, lane_change_rate=1.0, lane_change_motive="random",
                            density=45))
    s, lanes = smoothers_of(sim), sim.lane[smoothers_of(sim)].copy()
    for _ in range(30):
        sim.step(50)
        assert set(smoothers_of(sim)) == set(s) and (sim.lane[s] == lanes).all() and np.isinf(sim.exit_at[s]).all()
    assert sim.n_lane_changes > 0 and sim.n_exited > 0
    sim.set(exit_prob=0.3)  # every other car is routed again; a smoother's exit stays inf
    assert np.isinf(sim.exit_at[s]).all() and np.isfinite(sim.exit_at[sim.alive & (sim.kind == HUMAN)]).any()
    sim.set(density=5)  # closed mode: the density control removes cars, never a smoother
    sim.step()
    assert set(smoothers_of(sim)) == set(s) and sim.alive.sum() < 20
    openly = RingSim(SimConfig(seed=8, smoothers=3, traffic_mode="open", density=40))
    s = smoothers_of(openly)
    openly.set(density=4)  # open mode: removed at once (_sync_ring), never a smoother
    assert set(smoothers_of(openly)) == set(s) and openly.alive.sum() == openly.target_cars()
    test = RingSim(SimConfig(seed=9, smoothers=2, traffic_mode="open", subject_arm="on", entry_rate=0.05))
    test.step(200)
    assert np.isinf(test.exit_at[smoothers_of(test)]).all() and smoothers_of(test).size == 2


# ------------------------------------------------------------------ Smoother start
def test_before_smoother_start_a_smoother_drives_exactly_as_its_driver():
    """On the stern2018 ring (no exits, lane changes or density changes) a run with a smoother that starts at 150 s
    is the human run bit for bit until then (noise, multi-anticipation, reaction delay, no clip), and not after."""
    human = RingSim(SimConfig(seed=10, **STERN))
    smooth = RingSim(SimConfig(seed=10, **{**STERN, "smoothers": 1, "smoother_start": 150.0}))
    s = smoothers_of(smooth)
    for _ in range(1500):
        human.step()
        smooth.step()
    np.testing.assert_array_equal(smooth.theta, human.theta)
    np.testing.assert_array_equal(smooth.v, human.v)
    assert smooth.rng.bit_generator.state == human.rng.bit_generator.state and not smooth.smoother_on.any()
    assert smooth.speed_count[s] == 380  # (the history holds the last 38 s only)
    human.step(20)
    smooth.step(20)
    assert smooth.smoother_on[s].all() and not np.array_equal(smooth.v, human.v)


# ------------------------------------------------------------------ saved states
@pytest.mark.parametrize("config", [
    {**STERN, "smoothers": 1, "smoother_law": "pi", "smoother_start": 30.0},
    dict(smoothers=3, smoother_start=80.0, traffic_mode="open", entry_rate=0.2, truck_share=0.1),
    dict(smoothers=2, smoother_law="quiet", smoother_lane=1, density=60),
], ids=["pi-stern", "fs-open", "quiet"])
def test_a_run_with_smoothers_resumes_exactly(config, tmp_path):
    sim = RingSim(SimConfig(seed=11, **config))
    sim.step(500)  # (after the PI's start: its history and command under way; the others before theirs)
    assert_exact_resume(sim, tmp_path, more=600)  # (on past their starts)
    assert sim.smoother_on.any() == (config.get("smoother_law") != "quiet")


def test_a_damaged_speed_history_is_refused():
    sim = RingSim(SimConfig(seed=12, smoothers=1))
    sim.step(10)
    st = sim.state_dict()
    st["arrays"]["speed_hist"] = st["arrays"]["speed_hist"][:-1]
    with pytest.raises(StateError, match="speed_hist has shape"):
        RingSim.from_state(st)
    st = sim.state_dict()
    st["arrays"]["speed_count"][0] = 10_000
    with pytest.raises(StateError, match="speed_count"):
        RingSim.from_state(st)


# ------------------------------------------------------------------ smoothers 0
@pytest.mark.parametrize("name", sorted(OTHER))
def test_without_smoothers_their_settings_change_nothing(name):
    def run(cfg):
        sim = RingSim(cfg)
        sim.step(300)
        live = np.flatnonzero(sim.alive)
        return sim.uid[live].tolist(), sim.theta[live].tolist(), sim.v[live].tolist(), sim.n_emergency

    for base in (SimConfig(seed=3, truck_share=0.1, headway_std=0.3),
                 SimConfig(seed=3, auto_share=0.5, auto_kind="cav", companies=2)):
        assert run(base.updated(**{name: OTHER[name]})) == run(base), name
        assert not RingSim(base).smoothing


# ------------------------------------------------------------------ the viewer
def test_the_viewer_takes_every_value_and_shows_a_smoother(make_app, monkeypatch):
    app = make_app(smoothers=2, smoother_start=1.0)
    frames(app, 2)
    show_tab(app, "Smoothers")
    frames(app, 1)
    assert app.content_h <= app.content_rect.h  # (the tab fits without scrolling)
    for name in NEW:
        assert widget(app, "choice" if PARAM_BY_NAME[name].choices else "slider", name).rect.w > 0
    for name in NEW:
        spec = PARAM_BY_NAME[name]
        for value in spec.choices or np.linspace(spec.lo, spec.hi, 4).tolist():
            app.apply(name, value)
            frames(app, 1)
        app.apply(name, NEW[name] if name != "smoothers" else 2)
    for text in viewer.help_texts():  # (the help box still fits the longest help)
        assert len(app.fonts.wrap(text, app.status_rect.w)) <= app.status_lines, text[:60]
    app.apply("smoother_start", 0.0)
    frames(app, 30)
    app.paused = True
    sim, ring = app.sim, app.ring
    snap = sim.cars()
    on_top = []  # (the smoothers the mouse finds on their own pixel: not under a neighbour)
    for i in np.flatnonzero(snap["kind"] == SMOOTHER):
        app.mouse = (round(ring.cx + snap["x"][i] * ring.k), round(ring.cy - snap["y"][i] * ring.k))
        if app._car_at(app.mouse) == int(snap["slot"][i]):
            on_top.append((int(snap["slot"][i]), app.mouse))
    assert on_top
    slot, app.mouse = on_top[0]
    texts = rendered_texts(app, monkeypatch)
    assert "automated: smoother  ·  standstill gap 2.00 m" in texts
    assert any(t.startswith("smoother: FollowerStopper at 7.5 m/s, command") for t in texts), texts
    app.apply("smoother_start", 300.0)
    assert app._smoother_line(slot) == "smoother: its driver drives until 300 s"


# ------------------------------------------------------------------ behaviour: Stern's ring
def test_one_follower_stopper_damps_the_waves_of_the_stern_ring():
    """Stern 2018 Table 3, Exp. A: one FollowerStopper among 21 cars, switched on after the waves formed, cut the
    speed SD from 3.31 to 0.64 m/s. On the C1-calibrated ring (first waves ~116 s), switched on at 150 s, measured
    300-600 s (3 seeds): at the default 7.5 m/s it cuts speed_sd_pooled by well over a quarter and beats the
    placebo (its driver without noise); at 6.5 m/s by about three quarters (this model's waves run slower than
    Stern's, so its best set speed is lower: readme_notes S2)."""
    def sd(changes, seed):
        return experiments._sweep_job(({**STERN, **changes}, seed, 300.0, 300.0, 0.5))["speed_sd_pooled"]

    on = dict(smoothers=1, smoother_start=150.0)
    seeds = (0, 1, 2)
    human = np.mean([sd({}, k) for k in seeds])
    fs = np.mean([sd(on, k) for k in seeds])
    slow = [sd({**on, "smoother_speed": 6.5}, k) for k in seeds]
    quiet = np.mean([sd({**on, "smoother_law": "quiet"}, k) for k in seeds])
    assert human > 2.0  # (waves)
    assert fs < 0.7 * human and fs < quiet
    assert np.mean(slow) < 0.35 * human and max(slow) < 1.0
