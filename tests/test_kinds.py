"""Vehicle kinds and companies (SPEC 1, stage S1): the kind draws of rng_kinds at every creation site and their
monotone coupling (at the same seed the automated set at a lower share is a subset of that at a higher one,
and the human streams draw exactly the same), the queues carrying each vehicle's kind to its entry, the
automated vehicles' masked temperaments and noise and their direct-leader following, the settings (live,
rebuilding or idle; why they can do nothing) and auto_share = 0 being the model without kinds, bit for bit."""

from __future__ import annotations

import numpy as np
import pytest

from roundabout_sim import presets
from roundabout_sim.config import IDLE_WHY, PARAM_BY_NAME, PARAMS, SimConfig, inactive_reason
from roundabout_sim.engine import (ACC, ACC_GAP_SHARES, ACC_GAPS, AUTO_KINDS, AUTO_LAWS, CAV, CONNECTED, DRIVERLESS,
                                   GENERATORS, HUMAN, KIND_NAMES, QUEUE_FIELDS, RECOGNISABLE, ROBOTAXI, SMOOTHER,
                                   SUBJECT_ARM, SUBJECT_QUEUE, RingSim)
from tests.helpers import assert_valid

OPEN = dict(traffic_mode="open", entry_rate=0.15, n_lanes=2, truck_share=0.1)
HUMAN_STREAMS = ("rng", "rng_demand", "rng_drivers")


class Recording(RingSim):
    """A RingSim that keeps every (kind, company, gap) _draw_kinds gave (``drawn``; the capacity test's top-up's,
    from rng_kinds_test, ``drawn_test``), every queue field it filled
    (arm -> values) and took off (arm -> values), and every entrant's (kind given, kind it got)."""

    def _draw_kinds(self, k, rng=None):
        out = super()._draw_kinds(k, rng)
        self.__dict__.setdefault("drawn" if rng is None else "drawn_test", []).append(out)  # (rng: the top-up's)
        return out

    def _queue_kinds(self, arm, kinds, k=None):
        super()._queue_kinds(arm, kinds, k)
        joined = list(zip(*(values[:k].tolist() for values in kinds)))
        self.__dict__.setdefault("queued", {}).setdefault(int(arm), []).extend(joined)

    def _pop_kinds(self, arm):
        out = super()._pop_kinds(arm)
        self.__dict__.setdefault("popped", {}).setdefault(int(arm), []).append(tuple(v[0].item() for v in out))
        return out

    def _spawn(self, *args, trucks=None, kinds=None, **kw):
        slots = super()._spawn(*args, trucks=trucks, kinds=kinds, **kw)
        if kinds is not None:
            self.__dict__.setdefault("entrants", []).append((int(self.kind[slots[0]]), int(kinds[0][0])))
        return slots


def drawn(sim: Recording) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every kind, company and gap the run drew, in the order the vehicles were created."""
    return tuple(np.concatenate([d[j] for d in sim.drawn]) for j in range(3))


def kind_uniforms(sim: RingSim) -> int:
    """How many rng_kinds uniforms the vehicles created so far drew: three each. Created: every vehicle placed
    on the ring (initial placement, density fills, entrants) and every one that joined a queue (an arrival,
    blocked ones included, a closed-mode replacement), each once: an entrant was created when it joined. (A
    capacity-test top-up draws its three from rng_kinds_test instead.)"""
    return 3 * (sim._next_uid - sim.n_entered + sim.n_arrived)


def assert_drew(sim: RingSim, n: int, topped_up: int = 0) -> None:
    """rng_kinds has drawn exactly ``n`` uniforms since the seed, and rng_kinds_test ``topped_up``."""
    for name, key, count in (("rng_kinds", [sim.seed, 3], n), ("rng_kinds_test", [sim.seed, 3, 1], topped_up)):
        fresh = np.random.default_rng(key)
        fresh.random(count)
        assert getattr(sim, name).bit_generator.state == fresh.bit_generator.state, name


# ------------------------------------------------------------------ the kinds
def test_kind_codes_names_and_sets():
    assert (HUMAN, ACC, CAV, ROBOTAXI, SMOOTHER) == (0, 1, 2, 3, 4)
    assert KIND_NAMES == ("human", "acc", "cav", "robotaxi", "smoother")
    assert AUTO_KINDS == {"acc": ACC, "cav": CAV, "robotaxi": ROBOTAXI}
    assert set(AUTO_KINDS) == set(PARAM_BY_NAME["auto_kind"].choices) and SimConfig().auto_kind == "cav"
    assert RECOGNISABLE == {CAV, ROBOTAXI, SMOOTHER} and DRIVERLESS == {CAV, ROBOTAXI} and CONNECTED == {CAV}
    assert set(AUTO_LAWS) == set(AUTO_KINDS.values()) | {SMOOTHER}  # every automated kind has a law, smoothers too
    assert {"rng_kinds", "rng_kinds_test"} <= set(GENERATORS)
    assert {"queue_kinds", "queue_companies", "queue_acc_gaps"} <= set(QUEUE_FIELDS)
    assert ACC_GAPS == (1.1, 1.6, 2.2) and sum(ACC_GAP_SHARES) == pytest.approx(1.0)


def test_the_draws_give_the_shares_companies_and_gaps():
    """u1 < auto_share automates; companies are equally likely; the fallback gaps come in Liu 2018's shares."""
    sim = RingSim(SimConfig(seed=1, auto_share=0.3, auto_kind="robotaxi", companies=3, density=1))
    kind, company, gap = sim._draw_kinds(30_000)
    auto = kind != HUMAN
    assert set(np.unique(kind)) == {HUMAN, ROBOTAXI} and kind.dtype == company.dtype == np.int8
    assert auto.mean() == pytest.approx(0.3, abs=0.01)
    assert (company[~auto] == -1).all() and set(np.unique(company[auto])) == {0, 1, 2}
    assert np.bincount(company[auto]) / auto.sum() == pytest.approx([1 / 3] * 3, abs=0.02)
    assert np.isin(gap, ACC_GAPS).all()
    assert [np.mean(gap == g) for g in ACC_GAPS] == pytest.approx(list(ACC_GAP_SHARES), abs=0.01)
    one = RingSim(SimConfig(seed=1, auto_share=1.0, density=1))
    assert (one._draw_kinds(500)[0] == CAV).all()  # (the default kind; a share of 1 automates every vehicle)


def test_auto_share_zero_draws_nothing_and_is_the_model_without_kinds():
    """With auto_share 0 the other AV settings change nothing at all: no draw, every vehicle a human driver,
    the same run bit for bit (the golden fingerprints check the same for 12 configurations)."""
    plain = RingSim(SimConfig(seed=3, **OPEN))
    other = RingSim(SimConfig(seed=3, auto_kind="robotaxi", robotaxi_headway=1.6, companies=4, **OPEN))
    for sim in (plain, other):
        sim.step(800)
        assert not sim.automated and (sim.kind == HUMAN).all() and (sim.company == -1).all()
        assert np.isnan(sim.acc_gap).all() and (sim.s0[sim.alive] == sim.cfg.min_gap).all()
        assert_drew(sim, 0)
    for name in ("uid", "lane", "theta", "v", "truck", "a_now"):
        np.testing.assert_array_equal(getattr(plain, name)[plain.alive], getattr(other, name)[other.alive])
    assert plain.n_entered == other.n_entered > 0 and list(plain.conflict_log) == list(other.conflict_log)
    assert all(kind == HUMAN for *_, kind in plain.conflict_log)
    assert any(plain.queue_kinds) and all(set(fifo) <= {HUMAN} for fifo in plain.queue_kinds)


# --------------------------------------------------------------- monotone coupling
@pytest.mark.parametrize("kind", ["cav", "robotaxi", "acc"])
def test_a_higher_share_automates_the_same_vehicles_and_more(kind):
    """Open mode (initial placement, then every arrival): the n-th vehicle created draws the same three
    uniforms whatever the share, so the automated set at 0.2 is a subset of that at 0.4, with the same
    companies and gaps; and every vehicle on the ring at the start keeps its slot, so the subset holds by
    vehicle too."""
    sims = [Recording(SimConfig(seed=11, auto_share=share, auto_kind=kind, companies=3, max_queue=4, **OPEN))
            for share in (0.2, 0.4)]
    start = [sim.kind.copy() for sim in sims]
    for sim in sims:
        sim.step(1500)
        assert sim.n_blocked > 0  # (blocked arrivals draw too)
    (k1, c1, g1), (k2, c2, g2) = drawn(sims[0]), drawn(sims[1])
    assert k1.size == k2.size == kind_uniforms(sims[0]) // 3 == kind_uniforms(sims[1]) // 3 > 120
    a1, a2 = k1 != HUMAN, k2 != HUMAN
    assert (a1 <= a2).all() and a1.sum() < a2.sum()  # a subset, and a proper one
    assert a1.mean() == pytest.approx(0.2, abs=0.1) and a2.mean() == pytest.approx(0.4, abs=0.1)  # (~140 draws)
    np.testing.assert_array_equal(c1[a1], c2[a1])  # the same companies...
    np.testing.assert_array_equal(g1, g2)  # ...and fallback gaps
    assert set(np.unique(k2)) == {HUMAN, AUTO_KINDS[kind]}
    placed = start[0] != HUMAN
    assert placed.any() and (placed <= (start[1] != HUMAN)).all()  # by slot (= uid) at the start


def test_the_human_streams_draw_the_same_whatever_the_share():
    """Automated vehicles consume every draw a human in their place would (temperaments, trucks, cooldowns,
    routes) and mask the values afterwards: after the start every human stream is where it is without them,
    and the humans' values are the same; the arrivals (their own stream) are the same throughout."""
    base = dict(seed=5, truck_share=0.2, speed_std=1.5, headway_std=0.3, reaction_std=0.2, **dict(
        traffic_mode="open", entry_rate=0.2, n_lanes=3))
    plain, auto = RingSim(SimConfig(**base)), RingSim(SimConfig(**base, auto_share=0.4, auto_kind="robotaxi"))
    for name in HUMAN_STREAMS:
        assert plain.__dict__[name].bit_generator.state == auto.__dict__[name].bit_generator.state, name
    np.testing.assert_array_equal(plain.alive, auto.alive)
    human, robot = auto.alive & (auto.kind == HUMAN), auto.alive & (auto.kind == ROBOTAXI)
    assert human.any() and robot.any()
    for name in ("truck", "lc_cool", "exit_at", "theta", "uid"):
        np.testing.assert_array_equal(getattr(plain, name)[plain.alive], getattr(auto, name)[auto.alive])
    for name in ("z", "z_react", "z_gap"):  # the humans' own; the robotaxis' masked to 0 (but their time gap's)
        np.testing.assert_array_equal(getattr(plain, name)[human], getattr(auto, name)[human])
        if name == "z_gap":  # a robotaxi keeps the time-gap temperament it drew (Hu et al. 2023 Sec. III-B)
            np.testing.assert_array_equal(getattr(auto, name)[robot], getattr(plain, name)[robot])
        else:
            assert (getattr(auto, name)[robot] == 0).all() and (getattr(plain, name)[robot] != 0).any()
    for sim in (plain, auto):
        sim.step(600)
    assert plain.rng_demand.bit_generator.state == auto.rng_demand.bit_generator.state
    assert list(plain.arrival_log) == list(auto.arrival_log) and plain.n_arrived == auto.n_arrived


@pytest.mark.parametrize("mode", ["closed", "open", "open-subject"])
def test_every_new_vehicle_draws_three_uniforms_where_it_is_created(mode):
    """Initial placement, density fills (closed mode, a live Density raise in open mode, respacing near jam
    density), closed-mode replacements that join a queue, every arrival (blocked ones too) and the capacity
    test's top-up each draw three rng_kinds uniforms per vehicle (the top-up from rng_kinds_test, a stream of
    its own); an entrant draws none (its queue carries what it drew when it joined)."""
    if mode == "closed":
        cfg = SimConfig(seed=21, auto_share=0.3, exit_prob=0.4, n_lanes=2, truck_share=0.15)
    elif mode == "open":
        cfg = SimConfig(seed=22, auto_share=0.3, auto_kind="acc", max_queue=3, entry_rate=0.4, **{
            k: v for k, v in OPEN.items() if k != "entry_rate"})
    else:
        cfg = presets.apply("us_single_lane", SimConfig(seed=23)).updated(  # (a preset keeps only the seed)
            auto_share=0.3, auto_kind="robotaxi", subject_arm="on", entry_rate=0.1)
    sim = RingSim(cfg)
    assert_drew(sim, kind_uniforms(sim), 3 * sim.n_topped_up)
    sim.step(700)
    sim.set(density=95.0)  # closed: filled over the next steps; open: placed at once
    sim.step(300)
    assert_drew(sim, kind_uniforms(sim), 3 * sim.n_topped_up)
    assert sim.n_entered > 20 and sim.n_arrived > 20
    if mode == "closed":
        assert sim.inserted_at[sim.alive].max() > 0  # fills happened
    if mode == "open":
        assert sim.n_blocked > 0
    if mode == "open-subject":
        assert sim.n_topped_up > SUBJECT_QUEUE and (np.asarray(sim.queue_kinds[SUBJECT_ARM]) == ROBOTAXI).any()
    assert_valid(sim)


@pytest.mark.parametrize("mode", ["closed", "open", "open-subject"])
def test_queues_carry_each_vehicles_kind_to_its_entry(mode):
    """What joins a queue (kind, company, gap) leaves it in the same order and becomes the entrant's; what is
    still queued is the rest, and every queue field is as long as its queue."""
    extra = dict(auto_share=0.5, auto_kind="cav", companies=2)
    if mode == "closed":
        cfg = SimConfig(seed=31, exit_prob=0.5, n_lanes=1, **extra)
    elif mode == "open":
        cfg = SimConfig(seed=32, follow_up=3.0, max_queue=8, **{**OPEN, "entry_rate": 0.3}, **extra)
    else:
        cfg = presets.apply("us_single_lane", SimConfig(seed=33)).updated(subject_arm="on", entry_rate=0.08, **extra)
    sim = Recording(cfg)
    sim.step(1200)
    assert sum(len(p) for p in sim.popped.values()) == sim.n_entered > 30
    for arm, queued in sim.queued.items():
        popped = sim.popped.get(arm, [])
        assert queued[:len(popped)] == popped
        rest = list(zip(sim.queue_kinds[arm], sim.queue_companies[arm], sim.queue_acc_gaps[arm]))
        assert rest == queued[len(popped):len(popped) + int(sim.queue[arm])]
        for name in QUEUE_FIELDS:
            assert len(getattr(sim, name)[arm]) == sim.queue[arm]
    assert all(got == given for got, given in sim.entrants)
    assert {given for _, given in sim.entrants} == {HUMAN, CAV}
    if mode == "open-subject":
        assert len(sim.popped[SUBJECT_ARM]) == sim.arm_entered[SUBJECT_ARM] > 5


def test_a_live_max_queue_cut_drops_the_kinds_with_their_cars():
    sim = RingSim(SimConfig(seed=41, auto_share=0.5, follow_up=4.0, max_queue=40, **{**OPEN, "entry_rate": 0.5}))
    sim.step(600)
    assert sim.queue.max() > 5
    sim.set(max_queue=2)
    for name in ("queue_times", *QUEUE_FIELDS):
        assert [len(fifo) for fifo in getattr(sim, name)] == sim.queue.tolist(), name
    assert_valid(sim)


# --------------------------------------------------------------- automated driving
def test_automated_vehicles_are_one_software_without_noise():
    """Desired speed, time gap and reaction are the settings themselves for every automated vehicle (their
    temperaments are drawn, then masked), and the driver noise reaches only the human drivers: a ring of
    automated cars runs the same with any noise (drawn all the same, then masked)."""
    cfg = SimConfig(seed=51, auto_share=0.5, auto_kind="acc", speed_std=2.0, headway_std=0.4, reaction_std=0.3,
                    reaction_time=0.6, driver_noise=0.8, n_lanes=1, n_arms=0, density=40)
    sim = RingSim(cfg)
    auto = sim.alive & (sim.kind == ACC)
    human = sim.alive & (sim.kind == HUMAN)
    assert auto.sum() > 5 and human.sum() > 5
    slots = np.flatnonzero(auto)
    assert np.unique(sim.desired_speed(slots)).size == 1 and np.unique(sim.headway[auto]).size == 1
    # (an ACC car: ACC time gap and its dead time ACC dead time, 0.5 s, not a driver's: the controllers' settings)
    assert sim.headway[auto][0] == sim.cfg.acc_headway and (sim.delay[auto] == 5).all()
    assert np.unique(sim.headway[human]).size > 1 and np.unique(sim.delay[human]).size > 1
    runs = [RingSim(cfg.updated(auto_share=1.0, driver_noise=noise)) for noise in (0.0, 0.8)]
    for run in runs:
        run.step(400)
    assert (runs[1].noise[runs[1].alive] != 0).all()  # the noise process runs for every car...
    np.testing.assert_array_equal(runs[0].v, runs[1].v)  # ...and moves none of them
    np.testing.assert_array_equal(runs[0].theta, runs[1].theta)
    humans = [RingSim(cfg.updated(auto_share=0.0, driver_noise=noise)) for noise in (0.0, 0.8)]
    for run in humans:
        run.step(400)
    assert not np.array_equal(humans[0].v, humans[1].v)


def test_automated_vehicles_follow_their_direct_leader_only():
    """_car_following gives an automated car the plain IDM towards the car in front (its own a, T and s0)
    whatever the anticipation; human drivers still anticipate (so they differ from it in dense traffic)."""
    sim = RingSim(SimConfig(seed=61, auto_share=0.5, auto_kind="robotaxi", n_lanes=1, n_arms=0, density=85,
                            init_spacing="random", reaction_distance=60.0, anticipation=5))
    sim.step(100)
    idx = np.flatnonzero(sim.alive)
    lead, gap = sim._index_lanes(idx)
    v, v0, ld, g = sim.v[idx], sim.target_speed(idx), lead[idx], gap[idx]
    following = sim._car_following(idx, v, v0, ld, g, lead, gap)
    plain = sim._idm(v, v0, g, sim.v[ld], idx)
    robot = sim.kind[idx] == ROBOTAXI
    np.testing.assert_array_equal(following[robot], plain[robot])
    assert not np.allclose(following[~robot], plain[~robot])


def test_the_automated_command_is_a_dispatch_by_kind(monkeypatch):
    """_move asks _automated_command for the automated cars' commands only while there can be automated
    vehicles; each kind goes to its law (AUTO_LAWS), with that kind's cars only."""
    calls = []
    sim = RingSim(SimConfig(seed=71, auto_share=0.5, auto_kind="cav", n_arms=0, n_lanes=1, density=40))
    real = sim._law_cav  # (a CAV's law: PATH CACC behind a cooperating CAV, else the ACC law)
    monkeypatch.setattr(sim, "_law_cav", lambda slots, *a: calls.append(slots.copy()) or real(slots, *a))
    sim.step()
    assert len(calls) == 1 and (sim.kind[calls[0]] == CAV).all()
    assert calls[0].size == int((sim.alive & (sim.kind == CAV)).sum())
    plain = RingSim(SimConfig(seed=71, n_arms=0, n_lanes=1, density=40))
    monkeypatch.setattr(plain, "_automated_command", lambda *a: pytest.fail("called without automated vehicles"))
    plain.step(3)


def test_a_now_is_the_acceleration_each_car_realised():
    sim = RingSim(SimConfig(seed=81, auto_share=0.3, density=60))
    sim.step(100)
    v_before, alive = sim.v.copy(), sim.alive.copy()
    sim.step()
    same = alive & sim.alive
    assert sim.a_now[same] == pytest.approx((sim.v[same] - v_before[same]) / sim.cfg.dt)
    slot = sim._spawn(np.array([0]), np.array([1.0]), np.array([5.0]))
    assert sim.a_now[slot[0]] == 0.0  # reset at spawn


# ------------------------------------------------------------------ settings
def test_the_new_settings_rebuild_change_live_or_wait():
    sim = RingSim(SimConfig(seed=91))
    sim.step(20)
    gen = sim.generation
    assert not sim.set(auto_kind="robotaxi", companies=3)  # no automated vehicle before or after: kept for later
    assert sim.generation == gen and sim.cfg.auto_kind == "robotaxi" and sim.cfg.companies == 3
    assert sim.set(auto_share=0.5) and sim.generation == gen + 1  # rebuilds, with the kinds drawn
    assert (sim.kind[sim.alive] == ROBOTAXI).any() and set(np.unique(sim.company[sim.alive])) <= {-1, 0, 1, 2}
    t = sim.time
    sim.step(10)
    robot = sim.alive & (sim.kind == ROBOTAXI) & ~sim.truck
    assert not sim.set(robotaxi_headway=1.5)  # live: the robotaxis' time gap follows at once
    assert sim.time > t and sim.headway[robot] == pytest.approx(1.5 * sim.eff.time_headway)
    assert sim.set(auto_kind="cav") and (sim.kind[sim.alive] != ROBOTAXI).all()  # rebuilds: a new run
    assert sim.set(companies=2)
    before = sim.rng_kinds.bit_generator.state
    assert not sim.set(seed=1234)  # re-seeds every stream from now on, the kinds' too
    assert sim.rng_kinds.bit_generator.state == np.random.default_rng([1234, 3]).bit_generator.state != before


def test_settings_say_why_they_do_nothing():
    """Robotaxi time gap has one reason whenever there are no robotaxis, for want of automated vehicles or of
    that kind (so a sweep mixing both causes still warns that it does nothing)."""
    plain = SimConfig()
    for name in ("auto_kind", "companies"):
        assert inactive_reason(name, plain) == "no automated vehicles"
    assert inactive_reason("auto_share", plain) == ""
    cav = SimConfig(auto_share=0.3)
    assert inactive_reason("auto_kind", cav) == ""
    # (Companies acts once the companies drive differently: brand_spread or coop_between, config.companies_differ)
    assert inactive_reason("companies", cav) == "companies alike"
    assert inactive_reason("companies", cav.updated(brand_spread=0.2)) == ""
    for cfg in (plain, plain.updated(auto_kind="robotaxi"), cav):
        assert inactive_reason("robotaxi_headway", cfg) == "no robotaxis"
    assert inactive_reason("robotaxi_headway", cav.updated(auto_kind="robotaxi")) == ""
    assert {"no automated vehicles", "no robotaxis"} <= set(IDLE_WHY)


@pytest.mark.parametrize("bad", [dict(auto_share=1.5), dict(auto_share=-0.1), dict(auto_kind="bus"),
                                 dict(companies=0), dict(companies=1.5), dict(companies=200),
                                 dict(robotaxi_headway=0.0)])
def test_invalid_values_are_refused(bad):
    with pytest.raises(ValueError):
        SimConfig(**bad)


def test_random_settings_leave_the_av_tabs_alone_unless_named():
    """--random / --sample draw every group but AVs and Coop (and Entries, Smoothers) by default (human-only designs
    reproduce); named, those are drawn like any other, on their slider steps."""
    assert presets.OPT_IN_GROUPS == ("AVs", "Coop", "Entries", "Smoothers") and set(presets.DEFAULT_GROUPS) == \
        set(presets.GROUPS) - {"AVs", "Coop", "Entries", "Smoothers"}
    av_names = {p.name for p in PARAMS if p.group in ("AVs", "Coop")}  # (the ACC/CACC and cooperation ones too)
    assert {"auto_share", "auto_kind", "robotaxi_headway", "companies", "acc_headway", "coop_class"} <= av_names
    for seed in range(20):
        assert not set(presets.random_changes(seed)) & av_names
    assert not set(presets.plausible_ranges()) & av_names
    assert presets.plausible_ranges(["Coop"])["companies"] == [1, 2, 3, 4]
    assert set(presets.plausible_ranges(["Coop"])) == {p.name for p in PARAMS if p.group == "Coop"}
    shares = []
    for seed in range(200):
        drawn = presets.random_changes(seed, ["AVs", "coop"])
        assert set(drawn) == av_names
        SimConfig(**drawn)
        assert drawn["auto_kind"] in ("cav", "robotaxi", "acc") and 1.0 <= drawn["robotaxi_headway"] <= 1.6
        shares.append(drawn["auto_share"])
    assert 0 <= min(shares) < 0.1 and max(shares) > 0.8
