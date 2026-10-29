"""Stage S1 after review: the capacity test's top-up drawing its kinds from a stream of its own (rng_kinds_test),
so the conflicting stream and the test queue stay coupled across shares; an entrant let in only with room for
its own standstill gap (a robotaxi's 1.94 m longer); the post-merge slot for the automated limits (S2's clip
and safety layer); Robotaxi time gap's one idle reason and the sweep tools combining the picked settings it
needs; the sweep viewer's labels and fixes for the new columns and reasons; the viewer's origin keeping the
AVs and Coop settings that a draw of every tab (X, --random) leaves alone."""

from __future__ import annotations

import numpy as np
import pygame
import pytest

from roundabout_sim import app as viewer
from roundabout_sim import experiments, presets
from roundabout_sim import sweep_viewer as sv
from roundabout_sim.config import IDLE_WHY, PARAMS, SimConfig, inactive_inputs, inactive_reason
from roundabout_sim.engine import HUMAN, ROBOTAXI, ROBOTAXI_S0_EXTRA, SUBJECT_ARM, RingSim
from tests.helpers import frames
from tests.test_kinds import Recording


def capacity_test(seed: int, **changes) -> SimConfig:
    return presets.apply("us_single_lane", SimConfig(seed=seed)).updated(subject_arm="on", **changes)


# ------------------------------------------------------------------ coupling in the capacity test
@pytest.mark.parametrize("kind", ["robotaxi", "cav"])
def test_the_capacity_test_keeps_the_kinds_coupled_across_shares(kind):
    """How many vehicles the top-up adds follows the subject entries, which differ between shares; on a
    stream of its own it shifts nobody else's draws: the n-th arrival at each feeding entrance and the n-th
    vehicle topped up draw the same uniforms whatever the share, so both automated sets nest (SPEC 1, V11)."""
    shares = (0.2, 0.4, 1.0)
    sims = [Recording(capacity_test(7, entry_rate=0.06, max_queue=10_000, auto_share=share, auto_kind=kind,
                                    companies=2)) for share in shares]
    for sim in sims:
        sim.step(3000)
    assert len({sim.n_topped_up for sim in sims}) > 1  # the runs went apart (the top-ups differ)
    assert all(list(sim.arrival_log) == list(sims[0].arrival_log) for sim in sims)  # (rng_demand)
    for lo, hi in zip(sims, sims[1:]):
        for stream in ("drawn", "drawn_test"):  # placement and arrivals (rng_kinds); top-ups (rng_kinds_test)
            (k1, c1, g1), (k2, c2, g2) = (tuple(np.concatenate([d[j] for d in s.__dict__[stream]]) for j in range(3))
                                          for s in (lo, hi))
            n = min(k1.size, k2.size)
            if stream == "drawn":
                assert k1.size == k2.size  # the same vehicles drew (blocked arrivals too)
            a1, a2 = k1[:n] != HUMAN, k2[:n] != HUMAN
            assert n > 40 and (a1 <= a2).all(), stream
            np.testing.assert_array_equal(c1[:n][a1], c2[:n][a1])
            np.testing.assert_array_equal(g1[:n], g2[:n])
        for arm in range(1, lo.n_arms):  # by vehicle: each feeding entrance's k-th arrival (none turned away)
            q1, q2 = lo.queued[arm], hi.queued[arm]
            n = min(len(q1), len(q2))
            assert n > 10 and all(b[0] != HUMAN for a, b in zip(q1[:n], q2[:n]) if a[0] != HUMAN)
    assert all(sim.n_blocked == 0 for sim in sims)


# ------------------------------------------------------------------ entries with the entrant's own s0
class Entering(RingSim):
    """Records, for each vehicle let in, the bumper gap ahead of it and its own standstill gap (the room
    _entry_ok asks for is that + 1 m)."""

    def _entry_ok(self, angles, trucks=None, s0=None):
        ok, v_in = super()._entry_ok(angles, trucks, s0)
        nl, front, _, _ = self._neighbours_in(self.outer, angles, self.cfg.car_length)
        if nl[0] >= 0 and ok.any():
            k = np.flatnonzero(ok)[0]  # (the first entrance that may go is the one that does)
            own = self.cfg.min_gap if s0 is None else s0[k]
            self.__dict__.setdefault("fronts", []).append((float(front[k]), float(own)))
        return ok, v_in


@pytest.mark.parametrize("entry_rate", [0.0, 0.03])
def test_an_entrant_needs_room_for_its_own_standstill_gap(entry_rate):
    """A robotaxi queue is let in only with its own standstill gap + 1 m ahead (min_gap + 1.94 + 1 m): let in
    behind a driver's 2.5 m, as before, each braked to a stand at the entry point and the queue discharged at
    about half the drivers' rate (vc 0), with harsh braking at the entry once there was conflicting traffic."""
    runs = {}
    for label, extra in (("human", {}), ("robotaxi", dict(auto_share=1.0, auto_kind="robotaxi"))):
        sim = Entering(capacity_test(3, entry_rate=entry_rate, **extra))
        sim.step(4000)
        runs[label] = sim
    human, robot = runs["human"], runs["robotaxi"]
    assert robot.cfg.min_gap + ROBOTAXI_S0_EXTRA == pytest.approx(3.44)
    assert all(own == pytest.approx(3.44) for _, own in robot.fronts) and robot.fronts
    assert min(front - own for front, own in robot.fronts) >= 1.0 - 1e-9
    assert all(own == human.cfg.min_gap for _, own in human.fronts)
    entered = robot.arm_entered[SUBJECT_ARM]
    assert entered >= 0.85 * human.arm_entered[SUBJECT_ARM] > 60  # (as before: 0.56 x at vc 0, 0.54 x at ~300)
    assert robot.n_emergency <= 0.02 * entered  # (as before: 0.38 harsh brakings per entry at vc ~300)


def test_the_entry_check_keeps_min_gap_without_automated_vehicles():
    """No automated vehicle possible: _head_s0 is None and _entry_ok's arithmetic is the plain min_gap's
    (the golden runs check it bit for bit); a human queue head of an automated run needs min_gap too."""
    plain = RingSim(capacity_test(4, entry_rate=0.05))
    assert plain._head_s0(np.arange(plain.n_arms)) is None
    mixed = RingSim(capacity_test(4, entry_rate=0.05, auto_share=0.5, auto_kind="robotaxi"))
    mixed.step(200)
    arms = np.arange(mixed.n_arms)
    heads = [mixed.queue_kinds[arm][0] if mixed.queue_kinds[arm] else HUMAN for arm in arms]
    s0 = mixed._head_s0(arms)
    assert s0.tolist() == [mixed.cfg.min_gap + (ROBOTAXI_S0_EXTRA if k == ROBOTAXI else 0.0) for k in heads]


# ------------------------------------------------------------------ the automated limits' slot
def test_the_automated_limits_come_after_merging_with_the_direct_leader_idm():
    """_move calls _automated_limits (S2: the clip to [-max_decel, accel], then the safety layer) after
    cooperative merging and before the harsh-braking check, with what _car_following gave (an automated car's
    direct-leader IDM); only while automated vehicles are possible, and in S1 it changes nothing."""
    cfg = SimConfig(seed=5, n_lanes=3, density=70, exit_prob=0.3, auto_share=1.0, auto_kind="robotaxi")
    sim = RingSim(cfg)
    calls = []
    follow, merge = sim._car_following, sim._make_room_to_leave
    sim._car_following = lambda idx, *a: calls.append(("follow", follow(idx, *a))) or calls[-1][1].copy()
    sim._make_room_to_leave = lambda idx, v, v0, acc: (merge(idx, v, v0, acc), calls.append(("merge", acc.copy())))
    sim._automated_limits = lambda idx, acc, a_idm: calls.append(("limits", acc.copy(), a_idm.copy()))
    sim.step(30)
    assert [c[0] for c in calls] == ["follow", "merge", "limits"] * 30
    for j in range(0, len(calls), 3):
        np.testing.assert_array_equal(calls[j + 2][2], calls[j][1])  # a_idm: the car-following command
        np.testing.assert_array_equal(calls[j + 2][1], calls[j + 1][1])  # acc: after merging
    reference = RingSim(cfg)
    reference.step(30)
    np.testing.assert_array_equal(sim.theta, reference.theta)  # (S1: the slot is empty)
    human = RingSim(SimConfig(seed=5))
    human._automated_limits = lambda *a: pytest.fail("called without automated vehicles")
    human.step(10)


# ------------------------------------------------------------------ idle reasons
def test_robotaxi_time_gap_has_one_reason_and_the_sweep_notes_warn_of_it():
    """Robotaxi time gap does nothing without robotaxis, for want of automated vehicles or of the kind: one
    reason, so a sweep mixing both causes still gets its note (it split into two before, and none was shown)."""
    configs = [SimConfig(auto_share=a, robotaxi_headway=h) for a in (0.0, 0.5) for h in (1.0, 1.6)]
    assert {inactive_reason("robotaxi_headway", c) for c in configs} == {"no robotaxis"}
    notes = experiments.idle_notes(["auto_share", "robotaxi_headway"], configs)
    assert len(notes) == 1 and notes[0].startswith("robotaxi_headway acts only on robotaxis, and no run here has any")
    assert "set auto_kind=robotaxi and auto_share above 0" in notes[0]
    robotaxis = [c.updated(auto_kind="robotaxi") for c in configs]
    assert experiments.idle_notes(["auto_share", "robotaxi_headway"], robotaxis) == []
    assert set(IDLE_WHY) <= set(experiments.IDLE_FIX) and set(IDLE_WHY) <= set(sv.IDLE_FIX)


def test_inactive_inputs_name_every_setting_inactive_reason_reads():
    """config.inactive_inputs (what the sweep viewer combines) holds every setting inactive_reason reads: over
    random settings of every tab, each setting's reason follows from those settings alone."""
    default = SimConfig()
    for seed in range(60):
        drawn = presets.random_config(seed, groups=presets.GROUPS)
        for cfg in (drawn, drawn.updated(subject_arm="on")):
            for spec in PARAMS:
                alone = default.updated(**{n: getattr(cfg, n) for n in inactive_inputs(spec.name)})
                assert inactive_reason(spec.name, cfg) == inactive_reason(spec.name, alone), (seed, spec.name)
    assert inactive_inputs("robotaxi_headway") == {"auto_share", "auto_kind"}
    assert inactive_inputs("density") == set()


def test_the_new_sweep_form_combines_the_settings_robotaxi_time_gap_needs(tmp_path):
    """Robotaxi time gap acts only with Automated share above 0 AND AV kind robotaxi: the form warns until
    both are picked (each alone is not enough), then not, and says what to pick meanwhile."""
    view = sv.SweepViewer(None, size=(1100, 700), results_dir=tmp_path / "results")
    try:
        view.show_page("new")
        for name, idle in (("robotaxi_headway", "no robotaxis"), ("auto_kind", "no robotaxis"), ("auto_share", "")):
            view.toggle_param(name)
            view.layout()
            assert view.idle["robotaxi_headway"] == idle, name
            if idle:
                assert "pick Automated share and AV kind (with robotaxi) too" in " ".join(view.idle_lines[1])
        assert view.idle_lines[1] == [] and dict(view.form_variables())["auto_kind"] == ["cav", "robotaxi", "acc"]
        view.toggle_param("auto_kind")  # Automated share alone (the base kind is cav): idle again
        view.layout()
        assert view.idle["robotaxi_headway"] == "no robotaxis"
        view.base = {"auto_kind": "robotaxi"}  # robotaxi base settings: Automated share picked is enough
        view.layout()
        # (Companies: no combination of the picked settings makes the companies drive differently, robotaxis' only
        # label them, so the base settings' reason stands; config.companies_differ)
        assert view.idle["robotaxi_headway"] == "" and view.idle_reason("companies") == "no automated vehicles"
        view.base = {}
        view.layout()
        assert view.idle_reason("companies") == "no automated vehicles"  # (Automated share picked: nothing differs)
        view.base = {"brand_spread": 0.3}  # cav (the base kind) with a brand spread: the companies differ
        view.layout()
        assert view.idle_reason("companies") == ""
    finally:
        view.close()
        pygame.quit()


def test_the_new_columns_have_readable_labels():
    assert sv.column_label("automated") == "Automated vehicles on the ring"
    assert sv.column_label("human_speed") == "Human-driven speed (m/s)"
    assert sv.column_label("auto_speed") == "Automated vehicles' speed (m/s)"


# ------------------------------------------------------------------ the viewer's origin
def test_randomize_all_keeps_the_av_settings_in_the_origin(make_app, monkeypatch):
    """X draws every tab but AVs and Coop: their settings stay, are part of where the settings come from
    (given, when not at their defaults, with the flags that --random needs to start with them), and a later
    change to one counts as a change."""
    app = make_app(auto_share=0.4, auto_kind="robotaxi")
    frames(app, 2)
    monkeypatch.setattr(presets, "new_seed", lambda: 4321)
    app.randomize()
    cfg = app.sim.cfg
    assert cfg.auto_share == 0.4 and cfg.auto_kind == "robotaxi"
    assert "python -m roundabout_sim --random 4321 --auto_share 0.4 --auto_kind robotaxi starts with them" \
        in app.message[0]
    args = viewer.parse_args(["--random", "4321", "--auto_share", "0.4", "--auto_kind", "robotaxi", "--seed", "7"])
    assert viewer.config_from_args(args) == cfg  # (what the message says)
    assert app.origin_text() == "random settings, seed 4321 (given: Automated share, AV kind)"
    assert app.changed_since() == []
    app.apply("auto_share", 0.2)
    app.apply("companies", 3)  # a kept default changed counts too
    assert app.changed_since() == ["Automated share", "Companies"]
    _, txt = app.save_data()
    command = next(line for line in txt.read_text(encoding="utf-8").splitlines() if "experiments run" in line)
    assert "--random 4321" in command and "--set auto_share=0.2" in command and "--set companies=3" in command


def test_random_with_av_flags_names_them_as_given(new_app):
    args = viewer.parse_args(["--random", "5", "--auto_share", "0.3", "--seed", "1"])
    app = new_app(viewer.config_from_args(args), origin=viewer.origin_from_args(args))
    assert app.sim.cfg.auto_share == 0.3
    assert app.origin_text() == "random settings, seed 5 (given: Automated share)"
    app.apply("auto_share", 0.5)
    app.apply("auto_kind", "robotaxi")
    assert app.changed_since() == ["Automated share", "AV kind"]
    plain = viewer.origin_from_args(viewer.parse_args(["--random", "5"]))
    assert plain["given"] == [] and plain["values"]["auto_share"] == 0.0  # (the defaults, kept)
