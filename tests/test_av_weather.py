"""The driverless kinds' weather policy (SPEC 3, gap 1): av_weather_scale turns each of the drivers' behavioural road
factors f (after the Weather effect) into 1 + s (f - 1) for CAVs and robotaxis (av_road_factors: 1 = as the drivers,
exactly; the grip limits always the drivers'), in their desired speed, acceleration, comfortable braking, time gaps,
critical gap and follow-up; av_handover holds a driverless queue head at the yield line on snow or ice for that long
after it became the head. ACC cars (with a driver) and human drivers are untouched; every setting at its default, or
where it cannot act, changes nothing."""

from __future__ import annotations

import numpy as np
import pytest

from roundabout_sim import presets
from roundabout_sim.config import IDLE_WHY, PARAM_BY_NAME, SimConfig, inactive_inputs, inactive_reason
from roundabout_sim.engine import (ACC, CAV, HUMAN, ROAD_CONDITIONS, ROBOTAXI, SUBJECT_ARM, TWO_PI, WEATHER_FLOOR,
                                   WEATHER_SCALED, RingSim, av_road_factors, effective, road_factors)
from tests.test_entries import entry_scene
from tests.test_state_migration import trajectory

NEW = {"av_weather_scale": 1.0, "av_handover": 0.0}
OPEN = dict(traffic_mode="open", entry_rate=0.2)


# ------------------------------------------------------------------ settings
def test_the_av_weather_settings_defaults_ranges_and_tab():
    cfg = SimConfig()
    assert {name: getattr(cfg, name) for name in NEW} == NEW
    for name in NEW:
        spec = PARAM_BY_NAME[name]
        assert spec.group == "Entries" and not spec.reset and len(spec.help) <= 340, name
        assert presets.RANDOM_RANGES[name].basis.strip()
    assert (PARAM_BY_NAME["av_weather_scale"].lo, PARAM_BY_NAME["av_weather_scale"].hi) == (0.0, 2.0)
    assert (PARAM_BY_NAME["av_handover"].lo, PARAM_BY_NAME["av_handover"].hi) == (0, 120)
    for bad in (dict(av_weather_scale=-0.5), dict(av_handover=-1.0)):
        with pytest.raises(ValueError):
            SimConfig(**bad)


@pytest.mark.parametrize("name, config, reason", [
    ("av_weather_scale", dict(road_condition="snow"), "no driverless vehicles"),
    ("av_weather_scale", dict(road_condition="snow", auto_share=0.5, auto_kind="acc"), "no driverless vehicles"),
    ("av_weather_scale", dict(auto_share=0.5), "dry road"),
    ("av_weather_scale", dict(auto_share=0.5, road_condition="wet"), ""),
    ("av_weather_scale", dict(auto_share=0.5, road_condition="snow", weather_scale=0.0), "no weather effect"),
    ("av_handover", dict(auto_share=0.5, road_condition="wet"), "no snow or ice"),
    ("av_handover", dict(auto_share=0.5, road_condition="ice", auto_kind="robotaxi"), ""),
    ("av_handover", dict(auto_share=0.5, road_condition="ice", n_arms=0), "no entrances"),
])
def test_each_av_weather_setting_says_why_it_does_nothing(name, config, reason):
    assert inactive_reason(name, SimConfig(**config)) == reason
    assert reason == "" or reason in IDLE_WHY
    assert inactive_inputs("av_weather_scale") == {"auto_share", "auto_kind", "road_condition", "weather_scale"}
    assert inactive_inputs("av_handover") == {"n_arms", "auto_share", "auto_kind", "road_condition"}


# ------------------------------------------------------------------ the factors
def test_av_road_factors():
    for condition in ROAD_CONDITIONS:
        for ws in (1.0, 0.5, 1.7):
            humans = road_factors(condition, ws)
            assert av_road_factors(SimConfig(road_condition=condition, weather_scale=ws)) == humans  # (1: exactly)
            for s in (0.0, 0.4, 2.0):
                av = av_road_factors(SimConfig(road_condition=condition, weather_scale=ws, av_weather_scale=s))
                for name, f in humans.items():
                    want = max(WEATHER_FLOOR, 1.0 + s * (f - 1.0)) if name in WEATHER_SCALED else f
                    assert av[name] == want, (condition, ws, s, name)
    zero = av_road_factors(SimConfig(road_condition="snow", av_weather_scale=0.0))
    assert all(zero[name] == 1.0 for name in WEATHER_SCALED)
    assert zero["max_decel"] == 0.30 and zero["lateral_accel"] == 0.50  # (the grip limits: the drivers')


def test_at_the_defaults_the_driverless_vehicles_use_the_drivers_values():
    for cfg in (SimConfig(auto_share=0.5, road_condition="snow"), SimConfig(auto_share=0.5, av_weather_scale=0.3),
                SimConfig(auto_share=0.5, auto_kind="acc", road_condition="ice", av_weather_scale=0.3)):
        sim = RingSim(cfg.updated(seed=1))
        assert not sim.av_weather and sim.eff_av is sim.eff and sim.road_gap_av == sim.road_gap
        slots = np.flatnonzero(sim.alive)
        assert sim._comfort(slots) == sim.eff.comfort_decel and type(sim._comfort(slots)) is float
        assert sim._av_follow_up_steps() == sim.follow_up_steps()


def snow_pair(av: float, kind: str = "robotaxi", **config) -> tuple[RingSim, int, int]:
    """Snow, one lane of radius 200 m (so the cornering cap does not bind), no noise or spread: a driver and an
    automated vehicle of ``kind`` far apart. Returns (sim, driver slot, automated slot)."""
    cfg = SimConfig(**{**dict(seed=1, n_lanes=1, n_arms=4, radius=200.0, traffic_mode="open", entry_rate=0.0,
                              density=1, driver_noise=0.0, speed_std=0.0, road_condition="snow", auto_share=1.0,
                              auto_kind=kind, av_weather_scale=av), **config})
    sim = RingSim(cfg)
    sim.alive[:] = False
    code = {"robotaxi": ROBOTAXI, "cav": CAV, "acc": ACC}[kind]
    slots = sim._spawn(np.zeros(2, dtype=int), np.array([0.5, 3.5]), np.zeros(2),
                       kinds=(np.array([HUMAN, code], dtype=np.int8), np.array([-1, 0], dtype=np.int8),
                              np.array([np.nan, 1.6])))
    return sim, int(slots[0]), int(slots[1])


def test_a_robotaxi_on_snow_drives_by_its_own_factors():
    """Snow (speed x0.75, acceleration x0.7, comfortable braking x0.7, time gap x1.25): at av_weather_scale 0 a robotaxi
    drives as on a dry road (12 m/s, 1.2 m/s2, 2 m/s2, 1.3 x 1.2 s), at 2 twice the drivers' change; the driver keeps
    the snow values either way."""
    for av, speed, accel, comfort, gap in ((0.0, 12.0, 1.2, 2.0, 1.3 * 1.2), (2.0, 6.0, 0.48, 0.8, 1.3 * 1.8),
                                           (0.5, 10.5, 1.02, 1.7, 1.3 * 1.35)):
        sim, me, it = snow_pair(av)
        assert sim.av_weather
        assert sim.desired_speed(np.array([me, it])) == pytest.approx([9.0, speed])
        assert sim.accel[[me, it]] == pytest.approx([0.84, accel])
        assert sim._comfort(np.array([me, it])) == pytest.approx([1.4, comfort])
        assert sim.headway[[me, it]] == pytest.approx([1.5, gap])


def test_a_cavs_time_gaps_take_its_own_road_factor_and_an_acc_car_the_drivers():
    for kind, gap in (("cav", 1.6), ("acc", 1.1 * 1.25)):
        sim, me, it = snow_pair(0.0, kind)
        assert sim.headway[it] == pytest.approx(gap), kind  # (a CAV's fallback gap: x1 at 0; an ACC car's: x1.25)
    sim, me, it = snow_pair(0.0, "cav")
    assert sim._cacc_gap(0.7) == pytest.approx(0.7) and sim._acc_law_gap(1.1, -1) == pytest.approx(1.1 * 1.25)


def test_a_driverless_heads_critical_gap_and_follow_up_on_snow():
    """Snow: the drivers' entry gap x1.2 (2.4 s) and follow-up x1.2 (2.5 -> 3.0 s, 31 steps). A robotaxi head (x
    robotaxi_entry 1.3) at av_weather_scale 0 has the dry values: critical gap 2.0 x 1.3 = 2.6 s, follow-up round(1.3 x
    26) = 34 steps; at 1 the drivers': 2.4 x 1.3 = 3.12 s and round(1.3 x 31) = 40 steps."""
    v = 6.0
    for av in (0.0, 1.0):
        sim, _ = entry_scene({0: (ROBOTAXI, 0, np.nan), 1: (HUMAN, -1, np.nan)}, [(HUMAN, 60.0, v)],
                             auto_kind="robotaxi", road_condition="snow", av_weather_scale=av, follow_up=2.5)
        assert sim.follow_up_steps() == 31 and sim._av_follow_up_steps() == (26 if av == 0 else 31)
        slack, _ = sim._follow_up_slack(31, np.array([0, 1]))
        assert slack.tolist() == [31 - (34 if av == 0 else 40), 0]
    for av, lag, ok in ((0.0, 2.65, True), (0.0, 2.5, False), (1.0, 3.2, True), (1.0, 3.0, False)):
        sim, _ = entry_scene({0: (ROBOTAXI, 0, np.nan)}, [(HUMAN, -(lag * v + 4.0), v), (HUMAN, 60.0, v)],
                             auto_kind="robotaxi", road_condition="snow", av_weather_scale=av)
        arms = np.array([0])
        assert bool(sim._entry_check(sim.entry_angle[arms], None, sim._head_s0(arms), arms)[0][0]) == ok, (av, lag)


@pytest.mark.parametrize("config", [
    dict(auto_share=0.5, auto_kind="cav", road_condition="dry", av_weather_scale=0.2, av_handover=40.0),
    dict(auto_share=0.5, auto_kind="acc", road_condition="snow", av_weather_scale=0.2, av_handover=40.0),
    dict(auto_share=0.5, auto_kind="robotaxi", road_condition="wet", av_handover=40.0),
    dict(road_condition="ice", av_weather_scale=1.8, av_handover=40.0),
], ids=["dry", "acc-snow", "wet-handover", "humans-ice"])
def test_where_they_cannot_act_the_av_weather_settings_change_nothing(config):
    base = RingSim(SimConfig(seed=5, **OPEN, **{k: v for k, v in config.items() if not k.startswith("av_")}))
    other = RingSim(SimConfig(seed=5, **OPEN, **config))
    assert not (other.av_weather or other.handover)
    base.step(1200)
    other.step(1200)
    assert trajectory(other) == trajectory(base) and base.n_entered > 20


def test_without_a_weather_effect_av_weather_changes_nothing():
    """At weather_scale 0 the drivers' factors are all 1, so 1 + s (1 - 1) = 1 for every s: idle ("no weather
    effect"), and the run is the same."""
    runs = [RingSim(SimConfig(seed=3, auto_share=0.5, auto_kind="cav", road_condition="snow", weather_scale=0.0,
                              av_weather_scale=s, **OPEN)) for s in (2.0, 1.0)]
    for sim in runs:
        sim.step(800)
    assert trajectory(runs[0]) == trajectory(runs[1]) and runs[0].n_entered > 20


def test_a_live_av_weather_change_logs_no_wave_of_stops():
    """Robotaxis on ice at av_weather_scale 2 (desired speed 1.2 m/s at weather_scale 2) set to 0 live (12 m/s): set()
    re-baselines who counts as moving, as for the drivers' weather settings, so no vehicle is logged as stopping."""
    cfg = presets.apply("default").updated(road_condition="ice", av_weather_scale=2.0, weather_scale=2.0,
                                           auto_share=1.0, auto_kind="robotaxi", n_arms=0, n_lanes=1, density=20.0,
                                           seed=5)
    sim = RingSim(cfg)
    sim.step(600)
    assert sim.moving[sim.alive].any()
    n = sim.n_stops
    assert sim.set(av_weather_scale=0.0) is False
    sim.step(5)
    assert sim.n_stops == n


def test_av_weather_changes_a_driverless_run():
    base = RingSim(SimConfig(seed=5, auto_share=0.5, auto_kind="cav", road_condition="snow", **OPEN))
    other = RingSim(SimConfig(seed=5, auto_share=0.5, auto_kind="cav", road_condition="snow", av_weather_scale=0.0,
                              **OPEN))
    base.step(600)
    other.step(600)
    assert trajectory(other) != trajectory(base)


# ------------------------------------------------------------------ the handover at the yield line
def test_a_driverless_head_waits_for_its_handover():
    """On snow with av_handover 10 s a robotaxi head may try to enter only 10 s after it became the head: its arrival,
    or its entrance's last entry if later; a driver head at once."""
    sim, _ = entry_scene({0: (ROBOTAXI, 0, np.nan), 1: (HUMAN, -1, np.nan)}, [(HUMAN, 60.0, 6.0)],
                         auto_kind="robotaxi", road_condition="snow", av_handover=10.0)
    assert sim.handover
    sim.queue_times[0][0] = sim.queue_times[1][0] = 0.0
    sim.time = 9.9
    assert sim._handed_over(np.array([0, 1])).tolist() == [False, True]
    sim.time = 10.0
    assert sim._handed_over(np.array([0, 1])).tolist() == [True, True]
    sim.last_entry[0] = 5.0  # (it became the head at the last entry, 5 s after it arrived)
    assert sim._handed_over(np.array([0])).tolist() == [False]
    sim.time = 15.0
    assert sim._handed_over(np.array([0])).tolist() == [True]


def test_the_handover_spaces_a_driverless_queue_on_snow():
    """Capacity test on snow without conflicting flow: robotaxis enter every 10 s with a 10 s handover (each next head
    becomes the head at the last entry); without it at their follow-up; on wet roads it does nothing."""
    def spacings(road="snow", **config):
        cfg = presets.apply("us_single_lane", SimConfig(seed=4)).updated(
            subject_arm="on", entry_rate=0.0, auto_share=1.0, auto_kind="robotaxi", road_condition=road, **config)
        sim = RingSim(cfg)
        sim.step(1500)
        times = np.array([t for t, arm, _ in sim.entry_log if arm == SUBJECT_ARM])
        return np.round(np.diff(times), 6)

    held = spacings(av_handover=10.0)
    assert held.size >= 10 and held.min() == pytest.approx(10.0) and np.median(held) == pytest.approx(10.0)
    assert np.median(spacings()) < 5.0
    assert np.array_equal(spacings("wet", av_handover=10.0), spacings("wet"))
