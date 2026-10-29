"""The human-baseline settings of S0: anticipation (how many cars ahead drivers react to; 5 = the engine's
ANTICIPATION, exactly as before) and weather_scale (how strongly the road condition changes driving; 1 = the
ROAD_CONDITIONS factors exactly, 0 = dry behaviour with the road's grip, 2 = twice the change, floored), both
live, documented and drawn by the random settings; and the capacity test's switch never drawn."""

from __future__ import annotations

import numpy as np
import pytest

from roundabout_sim import experiments, presets
from roundabout_sim.config import PARAM_BY_NAME, SimConfig, inactive_reason
from roundabout_sim.engine import (ANTICIPATION, MAX_DECEL, ROAD_CONDITIONS, ROAD_SCALED, WEATHER_FLOOR,
                                   WEATHER_SCALED, RingSim, effective, road_factors)
from tests.helpers import assert_valid


def run(cfg: SimConfig, steps: int = 400) -> tuple:
    sim = RingSim(cfg)
    sim.step(steps)
    live = np.flatnonzero(sim.alive)
    return sim.uid[live].tolist(), sim.theta[live].tolist(), sim.v[live].tolist(), sim.n_entered, sim.n_stops


# ------------------------------------------------------------------ the settings
def test_the_settings_are_documented_and_validated():
    cfg = SimConfig()
    assert (cfg.anticipation, cfg.weather_scale, cfg.subject_arm) == (ANTICIPATION, 1.0, "off")
    a, w = PARAM_BY_NAME["anticipation"], PARAM_BY_NAME["weather_scale"]
    assert (a.group, a.lo, a.hi, a.step, a.reset) == ("Drivers", 1, 5, 1, False)
    assert (w.group, w.lo, w.hi, w.reset) == ("Road", 0.0, 2.0, False)
    for spec in (a, w, PARAM_BY_NAME["subject_arm"]):
        assert spec.help and len(spec.help) <= 460 and "Treiber" in a.help
    for bad in (dict(anticipation=0), dict(anticipation=2.5), dict(weather_scale=-0.1),
                dict(weather_scale=float("nan")), dict(subject_arm="yes")):
        with pytest.raises(ValueError):
            SimConfig(**bad)
    assert "Anticipation" in PARAM_BY_NAME["reaction_distance"].help


def test_random_settings_draw_them_and_never_the_capacity_test():
    for name in ("anticipation", "weather_scale", "subject_arm"):
        assert presets.RANDOM_RANGES[name].basis.strip()
    assert presets.RANDOM_RANGES["weather_scale"].typical == 1.0
    assert presets.plausible_ranges(["Demand"])["subject_arm"] == ["off"]
    draws = [presets.random_changes(seed) for seed in range(300)]
    assert {d["subject_arm"] for d in draws} == {"off"}
    assert {d["anticipation"] for d in draws} <= {1, 2, 3, 4, 5} and len({d["anticipation"] for d in draws}) >= 4
    assert all(0.5 <= d["weather_scale"] <= 2.0 for d in draws)


# ------------------------------------------------------------------ anticipation
@pytest.mark.parametrize("cfg", [SimConfig(seed=1, density=85, n_arms=0, n_lanes=1),
                                 presets.apply("us_two_lane", SimConfig(seed=2))], ids=["ring", "us_two_lane"])
def test_anticipation_5_is_the_model_as_it_was(cfg, monkeypatch):
    """5 cars ahead, given or by default, is bit for bit the constant the engine used (golden.py checks main)."""
    given = run(cfg.updated(anticipation=5))
    assert given == run(cfg)
    from roundabout_sim import engine
    monkeypatch.setattr(engine, "ANTICIPATION", 3)  # (the loop reads the setting, not the constant)
    assert run(cfg.updated(anticipation=5)) == given


def test_anticipation_1_is_the_plain_single_leader_idm():
    sim = RingSim(SimConfig(seed=3, density=90, n_arms=0, n_lanes=1, anticipation=1, driver_noise=0.0))
    idx = np.flatnonzero(sim.alive)
    lead, gap = sim._index_lanes(idx)
    v, v0 = sim.v[idx], sim.target_speed(idx)
    following = sim._car_following(idx, v, v0, lead[idx], gap[idx], lead, gap)
    np.testing.assert_allclose(following, sim._idm(v, v0, gap[idx], sim.v[lead[idx]], idx), rtol=1e-12)
    many = sim._car_following  # five cars ahead change it in dense traffic
    sim.cfg = sim.cfg.updated(anticipation=5)
    assert not np.allclose(many(idx, v, v0, lead[idx], gap[idx], lead, gap), following)


def test_anticipation_changes_live_without_a_rebuild():
    cfg = SimConfig(seed=4, density=85, n_arms=0, n_lanes=1)
    a, b = RingSim(cfg), RingSim(cfg)
    a.step(100)
    b.step(100)
    assert b.set(anticipation=1) is False and b.time == a.time
    a.step(300)
    b.step(300)
    assert not np.array_equal(a.v, b.v)
    assert_valid(b)


# ------------------------------------------------------------------ weather_scale
@pytest.mark.parametrize("condition", list(ROAD_CONDITIONS))
def test_weather_scale_1_is_the_road_factors_exactly(condition):
    assert road_factors(condition, 1.0) is ROAD_CONDITIONS[condition]
    cfg = SimConfig(road_condition=condition, weather_scale=1.0)
    k = ROAD_CONDITIONS[condition]
    eff = effective(cfg)
    for name in ROAD_SCALED:
        expected = MAX_DECEL * k[name] if name == "max_decel" else getattr(cfg, name) * k[name]
        assert getattr(eff, name) == expected  # the multiplication main does, exactly
    assert eff.speed_std == cfg.speed_std * k["mean_speed"] and eff.headway_std == cfg.headway_std * k["time_headway"]


@pytest.mark.parametrize("name", ["snow", "us_two_lane", "peak_trucks"])
def test_weather_scale_1_runs_bit_for_bit(name):
    cfg = {"snow": presets.apply("us_single_lane_snow", SimConfig(seed=5)),
           "us_two_lane": presets.apply("us_two_lane", SimConfig(seed=6, road_condition="ice")),
           "peak_trucks": presets.apply("us_single_lane_peak", SimConfig(seed=7, road_condition="wet"))}[name]
    assert run(cfg.updated(weather_scale=1.0)) == run(cfg)


def test_weather_scale_moves_the_behaviour_and_keeps_the_grip():
    snow = ROAD_CONDITIONS["snow"]
    for scale in (0.0, 0.5, 2.0):
        k = road_factors("snow", scale)
        for name in ROAD_SCALED:
            if name in WEATHER_SCALED:
                assert k[name] == pytest.approx(max(WEATHER_FLOOR, 1 + scale * (snow[name] - 1)))
            else:  # cornering and braking limits: the tyres' grip, in full
                assert k[name] == snow[name]
    assert all(v == 1.0 for v in road_factors("dry", 2.0).values())  # nothing on a dry road
    dry, as_if_dry = effective(SimConfig()), effective(SimConfig(road_condition="snow", weather_scale=0.0))
    assert as_if_dry.mean_speed == dry.mean_speed and as_if_dry.entry_gap == dry.entry_gap
    assert as_if_dry.max_decel == MAX_DECEL * snow["max_decel"] < dry.max_decel
    ice = road_factors("ice", 2.0)  # 1 + 2 (0.45 - 1) < 0: floored
    assert ice["comfort_decel"] == WEATHER_FLOOR and ice["max_accel"] == WEATHER_FLOOR
    sim = RingSim(presets.apply("us_single_lane", SimConfig(seed=8, road_condition="ice", weather_scale=2.0)))
    sim.step(1500)
    assert_valid(sim) and sim.n_entered > 0


def test_weather_scale_changes_live_and_does_nothing_on_a_dry_road():
    cfg = presets.apply("us_single_lane_snow", SimConfig(seed=9))
    a, b = RingSim(cfg), RingSim(cfg)
    assert b.set(weather_scale=2.0) is False
    assert b.eff.mean_speed < a.eff.mean_speed and b.eff.follow_up > a.eff.follow_up
    assert b.follow_up_steps() > a.follow_up_steps()
    dry = SimConfig(seed=9)
    assert run(dry.updated(weather_scale=0.3)) == run(dry)
    assert inactive_reason("weather_scale", dry) == "dry road" and inactive_reason("weather_scale", cfg) == ""
    notes = experiments.idle_notes(["weather_scale"], [dry, dry.updated(weather_scale=2.0)])
    assert notes and "on a dry road" in notes[0] and "road_condition" in notes[0]
