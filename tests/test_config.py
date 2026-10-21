"""SimConfig defaults, the PARAMS table that drives the UI/CLI, and value coercion."""

from __future__ import annotations

from dataclasses import fields

import pytest

from roundabout_sim.config import PARAM_BY_NAME, PARAMS, SimConfig, coerce

DEFAULTS = SimConfig()
FIELDS = {f.name for f in fields(SimConfig)}
NUMERIC = [p for p in PARAMS if p.choices is None]
INTEGER = [p for p in NUMERIC if isinstance(getattr(DEFAULTS, p.name), int)]
CATEGORICAL = [p for p in PARAMS if p.choices is not None]


def _name(spec) -> str:
    return spec.name


def test_param_names_are_unique_config_fields():
    names = [p.name for p in PARAMS]
    assert len(names) == len(set(names))
    assert set(names) <= FIELDS
    assert set(PARAM_BY_NAME) == set(names)


def test_requested_controls_are_exposed():
    """Every variable the project asked to vary has a control."""
    wanted = {"speed_std", "density", "n_lanes", "n_arms", "reaction_distance",
              "lane_change_rate", "init_spacing", "reaction_time", "car_length"}
    assert wanted <= set(PARAM_BY_NAME)


@pytest.mark.parametrize("spec", NUMERIC, ids=_name)
def test_numeric_default_within_range(spec):
    assert spec.lo < spec.hi
    assert spec.step > 0
    assert spec.lo <= getattr(DEFAULTS, spec.name) <= spec.hi


@pytest.mark.parametrize("spec", INTEGER, ids=_name)
def test_integer_params_have_integer_ranges(spec):
    assert float(spec.lo).is_integer() and float(spec.hi).is_integer()
    assert float(spec.step).is_integer()


@pytest.mark.parametrize("spec", CATEGORICAL, ids=_name)
def test_choice_default_is_one_of_the_choices(spec):
    assert len(spec.choices) >= 2
    assert len(set(spec.choices)) == len(spec.choices)
    assert getattr(DEFAULTS, spec.name) in spec.choices


@pytest.mark.parametrize("spec", PARAMS, ids=_name)
def test_param_is_documented(spec):
    assert spec.label.strip() and spec.group.strip() and spec.help.strip()


def test_geometry_needs_rebuild_and_traffic_is_live():
    for name in ("n_lanes", "n_arms", "radius", "car_length"):
        assert PARAM_BY_NAME[name].reset, name
    for name in ("density", "speed_std", "reaction_distance", "lane_change_rate", "reaction_time"):
        assert not PARAM_BY_NAME[name].reset, name


@pytest.mark.parametrize("spec", PARAMS, ids=_name)
def test_coerce_round_trips_default_through_text(spec):
    default = getattr(DEFAULTS, spec.name)
    value = coerce(spec.name, str(default))
    assert value == default
    assert type(value) is type(default)


@pytest.mark.parametrize("raw, expected", [("3", 3), ("2.6", 3), (4.4, 4), (2, 2)])
def test_coerce_int(raw, expected):
    value = coerce("n_lanes", raw)
    assert value == expected and type(value) is int


@pytest.mark.parametrize("raw, expected", [("42.5", 42.5), (7, 7.0), ("1e2", 100.0)])
def test_coerce_float(raw, expected):
    value = coerce("density", raw)
    assert value == expected and type(value) is float


def test_coerce_choice():
    assert coerce("init_spacing", "clustered") == "clustered"
    assert coerce("lane_change_motive", "random") == "random"
    with pytest.raises(ValueError):
        coerce("init_spacing", "diagonal")


@pytest.mark.parametrize("raw", [None, "", "none", "None"])
def test_coerce_seed_none(raw):
    assert coerce("seed", raw) is None


def test_coerce_seed_int():
    value = coerce("seed", "12")
    assert value == 12 and type(value) is int


def test_updated_copies_and_applies_changes():
    cfg = SimConfig()
    new = cfg.updated(n_lanes=5, density=99.0)
    assert (new.n_lanes, new.density) == (5, 99.0)
    assert (cfg.n_lanes, cfg.density) == (DEFAULTS.n_lanes, DEFAULTS.density)


def test_updated_rejects_unknown_names():
    with pytest.raises(KeyError, match="not_a_param"):
        SimConfig().updated(not_a_param=1)
