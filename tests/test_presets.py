"""Real-world presets, plausible random settings and Latin hypercube designs (roundabout_sim.presets):
the preset data against the sliders, apply/get/describe, every preset in the engine (the ring presets'
stop-and-go waves, the open presets' entries), random draws (valid, on the slider steps, reproducible,
linked like real roundabouts, distributed as specified) and the stratification of the sweep samples."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import fields

import numpy as np
import pytest

from roundabout_sim import presets
from roundabout_sim.config import PARAM_BY_NAME, PARAMS, SimConfig
from roundabout_sim.engine import ROAD_CONDITIONS, RingSim, mean_demand_factor
from roundabout_sim.textfmt import decimals
from tests.helpers import assert_valid, strata

DEFAULTS = SimConfig()
NAMES = presets.names()
PARAM_NAMES = [p.name for p in PARAMS]
OPEN_PRESETS = [n for n in NAMES if presets.apply(n).traffic_mode == "open"]
CHOICES = [n for n, spec in presets.RANDOM_RANGES.items() if isinstance(spec, presets.Choice)]
DRAWS = 3000  # seeds of the random-settings checks


def on_step(name: str, value) -> bool:
    """``value`` is one of the slider's values: lo + k x step, with no float artefacts
    (0.3, not 0.30000000000000004) beyond the decimals the viewer shows."""
    spec = PARAM_BY_NAME[name]
    k = (value - spec.lo) / spec.step
    return abs(k - round(k)) < 1e-9 and float(f"{value:.{decimals(spec.step)}f}") == value


def in_range(name: str, value) -> bool:
    spec = PARAM_BY_NAME[name]
    return value in spec.choices if spec.choices else spec.lo <= value <= spec.hi


@pytest.fixture(scope="module")
def draws() -> list[dict]:
    return [presets.random_changes(seed) for seed in range(DRAWS)]


# ------------------------------------------------------------------ presets
def test_presets_are_listed_in_display_order():
    assert NAMES == list(presets.PRESETS) and NAMES[0] == "default"
    assert 8 <= len(NAMES) <= 12  # what the viewer's selector is laid out for
    assert {"sugiyama2008", "tadaki2013", "stern2018", "us_single_lane", "us_two_lane", "etoile"} <= set(NAMES)
    assert all(presets.PRESETS[name].name == name for name in NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_preset_is_documented_and_sourced(name):
    p = presets.get(name)
    assert 0 < len(p.title) <= 24 and p.title == p.title.strip()
    assert p.description.strip() and 1 <= len(re.findall(r"[.!?](?:\s|$)", p.description)) <= 2
    assert p.sources and all(isinstance(s, str) and s.strip() for s in p.sources)
    if name != "default":
        assert any("https://" in s for s in p.sources)  # traceable to a citation with a URL
        assert p.notes.strip()
    for text in (p.title, p.description, p.notes, p.cite, *p.sources):
        assert text.isascii()  # safe for any console and the viewer's fonts
    # A short citation of the main sources for the viewer's help box (none for the model's own defaults).
    assert (0 < len(p.cite) <= 50) == (name != "default"), p.cite


@pytest.mark.parametrize("name", NAMES)
def test_preset_values_are_parameters_within_the_slider_ranges(name):
    for key, value in presets.get(name).changes.items():
        assert key in PARAM_BY_NAME, key  # a model parameter with a slider (never the seed or dt)
        assert type(value) is type(getattr(DEFAULTS, key)), key
        assert in_range(key, value), (key, value)


@pytest.mark.parametrize("name", NAMES)
def test_apply_puts_the_preset_on_top_of_the_defaults(name):
    p = presets.get(name)
    cfg = presets.apply(name)
    assert isinstance(cfg, SimConfig) and cfg.seed is None
    for f in fields(SimConfig):
        assert getattr(cfg, f.name) == p.changes.get(f.name, getattr(DEFAULTS, f.name)), f.name


def test_default_preset_is_the_model_defaults():
    assert presets.get("default").changes == {}
    assert presets.apply("default") == DEFAULTS


def test_apply_starts_from_the_defaults_and_keeps_only_the_seed():
    base = SimConfig(seed=7, n_lanes=5, n_arms=8, density=10.0, truck_share=0.2, road_condition="ice", dt=0.05,
                     lane_width=3.0, car_width=2.0)
    before = dict(presets.get("sugiyama2008").changes)
    cfg = presets.apply("sugiyama2008", base)
    assert cfg == presets.apply("sugiyama2008").updated(seed=7)
    assert (cfg.n_lanes, cfg.n_arms, cfg.radius, cfg.density) == (1, 0, 36.6, 95.7)
    assert (cfg.truck_share, cfg.road_condition, cfg.dt, cfg.lane_width, cfg.car_width) == \
        (0.0, "dry", DEFAULTS.dt, DEFAULTS.lane_width, DEFAULTS.car_width)
    assert base.n_lanes == 5 and presets.get("sugiyama2008").changes == before  # nothing changed in place
    assert presets.apply("etoile", base).updated(n_lanes=3).n_lanes == 3  # explicit flags go on top


@pytest.mark.parametrize("text, name", [
    ("sugiyama2008", "sugiyama2008"), ("SUGIYAMA2008", "sugiyama2008"), ("tadaki-2013", "tadaki2013"),
    ("us-single-lane", "us_single_lane"), ("US_Single_Lane_Peak", "us_single_lane_peak"),
    (" us single lane wet ", "us_single_lane_wet"), ("Place Charles de Gaulle", "etoile"), ("ETOILE", "etoile"),
    ("Stern 2018 ring", "stern2018")])
def test_get_accepts_case_separator_and_title_variants(text, name):
    assert presets.get(text) is presets.PRESETS[name]
    assert presets.get(presets.PRESETS[name]) is presets.PRESETS[name]
    assert presets.apply(text) == presets.apply(name)


def test_unknown_preset_lists_the_valid_names():
    with pytest.raises(KeyError) as err:
        presets.get("magic_roundabout")
    message = str(err.value)
    assert "magic_roundabout" in message and all(name in message for name in NAMES)
    assert not message.startswith(("'", '"'))  # reads as a plain command-line error
    with pytest.raises(KeyError):
        presets.apply("")


def test_find_names_the_preset_of_a_config():
    for name in NAMES:
        assert presets.find(presets.apply(name, SimConfig(seed=3))) == name
    assert presets.find(presets.apply("etoile").updated(n_lanes=4)) is None


@pytest.mark.parametrize("name", NAMES)
def test_describe_is_plain_ascii_with_everything(name):
    p = presets.get(name)
    text = presets.describe(name)
    assert text.isascii()
    for part in (p.title, p.name, p.description, p.notes, *p.sources):
        assert part in text
    for key in p.changes:
        assert f"{key}=" in text
    wrapped = presets.describe(name, width=72)
    assert wrapped.isascii()
    assert all(len(line) <= 72 or "http" in line for line in wrapped.splitlines())  # only URLs run over


def test_list_text_lists_every_preset_in_plain_ascii():
    text = presets.list_text()
    assert text.isascii() and text.endswith("\n")
    assert "(and --set in experiments)" in " ".join(text.split())  # the viewer has no --set
    for p in presets.PRESETS.values():
        assert f"{p.name:<24} {p.title}" in text
    assert text.index("sugiyama2008") < text.index("tadaki2013") < text.index("etoile")
    assert all(len(line) <= 100 or "http" in line for line in text.splitlines())
    full = presets.list_text(full=True)
    assert full.isascii() and all(presets.describe(name, 100) in full for name in NAMES)


# ------------------------------------------------------- presets in the engine
@pytest.mark.parametrize("mode", ["closed", "open"])
@pytest.mark.parametrize("name", NAMES)
def test_every_preset_runs_in_both_traffic_modes(name, mode):
    sim = RingSim(presets.apply(name, SimConfig(seed=1)).updated(traffic_mode=mode))
    cars = int(sim.alive.sum())
    assert cars > 0
    sim.step(300)
    assert_valid(sim)
    if sim.cfg.n_arms == 0:
        assert int(sim.alive.sum()) == cars and sim.n_entered == 0  # a closed circuit keeps its cars
    else:
        assert sim.n_entered > 0  # the entrances admit cars


@pytest.mark.parametrize("name", OPEN_PRESETS)
def test_open_presets_admit_their_demand(name):
    sim = RingSim(presets.apply(name, SimConfig(seed=2)))
    sim.step(1200)  # 2 minutes
    cfg = sim.cfg
    offered = cfg.entry_rate * cfg.n_arms * sim.time * mean_demand_factor(cfg.demand_profile, 0.0, sim.time)
    assert 0.6 * offered < sim.n_arrived < 1.4 * offered
    assert sim.n_entered > 0.6 * sim.n_arrived and sim.n_blocked == 0
    assert_valid(sim)


def wave_onset(cfg: SimConfig, horizon: float) -> float | None:
    """When the first stop-and-go wave shows (speed std across cars above 2.5 m/s, Stern 2018's
    detector, with a car nearly stopped), or None by ``horizon`` seconds."""
    sim = RingSim(cfg)
    while sim.time < horizon:
        sim.step(10)
        v = sim.v[sim.alive]
        if v.std() > 2.5 and v.min() < 1.0:
            return sim.time
    return None


@pytest.mark.parametrize("name, seed, cars, by", [
    ("sugiyama2008", 1, 22, 90.0),  # waves after 63 s (seed 1); Sugiyama: a jam in the 3-minute snapshot
    ("tadaki2013", 2, 30, 90.0),  # 55 s; Tadaki: 30 cars on the dome circuit jam
    ("stern2018", 2, 21, 150.0)])  # 100 s; Stern: first waves 55-79 s after the start (161 s with 22 cars)
def test_ring_presets_form_stop_and_go_waves_within_minutes(name, seed, cars, by):
    cfg = presets.apply(name, SimConfig(seed=seed))
    assert int(RingSim(cfg).alive.sum()) == cars  # the published number of cars
    onset = wave_onset(cfg, by)
    assert onset is not None and onset > 20.0  # from an even start, not at once


def test_tadaki_ring_flows_freely_below_the_measured_transition():
    """25 cars (79.6 veh/km) stayed free in the experiment; 30 jam (above)."""
    sim = RingSim(presets.apply("tadaki2013", SimConfig(seed=1)).updated(density=79.6))
    assert int(sim.alive.sum()) == 25
    worst = 0.0
    while sim.time < 240.0:
        sim.step(10)
        if sim.time > 60.0:
            worst = max(worst, float(sim.v[sim.alive].std()))
    assert worst < 2.0


def test_hcm_capacity_link_reproduces_the_hcm_6_saturation_flows():
    """At v/c 1, equal demand and exit chance 0.5 (circulating = entering flow) the entrance chance is
    HCM 6's self-consistent saturation flow: 685 veh/h single-lane, 750 one lane against two."""
    assert presets._hcm_entry_rate(1.0, 1, 4, 0.5, 0.0) * 3600 == pytest.approx(685, abs=2)
    assert presets._hcm_entry_rate(1.0, 2, 4, 0.5, 0.0) * 3600 == pytest.approx(750, abs=2)
    assert presets._hcm_entry_rate(0.5, 1, 4, 0.5, 0.6) < presets._hcm_entry_rate(0.5, 1, 4, 0.5, 0.0)


def test_hcm_capacity_link_follows_the_road_and_the_trucks():
    """The demand is a degree of saturation of the capacity on the drawn road, in vehicles: worse roads and
    more trucks (TRUCK_PCE cars each) get less, so snow and ice draws are not gridlock by construction."""
    assert set(presets.ROAD_CAPACITY) == set(ROAD_CONDITIONS) and presets.ROAD_CAPACITY["dry"] == 1.0
    rates = [presets._hcm_entry_rate(0.6, 1, 4, 0.5, 0.2, road) for road in ("dry", "wet", "snow", "ice")]
    assert rates == sorted(rates, reverse=True) and rates[3] < 0.8 * rates[0]
    dry = presets._hcm_entry_rate(0.6, 1, 4, 0.5, 0.2)
    assert presets._hcm_entry_rate(0.6, 1, 4, 0.5, 0.2, "dry", 0.1) == pytest.approx(dry / 1.1)
    # the full draw passes the drawn road and trucks on: seeds whose draw only differs there
    for seed in range(200):
        ch = presets.random_changes(seed)
        if ch["traffic_mode"] == "open" and ch["road_condition"] != "dry":
            dry_ch = presets.random_changes(seed, fixed={"road_condition": "dry"})
            assert ch["entry_rate"] <= dry_ch["entry_rate"]
            assert {k for k in ch if ch[k] != dry_ch[k]} <= {"road_condition", "entry_rate"}


# ----------------------------------------------------------- random settings
def test_random_ranges_cover_every_slider_parameter():
    assert set(presets.RANDOM_RANGES) == set(PARAM_NAMES)
    for name, spec in presets.RANDOM_RANGES.items():
        param = PARAM_BY_NAME[name]
        assert spec.basis.strip(), name  # why, and the source
        if isinstance(spec, presets.Choice):
            assert len(spec.choices) == len(spec.weights) and min(spec.weights) > 0
            assert all(in_range(name, choice) for choice in spec.choices), name
        else:
            assert param.lo <= spec.lo <= spec.typical <= spec.hi <= param.hi, name
    for lanes, spec in presets.RADIUS_BY_LANES.items():
        assert PARAM_BY_NAME["radius"].lo <= spec.lo <= spec.typical <= spec.hi <= PARAM_BY_NAME["radius"].hi


def test_random_changes_are_valid_settings_on_the_slider_steps(draws):
    for changes in draws:
        assert list(changes) == PARAM_NAMES  # every parameter with a slider, in PARAMS order
        SimConfig(**changes)  # valid
        for name, value in changes.items():
            assert type(value) is type(getattr(DEFAULTS, name)), name
            assert in_range(name, value), (name, value)
            if not PARAM_BY_NAME[name].choices:
                assert on_step(name, value), (name, value)


def test_random_changes_are_reproducible_by_seed():
    first = presets.random_changes(11)
    assert first == presets.random_changes(11)
    other = presets.random_changes(12)
    assert sum(first[n] != other[n] for n in PARAM_NAMES) > len(PARAM_NAMES) // 2
    assert presets.random_changes(np.int64(11)) == first
    np.random.seed(0)
    expected = np.random.random()
    np.random.seed(0)
    presets.random_changes(5)
    assert np.random.random() == expected  # leaves the global random state alone


@pytest.mark.parametrize("seed", [-1, 1.5, True, "3", None])
def test_random_seed_must_be_a_whole_number(seed):
    with pytest.raises(ValueError, match="seed"):
        presets.random_changes(seed)


def test_new_seed_is_short_and_usable():
    seeds = {presets.new_seed() for _ in range(20)}
    assert all(type(s) is int and 0 <= s < 100_000 for s in seeds) and len(seeds) > 1
    presets.random_changes(seeds.pop())


@pytest.mark.parametrize("group", presets.GROUPS)
def test_randomizing_one_group_draws_only_its_parameters(group):
    changes = presets.random_changes(8, [group])
    assert list(changes) == [p.name for p in PARAMS if p.group == group]
    # each parameter has its own stream: on top of the full draw, a group draw gives the same values
    full = presets.random_changes(8)
    assert presets.random_changes(8, [group], base=presets.random_config(8)) == {k: full[k] for k in changes}
    SimConfig().updated(**changes)


def test_group_names_are_checked():
    assert presets.random_changes(8, "drivers") == presets.random_changes(8, ["Drivers"])
    both = presets.random_changes(8, ["Road", "demand"])
    assert set(both) == {p.name for p in PARAMS if p.group in ("Road", "Demand")}
    assert presets.random_changes(8, []) == {}
    with pytest.raises(ValueError, match="Drivers"):
        presets.random_changes(8, ["Wheels"])  # the message lists the groups


def test_random_config_keeps_the_seed_and_the_other_groups():
    base = presets.apply("etoile", SimConfig(seed=9))
    cfg = presets.random_config(4, base=base)
    assert cfg.seed == 9 and cfg == base.updated(**presets.random_changes(4, base=base))
    assert presets.random_config(4).seed is None
    drivers = presets.random_config(4, base=base, groups=["Drivers"])
    for p in PARAMS:
        if p.group != "Drivers":
            assert getattr(drivers, p.name) == getattr(base, p.name), p.name
    assert any(getattr(drivers, p.name) != getattr(base, p.name) for p in PARAMS if p.group == "Drivers")
    fixed = presets.random_config(4, base=base, groups=["Drivers"], fixed={"n_lanes": 2, "reaction_time": 0.5})
    assert fixed.n_lanes == 2 and fixed.reaction_time == 0.5 and fixed.seed == 9


def test_base_groups_and_fixed_are_keywords():
    """random_changes(seed, groups) and random_config(seed) take the rest by keyword only, so the two
    cannot be mixed up (they used to take groups and base in opposite orders)."""
    with pytest.raises(TypeError):
        presets.random_changes(3, None, SimConfig())
    with pytest.raises(TypeError):
        presets.random_config(3, SimConfig())
    with pytest.raises(TypeError):
        presets.random_config(3, ["Drivers"])


def test_fixed_settings_are_kept_and_the_links_follow_them():
    """Settings given explicitly (--set or a flag with --random) stay as given and the linked draws use
    them, while every other setting keeps its own stream's value."""
    free = presets.random_changes(40)
    assert free["traffic_mode"] == "closed" and free["density"] > presets.DENSITY_OPEN.hi
    opened = presets.random_changes(40, fixed={"traffic_mode": "open"})
    assert opened["traffic_mode"] == "open" and opened["density"] <= presets.DENSITY_OPEN.hi
    assert {k for k in free if free[k] != opened[k]} <= {"traffic_mode", "density"}
    rates = {8: [], 4: []}
    for seed in range(60):
        one_lane = presets.random_changes(seed, fixed={"n_lanes": 1})
        spec = presets.RADIUS_BY_LANES[1]
        assert one_lane["n_lanes"] == 1 and spec.lo <= one_lane["radius"] <= spec.hi
        for arms in rates:
            ch = presets.random_changes(seed, fixed={"n_arms": arms, "traffic_mode": "open"})
            assert 0.75 * 2 / arms - 0.025 <= ch["exit_prob"] <= 1.25 * 2 / arms + 0.025
            rates[arms].append(ch["entry_rate"])
    # 8 arms: longer trips past more entrances, so more circulating traffic and less demand per entrance
    assert np.median(rates[8]) < np.median(rates[4])
    # groups and fixed together: fixed names of the groups come back unchanged, fixed text is converted
    drivers = presets.random_changes(5, ["Drivers"], fixed={"reaction_time": "0.5", "n_lanes": 1})
    assert list(drivers) == [p.name for p in PARAMS if p.group == "Drivers"] and drivers["reaction_time"] == 0.5
    assert drivers["reaction_std"] <= presets.REACTION_CV_MAX * 0.5 + 0.025
    with pytest.raises(ValueError, match="wheels"):
        presets.random_changes(5, fixed={"wheels": 3})
    with pytest.raises(ValueError):
        presets.random_changes(5, fixed={"road_condition": "mud"})


def test_random_values_are_linked_like_real_roundabouts(draws):
    half = {name: PARAM_BY_NAME[name].step / 2 + 1e-9 for name in PARAM_NAMES if not PARAM_BY_NAME[name].choices}
    for ch in draws:
        radius = presets.RADIUS_BY_LANES[ch["n_lanes"]]  # wider rings for more lanes
        assert radius.lo - half["radius"] <= ch["radius"] <= radius.hi + half["radius"]
        share = 2.0 / ch["n_arms"]  # a mean trip of about half the circle
        assert max(0.1, 0.75 * share) - half["exit_prob"] <= ch["exit_prob"] <= min(0.8, 1.25 * share) + \
            half["exit_prob"]
        length = ch["car_length"] + ch["truck_share"] * (ch["truck_length"] - ch["car_length"])
        assert ch["density"] <= presets.DENSITY_CAP * 1000.0 / (length + ch["min_gap"]) + half["density"]
        if ch["traffic_mode"] == "open":  # only the start: near where open traffic settles
            assert ch["density"] <= presets.DENSITY_OPEN.hi + half["density"]
        if ch["truck_share"] == 0:
            assert ch["truck_length"] == DEFAULTS.truck_length
        assert ch["reaction_std"] <= presets.REACTION_CV_MAX * ch["reaction_time"] + half["reaction_std"]
        assert ch["headway_std"] <= presets.HEADWAY_CV_MAX * ch["time_headway"] + half["headway_std"]
    # demand follows the layout's capacity: single-lane rings get less than multilane ones
    rates = {lanes: np.median([ch["entry_rate"] for ch in draws if ch["n_lanes"] == lanes and ch["n_arms"] == 4])
             for lanes in (1, 2)}
    assert rates[1] < rates[2]


def test_random_distributions_are_roughly_as_specified(draws):
    n = len(draws)
    for name in CHOICES:
        spec = presets.RANDOM_RANGES[name]
        counts = Counter(ch[name] for ch in draws)
        assert set(counts) <= set(spec.choices), name
        for choice, weight in zip(spec.choices, spec.weights):
            assert abs(counts[choice] / n - weight / sum(spec.weights)) < 0.03, (name, choice)
    independent = [name for name, spec in presets.RANDOM_RANGES.items()
                   if isinstance(spec, presets.Numeric) and name not in
                   ("radius", "density", "exit_prob", "entry_gap", "entry_rate", "lateral_accel", "truck_length",
                    "reaction_std", "headway_std")]
    assert len(independent) >= 15
    for name in independent:
        spec, step = presets.RANDOM_RANGES[name], PARAM_BY_NAME[name].step
        values = np.array([ch[name] for ch in draws], dtype=float)
        assert spec.lo - step / 2 <= values.min() and values.max() <= spec.hi + step / 2, name
        assert abs(values.mean() - (spec.lo + spec.typical + spec.hi) / 3) < 0.05 * (spec.hi - spec.lo), name
    rates = np.array([ch["entry_rate"] for ch in draws])
    assert rates.min() >= 0.03 and rates.max() <= 0.22 and 0.08 <= np.median(rates) <= 0.14  # veh/h 290-500
    closed = np.array([ch["density"] for ch in draws if ch["traffic_mode"] == "closed"])
    assert abs(closed.mean() - 50.0) < 3.0  # triangular 5-35-110, rarely capped
    gaps = np.array([ch["entry_gap"] for ch in draws])
    assert 1.5 <= gaps.min() and gaps.max() <= 3.5 and 2.2 <= np.median(gaps) <= 2.6


def test_random_settings_run(draws):
    for seed in range(8):
        sim = RingSim(SimConfig(seed=seed).updated(**draws[seed]))
        sim.step(200)
        assert_valid(sim)
        if sim.cfg.n_arms:
            assert sim.n_entered > 0 or sim.n_arrived == 0


def test_sampler_stays_in_its_bounds():
    spec = presets.Numeric(2.0, 8.0, 4.0)
    rng = np.random.default_rng(0)
    values = np.array([presets._sample(spec, rng) for _ in range(4000)])
    assert values.min() >= 2.0 and values.max() <= 8.0
    assert abs(values.mean() - 14 / 3) < 0.1  # triangular: (lo + typical + hi) / 3
    assert presets._sample(spec, rng, lo=5.0, hi=5.0) == 5.0


def test_specs_are_checked():
    with pytest.raises(ValueError):
        presets.Numeric(1.0, 2.0, 3.0)
    with pytest.raises(ValueError):
        presets.Choice(("a", "b"), (1.0,))


def test_snap_puts_values_on_the_slider():
    assert presets.snap("radius", 18.3) == 18.0 and presets.snap("radius", 18.5) == 18.0  # 1 m steps (round half even)
    assert repr(presets.snap("exit_prob", 0.1 + 0.2)) == "0.3"
    assert presets.snap("density", 1000) == 300.0 and presets.snap("density", 1000, clip=False) == 1000.0
    assert presets.snap("n_lanes", 2.6) == 3 and type(presets.snap("n_lanes", 2.6)) is int
    assert presets.snap("road_condition", "wet") == "wet"
    with pytest.raises(ValueError):
        presets.snap("road_condition", "mud")


def test_slider_ranges_hold_the_real_layouts():
    radius, arms = PARAM_BY_NAME["radius"], PARAM_BY_NAME["n_arms"]
    assert radius.lo <= 12.25 and arms.hi >= 12  # UK compact roundabouts; the Etoile's 12 avenues
    assert (DEFAULTS.radius, DEFAULTS.n_arms) == (80.0, 4)  # widening the ranges left the defaults alone
    # The presets keep their published radius between the 1 m steps, and the viewers show it exactly.
    from roundabout_sim.app import format_value
    for name, shown in (("us_single_lane", "18.25"), ("us_two_lane", "22.25"), ("etoile", "67.25"),
                        ("sugiyama2008", "36.6"), ("stern2018", "41.4"), ("tadaki2013", "50")):
        r = presets.apply(name).radius
        assert radius.lo <= r <= radius.hi and format_value(radius, r) == shown, name


# ---------------------------------------------------------- Latin hypercube
@pytest.mark.parametrize("n", [1, 10, 37])
def test_latin_hypercube_hits_every_stratum_once(n):
    ranges = {"x": (0.0, 1.0), "density": (5, 110), "radius": (12, 70), "entry_rate": (0.03, 0.5),
              "max_queue": (20, 100)}
    rows = presets.latin_hypercube(n, ranges, seed=3)
    assert len(rows) == n and all(list(row) == list(ranges) for row in rows)
    for name, (lo, hi) in ranges.items():
        values = [row[name] for row in rows]
        assert all(lo <= v <= hi for v in values), name
        assert strata(values, lo, hi, n) == list(range(n)), name
        if name in PARAM_BY_NAME:
            assert all(on_step(name, v) and type(v) is type(getattr(DEFAULTS, name)) for v in values), name


@pytest.mark.parametrize("name, lo, hi, n", [
    ("entry_rate", 0.0, 0.2, 20), ("exit_prob", 0.05, 0.3, 5), ("density", 10, 40, 30), ("radius", 20, 36, 16),
    ("time_headway", 0.7, 1.4, 7), ("max_queue", 20, 60, 40)])
def test_latin_hypercube_with_one_slider_step_per_stratum_never_repeats(name, lo, hi, n):
    """n = (hi - lo) / step: each stratum holds exactly one slider value, on its lower boundary, where float
    rounding used to put it in the stratum below (and the design repeated values)."""
    for seed in range(25):
        values = [row[name] for row in presets.latin_hypercube(n, {name: (lo, hi)}, seed)]
        assert len(set(values)) == n, (seed, sorted(values))
        assert strata(values, lo, hi, n) == list(range(n)) and all(on_step(name, v) for v in values)


def test_latin_hypercube_converts_settings_without_a_slider():
    """SimConfig fields without a slider get SimConfig's types (text levels as a sweep's comma list passes
    them, a whole-number seed); names that are no setting pass through."""
    rows = presets.latin_hypercube(4, {"lane_width": ["3.0", "3.5"], "dt": ("0.05", "0.1"), "x": ["a", "b"],
                                       "density": ["20", "30.5"]}, seed=1)
    assert sorted(row["lane_width"] for row in rows) == [3.0, 3.0, 3.5, 3.5]
    assert all(type(row["lane_width"]) is float and type(row["dt"]) is float for row in rows)
    assert sorted(row["density"] for row in rows) == [20.0, 20.0, 30.5, 30.5]
    assert sorted(row["x"] for row in rows) == ["a", "a", "b", "b"]
    for row in rows:
        DEFAULTS.updated(**{k: v for k, v in row.items() if k != "x"})
    seeds = [row["seed"] for row in presets.latin_hypercube(5, {"seed": (1, 50)}, seed=2)]
    assert all(type(s) is int and 1 <= s <= 50 for s in seeds) and strata(seeds, 1, 50, 5) == list(range(5))
    few = [row["seed"] for row in presets.latin_hypercube(6, {"seed": (1, 3)}, seed=2)]
    assert sorted(few) == [1, 1, 2, 2, 3, 3]  # fewer whole numbers than samples: dealt out evenly
    assert presets.latin_hypercube(2, {"seed": (4, 4)}, seed=0) == [{"seed": 4}] * 2
    with pytest.raises(ValueError, match="whole number"):
        presets.latin_hypercube(3, {"n_lanes": (2.2, 2.8)}, seed=0)


def test_latin_hypercube_deals_whole_numbers_and_choices_evenly():
    rows = presets.latin_hypercube(10, {"n_lanes": (1, 5), "road_condition": ["dry", "wet", "snow"],
                                        "traffic_mode": ("closed", "open"), "n_arms": [3, 4, 6],
                                        "time_headway": (0.7, 1.4)}, seed=1)
    assert Counter(row["n_lanes"] for row in rows) == {1: 2, 2: 2, 3: 2, 4: 2, 5: 2}
    assert all(type(row["n_lanes"]) is int and type(row["n_arms"]) is int for row in rows)
    assert sorted(Counter(row["road_condition"] for row in rows).values()) == [3, 3, 4]
    assert Counter(row["traffic_mode"] for row in rows) == {"closed": 5, "open": 5}
    assert sorted(Counter(row["n_arms"] for row in rows).values()) == [3, 3, 4]
    # 0.7-1.4 s holds 8 slider values for 10 samples: each used once or twice
    assert sorted(Counter(row["time_headway"] for row in rows).values()) == [1] * 6 + [2] * 2
    with pytest.raises(ValueError):
        presets.latin_hypercube(4, {"road_condition": ["dry", "slush"]}, seed=1)


def test_latin_hypercube_is_reproducible_and_columns_are_independent():
    ranges = {"density": (5, 110), "mean_speed": (8, 14)}
    first = presets.latin_hypercube(12, ranges, seed=5)
    assert first == presets.latin_hypercube(12, ranges, seed=5)
    assert first != presets.latin_hypercube(12, ranges, seed=6)
    more = presets.latin_hypercube(12, {**ranges, "politeness": (0, 1)}, seed=5)
    assert [{k: row[k] for k in ranges} for row in more] == first


def test_latin_hypercube_edge_cases():
    rows = presets.latin_hypercube(3, {"radius": (36.6, 36.6), "other": (2.0, 2.0)}, seed=0)
    assert [row["radius"] for row in rows] == [36.6] * 3 and [row["other"] for row in rows] == [2.0] * 3
    beyond = presets.latin_hypercube(4, {"density": (300, 400)}, seed=0)  # past the slider: the grid goes on
    assert all(300 <= row["density"] <= 400 and float(row["density"]).is_integer() for row in beyond)
    between = presets.latin_hypercube(4, {"radius": (36.3, 36.4)}, seed=0)  # no slider value inside
    assert all(36.3 <= row["radius"] <= 36.4 for row in between)
    for n, ranges, seed in ((0, {"x": (0, 1)}, 0), (3, {"x": (1, 0)}, 0), (3, {"x": []}, 0),
                            (3, {"x": (0, 1)}, -1), (3, {"x": (0, float("inf"))}, 0)):
        with pytest.raises(ValueError):
            presets.latin_hypercube(n, ranges, seed)


def test_latin_hypercube_over_the_plausible_ranges_gives_valid_settings():
    ranges = presets.plausible_ranges()
    assert list(ranges) == PARAM_NAMES
    rows = presets.latin_hypercube(40, ranges, seed=2)
    for row in rows:
        DEFAULTS.updated(**row)
        for name, value in row.items():
            assert in_range(name, value), (name, value)
            if not PARAM_BY_NAME[name].choices:
                assert on_step(name, value), (name, value)
    assert list(presets.plausible_ranges(["Drivers"])) == [p.name for p in PARAMS if p.group == "Drivers"]
