"""The stern2018 human ring after the C1 calibration (S0 validation V6): with waves it carries Stern et al.
2018's flow (Table 3: 1,755-1,828 veh/h, speed SD 2.36-3.85 m/s), and the mechanism behind the change: its
drivers are string-stable on the ring, so the waves are noise-driven and rarely stop anyone, whereas DRIVERS
alone turn one braking car into a lasting stop-and-go jam (the old preset's 1,250 veh/h). Deterministic: fixed
seeds."""

from __future__ import annotations

import numpy as np

from roundabout_sim import presets
from roundabout_sim.config import SimConfig
from roundabout_sim.engine import RingSim
from roundabout_sim.experiments import pooled_sd

# The stern2018 drivers before C1: DRIVERS and the defaults (mean_speed was the preset's 11 m/s).
DRIVERS_ALONE = dict(presets.DRIVERS, max_accel=SimConfig().max_accel, comfort_decel=SimConfig().comfort_decel,
                     anticipation=SimConfig().anticipation, mean_speed=11.0)


def ring_window(cfg: SimConfig, warm: float = 300.0, measure: float = 300.0) -> tuple[float, float, float]:
    """(flow [veh/h] = cars / lane length x mean speed, Stern's throughput; pooled speed SD [m/s]; share of the
    seconds with a car below 0.5 m/s) over ``measure`` s after ``warm`` s, sampled once a second."""
    sim = RingSim(cfg)
    sim.step(int(round(warm / cfg.dt)))
    means, stds, standing = [], [], 0
    for _ in range(int(measure)):
        sim.step(int(round(1.0 / cfg.dt)))
        v = sim.v[sim.alive]
        means.append(v.mean())
        stds.append(v.std())
        standing += v.min() < 0.5
    flow = 3600.0 * float(np.mean(means)) * int(sim.alive.sum()) / float(sim.lane_len[0])
    return flow, pooled_sd(stds, means), standing / measure


def test_stern_ring_with_waves_carries_sterns_flow():
    """Seeds 0-3, 300 s + 300 s: every run has Stern-sized waves and the flow is Stern's, not the old 35 % less
    (this version: 1,615-1,752 veh/h, SD 2.61-3.37 m/s; the old drivers 1,163-1,398 veh/h)."""
    runs = [ring_window(presets.apply("stern2018", SimConfig(seed=seed))) for seed in range(4)]
    flows, sds, _ = zip(*runs)
    assert all(2.36 <= sd <= 3.85 for sd in sds)  # Stern Table 3, with waves
    assert 1650.0 <= float(np.median(flows)) <= 1900.0  # Stern: 1,755-1,828 (a 5-minute window scatters more)
    old = [ring_window(presets.apply("stern2018", SimConfig(seed=seed)).updated(**DRIVERS_ALONE))[0]
           for seed in range(4)]
    assert float(np.median(flows)) > 1.3 * float(np.median(old))


def test_stern_drivers_are_string_stable_and_drivers_alone_are_not():
    """Without noise, one car braking hard for 2 s fades out with the calibrated drivers but grows into a lasting
    jam that stops cars with DRIVERS alone: the C1 mechanism (noise-driven waves against full stop-and-go)."""
    for changes, stable in (({}, True), (DRIVERS_ALONE, False)):
        sim = RingSim(presets.apply("stern2018", SimConfig(seed=0)).updated(**{**changes, "driver_noise": 0.0}))
        sim.step(600)
        assert sim.v[sim.alive].std() < 0.2  # an even start without noise stays (nearly) even
        sim.brake(int(np.flatnonzero(sim.alive)[0]), 2.0)
        lowest = np.inf
        while sim.time < 600.0:
            sim.step(10)
            if sim.time > 100.0:
                lowest = min(lowest, float(sim.v[sim.alive].min()))
        spread = float(sim.v[sim.alive].std())
        if stable:
            assert spread < 0.5 and lowest > 3.0
        else:
            assert spread > 1.5 and lowest < 0.5
