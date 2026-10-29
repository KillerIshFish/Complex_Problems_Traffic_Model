"""The robotaxi (SPEC 2, Hu et al. 2023, Waymo Open Dataset): the IDM with its direct leader at a driver's time
gap x robotaxi_headway (a truck body x TRUCK_HEADWAY_FACTOR on top) and a standstill gap ROBOTAXI_S0_EXTRA
longer, a driver's acceleration and comfortable braking, no noise, no spread between robotaxis, the reaction
delay of the Reaction time setting; its steady state is the IDM equilibrium with those values, and a ring of
robotaxis settles there."""

from __future__ import annotations

import numpy as np
import pytest

from roundabout_sim.config import SimConfig
from roundabout_sim.engine import (HUMAN, ROBOTAXI, ROBOTAXI_S0_EXTRA, TRUCK_ACCEL_FACTOR, TRUCK_HEADWAY_FACTOR,
                                   RingSim)

SPREAD = dict(speed_std=1.5, headway_std=0.3, reaction_std=0.2)
RING = dict(n_lanes=1, n_arms=0, radius=60.0)


def robotaxis(**config) -> RingSim:
    return RingSim(SimConfig(**{"seed": 1, "auto_share": 1.0, "auto_kind": "robotaxi", **config}))


def test_the_s0_extra_is_hus_jam_spacing_difference():
    # (10.0 - 4.78) - (7.88 - 4.6): Hu 2023 jam spacings AV-HV / HV-HV less the IDM-implied vehicle lengths.
    assert ROBOTAXI_S0_EXTRA == pytest.approx((10.0 - 4.78) - (7.88 - 4.6))
    assert SimConfig().robotaxi_headway == 1.3 and 3.49 / 2.69 == pytest.approx(1.3, abs=0.005)


def test_time_gap_standstill_gap_acceleration_and_delay():
    sim = robotaxis(truck_share=0.25, reaction_time=0.7, min_gap=1.5, robotaxi_headway=1.4, density=30, **SPREAD)
    live = sim.alive
    car, truck = live & ~sim.truck, live & sim.truck
    assert (sim.kind[live] == ROBOTAXI).all() and car.any() and truck.any()
    T, a = sim.eff.time_headway, sim.eff.max_accel
    assert (sim.headway[car] == 1.4 * T).all()  # the setting x the factor, no spread
    assert (sim.headway[truck] == TRUCK_HEADWAY_FACTOR * (1.4 * T)).all()  # the truck body on top
    assert (sim.s0[live] == 1.5 + ROBOTAXI_S0_EXTRA).all()  # a truck too
    assert (sim.accel[car] == a).all() and (sim.accel[truck] == TRUCK_ACCEL_FACTOR * a).all()
    assert (sim.delay[live] == 7).all()  # Reaction time, no spread
    v0 = sim.desired_speed(np.flatnonzero(car))
    assert np.unique(v0).size == 1 and v0[0] == min(sim.eff.mean_speed, np.sqrt(sim.eff.lateral_accel * sim.radii[0]))
    # The road condition scales its time gap like a driver's (the robotaxi weather policy is S3's).
    assert not sim.set(road_condition="wet")
    assert sim.headway[car] == pytest.approx(1.4 * sim.eff.time_headway) and sim.eff.time_headway > T
    # Humans beside them keep their own spread and the plain standstill gap.
    mixed = robotaxis(auto_share=0.5, min_gap=1.5, density=40, **SPREAD)
    human = mixed.alive & (mixed.kind == HUMAN)
    assert (mixed.s0[human] == 1.5).all() and np.unique(mixed.headway[human]).size > 1


def test_robotaxis_drive_without_noise():
    """A single-lane ring of robotaxis runs the same with or without driver noise (it is drawn, then masked;
    nothing else draws from the main stream there), unlike the same ring of drivers."""
    runs = {(share, noise): robotaxis(auto_share=share, driver_noise=noise, density=45, init_spacing="random",
                                      **RING)
            for share in (0.0, 1.0) for noise in (0.0, 0.7)}
    for sim in runs.values():
        sim.step(500)
    np.testing.assert_array_equal(runs[1.0, 0.0].theta, runs[1.0, 0.7].theta)
    np.testing.assert_array_equal(runs[1.0, 0.0].v, runs[1.0, 0.7].v)
    assert not np.array_equal(runs[0.0, 0.0].v, runs[0.0, 0.7].v)


def test_the_steady_state_is_the_idm_equilibrium_with_the_robotaxis_values():
    sim = robotaxis(density=40, **RING)
    slots = np.flatnonzero(sim.alive)[:3]
    T, s0 = 1.3 * sim.eff.time_headway, sim.cfg.min_gap + ROBOTAXI_S0_EXTRA
    gaps = np.array([s0 - 0.1, 10.0, 20.0])
    v0 = sim.desired_speed(slots)
    v = sim.equilibrium_speed(gaps, v0, slots)
    assert v[0] == 0.0  # closer than its standstill gap: it stands
    residual = 1 - (v[1:] / v0[1:]) ** 4 - ((s0 + v[1:] * T) / gaps[1:]) ** 2
    assert np.abs(residual).max() < 1e-9 and (v[1:] > 0).all()
    human = RingSim(SimConfig(seed=1, density=40, **RING))
    h = human.equilibrium_speed(gaps, v0, np.flatnonzero(human.alive)[:3])
    assert (v[1:] < h[1:]).all()  # longer time and standstill gaps: slower at the same gap
    # A ring of robotaxis started at random gaps settles there: evenly spaced at the equilibrium speed.
    ring = robotaxis(density=40, init_spacing="random", **RING)
    ring.step(6000)
    idx, gap = ring.bumper_gaps()
    assert gap.std() < 1e-2
    np.testing.assert_allclose(ring.v[idx], ring.equilibrium_speed(gap, ring.desired_speed(idx), idx), atol=1e-3)
    s, vv, v_free = gap.mean(), ring.v[idx].mean(), ring.desired_speed(idx)[0]
    assert s == pytest.approx((s0 + vv * T) / np.sqrt(1 - (vv / v_free) ** 4), rel=1e-3)


def test_a_stopped_robotaxi_keeps_its_longer_standstill_gap():
    """Behind a standing car, a robotaxi comes to rest about its standstill gap behind it (the IDM's s0)."""
    sim = robotaxis(density=1, driver_noise=0.0, traffic_mode="open", **RING)  # (open: no density control)
    sim.alive[:] = False
    lead, follow = sim._spawn(np.zeros(2, dtype=int), np.array([0.0, -60.0 / sim.radii[0]]), np.array([0.0, 8.0]))
    sim.exit_at[[lead, follow]] = np.inf
    for _ in range(1500):
        sim.v[lead] = 0.0
        sim.theta[lead] = 0.0
        sim.step()
    gap = (sim.theta[lead] - sim.theta[follow]) % (2 * np.pi) * sim.radii[0] - sim.cfg.car_length
    assert sim.v[follow] < 0.05 and gap == pytest.approx(sim.cfg.min_gap + ROBOTAXI_S0_EXTRA, abs=0.3)
