"""Vectorised multi-lane roundabout (ring road) traffic engine.

Cars drive anticlockwise on concentric circular lanes. A car's state is its
lane, its angle ``theta`` on the ring (radians, increasing in the driving
direction) and its speed. Every step:

1. **Car following** - each driver's acceleration comes from the Intelligent
   Driver Model (IDM) towards the cars ahead in its lane. Drivers react to
   every car within ``reaction_distance`` (multi-anticipation: seeing further
   ahead damps waves; too short and they notice the car in front too late),
   act on what they perceived ``reaction_time`` ago, and add a little random
   noise. A hard constraint stops anyone driving into the car ahead.
2. **Exits** - a car whose route ends at an exit leaves if it reached it on
   the outer lane; otherwise it misses the exit and goes round again. In
   closed mode every leaver is replaced by a car queued at a random entrance.
3. **Lane changes** - leavers move outwards in time for their exit; other
   drivers consider a change ``lane_change_rate`` times per second and make it
   if it is safe (and, in "advantage" mode, faster - the MOBIL model).
4. **Arrivals** (open mode) - cars arrive at each entrance at random
   (Poisson, ``entry_rate`` per second, shaped by ``demand_imbalance`` and
   ``demand_profile``) and join its queue; arrivals that find ``max_queue``
   cars waiting are turned away (blocked).
5. **Entries** - queued cars merge into the outer lane when the gap to
   circulating traffic is at least ``entry_gap`` seconds and more than
   ``follow_up`` seconds after the previous car from that entrance (on the
   first step after it: 1 s means 1.1 s at the default time step). In closed
   mode at most ``MAX_QUEUE`` cars wait per entrance; beyond that a leaver's
   replacement is slotted straight into the ring, so the ring never drains
   into queues. Each queue keeps its cars' arrival times (oldest first), so
   every entry knows how long the car waited.

In closed mode density is held as cars on the ring per km of lane: raising it
inserts cars into the largest gaps, lowering it removes random cars. In open
mode it only sets the cars at the start (and once more whenever it is changed
live); after that arrivals and exits decide how many cars are on the ring.

Vehicles and drivers differ: a ``truck_share`` of the vehicles are trucks
(``truck_length`` long, slower to accelerate, slower and keeping a longer time
gap, see TRUCK_*), and each driver gets their own reaction time and time gap
around the settings (``reaction_std``, ``headway_std``). A vehicle's ``theta``
is its centre, so the bumper gap to the leader is the distance between their
centres less half of each one's length. The ``road_condition`` scales the
settings the dynamics use (desired speed, acceleration, braking, time gap,
entry gap, follow-up, cornering and the braking limit; ROAD_CONDITIONS), all
through one set of effective values, ``RingSim.eff`` (dry: the settings
themselves); ``weather_scale`` sets how strongly the drivers' behaviour follows the
road (road_factors: the tyres' grip always applies in full).

Every circulating vehicle passing an entrance's entry point (where entering cars
merge: its conflict point) is logged (``conflict_log``) and counted per entrance
(``arm_passed``, ``arm_passed_outer``): the entrance's conflicting flow. With
``subject_arm`` on (open mode) entrance 0 is a capacity test: its queue is kept
full without arrivals and every car leaves at the first exit after it (with 2+
lanes the first far enough on to leave from any lane), so each car from the
other entrances passes it exactly once and its conflicting flow is their entry
flow, set by the Entrance chance (SUBJECT_*; the experiments' subject_* columns).

Every vehicle has a ``kind`` (KIND_NAMES: a human driver, or an automated
vehicle of the run's ``auto_kind``) and, if automated, a ``company``. An
``auto_share`` of the new vehicles are automated: each new vehicle draws three
uniforms of a stream of its own (``rng_kinds``; the capacity test's top-up
``rng_kinds_test``) when it is created, so at the same seed a higher share
automates the same vehicles and more (monotone coupling), and the other streams
draw exactly what they draw without automated vehicles (a vehicle's human draws
are made and then masked). Automated vehicles
follow only the car in front, without noise and without a spread between them
(one software; robotaxis keep the drivers' time-gap spread), with their kind's
own time gap and standstill gap (``_set_driving``) and command
(``_automated_command``: the IDM for a robotaxi, the PATH ACC law for an ACC
car, PATH CACC for a CAV behind a cooperating CAV in strings of at most
``platoon_max``, else the ACC law), clipped to the braking limit and passed
through a safety layer (``_automated_limits``); the truck body is independent
of the kind (truck factors apply on top). With ``auto_share`` 0 nothing is
drawn and every run is the model without kinds, bit for bit. Observers count
time-to-collision episodes and exposure by follower-leader pair type
(``_observe_following``), the safety layer's interventions and harsh braking
by group.

Wave smoothers (``smoothers``, a count) are drivers' cars picked without
chance after the initial placement, at evenly spaced angles (``_sync_smoothers``;
a live change converts or hands back the nearest). From ``smoother_start`` on
they drive by FollowerStopper or PI with saturation (Stern et al. 2018), or as
a noise-free placebo of their driver (``_law_smoother``), through the same clip
and safety layer; before it their drivers drive them. They never leave, change
lane or get removed by the density control. With ``smoothers`` 0 nothing changes.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import asdict, dataclass, fields

import numpy as np

from roundabout_sim.config import PARAM_BY_NAME, SimConfig

TWO_PI = 2.0 * np.pi
LC_ZONE = 50.0  # [m] of ring a leaver needs per lane it must cross to reach the outer lane
LC_COOLDOWN = 2.0  # [s] minimum time between two lane changes of one driver
LC_THRESHOLD = 0.2  # [m/s2] acceleration gain needed for an "advantage" lane change
LC_DURATION = 1.5  # [s] how long a lane change takes on screen (rendering only)
NOISE_TAU = 2.0  # [s] correlation time of the acceleration noise
SAFE_MARGIN = 0.3  # [m] bumper gap enforced by the no-overlap constraint
MAX_REACTION = 2.0  # [s] longest supported reaction time
PERTURB_DECEL = 4.0  # [m/s2] braking of a car the user tells to brake
MAX_DECEL = 9.0  # [m/s2] physical braking limit; needing more counts as harsh braking
SLOW_FRACTION = 0.3  # a car counts as jammed below this fraction of its free speed
RESUME_FRACTION = 0.6  # ...and as moving again above this (hysteresis for counting stops)
MAX_QUEUE = 2  # cars waiting per entrance in closed mode (open mode: cfg.max_queue)
PEAK_PERIOD = 600.0  # [s] one rise and fall of the "peak" demand profile
PEAK_SWING = 0.5  # the "peak" profile runs from (1 - this) to (1 + this) times the mean demand
LOG_HORIZON = 600.0  # [s] how long event logs are kept for rolling rates
EMERGENCY_HOLD = 2.0  # [s] an emergency episode ends after this long without one
LEAVER_PATIENCE = 5.0  # [s] a leaver held at a standstill by merging gives up its exit
SMALL_FILL = 3  # deficits up to this many cars are slotted into random gaps
ANTICIPATION = 5  # most cars ahead a driver reacts to (within the reaction distance): cfg.anticipation's default
HEADWAY_FLOOR = 0.5  # nobody keeps less than this share of the Following time gap (cf. desired speeds)
TEMPERAMENT_CLIP = 2.5  # per-driver draws are standard normals clipped to +- this
# Capacity test (subject_arm on, open mode): entrance SUBJECT_ARM keeps a standing queue and every car leaves at
# the first exit after it (with 2+ lanes the first far enough on: RingSim._subject_exit), so its conflicting flow
# is the other entrances' entry flow (RingSim._assign_routes).
# *model*: any queue of 2 or more keeps a car waiting at every step (the top-up comes before the entries, which
# take at most one car per entrance per step); 10 shows as a standing queue in the viewer.
SUBJECT_ARM = 0
SUBJECT_QUEUE = 10

# Vehicle kinds (per-car ``kind``, int8): a human driver (HUMAN) or one of four automated kinds. A run's automated
# vehicles are all of one kind (auto_kind: ACC, CAV or ROBOTAXI), an auto_share of the new vehicles; smoothers
# (wave-dampening automated vehicles) come separately, as a count.
HUMAN, ACC, CAV, ROBOTAXI, SMOOTHER = range(5)
KIND_NAMES = ("human", "acc", "cav", "robotaxi", "smoother")  # by code
AUTO_KINDS = {"acc": ACC, "cav": CAV, "robotaxi": ROBOTAXI}  # auto_kind -> its code (the auto_share kinds)
RECOGNISABLE = frozenset({CAV, ROBOTAXI, SMOOTHER})  # drivers can tell them apart (an ACC car looks like any car)
DRIVERLESS = frozenset({CAV, ROBOTAXI})  # nobody in them takes over (stalls, the AV weather policy)
CONNECTED = frozenset({CAV})  # talk to each other (V2V: CACC strings, cooperation)
# A robotaxi keeps a standstill gap this much longer than a driver's [m] (Hu et al. 2023, Waymo Open Dataset:
# jam spacings 10.0 m for an AV behind a human against 7.88 m for a human behind a human, Sec. III-C, Table IV,
# less the vehicle lengths the IDM calibration implies, 4.78 and 4.6 m, Sec. IV-B: 5.22 - 3.28 m; *derived*).
ROBOTAXI_S0_EXTRA = 1.94
# A CAV's time gap when it follows in ACC mode (no cooperating CAV ahead: _law_cav): the gaps the
# drivers of the PATH ACC cars chose and how often (Liu et al. 2018 p. 21, from Nowakowski et al. 2010: 1.1 s
# 50.4 %, 1.6 s 18.5 %, 2.2 s 31.1 %; mean 1.53 s, *derived*). Each vehicle's third kind uniform picks one.
ACC_GAPS = (1.1, 1.6, 2.2)  # [s]
ACC_GAP_SHARES = (0.504, 0.185, 0.311)
ACC_GAP_EDGES = (0.504, 0.689)  # the uniform's bounds between them (cumulative shares)
# Each automated kind's longitudinal law (RingSim._automated_command calls RingSim._law_<name> for its cars).
# robotaxi: the IDM with the direct leader (Hu et al. 2023 calibrate the IDM to Waymo's following, Sec. IV).
# acc: the PATH ACC law (_law_acc). cav: PATH CACC behind a cooperating CAV, else the ACC law (_law_cav).
# smoother: FollowerStopper, PI with saturation or the quiet placebo (_law_smoother).
AUTO_LAWS = {ROBOTAXI: "idm", ACC: "acc", CAV: "cav", SMOOTHER: "smoother"}
# ACC and CACC (RingSim._law_acc, _law_cav). Every gain and gap below was fitted or chosen for 15-34 m/s on
# freeways and test tracks (Milanes & Shladover 2014 Secs. 3-4; Gunter et al. 2019 Secs. III-IV; Liu et al.
# 2018): at roundabout speeds (3-12 m/s) these laws are an extrapolation, not a calibration.
# The PATH ACC gap law, fitted to a production car's realised speed, so it needs no lag of its own (Milanes &
# Shladover 2014 Eq. 5, Sec. 4.2; the same law: Xiao, Wang & van Arem 2017 Eq. 3, Liu et al. 2018 Eq. 34,
# FHWA-OR-RD-22-11 Eq. 7-2): a = ACC_K1 (gap - s0 - T v) + ACC_K2 (v_lead - v). It is string-unstable (M&S 2014
# Table 3 and Sec. 5; *derived*: the constant-time-gap condition k1 T^2 + 2 k2 T >= 2 needs T >= 2.66 s).
ACC_K1 = 0.23  # [1/s2]
ACC_K2 = 0.07  # [1/s]
# Its speed (cruising) mode a = ACC_SPEED_GAIN (v_set - v) (Xiao 2017 Eq. 2, p. 5: 0.3-0.4 in the literature; Liu
# 2018 Eq. 33). *assumption*: the smaller of the two commands applies (the sources switch modes at 120 m / 100 m
# clearance with hysteresis, Liu 2018 pp. 21-22; the minimum needs no stored mode and picks the speed mode far
# behind a leader and the gap mode close to it).
ACC_SPEED_GAIN = 0.4  # [1/s]
# The ACC spacing margin, Xiao 2017 Eq. 6 (its d0 less the 5 m vehicle length it includes, *derived*): ACC_S0 up to
# ACC_S0_LOW, ACC_S0_K / v - 5 m up to ACC_S0_HIGH, 0 above. It is the ACC car's standstill gap s0. As in Xiao's
# gap error it is taken at the car's own speed, so between 10.8 and 15 m/s its slope -75/v^2 enters the dynamics:
# the equilibrium spacing s0(v) + T v grows by only T - 75/v^2 per m/s, the effective time gap (*derived*: 0.46 s
# just above 10.8 m/s and 0.58 s at 12 m/s for T 1.1 s; 0.77 s at 15 m/s). Linearised (dt 0.1 s), per-car peak
# string gain at 12 m/s 2.44 without a dead time and 5.64 with 0.5 s, against 1.59 and 2.32 where the margin is flat
# (*derived*), and the dead time a lone follower tolerates falls (ACC_DEAD_MAX). Below 0.643 s (75 / 10.8^2) the
# equilibrium spacing would shrink as speed rises just above 10.8 m/s: ACC_GAP_MIN. (Xiao's CACC margin has the same
# effect, smaller: CACC_S0_SLOPE.)
ACC_S0 = 2.0  # [m] (7 m - 5 m)
ACC_S0_K = 75.0  # [m2/s]
ACC_S0_LOW, ACC_S0_HIGH = 10.8, 15.0  # [m/s]
# [s] The shortest time gap the ACC law keeps (an ACC car's ACC time gap, the slider's lowest value, and a CAV's
# ACC-fallback gap, after the brand and road factors; a truck x TRUCK_HEADWAY_FACTOR on top): *derived*, above 75 /
# 10.8^2 = 0.643 s, so the effective time gap stays positive across Xiao's band (at least 0.057 s) and the
# equilibrium spacing grows with speed, and the smallest on 0.05 s steps at which the default ACC settings (1.1 s,
# 0.5 s) stay locally stable at every brand factor (worst spectral radius 0.99991, at brand factors 0.55-0.6; factors
# under 0.64 bind it).
ACC_GAP_MIN = 0.7
# [s] The longest ACC dead time, the brand factor included (_set_driving): *derived* (linearised follower behind a
# steady leader as the engine steps it, the whole command delayed by the buffer, dt 0.1 s), the delayed M&S loop
# goes locally unstable, a lone follower oscillating behind a steady leader, from a dead time of 0.8 s at an
# effective time gap of 0.6 s, 1.2 s at 1.1 s and 1.5 s at 2-2.4 s, but already from 0.7 s at 0.46 s (T 1.1 s
# just above 10.8 m/s, Xiao's slope) and 0.3 s at 0. Up to 0.6 s it is locally stable wherever the effective time
# gap is at least 0.4 s: at every ACC time gap and dead time of the sliders and brand factor the settings can make
# (companies 2-4 x brand_spread on its 0.05 steps: 41 factors, 0.4-1.6) where the margin is flat (0 of the 10,045
# combinations unstable), and at ACC time gaps of 1.1 s or more across Xiao's band. *derived* too: just above
# 10.8 m/s shorter time gaps tolerate less (0.3 s at 0.7 s, 0.4 s at 0.8-0.9 s, 0.5 s at 1.0 s), so 387 of the
# 10,045 combinations (3.9 %) still oscillate there; a flat margin would remove them. Gunter et al. 2019
# (Table II) fit dead times up to 0.99 s, but with gains k1 of 0.012-0.071 s^-2; the M&S gains (ACC_K1 = 0.23)
# were fitted with none (M&S 2014 Table 3: the ACC car's transfer function has no delay), so the longer dead times
# do not carry over to this law.
ACC_DEAD_MAX = 0.6
# PATH CACC as used for the HCM 7 CAV factors (FHWA-OR-RD-22-11 Eqs. 7-3 to 7-6, Sec. 7.2.2.1 p. 108; Liu et al.
# 2018 Eqs. 35-38, pp. 21-22): each update v += kp e + kd edot, e = gap - s0 - Tc v, edot = (v_lead - v) - Tc a
# (a: the follower's own acceleration, a_now), i.e. a = (CACC_KP e + CACC_KD edot) / CACC_PERIOD = 4.5 e + 0.125
# edot (*derived*). No first-order lag: fitted to realised speed like the ACC law (*derived*: with a 0.5 s lag a
# 2.70 m/s dip in a string grows to 3.43 m/s over 8 cars, against 2.71 m/s without).
CACC_KP = 0.45
CACC_KD = 0.0125
# [s] Its update interval (Liu 2018 p. 10: "0.1 second"; FHWA's VISSIM at 10 steps/s, p. 113). A discretised speed
# law, stable only for steps up to its period: *derived* (linearised loop of a follower behind a steady leader, as
# the engine steps it), dividing by CACC_PERIOD whatever the step it goes unstable (spectral radius > 1) from dt
# 0.56 s at a 0.7 s gap, 0.24 s at 1.5 s and 0.13 s at 2.4 s. So _law_cav divides by max(CACC_PERIOD, dt): up to the
# 0.1 s default step the law as published, at a longer step one update per step with the published per-update
# gains (dt is a field without a slider, --set dt=...).
CACC_PERIOD = 0.1
# [s] Even at its own 0.1 s period the published law goes unstable from a 2.84 s time gap (*derived*, as above),
# which a truck (x 1.5) of a slow brand (up to x 1.6) at CAV_PLATOON_GAP, or on ice (x 1.6), can reach. Liu 2018
# (pp. 21-22) regulates gaps by CACC only below a 2 s time gap (above it, speed mode), so *assumption*: beyond
# CACC_GAIN_GAP the gains fall as CACC_GAIN_GAP / Tc (the per-update speed feedback kp Tc stays at its 2 s value);
# *derived*: then stable for every Tc of 0.5-8 s at steps up to 1 s (radius <= 0.99), 0.2-8 s up to 0.2 s.
CACC_GAIN_GAP = 2.0
# The CACC spacing margin, Xiao 2017 Eq. 7 (d0 = 6.25 - 0.125 v less 5 m, *derived*): CACC_S0 at standstill,
# falling by CACC_S0_SLOPE per m/s to 0 at 10 m/s. *assumption*: the FHWA law has no standstill term of its own.
# Below 10 m/s it shortens the regulated gap by 0.125 s x v, so the effective time gap is Tc - 0.125 s: *derived*
# (linearised, dt 0.1 s), at Tc 0.7 s the string is then mildly string-unstable at low frequency (peak |G| 1.04
# per car; stable from Tc ~0.8 s), while above 10 m/s it is string-stable (|G| <= 1, tending to 1 at low
# frequency). Roundabout speeds are below 10 m/s.
CACC_S0 = 1.25  # [m]
CACC_S0_SLOPE = 0.125  # [s]
# The time gap of a CAV that would be member platoon_max + 1 of a CACC string (the gap between strings in
# FHWA-OR-RD-22-11's roundabout runs, p. 108; Liu 2018 pp. 24-25: t1 = 1.5 s behind a string at its maximum
# length). It follows by the CACC law and leads the next string.
CAV_PLATOON_GAP = 1.5  # [s]
# Wave smoothers (RingSim._law_smoother; the smoothers setting, a count): Stern et al. 2018, Transp. Res. C 89
# (arXiv 1705.01693), where one automated car among 21 on a 260 m ring damped the stop-and-go waves (Sec. 2.1, Table 3).
# FollowerStopper (Sec. 3.1, Eqs. 1-2): with dv = v_lead - v, three boundaries dx_k = FS_DX0[k] + min(dv, 0)^2 /
# (2 FS_DECEL[k]) (p. 5: at dv = -3 m/s, 7.5, 9.75 and 15 m) split the bumper gap dx into a stop (v_cmd = 0 up to
# dx1), a ramp to vbar = min(max(v_lead, 0), U) at dx2, a ramp from vbar to U at dx3 (the paper's third branch, vbar +
# (U - vbar)(dx - dx2)/(dx3 - dx2); Flow's code has U - v there) and U beyond, U = smoother_speed.
FS_DX0 = (4.5, 5.25, 6.0)  # [m]
FS_DECEL = (1.5, 1.0, 0.5)  # [m/s2]
# The CAT Vehicle tracks its command speed by two PIDs (Stern 2018 Sec. 3.3, pp. 7-8: to a 1 m/s step the accelerator
# loop rises in about 1.6 s and settles in about 5.52 s, overshooting 5 %; the brake loop, used when v_cmd - v <=
# -0.25 m/s, rises in about 0.8 s and settles in about 1.94 s, overshooting 11 %): here first-order lags a = (v_cmd -
# v) / tau with tau = rise / 2.2 (the 10-90 % rise of a first-order lag; *derived*). The rise time is the fastest
# reading of those loops: fitted to the settling times instead (4 tau for 2 %), tau would be about 1.4 s and 0.5 s
# (*derived*). PI with saturation, a lagged follower of its leader's speed at short gaps, is sensitive to that choice;
# FollowerStopper is not (*derived*, stern2018 runs at tau 1.4 s). They stand for the whole speed response, so a
# tracking smoother has no reaction delay.
SMOOTHER_TAU_ACCEL = 0.73  # [s]
SMOOTHER_TAU_BRAKE = 0.36  # [s]
SMOOTHER_BRAKE_BAND = 0.25  # [m/s]
# PI with saturation (Stern 2018 Sec. 3.2, Eqs. 3-5, pp. 6-7): U = the car's mean own speed over the last PI_WINDOW;
# v_target = U + PI_V_CATCH clip((dx - PI_GAP_LOW) / (PI_GAP_HIGH - PI_GAP_LOW), 0, 1); alpha = clip((dx - dx_s) /
# PI_GAMMA, 0, 1), dx_s = max(PI_SAFE_TIME v, PI_SAFE_GAP) with the car's own speed v (Stern 2018 Sec. 3.2 p. 7:
# "max(2 s . v, 4 m) ... the recommended following distance according to the 2 seconds rule"; Flow's PISaturation
# uses 2 dv there, which departs from the paper); beta = 1 - alpha / 2; v_cmd <- beta (alpha v_target + (1 - alpha)
# v_lead) + (1 - beta) v_cmd, once per step (*model*: the paper's update rate is not stated; beta >= 1/2 makes it
# nearly memoryless). Where the gap is under 2 s x v (alpha = 0) the command is the leader's speed (Stern p. 7: "when
# the gap is short, the CAT Vehicle has the same speed of the lead vehicle"), so on a dense ring it damps mainly by
# its speed tracking's lag (SMOOTHER_TAU_*).
PI_WINDOW = 38.0  # [s]
PI_GAP_LOW, PI_GAP_HIGH = 7.0, 30.0  # [m] g_l, g_u
PI_V_CATCH = 1.0  # [m/s]
PI_GAMMA = 2.0  # [m]
PI_SAFE_TIME, PI_SAFE_GAP = 2.0, 4.0  # [s], [m]
# Gap cap (Lichtle, Jang, Shah, Vinitsky, Lee & Bayen 2024, "Traffic smoothing controllers for autonomous vehicles
# using deep reinforcement learning and real-world trajectory data", IEEE ITSC 2023, arXiv 2401.09666, Sec. III-A.2:
# the gap-closing wrapper h_max = max(120, 6 v); its prose says "space gap larger than 120 m or time gap larger than
# 6 s", which the formula contradicts: the formula is used): beyond max(SMOOTHER_CAP_GAP, SMOOTHER_CAP_TIME x v) to
# the car ahead a smoother follows its driver's IDM instead, so it never holds back at its set speed behind a huge gap.
# *model*: Lichtle's wrapper commands 1.5 m/s2 there (while the time to collision exceeds 6 s); the IDM's free-road
# term also accelerates, without overshooting the desired speed.
SMOOTHER_CAP_GAP = 120.0  # [m]
SMOOTHER_CAP_TIME = 6.0  # [s]
# Cooperation classes, lowest first (SAE J3216: A status-sharing, B intent-sharing, C agreement-seeking, D
# prescriptive): coop_class within a company, coop_between caps it between companies ("same": as within). What
# each does here (*assumption*, the capability table): none, no V2X (CAVs follow by the ACC law); A, CACC behind
# a cooperating CAV (COOP_CACC); B-D add cooperation at the entries (not modelled yet: they act as A).
COOP_CLASSES = ("none", "A", "B", "C", "D")
COOP_CACC = 1  # the lowest class with CACC strings (A)
COOP_INTENT = 2  # the lowest class that shares intent: route-conflict anticipation at the entries (B)
# Surrogate safety measures (observers, RingSim._observe_following): time to collision TTC = gap / (v - v_lead)
# of a follower closing on its direct leader; an episode is TTC at or below TTC_CRITICAL (SSAM's default, FHWA
# Surrogate Safety Assessment Model, Gettman et al. 2008 FHWA-HRT-08-051) with a per-car hold of EMERGENCY_HOLD,
# and time exposed TET (Minderhoud & Bovy 2001, Accid. Anal. Prev. 33(1)) the vehicle-seconds at or below it.
TTC_CRITICAL = 1.5  # [s]
# Pair types of a follower and its direct leader, the FOLLOWER's letter first (as Hu et al. 2023 name pairs, AV-HV =
# an AV behind a human): H = a human driver or an ACC car (it looks like any car), A = a recognisable AV
# (RECOGNISABLE). Index = 2 x (follower is A) + (leader is A): pair_type.
PAIR_TYPES = ("HH", "HA", "AH", "AA")
SEEN_AS_AV = np.array([code in RECOGNISABLE for code in range(len(KIND_NAMES))])  # by kind code


def pair_type(follower_kind, leader_kind):
    """Index into PAIR_TYPES of a follower of kind ``follower_kind`` behind a leader of kind ``leader_kind`` (codes,
    numbers or aligned arrays; int64): 2 x (the follower is seen as an AV) + (the leader is), the FOLLOWER's letter
    first (FOLLOWER-LEADER, as Hu et al. 2023 name pairs). At an entry the lag vehicle is the follower and the entrant
    the leader; never the passing order some studies use (Rahmani)."""
    return 2 * SEEN_AS_AV[follower_kind].astype(np.int64) + SEEN_AS_AV[leader_kind]
# A consequence of the robotaxi's values, not tuned (*derived*): at the SAME density its longer T and s0 give a
# lower IDM equilibrium speed, where the plain IDM is linearly string-unstable even without a reaction delay
# (S = f_v^2/2 - f_dv f_v - f_s, Hu et al. 2023 Eqs. 7-10, >= 0 stable: single-lane ring of radius 60 m at
# density 60, -0.025 for robotaxis against +0.030 for drivers; stern2018 -0.123 against -0.006). So rings of
# robotaxis at a fixed density form stop-and-go waves (the delay only worsens them), the reverse of Hu's Table XI
# (Sec. IV-C: the AV-HV pairs are the least string-unstable), which compares each group at its own observed
# equilibrium, not at equal density. Read fixed-density ring results with robotaxis in that light.

# Per-car arrays, one value per slot (RingSim._alloc); the acceleration history acc_hist holds one row
# per step of history on top.
CAR_ARRAYS = ("alive", "lane", "theta", "travel", "v", "z", "exit_at", "born", "lat", "lc_cool", "noise",
              "brake_until", "uid", "moving", "emergency_until", "stuck", "entered_at", "inserted_at", "truck",
              "length", "z_react", "z_gap", "accel", "headway", "delay", "kind", "company", "acc_gap", "s0",
              "a_now", "string_pos", "aeb_until", "ttc_until", "v_cmd", "smoother_on", "speed_count")
# Saved states (RingSim.state_dict): the layout version, and what they hold besides the per-car arrays.
STATE_VERSION = 4  # 1: before the capacity test; 2: + it, conflict points, weather strength and anticipation;
# 3: + vehicle kinds and companies (their draws, settings, queue fields and per-car arrays) and conflict_log's kind;
# 4: + the ACC/CACC settings and string positions, cooperation, the safety layer and the surrogate safety counts,
# and the wave smoothers (settings, command speeds, start flags and the history of their own speeds, speed_hist)
ARM_ARRAYS = {"queue": int, "entry_cool": np.int64, "arm_arrived": np.int64, "arm_entered": np.int64,
              "queue_busy": float, "arm_passed": np.int64, "arm_passed_outer": np.int64,
              "arm_passed_trucks": np.int64, "arm_entered_trucks": np.int64,
              "arm_entry_wait": float}  # one value per entrance, and its dtype
# What each queued vehicle carries besides its arrival time (queue_times), in the same order: saved as one array
# per field (every entrance's queue in turn) and a count per entrance, read back as Python values of the type.
QUEUE_FIELDS = {"queue_trucks": bool, "queue_kinds": int, "queue_companies": int, "queue_acc_gaps": float}
# One value per pair type of a follower and its leader (PAIR_TYPES), and its dtype: the surrogate safety counts
# (RingSim._observe_following), in time steps.
PAIR_ARRAYS = {"pair_follow_steps": np.int64, "pair_ttc_steps": np.int64, "pair_ttc_episodes": np.int64}
COUNTERS = ("n_entered", "n_exited", "n_missed", "n_lane_changes", "n_emergency", "n_stops", "n_arrived",
            "n_blocked", "n_gave_up", "n_topped_up", "n_aeb", "n_emergency_auto", "n_auto_steps", "n_human_steps",
            "n_cutins_auto")  # whole numbers
FLOAT_TOTALS = ("entry_wait_total",)  # sums of floats (each kept in its own accumulation order)
EVENT_LOGS = ("detector_log", "exit_log", "lane_change_log", "emergency_log", "stop_log", "arrival_log",
              "entry_log", "blocked_log", "gave_up_log", "conflict_log")  # tuples starting with the time
# (lane_change_log: plain times)
GENERATORS = ("rng", "rng_demand", "rng_drivers", "rng_kinds", "rng_kinds_test")  # every random stream


class FromState:
    """An ADDED fill that is no constant: ``fill(sim)`` works it out from the restored simulation (its settings
    and the arrays restored or filled before it), for a name whose right value in an older state depends on
    that state (a per-car array that follows from the settings, say)."""

    def __init__(self, fill, why: str):
        self.fill, self.why = fill, why

    def __repr__(self) -> str:
        return f"FromState({self.why})"


# Everything saved that was added after version 1, with what a state of an older version gets for it (the
# setting's default, zero counts, an empty log): RingSim._from_state fills only these names, and only into a
# state of an older version (a state of this version that lacks one is damaged). Settings (SimConfig fields:
# the value is the default), ARM_ARRAYS (the value per entrance), PAIR_ARRAYS (the value per pair type),
# QUEUE_FIELDS (the value per queued vehicle), CAR_ARRAYS (the value per slot, or a FromState fill), speed_hist (the
# value per entry: an older state has no smoothers, so it holds no rows), COUNTERS and FLOAT_TOTALS, EVENT_LOGS (an
# empty log) and GENERATORS (k: the stream is default_rng([seed, k]); a tuple: default_rng([seed, *k]), a sub-stream).
# Each stage that changes the saved layout adds its names here and raises STATE_VERSION. (Fields added to the
# entries of a log: LOG_WIDENED.)
ADDED: dict[str, object] = {
    # version 2: the capacity test, conflict points, weather strength and anticipation
    "weather_scale": 1.0, "subject_arm": "off", "anticipation": ANTICIPATION,  # settings
    "arm_passed": 0, "arm_passed_outer": 0, "arm_passed_trucks": 0, "arm_entered_trucks": 0,  # per entrance
    "arm_entry_wait": 0.0,
    "n_topped_up": 0,  # counter
    "conflict_log": (),  # event log
    # version 3: vehicle kinds and companies (every vehicle of an older state is a human driver)
    "auto_share": 0.0, "auto_kind": "cav", "robotaxi_headway": 1.3, "companies": 1,  # settings
    "queue_kinds": HUMAN, "queue_companies": -1, "queue_acc_gaps": np.nan,  # per queued vehicle
    "kind": HUMAN, "company": -1, "acc_gap": np.nan, "a_now": 0.0,  # per car
    # (a driver's standstill gap is the setting's, whatever it was: worked out again, not a constant)
    "s0": FromState(lambda sim: sim._standstill_gaps(np.arange(sim.cap)), "min_gap of the state's settings"),
    "rng_kinds": 3,  # the kind draws' stream: default_rng([seed, 3])
    "rng_kinds_test": (3, 1),  # ...and its sub-stream for the capacity test's top-up: default_rng([seed, 3, 1])
    # version 4: the ACC/CACC controllers, cooperation, the safety layer and the surrogate safety counts
    "acc_headway": 1.1, "acc_delay": 0.5, "cav_headway": 0.7, "platoon_max": 8,  # settings
    "coop_class": "B", "coop_between": "same", "brand_spread": 0.0,
    "string_pos": 0, "aeb_until": -np.inf, "ttc_until": -np.inf,  # per car (no string, no episode under way)
    "n_aeb": 0, "n_emergency_auto": 0, "n_auto_steps": 0, "n_human_steps": 0, "n_cutins_auto": 0,  # counters
    "pair_follow_steps": 0, "pair_ttc_steps": 0, "pair_ttc_episodes": 0,  # per pair type
    # ...and the wave smoothers (an older state has none)
    "smoothers": 0, "smoother_law": "followerstopper", "smoother_speed": 7.5, "smoother_start": 0.0,  # settings
    "smoother_lane": -1,
    "v_cmd": 0.0, "smoother_on": False, "speed_count": 0,  # per car (no command, not tracking, no speeds held)
    "speed_hist": 0.0,  # (rows x slots: no rows without smoothers)
}
# Fields added to the entries of an event log since version 1: (the entries' width now, what an older state's
# shorter entries get on the end). A state of this version must hold entries of the full width.
LOG_WIDENED: dict[str, tuple[int, tuple]] = {
    "conflict_log": (5, (HUMAN,)),  # version 3: + the passing vehicle's kind (an older state's: human drivers)
}

# Heavy vehicles (trucks): multiples of a car driver's setting. Truck drivers keep a car's standstill gap
# and comfortable braking (the sources disagree: x1-2 and x0.67-1) and a car's entry gap and follow-up
# time (field data give x1.7-1.9; here a truck's length and slow start already make it need more room).
# max acceleration: IDM a 0.5 vs 1.0 m/s2 (Treiber & Kesting 2013, errata 2017), 0.7 vs 1.4 (EIDM, Kesting 2010)
TRUCK_ACCEL_FACTOR = 0.5  # and trucks measured at 0.59 m/s2 from a stop (Yang 2016)
TRUCK_SPEED_FACTOR = 0.7  # desired speed: 80 vs 120 km/h (Treiber & Kesting; MOBIL, Kesting 2007), 85 vs 120 (EIDM)
# time gap: 2.0 vs 1.5 s (EIDM) to 1.8 vs 1.0 s (Treiber & Kesting)
TRUCK_HEADWAY_FACTOR = 1.5  # and truck time gaps 1.7-2.0x a car's on roundabouts (Macioszek 2019)
TRUCK_WIDTH = 2.55  # [m] EU maximum (Directive 96/53/EC Annex I); US WB-62/67 trucks are 2.59 m

# Road surface: multiple of each setting the dynamics use ("max_decel" scales MAX_DECEL; the speed and time-gap
# factors scale each driver's own value, so the spreads between drivers too). A synthesis of
# weather studies, mostly on freeways and signalised streets (Hranac et al. 2006 FHWA-HOP-07-073; Rakha
# et al. 2009 FHWA-JPO-09-066 and 2010 FHWA-JPO-11-020; Martin/Perrin et al. 2000 MPC-01-120; Wallman &
# Astrom 2001 VTI 911A); ice is the least certain. The standstill gap (jam density) and reaction time are
# not scaled: the studies find no weather effect on jam density, and the reaction-time evidence is thin.
ROAD_SCALED = ("mean_speed", "max_accel", "comfort_decel", "time_headway", "entry_gap", "follow_up",
               "lateral_accel", "max_decel")
ROAD_CONDITIONS: dict[str, dict[str, float]] = {
    "dry": dict.fromkeys(ROAD_SCALED, 1.0),
    "wet": dict(
        mean_speed=0.92,  # free speed -6 to -9 % in rain (FHWA 2006), -10 % on arterials (Perrin 2000)
        max_accel=0.95,  # assumption: wet/dry peak adhesion 0.90 (Rakha 2010 Tab. 4.1), start-up barely slower (Perrin)
        comfort_decel=0.88,  # approach deceleration x0.876 in 1.6 cm/h rain (Rakha 2009 Eq. 5.2)
        time_headway=1.12,  # capacity -10 to -11 % in rain (FHWA 2006); SHRP 2 heavy-rain factor 0.89
        entry_gap=1.10,  # accepted gaps +10 % at four roundabouts in rain (Lee 2018); 1.104 (Rakha 2010 Tab. 3.5)
        follow_up=1.05,  # queue discharge headways +2 to +6 % in rain (Sadek 2004, Perrin 2000)
        lateral_accel=0.90,  # wet/dry friction 0.9 (Rakha 2010)
        max_decel=0.80,  # friction wet 0.7-0.8 vs dry 0.8-1.0 (VTI 2001): 7.2 m/s2
    ),
    "snow": dict(
        mean_speed=0.75,  # arterial speeds -25 to -40 % on snowy or slushy roads (FHWA RWM, Perrin 2000)
        max_accel=0.70,  # assumption: start-up lost time +23 % (Perrin 2000); traction on packed snow 1.2-1.8 m/s2
        comfort_decel=0.70,  # comfortable braking 0.27 g -> 0.20 g on snow (x0.74, via Perrin 2000); x0.56 (Sun 2026)
        time_headway=1.25,  # discharge headways +13 to +26 % in snow and slush (Perrin 2000); SHRP 2 heavy snow 0.80
        entry_gap=1.20,  # critical gap x1.19-1.20 with snow on a snowy surface (Rakha 2010 Tab. 3.5)
        follow_up=1.20,  # discharge headways +13 to +26 % in snow (Perrin 2000)
        lateral_accel=0.50,  # 2 m/s2: about the packed-snow friction limit, 0.2-0.3 g (VTI 2001); Kageyama 2013
        max_decel=0.30,  # packed-snow friction 0.20-0.30 (VTI 2001): 2.7 m/s2
    ),
    "ice": dict(
        mean_speed=0.70,  # free speed -28 % on an icy test track (Rakha 2010 Tab. 2.3)
        max_accel=0.50,  # assumption: traction limit mu 0.1 x 0.6 x g = 0.59 m/s2, half the default 1.2
        comfort_decel=0.45,  # assumption: below snow (Perrin 2000, qualitative) and the 1.35 m/s2 limit
        time_headway=1.60,  # assumption between snow (1.25) and x1.87 at capacity on an icy track (Rakha 2010)
        entry_gap=1.12,  # critical gaps x1.11-1.12 on an icy surface (Rakha 2010 Tab. 3.5)
        follow_up=1.30,  # assumption: above snow (no field data)
        lateral_accel=0.30,  # 1.2 m/s2: about the ice friction limit, 0.1-0.15 g (VTI 2001, Rakha 2010)
        max_decel=0.15,  # black ice 0.15-0.30, packed snow or ice 0.10-0.25 (VTI 2001, Rakha 2010): 1.35 m/s2
    ),
}

# The behavioural factors weather_scale acts on (people's choices); lateral_accel and max_decel are the tyres'
# grip and always apply in full.
WEATHER_SCALED = ("mean_speed", "max_accel", "comfort_decel", "time_headway", "entry_gap", "follow_up")
# *assumption*: no scaled factor falls below this. It first acts on ice's comfortable braking at weather_scale
# 1.64 (1 - 1.64 x 0.55 = 0.098), and on its acceleration at 1.8 (1 - 1.8 x 0.5 = 0.10); without it they would
# reach 0 at 1.82 and 2.0 (derived). The studies give no such extremes.
WEATHER_FLOOR = 0.1


def road_factors(condition: str, scale: float = 1.0) -> dict[str, float]:
    """The multiple of each ROAD_SCALED setting on road ``condition``, with the behavioural factors
    (WEATHER_SCALED) k turned into 1 + ``scale`` (k - 1), at least WEATHER_FLOOR: 0 = drivers act as on a dry
    road, 1 = ROAD_CONDITIONS as given (exactly: 1 + 1 (k - 1) is not k for every float), 2 = twice the change.
    A dry road is all 1 whatever the scale."""
    k = ROAD_CONDITIONS[condition]
    if scale == 1:
        return k
    return {name: max(WEATHER_FLOOR, 1.0 + scale * (f - 1.0)) if name in WEATHER_SCALED else f
            for name, f in k.items()}


@dataclass(frozen=True)
class Effective:
    """The settings the dynamics use: ``SimConfig``'s, scaled by the road condition (dry: unchanged).

    The factors act on each driver's own desired speed and time gap, so they
    scale the spreads between drivers too.
    """
    mean_speed: float  # [m/s]
    speed_std: float  # [m/s]
    max_accel: float  # [m/s2]
    comfort_decel: float  # [m/s2]
    time_headway: float  # [s]
    headway_std: float  # [s]
    entry_gap: float  # [s]
    follow_up: float  # [s]
    lateral_accel: float  # [m/s2]
    max_decel: float  # [m/s2] physical braking limit (MAX_DECEL on a dry road)


def effective(cfg: SimConfig) -> Effective:
    """The settings ``cfg`` drives with on its road condition (one multiplier each, ROAD_CONDITIONS, the
    behavioural ones scaled by the weather effect: road_factors)."""
    k = road_factors(cfg.road_condition, cfg.weather_scale)
    return Effective(**{name: getattr(cfg, name) * k[name] for name in ROAD_SCALED if name != "max_decel"},
                     speed_std=cfg.speed_std * k["mean_speed"], headway_std=cfg.headway_std * k["time_headway"],
                     max_decel=MAX_DECEL * k["max_decel"])


def braking_limits(conditions=("wet", "snow", "ice")) -> str:
    """The physical braking limit [m/s2] on each of the road ``conditions``, e.g. "7.2 / 2.7 / 1.35"."""
    return " / ".join(f"{MAX_DECEL * ROAD_CONDITIONS[c]['max_decel']:.3g}" for c in conditions)


def mean_demand_factor(profile: str, t0: float, t1: float) -> float:
    """Mean of the demand profile (a multiple of the entrance chance) from ``t0`` to ``t1`` s.

    1 for "constant". For "peak" it is 1 only over whole PEAK_PERIODs (or a
    window placed symmetrically about a peak or trough): a sweep measuring
    120-420 s, for example, catches the rise to the peak and averages 1.30.
    """
    if profile != "peak" or t1 <= t0:
        return 1.0
    w = TWO_PI / PEAK_PERIOD
    return float(1.0 - PEAK_SWING * (np.sin(w * t1) - np.sin(w * t0)) / (w * (t1 - t0)))


# ---------------------------------------------------------------- saved states
class StateError(ValueError):
    """A saved state that cannot be restored: incomplete, damaged or from another version (the
    message says what is wrong)."""


# Types the fields of an event log's entries can have, by the name a saved state records them under.
FIELD_TYPES = {"float": float, "int": int, "bool": bool, "float64": np.float64, "int64": np.int64,
               "bool_": np.bool_}
_FIELD_NAMES = {kind: name for name, kind in FIELD_TYPES.items()}


def _pack_log(log: deque) -> tuple[np.ndarray, list[str]]:
    """An event log as an array (a row per entry, a column per field; 1-D for plain times) and the
    type of each field (as in its first entry), so :func:`_unpack_log` gives back equal entries of
    the same Python types. Whole numbers are exact in the float array up to 2**53."""
    if not log:
        return np.zeros(0), []
    first = log[0]
    types = [_FIELD_NAMES.get(type(value), "float") for value in (first if isinstance(first, tuple) else (first,))]
    return np.array(list(log), dtype=float), types


def _unpack_log(name: str, values: np.ndarray, types: list[str], plain: bool) -> deque:
    """The event log :func:`_pack_log` packed: tuples of ``types``, or plain values if ``plain``."""
    if values.shape[:1] == (0,):
        return deque()
    width = 1 if plain else (values.shape[1] if values.ndim == 2 else -1)
    if values.ndim != (1 if plain else 2) or len(types) != width or not set(types) <= set(FIELD_TYPES):
        raise StateError(f"{name} is not a valid event log")
    columns = []
    for column, kind in zip([values] if plain else values.T, types):
        if kind not in ("float", "float64") and not (np.isfinite(column).all() and (column == np.round(column)).all()):
            raise StateError(f"{name} holds a fraction where a whole number belongs")
        typed = column.astype(FIELD_TYPES[kind])
        columns.append(typed.tolist() if FIELD_TYPES[kind] in (float, int, bool) else list(typed))
    return deque(columns[0] if plain else zip(*columns))


def _generator(state) -> np.random.Generator:
    """A random generator set to a saved bit-generator ``state`` (numpy's PCG64 family)."""
    name = state.get("bit_generator") if isinstance(state, dict) else None
    if name not in ("PCG64", "PCG64DXSM"):
        raise StateError(f"unsupported random generator {name!r}")
    generator = np.random.Generator(getattr(np.random, name)())
    generator.bit_generator.state = state
    return generator


def _array(arrays: dict, name: str, shape: tuple | None, dtype) -> np.ndarray:
    """A copy of saved array ``name`` as ``dtype``, checking its shape (None: any) and kind of number."""
    if name not in arrays:
        raise StateError(f"the state lacks the array {name}")
    arr = np.asarray(arrays[name])
    if shape is not None and arr.shape != tuple(shape):
        raise StateError(f"{name} has shape {arr.shape}, expected {tuple(shape)}")
    if not np.can_cast(arr.dtype, dtype, casting="same_kind"):
        raise StateError(f"{name} holds {arr.dtype}, expected {np.dtype(dtype)}")
    return arr.astype(dtype, copy=True)


def _split(values: np.ndarray, counts: np.ndarray, name: str) -> list[np.ndarray]:
    """``values`` cut into consecutive parts of ``counts`` elements (one per entrance or lane)."""
    if values.ndim != 1 or (counts < 0).any() or counts.sum() != values.size:
        raise StateError(f"{name} does not match its counts")
    bounds = np.concatenate(([0], np.cumsum(counts)))
    return [values[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


def _chain_depth(joined: np.ndarray, lead: np.ndarray, key: np.ndarray | None = None, *, heads: bool = False):
    """Pointer jumping (list ranking) along chains of elements: element i is joined to ``lead[i]`` where
    ``joined[i]``, else it heads its chain. Returns (depth: the links from each element to its head, 0 for a
    head; loop: the elements whose chain never reaches a head, a closed loop of joined elements; low: the
    smallest ``key`` over each element's loop, None without ``key``), with ``heads`` a fourth: each element's
    head (itself for a head; meaningless in a loop). Each pass doubles the reach, so chains of any length
    resolve within ceil(log2 n) passes; one more pass, and a loop's ``low`` covers all of it."""
    n = joined.size
    up = np.where(joined, lead, np.arange(n))
    depth = joined.astype(np.int64)
    low = None if key is None else key.copy()
    for _ in range(n.bit_length() + 1):
        if not joined[up].any():
            break
        if low is not None:
            low = np.minimum(low, low[up])
        depth = depth + depth[up]
        up = up[up]
    return (depth, joined[up], low, up) if heads else (depth, joined[up], low)


class RingSim:
    """Agent-based traffic on a multi-lane roundabout.

    Per-car state lives in fixed-capacity numpy arrays indexed by slot; only
    slots with ``alive[i]`` are cars on the ring. Use :meth:`cars` for a
    snapshot of the live cars instead of reading the arrays directly.
    """

    def __init__(self, config: SimConfig | None = None, **overrides):
        self.cfg = (config or SimConfig()).updated(**overrides)
        self.reset()

    # ------------------------------------------------------------- set-up
    def reset(self, seed: int | None = None, **changes) -> None:
        """Rebuild the roundabout and place the initial cars."""
        if changes:
            self.cfg = self.cfg.updated(**changes)
        cfg = self.cfg
        if seed is None:
            seed = cfg.seed if cfg.seed is not None else int(np.random.SeedSequence().entropy % 2**31)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        # Open-mode arrivals draw from a stream of their own, so the demand settings never change a
        # closed-mode run and the same seed gives the same arrivals. What the arrivals cause (entries,
        # new drivers) still draws from self.rng, so different demand shifts the rest of an open run.
        self.rng_demand = np.random.default_rng([self.seed, 1])
        # Vehicle types and drivers' own reaction times and time gaps: a stream of their own too, so
        # with every vehicle a car and every driver alike the run is the same as without them.
        self.rng_drivers = np.random.default_rng([self.seed, 2])
        # Each new vehicle's kind, company and ACC-fallback gap (_draw_kinds): drawn only when auto_share > 0, so
        # without automated vehicles nothing changes, and with them every other stream draws what it would.
        self.rng_kinds = np.random.default_rng([self.seed, 3])
        # ...but the capacity test's top-up from a sub-stream of its own: how many it adds follows the subject
        # entries (the dynamics), so on rng_kinds it would shift every later arrival's draws once two runs
        # differ, and the conflicting stream's kinds would stop nesting across shares (_top_up).
        self.rng_kinds_test = np.random.default_rng([self.seed, 3, 1])
        self._update_effective()
        self.time = 0.0
        self.steps = 0
        self.generation = getattr(self, "generation", -1) + 1  # lets observers notice rebuilds

        self._geometry()
        n = self.n_arms
        self.queue = np.zeros(n, dtype=int)  # cars waiting at each entrance
        # Arrival time of every queued car, oldest first: len(queue_times[arm]) == queue[arm] always.
        self.queue_times: list[deque[float]] = [deque() for _ in range(n)]
        # Whether each queued vehicle is a truck (drawn when it joins), in the same order; and its kind, company
        # and ACC-fallback gap (_draw_kinds, also drawn when it joins: QUEUE_FIELDS).
        self.queue_trucks: list[deque[bool]] = [deque() for _ in range(n)]
        self.queue_kinds: list[deque[int]] = [deque() for _ in range(n)]
        self.queue_companies: list[deque[int]] = [deque() for _ in range(n)]
        self.queue_acc_gaps: list[deque[float]] = [deque() for _ in range(n)]
        self.entry_cool = np.zeros(n, dtype=np.int64)  # steps until each entrance may let its next car in
        self.demand_weight = self._demand_weights()

        self.n_entered = self.n_exited = self.n_missed = 0
        self.n_lane_changes = self.n_emergency = self.n_stops = 0
        # Every arrival (and car added to the capacity test's queue) ends up entered, still queued, blocked
        # or given up: n_arrived + n_topped_up == n_entered + queue.sum() + n_blocked + n_gave_up.
        self.n_arrived = self.n_blocked = 0
        self.n_gave_up = 0  # queued cars dropped when Max queue (open) or Density (closed) was lowered live
        self.n_topped_up = 0  # cars added to the subject entrance's queue (subject_arm on): not arrivals
        # Observers of the automated vehicles: safety-layer episodes (_automated_limits), harsh-braking episodes of
        # automated vehicles (n_emergency counts every vehicle's), vehicle-steps on the ring of automated vehicles
        # and of human drivers (exposure) and lane changes that cut in ahead of a recognisable AV.
        self.n_aeb = self.n_emergency_auto = self.n_auto_steps = self.n_human_steps = self.n_cutins_auto = 0
        # Per pair type (PAIR_TYPES): steps of car following, steps at TTC <= TTC_CRITICAL, TTC episodes.
        for name, dtype in PAIR_ARRAYS.items():
            setattr(self, name, np.zeros(len(PAIR_TYPES), dtype=dtype))
        self.entry_wait_total = 0.0  # [s] queue wait summed over every car that entered
        # Per entrance, for its demand / capacity ratio (metrics.entry_vc).
        self.arm_arrived = np.zeros(n, dtype=np.int64)  # arrivals (blocked ones included)
        self.arm_entered = np.zeros(n, dtype=np.int64)  # entries
        self.queue_busy = np.zeros(n)  # [s] time it has had cars waiting to enter
        # Its conflicting flow (observer only, _log_conflicts): circulating vehicles that passed its entry point,
        # on any lane, on the outer lane (the one entering cars merge into) and trucks; its truck entries (pc/h)
        # and the queue wait of its entrants [s] (apart from entry_wait_total, which keeps its own sum order).
        self.arm_passed = np.zeros(n, dtype=np.int64)
        self.arm_passed_outer = np.zeros(n, dtype=np.int64)
        self.arm_passed_trucks = np.zeros(n, dtype=np.int64)
        self.arm_entered_trucks = np.zeros(n, dtype=np.int64)
        self.arm_entry_wait = np.zeros(n)
        # Event logs for rolling rates, oldest first, pruned after LOG_HORIZON.
        self.detector_log: deque[tuple[float, int]] = deque()  # (time, lane) crossings
        self.exit_log: deque[tuple[float, int, float]] = deque()  # (time, arm, trip time)
        self.lane_change_log: deque[float] = deque()  # time
        self.emergency_log: deque[tuple[float, int]] = deque()  # (time, new emergencies)
        self.stop_log: deque[tuple[float, int]] = deque()  # (time, cars that stopped)
        self.arrival_log: deque[tuple[float, int]] = deque()  # (time, arm) of each arrival
        self.entry_log: deque[tuple[float, int, float]] = deque()  # (time, arm, queue wait)
        self.blocked_log: deque[tuple[float, int]] = deque()  # (time, arm) arrivals turned away
        self.gave_up_log: deque[tuple[float, int]] = deque()  # (time, arm) queued cars that gave up
        # (time, arm, lane, uid, kind) of each vehicle passing an entry point (its conflict point)
        self.conflict_log: deque[tuple[float, int, int, int, int]] = deque()
        # Once a second: (time, arm_arrived, arm_entered, queue_busy) copies, for rolling ratios.
        self.entrance_log: deque[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = deque(
            [(0.0, self.arm_arrived.copy(), self.arm_entered.copy(), self.queue_busy.copy())])
        self._next_uid = 0
        self._next_fill = 0.0
        # Whether each vehicle the density control drew but could not place yet is a truck (oldest
        # first): the next fill places them first, so a truck that fits no gap waits for one.
        self._pending: list[bool] = []
        self._next_prune = 0.0
        self._lanes: list[tuple[np.ndarray, np.ndarray]] = [(np.zeros(0, dtype=np.int64), np.zeros(0))] * self.n_lanes

        self._hist_len = int(round(MAX_REACTION / cfg.dt)) + 1
        self._hist_ptr = 0
        self._alloc(max(64, 2 * self.capacity()))
        self._place_initial(self.target_cars())
        if self.open_demand and int(self.alive.sum()) < self.target_cars():
            # Vehicles that fitted no lane of their own (a heavy truck mix, or a start at jam
            # density): closed mode tops the ring up on its first step, open mode places them
            # now, as a live Density raise does, so Density sets the cars at the start.
            self._sync_ring()
        if self.smoothing:  # the wave smoothers: picked from the drivers just placed (no draws)
            self._sync_smoothers()

    def _geometry(self) -> None:
        """Lanes, arms and the detector line from the settings (nothing random)."""
        cfg = self.cfg
        self.n_lanes = int(cfg.n_lanes)
        self.outer = self.n_lanes - 1
        self.radii = cfg.radius + cfg.lane_width * np.arange(self.n_lanes)
        self.lane_len = TWO_PI * self.radii
        self.r_out = float(self.radii[-1])
        n = int(cfg.n_arms)
        self.n_arms = n
        if n:
            self.arm_angle = (-np.pi / 2 + TWO_PI * np.arange(n) / n) % TWO_PI
            half = min(0.35 * TWO_PI / n, 1.2 * cfg.lane_width / self.r_out)
            self.exit_angle = (self.arm_angle - half) % TWO_PI  # exit comes first...
            self.entry_angle = (self.arm_angle + half) % TWO_PI  # ...then the entry
            self.detector_angle = float((-np.pi / 2 + np.pi / n) % TWO_PI)  # between arms
            # The capacity test's exit (_assign_routes): the first after the subject entrance's entry point.
            # With 2+ lanes, the first at least the innermost lane's lane-change zone beyond it (must_move_out),
            # so leaving does not herd every car onto the outer lane before that point: its lane use stays the
            # drivers' own (*model*; else all of its conflicting flow would be on the outer lane). None so far
            # within a lap: the last one before the point comes round again (the subject's own exit).
            arcs = (self.exit_angle - self.entry_angle[SUBJECT_ARM]) % TWO_PI
            far = arcs * self.r_out >= ((self.n_lanes - 1) * LC_ZONE + 20.0 if self.n_lanes > 1 else 0.0)
            self._subject_exit = int(np.argmin(np.where(far, arcs, np.inf))) if far.any() else int(np.argmax(arcs))
        else:
            self.arm_angle = self.exit_angle = self.entry_angle = np.zeros(0)
            self.detector_angle = 0.0
            self._subject_exit = 0
        # The entry points in angle order, then once more a lap on (_log_conflicts: the arc a car drives in a
        # step starts in [0, 2pi) and ends before 4pi), and the entrance each belongs to.
        self._entry_order = np.argsort(self.entry_angle, kind="stable")
        ordered = self.entry_angle[self._entry_order]
        self._entry_ext = np.concatenate((ordered, ordered + TWO_PI))

    def set(self, **changes) -> bool:
        """Change parameters live. Returns True if the change rebuilt the roundabout."""
        if not changes:
            return False
        old, old_seed = self.cfg, self.seed
        new_cfg = old.updated(**changes)  # validates names and values
        # Truck length matters only once there are trucks, and the capacity test only in open mode with an
        # entrance: changed while it cannot act, before and after, it is just kept for later (nothing on the
        # ring depends on it).
        idle = {"truck_length"} if old.truck_share == 0 and new_cfg.truck_share == 0 else set()
        if not (self._can_test(old) or self._can_test(new_cfg)):
            idle.add("subject_arm")
        if old.auto_share == 0 and new_cfg.auto_share == 0:  # (no vehicle is automated, before or after)
            idle |= {"auto_kind", "companies"}
        # Geometry/numerics fields without a slider (lane_width, dt, ...) also rebuild.
        rebuild = any((PARAM_BY_NAME[k].reset if k in PARAM_BY_NAME else k != "seed")
                      and getattr(old, k) != v for k, v in changes.items() if k not in idle)
        self.cfg = new_cfg
        if rebuild:
            try:
                self.reset(seed=new_cfg.seed if "seed" in changes else self.seed)
            except Exception:
                self.cfg = old
                self.reset(seed=old_seed)
                raise
            return True
        if "seed" in changes and new_cfg.seed is not None:  # applies from now on
            self.seed = int(new_cfg.seed)
            self.rng = np.random.default_rng(self.seed)
            self.rng_demand = np.random.default_rng([self.seed, 1])
            self.rng_drivers = np.random.default_rng([self.seed, 2])
            self.rng_kinds = np.random.default_rng([self.seed, 3])
            self.rng_kinds_test = np.random.default_rng([self.seed, 3, 1])
        alive = np.flatnonzero(self.alive)
        # Every driver keeps their temperaments; their own time gap, acceleration, standstill gap and
        # reaction delay follow the new settings and road condition at once (a robotaxi's Robotaxi time gap too).
        self._update_effective()
        self._set_driving(alive)
        if new_cfg.exit_prob != old.exit_prob and not self.subject:  # (the capacity test's routes ignore it)
            self._assign_routes(alive)  # geometric routes are memoryless (a smoother's stays: it never leaves)
        if self.speed_hist.shape[0] != self._speed_rows():  # smoothers on or off: their speed history (_record_speeds)
            self.speed_hist = np.zeros((self._speed_rows(), self.cap))
            self.speed_count[:] = 0
        if new_cfg.smoothers != old.smoothers or new_cfg.smoother_lane != old.smoother_lane:
            self._sync_smoothers()  # converts or hands back the nearest (after the driving values and routes)
        if new_cfg.smoother_law != old.smoother_law:  # a new law engages from each car's speed (_law_smoother)
            self.smoother_on[:] = False
        if any(getattr(new_cfg, k) != getattr(old, k) for k in
               ("mean_speed", "speed_std", "lateral_accel", "turn_speed", "exit_prob", "comfort_decel",
                "road_condition", "weather_scale")):
            # New free speeds: re-baseline who counts as moving, so the change
            # itself is not logged as a wave of stops.
            self.moving[alive] &= self.v[alive] >= SLOW_FRACTION * self.target_speed(alive)
        if new_cfg.demand_imbalance != old.demand_imbalance:
            self.demand_weight = self._demand_weights()
        if self.open_demand:
            if new_cfg.max_queue < old.max_queue:  # shorter queues: their latest arrivals give up
                for arm in range(self.n_arms):
                    if not (self.subject and arm == SUBJECT_ARM):  # (the capacity test's queue stays full)
                        self._drop_queued(arm, max(int(self.queue[arm]) - new_cfg.max_queue, 0))
            if new_cfg.density != old.density:
                self._sync_ring()  # Density sets the cars on the ring once, at this moment
        return False

    def _alloc(self, cap: int) -> None:
        self.cap = cap
        self.alive = np.zeros(cap, dtype=bool)
        self.lane = np.zeros(cap, dtype=np.int64)
        self.theta = np.zeros(cap)  # angle on the ring in [0, 2pi)
        self.travel = np.zeros(cap)  # unwrapped angle driven since the car appeared
        self.v = np.zeros(cap)
        self.z = np.zeros(cap)  # driver temperament ~ N(0, 1): desired speed = mean + z * std
        self.exit_at = np.full(cap, np.inf)  # value of ``travel`` at which the car's exit is
        self.born = np.zeros(cap)
        self.lat = np.zeros(cap)  # drawn lateral position in lanes (animates lane changes)
        self.lc_cool = np.zeros(cap)
        self.noise = np.zeros(cap)
        self.brake_until = np.full(cap, -np.inf)
        self.uid = np.full(cap, -1, dtype=np.int64)
        self.moving = np.zeros(cap, dtype=bool)  # above RESUME_FRACTION since its last stop
        self.emergency_until = np.full(cap, -np.inf)  # end of the current emergency episode
        self.stuck = np.zeros(cap)  # [s] a leaver has been held at a standstill by merging
        self.entered_at = np.full(cap, -np.inf)  # time the car came in from an entrance
        self.inserted_at = np.full(cap, -np.inf)  # time it was slotted in by the density control
        self.acc_hist = np.zeros((self._hist_len, cap))
        # Vehicle and driver: drawn at spawn from rng_drivers.
        self.truck = np.zeros(cap, dtype=bool)
        self.length = np.zeros(cap)  # [m] car_length or truck_length
        self.z_react = np.zeros(cap)  # temperaments ~ N(0, 1): own reaction time = setting + z * spread
        self.z_gap = np.zeros(cap)  # ...and own time gap
        # Kind (HUMAN ... SMOOTHER), company (0 to companies - 1; -1 for human drivers and smoothers) and the
        # CAV's ACC-fallback time gap [s] (ACC_GAPS; NaN when nothing was drawn): drawn at spawn from rng_kinds.
        self.kind = np.zeros(cap, dtype=np.int8)
        self.company = np.full(cap, -1, dtype=np.int8)
        self.acc_gap = np.full(cap, np.nan)
        # Derived from those and the effective settings (_set_driving), so the step need not.
        self.accel = np.zeros(cap)  # [m/s2] max acceleration
        self.headway = np.zeros(cap)  # [s] IDM time gap
        self.delay = np.zeros(cap, dtype=np.int64)  # [steps] reaction delay
        self.s0 = np.zeros(cap)  # [m] standstill gap
        # [m/s2] Acceleration the car realised in the last step ((v_new - v) / dt: after the braking limit, a
        # stop and the no-overlap constraint; *model*): what a CACC law's gap-error derivative and a smoother's
        # speed tracking take as the car's own acceleration.
        self.a_now = np.zeros(cap)
        # A CAV's place in its CACC string, worked out at the start of every step (_controller_gaps): 0 = no CAV, 1 =
        # string leader following by the ACC law, 2 to platoon_max = member at the CACC time gap, platoon_max + 1 =
        # the member that would be one too many, following by the CACC law at CAV_PLATOON_GAP and leading the next.
        self.string_pos = np.zeros(cap, dtype=np.int8)
        # End of the car's current safety-layer and low-TTC episodes (observers; like emergency_until).
        self.aeb_until = np.full(cap, -np.inf)
        self.ttc_until = np.full(cap, -np.inf)
        # A smoother's command speed [m/s] (_law_smoother: PI with saturation updates it from its last value) and
        # whether its controller tracks it (from Smoother start on, under FollowerStopper or PI; it starts from the
        # car's speed). Its own speeds of the last PI_WINDOW s, one row per step taken in turn (speed_hist, rows
        # only while there are smoothers: _speed_rows), and how many of those rows hold its own (speed_count).
        self.v_cmd = np.zeros(cap)
        self.smoother_on = np.zeros(cap, dtype=bool)
        self.speed_count = np.zeros(cap, dtype=np.int64)
        self.speed_hist = np.zeros((self._speed_rows(), cap))

    def _speed_rows(self) -> int:
        """Rows of the smoothers' speed history: PI_WINDOW in whole steps while there are smoothers, else none."""
        return int(round(PI_WINDOW / self.cfg.dt)) if self.cfg.smoothers > 0 else 0

    def _grow(self, need: int) -> None:
        old = {name: getattr(self, name) for name in CAR_ARRAYS}
        hist, speeds, old_cap = self.acc_hist, self.speed_hist, self.cap
        self._alloc(max(2 * old_cap, old_cap + need))
        for name, arr in old.items():
            getattr(self, name)[:old_cap] = arr
        self.acc_hist[:, :old_cap] = hist
        self.speed_hist[:, :old_cap] = speeds

    # -------------------------------------------------------- saved state
    def state_dict(self) -> dict:
        """Everything an exact resume needs: ``{"meta": a JSON-serialisable dict, "arrays": {name: array}}``.

        :meth:`from_state` turns it back into a simulation that carries on bit
        for bit as this one would. It holds the settings, seed, clock and
        generation; every per-car array (CAR_ARRAYS), the acceleration history
        and its pointer, the smoothers' speed history; the queues with each queued car's arrival time,
        vehicle type, kind, company and ACC-fallback gap (QUEUE_FIELDS); the
        entrance cooldowns, counters, event logs and
        once-a-second entrance totals; the uid counter, the fill and prune
        timers and the vehicles the density control still has to place; the
        lane index cache; and the bit-generator state of every random stream
        (GENERATORS). What follows from the settings alone (the geometry,
        ``eff``, the demand weights) is worked out again. Arrays are copies.
        The layout is STATE_VERSION's; states of older versions still load
        (ADDED: what was added since takes its fill).
        """
        arrays = {name: getattr(self, name).copy()
                  for name in (*CAR_ARRAYS, "acc_hist", "speed_hist", *ARM_ARRAYS, *PAIR_ARRAYS)}
        logs = {}
        for name in EVENT_LOGS:
            arrays[name], logs[name] = _pack_log(getattr(self, name))
        # Each queue's arrival times (and what its vehicles carry), all entrances in one array, and how many each has.
        for name, kind in (("queue_times", float), *QUEUE_FIELDS.items()):
            queues = getattr(self, name)
            arrays[name] = np.array([value for fifo in queues for value in fifo], dtype=kind)
            arrays[f"{name}_count"] = np.array([len(fifo) for fifo in queues], dtype=np.int64)
        k = len(self.entrance_log)
        arrays["entrance_log_time"] = np.array([entry[0] for entry in self.entrance_log], dtype=float)
        for i, (part, dtype) in enumerate((("arrived", np.int64), ("entered", np.int64), ("busy", float)), start=1):
            arrays[f"entrance_log_{part}"] = np.array([entry[i] for entry in self.entrance_log],
                                                      dtype=dtype).reshape(k, self.n_arms)
        arrays["lanes_order"] = np.concatenate([order for order, _ in self._lanes]).astype(np.int64)
        arrays["lanes_theta"] = np.concatenate([th for _, th in self._lanes]).astype(float)
        arrays["lanes_count"] = np.array([order.size for order, _ in self._lanes], dtype=np.int64)
        meta = {
            "version": STATE_VERSION,
            "config": {name: value.item() if isinstance(value, np.generic) else value
                       for name, value in asdict(self.cfg).items()},
            "seed": self.seed, "time": self.time, "steps": self.steps, "generation": self.generation,
            "cap": self.cap, "hist_ptr": self._hist_ptr, "next_uid": self._next_uid,
            "next_fill": self._next_fill, "next_prune": self._next_prune,
            "pending": [bool(truck) for truck in self._pending],
            "counters": {name: int(getattr(self, name)) for name in COUNTERS},
            **{name: float(getattr(self, name)) for name in FLOAT_TOTALS},
            "logs": logs,
            "generators": {name: getattr(self, name).bit_generator.state for name in GENERATORS},
        }
        return {"meta": meta, "arrays": arrays}

    @classmethod
    def from_state(cls, state: dict) -> RingSim:
        """The simulation a :meth:`state_dict` was taken from, carrying on bit for bit as it would have.

        Raises :class:`StateError` saying what is wrong if ``state`` is
        incomplete, damaged or from another version (e.g. settings this
        version does not know, or arrays of the wrong shape). A state of an
        older version (1 up to STATE_VERSION) loads with what was added since
        filled in (ADDED: default settings, zero counts, empty logs), so it
        carries on as it would have, its new measurements counted from then on.
        """
        try:
            return cls._from_state(state["meta"], state["arrays"])
        except StateError:
            raise
        except KeyError as e:
            raise StateError(f"the state lacks {e.args[0]!r}") from e
        except (TypeError, ValueError, AttributeError, IndexError, OverflowError) as e:
            raise StateError(f"the state does not fit together ({e})") from e

    @classmethod
    def _from_state(cls, meta: dict, arrays: dict) -> RingSim:
        version = meta.get("version")
        if isinstance(version, bool) or not isinstance(version, int) or not 1 <= version <= STATE_VERSION:
            raise StateError(f"simulation state version {version!r}; this version reads 1 to {STATE_VERSION}")

        def fill(name: str, present) -> bool:
            """Whether ``name``, missing from ``present``, takes its ADDED fill (in an older state only)."""
            return version < STATE_VERSION and name in ADDED and name not in present

        config, names = dict(meta["config"]), {f.name for f in fields(SimConfig)}
        config.update({name: ADDED[name] for name in sorted(names) if fill(name, config)})
        unknown, missing = sorted(set(config) - names), sorted(names - set(config))
        if unknown or missing:
            raise StateError("settings of another version (" + "; ".join(
                [f"unknown: {', '.join(unknown)}"] * bool(unknown) + [f"missing: {', '.join(missing)}"] * bool(missing))
                + ")")
        self = cls.__new__(cls)
        self.cfg = cfg = SimConfig(**config)  # validates the values
        self.seed = int(meta["seed"])
        generators = meta["generators"]
        for name in GENERATORS:
            key = ADDED.get(name)
            setattr(self, name, np.random.default_rng([self.seed, *(key if isinstance(key, tuple) else (key,))])
                    if fill(name, generators) else _generator(generators[name]))
        self._update_effective()
        self.time, self.steps, self.generation = float(meta["time"]), int(meta["steps"]), int(meta["generation"])
        if not (np.isfinite(self.time) and self.time >= 0 and self.steps >= 0 and self.generation >= 0):
            raise StateError("the clock (time, steps or generation) is out of range")
        self._geometry()
        n = self.n_arms
        for name, dtype in ARM_ARRAYS.items():
            setattr(self, name, np.full(n, ADDED[name], dtype=dtype) if fill(name, arrays)
                    else _array(arrays, name, (n,), dtype))
        for name, dtype in PAIR_ARRAYS.items():
            setattr(self, name, np.full(len(PAIR_TYPES), ADDED[name], dtype=dtype) if fill(name, arrays)
                    else _array(arrays, name, (len(PAIR_TYPES),), dtype))
        self.demand_weight = self._demand_weights()
        for name, dtype in (("queue_times", float), *QUEUE_FIELDS.items()):
            if fill(name, arrays):  # (an older state's queued vehicles: one fill each, whatever count it holds)
                setattr(self, name, [deque([ADDED[name]] * max(int(q), 0)) for q in self.queue])
                continue
            parts = _split(_array(arrays, name, None, dtype), _array(arrays, f"{name}_count", (n,), np.int64), name)
            setattr(self, name, [deque(part.tolist()) for part in parts])
        if (self.queue < 0).any() or any(len(fifo) != q for fifo, q in zip(self.queue_times, self.queue)) \
                or any(len(fifo) > q for name in QUEUE_FIELDS for fifo, q in zip(getattr(self, name), self.queue)):
            raise StateError("the queues do not match their arrival times and vehicle types")
        counters = meta["counters"]
        for name in COUNTERS:
            setattr(self, name, int(ADDED[name] if fill(name, counters) else counters[name]))
        for name in FLOAT_TOTALS:
            setattr(self, name, float(ADDED[name] if fill(name, meta) else meta[name]))
        for name in EVENT_LOGS:
            if fill(name, arrays) and fill(name, meta["logs"]):
                setattr(self, name, deque(ADDED[name]))
            else:
                setattr(self, name, _unpack_log(name, _array(arrays, name, None, float), list(meta["logs"][name]),
                                                plain=name == "lane_change_log"))
        for name, (width, extra) in LOG_WIDENED.items():  # entries given fields since: an older state's get the fills
            log = getattr(self, name)
            short = len(log[0]) if log else width
            if short != width:
                if version == STATE_VERSION or not width - len(extra) <= short < width:
                    raise StateError(f"{name} holds entries of {short} fields, expected {width}")
                setattr(self, name, deque(entry + extra[short - width:] for entry in log))
        when = _array(arrays, "entrance_log_time", None, float)
        k = when.shape[0] if when.ndim == 1 else -1
        parts = [_array(arrays, f"entrance_log_{part}", (k, n), dtype)
                 for part, dtype in (("arrived", np.int64), ("entered", np.int64), ("busy", float))]
        self.entrance_log = deque((t, *(row.copy() for row in rows)) for t, *rows in zip(when.tolist(), *parts))
        self._next_uid, self._next_fill = int(meta["next_uid"]), float(meta["next_fill"])
        self._pending = list(meta["pending"])
        if not all(isinstance(truck, bool) for truck in self._pending):
            raise StateError("pending holds something other than true/false")
        self._next_prune = float(meta["next_prune"])
        if not (np.isfinite(self._next_fill) and np.isfinite(self._next_prune)):
            raise StateError("the fill or prune timer is not a finite time")
        self._hist_len = int(round(MAX_REACTION / cfg.dt)) + 1
        self._hist_ptr = int(meta["hist_ptr"])
        cap = int(meta["cap"])
        if cap < 1 or np.shape(arrays.get("alive")) != (cap,) or not 0 <= self._hist_ptr < self._hist_len:
            raise StateError("the car arrays' size or history pointer is out of range")  # (before allocating)
        self._alloc(cap)
        derived = []  # FromState fills, worked out once every other array is in place
        for name in CAR_ARRAYS:
            if fill(name, arrays):
                if isinstance(ADDED[name], FromState):
                    derived.append(name)
                else:
                    getattr(self, name)[:] = ADDED[name]
            else:
                getattr(self, name)[:] = _array(arrays, name, (cap,), getattr(self, name).dtype)
        for name in derived:
            getattr(self, name)[:] = ADDED[name].fill(self)
        if version < 4:  # acc and cav cars saved before their controllers drove a driver's IDM: theirs from now on
            stale = np.flatnonzero(self.alive & ((self.kind == ACC) | (self.kind == CAV)))
            if stale.size:
                self._set_driving(stale)
        self.acc_hist[:] = _array(arrays, "acc_hist", (self._hist_len, cap), float)
        self.speed_hist[:] = ADDED["speed_hist"] if fill("speed_hist", arrays) else \
            _array(arrays, "speed_hist", self.speed_hist.shape, float)  # (its rows follow from the settings)
        if (self.speed_count < 0).any() or (self.speed_count > self.speed_hist.shape[0]).any():
            raise StateError("speed_count holds more speeds than the smoothers' history has rows")
        counts = _array(arrays, "lanes_count", (self.n_lanes,), np.int64)
        order = _array(arrays, "lanes_order", None, np.int64)
        if order.ndim != 1 or ((order < 0) | (order >= cap)).any():
            raise StateError("the lane index holds slots that do not exist")
        self._lanes = list(zip(_split(order, counts, "lanes_order"),
                               _split(_array(arrays, "lanes_theta", order.shape, float), counts, "lanes_theta")))
        lanes = self.lane[self.alive]
        if ((lanes < 0) | (lanes >= self.n_lanes)).any():
            raise StateError("a car is on a lane the ring does not have")
        return self

    # ----------------------------------------------------------- geometry
    @property
    def total_lane_length(self) -> float:
        return float(self.lane_len.sum())

    @property
    def mean_length(self) -> float:
        """[m] Expected vehicle length for the truck share (car_length when there are no trucks)."""
        cfg = self.cfg
        return cfg.car_length + cfg.truck_share * (cfg.truck_length - cfg.car_length)

    @property
    def jam_density(self) -> float:
        """Densest packing [veh/km/lane]: bumper to bumper at the standstill gap (vehicles of the mean length)."""
        return 1000.0 / (self.mean_length + self.cfg.min_gap)

    def capacity(self) -> int:
        """Most vehicles the lanes hold at jam density (of the mean length: a random mix with more
        trucks than average can need a few places more)."""
        pitch = self.mean_length + self.cfg.min_gap
        return int(np.floor(self.lane_len / pitch).sum())

    def target_cars(self) -> int:
        density = min(self.cfg.density, self.jam_density)
        return min(int(round(density * self.total_lane_length / 1000.0)), self.capacity())

    def lane_radius(self, lat) -> np.ndarray:
        return self.cfg.radius + self.cfg.lane_width * np.asarray(lat)

    # ------------------------------------------------------------- demand
    @property
    def open_demand(self) -> bool:
        """True in open mode: cars arrive at random and leavers are not replaced."""
        return self.cfg.traffic_mode == "open"

    @property
    def queue_limit(self) -> int:
        """Most cars that can wait at one entrance in the current mode."""
        return int(self.cfg.max_queue) if self.open_demand else MAX_QUEUE

    def _demand_weights(self) -> np.ndarray:
        """Each entrance's share of the demand, with mean 1 over the entrances.

        A main road runs along the axis through arm 0 (the bottom one) and, with
        an even number of arms, the opposite arm n/2: an arm's weight is
        1 + demand_imbalance * cos(2 * its angle from arm 0), so arms on the axis
        get the most and arms at right angles to it the least (none at
        imbalance 1). With one or two arms every arm is on the axis.
        """
        n = self.n_arms
        if n == 0:
            return np.zeros(0)
        w = 1.0 + self.cfg.demand_imbalance * np.cos(2 * TWO_PI * np.arange(n) / n)
        return w / w.mean()

    def demand_factor(self, t=None):
        """The demand profile at time ``t`` (default: now) as a multiple of the
        entrance chance; its mean over each whole PEAK_PERIOD is exactly 1
        (over other windows see :func:`mean_demand_factor`).

        "peak" is 1 - PEAK_SWING * cos(2 pi t / PEAK_PERIOD): half the demand at
        t = 0, 10, 20... min, 1.5 times it at 5, 15, 25... min.
        """
        t = self.time if t is None else t
        if self.cfg.demand_profile == "peak":
            return 1.0 - PEAK_SWING * np.cos(TWO_PI * np.asarray(t, dtype=float) / PEAK_PERIOD)
        return np.ones_like(np.asarray(t, dtype=float)) if np.ndim(t) else 1.0

    def arrival_rates(self) -> np.ndarray:
        """[1/s] Mean arrival rate at each entrance right now (open mode); none at the capacity test's subject
        entrance (its queue is topped up instead: _top_up)."""
        rates = self.cfg.entry_rate * float(self.demand_factor()) * self.demand_weight
        if self.subject:
            rates[SUBJECT_ARM] = 0.0
        return rates

    @staticmethod
    def _can_test(cfg: SimConfig) -> bool:
        """Whether ``cfg`` could run the capacity test (subject_arm): open mode, with an entrance to test."""
        return cfg.traffic_mode == "open" and int(cfg.n_arms) > 0

    # ------------------------------------------------- vehicles and drivers
    def _update_effective(self) -> None:
        """The effective settings for the road condition, and whether vehicles and drivers differ.

        ``uniform_length``: every vehicle is a car (no trucks), so lengths are
        the one ``car_length``. ``automated``: there can be automated vehicles
        (auto_share > 0, or wave smoothers: ``smoothing``, smoothers > 0), which
        drive by their kind's rules. ``alike``: neither,
        and every driver keeps the same time gap and reaction time. Then the
        dynamics use plain numbers, which keeps the arithmetic, and the speed,
        of a model without trucks, kinds or driver spread; otherwise they use
        each vehicle's own values (the standstill gap only with automated
        vehicles: ``_driving``). Also whether the capacity test is on
        (``subject``), whether the automated vehicles are ACC cars or CAVs,
        whose controllers set their margins and time gaps every step
        (``controllers``: _controller_gaps), and the road condition's
        time-gap factor their fixed time gaps take (``road_gap``). All from the
        settings, never per step.
        """
        cfg = self.cfg
        self.eff = effective(cfg)
        self.subject = cfg.subject_arm == "on" and self._can_test(cfg)  # the capacity test (subject_arm)
        self.uniform_length = cfg.truck_share == 0
        self.smoothing = cfg.smoothers > 0
        self.automated = cfg.auto_share > 0 or self.smoothing
        self.alike = self.uniform_length and cfg.headway_std == 0 and cfg.reaction_std == 0 and not self.automated
        self.controllers = cfg.auto_share > 0 and cfg.auto_kind in ("acc", "cav")
        # *assumption*: the ACC and CACC time gaps follow the road condition as a driver's time gap does (an ACC
        # driver picks a longer setting; no source gives a CACC's in rain or snow), as a robotaxi's does. 1 if dry.
        self.road_gap = road_factors(cfg.road_condition, cfg.weather_scale)["time_headway"]

    def _draw_trucks(self, k: int) -> np.ndarray:
        """Whether each of ``k`` new vehicles is a truck (chance ``truck_share``, from rng_drivers)."""
        return self.rng_drivers.random(k) < self.cfg.truck_share

    def _fill_types(self, k: int) -> np.ndarray:
        """Whether each of the next ``k`` vehicles the density control adds is a truck: first the
        ones it could not place before (oldest first), then new draws."""
        waiting = self._pending[:k]
        return np.concatenate((np.array(waiting, dtype=bool), self._draw_trucks(k - len(waiting))))

    def _draw_kinds(self, k: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(kind, company, ACC-fallback gap [s]) of ``k`` new vehicles, in the order they are created.

        With auto_share > 0 each draws three uniforms u1, u2, u3 from rng_kinds
        (``rng``: the capacity test's top-up gives rng_kinds_test) (in that
        order, vehicle after vehicle), whatever it turns out to be:
        automated (of the kind auto_kind) iff u1 < auto_share; company
        floor(u2 x companies) if automated, else -1; its ACC-fallback gap
        ACC_GAPS picked by u3 (Liu 2018's shares). So a vehicle's draws never
        depend on the share, and at the same seed a higher share automates the
        same vehicles and more (monotone coupling). With auto_share 0 nothing is
        drawn: human drivers, no company, no gap (NaN); smoothers are picked
        from the drivers instead (_sync_smoothers).
        """
        if self.cfg.auto_share <= 0 or k <= 0:
            return np.zeros(k, dtype=np.int8), np.full(k, -1, dtype=np.int8), np.full(k, np.nan)
        cfg = self.cfg
        u = (self.rng_kinds if rng is None else rng).random((k, 3))
        auto = u[:, 0] < cfg.auto_share
        kind = np.where(auto, AUTO_KINDS[cfg.auto_kind], HUMAN).astype(np.int8)
        company = np.where(auto, np.minimum(np.floor(u[:, 1] * cfg.companies), cfg.companies - 1), -1).astype(np.int8)
        gap = np.asarray(ACC_GAPS)[np.searchsorted(ACC_GAP_EDGES, u[:, 2], side="right")]
        return kind, company, gap

    def _queue_kinds(self, arm: int, kinds: tuple[np.ndarray, np.ndarray, np.ndarray], k: int | None = None) -> None:
        """Put the first ``k`` (default all) of ``kinds`` (:meth:`_draw_kinds`) at the back of an entrance's
        queue fields, beside their trucks and arrival times (as Python values)."""
        for name, values in zip(("queue_kinds", "queue_companies", "queue_acc_gaps"), kinds):
            getattr(self, name)[arm].extend(values[:k].tolist())

    def _set_driving(self, i) -> None:
        """Each driver's own time gap, max acceleration, standstill gap and reaction delay (in steps),
        from their temperaments, their vehicle, their kind and the effective settings.

        Time gap: Following time gap + z * spread, at least HEADWAY_FLOOR of the
        setting, times robotaxi_headway for a robotaxi, times TRUCK_HEADWAY_FACTOR
        in a truck. Reaction time: Reaction time + z * spread within
        0-MAX_REACTION, rounded to whole steps. Standstill gap: _standstill_gaps.
        An automated vehicle's desired-speed and reaction temperaments are 0
        (_spawn), so it gets the settings themselves (one software, no spread:
        *assumption*); a robotaxi keeps its time-gap temperament (Hu et al.
        2023 Sec. III-B find Waymo's time-headway variance comparable to
        drivers', which they credit to its controller adapting to the
        situation). Its acceleration and comfortable braking are a driver's (a
        robotaxi's too: Hu et al. 2023 Table X finds no significant difference)
        and a robotaxi's reaction delay is Reaction time (*assumption*: Hu's
        longer response time, 2.42 against 1.80 s, Sec. III-D Table V, is a
        stimulus-to-response time that includes its longer time gap, not a
        slower reaction). The controllers' settings are fixed (no spread),
        times the company's brand factor (_brand) and the road's time-gap
        factor (``road_gap``): an ACC car keeps ACC time gap (at least
        ACC_GAP_MIN: _acc_law_gap) and the dead time ACC dead time (at most
        ACC_DEAD_MAX, the brand factor included); a CAV its ACC-fallback gap
        (acc_gap, at least ACC_GAP_MIN) and no dead time (its CACC time gap
        within a string is set every step, _controller_gaps).
        A smoother keeps its driver's values, temperaments and all (*model*: it
        was a driver's car, picked after placement, _sync_smoothers): its driver
        drives it before Smoother start, the quiet law is that driver's model,
        and the driver's IDM is its safety layer's reference and its gap cap's
        fallback; its FollowerStopper or PI law has fixed settings, the same for
        every smoother, and tracks without a reaction delay (_move).
        """
        cfg, eff = self.cfg, self.eff
        own_gap = np.maximum(HEADWAY_FLOOR * eff.time_headway, eff.time_headway + self.z_gap[i] * eff.headway_std)
        kind = self.kind[i]
        if self.automated:
            own_gap = np.where(kind == ROBOTAXI, cfg.robotaxi_headway * own_gap, own_gap)
        if self.controllers:
            ctrl = (kind == ACC) | (kind == CAV)
            own_gap = np.where(ctrl, self._acc_law_gap(np.where(kind == ACC, cfg.acc_headway, self.acc_gap[i]),
                                                       self.company[i]), own_gap)
        truck = self.truck[i]
        self.headway[i] = np.where(truck, TRUCK_HEADWAY_FACTOR * own_gap, own_gap)
        self.accel[i] = np.where(truck, TRUCK_ACCEL_FACTOR * eff.max_accel, eff.max_accel)
        self.s0[i] = self._standstill_gaps(i)
        reaction = np.clip(cfg.reaction_time + self.z_react[i] * cfg.reaction_std, 0.0, MAX_REACTION)
        if self.controllers:  # the ACC car's dead time (Gunter 2019 Eq. 3's tau, at most ACC_DEAD_MAX), a CAV none
            dead = np.where(kind == ACC, np.clip(cfg.acc_delay * self._brand(self.company[i]), 0.0, ACC_DEAD_MAX), 0.0)
            reaction = np.where(ctrl, dead, reaction)
        self.delay[i] = np.rint(reaction / cfg.dt).astype(np.int64)

    def _brand(self, company):
        """The brand factor of vehicles of these companies on the ACC/CACC time gaps and the ACC dead time:
        company k of n gets 1 + brand_spread (2k/(n-1) - 1), evenly spaced around 1 and the same every run (with
        brand_spread 0.6 around a 1.3 s gap, 0.52-2.08 s: Gunter et al. 2019 Table II's 0.54-2.05 s over
        production ACC settings, *derived*). The plain number 1 with one company or no spread; 1 for no company
        (-1)."""
        n, spread = int(self.cfg.companies), self.cfg.brand_spread
        if n == 1 or spread == 0:
            return 1.0
        return np.where(company >= 0, 1.0 + spread * (2.0 * np.maximum(company, 0) / (n - 1) - 1.0), 1.0)

    def _auto_gap(self, gap, company):
        """[s] A controller's time gap setting ``gap`` (ACC time gap, an ACC-fallback gap, CACC time gap or
        CAV_PLATOON_GAP) as vehicles of these companies keep it: times the brand factor and the road's time-gap
        factor (truck body: the caller)."""
        return gap * self._brand(company) * self.road_gap

    def _acc_law_gap(self, gap, company):
        """[s] The time gap the ACC law keeps for the setting ``gap`` (ACC time gap, or a CAV's ACC-fallback gap):
        _auto_gap, at least ACC_GAP_MIN (so Xiao's margin never makes the equilibrium spacing shrink with speed;
        truck body: the caller)."""
        return np.maximum(self._auto_gap(gap, company), ACC_GAP_MIN)

    @staticmethod
    def acc_margin(v) -> np.ndarray:
        """[m] The ACC spacing margin at speed ``v`` (Xiao et al. 2017 Eq. 6, less the vehicle length): ACC_S0 up
        to ACC_S0_LOW, ACC_S0_K / v - 5 m up to ACC_S0_HIGH (1.94 m just above 10.8 m/s: the source's own step),
        0 above."""
        v = np.asarray(v, dtype=float)
        return np.where(v <= ACC_S0_LOW, ACC_S0,
                        np.where(v < ACC_S0_HIGH, ACC_S0_K / np.maximum(v, ACC_S0_LOW) - 5.0, 0.0))

    @staticmethod
    def cacc_margin(v) -> np.ndarray:
        """[m] The CACC spacing margin at speed ``v`` (Xiao et al. 2017 Eq. 7, less the vehicle length): CACC_S0 at
        standstill, CACC_S0_SLOPE less per m/s, 0 from 10 m/s."""
        return np.maximum(CACC_S0 - CACC_S0_SLOPE * np.asarray(v, dtype=float), 0.0)

    def _standstill_gaps(self, i) -> np.ndarray:
        """[m] The standstill gap of the vehicles in ``i``: Standstill gap, ROBOTAXI_S0_EXTRA more for a robotaxi,
        ACC_S0 (the controllers' margin at standstill) for an ACC car or a CAV (a truck keeps its driver's: see
        TRUCK_*; the road condition never scales it, ROAD_SCALED). The ACC cars' and CAVs' margins then follow
        their speed and mode every step (_controller_gaps)."""
        return self._kind_s0(self.kind[i])

    def _kind_s0(self, kind) -> np.ndarray:
        """[m] The standstill gap of vehicles of these kinds (codes): see _standstill_gaps."""
        min_gap = self.cfg.min_gap
        return np.where(kind == ROBOTAXI, min_gap + ROBOTAXI_S0_EXTRA,
                        np.where((kind == ACC) | (kind == CAV), ACC_S0, min_gap))

    def _driving(self, slots):
        """(max acceleration, time gap, standstill gap) of the drivers in ``slots`` for the IDM: plain
        numbers while every driver is alike, else arrays aligned with ``slots`` (the standstill gap stays
        the plain min_gap while no vehicle can be automated, so the human model keeps its arithmetic)."""
        if self.alike:
            return self.eff.max_accel, self.eff.time_headway, self.cfg.min_gap
        return self.accel[slots], self.headway[slots], self.s0[slots] if self.automated else self.cfg.min_gap

    def _pair_length(self, behind, ahead):
        """[m] Distance between the centres of two vehicles bumper to bumper: half of each one's length."""
        if self.uniform_length:
            return self.cfg.car_length
        return 0.5 * (self.length[behind] + self.length[ahead])

    def _extra_length(self, slots) -> np.ndarray:
        """[m] How much longer than a car each vehicle is (0 for a car, exactly)."""
        return np.where(self.truck[slots], self.cfg.truck_length - self.cfg.car_length, 0.0)

    # ------------------------------------------------------- driver model
    def desired_speed(self, idx, lane=None) -> np.ndarray:
        """Free-road desired speed, capped by comfortable cornering on the car's lane
        (or on ``lane`` if given). Trucks want TRUCK_SPEED_FACTOR of their driver's.

        Floored at half the mean: a huge spread would otherwise create drivers
        who barely move, i.e. moving bottlenecks rather than phantom jams.
        """
        eff = self.eff
        v0 = np.maximum(0.5 * eff.mean_speed, eff.mean_speed + self.z[idx] * eff.speed_std)
        if not self.uniform_length:
            v0 = np.where(self.truck[idx], TRUCK_SPEED_FACTOR * v0, v0)
        lanes = self.lane[idx] if lane is None else lane
        return np.minimum(v0, np.sqrt(eff.lateral_accel * self.radii)[lanes])

    def target_speed(self, idx) -> np.ndarray:
        """Desired speed, also slowing down comfortably to ``turn_speed`` before an exit.

        Only cars already on the outer lane slow for their exit: a leaver still
        on an inner lane keeps pace with traffic so it can merge outwards.
        """
        cfg = self.cfg
        v = self.desired_speed(idx)
        to_exit = (self.exit_at[idx] - self.travel[idx]) * self.r_out
        approach = np.sqrt(cfg.turn_speed**2 + 2 * self.eff.comfort_decel * np.maximum(to_exit - 5.0, 0.0))
        slowing = np.isfinite(to_exit) & (self.lane[idx] == self.outer)
        return np.where(slowing, np.minimum(v, approach), v)

    def _free_acc(self, v, v0, a) -> np.ndarray:
        """IDM free-road acceleration towards ``v0`` for max acceleration ``a`` (a number, or
        per car). Above the desired speed it brakes gently (improved IDM) instead of the
        plain IDM's hard (v/v0)^4."""
        b = self.eff.comfort_decel
        v = np.asarray(v, dtype=float)
        v0 = np.maximum(np.asarray(v0, dtype=float), 0.1)
        free = a * (1 - (v / v0) ** 4)
        over = v > v0
        if over.any():
            a_over = a[over] if np.ndim(a) else a
            free[over] = -b * (1 - (v0[over] / v[over]) ** (4 * a_over / b))
        return free

    def _interaction(self, v, gap, v_lead, a, T, s0=None) -> np.ndarray:
        """IDM braking towards a leader ``gap`` metres (bumper to bumper) ahead, for drivers
        with max acceleration ``a``, time gap ``T`` and standstill gap ``s0`` (numbers, or aligned
        with ``v``; ``s0`` by default Standstill gap)."""
        b = self.eff.comfort_decel
        s0 = self.cfg.min_gap if s0 is None else s0
        s_star = s0 + np.maximum(0.0, v * T + v * (v - v_lead) / (2 * np.sqrt(a * b)))
        return a * (s_star / np.maximum(gap, 0.1)) ** 2

    def _idm(self, v, v0, gap, v_lead, slots, T=None, s0=None) -> np.ndarray:
        """IDM acceleration towards one leader for the drivers in ``slots`` (aligned with
        ``v``). ``gap`` is bumper to bumper; leaders beyond the reaction distance (or
        ``inf``) are ignored. ``T``, ``s0``: time gaps and standstill gaps in place of the
        drivers' own (aligned; None: their own, _driving), e.g. those a CAV would keep
        behind another leader (_gaps_behind)."""
        a, own_T, own_s0 = self._driving(slots)
        T = own_T if T is None else T
        s0 = own_s0 if s0 is None else s0
        v = np.asarray(v, dtype=float)
        seen = np.isfinite(gap) & (gap <= self.cfg.reaction_distance)
        return self._free_acc(v, v0, a) - np.where(seen, self._interaction(v, np.where(seen, gap, 1.0), v_lead,
                                                                           a, T, s0), 0.0)

    def _car_following(self, idx, v, v0, ld, g, lead, gap) -> np.ndarray:
        """Car-following acceleration with multi-anticipation (as in Treiber's
        Human Driver Model): a driver reacts to every car ahead in its lane
        within the reaction distance (up to ``anticipation`` cars, by default
        ANTICIPATION; 1 = the plain IDM), not just the next one. The j-th car
        ahead is judged by the average gap per car to it
        and weighted 1/j^2; the weighted mean of these IDM braking terms
        replaces the single-leader term. In even traffic this changes nothing,
        but drivers who see further brake earlier and more gently for a slowdown
        a few cars ahead, which damps stop-and-go waves. Arrays are aligned with
        ``idx``; ``ld``/``g`` are each car's leader slot and bumper gap, and
        ``lead``/``gap`` the full-size arrays from :meth:`_index_lanes`.
        Each driver uses their own acceleration, time gap and standstill gap;
        the distance to the j-th car ahead counts the real lengths of the cars
        in between. Automated vehicles react to their direct leader only
        (*model*: what their sensors track; Hu et al. 2023 calibrate the plain
        IDM to Waymo's following, and the ACC and CACC laws use it too), but a
        smoother its driver drives, or under the quiet law (_anticipates).
        """
        cfg = self.cfg
        a, T, s0 = self._driving(idx)
        seen = np.isfinite(g) & (g <= cfg.reaction_distance)
        v_lead = np.where(ld >= 0, self.v[np.maximum(ld, 0)], v)
        braking = np.where(seen, self._interaction(v, np.where(seen, g, 1.0), v_lead, a, T, s0), 0.0)
        weight = seen.astype(float)
        cars_ahead = np.bincount(self.lane[idx], minlength=self.n_lanes)[self.lane[idx]] - 1
        car, gaps = np.maximum(ld, 0), g.copy()
        between = 0.0  # [m] summed lengths of the cars passed so far (vehicles of mixed lengths)
        anticipating = seen  # who looks beyond the car in front
        if self.automated:
            anticipating = seen & self._anticipates(self.kind[idx])
        for j in range(2, int(cfg.anticipation) + 1):
            gaps = gaps + gap[car]  # summed bumper gaps to the j-th car ahead
            if not self.uniform_length:
                between = between + self.length[car]
            car = lead[car]
            reach = gaps + ((j - 1) * cfg.car_length if self.uniform_length else between)
            sees = anticipating & (j <= cars_ahead) & (reach <= cfg.reaction_distance)
            if not sees.any():
                break
            w = 1.0 / j**2
            braking += np.where(sees, w * self._interaction(v, gaps / j, self.v[car], a, T, s0), 0.0)
            weight += np.where(sees, w, 0.0)
        return self._free_acc(v, v0, a) - np.where(weight > 0, braking / np.maximum(weight, 1e-12), 0.0)

    def _driven(self, kind) -> np.ndarray:
        """Which vehicles of these kinds (codes) a human drives this step: human drivers, and smoothers before
        Smoother start (each drives as its driver until then: Stern et al. 2018 let the waves form first). The
        others are under their controller: no noise (_move), the clip and the safety layer (_automated_limits)."""
        driven = kind == HUMAN
        if self.smoothing and self.time < self.cfg.smoother_start:
            driven = driven | (kind == SMOOTHER)
        return driven

    def _anticipates(self, kind) -> np.ndarray:
        """Which vehicles of these kinds look beyond the car in front (_car_following): those a human drives
        (_driven), and smoothers under the quiet law (the placebo: its driver's own model, multi-anticipation
        and all, without the noise)."""
        if self.smoothing and self.cfg.smoother_law == "quiet":
            return (kind == HUMAN) | (kind == SMOOTHER)
        return self._driven(kind)

    def _smoothers_track(self) -> bool:
        """Whether the smoothers' controllers track a command speed this step (FollowerStopper or PI, from
        Smoother start on): then without a reaction delay (_move; SMOOTHER_TAU_* stand for the whole response)."""
        return self.time >= self.cfg.smoother_start and self.cfg.smoother_law != "quiet"

    def _observed_kinds(self, kind) -> np.ndarray:
        """The kinds the observers count these vehicles (codes) as: a vehicle a human drives this step (_driven: a
        smoother before Smoother start) as a human driver's, so the automated exposure, harsh braking and the A of
        the pair types (PAIR_TYPES) start when its controller does."""
        if self.smoothing and self.time < self.cfg.smoother_start:
            return np.where(self._driven(kind), HUMAN, kind).astype(kind.dtype)
        return kind

    def _automated_command(self, idx, v, v0, ld, g, acc) -> None:
        """The automated vehicles' commands in place of ``acc`` (aligned with ``idx``: what
        :meth:`_car_following` gave every car), each kind by its law in AUTO_LAWS (``_law_<name>``).
        ``v``, ``v0``, ``ld``, ``g``: as there. Called only while automated vehicles are possible; what
        follows in ``_move`` applies on top of each kind's law: cooperative merging (_make_room_to_leave),
        then :meth:`_automated_limits` (the clip of every command to [-max_decel, accel] and the safety
        layer: after merging, which can lower a command again), then the reaction (dead) time, the braking
        limit and the no-overlap constraint as for every car."""
        kind = self.kind[idx]
        for code, law in AUTO_LAWS.items():
            m = np.flatnonzero(kind == code)
            if m.size:
                acc[m] = getattr(self, f"_law_{law}")(idx[m], v[m], v0[m], ld[m], g[m], acc[m])

    def _automated_limits(self, idx, acc, a_idm) -> None:
        """The last word on the automated vehicles' commands in ``acc`` (aligned with ``idx``, in place): after
        cooperative merging, before the harsh-braking check (``over_braking``) and the reaction delay.
        ``a_idm``: what _car_following gave each car before _automated_command, for an automated car the IDM
        towards its direct leader with its own T and s0 (its anticipation terms are masked; an ACC car's and a
        CAV's are the margin and time gap of its law this step, _controller_gaps).

        Each automated command is clipped to [-max_decel, accel] (the road's braking limit, the vehicle's
        acceleration), then passes the safety layer: where a_idm asks for more than comfortable braking, a =
        min(a, a_idm) (*assumption*: an automatic emergency braking layer; Milanes & Shladover 2014 and Xiao et
        al. 2017 note their laws are not collision-free). An activation is a step where a_idm asks for harder
        braking than the law's own command (after merging): counted per episode with a per-car hold of
        EMERGENCY_HOLD (``n_aeb``, ``aeb_until``). The IDM law of a robotaxi never activates it, and on a road
        whose braking limit is above comfortable braking (any road at the default settings) its commands pass
        unchanged. Here and not in _automated_command because _make_room_to_leave lowers a leaver's command to
        as little as -1.5 b and a yielding follower's to -b (b: the effective comfort_decel), which on snow or ice
        can lie beyond the road's braking limit (a firm Comfortable braking, or a Weather effect below 1).
        Smoothers pass it from Smoother start on, under every law (the quiet placebo's command is clipped as the
        others' are); before it their drivers drive them (_driven), as any driver."""
        auto = np.flatnonzero(~self._driven(self.kind[idx]))
        if auto.size == 0:
            return
        slots, law, ref = idx[auto], acc[auto], a_idm[auto]
        clipped = np.minimum(np.maximum(law, -self.eff.max_decel), self.accel[slots])
        unsafe = ref < -self.eff.comfort_decel
        acc[auto] = np.where(unsafe, np.minimum(clipped, ref), clipped)
        acting = slots[unsafe & (ref < law)]
        if acting.size:
            self.n_aeb += int((self.aeb_until[acting] < self.time).sum())
            self.aeb_until[acting] = self.time + EMERGENCY_HOLD

    def _law_idm(self, slots, v, v0, ld, g, a_idm) -> np.ndarray:
        """The IDM towards the direct leader with the car's own acceleration, time gap and standstill gap
        (_set_driving): what _car_following gave an automated car (its anticipation terms are masked there)."""
        return a_idm

    def _law_acc(self, slots, v, v0, ld, g, a_idm) -> np.ndarray:
        """The PATH ACC law (ACC_K1, ACC_K2: Milanes & Shladover 2014 Eq. 5) towards the direct leader at the
        car's time gap and the margin of its speed (headway, s0: _set_driving, _controller_gaps), never above
        the speed mode ACC_SPEED_GAIN (v0 - v) towards its target speed ``v0`` (cornering and exit slowing
        included: target_speed). The leader is the car in front whatever the gap (*assumption*: the radar's
        120-150 m reach, Liu et al. 2018 p. 22, covers every gap where the gap mode binds; far behind, the
        speed mode is the smaller). Its dead time is the reaction delay buffer's (delay, ACC dead time)."""
        return np.minimum(ACC_K1 * (g - self.s0[slots] - self.headway[slots] * v) + ACC_K2 * (self.v[ld] - v),
                          ACC_SPEED_GAIN * (v0 - v))

    def _law_cav(self, slots, v, v0, ld, g, a_idm) -> np.ndarray:
        """A CAV's command: PATH CACC (CACC_KP, CACC_KD per CACC_PERIOD: FHWA-OR-RD-22-11 Eqs. 7-3 to 7-6) behind a
        cooperating CAV (string_pos >= 2: _controller_gaps), with the gap error's derivative from its own
        acceleration last step (a_now); else the ACC law at its fallback gap, without a dead time (_law_acc's
        gap mode). Both at the time gap and margin _controller_gaps set this step, never above the speed mode
        (as the ACC law). The CACC law divides by max(CACC_PERIOD, dt) (see CACC_PERIOD), and beyond a time gap
        of CACC_GAIN_GAP by Tc / CACC_GAIN_GAP more (see there)."""
        T, v_lead = self.headway[slots], self.v[ld]
        error = g - self.s0[slots] - T * v
        cacc = self.string_pos[slots] >= 2
        period = max(CACC_PERIOD, self.cfg.dt) * np.maximum(1.0, T / CACC_GAIN_GAP)
        gap_law = np.where(cacc, (CACC_KP * error + CACC_KD * ((v_lead - v) - T * self.a_now[slots])) / period,
                           ACC_K1 * error + ACC_K2 * (v_lead - v))
        return np.minimum(gap_law, ACC_SPEED_GAIN * (v0 - v))

    def _law_smoother(self, slots, v, v0, ld, g, a_idm) -> np.ndarray:
        """A wave smoother's command (Smoother law): FollowerStopper (follower_stopper, set speed Smoother speed)
        or PI with saturation (pi_saturation, from its own mean speed over the last PI_WINDOW s: _mean_own_speed),
        tracked by smoother_tracking; where the bumper gap ``g`` exceeds max(SMOOTHER_CAP_GAP, SMOOTHER_CAP_TIME x
        v), its driver's IDM towards the direct leader instead (``a_idm``: Lichtle's gap cap, which keeps it from
        holding back behind a huge gap). Before Smoother start, and under the quiet law (its driver's own model
        without the noise: a placebo that tells the control law apart from the removal of one noisy driver),
        ``a_idm`` itself (_car_following gave it with the driver's multi-anticipation then: _anticipates).

        The command speed is kept per car (v_cmd, saved: PI updates it from its last value); a controller that
        engages (smoother_on was False: at Smoother start, on a switch of law, or a smoother picked later) starts
        from the car's speed. *model*: neither law commands more than the lane's comfortable cornering speed
        (sqrt(cornering limit x radius), as a driver's desired speed: desired_speed). What follows in _move
        applies as to every automated car: merging, then the clip and the safety layer (_automated_limits)."""
        if not self._smoothers_track():
            self.smoother_on[slots] = False
            return a_idm
        cfg = self.cfg
        v_lead = self.v[ld]
        engaging = slots[~self.smoother_on[slots]]
        self.v_cmd[engaging] = self.v[engaging]
        self.smoother_on[slots] = True
        cap = np.sqrt(self.eff.lateral_accel * self.radii)[self.lane[slots]]  # [m/s] the lane's cornering speed
        if cfg.smoother_law == "pi":
            v_cmd = np.minimum(self.pi_saturation(g, v, v_lead, self._mean_own_speed(slots), self.v_cmd[slots]), cap)
        else:
            v_cmd = self.follower_stopper(g, v, v_lead, np.minimum(cfg.smoother_speed, cap))
        self.v_cmd[slots] = v_cmd
        far = g > np.maximum(SMOOTHER_CAP_GAP, SMOOTHER_CAP_TIME * v)
        return np.where(far, a_idm, self.smoother_tracking(v_cmd - v, cfg.dt))

    @staticmethod
    def follower_stopper(dx, v, v_lead, U) -> np.ndarray:
        """[m/s] FollowerStopper's command speed (Stern et al. 2018 Eqs. 1-2, FS_DX0, FS_DECEL) for bumper gaps
        ``dx``, own speeds ``v``, leader speeds ``v_lead`` and set speed ``U`` (a number or aligned): 0 up to dx1,
        vbar (dx - dx1) / (dx2 - dx1) up to dx2, vbar + (U - vbar)(dx - dx2) / (dx3 - dx2) up to dx3, U beyond,
        with vbar = min(max(v_lead, 0), U) and dx_k = FS_DX0[k] + min(v_lead - v, 0)^2 / (2 FS_DECEL[k]) (the
        spans dx2 - dx1 and dx3 - dx2 are at least 0.75 m: no division by 0)."""
        dx, v, v_lead = (np.asarray(x, dtype=float) for x in (dx, v, v_lead))
        closing = np.minimum(v_lead - v, 0.0)
        dx1, dx2, dx3 = (x0 + closing * closing / (2.0 * d) for x0, d in zip(FS_DX0, FS_DECEL))
        vbar = np.minimum(np.maximum(v_lead, 0.0), U)
        return np.where(dx <= dx1, 0.0,
                        np.where(dx <= dx2, vbar * (dx - dx1) / (dx2 - dx1),
                                 np.where(dx <= dx3, vbar + (U - vbar) * (dx - dx2) / (dx3 - dx2), U)))

    @staticmethod
    def pi_saturation(dx, v, v_lead, U, v_cmd) -> np.ndarray:
        """[m/s] The next command speed of PI with saturation (Stern et al. 2018 Eqs. 3-5, PI_*) for bumper gaps
        ``dx``, own speeds ``v``, leader speeds ``v_lead``, mean own speeds ``U`` and last commands ``v_cmd``:
        beta (alpha v_target + (1 - alpha) v_lead) + (1 - beta) v_cmd, with v_target = U + PI_V_CATCH
        clip((dx - PI_GAP_LOW) / (PI_GAP_HIGH - PI_GAP_LOW), 0, 1), alpha = clip((dx - max(PI_SAFE_TIME v,
        PI_SAFE_GAP)) / PI_GAMMA, 0, 1) and beta = 1 - alpha / 2."""
        dx, v = np.asarray(dx, dtype=float), np.asarray(v, dtype=float)
        target = U + PI_V_CATCH * np.clip((dx - PI_GAP_LOW) / (PI_GAP_HIGH - PI_GAP_LOW), 0.0, 1.0)
        alpha = np.clip((dx - np.maximum(PI_SAFE_TIME * v, PI_SAFE_GAP)) / PI_GAMMA, 0.0, 1.0)
        beta = 1.0 - 0.5 * alpha
        return beta * (alpha * target + (1.0 - alpha) * v_lead) + (1.0 - beta) * v_cmd

    @staticmethod
    def smoother_tracking(dv, dt: float) -> np.ndarray:
        """[m/s2] A smoother's acceleration towards its command speed, ``dv`` = v_cmd - v above its speed: dv /
        tau, tau = SMOOTHER_TAU_BRAKE where dv <= -SMOOTHER_BRAKE_BAND (Stern's brake loop), else
        SMOOTHER_TAU_ACCEL; tau at least the step ``dt`` (*derived*: an explicit step of a first-order lag
        overshoots when dt > tau and diverges beyond 2 tau; at most there in one step; at the 0.1 s default the
        published values themselves)."""
        dv = np.asarray(dv, dtype=float)
        tau = np.where(dv <= -SMOOTHER_BRAKE_BAND, SMOOTHER_TAU_BRAKE, SMOOTHER_TAU_ACCEL)
        return dv / np.maximum(tau, dt)

    def _record_speeds(self, idx, v) -> None:
        """Observer of the smoothers' own speeds (``v``, aligned with ``idx``: at the start of this step), into
        their history: row ``steps`` mod its length (speed_hist: the last PI_WINDOW s in whole steps), and
        speed_count, how many rows hold the car's own (at most all). Every step, whatever the law and before
        Smoother start too, so PI with saturation starts from the mean of its last PI_WINDOW s (as Stern's car,
        driven by its driver until then)."""
        m = np.flatnonzero(self.kind[idx] == SMOOTHER)
        if m.size:
            slots, rows = idx[m], self.speed_hist.shape[0]
            self.speed_hist[self.steps % rows, slots] = v[m]
            self.speed_count[slots] = np.minimum(self.speed_count[slots] + 1, rows)

    def _mean_own_speed(self, slots) -> np.ndarray:
        """[m/s] Each smoother's mean own speed over the rows of its history that hold its own (_record_speeds:
        the last PI_WINDOW s, or the time since it was picked if shorter); PI with saturation's U."""
        rows = self.speed_hist.shape[0]
        age = (self.steps - np.arange(rows)) % rows  # steps since each row was written (0: this step's)
        count = self.speed_count[slots]
        own = age[:, None] < count[None, :]
        return np.where(own, self.speed_hist[:, slots], 0.0).sum(axis=0) / np.maximum(count, 1)

    def _coop_class(self, kind, company, other_kind, other) -> np.ndarray:
        """The cooperation class (index into COOP_CLASSES) of pairs of vehicles of kinds ``kind`` and ``other_kind``
        (codes) and companies ``company`` and ``other`` (numbers or aligned arrays): 0 ("none") unless both are
        CONNECTED (CAVs; a human driver's company -1 equals another's, so the kinds must be tested); for two CAVs
        coop_class within a company, the lower of coop_class and coop_between between two ("same": coop_class).
        Compare it with COOP_CACC (strings) and COOP_INTENT."""
        cfg = self.cfg
        within = COOP_CLASSES.index(cfg.coop_class)
        between = within if cfg.coop_between == "same" else min(within, COOP_CLASSES.index(cfg.coop_between))
        both = (np.asarray(kind) == CAV) & (np.asarray(other_kind) == CAV)  # (CONNECTED = {CAV})
        return np.where(both, np.where(company == other, within, between), 0)

    def _controller_gaps(self, idx, v, ld) -> None:
        """At the start of a step, before the car following: the ACC cars' and CAVs' spacing margins at their
        speed ``v`` (s0: acc_margin, cacc_margin), each CAV's place in its CACC string (string_pos:
        _string_positions) and its time gap (headway): CACC time gap as a string member, CAV_PLATOON_GAP as the
        member that would be one too many, else its ACC-fallback gap (each x brand x road factor; a truck x
        TRUCK_HEADWAY_FACTOR; the ACC-fallback gap at least ACC_GAP_MIN). So the laws, and the IDM _car_following
        gives these cars (the safety layer's reference, _automated_limits), use the margin and time gap the car
        regulates to behind its leader now. A lane change gives a CAV another leader, and with it another time gap
        and margin: MOBIL judges each car involved by those it would keep after the change (_gaps_behind). The entry
        checks (_entry_ok) use the lag vehicle's present ones. ``idx``, ``v``, ``ld``: as in _move."""
        kind = self.kind[idx]
        acc = np.flatnonzero(kind == ACC)
        if acc.size:
            self.s0[idx[acc]] = self.acc_margin(v[acc])
        cav = np.flatnonzero(kind == CAV)
        if cav.size == 0:
            return
        pos = self._string_positions(idx, ld, kind)
        self.string_pos[idx] = pos
        slots, p, vc = idx[cav], pos[cav], v[cav]
        self.headway[slots], self.s0[slots] = self._cav_gaps(slots, p, vc)

    def _cav_gaps(self, slots, pos, v) -> tuple[np.ndarray, np.ndarray]:
        """(time gap [s], margin [m]) of the CAVs in ``slots`` at string positions ``pos`` (_string_positions) and
        speeds ``v`` (aligned): a member (pos >= 2) CACC time gap, or CAV_PLATOON_GAP as the one too many (pos >
        platoon_max), with the CACC margin; a string leader (pos 1) its ACC-fallback gap (at least ACC_GAP_MIN) with
        the ACC margin. Each x brand x road factor, a truck x TRUCK_HEADWAY_FACTOR."""
        cacc = pos >= 2
        member = self._auto_gap(np.where(pos > self.cfg.platoon_max, CAV_PLATOON_GAP, self.cfg.cav_headway),
                                self.company[slots])
        gap = np.where(cacc, member, self._acc_law_gap(self.acc_gap[slots], self.company[slots]))
        return (np.where(self.truck[slots], TRUCK_HEADWAY_FACTOR * gap, gap),
                np.where(cacc, self.cacc_margin(v), self.acc_margin(v)))

    def _gaps_behind(self, cars, leaders, leader_pos=None):
        """(time gap, margin, string position) the vehicles in ``cars`` would keep behind ``leaders`` (aligned
        slots; -1: none; a car's own slot: alone in its lane), with each leader's string position ``leader_pos``
        (default its present string_pos), for MOBIL (_lane_changes): a lane change gives a CAV another leader,
        and the time gap and margin it regulates to follow it (_controller_gaps). A CAV behind a CAV it cooperates
        with at class COOP_CACC or more joins its string: the next position (after platoon_max, platoon_max + 1,
        the one too many; after that, 2; a CAV at platoon_max + 1 now keeps leading its string there,
        _string_positions, whether it would merge into the string ahead left out: the longer gap), so the CACC
        time gap or CAV_PLATOON_GAP with the CACC margin; behind anything else it leads (position 1) at its
        ACC-fallback gap with the ACC margin (_cav_gaps). Every other vehicle keeps its present ones (headway,
        s0; position 0, or its present one). (None, None, None) while no vehicle is a CAV here (each keeps its
        own: _driving)."""
        if not self.controllers or self.cfg.auto_kind != "cav":
            return None, None, None
        cav = np.flatnonzero(self.kind[cars] == CAV)
        if cav.size == 0:
            return None, None, None
        T, s0, pos = self.headway[cars], self.s0[cars], self.string_pos[cars].astype(np.int64)
        c, lead = cars[cav], leaders[cav]
        ahead = np.maximum(lead, 0)
        p_lead = (self.string_pos[ahead] if leader_pos is None else np.asarray(leader_pos)[cav]).astype(np.int64)
        joined = (lead >= 0) & (lead != c) & (self._coop_class(CAV, self.company[c], self.kind[ahead],
                                                               self.company[ahead]) >= COOP_CACC)
        pm = int(self.cfg.platoon_max)
        p_lead = np.maximum(p_lead, 1)  # (a CAV is at least the leader of its own string)
        p = np.where(joined, np.where(p_lead > pm, 2, np.where(p_lead == pm, pm + 1, p_lead + 1)), 1)
        p = np.where(joined & (pos[cav] == pm + 1), pm + 1, p)  # (a cut keeps leading its string)
        T[cav], s0[cav] = self._cav_gaps(c, p, self.v[c])
        pos[cav] = p
        return T, s0, pos

    def _string_positions(self, idx, ld, kind) -> np.ndarray:
        """Each car's place in its CACC string (aligned with ``idx``; ``ld`` its leader's slot, ``kind`` its
        kind; int8): 0 for any but a CAV; 1 for a CAV that leads a string (no cooperating CAV directly ahead:
        it follows by the ACC law); for a CAV joined to a cooperating CAV directly ahead (class COOP_CACC or
        more, _coop_class), d links from the head of its part of the chain: d mod platoon_max + 1, or
        platoon_max + 1 where d is a multiple of platoon_max (the CAV that would be one too many keeps
        CAV_PLATOON_GAP and leads the next string, FHWA-OR-RD-22-11 p. 108, Liu et al. 2018 pp. 24-25). The
        heads: string leaders and, *model*, every CAV that was platoon_max + 1 last step (string_pos) and is still
        joined: a string once cut keeps its head (FHWA-OR-RD-22-11 Sec. 4.2.2 p. 60: "a CAV can be either a
        platoon leader or a follower"; the sources give no rule for re-forming strings). So a vehicle joining or
        leaving the chain ahead of a cut (an entry, an exit, a lane change) re-counts the cars up to the cut only.
        (Counted afresh from the chain's leader, every cut behind such a change moved by one car: the member just
        behind a cut became the one too many and opened its CACC gap to CAV_PLATOON_GAP at once, braking hard,
        *derived*: 43 of the 54 harsh-braking onsets of CAVs at 100 % on two lanes at density 80, seeds 1-3, 300 s.)
        Such a head merges into the string ahead where both fit in one: the cars from the string ahead's head to
        its own leader, plus itself and the members behind it (up to platoon_max), at most platoon_max, so no car
        behind it changes its time gap; one merge per chain of strings per step. A string that grows past
        platoon_max (a vehicle joining it) is still cut by the count, and a string leader that comes to follow
        another string joins it by the count (two strings meeting: its tail may split off). *model*: membership
        follows the chain of direct leaders whatever the gap (far behind, its law's speed mode governs, as Liu's
        CACC switches to speed mode above a 2 s time gap); a lane in which every vehicle is a joined CAV and none a
        head forms a loop with no leader, which the CAV with the lowest uid leads (at CAV_PLATOON_GAP, whatever
        the loop's length). Worked out every step from the leader chain by pointer jumping (_chain_depth:
        vectorised, ceil(log2 n) + 1 passes at most, two or three times), so any chain gets its exact count."""
        n = idx.size
        me = np.arange(n)
        at = np.empty(self.cap, dtype=np.int64)
        at[idx] = me
        lp = at[ld]  # each car's leader's place in idx
        cav = kind == CAV
        joined = cav & (kind[lp] == CAV) & (lp != me)
        if joined.any():
            pair = self._coop_class(CAV, self.company[idx[joined]], CAV, self.company[ld[joined]])
            joined[joined] = pair >= COOP_CACC
        pos = cav.astype(np.int64)  # (string leaders 1)
        if not joined.any():
            return pos.astype(np.int8)
        pm = int(self.cfg.platoon_max)
        key = self.uid[idx]
        cut = joined & (self.string_pos[idx] == pm + 1)  # last step's heads after a full string, still joined
        head = cut.copy()
        depth, loop, low, up = _chain_depth(joined & ~head, lp, key, heads=True)
        if loop.any():  # every vehicle of a lane a joined CAV, none a head: its lowest uid leads the loop
            head |= loop & (key == low)
            depth, _, _, up = _chain_depth(joined & ~head, lp, heads=True)
        if cut.any():
            c = np.flatnonzero(cut)
            ahead = depth[lp[c]] % pm + 1  # the string ahead: its head to this head's leader
            own = np.minimum(np.bincount(up, minlength=n)[c], pm)  # this head's string: itself and its members
            merge = np.zeros(n, dtype=bool)
            merge[c] = ahead + own <= pm
            merge[c] &= ~merge[up[lp[c]]]  # (not while the string ahead's head merges too, nor into its own loop)
            if merge.any():
                depth, _, _ = _chain_depth(joined & ~(head & ~merge), lp)
        rest = depth % pm
        return np.where(joined, np.where(rest == 0, pm + 1, rest + 1), pos).astype(np.int8)

    def equilibrium_speed(self, gap, v0, slots=None) -> np.ndarray:
        """Steady-state IDM speed for a bumper gap (used to start cars smoothly), for the
        drivers in ``slots`` (aligned with ``gap``; default: the effective Following time gap and
        Standstill gap), with their own time gap and standstill gap."""
        cfg = self.cfg
        T, s0 = (self.eff.time_headway, cfg.min_gap) if slots is None else self._driving(slots)[1:]
        gap = np.asarray(gap, dtype=float)
        v0 = np.maximum(np.broadcast_to(np.asarray(v0, dtype=float), gap.shape), 0.1)
        lo, hi = np.zeros_like(gap), v0.copy()
        seen = gap <= cfg.reaction_distance
        for _ in range(40):  # bisection on the IDM steady state (acceleration = 0)
            mid = 0.5 * (lo + hi)
            f = 1 - (mid / v0) ** 4 \
                - np.where(seen, ((s0 + mid * T) / np.maximum(gap, 0.1)) ** 2, 0)
            lo, hi = np.where(f > 0, mid, lo), np.where(f > 0, hi, mid)
        return np.where(gap > s0, lo, 0.0)

    # ------------------------------------------------------ neighbourhood
    def _index_lanes(self, idx):
        """Sort live cars by angle within each lane.

        Returns full-size arrays ``lead`` (slot of the car ahead, -1 if none)
        and ``gap`` (bumper gap to it [m]: the distance between their centres
        less half of each one's length); a car alone in its lane follows
        itself round the ring. Also caches each lane's sorted slots/angles.
        """
        lead = np.full(self.cap, -1, dtype=np.int64)
        gap = np.full(self.cap, np.inf)
        # Stable sorts by angle, then by lane: grouped by lane, by angle within it.
        order = idx[np.argsort(self.theta[idx], kind="stable")]
        order = order[np.argsort(self.lane[order].astype(np.int16), kind="stable")]  # radix sort
        lanes, th = self.lane[order], self.theta[order]
        bounds = np.searchsorted(lanes, np.arange(self.n_lanes + 1))
        filled = bounds[1:] > bounds[:-1]
        nxt_pos = np.arange(1, order.size + 1)
        nxt_pos[bounds[1:][filled] - 1] = bounds[:-1][filled]  # the last car in a lane follows the first
        nxt = order[nxt_pos]
        dth = th[nxt_pos] - th
        dth[dth < 0] += TWO_PI  # == % TWO_PI on (-2pi, 2pi), but 3x cheaper
        dth[nxt == order] = TWO_PI
        lead[order] = nxt
        gap[order] = dth * self.radii[lanes] - self._pair_length(order, nxt)
        self._lanes = [(order[a:b], th[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]
        return lead, gap

    def bumper_gaps(self) -> tuple[np.ndarray, np.ndarray]:
        """(slots of the live cars, bumper gap from each to the car ahead in its lane [m])."""
        idx = np.flatnonzero(self.alive)
        _, gap = self._index_lanes(idx)
        return idx, gap[idx]

    def _neighbours_in(self, lane: int, angles: np.ndarray, length=None):
        """Leader/follower slots and bumper gaps for vehicles ``length`` m long (a number or
        aligned with ``angles``; default a car) centred at ``angles`` in ``lane``."""
        order, th = self._lanes[lane]
        n = angles.size
        if order.size == 0:
            return (np.full(n, -1), np.full(n, np.inf), np.full(n, -1), np.full(n, np.inf))
        pos = np.searchsorted(th, angles)
        ld = order[pos % order.size]
        fl = order[(pos - 1) % order.size]
        r = self.radii[lane]
        if self.uniform_length:
            front_len = back_len = self.cfg.car_length
        else:
            own = self.cfg.car_length if length is None else length
            front_len, back_len = 0.5 * (own + self.length[ld]), 0.5 * (own + self.length[fl])
        front = ((self.theta[ld] - angles) % TWO_PI) * r - front_len
        back = ((angles - self.theta[fl]) % TWO_PI) * r - back_len
        return ld, front, fl, back

    def _neighbours(self, lanes: np.ndarray, angles: np.ndarray, length=None):
        """:meth:`_neighbours_in` for positions spread over several lanes (``length`` a number
        or aligned with ``angles``)."""
        n = angles.size
        out = (np.full(n, -1, dtype=np.int64), np.full(n, np.inf), np.full(n, -1, dtype=np.int64), np.full(n, np.inf))
        for lane in range(self.n_lanes):
            m = lanes == lane
            if m.any():
                part_len = length[m] if np.ndim(length) else length
                for arr, part in zip(out, self._neighbours_in(lane, angles[m], part_len)):
                    arr[m] = part
        return out

    # ----------------------------------------------------------- spawning
    def _spawn(self, lanes, angles, speeds, lat=None, trucks=None, *, kinds=None) -> np.ndarray:
        """New vehicles centred at ``angles``; ``trucks`` says which are trucks (default: drawn), ``kinds``
        their (kind, company, ACC-fallback gap) arrays (default: drawn, :meth:`_draw_kinds`). Their drivers'
        temperaments come from rng_drivers (nothing extra from the main stream). An automated vehicle draws
        every temperament a driver in its place would and then sets its desired-speed and reaction ones to 0:
        one software, the settings themselves (*assumption*). A robotaxi keeps its time-gap temperament, so
        robotaxis spread their time gaps as drivers do (Hu et al. 2023 Sec. III-B find Waymo's time-headway
        variance comparable to drivers', which they credit to its controller adapting to the situation); the
        other automated kinds set it to 0 too (fixed controller settings). Every stream but the kinds'
        (rng_kinds, rng_kinds_test) draws as without automated vehicles."""
        k = len(angles)
        free = np.flatnonzero(~self.alive)
        if free.size < k:
            self._grow(k - free.size)
            free = np.flatnonzero(~self.alive)
        i = free[:k]
        self.kind[i], self.company[i], self.acc_gap[i] = self._draw_kinds(k) if kinds is None else kinds
        self.alive[i] = True
        self.lane[i] = lanes
        self.theta[i] = np.asarray(angles) % TWO_PI
        self.travel[i] = 0.0
        self.v[i] = speeds
        self.z[i] = np.clip(self.rng.standard_normal(k), -2.5, 2.5)
        self.born[i] = self.time
        self.lat[i] = lanes if lat is None else lat
        self.lc_cool[i] = self.rng.uniform(0, LC_COOLDOWN, k)
        self.noise[i] = 0.0
        self.brake_until[i] = -np.inf
        self.uid[i] = np.arange(self._next_uid, self._next_uid + k)
        self._next_uid += k
        self.moving[i] = False
        self.emergency_until[i] = -np.inf
        self.stuck[i] = 0.0
        self.entered_at[i] = -np.inf
        self.inserted_at[i] = -np.inf
        self.acc_hist[:, i] = 0.0
        self.a_now[i] = 0.0
        self.string_pos[i] = 0
        self.aeb_until[i] = -np.inf
        self.ttc_until[i] = -np.inf
        self.v_cmd[i] = 0.0  # (no smoother: smoothers are picked from the drivers, _sync_smoothers)
        self.smoother_on[i] = False
        self.speed_count[i] = 0
        self.truck[i] = self._draw_trucks(k) if trucks is None else trucks
        self.length[i] = np.where(self.truck[i], self.cfg.truck_length, self.cfg.car_length)
        self.z_react[i], self.z_gap[i] = np.clip(self.rng_drivers.standard_normal((2, k)),
                                                 -TEMPERAMENT_CLIP, TEMPERAMENT_CLIP)
        if self.automated:  # automated: drawn as a driver's, then masked (desired speed, reaction; time gap but a
            auto = self.kind[i] != HUMAN  # robotaxi's)
            for name in ("z", "z_react"):
                getattr(self, name)[i] = np.where(auto, 0.0, getattr(self, name)[i])
            self.z_gap[i] = np.where(auto & (self.kind[i] != ROBOTAXI), 0.0, self.z_gap[i])
        self._set_driving(i)
        self._assign_routes(i)
        return i

    def _assign_routes(self, i) -> None:
        """Draw each car's exit (_draw_routes); a smoother's is drawn as a driver's and then set to inf (it
        never leaves), every time, so a live Exit chance change re-routes the others only."""
        self._draw_routes(i)
        if self.smoothing:
            self.exit_at[i] = np.where(self.kind[i] == SMOOTHER, np.inf, self.exit_at[i])

    def _draw_routes(self, i) -> None:
        """Draw each car's exit: it leaves at each exit it passes with chance ``exit_prob``.

        In the capacity test (``subject``) nothing is drawn: every car leaves at
        the first exit after the subject entrance's entry point (``_subject_exit``),
        so the subject's own entrants never pass that point and every other car
        passes it once (an arm's exit comes before its entry). On one lane that
        is the next exit, taken the next time the car reaches it; with 2+ lanes
        it can be a later one, which a new car between the two (an entrant of an
        arm in between) reaches only after going round past the point. Routes
        are set once per car (a live Exit chance change leaves them alone).
        """
        self.exit_at[i] = np.inf
        if self.subject:
            exit_angle, entry = self.exit_angle[self._subject_exit], self.entry_angle[SUBJECT_ARM]
            if self._subject_exit == (SUBJECT_ARM + 1) % self.n_arms:
                ahead = (exit_angle - self.theta[i]) % TWO_PI
                self.exit_at[i] = self.travel[i] + np.where(ahead < 1e-6, ahead + TWO_PI, ahead)
            else:  # round to the entry point first (the subject's entrants start on it), then on to the exit
                to_entry = (entry - self.theta[i]) % TWO_PI
                to_entry = np.where(to_entry > TWO_PI - 1e-6, 0.0, to_entry)
                self.exit_at[i] = self.travel[i] + to_entry + (exit_angle - entry) % TWO_PI
            return
        p = self.cfg.exit_prob
        if self.n_arms == 0 or p <= 0 or len(i) == 0:
            return
        k = self.rng.geometric(min(p, 1.0), size=len(i))  # exits passed, >= 1
        ahead = (self.exit_angle[None, :] - self.theta[i][:, None]) % TWO_PI
        ahead = np.where(ahead < 1e-6, ahead + TWO_PI, ahead)
        ahead.sort(axis=1)
        laps, nth = np.divmod(k - 1, self.n_arms)
        self.exit_at[i] = self.travel[i] + ahead[np.arange(len(i)), nth] + TWO_PI * laps

    def _place_initial(self, n: int) -> None:
        """Lay out ``n`` vehicles by lane length in the init_spacing pattern, at their steady speeds.

        Positions are vehicle centres. Trucks are drawn first and each pattern
        makes room for their extra length (``extra``, 0 for a car): spacings
        grow by half the extra length of the two vehicles they join, so the
        bumper gaps follow the pattern whatever the mix.
        """
        if n <= 0:
            return
        cfg = self.cfg
        pitch = cfg.car_length + cfg.min_gap  # of a car
        share = n * self.lane_len / self.lane_len.sum()
        counts = np.floor(share).astype(int)
        counts[np.argsort(counts - share)[: n - counts.sum()]] += 1
        counts = np.minimum(counts, np.floor(self.lane_len / (self.mean_length + cfg.min_gap)).astype(int))
        # A clustered start is one packed block across all lanes, so it starts
        # at the same angle in every lane.
        start = self.rng.uniform(0, TWO_PI) if cfg.init_spacing == "clustered" else 0.0
        for lane, k in enumerate(counts):
            if k == 0:
                continue
            C, r = self.lane_len[lane], self.radii[lane]
            trucks = self._draw_trucks(k)
            extra = np.where(trucks, cfg.truck_length - cfg.car_length, 0.0)
            while not self.uniform_length and k and k * pitch + extra[:k].sum() > C:
                k -= 1  # more trucks than this lane's share holds: the last ones do not fit here
            if k == 0:
                continue
            trucks, extra = trucks[:k], extra[:k]
            pair = 0.5 * (extra + np.roll(extra, -1))  # half the extra length of each vehicle and the next
            if cfg.init_spacing == "random":
                offset = self.rng.uniform(0, C)
                gaps = pitch + (C - k * pitch - extra.sum()) * self.rng.dirichlet(np.ones(k)) + pair
                s = offset + np.concatenate(([0.0], np.cumsum(gaps[:-1])))
            elif cfg.init_spacing == "clustered":
                s = start * r + min(pitch * 1.05, (C - extra.sum()) / k) * np.arange(k) \
                    + np.concatenate(([0.0], np.cumsum(pair[:-1])))
            else:  # uniform
                s = self.rng.uniform(0, C) + C * np.arange(k) / k + self._even_shift(extra)
            self._spawn(np.full(k, lane), s / r, np.zeros(k), trucks=trucks)
        idx = np.flatnonzero(self.alive)
        _, gap = self._index_lanes(idx)
        self.v[idx] = self.equilibrium_speed(gap[idx], self.desired_speed(idx), idx)

    @staticmethod
    def _even_shift(extra: np.ndarray) -> np.ndarray:
        """[m] Shifts that turn evenly spaced centres into even bumper gaps for vehicles ``extra`` m
        longer than a car, in driving order round a lane: each moves on by half the extra length of
        every pair behind it, less the mean. All 0 (exactly) when every vehicle is a car."""
        pair = 0.5 * (extra + np.roll(extra, -1))
        return np.concatenate(([0.0], np.cumsum(pair[:-1] - extra.mean())))

    # ------------------------------------------------------- wave smoothers
    def _sync_smoothers(self) -> None:
        """Pick the smoothers (no draws, no _index_lanes: from the cars' angles): after the initial placement,
        and again when Smoothers or Smoother lane change live. Smoother k of n (Smoothers) has its place in lane
        k mod n_lanes (Smoother lane -1: round-robin from the innermost) or in Smoother lane (clipped to the
        ring's lanes), at the angle ref + 2 pi k / n. First the smoothers there are now keep the places nearest
        them in their lanes (pairs of a place and a smoother, nearest first, each used once), so a live raise
        keeps every one and a cut keeps those nearest the new places; each place left takes the nearest
        human-driven car in its lane that is no truck (in order of k; ties: the lowest slot; a lane with none
        left gets no more). ref is the angle of the present smoother with the lowest uid (0 at the start: it
        keeps its place, *model*).

        Cars that become smoothers turn kind SMOOTHER, company -1, and never leave (exit_at inf, kept by
        _assign_routes), change lane (_lane_changes) or get removed by the density control (_removable); each
        keeps its driver (_set_driving), who drives it until Smoother start, and starts with no speed history
        (_record_speeds). Smoothers left out are handed back to their drivers: kind HUMAN again, with a route
        drawn as any car's (from the main stream, at a live change)."""
        cfg = self.cfg
        current = self.alive & (self.kind == SMOOTHER)
        chosen = np.zeros(self.cap, dtype=bool)
        n = int(cfg.smoothers)
        if n > 0:
            mine = np.flatnonzero(current)
            ref = float(self.theta[mine[np.argmin(self.uid[mine])]]) if mine.size else 0.0
            lanes = np.arange(n) % self.n_lanes if cfg.smoother_lane < 0 else \
                np.full(n, min(int(cfg.smoother_lane), self.n_lanes - 1))
            angles = ref + TWO_PI * np.arange(n) / n
            free = np.ones(n, dtype=bool)  # places not taken yet

            def off(slots, angle):  # [rad] how far each car is from a place's angle
                return np.abs((self.theta[slots] - angle + np.pi) % TWO_PI - np.pi)

            if mine.size:  # the present smoothers keep the places nearest them, nearest pairs first
                d = np.where(lanes[:, None] == self.lane[mine][None, :], off(mine[None, :], angles[:, None]), np.inf)
                for flat in np.argsort(d, axis=None, kind="stable"):
                    k, j = divmod(int(flat), mine.size)
                    if not np.isfinite(d[k, j]):
                        break
                    if free[k] and not chosen[mine[j]]:
                        free[k], chosen[mine[j]] = False, True
            pool = self.alive & (self.kind == HUMAN) & ~self.truck
            for k in np.flatnonzero(free):
                cand = np.flatnonzero(pool & ~chosen & (self.lane == lanes[k]))
                if cand.size:
                    chosen[cand[np.argmin(off(cand, angles[k]))]] = True
        picked, dropped = np.flatnonzero(chosen & ~current), np.flatnonzero(current & ~chosen)
        if picked.size:
            self.kind[picked] = SMOOTHER
            self.company[picked] = -1
            self.exit_at[picked] = np.inf
            self.smoother_on[picked] = False
            self.speed_count[picked] = 0
            self.v_cmd[picked] = self.v[picked]
            self._set_driving(picked)
        if dropped.size:
            self.kind[dropped] = HUMAN
            self.smoother_on[dropped] = False
            self._set_driving(dropped)
            self._assign_routes(dropped)

    def _removable(self) -> np.ndarray:
        """Slots of the cars the density control may remove (_sync_population, _sync_ring): every live car but
        the smoothers."""
        if self.smoothing:
            return np.flatnonzero(self.alive & (self.kind != SMOOTHER))
        return np.flatnonzero(self.alive)

    # ------------------------------------------------------- user actions
    def brake(self, slot: int, duration: float = 3.0) -> None:
        """Make one driver brake hard for ``duration`` seconds (a perturbation)."""
        if 0 <= slot < self.cap and self.alive[slot]:
            self.brake_until[slot] = self.time + duration

    def brake_random(self, duration: float = 3.0) -> int | None:
        idx = np.flatnonzero(self.alive)
        if idx.size == 0:
            return None
        slot = int(self.rng.choice(idx))
        self.brake(slot, duration)
        return slot

    def car_near(self, x: float, y: float, max_dist: float = 4.0) -> int | None:
        """Slot of the live car closest to world point (x, y), if within ``max_dist``.

        A vehicle longer than a car counts as near all along its body: the
        distance along its heading is shortened by half its extra length.
        """
        idx = np.flatnonzero(self.alive)
        if idx.size == 0:
            return None
        cx, cy = self.positions(idx)
        dx, dy = x - cx, y - cy
        th = self.theta[idx]
        along = dx * -np.sin(th) + dy * np.cos(th)  # along the driving direction (anticlockwise)
        across = dx * np.cos(th) + dy * np.sin(th)
        d = np.hypot(np.maximum(np.abs(along) - 0.5 * self._extra_length(idx), 0.0), across)
        k = int(np.argmin(d))
        return int(idx[k]) if d[k] <= max_dist else None

    # -------------------------------------------------------------- views
    def positions(self, idx):
        r = self.lane_radius(self.lat[idx])
        return r * np.cos(self.theta[idx]), r * np.sin(self.theta[idx])

    def cars(self) -> dict[str, np.ndarray]:
        """Snapshot of the live cars (all arrays aligned)."""
        idx = np.flatnonzero(self.alive)
        x, y = self.positions(idx)
        to_exit = (self.exit_at[idx] - self.travel[idx]) * self.r_out
        # Angle cars across while a lane change is being animated (outwards = clockwise tilt).
        drift = np.clip(self.lane[idx] - self.lat[idx], -1, 1) * self.cfg.lane_width / LC_DURATION
        tilt = np.arctan2(np.where(np.abs(self.lane[idx] - self.lat[idx]) > 1e-6, drift, 0.0),
                          np.maximum(self.v[idx], 1.0))
        if not self.uniform_length:  # a truck turns no further than to put its front half a lane across from its back
            cap = np.where(self.truck[idx], np.arctan(0.5 * self.cfg.lane_width / self.length[idx]), np.pi / 2)
            tilt = np.clip(tilt, -cap, cap)
        return {
            "slot": idx,
            "uid": self.uid[idx],
            "lane": self.lane[idx],
            "lat": self.lat[idx],
            "theta": self.theta[idx],
            "x": x,
            "y": y,
            "heading": self.theta[idx] + np.pi / 2 - tilt,
            "v": self.v[idx],
            "v0": self.desired_speed(idx),
            "z": self.z[idx],
            "leaving": to_exit < 0.5 * TWO_PI * self.r_out,
            "braking": self.brake_until[idx] > self.time,
            "truck": self.truck[idx],
            "length": self.length[idx],
            "width": np.where(self.truck[idx], TRUCK_WIDTH, self.cfg.car_width),
            "kind": self.kind[idx],  # KIND_NAMES codes
            "company": self.company[idx],  # -1: none
        }

    # ------------------------------------------------------------ dynamics
    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            self._step()

    def _step(self) -> None:
        dt = self.cfg.dt
        open_demand = self.open_demand
        idx = np.flatnonzero(self.alive)
        if idx.size:
            self._move(idx)
            self._exits()
            self._lane_changes(np.flatnonzero(self.alive))
        if open_demand:
            self._arrivals()
        if self.subject:
            self._top_up()
        self._entries()
        if not open_demand:
            # Last, so that after every step the population matches the density
            # setting (density changes, and leavers that found their queue full).
            self._sync_population()
        # Whole arrays: cheaper than masking, and _spawn resets free slots.
        self.lc_cool -= dt
        self.entry_cool -= 1
        self.lat += np.clip(self.lane - self.lat, -dt / LC_DURATION, dt / LC_DURATION)
        self.time += dt
        self.steps += 1
        if self.time >= self._next_prune:
            self._prune_logs()

    def _prune_logs(self) -> None:
        """Drop events older than LOG_HORIZON and log the entrances' counters (once per simulated second)."""
        self._next_prune = self.time + 1.0
        cutoff = self.time - LOG_HORIZON
        if self.n_arms:
            self.entrance_log.append((self.time, self.arm_arrived.copy(), self.arm_entered.copy(),
                                      self.queue_busy.copy()))
        for log in (self.detector_log, self.exit_log, self.emergency_log, self.stop_log, self.arrival_log,
                    self.entry_log, self.blocked_log, self.gave_up_log, self.entrance_log, self.conflict_log):
            while log and log[0][0] < cutoff:
                log.popleft()
        while self.lane_change_log and self.lane_change_log[0] < cutoff:
            self.lane_change_log.popleft()

    def _move(self, idx) -> None:
        cfg, dt = self.cfg, self.cfg.dt
        lead, gap = self._index_lanes(idx)
        v = self.v[idx]
        ld = lead[idx]
        g = gap[idx]
        v0 = self.target_speed(idx)
        self._observe_following(idx, v, ld, g)
        if self.controllers:
            self._controller_gaps(idx, v, ld)
        if self.smoothing:
            self._record_speeds(idx, v)
        acc_now = self._car_following(idx, v, v0, ld, g, lead, gap)
        if self.automated:
            a_idm = acc_now.copy()  # (an automated car's: the IDM towards its direct leader, its own T and s0)
            self._automated_command(idx, v, v0, ld, g, acc_now)
        self._make_room_to_leave(idx, v, v0, acc_now)
        if self.automated:
            self._automated_limits(idx, acc_now, a_idm)
        # The driver model asking for more than the brakes can give (on this
        # road): harsh braking, typically over-reaction after a late or delayed response.
        max_decel = self.eff.max_decel
        over_braking = acc_now < -max_decel

        # Reaction time: act on the acceleration decided ``delay`` steps ago (each driver their own).
        if self.alike:
            delay = int(round(min(cfg.reaction_time, MAX_REACTION) / dt))
        else:
            delay = self.delay[idx]
            if self.smoothing and self._smoothers_track():  # (a tracking smoother: none, _smoothers_track)
                delay = np.where(self.kind[idx] == SMOOTHER, 0, delay)
        self._hist_ptr = (self._hist_ptr + 1) % self._hist_len
        self.acc_hist[self._hist_ptr, idx] = acc_now
        acc = self.acc_hist[(self._hist_ptr - delay) % self._hist_len, idx].copy()

        braking = self.brake_until[idx] > self.time
        if cfg.driver_noise > 0:
            f = np.exp(-dt / NOISE_TAU)
            self.noise[idx] = f * self.noise[idx] + cfg.driver_noise * np.sqrt(1 - f * f) * \
                self.rng.standard_normal(idx.size)
            noise = self.noise[idx]
            if self.automated:  # automated vehicles drive without it (drawn all the same, then masked; _driven)
                noise = np.where(self._driven(self.kind[idx]), noise, 0.0)
            acc += noise
        # Physical braking limit; a car told to brake brakes hard, but no harder than the road allows.
        acc = np.where(braking, -min(PERTURB_DECEL, max_decel), np.maximum(acc, -max_decel))

        v_new = v + acc * dt
        stops = v_new < 0
        ds = np.where(stops, v * v / (2 * np.maximum(-acc, 1e-9)), v * dt + 0.5 * acc * dt * dt)
        ds = np.maximum(ds, 0.0)
        v_new = np.maximum(v_new, 0.0)

        # No overlaps: nobody may end the step closer than SAFE_MARGIN behind
        # where its leader ends the step. Iterate until nothing shrinks: moves
        # only decrease and are bounded by 0, so this settles chains of any length.
        planned = ds
        pos = np.empty(self.cap, dtype=np.int64)
        pos[idx] = np.arange(idx.size)
        ahead = pos[ld]  # every live car has a leader (itself if alone in its lane)
        for _ in range(idx.size):
            limited = np.minimum(ds, np.maximum(g + ds[ahead] - SAFE_MARGIN, 0.0))
            if np.array_equal(limited, ds):
                break
            ds = limited
        clamped = ds < planned - 1e-3
        v_new = np.where(clamped, np.minimum(v_new, ds / dt), v_new)
        self.a_now[idx] = (v_new - v) / dt  # the acceleration each car realised this step (_alloc)

        # Harsh braking ("emergency"): the driver model asks for more than
        # the braking limit, or the no-overlap constraint has to stop the car
        # harder than that. Counted per episode, which ends after EMERGENCY_HOLD s.
        emergency = over_braking | ((v - v_new) / dt > max_decel + 1e-6)
        new = emergency & (self.emergency_until[idx] < self.time)
        onset = int(new.sum())
        self.emergency_until[idx[emergency]] = self.time + EMERGENCY_HOLD
        if onset:
            self.n_emergency += onset
            self.emergency_log.append((self.time, onset))
            if self.automated:  # (observer: the automated vehicles' share of them, under their controllers)
                self.n_emergency_auto += int((new & (self._observed_kinds(self.kind[idx]) != HUMAN)).sum())

        # A stop: dropping below SLOW_FRACTION of the target speed after having
        # been above RESUME_FRACTION (so crawling along in dense flow, or
        # slowing to turn off at an exit, is not a stop).
        moving = self.moving[idx]
        stopped = moving & (v_new < SLOW_FRACTION * v0)
        self.moving[idx] = ~stopped & (moving | (v_new > RESUME_FRACTION * v0))
        n_stopped = int(stopped.sum())
        if n_stopped:
            self.n_stops += n_stopped
            self.stop_log.append((self.time, n_stopped))

        dtheta = ds / self.radii[self.lane[idx]]
        old = self.theta[idx]
        new = old + dtheta
        new[new >= TWO_PI] -= TWO_PI  # == % TWO_PI: a step is far shorter than a lap
        self.theta[idx] = new
        self.travel[idx] += dtheta
        self.v[idx] = v_new

        det = self.detector_angle
        before, after = old - det, new - det
        before[before < 0] += TWO_PI
        after[after < 0] += TWO_PI
        crossed = (dtheta > 0) & (before > after)
        for lane in self.lane[idx[crossed]]:
            self.detector_log.append((self.time, int(lane)))
        if self.n_arms:
            self._log_conflicts(idx, old, dtheta)

    def _log_conflicts(self, idx, old, dtheta) -> None:
        """Log the circulating vehicles that passed an entrance's entry point this step (its conflict point
        with the entering cars): ``conflict_log`` (time, arm, lane, uid, kind) and the counts ``arm_passed``
        (any lane), ``arm_passed_outer`` (the outer lane) and ``arm_passed_trucks``. Observer only: it reads
        the positions and draws nothing.

        A vehicle passes a point when the arc it drove this step, from just
        after where it was (``old``, aligned with ``idx``) to where it is now,
        reaches it, as at the detector line; so a car placed on an entry point
        (an entrant) does not pass that one. Leavers leave at the arm's exit,
        which comes before its entry point, so a circulating flow never counts
        the cars leaving there (a step is far shorter than the exit-to-entry arc).
        """
        ext = self._entry_ext
        start = np.searchsorted(ext, old, side="right")  # the first entry point ahead of each vehicle
        passed = np.searchsorted(ext, old + dtheta, side="right") - start  # how many it reached: 0, rarely 1
        hit = np.flatnonzero(passed)
        if hit.size == 0:
            return
        for k in range(int(passed[hit].max())):  # (more than one point in a step only with very long steps)
            sub = hit[passed[hit] > k]
            arms = self._entry_order[(start[sub] + k) % self.n_arms]
            slots = idx[sub]
            lanes = self.lane[slots]
            np.add.at(self.arm_passed, arms, 1)
            np.add.at(self.arm_passed_outer, arms[lanes == self.outer], 1)
            np.add.at(self.arm_passed_trucks, arms[self.truck[slots]], 1)
            self.conflict_log.extend(zip([self.time] * sub.size, arms.tolist(), lanes.tolist(),
                                         self.uid[slots].tolist(), self.kind[slots].tolist()))

    def _observe_following(self, idx, v, ld, g) -> None:
        """Observer only (reads the state at the start of the step, draws nothing): the surrogate safety counts
        and the exposure of each group. ``v``, ``ld``, ``g``: each car's speed, its leader's slot and the bumper
        gap to it (aligned with ``idx``, as in _move).

        Following exposure: a car whose direct leader (another vehicle) is within the Reaction distance (the
        range in which the model's drivers react to it), or closing on it at TTC <= TTC_CRITICAL whatever the
        distance, follows it for this step, counted per pair type (pair_type, follower first) in
        ``pair_follow_steps``. Among those, a follower closing on its leader with TTC = gap / (v - v_lead) <=
        TTC_CRITICAL adds a step to ``pair_ttc_steps`` (TET = these steps x dt, Minderhoud & Bovy 2001) and starts
        a TTC episode (``pair_ttc_episodes``) unless one is under way for it (``ttc_until``: a per-car hold of
        EMERGENCY_HOLD, as harsh braking): TTC is a property of the pair, not of what the drivers see (a short
        Reaction distance must not hide conflicts). Also the vehicle-steps on the ring of automated vehicles and of
        human drivers (``n_auto_steps``, ``n_human_steps``). A smoother counts as its driver's car until Smoother
        start (_observed_kinds)."""
        v_lead = self.v[ld]
        low = (ld != idx) & (v > v_lead) & (g <= TTC_CRITICAL * (v - v_lead))
        follows = ((ld != idx) & (g <= self.cfg.reaction_distance)) | low
        if self.automated:
            kind = self._observed_kinds(self.kind[idx])
            n_auto = int((kind != HUMAN).sum())
            self.n_auto_steps += n_auto
            self.n_human_steps += idx.size - n_auto
            pair = pair_type(kind, self._observed_kinds(self.kind[ld]))
            self.pair_follow_steps += np.bincount(pair[follows], minlength=len(PAIR_TYPES))
        else:  # (every pair HH)
            self.n_human_steps += idx.size
            pair = None
            self.pair_follow_steps[0] += int(follows.sum())
        if not low.any():
            return
        hit = np.flatnonzero(low)
        onset = hit[self.ttc_until[idx[hit]] < self.time]
        if pair is None:
            self.pair_ttc_steps[0] += hit.size
            self.pair_ttc_episodes[0] += onset.size
        else:
            self.pair_ttc_steps += np.bincount(pair[hit], minlength=len(PAIR_TYPES))
            self.pair_ttc_episodes += np.bincount(pair[onset], minlength=len(PAIR_TYPES))
        self.ttc_until[idx[hit]] = self.time + EMERGENCY_HOLD

    def must_move_out(self, idx) -> np.ndarray:
        """Leavers not yet on the outer lane that are inside their lane-change zone."""
        to_go = self.outer - self.lane[idx]
        to_exit = (self.exit_at[idx] - self.travel[idx]) * self.r_out
        return (to_go > 0) & (to_exit < to_go * LC_ZONE + 20.0)

    def _make_room_to_leave(self, idx, v, v0, acc) -> None:
        """Cooperative merging for leavers that must move outwards.

        Driving alongside the target lane's traffic at its speed, a leaver
        would never find a gap. So it follows the car just ahead in that lane
        (braking at most 1.5x comfortably), and the car just behind it in that
        lane eases off to let it in (at most comfortable braking). Updates
        ``acc`` (aligned with ``idx``) in place.

        In dense traffic leavers and yielding cars can end up holding each
        other at a standstill; a leaver held there by merging (its own lane
        would let it go) for LEAVER_PATIENCE seconds gives up its exit and
        goes round again, which breaks the deadlock.
        """
        if self.n_lanes < 2:
            return
        must = self.must_move_out(idx)
        if not must.any():
            self.stuck[idx] = 0.0
            return
        own = acc.copy()
        m = np.flatnonzero(must)
        b = self.eff.comfort_decel
        leavers = idx[m]
        nl, front, nf, back = self._neighbours(self.lane[leavers] + 1, self.theta[leavers],
                                               None if self.uniform_length else self.length[leavers])
        v_front = np.where(nl >= 0, self.v[np.maximum(nl, 0)], v[m])
        follow = self._idm(v[m], v0[m], front, v_front, leavers)
        acc[m] = np.minimum(acc[m], np.maximum(follow, -1.5 * b))
        has = nf >= 0
        if has.any():
            pos = np.empty(self.cap, dtype=np.int64)
            pos[idx] = np.arange(idx.size)
            f = pos[nf[has]]
            yield_acc = self._idm(v[f], v0[f], back[has], v[m][has], idx[f])
            np.minimum.at(acc, f, np.maximum(yield_acc, -b))

        held = must & (v < 0.5) & (own > 0) & (acc <= 0)
        self.stuck[idx] = np.where(held, self.stuck[idx] + self.cfg.dt, 0.0)
        give_up = idx[self.stuck[idx] > LEAVER_PATIENCE]
        if give_up.size:
            self.exit_at[give_up] += TWO_PI
            self.n_missed += give_up.size
            self.stuck[give_up] = 0.0

    def _exits(self) -> None:
        if self.n_arms == 0:
            return
        idx = np.flatnonzero(self.alive)
        due = idx[self.travel[idx] >= self.exit_at[idx]]
        if due.size == 0:
            return
        on_outer = self.lane[due] == self.outer
        missed = due[~on_outer]
        self.exit_at[missed] += TWO_PI  # not on the outer lane in time: go round again
        self.n_missed += missed.size
        leaving = due[on_outer]
        if leaving.size == 0:
            return
        arms = np.argmin(np.abs(((self.theta[leaving][:, None] - self.exit_angle[None, :]) + np.pi)
                                % TWO_PI - np.pi), axis=1)
        for slot, arm in zip(leaving, arms):
            self.exit_log.append((self.time, int(arm), self.time - self.born[slot]))
        self.alive[leaving] = False
        self.n_exited += leaving.size
        if self.open_demand:
            return  # leavers are gone for good; new cars come from _arrivals
        # Each leaver is replaced at a random entrance; if that queue is full the
        # replacement is slotted into the ring instead (by _sync_population).
        for arm in self.rng.integers(self.n_arms, size=leaving.size):
            if self.queue[arm] < MAX_QUEUE:  # (one that finds it full is created when it is slotted in)
                self.queue[arm] += 1
                self.queue_times[arm].append(self.time)
                self.queue_trucks[arm].append(bool(self._draw_trucks(1)[0]))
                self._queue_kinds(arm, self._draw_kinds(1))
                self.n_arrived += 1
                self.arm_arrived[arm] += 1
                self.arrival_log.append((self.time, int(arm)))

    def _arrivals(self) -> None:
        """Open mode: cars arrive at each entrance at random (Poisson, from the demand
        stream) and join its queue; those that find it full are turned away (blocked).
        Every arrival draws its kind (blocked ones too, so the kinds' draws never depend
        on the queues); trucks are drawn, as always, only for those that join."""
        if self.n_arms == 0 or self.cfg.entry_rate <= 0:
            return
        arrived = self.rng_demand.poisson(self.arrival_rates() * self.cfg.dt)
        if not arrived.any():
            return
        limit, now = int(self.cfg.max_queue), self.time
        for arm in np.flatnonzero(arrived).tolist():
            k = int(arrived[arm])
            join = min(k, max(limit - int(self.queue[arm]), 0))
            kinds = self._draw_kinds(k)
            self.queue[arm] += join
            self.queue_times[arm].extend([now] * join)
            self.queue_trucks[arm].extend(self._draw_trucks(join).tolist())
            self._queue_kinds(arm, kinds, join)
            self.arrival_log.extend([(now, arm)] * k)
            if k > join:
                self.n_blocked += k - join
                self.blocked_log.extend([(now, arm)] * (k - join))
        self.n_arrived += int(arrived.sum())
        self.arm_arrived += arrived

    def _top_up(self) -> None:
        """The capacity test (``subject``): fill the subject entrance's queue back up to SUBJECT_QUEUE cars,
        each joining now (its queue wait counts from here), a truck as often as an arrival is (from
        rng_drivers) and of a kind as an arrival is (from rng_kinds_test, a stream of its own: how many join
        follows the subject entries, so the n-th vehicle topped up and the n-th arrival at each other entrance
        draw the same uniforms whatever the share, and the kinds of both stay nested across shares). No
        arrival is drawn or counted: n_topped_up counts them instead."""
        k = SUBJECT_QUEUE - int(self.queue[SUBJECT_ARM])
        if k > 0:
            self.queue[SUBJECT_ARM] += k
            self.queue_times[SUBJECT_ARM].extend([self.time] * k)
            self.queue_trucks[SUBJECT_ARM].extend(self._draw_trucks(k).tolist())
            self._queue_kinds(SUBJECT_ARM, self._draw_kinds(k, self.rng_kinds_test))
            self.n_topped_up += k

    def _drop_queued(self, arm: int, k: int) -> None:
        """Take the ``k`` latest arrivals out of an entrance's queue: they give up
        (counted in n_gave_up and gave_up_log, so every arrival stays accounted for)."""
        k = int(k)
        if k > 0:
            self.queue[arm] -= k
            for _ in range(k):
                self.queue_times[arm].pop()
            for name in QUEUE_FIELDS:
                carried = getattr(self, name)[arm]
                while len(carried) > self.queue[arm]:  # (a queue edited by hand may hold fewer types)
                    carried.pop()
            self.n_gave_up += k
            self.gave_up_log.extend([(self.time, int(arm))] * k)

    def _lane_changes(self, idx) -> None:
        cfg = self.cfg
        if self.n_lanes < 2 or idx.size == 0:
            return
        ready = idx[(self.lc_cool[idx] <= 0) & (self.brake_until[idx] <= self.time)]
        if ready.size == 0:
            return
        to_exit = (self.exit_at[ready] - self.travel[ready]) * self.r_out
        mandatory = self.must_move_out(ready)
        # (Automated vehicles consider and make changes as drivers do, with the same rules and draws: *model*, no
        # lane-change behaviour of their own; but a smoother keeps its lane, drawn all the same, then masked.)
        considering = mandatory | (self.rng.random(ready.size) < cfg.lane_change_rate * cfg.dt)
        if self.smoothing:
            considering &= self.kind[ready] != SMOOTHER
        cars = ready[considering]
        if cars.size == 0:
            return
        mandatory, to_exit = mandatory[considering], to_exit[considering]

        lead, gap = self._index_lanes(idx)
        ld = lead[idx]
        follower = np.full(self.cap, -1, dtype=np.int64)
        others = ld != idx
        follower[ld[others]] = idx[others]

        # Every (car, direction) option at once: outwards for cars[k] at k, inwards at n + k.
        n = cars.size
        target = np.concatenate((self.lane[cars] + 1, self.lane[cars] - 1))
        allowed = (target >= 0) & (target < self.n_lanes)
        # Move inwards only when the exit is far enough away.
        allowed[n:] &= ~mandatory & (to_exit > (self.outer - target[n:]) * LC_ZONE + 60.0)
        opt = np.flatnonzero(allowed)
        if opt.size == 0:
            return
        k = opt % n
        c = cars[k]
        own_length = cfg.car_length if self.uniform_length else self.length[c]
        nl, front, nf, back = self._neighbours(target[opt], self.theta[c], own_length)
        has_f = nf >= 0
        f = np.maximum(nf, 0)
        # The car that was following us gets our old leader.
        o = follower[c]
        has_o = o >= 0
        os_ = np.maximum(o, 0)
        # Desired speeds and current accelerations of everyone involved.
        need = np.concatenate((c, f, os_))
        v0 = np.zeros(self.cap)
        v0[need] = self.target_speed(need)
        acc = np.zeros(self.cap)
        acc[need] = self._idm(self.v[need], v0[need], gap[need], self.v[lead[need]], need)

        vc = self.v[c]
        v_front = np.where(nl >= 0, self.v[np.maximum(nl, 0)], vc)
        # Each car judged by the time gap and margin it would keep behind its leader after the change: a CAV's
        # follow its leader (CACC in a string, else its ACC-fallback gap: _gaps_behind); everyone else keeps theirs.
        T_c, s0_c, pos_c = self._gaps_behind(c, nl)
        T_f, s0_f, _ = self._gaps_behind(f, np.where(has_f, c, -1), pos_c)
        T_o, s0_o, _ = self._gaps_behind(os_, np.where(has_o, lead[c], -1))
        # Judge the target lane with the speed this driver could do there
        # (its cornering cap differs from the current lane's).
        a_self = self._idm(vc, self.desired_speed(c, target[opt]), front, v_front, c, T_c, s0_c)
        a_f_new = np.where(has_f, self._idm(self.v[f], v0[f], back, vc, f, T_f, s0_f), 0.0)
        a_f_old = np.where(has_f, acc[f], 0.0)
        # Once we are gone, the old follower's gap reaches over our length to our old leader.
        a_o_new = np.where(has_o, self._idm(self.v[os_], v0[os_], gap[os_] + own_length + gap[c],
                                            self.v[lead[c]], os_, T_o, s0_o), 0.0)
        a_o_old = np.where(has_o, acc[os_], 0.0)
        forced = mandatory[k] & (opt < n)  # a leaver must move out
        # Braking a change may force on anyone: more for a leaver, never beyond what the road allows.
        b_safe = np.minimum(self.eff.comfort_decel * np.where(mandatory[k], 2.5, 1.5), self.eff.max_decel)
        # Room kept to the new leader and from the new follower: half the standstill gap of the car behind each
        # gap (the changer's ahead of it, the new follower's behind it), at least SAFE_MARGIN + 0.2 m.
        room_front = room_back = max(0.5 * cfg.min_gap, SAFE_MARGIN + 0.2)
        if self.automated:
            room_front = np.maximum(0.5 * (self.s0[c] if s0_c is None else s0_c), SAFE_MARGIN + 0.2)
            room_back = np.where(has_f, np.maximum(0.5 * (self.s0[f] if s0_f is None else s0_f), SAFE_MARGIN + 0.2),
                                 room_back)
        safe = (front >= room_front) & (back >= room_back) & (a_f_new >= -b_safe) & (a_self >= -b_safe)
        gain = (a_self - acc[c]) + cfg.politeness * ((a_f_new - a_f_old) + (a_o_new - a_o_old))
        if cfg.lane_change_motive == "random":
            wants = np.ones(opt.size, dtype=bool)
            s = np.empty(opt.size)
            # Same draw order as one draw per (direction, target lane) group.
            s[np.lexsort((target[opt], opt >= n))] = self.rng.random(opt.size)
        else:
            wants = gain > LC_THRESHOLD
            s = gain
        ok = safe & (wants | forced)
        score = np.full(2 * n, -np.inf)
        score[opt] = np.where(ok, np.where(forced, np.inf, s), -np.inf)
        new_lead = np.full(2 * n, -1, dtype=np.int64)
        new_foll = np.full(2 * n, -1, dtype=np.int64)
        new_back = np.full(2 * n, np.inf)  # [m] bumper gap to the new follower (observer: cut-ins)
        new_lead[opt], new_foll[opt], new_back[opt] = nl, nf, back

        t_out, s_out, l_out, f_out = target[:n], score[:n], new_lead[:n], new_foll[:n]
        t_in, s_in, l_in, f_in = target[n:], score[n:], new_lead[n:], new_foll[n:]
        seen_av = SEEN_AS_AV[self._observed_kinds(self.kind)] if self.automated else None  # (observer: cut-ins)
        go_out = s_out >= s_in
        best = np.where(go_out, s_out, s_in)
        chosen = np.flatnonzero(best > -np.inf)
        if chosen.size == 0:
            return
        involved: set[int] = set()
        for k in self.rng.permutation(chosen):
            c = int(cars[k])
            nl = int(l_out[k] if go_out[k] else l_in[k])
            nf = int(f_out[k] if go_out[k] else f_in[k])
            lane = int(t_out[k] if go_out[k] else t_in[k])
            group = {c, nl, nf, int(follower[c])} - {-1}
            if nl < 0:  # empty target lane: nothing links two entrants, so allow one per step
                group.add(-10 - lane)
            if group & involved:
                continue  # a neighbour is already changing lane this step
            involved |= group
            self.lane[c] = lane
            self.lc_cool[c] = LC_COOLDOWN
            self.n_lane_changes += 1
            self.lane_change_log.append(self.time)
            # (observer: a cut-in ahead of a recognisable AV, which now follows the changer within the Reaction
            # distance, the range drivers react in; a change far ahead of it is no cut-in)
            behind = new_back[k if go_out[k] else n + k]
            if self.automated and nf >= 0 and seen_av[nf] and behind <= cfg.reaction_distance:
                self.n_cutins_auto += 1

    def follow_up_steps(self) -> int:
        """Steps an entrance waits after letting a car in: the next car may go on the
        first step after ``follow_up`` (scaled by the road condition) has passed (1 s takes 11
        steps of 0.1 s)."""
        return int(np.floor(self.eff.follow_up / self.cfg.dt + 1e-9)) + 1

    def _entries(self) -> None:
        """Queued cars merge in where the gap allows, at most one per entrance per follow-up time."""
        if self.n_arms == 0 or not self.queue.any():
            return
        self.queue_busy += self.cfg.dt * (self.queue > 0)  # entrances with cars waiting this step
        ready = np.flatnonzero((self.queue > 0) & (self.entry_cool <= 0))
        if ready.size == 0:
            return
        # Only the outer lane's index is needed (the others are rebuilt before their next use).
        cars = np.flatnonzero(self.alive & (self.lane == self.outer))
        order = cars[np.argsort(self.theta[cars], kind="stable")]
        self._lanes[self.outer] = (order, self.theta[order])
        follow_up = self.follow_up_steps()
        arms = self.rng.permutation(ready)
        while arms.size:
            # Check the remaining arms together; after an entry, re-check the ones after it.
            ok, v_in = self._entry_ok(self.entry_angle[arms], self._head_trucks(arms), self._head_s0(arms))
            hits = np.flatnonzero(ok)
            if hits.size == 0:
                break
            k = hits[0]
            arm, angle = arms[k], self.entry_angle[arms[k]]
            trucks = self.queue_trucks[arm]
            truck = trucks.popleft() if trucks else False  # (a queue edited by hand may lack the types)
            if truck:
                self.arm_entered_trucks[arm] += 1
            slot = self._spawn(np.array([self.outer]), np.array([angle]), v_in[k:k + 1],
                               lat=np.array([self.outer + 1.0]), trucks=np.array([truck]), kinds=self._pop_kinds(arm))
            self.entered_at[slot] = self.time
            self.queue[arm] -= 1
            wait = self.time - self.queue_times[arm].popleft()  # first come, first served
            self.entry_wait_total += wait
            self.arm_entry_wait[arm] += wait
            self.entry_log.append((self.time, int(arm), wait))
            self.entry_cool[arm] = follow_up
            self.n_entered += 1
            self.arm_entered[arm] += 1
            # Keep the outer lane's sorted index current for the next arm.
            order, th = self._lanes[self.outer]
            pos = np.searchsorted(th, angle)
            self._lanes[self.outer] = (np.insert(order, pos, slot[0]), np.insert(th, pos, angle % TWO_PI))
            arms = arms[k + 1:]

    def _pop_kinds(self, arm: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(kind, company, ACC-fallback gap) of the vehicle at the head of an entrance's queue, as one-element
        arrays for _spawn, taken off its queue fields (a human driver where a queue edited by hand lacks them:
        the fills of an older state, ADDED)."""
        return tuple(np.array([fifo.popleft() if fifo else ADDED[name]], dtype=getattr(self, per_car).dtype)
                     for name, per_car in (("queue_kinds", "kind"), ("queue_companies", "company"),
                                           ("queue_acc_gaps", "acc_gap"))
                     for fifo in [getattr(self, name)[arm]])

    def _head_trucks(self, arms) -> np.ndarray | None:
        """Whether the vehicle at the head of each of these entrances' queues is a truck
        (None when every vehicle is a car)."""
        if self.uniform_length:
            return None
        return np.array([bool(self.queue_trucks[arm][0]) if self.queue_trucks[arm] else False for arm in arms])

    def _head_s0(self, arms) -> np.ndarray | None:
        """[m] The standstill gap of the vehicle at the head of each of these entrances' queues, by the kind its
        queue carries (None while no vehicle can be automated: every one keeps Standstill gap)."""
        if not self.automated:
            return None
        return self._kind_s0(np.array([self.queue_kinds[arm][0] if self.queue_kinds[arm] else HUMAN for arm in arms]))

    def _entry_ok(self, angles: np.ndarray, trucks: np.ndarray | None = None,
                  s0: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Whether a vehicle may enter the outer lane at each of ``angles`` (``trucks``: which are
        trucks; None = all cars), and at what speed. The room it needs ahead is its own standstill gap
        (``s0``; None = Standstill gap for all) + 1 m, and behind it the lag vehicle's own + 1 m: an entrant
        let into less than its own (a robotaxi's is 1.94 m longer) would brake to a stand at the entry point
        and hold up the next one (the entry rules by kind are S3's)."""
        cfg, eff = self.cfg, self.eff
        if trucks is None:
            length, v_in = cfg.car_length, np.full(angles.size, min(cfg.turn_speed, eff.mean_speed))
        else:
            length = np.where(trucks, cfg.truck_length, cfg.car_length)
            v_in = np.minimum(cfg.turn_speed, eff.mean_speed * np.where(trucks, TRUCK_SPEED_FACTOR, 1.0))
        own = cfg.min_gap if s0 is None else s0
        nl, front, nf, back = self._neighbours_in(self.outer, angles, length)
        if nl[0] < 0:  # empty outer lane: room for anything shorter than the lane
            return np.broadcast_to(length + own <= self.lane_len[self.outer], angles.shape).copy(), v_in
        vf = self.v[nf]
        lag = cfg.min_gap if s0 is None else self.s0[nf]
        follower_ok = self._idm(vf, self.desired_speed(nf), back, v_in, nf) >= -eff.comfort_decel
        ok = ((front >= own + 1.0) & (back >= lag + 1.0)
              & (back >= eff.entry_gap * vf) & follower_ok)
        return ok, np.where(front < 15.0, np.minimum(v_in, self.v[nl]), v_in)

    def _sync_population(self) -> None:
        """Closed mode: match the number of cars (on the ring and queued) to the density setting."""
        target = self.target_cars()
        n_ring = int(self.alive.sum())
        surplus = n_ring + int(self.queue.sum()) - target
        if surplus >= 0:
            self._pending.clear()  # nobody missing: nothing left to place
        if surplus > 0:
            for arm in np.argsort(-self.queue):
                take = min(self.queue[arm], surplus)
                self._drop_queued(arm, take)
                surplus -= take
            if surplus > 0:
                pool = self._removable()
                if pool.size:
                    self.alive[self.rng.choice(pool, size=min(surplus, pool.size), replace=False)] = False
        elif surplus < 0 and self.time >= self._next_fill:
            got, full = self._insert_into_gaps(-surplus)
            if got < -surplus:
                # Respacing is for a ring where no gap takes even a car (near jam density). A truck
                # that fits no gap yet waits for one instead: it is the next vehicle placed.
                if not full or (not self._respace_lanes(-surplus - got) and got == 0):
                    self._next_fill = self.time + 1.0  # retry in a second

    def _sync_ring(self) -> None:
        """Open mode, when Density is changed live (or the start left vehicles out): bring
        the cars on the ring to the density setting at once (queues untouched); after
        that arrivals and exits decide again. Trucks that fit no gap are placed by respacing the lanes (a
        bulk raise teleports cars anyway), so the vehicle mix holds; only vehicles
        that fit no lane at all (a full ring) are left out."""
        target = self.target_cars()
        surplus = int(self.alive.sum()) - target
        if surplus > 0:
            pool = self._removable()
            if pool.size:
                self.alive[self.rng.choice(pool, size=min(surplus, pool.size), replace=False)] = False
        while surplus < 0:  # every round adds at least one car, or stops
            got, _ = self._insert_into_gaps(-surplus)
            # Nothing fitted a gap: no gap takes a car, or only trucks are left to place.
            if got == 0 and not self._respace_lanes(-surplus):
                break
            surplus = int(self.alive.sum()) - target
        self._pending.clear()  # (once only: open mode keeps no vehicles waiting to be placed)

    def _insert_into_gaps(self, k: int) -> tuple[int, bool]:
        """Insert up to ``k`` vehicles into gaps; returns how many fitted, and whether
        the ring ran out of gaps that take a car.

        A few cars (leavers' replacements whose queue was full) go into random
        gaps that are big enough for the car behind not to slow down, so they
        do not systematically fill the emptiest stretch of road that a "do cars
        congregate?" measure looks for. Bulk density raises, and small ones
        when too few such gaps exist (dense traffic), go into the largest gaps.

        Gaps are ranked by their bumper gap plus a car's length (the distance
        between the centres of two cars). A new vehicle goes halfway along the
        bumper gap, leaving the same room ahead and behind; trucks need their
        extra length on top. A truck that fits no gap waits for one: it stays
        first in line for the next fill (``_pending``, as do vehicles left over
        when no gap takes a car), and the vehicles after it may still go.
        """
        cfg = self.cfg
        # A gap must leave the new car and its follower the lane-change safety room.
        min_room = max(0.5 * cfg.min_gap, SAFE_MARGIN + 0.2)
        fits = 2 * (cfg.car_length + min_room)
        trucks = self._fill_types(k)
        extra = np.where(trucks, cfg.truck_length - cfg.car_length, 0.0)  # [m] longer than a car
        idx = np.flatnonzero(self.alive)
        _, gap = self._index_lanes(idx)
        # Gaps: (-(bumper gap + car length), lane, angle of the vehicle behind the gap, its slot,
        # its extra length). Slot -2: a vehicle inserted here; -3: an empty lane.
        heap: list[tuple[float, int, float, int, float]] = []
        for lane, (order, th) in enumerate(self._lanes):
            if order.size == 0:
                heap.append((-self.lane_len[lane], lane, float(self.rng.uniform(0, TWO_PI)), -3, 0.0))
            else:
                heap.extend(zip(-(gap[order] + cfg.car_length), [lane] * order.size,
                                th.tolist(), order.tolist(), self._extra_length(order).tolist()))
        small = k <= SMALL_FILL
        if small:
            # Only gaps where the car behind need not slow down for the new vehicles (the longest of
            # them that fits anywhere: a truck that fits no gap waits, and need not hold up the cars).
            longest = 0.0
            if extra.any():
                largest = -min(entry[0] for entry in heap)
                longest = float(max((e for e in extra if fits + e <= largest), default=0.0))
            cand = [entry for entry in heap if -entry[0] >= fits + longest and not self._near_exit(entry)]
            behind = np.array([entry[3] for entry in cand], dtype=np.int64)
            keep = behind < 0
            if (~keep).any():  # one vectorised check for all the cars behind
                room = np.array([-0.5 * (entry[0] + longest) - cfg.car_length for entry in cand])[~keep]
                keep[~keep] = self.equilibrium_speed(room, self.desired_speed(behind[~keep]),
                                                     behind[~keep]) >= self.v[behind[~keep]]
            ok = [entry for entry, fine in zip(cand, keep) if fine]
            if len(ok) >= k:
                heap = [ok[i] for i in self.rng.choice(len(ok), size=k, replace=False)]
        heapq.heapify(heap)
        placed = np.zeros(k, dtype=bool)
        full = False  # no gap left that takes a car
        lanes, angles, followers, arcs = [], [], [], []
        for j, e in enumerate(extra):
            if not heap or -heap[0][0] < fits:
                full = True
                break
            if -heap[0][0] < fits + e:
                continue  # a truck that fits not even the largest gap waits for one
            neg_arc, lane, start, foll, foll_extra = heapq.heappop(heap)
            arc = -neg_arc
            placed[j] = True
            lanes.append(lane)
            if foll == -3:  # empty lane: the first vehicle's gap is the whole lane
                angles.append(start)
                followers.append(-2)
                arcs.append(arc - e)
                heapq.heappush(heap, (neg_arc + e, lane, start, -2, e))
                continue
            # Halfway along the bumper gap: (gap + follower's length) / 2 from the follower's centre.
            angle = (start + 0.5 * (arc + foll_extra) / self.radii[lane]) % TWO_PI
            angles.append(angle)
            followers.append(foll)
            arcs.append(0.5 * (arc - e))
            heapq.heappush(heap, (-0.5 * (arc - e), lane, start, foll, foll_extra))
            heapq.heappush(heap, (-0.5 * (arc - e), lane, angle, -2, e))
        self._pending = trucks[~placed].tolist()
        if not lanes:
            return 0, full
        slots = self._spawn(np.array(lanes), np.array(angles), np.zeros(len(lanes)), trucks=trucks[placed])
        self.inserted_at[slots] = self.time
        # arcs: bumper gap ahead of each new vehicle + a car's length.
        eq = self.equilibrium_speed(np.array(arcs) - cfg.car_length, self.desired_speed(slots), slots)
        foll = np.array(followers)
        v_f = np.where(foll >= 0, self.v[np.maximum(foll, 0)], eq)
        self.v[slots] = np.minimum(eq, v_f)
        # The car now directly behind each new car (its old follower, or an
        # earlier new car in a gap split twice) slows to suit its new gap, so
        # the insertion does not set off a man-made braking wave.
        idx = np.flatnonzero(self.alive)
        lead, gap = self._index_lanes(idx)
        behind = idx[np.isin(lead[idx], slots) & (lead[idx] != idx)]
        # A replacement slotted in now and then: the car behind brakes physically
        # (the driver model does the rest). A bulk density raise from the slider
        # teleports many cars at once: settle their followers at once instead
        # of setting off a man-made braking wave.
        self._cap_speed(behind, gap[behind], max_cut=self.eff.max_decel * cfg.dt if small else np.inf)
        return len(lanes), full

    def _near_exit(self, gap_entry: tuple[float, int, float, int, float]) -> bool:
        """Whether a car slotted into this gap (outer lane) would sit just before an exit."""
        neg_arc, lane, start, _, foll_extra = gap_entry
        if lane != self.outer or self.n_arms == 0:
            return False
        mid = start + 0.5 * (-neg_arc + foll_extra) / self.radii[lane]
        before = (self.exit_angle - mid) % TWO_PI * self.radii[lane]
        return bool((before < 15.0).any())

    def _cap_speed(self, slots, gap, max_cut: float = np.inf) -> None:
        """Slow ``slots`` to the steady-state speed for ``gap`` (by at most
        ``max_cut`` m/s; the driver model brakes the rest), without logging stops.

        Slowed cars are marked as touched by the density control, so a jam they
        head is counted as "other", not "following".
        """
        eq = self.equilibrium_speed(gap, self.desired_speed(slots), slots)
        if np.isfinite(max_cut):
            eq = np.maximum(eq, self.v[slots] - max_cut)
        slowed = slots[self.v[slots] > eq]
        self.v[slots] = np.minimum(self.v[slots], eq)
        self.acc_hist[:, slowed] = np.minimum(self.acc_hist[:, slowed], 0.0)
        self.a_now[slowed] = np.minimum(self.a_now[slowed], 0.0)
        self.moving[slowed] &= self.v[slowed] >= SLOW_FRACTION * self.target_speed(slowed)
        self.inserted_at[slowed] = self.time

    def _respace_lanes(self, k: int) -> bool:
        """Fallback for near-jam densities, where no single gap can take a car.

        Gives up to ``k`` extra cars to the lanes with spare room (one more each
        round, to the lane whose cars would keep the most space), then spaces
        each changed lane's cars evenly, keeping their order. Returns False if
        every lane is full. With trucks (the vehicles still waiting to be placed
        first) a lane has room for one more vehicle while its lengths and
        standstill gaps add up to no more than the lane, a truck no lane has
        room for keeps waiting, and the spacing evens out the bumper gaps.
        """
        cfg = self.cfg
        pitch = cfg.car_length + cfg.min_gap
        trucks = self._fill_types(k)
        idx = np.flatnonzero(self.alive)
        self._index_lanes(idx)
        count = np.array([order.size for order, _ in self._lanes])
        room = np.floor(self.lane_len / pitch).astype(int)
        used = np.array([self.length[order].sum() for order, _ in self._lanes]) + count * cfg.min_gap
        added = np.zeros(self.n_lanes, dtype=int)
        given: list[list[bool]] = [[] for _ in range(self.n_lanes)]  # truck or not, per new vehicle
        placed = np.zeros(k, dtype=bool)
        for j, truck in enumerate(trucks):
            # At most one new car per existing gap in a lane.
            if self.uniform_length:
                free = count + added < room
            else:
                free = used + (cfg.truck_length if truck else cfg.car_length) + cfg.min_gap <= self.lane_len
            spare = free & ((count == 0) | (added < count))
            if not spare.any():
                if truck:
                    continue  # no room for a truck: it waits, a car may still fit
                break
            lane = np.argmax(np.where(spare, self.lane_len / (count + added + 1), -1.0))
            added[lane] += 1
            used[lane] += (cfg.truck_length if truck else cfg.car_length) + cfg.min_gap
            given[lane].append(bool(truck))
            placed[j] = True
        self._pending = trucks[~placed].tolist()
        if not added.any():
            return False
        for lane in np.flatnonzero(added):
            order, th = self._lanes[lane]
            n_new = order.size + added[lane]
            step = TWO_PI / n_new
            new_trucks = np.array(given[lane])
            if order.size:
                # New vehicles go into the largest gaps (bumper to bumper), one per gap.
                gaps = np.diff(np.append(th, th[0] + TWO_PI))
                if not self.uniform_length:
                    length = self.length[order]
                    gaps = gaps * self.radii[lane] - 0.5 * (length + np.roll(length, -1))
                take = np.zeros(order.size, dtype=bool)
                take[np.argsort(-gaps)[:added[lane]]] = True
                new = self._spawn(np.full(added[lane], lane), th[take], np.zeros(added[lane]), trucks=new_trucks)
                sequence = np.empty(n_new, dtype=np.int64)
                at = np.arange(order.size) + np.cumsum(np.concatenate(([0], take[:-1])))
                sequence[at] = order
                sequence[at[take] + 1] = new
                old = self.theta[sequence].copy()
                extra = self._extra_length(sequence)
                self.theta[sequence] = (th[0] + step * np.arange(n_new)
                                        + self._even_shift(extra) / self.radii[lane]) % TWO_PI
                # Keep each car's distance to its exit tied to its new position.
                self.travel[sequence] += (self.theta[sequence] - old + np.pi) % TWO_PI - np.pi
            else:
                new = sequence = self._spawn(np.full(n_new, lane), np.zeros(n_new), np.zeros(n_new), trucks=new_trucks)
                extra = self._extra_length(sequence)
                self.theta[sequence] = (self.rng.uniform(0, TWO_PI) + step * np.arange(n_new)
                                        + self._even_shift(extra) / self.radii[lane]) % TWO_PI
                self._assign_routes(new)
            self.inserted_at[new] = self.time
            self.v[new] = np.inf  # capped to the steady-state speed below
            even_gap = step * self.radii[lane] - cfg.car_length - extra.mean()  # every bumper gap
            self._cap_speed(sequence, np.full(n_new, even_gap))
        return True
