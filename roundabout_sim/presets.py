"""Real-world presets, plausible random settings and Latin hypercube samples.

``PRESETS`` are settings taken from published experiments, design guides and
traffic counts: the ring-road jam experiments (Sugiyama 2008, Tadaki 2013,
Stern 2018), US single- and two-lane roundabouts (NCHRP Report 672, HCM 6) at
several demands and on wet and snowy roads, and the Place Charles de Gaulle in
Paris. Every value carries its source in a comment, or is marked *derived*
(arithmetic on sourced numbers), *assumption* (a judgement where no source
gives it) or *model* (forced by what the preset represents). ``apply`` turns a
preset into a ``SimConfig``: the preset's values on top of the defaults, so a
preset gives the same model whatever was set before (only the seed is kept).
Preset values keep their published precision even where it falls between a
slider's steps (the Sugiyama ring's 36.6 m radius and 95.7 veh/km, the US
single-lane 18.25 m radius, the Etoile's exit chance 0.17); the viewers show
them exactly.

``random_changes`` draws a plausible value for every parameter (or for the
given groups; the automated-vehicle and cooperation tabs only when named,
OPT_IN_GROUPS) from ``RANDOM_RANGES``: real-world distributions rather than the
whole slider ranges, linked where real roundabouts link them (the radius
follows the lane count, the demand the layout's capacity, ...). It is
reproducible by seed, and its values sit on the sliders' steps.
``latin_hypercube`` spreads sweep samples evenly over given ranges.

Three findings of the research behind these values (2026-10-25) and of the
model checks shape the presets:

* The model's ``entry_gap`` is not the HCM critical headway. The engine lets a
  car merge when the bumper gap to the outer-lane car behind is at least
  ``entry_gap`` times its speed (plus clearances), so HCM values used literally
  give about 40 % too little entry capacity. The design-guide presets use the
  model value that reproduces the HCM 6 capacity and give the literal one in
  their notes.
* The engine lets the next queued car enter on the first time step after
  ``follow_up``, so the presets set it one 0.1 s step below the HCM follow-up
  headway (2.5 for HCM 6's 2.6 s): queued cars then enter the HCM headway apart.
* With the default drivers the ring experiments mostly stay smooth. Every
  real-world preset uses ``DRIVERS`` (means of calibrated car-following
  studies), with which the rings form waves within minutes at the measured
  densities and the free-to-jam transition falls where it was measured.
"""

from __future__ import annotations

import math
import re
import textwrap
import zlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from numbers import Integral

import numpy as np

from roundabout_sim.config import PARAM_BY_NAME, PARAMS, SimConfig, coerce
from roundabout_sim.textfmt import plain, short_value

DEFAULTS = SimConfig()
GROUPS: tuple[str, ...] = tuple(dict.fromkeys(p.group for p in PARAMS))  # slider groups, in tab order
# Groups drawn only when named (random_changes, plausible_ranges; --random and --sample draw the others by
# default), so human-only random settings and designs stay what they were before automated vehicles (SPEC 0.4).
OPT_IN_GROUPS = ("AVs", "Coop")
DEFAULT_GROUPS: tuple[str, ...] = tuple(g for g in GROUPS if g not in OPT_IN_GROUPS)
FIELDS: tuple[str, ...] = tuple(f.name for f in fields(SimConfig))  # every SimConfig field, also those without a slider


# ------------------------------------------------------------------ sources
def _cite(citation: str, url: str = "") -> str:
    return f"{citation}. {url}" if url else citation


SUGIYAMA_2008 = _cite(
    "Sugiyama Y, Fukui M, Kikuchi M, Hasebe K, Nakayama A, Nishinari K, Tadaki S, Yukawa S (2008) Traffic jams "
    "without bottlenecks - experimental evidence for the physical mechanism of the formation of a jam. "
    "New J. Phys. 10, 033001 (pp.4-5, Figs. 2-4)",
    "https://ir.library.osaka-u.ac.jp/repo/ouka/all/93262/NewJPhys_10_033001.pdf")
TADAKI_2013 = _cite(
    "Tadaki S, Kikuchi M, Fukui M, Nakayama A, Nishinari K, Shibata A, Sugiyama Y, Yosida T, Yukawa S (2013) "
    "Phase transition in traffic jam experiment on a circuit. New J. Phys. 15, 103034 (pp.3-12, Table 1, "
    "Figs. 2-7, App. C)",
    "https://ir.library.osaka-u.ac.jp/repo/ouka/all/93261/NewJPhys_15_103034.pdf")
STERN_2018 = _cite(
    "Stern RE, Cui S, Delle Monache ML, Bhadani R, Bunting M, Churchill M, Hamilton N, Haulcy R, Pohlmann H, "
    "Wu F, Piccoli B, Seibold B, Sprinkle J, Work DB (2018) Dissipation of stop-and-go waves via control of "
    "autonomous vehicles: Field experiments. Transportation Research Part C 89, 205-221 (Secs. 2.1-4.2, "
    "Tables 1-3, App. 5.2)",
    "https://arxiv.org/abs/1705.01693")
NCHRP_1061 = _cite(
    "NCHRP Research Report 1061 (2023) Highway and Street Design Vehicles: An Update (Table 1 p.10, "
    "Table 18 p.38)",
    "https://nap.nationalacademies.org/read/27236/page/38")
ZHU_2018 = _cite(
    "Zhu M, Wang X, Tarko A, Fang S (2018) Modeling car-following behavior on urban expressways in Shanghai: "
    "A naturalistic driving study. Transportation Research Part C 93, 425-445 (Tables 2 and 4)",
    "https://arxiv.org/abs/1811.06395")
KESTING_2008 = _cite(
    "Kesting A, Treiber M (2008) Calibrating car-following models by using trajectory data: methodological "
    "study. Transportation Research Record 2088 (Table 1 and Discussion)",
    "https://arxiv.org/abs/0803.4063")
JOHANSSON_1971 = _cite(
    "Johansson G, Rumar K (1971) Drivers' brake reaction times. Human Factors 13(1), 23-27 (median 0.66 s, "
    "321 drivers, as reported in AASHTO Green Book Ch. 3)",
    "https://www.eng.mu.edu/drakopoa/courses/172/Proj1%20AutoCAD/GDHS%20Ch%203%20txt.pdf")
TREIBER_2017 = _cite(
    "Treiber M, Kesting A (2017) The Intelligent Driver Model with stochasticity - new insights into traffic "
    "flow oscillations. ISTTT22 / Transportation Research Procedia (Secs. 2-4)",
    "https://arxiv.org/abs/1708.06952")
NCHRP_672 = _cite(
    "Rodegerdts L et al. (2010) NCHRP Report 672 'Roundabouts: An Informational Guide', 2nd ed. (Exhibits 1-9, "
    "3-12, 3-14, 4-5, 4-9, 6-9, 6-47; Sec. 3.5.1, 6.3.1; Eqs. 4-3, 4-10, 6-1). Superseded by NCHRP Research "
    "Report 1043 (2023)",
    "https://www.aldot.gov/publications/Design/pdf/TrafficSafetyOp/NCHRP672Roundabouts.pdf")
FHWA_SA_15_070 = _cite(
    "Rodegerdts L et al. (2015) FHWA-SA-15-070 'Accelerating Roundabout Implementation in the United States, "
    "Vol. II: Assessment of Roundabout Capacity Models for the HCM' (Tables 1-2, 10-14, 20-21)",
    "https://rosap.ntl.bts.gov/view/dot/49387")
LOBAZA_2022 = _cite(
    "Lobaza MG (2022) Roundabout with a strong U-turn traffic. Romanian Journal of Transport Infrastructure "
    "11(1), Tables 2-3, doi 10.2478/rjti-2022-0003 (HCM 6 capacity coefficients)",
    "https://reference-global.com/download/article/10.2478/rjti-2022-0003.pdf")
WISDOT_2020 = _cite(
    "Olsson, Campbell, Sternke / TADI (2020) Statewide Roundabout Traffic Operations Analysis, for WisDOT "
    "(Tables 6-8, 13-19)",
    "https://wisconsindot.gov/dtsdManuals/traffic-ops/programs/analysis/roundabout-analysis.pdf")
AKCELIK_2022 = _cite(
    "Akcelik R (prep.) / Besley (presenter) (2022) HCM 6 Extended Roundabout Capacity Model in SIDRA "
    "INTERSECTION, TRB 6th Int. Roundabout Conf., slide 6 (HCM 6 model unchanged in HCM 7)",
    "https://sidrasolutions.com/media/518/download")
GIUFFRE_2016 = _cite(
    "Giuffre O, Grana A, Tumminello ML (2016) Methodological frontier in operational analysis for "
    "roundabouts: a review. Frontiers in Built Environment 2:28",
    "https://doi.org/10.3389/fbuil.2016.00028")
FHWA_HPMS = _cite(
    "FHWA HPMS Field Manual, Appendix N 'Procedures for Estimating Highway Capacity' (PHF 0.92 urban, "
    "0.88 rural)",
    "https://www.fhwa.dot.gov/ohim/hpmsmanl/appn6.cfm")
FHWA_VM1 = _cite(
    "FHWA Highway Statistics 2022, Table VM-1 (vehicle-miles by vehicle type; shares computed)",
    "https://www.fhwa.dot.gov/policyinformation/statistics/2022/vm1.cfm")
DFT_2024 = _cite(
    "DfT (2025) Road traffic estimates in Great Britain, 2024 (traffic by road type; shares computed)",
    "https://www.gov.uk/government/statistics/road-traffic-estimates-in-great-britain-2024")
SIVAK_1982 = _cite(
    "Sivak M, Olson PL, Farmer KM (1982) Radar-measured reaction times of unalerted drivers to brake signals. "
    "Perceptual and Motor Skills 55(2), 594",
    "https://trid.trb.org/View/204069")
LEE_2018 = _cite(
    "Lee D, Hwang S, Ka E, Lee C (2018) Evaluation of the rain effects on gap acceptance behavior at "
    "roundabouts by a logit model. Journal of Advanced Transportation 2726732",
    "https://pure.uos.ac.kr/en/publications/evaluation-of-the-rain-effects-on-gap-acceptance-behavior-at-roun/")
RAKHA_2010 = _cite(
    "Rakha H, Zohdy I, Park S, Krechmer D (2010) Microscopic Analysis of Traffic Flow in Inclement Weather - "
    "Part 2, FHWA-JPO-11-020 (Tables 2.3, 3.5, 4.1, 4.10)",
    "https://rosap.ntl.bts.gov/view/dot/4030/dot_4030_DS1.pdf")
RAKHA_2009 = _cite(
    "Rakha H, Krechmer D, Cordahi G, Zohdy I, Sadek S, Arafeh M (2009) Microscopic Analysis of Traffic Flow "
    "in Inclement Weather, FHWA-JPO-09-066 (Table 4.8, Eq. 5.2, Table 5.4)",
    "https://rosap.ntl.bts.gov/view/dot/4003/dot_4003_DS1.pdf")
SHRP2_L08 = _cite(
    "SHRP 2 L08 / HCM 6th ed. weather adjustment factors, as reproduced in FHWA-HOP-13-042 Appendix A "
    "Tables 33-34",
    "https://ops.fhwa.dot.gov/publications/fhwahop13042/appa.htm")
HRANAC_2006 = _cite(
    "Hranac R, Sterzin E, Krechmer D, Rakha H, Farzaneh M (2006) Empirical Studies on Traffic Flow in "
    "Inclement Weather. FHWA-HOP-07-073 (Table ES.1, Fig. ES.3)",
    "https://ops.fhwa.dot.gov/publications/weatherempirical/sect0.htm")
WALLMAN_2001 = _cite(
    "Wallman C-G, Astrom H (2001) Friction measurement methods and the correlation between road friction "
    "and traffic safety: a literature review. VTI meddelande 911A (pp.26-37)",
    "https://www.diva-portal.org/smash/get/diva2:673366/FULLTEXT01.pdf")
MUNEHIRO_2017 = _cite(
    "Munehiro K, Takada T, Sato S, Takamoto S, Ishida T (2017) Issues in the planning and management of "
    "roundabouts in cold, snowy regions. TRB 96th Annual Meeting, paper 17-00960",
    "https://trid.trb.org/view/1437414")
PERRIN_2000 = _cite(
    "Martin PT, Perrin J, Hansen B, Quintana I (2000) Inclement Weather Signal Timings, MPC-01-120 "
    "(Tables 2.3, 4.2-4.6, pp.20-21)",
    "https://www.ugpti.org/resources/reports/downloads/mpc01-120.pdf")
SADEK_2004 = _cite(
    "Sadek AW, Agbolosu-Amison SJ (2004) Validating traffic simulation models to inclement weather travel "
    "conditions, NETCR47 (Table 13)",
    "https://onlinepubs.trb.org/onlinepubs/trispdfs/00983592.pdf")
FHWA_RWM = _cite(
    "FHWA Road Weather Management Program, 'How do weather events impact roads?'",
    "https://ops.fhwa.dot.gov/weather/q1_roadimpact.htm")
SUN_2026 = _cite(
    "Sun Y, Li W, Yang M, Zhang X (2026) Parameter calibration method for car-following models under snowy "
    "weather conditions. PLoS One 21(6): e0350550 (Tables 1, 3)",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC13249222/")
KAGEYAMA_2013 = _cite(
    "Kageyama H, Munehiro K, Ishida T (2013) Experiments to determine roundabout safety in consideration of "
    "driver behavior. Proc. EASTS Vol. 9",
    "https://easts.info/on-line/proceedings/vol9/PDF/P346.pdf")
WIKI_ETOILE = _cite(
    "Wikipedia FR 'Place Charles-de-Gaulle (Paris)' (12 avenues, 241 m place diameter)",
    "https://fr.wikipedia.org/wiki/Place_Charles-de-Gaulle_(Paris)")
OSM_ETOILE = _cite(
    "OpenStreetMap (ODbL) via Overpass, own circle fits 2026-10-25: ring ways (junction=circular, r 68.7 m), "
    "outer kerb way 588482470 (r 83.2 m), chain ring way 1356427943 (r 56.5 m), avenue bearings and oneway "
    "tags",
    "https://www.openstreetmap.org/way/184551793")
PARIS_COUNTS = _cite(
    "Ville de Paris open data 'Comptage routier - donnees trafic issues des capteurs permanents', own "
    "aggregation of arcs entering Pl_Charles_de_Gaulle, Sep 2025-Oct 2026",
    "https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents/")
LECUYER_2024 = _cite(
    "Lecuyer C (2024-03-18) 'Des embouteillages font leur apparition sur le rond-point de l'Etoile' (12 to 8 "
    "lanes; city press file of 11 May 2022)",
    "https://www.catherinelecuyer.fr/post/des-embouteillages-font-leur-apparition-sur-le-rond-point-de-etoile")
DEVISPERMIS_2026 = _cite(
    "DevisPermis (2026) Place de l'Etoile driving-test guide (entering vehicles have priority)",
    "https://devispermis.fr/blog/place-etoile-paris-rond-point-examen-permis-2026")
PARIS_30KMH = _cite(
    "Ville de Paris, generalisation of the 30 km/h limit (30 Aug 2021; Champs-Elysees, Foch and Grande Armee "
    "kept at 50 km/h)",
    "https://www.paris.fr/pages/generalisation-de-la-vitesse-a-30-km-h-les-parisiens-ont-donne-leur-avis-16967")
MODEL_CHECKS = ("Check runs of this model: the research for the presets (2026-10-25: entry capacity against "
                "HCM 6, ring wave onsets) and this version (the figures in the notes)")

DRIVER_SOURCES = (ZHU_2018, KESTING_2008, JOHANSSON_1971, TREIBER_2017)


# ------------------------------------------------------------------ presets
@dataclass(frozen=True)
class Preset:
    """A named set of settings from a real site or experiment.

    ``changes`` maps parameter names to values (everything else keeps its
    default); treat it as read-only, :func:`apply` copies it. ``sources`` are
    citations with URLs, the main one first, and ``cite`` names the main ones
    briefly for the viewer's help box; ``notes`` hold the caveats,
    validation targets and model checks. "This version" means runs of the
    current engine: for the open presets seeds 0-15, 20 min measured after 5
    min of warm-up, given as the median and the range across the seeds, as
    ``experiments sweep --preset NAME --x entry_rate=RATE --seeds 16 --warmup
    300 --measure 1200`` gives them; the rings as stated in their notes.
    """
    name: str
    title: str  # at most 24 characters (the viewer's preset selector)
    description: str  # one or two sentences for the viewer's help box
    sources: tuple[str, ...]
    changes: dict = field(hash=False)
    notes: str = ""
    cite: str = ""  # the main sources in a few words ("Sugiyama et al. 2008, New J. Phys. 10, 033001")


# The drivers of every real-world preset. With the defaults (T 1.2 s, s0 2 m, reaction 0.4 s, noise 0.35)
# the ring experiments mostly stay smooth; with these they form waves within 1-6 minutes at the measured
# densities. Max acceleration (1.2) and comfortable braking (2.0) keep the defaults: 1.2 lies between the
# Zhu-calibrated 0.84 and the Treiber-Kesting city value 1.5, 2.0 is the Treiber-Kesting city value.
DRIVERS = {
    "time_headway": 0.9,  # Zhu 2018: calibrated IDM time gap, mean 0.878 s (42 drivers)
    "min_gap": 1.5,  # Zhu 2018: calibrated s0 mean 1.555 m; Kesting & Treiber 2008: measured 1.39-1.64 m
    # assumption: between the median reactions 0.50 s (Zhu 2018) and 0.66 s (Johansson & Rumar 1971);
    # puts the model's closed-ring jam threshold between Tadaki's 80 and 89 veh/km (0.4: no waves)
    "reaction_time": 0.6,
    # derived: Treiber & Kesting 2017's noise intensity Q = 0.32 m2/s3 (fits their platoon experiment)
    # is 0.28 m/s2 with the engine's NOISE_TAU = 2 s, rounded to the 0.05 step
    "driver_noise": 0.3,
}
DRIVERS_NOTE = ("Drivers: the calibrated real-world driver set (presets.DRIVERS: time gap 0.9 s, standstill gap "
                "1.5 m, reaction 0.6 s, noise 0.3 m/s2); with the default drivers the rings mostly stay smooth.")

# A typical US single-lane roundabout; the us_single_lane_* presets vary its demand and road surface.
_US_SINGLE_LANE = {
    "n_lanes": 1,  # one entering and one circulating lane (NCHRP 672 Exhibit 1-9)
    # derived: inscribed circle 40 m (NCHRP 672 Sec. 6.3.1: 36-43 m is the usual start; a WB-67 truck needs
    # 40-46 m): 40 / 2 - 0.5 x 3.5 m lane = 18.25 m
    "radius": 18.25,
    "traffic_mode": "open",  # model: approach traffic arrives at random and leaves
    # assumption: 432 veh/h per approach, in the 'moderate' 400-500 veh/h band derived from NCHRP 672
    # Exhibit 1-9 (its single-lane service limit, 25,000 veh/day x K 0.09-0.10 / 4 legs, is 560-625)
    "entry_rate": 0.12,
    # assumption: the main road brings 60 % of the demand, a minor-street share of 0.40 inside the
    # 0.33-0.50 that NCHRP 672 Sec. 3.5.1 assumes (with 4 arms the two side roads together get (1 - x) / 2,
    # each (1 - x) / 4)
    "demand_imbalance": 0.2,
    # assumption: 2 / n_arms, a mean trip of 2 exits (straight on), so the flow past each entry equals
    # its entering flow; one constant chance cannot match real turning splits (NCHRP 672 Exhibit 3-12)
    "exit_prob": 0.5,
    # derived: the model equivalent of the HCM 6 critical headway 4.98 s (A 1380, B 1.02e-3): 2.5 s gives
    # about 660 veh/h per entrance at saturation against HCM 6's 685 (this version, seeds 0-3: 634-679;
    # the research: 643 with follow-up 2.6), the literal 5.0 s only about 400
    "entry_gap": 2.5,
    # derived: HCM 6 single-lane follow-up 2.609 s (3600 / 1380; FHWA-SA-15-070 field mean 2.6 s) less one
    # 0.1 s time step: the engine lets the next car go on the first step after the follow-up, so 2.5 gives
    # entries 2.6 s apart (measured at zero circulating flow; 2.6 would give 2.7 s)
    "follow_up": 2.5,
    "mean_speed": 11.0,  # NCHRP 672 Exhibit 6-47: 25 mph (11.1 m/s) maximum single-lane entry design speed
    "lateral_accel": 3.0,  # derived: NCHRP 672 Eq. 6-1 gives 7.47 m/s at R 18.25 m, v^2 / R = 3.06 m/s2
    "density": 30.0,  # assumption: start only; the density this layout settles at (30.2 veh/km/lane)
    **DRIVERS,
}
_US_SOURCES = (NCHRP_672, FHWA_SA_15_070, LOBAZA_2022, WISDOT_2020, AKCELIK_2022, *DRIVER_SOURCES, MODEL_CHECKS)
_ENTRY_GAP_NOTE = (
    "Entry gap is calibrated to this model, not the literal HCM value: the engine lets a car merge when the "
    "bumper gap to the outer-lane car behind is at least Entry gap x its speed (plus clearances), so the HCM "
    "critical headway used literally gives about 40 % too little entry capacity. {literal} Capacity drops "
    "steeply once Entry gap passes the Follow-up time. Follow-up time is set one 0.1 s time step below the "
    "HCM follow-up headway, as the engine lets the next car go on the first step after it: queued cars then "
    "enter the HCM headway apart (measured at zero circulating flow).")

PRESETS: dict[str, Preset] = {p.name: p for p in (
    Preset(
        "default", "Model defaults",
        "The model's own defaults: a 3-lane ring of radius 80 m with 4 entrances, held at 50 veh/km/lane. "
        "Not tied to a real site.",
        ("The model's own defaults (config.SimConfig), not tied to a real site",),
        {},
        "Kept so the viewer and the command line can always go back to the model's own settings "
        "(bit-identical to a run without a preset)."),

    # Sugiyama et al. 2008, the first phantom-jam experiment on a circuit (Nagoya, Japan).
    Preset(
        "sugiyama2008", "Sugiyama 2008 ring",
        "22 cars on a 230 m single-lane circuit (Japan, 2008): a jam forms with no bottleneck and moves "
        "backward at about 20 km/h while free cars drive at 40 km/h.",
        (SUGIYAMA_2008, *DRIVER_SOURCES, MODEL_CHECKS),
        {
            "n_lanes": 1,  # single-lane circuit
            "n_arms": 0,  # closed circuit: no entries or exits
            "radius": 36.6,  # derived: circumference 230 m (p.4, Fig. 2) / (2 pi) = 36.61 m
            "density": 95.7,  # derived: 22 cars / 0.230 km = 95.65 veh/km (the engine rounds to 22 cars)
            "mean_speed": 11.0,  # cars outside the jam drove at about 40 km/h = 11.1 m/s (p.5)
            "lane_change_rate": 0.0,  # model: one lane, so no lane changes
            **DRIVERS,
        },
        "Validation targets: cars in the jam stop completely; the jam moves backward at about 20 km/h "
        "(5.6 m/s); a 5-car jam shows in the 3-minute snapshot (a snapshot time, not a measured onset). "
        "Check of the research and this version (seeds 1-3, 600 s; a wave = speed std across cars above 2.5 "
        "m/s, Stern's detector): waves at 63, 156 and 339 s, then speed std 2.6-2.8 m/s at a mean speed of "
        "3.2-3.4 m/s. Car length keeps the 4 m default (assumption: the paper gives no vehicle "
        "size). The cornering cap sqrt(4.0 x 36.6) = 12.1 m/s stays above the 11.1 m/s free speed; keep "
        "Cornering limit >= 3.4 (the drivers' own 11.1^2 / 36.6). " + DRIVERS_NOTE,
        cite="Sugiyama et al. 2008, New J. Phys. 10, 033001"),

    # Tadaki et al. 2013, the same group's larger circuit in the Nagoya Dome with a sweep of car numbers.
    Preset(
        "tadaki2013", "Tadaki 2013 dome ring",
        "30 identical cars on the 314 m Nagoya Dome circuit (radius 50 m, 95.5 veh/km) give jammed flow. "
        "Sweep Density from 32 to 108 to find the free-to-jam transition, measured between 80 and 89 veh/km.",
        (TADAKI_2013, *DRIVER_SOURCES, MODEL_CHECKS),
        {
            "n_lanes": 1,  # single-lane circuit
            "n_arms": 0,  # closed circuit: no entries or exits
            "radius": 50.0,  # '50 m radius (314 m circumference) circuit' (Figs. 2-3)
            # derived: session 2030, 30 cars / 0.314 km = 95.5 veh/km, the run whose jam cluster moved
            # backward at 20 km/h (p.6, Fig. 4)
            "density": 95.5,
            "car_length": 3.9,  # all cars were Toyota Vitz, 3.885 m long (p.5)
            "mean_speed": 11.0,  # free-flow fit of the fundamental diagram: about 40 km/h (p.10, Fig. 7a)
            "lane_change_rate": 0.0,  # model: one lane, so no lane changes
            **DRIVERS,
        },
        "Cars = Density x 0.314: 10 -> 31.8, 20 -> 63.7, 25 -> 79.6, 28 -> 89.2, 30 -> 95.5, 34 -> 108.3 "
        "veh/km. Measured: free flow up to 25 cars, jams from 28 (critical density between 80 and 89 veh/km), "
        "both at 28-30 cars (metastable); jam clusters move backward at about 20 km/h. Do not use the 40-car "
        "(127 veh/km) 'homogeneous again' run as a target: a cautious first-day session left out of the "
        "analysis. Research check (3 seeds, 600 s): 20 and 25 cars stay free, 28-34 cars form waves (after "
        "55-339 s), so the model's transition falls where it was measured; its jammed mean speeds (3.4 m/s at "
        "30 cars) are low. This version: 30 cars form waves at 83, 55 and 89 s (seeds 1-3); 20 and 25 cars "
        "keep the speed std below 1.4 m/s (seeds 1-5, 300 s). " + DRIVERS_NOTE,
        cite="Tadaki et al. 2013, New J. Phys. 15, 103034"),

    # Stern et al. 2018, the US ring experiment (Tucson, Arizona) on which one automated car damped the waves.
    Preset(
        "stern2018", "Stern 2018 ring",
        "21 human-driven cars on a 260 m single-lane track in the US (radius 41.4 m, 80.8 veh/km): stop-and-go "
        "waves appeared 55-79 s after the start (161 s in a run with 22 cars).",
        (STERN_2018, NCHRP_1061, TADAKI_2013, *DRIVER_SOURCES, MODEL_CHECKS),
        {
            "n_lanes": 1,  # single-lane track
            "n_arms": 0,  # closed circuit: no entries or exits
            "radius": 41.4,  # 'radius 41.4 meters to the center of the lane (260 meter circumference)' (Sec. 2.1)
            # derived: 21 cars (Experiments A and B) / 0.260 km = 80.8 veh/km (Experiment C: 22 cars, 84.6)
            "density": 80.8,
            # assumption: not reported (the ring was enlarged for the bigger US cars); mean length of 2019 US
            # passenger models 4.86 m (NCHRP RR 1061 Table 18)
            "car_length": 4.9,
            "lane_change_rate": 0.0,  # model: one lane, so no lane changes
            **DRIVERS,
            # The drivers, calibrated to the human ring with waves (Stern Table 3: 1,755-1,828 veh/h, speed SD
            # 2.36-3.85 m/s; first waves 55-161 s, Tables 1, 2, 4) on top of DRIVERS, whose ring jams fully at
            # about 1,250 veh/h (see the notes). Each value lies within its calibration studies' spread; the
            # set sits near the ring's stability boundary, so every value matters (notes).
            # assumption: not reported (mean speed 6.3 m/s with waves, 7.2 smoothed); above the ring's cornering
            # cap sqrt(4.0 x 41.4) = 12.9 m/s (the default Cornering limit), so most drivers' free speed is that
            # cap; at 11 (the Japanese rings' 40 km/h) waves form in 8 % of the runs, at 13 in 79 % (this version)
            "mean_speed": 14.0,
            # Zhu 2018: calibrated IDM time gap 0.878 +/- 0.334 s; assumption: below the mean, as the drivers
            # were asked to catch up with the car ahead (Sec. 2.1)
            "time_headway": 0.8,
            # Zhu 2018: calibrated s0 1.555 +/- 0.96 m (Kesting & Treiber 2008: measured 1.39-1.64 m);
            # assumption: below the mean, drivers closing gaps (Sec. 2.1)
            "min_gap": 1.25,
            # assumption: the low end of measured car-following reaction times (Zhu 2018 GHR median 0.50 s;
            # RANDOM_RANGES 0.3-1.2 s), alert drivers on a closed track; 0.4 already gives full stop-and-go at
            # about 1,550 veh/h (this version)
            "reaction_time": 0.3,
            # derived: 0.4 m/s2 with the engine's NOISE_TAU = 2 s is a noise intensity Q = 2 x 0.4^2 x 2 = 0.64
            # m2/s3, twice Treiber & Kesting 2017's platoon fit 0.32 (assumption: this ring's drivers vary more)
            "driver_noise": 0.4,
            # Kesting & Treiber 2008: calibrated 1.52-1.58 m/s2 (Table 1); 1.5 in town (Treiber & Kesting)
            "max_accel": 1.5,
            # Kesting & Treiber 2008: calibrated 0.61-3.73 m/s2 (Table 1); comfortable braking 0.27 g = 2.65 m/s2
            # (Garber & Hoel)
            "comfort_decel": 2.5,
            # assumption: the plain IDM, only the car in front (drivers were asked to close any widening gap to
            # the vehicle ahead, Sec. 2.1); drivers anticipate 1-7 vehicles (Treiber, Kesting & Helbing 2006)
            "anticipation": 1,
        },
        "Drivers were told to drive as in rush hour, catching up with the car ahead; they started evenly spaced "
        "from a standstill and drove 7-10 minutes. Targets: first waves at 79 and 55 s in Experiments A and B, "
        "with the preset's 21 cars (one of them the CAT Vehicle, driven by a human until its controller was "
        "switched on); Experiment C had 22 cars (84.6 veh/km) and its first wave at 161 s. With waves (Table 3, "
        "the first interval after they start): speed SD over all cars and time 3.31 / 2.36 / 3.85 m/s, 1,827 / "
        "1,828 / 1,755 veh/h (mean speed 6.3 / 6.3 / 5.8 m/s); the flow hardly fell when the waves came in A and B "
        "(1,809 -> 1,827 and 1,665 -> 1,828 veh/h; C 2,160 -> 1,755). Drivers calibrated to that (C1): the time "
        "gap, standstill gap, reaction time, noise, acceleration, comfortable braking, anticipation and desired "
        "speed differ from DRIVERS (sources by each value). This version (seeds 0-31, 300 s warm-up + 600 s): "
        "median 1,781 veh/h (1,627-1,889) at 6.13 m/s, speed SD 2.55 m/s, first wave (speed std across cars above "
        "2.5 m/s for 5 s) at 116 s, waves in 29 of the 32 runs by 300 s, moving backward at 4.6-5.1 m/s (Sugiyama "
        "2008: about 5.6), a car below 0.5 m/s in 8 % of the samples, 0.4 harsh brakings a minute. Mechanism: "
        "these drivers are string-stable on this ring (one car braking for 2 s without noise fades out), so the "
        "waves are noise-driven oscillations that rarely stop anyone (Treiber & Kesting 2017). With DRIVERS alone "
        "the ring is string-unstable: the same braking grows into a lasting jam, cars stand in 76 % of the "
        "samples, a stopped car restarts only once its gap has opened and after its 0.6 s reaction, and the jam "
        "lets out only about one car per 2.9 s: 1,250 veh/h (1,150-1,420; seeds 0-11) at 4.3 m/s, about 30 % below "
        "Stern and below the 1,770 veh/h the same drivers keep without waves (so do the other ring presets: "
        "Sugiyama 1,180 veh/h, waves at 3.4 m/s). Near the stability boundary every value matters (seeds 0-23): "
        "anticipation 2 or desired speed 11 keep the ring smooth (waves in 0-8 % of the runs); comfortable braking "
        "2.0, standstill gap 1.0 m or noise 0.3 leave waves in 33-58 % of them; reaction 0.4 s, acceleration 1.2 "
        "or standstill gap 1.5 m bring back full stop-and-go at 1,370-1,550 veh/h (a car standing in 71-92 % of "
        "the samples).",
        cite="Stern et al. 2018, Transp. Res. C 89, 205-221"),

    # A typical US single-lane roundabout, NCHRP Report 672 geometry and HCM 6 gap acceptance.
    Preset(
        "us_single_lane", "US single-lane (HCM 6)",
        "Typical US single-lane roundabout (NCHRP 672: 40 m across, 4 legs) at a moderate 432 veh/h per "
        "approach. Follow-up 2.5 s (entries 2.6 s apart) and entry gap 2.5 s reproduce HCM 6 capacity in this "
        "model (HCM: follow-up 2.6 s, critical headway 5.0 s).",
        _US_SOURCES,
        dict(_US_SINGLE_LANE),
        _ENTRY_GAP_NOTE.format(literal="Literal HCM 6 values: critical headway 5.0 s, follow-up 2.6 s (2.5 s "
                                       "here gives about 660 veh/h per entrance at saturation against HCM 6's 685, "
                                       "5.0 s only about 400); use Entry gap 5.0 only if the entry rule is "
                                       "changed to HCM semantics.") +
        " HCM 6 check: the main approach (518 veh/h) against about 400 veh/h circulating has capacity 918 "
        "veh/h, v/c 0.56 (LOS A-B). This version: all of the demand entered (434 veh/h per entrance), median "
        "queue wait 4.8 s (3.4-6.7 s, LOS A), longest queue 7 cars (5-10). Cars only, so throughput compares "
        "with HCM pc/h directly. Geometry: one 3.5 m model lane; real circulatory roadways are 4.9-6.1 m wide. "
        "NCHRP 672 is superseded by NCHRP RR 1043 (2023); the HCM 6 capacity model is unchanged in HCM 7. "
        + DRIVERS_NOTE,
        cite="NCHRP Report 672; HCM 6 (FHWA-SA-15-070)"),

    Preset(
        "us_single_lane_offpeak", "US single-lane off-peak",
        "The US single-lane roundabout at off-peak demand of 216 veh/h per approach, where Poisson arrivals "
        "are realistic. Queues stay short (HCM LOS A).",
        (*_US_SOURCES[:2], GIUFFRE_2016, *DRIVER_SOURCES, MODEL_CHECKS),
        {
            **_US_SINGLE_LANE,
            # assumption: 216 veh/h per approach, in the 'low' 150-300 veh/h band (NCHRP 672 Exhibit 1-9);
            # Poisson arrivals are realistic up to about 400-500 veh/h (Giuffre et al. 2016)
            "entry_rate": 0.06,
            "density": 15.0,  # assumption: start only; the density it settles at (15.2 veh/km/lane)
        },
        "Same layout, entry rule and drivers as us_single_lane (see its notes). This version: all of the demand "
        "entered (215 veh/h per entrance), median queue wait 1.1 s (0.9-1.6 s), longest queue 3 cars (2-5).",
        cite="NCHRP Report 672 (2010); Giuffre et al. 2016"),

    Preset(
        "us_single_lane_peak", "US single-lane peak hour",
        "Peak hour at the NCHRP single-lane service limit (576 veh/h per approach, 5% trucks, varied drivers), "
        "with demand swinging between 0.5x and 1.5x every 10 minutes. Queues build at the surges.",
        (*_US_SOURCES[:2], FHWA_HPMS, FHWA_VM1, DFT_2024, NCHRP_1061, SIVAK_1982, *DRIVER_SOURCES, MODEL_CHECKS),
        {
            **_US_SINGLE_LANE,
            # derived: 576 veh/h per approach, inside the 560-625 veh/h single-lane service limit
            # (NCHRP 672 Exhibit 1-9: 25,000 veh/day x K 0.09-0.10 / 4 legs)
            "entry_rate": 0.16,
            # assumption: the model's stylised surge (0.5x-1.5x over 10 min, same mean); real peaks are
            # flatter (peak hour factor 0.92, FHWA HPMS: the busiest 15 min only 9 % above the hour)
            "demand_profile": "peak",
            # assumption: between GB A roads (5.0 % lorries, DfT 2024) and US urban roads (7.9 % of
            # vehicle-miles, FHWA VM-1 2022)
            "truck_share": 0.05,
            # derived: US single-unit SU-30 9.14 m and WB-67 22.40 m (NCHRP RR 1061) weighted by their urban
            # vehicle-mile shares 3.9 and 4.0 %: (3.9 x 9.14 + 4.0 x 22.40) / 7.9 = 15.9 m
            "truck_length": 16.0,
            "headway_std": 0.3,  # Zhu 2018: between-driver sd of the calibrated time gap 0.334 s
            # derived: coefficient of variation 0.39 (Zhu 2018) to 0.52 (Sivak 1982) x 0.6 s = 0.23-0.31 s
            "reaction_std": 0.25,
        },
        "HCM 6 check at the mean demand (cars only): the main approach (691 veh/h) against about 532 veh/h "
        "circulating has capacity 802 veh/h, v/c 0.86, at the 0.85-0.90 limit of satisfactory operation "
        "(NCHRP 672 Sec. 4.5.5); 5 % trucks at PCE 2 add 5 % in passenger-car units. Above about 400-500 "
        "veh/h real arrivals bunch, so Poisson arrivals are an approximation (Giuffre et al. 2016). Use "
        "Demand profile 'constant' for an HCM-style peak hour. This version: 550 of the 568 veh/h arriving per "
        "entrance entered, median queue wait 72 s (42-114 s; LOS F in 13 of 16 seeds, E in 3; queue wait "
        "only), longest queue 42 cars (28-50); in 5 of 16 seeds a queue reached the 50-car limit and turned "
        "arrivals away (up to 1 a minute). The 5 % trucks of 16 m take the busiest entrances to capacity: "
        "without them the median wait is 44 s (27-73 s) and no queue fills; with Demand profile 'constant' "
        "also 44 s. The research's check without trucks and driver spread (2 seeds, follow-up 2.6): 555 "
        "veh/h, 32.8 s (LOS D), 32 cars. Otherwise as us_single_lane.",
        cite="NCHRP Report 672 (2010); FHWA VM-1 (trucks)"),

    Preset(
        "us_single_lane_wet", "US single-lane, wet",
        "The US single-lane roundabout on a wet road in rain: slower, softer braking, longer time gaps and "
        "about 10% longer accepted entry gaps (measured at Korean roundabouts).",
        (LEE_2018, RAKHA_2010, RAKHA_2009, SHRP2_L08, HRANAC_2006, WALLMAN_2001, *_US_SOURCES[:2],
         *DRIVER_SOURCES, MODEL_CHECKS),
        {
            **_US_SINGLE_LANE,
            "road_condition": "wet",  # the engine's rain factors (ROAD_CONDITIONS, sourced there)
        },
        "Expected: entry gap 2.5 x 1.10 = 2.75 s (and follow-up x1.05, entries 2.7 s apart; time gaps x1.12), "
        "which in the model costs about 8 % of the entry capacity at saturation (606 against 660 veh/h per "
        "entrance, seeds 0-3), in line with the 10-11 % capacity loss in rain on freeways (FHWA 2006) and the "
        "SHRP 2 heavy-rain factor 0.89. The demand is unchanged, so the loss shows as longer queues; real "
        "volumes fall 15-30 % in bad weather (FHWA), which a lower Entrance chance adds. This version: all of "
        "the demand entered, median queue wait 6.5 s (4.4-10.9 s; dry 4.8), longest queue 9 cars (5-13; dry "
        "7). Otherwise as us_single_lane.",
        cite="Lee et al. 2018; Rakha et al. 2010 (FHWA)"),

    Preset(
        "us_single_lane_snow", "US single-lane, snow",
        "The same roundabout on packed snow: about 25% slower, softer braking and acceleration, longer gaps. "
        "A Hokkaido test roundabout lost about 300 veh/h of entry capacity on compacted snow, the model about "
        "130.",
        (MUNEHIRO_2017, RAKHA_2010, PERRIN_2000, SADEK_2004, FHWA_RWM, WALLMAN_2001, SUN_2026, KAGEYAMA_2013,
         *_US_SOURCES[:2], *DRIVER_SOURCES, MODEL_CHECKS),
        {
            **_US_SINGLE_LANE,
            "road_condition": "snow",  # the engine's packed-snow factors (ROAD_CONDITIONS, sourced there)
        },
        "Expected: entry gap 2.5 x 1.20 = 3.0 s (alone about 13 % less entry capacity in the model); with "
        "follow-up x1.20 (entries 3.1 s apart), time gaps x1.25 and the slower driving about 19 % less at "
        "saturation (660 -> 534 veh/h per entrance, -126 veh/h; seeds 0-3). Munehiro et al. 2017 measured "
        "about -300 veh/h on compacted snow (about -150 with non-slip grit), so the model loses less than half "
        "of that. Kageyama et al. 2013 saw winter speeds of 10 km/h or less on a much smaller test roundabout "
        "(26 m across), so do not expect that here. This version: nearly all of the demand entered (431 of 432 "
        "veh/h per entrance), median queue wait 15.7 s (8.7-32.6 s, LOS A-D; dry 4.8), longest queue 13 cars "
        "(7-26; dry 7). Otherwise as us_single_lane.",
        cite="Munehiro et al. 2017; Rakha et al. 2010 (FHWA)"),

    # A typical US two-lane roundabout with one-lane entries (the model's entries are always one lane).
    Preset(
        "us_two_lane", "US two-lane (HCM 6)",
        "US two-lane roundabout (NCHRP 672: 55 m across) with one-lane entries onto two circulating lanes at "
        "648 veh/h per approach (v/c about 0.8). Follow-up 2.4 s (entries 2.5 s apart) and entry gap 2.2 s "
        "reproduce HCM 6 capacity in this model (HCM: follow-up 2.5 s, critical headway 4.3 s).",
        _US_SOURCES,
        {
            "n_lanes": 2,  # two-lane (multilane) roundabout (NCHRP 672 Exhibits 1-9, 6-9)
            # derived: inscribed circle 55 m (NCHRP 672 Sec. 6.3.1: 49-55 m is the usual start for two
            # lanes): 55 / 2 - 1.5 x 3.5 m = 22.25 m (outer lane 25.75 m)
            "radius": 22.25,
            "traffic_mode": "open",  # model: open demand
            # assumption: 648 veh/h per approach, v/c about 0.79 under HCM 6 (one-lane entry, two circulating
            # lanes, circulating flow about equal to the entering flow); the two-lane service limit of
            # 1,010-1,125 veh/h needs two-lane entries, which the model does not have
            "entry_rate": 0.18,
            "exit_prob": 0.5,  # assumption: 2 / n_arms (a mean trip of 2 exits, straight on)
            # derived: model equivalent of the HCM 6 critical headway 4.32 s (one-lane entry against two
            # circulating lanes, A 1420, B 0.85e-3): about 750 veh/h per entrance at saturation vs HCM 6's 750
            # (this version, seeds 0-3: 742-753; the research: 759 with follow-up 2.5)
            "entry_gap": 2.2,
            # derived: HCM 6 follow-up 2.536 s (3600 / 1420; FHWA-SA-15-070 field mean 2.5 s) less the one 0.1 s
            # step the engine adds: entries 2.5 s apart
            "follow_up": 2.4,
            # derived: NCHRP 672 Eq. 6-1: 8.06 m/s at R 22.25 m (v^2 / R = 2.92 m/s2), rounded to the step
            "lateral_accel": 3.0,
            "density": 25.0,  # assumption: start only; the density it settles at (24.3 veh/km/lane)
            **DRIVERS,
        },
        _ENTRY_GAP_NOTE.format(literal="Literal HCM 6 values: critical headway 4.3 s, follow-up 2.5 s (2.2 s "
                                       "here gives about 750 veh/h per entrance at saturation, as HCM 6, 4.3 s only "
                                       "about 440).") +
        " The model's entries join and check gaps in the outer lane only; HCM's one-lane entry counts both "
        "circulating lanes as conflicting, which the calibrated entry gap absorbs. Mean desired speed keeps "
        "the 12 m/s default, inside the 40-50 km/h multilane entry design speeds (NCHRP 672 Exhibit 6-47); "
        "the cornering cap binds on the ring anyway. Equal demand on every approach (assumption) keeps the "
        "v/c check simple. HCM 6 check: capacity 1420 exp(-0.00085 x 648) = 819 veh/h, v/c 0.79, control "
        "delay about 23 s. Research check (2 seeds, follow-up 2.5): 639 veh/h entered per entrance, mean queue "
        "wait 16.4 s (LOS C, queue wait only), longest queue 17 cars. This version: all of the demand entered "
        "(644 veh/h per entrance), median queue wait 12.7 s (8.4-20.6 s, mostly LOS B), longest queue 14 cars "
        "(10-19). Two 3.5 m model lanes = 7 m; real two-lane circulatory roadways are 8.5-9.8 m wide. "
        + DRIVERS_NOTE,
        cite="NCHRP Report 672; HCM 6 (FHWA-SA-15-070)"),

    # The Place Charles de Gaulle (Etoile), Paris: 12 avenues on a very large unmarked ring.
    Preset(
        "etoile", "Place Charles de Gaulle",
        "Paris's Etoile approximated: 12 avenues, a 166 m inscribed circle, 5 of its 8 unmarked lanes, "
        "30 km/h and about 4,750 veh/h entering (Paris open-data counts). Entering drivers really have "
        "priority (priorite a droite), which the model cannot do, so entry gaps are set very short.",
        (WIKI_ETOILE, OSM_ETOILE, PARIS_COUNTS, LECUYER_2024, DEVISPERMIS_2026, PARIS_30KMH, PERRIN_2000,
         *DRIVER_SOURCES, MODEL_CHECKS),
        {
            # assumption: the model's maximum; the unmarked ring is driven as about 8 lanes since the 2023-24
            # works, 12 before (Lecuyer 2024, quoting the city's 2022 press file)
            "n_lanes": 5,
            # twelve avenues meet the ring (Wikipedia; OpenStreetMap); 26-33 degrees between neighbours, so
            # even spacing is within about 4 degrees
            "n_arms": 12,
            # derived: outer kerb radius 83 m (inscribed circle about 166 m, OSM fit): 166 / 2 - 4.5 x 3.5 m
            # = 67.25 m, so the outer model lane runs along the kerb
            "radius": 67.25,
            "traffic_mode": "open",  # model: open demand
            # assumption: the 7 counted approaches bring 4,020-4,270 veh/h in the daytime (Paris open data,
            # 08-18 UTC); all 12 together an estimated 4,500-5,000 -> 4,750 / 12 = 396 veh/h = 0.11 /s
            "entry_rate": 0.11,
            # derived: least-squares fit of the engine's weights 1 + x cos(2 theta) to the 7 counted
            # approaches, Grande Armee as arm 0 and the Champs-Elysees opposite: x = 0.58
            "demand_imbalance": 0.6,
            # assumption: 2 / n_arms, a mean trip of 6 exits, i.e. across the place (Grande Armee to the
            # Champs-Elysees); real exits are very uneven (Marceau and Wagram mostly exits)
            "exit_prob": 0.17,
            # assumption: the slider minimum, standing in for priorite a droite (entering cars have priority
            # over circulating ones), which the model cannot represent
            "entry_gap": 1.5,
            # assumption: entrants do not yield, so a queue-discharge headway: 1.99 s dry saturation headway
            # at stop lines (Perrin 2000), less the one 0.1 s step the engine adds (entries 2.0 s apart)
            "follow_up": 1.9,
            # assumption: the 30 km/h limit (8.33 m/s, Paris since 30 Aug 2021; OSM maxspeed=30 on the ring),
            # rounded to the 0.5 step; no measured circulating speeds were found
            "mean_speed": 8.5,
            "density": 25.0,  # assumption: start only; the density it settles at (23.3 veh/km/lane)
            **DRIVERS,
        },
        "An approximation: the model has evenly spaced arms with one-lane entries, at most 5 lanes and no "
        "priority to the right. Geometry (OpenStreetMap circle fits, about 1-2 m): outer kerb r 83.2 m; "
        "the central island is r 45.0 m before 2023 (38 m of carriageway) and the widened pedestrian ring "
        "r 56.5 m now (26.5 m); the model's 5 x 3.5 m lanes span r 65.5-83 m, so the inner 9 m are left out. "
        "Avenue Marceau is exit-only and Mac-Mahon entry-only; every model arm is both. Daytime entering flows "
        "(veh/h, 08-18 UTC): Grande Armee 866-1,029, Foch 786-1,022, Champs-Elysees 688-875, Mac-Mahon "
        "625-695, Friedland 352-571, Wagram 205-295, Victor Hugo 120-131 (Hoche, Carnot, Kleber and Iena "
        "uncounted); the popular '110,000 veh/day' has no stated source. Demand is a flat plateau from 08 to "
        "18 UTC, so 'constant' fits better than 'peak'. No trucks (no Paris vehicle mix found); the cobbles "
        "(adhesion about 0.40-0.50) are not modelled; the cornering limit does not matter at r 67-81 m. "
        "Demand: 0.11 x 1.6 = 634 veh/h on the main-axis arms 0 and 6 and 158 at right angles to them; the "
        "real Grande Armee and Foch bring 870-1,030, so the fitted shape under-represents them by about 30 %. "
        "The main-axis queues build up all the same, because with exit chance 0.17 a car passes about 5 "
        "entrances, so about 1,900 veh/h circulate past each entrance (396 x 0.83 / 0.17), more than a "
        "one-lane entry can merge 634 veh/h into; with Exit chance 0.3 (about 920 veh/h circulating) the same "
        "demand queues at most 18 cars. Research check (2 seeds): 376 of 396 veh/h per entrance entered, mean "
        "queue wait 47 s. This version: 377 of 395 veh/h per entrance entered, median queue wait 52 s (34-85 "
        "s, LOS F in every seed); in 8 of 16 seeds a main-axis queue reached the 50-car limit and turned "
        "arrivals away (up to 3 a minute). The real place has jammed too since the lane reduction (Lecuyer "
        "2024). " + DRIVERS_NOTE,
        cite="OpenStreetMap geometry; Paris open-data counts"),
)}

_ALIASES: dict[str, str] = {}


def _alias(text: str) -> str:
    """Lower case without separators: 'US-single_lane', 'us single lane' and 'us_single_lane' all match."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


for _p in PRESETS.values():
    for _text in (_p.name, _p.title):
        if _ALIASES.setdefault(_alias(_text), _p.name) != _p.name:
            raise ValueError(f"preset name or title {_text!r} matches two presets")


class UnknownPresetError(KeyError):
    """A preset name that matches none; its message lists the valid names."""

    def __str__(self) -> str:  # KeyError would show the message in quotes
        return str(self.args[0]) if self.args else ""


def names() -> list[str]:
    """The preset names in display order ('default' first)."""
    return list(PRESETS)


def get(name: str | Preset) -> Preset:
    """The preset called ``name``: case, '-', '_' and spaces do not matter, and its title works too."""
    if isinstance(name, Preset):
        return name
    key = _ALIASES.get(_alias(name))
    if key is None:
        raise UnknownPresetError(f"unknown preset {name!r}; choose from: {', '.join(PRESETS)}")
    return PRESETS[key]


def apply(name: str | Preset, base: SimConfig | None = None) -> SimConfig:
    """The model of preset ``name``: its changes on top of the defaults, with ``base``'s seed.

    Every other setting - also those without a slider (lane width, car width,
    time step) - comes from the defaults, not from ``base``, so a preset gives
    the same model whatever was set before. Apply explicit flags or ``--set``
    values afterwards, e.g. ``apply("etoile", cfg).updated(n_lanes=3)``.
    """
    preset = get(name)
    return SimConfig(**{**preset.changes, "seed": base.seed if base is not None else None})


def find(cfg: SimConfig) -> str | None:
    """Name of the preset ``cfg`` is (whatever its seed), or None if it matches none."""
    for name in PRESETS:
        if apply(name, cfg) == cfg:
            return name
    return None


def _wrap(text: str, width: int | None, indent: str = "", hang: str = "") -> list[str]:
    if not width:
        return [indent + text]
    return textwrap.wrap(text, width, initial_indent=indent, subsequent_indent=indent + hang,
                         break_long_words=False, break_on_hyphens=False) or [indent]


def describe(name: str | Preset, width: int | None = None) -> str:
    """Everything about a preset as plain ASCII text: title, description, settings, notes and sources.

    ``width`` wraps the lines (URLs are never broken); None keeps one line per
    paragraph, for a help box that wraps by itself.
    """
    p = get(name)
    lines = [*_wrap(f"{p.title} ({p.name})", width), *_wrap(p.description, width),
             *_wrap(f"Settings: {_settings(p)}", width, hang="  "),
             *_wrap(f"Notes: {p.notes}", width, hang="  "), "Sources:"]
    for source in p.sources:
        lines += _wrap(f"- {source}", width, "  ", "  ")
    return plain("\n".join(lines))


def _settings(p: Preset) -> str:
    if not p.changes:
        return "the model defaults."
    return ", ".join(f"{k}={short_value(v)}" for k, v in p.changes.items()) + "; everything else at the defaults."


def list_text(width: int = 100, full: bool = False) -> str:
    """All presets for ``--list-presets`` as plain ASCII: name, title, description, settings and main source
    (``full``: :func:`describe` each, with notes and every source)."""
    if full:
        return "\n\n".join(describe(name, width) for name in PRESETS) + "\n"
    lines = [*_wrap("Presets (--preset NAME). Each starts from the model defaults (keeping the seed) and sets "
                    "the values listed; the parameter flags given with it (and --set in experiments) override "
                    "them.", width), ""]
    for p in PRESETS.values():
        count = f" (1 of {len(p.sources)})" if len(p.sources) > 1 else ""
        lines += [f"{p.name:<24} {p.title}", *_wrap(p.description, width, "    "),
                  *_wrap(f"Settings: {_settings(p)}", width, "    ", "  "),
                  *_wrap(f"Source{count}: {p.sources[0]}", width, "    ", "  "), ""]
    return plain("\n".join(lines).rstrip() + "\n")


# ----------------------------------------------------------- random settings
@dataclass(frozen=True)
class Numeric:
    """Plausible values of a numeric parameter: a triangular distribution over ``lo``-``hi`` whose mode,
    the most likely value, is ``typical``."""
    lo: float
    hi: float
    typical: float
    basis: str = ""  # why, and the source

    def __post_init__(self) -> None:
        if not self.lo <= self.typical <= self.hi:
            raise ValueError(f"need lo <= typical <= hi, got {self}")


@dataclass(frozen=True)
class Choice:
    """Plausible values of a categorical (or small whole-number) parameter and how often each occurs."""
    choices: tuple
    weights: tuple[float, ...]
    basis: str = ""

    def __post_init__(self) -> None:
        if not self.choices or len(self.weights) != len(self.choices) or min(self.weights) < 0 \
                or sum(self.weights) <= 0:
            raise ValueError(f"need one weight >= 0 per choice (not all 0), got {self}")


# Plausible real-world values for every parameter with a slider (research of 2026-10-25), not the whole
# slider ranges (which allow extremes on purpose). random_changes links some of them (see there).
RANDOM_RANGES: dict[str, Numeric | Choice] = {
    # Road
    "n_lanes": Choice((1, 2, 3, 4, 5), (0.40, 0.35, 0.15, 0.05, 0.05),
                      "Most roundabouts are single- or two-lane; three or four lanes need 55-100 m across, five "
                      "only very large rings like the Etoile (NCHRP 672 Exhibits 1-9, 6-9; weights assumed)."),
    "n_arms": Choice((3, 4, 5, 6), (0.25, 0.60, 0.10, 0.05),
                     "4 legs is the standard case and 3 common; DMRB CD 116 2.3.4: normal roundabouts should not "
                     "have five or more arms. No closed ring (0): an experiment, not a roundabout (NCHRP 672 "
                     "Exhibit 1-9; weights assumed)."),
    "radius": Numeric(12.0, 70.0, 25.0,
                      "radius = across / 2 - (lanes - 0.5) x 3.5 m: single-lane 28-46 m across -> 12-21 m, "
                      "two-lane 46-67 m -> 18-28 m, three-lane 61-91 m -> 22-37 m, UK normal up to 100 m, the "
                      "Etoile 67 m (NCHRP 672; DMRB CD 116). Drawn from RADIUS_BY_LANES for the lane count."),
    "car_length": Numeric(4.0, 5.0, 4.5,
                          "EU production average 4.36 m (2022), US 2019 passenger models 4.86 m, Toyota Vitz "
                          "3.885 m; simulations use 4-5 m (Inovev 2023; NCHRP RR 1061 Table 18; Tadaki 2013)."),
    "road_condition": Choice(("dry", "wet", "snow", "ice"), (0.75, 0.17, 0.06, 0.02),
                             "Depends on the climate; weights assumed for a temperate one."),
    "weather_scale": Numeric(0.5, 2.0, 1.0,
                             "The studies behind the road factors span about 0.5-1.6 x each one's change from "
                             "dry (snow speeds -25 to -40 % against -25 %, snow discharge headways +13 to +26 % "
                             "against +25 %, rain speeds -6 to -10 % against -8 %: FHWA RWM, Perrin 2000; derived); "
                             "the model's snow capacity loss needs about 2 x to reach Munehiro 2017's -300 veh/h. "
                             "Typical 1: the factors as given."),
    # Traffic
    "density": Numeric(5.0, 110.0, 35.0,
                       "Closed rings flow freely below about 80 veh/km and jam above about 89 (Tadaki 2013); jam "
                       "density is 118-140 veh/km (HCS7; IDM). In open mode only the start: drawn from "
                       "DENSITY_OPEN then. Kept below DENSITY_CAP of the drawn vehicles' jam density."),
    "exit_prob": Numeric(0.1, 0.8, 0.5,
                         "Drawn as 2 / n_arms x 0.75-1.25 (EXIT_SPREAD): a mean trip of about half the circle, "
                         "through traffic. One constant chance cannot match real turning splits (left turns "
                         "0-40 %, NCHRP 672 Exhibit 3-12; derived)."),
    "entry_gap": Numeric(1.5, 3.5, 2.5,
                         "On the model's scale (the lag to the outer-lane car behind): field critical headways "
                         "of 3.9-5.9 s (US 4.7, HCM 6 5.0 / 4.3, Germany 4.1, Denmark 4.7-5.1) are about 2.1-2.5 s "
                         "more than the model value with the same capacity (FHWA-SA-15-070; HCM 6; engine "
                         "check). Mode 2.5 s single-lane, ENTRY_GAP_MULTILANE with more lanes."),
    "follow_up": Numeric(1.8, 3.1, 2.5,
                         "Field follow-up headways 1.9-3.2 s, typically 2.6: US single-lane site means 1.7-3.0 s "
                         "(weighted 2.6 s, FHWA-SA-15-070); HCM 6 2.5-2.7 s; HCM 2010 3.19 s; Germany 2.9 s; "
                         "Switzerland 2.27-2.76 s; Hungary 1.8-2.4 s. Set 0.1 s lower, as the engine adds one "
                         "time step (2.5 gives entries 2.6 s apart)."),
    "turn_speed": Numeric(4.0, 9.0, 6.0,
                          "Entering and leaving cars are slower than circulating ones (NCHRP 672 speed-radius "
                          "relation: 6.9-11 m/s at R 15-50 m) and queued ones start from rest; the typical value "
                          "is the model default (assumption)."),
    # Demand
    "traffic_mode": Choice(("closed", "open"), (0.25, 0.75),
                           "Real roundabouts have open demand; closed rings are experiments (weights assumed)."),
    "entry_rate": Numeric(0.03, 0.22, 0.11,
                          "Per approach: low 150-300 veh/h (0.04-0.08 /s), moderate 400-500 (0.11-0.14), the "
                          "single-lane service limit 560-625 (0.16-0.17) (NCHRP 672 Exhibit 1-9); the Etoile "
                          "averages 0.11 /s. Drawn as a degree of saturation (DEMAND_VC) of the layout's HCM 6 "
                          "capacity on the drawn road condition and weather effect (ROAD_CAPACITY), with the "
                          "drawn trucks (TRUCK_PCE)."),
    "demand_profile": Choice(("constant", "peak"), (0.7, 0.3),
                             "A peak hour factor of 0.92 means demand is nearly constant within an hour (FHWA "
                             "HPMS); the Etoile counts show a flat daytime plateau (weights assumed)."),
    "demand_imbalance": Numeric(0.0, 0.7, 0.2,
                                "NCHRP 672 planning assumes a minor-street share of 0.33-0.50 (0-0.34 with 4 arms); "
                                "fits to counts give 0.58 (Etoile) and 0.67 (Armdale Rotary, AM peak)."),
    "max_queue": Numeric(20, 100, 50,
                         "Approach storage before spill-back blocks the junctions upstream: 50 cars are about "
                         "300-425 m at 6-8.5 m per queued car (HCS7 jam density). A modelling boundary "
                         "(assumption)."),
    "subject_arm": Choice(("off",), (1.0,),
                          "A measuring instrument (a saturated test entrance), not a feature of real roundabouts: "
                          "never drawn on (model)."),
    # Vehicles
    "truck_share": Numeric(0.0, 0.15, 0.05,
                           "Lorries are 1.1 % of the traffic on GB minor roads and 5.0 % on A roads (DfT 2024); "
                           "trucks 7.9 % of US urban vehicle-miles (FHWA VM-1 2022); NCHRP 672 calls more than "
                           "10 % semi-trailers frequent."),
    "truck_length": Numeric(9.0, 18.75, 12.0,
                            "US SU-30 9.14 m; EU rigid limit and SU-40 12 m; EU articulated 16.5 m; EU road train "
                            "18.75 m (NCHRP RR 1061 Table 1; Directive 96/53/EC). The default when there are no "
                            "trucks."),
    # Drivers
    "mean_speed": Numeric(8.0, 14.0, 11.0,
                          "Maximum entry design speeds 30-40 km/h single-lane and 40-50 km/h multilane (NCHRP 672 "
                          "Exhibit 6-47); 40 km/h free speed on the ring experiments; 30 km/h city limits."),
    "speed_std": Numeric(0.8, 2.2, 1.4,
                         "11.5-14.7 % of the desired speed (Kesting 2007; Zhu 2018 Table 4), 1.3-1.6 m/s at "
                         "11 m/s; the range allows 7-20 %."),
    "reaction_time": Numeric(0.3, 1.2, 0.6,
                             "GHR-calibrated median 0.50 s (Zhu 2018); expected braking median 0.66 s (Johansson "
                             "& Rumar 1971) and 0.70-0.75 s (Green 2000); about 1.2 s for unexpected signals "
                             "(Sivak 1982)."),
    "reaction_std": Numeric(0.0, 0.35, 0.2,
                            "Coefficient of variation 0.39 (Zhu 2018) to 0.52 (Sivak 1982) on reaction times of "
                            "0.4-0.7 s; at most REACTION_CV_MAX x the drawn reaction time."),
    "reaction_distance": Numeric(15.0, 60.0, 25.0,
                                 "Drivers anticipate 1-7 vehicles (Treiber 2006); 15-60 m covers about 1-5 cars at "
                                 "roundabout spacings (assumption)."),
    "anticipation": Numeric(1, 5, 5,
                            "Drivers anticipate 1-7 vehicles (Treiber, Kesting & Helbing 2006, Physica A 360); the "
                            "slider stops at 5. Typical: the model's 5 (assumption: the value it was calibrated "
                            "with)."),
    "time_headway": Numeric(0.7, 1.8, 1.0,
                            "Calibrated IDM time gaps 0.88 +/- 0.33 s (Zhu 2018), 1.03-1.51 s (Kesting & Treiber "
                            "2008), 1.0 s city (Treiber & Kesting); 1.8 s is the German recommendation."),
    "headway_std": Numeric(0.0, 0.45, 0.3,
                           "Between-driver sd of calibrated time gaps 0.334 s (CV 0.38) and of steady-following "
                           "headways 0.349 s (Zhu 2018); at most HEADWAY_CV_MAX x the drawn time gap."),
    "min_gap": Numeric(1.0, 3.0, 1.75,
                       "Measured standstill gaps 1.39-1.64 m; calibrated 1.56 +/- 0.96 m; the IDM recommends 2 m "
                       "(Kesting & Treiber 2008; Zhu 2018)."),
    "max_accel": Numeric(0.8, 2.0, 1.3,
                         "Calibrated 0.84 +/- 0.28 m/s2 (Zhu 2018), 1.52-1.58 (Kesting & Treiber 2008), 1.5 in "
                         "town (Treiber & Kesting); realistic 0.8-2.5 m/s2."),
    "comfort_decel": Numeric(1.0, 3.0, 2.0,
                             "IDM 2.0 m/s2 in town, 1.5 on freeways (Treiber & Kesting); calibrated 0.61-3.73 "
                             "(Kesting & Treiber 2008); comfortable braking 0.27 g = 2.65 m/s2 (Garber & Hoel)."),
    "driver_noise": Numeric(0.15, 0.45, 0.28,
                            "Treiber & Kesting 2017: a noise intensity of about 0.2 m2/s3 is typical and 0.32 fits "
                            "their platoon experiment, 0.22-0.28 m/s2 with the engine's NOISE_TAU = 2 s."),
    "lateral_accel": Numeric(2.0, 4.0, 3.0,
                             "The NCHRP 672 speed-radius relation (Eq. 6-1) implies 3.0 m/s2 at 20 m and 2.4 at "
                             "50 m; the Sugiyama drivers used about 3.4. Drawn as Eq. 6-1 at the drawn radius "
                             "x LATERAL_SPREAD."),
    # Lane changing
    "lane_change_rate": Numeric(0.05, 0.5, 0.2,
                                "No source gives how often drivers consider a change (MOBIL defines the decision, "
                                "not the rate); around the model default (assumption)."),
    "lane_change_motive": Choice(("advantage", "random"), (0.85, 0.15),
                                 "Drivers mostly change lane for an advantage (MOBIL, Kesting 2007; weights "
                                 "assumed)."),
    "politeness": Numeric(0.0, 0.6, 0.2,
                          "MOBIL politeness 0-1 (Kesting 2007); traffic-simulation.de uses 0.1 and the model "
                          "0.3 (typical value assumed between them)."),
    # Initial state
    "init_spacing": Choice(("uniform", "random", "clustered"), (0.5, 0.4, 0.1),
                           "The ring experiments started evenly spaced (Sugiyama, Stern); real traffic is random "
                           "(weights assumed)."),
    # Automated vehicles and cooperation (OPT_IN_GROUPS: drawn only when their tab is asked for)
    "auto_share": Numeric(0.0, 1.0, 0.2,
                          "The research range: FHWA-OR-RD-22-11 Table 1.7 gives the HCM 7 CAV factors at 0-100 % "
                          "in 20 % steps. Today's shares are near 0; typical 0.2, a near-term mix (assumption)."),
    "auto_kind": Choice(("cav", "robotaxi", "acc"), (0.4, 0.3, 0.3),
                        "One automated kind per run: connected AVs (the HCM 7 CAV basis), driverless robotaxis "
                        "(Waymo, Hu et al. 2023) or driver-supervised ACC cars (weights assumed)."),
    "robotaxi_headway": Numeric(1.0, 1.6, 1.3,
                                "Waymo's median time headway over the human one: 3.49 / 2.69 s = 1.30 (mean ratio "
                                "1.26; Hu et al. 2023 Sec. III-B, Fig. 5; derived). 1 = as close as drivers, 1.6 a "
                                "more cautious fleet (assumption; Hu's AV jam spacing is 1.27 x a driver's)."),
    "companies": Choice((1, 2, 3, 4), (0.4, 0.3, 0.2, 0.1),
                        "Operators or brands sharing the roundabout: one in a fleet-only deployment, a few as "
                        "brands mix (weights assumed)."),
}

# Ring radius [m] by lane count: radius = across / 2 - (lanes - 0.5) x 3.5 m lanes, rounded to the slider's
# 1 m step (derived: 12.25-21.25, typically 18.25 for one lane; 17.75-44.75, typically 22.25 for two;
# 21.75-41.25, typically 29.25 for three; 33.25 and 34.25 at the low end for four and five).
RADIUS_BY_LANES: dict[int, Numeric] = {
    1: Numeric(12.0, 21.0, 18.0, basis="UK compact 28 m to US single-lane 46 m across, typically 40 m "
                                       "(DMRB CD 116; NCHRP 672 Sec. 6.3.1; derived)"),
    2: Numeric(18.0, 45.0, 22.0, basis="US two-lane 46-67 m across, typically 55 m; UK normal roundabouts up to "
                                       "100 m (NCHRP 672; DMRB CD 116; derived)"),
    3: Numeric(22.0, 41.0, 29.0, basis="three-lane 61-91 m across, UK normal up to 100 m (NCHRP 672 Exhibit 6-9; "
                                       "DMRB CD 116; derived)"),
    4: Numeric(33.0, 70.0, 45.0, basis="very large rings, 91 m across up to the Etoile (assumption)"),
    5: Numeric(34.0, 70.0, 50.0, basis="very large rings, 100 m across up to the Etoile, r 67 m with 5 lanes "
                                       "(assumption)"),
}
# Degree of saturation (demand / capacity) of the busiest entrance in open mode: from off-peak to capacity;
# NCHRP 672 Sec. 4.5.5 puts the limit of satisfactory operation at 0.85-0.90. The typical 0.45 is the
# research's typical 0.11 /s on a 4-arm single-lane roundabout (assumption). The capacity is the one on the
# drawn road (ROAD_CAPACITY) in vehicles, with the drawn trucks at TRUCK_PCE passenger cars each; against
# the dry, cars-only capacity most snow and ice draws would be gridlock rather than 'off-peak to capacity'.
DEMAND_VC = Numeric(0.1, 1.0, 0.45)
# HCM 6 entry capacity c = A exp(-B v_c) [pc/h] against v_c pc/h circulating: one entry lane against one
# circulating lane, and against two (the model's entries are one lane; Lobaza 2022 Tables 2-3).
HCM6_CAPACITY = {1: (1380.0, 1.02e-3), 2: (1420.0, 0.85e-3)}
# Entry capacity on each road relative to dry, as the model gives it at saturation (model check: us_single_lane,
# all entrances saturated, seeds 0-3: 660, 606, 534 and 460 veh/h; us_two_lane on ice 0.71). The engine's
# ROAD_CONDITIONS factors cause it; scaling only HCM 6's critical and follow-up headways by them gives 0.93,
# 0.83 and 0.84, as on ice the slower acceleration and speeds cost capacity too. Real roads: about -10 % in
# rain (FHWA 2006), -300 veh/h on compacted snow (Munehiro 2017).
ROAD_CAPACITY = {"dry": 1.0, "wet": 0.92, "snow": 0.81, "ice": 0.70}
# With weather_scale s the factor f becomes 1 + s (f - 1), as engine.road_factors scales the driving factors, at
# least 0.1 (engine.WEATHER_FLOOR). *derived* approximation: the model's capacity loss is not exactly linear in s
# (capacity test, us_single_lane, vc ~500 veh/h: snow at s 0.5 / 2 keeps 1.18 / 0.75 x its s = 1 capacity
# against 1.12 / 0.77 by this rule; ice 1.32 / 0.35 against 1.21 / 0.57, its ring flow failing at s = 2).
CAPACITY_FLOOR = 0.1
TRUCK_PCE = 2.0  # a heavy vehicle counts as 2 passenger cars in the HCM 6 roundabout method (E_T)
EXIT_SPREAD = (0.75, 1.25)  # exit_prob = 2 / n_arms x U(these) (derived, research), clipped to its range
ENTRY_GAP_MULTILANE = 2.2  # mode of the entry gap with 2+ lanes (the HCM 6 two-lane model equivalent)
DENSITY_CAP = 0.85  # random densities stay below this share of the jam density (assumption: no solid queue)
# Open mode: Density only sets the cars at the start, so start near the 15-30 veh/km/lane the open presets
# settle at rather than anywhere up to a jam that would keep the entrances shut for minutes (assumption).
DENSITY_OPEN = Numeric(5.0, 50.0, 20.0)
REACTION_CV_MAX = 0.52  # reaction-time spread at most this x the reaction time (Sivak 1982)
HEADWAY_CV_MAX = 0.4  # time-gap spread at most this x the time gap (Zhu 2018: CV 0.38)
LATERAL_SPREAD = (0.85, 1.15)  # cornering limit = NCHRP 672 Eq. 6-1 at the radius x U(these) (assumption)


def _stream(seed: int, name: str, salt: int = 0) -> np.random.Generator:
    """A random stream of its own for each parameter: adding or reordering parameters, or drawing only some,
    leaves the others' values for a seed unchanged."""
    return np.random.default_rng([seed, zlib.crc32(name.encode()), salt])


def _check_seed(seed) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError(f"seed must be a whole number >= 0, got {seed!r}")
    return int(seed)


def _sample(spec: Numeric | Choice, rng: np.random.Generator, **override):
    """One draw from ``spec`` (``lo``, ``typical``, ``hi`` may be overridden)."""
    if isinstance(spec, Choice):
        p = np.asarray(spec.weights, dtype=float)
        return spec.choices[int(rng.choice(len(spec.choices), p=p / p.sum()))]
    lo, hi = override.get("lo", spec.lo), override.get("hi", spec.hi)
    typical = min(max(override.get("typical", spec.typical), lo), hi)
    if hi <= lo:
        return lo
    return float(rng.triangular(lo, typical, hi))


def snap(name: str, value, clip: bool = True):
    """``value`` on parameter ``name``'s slider grid (lo + k x step), as SimConfig's type for it.

    ``clip`` keeps it inside the slider range (off: the grid goes on beyond it).
    """
    spec = PARAM_BY_NAME[name]
    if spec.choices:
        if value not in spec.choices:
            raise ValueError(f"{name} must be one of {', '.join(spec.choices)}")
        return value
    k = round((float(value) - spec.lo) / spec.step)
    if clip:
        k = min(max(k, 0), math.floor((spec.hi - spec.lo) / spec.step + 1e-9))
    return coerce(name, round(spec.lo + k * spec.step, 9))


def _clip(value: float, spec: Numeric) -> float:
    return min(max(value, spec.lo), spec.hi)


def _groups(groups: Iterable[str] | str | None) -> set[str]:
    """``groups`` as GROUPS names them (any case); None: DEFAULT_GROUPS (every group but the OPT_IN_GROUPS)."""
    if groups is None:
        return set(DEFAULT_GROUPS)
    by_key = {g.lower(): g for g in GROUPS}
    wanted = [groups] if isinstance(groups, str) else list(groups)
    unknown = [g for g in wanted if str(g).lower() not in by_key]
    if unknown:
        raise ValueError(f"unknown group(s) {', '.join(map(repr, unknown))}; choose from: {', '.join(GROUPS)}")
    return {by_key[str(g).lower()] for g in wanted}


def group_names(groups: Iterable[str] | str | None = None) -> list[str]:
    """``groups`` as GROUPS names them (any case), in tab order; DEFAULT_GROUPS for None (the OPT_IN_GROUPS
    only when named). ValueError for an unknown one."""
    wanted = _groups(groups)
    return [g for g in GROUPS if g in wanted]


def _nchrp_lateral(radius: float) -> float:
    """[m/s2] Sideways acceleration at the speed NCHRP 672 Eq. 6-1 predicts on a curve of ``radius`` m
    (V = 3.4415 R^0.3861 in mph and ft, superelevation +0.02)."""
    v = 3.4415 * (radius / 0.3048) ** 0.3861 * 0.44704
    return v * v / radius


def _hcm_entry_rate(vc: float, n_lanes: int, n_arms: int, exit_prob: float, imbalance: float,
                    road_condition: str = "dry", truck_share: float = 0.0, weather_scale: float = 1.0) -> float:
    """[1/s] Mean entrance chance that loads the busiest entrance to ``vc`` x its HCM 6 capacity.

    Each car passes 1 / exit_prob - 1 entrances before it leaves (an arm's exit
    comes before its entry), so a mean entering flow q puts about q (1 - p) / p
    on the ring at each entrance; the busiest entrance gets the largest demand
    weight w (as RingSim._demand_weights). Solves w q = vc f A exp(-B q (1 - p) / p)
    in passenger cars, with f = ROAD_CAPACITY[road_condition] (scaled by
    ``weather_scale``: 1 + scale (f - 1), at least CAPACITY_FLOOR), and converts
    back to vehicles: with ``truck_share`` s each vehicle is 1 + s (TRUCK_PCE - 1) cars.
    """
    a, b = HCM6_CAPACITY[1 if n_lanes <= 1 else 2]
    f = ROAD_CAPACITY[road_condition]
    a *= f if weather_scale == 1 else max(CAPACITY_FLOOR, 1.0 + weather_scale * (f - 1.0))
    p = max(exit_prob, 0.05)
    weights = 1.0 + imbalance * np.cos(2 * 2 * np.pi * np.arange(n_arms) / n_arms)
    w = float(weights.max() / weights.mean())
    lo, hi = 0.0, vc * a / w
    for _ in range(50):  # bisection: the left side grows with q, the right one falls
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if w * mid > vc * a * math.exp(-b * mid * (1 - p) / p) else (mid, hi)
    return lo / 3600.0 / (1.0 + truck_share * (TRUCK_PCE - 1.0))


def random_changes(seed: int, groups: Iterable[str] | str | None = None, *, base: SimConfig | None = None,
                   fixed: Mapping[str, object] | None = None) -> dict:
    """A plausible random value for every parameter (or those of the slider ``groups``), reproducible by seed.
    "Every" is that of DEFAULT_GROUPS: the automated vehicles' and cooperation tabs (OPT_IN_GROUPS) are drawn
    only when ``groups`` names them, so a draw of the human settings is what it was before them.

    Values come from RANDOM_RANGES, sit on the slider steps (so the viewer
    shows them exactly) and give a valid SimConfig. Some are linked, as on real
    roundabouts: the radius follows the lane count (RADIUS_BY_LANES); the exit
    chance the number of arms; the entrance chance a degree of saturation of
    the layout's HCM 6 capacity on the drawn road (and weather effect) with the drawn trucks; the
    entry gap's mode the lane count; the density is lower in open mode (where
    it only sets the start) and stays below the jam density of the vehicles;
    the spreads of reaction times and time gaps follow their means; the
    cornering limit follows the radius (NCHRP 672). A linked value uses the
    drawn value of what it depends on, or ``base``'s (default: the defaults)
    when that is not drawn. Truck length keeps its default when there are no
    trucks; settings that act only in the other traffic mode are drawn anyway,
    for when it is switched.

    ``fixed`` holds settings that are not drawn but kept as given (e.g. those
    set explicitly on the command line), and the linked draws use them: with
    ``fixed={"traffic_mode": "open"}`` the density is an open-mode start, with
    ``{"n_lanes": 1}`` the radius a single-lane one. Those of ``groups`` are
    returned unchanged.

    Each parameter has a random stream of its own, so drawing only some groups
    gives them the values of the full draw for the same seed (with
    ``base=random_config(seed)``), fixing some leaves the others' values alone
    (apart from the links), and new parameters leave the old ones' alone.
    """
    seed = _check_seed(seed)
    wanted = _groups(groups)
    drawn = [p.name for p in PARAMS if p.group in wanted]
    cfg = base if base is not None else DEFAULTS
    v = {name: getattr(cfg, name) for name in FIELDS}  # what the linked draws depend on
    for name, value in (fixed or {}).items():
        if name not in v:
            raise ValueError(f"fixed: unknown parameter {name!r}")
        v[name] = coerce(name, value)
    pending = set(drawn) - set(fixed or ())
    ranges = RANDOM_RANGES

    def pick(name: str, draw=None) -> None:
        if name in pending:
            pending.discard(name)
            rng = _stream(seed, name)
            v[name] = snap(name, draw(rng) if draw else _sample(ranges[name], rng))

    # The linked draws (each with the parameter's own stream).
    def radius(rng):
        return _sample(RADIUS_BY_LANES[min(max(int(v["n_lanes"]), 1), 5)], rng)

    def truck_length(rng):
        return _sample(ranges["truck_length"], rng) if v["truck_share"] > 0 else DEFAULTS.truck_length

    def reaction_std(rng):
        return min(_sample(ranges["reaction_std"], rng), REACTION_CV_MAX * v["reaction_time"])

    def headway_std(rng):
        return min(_sample(ranges["headway_std"], rng), HEADWAY_CV_MAX * v["time_headway"])

    def lateral_accel(rng):
        return _clip(_nchrp_lateral(v["radius"]) * rng.uniform(*LATERAL_SPREAD), ranges["lateral_accel"])

    def density(rng):
        length = v["car_length"] + v["truck_share"] * (v["truck_length"] - v["car_length"])  # mean vehicle
        spec = DENSITY_OPEN if v["traffic_mode"] == "open" else ranges["density"]
        return min(_sample(spec, rng), DENSITY_CAP * 1000.0 / (length + v["min_gap"]))

    def exit_prob(rng):
        if v["n_arms"] <= 0:  # no exits: any value will do
            return _sample(ranges["exit_prob"], rng)
        return _clip(2.0 / v["n_arms"] * rng.uniform(*EXIT_SPREAD), ranges["exit_prob"])

    def entry_gap(rng):
        mode = ENTRY_GAP_MULTILANE if v["n_lanes"] > 1 else ranges["entry_gap"].typical
        return _sample(ranges["entry_gap"], rng, typical=mode)

    def entry_rate(rng):
        if v["n_arms"] <= 0:  # no entrances: any value will do
            return _sample(ranges["entry_rate"], rng)
        rate = _hcm_entry_rate(_sample(DEMAND_VC, rng), int(v["n_lanes"]), int(v["n_arms"]), v["exit_prob"],
                               v["demand_imbalance"], v["road_condition"], v["truck_share"], v["weather_scale"])
        return _clip(rate, ranges["entry_rate"])

    # In order of dependence: each line may use the values drawn above it.
    for name in ("traffic_mode", "n_lanes", "n_arms"):
        pick(name)
    pick("radius", radius)
    for name in ("car_length", "road_condition", "weather_scale", "truck_share"):
        pick(name)
    pick("truck_length", truck_length)
    for name in ("mean_speed", "speed_std", "reaction_time", "reaction_distance", "time_headway", "min_gap",
                 "max_accel", "comfort_decel", "driver_noise"):
        pick(name)
    pick("reaction_std", reaction_std)
    pick("headway_std", headway_std)
    pick("lateral_accel", lateral_accel)
    pick("density", density)
    pick("exit_prob", exit_prob)
    pick("entry_gap", entry_gap)
    for name in ("follow_up", "turn_speed", "demand_profile", "demand_imbalance", "max_queue"):
        pick(name)
    pick("entry_rate", entry_rate)
    for name in sorted(pending):  # the rest are independent (lane changing, start)
        pick(name)
    return {name: v[name] for name in drawn}


def new_seed() -> int:
    """A fresh seed for :func:`random_changes` when none is given: short enough to note down and type back
    (``--random SEED``) to get the same settings again."""
    return int(np.random.default_rng().integers(100_000))


def random_config(seed: int, *, base: SimConfig | None = None, groups: Iterable[str] | str | None = None,
                  fixed: Mapping[str, object] | None = None) -> SimConfig:
    """``base`` (default: the defaults) with the ``fixed`` settings and :func:`random_changes` applied; its
    seed is kept."""
    base = base if base is not None else DEFAULTS
    changes = random_changes(seed, groups, base=base, fixed=fixed)
    return base.updated(**{**{k: coerce(k, x) for k, x in (fixed or {}).items()}, **changes})


def plausible_ranges(groups: Iterable[str] | str | None = None) -> dict[str, tuple | list]:
    """RANDOM_RANGES as :func:`latin_hypercube` ranges: (lo, hi) or the list of choices (of DEFAULT_GROUPS'
    parameters unless ``groups`` names others)."""
    wanted = _groups(groups)
    out: dict[str, tuple | list] = {}
    for p in PARAMS:
        if p.group in wanted:
            spec = RANDOM_RANGES[p.name]
            out[p.name] = list(spec.choices) if isinstance(spec, Choice) else (spec.lo, spec.hi)
    return out


# ---------------------------------------------------------- Latin hypercube
def _stratum(value: float, lo: float, hi: float, n: int) -> int:
    """Which of ``n`` equal strata of lo-hi ``value`` falls in (the top one includes hi). A value on a
    boundary belongs to the stratum above it, also when float rounding puts it a hair below (the 1e-9)."""
    return min(max(int(math.floor((value - lo) / (hi - lo) * n + 1e-9)), 0), n - 1)


def _is_range(name: str, spec) -> bool:
    """(lo, hi) of two numbers for a numeric parameter (or any other name), not a list of levels."""
    param = PARAM_BY_NAME.get(name)
    if param is not None and param.choices:
        return False
    return (isinstance(spec, (tuple, list)) and len(spec) == 2
            and all(isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) for x in spec))


def _python(value):
    return value.item() if isinstance(value, np.generic) else value


def _whole(name: str) -> bool:
    """A SimConfig field that takes whole numbers (the lanes, the seed, ...)."""
    return name in FIELDS and (type(getattr(DEFAULTS, name)) is int or getattr(DEFAULTS, name) is None)


def _grid(name: str) -> tuple[float, float] | None:
    """(origin, step) of the values ``name`` takes: its slider's grid, whole numbers for a whole-number
    SimConfig field without a slider (the seed), or None (any value)."""
    param = PARAM_BY_NAME.get(name)
    if param is not None:
        return None if param.choices else (float(param.lo), float(param.step))
    return (0.0, 1.0) if _whole(name) else None


def _grid_value(name: str, k: int):
    """Grid value number ``k`` of ``name`` (see _grid), as SimConfig's type for it."""
    if name in PARAM_BY_NAME:
        return snap(name, PARAM_BY_NAME[name].lo + k * PARAM_BY_NAME[name].step, clip=False)
    origin, step = _grid(name)
    return coerce(name, round(origin + k * step, 9))


def _typed(name: str, value):
    """``value`` as SimConfig's type for ``name`` (a choice checked against its choices); other names as given."""
    if name not in FIELDS:
        return value
    param = PARAM_BY_NAME.get(name)
    return snap(name, value, clip=False) if param is not None and param.choices else coerce(name, value)


def latin_hypercube(n: int, ranges: Mapping[str, object], seed: int) -> list[dict]:
    """``n`` samples (one dict each) of a Latin hypercube design over ``ranges``.

    ``ranges`` maps a name to (lo, hi) or to a list of levels (choices, or any
    list other than two numbers). Each range is split into ``n`` equal strata,
    each sampled once at a random point, in a random order per variable. A
    parameter's values are put on its slider step, within their stratum, so the
    design stays stratified (a whole-number setting, such as the lanes or the
    seed, gets whole numbers); when the range holds no more of those values
    than samples, they are used as levels. Levels are dealt out evenly (each
    ``n // m`` or ``n // m + 1`` times) in random order. Values of SimConfig
    fields get SimConfig's types (so text levels such as "3.5" work); other
    names are passed through. A range with lo == hi gives that value as it is.
    Ranges may go beyond the slider ranges (the grid continues). Every
    variable has a stream of its own from ``seed``, so adding one leaves the
    others' columns as they were.
    """
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError(f"n must be a whole number >= 1, got {n!r}")
    seed = _check_seed(seed)
    columns: dict[str, list] = {}
    for name, spec in ranges.items():
        rng = _stream(seed, name, 1)
        if _is_range(name, spec):
            lo, hi = float(spec[0]), float(spec[1])
            if not (math.isfinite(lo) and math.isfinite(hi)) or hi < lo:
                raise ValueError(f"{name}: need finite lo <= hi, got {spec!r}")
            if hi == lo:  # a fixed value, kept as given
                columns[name] = [_typed(name, lo)] * n
                continue
            grid = _grid(name)
            k0 = k1 = 0
            if grid is not None:
                origin, step = grid
                k0 = math.ceil((lo - origin) / step - 1e-9)
                k1 = math.floor((hi - origin) / step + 1e-9)
                if 0 < k1 - k0 + 1 <= n:  # too few grid values for n strata: deal them out as levels
                    columns[name] = _levels([_grid_value(name, k) for k in range(k0, k1 + 1)], n, rng)
                    continue
                if k1 < k0 and _whole(name):
                    raise ValueError(f"{name}: no whole number between {lo:g} and {hi:g}")
                if k1 < k0:  # a range between two slider values keeps its values as drawn
                    grid = None
            strata = rng.permutation(n)
            x = lo + (strata + rng.random(n)) * (hi - lo) / n
            if grid is not None:
                columns[name] = [_on_grid(name, xi, int(s), lo, hi, n, k0, k1) for xi, s in zip(x, strata)]
            else:
                columns[name] = [_typed(name, float(xi)) for xi in x]
        else:
            levels = list(spec) if isinstance(spec, (list, tuple)) else [spec]
            if not levels:
                raise ValueError(f"{name}: no values")
            levels = [_typed(name, lv) for lv in levels]
            columns[name] = _levels(list(dict.fromkeys(_python(lv) for lv in levels)), n, rng)
    return [{name: col[i] for name, col in columns.items()} for i in range(n)]


def _levels(levels: list, n: int, rng: np.random.Generator) -> list:
    """``n`` values dealt evenly from ``levels`` (which get the spare ones, and the order, at random)."""
    order = rng.permutation(len(levels))
    picks = order[np.arange(n) % len(levels)]
    return [levels[i] for i in rng.permutation(picks)]


def _on_grid(name: str, x: float, stratum: int, lo: float, hi: float, n: int, k0: int, k1: int):
    """The grid value (numbers ``k0``-``k1`` lie in lo-hi) nearest ``x`` that is in ``stratum``.

    Every stratum holds one when the range holds more grid values than
    strata: a stratum is then at least one step wide.
    """
    origin, step = _grid(name)
    k = min(max(round((x - origin) / step), k0), k1)
    for d in range(k1 - k0 + 1):
        for kk in ((k,) if d == 0 else (k - d, k + d)):
            if k0 <= kk <= k1:
                value = _grid_value(name, kk)
                if _stratum(float(value), lo, hi, n) == stratum:
                    return value
    raise AssertionError(f"{name}: no grid value in stratum {stratum} of {n} over {lo:g}-{hi:g}")
