# Roundabout traffic model

An agent-based model of a multi-lane roundabout, built to study the **backups
that form among cars already circulating**: phantom jams and stop-and-go waves
that appear with no accident and no bottleneck. Every car is an independent
driver with its own desired speed, reaction time, attention span and
lane-change decisions. Small cars on a large ring let the roundabout hold
hundreds to thousands of them.

You can change the number of lanes and entrances/exits, the density, driver
behaviour (and how much drivers differ), the share and length of trucks, the
road surface (dry, wet, snow or ice) and the starting layout, either live in a
viewer or in batch experiments that write CSVs and figures. By default the ring is a **closed**
system with a fixed number of cars; in **open** traffic mode cars arrive at the
entrances at random, queue, enter and leave for good, so the number of cars on
the ring varies with the demand, and the model reports queues, blocked
arrivals and entry delay with an HCM-style level of service.

On top of that:

- **Real-world presets** start from a published experiment, a design guide or
  traffic counts: the ring-road phantom-jam experiments (Sugiyama 2008, Tadaki
  2013, Stern 2018), US single- and two-lane roundabouts (NCHRP Report 672,
  HCM 6) at several demands and on wet and snowy roads, and the Place Charles
  de Gaulle in Paris, every value with its source.
- **Random settings** give every parameter a plausible real-world value,
  reproducible by a seed.
- **Saved sessions**: closing the viewer saves the simulation, its charts and
  the view, and the next launch carries on exactly where it stopped.
- **Sweeps of one or many settings** (full grids, plans from a CSV, Latin
  hypercube samples) and a **sweep viewer** to browse them or run new ones.

![The viewer: a closed ring at density 80 - phantom jams appear as orange stripes moving backwards in the space-time diagram](docs/viewer.png)

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m roundabout_sim              # interactive viewer (closing it saves the session; the next launch resumes it)
python -m roundabout_sim --preset sugiyama2008     # a published experiment (--list-presets: all of them)
python -m roundabout_sim --random 3   # plausible random settings, seed 3
python -m roundabout_sim --fresh      # ignore the saved session
python -m roundabout_sim.experiments run --baseline   # one headless run -> dashboard figure + CSVs
python -m roundabout_sim.experiments sweep --x density=10:150:10 --series n_lanes=1,2,3
python -m roundabout_sim.experiments view             # the sweep viewer: browse sweep CSVs or run new sweeps
python -m pytest -q                   # test suite (~3 min)
```

Every parameter can also be set from the command line, for example
`python -m roundabout_sim --n_lanes 1 --n_arms 0 --density 80 --seed 3`.
`--help` lists all of them. Settings are taken in this order: the defaults,
then `--preset` or `--random`, then the parameter flags (the same in the viewer
and in `experiments`).

## Things to try

| Experiment | How | What you should see |
|---|---|---|
| **Phantom jams from nothing** | Entrances/exits = 0, Lanes = 1, Density 65, then slowly raise it to 85 | At 65-70 the flow is smooth. From about 75-80, stop-and-go waves appear spontaneously and drift backwards at about -3 m/s (orange stripes sloping down in the space-time diagram). This is a phase transition. |
| **The Sugiyama experiment** | `python -m roundabout_sim --preset sugiyama2008` (or the Presets tab) | 22 cars on a 230 m circuit, as in the first phantom-jam experiment. A stop-and-go wave forms with no bottleneck and travels backwards against the traffic; the cars caught in it stop. The Stop-and-go chart rises within about half a minute (`experiments run --preset sugiyama2008` gives a *Time to first stop-and-go* of 33, 26 and 36 s for seeds 1-3); by Stern's detector (the speed spread across the cars above 2.5 m/s) the wave is there after 1-6 minutes (63, 156 and 339 s). `tadaki2013` and `stern2018` are the two later rings. |
| **Reaction distance** | Closed ring, 1 lane, Density 80; Drivers → Reaction distance 10 → 20 → 50 m | This is how far ahead drivers pay attention. At 10 m they notice the car in front too late and brake violently: over half the cars are jammed, with dozens of harsh-braking episodes per minute. At 20-25 m phantom jams form. From about 40 m drivers ease off early and the waves die down. |
| **Reaction time** | Drivers → Reaction time 0.3 → 1.0 s | Slower reactions make waves form at much lower density (from about 50 instead of 75-80 on a closed ring), cut peak flux by about a third and cause a lot of harsh braking. |
| **Trigger a jam yourself** | Click any car (or press **B**) | The car brakes for 3 s. Near the threshold the disturbance grows into a wave that outlives it. |
| **Are the backups natural?** | `python -m roundabout_sim.experiments run --baseline` | Repeats the run with exit chance 0 (same seed) and reports how much of the jamming the entrances/exits cause. At the defaults (density 50) it is all of it: the same ring without exits does not jam until about 75. |
| **Speed variance** | Drivers → Speed variance 0 → 4 m/s | Bigger differences between drivers lower the flux at low density (slow drivers hold others up), cause more lane changes, and in this model slightly damp the waves. |
| **Lanes and lane changing** | Road → Lanes 1-5; Lane chg tab → Lane-change chance, motive "random" | Lane changes are cut-ins that force the car behind to brake, which is another source of waves. |
| **Initial spacing** | Start → uniform / clustered, once at Density 70 on a closed 1-lane ring and once at 50 | The average gap is set by Density. Just below the jam threshold an uneven start locks in jams that an even start barely forms (density 70: about 17 % against 5 % jammed). At 50 every start ends up the same within 1-2 min (watch Spacing (Gini) fall back). |
| **Many cars** | Ring radius 300, Car length 2, Lanes 5, Density 120 | About 1,150 cars, with dozens of jams forming and dissolving at once. |
| **Open demand** | Demand → Traffic mode *open*; Entrance chance 0.05, then 0.1, then 0.2 /s | At 0.05 /s (180 veh/h per entrance) the ring drains to about 30 cars and entering cars wait about 2 s (level of service A). The default roundabout lets in only about 6 cars per minute per entrance (about 400 veh/h: entries merge into the outer lane, where the leavers gather), so 0.1 /s is right at capacity: queues come and go, and the level of service swings between A and F from run to run. At 0.2 /s the queues outgrow the entries: the "Cars and queues" chart climbs, queue tags at the entrances turn red ("full") at 50 cars and arrivals are blocked. |
| **Rush hour** | Open mode, Demand profile *peak*, Entrance chance 0.1 | The demand swings between 0.05 and 0.15 /s every 10 minutes (peaks at 5, 15, 25... min). The queues build up during each peak and are longest about 2 minutes after it (1-3 min), then drain, not always completely before the next one. |
| **A real roundabout at capacity** | `python -m roundabout_sim --preset us_single_lane_peak` | A US single-lane roundabout at its service limit, 576 veh/h per approach with 5 % trucks, the demand swinging between half and 1.5 times that every 10 minutes. The queues build at each surge (the check runs: median queue wait 72 s, LOS F in most seeds, the longest queue 42 cars); compare `us_single_lane` (queue wait about 5 s) and `us_single_lane_offpeak`. |
| **The Place Charles de Gaulle** | `--preset etoile`, then Traffic → Exit chance 0.17 → 0.3 | 12 avenues on a 5-lane ring with about 4,750 veh/h entering. With exit chance 0.17 a car passes about 5 entrances, so about 1,900 veh/h circulate past each one and the main-axis queues (bottom and top) grow towards the 50-car limit; at 0.3 they stay below about 20 cars. |
| **Random roundabouts** | Press **X** a few times; on the Drivers tab press **Shift+X** | Each X draws a plausible roundabout (lanes, radius, demand, road and drivers together; the seed shows in the help box, and `--random SEED` starts with the same settings). Shift+X redraws only the tab shown. |
| **Trucks** | Vehicles → Truck share 0 → 0.1 → 0.2; press **C** until the cars are coloured by type | A truck takes up about twice a car's road: it is 12 m long, keeps a longer gap and is slow off the mark. At the defaults the flux falls from about 1.0 to 0.87 and 0.78 veh/s and the jammed cars rise from about 19 % to 26-28 % (means of 4 seeds). |
| **Road condition** | Road → Road condition dry → wet → snow → ice | Everyone drives slower and keeps longer gaps: at the defaults the flux falls from about 1.0 veh/s (dry) to 0.93 (wet), 0.78 (snow) and 0.65 (ice). Harsh braking jumps from about 1.5 per minute to 14 on snow and 65 on ice, because comfortable braking there is close to what the road allows. |
| **Driver differences** | Closed 1-lane ring (Entrances/exits 0, Lanes 1), Density 70; Drivers → Reaction time spread 0 → 0.3 s | Just below the jam threshold a spread of reaction times (the slowest drivers take about 1 s) nearly triples the jammed cars, from about 5 % to 14 %, and raises stop-and-go. A time gap spread of 0.4 s alone changes little (means of 4 seeds; single runs vary a lot). |
| **Follow-up time** | Road → Lanes 1; Traffic → Exit chance 0.5, Density 10; open mode, Entrance chance 0.2; then Follow-up time 1 → 3 s | Real drivers leave about 2.5-3.2 s between cars entering from one queue. With little circulating traffic the follow-up time sets how fast a queue can empty, and so the entry capacity: with 1 s the entrances are just at capacity (v/c about 0.97, a queue wait of about 16 s, about 13 cars queued at the 4 entrances), with 3 s over it (v/c 1.2, about 35 s and 29 cars, level of service F; the queues keep growing). Means over 10 seeds of minutes 2-7; single runs vary a lot. On the default 3-lane ring the gaps in the circulating traffic limit the entries more than the follow-up time does. |
| **Sweep from the viewer** | Press **V** (or *Sweep viewer*), then on its *New sweep* tab pick a setting, e.g. Density, and *Run sweep* | The sweep viewer opens with the viewer's settings as the base of every run; the sweep runs in the background and its chart opens when it is done. |

## The viewer

The **ring view** draws every vehicle to scale; trucks are longer and wider,
and bend along their lane.
Car colour shows speed relative to the car's free speed: orange is stopped,
grey is half speed and blue is free flow. **C** switches to colouring by the
driver's desired speed, by lane or by vehicle type (car or truck). Hover over
a car to see its type, length, speed, time gap, reaction time and exit.

- A thin inner band shows how much of each sector of the road is covered.
- An arrow points at the side where cars bunch up, shown only when they bunch
  more than chance would explain.
- Stubs mark the entrances and exits; queued cars wait on them. A queue longer
  than its stub (or any queue on a ring so big that its cars are too small to
  count) gets a count at the stub's end, red and marked *full* when it has
  reached Max queue and turns arrivals away. Hover a count for its help.
- The header names the preset running (or the random settings' seed; "changed"
  once a setting was changed since), the layout, the truck share and the road
  condition when they are set, and in open mode the demand: veh/h per entrance,
  on average (and on the main road) when the demand imbalance or the peak
  profile is on. With trucks the stats card adds the trucks on the ring and
  their mean speed.
- A detector line counts cars passing a point.

The **controls** are in tabs: Presets, Road, Traffic, Demand, Vehicles,
Drivers, Lane chg and Start (on two rows). Sliders marked *rebuilds* restart the
roundabout when you release them; all others change every driver live, and
each change is marked on the charts. Controls marked *open mode only*,
*no entrances* (on a ring without entrances, such as the ring-road experiments)
or *no trucks* (dimmed) do nothing with the current settings, but can still be
set in advance. Each parameter tab ends with a *Randomize* button for its settings
(Shift+X). Values show with the slider step's decimals, and a value between the
steps (a preset's 18.25 m radius, say) shows exactly. Hover over any control,
chart or readout to see what it means. What a key or a button just did (a saved
state, the seed drawn, a refused action) shows in the help box even with the
mouse resting on the view or a chart, for a few seconds and until the mouse
moves; on an empty ring (open mode) the readouts that average over the cars
show – and the charts a gap.

The **Presets** tab chooses a [real-world preset](#real-world-presets) with ◀ ▶
or a click on its title (hover it for its description and main source) and
starts it with *Apply preset*: the
roundabout is rebuilt with the preset's settings, keeping the seed, and the
charts get a marker. Below, it lists the preset's description, its settings,
the notes (caveats, and what check runs of this model give) and its sources;
the wheel scrolls. *Randomize all* (X) draws [random settings](#random-settings),
and the *Running* line says where the running settings come from: a preset or
random settings with their seed, and which settings were changed since. The
saved settings (D) and the session keep it.

Keys:

| Key | Action |
|---|---|
| Space | Pause / resume |
| R | Restart with a new seed |
| B | Brake a random car |
| + / - | Simulation speed |
| C | Car colours (speed, desired speed, lane, vehicle type) |
| S | Save a screenshot to `results/` |
| D | Save the chart data (last 10 min) as CSV, plus a settings file listing the settings, where they come from (preset or random seed), the live changes and the equivalent `experiments run` command (with `--preset` or `--random` when the settings come from one) |
| X | Random settings for every parameter, with a fresh seed shown in the help box (also *Randomize all* on the Presets tab) |
| Shift+X | Random settings for the tab shown only, linked to the current settings of the other tabs (also the *Randomize* button at the end of each tab) |
| V | Open the sweep viewer with the current settings as the base of its new sweeps (also *Sweep viewer*) |
| F5 | Save the state (simulation, charts and markers) as `results/state_<date>_<time>.npz` (also *Save state*) |
| F9 | Carry on from the most recently saved `results/state_*.npz` (also *Load state*); the view stays as it is |
| Esc | Quit, saving the session (see below) |

*Sweep viewer* (V) writes the current settings as
`results/sweep_base_<date>_<time>.json` (a flat `{parameter: value, "seed": ...}`
object) and starts `python -m roundabout_sim.sweep_viewer --base-settings <that file>
--results <the results folder>` in a window of its own: the viewer does not wait for it, and
pressing V again opens another. If one stops with an error, the help box says
so; what it printed is in `results/sweep_viewer.log`.

Command line of the viewer (`python -m roundabout_sim`):

| Option | What it does |
|---|---|
| `--preset NAME` | Start from a preset (`--list-presets`; the name or the title, case, `-`, `_` and spaces ignored). Parameter flags given with it override its values. |
| `--random SEED` | Start from plausible random settings drawn with this seed. Parameter flags given with it stay as given and the settings linked to them follow (`--random 40 --traffic_mode open` draws an open-mode start density). |
| `--list-presets` | Print every preset (name, description, settings, main source) and exit. |
| `--<parameter> VALUE` | One flag per parameter (`--density 80`, `--n-lanes 2`; `--help` lists them). Read as `experiments` reads them: a whole-number setting takes only whole numbers (`--n_lanes 2.5` is an error, not rounded), and a value outside the slider's range gets a note. |
| `--seed N` | The simulation's random seed, a whole number >= 0 (default: random). |
| `--size WxH` | Window size in pixels (default: 90 % of the screen; at least 1100x700 where the screen allows, also when a smaller size is given, with a note). |
| `--fresh`, `--no-autosave`, `--state PATH` | See [Saved sessions](#saved-sessions). |

### Saved sessions

Closing the viewer (Esc or the window's close button) saves the **session** to
`results/session/`, and the next launch carries on exactly where it stopped:
the same cars, queues and random draws, the same charts with their markers and
braked cars, the same view (tab, scroll position, sim speed, car and
space-time colours, paused or not), and the preset or random settings the
simulation came from. The help box says what was resumed.

| Command line | What happens |
|---|---|
| `python -m roundabout_sim` | resumes the saved session (a first launch starts afresh) |
| `--fresh` | ignores the saved session, simulation and view; quitting replaces it |
| `--no-autosave` | does not save a session on quit (`results/session/` keeps the last one) |
| `--state PATH` | carries on from a saved state file (one saved with F5, or `results/session/state.npz`); it takes no model settings |
| model settings, e.g. `--density 80`, `--seed 3`, `--preset etoile` or `--random 3` | start a new simulation with them; the view settings still restore |

**F5** (or *Save state*) saves the state at any moment as
`results/state_<date>_<time>.npz` and **F9** (or *Load state*) carries on from
the most recent one, keeping the view; the help box confirms both.

A saved state holds everything an exact resume needs: the settings and seed,
the clock, every vehicle (position, lane and lane-change animation, speed,
truck or car, the driver's temperaments, route, cooldowns and acceleration
history for the reaction time), the queues with each queued vehicle's arrival
time and type, the entrance cooldowns, every counter and event log behind the
rolling rates, the vehicles the density control still has to place, and the
state of every random number stream. Carrying on from it gives bit for bit
the run that never stopped (the tests check this for closed and open traffic,
trucks on snow, the 1,900-car ring and live changes). The chart history (10
minutes), the markers, the braked cars and the settings' origin (preset or
random seed) come with it.

**Files.** A state is one `.npz` file (numpy's zip of arrays) that is read with
`np.load(path, allow_pickle=False)`, so it never runs pickled code. Next to the
arrays it holds `meta_json`, a JSON text with the format name and version, when
it was saved, every setting, the clock, counters and random-stream states, and
the viewer's markers, view and settings' origin. `results/session/session.json`
repeats the settings, their origin and the view in readable, standard JSON (an
infinite density is written as `"inf"`). Files are written to a temporary file
and then moved into place, so a crash or a full disk never leaves half a file.
A missing, damaged or incompatible file (another format version, settings this
version does not know, arrays of the wrong shape) never stops the viewer: it
starts afresh with whichever of the file's settings are still valid (or the
defaults) and says so in the help box. A saved session it could not resume is
first moved to `results/session/not_resumed_<date>_<time>.npz`, so quitting
does not replace it (`--state` carries on from it with a version that reads
it). Chart samples the viewer cannot show (taken at another sampling interval
than its 0.5 s, or without a metric it draws) are left out with a note, and
the simulation still carries on from the file. F9 on an unreadable file
changes nothing.

From Python:

```python
from roundabout_sim import state
from roundabout_sim.engine import RingSim
state.save("my_state.npz", sim, recorder)      # the recorder (chart samples) is optional
sim = state.load("my_state.npz").simulation()   # carries on bit for bit
copy = RingSim.from_state(sim.state_dict())     # the same in memory
```

### Live charts

The **live charts** share one time axis:

| Chart | What it shows |
|---|---|
| Mean speed | Actual average speed against the average speed drivers want. The gap is the cost of congestion. |
| Congestion over time | **The headline congestion measure.** Jammed cars (% below 30 % of their free speed; rises steadily with density) and speed deficit (% below free speed). |
| Stop-and-go (instability) | How often a moving car is forced to (nearly) stop, per car per minute. It is about 0 in smooth flow and also in a standstill, so read it with Congestion: it says how unstable the flow is, not how congested. Its header splits the current jams by what their front car is doing (hover over it). |
| Flux | Cars per second past a cross-section, summed over lanes: for the whole ring (from speeds) and at the detector line (cars actually counted). |
| Congregation | Spacing (Gini) and order parameter, with a dashed line for what randomly placed cars would give. Only an order parameter well above that line means the cars bunch on one side. |
| Cars and queues (open mode only) | Cars on the ring and cars waiting at all entrances. Its header gives the entry delay over the last 30 s, its level of service and the longest queue. |
| Space-time diagram | Angle on the ring (vertical) against time (horizontal), coloured by speed. Jams are stripes, and their slope is the wave speed. Toggle to occupancy to see where cars pile up. |

Markers on the charts name each live change ("Density 50→80"), each preset
applied and each random draw with its seed.

## Changeable variables

All parameters live in [`roundabout_sim/config.py`](roundabout_sim/config.py),
which is the single source of truth for the sliders, the command-line flags,
the experiments CLI and the CSV columns. The tables follow the viewer's tabs.

**Road tab** (changing the layout rebuilds the roundabout; the road condition applies at once)

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| Lanes (`n_lanes`) | 3 | 1-5 | Number of concentric lanes on the ring. |
| Entrances / exits (`n_arms`) | 4 | 0-12 | Evenly spaced entry/exit pairs on the outer lane (0 = closed ring; up to 12, as at the Place Charles de Gaulle). |
| Ring radius (`radius`) | 80 m | 10-300 m, 1 m steps | Centreline radius of the innermost lane. Real single-lane roundabouts are about 12-21 m (28-46 m across), two-lane ones 18-28 m, the ring-road jam experiments 37-50 m. Presets keep their published radius between the steps (18.25 m, 36.6 m). |
| Car length (`car_length`) | 4 m | 1-8 m | Space each car takes up along the lane. |
| Road condition (`road_condition`) | dry | dry / wet / snow / ice | The road surface. Wet, snow and ice scale what the drivers do by factors from weather studies (see [How the model works](#how-the-model-works)): lower desired speed, acceleration, comfortable braking, cornering limit and braking limit, longer time gap, entry gap and follow-up time. The other sliders keep showing the dry values. |

**Traffic tab**

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| Density (`density`) | 50 veh/km/lane | 1-300 veh/km/lane | Cars on the ring per km of lane, capped at jam density. Raising it slots cars into gaps; lowering it removes cars. On a closed ring, waves appear from about 75-80. In open mode it sets only the cars at the start (and, moved live, the cars on the ring once, at that moment). |
| Exit chance (`exit_prob`) | 0.1 | 0-1 | Chance that a car leaves at each exit it passes. In closed mode each leaver is replaced by a car at a random entrance, or slotted into the ring if that queue is full (2 cars); in open mode leavers are gone for good. |
| Entry gap (`entry_gap`) | 2 s | 1.5-6 s | Smallest time gap to circulating traffic that an entering car accepts (the critical gap; see [Real-world presets](#real-world-presets) for how it relates to the HCM's critical headway). |
| Follow-up time (`follow_up`) | 1 s | 1-5 s | Shortest time between two cars entering from the same entrance (the follow-up headway). Real roundabouts show about 2.5-3.2 s. The next car may go on the first time step after it has passed, so the spacing is the follow-up time plus one step (1 s gives 1.1 s at the 0.1 s step); the default of 1 s keeps the model's original behaviour. Applies in both modes. |
| Turn speed (`turn_speed`) | 6 m/s | 2-15 m/s | Speed at which cars enter and leave. |

**Demand tab** (open traffic mode)

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| Traffic mode (`traffic_mode`) | closed | closed / open | *closed*: the number of cars is held at Density; each leaver is replaced at a random entrance. *open*: cars arrive at the entrances at random, queue, enter and leave for good, so the number on the ring varies. Changing it rebuilds the roundabout. |
| Entrance chance (`entry_rate`) | 0.1 /s | 0-0.5 /s | Open mode: mean arrivals per second at each entrance (Poisson). Multiply by 3600 for veh/h: 0.1 /s = 360 veh/h per entrance, 0.5 /s = 1800 veh/h. |
| Demand profile (`demand_profile`) | constant | constant / peak | Open mode. *peak*: the entrance chance follows 1 − 0.5 cos(2πt / 600 s), rising smoothly from half to 1.5 times its value and back every 10 minutes. Its mean is the entrance chance over each whole 10-minute cycle, not over shorter windows: the sweep's default 120-420 s window catches the rise to a peak and averages 1.3 times it. `run` and `sweep` print a note then, and `demand_per_min` gives the demand actually offered. |
| Demand imbalance (`demand_imbalance`) | 0 | 0-1 | Open mode: 0 = every entrance gets the same demand. Higher puts more on a main road through the bottom entrance (and the top one): an entrance's demand is scaled by 1 + imbalance × cos(2 × its angle from the bottom entrance), normalised so the mean per entrance stays the entrance chance. At 1, entrances at right angles to the main road get none. |
| Max queue (`max_queue`) | 50 cars | 1-200 cars | Open mode: cars that can wait at each entrance. Arrivals that find the queue full are turned away and counted as blocked (spill-back). Lowered live, the latest arrivals of longer queues give up (counted as given up, not as blocked). |

**Vehicles tab** (changing these rebuilds the roundabout, because a vehicle's length is set when it appears; Truck length only while there are trucks)

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| Truck share (`truck_share`) | 0 | 0-0.3 | Share of the vehicles that are trucks: each new vehicle is one by chance, also the ones queuing at the entrances. Trucks are 2.55 m wide, accelerate at 0.5× a car's rate, want 0.7× their driver's desired speed and keep a 1.5× longer time gap. Real shares are about 1-5 % in towns and 5-15 % on main roads. |
| Truck length (`truck_length`) | 12 m | 6-23 m | 9 m single unit, 12 m rigid lorry (the EU limit), 16.5 m articulated, 18.75 m road train, 22.4 m US WB-67. Capacity and jam density use the mean vehicle length; Density still counts vehicles. |

**Drivers tab**

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| Mean desired speed (`mean_speed`) | 12 m/s | 3-30 m/s | Average speed drivers want to drive on a free road. |
| Speed variance (std) (`speed_std`) | 1.5 m/s | 0-6 m/s | Spread of desired speeds between drivers (floored at half the mean). |
| Reaction time (`reaction_time`) | 0.4 s | 0-2 s | Delay between perceiving the traffic and acting on it. |
| Reaction time spread (std) (`reaction_std`) | 0 s | 0-0.5 s | How much reaction times differ between drivers. Each driver's own is drawn once around Reaction time, kept within 0-2 s and rounded to the 0.1 s step. Real drivers differ by about 0.2 s. |
| Reaction distance (`reaction_distance`) | 25 m | 10-100 m | How far ahead drivers pay attention. They react to every car within this distance (up to 5), so looking further ahead lets them ease off early and damps stop-and-go waves. Very short (10-15 m) and they notice the car in front too late and brake violently. |
| Following time gap (`time_headway`) | 1.2 s | 0.5-3 s | Desired time gap to the car in front. |
| Time gap spread (std) (`headway_std`) | 0 s | 0-0.8 s | How much the following time gap differs between drivers. Each driver's own is drawn once around Following time gap (at least half of it; trucks keep 1.5× theirs). Observed spread: about 0.3 s. |
| Standstill gap (`min_gap`) | 2 m | 0.5-5 m | Bumper-to-bumper gap kept in a stopped queue. |
| Max acceleration (`max_accel`) | 1.2 m/s² | 0.3-4 m/s² | How briskly drivers speed up. |
| Comfortable braking (`comfort_decel`) | 2 m/s² | 0.5-5 m/s² | Braking drivers are happy to use. |
| Driver noise (`driver_noise`) | 0.35 m/s² | 0-1.5 m/s² | Random fluctuations in each driver's acceleration. More noise brings waves on earlier. |
| Cornering limit (`lateral_accel`) | 4 m/s² | 1-10 m/s² | Comfortable sideways acceleration; it makes tight inner lanes slower. |

Each driver keeps one draw for each spread (a temperament, as for the desired
speed), so moving a spread, a driver setting or the road condition live changes
every driver at once, and each keeps their place in the spread.

**Lane chg tab and Start tab**

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| Lane-change chance (`lane_change_rate`) | 0.2 /s | 0-1 /s | How often (per second) a driver considers changing lane. |
| Lane-change motive (`lane_change_motive`) | advantage | advantage / random | *advantage*: change only when it is faster (the MOBIL model); *random*: change into any safe gap. |
| Politeness (`politeness`) | 0.3 | 0-1 | How much drivers weigh the braking they force on others. |
| Initial spacing (`init_spacing`, Start tab) | uniform | uniform / random / clustered | The average gap is set by Density. *uniform*: evenly spaced; *random*: random gaps; *clustered*: one packed queue across all lanes. Near the jam threshold an uneven start can lock in jams; elsewhere the effect fades within 1-2 min. Changing it rebuilds the roundabout. |

Three settings have no slider and keep their defaults unless set with
`--set` in experiments: lane width 3.5 m (`lane_width`), car width 1.8 m
(`car_width`) and the time step 0.1 s (`dt`).

## Real-world presets

```powershell
python -m roundabout_sim --list-presets                 # name, description, settings and main source of each
python -m roundabout_sim --preset sugiyama2008          # the viewer, starting from a preset
python -m roundabout_sim --preset us_single_lane --entry_rate 0.16   # parameter flags go on top
python -m roundabout_sim.experiments run --preset stern2018 --seed 2
python -m roundabout_sim.experiments sweep --preset tadaki2013 --x density=32:108:4 --seeds 3
```

A preset is a set of settings taken from a published experiment, a design
guide or traffic counts. It starts from the model defaults and changes only
its own values, so it gives the same model whatever was set before (only the
seed is kept); explicit flags and `--set` then go on top. Names ignore case,
`-`, `_` and spaces, and the title works too (`--preset "US two-lane (HCM 6)"`).
Every value is traceable: `roundabout_sim/presets.py` gives the source of each
in a comment or marks it *derived* (arithmetic on sourced numbers),
*assumption* or *model*, and each preset's notes (on the viewer's Presets tab
and in `--list-presets`) give the caveats and what the check runs of this model
show. Values keep their published precision even between a slider's steps, and
the viewers show them exactly (the Sugiyama ring's radius 36.6 m and 95.7
veh/km).

| Preset | What it is | Main settings | Main sources |
|---|---|---|---|
| `default` | The model's own defaults (bit-identical to running without a preset). | - | - |
| `sugiyama2008` | 22 cars on a 230 m single-lane circuit: a jam forms with no bottleneck and moves backward at about 20 km/h. | 1 lane, no exits, radius 36.6 m, 95.7 veh/km, 11 m/s | [Sugiyama et al. 2008, New J. Phys. 10, 033001](https://ir.library.osaka-u.ac.jp/repo/ouka/all/93262/NewJPhys_10_033001.pdf) |
| `tadaki2013` | 30 cars on the 314 m Nagoya Dome circuit (jammed); sweep Density 32-108 for the free-to-jam transition, measured between 80 and 89 veh/km. | 1 lane, radius 50 m, 95.5 veh/km, 3.9 m cars | [Tadaki et al. 2013, New J. Phys. 15, 103034](https://ir.library.osaka-u.ac.jp/repo/ouka/all/93261/NewJPhys_15_103034.pdf) |
| `stern2018` | 21 cars on a 260 m track in the US: stop-and-go waves 55-79 s after the start (161 s in a run with 22 cars). | 1 lane, radius 41.4 m, 80.8 veh/km, 4.9 m cars | [Stern et al. 2018, Transp. Res. C 89](https://arxiv.org/abs/1705.01693) |
| `us_single_lane` | Typical US single-lane roundabout (40 m across, 4 legs) at a moderate 432 veh/h per approach. | open, 1 lane, radius 18.25 m, 0.12 /s, imbalance 0.2, exit 0.5, entry gap 2.5 s, follow-up 2.5 s, 11 m/s, cornering 3 m/s² | [NCHRP Report 672](https://www.aldot.gov/publications/Design/pdf/TrafficSafetyOp/NCHRP672Roundabouts.pdf); HCM 6 via [FHWA-SA-15-070](https://rosap.ntl.bts.gov/view/dot/49387) |
| `us_single_lane_offpeak` | The same at off-peak demand, 216 veh/h per approach (LOS A). | as above, 0.06 /s | as above; [Giuffrè et al. 2016](https://doi.org/10.3389/fbuil.2016.00028) |
| `us_single_lane_peak` | Peak hour at the single-lane service limit, 576 veh/h per approach, with trucks and varied drivers. | as above, 0.16 /s, peak profile, 5 % trucks of 16 m, reaction spread 0.25 s, time gap spread 0.3 s | as above; [FHWA VM-1](https://www.fhwa.dot.gov/policyinformation/statistics/2022/vm1.cfm), [NCHRP RR 1061](https://nap.nationalacademies.org/read/27236/page/38) |
| `us_single_lane_wet` | The single-lane roundabout on a wet road in rain (about 10 % longer accepted gaps). | `us_single_lane`, road wet | [Lee et al. 2018](https://pure.uos.ac.kr/en/publications/evaluation-of-the-rain-effects-on-gap-acceptance-behavior-at-roun/); [Rakha et al. 2010](https://rosap.ntl.bts.gov/view/dot/4030/dot_4030_DS1.pdf) |
| `us_single_lane_snow` | The same on packed snow. A Hokkaido test roundabout lost about 300 veh/h of entry capacity on compacted snow; the model loses about 130 at saturation (-19 %). | `us_single_lane`, road snow | [Munehiro et al. 2017](https://trid.trb.org/view/1437414); [Martin/Perrin et al. 2000](https://www.ugpti.org/resources/reports/downloads/mpc01-120.pdf) |
| `us_two_lane` | US two-lane roundabout (55 m across), one-lane entries onto two circulating lanes, 648 veh/h per approach (v/c about 0.8). | open, 2 lanes, radius 22.25 m, 0.18 /s, entry gap 2.2 s, follow-up 2.4 s | NCHRP Report 672; HCM 6 |
| `etoile` | The Place Charles de Gaulle (Paris) approximated: 12 avenues, 166 m across, about 4,750 veh/h entering. | open, 5 lanes, 12 arms, radius 67.25 m, 0.11 /s, imbalance 0.6, exit 0.17, entry gap 1.5 s, follow-up 1.9 s, 8.5 m/s | [OpenStreetMap](https://www.openstreetmap.org/way/184551793) circle fits; [Paris open-data counts](https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents/); [Lecuyer 2024](https://www.catherinelecuyer.fr/post/des-embouteillages-font-leur-apparition-sur-le-rond-point-de-etoile) |

Two findings shape the presets:

- **Real drivers.** With the default drivers the ring experiments mostly stay
  smooth. Every real-world preset therefore uses one calibrated driver set:
  following time gap 0.9 s and standstill gap 1.5 m (means of calibrated
  car-following studies, [Zhu et al. 2018](https://arxiv.org/abs/1811.06395),
  [Kesting & Treiber 2008](https://arxiv.org/abs/0803.4063)), reaction time
  0.6 s (between the medians 0.50 s and 0.66 s of Zhu 2018 and Johansson &
  Rumar 1971) and driver noise 0.3 m/s² (the noise intensity that fits
  [Treiber & Kesting 2017](https://arxiv.org/abs/1708.06952)'s platoon
  experiment). With it the three rings form stop-and-go waves within 1-6
  minutes, and the Tadaki ring's free-to-jam transition falls between 80 and
  89 veh/km, as measured.
- **Entry gap is calibrated to this model.** The engine lets a car merge when
  the bumper gap to the outer-lane car behind is at least Entry gap × its
  speed (plus clearances), so the HCM critical headway used literally gives
  about 40 % too little entry capacity (about 400 instead of 685 veh/h
  single-lane). The design-guide presets use the value that reproduces the
  HCM 6 capacity (2.5 s single-lane: about 660 veh/h; 2.2 s one lane against
  two: about 750) and give the literal HCM value (5.0 / 4.3 s) in their notes.
  Follow-up times are the HCM follow-up headways less one 0.1 s time step
  (2.5 s single-lane, 2.4 s two-lane), because the engine lets the next car go
  on the first step after the follow-up time: queued cars then enter 2.6 /
  2.5 s apart, as in HCM 6.

What the check runs of this model show. Open presets: seeds 0-15, 20 min
after 5 min of warm-up, the median and the range across the seeds, as
`python -m roundabout_sim.experiments sweep --preset NAME --x entry_rate=RATE --seeds 16 --warmup 300 --measure 1200`
gives them (single seeds vary a lot near capacity). Rings: seeds 1-3, 10 min.

| Preset | Result |
|---|---|
| `sugiyama2008` / `tadaki2013` / `stern2018` | Stop-and-go waves (the speed spread across the cars above 2.5 m/s, Stern's detector) after 63-339 / 55-89 / 100-133 s. The jammed rings carry about 35 % less flow than Stern measured (about 1,200 against 1,755-1,828 veh/h). |
| `us_single_lane` | All of the demand enters; queue wait 4.8 s (3.4-6.7), longest queue 7 cars (5-10) (HCM 6: v/c 0.56, LOS A-B). |
| `us_single_lane_offpeak` | Queue wait 1.1 s (0.9-1.6), longest queue 3 cars (2-5). |
| `us_single_lane_peak` | At capacity: queue wait 72 s (42-114, LOS F in 13 of 16 seeds), longest queue 42 cars (28-50); in 5 of 16 seeds a queue reaches the 50-car limit and turns arrivals away. The 5 % trucks of 16 m do it: without them 44 s and no full queue. |
| `us_single_lane_wet` / `_snow` | Queue wait 6.5 s (4.4-10.9) / 15.7 s (8.7-32.6) against 4.8 s dry; longest queue 9 / 13 cars against 7. At saturation the model loses 8 % / 19 % of the entry capacity. |
| `us_two_lane` | All of the demand (644 veh/h per entrance) enters; queue wait 12.7 s (8.4-20.6, mostly LOS B; HCM 6 control delay about 23 s). |
| `etoile` | 377 of 395 veh/h per entrance enter; queue wait 52 s (34-85, LOS F); in 8 of 16 seeds a main-axis queue reaches the 50-car limit. The main axis gets 634 veh/h per entrance (the real Grande Armée and Foch 870-1,030, so the model under-represents them by about 30 %), but with exit chance 0.17 about 1,900 veh/h circulate past each entrance, more than a one-lane entry can merge that into; with exit chance 0.3 the queues stay below 20 cars. The real place has jammed too since its lanes were cut from 12 to 8. |

## Random settings

```powershell
python -m roundabout_sim --random 3                      # the viewer with random settings (seed 3)
python -m roundabout_sim --random 3 --n_lanes 1          # one lane: the radius and demand drawn for it
python -m roundabout_sim.experiments run --random 3      # the same settings, headless
python -m roundabout_sim.experiments sweep --sample 50 --seed 1                                 # 50 random points (x 3 seeds = 150 runs) over the plausible ranges of every setting
python -m roundabout_sim.experiments sweep --sample 20 --sample-groups Drivers --seed 1         # only the Drivers tab's settings
python -m roundabout_sim.experiments sweep --sample 40 --vary density=20:100 --vary reaction_time=0.3:1.2 --seed 1
```

**X** (or *Randomize all* on the Presets tab) gives every setting a plausible
random value; **Shift+X** (or the *Randomize* button at the end of a tab) only
the settings of the tab shown. Each press draws a fresh seed, which the help
box shows and the saved settings (D) and the session keep; for **X**,
`--random SEED` gets the same settings back (a Shift+X draw also depends on the
settings of the other tabs). Randomizing rebuilds the roundabout when a
rebuilding setting changes (keeping the simulation's seed), else it changes the
drivers live; either way a chart marker names it with its seed. A Shift+X draw
on top of a preset (or of an X draw) keeps it as its base: the view's header
and the saved settings name both ("preset etoile (Place Charles de Gaulle) with
random Drivers settings, seed 77"), and the saved command is `--preset etoile`
with the drawn settings as `--set`. Shift+X draws of other tabs add up, each
named with its seed ("... with random Drivers settings, seed 77, and Road
settings, seed 412"); a tab drawn again replaces its earlier draw. Settings
given as flags with `--random` are named too ("random settings, seed 40
(given: Density)"), and changing one later counts as a change. The values are
not drawn uniformly over the slider ranges (which allow extremes on purpose)
but from real-world distributions (`presets.RANDOM_RANGES`, each with its
source), always inside the slider range and on the slider's step. Most are
triangular (lo, most often, hi):

| Setting | Distribution | Values |
|---|---|---|
| Lanes (`n_lanes`) | choice | 1 40 %, 2 35 %, 3 15 %, 4 5 %, 5 5 % |
| Entrances / exits (`n_arms`) | choice | 3 25 %, 4 60 %, 5 10 %, 6 5 % (no closed rings: an experiment, not a roundabout) |
| Ring radius (`radius`) | triangular | by lane count: 12-21 m (1 lane, typically 18), 18-45 m (2, typically 22), 22-41 m (3, typically 29), 33-70 m (4, typically 45), 34-70 m (5, typically 50) |
| Car length (`car_length`) | triangular | 4-5 m, typically 4.5 |
| Road condition (`road_condition`) | choice | dry 75 %, wet 17 %, snow 6 %, ice 2 % |
| Density (`density`) | triangular | closed mode 5-110 veh/km/lane, typically 35; open mode (only the start) 5-50, typically 20; below 85 % of the jam density |
| Exit chance (`exit_prob`) | uniform | 2 / entrances × 0.75-1.25 (a mean trip of half the circle), within 0.1-0.8 |
| Entry gap (`entry_gap`) | triangular | 1.5-3.5 s, typically 2.5 (one lane) or 2.2 (more) |
| Follow-up time (`follow_up`) | triangular | 1.8-3.1 s, typically 2.5 (field headways 1.9-3.2 s, typically 2.6, less the 0.1 s step the engine adds) |
| Turn speed (`turn_speed`) | triangular | 4-9 m/s, typically 6 |
| Traffic mode (`traffic_mode`) | choice | closed 25 %, open 75 % |
| Entrance chance (`entry_rate`) | triangular | a degree of saturation 0.1-1.0 (typically 0.45) of the layout's HCM 6 entry capacity on the drawn road (× 1 dry, 0.92 wet, 0.81 snow, 0.70 ice, as the model measures) with trucks counted as 2 cars, within 0.03-0.22 /s |
| Demand profile (`demand_profile`) | choice | constant 70 %, peak 30 % |
| Demand imbalance (`demand_imbalance`) | triangular | 0-0.7, typically 0.2 |
| Max queue (`max_queue`) | triangular | 20-100 cars, typically 50 |
| Truck share (`truck_share`) | triangular | 0-0.15, typically 0.05 |
| Truck length (`truck_length`) | triangular | 9-18.75 m, typically 12 (the default when there are no trucks) |
| Mean desired speed (`mean_speed`) | triangular | 8-14 m/s, typically 11 |
| Speed variance (`speed_std`) | triangular | 0.8-2.2 m/s, typically 1.4 |
| Reaction time (`reaction_time`) | triangular | 0.3-1.2 s, typically 0.6 |
| Reaction time spread (`reaction_std`) | triangular | 0-0.35 s, typically 0.2, at most 0.52 × the reaction time |
| Reaction distance (`reaction_distance`) | triangular | 15-60 m, typically 25 |
| Following time gap (`time_headway`) | triangular | 0.7-1.8 s, typically 1 |
| Time gap spread (`headway_std`) | triangular | 0-0.45 s, typically 0.3, at most 0.4 × the time gap |
| Standstill gap (`min_gap`) | triangular | 1-3 m, typically 1.75 |
| Max acceleration (`max_accel`) | triangular | 0.8-2 m/s², typically 1.3 |
| Comfortable braking (`comfort_decel`) | triangular | 1-3 m/s², typically 2 |
| Driver noise (`driver_noise`) | triangular | 0.15-0.45 m/s², typically 0.28 |
| Cornering limit (`lateral_accel`) | uniform | the NCHRP 672 speed-radius relation at the radius × 0.85-1.15, within 2-4 m/s² |
| Lane-change chance (`lane_change_rate`) | triangular | 0.05-0.5 /s, typically 0.2 |
| Lane-change motive (`lane_change_motive`) | choice | advantage 85 %, random 15 % |
| Politeness (`politeness`) | triangular | 0-0.6, typically 0.2 |
| Initial spacing (`init_spacing`) | choice | uniform 50 %, random 40 %, clustered 10 % |

Linked settings use the values drawn before them (or, when only one tab is
randomized, the current ones), and settings given explicitly with `--random`
(a parameter flag, or `--set` in experiments) stay as given, with the linked
ones following them: `--random 40 --traffic_mode open` starts at an open-mode
density, where `--random 40` alone draws a closed ring at 76 veh/km/lane. Each
setting has a random stream of its own, so on randomizing one tab its
unlinked settings get the same values as a full randomize with the same seed;
the five linked to other tabs (density, exit chance, entry gap, entrance
chance, cornering limit) follow the current settings of those tabs. Settings
that act only in the other traffic mode are drawn too, for when you switch.

`experiments sweep --sample N` runs a **Latin hypercube** design over the
`--vary` ranges, or without `--vary` over the plausible ranges of the table
above (`--sample-groups G1,G2`: only those tabs; settings given with `--set`
or a flag stay fixed). See [Sweeps of many settings](#sweeps-of-many-settings).
The design covers the ranges evenly, so the weights and the links of the
random settings do not apply (a one-lane ring may get a 70 m radius): use
`--random SEED` runs for linked settings.

### Python API

```python
from roundabout_sim import presets

presets.names()                         # ['default', 'sugiyama2008', ...] in display order
p = presets.get("US-single-lane")       # Preset(name, title, description, sources, changes, notes)
cfg = presets.apply("etoile", base)     # SimConfig: the defaults + the preset, base's seed kept
presets.describe("etoile", width=80)    # title, description, settings, notes and sources (ASCII)
presets.list_text()                     # all presets, for --list-presets
presets.find(cfg)                       # the preset a SimConfig is, or None

presets.random_changes(3)               # {parameter: value} for every slider, reproducible by seed
presets.random_changes(3, ["Drivers"], base=cfg)   # only one tab's settings, linked to cfg
presets.random_changes(3, fixed={"traffic_mode": "open"})   # kept as given; the links follow it
presets.random_config(3, base=base)     # base with random_changes applied (base's seed kept)
presets.new_seed()                      # a short fresh seed to show and reuse
presets.latin_hypercube(20, {"density": (20, 100), "n_lanes": (1, 3), "road_condition": ["dry", "wet"]}, seed=1)
presets.plausible_ranges()              # RANDOM_RANGES as latin_hypercube ranges (unlinked; sweep --sample)
```

## How congestion is measured

There is no single "congestion" number, so the model reports a few that answer
different questions.

**How congested is it?**

- **Jammed cars** (`jam_fraction`) is the headline measure. It is the share of
  cars crawling below 30 % of their free speed, in queues and in stop-and-go
  waves alike, and it rises steadily with density.
- **Speed deficit** (`speed_deficit`) is how far below free speed the cars
  drive in total.
- **Time to congestion** (`congestion_onset`, experiments only) is the first
  time the 30 s mean of jammed cars exceeds 30 %, counted from t = 0. A start
  that is already jammed scores about 0.

**How unstable is it?**

- **Stop-and-go index** (`stops_per_car_min`) counts stops per car per minute
  over the last 30 s. A *stop* is a car that had been moving (above 60 % of
  the speed it is aiming for) dropping below 30 % of it. Slowing down to turn
  off at an exit doesn't count.
  - In smooth flow, however dense, it is about 0.
  - In a standstill it is also about 0, because nobody is moving to be stopped.
  - In stop-and-go waves it is about 1-2.
- **Time to first stop-and-go** (`stop_go_onset`, experiments only) is the
  first time the index, counted over a full 30 s window, exceeds 0.5. That
  means a quarter of the cars were forced to stop within 30 s.

**What holds the jams up?**

- **Number of jams** (`n_jams`) counts runs of two or more jammed cars in a
  lane.
- Each jam is classified by what its front car is doing now:
  - *following*: just following the car ahead;
  - *merge*: a leaver moving out to its exit;
  - *exit*: slowing down to leave;
  - *entry*: just came in;
  - *other*: braked by you, or slotted in by the density control.
- The **following share** (`following_jam_share`) is only an **upper bound**
  on natural (phantom) jams. A wave set off at an exit keeps travelling
  backwards with ordinary followers at its front. With the defaults it is
  about 70-85 %, yet the same run with exit chance 0 has no jams at all.
- The honest test of what the entrances and exits cause is
  `experiments run --baseline`. Its `arm_jam_share` is
  1 − jammed(no exits) / jammed. Alternatively, sweep with
  `--series exit_prob=0,0.1`.

**Other dynamics**

- **Jam wave speed** (experiments) is estimated from the space-time diagram.
  Negative means the wave moves backwards against the traffic. The model gives
  about -2 to -4 m/s; real stop-and-go waves are about -4 to -5.5 m/s
  (-15 to -20 km/h).
- **Harsh braking** (`emergency_per_min`) counts episodes per minute. An
  episode is either the driver model asking for more than the braking limit
  (an over-reaction after a late or delayed response), or the no-overlap
  constraint stopping a car harder than that. The limit is 9 m/s² on a dry
  road, 7.2 when wet, 2.7 on snow and 1.35 on ice.
  - With the default drivers it is 0 on a ring without exits, and a few per
    minute from merging at the default 4 entrances/exits.
  - It grows sharply with shorter reaction distance, longer reaction time and
    higher density, and on snow and ice, where comfortable braking is close to
    the limit (see *Road conditions* below).

**Space and flow**

- **Flux** (`flux`, `detector_flux`): cars per second through a
  cross-section, both for the whole circle (Σ speed / lane length) and as
  counted at the detector line.
- **Area covered** (`occupancy`): the share of the road surface covered by
  vehicles, at their real lengths and widths.
- **Trucks** (`trucks`, `truck_speed`, `car_speed`): the trucks on the ring and
  the mean speed of the trucks and of the other vehicles (empty without them).
- **Spacing Gini** (`spacing_gini`): the Gini coefficient of the road length
  each car has (bumper gap ahead + the mean vehicle length, a car's without
  trucks, so a mix of cars and trucks alone does not raise it).
  - 0 means the road is shared evenly.
  - About 0.35 means random gaps.
  - Values towards 1 mean packed queues or platoons.

  It is good for initial spacing and platoons, but it is not a jam detector.
- **Order parameter** (`order_param`, `order_angle`): |mean of e^(iθ)| over
  all cars.
  - 0 means the cars are balanced around the ring; 1 means they are all on one
    side.
  - `order_angle` gives the side where they bunch up.
  - `order_baseline` = √(π/4N) is what N randomly placed cars give, so only
    values well above it (the viewer uses 2×) mean real one-sidedness.
- **Clustering** (`clustering`): the spread of car counts over 6-24 sectors,
  after subtracting random scatter. 0 means even or random placement, and about
  1 means half the ring is empty. It is often 0 in dense flow.

**Entrances and queues** (most useful in open mode)

- **Cars on the ring** (`cars`) and **queued** (`queued`: at all entrances;
  `longest_queue`: at the longest one; `peak_queue` in experiments: the longest
  queue seen in the measurement window). In open mode the number on the ring
  varies.
- **Arrivals** (`arrivals_per_min`): cars arriving at the entrances. In open
  mode every random arrival counts, blocked ones included; in closed mode it
  is the leavers' replacements that join a queue (the rest are slotted into the
  ring). `demand_per_min` is what the open-mode demand settings ask for (in
  the summaries: its mean over the window). Every arrival ends up entered,
  still queued, blocked, or given up (dropped from a queue when Max queue or
  Density is lowered live).
- **Entries** (`entries_per_min`): cars entering the ring from the queues.
- **Blocked** (`blocked_per_min`, open mode): arrivals that found their queue
  full (Max queue) and were turned away, i.e. spill-back onto the approach.
- **Entry delay** (`mean_entry_delay`): the mean queue wait of the cars that
  entered, from joining the queue to entering the ring (empty if none
  entered). It comes with an HCM-style **level of service**: A ≤ 10 s,
  B ≤ 15 s, C ≤ 25 s, D ≤ 35 s, E ≤ 50 s, F above (the HCM 6th edition's
  roundabout thresholds). It is only the wait in the queue, not the HCM
  *control delay*, which also counts slowing down to the yield line and
  speeding up again, so it reads somewhat better than an HCM analysis of the
  same traffic. As the HCM does for an entry lane, the letter is F whenever
  the demand exceeds the capacity (v/c > 1, see below), however short the
  wait of the cars that got in: with a short Max queue turning arrivals away,
  the wait alone would hide that. The letter (`entry_los`) is in the viewer,
  the printed summaries and the summary and sweep CSVs, not in the time
  series. Cars still queued when a window ends are not in its mean, so in an
  oversaturated run read it together with the queues.
- **Demand / capacity** (`entry_vc`, open mode): v/c of the busiest entrance,
  its arrivals per second (blocked ones included) over its entry rate while it
  had a queue. Below 1 it is about the share of the time the entrance had a
  queue; above 1 arrivals outrun what it lets in, so its queue grows or turns
  them away. The viewer and the time series use the last 5 minutes (over
  30 s it would hang on a few cars), the summaries the measurement window.
- **Throughput** (`throughput_per_h`): cars leaving the ring per hour.

## Batch experiments

```powershell
# one run: dashboard (space-time diagram, congestion over time, jam causes, ...) + CSVs
python -m roundabout_sim.experiments run --set n_lanes=1 --set n_arms=0 --set density=80 --duration 600

# how much of the jamming do the entrances/exits cause? (repeats the run with exit_prob = 0, same seed)
python -m roundabout_sim.experiments run --baseline --duration 300

# runs from a real-world preset and from random settings
python -m roundabout_sim.experiments run --preset sugiyama2008 --duration 300
python -m roundabout_sim.experiments run --random 3 --traffic-mode open    # flags take part in the random links

# sweep: any parameter on x, optionally one line per value of another; seeds run in parallel
python -m roundabout_sim.experiments sweep --x reaction_distance=10:60:10 --series density=70,80,90 --set n_arms=0 --set n_lanes=1

# open demand: queues, blocked arrivals and entry delay against the entrance chance, for two follow-up times
# (in the printed table and the CSV)
python -m roundabout_sim.experiments sweep --x entry_rate=0.02:0.2:0.02 --series follow_up=1,3 --set traffic_mode=open

# trucks and road surface: the truck share against dry, wet, snowy and icy roads, with varied drivers
python -m roundabout_sim.experiments sweep --x truck_share=0:0.3:0.05 --series road_condition=dry,wet,snow,ice --set reaction_std=0.2 --set headway_std=0.3

# three or more settings: a full grid; the first two are the figure's x and series, the rest averaged over
python -m roundabout_sim.experiments sweep --vary density=20:100:20 --vary n_lanes=1,2 --vary road_condition=dry,wet,snow

# from a real-world preset (flags and --set override it; the swept values override those)
python -m roundabout_sim.experiments sweep --preset us_single_lane --x entry_rate=0.04:0.24:0.04 --series follow_up=2,3.2
python -m roundabout_sim.experiments --list-presets

# one run per row of a CSV (columns = parameter names, optional seed; empty cells keep the base setting)
python -m roundabout_sim.experiments sweep --plan my_plan.csv --seeds 2

# random designs: a Latin hypercube over ranges, or over the plausible ranges of every parameter (or of some tabs)
python -m roundabout_sim.experiments sweep --sample 40 --vary density=10:120 --vary reaction_time=0.2:1.2 --vary n_lanes=1,2,3
python -m roundabout_sim.experiments sweep --sample 60 --sample-groups "Drivers,Lane chg" --seed 5

# browse sweeps in the sweep viewer (the most recent in results/, or a given one)
python -m roundabout_sim.experiments view
python -m roundabout_sim.experiments view results/sweep.csv --size 1600x900
```

In open mode the run summary also prints the arrivals, entries, blocked
arrivals, demand / capacity, entry delay with its level of service,
longest queue and throughput, and the dashboard's events panel adds arrivals,
entries and blocked arrivals. `run --baseline` is skipped in open mode (with exit chance 0
the arrivals would never leave). The sweep's printed table adds the entry
delay and its level of service, and in open mode the cars, queues, demand,
blocked arrivals, demand / capacity and throughput. With trucks the run summary
and the sweep table add the trucks on the ring and the mean speeds of cars and
trucks.

**Run dashboard in open mode.** With `traffic_mode=open` the `experiments run`
dashboard gets a fifth row for the entrances: the cars on the ring, queued at
all entrances and in the longest queue over time (with Max queue per entrance
as a line while the longest queue gets near it), and the entry delay over time with the
HCM-style level-of-service limits as dotted lines, its late-half value and
letter on the title line. The speed histogram marks the drivers' measured mean
desired speed (with trucks and road condition), as the mean speed chart and the
printed summary give it.

### Sweeps of many settings

`experiments sweep` has three designs. Every design point runs `--seeds` times
(seeds `--seed`, `--seed` + 1, ...), in parallel on `--workers` processes.

| Design | Flags | Points |
|---|---|---|
| grid | `--x`, `--series`, `--vary NAME=SPEC` (repeatable) | Every combination of the values (full factorial). The order is `--x`, `--series`, then the `--vary` flags as given: the 1st setting is the figure's x axis, the 2nd its series (at most 8 values, one colour each), the others are averaged over in the figure. `--vary` alone works too (`--vary a=.. --vary b=..` is the same sweep as `--x a=.. --series b=..`). SPEC = `start:stop:step` (stop included) or a comma list. |
| plan | `--plan FILE.csv` | One point per row. Columns are parameter names (an unknown one is an error that lists the valid names; a name given twice is an error too) plus an optional `seed` column, that row's base seed. An empty cell keeps the base setting, and so do the cells missing at the end of a row shorter than the header (a note names such rows); a column with neither a name nor a value (a trailing comma) is ignored. Columns with more than one value are the varied settings; the first is the x. Rows with the same settings and seed would run the same simulation twice: a note names them (give each row its own seed, or use `--seeds` for repeats). A plan in which nothing varies gets a figure of its runs against their number, without a mean line. Errors name the file's line. |
| sample | `--sample N` | N random points of a Latin hypercube, reproducible by `--seed`. With `--vary` (or `--x` / `--series`) over the given settings: `start:stop` (a step, if given, is not used) is a range split into N equal strata with one point in each, on the slider step; when the range holds no more slider values than N, those values are dealt out evenly instead (each N // m or N // m + 1 times), as is a comma list. Without `--vary`, over the plausible ranges of every parameter (`presets.plausible_ranges()`, the [random settings](#random-settings) table), or only of `--sample-groups` (the viewer's tabs: Road, Traffic, Demand, Vehicles, Drivers, Lane chg, Start): each range cut into N strata, the choices (lanes, entrances, road condition, ...) dealt out evenly. Parameters given with `--set` or their flag stay fixed. A `--preset` or `--random` base whose every setting the sample redraws is an error (use `--sample-groups` to keep some). |

**Base settings.** Every run starts from the defaults, then `--preset NAME`
(a real-world preset, `--list-presets` lists them) or `--random SEED`
(plausible random values of every parameter, with `--set` and the parameter
flags taking part in the links), then `--set` and the parameter flags, then
the values the sweep varies. The same `--preset`, `--random` and
`--list-presets` work for `experiments run`; its `PREFIX_summary.csv` then has a
`preset` or `random` column, and the dashboard's subtitle names it. Seeds
(`--seed`, `--random`, a plan's `seed` column) are whole numbers >= 0, and a
whole-number setting (lanes, entrances, max queue) takes only whole numbers
(`n_lanes=2.5` is an error, not rounded): every such mistake is reported
before anything runs.

**Printed output.** The first line gives the number of runs (`16 runs (2 density
x 2 n_lanes x 2 exit_prob x 2 seeds) ...`); a note follows when a varied
setting does nothing in any of the runs (an open-mode setting such as
`entry_rate` swept with every run closed, the entrance settings on a ring
without entrances, `truck_length` without trucks: every value would give the
same runs), then one progress line per run
(`[  3/16] exit_prob=0.0 n_lanes=1 density=40.0 seed=0: flux ...`; plans and
samples say `row 3:` / `point 3:` and show up to four settings). Grids and plans
end with the means across seeds, one row per combination of the varied
settings, and, when exit_prob is varied with 0 among its values, the share of
the jamming due to the entrances/exits (also with exit_prob as one of three or
more settings). A random sample instead prints the spread of the headline
metrics and the rank correlation (Spearman) of each varied setting with them,
strongest first. The tables show three significant digits, whole numbers from
1000 up (`1650` veh/h, not `1.65e+03`).

| Command | Output | Contents |
|---|---|---|
| `run` | `PREFIX.png` | Dashboard |
| | `PREFIX_timeseries.csv` | One row per sample (default every 0.5 s) |
| | `PREFIX_summary.csv` | Settings plus metrics over the second half. With `--baseline` it also has `baseline_*` columns and `arm_jam_share`; with `--preset` / `--random` a `preset` / `random` column. |
| | `PREFIX_spacetime.npz` | Space-time and occupancy arrays |
| `sweep` | `PREFIX.csv` | One row per run: the varied settings first (the last one first, so the x is the column just before `seed`; for one or two settings exactly as before: series, x, seed), the seed, every other setting, then the metrics averaged over the measurement window (event rates and the entry delay count exactly the events inside it). Rows in run order: the x varies fastest, the seeds within each point. |
| | `PREFIX_meta.json` | The design, for the sweep viewer (format below). |
| | `PREFIX.png` | Grid: the flux (the fundamental diagram when x is density) and twelve more metrics against the x (speed, jammed %, stop-and-go, onsets, jams, following share, harsh braking, wave speed, spacing Gini, clustering, order and lane changes), one line per series value, mean with a min-max band. With three or more settings each point pools the runs of every value of the others (the subtitle says which). In open mode a fifth row shows the entrances: mean entry delay (with the HCM level-of-service limits as dotted lines, the letters beside the chart), throughput, the longest queue (mean over time, and the longest seen dashed) and blocked arrivals. The harsh-braking chart names the braking limit on every road condition. Plan or sample: one row per varied setting (at most 8: those with the strongest rank correlation) and one column per headline metric; each dot is a run, the line the mean per value or per equal-width bin of the setting. |
| | `PREFIX_heatmap.png` | Grids of two or more settings: x by series cells coloured by the mean of the headline metrics (flux, jammed cars, stop-and-go; in open mode also the entry delay and the throughput), one colour bar each (light = low, dark = high). Text values keep the order given, as in `PREFIX.png`. The cells show their values at the largest size that fits every cell (none when even the smallest does not). A sweep that draws no heatmap removes a `PREFIX_heatmap.png` left by an earlier sweep with the same prefix. |

Settings columns are named like the parameters. A measurement that shares a
name gets a `measured_` prefix, for example `measured_density`. Runs are
deterministic for a given seed.

`PREFIX_meta.json` (format version 1):

```json
{
 "version": 1,
 "design": "grid",
 "varied": [{"name": "density", "values": [20.0, 40.0]}, {"name": "n_lanes", "values": [1, 2]}],
 "fixed": {"n_arms": 0},
 "preset": null,
 "random": null,
 "sample_groups": null,
 "seeds": 2,
 "base_seed": 0,
 "warmup": 120.0,
 "measure": 300.0,
 "interval": 0.5,
 "command": "python -m roundabout_sim.experiments sweep --vary density=20,40 --vary n_lanes=1,2 --set n_arms=0 --seeds 2"
}
```

A random sample (`sweep --sample 40 --vary density=10:120 --vary n_lanes=1,2`)
gives a sampled range as `range` instead:

```json
 "design": "sample",
 "varied": [{"name": "density", "range": [10.0, 120.0]}, {"name": "n_lanes", "values": [1, 2]}],
```

`design` is `grid`, `plan` or `sample`; `varied` lists the varied settings in
axis order (x first, then the series, then the rest), each with its `values`,
or for a sampled range its `range` [lo, hi] (without `--vary`: the plausible
range of each number, the choices actually drawn); `fixed` holds every other setting
that differs from the defaults; `preset` / `random` the base (or null);
`sample_groups` the `--sample-groups` of a random sample, as the tabs name them
(null for every group, and for a grid or plan); `seeds`,
`base_seed`, `warmup`, `measure` and `interval` the run settings; `command` the
command line. Both files are written atomically (to a temporary file, flushed to
disk, then renamed), the CSV first: a sweep stopped in between leaves its CSV
without the meta file, which the sweep viewer reads all the same.

`experiments view [CSV] [--results DIR] [--size WxH]` opens the
[sweep viewer](#sweep-viewer) (the same as `python -m roundabout_sim.sweep_viewer`).

![A sweep of reaction distance on a closed 1-lane ring](docs/sweep.png)

## Sweep viewer

```powershell
python -m roundabout_sim.sweep_viewer                       # the most recent sweep CSV in results/
python -m roundabout_sim.sweep_viewer results/sweep.csv     # a given sweep
python -m roundabout_sim.sweep_viewer --base-settings my_settings.json --results results --size 1600x900
```

The sweep viewer opens the CSV that `experiments sweep` writes (one row per
run), also CSVs written by older versions; the `PREFIX_meta.json` beside it is
optional. It sorts the columns into the **varied settings** (setting columns
with more than one value, or those the meta file lists), the **fixed
settings**, the **seed** and the **measured** columns, and draws any measured
column against the varied settings. Without a meta file it reads the layout
`experiments sweep` writes: the setting just before the `seed` column is the x
axis and the one before it the series (in other CSVs the x axis is the numeric
setting with the most values). The design comes from the design points (the
seeds' repeats dropped): every combination present is a **full grid**, however
unevenly its values are spaced; some combinations with a setting whose values
are mostly unique and unevenly spaced is a **random sample**, otherwise a
**plan**. With one varied setting a random sample also has every combination;
it is told apart by values off the slider step (`--sample` writes the meta
file, which settles it).

| View | What it shows |
|---|---|
| lines | The metric against the X setting: mean of the runs at each point (line and dots) with a band from the lowest to the highest run. With a **series** setting (up to 8 values) one line per value in the palette colours, with a legend. Reference lines as in the sweep figure, dashed: the stop-and-go onset, the level-of-service limits of the entry delay (with the letters beside the chart) and, for the order parameter, the random-placement level of each series (faint). |
| heatmap | Two varied settings: X × Y cells coloured by the mean (dark = low, light = high, empty = no runs), with a colour bar and the value in every cell when the cells are big enough for it. |
| scatter | Every run as a dot: x = a setting, y = the metric, colour = another setting (colour bar, or a legend for a choice such as the lane-change motive). The light line joins the mean at each x value; in a plan or random sample whose x takes many values (more than 12 in a plan, 6 in a sample), the means of equal-width bins of x, as in the sample figure. The only view of a random sample of several settings (one of a single setting has the lines view too). |

With three or more varied settings, each one that is neither on an axis nor the
series gets a control in the panel: **average** over all its values (every run
pooled) or **fix** it to one value (a slice through the sweep). The chart's
subtitle says which, and lists the fixed settings that differ from the
defaults.

The **metric list** has every numeric measured column, the experiments' sweep
charts first (titled as in the sweep figure, the two onset times each on its
own; in a sweep of open-mode runs also the entrance charts: entry delay,
throughput, longest queue and blocked arrivals; the rest by a readable name,
such as *Arrivals (per minute)* or *Entry v/c (busiest entrance)*, or else by
column name). Loading another CSV keeps the metric when that file has values
of it, else shows the flux; the list scrolls to the metric. Hover a metric for
its definition (from the experiments definitions) and its CSV column. Text
columns such as `entry_los` are counted in the hover readout instead, per
series line (e.g. `Entry LOS A ×1, F ×1`; the `-` of "no level of service" is
left out). Hover the chart for the exact values: the settings at that point,
mean, min and max and the number of runs (in the scatter: the run under the
mouse with all its varied settings, or the x value or bin). Values of a setting
show as in the main viewer: with the slider step's decimals, and exactly when
they lie between the steps. A setting with an infinite value (Density or
Reaction distance = inf, both valid) has no place on a scale, so its values sit
evenly spaced like a choice's, `∞` last (in the experiments' figures too). The
line under the chart's title names the slices, the runs per point, the run
lengths, the base (a preset or random settings, with its seed) and the
settings held fixed; when it is too long for the window it ends with an
ellipsis, and hovering it shows all of it (as hovering the panel's lines naming
the sweep shows every setting it varies). In the scatter a ring marks a mean
of a single run, so that its dot keeps its colour.

The **Files** tab lists the sweep CSVs in the results folder (and its
subfolders one level down), newest first; click one to load it.

The **New sweep** tab sets up a sweep without the command line: pick 1-3
settings from the list of all parameters (the 1st is the x axis, the 2nd draws
one line per value, so at most 8 values, the 3rd can be sliced or averaged
over), drag each one's range (within its slider range) and choose its number
of points, or for a choice which options to include; then the seeds per point,
the warm-up and the measured time. Every run starts from the base settings
(`--base-settings FILE`, a JSON object `{parameter: value}`, optionally with a
`seed`, as the main viewer's *Sweep viewer* button (V) writes it; names that
are not parameters are left out with a message, and a file without any
parameter is an error; otherwise the defaults). The form says how many of them
differ from the defaults and lists them, apart from the settings swept, above
the command. A setting that would do nothing in any run of the sweep set up is
dimmed in the list with the reason (*open mode only* on closed-mode base
settings, *no entrances*, *no trucks*), and picking one shows a warning, until
a picked setting makes it act (Traffic mode with open, say). **Run sweep** starts
`python -m roundabout_sim.experiments sweep ...` in the background (one or two
settings as `--x` / `--series`, three as one `--vary` each; the base settings
that differ from the defaults as `--set`; the command is shown under the form)
with a progress bar and the time left; the window stays responsive, **Cancel**
stops it with its worker processes (removing anything it had begun to write),
and an error shows the exception (or the command-line error) first, then the end
of the traceback. When it finishes the
new CSV opens; so does one cancelled while it draws its figures, since its CSV
is complete by then.

The keys Up/Down, Left/Right, Tab and H act on the chart page only.

| Key | Action |
|---|---|
| Up / Down | Previous / next metric |
| Left / Right | Previous / next setting on the x axis |
| Tab / Shift+Tab | Next / previous series (heatmap: Y axis; scatter: colour) setting |
| H | Heatmap on / off (two or more varied settings) |
| S | Save the chart as a PNG to the results folder (`sweepview_<csv>_<view>_<metric>_<time>.png`) |
| R / F5 | Reload the CSV |
| O | Files tab |
| N | New sweep tab |
| Esc | Quit (a sweep still running is stopped) |

## How the model works

Cars drive anticlockwise on concentric circular lanes. A car's state is its
lane, its angle on the ring and its speed. Each 0.1 s step works as follows:

1. **Car following (IDM, multi-anticipation).** Each driver's acceleration
   comes from the Intelligent Driver Model, towards the cars ahead in its lane.
   - It reacts to every car within the *reaction distance* (up to 5), as in
     Treiber's Human Driver Model, so looking further ahead damps waves. A very
     short distance means late, violent braking.
   - Drivers act on what they perceived *reaction time* ago (each their own,
     see below), and add correlated random noise.
   - A hard no-overlap constraint keeps cars from ever driving into each
     other. Every gap is a bumper gap: the distance between two vehicles'
     centres less half of each one's length.
2. **Exits.**
   - A car's route ends at an exit, chosen by giving it the *exit chance* at
     every exit it passes.
   - Leavers move out to the outer lane in time, slow to the turn speed and
     leave.
   - A leaver that cannot get out goes round again. So does a leaver held at a
     standstill by merging traffic for more than 5 s.
   - In closed mode each leaver is replaced by a car queued at a random
     entrance (or slotted into the ring if that queue already holds 2 cars),
     so the number of cars stays at the density setting. In open mode leavers
     are gone for good.
3. **Lane changes.**
   - Leavers merge cooperatively: they slot in behind the car beside them, and
     that car's follower eases off to let them in.
   - Other drivers consider a change *lane-change chance* times per second and
     make it if it is safe and, in "advantage" mode, faster (MOBIL).
   - A change is safe if the driver behind in the target lane (and the one
     changing) need brake no harder than 1.5× *comfortable braking*, 2.5× for
     a leaver that must get out, and never harder than the road's braking
     limit. On a dry road that limit matters only for leavers once
     comfortable braking is above 3.6 m/s².
4. **Arrivals** (open mode). Cars arrive at each entrance at random (a
   Poisson process at the *entrance chance*, shaped by the demand imbalance
   and profile) and join its queue; arrivals that find *max queue* cars
   waiting are blocked. They come from a random stream of their own, so the
   demand settings never change a closed-mode run, and in open mode the same
   seed gives the same arrivals. What the arrivals cause (entries, new
   drivers) still draws from the main stream, so two open runs with different
   demand soon differ in every other random draw too.
5. **Entries.** Queued cars merge into the outer lane, first come first
   served, when the gap to circulating traffic is at least the *entry gap*
   and more than the *follow-up time* has passed since the previous car from
   that entrance (the first time step after it). Each queue keeps its cars' arrival times, so every entry
   knows how long its car waited.

**Vehicles, drivers and the road.**

- A *truck share* of the vehicles are trucks, *truck length* long and 2.55 m
  wide (the EU limit). Their drivers accelerate at 0.5× a car's rate, want
  0.7× the desired speed and keep a 1.5× longer time gap: the IDM truck
  parameters of Treiber & Kesting and of Kesting et al.'s enhanced IDM, trucks
  measured at 0.59 m/s² from a stop (Yang et al. 2016) and truck time gaps on
  roundabouts of 1.7-2.0× those of cars (Macioszek 2019). Whether a vehicle is
  a truck is drawn when it appears, or when it joins an entrance queue, so a
  queued truck waits for a gap it fits. The driver model, lane changes,
  entries, the density control and the start layout all make room for the
  longer vehicles.
- Each driver draws their own reaction time and time gap once, around the
  settings (*reaction time spread*, *time gap spread*). The driver model, the
  lane-change and entry checks and the reaction delay use each driver's own
  values. Vehicle types and these draws come from a random stream of their
  own, so with no trucks and no spreads (the defaults) the model runs exactly
  as it did without them.
- The *road condition* multiplies what the drivers do, through one set of
  effective settings (it scales each driver's own desired speed and time gap,
  so the spreads between drivers too):

  | | desired speed | acceleration | comfortable braking | time gap | entry gap | follow-up | cornering | braking limit |
  |---|---|---|---|---|---|---|---|---|
  | dry | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 (9 m/s²) |
  | wet | 0.92 | 0.95 | 0.88 | 1.12 | 1.10 | 1.05 | 0.90 | 0.80 (7.2 m/s²) |
  | snow | 0.75 | 0.70 | 0.70 | 1.25 | 1.20 | 1.20 | 0.50 | 0.30 (2.7 m/s²) |
  | ice | 0.70 | 0.50 | 0.45 | 1.60 | 1.12 | 1.30 | 0.30 | 0.15 (1.35 m/s²) |

  The factors are a synthesis of weather studies, mostly on freeways and
  signalised streets (FHWA 2006, Rakha et al. 2009 and 2010, Perrin et al.
  2000, VTI 2001; Lee et al. 2018 for gap acceptance at roundabouts in rain).
  Ice is the least certain, and `engine.py` notes the source of each factor.
  The standstill gap (jam density) and the reaction time are not scaled. On
  snow and ice comfortable braking is close to the braking limit (1.9× and
  1.5× below it, against 4.5× on a dry road), so ordinary car following there
  often asks for more than the road allows: at the defaults harsh braking
  rises from about 1.5 episodes per minute (dry) to about 14 on snow and 65
  on ice.

The engine ([`engine.py`](roundabout_sim/engine.py)) is vectorised with numpy.
On a desktop PC it steps about 80 cars at over 250× real time and about
1,900 cars at over 100× real time (drawing the viewer comes on top), so the
viewer stays smooth with thousands of cars.

**Known simplifications.**

- Entrances and exits always come in pairs (one arm = one entrance + one exit);
  their numbers cannot be set separately.
- In closed mode (the default) each leaver is replaced at a random entrance,
  so arrivals always match departures and density is set as cars on the
  ring. Open mode has random arrivals instead, but they are a Poisson process
  at a set rate (optionally with a cosine "peak" profile), not real counts;
  no platoons arrive from upstream signals.
- Lane changes are instant in the model; the viewer animates them.
- Entrances and exits are points on the outer lane, without approach roads.
  Queued cars wait off the road (drawn on the entry stubs) with no approach
  dynamics of their own; blocked arrivals are simply lost, not rerouted.
- Entering cars merge only into the outer lane, where the leavers also
  gather, so entry capacity is low: about 400 veh/h per entrance at the
  defaults. Every entrance has the same entry gap and follow-up time.
- **Entry gap is not the HCM critical headway.** The presets use the model
  value that reproduces the HCM 6 capacity; the literal headway gives about
  40 % too little. Their follow-up times are one 0.1 s time step below the HCM
  headways, as the engine lets the next car go on the step after.
- The entry delay is the queue wait only, not the HCM control delay, so its
  level of service is only HCM-style. Its v/c estimates an entrance's
  capacity from its entry rate while cars wait, which overstates the capacity
  at light demand (an entrance with no queue lets each car in without any
  follow-up wait), so there v/c reads low; near and above capacity it is
  close.
- Drivers differ only in desired speed, reaction time and time gap; maximum
  acceleration differs only by vehicle type, and everyone brakes comfortably
  at the same rate (scaled by the road condition). Reaction times are rounded
  to the 0.1 s step.
- Trucks follow the lane's centreline like cars (and are drawn bent along
  it): they do not sweep wider on the curve, use a truck apron or straddle
  two lanes, and they change lanes as freely as cars. Their drivers keep a car's standstill
  gap and comfortable braking and accept a car's entry gap and follow-up time
  (field data give trucks 1.7-1.9× a car's critical gap and follow-up time;
  here only their length and slow start make them need more room).
- Capacity and jam density use the mean vehicle length. When the random mix
  holds more trucks than average, a ring near jam density can hold a few
  vehicles fewer than Density asks, and in dense traffic a truck the density
  control slots in waits until a gap fits it while cars may go first (with
  22 m trucks a few seconds at 0.8 × jam density, about 40 s on average at
  0.9 ×). They are added when room appears. In open mode, where a live
  Density change sets the ring only once, a truck that fits no gap is placed
  at once by respacing its lane; only a vehicle no lane has room for (from
  about 0.9 × jam density with a truck-heavy mix) is left out for good.
- The road condition is the same everywhere on the ring and for every driver.
  Its factors come mostly from studies away from roundabouts, and real
  drivers adapt to a slippery road (or fail to) in more ways than these eight
  multipliers.
- Jam causes describe what each jam's front car is doing now, not what started
  the jam (see above).
- **Presets are approximations.** Model entrances are a single lane, so
  roundabouts with two-lane entries (Los Alamitos Circle, the Armdale Rotary)
  are not presets, and at the Etoile the one-lane main-axis entries queue to
  the limit under the heavy circulating flow. The Etoile also has evenly
  spaced arms that are all both entry and exit (one constant exit chance, and
  a smooth demand shape that gives the busiest avenues about 30 % less than
  counted), at most 5 of its 8 unmarked lanes, no priority to the right
  (entering cars have priority there; approximated by a 1.5 s entry gap) and
  no cobbles.
- **Stop-and-go carries too little flow.** On the ring presets the waves form
  as measured, but the jammed state carries about 35 % less flow than in
  Stern's experiment.
- **Random settings are independent** apart from the links listed above; the
  weights of the choices (lanes, arms, road condition, traffic mode, ...) are
  assumptions.
- `--sample` without `--vary` covers the plausible ranges evenly (a Latin
  hypercube): every lane count, number of entrances, road condition and
  traffic mode equally often, each number spread evenly over its range. It
  does not follow the real-world weights and links of `--random`
  (`presets.random_changes`: mostly dry roads, the radius drawn for the lane
  count, the density kept below the jam density, ...), so it can pair a
  five-lane ring with a 12 m radius. That suits a sensitivity study; for
  single realistic settings use `--random`.
- A random sample's figure shows at most 8 varied settings, those whose rank
  correlation with the headline metrics is strongest; all of them are in the
  CSV and the sweep viewer. Its lines are means per bin with the other
  settings varying at the same time, so they show each setting's effect
  averaged over the others, not its effect with the others held.
- The sweep viewer pools the runs of a setting it averages over (mean, min and
  max over all of them), which equals the mean of the per-value means only when
  every value has the same number of runs, as in a full grid.
- A New sweep from the sweep viewer is a full grid of 1-3 settings; plans
  (`--plan`) and random samples (`--sample`) are run from the command line and
  then opened in the viewer. A sweep cancelled before its last run writes
  nothing (the CSV is written once every run is done).
- A saved state carries on exactly only in a version of the model with the
  same settings and state layout. A state from a version with other settings
  (for example one that added a parameter) restores only the settings both
  versions know, and the simulation starts afresh. The window size is not part
  of the session.

## Project layout

```
roundabout_sim/
  config.py       parameters: defaults, ranges, help text (drives sliders, CLIs and CSV columns)
  engine.py       RingSim - the vectorised multi-lane ring traffic engine (and its state_dict / from_state)
  metrics.py      flux, congestion (jammed cars, stop-and-go), jam causes, spacing / order, entrances; Recorder
  presets.py      real-world presets (sources in comments), random settings, Latin hypercube samples
  state.py        saved states: one .npz of arrays plus JSON metadata (sessions, F5 / F9, --state)
  palette.py      shared colours for viewer and figures
  textfmt.py      number and text formatting shared by the CLI, the figures and the viewers
  cli.py          command-line pieces shared by the viewer, the experiments CLI and the sweep viewer
  ui.py           pygame pieces shared by both viewers (window set-up, help box and its messages, tabs)
  app.py          pygame viewer (python -m roundabout_sim)
  experiments.py  run / sweep / view CLI (python -m roundabout_sim.experiments)
  sweep_viewer.py pygame sweep viewer and new-sweep runner (python -m roundabout_sim.sweep_viewer)
  plots.py        matplotlib dashboard, sweep, heatmap and sample figures
tests/            pytest suite:
  test_engine.py, test_metrics.py, test_config.py, test_review_fixes.py   engine invariants, metrics, regressions
  test_open_demand.py, test_vehicles_drivers.py                            open demand; trucks, driver spread, road
  test_state.py                                                            saved states, sessions, exact resume
  test_presets.py                                                          presets, random settings, Latin hypercube
  test_experiments.py, test_sweeps.py                                      run / sweep CLI, designs, meta file, figures
  test_app.py, test_sweep_viewer.py                                        viewer and sweep viewer (headless)
  test_integration.py                                                      presets / random / sweep button in the viewer
  conftest.py, helpers.py                                                  headless set-up, viewer factory, event helpers
docs/             images used in this README
```
