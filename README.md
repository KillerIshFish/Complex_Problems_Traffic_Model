# Roundabout traffic model

An agent-based model of a single-lane, four-arm roundabout. Every car is an
independent driver agent. You can adjust **traffic density** and **driver speed
variance** live, or sweep them in batch to see how they affect throughput,
speed, trip time and queueing.

Built on [highway-env](https://github.com/Farama-Foundation/HighwayEnv)
(road geometry, vehicle kinematics, IDM car-following), with roundabout
right-of-way rules added on top.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m roundabout_sim            # interactive viewer
python -m roundabout_sim.experiments  # batch sweep -> results/sweep.csv + .png
```

### Interactive viewer

| Control | Effect |
|---|---|
| **Cars in system** | Density. The model keeps this many cars present: a car that exits is replaced by a new one at a random entry. Lowering it lets cars drain away without replacing them. |
| **Mean desired speed** | Mean free-road speed of the drivers. |
| **Speed std. deviation** | Spread of desired speeds across drivers (speed variance). |
| **Roundabout speed factor** | Each driver slows to this fraction of their own desired speed on the roundabout's curves. |
| **Simulation speed** | Simulation steps per frame. |
| `Space` / `R` / `Esc` | Pause, restart with a new seed, quit. `--seed N` fixes the seed. |

Car colour shows each driver's speed temperament: orange is slow, grey is
average, blue is fast, and red has crashed. The side panel shows live
throughput, mean speed, ring density (veh/km), entry queues and trip times.

### Batch experiments

```powershell
python -m roundabout_sim.experiments --cars 5 10 20 30 45 60 --speed-std 0 2 4 --seeds 3
```

Each run warms up for `--warmup` simulated seconds, then measures for
`--measure` seconds. Runs are spread over `--workers` processes (default: all
cores but one). The tool writes one CSV row per run, a 2×2 figure of metrics
against density (one line per speed std. dev., averaged over seeds), and a
summary table to the terminal.

## How the model works

**Drivers** (`roundabout_sim/agents.py`, `YieldingIDMVehicle`)

- *Car-following:* the Intelligent Driver Model (IDM). The acceleration
  exponent is randomized per driver.
- *Desired speed:* each driver draws a fixed temperament `z ~ N(0, 1)`
  (clipped to ±2.5). Their desired speed is `mean + z · std`, so moving a
  slider retunes every existing driver consistently. On the ring, the merge
  lanes and the exit curves, the target is `ring factor × desired speed`, and
  drivers brake gently on the approach to reach it.
- *Gap acceptance:* entering cars give way to circulating traffic. They treat
  the yield line as a stopped obstacle unless every relevant ring car is at
  least `CRITICAL_GAP` (3 s) and `MIN_GAP_DISTANCE` (15 m) away, and nobody
  is just past the merge point. Ring cars that will leave at the exit just
  before the entry are ignored. Once a car is over the line, or can no longer
  stop comfortably, it commits.
- *Zipper merge:* a car that is entering is projected onto the ring at its
  merge point. Whichever of the two cars (entering or circulating) is
  virtually behind follows the other with IDM.

**Environment** (`roundabout_sim/env.py`, `RoundaboutTrafficEnv`)

- There is no ego or RL vehicle. Agents spawn at the four entry roads with
  random destinations and are removed when they reach the end of an exit road.
- The approach to the yield line and the spawn speed are capped so new cars
  don't launch into the back of a queue.
- `RoundaboutRoad` patches two quirks of highway-env's leader search:
  - Ring cars mistook cars waiting at a yield line for cars in their lane.
  - On circular lanes the next-segment lookup wrapped around, so cars far
    down an exit road became phantom leaders.
- All parameters live in `RoundaboutTrafficEnv.default_config()`. You can
  override any of them with `RoundaboutTrafficEnv(config={...})`.

**Known limits / extensions**

- Single-lane ring only. highway-env's two-lane ring has no lane discipline:
  inner-lane cars cut across the outer lane to exit.
- Density is controlled as "cars in system" (a closed loop). An open-boundary
  version would use an arrival rate per entry instead.
- Gap-acceptance parameters (`CRITICAL_GAP`, `MIN_GAP_DISTANCE`) are class
  constants. They are good candidates for per-driver heterogeneity or extra
  sliders.

## Layout

```
roundabout_sim/
  agents.py       driver agent (IDM + right-of-way + zipper merge)
  env.py          roundabout environment, spawning/removal, metrics
  app.py          pygame viewer with sliders and live charts
  experiments.py  parallel parameter sweeps -> CSV + figure
  __main__.py     `python -m roundabout_sim` launches the viewer
```
