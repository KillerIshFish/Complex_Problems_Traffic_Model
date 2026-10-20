"""Headless parameter sweeps: how density and speed variance shape the flow.

Example (≈ a few minutes on a multi-core machine)::

    python -m roundabout_sim.experiments --cars 5 10 20 30 45 60 --speed-std 0 2 4 --seeds 3

Each run warms up (so the initial placement is forgotten), then measures
throughput, mean speed, trip time, ring density and entry queues. Results go
to ``<out>.csv`` (one row per run) and ``<out>.png`` (means across seeds,
shaded band = min-max across seeds).
"""

from __future__ import annotations

import argparse
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from roundabout_sim.env import RoundaboutTrafficEnv

METRICS = {
    "throughput_per_min": "Throughput (cars/min)",
    "mean_speed": "Mean speed (m/s)",
    "mean_trip_time": "Mean trip time (s)",
    "queued": "Cars queued at entries",
}

# Categorical series colours (fixed order) and light chart surface tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def run_one(cars: int, speed_std: float, mean_speed: float, seed: int,
            warmup: float, measure: float) -> dict:
    env = RoundaboutTrafficEnv(config={
        "vehicles_count": cars, "speed_std": speed_std, "mean_speed": mean_speed,
    })
    env.reset(seed=seed)
    freq = env.config["simulation_frequency"]
    for _ in range(int(warmup * freq)):
        env.step()
    start = env.stats()
    samples = []
    for k in range(int(measure * freq)):
        stats, *_ = env.step()
        if k % freq == 0:  # once per simulated second
            samples.append(stats)
    end = env.stats()
    trips = [tt for t_exit, tt in env.trips if t_exit > start["time"]]
    s = pd.DataFrame(samples)
    return {
        "cars": cars,
        "speed_std": speed_std,
        "mean_speed_setting": mean_speed,
        "seed": seed,
        "throughput_per_min": (end["exited"] - start["exited"]) / (measure / 60.0),
        "mean_speed": s["mean_speed"].mean(),
        "mean_trip_time": float(np.mean(trips)) if trips else np.nan,
        "ring_density": s["ring_density"].mean(),
        "queued": s["queued"].mean(),
        "crashes": end["crashes"] - start["crashes"],
    }


def sweep(cars, stds, mean_speed, seeds, warmup, measure, workers) -> pd.DataFrame:
    jobs = list(itertools.product(cars, stds, range(seeds)))
    print(f"{len(jobs)} runs on {workers} workers "
          f"({warmup:.0f} s warm-up + {measure:.0f} s measured each)")
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, n, std, mean_speed, seed, warmup, measure)
                   for n, std, seed in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            r = f.result()
            rows.append(r)
            print(f"[{i:3d}/{len(jobs)}] cars={r['cars']:3d} std={r['speed_std']:.1f} "
                  f"seed={r['seed']}  {r['throughput_per_min']:5.1f} cars/min  "
                  f"v={r['mean_speed']:4.1f} m/s  ({time.time() - t0:.0f} s)")
    return pd.DataFrame(rows).sort_values(["speed_std", "cars", "seed"]).reset_index(drop=True)


def plot(df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stds = sorted(df["speed_std"].unique())
    if len(stds) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} speed-std values can be plotted at once")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=SURFACE)
    for ax, (metric, title) in zip(axes.flat, METRICS.items()):
        ax.set_facecolor(SURFACE)
        for color, std in zip(SERIES, stds):
            g = df[df["speed_std"] == std].groupby("cars")[metric].agg(["mean", "min", "max"])
            ax.fill_between(g.index, g["min"], g["max"], color=color, alpha=0.15, linewidth=0)
            ax.plot(g.index, g["mean"], color=color, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"σ = {std:g} m/s")
        ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=12)
        ax.set_xlabel("Cars in system", color=TEXT_SECONDARY)
        ax.set_ylim(bottom=0)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=TEXT_SECONDARY)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Desired-speed std. dev.", loc="upper center",
               ncol=len(stds), frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(f"Roundabout performance vs density "
                 f"(mean desired speed {df['mean_speed_setting'].iloc[0]:g} m/s, "
                 f"{df['seed'].nunique()} seeds; band = min-max)",
                 y=0.92, color=TEXT_SECONDARY, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep density and speed variance.")
    parser.add_argument("--cars", type=int, nargs="+", default=[5, 10, 20, 30, 45, 60],
                        help="cars in system (density) values")
    parser.add_argument("--speed-std", type=float, nargs="+", default=[0.0, 2.0, 4.0],
                        help="std. dev. of desired speed [m/s]")
    parser.add_argument("--mean-speed", type=float, default=12.0, help="mean desired speed [m/s]")
    parser.add_argument("--seeds", type=int, default=3, help="repetitions per setting")
    parser.add_argument("--warmup", type=float, default=60.0, help="simulated warm-up [s]")
    parser.add_argument("--measure", type=float, default=240.0, help="simulated measurement [s]")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", type=Path, default=Path("results/sweep"),
                        help="output path prefix (writes .csv and .png)")
    args = parser.parse_args()

    df = sweep(args.cars, args.speed_std, args.mean_speed, args.seeds,
               args.warmup, args.measure, args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    plot(df, args.out.with_suffix(".png"))
    print(f"wrote {args.out.with_suffix('.csv')} and {args.out.with_suffix('.png')}")
    summary = df.groupby(["speed_std", "cars"])[list(METRICS) + ["crashes"]].mean().round(1)
    print(summary.to_string())


if __name__ == "__main__":
    main()
