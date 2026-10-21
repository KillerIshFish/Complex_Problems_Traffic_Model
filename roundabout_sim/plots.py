"""Figures for the experiments: a single-run dashboard and sweep small multiples.

Both use the light chart tokens and the fixed categorical order from
:mod:`roundabout_sim.palette`, one y-axis per chart and a legend whenever a
chart has two or more series. Every panel title names the measure and its
unit; a smaller second line says how to read it. Figures are built with
``matplotlib.figure`` directly (no pyplot), so they never touch global GUI
state.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator

from roundabout_sim.engine import MAX_DECEL
from roundabout_sim.palette import PLOT, SERIES_HEX, speed_cmap

INK, INK2, GRID, SURFACE = PLOT["text"], PLOT["text2"], PLOT["grid"], PLOT["surface"]
LINE = 1.5  # [pt] ~2 px at the saved resolution
REF = 0.9  # [pt] reference lines (thresholds, markers)
DASHED = (0, (4, 3))  # faint comparison lines (a baseline run, random placement)
DPI = 120
NOTE_SIZE = 8.5

RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.edgecolor": GRID,
    "axes.titlesize": 11,
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.titlepad": 8,
    "axes.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.size": 10,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.labelcolor": INK,
    "lines.linewidth": LINE,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
}

# What the front car of a jam is doing (metrics.jam_origins), in palette order.
# "Following" is only an upper bound on natural jams: a wave set off at an exit
# travels backwards with ordinary followers at its front.
JAM_ORIGINS = (("jams_following", "Front car following"),
               ("jams_merge", "Leaver merging out to its exit"),
               ("jams_exit", "Slowing down to exit"),
               ("jams_entry", "Just entered the ring"),
               ("jams_other", "Braked by you / slotted in by the density control"))


def _line(ax, x, y, color, label=None, **kw):
    return ax.plot(x, y, color=color, linewidth=LINE, label=label, **kw)[0]


def _ref_line(ax, y: float, label: str | None = None) -> None:
    """Horizontal reference line (named in the legend or the panel note)."""
    ax.axhline(y, color=INK2, linewidth=REF, zorder=1, label=label)


@dataclass
class _Note:
    """A chart title with the smaller note under it (see :func:`_panel_title`)."""

    ax: object
    title: str
    text: str
    artist: object | None
    right: str
    right_color: str
    min_lines: int
    style: dict

    def place(self, lines: int) -> None:
        pad = 8 + 11.5 * max(lines, self.min_lines)  # one pad is shared by all of an axes' titles
        if self.artist is not None:  # hangs from just under the title
            self.artist.xyann = (0, pad - 3)
        self.ax.set_title(self.title, pad=pad, **self.style)
        if self.right:
            self.ax.set_title(self.right, loc="right", pad=pad, color=self.right_color, **self.style)


def _panel_title(ax, title: str, note: str = "", right: str = "", right_color: str = INK,
                 min_lines: int = 2, **style) -> None:
    """Title (measure and unit) with an optional smaller ``note`` under it saying how to read it.

    ``right`` is a headline result set right-aligned on the title line. The
    note stays out of the layout; :func:`_fit_notes` wraps it to the final
    axes width and pads the title to make room (at least ``min_lines``, so
    titles of neighbouring charts line up).
    """
    artist = None
    if note:
        artist = ax.annotate(note, xy=(0, 1), xycoords="axes fraction", xytext=(0, 0), textcoords="offset points",
                             ha="left", va="top", fontsize=NOTE_SIZE, color=INK2, linespacing=1.25)
        artist.set_in_layout(False)
    entry = _Note(ax, title, note, artist, right, right_color, min_lines, style)
    entry.place(1 if note else 0)
    ax.get_figure().__dict__.setdefault("_panel_notes", []).append(entry)


def _fit_notes(fig: Figure) -> None:
    """Wrap every chart note to its axes' width once the layout is known, and pad the titles to fit."""
    notes = fig.__dict__.pop("_panel_notes", [])
    if not notes:
        return
    canvas = fig.canvas if isinstance(fig.canvas, FigureCanvasAgg) else FigureCanvasAgg(fig)
    fig.draw_without_rendering()  # runs the layout: axes widths are final (titles only change heights)
    renderer = canvas.get_renderer()
    counts = []
    for note in notes:
        lines = []
        if note.artist is not None:
            char = note.artist.get_window_extent(renderer).width / max(len(note.text), 1)
            chars = max(12, int(note.ax.get_window_extent(renderer).width / char))
            lines = textwrap.wrap(note.text, chars, break_on_hyphens=False)
            note.artist.set_text("\n".join(lines))
        counts.append(len(lines))
    # Charts whose tops line up share the tallest note's padding, so their titles line up too.
    tops = [round(note.ax.get_position().y1, 3) for note in notes]
    row_max: dict[float, int] = {}
    for top, count in zip(tops, counts):
        row_max[top] = max(row_max.get(top, 0), count)
    for note, top in zip(notes, tops):
        note.place(row_max[top])


def _legend_band(ax, ncol: int, entries: int, handles: list | None = None) -> None:
    """Legend in a band added above the data, so it never covers a line.

    ``handles`` (with their labels) replaces the axes' labelled artists, e.g.
    for a key to line styles on a chart whose colours are a figure legend.
    """
    rows = -(-entries // ncol)
    band = 0.04 + 0.095 * rows  # share of the axes height the legend needs
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * band / (1 - band))
    extra = {"handles": handles} if handles is not None else {}
    ax.legend(loc="upper left", ncol=ncol, fontsize=8.5, handlelength=2.0 if handles else 1.5,
              columnspacing=1.2, borderaxespad=0.2, **extra)


def _wrap(subtitle: str, fig_width: float) -> list[str]:
    """Subtitle lines: each ``\\n``-separated part wrapped to the figure width (10 pt text)."""
    chars = int(fig_width * 12.5)
    return [line for part in subtitle.split("\n") for line in textwrap.wrap(part, chars, break_on_hyphens=False)]


def _header_ratio(lines: int, fig_height: float, rest: float, extra: float = 0.0) -> float:
    """Height ratio for a header row holding the title, ``lines`` of subtitle and ``extra`` inches."""
    need = 0.34 + 0.25 * lines + extra  # [in]
    return need * rest / (fig_height - 1.0 - need)


def _header(ax, title: str, lines: list[str]) -> None:
    """Title and subtitle lines at the top of an axis-less header row.

    The texts are left out of the layout (the row is sized for them instead):
    a long subtitle would otherwise make the layout engine squeeze every column.
    """
    ax.axis("off")
    for text in (ax.text(0, 1, title, transform=ax.transAxes, ha="left", va="top", fontsize=15,
                         fontweight="bold", color=INK),
                 ax.text(0, 1, "\n" + "\n".join(lines), transform=ax.transAxes, ha="left", va="top",
                         fontsize=10, color=INK2, linespacing=1.9)):
        text.set_in_layout(False)


def _time_axis(ax, frame: pd.DataFrame) -> None:
    ax.set_xlim(0, float(frame["time"].max()) if len(frame) else 1.0)
    ax.set_xlabel("Time (s)")


def _top(*series, floor: float = 1.0) -> float:
    """Upper axis limit for non-negative data (``floor`` if it is all zero or NaN)."""
    peaks = [np.nanmax(np.asarray(s, dtype=float)) for s in series if np.isfinite(np.asarray(s, float)).any()]
    return max(floor, 1.05 * max(peaks, default=0.0))


def _lane_name(lane: int, n_lanes: int) -> str:
    where = " (inner)" if lane == 0 else " (outer)" if lane == n_lanes - 1 else ""
    return f"Lane {lane + 1}{where}"


def _bin_width(duration: float, max_bins: int = 60) -> float:
    """Smallest round bin width [s] that splits ``duration`` into at most ``max_bins`` bins."""
    for width in (1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600, 1200, 3600):
        if duration / width <= max_bins:
            return float(width)
    return float(np.ceil(duration / max_bins))


# ----------------------------------------------------------------- dashboard
def dashboard(frame: pd.DataFrame, times: np.ndarray, speeds: np.ndarray, occupancy: np.ndarray,
              car_speeds: np.ndarray, *, title: str, subtitle: str, n_lanes: int,
              arm_deg: Sequence[float], detector_deg: float | None, v_free: float, late_from: float,
              stop_index: np.ndarray, onset: float, onset_threshold: float, following_share: float,
              wave_speed: float, rate_window: float, congestion_onset: float, congestion_threshold: float,
              baseline: pd.DataFrame | None = None, arm_share: float = float("nan")) -> Figure:
    """Single-run dashboard.

    ``frame`` is :meth:`Recorder.frame`; ``times``/``speeds`` the space-time
    array (speed per angular bin); ``occupancy`` the per-sector occupancy over
    time; ``car_speeds`` every car's speed sampled after ``late_from`` seconds
    (the histogram and the occupancy ring use that late part of the run).

    Congestion over time is the headline: the share of jammed cars, with
    ``congestion_onset`` the first time its ``rate_window`` mean exceeded
    ``congestion_threshold`` (NaN if never). ``baseline`` (optional) is the
    frame of the same run with exit_prob = 0, drawn as a faint dashed
    reference, and ``arm_share`` the share of the jamming it attributes to the
    entrances/exits. ``stop_index`` is the stop-and-go index per sample (the
    instability companion), ``onset`` the time it first exceeded
    ``onset_threshold``; ``following_share`` the share of jams whose front car
    is just following over the late half (an upper bound on natural jams).
    Rolling rates are hidden until their ``rate_window`` has filled.
    """
    t = frame["time"].to_numpy()

    def rate(col: str) -> pd.Series:
        return frame[col].where(frame["time"] >= rate_window)

    with matplotlib.rc_context(RC):
        fig = Figure(figsize=(20, 18.5), dpi=DPI, layout="constrained")
        fig.get_layout_engine().set(h_pad=0.08, w_pad=0.08, hspace=0.035, wspace=0.03)
        lines = _wrap(subtitle, fig.get_figwidth())
        heights = (2.0, 1.15, 1.0, 1.0)
        outer = fig.add_gridspec(2, 1, height_ratios=(
            _header_ratio(len(lines), fig.get_figheight(), sum(heights)), sum(heights)))
        _header(fig.add_subplot(outer[0]), title, lines)
        gs = outer[1].subgridspec(4, 12, height_ratios=heights)

        # Row 1: where the jams are, and where the cars are
        _spacetime(fig, fig.add_subplot(gs[0, :9]), times, speeds, arm_deg, detector_deg, v_free, wave_speed)
        late = times >= late_from
        occ = occupancy[late].mean(axis=0) if late.any() and occupancy.size else np.zeros(occupancy.shape[1])
        _occupancy_ring(fig.add_subplot(gs[0, 9:], projection="polar"), 100.0 * occ, arm_deg)

        # Row 2: congestion over time (the headline) and what the front car of each jam is doing
        _congestion_panel(fig.add_subplot(gs[1, :6]), frame, late_from, congestion_onset, congestion_threshold,
                          rate_window, baseline, arm_share)
        _jam_origins_panel(fig.add_subplot(gs[1, 6:]), frame, following_share, late_from)

        # Row 3: how unstable, how fast, how much gets through
        _stop_go_panel(fig.add_subplot(gs[2, 0:4]), frame, stop_index, onset, onset_threshold, rate_window)

        ax = fig.add_subplot(gs[2, 4:8])
        _panel_title(ax, "Mean speed (m/s)", "average over the cars on the ring; "
                                            "grey = what those drivers would like to drive")
        series = []
        if n_lanes > 1:
            series = [c for c in (f"lane{k}_speed" for k in range(n_lanes)) if c in frame]
            for lane in range(n_lanes):
                if f"lane{lane}_speed" in frame:
                    _line(ax, t, frame[f"lane{lane}_speed"], SERIES_HEX[lane % len(SERIES_HEX)],
                          _lane_name(lane, n_lanes))
            _line(ax, t, frame["mean_speed"], INK, "All cars")
        else:
            _line(ax, t, frame["mean_speed"], SERIES_HEX[0], "All cars")
        if "mean_free_speed" in frame:
            ax.plot(t, frame["mean_free_speed"], color=INK2, linewidth=REF, label="Desired speed", zorder=1)
            series.append("mean_free_speed")
        ax.set_ylim(0, _top(frame["mean_speed"], *(frame[c] for c in series)))
        entries = len(ax.get_legend_handles_labels()[1])
        _legend_band(ax, 4 if entries <= 4 else 3, entries)
        _time_axis(ax, frame)

        ax = fig.add_subplot(gs[2, 8:])
        _panel_title(ax, "Flux past a point (cars/s, all lanes)",
                     f"whole ring: from the speeds, every sample; detector: cars crossing one line, "
                     f"{rate_window:g} s average")
        _line(ax, t, frame["flux"], SERIES_HEX[0], "Whole ring")
        _line(ax, t, rate("detector_flux"), SERIES_HEX[1], "Detector line")
        ax.set_ylim(0, _top(frame["flux"], rate("detector_flux"), floor=0.1))
        _legend_band(ax, 2, 2)
        _time_axis(ax, frame)

        # Row 4: do the cars congregate, what happens on the road, how fast are they
        _congregation_panel(fig.add_subplot(gs[3, 0:4]), frame)

        ax = fig.add_subplot(gs[3, 4:8])
        _panel_title(ax, "Events per minute (whole ring)",
                     f"{rate_window:g} s rolling average; harsh braking = an episode of the driver model asking "
                     f"for more than {MAX_DECEL:g}\u00a0m/s² (over-reaction after a late or delayed response), or the "
                     f"no-overlap constraint stopping a car harder", min_lines=3)
        events = {"emergency_per_min": "Harsh braking (episodes/min)", "lane_changes_per_min": "Lane changes",
                  "exits_per_min": "Exits"}
        for k, (col, label) in enumerate(events.items()):
            _line(ax, t, rate(col), SERIES_HEX[k], label)
        ax.set_ylim(0, _top(*(rate(c) for c in events)))
        _legend_band(ax, 3, 3)
        if not any(rate(c).notna().any() for c in events):
            ax.text(0.5, 0.45, f"shown once {rate_window:g} s have passed", transform=ax.transAxes,
                    ha="center", va="center", color=INK2)
        _time_axis(ax, frame)

        _speed_histogram(fig.add_subplot(gs[3, 8:]), car_speeds, v_free, late_from)
        _fit_notes(fig)
    return fig


def _late_mean(frame: pd.DataFrame | None, col: str, late_from: float) -> float:
    if frame is None or col not in frame:
        return float("nan")
    late = frame.loc[frame["time"] >= late_from, col]
    return float(late.mean()) if len(late) else float("nan")


def _congestion_panel(ax, frame: pd.DataFrame, late_from: float, onset: float, threshold: float,
                      window: float, baseline: pd.DataFrame | None, arm_share: float) -> None:
    """Headline congestion chart: jammed cars and speed deficit over time, the onset, and the no-exit baseline."""
    t = frame["time"].to_numpy()
    jammed = 100 * _late_mean(frame, "jam_fraction", late_from)
    if not np.isfinite(jammed):
        right = ""
    elif baseline is not None and np.isfinite(arm_share):
        right = (f"{jammed:.0f} % jammed from {late_from:.0f} s on; "
                 f"{100 * arm_share:.0f} % of it due to entrances/exits")
    elif baseline is not None:
        right = (f"{jammed:.0f} % jammed from {late_from:.0f} s on "
                 f"(no exits: {100 * _late_mean(baseline, 'jam_fraction', late_from):.0f} %)")
    else:
        right = f"{jammed:.0f} % of cars jammed from {late_from:.0f} s on"
    note = (f"jammed: share of cars crawling below 30 % of their free speed (rises steadily with density); "
            f"speed deficit: how far below their free speed the cars drive, all together. Congested from the "
            f"first time the {window:g} s mean of jammed exceeds {100 * threshold:g} %")
    if baseline is not None:
        note += ("; dashed = the same run (seed and all else equal) with exit chance 0, so the gap between the "
                 "two jammed lines is what the entrances/exits cause")
    _panel_title(ax, "Congestion over time (%)", note, right=right, fontsize=12)
    _line(ax, t, 100 * frame["jam_fraction"], SERIES_HEX[0], "Jammed cars (% of cars)")
    _line(ax, t, 100 * frame["speed_deficit"], SERIES_HEX[1], "Speed deficit (% of free speed)")
    entries = 3
    if baseline is not None:
        ax.plot(baseline["time"], 100 * baseline["jam_fraction"], color=INK2, linewidth=REF + 0.3,
                linestyle=DASHED, zorder=1.5, label="Jammed: same run, no exits")
        entries += 1
    _ref_line(ax, 100 * threshold, f"Congestion threshold ({100 * threshold:g} %, {window:g} s mean)")
    if np.isfinite(onset):  # stops below the legend band
        ax.vlines(onset, 0, 100, color=INK, linewidth=REF, zorder=1, label=f"Congested from {onset:.0f} s")
        entries += 1
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    _legend_band(ax, 3, entries)
    _time_axis(ax, frame)


def _stop_go_panel(ax, frame: pd.DataFrame, stop_index: np.ndarray, onset: float, threshold: float,
                   rate_window: float) -> None:
    """Instability: the stop-and-go index over time, its onset threshold and the first stop-and-go jam."""
    t = frame["time"].to_numpy()
    if np.isfinite(onset):
        right, right_color = f"first at {onset:.0f} s", INK
    else:
        right, right_color = f"never above {threshold:g}", INK2
    _panel_title(ax, "Instability: stop-and-go index (stops/car/min)",
                 f"how often a moving car is forced to (nearly) stop, over the last {rate_window:g} s: ~0 when "
                 f"smooth and also in a standstill (nobody is moving to be stopped), 1-2 in stop-and-go waves. "
                 f"First stop-and-go jam = first time above {threshold:g}", right=right, right_color=right_color,
                 min_lines=3)
    _line(ax, t, stop_index, SERIES_HEX[0], "Stop-and-go index")
    _ref_line(ax, threshold, f"Onset threshold ({threshold:g})")
    top = _top(stop_index, floor=max(1.0, 2.0 * threshold))
    entries = 2
    if np.isfinite(onset):  # stops below the legend band
        ax.vlines(onset, 0, top, color=INK, linewidth=REF, zorder=1, label="First stop-and-go jam")
        entries = 3
    ax.set_ylim(0, top)
    _legend_band(ax, entries, entries)
    _time_axis(ax, frame)


def _congregation_panel(ax, frame: pd.DataFrame) -> None:
    """Do the cars congregate? Spacing Gini, clustering and the order parameter against its random level."""
    t = frame["time"].to_numpy()
    _panel_title(ax, "Do cars congregate? (dimensionless, 0 = even)",
                 "spacing Gini: how unevenly the road is shared, ~0.35 = random gaps, towards 1 = packed queues "
                 "or platoons; clustering: sector spread beyond random scatter (often 0 in dense flow); order "
                 "parameter: 1 = all on one side, real only well above the dashed level random placement gives",
                 min_lines=3)
    cols = {"spacing_gini": "Spacing Gini", "clustering": "Clustering", "order_param": "Order parameter"}
    shown = [c for c in cols if c in frame]
    for k, col in enumerate(cols):
        if col in frame:
            _line(ax, t, frame[col], SERIES_HEX[k], cols[col])
    if "order_baseline" in frame:
        ax.plot(t, frame["order_baseline"], color=INK2, linewidth=REF + 0.3, linestyle=DASHED, zorder=1.5,
                label="Order parameter of random placement")
        shown.append("order_baseline")
    ax.set_ylim(0, _top(*(frame[c] for c in shown), floor=1.0))
    entries = len(ax.get_legend_handles_labels()[1])
    _legend_band(ax, 2 if entries > 3 else entries, entries)
    _time_axis(ax, frame)


def _jam_origins_panel(ax, frame: pd.DataFrame, following_share: float, late_from: float) -> None:
    """Stacked columns per time bin: mean number of jams, split by what their front car is doing."""
    if np.isfinite(following_share):
        right, right_color = (f"front car following: {100 * following_share:.0f} % of jams from {late_from:.0f} s on",
                              INK)
    else:
        right, right_color = f"No jams after {late_from:.0f} s", INK2
    _panel_title(ax, "Jams by what their front car is doing (count)",
                 "following is only an upper bound on natural (phantom) jams: a wave set off at an exit travels "
                 "backwards with ordinary followers at its front. For what the entrances/exits really cause, "
                 "compare with exit chance 0 (run --baseline)",
                 right=right, right_color=right_color, fontsize=12)
    t_end = float(frame["time"].max()) if len(frame) else 1.0
    width = _bin_width(t_end)
    cols = [c for c, _ in JAM_ORIGINS if c in frame]
    means = frame.groupby((frame["time"] // width).astype(int))[cols].mean()
    left = means.index.to_numpy(float) * width
    bottom = np.zeros(len(means))
    for k, (col, label) in enumerate(JAM_ORIGINS):
        if col not in means:
            continue
        values = means[col].to_numpy(float)
        ax.bar(left, values, width=width, bottom=bottom, align="edge", color=SERIES_HEX[k],
               edgecolor=SURFACE, linewidth=1.2, label=label)
        bottom += values
    ax.set_ylim(0, _top(bottom, floor=1.0))
    if bottom.max(initial=0.0) <= 4:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="x", visible=False)
    _legend_band(ax, 3, len(cols))
    _time_axis(ax, frame)
    ax.set_xlabel(f"Time (s), {width:g} s bins")


def _spacetime(fig: Figure, ax, times, speeds, arm_deg, detector_deg, v_free, wave_speed) -> None:
    """Space-time diagram: x time, y position on the ring, colour = mean speed."""
    cmap = speed_cmap().with_extremes(bad=SURFACE)
    if times.size:
        half = 0.5 * (times[1] - times[0]) if times.size > 1 else 0.5
        extent = (times[0] - half, times[-1] + half, 0, 360)
        im = ax.imshow(np.ma.masked_invalid(speeds.T), origin="lower", aspect="auto", extent=extent,
                       cmap=cmap, vmin=0, vmax=v_free, interpolation="nearest")
        bar = fig.colorbar(im, ax=ax, pad=0.008, fraction=0.025, aspect=30)
        bar.set_label("Mean speed in the bin (m/s; orange = stopped, blue = free flow)", color=INK2)
        bar.outline.set_visible(False)
        bar.ax.tick_params(colors=INK2)
        ax.set_xlim(extent[0], extent[1])
    ax.grid(False)
    ax.set_ylim(0, 360)
    ax.yaxis.set_major_locator(MultipleLocator(45))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position on the ring (deg) - traffic drives upwards")
    marks = [(a, f"arm {k + 1}") for k, a in enumerate(arm_deg)]
    if detector_deg is not None:
        marks.append((detector_deg, "detector"))
    for angle, label in marks:
        ax.axhline(angle, color=INK, linewidth=0.8, alpha=0.55)
        ax.annotate(label, xy=(0, angle), xycoords=("axes fraction", "data"), xytext=(4, 2),
                    textcoords="offset points", fontsize=8.5, color=INK, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.8))
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title("Space-time diagram: speed by position on the ring (stripes sloping down = jams moving backwards)")
    if np.isfinite(wave_speed):
        ax.set_title(f"Jam wave speed {wave_speed:+.1f} m/s ({3.6 * wave_speed:+.0f} km/h)",
                     loc="right", fontsize=11, color=INK)
    else:
        ax.set_title("No jam waves measured", loc="right", fontsize=11, color=INK2)


def _occupancy_ring(ax, occ_pct: np.ndarray, arm_deg: Sequence[float]) -> None:
    """Polar bars: share of each sector's road covered by cars."""
    sectors = occ_pct.size
    width = 2 * np.pi / sectors
    centres = (np.arange(sectors) + 0.5) * width
    ax.bar(centres, occ_pct, width=0.86 * width, color=SERIES_HEX[0], linewidth=0, label="Sector")
    ring = np.linspace(0, 2 * np.pi, 361)
    ax.plot(ring, np.full_like(ring, occ_pct.mean()), color=INK, linewidth=1.1, label="Ring average")
    ax.set_ylim(0, max(1.0, float(occ_pct.max()) * 1.12))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_rlabel_position(135)
    ax.tick_params(axis="y", labelsize=8, colors=INK)
    for label in ax.get_yticklabels():
        label.set_bbox(dict(boxstyle="round,pad=0.1", fc=SURFACE, ec="none", alpha=0.85))
    if len(arm_deg):
        ax.set_xticks(np.radians(arm_deg))
        ax.set_xticklabels([f"arm {k + 1}\n{a:.0f}°" for k, a in enumerate(arm_deg)], fontsize=8.5)
    else:
        ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.tick_params(axis="x", colors=INK2, pad=2)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(False)  # keep the radial scale readable on top of the bars
    ax.spines["polar"].set_color(GRID)
    ax.set_title("Where cars congregate: road covered (%, late half)\nanticlockwise = direction of travel",
                 loc="center", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2)


def _speed_histogram(ax, car_speeds: np.ndarray, v_free: float, late_from: float) -> None:
    top = max(v_free * 1.25, float(np.max(car_speeds)) if car_speeds.size else 1.0)
    width = 0.5
    edges = np.arange(0.0, top + width, width)
    counts, _ = np.histogram(car_speeds, bins=edges)
    share = 100 * counts / max(counts.sum(), 1)
    ax.bar(edges[:-1], share, width=width, align="edge", color=SERIES_HEX[0], edgecolor=SURFACE, linewidth=1.0)
    ax.axvline(v_free, color=INK2, linewidth=REF)
    ax.annotate("mean desired speed", xy=(v_free, 0.97), xycoords=("data", "axes fraction"),
                xytext=(-4, 0), textcoords="offset points", ha="right", va="top", fontsize=8.5, color=INK2)
    _panel_title(ax, "Speed distribution (% of car samples)",
                 f"all cars' speeds after {late_from:.0f} s; a peak near 0 = cars stuck in jams")
    ax.set_xlabel("Speed (m/s)")
    ax.set_xlim(0, edges[-1])
    ax.set_ylim(0, max(1.0, float(share.max()) * 1.2))
    ax.grid(axis="x", visible=False)


# --------------------------------------------------------------------- sweep
@dataclass(frozen=True)
class Panel:
    """One sweep chart: a CSV column against the swept parameter."""

    column: str
    title: str  # measure (unit)
    note: str = ""  # how to read it
    scale: float = 1.0  # applied to the values (e.g. 100 for %)
    from_zero: bool = True
    min_top: float = 0.0  # the y axis reaches at least this, so noise near 0 stays flat
    ref: float | None = None  # horizontal reference line
    empty: str = "no data"  # shown when no run has a value
    second: str | None = None  # a second column with the same unit, drawn dashed in the series colour
    styles: tuple[str, str] = ("", "")  # legend names of the solid (column) and dashed (second) lines
    ref_column: str | None = None  # a faint dashed per-series reference read from this column
    ref_label: str = ""  # its legend name


def sweep_panels(x: str, onset_threshold: float, congestion_threshold: float = 0.3) -> tuple[Panel, ...]:
    """The sweep charts, hero (flux) first, then speed and congestion, jams, and congregation."""
    flux = (Panel("flux", "Fundamental diagram: flux (cars/s past a point, all lanes)",
                  "rises with density while traffic flows, peaks where jamming sets in, then falls",
                  min_top=0.1)
            if x == "density" else
            Panel("flux", "Flux past a point (cars/s, all lanes)", "cars crossing a fixed line per second",
                  min_top=0.1))
    return (
        flux,
        Panel("measured_mean_speed", "Mean speed (m/s)", "average over the cars on the ring"),
        Panel("jam_fraction", "Congestion: jammed cars (%)",
              "share of cars below 30 % of their free speed; rises steadily with density", scale=100.0,
              min_top=10.0),
        Panel("stops_per_car_min", "Instability: stop-and-go (stops/car/min)",
              f"~0 when smooth and also in a standstill, 1-2 in stop-and-go waves; line = onset {onset_threshold:g}",
              min_top=1.0, ref=onset_threshold),
        Panel("stop_go_onset", "Onset time (s from the start)",
              "mean over the runs that got there; a missing point = no run crossed that threshold; "
              "~0 = jammed from the start", empty="no run crossed either threshold",
              second="congestion_onset",
              styles=(f"stop-and-go index > {onset_threshold:g}",
                      f"jammed > {100 * congestion_threshold:g} % (30 s mean)")),
        Panel("n_jams", "Number of jams (count)", "runs of 2+ jammed cars in a lane, mean over time"),
        Panel("following_jam_share", "Front car following (% of jams)",
              "upper bound on natural jams (exit waves travel back behind followers); compare exit chance 0",
              scale=100.0, min_top=100.0, empty="no jams"),
        Panel("emergency_per_min", "Harsh braking (episodes/min)",
              f"asking for > {MAX_DECEL:g}\u00a0m/s² (a late or delayed over-reaction), or the no-overlap constraint",
              min_top=1.0),
        Panel("wave_speed", "Jam wave speed (m/s)", "< 0 = jams drift backwards against the traffic",
              from_zero=False, ref=0.0, empty="no runs jammed"),
        Panel("spacing_gini", "Spacing Gini (0-1)",
              "0 = even, ~0.35 = random gaps, towards 1 = packed queues or platoons; not a jam detector",
              min_top=0.5),
        Panel("clustering", "Clustering (dimensionless)",
              "sector spread beyond random: 0 = even or random, ~1 = half the ring empty", min_top=0.5),
        Panel("order_param", "Order parameter (0-1)", "0 = balanced, 1 = all on one side; only well above the dashed random-placement level is real",
              min_top=0.5, ref_column="order_baseline", ref_label="random placement"),
        Panel("lane_changes_per_min", "Lane changes (per minute)", "whole ring", min_top=1.0),
    )


def sweep_figure(df: pd.DataFrame, x: str, series: str | None, *, x_label: str,
                 series_label: Callable[[object], str], title: str, subtitle: str,
                 legend_title: str = "", onset_threshold: float = 0.5,
                 congestion_threshold: float = 0.3) -> Figure:
    """Small multiples of the sweep metrics against ``x``: mean across seeds, band = min-max.

    One line per value of ``series`` (in the order the values appear in
    ``df``). The flux chart (the fundamental diagram when ``x`` is density)
    takes a 2 x 2 block at the top left; twelve smaller charts fill the rest.
    Columns missing from ``df`` show as empty charts.
    """
    categorical = not pd.api.types.is_numeric_dtype(df[x])
    x_values = list(dict.fromkeys(df[x])) if categorical else sorted(df[x].unique())
    pos = {v: i for i, v in enumerate(x_values)} if categorical else None
    groups = [(None, df)] if series is None else [(v, df[df[series] == v]) for v in dict.fromkeys(df[series])]
    if len(groups) > len(SERIES_HEX):
        raise ValueError(f"at most {len(SERIES_HEX)} series values can be plotted at once")
    panels = sweep_panels(x, onset_threshold, congestion_threshold)

    def stats_of(g: pd.DataFrame, column: str, scale: float) -> tuple[np.ndarray, pd.DataFrame]:
        values = g[column] if column in g else pd.Series(np.nan, index=g.index)
        stats = values.groupby(g[x], sort=not categorical).agg(["mean", "min", "max"]) * scale
        xs = np.array([pos[v] for v in stats.index]) if categorical else stats.index.to_numpy(float)
        return xs, stats

    with matplotlib.rc_context({**RC, "axes.titlesize": 10.5}):
        fig = Figure(figsize=(20, 18), dpi=DPI, layout="constrained")
        fig.get_layout_engine().set(h_pad=0.08, w_pad=0.08, hspace=0.06, wspace=0.035)
        lines = _wrap(subtitle, fig.get_figwidth())
        legend_in = 0.65 if series else 0.0
        gs = fig.add_gridspec(5, 4, height_ratios=(
            _header_ratio(len(lines), fig.get_figheight(), 4.0, legend_in), 1, 1, 1, 1))
        header = fig.add_subplot(gs[0, :])
        _header(header, title, lines)
        slots = [gs[1:3, 0:2], gs[1, 2], gs[1, 3], gs[2, 2], gs[2, 3],
                 gs[3, 0], gs[3, 1], gs[3, 2], gs[3, 3], gs[4, 0], gs[4, 1], gs[4, 2], gs[4, 3]]
        axes = [fig.add_subplot(slot) for slot in slots]
        for ax in axes[1:]:
            ax.sharex(axes[0])
        labelled = {0, 9, 10, 11, 12}  # charts with no chart below them (and the hero)
        handles = []
        for i, (ax, panel) in enumerate(zip(axes, panels)):
            hero = i == 0
            peak = -np.inf  # largest |value| plotted; stays -inf if the panel has no data
            for k, (value, g) in enumerate(groups):
                color = SERIES_HEX[k]
                xs, stats = stats_of(g, panel.column, panel.scale)
                ax.fill_between(xs, stats["min"], stats["max"], color=color, alpha=0.12, linewidth=0)
                h = ax.plot(xs, stats["mean"], color=color, linewidth=LINE * (1.35 if hero else 1.0),
                            marker="o", markersize=7 if hero else 5, markeredgecolor=SURFACE,
                            markeredgewidth=1.2, label=series_label(value) if series else None)[0]
                if hero:
                    handles.append(h)
                drawn = [stats["max"]]
                if panel.second:
                    xs2, stats2 = stats_of(g, panel.second, panel.scale)
                    ax.plot(xs2, stats2["mean"], color=color, linewidth=LINE, linestyle=DASHED, marker="s",
                            markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1.0)
                    drawn.append(stats2["mean"])
                if panel.ref_column:
                    xs3, stats3 = stats_of(g, panel.ref_column, panel.scale)
                    ax.plot(xs3, stats3["mean"], color=color, linewidth=REF + 0.3, linestyle=DASHED, alpha=0.7,
                            zorder=1.5)
                    drawn.append(stats3["mean"])
                values = pd.concat(drawn).abs().dropna()
                if not values.empty:
                    peak = max(peak, float(values.max()))
            if not np.isfinite(peak):
                ax.text(0.5, 0.5, panel.empty, transform=ax.transAxes, ha="center", va="center", color=INK2)
            if panel.ref is not None:
                _ref_line(ax, panel.ref)
            if panel.from_zero:
                top = max(panel.min_top, 1.08 * peak) if np.isfinite(peak) and peak > 0 else max(panel.min_top, 1.0)
                ax.set_ylim(0, top)
            key = []  # line styles (colours are the figure legend)
            if panel.second:
                key = [Line2D([], [], color=INK, linewidth=LINE, marker="o", markersize=4.5, label=panel.styles[0]),
                       Line2D([], [], color=INK, linewidth=LINE, linestyle=DASHED, marker="s", markersize=4,
                              label=panel.styles[1])]
            elif panel.ref_column:
                key = [Line2D([], [], color=INK, linewidth=LINE, marker="o", markersize=4.5,
                              label=panel.title.split(" (")[0]),
                       Line2D([], [], color=INK2, linewidth=REF + 0.3, linestyle=DASHED, label=panel.ref_label)]
            if key:
                _legend_band(ax, 1, len(key), handles=key)
            _panel_title(ax, panel.title, panel.note,
                         **({"fontsize": 13} if hero else {}))
            ax.margins(x=0.03)
            if i in labelled:
                ax.set_xlabel(x_label)
                ax.tick_params(axis="x", labelbottom=True)
                if categorical:
                    ax.set_xticks(range(len(x_values)))
                    ax.set_xticklabels([str(v) for v in x_values])
            else:
                ax.tick_params(axis="x", labelbottom=False)
        if series:
            header.legend(handles, [h.get_label() for h in handles], loc="lower left", ncol=len(handles),
                          title=legend_title, alignment="left", title_fontsize=9.5, fontsize=9.5,
                          bbox_to_anchor=(0, 0), borderaxespad=0)
        _fit_notes(fig)
    return fig
