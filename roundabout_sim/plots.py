"""Figures for the experiments.

* :func:`dashboard`: one run over time; in open traffic mode a fifth row for
  the entrances (cars on the ring and queued, the entry delay with the
  level-of-service limits).
* :func:`sweep_figure`: sweep small multiples, each metric against the 1st
  varied setting with a line per value of the 2nd (any others averaged over,
  as the subtitle says); a sweep of open-mode runs adds the entrance charts
  (OPEN_PANELS).
* :func:`heatmap_figure`: a sweep of two or more settings as X x Y heatmaps of
  the headline metrics (:func:`headline_panels`).
* :func:`sample_figure`: a plan or random sample, every run as a dot against
  each varied setting, with the means per value or per bin.

All use the light chart tokens and the fixed categorical order from
:mod:`roundabout_sim.palette`, one y-axis per chart and a legend whenever a
chart has two or more series. Every panel title names the measure and its
unit; a smaller second line says how to read it. Figures are built with
``matplotlib.figure`` directly (no pyplot), so they never touch global GUI
state.
"""

from __future__ import annotations

import re
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

from roundabout_sim.config import PARAM_BY_NAME
from roundabout_sim.engine import MAX_DECEL, ROAD_CONDITIONS
from roundabout_sim.engine import braking_limits as road_braking_limits
from roundabout_sim.metrics import LOS_LIMITS, LOS_VC_TEXT, los_text
from roundabout_sim.palette import PLOT, SERIES_HEX, sequential_cmap, speed_cmap
from roundabout_sim.textfmt import significant

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


def braking_limits() -> str:
    """The braking limit on every road condition, e.g. "9 m/s² dry, 7.2 / 2.7 / 1.35 wet / snow / ice"."""
    others = [c for c in ROAD_CONDITIONS if c != "dry"]
    return f"{MAX_DECEL:g} m/s² dry, {road_braking_limits(others)} {' / '.join(others)}"


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


SUBTITLE_BREAKS = re.compile(r"(, |; | \| )")  # between the items of a subtitle ("Ring radius = 40 m, ...")


def _wrap(subtitle: str, fig_width: float) -> list[str]:
    """Subtitle lines: each ``\\n``-separated part wrapped to the figure width (10 pt text), between its
    items (at ", ", "; " and " | ") rather than inside one ("Mean desired speed = 11 m/s" stays whole),
    unless an item is too long to share a line."""
    chars = int(fig_width * 12.5)
    lines = []
    for part in subtitle.split("\n"):
        pieces = SUBTITLE_BREAKS.split(part)  # items at even places, separators at odd ones
        glued = "".join(p if i % 2 or len(p) > chars // 2 else p.replace(" ", "\xa0") for i, p in enumerate(pieces))
        lines += [line.replace("\xa0", " ") for line in textwrap.wrap(glued, chars, break_on_hyphens=False)]
    return lines


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


def _fit_header(fig: Figure, gs, header, gap: float = 0.15) -> None:
    """Size the header row to its title and subtitle, ``gap`` inches, then its legend (if any).

    :func:`_header_ratio` can only estimate it: the charts' titles, notes and
    labels take a share of the height that is known once the layout has run.
    So lay out once, measure the texts and the header the layout gave, and set
    the header's height ratio from that (call it after :func:`_fit_notes`).
    """
    canvas = fig.canvas if isinstance(fig.canvas, FigureCanvasAgg) else FigureCanvasAgg(fig)
    fig.draw_without_rendering()
    renderer = canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in header.texts]
    legend = header.get_legend()
    need = (max(b.y1 for b in boxes) - min(b.y0 for b in boxes)) / fig.dpi + gap
    if legend is not None:
        need += legend.get_window_extent(renderer).height / fig.dpi
    ratios = list(gs.get_height_ratios())
    height = fig.get_figheight()
    got = header.get_position().height * height
    margins = height - got * sum(ratios) / ratios[0]  # what the layout keeps for decorations and padding
    rest = sum(ratios[1:])
    gs.set_height_ratios([need * rest / max(height - margins - need, 0.1), *ratios[1:]])


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
              baseline: pd.DataFrame | None = None, arm_share: float = float("nan"),
              open_demand: bool = False, max_queue: int | None = None, entry_delay: float = float("nan"),
              entry_los: str = "") -> Figure:
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
    Rolling rates are hidden until their ``rate_window`` has filled. With
    ``open_demand`` (open traffic mode) the events panel also shows the
    arrivals, entries and blocked arrivals per minute, and a fifth row the
    entrances: the cars on the ring and in the queues (against ``max_queue``)
    and the entry delay with its level-of-service limits; ``entry_delay`` and
    ``entry_los`` are the run's late-half queue wait and its letter (as the
    printed summary gives them).
    """
    t = frame["time"].to_numpy()

    def rate(col: str) -> pd.Series:
        return frame[col].where(frame["time"] >= rate_window)

    with matplotlib.rc_context(RC):
        heights = (2.0, 1.15, 1.0, 1.0, 1.0) if open_demand else (2.0, 1.15, 1.0, 1.0)
        fig = Figure(figsize=(20, 18.5 + 3.6 * (len(heights) - 4)), dpi=DPI, layout="constrained")
        fig.get_layout_engine().set(h_pad=0.08, w_pad=0.08, hspace=0.035, wspace=0.03)
        lines = _wrap(subtitle, fig.get_figwidth())
        outer = fig.add_gridspec(2, 1, height_ratios=(
            _header_ratio(len(lines), fig.get_figheight(), sum(heights)), sum(heights)))
        _header(fig.add_subplot(outer[0]), title, lines)
        gs = outer[1].subgridspec(len(heights), 12, height_ratios=heights)

        # Row 1: where the jams are, and where the cars are
        _spacetime(fig, fig.add_subplot(gs[0, :9]), times, speeds, arm_deg, detector_deg, v_free, wave_speed)
        late = times >= late_from
        occ = occupancy[late].mean(axis=0) if late.any() and occupancy.size else np.zeros(occupancy.shape[1])
        _occupancy_ring(fig.add_subplot(gs[0, 9:], projection="polar"), 100.0 * occ, arm_deg)

        # Row 2: congestion over time (the headline) and what the front car of each jam is doing
        _congestion_panel(fig.add_subplot(gs[1, :6]), frame, late_from, congestion_onset, congestion_threshold,
                          rate_window, baseline, arm_share)
        _jam_origins_panel(fig.add_subplot(gs[1, 6:]), frame, following_share, late_from, open_demand)

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
                     f"for more than the braking limit ({braking_limits()}; an over-reaction after a late or "
                     f"delayed response), or the no-overlap constraint stopping a car harder", min_lines=3)
        events = {"emergency_per_min": "Harsh braking (episodes/min)", "lane_changes_per_min": "Lane changes",
                  "exits_per_min": "Exits"}
        if open_demand:  # what happens at the entrances when cars arrive at random
            events.update(arrivals_per_min="Arrivals", entries_per_min="Entries",
                          blocked_per_min="Blocked arrivals (queue full)")
        for k, (col, label) in enumerate(events.items()):
            _line(ax, t, rate(col), SERIES_HEX[k], label)
        ax.set_ylim(0, _top(*(rate(c) for c in events)))
        _legend_band(ax, 3, len(events))
        if not any(rate(c).notna().any() for c in events):
            ax.text(0.5, 0.45, f"shown once {rate_window:g} s have passed", transform=ax.transAxes,
                    ha="center", va="center", color=INK2)
        _time_axis(ax, frame)

        # The marker at the drivers' measured mean desired speed (trucks and spread included), as the mean
        # speed chart's grey line and the printed summary give it; v_free only sets the scale.
        desired = _late_mean(frame, "mean_free_speed", late_from)
        _speed_histogram(fig.add_subplot(gs[3, 8:]), car_speeds, v_free, late_from,
                         desired if np.isfinite(desired) else v_free)

        if open_demand:  # Row 5: the entrances, where open mode differs
            _queues_panel(fig.add_subplot(gs[4, :6]), frame, late_from, max_queue)
            _entry_delay_panel(fig.add_subplot(gs[4, 6:]), frame, late_from, rate_window, entry_delay, entry_los)
        _fit_notes(fig)
    return fig


def _queues_panel(ax, frame: pd.DataFrame, late_from: float, max_queue: int | None) -> None:
    """Open mode: the cars on the ring and waiting at the entrances over time, and the longest queue."""
    t = frame["time"].to_numpy()
    on_ring, queued = _late_mean(frame, "cars", late_from), _late_mean(frame, "queued", late_from)
    right = (f"{on_ring:.0f} cars on the ring, {queued:.0f} queued from {late_from:.0f} s on"
             if np.isfinite(on_ring) and np.isfinite(queued) else "")
    _panel_title(ax, "Cars on the ring and at the entrances (count)",
                 "open mode: cars arrive at random and leave, so the number on the ring varies; queued = waiting "
                 "at all entrances together; longest queue = the longest single one. An arrival finding its queue "
                 "at Max queue is turned away (blocked)", right=right, fontsize=12)
    cols = {"cars": "On the ring", "queued": "Queued (all entrances)", "longest_queue": "Longest queue"}
    shown = [c for c in cols if c in frame]
    for k, col in enumerate(cols):
        if col in frame:
            _line(ax, t, frame[col], SERIES_HEX[k], cols[col])
    top = _top(*(frame[c] for c in shown), floor=5.0)
    # The limit of each entrance's queue, while the longest queue gets near it (not where the cars on the ring
    # or all the queues together pass it: it is no limit of theirs), and only inside the axis it would squash.
    longest = float(frame["longest_queue"].max()) if "longest_queue" in frame else float("nan")
    if max_queue is not None and longest >= 0.5 * max_queue and max_queue <= top:
        _ref_line(ax, max_queue, f"Max queue per entrance ({max_queue} cars)")
    ax.set_ylim(0, top)
    entries = len(ax.get_legend_handles_labels()[1])
    _legend_band(ax, entries, entries)
    _time_axis(ax, frame)


def _entry_delay_panel(ax, frame: pd.DataFrame, late_from: float, window: float, delay: float,
                       los: str) -> None:
    """Open mode: the queue wait of the cars entering, over time, with the level-of-service limits."""
    t = frame["time"].to_numpy()
    if np.isfinite(delay):
        right, right_color = f"{delay:.0f} s queue wait from {late_from:.0f} s on: LOS {los}", INK
    else:
        right, right_color = f"No car entered after {late_from:.0f} s", INK2
    _panel_title(ax, "Entry delay (s, queue wait)",
                 f"mean wait of the cars that entered in the last {window:g} s, from joining the queue to entering "
                 f"the ring (not the full HCM control delay); dotted lines: HCM-style level-of-service limits "
                 f"{los_text()} (and {LOS_VC_TEXT})", right=right, right_color=right_color, fontsize=12)
    delays = frame["mean_entry_delay"] if "mean_entry_delay" in frame else pd.Series(np.nan, index=frame.index)
    _line(ax, t, delays, SERIES_HEX[0], f"Mean entry delay ({window:g} s)")
    top = _top(delays, floor=20.0)
    ax.set_ylim(0, top)
    _los_lines(ax, top)
    _legend_band(ax, 1, 1)
    if not delays.notna().any():  # (midway between the A and B limits: clear of the dotted lines)
        ax.text(0.5, 0.5 * (LOS_LIMITS[0][1] + LOS_LIMITS[1][1]), "no car entered", ha="center", va="center",
                transform=ax.get_yaxis_transform(), color=INK2, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2))
    _time_axis(ax, frame)


def _los_lines(ax, top: float) -> None:
    """The level-of-service limits (LOS_LIMITS) below ``top`` as dotted lines, and each band's letter just
    right of the chart (clear of the data, which runs to the edge) where the band is tall enough to hold it.
    Drawn after the y limits are set: they never raise the top."""
    edges = [0.0, *(v for _, v in LOS_LIMITS), np.inf]
    letters = [k for k, _ in LOS_LIMITS] + ["F"]
    for v in edges[1:-1]:
        if v < top:
            ax.axhline(v, color=INK2, linewidth=0.8, linestyle=":", zorder=1)
    for letter, lo, hi in zip(letters, edges[:-1], edges[1:]):
        hi = min(hi, top)
        if lo < top and (hi - lo) / top >= 0.08:
            ax.annotate(letter, xy=(1, 0.5 * (lo + hi)), xycoords=("axes fraction", "data"), xytext=(4, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=8.5, color=INK2,
                        annotation_clip=False)


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


def _jam_origins_panel(ax, frame: pd.DataFrame, following_share: float, late_from: float,
                       open_demand: bool = False) -> None:
    """Stacked columns per time bin: mean number of jams, split by what their front car is doing.

    In open mode there is no exit-chance-0 baseline to compare with (the
    arrivals would never leave) and no density control slotting cars in.
    """
    if np.isfinite(following_share):
        right, right_color = (f"front car following: {100 * following_share:.0f} % of jams from {late_from:.0f} s on",
                              INK)
    else:
        right, right_color = f"No jams after {late_from:.0f} s", INK2
    note = ("following is only an upper bound on natural (phantom) jams: a wave set off at an exit travels "
            "backwards with ordinary followers at its front")
    if not open_demand:
        note += ". For what the entrances/exits really cause, compare with exit chance 0 (run --baseline)"
    _panel_title(ax, "Jams by what their front car is doing (count)", note,
                 right=right, right_color=right_color, fontsize=12)
    labels = dict(JAM_ORIGINS)
    if open_demand:
        labels["jams_other"] = "Braked by you (in the viewer)"
    t_end = float(frame["time"].max()) if len(frame) else 1.0
    width = _bin_width(t_end)
    cols = [c for c, _ in JAM_ORIGINS if c in frame]
    means = frame.groupby((frame["time"] // width).astype(int))[cols].mean()
    left = means.index.to_numpy(float) * width
    bottom = np.zeros(len(means))
    for k, (col, _) in enumerate(JAM_ORIGINS):
        if col not in means:
            continue
        values = means[col].to_numpy(float)
        ax.bar(left, values, width=width, bottom=bottom, align="edge", color=SERIES_HEX[k],
               edgecolor=SURFACE, linewidth=1.2, label=labels[col])
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


def _speed_histogram(ax, car_speeds: np.ndarray, v_free: float, late_from: float, desired: float) -> None:
    """Share of the car samples per 0.5 m/s, with a marker at the drivers' mean ``desired`` speed (the scale
    reaches 1.25 x ``v_free``, the free speed of an average driver)."""
    top = max(v_free * 1.25, desired * 1.1, float(np.max(car_speeds)) if car_speeds.size else 1.0)
    width = 0.5
    edges = np.arange(0.0, top + width, width)
    counts, _ = np.histogram(car_speeds, bins=edges)
    share = 100 * counts / max(counts.sum(), 1)
    ax.bar(edges[:-1], share, width=width, align="edge", color=SERIES_HEX[0], edgecolor=SURFACE, linewidth=1.0)
    ax.axvline(desired, color=INK2, linewidth=REF)
    ax.annotate("mean desired speed", xy=(desired, 0.97), xycoords=("data", "axes fraction"),
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
    los: bool = False  # dotted lines at the level-of-service limits (an entry delay in s)
    lines: str = ""  # what its reference lines (ref, los) are: joined to the note where they are drawn

    def full_note(self) -> str:
        """The note with what the reference lines are, for a chart that draws them."""
        return "; ".join(part for part in (self.note, self.lines) if part)


def sweep_panels(x: str, onset_threshold: float, congestion_threshold: float = 0.3,
                 open_demand: bool = False) -> tuple[Panel, ...]:
    """The sweep charts, hero (flux) first, then speed and congestion, jams, and congregation; with
    ``open_demand`` the entrance charts (OPEN_PANELS) last."""
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
              "~0 when smooth and also in a standstill, 1-2 in stop-and-go waves",
              min_top=1.0, ref=onset_threshold, lines=f"line = onset {onset_threshold:g}"),
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
              f"asking for more than the braking limit ({braking_limits()}; a late or delayed over-reaction), "
              "or the no-overlap constraint", min_top=1.0),
        Panel("wave_speed", "Jam wave speed (m/s)", "< 0 = jams drift backwards against the traffic",
              from_zero=False, ref=0.0, empty="no runs jammed"),
        Panel("spacing_gini", "Spacing Gini (0-1)",
              "0 = even, ~0.35 = random gaps, towards 1 = packed queues or platoons; not a jam detector",
              min_top=0.5),
        Panel("clustering", "Clustering (dimensionless)",
              "sector spread beyond random: 0 = even or random, ~1 = half the ring empty", min_top=0.5),
        Panel("order_param", "Order parameter (0-1)",
              "0 = balanced, 1 = all on one side; only well above the random-placement level is real",
              min_top=0.5, ref_column="order_baseline", ref_label="random placement"),
        Panel("lane_changes_per_min", "Lane changes (per minute)", "whole ring", min_top=1.0),
        *(OPEN_PANELS if open_demand else ()),
    )


# Open mode (cars arrive at random): what happens at the entrances, a fifth row of the sweep figure.
OPEN_PANELS = (
    Panel("mean_entry_delay", "Mean entry delay (s, queue wait)",
          f"queue wait of the cars that entered (not the full control delay); HCM-style level of service "
          f"{los_text()}, and {LOS_VC_TEXT}",
          min_top=10.0, empty="no car entered", los=True, lines="lines = each limit"),
    Panel("throughput_per_h", "Throughput (veh/h)", "cars leaving the ring per hour, all exits", min_top=100.0),
    Panel("longest_queue", "Longest queue (cars)",
          "the longest entrance queue: mean over time and the longest seen", min_top=5.0,
          second="peak_queue", styles=("mean over time", "longest seen")),
    Panel("blocked_per_min", "Blocked arrivals (per minute)",
          "arrivals that found their queue full (max queue) and were turned away: spill-back", min_top=1.0),
)
# The headline metrics of the heatmaps and of the plan / sample scatter (+ OPEN_HEADLINE in open mode).
HEADLINE = ("flux", "jam_fraction", "stops_per_car_min")
OPEN_HEADLINE = ("mean_entry_delay", "throughput_per_h")


def headline_panels(open_demand: bool = False, onset_threshold: float = 0.5) -> list[Panel]:
    """The sweep charts of the headline metrics (with the entrance ones in open mode)."""
    by_column = {p.column: p for p in sweep_panels("", onset_threshold, open_demand=True)}
    return [by_column[c] for c in HEADLINE + (OPEN_HEADLINE if open_demand else ())]


def _ordered(col: pd.Series, by_choice: bool = True) -> tuple[list, bool]:
    """The values of a setting column in axis order and whether it is categorical: numbers ascending, text
    in the order its parameter lists the choices (``by_choice``; for a random sample, where the order the
    values appear in is random) or as the values appear (a grid: the order given, as in its line charts).
    Numbers with an infinite value (Density or Reaction distance = inf, both valid settings) have no place
    on a scale: they are categorical too, ascending with inf last, as the sweep viewer shows them."""
    if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
        values = sorted(col.dropna().unique())
        return values, not bool(np.isfinite(np.asarray(values, dtype=float)).all())
    seen = list(dict.fromkeys(col.dropna()))
    if not by_choice:
        return seen, True
    spec = PARAM_BY_NAME.get(str(col.name))
    order = list(spec.choices) if spec and spec.choices else []
    return sorted(seen, key=lambda v: (order.index(v) if v in order else len(order), seen.index(v))), True


def _tick_text(value_text: Callable[[object], str]) -> Callable[[object], str]:
    """``value_text`` for tick labels, with an infinite number as "∞" (not "inf")."""
    def text(v) -> str:
        if isinstance(v, (float, np.floating)) and np.isinf(v):
            return "∞" if v > 0 else "-∞"
        return value_text(v)
    return text


def _thin_ticks(ax, axis: str, labels: list[str], most: int = 14) -> None:
    """Category ticks on ``axis``, every k-th one labelled when there are more than ``most``."""
    every = -(-len(labels) // most)
    ticks = range(0, len(labels), every)
    (ax.set_xticks if axis == "x" else ax.set_yticks)(list(ticks))
    (ax.set_xticklabels if axis == "x" else ax.set_yticklabels)([labels[i] for i in ticks])


def _cell_text(value: float) -> str:
    """A heatmap cell's value: three significant digits, no exponent (99.96 -> "100", next to a "100")."""
    return significant(value)


def _scale_limits(values: np.ndarray, panel: Panel) -> tuple[float, float]:
    """Colour or axis range of a panel: from 0 (when it starts there) to at least its min_top."""
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if panel.from_zero and lo >= 0:
        lo, hi = 0.0, max(hi, panel.min_top)
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def heatmap_figure(df: pd.DataFrame, x: str, y: str, panels: Sequence[Panel], *, x_label: str, y_label: str,
                   title: str, subtitle: str, value_text: Callable[[object], str] = str) -> Figure:
    """One heatmap per headline metric: x by y cells coloured by the mean of the runs in each.

    Every run at a cell's pair of values counts (its seeds and, in a sweep of
    three or more settings, all the values of the others). Light = low, dark =
    high on each chart's own colour bar (from 0 for measures that start there);
    blank = no run with a value. Cells show their value at the largest size
    that fits them (none when even the smallest does not). Numbers run
    ascending, text in the order it appears in ``df`` (as in
    :func:`sweep_figure`).
    """
    x_values, _ = _ordered(df[x], by_choice=False)
    y_values, _ = _ordered(df[y], by_choice=False)
    cols = min(3, len(panels))
    rows = -(-len(panels) // cols)
    cmap = sequential_cmap().with_extremes(bad=SURFACE)
    labelled = len(x_values) <= 16 and len(y_values) <= 12
    cells = []  # (axes, its cell texts) of the labelled heatmaps
    with matplotlib.rc_context(RC):
        lines = _wrap(subtitle, 20)
        height = 1.4 + 0.25 * len(lines) + 5.6 * rows
        fig = Figure(figsize=(20, height), dpi=DPI, layout="constrained")
        fig.get_layout_engine().set(h_pad=0.08, w_pad=0.08, hspace=0.06, wspace=0.05)
        gs = fig.add_gridspec(rows + 1, cols, height_ratios=(_header_ratio(len(lines), height, rows), *[1] * rows))
        _header(fig.add_subplot(gs[0, :]), title, lines)
        for i, panel in enumerate(panels):
            ax = fig.add_subplot(gs[1 + i // cols, i % cols])
            values = df[panel.column] if panel.column in df else pd.Series(np.nan, index=df.index)
            grid = (pd.to_numeric(values, errors="coerce").groupby([df[y], df[x]]).mean().unstack()
                    .reindex(index=y_values, columns=x_values) * panel.scale)
            data = grid.to_numpy(float)
            if np.isfinite(data).any():
                lo, hi = _scale_limits(data, panel)
                im = ax.imshow(np.ma.masked_invalid(data), origin="lower", aspect="auto", cmap=cmap, vmin=lo,
                               vmax=hi, interpolation="nearest")
                bar = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.05, aspect=25)
                bar.outline.set_visible(False)
                bar.ax.tick_params(colors=INK2)
                if labelled:
                    texts = [ax.text(c, r, _cell_text(v), ha="center", va="center", fontsize=CELL_SIZES[0],
                                     color=SURFACE if (v - lo) / (hi - lo) > 0.55 else INK)
                             for (r, c), v in np.ndenumerate(data) if np.isfinite(v)]
                    for text in texts:
                        text.set_in_layout(False)
                    cells.append((ax, texts))
            else:
                ax.set_xlim(-0.5, len(x_values) - 0.5)
                ax.set_ylim(-0.5, len(y_values) - 0.5)
                ax.text(0.5, 0.5, panel.empty, transform=ax.transAxes, ha="center", va="center", color=INK2)
            _thin_ticks(ax, "x", [_tick_text(value_text)(v) for v in x_values])
            _thin_ticks(ax, "y", [_tick_text(value_text)(v) for v in y_values])
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            _panel_title(ax, panel.title, panel.note, fontsize=12)  # (no reference lines here)
        _fit_notes(fig)
        _fit_header(fig, gs, fig.axes[0])
        _fit_cell_texts(fig, cells, len(x_values), len(y_values))
    return fig


CELL_SIZES = (10.5, 8.5, 7.5)  # [pt] a heatmap's cell values: the largest that fits every cell, else none


def _fit_cell_texts(fig: Figure, cells: list, columns: int, rows: int) -> None:
    """Size the heatmaps' cell values to the largest of CELL_SIZES at which every chart's widest value fits
    85 % of a cell's width (and 80 % of its height); a chart where even the smallest does not fit shows
    none. Measured once the layout is final (call it after :func:`_fit_header`), so neighbouring values never
    run into each other or past the edge, and one size serves the whole figure."""
    if not cells:
        return
    canvas = fig.canvas if isinstance(fig.canvas, FigureCanvasAgg) else FigureCanvasAgg(fig)
    fig.draw_without_rendering()
    renderer = canvas.get_renderer()

    def fits(ax, texts, size: float) -> bool:
        for text in texts:
            text.set_fontsize(size)
        box = ax.get_window_extent(renderer)
        extents = [text.get_window_extent(renderer) for text in texts]
        return (max(e.width for e in extents) <= 0.85 * box.width / columns
                and max(e.height for e in extents) <= 0.8 * box.height / rows)

    best = {}  # chart -> index of the largest size that fits it
    for k, (ax, texts) in enumerate(cells):
        best[k] = next((i for i, size in enumerate(CELL_SIZES) if fits(ax, texts, size)), None)
    shared = max((i for i in best.values() if i is not None), default=None)
    for k, (_, texts) in enumerate(cells):
        for text in texts:
            text.set_fontsize(CELL_SIZES[shared if shared is not None else -1])
            text.set_visible(best[k] is not None)


# A number in a plan or a random sample with more distinct values than this is binned (equal-width bins
# of it), in the sample figure and the sweep viewer's scatter alike; with fewer, the mean at each value.
BIN_ABOVE = {"plan": 12, "sample": 6}


def binned_stats(x: np.ndarray, y: np.ndarray, bins: int | None = None):
    """Equal-width bins of ``x``: (centres, means, mins, maxs, counts, edges) of the finite ``y`` in each
    (NaN where a bin is empty); about sqrt(n) bins, 3-12, unless ``bins`` is given. A bin's centre is
    the mean x of its runs, so that its mean is drawn where they are. The sample figure and the sweep
    viewer's scatter both bin with it."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0:
        e = np.zeros(0)
        return e, e, e, e, np.zeros(0, dtype=int), e
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        one = np.array([lo])
        return one, np.array([y.mean()]), np.array([y.min()]), np.array([y.max()]), np.array([x.size]), \
            np.array([lo, hi])
    bins = bins or int(np.clip(round(np.sqrt(x.size)), 3, 12))
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)
    counts = np.bincount(idx, minlength=bins)
    sums = np.bincount(idx, weights=y, minlength=bins)
    x_sums = np.bincount(idx, weights=x, minlength=bins)
    mins, maxs = np.full(bins, np.inf), np.full(bins, -np.inf)
    np.minimum.at(mins, idx, y)
    np.maximum.at(maxs, idx, y)
    empty = counts == 0
    means = np.where(empty, np.nan, sums / np.maximum(counts, 1))
    centres = np.where(empty, (edges[:-1] + edges[1:]) / 2, x_sums / np.maximum(counts, 1))
    return centres, means, np.where(empty, np.nan, mins), np.where(empty, np.nan, maxs), counts, edges


def _bin_means(x: np.ndarray, y: np.ndarray, distinct_max: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Mean of ``y`` at each value of ``x`` when it takes at most ``distinct_max`` values, else in about
    sqrt(n) (3-12) equal-width bins of ``x`` (:func:`binned_stats`), each drawn at the mean ``x`` of its runs."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0:
        return np.zeros(0), np.zeros(0)
    values = np.unique(x)
    if values.size <= distinct_max:
        return values, np.array([y[x == v].mean() for v in values])
    centres, means, _, _, counts, _ = binned_stats(x, y)
    keep = counts > 0
    return centres[keep], means[keep]


def sample_figure(df: pd.DataFrame, params: Sequence[str], panels: Sequence[Panel], *,
                  param_label: Callable[[str], str], title: str, subtitle: str,
                  value_text: Callable[[object], str] = str, distinct_max: int = 12, means: bool = True) -> Figure:
    """A plan or random sample: every run as a dot against each varied setting, one row per setting and one
    column per headline metric, with the mean of the runs per bin (or per value) as a line (none without
    ``means``: a plan in which nothing varies shows its runs against their number).

    The other settings vary at the same time, so a line shows a setting's
    effect averaged over them. A choice gets its values side by side (the dots
    spread a little sideways so they do not hide each other). A number with at
    most ``distinct_max`` values gets the mean per value, one with more the
    means of equal-width bins (a plan: its values; a random sample: bins).
    """
    rows, cols = max(1, len(params)), len(panels)
    jitter = np.random.default_rng(0).uniform(-0.18, 0.18, len(df))  # its own generator: nothing else moves
    with matplotlib.rc_context({**RC, "axes.titlesize": 10.5}):
        lines = _wrap(subtitle, 20)
        legend_in = 0.45
        height = 1.6 + 0.25 * len(lines) + legend_in + 2.9 * rows
        fig = Figure(figsize=(20, height), dpi=DPI, layout="constrained")
        fig.get_layout_engine().set(h_pad=0.08, w_pad=0.08, hspace=0.06, wspace=0.04)
        gs = fig.add_gridspec(rows + 1, cols, height_ratios=(
            _header_ratio(len(lines), height, rows, legend_in), *[1] * rows))
        header = fig.add_subplot(gs[0, :])
        _header(header, title, lines)
        for r, name in enumerate(params):
            levels, categorical = _ordered(df[name])
            if categorical:
                codes = df[name].map({v: i for i, v in enumerate(levels)}).to_numpy(float)
                xs = codes + jitter
            else:
                codes = xs = df[name].to_numpy(float)
            for c, panel in enumerate(panels):
                ax = fig.add_subplot(gs[1 + r, c])
                values = df[panel.column] if panel.column in df else pd.Series(np.nan, index=df.index)
                ys = pd.to_numeric(values, errors="coerce").to_numpy(float) * panel.scale
                ax.scatter(xs, ys, s=16 if means else 24, color=SERIES_HEX[0], alpha=0.45 if means else 0.8,
                           linewidths=0, zorder=2)
                if means:
                    centres, mean = _bin_means(codes, ys, distinct_max)
                    ax.plot(centres, mean, color=INK, linewidth=LINE, linestyle="none" if categorical else "-",
                            marker="D" if categorical else "o", markersize=6 if categorical else 4.5,
                            markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
                if np.isfinite(ys).any():
                    lo, hi = _scale_limits(ys, panel)
                    ax.set_ylim(lo, hi * 1.08 if panel.from_zero and lo >= 0 else hi + 0.05 * (hi - lo))
                    if panel.los:
                        _los_lines(ax, ax.get_ylim()[1])
                else:
                    ax.text(0.5, 0.5, panel.empty, transform=ax.transAxes, ha="center", va="center", color=INK2)
                if panel.ref is not None:
                    _ref_line(ax, panel.ref)
                if categorical:
                    ax.set_xticks(range(len(levels)))
                    ax.set_xticklabels([_tick_text(value_text)(v) for v in levels])
                    ax.set_xlim(-0.5, len(levels) - 0.5)
                    ax.grid(axis="x", visible=False)
                elif pd.api.types.is_integer_dtype(df[name]):
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_xlabel(param_label(name))
                if r == 0:
                    _panel_title(ax, panel.title, panel.full_note(), min_lines=2)
        handles = [Line2D([], [], color=SERIES_HEX[0], marker="o", linestyle="none", markersize=5, alpha=0.6,
                          label="one run")]
        if means:
            handles.append(Line2D([], [], color=INK, linewidth=LINE, marker="o", markersize=4.5,
                                  label="mean of the runs per bin of the setting (per value when it takes few)"))
        header.legend(handles=handles, loc="lower left", ncol=2, fontsize=9.5, bbox_to_anchor=(0, 0),
                      borderaxespad=0)
        _fit_notes(fig)
        _fit_header(fig, gs, header)
    return fig


def sweep_figure(df: pd.DataFrame, x: str, series: str | None, *, x_label: str,
                 series_label: Callable[[object], str], title: str, subtitle: str,
                 legend_title: str = "", onset_threshold: float = 0.5,
                 congestion_threshold: float = 0.3, open_demand: bool = False,
                 value_text: Callable[[object], str] = str) -> Figure:
    """Small multiples of the sweep metrics against ``x``: mean across seeds, band = min-max.

    One line per value of ``series`` (in the order the values appear in
    ``df``). The flux chart (the fundamental diagram when ``x`` is density)
    takes a 2 x 2 block at the top left; twelve smaller charts fill the rest,
    and with ``open_demand`` (open traffic mode) a fifth row shows the
    entrances: entry delay, throughput, queues and blocked arrivals. Every
    other column of ``df`` is pooled: with more swept settings than ``x`` and
    ``series`` each point averages over them. Columns missing from ``df`` show
    as empty charts. Text values of ``x``, and numbers with an infinite one
    (:func:`_ordered`), sit evenly spaced, labelled with ``value_text``.
    """
    x_values, categorical = _ordered(df[x], by_choice=False)
    pos = {v: i for i, v in enumerate(x_values)} if categorical else None
    groups = [(None, df)] if series is None else [(v, df[df[series] == v]) for v in dict.fromkeys(df[series])]
    if len(groups) > len(SERIES_HEX):
        raise ValueError(f"at most {len(SERIES_HEX)} series values can be plotted at once")
    panels = sweep_panels(x, onset_threshold, congestion_threshold, open_demand)
    rows = 5 if open_demand else 4  # rows of charts below the header

    def stats_of(g: pd.DataFrame, column: str, scale: float) -> tuple[np.ndarray, pd.DataFrame]:
        values = g[column] if column in g else pd.Series(np.nan, index=g.index)
        stats = values.groupby(g[x], sort=not categorical).agg(["mean", "min", "max"]) * scale
        if not categorical:
            return stats.index.to_numpy(float), stats
        xs = np.array([pos[v] for v in stats.index], dtype=float)
        order = np.argsort(xs, kind="stable")  # along the axis, whatever order the runs came in
        return xs[order], stats.iloc[order]

    with matplotlib.rc_context({**RC, "axes.titlesize": 10.5}):
        fig = Figure(figsize=(20, 18 + 4 * (rows - 4)), dpi=DPI, layout="constrained")
        fig.get_layout_engine().set(h_pad=0.08, w_pad=0.08, hspace=0.06, wspace=0.035)
        lines = _wrap(subtitle, fig.get_figwidth())
        legend_in = 0.65 if series else 0.0
        gs = fig.add_gridspec(rows + 1, 4, height_ratios=(
            _header_ratio(len(lines), fig.get_figheight(), float(rows), legend_in), *[1] * rows))
        header = fig.add_subplot(gs[0, :])
        _header(header, title, lines)
        slots = [gs[1:3, 0:2], gs[1, 2], gs[1, 3], gs[2, 2], gs[2, 3],
                 *(gs[row, col] for row in range(3, rows + 1) for col in range(4))]
        axes = [fig.add_subplot(slot) for slot in slots]
        for ax in axes[1:]:
            ax.sharex(axes[0])
        labelled = {0, *range(len(slots) - 4, len(slots))}  # charts with no chart below them (and the hero)
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
                            markeredgewidth=1.2, label=_tick_text(series_label)(value) if series else None)[0]
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
                if panel.los:
                    _los_lines(ax, top)
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
            _panel_title(ax, panel.title, panel.full_note(),
                         **({"fontsize": 13} if hero else {}))
            ax.margins(x=0.03)
            if i in labelled:
                ax.set_xlabel(x_label)
                ax.tick_params(axis="x", labelbottom=True)
                if categorical:
                    ax.set_xticks(range(len(x_values)))
                    ax.set_xticklabels([_tick_text(value_text)(v) for v in x_values])
            else:
                ax.tick_params(axis="x", labelbottom=False)
        if series:
            header.legend(handles, [h.get_label() for h in handles], loc="lower left", ncol=len(handles),
                          title=legend_title, alignment="left", title_fontsize=9.5, fontsize=9.5,
                          bbox_to_anchor=(0, 0), borderaxespad=0)
        _fit_notes(fig)
        _fit_header(fig, gs, header)
    return fig
