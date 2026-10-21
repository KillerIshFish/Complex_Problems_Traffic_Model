"""Shared colours for the viewer and the plots, so both read the same way.

* Speed (relative to a driver's free speed) is diverging: orange = stopped,
  grey = half speed, blue = free flow.
* Lanes / series use the categorical order below (never cycled).
* Occupancy / density uses a sequential blue ramp (light = empty).
"""

from __future__ import annotations

import numpy as np


def _hex(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


SLOW = _hex("#eb6834")
MID = _hex("#c8c7c0")
FAST = _hex("#2a78d6")
CRITICAL = _hex("#d03b3b")  # braking / perturbed cars

SERIES_HEX = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SERIES = [_hex(h) for h in SERIES_HEX]

SEQUENTIAL_HEX = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Dark UI surface (viewer)
UI = {
    "surface": _hex("#1a1a19"),
    "raised": _hex("#282826"),
    "text": _hex("#f0efec"),
    "text2": _hex("#c3c2b7"),
    "muted": _hex("#8c8b84"),
    "grid": _hex("#3c3c39"),
    "accent": _hex("#3987e5"),
    "road": _hex("#4a4a47"),
    "road_edge": _hex("#8c8b84"),
    "background": _hex("#232322"),
}

# Light chart surface (matplotlib figures)
PLOT = {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e4e3df"}


def speed_rgb(frac) -> np.ndarray:
    """RGB (0-255) for speed as a fraction of free speed (NaN = stopped); array in, (n, 3) out."""
    t = np.clip(np.nan_to_num(np.asarray(frac, dtype=float)), 0.0, 1.0)[..., None]
    slow, mid, fast = (np.array(c, dtype=float) for c in (SLOW, MID, FAST))
    lower = slow + (mid - slow) * (t / 0.5)
    upper = mid + (fast - mid) * ((t - 0.5) / 0.5)
    return np.where(t < 0.5, lower, upper)


def sequential_rgb(frac, dark: bool = False) -> np.ndarray:
    """RGB (0-255) on the sequential ramp; 0 = lightest step (darkest when ``dark``,
    so that "empty" recedes into a dark surface; NaN = 0). Array in, (n, 3) out."""
    steps = np.array([_hex(h) for h in (SEQUENTIAL_HEX[::-1] if dark else SEQUENTIAL_HEX)], dtype=float)
    t = np.clip(np.nan_to_num(np.asarray(frac, dtype=float)), 0.0, 1.0) * (len(steps) - 1)
    i = np.minimum(t.astype(int), len(steps) - 2)
    w = (t - i)[..., None]
    return steps[i] * (1 - w) + steps[i + 1] * w


def speed_cmap():
    """The same speed scale as a matplotlib colormap (0 = stopped, 1 = free)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "speed", [np.array(c) / 255 for c in (SLOW, MID, FAST)])


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("occupancy", SEQUENTIAL_HEX)
