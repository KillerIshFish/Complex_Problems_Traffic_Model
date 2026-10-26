"""Number and text formatting shared by the experiments CLI, the figures, the presets and the viewers.

Standard library only, so that any module can use it.
"""

from __future__ import annotations

import math
from numbers import Real

# Console-safe replacements (Windows terminals often mangle the rest).
ASCII = (("±", "+/-"), ("²", "2"), ("×", "x"), ("–", "-"), ("—", "-"), ("→", "->"), ("é", "e"), ("è", "e"),
         ("à", "a"), ("É", "E"))


def significant(value, digits: int = 3, max_places: int = 3, missing: str = "-") -> str:
    """``value`` with ``digits`` significant digits and no exponent, at most ``max_places`` decimals
    (``missing`` for None, NaN or infinity); whole numbers keep all their digits (1234). A value that
    rounds up to the next power of ten takes that one's decimals: 99.96 reads "100", not "100.0"."""
    if value is None:
        return missing
    v = float(value)
    if not math.isfinite(v):
        return missing
    if v == 0:
        return "0"

    def places(x: float) -> int:
        return min(max(digits - 1 - math.floor(math.log10(abs(x))), 0), max_places)

    text = f"{v:.{places(v)}f}"
    r = float(text)
    if r and places(r) < places(v):  # rounded up to the next power of ten
        text = f"{r:.{places(r)}f}"
    return text


def table_number(value: float) -> str:
    """A cell of a printed table: three significant digits, without an exponent from 1000 up
    (1650 veh/h, not 1.65e+03)."""
    return f"{value:.0f}" if abs(value) >= 999.5 else f"{value:.3g}"


def decimals(step: float) -> int:
    """Decimals of a slider step (0.05 -> 2, 5 -> 0)."""
    return 0 if step >= 1 else len(f"{step:g}".split(".")[1])


DISPLAY_DIGITS = 6  # significant digits of a value shown off its slider step


def format_number(value: float, step: float = 1.0) -> str:
    """A setting's value as the viewers show it: with the step's decimals (exit chance 0.1 -> "0.10", a
    radius of 80 m -> "80"), and as many more as a value between the steps needs to show exactly (a
    preset's 36.6 m or 18.25 m radius, 95.7 veh/km), up to DISPLAY_DIGITS significant digits (a random
    sample's 37.123456 -> "37.1235")."""
    v = float(value)
    if not math.isfinite(v):
        return "–" if math.isnan(v) else ("∞" if v > 0 else "-∞")
    places = decimals(step)
    most = max(places, DISPLAY_DIGITS - 1 - math.floor(math.log10(abs(v)))) if v else places
    while places < most and abs(round(v, places) - v) > 1e-9 * max(1.0, abs(v)):
        places += 1
    return f"{v:.{places}f}"


def value_text(value) -> str:
    """A setting as the experiments CLI takes it (``--set name=value``)."""
    if isinstance(value, float):
        return format(value, ".10g")
    return str(value)


def short_value(value) -> str:
    """A setting's value in a sentence or a label: numbers to 6 significant digits ("0.1", "80")."""
    return f"{value:g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


def plural(n, word: str) -> str:
    """'1 run', '3 runs'."""
    n = int(n)
    return f"{n} {word}{'s' if n != 1 else ''}"


def and_join(items) -> str:
    """'a', 'a and b', 'a, b and c'."""
    items = [str(i) for i in items]
    return ", ".join(items[:-1]) + f" and {items[-1]}" if len(items) > 1 else "".join(items)


def base_text(preset: str | None = None, random=None) -> str:
    """Where a sweep's base settings come from: 'preset etoile', 'random settings (seed 3)' or ''. The
    experiments CLI prints it and the sweep viewer shows it from the meta file."""
    if preset:
        return f"preset {preset}"
    if isinstance(random, bool) or not isinstance(random, Real) or not math.isfinite(random):
        return ""
    return f"random settings (seed {random:g})"


def plain(text: str) -> str:
    """ASCII version for the console (Windows terminals often mangle the rest)."""
    for old, new in ASCII:
        text = text.replace(old, new)
    return text.encode("ascii", "replace").decode("ascii")
