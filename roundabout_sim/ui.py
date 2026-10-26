"""Pieces both pygame windows share: the main viewer (:mod:`roundabout_sim.app`) and the sweep viewer
(:mod:`roundabout_sim.sweep_viewer`).

The window set-up (DPI awareness, the first size, the smallest size a window may have), the fonts
and text fitting (cut with an ellipsis, wrapped), the help box's height and its passing messages,
the tabs and the rows of buttons that share out a width.
"""

from __future__ import annotations

import ctypes
import math
import sys
import time

import numpy as np
import pygame

from roundabout_sim import palette

UI = palette.UI
FONT_FACE = "segoeui,helveticaneue,arial"
MIN_WINDOW = (1100, 700)
STATUS_LINES = (6, 10)  # fewest / most lines of the help box
PIN_HOLD = 2.0  # [s] a message from a key shows ahead of the hover help at least this long


# ------------------------------------------------------------------ window
def dpi_aware() -> None:
    """Render at native resolution on high-DPI Windows displays instead of being upscaled blurrily."""
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass


def initial_size() -> tuple[int, int]:
    """About 90 % of the desktop, but at least ``MIN_WINDOW`` where the screen allows."""
    sizes = pygame.display.get_desktop_sizes()
    w, h = sizes[0] if sizes else (MIN_WINDOW[0], MIN_WINDOW[1])
    return (int(max(0.9 * w, min(MIN_WINDOW[0], w))), int(max(0.9 * h, min(MIN_WINDOW[1], h))))


def clamp_size(size: tuple[int, int]) -> tuple[int, int]:
    """``size`` raised to MIN_WINDOW where it is smaller (unless the screen itself is smaller): the
    smallest window whose layout stays legible."""
    desk = (pygame.display.get_desktop_sizes() or [MIN_WINDOW])[0]
    return (max(int(size[0]), min(MIN_WINDOW[0], desk[0])), max(int(size[1]), min(MIN_WINDOW[1], desk[1])))


def open_window(size: tuple[int, int] | None) -> pygame.Surface:
    """The resizable window at ``size`` (a --size; at least :func:`clamp_size`, with a note on stderr
    when that raised it), else at :func:`initial_size`."""
    if size is None:
        return pygame.display.set_mode(initial_size(), pygame.RESIZABLE)
    fitted = clamp_size(size)
    if fitted != tuple(size):
        print(f"note: window size {size[0]}x{size[1]} is below the smallest the layout supports; using "
              f"{fitted[0]}x{fitted[1]}", file=sys.stderr)
    return pygame.display.set_mode(fitted, pygame.RESIZABLE)


def clamp_resize(screen: pygame.Surface, size: tuple[int, int]) -> pygame.Surface:
    """The window after a resize to ``size``: never smaller than MIN_WINDOW (unless the screen
    itself is smaller, see :func:`clamp_size`); set anew when that differs from what it is (clamped,
    injected, or a driver that did not resize)."""
    size = clamp_size(size)
    return screen if screen.get_size() == size else pygame.display.set_mode(size, pygame.RESIZABLE)


# ------------------------------------------------------------------ text
class Fonts:
    """System fonts at one UI scale, with a cache of rendered strings."""

    def __init__(self, scale: float):
        self.scale = scale
        px = lambda size: max(10, round(size * scale))  # noqa: E731
        self.fonts = {
            "title": pygame.font.SysFont(FONT_FACE, px(18), bold=True),
            "base": pygame.font.SysFont(FONT_FACE, px(14)),
            "small": pygame.font.SysFont(FONT_FACE, px(12)),
        }
        self._cache: dict[tuple, pygame.Surface] = {}

    def render(self, text: str, font: str = "base", colour=UI["text"]) -> pygame.Surface:
        key = (text, font, colour)
        surf = self._cache.get(key)
        if surf is None:
            if len(self._cache) > 4000:
                self._cache.clear()
            surf = self._cache[key] = self.fonts[font].render(text, True, colour)
        return surf

    def blit(self, target: pygame.Surface, text: str, pos, font: str = "base", colour=UI["text"],
             anchor: str = "topleft") -> pygame.Rect:
        surf = self.render(text, font, colour)
        rect = surf.get_rect(**{anchor: (round(pos[0]), round(pos[1]))})
        target.blit(surf, rect)
        return rect

    def width(self, text: str, font: str = "base") -> int:
        return self.fonts[font].size(text)[0]

    def line(self, font: str = "base") -> int:
        return self.fonts[font].get_linesize()

    def wrap(self, text: str, width: int, font: str = "small") -> list[str]:
        lines, current = [], ""
        for word in text.split(" "):
            trial = f"{current} {word}".strip()
            if current and self.width(trial, font) > width:
                lines.append(current)
                current = word
            else:
                current = trial
            while len(current) > 1 and self.width(current, font) > width:  # a word longer than a line (a path)
                n = len(current) - 1
                while n > 1 and self.width(current[:n], font) > width:
                    n -= 1
                lines.append(current[:n])
                current = current[n:]
        return lines + [current] if current else lines


def ellipsize(fonts: Fonts, text: str, width: int, font: str = "small") -> str:
    """``text``, cut with an ellipsis to fit ``width`` px."""
    if fonts.width(text, font) <= width:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fonts.width(text[:mid].rstrip() + "…", font) <= width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + "…" if lo else ""


def wrap_lines(fonts: Fonts, text: str, width: int, lines: int, font: str = "small") -> list[str]:
    """``text`` wrapped to ``width``, at most ``lines`` lines (the last one cut with an ellipsis)."""
    wrapped = fonts.wrap(text, width, font)
    if len(wrapped) > lines:
        wrapped = wrapped[:lines - 1] + [ellipsize(fonts, " ".join(wrapped[lines - 1:]), width, font)]
    return wrapped


# ------------------------------------------------------------------ axes
def finite_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) index ranges of the runs of True in a boolean array (the stretches of a line
    drawn between its undefined values)."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], valid.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def nice_step(span: float, target: int) -> float:
    """A round tick step (1, 2, 2.5 or 5 times a power of ten) giving about ``target`` ticks over ``span``."""
    raw = max(span, 1e-9) / max(target, 1)
    if not math.isfinite(raw):
        return 1.0  # no scale for an infinite span: callers draw no ticks
    mag = 10.0 ** math.floor(math.log10(raw))
    return next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag * 1.0001)


def tick_text(value: float, step: float) -> str:
    """A tick label with as many decimals as the tick step has (2.5 -> one: 7.5, not a rounded 8)."""
    text = f"{step:.10f}".rstrip("0")
    places = min(len(text.split(".")[1]), 6) if "." in text else 0
    return f"{0.0 if abs(value) < 1e-9 * step else value:.{places}f}"


# ------------------------------------------------------------------ help box
def status_lines(fonts, width: int, texts, fit: tuple) -> tuple[tuple, int]:
    """Lines the help box needs so that the longest of ``texts`` (a callable giving them) fits
    ``width``, within STATUS_LINES. ``fit`` is the last answer, ((font scale, width), lines), given
    back as is while the font scale and width stay, so the box keeps its height whatever it shows."""
    key = (fonts.scale, width)
    if fit[0] != key:
        need = max(len(fonts.wrap(text, width)) for text in texts())
        fit = (key, int(np.clip(need, *STATUS_LINES)))
    return fit


class Messages:
    """The help box's passing message (``message``: its text and the wall-clock time it goes).

    Hover help comes first, so a message would hide whenever the mouse rests on a control, a chart
    or the view. It comes first instead while the mouse stays on the control whose click gave it,
    and, when it came from a key or by itself (the session resumed, a sweep finished), for
    PIN_HOLD s and after that until the mouse moves. The viewer sets ``mouse`` and calls
    :meth:`_motion` on every mouse move, :meth:`_clicked_on` before a click's action, and runs keys
    through :meth:`_by_key`."""

    mouse: tuple[int, int]

    def _init_messages(self) -> None:
        self.message: tuple[str, float] = ("", 0.0)
        self._clicked: tuple | None = None  # (kind, name, message before) of the control clicked last
        self._pinned: tuple | None = None  # (message, wall time): shown ahead of the hover help

    def _pin(self) -> None:
        """Show the current message ahead of the hover help (see the class docstring)."""
        self._pinned = (self.message, time.monotonic())

    def _by_key(self, act):
        """Run a key's action ``act``; a message it gives is pinned. Returns what ``act`` returns."""
        before = self.message
        result = act()
        if self.message is not before:
            self._pin()
        return result

    def _clicked_on(self, w) -> None:
        """Before the action of a click on ``w``: a message it gives shows while the mouse stays on it."""
        self._clicked = (w.kind, w.name, self.message)

    def _motion(self, pos, w) -> None:
        """The mouse moved to ``pos`` (over the control ``w``, or None)."""
        moved = tuple(pos) != tuple(self.mouse)
        self.mouse = pos
        if self._clicked and (w is None or (w.kind, w.name) != self._clicked[:2]):
            self._clicked = None  # the mouse left the control: its help shows again
        if self._pinned and moved and time.monotonic() - self._pinned[1] >= PIN_HOLD:
            self._pinned = None

    def _live_message(self, w) -> str | None:
        """The message, when it shows ahead of the hover help of ``w`` (the control under the mouse,
        or None) and of anything else under the mouse; else None."""
        text, until = self.message
        if not text or time.monotonic() >= until:
            return None
        if self._pinned and self._pinned[0] is self.message:
            return text
        clicked = self._clicked
        if w is not None and clicked and (w.kind, w.name) == clicked[:2] and self.message is not clicked[2]:
            return text  # what the control just did, until the mouse leaves it
        return None


# ------------------------------------------------------------------ widgets
def row_rects(x: int, y: int, w: int, h: int, n: int, gap: int) -> list[pygame.Rect]:
    """``n`` buttons sharing out the width ``w`` in one row, ``gap`` px apart."""
    bw = (w - (n - 1) * gap) / n
    rects = []
    for col in range(n):
        left, right = round(x + col * (bw + gap)), round(x + col * (bw + gap) + bw)
        rects.append(pygame.Rect(left, y, right - left, h))
    return rects


def draw_tab(screen: pygame.Surface, fonts, rect: pygame.Rect, label: str, active: bool, hot: bool) -> None:
    """A tab: its label, underlined in the accent colour when it is the one shown."""
    colour = UI["text"] if active else UI["text2"] if hot else UI["muted"]
    fonts.blit(screen, label, rect.center, "small", colour, "center")
    pygame.draw.line(screen, UI["accent"] if active else UI["grid"], (rect.x, rect.bottom - 1),
                     (rect.right - 1, rect.bottom - 1), 2 if active else 1)
