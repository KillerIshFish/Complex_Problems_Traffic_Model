"""Helpers shared by the test modules: pygame events as the viewers' event loop delivers them, a click,
the widget of a kind and name, a tab shown, frames of the main viewer's loop body and the texts it
renders (the main viewer, roundabout_sim.app, and the sweep viewer); the invariants of a simulation
with vehicles of mixed lengths; the strata of a Latin hypercube; a real PNG figure. (tests/conftest.py
makes pygame headless before any test module imports it.)"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygame

from roundabout_sim.engine import TWO_PI, RingSim

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def move(pos) -> pygame.event.Event:
    return pygame.event.Event(pygame.MOUSEMOTION, pos=tuple(int(v) for v in pos), rel=(0, 0), buttons=(0, 0, 0))


def down(pos) -> pygame.event.Event:
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=tuple(int(v) for v in pos), button=1)


def up(pos) -> pygame.event.Event:
    return pygame.event.Event(pygame.MOUSEBUTTONUP, pos=tuple(int(v) for v in pos), button=1)


def key(k: int, mod: int = 0) -> pygame.event.Event:
    return pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod, unicode="", scancode=0)


def wheel(y: int) -> pygame.event.Event:
    return pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=y, flipped=False, precise_x=0.0, precise_y=float(y))


def resize(size) -> pygame.event.Event:
    return pygame.event.Event(pygame.VIDEORESIZE, size=tuple(size), w=size[0], h=size[1])


def click(viewer, pos) -> None:
    """Move onto ``pos``, press and let go there (either viewer)."""
    for event in (move(pos), down(pos), up(pos)):
        assert viewer.handle(event)


def widget(viewer, kind: str, name: str):
    """The widget of this kind and name (either viewer)."""
    return next(w for w in viewer.widgets if w.kind == kind and w.name == name)


def frame(app, *events: pygame.event.Event) -> None:
    """One pass of the main viewer's loop body (App.run): events, layout, step (unless paused), draw, flip."""
    for event in events:
        assert app.handle(event), f"{event} asked to quit"
    if app.screen.get_size() != app._layout_size:
        app.layout()
    if not app.paused:
        app.advance(app.steps_per_frame)
    app.draw()
    pygame.display.flip()


def frames(app, n: int) -> None:
    for _ in range(n):
        frame(app)


def show_tab(app, tab: str) -> None:
    """Click the main viewer's tab ``tab`` (a parameter group or Presets)."""
    click(app, widget(app, "tab", tab).rect.center)
    assert app.group == tab


def rendered_texts(app, monkeypatch) -> list[str]:
    """Draw the main viewer once and return every text rendered."""
    rendered = []
    render = app.fonts.render
    monkeypatch.setattr(app.fonts, "render", lambda text, *a, **k: rendered.append(text) or render(text, *a, **k))
    app.draw()
    monkeypatch.setattr(app.fonts, "render", render)
    return rendered


# ------------------------------------------------------------------ simulations
def bumper_gaps(sim: RingSim) -> np.ndarray:
    """Every vehicle's bumper gap to the one ahead in its lane, worked out from the centres and
    lengths (independently of the engine's own lane index)."""
    gaps = []
    for lane in range(sim.n_lanes):
        slots = np.flatnonzero(sim.alive & (sim.lane == lane))
        if slots.size:
            order = slots[np.argsort(sim.theta[slots])]
            th, length = sim.theta[order], sim.length[order]
            centres = np.diff(np.append(th, th[0] + TWO_PI)) * sim.radii[lane]
            gaps.append(centres - 0.5 * (length + np.roll(length, -1)))
    return np.concatenate(gaps) if gaps else np.zeros(0)


def assert_valid(sim: RingSim) -> None:
    """No overlaps, finite speeds >= 0, and the vehicle bookkeeping consistent (lengths by type, a type
    for every queued vehicle, queues >= 0)."""
    gaps = bumper_gaps(sim)
    assert gaps.size == 0 or gaps.min() >= -1e-9, f"vehicles overlap by {-gaps.min():.3f} m"
    live = sim.alive
    assert np.isfinite(sim.v[live]).all() and (sim.v[live] >= 0).all()
    np.testing.assert_array_equal(sim.length[live], np.where(sim.truck[live], sim.cfg.truck_length,
                                                             sim.cfg.car_length))
    assert (sim.queue >= 0).all()
    assert [len(types) for types in sim.queue_trucks] == sim.queue.tolist()  # every queued vehicle has a type
    assert int(sim.truck[live].sum()) <= int(live.sum())


# ------------------------------------------------------------------ designs and figures
def strata(values, lo: float, hi: float, n: int) -> list[int]:
    """The strata (of ``n`` equal ones of lo-hi) of ``values``, sorted; a value on a boundary in the one
    above, despite float rounding."""
    return sorted(np.clip(np.floor((np.asarray(values, float) - lo) / (hi - lo) * n + 1e-9), 0, n - 1)
                  .astype(int).tolist())


def assert_png(path: Path, min_bytes: int = 20_000) -> None:
    """A real figure: a PNG of a sensible size with something drawn on it."""
    data = Path(path).read_bytes()
    assert data[:8] == PNG_MAGIC, path
    width, height = int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    assert width >= 600 and height >= 400, path
    assert len(data) >= min_bytes, path
