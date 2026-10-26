"""Shared set-up of the tests: the repository root importable however pytest is launched, pygame
headless (SDL's dummy video driver, no sound device) before any test module imports it, matplotlib
off-screen, factories of headless main viewers and a switch that skips the experiments' figures.
The other shared helpers are in tests/helpers.py."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Before pygame initialises its display: render off-screen, no sound device.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

@pytest.fixture
def new_app(tmp_path):
    """Factory for headless main viewers: ``new_app(config=None, size=(1280, 720), **options)`` (seed 7
    unless ``config`` is given; App's options such as autosave), with the results folder in
    tmp_path/results; pygame is shut down afterwards."""
    import pygame

    from roundabout_sim import app as viewer
    from roundabout_sim.config import SimConfig

    def make(config=None, size=(1280, 720), **options):
        return viewer.App(config or SimConfig(seed=7), size=size, results_dir=tmp_path / "results", **options)

    yield make
    pygame.quit()


@pytest.fixture
def make_app(new_app):
    """Factory for headless main viewers from settings: ``make_app(size=(1280, 720), **config)`` (seed 7
    unless given)."""
    from roundabout_sim.config import SimConfig

    return lambda size=(1280, 720), **config: new_app(SimConfig(**{"seed": 7, **config}), size)


@pytest.fixture
def no_figures(monkeypatch):
    """Skip drawing the experiments' figures (the sweep, heatmap and sample figures and the run dashboard)
    where a test checks only the CSV files and the printed text: they take most of a short run's time. Each
    call is recorded (its name, args and kwargs) and saving one writes just the PNG signature."""
    from roundabout_sim import plots
    from tests.helpers import PNG_MAGIC

    calls = []

    class Fake:
        def __init__(self, name, args, kwargs):
            self.name, self.args, self.kwargs = name, args, kwargs

        def savefig(self, path):
            Path(path).write_bytes(PNG_MAGIC)

    for name in ("sweep_figure", "heatmap_figure", "sample_figure", "dashboard"):
        def fake(*args, _name=name, **kwargs):
            calls.append(Fake(_name, args, kwargs))
            return calls[-1]
        monkeypatch.setattr(plots, name, fake)
    return calls
