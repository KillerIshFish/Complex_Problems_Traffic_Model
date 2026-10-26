"""Headless smoke tests of the pygame viewer (SDL's dummy video driver, no window).

Each test builds the viewer at 1280x720 with a fixed seed, runs frames of its
loop body (events, step, draw, flip) and injects mouse/keyboard events the way
pygame delivers them, then checks the effect on the simulation and the charts.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pygame
import pytest

from roundabout_sim import app as viewer
from roundabout_sim import palette
from roundabout_sim import experiments
from roundabout_sim.config import SimConfig, coerce
from tests.helpers import click, down, frame, frames, key, move, resize, show_tab, up, wheel, widget

SIZE = (1280, 720)
SEED = 7
BIG_RING = dict(n_lanes=5, radius=300, car_length=2, min_gap=1, density=200)  # ~1900 cars


# ------------------------------------------------------------------ helpers
# (make_app, from conftest: viewers at SIZE with seed SEED, screenshots into tmp_path)
def slider_pos(w: viewer.Widget, value: float) -> tuple[int, int]:
    """Screen point on a slider's track for ``value``."""
    spec = w.spec
    frac = (value - spec.lo) / (spec.hi - spec.lo)
    return round(w.track.x + frac * w.track.w), w.track.centery


def option_pos(w: viewer.Widget, choice: str) -> tuple[int, int]:
    return w.options[w.spec.choices.index(choice)].center


def car_screen_pos(app: viewer.App) -> tuple[int, tuple[int, int]]:
    """The live car farthest from any other (so a click cannot hit a neighbour), and its pixel."""
    snap = app.sim.cars()
    x, y = snap["x"], snap["y"]
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(d, np.inf)
    i = int(np.argmax(d.min(axis=1)))
    ring = app.ring
    return int(snap["slot"][i]), (round(ring.cx + x[i] * ring.k), round(ring.cy - y[i] * ring.k))


def assert_layout_fits(app: viewer.App) -> None:
    """Regions inside the window and apart; charts inside their column; the road inside the view."""
    screen = app.screen.get_rect()
    regions = (app.view_rect, app.stats_rect, app.charts_rect, app.panel_rect)
    for r in regions:
        assert screen.contains(r), r
    for i, a in enumerate(regions):
        for b in regions[i + 1:]:
            assert not a.colliderect(b), (a, b)
    for _, card, plot, *_ in app.chart_cards:
        assert app.charts_rect.contains(card) and card.contains(plot)
    assert app.charts_rect.contains(app.st_card) and app.st_card.contains(app.st_plot)
    for w in app.panel_widgets:
        if not w.scrolls:
            assert app.panel_rect.contains(w.rect), w.name
    ring = app.ring
    assert ring is not None and ring.rect == app.view_rect
    assert ring.k * (ring.r_edge + ring.stub) <= app.view_rect.w / 2


# --------------------------------------------------------------------- tests
def test_viewer_starts_runs_and_draws(make_app):
    app = make_app()
    assert app.screen.get_size() == SIZE
    assert app.sim.seed == SEED and app.sim.generation == 0
    frames(app, 40)
    sim = app.sim
    assert sim.time == pytest.approx(40 * sim.cfg.dt)
    assert int(sim.alive.sum()) + int(sim.queue.sum()) == sim.target_cars()
    # One chart sample every SAMPLE_INTERVAL of simulated time.
    t = app.hist.array("time")
    assert len(app.hist) >= 8 and np.all(np.diff(t) > 0)
    assert np.diff(t)[1:] == pytest.approx(viewer.SAMPLE_INTERVAL)
    assert {"stops_per_car_min", "mean_free_speed", "jams_following"} <= set(app._latest_row())
    assert app.stat_hits  # the stats card was drawn, every readout with its help text
    assert_layout_fits(app)


def test_same_seed_gives_the_same_run(make_app):
    a = make_app()
    frames(a, 20)
    theta = np.sort(a.sim.theta[a.sim.alive])
    b = make_app()
    frames(b, 20)
    np.testing.assert_allclose(np.sort(b.sim.theta[b.sim.alive]), theta)


def test_run_loop_handles_posted_events_and_quits(make_app, monkeypatch):
    app = make_app()
    drawn = 0
    draw = app.draw

    def counting_draw():
        nonlocal drawn
        draw()
        drawn += 1
        if drawn == 5:
            pygame.event.post(key(pygame.K_PLUS))  # handled at the start of frame 6
        if drawn == 30:
            pygame.event.post(pygame.event.Event(pygame.QUIT))

    monkeypatch.setattr(app, "draw", counting_draw)
    pygame.event.clear()
    app.run()
    assert drawn == 30
    assert app.steps_per_frame == 2
    assert app.sim.time == pytest.approx((5 + 25 * 2) * app.sim.cfg.dt)  # quit before stepping frame 31
    assert len(app.work_ms) == 30
    assert not pygame.get_init()  # run() shuts pygame down


def test_dragging_a_live_slider_changes_the_running_model(make_app):
    app = make_app()
    frames(app, 20)
    sim = app.sim
    gen, t0, target0 = sim.generation, sim.time, sim.target_cars()
    show_tab(app, "Traffic")
    w = widget(app, "slider", "density")
    assert not w.spec.reset
    tol = 2 * (w.spec.hi - w.spec.lo) / w.track.w  # two pixels of track
    frame(app, move(slider_pos(w, 50)), down(slider_pos(w, 50)), move(slider_pos(w, 100)))
    assert app.drag is not None
    assert sim.cfg.density == pytest.approx(100, abs=tol)  # applied while dragging
    frame(app, move(slider_pos(w, 120)), up(slider_pos(w, 120)))
    assert app.drag is None
    assert sim.cfg.density == pytest.approx(120, abs=tol)
    assert sim.generation == gen and sim.time > t0  # no rebuild: the run carries on
    assert sim.target_cars() > 2 * target0
    assert int(sim.alive.sum()) + int(sim.queue.sum()) == sim.target_cars()  # synced every step
    when, label = app.markers[-1]
    assert len(app.markers) == 1 and when == pytest.approx(sim.time - sim.cfg.dt)
    assert label == f"Density 50→{viewer.format_value(w.spec, sim.cfg.density)}"
    frames(app, 10)  # charts draw the marker
    assert app.hist.array("time")[0] < t0  # history kept


def test_dragging_the_exit_chance_to_zero_reroutes_every_car(make_app):
    app = make_app()
    frames(app, 10)
    show_tab(app, "Traffic")
    w = widget(app, "slider", "exit_prob")
    frame(app, move(slider_pos(w, 0.1)), down(slider_pos(w, 0.1)), move((w.track.x - 20, w.track.centery)),
          up((w.track.x - 20, w.track.centery)))
    assert app.sim.cfg.exit_prob == 0.0
    assert np.all(np.isinf(app.sim.exit_at[app.sim.alive]))


def test_steps_per_frame_slider(make_app):
    app = make_app()
    w = widget(app, "slider", "steps_per_frame")
    frame(app, move(slider_pos(w, 1)), down(slider_pos(w, 1)), move((w.track.right + 30, w.track.centery)))
    assert app.steps_per_frame == int(w.spec.hi) == 20
    frame(app, up((w.track.right + 30, w.track.centery)))
    t = app.sim.time
    frame(app)
    assert app.sim.time == pytest.approx(t + 20 * app.sim.cfg.dt)
    assert not app.markers  # viewer settings are not model changes


def test_rebuilding_slider_applies_only_on_release(make_app):
    app = make_app()
    frames(app, 20)
    sim = app.sim
    gen = sim.generation
    assert app.group == "Road"
    w = widget(app, "slider", "n_lanes")
    assert w.spec.reset
    end = (w.track.right, w.track.centery)
    frame(app, move(slider_pos(w, 3)), down(slider_pos(w, 3)), move(end))
    assert app.value("n_lanes") == 5  # the slider shows the new value...
    assert sim.cfg.n_lanes == 3 and sim.n_lanes == 3 and sim.generation == gen  # ...the model waits
    assert sim.time == pytest.approx(21 * sim.cfg.dt)  # and keeps running meanwhile
    frame(app, up(end))
    assert sim.cfg.n_lanes == 5 and sim.n_lanes == 5
    assert sim.generation == gen + 1
    # The charts restart with the rebuilt roundabout.
    assert sim.time == pytest.approx(sim.cfg.dt)
    assert list(app.hist.array("time")) == [0.0]
    assert not app.markers and not app.brakes
    assert "lane4_cars" in app._latest_row()
    assert app.ring.r_edge == pytest.approx(sim.r_out + sim.cfg.lane_width / 2)
    frames(app, 10)
    assert_layout_fits(app)


def test_releasing_a_rebuilding_slider_at_its_value_does_not_rebuild(make_app):
    app = make_app()
    frames(app, 5)
    w = widget(app, "slider", "n_arms")
    gen = app.sim.generation
    start = slider_pos(w, app.sim.cfg.n_arms)
    frame(app, move(start), down(start), move((start[0] + 60, start[1])), move(start), up(start))
    assert app.sim.generation == gen and app.sim.cfg.n_arms == 4
    assert app.sim.time == pytest.approx(6 * app.sim.cfg.dt)


def test_choice_buttons(make_app):
    app = make_app()
    frames(app, 10)
    sim = app.sim
    gen = sim.generation

    colours = widget(app, "choice", "colour_mode")
    click(app, option_pos(colours, "lane"))
    assert app.colour_mode == "lane"
    frame(app)

    spacetime = widget(app, "choice", "spacetime_mode")
    assert spacetime in app.chart_widgets
    click(app, option_pos(spacetime, "occupancy"))
    assert app.spacetime_mode == "occupancy"
    frame(app)
    assert not app.markers  # view settings leave no marker

    # A pressed option only counts if the button is released on it.
    app.handle(down(option_pos(colours, "desired")))
    app.handle(up(option_pos(colours, "speed")))
    assert app.colour_mode == "lane"

    # A live model choice: applied at once, marked on the charts, no rebuild.
    show_tab(app, "Lane chg")
    motive = widget(app, "choice", "lane_change_motive")
    click(app, option_pos(motive, "random"))
    assert sim.cfg.lane_change_motive == "random" and sim.generation == gen
    assert app.markers[-1][1] == "Lane-change motive advantage→random"
    frame(app)

    # A rebuilding choice: restarts the roundabout and the charts.
    show_tab(app, "Start")
    spacing = widget(app, "choice", "init_spacing")
    click(app, option_pos(spacing, "clustered"))
    assert sim.cfg.init_spacing == "clustered" and sim.generation == gen + 1
    assert sim.cfg.lane_change_motive == "random"  # earlier live change survives the rebuild
    assert not app.markers and sim.time == 0.0
    frames(app, 5)
    assert app.colour_mode == "lane" and app.spacetime_mode == "occupancy"


def test_keyboard_shortcuts(make_app):
    app = make_app()
    dt = app.sim.cfg.dt
    frame(app)

    frame(app, key(pygame.K_SPACE))  # pause: the model stops, the window keeps drawing
    assert app.paused
    t = app.sim.time
    frames(app, 5)
    assert app.sim.time == t
    frame(app, key(pygame.K_SPACE))
    assert not app.paused and app.sim.time == pytest.approx(t + dt)

    for k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):  # + (also = and keypad +)
        app.handle(key(k))
    assert app.steps_per_frame == 4
    t = app.sim.time
    frame(app)
    assert app.sim.time == pytest.approx(t + 4 * dt)
    for _ in range(30):
        app.handle(key(pygame.K_PLUS))
    assert app.steps_per_frame == 20  # clamped to the slider's range
    for k in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
        app.handle(key(k))
    assert app.steps_per_frame == 17
    for _ in range(30):
        app.handle(key(pygame.K_MINUS))
    assert app.steps_per_frame == 1

    seen = []
    for _ in range(4):  # C cycles the car colours
        frame(app, key(pygame.K_c))
        seen.append(app.colour_mode)
    assert seen == ["desired", "lane", "type", "speed"]

    frame(app, key(pygame.K_b))  # B brakes a random car for 3 s
    assert len(app.brakes) == 1
    assert int(app.sim.cars()["braking"].sum()) == 1
    frames(app, 3)

    seed, gen = app.sim.seed, app.sim.generation
    frame(app, key(pygame.K_r))  # R restarts with a new seed
    assert app.sim.generation == gen + 1 and app.sim.seed != seed
    assert app.sim.time == pytest.approx(dt) and len(app.hist) == 1 and not app.brakes

    assert app.handle(key(pygame.K_ESCAPE)) is False
    assert app.handle(pygame.event.Event(pygame.QUIT)) is False


def test_screenshots_go_to_the_results_folder(make_app, tmp_path):
    app = make_app()
    frames(app, 3)
    app.handle(key(pygame.K_s))
    click(app, widget(app, "button", "screenshot").rect.center)  # same second: gets a new name
    shots = sorted((tmp_path / "results").glob("roundabout_*.png"))
    assert len(shots) == 2
    assert pygame.image.load(str(shots[0])).get_size() == SIZE
    text = app.message[0]
    assert text.startswith("Saved ") and Path(text.removeprefix("Saved ")) in shots


def test_panel_buttons(make_app):
    app = make_app()
    frames(app, 10)
    sim = app.sim
    dt = sim.cfg.dt

    def press(name):
        click(app, widget(app, "button", name).rect.center)

    press("pause")
    assert app.paused
    t = sim.time
    press("step")  # one frame's worth, also while paused
    assert sim.time == pytest.approx(t + dt) and app.paused
    frames(app, 3)
    assert sim.time == pytest.approx(t + dt)
    press("pause")
    assert not app.paused

    # A button only fires when released on it.
    app.handle(down(widget(app, "button", "pause").rect.center))
    app.handle(up(app.view_rect.center))
    assert not app.paused

    press("brake")
    assert len(app.brakes) == 1 and app.brakes[-1][0] == sim.time
    frames(app, 2)

    gen, t = sim.generation, sim.time
    press("clear")  # forget the charts, keep the run
    assert sim.generation == gen and sim.time == t
    assert len(app.hist) == 1 and not app.brakes and not app.markers
    frames(app, 6)
    assert len(app.hist) == 2

    press("reset")
    assert sim.generation == gen + 1 and sim.time == 0.0
    frames(app, 3)


def test_clicking_a_car_makes_it_brake(make_app):
    app = make_app()
    frames(app, 10)
    sim = app.sim
    slot, pos = car_screen_pos(app)
    assert app.view_rect.collidepoint(pos) and app.widget_at(pos) is None
    app.handle(move(pos))
    app.draw()  # hovered car: outline and tooltip
    click(app, pos)
    assert sim.brake_until[slot] == pytest.approx(sim.time + 3.0)
    assert app.brakes[-1] == (sim.time, pytest.approx(float(sim.theta[slot])))
    frames(app, 5)  # braked car ringed in the view and the space-time diagram
    snap = sim.cars()
    assert sim.alive[slot] and snap["braking"][snap["slot"] == slot].all()

    n = len(app.brakes)
    click(app, (app.ring.cx, app.ring.cy))  # the central island: no car there
    click(app, app.view_rect.topleft)
    assert len(app.brakes) == n
    assert int(sim.cars()["braking"].sum()) == 1


def test_viewer_follows_a_rebuild_made_directly_on_the_model(make_app):
    app = make_app()
    frames(app, 20)
    old_ring = app.ring
    assert app.sim.set(n_arms=2) is True
    frame(app)
    assert list(app.hist.array("time")) == [0.0]
    assert app.ring is not old_ring and app.sim.n_arms == 2
    frames(app, 5)


@pytest.mark.parametrize("size", [(1100, 700), (1600, 900), (1920, 1080), (2400, 900)])
def test_window_resize_relays_out_everything(make_app, size):
    app = make_app()
    frames(app, 5)
    frame(app, resize(size))
    assert app.screen.get_size() == size
    assert app.panel_rect.right == size[0] and app.panel_rect.h == size[1]
    assert_layout_fits(app)
    columns = len({card.x for _, card, *_ in app.chart_cards})
    assert columns == (2 if app.charts_rect.w >= 900 * app.s else 1)
    if size == (2400, 900):
        assert columns == 2  # a very wide window puts the charts side by side
    frames(app, 3)
    frame(app, resize(SIZE))
    assert app.screen.get_size() == SIZE
    assert_layout_fits(app)


def test_wheel_scrolls_the_parameter_list(make_app):
    app = make_app()
    show_tab(app, "Drivers")
    overflow = app.content_h - app.content_rect.h
    assert overflow > 0
    first = widget(app, "slider", "mean_speed")
    y0 = first.rect.y
    frame(app, move(app.content_rect.center), wheel(-1))
    assert app.scroll > 0
    assert widget(app, "slider", "mean_speed").rect.y == y0 - app.scroll
    for _ in range(30):
        app.handle(wheel(-3))
    assert app.scroll == overflow  # stops at the end of the list
    frame(app)
    # The last slider is reachable now and still works.
    last = widget(app, "slider", "lateral_accel")
    assert app.content_rect.contains(last.rect)
    target = slider_pos(last, 8.0)
    frame(app, move(target), down(target), up(target))
    assert app.sim.cfg.lateral_accel == pytest.approx(8.0, abs=0.5)
    for _ in range(30):
        app.handle(wheel(3))
    assert app.scroll == 0


def test_hover_help_for_charts_view_controls_and_readouts(make_app):
    app = make_app()
    frames(app, 30)
    for spec, card, plot, *_ in app.chart_cards:
        frame(app, move(plot.center))  # crosshair and hovered values
        assert app._status_text() == spec.help
    for mode in ("speed", "occupancy"):
        app.spacetime_mode = mode
        frame(app, move(app.st_plot.center))
        assert app._status_text() == viewer.SPACETIME_HELP
    frame(app, move(app.ring.rect.move(3, 3).topleft))
    assert app._status_text() == viewer.VIEW_HELP

    show_tab(app, "Traffic")
    density = widget(app, "slider", "density")
    frame(app, move(density.rect.center))
    assert app._status_text().startswith("Density: ")
    show_tab(app, "Road")
    lanes = widget(app, "slider", "n_lanes")
    frame(app, move(lanes.rect.center))
    assert "rebuilds the roundabout" in app._status_text()

    rect, help_text = app.stat_hits[0]
    frame(app, move(rect.center))
    assert app._status_text() == help_text
    labels = {text.split(":")[0] for _, text in app.stat_hits}
    assert {"Stop-and-go", "Clustering", "Mean free speed", "Following share of jams",
            "Harsh braking (episodes/min)", "Spacing (Gini)"} <= labels
    assert not any(word in text for _, text in app.stat_hits for word in ("Natural", "Emergency", "near-crash"))


def test_key_messages_show_wherever_the_mouse_rests(make_app, monkeypatch):
    """F5, X, Shift+X, F9 and S confirm themselves in the help box even with the mouse resting on
    the view, a chart, the stats or a control (whose hover help would otherwise come first)."""
    from roundabout_sim import ui

    app = make_app()
    frames(app, 5)
    spots = {"view": app.ring.rect.move(3, 3).topleft, "chart": app.chart_cards[0][2].center,
             "stats": app.stat_hits[0][0].center, "slider": widget(app, "slider", "steps_per_frame").rect.center}
    for name, pos in spots.items():
        for k, mod in ((pygame.K_F5, 0), (pygame.K_x, 0), (pygame.K_x, pygame.KMOD_SHIFT), (pygame.K_F9, 0),
                       (pygame.K_s, 0)):
            show_tab(app, "Traffic")
            frame(app, move(pos))
            frame(app, pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod, unicode="", scancode=0))
            assert app._status_text() == app.message[0] != "", (name, k)
            frame(app, move((pos[0] + 1, pos[1])))  # a move within PIN_HOLD keeps it
            assert app._status_text() == app.message[0], (name, k)
    monkeypatch.setattr(ui, "PIN_HOLD", 0.0)
    frame(app, move(spots["view"]))  # held long enough: once the mouse moves, its help is back
    assert app._status_text() == viewer.VIEW_HELP


def test_chart_ticks_keep_the_decimals_of_a_2_5_step(make_app, monkeypatch):
    """A tick step of 2.5 labels 2.5 and 7.5 as such (they printed as 2 and 8, rounded half to even)."""
    app = make_app()
    frames(app, 5)
    monkeypatch.setattr(viewer, "nice_step", lambda span, target: 2.5)
    texts = []
    blit = app.fonts.blit
    monkeypatch.setattr(app.fonts, "blit", lambda scr, text, *a, **k: texts.append(text) or blit(scr, text, *a, **k))
    app.draw()
    assert {"2.5", "7.5"} <= set(texts)
    assert viewer.tick_text(7.5, 2.5) == "7.5" and viewer.tick_text(10.0, 5.0) == "10"


def test_resume_note_shows_with_the_mouse_on_the_view(make_app, tmp_path):
    first = viewer.App(SimConfig(seed=SEED), size=SIZE, results_dir=tmp_path / "res", autosave=True)
    frames(first, 3)
    assert first.save_session() is not None
    app = viewer.App(SimConfig(seed=SEED), size=SIZE, results_dir=tmp_path / "res", resume="session")
    frame(app)
    frame(app, move(app.view_rect.center))
    assert app._status_text().startswith("Resumed the saved session")


@pytest.mark.parametrize("config", [
    dict(n_arms=0, density=70),  # closed ring: phantom jams without entries/exits
    dict(n_lanes=1, n_arms=8, radius=20),  # tiny ring, many arms
    dict(n_lanes=1, radius=20, density=1),  # empty ring: no cars, NaN free speed
], ids=["closed-ring", "tiny-ring", "empty-ring"])
def test_unusual_roundabouts_draw(make_app, config):
    app = make_app(**config)
    for mode in ("speed", "desired", "lane"):
        app.colour_mode = mode
        frames(app, 4)
    app.spacetime_mode = "occupancy"
    frames(app, 4)
    assert_layout_fits(app)
    if config.get("density") == 1:
        assert app.sim.target_cars() == 0 and np.isnan(app._latest_row()["mean_free_speed"])
    else:
        slot, pos = car_screen_pos(app)
        click(app, pos)
        assert app.sim.brake_until[slot] > app.sim.time


def test_big_ring_of_about_2000_small_cars(make_app):
    app = make_app(**BIG_RING)
    sim = app.sim
    assert sim.target_cars() > 1800
    assert int(sim.alive.sum()) == sim.target_cars()
    for mode in ("speed", "desired", "lane"):
        app.colour_mode = mode
        frames(app, 3)
    app.spacetime_mode = "occupancy"
    frames(app, 3)
    assert int(sim.alive.sum()) + int(sim.queue.sum()) == sim.target_cars()
    assert len(app.hist) >= 2 and "lane4_speed" in app._latest_row()
    assert_layout_fits(app)
    slot, pos = car_screen_pos(app)  # even 2 m cars on a 300 m ring can be picked
    click(app, pos)
    assert sim.brake_until[slot] > sim.time
    frames(app, 2)


def test_command_line_options():
    args = viewer.parse_args(["--seed", "5", "--size", "1280x720", "--n-lanes", "2", "--density", "70",
                              "--init_spacing", "clustered", "--reaction-distance", "40"])
    assert args.size == (1280, 720)
    cfg = viewer.config_from_args(args)
    assert cfg == SimConfig(seed=5, n_lanes=2, density=70.0, init_spacing="clustered", reaction_distance=40.0)
    assert isinstance(cfg.n_lanes, int)
    assert viewer.config_from_args(viewer.parse_args([])) == SimConfig()
    for bad in (["--size", "big"], ["--init_spacing", "diagonal"], ["--density", "lots"]):
        with pytest.raises(SystemExit) as exc:
            viewer.parse_args(bad)
        assert exc.value.code == 2


def test_charts_read_in_order_with_the_new_measures(make_app):
    titles = [spec.title for spec in viewer.CHARTS]
    for title, start in zip(titles, ("Mean speed", "Congestion over time", "Stop-and-go (instability)", "Flux",
                                     "Congregation"), strict=True):
        assert title.startswith(start)
    congestion, stops, congregation = viewer.CHARTS[1], viewer.CHARTS[2], viewer.CHARTS[4]
    assert [s.key for s in congestion.series] == ["jam_fraction", "speed_deficit"]  # headline first
    assert stops.extra == "jams" and "standstill" in stops.help and "Congestion" in stops.help
    assert [(s.key, s.reference) for s in congregation.series] == [
        ("spacing_gini", False), ("order_param", False), ("order_baseline", True)]
    assert not any(s.key == "clustering" for spec in viewer.CHARTS for s in spec.series)  # a stats readout only
    # The data keep the palette order; the reference is a faint tint of the order colour, not a palette slot.
    colours = viewer.App._series_colours(congregation)
    assert colours[:2] == list(palette.SERIES[:2]) and colours[2] not in palette.SERIES

    app = make_app()
    frames(app, 40)
    row = app._latest_row()
    stats = {st.label: st for col in app._stat_columns(row) for st in col}
    assert stats["Order"].value == f"{row['order_param']:.2f} · random {row['order_baseline']:.2f}"
    assert stats["Spacing (Gini)"].value == f"{row['spacing_gini']:.2f}"
    assert "upper bound" in stats["Following share of jams"].help_text()
    assert "Exit chance = 0" in stats["Following share of jams"].help_text()


def test_jam_and_bunching_readouts():
    v = dict(n_jams=6, jams_following=4, jams_merge=0, jams_exit=2, jams_entry=0, jams_other=0)
    texts = viewer.App._extra_texts("jams", v)
    assert texts[0] == "jams 6 — following 4 · merge 0 · exit 2 · entry 0 · other 0"
    assert texts[-1] == "jams 6" and not any("natural" in t for t in texts)
    assert viewer.App._extra_texts("jams", {**v, "n_jams": 0}) == ["no jams"]
    # Order counts as one-sided only well above what randomly placed cars give.
    level = viewer.App._extra_texts("bunching", dict(order_param=0.15, order_angle=1.0, order_baseline=0.09))
    assert "random" in level[0]
    bunched = viewer.App._extra_texts("bunching", dict(order_param=0.4, order_angle=math.pi, order_baseline=0.1))
    assert bunched == ["bunched at 180° · 4.0× random", "bunched at 180°"]
    assert viewer.one_sided(0.4, 0.1) and not viewer.one_sided(0.15, 0.09) and not viewer.one_sided(0.04, np.nan)


def test_header_readouts_have_their_own_help(make_app):
    app = make_app(n_arms=0, n_lanes=1, density=80, reaction_distance=20)
    frames(app, 20)
    frame(app, key(pygame.K_SPACE))  # paused: the readouts stay put
    assert {text for _, text in app.chart_hits} == {viewer.JAMS_HELP, viewer.ORDER_HELP}
    for rect, text in list(app.chart_hits):
        frame(app, move((rect.right - 6, rect.centery)))
        assert app._status_text() == text
    assert "upper bound" in viewer.JAMS_HELP and "Exit chance = 0" in viewer.JAMS_HELP


def test_tabs_have_readable_tooltips(make_app):
    app = make_app()
    for group in viewer.GROUPS:
        w = widget(app, "tab", group)
        frame(app, move(w.rect.center))
        assert app._status_text() == w.help == viewer.tab_help(group)
        assert w.help.startswith("Show the ") and "chg" not in w.help
    assert widget(app, "tab", "Lane chg").help == "Show the lane-changing parameters."
    assert viewer.tab_help("Other stuff") == "Show the other stuff parameters."  # a future group


@pytest.mark.parametrize("size", [SIZE, (1600, 900), (2400, 1300)])
def test_every_help_text_fits_the_help_box(make_app, size):
    app = make_app(size=size)
    assert viewer.STATUS_LINES[0] <= app.status_lines <= viewer.STATUS_LINES[1]
    for text in viewer.help_texts():
        assert len(app.fonts.wrap(text, app.status_rect.w)) <= app.status_lines, text[:60]
    # A word longer than the box (a saved file's full path) is broken, not run off the edge.
    path = "C:\\" + "\\".join(["a_rather_long_folder_name"] * 12) + "\\roundabout_20260101_120000.csv"
    lines = app.fonts.wrap(f"Saved {path} (the last 10 min)", app.status_rect.w)
    assert "".join(lines).replace(" ", "") == f"Saved{path}(thelast10min)"
    assert all(app.fonts.width(line, "small") <= app.status_rect.w for line in lines)
    assert app.panel_rect.contains(app.status_rect) and app.content_rect.bottom < app.status_rect.y
    frames(app, 3)
    assert_layout_fits(app)


def test_marker_labels_keep_the_newest_and_hovering_names_any(make_app, monkeypatch):
    app = make_app()
    frames(app, 100)
    t_end, span = app._time_axis()
    names = ["Density 50→60", "Speed variance (std) 1.5→2.0", "Reaction distance 25→20"]
    for k, label in enumerate(names):  # oldest first, each a couple of pixels from the next
        app.markers.append((t_end - 5 + 0.3 * k, label))
    lh = app.fonts.line("small")
    one_row = pygame.Rect(100, 100, 400, 2 * lh)
    assert [label for _, label in app._marker_labels(one_row, t_end, span)] == [names[-1]]
    two_rows = pygame.Rect(100, 100, 400, 6 * lh)
    placed = app._marker_labels(two_rows, t_end, span)
    assert [label for _, label in placed] == names[:0:-1]  # the newest two, one per row
    assert placed[0][0].y < placed[1][0].y
    assert all(two_rows.contains(box) for box, _ in placed)

    # A label left out still shows in the hover readout of its line.
    _, card, plot, *_ = app.chart_cards[0]
    x = app._time_x(plot, app.markers[1][0], t_end, span)
    assert app._marker_near(plot, x, t_end, span) == names[1]
    assert app._marker_near(plot, x + 80, t_end, span) is None
    rendered = []
    render = app.fonts.render
    monkeypatch.setattr(app.fonts, "render", lambda text, *a, **k: rendered.append(text) or render(text, *a, **k))
    app.paused = True
    frame(app, move((round(x), plot.centery)))
    assert any(text.endswith(f"·  {names[1]}") and text.split()[1] == "s" for text in rendered)

    app.markers.clear()  # far apart: every label fits on the one row
    app.markers.extend((t_end - span + 15 * k, label) for k, label in enumerate(names))
    assert len(app._marker_labels(pygame.Rect(0, 100, 1000, 2 * lh), t_end, span)) == 3

    # A label sits right beside its own line and never covers a newer marker's line (it would read
    # as that line's label): here the middle one goes to the left of its line, the oldest is left out.
    app.markers.clear()
    app.markers.extend((t_end - span * f, label) for f, label in zip((0.7, 0.68, 0.5), names))
    plot = pygame.Rect(0, 100, 600, 2 * lh)
    lines = [round(app._time_x(plot, when, t_end, span)) for when, _ in app.markers]
    placed = {label: box for box, label in app._marker_labels(plot, t_end, span)}
    assert set(placed) == set(names[1:]) and placed[names[1]].right == lines[1] - 3
    for k, (_, label) in enumerate(app.markers):
        if label in placed:
            box = placed[label]
            assert box.x == lines[k] + 3 or box.right == lines[k] - 3
            assert not any(box.x <= x <= box.right for x in lines[k + 1:])


def test_save_csv_writes_the_chart_data_and_the_settings(make_app, tmp_path):
    app = make_app()
    frames(app, 30)
    show_tab(app, "Traffic")
    w = widget(app, "slider", "density")
    frame(app, move(slider_pos(w, 50)), down(slider_pos(w, 50)), move(slider_pos(w, 80)), up(slider_pos(w, 80)))
    frames(app, 10)
    sim = app.sim
    frame(app, key(pygame.K_d))
    click(app, widget(app, "button", "csv").rect.center)  # same second: a fresh pair of names
    results = tmp_path / "results"
    csvs = sorted(results.glob("roundabout_*.csv"))
    texts = sorted(results.glob("roundabout_*_settings.txt"))
    assert len(csvs) == 2 and len(texts) == 2
    assert {p.name.removesuffix(".csv") for p in csvs} == {p.name.removesuffix("_settings.txt") for p in texts}
    latest = Path(app.message[0].split()[1])  # "Saved <path> (...)"
    assert latest in csvs and "last 10 min" in app.message[0]

    data = pd.read_csv(latest)
    assert len(data) == len(app.rec.rows)
    np.testing.assert_allclose(data["time"], app.rec.frame()["time"])
    assert {"time", "jam_fraction", "speed_deficit", "stops_per_car_min", "n_jams",
            "jams_following", "emergency_per_min", "spacing_gini", "order_param", "order_baseline",
            "clustering", "occupancy", "flux", "lane0_speed"} <= set(data.columns)
    # Measurements named like a setting get the prefix the experiments CSVs use.
    assert {"measured_density", "measured_mean_speed", "measured_speed_std"} <= set(data.columns)
    assert not {"density", "mean_speed", "speed_std"} & set(data.columns)
    np.testing.assert_allclose(data["measured_mean_speed"], app.rec.frame()["mean_speed"])

    text = latest.with_name(f"{latest.stem}_settings.txt").read_text(encoding="utf-8")
    assert f"(seed {SEED})" in text and f"density = {sim.cfg.density:g}" in text
    assert "reaction_distance = 25" in text and "last 10 min" in text
    when, label = app.markers[-1]
    assert f"{when:8.1f} s  {label}" in text and label.startswith("Density 50→")
    command = next(line.strip() for line in text.splitlines()
                   if line.strip().startswith("python -m roundabout_sim.experiments run"))
    args = experiments.build_parser().parse_args(command.split()[3:])
    assert args.seed == SEED and args.duration == pytest.approx(sim.time)
    changes = {name: coerce(name, value) for name, value in args.set}
    assert SimConfig(seed=args.seed).updated(**changes) == sim.cfg  # reruns the final settings
    assert set(changes) == {"density"}  # only what differs from the defaults


def test_parameter_flags_are_read_as_strictly_as_the_experiments_cli(capsys):
    """--n_lanes 2.5 is an error, not 2 lanes (banker's rounding made 3.5 four); a seed below 0 is refused
    by --seed itself; a value off the slider range gets the experiments CLI's note."""
    for bad, why in ((["--n_lanes", "2.5"], "bad value '2.5' for n_lanes: must be a whole number"),
                     (["--n-arms", "1.5"], "bad value '1.5' for n_arms: must be a whole number"),
                     (["--max_queue", "2.7"], "bad value '2.7' for max_queue"),
                     (["--seed", "-2"], "argument --seed: seed must be a whole number >= 0, got '-2'")):
        with pytest.raises(SystemExit) as exc:
            viewer.parse_args(bad)
        assert exc.value.code == 2 and why in capsys.readouterr().err, bad
    cfg = viewer.config_from_args(viewer.parse_args(["--n_lanes", "9", "--n_arms", "3.0"]))
    assert (cfg.n_lanes, cfg.n_arms) == (9, 3)
    out = capsys.readouterr().out
    assert "note: n_lanes=9 is outside the usual range 1-5" in out and "n_arms" not in out


def test_small_readouts_read_naturally(make_app):
    """The congregation chart's axis stops at 1 (its measures all lie in 0-1), a lane with one car reads
    '1 car', and the short delay headers keep the decimal the stats card shows."""
    spec = next(c for c in viewer.CHARTS if c.title.startswith("Congregation"))
    assert spec.max_top == 1.0
    app = make_app(n_lanes=2, density=5.0, n_arms=0)
    row = {**app._latest_row(), "lane0_speed": 8.3, "lane0_cars": 1, "lane1_speed": np.nan, "lane1_cars": 0}
    lanes = [s for s in app._stat_columns(row)[2] if s.key == "Lane"]
    assert [s.value for s in lanes] == ["8.3 m/s · 1 car", "0 cars"]
    texts = viewer.App._extra_texts("delay", {"mean_entry_delay": 1.4, "longest_queue": 3, "entry_vc": 1.2})
    assert texts[-3:] == ["delay 1.4 s · LOS F", "1.4 s · LOS F", "LOS F"]
    assert not any("delay 1 s" in t for t in texts)


@pytest.mark.parametrize("size", [(1100, 700), (1920, 1080)])
def test_arms_of_a_crowded_small_ring_do_not_overlap_and_the_detector_label_is_clear(make_app, size):
    """Twelve arms on a 10 m ring: each arm is drawn at most as wide as its share of the ring's edge, and
    the detector label moves past the arms' ends when the arms beside it would cover it."""
    for config in (dict(n_arms=12, radius=10.0, n_lanes=1), dict(n_arms=12, radius=30.0, n_lanes=5), {}):
        app = make_app(size=size, **config)
        app.draw()
        ring, sim = app.ring, app.sim
        half = ring.arm_half_width(sim)
        assert half <= ring.r_edge * math.sin(math.pi / sim.n_arms) + 1e-9
        if not config:  # (four arms: as wide as their two lanes, as before)
            assert half == pytest.approx(ring.arm_offset + ring.lw / 2)
        assert not ring._on_arms(ring.detector_box, sim, half)
        assert ring.layer.get_rect().contains(ring.detector_box)


def test_a_window_size_below_the_minimum_is_raised_to_it(make_app, capsys):
    app = make_app(size=(400, 300))
    desk = (pygame.display.get_desktop_sizes() or [viewer.MIN_WINDOW])[0]
    assert app.screen.get_size() == (max(400, min(viewer.MIN_WINDOW[0], desk[0])),
                                     max(300, min(viewer.MIN_WINDOW[1], desk[1])))
    assert "window size 400x300 is below the smallest the layout supports" in capsys.readouterr().err
