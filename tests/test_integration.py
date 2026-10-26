"""How the features work together: presets and random settings in the viewer and on the command line,
the saved session of a preset, the viewer's Sweep viewer button and experiments' --random with
explicit settings.

The viewer runs headless (SDL's dummy video driver) at the minimum window, 1100x700, as in
tests/test_app.py; the sweep viewer it starts is never really launched (subprocess.Popen is replaced).
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pygame
import pytest

from roundabout_sim import app as viewer
from roundabout_sim import experiments, presets, state
from roundabout_sim import sweep_viewer as sv
from roundabout_sim.config import PARAM_BY_NAME, PARAMS, SimConfig
from tests.helpers import click, frame, key, move, show_tab, widget

SIZE = (1100, 700)
SEED = 7
NAMES = [f.name for f in fields(SimConfig) if f.name != "seed"]


# ------------------------------------------------------------------ helpers
@pytest.fixture
def make_app(new_app):
    """Factory for viewers at the minimum window (fixed seed, files into tmp_path/results)."""
    return lambda size=SIZE, autosave=False, **config: new_app(SimConfig(**{"seed": SEED, **config}), size,
                                                               autosave=autosave)


def settings(cfg: SimConfig) -> dict:
    return {n: getattr(cfg, n) for n in NAMES}


def assert_fits(app: viewer.App) -> None:
    """Every fixed control inside the panel; the list's controls side by side within its width."""
    for w in app.panel_widgets:
        assert app.panel_rect.contains(w.rect) or w.scrolls, w.name
        if w.scrolls:
            assert app.panel_rect.left < w.rect.left and w.rect.right < app.panel_rect.right, w.name
    tabs = [w for w in app.panel_widgets if w.kind == "tab"]
    assert {w.name for w in tabs} == set(viewer.TABS)
    for w in tabs:  # every tab's label fits its tab
        assert app.fonts.width(w.name, "small") <= w.rect.w, w.name


# ------------------------------------------------------------------ the preset selector
def test_preset_selector_applies_each_preset_and_shows_its_description(make_app):
    app = make_app()
    frame(app)
    show_tab(app, viewer.PRESETS_TAB)
    assert app.preset_choice == "default" and app.origin_text() == ""  # the defaults are no preset of note
    seed, names = app.sim.seed, presets.names()
    for i, name in enumerate(names):
        assert app.preset_choice == name
        p = presets.get(name)
        cycler = widget(app, "cycler", "preset")
        frame(app, move(cycler.rect.center))  # hovering the selector: its description in the help box
        assert p.description in app._status_text() and f"Preset {i + 1} of {len(names)}" in app._status_text()
        text = widget(app, "text", "preset_text")  # below: description, settings, notes and sources
        shown = " ".join(line for _, line, _ in text.lines)
        assert shown.startswith(p.description.split(" ")[0]) and "Sources:" in shown and "Notes:" in shown
        click(app, widget(app, "button", "apply_preset").rect.center)
        frame(app)
        assert app.sim.generation >= 1 and app.sim.seed == seed  # rebuilt, the seed kept
        assert settings(app.sim.cfg) == settings(presets.apply(name)) and presets.find(app.sim.cfg) == name
        assert app.origin_text() == f"preset {name} ({p.title})"
        assert list(app.markers) == [(0.0, f"Preset {p.title}")]  # the rebuild cleared the charts, then this
        assert "rebuilt with its settings" in app._status_text()  # the message answers the click
        assert_fits(app)
        click(app, widget(app, "cycler", "preset").options[1].center)  # ▶ the next one
    assert app.preset_choice == "default"  # round to the first again
    click(app, widget(app, "cycler", "preset").options[0].center)  # ◀
    assert app.preset_choice == names[-1]
    click(app, widget(app, "cycler", "preset").rect.center)  # the title steps forward, as in the sweep viewer
    assert app.preset_choice == "default"


def test_preset_help_names_its_main_source_and_keeps_the_help_box_height(make_app):
    for name in presets.names():
        p = presets.get(name)
        assert (f"Source: {p.cite}." in viewer.preset_help(name)) == (name != "default")
    for size, lines in ((SIZE, 8), ((1280, 720), 8), ((1920, 1080), 9)):  # as before the sources were added
        app = make_app(size=size)
        assert app.status_lines == lines
        for name in presets.names():
            assert len(app.fonts.wrap(viewer.preset_help(name), app.status_rect.w)) <= lines, (size, name)
        pygame.quit()


def test_preset_caption_and_running_line_show_in_full_at_every_size(make_app, monkeypatch):
    """The caption under the selector drops its less important parts rather than the state; the
    Running line grows to hold every changed setting."""
    monkeypatch.setattr(presets, "new_seed", lambda: 48213)
    for size in (SIZE, (1280, 720), (1920, 1080)):
        app = make_app(size=size)
        show_tab(app, viewer.PRESETS_TAB)
        for name in presets.names():
            app.preset_choice = name
            for status in ("", "running", "running, changed"):
                if status:
                    app.apply_preset(name)
                if status == "running, changed":
                    app.apply("exit_prob", 0.4 if app.sim.cfg.exit_prob != 0.4 else 0.45)
                app._layout_panel()
                caption = widget(app, "info", "caption")
                text = app._caption(caption.rect.w)
                assert app.fonts.width(text, "small") <= caption.rect.w, (size, name, text)
                assert text.startswith(f"{presets.names().index(name) + 1} of {len(presets.names())}"), text
                assert text.endswith(f"·  {status}") if status else "running" not in text, (size, name, text)
        # A long origin: a preset, one tab drawn on it, then four settings changed by hand.
        app.apply_preset("us_single_lane_peak")
        app.randomize(["Drivers"])
        for setting, value in (("density", 33.0), ("exit_prob", 0.41), ("entry_gap", 3.3), ("driver_noise", 0.11)):
            app.apply(setting, value)
        app._layout_panel()
        origin = widget(app, "info", "origin")
        lines = app.fonts.wrap(app._origin_line(), origin.rect.w)
        assert origin.rect.h == len(lines) * app.fonts.line("small") and len(lines) >= 2
        assert "Density, Exit chance, Entry gap and 1 more changed" in app._origin_line()
        frame(app)
        pygame.quit()


def test_preset_tab_lays_out_at_every_size(make_app):
    for size in (SIZE, (1280, 720), (1920, 1080)):
        app = make_app(size=size)
        show_tab(app, viewer.PRESETS_TAB)
        app.preset_choice = "etoile"  # the longest notes and source list
        app._layout_panel()
        frame(app)
        assert_fits(app)
        text = widget(app, "text", "preset_text")
        assert all(app.fonts.width(line, "small") <= text.rect.w for _, line, _ in text.lines)
        assert app.content_h > app.content_rect.h  # it scrolls ...
        for _ in range(200):
            app.handle(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-3, flipped=False, precise_x=0.0,
                                          precise_y=-3.0) if app.content_rect.collidepoint(app.mouse) else
                       move(app.content_rect.center))
        assert app.scroll == app.content_h - app.content_rect.h  # ... to its last source
        frame(app)
        pygame.quit()


def test_changes_after_a_preset_show_in_the_origin_markers_and_saved_settings(make_app, monkeypatch):
    app = make_app()
    app.apply_preset("etoile")
    frame(app)
    # A preset value between the slider's steps shows exactly, also in its marker.
    show_tab(app, "Traffic")
    exit_prob = widget(app, "slider", "exit_prob")
    assert viewer.format_value(exit_prob.spec, app.sim.cfg.exit_prob) == "0.17"
    track = exit_prob.track
    x = round(track.x + (0.2 - exit_prob.spec.lo) / (exit_prob.spec.hi - exit_prob.spec.lo) * track.w)
    click(app, (x, track.centery))
    assert app.sim.cfg.exit_prob == pytest.approx(0.2) and app.markers[-1][1] == "Exit chance 0.17→0.20"
    assert app.origin_text() == "preset etoile (Place Charles de Gaulle), then Exit chance changed"
    assert app.origin_text(short=True) == "Place Charles de Gaulle (changed)"
    _, txt = app.save_data()
    text = txt.read_text(encoding="utf-8")
    assert "  from preset etoile (Place Charles de Gaulle), then Exit chance changed" in text
    command = next(line for line in text.splitlines() if "experiments run" in line)
    assert "--preset etoile" in command and "--set exit_prob=0.2" in command and "n_lanes" not in command
    # The command gives the running settings again (exit chance as the slider left it).
    args = experiments.build_parser().parse_args(command.split()[3:])
    base = experiments._base(args, experiments.build_parser())
    assert settings(base.cfg) == settings(app.sim.cfg)
    # Random drivers on top of the preset: it stays the base, in the header and in the command.
    monkeypatch.setattr(presets, "new_seed", lambda: 55)
    app.apply_preset("etoile")
    show_tab(app, "Drivers")
    app.handle(key(pygame.K_x, pygame.KMOD_SHIFT))
    assert app.origin_text() == "preset etoile (Place Charles de Gaulle) with random Drivers settings, seed 55"
    assert app.origin_text(short=True) == "Place Charles de Gaulle, random Drivers"
    _, txt = app.save_data()
    command = next(line for line in txt.read_text(encoding="utf-8").splitlines() if "experiments run" in line)
    assert "--preset etoile" in command and "--set reaction_time=" in command
    args = experiments.build_parser().parse_args(command.split()[3:])
    assert settings(experiments._base(args, experiments.build_parser()).cfg) == settings(app.sim.cfg)


# ------------------------------------------------------------------ random settings
def test_randomize_all_and_one_tab_are_reproducible_by_their_seed(make_app, monkeypatch):
    app = make_app()
    frame(app)
    seeds = iter([1234, 77, 5])
    monkeypatch.setattr(presets, "new_seed", lambda: next(seeds))
    before = app.sim.cfg
    frame(app, key(pygame.K_x))  # X: every setting
    drawn = presets.random_changes(1234)
    assert {n: getattr(app.sim.cfg, n) for n in drawn} == drawn and app.sim.seed == SEED
    assert app.sim.cfg == viewer.config_from_args(viewer.parse_args(["--random", "1234", "--seed", str(SEED)]))
    assert "seed 1234" in app.message[0] and "--random 1234" in app.message[0]
    assert any(label == "Random settings, seed 1234" for _, label in app.markers)
    assert app.origin_text() == "random settings, seed 1234" and before != app.sim.cfg
    # Shift+X on the Drivers tab: only its settings, linked to the others as they are now.
    show_tab(app, "Drivers")
    other = app.sim.cfg
    frame(app, key(pygame.K_x, pygame.KMOD_LSHIFT))
    expected = presets.random_changes(77, ["Drivers"], base=other)
    for spec in PARAMS:
        assert getattr(app.sim.cfg, spec.name) == (expected[spec.name] if spec.group == "Drivers" else
                                                   getattr(other, spec.name)), spec.name
    assert app.markers[-1][1] == "Random Drivers settings, seed 77" and app.sim.time > 0  # live: no rebuild
    assert app.origin_text() == "random settings, seed 1234 with random Drivers settings, seed 77"
    # The button at the end of the tab (scrolled into view) does the same.
    app.scroll = 10 ** 6
    app._layout_panel()
    button = widget(app, "button", "randomize_tab")
    assert app.content_rect.contains(button.rect) and button.label == "Randomize Drivers (Shift+X)"
    click(app, button.rect.center)
    assert app.markers[-1][1] == "Random Drivers settings, seed 5"
    assert app.origin_text() == "random settings, seed 1234 with random Drivers settings, seed 5"
    assert app.origin_text(short=True) == "random, seed 1234, random Drivers"
    # The Presets tab has no settings of its own to randomize.
    show_tab(app, viewer.PRESETS_TAB)
    cfg = app.sim.cfg
    app.handle(key(pygame.K_x, pygame.KMOD_SHIFT))
    assert app.sim.cfg is cfg and "choose a parameter tab" in app.message[0]


def test_one_tab_draws_on_top_of_each_other_keep_every_seed_through_save_and_load(tmp_path, monkeypatch):
    seeds = iter(range(500, 600))
    monkeypatch.setattr(presets, "new_seed", lambda: next(seeds))
    for start, base in ((None, ""), ("etoile", "preset etoile (Place Charles de Gaulle) with "), ("X", "")):
        folder = tmp_path / str(start)
        app = viewer.App(SimConfig(seed=SEED), size=SIZE, results_dir=folder, autosave=True)
        if start == "etoile":
            app.apply_preset("etoile")
        elif start == "X":
            base = f"random settings, seed {app.randomize()} with "
        road, traffic = app.randomize(["Road"]), app.randomize(["Traffic"])
        assert app.origin_text() == f"{base}random Road settings, seed {road}, and Traffic settings, seed {traffic}"
        assert app.changed_since() == []  # the Road draw is not taken for changes by hand
        again = app.randomize(["Road"])  # a tab drawn again replaces its earlier draw
        assert app.origin_text() == f"{base}random Traffic settings, seed {traffic}, and Road settings, seed {again}"
        if start is None:
            assert app.origin_text(short=True) == f"random Traffic, Road, seeds {traffic}, {again}"
        app.apply("exit_prob", 0.42)
        live = app.origin_text()
        assert live.endswith("then Exit chance changed")
        app.save_state()
        app.save_session()
        pygame.quit()
        resumed = viewer.App(SimConfig(seed=SEED), size=SIZE, results_dir=folder, resume="session")
        assert resumed.origin_text() == live  # the session keeps every draw
        resumed.randomize()
        assert resumed.load_state()  # and so does a saved state (F9)
        assert resumed.origin_text() == live
        pygame.quit()


def test_random_flags_given_are_named_and_count_as_changes(tmp_path):
    def start(*argv: str) -> viewer.App:
        args = viewer.parse_args(list(argv))
        return viewer.App(viewer.config_from_args(args), size=SIZE, results_dir=tmp_path,
                          origin=viewer.origin_from_args(args))

    app = start("--random", "40", "--density", "30", "--seed", "1")
    assert app.origin_text() == "random settings, seed 40 (given: Density)"
    assert app.origin_text(short=True) == "random, seed 40, Density given"
    assert app.message[0].startswith("Started from random settings, seed 40 (given: Density):")
    _, txt = app.save_data()
    text = txt.read_text(encoding="utf-8")
    assert "  from random settings, seed 40 (given: Density)" in text
    command = next(line for line in text.splitlines() if "experiments run" in line)
    assert "--random 40" in command and command.endswith("--set density=30")  # the flag, as given
    app.apply("density", 55.0)  # a flag's value changed later counts, as with --preset
    assert app.changed_since() == ["Density"]
    assert app.origin_text() == "random settings, seed 40 (given: Density), then Density changed"
    pygame.quit()
    app = start("--random", "40", "--traffic_mode", "open")
    assert app.origin_text() == "random settings, seed 40 (given: Traffic mode)" and app.sim.cfg.traffic_mode == "open"
    pygame.quit()
    assert start("--random", "40").origin_text() == "random settings, seed 40"  # nothing given
    pygame.quit()


def test_randomize_button_and_saved_settings_name_the_seed(make_app, monkeypatch):
    app = make_app()
    monkeypatch.setattr(presets, "new_seed", lambda: 31)
    show_tab(app, viewer.PRESETS_TAB)
    click(app, widget(app, "button", "randomize").rect.center)
    frame(app)
    assert settings(app.sim.cfg) == {**settings(SimConfig()), **presets.random_changes(31)}
    _, txt = app.save_data()
    text = txt.read_text(encoding="utf-8")
    assert "  from random settings, seed 31" in text
    command = next(line for line in text.splitlines() if "experiments run" in line)
    assert "--random 31" in command and "--set" not in command  # exactly the settings --random 31 draws


# ------------------------------------------------------------------ command line
def test_preset_random_and_list_flags(capsys):
    cfg = viewer.config_from_args(viewer.parse_args(["--preset", "sugiyama2008"]))
    assert cfg == presets.apply("sugiyama2008")
    # Defaults < preset < explicit flags (as in experiments); the title works as a name.
    args = viewer.parse_args(["--preset", "US single-lane (HCM 6)", "--density", "40", "--seed", "3"])
    assert args.preset == "us_single_lane"
    assert viewer.config_from_args(args) == presets.apply("us_single_lane").updated(density=40.0, seed=3)
    assert viewer.explicit_settings(args) == ["--seed", "--preset", "--density"]
    assert viewer.origin_from_args(args)["preset"] == "us_single_lane"
    # --random: explicit flags take part in the links (an open-mode start density).
    args = viewer.parse_args(["--random", "40", "--traffic_mode", "open"])
    cfg = viewer.config_from_args(args)
    assert cfg.traffic_mode == "open" and cfg.density <= presets.DENSITY_OPEN.hi
    assert settings(cfg) == {**settings(SimConfig()), **presets.random_changes(40, fixed={"traffic_mode": "open"})}
    assert viewer.config_from_args(viewer.parse_args(["--random", "40"])).density > presets.DENSITY_OPEN.hi
    assert viewer.session_options(args) == {"resume": None, "restore_view": True}  # a new simulation
    assert viewer.origin_from_args(args)["random"] == 40
    for bad, why in ((["--preset", "nowhere"], "sugiyama2008"), (["--preset", "etoile", "--random", "3"],
                                                                  "not allowed with"),
                     (["--random", "-1"], "seed must be"), (["--state", "x.npz", "--preset", "etoile"],
                                                            "--preset"), (["--n_lanes", "0"], "n_lanes")):
        with pytest.raises(SystemExit) as exc:
            viewer.parse_args(bad)
        assert exc.value.code == 2 and why in capsys.readouterr().err, bad
    with pytest.raises(SystemExit) as exc:
        viewer.parse_args(["--list-presets"])
    assert exc.value.code in (0, None) and capsys.readouterr().out == presets.list_text()
    with pytest.raises(SystemExit):
        viewer.parse_args(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    for words in ("--preset NAME", "--random SEED", "--list-presets", "X random settings", "V open the sweep viewer"):
        assert words in text, words


def test_a_preset_session_resumes_with_its_origin(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # main() keeps its files in ./results
    apps = []
    run = viewer.App.run

    def run_a_little(self):
        apps.append(self)
        self.advance(20)
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        run(self)

    monkeypatch.setattr(viewer.App, "run", run_a_little)
    viewer.main(["--size", "1100x700", "--preset", "stern2018", "--seed", "2"])
    first = apps[0]
    assert first.origin_text() == "preset stern2018 (Stern 2018 ring)" and first.preset_choice == "stern2018"
    session = json.loads((tmp_path / "results" / "session" / "session.json").read_text(encoding="utf-8"))
    assert session["origin"] == "preset stern2018 (Stern 2018 ring)"
    viewer.main(["--size", "1100x700"])  # resumes it, origin and all
    second = apps[1]
    assert second.sim.cfg == first.sim.cfg and second.sim.time == pytest.approx(first.sim.time + 2.0)
    assert second.origin_text() == first.origin_text() and second.preset_choice == "stern2018"
    viewer.main(["--size", "1100x700", "--random", "9"])  # a new simulation from random settings
    assert apps[2].sim.time == pytest.approx(2.0) and apps[2].origin_text() == "random settings, seed 9"
    viewer.main(["--size", "1100x700"])
    assert apps[3].origin_text() == "random settings, seed 9" and apps[3].sim.time == pytest.approx(4.0)
    pygame.quit()


# ------------------------------------------------------------------ the sweep viewer button
class FakePopen:
    """Stands in for subprocess.Popen: records the launch, never runs anything."""
    launched: list["FakePopen"] = []

    def __init__(self, argv, **kwargs):
        self.argv, self.kwargs, self.code = list(argv), kwargs, None
        FakePopen.launched.append(self)

    def poll(self):
        return self.code


def test_sweep_viewer_key_writes_the_base_settings_and_does_not_wait(make_app, monkeypatch, tmp_path):
    FakePopen.launched = []
    monkeypatch.setattr(viewer.subprocess, "Popen", FakePopen)
    app = make_app()
    app.apply_preset("us_two_lane")
    frame(app, key(pygame.K_v))
    proc, = FakePopen.launched
    argv = proc.argv
    assert argv[:3] == [sys.executable, "-m", "roundabout_sim.sweep_viewer"]
    assert argv[3] == "--base-settings" and argv[5:] == ["--results", str((tmp_path / "results").resolve())]
    base = Path(argv[4])
    assert base.parent == (tmp_path / "results").resolve() and base.name.startswith("sweep_base_")
    assert proc.kwargs["stdin"] == subprocess.DEVNULL
    if sys.platform == "win32":
        assert proc.kwargs["creationflags"] & subprocess.CREATE_NO_WINDOW
    else:
        assert proc.kwargs["start_new_session"]
    # A flat {parameter: value, "seed": ...} file that the sweep viewer takes as it is.
    raw = json.loads(base.read_text(encoding="utf-8"))
    assert raw == state.settings_of(app.sim) and raw["seed"] == SEED
    changes, seed, ignored = sv.load_base(base)
    assert seed == SEED and not ignored and SimConfig().updated(**changes) == app.sim.cfg.updated(seed=None)
    argv = sv.sweep_args([("entry_rate", [0.1, 0.2])], seeds=1, warmup=0, measure=10, out="s", base=changes)
    sets = [argv[i + 1] for i, a in enumerate(argv) if a == "--set"]  # the preset's settings, not the swept one
    assert "n_lanes=2" in sets and "radius=22.25" in sets and not any(s.startswith("entry_rate=") for s in sets)
    assert sv.changed_settings(changes) == {n: v for n, v in changes.items() if v != getattr(SimConfig(), n)}
    assert "Opening the sweep viewer" in app.message[0]
    # A second press opens another; neither is waited for. One that fails says where its errors went.
    click(app, widget(app, "button", "sweep").rect.center)
    assert len(FakePopen.launched) == 2 and len(app.sweep_viewers) == 2
    FakePopen.launched[0].code = 1
    frame(app)
    assert "stopped with an error" in app.message[0] and viewer.SWEEP_LOG in app.message[0]
    assert app.sweep_viewers == [FakePopen.launched[1]]


def test_sweep_viewer_reads_a_base_file_of_every_setting(tmp_path):
    """The note counts only the settings that differ from the defaults (the file lists all of them)."""
    path = tmp_path / "base.json"
    state.write_json(path, {**settings(presets.apply("etoile")), "seed": 4})
    viewer_ = sv.SweepViewer(None, size=SIZE, results_dir=tmp_path / "results", base_settings=path)
    try:
        changed = sv.changed_settings(sv.load_base(path)[0])
        assert changed == presets.PRESETS["etoile"].changes  # (all of them differ from the defaults)
        assert viewer_.base_note == f"{len(changed)} changed from the defaults (base.json)"
        assert viewer_.base_seed == 4
    finally:
        pygame.quit()


# ------------------------------------------------------------------ experiments
def test_experiments_random_takes_the_explicit_settings_into_its_links():
    parser = experiments.build_parser()
    alone = experiments._base(parser.parse_args(["run", "--random", "40"]), parser)
    opened = experiments._base(parser.parse_args(["run", "--random", "40", "--traffic-mode", "open"]), parser)
    assert alone.cfg.traffic_mode == "closed" and alone.cfg.density > presets.DENSITY_OPEN.hi
    assert opened.cfg.traffic_mode == "open" and opened.cfg.density <= presets.DENSITY_OPEN.hi
    assert opened.cfg.density == presets.random_changes(40, fixed={"traffic_mode": "open"})["density"]
    # The viewer's --random gives the same settings.
    view = viewer.config_from_args(viewer.parse_args(["--random", "40", "--traffic-mode", "open"]))
    assert settings(view) == settings(opened.cfg)
    one_lane = experiments._base(parser.parse_args(["run", "--random", "40", "--set", "n_lanes=1"]), parser)
    spec = presets.RADIUS_BY_LANES[1]
    assert spec.lo <= one_lane.cfg.radius <= spec.hi  # a radius drawn for one lane


def test_values_between_the_slider_steps_show_exactly():
    radius = PARAM_BY_NAME["radius"]
    assert radius.step == 1 and viewer.format_value(radius, 80.0) == "80"
    assert [viewer.format_value(radius, r) for r in (18.25, 36.6, 67.25)] == ["18.25", "36.6", "67.25"]
    assert viewer.format_value(PARAM_BY_NAME["density"], 95.7) == "95.7"
    assert viewer.format_value(PARAM_BY_NAME["exit_prob"], 0.1) == "0.10"  # the step's decimals stay
    assert viewer.format_value(PARAM_BY_NAME["car_length"], 4.0) == "4.0"
    assert sv.setting_text("radius", 18.25) == "18.25" and sv.setting_phrase("radius", 80.0) == "Ring radius = 80 m"
