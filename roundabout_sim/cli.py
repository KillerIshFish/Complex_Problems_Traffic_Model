"""Command-line pieces shared by the viewer (``python -m roundabout_sim``), the experiments CLI and the
sweep viewer: the --preset / --random / --list-presets flags, the seed and window-size types, the
strict reading of a setting's value, and how a child process (a sweep, a sweep viewer) is started.

Imports neither pygame nor matplotlib, so that building a parser stays cheap.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from roundabout_sim.config import PARAM_BY_NAME, coerce
from roundabout_sim.textfmt import short_value

ROOT = Path(__file__).resolve().parents[1]  # the folder holding the package (importable by a child process)

LIST_PRESETS_HELP = "list the presets (name, description, settings, main source) and exit"


class ListPresetsAction(argparse.Action):
    """--list-presets: print the presets and exit (before any required argument is missed)."""

    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings, dest=dest, default=default, nargs=0, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        from roundabout_sim import presets

        print(presets.list_text(), end="")
        parser.exit()


def seed_arg(text: str) -> int:
    """A seed flag (--seed, --random): a whole number >= 0, checked before anything runs."""
    try:
        value = int(text)
    except ValueError:
        value = -1
    if value < 0:
        raise argparse.ArgumentTypeError(f"seed must be a whole number >= 0, got {text!r}")
    return value


def window_size(text: str) -> tuple[int, int]:
    """--size WxH: a window size in pixels (the viewers raise one below the smallest legible window,
    ui.MIN_WINDOW, to it)."""
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, e.g. 1600x900") from None
    return w, h


def setting_value(name: str, value):
    """config.coerce, with the parameter named in the error message; a whole-number setting (lanes,
    entrances, queue length) takes only whole numbers instead of rounding 2.5 silently. The viewer's
    parameter flags, the experiments' flags, --set, --vary and plan cells all read values with it."""
    try:
        out = coerce(name, value)
        if type(out) is int and not float(value).is_integer():
            raise ValueError("must be a whole number")
        return out
    except ValueError as err:
        shown = value if isinstance(value, str) else short_value(value)  # (as typed: '2.5' from either CLI)
        raise ValueError(f"bad value {shown!r} for {name}: {err}") from None


def range_note(name: str, value) -> str:
    """'note: n_lanes=9 is outside the usual range 1-5' for a value off its slider's range ("" else)."""
    spec = PARAM_BY_NAME.get(name)
    if spec and not spec.choices and not spec.lo <= value <= spec.hi:
        return f"note: {name}={value:g} is outside the usual range {spec.lo:g}-{spec.hi:g}"
    return ""


def child_process(extra_env: dict | None = None) -> dict:
    """Popen keyword arguments for a child process of the viewers (a sweep, a sweep viewer): the package
    importable from any working folder (ROOT on PYTHONPATH) and, on Windows, no console window of its
    own; elsewhere a session of its own, so that it outlives a viewer stopped with Ctrl+C and can be
    stopped with its workers."""
    env = {**os.environ, **(extra_env or {}),
           "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH", "")]))}
    detach = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else \
        {"start_new_session": True}
    return {"env": env, **detach}


def shell_line(argv: list[str]) -> str:
    """A command as one line to paste into this platform's shell (quoted for cmd.exe or a POSIX shell)."""
    return subprocess.list2cmdline(argv) if sys.platform == "win32" else shlex.join(argv)


def add_base_flags(parser: argparse.ArgumentParser, overrides: str, seed_note: str = "") -> None:
    """--preset / --random (the base settings, one or the other) and --list-presets. ``overrides``
    names the flags that override the base ("parameter flags", "--set and the parameter flags");
    ``seed_note`` says where the drawn seed shows, if anywhere."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preset", metavar="NAME",
                       help="start from a real-world preset (see --list-presets; name or title, case and - _ "
                            f"ignored); {overrides} given with it override its values")
    group.add_argument("--random", type=seed_arg, metavar="SEED",
                       help="start from plausible random values of every parameter (but the automated "
                            "vehicles' and cooperation ones, which keep their defaults), drawn with this seed"
                            f"{seed_note}; {overrides} given with it stay as given and the settings linked to "
                            "them follow (e.g. with --traffic-mode open, an open-mode start density)")
    parser.add_argument("--list-presets", action=ListPresetsAction, help=LIST_PRESETS_HELP)
