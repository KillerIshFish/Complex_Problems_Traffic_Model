"""Saved states: a simulation, its chart history and what the viewer keeps with them, in one file.

A state is a single ``.npz`` archive (numpy's zip of arrays, without pickle:
``np.load(path, allow_pickle=False)`` reads it) holding

* ``sim.<name>`` - the arrays of :meth:`RingSim.state_dict` (per-car arrays,
  acceleration history, queues, event logs, ...);
* ``rec.<name>`` - the arrays of :meth:`Recorder.state_dict` (the chart
  samples), when a recorder was saved too;
* ``meta_json`` - a JSON string stored as a numpy unicode array: the format
  name and FORMAT_VERSION, when it was saved, the settings (every SimConfig
  field, with the seed the run actually uses), the rest of the simulation's
  state (clock, counters, timers, the random generators' states: the
  ``meta`` of ``state_dict``), the recorder's, and whatever the caller adds
  under "app" (the viewer: chart markers, braked cars, view settings).

Files are written atomically: to a temporary file beside the target, then
moved over it (``os.replace``), so a crash or a full disk mid-save never leaves
half a file. :func:`load` raises :class:`StateError` with a readable reason for
a missing, damaged or incompatible file (another format version, settings
this version does not know, arrays of the wrong shape); :func:`saved_settings`
and :func:`valid_settings` still recover whatever settings such a file holds.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable

import numpy as np

from roundabout_sim.config import SimConfig, coerce
from roundabout_sim.engine import RingSim, StateError
from roundabout_sim.metrics import Recorder

__all__ = ["FORMAT", "FORMAT_VERSION", "SavedState", "StateError", "load", "pack", "read_json", "save",
           "saved_settings", "settings_of", "valid_settings", "write_json"]

FORMAT = "roundabout_sim state"  # names the file's kind in its meta_json
FORMAT_VERSION = 1  # layout of the file; RingSim states carry their own engine.STATE_VERSION too


@dataclass
class SavedState:
    """A state as read from (or about to be written to) a file: the parsed ``meta_json`` and the arrays."""
    meta: dict
    arrays: dict[str, np.ndarray]
    path: Path | None = None

    @property
    def saved(self) -> str:
        """When it was saved ("YYYY-MM-DD HH:MM:SS")."""
        return str(self.meta.get("saved", "at an unknown time"))

    @property
    def app(self) -> dict:
        """What the program that saved it kept with it (the viewer: markers, braked cars, view)."""
        app = self.meta.get("app")
        return app if isinstance(app, dict) else {}

    def part(self, prefix: str) -> dict | None:
        """``{"meta", "arrays"}`` of one part ("sim" or "rec"), None if the file has no such part."""
        meta = self.meta.get(prefix)
        if meta is None:
            return None
        start = prefix + "."
        return {"meta": meta, "arrays": {name[len(start):]: arr for name, arr in self.arrays.items()
                                         if name.startswith(start)}}

    def simulation(self) -> RingSim:
        """The saved simulation, carrying on exactly where it was (raises :class:`StateError`)."""
        part = self.part("sim")
        if part is None:
            raise StateError("it holds no simulation")
        return RingSim.from_state(part)


def settings_of(sim: RingSim) -> dict:
    """Every setting of ``sim`` as plain JSON values, with the seed the run actually uses."""
    values = {name: value.item() if isinstance(value, np.generic) else value
              for name, value in asdict(sim.cfg).items()}
    return {**values, "seed": sim.seed}


def pack(sim: RingSim, recorder: Recorder | None = None, app: dict | None = None) -> SavedState:
    """``sim`` (and its ``recorder``, and the JSON-serialisable ``app`` extras) as a :class:`SavedState`."""
    sim_state = sim.state_dict()
    meta = {"format": FORMAT, "version": FORMAT_VERSION, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
            "settings": settings_of(sim), "sim": sim_state["meta"], "rec": None, "app": app or {}}
    arrays = {f"sim.{name}": arr for name, arr in sim_state["arrays"].items()}
    if recorder is not None:
        rec_state = recorder.state_dict()
        meta["rec"] = rec_state["meta"]
        arrays.update({f"rec.{name}": arr for name, arr in rec_state["arrays"].items()})
    return SavedState(meta, arrays)


def save(path: str | Path, sim: RingSim, recorder: Recorder | None = None, app: dict | None = None) -> SavedState:
    """Write ``sim`` (with its ``recorder`` and ``app`` extras) to ``path`` atomically; returns what was saved."""
    saved = pack(sim, recorder, app)
    saved.path = Path(path)
    arrays = {**saved.arrays, "meta_json": np.array(json.dumps(saved.meta))}
    write_atomic(saved.path, lambda fh: np.savez_compressed(fh, **arrays))
    return saved


def load(path: str | Path) -> SavedState:
    """Read a state file. Raises :class:`StateError` (saying why) if it is missing, damaged, not a
    state of this model or of another format version; :meth:`SavedState.simulation` checks the rest."""
    path = Path(path)
    meta, arrays = _read(path)
    if meta.get("format") != FORMAT:
        raise StateError(f"{path.name} is not a saved state of this model")
    version = meta.get("version")
    if version != FORMAT_VERSION:
        raise StateError(f"{path.name} was saved in format version {version}; this version reads {FORMAT_VERSION}")
    return SavedState(meta, arrays, path)


def saved_settings(path: str | Path) -> dict | None:
    """The settings a state file holds, whatever else is wrong with it (None if not even they can be read).
    Never raises: callers use it to recover from a file that has just failed to load."""
    try:
        meta, _ = _read(Path(path), arrays=False)
    except Exception:  # (StateError for every failure _read foresees; anything else is no better)
        return None
    settings = meta.get("settings")
    return settings if isinstance(settings, dict) else None


def valid_settings(values: dict, base: SimConfig | None = None) -> tuple[SimConfig, list[str]]:
    """``base`` (default: the defaults) with each of ``values`` that is a known setting (``seed``
    included) with a valid value, and the names of the ones left out."""
    cfg, skipped = base or SimConfig(), []
    known = {f.name for f in fields(SimConfig)}
    for name, value in values.items():
        try:
            if name not in known:
                raise KeyError(name)
            cfg = cfg.updated(**{name: coerce(name, value)})
        except (KeyError, TypeError, ValueError, OverflowError):
            skipped.append(str(name))
    return cfg, skipped


def write_json(path: str | Path, data: dict) -> None:
    """Write ``data`` as readable, standard JSON to ``path``, atomically. Infinite and NaN floats
    (density = inf is a valid setting) are written as the strings "inf", "-inf" and "nan", which
    :func:`valid_settings` reads back; standard JSON has no such numbers."""
    text = json.dumps(_finite_json(data), indent=2, allow_nan=False) + "\n"
    write_atomic(path, lambda fh: fh.write(text.encode("utf-8")))


def _finite_json(value):
    """``value`` with every non-finite float (in dicts and lists, all the way down) as a string."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # "inf", "-inf", "nan"
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


def read_json(path: str | Path) -> dict:
    """A JSON object from ``path``; raises :class:`StateError` if it is missing or not one."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StateError(f"there is no {path}") from None
    except (OSError, UnicodeDecodeError, ValueError) as e:
        raise StateError(f"{path.name} cannot be read ({e})") from e
    if not isinstance(data, dict):
        raise StateError(f"{path.name} does not hold a JSON object")
    return data


def _read(path: Path, arrays: bool = True) -> tuple[dict, dict[str, np.ndarray]]:
    """(meta, arrays) of a state file (the arrays only if asked for), checking it is one at all."""
    if not path.is_file():
        raise StateError(f"there is no saved state at {path}")
    try:
        with open(path, "rb") as fh:
            archive = zipfile.is_zipfile(fh)
    except OSError as e:  # e.g. permission denied, or locked by another program
        raise StateError(f"{path.name} cannot be read ({e.strerror or e})") from e
    if not archive:  # (np.load would take it for a pickle and suggest loading it unsafely)
        raise StateError(f"{path.name} is not a readable state file (not an .npz archive)")
    try:
        with np.load(path, allow_pickle=False) as data:
            names = data.files if arrays else [name for name in data.files if name == "meta_json"]
            contents = {name: data[name] for name in names}
    except OSError as e:
        raise StateError(f"{path.name} cannot be read ({e.strerror or e})") from e
    except Exception as e:  # anything numpy or zipfile raise for an archive that is damaged
        # Never pass on numpy's advice to load a file with Python objects in it "unsafely".
        why = "it holds Python objects, which are never loaded" if "pickle" in str(e) else f"{type(e).__name__}: {e}"
        raise StateError(f"{path.name} is not a readable state file ({why})") from e
    text = contents.pop("meta_json", None)
    if not isinstance(text, np.ndarray) or text.dtype.kind != "U" or text.ndim != 0:
        raise StateError(f"{path.name} has no meta_json, so it is not a saved state")
    try:
        meta = json.loads(str(text[()]))
    except (ValueError, RecursionError) as e:  # (nested too deeply for the parser: not one of ours either)
        raise StateError(f"the meta_json of {path.name} is not valid JSON ({e})") from e
    if not isinstance(meta, dict):
        raise StateError(f"the meta_json of {path.name} is not a JSON object")
    return meta, contents


def new_paths(folder: str | Path, stem: str, *suffixes: str) -> list[Path]:
    """``folder/<stem>_<date>_<time><suffix>`` for each suffix, none of them taken yet (saving twice in
    the same second adds -1, -2, ...); creates ``folder``. The viewers name their screenshots, chart
    data, saved states, sweep bases, new sweeps and PNGs with it."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    n = 0
    while True:
        paths = [folder / f"{stem}{'-' + str(n) if n else ''}{suffix}" for suffix in suffixes]
        if not any(p.exists() for p in paths):
            return paths
        n += 1


def write_atomic(path: str | Path, write: Callable) -> None:
    """Write a file through ``write(binary file)`` into a temporary file beside ``path`` (named
    ``.<name>.<random>.tmp``), flushed to disk, then move it over ``path``: readers see the old file or
    the new one, never half of one, even after a crash. The saved states and sessions, and the
    experiments' sweep CSV and meta file, are all written this way."""
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
