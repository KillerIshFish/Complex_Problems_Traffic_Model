"""``python -m roundabout_sim`` launches the interactive viewer (``--help`` for options and keys)."""

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")  # no pygame banner before --help / the window

from roundabout_sim.app import main  # noqa: E402

main()
