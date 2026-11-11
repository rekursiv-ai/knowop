"""The entry point of the agent process: one session's sagent, live in Trackinizer.

Its contract lands ahead of the process itself, so the command line and a host's
daemon are written against one signature: this module declares what they call, and
the call raises until the process is built. ``codec`` names what crosses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

    from trackinizer.trax.profile import Profile


__all__ = ["main"]


def main(argv: Sequence[str], *, profile: Profile | None) -> int:
    """Run one session's agent process until it ends.

    Args:
      argv: What follows ``knowop run``: sagent's flags, and knowop's own.
      profile: The saved server and key it reports to; ``None`` when a daemon
        started it, which hands it both.

    Returns:
      code: 0 after a stop or a quit; 1 for a start error, a refused key or a
        session that is gone; 3 when another process holds the session; 4 when it
        ran headless too long without its socket.

    """
    del argv, profile
    raise NotImplementedError("The agent process is not built yet.")
