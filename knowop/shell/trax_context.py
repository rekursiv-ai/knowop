"""The trax context a host keeps out of what it starts for the server.

A ``knowop connect`` started where trax is configured inherits a server, a credential and a
profile. A terminal run or a command the server sends must never carry them: the
credential would reach the server's own assistant in the command's output.
"""

from __future__ import annotations

from typing import Final

import os


__all__ = ["inherited_trax_context"]

_ROUTING_ENV: Final = frozenset(
    {"TRACKINIZER_URL", "TRACKINIZER_TOKEN", "TRACKINIZER_PROFILE"},
)
"""The variables ``trax`` reads to choose its server, credential and profile."""


def inherited_trax_context() -> frozenset[str]:
    """Return the host's variables that would route a child off this station.

    Returns:
      names: The routing variables set here, and every ``TRAX_*`` and ``LC_TRAX_*``
        name, such as one left by a trax run the host itself sits in.

    """
    return frozenset(
        name
        for name in os.environ
        if name in _ROUTING_ENV or name.startswith(("TRAX_", "LC_TRAX_"))
    )
