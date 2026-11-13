"""Why a stored session could not be continued in a journal here.

Their own module because an exception must be IMPORTED to be caught. Deferred with
the work that raises it, through ``lazy_import``, ``except`` would receive a proxy
rather than a class, which raises ``TypeError: catching classes that do not inherit
from BaseException`` at the moment the handler runs. The failure lands only on the
error path, so it replaces the diagnostic the handler existed to print with a
traceback.

Nothing here imports anything, so a caller binds these eagerly and defers only
the work.
"""

from __future__ import annotations


__all__ = [
    "CiphertextDroppedError",
    "LossyConversionError",
    "NotResumableError",
]


class NotResumableError(Exception):
    """A stored part is not sagent's journal, so no journal here continues it."""


class CiphertextDroppedError(Exception):
    """A record's sealed reasoning is gone, so the journal cannot be replayed.

    Raised rather than written with an empty ``encrypted``: the provider
    validates the field, so a silently-hollowed transcript fails at the model
    with no indication that retention was the cause.
    """


class LossyConversionError(Exception):
    """A stored part holds records capture truncated, and the loss was not accepted."""
