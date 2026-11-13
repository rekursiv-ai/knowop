"""Write a stored sagent session back out as a journal, to continue it on this host.

An agent process continues a session in a journal on its own host. When the host holds
none whole, ``resume.session_journal`` writes the stored records here as a new journal,
which sagent then resumes as one it wrote: the IR's round trip makes the file sagent's
own format, not a rendering for a person.

The journal sits in a new folder, named by a fresh uuid's hex as sagent names a session,
so nothing this host already holds is overwritten. Ciphertext is spliced here: it lives
in its own table precisely so retention can drop it, and a session whose sealed
reasoning is gone cannot be replayed to the provider (:class:`CiphertextDroppedError`).
"""

from __future__ import annotations

from dataclasses import replace
from io import StringIO
from typing import TYPE_CHECKING
from uuid import uuid4

import json

from trackinizer.sagent.scripts.convert_session import sagent
from trackinizer.sagent.session.types import IncompleteRecord, Thinking

from knowop.agent.errors import CiphertextDroppedError


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from trackinizer.sagent.session.types import SessionRecord


__all__ = ["materialize"]


def materialize(
    records: Sequence[SessionRecord],
    *,
    sealed: Sequence[str | None],
    folder: Path,
) -> Path:
    """Write ``records`` as a new sagent journal under ``folder``; return its path.

    Args:
      records: The session's records, in order.
      sealed: Each record's ciphertext, positionally aligned with ``records``
        (``None`` where a record has none).
      folder: Where the journal's own folder is made: the session's journals folder.

    Returns:
      path: The journal, ``<folder>/<hex>/session.jsonl``.

    Raises:
      CiphertextDroppedError: A ``Thinking`` record needs sealed bytes that are
        not present.

    """
    # Whole before anything is written: a partly written journal left on disk would be
    # a session this host appears to hold.
    journal = StringIO()
    sagent.denormalize(
        [each for each in _spliced(records, sealed) if _writable(each)],
        journal,
    )
    path = folder / uuid4().hex / "session.jsonl"
    # Private, as sagent keeps its own sessions: what was said is its owner's alone.
    path.parent.mkdir(parents=True, mode=0o700)
    _ = path.write_bytes(journal.getvalue().encode())
    return path


# Only :class:`IncompleteRecord` can fail: every other member is structured, and the
# writer builds its line. This one carries raw bytes, and the writer emits them as they
# stand.
#
# A real capture stored 4498 characters -- a ``cost-state``, an ``atis-latch`` and a
# ``turn_context`` concatenated with no separators -- under a SINGLE record, which is
# not one line in any format: written back into a file it produced a line that parsed
# as none.
#
# Anything that IS one line replays verbatim, valid or not: a truncated final line is
# exactly what this record exists to preserve, and rewriting the file it came from must
# reproduce it byte for byte.
def _writable(record: SessionRecord) -> bool:
    """Whether ``record`` can be written as ONE line of the journal."""
    return not isinstance(record, IncompleteRecord) or _one_line(record.text)


def _one_line(text: str) -> bool:
    """Whether ``text`` is a single line holding at most one JSON value."""
    if "\n" in text.rstrip("\n"):
        return False
    try:
        end = json.JSONDecoder().raw_decode(text.strip())[1]
    except ValueError:
        # Unparsable as a whole is the TRUNCATED case this record preserves;
        # one line, so it replays.
        return True
    return not text.strip()[end:].strip()


def _spliced(
    records: Sequence[SessionRecord],
    sealed: Sequence[str | None],
) -> list[SessionRecord]:
    """Rejoin each record with its ciphertext, or refuse if any is missing."""
    out: list[SessionRecord] = []
    for idx, record in enumerate(records):
        bytes_ = sealed[idx] if idx < len(sealed) else None
        if not isinstance(record, Thinking):
            out.append(record)
            continue
        if bytes_ is not None:
            out.append(replace(record, encrypted=bytes_))
            continue
        # Empty marks split ciphertext; None means the record was never sealed.
        # A summary can accompany a seal, so it cannot establish replayability.
        content = record.content or ""
        if record.encrypted == "" and not content:
            raise CiphertextDroppedError(
                f"record {idx} is sealed reasoning whose ciphertext is no "
                "longer stored; the provider rejects a transcript with an "
                "empty 'encrypted' field, so this session cannot be resumed",
            )
        out.append(record)
    return out
