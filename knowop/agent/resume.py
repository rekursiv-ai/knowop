"""Pick the journal an agent process continues a stored session in, on this host.

A session is the parts the server stores, each one journal's records, named for the
journal's folder: ``<folder>.jsonl``. A process continues the newest part's own
journal when this host holds it whole; otherwise it restores every stored part into a
new journal (:func:`~knowop.agent.materialize.materialize`), so nothing
local is renamed, merged or deleted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final
from uuid import UUID, uuid4

import re

from trackinizer.client.conversation import read_part
from trackinizer.lib.userdirs import state_dir
from trackinizer.sagent.session.types import SessionRecord, UncategorizedRecord

from knowop.agent.errors import (
    LossyConversionError,
    NotResumableError,
)
from knowop.agent.materialize import materialize
from knowop.agent.truncate import was_truncated, without_note


if TYPE_CHECKING:
    from collections.abc import Sequence

    from trackinizer.client.conversation import PartReader
    from trackinizer.wire.wire_session_ir import PartBody


__all__ = ["JournalChoice", "session_journal"]


_STANDING_CHILDREN: Final = frozenset({"persistent_agent", "watcher"})
"""The standing records that make sagent start a child again from its journal, which
a restored session's host does not hold."""


@dataclass(frozen=True, slots=True, kw_only=True)
class JournalChoice:
    """The journal an agent process writes, and the part its records are stored in."""

    path: Path
    """The journal, ``<session's folder>/<folder>/session.jsonl``; it may not exist."""

    part: str
    """The part's name, ``<folder>.jsonl``."""

    start: int
    """The position of the journal's first record not yet stored."""

    restored: bool
    """Whether the journal was written afresh from the session's stored parts."""


def session_journal(
    session_id: UUID,
    parts: Sequence[PartBody],
    *,
    client: PartReader,
    lossy: bool,
) -> JournalChoice:
    """Return the journal an agent process continues ``session_id`` in, on this host.

    The newest part's own journal, when this host holds it whole; a new one, for a
    session with no parts; and otherwise one written from every stored part, in a
    new folder, so nothing local is renamed, merged or deleted.

    Args:
      session_id: The session.
      parts: Its parts, as the server lists them.
      client: Reads a part's records, to restore one.
      lossy: Whether to restore a part holding a record capture cut.

    Returns:
      choice: The journal, its part, and where that part's upload starts.

    Raises:
      NotResumableError: A part is not sagent's journal, so none can hold it.
      LossyConversionError: A part holds a cut record and ``lossy`` is false.

    """
    folder = state_dir() / "rekursiv-ai" / "knowop" / "journals" / str(session_id)
    if not parts:
        minted = uuid4().hex
        return JournalChoice(
            path=folder / minted / "session.jsonl",
            part=f"{minted}.jsonl",
            start=0,
            restored=False,
        )
    last = max(parts, key=lambda each: each.part)
    kept = _held_journal(folder, last)
    if kept is not None:
        return JournalChoice(
            path=kept,
            part=last.name,
            start=last.records,
            restored=False,
        )
    return _restored(session_id, parts, client=client, lossy=lossy, folder=folder)


# The part names its journal's folder, which this host may hold; a name that is not a
# folder's, such as one holding a ``..``, names none.
def _held_journal(folder: Path, part: PartBody) -> Path | None:
    """Return this host's journal of ``part``, if it holds every record stored."""
    if re.fullmatch(r"[0-9a-f]{32}\.jsonl", part.name) is None:
        return None
    path = folder / part.name.removesuffix(".jsonl") / "session.jsonl"
    return path if path.is_file() and _lines(path) >= part.records else None


# Each part replays as it was captured, in order, into one journal. The journal goes on
# where the last left off, so the records that restart a child are left out: the
# child's own journal is on the host that wrote it.
def _restored(
    session_id: UUID,
    parts: Sequence[PartBody],
    *,
    client: PartReader,
    lossy: bool,
    folder: Path,
) -> JournalChoice:
    """Write a new journal holding every stored part's records."""
    foreign = sorted({part.format for part in parts} - {"sagent"})
    if foreign:
        raise NotResumableError(
            f"session {session_id} holds a part of {', '.join(foreign)}, which no "
            "sagent journal continues; it stays readable",
        )
    records: list[SessionRecord] = []
    sealed: list[str | None] = []
    for part in sorted(parts, key=lambda each: each.part):
        read, seals = _captured(client, session_id, part, lossy=lossy)
        for record, seal in zip(read, seals, strict=True):
            if not (
                isinstance(record, UncategorizedRecord)
                and record.kind in _STANDING_CHILDREN
            ):
                records.append(record)
                sealed.append(seal)
    path = materialize(records, sealed=sealed, folder=folder)
    return JournalChoice(
        path=path,
        part=f"{path.parent.name}.jsonl",
        start=_lines(path),
        restored=True,
    )


def _captured(
    client: PartReader,
    session_id: UUID,
    part: PartBody,
    *,
    lossy: bool,
) -> tuple[list[SessionRecord], list[str | None]]:
    """Read one part as it was captured: each record, and each one's seal."""
    # Read WITH the ciphertext, unlike a viewer: a replay is the one caller that needs the
    # sealed half, since the provider validates it.
    records, sealed = read_part(client, session_id, part.part, plaintext_only=False)
    if not lossy and (cut := sum(map(was_truncated, records))):
        raise LossyConversionError(
            f"{cut} record(s) of session {session_id} were truncated at capture "
            "to fit an upload",
        )
    # The note is Trackinizer's, not sagent's: the writer replays an uncategorized
    # record's payload verbatim, note and all.
    return [without_note(record) for record in records], sealed


def _lines(path: Path) -> int:
    """Return how many records a journal holds: its lines, blank ones aside."""
    with path.open("rb") as journal:
        return sum(1 for line in journal if line.strip())
