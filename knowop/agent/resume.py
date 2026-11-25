"""Pick the journal an agent process continues a stored session in, on this host.

A session is the parts the server stores, each one journal's records, named for the
journal's folder: ``<folder>.jsonl``. A process continues the newest part's own
journal when this host holds it whole; otherwise it restores every stored part into a
new journal (:func:`~knowop.agent.materialize.materialize`), so nothing
local is renamed, merged or deleted.

A restore stands its model on what this host offers. A remote agent signs in with
keys alone, so a session that ran on a subscription (``anthropic-sub``) would not
resume there: sagent resumes on the model its journal stood, or not at all. The
restore forgets each cached model the host lacks, and moves a model it lacks to the
same model on the keyed provider its provider extends, else to a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict
from uuid import UUID, uuid4

import re

from trackinizer.client.conversation import read_part
from trackinizer.lib.userdirs import state_dir
from trackinizer.sagent.journal.types import SessionRecord, UncategorizedRecord
from treekle import from_plain, immutable

from knowop.agent.errors import (
    LossyConversionError,
    NotResumableError,
)
from knowop.agent.materialize import materialize
from knowop.agent.truncate import was_truncated, without_note


if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

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

    unoffered: str = ""
    """The provider the session's model stood on, which this host lacks, so another
    stands in for it; empty when none does."""


def session_journal(
    session_id: UUID,
    parts: Sequence[PartBody],
    *,
    client: PartReader,
    lossy: bool,
    offered: Collection[str] = (),
    keyed: Collection[tuple[str, str]] = (),
    fallback: str = "openai/sol-6.2",
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
      offered: The providers this host's agent may use, by their own names; empty
        for every provider, whose models a restore stands as they stood.
      keyed: Each provider ``offered`` lacks, paired with the offered keyed one it
        extends, which serves its models under the same names (``anthropic-sub``,
        ``anthropic``).
      fallback: ``PROVIDER/MODEL``, what a restore stands a model ``offered`` lacks
        on when ``keyed`` pairs its provider with none.

    Returns:
      choice: The journal, its part, and where that part's upload starts.

    Raises:
      NotResumableError: No part has a native format.
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
    return _restored(
        session_id,
        parts,
        client=client,
        lossy=lossy,
        folder=folder,
        offered=offered,
        keyed=keyed,
        fallback=fallback,
    )


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
    offered: Collection[str],
    keyed: Collection[tuple[str, str]],
    fallback: str,
) -> JournalChoice:
    """Write a new journal holding every stored part's records."""
    # Any CLI's part resumes: its records are sagent's. A formatless part holds a
    # capture's notes, never a conversation (``ManifestBody.format``).
    native = [part for part in parts if part.format]
    if not native:
        raise NotResumableError(
            f"session {session_id} holds no part with a native format, so there is "
            "no conversation to continue; it stays readable",
        )
    records: list[SessionRecord] = []
    sealed: list[str | None] = []
    for part in sorted(native, key=lambda each: each.part):
        read, seals = _captured(client, session_id, part, lossy=lossy)
        for record, seal in zip(read, seals, strict=True):
            if not (
                isinstance(record, UncategorizedRecord)
                and record.kind in _STANDING_CHILDREN
            ):
                records.append(record)
                sealed.append(seal)
    unoffered = (
        _offered_model(records, offered=offered, keyed=keyed, fallback=fallback)
        if offered
        else ""
    )
    path = materialize(records, sealed=sealed, folder=folder)
    # Counted as stored, so the host's model goes up in no part: a later restore on a
    # host that holds the session's own sign-in resumes on the model it stood.
    return JournalChoice(
        path=path,
        part=f"{path.parent.name}.jsonl",
        start=_lines(path),
        restored=True,
        unoffered=unoffered,
    )


class _Stood(TypedDict):
    """What of sagent's standing ``model`` record a restore reads."""

    model: str
    cache: list[str]


# Sagent resumes on the model its newest ``model`` record stands, and builds each
# line's model from that line's provider alone, the cache's too: one line this host
# lacks fails the whole resume.
def _offered_model(
    records: list[SessionRecord],
    *,
    offered: Collection[str],
    keyed: Collection[tuple[str, str]],
    fallback: str,
) -> str:
    """Stand the newest model on what ``offered`` holds, in place; return whose it moved."""
    stood = [
        at
        for at, record in enumerate(records)
        if isinstance(record, UncategorizedRecord) and record.kind == "model"
    ]
    if not stood:
        return ""
    record = records[stood[-1]]
    assert isinstance(record, UncategorizedRecord)
    was = from_plain(record.payload, _Stood)
    cache = [line for line in was["cache"] if _provider(line) in offered]
    model = was["model"]
    if _provider(model) not in offered:
        kept = [line for line in cache if _bare(line) == _bare(fallback)]
        model = _keyed(model, keyed=keyed) or (kept or [fallback])[0]
        cache = [line for line in cache if line != model]
    records[stood[-1]] = UncategorizedRecord(
        kind=record.kind,
        payload=immutable({**record.payload, "model": model, "cache": cache}),
    )
    return "" if model == was["model"] else _provider(was["model"])


def _keyed(line: str, *, keyed: Collection[tuple[str, str]]) -> str:
    """Return ``line`` on the keyed provider ``keyed`` pairs its provider with, else ``""``."""
    signed, _, rest = line.partition("/")
    account, at, provider = signed.rpartition("@")
    on = dict(keyed).get(provider, "")
    return f"{account}{at}{on}/{rest}" if on else ""


def _provider(line: str) -> str:
    """Return the provider a model line names: ``work@openai/sol`` is ``openai``'s."""
    return line.partition("/")[0].rpartition("@")[2]


def _bare(line: str) -> str:
    """Return a model line without its tags or choices: ``openai/sol+highthink`` is ``openai/sol``."""
    return line.partition("+")[0].partition(" ")[0]


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
