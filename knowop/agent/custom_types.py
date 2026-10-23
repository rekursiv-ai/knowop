"""Types shared across ``knowop run``, above the adapter package.

The ``Event`` record every sink consumes: one act a session performed, plus
where it came from. It lives here rather than beside the ``Adapter`` protocol
because the sinks never touch an adapter -- keeping it in ``adapters/`` made
every sink import the adapter package for one dataclass.

The act itself is the shared IR record (``trackinizer.lib.agent.types.sessions``)
widened by the scrape's stream records (:data:`TraxRecord`): one vocabulary
reads a session, stores it, and writes it back out as any CLI's native format.

The adapter-facing protocols are the sibling ``adapters/custom_types.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from trackinizer.types.streams import TraxRecord


__all__ = ["Event", "FlagValue", "HostRequest", "RunRequest"]


type FlagValue = str | int | bool | list[str] | None


class RunRequest(TypedDict):
    """Plain inputs from which the detached interpreter rebuilds a runner."""

    flags: dict[str, FlagValue]
    cli_argv: list[str]
    connection: dict[str, FlagValue] | None
    resume_path: str | None
    cli_session_id: str | None
    profile: str | None
    attach: str | None
    prefix_bytes: int
    compact_after_load: bool
    run_name: str


class HostRequest(TypedDict):
    """Plain inputs passed through the detached host's inherited descriptor."""

    paths: str
    address: str
    control: str
    resume: str
    command: str
    ready: int
    run: RunRequest


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """One captured act, in memory, before it becomes a wire body.

    Carries the FILE it came from because a session is several of them --
    claude splits on compaction, codex forks -- and each is stored as its own
    part with positions numbered from zero within it.
    """

    record: TraxRecord
    """The act itself: a message, a tool call, a result, a context change --
    or, from a scrape, one line and the stream it crossed."""

    path: Path
    """The session file this record was read from. The server resolves a part
    number from its BASENAME, since the absolute path differs across
    machines -- which is what lets a resumed run append to the part it
    materialized."""

    restart: bool = False
    """Whether the file was REPLACED immediately before this record.

    A claude compaction rewrites the transcript rather than appending, keeping
    the turns it did not summarize away. The record stored at a given position
    may therefore have changed, so a restarted batch overwrites rather than
    skipping what is already there.
    """

    @property
    def kind(self) -> str:
        """The record's class name, which selects its type on decode."""
        return type(self.record).__name__
