"""Send finished local agent sessions to Trackinizer, each as an AgentSession of its own.

``knowop upload PATH...`` reads Claude Code, Codex and sagent session files, named or
found in folders, and sends each as an AgentSession under the CLI's own id for it, so a
second upload lands on the same one and sends only the records it lacks. A session's
sub-agents go up as AgentSessions of their own, each after the session that spawned
it and linked to it by a ``produced_by`` edge, as a recorded run's were. A session
the server holds already keeps the audience it has, whatever the upload asks.

::

    PATH... ──plan──▶ Session: file, format, CLI id, what spawned it
            ──upload──▶ start ─▶ parts held ─▶ records not held ─▶ end ─▶ produced_by
                        (sessions side by side, ``jobs`` at a time, each a sub-agent's
                         spawner first)

Reading a transcript is the cost (normalizing it runs under the GIL), so sessions go
up in worker processes, each with a client of its own.

Typical usage example:

  sessions = plan([Path("~/.claude")])
  results = upload(sessions, connect=partial(Client, url, api_key=key),
                   private=True, jobs=8, redactor=redactor_from_environ(os.environ))
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, replace
from functools import partial
from itertools import batched, chain, islice
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Final, Protocol
from uuid import UUID, uuid4

import os
import re

from trackinizer.client.errors import ClientError
from trackinizer.client.session_upload import (
    RecordAppender,
    append_batch,
    record_bodies,
    send_batches,
)
from trackinizer.runs.adapters.claude import ClaudeAdapter
from trackinizer.runs.adapters.codex import CodexAdapter
from trackinizer.sagent.scripts.convert_session import claude, codex, sagent
from trackinizer.sagent.scripts.convert_session.convert import (
    die_with_parent,
    open_transcript,
    sniff,
    transcript_name,
    transcripts,
)
from trackinizer.sagent.session.jsonl import read_record
from trackinizer.sagent.session.projects import is_session
from trackinizer.sagent.session.types import TurnContext, UncategorizedRecord
from trackinizer.types.session_records import record_time
from trackinizer.wire.audience import PRIVATE
from trackinizer.wire.wire_session_ir import MAX_RECORD_BATCH
from trackinizer.wire.wire_sessions import SessionEnd, SessionStart
from treekle import from_plain

from knowop.agent.truncate import bounded


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from concurrent.futures import Future
    from datetime import datetime

    from trackinizer.runs.adapters.custom_types import Subagent
    from trackinizer.runs.redact import Redactor
    from trackinizer.sagent.session.types import SessionRecord
    from trackinizer.wire.wire_session_ir import PartBody
    from trackinizer.wire.wire_sessions import SessionStartResponse


__all__ = ["FORMATS", "Session", "SessionClient", "Uploaded", "plan", "upload"]


FORMATS: Final = ("claude", "codex", "sagent")
"""The formats a session file is uploaded from, as a session's ``cli`` names them."""

_READERS: Final = {
    "claude": claude.normalize,
    "codex": codex.normalize,
    "sagent": sagent.normalize,
}
"""Each format's session reader."""

_STANDING: Final = frozenset({"persistent_agent", "watcher"})
"""The records a sagent journal stands for each agent it spawned, naming its journal."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Session:
    """One session file a plan sends, and its place in the tree of sessions it spawned."""

    format: str
    """Its CLI: ``claude``, ``codex`` or ``sagent``; the AgentSession's ``cli``."""

    id: str
    """The CLI's own id for it; the AgentSession's ``cli_session_id``."""

    path: Path
    spawned_by: str = ""
    """The CLI's id for the session that spawned it; ``""`` for a session no other
    spawned."""

    note: str = ""
    """What spawned it, as the CLI records it: the edge to its spawner says it."""

    linked: bool = False
    """Whether the session that spawned it is in the plan, so is linked to; one whose
    file is not on disk goes up unlinked."""

    subagents: int = 0
    """How many sessions of the plan it spawned."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Uploaded:
    """What became of one session: its AgentSession, or why it did not go up."""

    session: Session
    agent_session: UUID | None = None
    """The AgentSession it is in; ``None`` when it failed before one was opened."""

    sent: int = 0
    """Records sent, stand-ins for the refused ones included."""

    held: int = 0
    """Records the server held already, which were not sent again."""

    refused: int = 0
    """Records the server refused, each sent as a stand-in saying so."""

    audience: str = ""
    """Who reads it, as the server holds it."""

    error: str = ""
    """Why it failed; ``""`` when it went up."""


class SessionClient(RecordAppender, Protocol):
    """The ``Client`` calls an upload makes."""

    def session_start(self, body: SessionStart) -> SessionStartResponse:
        """Open the session, or re-open the one the caller holds under its CLI id."""
        ...

    def read_session_parts(self, session_id: UUID) -> list[PartBody]:
        """Return the parts the server holds of the session."""
        ...

    def session_end(self, session_id: UUID, body: SessionEnd | None = None) -> object:
        """End the session."""
        ...

    def add_edge(
        self,
        from_id: UUID,
        to_id: UUID,
        edge_kind: str,
        *,
        actor: str,
        note: str = "",
    ) -> object:
        """Add an edge, or annotate the one there."""
        ...

    def close(self) -> None:
        """Release the connection."""
        ...


def plan(
    paths: Sequence[Path],
    *,
    read_as: str = "",
    subagents: bool = True,
) -> list[Session]:
    """Return the sessions ``paths`` name, each spawner before the sessions it spawned.

    A folder is searched for session files, nesting included; a file that is no
    session of the three formats is passed over. A session's whole tree goes: a file
    of a sub-agent stands for the session at its root, and a session for its
    sub-agents, found by where the CLI keeps them beside its file.

    Args:
      paths: Session files, or folders holding them, such as ``~/.claude``.
      read_as: The format to read every file as; ``""`` reads it from each file.
      subagents: Whether sub-agents go too; without them, only sessions that link to
        no other in the plan.

    Returns:
      sessions: Each once, by its file, a tree's root first and its sub-agents after.

    """
    layouts: dict[str, _Layout] = {
        "claude": _Claude(),
        "codex": _Codex(),
        "sagent": _Sagent(),
    }
    members: dict[Path, str] = {}
    for path in paths:
        for file in transcripts(path) if path.is_dir() else [path]:
            found = read_as or sniff(file)
            # A sagent session is its folder's ``session.jsonl``; the folder also keeps
            # an old sagent's log, whole, beside it, which is no session of its own.
            if found in FORMATS and (found != "sagent" or file.name == "session.jsonl"):
                for member in layouts[found].family(file):
                    members.setdefault(member, found)
    placed = [layouts[found].placed(file, found) for file, found in members.items()]
    held = {(session.format, session.id) for session in placed}
    linked = [
        replace(session, linked=(session.format, session.spawned_by) in held)
        for session in placed
    ]
    if not subagents:
        linked = [session for session in linked if not session.linked]
    return _ordered(linked)


def upload(
    sessions: Sequence[Session],
    *,
    connect: Callable[[], SessionClient],
    private: bool,
    jobs: int,
    redactor: Redactor | None,
) -> list[Uploaded]:
    """Send each session as an AgentSession, ``jobs`` at a time; return what became of each.

    A sub-agent goes after the session that spawned it, into its audience, and is
    linked to it; one whose spawner failed is not sent. Two files under one id go one
    after the other, into one AgentSession. A session the server holds keeps the
    audience it has. A failure is the session's alone: the others go on.

    Args:
      sessions: What :func:`plan` returned.
      connect: Makes a client; each session's worker makes its own. Picklable, as
        ``partial(Client, url, api_key=key)`` is, since workers are processes.
      private: Whether a new session that spawned none opens ``private`` rather
        than ``open``.
      jobs: How many sessions go up side by side.
      redactor: Masks secret values in every record; ``None`` keeps them.

    Returns:
      results: One per session, in the order of ``sessions``.

    """
    pool = _executor(max(1, min(jobs, len(sessions))))
    batch = _Batch(
        sessions,
        pool=pool,
        send=partial(_send_session, connect=connect, redactor=redactor),
        private=private,
    )
    try:
        for session in sessions:
            if not session.linked:
                batch.start(session, spawner=None)
        while batch.running:
            finished, _ = wait(batch.running, return_when=FIRST_COMPLETED)
            for future in finished:
                batch.finish(future)
    except BaseException:
        pool.shutdown(cancel_futures=True)
        raise
    pool.shutdown()
    return [batch.done[session] for session in sessions]


# One worker gains nothing from a process: it pays the start of an interpreter that
# imports the readers again, and its client would only cross a pipe.
def _executor(jobs: int) -> Executor:
    """Return the pool sessions go up in: a thread for one, else ``jobs`` processes."""
    if jobs == 1:
        return ThreadPoolExecutor(max_workers=1)
    return ProcessPoolExecutor(
        max_workers=jobs,
        initializer=die_with_parent,
        initargs=(os.getpid(), 1.0),
    )


class _Batch:
    """The sessions of one upload on their way: which run, which wait, what became of each.

    A session goes once the one that spawned it is up, and never beside another under
    its own id. Two files carry one id where a fork copied a sub-agent's transcript, or
    a session went on in another folder; the server opens a session at each start of an
    id it does not hold, so two at once would open two.

    Args:
      sessions: What :func:`plan` returned.
      pool: Where a session's upload runs.
      send: Uploads one session; :func:`_send_session` with its client and redactor.
      private: Whether a new session that spawned none opens ``private``.

    """

    def __init__(
        self,
        sessions: Sequence[Session],
        *,
        pool: Executor,
        send: Callable[..., Uploaded],
        private: bool,
    ) -> None:
        self._pool = pool
        self._send = send
        self._private = private
        self._spawned: dict[tuple[str, str], list[Session]] = {}
        for session in sessions:
            if session.linked:
                self._spawned.setdefault(
                    (session.format, session.spawned_by),
                    [],
                ).append(session)
        # Each with the session that spawned it, behind the one out under its id.
        self._waiting = defaultdict[
            tuple[str, str],
            list[tuple[Session, Uploaded | None]],
        ](list)
        self._out: set[tuple[str, str]] = set()
        self.running: dict[Future[Uploaded], Session] = {}
        """Each upload on its way, to its session."""

        self.done: dict[Session, Uploaded] = {}
        """What became of each session so far."""

    def start(self, session: Session, *, spawner: Uploaded | None) -> None:
        """Send ``session``, or queue it behind the one out under its id.

        Args:
          session: The session to send.
          spawner: What became of the session that spawned it, which it is linked to
            and whose audience it takes; ``None`` for a session that links to none.

        """
        key = (session.format, session.id)
        if key in self._out:
            self._waiting[key].append((session, spawner))
            return
        self._out.add(key)
        future = self._pool.submit(
            self._send,
            session,
            private=self._private if spawner is None else spawner.audience == PRIVATE,
            parent=None if spawner is None else spawner.agent_session,
        )
        self.running[future] = session

    def finish(self, future: Future[Uploaded]) -> None:
        """Record a finished upload; start what waited on its id or on its session.

        Args:
          future: One of :attr:`running`, done.

        """
        session = self.running.pop(future)
        key = (session.format, session.id)
        self._out.discard(key)
        if behind := self._waiting[key]:
            waiting, spawner = behind.pop(0)
            self.start(waiting, spawner=spawner)
        self._settle(future.result())

    # The sessions an id spawned go once, after the first file under it.
    def _settle(self, result: Uploaded) -> None:
        """Record ``result``; send what its session spawned, or record it as not sent."""
        settled = [result]
        while settled:
            result = settled.pop()
            self.done[result.session] = result
            key = (result.session.format, result.session.id)
            for child in self._spawned.pop(key, []):
                if result.agent_session is None:
                    settled.append(
                        Uploaded(
                            session=child,
                            error="not sent: the session that spawned it failed",
                        ),
                    )
                else:
                    self.start(child, spawner=result)


# A failure is reported against its session, and the batch goes on: a file that fails
# to read, a server that refuses the session.
def _send_session(
    session: Session,
    *,
    connect: Callable[[], SessionClient],
    private: bool,
    parent: UUID | None,
    redactor: Redactor | None,
) -> Uploaded:
    """Send ``session`` with a client of its own; return what became of it."""
    client = connect()
    try:
        return _sent(client, session, private=private, parent=parent, redactor=redactor)
    except (ClientError, OSError, TypeError, ValueError) as error:
        return Uploaded(session=session, error=str(error) or type(error).__name__)
    finally:
        client.close()


# Started at its first record's time and ended at its last, so the session sits where it
# happened. Ended even when sending failed: a start re-opens an ended session, and a
# second upload re-opens it again to send the rest.
def _sent(
    client: SessionClient,
    session: Session,
    *,
    private: bool,
    parent: UUID | None,
    redactor: Redactor | None,
) -> Uploaded:
    """Send ``session``'s records the server lacks, then link it to ``parent``."""
    with open_transcript(session.path, errors="replace") as handle:
        read = _Read(_READERS[session.format](handle))
        opened = client.session_start(
            SessionStart(
                cli=session.format,
                cli_session_id=session.id,
                # Its own, so no live session holds it and the server need not rename it.
                actor=f"{session.format}/{session.id}",
                started=read.started(),
                private=private,
                # The audience is for a session this start mints: one sent before, or
                # recorded live, stays with the readers it has.
                keep_audience=True,
            ),
        )
        try:
            sent, held, refused = _records(
                client,
                session,
                opened=opened,
                private=private,
                read=read,
                redactor=redactor,
            )
        finally:
            _ = client.session_end(opened.id, SessionEnd(ended=read.ended()))
    if parent is not None:
        _ = client.add_edge(
            opened.id,
            parent,
            "produced_by",
            actor=session.format,
            note=session.note,
        )
    return Uploaded(
        session=session,
        agent_session=opened.id,
        sent=sent,
        held=held,
        refused=refused,
        audience=opened.audience or "",
    )


# A server that predates hiding a session ignores ``private``, opens it to everyone and
# names no audience in its answer: not a record goes there. Any audience it does name is
# one the session had before this upload, which keeps it.
def _records(
    client: SessionClient,
    session: Session,
    *,
    opened: SessionStartResponse,
    private: bool,
    read: _Read,
    redactor: Redactor | None,
) -> tuple[int, int, int]:
    """Send the records of ``session``'s part the server lacks; return sent, held, refused."""
    if private and opened.audience is None:
        raise ClientError("the server predates hiding a session; nothing was sent")
    name = _part_name(session.path)
    part = next(
        (each for each in client.read_session_parts(opened.id) if each.name == name),
        None,
    )
    held = 0 if part is None else part.records
    # The id the part was stored under, kept: each append's manifest replaces it.
    ir_id = part.ir_id if part is not None and part.ir_id is not None else uuid4()
    sent = refused = 0
    for start, batch in read.batches(after=held, size=MAX_RECORD_BATCH):
        refused += len(
            send_batches(
                record_bodies(batch, start=start, redactor=redactor, bounded=bounded),
                send=partial(
                    append_batch,
                    client,
                    session=opened.id,
                    name=name,
                    ir_id=ir_id,
                    encoding=read.encoding,
                    format=session.format,
                ),
            ),
        )
        sent += len(batch)
    return sent, held, refused


# The server keeps one part per file NAME in a session, and claude's and codex's files
# are named for their session. sagent names every journal ``session.jsonl``, in a folder
# named for its session, so its part takes its folder's name, as a recorded run's did. A
# rollout codex compressed keeps the name it had, so its records land where they were.
def _part_name(path: Path) -> str:
    """Return the name the server keeps ``path``'s records under."""
    name = transcript_name(path)
    return f"{path.parent.name}.jsonl" if name == "session.jsonl" else name


class _Read:
    """One file's records, read once, with the times they state and the file's encoding.

    :meth:`started`, then :meth:`batches`, then :meth:`ended`.

    Args:
      records: The file's records, as its format's reader yields them.

    """

    def __init__(self, records: Iterator[SessionRecord]) -> None:
        self._records = records
        self._ahead: list[SessionRecord] = []
        self._last: datetime | None = None
        self.encoding: dict[str, object] = {}
        """How the file spells its bytes, as far as it has been read (``Tail.encoding``)."""

    # Read only as far as the first dated record, which is then the last read too.
    def started(self) -> datetime | None:
        """Return the first time a record states; ``None`` when none states one."""
        while self._last is None and (record := next(self._records, None)) is not None:
            self._ahead.append(self._seen(record))
        return self._last

    def ended(self) -> datetime | None:
        """Return the last time a record read so far states."""
        return self._last

    def batches(
        self,
        *,
        after: int,
        size: int,
    ) -> Iterator[tuple[int, list[SessionRecord]]]:
        """Yield the records from position ``after``, ``size`` at a time, each run's start.

        Reads the file to its end, so :meth:`ended` is its last record's time.

        Args:
          after: The position of the first record to yield; those before it are read
            and passed over.
          size: The most records one run holds.

        Yields:
          start: The position of the run's first record.
          batch: The run of records.

        """
        start = after
        records = chain(self._ahead, map(self._seen, self._records))
        for batch in batched(islice(records, after, None), size):
            yield start, list(batch)
            start += len(batch)

    def _seen(self, record: SessionRecord) -> SessionRecord:
        """Note the time ``record`` states, and the encoding a ``TurnContext`` restates."""
        if (time := record_time(record)) is not None:
            self._last = time
        if isinstance(record, TurnContext) and record.encoding:
            self.encoding = dict(record.encoding)
        return record


class _Layout(Protocol):
    """Where one CLI keeps a session's sub-agents, and how it names them."""

    def family(self, path: Path) -> list[Path]:
        """Return the files of every session in ``path``'s tree on disk, the root first."""
        ...

    def placed(self, path: Path, found: str) -> Session:
        """Return the session ``path`` is, of format ``found``, and what spawned it."""
        ...


class _Claude:
    """Claude Code: ``<session>.jsonl``, its sub-agents under ``<session>/subagents/``.

    The agents a workflow ran sit a folder deeper, ``subagents/workflows/<workflow>/``,
    beside the workflow's journal, which is no transcript.
    """

    def __init__(self) -> None:
        self._adapter = ClaudeAdapter()

    def family(self, path: Path) -> list[Path]:
        """Return the session at the root of ``path``'s tree and every sub-agent of it.

        Args:
          path: A session's file, or a file among its sub-agents'.

        Returns:
          files: The root's file, when it is on disk, then each sub-agent's; none for
            a file among a session's sub-agents' that is no transcript.

        """
        folder = self._adapter.session_folder(path)
        if folder is None:
            folder = path.parent / path.stem
        elif self._adapter.subagent(path) is None:
            return []
        root = folder.parent / f"{folder.name}.jsonl"
        agents = sorted(
            agent
            for agent in (folder / "subagents").rglob("agent-*.jsonl")
            if self._adapter.subagent(agent) is not None
        )
        return [root, *agents] if root.is_file() else agents

    # Claude's id for a sub-agent is its own only within its session: a fork carries its
    # sub-agents' transcripts under their ids and goes on with them its own way, and an
    # agent of a workflow that a message continued has a second transcript under its id
    # among the session's own sub-agents. Two files under one id would be one
    # AgentSession's one part, so a sub-agent is named from its session down.
    def placed(self, path: Path, found: str) -> Session:
        """Return the session ``path`` is, and the session that spawned it.

        Args:
          path: A session's file, or a sub-agent's.
          found: Its format.

        Returns:
          session: Named by its file's stem; a sub-agent's after its session's id, and
            its workflow's when one ran it.

        """
        spawn = self._adapter.subagent(path)
        folder = self._adapter.session_folder(path)
        if spawn is None or folder is None:
            return Session(format=found, id=path.stem, path=path)
        within = "" if path.parent.name == "subagents" else f"{path.parent.name}/"
        at_root = spawn.parent == folder.name
        return Session(
            format=found,
            id=f"{folder.name}/{within}{path.stem}",
            path=path,
            spawned_by=spawn.parent if at_root else f"{folder.name}/{spawn.parent}",
            note=spawn.note,
        )


def _spawned(session: Session, *, by: Subagent | None) -> Session:
    """Return ``session`` naming the session that spawned it, when one did."""
    if by is None:
        return session
    return replace(session, spawned_by=by.parent, note=by.note)


class _Codex:
    """Codex: each spawned thread a rollout of its own, naming its spawner's thread.

    A rollout is found by its id in its own folder, or in the dated folders it sits
    in (``sessions/<YYYY>/<MM>/<DD>/``), each folder's rollouts read once.
    """

    def __init__(self) -> None:
        self._adapter = CodexAdapter()
        # By folder searched, each thread's rollout and what spawned it, by its id.
        self._folders: dict[Path, dict[str, tuple[Path, Subagent | None]]] = {}

    def family(self, path: Path) -> list[Path]:
        """Return the rollouts of every thread in ``path``'s tree, the root's first.

        Args:
          path: A rollout.

        Returns:
          files: The rollout of the thread at the root, as far up as rollouts are
            found, then each thread it spawned, tier by tier.

        """
        threads = self._threads(path)
        root, _ = self._named(path)
        seen = {root}
        while (
            (spawn := threads[root][1]) is not None
            and spawn.parent in threads
            and spawn.parent not in seen
        ):
            root = spawn.parent
            seen.add(root)
        tree: list[str] = []
        frontier = [root]
        while frontier:
            tree += frontier
            frontier = [
                thread
                for thread, (_, spawn) in threads.items()
                if spawn is not None and spawn.parent in frontier and thread not in tree
            ]
        return [threads[thread][0] for thread in tree]

    def placed(self, path: Path, found: str) -> Session:
        """Return the thread ``path`` is the rollout of, and the thread that spawned it.

        Args:
          path: A rollout.
          found: Its format.

        Returns:
          session: Named by its thread's id: the one its first line spawns, else its
            file's name's, else its file's stem.

        """
        thread, spawn = self._named(path)
        return _spawned(Session(format=found, id=thread, path=path), by=spawn)

    def _named(self, path: Path) -> tuple[str, Subagent | None]:
        """Return the id of the thread ``path`` is the rollout of, and what spawned it."""
        spawn = self._adapter.subagent(path)
        if spawn is not None:
            return spawn.child, spawn
        return (
            self._adapter.session_id_from_path(path)
            or Path(transcript_name(path)).stem,
            None,
        )

    # ``path`` too: a rollout named otherwise is one its folder's pattern misses.
    def _threads(self, path: Path) -> dict[str, tuple[Path, Subagent | None]]:
        """Return every thread beside ``path`` by its id, ``path``'s own among them."""
        dated = [path.parent, path.parent.parent, path.parent.parent.parent]
        if all(re.fullmatch(r"[0-9]+", folder.name) for folder in dated):
            scope, pattern = dated[-1].parent, "*/*/*/rollout-*.jsonl"
        else:
            scope, pattern = path.parent, "rollout-*.jsonl"
        if scope not in self._folders:
            self._folders[scope] = {}
            # Codex compresses its older rollouts in place, ``.jsonl`` to ``.jsonl.zst``.
            for rollout in sorted(
                chain(scope.glob(pattern), scope.glob(f"{pattern}.zst")),
            ):
                thread, spawn = self._named(rollout)
                self._folders[scope].setdefault(thread, (rollout, spawn))
        thread, spawn = self._named(path)
        self._folders[scope].setdefault(thread, (path, spawn))
        return self._folders[scope]


class _Sagent:
    """sagent: ``<id>/session.jsonl``, each agent it spawned ``<id>/<uuid>/``, nested."""

    def __init__(self) -> None:
        self._labels: dict[Path, dict[str, str]] = {}

    def family(self, path: Path) -> list[Path]:
        """Return the journal at the root of ``path``'s tree and every one beneath it.

        Args:
          path: A session's journal, or a spawned agent's.

        Returns:
          files: The root session's journal, then each spawned agent's beneath it.

        """
        root = path
        while (up := _spawner_journal(root)) is not None:
            root = up
        beneath = sorted(
            journal
            for journal in root.parent.rglob("session.jsonl")
            if _spawner_journal(journal) is not None
        )
        return [root, *beneath]

    def placed(self, path: Path, found: str) -> Session:
        """Return the session ``path`` is, named by its folders from the root's down.

        Args:
          path: A journal.
          found: Its format.

        Returns:
          session: With the agent that spawned it, and the label that agent stood it
            under, if any.

        """
        spawner = _spawner_journal(path)
        session = Session(format=found, id=_sagent_id(path), path=path)
        if spawner is None:
            return session
        return replace(
            session,
            spawned_by=_sagent_id(spawner),
            note=self._label(spawner).get(path.parent.name, ""),
        )

    def _label(self, journal: Path) -> dict[str, str]:
        """Return each agent ``journal`` spawned, by its folder's name, to its label."""
        if journal not in self._labels:
            self._labels[journal] = _standing_labels(journal)
        return self._labels[journal]


# A spawned agent's folder sits in its spawner's, so a session's folder in a project
# folder that holds an old sagent's log is no spawned agent: sagent's own rule says so.
def _spawner_journal(path: Path) -> Path | None:
    """Return the journal of the agent that spawned ``path``'s, or ``None`` for a root."""
    up = path.parent.parent / "session.jsonl"
    return up if up.is_file() and not is_session(path) else None


# Its folders from the root session's down: a spawned agent's folder is a uuid today but
# was its label, and two sessions each spawn a ``watcher``.
def _sagent_id(path: Path) -> str:
    """Return the id of the session ``path`` is the journal of."""
    folders = [path.parent.name]
    here = path
    while (up := _spawner_journal(here)) is not None:
        folders.append(up.parent.name)
        here = up
    return "/".join(reversed(folders))


# A child stands in its spawner's journal as the record that starts it again: its label
# and its journal's path. Only those lines are parsed: a journal can be hundreds of MB.
def _standing_labels(journal: Path) -> dict[str, str]:
    """Return the label each child ``journal`` stood has, by its journal's folder."""
    labels: dict[str, str] = {}
    with journal.open("rb") as lines:
        for line in lines:
            if not any(f'"{kind}"'.encode() in line for kind in _STANDING):
                continue
            record = read_record(line)
            if isinstance(record, UncategorizedRecord) and record.kind in _STANDING:
                child = from_plain(
                    record.payload.get("journal"),
                    str,
                    default="",  # pragma: no mutate -- read for truth; None is as empty.
                )
                if child:
                    labels[PurePath(child).parent.name] = from_plain(
                        record.payload.get("label"),
                        str,
                        default="",
                    )
    return labels


def _ordered(sessions: Iterable[Session]) -> list[Session]:
    """Return ``sessions`` tree by tree, each session before those it spawned."""
    spawned: dict[tuple[str, str], list[Session]] = {}
    roots: list[Session] = []
    for session in sorted(sessions, key=lambda each: each.path):
        if session.linked:
            spawned.setdefault((session.format, session.spawned_by), []).append(session)
        else:
            roots.append(session)
    # Popped: where two files carry one id, what it spawned goes under the first.
    ordered: list[Session] = []
    pending = roots[::-1]
    while pending:
        session = pending.pop()
        children = spawned.pop((session.format, session.id), [])
        ordered.append(replace(session, subagents=len(children)))
        pending += children[::-1]
    return ordered
