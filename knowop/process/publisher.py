"""What an agent process tells Trackinizer: each agent's records, and the root's lines.

One observer on the root agent's runtime. Each event goes up the session's socket as
it is published, and the root's journal goes up as records from each ``SaveSession``.
What an agent sends ``user`` goes up as its ``AgentToAgentMessage``, Chat being the
user. The agent's state is sagent's own status line, and what a slash line can name is
sagent's own settings registry, each sent when it changes; while the agent works, the
line is read again each second, as sagent's screen polls it. A secret named in
``TRAX_REDACT_NAMES`` is masked in each line as it is in each record; a reply is
masked piece by piece as it streams, so a value the model writes across two pieces
is masked only in its record.

Each child an agent spawns, a watcher among them, journals in a folder of its
spawner's (``child_journal``). Its journal goes up from its own ``SaveSession`` to a
session of its own, started at its first save ``spawned_by`` its spawner's session,
its label the link's note, named by its folders from the root's down
(``projects.session_id``), as ``knowop upload`` names it. A refused child is let go;
the root's session goes on.

::

    runtime event ──Publisher──▶ EventLine, StateLine, SettingsLine ──▶ the socket
    SaveSession   ──_Journal───▶ journal lines ──▶ POST .../records (a worker thread)
      (each agent's own: the root's to the session, a child's to the session its
       first save starts, spawned_by its spawner's)

Typical usage example:

  publisher = Publisher(agent, team=team, client=client, session_id=session_id,
                        part=part, start=start, send=socket.send)
  agent.runtime.observers.append(publisher)
  commands = Commands(agent, ..., say=publisher.say)
  publisher.commands = commands           # The settings' source from now on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from io import StringIO
from typing import TYPE_CHECKING, Final
from uuid import UUID

import asyncio
import logging
import os
import time

from sagent.runtime import types
from sagent.session import projects
from sagent.session.jsonl import record_line
from sagent.session.types import AgentToAgentMessage
from sagent.tools.background_task import without_controls
from sagent.tui.status import Status
from trackinizer.client.errors import ClientError
from trackinizer.client.session_socket import retry_delays
from trackinizer.client.session_upload import (
    PartUploader,
    append_batch,
    record_bodies,
    send_batches,
)
from trackinizer.runs.redact import redactor_from_environ
from trackinizer.sagent.scripts.convert_session.sagent import normalize
from trackinizer.wire.wire_session_relay import (
    STALE_PART,
    AgentState,
    BackgroundJob,
    EventLine,
    Setting,
    SettingsLine,
    StateLine,
    TeamMember,
)
from trackinizer.wire.wire_sessions import SessionStart

from knowop.agent.truncate import (
    bounded,
    bounded_payload,
    encoded_size,
)
from knowop.process.codec import event_payload


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable
    from pathlib import Path

    from sagent.agent.agent import Agent
    from sagent.agent.team import Team
    from sagent.session.types import SessionRecord, ToolCall
    from sagent.tools.types import Tool
    from sagent.tui.slash import Commands
    from sagent.tui.types import CommandOutput
    from trackinizer.runs.redact import Redactor
    from trackinizer.wire.json_types import JSON
    from trackinizer.wire.wire_session_ir import RecordBody


__all__ = ["MAX_EVENT_BYTES", "Publisher"]


MAX_EVENT_BYTES: Final = 1 << 20
"""The most one event's line takes; its long strings are cut to fit, and a line still
over is not sent."""

_LINE_SEC: Final = 1.0
"""How often a state whose only change is its text goes up: at most, and while the
agent works, at least."""

_GONE: Final = frozenset({401, 403, 404, 410})
"""What the server answers a key it refuses, or a session it no longer has."""

logger = logging.getLogger(__name__)


class Publisher:
    """One sagent observer: uploads each agent's records and sends the root's lines.

    Args:
      agent: The root agent, whose journal is the session's part.
      team: The session's agents; each child that joins with a journal goes up too.
      client: Where the records are stored.
      session_id: The session.
      part: The part's name, ``<folder>.jsonl``, the folder a uuid's hex.
      start: The journal's first position not yet stored.
      send: Sends one line up; it never blocks.

    """

    def __init__(
        self,
        agent: Agent,
        *,
        team: Team,
        client: PartUploader,
        session_id: UUID,
        part: str,
        start: int,
        send: Callable[[EventLine | StateLine | SettingsLine], None],
    ) -> None:
        self._agent = agent
        self._team = team
        self._session_id = session_id
        self._send = send
        self._status = Status(agent.runtime, spend=agent.spend_usd)
        self._commands: Commands | None = None
        # The settings last sent: first sent, and set, with ``commands``.
        self._settings: tuple[Setting, ...]
        self._client = client
        self._redactor = redactor_from_environ(os.environ)
        self.ended: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        """The exit code, once an upload's answer ends the process: 1 for a key or
        a session the server refused, 3 for a part another host continued."""

        self.reason = ""
        """Why :attr:`ended`, once it is."""

        self._root = _Journal(
            agent,
            client=client,
            redactor=self._redactor,
            opened=partial(_part, _Part(session_id=session_id, name=part, start=start)),
            ends=self._ends,
        )
        # By journal, so a child finds its spawner's session (``spawner_journal``). Only
        # the root's may be ``None``: a child with no journal sends nothing.
        self._journals: dict[Path | None, _Journal] = {
            agent.session.journal.path: self._root,
        }
        # The children whose events this publisher hears, by label.
        self._heard: dict[str, Agent] = {}
        # Whether the root called its model since its turn last ended.
        self._worked = False  # pragma: no mutate[truthiness]
        self._turn_ended: datetime | None = None
        # The next read of the state while busy; none once closed.
        self._tick: asyncio.TimerHandle | None = None
        self._open = True
        team.changes.add(self._joined)
        self._sent(self._now(since=datetime.now(UTC)))

    @property
    def commands(self) -> Commands | None:
        """The commands whose settings a slash line can name; ``None`` until assigned."""
        return self._commands

    @commands.setter
    def commands(self, commands: Commands) -> None:
        self._commands = commands
        self._settings = _harvested(commands)
        self._send(SettingsLine(settings=self._settings))

    def __call__(self, event: types.RuntimeEvent) -> None:
        """Send ``event`` up, or upload the journal at a ``SaveSession``; then the state."""
        self._status(event)
        self._turn(event)
        inner = event
        while isinstance(inner, types.ChildEvent):
            inner = inner.inner
        if not isinstance(inner, types.SaveSession):
            self._event(event)
        elif inner is event:
            self._root.upload()
        self._check_state()

    def say(self, output: CommandOutput) -> None:
        """Send what a slash command said up, then the settings if they changed."""
        self._event(output)
        self.check_settings()

    def to_user(self, sender: str, text: str) -> None:
        """Send up what ``sender`` told the user, as its message to ``user``."""
        self._event(AgentToAgentMessage(sender=sender, recipient="user", content=text))

    def check_settings(self) -> None:
        """Send the settings a slash line can name, if they differ from the last sent."""
        if self._commands is None:
            return
        settings = _harvested(self._commands)
        if settings != self._settings:
            self._settings = settings
            self._send(SettingsLine(settings=settings))

    # One at a time, in the order the agents joined: a spawner before its children,
    # whose upload waits for its session. Not gathered: a gather yields to the loop
    # even when nothing runs, and the agent would publish meanwhile.
    async def caught_up(self) -> None:
        """Upload each agent's journal until it is caught up, or an answer ends it."""
        for each in list(self._journals.values()):
            await each.caught_up()

    async def aclose(self) -> None:
        """Stop each upload running and the state's reads, and hear of no more agents."""
        self._open = False  # pragma: no mutate[truthiness]
        if self._tick is not None:
            self._tick.cancel()
        self._team.changes.discard(self._joined)
        for child in self._heard.values():
            child.runtime.observers.remove(self._child_heard)
        for each in self._journals.values():
            await each.aclose()

    def _event(self, event: object) -> None:
        """Send ``event`` up as a line of its class's name, cut to fit or not at all."""
        line = EventLine(
            kind=type(event).__name__,
            payload=bounded_payload(self._plain(event), max_bytes=MAX_EVENT_BYTES),
        )
        if encoded_size(line) > MAX_EVENT_BYTES:
            logger.warning("did not send a %s line too large to cut", line.kind)
            return
        self._send(line)

    # An idle with no model call since is no turn's end: a runtime idles as it starts.
    # A failure or a halt ends the turn it stops, whatever came before it.
    def _turn(self, event: types.RuntimeEvent) -> None:
        """Note when the root agent's turn ends."""
        if isinstance(event, types.ModelCallStarted):
            self._worked = True
        elif isinstance(
            event,
            (types.ModelResponseError, types.ModelResponseCancelled),
        ) or (isinstance(event, types.AgentIdle) and self._worked):
            self._worked = False
            self._turn_ended = datetime.now(UTC)

    # Its text, once a second.
    def _check_state(self) -> None:
        """Send the state at once when anything but its text changed."""
        last = self._state
        state = self._now(since=last.since)
        if state.busy != last.busy:
            self._sent(state.model_copy(update={"since": datetime.now(UTC)}))
            self.check_settings()
        elif state.model_copy(update={"line": last.line}) != last or (
            state.line != last.line
            and time.monotonic() - self._state_sent_sec >= _LINE_SEC
        ):
            self._sent(state)

    # ``busy`` is whether sagent turns its spinner: a call, tool calls or a compaction.
    def _now(self, *, since: datetime) -> AgentState:
        """Return the agents' state now, ``busy`` last changed at ``since``."""
        line = self._status.line()
        jobs, team = team_work(
            self._team,
            self._agent,
            mask=self._masked,
            sessions=self._session,
        )
        return AgentState(
            line=line,
            busy=self._status.frame() != line,
            since=since,
            pending=tuple(
                self._plain(m) for m in self._agent.runtime.pending_mid_stream()
            ),
            jobs=jobs,
            team=team,
            turn_ended=self._turn_ended,
        )

    def _session(self, agent: Agent) -> UUID | None:
        """Return the session ``agent``'s records go to: the root's, or a child's once opened."""
        if agent is self._agent:
            return self._session_id
        journal = self._journals.get(agent.session.journal.path)
        return None if journal is None else journal.stored_session

    def _masked(self, text: str) -> str:
        """Return ``text`` with each secret value masked."""
        return text if self._redactor is None else self._redactor.redact(text)

    # Masked first, then cut, as a record is: a cut could split a value.
    def _plain(self, said: object) -> JSON:
        """Return ``said`` as the plain data a viewer reads, each secret value masked."""
        payload = event_payload(said)
        if self._redactor is None:
            return payload
        return self._redactor.redact_mapping(payload)

    def _sent(self, state: AgentState) -> None:
        """Send ``state`` up, as the one a viewer now holds."""
        self._state = state
        self._state_sent_sec = time.monotonic()
        self._send(StateLine(state=state))
        self._arm()

    # A model call that streams nothing publishes nothing until it ends, so a line read
    # only at events stood still for minutes, its wait unnamed, as if the agent hung.
    def _arm(self) -> None:
        """Read the state again in ``_LINE_SEC`` while busy, unless a read is due."""
        if self._open and self._state.busy and self._tick is None:
            self._tick = asyncio.get_running_loop().call_later(_LINE_SEC, self._ticked)

    def _ticked(self) -> None:
        """Read the state, and arm the next read while it stays busy."""
        self._tick = None
        self._check_state()
        self._arm()

    # A child joins before its journal opens (``AgentSpawn._start``), so its first save
    # is heard; one with no journal, a far caller's, has nothing to send.
    def _joined(self) -> None:
        """Hear each child that joined the team at work, and one with a journal save it."""
        for label in self._team:
            agent = self._team[label]
            if agent is not self._agent and self._heard.get(label) is not agent:
                self._heard[label] = agent
                agent.runtime.observers.append(self._child_heard)
            path = agent.session.journal.path
            if path is None or path in self._journals:
                continue
            journal = _Journal(
                agent,
                client=self._client,
                redactor=self._redactor,
                opened=partial(self._child_part, path, label=label),
                ends=partial(self._let_go, label),
            )
            self._journals[path] = journal
            agent.runtime.observers.append(journal.saved)

    # A child's own runtime tells when it starts and stops work; its spawner hears only
    # some of its events (``agent_spawn._Forward``), and none as its model is called.
    def _child_heard(self, event: types.RuntimeEvent) -> None:
        """Send the state if a child's event changed it."""
        del event
        self._check_state()

    # Its spawner's session is opened first: a spawner saves before its model is called,
    # so before it spawns, but its start may still be on its way.
    async def _child_part(self, journal: Path, *, label: str) -> _Part:
        """Start the session of the child whose journal ``journal`` is; return its part."""
        above = projects.spawner_journal(journal)
        spawner = None if above is None else self._journals.get(above)
        if spawner is None:
            raise _NoSpawnerError(f"no journal of this session spawned {label!r}")
        spawned_by = await spawner.session()
        return await asyncio.to_thread(
            self._started,
            journal,
            label=label,
            spawned_by=spawned_by,
        )

    def _started(self, journal: Path, *, label: str, spawned_by: UUID) -> _Part:
        """Start the child's session, or re-open it; return where its upload begins."""
        cli_session_id = projects.session_id(journal)
        opened = self._client.session_start(
            SessionStart(
                cli="sagent",
                cli_session_id=cli_session_id,
                # Its own, so no live session holds it, as ``knowop upload`` names one.
                actor=f"sagent/{cli_session_id}",
                spawned_by=spawned_by,
                note=label,
            ),
        )
        name = f"{journal.parent.name}.jsonl"
        stored = next(
            (
                part.records
                for part in self._client.read_session_parts(opened.id)
                if part.name == name
            ),
            0,
        )
        return _Part(session_id=opened.id, name=name, start=stored)

    def _let_go(self, label: str, error: ClientError | _NoSpawnerError) -> bool:
        """Stop uploading ``label``'s journal on an answer no retry changes."""
        if isinstance(error, ClientError) and error.status_code not in _GONE:
            return False
        logger.warning("did not store agent %s: %s", label, error)
        return True

    def _ends(self, error: ClientError | _NoSpawnerError) -> bool:
        """End the process on an answer no retry changes; say whether it did."""
        assert isinstance(error, ClientError)
        if error.status_code == 409 and error.code == STALE_PART:
            self._end(3, "another host continued the session; this journal is kept")
            return True
        if error.status_code in _GONE:
            self._end(1, f"the server refused the upload: {error}")
            return True
        return False

    def _end(self, code: int, reason: str) -> None:
        """Resolve :attr:`ended`, unless it is already done."""
        if not self.ended.done():
            self.reason = reason
            self.ended.set_result(code)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Part:
    """Where one journal's records are stored."""

    session_id: UUID
    name: str
    """``<folder>.jsonl``, the journal's folder."""

    start: int
    """The journal's first position not yet stored."""


class _NoSpawnerError(Exception):
    """A journal of the team whose spawner's journal this process does not upload."""


def team_work(
    team: Team,
    root: Agent,
    *,
    mask: Callable[[str], str],
    sessions: Callable[[Agent], UUID | None],
) -> tuple[tuple[BackgroundJob, ...], tuple[TeamMember, ...]]:
    """Return what ``team``'s agents have going, as sagent's tree screen shows it.

    Args:
      team: The agents of the session.
      root: Its root agent, whose jobs come first.
      mask: What each job's title is shown through, its secrets masked.
      sessions: The session each agent's records are stored in, if one is yet.

    Returns:
      jobs: Each background job of every agent, the root's first.
      members: Each agent of the team: each root, then those it spawned, depth
        first, in the order they joined.

    """
    under: dict[str | None, list[str]] = {}
    for label in team:
        under.setdefault(team.parent(label), []).append(label)
    walked = _depth_first(under, None)
    jobs = (
        *_jobs(root, label="", mask=mask),
        *(
            job
            for label, _ in walked
            if team[label] is not root
            for job in _jobs(team[label], label=label, mask=mask)
        ),
    )
    return jobs, tuple(
        TeamMember(
            label=label,
            parent=parent or "",
            busy=_at_work(team[label]),
            spend_usd=team[label].spend_usd,
            session=sessions(team[label]),
        )
        for label, parent in walked
    )


def _depth_first(
    under: dict[str | None, list[str]],
    parent: str | None,
) -> list[tuple[str, str | None]]:
    """Return each agent ``parent`` spawned, and its parent, each before its own."""
    return [
        each
        for label in under.get(parent, [])
        for each in ((label, parent), *_depth_first(under, label))
    ]


# As sagent's tree screen lists them: a call the model waits on shows in its turn, a
# spawn's child is a member of its own, and a call that manages jobs is no job, one
# that waits for a job marking it awaited.
def _jobs(
    agent: Agent,
    *,
    label: str,
    mask: Callable[[str], str],
) -> list[BackgroundJob]:
    """Return ``agent``'s background jobs, each title through ``mask``."""
    runtime = agent.runtime
    held = runtime.jobs.values()
    awaited = {
        waited
        for job in held
        if job.call.name == "BackgroundTask"
        and job.call.arguments.get("operation") == "foreground"
        and isinstance(waited := job.call.arguments.get("id"), str)
    }
    return [
        BackgroundJob(
            call_id=job.call.call_id,
            tool=job.call.name,
            title=mask(_title(runtime.tools.get(job.call.name), job.call)),
            agent=label,
            started=datetime.fromtimestamp(job.started, UTC),
            state="sleeping" if job.state == "sleeping" else "running",
            awaited=job.call.call_id in awaited,
        )
        for job in held
        if job.state != "running"
        and job.call.name not in {"AgentSpawn", "BackgroundTask"}
    ]


def _title(tool: Tool | None, call: ToolCall) -> str:
    """Return the tool's own label for ``call``, or its name for a tool not held."""
    if tool is None:
        return call.name
    return tool.summary(without_controls(call.arguments))


def _at_work(agent: Agent) -> bool:
    """Return whether ``agent``'s model call, its tool calls or a compaction runs."""
    runtime = agent.runtime
    return bool(
        runtime.model_call is not None or runtime.cohort or runtime.compact_task,
    )


# ``model`` offers each model its catalog serves, as ``PROVIDER/MODEL``, which sagent's
# own setting leaves to its help: Chat's Model picks among them.
def _harvested(commands: Commands) -> tuple[Setting, ...]:
    """Return what a slash line can name now, as the wire carries it."""
    return tuple(
        Setting(
            name=s.name,
            help_short=s.help_short,
            help_long=s.help_long,
            usage=s.usage,
            value=None if s.value is None else s.value(),
            options=_models(commands) if s.name == ("model",) else tuple(s.options()),
            flags=tuple(sorted(s.flags)),
        )
        for s in commands.reachable()
    )


def _models(commands: Commands) -> tuple[str, ...]:
    """Return each model ``commands``' catalog serves, as ``PROVIDER/MODEL``."""
    catalog = commands.catalog
    return tuple(
        f"{provider}/{model}"
        for provider in catalog.factories
        for model in catalog[provider]
    )


async def _part(part: _Part) -> _Part:
    """Return ``part``: the root's, known before its first upload."""
    return part


class _Journal:
    """One agent's journal, uploaded as one part of its session from where it was stored.

    Args:
      agent: The agent whose journal it is.
      client: Where the records are stored.
      redactor: Masks secret values in each record; ``None`` keeps them.
      opened: Returns where its records are stored; awaited once, at the first upload.
      ends: Says whether an answer no retry changes ends this journal's upload.

    """

    def __init__(
        self,
        agent: Agent,
        *,
        client: PartUploader,
        redactor: Redactor | None,
        opened: Callable[[], Awaitable[_Part]],
        ends: Callable[[ClientError | _NoSpawnerError], bool],
    ) -> None:
        self._agent = agent
        self._client = client
        self._redactor = redactor
        self._opened = opened
        self._ends = ends
        self._part: _Part | None = None
        self._next = 0
        self._gone = False
        self._uploading: asyncio.Task[None] | None = None
        self._saved = asyncio.Event()
        # Set once its session is opened, or it is let go.
        self._settled = asyncio.Event()

    async def session(self) -> UUID:
        """Return its session once its first upload opens it.

        Returns:
          session_id: The session its records are stored in.

        Raises:
          _NoSpawnerError: It was let go before its session opened, so none will.

        """
        _ = await self._settled.wait()
        if self._part is None:
            raise _NoSpawnerError("the session of its spawner was not stored")
        return self._part.session_id

    @property
    def stored_session(self) -> UUID | None:
        """The session its records are stored in; ``None`` until its first upload opens it."""
        return None if self._part is None else self._part.session_id

    def saved(self, event: types.RuntimeEvent) -> None:
        """Upload at the agent's own ``SaveSession``."""
        if isinstance(event, types.SaveSession):
            self.upload()

    def upload(self) -> None:
        """Upload from where the last upload got to; one running reads it again."""
        self._saved.set()
        if not self._gone and (self._uploading is None or self._uploading.done()):
            self._uploading = asyncio.create_task(self._uploaded())

    async def caught_up(self) -> None:
        """Upload until the journal is caught up, or an answer ends it."""
        self.upload()
        if self._uploading is not None:
            await self._uploading

    async def aclose(self) -> None:
        """Stop the upload running, if one is."""
        if self._uploading is not None:
            self._uploading.cancel()
            await asyncio.gather(self._uploading, return_exceptions=True)

    # Read on the runtime's loop, which writes the journal first; turned into lines and
    # sent on a worker thread. A pass reads the journal as it stood when the pass began,
    # so a save during it asks for one more: a record written meanwhile goes with this
    # upload, not with the next ``SaveSession``, which may be the person's next line.
    async def _uploaded(self) -> None:
        """Open the session once, then upload until caught up, retrying an outage."""
        delays = retry_delays()
        while True:
            self._saved.clear()
            try:
                part = await self._part_opened()
                records = self._agent.session.history(self._next)
                pending = await asyncio.to_thread(
                    self._bodies,
                    records,
                    start=self._next,
                )
                if not pending:
                    if not self._saved.is_set():
                        return
                    continue
                count = len(pending)
                try:
                    refused = await asyncio.to_thread(
                        send_batches,
                        pending,
                        send=partial(self._appended, part),
                    )
                finally:
                    self._next += count - len(pending)
            except (ClientError, _NoSpawnerError) as error:
                if self._ends(error):
                    self._gone = True
                    self._settled.set()
                    return
                logger.warning("upload failed, trying again: %s", error)
                await asyncio.sleep(next(delays))
                continue
            for body in refused:
                logger.warning(
                    "the server refused record %d; a stand-in holds it",
                    body.idx,
                )
            delays = retry_delays()

    async def _part_opened(self) -> _Part:
        """Return where the records are stored, opening it at the first call."""
        if self._part is None:
            self._part = await self._opened()
            self._next = self._part.start
            self._settled.set()
        return self._part

    # One reader over all the lines: a line sagent's ``record_line`` wrote reads as
    # exactly one record, so a record's position is its line's.
    def _bodies(
        self,
        records: Iterable[SessionRecord],
        *,
        start: int,
    ) -> list[RecordBody]:
        """Return each record from ``start`` as the body its journal line reads back as."""
        lines = b"".join(record_line(record) for record in records)
        return record_bodies(
            normalize(StringIO(lines.decode())),
            start=start,
            redactor=self._redactor,
            bounded=bounded,
        )

    def _appended(self, part: _Part, batch: list[RecordBody]) -> None:
        """Append one batch of the part's records."""
        append_batch(
            self._client,
            session=part.session_id,
            name=part.name,
            ir_id=UUID(hex=part.name.removesuffix(".jsonl")),
            encoding={},
            batch=batch,
            format="sagent",
        )
