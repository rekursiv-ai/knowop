"""What an agent process tells Trackinizer: the root agent's records, and its live lines.

One observer on the root agent's runtime. Each event goes up the session's socket as
it is published, and the journal goes up as records from each ``SaveSession``. What an
agent sends ``user`` goes up as its ``AgentToAgentMessage``, Chat being the user. The
agent's state is sagent's own status line, and what a slash line can name is sagent's
own settings registry, each sent when it changes. A secret named in
``TRAX_REDACT_NAMES`` is masked in each line as it is in each record; a reply is
masked piece by piece as it streams, so a value the model writes across two pieces
is masked only in its record.

::

    runtime event ──Publisher──▶ EventLine, StateLine, SettingsLine ──▶ the socket
    SaveSession   ──Publisher──▶ journal lines ──▶ POST .../records (a worker thread)

Typical usage example:

  publisher = Publisher(agent, client=client, session_id=session_id, part=part,
                        start=start, send=socket.send)
  agent.runtime.observers.append(publisher)
  commands = Commands(agent, ..., say=publisher.say)
  publisher.commands = commands           # The settings' source from now on.
"""

from __future__ import annotations

from datetime import UTC, datetime
from io import StringIO
from typing import TYPE_CHECKING, Final
from uuid import UUID

import asyncio
import logging
import os
import time

from sagent.runtime import types
from sagent.session.jsonl import record_line
from sagent.session.types import AgentToAgentMessage
from sagent.tui.status import Status
from trackinizer.client.errors import ClientError
from trackinizer.client.session_socket import retry_delays
from trackinizer.client.session_upload import (
    append_batch,
    record_bodies,
    send_batches,
)
from trackinizer.runs.redact import redactor_from_environ
from trackinizer.sagent.scripts.convert_session.sagent import normalize
from trackinizer.wire.wire_session_relay import (
    STALE_PART,
    AgentState,
    EventLine,
    Setting,
    SettingsLine,
    StateLine,
)

from knowop.agent.truncate import (
    bounded,
    bounded_payload,
    encoded_size,
)
from knowop.process.codec import event_payload


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from sagent.agent.agent import Agent
    from sagent.session.types import SessionRecord
    from sagent.tui.slash import Commands
    from sagent.tui.types import CommandOutput
    from trackinizer.client.session_upload import RecordAppender
    from trackinizer.wire.json_types import JSON
    from trackinizer.wire.wire_session_ir import RecordBody


__all__ = ["MAX_EVENT_BYTES", "Publisher"]


MAX_EVENT_BYTES: Final = 1 << 20
"""The most one event's line takes; its long strings are cut to fit, and a line still
over is not sent."""

_LINE_SEC: Final = 1.0
"""How often a state whose only change is its text goes up, at most."""

_GONE: Final = frozenset({401, 403, 404, 410})
"""What the server answers a key it refuses, or a session it no longer has."""

logger = logging.getLogger(__name__)


class Publisher:
    """One sagent observer: uploads the root agent's records and sends its live lines.

    Args:
      agent: The root agent, whose journal is the session's part.
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
        client: RecordAppender,
        session_id: UUID,
        part: str,
        start: int,
        send: Callable[[EventLine | StateLine | SettingsLine], None],
    ) -> None:
        self._agent = agent
        self._send = send
        self._status = Status(agent.runtime, spend=agent.spend_usd)
        self._commands: Commands | None = None
        # The settings last sent: first sent, and set, with ``commands``.
        self._settings: tuple[Setting, ...]
        self._next = start
        self._client = client
        self._session_id = session_id
        self._part = part
        self._redactor = redactor_from_environ(os.environ)
        self._uploading: asyncio.Task[None] | None = None
        self._saved = asyncio.Event()
        self.ended: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        """The exit code, once an upload's answer ends the process: 1 for a key or
        a session the server refused, 3 for a part another host continued."""

        self.reason = ""
        """Why :attr:`ended`, once it is."""

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
        inner = event
        while isinstance(inner, types.ChildEvent):
            inner = inner.inner
        if not isinstance(inner, types.SaveSession):
            self._event(event)
        elif inner is event:
            self.upload()
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

    def upload(self) -> None:
        """Upload the journal from where the last upload got to; one running reads it again."""
        self._saved.set()
        if self._uploading is None or self._uploading.done():
            self._uploading = asyncio.create_task(self._uploaded())

    async def caught_up(self) -> None:
        """Upload until the journal is caught up, or an answer ends the process."""
        self.upload()
        assert isinstance(self._uploading, asyncio.Task)
        await self._uploading

    async def aclose(self) -> None:
        """Stop the upload running, if one is."""
        if self._uploading is not None:
            self._uploading.cancel()
            await asyncio.gather(self._uploading, return_exceptions=True)

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

    def _check_state(self) -> None:
        """Send the state if ``busy`` or ``pending`` changed, or its text, once a second."""
        last = self._state
        state = self._now(since=last.since)
        if state.busy != last.busy:
            self._sent(state.model_copy(update={"since": datetime.now(UTC)}))
            self.check_settings()
        elif state.pending != last.pending or (
            state.line != last.line
            and time.monotonic() - self._state_sent_sec >= _LINE_SEC
        ):
            self._sent(state)

    # ``busy`` is whether sagent turns its spinner: a call, tool calls or a compaction.
    def _now(self, *, since: datetime) -> AgentState:
        """Return the agent's state now, ``busy`` last changed at ``since``."""
        line = self._status.line()
        return AgentState(
            line=line,
            busy=self._status.frame() != line,
            since=since,
            pending=tuple(
                self._plain(m) for m in self._agent.runtime.pending_mid_stream()
            ),
        )

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

    # Read on the runtime's loop, which writes the journal first; turned into lines and
    # sent on a worker thread. A pass reads the journal as it stood when the pass began,
    # so a save during it asks for one more: a record written meanwhile goes with this
    # upload, not with the next ``SaveSession``, which may be the person's next line.
    async def _uploaded(self) -> None:
        """Upload the journal until it is caught up, retrying an outage."""
        delays = retry_delays()
        while True:
            self._saved.clear()
            records = self._agent.session.history(self._next)
            pending = await asyncio.to_thread(self._bodies, records, start=self._next)
            if not pending:
                if not self._saved.is_set():
                    return
                continue
            count = len(pending)
            try:
                refused = await asyncio.to_thread(
                    send_batches,
                    pending,
                    send=self._appended,
                )
            except ClientError as error:
                self._next += count - len(pending)
                if self._ends(error):
                    return
                logger.warning("upload failed, trying again: %s", error)
                await asyncio.sleep(next(delays))
                continue
            for body in refused:
                logger.warning(
                    "the server refused record %d; a stand-in holds it",
                    body.idx,
                )
            self._next += count
            delays = retry_delays()

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

    def _appended(self, batch: list[RecordBody]) -> None:
        """Append one batch of the part's records."""
        append_batch(
            self._client,
            session=self._session_id,
            name=self._part,
            ir_id=UUID(hex=self._part.removesuffix(".jsonl")),
            encoding={},
            batch=batch,
            format="sagent",
        )

    def _ends(self, error: ClientError) -> bool:
        """End the process on an answer no retry changes; say whether it did."""
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


def _harvested(commands: Commands) -> tuple[Setting, ...]:
    """Return what a slash line can name now, as the wire carries it."""
    return tuple(
        Setting(
            name=s.name,
            help_short=s.help_short,
            help_long=s.help_long,
            usage=s.usage,
            value=None if s.value is None else s.value(),
            options=tuple(s.options()),
            flags=tuple(sorted(s.flags)),
        )
        for s in commands.reachable()
    )
