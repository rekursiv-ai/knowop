"""The relay to other sessions' agents: AgentSend through Trackinizer's ``/api/messages``.

An agent of another session is ``label#412``, the session by its number alone: the
server signs every message so, and the team books what is owed by the name, so a
debt booked under ``label#<UUID>`` would never be settled by the reply. A send to a
UUID is refused back to its sender, saying how to name it.
sagent's team sends each such label it holds no link for through its default relay
(``Team.relay``); in an agent process that relay is :class:`ServerRelay`. It posts each
message as ``trax send`` does, with the process's own key, naming its own session, the
sending agent's label and how the team booked it, so the server writes the sender as
``label#<this session's number>``. The other session's process takes the message as an
``Input``, which :meth:`ServerRelay.deliver` hands to the agent it names.

::

    AgentSend ─▶ Team ─▶ ServerRelay.send ─▶ POST /api/messages ─▶ Input
    Input (booked) ─▶ ServerRelay.deliver ─▶ Team.receive ─▶ the agent it names

A send never waits on the server: it is posted after the turn's tool call returns, one
at a time in the order sent, each under its own id as its ``Idempotency-Key``. A send
the server refuses, or that reaches no one, is told to its sender as a note of its
team's (``Team.refuse``), as sagent's own links tell one: never as a message from the
agent it was for, which never saw it. A send still waiting as the process ends is
posted within ``flush_sec``, or told as not sent.

Typical usage example:

  relay = ServerRelay(client, session_id=session_id, team=team, root="Agent")
  async with relay:
      team.relay = relay
      relay.deliver(message, booked=sent.booked)  # An agent's message from elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, override
from uuid import UUID, uuid4

import asyncio
import contextlib
import logging

from sagent.agent.team import MessageKind
from sagent.types import AsyncResource
from trackinizer.client.errors import ClientError
from trackinizer.wire.wire_sessions import SessionAgent


if TYPE_CHECKING:
    from sagent.agent.team import Team
    from sagent.session.types import AgentToAgentMessage


__all__ = ["AgentMessenger", "ServerRelay"]


logger = logging.getLogger(__name__)


class AgentMessenger(Protocol):
    """The call a relay posts each message with: a ``Client``'s own."""

    def message_agent(
        self,
        to: SessionAgent,
        text: str,
        *,
        from_session: UUID | None = None,
        from_label: str = "",
        booked: MessageKind | None = None,
        key: UUID | None = None,
    ) -> list[UUID]:
        """Send ``text`` to one agent of a session; return the sessions it went to."""
        ...


class ServerRelay(AsyncResource):
    """Carries this session's agents' messages to other sessions' agents, and back.

    Args:
      client: Posts each message, with the key this process serves its session with.
      session_id: This process's session, which each message names as its sender's.
      team: This session's agents.
      root: The root agent's label: the one a message naming no agent is for.

    """

    flush_sec: ClassVar[float] = 10.0
    """How long closing waits for the sends still waiting to be posted."""

    def __init__(
        self,
        client: AgentMessenger,
        *,
        session_id: UUID,
        team: Team,
        root: str,
    ) -> None:
        self._client = client
        self._session_id = session_id
        self._team = team
        self._root = root
        self._waiting: asyncio.Queue[_Note] = asyncio.Queue()
        self._posting: _Note | None = None
        self._task: asyncio.Task[None] | None = None
        """What posts each waiting send; ``None`` until opened."""

    @override
    async def aopen(self) -> None:
        self._task = asyncio.create_task(self._posted())

    # Sent, or told as not sent, before the process ends: never lost without a word.
    @override
    async def aclose(self) -> None:
        if self._task is not None:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(self.flush_sec):
                    await self._waiting.join()
            _ = self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._posting is not None:
            self._refused(
                self._posting,
                "this process ended before the server said whether it arrived",
            )
        while not self._waiting.empty():
            self._refused(
                self._waiting.get_nowait(),
                "this process ended before it was sent",
            )

    def send(
        self,
        *,
        sender: str,
        recipient: str,
        content: str,
        kind: MessageKind,
    ) -> None:
        """Queue ``content`` for ``recipient``, ``label#session``; it never waits.

        Args:
          sender: The agent here.
          recipient: The agent of another session.
          content: What it says.
          kind: How the team here booked it.

        """
        self._waiting.put_nowait(
            _Note(
                id=uuid4(),
                sender=sender,
                recipient=recipient,
                content=content,
                kind=kind,
            ),
        )

    def deliver(self, message: AgentToAgentMessage, *, booked: MessageKind) -> None:
        """Hand an agent's message from another session to the agent here it names.

        It is booked as its sender booked it: an ask is owed a reply, and a reply
        settles what was asked. One the team cannot take is refused back to its
        sender, unless it is news: a refusal is news, so none goes back and forth.

        Args:
          message: The message, from its sender as the server named it,
            ``label#number``; for the root when it names no agent.
          booked: How the sending agent's team booked it.

        """
        assert isinstance(message.sender, str)
        recipient = message.recipient or self._root
        try:
            self._team.receive(
                sender=message.sender,
                recipient=recipient,
                content=message.content or "",
                kind=booked,
            )
        except ValueError as error:
            logger.warning(
                "message from %s to %r refused: %s",
                message.sender,
                recipient,
                error,
            )
            if booked != "tell":
                self._team.tell(
                    recipient,
                    recipient=message.sender,
                    content=f"[message refused] {error}",
                )

    async def _posted(self) -> None:
        """Post each waiting message in turn, until cancelled."""
        while True:
            note = await self._waiting.get()
            # Left set when cancelled mid-post, so closing tells of it.
            self._posting = note
            await self._post(note)
            self._posting = None
            self._waiting.task_done()

    async def _post(self, note: _Note) -> None:
        """Post ``note``; tell its sender when it reached no one."""
        try:
            delivered = await asyncio.to_thread(
                self._client.message_agent,
                _numbered(note.recipient),
                note.content,
                from_session=self._session_id,
                from_label=note.sender,
                booked=note.kind,
                key=note.id,
            )
        # A pydantic validation error, as of a target that names no session, is a
        # ``ValueError``.
        except (ClientError, ValueError) as error:
            self._refused(note, str(error))
            return
        if not delivered:
            self._refused(
                note,
                f"no agent is connected to a live session {note.recipient} names",
            )

    def _refused(self, note: _Note, why: str) -> None:
        """Tell ``note``'s sender it was not delivered, as sagent's own links do."""
        logger.warning(
            "message from %s to %s not delivered: %s",
            note.sender,
            note.recipient,
            why,
        )
        self._team.refuse(note.sender, note.recipient, why)


def _numbered(recipient: str) -> SessionAgent:
    """Read ``label#412``; refuse a session named by its UUID, or by nothing."""
    agent = SessionAgent.parse(recipient)
    # Read off the words: a session ``parse`` took and that is no number is a UUID.
    if not recipient.rpartition("#")[2].isdecimal():
        raise ValueError(
            "name another session's agent by its session's number, such as "
            "lead#412, as its messages are signed",
        )
    return agent


@dataclass(frozen=True, slots=True, kw_only=True)
class _Note:
    """One message waiting to be posted."""

    id: UUID
    """Its own id, the post's ``Idempotency-Key``: a retried post is sent once."""

    sender: str
    recipient: str
    content: str
    kind: MessageKind
