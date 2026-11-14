"""The relay to other sessions' agents: what it posts, what it tells, what it takes in.

Against a fake ``Client`` that records each post and answers as the test says, and a
real team of scripted agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

import asyncio
import threading

from sagent.agent.agent import Agent
from sagent.agent.team import MessageKind, Team
from sagent.model.scripted import ScriptedProvider
from sagent.providers.catalog import ModelCatalog
from sagent.runtime import types
from sagent.session.types import AgentToAgentMessage
from sagent.tools.types import ToolContext
from trackinizer.client.errors import ClientError
from trackinizer.wire.wire_sessions import SessionAgent

import pytest
import pytest_asyncio

from knowop.process.server_relay import ServerRelay


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path


_SESSION = UUID(int=415)
_LEAD = UUID(int=412)


@dataclass(slots=True, kw_only=True)
class _Posted:
    """One post the relay made."""

    to: SessionAgent
    text: str
    from_session: UUID | None
    from_label: str
    booked: MessageKind | None
    key: UUID | None


@dataclass(slots=True, kw_only=True)
class _Server:
    """``Client.message_agent``, answering each post as ``answers`` says, in turn."""

    answers: list[list[UUID] | ClientError] = field(
        default_factory=list[list[UUID] | ClientError],
    )
    posted: list[_Posted] = field(default_factory=list[_Posted])
    held: threading.Event | None = None
    """Set, a post waits for it first: a server that does not answer."""

    entered: threading.Event = field(default_factory=threading.Event)
    """Set once a post has begun."""

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
        self.entered.set()
        if self.held is not None:
            _ = self.held.wait()
        self.posted.append(
            _Posted(
                to=to,
                text=text,
                from_session=from_session,
                from_label=from_label,
                booked=booked,
                key=key,
            ),
        )
        answer = self.answers.pop(0) if self.answers else [_LEAD]
        if isinstance(answer, ClientError):
            raise answer
        return answer


def _team(tmp_path: Path, *labels: str) -> Team:
    """Return a team of ``Agent`` and a child of it per label."""
    team = Team(catalog=ModelCatalog(factories={"scripted": ScriptedProvider}))
    for label in ("Agent", *labels):
        _ = team.join(
            Agent(
                model=ScriptedProvider()["scripted"],
                tool_context=ToolContext(cwd=tmp_path),
            ),
            label=label,
            parent=None if label == "Agent" else "Agent",
        )
    return team


def _told(team: Team, label: str) -> list[tuple[str | None, str | None]]:
    """Return who sent ``label`` what, each note of the team's as from no one."""
    return [
        (m.sender, m.content)
        if isinstance(m, AgentToAgentMessage)
        else (None, m.content)
        for m in team[label].runtime.inbox.drain_nowait()
        if isinstance(m, (AgentToAgentMessage, types.Nudge))
    ]


def _not_delivered(why: str) -> tuple[None, str]:
    """Return the team's note that a send to ``lead#412`` did not arrive."""
    return (None, f"Your message to lead#412 was not delivered: {why}")


@pytest_asyncio.fixture
async def team(tmp_path: Path) -> AsyncGenerator[Team]:
    async with _team(tmp_path, "worker") as built:
        yield built


@pytest.mark.asyncio
async def test_a_send_is_posted_later_in_order_as_this_session_s_agent(
    team: Team,
) -> None:
    server = _Server()
    async with ServerRelay(
        server,
        session_id=_SESSION,
        team=team,
        root="Agent",
    ) as relay:
        team.relay = relay
        team.send("worker", "lead#412", "status?")
        # A label may hold ``#``: the session is what follows the last one.
        team.send("worker", "lead#a#413", "and now?")
        # Nothing is posted while the turn that sent goes on.
        assert server.posted == []

    assert [(p.to, p.text) for p in server.posted] == [
        (SessionAgent(label="lead", session=412), "status?"),
        (SessionAgent(label="lead#a", session=413), "and now?"),
    ]
    assert {(p.from_session, p.from_label, p.booked) for p in server.posted} == {
        (_SESSION, "worker", "ask"),
    }
    keys = [p.key for p in server.posted]
    assert None not in keys
    assert len(set(keys)) == 2
    assert _told(team, "worker") == []


@pytest.mark.parametrize(
    ("answer", "why"),
    [
        (
            ClientError("POST /api/messages failed: All connection attempts failed"),
            "POST /api/messages failed: All connection attempts failed",
        ),
        (
            ClientError("POST /api/messages -> 403: not_its_process", status_code=403),
            "POST /api/messages -> 403: not_its_process",
        ),
        (
            ClientError("POST /api/messages -> 409: rooms", status_code=409),
            "POST /api/messages -> 409: rooms",
        ),
        (
            ClientError("POST /api/messages -> 422: one target", status_code=422),
            "POST /api/messages -> 422: one target",
        ),
        ([], "no agent is connected to a live session lead#412 names"),
    ],
    ids=["out of reach", "403", "409", "422", "delivered to no one"],
)
@pytest.mark.asyncio
async def test_a_send_that_reaches_no_one_is_told_to_its_sender_with_why(
    team: Team,
    answer: list[UUID] | ClientError,
    why: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = _Server(answers=[answer])
    async with ServerRelay(
        server,
        session_id=_SESSION,
        team=team,
        root="Agent",
    ) as relay:
        team.relay = relay
        team.send("worker", "lead#412", "status?")

    assert _told(team, "worker") == [_not_delivered(why)]
    assert caplog.messages == [
        f"message from worker to lead#412 not delivered: {why}",
    ]


@pytest.mark.asyncio
async def test_a_label_that_names_no_session_is_refused_unposted(team: Team) -> None:
    server = _Server()
    async with ServerRelay(
        server,
        session_id=_SESSION,
        team=team,
        root="Agent",
    ) as relay:
        team.relay = relay
        team.send("worker", "lead#next", "status?")

    assert server.posted == []
    [(sender, said)] = _told(team, "worker")
    assert sender is None
    assert said is not None
    assert said.startswith(
        "Your message to lead#next was not delivered: 'lead#next' names no agent",
    )


@pytest.mark.asyncio
async def test_a_send_waiting_as_the_process_ends_is_sent_or_told_as_not_sent(
    team: Team,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held = threading.Event()
    server = _Server(held=held)
    monkeypatch.setattr(ServerRelay, "flush_sec", 0)
    relay = ServerRelay(server, session_id=_SESSION, team=team, root="Agent")
    async with relay:
        team.relay = relay
        team.send("worker", "lead#412", "first")
        team.send("Agent", "lead#412", "second")
        assert await asyncio.to_thread(server.entered.wait, 5)
    held.set()

    assert _told(team, "worker") == [
        _not_delivered(
            "this process ended before the server said whether it arrived",
        ),
    ]
    assert _told(team, "Agent") == [
        _not_delivered("this process ended before it was sent"),
    ]


@pytest.mark.asyncio
async def test_a_send_waiting_as_the_process_ends_is_posted_within_the_flush(
    team: Team,
) -> None:
    server = _Server()
    async with ServerRelay(
        server,
        session_id=_SESSION,
        team=team,
        root="Agent",
    ) as relay:
        team.relay = relay
        team.send("worker", "lead#412", "last words")

    assert [p.text for p in server.posted] == ["last words"]
    assert _told(team, "worker") == []


@pytest.mark.asyncio
async def test_a_relay_closed_unopened_tells_each_send_it_was_not_sent(
    team: Team,
) -> None:
    server = _Server()
    relay = ServerRelay(server, session_id=_SESSION, team=team, root="Agent")
    team.relay = relay
    team.send("worker", "lead#412", "never posted")

    async with asyncio.timeout(1):
        await relay.aclose()

    assert server.posted == []
    assert _told(team, "worker") == [
        _not_delivered("this process ended before it was sent"),
    ]


@pytest.mark.asyncio
async def test_a_refusal_for_one_who_left_is_told_no_one(team: Team) -> None:
    server = _Server(answers=[[]])
    async with ServerRelay(
        server,
        session_id=_SESSION,
        team=team,
        root="Agent",
    ) as relay:
        team.relay = relay
        relay.send(sender="gone", recipient="lead#412", content="hi", kind="tell")

    assert (_told(team, "Agent"), _told(team, "worker")) == ([], [])


def _from(
    sender: str,
    content: str | None,
    recipient: str | None = None,
) -> AgentToAgentMessage:
    return AgentToAgentMessage(sender=sender, recipient=recipient, content=content)


@pytest.mark.asyncio
async def test_a_message_from_another_session_reaches_the_agent_it_names_booked(
    team: Team,
) -> None:
    server = _Server()
    async with ServerRelay(
        server,
        session_id=_SESSION,
        team=team,
        root="Agent",
    ) as relay:
        team.relay = relay
        relay.deliver(_from("lead#412", "status?"), booked="ask")
        relay.deliver(_from("lead#412", "and you?", "worker"), booked="ask")
        relay.deliver(_from("lead#412", None, "worker"), booked="tell")
        assert team.owes("Agent") == {"lead#412": "status?"}
        assert team.owes("worker") == {"lead#412": "and you?"}
        # Its sender is reached from then on: the reply settles what was asked.
        team.send("worker", "lead#412", "green")

    assert _told(team, "Agent") == [("lead#412", "status?")]
    assert _told(team, "worker") == [("lead#412", "and you?"), ("lead#412", "")]
    assert team.owes("worker") == {}
    assert [(p.to, p.from_label, p.booked) for p in server.posted] == [
        (SessionAgent(label="lead", session=412), "worker", "reply"),
    ]


@pytest.mark.asyncio
async def test_another_session_s_agent_is_named_by_its_session_s_number_alone(
    team: Team,
) -> None:
    """The server signs each message ``label#number``: a debt booked by UUID never settles.

    Asked by ``lead#412``, an answer to ``lead#<UUID>`` would book a new request and
    leave the old one owed, with a nudge at each idle.
    """
    server = _Server()
    async with ServerRelay(
        server,
        session_id=_SESSION,
        team=team,
        root="Agent",
    ) as relay:
        team.relay = relay
        relay.deliver(_from("lead#412", "status?", "worker"), booked="ask")
        team.send("worker", f"lead#{_LEAD}", "green")

    assert server.posted == []
    assert team.owes("worker") == {"lead#412": "status?"}
    assert team.owes(f"lead#{_LEAD}") == {}
    assert _told(team, "worker") == [
        ("lead#412", "status?"),
        (
            None,
            (
                f"Your message to lead#{_LEAD} was not delivered: name another session's "
                "agent by its session's number, such as lead#412, as its messages are "
                "signed"
            ),
        ),
    ]


@pytest.mark.asyncio
async def test_a_reply_from_another_session_settles_what_was_asked(team: Team) -> None:
    server = _Server()
    async with ServerRelay(
        server,
        session_id=_SESSION,
        team=team,
        root="Agent",
    ) as relay:
        team.relay = relay
        team.send("worker", "lead#412", "status?")
        assert team.owes("lead#412") == {"worker": "status?"}
        relay.deliver(_from("lead#412", "green", "worker"), booked="reply")

    assert team.owes("lead#412") == {}
    assert team.owes("worker") == {}


@pytest.mark.parametrize("kind", ["ask", "reply"])
@pytest.mark.asyncio
async def test_a_message_for_no_agent_here_is_refused_back_once(
    team: Team,
    kind: MessageKind,
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = _Server()
    async with ServerRelay(
        server,
        session_id=_SESSION,
        team=team,
        root="Agent",
    ) as relay:
        team.relay = relay
        relay.deliver(_from("lead#412", "status?", "ghost"), booked=kind)

    [refusal] = server.posted
    assert (refusal.to, refusal.from_label, refusal.booked) == (
        SessionAgent(label="lead", session=412),
        "ghost",
        "tell",
    )
    unknown = "Unknown agent 'ghost'. Active: ['Agent', 'worker']"
    assert refusal.text == f"[message refused] {unknown}"
    assert caplog.messages == [f"message from lead#412 to 'ghost' refused: {unknown}"]


@pytest.mark.asyncio
async def test_a_refusal_that_cannot_be_taken_is_not_refused_back(team: Team) -> None:
    server = _Server()
    async with ServerRelay(
        server,
        session_id=_SESSION,
        team=team,
        root="Agent",
    ) as relay:
        team.relay = relay
        relay.deliver(
            _from("ghost#412", "[message refused] Unknown agent 'gone'", "gone"),
            booked="tell",
        )

    assert server.posted == []


@pytest.mark.asyncio
async def test_an_agent_with_peer_messaging_off_hears_no_one_and_says_so(
    team: Team,
) -> None:
    server = _Server()
    async with ServerRelay(
        server,
        session_id=_SESSION,
        team=team,
        root="Agent",
    ) as relay:
        team.relay = relay
        team.set_peer_messaging("Agent", False)
        relay.deliver(_from("lead#412", "status?"), booked="ask")

    assert _told(team, "Agent") == []
    assert team.owes("Agent") == {}
    [refusal] = server.posted
    assert refusal.text == (
        "[message refused] Cross-session peer messaging is disabled for this agent."
    )


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
