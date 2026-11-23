"""The publisher: each runtime event up as a line, the journal up as records."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, Literal, cast
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import asyncio
import contextlib
import functools
import itertools
import threading
import time

from sagent.agent import team as team_module
from sagent.agent.agent import Agent
from sagent.agent.agents_md import AgentsMdConfig
from sagent.agent.team import Team
from sagent.journal.jsonl import JsonlJournal
from sagent.journal.types import (
    AgentIdle,
    AgentMessage,
    Cancelled,
    Failed,
    ModelIdle,
    Succeeded,
    ToolCall,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.model.exceptions import RetryableError
from sagent.model.scripted import ScriptedProvider, call, reply
from sagent.providers.catalog import ModelCatalog
from sagent.runtime import (
    runtime as runtime_module,
    types,
)
from sagent.settings.harvest import harvest
from sagent.tools.agent_spawn import AgentSpawn, _Forward
from sagent.tools.background_task import BackgroundTask
from sagent.tools.display import ToolDisplay
from sagent.tools.types import ToolContext, ToolOutput, whole_output
from sagent.tui import status as status_module
from sagent.tui.prompt import Panes
from sagent.tui.slash import Commands
from sagent.tui.types import CommandOutput
from trackinizer.client.errors import ClientError
from trackinizer.client.session_upload import send_batches
from trackinizer.wire.wire_session_ir import ManifestBody, PartBody, RecordBody
from trackinizer.wire.wire_session_relay import (
    STALE_PART,
    AgentState,
    BackgroundJob,
    EventLine,
    SettingsLine,
    StateLine,
    TeamMember,
)
from trackinizer.wire.wire_sessions import SessionStart, SessionStartResponse
from treekle import PlainTree

import pytest

from knowop.process import publisher as publisher_module
from knowop.process.codec import event_payload
from knowop.process.publisher import MAX_EVENT_BYTES, Publisher


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator, Sequence
    from pathlib import Path

    from sagent.agent.team import MessageKind
    from sagent.journal.types import SessionRecord
    from sagent.model.scripted import Reply
    from sagent.tools.types import Tool, ToolStreamingResult
    from trackinizer.wire.json_types import JSON


_SESSION: Final = UUID(int=9)
_FOLDER: Final = uuid4().hex
_PART: Final = f"{_FOLDER}.jsonl"


@dataclass(slots=True, kw_only=True)
class _Store:
    """Trackinizer as the publisher meets it: the root's records route, and a start
    and a parts read for each child's session, each keeping what it is sent or
    answering as told.
    """

    appended: list[tuple[str, ManifestBody, list[RecordBody]]] = field(
        default_factory=list[tuple[str, ManifestBody, list[RecordBody]]],
    )
    """The root session's appends."""

    failing: list[ClientError | None] = field(default_factory=list[ClientError | None])
    """What the next appends raise, in turn; ``None`` lets one through."""

    held: threading.Event | None = None
    """When set, an append waits for it."""

    refusing: str = ""
    """A kind of record the server cannot store; empty for none."""

    starts: list[SessionStart] = field(default_factory=list[SessionStart])
    """Each child's start, in order."""

    start_failing: list[ClientError | None] = field(
        default_factory=list[ClientError | None],
    )
    """What the next starts raise, in turn; ``None`` lets one through."""

    stored: int = 0
    """How many records the server holds of each child's part already."""

    children: dict[UUID, tuple[str, list[RecordBody]]] = field(
        default_factory=dict[UUID, tuple[str, list[RecordBody]]],
    )
    """Each child's session: its part's name, and the records appended to it."""

    child_appended: threading.Event = field(default_factory=threading.Event)
    """Set once any child's records are appended."""

    def session_start(self, body: SessionStart) -> SessionStartResponse:
        if self.start_failing and (error := self.start_failing.pop(0)) is not None:
            raise error
        self.starts.append(body)
        assert isinstance(body.cli_session_id, str)
        session_id = uuid5(NAMESPACE_URL, body.cli_session_id)
        part = f"{body.cli_session_id.rpartition('/')[2]}.jsonl"
        _ = self.children.setdefault(session_id, (part, []))
        return SessionStartResponse(id=session_id, seq=0)

    def read_session_parts(self, session_id: UUID) -> list[PartBody]:
        name, _ = self.children[session_id]
        if not self.stored:
            return []
        return [PartBody(part=0, name=name, format="sagent", records=self.stored)]

    def append_records(
        self,
        session_id: UUID,
        *,
        name: str,
        manifest: ManifestBody,
        records: Sequence[RecordBody],
    ) -> object:
        if session_id != _SESSION:
            assert (name, manifest.name) == (self.children[session_id][0],) * 2
            self.children[session_id][1].extend(records)
            self.child_appended.set()
            return None
        if self.held is not None:
            assert self.held.wait(5)
        if self.failing and (error := self.failing.pop(0)) is not None:
            raise error
        if any(r.kind == self.refusing for r in records):
            raise ClientError("cannot store it", status_code=422)
        self.appended.append((name, manifest, list(records)))
        return None

    def positions(self) -> list[int]:
        return [r.idx for _, _, batch in self.appended for r in batch]


@dataclass(slots=True, kw_only=True)
class _Screen:
    show_thinking: bool = False


@dataclass(slots=True, kw_only=True)
class _Published:
    agent: Agent
    team: Team
    publisher: Publisher
    store: _Store
    sent: list[EventLine | StateLine | SettingsLine]

    def events(self) -> list[EventLine]:
        return [line for line in self.sent if isinstance(line, EventLine)]

    def states(self) -> list[StateLine]:
        return [line for line in self.sent if isinstance(line, StateLine)]

    def settings(self) -> list[SettingsLine]:
        return [line for line in self.sent if isinstance(line, SettingsLine)]


# Given ``children``, it spawns, and they answer them in turn.
#
# Only a root that spawns joins the team: joining wakes it, which its other tests would
# see as one more ``AgentIdle``.
@contextlib.asynccontextmanager
async def _published(
    tmp_path: Path,
    *replies: Reply,
    start: int = 0,
    store: _Store | None = None,
    children: Sequence[Reply] = (),
    tools: Sequence[Tool] = (),
) -> AsyncGenerator[_Published]:
    """Publish a root agent; given ``children``, it spawns, and they answer them."""
    provider = ScriptedProvider(replies=replies)
    team = Team(
        catalog=ModelCatalog(
            factories={
                "scripted": functools.partial(ScriptedProvider, replies=children),
            },
        ),
    )
    journal = JsonlJournal(tmp_path / _FOLDER / "session.jsonl")
    agent = Agent(
        model=provider["scripted"],
        tools=[*tools, *([AgentSpawn(team=team)] if children else [])],
        journal=journal,
        tool_context=ToolContext(cwd=tmp_path),
        system_prompt="S",
        agents_md=AgentsMdConfig(
            system_dir=tmp_path / "system",
            user_dir=tmp_path / "user",
        ),
    )
    if children:
        _ = team.join(agent)
    sent: list[EventLine | StateLine | SettingsLine] = []
    async with journal, agent, team:
        store = store if store is not None else _Store()
        publisher = Publisher(
            agent,
            team=team,
            client=store,
            session_id=_SESSION,
            part=_PART,
            start=start,
            send=sent.append,
        )
        agent.runtime.observers.append(publisher)
        try:
            yield _Published(
                agent=agent,
                team=team,
                publisher=publisher,
                store=store,
                sent=sent,
            )
        finally:
            await publisher.aclose()


async def _told(p: _Published, event: types.RuntimeEvent) -> None:
    """Tell the publisher ``event``, then let it read the state once the loop turns."""
    p.publisher(event)
    await asyncio.sleep(0)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return what the publisher warned of, in order."""
    return [
        r.getMessage() for r in caplog.records if r.name == publisher_module.__name__
    ]


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Try a failed upload again at once; the delays are the socket module's to test."""
    monkeypatch.setattr(publisher_module, "retry_delays", lambda: itertools.repeat(0.0))


@pytest.mark.asyncio
async def test_a_turn_s_records_go_up_at_their_journal_positions(
    tmp_path: Path,
) -> None:
    async with _published(tmp_path, reply("hello")) as p:
        _ = await p.agent.send("hi")
        await p.publisher.caught_up()
        stored = list(p.agent.session.history())

    kinds = [r.kind for _, _, batch in p.store.appended for r in batch]
    assert p.store.positions() == list(range(len(stored)))
    assert kinds == [type(r).__name__ for r in stored]
    assert {"UserMessage", "AssistantMessage"} <= set(kinds)
    name, manifest, batch = p.store.appended[-1]
    assert (name, manifest.name, manifest.format) == (_PART, _PART, "sagent")
    assert (manifest.ir_id, manifest.records) == (UUID(hex=_FOLDER), batch[-1].idx + 1)


@pytest.mark.asyncio
async def test_an_upload_starts_where_the_part_was_stored_to(tmp_path: Path) -> None:
    async with _published(tmp_path, reply("one"), reply("two")) as first:
        _ = await first.agent.send("a")
        await first.publisher.caught_up()
        stored = len(list(first.agent.session.history()))
    async with _published(tmp_path, reply("two"), start=stored) as second:
        _ = await second.agent.send("b")
        await second.publisher.caught_up()

    assert second.store.positions()[0] == stored


@pytest.mark.asyncio
async def test_save_session_is_not_sent_and_each_other_event_is_by_its_name(
    tmp_path: Path,
) -> None:
    async with _published(tmp_path, reply("hello")) as p:
        _ = await p.agent.send("hi")
        await p.publisher.caught_up()

    kinds = [line.kind for line in p.events()]
    assert "SessionSaveDue" not in kinds
    assert {"ModelResponseStarted", "ModelResponseCompleted", "AgentIdle"} <= set(kinds)
    idle = next(line for line in p.events() if line.kind == "AgentIdle")
    assert idle.payload == event_payload(AgentIdle())


@pytest.mark.asyncio
async def test_a_child_s_save_session_is_not_sent_at_any_depth(tmp_path: Path) -> None:
    async with _published(tmp_path) as p:
        p.publisher(
            types.ChildEvent(
                label="a",
                event=types.ChildEvent(label="b", event=types.SessionSaveDue()),
            ),
        )
        p.publisher(types.ChildEvent(label="a", event=AgentIdle()))

    assert [line.kind for line in p.events()] == ["ChildEvent", "SessionQuitCompleted"]
    assert p.store.appended == []


# The root spawns ``kid``, who spawns ``grandkid``; each idles with what it says last.
_SPAWNS_KID: Final = (
    reply(calls=[call("AgentSpawn", "s1", prompt="dig", label="kid")]),
    reply("root done"),
)
_KID_SPAWNS_GRANDKID: Final = (
    reply(calls=[call("AgentSpawn", "s2", prompt="deeper", label="grandkid")]),
    reply("grandkid done"),
    reply("kid done"),
)


@pytest.mark.asyncio
async def test_each_child_goes_up_to_a_session_of_its_own_spawned_by_its_spawners(
    tmp_path: Path,
) -> None:
    async with _published(
        tmp_path,
        *_SPAWNS_KID,
        children=_KID_SPAWNS_GRANDKID,
    ) as p:
        _ = await p.agent.send("go")
        await p.publisher.caught_up()
        paths = {label: p.team[label].session.journal.path for label in p.team}
        histories = {label: list(p.team[label].session.history()) for label in p.team}

    kid, grandkid = paths["kid"], paths["grandkid"]
    assert kid is not None
    assert grandkid is not None
    kid_id = f"{_FOLDER}/{kid.parent.name}"
    grandkid_id = f"{kid_id}/{grandkid.parent.name}"
    assert [(s.cli_session_id, s.actor, s.note) for s in p.store.starts] == [
        (kid_id, f"sagent/{kid_id}", "kid"),
        (grandkid_id, f"sagent/{grandkid_id}", "grandkid"),
    ]
    assert [s.spawned_by for s in p.store.starts] == [
        _SESSION,
        uuid5(NAMESPACE_URL, kid_id),
    ]
    for label, cli_id in (("kid", kid_id), ("grandkid", grandkid_id)):
        _, stored = p.store.children[uuid5(NAMESPACE_URL, cli_id)]
        assert [r.idx for r in stored] == list(range(len(histories[label])))
    assert p.store.positions() == list(range(len(histories["Agent"])))


@pytest.mark.asyncio
async def test_a_child_s_upload_starts_where_its_part_was_stored_to(
    tmp_path: Path,
) -> None:
    store = _Store(stored=2)
    async with _published(
        tmp_path,
        *_SPAWNS_KID,
        children=_KID_SPAWNS_GRANDKID[1:],
        store=store,
    ) as p:
        _ = await p.agent.send("go")
        await p.publisher.caught_up()

    [(_, stored)] = store.children.values()
    assert stored[0].idx == 2


@pytest.mark.asyncio
async def test_an_outage_opening_a_child_s_session_is_tried_again(
    tmp_path: Path,
) -> None:
    store = _Store(start_failing=[ClientError("down", status_code=503)])
    async with _published(
        tmp_path,
        *_SPAWNS_KID,
        children=_KID_SPAWNS_GRANDKID[1:],
        store=store,
    ) as p:
        _ = await p.agent.send("go")
        await p.publisher.caught_up()

    assert [s.note for s in store.starts] == ["kid"]
    [(_, stored)] = store.children.values()
    assert stored


@pytest.mark.asyncio
async def test_a_child_the_server_refuses_is_let_go_and_the_root_goes_on(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _Store(start_failing=[ClientError("not yours", status_code=403)] * 9)
    async with _published(
        tmp_path,
        *_SPAWNS_KID,
        children=_KID_SPAWNS_GRANDKID[1:],
        store=store,
    ) as p:
        _ = await p.agent.send("go")
        await p.publisher.caught_up()
        stored = len(list(p.agent.session.history()))

    assert not p.publisher.ended.done()
    assert len(store.start_failing) == 8
    assert store.positions() == list(range(stored))
    assert any(
        "kid" in warned and "not yours" in warned for warned in _warnings(caplog)
    )


@pytest.mark.asyncio
async def test_a_child_of_a_refused_child_is_let_go_too(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Its spawner's session never opens, so it waits for none."""
    store = _Store(start_failing=[ClientError("not yours", status_code=403)])
    async with _published(
        tmp_path,
        *_SPAWNS_KID,
        children=_KID_SPAWNS_GRANDKID,
        store=store,
    ) as p:
        _ = await p.agent.send("go")
        await p.publisher.caught_up()

    assert store.starts == []
    assert any(
        "grandkid" in warned and "spawner was not stored" in warned
        for warned in _warnings(caplog)
    )


@pytest.mark.asyncio
async def test_an_agent_whose_spawner_this_session_does_not_hold_is_let_go(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with _published(tmp_path) as p:
        journal = JsonlJournal(tmp_path / "elsewhere" / str(uuid4()) / "session.jsonl")
        stray = Agent(
            model=ScriptedProvider()["scripted"],
            journal=journal,
            tool_context=ToolContext(cwd=tmp_path),
        )
        _ = p.team.join(stray, label="stray")
        async with journal, stray:
            stray.runtime.publish(types.SessionSaveDue())
            await p.publisher.caught_up()

    assert p.store.starts == []
    assert any(
        "stray" in warned and "no journal of this session" in warned
        for warned in _warnings(caplog)
    )


@pytest.mark.asyncio
async def test_an_agent_with_no_spawner_is_let_go_though_the_root_keeps_no_journal(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A root held in memory has no journal: no child's spawner is that missing one."""
    team = Team(catalog=ModelCatalog(factories={}))
    journal = JsonlJournal(tmp_path / "elsewhere" / str(uuid4()) / "session.jsonl")
    agent, stray = _scripted(tmp_path), _scripted(tmp_path, journal=journal)
    store = _Store()
    async with agent, team, journal, stray:
        _ = team.join(agent)
        publisher = Publisher(
            agent,
            team=team,
            client=store,
            session_id=_SESSION,
            part=_PART,
            start=0,
            send=lambda _: None,
        )
        try:
            _ = team.join(stray, label="stray")
            stray.runtime.publish(types.SessionSaveDue())
            await publisher.caught_up()
        finally:
            await publisher.aclose()

    assert store.starts == []
    assert any(
        "stray" in warned and "no journal of this session" in warned
        for warned in _warnings(caplog)
    )


@pytest.mark.asyncio
async def test_a_child_in_the_team_before_the_publisher_goes_up_too(
    tmp_path: Path,
) -> None:
    """A restarted process resumes its persistent children before it publishes."""
    team = Team(catalog=ModelCatalog(factories={}))
    journal = JsonlJournal(tmp_path / _FOLDER / "session.jsonl")
    folder = uuid4().hex
    kid_journal = JsonlJournal(tmp_path / _FOLDER / folder / "session.jsonl")
    agent = Agent(
        # What it answers its child's idle.
        model=ScriptedProvider(replies=[reply("noted")])["scripted"],
        journal=journal,
        tool_context=ToolContext(cwd=tmp_path),
    )
    kid = Agent(
        model=ScriptedProvider(replies=[reply("pong")])["scripted"],
        journal=kid_journal,
        tool_context=ToolContext(cwd=tmp_path),
    )
    store = _Store()
    async with journal, agent, team, kid_journal, kid:
        _ = team.join(agent)
        _ = team.join(kid, label="reviewer", parent="Agent")
        publisher = Publisher(
            agent,
            team=team,
            client=store,
            session_id=_SESSION,
            part=_PART,
            start=0,
            send=lambda _: None,
        )
        try:
            _ = await kid.send("ping")
            await publisher.caught_up()
            history = list(kid.session.history())
        finally:
            await publisher.aclose()

    kid_id = f"{_FOLDER}/{folder}"
    assert [(s.cli_session_id, s.spawned_by, s.note) for s in store.starts] == [
        (kid_id, _SESSION, "reviewer"),
    ]
    _, stored = store.children[uuid5(NAMESPACE_URL, kid_id)]
    assert [r.idx for r in stored] == list(range(len(history)))


@pytest.mark.asyncio
async def test_what_a_resumed_child_wrote_before_the_publisher_goes_up_with_no_new_turn(
    tmp_path: Path,
) -> None:
    """Its records go up as the publisher is built, not at its next save or the stop."""
    team = Team(catalog=ModelCatalog(factories={}))
    journal = JsonlJournal(tmp_path / _FOLDER / "session.jsonl")
    folder = uuid4().hex
    kid_journal = JsonlJournal(tmp_path / _FOLDER / folder / "session.jsonl")
    agent = Agent(
        # What it answers its child's idle.
        model=ScriptedProvider(replies=[reply("noted")])["scripted"],
        journal=journal,
        tool_context=ToolContext(cwd=tmp_path),
    )
    kid = Agent(
        model=ScriptedProvider(replies=[reply("pong")])["scripted"],
        journal=kid_journal,
        tool_context=ToolContext(cwd=tmp_path),
    )
    store = _Store()
    kid_session = uuid5(NAMESPACE_URL, f"{_FOLDER}/{folder}")
    async with journal, agent, team, kid_journal, kid:
        _ = team.join(agent)
        _ = team.join(kid, label="reviewer", parent="Agent")
        _ = await kid.send("ping")
        written = len(list(kid.session.history()))
        publisher = Publisher(
            agent,
            team=team,
            client=store,
            session_id=_SESSION,
            part=_PART,
            start=0,
            send=lambda _: None,
        )
        try:
            appended = await asyncio.to_thread(store.child_appended.wait, 1)
        finally:
            await publisher.aclose()

    assert appended
    _, stored = store.children[kid_session]
    assert [r.idx for r in stored] == list(range(written))


@pytest.mark.asyncio
async def test_what_an_agent_tells_the_user_goes_up_as_its_message(
    tmp_path: Path,
) -> None:
    async with _published(tmp_path) as p:
        p.publisher.to_user("pyq2", "in the repository")

    message = AgentMessage(
        sender="pyq2",
        recipient="user",
        content="in the repository",
    )
    assert p.events()[:-1] == [
        EventLine(kind="AgentMessage", payload=event_payload(message)),
    ]
    assert p.events()[-1].kind == "SessionQuitCompleted"


@pytest.mark.asyncio
async def test_a_command_s_output_goes_up_as_an_event(tmp_path: Path) -> None:
    output = CommandOutput(text="[/effort] high", tone="note", command="/effort")
    async with _published(tmp_path) as p:
        p.publisher.print_output(output)

    assert p.events()[:-1] == [
        EventLine(kind="CommandOutput", payload=event_payload(output)),
    ]
    assert p.events()[-1].kind == "SessionQuitCompleted"


@pytest.mark.asyncio
async def test_a_long_event_is_cut_to_fit_and_one_that_cannot_be_is_not_sent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = 64 << 10
    monkeypatch.setattr(publisher_module, "MAX_EVENT_BYTES", bound)
    async with _published(tmp_path) as p:
        p.publisher(types.ModelResponseTextStreamed(text="x" * 3 * bound))
        # Each string is cut to no less than a kibibyte, so these stay over.
        p.publisher(
            types.ModelResponseSuspended(
                exception=RetryableError("busy", retry_after_sec=1.0),
                retry_at=datetime.fromtimestamp(0.0, UTC),
                failures_in_a_row=1,
                provider="p",
                reset_at_by_login={
                    f"{n:04}" + "y" * 1_100: datetime.fromtimestamp(0.0, UTC)
                    for n in range(80)
                },
            ),
        )

    assert [line.kind for line in p.events()] == [
        "ModelResponseTextStreamed",
        "SessionQuitCompleted",
    ]
    assert len(p.events()[0].model_dump_json().encode()) <= bound
    assert MAX_EVENT_BYTES == 1 << 20


@pytest.mark.asyncio
async def test_a_state_is_sent_when_built_and_at_once_when_busy_changes(
    tmp_path: Path,
) -> None:
    async with _published(tmp_path) as p:
        built = list(p.states())
        await _told(p, types.ModelResponseStarted())
        started = p.states()[-1]
        await _told(p, types.ModelResponseCompleted(outcome=Cancelled()))
        # A halted call ends its turn too, as the runtime tells it.
        await _told(p, ModelIdle(outcome=Cancelled()))

    assert [s.state.busy for s in built] == [False]
    assert started.state.busy
    assert started.state.since > built[0].state.since
    assert [s.state.busy for s in p.states()] == [False, True, False]


# The runtime tells of a reply's end before the same step starts its calls, and of the
# calls' end before it calls the model again: read there, the agent seemed idle.
@pytest.mark.asyncio
async def test_a_turn_with_a_call_is_busy_throughout_from_its_reply_to_its_calls_and_back(
    tmp_path: Path,
) -> None:
    waiting = _Waiting()
    waiting.release.set()
    async with _published(
        tmp_path,
        reply(calls=[call("wait", "w1")]),
        reply("done"),
        tools=[waiting],
    ) as p:
        _ = p.team.join(p.agent)
        _ = await p.agent.send("go")
        await asyncio.sleep(0)

    states = [line.state for line in p.states()]
    members = [all(m.busy for m in s.team or ()) for s in states if s.busy]
    assert [busy for busy, _ in itertools.groupby(s.busy for s in states)] == [
        False,
        True,
        False,
    ]
    assert members == [True] * len(members)
    assert len({s.since for s in states if s.busy}) == 1


async def _ranks_logins_on_a_thread(turn_start: bool) -> None:
    """Wait on a worker thread before a model call, as a team's ``before_call`` does."""
    del turn_start
    await asyncio.to_thread(time.sleep, 0.005)


# A team agent's runtime awaits ``before_call`` before each model call, and the team's
# hook ranks its provider's logins on a worker thread (``team._Precedence``): the loop
# turns there with no call running, after a reply's calls start or end.
@pytest.mark.parametrize(
    "arguments",
    [{}, {"background": True}],
    ids=["waited", "background"],
)
@pytest.mark.asyncio
async def test_a_turn_is_busy_until_it_ends_though_each_model_call_waits_on_a_thread(
    tmp_path: Path,
    arguments: dict[str, PlainTree],
) -> None:
    waiting = _Waiting()
    waiting.release.set()
    async with _published(
        tmp_path,
        reply(calls=[call("wait", "w1", **arguments)]),
        reply(calls=[call("wait", "w2", **arguments)]),
        reply("done"),
        # What a background call's result wakes the model to say, once the turn ended.
        reply("noted"),
        reply("noted"),
        tools=[waiting],
    ) as p:
        _ = p.team.join(p.agent)
        p.agent.runtime.before_call = _ranks_logins_on_a_thread
        _ = await p.agent.send("go")
        await asyncio.sleep(0)

    ended = next(
        n
        for n, line in enumerate(p.sent)
        if isinstance(line, EventLine) and line.kind == "ModelIdle"
    )
    told = next(
        n for n, line in enumerate(p.sent) if n > ended and isinstance(line, StateLine)
    )
    turn = [
        line.state.busy for line in p.sent[: told + 1] if isinstance(line, StateLine)
    ]
    assert [busy for busy, _ in itertools.groupby(turn)] == [False, True, False]


# A child's runtime waits on its own ``before_call`` before each of its calls, as its
# parent's does; neither member is told idle between the steps of its turn.
@pytest.mark.asyncio
async def test_each_member_is_busy_until_its_turn_ends_though_each_call_waits_on_a_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        team_module._Precedence,
        "before_call",
        staticmethod(_ranks_logins_on_a_thread),
    )
    waiting = _Waiting()
    waiting.release.set()
    async with _published(
        tmp_path,
        reply(calls=[call("AgentSpawn", "s1", prompt="dig", label="kid")]),
        reply(calls=[call("wait", "w1")]),
        reply("root done"),
        children=(
            reply(calls=[call("wait", "w2")]),
            reply(calls=[call("wait", "w3")]),
            reply("kid done"),
        ),
        tools=[waiting],
    ) as p:
        _ = await p.agent.send("go")
        await asyncio.sleep(0)

    states = [line.state for line in p.states()]
    for label in ("Agent", "kid"):
        seen = (m.busy for s in states for m in s.team or () if m.label == label)
        runs = [busy for busy, _ in itertools.groupby(seen)]
        assert (label, [busy for busy in runs if busy], runs[-1]) == (
            label,
            [True],
            False,
        )


@pytest.fixture
def five_hours_east_of_utc(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run on a host whose local time is five hours ahead of UTC."""
    monkeypatch.setenv("TZ", "Etc/GMT-5")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.mark.usefixtures("five_hours_east_of_utc")
@pytest.mark.asyncio
async def test_the_first_state_is_dated_now_on_a_host_away_from_utc(
    tmp_path: Path,
) -> None:
    async with _published(tmp_path) as p:
        built = p.states()[0].state.since

    assert abs(built - datetime.now(UTC)) < timedelta(minutes=1)


# A naive local time reads as UTC on the wire, so on a host at UTC it passes unseen.
@pytest.mark.usefixtures("five_hours_east_of_utc")
@pytest.mark.asyncio
async def test_each_time_a_turn_tells_is_now_on_a_host_away_from_utc(
    tmp_path: Path,
) -> None:
    async with _published(tmp_path, reply("hello")) as p:
        _ = await p.agent.send("hi")
        await asyncio.sleep(0)

    states = [line.state for line in p.states()]
    ended = [s.turn_ended for s in states if s.turn_ended is not None]
    assert [s.busy for s in states] == [False, True, False]
    assert ended
    for each in [*(s.since for s in states), *ended]:
        assert abs(each - datetime.now(UTC)) < timedelta(minutes=1)


@pytest.mark.asyncio
async def test_the_first_state_counts_what_a_resumed_session_spent(
    tmp_path: Path,
) -> None:
    sent: list[EventLine | StateLine | SettingsLine] = []
    async with _published(tmp_path) as p:
        p.agent.spend_usd = 1.25
        _ = Publisher(
            p.agent,
            team=p.team,
            client=p.store,
            session_id=_SESSION,
            part=_PART,
            start=0,
            send=sent.append,
        )

    [first] = [line for line in sent if isinstance(line, StateLine)]
    assert "$1.25" in first.state.line


@pytest.mark.asyncio
async def test_a_change_of_text_alone_goes_at_most_once_a_second(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _published(tmp_path) as p:
        await _told(p, types.ModelResponseStarted())
        before = len(p.states())
        await _told(p, types.ModelResponseTextStreamed(text="x" * 4_000))
        withheld = len(p.states()) - before
        monkeypatch.setattr(publisher_module, "_LINE_SEC", 0.0)
        await _told(p, types.ModelResponseTextStreamed(text="x" * 4_000))

    assert withheld == 0
    assert len(p.states()) == before + 1
    assert p.states()[-1].state.busy


@dataclass(slots=True, kw_only=True)
class _Clock:
    """The status line's clock, which the test moves on."""

    now: float = 1_000.0

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return time.time()


@dataclass(frozen=True, slots=True, kw_only=True)
class _Read:
    """A read of the state the publisher armed, held back until the test fires it."""

    delay_sec: float
    handle: asyncio.TimerHandle
    fire: Callable[[], object]


def _held_reads(monkeypatch: pytest.MonkeyPatch) -> list[_Read]:
    """Hold back each read of the state a publisher arms on the running loop."""
    reads: list[_Read] = []
    loop = asyncio.get_running_loop()
    later = loop.call_later

    def held(
        delay: float,
        callback: Callable[..., object],
        *args: object,
    ) -> asyncio.TimerHandle:
        if not isinstance(getattr(callback, "__self__", None), Publisher):
            return later(delay, callback, *args)
        handle = later(3_600, callback, *args)
        fire = functools.partial(callback, *args)
        reads.append(_Read(delay_sec=delay, handle=handle, fire=fire))
        return handle

    monkeypatch.setattr(loop, "call_later", held)
    return reads


@pytest.mark.asyncio
async def test_a_call_that_streams_nothing_still_moves_the_line_until_idle_or_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(status_module, "time", clock)
    reads = _held_reads(monkeypatch)
    async with _published(tmp_path) as p:
        idle = len(reads)
        await _told(p, types.ModelResponseStarted())
        started = p.states()[-1].state.line
        [first] = reads
        clock.now += 20
        monkeypatch.setattr(publisher_module, "_LINE_SEC", 0.0)
        first.fire()
        moved = p.states()[-1].state
        await _told(p, types.ModelResponseCompleted(outcome=Cancelled()))
        # A halted call ends its turn too, as the runtime tells it.
        await _told(p, ModelIdle(outcome=Cancelled()))
        reads[-1].fire()
        after_idle = len(reads)
        await _told(p, types.ModelResponseStarted())
        last = reads[-1]
    p.publisher(types.ModelResponseCompleted(outcome=Cancelled()))
    # A read already under way as it closed arms no other, nor does a call after it.
    last.fire()
    p.publisher(types.ModelResponseStarted())

    assert (idle, first.delay_sec) == (0, 1.0)
    assert "waiting on model." not in started
    assert moved.busy
    assert moved.line.startswith("[20s ")
    assert moved.line.endswith("; waiting on model.]")
    assert after_idle == 2
    assert last.handle.cancelled()
    assert len(reads) == 3
    assert all(getattr(f, "__self__", None) is not p.publisher for f in p.team.changes)


@dataclass(slots=True, kw_only=True)
class _Waiting:
    """A tool whose calls run until the test lets them end."""

    name: str = "wait"
    description: str = "Wait for tests."
    schema: Mapping[str, PlainTree] = field(default_factory=dict[str, "PlainTree"])
    display: ToolDisplay = field(default_factory=ToolDisplay)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    runs_until_stopped: bool = False
    """Whether it says its calls end only when stopped, as a tool that watches does."""

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        return f"wait for {arguments}"

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        del context, arguments
        return None

    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[UncategorizedToolResult, object, object]:
        del context, arguments
        return whole_output(
            self._run(),
            runs_until_stopped=self.runs_until_stopped,
        )

    async def _run(self) -> ToolOutput[UncategorizedToolResult, object, object]:
        _ = await self.release.wait()
        return ToolOutput(stdout=("done",), result=UncategorizedToolResult())


@dataclass(slots=True, kw_only=True)
class _Heard:
    """An observer set once it hears an event of ``kind``."""

    kind: type
    seen: asyncio.Event = field(default_factory=asyncio.Event)

    def __call__(self, event: types.RuntimeEvent) -> None:
        if isinstance(event, self.kind):
            self.seen.set()


def _when(agent: Agent, kind: type) -> asyncio.Event:
    """Return an event set once ``agent`` publishes an event of ``kind``."""
    heard = _Heard(kind=kind)
    agent.runtime.observers.append(heard)
    return heard.seen


@pytest.mark.asyncio
async def test_a_background_job_goes_up_at_once_and_its_turn_ends_with_its_result(
    tmp_path: Path,
) -> None:
    waiting = _Waiting()
    async with _published(
        tmp_path,
        reply(calls=[call("wait", "w1", what="the build", background=True)]),
        reply("started it"),
        reply("it is built"),
        tools=[waiting],
    ) as p:
        answered = _when(p.agent, ModelIdle)
        turn = asyncio.create_task(p.agent.send("go"))
        _ = await answered.wait()
        running = p.states()[-1].state
        waiting.release.set()
        _ = await turn

    assert running.jobs is not None
    [job] = running.jobs
    assert (job.call_id, job.tool, job.title, job.agent, job.state) == (
        "w1",
        "wait",
        "wait for {'what': 'the build'}",
        "",
        "running",
    )
    assert abs(job.started - datetime.now(UTC)) < timedelta(minutes=1)
    assert (running.busy, running.working, running.turn_ended) == (False, True, None)
    ended = p.states()[-1].state
    assert (ended.jobs, ended.working) == ((), False)
    assert ended.turn_ended is not None
    assert ended.turn_ended > job.started


@pytest.mark.asyncio
async def test_a_member_whose_turn_ended_is_not_busy_while_only_a_background_job_runs(
    tmp_path: Path,
) -> None:
    waiting = _Waiting()
    async with _published(
        tmp_path,
        reply(calls=[call("wait", "w1", background=True)]),
        reply("started it"),
        reply("it is built"),
        tools=[waiting],
    ) as p:
        _ = p.team.join(p.agent)
        answered = _when(p.agent, ModelIdle)
        turn = asyncio.create_task(p.agent.send("go"))
        _ = await answered.wait()
        await asyncio.sleep(0)
        running = p.states()[-1].state
        waiting.release.set()
        _ = await turn

    assert [(m.label, m.busy) for m in running.team or ()] == [("Agent", False)]
    assert (running.busy, running.working) == (False, True)


def _scripted(
    tmp_path: Path,
    *replies: Reply,
    journal: JsonlJournal | None = None,
) -> Agent:
    """Return an agent that answers ``replies`` in turn, writing ``journal`` if one."""
    return Agent(
        model=ScriptedProvider(replies=replies)["scripted"],
        journal=journal,
        tool_context=ToolContext(cwd=tmp_path),
    )


@pytest.mark.asyncio
async def test_a_closed_publisher_tells_nothing_of_a_child_heard_as_the_team_changed(
    tmp_path: Path,
) -> None:
    """Each change of the team looks again at who joined; a child is heard once."""
    team = Team(catalog=ModelCatalog(factories={}))
    # The root answers its child's idle.
    agent, kid = _scripted(tmp_path, reply("noted")), _scripted(tmp_path, reply("pong"))
    sent: list[EventLine | StateLine | SettingsLine] = []
    async with agent, team, kid:
        _ = team.join(agent)
        _ = team.join(kid, label="kid", parent="Agent")
        publisher = Publisher(
            agent,
            team=team,
            client=_Store(),
            session_id=_SESSION,
            part=_PART,
            start=0,
            send=sent.append,
        )
        team.set_peer_messaging("Agent", False)
        await publisher.aclose()
        told = len(sent)
        _ = await kid.send("ping")
        await asyncio.sleep(0)

    assert len(sent) == told


@pytest.mark.parametrize("before", [True, False], ids=["just before", "after"])
@pytest.mark.asyncio
async def test_a_read_an_event_asks_for_around_the_close_tells_nothing_after_it(
    tmp_path: Path,
    before: bool,
) -> None:
    """An event reads the state once the loop next turns, which may be after the close."""
    async with _published(tmp_path) as p:
        if before:
            p.publisher(types.ModelResponseStarted())
        await p.publisher.aclose()
        told = p.states()
        # The runtime goes on telling of events, as the agent closes after its publisher.
        if not before:
            p.publisher(types.ModelResponseStarted())
        await asyncio.sleep(0)
        after = p.states()

    assert after == told


# Each child tells its spawner's runtime what sagent shows of it (``agent_spawn._Forward``),
# but tells no child's on: a grandchild's stops at its parent, below the root's runtime.
@pytest.mark.asyncio
async def test_what_sagent_shows_of_an_agent_below_a_child_goes_up_once_under_its_labels(
    tmp_path: Path,
) -> None:
    team = Team(catalog=ModelCatalog(factories={}))
    # Each spawner answers what its child's failure tells it.
    agent = _scripted(tmp_path, reply("noted"), reply("noted"))
    kid, grandkid = _scripted(tmp_path, reply("noted")), _scripted(tmp_path)
    shown = (
        ModelIdle(outcome=Failed(exception=RuntimeError("down"))),
        types.ModelResponseSuspended(
            exception=RetryableError("busy", retry_after_sec=1.0),
            retry_at=datetime.fromtimestamp(0.0, UTC),
            failures_in_a_row=1,
        ),
        types.LoginSwitched(provider="p", from_login="a", to_login="b"),
    )
    sent: list[EventLine | StateLine | SettingsLine] = []
    async with agent, team, kid, grandkid:
        _ = team.join(agent)
        _ = team.join(kid, label="kid", parent="Agent")
        _ = team.join(grandkid, label="grandkid", parent="kid")
        # As a spawn wires them.
        kid.runtime.observers.append(_Forward(label="kid", parent=agent))
        grandkid.runtime.observers.append(_Forward(label="grandkid", parent=kid))
        publisher = Publisher(
            agent,
            team=team,
            client=_Store(),
            session_id=_SESSION,
            part=_PART,
            start=0,
            send=sent.append,
        )
        agent.runtime.observers.append(publisher)
        try:
            # The team changes again after each joined: each is heard once all the same.
            team.set_peer_messaging("Agent", False)
            for each in (kid, grandkid):
                for event in (*shown, ModelIdle(outcome=Succeeded(value=None))):
                    each.runtime.publish(event)
        finally:
            await publisher.aclose()

    told = [line.payload for line in sent if isinstance(line, EventLine)]

    def below(*labels: str, event: types.RuntimeEvent) -> JSON:
        for label in reversed(labels):
            event = types.ChildEvent(label=label, event=event)
        return event_payload(event)

    succeeded = ModelIdle(outcome=Succeeded(value=None))
    assert [told.count(below("kid", event=e)) for e in shown] == [1, 1, 1]
    assert [told.count(below("kid", "grandkid", event=e)) for e in shown] == [1, 1, 1]
    assert [below("grandkid", event=e) in told for e in shown] == [False] * 3
    assert below("kid", "grandkid", event=succeeded) not in told


@dataclass(frozen=True, slots=True, kw_only=True)
class _Far:
    """An agent on another machine, which takes what it is sent and says nothing."""

    def send(
        self,
        *,
        sender: str,
        recipient: str,
        content: str,
        kind: MessageKind,
    ) -> None:
        del sender, recipient, content, kind


@pytest.mark.asyncio
async def test_a_child_an_agent_of_another_machine_spawned_is_told_of_by_no_one_here(
    tmp_path: Path,
) -> None:
    """Its spawner is no member: what sagent shows of it goes to that spawner alone."""
    team = Team(catalog=ModelCatalog(factories={}))
    agent, kid = _scripted(tmp_path), _scripted(tmp_path)
    sent: list[EventLine | StateLine | SettingsLine] = []
    async with agent, team, kid:
        _ = team.join(agent)
        team.reach("far@box", relay=_Far())
        _ = team.join(kid, label="kid", parent="far@box")
        publisher = Publisher(
            agent,
            team=team,
            client=_Store(),
            session_id=_SESSION,
            part=_PART,
            start=0,
            send=sent.append,
        )
        try:
            kid.runtime.publish(
                ModelIdle(outcome=Failed(exception=RuntimeError("down"))),
            )
        finally:
            await publisher.aclose()

    assert [line for line in sent if isinstance(line, EventLine)] == []


@pytest.mark.asyncio
async def test_an_agent_that_takes_the_label_another_left_is_heard_as_its_own(
    tmp_path: Path,
) -> None:
    team = Team(catalog=ModelCatalog(factories={}))
    agent = _scripted(tmp_path, reply("noted"))
    gone, kid = _scripted(tmp_path), _scripted(tmp_path, reply("pong"))
    sent: list[EventLine | StateLine | SettingsLine] = []
    async with agent, team, gone, kid:
        _ = team.join(agent)
        _ = team.join(gone, label="kid", parent="Agent")
        publisher = Publisher(
            agent,
            team=team,
            client=_Store(),
            session_id=_SESSION,
            part=_PART,
            start=0,
            send=sent.append,
        )
        try:
            _ = team.leave("kid")
            _ = team.join(kid, label="kid", parent="Agent")
            # Only the child works: what is read of it, its own events ask for.
            _ = await kid.send("ping")
            await asyncio.sleep(0)
        finally:
            await publisher.aclose()

    assert True in [
        m.busy
        for line in sent
        if isinstance(line, StateLine)
        for m in line.state.team or ()
        if m.label == "kid"
    ]


@pytest.mark.asyncio
async def test_a_child_that_joins_at_work_goes_up_at_its_own_save_its_secrets_masked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No catch-up asks for it, nor does it resume before the publisher is built."""
    monkeypatch.setenv("TRAX_REDACT_NAMES", "KEY")
    monkeypatch.setenv("KEY", "hunter2-hunter2")
    team = Team(catalog=ModelCatalog(factories={}))
    journal = JsonlJournal(tmp_path / _FOLDER / "session.jsonl")
    kid_journal = JsonlJournal(tmp_path / _FOLDER / uuid4().hex / "session.jsonl")
    agent = _scripted(tmp_path, reply("noted"), journal=journal)
    kid = _scripted(tmp_path, reply("pong"), journal=kid_journal)
    store = _Store()
    async with journal, agent, team, kid_journal, kid:
        _ = team.join(agent)
        publisher = Publisher(
            agent,
            team=team,
            client=store,
            session_id=_SESSION,
            part=_PART,
            start=0,
            send=lambda _: None,
        )
        try:
            _ = team.join(kid, label="kid", parent="Agent")
            _ = await kid.send("the key is hunter2-hunter2")
            appended = await asyncio.to_thread(store.child_appended.wait, 1)
        finally:
            await publisher.aclose()

    assert appended
    [(_, stored)] = store.children.values()
    bodies = [r.model_dump_json() for r in stored]
    assert any("[redacted:KEY]" in body for body in bodies)
    assert not any("hunter2-hunter2" in body for body in bodies)


@pytest.mark.asyncio
async def test_a_job_that_runs_until_stopped_keeps_nothing_working_and_no_turn_open(
    tmp_path: Path,
) -> None:
    watching = _Waiting(runs_until_stopped=True)
    async with _published(
        tmp_path,
        reply(calls=[call("wait", "w1", what="news", background=True)]),
        reply("watching"),
        tools=[watching],
    ) as p:
        # The turn ends with the call still running: it has no result to wait for.
        assert await p.agent.send("go") == "watching"
        idle = p.states()[-1].state
        watching.release.set()

    assert idle.jobs is not None
    assert [(job.call_id, job.runs_until_stopped) for job in idle.jobs] == [
        ("w1", True),
    ]
    assert (idle.busy, idle.working) == (False, False)
    assert idle.turn_ended is not None


@pytest.mark.asyncio
async def test_a_job_its_agent_waits_for_goes_up_awaited_and_the_wait_is_no_job(
    tmp_path: Path,
) -> None:
    waiting = _Waiting()
    async with _published(
        tmp_path,
        reply(calls=[call("wait", "w1", what="the build", background=True)]),
        reply(calls=[call("BackgroundTask", "f1", operation="foreground", id="w1")]),
        reply("it is built"),
        tools=[waiting, BackgroundTask()],
    ) as p:
        says = _StateSays(published=p)
        p.agent.runtime.observers.append(says)
        turn = asyncio.create_task(p.agent.send("go"))
        async with asyncio.timeout(5):
            _ = await says.seen.wait()
        awaited = next(line.state for line in p.states() if _awaits(line.state))
        waiting.release.set()
        _ = await turn

    assert [(job.call_id, job.awaited) for job in awaited.jobs or ()] == [("w1", True)]


def _awaits(state: AgentState) -> bool:
    """Return whether ``state`` says a job is awaited."""
    return any(job.awaited for job in state.jobs or ())


@dataclass(slots=True, kw_only=True)
class _StateSays:
    """An observer heard after the publisher: set once a state it sent awaits a job."""

    published: _Published
    seen: asyncio.Event = field(default_factory=asyncio.Event)

    # The publisher reads the state an event asks for once the loop turns: look after.
    def __call__(self, event: types.RuntimeEvent) -> None:
        del event
        _ = asyncio.get_running_loop().call_soon(self._look)

    def _look(self) -> None:
        if any(_awaits(line.state) for line in self.published.states()):
            self.seen.set()


@dataclass(frozen=True, slots=True, kw_only=True)
class _Lineage:
    """A team as the tree reads it: its agents by label, and who spawned whom."""

    agents: dict[str, Agent]
    parents: dict[str, str | None]

    def __iter__(self) -> Iterator[str]:
        return iter(self.agents)

    def __getitem__(self, label: str) -> Agent:
        return self.agents[label]

    def parent(self, label: str) -> str | None:
        return self.parents[label]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Held:
    """A job as a runtime lists it."""

    call: ToolCall
    status: str
    runs_until_stopped: bool = False


def _job(
    call_id: str,
    status: str,
    name: str = "Bash",
    **arguments: PlainTree,
) -> _Held:
    """Return a job of ``name``'s, asked for at 10 s past the epoch."""
    return _Held(
        call=ToolCall(
            call_id=call_id,
            name=name,
            arguments=arguments,
            timestamp=datetime.fromtimestamp(10.0, UTC),
        ),
        status=status,
    )


def _member(
    *held: _Held,
    spend_usd: float = 0.0,
    tools: Mapping[str, object] | None = None,
) -> Agent:
    """Return an agent running ``held``, those ``running`` in its turn, holding ``tools``."""
    runtime = SimpleNamespace(
        jobs={job.call.call_id: job for job in held},
        at_work=any(job.status == "running" for job in held),
        tools=dict(tools or {}),
    )
    return cast("Agent", SimpleNamespace(runtime=runtime, spend_usd=spend_usd))


def test_spawns_and_job_management_are_no_jobs_and_a_wait_marks_the_job_it_waits_for() -> (
    None
):
    summarized: list[Mapping[str, PlainTree]] = []

    def summary(arguments: Mapping[str, PlainTree]) -> str:
        summarized.append(arguments)
        return "Bash make"

    root = _member(
        _job("w1", "detached", command="make", background=True),
        _job("w2", "sleeping", "gone"),
        _job("s1", "detached", "AgentSpawn"),
        _job("f1", "detached", "BackgroundTask", operation="foreground", id="w1"),
        _job("l1", "detached", "BackgroundTask", operation="list"),
        tools={"Bash": SimpleNamespace(summary=summary)},
    )
    team = _Lineage(agents={"Agent": root}, parents={"Agent": None})

    jobs, _ = publisher_module.team_work(
        cast("Team", team),
        root,
        mask=str,
        sessions=lambda _: None,
        calling=lambda _: False,
    )

    assert [(job.call_id, job.title, job.state, job.awaited) for job in jobs] == [
        ("w1", "Bash make", "running", True),
        # A call detached with its group may name a tool the agent does not hold.
        ("w2", "gone", "sleeping", False),
    ]
    assert summarized == [{"command": "make"}]


# Away from UTC, so a job's start read as local time would show another instant.
@pytest.mark.usefixtures("five_hours_east_of_utc")
def test_each_agent_s_jobs_go_up_by_its_label_masked_those_its_turn_waits_on_left_out() -> (
    None
):
    root = _member(_job("r1", "detached"), spend_usd=1.5)
    kid = _member(_job("k1", "sleeping"), _job("k2", "running"))
    team = _Lineage(
        agents={"Agent": root, "kid": kid},
        parents={"Agent": None, "kid": "Agent"},
    )

    jobs, members = publisher_module.team_work(
        cast("Team", team),
        root,
        mask=str.lower,
        sessions=lambda agent: _SESSION if agent is root else None,
        # The root's model works on its turn, between two of its steps.
        calling=lambda agent: agent is root,
    )

    started = datetime.fromtimestamp(10.0, UTC)
    assert jobs == (
        BackgroundJob(
            call_id="r1",
            tool="Bash",
            title="bash",
            started=started,
            state="running",
        ),
        BackgroundJob(
            call_id="k1",
            tool="Bash",
            title="bash",
            agent="kid",
            started=started,
            state="sleeping",
        ),
    )
    assert members == (
        TeamMember(label="Agent", busy=True, spend_usd=1.5, session=_SESSION),
        TeamMember(label="kid", parent="Agent", busy=True),
    )


# Away from UTC, so an unstamped call's start read as local time would show another instant.
@pytest.mark.usefixtures("five_hours_east_of_utc")
@pytest.mark.asyncio
async def test_only_calls_run_apart_are_jobs_each_under_its_agent_s_label(
    tmp_path: Path,
) -> None:
    """A call the model waits on is no background job."""
    agent = Agent(
        model=ScriptedProvider()["scripted"],
        tool_context=ToolContext(cwd=tmp_path),
    )
    at_zero = datetime.fromtimestamp(0.0, UTC)
    calls: tuple[tuple[str, datetime | None, Literal["waited", "apart"], bool], ...] = (
        ("running", at_zero, "waited", True),
        ("detached", at_zero, "apart", True),
        ("sleeping", at_zero, "apart", False),
        # A call no runtime stamped, as a test's own may be, starts now, in UTC.
        ("unstamped", None, "apart", True),
    )
    for call_id, stamp, status, started in calls:
        agent.runtime._calls[call_id] = runtime_module._Call(
            call=ToolCall(call_id=call_id, name="gone", timestamp=stamp),
            key=None,
            task=asyncio.create_task(asyncio.wait_for(asyncio.Future[None](), None)),
            background=status == "apart",
            status=status,
            refusal=None,
            sleeping=not started,
        )
    before = datetime.now(UTC)

    jobs = publisher_module._jobs(agent, label="kid", mask=str)
    for each in agent.runtime._calls.values():
        _ = each.task.cancel()

    assert [(job.call_id, job.agent, job.state) for job in jobs] == [
        ("detached", "kid", "running"),
        ("sleeping", "kid", "sleeping"),
        ("unstamped", "kid", "running"),
    ]
    assert jobs[0].started == datetime.fromtimestamp(0.0, UTC)
    assert timedelta(0) <= jobs[-1].started - before < timedelta(minutes=1)


@pytest.mark.asyncio
async def test_a_delayed_job_goes_up_sleeping(tmp_path: Path) -> None:
    async with _published(
        tmp_path,
        reply(calls=[call("wait", "w1", delay=3_600)]),
        reply("later"),
        tools=[_Waiting()],
    ) as p:
        answered = _when(p.agent, ModelIdle)
        turn = asyncio.create_task(p.agent.send("go"))
        _ = await answered.wait()
        sleeping = p.states()[-1].state
        turn.cancel()

    assert [job.state for job in sleeping.jobs or ()] == ["sleeping"]


@pytest.mark.asyncio
async def test_the_team_goes_up_as_sagent_s_tree_each_agent_with_its_session(
    tmp_path: Path,
) -> None:
    async with _published(
        tmp_path,
        *_SPAWNS_KID,
        children=_KID_SPAWNS_GRANDKID[1:],
    ) as p:
        _ = await p.agent.send("go")
        await p.publisher.caught_up()
        # The kid's session opened at its first save; the next event takes it up.
        await _told(p, AgentIdle())
        kid = p.team["kid"].session.journal.path
        # Read before the agents close: the quit is told, of a team gone.
        teams = [
            [(m.label, m.parent, m.busy, m.session) for m in line.state.team or ()]
            for line in p.states()
        ]

    assert kid is not None
    assert [("Agent", "", True, _SESSION), ("kid", "Agent", True, None)] in teams
    assert teams[-1] == [
        ("Agent", "", False, _SESSION),
        ("kid", "Agent", False, uuid5(NAMESPACE_URL, f"{_FOLDER}/{kid.parent.name}")),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ending",
    [
        AgentIdle(),
        ModelIdle(outcome=Failed(exception=RuntimeError("down"))),
        ModelIdle(outcome=Cancelled()),
    ],
    ids=["idle", "failed", "halted"],
)
async def test_a_turn_ends_when_the_root_idles_after_a_call_or_stops_short(
    tmp_path: Path,
    ending: types.RuntimeEvent,
) -> None:
    async with _published(tmp_path) as p:
        await _told(p, AgentIdle())
        await _told(
            p,
            types.ChildEvent(label="kid", event=types.ModelResponseStarted()),
        )
        await _told(p, types.ChildEvent(label="kid", event=ending))
        unended = p.states()[-1].state.turn_ended
        await _told(p, types.ModelResponseStarted())
        before = datetime.now(UTC)
        await _told(p, ending)
        ended = p.states()[-1].state.turn_ended
        await _told(p, AgentIdle())
        again = p.states()[-1].state.turn_ended

    assert unended is None
    assert ended is not None
    assert ended >= before
    assert again == ended


# Only the runtime knows whether a turn stopped: a compaction cancels a call, a
# failure may be got past, and a halt or a clear of an agent at rest stops nothing.
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event",
    [
        types.ModelResponseCompleted(outcome=Cancelled()),
        types.ModelResponseCompleted(
            outcome=Failed(exception=RuntimeError("down")),
        ),
        types.ModelHalt(),
        types.SessionClearCompleted(outcome=Succeeded(value=None)),
        ModelIdle(outcome=Succeeded(value=None)),
    ],
    ids=["a-call-cancelled", "a-call-failed", "a-halt-asked", "a-clear", "finished"],
)
async def test_nothing_but_an_idle_or_a_turn_that_stopped_short_ends_a_turn(
    tmp_path: Path,
    event: types.RuntimeEvent,
) -> None:
    async with _published(tmp_path) as p:
        # Its runtime idles as it starts, which ends no turn; let it, before the call.
        await asyncio.sleep(0)
        await _told(p, types.ModelResponseStarted())
        await _told(p, event)
        unended = p.states()[-1].state.turn_ended
        await _told(p, AgentIdle())
        idled = p.states()[-1].state.turn_ended

    assert unended is None
    assert idled is not None


@pytest.mark.asyncio
async def test_a_turn_that_failed_and_is_then_finished_ends_again_at_its_idle(
    tmp_path: Path,
) -> None:
    """A retry may get past the failure: the same turn then ends a second time."""
    async with _published(tmp_path) as p:
        # Its runtime idles as it starts, which ends no turn; let it, before the call.
        await asyncio.sleep(0)
        await _told(p, types.ModelResponseStarted())
        await _told(p, ModelIdle(outcome=Failed(exception=RuntimeError("x"))))
        failed = p.states()[-1].state.turn_ended
        await _told(p, types.ModelResponseStarted())
        await _told(p, ModelIdle(outcome=Succeeded(value=None)))
        retried = p.states()[-1].state.turn_ended
        await _told(p, AgentIdle())
        finished = p.states()[-1].state.turn_ended

    assert failed is not None
    assert retried == failed
    assert finished is not None
    assert finished > failed


@pytest.mark.asyncio
async def test_a_message_waiting_mid_stream_goes_at_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waiting = UserMessage(content="also this")
    async with _published(tmp_path) as p:
        monkeypatch.setattr(p.agent.runtime, "pending_mid_stream", lambda: (waiting,))
        await _told(p, AgentIdle())

    assert p.states()[-1].state.pending == (event_payload(waiting),)


@pytest.mark.asyncio
async def test_the_settings_go_when_assigned_and_again_only_when_they_change(
    tmp_path: Path,
) -> None:
    async with _published(tmp_path) as p:
        commands = Commands(
            p.agent,
            catalog=ModelCatalog(factories={"scripted": ScriptedProvider}),
            print_output=p.publisher.print_output,
            screen=_Screen(),
            panes=Panes(),
        )
        p.publisher.commands = commands
        assert p.publisher.commands is commands
        p.publisher.print_output(CommandOutput(text="nothing changed", tone="note"))
        _ = await commands("/screen.thinking show")
        p.publisher.check_settings()
        # A setting with words of every kind, so a field left out shows.
        source = next(s for s in harvest(commands.owners()) if s.help_long and s.usage)

    first, second = p.settings()
    thinking = {s.name: s for s in first.settings}[("screen", "thinking")]
    assert (thinking.value, thinking.options) == (
        "hide (default)",
        ("default", "show", "hide"),
    )
    sent = {s.name: s for s in first.settings}[source.name]
    assert (sent.help_long, sent.usage) == (source.help_long, source.usage)
    assert {s.name: s for s in second.settings}[("screen", "thinking")].value == "show"
    model = {s.name: s for s in first.settings}[("model",)]
    assert model.options == ("scripted/scripted",)
    login = {s.name: s for s in first.settings}[("login",)]
    assert (login.value, login.flags) == (
        None,
        ("--headless", "--help", "--no-headless", "-h"),
    )


@pytest.mark.asyncio
async def test_a_setting_changed_with_nothing_said_here_goes_when_busy_next_changes(
    tmp_path: Path,
) -> None:
    """Other commands on the same screen change it, and say so elsewhere."""
    shown: list[CommandOutput] = []
    screen = _Screen()
    async with _published(tmp_path) as p:
        catalog = ModelCatalog(factories={"scripted": ScriptedProvider})
        elsewhere = Commands(
            p.agent,
            catalog=catalog,
            print_output=shown.append,
            screen=screen,
            panes=Panes(),
        )
        p.publisher.commands = Commands(
            p.agent,
            catalog=catalog,
            print_output=p.publisher.print_output,
            screen=screen,
            panes=Panes(),
        )
        _ = await elsewhere("/screen.thinking show")
        unsent = len(p.settings())
        await _told(p, types.ModelResponseStarted())

    assert (unsent, len(p.settings())) == (1, 2)
    settings = {s.name: s for s in p.settings()[-1].settings}
    assert (settings[("screen", "thinking")].value or "").startswith("show")


@pytest.mark.asyncio
async def test_an_outage_is_tried_again_until_the_records_land(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    outages = [ClientError("down"), ClientError("down", status_code=503)]
    store = _Store(failing=[*outages])
    async with _published(tmp_path, reply("hello"), store=store) as p:
        _ = await p.agent.send("hi")
        await p.publisher.caught_up()
        stored = list(p.agent.session.history())

    assert p.store.positions() == list(range(len(stored)))
    assert not p.publisher.ended.done()
    assert _warnings(caplog) == [f"upload failed, trying again: {e}" for e in outages]


@pytest.mark.asyncio
async def test_an_outage_partway_through_a_pass_resends_only_what_did_not_land(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _published(tmp_path, reply("one"), reply("two")) as first:
        _ = await first.agent.send("a")
        await first.publisher.caught_up()
        start = len(list(first.agent.session.history()))
        _ = await first.agent.send("b")
        await first.publisher.caught_up()
    # One record a batch, so the second batch's outage comes after the first landed.
    monkeypatch.setattr(
        publisher_module,
        "send_batches",
        functools.partial(send_batches, batch_size=1),
    )
    store = _Store(failing=[None, ClientError("down")])
    async with _published(tmp_path, start=start, store=store) as second:
        await second.publisher.caught_up()
        stored = len(list(second.agent.session.history()))

    assert second.store.positions() == list(range(start, stored))


@pytest.mark.asyncio
async def test_an_outage_after_a_pass_landed_is_tried_again(tmp_path: Path) -> None:
    held = threading.Event()
    async with _published(tmp_path, reply("one"), reply("two")) as p:
        _ = await p.agent.send("a")
        await p.publisher.caught_up()
        # The turn's first save starts a pass that waits to land its records; the
        # rest of the turn is saved meanwhile, and its pass meets the outage.
        p.store.held = held
        _ = await p.agent.send("b")
        p.store.failing[:] = [None, ClientError("down")]
        held.set()
        await p.publisher.caught_up()
        stored = list(p.agent.session.history())

    assert p.store.positions() == list(range(len(stored)))


@pytest.mark.parametrize("status", [401, 403, 404, 410])
@pytest.mark.asyncio
async def test_a_refused_key_or_a_gone_session_ends_the_process(
    tmp_path: Path,
    status: int,
) -> None:
    store = _Store(failing=[ClientError("no", status_code=status)])
    async with _published(tmp_path, reply("hello"), store=store) as p:
        _ = await p.agent.send("hi")
        await p.publisher.caught_up()

    assert p.publisher.ended.result() == 1
    assert "refused the upload" in p.publisher.reason
    assert store.appended == []


@pytest.mark.asyncio
async def test_a_part_another_host_continued_ends_the_process_with_3(
    tmp_path: Path,
) -> None:
    stale = ClientError("stale", status_code=409, code=STALE_PART)
    store = _Store(failing=[stale])
    async with _published(tmp_path, reply("hello"), store=store) as p:
        _ = await p.agent.send("hi")
        await p.publisher.caught_up()

    assert p.publisher.ended.result() == 3
    assert p.publisher.reason == (
        "another host continued the session; this journal is kept"
    )
    assert store.appended == []


@pytest.mark.asyncio
async def test_another_conflict_is_tried_again_and_ends_nothing(
    tmp_path: Path,
) -> None:
    other = ClientError("busy", status_code=409, code="busy")
    store = _Store(failing=[other])
    async with _published(tmp_path, reply("hello"), store=store) as p:
        _ = await p.agent.send("hi")
        await p.publisher.caught_up()

    assert not p.publisher.ended.done()
    assert store.appended


@pytest.mark.asyncio
async def test_a_record_the_server_refuses_goes_as_its_stand_in(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _Store(refusing="AssistantMessage")
    async with _published(tmp_path, reply("hello"), store=store) as p:
        _ = await p.agent.send("hi")
        await p.publisher.caught_up()
        stored = list(p.agent.session.history())

    kinds = {r.idx: r.kind for _, _, batch in p.store.appended for r in batch}
    said = next(
        i for i, r in enumerate(stored) if type(r).__name__ == "AssistantMessage"
    )
    assert sorted(kinds) == list(range(len(stored)))
    assert kinds[said] == "UncategorizedRecord"
    assert not p.publisher.ended.done()
    assert _warnings(caplog) == [
        f"the server refused record {said}; a stand-in holds it",
    ]


@pytest.mark.asyncio
async def test_an_end_no_one_waits_on_any_more_is_kept_to_itself(
    tmp_path: Path,
) -> None:
    refused: list[ClientError | None] = [ClientError("no", status_code=401)]
    async with _published(tmp_path, reply("hello"), store=_Store(failing=refused)) as p:
        _ = p.publisher.ended.cancel()
        _ = await p.agent.send("hi")
        await p.publisher.caught_up()

    assert p.publisher.reason == ""


@pytest.mark.asyncio
async def test_one_upload_runs_at_a_time_and_catches_up_on_what_came_meanwhile(
    tmp_path: Path,
) -> None:
    held = threading.Event()
    store = _Store(held=held)
    async with _published(tmp_path, reply("one"), reply("two"), store=store) as p:
        _ = await p.agent.send("a")
        _ = await p.agent.send("b")
        held.set()
        await p.publisher.caught_up()
        stored = list(p.agent.session.history())

    assert p.store.positions() == list(range(len(stored)))


@pytest.mark.asyncio
async def test_records_saved_while_a_pass_finds_nothing_new_still_go_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _published(tmp_path, reply("one"), reply("two")) as p:
        _ = await p.agent.send("a")
        await p.publisher.caught_up()
        journal = p.agent.session
        history = journal.history
        reading, read = threading.Event(), threading.Event()
        waited: list[int] = []

        # A read is of the journal as it stands; its records are taken on the upload's
        # thread, which the first read holds there, as a slow hop does, while the
        # turn's saves land.
        def first_read_waits(start: int = 0) -> Iterator[SessionRecord]:
            records = tuple(history(start))
            if waited:
                return iter(records)
            waited.append(start)

            def waiting() -> Iterator[SessionRecord]:
                reading.set()
                assert read.wait(5)
                yield from records

            return waiting()

        monkeypatch.setattr(journal, "history", first_read_waits)
        p.publisher(types.SessionSaveDue())
        assert await asyncio.to_thread(reading.wait, 5)
        _ = await p.agent.send("b")
        read.set()
        await p.publisher.caught_up()
        stored = list(history())

    assert p.store.positions() == list(range(len(stored)))


@pytest.mark.asyncio
async def test_a_secret_named_for_masking_never_goes_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAX_REDACT_NAMES", "KEY")
    monkeypatch.setenv("KEY", "hunter2-hunter2")
    waiting = UserMessage(content="and again hunter2-hunter2")
    async with _published(tmp_path, reply("hello")) as p:
        _ = await p.agent.send("the key is hunter2-hunter2")
        await p.publisher.caught_up()
        p.publisher.print_output(
            CommandOutput(text="[/login] hunter2-hunter2", tone="note"),
        )
        monkeypatch.setattr(p.agent.runtime, "pending_mid_stream", lambda: (waiting,))
        await _told(p, types.ModelResponseStarted())

    sent = [r.model_dump_json() for _, _, batch in p.store.appended for r in batch]
    assert any("[redacted:KEY]" in body for body in sent)
    assert not any("hunter2-hunter2" in body for body in sent)
    live = {line.kind: line.model_dump_json() for line in p.events()}
    assert "[redacted:KEY]" in live["UserMessage"]
    assert "[redacted:KEY]" in live["CommandOutput"]
    assert "[redacted:KEY]" in p.states()[-1].model_dump_json()
    assert not any("hunter2-hunter2" in line.model_dump_json() for line in p.sent)


@pytest.mark.asyncio
async def test_with_no_secret_named_an_event_goes_up_as_it_is(tmp_path: Path) -> None:
    async with _published(tmp_path) as p:
        p.publisher(types.ModelResponseTextStreamed(text="hunter2-hunter2"))

    assert p.events()[0].payload == event_payload(
        types.ModelResponseTextStreamed(text="hunter2-hunter2"),
    )


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
