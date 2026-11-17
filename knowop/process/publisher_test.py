"""The publisher: each runtime event up as a line, the journal up as records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import asyncio
import contextlib
import functools
import itertools
import threading
import time

from sagent.agent.agent import Agent
from sagent.agent.agents_md import AgentsMdConfig
from sagent.agent.team import Team
from sagent.model.scripted import ScriptedProvider, call, reply
from sagent.providers.catalog import ModelCatalog
from sagent.runtime import types
from sagent.session.jsonl import JsonlJournal
from sagent.session.types import (
    AgentToAgentMessage,
    ToolCall,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.settings.harvest import harvest
from sagent.tools.agent_spawn import AgentSpawn
from sagent.tools.display import ToolDisplay
from sagent.tools.types import ToolContext, ToolOutput, whole_output
from sagent.tui.prompt import Panes
from sagent.tui.slash import Commands
from sagent.tui.types import CommandOutput
from trackinizer.client.errors import ClientError
from trackinizer.client.session_upload import send_batches
from trackinizer.wire.wire_session_ir import ManifestBody, PartBody, RecordBody
from trackinizer.wire.wire_session_relay import (
    STALE_PART,
    EventLine,
    SettingsLine,
    StateLine,
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

    from sagent.model.scripted import Reply
    from sagent.session.types import SessionRecord
    from sagent.tools.types import Tool, ToolStreamingResult


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
    assert "SaveSession" not in kinds
    assert {"ModelCallStarted", "ModelResponseComplete", "AgentIdle"} <= set(kinds)
    idle = next(line for line in p.events() if line.kind == "AgentIdle")
    assert idle.payload == event_payload(types.AgentIdle())


@pytest.mark.asyncio
async def test_a_child_s_save_session_is_not_sent_at_any_depth(tmp_path: Path) -> None:
    async with _published(tmp_path) as p:
        p.publisher(
            types.ChildEvent(
                label="a",
                inner=types.ChildEvent(label="b", inner=types.SaveSession()),
            ),
        )
        p.publisher(types.ChildEvent(label="a", inner=types.AgentIdle()))

    assert [line.kind for line in p.events()] == ["ChildEvent"]
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
            stray.runtime.publish(types.SaveSession())
            await p.publisher.caught_up()

    assert p.store.starts == []
    assert any(
        "stray" in warned and "no journal of this session" in warned
        for warned in _warnings(caplog)
    )


@pytest.mark.asyncio
async def test_what_an_agent_tells_the_user_goes_up_as_its_message(
    tmp_path: Path,
) -> None:
    async with _published(tmp_path) as p:
        p.publisher.to_user("pyq2", "in the repository")

    said = AgentToAgentMessage(
        sender="pyq2",
        recipient="user",
        content="in the repository",
    )
    assert p.events() == [
        EventLine(kind="AgentToAgentMessage", payload=event_payload(said)),
    ]


@pytest.mark.asyncio
async def test_a_command_s_output_goes_up_as_an_event(tmp_path: Path) -> None:
    said = CommandOutput(text="[/effort] high", tone="note", command="/effort")
    async with _published(tmp_path) as p:
        p.publisher.say(said)

    assert p.events() == [EventLine(kind="CommandOutput", payload=event_payload(said))]


@pytest.mark.asyncio
async def test_a_long_event_is_cut_to_fit_and_one_that_cannot_be_is_not_sent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = 64 << 10
    monkeypatch.setattr(publisher_module, "MAX_EVENT_BYTES", bound)
    async with _published(tmp_path) as p:
        p.publisher(types.ModelResponsePartial(text="x" * 3 * bound))
        # Each string is cut to no less than a kibibyte, so these stay over.
        p.publisher(
            types.LoginsSpent(
                provider="p",
                resets=tuple((f"{n:04}" + "y" * 1_100, 0.0) for n in range(80)),
            ),
        )

    assert [line.kind for line in p.events()] == ["ModelResponsePartial"]
    assert len(p.events()[0].model_dump_json().encode()) <= bound
    assert MAX_EVENT_BYTES == 1 << 20


@pytest.mark.asyncio
async def test_a_state_is_sent_when_built_and_at_once_when_busy_changes(
    tmp_path: Path,
) -> None:
    async with _published(tmp_path) as p:
        built = list(p.states())
        p.publisher(types.ModelCallStarted())
        started = p.states()[-1]
        p.publisher(types.ModelResponseCancelled())

    assert [s.state.busy for s in built] == [False]
    assert started.state.busy
    assert started.state.since > built[0].state.since
    assert [s.state.busy for s in p.states()] == [False, True, False]


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
        p.publisher(types.ModelCallStarted())
        before = len(p.states())
        p.publisher(types.ModelResponsePartial(text="x" * 4_000))
        withheld = len(p.states()) - before
        monkeypatch.setattr(publisher_module, "_LINE_SEC", 0.0)
        p.publisher(types.ModelResponsePartial(text="x" * 4_000))

    assert withheld == 0
    assert len(p.states()) == before + 1
    assert p.states()[-1].state.busy


@dataclass(slots=True, kw_only=True)
class _Waiting:
    """A tool whose calls run until the test lets them end."""

    name: str = "wait"
    description: str = "Wait for tests."
    schema: Mapping[str, PlainTree] = field(default_factory=dict[str, "PlainTree"])
    display: ToolDisplay = field(default_factory=ToolDisplay)
    release: asyncio.Event = field(default_factory=asyncio.Event)

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
        return whole_output(self._run())

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
        answered = _when(p.agent, types.ModelIdle)
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


def test_a_job_whose_tool_the_agent_no_longer_holds_is_titled_by_its_name() -> None:
    """A call detached with its group may name a tool the agent does not hold."""
    called = ToolCall(call_id="w1", name="gone", arguments={"what": "x"})

    assert publisher_module._title(None, called) == "gone"


@pytest.mark.asyncio
async def test_a_delayed_job_goes_up_sleeping(tmp_path: Path) -> None:
    async with _published(
        tmp_path,
        reply(calls=[call("wait", "w1", delay=3_600)]),
        reply("later"),
        tools=[_Waiting()],
    ) as p:
        answered = _when(p.agent, types.ModelIdle)
        turn = asyncio.create_task(p.agent.send("go"))
        _ = await answered.wait()
        sleeping = p.states()[-1].state
        turn.cancel()

    assert [job.state for job in sleeping.jobs or ()] == ["sleeping"]


@pytest.mark.asyncio
async def test_a_child_at_work_goes_up_by_its_label(tmp_path: Path) -> None:
    async with _published(
        tmp_path,
        *_SPAWNS_KID,
        children=_KID_SPAWNS_GRANDKID[1:],
    ) as p:
        _ = await p.agent.send("go")

    agents = [s.state.agents for s in p.states()]
    assert ("kid",) in agents
    assert agents[-1] == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ending",
    [
        types.AgentIdle(),
        types.ModelResponseError(exception=RuntimeError("down")),
        types.ModelResponseCancelled(),
    ],
    ids=["idle", "failed", "halted"],
)
async def test_a_turn_ends_when_the_root_idles_after_a_call_fails_or_halts(
    tmp_path: Path,
    ending: types.RuntimeEvent,
) -> None:
    async with _published(tmp_path) as p:
        p.publisher(types.AgentIdle())
        p.publisher(types.ChildEvent(label="kid", inner=types.ModelCallStarted()))
        p.publisher(types.ChildEvent(label="kid", inner=ending))
        unended = p.states()[-1].state.turn_ended
        p.publisher(types.ModelCallStarted())
        before = datetime.now(UTC)
        p.publisher(ending)
        ended = p.states()[-1].state.turn_ended
        p.publisher(types.AgentIdle())
        again = p.states()[-1].state.turn_ended

    assert unended is None
    assert ended is not None
    assert ended >= before
    assert again == ended


@pytest.mark.asyncio
async def test_a_message_waiting_mid_stream_goes_at_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waiting = UserMessage(content="also this")
    async with _published(tmp_path) as p:
        monkeypatch.setattr(p.agent.runtime, "pending_mid_stream", lambda: (waiting,))
        p.publisher(types.AgentIdle())

    assert p.states()[-1].state.pending == (event_payload(waiting),)


@pytest.mark.asyncio
async def test_the_settings_go_when_assigned_and_again_only_when_they_change(
    tmp_path: Path,
) -> None:
    async with _published(tmp_path) as p:
        commands = Commands(
            p.agent,
            catalog=ModelCatalog(factories={"scripted": ScriptedProvider}),
            say=p.publisher.say,
            screen=_Screen(),
            panes=Panes(),
        )
        p.publisher.commands = commands
        assert p.publisher.commands is commands
        p.publisher.say(CommandOutput(text="nothing changed", tone="note"))
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
            say=shown.append,
            screen=screen,
            panes=Panes(),
        )
        p.publisher.commands = Commands(
            p.agent,
            catalog=catalog,
            say=p.publisher.say,
            screen=screen,
            panes=Panes(),
        )
        _ = await elsewhere("/screen.thinking show")
        unsent = len(p.settings())
        p.publisher(types.ModelCallStarted())

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
        p.publisher(types.SaveSession())
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
        p.publisher.say(CommandOutput(text="[/login] hunter2-hunter2", tone="note"))
        monkeypatch.setattr(p.agent.runtime, "pending_mid_stream", lambda: (waiting,))
        p.publisher(types.ModelCallStarted())

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
        p.publisher(types.ModelResponsePartial(text="hunter2-hunter2"))

    assert p.events()[0].payload == event_payload(
        types.ModelResponsePartial(text="hunter2-hunter2"),
    )


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
