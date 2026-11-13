"""The publisher: each runtime event up as a line, the journal up as records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final
from uuid import UUID, uuid4

import asyncio
import contextlib
import functools
import itertools
import threading
import time

from sagent.agent.agent import Agent
from sagent.agent.agents_md import AgentsMdConfig
from sagent.model.scripted import ScriptedProvider, reply
from sagent.providers.catalog import ModelCatalog
from sagent.runtime import types
from sagent.session.jsonl import JsonlJournal
from sagent.session.types import AgentToAgentMessage, UserMessage
from sagent.settings.harvest import harvest
from sagent.tools.types import ToolContext
from sagent.tui.prompt import Panes
from sagent.tui.slash import Commands
from sagent.tui.types import CommandOutput
from trackinizer.client.errors import ClientError
from trackinizer.client.session_upload import send_batches
from trackinizer.wire.wire_session_ir import ManifestBody, RecordBody
from trackinizer.wire.wire_session_relay import (
    STALE_PART,
    EventLine,
    SettingsLine,
    StateLine,
)

import pytest

from knowop.process import publisher as publisher_module
from knowop.process.codec import event_payload
from knowop.process.publisher import MAX_EVENT_BYTES, Publisher


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator, Sequence
    from pathlib import Path

    from sagent.model.scripted import Reply
    from sagent.session.types import SessionRecord


_SESSION: Final = UUID(int=9)
_FOLDER: Final = uuid4().hex
_PART: Final = f"{_FOLDER}.jsonl"


@dataclass(slots=True, kw_only=True)
class _Store:
    """Trackinizer's records route: it keeps what is appended, or answers as told."""

    appended: list[tuple[str, ManifestBody, list[RecordBody]]] = field(
        default_factory=list[tuple[str, ManifestBody, list[RecordBody]]],
    )
    failing: list[ClientError | None] = field(default_factory=list[ClientError | None])
    """What the next appends raise, in turn; ``None`` lets one through."""

    held: threading.Event | None = None
    """When set, an append waits for it."""

    refusing: str = ""
    """A kind of record the server cannot store; empty for none."""

    def append_records(
        self,
        session_id: UUID,
        *,
        name: str,
        manifest: ManifestBody,
        records: Sequence[RecordBody],
    ) -> object:
        assert session_id == _SESSION
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
    publisher: Publisher
    store: _Store
    sent: list[EventLine | StateLine | SettingsLine]

    def events(self) -> list[EventLine]:
        return [line for line in self.sent if isinstance(line, EventLine)]

    def states(self) -> list[StateLine]:
        return [line for line in self.sent if isinstance(line, StateLine)]

    def settings(self) -> list[SettingsLine]:
        return [line for line in self.sent if isinstance(line, SettingsLine)]


@contextlib.asynccontextmanager
async def _published(
    tmp_path: Path,
    *replies: Reply,
    start: int = 0,
    store: _Store | None = None,
) -> AsyncGenerator[_Published]:
    provider = ScriptedProvider(replies=replies)
    journal = JsonlJournal(tmp_path / _FOLDER / "session.jsonl")
    agent = Agent(
        model=provider["scripted"],
        journal=journal,
        tool_context=ToolContext(cwd=tmp_path),
        system_prompt="S",
        agents_md=AgentsMdConfig(
            system_dir=tmp_path / "system",
            user_dir=tmp_path / "user",
        ),
    )
    sent: list[EventLine | StateLine | SettingsLine] = []
    async with journal, agent:
        store = store if store is not None else _Store()
        publisher = Publisher(
            agent,
            client=store,
            session_id=_SESSION,
            part=_PART,
            start=start,
            send=sent.append,
        )
        agent.runtime.observers.append(publisher)
        try:
            yield _Published(agent=agent, publisher=publisher, store=store, sent=sent)
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


@pytest.mark.asyncio
async def test_a_part_another_host_continued_ends_the_process_with_3(
    tmp_path: Path,
) -> None:
    stale = ClientError("stale", status_code=409, code=STALE_PART)
    async with _published(tmp_path, reply("hello"), store=_Store(failing=[stale])) as p:
        _ = await p.agent.send("hi")
        await p.publisher.caught_up()

    assert p.publisher.ended.result() == 3
    assert "another host" in p.publisher.reason


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
