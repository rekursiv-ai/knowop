"""The agent process's main: its flags, the live session, its inputs and its exits.

Against a small relay in the test that keeps each frame the process sends and sends
it inputs, a fake ``Client`` for the calls the process makes, and a scripted agent
standing in for the one sagent's ``opened`` builds.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from http import HTTPStatus
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Final,
    NoReturn,
    Self,
    cast,
    override,
)
from uuid import UUID, uuid4

import asyncio
import contextlib
import io
import itertools
import json
import logging
import os
import re

from sagent.agent.agent import Agent
from sagent.agent.agents_md import AgentsMdConfig
from sagent.agent.team import Team
from sagent.model.scripted import Reply, ScriptedProvider, call, reply
from sagent.providers.catalog import ModelCatalog
from sagent.session.jsonl import JournalInUseError, JsonlJournal
from sagent.session.types import (
    AgentToAgentMessage,
    ReadResult,
    ToolResult,
    UncategorizedRecord,
)
from sagent.tools import mcp
from sagent.tools.read import Read
from sagent.tools.remote import RemoteSession, RemoteTool
from sagent.tools.types import Tool, ToolContext
from sagent.tui.main import Flags, OpenAgent, StartError
from sagent.tui.prompt import Panes
from sagent.tui.render import error_text
from sagent.tui.slash import Commands
from sagent.tui.theme import ThemeState
from trackinizer.client.client import Client
from trackinizer.client.errors import ClientError
from trackinizer.client.session_socket import SessionSocket, SocketClient
from trackinizer.trax.profile import Profile
from trackinizer.wire.wire_process_keys import PROCESS_KEY_ROUTES, WEBSOCKET
from trackinizer.wire.wire_session_ir import ManifestBody, PartBody, RecordBody
from trackinizer.wire.wire_session_relay import (
    SESSION_PROCESS_WS_PATH,
    STALE_PART,
    AgentProcess,
    Hello,
    Input,
    Opened,
    ProcessLine,
    ProcessSpec,
    ServeSession,
    SettingsLine,
    ViewerFrame,
)
from trackinizer.wire.wire_sessions import (
    GitRef,
    MessageKind,
    SessionAgent,
    SessionStart,
    SessionStartResponse,
)
from trackinizer.wire.wire_terminals import (
    CLOSE_ENDED,
    CLOSE_HOST_REVOKED,
    CLOSE_REFUSED,
    CLOSE_TAKEN_OVER,
)
from trackinizer.wire.wire_tools import TOOLS_WS_PATH
from websockets.asyncio.server import (
    ServerConnection,
    serve as serve_websockets,
)
from websockets.exceptions import ConnectionClosed

import httpx2
import pytest

from knowop.agent import placement
from knowop.agent.errors import NotResumableError
from knowop.process import (
    main as main_module,
    publisher,
)
from knowop.process.codec import event_payload, input_message
from knowop.process.host_follower import HostFollower
from knowop.process.host_follower_test import _Client, _daemons, _host
from knowop.process.main import (
    Ended,
    ServeClient,
    first_done,
    main,
    serve,
)
from knowop.process.publisher import Publisher
from knowop.process.remote_commands import REMOTE_AGENT


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator

    from mcp.client.stdio import StdioServerParameters
    from sagent.runtime.types import RuntimeEvent
    from sagent.tui.types import CommandOutput
    from trackinizer.wire.json_types import JSON
    from trackinizer.wire.wire_session_relay import InputKind
    from websockets.http11 import Request, Response


_SESSION: Final = UUID(int=11)
_PROCESS: Final = UUID(int=12)
_HOST: Final = "laptop"
_AT: Final = "2026-11-10T00:00:00Z"
_PROFILE: Final = Profile(url="http://127.0.0.1:1", author="ada@x.y", api_key="k")
_WORKSPACE: Final = UUID(int=13)
_PLACED_ON: Final = UUID(int=14)
_CHECKOUT: Final = GitRef(origin="https://github.com/o/r.git", branch="main")


@dataclass(slots=True, kw_only=True)
class _Relay:
    """Trackinizer's process socket, as the process sees it."""

    port: int = 0
    connections: list[list[str]] = field(default_factory=list[list[str]])
    paths: list[str] = field(default_factory=list[str])
    refusing: HTTPStatus | None = None
    live: ServerConnection | None = None
    arrived: asyncio.Condition = field(default_factory=asyncio.Condition)

    async def handle(self, connection: ServerConnection) -> None:
        frames: list[str] = []
        self.connections.append(frames)
        self.live = connection
        with contextlib.suppress(ConnectionClosed):
            async for message in connection:
                assert isinstance(message, str)
                async with self.arrived:
                    frames.append(message)
                    self.arrived.notify_all()

    def refuse(self, connection: ServerConnection, request: Request) -> Response | None:
        self.paths.append(request.path)
        if self.refusing is None:
            return None
        return connection.respond(self.refusing, "refused\n")

    async def seen(
        self,
        wanted: Callable[[dict[str, object]], bool],
    ) -> dict[str, object]:
        """Return the first frame ``wanted`` takes, once one has come."""
        async with asyncio.timeout(5), self.arrived:
            await self.arrived.wait_for(lambda: self.found(wanted) is not None)
        found = self.found(wanted)
        assert found is not None
        return found

    def found(
        self,
        wanted: Callable[[dict[str, object]], bool],
    ) -> dict[str, object] | None:
        """Return the first frame ``wanted`` takes of those come so far, if any."""
        return next(iter(self.every(wanted)), None)

    def every(
        self,
        wanted: Callable[[dict[str, object]], bool],
    ) -> list[dict[str, object]]:
        """Return each frame ``wanted`` takes of those come so far, in order."""
        parsed = (
            cast("dict[str, object]", json.loads(frame))
            for frames in self.connections
            for frame in frames
        )
        return [frame for frame in parsed if wanted(frame)]

    async def send(
        self,
        kind: InputKind,
        payload: JSON,
        *,
        author: str = "ada@x.y",
        recipient: str = "",
        booked: MessageKind | None = None,
    ) -> UUID:
        sent = uuid4()
        assert self.live is not None
        await self.live.send(
            Input(
                id=sent,
                author=author,
                kind=kind,
                payload=payload,
                recipient=recipient,
                booked=booked,
            ).model_dump_json(),
        )
        return sent

    async def close(self, code: int) -> None:
        _ = await self.seen(_is("hello"))
        assert self.live is not None
        await self.live.close(code=code, reason=f"closed with {code}")


def _is(kind: str, *, event: str = "") -> Callable[[dict[str, object]], bool]:
    def wanted(frame: dict[str, object]) -> bool:
        return frame["type"] == kind and event in ("", frame.get("kind"))

    return wanted


@contextlib.asynccontextmanager
async def _relayed() -> AsyncGenerator[_Relay]:
    relay = _Relay()
    async with serve_websockets(
        relay.handle,
        "127.0.0.1",
        0,
        process_request=relay.refuse,
    ) as server:
        relay.port = server.sockets[0].getsockname()[1]
        yield relay


@dataclass(slots=True, kw_only=True)
class _Server(_Client):
    """Trackinizer's calls an agent process makes, as a fake ``Client``."""

    base_url: str = "http://127.0.0.1:1"
    api_key: str = "process-key"
    parts: list[PartBody] = field(default_factory=list[PartBody])
    stored: list[RecordBody] = field(default_factory=list[RecordBody])
    failing: list[ClientError] = field(default_factory=list[ClientError])
    started: list[ProcessSpec] = field(default_factory=list[ProcessSpec])
    answer: AgentProcess | None = None
    frames: list[ViewerFrame] = field(default_factory=list[ViewerFrame])
    lines_refused: bool = False
    start_refused: bool = False
    placed: list[tuple[str, object]] = field(default_factory=list[tuple[str, object]])
    place_refused: bool = False
    stopped: list[UUID] = field(default_factory=list[UUID])
    stop_answer: AgentProcess | ClientError | None = None
    messaged: list[tuple[SessionAgent, str, str, MessageKind | None]] = field(
        default_factory=list[tuple[SessionAgent, str, str, "MessageKind | None"]],
    )
    """Each message an agent here sent another session's: to, text, label, booking."""

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
        assert (from_session, key is None) == (_SESSION, False)
        self.messaged.append((to, text, from_label, booked))
        return [UUID(int=412)]

    def stop_process(self, session_id: UUID) -> AgentProcess | None:
        self.stopped.append(session_id)
        if isinstance(self.stop_answer, ClientError):
            raise self.stop_answer
        return self.stop_answer

    def post(self, path: str, *, body: object = None) -> dict[str, str]:
        self.placed.append((path, body))
        if self.place_refused:
            raise ClientError("POST place -> 403", status_code=403)
        return {"workspace_id": str(_WORKSPACE), "canvas_id": str(_PLACED_ON)}

    def read_session_parts(self, session_id: UUID) -> list[PartBody]:
        del session_id
        return self.parts

    def read_session_records(
        self,
        session_id: UUID,
        *,
        part: int = 0,
        plaintext_only: bool = False,
    ) -> list[RecordBody]:
        del session_id, part, plaintext_only
        return []

    def append_records(
        self,
        session_id: UUID,
        *,
        name: str,
        manifest: ManifestBody,
        records: Sequence[RecordBody],
    ) -> object:
        assert session_id == _SESSION
        del name, manifest
        if self.failing:
            raise self.failing.pop(0)
        self.stored.extend(records)
        return None

    def session_start(self, body: SessionStart) -> SessionStartResponse:
        assert body.cli == "sagent"
        if self.start_refused:
            raise ClientError("no", status_code=403)
        return SessionStartResponse.model_validate(
            {"id": str(_SESSION), "actor": "ada", "seq": 1},
        )

    def start_process(self, session_id: UUID, spec: ProcessSpec) -> AgentProcess:
        assert session_id == _SESSION
        self.started.append(spec)
        assert self.answer is not None
        return self.answer

    def open_lines(self, session_id: UUID) -> Iterator[ViewerFrame]:
        assert session_id == _SESSION
        if self.lines_refused:
            raise ClientError("GET lines -> 501", status_code=501)
        yield from self.frames

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.change(closed=True)


@dataclass(slots=True, kw_only=True)
class _Screen:
    show_thinking: bool = False


@dataclass(slots=True, kw_only=True)
class _Opened:
    """Stands in for sagent's ``opened``: a scripted agent on the journal flags name."""

    folder: Path
    monkeypatch: pytest.MonkeyPatch
    replies: list[Reply] = field(default_factory=list[Reply])
    tools: list[Tool] = field(default_factory=list[Tool])
    remote_tools_only: bool = False
    """Give the agent ``RemoteTool`` alone, over the hosts the main links."""

    show_thinking: bool = False
    """Whether the screen it builds shows reasoning."""

    raises: BaseException | None = None
    flags: list[Flags] = field(default_factory=list[Flags])
    given: list[Sequence[tuple[str, str]]] = field(
        default_factory=list[Sequence[tuple[str, str]]],
    )
    interactive: list[bool] = field(default_factory=list[bool])
    use_daemon: list[bool] = field(default_factory=list[bool])
    agent: Agent | None = None
    team: Team | None = None
    theme: ThemeState | None = None
    remote: RemoteSession | None = None
    halts: int = 0
    steps: list[str] = field(default_factory=list[str])

    @contextlib.asynccontextmanager
    async def __call__(
        self,
        flags: Flags,
        *,
        given: Sequence[tuple[str, str]],
        interactive: bool,
        use_daemon: bool = True,
    ) -> AsyncGenerator[OpenAgent]:
        self.flags.append(flags)
        self.given.append(given)
        self.interactive.append(interactive)
        self.use_daemon.append(use_daemon)
        if self.raises is not None:
            raise self.raises
        assert flags.session is not None
        provider = ScriptedProvider(replies=self.replies)
        catalog = ModelCatalog(factories={"scripted": ScriptedProvider})
        journal = JsonlJournal(flags.session)
        async with (
            Team(catalog=catalog) as team,
            RemoteSession([], retry_sec=0.01) as remote,
            journal,
        ):
            agent = Agent(
                model=provider["scripted"],
                journal=journal,
                tools=[RemoteTool(remote=remote)]
                if self.remote_tools_only
                else self.tools,
                tool_context=ToolContext(cwd=self.folder),
                system_prompt="S",
                agents_md=AgentsMdConfig(
                    system_dir=self.folder / "system",
                    user_dir=self.folder / "user",
                ),
            )
            _ = team.join(agent, label="Agent")
            self.monkeypatch.setattr(agent.runtime, "halt", self._halted(agent))
            async with agent:
                self.agent, self.team, self.remote = agent, team, remote
                self.theme = ThemeState()
                yield OpenAgent(
                    agent=agent,
                    catalog=catalog,
                    team=team,
                    remote=remote,
                    links=(),
                    screen=_Screen(show_thinking=self.show_thinking),
                    theme=self.theme,
                )

    def _halted(self, agent: Agent) -> Callable[[], None]:
        halt = agent.runtime.halt

        def counted() -> None:
            self.halts += 1
            self.steps.append("halt")
            halt()

        return counted


@pytest.fixture(autouse=True)
def this_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on a host of a known name; try a failed upload again at once."""
    monkeypatch.setattr(main_module, "host_name", lambda: _HOST)
    monkeypatch.setattr(publisher, "retry_delays", lambda: itertools.repeat(0.0))


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Opened:
    stand_in = _Opened(folder=tmp_path, monkeypatch=monkeypatch)
    monkeypatch.setattr(main_module, "opened", stand_in)
    return stand_in


@contextlib.asynccontextmanager
async def _serving(
    relay: _Relay,
    *,
    args: Sequence[str] = (),
    tui: bool = False,
    server: _Server | None = None,
    time_limit: timedelta | None = None,
    remote: bool = False,
) -> AsyncGenerator[tuple[asyncio.Task[int], _Server]]:
    """Run ``serve`` in a task against ``relay``; yield it and the fake server."""
    server = server if server is not None else _Server(version=1)
    server.base_url = f"http://127.0.0.1:{relay.port}"
    with server:
        task = asyncio.create_task(
            serve(
                args,
                session_id=_SESSION,
                process_id=_PROCESS,
                client=server,
                tui=tui,
                host=_HOST,
                time_limit=time_limit,
                remote=remote,
            ),
        )
        try:
            yield task, server
        finally:
            _ = task.cancel()
            _ = await asyncio.gather(task, return_exceptions=True)


async def _exited(task: asyncio.Task[int]) -> int:
    async with asyncio.timeout(5):
        return await task


@pytest.mark.asyncio
async def test_a_headless_run_says_hello_and_prints_its_session_first(
    opened: _Opened,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async with _relayed() as relay, _serving(relay) as (task, _):
        hello = Hello.model_validate(await relay.seen(_is("hello")))
        await relay.close(CLOSE_ENDED)
        code = await _exited(task)

    assert (hello.process_id, hello.host) == (_PROCESS, _HOST)
    assert relay.paths == [f"/api/sessions/{_SESSION}/process/ws"]
    assert capsys.readouterr().out.splitlines()[0] == str(_SESSION)
    assert code == 0
    assert opened.interactive == [False]
    # sagent's own daemon starts beside it, as for any agent a person runs.
    assert opened.use_daemon == [True]


@pytest.mark.asyncio
async def test_a_remote_agent_opens_without_sagents_daemon(opened: _Opened) -> None:
    async with _relayed() as relay, _serving(relay, remote=True) as (task, _):
        await relay.close(CLOSE_ENDED)
        _ = await _exited(task)

    assert opened.use_daemon == [False]


@pytest.mark.asyncio
async def test_the_journal_is_the_sessions_own_and_sagents_flags_reach_the_build(
    opened: _Opened,
) -> None:
    args = ["--name", "Lead", "--model.thinking.effort", "high"]
    async with _relayed() as relay, _serving(relay, args=args) as (task, _):
        await relay.close(CLOSE_ENDED)
        _ = await _exited(task)

    (flags,) = opened.flags
    assert flags.name == "Lead"
    assert opened.given == [[("model.thinking.effort", "high")]]
    assert flags.session is not None
    assert flags.session.parent.parent.name == str(_SESSION)


@pytest.mark.asyncio
async def test_a_line_from_chat_is_the_authors_and_the_agent_answers_it(
    opened: _Opened,
) -> None:
    opened.replies = [reply("hello ada")]
    async with _relayed() as relay, _serving(relay) as (task, server):
        _ = await relay.seen(_is("settings"))
        sent = await relay.send(
            "UserMessage",
            {"py/object": "UserMessage", "content": "hi", "extra": {}},
        )
        _ = await relay.seen(_is("event", event="ModelResponseComplete"))
        assert opened.agent is not None
        said = [
            r
            for r in opened.agent.session.history()
            if isinstance(r, AgentToAgentMessage)
        ]
        await relay.close(CLOSE_ENDED)
        _ = await _exited(task)

    assert [(m.sender, m.content, dict(m.extra)) for m in said] == [
        ("ada@x.y", "hi", {"author": "ada@x.y", "input": str(sent)}),
    ]
    assert "AgentToAgentMessage" in [r.kind for r in server.stored]


_STATUS: Final = {"py/object": "UserMessage", "content": "status?"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "remote",
    [False, True],
    ids=["beside sagent's daemon", "a remote agent, with none"],
)
async def test_another_session_s_agent_is_asked_and_answered_through_the_server(
    opened: _Opened,
    remote: bool,
) -> None:
    """Its message is owed a reply, and the reply goes back as one, daemon or none."""
    async with (
        _relayed() as relay,
        _serving(relay, remote=remote) as (task, server),
    ):
        _ = await relay.seen(_is("settings"))
        assert opened.team is not None
        assert opened.agent is not None
        team = opened.team
        # Held at the inbox, as the scripted model answers nothing; a halt goes on.
        arrived = asyncio.Queue[AgentToAgentMessage]()
        push = opened.agent.runtime.push

        def held(message: RuntimeEvent) -> None:
            if isinstance(message, AgentToAgentMessage):
                arrived.put_nowait(message)
            else:
                push(message)

        opened.monkeypatch.setattr(opened.agent.runtime, "push", held)
        _ = await relay.send("UserMessage", _STATUS, author="lead#412", booked="ask")
        message = await asyncio.wait_for(arrived.get(), 5)
        owed = team.owes("Agent")
        team.send("Agent", "lead#412", "green")
        await relay.close(CLOSE_ENDED)
        _ = await _exited(task)

    assert (message.sender, message.recipient, message.content) == (
        "lead#412",
        "Agent",
        "status?",
    )
    assert owed == {"lead#412": "status?"}
    assert server.messaged == [
        (SessionAgent(label="lead", session=412), "green", "Agent", "reply"),
    ]
    assert opened.use_daemon == [not remote]


@pytest.mark.asyncio
async def test_a_person_s_line_reaches_the_agent_it_names_or_says_none_took_it(
    opened: _Opened,
) -> None:
    async with _relayed() as relay, _serving(relay) as (task, _):
        _ = await relay.seen(_is("settings"))
        assert opened.team is not None
        helper = Agent(
            model=ScriptedProvider()["scripted"],
            tool_context=ToolContext(cwd=opened.folder),
        )
        _ = opened.team.join(helper, label="helper", parent="Agent")
        pushed: list[object] = []
        opened.monkeypatch.setattr(helper.runtime, "push", pushed.append)
        _ = await relay.send("UserMessage", _STATUS, recipient="helper")
        _ = await relay.send("UserMessage", _STATUS, recipient="ghost")
        said = await relay.seen(_is("event", event="CommandOutput"))
        await relay.close(CLOSE_ENDED)
        _ = await _exited(task)

    assert [
        (m.sender, m.recipient, m.content)
        for m in pushed
        if isinstance(m, AgentToAgentMessage)
    ] == [("ada@x.y", "helper", "status?")]
    payload = cast("dict[str, object]", said["payload"])
    assert payload["text"] == "No agent 'ghost' here took this line."


@pytest.mark.asyncio
async def test_what_an_agent_sends_the_user_goes_to_chat(opened: _Opened) -> None:
    async with _relayed() as relay, _serving(relay) as (task, _):
        _ = await relay.seen(_is("settings"))
        assert opened.team is not None
        # As a child answers a /send from Chat: to ``user``, whom no screen shows here.
        opened.team.send("Agent", "user", "in the repository")
        said = await relay.seen(_is("event", event="AgentToAgentMessage"))
        await relay.close(CLOSE_ENDED)
        _ = await _exited(task)

    assert said["payload"] == event_payload(
        AgentToAgentMessage(
            sender="Agent",
            recipient="user",
            content="in the repository",
        ),
    )


@pytest.mark.asyncio
async def test_a_halt_from_chat_halts_the_turn(opened: _Opened) -> None:
    async with _relayed() as relay, _serving(relay) as (task, _):
        _ = await relay.seen(_is("settings"))
        _ = await relay.send("Halt", {"py/object": "Halt"})
        _ = await relay.seen(_is("event", event="Halt"))
        await relay.close(CLOSE_TAKEN_OVER)
        _ = await _exited(task)

    assert opened.halts == 1


@pytest.mark.asyncio
async def test_a_slash_line_runs_and_says_what_it_did_and_the_settings_follow(
    opened: _Opened,
) -> None:
    async with _relayed() as relay, _serving(relay) as (task, _):
        first = await relay.seen(_is("settings"))
        _ = await relay.send("Command", {"line": "/screen.thinking show"})
        said = await relay.seen(_is("event", event="CommandOutput"))
        changed = await relay.seen(lambda f: f["type"] == "settings" and f != first)
        await relay.close(CLOSE_ENDED)
        _ = await _exited(task)

    assert "show" in json.dumps(said["payload"])
    settings = SettingsLine.model_validate(changed).settings
    thinking = {s.name: s for s in settings}[("screen", "thinking")]
    assert (thinking.value or "").startswith("show")
    del opened


_SERVER_PROBE: Final = (
    "/mcp.list --strict-mcp-config --mcp-config "
    '\'{"mcpServers":{"probe":{"command":"/usr/bin/touch","args":["/tmp/marker"]}}}\''
)
"""The line that ran a program on alpha-staging: ``mcp.list`` starts each server."""


@pytest.mark.asyncio
@pytest.mark.parametrize("remote", [True, False], ids=["remote", "own-host"])
async def test_a_line_that_starts_a_server_starts_none_on_a_remote_agent(
    opened: _Opened,
    monkeypatch: pytest.MonkeyPatch,
    *,
    remote: bool,
) -> None:
    launched: list[str] = []

    def launcher(parameters: StdioServerParameters, *, errlog: object) -> NoReturn:
        """Stand in for the stdio launcher: say what it would run, and run nothing."""
        del errlog
        launched.append(parameters.command)
        raise OSError("not run in a test")

    monkeypatch.setattr(mcp, "stdio_client", launcher)
    async with _relayed() as relay, _serving(relay, remote=remote) as (task, _):
        _ = await relay.seen(_is("settings"))
        _ = await relay.send("Command", {"line": _SERVER_PROBE})
        said = await relay.seen(_is("event", event="CommandOutput"))
        await relay.close(CLOSE_ENDED)
        _ = await _exited(task)

    told = cast("dict[str, object]", said["payload"])
    if remote:
        assert launched == []
        assert told["text"] == "/mcp.list is not available for a remote agent"
    else:
        assert launched == ["/usr/bin/touch"]
        assert told["command"] == "/mcp.list"
    del opened


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("line", "name"),
    [
        ("/mcp.list", "mcp.list"),
        ("/host.list", "host.list"),
        ("/remote.list", "remote.list"),
        ("/login", "login"),
        ("/login.precedence off", "login.precedence"),
        ("/model.provider.account work", "model.provider.account"),
        ("/peer-messaging off", "peer-messaging"),
        ("/screen.theme light", "screen.theme"),
    ],
)
@pytest.mark.parametrize("remote", [True, False], ids=["remote", "own-host"])
async def test_each_refused_family_is_refused_for_a_remote_agent_alone(
    opened: _Opened,
    line: str,
    name: str,
    *,
    remote: bool,
) -> None:
    async with _relayed() as relay, _serving(relay, remote=remote) as (task, _):
        _ = await relay.seen(_is("settings"))
        _ = await relay.send("Command", {"line": line})
        said = await relay.seen(_is("event", event="CommandOutput"))
        await relay.close(CLOSE_ENDED)
        _ = await _exited(task)

    told = cast("dict[str, object]", said["payload"])
    refused = f"/{name} is not available for a remote agent"
    assert (told["text"] == refused) is remote
    if not remote:
        # It ran, and said what came of it under its own name.
        assert told["command"] == f"/{name}"
    del opened


@pytest.mark.asyncio
async def test_a_remote_agent_tells_a_viewer_only_the_lines_it_runs(
    opened: _Opened,
) -> None:
    async with _relayed() as relay, _serving(relay, remote=True) as (task, _):
        first = SettingsLine.model_validate(await relay.seen(_is("settings")))
        await relay.close(CLOSE_ENDED)
        _ = await _exited(task)

    named = [setting.name for setting in first.settings]
    assert ("compact",) in named
    assert ("model",) in named
    assert all(REMOTE_AGENT.allows(name) for name in named)
    assert not [name for name in named if name[0] in {"mcp", "host", "remote"}]
    del opened


@pytest.mark.asyncio
async def test_a_slash_line_that_says_nothing_still_sends_the_settings_it_changed(
    opened: _Opened,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def quiet(self: Commands, line: str) -> bool:
        del line
        self.screen.show_thinking = True
        return False

    monkeypatch.setattr(Commands, "__call__", quiet)
    async with _relayed() as relay, _serving(relay) as (task, _):
        first = await relay.seen(_is("settings"))
        _ = await relay.send("Command", {"line": "/quiet"})
        changed = await relay.seen(lambda f: f["type"] == "settings" and f != first)
        await relay.close(CLOSE_ENDED)
        _ = await _exited(task)

    settings = SettingsLine.model_validate(changed).settings
    thinking = {s.name: s for s in settings}[("screen", "thinking")]
    assert (thinking.value or "").startswith("show")
    assert relay.found(_is("event", event="CommandOutput")) is None
    del opened


@pytest.mark.asyncio
async def test_quit_from_chat_ends_the_process_as_quit_at_its_screen(
    opened: _Opened,
) -> None:
    async with _relayed() as relay, _serving(relay) as (task, _):
        _ = await relay.seen(_is("settings"))
        _ = await relay.send("Command", {"line": "/quit"})
        code = await _exited(task)
        assert relay.live is not None
        await relay.live.wait_closed()
        closed = relay.live.close_code

    assert (code, closed) == (0, 1000)
    del opened


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        ("UserMessage", {"py/object": "UserMessage", "sender": "eve", "content": "x"}),
        ("Halt", {"py/object": "Nope"}),
    ],
    ids=["names-its-own-sender", "not-its-kind"],
)
@pytest.mark.asyncio
async def test_an_input_that_does_not_read_is_answered_as_an_error(
    opened: _Opened,
    kind: InputKind,
    payload: JSON,
) -> None:
    async with _relayed() as relay, _serving(relay) as (task, _):
        _ = await relay.seen(_is("settings"))
        sent = await relay.send(kind, payload)
        said = await relay.seen(_is("event", event="CommandOutput"))
        # The next input is still delivered.
        _ = await relay.send("Halt", {"py/object": "Halt"})
        _ = await relay.seen(_is("event", event="Halt"))
        await relay.close(CLOSE_ENDED)
        _ = await _exited(task)

    with pytest.raises(ValueError, match=r".") as unread:
        _ = input_message(Input(id=sent, author="ada@x.y", kind=kind, payload=payload))
    told = cast("dict[str, object]", said["payload"])
    assert (told["text"], told["tone"]) == (error_text(unread.value), "error")
    # The halt sent after it, then the stop's own.
    assert opened.halts == 2


@pytest.mark.asyncio
async def test_a_slash_line_that_raises_is_answered_as_an_error_and_logged(
    opened: _Opened,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    raised = RuntimeError("/model broke")

    async def broken(self: Commands, line: str) -> bool:
        del self, line
        raise raised

    monkeypatch.setattr(Commands, "__call__", broken)
    caplog.set_level(logging.WARNING, logger=main_module.__name__)
    async with _relayed() as relay, _serving(relay) as (task, _):
        _ = await relay.seen(_is("settings"))
        _ = await relay.send("Command", {"line": "/model"})
        said = await relay.seen(_is("event", event="CommandOutput"))
        # The next input is still delivered.
        _ = await relay.send("Halt", {"py/object": "Halt"})
        _ = await relay.seen(_is("event", event="Halt"))
        await relay.close(CLOSE_ENDED)
        _ = await _exited(task)

    told = cast("dict[str, object]", said["payload"])
    assert (told["text"], told["tone"]) == (error_text(raised), "error")
    assert [
        (record.getMessage(), record.exc_info and record.exc_info[1])
        for record in caplog.records
        if record.name == main_module.__name__
    ] == [("command '/model' failed", raised)]
    del opened


@pytest.mark.parametrize(
    ("code", "exit_code"),
    [
        (CLOSE_ENDED, 0),
        (CLOSE_REFUSED, 1),
        (CLOSE_HOST_REVOKED, 1),
        (CLOSE_TAKEN_OVER, 3),
    ],
)
@pytest.mark.asyncio
async def test_each_close_code_ends_the_process_with_its_exit_code_and_reason(
    opened: _Opened,
    capsys: pytest.CaptureFixture[str],
    code: int,
    exit_code: int,
) -> None:
    async with _relayed() as relay, _serving(relay) as (task, _):
        await relay.close(code)
        exited = await _exited(task)

    assert exited == exit_code
    assert f"closed with {code}" in capsys.readouterr().err
    del opened


@pytest.mark.asyncio
async def test_a_stop_halts_the_turn_and_uploads_what_the_journal_holds(
    opened: _Opened,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Recorded(Publisher):
        @override
        async def caught_up(self) -> None:
            opened.steps.append("caught up")
            await super().caught_up()

    monkeypatch.setattr(main_module, "Publisher", Recorded)
    async with _relayed() as relay, _serving(relay) as (task, server):
        _ = await relay.seen(_is("settings"))
        assert opened.agent is not None
        # Journaled with no ``SaveSession``, so only the stop's own upload sends it.
        opened.agent.session.stand(
            "note",
            UncategorizedRecord(kind="note", payload={"kept": True}),
        )
        await relay.close(CLOSE_ENDED)
        code = await _exited(task)

    assert code == 0
    assert opened.steps == ["halt", "caught up"]
    assert "UncategorizedRecord" in [r.kind for r in server.stored]
    journal = opened.flags[0].session
    assert journal is not None
    lines = [line for line in journal.read_bytes().splitlines() if line.strip()]
    assert sorted({r.idx for r in server.stored}) == list(range(len(lines)))


@pytest.mark.asyncio
@pytest.mark.parametrize("remote", [False, True], ids=["own host", "remote agent"])
async def test_a_quit_from_chat_uploads_what_the_journal_holds_before_it_ends(
    opened: _Opened,
    *,
    remote: bool,
) -> None:
    """A remote agent's folder goes as it ends: what is not stored by then is lost."""
    async with _relayed() as relay, _serving(relay, remote=remote) as (task, server):
        _ = await relay.seen(_is("settings"))
        assert opened.agent is not None
        # Journaled with no ``SaveSession``: an upload still owed as ``/quit`` comes.
        opened.agent.session.stand(
            "note",
            UncategorizedRecord(kind="note", payload={"kept": True}),
        )
        _ = await relay.send("Command", {"line": "/quit"})
        code = await _exited(task)

    assert code == 0
    assert "UncategorizedRecord" in [r.kind for r in server.stored]
    assert opened.steps == ["halt"]


@pytest.mark.asyncio
async def test_at_its_time_limit_a_process_ends_itself_as_a_stop_ends_it(
    opened: _Opened,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Recorded(Publisher):
        @override
        async def caught_up(self) -> None:
            opened.steps.append("caught up")
            await super().caught_up()

    monkeypatch.setattr(main_module, "Publisher", Recorded)
    limit = timedelta(milliseconds=50)
    async with _relayed() as relay, _serving(relay, time_limit=limit) as (task, _):
        _ = await relay.seen(_is("settings"))
        code = await _exited(task)

    assert code == 0
    assert opened.steps == ["halt", "caught up"]
    assert capsys.readouterr().err == "ended at its time limit, 0-00:00:00\n"


@pytest.mark.asyncio
async def test_a_process_runs_until_its_time_limit(opened: _Opened) -> None:
    limit = timedelta(hours=1)
    async with _relayed() as relay, _serving(relay, time_limit=limit) as (task, _):
        _ = await relay.seen(_is("settings"))
        await asyncio.sleep(
            0.05,
        )  # house-ignore[sleep] -- Proves nothing ends it early.
        assert not task.done()
        await relay.close(CLOSE_ENDED)
        assert await _exited(task) == 0
    del opened


@pytest.mark.asyncio
async def test_a_stop_ends_the_process_when_its_turn_will_not_halt_in_time(
    opened: _Opened,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    halting = asyncio.Event()

    async def never_halts(agent: Agent) -> None:
        del agent
        halting.set()
        _ = await asyncio.Event().wait()

    monkeypatch.setattr(main_module, "_halted", never_halts)
    monkeypatch.setattr(main_module, "_STOPPED_SEC", 0.05)
    async with _relayed() as relay, _serving(relay) as (task, _):
        _ = await relay.seen(_is("settings"))
        await relay.close(CLOSE_ENDED)
        code = await _exited(task)

    assert halting.is_set()
    assert code == 0
    del opened


@pytest.mark.asyncio
async def test_hosts_are_followed_on_the_processs_own_server_and_key_beside_its_links(
    opened: _Opened,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    followed: list[dict[str, object]] = []

    class Followed:
        Config = HostFollower.Config

        def __init__(self, config: HostFollower.Config, **kwargs: object) -> None:
            del config
            followed.append(kwargs)

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *exc: object) -> None:
            del exc

    monkeypatch.setattr(main_module, "HostFollower", Followed)
    async with _relayed() as relay, _serving(relay) as (task, server):
        await relay.close(CLOSE_ENDED)
        _ = await _exited(task)

    [given] = followed
    assert (given["url"], given["api_key"]) == (server.base_url, server.api_key)
    assert (given["keep"], given["own_host"]) == ((), _HOST)
    assert given["remote"] is opened.remote


@pytest.mark.asyncio
async def test_a_slash_line_from_chat_has_the_team_the_panes_and_the_theme_of_its_agent(
    opened: _Opened,
) -> None:
    lines = ("/send nobody hi", "/kill all", "/screen.theme light")
    async with _relayed() as relay, _serving(relay) as (task, _):
        _ = await relay.seen(_is("settings"))
        for line in lines:
            _ = await relay.send("Command", {"line": line})
        said = _is("event", event="CommandOutput")
        async with asyncio.timeout(5), relay.arrived:
            _ = await relay.arrived.wait_for(
                lambda: len(relay.every(said)) == len(lines),
            )
        outputs = [cast("dict[str, str]", f["payload"]) for f in relay.every(said)]
        await relay.close(CLOSE_ENDED)
        _ = await _exited(task)

    assert [(o["tone"], o["text"]) for o in outputs[:2]] == [
        # Looked up in the session's team, which holds no other agent; with none at
        # all the line would say "there are no other agents".
        ("error", "no agent 'nobody'; agents: none"),
        ("note", "cancelled all tool calls"),
    ]
    assert opened.theme is not None
    assert opened.theme.current.name == "light"


@pytest.mark.asyncio
async def test_a_headless_process_without_its_socket_too_long_exits_4(
    opened: _Opened,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "_GIVE_UP_SEC", 0.05)
    async with _relayed() as relay, _serving(relay) as (task, _):
        _ = await relay.seen(_is("settings"))
        relay.refusing = HTTPStatus.SERVICE_UNAVAILABLE
        assert relay.live is not None
        await relay.live.close(code=1011)
        code = await _exited(task)

    assert code == 4
    del opened


@pytest.mark.asyncio
async def test_a_refused_key_exits_1_before_any_agent_is_built(opened: _Opened) -> None:
    async with _relayed() as relay:
        relay.refusing = HTTPStatus.UNAUTHORIZED
        async with _serving(relay) as (task, _):
            code = await _exited(task)

    assert code == 1
    assert opened.flags == []


@pytest.mark.parametrize(
    ("answer", "exit_code"),
    [
        (ClientError("no", status_code=401), 1),
        (ClientError("gone", status_code=410), 1),
        (ClientError("stale", status_code=409, code=STALE_PART), 3),
    ],
    ids=["refused-key", "session-gone", "continued-elsewhere"],
)
@pytest.mark.asyncio
async def test_an_upload_the_server_refuses_ends_the_process(
    opened: _Opened,
    answer: ClientError,
    exit_code: int,
) -> None:
    opened.replies = [reply("hello")]
    async with _relayed() as relay, _serving(relay) as (task, server):
        _ = await relay.seen(_is("settings"))
        server.failing = [answer]
        _ = await relay.send(
            "UserMessage",
            {"py/object": "UserMessage", "content": "x"},
        )
        code = await _exited(task)

    assert code == exit_code


@pytest.mark.parametrize(
    ("raised", "exit_code"),
    [
        (StartError("unknown tools Nope"), 1),
        (JournalInUseError("Journal j is open in pid 7."), 3),
    ],
)
@pytest.mark.asyncio
async def test_a_start_error_exits_1_and_a_journal_held_elsewhere_exits_3(
    opened: _Opened,
    capsys: pytest.CaptureFixture[str],
    raised: BaseException,
    exit_code: int,
) -> None:
    opened.raises = raised
    async with _relayed() as relay, _serving(relay) as (task, _):
        code = await _exited(task)

    assert code == exit_code
    assert str(raised) in capsys.readouterr().err


@pytest.mark.asyncio
async def test_a_session_that_cannot_continue_here_exits_1(
    opened: _Opened,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refused(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise NotResumableError("a claude part")

    monkeypatch.setattr(main_module, "session_journal", refused)
    async with _relayed() as relay, _serving(relay) as (task, _):
        code = await _exited(task)

    assert code == 1
    assert opened.flags == []


@pytest.mark.asyncio
async def test_at_a_terminal_the_screen_runs_and_its_quit_ends_the_process(
    opened: _Opened,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    quit_now = asyncio.Event()
    shown: list[tuple[Agent, dict[str, object]]] = []

    async def screen(agent: Agent, **kwargs: object) -> None:
        shown.append((agent, kwargs))
        _ = await quit_now.wait()

    gives_up: list[float | None] = []

    class Recorded(SessionSocket):
        def __init__(
            self,
            client: SocketClient,
            *,
            session_id: UUID,
            hello: Hello,
            give_up_sec: float | None = None,
        ) -> None:
            gives_up.append(give_up_sec)
            super().__init__(
                client,
                session_id=session_id,
                hello=hello,
                give_up_sec=give_up_sec,
            )

    monkeypatch.setattr(main_module, "run", screen)
    monkeypatch.setattr(main_module, "SessionSocket", Recorded)
    opened.show_thinking = True
    history = tmp_path / "history"
    args = ["--history", str(history)]
    async with (
        _relayed() as relay,
        _serving(relay, args=args, tui=True) as (task, server),
    ):
        _ = await relay.seen(_is("state"))
        assert opened.agent is not None
        opened.agent.session.stand(
            "note",
            UncategorizedRecord(kind="note", payload={"kept": True}),
        )
        quit_now.set()
        code = await _exited(task)

    assert code == 0
    # A quit halts the turn and uploads what the journal holds, as a stop does.
    assert opened.steps == ["halt"]
    assert "UncategorizedRecord" in [r.kind for r in server.stored]
    assert opened.interactive == [True]
    # A person at the screen ends it; a headless one gives up (exits 4).
    assert gives_up == [None]
    [(agent, given)] = shown
    assert agent is opened.agent
    # The screen draws the session's team, history, theme and reasoning as built.
    assert given["team"] is opened.team
    assert given["history"] == history
    assert given["show_thinking"] is True
    assert given["theme"] is opened.theme


@pytest.mark.asyncio
async def test_at_a_terminal_chat_and_the_screen_run_slash_lines_on_the_same_commands(
    opened: _Opened,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quit_now = asyncio.Event()
    printed: list[CommandOutput] = []
    drawn = _Screen()
    built: list[Commands] = []

    # As sagent's ``Views`` builds each view's commands: the root's, then a child's.
    async def screen(
        agent: Agent,
        *,
        catalog: ModelCatalog,
        commands: Callable[..., Commands],
        **kwargs: object,
    ) -> None:
        del kwargs
        child = Agent(model=ScriptedProvider()["scripted"])
        built.extend(
            commands(
                each,
                catalog=catalog,
                say=printed.append,
                screen=drawn,
                panes=Panes(),
            )
            for each in (agent, child)
        )
        _ = await quit_now.wait()

    monkeypatch.setattr(main_module, "run", screen)
    async with _relayed() as relay, _serving(relay, tui=True) as (task, _):
        _ = await relay.seen(_is("settings"))
        _ = await relay.send("Command", {"line": "/screen.thinking show"})
        from_chat = await relay.seen(_is("event", event="CommandOutput"))
        shown_from_chat = drawn.show_thinking
        root, child = built
        _ = await child("/runtime.retry off")
        _ = await root("/screen.thinking hide")
        typed = await relay.seen(
            lambda frame: frame.get("kind") == "CommandOutput" and frame != from_chat,
        )
        quit_now.set()
        code = await _exited(task)

    assert code == 0
    # Chat's line changed the screen that draws, and printed there as a typed one does.
    assert shown_from_chat
    assert [said.command for said in printed] == [
        "/screen.thinking",
        "/runtime.retry",
        "/screen.thinking",
    ]
    # Chat was told what its own line did and what the root's typed line did, and
    # nothing of a child's.
    assert "show" in json.dumps(from_chat["payload"])
    assert "hide" in json.dumps(typed["payload"])
    assert relay.every(_is("event", event="CommandOutput")) == [from_chat, typed]
    del opened


@pytest.mark.asyncio
async def test_a_halt_from_chat_halts_while_a_slash_line_runs_and_a_line_waits_its_turn(
    opened: _Opened,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started, release = asyncio.Event(), asyncio.Event()

    async def slow(self: Commands, line: str) -> bool:
        del self, line
        started.set()
        _ = await release.wait()
        return False

    monkeypatch.setattr(Commands, "__call__", slow)
    opened.replies = [reply("hello ada")]
    async with _relayed() as relay, _serving(relay) as (task, _):
        _ = await relay.seen(_is("settings"))
        _ = await relay.send("Command", {"line": "/login"})
        async with asyncio.timeout(5):
            _ = await started.wait()
        _ = await relay.send(
            "UserMessage",
            {"py/object": "UserMessage", "content": "hi", "extra": {}},
        )
        _ = await relay.send("Halt", {"py/object": "Halt"})
        _ = await relay.seen(_is("event", event="Halt"))
        halts_meanwhile = opened.halts
        # The halt was read after the line, which has not reached the agent.
        said_meanwhile = relay.found(_is("event", event="AgentToAgentMessage"))
        release.set()
        _ = await relay.seen(_is("event", event="AgentToAgentMessage"))
        await relay.close(CLOSE_TAKEN_OVER)
        _ = await _exited(task)

    assert (halts_meanwhile, said_meanwhile) == (1, None)


@pytest.mark.asyncio
async def test_a_slash_line_still_running_has_ended_before_what_it_uses_is_closed(
    opened: _Opened,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    steps: list[str] = []

    async def slow(self: Commands, line: str) -> bool:
        del self, line
        started.set()
        try:
            _ = await asyncio.Event().wait()
        finally:
            # A cleanup of several steps, as a command that ends its own tasks takes.
            for _step in range(5):
                await asyncio.sleep(0)
            steps.append("command ended")
        return False

    class Recorded(Publisher):
        @override
        async def aclose(self) -> None:
            steps.append("publisher closed")
            await super().aclose()

    monkeypatch.setattr(Commands, "__call__", slow)
    monkeypatch.setattr(main_module, "Publisher", Recorded)
    async with _relayed() as relay, _serving(relay) as (task, _):
        _ = await relay.seen(_is("settings"))
        _ = await relay.send("Command", {"line": "/login"})
        async with asyncio.timeout(5):
            _ = await started.wait()
        await relay.close(CLOSE_TAKEN_OVER)
        code = await _exited(task)

    assert (code, steps) == (3, ["command ended", "publisher closed"])
    del opened


@pytest.mark.asyncio
async def test_a_stop_ends_the_screen_too(
    opened: _Opened,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ended: list[bool] = []

    async def screen(agent: Agent, **kwargs: object) -> None:
        del agent, kwargs
        try:
            _ = await asyncio.Event().wait()
        finally:
            ended.append(True)

    monkeypatch.setattr(main_module, "run", screen)
    async with _relayed() as relay, _serving(relay, tui=True) as (task, _):
        _ = await relay.seen(_is("hello"))
        # A slash line that waits on commands this screen never builds ends with it.
        _ = await relay.send("Command", {"line": "/effort"})
        _ = await relay.send("Halt", {"py/object": "Halt"})
        _ = await relay.seen(_is("event", event="Halt"))
        await relay.close(CLOSE_TAKEN_OVER)
        code = await _exited(task)

    assert (code, ended) == (3, [True])
    del opened


@pytest.mark.asyncio
async def test_a_local_model_runs_a_local_tool_here(
    opened: _Opened,
    tmp_path: Path,
) -> None:
    here = tmp_path / "here.txt"
    here.write_text("here\n")
    ran = call("Read", "c1", file_path=str(here))
    opened.replies = [reply(calls=[ran]), reply("ran it")]
    opened.tools = [Read()]
    async with _relayed() as relay, _serving(relay) as (task, server):
        hello = Hello.model_validate(await relay.seen(_is("hello")))
        _ = await relay.seen(_is("settings"))
        assert opened.agent is not None
        assert await opened.agent.send("run it") == "ran it"
        results = [
            r for r in opened.agent.session.history() if isinstance(r, ToolResult)
        ]
        await relay.close(CLOSE_ENDED)
        _ = await _exited(task)

    assert hello.host == _HOST
    assert [type(r.result) for r in results] == [ReadResult]
    assert "here" in results[0].text()
    assert "ReadResult" in [r.kind for r in server.stored]
    assert server.asked == [None]


@pytest.mark.asyncio
async def test_a_local_model_runs_a_tool_on_another_host_with_remote_tools_only(
    opened: _Opened,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran = call(
        "RemoteTool",
        "c1",
        endpoint="lab-box",
        tool="Bash",
        arguments={"command": "hostname", "description": "name it"},
    )
    opened.replies = [reply(calls=[ran]), reply("it is lab-box")]
    opened.remote_tools_only = True
    server = _Server(version=1, hosts=[_host("lab-box"), _host(_HOST)])
    async with (
        _daemons(monkeypatch, "lab-box", answer=True) as daemons,
        _relayed() as relay,
        _serving(relay, server=server) as (task, _),
    ):
        _ = await relay.seen(_is("settings"))
        assert opened.remote is not None
        assert list(opened.remote.endpoints()) == ["lab-box"]
        endpoint = opened.remote.endpoints()["lab-box"]
        async with asyncio.timeout(5):
            await endpoint.until(lambda e: e.up)
        assert opened.agent is not None
        assert await opened.agent.send("which host?") == "it is lab-box"
        results = [
            r for r in opened.agent.session.history() if isinstance(r, ToolResult)
        ]
        await relay.close(CLOSE_ENDED)
        _ = await _exited(task)

    assert [c.tool for c in daemons.calls["lab-box"]] == ["Bash"]
    assert [asker for _, _, _, asker in daemons.dialed] == [None]
    assert "lab-box" in results[0].text()


def test_a_model_elsewhere_is_not_reached_from_here_yet(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--login-host", "desk"], profile=_PROFILE) == 1
    assert "no host answers another's model calls yet" in capsys.readouterr().err
    assert server.started == []


def _attached(state: str = "attached", *, process: UUID = _PROCESS) -> AgentProcess:
    return AgentProcess.model_validate(
        {
            "session_id": str(_SESSION),
            "process_id": str(process),
            "host": "vm",
            "state": state,
            "reason": "it crashed" if state == "detached" else "",
            "since": _AT,
        },
    )


@dataclass(slots=True, kw_only=True)
class _Served:
    """``serve``, recording how ``main`` called it."""

    calls: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    raises: BaseException | None = None

    async def __call__(self, args: Sequence[str], **kwargs: object) -> int:
        self.calls.append({"args": list(args), **kwargs})
        if self.raises is not None:
            raise self.raises
        return 0


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> _Served:
    stand_in = _Served()
    monkeypatch.setattr(main_module, "serve", stand_in)
    return stand_in


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Server]:
    fake = _Server(version=1)

    def client(url: str, *, author: str = "", api_key: str = "") -> _Server:
        del author
        fake.base_url, fake.api_key = url, api_key
        return fake

    monkeypatch.setattr(main_module, "Client", client)
    # A run started from no canvas, in a checkout the test names: set, so that what a
    # run writes of its placement is undone with the test.
    monkeypatch.setenv("TRAX_CANVAS", "")
    monkeypatch.setenv("TRAX_WORKSPACE", "")

    def checkout(cwd: Path) -> GitRef:
        del cwd
        return _CHECKOUT

    monkeypatch.setattr(placement, "git_ref", checkout)
    yield fake
    fake.change(closed=True)


def test_a_run_here_places_its_session_and_tells_its_tools_where(
    served: _Served,
    server: _Server,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started_from = uuid4()
    monkeypatch.setenv("TRAX_CANVAS", str(started_from))

    assert main(["--headless"], profile=_PROFILE) == 0

    assert server.placed == [
        (
            f"/api/sessions/{_SESSION}/place",
            {
                "canvas": str(started_from),
                "git": {"origin": _CHECKOUT.origin, "branch": _CHECKOUT.branch},
            },
        ),
    ]
    assert (os.environ["TRAX_WORKSPACE"], os.environ["TRAX_CANVAS"]) == (
        str(_WORKSPACE),
        str(_PLACED_ON),
    )
    assert len(served.calls) == 1


def test_a_run_the_server_will_not_place_runs_unplaced(
    served: _Served,
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.place_refused = True

    assert main(["--headless"], profile=_PROFILE) == 0

    assert len(served.calls) == 1
    assert (os.environ["TRAX_WORKSPACE"], os.environ["TRAX_CANVAS"]) == ("", "")
    assert "running unplaced" in capsys.readouterr().err


@pytest.mark.parametrize(
    "case",
    [([], True, True), (["--headless"], True, False), ([], False, False)],
    ids=["terminal", "headless-at-a-terminal", "no-terminal"],
)
def test_a_run_starts_a_session_here_with_a_screen_only_at_a_terminal(
    served: _Served,
    server: _Server,
    monkeypatch: pytest.MonkeyPatch,
    case: tuple[list[str], bool, bool],
) -> None:
    argv, isatty, tui = case
    stdin = io.StringIO()
    monkeypatch.setattr(stdin, "isatty", lambda: isatty)
    monkeypatch.setattr("sys.stdin", stdin)

    assert main([*argv, "--model", "opus"], profile=_PROFILE) == 0

    (called,) = served.calls
    assert called["args"] == ["--model", "opus"]
    assert (called["session_id"], called["tui"], called["host"]) == (
        _SESSION,
        tui,
        _HOST,
    )
    # A person's own run is no remote agent: it starts sagent's own daemon.
    assert called["remote"] is False
    assert called["client"] is server
    assert server.api_key == "k"


def test_a_session_named_is_continued_here(served: _Served, server: _Server) -> None:
    other = uuid4()

    assert main(["--session", str(other), "--headless"], profile=_PROFILE) == 0

    assert served.calls[0]["session_id"] == other
    del server


@pytest.mark.parametrize(
    "picked",
    [
        ["--resume"],
        ["--continue"],
        ["--resume-all"],
        ["--continue-all"],
        ["--ephemeral"],
        ["--hosted-session", "x/session.jsonl"],
    ],
)
def test_a_flag_that_picks_a_journal_is_refused(
    served: _Served,
    capsys: pytest.CaptureFixture[str],
    picked: list[str],
) -> None:
    with pytest.raises(SystemExit) as refused:
        _ = main(picked, profile=_PROFILE)

    assert refused.value.code == 2
    assert f"drop {picked[0]}" in capsys.readouterr().err
    assert served.calls == []


@pytest.mark.parametrize("remote", [False, True])
def test_a_daemons_start_serves_the_session_it_hands_with_its_key(
    served: _Served,
    server: _Server,
    monkeypatch: pytest.MonkeyPatch,
    *,
    remote: bool,
) -> None:
    """A start the server placed for a remote agent has no host where it runs."""
    handed = ServeSession(
        session_id=_SESSION,
        process_id=_PROCESS,
        token=str(uuid4()),
        spec=ProcessSpec(host="vm"),
        remote=remote,
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(handed.model_dump_json() + "\n"))

    code = main(
        [
            "--serve",
            "https://box.example",
            "--time",
            "120",
            "--model",
            "opus",
        ],
        profile=None,
    )

    assert code == 0
    (called,) = served.calls
    assert called["args"] == ["--model", "opus"]
    assert called["time_limit"] == timedelta(hours=2)
    assert (called["session_id"], called["process_id"]) == (_SESSION, _PROCESS)
    assert (called["host"], called["tui"]) == ("vm", False)
    assert called["remote"] is remote
    assert (server.base_url, server.api_key) == ("https://box.example", handed.token)


def test_a_daemons_start_with_no_session_on_stdin_exits_1(
    served: _Served,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("not a session\n"))

    assert main(["--serve", "https://box.example"], profile=None) == 1
    assert served.calls == []


@pytest.mark.parametrize("argv", [["--headless"], ["--on", "vm"]])
def test_a_session_the_server_will_not_start_exits_1(
    served: _Served,
    server: _Server,
    argv: list[str],
) -> None:
    server.start_refused = True

    assert main(argv, profile=_PROFILE) == 1
    assert (served.calls, server.started) == ([], [])


def test_a_run_with_no_profile_and_no_daemon_exits_1(served: _Served) -> None:
    assert main([], profile=None) == 1
    assert served.calls == []


@pytest.mark.parametrize("raised", [KeyboardInterrupt(), asyncio.CancelledError()])
def test_ctrl_c_or_sigterm_exits_130(
    served: _Served,
    server: _Server,
    raised: BaseException,
) -> None:
    served.raises = raised

    assert main(["--headless"], profile=_PROFILE) == 130
    del server


@pytest.mark.parametrize(
    "case",
    [
        (_attached(), [], 0),
        (
            _attached("starting"),
            [
                ProcessLine(process=_attached(process=uuid4())),
                ProcessLine(process=_attached()),
            ],
            0,
        ),
        (_attached("starting"), [ProcessLine(process=_attached("detached"))], 1),
        (_attached("starting"), [], 1),
    ],
    ids=["attached-at-once", "attached-later", "detached", "never"],
)
def test_on_starts_it_through_a_daemon_and_waits_for_it_to_attach(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    case: tuple[AgentProcess, list[ViewerFrame], int],
) -> None:
    answer, frames, exit_code = case
    monkeypatch.setattr(main_module, "_ATTACHED_SEC", 0.05)
    server.answer = answer
    server.frames = [Opened(), *frames]

    code = main(
        ["--on", "vm", "--cwd", "~/src", "--login-host", "desk", "--model", "opus"],
        profile=_PROFILE,
    )

    assert code == exit_code
    assert server.started == [
        ProcessSpec(
            host="vm",
            login_host="desk",
            cwd="~/src",
            args=("--model", "opus"),
        ),
    ]
    assert (capsys.readouterr().out == f"{_SESSION}\n") == (exit_code == 0)
    # Its checkout is on the host it starts on, which this command cannot read.
    assert server.placed == []


def test_on_with_no_name_starts_it_where_the_server_runs_agents(
    server: _Server,
) -> None:
    server.answer = _attached()
    server.frames = [Opened()]

    assert main(["--on"], profile=_PROFILE) == 0
    assert server.started == [ProcessSpec()]


def test_on_hands_the_time_limit_to_the_process_it_starts_as_its_own_flag(
    server: _Server,
) -> None:
    server.answer = _attached()
    server.frames = [Opened()]

    assert (
        main(["--on", "vm", "--time", "90", "--model", "opus"], profile=_PROFILE) == 0
    )
    assert server.started == [
        ProcessSpec(host="vm", args=("--time", "0-01:30:00", "--model", "opus")),
    ]


@pytest.mark.parametrize(
    ("given", "limit"),
    [
        ("120", timedelta(minutes=120)),
        ("90:30", timedelta(minutes=90, seconds=30)),
        ("2:05:07", timedelta(hours=2, minutes=5, seconds=7)),
        ("1-0", timedelta(days=1)),
        ("1-2:30", timedelta(days=1, hours=2, minutes=30)),
        ("1-2:30:15", timedelta(days=1, hours=2, minutes=30, seconds=15)),
    ],
    ids=["M", "M:S", "H:M:S", "D-H", "D-H:M", "D-H:M:S"],
)
def test_a_time_limit_reads_as_slurms_time_does(
    served: _Served,
    server: _Server,
    given: str,
    limit: timedelta,
) -> None:
    assert main(["--headless", "--time", given], profile=_PROFILE) == 0
    assert main(["--headless"], profile=_PROFILE) == 0
    assert [call["time_limit"] for call in served.calls] == [limit, None]
    del server


@pytest.mark.parametrize(
    "given",
    ["2h", "1:2:3:4", "1-2:3:4:5", "-30", "1-", "", "0", "0-0:00"],
    ids=[
        "a unit",
        "four fields",
        "days and four",
        "no days",
        "no hours",
        "empty",
        "zero",
        "zero days",
    ],
)
def test_a_time_limit_slurm_would_not_read_or_of_no_time_is_refused(
    served: _Served,
    capsys: pytest.CaptureFixture[str],
    given: str,
) -> None:
    with pytest.raises(SystemExit) as refused:
        _ = main(["--headless", "--time", given], profile=_PROFILE)

    assert refused.value.code == 2
    said = capsys.readouterr().err
    assert (
        f"argument --time: {given!r} is no time limit, such as 120, 2:00:00 or 1-0"
        in said
    )
    assert served.calls == []


def test_help_names_each_of_knowops_own_flags_with_what_it_takes_and_does(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLUMNS", "200")

    with pytest.raises(SystemExit) as exited:
        _ = main(["--help"], profile=None)

    help_text = capsys.readouterr().out
    options = help_text[help_text.index("\noptions:\n") :].splitlines()[2:]
    assert exited.value.code == 0
    assert [" ".join(line.split()) for line in options] == [
        "-h, --help show this help message and exit",
        "--headless Run with no screen, at a terminal too.",
        "--session SESSION The session to continue here.",
        "--on [NAME] Start on host NAME through its daemon; with no name, the server's.",
        "--cwd DIR With --on, the folder there to start in (default: ~).",
        "--login-host NAME Use host NAME's model login.",
        "--stop ID Stop session ID's agent, wherever it runs, and exit.",
        "--time TIME End TIME after it starts, as a stop ends it; Slurm's --time: 120, 2:00:00 or 1-0.",
        "--serve URL Serve the session a daemon hands on stdin, through the server at URL.",
    ]


def test_stop_stops_the_sessions_agent_wherever_it_runs_and_starts_none(
    served: _Served,
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.stop_answer = _attached("detached")

    assert main(["--stop", str(_SESSION)], profile=_PROFILE) == 0
    assert server.stopped == [_SESSION]
    assert capsys.readouterr().out == f"stopped {_SESSION} on vm\n"
    assert (served.calls, server.started, server.placed) == ([], [], [])


@pytest.mark.parametrize(
    ("given", "named"),
    [
        (["--on", "vm"], "--on"),
        (["--session", str(_SESSION)], "--session"),
        (["--headless"], "--headless"),
        (["--time", "90"], "--time"),
        (["--cwd", "~"], "--cwd"),
        (["--model", "opus", "--login-host=desk"], "--model, --login-host"),
        (["--sto", "x"], "--sto"),
    ],
    ids=[
        "on",
        "session",
        "headless",
        "time",
        "cwd-at-its-default",
        "sagents-and-own",
        "no-abbreviation",
    ],
)
def test_stop_refuses_any_other_flag_and_stops_nothing(
    served: _Served,
    server: _Server,
    capsys: pytest.CaptureFixture[str],
    given: list[str],
    named: str,
) -> None:
    with pytest.raises(SystemExit) as refused:
        _ = main(["--stop", str(_SESSION), *given], profile=_PROFILE)

    assert refused.value.code == 2
    assert capsys.readouterr().err.endswith(
        f"\nknowop run: error: --stop takes no other flag: {named}\n",
    )
    assert (server.stopped, served.calls) == ([], [])


@pytest.mark.parametrize(
    ("answer", "said"),
    [
        (
            ClientError("DELETE process -> 403: not yours", status_code=403),
            "DELETE process -> 403: not yours",
        ),
        (None, f"session {_SESSION} has had no agent process"),
    ],
    ids=["refused", "never-ran"],
)
def test_a_stop_the_server_refuses_or_of_a_session_that_never_ran_exits_1(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
    answer: ClientError | None,
    said: str,
) -> None:
    server.stop_answer = answer

    assert main(["--stop", str(_SESSION)], profile=_PROFILE) == 1
    printed = capsys.readouterr()
    assert (printed.out, printed.err) == ("", f"Error: {said}\n")


def test_on_whose_lines_fail_exits_1_and_starts_nothing(server: _Server) -> None:
    server.answer = _attached()
    server.lines_refused = True

    assert main(["--on", "vm"], profile=_PROFILE) == 1
    assert server.started == []


@pytest.mark.asyncio
async def test_first_done_returns_the_first_to_finish_and_cancels_the_rest() -> None:
    cancelled: list[bool] = []

    async def slow() -> Ended:
        try:
            _ = await asyncio.Event().wait()
        finally:
            cancelled.append(True)
        return Ended(code=9)

    async def quick() -> Ended:
        return Ended(code=0)

    assert await first_done(slow(), quick()) == Ended(code=0)
    assert cancelled == [True]


def test_each_call_a_process_makes_is_a_route_its_key_reaches() -> None:
    """A process key is refused on every route the server does not list for it.

    Each request a ``ServeClient`` call sends is recorded off a real ``Client``. Its
    server and key are dialed besides: the session's socket and the tool connections.
    """
    sent: list[tuple[str, str]] = []

    def refused(request: httpx2.Request) -> httpx2.Response:
        sent.append((request.method, request.url.path))
        return httpx2.Response(403)

    session_id = UUID(int=1)
    calls: dict[str, Callable[[Client], object]] = {
        "read_session_parts": lambda c: c.read_session_parts(session_id),
        "read_session_records": lambda c: c.read_session_records(session_id),
        "append_records": lambda c: c.append_records(session_id),
        "terminal_hosts": lambda c: c.terminal_hosts(),
        "hosts_version": lambda c: c.hosts_version(since=None, wait_sec=0),
        "message_agent": lambda c: c.message_agent(
            SessionAgent(label="lead", session=412),
            "hi",
            from_session=session_id,
            from_label="Agent",
            booked="ask",
        ),
    }
    with Client(
        "https://trax.test",
        api_key="k",
        transport=httpx2.MockTransport(refused),
    ) as client:
        for call in calls.values():
            with pytest.raises(ClientError):
                _ = call(client)

    members = {name for name in dir(ServeClient) if not name.startswith("_")}
    routes = [
        *sent,
        (WEBSOCKET, SESSION_PROCESS_WS_PATH),
        (WEBSOCKET, TOOLS_WS_PATH),
    ]
    assert members == {*calls, "base_url", "api_key"}
    assert len(sent) == len(calls)
    assert all(
        any(
            method == listed and re.fullmatch(_template_pattern(path), url)
            for listed, path in PROCESS_KEY_ROUTES
        )
        for method, url in routes
    ), routes


def _template_pattern(path: str) -> str:
    """Return a pattern that each URL a route's template names matches in full."""
    return re.sub(r"\\\{[a-z_]+\\\}", "[^/]+", re.escape(path))


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
