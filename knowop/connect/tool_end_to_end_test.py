"""A real ``Endpoint`` reaches a real ToolServing through Trackinizer's relay.

The whole path, over loopback sockets: an ``Endpoint`` dials with ``dial_tools``; a
fake server routes each socket into Trackinizer's real ``ToolRelay``, which reads each line
with its real ``ToolLines``; ``ToolServing`` holds a multiplexed websocket and hands
each tool stream to ``Server.adopt``. The fake stands in for the routes and database,
which Trackinizer's own route tests cover. Every daemon runs in a
``LocalHost``, in this process: ``tool_serving_test.py`` covers the real child.

It lives here, not beside the relay, as trackinizer may not import ``knowop``.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, cast, override
from urllib.parse import parse_qs, unquote, urlsplit

import asyncio
import os
import re
import resource
import shlex
import threading
import time
import uuid

from sagent.agent.agent import Agent
from sagent.host.testing import Gate, LocalHost
from sagent.host.wire import (
    MAX_LINE_BYTES,
    PING_SEC,
    Call,
    Hello,
    Result,
    Welcome,
    encoded,
    received,
)
from sagent.model.scripted import (
    ScriptedProvider,
    call as scripted_call,
    reply,
)
from sagent.session.jsonl import JsonlJournal, read_record
from sagent.session.types import (
    BashResult,
    ReadResult,
    RemoteToolResult,
    ToolMessage,
    UncategorizedToolResult,
)
from sagent.tools.bash import Bash
from sagent.tools.display import ToolDisplay
from sagent.tools.edit import Edit
from sagent.tools.glob_tool import Glob
from sagent.tools.grep import Grep
from sagent.tools.list import List
from sagent.tools.read import Read
from sagent.tools.remote import Endpoint, Link, RemoteSession, RemoteTool
from sagent.tools.types import ToolOutput, whole_output
from sagent.tools.write import Write
from trackinizer.client.tool_bridge import dial_tools
from trackinizer.server.config import Config
from trackinizer.server.terminals import SocketClosedError
from trackinizer.server.tool_lines import (
    GatedCall,
    ToolLines,
    Verdict,
    authorize,
    refused_line,
)
from trackinizer.server.tool_relay import (
    ToolConnectionBusyError,
    ToolConnectionEndError,
    ToolConnectionUnavailableError,
    ToolRelay,
)
from trackinizer.trax.connection import server_dir
from trackinizer.wire import wire_terminals
from trackinizer.wire.wire_tools import (
    HOST_TOOLS_MUX_PATH,
    MUX_CALLS,
    TOOLS_WS_PATH,
    ToolMux,
)
from treekle import PlainTree
from websockets.asyncio.client import connect as connect_socket
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

import pytest

from knowop.connect import tool_serving
from knowop.connect.tool_serving import ToolServing, tool_dirs


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Iterator
    from pathlib import Path

    from sagent.host.server import Server
    from sagent.host.wire import Streams
    from sagent.tools.types import Tool, ToolContext, ToolStreamingResult
    from trackinizer.server.tool_lines import Authorize
    from trackinizer.server.tool_relay import Inbound, ToolConnection
    from trackinizer.wire.wire_tools import MuxStream
    from websockets.asyncio.server import (
        Server as WebsocketServer,
        ServerConnection,
    )


pytestmark = pytest.mark.asyncio

_ADA: Final = "ada@x.y"
_NAME: Final = "gx10"
_KEY: Final = uuid.UUID(int=1)
_CREDENTIAL: Final = "trax_machine_" + "0" * 32 + "_" + "c" * 43

_LIMIT_SEC: Final = 30.0
"""How long anything awaited here may take: long enough for a loaded box."""

_RETRY_SEC: Final = 0.005
"""How long an agent or daemon waits to try again, never backing off past twice that."""

_AGENT_PATH: Final = re.compile(TOOLS_WS_PATH.format(name="(?P<name>[^/]+)"))
_MUX_PATH: Final = re.compile(HOST_TOOLS_MUX_PATH.format(host_id="(?P<host>[^/]+)"))


@pytest.fixture(autouse=True)
def _empty_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep what a real tool reads or writes under home off the developer's own."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


@dataclass(frozen=True, slots=True, kw_only=True)
class _Host:
    name: str
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    instance: uuid.UUID = field(default_factory=uuid.uuid4)


class _Socket:
    """A websocket the relay serves, as the relay's ``ToolSocket``."""

    def __init__(self, websocket: ServerConnection) -> None:
        self._socket = websocket

    async def send_bytes(self, data: bytes) -> None:
        try:
            await self._socket.send(data)
        except ConnectionClosed as error:
            raise SocketClosedError from error

    async def close(self, code: int, reason: str = "") -> None:
        await self._socket.close(code, reason)

    async def read(self) -> bytes | None:
        try:
            return await self._socket.recv(decode=False)
        except ConnectionClosed:
            return None


class _Relay:
    """Trackinizer's relay without its routes: a real relay and line reader on a port.

    Every host it knows is the asker's, confirmed and serving tools, until a test
    takes it out of ``allowed``.
    """

    def __init__(
        self,
        *,
        line_bytes: int,
        clock: Callable[[], float],
        agent_sockets: int,
        streams: int,
    ) -> None:
        self.port = 0
        self.line_bytes = line_bytes
        self.clock = clock
        self.agent_socket_limit = agent_sockets
        self.stream_limit = streams
        self.mux: dict[uuid.UUID, ToolMux] = {}
        self.hosts: dict[str, _Host] = {}
        self.allowed: set[uuid.UUID] = set()
        self.calls: list[GatedCall] = []
        """Each call the relay's gate judged, oldest first, as its log line says."""

        self.authorize: Authorize = authorize
        self.relay: ToolRelay | None = None
        self._server: WebsocketServer | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> None:
        """Serve on this relay's port, or on a new one the first time, with a new relay."""
        self.relay = ToolRelay(
            alive=self._alive,
            clock=self.clock,
            line_bytes=self.line_bytes,
            agent_sockets_per_host=self.agent_socket_limit,
            streams_per_host=self.stream_limit,
        )
        self._server = await serve(self._handle, "127.0.0.1", self.port)
        self.port = cast(int, next(iter(self._server.sockets)).getsockname()[1])

    async def stop(self) -> None:
        """Stop serving, as a restart does: every connection goes, and the relay with it."""
        assert self._server is not None
        self.relay = None
        self._server.close()
        await self._server.wait_closed()

    def drop_connections(self) -> None:
        """Cut every connection with no goodbye, as a network does; the server stays."""
        assert self._server is not None
        for connection in tuple(self._server.connections):
            connection.transport.abort()

    def agent_sockets(self) -> set[ServerConnection]:
        assert self._server is not None
        return {
            connection
            for connection in self._server.connections
            if connection.request is not None
            and _AGENT_PATH.fullmatch(urlsplit(connection.request.path).path)
        }

    def host(self, name: str = _NAME) -> _Host:
        """Know a host of the asker's, confirmed and serving tools."""
        found = _Host(name=name)
        self.hosts[name] = found
        self.allowed.add(found.id)
        return found

    async def _alive(self, connection: ToolConnection) -> bool:
        return connection.host_id in self.allowed

    async def _handle(self, websocket: ServerConnection) -> None:
        request = websocket.request
        assert request is not None
        url = urlsplit(request.path)
        query = parse_qs(url.query)
        socket = _Socket(websocket)
        relay = self.relay
        assert relay is not None
        if (agent := _AGENT_PATH.fullmatch(url.path)) is not None:
            await self._agent(
                relay,
                socket,
                unquote(agent["name"]),
                asker=query["asker"][0],
            )
        elif (mux := _MUX_PATH.fullmatch(url.path)) is not None:
            host_id = uuid.UUID(mux["host"])
            transport = ToolMux(websocket.send)
            self.mux[host_id] = transport
            try:
                await relay.serve_mux(
                    host_id,
                    uuid.UUID(query["instance"][0]),
                    transport,
                    socket.read,
                )
            finally:
                if self.mux.get(host_id) is transport:
                    del self.mux[host_id]
        else:
            await socket.close(wire_terminals.CLOSE_REFUSED, "no such route")

    async def _agent(
        self,
        relay: ToolRelay,
        socket: _Socket,
        name: str,
        *,
        asker: str,
    ) -> None:
        host = self.hosts[name]
        if host.id not in self.allowed:
            await socket.close(wire_terminals.CLOSE_REFUSED, f"{name} is not confirmed")
            return
        try:
            connection = relay.join(
                host.id,
                agent=socket,
                asker=asker,
                api_key_id=_KEY,
                inbound=self._gated(relay, asker=asker, host=name),
            )
        except (ToolConnectionBusyError, ToolConnectionUnavailableError) as error:
            await socket.close(wire_terminals.CLOSE_REFUSED, str(error))
            return
        await relay.serve_agent(connection, socket.read)

    def _gated(self, relay: ToolRelay, *, asker: str, host: str) -> Inbound:
        """Read the agent's lines as the agent route does, keeping each call judged."""
        lines = ToolLines(
            asker=asker,
            host=host,
            authorize=self._judged,
            line_bytes=self.line_bytes,
        )

        async def inbound(
            connection: ToolConnection,
            frame: bytes,
        ) -> tuple[bytes, bytes]:
            try:
                routed, calls = lines.feed(frame)
            except ValueError as error:
                reason = f"the server will not relay a line: {error}"
                raise ToolConnectionEndError(reason) from error
            self.calls.extend(calls)
            for call in calls:
                if call.outcome == "allow":
                    relay.called(connection, call.call_id)
            for call_id in routed.acks:
                relay.acked(connection, call_id)
            return routed.to_daemon, routed.to_agent

        return inbound

    def _judged(self, call: GatedCall) -> Verdict:
        """Judge with the rule set now, which a test may change once a tool connection is open."""
        return self.authorize(call)


@asynccontextmanager
async def _serving(
    *,
    line_bytes: int = Config().tool_line_bytes,
    clock: Callable[[], float] = time.monotonic,
    agent_sockets: int = Config().tool_agent_sockets_per_host,
    streams: int = Config().tool_streams_per_host,
) -> AsyncGenerator[_Relay]:
    """Run a relay and verify that its agent connections are cleaned up."""
    relay = _Relay(
        line_bytes=line_bytes,
        clock=clock,
        agent_sockets=agent_sockets,
        streams=streams,
    )
    await relay.start()
    try:
        yield relay
        await _until(lambda: relay.relay is None or not relay.relay._connections)
    finally:
        if relay.relay is not None:
            await relay.stop()


@dataclass(frozen=True, slots=True, kw_only=True)
class _Gated(Gate):
    """A ``Gate`` opened from outside the daemon's thread, which owns its event loop."""

    loops: list[asyncio.AbstractEventLoop] = field(
        default_factory=list[asyncio.AbstractEventLoop],
    )

    # ``call`` is invoked on the daemon's event loop, the loop ``open`` must wake.
    @override
    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[UncategorizedToolResult, str, str]:
        self.loops.append(asyncio.get_running_loop())
        return Gate.call(self, context, arguments)

    def open(self) -> None:
        _ = self.loops[-1].call_soon_threadsafe(self.opened.set)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Echo:
    """A tool that answers with the text it was given."""

    name = "Echo"
    description = "Say the text back."
    schema: Mapping[str, PlainTree] = field(default_factory=lambda: {"type": "object"})
    display: ToolDisplay = field(default=ToolDisplay())

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        del arguments
        return "Echo"

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
    ) -> ToolStreamingResult[UncategorizedToolResult, str, str]:
        del context
        return whole_output(self._run(str(arguments["text"])))

    async def _run(self, text: str) -> ToolOutput[UncategorizedToolResult, str, str]:
        return ToolOutput(stdout=(text,), result=UncategorizedToolResult())


def _local(monkeypatch: pytest.MonkeyPatch, **tools: Tool) -> LocalHost:
    """Serve ``tools`` from sagent's host in ToolServing's own event loop."""
    sagent_host = LocalHost(tools=tools)
    monkeypatch.setattr(tool_serving, "scrubbed_host", sagent_host)
    return sagent_host


@asynccontextmanager
async def _hosting(
    relay: _Relay,
    host: _Host,
    *,
    url: str = "",
) -> AsyncGenerator[ToolServing]:
    """Run a ``ToolServing`` for ``host``, retrying promptly after a disconnect."""
    url = url or relay.url
    daemon = ToolServing(
        url,
        credential=_CREDENTIAL,
        name=host.name,
        dirs=tool_dirs(host.name, server=url),
        retry_sec=_RETRY_SEC,
    )
    daemon.start(host.id, host.instance)
    try:
        yield daemon
    finally:
        await asyncio.to_thread(daemon.stop)


def _link(
    relay: _Relay,
    *,
    host: str = _NAME,
    dial_allowed: asyncio.Event | None = None,
) -> Link:
    async def dial() -> Streams:
        if dial_allowed is not None:
            await dial_allowed.wait()
        return await dial_tools(
            relay.url,
            api_key="assistant-key",
            host=host,
            asker=_ADA,
            limit=MAX_LINE_BYTES,
        )

    return Link(dial=dial, name=host, where="relay")


@asynccontextmanager
async def _agent(
    *links: Link,
    ping_sec: float = PING_SEC,
    journal: Path | None = None,
) -> AsyncGenerator[RemoteSession]:
    async with RemoteSession(
        links,
        journal=journal,
        client="laptop",
        retry_sec=_RETRY_SEC,
        ping_sec=ping_sec,
        max_retry_sec=2 * _RETRY_SEC,
    ) as session:
        yield session


async def _until(done: Callable[[], bool]) -> None:
    deadline = time.monotonic() + _LIMIT_SEC
    while not done():
        assert time.monotonic() < deadline, "the awaited change never came"
        # Polls state the daemon's threads change; nothing signals it.
        await asyncio.sleep(0.005)  # house-ignore[sleep] -- See above.


async def _up(session: RemoteSession, name: str = _NAME) -> Endpoint:
    endpoint = session.endpoints()[name]
    async with asyncio.timeout(_LIMIT_SEC):
        await endpoint.until(lambda each: each.up)
    return endpoint


async def _said(endpoint: Endpoint, text: str) -> None:
    """Return once the endpoint's error holds ``text``."""
    async with asyncio.timeout(_LIMIT_SEC):
        await endpoint.until(lambda each: text in each.error)


def _failed(result: ToolMessage) -> str:
    """Return the error text of a failed ``RemoteTool`` call."""
    assert isinstance(result.result, RemoteToolResult), result
    assert result.is_error
    return result.text()


def _text(result: ToolMessage) -> str:
    """Return the text of a ``Read`` call's result."""
    assert isinstance(result.result, ReadResult), result
    return result.text()


async def _call(
    endpoint: Endpoint,
    tool: str,
    *,
    agent: str = "Agent",
    **arguments: PlainTree,
) -> ToolMessage:
    async with asyncio.timeout(_LIMIT_SEC):
        return await endpoint.call(
            agent=agent,
            window="w1",
            tool=tool,
            arguments=arguments,
        )


def _running(
    endpoint: Endpoint,
    tool: str,
    **arguments: PlainTree,
) -> asyncio.Task[ToolMessage]:
    return asyncio.create_task(
        endpoint.call(agent="Agent", window="w1", tool=tool, arguments=arguments),
    )


# -- The brief's path --------------------------------------------------------------------


async def test_a_real_endpoint_reaches_a_real_tool_serving(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _ = _local(monkeypatch, Read=Read())
    readme = tmp_path / "readme.txt"
    readme.write_text("from this host\n")
    async with (
        _serving() as relay,
        _hosting(relay, relay.host()),
        _agent(_link(relay)) as session,
    ):
        endpoint = await _up(session)
        spec = await endpoint.spec("Read")
        assert spec is not None
        assert spec.digest == endpoint.tools["Read"]
        result = await _call(endpoint, "Read", file_path=str(readme))
    assert "from this host" in _text(result)
    (call,) = relay.calls
    assert (call.asker, call.agent, call.host, call.tool, call.outcome) == (
        _ADA,
        "Agent@laptop",
        _NAME,
        "Read",
        "allow",
    )
    assert call.session == session.session_id
    # The id of the agent's ``Call`` line, as the agent wrote it.
    assert re.fullmatch(r"[0-9a-f]{32}", call.call_id)


async def test_a_calls_session_record_names_it_by_the_id_the_relay_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """What the agent ran is in its session, joined to the relay's log by session and id."""
    _ = _local(monkeypatch, Read=Read())
    readme = tmp_path / "readme.txt"
    readme.write_text("from this host\n")
    asked = scripted_call(
        "RemoteTool",
        "toolu_1",
        endpoint=_NAME,
        tool="Read",
        arguments={"file_path": str(readme)},
    )
    provider = ScriptedProvider(replies=[reply(calls=[asked]), reply("read")])
    path = tmp_path / "s1" / "session.jsonl"
    async with (
        _serving() as relay,
        _hosting(relay, relay.host()),
        JsonlJournal(path) as journal,
        _agent(_link(relay), journal=path) as session,
    ):
        _ = await _up(session)
        async with Agent(
            model=provider["scripted"],
            tools=[RemoteTool(remote=session)],
            journal=journal,
            environment=False,
        ) as agent:
            async with asyncio.timeout(_LIMIT_SEC):
                assert await agent.send("read it") == "read"
    (record,) = [
        r
        for r in map(read_record, path.read_bytes().splitlines())
        if isinstance(r, ToolMessage) and isinstance(r.result, ReadResult)
    ]
    (call,) = relay.calls
    assert "from this host" in _text(record)
    assert (record.call_id, call.session, call.call_id) == (
        "toolu_1",
        "s1",
        record.wire_id,
    )


@pytest.mark.parametrize("drop", ["restart", "cut"])
async def test_a_call_survives_the_server_dropping(
    monkeypatch: pytest.MonkeyPatch,
    drop: str,
) -> None:
    gate = _Gated()
    daemon = _local(monkeypatch, Bash=gate)
    async with (
        _serving() as relay,
        _hosting(relay, relay.host()),
        _agent(_link(relay)) as session,
    ):
        endpoint = await _up(session)
        call = _running(endpoint, "Bash")
        await _until(gate.started.is_set)

        if drop == "restart":
            await relay.stop()
            await _until(lambda: not endpoint.up)
            gate.open()
            await relay.start()
        else:
            relay.drop_connections()
            await _until(lambda: not endpoint.up)
            gate.open()

        async with asyncio.timeout(_LIMIT_SEC):
            result = await call
    assert (result.state, result.result) == ("done", UncategorizedToolResult())
    assert result.text() == "open for Agent@laptop"
    assert len(daemon.started) == 1
    assert len(relay.calls) == 1


async def test_two_sessions_on_one_host_get_their_own_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _Gated()
    daemon = _local(monkeypatch, Bash=gate)
    async with (
        _serving() as relay,
        _hosting(relay, relay.host()),
        _agent(_link(relay)) as first,
        _agent(_link(relay)) as second,
    ):
        calls = [
            asyncio.create_task(_call(await _up(session), "Bash", agent=agent))
            for session, agent in ((first, "One"), (second, "Two"))
        ]
        await _until(
            lambda: sum(not worker.task.done() for worker in daemon.started) == 2,
        )
        await _until(gate.started.is_set)
        gate.open()
        one, two = await asyncio.gather(*calls)
    assert (one.state, one.result) == ("done", UncategorizedToolResult())
    assert (two.state, two.result) == ("done", UncategorizedToolResult())
    assert one.text() == "open for One@laptop"
    assert two.text() == "open for Two@laptop"
    assert {call.session for call in relay.calls} == {
        first.session_id,
        second.session_id,
    }


@pytest.fixture
def _relay_test_fd_budget() -> Iterator[None]:
    # This harness runs the relay, daemon, and every agent in one process.
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    needed = 4096
    if hard != resource.RLIM_INFINITY and hard < needed:
        pytest.skip("the single-process harness needs a larger descriptor budget")
    raised = soft != resource.RLIM_INFINITY and soft < needed
    if raised:
        resource.setrlimit(resource.RLIMIT_NOFILE, (needed, hard))
    try:
        yield
    finally:
        if raised:
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


@pytest.mark.usefixtures("_relay_test_fd_budget")
async def test_more_than_32_activated_agents_keep_native_state_on_one_mux(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _ = _local(monkeypatch, Read=Read())
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("from the daemon\n")
    now = 0.0
    async with (
        _serving(clock=lambda: now, streams=40) as relay,
        _hosting(relay, host := relay.host()),
        AsyncExitStack() as stack,
    ):
        sessions = [
            await stack.enter_async_context(_agent(_link(relay))) for _ in range(33)
        ]
        endpoints = await asyncio.gather(*(_up(session) for session in sessions))
        transport = relay.mux[host.id]
        sockets = relay.agent_sockets()
        assert len(sockets) == len(transport._streams) == 33
        result = await _call(endpoints[0], "Read", file_path=str(evidence))
        assert "from the daemon" in _text(result)
        assert "from the daemon" in _text(
            await _call(endpoints[-1], "Read", file_path=str(evidence)),
        )

        now += 60
        assert relay.relay is not None
        await relay.relay.sweep()
        assert relay.agent_sockets() == sockets
        assert relay.mux[host.id] is transport
        assert len(transport._streams) == 33
        assert "unchanged since last read" in _text(
            await _call(endpoints[0], "Read", file_path=str(evidence)),
        )


@pytest.mark.usefixtures("_relay_test_fd_budget")
@pytest.mark.parametrize(
    "limit",
    [4, Config().tool_agent_sockets_per_host],
    ids=["small", "default"],
)
async def test_full_agent_socket_budget_recovers_without_interrupting_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    limit: int,
) -> None:
    gate = _Gated()
    _local(monkeypatch, Bash=gate, Read=Read())
    path = tmp_path / "evidence.txt"
    path.write_text("after saturation\n")
    async with (
        _serving(agent_sockets=limit, streams=2) as relay,
        _hosting(relay, host := relay.host()),
        _agent(_link(relay)) as active,
        AsyncExitStack() as idle,
    ):
        endpoint = await _up(active)
        physical = relay.mux[host.id]
        running = _running(endpoint, "Bash")
        await _until(gate.started.is_set)
        socket_url = (
            relay.url.replace("http:", "ws:")
            + TOOLS_WS_PATH.format(name=host.name)
            + "?asker="
            + _ADA
        )
        parked = [
            await idle.enter_async_context(connect_socket(socket_url))
            for _ in range(limit - 1)
        ]
        async with connect_socket(socket_url) as overflow:
            with pytest.raises(ConnectionClosed) as closed:
                _ = await overflow.recv()
            assert closed.value.rcvd is not None
            assert closed.value.rcvd.code == wire_terminals.CLOSE_REFUSED
            assert f"{limit} tool connections" in closed.value.rcvd.reason
        await parked[0].close()
        await _until(
            lambda: (
                relay.relay is not None and len(relay.relay._connections) == limit - 1
            ),
        )
        async with _agent(_link(relay)) as fresh:
            answer = await _call(await _up(fresh), "Read", file_path=str(path))
            assert "after saturation" in _text(answer)
            assert not running.done()
            assert relay.mux[host.id] is physical
        gate.open()
        assert (await running).state == "done"


@pytest.mark.cli_bash
@pytest.mark.usefixtures("_relay_test_fd_budget")
async def test_detached_commands_release_permits_before_any_client_reconnects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _ = _local(monkeypatch, Bash=Bash())
    completed: set[str] = set()
    complete = tool_serving._CallPermits.completed
    loop = asyncio.get_running_loop()

    def completed_call(
        permits: tool_serving._CallPermits,
        worker: tool_serving._PermittedHost,
        call_id: str,
    ) -> None:
        complete(permits, worker, call_id)
        _ = loop.call_soon_threadsafe(completed.add, call_id)

    monkeypatch.setattr(tool_serving._CallPermits, "completed", completed_call)
    dial_allowed = asyncio.Event()
    dial_allowed.set()
    async with (
        _serving() as relay,
        _hosting(relay, host := relay.host()),
        AsyncExitStack() as stack,
    ):
        endpoints = [
            await _up(
                await stack.enter_async_context(
                    _agent(_link(relay, dial_allowed=dial_allowed)),
                ),
            )
            for _ in range(MUX_CALLS)
        ]
        assert len(endpoints) == MUX_CALLS
        mux = relay.mux[host.id]
        calls: list[asyncio.Task[ToolMessage]] = []
        writes: list[Path] = []
        releases: list[int] = []
        for index, endpoint in enumerate(endpoints):
            fifo = tmp_path / f"release-{index}"
            os.mkfifo(fifo)
            # Keep the pipe open so releasing the shell never blocks this event loop.
            fd = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
            stack.callback(os.close, fd)
            releases.append(fd)
            output = tmp_path / f"command-{index}"
            writes.append(output)
            path = shlex.quote(str(output))
            calls.append(
                _running(
                    endpoint,
                    "Bash",
                    command=(
                        f"printf 'started\\n' >> {path}; "
                        f"read -r release < {shlex.quote(str(fifo))} && "
                        f"printf 'finished\\n' >> {path} && printf 'command {index:02}\\n'"
                    ),
                    description="Wait for the detached command test",
                ),
            )
        try:
            await _until(lambda: all(path.exists() for path in writes))
            assert not completed
            assert len(relay.calls) == len(calls)
            call_ids = {call.call_id for call in relay.calls}
            sockets = relay.agent_sockets()
            assert len(sockets) == len(calls)
            dial_allowed.clear()
            for socket in sockets:
                socket.transport.abort()
            await _until(lambda: not relay.agent_sockets())
            await _until(lambda: all(not endpoint.up for endpoint in endpoints))
            assert not any(call.done() for call in calls)
            assert not completed

            for fd in releases:
                assert os.write(fd, b"finish\n") == len(b"finish\n")
            await _until(lambda: completed == call_ids)
            assert all(path.read_text() == "started\nfinished\n" for path in writes)
            assert not relay.agent_sockets()
            assert not any(call.done() for call in calls)

            async with _agent(_link(relay)) as fresh:
                result = await _call(
                    await _up(fresh),
                    "Bash",
                    command="printf 'fresh command'",
                    description="Verify a fresh permit is available",
                )
                assert isinstance(result.result, BashResult)
                assert result.result.exit_code == 0
                assert "fresh command" in result.text()
                assert len(relay.agent_sockets()) == 1
                assert all(not endpoint.up for endpoint in endpoints)
                assert relay.mux[host.id] is mux

            dial_allowed.set()
            async with asyncio.timeout(_LIMIT_SEC):
                for index, result in enumerate(await asyncio.gather(*calls)):
                    assert isinstance(result.result, BashResult)
                    assert result.result.exit_code == 0
                    assert f"command {index:02}" in result.text()
            assert all(path.read_text() == "started\nfinished\n" for path in writes)
            assert len(relay.calls) == len(calls) + 1
        finally:
            for call in calls:
                _ = call.cancel()
            _ = await asyncio.gather(*calls, return_exceptions=True)


async def test_a_host_connected_mid_session_shows_in_set_links(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _ = _local(monkeypatch, Read=Read())
    readme = tmp_path / "readme.txt"
    readme.write_text("first\n")
    async with (
        _serving() as relay,
        _hosting(relay, relay.host()),
        _agent(_link(relay)) as session,
    ):
        kept = await _up(session)
        async with _hosting(relay, relay.host("later")):
            session.set_links([_link(relay), _link(relay, host="later")])
            assert set(session.endpoints()) == {_NAME, "later"}
            assert session.endpoints()[_NAME] is kept
            result = await _call(
                await _up(session, "later"),
                "Read",
                file_path=str(readme),
            )
    assert "first" in _text(result)
    assert [call.host for call in relay.calls] == ["later"]


async def test_a_refused_line_decodes_as_a_result() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(refused_line("c1", "no Bash"))
    reader.feed_eof()
    message = await received(reader)
    assert isinstance(message, Result)
    assert message.id == "c1"
    assert isinstance(message.record, ToolMessage)
    assert message.record.state == "failed"
    assert (message.record.text(), message.record.is_error) == ("no Bash", True)


# -- What ends a tool connection, and what heals it -------------------------------------------------


async def test_a_sweep_ends_a_connection_whose_rule_no_longer_holds_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _Gated()
    _ = _local(monkeypatch, Bash=gate)
    async with _serving() as relay:
        host = relay.host()
        async with (
            _hosting(relay, host),
            _agent(_link(relay)) as session,
        ):
            endpoint = await _up(session)
            call = _running(endpoint, "Bash")
            await _until(gate.started.is_set)

            relay.allowed.discard(host.id)
            assert relay.relay is not None
            await relay.relay.sweep()

            await _said(
                endpoint,
                f"tool connection with {wire_terminals.CLOSE_REFUSED}: ",
            )
            await _said(endpoint, f"{_NAME} is not confirmed")

            # Allowed again, the session is still there, and so is the call in it.
            relay.allowed.add(host.id)
            gate.open()
            async with asyncio.timeout(_LIMIT_SEC):
                result = await call
    assert (result.state, result.result) == ("done", UncategorizedToolResult())


# -- What else crosses the relay ---------------------------------------------------------


async def test_a_line_longer_than_a_frame_crosses_both_ways(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _local(monkeypatch, Bash=_Echo())
    # Many frames' worth, with characters a frame may split.
    text = "é€" * (1 << 15)
    async with (
        _serving() as relay,
        _hosting(relay, relay.host()),
        _agent(_link(relay)) as session,
    ):
        result = await _call(await _up(session), "Bash", text=text)
    assert isinstance(result.result, UncategorizedToolResult)
    assert result.text() == text
    (call,) = relay.calls
    assert call.argument_bytes > len(text)


async def test_a_line_over_the_limit_ends_the_connection_and_the_agent_hears_it_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = 1 << 14
    _ = _local(monkeypatch, Bash=_Echo())
    async with (
        _serving(line_bytes=limit) as relay,
        _hosting(relay, relay.host()),
        _agent(_link(relay)) as session,
    ):
        endpoint = await _up(session)
        lost = await _call(endpoint, "Bash", text="x" * (limit + 1))
        assert "lost this call" in _failed(lost)
        answered = await _call(await _up(session), "Bash", text="next")
    assert isinstance(answered.result, UncategorizedToolResult)
    assert answered.text() == "next"
    assert len(relay.calls) == 1


async def test_a_read_of_an_image_past_the_relays_line_cap_returns_whole(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    limit = 1 << 14
    _ = _local(monkeypatch, Read=Read())
    image = tmp_path / "big.png"
    image.write_bytes(os.urandom(4 * limit))
    async with (
        _serving(line_bytes=limit) as relay,
        _hosting(relay, relay.host()),
        _agent(_link(relay)) as session,
    ):
        result = await _call(await _up(session), "Read", file_path=str(image))
    assert isinstance(result.result, ReadResult), result
    (attachment,) = result.attachments
    assert attachment.data == image.read_bytes()


async def test_a_cancelled_call_stops_the_tool_and_the_connection_goes_on(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gate = _Gated()
    _ = _local(monkeypatch, Bash=gate, Read=Read())
    readme = tmp_path / "readme.txt"
    readme.write_text("still here\n")
    async with (
        _serving() as relay,
        _hosting(relay, relay.host()),
        _agent(_link(relay)) as session,
    ):
        endpoint = await _up(session)
        call = _running(endpoint, "Bash")
        await _until(gate.started.is_set)

        _ = call.cancel()

        await _until(gate.stopped.is_set)
        async with _agent(_link(relay)) as other:
            assert "still here" in _text(
                await _call(await _up(other), "Read", file_path=str(readme)),
            )
        result = await _call(endpoint, "Read", file_path=str(readme))
    assert "still here" in _text(result)


async def test_an_agent_that_leaves_says_goodbye_and_its_daemon_stops(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    daemon = _local(monkeypatch, Read=Read())
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("read before goodbye\n")
    async with _serving() as relay, _hosting(relay, relay.host()):
        async with _agent(_link(relay)) as session:
            result = await _call(await _up(session), "Read", file_path=str(evidence))
            assert "read before goodbye" in _text(result)
            (running,) = [worker for worker in daemon.started if not worker.task.done()]
        await _until(running.task.done)


# -- The rule at the relay ---------------------------------------------------------------


async def test_a_tool_a_rule_denies_is_refused_and_never_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def rule(call: GatedCall) -> Verdict:
        if call.tool == "Bash":
            return Verdict(outcome="deny", reason="Ada's rule denies Bash.")
        return Verdict(outcome="allow")

    gate = _Gated()
    daemon = _local(monkeypatch, Bash=gate, Read=Read())
    readme = tmp_path / "readme.txt"
    readme.write_text("allowed\n")
    async with _serving() as relay:
        relay.authorize = rule
        async with _hosting(relay, relay.host()), _agent(_link(relay)) as session:
            endpoint = await _up(session)
            denied = await _call(endpoint, "Bash")
            assert not gate.started.is_set()
            allowed = await _call(endpoint, "Read", file_path=str(readme))
    assert denied.state == "failed"
    assert denied.text() == "Ada's rule denies Bash."
    assert "allowed" in _text(allowed)
    assert not gate.started.is_set()
    assert len(daemon.started) == 1
    assert [(call.tool, call.outcome) for call in relay.calls] == [
        ("Bash", "deny"),
        ("Read", "allow"),
    ]


# -- A daemon stuck mid-line ---------------------------------------------------------------


@dataclass(slots=True, kw_only=True)
class _Stuck:
    """The first tool connection's daemon, which answers a call with half a line, and hangs."""

    sent: threading.Event = field(default_factory=threading.Event)
    ended: threading.Event = field(default_factory=threading.Event)

    async def serve(self, stream: MuxStream) -> None:
        held = b""
        try:
            async for frame in stream:
                held += frame
                while b"\n" in held:
                    line, _, held = held.partition(b"\n")
                    await self._answer(stream, line + b"\n")
        finally:
            await stream.close()
            self.ended.set()

    async def _answer(self, stream: MuxStream, line: bytes) -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(line)
        message = await received(reader)
        if isinstance(message, Hello):
            await stream.send(
                encoded(
                    Welcome(
                        name=_NAME,
                        resumed=False,
                        calls=(),
                        tools={"Bash": "digest"},
                    ),
                ),
            )
        elif isinstance(message, Call):
            whole = encoded(
                Result(
                    id=message.id,
                    record=ToolMessage(
                        call_id="",
                        state="done",
                        stdout=("corrupt",),
                        result=UncategorizedToolResult(),
                    ),
                ),
            )
            await stream.send(whole[: len(whole) // 2])
            self.sent.set()


def _stuck_connections(
    monkeypatch: pytest.MonkeyPatch,
    *,
    every: bool = False,
) -> _Stuck:
    """Have the daemon's first tool connection stuck, or ``every`` one; any other is served as usual."""
    real = tool_serving._adopted_stream
    stuck = _Stuck()
    taken: list[str] = []

    async def adopted(
        server: Server,
        stream: MuxStream,
        *,
        peer: str,
        end_sec: float,
        line_bytes: int,
    ) -> None:
        if taken and not every:
            await real(
                server,
                stream,
                peer=peer,
                end_sec=end_sec,
                line_bytes=line_bytes,
            )
            return
        taken.append(peer)
        await stuck.serve(stream)

    monkeypatch.setattr(tool_serving, "_adopted_stream", adopted)
    return stuck


@pytest.mark.parametrize("ending", ["sweep", "stop"])
async def test_a_daemon_stuck_mid_line_harms_no_one_and_still_ends(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ending: str,
) -> None:
    stuck = _stuck_connections(monkeypatch)
    _ = _local(monkeypatch, Read=Read())
    readme = tmp_path / "readme.txt"
    readme.write_text("carried on\n")
    async with _serving() as relay:
        host = relay.host()
        async with (
            _hosting(relay, host) as daemon,
            _agent(_link(relay)) as stuck_agent,
            _agent(_link(relay)) as other_agent,
        ):
            endpoint = await _up(stuck_agent)
            call = _running(endpoint, "Bash")
            await _until(stuck.sent.is_set)

            other = await _up(other_agent)
            carried = await _call(other, "Read", file_path=str(readme))
            assert "carried on" in _text(carried)
            assert not call.done()

            if ending == "sweep":
                relay.allowed.discard(host.id)
                assert relay.relay is not None
                await relay.relay.sweep()
                await _until(stuck.ended.is_set)
                await _said(
                    endpoint,
                    f"tool connection with {wire_terminals.CLOSE_REFUSED}: ",
                )
                await _until(lambda: not endpoint.up)
                assert not call.done()
                relay.allowed.add(host.id)
                # The real daemon holds nothing of it, so the call fails, not reads half.
                async with asyncio.timeout(_LIMIT_SEC):
                    assert "lost this call" in _failed(await call)
            else:
                await asyncio.to_thread(daemon.stop)
                assert stuck.ended.is_set()
                await _until(lambda: not endpoint.up)
                assert not call.done()
                _ = call.cancel()


async def test_a_daemon_stuck_mid_line_cannot_hold_an_agent_past_its_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A silence this short also times out a handshake slowed by load. Were only the first
    # tool connection stuck, the agent would then redial onto a served one, and never be held.
    stuck = _stuck_connections(monkeypatch, every=True)
    _ = _local(monkeypatch, Read=Read())
    async with (
        _serving() as relay,
        _hosting(relay, relay.host()),
        _agent(_link(relay), ping_sec=0.01) as session,
    ):
        endpoint = await _up(session)
        call = _running(endpoint, "Bash")
        await _until(stuck.sent.is_set)
        async with asyncio.timeout(_LIMIT_SEC):
            assert "lost this call" in _failed(await call)
        await _until(stuck.ended.is_set)


# -- Two URLs, one host name ----------------------------------------------------------


async def test_two_urls_of_one_host_name_keep_their_state_apart(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _ = _local(monkeypatch, Read=Read())
    readme = tmp_path / "readme.txt"
    readme.write_text("either\n")
    async with _serving() as relay, _serving() as other:
        other_url = f"http://localhost:{other.port}"
        async with (
            _hosting(relay, relay.host()),
            _hosting(other, other.host(), url=other_url),
            _agent(_link(relay)) as first,
            _agent(_link(other)) as second,
        ):
            for session in (first, second):
                result = await _call(await _up(session), "Read", file_path=str(readme))
                assert "either" in _text(result)
            roots = {
                tool_dirs(_NAME, server=url).state for url in (relay.url, other_url)
            }
            assert len(roots) == 2
            assert all((root / "server.pid").exists() for root in roots)
            assert server_dir(relay.url) != server_dir(other_url)
    assert "tools: off" not in capsys.readouterr().err


# -- Every real tool ---------------------------------------------------------------------


@pytest.mark.cli_bash
async def test_each_real_tool_answers_through_the_relay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _ = _local(
        monkeypatch,
        Bash=Bash(),
        Edit=Edit(),
        Glob=Glob(),
        Grep=Grep(),
        List=List(),
        Read=Read(),
        Write=Write(),
    )
    note = tmp_path / "note.txt"
    async with (
        _serving() as relay,
        _hosting(relay, relay.host()),
        _agent(_link(relay)) as session,
    ):
        endpoint = await _up(session)
        calls: tuple[tuple[str, dict[str, PlainTree]], ...] = (
            ("Write", {"file_path": str(note), "content": "é first\n"}),
            ("Read", {"file_path": str(note)}),
            (
                "Edit",
                {
                    "file_path": str(note),
                    "old_string": "first",
                    "new_string": "second",
                },
            ),
            ("Grep", {"pattern": "second", "path": str(tmp_path)}),
            ("Glob", {"pattern": "*.txt", "path": str(tmp_path)}),
            ("List", {"path": str(tmp_path)}),
        )
        for tool, arguments in calls:
            async with asyncio.timeout(_LIMIT_SEC):
                result = await endpoint.call(
                    agent="Agent",
                    window="w1",
                    tool=tool,
                    arguments=arguments,
                )
            assert not result.is_error, (tool, result)
        # More than a frame, and more than a tool keeps in its answer.
        spilled = await _call(
            endpoint,
            "Bash",
            command="head -c 100000 /dev/zero | tr '\\0' x",
            description="Print a lot",
        )
    assert isinstance(spilled.result, BashResult)
    assert spilled.result.exit_code == 0
    assert any(isinstance(part, str) and part for part in spilled.stdout)
    assert note.read_text() == "é second\n"
    assert len(relay.calls) == 7


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
