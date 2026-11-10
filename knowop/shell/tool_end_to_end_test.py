"""A real ``Endpoint`` reaches a real tool host through Trackinizer's relay.

The whole path, over loopback sockets: an ``Endpoint`` dials with ``dial_tools``; a
fake server routes each leg into Trackinizer's real ``PipeHub``, which reads each line
with its real ``ToolLines``; ``ToolHost`` takes the hub's offer and dials back, and hands
the pipe to ``Server.adopt``. The fake stands in for the routes, the database and the
long poll alone, which Trackinizer's own route tests cover. Every host runs in a
``LocalHost``, in this process: ``tool_host_test.py`` covers the real child.

It lives here, not beside the relay, as trackinizer may not import ``knowop``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Final, cast, override
from urllib.parse import parse_qs, unquote, urlsplit

import asyncio
import os
import re
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
    FunctionResult,
    ReadResult,
    RemoteToolResult,
    ToolErrorResult,
)
from sagent.tools.bash import Bash
from sagent.tools.edit import Edit
from sagent.tools.glob_tool import Glob
from sagent.tools.grep import Grep
from sagent.tools.list import List
from sagent.tools.read import Read
from sagent.tools.remote import Endpoint, Link, RemoteSession, RemoteTool
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
from trackinizer.server.tool_pipes import PipeEndError, PipeHub
from trackinizer.trax.connection import server_dir
from trackinizer.wire import wire_terminals
from trackinizer.wire.wire_tools import MACHINE_TOOLS_WS_PATH, TOOLS_WS_PATH
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

import pytest

from knowop.shell import tool_host
from knowop.shell.tool_host import ToolClient, ToolHost, tool_dirs


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Mapping
    from pathlib import Path

    from sagent.host.server import Server
    from sagent.session.types import AnyToolResult
    from sagent.tools.types import Tool, ToolContext
    from trackinizer.server.tool_lines import Authorize
    from trackinizer.server.tool_pipes import Inbound, Pipe
    from trackinizer.wire.wire_tools import ToolOffer
    from treekle import PlainTree
    from websockets.asyncio.client import ClientConnection
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
"""How long an agent or host waits to try again, never backing off past twice that."""

_WAIT_SEC: Final = 0.005
"""How long a host's wait is held: short, so a stopped host's wait ends at once."""

_AGENT_PATH: Final = re.compile(TOOLS_WS_PATH.format(name="(?P<name>[^/]+)"))
_HOST_PATH: Final = re.compile(
    MACHINE_TOOLS_WS_PATH.format(
        machine_id="(?P<machine>[^/]+)",
        pipe_id="(?P<pipe>[^/]+)",
    ),
)


@pytest.fixture(autouse=True)
def _empty_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep what a real tool reads or writes under home off the developer's own."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


@dataclass(frozen=True, slots=True, kw_only=True)
class _Machine:
    name: str
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    instance: uuid.UUID = field(default_factory=uuid.uuid4)


class _Leg:
    """A websocket the relay serves, as the hub's ``PipeSocket``."""

    def __init__(self, socket: ServerConnection) -> None:
        self._socket = socket

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
    """Trackinizer's relay without its routes: a real hub and line reader on a port.

    Every machine it knows is the asker's, confirmed and serving tools, until a test
    takes it out of ``allowed``. A host's wait reads the hub directly, on this loop,
    in place of the HTTP long poll.
    """

    def __init__(self, *, line_bytes: int, dial_back_sec: float) -> None:
        self.port = 0
        self.line_bytes = line_bytes
        self.dial_back_sec = dial_back_sec
        self.machines: dict[str, _Machine] = {}
        self.allowed: set[uuid.UUID] = set()
        self.calls: list[GatedCall] = []
        """Each call the relay's gate judged, oldest first, as its log line says."""

        self.authorize: Authorize = authorize
        self.hub: PipeHub | None = None
        self.loop = asyncio.get_running_loop()
        self._server: WebsocketServer | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> None:
        """Serve on this relay's port, or on a new one the first time, with a new hub."""
        self.hub = PipeHub(
            alive=self._alive,
            pipes_per_machine=Config().tool_pipes_per_machine,
            dial_back_sec=self.dial_back_sec,
            line_bytes=Config().tool_line_bytes,
        )
        self._server = await serve(self._handle, "127.0.0.1", self.port)
        self.port = cast(int, next(iter(self._server.sockets)).getsockname()[1])

    async def stop(self) -> None:
        """Stop serving, as a restart does: every connection goes, and the hub with it."""
        assert self._server is not None
        self.hub = None
        self._server.close()
        await self._server.wait_closed()

    def drop_connections(self) -> None:
        """Cut every connection with no goodbye, as a network does; the server stays."""
        assert self._server is not None
        for connection in tuple(self._server.connections):
            connection.transport.abort()

    def machine(self, name: str = _NAME) -> _Machine:
        """Know a machine of the asker's, confirmed and serving tools."""
        found = _Machine(name=name)
        self.machines[name] = found
        self.allowed.add(found.id)
        return found

    async def _alive(self, pipe: Pipe) -> bool:
        return pipe.machine_id in self.allowed

    async def _handle(self, socket: ServerConnection) -> None:
        request = socket.request
        assert request is not None
        url = urlsplit(request.path)
        query = parse_qs(url.query)
        leg = _Leg(socket)
        hub = self.hub
        assert hub is not None
        if (agent := _AGENT_PATH.fullmatch(url.path)) is not None:
            await self._agent(hub, leg, unquote(agent["name"]), asker=query["asker"][0])
        elif (host := _HOST_PATH.fullmatch(url.path)) is not None:
            pipe = hub.attach(
                uuid.UUID(host["machine"]),
                uuid.UUID(host["pipe"]),
                leg,
                instance=uuid.UUID(query["instance"][0]),
            )
            if pipe is None:
                await leg.close(wire_terminals.CLOSE_REFUSED, "no such pipe waits")
                return
            await hub.serve_host(pipe, leg.read)
        else:
            await leg.close(wire_terminals.CLOSE_REFUSED, "no such route")

    async def _agent(self, hub: PipeHub, leg: _Leg, name: str, *, asker: str) -> None:
        machine = self.machines[name]
        if machine.id not in self.allowed:
            await leg.close(wire_terminals.CLOSE_REFUSED, f"{name} is not confirmed")
            return
        pipe = hub.join(
            machine.id,
            agent=leg,
            asker=asker,
            api_key_id=_KEY,
            inbound=self._gated(asker=asker, machine=name),
        )
        await hub.serve_agent(pipe, leg.read)

    def _gated(self, *, asker: str, machine: str) -> Inbound:
        """Read the agent's lines as the agent route does, keeping each call judged."""
        lines = ToolLines(
            asker=asker,
            machine=machine,
            authorize=self._judged,
            line_bytes=self.line_bytes,
        )

        async def inbound(pipe: Pipe, frame: bytes) -> tuple[bytes, bytes]:
            del pipe
            try:
                routed, calls = lines.feed(frame)
            except ValueError as error:
                reason = f"the server will not relay a line: {error}"
                raise PipeEndError(reason) from error
            self.calls.extend(calls)
            return routed.to_host, routed.to_agent

        return inbound

    def _judged(self, call: GatedCall) -> Verdict:
        """Judge with the rule set now, which a test may change once a pipe is open."""
        return self.authorize(call)


class _Waits:
    """A host's client: each wait is held on the relay's hub, on the relay's loop."""

    def __init__(self, relay: _Relay) -> None:
        self._relay = relay

    def wait_tools(
        self,
        machine_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        wait_sec: float,
    ) -> ToolOffer | None:
        del instance
        hub = self._relay.hub
        if hub is None:
            msg = "the server is down"
            raise ConnectionError(msg)
        held = hub.wait(machine_id, wait_sec=wait_sec)
        return asyncio.run_coroutine_threadsafe(held, self._relay.loop).result()

    def close(self) -> None:
        return


@asynccontextmanager
async def _serving(
    *,
    line_bytes: int = Config().tool_line_bytes,
    dial_back_sec: float = Config().tool_dial_back_sec,
) -> AsyncGenerator[_Relay]:
    """Yield a running relay; at its end, every pipe and host wait through it has ended."""
    relay = _Relay(line_bytes=line_bytes, dial_back_sec=dial_back_sec)
    await relay.start()
    try:
        yield relay
        await _until(lambda: relay.hub is None or not relay.hub._pipes)
        # A stopped host's listener ends with the wait it holds, which this loop answers.
        await _until(lambda: not _tool_threads())
    finally:
        if relay.hub is not None:
            await relay.stop()


@dataclass(frozen=True, slots=True, kw_only=True)
class _Gated(Gate):
    """A ``Gate`` opened from outside the host's thread, which owns its event loop."""

    loops: list[asyncio.AbstractEventLoop] = field(
        default_factory=list[asyncio.AbstractEventLoop],
    )

    @override
    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> FunctionResult:
        self.loops.append(asyncio.get_running_loop())
        return await Gate.call(self, context, arguments)

    def open(self) -> None:
        _ = self.loops[-1].call_soon_threadsafe(self.opened.set)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Echo(Gate):
    """A tool that answers with the text it was given."""

    @override
    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> FunctionResult:
        del context
        return FunctionResult(call_id="", name="Echo", content=str(arguments["text"]))


def _local(monkeypatch: pytest.MonkeyPatch, **tools: Tool) -> LocalHost:
    """Serve ``tools`` from a host in the tool host's own event loop."""
    host = LocalHost(tools=tools)
    monkeypatch.setattr(tool_host, "scrubbed_host", host)
    return host


@asynccontextmanager
async def _hosting(
    relay: _Relay,
    machine: _Machine,
    *,
    url: str = "",
    client_for: Callable[[str], ToolClient] | None = None,
) -> AsyncGenerator[ToolHost]:
    """Run a ``ToolHost`` for ``machine``, asking again at once after a failed wait."""
    url = url or relay.url
    host = ToolHost(
        url,
        credential=_CREDENTIAL,
        name=machine.name,
        client_for=client_for or (lambda _: _Waits(relay)),
        dirs=tool_dirs(machine.name, server=url),
        retry_sec=_RETRY_SEC,
        wait_sec=_WAIT_SEC,
    )
    host.start(machine.id, machine.instance)
    try:
        yield host
    finally:
        await asyncio.to_thread(host.stop)


def _link(relay: _Relay, *, machine: str = _NAME) -> Link:
    return Link(
        dial=partial(
            dial_tools,
            relay.url,
            api_key="assistant-key",
            machine=machine,
            asker=_ADA,
            limit=MAX_LINE_BYTES,
        ),
        name=machine,
        where="relay",
    )


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
        await asyncio.sleep(0.005)


async def _up(session: RemoteSession, name: str = _NAME) -> Endpoint:
    endpoint = session.endpoints()[name]
    async with asyncio.timeout(_LIMIT_SEC):
        await endpoint.until(lambda each: each.up)
    return endpoint


async def _said(endpoint: Endpoint, text: str) -> None:
    """Return once the endpoint's error holds ``text``."""
    async with asyncio.timeout(_LIMIT_SEC):
        await endpoint.until(lambda each: text in each.error)


def _failed(result: AnyToolResult) -> str:
    """Return why a call came to no result, as its agent is told."""
    assert isinstance(result, RemoteToolResult), result
    assert result.is_error is True
    return result.text()


def _text(result: AnyToolResult) -> str:
    """Return what a real ``Read`` read."""
    assert isinstance(result, ReadResult), result
    return str(result.content)


async def _call(
    endpoint: Endpoint,
    tool: str,
    *,
    agent: str = "Agent",
    **arguments: PlainTree,
) -> AnyToolResult:
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
) -> asyncio.Task[AnyToolResult]:
    return asyncio.create_task(
        endpoint.call(agent="Agent", window="w1", tool=tool, arguments=arguments),
    )


def _tool_threads() -> set[str]:
    return {t.name for t in threading.enumerate() if t.name.startswith("tool-")}


# -- The brief's path --------------------------------------------------------------------


async def test_a_real_endpoint_reaches_a_real_tool_host(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _ = _local(monkeypatch, Read=Read())
    readme = tmp_path / "readme.txt"
    readme.write_text("from this machine\n")
    async with (
        _serving() as relay,
        _hosting(relay, relay.machine()),
        _agent(_link(relay)) as session,
    ):
        endpoint = await _up(session)
        spec = await endpoint.spec("Read")
        assert spec is not None
        assert spec.digest == endpoint.tools["Read"]
        result = await _call(endpoint, "Read", file_path=str(readme))
    assert "from this machine" in _text(result)
    (call,) = relay.calls
    assert (call.asker, call.agent, call.machine, call.tool, call.outcome) == (
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
    readme.write_text("from this machine\n")
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
        _hosting(relay, relay.machine()),
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
        if isinstance(r, ReadResult)
    ]
    (call,) = relay.calls
    assert "from this machine" in _text(record)
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
    host = _local(monkeypatch, Bash=gate)
    async with (
        _serving() as relay,
        _hosting(relay, relay.machine()),
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
    assert isinstance(result, FunctionResult)
    assert result.content == "open for Agent@laptop"
    assert len(host.started) == 1
    assert len(relay.calls) == 1


async def test_two_sessions_on_one_machine_get_their_own_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _Gated()
    host = _local(monkeypatch, Bash=gate)
    async with (
        _serving() as relay,
        _hosting(relay, relay.machine()),
        _agent(_link(relay)) as first,
        _agent(_link(relay)) as second,
    ):
        calls = [
            asyncio.create_task(_call(await _up(session), "Bash", agent=agent))
            for session, agent in ((first, "One"), (second, "Two"))
        ]
        await _until(lambda: len(host.started) == 2)
        await _until(gate.started.is_set)
        gate.open()
        one, two = await asyncio.gather(*calls)
    assert isinstance(one, FunctionResult)
    assert isinstance(two, FunctionResult)
    assert one.content == "open for One@laptop"
    assert two.content == "open for Two@laptop"
    assert {call.session for call in relay.calls} == {
        first.session_id,
        second.session_id,
    }


async def test_a_machine_connected_mid_session_shows_in_set_links(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _ = _local(monkeypatch, Read=Read())
    readme = tmp_path / "readme.txt"
    readme.write_text("first\n")
    async with (
        _serving() as relay,
        _hosting(relay, relay.machine()),
        _agent(_link(relay)) as session,
    ):
        kept = await _up(session)
        async with _hosting(relay, relay.machine("later")):
            session.set_links([_link(relay), _link(relay, machine="later")])
            assert set(session.endpoints()) == {_NAME, "later"}
            assert session.endpoints()[_NAME] is kept
            result = await _call(
                await _up(session, "later"),
                "Read",
                file_path=str(readme),
            )
    assert "first" in _text(result)
    assert [call.machine for call in relay.calls] == ["later"]


async def test_a_refused_line_decodes_as_a_result() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(refused_line("c1", "no Bash"))
    reader.feed_eof()
    message = await received(reader)
    assert isinstance(message, Result)
    assert message.id == "c1"
    assert isinstance(message.record, ToolErrorResult)
    assert (message.record.content, message.record.is_error) == ("no Bash", True)


# -- What ends a pipe, and what heals it -------------------------------------------------


async def test_a_sweep_ends_a_pipe_whose_rule_no_longer_holds_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _Gated()
    _ = _local(monkeypatch, Bash=gate)
    async with _serving() as relay:
        machine = relay.machine()
        async with (
            _hosting(relay, machine),
            _agent(_link(relay)) as session,
        ):
            endpoint = await _up(session)
            call = _running(endpoint, "Bash")
            await _until(gate.started.is_set)

            relay.allowed.discard(machine.id)
            assert relay.hub is not None
            await relay.hub.sweep()

            await _said(endpoint, f"pipe with {wire_terminals.CLOSE_REFUSED}: ")
            await _said(endpoint, f"{_NAME} is not confirmed")

            # Allowed again, the session is still there, and so is the call in it.
            relay.allowed.add(machine.id)
            gate.open()
            async with asyncio.timeout(_LIMIT_SEC):
                result = await call
    assert isinstance(result, FunctionResult)


class _LosesTheFirstOffer:
    """A host's client whose first offer is lost on its way, as a dropped response is."""

    def __init__(self, client: ToolClient) -> None:
        self._client = client
        self.lost = 0

    def wait_tools(
        self,
        machine_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        wait_sec: float,
    ) -> ToolOffer | None:
        offer = self._client.wait_tools(
            machine_id,
            instance=instance,
            wait_sec=wait_sec,
        )
        if offer is not None and not self.lost:
            self.lost += 1
            return None
        return offer

    def close(self) -> None:
        self._client.close()


async def test_an_offer_lost_on_its_way_heals_by_the_dial_back_and_a_redial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _ = _local(monkeypatch, Read=Read())
    readme = tmp_path / "readme.txt"
    readme.write_text("healed\n")
    clients: list[_LosesTheFirstOffer] = []
    async with _serving(dial_back_sec=0.01) as relay:

        def losing_client(key: str) -> _LosesTheFirstOffer:
            del key
            clients.append(_LosesTheFirstOffer(_Waits(relay)))
            return clients[-1]

        async with (
            _hosting(relay, relay.machine(), client_for=losing_client),
            _agent(_link(relay)) as session,
        ):
            result = await _call(await _up(session), "Read", file_path=str(readme))
    assert "healed" in _text(result)
    assert [client.lost for client in clients] == [1]


# -- What else crosses the relay ---------------------------------------------------------


async def test_a_line_longer_than_a_frame_crosses_both_ways(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _local(monkeypatch, Bash=_Echo())
    # Many frames' worth, with characters a frame may split.
    text = "é€" * (1 << 15)
    async with (
        _serving() as relay,
        _hosting(relay, relay.machine()),
        _agent(_link(relay)) as session,
    ):
        result = await _call(await _up(session), "Bash", text=text)
    assert isinstance(result, FunctionResult)
    assert result.content == text
    (call,) = relay.calls
    assert call.argument_bytes > len(text)


async def test_a_line_over_the_limit_ends_the_pipe_and_the_agent_hears_it_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = 1 << 14
    _ = _local(monkeypatch, Bash=_Echo())
    async with (
        _serving(line_bytes=limit) as relay,
        _hosting(relay, relay.machine()),
        _agent(_link(relay)) as session,
    ):
        endpoint = await _up(session)
        lost = await _call(endpoint, "Bash", text="x" * (limit + 1))
        assert "lost this call" in _failed(lost)
        answered = await _call(await _up(session), "Bash", text="next")
    assert isinstance(answered, FunctionResult)
    assert answered.content == "next"
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
        _hosting(relay, relay.machine()),
        _agent(_link(relay)) as session,
    ):
        result = await _call(await _up(session), "Read", file_path=str(image))
    assert isinstance(result, ReadResult), result
    (attachment,) = result.attachments
    assert attachment.data == image.read_bytes()


async def test_a_cancelled_call_stops_the_tool_and_the_pipe_goes_on(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gate = _Gated()
    _ = _local(monkeypatch, Bash=gate, Read=Read())
    readme = tmp_path / "readme.txt"
    readme.write_text("still here\n")
    async with (
        _serving() as relay,
        _hosting(relay, relay.machine()),
        _agent(_link(relay)) as session,
    ):
        endpoint = await _up(session)
        call = _running(endpoint, "Bash")
        await _until(gate.started.is_set)

        _ = call.cancel()

        await _until(gate.stopped.is_set)
        result = await _call(endpoint, "Read", file_path=str(readme))
    assert "still here" in _text(result)


async def test_an_agent_that_leaves_says_goodbye_and_its_host_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _local(monkeypatch, Read=Read())
    async with _serving() as relay, _hosting(relay, relay.machine()):
        async with _agent(_link(relay)) as session:
            _ = await _up(session)
            (running,) = host.started
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
    host = _local(monkeypatch, Bash=gate, Read=Read())
    readme = tmp_path / "readme.txt"
    readme.write_text("allowed\n")
    async with _serving() as relay:
        relay.authorize = rule
        async with _hosting(relay, relay.machine()), _agent(_link(relay)) as session:
            endpoint = await _up(session)
            denied = await _call(endpoint, "Bash")
            # What the rule allows still crosses, so the denied line would have by now.
            allowed = await _call(endpoint, "Read", file_path=str(readme))
    assert isinstance(denied, ToolErrorResult)
    assert denied.content == "Ada's rule denies Bash."
    assert "allowed" in _text(allowed)
    assert not gate.started.is_set()
    assert len(host.started) == 1
    assert [(call.tool, call.outcome) for call in relay.calls] == [
        ("Bash", "deny"),
        ("Read", "allow"),
    ]


# -- A host stuck mid-line ---------------------------------------------------------------


@dataclass(slots=True, kw_only=True)
class _Stuck:
    """The first pipe's host, which answers a call with half a line, and hangs."""

    sent: threading.Event = field(default_factory=threading.Event)
    ended: threading.Event = field(default_factory=threading.Event)
    close_code: int | None = None

    async def serve(self, socket: ClientConnection) -> None:
        held = b""
        try:
            while True:
                held += await socket.recv(decode=False)
                while b"\n" in held:
                    line, _, held = held.partition(b"\n")
                    await self._answer(socket, line + b"\n")
        except ConnectionClosed:
            pass
        finally:
            await socket.close()
            self.close_code = socket.close_code
            self.ended.set()

    async def _answer(self, socket: ClientConnection, line: bytes) -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(line)
        message = await received(reader)
        if isinstance(message, Hello):
            await socket.send(
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
                    record=FunctionResult(call_id="", name="Bash", content="corrupt"),
                ),
            )
            await socket.send(whole[: len(whole) // 2])
            self.sent.set()


def _stuck_pipes(monkeypatch: pytest.MonkeyPatch, *, every: bool = False) -> _Stuck:
    """Have the host's first pipe stuck, or ``every`` one; any other is served as usual."""
    real = tool_host._adopted
    stuck = _Stuck()
    taken: list[str] = []

    async def adopted(
        server: Server,
        socket: ClientConnection,
        *,
        peer: str,
        end_sec: float,
        line_bytes: int,
    ) -> None:
        if taken and not every:
            await real(
                server,
                socket,
                peer=peer,
                end_sec=end_sec,
                line_bytes=line_bytes,
            )
            return
        taken.append(peer)
        await stuck.serve(socket)

    monkeypatch.setattr(tool_host, "_adopted", adopted)
    return stuck


@pytest.mark.parametrize("ending", ["sweep", "stop"])
async def test_a_host_stuck_mid_line_harms_no_one_and_still_ends(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ending: str,
) -> None:
    stuck = _stuck_pipes(monkeypatch)
    _ = _local(monkeypatch, Read=Read())
    readme = tmp_path / "readme.txt"
    readme.write_text("carried on\n")
    async with _serving() as relay:
        machine = relay.machine()
        async with (
            _hosting(relay, machine) as host,
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
                relay.allowed.discard(machine.id)
                assert relay.hub is not None
                await relay.hub.sweep()
                await _until(stuck.ended.is_set)
                assert stuck.close_code == wire_terminals.CLOSE_REFUSED
                await _until(lambda: not endpoint.up)
                assert not call.done()
                relay.allowed.add(machine.id)
                # The real host holds nothing of it, so the call fails, not reads half.
                async with asyncio.timeout(_LIMIT_SEC):
                    assert "lost this call" in _failed(await call)
            else:
                await asyncio.to_thread(host.stop)
                assert stuck.ended.is_set()
                await _until(lambda: not endpoint.up)
                assert not call.done()
                _ = call.cancel()


async def test_a_host_stuck_mid_line_cannot_hold_an_agent_past_its_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A silence this short also times out a handshake slowed by load. Were only the first
    # pipe stuck, the agent would then redial onto a served one, and never be held.
    stuck = _stuck_pipes(monkeypatch, every=True)
    _ = _local(monkeypatch, Read=Read())
    async with (
        _serving() as relay,
        _hosting(relay, relay.machine()),
        _agent(_link(relay), ping_sec=0.01) as session,
    ):
        endpoint = await _up(session)
        call = _running(endpoint, "Bash")
        await _until(stuck.sent.is_set)
        async with asyncio.timeout(_LIMIT_SEC):
            assert "lost this call" in _failed(await call)
        await _until(stuck.ended.is_set)


# -- Two URLs, one machine name ----------------------------------------------------------


async def test_two_urls_of_one_machine_name_keep_their_state_apart(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _ = _local(monkeypatch, Read=Read())
    readme = tmp_path / "readme.txt"
    readme.write_text("either\n")
    async with _serving() as relay:
        machine = relay.machine()
        other_url = f"http://localhost:{relay.port}"
        async with (
            _hosting(relay, machine),
            _hosting(relay, machine, url=other_url),
            _agent(_link(relay)) as first,
            _agent(_link(relay)) as second,
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
        _hosting(relay, relay.machine()),
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
            assert result.is_error is not True, (tool, result)
        # More than a frame, and more than a tool keeps in its answer.
        spilled = await _call(
            endpoint,
            "Bash",
            command="head -c 100000 /dev/zero | tr '\\0' x",
            description="Print a lot",
        )
    assert isinstance(spilled, BashResult)
    assert spilled.exit_code == 0
    assert spilled.stdout
    assert note.read_text() == "é second\n"
    assert len(relay.calls) == 7


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
