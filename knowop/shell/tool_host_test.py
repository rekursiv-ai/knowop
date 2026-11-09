"""The tool host: a real websocket and bridge, each session's host in this process.

A real child is sagent's to test (``host/server_test.py``, ``tools/bash_test.py``); here,
``scrubbed_host`` is pinned by the command it starts the child with.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar, Final, Self, cast, override
from unittest.mock import ANY, AsyncMock, Mock

import asyncio
import contextlib
import fcntl
import json
import logging
import os
import queue
import re
import stat
import sys
import threading
import time
import uuid

from sagent.host.server import HOST_COMMAND
from sagent.host.strings import ServeDirs
from sagent.host.testing import Gate, LocalHost
from sagent.host.wire import (
    MAX_LINE_BYTES,
    VERSION,
    Call,
    Describe,
    Described,
    Hello,
    Note,
    Refused,
    Result,
    Welcome,
    encoded,
    received,
)
from sagent.session.types import FunctionResult, ReadResult
from sagent.tools.display import ToolDisplay
from sagent.tools.read import Read
from trackinizer.client import tool_bridge
from trackinizer.client.errors import ClientError
from trackinizer.lib.userdirs import state_dir
from trackinizer.server.config import Config
from trackinizer.trax.connection import server_dir
from trackinizer.wire import wire_terminals
from trackinizer.wire.wire_machine_host import HEARTBEAT_SEC
from trackinizer.wire.wire_tools import (
    FRAME_BYTES,
    MACHINE_TOOLS_WS_PATH,
    WAIT_MAX_SEC,
    ToolOffer,
)
from treekle import PlainTree
from websockets.asyncio.client import connect
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.frames import CloseCode
from wrapt import lazy_import

import pytest

from knowop.lib.testing.import_graph import Counted, import_graph
from knowop.shell import tool_host
from knowop.shell.tool_host import ToolHost, scrubbed_host, tool_dirs


_THIS: Final = Path(__file__).resolve()
_CWD: Final = _THIS.parent

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Coroutine, Generator, Sequence

    from sagent.host.server import Host
    from sagent.host.wire import Message
    from sagent.tools.types import Tool, ToolContext
    from websockets.asyncio.client import ClientConnection
    from websockets.datastructures import Headers
    from websockets.http11 import Request, Response


pytestmark = pytest.mark.asyncio

_MACHINE: Final = uuid.UUID(int=1)
_INSTANCE: Final = uuid.UUID(int=2)
_CREDENTIAL: Final = "trax_machine_" + "0" * 32 + "_" + "c" * 43
_REVOKED: Final = ClientError("-> 410", status_code=410)
_NAME: Final = "box-1"

_ROOT: Final = _THIS.parents[__name__.count(".")]
"""The directory holding this package's top level, as an import finds it."""

_FIRST_PARTY: Final = frozenset(
    module.split(".")[0]
    for module in (__name__, ClientError.__module__, ServeDirs.__module__)
)
"""The packages of knowop, trackinizer and sagent: an import graph walks these alone."""


@pytest.fixture(autouse=True)
def _empty_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep what a real tool reads or writes under home off the developer's own."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


@pytest.fixture(autouse=True)
def _no_thread_left() -> Generator[None]:
    """End each test with its host's threads gone, however it stopped them."""
    yield
    deadline = time.monotonic() + 10
    while _threads():
        assert time.monotonic() < deadline, f"left running: {_threads()}"
        time.sleep(0.001)


class _Client:
    """The ``Client`` calls a host makes: each wait takes the next item put here."""

    def __init__(self) -> None:
        self.answers: queue.Queue[ToolOffer | Exception] = queue.Queue()
        self.waits: list[tuple[uuid.UUID, uuid.UUID, float]] = []
        self.keys: list[str] = []
        self.closed = threading.Event()

    def __call__(self, key: str) -> _Client:
        self.keys.append(key)
        return self

    def wait_tools(
        self,
        machine_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        wait_sec: float,
    ) -> ToolOffer | None:
        self.waits.append((machine_id, instance, wait_sec))
        try:
            answer = self.answers.get(timeout=0.005)
        except queue.Empty:
            return None
        if isinstance(answer, Exception):
            raise answer
        return answer

    def close(self) -> None:
        self.closed.set()


class _Leg:
    """The server's end of one host leg."""

    def __init__(self, socket: ServerConnection) -> None:
        self.socket = socket
        self._held = b""

    async def say(self, message: Message) -> None:
        await self.socket.send(encoded(message))

    async def next(self) -> Message | None:
        """Return the next line, or ``None`` when the host closed its leg."""
        try:
            while b"\n" not in self._held:
                frame = await asyncio.wait_for(self.socket.recv(decode=False), 10)
                self._held += frame
        except ConnectionClosed:
            return None
        line, _, self._held = self._held.partition(b"\n")
        reader = asyncio.StreamReader()
        reader.feed_data(line + b"\n")
        return await received(reader)

    async def closed(self) -> None:
        await asyncio.wait_for(self.socket.wait_closed(), 10)

    async def open_session(self, session_id: str = "s1") -> Message | None:
        await self.say(_hello(session_id))
        return await self.next()


class _Relay:
    """The server's side of a host: its offers, and a real websocket for each dial."""

    def __init__(self) -> None:
        self.url = ""
        self.client = _Client()
        self.legs: asyncio.Queue[_Leg] = asyncio.Queue()
        self.dials: list[tuple[str, str]] = []
        self.refusing = False
        self.silent = False
        self.spoken = asyncio.Event()
        self.offered: list[uuid.UUID] = []

    async def gate(
        self,
        connection: ServerConnection,
        request: Request,
    ) -> Response | None:
        del request
        if self.silent:
            # Answers no handshake until the test says, as a server too busy to.
            _ = await self.spoken.wait()
        if self.refusing:
            self.refusing = False
            return connection.respond(HTTPStatus.FORBIDDEN, "refused\n")
        return None

    async def handle(self, socket: ServerConnection) -> None:
        request = socket.request
        assert request is not None
        headers: Headers = request.headers
        self.dials.append((request.path, headers["Authorization"]))
        await self.legs.put(_Leg(socket))
        await socket.wait_closed()

    def offer(self) -> None:
        self.offered.append(uuid.uuid4())
        self.client.answers.put(ToolOffer(pipe_id=self.offered[-1]))

    async def leg(self) -> _Leg:
        """Offer a pipe and return the leg the host dials back for it."""
        self.offer()
        return await asyncio.wait_for(self.legs.get(), 10)


@asynccontextmanager
async def _relay() -> AsyncGenerator[_Relay]:
    relay = _Relay()
    async with serve(
        relay.handle,
        "127.0.0.1",
        0,
        process_request=relay.gate,
    ) as server:
        port = cast(int, next(iter(server.sockets)).getsockname()[1])
        relay.url = f"http://127.0.0.1:{port}"
        yield relay


def _host(
    relay: _Relay,
    *,
    dirs: ServeDirs | None = None,
    lock_wait_sec: float = 0.0,
    retry_sec: float = 0.005,
    wait_sec: float = 0.0,
    max_pipes: int = 16,
    dial_sec: float | None = None,
    line_bytes: int = 0,
) -> ToolHost:
    return ToolHost(
        relay.url,
        credential=_CREDENTIAL,
        name=_NAME,
        client_for=relay.client,
        dirs=dirs or _dirs(relay),
        retry_sec=retry_sec,
        lock_wait_sec=lock_wait_sec,
        # A close this relay cannot answer, as its loop is blocked, ends at once.
        close_sec=0.05,
        wait_sec=wait_sec,
        max_pipes=max_pipes,
        dial_sec=dial_sec,
        line_bytes=line_bytes,
    )


@asynccontextmanager
async def _serving(
    relay: _Relay,
    *,
    lock_wait_sec: float = 0.0,
    max_pipes: int = 16,
    dial_sec: float | None = None,
    line_bytes: int = 0,
) -> AsyncGenerator[ToolHost]:
    host = _host(
        relay,
        lock_wait_sec=lock_wait_sec,
        max_pipes=max_pipes,
        dial_sec=dial_sec,
        line_bytes=line_bytes,
    )
    host.start(_MACHINE, _INSTANCE)
    try:
        yield host
    finally:
        await asyncio.to_thread(host.stop)


def _dirs(relay: _Relay) -> ServeDirs:
    return tool_dirs(_NAME, server=relay.url)


def _local(monkeypatch: pytest.MonkeyPatch, **tools: Tool) -> None:
    """Serve ``tools`` from a host in the tool host's own event loop."""
    monkeypatch.setattr(tool_host, "scrubbed_host", LocalHost(tools=tools))


async def _until(done: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 10
    while not done():
        assert time.monotonic() < deadline, "the awaited change never came"
        await asyncio.sleep(0.005)


async def _result(leg: _Leg, call_id: str) -> Result:
    while True:
        message = await leg.next()
        assert message is not None, f"the pipe ended before {call_id} answered"
        if isinstance(message, Result) and message.id == call_id:
            return message


def _hello(session_id: str, *, version: int = VERSION) -> Hello:
    return Hello(
        version=version,
        token="",
        session_id=session_id,
        client="agent",
        tools=None,
    )


def _call(call_id: str, tool: str, **arguments: PlainTree) -> Call:
    return Call(id=call_id, agent="Agent", window="w1", tool=tool, arguments=arguments)


def _content(result: Result) -> str:
    assert isinstance(result.record, FunctionResult | ReadResult), result.record
    return str(result.record.content)


def _threads() -> set[str]:
    return {t.name for t in threading.enumerate() if t.name.startswith("tool-")}


async def test_each_offer_dials_back_and_serves_a_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _local(monkeypatch, Read=Read())
    dialed = Mock(wraps=connect)
    adopted = AsyncMock(wraps=tool_host._adopted)
    monkeypatch.setattr(tool_host, "connect", dialed)
    monkeypatch.setattr(tool_host, "_adopted", adopted)
    readme = tmp_path / "readme.txt"
    readme.write_text("from this machine\n")
    async with _relay() as relay, _serving(relay):
        for session in ("s1", "s2"):
            leg = await relay.leg()
            welcome = await leg.open_session(session)
            assert isinstance(welcome, Welcome)
            assert (welcome.name, list(welcome.tools)) == (_NAME, ["Read"])
            await leg.say(_call("1", "Read", file_path=str(readme)))
            assert "from this machine" in _content(await _result(leg, "1"))
            await leg.socket.close()

    assert relay.client.keys == [_CREDENTIAL]
    assert relay.client.waits[0] == (_MACHINE, _INSTANCE, WAIT_MAX_SEC)
    assert [path for path, _ in relay.dials] == [
        MACHINE_TOOLS_WS_PATH.format(machine_id=_MACHINE, pipe_id=pipe_id)
        + f"?instance={_INSTANCE}"
        for pipe_id in relay.offered
    ]
    assert {authorization for _, authorization in relay.dials} == {
        f"Bearer {_CREDENTIAL}",
    }
    dialed.assert_called_with(
        ANY,
        additional_headers={"Authorization": f"Bearer {_CREDENTIAL}"},
        max_size=FRAME_BYTES,
        ping_interval=20,
        close_timeout=0.05,
    )
    adopted.assert_awaited_with(
        ANY,
        ANY,
        peer=f"relay:{relay.offered[-1]}",
        end_sec=5.0,
        line_bytes=MAX_LINE_BYTES,
    )


class _Waits(threading.Event):
    """A stop event that keeps how long each wait on it was for."""

    def __init__(self) -> None:
        super().__init__()
        self.waited: list[float | None] = []

    @override
    def wait(self, timeout: float | None = None) -> bool:
        self.waited.append(timeout)
        return super().wait(timeout)


async def test_a_host_with_its_defaults_dials_sagents_wire_as_the_server_names_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _local(monkeypatch, Read=Read())
    socket = SimpleNamespace(close_code=1000, close_reason="")
    dialed = AsyncMock(return_value=socket)
    adopted = AsyncMock()
    stops: list[_Waits] = []

    def event() -> _Waits:
        stops.append(_Waits())
        return stops[-1]

    monkeypatch.setattr(tool_host, "connect", dialed)
    monkeypatch.setattr(tool_host, "_adopted", adopted)
    monkeypatch.setattr(
        tool_host,
        "threading",
        SimpleNamespace(Event=event, Thread=threading.Thread),
    )
    client = _Client()
    # ``http`` twice, a trailing slash, and a last letter ``rstrip`` might take too.
    url = "http://http.X/"
    host = ToolHost(
        url,
        credential=_CREDENTIAL,
        name=_NAME,
        client_for=client,
        dirs=tool_dirs(_NAME, server=url),
    )
    pipe = uuid.UUID(int=3)
    client.answers.put(ToolOffer(pipe_id=pipe))
    host.start(_MACHINE, _INSTANCE)
    # The next wait, answered at once, waits the default before it asks again.
    await _until(lambda: 5.0 in stops[0].waited)
    await asyncio.to_thread(host.stop)

    assert client.waits[0][2] == WAIT_MAX_SEC
    dialed.assert_awaited_once_with(
        "ws://http.X"
        + MACHINE_TOOLS_WS_PATH.format(machine_id=_MACHINE, pipe_id=pipe)
        + f"?instance={_INSTANCE}",
        additional_headers={"Authorization": f"Bearer {_CREDENTIAL}"},
        max_size=FRAME_BYTES,
        ping_interval=20,
        close_timeout=1.0,
    )
    adopted.assert_awaited_once_with(
        ANY,
        socket,
        peer=f"relay:{pipe}",
        end_sec=5.0,
        line_bytes=MAX_LINE_BYTES,
    )


async def test_stopping_a_host_never_started_returns_at_once() -> None:
    host = ToolHost(
        "http://h",
        credential=_CREDENTIAL,
        name=_NAME,
        client_for=_Client(),
        dirs=tool_dirs(_NAME, server="http://h"),
    )

    host.stop()

    assert not _threads()


async def test_no_thread_of_the_tools_keeps_the_process_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay() as relay, _serving(relay):
        await _until(lambda: {"tool-host", "tool-listener"} <= _threads())
        threads = [t for t in threading.enumerate() if t.name.startswith("tool-")]
        assert all(thread.daemon for thread in threads)


async def test_a_410_stops_the_host(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = Gate()
    _local(monkeypatch, Bash=gate)
    async with _relay() as relay:
        host = _host(relay)
        host.start(_MACHINE, _INSTANCE)
        leg = await relay.leg()
        assert isinstance(await leg.open_session(), Welcome)
        await leg.say(_call("1", "Bash"))
        await _until(gate.started.is_set)

        relay.client.answers.put(_REVOKED)

        await leg.closed()
        await _until(gate.stopped.is_set)
        await _until(lambda: not _threads())
        assert relay.client.closed.is_set()
        # Nothing was left to stop, and the state's lock is free for a next host.
        await asyncio.to_thread(host.stop)
        _assert_lock_free(_dirs(relay))


async def test_a_wait_that_fails_is_retried_and_says_so(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay() as relay, _serving(relay):
        relay.client.answers.put(ClientError("-> 503", status_code=503))
        relay.client.answers.put(ValueError("no json\nsecond line"))
        leg = await relay.leg()
        assert isinstance(await leg.open_session(), Welcome)
    said = capsys.readouterr().err.splitlines()
    assert sum("tool wait failed" in line for line in said) == 2
    assert not any(line.startswith("second line") for line in said)


async def test_stop_ends_running_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = Gate()
    _local(monkeypatch, Bash=gate)
    async with _relay() as relay:
        host = _host(relay)
        host.start(_MACHINE, _INSTANCE)
        running = await relay.leg()
        assert isinstance(await running.open_session("s1"), Welcome)
        await running.say(_call("1", "Bash"))
        await _until(gate.started.is_set)
        idle = await relay.leg()
        assert isinstance(await idle.open_session("s2"), Welcome)
        # A leg that never said hello is a bridge too.
        silent = await relay.leg()

        started = time.monotonic()
        await asyncio.to_thread(host.stop)

        assert time.monotonic() - started < HEARTBEAT_SEC
        assert gate.stopped.is_set()
        for leg in (running, idle, silent):
            await leg.closed()
        await _until(lambda: not _threads())
        assert relay.client.closed.is_set()
        _assert_lock_free(_dirs(relay))


async def test_stop_waits_for_a_bridge_still_closing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay() as relay:
        host = _host(relay)
        host.start(_MACHINE, _INSTANCE)
        leg = await relay.leg()
        await leg.say(_hello("s1", version=VERSION + 1))
        assert isinstance(await leg.next(), Refused)

        # Its pipe is over, and its bridge waits for the relay to answer its close. On
        # the relay's own event loop, no answer comes before this returns.
        host.stop()

        await leg.closed()


async def test_an_offer_handed_over_as_the_host_stops_dials_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _local(monkeypatch, Read=Read())

    class _LateOffer(_Client):
        """Its wait, still held at the stop, answers with an offer only after it."""

        @override
        def wait_tools(
            self,
            machine_id: uuid.UUID,
            *,
            instance: uuid.UUID,
            wait_sec: float,
        ) -> ToolOffer | None:
            self.waits.append((machine_id, instance, wait_sec))
            _ = host_stopped.wait(10)
            return ToolOffer(pipe_id=uuid.uuid4())

    host_stopped = threading.Event()
    async with _relay() as relay:
        relay.client = _LateOffer()
        host = _host(relay)
        host.start(_MACHINE, _INSTANCE)
        await _until(lambda: bool(relay.client.waits))
        await asyncio.to_thread(host.stop)
        host_stopped.set()
        await _until(lambda: not _threads())
        assert relay.dials == []


@pytest.mark.parametrize("raised", [AttributeError, RecursionError, TypeError])
async def test_what_a_pipe_raises_ends_only_that_pipe(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    raised: type[Exception],
) -> None:
    gate = Gate()
    local = LocalHost(tools={"Bash": gate})
    loops: list[asyncio.AbstractEventLoop] = []

    async def start(tools: Sequence[str], directory: Path) -> Host:
        loops.append(asyncio.get_running_loop())
        if directory.name == "bad":
            raise raised("hostile\n" + "x" * 10_000)
        return await local(tools, directory)

    monkeypatch.setattr(tool_host, "scrubbed_host", start)
    async with _relay() as relay, _serving(relay):
        other = await relay.leg()
        assert isinstance(await other.open_session("s1"), Welcome)
        await other.say(_call("1", "Bash"))
        await _until(gate.started.is_set)

        bad = await relay.leg()
        bad_id = relay.offered[-1]
        await bad.say(_hello("bad"))
        await bad.closed()

        # The other pipe's call goes on, and the host still takes offers.
        loops[0].call_soon_threadsafe(gate.opened.set)
        assert "open for" in _content(await _result(other, "1"))
        after = await relay.leg()
        assert isinstance(await after.open_session("s2"), Welcome)

    said = [
        line
        for line in capsys.readouterr().err.splitlines()
        if "tool pipe failed" in line
    ]
    # The type alone: what a hostile line raises may hold its text.
    assert said == [f"knowop: tool pipe failed: pipe={bad_id} error={raised.__name__}"]


async def test_a_pipe_says_once_when_it_is_adopted_and_once_how_it_ended(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _local(monkeypatch, Read=Read())
    said: list[str] = []

    def heard(text: str) -> bool:
        said.extend(capsys.readouterr().err.splitlines())
        return any(text in line for line in said)

    async with _relay() as relay, _serving(relay):
        leg = await relay.leg()
        pipe_id = relay.offered[-1]
        assert isinstance(await leg.open_session(), Welcome)
        await leg.socket.close(wire_terminals.CLOSE_ENDED, "the agent left")
        await _until(lambda: heard("tool pipe ended"))
    _ = heard("")

    assert [line for line in said if "tool pipe" in line] == [
        f"knowop: tool pipe adopted: pipe={pipe_id}",
        (
            f"knowop: tool pipe ended: pipe={pipe_id} code={wire_terminals.CLOSE_ENDED} "
            "reason='the agent left'"
        ),
    ]


async def test_a_line_too_deep_to_decode_ends_only_its_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay() as relay, _serving(relay):
        bad = await relay.leg()
        for start in range(0, 200_000, FRAME_BYTES):
            await bad.socket.send(b"[" * min(FRAME_BYTES, 200_000 - start))
        await bad.socket.send(b"\n")
        await bad.closed()
        good = await relay.leg()
        assert isinstance(await good.open_session(), Welcome)


async def test_a_refusal_reaches_the_server_before_the_leg_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay() as relay, _serving(relay):
        leg = await relay.leg()
        await leg.say(_hello("s1", version=VERSION + 1))
        assert isinstance(await leg.next(), Refused)
        await leg.closed()


async def test_a_refused_dial_is_one_line_and_the_host_goes_on(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay() as relay, _serving(relay):
        relay.refusing = True
        relay.offer()
        await _until(lambda: "tool pipe" in capsys.readouterr().err)
        leg = await relay.leg()
        assert isinstance(await leg.open_session(), Welcome)


async def test_a_held_lock_says_so_once_and_serves_when_it_frees(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay() as relay:
        holder = _HeldLock(_dirs(relay))
        async with _serving(relay):
            await _until(lambda: "tools: off" in capsys.readouterr().err)
            # It goes on trying, saying nothing more, and the rest of the host goes on.
            await asyncio.sleep(0.05)
            assert capsys.readouterr().err == ""
            assert relay.client.waits == []

            holder.close()

            leg = await relay.leg()
            assert isinstance(await leg.open_session(), Welcome)
            assert "tools: on" in capsys.readouterr().err


async def test_a_lock_freed_within_the_wait_serves_after_all(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay() as relay:
        holder = _HeldLock(_dirs(relay))
        async with _serving(relay, lock_wait_sec=30):
            await asyncio.sleep(0.02)
            assert relay.client.waits == []
            holder.close()
            leg = await relay.leg()
            assert isinstance(await leg.open_session(), Welcome)
    said = capsys.readouterr().err
    assert "tools: off" not in said
    assert "tools: on" not in said


async def test_stop_ends_a_wait_for_the_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay() as relay:
        with _HeldLock(_dirs(relay)):
            host = _host(relay, lock_wait_sec=600)
            host.start(_MACHINE, _INSTANCE)
            await asyncio.sleep(0.02)
            started = time.monotonic()
            await asyncio.to_thread(host.stop)
            assert time.monotonic() - started < 5
            await _until(lambda: not _threads())


async def test_a_state_folder_it_cannot_make_is_one_line_and_no_tools(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _local(monkeypatch, Read=Read())
    file = tmp_path / "a-file"
    file.touch()
    dirs = ServeDirs(secrets=file / "m" / "secrets", state=file / "m" / "state")
    async with _relay() as relay:
        host = _host(relay, dirs=dirs)
        host.start(_MACHINE, _INSTANCE)
        await _until(lambda: "tools: off" in capsys.readouterr().err)
        await asyncio.to_thread(host.stop)
        assert relay.client.waits == []


async def test_it_opens_no_port(monkeypatch: pytest.MonkeyPatch) -> None:
    _local(monkeypatch, Read=Read())
    tcp = AsyncMock(side_effect=AssertionError("a port was opened"))
    unix = AsyncMock(side_effect=AssertionError("a socket was opened"))
    monkeypatch.setattr(asyncio, "start_server", tcp)
    monkeypatch.setattr(asyncio, "start_unix_server", unix)
    async with _relay() as relay, _serving(relay):
        leg = await relay.leg()
        assert isinstance(await leg.open_session(), Welcome)
        config = cast(
            dict[str, object],
            json.loads((_dirs(relay).state / "server.json").read_text()),
        )
    assert config["listen"] == []
    tcp.assert_not_awaited()
    unix.assert_not_awaited()


async def test_its_folders_are_this_users_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay() as relay:
        dirs = _dirs(relay)
        tools = state_dir() / "rekursiv-ai" / "knowop" / "tools"
        root = tools / server_dir(relay.url) / _NAME
        assert (dirs.secrets.parent, dirs.state.parent) == (root, root)
        # An earlier run under another umask made it wider; this one narrows it.
        root.mkdir(parents=True, mode=0o755)
        root.chmod(0o755)
        async with _serving(relay):
            leg = await relay.leg()
            assert isinstance(await leg.open_session(), Welcome)
            assert stat.S_IMODE(root.stat().st_mode) == 0o700
            assert stat.S_IMODE(dirs.state.stat().st_mode) == 0o700
            assert stat.S_IMODE((dirs.state / "server.pid").stat().st_mode) == 0o600
            sessions = dirs.state / "sessions"
            assert stat.S_IMODE((sessions / "s1").stat().st_mode) == 0o700


async def test_new_tools_folders_are_this_users_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay() as relay, _serving(relay):
        leg = await relay.leg()
        assert isinstance(await leg.open_session(), Welcome)
        root = _dirs(relay).state.parent
    assert {
        stat.S_IMODE(folder.stat().st_mode)
        for folder in (root, root.parent, root.parent.parent)
    } == {0o700}


async def test_one_machine_name_on_two_servers_serves_both(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay() as first, _relay() as second:
        assert _dirs(first) != _dirs(second)
        async with _serving(first), _serving(second):
            for relay in (first, second):
                leg = await relay.leg()
                assert isinstance(await leg.open_session(), Welcome)
    assert "tools: off" not in capsys.readouterr().err


async def test_the_host_program_runs_under_env_without_the_trax_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in list(os.environ):
        if name.startswith(("TRACKINIZER_", "TRAX_", "LC_TRAX_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("TRACKINIZER_TOKEN", "secret")
    monkeypatch.setenv("TRACKINIZER_SOMETHING_NEW", "new")
    monkeypatch.setenv("TRAX_RUN", "r")
    monkeypatch.setenv("LC_TRAX_SERVER", "s")
    monkeypatch.setenv("KNOWOP_TOOLS_TOKEN", "t")
    started = AsyncMock(return_value="the host")
    monkeypatch.setattr("sagent.host.server.child_host", started)

    assert await scrubbed_host(("Read",), tmp_path) == "the host"

    started.assert_awaited_once_with(
        ("Read",),
        tmp_path,
        command=(
            "/usr/bin/env",
            "-u",
            "KNOWOP_TOOLS_TOKEN",
            "-u",
            "LC_TRAX_SERVER",
            "-u",
            "TRACKINIZER_SOMETHING_NEW",
            "-u",
            "TRACKINIZER_TOKEN",
            "-u",
            "TRAX_RUN",
            *HOST_COMMAND,
        ),
    )


@pytest.mark.parametrize(
    ("seam", "module"),
    [
        ("sagent_server", "sagent.host.absent_from_this_install"),
        ("tool_bridge", "trackinizer.client.absent_from_this_install"),
        ("wire_tools", "trackinizer.wire.absent_from_this_install"),
    ],
)
async def test_an_install_without_what_tools_need_says_so_once(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    seam: str,
    module: str,
) -> None:
    assert tool_host.can_serve()
    assert capsys.readouterr().err == ""

    monkeypatch.setattr(tool_host, seam, lazy_import(module))

    assert not tool_host.can_serve()
    said = capsys.readouterr().err.splitlines()
    assert len(said) == 1
    assert "tools: off" in said[0]


async def test_a_sagent_without_server_adopt_serves_no_tools(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _OlderServer:
        """sagent 0.2.3's ``Server``: a host module, and no ``adopt``."""

    monkeypatch.setattr(
        tool_host,
        "sagent_server",
        SimpleNamespace(HOST_COMMAND=HOST_COMMAND, Server=_OlderServer),
    )

    assert not tool_host.can_serve()
    (said,) = capsys.readouterr().err.splitlines()
    assert "tools: off" in said
    assert "adopt" in said


@pytest.mark.parametrize(
    "command",
    [
        pytest.param(("/nowhere/python",), id="missing-program"),
        pytest.param(("/usr/bin/false",), id="exits-at-once"),
    ],
)
async def test_a_host_program_that_will_not_start_says_so(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: tuple[str, ...],
) -> None:
    monkeypatch.setattr("sagent.host.server.HOST_COMMAND", command)
    # With no handler configured, as for ``knowop connect``, a warning reaches stderr.
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    async with _relay() as relay, _serving(relay):
        leg = await relay.leg()
        await leg.say(_hello("s1"))
        await leg.closed()
    said = capsys.readouterr().err.splitlines()
    assert sum("did not start" in line for line in said) == 1


async def test_pipes_stop_at_the_cap_and_a_slot_frees_when_one_ends(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _local(monkeypatch, Read=Read())
    caplog.set_level(logging.WARNING, logger=tool_host.__name__)
    async with _relay() as relay, _serving(relay, max_pipes=2):
        legs = [await relay.leg() for _ in range(2)]

        relay.offer()

        refused = f"tool pipe refused: pipe={relay.offered[-1]} open=2 max=2"
        await _until(
            lambda: [r.getMessage() for r in caplog.records] == [refused],
        )
        assert len(relay.dials) == 2

        await legs[0].socket.close()

        async def next_leg() -> _Leg:
            while True:
                relay.offer()
                with contextlib.suppress(TimeoutError):
                    return await asyncio.wait_for(relay.legs.get(), 0.02)

        leg = await asyncio.wait_for(next_leg(), 10)
        assert isinstance(await leg.open_session(), Welcome)
        assert len(relay.dials) == 3


async def test_a_dial_back_the_server_does_not_answer_ends_at_its_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay() as relay, _serving(relay, dial_sec=0.05):
        relay.silent = True
        relay.offer()

        async with asyncio.timeout(5):
            await _until(lambda: "TimeoutError" in capsys.readouterr().err)
        relay.spoken.set()


class _AnsweredAtOnce(_Client):
    """A server that does not hold the wait."""

    @override
    def wait_tools(
        self,
        machine_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        wait_sec: float,
    ) -> ToolOffer | None:
        self.waits.append((machine_id, instance, wait_sec))
        return None


async def test_a_wait_answered_at_once_is_not_asked_again_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay() as relay:
        relay.client = _AnsweredAtOnce()
        host = _host(relay, retry_sec=0.01)
        host.start(_MACHINE, _INSTANCE)
        await asyncio.sleep(0.05)
        await asyncio.to_thread(host.stop)
        # Asked again at once, it would ask thousands of times in this time.
        assert 1 <= len(relay.client.waits) <= 10


async def test_a_wait_held_to_its_end_is_asked_again_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _local(monkeypatch, Read=Read())

    class _Held(_Client):
        @override
        def wait_tools(
            self,
            machine_id: uuid.UUID,
            *,
            instance: uuid.UUID,
            wait_sec: float,
        ) -> ToolOffer | None:
            self.waits.append((machine_id, instance, wait_sec))
            time.sleep(wait_sec)
            return None

    async with _relay() as relay:
        relay.client = _Held()
        async with asyncio.timeout(10):
            host = _host(relay, retry_sec=60, wait_sec=0.01)
            host.start(_MACHINE, _INSTANCE)
            await _until(lambda: len(relay.client.waits) >= 3)
            await asyncio.to_thread(host.stop)
        assert {wait_sec for _, _, wait_sec in relay.client.waits} == {0.01}


async def test_a_wait_failing_the_same_way_is_said_once(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _local(monkeypatch, Read=Read())
    refused = ClientError("-> 403 stale", status_code=403)
    async with _relay() as relay, _serving(relay):
        for _ in range(3):
            relay.client.answers.put(refused)
        await _until(relay.client.answers.empty)
        leg = await relay.leg()
        assert isinstance(await leg.open_session(), Welcome)
        # A failure after a wait that worked is news again.
        relay.client.answers.put(refused)
        await _until(relay.client.answers.empty)
        await _until(lambda: len(relay.client.waits) > 6)
    said = capsys.readouterr().err.splitlines()
    assert sum("tool wait failed" in line for line in said) == 2


async def test_a_client_that_cannot_be_made_is_one_line_and_ends_the_tools(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _local(monkeypatch, Read=Read())

    def no_client(key: str) -> _Client:
        raise ClientError(f"no client for {key}")

    async with _relay() as relay:
        monkeypatch.setattr(relay, "client", no_client)
        host = _host(relay)
        host.start(_MACHINE, _INSTANCE)
        await _until(lambda: not _threads())
        assert "tools: off" in capsys.readouterr().err
        await asyncio.to_thread(host.stop)
        _assert_lock_free(_dirs(relay))


class _Failing(ToolHost):
    """A host whose serving loop raises, as a bug in it would."""

    @override
    async def _serve(self, machine_id: uuid.UUID, instance: uuid.UUID) -> None:
        raise RuntimeError("the serving loop broke")


async def test_stop_says_so_when_the_tools_outlast_its_wait(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # This test's own, so a second run in one process (as mutmut's) finds it unset.
    release = threading.Event()

    class _Unstoppable(ToolHost):
        """A host whose serving thread ignores the stop until the test lets it go."""

        @override
        async def _serve(self, machine_id: uuid.UUID, instance: uuid.UUID) -> None:
            _ = await asyncio.to_thread(release.wait, 10)

    async with _relay() as relay:
        host = _Unstoppable(
            relay.url,
            credential=_CREDENTIAL,
            name=_NAME,
            client_for=relay.client,
            dirs=_dirs(relay),
            stop_sec=0.05,
        )
        host.start(_MACHINE, _INSTANCE)
        try:
            await asyncio.to_thread(host.stop)
            assert "still stopping" in capsys.readouterr().err
        finally:
            release.set()
            await _until(lambda: not _threads())


async def test_a_serving_loop_that_raises_says_so_and_ends_the_listener(
    capsys: pytest.CaptureFixture[str],
) -> None:
    async with _relay() as relay:
        host = _Failing(
            relay.url,
            credential=_CREDENTIAL,
            name=_NAME,
            client_for=relay.client,
            dirs=_dirs(relay),
        )
        host.start(_MACHINE, _INSTANCE)
        await _until(lambda: not _threads())
        assert "tools: stopped: RuntimeError('the serving loop broke')" in (
            capsys.readouterr().err
        )
        await asyncio.to_thread(host.stop)


class _OfferedAsStopped(ToolHost):
    """A host whose listener hands over an offer in the same breath as the stop."""

    @override
    def _listen(
        self,
        machine_id: uuid.UUID,
        instance: uuid.UUID,
        loop: asyncio.AbstractEventLoop,
        start_pipe: Callable[[ToolOffer], None],
    ) -> None:
        # The loop is held while the offer and the stop are queued, so the offer
        # is always started after the stop: an offer racing a stop is made certain.
        held = threading.Event()
        _ = loop.call_soon_threadsafe(held.wait, 10)
        _ = loop.call_soon_threadsafe(start_pipe, ToolOffer(pipe_id=uuid.uuid4()))
        self._stop.set()
        held.set()


async def test_an_offer_started_after_the_stop_opens_no_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[ToolOffer] = []

    def pipe(
        self: ToolHost,
        server: object,
        offer: ToolOffer,
        **_: object,
    ) -> Coroutine[None, None, None]:
        del self, server
        opened.append(offer)
        return asyncio.sleep(0)

    monkeypatch.setattr(ToolHost, "_pipe", pipe)
    async with _relay() as relay:
        host = _OfferedAsStopped(
            relay.url,
            credential=_CREDENTIAL,
            name=_NAME,
            client_for=relay.client,
            dirs=_dirs(relay),
        )
        host.start(_MACHINE, _INSTANCE)
        await _until(lambda: not _threads())
        await asyncio.to_thread(host.stop)
    assert opened == []


@pytest.mark.parametrize("raised", [OSError, asyncio.CancelledError])
async def test_a_pipe_whose_bridge_cannot_start_closes_its_leg_and_the_host_goes_on(
    monkeypatch: pytest.MonkeyPatch,
    raised: type[BaseException],
) -> None:
    bridged = tool_bridge.bridged
    failed: list[bool] = []

    async def failing_once(
        socket: ClientConnection,
        *,
        limit: int,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        if not failed:
            failed.append(True)
            raise raised
        return await bridged(socket, limit=limit)

    monkeypatch.setattr(tool_bridge, "bridged", failing_once)
    _local(monkeypatch, Read=Read())
    async with _relay() as relay, _serving(relay):
        await (await relay.leg()).closed()
        leg = await relay.leg()
        assert isinstance(await leg.open_session(), Welcome)


async def test_a_frame_over_the_limit_ends_its_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay() as relay, _serving(relay):
        leg = await relay.leg()
        await leg.socket.send(b"x" * (FRAME_BYTES + 1))
        await leg.closed()
        assert leg.socket.close_code == CloseCode.MESSAGE_TOO_BIG
        good = await relay.leg()
        assert isinstance(await good.open_session(), Welcome)


async def test_a_line_over_the_limit_ends_its_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _local(monkeypatch, Read=Read())
    limit = 1 << 20
    async with _relay() as relay, _serving(relay, line_bytes=limit):
        leg = await relay.leg()
        frame = b"x" * FRAME_BYTES
        with contextlib.suppress(ConnectionClosed):
            for _ in range(limit // FRAME_BYTES + 2):
                await leg.socket.send(frame)
        await leg.closed()


async def test_the_hosts_reader_takes_sagents_own_limit_past_the_relays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The relay caps what an agent sends; what the host reads is sagent's to cap."""
    _local(monkeypatch, Read=Read())
    padded = b'", "pad": "' + b"x" * Config().tool_line_bytes + b'"}\n'
    line = encoded(Describe(id="d1", tool="Read")).removesuffix(b'"}\n') + padded
    async with _relay() as relay, _serving(relay):
        leg = await relay.leg()
        assert isinstance(await leg.open_session(), Welcome)
        for start in range(0, len(line), FRAME_BYTES):
            await leg.socket.send(line[start : start + FRAME_BYTES])
        answer = await leg.next()
    assert isinstance(answer, Described)
    assert answer.id == "d1"


async def test_a_note_to_another_session_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay() as relay, _serving(relay):
        sender = await relay.leg()
        assert isinstance(await sender.open_session("s1"), Welcome)
        other = await relay.leg()
        assert isinstance(await other.open_session("s2"), Welcome)

        await sender.say(
            Note(
                id="n1",
                sender="Agent",
                recipient="Agent#s2",
                content="hi",
                kind="tell",
            ),
        )

        refused = await sender.next()
        assert isinstance(refused, Refused)
        assert refused.id == "n1"
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(other.socket.recv(), 0.05)


class _Canary:
    """What a line must never build or call: each says so."""

    touched: ClassVar[list[str]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        _Canary.touched.append("built")


def _canary_called(*args: object) -> None:
    del args
    _Canary.touched.append("called")


@dataclass(frozen=True, slots=True, kw_only=True)
class _Spy:
    """A tool that notes each call it ran."""

    name = "Spy"
    description = "Note the call."
    schema: Mapping[str, PlainTree] = field(default_factory=lambda: {"type": "object"})
    display: ToolDisplay = field(default=ToolDisplay())
    ran: list[str] = field(default_factory=list[str])

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        del arguments
        return "Spy"

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        del context, arguments

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> FunctionResult:
        del context
        self.ran.append(repr(arguments)[:100])
        return FunctionResult(call_id="", name="Spy", content="ran")


@dataclass(frozen=True, slots=True, kw_only=True)
class _Hostile:
    """A line an attacker holding the assistant's key could send a host."""

    line: bytes
    hello: bytes = b""
    """What goes first, to have a session; empty when the line is the first."""

    runs: bool | None = False
    """Whether the granted tool runs; ``None`` when either is fine."""


_SENTINEL: Final = "@SENTINEL@"
"""In a line, for a file the test names; a line that acts makes it."""

_HELLO_LINE: Final[dict[str, object]] = {
    "py/object": "Hello",
    "version": VERSION,
    "token": "",
    "session_id": "hostile",
    "client": "c",
    "tools": ["Read", "Write"],
}

_CALL_LINE: Final[dict[str, object]] = {
    "py/object": "Call",
    "id": "h1",
    "agent": "a",
    "window": "w",
    "tool": "Write",
    "arguments": {},
    "address_book": ["a"],
    "granted": ["Write"],
}

_FIELD_LINES: Final[tuple[dict[str, object], ...]] = (
    _HELLO_LINE,
    _CALL_LINE,
    {"py/object": "Cancel", "id": "h1"},
    {"py/object": "Ack", "id": "h1"},
    {"py/object": "Describe", "id": "d1", "tool": "Read"},
    {
        "py/object": "Note",
        "id": "n1",
        "sender": "a",
        "recipient": "b",
        "content": "hi",
        "kind": "tell",
    },
    {"py/object": "Peers", "labels": ["a"], "public": ["a"]},
)

_OUTSIDE_TAGS: Final[dict[str, dict[str, object]]] = {
    "object": {"py/object": f"{__name__}._Canary"},
    "popen": {"py/object": "subprocess.Popen", "args": ["touch", _SENTINEL]},
    "reduce": {
        "py/reduce": [
            {"py/function": "os.system"},
            {"py/tuple": [f"touch {_SENTINEL}"]},
        ],
    },
    "reduce-class": {
        "py/reduce": [{"py/type": f"{__name__}._Canary"}, {"py/tuple": []}],
    },
    "function": {"py/function": f"{__name__}.{_canary_called.__name__}"},
    "type": {"py/type": f"{__name__}._Canary"},
    "hook": {"py/hook": [f"{__name__}.{_canary_called.__name__}", {}]},
}
"""Each way a tag might name code. No field reads any of these."""

_READ_TAGS: Final[dict[str, dict[str, object]]] = {
    "allowed-object": {"py/object": "DaemonControl", "stop": True},
    "path": {"py/path": "/etc/passwd"},
}
"""Tags the wire reads, of a type the field does not declare: below the root a tag decides."""

_TAG_DOES_NOT_STOP_A_CALL: Final = frozenset(
    {"id", "agent", "window", "arguments", "granted"},
)
"""The ``Call`` fields no step before the tool looks into, so a read tag there runs it."""

_WRONG_VALUES: Final[dict[str, object]] = {
    "int": 5,
    "null": None,
    "bool": True,
    "list": [],
    "dict": {},
}


def _json_line(fields: Mapping[str, object]) -> bytes:
    return json.dumps(fields).encode() + b"\n"


def _in_a_session(
    line: Mapping[str, object] | bytes,
    *,
    runs: bool | None = False,
) -> _Hostile:
    return _Hostile(
        line=line if isinstance(line, bytes) else _json_line(line),
        hello=_json_line(_HELLO_LINE),
        runs=runs,
    )


def _with_field(
    line: Mapping[str, object],
    key: str,
    value: object,
    *,
    runs: bool,
) -> _Hostile:
    changed = {**line, key: value}
    if line is _HELLO_LINE:
        return _Hostile(line=_json_line(changed))
    return _in_a_session(changed, runs=runs if line is _CALL_LINE else False)


def _each_field() -> dict[str, _Hostile]:
    """Return every field of every line a client sends, set to each bad value."""
    cases: dict[str, _Hostile] = {}
    for line in _FIELD_LINES:
        kind = str(line["py/object"])
        for key in line:
            if key == "py/object":
                continue
            for name in ("object", "reduce"):
                cases[f"{kind}-{key}-outside-{name}"] = _with_field(
                    line,
                    key,
                    _OUTSIDE_TAGS[name],
                    runs=key == "arguments",
                )
            for name, tag in _READ_TAGS.items():
                runs = key in _TAG_DOES_NOT_STOP_A_CALL
                cases[f"{kind}-{key}-read-{name}"] = _with_field(
                    line,
                    key,
                    tag,
                    runs=runs,
                )
            for name, value in _WRONG_VALUES.items():
                # What its field does take is not a wrong type.
                if (key, name) not in {
                    ("tools", "null"),
                    ("tools", "list"),
                    ("public", "null"),
                    ("public", "list"),
                    ("labels", "list"),
                    ("address_book", "list"),
                    ("granted", "list"),
                    ("arguments", "dict"),
                }:
                    cases[f"{kind}-{key}-wrong-{name}"] = _with_field(
                        line,
                        key,
                        value,
                        runs=False,
                    )
    return cases


# Below ``arguments`` a tag outside the allow-list reads as data, so the tool gets the
# plain mapping; as a ``json://`` key it does not read at all.
def _inside_arguments() -> dict[str, _Hostile]:
    """Return a tag at each place a ``Call``'s arguments can hold one."""
    cases: dict[str, _Hostile] = {}
    for name, tag in {**_OUTSIDE_TAGS, **_READ_TAGS}.items():
        for where, arguments in {
            "value": {"a": tag},
            "nested": {"a": [{"b": [tag]}]},
            "py-dict": {"py/dict": {"a": tag}},
        }.items():
            cases[f"arguments-{where}-{name}"] = _in_a_session(
                {**_CALL_LINE, "arguments": arguments},
                runs=True,
            )
        cases[f"arguments-key-{name}"] = _in_a_session(
            {**_CALL_LINE, "arguments": {"json://" + json.dumps(tag): 1}},
            runs=name in _READ_TAGS,
        )
        cases[f"root-{name}"] = _in_a_session(tag)
        cases[f"first-line-{name}"] = _Hostile(line=_json_line(tag))
    return cases


def _deep_lines() -> dict[str, _Hostile]:
    """Return lines nested from where the writers recurse to past any stack."""
    deep = 200_000
    call = _json_line({**_CALL_LINE, "arguments": "@ARGUMENTS@"})
    cases = {
        "deep-open-array": _in_a_session(b"[" * deep + b"\n"),
        "deep-array": _in_a_session(b"[" * deep + b"]" * deep + b"\n"),
        "deep-object": _in_a_session(b'{"a":' * deep + b"1" + b"}" * deep + b"\n"),
        "deep-first-line": _Hostile(line=b"[" * deep + b"\n"),
        "deep-json-key": _in_a_session(
            {**_CALL_LINE, "arguments": {"json://" + "[" * deep: 1}},
        ),
    }
    # 900 fits every reader and writer on the way, and a tool gets it. Past that, whichever
    # gives out first ends the pipe, and where is the platform's stack: either is fine.
    cases["deep-arguments-900"] = _in_a_session(
        call.replace(b'"@ARGUMENTS@"', b'{"a":' + b"[" * 900 + b"]" * 900 + b"}"),
        runs=True,
    )
    for depth in (1100, 5000, deep):
        cases[f"deep-arguments-{depth}"] = _in_a_session(
            call.replace(
                b'"@ARGUMENTS@"',
                b'{"a":' + b"[" * depth + b"]" * depth + b"}",
            ),
            runs=None,
        )
        cases[f"deep-tags-{depth}"] = _in_a_session(
            call.replace(
                b'"@ARGUMENTS@"',
                b'{"a":' + b'{"py/tuple":[' * depth + b"]}" * depth + b"}",
            ),
            runs=None,
        )
    return cases


def _number_lines() -> dict[str, _Hostile]:
    """Return numbers past what a parser or a tool can hold, wherever a line takes one."""
    call = _json_line({**_CALL_LINE, "arguments": "@ARGUMENTS@", "id": "@ID@"})
    hello = _json_line({**_HELLO_LINE, "version": "@VERSION@"})
    numbers = {
        # A longer integer is more than Python reads from text by default.
        "int-4000": (b"1" * 4000, True),
        "int-5000": (b"1" * 5000, False),
        "exp": (b"1e999", True),
        "negative-exp": (b"-1e999", True),
        "nan": (b"NaN", True),
        "infinity": (b"Infinity", True),
        "negative-infinity": (b"-Infinity", True),
    }
    cases: dict[str, _Hostile] = {}
    for name, (number, in_arguments_runs) in numbers.items():
        cases[f"number-arguments-{name}"] = _in_a_session(
            call.replace(b'"@ARGUMENTS@"', b'{"a":' + number + b"}").replace(
                b'"@ID@"',
                b'"h1"',
            ),
            runs=in_arguments_runs,
        )
        cases[f"number-id-{name}"] = _in_a_session(
            call.replace(b'"@ARGUMENTS@"', b"{}").replace(b'"@ID@"', number),
        )
        cases[f"number-version-{name}"] = _Hostile(
            line=hello.replace(b'"@VERSION@"', number),
        )
    return cases


def _duplicate_lines() -> dict[str, _Hostile]:
    """Return lines whose keys repeat: the last of a key is the one every reader takes."""
    call = _json_line(_CALL_LINE)
    write = b'"tool": "Write"'
    kind = b'"py/object": "Call"'
    arguments = b'"arguments": {}'
    return {
        "duplicate-tool-withheld-last": _in_a_session(
            call.replace(write, write + b', "tool": "Edit"'),
        ),
        "duplicate-tool-withheld-first": _in_a_session(
            call.replace(write, b'"tool": "Edit", ' + write),
            runs=True,
        ),
        "duplicate-tool-escaped": _in_a_session(
            call.replace(write, write + b', "\\u0074ool": "Edit"'),
        ),
        "duplicate-kind-daemon-last": _in_a_session(
            call.replace(kind, kind + b', "py/object": "DaemonControl"'),
        ),
        "duplicate-kind-daemon-first": _in_a_session(
            call.replace(kind, b'"py/object": "DaemonControl", ' + kind),
            runs=True,
        ),
        "duplicate-arguments-tag-first": _in_a_session(
            call.replace(
                arguments,
                b'"arguments": {"py/function": "os.system"}, ' + arguments,
            ),
            runs=True,
        ),
    }


def _tool_lines() -> dict[str, _Hostile]:
    """Return calls of what this session may not have."""
    return {
        "tool-not-granted": _in_a_session({**_CALL_LINE, "tool": "Edit"}),
        "tool-not-served": _in_a_session({**_CALL_LINE, "tool": "Nothing"}),
        "tool-claims-a-grant": _in_a_session(
            {**_CALL_LINE, "tool": "Edit", "granted": ["Edit"]},
        ),
        "tool-grant-not-asked-for": _Hostile(
            hello=_json_line({**_HELLO_LINE, "tools": ["Read"]}),
            line=_json_line(_CALL_LINE),
        ),
        "tool-nul-in-name": _in_a_session({**_CALL_LINE, "tool": "Wri\x00te"}),
    }


def _kind_lines() -> dict[str, _Hostile]:
    """Return lines no client should send a host, and notes it should not take."""
    return {
        "kind-not-json": _in_a_session(b"not json\n"),
        "kind-blank": _in_a_session(b"\n"),
        "kind-not-utf8": _in_a_session(b'{"py/object": "Ping"}\xff\n'),
        "kind-array": _in_a_session(b"[1]\n"),
        "kind-text": _in_a_session(b'"Call"\n'),
        "kind-none": _in_a_session(b'{"id": "x"}\n'),
        "kind-number": _in_a_session(b'{"py/object": 5}\n'),
        "kind-unknown": _in_a_session(b'{"py/object": "Nothing"}\n'),
        "kind-daemon-control": _in_a_session(
            {"py/object": "DaemonControl", "stop": True},
        ),
        "kind-gate-call": _in_a_session(
            {"py/object": "GateCall", "op": "reset", "key": "k"},
        ),
        "kind-pause": _in_a_session({"py/object": "Pause"}),
        "kind-tools": _in_a_session({"py/object": "Tools", "tools": {}}),
        "note-to-another-session": _in_a_session(
            {
                "py/object": "Note",
                "id": "n1",
                "sender": "a",
                "recipient": "a#witness",
                "content": "hi",
                "kind": "tell",
            },
        ),
        "note-kind-unknown": _in_a_session(
            {
                "py/object": "Note",
                "id": "n1",
                "sender": "a",
                "recipient": "b",
                "content": "hi",
                "kind": "nothing",
            },
        ),
        "note-recipient-holds-the-mark": _in_a_session(
            {
                "py/object": "Note",
                "id": "n1",
                "sender": "a",
                "recipient": {"py/list": ["#"]},
                "content": "hi",
                "kind": "tell",
            },
        ),
        "unknown-field-call": _in_a_session({**_CALL_LINE, "x": 1}, runs=None),
        "unknown-field-hello": _Hostile(line=_json_line({**_HELLO_LINE, "x": 1})),
        "second-hello": _in_a_session(_HELLO_LINE),
        "second-hello-other-session": _in_a_session(
            {**_HELLO_LINE, "session_id": "other"},
        ),
    }


def _session_lines() -> dict[str, _Hostile]:
    """Return a ``Hello`` naming each session a folder must not be made for, or may be."""
    odd = {
        "empty": "",
        "traversal": "../../etc",
        "dots": "..",
        "dot": ".",
        "slash": "a/b",
        "absolute": "/etc/passwd",
        "space": "a b",
        "nul": "a\x00",
        "newline": "a\nb",
        "unicode": "é",
        "surrogate": "\ud800",
        "flag": "--help",
        "longest-name": "x" * 255,
        "over-a-name": "x" * 256,
        "long": "x" * 5000,
        "huge": "x" * 1_000_000,
    }
    return {
        f"session-{name}": _Hostile(
            line=_json_line({**_HELLO_LINE, "session_id": session}),
        )
        for name, session in odd.items()
    }


def _wire_class_lines() -> dict[str, _Hostile]:
    """Return a field that takes text holding a wire message: a real child exits on it."""
    return {
        "wire-class-address-book": _in_a_session(
            {**_CALL_LINE, "address_book": {"py/object": "Ping"}},
            runs=None,
        ),
        "wire-class-note-sender": _in_a_session(
            {
                "py/object": "Note",
                "id": "n1",
                "sender": {"py/object": "Ping"},
                "recipient": "b",
                "content": "hi",
                "kind": "tell",
            },
        ),
    }


_HOSTILE: Final = {
    **_each_field(),
    **_inside_arguments(),
    **_deep_lines(),
    **_number_lines(),
    **_duplicate_lines(),
    **_tool_lines(),
    **_kind_lines(),
    **_session_lines(),
    **_wire_class_lines(),
}


async def _send_framed(leg: _Leg, raw: bytes) -> None:
    """Send ``raw`` in frames a host's websocket takes; a pipe already ended is fine."""
    with contextlib.suppress(ConnectionClosed):
        for start in range(0, len(raw), FRAME_BYTES):
            await leg.socket.send(raw[start : start + FRAME_BYTES])


# The host reads a pipe's lines in order, and starts each call in order. So a ``Read``
# sent after a line is answered after whatever that line did has begun.
async def _settled(leg: _Leg, readable: Path) -> None:
    """Return once the host has dealt with what it was sent, or has ended the pipe."""
    await _send_framed(leg, encoded(_call("settle", "Read", file_path=str(readable))))
    while True:
        try:
            message = await leg.next()
        except (
            ValueError
        ):  # A line longer than this test's reader takes, a hostile id's.
            continue
        if message is None or (isinstance(message, Result) and message.id == "settle"):
            return


@pytest.mark.parametrize("name", list(_HOSTILE))
async def test_a_hostile_line_acts_on_nothing_and_stays_on_its_pipe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
) -> None:
    hostile = _HOSTILE[name]
    readable = tmp_path / "readable.txt"
    readable.write_text("readable\n")
    sentinel = tmp_path / "sentinel"
    gate, spy, withheld = Gate(), _Spy(), _Spy()
    local = LocalHost(
        tools={"Bash": gate, "Write": spy, "Edit": withheld, "Read": Read()},
    )
    loops: list[asyncio.AbstractEventLoop] = []

    async def start(tools: Sequence[str], directory: Path) -> Host:
        loops.append(asyncio.get_running_loop())
        return await local(tools, directory)

    monkeypatch.setattr(tool_host, "scrubbed_host", start)
    _Canary.touched.clear()
    async with _relay() as relay, _serving(relay):
        witness = await relay.leg()
        assert isinstance(await witness.open_session("witness"), Welcome)
        await witness.say(_call("w1", "Bash"))
        await _until(gate.started.is_set)

        leg = await relay.leg()
        await _send_framed(leg, hostile.hello)
        await _send_framed(
            leg,
            hostile.line.replace(_SENTINEL.encode(), str(sentinel).encode()),
        )
        await _settled(leg, readable)
        await leg.socket.close()

        # Another pipe takes a new session, and reads.
        fresh = await relay.leg()
        assert isinstance(await fresh.open_session("fresh"), Welcome)
        await fresh.say(_call("f1", "Read", file_path=str(readable)))
        assert "readable" in _content(await _result(fresh, "f1"))
        # The one that was already running went on through it all.
        assert not gate.stopped.is_set()
        loops[0].call_soon_threadsafe(gate.opened.set)
        assert "open for" in _content(await _result(witness, "w1"))

        state = _dirs(relay).state
        assert {folder.name for folder in state.parent.iterdir()} <= {
            "secrets",
            "state",
        }
        for folder in (state / "sessions").iterdir():
            assert re.fullmatch(r"[A-Za-z0-9_-]+", folder.name)

    assert _Canary.touched == []
    assert not sentinel.exists()
    assert withheld.ran == []
    if hostile.runs is not None:
        assert bool(spy.ran) == hostile.runs


async def test_importing_the_tool_host_does_not_load_sagents_host() -> None:
    graph = import_graph(
        [tool_host.__name__],
        path=[str(_ROOT), *sys.path],
        follow=_first_party,
        counted=Counted.IMPORT_TIME,
    )
    reached = {*graph, *(name for names in graph.values() for name in names)}
    assert ClientError.__module__ in reached
    host = ServeDirs.__module__.rpartition(".")[0]
    assert sorted(name for name in reached if name.startswith(host)) == []


def _first_party(module: str) -> bool:
    return module.split(".", maxsplit=1)[0] in _FIRST_PARTY


class _HeldLock:
    """Hold ``server.pid``'s lock, as another host of this machine would."""

    def __init__(self, dirs: ServeDirs) -> None:
        dirs.state.mkdir(parents=True, mode=0o700)
        self._fd = os.open(dirs.state / "server.pid", os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _ = os.pwrite(self._fd, b"4242", 0)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        os.close(self._fd)


def _assert_lock_free(dirs: ServeDirs) -> None:
    fd = os.open(dirs.state / "server.pid", os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
