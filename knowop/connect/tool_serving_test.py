"""ToolServing over an in-memory websocket, each session's sagent host in this process.

A stand-in relay answers each dial with one end of an in-memory websocket, and each
daemon's stop waits out no retry, so no test opens a port or waits real time. The
same checks over real sockets, real children and real timeouts are hand-run:
``private/tool_serving_parity.py``. A real child is sagent's to test
(``daemon/server_test.py``, ``tools/bash_test.py``); here, ``scrubbed_host`` is pinned
by the command it starts the child with.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar, Final, Protocol, Self, cast, override
from unittest.mock import ANY, AsyncMock, Mock

import asyncio
import contextlib
import fcntl
import json
import logging
import os
import re
import stat
import sys
import threading
import time
import uuid

from sagent.host import server as sagent_host_server
from sagent.host.server import DEFAULT_TOOLS, HOST_COMMAND
from sagent.host.strings import Grant, ServeDirs
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
from sagent.session.types import (
    ReadResult,
    UncategorizedToolResult,
)
from sagent.tools.display import ToolDisplay
from sagent.tools.read import Read
from sagent.tools.types import ToolOutput, whole_output
from trackinizer.client import tool_bridge
from trackinizer.client.errors import ClientError
from trackinizer.lib.userdirs import state_dir
from trackinizer.trax.connection import server_dir
from trackinizer.trax.render import echo
from trackinizer.wire import wire_terminals
from trackinizer.wire.wire_daemon import HEARTBEAT_SEC
from trackinizer.wire.wire_tools import (
    FRAME_BYTES,
    HOST_TOOLS_WS_PATH,
    WAIT_MAX_SEC,
    ToolOffer,
)
from treekle import PlainTree
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
)
from websockets.frames import Close, CloseCode
from wrapt import lazy_import

import pytest

from knowop.connect import tool_serving
from knowop.connect.conftest import _status
from knowop.connect.tool_serving import (
    ToolServing,
    scrubbed_host,
    tool_dirs,
)
from knowop.connect.wake import Wake
from knowop.lib.testing.import_graph import Counted, import_graph


_THIS: Final[Path] = Path(__file__).resolve()
_CWD: Final = _THIS.parent

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterator,
        Callable,
        Coroutine,
        Generator,
        Sequence,
    )

    from sagent.host.server import Host, Server
    from sagent.host.wire import Message
    from sagent.tools.types import Tool, ToolContext, ToolStreamingResult
    from trackinizer.client.tool_bridge import Streams, Websocket


pytestmark = pytest.mark.asyncio

_HOST: Final = uuid.UUID(int=1)
_INSTANCE: Final = uuid.UUID(int=2)
_CREDENTIAL: Final = "trax_machine_" + "0" * 32 + "_" + "c" * 43
_REVOKED: Final = ClientError("-> 410", status_code=410)
_NAME: Final = "box-1"
_CLEAN_CODES: Final = frozenset({CloseCode.NORMAL_CLOSURE, CloseCode.GOING_AWAY})
"""The close codes websockets ends a read on quietly; any other raises."""

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
    """End each test with its daemon's threads gone, however it stopped them."""
    yield
    _joined()


@pytest.fixture(autouse=True)
def relays(monkeypatch: pytest.MonkeyPatch) -> _Board:
    """Answer every daemon's dial from a stand-in relay; record its stops and lines."""
    board = _Board()
    monkeypatch.setattr(tool_serving, "connect", board.connect)
    monkeypatch.setattr(tool_serving, "echo", board.echo)
    monkeypatch.setattr(
        tool_serving,
        "threading",
        SimpleNamespace(Event=board.stop, Thread=threading.Thread),
    )
    return board


class _Client:
    """The ``Client`` calls a daemon makes: each wait takes the next answer put here.

    A wait with no answer to take is held until one is put, or for ``hold_sec``;
    after :meth:`release`, every wait is answered with nothing at once, as a held
    request a stop leaves behind ends.

    Args:
      hold_sec: How long a wait is held for; ``None`` holds it until an answer.

    """

    def __init__(self, *, hold_sec: float | None = None) -> None:
        self.waits: list[tuple[uuid.UUID, uuid.UUID, float]] = []
        self.keys: list[str] = []
        self.closed = threading.Event()
        self.changed = threading.Condition()
        """Notified as each wait is asked or answer put; the daemon asks from its
        listener thread."""
        self._answers: list[ToolOffer | Exception | None] = []
        self._released = False
        self._hold_sec = hold_sec

    def __call__(self, key: str) -> _Client:
        self.keys.append(key)
        return self

    def put(self, answer: ToolOffer | Exception | None) -> None:
        """Answer the next wait with ``answer``: an offer, a failure, or nothing."""
        with self.changed:
            self._answers.append(answer)
            self.changed.notify_all()

    def release(self) -> None:
        """Answer every wait, held or to come, with nothing."""
        with self.changed:
            self._released = True
            self.changed.notify_all()

    def wait_tools(
        self,
        host_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        wait_sec: float,
    ) -> ToolOffer | None:
        self.asked(host_id, instance, wait_sec)
        with self.changed:
            _ = self.changed.wait_for(
                lambda: self._answers or self._released,
                timeout=self._hold_sec,
            )
            answer = self._answers.pop(0) if self._answers else None
        if isinstance(answer, Exception):
            raise answer
        return answer

    def close(self) -> None:
        self.closed.set()

    def asked(self, host_id: uuid.UUID, instance: uuid.UUID, wait_sec: float) -> None:
        """Record one wait and wake whoever waits for it."""
        with self.changed:
            self.waits.append((host_id, instance, wait_sec))
            self.changed.notify_all()

    async def reached(self, count: int) -> None:
        """Return once ``count`` waits were asked."""
        assert await asyncio.to_thread(self._reached, count), f"never asked {count}"

    def _reached(self, count: int) -> bool:
        with self.changed:
            return self.changed.wait_for(lambda: len(self.waits) >= count, timeout=10)


class _Waits(threading.Event):
    """A daemon's stop that keeps how long each wait on it was for.

    A wait with a timeout returns at once, as though its time had passed, so a retry
    costs no real time; a wait without one waits for the stop.
    """

    def __init__(self) -> None:
        super().__init__()
        self.waited: list[float | None] = []
        self.changed = threading.Condition()

    @override
    def wait(self, timeout: float | None = None) -> bool:
        with self.changed:
            self.waited.append(timeout)
            self.changed.notify_all()
        if timeout is None:
            return super().wait()
        return self.is_set()

    async def entered(self, seconds: float, count: int) -> None:
        """Return once ``count`` waits of ``seconds`` have begun."""
        assert await asyncio.to_thread(self._entered, seconds, count)

    def _entered(self, seconds: float, count: int) -> bool:
        with self.changed:
            return self.changed.wait_for(
                lambda: self.waited.count(seconds) >= count,
                timeout=10,
            )


class _End:
    """One end of an in-memory websocket, read on the event loop it was made for.

    What the other end sends or closes with reaches this one through its loop, so
    each end may live on a loop of its own: the daemon's, or the test's.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.peer: _End | None = None
        self.closed = False
        """Set once either end closed: before :meth:`close` returns, on its own end."""
        self.close_code: int | None = None
        self.close_reason: str | None = None
        self.held = False
        """Whether a close here waits for :meth:`release`, as one the peer is slow to
        answer does."""
        self.closing = threading.Event()
        """Set once a close here has begun."""
        self.closes = 0
        """How many times this end was closed, closed already or not."""
        self._loop = loop
        self._inbox: asyncio.Queue[bytes | Close] = asyncio.Queue()
        self._ended = asyncio.Event()
        self._released = asyncio.Event()

    async def send(self, message: str | bytes, /) -> None:
        if self.closed:
            raise ConnectionClosedOK(None, None)
        assert self.peer is not None
        self.peer.deliver(message.encode() if isinstance(message, str) else message)

    async def recv(self) -> bytes:
        """Return the next message; raise as websockets does once the socket closed."""
        item = await self._inbox.get()
        if isinstance(item, Close):
            # Every later read finds the socket closed too.
            self._inbox.put_nowait(item)
            if item.code in _CLEAN_CODES:
                raise ConnectionClosedOK(item, None)
            raise ConnectionClosedError(item, None)
        return item

    async def close(
        self,
        code: int = CloseCode.NORMAL_CLOSURE,
        reason: str = "",
    ) -> None:
        self.closes += 1
        if self.closed:
            return
        self.closing.set()
        if self.held:
            _ = await self._released.wait()
        assert self.peer is not None
        frame = Close(code, reason)
        self._close(frame)
        self.peer.deliver(frame)

    async def release(self) -> None:
        """Let each close held here finish; run on this end's loop."""
        self._released.set()

    async def wait_closed(self) -> None:
        _ = await self._ended.wait()

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._messages()

    def deliver(self, item: bytes | Close) -> None:
        """Hand ``item`` to this end from any thread; an end whose loop ended drops it."""
        with contextlib.suppress(RuntimeError):
            _ = self._loop.call_soon_threadsafe(self._take, item)

    async def _messages(self) -> AsyncIterator[bytes]:
        while True:
            try:
                yield await self.recv()
            except ConnectionClosedOK:
                return

    def _take(self, item: bytes | Close) -> None:
        if isinstance(item, Close):
            self._close(item)
        elif not self.closed:
            self._inbox.put_nowait(item)

    def _close(self, frame: Close) -> None:
        """End this end with ``frame``: the code either end closed with first."""
        if self.closed:
            return
        self.closed = True
        self.close_code, self.close_reason = frame.code, frame.reason
        self._inbox.put_nowait(frame)
        self._ended.set()


class _Socket:
    """The server's end of one daemon socket."""

    def __init__(self, websocket: _End) -> None:
        self.websocket = websocket
        self._held = b""

    async def say(self, message: Message) -> None:
        await self.websocket.send(encoded(message))

    async def next(self) -> Message | None:
        """Return the next line, or ``None`` when the daemon closed its socket."""
        try:
            while b"\n" not in self._held:
                self._held += await asyncio.wait_for(self.websocket.recv(), 10)
        except ConnectionClosed:
            return None
        line, _, self._held = self._held.partition(b"\n")
        reader = asyncio.StreamReader()
        reader.feed_data(line + b"\n")
        return await received(reader)

    async def closed(self) -> None:
        await asyncio.wait_for(self.websocket.wait_closed(), 10)

    async def open_session(self, session_id: str = "s1") -> Message | None:
        await self.say(_hello(session_id))
        return await self.next()


class _Relay:
    """The server's side of a daemon: its offers, and an in-memory socket for each dial.

    Args:
      url: The server's URL, which names this relay alone.

    """

    def __init__(self, url: str) -> None:
        self.url = url
        self.client = _Client()
        self.sockets: asyncio.Queue[_Socket] = asyncio.Queue()
        self.dials: list[tuple[str, str]] = []
        self.ends: list[_End] = []
        """The daemon's end of each socket dialed, oldest first."""
        self.refusing = False
        self.silent = False
        self.dialing = threading.Event()
        """Set once the daemon has begun to dial."""
        self.offered: list[uuid.UUID] = []
        self.daemon_loop: asyncio.AbstractEventLoop | None = None
        """The daemon's event loop, once it has dialed."""
        self._loop = asyncio.get_running_loop()

    async def dial(self, path: str, *, authorization: str) -> _End:
        """Answer a dial on the daemon's loop as the server would; return its end."""
        self.daemon_loop = asyncio.get_running_loop()
        self.dialing.set()
        if self.silent:
            # Answers no handshake, as a server too busy to.
            _ = await asyncio.Event().wait()
        if self.refusing:
            self.refusing = False
            raise _status(HTTPStatus.FORBIDDEN)
        self.dials.append((path, authorization))
        near, far = _End(self.daemon_loop), _End(self._loop)
        near.peer, far.peer = far, near
        self.ends.append(near)
        _ = self._loop.call_soon_threadsafe(self.sockets.put_nowait, _Socket(far))
        return near

    def offer(self) -> None:
        self.offered.append(uuid.uuid4())
        self.client.put(ToolOffer(connection_id=self.offered[-1]))

    async def socket(self) -> _Socket:
        """Offer a tool connection and return the socket the daemon dials back for it."""
        self.offer()
        return await asyncio.wait_for(self.sockets.get(), 10)

    async def on_daemon[T](self, awaited: Coroutine[None, None, T]) -> T:
        """Return what ``awaited`` returns, run on the daemon's own event loop."""
        assert self.daemon_loop is not None
        return await _on(self.daemon_loop, awaited)


class _Relaying(Protocol):
    """What a daemon is made from: a server's URL and the client it waits on."""

    url: str
    client: _Client


class _Board:
    """Every relay of one test, each daemon's stop, and each line the daemons say."""

    def __init__(self) -> None:
        self.relays: list[_Relay] = []
        self.stops: list[_Waits] = []
        self._said: list[str] = []
        self._changed = threading.Condition()

    async def connect(
        self,
        url: str,
        *,
        additional_headers: Mapping[str, str],
        max_size: int,
        ping_interval: float,
        close_timeout: float,
    ) -> _End:
        """Stand in for websockets' ``connect``: the relay ``url`` names answers."""
        del max_size, ping_interval, close_timeout
        for relay in self.relays:
            base = relay.url.replace("http", "ws", 1)
            if url.startswith(f"{base}/"):
                return await relay.dial(
                    url.removeprefix(base),
                    authorization=additional_headers["Authorization"],
                )
        raise ConnectionRefusedError(url)

    def stop(self) -> _Waits:
        """Make a daemon's stop and keep it."""
        self.stops.append(_Waits())
        return self.stops[-1]

    def echo(self, message: object = "", *, err: bool = False) -> None:
        """Say ``message`` as the daemon does, and keep it."""
        echo(message, err=err)
        with self._changed:
            self._said.append(str(message))
            self._changed.notify_all()

    async def heard(self, text: str, count: int = 1) -> None:
        """Return once ``count`` lines said hold ``text``."""
        assert await asyncio.to_thread(self._heard, text, count), f"never said {text}"

    def _heard(self, text: str, count: int) -> bool:
        with self._changed:
            return self._changed.wait_for(
                lambda: sum(text in line for line in self._said) >= count,
                timeout=10,
            )


@asynccontextmanager
async def _relay(board: _Board) -> AsyncGenerator[_Relay]:
    relay = _Relay(f"http://relay-{len(board.relays)}.test")
    board.relays.append(relay)
    try:
        yield relay
    finally:
        # A daemon stopped by hand leaves its listener in a held wait.
        relay.client.release()


def _daemon(
    relay: _Relaying,
    *,
    dirs: ServeDirs | None = None,
    lock_wait_sec: float = 0.0,
    retry_sec: float = 0.005,
    wait_sec: float = 0.0,
    max_connections: int = 16,
    dial_sec: float | None = None,
    line_bytes: int = 0,
    wake: Wake | None = None,
) -> ToolServing:
    return ToolServing(
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
        max_connections=max_connections,
        dial_sec=dial_sec,
        line_bytes=line_bytes,
        wake=wake,
    )


@asynccontextmanager
async def _serving(
    relay: _Relaying,
    *,
    lock_wait_sec: float = 0.0,
    max_connections: int = 16,
    dial_sec: float | None = None,
    line_bytes: int = 0,
) -> AsyncGenerator[ToolServing]:
    daemon = _daemon(
        relay,
        lock_wait_sec=lock_wait_sec,
        max_connections=max_connections,
        dial_sec=dial_sec,
        line_bytes=line_bytes,
    )
    daemon.start(_HOST, _INSTANCE)
    try:
        yield daemon
    finally:
        await asyncio.to_thread(daemon.stop)
        relay.client.release()


def _dirs(relay: _Relaying) -> ServeDirs:
    return tool_dirs(_NAME, server=relay.url)


def _local(monkeypatch: pytest.MonkeyPatch, **tools: Tool) -> None:
    """Serve ``tools`` from sagent's host in ToolServing's own event loop."""
    monkeypatch.setattr(tool_serving, "scrubbed_host", LocalHost(tools=tools))


async def _on[T](
    loop: asyncio.AbstractEventLoop,
    awaited: Coroutine[None, None, T],
) -> T:
    """Return what ``awaited`` returns, run on ``loop``: another thread's."""
    return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(awaited, loop))


async def _nothing() -> None:
    """Do nothing: run on a loop, it returns after every callback queued before it."""


def _joined() -> None:
    """Join the daemon's threads, failing if any outlives 10s."""
    for thread in threading.enumerate():
        if thread.name.startswith("tool-"):
            thread.join(10)
    assert not _threads(), f"left running: {_threads()}"


async def _result(socket: _Socket, call_id: str) -> Result:
    while True:
        message = await socket.next()
        assert message is not None, (
            f"the tool connection ended before {call_id} answered"
        )
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
    """Return the text of a done Gate or Read call's record."""
    tool_result = result.record.result
    assert tool_result == UncategorizedToolResult() or isinstance(
        tool_result,
        ReadResult,
    ), result.record
    return result.record.text()


def _threads() -> set[str]:
    return {t.name for t in threading.enumerate() if t.name.startswith("tool-")}


async def test_each_offer_dials_back_and_serves_a_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    dialed = Mock(wraps=relays.connect)
    adopted = AsyncMock(wraps=tool_serving._adopted)
    served: list[tuple[str, Grant]] = []
    monkeypatch.setattr(tool_serving, "connect", dialed)
    monkeypatch.setattr(tool_serving, "_adopted", adopted)
    monkeypatch.setattr(sagent_host_server.Server, "adopt", _recorded_adopt(served))
    readme = tmp_path / "readme.txt"
    readme.write_text("from this host\n")
    async with _relay(relays) as relay, _serving(relay):
        for session in ("s1", "s2"):
            socket = await relay.socket()
            welcome = await socket.open_session(session)
            assert isinstance(welcome, Welcome)
            assert (welcome.name, list(welcome.tools)) == (_NAME, ["Read"])
            await socket.say(_call("1", "Read", file_path=str(readme)))
            assert "from this host" in _content(await _result(socket, "1"))
            await socket.websocket.close()

    assert relay.client.keys == [_CREDENTIAL]
    assert relay.client.waits[0] == (_HOST, _INSTANCE, WAIT_MAX_SEC)
    assert [path for path, _ in relay.dials] == [
        HOST_TOOLS_WS_PATH.format(host_id=_HOST, connection_id=connection_id)
        + f"?instance={_INSTANCE}"
        for connection_id in relay.offered
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
    # The relay's grant: every tool the server serves, as the relay decides.
    assert served == [
        (f"relay:{offered}", Grant(id="relay", sha256="", tools=None, added=""))
        for offered in relay.offered
    ]


class _Adopt(Protocol):
    """``Server.adopt``, as a function of the server it adopts for."""

    async def __call__(
        self,
        server: Server,
        streams: Streams,
        /,
        *,
        peer: str,
        grant: Grant,
    ) -> None: ...


def _recorded_adopt(served: list[tuple[str, Grant]]) -> _Adopt:
    """Return sagent's ``Server.adopt``, keeping each peer and grant in ``served``."""
    adopt = sagent_host_server.Server.adopt

    async def recorded(
        server: Server,
        streams: Streams,
        *,
        peer: str,
        grant: Grant,
    ) -> None:
        served.append((peer, grant))
        await adopt(server, streams, peer=peer, grant=grant)

    return recorded


async def test_a_daemon_with_its_defaults_dials_sagents_wire_as_the_server_names_it(
    monkeypatch: pytest.MonkeyPatch,
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    websocket = SimpleNamespace(close_code=1000, close_reason="")
    dialed = AsyncMock(return_value=websocket)
    served = threading.Event()

    def serve(*args: object, **kwargs: object) -> None:
        del args, kwargs
        served.set()

    adopted = AsyncMock(side_effect=serve)
    stops = relays.stops
    monkeypatch.setattr(tool_serving, "connect", dialed)
    monkeypatch.setattr(tool_serving, "_adopted", adopted)
    client = _Client()
    # ``http`` twice, a trailing slash, and a last letter ``rstrip`` might take too.
    url = "http://http.X/"
    daemon = ToolServing(
        url,
        credential=_CREDENTIAL,
        name=_NAME,
        client_for=client,
        dirs=tool_dirs(_NAME, server=url),
    )
    connection = uuid.UUID(int=3)
    client.put(ToolOffer(connection_id=connection))
    client.put(None)
    daemon.start(_HOST, _INSTANCE)
    # The next wait, answered at once, waits the default before it asks again.
    await stops[0].entered(5.0, 1)
    # The offer's connection runs on the daemon's loop, apart from the listener.
    assert await asyncio.to_thread(served.wait, 10)
    await asyncio.to_thread(daemon.stop)
    client.release()

    assert client.waits[0][2] == WAIT_MAX_SEC
    dialed.assert_awaited_once_with(
        "ws://http.X"
        + HOST_TOOLS_WS_PATH.format(host_id=_HOST, connection_id=connection)
        + f"?instance={_INSTANCE}",
        additional_headers={"Authorization": f"Bearer {_CREDENTIAL}"},
        max_size=FRAME_BYTES,
        ping_interval=20,
        close_timeout=1.0,
    )
    adopted.assert_awaited_once_with(
        ANY,
        websocket,
        peer=f"relay:{connection}",
        end_sec=5.0,
        line_bytes=MAX_LINE_BYTES,
    )


async def test_stopping_a_daemon_never_started_returns_at_once() -> None:
    daemon = ToolServing(
        "http://h",
        credential=_CREDENTIAL,
        name=_NAME,
        client_for=_Client(),
        dirs=tool_dirs(_NAME, server="http://h"),
    )

    daemon.stop()

    assert not _threads()


async def test_no_thread_of_the_tools_keeps_the_process_alive(
    monkeypatch: pytest.MonkeyPatch,
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay(relays) as relay, _serving(relay):
        # The listener asks from its thread, which the daemon's thread started.
        await relay.client.reached(1)
        assert {"tool-daemon", "tool-listener"} <= _threads()
        threads = [t for t in threading.enumerate() if t.name.startswith("tool-")]
        assert all(thread.daemon for thread in threads)


async def test_a_410_stops_the_daemon(
    monkeypatch: pytest.MonkeyPatch,
    relays: _Board,
) -> None:
    gate = Gate()
    _local(monkeypatch, Bash=gate)
    async with _relay(relays) as relay:
        daemon = _daemon(relay)
        daemon.start(_HOST, _INSTANCE)
        socket = await relay.socket()
        assert isinstance(await socket.open_session(), Welcome)
        await socket.say(_call("1", "Bash"))
        _ = await relay.on_daemon(gate.started.wait())

        relay.client.put(_REVOKED)

        await socket.closed()
        await asyncio.to_thread(_joined)
        assert gate.stopped.is_set()
        assert relay.client.closed.is_set()
        # Nothing was left to stop, and the state's lock is free for a next daemon.
        await asyncio.to_thread(daemon.stop)
        _assert_lock_free(_dirs(relay))


async def test_a_wait_that_fails_is_retried_and_says_so(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay(relays) as relay, _serving(relay):
        relay.client.put(ClientError("-> 503", status_code=503))
        relay.client.put(ValueError("no json\nsecond line"))
        socket = await relay.socket()
        assert isinstance(await socket.open_session(), Welcome)
    # Each failure waits its retry before the next ask.
    assert relays.stops[0].waited.count(0.005) == 2
    said = capsys.readouterr().err.splitlines()
    assert sum("tool wait failed" in line for line in said) == 2
    assert not any(line.startswith("second line") for line in said)


async def test_stop_ends_running_tools(
    monkeypatch: pytest.MonkeyPatch,
    relays: _Board,
) -> None:
    gate = Gate()
    _local(monkeypatch, Bash=gate)
    async with _relay(relays) as relay:
        daemon = _daemon(relay)
        daemon.start(_HOST, _INSTANCE)
        running = await relay.socket()
        assert isinstance(await running.open_session("s1"), Welcome)
        await running.say(_call("1", "Bash"))
        _ = await relay.on_daemon(gate.started.wait())
        idle = await relay.socket()
        assert isinstance(await idle.open_session("s2"), Welcome)
        # A socket that never said hello is a bridge too.
        silent = await relay.socket()

        started = time.monotonic()
        await asyncio.to_thread(daemon.stop)

        assert time.monotonic() - started < HEARTBEAT_SEC
        assert gate.stopped.is_set()
        for socket in (running, idle, silent):
            await socket.closed()
        # The wait the server held at the stop ends, with nothing.
        relay.client.release()
        await asyncio.to_thread(_joined)
        assert relay.client.closed.is_set()
        _assert_lock_free(_dirs(relay))


async def test_stop_returns_once_each_bridge_has_closed(
    monkeypatch: pytest.MonkeyPatch,
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay(relays) as relay:
        daemon = _daemon(relay)
        daemon.start(_HOST, _INSTANCE)
        sockets = [await relay.socket() for _ in range(2)]
        await sockets[0].say(_hello("s1", version=VERSION + 1))
        assert isinstance(await sockets[0].next(), Refused)
        assert isinstance(await sockets[1].open_session("s2"), Welcome)

        # Called on this loop, which runs nothing of the relay's until it returns.
        daemon.stop()

        assert [end.closed for end in relay.ends] == [True, True]
        for socket in sockets:
            await socket.closed()


async def test_stop_waits_for_a_close_the_relay_is_slow_to_answer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay(relays) as relay:
        daemon = _daemon(relay)
        daemon.start(_HOST, _INSTANCE)
        socket = await relay.socket()
        assert isinstance(await socket.open_session(), Welcome)
        [end] = relay.ends
        end.held = True

        stopping = asyncio.ensure_future(asyncio.to_thread(daemon.stop))
        assert await asyncio.to_thread(end.closing.wait, 10)
        assert not stopping.done()
        await relay.on_daemon(end.release())
        await stopping

        assert end.closed
        # The bridge closed it, and the daemon closed it again once the bridge ended.
        assert end.closes == 2
        await socket.closed()
    assert "tools: stopped" not in capsys.readouterr().err


async def test_an_offer_handed_over_as_the_daemon_stops_dials_nothing(
    monkeypatch: pytest.MonkeyPatch,
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())

    class _LateOffer(_Client):
        """Its wait, still held at the stop, answers with an offer only after it."""

        @override
        def wait_tools(
            self,
            host_id: uuid.UUID,
            *,
            instance: uuid.UUID,
            wait_sec: float,
        ) -> ToolOffer | None:
            self.asked(host_id, instance, wait_sec)
            _ = daemon_stopped.wait(10)
            return ToolOffer(connection_id=uuid.uuid4())

    daemon_stopped = threading.Event()
    async with _relay(relays) as relay:
        relay.client = _LateOffer()
        daemon = _daemon(relay)
        daemon.start(_HOST, _INSTANCE)
        await relay.client.reached(1)
        await asyncio.to_thread(daemon.stop)
        daemon_stopped.set()
        await asyncio.to_thread(_joined)
        assert relay.dials == []


@pytest.mark.parametrize("raised", [AttributeError, RecursionError, TypeError])
async def test_what_a_connection_raises_ends_only_that_connection(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    raised: type[Exception],
    relays: _Board,
) -> None:
    gate = Gate()
    local = LocalHost(tools={"Bash": gate})
    loops: list[asyncio.AbstractEventLoop] = []

    async def start(tools: Sequence[str], directory: Path) -> Host:
        loops.append(asyncio.get_running_loop())
        if directory.name == "bad":
            raise raised("hostile\n" + "x" * 10_000)
        return await local(tools, directory)

    monkeypatch.setattr(tool_serving, "scrubbed_host", start)
    async with _relay(relays) as relay, _serving(relay):
        other = await relay.socket()
        assert isinstance(await other.open_session("s1"), Welcome)
        await other.say(_call("1", "Bash"))
        _ = await _on(loops[0], gate.started.wait())

        bad = await relay.socket()
        bad_id = relay.offered[-1]
        await bad.say(_hello("bad"))
        await bad.closed()

        # The other tool connection's call goes on, and the daemon still takes offers.
        loops[0].call_soon_threadsafe(gate.opened.set)
        assert "open for" in _content(await _result(other, "1"))
        after = await relay.socket()
        assert isinstance(await after.open_session("s2"), Welcome)

    said = [
        line
        for line in capsys.readouterr().err.splitlines()
        if "tool connection failed" in line
    ]
    # The type alone: what a hostile line raises may hold its text.
    assert said == [
        f"knowop: tool connection failed: connection={bad_id} error={raised.__name__}",
    ]


async def test_a_connection_says_once_when_it_is_adopted_and_once_how_it_ended(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay(relays) as relay, _serving(relay):
        socket = await relay.socket()
        connection_id = relay.offered[-1]
        assert isinstance(await socket.open_session(), Welcome)
        await socket.websocket.close(wire_terminals.CLOSE_ENDED, "the agent left")
        await relays.heard("tool connection ended")
    said = capsys.readouterr().err.splitlines()

    assert [line for line in said if "tool connection" in line] == [
        f"knowop: tool connection adopted: connection={connection_id}",
        (
            f"knowop: tool connection ended: connection={connection_id} code={wire_terminals.CLOSE_ENDED} "
            "reason='the agent left'"
        ),
    ]


async def test_a_line_too_deep_to_decode_ends_only_its_connection(
    monkeypatch: pytest.MonkeyPatch,
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay(relays) as relay, _serving(relay):
        bad = await relay.socket()
        for start in range(0, 200_000, FRAME_BYTES):
            await bad.websocket.send(b"[" * min(FRAME_BYTES, 200_000 - start))
        await bad.websocket.send(b"\n")
        await bad.closed()
        good = await relay.socket()
        assert isinstance(await good.open_session(), Welcome)


async def test_a_refusal_reaches_the_server_before_the_socket_closes(
    monkeypatch: pytest.MonkeyPatch,
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay(relays) as relay, _serving(relay):
        socket = await relay.socket()
        await socket.say(_hello("s1", version=VERSION + 1))
        assert isinstance(await socket.next(), Refused)
        await socket.closed()


async def test_a_refused_dial_is_one_line_and_the_daemon_goes_on(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay(relays) as relay, _serving(relay):
        relay.refusing = True
        relay.offer()
        await relays.heard("tool connection")
        assert "tool connection failed" in capsys.readouterr().err
        socket = await relay.socket()
        assert isinstance(await socket.open_session(), Welcome)


async def test_a_held_lock_says_so_once_and_serves_when_it_frees(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    stops = relays.stops
    # A clock that stands still: the wait for the lock ends at its deadline, not after.
    monkeypatch.setattr(tool_serving, "time", SimpleNamespace(monotonic=lambda: 5.0))
    async with _relay(relays) as relay:
        holder = _HeldLock(_dirs(relay))
        async with _serving(relay):
            await relays.heard("tools: off")
            [off] = capsys.readouterr().err.splitlines()
            assert off.startswith("knowop: tools: off: ValueError(")
            # It goes on trying, saying nothing more, and the rest of the daemon goes on.
            tried = stops[0].waited.count(0.005)
            await stops[0].entered(0.005, tried + 3)
            assert capsys.readouterr().err == ""
            assert relay.client.waits == []

            holder.close()

            socket = await relay.socket()
            assert isinstance(await socket.open_session(), Welcome)
            assert (
                "knowop: tools: on: the lock came free"
                in capsys.readouterr().err.splitlines()
            )


async def test_a_lock_freed_within_the_wait_serves_after_all(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    stops = relays.stops
    async with _relay(relays) as relay:
        holder = _HeldLock(_dirs(relay))
        async with _serving(relay, lock_wait_sec=30):
            # Retried twice on the held lock, and asked for nothing meanwhile.
            await stops[0].entered(0.005, 2)
            assert relay.client.waits == []
            holder.close()
            socket = await relay.socket()
            assert isinstance(await socket.open_session(), Welcome)
    said = capsys.readouterr().err
    assert "tools: off" not in said
    assert "tools: on" not in said


async def test_stop_ends_a_wait_for_the_lock(
    monkeypatch: pytest.MonkeyPatch,
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    stops = relays.stops
    async with _relay(relays) as relay:
        with _HeldLock(_dirs(relay)):
            daemon = _daemon(relay, lock_wait_sec=600)
            daemon.start(_HOST, _INSTANCE)
            await stops[0].entered(0.005, 1)
            started = time.monotonic()
            await asyncio.to_thread(daemon.stop)
            assert time.monotonic() - started < 5
            await asyncio.to_thread(_joined)


async def test_a_state_folder_it_cannot_make_is_one_line_and_no_tools(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    file = tmp_path / "a-file"
    file.touch()
    dirs = ServeDirs(secrets=file / "m" / "secrets", state=file / "m" / "state")
    async with _relay(relays) as relay:
        daemon = _daemon(relay, dirs=dirs)
        daemon.start(_HOST, _INSTANCE)
        await relays.heard("tools: off")
        await asyncio.to_thread(daemon.stop)
        assert relay.client.waits == []
    [off] = capsys.readouterr().err.splitlines()
    assert off.startswith("knowop: tools: off: NotADirectoryError(")


async def test_it_opens_no_port(
    monkeypatch: pytest.MonkeyPatch,
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    tcp = AsyncMock(side_effect=AssertionError("a port was opened"))
    unix = AsyncMock(side_effect=AssertionError("a socket was opened"))
    monkeypatch.setattr(asyncio, "start_server", tcp)
    monkeypatch.setattr(asyncio, "start_unix_server", unix)
    async with _relay(relays) as relay, _serving(relay):
        socket = await relay.socket()
        assert isinstance(await socket.open_session(), Welcome)
        config = cast(
            dict[str, object],
            json.loads((_dirs(relay).state / "server.json").read_text()),
        )
    assert config["listen"] == []
    tcp.assert_not_awaited()
    unix.assert_not_awaited()


async def test_it_serves_sagents_defaults_and_the_web_and_paper_tools(
    monkeypatch: pytest.MonkeyPatch,
    relays: _Board,
) -> None:
    """A remote agent reaches the web and papers through a host alone."""
    _local(monkeypatch, Read=Read())
    made = Mock(wraps=sagent_host_server.Server)
    monkeypatch.setattr(sagent_host_server, "Server", made)
    async with _relay(relays) as relay, _serving(relay):
        # The listener starts once the tool server is open.
        await relay.client.reached(1)

    assert made.call_args.kwargs["tools"] == (
        *DEFAULT_TOOLS,
        "WebSearch",
        "WebFetch",
        "PaperSearch",
        "PaperDetails",
        "PaperAuthor",
        "PaperFetch",
    )


async def test_its_folders_are_this_users_alone(
    monkeypatch: pytest.MonkeyPatch,
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay(relays) as relay:
        dirs = _dirs(relay)
        tools = state_dir() / "rekursiv-ai" / "knowop" / "tools"
        root = tools / server_dir(relay.url) / _NAME
        assert dirs == ServeDirs(secrets=root / "secrets", state=root / "state")
        # An earlier run under another umask made it wider; this one narrows it.
        root.mkdir(parents=True, mode=0o755)
        root.chmod(0o755)
        async with _serving(relay):
            socket = await relay.socket()
            assert isinstance(await socket.open_session(), Welcome)
            assert stat.S_IMODE(root.stat().st_mode) == 0o700
            assert stat.S_IMODE(dirs.state.stat().st_mode) == 0o700
            assert stat.S_IMODE((dirs.state / "server.pid").stat().st_mode) == 0o600
            sessions = dirs.state / "sessions"
            assert stat.S_IMODE((sessions / "s1").stat().st_mode) == 0o700


async def test_new_tools_folders_are_this_users_alone(
    monkeypatch: pytest.MonkeyPatch,
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay(relays) as relay, _serving(relay):
        socket = await relay.socket()
        assert isinstance(await socket.open_session(), Welcome)
        root = _dirs(relay).state.parent
    assert {
        stat.S_IMODE(folder.stat().st_mode)
        for folder in (root, root.parent, root.parent.parent)
    } == {0o700}


async def test_one_host_name_on_two_servers_serves_both(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay(relays) as first, _relay(relays) as second:
        assert _dirs(first) != _dirs(second)
        async with _serving(first), _serving(second):
            for relay in (first, second):
                socket = await relay.socket()
                assert isinstance(await socket.open_session(), Welcome)
    assert "tools: off" not in capsys.readouterr().err


async def test_the_daemon_program_runs_under_env_without_the_trax_context(
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
    started = AsyncMock(return_value="the daemon")
    monkeypatch.setattr("sagent.host.server.child_host", started)

    assert await scrubbed_host(("Read",), tmp_path) == "the daemon"

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
    assert tool_serving.can_serve()
    assert capsys.readouterr().err == ""

    monkeypatch.setattr(tool_serving, seam, lazy_import(module))

    assert not tool_serving.can_serve()
    said = capsys.readouterr().err.splitlines()
    assert len(said) == 1
    assert "tools: off" in said[0]


async def test_a_sagent_without_server_adopt_serves_no_tools(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _OlderServer:
        """sagent 0.2.3's ``Server``: a daemon module, and no ``adopt``."""

    monkeypatch.setattr(
        tool_serving,
        "sagent_server",
        SimpleNamespace(HOST_COMMAND=HOST_COMMAND, Server=_OlderServer),
    )

    assert not tool_serving.can_serve()
    (said,) = capsys.readouterr().err.splitlines()
    assert "tools: off" in said
    assert "adopt" in said


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(FileNotFoundError(2, "No such file", "/nowhere"), id="missing"),
        pytest.param(ValueError("the host said no hello"), id="exits-at-once"),
    ],
)
async def test_a_session_host_that_will_not_start_says_so(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    relays: _Board,
    failure: Exception,
) -> None:
    async def start(tools: Sequence[str], directory: Path) -> Host:
        del tools, directory
        raise failure

    monkeypatch.setattr(tool_serving, "scrubbed_host", start)
    # With no handler configured, as for ``knowop connect``, a warning reaches stderr.
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    async with _relay(relays) as relay, _serving(relay):
        socket = await relay.socket()
        await socket.say(_hello("s1"))
        await socket.closed()
    said = capsys.readouterr().err.splitlines()
    assert sum("did not start" in line for line in said) == 1


async def test_connections_stop_at_the_cap_and_a_slot_frees_when_one_ends(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    caplog.set_level(logging.WARNING, logger=tool_serving.__name__)
    async with _relay(relays) as relay, _serving(relay, max_connections=2):
        sockets = [await relay.socket() for _ in range(2)]

        relay.offer()

        refused = (
            f"tool connection refused: connection={relay.offered[-1]} open=2 max=2"
        )
        # The listener hands the offer to the daemon's loop before it asks again, and
        # the loop runs what was handed to it in order.
        await relay.client.reached(4)
        await relay.on_daemon(_nothing())
        assert [r.getMessage() for r in caplog.records] == [refused]
        assert len(relay.dials) == 2

        await sockets[0].websocket.close()
        await relays.heard("tool connection ended")
        # The ended connection's task gives back its slot once it is done.
        await relay.on_daemon(_nothing())

        socket = await relay.socket()
        assert isinstance(await socket.open_session(), Welcome)
        assert len(relay.dials) == 3


async def test_a_dial_back_the_server_does_not_answer_ends_at_its_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    # A timeout of none at all: the dial times out as soon as it waits.
    async with _relay(relays) as relay, _serving(relay, dial_sec=0):
        relay.silent = True
        relay.offer()

        await relays.heard("TimeoutError")
    assert (
        f"knowop: tool connection failed: connection={relay.offered[-1]} "
        "error=TimeoutError"
    ) in capsys.readouterr().err.splitlines()
    assert relay.dials == []


class _AnsweredAtOnce(_Client):
    """A server that does not hold its first three waits; it holds the rest."""

    @override
    def wait_tools(
        self,
        host_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        wait_sec: float,
    ) -> ToolOffer | None:
        if len(self.waits) >= 3:
            return super().wait_tools(host_id, instance=instance, wait_sec=wait_sec)
        self.asked(host_id, instance, wait_sec)
        return None


async def test_a_wait_answered_at_once_is_not_asked_again_at_once(
    monkeypatch: pytest.MonkeyPatch,
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    stops = relays.stops
    async with _relay(relays) as relay:
        relay.client = _AnsweredAtOnce()
        daemon = _daemon(relay, retry_sec=0.01)
        daemon.start(_HOST, _INSTANCE)
        await relay.client.reached(3)
        await asyncio.to_thread(daemon.stop)
        # Each ask answered at once is followed by a wait of retry_sec before the next.
        asked = len(relay.client.waits)
        assert stops[0].waited.count(0.01) >= asked - 1


async def test_a_wait_held_to_its_end_is_asked_again_at_once(
    monkeypatch: pytest.MonkeyPatch,
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())

    class _Held(_Client):
        @override
        def wait_tools(
            self,
            host_id: uuid.UUID,
            *,
            instance: uuid.UUID,
            wait_sec: float,
        ) -> ToolOffer | None:
            if len(self.waits) >= 3:
                return super().wait_tools(host_id, instance=instance, wait_sec=wait_sec)
            self.asked(host_id, instance, wait_sec)
            # A long-poll held for all of its wait, on the clock the daemon reads.
            clock.now += wait_sec
            return None

    clock = _Clock()
    monkeypatch.setattr(tool_serving, "time", SimpleNamespace(monotonic=clock))
    async with _relay(relays) as relay:
        relay.client = _Held()
        async with asyncio.timeout(10):
            daemon = _daemon(relay, retry_sec=60, wait_sec=0.01)
            daemon.start(_HOST, _INSTANCE)
            await relay.client.reached(3)
            await asyncio.to_thread(daemon.stop)
        assert {wait_sec for _, _, wait_sec in relay.client.waits} == {0.01}


class _HeldFor(_Client):
    """A server that holds each of its first waits for a time it is given, then all.

    Args:
      clock: The daemon's clock, which each wait moves on by its time.
      held_sec: How long each of the first waits is held, in order.

    """

    def __init__(self, clock: _Clock, *held_sec: float) -> None:
        super().__init__()
        self._clock = clock
        self._held_sec = list(held_sec)

    @override
    def wait_tools(
        self,
        host_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        wait_sec: float,
    ) -> ToolOffer | None:
        if not self._held_sec:
            return super().wait_tools(host_id, instance=instance, wait_sec=wait_sec)
        self.asked(host_id, instance, wait_sec)
        self._clock.now += self._held_sec.pop(0)
        return None


async def test_a_wait_held_under_half_its_time_waits_before_it_asks_again(
    monkeypatch: pytest.MonkeyPatch,
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    clock = _Clock()
    monkeypatch.setattr(tool_serving, "time", SimpleNamespace(monotonic=clock))
    async with _relay(relays) as relay:
        # Held for under half of its 2 s, then for half exactly.
        relay.client = _HeldFor(clock, 0.75, 1.0)
        daemon = _daemon(relay, retry_sec=60, wait_sec=2.0)
        daemon.start(_HOST, _INSTANCE)
        await relay.client.reached(3)
        await asyncio.to_thread(daemon.stop)
    assert relays.stops[0].waited.count(60) == 1


async def test_a_closed_connection_whose_relay_never_answers_closes_after_its_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed_writer = Mock()
    writer = Mock(close=closed_writer)

    async def bridged(websocket: Websocket, *, limit: int) -> tuple[object, Mock]:
        del websocket, limit
        return asyncio.StreamReader(), writer

    monkeypatch.setattr(tool_bridge, "bridged", bridged)
    adopt = AsyncMock()
    closed = AsyncMock()
    # A relay that answers no close: its socket never says it closed.
    socket = Mock(wait_closed=AsyncMock(side_effect=asyncio.Event().wait), close=closed)

    async with asyncio.timeout(1):
        await tool_serving._adopted(
            Mock(adopt=adopt),
            socket,
            peer="p",
            end_sec=0,
            line_bytes=8,
        )

    adopt.assert_awaited_once_with(
        (ANY, writer),
        peer="p",
        grant=Grant(id="relay", sha256="", tools=None, added=""),
    )
    closed_writer.assert_called_once_with()
    closed.assert_awaited_once_with()


async def test_stop_cancels_a_dial_the_server_has_not_answered(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay(relays) as relay:
        relay.silent = True
        daemon = ToolServing(
            relay.url,
            credential=_CREDENTIAL,
            name=_NAME,
            client_for=relay.client,
            dirs=_dirs(relay),
            stop_sec=1.0,
        )
        daemon.start(_HOST, _INSTANCE)
        relay.offer()
        assert await asyncio.to_thread(relay.dialing.wait, 10)

        await asyncio.to_thread(daemon.stop)

        assert "still stopping" not in capsys.readouterr().err
        relay.client.release()
        await asyncio.to_thread(_joined)


async def test_a_wake_from_a_pause_asks_again_at_once_beside_the_dead_wait(
    monkeypatch: pytest.MonkeyPatch,
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    dead = threading.Event()
    cut = threading.Event()
    anew = threading.Event()
    listeners: list[threading.Thread] = []

    # Each listener is known by the thread its waits come from, not by a count of
    # threads or waits, which other tests' threads and the scheduler would sway.
    class _DeadFirst(_Client):
        @override
        def wait_tools(
            self,
            host_id: uuid.UUID,
            *,
            instance: uuid.UUID,
            wait_sec: float,
        ) -> ToolOffer | None:
            self.asked(host_id, instance, wait_sec)
            here = threading.current_thread()
            if not listeners:
                listeners.append(here)
                cut.set()
                # A wait a pause cut: its socket answers nothing until released.
                assert dead.wait(10)
                return None
            if here is not listeners[0]:
                listeners.append(here)
                anew.set()
                # Held like any wait the server holds, until the test ends.
                return super().wait_tools(host_id, instance=instance, wait_sec=wait_sec)
            return None

    wall = [0.0]
    wake = Wake(wall=lambda: wall[0], clock=lambda: 0.0)
    async with _relay(relays) as relay:
        relay.client = _DeadFirst()
        daemon = _daemon(relay, wake=wake)
        try:
            async with asyncio.timeout(10):
                daemon.start(_HOST, _INSTANCE)
                assert await asyncio.to_thread(cut.wait, 10)
                wall[0] += 30.0
                # Called on this thread, which is no daemon: the new listener is one anyway.
                assert wake.check()
                assert await asyncio.to_thread(anew.wait, 10)
                old, new = listeners[0], listeners[1]
                assert (new.name, new.daemon, old.is_alive()) == (
                    "tool-listener",
                    True,
                    True,
                )
                # The listener the wake replaced ends once its cut wait returns.
                dead.set()
                await asyncio.to_thread(old.join, 10)
                assert (old.is_alive(), new.is_alive()) == (False, True)
        finally:
            dead.set()
            await asyncio.to_thread(daemon.stop)
        wall[0] += 30.0
        assert wake.check()


@pytest.mark.parametrize("text", ["-> 403 stale", ""])
async def test_a_wait_failing_the_same_way_is_said_once(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    relays: _Board,
    text: str,
) -> None:
    """A failure is said once while it lasts, and again after a wait that worked."""
    _local(monkeypatch, Read=Read())
    # Even a failure with no text is said, once.
    refused = ClientError(text, status_code=403)
    async with _relay(relays) as relay, _serving(relay):
        for _ in range(3):
            relay.client.put(refused)
        await relay.client.reached(3)
        socket = await relay.socket()
        assert isinstance(await socket.open_session(), Welcome)
        # A failure after a wait that worked is news again.
        relay.client.put(refused)
        await relays.heard("tool wait failed", 2)
    said = capsys.readouterr().err.splitlines()
    assert sum("tool wait failed" in line for line in said) == 2
    # Each names the failure by its type and text.
    assert said.count(f"knowop: tool wait failed: ClientError({text!r})") == 2


class _NoClient(_Client):
    """A credential no client can be made for."""

    @override
    def __call__(self, key: str) -> _Client:
        raise ClientError(f"no client for {key}")


async def test_a_client_that_cannot_be_made_is_one_line_and_ends_the_tools(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay(relays) as relay:
        relay.client = _NoClient()
        daemon = _daemon(relay)
        daemon.start(_HOST, _INSTANCE)
        await asyncio.to_thread(_joined)
        # The line names what failed.
        assert "knowop: tools: off: ClientError('no client for trax_machine_" in (
            capsys.readouterr().err
        )
        await asyncio.to_thread(daemon.stop)
        _assert_lock_free(_dirs(relay))


class _Failing(ToolServing):
    """A daemon whose serving loop raises, as a bug in it would."""

    @override
    async def _serve(self, host_id: uuid.UUID, instance: uuid.UUID) -> None:
        raise RuntimeError("the serving loop broke")


async def test_stop_says_so_when_the_tools_outlast_its_wait(
    capsys: pytest.CaptureFixture[str],
    relays: _Board,
) -> None:
    # This test's own, so a second run in one process (as mutmut's) finds it unset.
    release = threading.Event()

    class _Unstoppable(ToolServing):
        """A daemon whose serving thread ignores the stop until the test lets it go."""

        @override
        async def _serve(self, host_id: uuid.UUID, instance: uuid.UUID) -> None:
            _ = await asyncio.to_thread(release.wait, 10)

    async with _relay(relays) as relay:
        daemon = _Unstoppable(
            relay.url,
            credential=_CREDENTIAL,
            name=_NAME,
            client_for=relay.client,
            dirs=_dirs(relay),
            # Its wait is out at once: the serving thread is still alive after it.
            stop_sec=0,
        )
        daemon.start(_HOST, _INSTANCE)
        try:
            await asyncio.to_thread(daemon.stop)
            assert capsys.readouterr().err.splitlines() == [
                "knowop: tools: still stopping; leaving them to end",
            ]
        finally:
            release.set()
            await asyncio.to_thread(_joined)


async def test_a_serving_loop_that_raises_says_so_and_ends_the_listener(
    capsys: pytest.CaptureFixture[str],
    relays: _Board,
) -> None:
    async with _relay(relays) as relay:
        daemon = _Failing(
            relay.url,
            credential=_CREDENTIAL,
            name=_NAME,
            client_for=relay.client,
            dirs=_dirs(relay),
        )
        daemon.start(_HOST, _INSTANCE)
        await asyncio.to_thread(_joined)
        assert "tools: stopped: RuntimeError('the serving loop broke')" in (
            capsys.readouterr().err
        )
        await asyncio.to_thread(daemon.stop)


class _OfferedAsStopped(ToolServing):
    """A daemon whose listener hands over an offer in the same breath as the stop."""

    @override
    def _listen(
        self,
        host_id: uuid.UUID,
        instance: uuid.UUID,
        loop: asyncio.AbstractEventLoop,
        start_connection: Callable[[ToolOffer], None],
        current: Callable[[], bool],
    ) -> None:
        del current
        # The loop is held while the offer and the stop are queued, so the offer
        # is always started after the stop: an offer racing a stop is made certain.
        held = threading.Event()
        _ = loop.call_soon_threadsafe(held.wait, 10)
        _ = loop.call_soon_threadsafe(
            start_connection,
            ToolOffer(connection_id=uuid.uuid4()),
        )
        self._stop.set()
        held.set()


async def test_an_offer_started_after_the_stop_opens_no_connection(
    monkeypatch: pytest.MonkeyPatch,
    relays: _Board,
) -> None:
    opened: list[ToolOffer] = []

    def connection(
        self: ToolServing,
        server: object,
        offer: ToolOffer,
        **_: object,
    ) -> Coroutine[None, None, None]:
        del self, server
        opened.append(offer)
        return asyncio.sleep(0)

    monkeypatch.setattr(ToolServing, "_pipe", connection)
    async with _relay(relays) as relay:
        daemon = _OfferedAsStopped(
            relay.url,
            credential=_CREDENTIAL,
            name=_NAME,
            client_for=relay.client,
            dirs=_dirs(relay),
        )
        daemon.start(_HOST, _INSTANCE)
        await asyncio.to_thread(_joined)
        await asyncio.to_thread(daemon.stop)
    assert opened == []


@pytest.mark.parametrize("raised", [OSError, asyncio.CancelledError])
async def test_a_connection_whose_bridge_cannot_start_closes_its_socket_and_the_daemon_goes_on(
    monkeypatch: pytest.MonkeyPatch,
    raised: type[BaseException],
    relays: _Board,
) -> None:
    bridged = tool_bridge.bridged
    failed: list[bool] = []

    async def failing_once(
        websocket: Websocket,
        *,
        limit: int,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        if not failed:
            failed.append(True)
            raise raised
        return await bridged(websocket, limit=limit)

    monkeypatch.setattr(tool_bridge, "bridged", failing_once)
    _local(monkeypatch, Read=Read())
    async with _relay(relays) as relay, _serving(relay):
        await (await relay.socket()).closed()
        socket = await relay.socket()
        assert isinstance(await socket.open_session(), Welcome)


async def test_a_line_over_the_limit_ends_its_connection(
    monkeypatch: pytest.MonkeyPatch,
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    limit = 1 << 20
    async with _relay(relays) as relay, _serving(relay, line_bytes=limit):
        socket = await relay.socket()
        frame = b"x" * FRAME_BYTES
        with contextlib.suppress(ConnectionClosed):
            for _ in range(limit // FRAME_BYTES + 2):
                await socket.websocket.send(frame)
        await socket.closed()


async def test_a_note_to_another_session_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    relays: _Board,
) -> None:
    _local(monkeypatch, Read=Read())
    async with _relay(relays) as relay, _serving(relay):
        sender = await relay.socket()
        assert isinstance(await sender.open_session("s1"), Welcome)
        other = await relay.socket()
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
        # The other session's next line answers what it asks after the refusal: no
        # note came before it.
        await other.say(Describe(id="d1", tool="Read"))
        described = await other.next()
        assert isinstance(described, Described)
        assert described.id == "d1"


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

    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[UncategorizedToolResult, str, str]:
        del context
        return whole_output(self._run(arguments))

    async def _run(
        self,
        arguments: Mapping[str, PlainTree],
    ) -> ToolOutput[UncategorizedToolResult, str, str]:
        self.ran.append(repr(arguments)[:100])
        return ToolOutput(stdout=("ran",), result=UncategorizedToolResult())


@dataclass(frozen=True, slots=True, kw_only=True)
class _Hostile:
    """A line an attacker holding the assistant's key could send a daemon."""

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
"""Tags the wire reads, of a type the field does not declare: it refuses each one."""

_TAG_DOES_NOT_STOP_A_CALL: Final = frozenset({"arguments"})
"""The ``Call`` field of plain data, where a read tag is data and the tool runs."""

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
# plain mapping; as a ``json://`` key it is refused, as no key but text is.
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
    # gives out first ends the tool connection, and where is the platform's stack: either is fine.
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
    """Return lines no client should send a daemon, and notes it should not take."""
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
    """Return a field that takes text holding a wire message, which ended its daemon."""
    return {
        "wire-class-address-book": _in_a_session(
            {**_CALL_LINE, "address_book": {"py/object": "Ping"}},
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


async def _send_framed(socket: _Socket, raw: bytes) -> None:
    """Send ``raw`` in frames a daemon's websocket takes; a tool connection already ended is fine."""
    with contextlib.suppress(ConnectionClosed):
        for start in range(0, len(raw), FRAME_BYTES):
            await socket.websocket.send(raw[start : start + FRAME_BYTES])


# The daemon reads a tool connection's lines in order, and starts each call in order. So a ``Read``
# sent after a line is answered after whatever that line did has begun.
async def _settled(socket: _Socket, readable: Path) -> None:
    """Return once the daemon has dealt with what it was sent, or has ended the tool connection."""
    await _send_framed(
        socket,
        encoded(_call("settle", "Read", file_path=str(readable))),
    )
    while True:
        try:
            message = await socket.next()
        except (
            ValueError
        ):  # A line longer than this test's reader takes, a hostile id's.
            continue
        if message is None or (isinstance(message, Result) and message.id == "settle"):
            return


@pytest.mark.parametrize("name", list(_HOSTILE))
async def test_a_hostile_line_acts_on_nothing_and_stays_on_its_connection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
    relays: _Board,
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

    monkeypatch.setattr(tool_serving, "scrubbed_host", start)
    _Canary.touched.clear()
    async with _relay(relays) as relay, _serving(relay):
        witness = await relay.socket()
        assert isinstance(await witness.open_session("witness"), Welcome)
        await witness.say(_call("w1", "Bash"))
        _ = await _on(loops[0], gate.started.wait())

        socket = await relay.socket()
        await _send_framed(socket, hostile.hello)
        await _send_framed(
            socket,
            hostile.line.replace(_SENTINEL.encode(), str(sentinel).encode()),
        )
        await _settled(socket, readable)
        await socket.websocket.close()

        # Another tool connection takes a new session, and reads.
        fresh = await relay.socket()
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
    # No line ends a session's sagent host: each one stopped, when ToolServing did, unbroken.
    assert [
        repr(daemon.task.exception())
        for daemon in local.started
        if not daemon.task.cancelled() and daemon.task.exception() is not None
    ] == []


async def test_importing_the_tool_serving_does_not_load_sagents_daemon() -> None:
    graph = import_graph(
        [tool_serving.__name__],
        path=[str(_ROOT), *sys.path],
        follow=_first_party,
        counted=Counted.IMPORT_TIME,
    )
    reached = {*graph, *(name for names in graph.values() for name in names)}
    assert ClientError.__module__ in reached
    daemon = ServeDirs.__module__.rpartition(".")[0]
    assert sorted(name for name in reached if name.startswith(daemon)) == []


def _first_party(module: str) -> bool:
    return module.split(".", maxsplit=1)[0] in _FIRST_PARTY


class _Clock:
    """A monotonic clock that moves only when a test moves it."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _HeldLock:
    """Hold ``server.pid``'s lock, as another daemon of this host would."""

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
