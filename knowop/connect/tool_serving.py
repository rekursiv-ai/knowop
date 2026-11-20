"""Serve sagent tools over the daemon's outbound relay mux.

The mux only carries adopted streams. Sagent owns logical sessions, their workers, and
their retained results across relay connections. The daemon budgets calls and caps
resident workers without evicting any session.

A session's worker runs as the user who started ``knowop connect``, without
``TRACKINIZER_*``, ``TRAX_*``, ``LC_TRAX_*`` or a run's token in its environment.
It can still read that user's files, including the saved host credential. A call's
background processes (``nohup``, ``setsid``, ``&``) are not tracked.

The host and tool wire are loaded only when serving tools. An install whose
sagent or trackinizer predates those APIs serves no tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, override

import asyncio
import contextlib
import logging
import math
import os
import threading
import time

from trackinizer.lib.posix.child import finish
from trackinizer.lib.userdirs import state_dir
from trackinizer.trax.connection import server_dir
from trackinizer.trax.render import echo
from trackinizer.wire.wire_daemon import HEARTBEAT_SEC
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed


if TYPE_CHECKING:
    from collections.abc import Sequence

    import uuid

    from sagent.host import (
        server as sagent_server,
        strings,
        wire,
    )
    from sagent.host.server import Host, Server, _Session
    from sagent.host.strings import ServeDirs
    from trackinizer.client import tool_bridge
    from trackinizer.sagent.session.types import ToolResult
    from trackinizer.wire import wire_tools
    from websockets.asyncio.client import ClientConnection
else:
    from wrapt import lazy_import

    # Absent from a released sagent or trackinizer that predates them: ``can_serve`` says.
    sagent_server = lazy_import("sagent.host.server")
    strings = lazy_import("sagent.host.strings")
    wire = lazy_import("sagent.host.wire")
    ToolResult = lazy_import("trackinizer.sagent.session.types", "ToolResult")
    tool_bridge = lazy_import("trackinizer.client.tool_bridge")
    wire_tools = lazy_import("trackinizer.wire.wire_tools")


__all__ = [
    "ToolServing",
    "can_serve",
    "scrubbed_host",
    "scrubbed_names",
    "tool_dirs",
]

_logger = logging.getLogger(__name__)

_LOCK_WAIT_SEC: Final = 2 * HEARTBEAT_SEC
"""How long a new daemon waits for the state's lock before it says tools are off: a daemon
it took the host from holds it until its next heartbeat says so, and then stops."""

_STOP_SEC: Final = 2 * HEARTBEAT_SEC
"""How long ``stop`` waits for every session's child to end."""

_BRIDGE_END_SEC: Final = 5.0
"""How long a closed tool connection's bridge may take to send what it holds."""

_CLOSE_SEC: Final = 1.0
"""How long a tool connection's websocket waits for the server to answer its close."""

_QUEUED_CALLS: Final = 256
"""Default limit on calls waiting for a worker permit, across all sessions."""

_QUEUED_MIB: Final = 128
"""Default limit in MiB on call lines waiting for a worker permit."""

_QUEUE_SEC: Final = 30.0
"""How long a call may wait for a permit before receiving a failure result."""

_RESIDENT_WORKERS: Final = 128
"""The cap on worker children, including those still starting or stopping."""

_SAID_CHARS: Final = 200
"""The most of an error one line says: a hostile line's text may be in it."""

_SCRUBBED_PREFIXES: Final = ("TRACKINIZER_", "TRAX_", "LC_TRAX_")
"""What no tool inherits: any trackinizer or trax variable, named in a list or not."""

_BEARER_ENV: Final = "KNOWOP_TOOLS_TOKEN"
"""The variable a run handed its CLI its own tool server's bearer token in."""


def can_serve() -> bool:
    """Say whether this install can serve tools; when not, say so in one line too.

    Returns:
      can: Whether sagent's host, as new as ToolServing uses, and the tool wire
        import here.

    """
    try:
        _ = (
            sagent_server.Server.adopt,
            tool_bridge.bridged,
            wire_tools.ToolMux,
            wire_tools.MUX_CALLS,
        )
    # An older sagent has the daemon module but not ``Server.adopt``.
    except (ImportError, AttributeError) as error:
        echo(f"knowop: tools: off, this install lacks them: {error}", err=True)
        return False
    return True


def tool_dirs(host: str, *, server: str) -> ServeDirs:
    """Return where the tool server of ``host`` on ``server`` keeps what it must.

    Args:
      host: The host's name.
      server: The server's URL: the same name on another server is another host.

    Returns:
      dirs: Two folders of one that :class:`ToolServing` makes for this user alone.

    """
    root = state_dir() / "rekursiv-ai" / "knowop" / "tools" / server_dir(server) / host
    return strings.ServeDirs(secrets=root / "secrets", state=root / "state")


def scrubbed_names() -> list[str]:
    """Return the variables no child the server asks this daemon for inherits.

    Returns:
      names: Every ``TRACKINIZER_*``, ``TRAX_*`` and ``LC_TRAX_*`` variable set here,
        and a run's token, set or not; sorted.

    """
    return sorted(
        {
            *(name for name in os.environ if name.startswith(_SCRUBBED_PREFIXES)),
            _BEARER_ENV,
        },
    )


async def scrubbed_host(tools: Sequence[str], directory: Path) -> Host:
    """Start a session's child without a trackinizer or trax variable or a run's token.

    Args:
      tools: What it serves.
      directory: Where it keeps its log and the outputs its tools spill.

    Returns:
      daemon: The child.

    """
    unset = [word for name in scrubbed_names() for word in ("-u", name)]
    return await sagent_server.child_host(
        tools,
        directory,
        command=("/usr/bin/env", *unset, *sagent_server.HOST_COMMAND),
    )


class ToolServing:
    """Serve tool sessions on one outbound mux and one event loop of its own.

    Each daemon reads ``KNOWOP_TOOL_ACTIVE_CALLS`` (32), ``KNOWOP_TOOL_QUEUED_CALLS``
    (256), and ``KNOWOP_TOOL_QUEUED_MIB`` (128) at construction. Invalid values
    are logged and replaced with the default.

    Args:
      url: The server's URL.
      credential: The host credential sent when dialing the mux.
      name: The host's name, which an agent reads as the endpoint's.
      dirs: Where the tool server keeps what it must.
      retry_sec: How long to wait before reconnecting or retrying the state's lock.
      lock_wait_sec: How long to wait for the state's lock before saying tools are off.
        They come on if the lock frees later.
      stop_sec: How long ``stop`` waits for every session's child to end.
      bridge_end_sec: How long a closed tool connection's bridge may take to send what it holds.
      close_sec: How long a tool connection's websocket waits for the server to answer its close.
      dial_sec: How long a mux dial may take to open; ``None`` for no timeout.
      line_bytes: The longest line it reads from an agent; ``0`` for sagent's own
        ``MAX_LINE_BYTES``, as the agent reads with. The relay caps what reaches it.

    """

    def __init__(
        self,
        url: str,
        *,
        credential: str,
        name: str,
        dirs: ServeDirs,
        retry_sec: float = 5.0,
        lock_wait_sec: float = _LOCK_WAIT_SEC,
        stop_sec: float = _STOP_SEC,
        bridge_end_sec: float = _BRIDGE_END_SEC,
        close_sec: float = _CLOSE_SEC,
        dial_sec: float | None = 10.0,
        line_bytes: int = 0,
    ) -> None:
        self._url = url.replace("http", "ws", 1).rstrip("/")
        self._credential = credential
        self._name = name
        self._dirs = dirs
        self._retry_sec = retry_sec
        self._lock_wait_sec = lock_wait_sec
        self._stop_sec = stop_sec
        self._bridge_end_sec = bridge_end_sec
        self._close_sec = close_sec
        # The wire is imported only once a daemon is made: ``can_serve`` guards it.
        self._dial_sec = dial_sec
        self._line_bytes = line_bytes or wire.MAX_LINE_BYTES
        self._stop = threading.Event()
        self._serving: threading.Thread | None = None
        self._permits = _CallPermits()

    def start(self, host_id: uuid.UUID, instance: uuid.UUID) -> None:
        """Start serving on a thread; what stops it from serving does not raise here.

        Args:
          host_id: This host's id.
          instance: This daemon's instance.

        """
        self._serving = threading.Thread(
            name="tool-daemon",
            target=self._run,
            args=(host_id, instance),
            daemon=True,
        )
        self._serving.start()

    def stop(self) -> None:
        """Close the mux and end the session workers, waiting up to the stop timeout."""
        self._stop.set()
        if self._serving is not None:
            self._serving.join(timeout=self._stop_sec)
            if self._serving.is_alive():
                echo("knowop: tools: still stopping; leaving them to end", err=True)

    def _run(self, host_id: uuid.UUID, instance: uuid.UUID) -> None:
        """Serve until stopped."""
        try:
            asyncio.run(self._serve(host_id, instance))
        except Exception as error:  # noqa: BLE001 -- Tools ending must not end the connect.
            echo(f"knowop: tools: stopped: {_said(error)}", err=True)
        finally:
            self._stop.set()

    async def _serve(self, host_id: uuid.UUID, instance: uuid.UUID) -> None:
        """Keep the server and its workers alive across mux reconnects."""
        server = await self._opened_server()
        if server is None:
            return
        mux = asyncio.create_task(self._mux_loop(server, host_id, instance))
        try:
            _ = await asyncio.to_thread(self._stop.wait)
        finally:
            self._permits.close()
            try:
                await server.aclose()
            finally:
                _ = mux.cancel()
                _ = await asyncio.gather(mux, return_exceptions=True)

    async def _mux_loop(
        self,
        server: Server,
        host_id: uuid.UUID,
        instance: uuid.UUID,
    ) -> None:
        path = wire_tools.HOST_TOOLS_MUX_PATH.format(host_id=host_id)
        while not self._stop.is_set():
            try:
                async with connect(
                    f"{self._url}{path}?instance={instance}",
                    additional_headers={"Authorization": f"Bearer {self._credential}"},
                    max_size=wire_tools.MUX_FRAME_BYTES,
                    ping_interval=20,
                    open_timeout=self._dial_sec,
                    close_timeout=self._close_sec,
                ) as socket:
                    await self._mux_socket(server, socket)
            except Exception as error:  # noqa: BLE001 -- Retry a failed daemon connection.
                _logger.warning(
                    "daemon tool mux failed: error=%s",
                    type(error).__name__,
                )
            await asyncio.sleep(self._retry_sec)

    async def _mux_socket(self, server: Server, socket: ClientConnection) -> None:
        tasks: set[asyncio.Task[None]] = set()

        def accept(stream: wire_tools.MuxStream) -> None:
            task = asyncio.create_task(self._mux_stream(server, stream))
            tasks.add(task)
            task.add_done_callback(tasks.discard)

        async def read() -> bytes | None:
            try:
                data = await socket.recv()
            except ConnectionClosed:
                return None
            if not isinstance(data, bytes):
                raise TypeError("daemon tool frames must be binary")
            return data

        async def send(data: bytes) -> None:
            try:
                await socket.send(data)
            except ConnectionClosed as error:
                raise ConnectionError("the relay left") from error

        try:
            await wire_tools.ToolMux(send).run(read, accept=accept)
        finally:
            _ = await asyncio.gather(*tasks, return_exceptions=True)

    async def _mux_stream(self, server: Server, stream: wire_tools.MuxStream) -> None:
        try:
            await _adopted_stream(
                server,
                stream,
                peer=f"relay:{stream.id}",
                end_sec=self._bridge_end_sec,
                line_bytes=self._line_bytes,
            )
        except Exception as error:  # noqa: BLE001 -- One adopted stream must not end its siblings.
            _logger.warning(
                "daemon tool stream failed: stream=%s error=%s",
                stream.id,
                type(error).__name__,
            )

    async def _opened_server(self) -> Server | None:
        """Return the open tool server, waiting for its lock; ``None`` if it cannot open."""
        server_type = _relay_server_type()
        deadline = time.monotonic() + self._lock_wait_sec
        while not self._stop.is_set():
            server = server_type(
                self._dirs,
                listen=[],
                name=self._name,
                # A remote agent is given no tool of its own that reaches past its
                # process, so it searches the web and reads papers here.
                tools=(
                    *sagent_server.DEFAULT_TOOLS,
                    "WebSearch",
                    "WebFetch",
                    "PaperSearch",
                    "PaperDetails",
                    "PaperAuthor",
                    "PaperFetch",
                ),
                start_host=self._permits.start_host,
            )
            try:
                _private(self._dirs.state.parent)
                await server.aopen()
            except OSError as error:
                echo(f"knowop: tools: off: {_said(error)}", err=True)
                return None
            except ValueError as error:
                # Another daemon holds the lock, and a slow one may yet let it go.
                if time.monotonic() >= deadline:
                    echo(f"knowop: tools: off: {_said(error)}", err=True)
                    # Said once: no time reaches the deadline after this.
                    deadline = math.inf
                _ = await asyncio.to_thread(self._stop.wait, self._retry_sec)
            else:
                if deadline == math.inf:
                    echo("knowop: tools: on: the lock came free", err=True)
                return server
        return None


def _relay_server_type() -> type[Server]:
    # Defining the subclass at import time would make sagent a required dependency.
    class _RelayServer(sagent_server.Server):
        @override
        async def _admit(
            self,
            hello: wire.Message | None,
            *,
            peer: str,
            writer: asyncio.StreamWriter,
            grant: strings.Grant | None,
        ) -> _Session | None:
            try:
                return await super()._admit(
                    hello,
                    peer=peer,
                    writer=writer,
                    grant=grant,
                )
            except _WorkersBusyError as error:
                # Native admission would otherwise close without telling the client why.
                writer.write(wire.encoded(wire.Refused(reason=str(error))))
                return None

    return _RelayServer


class _WorkersBusyError(ConnectionError):
    """No worker slot is available for a new logical session."""


def _positive_env(name: str, *, default: int) -> int:
    """Read a positive integer from the environment or keep the default."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value > 0:
        return value
    _logger.warning("%s must be a positive integer; using %d", name, default)
    return default


class _CallPermits:
    """One FIFO and one active-call budget, shared by all workers in this daemon."""

    def __init__(self) -> None:
        self._active_limit = _positive_env(
            "KNOWOP_TOOL_ACTIVE_CALLS",
            default=wire_tools.MUX_CALLS,
        )
        self._queued_limit = _positive_env(
            "KNOWOP_TOOL_QUEUED_CALLS",
            default=_QUEUED_CALLS,
        )
        self._queued_byte_limit = (
            _positive_env("KNOWOP_TOOL_QUEUED_MIB", default=_QUEUED_MIB) << 20
        )
        self._active: set[tuple[_PermittedHost, str]] = set()
        self._queued: dict[tuple[_PermittedHost, str], _QueuedCall] = {}
        self._queued_bytes = 0
        self._closed = False
        self._residents = 0

    async def start_host(self, tools: Sequence[str], directory: Path) -> Host:
        """Reserve a slot and start a scrubbed worker for a new session.

        Args:
          tools: Tools the worker can serve.
          directory: Where the worker keeps its log and spill files.

        Returns:
          host: The worker with its calls subject to the daemon's shared limits.

        Raises:
          ConnectionError: The daemon is busy or stopping.

        """
        if self._closed:
            raise ConnectionError("Tool serving is stopping.")
        if self._residents >= _RESIDENT_WORKERS:
            raise _WorkersBusyError(
                "All tool workers are busy. Retry the connection later.",
            )
        self._residents += 1
        starting = asyncio.create_task(self._start_worker(tools, directory=directory))
        try:
            worker = await asyncio.shield(starting)
        except asyncio.CancelledError:
            await finish(_stop_abandoned_start(starting))
            raise
        if self._closed:
            await _stop_unadmitted(worker)
            raise ConnectionError("Tool serving is stopping.")
        return worker

    def submit(self, worker: _PermittedHost, call_id: str, *, line: bytes) -> None:
        """Write a call to its worker now, or put it in the bounded FIFO.

        Args:
          worker: Where the call would run.
          call_id: The worker-local ID of this call.
          line: The original call line to forward.

        """
        key = (worker, call_id)
        if key in self._active or key in self._queued:
            return
        if self._closed or worker.stopping:
            worker.reject(call_id, outcome="failed", reason="Tool serving is stopping.")
        elif len(self._active) < self._active_limit:
            self._send(key, line=line)
        elif (
            len(self._queued) >= self._queued_limit
            or self._queued_bytes + len(line) > self._queued_byte_limit
        ):
            worker.reject(
                call_id,
                outcome="failed",
                reason="The tool call queue is full.",
            )
        else:
            expiry = asyncio.get_running_loop().call_later(
                _QUEUE_SEC,
                self._expire,
                key,
            )
            self._queued[key] = _QueuedCall(line=line, expiry=expiry)
            self._queued_bytes += len(line)

    def cancel(self, worker: _PermittedHost, call_id: str, *, line: bytes) -> None:
        """Cancel queued work locally; an executing call still owns its permit.

        Args:
          worker: Where the call runs or would run.
          call_id: The worker-local ID of the cancelled call.
          line: The original cancel line to forward if it has started.

        """
        key = (worker, call_id)
        if key in self._queued:
            _ = self._dequeue(key)
            worker.reject(
                call_id,
                outcome="cancelled",
                reason="Cancelled before starting.",
            )
        elif key in self._active:
            worker.host.writer.write(line)

    def completed(self, worker: _PermittedHost, call_id: str) -> None:
        """Free the permit for a Result consumed from a worker, not from a client."""
        self._active.discard((worker, call_id))
        self._advance()

    def stopping(self, worker: _PermittedHost) -> None:
        """Remove queued calls for a worker that no longer takes calls."""
        for key in tuple(self._queued):
            if key[0] is worker:
                _ = self._dequeue(key)

    def exited(self, worker: _PermittedHost) -> None:
        """Release the stopped worker's slot and calls still missing results."""
        self._residents -= 1
        self.stopping(worker)
        self._active = {key for key in self._active if key[0] is not worker}
        self._advance()

    def close(self) -> None:
        """Do not start workers or queued calls during daemon shutdown."""
        self._closed = True
        for key in tuple(self._queued):
            _ = self._dequeue(key)

    async def _start_worker(
        self,
        tools: Sequence[str],
        *,
        directory: Path,
    ) -> _PermittedHost:
        try:
            host = await scrubbed_host(tools, directory)
        except BaseException:
            self._residents -= 1
            raise
        return _PermittedHost(host, permits=self)

    def _send(self, key: tuple[_PermittedHost, str], *, line: bytes) -> None:
        worker, _ = key
        self._active.add(key)
        worker.host.writer.write(line)

    def _advance(self) -> None:
        while self._queued and len(self._active) < self._active_limit:
            key = next(iter(self._queued))
            if self._queued[key].expiry.when() <= asyncio.get_running_loop().time():
                self._expire(key)
            else:
                self._send(key, line=self._dequeue(key))

    def _dequeue(self, key: tuple[_PermittedHost, str]) -> bytes:
        queued = self._queued.pop(key)
        queued.expiry.cancel()
        self._queued_bytes -= len(queued.line)
        return queued.line

    def _expire(self, key: tuple[_PermittedHost, str]) -> None:
        _ = self._dequeue(key)
        worker, call_id = key
        worker.reject(
            call_id,
            outcome="failed",
            reason="The tool call queue wait expired.",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _QueuedCall:
    line: bytes
    expiry: asyncio.TimerHandle


class _PermittedHost:
    """Wrap a worker without adding another socket or changing its wire."""

    def __init__(self, host: Host, *, permits: _CallPermits) -> None:
        self.host = host
        self.permits = permits
        self.stopping = False
        self.reader = _WorkerReader(self)
        self.writer = asyncio.StreamWriter(
            _WorkerTransport(self),
            asyncio.Protocol(),
            None,
            asyncio.get_running_loop(),
        )
        self._stopped: asyncio.Task[None] | None = None

    async def stop(self) -> None:
        """Fence new work, then let the worker stop before releasing its permits."""
        if self._stopped is None:
            self.fence()
            self._stopped = asyncio.create_task(self._stop_worker())
        await finish(self._stopped)

    def fence(self) -> None:
        """Prevent admission to a worker whose stream is closing or unreadable."""
        self.stopping = True
        self.permits.stopping(self)

    def reject(
        self,
        call_id: str,
        *,
        outcome: Literal["failed", "cancelled"],
        reason: str,
    ) -> None:
        """Supply a normal Result for a call the worker did not run.

        Args:
          call_id: The worker-local ID of the rejected call.
          outcome: Whether it failed or was cancelled while queued.
          reason: The reason sent to the client.

        """
        self.reader.feed_data(
            wire.encoded(
                wire.Result(
                    id=call_id,
                    record=ToolResult(call_id="", state=outcome, stderr=(reason,)),
                ),
            ),
        )

    async def _stop_worker(self) -> None:
        await self.host.stop()
        self.permits.exited(self)


async def _stop_abandoned_start(starting: asyncio.Task[_PermittedHost]) -> None:
    [worker] = await asyncio.gather(starting, return_exceptions=True)
    if isinstance(worker, _PermittedHost):
        await _stop_unadmitted(worker)


async def _stop_unadmitted(worker: _PermittedHost) -> None:
    try:
        await worker.stop()
    finally:
        worker.writer.close()


class _WorkerReader(asyncio.StreamReader):
    """Merge child output and daemon results into the logical host's stream."""

    def __init__(self, worker: _PermittedHost) -> None:
        super().__init__(limit=wire.MAX_LINE_BYTES)
        self._worker = worker
        self._source: asyncio.StreamReader | None = worker.host.reader
        self._reading: asyncio.Task[None] | None = None

    @override
    async def readline(self) -> bytes:
        if self._source is not None and (self._reading is None or self._reading.done()):
            self._reading = asyncio.create_task(self._read_worker(self._source))
        return await super().readline()

    async def _read_worker(self, source: asyncio.StreamReader) -> None:
        worker = self._worker
        try:
            line = await source.readline()
            if line:
                message = wire.decoded(line)
                self.feed_data(line)
                if isinstance(message, wire.Result):
                    worker.permits.completed(worker, message.id)
                return
        except (OSError, ValueError) as error:
            self.set_exception(error)
        else:
            self.feed_eof()
        self._source = None
        # A broken output stream does not prove the worker's calls have stopped.
        worker.fence()


class _WorkerTransport(asyncio.WriteTransport):
    """Intercept the complete lines Server writes to a worker."""

    def __init__(self, worker: _PermittedHost) -> None:
        super().__init__()
        self._worker = worker

    @override
    def write(self, data: bytes | bytearray | memoryview) -> None:
        worker = self._worker
        line = bytes(data)
        message = wire.decoded(line)
        if isinstance(message, wire.Call):
            worker.permits.submit(worker, message.id, line=line)
        elif isinstance(message, wire.Cancel):
            worker.permits.cancel(worker, message.id, line=line)
        else:
            worker.host.writer.write(line)

    @override
    def is_closing(self) -> bool:
        worker = self._worker
        return worker.stopping or worker.host.writer.is_closing()

    @override
    def close(self) -> None:
        worker = self._worker
        worker.fence()
        worker.host.writer.close()


# Closing the writer ends the bridge, which sends what the writer holds and only then
# closes the websocket, so a refusal arrives. The caller outlasts the bridge, whatever
# cancels it: a bridge still closing when the event loop ends would leave its socket open.
async def _adopted_stream(
    server: Server,
    socket: wire_tools.MuxStream,
    *,
    peer: str,
    end_sec: float,
    line_bytes: int,
) -> None:
    """Serve ``socket``'s tool connection under the relay's grant, and close it within ``end_sec``."""
    try:
        streams = await tool_bridge.bridged(socket, limit=line_bytes)
    except BaseException:
        await finish(socket.close())
        raise
    try:
        await server.adopt(
            streams,
            peer=peer,
            grant=strings.Grant(id="relay", sha256="", tools=None, added=""),
        )
    finally:
        streams[1].close()
        await finish(_bridge_ended(socket, end_sec=end_sec))


async def _bridge_ended(
    socket: wire_tools.MuxStream,
    *,
    end_sec: float,
) -> None:
    """Wait for a closed tool connection's bridge; a server not reading loses its websocket."""
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(socket.wait_closed(), end_sec)
    await socket.close()


def _private(root: Path) -> None:
    """Make ``root``, and each missing folder above it, this user's alone."""
    # ``mkdir(parents=True, mode=...)`` gives the mode to the last one alone.
    for folder in reversed(
        [each for each in (root, *root.parents) if not each.exists()],
    ):
        folder.mkdir(mode=0o700, exist_ok=True)
    # A folder an earlier run made keeps its mode through ``mkdir``.
    Path.chmod(root, 0o700)


def _said(error: Exception) -> str:
    """Return ``error`` on one short line."""
    return repr(error)[:_SAID_CHARS]
