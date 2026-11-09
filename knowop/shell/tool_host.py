"""Serve sagent's tools to the agents the server pairs with this machine.

A machine host that serves tools holds a request on the server for the next pipe. For
each one it dials the host leg back and hands the bridged streams to one of sagent's
tool servers (``Server.adopt``), which listens on nothing and does the rest: sessions,
acks, resumes, and a child process per session whose tools run as the user who started
``knowop connect``. A session's child starts with no ``TRACKINIZER_*``, ``TRAX_*`` or
``LC_TRAX_*`` variable, whatever its name, and no run's token. That covers the
environment alone: a tool runs as this user, so it can read this user's files, the saved
machine credential among them.

The server hands this machine its lines, so what one line does stays on its pipe: each
pipe's task catches whatever comes out of ``adopt``, says so in one line, and closes
that pipe alone. A revoke or a stop closes every pipe and ends every running call. What
a call left running in the background (``nohup``, ``setsid``, ``&``) is not tracked, and
goes on.

This module imports sagent's host and the tool wire only when it is used, so ``knowop``'s
other commands do not load them. An install whose sagent or trackinizer predates them,
which knowop's floors allow, makes :func:`can_serve` say so, and the machine serves no
tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

import asyncio
import contextlib
import logging
import os
import threading
import time

from trackinizer.client.errors import ClientError
from trackinizer.lib.posix.child import finish
from trackinizer.lib.userdirs import state_dir
from trackinizer.trax.connection import server_dir
from trackinizer.trax.render import echo
from trackinizer.wire.wire_machine_host import HEARTBEAT_SEC
from websockets.asyncio.client import connect

from knowop.agent.tool_server import BEARER_ENV


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import uuid

    from sagent.host import (
        server as sagent_server,
        strings,
        wire,
    )
    from sagent.host.server import Host, Server
    from sagent.host.strings import ServeDirs
    from trackinizer.client import tool_bridge
    from trackinizer.wire import wire_tools
    from trackinizer.wire.wire_tools import ToolOffer
    from websockets.asyncio.client import ClientConnection
else:
    from wrapt import lazy_import

    # Absent from a released sagent or trackinizer that predates them: ``can_serve`` says.
    sagent_server = lazy_import("sagent.host.server")
    strings = lazy_import("sagent.host.strings")
    wire = lazy_import("sagent.host.wire")
    tool_bridge = lazy_import("trackinizer.client.tool_bridge")
    wire_tools = lazy_import("trackinizer.wire.wire_tools")


__all__ = ["ToolClient", "ToolHost", "can_serve", "scrubbed_host", "tool_dirs"]

_logger = logging.getLogger(__name__)

_LOCK_WAIT_SEC: Final = 2 * HEARTBEAT_SEC
"""How long a new host waits for the state's lock before it says tools are off: a host
it took the machine from holds it until its next heartbeat says so, and then stops."""

_STOP_SEC: Final = 2 * HEARTBEAT_SEC
"""How long ``stop`` waits for every session's child to end."""

_BRIDGE_END_SEC: Final = 5.0
"""How long a closed pipe's bridge may take to send what it holds."""

_CLOSE_SEC: Final = 1.0
"""How long a pipe's websocket waits for the server to answer its close."""

_MAX_PIPES: Final = 16
"""The most pipes this host serves at once: its own defence against a server that offers
more than it should. An offer past it is refused, with a warning here; the server, which
is not told, ends that pipe when its dial-back does not come."""

_SAID_CHARS: Final = 200
"""The most of an error one line says: a hostile line's text may be in it."""

_SCRUBBED_PREFIXES: Final = ("TRACKINIZER_", "TRAX_", "LC_TRAX_")
"""What no tool inherits: any trackinizer or trax variable, named in a list or not."""


def can_serve() -> bool:
    """Say whether this install can serve tools; when not, say so in one line too.

    Returns:
      can: Whether sagent's host, as new as the tool host uses, and the tool wire
        import here.

    """
    try:
        _ = (
            sagent_server.Server.adopt,
            tool_bridge.bridged,
            wire_tools.WAIT_MAX_SEC,
        )
    # An older sagent has the host module but not ``Server.adopt``.
    except (ImportError, AttributeError) as error:
        echo(f"knowop: tools: off, this install lacks them: {error}", err=True)
        return False
    return True


def tool_dirs(machine: str, *, server: str) -> ServeDirs:
    """Return where the tool server of ``machine`` on ``server`` keeps what it must.

    Args:
      machine: The machine's name.
      server: The server's URL: the same name on another server is another machine.

    Returns:
      dirs: Two folders of one that :class:`ToolHost` makes for this user alone.

    """
    root = (
        state_dir() / "rekursiv-ai" / "knowop" / "tools" / server_dir(server) / machine
    )
    return strings.ServeDirs(secrets=root / "secrets", state=root / "state")


async def scrubbed_host(tools: Sequence[str], directory: Path) -> Host:
    """Start a session's child without a trackinizer or trax variable or a run's token.

    Args:
      tools: What it serves.
      directory: Where it keeps its log and the outputs its tools spill.

    Returns:
      host: The child.

    """
    names = sorted(
        {
            *(name for name in os.environ if name.startswith(_SCRUBBED_PREFIXES)),
            BEARER_ENV,
        },
    )
    unset = [word for name in names for word in ("-u", name)]
    return await sagent_server.child_host(
        tools,
        directory,
        command=("/usr/bin/env", *unset, *sagent_server.HOST_COMMAND),
    )


class ToolClient(Protocol):
    """The calls the tool host makes: the ``Client`` method, and closing it."""

    def wait_tools(
        self,
        machine_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        wait_sec: float,
    ) -> ToolOffer | None:
        """Hold for the next pipe."""
        ...

    def close(self) -> None:
        """Release the connection."""
        ...


class ToolHost:
    """Serves each tool pipe the server offers, on one event loop of its own.

    Args:
      url: The server's URL.
      credential: The machine credential, which the client and each dial send.
      name: The machine's name, which an agent reads as the endpoint's.
      client_for: Returns a client that sends the given credential.
      dirs: Where the tool server keeps what it must.
      retry_sec: How long to wait before asking again after a failed wait, or one the
        server did not hold, or for the state's lock.
      lock_wait_sec: How long to wait for the state's lock before saying tools are off.
        They come on if the lock frees later.
      stop_sec: How long ``stop`` waits for every session's child to end.
      bridge_end_sec: How long a closed pipe's bridge may take to send what it holds.
      close_sec: How long a pipe's websocket waits for the server to answer its close.
      wait_sec: How long to ask the server to hold each wait; ``0`` asks for the
        longest it holds, ``WAIT_MAX_SEC``.
      max_pipes: The most pipes it serves at once; an offer past them is refused.
      dial_sec: How long a dial-back may take to open; ``None`` for websockets' own.
      line_bytes: The longest line it reads from an agent; ``0`` for sagent's own
        ``MAX_LINE_BYTES``, as the agent reads with. The relay caps what reaches it.

    """

    def __init__(
        self,
        url: str,
        *,
        credential: str,
        name: str,
        client_for: Callable[[str], ToolClient],
        dirs: ServeDirs,
        retry_sec: float = 5.0,
        lock_wait_sec: float = _LOCK_WAIT_SEC,
        stop_sec: float = _STOP_SEC,
        bridge_end_sec: float = _BRIDGE_END_SEC,
        close_sec: float = _CLOSE_SEC,
        wait_sec: float = 0.0,
        max_pipes: int = _MAX_PIPES,
        dial_sec: float | None = None,
        line_bytes: int = 0,
    ) -> None:
        self._url = url.replace("http", "ws", 1).rstrip("/")
        self._credential = credential
        self._name = name
        self._client_for = client_for
        self._dirs = dirs
        self._retry_sec = retry_sec
        self._lock_wait_sec = lock_wait_sec
        self._stop_sec = stop_sec
        self._bridge_end_sec = bridge_end_sec
        self._close_sec = close_sec
        # The wire is imported only once a host is made: ``can_serve`` guards it.
        self._wait_sec = wait_sec or wire_tools.WAIT_MAX_SEC
        self._max_pipes = max_pipes
        self._dial_sec = dial_sec
        self._line_bytes = line_bytes or wire.MAX_LINE_BYTES
        self._stop = threading.Event()
        self._serving: threading.Thread | None = None

    def start(self, machine_id: uuid.UUID, instance: uuid.UUID) -> None:
        """Start serving on a thread; what stops it from serving does not raise here.

        Args:
          machine_id: This machine's id.
          instance: This host's instance.

        """
        self._serving = threading.Thread(
            name="tool-host",
            target=self._run,
            args=(machine_id, instance),
            daemon=True,
        )
        self._serving.start()

    def stop(self) -> None:
        """Stop listening, close every pipe and end every tool; returns once they have.

        A wait the server is still holding stays on its thread until it answers, up to
        its hold, and whatever it answers is dropped.
        """
        self._stop.set()
        if self._serving is not None:
            self._serving.join(timeout=self._stop_sec)
            if self._serving.is_alive():
                echo("knowop: tools: still stopping; leaving them to end", err=True)

    def _run(self, machine_id: uuid.UUID, instance: uuid.UUID) -> None:
        """Serve until stopped; what ends it early also ends the listener."""
        try:
            asyncio.run(self._serve(machine_id, instance))
        except Exception as error:  # noqa: BLE001 -- Tools ending must not end the connect.
            echo(f"knowop: tools: stopped: {_said(error)}", err=True)
        finally:
            self._stop.set()

    async def _serve(self, machine_id: uuid.UUID, instance: uuid.UUID) -> None:
        """Hold the tool server and start each pipe's task, until stopped."""
        server = await self._opened_server()
        if server is None:
            return
        loop = asyncio.get_running_loop()
        pipes: set[asyncio.Task[None]] = set()

        def start_pipe(offer: ToolOffer) -> None:
            if self._stop.is_set():
                return
            if len(pipes) >= self._max_pipes:
                _logger.warning(
                    "tool pipe refused: pipe=%s open=%d max=%d",
                    offer.pipe_id,
                    len(pipes),
                    self._max_pipes,
                )
                return
            pipe = loop.create_task(
                self._pipe(
                    server,
                    offer,
                    machine_id=machine_id,
                    instance=instance,
                ),
            )
            pipes.add(pipe)
            pipe.add_done_callback(pipes.discard)

        try:
            threading.Thread(
                name="tool-listener",
                target=self._listen,
                args=(machine_id, instance, loop, start_pipe),
                daemon=True,
            ).start()
            # Waited for on a thread: a stop that came before this line needs no loop.
            _ = await asyncio.to_thread(self._stop.wait)
        finally:
            try:
                await server.aclose()
            finally:
                for pipe in tuple(pipes):
                    _ = pipe.cancel()
                _ = await asyncio.gather(*pipes, return_exceptions=True)

    async def _opened_server(self) -> Server | None:
        """Return the open tool server, waiting for its lock; ``None`` if it cannot open."""
        deadline = time.monotonic() + self._lock_wait_sec
        told = False
        while not self._stop.is_set():
            server = sagent_server.Server(
                self._dirs,
                listen=[],
                name=self._name,
                start_host=scrubbed_host,
            )
            try:
                _private(self._dirs.state.parent)
                await server.aopen()
            except OSError as error:
                echo(f"knowop: tools: off: {_said(error)}", err=True)
                return None
            except ValueError as error:
                # Another host holds the lock, and a slow one may yet let it go.
                if not told and time.monotonic() >= deadline:
                    echo(f"knowop: tools: off: {_said(error)}", err=True)
                    told = True
                _ = await asyncio.to_thread(self._stop.wait, self._retry_sec)
            else:
                if told:
                    echo("knowop: tools: on: the lock came free", err=True)
                return server
        return None

    def _listen(
        self,
        machine_id: uuid.UUID,
        instance: uuid.UUID,
        loop: asyncio.AbstractEventLoop,
        start_pipe: Callable[[ToolOffer], None],
    ) -> None:
        """Wait for offers until stopped or revoked, starting a pipe for each."""
        try:
            client = self._client_for(self._credential)
        except Exception as error:  # noqa: BLE001 -- A host left serving nothing would not say so.
            echo(f"knowop: tools: off: {_said(error)}", err=True)
            self._stop.set()
            return
        said = ""
        try:
            while not self._stop.is_set():
                asked = time.monotonic()
                try:
                    offer = client.wait_tools(
                        machine_id,
                        instance=instance,
                        wait_sec=self._wait_sec,
                    )
                except Exception as error:  # noqa: BLE001 -- An answer that cannot be read must not end the wait.
                    if isinstance(error, ClientError) and error.status_code == 410:
                        self._stop.set()
                        return
                    # A refusal that lasts would say so every few seconds.
                    if _said(error) != said:
                        said = _said(error)
                        echo(f"knowop: tool wait failed: {said}", err=True)
                    _ = self._stop.wait(self._retry_sec)
                    continue
                said = ""
                if offer is not None:
                    # A loop closed meanwhile has stopped: the offer has no one to serve.
                    with contextlib.suppress(RuntimeError):
                        _ = loop.call_soon_threadsafe(start_pipe, offer)
                elif time.monotonic() - asked < self._wait_sec / 2:
                    # Not held: asking at once would spin on this server.
                    _ = self._stop.wait(self._retry_sec)
        finally:
            client.close()

    # It says when the pipe is adopted, and how it ended or failed. A failure names the
    # error's type alone: what a hostile line raises may hold that line's text.
    async def _pipe(
        self,
        server: Server,
        offer: ToolOffer,
        *,
        machine_id: uuid.UUID,
        instance: uuid.UUID,
    ) -> None:
        """Dial the host leg of ``offer`` and serve it; what it raises stays here."""
        path = wire_tools.MACHINE_TOOLS_WS_PATH.format(
            machine_id=machine_id,
            pipe_id=offer.pipe_id,
        )
        try:
            # ``None`` leaves the dial to websockets' own open timeout.
            async with asyncio.timeout(self._dial_sec):
                socket = await connect(
                    f"{self._url}{path}?instance={instance}",
                    additional_headers={"Authorization": f"Bearer {self._credential}"},
                    max_size=wire_tools.FRAME_BYTES,
                    ping_interval=20,
                    close_timeout=self._close_sec,
                )
            echo(f"knowop: tool pipe adopted: pipe={offer.pipe_id}", err=True)
            await _adopted(
                server,
                socket,
                peer=f"relay:{offer.pipe_id}",
                end_sec=self._bridge_end_sec,
                line_bytes=self._line_bytes,
            )
        except Exception as error:  # noqa: BLE001 -- A hostile line raises anything; it stays on its pipe.
            echo(
                f"knowop: tool pipe failed: pipe={offer.pipe_id} "
                f"error={type(error).__name__}",
                err=True,
            )
        else:
            echo(
                f"knowop: tool pipe ended: pipe={offer.pipe_id} "
                f"code={socket.close_code} reason={socket.close_reason!r}",
                err=True,
            )


# Closing the writer ends the bridge, which sends what the writer holds and only then
# closes the websocket, so a refusal arrives. The caller outlasts the bridge, whatever
# cancels it: a bridge still closing when the event loop ends would leave its socket open.
async def _adopted(
    server: Server,
    socket: ClientConnection,
    *,
    peer: str,
    end_sec: float,
    line_bytes: int,
) -> None:
    """Serve ``socket``'s pipe under the relay's grant, and close it within ``end_sec``."""
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


async def _bridge_ended(socket: ClientConnection, *, end_sec: float) -> None:
    """Wait for a closed pipe's bridge; a server not reading loses its websocket."""
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
