"""Serve terminals on this host: wait for an open, then bridge a shell to the server.

A daemon (``knowop connect``) holds a request on the server for the next terminal to
open. Each open runs a login shell under a host of its own (``trackinizer.lib.posix.host``): a
process in a session, and where systemd can, a scope of its own, that keeps the shell's
pty and its recent output. The daemon bridges that host to a WebSocket it dials out to
the server, which relays it to the owner's browser. The daemon listens on nothing,
except that a browser may offer it a direct link (``terminal_direct.py``) when a WebRTC
library is installed.

The shell outlives what carries it. A dropped socket, or a daemon that stops or
restarts, only lets go of the host: the daemon dials again with ``resume`` and the
server's offset, and a newer daemon of the host takes up every shell an earlier one
left, from the record in the shell's folder. A shell ends when it exits, when the
server says its terminal ended, when the host is revoked, or when no daemon took it up
for :data:`~trackinizer.wire.wire_terminals.DAEMON_GRACE_SEC`.

Every open and close is written to a log only this user can read, with who the server
says opened it, so a dishonest server can lie about the opener but cannot hide an open.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

import asyncio
import contextlib
import enum
import logging
import math
import os
import pwd
import shutil
import threading
import uuid

from pydantic import BaseModel, ConfigDict
from trackinizer.client.errors import ClientError
from trackinizer.lib.posix import attach
from trackinizer.lib.posix.host import (
    Frame,
    Hosting,
    decode_frame,
    encode_frame,
    launch,
)
from trackinizer.lib.posix.sockets import accepts, socket_address
from trackinizer.lib.userdirs import state_dir
from trackinizer.trax.render import echo
from trackinizer.wire import wire_terminals
from trackinizer.wire.wire_terminals import TerminalWait
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus

from knowop.connect.job_limits import Run, Scoping, run
from knowop.connect.terminal_direct import DirectRouter
from knowop.connect.wake import Listener, Wake


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from knowop.connect.terminal_direct import PeerFactory


__all__ = ["TerminalClient", "TerminalServing", "start_folder"]

_logger = logging.getLogger(__name__)

_RECORD: Final = "terminal.json"
"""In a shell's folder: the host it serves and the open it was started for."""

_KILLING_CODES: Final = frozenset(
    {
        wire_terminals.CLOSE_ENDED,
        wire_terminals.CLOSE_DAEMON_LOST,
        wire_terminals.CLOSE_HOST_REVOKED,
    },
)
"""Close codes that say the server ended the terminal, so its shell goes too."""


class TerminalClient(Protocol):
    """The call the listener makes."""

    def wait_terminals(
        self,
        host_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        wait_sec: float,
    ) -> TerminalWait | None:
        """Hold for the next terminal to open."""
        ...

    def close(self) -> None:
        """Release the connection."""
        ...


class TerminalServing:
    """Opens the terminals the server asks for, each on its own bridge.

    Args:
      name: The host's name, which names the log.
      server_url: The server's URL; the sockets use its host over ``ws``/``wss``.
      credential: The host credential, sent as a bearer on each socket.
      credential_client: Returns a client that sends the given credential.
      trax_env: The trax context every shell starts with, which names no secret.
      shell: The command a terminal runs; the user's login shell when empty.
      direct: Makes the peer that answers a browser's offer of a direct link; ``None``
        when no WebRTC library is installed, which keeps every terminal on the relay.
      wake: Tells when this computer wakes from a pause: the wait held across it then
        waits on a dead socket, so a new one is started at once. ``None`` watches not.
      run: Runs ``systemd-run`` once, to learn whether a shell's host can have a
        scope of its own, which a service manager stopping the daemon leaves be.

    """

    def __init__(
        self,
        *,
        name: str,
        server_url: str,
        credential: str,
        credential_client: Callable[[str], TerminalClient],
        trax_env: dict[str, str],
        shell: Sequence[str] = (),
        direct: PeerFactory | None = None,
        wake: Wake | None = None,
        run: Run = run,
    ) -> None:
        self.direct = direct
        """Makes the peer for a browser's offer; ``None`` keeps every terminal on the relay."""
        self._name = name
        self._url = server_url.replace("http", "ws", 1).rstrip("/")
        self._credential = credential
        self._client_for = credential_client
        self.trax_env = trax_env
        """The trax context every shell starts with."""
        self._shell = list(shell)
        self._scoping = Scoping(run)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._bridges: dict[uuid.UUID, threading.Thread] = {}
        self._loops: dict[
            uuid.UUID,
            tuple[asyncio.AbstractEventLoop, asyncio.Event],
        ] = {}
        self._wake = wake

    def start(self, host_id: uuid.UUID, instance: uuid.UUID) -> None:
        """Take up each shell an earlier daemon of the host left, then listen for opens.

        Args:
          host_id: This host's id.
          instance: This daemon's instance.

        """
        shells = state_dir() / "rekursiv-ai" / "knowop" / "terminals"
        for record in sorted(shells.glob(f"*/{_RECORD}")):
            held = _held(record, host_id=host_id)
            if held is None:
                continue
            if accepts(socket_address(record.parent / "host.sock")):
                self._start_bridge(
                    held,
                    host_id=host_id,
                    instance=instance,
                    resume=True,
                )
            else:
                # Its shell ended while no daemon held it; the server ends the terminal
                # once this daemon waits without taking it up.
                shutil.rmtree(record.parent, ignore_errors=True)
        Listener(
            name="terminal-listener",
            listen=partial(self._listen, host_id, instance),
            stop=self._stop,
        ).start(self._wake)

    def stop(self) -> None:
        """Stop listening and let go of every shell; returns once the bridges have ended.

        Each shell runs on in its host, for the next daemon of this host to take up.
        """
        self._stop.set()
        with self._lock:
            loops = list(self._loops.values())
            bridges = list(self._bridges.values())
        for loop, ended in loops:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(ended.set)
        for bridge in bridges:
            bridge.join(timeout=10)

    def _listen(
        self,
        host_id: uuid.UUID,
        instance: uuid.UUID,
        current: Callable[[], bool],
    ) -> None:
        """Wait for opens until stopped or replaced, starting a bridge for each."""
        client = self._client_for(self._credential)
        try:
            while not self._stop.is_set() and current():
                try:
                    opened = client.wait_terminals(
                        host_id,
                        instance=instance,
                        wait_sec=wire_terminals.WAIT_MAX_SEC,
                    )
                except ClientError as err:
                    if err.status_code == 410:
                        _end_every_shell(host_id)
                        return
                    echo(f"knowop: terminal wait failed: {err}", err=True)
                    self._stop.wait(5.0)
                    continue
                if opened is not None:
                    self._start_bridge(opened, host_id=host_id, instance=instance)
        finally:
            client.close()

    def _start_bridge(
        self,
        opened: TerminalWait,
        *,
        host_id: uuid.UUID,
        instance: uuid.UUID,
        resume: bool = False,
    ) -> None:
        """Start a bridge thread for an open, or for a shell to take up again."""
        if self._stop.is_set():
            return
        with self._lock:
            thread = threading.Thread(
                name=f"terminal-{opened.terminal_id}",
                target=self._run,
                args=(opened, host_id, instance, resume),
                daemon=True,
            )
            self._bridges[opened.terminal_id] = thread
            thread.start()

    def _run(
        self,
        opened: TerminalWait,
        host_id: uuid.UUID,
        instance: uuid.UUID,
        resume: bool,
    ) -> None:
        """Bridge one terminal, then forget its thread."""
        try:
            self.serve(opened, host_id=host_id, instance=instance, resume=resume)
        finally:
            with self._lock:
                _ = self._bridges.pop(opened.terminal_id)

    def serve(
        self,
        opened: TerminalWait,
        *,
        host_id: uuid.UUID,
        instance: uuid.UUID,
        resume: bool,
    ) -> None:
        """Bridge one terminal on the calling thread until it ends or the daemon stops.

        Args:
          opened: The terminal the server asked for.
          host_id: This host's id.
          instance: This daemon's instance.
          resume: Take up the shell an earlier bridge started, rather than start one.

        """
        asyncio.run(
            self._bridge(opened, host_id=host_id, instance=instance, resume=resume),
        )

    async def _bridge(
        self,
        opened: TerminalWait,
        *,
        host_id: uuid.UUID,
        instance: uuid.UUID,
        resume: bool,
    ) -> None:
        """Register cancellation before dialing, then serve one shell at most."""
        ended = asyncio.Event()
        with self._lock:
            self._loops[opened.terminal_id] = (asyncio.get_running_loop(), ended)
        if self._stop.is_set():
            ended.set()
        work = asyncio.create_task(
            self._serve(
                opened,
                host_id=host_id,
                instance=instance,
                ended=ended,
                resume=resume,
            ),
        )
        stopped = asyncio.create_task(ended.wait())
        try:
            await asyncio.wait({work, stopped}, return_when=asyncio.FIRST_COMPLETED)
            if work.done():
                await work
        finally:
            work.cancel()
            stopped.cancel()
            await asyncio.gather(work, stopped, return_exceptions=True)
            with self._lock:
                _ = self._loops.pop(opened.terminal_id)

    # A first claim must come before any shell starts. A ``resume`` claim takes up a
    # shell already running, and the server's acknowledgement then carries the offset
    # its relay reached.
    async def _connect(
        self,
        opened: TerminalWait,
        *,
        host_id: uuid.UUID,
        instance: uuid.UUID,
        resume: bool = False,
        open_timeout_sec: float = 10.0,
        startup_timeout_sec: float = 45.0,
        retry_sec: float = 1.0,
    ) -> _Claim | _Refusal:
        """Retry the claim until the server acknowledges it, refuses it, or time is up."""
        path = wire_terminals.HOST_TERMINAL_WS_PATH.format(
            host_id=host_id,
            terminal_id=opened.terminal_id,
        )
        query = f"instance={instance}&ready=true" + ("&resume=true" if resume else "")
        deadline = asyncio.get_running_loop().time() + startup_timeout_sec
        reason = "connection deadline exceeded"
        code: int | None = None
        while (remaining := deadline - asyncio.get_running_loop().time()) > 0:
            socket: ClientConnection | None = None
            try:
                async with asyncio.timeout(min(open_timeout_sec, remaining)):
                    socket = await connect(
                        f"{self._url}{path}?{query}",
                        additional_headers={
                            "Authorization": f"Bearer {self._credential}",
                        },
                        ping_interval=20,
                        open_timeout=min(open_timeout_sec, remaining),
                        close_timeout=1,
                    )
                    await socket.send(encode_frame(Frame.ACK))
                    offset = _acknowledged(await socket.recv(), resume=resume)
                    if offset is None:
                        await socket.close()
                        reason = "invalid startup acknowledgement"
                        break
                return _Claim(socket=socket, offset=offset, resume=resume)
            except asyncio.CancelledError:
                if socket is not None:
                    await socket.close()
                raise
            except (OSError, InvalidHandshake, ConnectionClosed) as error:
                if socket is not None:
                    await socket.close()
                reason = str(error) or type(error).__name__
                if (
                    isinstance(error, ConnectionClosed)
                    and error.rcvd is not None
                    and error.rcvd.code
                    in {
                        wire_terminals.CLOSE_REFUSED,
                        wire_terminals.CLOSE_HOST_REVOKED,
                        wire_terminals.CLOSE_ENDED,
                    }
                ):
                    code = error.rcvd.code
                    break
                if isinstance(error, InvalidStatus) and (
                    error.response.status_code < 500
                    and error.response.status_code != 429
                ):
                    break
                _logger.debug(
                    "terminal %s: startup retry: %s",
                    opened.terminal_id,
                    reason,
                )
            await asyncio.sleep(
                min(retry_sec, max(0, deadline - asyncio.get_running_loop().time())),
            )
        what = "resume" if resume else "startup"
        echo(
            f"knowop: terminal {what} failed ({opened.terminal_id}): {reason}",
            err=True,
        )
        self._log(f"{what}_failed", opened, status=1)
        return _Refusal(code=code)

    # Bridge it to its end.
    #
    # A launched shell is never launched again: a lost socket dials back with ``resume``
    # for as long as the server has the terminal.
    async def _serve(
        self,
        opened: TerminalWait,
        *,
        host_id: uuid.UUID,
        instance: uuid.UUID,
        ended: asyncio.Event,
        resume: bool,
    ) -> None:
        """Start the shell once its relay is ready, or take it up."""
        folder = (
            state_dir()
            / "rekursiv-ai"
            / "knowop"
            / "terminals"
            / str(opened.terminal_id)
        )
        dial = partial(self._connect, opened, host_id=host_id, instance=instance)
        claim = await (
            dial(resume=True, startup_timeout_sec=math.inf) if resume else dial()
        )
        # Closed however the bridge ends, a stop that cancels it included: the server
        # then knows at once that this daemon let go.
        try:
            if not resume and not (
                isinstance(claim, _Claim)
                and await self._started(opened, claim, folder=folder, host_id=host_id)
            ):
                return
            address = await asyncio.to_thread(socket_address, folder / "host.sock")
            ending, status = _Ending.LOST, None
            while ending is _Ending.LOST:
                if isinstance(claim, _Refusal):
                    killed = claim.code in _KILLING_CODES
                    ending = _Ending.KILLED if killed else _Ending.LET_GO
                    break
                ending, status = await self._relay(opened, claim, address, ended)
                if ending is _Ending.LOST:
                    await claim.socket.close()
                    claim = await dial(resume=True, startup_timeout_sec=math.inf)
        finally:
            if isinstance(claim, _Claim):
                await claim.socket.close()
        if ending is _Ending.KILLED:
            status = await _stopped(address)
        if ending in {_Ending.EXITED, _Ending.KILLED}:
            shutil.rmtree(folder, ignore_errors=True)
            self._log("close", opened, status=status)

    async def _started(
        self,
        opened: TerminalWait,
        claim: _Claim,
        *,
        folder: Path,
        host_id: uuid.UUID,
    ) -> bool:
        """Log the open, have the server flush what was typed, and launch the shell."""
        self._log("open", opened)
        try:
            await claim.socket.send(encode_frame(Frame.ACK))
            return await self._launch(opened, claim, folder=folder, host_id=host_id)
        except ConnectionClosed:
            self._log("close", opened, status=1)
            return False

    async def _launch(
        self,
        opened: TerminalWait,
        claim: _Claim,
        *,
        folder: Path,
        host_id: uuid.UUID,
    ) -> bool:
        """Start the shell under its own host; say why in the terminal when it cannot."""
        cwd = await asyncio.to_thread(start_folder, opened.cwd)
        if cwd is None:
            await _end_with_line(
                claim.socket,
                f"knowop: {opened.cwd} is not a folder on this host",
            )
            self._log("close", opened, status=1)
            return False
        address = await asyncio.to_thread(
            _record,
            folder,
            host_id=host_id,
            opened=opened,
        )
        log = folder / "host.log"
        ready = await asyncio.to_thread(
            launch,
            address,
            self._shell or [pwd.getpwuid(os.getuid()).pw_shell, "-l"],
            log=log,
            winsize=(opened.rows, opened.cols),
            timeout_sec=10.0,
            hosting=Hosting(
                cwd=str(cwd),
                env=self._env(opened),
                clean_env=True,
                terminate_grace_sec=2.0,
                unattended_sec=wire_terminals.DAEMON_GRACE_SEC,
            ),
            wrap=partial(
                self._scoping.argv,
                unit=f"knowop-terminal-{opened.terminal_id}",
            ),
        )
        if ready is True:
            return True
        await _end_with_line(
            claim.socket,
            f"knowop: the shell did not start; see {log}",
        )
        shutil.rmtree(folder, ignore_errors=True)
        self._log("close", opened, status=1)
        return False

    async def _relay(
        self,
        opened: TerminalWait,
        claim: _Claim,
        address: Path,
        ended: asyncio.Event,
    ) -> tuple[_Ending, int | None]:
        """Bridge the shell's host to the server until one of them, or the daemon, ends."""
        socket = claim.socket
        shell = attach.RemoteTerminal(address, detach_key=b"", offset=claim.offset)
        try:
            if claim.resume:
                # The server flushes what was typed while no daemon held the terminal.
                await socket.send(encode_frame(Frame.ACK))
            await shell.start()
            if shell.resume != claim.offset:
                await socket.send(encode_frame(Frame.ACK, shell.resume.to_bytes(8)))
        except ConnectionClosed:
            await shell.close()
            return _lost(socket.close_code), None
        except (OSError, asyncio.IncompleteReadError):
            # The host is gone, and the shell's status with it.
            with contextlib.suppress(ConnectionClosed):
                await socket.send(encode_frame(Frame.EXIT, (1).to_bytes(4)))
            return _Ending.EXITED, 1
        router = DirectRouter(
            relay=socket.send,
            shell=shell,
            factory=self.direct,
            ice_servers=tuple(opened.ice_servers),
            start=shell.resume,
        )
        pump = asyncio.create_task(_pump(shell, router))
        reader = asyncio.create_task(_read(router, socket))
        stopped = asyncio.create_task(ended.wait())
        try:
            _ = await asyncio.wait(
                {pump, reader, stopped},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stopped.done():
                return _Ending.LET_GO, None
            if pump.done() and pump.result():
                status = shell.exit_status
            elif reader.done() and reader.result():
                status = await _stopped(address)
            else:
                return _lost(socket.close_code), None
            # Before the router closes: a direct link is told as well as the relay.
            with contextlib.suppress(ConnectionClosed):
                await router.exit(1 if status is None else status)
            return _Ending.EXITED, status
        finally:
            for task in (pump, reader, stopped):
                _ = task.cancel()
            _ = await asyncio.gather(pump, reader, stopped, return_exceptions=True)
            await router.close()
            await shell.close()

    # A shell starts clean, so this and the essentials are all it has.
    def _env(self, opened: TerminalWait) -> dict[str, str]:
        """Return the environment laid over the child's; no secret in it."""
        env = {
            "TERM": "xterm-256color",
            "COLORTERM": "truecolor",
            **self.trax_env,
        }
        if opened.canvas is not None:
            env["TRAX_CANVAS"] = str(opened.canvas)
            env["LC_TRAX_CANVAS"] = str(opened.canvas)
        if opened.workspace is not None:
            env["TRAX_WORKSPACE"] = str(opened.workspace)
            env["LC_TRAX_WORKSPACE"] = str(opened.workspace)
        return env

    def _log(
        self,
        event: str,
        opened: TerminalWait,
        *,
        status: int | None = None,
    ) -> None:
        """Append one line to the daemon's log, readable by this user only."""
        path = (
            state_dir()
            / "rekursiv-ai"
            / "trax"
            / "hosts"
            / f"{self._name}.terminals.log"
        )
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        line = _LogLine(
            at=datetime.now(UTC),
            event=event,
            terminal_id=opened.terminal_id,
            opener=opened.opener,
            requested_by=opened.requested_by,
            status=status,
        )
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a") as file:
            _ = file.write(line.model_dump_json() + "\n")
        if event == "open":
            echo(f"terminal opened by {opened.opener}")


def start_folder(cwd: str | None) -> Path | None:
    """Return the folder a run starts in on this host, or ``None`` when it is none.

    Args:
      cwd: As the server sends it: absolute or ``~``-relative; ``None`` for home.

    Returns:
      folder: The folder, ``~`` resolved; ``None`` when ``cwd`` names no folder here.

    """
    if cwd is None:
        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    try:
        folder = Path(cwd).expanduser()
    except RuntimeError:
        # ``~name`` names a user this host does not have: no home to resolve against.
        return None
    return folder if folder.is_dir() else None


async def _pump(shell: attach.RemoteTerminal, router: DirectRouter) -> bool:
    """Send the shell's output until it ends; False when the server's socket closed."""
    try:
        async for chunk in shell.output():
            await router.output(chunk)
    except ConnectionClosed:
        return False
    return True


# True on ``STOP``, False once it closed.
#
# A message that is no frame ends the terminal as ``STOP`` does: the server broke the
# protocol, and nothing it sends after can be trusted.
async def _read(router: DirectRouter, socket: ClientConnection) -> bool:
    """Apply the server's frames to the shell."""
    try:
        async for message in socket:
            frame = _decode(message)
            if frame is None or frame[0] is Frame.STOP:
                return True
            await router.from_relay(*frame)
    except ConnectionClosed:
        return False
    return False


def _decode(message: str | bytes) -> tuple[Frame, bytes] | None:
    """Return the frame a message carries, or ``None`` when it is not one."""
    if isinstance(message, str):
        return None
    try:
        return decode_frame(message)
    except ValueError:
        return None


class _Socket(Protocol):
    """The one send a refusal makes on the daemon's socket."""

    async def send(self, message: bytes) -> None:
        """Send one binary message."""
        ...


# The shell never starts, so the browser is told the same way as after a shell that
# exits: output, then ``EXIT``.
async def _end_with_line(socket: _Socket, line: str) -> None:
    """Show ``line`` in the terminal, then end it with a failed status."""
    with contextlib.suppress(ConnectionClosed):
        await socket.send(encode_frame(Frame.OUTPUT, f"{line}\r\n".encode()))
        await socket.send(encode_frame(Frame.EXIT, (1).to_bytes(4)))


@dataclass(frozen=True, slots=True, kw_only=True)
class _Claim:
    """A terminal's socket the server acknowledged, and the offset its relay reached."""

    socket: ClientConnection
    offset: int
    resume: bool
    """Whether it takes up a shell already running."""


@dataclass(frozen=True, slots=True, kw_only=True)
class _Refusal:
    """A claim that will not succeed; ``code`` is the server's close code, if it sent one."""

    code: int | None


class _Ending(enum.Enum):
    """How a bridge to a shell's host ended."""

    EXITED = enum.auto()
    """The shell exited, or was stopped, and the server was told its status."""

    KILLED = enum.auto()
    """The server ended the terminal, so its shell is stopped."""

    LOST = enum.auto()
    """The server's socket dropped; the shell runs on and is claimed again."""

    LET_GO = enum.auto()
    """This daemon stops, or another holds the terminal; the shell runs on."""


async def _stopped(address: Path) -> int | None:
    """Stop the shell its host at ``address`` serves; its status, ``None`` if gone."""
    try:
        return await attach.stop(address)
    except OSError:
        return None


def _lost(code: int | None) -> _Ending:
    """Return what a server socket that closed with ``code`` means for its shell."""
    if code in _KILLING_CODES:
        return _Ending.KILLED
    if code == wire_terminals.CLOSE_TAKEN_OVER:
        return _Ending.LET_GO
    return _Ending.LOST


# ``None`` for no ``ACK``.
#
# A first claim's is empty, at offset 0; a ``resume`` claim's holds the u64 offset the
# server's relay reached.
def _acknowledged(message: str | bytes, *, resume: bool) -> int | None:
    """Return the offset the server's acknowledgement carries."""
    frame = _decode(message)
    if frame is None or frame[0] is not Frame.ACK:
        return None
    payload = frame[1]
    if resume:
        return int.from_bytes(payload) if len(payload) == 8 else None
    return None if payload else 0


class _Record(BaseModel):
    """What a shell's folder says of it, for a later daemon to take it up."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    host_id: uuid.UUID
    opened: TerminalWait


def _record(folder: Path, *, host_id: uuid.UUID, opened: TerminalWait) -> Path:
    """Make a new shell's folder, private, and write its record; return its address."""
    # A long folder's socket alias makes it too, as private.
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    address = socket_address(folder / "host.sock")
    fd = os.open(folder / _RECORD, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as file:
        _ = file.write(_Record(host_id=host_id, opened=opened).model_dump_json())
    return address


def _held(record: Path, *, host_id: uuid.UUID) -> TerminalWait | None:
    """Return the open ``record`` names when it is this host's; ``None`` otherwise."""
    try:
        held = _Record.model_validate_json(record.read_bytes())
    except (OSError, ValueError):
        return None
    return held.opened if held.host_id == host_id else None


def _end_every_shell(host_id: uuid.UUID) -> None:
    """Stop every shell of a revoked host, and forget each."""
    shells = state_dir() / "rekursiv-ai" / "knowop" / "terminals"
    for record in sorted(shells.glob(f"*/{_RECORD}")):
        if _held(record, host_id=host_id) is None:
            continue
        _ = asyncio.run(_stopped(socket_address(record.parent / "host.sock")))
        shutil.rmtree(record.parent, ignore_errors=True)


class _LogLine(BaseModel):
    """One line of the daemon-local terminal log."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    at: datetime
    event: str
    terminal_id: uuid.UUID
    opener: str
    requested_by: str | None = None
    status: int | None = None
