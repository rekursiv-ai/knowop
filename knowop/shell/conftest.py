"""Fakes the shell tests share: OS process replies, scripted servers and peers.

Each test module keeps its own tests; what more than one reads lives here, so no test
module imports another.
"""

from __future__ import annotations

from asyncio.subprocess import Process
from functools import partial
from typing import TYPE_CHECKING, Final, Protocol, Self, override
from unittest.mock import AsyncMock, Mock

import asyncio
import base64
import queue
import sys
import threading
import time
import uuid

from trackinizer.client.errors import ClientError
from trackinizer.lib.posix.host import Frame, decode_frame, encode_frame
from trackinizer.lib.userdirs import state_dir
from trackinizer.wire import wire_terminals
from trackinizer.wire.wire_exec import ExecRequest, ExecResult, ExecWait
from trackinizer.wire.wire_machine_job import MachineJobSpec
from trackinizer.wire.wire_terminals import TerminalWait
from treekle import json_loads
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.http11 import Response

from knowop.shell import terminal_host
from knowop.shell.exec_command import run_command
from knowop.shell.exec_host import ExecHost
from knowop.shell.job_folder import JobFolder
from knowop.shell.terminal_host import TerminalHost


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from pathlib import Path

    from trackinizer.mock_test_support import AsyncMockProtocol, MockProtocol

    import pytest

    from knowop.shell.files_channel import FilesChannel
    from knowop.shell.terminal_direct import PeerFactory


__all__ = [
    "_CREDENTIAL",
    "_INSTANCE",
    "_MACHINE",
    "_REVOKED",
    "_Client",
    "_FakeSocket",
    "_Messages",
    "_Process",
    "_ScriptedPeer",
    "_Server",
    "_SetOnceHeld",
    "_command",
    "_connect",
    "_connector",
    "_err",
    "_fail_files",
    "_finish_files",
    "_folder",
    "_host",
    "_kill_group",
    "_log_lines",
    "_opened",
    "_out",
    "_process",
    "_python",
    "_request",
    "_run",
    "_signal_process",
    "_startup_log",
    "_status",
    "_terminal_host",
    "_terminal_run",
    "_unscoped",
    "_wait_until",
]


class _Stdin(Protocol):
    write: MockProtocol[object]
    close: MockProtocol[object]
    drain: AsyncMockProtocol[object]


class _Process(Mock):
    pid: int
    returncode: int | None
    stdin: _Stdin
    stdout: asyncio.StreamReader
    stderr: asyncio.StreamReader
    wait: AsyncMockProtocol[object]
    send_signal: MockProtocol[None]


def _process(
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
    code: int | None = 0,
) -> _Process:
    out, err = asyncio.StreamReader(), asyncio.StreamReader()
    out.feed_data(stdout)
    err.feed_data(stderr)
    if code is not None:
        out.feed_eof()
        err.feed_eof()
    process = _Process(
        spec=Process,
        pid=7,
        returncode=code,
        stdin=Mock(spec=asyncio.StreamWriter, drain=AsyncMock()),
        stdout=out,
        stderr=err,
        wait=AsyncMock(side_effect=AssertionError("The pipe-bound wait was used.")),
    )
    process.send_signal.side_effect = partial(_signal_process, process)
    return process


def _signal_process(process: _Process, signum: int) -> None:
    process.returncode = -signum
    process.stdout.feed_eof()
    process.stderr.feed_eof()


def _kill_group(process: _Process, pgid: int, signum: int) -> None:
    if pgid != process.pid:
        raise ValueError("Expected pgid == process.pid.")
    _signal_process(process, signum)


_MACHINE: Final = uuid.UUID(int=1)
_INSTANCE: Final = uuid.UUID(int=2)
_CREDENTIAL: Final = "trax_machine_" + "0" * 32 + "_" + "c" * 43
_REVOKED: Final = ClientError("-> 410", status_code=410)


def _request(
    command: str,
    *,
    cwd: Path | str,
    timeout_sec: float = 10,
    stdin: bytes | None = None,
    env: dict[str, str] | None = None,
) -> ExecRequest:
    # A fraction of a second, which the wire's whole seconds cannot say, keeps the
    # timeout tests from waiting out a real one.
    return ExecRequest.model_construct(
        command=command,
        cwd=str(cwd),
        timeout_sec=timeout_sec,
        stdin_b64=None if stdin is None else base64.b64encode(stdin).decode(),
        env=env,
    )


def _run(request: ExecRequest) -> ExecResult:
    """Run the request; a kill that does not take effect shows as a slow run."""
    started = time.monotonic()
    result = asyncio.run(run_command(request))
    if time.monotonic() - started >= 5:
        raise ValueError("Expected time.monotonic() - started < 5.")
    return result


def _out(result: ExecResult) -> bytes:
    return base64.b64decode(result.stdout_b64)


def _err(result: ExecResult) -> bytes:
    return base64.b64decode(result.stderr_b64)


def _wait_until(done: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5
    while not done():
        if time.monotonic() >= deadline:
            raise TimeoutError("the awaited change never came")
        time.sleep(0.005)


class _Server:
    """A scripted server: hands out commands, then ends the host as a revoke would."""

    def __init__(self) -> None:
        self.waits: queue.Queue[ExecWait | ClientError | None] = queue.Queue()
        self.results: list[tuple[uuid.UUID, ExecResult]] = []
        self.post_errors: list[ClientError] = []
        self.posts = 0
        self.posted = threading.Event()
        self.keys: list[str] = []
        self.closed = 0

    def __call__(self, key: str) -> _Server:
        self.keys.append(key)
        return self

    def wait_exec(
        self,
        machine_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        wait_sec: float,
    ) -> ExecWait | None:
        if (machine_id, instance, wait_sec) != (_MACHINE, _INSTANCE, 25):
            raise ValueError(
                "Expected (machine_id, instance, wait_sec) == (_MACHINE, _INSTANCE, 25).",
            )
        # A wait with nothing to hand out ends empty, as the server's does. Raising
        # instead would fail the listener of a test that stopped early, after the test.
        try:
            answer = self.waits.get(timeout=5)
        except queue.Empty:
            return None
        if isinstance(answer, ClientError):
            raise answer
        return answer

    def post_exec_result(
        self,
        machine_id: uuid.UUID,
        exec_id: uuid.UUID,
        result: ExecResult,
    ) -> None:
        if machine_id != _MACHINE:
            raise ValueError("Expected machine_id == _MACHINE.")
        self.posts += 1
        if self.post_errors:
            raise self.post_errors.pop(0)
        self.results.append((exec_id, result))
        self.posted.set()

    def close(self) -> None:
        self.closed += 1


def _command(
    command: str,
    *,
    cwd: Path,
    timeout_sec: int = 10,
    env: dict[str, str] | None = None,
) -> ExecWait:
    return ExecWait(
        exec_id=uuid.uuid4(),
        asker="ada@example.com",
        command=command,
        cwd=str(cwd),
        timeout_sec=timeout_sec,
        env=env,
    )


def _host(server: _Server) -> ExecHost:
    return ExecHost(
        name="box-1",
        credential=_CREDENTIAL,
        credential_client=server,
        retry_sec=0,
    )


def _log_lines() -> list[dict[str, object]]:
    path = state_dir() / "rekursiv-ai" / "trax" / "machines" / "box-1.exec.log"
    return [
        json_loads(line, dict[str, object]) for line in path.read_text().splitlines()
    ]


class _Client:
    """Hands out scripted waits; once they run out it blocks like a held request."""

    def __init__(self, answers: Sequence[TerminalWait | ClientError]) -> None:
        self._answers = list(answers)
        self.idle = threading.Event()
        self.closed = threading.Event()
        self.failed = threading.Event()
        """Set once a scripted error has been raised."""

    def wait_terminals(
        self,
        machine_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        wait_sec: float,
    ) -> TerminalWait | None:
        if (machine_id, instance, wait_sec) != (_MACHINE, _INSTANCE, 25):
            raise ValueError(
                "Expected (machine_id, instance, wait_sec) == (_MACHINE, _INSTANCE, 25).",
            )
        if self._answers:
            answer = self._answers.pop(0)
            if isinstance(answer, ClientError):
                self.failed.set()
                raise answer
            return answer
        self.idle.wait(0.005)
        return None

    def close(self) -> None:
        self.closed.set()


def _opened(**fields: object) -> TerminalWait:
    return TerminalWait.model_validate(
        {
            "terminal_id": uuid.uuid4(),
            "opener": "ada@x.y",
            "rows": 24,
            "cols": 80,
            **fields,
        },
    )


def _terminal_host(
    url: str,
    *shell: str,
    client: _Client | None = None,
    trax_env: dict[str, str] | None = None,
    drain_sec: float = 0.5,
    direct: PeerFactory | None = None,
) -> TerminalHost:
    return TerminalHost(
        name="box",
        server_url=url,
        credential=_CREDENTIAL,
        credential_client=lambda _key: client or _Client([]),
        trax_env=trax_env or {},
        shell=shell,
        drain_sec=drain_sec,
        direct=direct,
    )


async def _terminal_run(host: TerminalHost, opened: TerminalWait) -> None:
    await asyncio.to_thread(
        host.serve,
        opened,
        machine_id=_MACHINE,
        instance=_INSTANCE,
    )


async def _finish_files(
    channel: FilesChannel,
    send: Callable[[bytes], Awaitable[None]],
) -> None:
    del channel
    await send(b"finished\n")


async def _fail_files(
    channel: FilesChannel,
    send: Callable[[bytes], Awaitable[None]],
) -> None:
    del channel, send
    raise RuntimeError("file service failed")


class _ScriptedPeer:
    """A peer that, once it has answered, plays the browser: switch the route, type."""

    def __init__(self) -> None:
        self.sent: list[tuple[Frame, bytes]] = []
        self.ice_servers: tuple[str, ...] = ()
        self.is_open = True

    async def accept(
        self,
        offer_sdp: str,
        *,
        ice_servers: tuple[str, ...],
        on_message: Callable[[bytes], None],
        on_close: Callable[[], None],
    ) -> str:
        del offer_sdp, on_close
        self.ice_servers = ice_servers
        loop = asyncio.get_running_loop()
        route = wire_terminals.encode_route(wire_terminals.ROUTE_DIRECT, 0)
        loop.call_soon(on_message, encode_frame(Frame.ROUTE, route))
        loop.call_soon(on_message, encode_frame(Frame.INPUT, b"go\n"))
        return "answer-sdp"

    def send(self, data: bytes) -> None:
        self.sent.append(decode_frame(data))

    async def close(self) -> None:
        self.is_open = False


class _FakeSocket:
    """A leg that answers the startup handshake as scripted."""

    def __init__(
        self,
        *,
        reply: bytes = encode_frame(Frame.ACK),
        send_error: ConnectionClosed | None = None,
        hang: bool = False,
        messages: tuple[bytes, ...] = (),
        closes: bool = False,
        end_after_output: bool = False,
    ) -> None:
        self.reply = reply
        self.send_error = send_error
        self.hang = hang
        self.closed = False
        self.waiting = asyncio.Event()
        self.received = threading.Event()
        self.sent: list[bytes] = []
        self.messages = messages
        self.closes = closes
        self.end_after_output = end_after_output
        self.output_sent = asyncio.Event()

    def __aiter__(self) -> _Messages:
        return _Messages(
            *self.messages,
            closes=self.closes,
            hold=not self.end_after_output,
            until=self.output_sent if self.end_after_output else None,
        )

    async def send(self, data: bytes) -> None:
        self.sent.append(data)
        if decode_frame(data)[0] is Frame.OUTPUT:
            self.output_sent.set()
        if self.send_error is not None:
            raise self.send_error

    async def recv(self) -> bytes:
        self.received.set()
        if self.hang:
            self.waiting.set()
            await asyncio.Event().wait()
        return self.reply

    async def close(self) -> None:
        self.closed = True


class _Connector:
    """Stands in for ``connect``: scripted outcomes, the last one repeating."""

    def __init__(self, *outcomes: _FakeSocket | Exception) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []
        self.urls: list[str] = []

    async def __call__(self, url: str, **kwargs: object) -> _FakeSocket:
        self.urls.append(url)
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _connector(
    monkeypatch: pytest.MonkeyPatch,
    *outcomes: _FakeSocket | Exception,
) -> _Connector:
    connector = _Connector(*outcomes)
    monkeypatch.setattr(terminal_host, "connect", connector)
    return connector


async def _connect(
    opened: TerminalWait | None = None,
    **timing: float,
) -> object:
    host = _terminal_host("http://box.invalid")
    return await host._connect(  # noqa: SLF001 -- The startup handshake the tests time.
        opened or _opened(),
        machine_id=_MACHINE,
        instance=_INSTANCE,
        **timing,
    )


def _status(code: int) -> InvalidStatus:
    return InvalidStatus(Response(code, "x", Headers(), b""))


def _startup_log() -> str:
    path = state_dir() / "rekursiv-ai" / "trax" / "machines" / "box.terminals.log"
    return path.read_text()


class _Messages:
    """The server's leg as it is read: scripted messages, then its end."""

    def __init__(
        self,
        *messages: str | bytes,
        closes: bool,
        hold: bool = False,
        until: asyncio.Event | None = None,
    ) -> None:
        self._messages = list(messages)
        self._closes = closes
        self._hold = hold
        self._until = until

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> str | bytes:
        if self._messages:
            return self._messages.pop(0)
        if self._closes:
            raise ConnectionClosed(None, None)
        if self._until is not None:
            await self._until.wait()
        if self._hold:
            await asyncio.Event().wait()
        raise StopAsyncIteration


def _unscoped(
    *,
    unit: str,
    memory_mb: int,
    argv: Sequence[str],
    runtime_sec: float | None = None,
) -> list[str]:
    if not unit.startswith("knowop-prepare-"):
        raise ValueError('Expected unit.startswith("knowop-prepare-").')
    if memory_mb != 256:
        raise ValueError("Expected memory_mb == 256.")
    if runtime_sec is None:
        raise ValueError("Expected runtime_sec is not None.")
    return list(argv)


class _SetOnceHeld(threading.Event):
    """A cancel that is clear when the mirrors' lock is taken, and set from then on."""

    def __init__(self) -> None:
        super().__init__()
        self.asked = 0

    @override
    def is_set(self) -> bool:
        self.asked += 1
        return self.asked > 1


def _folder(tmp_path: Path, *argv: str, minutes: int = 5) -> JobFolder:
    folder = JobFolder.create(
        tmp_path / "7",
        MachineJobSpec(
            id=7,
            owner="jan@example.com",
            argv=list(argv),
            remote="https://github.com/o/r.git",
            ref="refs/heads/main",
            commit="a" * 40,
            gpus=0,
            memory_mb=1_024,
            minutes=minutes,
            label="",
        ),
    )
    folder.src.mkdir()
    return folder


def _python(code: str) -> list[str]:
    return [sys.executable, "-c", code]
