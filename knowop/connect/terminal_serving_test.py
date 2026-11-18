"""TerminalServing with process and WebSocket boundaries supplied inline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, override
from unittest.mock import AsyncMock, Mock

import asyncio
import functools
import logging
import os
import pwd
import stat
import threading
import uuid

from trackinizer.client.errors import ClientError
from trackinizer.lib.posix import attach
from trackinizer.lib.posix.host import (
    Frame,
    Hosting,
    decode_frame,
    encode_frame,
    encode_winsize,
)
from trackinizer.lib.posix.sockets import socket_address
from trackinizer.lib.posix.terminal import Terminal
from trackinizer.lib.userdirs import state_dir
from trackinizer.wire import wire_terminals
from websockets.exceptions import ConnectionClosed
from websockets.frames import Close

import pytest
import pytest_asyncio

from knowop.connect import terminal_serving
from knowop.connect.conftest import (
    _CREDENTIAL,
    _HOST,
    _INSTANCE,
    _Client,
    _connect,
    _connector,
    _FakeSocket,
    _Messages,
    _opened,
    _ScriptedPeer,
    _startup_log,
    _status,
    _terminal_run,
    _terminal_serving,
)
from knowop.connect.terminal_direct import DirectRouter
from knowop.connect.terminal_serving import (
    TerminalServing,
    _acknowledged,
    _Claim,
    _end_with_line,
    _Ending,
    _LogLine,
    _pump,
    _read,
    _Refusal,
)
from knowop.connect.wake import Wake


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from trackinizer.wire.wire_terminals import TerminalWait
    from websockets.asyncio.client import ClientConnection

pytestmark = pytest.mark.asyncio

_INPUT = encode_frame(Frame.INPUT, b"typed\n")


class _HeldShell:
    """A shell's host as a bridge reaches it: scripted output, then its exit or a hold.

    Args:
      chunks: What the shell wrote.
      status: Its exit status, once the chunks are out; ``None`` holds the output open
        until the bridge lets go, as a shell that runs on.
      resume: Where its host resumes the stream.
      after_input: Write nothing until something is typed.
      gone: Raised by ``start``, as a host that is no longer there.

    """

    def __init__(
        self,
        *chunks: bytes,
        status: int | None = 0,
        resume: int = 0,
        after_input: bool = False,
        gone: Exception | None = None,
    ) -> None:
        self.chunks = chunks
        self.status = status
        self.resume = resume
        self.after_input = after_input
        self.gone = gone
        self.exit_status: int | None = None
        self.written: list[bytes] = []
        self.sizes: list[tuple[int, int]] = []
        self.closed = False
        self._typed: asyncio.Event | None = None

    async def start(self) -> None:
        if self.gone is not None:
            raise self.gone
        self._typed = asyncio.Event()

    async def output(self) -> AsyncIterator[bytes]:
        assert self._typed is not None
        if self.after_input:
            _ = await self._typed.wait()
        for chunk in self.chunks:
            yield chunk
        if self.status is None:
            _ = await asyncio.Event().wait()
        self.exit_status = self.status

    async def write(self, data: bytes) -> bool:
        self.written.append(data)
        assert self._typed is not None
        self._typed.set()
        return True

    def set_winsize(self, rows: int, cols: int) -> None:
        self.sizes.append((rows, cols))

    async def close(self) -> None:
        self.closed = True


@dataclass(frozen=True, slots=True, kw_only=True)
class _Launch:
    """One shell's host the daemon started."""

    address: Path
    command: list[str]
    winsize: tuple[int, int]
    hosting: Hosting
    argv: list[str]


@dataclass(slots=True, kw_only=True)
class _Hosts:
    """The shells' hosts: each launch, attach and stop, and the shells they reach."""

    shells: list[_HeldShell]
    ready: bool | None = True
    stop_status: int = 143
    launches: list[_Launch] = field(default_factory=list[_Launch])
    attaches: list[tuple[Path, bytes, int | None]] = field(
        default_factory=list[tuple[Path, bytes, int | None]],
    )
    stops: list[Path] = field(default_factory=list[Path])
    records: list[tuple[int, int, str]] = field(
        default_factory=list[tuple[int, int, str]],
    )
    """The modes of each shell's folder and record, and the record, as it launched."""

    def launch(
        self,
        address: Path,
        command: list[str],
        *,
        log: Path,
        winsize: tuple[int, int],
        timeout_sec: float,
        hosting: Hosting,
        wrap: Callable[[list[str]], list[str]],
    ) -> bool | None:
        del log, timeout_sec
        record = address.parent / "terminal.json"
        self.records.append(
            (
                stat.S_IMODE(address.parent.stat().st_mode),
                stat.S_IMODE(record.stat().st_mode),
                record.read_text(),
            ),
        )
        self.launches.append(
            _Launch(
                address=address,
                command=list(command),
                winsize=winsize,
                hosting=hosting,
                argv=wrap(["host"]),
            ),
        )
        return self.ready

    def attach(self, address: Path, *, detach_key: bytes, offset: int | None) -> object:
        self.attaches.append((address, detach_key, offset))
        return self.shells.pop(0) if len(self.shells) > 1 else self.shells[0]

    async def stop(self, address: Path) -> int | None:
        self.stops.append(address)
        return self.stop_status


def _hosts(
    monkeypatch: pytest.MonkeyPatch,
    *shells: _HeldShell,
    ready: bool | None = True,
) -> _Hosts:
    hosts = _Hosts(shells=list(shells) or [_HeldShell()], ready=ready)
    monkeypatch.setattr(terminal_serving, "launch", hosts.launch)
    monkeypatch.setattr(attach, "RemoteTerminal", hosts.attach)
    monkeypatch.setattr(attach, "stop", hosts.stop)
    return hosts


def _held_record(opened: TerminalWait, *, host_id: uuid.UUID = _HOST) -> Path:
    """Write the record an earlier bridge left for ``opened``; return its folder."""
    folder = (
        state_dir() / "rekursiv-ai" / "knowop" / "terminals" / str(opened.terminal_id)
    )
    folder.mkdir(parents=True)
    _ = (folder / "terminal.json").write_text(
        terminal_serving._Record(host_id=host_id, opened=opened).model_dump_json(),
    )
    return folder


def _result(socket: _FakeSocket) -> tuple[bytes, int]:
    output = b""
    for message in socket.sent:
        kind, payload = decode_frame(message)
        if kind is Frame.OUTPUT:
            output += payload
        elif kind is Frame.EXIT:
            return output, int.from_bytes(payload)
    raise AssertionError("closed before the shell exited")


def _log_lines() -> list[_LogLine]:
    path = state_dir() / "rekursiv-ai" / "trax" / "hosts" / "box.terminals.log"
    return [
        _LogLine.model_validate_json(line) for line in path.read_text().splitlines()
    ]


def _path(opened: TerminalWait) -> str:
    return wire_terminals.HOST_TERMINAL_WS_PATH.format(
        host_id=_HOST,
        terminal_id=opened.terminal_id,
    )


async def test_a_shells_output_and_exit_status_arrive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _FakeSocket()
    connector = _connector(monkeypatch, socket)
    hosts = _hosts(monkeypatch, _HeldShell(b"ok", status=3))
    opened = _opened()

    await _terminal_run(
        _terminal_serving("http://box.invalid", "/bin/sh", "-c", "printf ok; exit 3"),
        opened,
    )

    assert connector.urls == [
        f"ws://box.invalid{_path(opened)}?instance={_INSTANCE}&ready=true",
    ]
    assert connector.calls[0]["additional_headers"] == {
        "Authorization": f"Bearer {_CREDENTIAL}",
    }
    assert _result(socket) == (b"ok", 3)
    # The shell is gone, so nothing is left to take up.
    assert not (
        state_dir() / "rekursiv-ai" / "knowop" / "terminals" / str(opened.terminal_id)
    ).exists()
    (launched,) = hosts.launches
    assert launched.command == ["/bin/sh", "-c", "printf ok; exit 3"]
    assert launched.address == socket_address(
        (state_dir() / "rekursiv-ai" / "knowop" / "terminals" / str(opened.terminal_id))
        / "host.sock",
    )
    # Every byte typed is the shell's, and the stream is numbered from its start.
    assert hosts.attaches == [(launched.address, b"", 0)]
    assert socket.closed


async def test_a_shell_runs_under_a_host_that_outlives_the_daemon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _connector(monkeypatch, _FakeSocket())
    hosts = _hosts(monkeypatch)
    opened = _opened(rows=30, cols=100)

    await _terminal_run(_terminal_serving("http://box.invalid", "/bin/sh"), opened)

    (launched,) = hosts.launches
    assert launched.winsize == (30, 100)
    assert launched.hosting.clean_env
    assert launched.hosting.terminate_grace_sec == 2.0
    # A host no daemon takes up again ends its shell after the server's grace.
    assert launched.hosting.unattended_sec == wire_terminals.DAEMON_GRACE_SEC
    # No systemd here, so no scope: the host runs as itself.
    assert launched.argv == ["host"]


async def test_where_systemd_runs_scopes_a_shells_host_is_put_in_one_named_for_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _connector(monkeypatch, _FakeSocket())
    hosts = _hosts(monkeypatch)
    opened = _opened()
    daemon = TerminalServing(
        name="box",
        server_url="http://box.invalid",
        credential=_CREDENTIAL,
        credential_client=lambda _key: _Client([]),
        trax_env={},
        run=_systemd,
    )

    await _terminal_run(daemon, opened)

    (launched,) = hosts.launches
    assert launched.argv[:3] == ["systemd-run", "--user", "--scope"]
    assert f"knowop-terminal-{opened.terminal_id}" in launched.argv
    assert launched.argv[-2:] == ["--", "host"]


async def test_startup_retries_handshake_timeouts_and_launches_the_shell_once(
    monkeypatch: pytest.MonkeyPatch,
    startup_clock: _StartupClock,
) -> None:
    monkeypatch.setattr(
        TerminalServing,
        "_connect",
        functools.partialmethod(
            TerminalServing._connect,
            open_timeout_sec=0.02,
            retry_sec=0.001,
        ),
    )
    socket = _FakeSocket()
    connector = _connector(monkeypatch, TimeoutError(), TimeoutError(), socket)
    hosts = _hosts(monkeypatch, _HeldShell(b"ready"))
    opened = _opened()

    await asyncio.wait_for(
        _terminal_run(
            _terminal_serving("http://box.invalid", "/bin/sh", "-c", "printf ready"),
            opened,
        ),
        3,
    )

    assert (
        connector.urls
        == [f"ws://box.invalid{_path(opened)}?instance={_INSTANCE}&ready=true"] * 3
    )
    # Each timeout waits its retry before the next try.
    assert len(startup_clock.pauses) == 2
    assert sum(startup_clock.pauses) == pytest.approx(0.002)
    assert all(
        0 < cast(float, kwargs["open_timeout"]) <= 0.02 for kwargs in connector.calls
    )
    assert len(hosts.launches) == 1
    # The handshake's ACK, then the one that has the server flush what was typed.
    assert socket.sent[:2] == [encode_frame(Frame.ACK)] * 2
    assert _result(socket) == (b"ready", 0)
    assert socket.closed


async def test_a_refused_startup_never_launches_the_shell_and_reports_one_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ = _connector(
        monkeypatch,
        ConnectionClosed(Close(wire_terminals.CLOSE_REFUSED, ""), None),
    )
    hosts = _hosts(monkeypatch)

    await _terminal_run(_terminal_serving("http://box.invalid"), _opened())

    assert hosts.launches == []
    error = capsys.readouterr().err
    assert "terminal startup failed" in error
    assert "Traceback" not in error


async def test_stop_cancels_startup_while_waiting_for_the_server_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _FakeSocket(hang=True)
    _ = _connector(monkeypatch, socket)
    hosts = _hosts(monkeypatch)
    daemon = _terminal_serving("http://box.invalid")
    running = asyncio.create_task(_terminal_run(daemon, _opened()))
    try:
        assert await asyncio.wait_for(asyncio.to_thread(socket.received.wait, 2), 2)
        await asyncio.to_thread(daemon.stop)
        await asyncio.wait_for(running, 1)
    finally:
        await asyncio.to_thread(daemon.stop)
        await running

    assert socket.sent == [encode_frame(Frame.ACK)]
    assert socket.closed
    assert hosts.launches == []


async def test_input_reaches_the_shell_and_a_resize_is_clamped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _FakeSocket(
        messages=(
            encode_frame(Frame.RESIZE, encode_winsize(0, 5_000)),
            encode_frame(Frame.INPUT, b"go\n"),
        ),
    )
    _ = _connector(monkeypatch, socket)
    shell = _HeldShell(b"got-go\r\n", after_input=True)
    _ = _hosts(monkeypatch, shell)

    await _terminal_run(_terminal_serving("http://box.invalid", "/bin/sh"), _opened())

    assert _result(socket) == (b"got-go\r\n", 0)
    assert shell.written == [b"go\n"]
    assert shell.sizes == [(1, 1000)]


async def test_the_servers_stop_stops_the_shell_and_reports_its_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _FakeSocket(messages=(encode_frame(Frame.STOP),))
    _ = _connector(monkeypatch, socket)
    shell = _HeldShell(b"17\n", status=None)
    hosts = _hosts(monkeypatch, shell)
    opened = _opened()

    await _terminal_run(_terminal_serving("http://box.invalid", "/bin/sh"), opened)

    assert hosts.stops == [hosts.launches[0].address]
    assert socket.sent[-1] == encode_frame(Frame.EXIT, (143).to_bytes(4))
    assert shell.closed
    assert socket.closed
    assert not (
        state_dir() / "rekursiv-ai" / "knowop" / "terminals" / str(opened.terminal_id)
    ).exists()
    assert [(line.event, line.status) for line in _log_lines()] == [
        ("open", None),
        ("close", 143),
    ]


async def test_a_stop_of_a_host_already_gone_reports_a_failed_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _FakeSocket(messages=(encode_frame(Frame.STOP),))
    _ = _connector(monkeypatch, socket)
    _ = _hosts(monkeypatch, _HeldShell(status=None))
    monkeypatch.setattr(attach, "stop", AsyncMock(side_effect=FileNotFoundError()))

    await _terminal_run(_terminal_serving("http://box.invalid", "/bin/sh"), _opened())

    assert socket.sent[-1] == encode_frame(Frame.EXIT, (1).to_bytes(4))


async def test_a_dropped_socket_lets_go_of_the_shell_and_takes_it_up_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shell runs on; the daemon dials back with ``resume`` from the server's offset."""
    dropped = _FakeSocket(closes=True)
    again = _FakeSocket(reply=encode_frame(Frame.ACK, (2).to_bytes(8)))
    connector = _connector(monkeypatch, dropped, again)
    first, second = _HeldShell(b"ab", status=None), _HeldShell(b"cd", resume=2)
    hosts = _hosts(monkeypatch, first, second)
    opened = _opened()

    await _terminal_run(_terminal_serving("http://box.invalid", "/bin/sh"), opened)

    base = f"ws://box.invalid{_path(opened)}?instance={_INSTANCE}&ready=true"
    assert connector.urls == [base, f"{base}&resume=true"]
    assert len(hosts.launches) == 1
    assert hosts.stops == []
    assert [offset for _, _, offset in hosts.attaches] == [0, 2]
    assert first.closed
    assert dropped.closed
    # The handshake, the flush of what was typed meanwhile, then the shell's output.
    assert again.sent[:2] == [encode_frame(Frame.ACK)] * 2
    assert _result(again) == (b"cd", 0)
    assert [line.event for line in _log_lines()] == ["open", "close"]


async def test_a_host_that_kept_less_than_the_server_has_says_where_it_resumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _FakeSocket(reply=encode_frame(Frame.ACK, (2).to_bytes(8)))
    _ = _connector(monkeypatch, socket)
    hosts = _hosts(monkeypatch, _HeldShell(b"later", resume=9))
    opened = _opened()
    _ = _held_record(opened)

    await _terminal_run(_terminal_serving("http://box.invalid"), opened, resume=True)

    assert hosts.launches == []
    assert socket.sent[:3] == [
        encode_frame(Frame.ACK),
        encode_frame(Frame.ACK),
        encode_frame(Frame.ACK, (9).to_bytes(8)),
    ]
    assert _result(socket) == (b"later", 0)


@pytest.mark.parametrize(
    "code",
    [
        wire_terminals.CLOSE_ENDED,
        wire_terminals.CLOSE_DAEMON_LOST,
        wire_terminals.CLOSE_HOST_REVOKED,
    ],
)
async def test_a_socket_the_server_closed_as_ended_stops_the_shell(
    monkeypatch: pytest.MonkeyPatch,
    code: int,
) -> None:
    socket = _FakeSocket(closes=True, close_code=code)
    connector = _connector(monkeypatch, socket)
    hosts = _hosts(monkeypatch, _HeldShell(status=None))
    opened = _opened()

    await _terminal_run(_terminal_serving("http://box.invalid", "/bin/sh"), opened)

    assert len(connector.urls) == 1
    assert hosts.stops == [hosts.launches[0].address]
    assert not (
        state_dir() / "rekursiv-ai" / "knowop" / "terminals" / str(opened.terminal_id)
    ).exists()
    assert _log_lines()[-1].status == 143


async def test_a_socket_another_daemon_took_over_leaves_the_shell_to_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _FakeSocket(closes=True, close_code=wire_terminals.CLOSE_TAKEN_OVER)
    connector = _connector(monkeypatch, socket)
    hosts = _hosts(monkeypatch, _HeldShell(status=None))
    opened = _opened()

    await _terminal_run(_terminal_serving("http://box.invalid", "/bin/sh"), opened)

    assert len(connector.urls) == 1
    assert hosts.stops == []
    assert (
        (state_dir() / "rekursiv-ai" / "knowop" / "terminals" / str(opened.terminal_id))
        / "terminal.json"
    ).exists()
    assert [line.event for line in _log_lines()] == ["open"]


@pytest.mark.parametrize(
    ("code", "stopped"),
    [
        (wire_terminals.CLOSE_ENDED, True),
        (wire_terminals.CLOSE_HOST_REVOKED, True),
        # A daemon the host replaced is refused too: its shells are the new one's.
        (wire_terminals.CLOSE_REFUSED, False),
    ],
)
async def test_a_refused_resume_stops_the_shell_only_when_its_terminal_ended(
    monkeypatch: pytest.MonkeyPatch,
    code: int,
    stopped: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ = _connector(monkeypatch, ConnectionClosed(Close(code, ""), None))
    hosts = _hosts(monkeypatch)
    opened = _opened()
    folder = _held_record(opened)
    address = socket_address(folder / "host.sock")

    await _terminal_run(_terminal_serving("http://box.invalid"), opened, resume=True)

    assert (hosts.stops == [address]) is stopped
    assert folder.exists() is not stopped
    assert hosts.attaches == []
    assert "knowop: terminal resume failed" in capsys.readouterr().err


async def test_a_host_gone_before_it_is_taken_up_ends_the_terminal_with_status_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _FakeSocket(reply=encode_frame(Frame.ACK, (0).to_bytes(8)))
    _ = _connector(monkeypatch, socket)
    _ = _hosts(monkeypatch, _HeldShell(gone=FileNotFoundError()))
    opened = _opened()
    folder = _held_record(opened)

    await _terminal_run(_terminal_serving("http://box.invalid"), opened, resume=True)

    assert _result(socket) == (b"", 1)
    assert not folder.exists()


async def test_a_socket_that_drops_while_a_shell_is_taken_up_is_dialed_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dropping = _ClosingSocket(reply=encode_frame(Frame.ACK, (0).to_bytes(8)))
    again = _FakeSocket(reply=encode_frame(Frame.ACK, (0).to_bytes(8)))
    connector = _connector(monkeypatch, dropping, again)
    hosts = _hosts(monkeypatch, _HeldShell(b"x"))
    opened = _opened()
    _ = _held_record(opened)

    await _terminal_run(_terminal_serving("http://box.invalid"), opened, resume=True)

    assert len(connector.urls) == 2
    assert len(hosts.attaches) == 2
    assert _result(again) == (b"x", 0)


async def test_a_shell_that_cannot_start_says_where_its_log_is_and_leaves_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _FakeSocket()
    _ = _connector(monkeypatch, socket)
    hosts = _hosts(monkeypatch, ready=False)
    opened = _opened()

    await _terminal_run(_terminal_serving("http://box.invalid", "/bin/sh"), opened)

    log = (
        state_dir() / "rekursiv-ai" / "knowop" / "terminals" / str(opened.terminal_id)
    ) / "host.log"
    assert _result(socket) == (
        f"knowop: the shell did not start; see {log}\r\n".encode(),
        1,
    )
    assert hosts.attaches == []
    assert not (
        state_dir() / "rekursiv-ai" / "knowop" / "terminals" / str(opened.terminal_id)
    ).exists()
    assert [(line.event, line.status) for line in _log_lines()] == [
        ("open", None),
        ("close", 1),
    ]


async def test_a_shells_folder_and_record_are_for_this_user_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _connector(monkeypatch, _FakeSocket())
    hosts = _hosts(monkeypatch)
    opened = _opened()

    await _terminal_run(_terminal_serving("http://box.invalid"), opened)

    ((folder_mode, record_mode, record),) = hosts.records
    assert (folder_mode, record_mode) == (0o700, 0o600)
    assert terminal_serving._Record.model_validate_json(
        record,
    ) == terminal_serving._Record(
        host_id=_HOST,
        opened=opened,
    )


async def test_the_shell_starts_clean_with_the_trax_context_and_no_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sentinel-secret")
    monkeypatch.setenv("TRACKINIZER_TOKEN", "sentinel-token")
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    _ = _connector(monkeypatch, _FakeSocket())
    hosts = _hosts(monkeypatch)
    url = "http://box.invalid"
    canvas, workspace = uuid.uuid4(), uuid.uuid4()
    daemon = _terminal_serving(
        url,
        "/bin/sh",
        trax_env={"LC_TRAX_SERVER": url, "TRACKINIZER_PROFILE": "hosted"},
    )
    await _terminal_run(daemon, _opened(canvas=canvas, workspace=workspace))

    (launched,) = hosts.launches
    # The environment the host's child starts with, as the host builds it.
    child = Terminal(
        ["/bin/sh"],
        env=launched.hosting.env,
        clean_env=launched.hosting.clean_env,
    )._child_env()
    lines = {f"{name}={value}" for name, value in child.items()}
    assert {
        "TERM=xterm-256color",
        "COLORTERM=truecolor",
        "LANG=en_US.UTF-8",
        f"LC_TRAX_SERVER={url}",
        "TRACKINIZER_PROFILE=hosted",
        f"TRAX_CANVAS={canvas}",
        f"LC_TRAX_CANVAS={canvas}",
        f"TRAX_WORKSPACE={workspace}",
        f"LC_TRAX_WORKSPACE={workspace}",
    } <= lines
    assert not [line for line in lines if "sentinel" in line]


async def test_each_open_and_close_is_logged_once_for_this_user_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ = _connector(monkeypatch, _FakeSocket())
    _ = _hosts(monkeypatch, _HeldShell(status=5))
    opened = _opened(requested_by="helper-1")
    await _terminal_run(
        _terminal_serving("http://box.invalid", "/bin/sh", "-c", "exit 5"),
        opened,
    )

    path = state_dir() / "rekursiv-ai" / "trax" / "hosts" / "box.terminals.log"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    first, second = path.read_text().splitlines()
    assert f'"event":"open","terminal_id":"{opened.terminal_id}"' in first
    assert '"opener":"ada@x.y"' in first
    assert '"requested_by":"helper-1"' in first
    assert '"event":"close"' in second
    assert '"status":5' in second
    # Each line says when, in UTC.
    assert _LogLine.model_validate_json(first).at.utcoffset() == timedelta(0)
    # The open, not the close, is said where the user runs the daemon.
    assert capsys.readouterr().out == "terminal opened by ada@x.y\n"


async def test_a_terminal_without_a_folder_starts_in_the_home_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _connector(monkeypatch, _FakeSocket())
    hosts = _hosts(monkeypatch)

    await _terminal_run(_terminal_serving("http://box.invalid"), _opened())

    assert hosts.launches[0].hosting.cwd == pwd.getpwuid(os.getuid()).pw_dir


async def test_a_terminal_starts_in_the_folder_it_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = tmp_path / "project"
    folder.mkdir()
    _ = _connector(monkeypatch, _FakeSocket())
    hosts = _hosts(monkeypatch)

    await _terminal_run(
        _terminal_serving("http://box.invalid"),
        _opened(cwd=str(folder)),
    )

    assert hosts.launches[0].hosting.cwd == str(folder)


async def test_a_tilde_folder_is_resolved_against_the_home_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    (home / "project").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    _ = _connector(monkeypatch, _FakeSocket())
    hosts = _hosts(monkeypatch)

    await _terminal_run(
        _terminal_serving("http://box.invalid"),
        _opened(cwd="~/project"),
    )

    assert hosts.launches[0].hosting.cwd == str(home / "project")


async def test_a_folder_that_is_missing_or_a_file_ends_the_terminal_and_runs_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file = tmp_path / "notes.txt"
    _ = file.write_text("not a folder")
    refusals: list[tuple[bytes, int]] = []
    hosts = _hosts(monkeypatch)

    # A ``~name`` of a user this host lacks has no home to resolve against.
    stranger = "~no-such-user-c8/project"
    for folder in (str(tmp_path / "gone"), str(file), stranger):
        socket = _FakeSocket()
        _ = _connector(monkeypatch, socket)
        await _terminal_run(
            _terminal_serving("http://box.invalid", "/bin/sh", "-c", "echo ran"),
            _opened(cwd=folder),
        )
        refusals.append(_result(socket))
    assert hosts.launches == []

    assert refusals == [
        (
            f"knowop: {tmp_path / 'gone'} is not a folder on this host\r\n".encode(),
            1,
        ),
        (f"knowop: {file} is not a folder on this host\r\n".encode(), 1),
        (f"knowop: {stranger} is not a folder on this host\r\n".encode(), 1),
    ]
    # The refusal is logged as a close with the status it reported, so the log pairs
    # every open with its close as it does for a shell.
    closes = [line for line in _log_lines() if line.event == "close"]
    assert [line.status for line in closes] == [1, 1, 1]


class _Socket:
    """A WebSocket socket a refusal writes to; ``gone`` makes every send find it closed."""

    def __init__(self, *, gone: bool = False) -> None:
        self.gone = gone
        self.sent: list[bytes] = []

    async def send(self, message: bytes) -> None:
        if self.gone:
            raise ConnectionClosed(None, None)
        self.sent.append(message)


async def test_a_refusal_sends_its_line_then_exit_one_to_a_socket_still_open() -> None:
    socket = _Socket()

    await _end_with_line(socket, "knowop: x is not a folder on this host")

    assert socket.sent == [
        encode_frame(Frame.OUTPUT, b"knowop: x is not a folder on this host\r\n"),
        encode_frame(Frame.EXIT, (1).to_bytes(4)),
    ]


async def test_a_refusal_to_a_socket_already_closed_is_not_an_error() -> None:
    await _end_with_line(
        _Socket(gone=True),
        "knowop: x is not a folder on this host",
    )


class _RelayingShell(_HeldShell):
    """A shell that runs on, and counts its bridge at ``relaying`` once it is reached."""

    def __init__(self, relaying: threading.Barrier) -> None:
        super().__init__(b"17\n", status=None)
        self._relaying = relaying

    @override
    async def start(self) -> None:
        await super().start()
        _ = await asyncio.to_thread(self._relaying.wait, 10)


async def _five_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TerminalServing, _Client, list[_FakeSocket], _Hosts]:
    """Open five independent fake sockets and shells through the real daemon threads."""
    sockets = [_FakeSocket() for _ in range(5)]
    _ = _connector(monkeypatch, *sockets)
    # Each bridge relays on a thread of its own; all five must be relaying, where a
    # stop reaches them, before the test goes on.
    relaying = threading.Barrier(6)
    hosts = _hosts(monkeypatch, *(_RelayingShell(relaying) for _ in range(5)))
    client = _Client([_opened() for _ in range(5)])
    daemon = _terminal_serving("http://box.invalid", "/bin/sh", client=client)
    daemon.start(_HOST, _INSTANCE)
    _ = await asyncio.to_thread(relaying.wait, 10)
    return daemon, client, sockets, hosts


async def test_a_daemon_serves_five_terminals_at_once_and_refuses_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, _, sockets, hosts = await _five_opens(monkeypatch)

    open_now = [not socket.closed for socket in sockets]
    await asyncio.to_thread(daemon.stop)

    assert open_now == [True] * 5
    assert len(hosts.launches) == 5


async def test_stop_lets_go_of_every_shell_and_the_listener_of_its_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stopping daemon closes its sockets and leaves each shell for the next one."""
    daemon, client, sockets, hosts = await _five_opens(monkeypatch)

    await asyncio.to_thread(daemon.stop)

    assert all(socket.closed for socket in sockets)
    assert hosts.stops == []
    # The listener's held request ends by itself, then it lets go of its client.
    assert await asyncio.to_thread(client.closed.wait, 10)
    records = _log_lines()
    assert [record.event for record in records] == ["open"] * 5
    shells = state_dir() / "rekursiv-ai" / "knowop" / "terminals"
    assert len(list(shells.glob("*/terminal.json"))) == 5


def _pause_bridge_start(
    thread: threading.Thread,
    *,
    start: Callable[[threading.Thread], None],
    arrived: threading.Event,
    resume: threading.Event,
) -> None:
    if thread.name.startswith("terminal-"):
        arrived.set()
        assert resume.wait(2)
    start(thread)


async def test_shutdown_cannot_join_a_bridge_before_its_thread_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrived, resume = threading.Event(), threading.Event()
    monkeypatch.setattr(
        threading.Thread,
        "start",
        functools.partialmethod(
            _pause_bridge_start,
            start=threading.Thread.start,
            arrived=arrived,
            resume=resume,
        ),
    )
    _ = _connector(monkeypatch, _FakeSocket(hang=True))
    daemon = _terminal_serving("http://127.0.0.1:1")
    opening = asyncio.create_task(
        asyncio.to_thread(
            daemon._start_bridge,
            _opened(),
            host_id=_HOST,
            instance=_INSTANCE,
        ),
    )
    try:
        assert await asyncio.to_thread(arrived.wait, 2)
        stopping = asyncio.create_task(asyncio.to_thread(daemon.stop))
        # Stop sets its flag, then waits on the lock the paused start holds.
        assert await asyncio.to_thread(daemon._stop.wait, 2)
        resume.set()
        await asyncio.wait_for(asyncio.gather(opening, stopping), 2)
    finally:
        resume.set()
        await opening
        await asyncio.to_thread(daemon.stop)


async def test_a_closed_loop_in_the_shutdown_snapshot_does_not_skip_other_bridges() -> (
    None
):
    daemon = _terminal_serving("http://127.0.0.1:1")
    closed = asyncio.new_event_loop()
    closed.close()
    ended = asyncio.Event()
    daemon._loops[uuid.uuid4()] = (closed, asyncio.Event())
    daemon._loops[uuid.uuid4()] = (asyncio.get_running_loop(), ended)

    await asyncio.to_thread(daemon.stop)
    await asyncio.wait_for(ended.wait(), 1)


class _Waited(threading.Thread):
    """A bridge that never runs: it keeps how long each wait for its end may take."""

    def __init__(self) -> None:
        super().__init__()
        self.waits: list[float | None] = []

    @override
    def join(self, timeout: float | None = None) -> None:
        self.waits.append(timeout)


async def test_stop_waits_ten_seconds_for_each_bridge_and_no_longer() -> None:
    """A bridge that will not end must not hold the daemon's exit for good."""
    daemon = _terminal_serving("http://127.0.0.1:1")
    bridges = [_Waited(), _Waited()]
    daemon._bridges.update({uuid.uuid4(): bridge for bridge in bridges})

    await asyncio.to_thread(daemon.stop)

    assert [bridge.waits for bridge in bridges] == [[10], [10]]


async def test_start_takes_up_each_live_shell_of_this_host_and_forgets_each_dead_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live, dead, other = _opened(), _opened(), _opened()
    live_folder = _held_record(live)
    dead_folder = _held_record(dead)
    other_folder = _held_record(other, host_id=uuid.uuid4())
    corrupt = state_dir() / "rekursiv-ai" / "knowop" / "terminals" / "corrupt"
    corrupt.mkdir()
    _ = (corrupt / "terminal.json").write_text("{")
    alive = {socket_address(live_folder / "host.sock")}
    monkeypatch.setattr(terminal_serving, "accepts", alive.__contains__)
    client = _Client([])
    daemon = _terminal_serving("http://127.0.0.1:1", client=client)
    started = Mock()
    monkeypatch.setattr(daemon, "_start_bridge", started)

    daemon.start(_HOST, _INSTANCE)
    daemon.stop()
    assert await asyncio.to_thread(client.closed.wait, 10)

    started.assert_called_once_with(
        live,
        host_id=_HOST,
        instance=_INSTANCE,
        resume=True,
    )
    assert not dead_folder.exists()
    assert other_folder.exists()
    assert corrupt.exists()


async def test_a_410_stops_every_shell_of_this_host_and_ends_the_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hosts = _hosts(monkeypatch)
    mine, other = _opened(), _opened()
    mine_folder = _held_record(mine)
    mine_address = socket_address(mine_folder / "host.sock")
    other_folder = _held_record(other, host_id=uuid.uuid4())
    monkeypatch.setattr(terminal_serving, "accepts", _accepting)
    client = _Client([ClientError("-> 410", status_code=410)])
    daemon = _terminal_serving("http://127.0.0.1:1", client=client)
    monkeypatch.setattr(daemon, "_start_bridge", Mock())

    daemon.start(_HOST, _INSTANCE)
    assert await asyncio.to_thread(client.closed.wait, 10)

    assert hosts.stops == [mine_address]
    assert not mine_folder.exists()
    assert other_folder.exists()


async def test_a_wait_that_fails_otherwise_is_said_and_waited_out_until_stopped(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _Client([ClientError("-> 503", status_code=503)])
    daemon = _terminal_serving("http://127.0.0.1:1", client=client)

    daemon.start(_HOST, _INSTANCE)
    assert await asyncio.to_thread(client.failed.wait, 10)
    daemon.stop()
    assert await asyncio.to_thread(client.closed.wait, 10)

    assert "knowop: terminal wait failed: -> 503" in capsys.readouterr().err


class _Held:
    """Answers ``answers`` in order, then holds the next wait until the test lets it go.

    A held wait is one the server holds, or one a pause cut.
    """

    def __init__(self, *answers: TerminalWait | ClientError) -> None:
        self.asks = 0
        self.asked = threading.Event()
        """Set once a wait is held."""
        self.release = threading.Event()
        self.closed = threading.Event()
        self.threads: list[str] = []
        """The thread each wait was asked on."""
        self._answers = list(answers)

    def wait_terminals(
        self,
        host_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        wait_sec: float,
    ) -> TerminalWait | None:
        del host_id, instance, wait_sec
        self.asks += 1
        self.threads.append(threading.current_thread().name)
        if self._answers:
            answer = self._answers.pop(0)
            if isinstance(answer, ClientError):
                raise answer
            return answer
        self.asked.set()
        assert self.release.wait(10)
        return None

    def close(self) -> None:
        self.closed.set()


def _listening(
    *clients: _Held,
    wake: Wake | None = None,
) -> tuple[TerminalServing, list[str]]:
    """Return a daemon whose each listener takes the next client, and the keys given."""
    keys: list[str] = []
    waiting = list(clients)

    def next_client(key: str) -> _Held:
        keys.append(key)
        return waiting.pop(0)

    daemon = TerminalServing(
        name="box",
        server_url="http://127.0.0.1:1",
        credential=_CREDENTIAL,
        credential_client=next_client,
        trax_env={},
        wake=wake,
    )
    return daemon, keys


class _Stop(threading.Event):
    """A daemon's stop that keeps each timed wait on it and waits none of them out."""

    def __init__(self) -> None:
        super().__init__()
        self.waited: list[float | None] = []

    @override
    def wait(self, timeout: float | None = None) -> bool:
        self.waited.append(timeout)
        if timeout is None:
            return super().wait()
        return self.is_set()


async def test_a_wait_that_fails_is_waited_out_five_seconds_then_asked_again() -> None:
    client = _Held(ClientError("-> 503", status_code=503))
    daemon, _ = _listening(client)
    daemon._stop = stop = _Stop()
    daemon.start(_HOST, _INSTANCE)

    assert await asyncio.to_thread(client.asked.wait, 10)
    daemon.stop()
    client.release.set()
    assert await asyncio.to_thread(client.closed.wait, 10)

    assert (client.asks, stop.waited) == (2, [5.0])


async def test_each_open_starts_a_bridge_for_this_host_and_daemon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = _opened()
    client = _Held(opened)
    daemon, _ = _listening(client)
    started = Mock()
    monkeypatch.setattr(daemon, "_start_bridge", started)
    daemon.start(_HOST, _INSTANCE)

    assert await asyncio.to_thread(client.asked.wait, 10)
    daemon.stop()
    client.release.set()
    assert await asyncio.to_thread(client.closed.wait, 10)

    started.assert_called_once_with(opened, host_id=_HOST, instance=_INSTANCE)


async def test_a_wake_from_a_pause_starts_a_new_wait_and_the_dead_one_ends_after() -> (
    None
):
    dead, fresh = _Held(), _Held()
    wall = [0.0]
    wake = Wake(wall=lambda: wall[0], clock=lambda: 0.0)
    daemon, keys = _listening(dead, fresh, wake=wake)
    daemon.start(_HOST, _INSTANCE)
    assert await asyncio.to_thread(dead.asked.wait, 10)

    wall[0] += 30.0
    assert wake.check()
    assert await asyncio.to_thread(fresh.asked.wait, 10)
    dead.release.set()
    assert await asyncio.to_thread(dead.closed.wait, 10)
    daemon.stop()
    fresh.release.set()
    assert await asyncio.to_thread(fresh.closed.wait, 10)
    wall[0] += 30.0
    assert wake.check()

    assert dead.asks == 1
    # Each listener asks with the host's own key, on a thread named for it.
    assert keys == [_CREDENTIAL, _CREDENTIAL]
    assert dead.threads == fresh.threads == ["terminal-listener"]


async def test_an_open_that_arrives_once_the_daemon_stopped_starts_no_bridge() -> None:
    daemon = _terminal_serving("http://127.0.0.1:1")
    opened = _opened()
    daemon.stop()

    daemon._start_bridge(opened, host_id=_HOST, instance=_INSTANCE)

    assert f"terminal-{opened.terminal_id}" not in {
        thread.name for thread in threading.enumerate()
    }


async def test_a_socket_closed_before_the_shell_starts_runs_nothing_and_logs_a_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _ClosingSocket()
    _ = _connector(monkeypatch, socket)
    hosts = _hosts(monkeypatch)

    await _terminal_run(_terminal_serving("http://box.invalid", "/bin/sh"), _opened())

    assert socket.closed
    assert hosts.launches == []
    assert '"event":"close"' in _startup_log()
    assert '"status":1' in _startup_log()


async def test_the_shells_output_ends_the_pump_and_a_closed_relay_says_so() -> None:
    """Whatever the shell still writes has nowhere to go; the pump says the relay closed."""
    gone = DirectRouter(relay=_relay_gone, shell=_Shell(), factory=None)
    kept = DirectRouter(relay=_relay_kept, shell=_Shell(), factory=None)

    assert not await _pump(cast("attach.RemoteTerminal", _Output(b"late")), gone)
    assert await _pump(cast("attach.RemoteTerminal", _Output(b"late")), kept)


@pytest.mark.parametrize(
    ("messages", "closes", "ends"),
    [
        (("text, not a frame", _INPUT), False, True),
        ((b"\xff", _INPUT), False, True),
        ((encode_frame(Frame.STOP), _INPUT), False, True),
        ((), True, False),
        ((), False, False),
    ],
    ids=["text", "no frame", "stop", "closed", "ended"],
)
async def test_the_server_socket_is_read_until_a_stop_what_is_no_frame_or_its_close(
    messages: tuple[str | bytes, ...],
    closes: bool,
    ends: bool,
) -> None:
    shell = _Shell()
    router = DirectRouter(relay=_relay_gone, shell=shell, factory=None)

    read = await _read(
        router,
        cast("ClientConnection", _Messages(*messages, closes=closes)),
    )

    assert read is ends
    assert shell.written == []


async def test_an_offer_is_answered_and_the_shells_output_then_goes_direct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = _ScriptedPeer()
    offer = wire_terminals.DirectSignal(gen=1, type="offer", sdp="offer-sdp")
    socket = _FakeSocket(
        messages=(encode_frame(Frame.SIGNAL, offer.model_dump_json().encode()),),
    )
    _ = _connector(monkeypatch, socket)
    shell = _HeldShell(b"got-go\r\n", after_input=True)
    _ = _hosts(monkeypatch, shell)
    daemon = _terminal_serving("http://box.invalid", "/bin/sh", direct=lambda: peer)

    await _terminal_run(daemon, _opened(ice_servers=["stun:s.test:3478"]))

    assert shell.written == [b"go\n"]
    relayed = [
        frame
        for message in socket.sent
        if (frame := decode_frame(message))[0] is not Frame.ACK
    ]
    answer = wire_terminals.DirectSignal.model_validate_json(relayed[0][1])
    assert (relayed[0][0], answer.type, answer.sdp) == (
        Frame.SIGNAL,
        "answer",
        "answer-sdp",
    )
    assert peer.ice_servers == ("stun:s.test:3478",)
    assert b"got-go" in b"".join(p for kind, p in peer.sent if kind is Frame.OUTPUT)
    assert (Frame.EXIT, (0).to_bytes(4)) in peer.sent
    assert all(kind is not Frame.OUTPUT for kind, _ in relayed)
    assert relayed[-1] == (Frame.EXIT, (0).to_bytes(4))


async def test_a_terminal_without_a_shell_of_its_own_runs_the_login_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _connector(monkeypatch, _FakeSocket())
    hosts = _hosts(monkeypatch)

    await _terminal_run(_terminal_serving("http://box.invalid"), _opened())

    assert hosts.launches[0].command == [pwd.getpwuid(os.getuid()).pw_shell, "-l"]


@dataclass(slots=True, kw_only=True)
class _StartupClock:
    now: float = 100.0
    pauses: list[float] = field(default_factory=list[float])

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.pauses.append(delay)
        self.now += delay


@pytest_asyncio.fixture
async def startup_clock(monkeypatch: pytest.MonkeyPatch) -> _StartupClock:
    """Advance startup time only by the delays the real retry loop requests."""
    clock = _StartupClock()
    monkeypatch.setattr(asyncio.get_running_loop(), "time", clock)
    monkeypatch.setattr(asyncio, "sleep", clock.sleep)
    return clock


def _claimed(result: object) -> _Claim:
    assert isinstance(result, _Claim)
    return result


async def test_connect_sends_the_credential_and_the_timeouts_a_socket_needs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _FakeSocket()
    connector = _connector(monkeypatch, socket)
    assert _claimed(await _connect()) == _Claim(
        socket=cast("ClientConnection", socket),
        offset=0,
        resume=False,
    )
    (kwargs,) = connector.calls
    assert kwargs["additional_headers"] == {"Authorization": f"Bearer {_CREDENTIAL}"}
    assert kwargs["ping_interval"] == 20
    assert kwargs["close_timeout"] == 1
    assert kwargs["open_timeout"] == 10.0


async def test_a_resume_claim_reads_the_offset_the_servers_relay_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _FakeSocket(reply=encode_frame(Frame.ACK, (77).to_bytes(8)))
    connector = _connector(monkeypatch, socket)
    opened = _opened()

    claim = _claimed(await _connect(opened, resume=True))

    assert (claim.offset, claim.resume) == (77, True)
    assert connector.urls == [
        f"ws://box.invalid{_path(opened)}?instance={_INSTANCE}&ready=true&resume=true",
    ]


@pytest.mark.parametrize(
    ("reply", "resume", "offset"),
    [
        (encode_frame(Frame.ACK), False, 0),
        (encode_frame(Frame.ACK, (5).to_bytes(8)), True, 5),
        (encode_frame(Frame.ACK, (5).to_bytes(8)), False, None),
        (encode_frame(Frame.ACK), True, None),
        (encode_frame(Frame.OUTPUT), False, None),
        ("ACK", False, None),
    ],
    ids=["first", "resume", "offset-to-a-first", "empty-to-a-resume", "no-ack", "text"],
)
async def test_an_acknowledgement_names_the_offset_its_claim_expects(
    reply: str | bytes,
    resume: bool,
    offset: int | None,
) -> None:
    assert _acknowledged(reply, resume=resume) == offset


async def test_connect_caps_the_open_timeout_by_the_time_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = _connector(monkeypatch, _FakeSocket())
    _ = await _connect(open_timeout_sec=1000)
    open_timeout = connector.calls[0]["open_timeout"]
    assert isinstance(open_timeout, float)
    assert 44 < open_timeout <= 45


async def test_connect_pauses_between_attempts_for_the_retry_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _connector(monkeypatch, OSError("down"), _FakeSocket())
    pauses: list[float] = []

    async def pause(delay: float) -> None:
        pauses.append(delay)

    monkeypatch.setattr(asyncio, "sleep", pause)
    _ = _claimed(await _connect())
    assert pauses == [1.0]


async def test_connect_starts_an_attempt_with_under_a_second_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _FakeSocket()
    _ = _connector(monkeypatch, socket)
    assert _claimed(await _connect(startup_timeout_sec=0.5)).socket is socket


async def test_connect_without_time_never_dials_and_says_the_deadline_passed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    connector = _connector(monkeypatch, _FakeSocket())
    # A frozen clock makes the time left exactly zero, which is no time at all.
    now = asyncio.get_running_loop().time()
    monkeypatch.setattr(asyncio.get_running_loop(), "time", lambda: now)
    assert await _connect(startup_timeout_sec=0) == _Refusal(code=None)
    assert not connector.calls
    assert "): connection deadline exceeded" in capsys.readouterr().err
    line = _startup_log()
    assert '"event":"startup_failed"' in line
    assert '"status":1' in line


async def test_connect_stops_on_a_wrong_acknowledgement_and_closes_the_socket(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    socket = _FakeSocket(reply=b"nope")
    connector = _connector(monkeypatch, socket)
    assert await _connect() == _Refusal(code=None)
    assert len(connector.calls) == 1
    assert socket.closed
    assert "): invalid startup acknowledgement" in capsys.readouterr().err
    assert '"event":"startup_failed"' in _startup_log()


@pytest.mark.parametrize(
    "code",
    [
        wire_terminals.CLOSE_REFUSED,
        wire_terminals.CLOSE_HOST_REVOKED,
        wire_terminals.CLOSE_ENDED,
    ],
)
async def test_connect_stops_at_a_refusal_and_keeps_its_code(
    monkeypatch: pytest.MonkeyPatch,
    code: int,
) -> None:
    connector = _connector(monkeypatch, ConnectionClosed(Close(code, ""), None))
    assert await _connect(resume=True) == _Refusal(code=code)
    assert len(connector.calls) == 1
    assert '"event":"resume_failed"' in _startup_log()


async def test_connect_names_the_error_that_ended_the_attempts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    startup_clock: _StartupClock,
) -> None:
    _ = _connector(monkeypatch, OSError("boom"))
    assert await _connect(startup_timeout_sec=0.03, retry_sec=0.005) == _Refusal(
        code=None,
    )
    assert capsys.readouterr().err.endswith("): boom\n")
    assert sum(startup_clock.pauses) == pytest.approx(0.03)


async def test_connect_names_an_error_without_a_message_by_its_type(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    startup_clock: _StartupClock,
) -> None:
    _ = _connector(monkeypatch, OSError())
    assert await _connect(startup_timeout_sec=0.03, retry_sec=0.005) == _Refusal(
        code=None,
    )
    assert capsys.readouterr().err.endswith("): OSError\n")
    assert sum(startup_clock.pauses) == pytest.approx(0.03)


async def test_connect_logs_each_retry_with_the_terminal_and_reason(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    startup_clock: _StartupClock,
) -> None:
    _ = _connector(monkeypatch, OSError("boom"), _FakeSocket())
    caplog.set_level(logging.DEBUG)
    opened = _opened()
    _ = await _connect(opened, retry_sec=0.001)
    assert startup_clock.pauses == [0.001]
    retry = [r for r in caplog.records if "startup retry" in r.getMessage()]
    (record,) = retry
    assert record.levelno == logging.DEBUG
    assert record.getMessage() == f"terminal {opened.terminal_id}: startup retry: boom"


@pytest.mark.parametrize(
    ("code", "retried"),
    [(404, False), (499, False), (429, True), (500, True), (503, True)],
)
async def test_connect_retries_only_what_a_retry_can_fix(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    code: int,
    retried: bool,
    startup_clock: _StartupClock,
) -> None:
    connector = _connector(monkeypatch, _status(code))
    assert await _connect(startup_timeout_sec=0.05, retry_sec=0.002) == _Refusal(
        code=None,
    )
    assert (len(connector.calls) > 2) is retried
    if not retried:
        assert len(connector.calls) == 1
    assert sum(startup_clock.pauses) == pytest.approx(0.05 if retried else 0)
    assert "knowop: terminal startup failed" in capsys.readouterr().err
    assert '"event":"startup_failed"' in _startup_log()


async def test_connect_never_sleeps_past_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
    startup_clock: _StartupClock,
) -> None:
    _ = _connector(monkeypatch, OSError("down"))
    result = await asyncio.wait_for(
        _connect(startup_timeout_sec=0.05, retry_sec=5.0),
        1,
    )
    assert result == _Refusal(code=None)
    assert startup_clock.pauses == [pytest.approx(0.05)]


async def test_connect_closes_a_socket_that_fails_during_the_handshake(
    monkeypatch: pytest.MonkeyPatch,
    startup_clock: _StartupClock,
) -> None:
    socket = _FakeSocket(send_error=ConnectionClosed(None, None))
    _ = _connector(monkeypatch, socket)
    assert await _connect(startup_timeout_sec=0.03, retry_sec=0.005) == _Refusal(
        code=None,
    )
    assert socket.closed
    assert sum(startup_clock.pauses) == pytest.approx(0.03)


async def test_connect_closes_a_socket_when_cancelled_mid_handshake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _FakeSocket(hang=True)
    _ = _connector(monkeypatch, socket)
    task = asyncio.create_task(_connect())
    _ = await asyncio.wait_for(socket.waiting.wait(), 1)
    _ = task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert socket.closed


async def test_the_socket_dials_the_servers_own_host_over_wss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = _connector(monkeypatch, _FakeSocket())
    _ = _hosts(monkeypatch)
    opened = _opened()

    # ``http`` twice, a trailing slash, and a last letter ``rstrip`` might take too.
    await _terminal_run(_terminal_serving("https://http.boX/", "/bin/true"), opened)

    assert connector.urls == [
        f"wss://http.boX{_path(opened)}?instance={_INSTANCE}&ready=true",
    ]


class _SlowClose(_FakeSocket):
    """A socket whose close waits until the test lets it finish, as a slow peer's."""

    def __init__(self) -> None:
        super().__init__()
        self.closing = threading.Event()
        self.release = asyncio.Event()

    @override
    async def close(self) -> None:
        self.closing.set()
        _ = await self.release.wait()
        self.closed = True


async def test_a_bridge_is_a_daemon_thread_whose_stop_waits_out_its_close(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    socket = _SlowClose()
    connector = _connector(monkeypatch, socket)
    relaying = threading.Barrier(2)
    _ = _hosts(monkeypatch, _RelayingShell(relaying))
    daemon = _terminal_serving("http://box.invalid", "/bin/sh")
    opened = _opened()

    daemon._start_bridge(opened, host_id=_HOST, instance=_INSTANCE)
    [bridge] = [
        thread
        for thread in threading.enumerate()
        if thread.name == f"terminal-{opened.terminal_id}"
    ]
    _ = await asyncio.to_thread(relaying.wait, 10)
    # Said as the terminal opens, while its shell runs.
    assert capsys.readouterr().out == "terminal opened by ada@x.y\n"
    stopping = asyncio.ensure_future(asyncio.to_thread(daemon.stop))
    assert await asyncio.to_thread(socket.closing.wait, 10)
    assert not stopping.done()
    loop, _ = daemon._loops[opened.terminal_id]
    _ = loop.call_soon_threadsafe(socket.release.set)
    await stopping

    assert bridge.daemon
    assert socket.closed
    # Each bridge is forgotten once it ends.
    assert (daemon._bridges, daemon._loops) == ({}, {})
    assert connector.urls == [
        f"ws://box.invalid{_path(opened)}?instance={_INSTANCE}&ready=true",
    ]


class _Unanswered:
    """A dial no server answers: it yields until a timeout cancels it, or fails."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, url: str, **kwargs: object) -> _FakeSocket:
        del url, kwargs
        self.calls += 1
        await _turns(100)
        raise AssertionError("no timeout ended the dial")


async def _turns(count: int) -> None:
    """Let the loop run ``count`` times, with no clock: ``startup_clock`` stops one."""
    loop = asyncio.get_running_loop()
    for _ in range(count):
        turn = loop.create_future()
        _ = loop.call_soon(turn.set_result, None)
        await turn


async def _slow_to_end(*chunks: bytes, turns: int) -> AsyncIterator[bytes]:
    """Yield ``chunks``, then hold; cancelled, it takes ``turns`` of the loop to end."""
    try:
        for chunk in chunks:
            yield chunk
        _ = await asyncio.Event().wait()
    finally:
        await _turns(turns)


@dataclass(frozen=True, slots=True, kw_only=True)
class _SlowToEndSocket:
    """A server socket that sends nothing, and takes ``turns`` to stop being read."""

    turns: int
    close_code: int | None = None

    def __aiter__(self) -> AsyncIterator[bytes]:
        return _slow_to_end(turns=self.turns)

    async def send(self, message: bytes) -> None:
        del message


class _SlowToEndStop(asyncio.Event):
    """A stop never set, whose wait takes ``turns`` of the loop to end once cancelled."""

    def __init__(self, *, turns: int) -> None:
        super().__init__()
        self._turns = turns

    @override
    async def wait(self) -> Literal[True]:
        try:
            return await super().wait()
        finally:
            await _turns(self._turns)


class _SlowToEndEvent(_SlowToEndStop):
    """``asyncio.Event`` as a bridge makes its own stop: slow to end once cancelled."""

    def __init__(self) -> None:
        super().__init__(turns=10)


async def test_a_bridge_returns_with_nothing_of_its_own_left_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its stop's watcher, cancelled as the bridge ends, is done before it returns."""
    # A dial refused for good: the work ends at once, and the watcher is left.
    _ = _connector(monkeypatch, _status(404))
    monkeypatch.setattr(asyncio, "Event", _SlowToEndEvent)
    daemon = _terminal_serving("http://box.invalid")
    before = asyncio.all_tasks()

    await daemon._bridge(_opened(), host_id=_HOST, instance=_INSTANCE, resume=False)

    assert asyncio.all_tasks() - before - {asyncio.current_task()} == set()


async def test_connect_times_out_each_dial_no_server_answers(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    startup_clock: _StartupClock,
) -> None:
    dial = _Unanswered()
    monkeypatch.setattr(terminal_serving, "connect", dial)

    assert await _connect(
        open_timeout_sec=0,
        startup_timeout_sec=0.002,
        retry_sec=0.001,
    ) == _Refusal(code=None)

    assert dial.calls == 2
    assert len(startup_clock.pauses) == 2
    assert sum(startup_clock.pauses) == pytest.approx(0.002)
    assert capsys.readouterr().err.endswith("): TimeoutError\n")


class _SlowToEndShell(_HeldShell):
    """A shell whose output, once cancelled, takes ``turns`` of the loop to end."""

    def __init__(self, *, turns: int) -> None:
        super().__init__(b"17", status=None)
        self._turns = turns

    @override
    async def output(self) -> AsyncIterator[bytes]:
        async for chunk in _slow_to_end(*self.chunks, turns=self._turns):
            yield chunk


@pytest.mark.parametrize("slow", ["output", "reader", "stop"])
async def test_a_relay_the_daemon_lets_go_of_returns_with_nothing_left_running(
    monkeypatch: pytest.MonkeyPatch,
    slow: str,
) -> None:
    """Its pump, reader and stop end before it returns."""
    # Each still pending as the daemon stops; the slow one, not waited for, would still
    # be running at the return, as the others end at once.
    turns = {name: 10 if name == slow else 0 for name in ("output", "reader", "stop")}
    shell = _SlowToEndShell(turns=turns["output"])
    _ = _hosts(monkeypatch, shell)
    stop = _SlowToEndStop(turns=turns["stop"])
    socket = _SlowToEndSocket(turns=turns["reader"])
    daemon = _terminal_serving("http://box.invalid")
    before = asyncio.all_tasks()
    letting_go = asyncio.get_running_loop().call_later(0.01, stop.set)

    ending = await daemon._relay(
        _opened(),
        _Claim(socket=cast("ClientConnection", socket), offset=0, resume=False),
        Path("unused"),
        stop,
    )

    letting_go.cancel()
    assert ending == (_Ending.LET_GO, None)
    assert shell.closed
    assert asyncio.all_tasks() - before - {asyncio.current_task()} == set()


class _ClosingSocket(_FakeSocket):
    """A socket that takes the startup handshake, then finds every later send closed."""

    def __init__(self, *, reply: bytes = encode_frame(Frame.ACK)) -> None:
        super().__init__(reply=reply)
        self.sends = 0

    @override
    async def send(self, data: bytes) -> None:
        del data
        self.sends += 1
        if self.sends > 1:
            raise ConnectionClosed(None, None)


@dataclass(slots=True, kw_only=True)
class _Shell:
    """The pty a router types into, recording what it was sent."""

    written: list[bytes] = field(default_factory=list[bytes])

    async def write(self, data: bytes) -> bool:
        self.written.append(data)
        return True

    def set_winsize(self, rows: int, cols: int) -> None:
        del rows, cols


class _Output:
    """A pty that wrote ``chunks``, then closed."""

    def __init__(self, *chunks: bytes) -> None:
        self._chunks = chunks

    async def output(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


def _systemd(argv: Sequence[str]) -> tuple[int, str]:
    """Answer as a computer where systemd runs every scope it is asked for."""
    del argv
    return 0, ""


def _accepting(address: Path) -> bool:
    """Say that a host listens at every address."""
    del address
    return True


async def _relay_gone(data: bytes) -> None:
    """Send to a relay the server has already closed."""
    del data
    raise ConnectionClosed(None, None)


async def _relay_kept(data: bytes) -> None:
    """Send to a relay that takes everything."""
    del data


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
