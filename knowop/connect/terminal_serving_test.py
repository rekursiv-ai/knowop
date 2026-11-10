"""TerminalServing with process and WebSocket boundaries supplied inline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast, override
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, call

import asyncio
import functools
import logging
import os
import pwd
import shutil
import stat
import sys
import threading
import uuid

from trackinizer.client.errors import ClientError
from trackinizer.lib.posix.host import Frame, decode_frame, encode_frame, encode_winsize
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
    _fail_files,
    _FakeSocket,
    _finish_files,
    _Messages,
    _opened,
    _ScriptedPeer,
    _startup_log,
    _status,
    _terminal_run,
    _terminal_serving,
)
from knowop.connect.files_channel import FilesChannel, Info
from knowop.connect.terminal_direct import DirectRouter
from knowop.connect.terminal_serving import (
    TerminalServing,
    _end_with_line,
    _LogLine,
    _pump,
    _read,
)
from knowop.connect.wake import Wake


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from trackinizer.mock_test_support import AsyncMockProtocol, MockProtocol
    from trackinizer.wire.wire_terminals import TerminalWait
    from websockets.asyncio.client import ClientConnection

pytestmark = pytest.mark.asyncio

_INPUT = encode_frame(Frame.INPUT, b"typed\n")


async def test_a_shells_output_and_exit_status_arrive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _FakeSocket()
    connector = _connector(monkeypatch, socket)
    _, launch = _terminal(monkeypatch, b"ok", status=3)
    opened = _opened()

    await _terminal_run(
        _terminal_serving("http://box.invalid", "/bin/sh", "-c", "printf ok; exit 3"),
        opened,
    )

    path = wire_terminals.HOST_TERMINAL_WS_PATH.format(
        host_id=_HOST,
        terminal_id=opened.terminal_id,
    )
    assert connector.urls == [f"ws://box.invalid{path}?instance={_INSTANCE}&ready=true"]
    assert connector.calls[0]["additional_headers"] == {
        "Authorization": f"Bearer {_CREDENTIAL}",
    }
    assert _result(socket) == (b"ok", 3)
    assert launch.call_args.args == (["/bin/sh", "-c", "printf ok; exit 3"],)


async def test_startup_retries_handshake_timeouts_and_launches_the_cli_once(
    monkeypatch: pytest.MonkeyPatch,
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
    _, launch = _terminal(monkeypatch, b"ready")
    opened = _opened(command="claude", run_name="chat-retry")

    await asyncio.wait_for(
        _terminal_run(_terminal_serving("http://box.invalid"), opened),
        3,
    )

    path = wire_terminals.HOST_TERMINAL_WS_PATH.format(
        host_id=_HOST,
        terminal_id=opened.terminal_id,
    )
    assert (
        connector.urls
        == [f"ws://box.invalid{path}?instance={_INSTANCE}&ready=true"] * 3
    )
    assert len(connector.calls) == 3
    assert all(
        0 < cast(float, kwargs["open_timeout"]) <= 0.02 for kwargs in connector.calls
    )
    launch.assert_called_once_with(
        [sys.executable, *_KNOWOP, "claude", "--detach", "--name", "chat-retry"],
        cwd=ANY,
        env=ANY,
        clean_env=False,
        drop_env=ANY,
        terminate_grace_sec=2.0,
        winsize=(24, 80),
    )
    assert socket.sent[:2] == [encode_frame(Frame.ACK)] * 2
    assert b"".join(
        payload
        for kind, payload in map(decode_frame, socket.sent)
        if kind is Frame.OUTPUT
    ).endswith(b"ready")
    assert socket.sent[-1] == encode_frame(Frame.EXIT, (0).to_bytes(4))
    assert socket.closed


async def test_a_refused_startup_never_launches_the_cli_and_reports_one_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ = _connector(
        monkeypatch,
        ConnectionClosed(Close(wire_terminals.CLOSE_REFUSED, ""), None),
    )
    _, launch = _terminal(monkeypatch)

    await _terminal_run(
        _terminal_serving("http://box.invalid"),
        _opened(command="claude", run_name="chat-refused"),
    )

    launch.assert_not_called()
    error = capsys.readouterr().err
    assert "terminal startup failed" in error
    assert "Traceback" not in error


async def test_stop_cancels_startup_while_waiting_for_the_server_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _FakeSocket(hang=True)
    _ = _connector(monkeypatch, socket)
    _, launch = _terminal(monkeypatch)
    daemon = _terminal_serving("http://box.invalid")
    running = asyncio.create_task(
        _terminal_run(daemon, _opened(command="claude", run_name="chat-stopped")),
    )
    try:
        assert await asyncio.wait_for(asyncio.to_thread(socket.received.wait, 2), 2)
        await asyncio.to_thread(daemon.stop)
        await asyncio.wait_for(running, 1)
    finally:
        await asyncio.to_thread(daemon.stop)
        await running

    assert socket.sent == [encode_frame(Frame.ACK)]
    assert socket.closed
    launch.assert_not_called()


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
    terminal, launch = _terminal(monkeypatch)
    typed, finished = asyncio.Event(), asyncio.Event()
    terminal.write.side_effect = functools.partial(_typed, typed)
    terminal.output.return_value = _output_when(
        typed,
        finished,
        b"1 1000\r\ngot-go\r\n",
    )
    terminal.wait.side_effect = functools.partial(_terminal_status, finished, status=0)

    await _terminal_run(
        _terminal_serving(
            "http://box.invalid",
            "/bin/sh",
            "-c",
            "read x; stty size; echo got-$x",
        ),
        _opened(),
    )

    output, status = _result(socket)
    assert status == 0
    assert b"1 1000" in output
    assert b"got-go" in output
    terminal.write.assert_awaited_once_with(b"go\n")
    terminal.set_winsize.assert_called_once_with(1, 1000)
    assert launch.call_args.args == (
        ["/bin/sh", "-c", "read x; stty size; echo got-$x"],
    )


async def test_stop_and_a_closed_socket_each_terminate_and_reap_the_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = "printf %s $$; sleep 0.02; echo; sleep 30"
    for messages, closes in (((encode_frame(Frame.STOP),), False), ((), True)):
        socket = _FakeSocket(messages=messages, closes=closes)
        _ = _connector(monkeypatch, socket)
        finished, drained = asyncio.Event(), asyncio.Event()
        terminal, launch = _terminal(monkeypatch, status=143)
        terminal.output.return_value = _held_output(finished, drained)
        terminal.wait.side_effect = functools.partial(_terminal_status, finished)
        terminal.terminate.side_effect = finished.set

        await _terminal_run(
            _terminal_serving("http://box.invalid", "/bin/sh", "-c", command),
            _opened(),
        )

        launch.assert_called_once_with(
            ["/bin/sh", "-c", command],
            cwd=ANY,
            env=ANY,
            clean_env=True,
            drop_env=(),
            terminate_grace_sec=2.0,
            winsize=(24, 80),
        )
        assert terminal.mock_calls.index(
            type(call)(("terminate", (), {})),
        ) < terminal.mock_calls.index(
            type(call)(("__aexit__", (None, None, None), {})),
        )
        terminal.terminate.assert_awaited_once_with()
        terminal.wait.assert_awaited_once_with()
        assert drained.is_set()
        assert socket.closed
        assert socket.sent[-1] == encode_frame(Frame.EXIT, (143).to_bytes(4))


class _TerminalMock(MagicMock):
    __aenter__: AsyncMockProtocol[Self]
    __aexit__: AsyncMockProtocol[bool]
    wait: AsyncMockProtocol[int]
    terminate: AsyncMockProtocol[object]
    write: AsyncMockProtocol[object]
    set_winsize: MockProtocol[object]
    output: MockProtocol[AsyncIterator[bytes]]


def _terminal(
    monkeypatch: pytest.MonkeyPatch,
    *chunks: bytes,
    status: int = 0,
) -> tuple[_TerminalMock, Mock]:
    terminal = _TerminalMock(spec=Terminal)
    terminal.__aenter__.return_value = terminal
    terminal.wait.return_value = status
    terminal.output.return_value = _Output(*chunks).output()
    launch = Mock(return_value=terminal)
    monkeypatch.setattr(terminal_serving, "Terminal", launch)
    return terminal, launch


def _result(socket: _FakeSocket) -> tuple[bytes, int]:
    output = b""
    for message in socket.sent:
        kind, payload = decode_frame(message)
        if kind is Frame.OUTPUT:
            output += payload
        elif kind is Frame.EXIT:
            return output, int.from_bytes(payload)
    raise AssertionError("closed before the shell exited")


async def _typed(ready: asyncio.Event, data: bytes) -> bool:
    del data
    ready.set()
    return True


async def _output_when(
    ready: asyncio.Event,
    finished: asyncio.Event,
    *chunks: bytes,
) -> AsyncIterator[bytes]:
    await ready.wait()
    for chunk in chunks:
        yield chunk
    finished.set()


def _environment(monkeypatch: pytest.MonkeyPatch) -> Mock:
    monkeypatch.setattr(Terminal, "start", AsyncMock())
    monkeypatch.setattr(Terminal, "output", _environment_output)
    launch = Mock(wraps=Terminal)
    monkeypatch.setattr(terminal_serving, "Terminal", launch)
    return launch


async def _environment_output(terminal: Terminal) -> AsyncIterator[bytes]:
    yield "\r\n".join(
        f"{name}={value}" for name, value in terminal._child_env().items()
    ).encode()


async def _held_output(
    finished: asyncio.Event,
    drained: asyncio.Event,
) -> AsyncIterator[bytes]:
    yield b"17"
    yield b"\n"
    await finished.wait()
    drained.set()


async def _terminal_status(finished: asyncio.Event, *, status: int = 143) -> int:
    await finished.wait()
    return status


async def test_the_shell_starts_clean_with_the_trax_context_and_no_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sentinel-secret")
    monkeypatch.setenv("TRACKINIZER_TOKEN", "sentinel-token")
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    socket = _FakeSocket()
    _ = _connector(monkeypatch, socket)
    launch = _environment(monkeypatch)
    url = "http://box.invalid"
    canvas, workspace = uuid.uuid4(), uuid.uuid4()
    daemon = _terminal_serving(
        url,
        "/bin/sh",
        "-c",
        "env",
        trax_env={"LC_TRAX_SERVER": url, "TRACKINIZER_PROFILE": "hosted"},
    )
    await _terminal_run(daemon, _opened(canvas=canvas, workspace=workspace))

    assert launch.call_args.kwargs["clean_env"] is True
    assert launch.call_args.kwargs["drop_env"] == ()
    lines = set(_result(socket)[0].decode().split())
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


async def _run_env(
    monkeypatch: pytest.MonkeyPatch,
    trax_env: dict[str, str],
    canvas: uuid.UUID,
) -> set[str]:
    """Return the real child's environment without spawning the process."""
    socket = _FakeSocket()
    _ = _connector(monkeypatch, socket)
    launch = _environment(monkeypatch)
    daemon = _terminal_serving("http://box.invalid", trax_env=trax_env)
    await _terminal_run(
        daemon,
        _opened(command="claude", run_name="run-1a2b3c4d", canvas=canvas),
    )
    assert launch.call_args.kwargs["clean_env"] is False
    return set(_result(socket)[0].decode().split())


async def test_a_run_inherits_the_daemon_environment_like_a_typed_knowop_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1/bus")
    monkeypatch.setenv("SENTINEL_FROM_DAEMON", "kept")
    canvas = uuid.uuid4()
    lines = await _run_env(
        monkeypatch,
        {"TRACKINIZER_PROFILE": "hosted"},
        canvas,
    )
    assert {
        "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1/bus",
        "SENTINEL_FROM_DAEMON=kept",
        "TRACKINIZER_PROFILE=hosted",
        f"TRAX_CANVAS={canvas}",
    } <= lines


async def test_the_canvases_trax_context_replaces_any_the_daemon_inherited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRACKINIZER_URL", "http://elsewhere.invalid")
    monkeypatch.setenv("TRACKINIZER_TOKEN", "sentinel-token")
    monkeypatch.setenv("TRACKINIZER_PROFILE", "elsewhere")
    monkeypatch.setenv("TRAX_ACTOR", "sentinel-actor")
    monkeypatch.setenv("LC_TRAX_CANVAS", "sentinel-canvas")
    canvas = uuid.uuid4()
    lines = await _run_env(
        monkeypatch,
        {"TRACKINIZER_PROFILE": "hosted"},
        canvas,
    )
    assert {
        "TRACKINIZER_PROFILE=hosted",
        f"LC_TRAX_CANVAS={canvas}",
    } <= lines
    assert not [line for line in lines if "sentinel" in line or "elsewhere" in line]


async def test_each_open_and_close_is_logged_once_for_this_user_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _connector(monkeypatch, _FakeSocket())
    _, launch = _terminal(monkeypatch, status=5)
    opened = _opened(requested_by="helper-1")
    await _terminal_run(
        _terminal_serving("http://box.invalid", "/bin/sh", "-c", "exit 5"),
        opened,
    )

    launch.assert_called_once()
    path = state_dir() / "rekursiv-ai" / "trax" / "hosts" / "box.terminals.log"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    first, second = path.read_text().splitlines()
    assert f'"event":"open","terminal_id":"{opened.terminal_id}"' in first
    assert '"opener":"ada@x.y"' in first
    assert '"requested_by":"helper-1"' in first
    assert '"event":"close"' in second
    assert '"status":5' in second


async def test_a_terminal_without_a_folder_starts_in_the_home_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``pwd -P`` prints the physical path, which a symlinked home does not share.
    home = await asyncio.to_thread(os.path.realpath, pwd.getpwuid(os.getuid()).pw_dir)
    socket = _FakeSocket()
    _ = _connector(monkeypatch, socket)
    _, launch = _terminal(monkeypatch, f"{home}\r\n".encode())

    await _terminal_run(
        _terminal_serving("http://box.invalid", "/bin/sh", "-c", "pwd -P"),
        _opened(),
    )

    assert _result(socket) == (f"{home}\r\n".encode(), 0)
    assert launch.call_args.kwargs["cwd"] == Path(pwd.getpwuid(os.getuid()).pw_dir)


async def test_a_run_starts_in_the_folder_it_names_and_says_so_on_its_first_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = tmp_path / "project"
    folder.mkdir()
    physical = await asyncio.to_thread(folder.resolve)
    socket = _FakeSocket()
    _ = _connector(monkeypatch, socket)
    _, launch = _terminal(monkeypatch, f"{physical}\r\n".encode())

    await _terminal_run(
        _terminal_serving("http://box.invalid"),
        _opened(command="claude", run_name="run-1a2b3c4d", cwd=str(folder)),
    )

    output, status = _result(socket)
    first, _, shell = output.partition(b"\r\n")
    assert status == 0
    assert first == f"knowop: started in {folder}".encode()
    assert shell.split() == [str(physical).encode()]
    assert launch.call_args.kwargs["cwd"] == folder


async def test_a_tilde_folder_is_resolved_against_the_home_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    (home / "project").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    physical = await asyncio.to_thread((home / "project").resolve)
    socket = _FakeSocket()
    _ = _connector(monkeypatch, socket)
    _, launch = _terminal(monkeypatch, f"{physical}\r\n".encode())

    await _terminal_run(
        _terminal_serving("http://box.invalid"),
        _opened(command="claude", run_name="run-1a2b3c4d", cwd="~/project"),
    )

    first, _, shell = _result(socket)[0].partition(b"\r\n")
    assert first == f"knowop: started in {home / 'project'}".encode()
    assert shell.split() == [str(physical).encode()]
    assert launch.call_args.kwargs["cwd"] == home / "project"


async def test_a_run_without_a_folder_says_it_starts_in_the_home_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _FakeSocket()
    _ = _connector(monkeypatch, socket)
    _, launch = _terminal(monkeypatch)

    await _terminal_run(
        _terminal_serving("http://box.invalid"),
        _opened(command="codex", run_name="run-1a2b3c4d"),
    )

    home = pwd.getpwuid(os.getuid()).pw_dir
    assert _result(socket) == (f"knowop: started in {home}\r\n".encode(), 0)
    assert launch.call_args.kwargs["cwd"] == Path(home)


async def test_a_folder_that_is_missing_or_a_file_ends_the_terminal_and_runs_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file = tmp_path / "notes.txt"
    _ = file.write_text("not a folder")
    refusals: list[tuple[bytes, int]] = []
    _, launch = _terminal(monkeypatch)

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
    launch.assert_not_called()

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
    path = state_dir() / "rekursiv-ai" / "trax" / "hosts" / "box.terminals.log"
    closes = [
        line for line in path.read_text().splitlines() if '"event":"close"' in line
    ]
    assert len(closes) == 3
    assert all('"status":1' in line for line in closes)


async def test_a_files_open_answers_requests_and_starts_no_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "a.txt").write_text("a")
    marker = tmp_path / "shell-ran"
    request = f'{{"id": 1, "op": "info", "path": "{tmp_path}"}}\n'
    socket = _FakeSocket(
        messages=(encode_frame(Frame.INPUT, request.encode()),),
        end_after_output=True,
    )
    _ = _connector(monkeypatch, socket)
    _, launch = _terminal(monkeypatch)

    await _terminal_run(
        _terminal_serving("http://box.invalid", "/bin/sh", "-c", f"touch {marker}"),
        _opened(kind="files"),
    )

    output, status = _result(socket)
    reply = Info.model_validate_json(output)
    assert (reply.id, reply.kind) == (1, "dir")
    assert reply.entries is not None
    assert reply.entries[0].name == "a.txt"
    assert status == 0
    assert not marker.exists()
    launch.assert_not_called()
    path = state_dir() / "rekursiv-ai" / "trax" / "hosts" / "box.terminals.log"
    assert '"kind":"files"' in path.read_text().splitlines()[0]


async def test_a_completed_files_service_reports_a_successful_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(FilesChannel, "serve", _finish_files)
    socket = _FakeSocket()
    _ = _connector(monkeypatch, socket)

    await _terminal_run(_terminal_serving("http://box.invalid"), _opened(kind="files"))

    assert _result(socket) == (b"finished\n", 0)
    assert socket.closed


async def test_a_failed_files_service_propagates_its_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(FilesChannel, "serve", _fail_files)
    socket = _FakeSocket()
    _ = _connector(monkeypatch, socket)

    with pytest.raises(RuntimeError, match="file service failed"):
        await _terminal_run(
            _terminal_serving("http://box.invalid"),
            _opened(kind="files"),
        )
    assert socket.closed


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


async def _five_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TerminalServing, _Client, list[_FakeSocket], list[_TerminalMock]]:
    """Open five independent fake sockets and terminals through the real daemon threads."""
    sockets = [_FakeSocket() for _ in range(5)]
    _ = _connector(monkeypatch, *sockets)
    terminals: list[_TerminalMock] = []
    for _ in range(5):
        finished, drained = asyncio.Event(), asyncio.Event()
        terminal, _ = _terminal(monkeypatch, status=143)
        terminal.output.return_value = _held_output(finished, drained)
        terminal.wait.side_effect = functools.partial(_terminal_status, finished)
        terminal.terminate.side_effect = finished.set
        terminals.append(terminal)
    monkeypatch.setattr(terminal_serving, "Terminal", Mock(side_effect=terminals))
    client = _Client([_opened() for _ in range(5)])
    daemon = _terminal_serving(
        "http://box.invalid",
        "/bin/sh",
        "-c",
        "sleep 30",
        client=client,
    )
    daemon.start(_HOST, _INSTANCE)
    for _ in range(500):
        await asyncio.sleep(0.002)
        if all(terminal.__aenter__.await_count == 1 for terminal in terminals):
            break
    return daemon, client, sockets, terminals


async def test_a_daemon_serves_five_terminals_at_once_and_refuses_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, _, sockets, terminals = await _five_opens(monkeypatch)

    open_now = [
        not socket.closed and terminal.__aenter__.await_count == 1
        for socket, terminal in zip(sockets, terminals, strict=True)
    ]
    await asyncio.to_thread(daemon.stop)

    assert open_now == [True] * 5


async def test_stop_ends_every_bridge_and_the_listener_lets_go_of_its_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, client, sockets, terminals = await _five_opens(monkeypatch)

    await asyncio.to_thread(daemon.stop)

    assert all(socket.closed for socket in sockets)
    for terminal in terminals:
        terminal.terminate.assert_awaited_once_with()
        terminal.wait.assert_awaited_once_with()
    # The listener's held request ends by itself, then it lets go of its client.
    assert await asyncio.to_thread(client.closed.wait, 10)
    path = state_dir() / "rekursiv-ai" / "trax" / "hosts" / "box.terminals.log"
    records = [
        _LogLine.model_validate_json(line) for line in path.read_text().splitlines()
    ]
    opened = [record.terminal_id for record in records if record.event == "open"]
    closed = [record.terminal_id for record in records if record.event == "close"]
    assert len(opened) == 5
    assert sorted(opened) == sorted(closed)


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
        await asyncio.sleep(0.01)
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


async def test_the_exit_is_reported_though_a_job_the_shell_left_still_holds_the_pty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _FakeSocket()
    _ = _connector(monkeypatch, socket)
    terminal, launch = _terminal(monkeypatch, status=4)
    drained = asyncio.Event()
    terminal.output.return_value = _held_output(asyncio.Event(), drained)
    daemon = _terminal_serving(
        "http://box.invalid",
        "/bin/sh",
        "-c",
        "setsid sleep 7.31 & exit 4",
        drain_sec=0.02,
    )

    await asyncio.wait_for(_terminal_run(daemon, _opened()), 3)

    assert _result(socket)[1] == 4
    assert not drained.is_set()
    assert launch.call_args.args == (["/bin/sh", "-c", "setsid sleep 7.31 & exit 4"],)
    terminal.terminate.assert_awaited_once_with()
    terminal.wait.assert_awaited_once_with()
    assert socket.closed


async def test_a_410_ends_the_listener_and_any_other_failure_is_retried() -> None:
    client = _Client([ClientError("-> 410", status_code=410)])
    daemon = _terminal_serving("http://127.0.0.1:1", client=client)

    daemon.start(_HOST, _INSTANCE)
    await asyncio.to_thread(client.closed.wait, 10)

    assert client.closed.is_set()


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
    """A wait the server holds until the test lets it go, as one a pause cut is."""

    def __init__(self) -> None:
        self.asks = 0
        self.asked = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()

    def wait_terminals(
        self,
        host_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        wait_sec: float,
    ) -> TerminalWait | None:
        del host_id, instance, wait_sec
        self.asks += 1
        self.asked.set()
        assert self.release.wait(10)
        return None

    def close(self) -> None:
        self.closed.set()


async def test_a_wake_from_a_pause_starts_a_new_wait_and_the_dead_one_ends_after() -> (
    None
):
    dead, fresh = _Held(), _Held()
    clients = [dead, fresh]
    wall = [0.0]
    wake = Wake(wall=lambda: wall[0], clock=lambda: 0.0)
    daemon = TerminalServing(
        name="box",
        server_url="http://127.0.0.1:1",
        credential=_CREDENTIAL,
        credential_client=lambda _key: clients.pop(0),
        trax_env={},
        wake=wake,
    )
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

    assert (dead.asks, clients) == (1, [])


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
    tmp_path: Path,
) -> None:
    socket = _ClosingSocket()
    _ = _connector(monkeypatch, socket)

    await _terminal_run(
        _terminal_serving(
            "http://box.invalid",
            "/bin/sh",
            "-c",
            f"touch {tmp_path / 'ran'}",
        ),
        _opened(),
    )

    assert socket.closed
    assert not (tmp_path / "ran").exists()
    assert '"event":"close"' in _startup_log()
    assert '"status":1' in _startup_log()


async def test_the_shells_output_stops_when_the_relay_has_closed() -> None:
    """Whatever the shell still writes has nowhere to go; the pump just ends."""
    router = DirectRouter(relay=_relay_gone, shell=_Shell(), factory=None)

    await _pump(cast("Terminal", _Output(b"late")), router)


@pytest.mark.parametrize(
    ("messages", "closes"),
    [(("text, not a frame", _INPUT), False), ((b"\xff", _INPUT), False), ((), True)],
    ids=["text", "no frame", "closed"],
)
async def test_the_server_socket_is_read_until_it_sends_what_is_no_frame_or_closes(
    messages: tuple[str | bytes, ...],
    closes: bool,
) -> None:
    shell = _Shell()
    router = DirectRouter(relay=_relay_gone, shell=shell, factory=None)

    await _read(router, cast("ClientConnection", _Messages(*messages, closes=closes)))

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
    terminal, _ = _terminal(monkeypatch)
    typed, finished = asyncio.Event(), asyncio.Event()
    terminal.write.side_effect = functools.partial(_typed, typed)
    terminal.output.return_value = _output_when(typed, finished, b"got-go\r\n")
    terminal.wait.side_effect = functools.partial(_terminal_status, finished, status=0)
    daemon = _terminal_serving(
        "http://box.invalid",
        "/bin/sh",
        "-c",
        "read x; echo got-$x",
        direct=lambda: peer,
    )

    await _terminal_run(daemon, _opened(ice_servers=["stun:s.test:3478"]))

    terminal.write.assert_awaited_once_with(b"go\n")
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


def _in_opt_bin(name: str) -> str | None:
    """Stand in for ``shutil.which`` on a computer with every command in /opt/bin."""
    return f"/opt/bin/{name}"


def _login_shell() -> list[str]:
    return [pwd.getpwuid(os.getuid()).pw_shell, "-l"]


_KNOWOP = ("-m", "knowop", "run")
"""What follows the interpreter in a run's command line: this install's ``knowop run``."""


async def test_a_claude_run_starts_detached_and_attaches_its_chat_session() -> None:
    session_id = uuid.UUID("9f1c2e3a-0000-4000-8000-000000000001")

    argv = _terminal_serving("http://127.0.0.1:1")._argv(
        _opened(command="claude", attach=session_id, run_name="chat-9f1c2e3a"),
    )

    assert argv == [
        sys.executable,
        *_KNOWOP,
        "claude",
        "--detach",
        "--name",
        "chat-9f1c2e3a",
        "--attach",
        str(session_id),
    ]


async def test_a_codex_run_without_a_session_attaches_nothing() -> None:
    argv = _terminal_serving("http://127.0.0.1:1")._argv(
        _opened(command="codex", run_name="run-1a2b3c4d"),
    )

    assert argv == [
        sys.executable,
        *_KNOWOP,
        "codex",
        "--detach",
        "--name",
        "run-1a2b3c4d",
    ]


async def test_a_run_is_started_by_this_install_whatever_path_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Another knowop on ``PATH``, or a ``trax`` that forwards, never runs it."""
    monkeypatch.setattr(shutil, "which", _in_opt_bin)

    argv = _terminal_serving("http://127.0.0.1:1")._argv(
        _opened(command="claude", run_name="run-1a2b3c4d"),
    )

    assert argv[:5] == [sys.executable, *_KNOWOP, "claude"]


async def test_a_terminal_without_a_command_runs_the_login_shell() -> None:
    assert _terminal_serving("http://127.0.0.1:1")._argv(_opened()) == _login_shell()
    shell = _terminal_serving("http://127.0.0.1:1", "/bin/sh", "-c", "true")._argv(
        _opened(),
    )
    assert shell == ["/bin/sh", "-c", "true"]


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


async def test_connect_sends_the_credential_and_the_timeouts_a_socket_needs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _FakeSocket()
    connector = _connector(monkeypatch, socket)
    assert await _connect() is socket
    (kwargs,) = connector.calls
    assert kwargs["additional_headers"] == {"Authorization": f"Bearer {_CREDENTIAL}"}
    assert kwargs["ping_interval"] == 20
    assert kwargs["close_timeout"] == 1
    assert kwargs["open_timeout"] == 10.0


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
    assert await _connect() is not None
    assert pauses == [1.0]


async def test_connect_starts_an_attempt_with_under_a_second_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = _FakeSocket()
    _ = _connector(monkeypatch, socket)
    assert await _connect(startup_timeout_sec=0.5) is socket


async def test_connect_without_time_never_dials_and_says_the_deadline_passed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    connector = _connector(monkeypatch, _FakeSocket())
    # A frozen clock makes the time left exactly zero, which is no time at all.
    now = asyncio.get_running_loop().time()
    monkeypatch.setattr(asyncio.get_running_loop(), "time", lambda: now)
    assert await _connect(startup_timeout_sec=0) is None
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
    assert await _connect() is None
    assert len(connector.calls) == 1
    assert socket.closed
    assert "): invalid startup acknowledgement" in capsys.readouterr().err
    assert '"event":"startup_failed"' in _startup_log()


async def test_connect_names_the_error_that_ended_the_attempts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    startup_clock: _StartupClock,
) -> None:
    _ = _connector(monkeypatch, OSError("boom"))
    assert await _connect(startup_timeout_sec=0.03, retry_sec=0.005) is None
    assert capsys.readouterr().err.endswith("): boom\n")
    assert sum(startup_clock.pauses) == pytest.approx(0.03)


async def test_connect_names_an_error_without_a_message_by_its_type(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    startup_clock: _StartupClock,
) -> None:
    _ = _connector(monkeypatch, OSError())
    assert await _connect(startup_timeout_sec=0.03, retry_sec=0.005) is None
    assert capsys.readouterr().err.endswith("): OSError\n")
    assert sum(startup_clock.pauses) == pytest.approx(0.03)


async def test_connect_logs_each_retry_with_the_terminal_and_reason(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _ = _connector(monkeypatch, OSError("boom"), _FakeSocket())
    caplog.set_level(logging.DEBUG)
    opened = _opened()
    _ = await _connect(opened, retry_sec=0.001)
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
    assert await _connect(startup_timeout_sec=0.05, retry_sec=0.002) is None
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
    assert result is None
    assert startup_clock.pauses == [pytest.approx(0.05)]


async def test_connect_closes_a_socket_that_fails_during_the_handshake(
    monkeypatch: pytest.MonkeyPatch,
    startup_clock: _StartupClock,
) -> None:
    socket = _FakeSocket(send_error=ConnectionClosed(None, None))
    _ = _connector(monkeypatch, socket)
    assert await _connect(startup_timeout_sec=0.03, retry_sec=0.005) is None
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


class _ClosingSocket(_FakeSocket):
    """A socket that takes the startup handshake, then finds every later send closed."""

    def __init__(self) -> None:
        super().__init__()
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


async def _relay_gone(data: bytes) -> None:
    """Send to a relay the server has already closed."""
    del data
    raise ConnectionClosed(None, None)


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
