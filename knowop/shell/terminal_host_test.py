"""The terminal host: real shells on real ptys, bridged to a local WebSocket server."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self, cast, override

import asyncio
import functools
import logging
import os
import pwd
import shutil
import stat
import subprocess
import sys
import threading
import uuid

from trackinizer.client.errors import ClientError
from trackinizer.lib.posix.host import Frame, decode_frame, encode_frame, encode_winsize
from trackinizer.lib.userdirs import state_dir
from trackinizer.wire import wire_terminals
from trackinizer.wire.wire_terminals import TerminalWait
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.http11 import Response

import pytest

from knowop.shell import terminal_host
from knowop.shell.terminal_direct import DirectRouter
from knowop.shell.terminal_host import (
    TerminalHost,
    _end_with_line,
    _LogLine,
    _pump,
    _read,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
    from pathlib import Path

    from trackinizer.lib.posix.terminal import Terminal
    from websockets.asyncio.client import ClientConnection

    from knowop.shell.terminal_direct import PeerFactory

pytestmark = pytest.mark.asyncio

_MACHINE = uuid.UUID(int=1)
_INSTANCE = uuid.UUID(int=2)
_CREDENTIAL = "trax_machine_" + "0" * 32 + "_" + "c" * 43
_INPUT = encode_frame(Frame.INPUT, b"typed\n")

type _Handler = Callable[[ServerConnection], Awaitable[None]]


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
        assert (machine_id, instance, wait_sec) == (_MACHINE, _INSTANCE, 25)
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


def _host(
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
        client_for=lambda _key: client or _Client([]),
        trax_env=trax_env or {},
        shell=shell,
        drain_sec=drain_sec,
        direct=direct,
    )


async def _listening(handler: _Handler) -> tuple[Server, str]:
    server = await serve(functools.partial(_ready_handler, handler), "127.0.0.1", 0)
    return server, f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"


async def _ready_handler(handler: _Handler, ws: ServerConnection) -> None:
    assert await ws.recv() == encode_frame(Frame.ACK)
    await ws.send(encode_frame(Frame.ACK))
    assert await ws.recv() == encode_frame(Frame.ACK)
    await handler(ws)


async def _until_exit(ws: ServerConnection) -> tuple[bytes, int]:
    """Collect output until the shell's exit frame; return both."""
    output = b""
    async for message in ws:
        assert isinstance(message, bytes)
        kind, payload = decode_frame(message)
        if kind is Frame.OUTPUT:
            output += payload
        elif kind is Frame.EXIT:
            return output, int.from_bytes(payload)
    raise AssertionError("closed before the shell exited")


async def _first_line(ws: ServerConnection) -> int:
    """Collect output to its first newline, which may span frames; parse it."""
    output = b""
    async for message in ws:
        assert isinstance(message, bytes)
        kind, payload = decode_frame(message)
        if kind is Frame.OUTPUT:
            output += payload
        if b"\n" in output:
            return int(output.split()[0])
    raise AssertionError("closed before the shell wrote a line")


async def _run(host: TerminalHost, opened: TerminalWait) -> None:
    await asyncio.to_thread(
        host.serve,
        opened,
        machine_id=_MACHINE,
        instance=_INSTANCE,
    )


async def test_a_shell_runs_on_a_real_pty_and_its_output_and_exit_status_arrive() -> (
    None
):
    seen: dict[str, object] = {}

    async def handler(ws: ServerConnection) -> None:
        assert ws.request is not None
        seen["path"] = ws.request.path
        seen["auth"] = ws.request.headers["Authorization"]
        seen["result"] = await _until_exit(ws)

    server, url = await _listening(handler)
    opened = _opened()
    await _run(_host(url, "/bin/sh", "-c", "printf ok; exit 3"), opened)
    server.close()

    path = wire_terminals.MACHINE_TERMINAL_WS_PATH.format(
        machine_id=_MACHINE,
        terminal_id=opened.terminal_id,
    )
    assert seen["path"] == f"{path}?instance={_INSTANCE}&ready=true"
    assert seen["auth"] == f"Bearer {_CREDENTIAL}"
    assert seen["result"] == (b"ok", 3)


async def test_startup_retries_handshake_timeouts_and_launches_the_cli_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        TerminalHost,
        "_connect",
        functools.partialmethod(
            TerminalHost._connect,
            open_timeout_sec=0.02,
            retry_sec=0.001,
        ),
    )
    launches = tmp_path / "launches"
    trax = _fake_knowop(tmp_path, f"echo started >> '{launches}'\nprintf ready\n")
    _runs_as_knowop(monkeypatch, trax)
    attempts: list[str] = []
    output: list[tuple[bytes, int]] = []

    async def handshake(ws: ServerConnection) -> None:
        assert await ws.recv() == encode_frame(Frame.ACK)
        attempts.append("connected")
        if len(attempts) < 3:
            await ws.wait_closed()
            return
        await ws.send(encode_frame(Frame.ACK))
        assert await ws.recv() == encode_frame(Frame.ACK)
        output.append(await _until_exit(ws))

    server = await serve(handshake, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    try:
        await asyncio.wait_for(
            _run(_host(url), _opened(command="claude", run_name="chat-retry")),
            3,
        )
    finally:
        server.close()
        await server.wait_closed()

    assert attempts == ["connected"] * 3
    assert launches.read_text().splitlines() == ["started"]
    assert output[0][0].endswith(b"ready")
    assert output[0][1] == 0


async def test_a_refused_startup_never_launches_the_cli_and_reports_one_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    launches = tmp_path / "launches"
    trax = _fake_knowop(tmp_path, f"echo started >> '{launches}'\n")
    _runs_as_knowop(monkeypatch, trax)

    async def refuse(ws: ServerConnection) -> None:
        await ws.close(wire_terminals.CLOSE_REFUSED)

    server = await serve(refuse, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    try:
        await _run(_host(url), _opened(command="claude", run_name="chat-refused"))
    finally:
        server.close()
        await server.wait_closed()

    assert not launches.exists()
    error = capsys.readouterr().err
    assert "terminal startup failed" in error
    assert "Traceback" not in error


async def test_stop_cancels_startup_while_waiting_for_the_server_acknowledgement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launches = tmp_path / "launches"
    trax = _fake_knowop(tmp_path, f"echo started >> '{launches}'\n")
    _runs_as_knowop(monkeypatch, trax)
    arrived = asyncio.Event()
    requests: list[str | bytes] = []

    async def stall(ws: ServerConnection) -> None:
        requests.append(await ws.recv())
        arrived.set()
        await ws.wait_closed()

    server = await serve(stall, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    host = _host(url)
    running = asyncio.create_task(
        _run(host, _opened(command="claude", run_name="chat-stopped")),
    )
    try:
        await asyncio.wait_for(arrived.wait(), 2)
        await asyncio.to_thread(host.stop)
        await asyncio.wait_for(running, 1)
    finally:
        server.close()
        await server.wait_closed()

    assert requests == [encode_frame(Frame.ACK)]
    assert not launches.exists()


async def test_input_reaches_the_shell_and_a_resize_is_clamped() -> None:
    result: list[tuple[bytes, int]] = []

    async def handler(ws: ServerConnection) -> None:
        await ws.send(encode_frame(Frame.RESIZE, encode_winsize(0, 5_000)))
        await ws.send(encode_frame(Frame.INPUT, b"go\n"))
        result.append(await _until_exit(ws))

    server, url = await _listening(handler)
    await _run(_host(url, "/bin/sh", "-c", "read x; stty size; echo got-$x"), _opened())
    server.close()

    output, status = result[0]
    assert status == 0
    assert b"1 1000" in output
    assert b"got-go" in output


async def test_stop_and_a_closed_socket_each_kill_the_shells_whole_group() -> None:
    pids: list[int] = []

    # The pid and its newline are written apart, as a pty may deliver them anyway.
    pid_line = "printf %s $$; sleep 0.02; echo; sleep 30"

    async def stopper(ws: ServerConnection) -> None:
        pids.append(await _first_line(ws))
        await ws.send(encode_frame(Frame.STOP))
        _, status = await _until_exit(ws)
        pids.append(status)

    server, url = await _listening(stopper)
    await _run(_host(url, "/bin/sh", "-c", pid_line), _opened())
    server.close()
    pid, status = pids
    assert status > 128
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)

    gone: list[int] = []

    async def leaver(ws: ServerConnection) -> None:
        gone.append(await _first_line(ws))
        await ws.close()

    server, url = await _listening(leaver)
    await _run(_host(url, "/bin/sh", "-c", pid_line), _opened())
    server.close()
    with pytest.raises(ProcessLookupError):
        os.kill(gone[0], 0)


async def test_the_shell_starts_clean_with_the_trax_context_and_no_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sentinel-secret")
    monkeypatch.setenv("TRACKINIZER_TOKEN", "sentinel-token")
    result: list[tuple[bytes, int]] = []

    async def handler(ws: ServerConnection) -> None:
        result.append(await _until_exit(ws))

    server, url = await _listening(handler)
    canvas, workspace = uuid.uuid4(), uuid.uuid4()
    host = _host(
        url,
        "/bin/sh",
        "-c",
        "env",
        trax_env={"LC_TRAX_SERVER": url, "TRACKINIZER_PROFILE": "hosted"},
    )
    await _run(host, _opened(canvas=canvas, workspace=workspace))
    server.close()

    lines = set(result[0][0].decode().split())
    assert {
        "TERM=xterm-256color",
        "COLORTERM=truecolor",
        f"LC_TRAX_SERVER={url}",
        "TRACKINIZER_PROFILE=hosted",
        f"TRAX_CANVAS={canvas}",
        f"LC_TRAX_CANVAS={canvas}",
        f"TRAX_WORKSPACE={workspace}",
        f"LC_TRAX_WORKSPACE={workspace}",
    } <= lines
    assert not [line for line in lines if "sentinel" in line]


async def _run_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trax_env: dict[str, str],
    canvas: uuid.UUID,
) -> set[str]:
    """Return the environment a run dumps; the fake knowop is ``env``."""
    trax = _fake_knowop(tmp_path, "env\n")
    _runs_as_knowop(monkeypatch, trax)
    result: list[tuple[bytes, int]] = []

    async def handler(ws: ServerConnection) -> None:
        result.append(await _until_exit(ws))

    server, url = await _listening(handler)
    host = _host(url, trax_env=trax_env)
    await _run(
        host,
        _opened(command="claude", run_name="run-1a2b3c4d", canvas=canvas),
    )
    server.close()
    return set(result[0][0].decode().split())


async def test_a_run_inherits_the_host_environment_like_a_typed_knowop_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1/bus")
    monkeypatch.setenv("SENTINEL_FROM_HOST", "kept")
    canvas = uuid.uuid4()
    lines = await _run_env(
        tmp_path,
        monkeypatch,
        {"TRACKINIZER_PROFILE": "hosted"},
        canvas,
    )
    assert {
        "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1/bus",
        "SENTINEL_FROM_HOST=kept",
        "TRACKINIZER_PROFILE=hosted",
        f"TRAX_CANVAS={canvas}",
    } <= lines


async def test_the_canvases_trax_context_replaces_any_the_host_inherited(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRACKINIZER_URL", "http://elsewhere.invalid")
    monkeypatch.setenv("TRACKINIZER_TOKEN", "sentinel-token")
    monkeypatch.setenv("TRACKINIZER_PROFILE", "elsewhere")
    monkeypatch.setenv("TRAX_ACTOR", "sentinel-actor")
    monkeypatch.setenv("LC_TRAX_CANVAS", "sentinel-canvas")
    canvas = uuid.uuid4()
    lines = await _run_env(
        tmp_path,
        monkeypatch,
        {"TRACKINIZER_PROFILE": "hosted"},
        canvas,
    )
    assert {
        "TRACKINIZER_PROFILE=hosted",
        f"LC_TRAX_CANVAS={canvas}",
    } <= lines
    assert not [line for line in lines if "sentinel" in line or "elsewhere" in line]


async def test_each_open_and_close_is_logged_once_for_this_user_only() -> None:
    async def handler(ws: ServerConnection) -> None:
        await _until_exit(ws)

    server, url = await _listening(handler)
    opened = _opened(requested_by="helper-1")
    await _run(_host(url, "/bin/sh", "-c", "exit 5"), opened)
    server.close()

    path = state_dir() / "rekursiv-ai" / "trax" / "machines" / "box.terminals.log"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    first, second = path.read_text().splitlines()
    assert f'"event":"open","terminal_id":"{opened.terminal_id}"' in first
    assert '"opener":"ada@x.y"' in first
    assert '"requested_by":"helper-1"' in first
    assert '"event":"close"' in second
    assert '"status":5' in second


async def test_a_terminal_without_a_folder_starts_in_the_home_directory() -> None:
    result: list[tuple[bytes, int]] = []

    async def handler(ws: ServerConnection) -> None:
        result.append(await _until_exit(ws))

    server, url = await _listening(handler)
    await _run(_host(url, "/bin/sh", "-c", "pwd -P"), _opened())
    server.close()

    # ``pwd -P`` prints the physical path, which a symlinked home does not share.
    home = await asyncio.to_thread(os.path.realpath, pwd.getpwuid(os.getuid()).pw_dir)
    assert result[0] == (f"{home}\r\n".encode(), 0)


async def test_a_run_starts_in_the_folder_it_names_and_says_so_on_its_first_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = tmp_path / "project"
    folder.mkdir()
    # The fake knowop prints the folder its run starts in, as the CLI would see it.
    trax = _fake_knowop(tmp_path, "pwd -P\n")
    _runs_as_knowop(monkeypatch, trax)
    result: list[tuple[bytes, int]] = []

    async def handler(ws: ServerConnection) -> None:
        result.append(await _until_exit(ws))

    server, url = await _listening(handler)
    await _run(
        _host(url),
        _opened(command="claude", run_name="run-1a2b3c4d", cwd=str(folder)),
    )
    server.close()

    output, status = result[0]
    first, _, shell = output.partition(b"\r\n")
    physical = await asyncio.to_thread(folder.resolve)
    assert status == 0
    assert first == f"knowop: started in {folder}".encode()
    assert shell.split() == [str(physical).encode()]


async def test_a_tilde_folder_is_resolved_against_the_home_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    (home / "project").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    trax = _fake_knowop(tmp_path, "pwd -P\n")
    _runs_as_knowop(monkeypatch, trax)
    result: list[tuple[bytes, int]] = []

    async def handler(ws: ServerConnection) -> None:
        result.append(await _until_exit(ws))

    server, url = await _listening(handler)
    await _run(
        _host(url),
        _opened(command="claude", run_name="run-1a2b3c4d", cwd="~/project"),
    )
    server.close()

    first, _, shell = result[0][0].partition(b"\r\n")
    physical = await asyncio.to_thread((home / "project").resolve)
    assert first == f"knowop: started in {home / 'project'}".encode()
    assert shell.split() == [str(physical).encode()]


async def test_a_run_without_a_folder_says_it_starts_in_the_home_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trax = _fake_knowop(tmp_path)
    _runs_as_knowop(monkeypatch, trax)
    result: list[tuple[bytes, int]] = []

    async def handler(ws: ServerConnection) -> None:
        result.append(await _until_exit(ws))

    server, url = await _listening(handler)
    await _run(_host(url), _opened(command="codex", run_name="run-1a2b3c4d"))
    server.close()

    home = pwd.getpwuid(os.getuid()).pw_dir
    assert result[0] == (f"knowop: started in {home}\r\n".encode(), 0)


async def test_a_folder_that_is_missing_or_a_file_ends_the_terminal_and_runs_nothing(
    tmp_path: Path,
) -> None:
    file = tmp_path / "notes.txt"
    _ = file.write_text("not a folder")
    refusals: list[tuple[bytes, int]] = []

    async def handler(ws: ServerConnection) -> None:
        refusals.append(await _until_exit(ws))

    # A ``~name`` of a user this machine lacks has no home to resolve against.
    stranger = "~no-such-user-c8/project"
    server, url = await _listening(handler)
    for folder in (str(tmp_path / "gone"), str(file), stranger):
        await _run(
            _host(url, "/bin/sh", "-c", "echo ran"),
            _opened(cwd=folder),
        )
    server.close()

    assert refusals == [
        (
            f"knowop: {tmp_path / 'gone'} is not a folder on this machine\r\n".encode(),
            1,
        ),
        (f"knowop: {file} is not a folder on this machine\r\n".encode(), 1),
        (f"knowop: {stranger} is not a folder on this machine\r\n".encode(), 1),
    ]
    # The refusal is logged as a close with the status it reported, so the log pairs
    # every open with its close as it does for a shell.
    path = state_dir() / "rekursiv-ai" / "trax" / "machines" / "box.terminals.log"
    closes = [
        line for line in path.read_text().splitlines() if '"event":"close"' in line
    ]
    assert len(closes) == 3
    assert all('"status":1' in line for line in closes)


class _Leg:
    """A WebSocket leg a refusal writes to; ``gone`` makes every send find it closed."""

    def __init__(self, *, gone: bool = False) -> None:
        self.gone = gone
        self.sent: list[bytes] = []

    async def send(self, message: bytes) -> None:
        if self.gone:
            raise ConnectionClosed(None, None)
        self.sent.append(message)


async def test_a_refusal_sends_its_line_then_exit_one_to_a_leg_still_open() -> None:
    leg = _Leg()

    await _end_with_line(leg, "knowop: x is not a folder on this machine")

    assert leg.sent == [
        encode_frame(Frame.OUTPUT, b"knowop: x is not a folder on this machine\r\n"),
        encode_frame(Frame.EXIT, (1).to_bytes(4)),
    ]


async def test_a_refusal_to_a_leg_already_closed_is_not_an_error() -> None:
    await _end_with_line(
        _Leg(gone=True),
        "knowop: x is not a folder on this machine",
    )


async def _five_opens(
    wait_for: Callable[[list[ServerConnection]], bool],
) -> tuple[TerminalHost, _Client, list[ServerConnection], Server]:
    """Start a host that is asked for five terminals; return once ``wait_for`` is true."""
    conns: list[ServerConnection] = []
    arrived = asyncio.Event()

    async def handler(ws: ServerConnection) -> None:
        conns.append(ws)
        arrived.set()
        await ws.wait_closed()

    server, url = await _listening(handler)
    client = _Client([_opened() for _ in range(5)])
    host = _host(url, "/bin/sh", "-c", "sleep 30", client=client)
    host.start(_MACHINE, _INSTANCE)
    for _ in range(500):
        await asyncio.sleep(0.002)
        if wait_for(conns):
            break
    return host, client, conns, server


async def test_a_host_serves_five_terminals_at_once_and_refuses_none() -> None:
    host, _, conns, server = await _five_opens(lambda conns: len(conns) == 5)

    open_now = [ws.close_code is None for ws in conns]
    await asyncio.to_thread(host.stop)
    server.close()

    assert open_now == [True] * 5


async def test_stop_ends_every_bridge_and_the_listener_lets_go_of_its_client() -> None:
    host, client, conns, server = await _five_opens(
        lambda conns: len(conns) == 5,
    )

    await asyncio.to_thread(host.stop)
    server.close()

    assert all(ws.close_code is not None for ws in conns)
    # The listener's held request ends by itself, then it lets go of its client.
    assert await asyncio.to_thread(client.closed.wait, 10)
    path = state_dir() / "rekursiv-ai" / "trax" / "machines" / "box.terminals.log"
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
    host = _host("http://127.0.0.1:1")
    opening = asyncio.create_task(
        asyncio.to_thread(
            host._start_bridge,
            _opened(),
            machine_id=_MACHINE,
            instance=_INSTANCE,
        ),
    )
    try:
        assert await asyncio.to_thread(arrived.wait, 2)
        stopping = asyncio.create_task(asyncio.to_thread(host.stop))
        await asyncio.sleep(0.01)
        resume.set()
        await asyncio.wait_for(asyncio.gather(opening, stopping), 2)
    finally:
        resume.set()
        await opening
        await asyncio.to_thread(host.stop)


async def test_a_closed_loop_in_the_shutdown_snapshot_does_not_skip_other_bridges() -> (
    None
):
    host = _host("http://127.0.0.1:1")
    closed = asyncio.new_event_loop()
    closed.close()
    ended = asyncio.Event()
    host._loops[uuid.uuid4()] = (closed, asyncio.Event())
    host._loops[uuid.uuid4()] = (asyncio.get_running_loop(), ended)

    await asyncio.to_thread(host.stop)
    await asyncio.wait_for(ended.wait(), 1)


async def test_the_exit_is_reported_though_a_job_the_shell_left_still_holds_the_pty() -> (
    None
):
    result: list[tuple[bytes, int]] = []

    async def handler(ws: ServerConnection) -> None:
        result.append(await _until_exit(ws))

    server, url = await _listening(handler)
    # ``setsid`` puts the job in a group of its own, as a job of an interactive shell
    # is, so it holds the pty open after the shell is gone.
    host = _host(
        url,
        "/bin/sh",
        "-c",
        "setsid sleep 7.31 & exit 4",
        drain_sec=0.02,
    )
    try:
        await asyncio.wait_for(_run(host, _opened()), 3)
    finally:
        server.close()
        await asyncio.to_thread(
            subprocess.run,
            ["pkill", "-f", "^sleep 7[.]31$"],
            check=False,
        )

    assert result[0][1] == 4


async def test_a_410_ends_the_listener_and_any_other_failure_is_retried() -> None:
    client = _Client([ClientError("-> 410", status_code=410)])
    host = _host("http://127.0.0.1:1", client=client)

    host.start(_MACHINE, _INSTANCE)
    await asyncio.to_thread(client.closed.wait, 10)

    assert client.closed.is_set()


async def test_a_wait_that_fails_otherwise_is_said_and_waited_out_until_stopped(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _Client([ClientError("-> 503", status_code=503)])
    host = _host("http://127.0.0.1:1", client=client)

    host.start(_MACHINE, _INSTANCE)
    assert await asyncio.to_thread(client.failed.wait, 10)
    host.stop()
    assert await asyncio.to_thread(client.closed.wait, 10)

    assert "knowop: terminal wait failed: -> 503" in capsys.readouterr().err


async def test_an_open_that_arrives_once_the_host_stopped_starts_no_bridge() -> None:
    host = _host("http://127.0.0.1:1")
    opened = _opened()
    host.stop()

    host._start_bridge(opened, machine_id=_MACHINE, instance=_INSTANCE)

    assert f"terminal-{opened.terminal_id}" not in {
        thread.name for thread in threading.enumerate()
    }


async def test_a_leg_closed_before_the_shell_starts_runs_nothing_and_logs_a_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    leg = _ClosingSocket()
    _ = _connector(monkeypatch, leg)

    await _run(
        _host("http://box.invalid", "/bin/sh", "-c", f"touch {tmp_path / 'ran'}"),
        _opened(),
    )

    assert leg.closed
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
async def test_the_server_leg_is_read_until_it_sends_what_is_no_frame_or_closes(
    messages: tuple[str | bytes, ...],
    closes: bool,
) -> None:
    shell = _Shell()
    router = DirectRouter(relay=_relay_gone, shell=shell, factory=None)

    await _read(router, cast("ClientConnection", _Messages(*messages, closes=closes)))

    assert shell.written == []


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
        loop.call_later(0.05, on_message, encode_frame(Frame.ROUTE, route))
        loop.call_later(0.1, on_message, encode_frame(Frame.INPUT, b"go\n"))
        return "answer-sdp"

    def send(self, data: bytes) -> None:
        self.sent.append(decode_frame(data))

    async def close(self) -> None:
        self.is_open = False


async def test_an_offer_is_answered_and_the_shells_output_then_goes_direct() -> None:
    peer = _ScriptedPeer()
    relayed: list[tuple[Frame, bytes]] = []

    async def handler(ws: ServerConnection) -> None:
        offer = wire_terminals.DirectSignal(gen=1, type="offer", sdp="offer-sdp")
        await ws.send(encode_frame(Frame.SIGNAL, offer.model_dump_json().encode()))
        async for message in ws:
            assert isinstance(message, bytes)
            relayed.append(decode_frame(message))
            if relayed[-1][0] is Frame.EXIT:
                return

    server, url = await _listening(handler)
    host = _host(url, "/bin/sh", "-c", "read x; echo got-$x", direct=lambda: peer)
    await _run(host, _opened(ice_servers=["stun:s.test:3478"]))
    server.close()

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


def _fake_knowop(tmp_path: Path, body: str = "") -> Path:
    trax = tmp_path / "trax"
    _ = trax.write_text(f"#!/bin/sh\n{body}")
    trax.chmod(0o755)
    return trax


def _runs_as_knowop(monkeypatch: pytest.MonkeyPatch, trax: Path) -> None:
    """Make the given script the interpreter a run starts, so it stands in for knowop."""
    monkeypatch.setattr(sys, "executable", str(trax))


def _in_opt_bin(name: str) -> str | None:
    """Stand in for ``shutil.which`` on a computer with every command in /opt/bin."""
    return f"/opt/bin/{name}"


def _login_shell() -> list[str]:
    return [pwd.getpwuid(os.getuid()).pw_shell, "-l"]


_KNOWOP = ("-m", "knowop", "run")
"""What follows the interpreter in a run's command line: this install's ``knowop run``."""


async def test_a_claude_run_starts_detached_and_attaches_its_chat_session() -> None:
    session_id = uuid.UUID("9f1c2e3a-0000-4000-8000-000000000001")

    argv = _host("http://127.0.0.1:1")._argv(
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
    argv = _host("http://127.0.0.1:1")._argv(
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

    argv = _host("http://127.0.0.1:1")._argv(
        _opened(command="claude", run_name="run-1a2b3c4d"),
    )

    assert argv[:5] == [sys.executable, *_KNOWOP, "claude"]


async def test_a_terminal_without_a_command_runs_the_login_shell() -> None:
    assert _host("http://127.0.0.1:1")._argv(_opened()) == _login_shell()
    shell = _host("http://127.0.0.1:1", "/bin/sh", "-c", "true")._argv(_opened())
    assert shell == ["/bin/sh", "-c", "true"]


class _FakeSocket:
    """A leg that answers the startup handshake as scripted."""

    def __init__(
        self,
        *,
        reply: bytes = encode_frame(Frame.ACK),
        send_error: ConnectionClosed | None = None,
        hang: bool = False,
    ) -> None:
        self.reply = reply
        self.send_error = send_error
        self.hang = hang
        self.closed = False
        self.waiting = asyncio.Event()

    async def send(self, data: bytes) -> None:
        del data
        if self.send_error is not None:
            raise self.send_error

    async def recv(self) -> bytes:
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

    async def __call__(self, url: str, **kwargs: object) -> _FakeSocket:
        del url
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
    return await _host("http://box.invalid")._connect(
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


async def test_connect_sends_the_credential_and_the_timeouts_a_leg_needs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leg = _FakeSocket()
    connector = _connector(monkeypatch, leg)
    assert await _connect() is leg
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
    leg = _FakeSocket()
    _ = _connector(monkeypatch, leg)
    assert await _connect(startup_timeout_sec=0.5) is leg


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


async def test_connect_stops_on_a_wrong_acknowledgement_and_closes_the_leg(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    leg = _FakeSocket(reply=b"nope")
    connector = _connector(monkeypatch, leg)
    assert await _connect() is None
    assert len(connector.calls) == 1
    assert leg.closed
    assert "): invalid startup acknowledgement" in capsys.readouterr().err
    assert '"event":"startup_failed"' in _startup_log()


async def test_connect_names_the_error_that_ended_the_attempts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ = _connector(monkeypatch, OSError("boom"))
    assert await _connect(startup_timeout_sec=0.03, retry_sec=0.005) is None
    assert capsys.readouterr().err.endswith("): boom\n")


async def test_connect_names_an_error_without_a_message_by_its_type(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ = _connector(monkeypatch, OSError())
    assert await _connect(startup_timeout_sec=0.03, retry_sec=0.005) is None
    assert capsys.readouterr().err.endswith("): OSError\n")


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
) -> None:
    connector = _connector(monkeypatch, _status(code))
    assert await _connect(startup_timeout_sec=0.05, retry_sec=0.002) is None
    assert (len(connector.calls) > 2) is retried
    if not retried:
        assert len(connector.calls) == 1
    assert "knowop: terminal startup failed" in capsys.readouterr().err
    assert '"event":"startup_failed"' in _startup_log()


async def test_connect_never_sleeps_past_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _connector(monkeypatch, OSError("down"))
    result = await asyncio.wait_for(
        _connect(startup_timeout_sec=0.05, retry_sec=5.0),
        1,
    )
    assert result is None


async def test_connect_closes_a_leg_that_fails_during_the_handshake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leg = _FakeSocket(send_error=ConnectionClosed(None, None))
    _ = _connector(monkeypatch, leg)
    assert await _connect(startup_timeout_sec=0.03, retry_sec=0.005) is None
    assert leg.closed


async def test_connect_closes_a_leg_when_cancelled_mid_handshake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leg = _FakeSocket(hang=True)
    _ = _connector(monkeypatch, leg)
    task = asyncio.create_task(_connect())
    _ = await asyncio.wait_for(leg.waiting.wait(), 1)
    _ = task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert leg.closed


class _ClosingSocket(_FakeSocket):
    """A leg that takes the startup handshake, then finds every later send closed."""

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


class _Messages:
    """The server's leg as it is read: scripted messages, then its end."""

    def __init__(self, *messages: str | bytes, closes: bool) -> None:
        self._messages = list(messages)
        self._closes = closes

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> str | bytes:
        if self._messages:
            return self._messages.pop(0)
        if self._closes:
            raise ConnectionClosed(None, None)
        raise StopAsyncIteration


async def _relay_gone(data: bytes) -> None:
    """Send to a relay the server has already closed."""
    del data
    raise ConnectionClosed(None, None)


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
