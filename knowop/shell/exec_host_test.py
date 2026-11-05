"""Exercise host execution and private logs with scripted process replies."""

from __future__ import annotations

from asyncio.subprocess import Process
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast
from unittest.mock import AsyncMock, Mock

import asyncio
import base64
import os
import queue
import signal
import stat
import threading
import time
import uuid

from trackinizer.client.errors import ClientError
from trackinizer.lib.userdirs import state_dir
from trackinizer.wire.wire_exec import ExecRequest, ExecResult, ExecWait
from treekle import json_loads

import pytest

from knowop.shell import exec_host
from knowop.shell.exec_command import run_command
from knowop.shell.exec_command_test import _kill_group, _process
from knowop.shell.exec_host import ExecHost


if TYPE_CHECKING:
    from collections.abc import Callable


_MACHINE: Final = uuid.UUID(int=1)
_INSTANCE: Final = uuid.UUID(int=2)
_CREDENTIAL: Final = "trax_machine_" + "0" * 32 + "_" + "c" * 43
_REVOKED: Final = ClientError("-> 410", status_code=410)


@pytest.fixture(autouse=True)
def empty_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Give the login shell a home with no profile, so it prints nothing of its own."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def shell(monkeypatch: pytest.MonkeyPatch) -> _Shell:
    """Keep the command runner and Child real, supplying only OS process replies."""
    reply = _Shell()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", reply.start)
    monkeypatch.setattr(os, "killpg", reply.kill)
    return reply


class _Shell:
    """Explicit stdout/status replies and the processes started to deliver them."""

    def __init__(self) -> None:
        self.replies: dict[str, tuple[bytes, int | None]] = {}
        self.processes: list[Mock] = []
        self.ready = threading.Event()
        self.start = AsyncMock(side_effect=self._spawn)
        self.kill = Mock(side_effect=self._signal)

    async def _spawn(self, *argv: str, **options: object) -> Process:
        del options
        stdout, code = self.replies.get(argv[2].partition("\n")[2], (b"", 0))
        process = _process(stdout=stdout, code=code)
        process.pid += len(self.processes)
        self.processes.append(process)
        self.ready.set()
        return cast(Process, process)

    def _signal(self, pgid: int, signum: int) -> None:
        process = next(each for each in self.processes if each.pid == pgid)
        if process.returncode is not None:
            raise ProcessLookupError
        if signum:
            _kill_group(process, pgid, signum)


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
    assert time.monotonic() - started < 5
    return result


def _out(result: ExecResult) -> bytes:
    return base64.b64decode(result.stdout_b64)


def _err(result: ExecResult) -> bytes:
    return base64.b64decode(result.stderr_b64)


def _wait_until(done: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5
    while not done():
        assert time.monotonic() < deadline, "the awaited change never came"
        time.sleep(0.005)


def test_a_folder_that_does_not_exist_runs_nothing(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere"

    result = _run(_request(f"touch {tmp_path}/ran", cwd=missing))

    assert result.exit_code == 1
    assert _err(result) == f"{missing} is not a folder on this machine\n".encode()
    assert result.cwd == str(missing)
    assert not (tmp_path / "ran").exists()


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
        assert (machine_id, instance, wait_sec) == (_MACHINE, _INSTANCE, 25)
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
        assert machine_id == _MACHINE
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


def _serve_one(server: _Server, host: ExecHost, command: ExecWait) -> None:
    """Run one command through the host and stop it once the result is posted."""
    host.start(_MACHINE, _INSTANCE)
    server.waits.put(command)
    assert server.posted.wait(5)
    server.waits.put(_REVOKED)
    host.stop()


def test_a_command_is_run_and_its_result_posted_under_the_commands_id(
    tmp_path: Path,
    shell: _Shell,
) -> None:
    shell.replies["echo hello"] = (b"hello\n", 0)
    server = _Server()
    host = _host(server)
    command = _command("echo hello", cwd=tmp_path)

    host.start(_MACHINE, _INSTANCE)
    server.waits.put(command)
    assert server.posted.wait(5)
    server.waits.put(_REVOKED)
    host.stop()

    [(exec_id, result)] = server.results
    assert exec_id == command.exec_id
    assert (result.exit_code, _out(result)) == (0, b"hello\n")
    assert server.keys == [_CREDENTIAL, _CREDENTIAL]
    # The listener closes its client as it leaves, which may follow ``stop``.
    _wait_until(lambda: server.closed >= 2)
    _, options = shell.start.call_args
    (report_fd,) = options["pass_fds"]
    shell.start.assert_awaited_once_with(
        "bash",
        "-lc",
        f"trap 'printf %s \"$PWD\" >&{report_fd}' EXIT\necho hello",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=tmp_path,
        env={
            name: value
            for name, value in os.environ.items()
            if name
            not in {"TRACKINIZER_URL", "TRACKINIZER_TOKEN", "TRACKINIZER_PROFILE"}
            and not name.startswith(("TRAX_", "LC_TRAX_"))
        },
        pass_fds=(report_fd,),
        limit=2**16,
        start_new_session=True,
    )


def test_each_command_leaves_one_private_log_line_naming_who_asked(
    tmp_path: Path,
    shell: _Shell,
) -> None:
    shell.replies.update({"exit 4": (b"", 4), "true": (b"", 0)})
    server = _Server()
    host = _host(server)
    first = _command("exit 4", cwd=tmp_path)
    second = _command("true", cwd=tmp_path)

    host.start(_MACHINE, _INSTANCE)
    server.waits.put(first)
    server.waits.put(second)
    _wait_until(lambda: len(server.results) >= 2)
    server.waits.put(_REVOKED)
    host.stop()

    lines = {line["exec_id"]: line for line in _log_lines()}
    assert set(lines) == {str(first.exec_id), str(second.exec_id)}
    line = lines[str(first.exec_id)]
    assert line["who"] == "ada@example.com"
    assert line["command"] == "exit 4"
    assert line["cwd"] == str(tmp_path)
    assert line["exit_code"] == 4
    assert isinstance(line["duration_ms"], int)
    assert datetime.fromisoformat(str(line["at"])).tzinfo is not None
    path = state_dir() / "rekursiv-ai" / "trax" / "machines" / "box-1.exec.log"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_a_revoked_machine_kills_the_commands_still_running(
    tmp_path: Path,
    shell: _Shell,
) -> None:
    shell.replies["sleep 30 & wait"] = (b"", None)
    server = _Server()
    host = _host(server)

    host.start(_MACHINE, _INSTANCE)
    server.waits.put(_command("sleep 30 & wait", cwd=tmp_path))
    assert shell.ready.wait(5)
    server.waits.put(_REVOKED)
    _wait_until(lambda: shell.processes[0].returncode is not None)
    host.stop()

    assert server.results == []
    [line] = _log_lines()
    assert line["exit_code"] == 128 + 9
    shell.kill.assert_called_once_with(shell.processes[0].pid, signal.SIGKILL)


def test_a_command_that_starts_after_a_stop_is_ended_at_once(
    tmp_path: Path,
    shell: _Shell,
) -> None:
    shell.replies["sleep 30 & wait"] = (b"", None)
    host = _host(_Server())
    host._stop.set()

    started = time.monotonic()
    result, killed = asyncio.run(
        host._execute(_command("sleep 30 & wait", cwd=tmp_path)),
    )

    assert killed
    assert time.monotonic() - started < 5
    assert result.exit_code == 128 + 9
    shell.kill.assert_called_once_with(shell.processes[0].pid, signal.SIGKILL)


def test_stopping_the_host_kills_the_commands_still_running(
    tmp_path: Path,
    shell: _Shell,
) -> None:
    shell.replies["sleep 30 & wait"] = (b"", None)
    server = _Server()
    host = _host(server)

    host.start(_MACHINE, _INSTANCE)
    server.waits.put(_command("sleep 30 & wait", cwd=tmp_path))
    assert shell.ready.wait(5)
    started = time.monotonic()
    host.stop()
    server.waits.put(_REVOKED)
    _wait_until(lambda: server.closed >= 1)

    assert time.monotonic() - started < 5
    assert shell.processes[0].returncode == -signal.SIGKILL
    assert server.results == []
    shell.kill.assert_called_once_with(shell.processes[0].pid, signal.SIGKILL)


def test_a_result_the_server_no_longer_wants_is_dropped_without_a_retry(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server = _Server()
    server.post_errors.append(ClientError("-> 404", status_code=404))
    host = _host(server)

    host.start(_MACHINE, _INSTANCE)
    server.waits.put(_command("true", cwd=tmp_path))
    _wait_until(lambda: server.posts >= 1)
    server.waits.put(_REVOKED)
    host.stop()

    assert (server.posts, server.results) == (1, [])
    assert capsys.readouterr().err == ""


def test_a_result_that_failed_in_transit_is_sent_again(
    tmp_path: Path,
    shell: _Shell,
) -> None:
    shell.replies["echo again"] = (b"again\n", 0)
    server = _Server()
    server.post_errors.extend(
        [
            ClientError("POST /x failed: refused"),
            ClientError("-> 503", status_code=503),
        ],
    )
    host = _host(server)

    host.start(_MACHINE, _INSTANCE)
    server.waits.put(_command("echo again", cwd=tmp_path))
    assert server.posted.wait(5)
    server.waits.put(_REVOKED)
    host.stop()

    assert server.posts == 3
    assert _out(server.results[0][1]) == b"again\n"


def test_a_result_that_never_gets_through_is_reported_on_stderr(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server = _Server()
    server.post_errors.extend([ClientError("-> 503", status_code=503)] * 3)
    host = _host(server)

    host.start(_MACHINE, _INSTANCE)
    server.waits.put(_command("true", cwd=tmp_path))
    _wait_until(lambda: server.posts >= 3)
    server.waits.put(_REVOKED)
    host.stop()

    assert server.results == []
    assert "result not delivered" in capsys.readouterr().err


def test_a_failed_wait_is_reported_and_the_host_goes_on(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server = _Server()
    host = _host(server)

    host.start(_MACHINE, _INSTANCE)
    server.waits.put(ClientError("GET /x failed: refused"))
    server.waits.put(None)
    server.waits.put(_command("true", cwd=tmp_path))
    assert server.posted.wait(5)
    server.waits.put(_REVOKED)
    host.stop()

    assert "exec wait failed" in capsys.readouterr().err


# ``/dev/fd`` lists this process's descriptors on Linux and macOS alike; ``/proc``
# exists on Linux alone.
def _open_descriptors() -> int:
    return len(list(Path("/dev/fd").iterdir()))


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (PermissionError(13, "Permission denied"), 126),
        (FileNotFoundError(2, "No such file or directory"), 127),
        (OSError(12, "Cannot allocate memory"), 126),
    ],
)
def test_a_command_that_cannot_spawn_reports_why_and_leaks_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: OSError,
    code: int,
) -> None:
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(side_effect=error))
    before = _open_descriptors()

    result = _run(_request("true", cwd=tmp_path))

    assert result.exit_code == code
    assert b"cannot start the command" in _err(result)
    assert error.strerror is not None
    assert error.strerror.encode() in _err(result)
    assert result.cwd == str(tmp_path)
    assert _open_descriptors() == before


def test_a_command_that_cannot_spawn_is_still_logged_and_posted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        AsyncMock(side_effect=PermissionError(13, "Permission denied")),
    )
    server = _Server()

    _serve_one(server, _host(server), _command("true", cwd=tmp_path))

    [(_, result)] = server.results
    assert result.exit_code == 126
    [line] = _log_lines()
    assert line["exit_code"] == 126


def test_a_log_that_cannot_be_written_never_holds_the_result_back(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    shell: _Shell,
) -> None:
    shell.replies["echo ran"] = (b"ran\n", 0)
    log = state_dir() / "rekursiv-ai" / "trax" / "machines" / "box-1.exec.log"
    # A file where the machines folder belongs.
    log.parent.parent.mkdir(parents=True)
    _ = log.parent.write_text("")
    server = _Server()

    _serve_one(server, _host(server), _command("echo ran", cwd=tmp_path))

    [(_, result)] = server.results
    assert _out(result) == b"ran\n"
    assert "exec log not written" in capsys.readouterr().err


def test_a_new_log_is_private_from_its_creation_and_named_as_the_layout_says(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Private before anything fixes its mode; its names as the filesystem stores them.

    macOS opens the log under any casing of its path, so only the names a listing
    returns tell a misspelled directory apart.
    """
    monkeypatch.setattr(exec_host.os, "fchmod", _keep_mode)
    server = _Server()

    _serve_one(server, _host(server), _command("true", cwd=tmp_path))

    [log] = state_dir().rglob("*.exec.log")
    assert log.relative_to(state_dir()).as_posix() == (
        "rekursiv-ai/trax/machines/box-1.exec.log"
    )
    assert stat.S_IMODE(log.stat().st_mode) == 0o600


def _keep_mode(descriptor: int, mode: int) -> None:
    """Stand in for ``os.fchmod``: leave the mode the log was created with."""
    del descriptor, mode


def test_a_log_that_already_exists_is_made_private(tmp_path: Path) -> None:
    log = state_dir() / "rekursiv-ai" / "trax" / "machines" / "box-1.exec.log"
    log.parent.mkdir(parents=True)
    _ = log.write_text("")
    log.chmod(0o644)
    server = _Server()

    _serve_one(server, _host(server), _command("true", cwd=tmp_path))

    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    assert len(_log_lines()) == 1


def test_a_log_that_is_a_link_is_not_followed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = state_dir() / "rekursiv-ai" / "trax" / "machines" / "box-1.exec.log"
    elsewhere = tmp_path / "elsewhere"
    _ = elsewhere.write_text("kept\n")
    log.parent.mkdir(parents=True)
    log.symlink_to(elsewhere)
    server = _Server()

    _serve_one(server, _host(server), _command("true", cwd=tmp_path))

    assert elsewhere.read_text() == "kept\n"
    assert len(server.results) == 1
    assert "exec log not written" in capsys.readouterr().err


def test_the_log_names_the_variables_a_request_set_and_not_their_values(
    tmp_path: Path,
) -> None:
    log = state_dir() / "rekursiv-ai" / "trax" / "machines" / "box-1.exec.log"
    server = _Server()
    command = _command(
        "true",
        cwd=tmp_path,
        env={"LD_PRELOAD": "evil.so", "BASH_ENV": "rc"},
    )

    _serve_one(server, _host(server), command)

    [line] = _log_lines()
    assert line["env"] == ["BASH_ENV", "LD_PRELOAD"]
    assert "evil.so" not in log.read_text()


def test_a_command_handed_over_as_the_host_stops_is_answered_not_dropped(
    tmp_path: Path,
) -> None:
    server = _Server()
    host = _host(server)
    host._stop.set()
    command = _command("echo never", cwd=tmp_path)

    host._start_worker(command, machine_id=_MACHINE)

    [(exec_id, result)] = server.results
    assert exec_id == command.exec_id
    assert result.exit_code == 1
    assert b"stopped" in _err(result)
    assert _out(result) == b""


def test_a_command_that_finished_as_the_host_stopped_is_still_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _host(_Server())
    finished = _result_of_nothing(tmp_path)

    async def exits_as_the_stop_lands(
        request: ExecRequest,
        *,
        ended: asyncio.Event,
    ) -> tuple[ExecResult, str]:
        del request
        ended.set()
        return finished, "exited"

    monkeypatch.setattr(exec_host, "run_until_ended", exits_as_the_stop_lands)

    _, killed = asyncio.run(host._execute(_command("true", cwd=tmp_path)))

    assert not killed


def _result_of_nothing(cwd: Path) -> ExecResult:
    return ExecResult(
        exit_code=0,
        stdout_b64="",
        stderr_b64="",
        cwd=str(cwd),
        truncated=False,
        duration_ms=0,
    )


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
