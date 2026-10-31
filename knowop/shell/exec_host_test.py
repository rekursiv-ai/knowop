"""The exec host and its command runner, with real shell commands."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

import asyncio
import base64
import gc
import json
import os
import queue
import stat
import subprocess
import threading
import time
import uuid

from trackinizer.client.errors import ClientError
from trackinizer.lib.userdirs import state_dir
from trackinizer.wire.wire_exec import (
    MAX_STREAM_BYTES,
    TIMEOUT_EXIT_CODE,
    ExecRequest,
    ExecResult,
    ExecWait,
)

import pytest

from knowop.shell import exec_host
from knowop.shell.exec_command import run_command
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


def _alive(pid: int) -> bool:
    """Whether ``pid`` still runs; a process that only awaits its reaping is dead."""
    state = subprocess.run(  # noqa: S603 -- fixed command, an int argument.
        ["ps", "-o", "stat=", "-p", str(pid)],  # noqa: S607 -- ps is on every PATH.
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return bool(state) and not state.startswith("Z")


def _wait_until(done: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5
    while not done():
        assert time.monotonic() < deadline, "the awaited change never came"
        time.sleep(0.005)


def _wait_for(path: Path) -> None:
    """Wait until ``path`` holds output; the shell creates it before it writes."""
    _wait_until(lambda: path.exists() and path.stat().st_size > 0)


def test_the_commands_exit_code_and_both_streams_are_reported(tmp_path: Path) -> None:
    result = _run(_request("echo out; echo err >&2; exit 3", cwd=tmp_path))

    assert (result.exit_code, _out(result), _err(result)) == (3, b"out\n", b"err\n")
    assert not result.truncated


def test_a_command_that_fails_to_parse_reports_the_shells_status(
    tmp_path: Path,
) -> None:
    result = _run(_request("if then", cwd=tmp_path))

    assert result.exit_code == 2
    assert b"syntax error" in _err(result)


def test_a_command_killed_by_a_signal_reports_the_shell_convention(
    tmp_path: Path,
) -> None:
    result = _run(_request("kill -9 $$", cwd=tmp_path))

    assert result.exit_code == 128 + 9


def test_the_command_runs_in_a_login_shell(tmp_path: Path) -> None:
    result = _run(_request("shopt -q login_shell && echo login", cwd=tmp_path))

    assert _out(result) == b"login\n"


def test_stdin_comes_from_the_request(tmp_path: Path) -> None:
    result = _run(_request("cat", cwd=tmp_path, stdin=b"from the request\n"))

    assert _out(result) == b"from the request\n"


def test_a_command_with_no_stdin_reads_end_of_file(tmp_path: Path) -> None:
    result = _run(_request("cat; echo done", cwd=tmp_path))

    assert (result.exit_code, _out(result)) == (0, b"done\n")


def test_a_command_that_never_reads_a_large_stdin_still_finishes(
    tmp_path: Path,
) -> None:
    result = _run(_request("echo done", cwd=tmp_path, stdin=b"x" * 900_000))

    assert (result.exit_code, _out(result)) == (0, b"done\n")


def test_the_command_starts_in_the_folder_and_reports_where_it_ended(
    tmp_path: Path,
) -> None:
    (tmp_path / "sub").mkdir()

    result = _run(_request("pwd; cd sub", cwd=tmp_path))

    assert _out(result) == f"{tmp_path}\n".encode()
    assert result.cwd == str(tmp_path / "sub")


def test_a_command_that_exits_inside_a_folder_still_reports_it(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()

    result = _run(_request("cd sub; exit 5", cwd=tmp_path))

    assert (result.exit_code, result.cwd) == (5, str(tmp_path / "sub"))


def test_a_home_relative_folder_is_this_users_home(empty_home: Path) -> None:
    result = _run(_request("pwd", cwd="~"))

    assert _out(result) == f"{empty_home}\n".encode()
    assert result.cwd == str(empty_home)


def test_a_folder_that_does_not_exist_runs_nothing(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere"

    result = _run(_request(f"touch {tmp_path}/ran", cwd=missing))

    assert result.exit_code == 1
    assert _err(result) == f"{missing} is not a folder on this machine\n".encode()
    assert result.cwd == str(missing)
    assert not (tmp_path / "ran").exists()


def test_a_folder_with_a_control_character_is_not_reported_as_the_end(
    tmp_path: Path,
) -> None:
    odd = tmp_path / "a\nb"
    odd.mkdir()

    result = _run(_request("cd 'a\nb'", cwd=tmp_path))

    assert result.cwd == str(tmp_path)


def test_the_request_adds_variables_to_the_hosts_environment(tmp_path: Path) -> None:
    result = _run(
        _request('echo "$GREETING $HOME"', cwd=tmp_path, env={"GREETING": "hi"}),
    )

    assert _out(result).startswith(b"hi /")


def test_a_command_past_its_timeout_is_killed_with_its_children(
    tmp_path: Path,
) -> None:
    pids = tmp_path / "pid"

    result = _run(
        _request(f"sleep 30 & echo $! > {pids}; wait", cwd=tmp_path, timeout_sec=0.3),
    )

    assert result.exit_code == TIMEOUT_EXIT_CODE
    assert b"timed out after 0.3 s" in _err(result)
    assert not _alive(int(pids.read_text()))


def test_a_child_that_ignores_termination_is_killed_at_the_timeout_too(
    tmp_path: Path,
) -> None:
    pids = tmp_path / "pid"

    result = _run(
        _request(
            f"(trap '' TERM; exec sleep 30) & echo $! > {pids}; wait",
            cwd=tmp_path,
            timeout_sec=0.3,
        ),
    )

    assert result.exit_code == TIMEOUT_EXIT_CODE
    assert not _alive(int(pids.read_text()))


def test_output_before_the_timeout_is_kept(tmp_path: Path) -> None:
    result = _run(_request("echo started; sleep 30", cwd=tmp_path, timeout_sec=0.3))

    assert _out(result) == b"started\n"
    assert result.exit_code == TIMEOUT_EXIT_CODE


def test_a_background_job_that_holds_the_pipes_does_not_hold_the_result(
    tmp_path: Path,
) -> None:
    pids = tmp_path / "pid"

    result = _run(
        _request(f"sleep 30 & echo $! > {pids}; echo done", cwd=tmp_path),
    )

    try:
        assert (result.exit_code, _out(result)) == (0, b"done\n")
    finally:
        os.kill(int(pids.read_text()), 9)


def test_ending_the_wait_kills_the_command_and_reports_what_it_had_printed(
    tmp_path: Path,
) -> None:
    pids = tmp_path / "pid"

    async def end_after_start() -> ExecResult:
        ended = asyncio.Event()
        running = asyncio.create_task(
            run_command(
                _request(f"echo up; sleep 30 & echo $! > {pids}; wait", cwd=tmp_path),
                ended=ended,
            ),
        )
        await asyncio.to_thread(_wait_for, pids)
        ended.set()
        return await running

    started = time.monotonic()
    result = asyncio.run(end_after_start())

    assert time.monotonic() - started < 5
    assert _out(result) == b"up\n"
    assert not _alive(int(pids.read_text()))


def test_output_over_the_limit_is_cut_from_the_middle_and_marked(
    tmp_path: Path,
) -> None:
    result = _run(
        _request(
            "printf HEAD; head -c 2000000 /dev/zero | tr '\\0' x; printf TAIL",
            cwd=tmp_path,
        ),
    )

    out = _out(result)
    assert result.truncated
    assert len(out) <= MAX_STREAM_BYTES
    assert out.startswith(b"HEADxxxx")
    assert out.endswith(b"xxxxTAIL")
    assert b"bytes cut" in out


def test_each_stream_is_cut_on_its_own(tmp_path: Path) -> None:
    result = _run(
        _request(
            "echo small; head -c 1100000 /dev/zero | tr '\\0' e >&2",
            cwd=tmp_path,
        ),
    )

    assert _out(result) == b"small\n"
    assert len(_err(result)) <= MAX_STREAM_BYTES
    assert b"bytes cut" in _err(result)
    assert result.truncated


def test_output_exactly_at_the_limit_is_not_cut(tmp_path: Path) -> None:
    result = _run(
        _request(f"head -c {MAX_STREAM_BYTES} /dev/zero | tr '\\0' x", cwd=tmp_path),
    )

    assert len(_out(result)) == MAX_STREAM_BYTES
    assert not result.truncated


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
    return [json.loads(line) for line in path.read_text().splitlines()]


def _serve_one(server: _Server, host: ExecHost, command: ExecWait) -> None:
    """Run one command through the host and stop it once the result is posted."""
    host.start(_MACHINE, _INSTANCE)
    server.waits.put(command)
    assert server.posted.wait(5)
    server.waits.put(_REVOKED)
    host.stop()


def test_a_command_is_run_and_its_result_posted_under_the_commands_id(
    tmp_path: Path,
) -> None:
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


def test_each_command_leaves_one_private_log_line_naming_who_asked(
    tmp_path: Path,
) -> None:
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


def test_a_revoked_machine_kills_the_commands_still_running(tmp_path: Path) -> None:
    server = _Server()
    host = _host(server)
    pids = tmp_path / "pid"

    host.start(_MACHINE, _INSTANCE)
    server.waits.put(
        _command(f"sleep 30 & echo $! > {pids}; wait", cwd=tmp_path),
    )
    _wait_for(pids)
    server.waits.put(_REVOKED)
    # Before the host is stopped: the revoke alone ends the command.
    _wait_until(lambda: not _alive(int(pids.read_text())))
    host.stop()

    assert server.results == []
    [line] = _log_lines()
    assert line["exit_code"] == 128 + 9


def test_a_command_that_starts_after_a_stop_is_ended_at_once(tmp_path: Path) -> None:
    host = _host(_Server())
    host._stop.set()
    pids = tmp_path / "pid"

    started = time.monotonic()
    result, killed = asyncio.run(
        host._execute(_command(f"sleep 30 & echo $! > {pids}; wait", cwd=tmp_path)),
    )

    assert killed
    assert time.monotonic() - started < 5
    assert result.exit_code == 128 + 9


def test_stopping_the_host_kills_the_commands_still_running(tmp_path: Path) -> None:
    server = _Server()
    host = _host(server)
    pids = tmp_path / "pid"

    host.start(_MACHINE, _INSTANCE)
    server.waits.put(
        _command(f"sleep 30 & echo $! > {pids}; wait", cwd=tmp_path),
    )
    _wait_for(pids)
    server.waits.put(_REVOKED)
    started = time.monotonic()
    host.stop()

    assert time.monotonic() - started < 5
    assert not _alive(int(pids.read_text()))
    assert server.results == []


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


def test_a_result_that_failed_in_transit_is_sent_again(tmp_path: Path) -> None:
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


def _cannot_spawn(error: OSError) -> Callable[..., object]:
    """Stand in for the event loop's ``subprocess_exec``, which the runner starts with."""

    async def subprocess_exec(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise error

    return subprocess_exec


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
    monkeypatch.setattr(asyncio.BaseEventLoop, "subprocess_exec", _cannot_spawn(error))
    # An earlier test's command, whose background job still held its pipes, left its
    # transport to the collector, which closes those pipe ends whenever it runs.
    # Collected first, they cannot count as descriptors this run closed.
    _ = gc.collect()
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
        asyncio.BaseEventLoop,
        "subprocess_exec",
        _cannot_spawn(PermissionError(13, "Permission denied")),
    )
    server = _Server()

    _serve_one(server, _host(server), _command("true", cwd=tmp_path))

    [(_, result)] = server.results
    assert result.exit_code == 126
    [line] = _log_lines()
    assert line["exit_code"] == 126


def test_stderr_at_the_limit_and_a_timeout_stay_within_the_limit(
    tmp_path: Path,
) -> None:
    result = _run(
        _request(
            f"head -c {MAX_STREAM_BYTES - 5} /dev/zero | tr '\\0' e >&2; sleep 30",
            cwd=tmp_path,
            timeout_sec=0.5,
        ),
    )

    assert result.exit_code == TIMEOUT_EXIT_CODE
    assert len(_err(result)) <= MAX_STREAM_BYTES
    assert _err(result).endswith(b"timed out after 0.5 s\n")
    assert result.truncated


def test_a_timeout_message_that_fits_leaves_the_stream_whole(tmp_path: Path) -> None:
    result = _run(_request("echo before >&2; sleep 30", cwd=tmp_path, timeout_sec=0.5))

    assert _err(result) == b"before\ntimed out after 0.5 s\n"
    assert not result.truncated


def test_the_hosts_trax_server_credential_and_profile_never_reach_a_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("TRACKINIZER_TOKEN", "TRACKINIZER_URL", "TRACKINIZER_PROFILE"):
        monkeypatch.setenv(name, "secret-" + name)
    monkeypatch.setenv("TRAX_CANVAS", "w")
    monkeypatch.setenv("LC_TRAX_SERVER", "https://hidden.example")
    monkeypatch.setenv("KEPT_BY_THE_HOST", "kept")

    result = _run(
        _request(
            "env | sort",
            cwd=tmp_path,
            env={"TRAX_CANVAS": "from-the-request"},
        ),
    )

    shown = _out(result).decode()
    assert "secret-" not in shown
    assert "hidden.example" not in shown
    assert "KEPT_BY_THE_HOST=kept" in shown
    # What the request sets is the server's own word, and it is not withheld.
    assert "TRAX_CANVAS=from-the-request" in shown


def test_a_log_that_cannot_be_written_never_holds_the_result_back(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
