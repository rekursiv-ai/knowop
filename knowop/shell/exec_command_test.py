"""Exercise command supervision with explicit process replies and real streams."""

from __future__ import annotations

from asyncio.subprocess import Process
from functools import partial
from typing import TYPE_CHECKING, TypedDict, Unpack, cast
from unittest.mock import AsyncMock, Mock

import asyncio
import base64
import os
import signal

from trackinizer.lib.posix.child import Child
from trackinizer.wire.wire_exec import (
    MAX_STREAM_BYTES,
    TIMEOUT_EXIT_CODE,
    ExecRequest,
    ExecResult,
)

import pytest

from knowop.shell.exec_command import (
    _supervise,
    run_command,
    run_until_ended,
)
from knowop.shell.exec_stream import Cut


if TYPE_CHECKING:
    from asyncio import Future
    from collections.abc import Awaitable, Callable, Iterable
    from pathlib import Path


@pytest.fixture(autouse=True)
def empty_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Give the login shell a home with no profile, so it prints nothing of its own."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def test_the_shells_exit_ends_the_wait_where_waiting_waits_for_the_pipes(
    tmp_path: Path,
    empty_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        os,
        "environ",
        {
            "HOME": str(empty_home),
            "PATH": os.defpath,
            "KEPT": "yes",
            "TRACKINIZER_TOKEN": "secret",
            "TRAX_CANVAS": "hidden",
        },
    )
    request = _request("echo done", cwd=tmp_path)

    result, process, start = asyncio.run(
        asyncio.wait_for(
            _run_reply(request, monkeypatch=monkeypatch, stdout=b"done\n"),
            5,
        ),
    )

    assert (result.exit_code, _out(result)) == (0, b"done\n")
    process.wait.assert_not_called()
    process.stdin.write.assert_not_called()
    process.stdin.close.assert_called_once_with()
    _, options = start.call_args
    (report_fd,) = options["pass_fds"]
    start.assert_awaited_once_with(
        "bash",
        "-lc",
        f"trap 'printf %s \"$PWD\" >&{report_fd}' EXIT\necho done",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=tmp_path,
        env={"HOME": str(empty_home), "PATH": os.defpath, "KEPT": "yes"},
        pass_fds=(report_fd,),
        limit=2**16,
        start_new_session=True,
    )
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(report_fd)


def test_a_command_past_its_timeout_is_reported_as_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, process, _ = asyncio.run(
        _run_reply(
            _request("echo started; sleep 30", cwd=tmp_path, timeout_sec=0.2),
            monkeypatch=monkeypatch,
            stdout=b"started\n",
            code=None,
            expired=True,
        ),
    )

    assert result.exit_code == TIMEOUT_EXIT_CODE
    assert (_out(result), _err(result)) == (b"started\n", b"timed out after 0.2 s\n")
    assert process.returncode == -signal.SIGKILL


def test_both_streams_are_reported_and_either_one_cut_marks_the_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long = MAX_STREAM_BYTES + 1

    result, _, _ = asyncio.run(
        _run_reply(
            _request(f"echo oops >&2; head -c {long} /dev/zero", cwd=tmp_path),
            monkeypatch=monkeypatch,
            stdout=b"\x00" * long,
            stderr=b"oops\n",
        ),
    )

    assert result.exit_code == 0
    assert _err(result) == b"oops\n"
    assert result.truncated


def test_a_home_folder_of_a_user_this_machine_lacks_runs_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = "~no-such-user-of-knowop"
    start = AsyncMock(side_effect=AssertionError("A refused request started a shell."))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)

    result, ending = asyncio.run(
        run_until_ended(_request(f"touch {tmp_path}/ran", cwd=folder), ended=None),
    )

    assert ending == "exited"
    assert result.exit_code == 1
    assert _err(result) == f"{folder} is not a folder on this machine\n".encode()
    assert not (tmp_path / "ran").exists()
    start.assert_not_called()


def test_a_command_started_without_its_pipes_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="started without its pipes"):
        asyncio.run(_supervise_without_pipes(tmp_path))


@pytest.mark.asyncio
async def test_a_cancelled_run_kills_the_command_and_what_it_started(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _process(code=None)
    ready = asyncio.Event()
    process.stdin.close.side_effect = ready.set
    start = AsyncMock(return_value=cast(Process, process))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    kill = Mock(side_effect=partial(_kill_group, process))
    monkeypatch.setattr(os, "killpg", kill)
    running = asyncio.create_task(
        run_command(_request("sleep 60 & wait", cwd=tmp_path)),
    )
    await ready.wait()

    running.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, timeout=5)
    kill.assert_called_once_with(process.pid, signal.SIGKILL)
    assert process.returncode == -signal.SIGKILL
    assert process.stdout.at_eof()
    assert process.stderr.at_eof()


@pytest.mark.asyncio
async def test_a_run_cancelled_while_its_shell_starts_kills_what_it_started(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _process(code=None)
    starting, handed = asyncio.Event(), asyncio.Event()
    start = AsyncMock(side_effect=partial(_hand_over, process, starting, handed))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    kill = Mock(side_effect=partial(_kill_group, process))
    monkeypatch.setattr(os, "killpg", kill)
    running = asyncio.create_task(
        run_command(_request("sleep 60 & wait", cwd=tmp_path)),
    )
    await starting.wait()

    running.cancel()
    handed.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, timeout=5)
    kill.assert_called_once_with(process.pid, signal.SIGKILL)
    assert process.returncode == -signal.SIGKILL
    assert process.stdout.at_eof()
    assert process.stderr.at_eof()


async def _hand_over(
    process: Mock,
    starting: asyncio.Event,
    handed: asyncio.Event,
    *argv: str,
    **options: object,
) -> Process:
    del argv, options
    starting.set()
    await handed.wait()
    return cast(Process, process)


def _process(*, stdout: bytes = b"", stderr: bytes = b"", code: int | None = 0) -> Mock:
    out, err = asyncio.StreamReader(), asyncio.StreamReader()
    out.feed_data(stdout)
    err.feed_data(stderr)
    if code is not None:
        out.feed_eof()
        err.feed_eof()
    process = Mock(
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


def _signal_process(process: Mock, signum: int) -> None:
    process.returncode = -signum
    process.stdout.feed_eof()
    process.stderr.feed_eof()


def _kill_group(process: Mock, pgid: int, signum: int) -> None:
    assert pgid == process.pid
    _signal_process(process, signum)


class _WaitOptions(TypedDict, total=False):
    timeout: float | None
    return_when: str


async def _expire(
    wait: Callable[..., Awaitable[tuple[set[Future[object]], set[Future[object]]]]],
    futures: Iterable[Future[object]],
    **options: Unpack[_WaitOptions],
) -> tuple[set[Future[object]], set[Future[object]]]:
    if options.get("return_when") == asyncio.FIRST_COMPLETED:
        assert options["timeout"] == 0.2
        await asyncio.sleep(0)
        return set(), set(futures)
    return await wait(futures, **options)


async def _run_reply(
    request: ExecRequest,
    *,
    monkeypatch: pytest.MonkeyPatch,
    stdout: bytes = b"",
    stderr: bytes = b"",
    code: int | None = 0,
    expired: bool = False,
) -> tuple[ExecResult, Mock, AsyncMock]:
    process = _process(stdout=stdout, stderr=stderr, code=code)
    start = AsyncMock(return_value=cast(Process, process))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    if expired:
        monkeypatch.setattr(asyncio, "wait", partial(_expire, asyncio.wait))
        monkeypatch.setattr(os, "killpg", partial(_kill_group, process))
    return await run_command(request), process, start


async def _supervise_without_pipes(cwd: Path) -> None:
    process = Mock(spec=Process, stdin=None, stdout=None, stderr=None, returncode=0)
    async with Child(
        process=cast(Process, process),
        group=False,
        grace_sec=0.0,
    ) as child:
        await _supervise(
            child,
            _request("true", cwd=cwd),
            ended=None,
            out=Cut(),
            err=Cut(),
        )


def _request(command: str, *, cwd: Path | str, timeout_sec: float = 10) -> ExecRequest:
    return ExecRequest.model_construct(
        command=command,
        cwd=str(cwd),
        timeout_sec=timeout_sec,
        stdin_b64=None,
        env=None,
    )


def _out(result: ExecResult) -> bytes:
    return base64.b64decode(result.stdout_b64)


def _err(result: ExecResult) -> bytes:
    return base64.b64decode(result.stderr_b64)


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
