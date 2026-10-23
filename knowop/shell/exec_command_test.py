"""The command runner's edges; the exec host's tests run it end to end."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

import asyncio
import base64
import os
import signal

from trackinizer.wire.wire_exec import (
    MAX_STREAM_BYTES,
    TIMEOUT_EXIT_CODE,
    ExecRequest,
    ExecResult,
)

import pytest

from knowop.shell.exec_command import (
    _kill_group,
    _Shell,
    _supervise,
    run_command,
)
from knowop.shell.exec_stream import Cut


if TYPE_CHECKING:
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before Python 3.13, ``Process.wait`` returns only once the pipes close.

    A job the shell left in the background holds them, so the runner must not
    wait on it: here it never returns at all.
    """
    monkeypatch.setattr(asyncio.subprocess.Process, "wait", _never)

    result = asyncio.run(
        asyncio.wait_for(run_command(_request("echo done", cwd=tmp_path)), 5),
    )

    assert (result.exit_code, _out(result)) == (0, b"done\n")


def test_a_command_past_its_timeout_is_reported_as_one(tmp_path: Path) -> None:
    result = asyncio.run(
        run_command(_request("echo started; sleep 30", cwd=tmp_path, timeout_sec=0.2)),
    )

    assert result.exit_code == TIMEOUT_EXIT_CODE
    assert (_out(result), _err(result)) == (b"started\n", b"timed out after 0.2 s\n")


def test_both_streams_are_reported_and_either_one_cut_marks_the_result(
    tmp_path: Path,
) -> None:
    long = MAX_STREAM_BYTES + 1

    result = asyncio.run(
        run_command(_request(f"echo oops >&2; head -c {long} /dev/zero", cwd=tmp_path)),
    )

    assert result.exit_code == 0
    assert _err(result) == b"oops\n"
    assert result.truncated


def test_a_home_folder_of_a_user_this_machine_lacks_runs_nothing(
    tmp_path: Path,
) -> None:
    folder = "~no-such-user-of-knowop"

    result = asyncio.run(run_command(_request(f"touch {tmp_path}/ran", cwd=folder)))

    assert result.exit_code == 1
    assert _err(result) == f"{folder} is not a folder on this machine\n".encode()
    assert not (tmp_path / "ran").exists()


def test_a_command_started_without_its_pipes_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="started without its pipes"):
        asyncio.run(_supervise_without_pipes(tmp_path))


@pytest.mark.parametrize("refusal", [ProcessLookupError, PermissionError])
def test_a_group_already_gone_or_not_this_users_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
    refusal: type[OSError],
) -> None:
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "killpg", partial(_refuse_kill, refusal, killed))

    _kill_group(cast("asyncio.subprocess.Process", _Running(pid=4242)))

    assert killed == [(4242, signal.SIGKILL)]


async def _supervise_without_pipes(cwd: Path) -> None:
    """Supervise a process started with none of the pipes the runner feeds and reads."""
    process = await asyncio.create_subprocess_exec("true")
    try:
        _ = await _supervise(
            process,
            _request("true", cwd=cwd),
            shell=_Shell(loop=asyncio.get_running_loop()),
            ended=None,
            out=Cut(),
            err=Cut(),
        )
    finally:
        _ = await process.wait()


async def _never(process: asyncio.subprocess.Process) -> int:
    """Stand in for ``Process.wait`` while something still holds the pipes."""
    del process
    await asyncio.Event().wait()
    return 0


@dataclass(frozen=True, slots=True, kw_only=True)
class _Running:
    """A shell still running, as far as its parent knows."""

    pid: int
    returncode: int | None = None


def _refuse_kill(
    refusal: type[OSError],
    killed: list[tuple[int, int]],
    pgid: int,
    signum: int,
) -> None:
    """Stand in for ``os.killpg`` on a group the kernel will not signal."""
    killed.append((pgid, signum))
    raise refusal


def _request(command: str, *, cwd: Path | str, timeout_sec: float = 10) -> ExecRequest:
    # A fraction of a second, which the wire's whole seconds cannot say, keeps the
    # timeout test from waiting out a real one.
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
