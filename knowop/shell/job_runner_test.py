"""The runner's exit, deadline and memory-cap decisions with OS process replies."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast
from unittest.mock import Mock, call

import os
import runpy
import signal
import subprocess
import sys
import time

import pytest

from knowop.shell import job_runner
from knowop.shell.conftest import _folder, _python
from knowop.shell.job_folder import Ended
from knowop.shell.job_runner import run_job


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import BinaryIO

    from trackinizer.mock_test_support import MockProtocol


class _Child(Protocol):
    pid: int
    args: list[str]
    wait: MockProtocol[int]


class _Popen(Mock):
    return_value: _Child


@pytest.fixture(autouse=True)
def popen(monkeypatch: pytest.MonkeyPatch) -> _Popen:
    """Supply process wait outcomes without replacing the runner's decisions or files."""
    child = Mock(spec=subprocess.Popen, pid=7, args=["job"], wait=Mock(return_value=0))
    started = _Popen(return_value=child)
    monkeypatch.setattr(subprocess, "Popen", started)
    return started


@pytest.fixture(autouse=True)
def signal_group(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Record the runner's signals without signalling a real process group."""
    sent = Mock()
    monkeypatch.setattr(os, "killpg", sent)
    return sent


def _spawn_output(
    child: _Child,
    data: bytes,
    argv: Sequence[str],
    **options: object,
) -> _Child:
    del argv
    _ = cast("BinaryIO", options["stdout"]).write(data)
    return child


def test_a_job_runs_in_its_worktree_and_its_output_joins_the_log(
    tmp_path: Path,
    popen: _Popen,
) -> None:
    argv = _python(
        "import os, sys; print(os.getcwd(), flush=True); print('oops', file=sys.stderr)",
    )
    folder = _folder(tmp_path, *argv)
    folder.note("worktree ready")
    popen.side_effect = partial(
        _spawn_output,
        popen.return_value,
        f"{folder.src}\noops\n".encode(),
    )

    ended = run_job(folder, events=None, limit_sec=30)

    assert ended == Ended(state="completed", exit_code=0, reason="")
    assert folder.log.read_text() == (f"knowop: worktree ready\n{folder.src}\noops\n")
    assert popen.call_args is not None
    log = cast("BinaryIO", popen.call_args.kwargs["stdout"])
    assert log.name == str(folder.log)
    assert log.closed
    popen.assert_called_once_with(
        argv,
        cwd=folder.src,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        process_group=0,
    )
    popen.return_value.wait.assert_called_once_with(timeout=30)


def test_a_job_that_exits_badly_is_failed_with_its_code(
    tmp_path: Path,
    popen: _Popen,
) -> None:
    folder = _folder(tmp_path, *_python("raise SystemExit(3)"))
    popen.return_value.wait.return_value = 3

    assert run_job(folder, events=None, limit_sec=30) == Ended(
        state="failed",
        exit_code=3,
        reason="exit 3",
    )


def test_a_job_a_signal_killed_reports_128_plus_the_signal(
    tmp_path: Path,
    popen: _Popen,
) -> None:
    folder = _folder(tmp_path, *_python("import os, signal; os.kill(os.getpid(), 9)"))
    popen.return_value.wait.return_value = -signal.SIGKILL

    assert run_job(folder, events=None, limit_sec=30) == Ended(
        state="failed",
        exit_code=137,
        reason="exit 137",
    )


def test_a_command_that_cannot_start_is_failed_with_why(
    tmp_path: Path,
    popen: _Popen,
) -> None:
    missing = _folder(tmp_path / "a", "no-such-command-knowop")
    script = tmp_path / "script"
    _ = script.write_text("#!/bin/sh\n")
    not_runnable = _folder(tmp_path / "b", str(script))
    popen.side_effect = [
        FileNotFoundError(2, "No such file or directory"),
        PermissionError(13, "Permission denied"),
    ]

    lost = run_job(missing, events=None, limit_sec=30)
    refused = run_job(not_runnable, events=None, limit_sec=30)

    assert (lost.state, lost.exit_code) == ("failed", 127)
    assert lost.reason.startswith("cannot start the command: [Errno 2]")
    assert (refused.state, refused.exit_code) == ("failed", 126)


def test_a_job_past_its_limit_is_stopped_with_its_whole_group(
    tmp_path: Path,
    popen: _Popen,
    signal_group: Mock,
) -> None:
    pid = tmp_path / "pid"
    folder = _folder(
        tmp_path,
        "sh",
        "-c",
        f"sleep 30 & echo $! > {pid}; wait",
        minutes=3,
    )
    child = popen.return_value
    child.wait.side_effect = [
        subprocess.TimeoutExpired(child.args, 0.05),
        -signal.SIGTERM,
    ]

    ended = run_job(folder, events=None, limit_sec=0.05, grace_sec=5)

    assert ended == Ended(
        state="timeout",
        exit_code=124,
        reason="stopped at its 3 min limit",
    )
    signal_group.assert_called_once_with(child.pid, signal.SIGTERM)
    assert child.wait.call_args_list == [call(timeout=0.05), call(timeout=5)]


def test_a_job_that_ignores_sigterm_is_killed_and_reaped_once_the_grace_is_up(
    tmp_path: Path,
    popen: _Popen,
    signal_group: Mock,
) -> None:
    pid, shell = tmp_path / "pid", tmp_path / "shell"
    folder = _folder(
        tmp_path,
        "sh",
        "-c",
        f"echo $$ > {shell}; trap '' TERM; sleep 30 & echo $! > {pid}; wait",
    )
    child = popen.return_value
    child.wait.side_effect = [
        subprocess.TimeoutExpired(child.args, 0.05),
        subprocess.TimeoutExpired(child.args, 0.05),
        -signal.SIGKILL,
    ]

    started = time.monotonic()
    ended = run_job(folder, events=None, limit_sec=0.05, grace_sec=0.05)

    assert ended.state == "timeout"
    assert time.monotonic() - started < 5
    assert signal_group.call_args_list == [
        call(child.pid, signal.SIGTERM),
        call(child.pid, signal.SIGKILL),
    ]
    assert child.wait.call_args_list == [call(timeout=0.05), call(timeout=0.05), call()]


def test_a_process_the_kernel_killed_at_the_cap_names_the_cap(
    tmp_path: Path,
    popen: _Popen,
) -> None:
    folder = _folder(tmp_path, *_python("raise SystemExit(137)"))
    popen.return_value.wait.return_value = 137
    events = tmp_path / "memory.events"
    _ = events.write_text("low 0\nhigh 0\nmax 3\noom 1\noom_kill 1\noom_group_kill 0\n")
    quiet = tmp_path / "quiet.events"
    _ = quiet.write_text("low 0\noom 0\noom_kill 0\n")

    capped = run_job(folder, events=events, limit_sec=30)
    exited = run_job(folder, events=quiet, limit_sec=30)

    assert capped == Ended(
        state="failed",
        exit_code=137,
        reason="killed at its 1024 MiB memory cap",
    )
    assert exited.reason == "exit 137"


def test_an_events_file_without_a_kill_count_counts_none(tmp_path: Path) -> None:
    folder = _folder(tmp_path, *_python("pass"))
    events = tmp_path / "memory.events"
    _ = events.write_text("low 0\n")

    assert run_job(folder, events=events, limit_sec=30).state == "completed"


def test_the_scopes_events_are_found_from_the_process_cgroup(tmp_path: Path) -> None:
    cgroup = tmp_path / "cgroup"
    _ = cgroup.write_text("0::/user.slice/run-u7.scope\n")
    scope = tmp_path / "sys" / "user.slice" / "run-u7.scope"
    scope.mkdir(parents=True)

    absent = job_runner._memory_events(cgroup, root=tmp_path / "sys")
    _ = (scope / "memory.events").write_text("oom_kill 0\n")
    present = job_runner._memory_events(cgroup, root=tmp_path / "sys")

    assert (absent, present) == (None, scope / "memory.events")


def test_main_runs_the_folders_job_niced_under_its_own_scope_and_records_its_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = _folder(tmp_path, *_python("pass"), minutes=1)
    niced: list[int] = []
    asked: list[Path] = []

    def nice(increment: int) -> int:
        niced.append(increment)
        return increment

    def no_events(cgroup: Path) -> None:
        asked.append(cgroup)

    # The real ``os.nice`` would lower this test worker for every later test.
    monkeypatch.setattr(os, "nice", nice)
    monkeypatch.setattr(job_runner, "_memory_events", no_events)

    code = job_runner.main([str(folder.path)])

    assert (code, niced, asked) == (0, [10], [Path("/proc/self/cgroup")])
    assert folder.ended() == Ended(state="completed", exit_code=0, reason="")


@pytest.mark.cli_python_subprocess
def test_the_runner_records_the_jobs_end_as_a_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    popen: _Popen,
) -> None:
    folder = _folder(tmp_path, *_python("print('module-job')"), minutes=1)
    popen.side_effect = partial(_spawn_output, popen.return_value, b"module-job\n")
    niced: list[int] = []

    def nice(increment: int) -> int:
        niced.append(increment)
        return increment

    monkeypatch.setattr(os, "nice", nice)
    monkeypatch.setattr(sys, "argv", [job_runner.__name__, str(folder.path)])
    monkeypatch.delitem(sys.modules, job_runner.__name__)
    monkeypatch.chdir(folder.path)
    with pytest.raises(SystemExit) as exited:
        runpy.run_module(job_runner.__name__, run_name="__main__", alter_sys=True)

    assert exited.value.code == 0
    assert folder.ended() == Ended(state="completed", exit_code=0, reason="")
    assert folder.log.read_text() == "module-job\n"
    assert niced == [10]


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
