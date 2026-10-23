"""The runner in a job's scope: how a job's end is told, at its limit and at its cap."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import os
import subprocess
import sys
import time

from trackinizer.wire.wire_machine_job import MachineJobSpec

import pytest

from knowop.shell import job_runner
from knowop.shell.job_folder import Ended, JobFolder
from knowop.shell.job_runner import run_job


# The export shortens the namespace, so the child needs its matching import root.
_REPO: Final = (
    Path(job_runner.__file__).resolve().parents[len(job_runner.__name__.split(".")) - 1]
)


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


def test_a_job_runs_in_its_worktree_and_its_output_joins_the_log(
    tmp_path: Path,
) -> None:
    folder = _folder(
        tmp_path,
        *_python(
            "import os, sys; print(os.getcwd(), flush=True); print('oops', file=sys.stderr)",
        ),
    )
    folder.note("worktree ready")

    ended = run_job(folder, events=None, limit_sec=30)

    assert ended == Ended(state="completed", exit_code=0, reason="")
    assert folder.log.read_text() == (f"knowop: worktree ready\n{folder.src}\noops\n")


def test_a_job_that_exits_badly_is_failed_with_its_code(tmp_path: Path) -> None:
    folder = _folder(tmp_path, *_python("raise SystemExit(3)"))

    assert run_job(folder, events=None, limit_sec=30) == Ended(
        state="failed",
        exit_code=3,
        reason="exit 3",
    )


def test_a_job_a_signal_killed_reports_128_plus_the_signal(tmp_path: Path) -> None:
    folder = _folder(tmp_path, *_python("import os, signal; os.kill(os.getpid(), 9)"))

    assert run_job(folder, events=None, limit_sec=30) == Ended(
        state="failed",
        exit_code=137,
        reason="exit 137",
    )


def test_a_command_that_cannot_start_is_failed_with_why(tmp_path: Path) -> None:
    missing = _folder(tmp_path / "a", "no-such-command-knowop")
    script = tmp_path / "script"
    _ = script.write_text("#!/bin/sh\n")
    not_runnable = _folder(tmp_path / "b", str(script))

    lost = run_job(missing, events=None, limit_sec=30)
    refused = run_job(not_runnable, events=None, limit_sec=30)

    assert (lost.state, lost.exit_code) == ("failed", 127)
    assert lost.reason.startswith("cannot start the command: [Errno 2]")
    assert (refused.state, refused.exit_code) == ("failed", 126)


def _gone(pid: int) -> bool:
    """Whether ``pid`` ends within a second."""
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.01)
    return False


def test_a_job_past_its_limit_is_stopped_with_its_whole_group(tmp_path: Path) -> None:
    pid = tmp_path / "pid"
    folder = _folder(
        tmp_path,
        "sh",
        "-c",
        f"sleep 30 & echo $! > {pid}; wait",
        minutes=3,
    )

    ended = run_job(folder, events=None, limit_sec=0.05, grace_sec=5)

    assert ended == Ended(
        state="timeout",
        exit_code=124,
        reason="stopped at its 3 min limit",
    )
    assert _gone(int(pid.read_text()))


def test_a_job_that_ignores_sigterm_is_killed_and_reaped_once_the_grace_is_up(
    tmp_path: Path,
) -> None:
    pid, shell = tmp_path / "pid", tmp_path / "shell"
    folder = _folder(
        tmp_path,
        "sh",
        "-c",
        f"echo $$ > {shell}; trap '' TERM; sleep 30 & echo $! > {pid}; wait",
    )

    started = time.monotonic()
    ended = run_job(folder, events=None, limit_sec=0.05, grace_sec=0.05)

    assert ended.state == "timeout"
    assert time.monotonic() - started < 5
    assert _gone(int(pid.read_text()))
    assert _gone(int(shell.read_text()))


def test_a_process_the_kernel_killed_at_the_cap_names_the_cap(tmp_path: Path) -> None:
    folder = _folder(tmp_path, *_python("raise SystemExit(137)"))
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
    monkeypatch.setattr(job_runner.os, "nice", nice)
    monkeypatch.setattr(job_runner, "_memory_events", no_events)

    code = job_runner.main([str(folder.path)])

    assert (code, niced, asked) == (0, [10], [Path("/proc/self/cgroup")])
    assert folder.ended() == Ended(state="completed", exit_code=0, reason="")


@pytest.mark.cli_python_subprocess
def test_the_runner_records_the_jobs_end_as_a_module(tmp_path: Path) -> None:
    folder = _folder(tmp_path, *_python("import os; print(os.nice(0))"), minutes=1)

    done = subprocess.run(  # noqa: S603 -- the runner, by module, as the host starts it.
        [sys.executable, "-m", job_runner.__name__, str(folder.path)],
        cwd=folder.path,
        env={**os.environ, "PYTHONPATH": str(_REPO)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert done.returncode == 0, done.stderr
    assert folder.ended() == Ended(state="completed", exit_code=0, reason="")
    assert int(folder.log.read_text()) >= 10


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
