"""Real user scopes enforce cgroup membership and an offline kill-all deadline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import os
import signal
import subprocess
import time
import uuid

import pytest

from knowop.shell.job_host import SystemdLauncher
from knowop.shell.job_limits import cap_problem, run, scope_argv


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="module")
def capped_user_manager() -> None:
    """Require the same working cap as the live scope check."""
    if (problem := cap_problem()) is not None:
        pytest.skip(f"no user manager caps a scope here: {problem}")


@pytest.fixture
def unit(capped_user_manager: None) -> Iterator[str]:
    """Give each scope its own name and kill it even after an assertion fails."""
    del capped_user_manager
    name = f"knowop-primitive-{uuid.uuid4().hex}"
    try:
        yield name
    finally:
        SystemdLauncher().kill(name, signal.SIGKILL)
        _ = run(["systemctl", "--user", "reset-failed", f"{name}.scope"])


def test_a_real_scope_runs_its_command_under_its_unit(unit: str) -> None:
    code, out = run(
        scope_argv(unit=unit, memory_mb=64, argv=["cat", "/proc/self/cgroup"]),
    )

    assert code == 0
    assert f"{unit}.scope" in out


def test_a_real_scope_deadline_kills_a_detached_descendant(
    unit: str,
    tmp_path: Path,
) -> None:
    pidfile = tmp_path / "descendant"
    argv = scope_argv(
        unit=unit,
        memory_mb=64,
        runtime_sec=0.02,
        argv=[
            "sh",
            "-c",
            (
                'setsid sh -c \'trap "" TERM; printf "%s\\n" "$$" > "$1.tmp"; '
                'mv "$1.tmp" "$1"; exec sleep 30\' sh "$1" & wait'
            ),
            "sh",
            str(pidfile),
        ],
    )
    with subprocess.Popen(  # noqa: S603 -- Fixed scope command and tmp_path argv.
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as child:
        try:
            _wait_ready(pidfile, child=child)
            pid = int(pidfile.read_text())
            assert os.getpgid(pid) == pid
            assert _running(pid)
            assert child.wait(timeout=2) != 0
            _wait_dead(pid)
            assert not _running(pid)
            assert not SystemdLauncher().alive(unit)
        finally:
            SystemdLauncher().kill(unit, signal.SIGKILL)
            child.wait(timeout=2)


def test_systemd_keeps_a_timeout_result_for_a_scope_its_deadline_stopped(
    unit: str,
) -> None:
    code, _ = run(
        scope_argv(
            unit=unit,
            memory_mb=64,
            runtime_sec=0.02,
            argv=["sleep", "5"],
        ),
    )

    assert code != 0
    assert SystemdLauncher().timed_out(unit)


def _wait_ready(path: Path, *, child: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 1
    while not (path.exists() and path.read_text().strip()):
        assert child.poll() is None, child.communicate()
        assert time.monotonic() < deadline, "descendant did not start"
        time.sleep(0.001)


def _running(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


def _wait_dead(pid: int) -> None:
    deadline = time.monotonic() + 1
    while _running(pid) and time.monotonic() < deadline:
        time.sleep(0.001)


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
