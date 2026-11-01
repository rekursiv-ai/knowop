"""The scope's deadline and forced kill are passed to systemd as unit properties."""

from pathlib import Path
from unittest.mock import ANY, patch

import subprocess
import sys

from trackinizer.wire.wire_machine_job import MachineJobSpec

import pytest

from knowop.shell import job_host
from knowop.shell.job_folder import JobFolder
from knowop.shell.job_host import SystemdLauncher
from knowop.shell.job_limits import scope_argv


@pytest.mark.parametrize(
    ("runtime_sec", "runtime_property"),
    [(70, "RuntimeMaxSec=70"), (0.5, "RuntimeMaxSec=0.5")],
)
def test_a_scope_deadline_kills_the_scope_at_its_runtime_limit(
    runtime_sec: float,
    runtime_property: str,
) -> None:
    argv = scope_argv(
        unit="knowop-job-test",
        memory_mb=128,
        runtime_sec=runtime_sec,
        argv=["true"],
    )

    assert argv[
        argv.index(runtime_property) - 1 : argv.index(runtime_property) + 3
    ] == ["-p", runtime_property, "-p", "KillSignal=SIGKILL"]


def test_the_launcher_deadline_includes_the_limit_and_ten_second_grace(
    tmp_path: Path,
) -> None:
    folder = JobFolder.create(
        tmp_path / "7",
        MachineJobSpec(
            id=7,
            owner="jan@example.com",
            argv=["true"],
            remote="https://example.invalid/o/r.git",
            ref="refs/heads/main",
            commit="a" * 40,
            gpus=0,
            memory_mb=128,
            minutes=1,
            label="",
        ),
    )
    with patch.object(job_host.subprocess, "Popen") as popen:
        SystemdLauncher().start(folder, unit="deadline-test", memory_mb=128, env={})

    popen.assert_called_once_with(
        [
            "systemd-run",
            "--user",
            "--scope",
            "--quiet",
            "--unit",
            "deadline-test",
            "-p",
            "MemoryMax=128M",
            "-p",
            "MemorySwapMax=0",
            "-p",
            "RuntimeMaxSec=70",
            "-p",
            "KillSignal=SIGKILL",
            "--",
            sys.executable,
            "-m",
            "knowop.shell.job_runner",
            str(folder.path),
        ],
        cwd=folder.path,
        env={},
        stdin=subprocess.DEVNULL,
        stdout=ANY,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
