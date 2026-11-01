"""Runner children inherit the environment the machine host prepared."""

from pathlib import Path

from trackinizer.wire.wire_machine_job import MachineJobSpec

import pytest

from knowop.shell.job_folder import JobFolder
from knowop.shell.job_runner import run_job


def test_a_job_inherits_the_runners_empty_cuda_device_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    folder = JobFolder.create(
        tmp_path / "job",
        MachineJobSpec(
            id=7,
            owner="jan@example.com",
            argv=["sh", "-c", 'printf "[%s]\\n" "$CUDA_VISIBLE_DEVICES"'],
            remote="https://github.com/o/r.git",
            ref="refs/heads/main",
            commit="a" * 40,
            gpus=0,
            memory_mb=1_024,
            minutes=1,
            label="",
        ),
    )
    folder.src.mkdir()

    ended = run_job(folder, events=None, limit_sec=1)

    assert ended.state == "completed"
    assert folder.log.read_text() == "[]\n"


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
