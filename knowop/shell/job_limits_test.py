"""What a computer lets its jobs use: the cap probe, the scope, sizes and idle GPUs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import os
import shutil
import subprocess
import sys

import pytest

from knowop.shell import job_limits
from knowop.shell.job_limits import (
    DEFAULT_RESULTS_MAX,
    Gpu,
    cap_problem,
    gpus,
    idle_gpus,
    memory_budget_mb,
    parse_size,
    run,
    scope_argv,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


_GB10 = "0, GPU-3230181f-635c-b306-a4c5-40cdd3c05561, NVIDIA GB10, [N/A]\n"
_H100S = (
    "0, GPU-aaaa, NVIDIA H100 80GB HBM3, 0\n"
    "1, GPU-bbbb, NVIDIA H100 80GB HBM3, 40000\n"
    "2, GPU-cccc, NVIDIA H100 80GB HBM3, 512\n"
    "3, GPU-dddd, NVIDIA H100 80GB HBM3, 513\n"
)


class _Run:
    """Answers each program by its first word, recording every argv."""

    def __init__(self, **answers: tuple[int, str]) -> None:
        self.answers = answers
        self.seen: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> tuple[int, str]:
        self.seen.append(list(argv))
        key = argv[0].replace("-", "_")
        if key == "nvidia_smi" and any("compute-apps" in each for each in argv):
            key = "apps"
        return self.answers.get(key, (127, ""))


def test_the_results_a_job_sends_back_are_at_most_256_mib_by_default() -> None:
    assert DEFAULT_RESULTS_MAX == 268_435_456


@pytest.mark.parametrize(
    ("text", "size"),
    [
        ("32G", 32 * 2**30),
        ("512m", 512 * 2**20),
        (" 1T ", 2**40),
        ("7", 7),
        ("2K", 2_048),
    ],
)
def test_a_size_names_binary_units(text: str, size: int) -> None:
    assert parse_size(text) == size


@pytest.mark.parametrize("text", ["", "G", "1.5G", "32GB", "-1G", "32 G"])
def test_a_size_that_is_not_one_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match="is not a size"):
        _ = parse_size(text)


def test_a_scope_caps_memory_with_no_swap_and_runs_the_command() -> None:
    argv = scope_argv(
        unit="knowop-job-ab-7",
        memory_mb=2_048,
        argv=["python", "-m", "x"],
    )

    assert argv == [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--unit",
        "knowop-job-ab-7",
        "-p",
        "MemoryMax=2048M",
        "-p",
        "MemorySwapMax=0",
        "--",
        "python",
        "-m",
        "x",
    ]


def test_the_cap_holds_when_a_probe_scope_reads_its_own_cap() -> None:
    fake = _Run(systemd_run=(0, "67108864\n"))

    assert cap_problem(fake) is None
    [argv] = fake.seen
    assert argv[:4] == ["systemd-run", "--user", "--scope", "--quiet"]
    assert "--unit" not in argv
    assert argv[argv.index("--") + 1 :][:2] == ["sh", "-c"]
    assert "MemoryMax=64M" in argv


@pytest.mark.parametrize(
    ("answer", "problem"),
    [
        ((127, ""), "could not start a capped scope"),
        ((1, ""), "could not start a capped scope"),
        ((0, "max\n"), "read its cap as 'max'"),
    ],
)
def test_no_cap_is_said_plainly(answer: tuple[int, str], problem: str) -> None:
    found = cap_problem(_Run(systemd_run=answer))

    assert found is not None
    assert problem in found


def test_gpus_are_read_from_nvidia_smi_and_none_without_it() -> None:
    found = gpus(_Run(nvidia_smi=(0, _GB10 + _H100S)))

    assert found[0] == Gpu(
        index=0,
        uuid="GPU-3230181f-635c-b306-a4c5-40cdd3c05561",
        name="NVIDIA GB10",
        used_mb=None,
    )
    assert found[0].unified
    assert [(g.index, g.used_mb, g.unified) for g in found[1:]] == [
        (0, 0, False),
        (1, 40_000, False),
        (2, 512, False),
        (3, 513, False),
    ]
    assert gpus(_Run()) == []
    assert gpus(_Run(nvidia_smi=(9, _H100S))) == []


def test_an_idle_gpu_holds_no_process_of_any_program_and_little_memory() -> None:
    fake = _Run(nvidia_smi=(0, _H100S), apps=(0, "GPU-aaaa\n\n"))

    # 0 runs another program's process, 1 holds 40 GB, 3 holds just over the line.
    assert idle_gpus(fake) == [2]


def test_a_gpu_that_shares_the_cpus_memory_is_never_idle() -> None:
    assert idle_gpus(_Run(nvidia_smi=(0, _GB10), apps=(0, ""))) == []


def test_no_gpu_is_idle_when_the_processes_cannot_be_listed() -> None:
    assert idle_gpus(_Run(nvidia_smi=(0, _H100S), apps=(6, ""))) == []


def test_jobs_may_hold_half_of_the_computers_memory() -> None:
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2**20

    assert memory_budget_mb() == total // 2


@pytest.mark.cli_python_subprocess
def test_run_returns_the_exit_code_and_output_and_127_for_a_missing_program() -> None:
    assert run([sys.executable, "-c", "print('hi'); raise SystemExit(3)"]) == (
        3,
        "hi\n",
    )
    assert run(["no-such-program-knowop"]) == (127, "")


def test_run_returns_the_programs_exit_code_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []

    def finished(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 3, stdout="hi\n", stderr="")

    monkeypatch.setattr(job_limits.subprocess, "run", finished)

    assert run(["sh", "-c", "exit 3"]) == (3, "hi\n")
    assert run(["no-such-program-knowop"]) == (127, "")
    assert seen == [[shutil.which("sh"), "-c", "exit 3"]]


def test_run_says_124_for_a_program_that_runs_out_of_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def too_slow(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise job_limits.subprocess.TimeoutExpired(cmd="sleep", timeout=30)

    monkeypatch.setattr(job_limits.subprocess, "run", too_slow)

    assert run(["sh"]) == (124, "")


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
