"""What this computer lets jobs use: a memory cap systemd enforces, its GPUs, its memory.

``knowop connect --jobs`` proves the cap before it serves a job: it starts a 64 MiB
scope under systemd's user manager and reads that scope's ``memory.max`` back. A
computer with no user manager to ask (macOS, or Linux outside a user session) cannot
cap a job, and runs none.

A job's scope has no swap (``MemorySwapMax=0``), unlike ``runcap``'s: measured on gx10
on 2026-10-23, a 3 GiB allocation under ``MemoryMax=2G MemorySwapMax=2G`` lived by
swapping its last gibibyte, and with no swap it was killed (exit 137). A job's cap is
then the memory it may hold.

A GPU a job is given must be idle: no compute process on it, knowop's or another
program's, and almost no memory in use, so a service already on it keeps it. A GPU
whose memory ``nvidia-smi`` reports as ``[N/A]`` shares the CPU's memory (GB10), and a
cgroup does not bound what CUDA allocates there: measured on gx10 on 2026-10-23, a
3 GiB CUDA allocation under a 2 GiB cap held while the scope's ``memory.current`` stayed
at 453 MB. Such a GPU is given to no job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

import os
import re
import shutil
import subprocess


if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = [
    "DEFAULT_RESULTS_MAX",
    "Gpu",
    "Run",
    "cap_problem",
    "gpus",
    "idle_gpus",
    "memory_budget_mb",
    "parse_size",
    "run",
    "scope_argv",
]

DEFAULT_RESULTS_MAX: Final = 256 * 2**20
"""The most result bytes one job sends back, unless the host's owner sets another."""

IDLE_MEMORY_MB: Final = 512
"""A GPU that holds more memory than this is in use, whatever lists it."""

_PROBE_MB: Final = 64


class Run(Protocol):
    """Runs one argv list and returns its exit code and standard output."""

    def __call__(self, argv: Sequence[str]) -> tuple[int, str]:
        """Run ``argv`` to its end; a program that is missing exits 127."""
        ...


def run(argv: Sequence[str]) -> tuple[int, str]:
    """Run ``argv`` with no input, for at most 30 seconds.

    Args:
      argv: The program and its arguments.

    Returns:
      code: Its exit code; 127 when the program is not on ``PATH``, and 124 when it
        ran out of time.
      out: Its standard output.

    """
    program = shutil.which(argv[0])
    if program is None:
        return 127, ""
    try:
        done = subprocess.run(  # noqa: S603 -- fixed argv.
            [program, *argv[1:]],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return 124, ""
    return done.returncode, done.stdout


def parse_size(text: str) -> int:
    """Return the bytes a size such as ``32G``, ``512M`` or ``1T`` names.

    Args:
      text: A whole number and one of K, M, G or T, binary; a bare number is bytes.

    Returns:
      size: The bytes.

    Raises:
      ValueError: It is not such a size.

    """
    found = re.fullmatch(r"(\d+)([KMGT]?)", text.strip().upper())
    if found is None:
        raise ValueError(f"{text!r} is not a size such as 32G or 512M")
    return int(found[1]) << (10 * " KMGT".index(found[2] or " "))


def cap_problem(run: Run = run) -> str | None:
    """Say why systemd cannot cap a job's memory here, or ``None`` when it can.

    Args:
      run: Runs ``systemd-run``.

    Returns:
      problem: One sentence, or ``None`` once a scope read back its own cap.

    """
    code, out = run(
        scope_argv(
            unit="",
            memory_mb=_PROBE_MB,
            argv=(
                "sh",
                "-c",
                'cat "/sys/fs/cgroup$(sed -n "s/^0:://p" /proc/self/cgroup)/memory.max"',
            ),
        ),
    )
    if code:
        return "systemd-run --user could not start a capped scope"
    if out.strip() != str(_PROBE_MB * 2**20):
        return f"a {_PROBE_MB}M scope read its cap as {out.strip()!r}"
    return None


def scope_argv(
    *,
    unit: str,
    memory_mb: int,
    argv: Sequence[str],
    runtime_sec: float | None = None,
) -> list[str]:
    """Return the ``systemd-run`` command that runs ``argv`` in a capped scope.

    Args:
      unit: The scope's name, ``NAME.scope``; empty lets systemd choose one.
      memory_mb: The cap, in MiB.
      argv: The command to run in it.
      runtime_sec: Kill the entire scope at this deadline, even without its host.

    Returns:
      argv: The whole command.

    """
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        *(["--unit", unit] if unit else []),
        "-p",
        f"MemoryMax={memory_mb}M",
        "-p",
        "MemorySwapMax=0",
        *(
            []
            if runtime_sec is None
            else [
                "-p",
                f"RuntimeMaxSec={runtime_sec:g}",
                "-p",
                "KillSignal=SIGKILL",
            ]
        ),
        "--",
        *argv,
    ]


@dataclass(frozen=True, slots=True, kw_only=True)
class Gpu:
    """One GPU, as ``nvidia-smi`` lists it."""

    index: int
    uuid: str
    name: str
    used_mb: int | None
    """The memory in use; ``None`` when the GPU shares the CPU's memory."""

    @property
    def unified(self) -> bool:
        """Whether the GPU's memory is the CPU's, which no cgroup bounds for CUDA."""
        return self.used_mb is None


def gpus(run: Run = run) -> list[Gpu]:
    """Return this computer's GPUs; none when ``nvidia-smi`` is missing or fails.

    Args:
      run: Runs ``nvidia-smi``.

    Returns:
      gpus: In index order.

    """
    code, out = run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used",
            "--format=csv,noheader,nounits",
        ],
    )
    if code:
        return []
    found = []
    for line in out.splitlines():
        index, uuid, name, used = (part.strip() for part in line.split(","))
        found.append(
            Gpu(
                index=int(index),
                uuid=uuid,
                name=name,
                used_mb=int(used) if used.isdigit() else None,
            ),
        )
    return found


def idle_gpus(run: Run = run) -> list[int]:
    """Return the indexes of the GPUs a job may be given now.

    A GPU is idle when it does not share the CPU's memory, holds no compute process
    of any program, and uses at most :data:`IDLE_MEMORY_MB`. When the processes cannot
    be listed, no GPU is idle.

    Args:
      run: Runs ``nvidia-smi``.

    Returns:
      indexes: In index order.

    """
    code, out = run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
    )
    if code:
        return []
    busy = {line.strip() for line in out.splitlines() if line.strip()}
    return [
        gpu.index
        for gpu in gpus(run)
        if gpu.used_mb is not None
        and gpu.used_mb <= IDLE_MEMORY_MB
        and gpu.uuid not in busy
    ]


def memory_budget_mb() -> int:
    """Return the memory this computer's jobs may hold together: half of its RAM."""
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2**20 // 2
