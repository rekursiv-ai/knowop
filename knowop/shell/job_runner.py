#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Run one job inside its capped scope: start its command, hold it to its time, record its end.

The machine host starts ``systemd-run --user --scope ... -- python -m <this module>
FOLDER`` in the job's folder (``job_folder.py``), so this process and everything the
job starts share the scope's memory cap. It runs the job's command without a shell,
in the job's worktree, at nice 10, with its output appended to the job's log. At the
job's time limit it sends the command's process group SIGTERM, then SIGKILL ten
seconds later. It then writes ``exit.json``: the exit code and how the job ended,
naming the memory cap when the kernel killed a process of the scope at it.

It outlives the host that started it: a ``knowop connect`` that stops or restarts
leaves it running, and the next one reads its ``exit.json``.
'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import os
import signal
import subprocess

from knowop.shell.job_folder import Ended, JobFolder


if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = ["main", "run_job"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the job in the folder the arguments name; return the process exit code.

    Args:
      argv: The arguments; ``sys.argv[1:]`` when omitted.

    Returns:
      code: 0 once the job's end is recorded, whatever the job's own exit code.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args(argv))
    # Below the host's services, so a busy job slows itself first.
    _ = os.nice(10)
    folder = JobFolder(flags.folder)
    ended = run_job(
        folder,
        events=_memory_events(Path("/proc/self/cgroup")),
        limit_sec=folder.spec().minutes * 60,
    )
    folder.write_ended(ended)
    return 0


class Flags(Protocol):
    """Parsed command-line flags."""

    folder: Path


def run_job(
    folder: JobFolder,
    *,
    events: Path | None,
    limit_sec: float,
    grace_sec: float = 10.0,
) -> Ended:
    """Run the folder's job to its end or its time limit, and say how it ended.

    Args:
      folder: The job's folder, holding its spec and worktree.
      events: The scope's ``memory.events``; ``None`` where there is none to read.
      limit_sec: How long the job may run: its ``minutes``.
      grace_sec: How long a job told to stop at its limit has before it is killed.

    Returns:
      ended: ``completed`` for exit 0, ``timeout`` (exit 124) at the limit, else
        ``failed`` with the exit code, and the memory cap named when the kernel
        killed a process at it.

    """
    spec = folder.spec()
    with folder.log.open("ab") as log:
        try:
            child = subprocess.Popen(  # noqa: S603 -- the owner's own command, no shell.
                spec.argv,
                cwd=folder.src,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                process_group=0,
            )
        except OSError as failure:
            code = 127 if isinstance(failure, FileNotFoundError) else 126
            return Ended(
                state="failed",
                exit_code=code,
                reason=f"cannot start the command: {failure}",
            )
        try:
            code = child.wait(timeout=limit_sec)
        except subprocess.TimeoutExpired:
            _stop(child, grace_sec=grace_sec)
            return Ended(
                state="timeout",
                exit_code=124,
                reason=f"stopped at its {spec.minutes} min limit",
            )
    status = 128 - code if code < 0 else code
    if _oom_kills(events):
        return Ended(
            state="failed",
            exit_code=status,
            reason=f"killed at its {spec.memory_mb} MiB memory cap",
        )
    if status == 0:
        return Ended(state="completed", exit_code=0, reason="")
    return Ended(state="failed", exit_code=status, reason=f"exit {status}")


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument("folder", type=Path, help="the job's folder")


def _stop(child: subprocess.Popen[bytes], *, grace_sec: float) -> None:
    """Send the job's process group SIGTERM, then SIGKILL once the grace is up."""
    os.killpg(child.pid, signal.SIGTERM)
    try:
        _ = child.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        _ = child.wait()


def _oom_kills(events: Path | None) -> int:
    """Return how many processes the kernel killed at the scope's memory cap."""
    if events is None:
        return 0
    for line in events.read_text().splitlines():
        name, _, count = line.partition(" ")
        if name == "oom_kill":
            return int(count)
    return 0


def _memory_events(
    cgroup: Path,
    *,
    root: Path = Path("/sys/fs/cgroup"),
) -> Path | None:
    """Return the ``memory.events`` of the cgroup a ``/proc/*/cgroup`` file names."""
    found = root / cgroup.read_text().split("::", 1)[-1].strip().lstrip("/")
    events = found / "memory.events"
    return events if events.exists() else None


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
