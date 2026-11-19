"""Run the watch on this host until stopped: one serve per host and server.

A serve holds an ``flock`` on its lock file for as long as it runs, and writes its pid
there, so a second serve of the same server on this host refuses to start, and
``knowop watch-pr status`` says whether one runs. A serve on another host is no
bother: the rows' compare-and-set tells each piece of news once (:mod:`.watch`).

A serve with nothing to watch for :data:`IDLE_SEC` ends; ``knowop watch-pr add`` starts
one again when none runs.

:func:`detach` starts a serve that outlives the command that started it: in a session
of its own, its output to its log, holding none of the caller's pipes, so a tool call
that runs ``knowop watch-pr serve --detach`` returns at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import contextlib
import fcntl
import logging
import os
import subprocess
import time

from trackinizer.client.errors import ClientError
from trackinizer.lib.posix.pids import locked
from trackinizer.lib.userdirs import state_dir

from knowop.service import server_tag
from knowop.watch_pr.watch import POLL_SEC, Watch


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence
    from threading import Event

    from knowop.watch_pr.watch import WatchClient


__all__ = [
    "IDLE_SEC",
    "ServeFiles",
    "ServingError",
    "detach",
    "serve",
    "serve_files",
    "serving",
]

IDLE_SEC: Final = 3 * 3600.0
"""Seconds a serve with nothing to watch keeps running, as Gradle's daemon and Bazel's
server end after 3 idle hours; the next ``knowop watch-pr add`` starts one again."""

_logger = logging.getLogger(__name__)

_START_SEC: Final = 10.0
"""The longest :func:`detach` waits for the serve it started to hold its lock."""


class ServingError(Exception):
    """A serve could not start; the message says why."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ServeFiles:
    """Where a serve of one server keeps its lock and its log on this host.

    Attributes:
      lock: The file it holds an ``flock`` on, which names its pid.
      log: What a detached serve writes.

    """

    lock: Path
    log: Path


def serve_files(server: str) -> ServeFiles:
    """Return where a serve keeps its lock and its log on this host.

    Args:
      server: The server it serves, whose own files these are.

    Returns:
      files: The lock and the log, in knowop's state folder.

    """
    folder = state_dir() / "rekursiv-ai" / "knowop" / "watch-pr"
    tag = server_tag(server)
    return ServeFiles(
        lock=folder / f"serve-{tag}.lock",
        log=folder / f"serve-{tag}.log",
    )


def serving(files: ServeFiles) -> int | None:
    """Return the pid of the serve that holds ``files``' lock; ``None`` for none."""
    if not locked(files.lock):
        return None
    text = files.lock.read_text().strip()
    return int(text) if text.isdigit() else None


def serve(
    client: Callable[[], WatchClient],
    *,
    files: ServeFiles,
    host: str,
    stop: Event,
    poll_sec: float = POLL_SEC,
    idle_sec: float = IDLE_SEC,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Run passes of the watch until ``stop`` is set, or nothing is watched for long.

    A failed pass costs one pass, and counts neither as idle nor as busy.

    Args:
      client: Makes the server's client, once.
      files: Where this serve holds its lock.
      host: This host's name, which the rows it writes name.
      stop: Set to end the serve after the pass it is in.
      poll_sec: Seconds between two passes.
      idle_sec: Seconds of passes that read nothing after which the serve ends.
      clock: Monotonic seconds, which time the idle.

    Raises:
      ServingError: Another serve of this server runs on this host.

    """
    with _held(files.lock):
        watch = Watch(client=client(), host=host)
        busy = clock()
        while not stop.is_set():
            try:
                read = watch.sweep()
            except (ClientError, OSError) as error:
                _logger.warning("knowop watch-pr: pass failed: %s", error)
            else:
                _logger.info("knowop watch-pr: read %d pull requests", read)
                if read:
                    busy = clock()
                elif clock() - busy >= idle_sec:
                    _logger.info(
                        "knowop watch-pr: nothing watched for %d s; the serve ends",
                        idle_sec,
                    )
                    return
            _ = stop.wait(poll_sec)


def detach(
    command: Sequence[str],
    *,
    files: ServeFiles,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Start ``command``, a serve, on its own; return its pid once it serves.

    Args:
      command: The argv that runs a serve in the foreground.
      files: Where that serve holds its lock and writes its log.
      sleep: Waits between two looks at the lock.
      clock: Monotonic seconds.

    Returns:
      pid: The serve's, once it holds the lock: this one's, or one that ran already.

    Raises:
      ServingError: The serve ended, or did not hold the lock in time; its log says why.

    """
    if (running := serving(files)) is not None:
        return running
    files.log.parent.mkdir(parents=True, exist_ok=True)
    with files.log.open("ab") as log:
        started = subprocess.Popen(  # noqa: S603 -- knowop's own argv, no shell.
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            # Not the caller's folder, which may be a worktree deleted under it.
            cwd=files.log.parent,
        )
    deadline = clock() + _START_SEC
    while (pid := serving(files)) is None:
        if started.poll() is not None or clock() > deadline:
            raise ServingError(
                f"knowop watch-pr serve did not start; see {files.log}",
            )
        sleep(0.1)
    return pid


@contextlib.contextmanager
def _held(lock: Path) -> Generator[None]:
    """Hold ``lock`` with this pid in it, or refuse when another process holds it."""
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            held = os.pread(fd, 32, 0).decode().strip()
            raise ServingError(
                f"knowop watch-pr serve runs on this host already (pid {held})",
            ) from None
        os.ftruncate(fd, 0)
        _ = os.pwrite(fd, f"{os.getpid()}\n".encode(), 0)
        yield
    finally:
        os.close(fd)
