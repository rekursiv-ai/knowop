"""One serve per host: its lock, its passes until stopped, and a detached start."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import fcntl
import os
import subprocess
import sys
import threading

from trackinizer.client.errors import ClientError

import pytest

from knowop.watch_pr.rows_test import Server
from knowop.watch_pr.serve import (
    ServeFiles,
    ServingError,
    detach,
    serve,
    serve_files,
    serving,
)


if TYPE_CHECKING:
    from pathlib import Path


def files(tmp_path: Path) -> ServeFiles:
    return ServeFiles(
        lock=tmp_path / "w" / "serve.lock",
        log=tmp_path / "w" / "serve.log",
    )


class Stop(threading.Event):
    """Set once ``passes`` waits went by, each wait one pass."""

    def __init__(self, passes: int) -> None:
        super().__init__()
        self.passes = passes
        self.waits: list[float | None] = []

    @override
    def wait(self, timeout: float | None = None) -> bool:
        self.waits.append(timeout)
        if len(self.waits) >= self.passes:
            self.set()
        return self.is_set()


def test_a_serve_of_a_server_keeps_its_files_apart_from_anothers() -> None:
    one, two = serve_files("https://a.test"), serve_files("https://b.test")
    assert one.lock.parent == one.log.parent == two.lock.parent
    assert one.lock.name.startswith("serve-")
    assert one.lock.suffix == ".lock"
    assert one.log.suffix == ".log"
    assert one.lock != two.lock


def test_a_serve_runs_passes_until_stopped_and_names_its_pid(tmp_path: Path) -> None:
    held = files(tmp_path)
    seen: list[int | None] = []
    server = Server()

    def client() -> Server:
        seen.append(serving(held))
        return server

    stop = Stop(passes=2)
    serve(client, files=held, host="laptop", stop=stop, poll_sec=5.0)
    assert stop.waits == [5.0, 5.0]
    assert seen == [os.getpid()]
    assert held.lock.read_text() == f"{os.getpid()}\n"
    assert serving(held) is None


def test_a_pass_that_fails_costs_one_pass(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = Server()
    server.down = ClientError("GET -> 502", status_code=502)

    def client() -> Server:
        return server

    serve(client, files=files(tmp_path), host="laptop", stop=Stop(passes=2))
    assert [each.getMessage() for each in caplog.records].count(
        "knowop watch-pr: pass failed: GET -> 502",
    ) == 2


def test_a_second_serve_on_this_host_is_refused(tmp_path: Path) -> None:
    held = files(tmp_path)
    refused: list[str] = []

    def client() -> Server:
        with pytest.raises(ServingError, match=f"pid {os.getpid()}") as raised:
            serve(Server, files=held, host="laptop", stop=Stop(passes=1))
        refused.append(str(raised.value))
        return Server()

    serve(client, files=held, host="laptop", stop=Stop(passes=1))
    assert len(refused) == 1


# A child that takes the lock as a serve does, then waits to be killed.
_HOLDER = (
    "import fcntl, os, sys, time\n"
    "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT)\n"
    "fcntl.flock(fd, fcntl.LOCK_EX)\n"
    "os.write(fd, f'{os.getpid()}\\n'.encode())\n"
    "time.sleep(60)\n"
)


def test_a_detached_serve_is_started_once_and_outlives_the_call(tmp_path: Path) -> None:
    held = files(tmp_path)
    command = [sys.executable, "-c", _HOLDER, str(held.lock)]
    pid = detach(command, files=held)
    try:
        assert pid != os.getpid()
        assert serving(held) == pid
        assert detach(["false"], files=held) == pid
        assert os.getsid(pid) == pid
    finally:
        os.kill(pid, 9)


def test_a_detached_serve_that_ends_says_where_its_log_is(tmp_path: Path) -> None:
    held = files(tmp_path)
    with pytest.raises(ServingError, match=r"see .*serve\.log"):
        _ = detach([sys.executable, "-c", "print('no gh')"], files=held)
    assert held.log.read_text() == "no gh\n"


def test_a_detached_serve_that_never_serves_is_given_up(tmp_path: Path) -> None:
    held = files(tmp_path)
    ticks = iter([0.0, 5.0, 11.0])
    slept: list[float] = []
    started: list[object] = []

    class Running:
        def __init__(self, command: list[str], **options: object) -> None:
            del options
            started.append(command)

        def poll(self) -> None:
            return None

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(subprocess, "Popen", Running)
        with pytest.raises(ServingError, match="did not start"):
            _ = detach(
                ["knowop"],
                files=held,
                sleep=slept.append,
                clock=lambda: next(ticks),
            )
    assert slept == [0.1]
    assert started == [["knowop"]]


def test_a_held_lock_that_names_no_pid_yet_names_none(tmp_path: Path) -> None:
    held = files(tmp_path)
    held.lock.parent.mkdir(parents=True)
    fd = os.open(held.lock, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        assert serving(held) is None
    finally:
        os.close(fd)


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
