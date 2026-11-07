"""Host bookkeeping over process and socket fakes; private/detach_parity.py is live."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal
from unittest.mock import Mock

import contextlib
import errno
import itertools
import json
import os
import shlex
import socket
import stat
import subprocess
import sys
import time

from trackinizer.lib.posix.detached import Readiness, await_ready
from trackinizer.lib.posix.sockets import UNIX_SOCKET_PATH_MAX_BYTES, socket_address
from treekle import from_plain, json_loads, to_plain

import pytest

from knowop.agent import detach, headless, session
from knowop.agent.custom_types import HostRequest, RunRequest
from knowop.agent.detach import (
    HeadlessSpec,
    HostRecord,
    main,
)
from knowop.agent.headless import Reply, Request


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping

    from trackinizer.lib.posix.host import HostSpec


# A slug as long as the campaign names whose launch failed when tmux could not
# bind its socket ("File name too long"), and then some.
_LONG_NAME = "nightly-regression-sweep-shards-20260101-0900-" + "x" * 80


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.on_sleep: Callable[[], None] | None = None

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep()


class _Hosts:
    """The process and socket table at detach's external boundaries."""

    def __init__(self) -> None:
        self.clock = _Clock()
        self._listening: set[Path] = set()
        self._live: set[int] = set()
        self._runs: dict[Path, _Control] = {}
        self._pids = itertools.count((1 << 30) + 1)

    def start(self) -> int:
        pid = next(self._pids)
        self._live.add(pid)
        return pid

    def end(self, pid: int) -> None:
        self._live.discard(pid)

    def accepts(self, address: Path, *, timeout_sec: float = 0.5) -> bool:
        del timeout_sec
        return address in self._listening or address in self._runs

    def running(self, pid: int) -> bool:
        return pid == os.getpid() or pid in self._live

    def request(
        self,
        address: Path,
        op: Literal["state", "takeover", "stop"],
        *,
        timeout_sec: float = 60.0,
    ) -> Reply:
        del timeout_sec
        return self._run(address).answer(Request(op=op))

    def say(self, address: Path, text: str, *, timeout_sec: float = 60.0) -> Reply:
        del timeout_sec
        return self._run(address).answer(Request(op="send", text=text))

    @contextlib.contextmanager
    def listen(self, path: Path) -> Generator[Path]:
        address = socket_address(path)
        self._listening.add(address)
        try:
            yield address
        finally:
            self._listening.discard(address)

    @contextlib.contextmanager
    def serve(
        self,
        path: Path,
        reply: Reply | Mapping[str, Reply],
        steps: list[str] | None = None,
    ) -> Generator[_Control]:
        control = _Control(socket_address(path), reply, steps)
        self._runs[control.address] = control
        try:
            yield control
        finally:
            del self._runs[control.address]

    def _run(self, address: Path) -> _Control:
        control = self._runs.get(address)
        if control is None:
            raise ConnectionRefusedError(errno.ECONNREFUSED, "refused", str(address))
        return control


@pytest.fixture(autouse=True)
def hosts(monkeypatch: pytest.MonkeyPatch) -> _Hosts:
    """Replace liveness, socket calls and polling time with the host table."""
    table = _Hosts()
    monkeypatch.setattr(detach, "accepts", table.accepts)
    monkeypatch.setattr(detach, "running", table.running)
    monkeypatch.setattr(detach, "time", table.clock)
    monkeypatch.setattr(headless, "request", table.request)
    monkeypatch.setattr(headless, "say", table.say)
    return table


@pytest.fixture
def live_child(hosts: _Hosts) -> Generator[int]:
    """Yield a synthetic live pid, then end it."""
    pid = hosts.start()
    try:
        yield pid
    finally:
        hosts.end(pid)


class TestNames:
    """A host name becomes a directory, so it must be one path component."""

    @pytest.mark.parametrize(
        "name",
        ["", "../up", "a/b", ".hidden", "_under", "x" * 129],
    )
    def test_rejects_a_name_that_is_not_a_filename(self, name: str) -> None:
        with pytest.raises(SystemExit, match="invalid host name"):
            _ = main(["log", name])

    @pytest.mark.parametrize("name", ["scientist#2", "lead@lab", _LONG_NAME])
    def test_accepts_routing_names_and_long_slugs(self, name: str) -> None:
        with pytest.raises(SystemExit, match="no host named"):
            _ = main(["log", name])


class TestSocketAddress:
    """A long name must not cost the host its socket, as it did under tmux."""

    def test_a_long_name_still_binds(self) -> None:
        root = detach._paths(_LONG_NAME)
        root.mkdir(parents=True)
        logical = root / "host.sock"
        assert len(os.fsencode(logical)) > 103, "the regression needs a long path"
        address = socket_address(logical)
        assert len(os.fsencode(address)) <= UNIX_SOCKET_PATH_MAX_BYTES
        assert address.name == logical.name
        assert address.parent.resolve() == root.resolve()


class TestRecords:
    """``host.json``: what each command reads to find and judge a host."""

    def test_round_trips(self, tmp_path: Path) -> None:
        record = _record(tmp_path, state="exited", exit_code=3)
        detach._write_record(tmp_path, record)
        assert detach._read_record(tmp_path) == record

    @pytest.mark.parametrize(
        "tagged_type",
        [
            "loop." + "knowop.client.agent.detach.HostRecord",
            ("loop." + "experimental.") + "knowop.client.agent.detach.HostRecord",
        ],
    )
    def test_reads_records_tagged_with_a_retired_type_path(
        self,
        tmp_path: Path,
        tagged_type: str,
    ) -> None:
        """Host records from either former module path keep every field."""
        text = (
            '{"py/object":"__HOST_TYPE__","name":"legacy",'
            '"command":"python worker","cwd":"/srv/work","pid":42,'
            '"address":"/var/run/host.sock","started":"2026-10-03T00:00:00+00:00",'
            '"state":"exited","exit_code":7,'
            '"ended":"2026-10-03T00:01:00+00:00","machine":"host-a",'
            '"control":"/var/run/control.sock","session":"session-1"}'
        ).replace("__HOST_TYPE__", tagged_type)
        (tmp_path / "host.json").write_text(text)

        assert detach._read_record(tmp_path) == HostRecord(
            name="legacy",
            command="python worker",
            cwd="/srv/work",
            pid=42,
            address="/var/run/host.sock",
            started="2026-10-03T00:00:00+00:00",
            state="exited",
            exit_code=7,
            ended="2026-10-03T00:01:00+00:00",
            machine="host-a",
            control="/var/run/control.sock",
            session="session-1",
        )

    def test_an_unreadable_record_is_absent(self, tmp_path: Path) -> None:
        (tmp_path / "host.json").write_text("{not json")
        assert detach._read_record(tmp_path) is None

    def test_the_record_is_json(self, tmp_path: Path) -> None:
        detach._write_record(tmp_path, _record(tmp_path))
        assert json_loads((tmp_path / "host.json").read_text(), HostRecord).name == "h"


class TestState:
    """The record's word, checked against whether anything still answers."""

    def test_exited_reports_the_status(self, tmp_path: Path) -> None:
        assert (
            detach._state(_record(tmp_path, state="exited", exit_code=7)) == "exited 7"
        )

    def test_a_listening_host_is_running(self, tmp_path: Path, hosts: _Hosts) -> None:
        with hosts.listen(tmp_path / "s") as address:
            record = _record(tmp_path, address=address)
            assert detach._state(record) == "running"

    def test_a_live_host_that_stopped_listening_is_exiting(
        self,
        tmp_path: Path,
        live_child: int,
    ) -> None:
        record = _record(tmp_path, state="running", pid=live_child)
        assert detach._state(record) == "exiting"

    def test_a_live_host_not_yet_listening_is_starting(
        self,
        tmp_path: Path,
        live_child: int,
    ) -> None:
        record = _record(tmp_path, state="starting", pid=live_child)
        assert detach._state(record) == "starting"

    def test_a_vanished_host_is_lost(self, tmp_path: Path) -> None:
        """Killed outright, a host never records ``exited``."""
        record = _record(tmp_path, state="running", pid=_dead_pid())
        assert detach._state(record) == "lost"


class TestTail:
    """What the launcher repeats from a host's log."""

    def test_the_last_lines_are_kept(self, tmp_path: Path) -> None:
        log = tmp_path / "host.log"
        log.write_text("one\ntwo\nthree\n")

        assert detach._tail(log, lines=2) == "two\nthree\n"

    def test_an_unreadable_log_says_nothing(self, tmp_path: Path) -> None:
        assert detach._tail(tmp_path / "absent.log") == ""


class TestExitStatus:
    """A ``SystemExit`` raised inside the host, as the status it reports."""

    def test_none_is_success(self) -> None:
        assert detach._exit_status(None) == 0

    def test_an_int_is_the_status(self) -> None:
        assert detach._exit_status(3) == 3

    def test_a_message_is_logged_and_fails(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The launcher shows the host log, so the reason must reach it."""
        assert detach._exit_status("knowop run: claude not found in PATH") == 1
        assert capsys.readouterr().err == "knowop run: claude not found in PATH\n"


def test_launch_execs_a_new_interpreter_without_forking_python(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(os, "fork", _forbid_fork)
    spawned: list[tuple[HostRequest, Path]] = []
    monkeypatch.setattr(detach, "_spawn", partial(_spawn_accepting, spawned))
    argv = [sys.executable, "-c", "input()"]

    assert (
        session.main(
            ["--detach", "--no-sync", "--name", "exec-host", "sh", "--", *argv],
        )
        == 0
    )
    assert "exec-host is running detached" in capsys.readouterr().err
    [(request, log)] = spawned
    paths = detach._paths("exec-host")
    assert (request["paths"], log) == (str(paths), paths / "host.log")
    assert request["command"] == shlex.join(["sh", *argv])
    assert (request["run"]["cli_argv"], request["run"]["run_name"]) == (
        argv,
        "exec-host",
    )
    assert request["run"]["flags"]["sync"] is False


def _spawn_accepting(
    spawned: list[tuple[HostRequest, Path]],
    request: HostRequest,
    *,
    ready: int,
    log: Path,
) -> None:
    spawned.append((request, log))
    _ = os.write(ready, b"ok\n")


def _forbid_fork() -> int:
    raise AssertionError("detached hosts must exec, never fork and run Python")


def test_spawn_passes_the_payload_and_readiness_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[tuple[list[str], Path, tuple[int, ...], bytes]] = []
    monkeypatch.setattr(detach, "spawn_detached", partial(_spawn_noted, started))
    request = HostRequest(
        paths=str(tmp_path),
        address=str(tmp_path / "host.sock"),
        control=str(tmp_path / "control.sock"),
        resume="sess-1",
        command="sh",
        ready=9,
        run=_run_request("host"),
    )

    detach._spawn(request, ready=9, log=tmp_path / "host.log")

    [(argv, log, pass_fds, payload)] = started
    assert argv[:4] == [sys.executable, "-m", "knowop.agent", "--_host"]
    assert (log, pass_fds) == (tmp_path / "host.log", (int(argv[4]), 9))
    assert json_loads(payload.decode(), HostRequest) == request
    with pytest.raises(OSError, match="Bad file descriptor"):
        _ = os.fstat(int(argv[4]))


def _spawn_noted(
    started: list[tuple[list[str], Path, tuple[int, ...], bytes]],
    argv: list[str],
    *,
    log: Path,
    pass_fds: tuple[int, ...],
) -> int:
    started.append((argv, log, pass_fds, os.pread(int(argv[4]), 1 << 16, 0)))
    return _dead_pid()


def test_a_failed_spawn_closes_both_ends_of_the_ready_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptors = os.pipe()
    pipe = Mock(return_value=descriptors)
    monkeypatch.setattr(os, "pipe", pipe)
    monkeypatch.setattr(subprocess, "Popen", Mock(side_effect=OSError("cannot exec")))

    with pytest.raises(OSError, match="cannot exec"):
        _ = session.main(["--detach", "--no-sync", "--name", "failed-exec", "sh"])

    pipe.assert_called_once_with()
    for fd in descriptors:
        with pytest.raises(OSError, match="Bad file descriptor"):
            _ = os.fstat(fd)


def test_run_host_reads_the_payload_closes_it_and_runs_its_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The child rebuilds its start from the request's run, and hosts with it."""
    request = HostRequest(
        paths=str(tmp_path),
        address=str(tmp_path / "host.sock"),
        control=str(tmp_path / "control.sock"),
        resume="sess-1",
        command="sh",
        ready=9,
        run=_run_request("host"),
    )
    read_fd, write_fd = os.pipe()
    _ = os.write(write_fd, json.dumps(to_plain(request)).encode())
    os.close(write_fd)
    hosted: list[tuple[Path, dict[str, object]]] = []
    monkeypatch.setattr(detach, "_detach", partial(_detach_noted, hosted))
    rebuilt: list[RunRequest] = []

    def rebuild(run: RunRequest) -> Callable[[HostSpec | None], int]:
        rebuilt.append(run)
        return _start

    assert detach.run_host(read_fd, rebuild=rebuild) == 7
    assert rebuilt == [request["run"]]
    [(paths, kwargs)] = hosted
    assert paths == tmp_path
    assert kwargs == {
        "address": tmp_path / "host.sock",
        "control": tmp_path / "control.sock",
        "resume": "sess-1",
        "command": "sh",
        "ready": 9,
        "start": _start,
    }
    with pytest.raises(OSError, match="Bad file descriptor"):
        _ = os.fstat(read_fd)


def _start(spec: HostSpec | None) -> int:
    raise AssertionError(spec)


def _detach_noted(
    hosted: list[tuple[Path, dict[str, object]]],
    paths: Path,
    **kwargs: object,
) -> int:
    hosted.append((paths, kwargs))
    return 7


def launch(name: str, *, command: str, start: Callable[[HostSpec], int]) -> int:
    """Exercise host bookkeeping with an in-process runner at the spawn boundary."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(detach, "_spawn", partial(_spawn_here, start))
        return detach.launch(name, command=command, request=_run_request(name))


def _run_request(name: str) -> RunRequest:
    """Return the plain inputs of a run named ``name`` that wraps nothing."""
    return RunRequest(
        flags={},
        cli_argv=[],
        connection=None,
        resume_path=None,
        cli_session_id=None,
        profile=None,
        attach=None,
        prefix_bytes=0,
        compact_after_load=False,
        run_name=name,
    )


def _spawn_refused(
    spawned: list[tuple[HostRequest, Path]],
    request: HostRequest,
    *,
    ready: int,
    log: Path,
) -> None:
    """Note what a spawn was asked for, then fail as an exec would."""
    del ready
    spawned.append((request, log))
    raise OSError("cannot exec")


def _spawn_silent(
    held: list[int],
    request: HostRequest,
    *,
    ready: int,
    log: Path,
) -> None:
    """Stand in for a host that starts and never answers: hold its pipe open."""
    del request, log
    held.append(os.dup(ready))


def _spawn_here(
    start: Callable[[HostSpec], int],
    request: HostRequest,
    *,
    ready: int,
    log: Path,
) -> None:
    del log
    saved = [os.dup(fd) for fd in (0, 1, 2)]
    try:
        _ = detach._detach(
            Path(request["paths"]),
            address=Path(request["address"]),
            control=Path(request["control"]),
            resume=request["resume"],
            command=request["command"],
            ready=os.dup(ready),
            start=start,
        )
    finally:
        for fd, copy in enumerate(saved):
            _ = os.dup2(copy, fd)
            os.close(copy)


class TestTheHost:
    """The host, run in this process: its record, its status, its log."""

    @pytest.mark.parametrize("how", ["returns", "refuses", "fails"])
    def test_the_launchers_child_records_the_host_and_exits_with_its_status(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
        how: str,
    ) -> None:
        """The host's stdio stays the test's."""
        start, status, logged = _HOST_ENDS[how]
        redirected: list[Path] = []
        monkeypatch.setattr(detach, "redirect_stdio", redirected.append)
        paths = detach._paths("upsilon")
        paths.mkdir(parents=True)
        read_fd, write_fd = os.pipe()
        try:
            actual = detach._detach(
                paths,
                address=socket_address(paths / "host.sock"),
                control=socket_address(paths / "control.sock"),
                resume="",
                command="sh -- true",
                ready=write_fd,
                start=start,
            )
        finally:
            os.close(read_fd)

        assert actual == status
        assert redirected == [paths / "host.log"]
        record = detach._read_record(paths)
        assert record is not None
        assert (record.state, record.exit_code, record.command) == (
            "exited",
            status,
            "sh -- true",
        )
        assert record.pid == os.getpid()
        assert record.cwd == str(Path.cwd())
        assert record.ended is not None
        assert logged in capsys.readouterr().err + caplog.text

    def test_listening_records_the_host_running_and_releases_the_launcher(
        self,
        tmp_path: Path,
    ) -> None:
        read_fd, write_fd = os.pipe()

        detach._listening(tmp_path, _record(tmp_path), Readiness(write_fd))

        record = detach._read_record(tmp_path)
        assert record is not None
        assert record.state == "running"
        assert await_ready(read_fd, timeout_sec=1.0) is True


class TestCommands:
    """``knowop run ls|attach|log|send|stop`` against recorded hosts."""

    def test_ls_lists_the_live_hosts_one_line_each_and_all_lists_every_host(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # A host directory with no record yet, listed first, is passed over.
        detach._paths("a-new").mkdir(parents=True)
        alpha = detach._paths("alpha")
        alpha.mkdir(parents=True)
        detach._write_record(
            alpha,
            _record(alpha, name="alpha", state="exited", exit_code=0),
        )
        beta = detach._paths("beta")
        beta.mkdir(parents=True)
        # This test's own pid: a starting host whose process is still there, running
        # a script that spans lines.
        detach._write_record(
            beta,
            replace(
                _record(beta, name="beta", state="starting", pid=os.getpid()),
                command="bash -c 'echo a\necho b'",
            ),
        )

        assert main(["ls"]) == 0
        live = capsys.readouterr().out.splitlines()[1:]
        assert main(["ls", "--all"]) == 0
        every = capsys.readouterr().out.splitlines()[1:]

        assert [line.split()[:2] for line in live] == [["beta", "starting"]]
        assert live[0].endswith("bash -c 'echo a echo b'")
        assert [line.split()[:2] for line in every] == [
            ["alpha", "exited"],
            ["beta", "starting"],
        ]

    def test_the_live_hosts_are_this_computers_hosts_still_there(self) -> None:
        here = socket.gethostname()
        for name, state, machine in (
            ("alpha", "exited", here),
            ("beta", "starting", here),
            ("gamma", "starting", f"not-{here}"),
            ("delta", "starting", ""),
        ):
            root = detach._paths(name)
            root.mkdir(parents=True)
            detach._write_record(
                root,
                _record(root, name=name, state=state, pid=os.getpid(), machine=machine),
            )

        assert [host.name for host in detach.live_hosts()] == ["beta", "delta"]

    @pytest.mark.parametrize(("failed", "status"), [(set[str](), 0), ({"beta"}, 1)])
    def test_stop_all_stops_each_live_host_and_fails_when_one_did_not_stop(
        self,
        monkeypatch: pytest.MonkeyPatch,
        failed: set[str],
        status: int,
    ) -> None:
        for name in ("beta", "delta"):
            root = detach._paths(name)
            root.mkdir(parents=True)
            detach._write_record(
                root,
                _record(root, name=name, state="starting", pid=os.getpid()),
            )
        stopped: list[str] = []

        def stop(name: str) -> int:
            stopped.append(name)
            return int(name in failed)

        monkeypatch.setattr(detach, "_stop", stop)

        assert main(["stop", "--all"]) == status
        assert stopped == ["beta", "delta"]

    def test_stop_all_with_no_hosts_succeeds_and_says_so(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert main(["stop", "--all"]) == 0
        assert capsys.readouterr().err == (
            "[knowop run] no host is running on this computer\n"
        )

    def test_stop_names_a_host_or_stops_them_all(self) -> None:
        with pytest.raises(SystemExit, match="name a host, or stop them all"):
            _ = main(["stop"])

    def test_ls_reads_the_directory_hosts_are_made_in(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Compared as text: a case-insensitive disk finds hosts under any spelling."""
        made = detach._paths("alpha").parent
        asked: list[Path] = []
        monkeypatch.setattr(Path, "is_dir", _noting_is_dir(asked, Path.is_dir))
        assert main(["ls"]) == 0
        assert made in asked

    def test_ls_all_lists_each_host(self, capsys: pytest.CaptureFixture[str]) -> None:
        root = detach._paths("alpha")
        root.mkdir(parents=True)
        detach._write_record(
            root,
            _record(root, name="alpha", state="exited", exit_code=0),
        )
        assert main(["ls", "--all"]) == 0
        header, row = capsys.readouterr().out.splitlines()
        assert row.split() == [
            "alpha",
            "exited",
            "0",
            "1",
            "2026-10-03T00:00:00+00:00",
            "sh",
            "--",
            "true",
        ]
        # Columns line up under their headings, left-aligned.
        assert [header.index(title) for title in ("STATE", "PID", "COMMAND")] == [
            row.index("exited"),
            row.index(" 1 ") + 1,
            row.index("sh --"),
        ]

    def test_ls_with_no_hosts_prints_the_header(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert main(["ls"]) == 0
        assert capsys.readouterr().out.split() == [
            "NAME",
            "STATE",
            "PID",
            "STARTED",
            "COMMAND",
        ]

    def test_log_prints_the_scrollback(
        self,
        capfdbinary: pytest.CaptureFixture[bytes],
    ) -> None:
        root = detach._paths("beta")
        root.mkdir(parents=True)
        detach._write_record(
            root,
            _record(root, name="beta", state="exited", exit_code=0),
        )
        (root / "scrollback.log").write_bytes(b"\x1b[1mbold\x1b[0m\r\n")
        assert main(["log", "beta"]) == 0
        assert capfdbinary.readouterr().out == b"\x1b[1mbold\x1b[0m\r\n"

    @pytest.mark.parametrize("command", ["attach", "send", "stop"])
    def test_a_host_that_is_gone_is_named_as_such(self, command: str) -> None:
        root = detach._paths("gamma")
        root.mkdir(parents=True)
        detach._write_record(
            root,
            _record(root, name="gamma", state="running", pid=_dead_pid()),
        )
        argv = [command, "gamma", *(["two", "words"] if command == "send" else [])]
        with pytest.raises(SystemExit, match="'gamma' is lost"):
            _ = main(argv)

    def test_help_names_every_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exited:
            _ = main(["--help"])
        assert exited.value.code == 0
        out = " ".join(capsys.readouterr().out.split())
        for line in (
            "ls List the live hosts and their state.",
            "attach Drive a host from this terminal; Ctrl-\\ detaches.",
            "log Print a host's scrollback.",
            "send Type a message into a host and press Enter.",
            "stop Stop a host and wait for it.",
        ):
            assert line in out

    def test_a_command_is_required(self) -> None:
        with pytest.raises(SystemExit) as exited:
            _ = main([])
        assert exited.value.code == 2

    @pytest.mark.parametrize("flag", ["-f", "--follow"])
    def test_following_an_exited_host_prints_it_all_and_returns(
        self,
        capfdbinary: pytest.CaptureFixture[bytes],
        flag: str,
    ) -> None:
        root = detach._paths("epsilon")
        root.mkdir(parents=True)
        detach._write_record(
            root,
            _record(root, name="epsilon", state="exited", exit_code=0),
        )
        (root / "scrollback.log").write_bytes(b"done\r\n")
        assert main(["log", "epsilon", flag]) == 0
        assert capfdbinary.readouterr().out == b"done\r\n"

    def test_following_a_running_host_waits_for_more_until_it_stops_accepting(
        self,
        hosts: _Hosts,
        monkeypatch: pytest.MonkeyPatch,
        capfdbinary: pytest.CaptureFixture[bytes],
    ) -> None:
        """Each wait finds the bytes the host wrote; it reads once more after it ends."""
        root = detach._paths("zeta")
        root.mkdir(parents=True)
        scrollback = root / "scrollback.log"
        scrollback.write_bytes(b"one\r\n")
        with contextlib.ExitStack() as host:
            address = host.enter_context(hosts.listen(root / "s"))
            detach._write_record(root, _record(root, name="zeta", address=address))
            monkeypatch.setattr(
                hosts.clock,
                "sleep",
                partial(_write_then_close, scrollback, host),
            )

            assert main(["log", "zeta", "--follow"]) == 0

        assert capfdbinary.readouterr().out == b"one\r\ntwo\r\n"

    def test_a_host_with_no_scrollback_yet_prints_nothing(
        self,
        capfdbinary: pytest.CaptureFixture[bytes],
    ) -> None:
        root = detach._paths("phi")
        root.mkdir(parents=True)
        detach._write_record(root, _record(root, name="phi", state="starting"))

        assert main(["log", "phi"]) == 0
        assert capfdbinary.readouterr().out == b""

    def test_a_host_that_ends_before_accepting_fails_the_launch_with_its_log(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert launch("sigma", command="sh", start=_say_then_end) == 1

        assert capsys.readouterr().err == (
            "knowop run: host 'sigma' failed to start; its log follows\n"
            "[knowop run] claude not found\n"
        )

    def test_the_host_is_spawned_in_a_private_directory_with_its_socket_and_log(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spawned: list[tuple[HostRequest, Path]] = []
        monkeypatch.setattr(detach, "_spawn", partial(_spawn_refused, spawned))

        with pytest.raises(OSError, match="cannot exec"):
            _ = detach.launch("lambda", command="sh", request=_run_request("lambda"))

        paths = detach._paths("lambda")
        [(request, log)] = spawned
        assert stat.S_IMODE(paths.stat().st_mode) == 0o700
        assert request["address"] == str(socket_address(paths / "host.sock"))
        assert log == paths / "host.log"

    def test_a_host_silent_past_the_wait_is_reported_not_killed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        held: list[int] = []
        monkeypatch.setattr(detach, "_READY_TIMEOUT_SEC", 0.01)
        monkeypatch.setattr(detach, "_spawn", partial(_spawn_silent, held))
        try:
            status = detach.launch("kappa", command="sh", request=_run_request("kappa"))
        finally:
            for fd in held:
                os.close(fd)

        assert status == 1
        assert capsys.readouterr().err == (
            "knowop run: host 'kappa' did not accept in time; its log follows\n"
        )

    def test_launch_refuses_a_running_name(self, hosts: _Hosts) -> None:
        """Checked before forking, so a refusal never leaves a stray process."""
        root = detach._paths("delta")
        root.mkdir(parents=True)
        with hosts.listen(root / "s") as address:
            detach._write_record(root, _record(root, name="delta", address=address))
            with pytest.raises(SystemExit) as refused:
                _ = launch("delta", command="sh", start=_never)
            assert str(refused.value) == (
                "knowop run: host 'delta' is running; "
                "stop it (knowop run stop delta) or choose another --name"
            )

    def test_launch_repeats_what_the_host_said_before_it_accepted(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The host loads a chat after the fork; its line still reaches the person."""
        assert launch("eta", command="sh", start=_say_then_accept) == 0
        said, up = capsys.readouterr().err.splitlines()
        assert said == "[knowop run] loaded 2 records"
        assert (
            up
            == "[knowop run] eta is running detached; attach with: knowop run attach eta"
        )


class TestAHeadlessHostIsRecordedAndFound:
    """A chat's run has no terminal until taken over; its record says how to reach it."""

    def test_the_host_is_told_where_the_control_socket_goes(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert launch("theta", command="sh", start=_say_the_spec) == 0
        said = capsys.readouterr().err.splitlines()[0]
        control = socket_address(detach._paths("theta") / "control.sock")
        assert said == f"[knowop run] control {control} resume ''"

    def test_the_record_names_the_machine_and_the_socket(self) -> None:
        assert launch("iota", command="sh", start=_say_the_spec) == 0
        record = detach._read_record(detach._paths("iota"))
        assert record is not None
        assert record.machine == socket.gethostname()
        assert record.control == str(
            socket_address(detach._paths("iota") / "control.sock"),
        )

    def test_a_host_listening_on_its_control_socket_alone_is_running(
        self,
        tmp_path: Path,
        hosts: _Hosts,
    ) -> None:
        with hosts.serve(tmp_path / "c", Reply(ok=True)) as control:
            record = _record(tmp_path, control=control.address)
            assert detach._state(record) == "running"

    def test_a_control_socket_nothing_listens_on_does_not_make_a_host_run(
        self,
        tmp_path: Path,
        live_child: int,
    ) -> None:
        record = _record(
            tmp_path,
            state="running",
            pid=live_child,
            control=tmp_path / "dead.sock",
        )
        assert detach._state(record) == "exiting"

    def test_a_record_from_before_headless_runs_still_reads(
        self,
        tmp_path: Path,
    ) -> None:
        written = from_plain(to_plain(_record(tmp_path)), dict[str, object])
        old = {
            key: value
            for key, value in written.items()
            if key not in {"machine", "control"}
        }
        (tmp_path / "host.json").write_text(json.dumps(old))
        record = detach._read_record(tmp_path)
        assert record is not None
        assert (record.machine, record.control) == ("", "")


class TestTakingAHeadlessRunOver:
    """``knowop run attach`` stops the driver, then drives the CLI's own terminal."""

    def test_the_run_is_asked_to_hand_over_before_the_terminal_is_attached(
        self,
        tmp_path: Path,
        hosts: _Hosts,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        steps: list[str] = []
        with _headless(hosts, tmp_path, monkeypatch, steps, Reply(ok=True)) as control:
            assert main(["attach", "kappa"]) == 0
            assert [asked.op for asked in control.asked] == ["takeover"]
        assert steps == ["takeover", "attach"]

    def test_it_says_how_the_chat_gets_the_cli_back_once_the_terminal_exits(
        self,
        tmp_path: Path,
        hosts: _Hosts,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with _headless(hosts, tmp_path, monkeypatch, [], Reply(ok=True), status=0):
            assert main(["attach", "kappa"]) == 0
        assert capsys.readouterr().err == (
            "\n[knowop run] kappa exited with status 0\n"
            "[knowop run] Chat stays taken over; use Reattach to Chat in the chat or the Console to give it back.\n"
        )

    def test_the_status_the_terminal_ended_with_is_the_commands(
        self,
        tmp_path: Path,
        hosts: _Hosts,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with _headless(hosts, tmp_path, monkeypatch, [], Reply(ok=True), status=3):
            assert main(["attach", "kappa"]) == 3

    def test_a_terminal_left_with_ctrl_backslash_leaves_the_cli_running(
        self,
        tmp_path: Path,
        hosts: _Hosts,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with _headless(hosts, tmp_path, monkeypatch, [], Reply(ok=True)):
            assert main(["attach", "kappa"]) == 0
        assert capsys.readouterr().err == (
            "\n[knowop run] detached from kappa; it is still running\n"
            "[knowop run] Chat stays taken over; use Reattach to Chat in the chat or the Console to give it back.\n"
        )

    def test_a_run_that_will_not_hand_over_is_not_attached(
        self,
        tmp_path: Path,
        hosts: _Hosts,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        steps: list[str] = []
        refusal = Reply(ok=False, reason="claude could not start")
        with (
            _headless(hosts, tmp_path, monkeypatch, steps, refusal),
            pytest.raises(SystemExit, match="cannot be taken over: claude could"),
        ):
            _ = main(["attach", "kappa"])
        assert steps == ["takeover"]

    def test_a_pty_host_is_attached_as_it_always_was(
        self,
        hosts: _Hosts,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        steps: list[str] = []
        _fake_attach(monkeypatch, steps, status=None)
        root = detach._paths("lambda")
        root.mkdir(parents=True)
        with hosts.listen(root / "s") as address:
            detach._write_record(
                root,
                _record(root, name="lambda", address=address, machine=_HERE),
            )
            assert main(["attach", "lambda"]) == 0
        assert steps == ["attach"]
        said = capsys.readouterr().err
        assert "detached from lambda; it is still running" in said
        assert "Reattach" not in said

    def test_a_host_of_another_machine_is_refused_with_its_name(
        self,
        tmp_path: Path,
        hosts: _Hosts,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        steps: list[str] = []
        with (
            _headless(
                hosts,
                tmp_path,
                monkeypatch,
                steps,
                Reply(ok=True),
                machine="elsewhere",
            ),
            pytest.raises(SystemExit) as refused,
        ):
            _ = main(["attach", "kappa"])
        said = str(refused.value)
        assert "runs on elsewhere" in said
        assert _HERE in said
        assert steps == []

    @pytest.mark.parametrize("command", ["send", "stop"])
    def test_stop_and_send_refuse_another_machines_host_too(
        self,
        command: str,
        tmp_path: Path,
        hosts: _Hosts,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        argv = [command, "kappa", *(["hi"] if command == "send" else [])]
        with (
            _headless(
                hosts,
                tmp_path,
                monkeypatch,
                [],
                Reply(ok=True),
                machine="elsewhere",
            ),
            pytest.raises(SystemExit, match="runs on elsewhere"),
        ):
            _ = main(argv)

    def test_a_record_that_names_no_machine_is_attached_anywhere(
        self,
        tmp_path: Path,
        hosts: _Hosts,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        steps: list[str] = []
        with _headless(hosts, tmp_path, monkeypatch, steps, Reply(ok=True), machine=""):
            assert main(["attach", "kappa"]) == 0
        assert steps == ["takeover", "attach"]


class TestReach:
    """A control call that fails reads as a host with no control socket."""

    @pytest.mark.parametrize("error", [OSError("refused"), ValueError("not a reply")])
    def test_a_failed_call_is_no_reply(self, tmp_path: Path, error: Exception) -> None:
        record = _record(tmp_path, control=tmp_path / "control.sock")

        assert detach._reach(record, partial(_raise, error)) is None


class TestReachingAHeadlessRun:
    """``knowop run stop|send`` and a second launch reach the run on its control socket."""

    def test_stop_ends_the_run_through_its_socket_and_waits_for_the_process(
        self,
        tmp_path: Path,
        hosts: _Hosts,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        stopped: list[Path] = []

        async def stop_terminal(address: Path) -> int | None:
            stopped.append(address)
            return 0

        monkeypatch.setattr(detach, "stop", stop_terminal)
        with _headless(hosts, tmp_path, monkeypatch, [], Reply(ok=True)) as control:
            assert main(["stop", "kappa"]) == 0
            assert [asked.op for asked in control.asked] == ["stop"]
        assert stopped == []
        assert capsys.readouterr().err == "[knowop run] kappa stopped\n"

    def test_a_host_with_a_terminal_of_its_own_is_stopped_through_it(
        self,
        hosts: _Hosts,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        stopped: list[Path] = []

        async def stop_terminal(address: Path) -> int | None:
            stopped.append(address)
            return 143

        monkeypatch.setattr(detach, "stop", stop_terminal)
        root = detach._paths("omicron")
        root.mkdir(parents=True)
        with hosts.listen(root / "s") as address:
            detach._write_record(
                root,
                _record(root, name="omicron", address=address, pid=_dead_pid()),
            )
            assert main(["stop", "omicron"]) == 0
        assert stopped == [address]
        assert capsys.readouterr().err == "[knowop run] omicron stopped (status 143)\n"

    def test_send_types_through_the_socket(
        self,
        tmp_path: Path,
        hosts: _Hosts,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with _headless(hosts, tmp_path, monkeypatch, [], Reply(ok=True)) as control:
            assert main(["send", "kappa", "two", "words"]) == 0
            assert [(asked.op, asked.text) for asked in control.asked] == [
                ("send", "two words"),
            ]

    @pytest.mark.parametrize(("taken", "code"), [(True, 0), (False, 1)])
    def test_a_host_with_a_terminal_of_its_own_takes_a_line_through_it(
        self,
        hosts: _Hosts,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        taken: bool,
        code: int,
    ) -> None:
        async def submit_line(address: Path, *, text: str) -> bool:
            del address
            return taken and text == "two words"

        monkeypatch.setattr(detach, "submit", submit_line)
        root = detach._paths("psi")
        root.mkdir(parents=True)
        detach._write_record(root, _record(root, name="psi"))
        with hosts.listen(root / "s") as address:
            detach._write_record(root, _record(root, name="psi", address=address))
            assert main(["send", "psi", "two", "words"]) == code
        assert capsys.readouterr().err == (
            "" if taken else "knowop run: psi did not take the message; it is exiting\n"
        )

    def test_a_host_that_will_not_exit_is_said_to_be_still_closing(
        self,
        live_child: int,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        async def stop_terminal(address: Path) -> int | None:
            del address
            return 0

        monkeypatch.setattr(detach, "stop", stop_terminal)
        monkeypatch.setattr(detach, "_STOP_TIMEOUT_SEC", 0.05)
        root = detach._paths("omega")
        root.mkdir(parents=True)
        detach._write_record(
            root,
            _record(root, name="omega", pid=live_child),
        )
        assert main(["stop", "omega"]) == 1
        assert capsys.readouterr().err == (
            f"knowop run: omega stopped its CLI but the host (pid {live_child}) "
            "is still closing after 0s\n"
        )

    def test_a_line_the_run_refuses_is_said_and_fails(
        self,
        tmp_path: Path,
        hosts: _Hosts,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        refusal = Reply(ok=False, reason="taken over in a terminal")
        with _headless(hosts, tmp_path, monkeypatch, [], refusal):
            assert main(["send", "kappa", "hi"]) == 1
        assert capsys.readouterr().err == (
            "knowop run: kappa did not take the message: taken over in a terminal\n"
        )


class TestALaunchOfATakenOverName:
    """A taken-over run is replaced by the launch that gives the chat its CLI back."""

    def test_the_run_is_ended_and_the_new_one_resumes_its_session(
        self,
        tmp_path: Path,
        hosts: _Hosts,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root = detach._paths("mu")
        root.mkdir(parents=True)
        replies = {
            "state": Reply(ok=True, state="taken_over", cli="claude", session="sess-1"),
            "stop": Reply(ok=True),
        }
        with hosts.serve(tmp_path / "c", replies) as control:
            detach._write_record(
                root,
                _record(
                    root,
                    name="mu",
                    control=control.address,
                    pid=_dead_pid(),
                    command="claude",
                ),
            )
            assert launch("mu", command="claude", start=_say_the_spec) == 0
            assert [asked.op for asked in control.asked] == ["state", "stop"]
        assert capsys.readouterr().err.splitlines()[0].endswith("resume 'sess-1'")

    def test_a_run_that_is_not_taken_over_keeps_its_name(
        self,
        tmp_path: Path,
        hosts: _Hosts,
    ) -> None:
        root = detach._paths("nu")
        root.mkdir(parents=True)
        replies = {"state": Reply(ok=True, state="running"), "stop": Reply(ok=True)}
        with hosts.serve(tmp_path / "c", replies) as control:
            detach._write_record(
                root,
                _record(root, name="nu", control=control.address, command="claude"),
            )
            with pytest.raises(SystemExit) as refused:
                _ = launch("nu", command="claude", start=_never)
            assert [asked.op for asked in control.asked] == ["state"]
        said = str(refused.value)
        assert "'nu' is running (the chat is not taken over)" in said
        assert "knowop run stop nu" in said

    def test_a_taken_over_run_on_its_way_out_is_waited_for(
        self,
        hosts: _Hosts,
        live_child: int,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The server turns the old run away as it opens the new one, so both race."""
        root = detach._paths("xi")
        root.mkdir(parents=True)
        detach._write_record(
            root,
            _record(
                root,
                name="xi",
                pid=live_child,
                session="sess-2",
                command="claude",
            ),
        )
        hosts.clock.on_sleep = partial(hosts.end, live_child)
        began = time.monotonic()
        assert launch("xi", command="claude", start=_say_the_spec) == 0
        assert time.monotonic() - began < 0.5
        assert hosts.clock.now == 0.05
        assert capsys.readouterr().err.splitlines()[0].endswith("resume 'sess-2'")

    def test_a_run_that_will_not_go_keeps_its_name(
        self,
        live_child: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(detach, "_STOP_TIMEOUT_SEC", 0.05)
        root = detach._paths("pi")
        root.mkdir(parents=True)
        detach._write_record(
            root,
            _record(
                root,
                name="pi",
                pid=live_child,
                session="sess-3",
                command="claude",
            ),
        )
        with pytest.raises(SystemExit, match="'pi' is still closing"):
            _ = launch("pi", command="claude", start=_never)

    @pytest.mark.parametrize(
        ("command", "resumed"),
        [("claude", "sess-4"), ("claude --model opus", "sess-4"), ("codex", "")],
    )
    def test_a_run_that_ended_taken_over_is_resumed_by_the_same_cli_alone(
        self,
        capsys: pytest.CaptureFixture[str],
        command: str,
        resumed: str,
    ) -> None:
        root = detach._paths("rho")
        root.mkdir(parents=True)
        detach._write_record(
            root,
            _record(
                root,
                name="rho",
                state="exited",
                exit_code=0,
                session="sess-4",
                command="claude",
                ended=_ago(30),
            ),
        )
        assert launch("rho", command=command, start=_say_the_spec) == 0
        assert capsys.readouterr().err.splitlines()[0].endswith(f"resume {resumed!r}")

    def test_a_run_that_started_by_itself_and_has_not_answered_keeps_its_name(
        self,
        live_child: int,
    ) -> None:
        root = detach._paths("phi")
        root.mkdir(parents=True)
        detach._write_record(
            root,
            _record(root, name="phi", state="starting", pid=live_child),
        )
        with pytest.raises(SystemExit) as refused:
            _ = launch("phi", command="claude", start=_never)
        assert str(refused.value) == (
            "knowop run: host 'phi' is starting; "
            "stop it (knowop run stop phi) or choose another --name"
        )

    def test_a_run_started_with_arguments_is_still_the_same_cli(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root = detach._paths("chi")
        root.mkdir(parents=True)
        detach._write_record(
            root,
            _record(
                root,
                name="chi",
                state="exited",
                exit_code=0,
                session="sess-7",
                command="claude --model opus",
                ended=_ago(30),
            ),
        )
        assert launch("chi", command="claude", start=_say_the_spec) == 0
        assert capsys.readouterr().err.splitlines()[0].endswith("resume 'sess-7'")

    def test_a_session_is_recent_until_the_window_has_passed(self) -> None:
        ended = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
        record = _record(Path("/h"), ended=ended.isoformat(timespec="seconds"))
        window = timedelta(seconds=120)

        assert detach._recent(record, now=ended + window - timedelta(seconds=1))
        assert not detach._recent(record, now=ended + window)
        assert detach._recent(_record(Path("/h")), now=ended + timedelta(days=9))

    def test_a_run_that_ended_long_ago_is_not_resumed(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The chat may have been answered by another provider since."""
        root = detach._paths("upsilon")
        root.mkdir(parents=True)
        detach._write_record(
            root,
            _record(
                root,
                name="upsilon",
                state="exited",
                exit_code=0,
                session="sess-6",
                command="claude",
                ended=_ago(3_600),
            ),
        )
        assert launch("upsilon", command="claude", start=_say_the_spec) == 0
        assert capsys.readouterr().err.splitlines()[0].endswith("resume ''")

    def test_a_run_that_was_never_taken_over_leaves_nothing_to_resume(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        root = detach._paths("sigma")
        root.mkdir(parents=True)
        detach._write_record(
            root,
            _record(
                root,
                name="sigma",
                state="exited",
                exit_code=0,
                command="claude",
            ),
        )
        assert launch("sigma", command="claude", start=_say_the_spec) == 0
        assert capsys.readouterr().err.splitlines()[0].endswith("resume ''")

    def test_a_takeover_is_written_into_the_hosts_record_beside_what_it_holds(
        self,
        tmp_path: Path,
    ) -> None:
        detach._write_record(tmp_path, _record(tmp_path, state="running", pid=7))

        detach._taken_over(tmp_path, "sess-8")

        record = detach._read_record(tmp_path)
        assert record is not None
        assert (record.session, record.state, record.pid) == ("sess-8", "running", 7)

    def test_a_takeover_in_a_host_with_no_record_writes_none(
        self,
        tmp_path: Path,
    ) -> None:
        detach._taken_over(tmp_path, "sess-9")

        assert not (tmp_path / "host.json").exists()

    def test_the_takeover_is_kept_in_the_record_through_the_hosts_end(self) -> None:
        assert launch("tau", command="claude", start=_take_over_then_end) == 0
        # The launcher returns once the host is up; the host ends a moment after.
        end = time.monotonic() + 10
        record = detach._read_record(detach._paths("tau"))
        while (record is None or record.state != "exited") and time.monotonic() < end:
            time.sleep(0.01)
            record = detach._read_record(detach._paths("tau"))
        assert record is not None
        assert (record.session, record.state) == ("sess-5", "exited")


def _ago(seconds: int) -> str:
    """Return the time ``seconds`` ago, as a host records when it ended."""
    return (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat(
        timespec="seconds",
    )


def _record(
    root: Path,
    *,
    name: str = "h",
    state: str = "running",
    pid: int = 1,
    address: Path | None = None,
    exit_code: int | None = None,
    machine: str = "",
    control: Path | None = None,
    session: str = "",
    command: str = "sh -- true",
    ended: str | None = None,
) -> HostRecord:
    """Return a host record with test defaults."""
    return HostRecord(
        name=name,
        command=command,
        cwd=str(root),
        pid=pid,
        address=str(address or root / "absent.sock"),
        started="2026-10-03T00:00:00+00:00",
        state=state,
        exit_code=exit_code,
        machine=machine,
        control="" if control is None else str(control),
        session=session,
        ended=ended,
    )


def _dead_pid() -> int:
    """Return a synthetic pid absent from the host table."""
    return 1 << 30


def _noting_is_dir(
    asked: list[Path],
    is_dir: Callable[[Path], bool],
) -> Callable[[Path], bool]:
    """Return ``is_dir`` that first notes each path it is asked about."""

    def noted(path: Path) -> bool:
        asked.append(path)
        return is_dir(path)

    return noted


# Written to the descriptor, not ``sys.stderr``: the host points descriptor 2 at its
# log, and pytest's capture replaces ``sys.stderr`` with a file of its own.
def _say_then_accept(spec: HostSpec) -> int:
    """Say one line, as a host loading a chat does, then accept and end."""
    _ = os.write(2, b"[knowop run] loaded 2 records\n")
    assert spec.on_listening is not None
    spec.on_listening()
    return 0


def _take_over_then_end(spec: HostSpec) -> int:
    """Be taken over on a session, accept, and end."""
    assert isinstance(spec, HeadlessSpec)
    assert spec.on_takeover is not None
    assert spec.on_listening is not None
    spec.on_listening()
    spec.on_takeover("sess-5")
    return 0


def _never(spec: object) -> int:
    """Fail: the refusal must stop the launch before ``start`` runs."""
    raise AssertionError(f"launched despite a running host: {spec}")


def _say_the_spec(spec: HostSpec) -> int:
    """Say where the host was told to listen for control requests, then accept."""
    assert isinstance(spec, HeadlessSpec)
    _ = os.write(
        2,
        f"[knowop run] control {spec.control} resume {spec.resume!r}\n".encode(),
    )
    assert spec.on_listening is not None
    spec.on_listening()
    return 0


_HERE = socket.gethostname()


class _Control:
    """A headless run's canned control replies and the requests it received."""

    def __init__(
        self,
        address: Path,
        reply: Reply | Mapping[str, Reply],
        steps: list[str] | None,
    ) -> None:
        self.address = address
        self.asked: list[Request] = []
        self._reply = reply
        self._steps = steps

    def answer(self, asked: Request) -> Reply:
        self.asked.append(asked)
        if self._steps is not None:
            self._steps.append(asked.op)
        return self._reply if isinstance(self._reply, Reply) else self._reply[asked.op]


def _fake_attach(
    monkeypatch: pytest.MonkeyPatch,
    steps: list[str],
    *,
    status: int | None,
) -> None:
    """Make the terminal attach note itself and return ``status``."""

    async def attached(address: Path) -> int | None:
        del address
        steps.append("attach")
        return status

    monkeypatch.setattr(detach, "attach", attached)


# Its process is already gone, so ``stop`` has nothing to wait for, and the terminal
# attach is faked.
@contextlib.contextmanager
def _headless(
    hosts: _Hosts,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: list[str],
    reply: Reply,
    *,
    status: int | None = None,
    machine: str = _HERE,
) -> Generator[_Control]:
    """Record a live headless host named ``kappa`` whose control socket gives ``reply``."""
    _fake_attach(monkeypatch, steps, status=status)
    root = detach._paths("kappa")
    root.mkdir(parents=True)
    with hosts.serve(tmp_path / "c", reply, steps) as control:
        detach._write_record(
            root,
            _record(
                root,
                name="kappa",
                pid=_dead_pid(),
                machine=machine,
                control=control.address,
            ),
        )
        yield control


def _accept_then_end(spec: HostSpec) -> int:
    """Accept, then end with a status of the CLI's."""
    assert spec.on_listening is not None
    spec.on_listening()
    return 3


def _refuse_in_the_host(spec: HostSpec) -> int:
    """Refuse cleanly, as the runner does for a CLI not on PATH."""
    del spec
    raise SystemExit("knowop run: claude not found in PATH")


def _fail_in_the_host(spec: HostSpec) -> int:
    """Fail with an error nothing above the host would report."""
    del spec
    raise RuntimeError("the CLI broke")


_HOST_ENDS: Final = {
    "returns": (_accept_then_end, 3, ""),
    "refuses": (_refuse_in_the_host, 1, "knowop run: claude not found in PATH\n"),
    "fails": (_fail_in_the_host, 1, "knowop run: the host failed"),
}
"""How a host's ``start`` can end: the start, the host's status, and what it logs."""


def _write_then_close(
    scrollback: Path,
    host: contextlib.ExitStack,
    seconds: float,
) -> None:
    """Stand in for the follower's wait: the host writes once more, then exits."""
    del seconds
    with scrollback.open("ab") as tail:
        _ = tail.write(b"two\r\n")
    host.close()


# Written to the descriptor, not ``sys.stderr``: the host points descriptor 2 at its
# log, and pytest's capture replaces ``sys.stderr`` with a file of its own.
def _say_then_end(spec: HostSpec) -> int:
    """Say why, then end without ever accepting."""
    del spec
    _ = os.write(2, b"[knowop run] claude not found\n")
    return 1


def _raise(error: Exception, address: Path) -> Reply:
    """Stand in for a control call that fails at ``address``."""
    del address
    raise error


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
