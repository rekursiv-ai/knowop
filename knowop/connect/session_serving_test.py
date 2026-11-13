"""SessionServing: a scripted server's starts, each run as a real stub of the process.

The stub stands in for ``knowop run --serve``: it prints its folder, the folder's mode,
its environment, its arguments, its input, its pid and its session, then exits with the
code it is given, or holds until it is killed. Nothing here waits on a clock: each
test waits on the fake server's condition, a queue, or a thread's end.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, TypedDict, override
from unittest.mock import Mock

import asyncio
import os
import queue
import signal
import stat
import sys
import threading
import uuid

from trackinizer.client.errors import ClientError
from trackinizer.lib.posix.detached import spawn_detached
from trackinizer.lib.userdirs import state_dir
from trackinizer.server.session_relay import SessionRelay
from trackinizer.wire.wire_session_relay import (
    MAX_REASON_CHARS,
    ProcessEnded,
    ProcessSpec,
    ServeSession,
)
from trackinizer.wire.wire_tools import WAIT_MAX_SEC
from treekle import json_loads

import pytest

from knowop.connect import session_serving
from knowop.connect.session_serving import (
    SERVE_COMMAND,
    Place,
    RefusedError,
    SessionServing,
    own_place,
)
from knowop.connect.wake import Wake


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, Sequence

    from knowop.connect.job_limits import Run


_HOST: Final = uuid.UUID(int=1)
_INSTANCE: Final = uuid.UUID(int=2)
_CREDENTIAL: Final = "trax_machine_" + "0" * 32 + "_" + "c" * 43
_TOKEN: Final = "trax_" + "t" * 43
_URL: Final = "https://know.example"
_REVOKED: Final = ClientError("-> 410", status_code=410)

_STUB: Final = (
    "import json, os, stat, sys, time;"
    "print(json.dumps({'cwd': os.getcwd(), 'mode': stat.S_IMODE(os.stat('.').st_mode),"
    " 'env': dict(os.environ), 'argv': sys.argv[2:], 'stdin': sys.stdin.read(),"
    " 'pid': os.getpid(), 'session': os.getsid(0)}), flush=True);"
    "time.sleep(60) if sys.argv[1] == 'hold' else None;"
    "raise SystemExit(int(sys.argv[1]))"
)
"""``knowop run --serve`` as a stub: what it was handed, then the exit it is given."""

_PYTHON_ADDS: Final = frozenset({"__CF_USER_TEXT_ENCODING", "LC_CTYPE"})
"""What a Python child sets in its own environment on macOS, whatever it was handed."""

_PATIENCE_SEC: Final = 3.0
"""How long a test waits for what a working serving does at once. A stub child starts
in tens of milliseconds; each mutant that breaks a report or a stop pays all of it."""


class _Said(TypedDict):
    """What the stub prints, first thing."""

    cwd: str
    mode: int
    env: dict[str, str]
    argv: list[str]
    stdin: str
    pid: int
    session: int


# Pytest says a thread's error as a warning, which this suite makes an error: one said
# after the last test fails the whole session, a mutation run's among them.
@pytest.fixture(autouse=True)
def _no_thread_left(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """End each test with its threads gone, and fail it for what one of them raised."""
    raised: list[threading.ExceptHookArgs] = []
    monkeypatch.setattr(threading, "excepthook", raised.append)
    yield
    _joined()
    assert [hook.exc_value for hook in raised] == []


class _Echoes:
    """What the serving says: each line, and whether it went to stderr."""

    def __init__(self) -> None:
        self.lines: queue.Queue[tuple[str, bool]] = queue.Queue()

    def __call__(self, message: object = "", *, err: bool = False) -> None:
        self.lines.put((str(message), err))

    def until(self, line: str) -> list[tuple[str, bool]]:
        """Return every line said up to ``line``, which must come in time."""
        said = [self.lines.get(timeout=_PATIENCE_SEC)]
        while said[-1][0] != line:
            said.append(self.lines.get(timeout=_PATIENCE_SEC))
        return said

    def said(self) -> list[tuple[str, bool]]:
        """Return every line said so far."""
        said: list[tuple[str, bool]] = []
        while not self.lines.empty():
            said.append(self.lines.get())
        return said


@pytest.fixture
def echoes(monkeypatch: pytest.MonkeyPatch) -> _Echoes:
    recorder = _Echoes()
    monkeypatch.setattr(session_serving, "echo", recorder)
    return recorder


class _Client:
    """The calls SessionServing makes: each wait takes the next start put here."""

    def __init__(self) -> None:
        self.answers: queue.Queue[ServeSession | Exception] = queue.Queue()
        self.waits: list[tuple[uuid.UUID, uuid.UUID, float]] = []
        self.changed = threading.Condition()
        """Notified as each wait is asked; the serving asks from its listener thread."""

        self.keys: list[str] = []
        self.ended: queue.Queue[tuple[uuid.UUID, ProcessEnded]] = queue.Queue()
        self.refusal: ClientError | None = None
        """What reporting an end raises, after the client's own retries."""

        self.closed = 0

    def __call__(self, key: str) -> _Client:
        self.keys.append(key)
        return self

    def wait_session(
        self,
        host_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        wait_sec: float,
    ) -> ServeSession | None:
        self.asked(host_id, instance, wait_sec)
        try:
            answer = self.answers.get(timeout=0.005)
        except queue.Empty:
            return None
        if isinstance(answer, Exception):
            raise answer
        return answer

    def report_process_ended(self, host_id: uuid.UUID, ended: ProcessEnded) -> None:
        if self.refusal is not None:
            raise self.refusal
        self.ended.put((host_id, ended))

    def close(self) -> None:
        self.closed += 1

    def asked(self, host_id: uuid.UUID, instance: uuid.UUID, wait_sec: float) -> None:
        """Record one wait and wake whoever waits for it."""
        with self.changed:
            self.waits.append((host_id, instance, wait_sec))
            self.changed.notify_all()

    def reached(self, count: int) -> None:
        """Return once ``count`` waits were asked."""
        with self.changed:
            asked = self.changed.wait_for(
                lambda: len(self.waits) >= count,
                timeout=_PATIENCE_SEC,
            )
        assert asked, f"never asked {count} waits"

    def end(self) -> ProcessEnded:
        """Return the next end reported, which must come from this host."""
        host_id, ended = self.ended.get(timeout=_PATIENCE_SEC)
        assert host_id == _HOST
        return ended


class _Waits(threading.Event):
    """A stop event that keeps how long each wait on it was for."""

    def __init__(self) -> None:
        super().__init__()
        self.waited: list[float | None] = []
        self.changed = threading.Condition()

    @override
    def wait(self, timeout: float | None = None) -> bool:
        with self.changed:
            self.waited.append(timeout)
            self.changed.notify_all()
        return super().wait(timeout)

    def entered(self, seconds: float) -> None:
        """Return once a wait of ``seconds`` has begun."""
        with self.changed:
            began = self.changed.wait_for(
                lambda: seconds in self.waited,
                timeout=_PATIENCE_SEC,
            )
        assert began, f"never rested {seconds} s"


class _Clock:
    """A monotonic clock that moves only when a test moves it."""

    def __init__(self) -> None:
        # Not zero: an elapsed time read as a sum would then pass for a difference.
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _stops(monkeypatch: pytest.MonkeyPatch) -> list[_Waits]:
    """Give each serving a recording stop event; return them as they are made."""
    stops: list[_Waits] = []

    def made() -> _Waits:
        stops.append(_Waits())
        return stops[-1]

    monkeypatch.setattr(session_serving, "threading", SimpleNamespace(Event=made))
    return stops


def _start(
    *,
    cwd: str = "~",
    args: Sequence[str] = (),
    login_host: str = "",
    remote: bool = False,
) -> ServeSession:
    return ServeSession(
        session_id=uuid.uuid4(),
        process_id=uuid.uuid4(),
        token=_TOKEN,
        spec=ProcessSpec(host="box", login_host=login_host, cwd=cwd, args=tuple(args)),
        remote=remote,
    )


def _no_systemd(argv: Sequence[str]) -> tuple[int, str]:
    """Answer as a computer with no ``systemd-run`` on ``PATH``."""
    del argv
    return 127, ""


def _serving(
    client: _Client,
    *,
    place: Callable[[ServeSession], Place] = own_place,
    exit_with: str = "0",
    command: Sequence[str] = (),
    run: Run = _no_systemd,
    retry_sec: float = 0.005,
    wait_sec: float = 0.0,
    wake: Wake | None = None,
) -> SessionServing:
    return SessionServing(
        _URL,
        credential=_CREDENTIAL,
        client_for=client,
        place=place,
        retry_sec=retry_sec,
        wait_sec=wait_sec,
        command=command or (sys.executable, "-I", "-S", "-c", _STUB, exit_with),
        run=run,
        wake=wake,
    )


def _ends(
    *starts: ServeSession,
    place: Callable[[ServeSession], Place] = own_place,
    command: Sequence[str] = (),
) -> list[ProcessEnded]:
    """Serve ``starts`` in order; return each end reported, in the order reported."""
    client = _Client()
    serving = _serving(client, place=place, command=command)
    for start in starts:
        client.answers.put(start)
    serving.start(_HOST, _INSTANCE)
    try:
        return [client.end() for _ in starts]
    finally:
        serving.stop()


def _served(
    start: ServeSession,
    *,
    place: Callable[[ServeSession], Place] = own_place,
) -> tuple[_Said, ProcessEnded]:
    """Serve ``start`` once; return what its stub said and how it was reported."""
    [ended] = _ends(start, place=place)
    return _said(start), ended


def _confined(serve: ServeSession) -> Place:
    """Return a confined place, as a remote agent service makes one, for ``serve``."""
    folder = state_dir() / "rekursiv-ai" / "confined" / str(serve.process_id)
    folder.mkdir(parents=True, mode=0o700)
    return Place(
        folder=folder,
        env={"PATH": os.environ["PATH"], "ONLY": "listed"},
        flags=("--tools", "RemoteTool"),
        owned=folder,
    )


def _said(start: ServeSession) -> _Said:
    """Return what the stub of ``start``'s process printed first."""
    processes = state_dir() / "rekursiv-ai" / "knowop" / "processes"
    printed = (processes / str(start.process_id) / "output.log").read_text()
    return json_loads(printed.splitlines()[0], _Said)


def _env(said: _Said) -> dict[str, str]:
    """Return the stub's environment, less what Python set there itself."""
    return {
        name: value for name, value in said["env"].items() if name not in _PYTHON_ADDS
    }


def _listeners() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "session-listener"]


def _joined() -> None:
    """Join the listeners and the children's reapers; fail if a listener outlives them."""
    for thread in threading.enumerate():
        if thread.name == "session-listener" or thread.name.startswith("reaper-"):
            thread.join(_PATIENCE_SEC)
    assert not _listeners(), "a listener was left running"


def test_the_command_is_this_installs_knowop_run_serve() -> None:
    # The client package, as this test's own module names it: the export renames it.
    package = __name__.rsplit(".", 2)[0]

    assert (sys.executable, "-P", "-m", package, "run", "--serve") == SERVE_COMMAND


def test_it_holds_each_wait_with_the_host_credential_and_closes_the_client() -> None:
    client = _Client()
    serving = _serving(client)

    serving.start(_HOST, _INSTANCE)
    client.reached(2)
    serving.stop()
    _joined()

    assert client.keys == [_CREDENTIAL]
    assert set(client.waits) == {(_HOST, _INSTANCE, WAIT_MAX_SEC)}
    assert client.closed == 1


def test_stopping_a_serving_never_started_returns_at_once() -> None:
    _serving(_Client()).stop()

    assert not _listeners()


def test_by_default_it_serves_its_own_account_asks_the_longest_hold_and_rests_5_s(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stops = _stops(monkeypatch)
    client = _Client()
    serving = SessionServing(
        _URL,
        credential=_CREDENTIAL,
        client_for=client,
        command=(sys.executable, "-I", "-S", "-c", _STUB, "0"),
        run=_no_systemd,
    )
    start = _start(cwd=str(tmp_path), args=("--model", "opus"))
    client.answers.put(start)

    serving.start(_HOST, _INSTANCE)
    ended = client.end()
    # The next wait, not held by this fake server, rests before the one after.
    stops[0].entered(5.0)
    serving.stop()
    _joined()

    # Its own host's start: the flags are passed on.
    assert (ended.process_id, ended.exit_code) == (start.process_id, 0)
    assert _said(start)["argv"] == [_URL, "--model", "opus"]
    assert {wait_sec for _, _, wait_sec in client.waits} == {WAIT_MAX_SEC}


class _Held(_Client):
    """A server that holds each wait for ``held_sec``, on the clock the serving reads."""

    def __init__(self, clock: _Clock, *, held_sec: float) -> None:
        super().__init__()
        self.clock = clock
        self.held_sec = held_sec

    @override
    def wait_session(
        self,
        host_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        wait_sec: float,
    ) -> ServeSession | None:
        self.asked(host_id, instance, wait_sec)
        self.clock.now += self.held_sec
        return None


@pytest.mark.parametrize(
    ("held_sec", "rests"),
    [(0.0, True), (0.49, True), (0.5, False), (1.0, False)],
    ids=["answered at once", "held under half", "held half", "held whole"],
)
def test_a_wait_held_half_its_time_is_asked_again_at_once_and_one_not_after_a_rest(
    monkeypatch: pytest.MonkeyPatch,
    *,
    held_sec: float,
    rests: bool,
) -> None:
    """A server that answers at once would otherwise be asked in a spin."""
    stops = _stops(monkeypatch)
    clock = _Clock()
    monkeypatch.setattr(session_serving, "time", SimpleNamespace(monotonic=clock))
    client = _Held(clock, held_sec=held_sec)
    serving = _serving(client, retry_sec=0.01, wait_sec=1.0)

    serving.start(_HOST, _INSTANCE)
    client.reached(3)
    serving.stop()
    _joined()

    assert {wait_sec for _, _, wait_sec in client.waits} == {1.0}
    assert (0.01 in stops[0].waited) is rests


@pytest.mark.parametrize("text", ["-> 503 busy", ""])
def test_a_wait_that_fails_is_retried_and_the_same_failure_said_once(
    echoes: _Echoes,
    text: str,
) -> None:
    client = _Client()
    serving = _serving(client)
    refused = ClientError(text, status_code=503)
    for _ in range(3):
        client.answers.put(refused)
    client.answers.put(_start())
    client.answers.put(refused)

    serving.start(_HOST, _INSTANCE)
    _ = client.end()
    client.reached(5)
    serving.stop()
    _joined()

    # A failure after a wait that worked is news again.
    assert echoes.said().count((f"knowop: session wait failed: {text}", True)) == 2


class _Refusing(_Client):
    """A server that refuses every wait, at once."""

    @override
    def wait_session(
        self,
        host_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        wait_sec: float,
    ) -> ServeSession | None:
        self.asked(host_id, instance, wait_sec)
        raise ClientError("-> 503 busy", status_code=503)


def test_a_wait_that_fails_is_asked_again_after_a_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stops = _stops(monkeypatch)
    client = _Refusing()
    serving = _serving(client, retry_sec=0.01)

    serving.start(_HOST, _INSTANCE)
    client.reached(3)
    serving.stop()
    _joined()

    assert stops[0].waited.count(0.01) >= len(client.waits) - 1


def test_a_wake_from_a_pause_starts_a_new_wait_on_a_client_of_its_own() -> None:
    """The wait held across a pause waits on a socket the pause killed."""
    wall = [0.0]
    wake = Wake(wall=lambda: wall[0], clock=lambda: 0.0)
    client = _Client()
    serving = _serving(client, wake=wake)
    serving.start(_HOST, _INSTANCE)
    client.reached(1)

    wall[0] += 30.0
    assert wake.check()
    serving.stop()
    _joined()

    assert (client.keys, client.closed) == ([_CREDENTIAL] * 2, 2)


def test_a_revoked_host_stops_waiting() -> None:
    client = _Client()
    serving = _serving(client)
    client.answers.put(_REVOKED)

    serving.start(_HOST, _INSTANCE)
    _joined()

    assert client.closed == 1
    assert len(client.waits) == 1


def test_each_start_is_started_once_its_end_reported_once_and_both_said(
    echoes: _Echoes,
) -> None:
    client = _Client()
    serving = _serving(client, exit_with="3")
    first, second = _start(), _start()
    client.answers.put(first)
    client.answers.put(second)

    serving.start(_HOST, _INSTANCE)
    ends = {client.end().process_id, client.end().process_id}
    serving.stop()
    _joined()

    assert ends == {first.process_id, second.process_id}
    assert client.ended.empty()
    assert [
        ServeSession.model_validate_json(_said(start)["stdin"])
        for start in (first, second)
    ] == [first, second]
    # One client waits; each report goes on a client of its own, as the host.
    assert client.keys == [_CREDENTIAL] * 3
    assert sorted(echoes.said()) == sorted(
        [
            *(
                (
                    (
                        f"knowop: agent process started: process={start.process_id} "
                        f"session={start.session_id}"
                    ),
                    True,
                )
                for start in (first, second)
            ),
            *(
                (
                    f"knowop: agent process ended: process={start.process_id} exit=3",
                    True,
                )
                for start in (first, second)
            ),
        ],
    )


def test_a_start_runs_in_its_folder_with_its_flags_and_reads_the_start_on_stdin(
    tmp_path: Path,
) -> None:
    start = _start(cwd=str(tmp_path), args=("--model", "opus"))

    said, ended = _served(start)

    assert Path(said["cwd"]).resolve() == tmp_path.resolve()
    assert said["argv"] == [_URL, "--model", "opus"]
    assert said["stdin"] == start.model_dump_json() + "\n"
    assert ServeSession.model_validate_json(said["stdin"]).token == _TOKEN
    assert (ended.process_id, ended.exit_code) == (start.process_id, 0)


def test_a_home_folder_is_the_daemons_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    said, _ = _served(_start(cwd="~"))

    assert Path(said["cwd"]).resolve() == tmp_path.resolve()


@pytest.mark.parametrize("cwd", ["/no/such/folder", "~nosuchuser12345/x"])
def test_a_folder_that_is_none_here_is_a_start_that_failed_naming_it(
    echoes: _Echoes,
    cwd: str,
) -> None:
    start = _start(cwd=cwd)

    [ended] = _ends(start)

    assert ended == ProcessEnded(
        process_id=start.process_id,
        exit_code=None,
        reason=f"{cwd} is not a folder on this host",
    )
    assert not (state_dir() / "rekursiv-ai" / "knowop" / "processes").exists()
    assert echoes.until(
        f"knowop: agent process ended: process={start.process_id} exit=None",
    ) == [(f"knowop: agent process ended: process={start.process_id} exit=None", True)]


def test_a_start_that_cannot_run_leaves_the_next_one_to_run(tmp_path: Path) -> None:
    refused, started = _start(cwd="/no/such/folder"), _start(cwd=str(tmp_path))

    ends = _ends(refused, started)

    assert [(end.process_id, end.exit_code) for end in ends] == [
        (refused.process_id, None),
        (started.process_id, 0),
    ]


def test_the_child_has_the_daemons_environment_less_trackinizers_and_traxs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRACKINIZER_TOKEN", "secret")
    monkeypatch.setenv("TRACKINIZER_SOMETHING_NEW", "new")
    monkeypatch.setenv("TRAX_RUN", "r")
    monkeypatch.setenv("LC_TRAX_SERVER", "s")
    monkeypatch.setenv("KNOWOP_TOOLS_TOKEN", "t")
    monkeypatch.setenv("ANY_OTHER", "kept")

    said, _ = _served(_start())

    env = _env(said)
    kept = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("TRACKINIZER_", "TRAX_", "LC_TRAX_"))
        and name not in {"KNOWOP_TOOLS_TOKEN", *_PYTHON_ADDS}
    }
    assert env == kept
    # Its user folders are the daemon's, so it finds the host's own sagent logins.
    assert env["XDG_STATE_HOME"] == os.environ["XDG_STATE_HOME"]


def test_the_output_is_kept_private_under_the_process(tmp_path: Path) -> None:
    start = _start(cwd=str(tmp_path))

    said, _ = _served(start)

    knowop = state_dir() / "rekursiv-ai" / "knowop"
    folder = knowop / "processes" / str(start.process_id)
    # Names as listed: macOS's file system opens a path under any casing of it.
    assert [each.name for each in state_dir().iterdir()] == ["rekursiv-ai"]
    assert [each.name for each in knowop.parent.iterdir()] == ["knowop"]
    assert [each.name for each in knowop.iterdir()] == ["processes"]
    assert [each.name for each in folder.iterdir()] == ["output.log"]
    assert stat.S_IMODE((folder / "output.log").stat().st_mode) == 0o600
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    # It leads a session of its own, so the daemon's terminal signals nothing to it.
    assert said["pid"] == said["session"]


@pytest.mark.parametrize(
    ("printed", "reason"),
    [
        # Trimmed, and a byte that is no character is said as the replacement one.
        (r"b'\xffbye\n'", "�bye"),
        (f"b'x' * {MAX_REASON_CHARS} + b'END'", "x" * (MAX_REASON_CHARS - 3) + "END"),
    ],
)
def test_an_exit_is_reported_with_its_code_and_the_end_of_its_output(
    printed: str,
    reason: str,
) -> None:
    start = _start()
    shout = f"import sys; sys.stdout.buffer.write({printed}); raise SystemExit(7)"

    [ended] = _ends(start, command=(sys.executable, "-I", "-S", "-c", shout))

    assert ended == ProcessEnded(
        process_id=start.process_id,
        exit_code=7,
        reason=reason,
    )


def test_a_child_that_cannot_be_started_is_reported_with_no_exit_code(
    tmp_path: Path,
) -> None:
    start = _start()
    missing = tmp_path / "missing"

    [ended] = _ends(start, command=(str(missing),))

    assert ended == ProcessEnded(
        process_id=start.process_id,
        exit_code=None,
        reason=f"[Errno 2] No such file or directory: '{missing}'",
    )


def test_a_start_that_raises_is_reported_and_the_next_one_still_starts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``Popen`` refuses a NUL the wire lets through; it once ended the listener."""
    spec = ProcessSpec.model_validate_json('{"args": ["a\\u0000b"]}')
    bad, good = _start(args=spec.args), _start()

    ends = _ends(bad, good)

    assert spec.args == ("a\x00b",)
    assert ends[0] == ProcessEnded(
        process_id=bad.process_id,
        exit_code=None,
        reason="embedded null byte",
    )
    assert (ends[1].process_id, ends[1].exit_code) == (good.process_id, 0)
    [failed] = [
        record for record in caplog.records if record.name == session_serving.__name__
    ]
    assert (
        failed.getMessage() == f"agent process start failed: process={bad.process_id}"
    )
    # With its traceback: what raised is no refusal this module wrote.
    assert isinstance(failed.exc_info, tuple)
    assert failed.exc_info[0] is ValueError


class _Unlogged:
    """Stands in for ``spawn_detached``: a child whose output is gone when it ends."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        exited: Callable[[int], None],
        **kwargs: object,
    ) -> int:
        del argv, kwargs
        # Named as the real reaper is, so each test joins it.
        threading.Thread(name="reaper-unlogged", target=exited, args=(0,)).start()
        return 1


def test_a_child_whose_output_is_gone_is_still_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_serving, "spawn_detached", _Unlogged())
    start = _start()

    [ended] = _ends(start)

    assert ended == ProcessEnded(process_id=start.process_id, exit_code=0, reason="")


def test_an_end_the_server_will_not_take_is_said_and_dropped(echoes: _Echoes) -> None:
    client = _Client()
    client.refusal = ClientError("-> 503 down", status_code=503)
    serving = _serving(client)
    start = _start()
    client.answers.put(start)
    dropped = (
        f"knowop: agent process end not reported: process={start.process_id} "
        "error=-> 503 down"
    )

    serving.start(_HOST, _INSTANCE)
    said = echoes.until(dropped)
    serving.stop()

    assert client.ended.empty()
    assert said[-2:] == [
        (f"knowop: agent process ended: process={start.process_id} exit=0", True),
        (dropped, True),
    ]


@pytest.mark.parametrize("login_host", ["", "laptop"])
def test_a_persons_own_host_gives_the_child_its_folder_environment_and_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    login_host: str,
) -> None:
    """A login named or not: it rides the start, whole, and changes nothing here."""
    monkeypatch.setenv("NOT_LISTED", "x")
    start = _start(cwd=str(tmp_path), login_host=login_host)

    said, ended = _served(start)

    env = _env(said)
    assert ServeSession.model_validate_json(said["stdin"]) == start
    assert ended.exit_code == 0
    assert Path(said["cwd"]).resolve() == tmp_path.resolve()
    assert (env["NOT_LISTED"], env["HOME"]) == ("x", os.environ["HOME"])
    assert said["argv"] == [_URL]
    # Not the process's own: a person's folder outlives it.
    assert tmp_path.is_dir()


def test_a_place_gives_the_child_its_folder_its_environment_alone_and_its_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOT_LISTED", "x")
    start = _start(cwd="/no/such/folder", args=())

    said, ended = _served(start, place=_confined)

    owned = state_dir() / "rekursiv-ai" / "confined" / str(start.process_id)
    assert (ended.process_id, ended.exit_code) == (start.process_id, 0)
    assert Path(said["cwd"]).resolve() == owned.resolve()
    assert _env(said) == {"PATH": os.environ["PATH"], "ONLY": "listed"}
    assert said["argv"] == [_URL, "--tools", "RemoteTool"]
    assert ServeSession.model_validate_json(said["stdin"]) == start


def test_a_folder_the_place_owns_is_gone_once_its_process_ends() -> None:
    first, second = _start(), _start()

    ends = _ends(first, second, place=_confined)

    confined = state_dir() / "rekursiv-ai" / "confined"
    assert {end.process_id for end in ends} == {first.process_id, second.process_id}
    assert list(confined.iterdir()) == []
    # Each ran in a folder of its own.
    assert {Path(_said(each)["cwd"]).name for each in (first, second)} == {
        str(first.process_id),
        str(second.process_id),
    }


def test_a_folder_the_place_owns_is_gone_when_its_child_cannot_start(
    tmp_path: Path,
) -> None:
    start = _start()

    [ended] = _ends(start, place=_confined, command=(str(tmp_path / "missing"),))

    assert (ended.process_id, ended.exit_code) == (start.process_id, None)
    assert list((state_dir() / "rekursiv-ai" / "confined").iterdir()) == []


def test_a_start_whose_owned_folder_is_already_gone_is_still_reported(
    tmp_path: Path,
) -> None:
    """Its place may own a folder something removed: the end is said all the same."""
    start = _start()

    def gone(serve: ServeSession) -> Place:
        del serve
        return Place(folder=tmp_path, env={}, flags=(), owned=tmp_path / "gone")

    [ended] = _ends(start, place=gone, command=(str(tmp_path / "missing"),))

    assert (ended.process_id, ended.exit_code) == (start.process_id, None)


def test_a_child_that_deleted_its_own_folder_is_still_reported() -> None:
    start = _start()
    folder = state_dir() / "rekursiv-ai" / "confined" / str(start.process_id)
    # Named, not ``os.getcwd()``: a mutation run's child may start anywhere, its
    # runner's own folder among them.
    gone = f"import shutil; shutil.rmtree({str(folder)!r})"

    [ended] = _ends(
        start,
        place=_confined,
        command=(sys.executable, "-I", "-S", "-c", gone),
    )

    assert ended == ProcessEnded(process_id=start.process_id, exit_code=0, reason="")


def _refusing(serve: ServeSession) -> Place:
    """Refuse ``serve``, as a place that will not run it does."""
    raise RefusedError(f"no room for {serve.process_id}")


def test_a_start_its_place_refuses_is_reported_and_starts_nothing(
    echoes: _Echoes,
) -> None:
    start = _start()

    [ended] = _ends(start, place=_refusing)

    assert ended == ProcessEnded(
        process_id=start.process_id,
        exit_code=None,
        reason=f"no room for {start.process_id}",
    )
    assert not (state_dir() / "rekursiv-ai" / "knowop" / "processes").exists()
    # A refusal is the place's own word, logged as no failure.
    assert (
        f"knowop: agent process ended: process={start.process_id} exit=None",
        True,
    ) in echoes.said()


def _broken(serve: ServeSession, *, failing: uuid.UUID) -> Place:
    """Fail for the start ``failing`` as a bug does; place any other as one's own."""
    if serve.process_id == failing:
        raise LookupError("a bug")
    return own_place(serve)


def test_a_place_that_fails_is_reported_and_the_next_start_still_runs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bad, good = _start(), _start()

    ends = _ends(bad, good, place=partial(_broken, failing=bad.process_id))

    assert ends[0] == ProcessEnded(
        process_id=bad.process_id,
        exit_code=None,
        reason="a bug",
    )
    assert (ends[1].process_id, ends[1].exit_code) == (good.process_id, 0)
    [failed] = [
        record for record in caplog.records if record.name == session_serving.__name__
    ]
    assert isinstance(failed.exc_info, tuple)
    assert failed.exc_info[0] is LookupError


def test_a_persons_own_host_refuses_a_start_for_a_remote_agent() -> None:
    """Here it would run with this host's files, environment and tools, for another."""
    start = _start(remote=True)

    with pytest.raises(RefusedError) as refused:
        _ = own_place(start)

    assert str(refused.value) == (
        "a remote agent's start runs on a remote agent service, not on a person's "
        "own host"
    )


class _Relayed(_Client):
    """A server whose waits are answered by Trackinizer's own relay."""

    def __init__(self, relay: SessionRelay) -> None:
        super().__init__()
        self.relay = relay

    @override
    def wait_session(
        self,
        host_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        wait_sec: float,
    ) -> ServeSession | None:
        self.asked(host_id, instance, wait_sec)
        return asyncio.run(
            self.relay.wait(host_id, wait_sec=0.005, dropped=_still_there),
        )


async def _still_there() -> bool:
    """Answer that the wait's caller has not gone."""
    return False


def test_a_start_the_server_placed_for_remote_agents_is_refused_on_a_persons_host() -> (
    None
):
    """Its daemon may hold the name a remote agent service enrolled under."""
    relay = SessionRelay()
    # As the route starts it once it found a remote agent service.
    started = relay.start(
        uuid.uuid4(),
        ProcessSpec(),
        process_id=uuid.uuid4(),
        host_id=_HOST,
        host="remote-agents",
        token=_TOKEN,
        remote=True,
    )
    client = _Relayed(relay)
    serving = _serving(client)

    serving.start(_HOST, _INSTANCE)
    ended = client.end()
    serving.stop()
    _joined()

    assert (ended.process_id, ended.exit_code) == (started.process_id, None)
    assert not (state_dir() / "rekursiv-ai" / "knowop" / "processes").exists()


class _Spawned:
    """Starts each child with ``spawn_detached``, and hands on its pid."""

    def __init__(self) -> None:
        self.pids: queue.Queue[int] = queue.Queue()

    def __call__(
        self,
        argv: Sequence[str],
        *,
        log: Path,
        cwd: Path | None,
        stdin: bytes,
        env: Mapping[str, str] | None,
        exited: Callable[[int], None] | None,
        pass_fds: Sequence[int] = (),
    ) -> int:
        pid = spawn_detached(
            argv,
            log=log,
            pass_fds=pass_fds,
            cwd=cwd,
            stdin=stdin,
            env=env,
            exited=exited,
        )
        self.pids.put(pid)
        return pid


def test_stopping_the_daemon_leaves_its_processes_running_and_their_ends_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned = _Spawned()
    monkeypatch.setattr(session_serving, "spawn_detached", spawned)
    client = _Client()
    serving = _serving(client, exit_with="hold")
    start = _start()
    client.answers.put(start)
    serving.start(_HOST, _INSTANCE)
    pid = spawned.pids.get(timeout=_PATIENCE_SEC)

    serving.stop()
    for listener in _listeners():
        listener.join(_PATIENCE_SEC)
    try:
        assert not _listeners()
        os.kill(pid, 0)
    finally:
        os.kill(pid, signal.SIGKILL)

    ended = client.end()
    assert (ended.process_id, ended.exit_code) == (start.process_id, -signal.SIGKILL)


class _Recorded:
    """Stands in for ``spawn_detached``: records each command, and ends it at once."""

    def __init__(self) -> None:
        self.argvs: list[list[str]] = []

    def __call__(
        self,
        argv: Sequence[str],
        *,
        log: Path,
        exited: Callable[[int], None],
        **kwargs: object,
    ) -> int:
        del kwargs
        self.argvs.append(list(argv))
        log.touch()
        exited(0)
        return 1


class _Systemd:
    """Answers ``systemd-run`` as a computer whose user manager does, or does not."""

    def __init__(self, code: int) -> None:
        self.code = code
        self.asked: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> tuple[int, str]:
        self.asked.append(list(argv))
        return self.code, ""


@pytest.mark.parametrize("code", [0, 1])
def test_a_process_gets_a_scope_of_its_own_where_systemd_run_works(
    monkeypatch: pytest.MonkeyPatch,
    *,
    code: int,
) -> None:
    recorded = _Recorded()
    monkeypatch.setattr(session_serving, "spawn_detached", recorded)
    systemd = _Systemd(code)
    client = _Client()
    serving = _serving(client, command=("knowop-stub",), run=systemd)
    # A ``$`` in a flag asks this systemd whether it would rewrite it.
    starts = (_start(args=("--add-dir", "$work")), _start(args=("--add-dir", "$work")))
    for start in starts:
        client.answers.put(start)

    serving.start(_HOST, _INSTANCE)
    _ = (client.end(), client.end())
    serving.stop()

    # Asked once, with a scope that runs nothing and caps nothing.
    probe = ["systemd-run", "--user", "--scope", "--quiet", "--", "true"]
    if code:
        assert systemd.asked == [probe]
        assert recorded.argvs == [["knowop-stub", _URL, "--add-dir", "$work"]] * 2
        return
    assert systemd.asked == [probe, *[["systemd-run", "--version"]] * 2]
    assert recorded.argvs == [
        [
            "systemd-run",
            "--user",
            "--scope",
            "--quiet",
            "--unit",
            f"knowop-process-{start.process_id}",
            "--",
            "knowop-stub",
            _URL,
            "--add-dir",
            "$work",
        ]
        for start in starts
    ]


# -- A daemon that restarts while its processes run ---------------------------------


_READS_ITS_FIFO: Final = "import sys; open(sys.argv[1]).read()"
"""A process that runs until its fifo, its first argument, is opened and closed."""


def _outliving(
    tmp_path: Path,
    echoes: _Echoes,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ServeSession, Path, threading.Event]:
    """Start a process under a daemon that dies before it ends, as a restart leaves one."""
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    start = _start()
    client = _Client()
    serving = _serving(
        client,
        place=_confined,
        command=(sys.executable, "-I", "-S", "-c", _READS_ITS_FIFO, str(fifo)),
    )
    exited = threading.Event()

    def died(*args: object, **kwargs: object) -> None:
        del args, kwargs
        exited.set()

    # The daemon that started it is gone: nothing of its own reports the end.
    monkeypatch.setattr(serving, "_ended", Mock(side_effect=died))
    client.answers.put(start)
    serving.start(_HOST, _INSTANCE)
    _ = echoes.until(
        f"knowop: agent process started: process={start.process_id} "
        f"session={start.session_id}",
    )
    serving.stop()
    for listener in _listeners():
        listener.join(_PATIENCE_SEC)
    return start, fifo, exited


def _released(fifo: Path) -> None:
    """End the process reading ``fifo``."""
    with fifo.open("w"):
        pass


@pytest.mark.parametrize("ends", ["after", "before"], ids=["still runs", "ended"])
def test_a_new_daemon_reports_the_end_of_a_process_an_earlier_one_started(
    tmp_path: Path,
    echoes: _Echoes,
    monkeypatch: pytest.MonkeyPatch,
    ends: str,
) -> None:
    """Its owned folder goes with it, whether it ends before the new daemon starts or after."""
    start, fifo, exited = _outliving(tmp_path, echoes, monkeypatch)
    owned = state_dir() / "rekursiv-ai" / "confined" / str(start.process_id)
    assert owned.is_dir()
    if ends == "before":
        _released(fifo)
        assert exited.wait(_PATIENCE_SEC)
    client = _Client()
    serving = _serving(client)

    serving.start(_HOST, _INSTANCE)
    if ends == "after":
        assert owned.is_dir()
        _released(fifo)
    try:
        ended = client.end()
    finally:
        serving.stop()

    assert (ended.process_id, ended.exit_code) == (start.process_id, None)
    assert not owned.exists()
    assert not (
        state_dir()
        / "rekursiv-ai"
        / "knowop"
        / "processes"
        / str(start.process_id)
        / "process.json"
    ).exists()


def _written(path: Path, *, host_id: uuid.UUID, pid: int) -> None:
    """Write a record no process holds, as a daemon that started ``pid`` would have."""
    path.parent.mkdir(parents=True)
    held = session_serving._held(path, host_id=host_id, owned=None)
    session_serving._write(held, host_id=host_id, owned=None, pid=pid)
    os.close(held)


@pytest.mark.parametrize(
    "case",
    ["another host", "free yet running", "unreadable"],
)
def test_a_record_a_new_daemon_cannot_take_up_is_left_as_found(
    caplog: pytest.LogCaptureFixture,
    case: str,
) -> None:
    """Free of its lock while its pid runs, it may be a launcher that closed it."""
    record = (
        state_dir()
        / "rekursiv-ai"
        / "knowop"
        / "processes"
        / str(uuid.uuid4())
        / "process.json"
    )
    _written(
        record,
        host_id=uuid.uuid4() if case == "another host" else _HOST,
        pid=os.getpid(),
    )
    if case == "unreadable":
        record.write_text("{")
    client = _Client()
    serving = _serving(client)

    serving.start(_HOST, _INSTANCE)
    serving.stop()

    assert record.exists()
    assert client.ended.empty()
    assert [
        r.getMessage() for r in caplog.records if "left as found" in r.getMessage()
    ] == (
        [f"agent process left as found: process={record.parent.name} pid={os.getpid()}"]
        if case == "free yet running"
        else []
    )


def test_a_record_gone_before_it_is_read_is_passed_over(tmp_path: Path) -> None:
    client = _Client()

    _serving(client)._adopt(tmp_path / "gone" / "process.json", host_id=_HOST)

    assert client.ended.empty()


def test_a_start_that_cannot_run_leaves_no_record_behind(tmp_path: Path) -> None:
    start = _start()

    [ended] = _ends(start, command=(str(tmp_path / "missing"),))

    assert ended.exit_code is None
    assert not (
        state_dir()
        / "rekursiv-ai"
        / "knowop"
        / "processes"
        / str(start.process_id)
        / "process.json"
    ).exists()


class _SeesRecord:
    """Stands in for ``spawn_detached``: keeps the record a child is started holding."""

    def __init__(self) -> None:
        self.found: list[tuple[str, int]] = []
        """Each record's text and its file's permission bits, as its child starts."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        log: Path,
        exited: Callable[[int], None],
        **kwargs: object,
    ) -> int:
        del argv, kwargs
        record = log.parent / "process.json"
        self.found.append((record.read_text(), record.stat().st_mode & 0o777))
        log.touch()
        exited(0)
        return 1


def test_a_record_names_its_host_and_owned_folder_before_its_process_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A daemon that dies as it starts a process leaves the next one what to clean up."""
    seen = _SeesRecord()
    monkeypatch.setattr(session_serving, "spawn_detached", seen)
    start = _start()

    _ = _ends(start, place=_confined)

    owned = state_dir() / "rekursiv-ai" / "confined" / str(start.process_id)
    [(text, mode)] = seen.found
    assert json_loads(text, dict[str, object]) == {
        "host_id": str(_HOST),
        "owned": str(owned),
        "pid": 0,
    }
    # The record names a folder to delete: its account alone may write it.
    assert mode == 0o600


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
