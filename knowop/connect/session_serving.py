"""Start the agent processes the server asks this host for, and report each one's end.

A daemon that serves sessions holds a request on the server for the next start, as it
holds one for the next tool connection (``tool_serving.py``); the server answers it
the moment a start is asked. Each start is a ``ServeSession``, handed once. For each,
the daemon starts this install's ``knowop run --serve URL`` detached: in a session of
its own and, where ``systemd-run --user`` works, a scope of its own, so stopping,
upgrading or crashing the daemon leaves it running. The child reads the
``ServeSession`` as one JSON line on stdin, and its output goes to
``state_dir()/rekursiv-ai/knowop/processes/<process>/output.log``. When it exits the
daemon reports why in its own words, its exit code, which every reader of the session
is told, and the end of that log apart, which only a writer of the session reads; then
the folder is deleted. A start that fails before anything runs is reported with no
exit code, and the daemon's own words for why.

A daemon may restart while its processes run on, as the remote agent service does at
each deploy. So beside the log the daemon writes ``process.json``, which names the
host and the folder the process owns, and holds an exclusive ``flock`` on it that
the child inherits and keeps until it exits. A daemon that starts takes up each such
file of its host: one no process holds any more is reported ended, with no exit
code, and its folders deleted; one still held is waited on, on a thread blocked on
the lock, and reported so when its process ends. The lock is the daemon's own and
names no pid, so a pid used again names nothing. A file free of its lock whose
process still runs, which only a launcher that closed the descriptor would leave,
is left alone, as before.

Where and with what each process starts is its :class:`Place`, which the serving's
``place`` returns for each start. On a person's own host (:func:`own_place`, the
default) the child starts in the start's folder, with the daemon's environment less
every variable no tool inherits either (:func:`scrubbed_names`), and sagent's flags
after ``--``; a start the server placed for a remote agent is refused there. A place
may own a folder made for its process alone, which is deleted when the process ends
or its start fails, and may name a launcher that runs the command: the remote agent
service's jail, which starts each agent in a systemd unit of its own and waits for it,
so the launcher's exit is the agent's.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, TypedDict

import fcntl
import logging
import os
import shutil
import sys
import threading
import time
import uuid

from trackinizer.client.errors import ClientError
from trackinizer.lib.posix.detached import spawn_detached
from trackinizer.lib.posix.pids import running
from trackinizer.lib.userdirs import state_dir
from trackinizer.trax.render import echo
from trackinizer.wire.wire_session_relay import (
    MAX_REASON_CHARS,
    ProcessEnded,
    StopProcess,
)
from trackinizer.wire.wire_tools import WAIT_MAX_SEC
from treekle import json_dumps, json_loads

from knowop.connect.job_limits import Run, Scoping, run
from knowop.connect.terminal_serving import start_folder
from knowop.connect.tool_serving import scrubbed_names
from knowop.connect.wake import Listener


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from trackinizer.wire.wire_session_relay import ServeSession

    from knowop.connect.wake import Wake


__all__ = [
    "SERVE_COMMAND",
    "Place",
    "RefusedError",
    "SessionClient",
    "SessionServing",
    "own_place",
]

_logger = logging.getLogger(__name__)

SERVE_COMMAND: Final = (
    sys.executable,
    "-P",
    "-m",
    __name__.rsplit(".", 2)[0],
    "run",
    "--serve",
)
"""How the daemon starts an agent process: this install's ``knowop run --serve``, which
the server's URL, a time limit if the start has one, ``--`` and sagent's flags follow. ``PATH`` may hold another knowop, or none,
and the published package is named otherwise, so the package is named from this
module; ``-P`` keeps a module in the child's folder from running in its place."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Place:
    """Where and with what one agent process starts.

    Attributes:
      folder: The folder it starts in.
      env: Its whole environment.
      flags: The start's flags: a leading ``--time TIME`` of knowop's, then sagent's.
      owned: A folder made for the process alone, deleted when it ends or its start
        fails; ``None`` for none.
      launcher: The words that run the command, before its own: a jail's program,
        which starts it in a unit of its own; empty runs the command itself.

    """

    folder: Path
    env: Mapping[str, str]
    flags: tuple[str, ...]
    owned: Path | None = None
    launcher: tuple[str, ...] = ()


class RefusedError(Exception):
    """A start a place will not run; its text is the reason reported."""


def own_place(serve: ServeSession) -> Place:
    """Return a person's own host's place for ``serve``: its folder, its environment.

    Args:
      serve: The start.

    Returns:
      place: The start's folder, the daemon's environment less what no tool inherits,
        and the start's own sagent flags.

    Raises:
      RefusedError: The server placed it for a remote agent, or its folder is none
        on this host.

    """
    # A connect of a person's may hold the name a remote agent service enrolled
    # under, signed in as its account: here the start would run with this host's
    # files, environment and tools, for another account.
    if serve.remote:
        raise RefusedError(
            "a remote agent's start runs on a remote agent service, not on a "
            "person's own host",
        )
    folder = start_folder(serve.spec.cwd)
    if folder is None:
        raise RefusedError(f"{serve.spec.cwd} is not a folder on this host")
    withheld = set(scrubbed_names())
    env = {name: value for name, value in os.environ.items() if name not in withheld}
    return Place(folder=folder, env=env, flags=serve.spec.args)


class SessionClient(Protocol):
    """The calls SessionServing makes: the ``Client`` methods, and closing it."""

    def wait_session(
        self,
        host_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        wait_sec: float,
    ) -> ServeSession | StopProcess | None:
        """Hold for the next start, or a stop of a process it runs."""
        ...

    def report_process_ended(self, host_id: uuid.UUID, ended: ProcessEnded) -> None:
        """Report that a process this daemon started exited, or never started."""
        ...

    def close(self) -> None:
        """Release the connection."""
        ...


# Before the class, whose default it is: a person's own daemon is handed no stop.
def _stops_none(process_id: uuid.UUID) -> None:
    """End no process."""
    del process_id


class SessionServing:
    """Starts each agent process the server asks this host for; see the module docstring.

    Args:
      url: The server's URL, which each process is handed.
      credential: The host credential, which the client sends.
      client_for: Returns a client that sends the given credential.
      place: Returns where and with what a start runs, or raises
        :class:`RefusedError`; a person's own host's by default.
      retry_sec: How long to wait before asking again after a failed wait, or one the
        server did not hold.
      wait_sec: How long to ask the server to hold each wait; ``0`` asks for the
        longest it holds.
      command: What starts an agent process, after the place's launcher: the URL, a
        time limit, ``--`` and sagent's flags follow it, as ``knowop run --serve``
        takes them.
      run: Runs ``systemd-run`` once, to learn whether a process can have a scope.
      wake: Tells when this computer wakes from a pause: the wait held across it then
        waits on a dead socket, so a new one is started at once. ``None`` watches not.
      stop: Ends a process the server stopped, whole, when the wait hands its stop:
        the remote agent service's jail stops its unit. A person's own daemon is
        handed none, and ends none.
      running: Returns the processes the place runs now. A start of the serving stops
        each one no record names: its launcher is gone, so no one waits on it or
        reports its end.
      reports_output: Whether a process's end is reported in its own words: why its
        start failed, and the end of its output. The remote agent service says
        neither, since both name its folders, which are not the people's it serves;
        its log keeps them. A place's refusal, written for the person, is told all
        the same.

    """

    def __init__(
        self,
        url: str,
        *,
        credential: str,
        client_for: Callable[[str], SessionClient],
        place: Callable[[ServeSession], Place] = own_place,
        retry_sec: float = 5.0,
        wait_sec: float = 0.0,
        command: Sequence[str] = SERVE_COMMAND,
        run: Run = run,
        wake: Wake | None = None,
        stop: Callable[[uuid.UUID], None] = _stops_none,
        running: Callable[[], Iterable[uuid.UUID]] = tuple,
        reports_output: bool = True,
    ) -> None:
        self._url = url
        self._credential = credential
        self._client_for = client_for
        self._place = place
        self._retry_sec = retry_sec
        # The session wait is held as the tool wait is, to the same longest hold.
        self._wait_sec = wait_sec or WAIT_MAX_SEC
        self._command = tuple(command)
        self._scoping = Scoping(run)
        self._wake = wake
        self._stopper = stop
        self._running = running
        self._reports_output = reports_output
        self._stop = threading.Event()

    def start(self, host_id: uuid.UUID, instance: uuid.UUID) -> None:
        """Start waiting for starts on a thread.

        First takes up each process an earlier daemon of the host started: one
        that ended meanwhile is reported now, one still running when it ends.

        Args:
          host_id: This host's id.
          instance: This daemon's instance.

        """
        processes = state_dir() / "rekursiv-ai" / "knowop" / "processes"
        for record in sorted(processes.glob(f"*/{_RECORD}")):
            self._adopt(record, host_id=host_id)
        for process in self._running():
            if not (processes / str(process) / _RECORD).exists():
                echo(
                    f"knowop: agent process stopped: process={process}: it ran with no "
                    "record, so nothing waited on it",
                    err=True,
                )
                self._stopper(process)
        Listener(
            name="session-listener",
            listen=partial(self._listen, host_id, instance),
            stop=self._stop,
        ).start(self._wake)

    def stop(self) -> None:
        """Stop waiting for starts; the processes it started go on.

        A wait the server is still holding returns on its own, and a start it brings
        is still started: a process outlives its daemon.
        """
        self._stop.set()

    def _listen(
        self,
        host_id: uuid.UUID,
        instance: uuid.UUID,
        current: Callable[[], bool],
    ) -> None:
        """Wait for starts until stopped, revoked or replaced, starting each."""
        client = self._client_for(self._credential)
        # The failure last said while the waits keep failing, even one with no text.
        said: str | None = None
        try:
            while not self._stop.is_set() and current():
                asked = time.monotonic()
                try:
                    serve = client.wait_session(
                        host_id,
                        instance=instance,
                        wait_sec=self._wait_sec,
                    )
                except ClientError as error:
                    if error.status_code == 410:
                        self._stop.set()
                        return
                    # A refusal that lasts would say so every few seconds.
                    if str(error) != said:
                        said = str(error)
                        echo(f"knowop: session wait failed: {error}", err=True)
                    _ = self._stop.wait(self._retry_sec)
                    continue
                said = None
                if isinstance(serve, StopProcess):
                    echo(
                        f"knowop: agent process stopped: process={serve.process_id}: "
                        "its server stopped it",
                        err=True,
                    )
                    self._stopper(serve.process_id)
                elif serve is not None:
                    self._start_process(serve, host_id=host_id)
                elif time.monotonic() - asked < self._wait_sec / 2:
                    # Not held: asking at once would spin on this server.
                    _ = self._stop.wait(self._retry_sec)
        finally:
            client.close()

    def _start_process(self, serve: ServeSession, *, host_id: uuid.UUID) -> None:
        """Start ``serve``'s process; one that cannot start is reported at once."""
        process = str(serve.process_id)
        log = (
            state_dir()
            / "rekursiv-ai"
            / "knowop"
            / "processes"
            / process
            / "output.log"
        )
        place: Place | None = None
        # The folder this start made, once it has: one there already is another's.
        made: list[Path] = []
        try:
            place = self._place(serve)
            # A process's folder is new: the server hands each start once.
            log.parent.mkdir(parents=True, mode=0o700)
            made.append(log.parent)
            held = _held(log.parent / _RECORD, host_id=host_id, owned=place.owned)
            try:
                pid = spawn_detached(
                    self._scoping.argv(
                        [
                            *place.launcher,
                            *self._command,
                            *_serve_args(self._url, place.flags),
                        ],
                        unit=f"knowop-process-{process}",
                    ),
                    log=log,
                    pass_fds=(held,),
                    cwd=place.folder,
                    stdin=serve.model_dump_json().encode() + b"\n",
                    env=place.env,
                    exited=partial(
                        self._ended,
                        serve.process_id,
                        host_id=host_id,
                        log=log,
                        owned=place.owned,
                    ),
                )
                _write(held, host_id=host_id, owned=place.owned, pid=pid)
            finally:
                # The child holds the lock from here on, through its own copy.
                os.close(held)
        except (RefusedError, OSError) as error:
            failure: Exception = error
        except Exception as error:
            # ``Popen`` refuses an argument holding a NUL, which the wire lets through.
            # A start that raises anything ends alone, as a tool connection does in
            # ``ToolServing``: the listener, and every later start, go on.
            _logger.warning(
                "agent process start failed: process=%s",
                process,
                exc_info=True,
            )
            failure = error
        else:
            echo(
                f"knowop: agent process started: process={process} "
                f"session={serve.session_id}",
                err=True,
            )
            return
        if place is not None and place.owned is not None:
            shutil.rmtree(place.owned, ignore_errors=True)
        for folder in made:
            shutil.rmtree(folder, ignore_errors=True)
        said = self._reports_output or isinstance(failure, RefusedError)
        if not said:
            echo(
                f"knowop: agent process not started: process={process}: {failure}",
                err=True,
            )
        self._report(
            host_id,
            ProcessEnded(
                process_id=serve.process_id,
                exit_code=None,
                reason=str(failure)[:MAX_REASON_CHARS] if said else _UNSTARTED,
            ),
        )

    def _adopt(self, record: Path, *, host_id: uuid.UUID) -> None:
        """Report the process ``record`` names if it ended; else wait for its end."""
        try:
            held = os.open(record, os.O_RDONLY)
        except FileNotFoundError:
            return
        try:
            written = json_loads(os.read(held, _RECORD_BYTES), _Record)
        except ValueError:
            written = None
        if written is None or written["host_id"] != str(host_id):
            os.close(held)
            return
        ended = partial(
            self._ended,
            uuid.UUID(record.parent.name),
            None,
            host_id=host_id,
            log=record.parent / "output.log",
            owned=Path(written["owned"]) if written["owned"] else None,
        )
        try:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            threading.Thread(
                name=f"reaper-adopted-{record.parent.name}",
                target=_outlived,
                args=(held, ended),
                daemon=True,
            ).start()
            return
        os.close(held)
        if written["pid"] and running(written["pid"]):
            # Free of its lock yet running: its launcher closed the descriptor, or
            # the pid names another process now. Neither says it ended.
            _logger.warning(
                "agent process left as found: process=%s pid=%d",
                record.parent.name,
                written["pid"],
            )
            return
        ended()

    def _ended(
        self,
        process_id: uuid.UUID,
        code: int | None,
        *,
        host_id: uuid.UUID,
        log: Path,
        owned: Path | None,
        reason: str = "",
    ) -> None:
        """Delete the folders a process owned and wrote to, then report its exit and output's end."""
        if owned is not None:
            shutil.rmtree(owned, ignore_errors=True)
        output = _tail(log)
        shutil.rmtree(log.parent, ignore_errors=True)
        if output and not self._reports_output:
            echo(
                f"knowop: agent process output: process={process_id}\n{output}",
                err=True,
            )
            output = ""
        self._report(
            host_id,
            ProcessEnded(
                process_id=process_id,
                exit_code=code,
                reason=reason
                or (
                    "it ended while its daemon was away"
                    if code is None
                    else f"exited with code {code}"
                ),
                output=output,
            ),
        )

    def _report(self, host_id: uuid.UUID, ended: ProcessEnded) -> None:
        """Report ``ended`` with the client's own retries; one that still fails is said."""
        echo(
            f"knowop: agent process ended: process={ended.process_id} "
            f"exit={ended.exit_code}",
            err=True,
        )
        try:
            with closing(self._client_for(self._credential)) as client:
                client.report_process_ended(host_id, ended)
        except ClientError as error:
            echo(
                f"knowop: agent process end not reported: "
                f"process={ended.process_id} error={error}",
                err=True,
            )


_UNSTARTED: Final = "the agent could not start; the service's log says why"
"""Why a start failed, as a service that reports no output says it (``reports_output``)."""

_RECORD: Final = "process.json"
"""Beside a process's log: its host and owned folder, locked for as long as it runs."""

_RECORD_BYTES: Final = 1 << 16
"""The most of a record read: a host id, a pid and a folder's path."""


class _Record(TypedDict):
    """What ``process.json`` says of the process it is locked for."""

    host_id: str
    owned: str
    """The folder it owns; empty for none."""

    pid: int
    """Its pid once spawned; 0 before."""


# ``knowop run --on HOST --time TIME`` hands its time limit at the head of the start's
# args (``process/main.py::_started_on``); it is knowop's flag, so it goes before
# ``--``, where sagent would refuse it as a setting it has not.
def _serve_args(url: str, flags: Sequence[str]) -> list[str]:
    """Return what follows ``knowop run --serve``: the URL, a time limit, ``--``, sagent's."""
    timed = list(flags[:2]) if flags[:1] == ("--time",) else []
    return [url, *timed, "--", *flags[len(timed) :]]


def _held(path: Path, *, host_id: uuid.UUID, owned: Path | None) -> int:
    """Write a process's record, locked exclusively; return its descriptor."""
    held = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    fcntl.flock(held, fcntl.LOCK_EX)
    _write(held, host_id=host_id, owned=owned, pid=0)
    return held


def _write(held: int, *, host_id: uuid.UUID, owned: Path | None, pid: int) -> None:
    """Write the record through ``held``, whole."""
    record = _Record(
        host_id=str(host_id),
        owned="" if owned is None else str(owned),
        pid=pid,
    )
    os.ftruncate(held, 0)
    _ = os.pwrite(held, json_dumps(record, target=_Record).encode(), 0)


# A blocked ``flock`` is the wait: the lock is freed when the process, the last holder
# of the descriptor, exits. No portable event marks the exit of a process that is not
# this one's child, and this needs none.
def _outlived(held: int, ended: Callable[..., None]) -> None:
    """Wait for the process holding ``held`` to exit; then report it ended."""
    try:
        fcntl.flock(held, fcntl.LOCK_EX)
    finally:
        os.close(held)
    # This daemon saw the end, but not the exit code: an earlier daemon started it.
    ended(reason="it ended after its daemon restarted, so its exit code is not known")


# Read whole: a headless agent prints its session's id and its errors, no more. A log
# that is gone or cannot be read is said as nothing: the end is reported all the same,
# from the reaper's thread, where a raise would report none.
def _tail(log: Path) -> str:
    """Return the last ``MAX_REASON_CHARS`` characters ``log`` holds, trimmed."""
    try:
        held = log.read_bytes()
    except OSError:
        return ""
    return held.decode(errors="replace").strip()[-MAX_REASON_CHARS:]
