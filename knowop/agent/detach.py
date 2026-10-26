r"""Keep ``knowop run`` alive past its terminal: launch detached, attach by name.

``knowop run --detach`` hosts the wrapped CLI in a background process no terminal
owns (:class:`~trackinizer.lib.posix.host.Host`), so closing the shell, dropping ssh,
or a launcher exiting leaves the agent running -- the job tmux did for
long-running agents. Capture, sync, and inbound delivery run in that process
exactly as in a foreground run. Each host has a name and a directory,
``state_dir()/rekursiv-ai/trax/run/hosts/<name>/``, holding:

- ``host.json``: what it runs, where it listens, and how it ended.
- ``scrollback.log``: every byte the CLI wrote, in order.
- ``host.log``: the runner's own messages.

These reach a host by name:

- ``knowop run ls`` lists the live hosts; ``--all`` lists every host and its state.
- ``knowop run attach NAME`` drives it from this terminal; Ctrl-\ detaches. A chat's run
  has no terminal until then: attach takes it over, and the chat gets it back when it
  is reattached from Chat, which launches the name again.
- ``knowop run log NAME [--follow]`` prints its scrollback.
- ``knowop run send NAME TEXT`` types a message in and presses Enter.
- ``knowop run stop NAME`` stops it and waits until it has exited.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast

import argparse
import asyncio
import logging
import os
import re
import select
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

from trackinizer.lib.posix.attach import attach, stop, submit
from trackinizer.lib.posix.host import HostSpec
from trackinizer.lib.posix.terminal import write_all
from trackinizer.lib.userdirs import state_dir
from trackinizer.trax.daemon.protocol import socket_address
from trackinizer.wire.wire_sessions import RUN_NAME_PATTERN
from treekle import json_dumps, json_loads

from knowop.agent import headless
from knowop.agent.custom_types import HostRequest, RunRequest
from knowop.agent.program import PROGRAM


if TYPE_CHECKING:
    from collections.abc import Sequence

    from knowop.agent.headless import Reply


__all__ = [
    "HOST_COMMANDS",
    "HeadlessSpec",
    "HostRecord",
    "build_parser",
    "launch",
    "live_hosts",
    "main",
    "run_host",
    "stop_all",
]

_logger = logging.getLogger(__name__)


HOST_COMMANDS: Final = frozenset({"attach", "log", "ls", "send", "stop"})
"""``knowop run`` first words that reach an existing host rather than wrap a CLI."""

_LIVE: Final = frozenset({"starting", "running", "exiting"})
"""The states ``knowop run ls`` lists without ``--all``: a host that is still there."""

# A path component: no separator, no leading dot, and short enough that the
# directory stays far inside every filesystem's 255-byte name limit. ``#`` and
# ``@`` are allowed because routing names (``scientist#2``) make natural host
# names.
_NAME: Final = re.compile(RUN_NAME_PATTERN)

# How long ``--detach`` waits for the host to accept. Opening a synced session
# is a server round trip with retries before the CLI is even spawned, so this is
# generous; a host still starting past it is reported, not killed.
_READY_TIMEOUT_SEC: Final = 120.0

# How long ``stop`` waits for the host process to finish once the CLI is gone:
# its teardown drains capture and closes the session under a 30s deadline of
# its own, and the name is not reusable until it has.
_RESUME_WINDOW_SEC: Final = 120.0
"""How long after a taken-over run ended a launch of its name still resumes its session."""

_STOP_TIMEOUT_SEC: Final = 60.0


@dataclass(frozen=True, slots=True, kw_only=True)
class HostRecord:
    """What ``host.json`` says about one host."""

    name: str
    command: str
    """The wrapped command line, for display."""

    cwd: str
    pid: int
    """The host process -- not the CLI, which is its child."""

    address: str
    """The socket it listens on. Recorded, not re-derived: a long state path
    maps to an alias under the temp dir, and a client with a different TMPDIR
    would derive a different one."""

    started: str
    state: str = "starting"
    """``starting``, then ``running`` once it accepts, then ``exited``."""

    exit_code: int | None = None
    ended: str | None = None
    machine: str = ""
    """The computer the host runs on, which alone can attach to it; empty in a record
    from before hosts named theirs."""

    control: str = ""
    """The socket a chat's headless run listens on for attach, send and stop, recorded
    for the reason ``address`` is. Nothing listens there for a run that has a terminal
    of its own."""

    session: str = ""
    """The CLI session a person took the run over on; empty for a run never taken over.
    A launch of the same name resumes it, so the chat's turns go on in the session the
    terminal continued."""


@dataclass(frozen=True, slots=True, kw_only=True)
class HeadlessSpec(HostSpec):
    """Where a host is served: the terminal's socket, and the control socket."""

    control: Path
    """Where a headless run listens for control requests."""

    resume: str = ""
    """The CLI session of the taken-over run this host replaces; empty for none."""

    on_takeover: Callable[[str], None] | None = None
    """Called with the CLI session once a person has taken the run over on it."""


def launch(
    name: str,
    *,
    command: str,
    request: RunRequest,
) -> int:
    """Start ``request`` in a detached host named ``name``; return once it accepts.

    Args:
      name: The host's name; must not belong to a live host.
      command: The wrapped command line, recorded for ``knowop run ls``.
      request: Plain runner inputs. The new interpreter rebuilds the runner and
        opens its own connections. What it writes to stderr before the host
        accepts is repeated here.

    Returns:
      exit_code: 0 once the host accepts; 1 when it failed or is not up yet.

    """
    paths = _paths(name)
    previous = _read_record(paths)
    resume = "" if previous is None else _replace(name, previous, command=command)
    shutil.rmtree(paths, ignore_errors=True)
    paths.mkdir(parents=True, mode=0o700)
    address = socket_address(paths / "host.sock")
    control = socket_address(paths / "control.sock")
    ready_r, ready_w = os.pipe()
    try:
        _spawn(
            HostRequest(
                paths=str(paths),
                address=str(address),
                control=str(control),
                resume=resume,
                command=command,
                ready=ready_w,
                run=request,
            ),
            ready=ready_w,
            log=paths / "host.log",
        )
    except BaseException:
        os.close(ready_r)
        raise
    finally:
        os.close(ready_w)
    message = _await_ready(ready_r)
    if message == b"ok":
        # What the host said before it accepted, such as how much of a chat it
        # loaded, is said to the person who started it.
        sys.stderr.write(_tail(paths / "host.log"))
        sys.stderr.write(
            f"[{PROGRAM}] {name} is running detached; "
            f"attach with: {PROGRAM} attach {name}\n",
        )
        return 0
    reason = "did not accept in time" if message is None else "failed to start"
    sys.stderr.write(f"{PROGRAM}: host {name!r} {reason}; its log follows\n")
    sys.stderr.write(_tail(paths / "host.log"))
    return 1


def run_host(
    fd: int,
    *,
    rebuild: Callable[[RunRequest], Callable[[HostSpec | None], int]],
) -> int:
    """Read and close the inherited JSON descriptor, then run its host.

    Args:
      fd: The launcher's inherited payload descriptor.
      rebuild: The runner's fixed factory for rebuilding its start callable.

    Returns:
      exit_code: The hosted runner's status.

    """
    with os.fdopen(fd, "r") as source:
        request = json_loads(source.read(), HostRequest)
    return _detach(
        Path(request["paths"]),
        address=Path(request["address"]),
        control=Path(request["control"]),
        resume=request["resume"],
        command=request["command"],
        ready=request["ready"],
        start=rebuild(request["run"]),
    )


def main(argv: Sequence[str], *, prog: str = PROGRAM) -> int:
    """Run ``knowop run ls|attach|log|send|stop``; return the exit code.

    Args:
      argv: The arguments after ``knowop run``, starting with the command.
      prog: The program name the usage shows, ``knowop run``.

    Returns:
      exit_code: The command's status; for ``attach``, the CLI's when it
        exited while attached.

    """
    flags = cast(_Flags, build_parser(prog).parse_args(argv))
    if flags.command == "ls":
        return _ls(every=flags.all)
    if flags.command == "attach":
        return _attach(flags.name)
    if flags.command == "log":
        return _log(flags.name, follow=flags.follow)
    if flags.command == "send":
        return _send(flags.name, " ".join(flags.text))
    if flags.all:
        return stop_all()
    if not flags.name:
        raise SystemExit(f"{prog} stop: name a host, or stop them all with --all")
    return _stop(flags.name)


def build_parser(prog: str = PROGRAM) -> argparse.ArgumentParser:
    """Return the parser for ``knowop run ls|attach|log|send|stop``.

    Args:
      prog: The program name the usage shows.

    Returns:
      parser: The parser, with one subcommand per host command.

    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    return parser


class _Readiness:
    """The launcher's end of the readiness pipe, answered at most once."""

    def __init__(self, fd: int) -> None:
        self._fd = fd

    def accept(self) -> None:
        """Tell the launcher the host is listening."""
        self._finish(b"ok")

    def close(self) -> None:
        """Release the pipe; a launcher still waiting reads a failure."""
        self._finish(b"")

    def _finish(self, message: bytes) -> None:
        if self._fd < 0:
            return
        _ = write_all(self._fd, message)
        os.close(self._fd)
        self._fd = -1


def live_hosts() -> list[HostRecord]:
    """Return the hosts still there on this computer, by name.

    Returns:
      hosts: Each host starting, running or exiting here; a host another computer
        sharing this home directory runs is not one.

    """
    here = socket.gethostname()
    return [
        record
        for record in _records()
        if (not record.machine or record.machine == here) and _state(record) in _LIVE
    ]


def stop_all() -> int:
    """Stop every host still there on this computer, one at a time.

    Returns:
      exit_code: 0 when each stopped, else 1.

    """
    hosts = live_hosts()
    if not hosts:
        sys.stderr.write(f"[{PROGRAM}] no host is running on this computer\n")
    status = 0
    for record in hosts:
        status = max(status, _stop(record.name))
    return status


def _records() -> list[HostRecord]:
    """Return every host's record, by name; a directory with none is skipped."""
    hosts = state_dir() / "rekursiv-ai" / "trax" / "run" / "hosts"
    found = (
        _read_record(directory)
        for directory in sorted(hosts.iterdir() if hosts.is_dir() else ())
    )
    return [record for record in found if record is not None]


def _ls(*, every: bool) -> int:
    """List the live hosts, or with ``every`` all of them, one per line."""
    rows = [("NAME", "STATE", "PID", "STARTED", "COMMAND")]
    for record in _records():
        state = _state(record)
        if every or state in _LIVE:
            rows.append(
                (
                    record.name,
                    state,
                    str(record.pid),
                    record.started,
                    # One host, one line: a command with a newline in it (a script
                    # given to ``bash -c``) would spill lines that read as hosts.
                    " ".join(record.command.split()),
                ),
            )
    # The command, last, is left ragged: padding it would only trail spaces.
    widths = [
        max(len(row[column]) for row in rows) for column in range(len(rows[0]) - 1)
    ]
    for row in rows:
        cells = [row[column].ljust(width) for column, width in enumerate(widths)]
        sys.stdout.write("  ".join([*cells, row[-1]]) + "\n")
    return 0


# A chat's run is headless: it has no terminal until a person asks for one. The run is
# asked to stop its driver and host the CLI's own terminal on the same session, and only
# then is this terminal attached to it, the way it is to any host. A host with a
# terminal of its own answers no such request and is attached as it always was.
def _attach(name: str) -> int:
    """Drive the host from this terminal until it exits or is detached."""
    record = _require_local(name)
    taken = _control(record, "takeover")
    if taken is not None and not taken.ok:
        raise SystemExit(
            f"{PROGRAM}: {name} cannot be taken over: {taken.reason}",
        )
    try:
        status = asyncio.run(attach(Path(record.address)))
    except OSError:
        raise SystemExit(
            f"{PROGRAM}: host {name!r} is {_state(record)}",
        ) from None
    if status is None:
        sys.stderr.write(
            f"\n[{PROGRAM}] detached from {name}; it is still running\n",
        )
    else:
        sys.stderr.write(f"\n[{PROGRAM}] {name} exited with status {status}\n")
    if taken is not None:
        sys.stderr.write(
            f"[{PROGRAM}] Chat stays taken over; use Reattach to Chat in the chat or "
            "the Console to give it back.\n",
        )
    return 0 if status is None else status


# A follower stops once the host no longer accepts -- and reads once more after, since
# the host writes its last bytes before it closes the socket and the read before the
# check can have missed them.
def _log(name: str, *, follow: bool) -> int:
    """Print the host's scrollback; with ``follow``, until the host exits."""
    record = _require_record(name)
    scrollback = _paths(name) / "scrollback.log"
    if not scrollback.exists():
        return 0
    out = sys.stdout.buffer
    with scrollback.open("rb") as source:
        while True:
            chunk = source.read(65_536)
            if chunk:
                _ = out.write(chunk)
                out.flush()
                continue
            if not follow or not _accepts(Path(record.address)):
                _ = out.write(source.read())
                out.flush()
                return 0
            time.sleep(0.2)


def _send(name: str, text: str) -> int:
    """Type ``text`` into the host and press Enter."""
    record = _require_local(name)
    sent = _reach(record, lambda address: headless.say(address, text))
    if sent is not None:
        if not sent.ok:
            sys.stderr.write(
                f"{PROGRAM}: {name} did not take the message: {sent.reason}\n",
            )
        return 0 if sent.ok else 1
    try:
        submitted = asyncio.run(submit(Path(record.address), text=text))
    except OSError:
        raise SystemExit(
            f"{PROGRAM}: host {name!r} is {_state(record)}",
        ) from None
    if not submitted:
        sys.stderr.write(
            f"{PROGRAM}: {name} did not take the message; it is exiting\n",
        )
        return 1
    return 0


def _stop(name: str) -> int:
    """Stop the host's CLI, then wait until the host itself has exited."""
    record = _require_local(name)
    # A headless run ends whole: stopping its terminal alone would leave it running.
    ended = _control(record, "stop")
    detail = ""
    if ended is None:
        try:
            detail = f" (status {asyncio.run(stop(Path(record.address)))})"
        except OSError:
            raise SystemExit(
                f"{PROGRAM}: host {name!r} is {_state(record)}",
            ) from None
    if not _wait_exit(record):
        sys.stderr.write(
            f"{PROGRAM}: {name} stopped its CLI but the host (pid {record.pid}) "
            f"is still closing after {_STOP_TIMEOUT_SEC:.0f}s\n",
        )
        return 1
    sys.stderr.write(f"[{PROGRAM}] {name} stopped{detail}\n")
    return 0


def _wait_exit(record: HostRecord) -> bool:
    """Wait for the host process to end; return whether it did within the time."""
    deadline = time.monotonic() + _STOP_TIMEOUT_SEC
    while _alive(record.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not _alive(record.pid)


# The launch that gives a chat its CLI back finds the run a person took over holding the
# name. The server turns that run away as it opens the new one, so the old run may be
# ending already, or still answering: either way it is gone before the new one starts,
# and the new one resumes the session the terminal continued. A run never taken over,
# or one on another CLI, leaves nothing to resume; one that is live and not taken over
# keeps its name.
def _replace(name: str, previous: HostRecord, *, command: str) -> str:
    """End the taken-over run that holds ``name``; return the session to resume."""
    state = _state(previous)
    session = previous.session
    if state in {"starting", "running", "exiting"}:
        reply = _control(previous, "state")
        if reply is not None and reply.state == "taken_over":
            session = reply.session
            _ = _control(previous, "stop")
        elif reply is not None or not previous.session:
            why = "" if reply is None else " (the chat is not taken over)"
            raise SystemExit(
                f"{PROGRAM}: host {name!r} is {state}{why}; "
                f"stop it ({PROGRAM} stop {name}) or choose another --name",
            )
        if not _wait_exit(previous):
            raise SystemExit(
                f"{PROGRAM}: host {name!r} is still closing; "
                f"stop it ({PROGRAM} stop {name}) or choose another --name",
            )
    same_cli = previous.command.split()[:1] == command.split()[:1]
    return session if same_cli and _recent(previous, now=datetime.now(UTC)) else ""


# A session a person took over is resumed by the launch that follows it, not by one that
# comes after the chat has been answered by another provider, whose turns it lacks.
def _recent(record: HostRecord, *, now: datetime) -> bool:
    """Whether the host is still there or ended no longer ago than a reattach takes."""
    if record.ended is None:
        return True
    return now - datetime.fromisoformat(record.ended) < timedelta(
        seconds=_RESUME_WINDOW_SEC,
    )


def _require_record(name: str) -> HostRecord:
    """Return ``name``'s record, or exit naming the missing host."""
    record = _read_record(_paths(name))
    if record is None:
        raise SystemExit(
            f"{PROGRAM}: no host named {name!r}; see {PROGRAM} ls --all",
        )
    return record


# A home directory shared between computers shows each one the others' hosts, whose
# sockets and processes are not here to reach.
def _require_local(name: str) -> HostRecord:
    """Return ``name``'s record, or exit when the host runs on another computer."""
    record = _require_record(name)
    here = socket.gethostname()
    if record.machine and record.machine != here:
        raise SystemExit(
            f"{PROGRAM}: host {name!r} runs on {record.machine}, not on this "
            f"machine ({here}); run this command there",
        )
    return record


# A host with a terminal of its own, or one from before runs were headless, records no
# control socket or has nothing listening on it.
def _control(
    record: HostRecord,
    op: Literal["state", "takeover", "stop"],
) -> Reply | None:
    """Ask a headless run to do ``op``; ``None`` when the host is not one."""
    return _reach(record, lambda address: headless.request(address, op))


# A host with a terminal of its own, or one from before runs were headless, records no
# control socket or has nothing listening on it, and is not one.
def _reach(record: HostRecord, call: Callable[[Path], Reply]) -> Reply | None:
    """Make ``call`` at the host's control socket; ``None`` when the host has none."""
    if not record.control:
        return None
    try:
        return call(Path(record.control))
    except (OSError, ValueError):
        return None


def _read_record(paths: Path) -> HostRecord | None:
    """Return the host's record, or None when it has none (or none readable)."""
    try:
        text = (paths / "host.json").read_text()
        return json_loads(text, HostRecord)
    except (OSError, ValueError, TypeError):
        return None


# Written whole and renamed into place, so a reader racing a state change sees the old
# record or the new one, never half of either.
def _write_record(paths: Path, record: HostRecord) -> None:
    """Persist ``record`` as the host's ``host.json``."""
    staged = paths / "host.json.tmp"
    _ = staged.write_text(json_dumps(record, target=HostRecord, indent=2) + "\n")
    _ = staged.replace(paths / "host.json")


# The record says what the host last wrote; the socket and the pid say whether it is
# still there to say otherwise. A host killed outright never writes ``exited``, and
# reports ``lost`` rather than a ``running`` nothing answers for.
def _state(record: HostRecord) -> str:
    """Return ``record``'s live state: its own, ``exiting``, or ``lost``."""
    if record.state == "exited":
        return f"exited {record.exit_code}"
    if _accepts(Path(record.address)) or (
        record.control and _accepts(Path(record.control))
    ):
        return "running"
    if not _alive(record.pid):
        return "lost"
    return "exiting" if record.state == "running" else record.state


def _accepts(address: Path) -> bool:
    """Whether a host is accepting connections at ``address``."""
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.5)
        probe.connect(str(address))
    except OSError:
        return False
    else:
        return True
    finally:
        probe.close()


def _alive(pid: int) -> bool:
    """Whether a process with ``pid`` exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _tail(path: Path, *, lines: int = 20) -> str:
    """Return the last ``lines`` lines of ``path``, or nothing if unreadable."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    return "".join(text.splitlines(keepends=True)[-lines:])


def _now() -> str:
    """Return the current UTC time to the second, ISO formatted."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _spawn(request: HostRequest, *, ready: int, log: Path) -> None:
    """Exec a host in a new session, passing JSON inputs and its readiness pipe."""
    # macOS TLS/framework state cannot survive fork without exec. Pass plain
    # inputs, never the launcher's running interpreter or clients.
    with (
        tempfile.TemporaryFile() as payload,
        os.fdopen(
            os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600),
            "wb",
        ) as out,
    ):
        _ = payload.write(json_dumps(request, target=HostRequest).encode())
        _ = payload.seek(0)
        process = subprocess.Popen(  # noqa: S603 -- Fixed interpreter and package; only an inherited fd is passed.
            [
                sys.executable,
                "-m",
                __name__.rsplit(".", 1)[0],
                "--_host",
                str(payload.fileno()),
            ],
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=out,
            start_new_session=True,
            pass_fds=(payload.fileno(), ready),
        )
    # Reap while the launcher lives, without making its exit wait for the host.
    threading.Thread(target=process.wait, daemon=True).start()


class _Flags(Protocol):
    """Parsed host-command flags."""

    command: str
    name: str
    follow: bool
    text: list[str]
    all: bool


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the host commands on ``parser``."""
    commands = parser.add_subparsers(dest="command", required=True)
    ls_parser = commands.add_parser("ls", help="List the live hosts and their state.")
    ls_parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="List every host, those that exited or were lost too.",
    )
    attach_parser = commands.add_parser(
        "attach",
        help="Drive a host from this terminal; Ctrl-\\ detaches.",
    )
    attach_parser.add_argument("name")
    log_parser = commands.add_parser("log", help="Print a host's scrollback.")
    log_parser.add_argument("name")
    log_parser.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="Keep printing new output until the host exits.",
    )
    send_parser = commands.add_parser(
        "send",
        help="Type a message into a host and press Enter.",
    )
    send_parser.add_argument("name")
    send_parser.add_argument("text", nargs="+")
    stop_parser = commands.add_parser("stop", help="Stop a host and wait for it.")
    stop_parser.add_argument("name", nargs="?", default="")
    stop_parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Stop every live host on this computer.",
    )


def _paths(name: str) -> Path:
    """Return ``name``'s host directory, rejecting a name that is not a filename."""
    if _NAME.fullmatch(name) is None:
        raise SystemExit(
            f"{PROGRAM}: invalid host name {name!r}; use letters, digits, and "
            "._@#+=:- (at most 128, not starting with punctuation)",
        )
    return state_dir() / "rekursiv-ai" / "trax" / "run" / "hosts" / name


def _detach(
    paths: Path,
    *,
    address: Path,
    control: Path,
    resume: str,
    command: str,
    ready: int,
    start: Callable[[HostSpec], int],
) -> int:
    """Serve as the exec'd host in its new session; return its status."""
    os.set_inheritable(ready, False)
    _redirect_stdio(paths / "host.log")
    record = HostRecord(
        name=paths.name,
        command=command,
        cwd=str(Path.cwd()),
        pid=os.getpid(),
        address=str(address),
        started=_now(),
        machine=socket.gethostname(),
        control=str(control),
    )
    _write_record(paths, record)
    readiness = _Readiness(ready)
    status = 1
    try:
        status = start(
            HeadlessSpec(
                address=address,
                scrollback=paths / "scrollback.log",
                on_listening=partial(_listening, paths, record, readiness),
                control=control,
                resume=resume,
                on_takeover=partial(_taken_over, paths),
            ),
        )
    except SystemExit as exit_:
        status = _exit_status(exit_.code)
    except Exception:
        # Nothing above this frame will report it: the host has no caller, and
        # the launcher shows only the log.
        _logger.exception("knowop run: the host failed")
    finally:
        readiness.close()
        # Read again: the record now may hold a session the first one did not.
        _write_record(
            paths,
            replace(
                _read_record(paths) or record,
                state="exited",
                exit_code=status,
                ended=_now(),
            ),
        )
        sys.stdout.flush()
        sys.stderr.flush()
    return status


# The runner's clean refusals (a binary not on PATH) arrive as a ``SystemExit`` with a
# message, which only the interpreter's top level would print -- and the launcher shows
# the log, so the reason must land there.
def _exit_status(code: object) -> int:
    """Return the status a ``SystemExit`` code means, logging a message code."""
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    sys.stderr.write(f"{code}\n")
    return 1


def _taken_over(paths: Path, session: str) -> None:
    """Record the CLI session a person took the host over on."""
    record = _read_record(paths)
    if record is not None:
        _write_record(paths, replace(record, session=session))


def _listening(paths: Path, record: HostRecord, readiness: _Readiness) -> None:
    """Record the host as running, then release the launcher."""
    _write_record(paths, replace(record, state="running"))
    readiness.accept()


def _redirect_stdio(log: Path) -> None:
    """Point stdin at /dev/null and stdout and stderr at ``log``."""
    null = os.open(os.devnull, os.O_RDONLY)
    out = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    _ = os.dup2(null, 0)
    _ = os.dup2(out, 1)
    _ = os.dup2(out, 2)
    os.close(null)
    os.close(out)


def _await_ready(fd: int) -> bytes | None:
    """Return the host's readiness message, or None if it never answered."""
    try:
        readable, _, _ = select.select([fd], (), (), _READY_TIMEOUT_SEC)
        return os.read(fd, 2) if readable else None
    finally:
        os.close(fd)
