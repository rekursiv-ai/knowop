"""Run one command in a login shell of its own, and report what it produced.

The command runs as the user who started ``knowop connect``, in ``bash -lc``, in a process
group of its own, with standard input from the request and the host's environment less
its trax server, credential and profile. The group is killed at the command's timeout,
which is reported as exit code 124 with the reason on stderr, and when the host ends it.
A stream longer than 1 MiB is cut from its middle.
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Final, Literal

import asyncio
import base64
import os
import time

from trackinizer.lib.posix.child import Child
from trackinizer.wire import wire_exec
from trackinizer.wire.wire_exec import ExecRequest, ExecResult

from knowop.shell.exec_stream import Cut, read_report, script
from knowop.shell.trax_context import inherited_trax_context


__all__ = ["Ending", "exec_result", "run_command", "run_until_ended"]

_DRAIN_SEC: Final = 0.5
"""How long output may keep flowing after the shell exits. A job it left running in the
background holds the pipes open, and must not hold the result back."""

_CANNOT_RUN: Final = 126
"""The status a shell reports for a command it found but could not execute."""

_NOT_FOUND: Final = 127
"""The status a shell reports for a command it could not find."""

type Ending = Literal["exited", "timeout", "ended"]
"""How a command's supervision ended: it exited, ran out of time, or the host ended it."""


async def run_command(
    request: ExecRequest,
    *,
    ended: asyncio.Event | None = None,
) -> ExecResult:
    """Run a command in a login shell and report what it produced.

    A command that outlives ``request.timeout_sec``, or whose ``ended`` event is set,
    has its whole process group killed. A command that cannot start, because its
    folder is missing, is not executable by this user, or has no shell to run in,
    reports exit code 1, 126 or 127 with the reason on stderr.

    Args:
      request: The command, its folder and timeout, and optionally stdin and variables.
      ended: Set by the host to end the command early: a revoke, or a stop.

    Returns:
      result: The exit code (124 for a timeout, 128 plus the signal for a command
        that a signal killed), both streams, the folder it ended in, and whether
        either stream was cut.

    """
    result, _ = await run_until_ended(request, ended=ended)
    return result


async def run_until_ended(
    request: ExecRequest,
    *,
    ended: asyncio.Event | None,
) -> tuple[ExecResult, Ending]:
    """Run the command; also say how it ended.

    Args:
      request: The command, its folder and timeout, and optionally stdin and variables.
      ended: Set by the host to end the command early: a revoke, or a stop.

    Returns:
      result: What ``run_command`` returns.
      ending: ``exited`` for a command that ran to its end or never started,
        ``timeout`` for one killed at its timeout, ``ended`` for one the host killed.

    """
    started = time.monotonic()
    folder = _start_folder(request.cwd)
    if folder is None:
        return _refused(
            request,
            1,
            f"{request.cwd} is not a folder on this machine",
            started,
        )
    report_read, report_write = os.pipe()
    try:
        child = await _spawn(request, folder, report_write=report_write)
    except OSError as failure:
        os.close(report_read)
        status = _NOT_FOUND if isinstance(failure, FileNotFoundError) else _CANNOT_RUN
        return _refused(
            request,
            status,
            f"cannot start the command: {failure}",
            started,
        )
    finally:
        os.close(report_write)
    try:
        out, err = Cut(), Cut()
        ending, code = await _supervise(
            child,
            request,
            ended=ended,
            out=out,
            err=err,
        )
        ended_in = read_report(report_read, fallback=request.cwd)
    finally:
        os.close(report_read)
    result = _finished(
        code,
        request,
        ending=ending,
        streams=(out, err),
        ended_in=ended_in,
        started=started,
    )
    return result, ending


def exec_result(
    exit_code: int,
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
    cwd: str,
    truncated: bool = False,
    started: float,
) -> ExecResult:
    """Build the wire result from the raw parts.

    Args:
      exit_code: The command's status.
      stdout: What it printed, already within the wire's limit.
      stderr: What it printed on its error stream, already within the limit.
      cwd: The folder it ended in.
      truncated: Whether either stream was cut.
      started: The monotonic time it started, for the duration.

    Returns:
      result: The wire result, with both streams as base64.

    """
    return ExecResult(
        exit_code=exit_code,
        stdout_b64=base64.b64encode(stdout).decode(),
        stderr_b64=base64.b64encode(stderr).decode(),
        cwd=cwd,
        truncated=truncated,
        duration_ms=round((time.monotonic() - started) * 1000),
    )


def _finished(
    code: int,
    request: ExecRequest,
    *,
    ending: Ending,
    streams: tuple[Cut, Cut],
    ended_in: str,
    started: float,
) -> ExecResult:
    """Build the result of a command that ran and exited with ``code``."""
    out, err = streams
    # The reason is fed to the stream, so the cut keeps the whole result in the limit.
    if ending == "timeout":
        err.feed(f"timed out after {request.timeout_sec:g} s\n".encode())
    stderr, stderr_cut = err.value()
    stdout, stdout_cut = out.value()
    return exec_result(
        wire_exec.TIMEOUT_EXIT_CODE if ending == "timeout" else _exit_code(code),
        stdout=stdout,
        stderr=stderr,
        cwd=ended_in,
        truncated=stdout_cut or stderr_cut,
        started=started,
    )


async def _spawn(request: ExecRequest, folder: Path, *, report_write: int) -> Child:
    """Start the login shell for the command, in a process group of its own."""
    return await Child.start(
        "bash",
        "-lc",
        script(request.command, report_fd=report_write),
        # Its own group, so a kill reaches what the command started.
        group=True,
        # Killed outright, whatever it does with a termination signal.
        grace_sec=0.0,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=folder,
        env=_environment(request),
        pass_fds=(report_write,),
    )


# The server's assistant reads every stream back, so a credential the host holds in its
# environment must not reach the command; the request's own variables are the server's.
def _environment(request: ExecRequest) -> dict[str, str]:
    """Return the host's environment less its trax context, plus the request's."""
    withheld = inherited_trax_context()
    inherited = {
        name: value for name, value in os.environ.items() if name not in withheld
    }
    return {**inherited, **(request.env or {})}


async def _supervise(
    child: Child,
    request: ExecRequest,
    *,
    ended: asyncio.Event | None,
    out: Cut,
    err: Cut,
) -> tuple[Ending, int]:
    """Feed the command, collect its output, and kill it when it must end."""
    process = child.process
    if process.stdin is None or process.stdout is None or process.stderr is None:
        raise ValueError("The command was started without its pipes.")
    streams = [
        asyncio.create_task(_feed(process.stdin, _request_stdin(request))),
        asyncio.create_task(_collect(process.stdout, out)),
        asyncio.create_task(_collect(process.stderr, err)),
    ]
    exited = asyncio.create_task(child.exited())
    stop = asyncio.create_task((ended or asyncio.Event()).wait())
    try:
        done, _ = await asyncio.wait(
            {exited, stop},
            timeout=request.timeout_sec,
            return_when=asyncio.FIRST_COMPLETED,
        )
        ending: Ending = (
            "exited" if exited in done else "ended" if stop in done else "timeout"
        )
        code = await exited if ending == "exited" else await child.stop()
        # What the shell left running keeps the pipes open, so the streams get a
        # moment to finish and no more.
        _ = await asyncio.wait(streams, timeout=_DRAIN_SEC)
    except asyncio.CancelledError:
        _ = await child.stop()
        raise
    finally:
        tasks = [*streams, stop, exited]
        for task in tasks:
            _ = task.cancel()
        _ = await asyncio.wait(tasks)
    return ending, code


async def _feed(stdin: asyncio.StreamWriter, data: bytes) -> None:
    """Write the request's standard input, then close it so a reader sees the end."""
    try:
        if data:
            stdin.write(data)
            await stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        # The command finished without reading it all.
        pass
    finally:
        with suppress(BrokenPipeError, ConnectionResetError, OSError):
            stdin.close()


async def _collect(stream: asyncio.StreamReader, sink: Cut) -> None:
    """Read a stream to its end into ``sink``."""
    while chunk := await stream.read(65_536):
        sink.feed(chunk)


def _request_stdin(request: ExecRequest) -> bytes:
    """Return the bytes the request gives the command on standard input."""
    return b"" if request.stdin_b64 is None else base64.b64decode(request.stdin_b64)


def _start_folder(cwd: str) -> Path | None:
    """Return the folder a command starts in, or ``None`` when ``cwd`` is not one."""
    # ``~name`` names a user this machine may not have, which raises.
    try:
        folder = Path(cwd).expanduser()
    except RuntimeError:
        return None
    return folder if folder.is_dir() else None


def _exit_code(returncode: int) -> int:
    """Return the status a shell reports for a process: 128 plus the signal."""
    return 128 - returncode if returncode < 0 else returncode


def _refused(
    request: ExecRequest,
    exit_code: int,
    reason: str,
    started: float,
) -> tuple[ExecResult, Ending]:
    """Return the result of a command that never ran, with its reason on stderr."""
    result = exec_result(
        exit_code,
        stderr=f"{reason}\n".encode(),
        cwd=request.cwd,
        started=started,
    )
    return result, "exited"
