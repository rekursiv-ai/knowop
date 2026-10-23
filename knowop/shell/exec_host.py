"""Run the commands the server's assistant sends to this machine, and report each result.

A machine host that serves commands holds a request on the server for the next one.
Each command runs as the user who started ``knowop connect`` (``exec_command.py``), and is
killed when the machine is revoked or the host stops.

Every command is written to a log only this user can read, with who the server says
asked for it and the names of the variables it set. The line is written when the
command ends, so a dishonest server can lie about the asker but cannot run a command
that leaves no line, unless the command takes this host down with it.
"""

from __future__ import annotations

from contextlib import closing, suppress
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Protocol

import asyncio
import os
import threading
import time
import uuid

from pydantic import BaseModel, ConfigDict
from trackinizer.client.errors import ClientError
from trackinizer.lib.userdirs import state_dir
from trackinizer.trax.render import echo
from trackinizer.wire import wire_exec

from knowop.shell.exec_command import (
    result_of,
    run_until_ended,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from trackinizer.wire.wire_exec import ExecResult, ExecWait


__all__ = ["ExecClient", "ExecHost"]

_POST_TRIES: Final = 3
"""Times a result is sent before it is given up on."""


class ExecClient(Protocol):
    """The calls the exec host makes: the ``Client`` methods, and closing it."""

    def wait_exec(
        self,
        machine_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        wait_sec: float,
    ) -> ExecWait | None:
        """Hold for the next command."""
        ...

    def post_exec_result(
        self,
        machine_id: uuid.UUID,
        exec_id: uuid.UUID,
        result: ExecResult,
    ) -> None:
        """Report what a command produced."""
        ...

    def close(self) -> None:
        """Release the connection."""
        ...


class ExecHost:
    """Runs the commands the server hands this machine, each on its own thread.

    Args:
      name: The machine's name, which names the log.
      credential: The machine credential, which the client sends.
      client_for: Returns a client that sends the given credential.
      retry_sec: How long to wait before asking again after a failed wait, or before
        sending a result again.

    """

    def __init__(
        self,
        *,
        name: str,
        credential: str,
        client_for: Callable[[str], ExecClient],
        retry_sec: float = 5.0,
    ) -> None:
        self._name = name
        self._credential = credential
        self._client_for = client_for
        self._retry_sec = retry_sec
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._workers: dict[uuid.UUID, threading.Thread] = {}
        self._ends: dict[
            uuid.UUID,
            tuple[asyncio.AbstractEventLoop, asyncio.Event],
        ] = {}
        self._listener: threading.Thread | None = None

    def start(self, machine_id: uuid.UUID, instance: uuid.UUID) -> None:
        """Start listening for commands on a thread.

        Args:
          machine_id: This machine's id.
          instance: This host's instance.

        """
        self._listener = threading.Thread(
            name="exec-listener",
            target=self._listen,
            args=(machine_id, instance),
            daemon=True,
        )
        self._listener.start()

    def stop(self) -> None:
        """Stop listening and kill every command; returns once they have ended."""
        self._stop.set()
        self._end_all()
        with self._lock:
            workers = list(self._workers.values())
        for worker in workers:
            worker.join(timeout=10)

    def _listen(self, machine_id: uuid.UUID, instance: uuid.UUID) -> None:
        """Wait for commands until stopped or revoked, starting a thread for each."""
        client = self._client_for(self._credential)
        try:
            while not self._stop.is_set():
                try:
                    command = client.wait_exec(
                        machine_id,
                        instance=instance,
                        wait_sec=wire_exec.WAIT_MAX_SEC,
                    )
                except ClientError as err:
                    if err.status_code == 410:
                        # Revoked: nothing more starts, and a command that has not yet
                        # registered to be ended is ended as it does.
                        self._stop.set()
                        self._end_all()
                        return
                    echo(f"knowop: exec wait failed: {err}", err=True)
                    self._stop.wait(self._retry_sec)
                    continue
                if command is not None:
                    self._start_worker(command, machine_id=machine_id)
        finally:
            client.close()

    def _start_worker(self, command: ExecWait, *, machine_id: uuid.UUID) -> None:
        """Run a command on a thread of its own."""
        if self._stop.is_set():
            # A wait already in flight when the host stopped handed this over; the
            # server took it, so its caller is told rather than left to time out.
            self._post(
                machine_id,
                command.exec_id,
                result_of(
                    1,
                    stderr=b"knowop connect stopped before the command started\n",
                    cwd=command.cwd,
                    started=time.monotonic(),
                ),
            )
            return
        with self._lock:
            worker = threading.Thread(
                name=f"exec-{command.exec_id}",
                target=self._run,
                args=(command, machine_id),
                daemon=True,
            )
            self._workers[command.exec_id] = worker
            worker.start()

    def _end_all(self) -> None:
        """Ask every running command to end; each kills its own process group."""
        with self._lock:
            ends = list(self._ends.values())
        for loop, ended in ends:
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(ended.set)

    def _run(self, command: ExecWait, machine_id: uuid.UUID) -> None:
        """Run one command, log it, report it, then forget its thread."""
        try:
            started = time.monotonic()
            result, killed = asyncio.run(self._execute(command))
            self._record(
                command,
                result,
                took_ms=round((time.monotonic() - started) * 1000),
            )
            # A command ended by a revoke or a stop has no one waiting for its result.
            if not killed:
                self._post(machine_id, command.exec_id, result)
        finally:
            with self._lock:
                self._workers.pop(command.exec_id, None)

    async def _execute(self, command: ExecWait) -> tuple[ExecResult, bool]:
        """Run the command; also say whether the host ended it."""
        ended = asyncio.Event()
        with self._lock:
            self._ends[command.exec_id] = (asyncio.get_running_loop(), ended)
        if self._stop.is_set():
            ended.set()
        try:
            result, ending = await run_until_ended(command, ended=ended)
            return result, ending == "ended"
        finally:
            with self._lock:
                self._ends.pop(command.exec_id, None)

    def _post(
        self,
        machine_id: uuid.UUID,
        exec_id: uuid.UUID,
        result: ExecResult,
    ) -> None:
        """Send a result, again when it fails in transit."""
        with closing(self._client_for(self._credential)) as client:
            for attempt in range(1, _POST_TRIES + 1):
                try:
                    client.post_exec_result(machine_id, exec_id, result)
                except ClientError as err:
                    # An answer a retry cannot change: the caller gave up, or the
                    # machine was revoked.
                    if err.status_code is not None and 400 <= err.status_code < 500:
                        return
                    if attempt == _POST_TRIES:
                        echo(f"knowop: exec result not delivered: {err}", err=True)
                        return
                    self._stop.wait(self._retry_sec)
                else:
                    return

    def _record(self, command: ExecWait, result: ExecResult, *, took_ms: int) -> None:
        """Log a finished command; a log that cannot be written never holds the result."""
        try:
            self._log(command, result, took_ms=took_ms)
        except OSError as err:
            # The command has run and its caller waits for the result.
            echo(f"knowop: exec log not written: {err}", err=True)

    def _log(self, command: ExecWait, result: ExecResult, *, took_ms: int) -> None:
        """Append one line to the host's log, readable by this user only."""
        path = (
            state_dir() / "rekursiv-ai" / "trax" / "machines" / f"{self._name}.exec.log"
        )
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        line = _LogLine(
            at=datetime.now(UTC),
            exec_id=command.exec_id,
            who=command.asker,
            command=command.command,
            cwd=command.cwd,
            env=sorted(command.env or {}),
            exit_code=result.exit_code,
            duration_ms=took_ms,
        )
        # A log that already exists keeps its mode through open(), and a link in its
        # place would send the line elsewhere.
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(fd, "a") as file:
            os.fchmod(file.fileno(), 0o600)
            _ = file.write(line.model_dump_json() + "\n")


class _LogLine(BaseModel):
    """One line of the host-local exec log."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    at: datetime
    exec_id: uuid.UUID
    who: str
    command: str
    cwd: str
    env: list[str]
    exit_code: int
    duration_ms: int
