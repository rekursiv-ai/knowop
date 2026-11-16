"""Shared process arguments and start/attach handshake for CLI and agent tools."""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING, Final, Protocol

import queue
import threading
import time

from trackinizer.client.errors import ClientError
from trackinizer.wire.wire_session_relay import ProcessLine


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from contextlib import AbstractContextManager
    from datetime import timedelta
    from uuid import UUID

    from trackinizer.wire.wire_session_relay import (
        AgentProcess,
        ProcessSpec,
        ViewerFrame,
    )


_ATTACHED_SEC: Final = 120.0


class ProcessClient(Protocol):
    """The client calls needed to start and follow an agent process."""

    def start_process(self, session_id: UUID, spec: ProcessSpec) -> AgentProcess:
        """Start a process."""
        ...

    def lines_stream(
        self,
        session_id: UUID,
    ) -> AbstractContextManager[Iterator[ViewerFrame]]:
        """Open live frames, closing their HTTP stream on exit."""
        ...


def started_process(
    client: ProcessClient,
    *,
    session_id: UUID,
    spec: ProcessSpec,
) -> AgentProcess:
    """Start once the stream opens; return when this process attaches.

    Args:
      client: The session's client.
      session_id: The session to serve.
      spec: Where and how to run it.

    Returns:
      process: The attached process.

    Raises:
      ClientError: The server refused the start or the process detached.
      TimeoutError: The attachment deadline elapsed.

    """
    frames: queue.Queue[ViewerFrame | ClientError] = queue.Queue()
    stopped = threading.Event()
    with ExitStack() as cleanup, client.lines_stream(session_id) as lines:
        reader = threading.Thread(
            target=_read_lines,
            args=(lines, frames, stopped),
            name="lines",
            daemon=True,
        )
        reader.start()
        cleanup.callback(reader.join)
        try:
            deadline = time.monotonic() + _ATTACHED_SEC
            _ = _next_frame(frames, deadline=deadline)
            started = client.start_process(session_id, spec)
            process = started
            while process.state not in {"detached", "attached"}:
                frame = _next_frame(frames, deadline=deadline)
                if (
                    isinstance(frame, ProcessLine)
                    and frame.process.process_id == started.process_id
                ):
                    process = frame.process
            if process.state == "detached":
                raise ClientError(f"the process did not start: {process.reason}")
            return process
        finally:
            stopped.set()


def start_args(limit: timedelta | None, args: Sequence[str]) -> tuple[str, ...]:
    """Return the process's time limit followed by sagent's arguments."""
    timed = () if limit is None else ("--time", slurm_time(limit))
    return (*timed, *args)


def slurm_time(limit: timedelta) -> str:
    """Write ``limit`` as Slurm's ``D-HH:MM:SS``."""
    minutes, seconds = divmod(limit.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{limit.days}-{hours:02}:{minutes:02}:{seconds:02}"


def _read_lines(
    lines: Iterator[ViewerFrame],
    frames: queue.Queue[ViewerFrame | ClientError],
    stopped: threading.Event,
) -> None:
    try:
        for frame in lines:
            if stopped.is_set():
                break
            frames.put(frame)
    except ClientError as error:
        frames.put(error)


def _next_frame(
    frames: queue.Queue[ViewerFrame | ClientError],
    *,
    deadline: float,
) -> ViewerFrame:
    try:
        frame = frames.get(timeout=max(0.0, deadline - time.monotonic()))
    except queue.Empty:
        raise TimeoutError(
            f"the process did not attach within {_ATTACHED_SEC:g} s",
        ) from None
    if isinstance(frame, ClientError):
        raise frame
    return frame
