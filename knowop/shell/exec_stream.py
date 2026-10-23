"""The streams of a command the exec host runs: cutting them, and the folder report.

A command's output is kept to the wire's limit as it arrives, so a command that prints
without end never grows this process. The shell reports the folder it ended in on a
pipe of its own, since standard output belongs to the command.
"""

from __future__ import annotations

from collections import deque
from typing import Final

import os
import re

from trackinizer.wire import wire_exec


__all__ = ["Cut", "read_report", "script"]

_CUT_MARK_BYTES: Final = 64
"""Room kept inside the stream limit for the line that says how much was cut."""

_ENDED_IN: Final = re.compile(r"^/[^\x00-\x1f\x7f]{0,1023}$")
"""A folder the wire carries: absolute, with no control character."""


class Cut:
    """Keeps the start and the end of a stream that outgrows the wire's limit."""

    def __init__(self) -> None:
        self._half = (wire_exec.MAX_STREAM_BYTES - _CUT_MARK_BYTES) // 2
        self._head = bytearray()
        self._tail: deque[bytes] = deque()
        self._tail_len = 0
        self._total = 0

    def feed(self, chunk: bytes) -> None:
        """Take the next bytes of the stream.

        Args:
          chunk: Bytes just read from the stream.

        """
        self._total += len(chunk)
        room = self._half - len(self._head)
        if room > 0:
            self._head += chunk[:room]
            chunk = chunk[room:]
        if not chunk:
            return
        self._tail.append(chunk)
        self._tail_len += len(chunk)
        # Enough is kept that a stream at the limit loses nothing; only one past it
        # gives up its middle.
        keep = wire_exec.MAX_STREAM_BYTES - self._half
        while self._tail_len - len(self._tail[0]) >= keep:
            self._tail_len -= len(self._tail.popleft())

    def value(self) -> tuple[bytes, bool]:
        """Return the stream, cut from the middle when it passed the limit.

        Returns:
          stream: Everything read, or its first and last halves around a line saying
            how many bytes were cut.
          cut: Whether the stream passed the limit.

        """
        kept = bytes(self._head) + b"".join(self._tail)
        if self._total <= wire_exec.MAX_STREAM_BYTES:
            return kept, False
        cut = self._total - 2 * self._half
        mark = f"\n[... {cut} bytes cut ...]\n".encode()
        return kept[: self._half] + mark + kept[-self._half :], True


# The shell reports its own folder as it exits, so a ``cd`` anywhere in the command
# persists, and so does the folder of a command that calls ``exit``. A command that
# replaces the shell (``exec``) never reports, and the folder stays where it started.
def script(command: str, *, report_fd: int) -> str:
    """Return the shell text: report the folder on exit, then the command itself."""
    return f"trap 'printf %s \"$PWD\" >&{report_fd}' EXIT\n{command}"


def read_report(fd: int, *, fallback: str) -> str:
    """Return the folder the shell reported, or ``fallback`` when it reported none.

    Args:
      fd: The read end of the pipe the shell reports on, whose writer has closed.
      fallback: The folder to return when nothing sound was reported.

    Returns:
      folder: What the shell wrote, when it is a folder the wire carries.

    """
    os.set_blocking(fd, False)
    try:
        reported = os.read(fd, 4096).decode(errors="replace")
    except BlockingIOError:
        return fallback
    return reported if _ENDED_IN.fullmatch(reported) else fallback
