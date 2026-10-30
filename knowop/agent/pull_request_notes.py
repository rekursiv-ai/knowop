"""Note what a session's shell calls did to pull requests, from the records it emits.

The run's drain thread shows each record to :meth:`RecordNotes.note`. A call that ran
``gh pr create`` names its pull request in what it printed, and one that ran ``git
push`` sends the watch to look for the pull request of the branch. claude records a
shell result without its command, so the command is read from the call and its result
found by the call's id. A call that failed did neither.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, cast

import re
import threading

from trackinizer.sagent.session.types import ShellCommandResult, ToolCall

from knowop.agent.pull_requests import links


if TYPE_CHECKING:
    from trackinizer.types.streams import TraxRecord
    from treekle import PlainTree


__all__ = ["Noted", "RecordNotes"]


type _Act = Literal["create", "push"]
"""What a shell call did: opened a pull request, or pushed a branch."""

_CREATE: Final = re.compile(r"\bgh\s+pr\s+create\b")
_PUSH: Final = re.compile(r"\bgit\b[^\n;&|]*\bpush\b")


@dataclass(frozen=True, slots=True, kw_only=True)
class Noted:
    """What the session's shell calls did since the watch last took its notes.

    Attributes:
      opened: The links ``gh pr create`` printed.
      pushed: Whether a ``git push`` succeeded.

    """

    opened: tuple[str, ...] = ()
    pushed: bool = False


class RecordNotes:
    """The notes the drain thread takes and the watch's thread reads."""

    def __init__(self) -> None:
        """Hold no notes yet."""
        self._lock = threading.Lock()
        # Only the drain thread reads or writes it.
        self._calls: dict[str, _Act] = {}
        self._opened: list[str] = []
        self._pushed = False

    def note(self, record: TraxRecord) -> bool:
        """Note a shell call's result that opened a pull request or pushed a branch.

        Args:
          record: A record the session just emitted.

        Returns:
          noted: Whether the watch has something new to act on.

        """
        if isinstance(record, ToolCall):
            act = _act(" ".join(_strings(record.arguments)))
            if act is not None:
                self._calls[record.call_id] = act
            return False
        if not isinstance(record, ShellCommandResult):
            return False
        act = self._calls.pop(record.call_id, None) or _act(
            " ".join(record.command or ()),
        )
        if act is None or record.is_error is True or record.exit_code not in {None, 0}:
            return False
        with self._lock:
            if act == "create":
                self._opened.extend(links(record.stdout))
            else:
                self._pushed = True
        return True

    def take(self) -> Noted:
        """Return the notes taken since the last call, each once."""
        with self._lock:
            noted = Noted(opened=tuple(self._opened), pushed=self._pushed)
            self._opened, self._pushed = list[str](), False
        return noted


def _act(command: str) -> _Act | None:
    """Say whether a shell command opened a pull request or pushed a branch."""
    if _CREATE.search(command):
        return "create"
    return "push" if _PUSH.search(command) else None


def _strings(tree: Mapping[str, PlainTree]) -> list[str]:
    """Return every string in a tool call's arguments: its command, however spelled."""
    found: list[str] = []
    stack: list[object] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            found.append(node)
        elif isinstance(node, Mapping):
            stack.extend(cast("Mapping[str, object]", node).values())
        elif isinstance(node, Sequence):
            stack.extend(cast("Sequence[object]", node))
    return found
