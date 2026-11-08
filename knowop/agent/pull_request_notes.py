"""Note what a session's shell calls did to pull requests, from the records it emits.

The run's drain thread shows each record of the run's agents, its sub-agents' too, to
:meth:`RecordNotes.note`. A call that ran ``gh pr create`` names its pull request in
what it printed, and one that ran ``git push`` sends the watch to look for the pull
request of the branch. claude records a shell result without its command, so the
command is read from the call and its result found by the call's id. A call that
failed did neither, and one that only names a command (``grep "gh pr create"``) ran
neither: each counts where a shell runs a command, first or after a separator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import re
import threading

from trackinizer.sagent.session.types import BashResult, ToolCall

from knowop.agent.pull_requests import links


if TYPE_CHECKING:
    from trackinizer.types.streams import TraxRecord
    from treekle import PlainTree


__all__ = ["Noted", "RecordNotes"]


type _Act = Literal["create", "push"]
"""What a shell call did: opened a pull request, or pushed a branch."""

# Where a shell runs a command: first, or after a separator, a ``then`` or a ``do``,
# behind any variable assignments.
_RUN: Final = r"(?:^|[\n;&|(]|\bthen|\bdo)\s*(?:\w+=\S*\s+)*"
_CREATE: Final = re.compile(rf"{_RUN}gh\s+pr\s+create\b")
_PUSH: Final = re.compile(rf"{_RUN}git\b[^\n;&|]*\bpush\b")


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
            act = _act(_strings(record.arguments))
            if act is not None:
                self._calls[record.call_id] = act
            return False
        if not isinstance(record, BashResult):
            return False
        act = self._calls.pop(record.call_id, None) or _act(record.command or ())
        if act is None or record.is_error is True or record.exit_code not in {None, 0}:
            return False
        with self._lock:
            if act == "create":
                self._opened.extend(links(record.text()))
            else:
                self._pushed = True
        return True

    def take(self) -> Noted:
        """Return the notes taken since the last call, each once."""
        with self._lock:
            noted = Noted(opened=tuple(self._opened), pushed=self._pushed)
            self._opened, self._pushed = list[str](), False
        return noted


# Each word alone, and all of them as one line: an argv (``["git", "push"]``) is one
# command only joined, and the script of ``["bash", "-lc", "git push"]`` starts its
# own word.
def _act(words: Sequence[str]) -> _Act | None:
    """Say whether a shell command opened a pull request or pushed a branch."""
    lines = (*words, " ".join(words))
    if any(_CREATE.search(line) for line in lines):
        return "create"
    return "push" if any(_PUSH.search(line) for line in lines) else None


def _strings(tree: Mapping[str, PlainTree]) -> list[str]:
    """Return every string in a tool call's arguments: its command, however spelled."""
    found: list[str] = []
    stack: list[PlainTree] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            found.append(node)
        elif isinstance(node, Mapping):
            stack.extend(reversed(list(node.values())))
        elif isinstance(node, Sequence):
            # Reversed onto the stack so the words pop in order: ``["git", "push"]``
            # read backwards is no push.
            stack.extend(reversed(node))
    return found
