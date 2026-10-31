"""Each sub-agent of a run in an AgentSession of its own, linked to its parent.

A CLI writes a sub-agent's transcript beside the run's own: Claude Code under
``<session>/subagents/``, Codex as a rollout of its own (:meth:`Adapter.subagent`).
:class:`SubagentSinks` is the run's sink, and sends each sub-agent's lines to a sink of
its own instead, opened under the sub-agent's own id, so a re-read or a resume lands on
the same session. Once both sessions are open, it links the sub-agent's to the one that
spawned it with a ``produced_by`` edge, so the sessions of a run form a tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import logging

from knowop.agent.sink import Sink


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from datetime import datetime
    from pathlib import Path
    from uuid import UUID

    from trackinizer.runs.adapters.custom_types import Adapter, Subagent
    from trackinizer.sagent.session.types import FileEditResult
    from trackinizer.types.streams import TraxRecord
    from trackinizer.wire.wire_session_ir import RecordBody

    from knowop.agent.custom_types import Event
    from knowop.agent.slash import SlashCommand


__all__ = ["SubagentSinks"]


_logger = logging.getLogger(__name__)


class SubagentSinks(Sink):
    """The run's sink, and one more per sub-agent, each linked to its parent.

    Args:
      root: The run's own sink: its session, its commands, its records.
      open_child: Opens the sink a sub-agent's records go to, under its own id.
      link: Links a sub-agent's session to its parent's, given both session ids and
        what spawned it; a failure is tried once more at close.

    """

    def __init__(
        self,
        root: Sink,
        *,
        open_child: Callable[[Subagent], Sink],
        link: Callable[[UUID, UUID, str], None],
    ) -> None:
        self._root = root
        self._open_child = open_child
        self._link = link
        # Read once per file: codex declares a sub-agent on its rollout's first line.
        self._subagents: dict[Path, Subagent | None] = {}
        # By the CLI's own id for each sub-agent, in the order they were seen, so a
        # parent is always linked before the children it spawned.
        self._children: dict[str, Sink] = {}
        self._spawned: dict[str, Subagent] = {}
        self._linked: set[str] = set()
        self._refused: set[str] = set()

    @property
    @override
    def session_id(self) -> UUID | None:
        return self._root.session_id

    @override
    def open(self) -> str | None:
        return self._root.open()

    # A sub-agent's file names it too, and the run's session must never take its id.
    # Its first line is named before it is fed, but the run's own id is already held
    # by then: claude's is named before the CLI starts, and codex's rollout is claimed
    # and read before a thread it spawned is owned (``_SessionOwner.owns``).
    @override
    def set_cli_session_id(self, cli_session_id: str) -> None:
        if cli_session_id not in self._children:
            self._root.set_cli_session_id(cli_session_id)

    @override
    def emit(self, adapter_name: str, event: Event) -> None:
        self._path_sink(event.path).emit(adapter_name, event)

    @override
    def restart(self, path: Path) -> None:
        self._path_sink(path).restart(path)

    @override
    def pass_over(self, path: Path) -> None:
        self._path_sink(path).pass_over(path)

    @override
    def emit_slash_command(self, command: SlashCommand, at: datetime) -> None:
        self._root.emit_slash_command(command, at)

    @override
    def emit_worktree_diff(self, records: Sequence[FileEditResult]) -> None:
        self._root.emit_worktree_diff(records)

    @override
    def emit_run_record(self, event: Event) -> None:
        self._root.emit_run_record(event)

    # The run's observers watch its own session: the worktree diff is the run's, and
    # the lines it typed are stamped on its own transcript, not a sub-agent's.
    @override
    def feed(
        self,
        adapter: Adapter,
        path: Path,
        raw: bytes,
        *,
        restart: bool = False,
        on_record: Callable[[TraxRecord], None] | None = None,
        upload: bool = True,
        annotate: Callable[[TraxRecord], TraxRecord] | None = None,
    ) -> list[str]:
        if path not in self._subagents:
            self._subagents[path] = adapter.subagent(path)
        subagent = self._subagents[path]
        if subagent is None:
            return self._root.feed(
                adapter,
                path,
                raw,
                restart=restart,
                on_record=on_record,
                upload=upload,
                annotate=annotate,
            )
        child = self._children.get(subagent.child)
        if child is None:
            child = self._open_child(subagent)
            self._children[subagent.child] = child
            self._spawned[subagent.child] = subagent
        return child.feed(adapter, path, raw, restart=restart, upload=upload)

    @override
    def flush(self) -> None:
        self._root.flush()
        for child in self._children.values():
            child.flush()
        self._link_open()

    @override
    def sync(self) -> None:
        self._root.sync()
        for child in self._children.values():
            child.sync()
        self._link_open()

    @override
    def pending(self) -> list[tuple[Path, RecordBody]]:
        return self._root.pending()

    @override
    def positions(self) -> dict[Path, int]:
        return self._root.positions()

    @override
    def keep_session_open(self) -> None:
        self._root.keep_session_open()
        for child in self._children.values():
            child.keep_session_open()

    # The children close first and the run's session after them. Closing flushes what
    # a sink holds, which opens its session if it had none, so the links come last.
    @override
    def close(self) -> None:
        for child in self._children.values():
            child.close()
        self._root.close()
        self._link_open(last=True)

    def _path_sink(self, path: Path) -> Sink:
        """Return the sink ``path``'s records go to: a sub-agent's, else the run's."""
        subagent = self._subagents.get(path)
        return self._root if subagent is None else self._children[subagent.child]

    # A refused link waits for close, then is logged: no logging is set up while the
    # CLI draws on the terminal, so a warning there lands on its screen, once a flush.
    def _link_open(self, *, last: bool = False) -> None:
        """Link each sub-agent whose session and parent's session are both open."""
        for child_id, subagent in self._spawned.items():
            if child_id in self._linked or (child_id in self._refused and not last):
                continue
            parent = self._children.get(subagent.parent, self._root).session_id
            child = self._children[child_id].session_id
            if child is None or parent is None:
                continue
            try:
                self._link(child, parent, subagent.note)
            except Exception:
                self._refused.add(child_id)
                if last:
                    _logger.warning(
                        "knowop run: could not link sub-agent %s to its parent",
                        child_id,
                        exc_info=True,
                    )
                continue
            self._linked.add(child_id)
