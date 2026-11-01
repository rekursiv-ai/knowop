"""Tests for the sub-agent sessions a run records beside its own."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, override
from uuid import UUID, uuid4

from trackinizer.runs.adapters.custom_types import Subagent

import pytest

from knowop.agent.sink import Sink
from knowop.agent.subagents import SubagentSinks


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from datetime import datetime

    from trackinizer.lib.agent.sessions.tail import Tail
    from trackinizer.runs.adapters.custom_types import Adapter
    from trackinizer.sagent.session.types import FileEditResult
    from trackinizer.types.streams import TraxRecord
    from trackinizer.wire.wire_session_ir import RecordBody

    from knowop.agent.custom_types import Event
    from knowop.agent.slash import SlashCommand


_RUN: Path = Path("/projects/-w/s1.jsonl")
_KID: Path = Path("/projects/-w/s1/subagents/agent-a1.jsonl")
_GRANDKID: Path = Path("/projects/-w/s1/subagents/agent-a2.jsonl")


class _Recorded(Sink):
    """A sink that records what reaches it; its session opens on its first flush."""

    def __init__(
        self,
        *,
        opened: UUID | None = None,
        log: list[str] | None = None,
    ) -> None:
        self.opened = opened
        self.fed: list[tuple[Path, bytes, bool, bool]] = []
        self.observed: list[bool] = []
        self.cli_ids: list[str] = []
        self.restarted: list[Path] = []
        self.passed: list[Path] = []
        self.commands = 0
        self.flushed = 0
        self.kept = 0
        self.failed: list[str] = []
        self.closed = 0
        self._log = log if log is not None else []

    @property
    @override
    def session_id(self) -> UUID | None:
        return self.opened

    @override
    def open(self) -> str | None:
        return None

    @override
    def set_cli_session_id(self, cli_session_id: str) -> None:
        self.cli_ids.append(cli_session_id)

    @override
    def emit(self, adapter_name: str, event: Event) -> None:
        del adapter_name, event

    @override
    def restart(self, path: Path) -> None:
        self.restarted.append(path)

    @override
    def pass_over(self, path: Path) -> None:
        self.passed.append(path)

    @override
    def emit_slash_command(self, command: SlashCommand, at: datetime) -> None:
        del command, at
        self.commands += 1

    @override
    def emit_worktree_diff(self, records: Sequence[FileEditResult]) -> None:
        del records

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
        del adapter
        self.fed.append((path, raw, restart, upload))
        self.observed.append(on_record is not None or annotate is not None)
        return ["UserMessage"]

    @override
    def flush(self) -> None:
        self.flushed += 1
        if self.opened is None:
            self.opened = uuid4()

    @override
    def pending(self) -> list[tuple[Path, RecordBody]]:
        return []

    @override
    def keep_session_open(self) -> None:
        self.kept += 1

    @override
    def fail(self, reason: str) -> None:
        self.failed.append(reason)

    # Closing flushes, so it opens the session as a flush does.
    @override
    def close(self) -> None:
        self.flush()
        self.closed += 1
        self._log.append(f"close {self.cli_ids[0]}")


class _Agents:
    """An adapter that names its sub-agents by path; it reads nothing."""

    name: str = "fake"
    cli_binary: str = "fake"
    whole_file: bool = False
    parent_session_env: frozenset[str] = frozenset[str]()
    session_switch_commands: frozenset[str] = frozenset[str]()

    def __init__(self, subagents: dict[Path, Subagent]) -> None:
        self.subagents = subagents
        self.asked: list[Path] = []

    def resumed_session(self, cli_args: tuple[str, ...]) -> str | None:
        del cli_args
        return None

    def session_dirs(self) -> Iterable[Path]:
        return ()

    def matches_session_file(self, path: Path) -> bool:
        return path.suffix == ".jsonl"

    def session_scope(self) -> Path | None:
        return None

    def session_id_from_path(self, path: Path) -> str | None:
        return path.stem

    def subagent(self, path: Path) -> Subagent | None:
        self.asked.append(path)
        return self.subagents.get(path)

    def reader(self) -> Tail[TraxRecord]:
        raise NotImplementedError


class _Tree:
    """A :class:`SubagentSinks` over recorded sinks, with every link it made."""

    def __init__(self, log: list[str] | None = None) -> None:
        self.log = log if log is not None else []
        self.run = _Recorded(opened=uuid4(), log=self.log)
        self.children: dict[str, _Recorded] = {}
        self.links: list[tuple[UUID, UUID, str]] = []
        self.refusals = 0
        self.sinks = SubagentSinks(self.run, open_child=self._open, link=self._link)
        self.sinks.set_cli_session_id("s1")

    def _open(self, subagent: Subagent) -> Sink:
        child = _Recorded(log=self.log)
        child.set_cli_session_id(subagent.child)
        self.children[subagent.child] = child
        return child

    def _link(self, child: UUID, parent: UUID, note: str) -> None:
        if self.refusals:
            self.refusals -= 1
            raise RuntimeError("the server is down")
        self.links.append((child, parent, note))
        self.log.append("link")


def _ignore(record: TraxRecord) -> None:
    """Observe a record, as the run's own observers do."""
    del record


def _agents() -> _Agents:
    return _Agents(
        {
            _KID: Subagent(child="a1", parent="s1", note="Explore: read it"),
            _GRANDKID: Subagent(child="a2", parent="a1", note="dig"),
        },
    )


def test_the_runs_own_lines_reach_its_own_sink_with_its_observers() -> None:
    tree = _Tree()

    kinds = tree.sinks.feed(_agents(), _RUN, b"hi\n", on_record=_ignore)

    assert kinds == ["UserMessage"]
    assert tree.run.fed == [(_RUN, b"hi\n", False, True)]
    assert tree.run.observed == [True]
    assert tree.children == {}


def test_a_subagents_lines_reach_one_sink_opened_under_its_own_id() -> None:
    """Two lines, one session: a re-read or a resume lands on the same one."""
    tree = _Tree()
    agents = _agents()

    _ = tree.sinks.feed(agents, _KID, b"one\n", on_record=_ignore)
    _ = tree.sinks.feed(agents, _KID, b"two\n", restart=True, upload=False)

    kid = tree.children["a1"]
    assert list(tree.children) == ["a1"]
    assert kid.cli_ids == ["a1"]
    assert kid.fed == [(_KID, b"one\n", False, True), (_KID, b"two\n", True, False)]
    # The run's observers watch its own session: a child's edits and lines are its own.
    assert kid.observed == [False, False]
    assert tree.run.fed == []
    assert agents.asked == [_KID]


def test_a_child_is_linked_to_its_parent_once_both_sessions_are_open() -> None:
    tree = _Tree()
    _ = tree.sinks.feed(_agents(), _KID, b"one\n")
    assert tree.links == []

    tree.sinks.flush()
    tree.sinks.flush()

    kid = tree.children["a1"].session_id
    assert kid is not None
    assert tree.run.session_id is not None
    assert tree.links == [(kid, tree.run.session_id, "Explore: read it")]


def test_a_nested_child_is_linked_to_the_child_that_spawned_it() -> None:
    tree = _Tree()
    agents = _agents()
    _ = tree.sinks.feed(agents, _KID, b"one\n")
    _ = tree.sinks.feed(agents, _GRANDKID, b"two\n")

    tree.sinks.flush()

    kid, grandkid = tree.children["a1"].session_id, tree.children["a2"].session_id
    assert kid is not None
    assert grandkid is not None
    assert tree.run.session_id is not None
    assert tree.links == [
        (kid, tree.run.session_id, "Explore: read it"),
        (grandkid, kid, "dig"),
    ]


def test_a_refused_link_is_tried_once_more_at_close_and_says_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The CLI draws on the terminal until close: a refusal there is not retried."""
    tree = _Tree()
    tree.refusals = 1
    _ = tree.sinks.feed(_agents(), _KID, b"one\n")

    tree.sinks.flush()
    tree.sinks.sync()
    assert (tree.links, tree.refusals) == ([], 0)
    tree.sinks.close()

    assert len(tree.links) == 1
    assert caplog.records == []


def test_a_link_refused_at_close_too_is_logged_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tree = _Tree()
    tree.refusals = 2
    _ = tree.sinks.feed(_agents(), _KID, b"one\n")
    tree.sinks.flush()

    tree.sinks.close()

    assert tree.links == []
    assert [record.getMessage() for record in caplog.records] == [
        "knowop run: could not link sub-agent a1 to its parent",
    ]


def test_closing_closes_the_children_then_the_run_then_links_them() -> None:
    """A session first opened by its closing flush is linked all the same."""
    log: list[str] = []
    tree = _Tree(log)
    tree.run.opened = None
    _ = tree.sinks.feed(_agents(), _KID, b"one\n")

    tree.sinks.close()

    assert log == ["close a1", "close s1", "link"]


def test_restarts_and_passed_over_lines_reach_the_sink_their_path_feeds() -> None:
    tree = _Tree()
    _ = tree.sinks.feed(_agents(), _KID, b"one\n")

    tree.sinks.restart(_KID)
    tree.sinks.pass_over(_KID)
    tree.sinks.restart(_RUN)

    assert tree.children["a1"].restarted == [_KID]
    assert tree.children["a1"].passed == [_KID]
    assert tree.run.restarted == [_RUN]


def test_the_runs_own_records_and_choices_stay_with_its_session() -> None:
    tree = _Tree()
    _ = tree.sinks.feed(_agents(), _KID, b"one\n")

    tree.sinks.keep_session_open()

    assert tree.sinks.session_id == tree.run.session_id
    assert tree.run.cli_ids == ["s1"]
    assert (tree.run.kept, tree.children["a1"].kept) == (1, 1)


def test_why_the_run_failed_is_said_by_its_own_session_alone() -> None:
    """Chat waits on the run's session; a sub-agent's never was the run's to start."""
    tree = _Tree()
    _ = tree.sinks.feed(_agents(), _KID, b"one\n")

    tree.sinks.fail("exited with status 1")

    assert (tree.run.failed, tree.children["a1"].failed) == (
        ["exited with status 1"],
        [],
    )


@pytest.mark.parametrize("seen", [None, "a1"], ids=["first", "child"])
def test_the_run_keeps_the_first_id_it_is_told_and_never_a_childs(
    seen: str | None,
) -> None:
    tree = _Tree()
    if seen is not None:
        _ = tree.sinks.feed(_agents(), _KID, b"one\n")

    tree.sinks.set_cli_session_id("a1")

    assert tree.run.cli_ids == (["s1"] if seen is not None else ["s1", "a1"])


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
