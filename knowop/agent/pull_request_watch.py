"""Watch a run session's pull requests, and wake the agent on their failures.

The watch learns of a pull request four ways, as Claude Code's app binds one:

- ``gh pr create`` printed its link (:class:`RecordNotes` reads the session's records);
- the branch the run started on has an open pull request the person opened;
- a branch the session pushed has an open one, whoever opened it
  (:class:`BranchLookup`);
- the agent called ``watch_pull_request``, one of the tools the run serves it
  (:meth:`PullRequestWatch.tools`).

A person lets one go from Chat with its ×, and the agent with ``unwatch_pull_request``:
both put the session's dismissal on its row (:mod:`.pull_request_rows`). While it
watches anything, the watch reads the session's rows at every pass, lets a dismissed
one go, and none of the ways above brings it back but ``watch_pull_request``. A tool
call waits for the pass that does what it asked, and answers what that pass saw.

Every :data:`POLL_SEC`, and at once after a call that acted, the watch reads each
pull request it watches and writes its row when what GitHub says changed. A closed
one is read every :data:`CLOSED_PASSES` passes instead, as Claude Code's app reads
one, and watched again once it reopens; a merged one is read no more.

With Auto-fix on, the failures :func:`due_failures` names become one line from
``knowop-autofix``, which the inbound loop takes (:meth:`PullRequestWatch.take`) and
types as it types a Chat line: the server cannot put a line into a chat's run, and
this run already sits in the agent's checkout, under the agent's own credentials.
Several sessions may watch one pull request, and only one is woken for it: the first
to watch it, its ``fixer``, until it lets the pull request go.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, Final, Literal
from uuid import UUID

import logging
import threading
import time

from trackinizer.wire.wire_sessions import InboundDrainItem
from treekle import from_plain

from knowop.agent.pull_request_branches import BranchLookup
from knowop.agent.pull_request_notes import RecordNotes
from knowop.agent.pull_request_rows import PullRequestRows
from knowop.agent.pull_requests import (
    AUTOFIX_SOURCE,
    GhError,
    due_failures,
    links,
    read_pull_request,
    run_gh,
    wake_text,
)
from knowop.agent.tool_server import AgentTool, ToolError
from knowop.agent.worktree_diff import run_git


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from trackinizer.types.streams import TraxRecord

    from knowop.agent.pull_request_rows import PullRequestClient
    from knowop.agent.pull_requests import GhRunner, PullRequest
    from knowop.agent.worktree_diff import Runner


__all__ = ["ASK_SEC", "CLOSED_PASSES", "POLL_SEC", "Ask", "PullRequestWatch"]

_logger = logging.getLogger(__name__)

POLL_SEC: Final = 60.0
"""Seconds between two reads of a pull request, as Codex and Claude's apps poll."""

CLOSED_PASSES: Final = 30
"""Passes between two reads of a closed pull request, which may be reopened."""

ASK_SEC: Final = 120.0
"""The longest a tool call waits for the pass that answers it: a pass reads each pull
request it watches, each read up to ``GH_SEC``."""

type Ask = Literal["watch", "unwatch"]
"""What a tool call asks of the watch."""

_STATUS: Final = {"OPEN": "active", "MERGED": "complete", "CLOSED": "abandoned"}
"""A pull request row's status for each state GitHub gives it."""

_LINK_SCHEMA: Final = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "The pull request's link: https://github.com/OWNER/REPO/pull/N",
        },
    },
    "required": ["url"],
    "additionalProperties": False,
}

_WATCH_TOOL: Final = (
    "Watch a GitHub pull request in this chat. Chat shows it over the session, and "
    "while its Auto-fix is on the run sends you a line from knowop-autofix when its "
    "checks fail. The run already watches a pull request this session opens with gh pr "
    "create, or one open on a branch it pushes: call this for any other the person "
    "asks you to watch, or to bring back one that was let go. Answers what the run now "
    "sees of it."
)

_UNWATCH_TOOL: Final = (
    "Stop watching a GitHub pull request in this chat, as the person's x in Chat does: "
    "Chat no longer shows it, no Auto-fix line comes for it, and neither a push nor gh "
    "pr create brings it back; only watch_pull_request does."
)


class PullRequestWatch:
    """The pull requests of one run's session: found, stored, read, and woken for.

    :meth:`note` runs on the drain thread, :meth:`take` on the inbound thread, each
    tool's call on the tool server's, and :meth:`run` on a thread of its own; the
    notes, the wakes and the calls go under locks, and the rest is :meth:`run`'s alone.

    Args:
      client: The server the session syncs to.
      session: The session's id, once it is open.
      actor: Who the row writes are recorded as.
      cwd: The checkout ``gh`` and ``git`` run in.
      resumed: Whether the run continues a session, whose earlier runs may have stored
        pull requests to go on watching; a new session has none to look for.
      gh: Runs ``gh``.
      git: Runs ``git``.
      clock: Monotonic seconds.
      poll_sec: Seconds between two reads of each pull request.

    """

    def __init__(
        self,
        *,
        client: PullRequestClient,
        session: Callable[[], UUID | None],
        actor: str,
        cwd: Path,
        resumed: bool = True,
        gh: GhRunner = run_gh,
        git: Runner = run_git,
        clock: Callable[[], float] = time.monotonic,
        poll_sec: float = POLL_SEC,
    ) -> None:
        """Watch nothing yet."""
        self._rows = PullRequestRows(client=client, actor=actor)
        self._session = session
        self._cwd = cwd
        self._gh = gh
        self._clock = clock
        self._poll_sec = poll_sec
        self._notes = RecordNotes()
        self._branches = BranchLookup(cwd=cwd, gh=gh, git=git, say=self._say)
        self._lock = threading.Lock()
        self._poke = threading.Event()
        self._closed = threading.Event()
        self._wakes: deque[str] = deque()
        self._asked: list[_Asked] = []
        # An insertion-ordered set: the links are read oldest first.
        self._watched: dict[str, None] = {}
        # Each closed one, and the passes until it is read again.
        self._closed_links: dict[str, int] = {}
        self._dismissed: set[str] = set()
        self._first_failed: dict[str, tuple[str, float]] = {}
        self._started = False
        self._seeded = not resumed
        self._said: set[str] = set()

    def note(self, record: TraxRecord) -> None:
        """Note a record the session just emitted; one that acted wakes the watch.

        Args:
          record: A record the session just emitted.

        """
        if self._notes.note(record):
            self._poke.set()

    def take(self) -> list[InboundDrainItem]:
        """Return the Auto-fix lines due, each taken once.

        Returns:
          lines: Lines from ``knowop-autofix``, oldest first, for the inbound loop to
            type as it types a Chat line.

        """
        with self._lock:
            texts = list(self._wakes)
            self._wakes.clear()
        return [
            InboundDrainItem(text=text, kind="line", source=AUTOFIX_SOURCE)
            for text in texts
        ]

    def tools(self) -> tuple[AgentTool, ...]:
        """Return the tools the agent watches pull requests with, and lets them go.

        Returns:
          tools: ``watch_pull_request`` and ``unwatch_pull_request``, each taking a
            pull request's ``url`` and answering once a pass has done what it asked.

        """
        return (
            AgentTool(
                name="watch_pull_request",
                description=_WATCH_TOOL,
                schema=_LINK_SCHEMA,
                call=partial(self._called, "watch"),
            ),
            AgentTool(
                name="unwatch_pull_request",
                description=_UNWATCH_TOOL,
                schema=_LINK_SCHEMA,
                call=partial(self._called, "unwatch"),
            ),
        )

    def start(self) -> None:
        """Run :meth:`run` on a thread of its own, which the process's exit ends."""
        threading.Thread(
            name="pull request watch",
            target=self.run,
            daemon=True,
        ).start()

    def run(self) -> None:
        """Read and store the pull requests until :meth:`close`; a failure costs a pass.

        Each pass follows a wait, so a run that ends inside its first
        :data:`POLL_SEC` and noted nothing never asks the server, ``git`` or ``gh``.
        """
        while not self._closed.is_set():
            _ = self._poke.wait(timeout=self._poll_sec)
            self._poke.clear()
            if self._closed.is_set():
                return
            try:
                self._sweep()
            except Exception:
                _logger.warning("knowop run: pull request watch failed", exc_info=True)

    def close(self) -> None:
        """Stop :meth:`run` at its next wait."""
        self._closed.set()
        self._poke.set()

    def _called(self, ask: Ask, arguments: Mapping[str, object]) -> str:
        """Answer a tool call once the pass that does what it asks has run."""
        found = links(from_plain(arguments.get("url"), str, default=""))
        if len(found) != 1:
            raise ToolError(
                "Pass url, one GitHub pull request's link: "
                "https://github.com/OWNER/REPO/pull/N",
            )
        asked = _Asked(ask=ask, url=found[0])
        with self._lock:
            self._asked.append(asked)
        self._poke.set()
        if not asked.done.wait(timeout=ASK_SEC):
            raise ToolError(
                f"The run is still at {found[0]}; Chat shows it once it is done.",
            )
        return asked.answer

    def _sweep(self) -> None:
        """Make one pass, and answer each tool call it took, whatever became of it."""
        with self._lock:
            asked, self._asked = self._asked, list[_Asked]()
        try:
            self._pass(asked)
        finally:
            for each in asked:
                each.end("The run could not do that; its log says why.")

    def _pass(self, asked: Sequence[_Asked]) -> None:
        """Find the session's pull requests, then read each and store what changed."""
        session = self._session()
        if session is None:
            return
        reopening = self._closed_due()
        if not self._find(session, reading=bool(asked or reopening)):
            return
        profile = self._rows.profile()
        for each in asked:
            if each.ask == "watch":
                self._restore(each.url, session=session)
            else:
                self._dismiss(each.url, session=session, me=profile["email"])
        default = profile["preferences"].get("autofix", True)
        read = [
            *self._watched,
            *(url for url in reopening if url in self._closed_links),
        ]
        seen = {
            url: self._refresh(
                url,
                session=session,
                me=profile["email"],
                default=default,
            )
            for url in read
        }
        for each in asked:
            each.end(
                seen.get(each.url, f"The run let {each.url} go.")
                if each.ask == "watch"
                else f"Let go of {each.url}: Chat no longer shows it, and no Auto-fix "
                "line comes for it. watch_pull_request brings it back.",
            )

    def _closed_due(self) -> list[str]:
        """Return the closed pull requests this pass reads, and count the rest down."""
        due: list[str] = []
        for url, left in self._closed_links.items():
            if left <= 1:
                due.append(url)
            self._closed_links[url] = CLOSED_PASSES if left <= 1 else left - 1
        return due

    # ``reading`` says whether the pass reads one it does not watch: a tool call's, or a
    # closed one's.
    def _find(self, session: UUID, *, reading: bool) -> bool:
        """Watch what was found since the last pass; say whether any is read."""
        noted = self._notes.take()
        if not self._started:
            self._started = True
            self._branches.note(anyone=False)
        if noted.pushed:
            self._branches.note(anyone=True)
        if not self._seeded or self._watched or reading:
            self._sync(session)
            self._seeded = True
        for url in (*noted.opened, *self._branches.found()):
            if url not in self._dismissed:
                self._watch(url)
        return bool(self._watched) or reading

    def _sync(self, session: UUID) -> None:
        """Read the session's rows: which were dismissed, and which to go on watching when the run continues a session."""
        dismissed: set[str] = set()
        for row in self._rows.session_rows(session):
            if row.dismissed:
                dismissed.add(row.url)
                self._unwatch(row.url)
            elif not self._seeded and row.status == "active":
                self._watch(row.url)
            elif not self._seeded and row.status == "abandoned":
                # Read on the next pass: it may have reopened while no run watched it.
                self._closed_links.setdefault(row.url, 1)
        self._dismissed = dismissed

    def _restore(self, url: str, *, session: UUID) -> None:
        """Watch ``url`` as the agent asked, a dismissal from this session undone."""
        row = self._rows.linked(url)
        if url in self._dismissed and row is not None:
            self._rows.dismiss(row, session=session, dismissed=False)
        self._dismissed.discard(url)
        self._watch(url)

    def _dismiss(self, url: str, *, session: UUID, me: str) -> None:
        """Let ``url`` go as the agent asked, and keep it from coming back on its own."""
        self._unwatch(url)
        if url in self._dismissed:
            return
        row = self._rows.linked(url)
        if row is None:
            row = self._rows.row(url, session=session, me=me, pull_request=None).row
        self._rows.dismiss(row, session=session, dismissed=True)
        self._dismissed.add(url)

    # The row's status follows GitHub, a closed pull request's back to ``active`` once
    # it is reopened.
    def _refresh(self, url: str, *, session: UUID, me: str, default: bool) -> str:
        """Read one pull request, store what changed, and queue a wake when one is due."""
        try:
            pull_request: PullRequest | None = read_pull_request(
                self._gh,
                self._cwd,
                ref=url,
            )
            error = None
        except GhError as failed:
            pull_request, error = None, str(failed)
            self._say(f"could not read {url}: {error}")
        stored = self._rows.row(url, session=session, me=me, pull_request=pull_request)
        written = self._rows.write_pr(
            stored,
            partial(
                self._changed,
                row=stored.row,
                pull_request=pull_request,
                error=error,
                session=session,
                default=default,
            ),
        )
        if written is None:
            self._unwatch(url)
            return f"{url} has no row any more; there is nothing left to watch."
        if pull_request is None:
            return f"gh could not read {url}: {error}. The run tries again each minute."
        pr, due = written
        status = _STATUS.get(pull_request.state, "active")
        self._rows.move(stored, status=status)
        match status:
            case "active":
                self._watch(url)
            case "abandoned":
                self._close(url)
            case _:
                self._unwatch(url)
        if due:
            with self._lock:
                self._wakes.append(wake_text(pull_request, due))
        switch = from_plain(pr.get("autofix"), bool, default=None)
        return _described(
            pull_request,
            autofix=default if switch is None else switch,
            fixing=pr.get("fixer") == str(session),
        )

    # Only the session that fixes it names failures in ``told``, and only it wakes.
    def _changed(
        self,
        kept: dict[str, object],
        *,
        row: UUID,
        pull_request: PullRequest | None,
        error: str | None,
        session: UUID,
        default: bool,
    ) -> tuple[dict[str, object], tuple[str, ...]]:
        """Return the row's ``app.knowop.pr`` as this read leaves it, and the failures to wake for."""
        new = dict(kept)
        due: tuple[str, ...] = ()
        if pull_request is None:
            new["error"] = error
        else:
            new |= pull_request.app() | {"error": None}
            switch = from_plain(kept.get("autofix"), bool, default=None)
            failing = (default if switch is None else switch) and bool(
                pull_request.failed,
            )
            fixer = self._fixer(kept, row=row, session=session, failing=failing)
            new["fixer"] = fixer
            if fixer == str(session):
                told = from_plain(kept.get("told"), list[str], default=[])
                due = self._due(pull_request, told=told) if failing else ()
                head = f"{pull_request.head_sha}:"
                new["told"] = [each for each in told if each.startswith(head)] + [
                    f"{head}{name}" for name in due
                ]
        if new != kept:
            new["read_at"] = datetime.now(UTC).isoformat()
        return new, due

    # The first session to watch it fixes it: the one its row names, until that one
    # lets it go. Whether it did is read only when a failure would wake it.
    def _fixer(
        self,
        kept: Mapping[str, object],
        *,
        row: UUID,
        session: UUID,
        failing: bool,
    ) -> str:
        """Return the session that fixes a pull request, this one if none does."""
        fixer = from_plain(kept.get("fixer"), str, default="")
        if not fixer or (
            failing
            and fixer != str(session)
            and self._rows.let_go(row, session=UUID(fixer))
        ):
            return str(session)
        return fixer

    def _due(
        self,
        pull_request: PullRequest,
        *,
        told: Sequence[str],
    ) -> tuple[str, ...]:
        """Return the failures to wake for, as :func:`due_failures` names them."""
        now = self._clock()
        head, seen = self._first_failed.get(pull_request.url, ("", now))
        first = seen if head == pull_request.head_sha else now
        if pull_request.failed:
            self._first_failed[pull_request.url] = (pull_request.head_sha, first)
        return due_failures(pull_request, told=told, first_seen=first, now=now)

    def _watch(self, url: str) -> None:
        """Read ``url`` at every pass."""
        self._closed_links.pop(url, None)
        self._watched.setdefault(url)

    def _close(self, url: str) -> None:
        """Read ``url``, which is closed, every :data:`CLOSED_PASSES` passes."""
        self._watched.pop(url, None)
        self._first_failed.pop(url, None)
        self._closed_links.setdefault(url, CLOSED_PASSES)

    def _unwatch(self, url: str) -> None:
        """Stop reading ``url``."""
        self._watched.pop(url, None)
        self._closed_links.pop(url, None)
        self._first_failed.pop(url, None)

    def _say(self, message: str) -> None:
        """Log ``message`` once a run, so a lasting failure is one line, not one a pass."""
        if message not in self._said:
            self._said.add(message)
            _logger.warning("knowop run: %s", message)


@dataclass(slots=True, kw_only=True)
class _Asked:
    """A tool call waiting on the next pass: what it asked, and once ended, the answer."""

    ask: Ask
    url: str
    answer: str = ""
    done: threading.Event = field(default_factory=threading.Event)

    def end(self, answer: str) -> None:
        """Answer the call, unless the pass answered it already."""
        if not self.done.is_set():
            self.answer = answer
            self.done.set()


def _described(pull_request: PullRequest, *, autofix: bool, fixing: bool) -> str:
    """Say what a read found of a pull request, as a tool call's answer."""
    name = f"{pull_request.repo}#{pull_request.number}"
    match pull_request.state:
        case "MERGED":
            return f"{name} is merged; there is nothing left to watch."
        case "CLOSED":
            minutes = round(CLOSED_PASSES * POLL_SEC / 60)
            return (
                f"{name} is closed; the run reads it every {minutes} minutes, and "
                "watches it again once it is reopened."
            )
    checks = [f"{pull_request.passed} of {pull_request.total} checks passed"]
    if pull_request.failed:
        checks.append(f"{len(pull_request.failed)} failed")
    if pull_request.pending:
        checks.append(f"{len(pull_request.pending)} running")
    if not autofix:
        return f"Watching {name}: {', '.join(checks)}; Auto-fix off."
    if not fixing:
        return (
            f"Watching {name}: {', '.join(checks)}; Auto-fix on, in the chat that "
            "watched it first: its agent gets the Auto-fix lines."
        )
    return f"Watching {name}: {', '.join(checks)}; Auto-fix on."
