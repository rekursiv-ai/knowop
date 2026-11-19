"""One pass over the account's watched pull requests: read, written, told.

Each pass reads every open pull request a session watches (a closed one every
:data:`CLOSED_PASSES` passes, since it may be reopened), through ``gh`` on this host.
It writes what GitHub says to the row with a compare-and-set, and the news that write
named in ``told`` goes to each watching session's agent as a line, as a person's line
goes (``POST /api/sessions/ID/inputs``). A session with no agent running is told
nothing: the line is dropped, as a message to a stopped agent is. One with none for
:data:`AWAY_DAYS` lets go of what it watches, as a subscription with no subscriber
expires; the row's ``away`` keeps since when, so a serve that restarts or moves to
another host times it on. A pull request this host's ``gh`` cannot read is news too,
so the agents tell the person.

Only the write that names news in ``told`` tells it, so a second watch, on this host
or another, that read the same state tells nothing twice. A watch that stopped leaves
the rows as they were last read; the next one to run reads them again, and tells
whatever is news then.

Auto-merge is GitHub's own: a row's box, in Chat or on the Pull requests page, or the
person's ``automerge`` preference for one never asked about, turns it on (``gh pr
merge --auto --squash``) or off once per ask, as that person in ``gh``, and the row
says why GitHub refused. GitHub then merges once the required checks pass and the
reviews allow; a draft's ask waits until it is marked ready.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, Final, Protocol, TypedDict
from uuid import UUID

import logging
import time

from trackinizer.client.errors import ClientError
from trackinizer.wire.wire_session_relay import NewInput
from treekle import from_plain

from knowop.watch_pr.github import (
    GhError,
    read_login,
    read_pull_request,
    run_gh,
    set_auto_merge,
)
from knowop.watch_pr.news import Audience, News, news, unreadable
from knowop.watch_pr.rows import (
    Profile,
    PullRequestRows,
    RowsClient,
    Stored,
    Watcher,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from trackinizer.wire.wire_session_relay import AgentProcess

    from knowop.watch_pr.github import GhRunner, PullRequest


__all__ = [
    "AWAY_DAYS",
    "AWAY_SEC",
    "CLOSED_PASSES",
    "HEARTBEAT_SEC",
    "POLL_SEC",
    "Watch",
    "WatchClient",
]

_logger = logging.getLogger(__name__)

POLL_SEC: Final = 60.0
"""Seconds between two passes, as Codex and Claude's apps poll."""

CLOSED_PASSES: Final = 30
"""Passes between two reads of a closed pull request, which may be reopened."""

HEARTBEAT_SEC: Final = 600.0
"""Seconds after which a row is written again though nothing changed, so ``read_at``
says a watch still reads it."""

AWAY_DAYS: Final = 31
"""Days a watching session keeps its watches with no agent running, as long as Pub/Sub
keeps a subscription no subscriber touched (31 days by default).

One window for every reason an agent is not running, as the relay tells them apart too
little to time them apart: it forgets an ended process once its key is gone, and reads
that session as one with none; it keeps processes in memory, so knows none after the
server restarts; and one only disconnected (a host asleep) may say ``Hello`` again as
the same process (``trackinizer/server/session_relay.py``)."""

AWAY_SEC: Final = AWAY_DAYS * 24 * 3600.0
"""Seconds of :data:`AWAY_DAYS`."""

_STATUS: Final = {"OPEN": "active", "MERGED": "complete", "CLOSED": "abandoned"}
"""A pull request row's status for each state GitHub gives it."""


class WatchClient(RowsClient, Protocol):
    """What a watch asks of the server: the rows', and a session's agent."""

    def send_input(self, session_id: UUID, new: NewInput) -> UUID:
        """Send a session's root agent a line; return its id."""
        ...

    def read_process(self, session_id: UUID) -> AgentProcess | None:
        """Return the agent process serving a session, or the last that did."""
        ...


class Watch:
    """The account's watched pull requests, read through this host's ``gh``.

    Args:
      client: The server.
      host: This host's name, which the rows it writes name.
      gh: Runs ``gh``.
      clock: Monotonic seconds, which hold a failure for its workflow.
      wall: Seconds since the epoch, which ``read_at`` records.

    """

    def __init__(
        self,
        *,
        client: WatchClient,
        host: str,
        gh: GhRunner = run_gh,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        """Read nothing yet."""
        self._client = client
        self._rows = PullRequestRows(client=client, actor="knowop-watch-pr")
        self._host = host
        self._gh = gh
        self._clock = clock
        self._wall = wall
        self._passes = 0
        self._login: str | None = None
        self._first_failed: dict[str, tuple[str, float]] = {}
        self._numbers: dict[UUID, int] = {}
        self._said: set[str] = set()
        self._unreadable: set[str] = set()

    def sweep(self) -> int:
        """Read and write each watched pull request once; one that fails is skipped.

        Every :data:`CLOSED_PASSES` passes, each watching session's agent is asked
        after first, once a pass, and one with none for :data:`AWAY_DAYS` lets go.

        Returns:
          read: How many pull requests this pass read.

        """
        profile = self._rows.profile()
        me = profile["email"]
        autofix, automerge = _preferences(profile)
        rows = self._rows.listed(me, status="active")
        checked = self._passes % CLOSED_PASSES == 0
        if checked:
            rows += self._rows.listed(me, status="abandoned")
        self._passes += 1
        running: dict[UUID, bool | None] = {}
        read = 0
        for listed in rows:
            if not listed.watchers or not listed.url:
                continue
            stored = listed
            try:
                if checked:
                    stored = self._let_go(listed, running=running)
                if stored.watchers:
                    self.refresh(stored, autofix=autofix, automerge=automerge)
            except ClientError as error:
                self._say(f"could not write {listed.url}: {error}")
            read += bool(stored.watchers)
        return read

    def read(self, stored: Stored) -> None:
        """Read one watched pull request now, as the next pass would, so it shows at once."""
        autofix, automerge = _preferences(self._rows.profile())
        self.refresh(stored, autofix=autofix, automerge=automerge)

    def refresh(self, stored: Stored, *, autofix: bool, automerge: bool) -> None:
        """Read one pull request, write what changed, and tell its watchers the news.

        Args:
          stored: Its row as listed.
          autofix: The person's ``autofix`` preference.
          automerge: The person's ``automerge`` preference.

        """
        try:
            pull_request = read_pull_request(self._gh, stored.url)
        except GhError as failed:
            self._unread(stored, error=str(failed))
            return
        self._unreadable.discard(stored.url)
        enacted = self._auto_merge(pull_request, kept=stored.pr, default=automerge)
        written = self._rows.write_pr(
            stored,
            partial(
                self._changed,
                pull_request=pull_request,
                default=autofix,
                enacted=enacted,
            ),
        )
        if written is None:
            return
        pr, outcome = written
        self._rows.move(
            outcome.stored,
            status=_STATUS.get(pull_request.state, "active"),
        )
        if not outcome.stored.watchers:
            return
        # The fixer is one of the watchers the write landed on (``_fixer``), so its
        # name is among theirs.
        fixer = from_plain(pr["fixer"], str)
        names = {
            str(each.session): self._name(each) for each in outcome.stored.watchers
        }
        for each in outcome.news:
            for session, name in names.items():
                audience = Audience(
                    to=name,
                    fixer=names[fixer],
                    watchers=tuple(names.values()),
                )
                self._send(UUID(session), each.text(pull_request, audience))

    def _changed(
        self,
        stored: Stored,
        *,
        pull_request: PullRequest,
        default: bool,
        enacted: dict[str, object],
    ) -> tuple[dict[str, object], _Outcome]:
        """Return the row's ``app.knowop.pr`` as this read leaves it, and its news."""
        kept = stored.pr
        new = dict(kept) | pull_request.app() | {"error": None} | enacted
        # ``gh`` reads with the person's own login, so a pull request whose author is
        # that login is theirs: the Pull requests page lists it under theirs.
        if (login := self._gh_login()) is not None:
            new["authored"] = pull_request.author == login
        switch = from_plain(kept.get("autofix"), bool, default=None)
        # Another's pull request, in a repository the person may not push to, is fixed
        # only when its switch asks.
        authored = new.get("authored") is True
        told, found = news(
            pull_request,
            kept=kept,
            told=from_plain(kept.get("told"), list[str], default=list[str]()),
            autofix=default and authored if switch is None else switch,
            first_seen=self._first_seen(pull_request),
            now=self._clock(),
        )
        new["told"] = told
        new["fixer"] = _fixer(kept, stored=stored)
        return self._stamped(kept, new), _Outcome(stored=stored, news=found)

    # Told where the row had no error, so two serves tell it once; and once a serve
    # until it reads the pull request again, so a serve whose gh fails, beside one
    # whose gh reads, does not tell it each pass as the other clears the error.
    def _unread(self, stored: Stored, *, error: str) -> None:
        """Write that this host's ``gh`` cannot read the PR, and tell its watchers."""
        self._say(f"could not read {stored.url}: {error}")
        written = self._rows.write_pr(stored, partial(self._errored, error=error))
        if written is None or stored.url in self._unreadable:
            return
        landed, newly = written[1]
        if not newly:
            return
        self._unreadable.add(stored.url)
        for each in landed.watchers:
            self._send(
                each.session,
                unreadable(stored.url, host=self._host, error=error),
            )

    def _errored(
        self,
        stored: Stored,
        *,
        error: str,
    ) -> tuple[dict[str, object], tuple[Stored, bool]]:
        """Return the row's ``app.knowop.pr`` with ``error``, and whether it is new."""
        kept = stored.pr
        return self._stamped(kept, kept | {"error": error}), (
            stored,
            kept.get("error") is None,
        )

    def _stamped(
        self,
        kept: Mapping[str, object],
        new: dict[str, object],
    ) -> dict[str, object]:
        """Return ``new`` stamped as read here, if it changed or the last read is old."""
        written = datetime.fromtimestamp(self._wall(), UTC)
        stale = _read_at(kept) < written - timedelta(seconds=HEARTBEAT_SEC)
        if new != kept or stale:
            return new | {"read_at": written.isoformat(), "read_by": self._host}
        return new

    # The first pass to find a watcher's agent not running marks when, on the row
    # (``away``), so the time is kept across serves, hosts and restarts.
    def _let_go(self, stored: Stored, *, running: dict[UUID, bool | None]) -> Stored:
        """Let each watcher with no agent for :data:`AWAY_SEC` go of the PR."""
        marked = from_plain(stored.pr.get("away"), dict[str, str], default={})
        now = datetime.fromtimestamp(self._wall(), UTC)
        away: dict[str, str] = {}
        staying: list[Watcher] = []
        for each in stored.watchers:
            if each.session not in running:
                running[each.session] = self._running(each.session)
            key, runs, gone = str(each.session), running[each.session], False
            if runs is True:
                mark = None
            elif runs is None:
                mark = marked.get(key)
            else:
                mark = marked.get(key, now.isoformat())
                gone = now - datetime.fromisoformat(mark) >= timedelta(seconds=AWAY_SEC)
            if not (gone and self._dismissed(stored, each.session)):
                staying.append(each)
                if mark is not None:
                    away[key] = mark
        # The pass's own write next finds the row changed and reads it again.
        if away != marked:
            _ = self._rows.write_pr(stored, partial(_marked, away=away))
        return replace(stored, watchers=tuple(staying))

    # Its agent is gone, so a line saying so would be dropped: the log says it.
    def _dismissed(self, stored: Stored, session: UUID) -> bool:
        """Let ``session`` go of the PR, as Chat's x does; return whether it did."""
        try:
            self._rows.dismiss(stored.row, session=session)
        except ClientError as error:
            self._say(f"could not let session {session} go: {error}")
            return False
        _logger.info(
            "knowop watch-pr: session %s, with no agent for %d days, let go of %s",
            session,
            AWAY_DAYS,
            stored.url,
        )
        return True

    def _running(self, session: UUID) -> bool | None:
        """Return whether ``session`` runs an agent now; ``None`` when unread."""
        try:
            process = self._client.read_process(session)
        except ClientError as error:
            self._say(f"could not read session {session}'s agent: {error}")
            return None
        return process is not None and process.state != "detached"

    def _send(self, session: UUID, text: str) -> None:
        """Send ``text`` to the session's agent; one with none running is told nothing."""
        line = NewInput(
            kind="UserMessage",
            payload={"py/object": "UserMessage", "content": text},
        )
        try:
            _ = self._client.send_input(session, line)
        except ClientError as error:
            self._say(f"told session {session} nothing: {error}")

    def _name(self, watcher: Watcher) -> str:
        """Return the watcher's agent as ``LABEL#N``, its session's number read once."""
        if watcher.session not in self._numbers:
            self._numbers[watcher.session] = from_plain(
                self._client.get(f"/api/inquiries/{watcher.session}"),
                _Numbered,
            )["seq"]
        return f"{watcher.label}#{self._numbers[watcher.session]}"

    def _gh_login(self) -> str | None:
        """Return the login ``gh`` reads with, asked once; ``None`` when it says none."""
        if self._login is None:
            try:
                self._login = read_login(self._gh)
            except GhError as error:
                self._say(f"gh names no login: {error}")
                return None
        return self._login

    def _first_seen(self, pull_request: PullRequest) -> float:
        """Return when a failure of the head was first seen, this read if none was."""
        now = self._clock()
        head, seen = self._first_failed.get(pull_request.url, ("", now))
        first = seen if head == pull_request.head_sha else now
        if pull_request.failed:
            self._first_failed[pull_request.url] = (pull_request.head_sha, first)
        return first

    # Run before the row is written, and not in ``_changed``, which a lost write runs
    # again: ``gh`` is called once a read. An ask is done once, so GitHub's own state
    # rules after it, a change on github.com too; a draft's ask waits, as GitHub turns
    # no draft's auto-merge on.
    def _auto_merge(
        self,
        pull_request: PullRequest,
        *,
        kept: Mapping[str, object],
        default: bool,
    ) -> dict[str, object]:
        """Do the Auto-merge ask due on a pull request; return what the row keeps of it."""
        done = from_plain(kept.get("automerge_done"), str, default="")
        ask = from_plain(kept.get("automerge_ask"), _MergeAsk, default=None)
        stamp, on = (ask["at"], ask["on"]) if ask else ("default", default)
        if (
            stamp == done
            or not (ask or (default and not done))
            or pull_request.state != "OPEN"
            or (on and pull_request.draft)
        ):
            return {}
        if on == pull_request.auto_merge:
            return {"automerge_done": stamp, "automerge_error": None}
        error = set_auto_merge(self._gh, url=pull_request.url, on=on)
        return {
            "automerge_done": stamp,
            "automerge_error": error,
            "auto_merge": on if error is None else pull_request.auto_merge,
        }

    def _say(self, message: str) -> None:
        """Log ``message`` once a serve, so a lasting failure is one line, not one a pass."""
        if message not in self._said:
            self._said.add(message)
            _logger.warning("knowop watch-pr: %s", message)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Outcome:
    """The row a write landed on, and the news that write named in ``told``."""

    stored: Stored
    news: list[News]


class _MergeAsk(TypedDict):
    """What a row's Auto-merge box writes: on or off, and when, which names the ask."""

    on: bool
    at: str


class _Numbered(TypedDict):
    seq: int


# The first session to watch it fixes it: the one its row names, until that one lets
# it go; then the first that still watches it. A person or an agent names another with
# ``knowop watch-pr fix``.
def _fixer(kept: Mapping[str, object], *, stored: Stored) -> str | None:
    """Return the session that fixes a pull request; ``None`` while none watches it."""
    named = from_plain(kept.get("fixer"), str, default="")
    watching = [str(each.session) for each in stored.watchers]
    return named if named in watching else next(iter(watching), None)


def _marked(stored: Stored, *, away: dict[str, str]) -> tuple[dict[str, object], None]:
    """Return the row's ``app.knowop.pr`` with ``away`` as its marks."""
    return stored.pr | {"away": away}, None


def _preferences(profile: Profile) -> tuple[bool, bool]:
    """Return the person's ``autofix`` preference, on unless set, and ``automerge``."""
    preferences = profile["preferences"]
    return preferences.get("autofix", True), preferences.get("automerge") is True


def _read_at(kept: Mapping[str, object]) -> datetime:
    """Return when a watch last wrote the row; long ago for one never written."""
    at = from_plain(kept.get("read_at"), str, default="")
    return datetime.fromisoformat(at) if at else datetime.min.replace(tzinfo=UTC)
