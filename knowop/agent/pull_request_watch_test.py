"""Tests for the watch of a run's pull requests: finding, storing, reading, waking.

The server is an in-memory :class:`_Server` holding rows, edges with their labels and
the person's profile; ``gh`` is a :class:`_Gh` that prints what the test set for each
pull request, and ``git`` a :class:`_Git` on the branch the test set. Each test drives
:meth:`PullRequestWatch._sweep` by hand, as :meth:`run` would, but a test of a tool,
whose call waits on the watch's own thread.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict
from uuid import UUID, uuid4

import json

from trackinizer.client.errors import ClientError
from trackinizer.sagent.session.types import ShellCommandResult, ToolCall
from treekle import from_plain

import pytest

from knowop.agent.pull_request_rows import DISMISSED, LABEL
from knowop.agent.pull_request_watch import (
    CLOSED_PASSES,
    PullRequestWatch,
)
from knowop.agent.pull_requests import (
    AUTOFIX_SOURCE,
    HOLD_SEC,
    GhOutput,
)
from knowop.agent.tool_server import AgentTool, ToolError
from knowop.agent.worktree_diff import Output


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from trackinizer.wire.filters import Filter
    from treekle import PlainTree


_SESSION = UUID("00000000-0000-0000-0000-00000000000a")
_OTHER = UUID("00000000-0000-0000-0000-00000000000b")
_ME = "me@acme.test"
_URL = "https://github.com/acme/app/pull/7"
_SHA = "a" * 40
_BRANCH = "agent/fix"

type _Edge = tuple[UUID, UUID, str]


class _LabelOp(TypedDict):
    op: str
    value: str


@dataclass(slots=True, kw_only=True)
class _Server:
    """Rows, edges and a profile, answering as ``Client`` does."""

    rows: dict[UUID, dict[str, object]] = field(default_factory=dict)
    edges: dict[_Edge, list[str]] = field(default_factory=dict)
    autofix: bool | None = None
    down: bool = False
    racing: Callable[[], None] | None = None
    """What another writer does between the watch's read and its write, which is then
    refused with 409 once; ``None`` for no race."""

    def get(
        self,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
    ) -> PlainTree:
        del params
        if path == "/api/me/profile":
            if self.down:
                raise ClientError("down", status_code=503)
            preferences = {} if self.autofix is None else {"autofix": self.autofix}
            return {"email": _ME, "preferences": preferences}
        target = UUID(path.removeprefix("/api/web/get/"))
        # A row's view names the sessions it was produced by; a session's, its rows.
        produced = [
            {
                "id": str(row),
                "kind": "WebResult",
                "status": str(self.rows[row]["status"]),
                **({"labels": list(labels)} if labels else {}),
            }
            for (row, to, kind), labels in self.edges.items()
            if to == target and kind == "produced_by"
        ]
        producers = [
            {
                "id": str(to),
                "kind": "AgentSession",
                "status": "active",
                **({"labels": list(labels)} if labels else {}),
            }
            for (row, to, kind), labels in self.edges.items()
            if row == target and kind == "produced_by"
        ]
        return {
            "backlinks": {"produced_by": produced},
            "edges": {"produced_by": producers},
        }

    def patch(self, path: str, *, body: object = None) -> PlainTree:
        _, _, _, row, kind, to, _ = path.split("/")
        # A label on an edge that is not there is the server's 404.
        labels = self.edges[UUID(row), UUID(to), kind]
        change = from_plain(body, _LabelOp)
        if change["op"] == "add" and change["value"] not in labels:
            labels.append(change["value"])
        if change["op"] == "sub" and change["value"] in labels:
            labels.remove(change["value"])
        return None

    def list_kind(
        self,
        kind: str,
        *,
        filters: Sequence[Filter] = (),
    ) -> list[dict[str, PlainTree]]:
        assert kind == "WebResult"
        names = {"id": "id", "webresult_url": "url", "account": "account"}
        return [
            json.loads(json.dumps(row))
            for row in self.rows.values()
            if all(str(row[names[each.field]]) == each.value for each in filters)
        ]

    def submit(self, kind: str, body: Mapping[str, object]) -> UUID:
        assert kind == "WebResult"
        for row in self.rows.values():
            if row["key"] == body["idempotency_key"]:
                return UUID(str(row["id"]))
        row = uuid4()
        self.rows[row] = {
            "id": str(row),
            "key": body["idempotency_key"],
            "url": body["url"],
            "title": body["title"],
            "labels": body["labels"],
            "account": _ME,
            "status": "active",
            "app": None,
        }
        return row

    def add_edge(
        self,
        from_id: UUID,
        to_id: UUID,
        edge_kind: str,
        *,
        actor: str,
        note: str = "",
    ) -> object:
        del actor, note
        _ = self.edges.setdefault((from_id, to_id, edge_kind), [])
        return None

    def transition_app(
        self,
        target_id: UUID,
        *,
        expected_from: dict[str, object] | None,
        to: dict[str, object] | None,
        actor: str,
    ) -> None:
        del actor
        if self.racing is not None:
            race, self.racing = self.racing, None
            race()
            raise ClientError("changed since", status_code=409)
        assert self.rows[target_id]["app"] == expected_from
        self.rows[target_id]["app"] = json.loads(json.dumps(to))

    def transition_status(
        self,
        target_id: UUID,
        *,
        expected_from: str,
        to: str,
        actor: str,
    ) -> None:
        del actor
        assert self.rows[target_id]["status"] == expected_from
        self.rows[target_id]["status"] = to

    def pr(self) -> dict[str, object]:
        """Return the one row's ``app.knowop.pr``."""
        (row,) = self.rows.values()
        return from_plain(row["app"], dict[str, dict[str, dict[str, object]]])[
            "knowop"
        ]["pr"]

    def labels(self, session: UUID = _SESSION) -> list[str]:
        """Return the labels of the one row's edge to ``session``."""
        ((labels),) = (
            labels for (_, to, _), labels in self.edges.items() if to == session
        )
        return labels


@dataclass(slots=True, kw_only=True)
class _Gh:
    """Prints a pull request as ``gh pr view --jq`` does, and the login as ``gh api``."""

    state: str = "OPEN"
    head_sha: str = _SHA
    author: str = "me-on-github"
    login: str = "me-on-github"
    checks: list[dict[str, object]] = field(default_factory=list)
    fails: str = ""
    none_for: set[str] = field(default_factory=set)
    """Branches with no pull request yet."""

    refs: list[str] = field(default_factory=list)

    def __call__(self, args: Sequence[str], cwd: Path) -> GhOutput:
        del cwd
        if args[0] == "api":
            return GhOutput(returncode=0, stdout=f"{self.login}\n", stderr="")
        ref = "" if args[2] == "--json" else args[2]
        self.refs.append(ref)
        if ref in self.none_for:
            said = f'no pull requests found for branch "{ref}"\n'
            return GhOutput(returncode=1, stdout="", stderr=said)
        if self.fails:
            return GhOutput(returncode=1, stdout="", stderr=self.fails)
        view = {
            "number": 7,
            "url": _URL,
            "title": "Fix it",
            "state": self.state,
            "author": self.author,
            "draft": False,
            "head": _BRANCH,
            "head_sha": self.head_sha,
            "base": "main",
            "additions": 1,
            "deletions": 0,
            "mergeable": "MERGEABLE",
            "review": "",
            "checks": self.checks,
        }
        return GhOutput(returncode=0, stdout=json.dumps(view), stderr="")


@dataclass(slots=True, kw_only=True)
class _Git:
    """Answers as ``git`` does in a checkout on ``branch``; empty for a detached HEAD."""

    branch: str = ""

    def __call__(self, argv: Sequence[str], max_bytes: int) -> Output:
        del max_bytes
        answers = {
            "HEAD": self.branch,
            "refs/remotes/origin/HEAD": "origin/main",
            "remote.origin.url": "git@github.com:acme/app.git",
        }
        said = answers[argv[-1]]
        return Output(
            returncode=0 if said else 1,
            stdout=f"{said}\n".encode(),
            cut=False,
        )


def _check(
    name: str,
    *,
    conclusion: str = "SUCCESS",
    status: str = "COMPLETED",
) -> dict[str, object]:
    return {
        "name": name,
        "workflow": "CI",
        "commit_status": False,
        "status": status,
        "conclusion": conclusion,
        "state": "",
    }


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _watch(
    server: _Server,
    gh: _Gh,
    clock: _Clock | None = None,
    *,
    git: _Git | None = None,
    session: UUID = _SESSION,
) -> PullRequestWatch:
    return PullRequestWatch(
        client=server,
        session=lambda: session,
        actor="Agent",
        cwd=Path("/work"),
        gh=gh,
        git=git or _Git(),
        clock=clock or _Clock(),
        poll_sec=3600.0,
    )


def _opened(watch: PullRequestWatch, *, call_id: str = "c1") -> None:
    """Show the watch a claude Bash call of ``gh pr create`` and its result."""
    watch.note(
        ToolCall(
            call_id=call_id,
            name="Bash",
            arguments={"command": "gh pr create -f"},
        ),
    )
    watch.note(ShellCommandResult(call_id=call_id, stdout=f"{_URL}\n"))


def _pushed(watch: PullRequestWatch) -> None:
    watch.note(
        ToolCall(call_id="p", name="Bash", arguments={"command": "git push -u origin"}),
    )
    watch.note(ShellCommandResult(call_id="p", stdout=""))


def _tool(watch: PullRequestWatch, name: str) -> AgentTool:
    (tool,) = (each for each in watch.tools() if each.name == name)
    return tool


def test_a_pull_request_the_session_opened_gets_a_row_produced_by_the_session() -> None:
    server, gh = _Server(), _Gh(checks=[_check("lint")])
    watch = _watch(server, gh)
    _opened(watch)
    watch._sweep()
    ((row_id, row),) = server.rows.items()
    assert row["url"] == _URL
    assert row["title"] == "acme/app#7 Fix it"
    assert row["labels"] == [LABEL]
    assert server.edges == {(row_id, _SESSION, "produced_by"): [LABEL]}
    pr = server.pr()
    assert pr["state"] == "OPEN"
    assert pr["checks"] == {"passed": 1, "failed": [], "pending": [], "total": 1}
    assert pr["error"] is None


def test_a_codex_result_names_its_own_command_and_a_failed_one_counts_for_nothing() -> (
    None
):
    server, gh = _Server(), _Gh()
    watch = _watch(server, gh)
    watch.note(
        ShellCommandResult(
            call_id="x",
            command=("bash", "-lc", "gh pr create --fill"),
            stdout=_URL,
            exit_code=1,
        ),
    )
    watch._sweep()
    assert server.rows == {}
    watch.note(
        ShellCommandResult(call_id="y", command=("gh", "pr", "create"), stdout=_URL),
    )
    watch._sweep()
    assert len(server.rows) == 1


def test_the_branch_the_run_started_on_brings_only_my_own_pull_request() -> None:
    server, gh = _Server(), _Gh(author="someone-else")
    watch = _watch(server, gh, git=_Git(branch=_BRANCH))
    watch._sweep()
    assert gh.refs == [_BRANCH]
    assert server.rows == {}
    # Found, if not mine: the branch is looked for no more until it is pushed.
    watch._sweep()
    assert gh.refs == [_BRANCH]
    _pushed(watch)
    watch._sweep()
    assert [row["url"] for row in server.rows.values()] == [_URL]


def test_my_own_pull_request_on_the_start_branch_is_watched() -> None:
    server, gh = _Server(), _Gh()
    watch = _watch(server, gh, git=_Git(branch=_BRANCH))
    watch._sweep()
    assert gh.refs == [_BRANCH, _URL]
    assert len(server.rows) == 1


@pytest.mark.parametrize("branch", ["", "main"])
def test_a_detached_head_or_the_default_branch_is_never_looked_for(branch: str) -> None:
    server, gh = _Server(), _Gh()
    watch = _watch(server, gh, git=_Git(branch=branch))
    _pushed(watch)
    watch._sweep()
    assert gh.refs == []


def test_a_pushed_branch_is_looked_for_until_its_pull_request_opens(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server, gh = _Server(), _Gh(none_for={_BRANCH})
    watch = _watch(server, gh, git=_Git(branch=_BRANCH))
    _pushed(watch)
    with caplog.at_level("WARNING"):
        watch._sweep()
        watch._sweep()
    assert server.rows == {}
    assert gh.refs == [_BRANCH, _BRANCH]
    # No pull request yet is no failure to log.
    assert caplog.records == []
    gh.none_for.clear()
    watch._sweep()
    assert len(server.rows) == 1


def test_nothing_noted_reads_nothing() -> None:
    server, gh = _Server(), _Gh()
    watch = _watch(server, gh)
    watch.note(ToolCall(call_id="l", name="Bash", arguments={"command": "ls"}))
    watch.note(ShellCommandResult(call_id="l", stdout=_URL))
    watch._sweep()
    assert gh.refs == []
    assert server.rows == {}


def test_a_failed_check_wakes_the_agent_once_per_head() -> None:
    server, gh = _Server(), _Gh(checks=[_check("unit", conclusion="FAILURE")])
    watch = _watch(server, gh)
    _opened(watch)
    watch._sweep()
    (wake,) = watch.take()
    assert wake.source == AUTOFIX_SOURCE
    assert wake.kind == "line"
    assert "> CI / unit" in wake.text
    assert server.pr()["told"] == [f"{_SHA}:CI / unit"]
    watch._sweep()
    assert watch.take() == []
    gh.head_sha = "b" * 40
    watch._sweep()
    assert len(watch.take()) == 1
    assert server.pr()["told"] == [f"{'b' * 40}:CI / unit"]


def test_a_new_run_of_the_session_keeps_watching_and_does_not_wake_twice() -> None:
    server, gh = _Server(), _Gh(checks=[_check("unit", conclusion="FAILURE")])
    first = _watch(server, gh)
    _opened(first)
    first._sweep()
    assert len(first.take()) == 1
    again = _watch(server, gh)
    again._sweep()
    assert gh.refs[-1] == _URL
    assert again.take() == []


@pytest.mark.parametrize(
    ("preference", "switch", "wakes"),
    [(None, None, 1), (False, None, 0), (False, True, 1), (True, False, 0)],
)
def test_the_pull_requests_switch_overrides_the_persons_preference(
    preference: bool | None,
    switch: bool | None,
    wakes: int,
) -> None:
    server, gh = _Server(autofix=preference), _Gh(checks=[_check("lint")])
    watch = _watch(server, gh)
    _opened(watch)
    watch._sweep()
    (row,) = server.rows.values()
    app = row["app"]
    assert isinstance(app, dict)
    app["knowop"]["pr"]["autofix"] = switch
    gh.checks = [_check("unit", conclusion="FAILURE")]
    watch._sweep()
    assert len(watch.take()) == wakes


def test_a_failure_waits_while_its_workflow_runs_another_check() -> None:
    clock = _Clock()
    server = _Server()
    gh = _Gh(
        checks=[
            _check("unit", conclusion="FAILURE"),
            _check("e2e", status="IN_PROGRESS", conclusion=""),
        ],
    )
    watch = _watch(server, gh, clock)
    _opened(watch)
    watch._sweep()
    assert watch.take() == []
    clock.now += HOLD_SEC
    watch._sweep()
    assert len(watch.take()) == 1


def test_a_merged_pull_request_completes_its_row_and_is_read_no_more() -> None:
    server, gh = _Server(), _Gh()
    watch = _watch(server, gh)
    _opened(watch)
    watch._sweep()
    gh.state = "MERGED"
    watch._sweep()
    (row,) = server.rows.values()
    assert row["status"] == "complete"
    reads = len(gh.refs)
    watch._sweep()
    assert len(gh.refs) == reads


def test_a_closed_one_is_read_now_and_then_and_watched_again_once_reopened() -> None:
    server, gh = _Server(), _Gh()
    watch = _watch(server, gh)
    _opened(watch)
    watch._sweep()
    gh.state = "CLOSED"
    watch._sweep()
    (row,) = server.rows.values()
    assert row["status"] == "abandoned"
    reads = len(gh.refs)
    for _ in range(CLOSED_PASSES - 1):
        watch._sweep()
    assert len(gh.refs) == reads
    gh.state = "OPEN"
    watch._sweep()
    assert len(gh.refs) == reads + 1
    assert row["status"] == "active"
    watch._sweep()
    assert len(gh.refs) == reads + 2


def test_a_new_run_reads_a_closed_row_of_its_session_at_its_second_pass() -> None:
    server, gh = _Server(), _Gh()
    first = _watch(server, gh)
    _opened(first)
    first._sweep()
    gh.state = "CLOSED"
    first._sweep()
    reads = len(gh.refs)
    gh.state = "OPEN"
    again = _watch(server, gh)
    again._sweep()
    assert len(gh.refs) == reads
    again._sweep()
    (row,) = server.rows.values()
    assert row["status"] == "active"
    assert len(gh.refs) == reads + 1


def test_the_watch_tool_watches_one_reopened_since_its_row_was_closed() -> None:
    server, gh = _Server(), _Gh(checks=[_check("lint")])
    watch = _watch(server, gh)
    _opened(watch)
    watch._sweep()
    gh.state = "CLOSED"
    watch._sweep()
    gh.state = "OPEN"
    watch.start()
    try:
        answer = _tool(watch, "watch_pull_request").call({"url": _URL})
    finally:
        watch.close()
    assert answer == "Watching acme/app#7: 1 of 1 checks passed; Auto-fix on."
    (row,) = server.rows.values()
    assert row["status"] == "active"


def test_only_the_first_session_watching_is_woken_until_it_lets_go() -> None:
    server, gh = _Server(), _Gh(checks=[_check("unit", conclusion="FAILURE")])
    first, other = _watch(server, gh), _watch(server, gh, session=_OTHER)
    _opened(first)
    first._sweep()
    assert len(first.take()) == 1
    _opened(other)
    gh.head_sha = "b" * 40
    other._sweep()
    first._sweep()
    assert (len(first.take()), len(other.take())) == (1, 0)
    assert server.pr()["fixer"] == str(_SESSION)
    # Let go in Chat while its run is down: the next session to read a failure fixes it.
    server.labels(_SESSION).append(DISMISSED)
    gh.head_sha = "c" * 40
    other._sweep()
    assert len(other.take()) == 1
    assert server.pr()["fixer"] == str(_OTHER)
    assert server.pr()["told"] == [f"{'c' * 40}:CI / unit"]


def test_two_sessions_reading_one_failure_at_once_wake_only_the_one_that_wrote_first() -> (
    None
):
    server, gh = _Server(), _Gh()
    first, other = _watch(server, gh), _watch(server, gh, session=_OTHER)
    _opened(first)
    first._sweep()
    server.pr().pop("fixer")
    _opened(other)
    gh.checks = [_check("unit", conclusion="FAILURE")]
    server.racing = first._sweep
    other._sweep()
    assert (len(first.take()), len(other.take())) == (1, 0)
    assert server.pr()["fixer"] == str(_SESSION)
    assert server.pr()["told"] == [f"{_SHA}:CI / unit"]


def test_the_watch_tool_says_when_another_chat_fixes_it() -> None:
    server, gh = _Server(), _Gh(checks=[_check("lint")])
    first, other = _watch(server, gh), _watch(server, gh, session=_OTHER)
    _opened(first)
    first._sweep()
    other.start()
    try:
        answer = _tool(other, "watch_pull_request").call({"url": _URL})
    finally:
        other.close()
    assert answer == (
        "Watching acme/app#7: 1 of 1 checks passed; Auto-fix on, in the chat that "
        "watched it first: its agent gets the Auto-fix lines."
    )


def test_one_dismissed_in_chat_is_let_go_and_comes_back_on_no_push() -> None:
    server, gh = _Server(), _Gh(checks=[_check("unit", conclusion="FAILURE")])
    watch = _watch(server, gh, git=_Git(branch=_BRANCH))
    _opened(watch)
    watch._sweep()
    assert len(watch.take()) == 1
    server.labels().append(DISMISSED)
    gh.head_sha = "b" * 40
    watch._sweep()
    reads = len(gh.refs)
    _opened(watch, call_id="c2")
    _pushed(watch)
    watch._sweep()
    watch._sweep()
    # The push looks its branch up once, and the pull request it finds stays let go.
    assert gh.refs[reads:] == [_BRANCH]
    assert watch.take() == []


def test_the_watch_tool_brings_one_back_and_answers_what_it_sees() -> None:
    server, gh = _Server(), _Gh(checks=[_check("lint")])
    watch = _watch(server, gh)
    _opened(watch)
    watch._sweep()
    server.labels().append(DISMISSED)
    watch._sweep()
    watch.start()
    try:
        answer = _tool(watch, "watch_pull_request").call({"url": f"see {_URL}"})
    finally:
        watch.close()
    assert answer == "Watching acme/app#7: 1 of 1 checks passed; Auto-fix on."
    assert server.labels() == [LABEL]


@pytest.mark.parametrize(
    ("state", "answer"),
    [
        (
            "OPEN",
            (
                "Watching acme/app#7: 0 of 2 checks passed, 1 failed, 1 running; "
                "Auto-fix off."
            ),
        ),
        ("MERGED", "acme/app#7 is merged; there is nothing left to watch."),
    ],
)
def test_the_watch_tool_answers_failures_runs_and_a_switch_off(
    state: str,
    answer: str,
) -> None:
    server = _Server(autofix=False)
    gh = _Gh(
        state=state,
        checks=[
            _check("unit", conclusion="FAILURE"),
            _check("e2e", status="IN_PROGRESS", conclusion=""),
        ],
    )
    watch = _watch(server, gh)
    watch.start()
    try:
        assert _tool(watch, "watch_pull_request").call({"url": _URL}) == answer
    finally:
        watch.close()


def test_the_unwatch_tool_lets_go_of_one_never_watched_and_keeps_it_gone() -> None:
    server, gh = _Server(), _Gh()
    watch = _watch(server, gh)
    watch.start()
    try:
        answer = _tool(watch, "unwatch_pull_request").call({"url": _URL})
    finally:
        watch.close()
    assert answer.startswith(f"Let go of {_URL}: Chat no longer shows it")
    assert server.labels() == [LABEL, DISMISSED]
    _opened(watch)
    watch._sweep()
    assert gh.refs == []


def test_a_tool_call_naming_no_pull_request_is_refused() -> None:
    watch = _watch(_Server(), _Gh())
    with pytest.raises(ToolError, match="one GitHub pull request's link"):
        _ = _tool(watch, "watch_pull_request").call({"url": "acme/app#7"})


def test_a_tool_call_is_answered_when_its_pass_fails() -> None:
    server, gh = _Server(down=True), _Gh()
    watch = _watch(server, gh)
    watch.start()
    try:
        answer = _tool(watch, "watch_pull_request").call({"url": _URL})
    finally:
        watch.close()
    assert answer == "The run could not do that; its log says why."


def test_a_read_that_fails_says_why_on_the_row() -> None:
    server, gh = _Server(), _Gh(fails="HTTP 401: Bad credentials\n")
    watch = _watch(server, gh)
    _opened(watch)
    watch._sweep()
    (row,) = server.rows.values()
    assert row["title"] == _URL
    assert server.pr()["error"] == "HTTP 401: Bad credentials"
    gh.fails = ""
    watch._sweep()
    assert server.pr()["error"] is None


def test_a_lasting_failure_is_logged_once_with_its_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server, gh = _Server(), _Gh(fails="HTTP 401: Bad credentials\n")
    watch = _watch(server, gh)
    _opened(watch)
    with caplog.at_level("WARNING"):
        for _ in range(3):
            watch._sweep()
    assert [record.getMessage() for record in caplog.records] == [
        f"knowop run: could not read {_URL}: HTTP 401: Bad credentials",
    ]


def test_a_write_that_lost_a_race_keeps_the_persons_switch() -> None:
    server, gh = _Server(), _Gh()
    watch = _watch(server, gh)
    _opened(watch)
    watch._sweep()

    def switch_off() -> None:
        (row,) = server.rows.values()
        app = row["app"]
        assert isinstance(app, dict)
        app["knowop"]["pr"]["autofix"] = False

    server.racing = switch_off
    gh.checks = [_check("lint")]
    watch._sweep()
    assert server.pr()["autofix"] is False
    assert from_plain(server.pr()["checks"], dict[str, object])["passed"] == 1


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
