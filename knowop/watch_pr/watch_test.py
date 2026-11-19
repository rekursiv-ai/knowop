"""A pass of the watch: read through gh, written once, each piece of news told once."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, override
from uuid import UUID

from trackinizer.client.errors import ClientError

from knowop.watch_pr.github import GhOutput
from knowop.watch_pr.github_test import URL, check, printed, view
from knowop.watch_pr.news import unreadable
from knowop.watch_pr.rows import DISMISSED, LABEL
from knowop.watch_pr.rows_test import (
    FIXER,
    OTHER,
    Peer,
    Server,
    stored_row,
)
from knowop.watch_pr.watch import CLOSED_PASSES, HEARTBEAT_SEC, Watch


if TYPE_CHECKING:
    from collections.abc import Sequence

    import pytest


LOGIN: Final = GhOutput(returncode=0, stdout="ada\n", stderr="")

FAILING: Final = view(checks=[check("unit"), check("lint", conclusion="FAILURE")])

TWO: Final = (Peer(session=FIXER), Peer(session=OTHER, note="lead"))


class Gh:
    """A ``gh`` whose pull request reads as ``answer`` says, its calls kept."""

    def __init__(self, answer: dict[str, object] | GhOutput = FAILING) -> None:
        self.answer = answer
        self.calls: list[list[str]] = []
        self.login = LOGIN
        self.merge = GhOutput(returncode=0, stdout="", stderr="")

    def __call__(self, args: Sequence[str]) -> GhOutput:
        self.calls.append(list(args))
        if args[0] == "api":
            return self.login
        if args[:2] == ["pr", "merge"]:
            return self.merge
        if isinstance(self.answer, GhOutput):
            return self.answer
        return printed(self.answer)

    def merges(self) -> list[list[str]]:
        """Return each ``gh pr merge`` call."""
        return [call for call in self.calls if call[:2] == ["pr", "merge"]]


class Clock:
    """Seconds, which the test moves."""

    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def watch(server: Server, gh: Gh, clock: Clock | None = None) -> Watch:
    held = clock or Clock()
    return Watch(client=server, host="laptop", gh=gh, clock=held, wall=held)


def test_a_failure_is_written_and_told_to_each_watcher_by_its_role() -> None:
    server = Server()
    row = server.row(peers=TWO)
    assert watch(server, Gh()).sweep() == 1
    pr = row.pr
    assert pr["checks"] == {
        "passed": 1,
        "failed": ["CI / lint"],
        "pending": [],
        "total": 2,
    }
    assert pr["told"] == ["0123456789abcdef:CI / lint"]
    assert (pr["fixer"], pr["authored"], pr["error"], pr["read_by"]) == (
        str(FIXER),
        True,
        None,
        "laptop",
    )
    assert pr["read_at"] == "1970-01-12T13:46:40+00:00"
    [(fixer, fixing), (other, watching)] = server.lines
    assert (fixer, other) == (FIXER, OTHER)
    assert "Auto-fix is on for this pull request" in fixing
    assert "Also watching it: lead#402; Agent#401 fixes it." in fixing
    assert "Agent#401 fixes it: do not push" in watching
    assert "--session lead#402" in watching


def test_a_second_watch_tells_nothing_the_first_told() -> None:
    server = Server()
    _ = server.row(peers=TWO)
    clock = Clock()
    _ = watch(server, Gh(), clock).sweep()
    writes = server.writes
    _ = watch(server, Gh(), clock).sweep()
    assert len(server.lines) == 2
    assert server.writes == writes


def test_a_race_another_watch_won_is_read_again_and_tells_nothing() -> None:
    server = Server()
    row = server.row(peers=TWO)
    clock = Clock()
    _ = watch(server, Gh(), clock).sweep()
    won, row.app = row.app, None
    server.lines.clear()
    server.racers = [won]
    _ = watch(server, Gh(), clock).sweep()
    assert server.lines == []


def test_a_session_that_let_it_go_hears_nothing_and_a_row_none_watches_is_not_read() -> (
    None
):
    server = Server()
    _ = server.row(
        peers=(Peer(session=FIXER), Peer(session=OTHER, labels=[LABEL, DISMISSED])),
    )
    _ = server.row(
        url="https://github.com/acme/app/pull/8",
        peers=(Peer(session=OTHER, labels=[LABEL, DISMISSED]),),
    )
    gh = Gh()
    assert watch(server, gh).sweep() == 1
    assert [session for session, _ in server.lines] == [FIXER]
    assert [call[2] for call in gh.calls if call[0] == "pr"] == [URL]


class Emptied(Server):
    """A server whose row every watcher lets go of just before the first write."""

    @override
    def transition_app(
        self,
        target_id: UUID,
        *,
        expected_from: dict[str, object] | None,
        to: dict[str, object] | None,
        actor: str,
    ) -> None:
        row = self.rows[target_id]
        if row.peers:
            row.peers.clear()
            raise ClientError("PUT app -> 409", status_code=409)
        super().transition_app(
            target_id,
            expected_from=expected_from,
            to=to,
            actor=actor,
        )


def test_a_row_every_watcher_let_go_of_meanwhile_is_written_and_tells_none() -> None:
    server = Emptied()
    row = server.row()
    _ = watch(server, Gh()).sweep()
    assert server.writes == 1
    assert (row.pr["fixer"], row.pr["told"]) == (None, ["0123456789abcdef:CI / lint"])
    assert server.lines == []


def test_a_session_with_no_agent_running_is_told_nothing_and_the_rest_are() -> None:
    server = Server()
    _ = server.row(peers=TWO)
    server.asleep = {FIXER}
    _ = watch(server, Gh()).sweep()
    assert [session for session, _ in server.lines] == [OTHER]


def test_auto_fix_off_by_switch_or_preference_tells_no_failure() -> None:
    server = Server()
    _ = server.row(app={"knowop": {"pr": {"autofix": False}}})
    _ = watch(server, Gh()).sweep()
    assert server.lines == []
    server = Server()
    server.preferences = {"autofix": False}
    _ = server.row()
    _ = watch(server, Gh()).sweep()
    assert server.lines == []


def test_the_switch_wins_over_the_preference() -> None:
    server = Server()
    server.preferences = {"autofix": False}
    _ = server.row(app={"knowop": {"pr": {"autofix": True}}})
    _ = watch(server, Gh()).sweep()
    assert len(server.lines) == 1


def test_anothers_pull_request_is_fixed_only_when_its_switch_asks() -> None:
    for login in (GhOutput(returncode=0, stdout="grace\n", stderr=""), None):
        server = Server()
        row = server.row()
        gh = Gh()
        if login is not None:
            gh.login = login
        else:
            gh.login = GhOutput(returncode=1, stdout="", stderr="not logged in")
        _ = watch(server, gh).sweep()
        assert server.lines == []
        assert row.pr["told"] == []
        row.app = {"knowop": {"pr": row.pr | {"autofix": True}}}
        _ = watch(server, gh).sweep()
        assert len(server.lines) == 1


LOGGED_OUT: Final = GhOutput(returncode=4, stdout="", stderr="gh auth login")


def test_a_read_that_fails_is_kept_on_the_row_and_told_once_to_each_watcher() -> None:
    server = Server()
    row = server.row(app={"knowop": {"pr": {"state": "OPEN"}}}, peers=TWO)
    watching = watch(server, Gh(LOGGED_OUT))
    _ = watching.sweep()
    assert (row.pr["error"], row.status) == ("gh auth login", "active")
    assert [session for session, _ in server.lines] == [FIXER, OTHER]
    assert server.lines[0][1] == unreadable(URL, host="laptop", error="gh auth login")
    _ = watching.sweep()
    _ = watch(server, Gh(LOGGED_OUT)).sweep()
    assert len(server.lines) == 2


def test_a_row_gone_meanwhile_tells_nothing_read_or_not() -> None:
    for gh in (Gh(), Gh(LOGGED_OUT)):
        server = Server()
        row = server.row()
        stored = stored_row(server, row)
        del server.rows[row.id]
        server.app_refusal = ClientError("PUT app -> 409", status_code=409)
        watch(server, gh).refresh(stored, autofix=True, automerge=False)
        assert server.lines == []


def test_a_serve_that_cannot_read_tells_once_while_another_reads_it() -> None:
    server = Server()
    row = server.row()
    clock = Clock()
    gh = Gh(LOGGED_OUT)
    failing = watch(server, gh, clock)
    reading = watch(server, Gh(view()), clock)
    for _ in range(2):
        _ = failing.sweep()
        _ = reading.sweep()
        assert row.pr["error"] is None
    assert len(server.lines) == 1
    gh.answer = view()
    _ = failing.sweep()
    gh.answer = LOGGED_OUT
    _ = failing.sweep()
    assert len(server.lines) == 2


def test_a_lasting_failure_is_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    server = Server()
    _ = server.row()
    watching = watch(server, Gh(GhOutput(returncode=1, stdout="", stderr="HTTP 404")))
    _ = watching.sweep()
    _ = watching.sweep()
    said = [each.getMessage() for each in caplog.records]
    assert said == [f"knowop watch-pr: could not read {URL}: HTTP 404"]


def test_an_unknown_login_leaves_authored_unset() -> None:
    server = Server()
    row = server.row()
    gh = Gh()
    gh.login = GhOutput(returncode=1, stdout="", stderr="not logged in")
    _ = watch(server, gh).sweep()
    assert "authored" not in row.pr


def test_the_login_is_asked_once() -> None:
    server = Server()
    _ = server.row()
    gh = Gh()
    watching = watch(server, gh)
    _ = watching.sweep()
    _ = watching.sweep()
    assert [call for call in gh.calls if call[0] == "api"] == [
        ["api", "user", "--jq", ".login"],
    ]


def test_an_unchanged_row_is_written_again_only_once_its_read_goes_stale() -> None:
    server = Server()
    _ = server.row()
    clock = Clock()
    watching = watch(server, Gh(), clock)
    _ = watching.sweep()
    writes = server.writes
    clock.now += HEARTBEAT_SEC - 1
    _ = watching.sweep()
    assert server.writes == writes
    clock.now += 2
    _ = watching.sweep()
    assert server.writes == writes + 1


def test_a_merge_seen_open_moves_the_row_and_is_told() -> None:
    server = Server()
    row = server.row(app={"knowop": {"pr": {"state": "OPEN"}}})
    _ = watch(server, Gh(view(state="MERGED"))).sweep()
    assert row.status == "complete"
    [(_, merged)] = server.lines
    assert merged.startswith("<knowop-merged ")


def test_a_state_github_never_names_keeps_the_row_open() -> None:
    server = Server()
    row = server.row()
    _ = watch(server, Gh(view(state="DRAFTED"))).sweep()
    assert row.status == "active"


def test_a_closed_row_is_read_on_the_first_pass_and_every_thirtieth_after() -> None:
    server = Server()
    _ = server.row(status="abandoned")
    watching = watch(server, Gh(view(state="CLOSED")))
    reads = [watching.sweep() for _ in range(CLOSED_PASSES + 1)]
    assert reads == [1] + [0] * (CLOSED_PASSES - 1) + [1]


def test_a_failure_waits_for_its_workflow_then_is_told() -> None:
    server = Server()
    _ = server.row()
    clock = Clock()
    gh = Gh(
        view(
            checks=[
                check("lint", conclusion="FAILURE"),
                check("unit", status="IN_PROGRESS", conclusion=""),
            ],
        ),
    )
    watching = watch(server, gh, clock)
    _ = watching.sweep()
    assert server.lines == []
    clock.now += 299
    _ = watching.sweep()
    assert server.lines == []
    clock.now += 1
    _ = watching.sweep()
    assert len(server.lines) == 1


def test_a_new_head_starts_its_own_hold() -> None:
    server = Server()
    _ = server.row()
    clock = Clock()
    held = view(
        checks=[
            check("lint", conclusion="FAILURE"),
            check("unit", status="IN_PROGRESS", conclusion=""),
        ],
    )
    gh = Gh(held)
    watching = watch(server, gh, clock)
    _ = watching.sweep()
    clock.now += 299
    gh.answer = held | {"head_sha": "f" * 16}
    _ = watching.sweep()
    clock.now += 1
    _ = watching.sweep()
    assert server.lines == []


def test_the_fixer_stays_until_it_lets_go_then_the_next_watcher_fixes() -> None:
    server = Server()
    row = server.row(
        peers=(Peer(session=OTHER, note="lead"), Peer(session=FIXER)),
        app={"knowop": {"pr": {"fixer": str(FIXER)}}},
    )
    _ = watch(server, Gh()).sweep()
    assert row.pr["fixer"] == str(FIXER)
    [(_, other), (_, fixer)] = server.lines
    assert "Auto-fix is on for this pull request" in fixer
    assert "Agent#401 fixes it: do not push" in other
    row.peers[1].labels.append(DISMISSED)
    _ = watch(server, Gh()).sweep()
    assert row.pr["fixer"] == str(OTHER)


def test_the_automerge_preference_turns_github_auto_merge_on_once() -> None:
    server = Server()
    server.preferences = {"automerge": True}
    row = server.row()
    gh = Gh(view())
    _ = watch(server, gh).sweep()
    first = row.pr
    _ = watch(server, gh).sweep()
    assert gh.merges() == [["pr", "merge", URL, "--auto", "--squash"]]
    assert (first["automerge_done"], first["automerge_error"], first["auto_merge"]) == (
        "default",
        None,
        True,
    )


def test_an_ask_github_refuses_is_kept_and_one_github_already_did_needs_no_call() -> (
    None
):
    server = Server()
    row = server.row(
        app={"knowop": {"pr": {"automerge_ask": {"on": False, "at": "t1"}}}},
    )
    gh = Gh(view(auto_merge=True))
    gh.merge = GhOutput(returncode=1, stdout="", stderr="not allowed")
    _ = watch(server, gh).sweep()
    assert gh.merges() == [["pr", "merge", URL, "--disable-auto"]]
    pr = row.pr
    assert (pr["automerge_done"], pr["automerge_error"], pr["auto_merge"]) == (
        "t1",
        "not allowed",
        True,
    )
    server = Server()
    row = server.row(
        app={"knowop": {"pr": {"automerge_ask": {"on": True, "at": "t2"}}}},
    )
    gh = Gh(view(auto_merge=True))
    _ = watch(server, gh).sweep()
    assert gh.merges() == []
    assert (row.pr["automerge_done"], row.pr["automerge_error"]) == ("t2", None)


def test_no_ask_and_no_preference_and_a_closed_one_need_no_call() -> None:
    server = Server()
    _ = server.row()
    gh = Gh(view())
    _ = watch(server, gh).sweep()
    server = Server()
    _ = server.row(app={"knowop": {"pr": {"automerge_ask": {"on": True, "at": "t"}}}})
    _ = watch(server, gh := Gh(view(state="CLOSED"))).sweep()
    assert gh.merges() == []


def test_a_drafts_ask_to_turn_auto_merge_on_waits() -> None:
    server = Server()
    row = server.row(
        app={"knowop": {"pr": {"automerge_ask": {"on": True, "at": "t1"}}}},
    )
    gh = Gh(view(draft=True))
    _ = watch(server, gh).sweep()
    assert gh.merges() == []
    assert "automerge_done" not in row.pr


def test_a_write_the_server_refuses_skips_the_row_and_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = Server()
    _ = server.row()
    server.app_refusal = ClientError("PUT app -> 403", status_code=403)
    assert watch(server, Gh()).sweep() == 1
    assert server.lines == []
    assert [each.getMessage() for each in caplog.records] == [
        f"knowop watch-pr: could not write {URL}: PUT app -> 403",
    ]


def test_a_session_with_no_agent_is_logged_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = Server()
    _ = server.row()
    server.asleep = {FIXER}
    _ = watch(server, Gh()).sweep()
    assert [each.getMessage() for each in caplog.records] == [
        f"knowop watch-pr: told session {FIXER} nothing: POST inputs -> 409",
    ]


def test_a_session_is_named_by_its_number_read_once() -> None:
    server = Server()
    row = server.row(peers=TWO)
    clock = Clock()
    watching = watch(server, Gh(), clock)
    _ = watching.sweep()
    row.app = None
    _ = watching.sweep()
    assert server.gets.count(f"/api/inquiries/{UUID(int=1)}") == 1
    assert len(server.lines) == 4


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
