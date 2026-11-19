"""The pull request rows: listed with their watchers, made once, watched, let go, written."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Final, cast
from uuid import UUID, uuid4

from trackinizer.client.errors import ClientError
from trackinizer.wire.refs import SeqRef

import pytest

from knowop.watch_pr.rows import (
    DISMISSED,
    LABEL,
    PullRequestRows,
    Stored,
    Watcher,
)


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from trackinizer.types.inquiries import Inquiry
    from trackinizer.wire.filters import Filter
    from trackinizer.wire.refs import Ref
    from trackinizer.wire.wire_session_relay import NewInput
    from treekle import PlainTree


ME: Final = "ada@x.test"
ACTOR: Final = "knowop-watch-pr"
URL: Final = "https://github.com/acme/app/pull/7"
FIXER: Final = UUID(int=1)
OTHER: Final = UUID(int=2)


@dataclass(slots=True, kw_only=True)
class Peer:
    """An edge from a row to a session: its note and its labels."""

    session: UUID
    note: str = "Agent"
    labels: list[str] = field(default_factory=lambda: [LABEL])


@dataclass(slots=True, kw_only=True)
class Row:
    """A pull request row as the server holds it."""

    id: UUID = field(default_factory=uuid4)
    url: str = URL
    status: str = "active"
    app: dict[str, object] | None = None
    title: str = ""
    peers: list[Peer] = field(default_factory=lambda: [Peer(session=FIXER)])

    def plain(self) -> dict[str, PlainTree]:
        """Return the row as ``GET /api/inquiries`` lists it."""
        return {
            "id": str(self.id),
            "url": self.url,
            "status": self.status,
            "account": ME,
            "labels": [LABEL],
            "app": cast("PlainTree", self.app),
            "produced_by": [
                {
                    "id": str(peer.session),
                    "kind": "AgentSession",
                    "note": peer.note,
                    "labels": list(peer.labels),
                }
                for peer in self.peers
            ],
        }

    @property
    def pr(self) -> dict[str, object]:
        """The row's ``app.knowop.pr``."""
        knowop = cast("dict[str, object]", (self.app or {})["knowop"])
        return cast("dict[str, object]", knowop["pr"])


class Server:
    """The server's calls a watch makes, over rows held here, as Trackinizer answers.

    Attributes:
      rows: Each row by id.
      lines: Each line a session's agent was sent, as ``(session, text)``.
      asleep: Sessions with no agent running: a line to one is refused.
      preferences: The person's preferences.
      writes: How many ``app`` writes landed.
      racers: What another writer writes to the row's ``app`` just before each of
        the next ``app`` writes.
      app_refusal: What every ``app`` write raises, when set.
      status_refusal: What every status write raises, when set.
      down: What every ``GET`` raises, when set.
      gets: Each path read.

    """

    def __init__(self) -> None:
        self.rows: dict[UUID, Row] = {}
        self.lines: list[tuple[UUID, str]] = []
        self.asleep: set[UUID] = set()
        self.preferences: dict[str, bool] = {}
        self.writes = 0
        self.racers: list[dict[str, object] | None] = []
        self.app_refusal: ClientError | None = None
        self.status_refusal: ClientError | None = None
        self.down: ClientError | None = None
        self.gets: list[str] = []

    def row(
        self,
        *,
        url: str = URL,
        status: str = "active",
        app: dict[str, object] | None = None,
        peers: Sequence[Peer] = (Peer(session=FIXER),),
    ) -> Row:
        """Hold a pull request row, with copies of ``peers``; return it."""
        copies = [replace(peer, labels=list(peer.labels)) for peer in peers]
        row = Row(url=url, status=status, app=app, peers=copies)
        self.rows[row.id] = row
        return row

    def get(
        self,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
    ) -> PlainTree:
        assert params is None
        self.gets.append(path)
        if self.down is not None:
            raise self.down
        if path == "/api/me/profile":
            return {"email": ME, "preferences": dict(self.preferences)}
        session = UUID(path.removeprefix("/api/inquiries/"))
        return {"id": str(session), "seq": 400 + session.int}

    def patch(self, path: str, *, body: object = None) -> PlainTree:
        _, _, _, row, kind, session, _ = path.split("/")
        assert kind == "produced_by"
        assert body == {"op": "add", "value": DISMISSED, "actor": ACTOR}
        for peer in self.rows[UUID(row)].peers:
            if peer.session == UUID(session):
                peer.labels.append(DISMISSED)
        return None

    def list_kind(
        self,
        kind: Inquiry.InquiryKind,
        *,
        status: Inquiry.Status | None = None,
        limit: int = 50,
        filters: Sequence[Filter] = (),
    ) -> list[dict[str, PlainTree]]:
        assert kind == "WebResult"
        assert limit > 0
        return [
            row.plain()
            for row in self.rows.values()
            if (status is None or row.status == status)
            and all(_matches(row, each) for each in filters)
        ]

    def submit(self, kind: Inquiry.InquiryKind, body: Mapping[str, object]) -> UUID:
        assert kind == "WebResult"
        assert body["labels"] == [LABEL]
        assert isinstance(body["idempotency_key"], str)
        row = self.row(url=str(body["url"]), peers=())
        row.title = str(body["title"])
        return row.id

    def add_edge(
        self,
        from_id: UUID,
        to_id: UUID,
        edge_kind: str,
        *,
        actor: Inquiry.Actor,
        note: str = "",
        labels: Sequence[str] | None = (),
    ) -> object:
        assert (edge_kind, actor) == ("produced_by", ACTOR)
        peers = self.rows[from_id].peers
        kept = next((peer for peer in peers if peer.session == to_id), None)
        if kept is None:
            peers.append(Peer(session=to_id, note=note, labels=list(labels or ())))
        else:
            kept.note, kept.labels = note, list(labels or ())
        return None

    def transition_app(
        self,
        target_id: UUID,
        *,
        expected_from: dict[str, object] | None,
        to: dict[str, object] | None,
        actor: Inquiry.Actor,
    ) -> None:
        assert actor == ACTOR
        if self.app_refusal is not None:
            raise self.app_refusal
        row = self.rows[target_id]
        if self.racers:
            row.app = self.racers.pop(0)
        if row.app != expected_from:
            raise ClientError("PUT app -> 409", status_code=409)
        row.app = to
        self.writes += 1

    def transition_status(
        self,
        target_id: UUID,
        *,
        expected_from: str,
        to: str,
        actor: Inquiry.Actor,
    ) -> None:
        assert actor == ACTOR
        if self.status_refusal is not None:
            raise self.status_refusal
        row = self.rows[target_id]
        if row.status != expected_from:
            raise ClientError("PUT status -> 409", status_code=409)
        row.status = to

    def send_input(self, session_id: UUID, new: NewInput) -> UUID:
        assert new.kind == "UserMessage"
        assert new.payload["py/object"] == "UserMessage"
        if session_id in self.asleep:
            raise ClientError("POST inputs -> 409", status_code=409)
        self.lines.append((session_id, str(new.payload["content"])))
        return uuid4()

    def resolve_id(self, ref: Ref) -> tuple[Inquiry.InquiryKind, UUID]:
        assert isinstance(ref, SeqRef)
        return "AgentSession", UUID(int=ref.seq - 400)


def _matches(row: Row, each: Filter) -> bool:
    """Return whether ``row`` passes one filter, as the server reads it."""
    match each.field:
        case "labels":
            return each.value == LABEL
        case "account":
            return each.value == ME
        case "webresult_url":
            return each.value == row.url
        case _:
            return each.value == str(row.id)


def rows(server: Server) -> PullRequestRows:
    return PullRequestRows(client=server, actor=ACTOR)


def stored_row(server: Server, row: Row) -> Stored:
    found = rows(server).find(row.url, me=ME)
    assert found is not None
    return found


def test_a_listed_row_names_the_sessions_that_still_watch_it() -> None:
    server = Server()
    _ = server.row(
        peers=(
            Peer(session=FIXER, note="lead"),
            Peer(session=OTHER, note="", labels=[LABEL, DISMISSED]),
            Peer(session=UUID(int=3), labels=["other"]),
        ),
    )
    [stored] = rows(server).listed(ME, status="active")
    assert stored.watchers == (Watcher(session=FIXER, label="lead"),)
    assert (stored.url, stored.app, stored.pr) == (URL, {}, {})
    assert rows(server).listed(ME, status="abandoned") == []


def test_a_watcher_with_no_note_is_the_default_agent() -> None:
    server = Server()
    _ = server.row(peers=(Peer(session=FIXER, note=""),))
    [stored] = rows(server).listed(ME, status="active")
    assert stored.watchers == (Watcher(session=FIXER, label="Agent"),)


def test_a_row_is_made_once_per_link() -> None:
    server = Server()
    made = rows(server).made(URL, me=ME, title="acme/app#7 Fix")
    assert rows(server).made(URL, me=ME, title="other") == made
    assert server.rows[made].title == "acme/app#7 Fix"
    assert rows(server).find("https://github.com/acme/app/pull/8", me=ME) is None


def test_watching_sets_the_label_and_undoes_a_dismissal() -> None:
    server = Server()
    row = server.row(peers=(Peer(session=FIXER, labels=[LABEL, DISMISSED]),))
    rows(server).watch(row.id, watcher=Watcher(session=FIXER, label="lead"))
    rows(server).watch(row.id, watcher=Watcher(session=OTHER, label="Agent"))
    assert stored_row(server, row).watchers == (
        Watcher(session=FIXER, label="lead"),
        Watcher(session=OTHER, label="Agent"),
    )


def test_letting_go_adds_the_dismissed_label_alone() -> None:
    server = Server()
    row = server.row(peers=(Peer(session=FIXER, labels=[LABEL, "kept"]),))
    rows(server).dismiss(row.id, session=FIXER)
    assert row.peers[0].labels == [LABEL, "kept", DISMISSED]
    assert stored_row(server, row).watchers == ()


def test_a_write_keeps_the_other_app_keys_and_makes_none_when_nothing_changed() -> None:
    server = Server()
    row = server.row(app={"other": 1, "knowop": {"x": 2, "pr": {"a": 1}}})
    same = rows(server).write_pr(
        stored_row(server, row),
        lambda kept: (kept.pr, "same"),
    )
    assert (same, server.writes) == (({"a": 1}, "same"), 0)
    written = rows(server).write_pr(
        stored_row(server, row),
        lambda kept: (kept.pr | {"b": 2}, "made"),
    )
    assert written == ({"a": 1, "b": 2}, "made")
    assert row.app == {"other": 1, "knowop": {"x": 2, "pr": {"a": 1, "b": 2}}}


def test_a_first_write_names_no_app_it_was_guarded_by() -> None:
    server = Server()
    row = server.row(app=None)
    _ = rows(server).write_pr(
        stored_row(server, row),
        lambda kept: (kept.pr | {"a": 1}, None),
    )
    assert row.app == {"knowop": {"pr": {"a": 1}}}


def test_a_write_another_beat_is_made_again_over_theirs() -> None:
    server = Server()
    row = server.row()
    server.racers = [{"knowop": {"pr": {"autofix": False}}}]
    written = rows(server).write_pr(
        stored_row(server, row),
        lambda kept: (kept.pr | {"read": True}, kept.pr.get("autofix")),
    )
    assert written == ({"autofix": False, "read": True}, False)
    assert server.writes == 1


def test_a_write_to_a_row_gone_meanwhile_is_none() -> None:
    server = Server()
    row = server.row()
    stored = stored_row(server, row)
    del server.rows[row.id]
    server.app_refusal = ClientError("PUT app -> 409", status_code=409)
    assert (
        rows(server).write_pr(stored, lambda kept: (kept.pr | {"a": 1}, None)) is None
    )


def test_a_write_others_keep_beating_or_the_server_refuses_raises() -> None:
    server = Server()
    row = server.row()
    server.racers = [{"n": n} for n in range(3)]
    with pytest.raises(ClientError, match="409"):
        _ = rows(server).write_pr(
            stored_row(server, row),
            lambda kept: (kept.pr | {"a": 1}, None),
        )
    assert server.racers == []
    server.app_refusal = ClientError("PUT app -> 403", status_code=403)
    with pytest.raises(ClientError, match="403"):
        _ = rows(server).write_pr(
            stored_row(server, row),
            lambda kept: (kept.pr | {"a": 1}, None),
        )


def test_a_move_is_made_once_and_one_another_made_first_is_kept() -> None:
    server = Server()
    row = server.row()
    stored = stored_row(server, row)
    rows(server).move(stored, status="active")
    rows(server).move(stored, status="complete")
    assert row.status == "complete"
    rows(server).move(stored, status="abandoned")
    assert row.status == "complete"


def test_a_move_the_server_refuses_raises() -> None:
    server = Server()
    row = server.row()
    server.status_refusal = ClientError("PUT status -> 403", status_code=403)
    with pytest.raises(ClientError, match="403"):
        rows(server).move(stored_row(server, row), status="complete")


def test_the_profile_is_the_persons() -> None:
    server = Server()
    server.preferences = {"autofix": False}
    assert rows(server).profile() == {"email": ME, "preferences": {"autofix": False}}


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
