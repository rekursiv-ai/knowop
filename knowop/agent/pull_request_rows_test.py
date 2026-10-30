"""Tests for a run session's pull request rows on the server.

The server is a :class:`_Client` that keeps the calls made and answers from what the
test set; the watch's own tests drive the rows end to end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

from trackinizer.client.errors import ClientError
from treekle import PlainTree

import pytest

from knowop.agent.pull_request_rows import (
    DISMISSED,
    LABEL,
    PullRequestRows,
    SessionRow,
    Stored,
)


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from trackinizer.wire.filters import Filter


_SESSION = UUID("00000000-0000-0000-0000-00000000000a")
_ROW = UUID("00000000-0000-0000-0000-0000000000b1")
_OTHER = UUID("00000000-0000-0000-0000-0000000000b2")
_URL = "https://github.com/acme/app/pull/7"


@dataclass(slots=True, kw_only=True)
class _Client:
    peers: list[PlainTree] = field(default_factory=list)
    """A session's backlinks, and a row's edges, as ``/api/web/get`` answers them."""

    rows: dict[str, dict[str, PlainTree]] = field(default_factory=dict)
    calls: list[tuple[str, object]] = field(default_factory=list)
    refusals: list[int] = field(default_factory=list)
    """The status each ``transition_*`` call is refused with, in turn; then none."""

    def get(
        self,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
    ) -> PlainTree:
        del params
        self.calls.append(("get", path))
        return {
            "backlinks": {"produced_by": self.peers},
            "edges": {"produced_by": self.peers},
        }

    def patch(self, path: str, *, body: object = None) -> PlainTree:
        self.calls.append((path, body))
        return None

    def list_kind(
        self,
        kind: str,
        *,
        filters: Sequence[Filter] = (),
    ) -> list[dict[str, PlainTree]]:
        del kind
        one = filters[0]  # A row by its id, or by its link and its account.
        self.calls.append(("list", one.value))
        return [self.rows[one.value]] if one.value in self.rows else []

    def submit(self, kind: str, body: Mapping[str, object]) -> UUID:
        del kind, body
        raise AssertionError

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
        self.calls.append(("edge", (from_id, to_id, edge_kind)))
        return None

    def transition_app(
        self,
        target_id: UUID,
        *,
        expected_from: dict[str, object] | None,
        to: dict[str, object] | None,
        actor: str,
    ) -> None:
        del target_id, actor
        self.calls.append(("app", (expected_from, to)))
        if self.refusals:
            raise ClientError("refused", status_code=self.refusals.pop(0))

    def transition_status(
        self,
        target_id: UUID,
        *,
        expected_from: str,
        to: str,
        actor: str,
    ) -> None:
        del target_id, actor
        self.calls.append(("status", (expected_from, to)))
        if self.refusals:
            raise ClientError("refused", status_code=self.refusals.pop(0))


def _row(
    row: UUID,
    *,
    url: str | None,
    labels: list[str],
    app: dict[str, PlainTree] | None = None,
) -> dict[str, PlainTree]:
    return {
        "id": str(row),
        "url": url,
        "labels": labels,
        "status": "active",
        "app": app,
    }


def _pr(**pr: PlainTree) -> dict[str, PlainTree]:
    return {"knowop": {"pr": pr}}


def test_the_sessions_rows_read_each_link_once_and_only_a_pull_requests() -> None:
    client = _Client(
        peers=[
            {
                "id": str(_ROW),
                "kind": "WebResult",
                "status": "active",
                "labels": [LABEL, DISMISSED],
            },
            # Not one this session's run linked: a search's result, an Issue.
            {"id": str(_OTHER), "kind": "WebResult", "status": "active"},
            {"id": str(_SESSION), "kind": "Issue", "labels": ["x"]},
        ],
        rows={str(_ROW): _row(_ROW, url=_URL, labels=[LABEL])},
    )
    rows = PullRequestRows(client=client, actor="Agent")
    expected = [SessionRow(url=_URL, status="active", dismissed=True)]
    assert rows.session_rows(_SESSION) == expected
    assert rows.session_rows(_SESSION) == expected
    reads = [call for call in client.calls if call[0] == "list"]
    assert reads == [("list", str(_ROW))]
    assert rows.linked(_URL) == _ROW


def test_a_row_is_linked_to_the_session_once_with_the_pull_request_label() -> None:
    client = _Client(rows={_URL: _row(_ROW, url=_URL, labels=[LABEL])})
    rows = PullRequestRows(client=client, actor="Agent")
    for _ in range(2):
        _ = rows.row(_URL, session=_SESSION, me="me@acme.test", pull_request=None)
    path = f"/api/edges/{_ROW}/produced_by/{_SESSION}/labels"
    assert [call for call in client.calls if call[0] != "list"] == [
        ("edge", (_ROW, _SESSION, "produced_by")),
        (path, {"op": "add", "value": LABEL, "actor": "Agent"}),
    ]


def _bump(pr: dict[str, object]) -> tuple[dict[str, object], int]:
    """Count one more read, and say how many the row held."""
    seen = pr.get("reads", 0)
    assert isinstance(seen, int)
    return pr | {"reads": seen + 1}, seen


def test_a_write_over_a_newer_row_changes_that_one_and_keeps_its_switch() -> None:
    newer = _pr(reads=5, autofix=False)
    client = _Client(
        rows={str(_ROW): _row(_ROW, url=_URL, labels=[LABEL], app=newer)},
        refusals=[409],
    )
    rows = PullRequestRows(client=client, actor="Agent")
    stored = Stored(row=_ROW, app=dict[str, object](_pr(reads=1)), status="active")
    assert rows.write_pr(stored, _bump) == ({"reads": 6, "autofix": False}, 5)
    writes = [call[1] for call in client.calls if call[0] == "app"]
    assert writes == [
        (_pr(reads=1), _pr(reads=2)),
        (newer, _pr(reads=6, autofix=False)),
    ]


def test_a_write_to_a_row_that_is_gone_says_so() -> None:
    client = _Client(refusals=[409])
    rows = PullRequestRows(client=client, actor="Agent")
    assert rows.write_pr(Stored(row=_ROW, app={}, status="active"), _bump) is None


@pytest.mark.parametrize("refusals", [[409, 409, 409], [500]])
def test_a_write_others_keep_beating_or_the_server_refuses_fails_the_pass(
    refusals: list[int],
) -> None:
    client = _Client(
        rows={str(_ROW): _row(_ROW, url=_URL, labels=[LABEL])},
        refusals=list(refusals),
    )
    rows = PullRequestRows(client=client, actor="Agent")
    with pytest.raises(ClientError):
        _ = rows.write_pr(Stored(row=_ROW, app={}, status="active"), _bump)
    assert [call[0] for call in client.calls].count("app") == len(refusals)


def test_an_unchanged_pr_is_not_written() -> None:
    client = _Client()
    rows = PullRequestRows(client=client, actor="Agent")
    stored = Stored(row=_ROW, app=dict[str, object](_pr(reads=1)), status="active")
    assert rows.write_pr(stored, lambda pr: (pr, "kept")) == ({"reads": 1}, "kept")
    assert client.calls == []


def test_a_row_moves_from_the_status_it_was_read_with() -> None:
    client = _Client(refusals=[409])
    rows = PullRequestRows(client=client, actor="Agent")
    rows.move(Stored(row=_ROW, app={}, status="abandoned"), status="abandoned")
    # Moved first by another session's run: refused, and nothing more to do.
    rows.move(Stored(row=_ROW, app={}, status="active"), status="abandoned")
    rows.move(Stored(row=_ROW, app={}, status="abandoned"), status="active")
    assert client.calls == [
        ("status", ("active", "abandoned")),
        ("status", ("abandoned", "active")),
    ]


@pytest.mark.parametrize(
    ("labels", "gone"),
    [(None, True), ([LABEL], False), ([LABEL, DISMISSED], True)],
)
def test_a_session_let_go_once_its_edge_is_dismissed_or_not_there(
    labels: list[str] | None,
    gone: bool,
) -> None:
    peers: list[PlainTree] = [{"id": str(_OTHER), "labels": [LABEL]}]
    if labels is not None:
        peers.append({"id": str(_SESSION), "labels": list(labels)})
    rows = PullRequestRows(client=_Client(peers=peers), actor="Agent")
    assert rows.let_go(_ROW, session=_SESSION) is gone


def test_a_dismissal_changes_one_label_of_the_edge_to_the_session() -> None:
    client = _Client()
    rows = PullRequestRows(client=client, actor="Agent")
    rows.dismiss(_ROW, session=_SESSION, dismissed=True)
    rows.dismiss(_ROW, session=_SESSION, dismissed=False)
    path = f"/api/edges/{_ROW}/produced_by/{_SESSION}/labels"
    assert client.calls == [
        (path, {"op": "add", "value": DISMISSED, "actor": "Agent"}),
        (path, {"op": "sub", "value": DISMISSED, "actor": "Agent"}),
    ]


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
