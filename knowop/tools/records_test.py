"""Tests for KnowOp's trax reads: trax's read commands, run for one agent."""

from __future__ import annotations

from typing import Final, override
from urllib.parse import parse_qsl
from uuid import UUID

from trackinizer.trax.cli import DISPATCHERS
from trackinizer.trax.grammar import VALID_KINDS
from trackinizer.wire.wire_process_keys import RECORDS_ROUTES, listed_route

import httpx2

from knowop.tools.knowop import KNOWOP_DESCRIPTION
from knowop.tools.records import READ_DISPATCHERS, RecordReads


_ROW: Final = "0b4c2c1e-0000-4000-8000-000000000007"

_READ_FORMS: Final = (
    "issue",
    "title re cache",
    "issue 7",
    f"issue {_ROW}",
    "issue 7 requires",
    "issue 7 description",
    "agentsession 3 records",
    "recent --limit 5",
    'search-sessions "tile cache"',
    "next",
    "blocked",
    "graph --open-only",
    "board",
    "cost issue 7",
    "confidence belief 3",
    "authority belief 3",
    f"id {_ROW}",
    "version",
)
"""One command per read form the tool runs; between them they call every route."""

_BODIES: Final[dict[str, object]] = {
    "/api/inquiries/{kind}/{seq}": {"id": _ROW, "kind": "Issue", "seq": 7},
    "/api/web/lookup/{target_id}": {"kind": "Issue"},
    "/api/web/get/{target_id}": {
        "self": {
            "id": _ROW,
            "kind": "Issue",
            "seq": 7,
            "title": "Tile cache",
            "description": "Why the tile cache misses.",
        },
        "edges": {},
        "backlinks": {},
        "changes": [],
    },
    "/api/sessions/{session_id}/parts": {
        "parts": [{"part": 0, "name": "main.jsonl", "format": "sagent", "records": 0}],
    },
    "/api/sessions/{session_id}/records": {"part": 0, "records": []},
    "/api/web/search_sessions": {"hits": [], "semantic": False, "degraded": False},
    "/api/inquiries/next_issue": None,
    "/api/inquiries/{target_id}/cost": {},
    "/api/inquiries/{target_id}/confidence": {"confidence": 0.5},
    "/api/inquiries/{target_id}/authority": {},
    "/api/version": {"sha": "abc123"},
}
"""Each route's smallest body its reader accepts, by template; any other is ``[]``."""


def test_runs_a_read_command_with_its_own_client() -> None:
    """A leading ``trax`` is dropped; the agent's key goes on every request."""
    transport = _Server()

    said = _answer("trax version", transport)

    assert said == "abc123\n"
    assert transport.sent == [("GET", "/api/version", "Bearer agent-key")]


def test_every_read_form_runs_and_between_them_they_call_every_listed_route() -> None:
    """The list is measured by the reads that call it, so the two cannot drift.

    A form whose route is missing fails with the client's refusal, and a listed route
    no form calls would widen what an agent's key reaches for nothing. Each request
    counts only when its parameters are ones that route accepts (``_routed``), so a
    path the client's matcher takes for a listed template but the server routes or
    rejects otherwise fails here.
    """
    transport = _Server()

    failed = {
        command: said
        for command in _READ_FORMS
        if _failed(said := _answer(command, transport))
    }

    assert failed == {}
    assert None not in transport.routes
    assert set(transport.routes) == set(RECORDS_ROUTES)


def test_splits_the_command_as_a_shell_would() -> None:
    transport = _Server()

    _answer('search-sessions "tile cache" --limit 3', transport)

    assert transport.queries[0]["q"] == "tile cache"


def test_answers_a_command_it_cannot_split() -> None:
    transport = _Server()

    said = _answer('issue title re "unclosed', transport)

    assert said == "The command could not be read: No closing quotation\n"
    assert transport.sent == []


def test_refuses_a_verb_that_acts_beyond_reading_by_name() -> None:
    transport = _Server()

    said = [_answer(command, transport) for command in ("send lead#1 hi", "export")]

    assert said == [
        "trax: send is not available here\n",
        "trax: export is not available here\n",
    ]
    assert transport.sent == []


def test_sends_no_write_under_any_key() -> None:
    """The resolving read goes out; the write it leads to never does."""
    transport = _Server()

    said = _answer("issue 7 status to complete", transport)

    assert transport.routes == [("GET", "/api/inquiries/{kind}/{seq}")]
    assert "is not available to this client" in said


def test_refuses_a_metric_read_its_description_names() -> None:
    """A metric read is a POST, which no Records route lists; the model is told so."""
    transport = _Server()

    said = _answer("experiment 42 metric at loss", transport)

    assert "POST" not in {method for method, _, _ in transport.sent}
    assert "is not available to this client" in said
    assert "`experiment 3 metric ...`" in KNOWOP_DESCRIPTION


def test_reads_no_local_input() -> None:
    transport = _Server()

    said = [
        _answer(command, transport)
        for command in ("issue title to -", "issue title to @x")
    ]

    assert all("reads no local input here" in text for text in said)
    assert transport.sent == []


def test_closes_its_client_after_every_call() -> None:
    transport = _Server()

    _answer("version", transport)
    _answer("version", transport)

    assert transport.closed == 2


def test_opens_no_client_for_a_command_that_sends_nothing() -> None:
    transport = _Server()

    said = [_answer(command, transport) for command in ("help", "send lead#1 hi")]

    assert said[0].startswith("Usage: trax")
    assert (transport.sent, transport.closed) == ([], 0)


def test_lays_a_table_out_to_its_own_width() -> None:
    """160 columns, whatever terminal the agent's process has or lacks."""
    row = {
        "id": _ROW,
        "kind": "Issue",
        "seq": 7,
        "status": "active",
        "title": "Why the tile cache misses " * 20,
        "priority": 20,
        "description": "",
    }
    transport = _Server({"/api/inquiries": [row]})

    said = _answer("issue", transport)

    assert max(len(line) for line in said.splitlines()) == 160


def test_answers_a_client_it_cannot_build_as_the_commands_error() -> None:
    """A key never goes over plain http to another computer; the model reads why."""
    transport = _Server()
    reads = RecordReads(
        "http://records.test",
        api_key="agent-key",
        transport=transport,
    )

    said = reads.read("version")

    assert said.startswith("trax: base_url 'http://records.test' is plain http")
    assert transport.sent == []


def test_runs_every_reading_verb_and_names_each_it_refuses() -> None:
    names = {name for dispatcher in READ_DISPATCHERS for name in dispatcher.names}
    refused = {
        name
        for dispatcher in DISPATCHERS
        if dispatcher not in READ_DISPATCHERS
        for name in dispatcher.names
    }

    assert {"recent", "search-sessions", "next", "blocked", "graph", "board"} <= names
    assert {"cost", "confidence", "authority", "id", "version", "help"} <= names
    assert refused == {"send", "export", "canvas", "doctor", "profile", "env"}
    assert all(f"`{name}`" in KNOWOP_DESCRIPTION for name in sorted(refused))


def test_the_fake_server_lists_a_path_only_with_parameters_its_route_accepts() -> None:
    """A path shaped like a listed template the server routes or rejects otherwise.

    ``proves_belief`` is a route of its own there; a ``{seq}`` that is no number and
    a ``{target_id}`` that is no UUID are refused 422 by the listed route's handler.
    """
    elsewhere = [
        f"/api/inquiries/{_ROW}/proves_belief",
        "/api/inquiries/Issue/seven",
        "/api/web/get/not-a-uuid",
    ]

    assert [_routed("GET", path) for path in elsewhere] == [None, None, None]
    assert _routed("GET", f"/api/sessions/{_ROW}/parts") == (
        "GET",
        "/api/sessions/{session_id}/parts",
    )


def _answer(command: str, transport: httpx2.BaseTransport) -> str:
    """Run ``command`` through fresh reads over ``transport``."""
    return RecordReads(
        "https://records.test",
        api_key="agent-key",
        transport=transport,
    ).read(command)


# The client's matcher takes any segment for a parameter; the server's handler converts
# each and refuses one it cannot, and a route registered first takes its own path.
def _routed(method: str, path: str) -> tuple[str, str] | None:
    """Return the Records route ``path`` is routed to and accepted by, or ``None``."""
    route = listed_route(method, path, RECORDS_ROUTES)
    if route is None:
        return None
    shaped = all(
        _reads(part, value)
        for part, value in zip(route[1].split("/"), path.split("/"), strict=True)
    )
    return route if shaped else None


def _reads(part: str, value: str) -> bool:
    """Whether the route's handler accepts ``value`` as its parameter ``part``."""
    match part:
        case "{kind}":
            return value in VALID_KINDS
        case "{seq}":
            return value.isdigit()
        case "{target_id}" | "{session_id}":
            try:
                _ = UUID(value)
            except ValueError:
                return False
            return True
        case _:
            return True


def _failed(said: str) -> bool:
    """Whether a command's output reports a refusal, a parse error or a crash."""
    return any(mark in said for mark in ("trax:", ": error:", "Traceback"))


class _Server(httpx2.BaseTransport):
    """Answers each Records route from ``_BODIES`` or the bodies given, 404 otherwise."""

    def __init__(self, bodies: dict[str, object] | None = None) -> None:
        self.bodies = {**_BODIES, **(bodies or {})}
        self.sent: list[tuple[str, str, str]] = []
        self.routes: list[tuple[str, str] | None] = []
        self.queries: list[dict[str, str]] = []
        self.closed = 0

    @override
    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        route = _routed(request.method, request.url.path)
        self.sent.append(
            (
                request.method,
                request.url.path,
                request.headers.get("Authorization", ""),
            ),
        )
        self.routes.append(route)
        self.queries.append(dict(parse_qsl(request.url.query.decode())))
        if route is None:
            return httpx2.Response(404, json={"detail": "not found"})
        return httpx2.Response(200, json=self.bodies.get(route[1], []))

    @override
    def close(self) -> None:
        self.closed += 1


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
