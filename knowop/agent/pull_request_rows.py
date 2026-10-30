"""A run session's pull request rows on the server: read, made, linked and written.

Each pull request is one WebResult per account: ``url`` its link, labelled
:data:`LABEL`, produced by each session that opened, pushed or watched it, and its
status what GitHub says: ``active`` while open, ``complete`` once merged and
``abandoned`` while closed. ``app.knowop.pr`` holds what GitHub last said (with
``error`` when the last read failed), the person's switch (``autofix``: true, false,
or absent for their ``autofix`` preference), the session that Auto-fixes it
(``fixer``) and the failures that session's wakes already named (``told``), so a run
that starts again does not wake twice.

The edge from the row to each session is labelled :data:`LABEL` too, so a session's
pull requests are found among what it produced without reading every row. A session
lets one go with :data:`DISMISSED` on that edge, which Chat's × writes too. Chat
draws the pill over a session's transcript from these rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, NotRequired, Protocol, TypedDict
from uuid import NAMESPACE_URL, UUID, uuid5

from trackinizer.client.errors import ClientError
from trackinizer.wire.filters import Filter
from treekle import from_plain


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from trackinizer.types.inquiries import Inquiry
    from treekle import PlainTree

    from knowop.agent.pull_requests import PullRequest


__all__ = [
    "DISMISSED",
    "LABEL",
    "Profile",
    "PullRequestClient",
    "PullRequestRows",
    "SessionRow",
    "Stored",
]

LABEL: Final = "pull-request"
"""The label of a pull request's row, and of its edge to each session it is linked to."""

_WRITES: Final = 3
"""Tries at a row's ``app`` that others keep writing between the read and the write."""

DISMISSED: Final = "dismissed"
"""The label of the edge from a pull request's row to a session that let it go."""


class PullRequestClient(Protocol):
    """What the rows ask of the server: ``Client``, or a narrower one."""

    def get(
        self,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
    ) -> PlainTree:
        """GET ``path``; return the decoded answer."""
        ...

    def patch(self, path: str, *, body: object = None) -> PlainTree:
        """PATCH ``body`` to ``path``; return the decoded answer."""
        ...

    def list_kind(
        self,
        kind: Inquiry.InquiryKind,
        *,
        filters: Sequence[Filter] = (),
    ) -> list[dict[str, PlainTree]]:
        """Return the rows of ``kind`` that ``filters`` match."""
        ...

    def submit(self, kind: Inquiry.InquiryKind, body: Mapping[str, object]) -> UUID:
        """Create a row; return its id."""
        ...

    def add_edge(
        self,
        from_id: UUID,
        to_id: UUID,
        edge_kind: str,
        *,
        actor: Inquiry.Actor,
        note: str = "",
    ) -> object:
        """Add an edge; a repeat changes nothing."""
        ...

    def transition_app(
        self,
        target_id: UUID,
        *,
        expected_from: dict[str, object] | None,
        to: dict[str, object] | None,
        actor: Inquiry.Actor,
    ) -> None:
        """Compare-and-set a row's ``app``."""
        ...

    def transition_status(
        self,
        target_id: UUID,
        *,
        expected_from: str,
        to: str,
        actor: Inquiry.Actor,
    ) -> None:
        """Compare-and-set a row's ``status``."""
        ...


class Preferences(TypedDict):
    """The preferences the watch reads."""

    autofix: NotRequired[bool]


class Profile(TypedDict):
    """The person the run acts for, as ``/api/me/profile`` answers."""

    email: str
    preferences: Preferences


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionRow:
    """A pull request row the session's backlinks name.

    Attributes:
      url: Its link.
      status: The row's status.
      dismissed: Whether the session let it go.

    """

    url: str
    status: str
    dismissed: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class Stored:
    """A pull request's row as last read.

    Attributes:
      row: Its id.
      app: Its ``app``, empty for none.
      status: Its status.

    """

    row: UUID
    app: dict[str, object]
    status: str


class PullRequestRows:
    """The rows of one run's session, and what they are linked to.

    Args:
      client: The server the session syncs to.
      actor: Who the writes are recorded as.

    """

    def __init__(self, *, client: PullRequestClient, actor: str) -> None:
        """Know no row yet."""
        self._client = client
        self._actor = actor
        self._linked: dict[str, UUID] = {}
        self._links: dict[UUID, str | None] = {}

    def profile(self) -> Profile:
        """Return the person the run acts for, and their preferences."""
        return from_plain(self._client.get("/api/me/profile"), Profile)

    def session_rows(self, session: UUID) -> list[SessionRow]:
        """Return the pull request rows linked to ``session``, each row's link read once.

        Args:
          session: The session's id.

        Returns:
          rows: Each pull request row its backlinks name, as linked now.

        """
        view = from_plain(self._client.get(f"/api/web/get/{session}"), _Detail)
        rows: list[SessionRow] = []
        for peer in view["backlinks"].get("produced_by", ()):
            if LABEL not in peer.get("labels", ()):
                continue
            url = self._link_of(peer)
            if url is None:
                continue
            self._linked[url] = UUID(peer["id"])
            rows.append(
                SessionRow(
                    url=url,
                    status=peer.get("status", ""),
                    dismissed=DISMISSED in peer.get("labels", ()),
                ),
            )
        return rows

    def linked(self, url: str) -> UUID | None:
        """Return the row of ``url`` linked to the session, if one is known."""
        return self._linked.get(url)

    def row(
        self,
        url: str,
        *,
        session: UUID,
        me: str,
        pull_request: PullRequest | None,
    ) -> Stored:
        """Return a pull request's row as stored, made and linked to the session once.

        Args:
          url: Its link.
          session: The session's id.
          me: The person's email, whose row it is.
          pull_request: What GitHub said of it, which titles a row made now.

        Returns:
          stored: Its row, ``app`` and status.

        """
        rows = self._client.list_kind(
            "WebResult",
            filters=(
                Filter(field="webresult_url", op="is", value=url),
                Filter(field="account", op="is", value=me),
            ),
        )
        if rows:
            read = from_plain(rows[0], _Row)
            stored = Stored(
                row=UUID(read["id"]),
                app=read["app"] or {},
                status=read["status"],
            )
        else:
            title = (
                url
                if pull_request is None
                else f"{pull_request.repo}#{pull_request.number} {pull_request.title}"
            )
            # Keyed to the person and the link, so two runs that find it at once make one.
            made = self._client.submit(
                "WebResult",
                {
                    "title": title,
                    "url": url,
                    "labels": [LABEL],
                    "idempotency_key": str(uuid5(NAMESPACE_URL, f"{me}\n{url}")),
                },
            )
            stored = Stored(row=made, app={}, status="active")
        if self._linked.get(url) != stored.row:
            _ = self._client.add_edge(
                stored.row,
                session,
                "produced_by",
                actor=self._actor,
            )
            self._label(stored.row, session=session, label=LABEL, on=True)
            self._linked[url] = stored.row
        return stored

    def write_pr[T](
        self,
        stored: Stored,
        change: Callable[[dict[str, object]], tuple[dict[str, object], T]],
    ) -> tuple[dict[str, object], T] | None:
        """Write what ``change`` makes of the row's ``app.knowop.pr``.

        A row written since it was read, by another session's run or the person's
        switch, is read again and changed again, so what each wrote stays.

        Args:
          stored: The row as read.
          change: Returns the ``app.knowop.pr`` to keep, given the one the row holds,
            and what else it made of it.

        Returns:
          written: The ``app.knowop.pr`` the row now holds and what ``change`` made of
            it, or ``None`` when the row is gone.

        Raises:
          ClientError: The server refused a write, or others wrote the row first
            :data:`_WRITES` times over.

        """
        app, tried = stored.app, 0
        while True:
            knowop = from_plain(app.get("knowop"), dict[str, object], default={})
            kept = from_plain(knowop.get("pr"), dict[str, object], default={})
            pr, made = change(kept)
            if pr == kept:
                return pr, made
            try:
                self._client.transition_app(
                    stored.row,
                    expected_from=app or None,
                    to=app | {"knowop": knowop | {"pr": pr}},
                    actor=self._actor,
                )
            except ClientError as error:
                tried += 1
                if error.status_code != 409 or tried == _WRITES:
                    raise
            else:
                return pr, made
            rows = self._client.list_kind(
                "WebResult",
                filters=(Filter(field="id", op="is", value=str(stored.row)),),
            )
            if not rows:
                return None
            app = from_plain(rows[0], _Row)["app"] or {}

    def move(self, stored: Stored, *, status: str) -> None:
        """Move a row to ``status``, as GitHub says the pull request is now.

        Args:
          stored: The row as read.
          status: ``active`` while it is open, ``complete`` once merged and
            ``abandoned`` while closed.

        """
        if stored.status == status:
            return
        try:
            self._client.transition_status(
                stored.row,
                expected_from=stored.status,
                to=status,
                actor=self._actor,
            )
        except ClientError as error:
            # Another session's run moved it first, as its read of GitHub said.
            if error.status_code != 409:
                raise

    def dismiss(self, row: UUID, *, session: UUID, dismissed: bool) -> None:
        """Put the session's dismissal on the edge from ``row``, or take it off.

        Args:
          row: The pull request's row.
          session: The session it is linked to.
          dismissed: Whether the session lets it go.

        """
        self._label(row, session=session, label=DISMISSED, on=dismissed)

    def let_go(self, row: UUID, *, session: UUID) -> bool:
        """Return whether ``session`` let go of the pull request at ``row``, or never had it.

        Args:
          row: The pull request's row.
          session: A session that watched it.

        Returns:
          gone: Whether the row's edge to ``session`` is dismissed, or not there.

        """
        view = from_plain(self._client.get(f"/api/web/get/{row}"), _Detail)
        return not any(
            peer["id"] == str(session) and DISMISSED not in peer.get("labels", ())
            for peer in view["edges"].get("produced_by", ())
        )

    def _label(self, row: UUID, *, session: UUID, label: str, on: bool) -> None:
        """Put ``label`` on the edge from ``row`` to ``session``, or take it off."""
        # One label, not the edge's whole list, which another writer may have changed:
        # Chat's × writes the same way.
        _ = self._client.patch(
            f"/api/edges/{row}/produced_by/{session}/labels",
            body={"op": "add" if on else "sub", "value": label, "actor": self._actor},
        )

    def _link_of(self, peer: _Peer) -> str | None:
        """Return the link of a pull request's row, read once; ``None`` for no row."""
        row = UUID(peer["id"])
        if row not in self._links:
            rows = self._client.list_kind(
                "WebResult",
                filters=(Filter(field="id", op="is", value=peer["id"]),),
            )
            self._links[row] = next(
                (from_plain(each, _Row)["url"] for each in rows),
                None,
            )
        return self._links[row]


class _Peer(TypedDict):
    id: str
    status: NotRequired[str]
    labels: NotRequired[list[str]]
    """The edge's labels: :data:`LABEL` for a pull request's, and :data:`DISMISSED`
    too once the session let it go."""


class _Links(TypedDict):
    """A row's edges of one direction, by kind."""

    produced_by: NotRequired[list[_Peer]]


class _Detail(TypedDict):
    backlinks: _Links
    """The edges to the row: a session's, from what it produced."""

    edges: _Links
    """The edges from the row: a pull request's, to the sessions that watch it."""


class _Row(TypedDict):
    id: str
    url: str | None
    labels: list[str] | None
    status: str
    app: dict[str, object] | None
