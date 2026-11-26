"""The account's pull request rows on the server: listed, made, watched and written.

Each pull request is one WebResult per account: ``url`` its link, labelled
:data:`LABEL`, and its status what GitHub says: ``active`` while open, ``complete`` once
merged and ``abandoned`` while closed. Each session that watches it is a
``produced_by`` edge from the row, labelled :data:`LABEL` too, its note the label of
the session's agent; a session lets it go with :data:`DISMISSED` on that edge, which
Chat's x writes too. Chat's pill, and the Pull requests page (the server's
``private/pull_requests.py``), draw the rows from these two labels.

``app.knowop.pr`` holds what GitHub last said, and the watch's own state:

- ``error``: why the watch cannot read it, and the fix, in plain words
  (``news.unread_reason``); ``None`` once a read succeeds.
- ``authored``: whether the person opened it, once their GitHub login is known.
- ``autofix``: the person's switch: true, false, or absent for their ``autofix``
  preference, which a pull request they did not open never takes.
- ``fixer``: the session whose agent fixes it; the others are told who does.
- ``away``: each watching session found with no agent running, and since when, as
  ISO time; gone once its agent runs again, or once it lets go.
- ``told``: ``head_sha:name`` of each piece of news already told, for the head now.
- ``automerge_ask``, ``automerge_done``, ``automerge_error``: the person's
  Auto-merge box (``on`` and the ``at`` that names the ask), the ask the watch did,
  or ``default`` for their ``automerge`` preference, and why GitHub refused.
- ``read_at``, ``read_by``: when a watch last wrote the row, and on which host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, NotRequired, Protocol, TypedDict
from uuid import NAMESPACE_URL, UUID, uuid5

from trackinizer.client.errors import ClientError
from trackinizer.wire.filters import Filter
from trackinizer.wire.routes import MAX_LIST_LIMIT
from treekle import from_plain


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from trackinizer.types.inquiries import Inquiry
    from treekle import PlainTree


__all__ = [
    "DISMISSED",
    "LABEL",
    "Profile",
    "PullRequestRows",
    "RowsClient",
    "Stored",
    "Watcher",
]

LABEL: Final = "pull-request"
"""The label of a pull request's row, and of its edge to each session that watches it."""

DISMISSED: Final = "dismissed"
"""The label of the edge from a pull request's row to a session that let it go."""

_WRITES: Final = 3
"""Tries at a row's ``app`` that others keep writing between the read and the write."""


class RowsClient(Protocol):
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
        status: Inquiry.Status | None = None,
        limit: int = ...,
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
        labels: Sequence[str] | None = (),
    ) -> object:
        """Add an edge, or set its note and labels."""
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
    automerge: NotRequired[bool]


class Profile(TypedDict):
    """The person the watch acts for, as ``/api/me/profile`` answers."""

    email: str
    preferences: Preferences


@dataclass(frozen=True, slots=True, kw_only=True)
class Watcher:
    """A session that watches a pull request.

    Attributes:
      session: The session's id.
      label: Its agent's label, as ``LABEL#N`` names the agent.

    """

    session: UUID
    label: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Stored:
    """A pull request's row as last read.

    Attributes:
      row: Its id.
      url: The pull request's link.
      status: Its status.
      app: Its ``app``, empty for none.
      watchers: The sessions that watch it, less those that let it go.

    """

    row: UUID
    url: str
    status: str
    app: dict[str, object]
    watchers: tuple[Watcher, ...] = ()

    @property
    def pr(self) -> dict[str, object]:
        """The row's ``app.knowop.pr``, empty for none."""
        return _pr(self.app)


class PullRequestRows:
    """The pull request rows of one account.

    Args:
      client: The server.
      actor: Who the writes are recorded as.

    """

    def __init__(self, *, client: RowsClient, actor: str) -> None:
        """Write as ``actor``."""
        self._client = client
        self._actor = actor

    def profile(self) -> Profile:
        """Return the person the watch acts for, and their preferences."""
        return from_plain(self._client.get("/api/me/profile"), Profile)

    def listed(self, me: str, *, status: Inquiry.Status) -> list[Stored]:
        """Return the account's pull request rows of ``status``, each with its watchers.

        Args:
          me: The person's email, whose rows they are.
          status: ``active`` for the open ones, ``abandoned`` for the closed.

        Returns:
          rows: Each row, newest first, as the server lists them.

        """
        return [
            _stored(row)
            for row in self._client.list_kind(
                "WebResult",
                status=status,
                limit=MAX_LIST_LIMIT,
                filters=(
                    Filter(field="labels", op="is", value=LABEL),
                    Filter(field="account", op="is", value=me),
                ),
            )
        ]

    def find(self, url: str, *, me: str) -> Stored | None:
        """Return the person's row of a pull request.

        Args:
          url: The pull request's link.
          me: The person's email, whose row it is.

        Returns:
          stored: The row, or ``None`` when there is none.

        """
        rows = self._client.list_kind(
            "WebResult",
            filters=(
                Filter(field="webresult_url", op="is", value=url),
                Filter(field="account", op="is", value=me),
            ),
        )
        return _stored(rows[0]) if rows else None

    def made(self, url: str, *, me: str, title: str) -> UUID:
        """Return the person's row of ``url``, made now if there is none.

        Args:
          url: The pull request's link.
          me: The person's email, whose row it is.
          title: The title a row made now gets.

        Returns:
          row: Its id.

        """
        found = self.find(url, me=me)
        if found is not None:
            return found.row
        # Keyed to the person and the link, so two that make it at once make one.
        return self._client.submit(
            "WebResult",
            {
                "title": title,
                "url": url,
                "labels": [LABEL],
                "idempotency_key": str(uuid5(NAMESPACE_URL, f"{me}\n{url}")),
            },
        )

    def watch(self, row: UUID, *, watcher: Watcher) -> None:
        """Have a session watch a pull request, a dismissal of its own undone.

        Args:
          row: The pull request's row.
          watcher: The session, and its agent's label, which the edge's note keeps.

        """
        # The labels are set whole, which takes a dismissal off with the same write.
        _ = self._client.add_edge(
            row,
            watcher.session,
            "produced_by",
            actor=self._actor,
            note=watcher.label,
            labels=[LABEL],
        )

    def dismiss(self, row: UUID, *, session: UUID) -> None:
        """Let the session go of the pull request, as Chat's x does."""
        # One label, not the edge's whole list, which another writer may have changed.
        _ = self._client.patch(
            f"/api/edges/{row}/produced_by/{session}/labels",
            body={"op": "add", "value": DISMISSED, "actor": self._actor},
        )

    def write_pr[T](
        self,
        stored: Stored,
        change: Callable[[Stored], tuple[dict[str, object], T]],
    ) -> tuple[dict[str, object], T] | None:
        """Write what ``change`` makes of the row's ``app.knowop.pr``.

        A row written since it was read, by another watch or the person's switch, is
        read again and changed again, so what each wrote stays, and what ``change``
        made of the row the write lands on is what it returns.

        Args:
          stored: The row as read.
          change: Returns the ``app.knowop.pr`` to keep, given the row, and what
            else it made of it.

        Returns:
          written: The ``app.knowop.pr`` the row now holds and what ``change`` made of
            it, or ``None`` when the row is gone.

        Raises:
          ClientError: The server refused a write, or others wrote the row first
            :data:`_WRITES` times over.

        """
        tried = 0
        while True:
            pr, made = change(stored)
            if pr == stored.pr:
                return pr, made
            knowop = from_plain(
                stored.app.get("knowop"),
                dict[str, object],
                default=dict[str, object](),
            )
            try:
                self._client.transition_app(
                    stored.row,
                    expected_from=stored.app or None,
                    to=stored.app | {"knowop": knowop | {"pr": pr}},
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
            stored = _stored(rows[0])

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
            # Another watch moved it first, as its read of GitHub said.
            if error.status_code != 409:
                raise


class _Peer(TypedDict):
    id: str
    kind: str
    note: NotRequired[str | None]
    labels: NotRequired[list[str] | None]


class _Row(TypedDict):
    id: str
    url: str | None
    status: str
    app: dict[str, object] | None
    produced_by: NotRequired[list[_Peer]]


def _stored(plain: dict[str, PlainTree]) -> Stored:
    """Return a row as the server listed it."""
    row = from_plain(plain, _Row)
    return Stored(
        row=UUID(row["id"]),
        url=row["url"] or "",
        status=row["status"],
        app=row["app"] or {},
        watchers=tuple(
            Watcher(session=UUID(peer["id"]), label=peer.get("note") or "Agent")
            for peer in row.get("produced_by", ())
            if peer["kind"] == "AgentSession"
            and LABEL in (labels := peer.get("labels") or ())
            and DISMISSED not in labels
        ),
    )


def _pr(app: Mapping[str, object]) -> dict[str, object]:
    """Return ``app.knowop.pr``, empty for none."""
    knowop = from_plain(
        app.get("knowop"),
        dict[str, object],
        default=dict[str, object](),
    )
    return from_plain(knowop.get("pr"), dict[str, object], default=dict[str, object]())
