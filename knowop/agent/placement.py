"""How a ``knowop run`` session finds its canvas and tells the child CLI about it.

The server decides the placement (``POST /api/sessions/{id}/place``); this reads the
two things only the client knows, the canvas the run was started from and its git
checkout, and turns the answer into the child's environment. A chat's session is
never placed: it sits on its chat's canvas, which the caller's session list names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
from uuid import UUID

import sys

from trackinizer.client.errors import ClientError
from trackinizer.trax.canvas import canvas_from_env
from trackinizer.wire.wire_sessions import (
    GitRef,
    PlaceSession,
    SessionPlacement,
    session_place_path,
)
from treekle import from_plain

from knowop.agent.program import PROGRAM
from knowop.agent.worktree_diff import run_git


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from treekle import PlainTree

    from knowop.agent.worktree_diff import Runner


class PlacementClient(Protocol):
    """What placing a run needs of a client: ``Client``, or a narrower one."""

    def post(self, path: str, *, body: object = None) -> PlainTree:
        """POST ``body`` to ``path``; return the decoded answer."""
        ...


class SessionLister(Protocol):
    """What finding a chat's canvas needs of a client: ``Client``, or a narrower one."""

    def get(
        self,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
    ) -> PlainTree:
        """GET ``path`` with ``params``; return the decoded answer."""
        ...


def git_ref(cwd: Path, *, run: Runner = run_git) -> GitRef | None:
    """Return the checkout's origin URL and branch.

    Args:
      cwd: The run's working directory.
      run: Runs each git command; the real subprocess runner by default.

    Returns:
      ref: ``None`` when ``cwd`` is not in a git work tree, has no ``origin`` or is on
        a detached HEAD.

    """
    where = ["git", "-C", str(cwd)]
    branch = run([*where, "symbolic-ref", "--short", "-q", "HEAD"], 4096)
    origin = run([*where, "config", "--get", "remote.origin.url"], 4096)
    if branch.returncode != 0 or origin.returncode != 0:
        return None
    name = branch.stdout.decode(errors="replace").strip()
    url = origin.stdout.decode(errors="replace").strip()
    return GitRef(origin=url, branch=name) if name and url else None


def place_run(
    client: PlacementClient,
    *,
    session_id: UUID,
    environ: Mapping[str, str],
    cwd: Path,
) -> SessionPlacement | None:
    """Place the run's session; on a refusal warn once and return ``None``.

    Args:
      client: The run's client.
      session_id: The session the run opened.
      environ: The process environment.
      cwd: The run's working directory.

    Returns:
      placement: The server's answer, or ``None`` when it could not be had.

    """
    body = PlaceSession(canvas=canvas_from_env(environ), git=git_ref(cwd))
    try:
        answer = client.post(
            session_place_path(session_id),
            body=body.model_dump(mode="json"),
        )
        return SessionPlacement.model_validate(answer)
    except (ClientError, ValueError) as error:
        sys.stderr.write(
            f"[{PROGRAM}] could not place this session ({error}); running unplaced\n",
        )
        return None


# The path is written out: ``SESSIONS_PATH`` is newer than trackinizer 0.1.10, the
# release knowop's wheel requires, so naming it would fail the import on such an
# install. Activity over one day suffices: the list keeps every open session, and the
# run's attach has just reopened this one. A listed session names its canvas and that
# canvas's workspace by a placement's field names, beside fields a placement ignores.
def chat_placement(
    client: SessionLister,
    *,
    session_id: UUID,
) -> SessionPlacement | None:
    """Return the canvas a chat's session sits on, as the session list names it.

    Args:
      client: The run's client.
      session_id: The chat's session, which the run attached to.

    Returns:
      placement: The canvas and its workspace, each ``None`` for a session on no
        canvas; ``None`` when the list leaves the session out, or on a refusal, which
        it warns of once.

    """
    try:
        answer = from_plain(
            client.get("/api/sessions", params={"days": 1}),
            dict[str, object],
        )
        rows = from_plain(answer.get("sessions"), list[dict[str, object]], default=[])
        return next(
            (
                SessionPlacement.model_validate(row)
                for row in rows
                if from_plain(row.get("id"), UUID) == session_id
            ),
            None,
        )
    except (ClientError, ValueError) as error:
        sys.stderr.write(
            f"[{PROGRAM}] could not find this chat's canvas ({error}); "
            "running unplaced\n",
        )
        return None


def placement_env(placement: SessionPlacement | None) -> dict[str, str]:
    """Return the variables that tell the child CLI where it was placed.

    Args:
      placement: The server's answer, or ``None`` for a run that was not placed.

    Returns:
      env: ``TRAX_WORKSPACE`` and ``TRAX_CANVAS`` for each id the server named.

    """
    env: dict[str, str] = {}
    if placement is None:
        return env
    if placement.workspace_id is not None:
        env["TRAX_WORKSPACE"] = str(placement.workspace_id)
    if placement.canvas_id is not None:
        env["TRAX_CANVAS"] = str(placement.canvas_id)
    return env
