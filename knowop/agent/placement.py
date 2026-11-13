"""How a ``knowop run`` session finds its canvas: the server places it at start.

The server decides the placement (``POST /api/sessions/{id}/place``); this reads the
two things only the client knows, the canvas the run was started from and its git
checkout, and turns the answer into the variables the run's tools read it from. Git is
asked so it changes nothing: no optional lock, no prompt, bounded in time and output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

import os
import subprocess
import sys

from trackinizer.client.errors import ClientError
from trackinizer.trax.canvas import canvas_from_env
from trackinizer.wire.wire_sessions import (
    GitRef,
    PlaceSession,
    SessionPlacement,
    session_place_path,
)

from knowop.agent.program import PROGRAM


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path
    from uuid import UUID

    from treekle import PlainTree


__all__ = [
    "GIT_SEC",
    "Output",
    "PlacementClient",
    "Runner",
    "git_ref",
    "place_run",
    "placement_env",
    "run_git",
]


GIT_SEC: Final = 20.0
"""Seconds one git command may run before it is killed."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Output:
    """What one git command printed, and how it ended.

    Attributes:
      returncode: Its exit status; ``-1`` when it ran out of time or could not
        start.
      stdout: What it printed, at most the byte bound it was read to.
      cut: Whether the output reached that bound and was truncated.

    """

    returncode: int
    stdout: bytes
    cut: bool


type Runner = Callable[[Sequence[str], int], Output]
"""Runs one git command ``argv``, its output kept to ``max_bytes``."""


class PlacementClient(Protocol):
    """What placing a run needs of a client: ``Client``, or a narrower one."""

    def post(self, path: str, *, body: object = None) -> PlainTree:
        """POST ``body`` to ``path``; return the decoded answer."""
        ...


def run_git(argv: Sequence[str], max_bytes: int) -> Output:
    """Run one git command, read-only, bounded in time and output.

    Args:
      argv: The full command, ``git`` first.
      max_bytes: Where its output is cut.

    Returns:
      output: What it printed and how it ended.

    """
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"}
    try:
        done = subprocess.run(  # noqa: S603 -- fixed argv, no shell.
            list(argv),
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=GIT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, bytes) else b""
        return Output(
            returncode=-1,
            stdout=stdout[:max_bytes],
            cut=len(stdout) > max_bytes,
        )
    except OSError:
        return Output(returncode=-1, stdout=b"", cut=False)
    return Output(
        returncode=done.returncode,
        stdout=done.stdout[:max_bytes],
        cut=len(done.stdout) > max_bytes,
    )


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


def placement_env(placement: SessionPlacement | None) -> dict[str, str]:
    """Return the variables that tell a run's tools where its session was placed.

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
