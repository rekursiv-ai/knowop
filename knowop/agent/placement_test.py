"""What a `knowop run` session tells the server about where it started, and the child."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from trackinizer.client.errors import ClientError
from trackinizer.trax.canvas import canvas_from_env
from trackinizer.wire.wire_sessions import SessionPlacement

import pytest

from knowop.agent.placement import (
    chat_placement,
    git_ref,
    place_run,
    placement_env,
)
from knowop.agent.worktree_diff import run_git


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from treekle import PlainTree


_CANVAS = UUID(int=5)
_SESSION = UUID(int=1)
_WORKSPACE = UUID(int=2)


class _Client:
    def __init__(self, *, fail: bool = False) -> None:
        self.asked: list[tuple[str, object]] = []
        self._fail = fail

    def post(self, path: str, *, body: object = None) -> dict[str, str]:
        self.asked.append((path, body))
        if self._fail:
            raise ClientError("404 Not Found")
        return {"workspace_id": str(UUID(int=2))}


class _Lister:
    """A client whose session list holds ``sessions``, or that refuses to list."""

    def __init__(self, sessions: list[dict[str, str | bool | None]] | None) -> None:
        self.asked: list[tuple[str, object]] = []
        self._sessions = sessions

    def get(
        self,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
    ) -> PlainTree:
        self.asked.append((path, params))
        if self._sessions is None:
            raise ClientError("404 Not Found")
        listed: dict[str, PlainTree] = {
            "sessions": self._sessions,
            "issues": [],
            "more": False,
        }
        return listed


def _listed(
    session_id: UUID,
    *,
    canvas: UUID | None,
) -> dict[str, str | bool | None]:
    """Return a session as the session list names it, with the fields it is read for."""
    return {
        "id": str(session_id),
        "title": "chat",
        "conversation_id": str(UUID(int=9)),
        "canvas_id": None if canvas is None else str(canvas),
        "workspace_id": None if canvas is None else str(_WORKSPACE),
        "live": True,
    }


def _git(cwd: Path, *argv: str) -> None:
    done = run_git(
        ["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@t", *argv],
        4096,
    )
    assert done.returncode == 0, done.stdout
    assert not done.cut


@pytest.fixture(autouse=True)
def _own_git_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def test_the_canvas_comes_from_trax_canvas_before_its_ssh_twin() -> None:
    both = {"TRAX_CANVAS": str(_CANVAS), "LC_TRAX_CANVAS": str(UUID(int=6))}

    assert canvas_from_env(both) == _CANVAS
    assert canvas_from_env({"LC_TRAX_CANVAS": str(_CANVAS)}) == _CANVAS
    assert canvas_from_env({"TRAX_CANVAS": "not-a-uuid"}) is None
    assert canvas_from_env({}) is None


def test_a_checkout_on_a_branch_names_its_origin_and_branch(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q", "-b", "feature")
    _git(tmp_path, "remote", "add", "origin", "https://github.com/o/r.git")

    ref = git_ref(tmp_path)

    assert ref is not None
    assert (ref.origin, ref.branch) == ("https://github.com/o/r.git", "feature")


def test_no_ref_for_a_directory_that_is_not_a_checkout_on_a_branch(
    tmp_path: Path,
) -> None:
    assert git_ref(tmp_path) is None
    _git(tmp_path, "init", "-q", "-b", "main")
    assert git_ref(tmp_path) is None  # No origin.
    _git(tmp_path, "remote", "add", "origin", "https://github.com/o/r.git")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "x")
    _git(tmp_path, "checkout", "-q", "--detach")
    assert git_ref(tmp_path) is None  # Detached HEAD.


def test_a_run_asks_the_server_with_its_canvas_and_checkout(tmp_path: Path) -> None:
    client = _Client()
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "remote", "add", "origin", "https://github.com/o/r.git")

    placed = place_run(
        client,
        session_id=_SESSION,
        environ={"TRAX_CANVAS": str(_CANVAS)},
        cwd=tmp_path,
    )

    assert placed == SessionPlacement(workspace_id=UUID(int=2))
    assert client.asked == [
        (
            f"/api/sessions/{_SESSION}/place",
            {
                "canvas": str(_CANVAS),
                "git": {"origin": "https://github.com/o/r.git", "branch": "main"},
            },
        ),
    ]


def test_a_failed_ask_warns_and_leaves_the_run_unplaced(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    placed = place_run(
        _Client(fail=True),
        session_id=_SESSION,
        environ={},
        cwd=tmp_path,
    )

    assert placed is None
    assert "running unplaced" in capsys.readouterr().err


def test_a_chats_session_sits_on_the_canvas_its_session_list_names() -> None:
    client = _Lister(
        [
            _listed(UUID(int=3), canvas=UUID(int=4)),
            _listed(_SESSION, canvas=_CANVAS),
        ],
    )

    placed = chat_placement(client, session_id=_SESSION)

    assert placed == SessionPlacement(workspace_id=_WORKSPACE, canvas_id=_CANVAS)
    assert client.asked == [("/api/sessions", {"days": 1})]


def test_a_session_on_no_canvas_or_left_off_the_list_has_no_canvas() -> None:
    unplaced = _Lister([_listed(_SESSION, canvas=None)])
    elsewhere = _Lister([_listed(UUID(int=3), canvas=_CANVAS)])

    assert chat_placement(unplaced, session_id=_SESSION) == (SessionPlacement())
    assert chat_placement(elsewhere, session_id=_SESSION) is None


def test_a_refused_list_warns_and_leaves_the_run_unplaced(
    capsys: pytest.CaptureFixture[str],
) -> None:
    placed = chat_placement(_Lister(None), session_id=_SESSION)

    assert placed is None
    assert "running unplaced" in capsys.readouterr().err


def test_the_child_env_holds_only_what_the_server_placed() -> None:
    assert placement_env(None) == {}
    assert placement_env(SessionPlacement()) == {}
    assert placement_env(
        SessionPlacement(workspace_id=UUID(int=2), canvas_id=_CANVAS),
    ) == {"TRAX_WORKSPACE": str(UUID(int=2)), "TRAX_CANVAS": str(_CANVAS)}


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
