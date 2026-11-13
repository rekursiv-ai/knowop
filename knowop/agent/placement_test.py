"""What a `knowop run` session tells the server about where it started."""

from __future__ import annotations

from functools import partial
from subprocess import CompletedProcess, TimeoutExpired
from typing import TYPE_CHECKING
from unittest.mock import Mock, call
from uuid import UUID

import os
import subprocess

from trackinizer.trax.canvas import canvas_from_env
from trackinizer.wire.wire_sessions import GitRef, SessionPlacement

import pytest

from knowop.agent import placement
from knowop.agent.conftest import _CANVAS, _SESSION, _Client
from knowop.agent.placement import (
    Output,
    git_ref,
    place_run,
    placement_env,
    run_git,
)


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def git_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Mock:
    """Keep placement on explicit Git replies, isolated from the launching checkout."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    read = Mock(return_value=Output(returncode=128, stdout=b"", cut=False))
    monkeypatch.setattr(placement, "git_ref", partial(git_ref, run=read))
    return read


def test_the_canvas_comes_from_trax_canvas_before_its_ssh_twin() -> None:
    both = {"TRAX_CANVAS": str(_CANVAS), "LC_TRAX_CANVAS": str(UUID(int=6))}

    assert canvas_from_env(both) == _CANVAS
    assert canvas_from_env({"LC_TRAX_CANVAS": str(_CANVAS)}) == _CANVAS
    assert canvas_from_env({"TRAX_CANVAS": "not-a-uuid"}) is None
    assert canvas_from_env({}) is None


def test_a_checkout_on_a_branch_names_its_origin_and_branch(
    tmp_path: Path,
    git_commands: Mock,
) -> None:
    git_commands.side_effect = [
        Output(returncode=0, stdout=b"feature\n", cut=False),
        Output(returncode=0, stdout=b"https://github.com/o/r.git\n", cut=False),
    ]

    ref = git_ref(tmp_path, run=git_commands)

    assert ref is not None
    assert (ref.origin, ref.branch) == ("https://github.com/o/r.git", "feature")
    assert git_commands.call_args_list == [
        call(
            ["git", "-C", str(tmp_path), "symbolic-ref", "--short", "-q", "HEAD"],
            4096,
        ),
        call(
            ["git", "-C", str(tmp_path), "config", "--get", "remote.origin.url"],
            4096,
        ),
    ]


def test_no_ref_for_a_directory_that_is_not_a_checkout_on_a_branch(
    tmp_path: Path,
    git_commands: Mock,
) -> None:
    # A failed command's output is never read, whatever it printed.
    git_commands.side_effect = [
        Output(returncode=128, stdout=b"", cut=False),
        Output(returncode=128, stdout=b"", cut=False),
        Output(returncode=0, stdout=b"main\n", cut=False),
        Output(returncode=1, stdout=b"https://github.com/o/r.git\n", cut=False),
        Output(returncode=1, stdout=b"main\n", cut=False),
        Output(returncode=0, stdout=b"https://github.com/o/r.git\n", cut=False),
        Output(returncode=0, stdout=b"\n", cut=False),
        Output(returncode=0, stdout=b"https://github.com/o/r.git\n", cut=False),
    ]
    assert git_ref(tmp_path, run=git_commands) is None
    assert git_ref(tmp_path, run=git_commands) is None  # No origin.
    assert git_ref(tmp_path, run=git_commands) is None  # Detached HEAD.
    assert git_ref(tmp_path, run=git_commands) is None  # A branch with no name.
    assert (
        git_commands.call_args_list
        == [
            call(
                ["git", "-C", str(tmp_path), "symbolic-ref", "--short", "-q", "HEAD"],
                4096,
            ),
            call(
                ["git", "-C", str(tmp_path), "config", "--get", "remote.origin.url"],
                4096,
            ),
        ]
        * 4
    )


def test_a_name_git_printed_in_another_encoding_is_read_with_stand_ins(
    tmp_path: Path,
    git_commands: Mock,
) -> None:
    git_commands.side_effect = [
        Output(returncode=0, stdout=b"na\xefve\n", cut=False),
        Output(returncode=0, stdout=b"https://h\xe9.example/r.git\n", cut=False),
    ]

    ref = git_ref(tmp_path, run=git_commands)

    assert ref == GitRef(origin="https://h\ufffd.example/r.git", branch="na\ufffdve")


def test_a_run_asks_the_server_with_its_canvas_and_checkout(
    tmp_path: Path,
    git_commands: Mock,
) -> None:
    client = _Client()
    git_commands.side_effect = [
        Output(returncode=0, stdout=b"main\n", cut=False),
        Output(returncode=0, stdout=b"https://github.com/o/r.git\n", cut=False),
    ]

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
    assert git_commands.call_args_list == [
        call(
            ["git", "-C", str(tmp_path), "symbolic-ref", "--short", "-q", "HEAD"],
            4096,
        ),
        call(
            ["git", "-C", str(tmp_path), "config", "--get", "remote.origin.url"],
            4096,
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


def test_a_placement_names_only_the_ids_the_server_gave() -> None:
    workspace, canvas = UUID(int=2), UUID(int=3)

    assert placement_env(None) == {}
    assert placement_env(SessionPlacement()) == {}
    assert placement_env(SessionPlacement(workspace_id=workspace)) == {
        "TRAX_WORKSPACE": str(workspace),
    }
    assert placement_env(
        SessionPlacement(workspace_id=workspace, canvas_id=canvas),
    ) == {"TRAX_WORKSPACE": str(workspace), "TRAX_CANVAS": str(canvas)}


class TestRunGit:
    """The subprocess runner's output bounds, missing binary, and deadline result."""

    def test_reads_output_up_to_the_byte_cap(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Output is kept to ``max_bytes`` and the cut is reported."""
        execute = Mock(return_value=CompletedProcess(["printf", "hello"], 0, b"hello"))
        monkeypatch.setattr(subprocess, "run", execute)

        done = run_git(["printf", "hello"], 100)
        assert done == Output(returncode=0, stdout=b"hello", cut=False)
        # Exactly the bound is whole: nothing was left out.
        assert run_git(["printf", "hello"], 5) == done
        cut = run_git(["printf", "hello"], 2)
        assert cut == Output(returncode=0, stdout=b"he", cut=True)
        assert (
            execute.call_args_list
            == [
                call(
                    ["printf", "hello"],
                    env={
                        **os.environ,
                        "GIT_OPTIONAL_LOCKS": "0",
                        "GIT_TERMINAL_PROMPT": "0",
                    },
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=20.0,
                    check=False,
                ),
            ]
            * 3
        )

    def test_a_missing_binary_is_not_an_exception(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A git that cannot start ends as ``-1``, not a crash."""
        execute = Mock(side_effect=FileNotFoundError("trax-run-no-such-binary-xyz"))
        monkeypatch.setattr(subprocess, "run", execute)

        done = run_git(["trax-run-no-such-binary-xyz"], 100)
        assert done == Output(returncode=-1, stdout=b"", cut=False)
        assert execute.call_args.args == (["trax-run-no-such-binary-xyz"],)

    def test_a_git_past_its_time_ends_as_minus_one_with_what_it_printed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(placement, "GIT_SEC", 0.02)
        argv = ["sh", "-c", "printf partial; exec sleep 30"]
        execute = Mock(side_effect=TimeoutExpired(argv, 0.02, output=b"partial"))
        monkeypatch.setattr(subprocess, "run", execute)

        done = run_git(argv, 4)

        assert done == Output(returncode=-1, stdout=b"part", cut=True)
        execute.assert_called_once_with(
            argv,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=0.02,
            check=False,
        )

    @pytest.mark.parametrize(
        ("printed", "kept"),
        [(b"part", b"part"), (None, b"")],
        ids=["exactly-the-bound", "nothing-printed"],
    )
    def test_a_git_past_its_time_keeps_what_it_printed_whole(
        self,
        monkeypatch: pytest.MonkeyPatch,
        printed: bytes | None,
        kept: bytes,
    ) -> None:
        argv = ["git", "status"]
        monkeypatch.setattr(
            subprocess,
            "run",
            Mock(side_effect=TimeoutExpired(argv, 20.0, output=printed)),
        )

        done = run_git(argv, 4)

        assert done == Output(returncode=-1, stdout=kept, cut=False)


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
