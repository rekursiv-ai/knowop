"""Tests for looking up the pull request of each branch a session works on.

``gh`` and ``git`` are fakes: :class:`_Gh` answers one pull request per branch as the
test set it, and :class:`_Git` names the checkout's branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import json

from knowop.agent.pull_request_branches import BRANCHES, BranchLookup
from knowop.agent.pull_requests import GhOutput
from knowop.agent.worktree_diff import Output


if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(slots=True, kw_only=True)
class _Gh:
    """One open pull request on each branch but those in ``none``, by ``author``."""

    author: str = "me"
    login: str = "me"
    head: str = ""
    none: str = "no pull requests found for branch"
    error: str = ""
    asked: list[str] = field(default_factory=list)

    def __call__(self, args: Sequence[str], cwd: Path) -> GhOutput:
        del cwd
        if args[0] == "api":
            self.asked.append("login")
            failed = GhOutput(returncode=1, stdout="", stderr="HTTP 401\n")
            return (
                GhOutput(returncode=0, stdout=self.login, stderr="")
                if self.login
                else failed
            )
        branch = args[2]
        self.asked.append(branch)
        if self.error:
            return GhOutput(returncode=1, stdout="", stderr=self.error)
        view = {
            "number": 1,
            "url": f"https://github.com/acme/app/pull/{len(self.asked)}",
            "title": "t",
            "state": "OPEN",
            "author": self.author,
            "draft": False,
            "head": self.head or branch,
            "head_sha": "a" * 40,
            "base": "main",
            "additions": 0,
            "deletions": 0,
            "mergeable": "MERGEABLE",
            "review": "",
            "checks": list[object](),
        }
        return GhOutput(returncode=0, stdout=json.dumps(view), stderr="")


@dataclass(slots=True, kw_only=True)
class _Git:
    branch: str = "b0"

    def __call__(self, argv: Sequence[str], max_bytes: int) -> Output:
        del max_bytes
        said = {
            "HEAD": self.branch,
            "refs/remotes/origin/HEAD": "origin/main",
            "remote.origin.url": "u",
        }[argv[-1]]
        return Output(returncode=0, stdout=said.encode(), cut=False)


def _lookup(gh: _Gh, git: _Git, said: list[str]) -> BranchLookup:
    return BranchLookup(cwd=Path("/work"), gh=gh, git=git, say=said.append)


def test_the_newest_branches_are_kept_and_the_oldest_let_go() -> None:
    gh, git = _Gh(error="no pull requests found for branch"), _Git()
    lookup = _lookup(gh, git, list[str]())
    for n in range(BRANCHES + 1):
        git.branch = f"b{n}"
        lookup.note(anyone=True)
    assert lookup.found() == []
    assert gh.asked == [f"b{n}" for n in range(1, BRANCHES + 1)]


def test_anothers_pull_request_ends_an_own_only_branch_and_login_is_read_once() -> None:
    gh, git = _Gh(author="other"), _Git()
    lookup = _lookup(gh, git, list[str]())
    lookup.note(anyone=False)
    git.branch = "b1"
    lookup.note(anyone=False)
    assert lookup.found() == []
    assert gh.asked == ["b0", "login", "b1"]
    assert lookup.found() == []
    assert gh.asked == ["b0", "login", "b1"]


def test_an_unread_login_keeps_an_own_only_branch_and_says_so_once() -> None:
    gh, git, said = _Gh(login=""), _Git(), list[str]()
    lookup = _lookup(gh, git, said)
    lookup.note(anyone=False)
    assert lookup.found() == []
    assert lookup.found() == []
    assert gh.asked == ["b0", "login", "b0", "login"]
    assert said == ["could not read your GitHub login: HTTP 401"] * 2


def test_a_pull_request_of_another_head_keeps_the_branch_looked_for() -> None:
    gh, git, said = _Gh(head="other"), _Git(), list[str]()
    lookup = _lookup(gh, git, said)
    lookup.note(anyone=True)
    assert lookup.found() == []
    gh.head = ""
    (found,) = lookup.found()
    assert found.startswith("https://github.com/acme/app/pull/")
    assert said == []


def test_a_failure_other_than_no_pull_request_is_said() -> None:
    gh, git, said = _Gh(error="HTTP 502\n"), _Git(), list[str]()
    lookup = _lookup(gh, git, said)
    lookup.note(anyone=True)
    assert lookup.found() == []
    assert said == ["could not look for a pull request of b0: HTTP 502"]


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
