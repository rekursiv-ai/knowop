"""Look for the pull request of each branch a run's session works on, until one opens.

Claude Code's app binds a session the open pull request of the branch it started on
when the person opened it, and of a branch it pushed whoever opened it, looking again
until one appears. So does this: :meth:`BranchLookup.note` notes the checkout's branch
(the one the run starts on, then each one pushed), and :meth:`BranchLookup.found`
looks for each noted branch's pull request at every pass, the newest :data:`BRANCHES`
of them, never the default branch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from knowop.agent.placement import git_ref
from knowop.agent.pull_requests import (
    GhError,
    read_login,
    read_pull_request,
)


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from knowop.agent.pull_requests import GhRunner
    from knowop.agent.worktree_diff import Runner


__all__ = ["BRANCHES", "BranchLookup"]

BRANCHES: Final = 20
"""Branches looked for at once, as Claude Code's app keeps the last 20 a session wrote."""


class BranchLookup:
    """The noted branches still without a pull request, oldest first.

    Args:
      cwd: The checkout ``gh`` and ``git`` run in.
      gh: Runs ``gh``.
      git: Runs ``git``.
      say: Logs a lasting failure once.

    """

    def __init__(
        self,
        *,
        cwd: Path,
        gh: GhRunner,
        git: Runner,
        say: Callable[[str], None],
    ) -> None:
        """Note no branch yet."""
        self._cwd = cwd
        self._gh = gh
        self._git = git
        self._say = say
        # Each branch, and whether a pull request anyone opened counts.
        self._branches: dict[str, bool] = {}
        self._default: str | None = None
        self._login = ""

    def note(self, *, anyone: bool) -> None:
        """Look for a pull request of the checkout's branch at each pass until one opens.

        Args:
          anyone: Take one whoever opened it, as for a branch the session pushed; else
            only the person's own, as for the branch the run started on.

        """
        ref = git_ref(self._cwd, run=self._git)
        if ref is None or ref.branch == self._default_branch():
            return
        self._branches[ref.branch] = self._branches.pop(ref.branch, False) or anyone
        while len(self._branches) > BRANCHES:
            del self._branches[next(iter(self._branches))]

    def found(self) -> list[str]:
        """Return the open pull requests the noted branches now have, each branch once.

        Returns:
          links: The link of each one found; its branch is looked for no more.

        """
        links: list[str] = []
        for branch, anyone in list(self._branches.items()):
            try:
                pull_request = read_pull_request(self._gh, self._cwd, ref=branch)
            except GhError as error:
                # No pull request yet is what most branches have, at most passes.
                if "no pull requests found" not in str(error):
                    self._say(f"could not look for a pull request of {branch}: {error}")
                continue
            if pull_request.state != "OPEN" or pull_request.head != branch:
                continue
            if not (anyone or self._github_login()):
                continue
            del self._branches[branch]
            if anyone or pull_request.author == self._login:
                links.append(pull_request.url)
        return links

    def _default_branch(self) -> str:
        """Return the branch ``origin`` names as its default, read once; empty for none."""
        if self._default is None:
            out = self._git(
                [
                    *("git", "-C", str(self._cwd), "symbolic-ref", "--short", "-q"),
                    "refs/remotes/origin/HEAD",
                ],
                4096,
            )
            named = out.stdout.decode(errors="replace").strip()
            self._default = named.removeprefix("origin/") if out.returncode == 0 else ""
        return self._default

    def _github_login(self) -> str:
        """Return the login ``gh`` reads with, read once it can be; empty until then."""
        if not self._login:
            try:
                self._login = read_login(self._gh, self._cwd)
            except GhError as error:
                self._say(f"could not read your GitHub login: {error}")
        return self._login
