"""Read a pull request through ``gh``, and turn GitHub's auto-merge on or off.

``gh`` runs on the host that serves the watch, with that host's own login: the one
credential the watch reads with. A read that fails says why (:class:`GhError`), and the
watch keeps that on the pull request's row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TypedDict

import os
import re
import shutil
import subprocess

from treekle import json_loads


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence


__all__ = [
    "GhError",
    "GhOutput",
    "GhRunner",
    "PullRequest",
    "link",
    "read_pull_request",
    "run_gh",
    "set_auto_merge",
]


GH_SEC: Final = 30.0
"""Seconds one ``gh`` call may run before it is killed."""

_FIELDS: Final = ",".join(
    (
        "number",
        "url",
        "title",
        "state",
        "author",
        "isDraft",
        "headRefName",
        "headRefOid",
        "baseRefName",
        "additions",
        "deletions",
        "mergeable",
        "reviewDecision",
        "autoMergeRequest",
        "statusCheckRollup",
        "updatedAt",
    ),
)

# gh's own camelCase names, renamed in gh's jq to the names :class:`_View` reads.
_JQ: Final = (
    '{number, url, title, state, author: (.author.login // ""), draft: .isDraft, '
    'head: .headRefName, updated: (.updatedAt // ""), '
    "head_sha: .headRefOid, base: .baseRefName, additions, deletions, mergeable, "
    'review: (.reviewDecision // ""), auto_merge: (.autoMergeRequest != null), '
    "checks: [(.statusCheckRollup // [])[] | "
    '{name: (.name // .context // ""), workflow: (.workflowName // ""), '
    'commit_status: (.__typename == "StatusContext"), status: (.status // ""), '
    'conclusion: (.conclusion // ""), state: (.state // "")}]}'
)

_LINK: Final = re.compile(
    r"https://github\.com/(?P<repo>[A-Za-z0-9._-]+/[A-Za-z0-9._-]+)"
    r"/pull/(?P<number>[0-9]+)",
)

# A conclusion GitHub gives a check that went wrong in its own run. ``CANCELLED`` is not
# one: a newer push cancels the runs of the head it replaced.
_FAILED: Final = frozenset(
    {"FAILURE", "TIMED_OUT", "ACTION_REQUIRED", "STARTUP_FAILURE", "ERROR"},
)

_RUNNING: Final = frozenset({"PENDING", "EXPECTED"})
"""A commit status's states while it has no verdict yet."""


class GhError(Exception):
    """``gh`` could not read a pull request; the message is its first line of why."""


@dataclass(frozen=True, slots=True, kw_only=True)
class GhOutput:
    """What one ``gh`` call printed, and how it ended.

    Attributes:
      returncode: Its exit status; ``-1`` when it ran out of time or could not start.
      stdout: What it printed.
      stderr: What it said went wrong.

    """

    returncode: int
    stdout: str
    stderr: str


type GhRunner = Callable[[Sequence[str]], GhOutput]
"""Runs ``gh`` with ``args``."""


def run_gh(args: Sequence[str]) -> GhOutput:
    """Run the ``gh`` on the PATH, never asking at a prompt.

    Args:
      args: Its arguments, ``gh`` itself not among them.

    Returns:
      output: What it printed and how it ended.

    """
    gh = shutil.which("gh")
    if gh is None:
        return GhOutput(
            returncode=-1,
            stdout="",
            stderr="gh is not on this host's PATH",
        )
    env = {**os.environ, "GH_PROMPT_DISABLED": "1", "GIT_TERMINAL_PROMPT": "0"}
    # Every call names the pull request by its link, so no checkout's remote is read,
    # and any working directory serves.
    try:
        done = subprocess.run(  # noqa: S603 -- fixed argv, no shell.
            [gh, *args],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=GH_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return GhOutput(returncode=-1, stdout="", stderr=f"gh took over {GH_SEC:.0f}s")
    except OSError as error:
        return GhOutput(returncode=-1, stdout="", stderr=f"gh did not start: {error}")
    return GhOutput(returncode=done.returncode, stdout=done.stdout, stderr=done.stderr)


@dataclass(frozen=True, slots=True, kw_only=True)
class PullRequest:
    """A pull request as GitHub last said it was.

    Attributes:
      url: Its link, which names it.
      repo: ``owner/name``.
      number: Its number.
      title: Its title.
      state: ``OPEN``, ``MERGED`` or ``CLOSED``.
      author: The GitHub login that opened it; empty when GitHub names none.
      draft: Whether it is a draft.
      head: Its branch.
      head_sha: Its head commit.
      base: The branch it merges into.
      additions: Lines it adds.
      deletions: Lines it removes.
      mergeable: ``MERGEABLE``, ``CONFLICTING`` or ``UNKNOWN``.
      review: Its review decision; empty for none.
      updated: When it last changed, as GitHub stamps it (ISO 8601); empty for none.
      auto_merge: Whether GitHub merges it once its checks and reviews allow.
      passed: Checks of the head that passed or need nothing.
      failed: Names of the head's checks that failed.
      pending: Names of the head's checks still running.
      waiting: Names of the failed checks whose workflow still runs another check.
      total: All of the head's checks.

    """

    url: str
    repo: str
    number: int
    title: str
    state: str
    author: str
    draft: bool
    head: str
    head_sha: str
    base: str
    additions: int
    deletions: int
    mergeable: str
    review: str
    updated: str
    auto_merge: bool
    passed: int
    failed: tuple[str, ...]
    pending: tuple[str, ...]
    waiting: tuple[str, ...]
    total: int

    @property
    def passing(self) -> bool:
        """Whether the head has checks and every one passed."""
        return self.total > 0 and self.passed == self.total

    @property
    def name(self) -> str:
        """``owner/name#N``, as a line names it."""
        return f"{self.repo}#{self.number}"

    def app(self) -> dict[str, object]:
        """Return what a pull request's row keeps of it.

        Returns:
          pr: The fields of ``app.knowop.pr`` that GitHub's word fills, which Chat's
            pill and the Pull requests page draw (``ui/src/visuals/pullRequests.ts``).

        """
        return {
            "url": self.url,
            "repo": self.repo,
            "number": self.number,
            "title": self.title,
            "state": self.state,
            "draft": self.draft,
            "head": self.head,
            "head_sha": self.head_sha,
            "base": self.base,
            "additions": self.additions,
            "deletions": self.deletions,
            "mergeable": self.mergeable,
            "review": self.review,
            "author": self.author,
            "updated": self.updated,
            "auto_merge": self.auto_merge,
            "checks": {
                "passed": self.passed,
                "failed": list(self.failed),
                "pending": list(self.pending),
                "total": self.total,
            },
        }


def link(text: str) -> str:
    """Return the pull request ``text`` names by its link, as GitHub spells it.

    Args:
      text: ``https://github.com/OWNER/REPO/pull/N``, with anything after the number.

    Returns:
      url: The link up to the number.

    Raises:
      ValueError: ``text`` is no GitHub pull request's link.

    """
    found = _LINK.match(text.strip())
    if found is None:
        raise ValueError(
            f"{text!r} is no GitHub pull request: give its link, "
            "https://github.com/OWNER/REPO/pull/N",
        )
    return found[0]


def read_pull_request(run: GhRunner, url: str) -> PullRequest:
    """Read one pull request through ``gh pr view``.

    Args:
      run: Runs ``gh``.
      url: The pull request's link.

    Returns:
      pull_request: What GitHub says of it now.

    Raises:
      GhError: ``gh`` failed, found none, or printed what is no pull request.

    """
    out = run(["pr", "view", url, "--json", _FIELDS, "--jq", _JQ])
    if out.returncode != 0:
        raise GhError(_why(out))
    try:
        view = json_loads(out.stdout, _View)
    # Not JSON (``JSONDecodeError``), or JSON of another shape (``ReadError``).
    except ValueError as error:
        raise GhError(f"gh printed no pull request: {error}"[:300]) from error
    found = _LINK.match(view["url"])
    if found is None:
        raise GhError(f"gh named no GitHub pull request: {view['url'][:200]}")
    verdicts = [(_verdict(check), check) for check in view["checks"]]
    failed = [check for verdict, check in verdicts if verdict == "failed"]
    pending = [check for verdict, check in verdicts if verdict == "pending"]
    running = {check["workflow"] for check in pending if check["workflow"]}
    return PullRequest(
        url=view["url"],
        repo=found["repo"],
        number=view["number"],
        title=view["title"],
        state=view["state"],
        author=view["author"],
        draft=view["draft"],
        head=view["head"],
        head_sha=view["head_sha"],
        base=view["base"],
        additions=view["additions"],
        deletions=view["deletions"],
        mergeable=view["mergeable"],
        review=view["review"],
        updated=view["updated"],
        auto_merge=view["auto_merge"],
        passed=sum(verdict == "passed" for verdict, _ in verdicts),
        failed=_names(failed),
        pending=_names(pending),
        waiting=_names(check for check in failed if check["workflow"] in running),
        total=len(view["checks"]),
    )


# GitHub's own auto-merge, not a merge the watch makes: GitHub merges once the required
# checks pass and the reviews allow, and a change made on github.com shows in the next
# read. Squash, as this repository's pull requests merge.
def set_auto_merge(run: GhRunner, *, url: str, on: bool) -> str | None:
    """Turn GitHub's auto-merge of a pull request on or off through ``gh pr merge``.

    Args:
      run: Runs ``gh``.
      url: The pull request's link.
      on: Whether GitHub merges it once it may.

    Returns:
      error: The first line of why GitHub refused, or ``None`` once it took it.

    """
    flags = ("--auto", "--squash") if on else ("--disable-auto",)
    out = run(["pr", "merge", url, *flags])
    return None if out.returncode == 0 else _why(out)


def read_login(run: GhRunner) -> str:
    """Return the GitHub login ``gh`` reads with, as ``gh api user`` names it.

    Args:
      run: Runs ``gh``.

    Returns:
      login: The login.

    Raises:
      GhError: ``gh`` failed or printed no login.

    """
    out = run(["api", "user", "--jq", ".login"])
    login = out.stdout.strip()
    if out.returncode != 0 or not login:
        raise GhError(_why(out))
    return login


class _Check(TypedDict):
    """One check of the head, as :data:`_JQ` names its fields."""

    name: str
    workflow: str
    commit_status: bool
    """A commit status (``state``), not a check run (``status``, ``conclusion``)."""

    status: str
    conclusion: str
    state: str


class _View(TypedDict):
    """What ``gh pr view`` prints through :data:`_JQ`."""

    number: int
    url: str
    title: str
    state: str
    author: str
    draft: bool
    head: str
    head_sha: str
    base: str
    additions: int
    deletions: int
    mergeable: str
    review: str
    updated: str
    auto_merge: bool
    checks: list[_Check]


def _verdict(check: _Check) -> str:
    """Say whether one check ``passed``, ``failed`` or is ``pending``."""
    if check["commit_status"]:
        if check["state"] in _RUNNING:
            return "pending"
        return "failed" if check["state"] in _FAILED else "passed"
    if check["status"] != "COMPLETED":
        return "pending"
    return "failed" if check["conclusion"] in _FAILED else "passed"


def _names(checks: Iterable[_Check]) -> tuple[str, ...]:
    """Name each of ``checks`` once, as :func:`_name` does."""
    return tuple(dict.fromkeys(map(_name, checks)))


def _name(check: _Check) -> str:
    """Name a check as the Checks tab does: its workflow, then its job."""
    name = check["name"] or "(unnamed check)"
    return f"{check['workflow']} / {name}" if check["workflow"] else name


def _why(out: GhOutput) -> str:
    """Return the first line of why ``gh`` failed."""
    said = (out.stderr.strip() or out.stdout.strip()).splitlines()
    return (said or [f"gh exited {out.returncode}"])[0][:300]
