"""Reading a pull request through ``gh``: its checks, its refusals, its auto-merge."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import shutil
import subprocess

from treekle import json_dumps

import pytest

from knowop.watch_pr import github
from knowop.watch_pr.github import (
    GhError,
    GhOutput,
    link,
    read_login,
    read_pull_request,
    run_gh,
    set_auto_merge,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


URL: Final = "https://github.com/acme/app/pull/7"


def view(**overrides: object) -> dict[str, object]:
    """Return what ``gh pr view`` prints through the watch's jq, open and checked."""
    unchecked: list[dict[str, object]] = []
    return {
        "number": 7,
        "url": URL,
        "title": "Fix the loader",
        "state": "OPEN",
        "author": "ada",
        "draft": False,
        "head": "fix-loader",
        "head_sha": "0123456789abcdef",
        "base": "main",
        "additions": 3,
        "deletions": 1,
        "mergeable": "MERGEABLE",
        "review": "",
        "updated": "2026-11-16T10:00:00Z",
        "auto_merge": False,
        "checks": unchecked,
    } | overrides


def check(name: str, *, workflow: str = "CI", **fields: object) -> dict[str, object]:
    """Return one check run, completed and passed unless ``fields`` say otherwise."""
    return {
        "name": name,
        "workflow": workflow,
        "commit_status": False,
        "status": "COMPLETED",
        "conclusion": "SUCCESS",
        "state": "",
    } | fields


class Gh:
    """A ``gh`` that prints what it is given, and keeps each call's arguments."""

    def __init__(self, *outputs: GhOutput) -> None:
        self.outputs = list(outputs)
        self.calls: list[list[str]] = []

    def __call__(self, args: Sequence[str]) -> GhOutput:
        self.calls.append(list(args))
        return self.outputs.pop(0)


def printed(value: object) -> GhOutput:
    return GhOutput(returncode=0, stdout=json_dumps(value), stderr="")


def test_a_read_sorts_the_heads_checks_and_names_each_once() -> None:
    gh = Gh(
        printed(
            view(
                checks=[
                    check("unit"),
                    check("lint", conclusion="FAILURE"),
                    check("lint", conclusion="FAILURE"),
                    check("types", status="IN_PROGRESS", conclusion=""),
                    check("old", conclusion="CANCELLED"),
                    check("deploy", workflow="", conclusion="TIMED_OUT"),
                    check("ci/x", workflow="", commit_status=True, state="PENDING"),
                    check("ci/y", workflow="", commit_status=True, state="ERROR"),
                    check("ci/z", workflow="", commit_status=True, state="SUCCESS"),
                    check("", workflow=""),
                ],
            ),
        ),
    )
    pr = read_pull_request(gh, URL)
    assert gh.calls[0][:3] == ["pr", "view", URL]
    assert (pr.repo, pr.number, pr.name) == ("acme/app", 7, "acme/app#7")
    assert pr.failed == ("CI / lint", "deploy", "ci/y")
    assert pr.pending == ("CI / types", "ci/x")
    assert pr.waiting == ("CI / lint",)
    assert (pr.passed, pr.total) == (4, 10)
    assert not pr.passing


def test_a_pull_request_passes_once_every_check_did() -> None:
    passing = read_pull_request(Gh(printed(view(checks=[check("unit")]))), URL)
    unchecked = read_pull_request(Gh(printed(view())), URL)
    assert passing.passing
    assert not unchecked.passing


def test_a_rows_app_keeps_what_github_said() -> None:
    pr = read_pull_request(Gh(printed(view(checks=[check("unit")]))), URL)
    assert pr.app() == {
        "url": URL,
        "repo": "acme/app",
        "number": 7,
        "title": "Fix the loader",
        "state": "OPEN",
        "draft": False,
        "head": "fix-loader",
        "head_sha": "0123456789abcdef",
        "base": "main",
        "additions": 3,
        "deletions": 1,
        "mergeable": "MERGEABLE",
        "review": "",
        "author": "ada",
        "updated": "2026-11-16T10:00:00Z",
        "auto_merge": False,
        "checks": {"passed": 1, "failed": [], "pending": [], "total": 1},
    }


@pytest.mark.parametrize(
    ("output", "said"),
    [
        (GhOutput(returncode=1, stdout="", stderr="no pull\nmore"), "no pull"),
        (GhOutput(returncode=1, stdout="printed", stderr=""), "printed"),
        (GhOutput(returncode=4, stdout="", stderr=""), "gh exited 4"),
        (
            GhOutput(returncode=0, stdout="not json", stderr=""),
            "gh printed no pull request",
        ),
        (printed({"number": 7}), "gh printed no pull request"),
        (printed(view(url="https://gitlab.com/a/b/7")), "gh named no GitHub pull"),
    ],
)
def test_a_read_that_fails_says_why(output: GhOutput, said: str) -> None:
    with pytest.raises(GhError, match=said):
        _ = read_pull_request(Gh(output), URL)


def test_a_link_is_read_up_to_its_number() -> None:
    assert link(f" {URL}/files#diff ") == URL
    with pytest.raises(ValueError, match="is no GitHub pull request") as raised:
        _ = link("https://github.com/acme/app/issues/7")
    assert str(raised.value) == (
        "'https://github.com/acme/app/issues/7' is no GitHub pull request: give its "
        "link, https://github.com/OWNER/REPO/pull/N"
    )


def test_auto_merge_turns_on_by_squash_and_off_and_says_a_refusal() -> None:
    gh = Gh(
        GhOutput(returncode=0, stdout="", stderr=""),
        GhOutput(returncode=1, stdout="", stderr="auto-merge is not allowed"),
    )
    assert set_auto_merge(gh, url=URL, on=True) is None
    assert set_auto_merge(gh, url=URL, on=False) == "auto-merge is not allowed"
    assert gh.calls == [
        ["pr", "merge", URL, "--auto", "--squash"],
        ["pr", "merge", URL, "--disable-auto"],
    ]


def test_the_login_is_gh_apis_user() -> None:
    gh = Gh(
        GhOutput(returncode=0, stdout="ada\n", stderr=""),
        GhOutput(returncode=0, stdout="", stderr=""),
    )
    assert read_login(gh) == "ada"
    assert gh.calls == [["api", "user", "--jq", ".login"]]
    with pytest.raises(GhError, match="gh exited 0"):
        _ = read_login(gh)


def on_path(name: str) -> str:
    return f"/bin/{name}"


def nowhere(name: str) -> None:
    del name


def test_gh_runs_without_a_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def run(argv: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        seen.update(options, argv=argv)
        return subprocess.CompletedProcess(argv, 0, stdout="out", stderr="")

    monkeypatch.setattr(shutil, "which", on_path)
    monkeypatch.setattr(subprocess, "run", run)
    assert run_gh(["pr", "view"]) == GhOutput(returncode=0, stdout="out", stderr="")
    assert seen["argv"] == ["/bin/gh", "pr", "view"]
    env = seen["env"]
    assert isinstance(env, dict)
    assert (env["GH_PROMPT_DISABLED"], env["GIT_TERMINAL_PROMPT"]) == ("1", "0")
    assert (seen["timeout"], seen["stdin"], seen["check"]) == (
        github.GH_SEC,
        subprocess.DEVNULL,
        False,
    )


def test_gh_that_is_missing_slow_or_unstartable_is_a_failed_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", nowhere)
    assert run_gh(["pr"]) == GhOutput(
        returncode=-1,
        stdout="",
        stderr="gh is not on this host's PATH",
    )
    monkeypatch.setattr(shutil, "which", on_path)

    def slow(argv: list[str], **options: object) -> None:
        del options
        raise subprocess.TimeoutExpired(argv, 30)

    monkeypatch.setattr(subprocess, "run", slow)
    assert run_gh(["pr"]) == GhOutput(
        returncode=-1,
        stdout="",
        stderr="gh took over 30s",
    )

    def broken(argv: list[str], **options: object) -> None:
        del argv, options
        raise PermissionError("denied")

    monkeypatch.setattr(subprocess, "run", broken)
    assert run_gh(["pr"]) == GhOutput(
        returncode=-1,
        stdout="",
        stderr="gh did not start: denied",
    )


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
