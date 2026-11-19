"""``knowop watch-pr``'s commands, over a server and a gh of the test's."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import os
import signal

from trackinizer.client.errors import ClientError
from trackinizer.trax.profile import Profile

import pytest

from knowop.watch_pr import cli
from knowop.watch_pr.github import GhOutput
from knowop.watch_pr.github_test import URL, printed, view
from knowop.watch_pr.rows import DISMISSED, LABEL
from knowop.watch_pr.rows_test import FIXER, OTHER, Peer, Row, Server
from knowop.watch_pr.serve import ServeFiles, ServingError


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    import threading


PROFILE: Final = Profile(url="https://lab.test", api_key="k")


def laptop() -> str:
    return "laptop"


def login() -> tuple[str, Profile]:
    return "lab", PROFILE


def no_serve(files: ServeFiles) -> None:
    del files


class Run:
    """Runs commands against ``server``; a detached serve is recorded, not started."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.server = Server()
        self.files = ServeFiles(
            lock=tmp_path / "serve.lock",
            log=tmp_path / "serve.log",
        )
        self.detached: list[list[str]] = []
        self.gh_answer: GhOutput = printed(view())
        monkeypatch.setattr(cli, "serve_files", self._files)
        monkeypatch.setattr(cli, "host_name", laptop)
        monkeypatch.setattr(cli, "detach", self._detach)
        monkeypatch.delenv("TRAX_SESSION", raising=False)

    def __call__(self, *args: str) -> int:
        return cli.main(args, login=login, connect=self._connect, gh=self._gh)

    def row(self) -> Row:
        """Return the one row the server holds."""
        [row] = self.server.rows.values()
        return row

    def _files(self, server: str) -> ServeFiles:
        assert server == PROFILE.url
        return self.files

    def _detach(self, command: Sequence[str], *, files: ServeFiles) -> int:
        assert files == self.files
        self.detached.append(list(command))
        return 4242

    def _gh(self, args: Sequence[str]) -> GhOutput:
        assert args[:3] == ["pr", "view", URL]
        return self.gh_answer

    def _connect(self, profile: Profile) -> Server:
        assert profile == PROFILE
        return self.server


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Run:
    return Run(tmp_path, monkeypatch)


def test_add_makes_the_row_watches_it_and_starts_this_hosts_serve(
    run: Run,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run("add", f"{URL}/files", "--session", "lead#401") == 0
    row = run.row()
    assert (row.title, row.url) == ("acme/app#7 Fix the loader", URL)
    assert row.peers == [Peer(session=FIXER, note="lead")]
    [command] = run.detached
    assert command[1:4] == ["-P", "-m", "knowop"]
    assert command[4:] == ["--profile", "lab", "watch-pr", "serve"]
    assert capsys.readouterr().out.splitlines() == [
        f"watching {URL} for session {FIXER}",
        f"serve on this host: pid 4242; log {run.files.log}",
    ]


@pytest.mark.parametrize(
    ("session", "label"),
    [
        ("401", "Agent"),
        ("AgentSession#401", "Agent"),
        ("Agent#401", "Agent"),
        (str(FIXER), "Agent"),
        (f"lead#{FIXER}", "lead"),
    ],
)
def test_a_session_is_named_by_number_ref_uuid_or_agent(
    run: Run,
    session: str,
    label: str,
) -> None:
    assert run("add", URL, "--session", session) == 0
    assert run.row().peers == [Peer(session=FIXER, note=label)]


def test_trax_session_names_the_session_by_default(
    run: Run,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAX_SESSION", str(OTHER))
    assert run("add", URL) == 0
    assert run.row().peers == [Peer(session=OTHER)]


def test_no_session_is_refused(run: Run, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("add", URL) == 1
    assert "session.self" in capsys.readouterr().err
    assert run.server.rows == {}


def test_add_refuses_a_pull_request_this_hosts_gh_cannot_read(
    run: Run,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run.gh_answer = GhOutput(returncode=1, stdout="", stderr="gh auth login")
    assert run("add", URL, "--session", "401") == 1
    assert capsys.readouterr().err == (
        f"knowop watch-pr: gh on laptop cannot read {URL}: gh auth login\n"
        "Nothing is watched. Tell the person, naming laptop: there `gh auth login` "
        "fixes a gh logged out, and a gh missing needs installing; or add it from "
        "another host whose gh is logged in.\n"
    )
    assert (run.server.rows, run.detached) == ({}, [])


def test_a_link_that_is_no_pull_request_is_refused(
    run: Run,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run("add", "https://example.test/7", "--session", "401") == 1
    assert "is no GitHub pull request" in capsys.readouterr().err


def test_remove_lets_the_session_go(
    run: Run,
    capsys: pytest.CaptureFixture[str],
) -> None:
    row = run.server.row()
    assert run("remove", URL, "--session", "401") == 0
    assert row.peers[0].labels == [LABEL, DISMISSED]
    assert capsys.readouterr().out == f"let go of {URL} for session {FIXER}\n"


def test_remove_or_fix_of_a_pull_request_none_watched_is_refused(
    run: Run,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run("remove", URL, "--session", "401") == 1
    assert run("fix", URL, "--session", "401") == 1
    assert capsys.readouterr().err == 2 * (
        f"knowop watch-pr: no session watches {URL}; knowop watch-pr add watches it\n"
    )


def test_fix_names_the_fixer_and_watches_for_it(
    run: Run,
    capsys: pytest.CaptureFixture[str],
) -> None:
    row = run.server.row()
    assert run("fix", URL, "--session", "lead#402") == 0
    assert row.pr == {"fixer": str(OTHER)}
    assert row.peers == [Peer(session=FIXER), Peer(session=OTHER, note="lead")]
    assert run("fix", URL, "--session", "401") == 0
    assert row.pr == {"fixer": str(FIXER)}
    assert len(row.peers) == 2
    assert capsys.readouterr().out.splitlines() == [
        f"session {OTHER} fixes {URL}",
        f"session {FIXER} fixes {URL}",
    ]


def test_status_lists_who_watches_who_fixes_and_when_it_was_read(
    run: Run,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pr = {
        "state": "OPEN",
        "checks": {"passed": 1, "total": 2, "failed": ["CI / lint"]},
        "fixer": str(OTHER),
        "error": "HTTP 502",
        "read_at": "2026-11-16T10:00:00+00:00",
        "read_by": "laptop",
    }
    _ = run.server.row(
        app={"knowop": {"pr": pr}},
        peers=(Peer(session=FIXER), Peer(session=OTHER, note="lead")),
    )
    _ = run.server.row(url="https://github.com/acme/app/pull/8", peers=())
    assert run("status") == 0
    assert capsys.readouterr().out.splitlines() == [
        f"{URL}  OPEN  checks 1/2, 1 failed  error: HTTP 502",
        f"  watched by Agent of session {FIXER}",
        f"  watched by lead of session {OTHER} (fixes it)",
        "  read 2026-11-16T10:00:00+00:00 by laptop",
        "serve on this host: none; knowop watch-pr serve --detach starts one",
    ]


def test_status_of_one_unread_pull_request_and_a_serve_here(
    run: Run,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ = run.server.row()
    assert run("status", "https://github.com/acme/app/pull/8") == 0
    assert capsys.readouterr().out.splitlines()[0] == (
        "no session watches https://github.com/acme/app/pull/8"
    )
    monkeypatch.setattr(cli, "serving", serving_at(77, run))
    assert run("status", URL) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"{URL}  unread  checks 0/0",
        f"  watched by Agent of session {FIXER}",
        "  read never",
        "serve on this host: pid 77",
    ]


def test_status_names_a_merged_pull_request_when_asked_and_lists_only_open_ones(
    run: Run,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ = run.server.row(status="complete", app={"knowop": {"pr": {"state": "MERGED"}}})
    assert run("status") == 0
    assert capsys.readouterr().out.splitlines()[0] == "no open pull request is watched"
    assert run("status", URL) == 0
    assert capsys.readouterr().out.splitlines()[0] == f"{URL}  MERGED  checks 0/0"


def test_add_of_a_merged_pull_request_says_no_news_will_come(
    run: Run,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run.gh_answer = printed(view(state="MERGED"))
    assert run("add", URL, "--session", "401") == 0
    assert capsys.readouterr().out.splitlines()[:2] == [
        f"watching {URL} for session {FIXER}",
        "it is merged: no news of it will come",
    ]


def test_serve_detach_prints_the_pid_and_its_log(
    run: Run,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run("serve", "--detach") == 0
    assert (
        capsys.readouterr().out
        == f"serve on this host: pid 4242; log {run.files.log}\n"
    )
    [command] = run.detached
    assert command[4:] == ["--profile", "lab", "watch-pr", "serve"]


def test_serve_in_the_foreground_ends_on_a_signal(
    run: Run,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    passes: list[Server] = []

    def serve(
        client: Callable[[], Server],
        *,
        files: ServeFiles,
        host: str,
        stop: threading.Event,
    ) -> None:
        assert (files, host) == (run.files, "laptop")
        passes.append(client())
        os.kill(os.getpid(), signal.SIGTERM)
        assert stop.is_set()

    before = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(cli, "serve", serve)
    assert run("serve") == 0
    assert passes == [run.server]
    assert signal.getsignal(signal.SIGTERM) == before


def test_a_second_foreground_serve_is_refused(
    run: Run,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def serve(
        client: Callable[[], Server],
        *,
        files: ServeFiles,
        host: str,
        stop: threading.Event,
    ) -> None:
        del client, files, host, stop
        raise ServingError("knowop watch-pr serve runs on this host already (pid 7)")

    monkeypatch.setattr(cli, "serve", serve)
    assert run("serve") == 1
    assert capsys.readouterr().err == (
        "knowop watch-pr: knowop watch-pr serve runs on this host already (pid 7)\n"
    )


def serving_at(pid: int | None, run: Run) -> Callable[[ServeFiles], int | None]:
    def serving(files: ServeFiles) -> int | None:
        assert files == run.files
        return pid

    return serving


def test_stop_signals_this_hosts_serve(
    run: Run,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    killed: list[tuple[int, int]] = []

    def kill(pid: int, sig: int) -> None:
        killed.append((pid, sig))

    monkeypatch.setattr(os, "kill", kill)
    monkeypatch.setattr(cli, "serving", serving_at(None, run))
    assert run("stop") == 0
    monkeypatch.setattr(cli, "serving", serving_at(77, run))
    assert run("stop") == 0
    assert killed == [(77, signal.SIGTERM)]
    assert capsys.readouterr().out.splitlines() == [
        "serve on this host: none",
        "stopped the serve on this host (pid 77)",
    ]


def connect_none(profile: Profile) -> Server:
    raise AssertionError(f"connected to {profile.url}")


def test_a_login_that_fails_is_said(capsys: pytest.CaptureFixture[str]) -> None:
    def refused() -> tuple[str, Profile]:
        raise ClientError("knowop is not logged in")

    assert cli.main(["status"], login=refused, connect=connect_none) == 1
    assert capsys.readouterr().err == "knowop watch-pr: knowop is not logged in\n"


def test_help_reads_no_login() -> None:
    def unread() -> tuple[str, Profile]:
        raise AssertionError("help read the login")

    with pytest.raises(SystemExit) as raised:
        _ = cli.main(["--help"], login=unread, connect=connect_none)
    assert raised.value.code == 0


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
