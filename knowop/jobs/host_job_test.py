"""A job on one's own host: what may be submitted, and logs and results by relay."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, call

import hashlib
import io
import os
import shutil
import subprocess

from trackinizer.client.errors import ClientError
from trackinizer.wire.wire_host_job import (
    Chunk,
    HostJobView,
    ReadRefused,
    ReadReply,
    ResultManifest,
)

import pytest

from knowop.jobs import host_job
from knowop.jobs.conftest import (
    _REMOTE,
    _Clock,
    _done,
    _Host,
    _submit,
    _view,
)
from knowop.jobs.host_job import (
    Pushed,
    pushed,
    run_logs,
    run_results,
)
from knowop.jobs.job_fetch import IntegrityError


if TYPE_CHECKING:
    from pathlib import Path
    from subprocess import CompletedProcess


_HEAD = "a" * 40


@pytest.fixture
def git(monkeypatch: pytest.MonkeyPatch) -> Mock:
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(shutil, "which", Mock(return_value="/tools/git"))
    monkeypatch.setattr(
        os,
        "environ",
        {"PATH": "/tools", "KEEP": "value", "GIT_DIR": "/callers/repository"},
    )
    return run


@pytest.fixture(autouse=True)
def replies(repo: Path, git: Mock) -> dict[str, CompletedProcess[bytes]]:
    answers = {
        "top": _done(f" {repo}\n".encode()),
        "branch": _done(b" main\n"),
        "remote": _done(b" origin\n"),
        "ref": _done(b" refs/heads/main\n"),
        "url": _done(f" {_REMOTE}\n".encode()),
        "status": _done(),
        "head": _done(f" {_HEAD}\n".encode()),
        "tip": _done(f"{_HEAD}\trefs/heads/main\n".encode()),
    }
    git.side_effect = answers.values()
    return answers


def test_a_pushed_clean_branch_names_its_remote_branch_and_commit(
    repo: Path,
    git: Mock,
) -> None:
    (repo / "sub").mkdir()

    found = pushed(repo / "sub")

    assert found == Pushed(
        remote=_REMOTE,
        ref="refs/heads/main",
        commit=_HEAD,
        branch="main",
    )
    env = {
        "PATH": "/tools",
        "KEEP": "value",
        "PYTEST_CURRENT_TEST": os.environ["PYTEST_CURRENT_TEST"],
    }
    git.assert_has_calls(
        [
            call(
                ["/tools/git", "rev-parse", "--show-toplevel"],
                cwd=repo / "sub",
                env=env,
                input=b"",
                capture_output=True,
                check=False,
            ),
            *[
                call(
                    ["/tools/git", *args],
                    cwd=repo,
                    env=env,
                    input=b"",
                    capture_output=True,
                    check=False,
                )
                for args in (
                    ("symbolic-ref", "--short", "HEAD"),
                    ("config", "branch.main.remote"),
                    ("config", "branch.main.merge"),
                    ("config", "remote.origin.url"),
                    ("status", "--porcelain=v1", "--untracked-files=normal", "-z"),
                    ("rev-parse", "HEAD"),
                    ("ls-remote", _REMOTE, "refs/heads/main"),
                )
            ],
        ],
    )
    assert git.call_count == 8


def test_a_file_that_differs_from_the_pushed_commit_is_refused(
    repo: Path,
    replies: dict[str, CompletedProcess[bytes]],
    git: Mock,
) -> None:
    replies["status"] = _done(b" M train.py\0?? new.py\0")

    with pytest.raises(
        ClientError,
        match=r"^2 files differ from the pushed commit: ",
    ) as refused:
        _ = pushed(repo)

    assert str(refused.value) == (
        "2 files differ from the pushed commit: train.py, new.py; a job on a "
        "host runs what was pushed, so commit and push them first"
    )
    assert git.call_count == 6


def test_more_than_ten_changed_files_are_counted_not_all_named(
    repo: Path,
    replies: dict[str, CompletedProcess[bytes]],
) -> None:
    replies["status"] = _done("".join(f"?? f{n:02d}.py\0" for n in range(12)).encode())

    with pytest.raises(ClientError, match=r"f09\.py and 2 more; a job on a host"):
        _ = pushed(repo)


def test_a_commit_not_yet_pushed_is_refused(
    repo: Path,
    replies: dict[str, CompletedProcess[bytes]],
) -> None:
    head, tip = "b" * 40, _HEAD
    replies["head"] = _done(f"{head}\n".encode())

    with pytest.raises(
        ClientError,
        match=f"origin/main is at {tip[:12]} but HEAD is at {head[:12]}; push HEAD first",
    ):
        _ = pushed(repo)


def test_a_branch_the_remote_does_not_hold_is_refused(
    repo: Path,
    replies: dict[str, CompletedProcess[bytes]],
    git: Mock,
) -> None:
    replies["branch"] = _done(b"side\n")
    replies["ref"] = _done(b"refs/heads/side\n")
    replies["tip"] = _done()

    with pytest.raises(ClientError, match="origin/side is at nothing"):
        _ = pushed(repo)

    assert git.call_args.args[0] == [
        "/tools/git",
        "ls-remote",
        _REMOTE,
        "refs/heads/side",
    ]


def test_a_detached_head_is_refused(
    repo: Path,
    replies: dict[str, CompletedProcess[bytes]],
    git: Mock,
) -> None:
    replies["branch"] = _done(b"discard failed output", code=1)

    with pytest.raises(
        ClientError,
        match=r"^HEAD is detached; check out the branch you pushed$",
    ):
        _ = pushed(repo)

    assert git.call_count == 2


def test_a_branch_with_no_upstream_is_refused(
    repo: Path,
    replies: dict[str, CompletedProcess[bytes]],
    git: Mock,
) -> None:
    replies["branch"] = _done(b"local\n")
    for key in ("remote", "ref", "url"):
        replies[key] = _done(code=1)

    with pytest.raises(ClientError, match="push it with git push -u origin local"):
        _ = pushed(repo)

    assert git.call_count == 5


@pytest.mark.parametrize(
    ("key", "value"),
    [("branch.local.merge", "refs/heads/local"), ("branch.local.remote", "origin")],
)
def test_a_branch_with_half_an_upstream_has_no_upstream(
    repo: Path,
    replies: dict[str, CompletedProcess[bytes]],
    key: str,
    value: str,
) -> None:
    replies["branch"] = _done(b"local\n")
    replies["remote"] = _done(code=1)
    replies["ref"] = _done(code=1)
    replies["url"] = _done(code=1)
    if key.endswith(".merge"):
        replies["ref"] = _done(f"{value}\n".encode())
    else:
        replies["remote"] = _done(f"{value}\n".encode())
        replies["url"] = _done(f"{_REMOTE}\n".encode())

    with pytest.raises(ClientError, match=r"^branch local has no upstream on a remote"):
        _ = pushed(repo)


@pytest.mark.parametrize("merge", ["refs/tags/v1", "--upload-pack=touch x"])
def test_an_upstream_that_is_not_a_branch_is_refused_before_git_reads_it(
    repo: Path,
    replies: dict[str, CompletedProcess[bytes]],
    git: Mock,
    merge: str,
) -> None:
    replies["ref"] = _done(f"{merge}\n".encode())

    with pytest.raises(
        ClientError,
        match=r"^origin/main cannot be named in a job: the ref must be a branch",
    ):
        _ = pushed(repo)

    assert git.call_count == 5


def test_a_remote_whose_url_holds_a_token_is_refused(
    repo: Path,
    replies: dict[str, CompletedProcess[bytes]],
    git: Mock,
) -> None:
    replies["url"] = _done(b"https://ghp_x@example.invalid/o/r.git\n")

    with pytest.raises(
        ClientError,
        match=r"^origin/main cannot be named in a job: a remote URL must not carry",
    ):
        _ = pushed(repo)

    assert git.call_count == 5


def test_a_folder_outside_any_repository_is_refused(tmp_path: Path, git: Mock) -> None:
    (tmp_path / "plain").mkdir()
    git.side_effect = [_done(code=128)]

    with pytest.raises(
        ClientError,
        match=r"^submit --host runs from inside a git repository$",
    ):
        _ = pushed(tmp_path / "plain")

    assert git.call_count == 1


def test_submit_names_the_pushed_commit_and_waits_until_the_host_takes_it(
    host: _Host,
    repo: Path,
    git: Mock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    host.views = [_view("queued")]
    clock = _Clock()

    job = _submit(host, repo, clock)

    assert git.call_args_list[0].kwargs["cwd"] == repo
    assert job == 9
    [body] = host.submitted
    head = _HEAD
    assert (body.host, body.argv, body.label) == (
        "gx10",
        ["python", "train.py"],
        "smoke",
    )
    assert (body.remote, body.ref, body.commit) == (_REMOTE, "refs/heads/main", head)
    assert clock.now == 1
    assert capsys.readouterr().err == (
        f"code: {head[:12]} of main, pushed to {_REMOTE}\n"
        "job 9: requested on gx10\n"
        "job 9: queued on gx10\n"
    )


def test_a_host_that_does_not_take_the_job_in_a_minute_is_said_plainly(
    host: _Host,
    repo: Path,
) -> None:
    host.views = [_view("requested")]

    with pytest.raises(ClientError, match=r"gx10 has not taken it in 60 s.*cancel 9"):
        _ = _submit(host, repo, _Clock())


def test_a_job_that_ended_at_once_exits_one_with_its_reason(
    host: _Host,
    repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    host.views = [_view("failed", reason="git fetch failed: fatal: no access")]

    with pytest.raises(SystemExit) as exited:
        _ = _submit(host, repo, _Clock())

    assert exited.value.code == 1
    assert capsys.readouterr().err.endswith(
        "job 9 failed: git fetch failed: fatal: no access\n",
    )


def test_a_job_that_completed_at_once_is_fine(host: _Host, repo: Path) -> None:
    host.views = [_view("completed")]

    assert _submit(host, repo, _Clock()) == 9


def test_a_job_the_wire_refuses_is_named_field_by_field(
    host: _Host,
    repo: Path,
) -> None:
    with pytest.raises(
        ClientError,
        match=r"^invalid job: gpus: .*less than or equal to 8",
    ):
        _ = _submit(host, repo, _Clock(), gpus=9)

    assert host.submitted == []


def test_logs_print_what_the_log_holds_now(host: _Host) -> None:
    host.folder.note("one")
    sink = io.BytesIO()

    run_logs(host, 9, follow=False, sink=sink)

    assert sink.getvalue() == b"knowop: one\n"
    assert host.reads == [("log", 0)]


def test_logs_follow_until_the_job_ends(host: _Host) -> None:
    host.grow = [b"two", b"three"]
    sink = io.BytesIO()

    run_logs(host, 9, follow=True, sink=sink)

    assert sink.getvalue() == b"knowop: two\nknowop: three\n"
    assert [offset for _, offset in host.reads] == [0, 0, 12]


@pytest.mark.parametrize(
    ("reply", "why"),
    [
        (ReadRefused(reason="job 9 is not on gx10"), "^job 9 is not on gx10$"),
        (ResultManifest(entries=[], total_bytes=0), "answered a read with something"),
    ],
)
def test_logs_the_host_will_not_give_say_why(
    host: _Host,
    reply: ReadRefused | ResultManifest,
    why: str,
) -> None:
    host.log_reply = ReadReply(reply=reply)

    with pytest.raises(ClientError, match=why):
        run_logs(host, 9, follow=False, sink=io.BytesIO())


def _results(host: _Host) -> None:
    (host.folder.out / "d").mkdir()
    _ = (host.folder.out / "d" / "a.txt").write_bytes(b"alpha")
    _ = (host.folder.out / "b.bin").write_bytes(bytes(range(256)) * 3)


def test_results_download_every_file_checked_and_keep_the_job_beside_them(
    host: _Host,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _results(host)
    view = _view("completed", exit_code=0)
    results = tmp_path / "results"

    run_results(host, view, results=results)
    first = capsys.readouterr().out
    run_results(host, view, results=results)

    assert (results / "out" / "d" / "a.txt").read_bytes() == b"alpha"
    assert (results / "out" / "b.bin").read_bytes() == bytes(range(256)) * 3
    assert HostJobView.model_validate_json((results / "job.json").read_text()) == view
    assert first == f"results: {results / 'out'} (2 downloaded, 0 already held)\n"
    assert capsys.readouterr().out == (
        f"results: {results / 'out'} (0 downloaded, 2 already held)\n"
    )


def test_a_large_result_is_read_in_chunks_of_a_mebibyte(
    host: _Host,
    tmp_path: Path,
) -> None:
    data = hashlib.sha256(b"seed").digest() * (2**15 + 1)
    _ = (host.folder.out / "big.bin").write_bytes(data)
    host.results_max = 2**21

    run_results(host, _view("completed"), results=tmp_path / "r")

    assert (tmp_path / "r" / "out" / "big.bin").read_bytes() == data
    assert [offset for name, offset in host.reads if name == "big.bin"] == [0, 2**20]


def test_a_remote_result_reads_as_a_stream_that_ends_at_its_size(
    host: _Host,
) -> None:
    _ = (host.folder.out / "a.bin").write_bytes(b"abc")
    remote = host_job._RemoteFile(host, 9, "a.bin", 3)

    with io.BufferedReader(remote) as stream:
        assert (stream.read(), stream.read()) == (b"abc", b"")

    assert host.reads == [("a.bin", 0)]


def test_results_over_the_hosts_limit_are_refused_with_where_they_stay(
    host: _Host,
    tmp_path: Path,
) -> None:
    _results(host)
    host.results_max = 100

    with pytest.raises(
        ClientError,
        match=r"over the 100 bytes gx10 sends back; they stay",
    ):
        run_results(host, _view("completed"), results=tmp_path / "r")

    assert not (tmp_path / "r").exists()


def test_a_result_whose_bytes_changed_on_the_way_is_discarded(
    host: _Host,
    tmp_path: Path,
) -> None:
    _results(host)
    host.corrupt = "d/a.txt"

    with pytest.raises(IntegrityError, match="does not match its sha256"):
        run_results(host, _view("completed"), results=tmp_path / "r")

    assert not (tmp_path / "r" / "out" / "d" / "a.txt").exists()


def test_a_result_that_shrank_after_its_listing_is_said_plainly(
    host: _Host,
    tmp_path: Path,
) -> None:
    _results(host)
    listing = host.folder.manifest(results_max=1_000, where="gx10")
    assert isinstance(listing, ResultManifest)
    _ = (host.folder.out / "b.bin").write_bytes(b"")
    host.listing = ReadReply(reply=listing)

    with pytest.raises(ClientError, match=r"b\.bin ended before its 768 bytes"):
        run_results(host, _view("completed"), results=tmp_path / "r")


def test_a_manifest_answered_with_a_chunk_is_refused(
    host: _Host,
    tmp_path: Path,
) -> None:
    host.listing = ReadReply(reply=Chunk(data_b64="", offset=0, size=0, ended=True))

    with pytest.raises(ClientError, match="answered a manifest with something else"):
        run_results(host, _view("completed"), results=tmp_path / "r")


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
