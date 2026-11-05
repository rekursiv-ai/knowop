"""A job on one's own machine: what may be submitted, and logs and results by relay."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import Mock, call

import base64
import hashlib
import io
import os
import shutil
import subprocess

from trackinizer.client.errors import ClientError
from trackinizer.wire.wire_machine_job import (
    Chunk,
    MachineJobSpec,
    MachineJobSubmit,
    MachineJobView,
    ReadRefused,
    ReadReply,
    ResultManifest,
)

import pytest

from knowop.jobs import machine_job
from knowop.jobs.job_fetch import IntegrityError
from knowop.jobs.job_source_test import _done
from knowop.jobs.machine_job import (
    Pushed,
    pushed,
    run_logs,
    run_results,
    run_submit,
)
from knowop.shell.job_folder import Ended, JobFolder


if TYPE_CHECKING:
    from pathlib import Path
    from subprocess import CompletedProcess


_REMOTE = "https://example.invalid/o/r.git"
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


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    work.mkdir()
    return work


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
        "machine runs what was pushed, so commit and push them first"
    )
    assert git.call_count == 6


def test_more_than_ten_changed_files_are_counted_not_all_named(
    repo: Path,
    replies: dict[str, CompletedProcess[bytes]],
) -> None:
    replies["status"] = _done("".join(f"?? f{n:02d}.py\0" for n in range(12)).encode())

    with pytest.raises(ClientError, match=r"f09\.py and 2 more; a job on a machine"):
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
        match=r"^submit --machine runs from inside a git repository$",
    ):
        _ = pushed(tmp_path / "plain")

    assert git.call_count == 1


def _view(state: str, **fields: object) -> MachineJobView:
    return MachineJobView.model_validate(
        {
            "id": 9,
            "machine": "gx10",
            "state": state,
            "gpus": 0,
            "memory_mb": 2_048,
            "minutes": 5,
            "label": "",
            "argv": ["python", "train.py"],
            "remote": _REMOTE,
            "ref": "refs/heads/main",
            "commit": "a" * 40,
            "exit_code": None,
            "reason": "",
            "created_at": datetime(2026, 10, 23, 12, tzinfo=UTC),
            "queued_at": None,
            "started_at": None,
            "ended_at": None,
            **fields,
        },
    )


@dataclass(slots=True, kw_only=True)
class _Machine:
    """The server's machine-job calls, answered from a job folder as the host would."""

    folder: JobFolder
    views: list[MachineJobView] = field(default_factory=list)
    submitted: list[MachineJobSubmit] = field(default_factory=list)
    reads: list[tuple[str, int]] = field(default_factory=list)
    results_max: int = 1_000
    grow: list[bytes] = field(default_factory=list)
    """Bytes the job writes to its log after each read, the last of which ends it."""

    corrupt: str = ""
    """A result file whose bytes the relay changes."""

    log_reply: ReadReply | None = None
    """What every log read answers, in place of the folder's log."""

    listing: ReadReply | None = None
    """What the manifest answers, in place of the folder's."""

    def machine_job_submit(self, body: MachineJobSubmit) -> MachineJobView:
        self.submitted.append(body)
        return _view("requested")

    def machine_job_view(self, job_id: int) -> MachineJobView:
        assert job_id == 9
        return self.views.pop(0) if len(self.views) > 1 else self.views[0]

    def machine_job_log(self, job_id: int, *, offset: int, wait: float) -> ReadReply:
        assert job_id == 9
        del wait
        self.reads.append(("log", offset))
        if self.log_reply is not None:
            return self.log_reply
        reply = self.folder.read_log(offset)
        if self.grow:
            self.folder.note(self.grow.pop(0).decode())
            if not self.grow:
                self.folder.write_ended(
                    Ended(state="completed", exit_code=0, reason=""),
                )
        return ReadReply(reply=reply)

    def machine_job_manifest(self, job_id: int) -> ReadReply:
        assert job_id == 9
        if self.listing is not None:
            return self.listing
        return ReadReply(
            reply=self.folder.manifest(results_max=self.results_max, where="gx10"),
        )

    def machine_job_file(self, job_id: int, path: str, *, offset: int) -> ReadReply:
        assert job_id == 9
        self.reads.append((path, offset))
        reply = self.folder.read_file(
            path,
            offset,
            results_max=self.results_max,
            where="gx10",
        )
        if path == self.corrupt and isinstance(reply, Chunk):
            data = base64.b64decode(reply.data_b64).upper()
            reply = reply.model_copy(
                update={"data_b64": base64.b64encode(data).decode()},
            )
        return ReadReply(reply=reply)


@pytest.fixture
def machine(tmp_path: Path) -> _Machine:
    spec = MachineJobSpec(
        id=9,
        owner="jan@example.com",
        argv=["python", "train.py"],
        remote=_REMOTE,
        ref="refs/heads/main",
        commit="a" * 40,
        gpus=0,
        memory_mb=2_048,
        minutes=5,
        label="",
    )
    return _Machine(folder=JobFolder.create(tmp_path / "machine" / "9", spec))


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _submit(machine: _Machine, repo: Path, clock: _Clock, *, gpus: int = 0) -> int:
    return run_submit(
        machine,
        repo=repo,
        machine="gx10",
        command=("python", "train.py"),
        gpus=gpus,
        memory_mb=2_048,
        minutes=5,
        label="smoke",
        sleep=clock.sleep,
        clock=clock,
    )


def test_submit_names_the_pushed_commit_and_waits_until_the_machine_takes_it(
    machine: _Machine,
    repo: Path,
    git: Mock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    machine.views = [_view("queued")]
    clock = _Clock()

    job = _submit(machine, repo, clock)

    assert git.call_args_list[0].kwargs["cwd"] == repo
    assert job == 9
    [body] = machine.submitted
    head = _HEAD
    assert (body.machine, body.argv, body.label) == (
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


def test_a_machine_that_does_not_take_the_job_in_a_minute_is_said_plainly(
    machine: _Machine,
    repo: Path,
) -> None:
    machine.views = [_view("requested")]

    with pytest.raises(ClientError, match=r"gx10 has not taken it in 60 s.*cancel 9"):
        _ = _submit(machine, repo, _Clock())


def test_a_job_that_ended_at_once_exits_one_with_its_reason(
    machine: _Machine,
    repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    machine.views = [_view("failed", reason="git fetch failed: fatal: no access")]

    with pytest.raises(SystemExit) as exited:
        _ = _submit(machine, repo, _Clock())

    assert exited.value.code == 1
    assert capsys.readouterr().err.endswith(
        "job 9 failed: git fetch failed: fatal: no access\n",
    )


def test_a_job_that_completed_at_once_is_fine(machine: _Machine, repo: Path) -> None:
    machine.views = [_view("completed")]

    assert _submit(machine, repo, _Clock()) == 9


def test_a_job_the_wire_refuses_is_named_field_by_field(
    machine: _Machine,
    repo: Path,
) -> None:
    with pytest.raises(
        ClientError,
        match=r"^invalid job: gpus: .*less than or equal to 8",
    ):
        _ = _submit(machine, repo, _Clock(), gpus=9)

    assert machine.submitted == []


def test_logs_print_what_the_log_holds_now(machine: _Machine) -> None:
    machine.folder.note("one")
    sink = io.BytesIO()

    run_logs(machine, 9, follow=False, sink=sink)

    assert sink.getvalue() == b"knowop: one\n"
    assert machine.reads == [("log", 0)]


def test_logs_follow_until_the_job_ends(machine: _Machine) -> None:
    machine.grow = [b"two", b"three"]
    sink = io.BytesIO()

    run_logs(machine, 9, follow=True, sink=sink)

    assert sink.getvalue() == b"knowop: two\nknowop: three\n"
    assert [offset for _, offset in machine.reads] == [0, 0, 12]


@pytest.mark.parametrize(
    ("reply", "why"),
    [
        (ReadRefused(reason="job 9 is not on gx10"), "^job 9 is not on gx10$"),
        (ResultManifest(entries=[], total_bytes=0), "answered a read with something"),
    ],
)
def test_logs_the_machine_will_not_give_say_why(
    machine: _Machine,
    reply: ReadRefused | ResultManifest,
    why: str,
) -> None:
    machine.log_reply = ReadReply(reply=reply)

    with pytest.raises(ClientError, match=why):
        run_logs(machine, 9, follow=False, sink=io.BytesIO())


def _results(machine: _Machine) -> None:
    (machine.folder.out / "d").mkdir()
    _ = (machine.folder.out / "d" / "a.txt").write_bytes(b"alpha")
    _ = (machine.folder.out / "b.bin").write_bytes(bytes(range(256)) * 3)


def test_results_download_every_file_checked_and_keep_the_job_beside_them(
    machine: _Machine,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _results(machine)
    view = _view("completed", exit_code=0)
    results = tmp_path / "results"

    run_results(machine, view, results=results)
    first = capsys.readouterr().out
    run_results(machine, view, results=results)

    assert (results / "out" / "d" / "a.txt").read_bytes() == b"alpha"
    assert (results / "out" / "b.bin").read_bytes() == bytes(range(256)) * 3
    assert (
        MachineJobView.model_validate_json((results / "job.json").read_text()) == view
    )
    assert first == f"results: {results / 'out'} (2 downloaded, 0 already held)\n"
    assert capsys.readouterr().out == (
        f"results: {results / 'out'} (0 downloaded, 2 already held)\n"
    )


def test_a_large_result_is_read_in_chunks_of_a_mebibyte(
    machine: _Machine,
    tmp_path: Path,
) -> None:
    data = hashlib.sha256(b"seed").digest() * (2**15 + 1)
    _ = (machine.folder.out / "big.bin").write_bytes(data)
    machine.results_max = 2**21

    run_results(machine, _view("completed"), results=tmp_path / "r")

    assert (tmp_path / "r" / "out" / "big.bin").read_bytes() == data
    assert [offset for name, offset in machine.reads if name == "big.bin"] == [0, 2**20]


def test_a_remote_result_reads_as_a_stream_that_ends_at_its_size(
    machine: _Machine,
) -> None:
    _ = (machine.folder.out / "a.bin").write_bytes(b"abc")
    remote = machine_job._RemoteFile(machine, 9, "a.bin", 3)

    with io.BufferedReader(remote) as stream:
        assert (stream.read(), stream.read()) == (b"abc", b"")

    assert machine.reads == [("a.bin", 0)]


def test_results_over_the_machines_limit_are_refused_with_where_they_stay(
    machine: _Machine,
    tmp_path: Path,
) -> None:
    _results(machine)
    machine.results_max = 100

    with pytest.raises(
        ClientError,
        match=r"over the 100 bytes gx10 sends back; they stay",
    ):
        run_results(machine, _view("completed"), results=tmp_path / "r")

    assert not (tmp_path / "r").exists()


def test_a_result_whose_bytes_changed_on_the_way_is_discarded(
    machine: _Machine,
    tmp_path: Path,
) -> None:
    _results(machine)
    machine.corrupt = "d/a.txt"

    with pytest.raises(IntegrityError, match="does not match its sha256"):
        run_results(machine, _view("completed"), results=tmp_path / "r")

    assert not (tmp_path / "r" / "out" / "d" / "a.txt").exists()


def test_a_result_that_shrank_after_its_listing_is_said_plainly(
    machine: _Machine,
    tmp_path: Path,
) -> None:
    _results(machine)
    listing = machine.folder.manifest(results_max=1_000, where="gx10")
    assert isinstance(listing, ResultManifest)
    _ = (machine.folder.out / "b.bin").write_bytes(b"")
    machine.listing = ReadReply(reply=listing)

    with pytest.raises(ClientError, match=r"b\.bin ended before its 768 bytes"):
        run_results(machine, _view("completed"), results=tmp_path / "r")


def test_a_manifest_answered_with_a_chunk_is_refused(
    machine: _Machine,
    tmp_path: Path,
) -> None:
    machine.listing = ReadReply(reply=Chunk(data_b64="", offset=0, size=0, ended=True))

    with pytest.raises(ClientError, match="answered a manifest with something else"):
        run_results(machine, _view("completed"), results=tmp_path / "r")


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
