"""A job on one's own machine: what may be submitted, and logs and results by relay."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import base64
import hashlib
import io
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
from knowop.jobs.machine_job import (
    Pushed,
    pushed,
    run_logs,
    run_results,
    run_submit,
)
from knowop.shell.job_folder import Ended, JobFolder


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


_REMOTE = "https://example.invalid/o/r.git"


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(  # noqa: S603 -- fixed git argv on a tmp_path repo.
        ["git", "-C", str(repo), *args],  # noqa: S607 -- git is the tool under test.
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout.strip()


@pytest.fixture
def bare(tmp_path: Path) -> Path:
    found = tmp_path / "remote.git"
    _ = _git(tmp_path, "init", "--quiet", "--bare", str(found))
    return found


@pytest.fixture(autouse=True)
def _git_config(tmp_path: Path, bare: Path) -> Iterator[None]:
    """Set a git identity; serve the https remote from a folder by ``insteadOf``."""
    config = tmp_path / "gitconfig"
    _ = config.write_text(
        "[user]\n\tname = t\n\temail = t@example.com\n"
        f'[url "file://{bare}"]\n\tinsteadOf = {_REMOTE}\n',
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("GIT_CONFIG_GLOBAL", str(config))
        patch.setenv("GIT_CONFIG_NOSYSTEM", "1")
        patch.delenv("GIT_DIR", raising=False)
        yield


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Return a repository whose ``main`` is pushed to the remote as it stands."""
    work = tmp_path / "work"
    work.mkdir()
    _ = _git(work, "init", "--quiet", "--initial-branch=main")
    _ = (work / "train.py").write_text("print(1)\n")
    _ = (work / ".gitignore").write_text("*.log\n")
    _ = _git(work, "add", "-A")
    _ = _git(work, "commit", "--quiet", "-m", "one")
    _ = _git(work, "remote", "add", "origin", _REMOTE)
    _ = _git(work, "push", "--quiet", "-u", "origin", "main")
    return work


def test_a_pushed_clean_branch_names_its_remote_branch_and_commit(repo: Path) -> None:
    (repo / "sub").mkdir()
    _ = (repo / "run.log").write_text("ignored\n")

    found = pushed(repo / "sub")

    assert found == Pushed(
        remote=_REMOTE,
        ref="refs/heads/main",
        commit=_git(repo, "rev-parse", "HEAD"),
        branch="main",
    )


def test_a_file_that_differs_from_the_pushed_commit_is_refused(repo: Path) -> None:
    _ = (repo / "train.py").write_text("print(2)\n")
    _ = (repo / "new.py").write_text("x\n")

    with pytest.raises(ClientError, match=r"^2 files differ from the pushed commit: "):
        _ = pushed(repo)


def test_more_than_ten_changed_files_are_counted_not_all_named(repo: Path) -> None:
    for n in range(12):
        _ = (repo / f"f{n:02d}.py").write_text("x\n")

    with pytest.raises(ClientError, match=r"f09\.py and 2 more; a job on a machine"):
        _ = pushed(repo)


def test_a_commit_not_yet_pushed_is_refused(repo: Path) -> None:
    _ = _git(repo, "commit", "--quiet", "--allow-empty", "-m", "two")
    head = _git(repo, "rev-parse", "HEAD")
    tip = _git(repo, "rev-parse", "origin/main")

    with pytest.raises(
        ClientError,
        match=f"origin/main is at {tip[:12]} but HEAD is at {head[:12]}; push HEAD first",
    ):
        _ = pushed(repo)


def test_a_branch_the_remote_does_not_hold_is_refused(repo: Path) -> None:
    _ = _git(repo, "checkout", "--quiet", "-b", "side")
    _ = _git(repo, "config", "branch.side.remote", "origin")
    _ = _git(repo, "config", "branch.side.merge", "refs/heads/side")

    with pytest.raises(ClientError, match="origin/side is at nothing"):
        _ = pushed(repo)


def test_a_detached_head_is_refused(repo: Path) -> None:
    _ = _git(repo, "checkout", "--quiet", "--detach")

    with pytest.raises(
        ClientError,
        match=r"^HEAD is detached; check out the branch you pushed$",
    ):
        _ = pushed(repo)


def test_a_branch_with_no_upstream_is_refused(repo: Path) -> None:
    _ = _git(repo, "checkout", "--quiet", "-b", "local")

    with pytest.raises(ClientError, match="push it with git push -u origin local"):
        _ = pushed(repo)


@pytest.mark.parametrize(
    ("key", "value"),
    [("branch.local.merge", "refs/heads/local"), ("branch.local.remote", "origin")],
)
def test_a_branch_with_half_an_upstream_has_no_upstream(
    repo: Path,
    key: str,
    value: str,
) -> None:
    _ = _git(repo, "checkout", "--quiet", "-b", "local")
    _ = _git(repo, "config", key, value)

    with pytest.raises(ClientError, match=r"^branch local has no upstream on a remote"):
        _ = pushed(repo)


@pytest.mark.parametrize("merge", ["refs/tags/v1", "--upload-pack=touch x"])
def test_an_upstream_that_is_not_a_branch_is_refused_before_git_reads_it(
    repo: Path,
    merge: str,
) -> None:
    _ = _git(repo, "config", "branch.main.merge", merge)

    with pytest.raises(
        ClientError,
        match=r"^origin/main cannot be named in a job: the ref must be a branch",
    ):
        _ = pushed(repo)


def test_a_remote_whose_url_holds_a_token_is_refused(repo: Path) -> None:
    _ = _git(
        repo,
        "remote",
        "set-url",
        "origin",
        "https://ghp_x@example.invalid/o/r.git",
    )

    with pytest.raises(
        ClientError,
        match=r"^origin/main cannot be named in a job: a remote URL must not carry",
    ):
        _ = pushed(repo)


def test_a_folder_outside_any_repository_is_refused(tmp_path: Path) -> None:
    (tmp_path / "plain").mkdir()

    with pytest.raises(
        ClientError,
        match=r"^submit --machine runs from inside a git repository$",
    ):
        _ = pushed(tmp_path / "plain")


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
    capsys: pytest.CaptureFixture[str],
) -> None:
    machine.views = [_view("queued")]
    clock = _Clock()

    job = _submit(machine, repo, clock)

    assert job == 9
    [body] = machine.submitted
    head = _git(repo, "rev-parse", "HEAD")
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
