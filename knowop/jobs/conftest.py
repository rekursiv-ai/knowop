"""Fakes the job tests share: a host's job calls, a checkout, a clock.

Each test module keeps its own tests; what more than one reads lives here, so no test
module imports another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from subprocess import CompletedProcess
from typing import IO, TYPE_CHECKING, Final, cast

import base64
import hashlib
import io
import json
import tarfile

from trackinizer.client.errors import ClientError
from trackinizer.wire.wire_host_job import (
    Chunk,
    HostJobList,
    HostJobSpec,
    HostJobSubmit,
    HostJobView,
    ReadReply,
)
from trackinizer.wire.wire_job import (
    Grant,
    GrantStatus,
    GrantTicket,
    JobList,
    JobView,
    ReadGrantBody,
    SecretNames,
    StartBody,
    SubmitBody,
)

import pytest

from knowop.connect.job_folder import Ended, JobFolder
from knowop.jobs import job_source
from knowop.jobs.host_job import run_submit
from knowop.jobs.job import Rig
from knowop.jobs.job_source import BundleInfo, Snapshot
from knowop.jobs.job_ssh import Tools


if TYPE_CHECKING:
    from collections.abc import Sequence

    from trackinizer.client.client import Client


__all__ = [
    "_COMMIT",
    "_PUBLIC",
    "_REMOTE",
    "_TOOLS",
    "_Clock",
    "_Host",
    "_Rig",
    "_Server",
    "_Time",
    "_done",
    "_fake_source",
    "_grant",
    "_job",
    "_submit",
    "_view",
    "allowed",
    "outs",
    "snapshotted",
]


def _done(
    stdout: bytes = b"",
    *,
    code: int = 0,
    stderr: bytes = b"",
) -> CompletedProcess[bytes]:
    return CompletedProcess([], code, stdout, stderr)


_REMOTE: Final = "https://example.invalid/o/r.git"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Return an empty checkout folder."""
    work = tmp_path / "work"
    work.mkdir()
    return work


def _view(state: str, **fields: object) -> HostJobView:
    return HostJobView.model_validate(
        {
            "id": 9,
            "host": "gx10",
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
class _Host:
    """The server's host-job calls, answered from a job folder as the host would."""

    folder: JobFolder
    views: list[HostJobView] = field(default_factory=list)
    submitted: list[HostJobSubmit] = field(default_factory=list)
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

    def host_job_submit(self, body: HostJobSubmit) -> HostJobView:
        self.submitted.append(body)
        return _view("requested")

    def host_job_view(self, job_id: int) -> HostJobView:
        if job_id != 9:
            raise ValueError("Expected job_id == 9.")
        return self.views.pop(0) if len(self.views) > 1 else self.views[0]

    def host_job_log(self, job_id: int, *, offset: int, wait: float) -> ReadReply:
        if job_id != 9:
            raise ValueError("Expected job_id == 9.")
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

    def host_job_manifest(self, job_id: int) -> ReadReply:
        if job_id != 9:
            raise ValueError("Expected job_id == 9.")
        if self.listing is not None:
            return self.listing
        return ReadReply(
            reply=self.folder.manifest(results_max=self.results_max, where="gx10"),
        )

    def host_job_file(self, job_id: int, path: str, *, offset: int) -> ReadReply:
        if job_id != 9:
            raise ValueError("Expected job_id == 9.")
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
def host(tmp_path: Path) -> _Host:
    """Return the server's calls for job 9, answered from a fresh job folder."""
    spec = HostJobSpec(
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
    return _Host(folder=JobFolder.create(tmp_path / "host" / "9", spec))


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _submit(host: _Host, repo: Path, clock: _Clock, *, gpus: int = 0) -> int:
    return run_submit(
        host,
        repo=repo,
        host="gx10",
        command=("python", "train.py"),
        gpus=gpus,
        memory_mb=2_048,
        minutes=5,
        label="smoke",
        sleep=clock.sleep,
        clock=clock,
    )


_PUBLIC: Final = (
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPtWmP3ce7m3hrwpxvm9x5s3WnCClGsA5R5nJ9J25ir1"
)
_COMMIT: Final = "a" * 40


_TOOLS = Tools(ssh="ssh", ssh_keygen="ssh-keygen")


def _grant(role: str = "rw") -> Grant:
    return Grant.model_validate(
        {
            "job_id": 7,
            "serial": 3,
            "role": role,
            "node": {
                "name": "gpu-node-b",
                "host": "203.0.113.7",
                "port": 22,
                "user": "rai-ext-job-7",
                "host_key": _PUBLIC,
            },
            "certificate": (
                "ssh-ed25519-cert-v01@openssh.com AAAAIHNzaC1lZDI1NTE5LWNlcnQtdjAx"
            ),
            "valid_before": "2026-10-21T12:00:00Z",
            "max_bundle_bytes": 100_000,
        },
    )


def _job(state: str = "queued", **fields: object) -> JobView:
    return JobView.model_validate(
        {
            "id": 7,
            "state": state,
            "node": "gpu-node-b",
            "gpus": 1,
            "minutes": 5,
            "label": "try",
            "argv": ["python", "x.py"],
            "commit": _COMMIT,
            "bundle_sha256": None,
            "created_at": datetime(2026, 10, 21, 12, tzinfo=UTC),
            "started_at": None,
            "ended_at": None,
            "gpu_seconds": None,
            "usd": None,
            "reason": "",
            "upload_serial": 3,
            **fields,
        },
    )


class _Time:
    """A clock that only moves when something waits."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class _Server:
    """The server's ``job_*`` methods, scripted, recording every call."""

    base_url = "https://trax.example"

    def __init__(self, time: _Time) -> None:
        self.time = time
        self.calls: list[tuple[str, object]] = []
        self.grants = [GrantStatus(state="ready", grant=_grant())]
        self.read_grants = [GrantStatus(state="ready", grant=_grant("ro"))]
        self.views = [_job("queued")]
        self.names: list[str] = []

    def _next[T](self, queue: list[T]) -> T:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def job_submit(self, body: SubmitBody) -> JobView:
        self.calls.append(("submit", body))
        return _job("requested")

    def job_grant(self, job_id: int, *, wait: float) -> GrantStatus:
        self.calls.append(("grant", (job_id, wait)))
        status = self._next(self.grants)
        if status.state == "pending":
            self.time.now += wait
        return status

    def job_read_grant(self, job_id: int, body: ReadGrantBody) -> GrantTicket:
        self.calls.append(("read_grant", (job_id, body.public_key)))
        return GrantTicket(serial=9)

    def job_grant_serial(self, job_id: int, serial: int, *, wait: float) -> GrantStatus:
        self.calls.append(("grant_serial", (job_id, serial, wait)))
        return self._next(self.read_grants)

    def job_start(self, job_id: int, body: StartBody) -> JobView:
        self.calls.append(("start", (job_id, body)))
        return _job("starting")

    def job_view(self, job_id: int) -> JobView:
        self.calls.append(("job", job_id))
        return self._next(self.views)

    def job_list(self) -> JobList:
        self.calls.append(("jobs", None))
        return JobList(jobs=[_job("running"), _job("completed", id=6, node=None)])

    def job_cancel(self, job_id: int) -> JobView:
        self.calls.append(("cancel", job_id))
        return _job("cancelled")

    # Every job here is a GPU job: the server holds no host job of the caller's.
    def host_job_view(self, job_id: int) -> JobView:
        self.calls.append(("host_job", job_id))
        raise ClientError("-> 404", status_code=404)

    def host_job_cancel(self, job_id: int) -> JobView:
        self.calls.append(("host_cancel", job_id))
        raise ClientError("-> 404", status_code=404)

    def host_job_list(self) -> HostJobList:
        self.calls.append(("host_jobs", None))
        return HostJobList(jobs=[])

    def job_secret_names(self) -> SecretNames:
        return SecretNames(names=self.names)

    def job_put_secret(self, name: str, value: str) -> None:
        self.calls.append(("put_secret", (name, value)))

    def job_delete_secret(self, name: str) -> None:
        self.calls.append(("delete_secret", name))


class _Node:
    """Plays ssh-keygen and the node's gate for every verb."""

    def __init__(self) -> None:
        self.words: list[list[str]] = []
        self.keygen_dirs: list[Path] = []
        self.bases: list[str] = []
        self.upload_codes = [0]
        self.uploaded: list[bytes] = []
        self.reply_commit: str | None = None
        self.files: dict[str, bytes] = {"a.txt": b"alpha", "d/b.bin": b"beta"}
        self.interrupt = False
        self.log_chunks = ["é".encode()[:1], "é".encode()[1:] + b" line\n"]

    def run(
        self,
        argv: Sequence[str],
        *,
        stdin: bytes | IO[bytes] | None = None,
        stdout: IO[bytes] | None = None,
    ) -> int:
        if argv[0] == "ssh-keygen":
            key = Path(argv[argv.index("-f") + 1])
            self.keygen_dirs.append(key.parent)
            key.write_text("PRIVATE\n")
            key.with_suffix(".pub").write_text(f"{_PUBLIC} knowop-job\n")
            return 0
        if stdout is None:
            raise ValueError("Expected stdout is not None.")
        words = list(argv[argv.index("--") + 1 :])
        self.words.append(words)
        match words[0]:
            case "bases":
                _ = stdout.write("".join(f"{each}\n" for each in self.bases).encode())
            case "upload":
                if stdin is None:
                    raise ValueError("Expected stdin is not None.")
                if isinstance(stdin, bytes):
                    raise TypeError("Expected not isinstance(stdin, bytes).")
                data = stdin.read()
                self.uploaded.append(data)
                if code := self.upload_codes.pop(0):
                    return code
                reply = {
                    "commit": self.reply_commit or words[3],
                    "bundle_sha256": words[2],
                    "bytes": len(data),
                    "files": 2,
                }
                _ = stdout.write(json.dumps(reply).encode() + b"\n")
            case "logs":
                if self.interrupt:
                    raise KeyboardInterrupt
                for chunk in self.log_chunks:
                    _ = stdout.write(chunk)
            case "manifest":
                for path, data in self.files.items():
                    entry = {
                        "path": path,
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                    _ = stdout.write(json.dumps(entry).encode() + b"\n")
            case "download":
                assert isinstance(stdin, bytes)
                wanted = stdin.decode().split()
                with tarfile.open(fileobj=stdout, mode="w|") as tar:
                    for path in wanted:
                        info = tarfile.TarInfo(path)
                        info.size = len(self.files[path])
                        tar.addfile(info, io.BytesIO(self.files[path]))
            case _:
                raise AssertionError(words)
        return 0


class _Rig:
    def __init__(self) -> None:
        self.time = _Time()
        self.server = _Server(self.time)
        self.node = _Node()

    def factory(self) -> Client:
        return cast("Client", self.server)

    @property
    def rig(self) -> Rig:
        return Rig(
            client=cast("Client", self.server),
            runner=self.node,
            tools=_TOOLS,
            sleep=self.time.sleep,
            clock=self.time.clock,
        )


outs: list[str] = []
"""The file names the fake bundler was asked to write."""


snapshotted: list[Path] = []
"""The directories the fake snapshotter was asked to snapshot."""


allowed: list[bool] = []
"""Whether each fake snapshot was told to send secret-named untracked files."""


def _fake_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    size: int = 12,
    base: str | None = None,
    untracked: tuple[str, ...] = (),
    head: str = "",
    shallow: bool = False,
) -> list[list[str]]:
    """Replace git: the snapshot is a commit, the bundle a 12-byte file."""
    bundled: list[list[str]] = []
    outs.clear()
    snapshotted.clear()
    allowed.clear()
    private = tmp_path / "private"
    private.mkdir(exist_ok=True)

    def snapshot(
        repo: Path,
        tmp: Path,
        *,
        allow_secret_names: bool = False,
    ) -> Snapshot:
        del tmp
        snapshotted.append(repo)
        allowed.append(allow_secret_names)
        return Snapshot(
            commit=_COMMIT,
            private_git_dir=private,
            files=5,
            untracked=untracked,
            head=head,
            shallow=shallow,
        )

    def bundle(snap: Snapshot, bases: Sequence[str], out: Path) -> BundleInfo:
        if snap.commit != _COMMIT:
            raise ValueError("Expected snap.commit == _COMMIT.")
        outs.append(out.name)
        bundled.append(list(bases))
        data = b"b" * size
        out.write_bytes(data)
        return BundleInfo(
            path=out,
            bytes=size,
            sha256=hashlib.sha256(data).hexdigest(),
            base=base if bases else None,
        )

    monkeypatch.setattr(job_source, "snapshot", snapshot)
    monkeypatch.setattr(job_source, "bundle", bundle)
    return bundled
