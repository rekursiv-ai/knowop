"""One job's folder on the machine that runs it, and the reads the server relays from it.

``<root>/<id>/`` holds ``spec.json`` (the job as the server handed it), ``launch.json``
(the scope it runs in, its GPUs, its cap), ``log`` (the host's ``knowop:`` lines, then
the job's output), ``exit.json`` (how it ended, which the runner writes), ``out/`` (the
results, ``$KNOWOP_JOB_OUT``), ``src/`` (the worktree, removed when the job ends) and
``ended`` (when the host reported the end, which the retention sweep counts from).

A read reaches nothing else. A result's path is relative with no ``.`` or ``..`` part,
and every part of it is opened with ``O_NOFOLLOW`` from the open ``out/`` folder, as
``ops/gpunode/gate.py`` does on a GPU node: a link the job left there is never
followed, so a read cannot return ``~/.ssh`` or any other file of the machine's user.
The results a job sends back are bounded by the machine's owner (256 MiB unless they
set another); over that, a read is refused with the folder's path, and the files stay.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Final, Literal

import base64
import errno
import hashlib
import os
import stat
import time
import uuid

from pydantic import BaseModel, ConfigDict
from trackinizer.wire.wire_job import ManifestEntry
from trackinizer.wire.wire_machine_job import (
    MAX_CHUNK_BYTES,
    MAX_MANIFEST_ENTRIES,
    Chunk,
    MachineJobSpec,
    ReadRefused,
    ResultManifest,
)


__all__ = ["Ended", "JobBinding", "JobFolder", "Launch"]

_NO_FOLLOW: Final = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK


class JobBinding(BaseModel):
    """The server, account and machine authorized to manage a persisted job."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    server: str
    account: str
    machine_id: uuid.UUID


class Launch(BaseModel):
    """How a job was started: its scope, its GPUs and its cap."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit: str
    gpus: list[int]
    memory_mb: int


class Ended(BaseModel):
    """How a job ended, as its runner saw it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: Literal["completed", "failed", "timeout"]
    exit_code: int | None
    reason: str


class JobFolder:
    """One job's folder.

    Args:
      path: ``<root>/<id>``.

    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.src = path / "src"
        """The job's worktree, and the folder its command starts in."""

        self.out = path / "out"
        """The job's results: ``$KNOWOP_JOB_OUT``."""

        self.log = path / "log"

    @classmethod
    def create(cls, path: Path, spec: MachineJobSpec) -> JobFolder:
        """Make a job's folder, readable by this user only, and keep its spec in it.

        Args:
          path: ``<root>/<id>``; it must not exist yet.
          spec: The job as the server handed it.

        Returns:
          folder: The new folder, with an empty ``out/``.

        """
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.mkdir(mode=0o700)
        folder = cls(path)
        folder.out.mkdir()
        _ = (path / "spec.json").write_text(spec.model_dump_json())
        return folder

    def spec(self) -> MachineJobSpec:
        """Return the job as the server handed it."""
        return MachineJobSpec.model_validate_json((self.path / "spec.json").read_text())

    def bind(self, binding: JobBinding) -> None:
        """Persist the identity that accepted this job."""
        _write(self.path / "binding.json", binding.model_dump_json())

    def binding(self) -> JobBinding | None:
        """Return the accepting identity; legacy folders have no binding."""
        return _read(self.path / "binding.json", JobBinding)

    def write_launch(self, launch: Launch) -> None:
        """Record how the job was started, before it starts."""
        _write(self.path / "launch.json", launch.model_dump_json())

    def launch(self) -> Launch | None:
        """Return how the job was started, or ``None`` before it was."""
        return _read(self.path / "launch.json", Launch)

    def write_ended(self, ended: Ended) -> None:
        """Record how the job ended; written whole, or not at all."""
        _write(self.path / "exit.json", ended.model_dump_json())

    def ended(self) -> Ended | None:
        """Return how the job ended, or ``None`` while it has not."""
        return _read(self.path / "exit.json", Ended)

    def mark_reported(self, at: float) -> None:
        """Record that the host reported the job's end, at ``at`` (epoch seconds)."""
        _write(self.path / "ended", f"{at}\n")

    def reported_at(self) -> float | None:
        """Return when the host reported the job's end, or ``None`` before it did."""
        try:
            return float((self.path / "ended").read_text())
        except FileNotFoundError:
            return None

    def note(self, line: str) -> None:
        """Append one of the host's own lines to the log, marked ``knowop:``."""
        with self.log.open("a") as log:
            _ = log.write(f"knowop: {line}\n")

    def read_log(self, offset: int) -> Chunk:
        """Return at most 1 MiB of the log from ``offset``.

        Args:
          offset: The first byte wanted.

        Returns:
          chunk: The bytes, the log's length, and whether the job has ended.

        """
        try:
            fd = os.open(self.log, _NO_FOLLOW)
        except FileNotFoundError:
            return _chunk(b"", offset=offset, size=0, ended=self._over())
        with os.fdopen(fd, "rb") as log:
            info = os.fstat(log.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise OSError(errno.EINVAL, "The log is not a regular file")
            size = info.st_size
            _ = log.seek(offset)
            data = log.read(MAX_CHUNK_BYTES)
        return _chunk(data, offset=offset, size=size, ended=self._over())

    def manifest(self, *, results_max: int, where: str) -> ResultManifest | ReadRefused:
        """List every regular file under ``out/`` with its size and sha256.

        Args:
          results_max: The most result bytes this machine sends back.
          where: The machine's name, for a refusal.

        Returns:
          manifest: The files by path, or why they are not sent.

        """
        listing = self._listing(results_max=results_max, where=where)
        if isinstance(listing, ReadRefused):
            return listing
        try:
            entries = [
                ManifestEntry(path=name, size=size, sha256=self._digest(name))
                for name, size in listing
            ]
            return ResultManifest(
                entries=entries,
                total_bytes=sum(size for _, size in listing),
            )
        except (OSError, ValueError):
            return ReadRefused(reason="a result file changed while it was listed")

    def read_file(
        self,
        name: str,
        offset: int,
        *,
        results_max: int,
        where: str,
    ) -> Chunk | ReadRefused:
        """Return at most 1 MiB of one result file from ``offset``.

        Args:
          name: The file's path under ``out/``.
          offset: The first byte wanted.
          results_max: The most result bytes this machine sends back.
          where: The machine's name, for a refusal.

        Returns:
          chunk: The bytes and the file's length, or why they are not sent.

        """
        listing = self._listing(results_max=results_max, where=where)
        if isinstance(listing, ReadRefused):
            return listing
        if name not in dict(listing):
            return ReadRefused(reason=f"{name!r} is not a result file of this job")
        try:
            with ExitStack() as stack:
                fd = _open_under(self.out, name, stack=stack)
                size = os.fstat(fd).st_size
                data = os.pread(fd, MAX_CHUNK_BYTES, offset)
        except (OSError, ValueError):
            return ReadRefused(reason=f"{name!r} changed while it was read")
        return _chunk(data, offset=offset, size=size, ended=True)

    def _over(self) -> bool:
        """Whether the job has ended, so its log will not grow."""
        return self.ended() is not None or self.reported_at() is not None

    def _listing(
        self,
        *,
        results_max: int,
        where: str,
    ) -> list[tuple[str, int]] | ReadRefused:
        """Return each regular file under ``out/`` and its size, or why none is sent."""
        found: list[tuple[str, int]] = []
        for top, _, files in os.walk(self.out):
            for file in files:
                path = Path(top) / file
                name = path.relative_to(self.out).as_posix()
                info = path.lstat()
                if stat.S_ISREG(info.st_mode) and _plain(name):
                    found.append((name, info.st_size))
            if len(found) > MAX_MANIFEST_ENTRIES:
                return ReadRefused(
                    reason=(
                        f"the results hold more than {MAX_MANIFEST_ENTRIES} files; "
                        f"they stay in {self.out} on {where}"
                    ),
                )
        total = sum(size for _, size in found)
        if total > results_max:
            return ReadRefused(
                reason=(
                    f"the results are {total} bytes, over the {results_max} bytes "
                    f"{where} sends back; they stay in {self.out} on {where}"
                ),
            )
        return sorted(found)

    def _digest(self, name: str) -> str:
        """Return the sha256 of one result file, opened as :meth:`read_file` does."""
        with ExitStack() as stack:
            fd = _open_under(self.out, name, stack=stack)
            with os.fdopen(os.dup(fd), "rb") as file:
                return hashlib.file_digest(file, hashlib.sha256).hexdigest()


def _plain(name: str) -> bool:
    """Whether a result path can travel: UTF-8, no newline, no NUL, short enough."""
    try:
        _ = name.encode()
    except UnicodeEncodeError:
        return False
    return len(name) <= 4_096 and "\n" not in name and "\0" not in name


def _open_under(top: Path, name: str, *, stack: ExitStack) -> int:
    """Open ``top/name`` one part at a time, never through a link; return its fd."""
    parts = name.split("/")
    folder = os.open(top, _NO_FOLLOW | os.O_DIRECTORY)
    stack.callback(os.close, folder)
    for part in parts[:-1]:
        folder = os.open(part, _NO_FOLLOW | os.O_DIRECTORY, dir_fd=folder)
        stack.callback(os.close, folder)
    fd = os.open(parts[-1], _NO_FOLLOW, dir_fd=folder)
    stack.callback(os.close, fd)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        raise ValueError(f"{name!r} is not a regular file")
    return fd


def _chunk(data: bytes, *, offset: int, size: int, ended: bool) -> Chunk:
    """Build a chunk of the bytes read."""
    return Chunk(
        data_b64=base64.b64encode(data).decode(),
        offset=offset,
        size=size,
        ended=ended,
    )


def _write(path: Path, text: str) -> None:
    """Write ``path`` whole: a reader sees the old file or the new one, never half."""
    partial = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}")
    _ = partial.write_text(text)
    _ = partial.replace(path)


def _read[M: BaseModel](path: Path, model: type[M]) -> M | None:
    """Read one of the folder's records, or ``None`` when it is not there yet."""
    try:
        return model.model_validate_json(path.read_text())
    except FileNotFoundError:
        return None
