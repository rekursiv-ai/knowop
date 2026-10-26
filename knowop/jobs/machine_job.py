"""``knowop job`` on a computer of the person's own: queued by git, read back by relay.

``submit --machine NAME`` names the commit your branch was pushed at, and the machine
fetches it itself; no code passes through the server. So the worktree must be clean and
the branch's upstream must be at ``HEAD``: what runs is what was pushed. ``logs`` and
``results`` read the job's files from the machine through the server, at most 1 MiB a
request, while ``knowop connect --jobs`` runs there, and the server keeps none of them.
``results`` keeps :mod:`job_fetch`'s checks: every file is checked against the
machine's manifest, and a download that stopped resumes where it stopped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Protocol, override

import base64
import io
import tarfile

from trackinizer.client.errors import ClientError
from trackinizer.trax.render import echo
from trackinizer.wire.wire_machine_job import (
    MACHINE_TERMINAL_STATES,
    READ_HOLD_MAX_SEC,
    Chunk,
    MachineJobSubmit,
    ReadRefused,
    ResultManifest,
    branch_ref,
    remote_url,
)

from knowop.jobs import job_fetch
from knowop.jobs.job_source import Git, clean_env, plain_name


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from _typeshed import WriteableBuffer
    from trackinizer.wire.wire_machine_job import MachineJobView, ReadReply

    import pydantic
else:
    from wrapt import lazy_import

    # Only a submit that fails validation reads pydantic's error type.
    pydantic = lazy_import("pydantic")


__all__ = [
    "MachineJobClient",
    "Pushed",
    "pushed",
    "run_logs",
    "run_results",
    "run_submit",
]


class MachineJobClient(Protocol):
    """What a machine job needs of a client: ``Client``, or a narrower one."""

    def machine_job_submit(self, body: MachineJobSubmit) -> MachineJobView:
        """Queue a job on one of the caller's machines."""
        ...

    def machine_job_view(self, job_id: int) -> MachineJobView:
        """Return the job as it stands."""
        ...

    def machine_job_log(self, job_id: int, *, offset: int, wait: float) -> ReadReply:
        """Read the job's log from ``offset``, waiting up to ``wait`` seconds."""
        ...

    def machine_job_manifest(self, job_id: int) -> ReadReply:
        """Read the job's result manifest."""
        ...

    def machine_job_file(self, job_id: int, path: str, *, offset: int) -> ReadReply:
        """Read one result file from ``offset``."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class Pushed:
    """Where the commit a job runs was pushed."""

    remote: str
    """The remote's URL as this repository names it; the machine fetches from it."""

    ref: str
    """The upstream branch, ``refs/heads/NAME``."""

    commit: str
    branch: str
    """The local branch, for messages."""


def pushed(repo: Path) -> Pushed:
    """Return the pushed commit a job from ``repo`` runs, or refuse plainly.

    Args:
      repo: Any folder inside the repository.

    Returns:
      pushed: The remote, the upstream branch and the commit.

    Raises:
      ClientError: ``repo`` is in no repository, ``HEAD`` is detached, the branch has
        no upstream, the remote's URL is not one a job may name, a file differs from
        the commit, or the upstream is not at ``HEAD``.

    """
    found = Git(cwd=repo, env=clean_env()).run("rev-parse", "--show-toplevel")
    if found.returncode:
        raise ClientError("submit --machine runs from inside a git repository")
    git = Git(cwd=Path(found.stdout.decode().strip()), env=clean_env())
    branch = _config(git, "symbolic-ref", "--short", "HEAD")
    if not branch:
        raise ClientError("HEAD is detached; check out the branch you pushed")
    remote = _config(git, "config", f"branch.{branch}.remote")
    ref = _config(git, "config", f"branch.{branch}.merge")
    # The URL as configured, before any insteadOf: what the machine is told to fetch.
    url = _config(git, "config", f"remote.{remote}.url")
    if not url or not ref:
        raise ClientError(
            f"branch {branch} has no upstream on a remote; push it with "
            f"git push -u origin {branch}",
        )
    try:
        _ = remote_url(url)
        _ = branch_ref(ref)
    except ValueError as error:
        raise ClientError(
            f"{remote}/{branch} cannot be named in a job: {error}",
        ) from None
    _refuse_changes(git)
    head = git("rev-parse", "HEAD").strip()
    # Neither the URL nor the ref can start with ``-``, so git reads neither as an
    # option.
    listed = git("ls-remote", url, ref).split()
    tip = listed[0] if listed else ""
    if tip != head:
        raise ClientError(
            f"{remote}/{branch} is at {tip[:12] or 'nothing'} but HEAD is at "
            f"{head[:12]}; push HEAD first",
        )
    return Pushed(remote=url, ref=ref, commit=head, branch=branch)


def run_submit(
    client: MachineJobClient,
    *,
    repo: Path,
    machine: str,
    command: Sequence[str],
    gpus: int,
    memory_mb: int,
    minutes: int,
    label: str,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
    take_sec: float = 60.0,
) -> int:
    """Queue a job on one of the caller's machines; return its id once it is taken.

    Args:
      client: The server.
      repo: The folder the job's repository is in.
      machine: The machine's name.
      command: The job's command line.
      gpus: How many GPUs.
      memory_mb: The memory cap, in MiB.
      minutes: The wall-clock limit.
      label: A short name for the job.
      sleep: Waits between looks at the job.
      clock: Monotonic seconds.
      take_sec: How long to wait for the machine to take the job.

    Returns:
      job: The job's id.

    Raises:
      ClientError: The repository is not pushed as it stands, the job is not one the
        wire takes, the server refused it, or the machine did not take it in time.
      SystemExit: 1 when the job ended at once, as a fetch that failed ends it.

    """
    where = pushed(repo)
    echo(
        f"code: {where.commit[:12]} of {where.branch}, pushed to {where.remote}",
        err=True,
    )
    try:
        body = MachineJobSubmit.model_validate(
            {
                "machine": machine,
                "argv": list(command),
                "gpus": gpus,
                "memory_mb": memory_mb,
                "minutes": minutes,
                "label": label,
                "remote": where.remote,
                "ref": where.ref,
                "commit": where.commit,
            },
        )
    except pydantic.ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()
        )
        raise ClientError(f"invalid job: {problems}") from None
    view = client.machine_job_submit(body)
    echo(f"job {view.id}: requested on {machine}", err=True)
    deadline = clock() + take_sec
    while view.state == "requested":
        if clock() >= deadline:
            raise ClientError(
                f"job {view.id}: {machine} has not taken it in {take_sec:g} s; is "
                f"knowop connect --jobs running there? 'knowop job cancel {view.id}' "
                "withdraws it",
            )
        sleep(1)
        view = client.machine_job_view(view.id)
    echo(f"job {view.id}: {view.state} on {machine}", err=True)
    if view.state in MACHINE_TERMINAL_STATES and view.state != "completed":
        echo(
            f"job {view.id} {view.state}: {view.reason or 'no reason given'}",
            err=True,
        )
        raise SystemExit(1)
    return view.id


def run_logs(
    client: MachineJobClient,
    job: int,
    *,
    follow: bool,
    sink: IO[bytes],
) -> None:
    """Copy a machine job's log to ``sink``; with ``follow``, until the job ends.

    Args:
      client: The server.
      job: The job's id.
      follow: Keep reading until the job ends.
      sink: Receives the log's bytes.

    Raises:
      ClientError: The machine refused the read, or the server could not relay it.

    """
    offset = 0
    while True:
        chunk = _chunk(
            client.machine_job_log(
                job,
                offset=offset,
                wait=READ_HOLD_MAX_SEC if follow else 0,
            ).reply,
        )
        data = base64.b64decode(chunk.data_b64)
        _ = sink.write(data)
        offset += len(data)
        if offset >= chunk.size and (chunk.ended or not follow):
            return


def run_results(
    client: MachineJobClient,
    view: MachineJobView,
    *,
    results: Path,
) -> None:
    """Download a machine job's results into ``results/out``, checking every file.

    Args:
      client: The server.
      view: The job.
      results: The folder to fill; ``job.json`` is written beside ``out/``.

    Raises:
      ClientError: The machine refused the results, as it does past its limit, or
        the download failed a check or stopped.

    """
    listing = client.machine_job_manifest(view.id).reply
    if isinstance(listing, ReadRefused):
        raise ClientError(listing.reason)
    if not isinstance(listing, ResultManifest):
        raise ClientError("the machine answered a manifest with something else")
    sizes = {entry.path: entry.size for entry in listing.entries}
    done = job_fetch.fetch(
        _Relay(client, view.id, sizes),
        listing.entries,
        results=results,
    )
    _ = (results / "job.json").write_text(f"{view.model_dump_json(indent=2)}\n")
    echo(
        f"results: {results / 'out'} "
        f"({done.downloaded} downloaded, {done.skipped} already held)",
    )


def _config(git: Git, *args: str) -> str:
    """Return what git prints, or empty when it finds nothing."""
    done = git.run(*args)
    return "" if done.returncode else done.stdout.decode().strip()


def _refuse_changes(git: Git) -> None:
    """Refuse a worktree with a file that differs from its commit."""
    listed = git("status", "--porcelain=v1", "--untracked-files=normal", "-z")
    changed = [entry[3:] for entry in listed.split("\0") if len(entry) > 3]
    if not changed:
        return
    shown = ", ".join(plain_name(name) for name in changed[:10])
    more = len(changed) - 10
    names = f"{shown} and {more} more" if more > 0 else shown
    raise ClientError(
        f"{len(changed)} files differ from the pushed commit: {names}; a job on a "
        "machine runs what was pushed, so commit and push them first",
    )


def _chunk(reply: Chunk | ResultManifest | ReadRefused) -> Chunk:
    """Return the chunk a read answered, or raise the machine's refusal."""
    if isinstance(reply, ReadRefused):
        raise ClientError(reply.reason)
    if not isinstance(reply, Chunk):
        raise ClientError("the machine answered a read with something else")
    return reply


class _Relay:
    """The results' source for ``job_fetch``: a tar stream of files read by relay."""

    def __init__(
        self,
        client: MachineJobClient,
        job: int,
        sizes: dict[str, int],
    ) -> None:
        self._client = client
        self._job = job
        self._sizes = sizes

    def download(self, paths: Sequence[str], out: IO[bytes]) -> None:
        """Write a tar stream of the named result files to ``out``.

        Args:
          paths: Paths under the job's ``out/``.
          out: Receives the stream.

        """
        with tarfile.open(fileobj=out, mode="w|") as tar:
            for path in paths:
                info = tarfile.TarInfo(path)
                info.size = self._sizes[path]
                tar.addfile(info, _RemoteFile(self._client, self._job, path, info.size))


class _RemoteFile(io.RawIOBase):
    """One result file, fetched from the machine 1 MiB at a time as it is consumed."""

    def __init__(
        self,
        client: MachineJobClient,
        job: int,
        path: str,
        size: int,
    ) -> None:
        super().__init__()
        self._client = client
        self._job = job
        self._path = path
        self._size = size
        self._fetched = 0
        self._pending = memoryview(b"")

    @override
    def readable(self) -> bool:
        return True

    @override
    def readinto(self, buffer: WriteableBuffer, /) -> int:
        if not self._pending:
            if self._fetched >= self._size:
                return 0
            reply = self._client.machine_job_file(
                self._job,
                self._path,
                offset=self._fetched,
            ).reply
            data = base64.b64decode(_chunk(reply).data_b64)
            if not data:
                raise ClientError(f"{self._path} ended before its {self._size} bytes")
            self._pending = memoryview(data)
            self._fetched += len(data)
        view = memoryview(buffer).cast("B")
        count = min(len(view), len(self._pending))
        view[:count] = self._pending[:count]
        self._pending = self._pending[count:]
        return count
