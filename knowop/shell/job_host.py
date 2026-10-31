"""Run the jobs the server hands this machine: fetch, cap, start, watch, report, read.

A machine host started with ``knowop connect --jobs`` holds a wait on the server for work
(``wire_machine_job``). A start is taken only for the account this host's credential
belongs to: the job's folder is made (``job_folder.py``) and the job reported ``queued``
at once, before its code is fetched (``job_repo.py``), so the server hands it out no
more. Once its GPUs are idle and its cap fits the machine's budget, the job starts in a
capped scope of its own (``systemd-run --user --scope``, ``job_runner.py``), is reported
``running``, and is watched until its runner records its end; the host then reports the
end, kills whatever the job left in its scope, and removes its worktree. A cancel stops
the scope: SIGTERM, then SIGKILL ten seconds later. A read is answered from the job's
folder, on a thread of its own.

A job outlives its host. Stopping ``knowop connect`` leaves a running job running, and
the next host started on this computer adopts every job whose end it has not reported.
A revoked machine stops them all. A job's folder is deleted 14 days after its end is
reported.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, Protocol

import hashlib
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

from trackinizer.client.errors import ClientError
from trackinizer.trax.render import echo
from trackinizer.wire.wire_machine_job import (
    WAIT_MAX_SEC,
    CancelWork,
    JobRecovery,
    JobReport,
    MachineJobSpec,
    ReadRefused,
    ReadReply,
    ReadWork,
    StartWork,
)

from knowop.shell import job_runner
from knowop.shell.job_folder import (
    Ended,
    JobBinding,
    JobFolder,
    Launch,
)
from knowop.shell.job_limits import Run, run, scope_argv
from knowop.shell.job_repo import FetchError
from knowop.shell.trax_context import inherited_trax_context


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    import uuid

    from trackinizer.wire.wire_machine_job import (
        Chunk,
        ResultManifest,
        WorkOffer,
    )


__all__ = ["JobClient", "JobHost", "Launcher", "SystemdLauncher", "Worktrees"]

_STOP_GRACE_SEC: Final = 10.0
"""How long a cancelled or revoked job has after SIGTERM before SIGKILL."""

_CANCELLED: Final = JobReport(state="cancelled", reason="Cancelled.")
_logger = logging.getLogger(__name__)


class JobClient(Protocol):
    """The calls the job host makes: the ``Client`` methods, and closing it."""

    def wait_machine_work(
        self,
        machine_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        wait_sec: float,
    ) -> WorkOffer:
        """Hold for the next piece of work."""
        ...

    def report_machine_job(
        self,
        machine_id: uuid.UUID,
        job_id: int,
        report: JobReport,
    ) -> None:
        """Report one of the machine's jobs."""
        ...

    def post_machine_read(
        self,
        machine_id: uuid.UUID,
        read_id: uuid.UUID,
        reply: ReadReply,
    ) -> None:
        """Answer a read."""
        ...

    def close(self) -> None:
        """Release the connection."""
        ...


class Worktrees(Protocol):
    """Fetches a job's commit into its worktree, and deletes the worktree."""

    def checkout(
        self,
        *,
        remote: str,
        ref: str,
        commit: str,
        dest: Path,
        cancel: threading.Event | None = None,
    ) -> None:
        """Make ``dest`` a worktree of ``commit``; raise ``FetchError`` saying why not."""
        ...

    def remove(self, *, remote: str, dest: Path) -> None:
        """Delete a worktree made from ``remote``."""
        ...


class Launcher(Protocol):
    """Starts a job's runner in a capped scope, and asks and tells that scope."""

    def start(
        self,
        folder: JobFolder,
        *,
        unit: str,
        memory_mb: int,
        env: Mapping[str, str],
    ) -> None:
        """Start the job's runner in a new scope named ``unit``."""
        ...

    def alive(self, unit: str) -> bool:
        """Say whether anything still runs in the scope."""
        ...

    def kill(self, unit: str, signum: int) -> None:
        """Send every process in the scope ``signum``."""
        ...

    def timed_out(self, unit: str) -> bool:
        """Say whether the scope's own deadline stopped it."""
        ...


class SystemdLauncher:
    """Starts runners in user scopes, and asks systemd about a scope it did not start.

    Args:
      run: Runs ``systemctl``.

    """

    def __init__(self, *, run: Run = run) -> None:
        self._run = run
        self._children: dict[str, subprocess.Popen[bytes]] = {}

    def start(
        self,
        folder: JobFolder,
        *,
        unit: str,
        memory_mb: int,
        env: Mapping[str, str],
    ) -> None:
        """Start ``job_runner`` on the folder, capped, in a session of its own.

        Args:
          folder: The job's folder; the runner starts there.
          unit: The scope's name.
          memory_mb: The cap, in MiB.
          env: The job's environment.

        """
        argv = scope_argv(
            unit=unit,
            memory_mb=memory_mb,
            runtime_sec=folder.spec().minutes * 60 + _STOP_GRACE_SEC,
            argv=[sys.executable, "-m", job_runner.__name__, str(folder.path)],
        )
        with folder.log.open("ab") as log:
            # A session of its own: a Ctrl-C to ``connect`` reaches its process group,
            # which the job must outlive.
            self._children[unit] = subprocess.Popen(  # noqa: S603 -- fixed argv.
                argv,
                cwd=folder.path,
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    def alive(self, unit: str) -> bool:
        """Say whether the runner, or anything else in its scope, still runs.

        Args:
          unit: The scope's name.

        Returns:
          alive: For a runner this host started, whether it has not exited; for one
            an earlier host started, whether systemd says its scope is active.

        """
        child = self._children.get(unit)
        if child is not None:
            if child.poll() is None:
                return True
            del self._children[unit]
            return False
        code, _ = self._run(
            ["systemctl", "--user", "is-active", "--quiet", f"{unit}.scope"],
        )
        return code == 0

    def kill(self, unit: str, signum: int) -> None:
        """Send every process in the scope ``signum``; a scope that is gone is fine."""
        _ = self._run(
            ["systemctl", "--user", "kill", f"--signal={signum}", f"{unit}.scope"],
        )

    def timed_out(self, unit: str) -> bool:
        """Read systemd's retained result when its deadline killed the runner.

        Args:
          unit: The scope's name.

        Returns:
          timed_out: Whether systemd recorded a scope timeout.

        """
        code, result = self._run(
            [
                "systemctl",
                "--user",
                "show",
                "--property=Result",
                "--value",
                f"{unit}.scope",
            ],
        )
        return code == 0 and result.strip() == "timeout"


type _Phase = Literal["waiting", "fetching", "ready", "running", "cleaning", "ended"]


@dataclass(slots=True, kw_only=True)
class _Job:
    """One job this host holds, and where it stands."""

    folder: JobFolder
    spec: MachineJobSpec
    phase: _Phase = "waiting"
    launch: Launch | None = None
    cancel: bool = False
    preparation_cancel: threading.Event = field(default_factory=threading.Event)
    fetch_thread: threading.Thread | None = None
    cleanup_thread: threading.Thread | None = None
    ending: JobReport | None = None
    recovering: bool = False
    recovery_at: float = 0.0
    needs_cleanup: bool = False
    failure: JobReport | None = None
    """Why the job ended before it started, as its fetch found."""

    stop_sent: float | None = None
    """When the host sent the scope SIGTERM."""

    pending: JobReport | None = None
    """A report the server has not taken yet."""

    waiting_for: str = ""


class JobHost:
    """Runs the jobs the server hands this machine; see the module docstring.

    Args:
      name: The machine's name, for the log lines and refusals.
      server_url: The server's URL, which the scopes' names are told apart by.
      credential: The machine credential, which the client sends.
      credential_client: Returns a client that sends the given credential.
      account: The account whose jobs this machine runs.
      root: The folder that holds a folder per job.
      repos: Fetches a job's commit into its worktree, and deletes it.
      launcher: Starts and stops the scopes.
      gpus: The indexes of the GPUs this machine offers jobs.
      idle_gpus: Returns the indexes of the GPUs idle now.
      memory_budget_mb: The most memory the running jobs may hold together.
      results_max: The most result bytes one job sends back.
      clock: Epoch seconds.
      retention_sec: How long a job's folder is kept after its end is reported.
      tick_sec: How often the jobs are looked at.
      retry_sec: How long to wait after a failed wait or answer.

    """

    def __init__(
        self,
        *,
        name: str,
        server_url: str,
        credential: str,
        credential_client: Callable[[str], JobClient],
        account: str,
        root: Path,
        repos: Worktrees,
        launcher: Launcher,
        gpus: Sequence[int],
        idle_gpus: Callable[[], Sequence[int]],
        memory_budget_mb: int,
        results_max: int,
        clock: Callable[[], float] = time.time,
        retention_sec: float = 14 * 86_400,
        tick_sec: float = 1.0,
        retry_sec: float = 5.0,
    ) -> None:
        self._name = name
        self._server_url = server_url
        self._tag = hashlib.sha256(server_url.encode()).hexdigest()[:8]
        self._credential = credential
        self._client_for = credential_client
        self._account = account.lower()
        self._root = root
        self._repos = repos
        self._launcher = launcher
        self._gpus = frozenset(gpus)
        self._idle_gpus = idle_gpus
        self._budget_mb = memory_budget_mb
        self._results_max = results_max
        self._clock = clock
        self._retention_sec = retention_sec
        self._tick_sec = tick_sec
        self._retry_sec = retry_sec
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._read_slots = threading.BoundedSemaphore(4)
        self._fetch_slots = threading.BoundedSemaphore(1)
        self._cleanup_slots = threading.BoundedSemaphore(1)
        self._jobs: dict[int, _Job] = {}
        self._machine: tuple[uuid.UUID, uuid.UUID] | None = None
        self._swept_at = 0.0
        self._scheduler: threading.Thread | None = None

    def start(self, machine_id: uuid.UUID, instance: uuid.UUID) -> None:
        """Adopt the jobs an earlier host left, then wait for work and run the jobs.

        Args:
          machine_id: This machine's id.
          instance: This host's instance.

        """
        self._machine = (machine_id, instance)
        self._adopt()
        self.sweep()
        threading.Thread(name="jobs-listener", target=self._listen, daemon=True).start()
        self._scheduler = threading.Thread(
            name="jobs",
            target=self._schedule,
            daemon=True,
        )
        self._scheduler.start()

    def stop(self) -> None:
        """Stop waiting for work and leave the running jobs running, unless revoked.

        The listener is left to its held wait, whose answer it drops. A machine
        revoked while its host waited learns of it here, and stops its jobs.

        """
        self._stop.set()
        # Worker joins may wait on repository locks; learn revocation before them.
        try:
            _ = self._wait(wait_sec=0)
        except ClientError as error:
            if error.status_code == 410:
                self._revoked()
        finally:
            self._stop_preparation()
            if self._scheduler is not None:
                self._scheduler.join(timeout=_STOP_GRACE_SEC)

    def handle(self, work: StartWork | CancelWork | ReadWork) -> None:
        """Act on one piece of work the server handed this host.

        Args:
          work: A job to take, a job to cancel, or a read to answer.

        """
        match work:
            case StartWork():
                self._accept(work.job)
            case CancelWork():
                self._cancel(work.job_id)
            case ReadWork():
                if not self._read_slots.acquire(blocking=False):
                    return
                threading.Thread(
                    name=f"jobs-read-{work.read_id}",
                    target=self._answer_bounded,
                    args=(work,),
                    daemon=True,
                ).start()

    def step(self) -> None:
        """Move every job on as far as it can go now, and sweep once an hour."""
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda job: job.spec.id)
        for job in jobs:
            self._advance(job)
        if self._clock() - self._swept_at >= 3_600.0:
            self.sweep()

    def sweep(self) -> None:
        """Delete the folders of the jobs whose end was reported over 14 days ago."""
        self._swept_at = self._clock()
        if not self._root.exists():
            return
        for path in self._root.iterdir():
            folder = JobFolder(path)
            reported = (
                folder.reported_at()
                if path.name.isdigit() and self._belongs(folder)
                else None
            )
            if reported is not None and self._clock() - reported >= self._retention_sec:
                shutil.rmtree(path, ignore_errors=True)

    def _listen(self) -> None:
        """Wait for work until stopped or revoked, acting on each piece."""
        while not self._stop.is_set():
            try:
                offer = self._wait(wait_sec=WAIT_MAX_SEC)
            except ClientError as error:
                if error.status_code == 410:
                    self._revoked()
                    return
                echo(f"knowop: job wait failed: {error}", err=True)
                _ = self._stop.wait(self._retry_sec)
                continue
            # Work that came as the host stopped is left to its lease: the server hands
            # it to the next host.
            if offer.work is not None and not self._stop.is_set():
                self.handle(offer.work)

    def _schedule(self) -> None:
        """Move the jobs on every tick until stopped."""
        while not self._stop.wait(self._tick_sec):
            self.step()

    def _wait(self, *, wait_sec: float) -> WorkOffer:
        """Hold for the next piece of work."""
        machine_id, instance = self._where()
        with closing(self._client_for(self._credential)) as client:
            return client.wait_machine_work(
                machine_id,
                instance=instance,
                wait_sec=wait_sec,
            )

    def _where(self) -> tuple[uuid.UUID, uuid.UUID]:
        """Return this machine's id and this host's instance."""
        if self._machine is None:
            raise RuntimeError("Expected the host to be started.")
        return self._machine

    def _accept(self, spec: MachineJobSpec) -> None:
        """Take a job: make its folder and report it queued, or refuse it."""
        with self._lock:
            held = spec.id in self._jobs
        if held:
            # The server handed it again: the answer to the first report was lost.
            _ = self._send(spec.id, JobReport(state="queued"))
            return
        refusal = self._refusal(spec)
        if refusal:
            _ = self._send(spec.id, JobReport(state="failed", reason=refusal))
            return
        folder = JobFolder.create(self._root / str(spec.id), spec)
        folder.bind(self._binding())
        folder.note(f"accepted on {self._name}")
        echo(f"job {spec.id}: accepted ({' '.join(spec.argv)[:60]})")
        job = _Job(folder=folder, spec=spec)
        self._tell(job, JobReport(state="queued"))
        with self._lock:
            self._jobs[spec.id] = job

    def _refusal(self, spec: MachineJobSpec) -> str:
        """Return why this machine will not run ``spec``; empty when it will."""
        if spec.owner.lower() != self._account:
            return f"{self._name} runs only {self._account}'s jobs"
        if (self._root / str(spec.id)).exists():
            return f"job {spec.id} already has a folder on {self._name}"
        if spec.gpus > len(self._gpus):
            return f"{self._name} offers {len(self._gpus)} GPUs to jobs"
        if spec.memory_mb > self._budget_mb:
            return f"{self._name} lets its jobs hold {self._budget_mb} MiB"
        return ""

    def _cancel(self, job_id: int) -> None:
        """Mark a job to be stopped; one this host does not hold is told cancelled."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.cancel = True
                job.preparation_cancel.set()
        if job is None:
            _ = self._send(job_id, _CANCELLED)

    def _advance(self, job: _Job) -> None:
        """Move one job on as far as it can go now."""
        if job.phase == "cleaning":
            self._clean(job)
            return
        if job.recovering and not job.cancel and not self._recover(job):
            return
        if job.pending is not None:
            if not self._send(job.spec.id, job.pending) and not (
                job.cancel and job.phase == "waiting"
            ):
                return
            job.pending = None
            if job.phase == "ended":
                self._forget(job)
        match job.phase:
            case "waiting" if job.cancel:
                self._finish(job, _CANCELLED)
            case "waiting":
                if not self._fetch_slots.acquire(blocking=False):
                    return
                job.phase = "fetching"
                job.fetch_thread = threading.Thread(
                    name=f"jobs-fetch-{job.spec.id}",
                    target=self._fetch,
                    args=(job,),
                    daemon=True,
                )
                job.fetch_thread.start()
            case "ready" if job.cancel:
                self._finish(job, _CANCELLED)
            case "ready" if job.failure is not None:
                self._finish(job, job.failure)
            case "ready":
                self._launch(job)
            case "running":
                self._watch(job)
            case _:
                pass

    def _fetch(self, job: _Job) -> None:
        """Fetch the job's commit into its worktree, on a thread of its own."""
        spec = job.spec
        failure = JobReport(state="failed", reason="Preparation did not complete.")
        try:
            job.folder.note(
                f"fetching {spec.commit[:12]} of {spec.ref} from {spec.remote}",
            )
            if job.needs_cleanup:
                self._repos.remove(remote=spec.remote, dest=job.folder.src)
            self._repos.checkout(
                remote=spec.remote,
                ref=spec.ref,
                commit=spec.commit,
                dest=job.folder.src,
                cancel=job.preparation_cancel,
            )
            job.folder.note("worktree ready")
            failure = None
        except FetchError as error:
            failure = JobReport(state="failed", reason=str(error)[:1_000])
        except Exception as error:
            _logger.exception("Job %s preparation failed", spec.id)
            failure = JobReport(state="failed", reason=str(error)[:1_000])
        finally:
            with self._lock:
                job.failure = failure
                job.phase = "ready"
            self._fetch_slots.release()

    def _launch(self, job: _Job) -> None:
        """Start the job in its scope once its GPUs and its memory are free."""
        spec = job.spec
        with self._lock:
            launches = [
                each.launch
                for each in self._jobs.values()
                if each.phase == "running" and each.launch is not None
            ]
        held_mb = sum(launch.memory_mb for launch in launches)
        held = {gpu for launch in launches for gpu in launch.gpus}
        free = sorted(set(self._idle_gpus()) & self._gpus - held) if spec.gpus else []
        if len(free) < spec.gpus:
            self._wait_for(job, f"waiting for {spec.gpus} idle GPUs")
            return
        if held_mb + spec.memory_mb > self._budget_mb:
            self._wait_for(job, f"waiting for {spec.memory_mb} MiB of memory")
            return
        if not self._send(spec.id, JobReport(state="running")):
            return
        launch = Launch(
            unit=f"knowop-job-{self._tag}-{spec.id}",
            gpus=free[: spec.gpus],
            memory_mb=spec.memory_mb,
        )
        job.folder.write_launch(launch)
        job.folder.note(
            f"starting in {launch.unit}.scope under a {spec.memory_mb} MiB cap, "
            f"GPUs: {','.join(map(str, launch.gpus)) or 'none'}",
        )
        try:
            self._launcher.start(
                job.folder,
                unit=launch.unit,
                memory_mb=spec.memory_mb,
                env=self._env(job, launch.gpus),
            )
        except OSError as error:
            self._finish(
                job,
                JobReport(state="failed", reason=f"the scope did not start: {error}"),
            )
            return
        with self._lock:
            job.launch = launch
            job.phase = "running"
        echo(f"job {spec.id}: running in {launch.unit}.scope")

    def _wait_for(self, job: _Job, what: str) -> None:
        """Say in the job's log what it waits for, once each time that changes."""
        if job.waiting_for != what:
            job.waiting_for = what
            job.folder.note(what)

    def _watch(self, job: _Job) -> None:
        """Report a job whose runner recorded its end; carry a cancel through."""
        launch = job.launch
        if launch is None:
            raise ValueError("Expected a running job to have a launch.")
        if job.cancel:
            self._stop_scope(job, launch)
            return
        ended = job.folder.ended()
        if ended is None and not self._launcher.alive(launch.unit):
            # The runner may have written its end just before it exited.
            ended = job.folder.ended()
            if ended is None and self._launcher.timed_out(launch.unit):
                ended = Ended(
                    state="timeout",
                    exit_code=124,
                    reason=f"stopped at its {job.spec.minutes} min scope limit",
                )
                job.folder.write_ended(ended)
            if ended is None:
                self._finish(
                    job,
                    JobReport(
                        state="failed",
                        reason="the job ended without recording how",
                    ),
                )
                return
        if ended is not None:
            self._finish(
                job,
                JobReport(
                    state=ended.state,
                    exit_code=ended.exit_code,
                    reason=ended.reason,
                ),
            )

    def _stop_scope(self, job: _Job, launch: Launch) -> None:
        """SIGTERM a cancelled job's scope, SIGKILL it after the grace, then report."""
        if job.stop_sent is None:
            job.stop_sent = self._clock()
            job.folder.note("cancelled; stopping")
            self._launcher.kill(launch.unit, signal.SIGTERM)
        elif not self._launcher.alive(launch.unit):
            ended = job.folder.ended()
            self._finish(
                job,
                JobReport(
                    state="cancelled",
                    exit_code=None if ended is None else ended.exit_code,
                    reason="Cancelled.",
                ),
            )
        elif self._clock() - job.stop_sent >= _STOP_GRACE_SEC:
            self._launcher.kill(launch.unit, signal.SIGKILL)

    def _finish(self, job: _Job, report: JobReport) -> None:
        """End a job here: kill what it left, drop its worktree, and report its end."""
        if job.launch is not None:
            self._launcher.kill(job.launch.unit, signal.SIGKILL)
        with self._lock:
            job.ending = report
            job.phase = "cleaning"
        job.pending = None if self._send(job.spec.id, report) else report
        self._clean(job)

    def _clean(self, job: _Job) -> None:
        """Start at most one cleanup worker, leaving the scheduler free to cancel."""
        if job.cleanup_thread is not None and job.cleanup_thread.is_alive():
            return
        if not self._cleanup_slots.acquire(blocking=False):
            return
        job.cleanup_thread = threading.Thread(
            name=f"jobs-clean-{job.spec.id}",
            target=self._cleanup,
            args=(job,),
            daemon=True,
        )
        job.cleanup_thread.start()

    def _cleanup(self, job: _Job) -> None:
        """Remove a stopped job's worktree, then retire it once its report landed."""
        try:
            self._repos.remove(remote=job.spec.remote, dest=job.folder.src)
            report = job.ending
            assert isinstance(report, JobReport)
            job.folder.note(f"{report.state}: {report.reason or 'exit 0'}")
            echo(f"job {job.spec.id}: {report.state}")
            if job.pending is None:
                self._forget(job)
            else:
                with self._lock:
                    job.phase = "ended"
        except Exception:
            _logger.exception("Job %s cleanup failed; will retry", job.spec.id)
        finally:
            self._cleanup_slots.release()

    def _forget(self, job: _Job) -> None:
        """Note that the job's end was reported, and stop holding it."""
        job.folder.mark_reported(self._clock())
        with self._lock:
            job.phase = "ended"
            _ = self._jobs.pop(job.spec.id, None)

    def _tell(self, job: _Job, report: JobReport) -> None:
        """Report a job's state now, or keep it to report again."""
        if not self._send(job.spec.id, report):
            job.pending = report

    def _send(self, job_id: int, report: JobReport) -> bool:
        """Require affirmative authorization for starts; retire rejected end reports."""
        machine_id, _ = self._where()
        try:
            with closing(self._client_for(self._credential)) as client:
                client.report_machine_job(machine_id, job_id, report)
        except ClientError as error:
            if report.state not in {"queued", "running"} and error.status_code in {
                403,
                404,
                409,
                410,
            }:
                return True
            echo(f"knowop: job {job_id} report not delivered: {error}", err=True)
            return False
        return True

    def _env(self, job: _Job, gpus: Sequence[int]) -> dict[str, str]:
        """Return the job's environment: the host's, less its trax context, and its own."""
        withheld = inherited_trax_context()
        return {
            **{k: v for k, v in os.environ.items() if k not in withheld},
            "KNOWOP_JOB_ID": str(job.spec.id),
            "KNOWOP_JOB_OUT": str(job.folder.out),
            # Empty hides every GPU, so a job that asked for none takes none.
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
        }

    def _answer_bounded(self, work: ReadWork) -> None:
        """Hold a host read slot through reading and posting, including failures."""
        try:
            self._answer(work)
        finally:
            self._read_slots.release()

    def _answer(self, work: ReadWork) -> None:
        """Answer a read from the job's folder, and send the answer."""
        reply = ReadReply(reply=self._read(work))
        machine_id, _ = self._where()
        for attempt in range(3):
            try:
                with closing(self._client_for(self._credential)) as client:
                    client.post_machine_read(machine_id, work.read_id, reply)
            except ClientError as error:
                if error.status_code is not None and 400 <= error.status_code < 500:
                    return
                if attempt == 2:
                    echo(f"knowop: read not delivered: {error}", err=True)
                    return
                _ = self._stop.wait(self._retry_sec)
            else:
                return

    def _read(self, work: ReadWork) -> Chunk | ResultManifest | ReadRefused:
        """Read what ``work`` asks of a job of this host's account."""
        folder = JobFolder(self._root / str(work.job_id))
        if not self._belongs(folder):
            return ReadRefused(reason=f"job {work.job_id} is not on {self._name}")
        limit, where = self._results_max, self._name
        try:
            match work.what:
                case "log":
                    return self._log(folder, work)
                case "manifest":
                    return folder.manifest(results_max=limit, where=where)
                case _:
                    return folder.read_file(
                        work.path,
                        work.offset,
                        results_max=limit,
                        where=where,
                    )
        except OSError as error:
            return ReadRefused(
                reason=f"job {work.job_id}'s files could not be read: {error.strerror}",
            )

    def _log(self, folder: JobFolder, work: ReadWork) -> Chunk:
        """Read the log, holding up to the read's wait for bytes past its offset."""
        deadline = self._clock() + work.wait_sec
        while True:
            chunk = folder.read_log(work.offset)
            if (
                chunk.size > work.offset
                or chunk.ended
                or self._clock() >= deadline
                or self._stop.wait(0.5)
            ):
                return chunk

    def _adopt(self) -> None:
        """Hold again every job an earlier host left with its end unreported."""
        if not self._root.exists():
            return
        for path in sorted(self._root.iterdir()):
            folder = JobFolder(path)
            if (
                not path.name.isdigit()
                or not (path / "spec.json").exists()
                or not self._belongs(folder)
                or folder.reported_at() is not None
            ):
                continue
            spec = folder.spec()
            launch = folder.launch()
            if launch is not None:
                job = _Job(
                    folder=folder,
                    spec=spec,
                    launch=launch,
                    phase="running",
                    recovering=True,
                )
                self._jobs[spec.id] = job
                self._recover(job)
                continue
            if not self._send(
                spec.id,
                JobRecovery(state="queued"),
            ):
                continue
            folder.note(f"adopted by a new host on {self._name}")
            self._jobs[spec.id] = _Job(
                folder=folder,
                spec=spec,
                pending=JobReport(state="queued"),
                needs_cleanup=True,
            )

    def _recover(self, job: _Job) -> bool:
        """Keep a bound scope accounted until authorized, or stop it on refusal."""
        if self._clock() < job.recovery_at:
            return False
        machine_id, _ = self._where()
        try:
            with closing(self._client_for(self._credential)) as client:
                client.report_machine_job(
                    machine_id,
                    job.spec.id,
                    JobRecovery(state="running"),
                )
        except ClientError as error:
            if error.status_code in {403, 404, 410}:
                if job.launch is not None:
                    self._launcher.kill(job.launch.unit, signal.SIGKILL)
                job.cancel = True
                job.recovering = False
                return True
            job.recovery_at = self._clock() + self._retry_sec
            return False
        job.recovering = False
        job.folder.note(f"adopted by a new host on {self._name}")
        return True

    def _binding(self) -> JobBinding:
        """Return this connection's complete persisted identity."""
        machine_id, _ = self._where()
        return JobBinding(
            server=self._server_url,
            account=self._account,
            machine_id=machine_id,
        )

    def _belongs(self, folder: JobFolder) -> bool:
        """Refuse foreign and legacy folders before any mutation or file relay."""
        return (
            folder.binding() == self._binding()
            and folder.spec().owner.lower() == self._account
        )

    def _revoked(self) -> None:
        """Stop every job of a revoked machine: SIGTERM, then SIGKILL after the grace."""
        self._stop.set()
        with self._lock:
            jobs = list(self._jobs.values())
        launched = [job.launch.unit for job in jobs if job.launch is not None]
        for unit in launched:
            self._launcher.kill(unit, signal.SIGTERM)
        deadline = time.monotonic() + _STOP_GRACE_SEC
        self._stop_preparation()
        with self._lock:
            self._jobs.clear()
        while any(self._launcher.alive(unit) for unit in launched):
            if time.monotonic() >= deadline:
                for unit in launched:
                    self._launcher.kill(unit, signal.SIGKILL)
                break
            time.sleep(0.2)
        for job in jobs:
            self._repos.remove(remote=job.spec.remote, dest=job.folder.src)
            job.folder.note("the machine was revoked; the job was stopped")
            job.folder.mark_reported(self._clock())
        echo(f"jobs: {len(jobs)} stopped, the machine was revoked")

    def _stop_preparation(self) -> None:
        """Stop fetches before shutdown or revocation touches their worktrees."""
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            job.preparation_cancel.set()
        for job in jobs:
            if job.fetch_thread is not None:
                job.fetch_thread.join(timeout=_STOP_GRACE_SEC)
            if job.cleanup_thread is not None:
                job.cleanup_thread.join(timeout=_STOP_GRACE_SEC)
