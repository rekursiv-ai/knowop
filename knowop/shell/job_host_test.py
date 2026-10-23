"""The job host: which jobs it takes, how it starts, watches and stops them, and reads."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import base64
import fcntl
import hashlib
import os
import signal
import subprocess
import threading
import time
import uuid

from trackinizer.client.errors import ClientError
from trackinizer.wire.wire_machine_job import (
    CancelWork,
    Chunk,
    JobRecovery,
    JobReport,
    MachineJobSpec,
    ReadRefused,
    ReadReply,
    ReadWork,
    ResultManifest,
    StartWork,
    WorkOffer,
)

import pytest

from knowop.shell import job_host
from knowop.shell.job_folder import (
    Ended,
    JobBinding,
    JobFolder,
    Launch,
)
from knowop.shell.job_host import JobHost, SystemdLauncher
from knowop.shell.job_repo import FetchError, Repos


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence


_JAN = "jan@example.com"
_URL = "https://alpha.knowop.ai"
_TAG = hashlib.sha256(_URL.encode()).hexdigest()[:8]
_MACHINE = uuid.UUID(int=1)
_INSTANCE = uuid.UUID(int=2)


def _spec(job_id: int = 7, /, **fields: object) -> MachineJobSpec:
    return MachineJobSpec.model_validate(
        {
            "id": job_id,
            "owner": _JAN,
            "argv": ["python", "train.py"],
            "remote": "https://github.com/o/r.git",
            "ref": "refs/heads/main",
            "commit": "a" * 40,
            "gpus": 0,
            "memory_mb": 1_024,
            "minutes": 5,
            "label": "",
        }
        | fields,
    )


class _Server:
    """The server's three host calls, scripted, recording every report and answer."""

    def __init__(self) -> None:
        self.machine_id = _MACHINE
        self.offers: list[WorkOffer | ClientError] = []
        self.reports: list[tuple[int, JobReport]] = []
        self.replies: list[tuple[uuid.UUID, ReadReply]] = []
        self.report_errors: list[ClientError] = []
        self.reply_errors: list[ClientError] = []
        self.stop_answer: WorkOffer | ClientError = WorkOffer()
        self.idle = threading.Event()

    def wait_machine_work(
        self,
        machine_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        wait_sec: float,
    ) -> WorkOffer:
        assert (machine_id, instance) == (self.machine_id, _INSTANCE)
        if wait_sec == 0:
            return _answer(self.stop_answer)
        if self.offers:
            return _answer(self.offers.pop(0))
        _ = self.idle.wait(0.05)
        return WorkOffer()

    def report_machine_job(
        self,
        machine_id: uuid.UUID,
        job_id: int,
        report: JobReport,
    ) -> None:
        assert machine_id == self.machine_id
        if self.report_errors:
            raise self.report_errors.pop(0)
        if not isinstance(report, JobRecovery):
            self.reports.append((job_id, report))

    def post_machine_read(
        self,
        machine_id: uuid.UUID,
        read_id: uuid.UUID,
        reply: ReadReply,
    ) -> None:
        assert machine_id == self.machine_id
        if self.reply_errors:
            raise self.reply_errors.pop(0)
        self.replies.append((read_id, reply))

    def close(self) -> None:
        pass

    def states(self) -> list[tuple[int, str]]:
        return [(job_id, report.state) for job_id, report in self.reports]


def _answer(found: WorkOffer | ClientError) -> WorkOffer:
    if isinstance(found, ClientError):
        raise found
    return found


@dataclass(slots=True, kw_only=True)
class _Launcher:
    """Scopes held in memory: started, alive until killed, unless told to ignore TERM."""

    started: list[tuple[str, int, dict[str, str]]] = field(default_factory=list)
    signals: list[tuple[str, int]] = field(default_factory=list)
    alive_units: set[str] = field(default_factory=set)
    timed_out_units: set[str] = field(default_factory=set)
    stubborn: bool = False
    fail: bool = False

    def start(
        self,
        folder: JobFolder,
        *,
        unit: str,
        memory_mb: int,
        env: Mapping[str, str],
    ) -> None:
        del folder
        if self.fail:
            raise OSError(2, "No such file or directory", "systemd-run")
        self.started.append((unit, memory_mb, dict(env)))
        self.alive_units.add(unit)

    def alive(self, unit: str) -> bool:
        return unit in self.alive_units

    def timed_out(self, unit: str) -> bool:
        return unit in self.timed_out_units

    def kill(self, unit: str, signum: int) -> None:
        self.signals.append((unit, signum))
        if signum == signal.SIGKILL or not self.stubborn:
            self.alive_units.discard(unit)


@dataclass(slots=True, kw_only=True)
class _Repos:
    """Worktrees made as a folder with one file; a fetch or a removal may fail."""

    failure: str = ""
    fetching: list[Path] = field(default_factory=list)
    removed: list[Path] = field(default_factory=list)
    gate: threading.Event | None = None
    removals_failing: int = 0

    def checkout(
        self,
        *,
        remote: str,
        ref: str,
        commit: str,
        dest: Path,
        cancel: threading.Event | None = None,
    ) -> None:
        del remote, ref
        self.fetching.append(dest)
        if self.gate is not None:
            deadline = time.monotonic() + 5
            while not self.gate.wait(0.01) and time.monotonic() < deadline:
                if cancel is not None and cancel.is_set():
                    raise FetchError("preparation cancelled")
        if self.failure:
            raise FetchError(self.failure)
        dest.mkdir()
        _ = (dest / "COMMIT").write_text(commit)

    def remove(self, *, remote: str, dest: Path) -> None:
        del remote
        self.removed.append(dest)
        if self.removals_failing:
            self.removals_failing -= 1
            raise OSError(16, "Device or resource busy", str(dest))
        if dest.exists():
            for each in dest.iterdir():
                each.unlink()
            dest.rmdir()


@dataclass(slots=True, kw_only=True)
class _Clock:
    now: float = 1_700_000_000.0

    def __call__(self) -> float:
        return self.now


@dataclass(slots=True, kw_only=True)
class _Rig:
    host: JobHost
    server: _Server
    launcher: _Launcher
    repos: _Repos
    clock: _Clock
    root: Path


def test_preparation_has_only_one_active_fetch(tmp_path: Path) -> None:
    gate = threading.Event()
    repos = _Repos(gate=gate)
    rig = _host(tmp_path, repos=repos)
    rig.host.start(_MACHINE, _INSTANCE)
    try:
        for job_id in range(7, 15):
            rig.host.handle(StartWork(job=_spec(job_id)))
        rig.host.step()
        _until(lambda: bool(repos.fetching))
        time.sleep(0.05)
        assert len(repos.fetching) == 1
    finally:
        gate.set()
        rig.host.stop()


def test_cancellation_interrupts_preparation(tmp_path: Path) -> None:
    gate = threading.Event()
    repos = _Repos(gate=gate)
    rig = _host(tmp_path, repos=repos)
    rig.host.start(_MACHINE, _INSTANCE)
    try:
        _start(rig)
        rig.host.step()
        _until(lambda: bool(repos.fetching))
        rig.host.handle(CancelWork(job_id=7))
        deadline = time.monotonic() + 1
        while _phase(rig, 7) == "fetching" and time.monotonic() < deadline:
            time.sleep(0.01)
        rig.host.step()
        _cleaned(rig)
        assert (7, "cancelled") in rig.server.states()
        assert not rig.launcher.started
    finally:
        gate.set()
        rig.host.stop()


def test_scheduler_cannot_pass_an_unanswered_queued_report(
    rig: _Rig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, release = threading.Event(), threading.Event()

    def paused(*args: object, **kwargs: object) -> None:
        del args, kwargs
        entered.set()
        release.wait(5)
        raise ClientError("rejected", status_code=409)

    monkeypatch.setattr(rig.server, "report_machine_job", paused)
    listener = threading.Thread(target=_start, args=(rig,))
    listener.start()
    try:
        assert entered.wait(1)
        rig.host.step()
        time.sleep(0.05)
        assert not rig.repos.fetching
        assert not rig.launcher.started
    finally:
        release.set()
        listener.join(timeout=5)


def test_a_cancelled_jobs_end_is_retried_after_transport_failure(rig: _Rig) -> None:
    _start(rig)
    rig.host.handle(CancelWork(job_id=7))
    rig.server.report_errors = [ClientError("offline")] * 2
    rig.host.step()
    _cleaned(rig)
    rig.host.step()
    assert JobFolder(rig.root / "7").reported_at() is None
    assert _phase(rig, 7) == "ended"
    rig.host.step()
    assert rig.server.states()[-1] == (7, "cancelled")


def _until(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5
    while not condition():
        assert time.monotonic() < deadline, "the condition never held"
        time.sleep(0.005)


def _host(
    tmp_path: Path,
    *,
    server: _Server | None = None,
    launcher: _Launcher | None = None,
    repos: _Repos | None = None,
    gpus: Sequence[int] = (),
    idle: Sequence[int] = (),
    budget_mb: int = 8_192,
    results_max: int = 1_000,
    account: str = "Jan@Example.com",
    server_url: str = _URL,
    tick_sec: float = 3_600,
    retry_sec: float = 0,
) -> _Rig:
    found_server = server or _Server()
    found_launcher = launcher or _Launcher()
    found_repos = repos or _Repos()
    clock = _Clock()

    def client_for(credential: str) -> _Server:
        assert credential == "trax_machine_x"
        return found_server

    host = JobHost(
        name="gx10",
        server_url=server_url,
        credential="trax_machine_x",
        client_for=client_for,
        account=account,
        root=tmp_path / "jobs",
        repos=found_repos,
        launcher=found_launcher,
        gpus=gpus,
        idle_gpus=lambda: idle,
        memory_budget_mb=budget_mb,
        results_max=results_max,
        clock=clock,
        tick_sec=tick_sec,
        retry_sec=retry_sec,
    )
    return _Rig(
        host=host,
        server=found_server,
        launcher=found_launcher,
        repos=found_repos,
        clock=clock,
        root=tmp_path / "jobs",
    )


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[_Rig]:
    found = _host(tmp_path)
    found.host.start(_MACHINE, _INSTANCE)
    yield found
    found.host.stop()


def _start(rig: _Rig, job_id: int = 7, /, **fields: object) -> MachineJobSpec:
    spec = _spec(job_id, **fields)
    rig.host.handle(StartWork(job=spec))
    return spec


def _fetched(rig: _Rig, job_id: int = 7) -> JobFolder:
    """Step until the job's worktree is made."""
    rig.host.step()
    folder = JobFolder(rig.root / str(job_id))
    _until(lambda: _phase(rig, job_id) == "ready")
    return folder


def _cleaned(rig: _Rig, job_id: int = 7) -> None:
    """Wait for the cleanup worker without retrying the end report."""
    _until(lambda: _phase(rig, job_id) in {"ended", "gone"})


def _phase(rig: _Rig, job_id: int) -> str:
    job = rig.host._jobs.get(job_id)
    return "gone" if job is None else job.phase


def _running(rig: _Rig, job_id: int = 7, /, **fields: object) -> JobFolder:
    _ = _start(rig, job_id, **fields)
    folder = _fetched(rig, job_id)
    rig.host.step()
    return folder


def test_a_job_of_this_hosts_account_is_taken_and_reported_queued(rig: _Rig) -> None:
    spec = _start(rig)

    folder = JobFolder(rig.root / "7")
    assert folder.spec() == spec
    assert folder.log.read_text() == "knowop: accepted on gx10\n"
    assert rig.server.states() == [(7, "queued")]


def test_the_host_says_which_job_it_took_and_the_start_of_its_command(
    rig: _Rig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ = _start(rig, argv=["python", "train.py", "--steps", "1" * 60])

    assert capsys.readouterr().out == (
        f"job 7: accepted (python train.py --steps {'1' * 36})\n"
    )


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"owner": "ana@example.com"}, "gx10 runs only jan@example.com's jobs"),
        ({"gpus": 1}, "gx10 offers 0 GPUs to jobs"),
        ({"memory_mb": 8_193}, "gx10 lets its jobs hold 8192 MiB"),
    ],
)
def test_a_job_this_machine_will_not_run_is_failed_with_why_and_kept_nowhere(
    rig: _Rig,
    fields: dict[str, object],
    reason: str,
) -> None:
    _ = _start(rig, **fields)

    assert rig.server.reports == [(7, JobReport(state="failed", reason=reason))]
    assert not (rig.root / "7").exists()


def test_a_job_handed_again_is_reported_queued_again_and_taken_once(rig: _Rig) -> None:
    _ = _start(rig)
    _ = _start(rig)

    assert rig.server.states() == [(7, "queued"), (7, "queued")]
    assert len(rig.host._jobs) == 1


def test_a_job_whose_folder_is_left_from_before_is_refused(rig: _Rig) -> None:
    (rig.root / "7").mkdir(parents=True)

    _ = _start(rig)

    assert rig.server.reports == [
        (7, JobReport(state="failed", reason="job 7 already has a folder on gx10")),
    ]


def test_a_job_runs_in_a_capped_scope_with_its_own_environment_and_ends_reported(
    rig: _Rig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRACKINIZER_TOKEN", "secret")
    monkeypatch.setenv("TRAX_CANVAS", "w")
    monkeypatch.setenv("KEEP_ME", "1")

    folder = _running(rig)

    [(unit, memory_mb, env)] = rig.launcher.started
    assert (unit, memory_mb) == (f"knowop-job-{_TAG}-7", 1_024)
    assert folder.launch() == Launch(unit=unit, gpus=[], memory_mb=1_024)
    assert env["KNOWOP_JOB_ID"] == "7"
    assert env["KNOWOP_JOB_OUT"] == str(folder.out)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["KEEP_ME"] == "1"
    assert "TRACKINIZER_TOKEN" not in env
    assert "TRAX_CANVAS" not in env
    assert rig.server.states() == [(7, "queued"), (7, "running")]
    assert (folder.src / "COMMIT").read_text() == "a" * 40

    rig.host.step()
    assert rig.server.states()[-1] == (7, "running")
    folder.write_ended(Ended(state="completed", exit_code=0, reason=""))
    rig.host.step()
    _cleaned(rig)

    assert rig.server.reports[-1] == (7, JobReport(state="completed", exit_code=0))
    assert (unit, signal.SIGKILL) in rig.launcher.signals
    assert not folder.src.exists()
    assert folder.reported_at() == rig.clock.now
    assert rig.host._jobs == {}
    log = folder.log.read_text().splitlines()
    assert log[1:] == [
        "knowop: fetching aaaaaaaaaaaa of refs/heads/main from https://github.com/o/r.git",
        "knowop: worktree ready",
        f"knowop: starting in {unit}.scope under a 1024 MiB cap, GPUs: none",
        "knowop: completed: exit 0",
    ]


def test_a_job_is_given_idle_gpus_no_other_job_holds(tmp_path: Path) -> None:
    rig = _host(tmp_path, gpus=[0, 1, 2], idle=[1, 2, 3])
    rig.host.start(_MACHINE, _INSTANCE)
    try:
        first = _running(rig, 7, gpus=1)
        second = _running(rig, 8, gpus=1)
        third = _running(rig, 9, gpus=1)
    finally:
        rig.host.stop()

    launches = [f.launch() for f in (first, second, third)]
    assert [None if each is None else each.gpus for each in launches] == [
        [1],
        [2],
        None,
    ]
    assert [env["CUDA_VISIBLE_DEVICES"] for _, _, env in rig.launcher.started] == [
        "1",
        "2",
    ]
    assert third.log.read_text().endswith("knowop: waiting for 1 idle GPUs\n")
    rig.host.step()
    assert third.log.read_text().count("waiting for 1 idle GPUs") == 1


def test_a_job_waits_until_the_running_jobs_leave_room_in_the_memory_budget(
    tmp_path: Path,
) -> None:
    rig = _host(tmp_path, budget_mb=4_096)
    rig.host.start(_MACHINE, _INSTANCE)
    try:
        first = _running(rig, 7, memory_mb=3_000)
        second = _running(rig, 8, memory_mb=3_000)
        assert second.launch() is None
        assert second.log.read_text().endswith(
            "knowop: waiting for 3000 MiB of memory\n",
        )

        first.write_ended(Ended(state="completed", exit_code=0, reason=""))
        rig.host.step()
        rig.host.step()
    finally:
        rig.host.stop()

    assert [unit[-1] for unit, _, _ in rig.launcher.started] == ["7", "8"]


def test_a_commit_that_cannot_be_fetched_fails_the_job_with_why(tmp_path: Path) -> None:
    rig = _host(tmp_path, repos=_Repos(failure="git fetch failed: fatal: no access"))
    rig.host.start(_MACHINE, _INSTANCE)
    try:
        _ = _start(rig)
        _ = _fetched(rig)
        rig.host.step()
        _cleaned(rig)
    finally:
        rig.host.stop()

    assert rig.server.reports[-1] == (
        7,
        JobReport(state="failed", reason="git fetch failed: fatal: no access"),
    )
    assert rig.launcher.started == []


def test_a_scope_that_cannot_start_fails_the_job(tmp_path: Path) -> None:
    rig = _host(tmp_path, launcher=_Launcher(fail=True))
    rig.host.start(_MACHINE, _INSTANCE)
    try:
        _ = _running(rig)
        _cleaned(rig)
    finally:
        rig.host.stop()

    [(job_id, report)] = rig.server.reports[-1:]
    assert (job_id, report.state) == (7, "failed")
    assert report.reason.startswith("the scope did not start: [Errno 2]")


def test_a_job_cancelled_before_it_started_ends_at_once(tmp_path: Path) -> None:
    gate = threading.Event()
    rig = _host(tmp_path, repos=_Repos(gate=gate))
    rig.host.start(_MACHINE, _INSTANCE)
    try:
        _ = _start(rig)
        rig.host.handle(CancelWork(job_id=7))
        rig.host.step()
        _cleaned(rig)
        _ = _start(rig, 8)
        rig.host.step()
        rig.host.handle(CancelWork(job_id=8))
        gate.set()
        _until(lambda: _phase(rig, 8) == "ready")
        rig.host.step()
        _cleaned(rig, 8)
    finally:
        rig.host.stop()

    assert rig.server.states() == [
        (7, "queued"),
        (7, "cancelled"),
        (8, "queued"),
        (8, "cancelled"),
    ]
    assert rig.launcher.started == []


def test_a_cancel_of_a_job_this_host_does_not_hold_is_told_cancelled(rig: _Rig) -> None:
    rig.host.handle(CancelWork(job_id=99))

    assert rig.server.reports == [
        (99, JobReport(state="cancelled", reason="Cancelled.")),
    ]


def test_a_cancelled_job_gets_sigterm_then_sigkill_after_ten_seconds(
    tmp_path: Path,
) -> None:
    rig = _host(tmp_path, launcher=_Launcher(stubborn=True))
    rig.host.start(_MACHINE, _INSTANCE)
    try:
        folder = _running(rig)
        unit = f"knowop-job-{_TAG}-7"
        rig.host.handle(CancelWork(job_id=7))
        rig.host.step()
        rig.host.step()
        rig.clock.now += 9.9
        rig.host.step()
        terms = list(rig.launcher.signals)
        rig.clock.now += 0.1
        rig.host.step()
        rig.host.step()
        _cleaned(rig)
    finally:
        rig.host.stop()

    assert terms == [(unit, signal.SIGTERM)]
    assert rig.launcher.signals[1] == (unit, signal.SIGKILL)
    assert rig.server.reports[-1] == (
        7,
        JobReport(state="cancelled", reason="Cancelled."),
    )
    assert "knowop: cancelled; stopping" in folder.log.read_text()


def test_a_cancelled_job_that_stops_keeps_the_exit_code_its_runner_wrote(
    rig: _Rig,
) -> None:
    folder = _running(rig)
    rig.host.handle(CancelWork(job_id=7))
    rig.host.step()
    folder.write_ended(Ended(state="failed", exit_code=143, reason="exit 143"))
    rig.host.step()
    _cleaned(rig)

    assert rig.server.reports[-1] == (
        7,
        JobReport(state="cancelled", exit_code=143, reason="Cancelled."),
    )


def test_a_runner_that_vanished_without_a_word_fails_the_job(rig: _Rig) -> None:
    _ = _running(rig)
    rig.launcher.alive_units.clear()

    rig.host.step()
    _cleaned(rig)

    assert rig.server.reports[-1] == (
        7,
        JobReport(state="failed", reason="the job ended without recording how"),
    )


def test_a_runner_its_scope_deadline_killed_is_recorded_and_reported_timed_out(
    rig: _Rig,
) -> None:
    folder = _running(rig)
    rig.launcher.alive_units.clear()
    rig.launcher.timed_out_units.add(f"knowop-job-{_TAG}-7")

    rig.host.step()
    _cleaned(rig)

    ended = Ended(
        state="timeout",
        exit_code=124,
        reason="stopped at its 5 min scope limit",
    )
    assert folder.ended() == ended
    assert rig.server.reports[-1] == (
        7,
        JobReport(state="timeout", exit_code=124, reason=ended.reason),
    )


def test_watching_a_job_that_was_never_launched_is_refused(rig: _Rig) -> None:
    _ = _start(rig)
    _ = _fetched(rig)

    with pytest.raises(ValueError, match="running job to have a launch"):
        rig.host._watch(rig.host._jobs[7])


def test_a_report_the_server_did_not_take_is_sent_again(rig: _Rig) -> None:
    rig.server.report_errors = [ClientError("down", status_code=502)]
    _ = _start(rig)
    assert rig.server.reports == []

    _ = _fetched(rig)

    assert rig.server.states() == [(7, "queued")]


@pytest.mark.parametrize("status", [None, 400, 401, 403, 404, 409, 410, 502])
def test_rejected_queued_report_prevents_checkout_and_launch(
    rig: _Rig,
    status: int | None,
) -> None:
    rig.server.report_errors = [ClientError("refused", status_code=status)] * 10
    _ = _start(rig)
    rig.host.step()
    time.sleep(0.05)
    rig.host.step()

    assert not (rig.root / "7" / "src").exists()
    assert rig.launcher.started == []


@pytest.mark.parametrize("status", [None, 403, 404, 409, 410, 502])
def test_rejected_running_report_prevents_launch(rig: _Rig, status: int | None) -> None:
    _ = _start(rig)
    _ = _fetched(rig)
    rig.server.report_errors = [ClientError("refused", status_code=status)] * 10

    rig.host.step()

    assert rig.launcher.started == []


def test_an_end_the_server_did_not_take_is_kept_until_it_does(rig: _Rig) -> None:
    folder = _running(rig)
    folder.write_ended(Ended(state="failed", exit_code=2, reason="exit 2"))
    rig.server.report_errors = [ClientError("down", status_code=None)]

    rig.host.step()
    _cleaned(rig)
    assert folder.reported_at() is None
    rig.host.step()

    assert rig.server.reports[-1] == (
        7,
        JobReport(state="failed", exit_code=2, reason="exit 2"),
    )
    assert folder.reported_at() is not None
    assert rig.host._jobs == {}


def test_a_report_the_server_refused_is_not_sent_again(rig: _Rig) -> None:
    folder = _running(rig)
    folder.write_ended(Ended(state="completed", exit_code=0, reason=""))
    rig.server.report_errors = [ClientError("ended", status_code=409)]

    rig.host.step()
    _cleaned(rig)

    assert folder.reported_at() is not None


def _launched(root: Path, job_id: int, *, ended: Ended | None = None) -> JobFolder:
    folder = JobFolder.create(root / str(job_id), _spec(job_id))
    folder.bind(JobBinding(server=_URL, account=_JAN, machine_id=_MACHINE))
    folder.src.mkdir()
    folder.write_launch(
        Launch(unit=f"knowop-job-{_TAG}-{job_id}", gpus=[], memory_mb=1_024),
    )
    if ended is not None:
        folder.write_ended(ended)
    return folder


def test_a_new_host_adopts_every_job_whose_end_was_not_reported(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    done = _launched(root, 7, ended=Ended(state="completed", exit_code=0, reason=""))
    alive = _launched(root, 8)
    taken = JobFolder.create(root / "9", _spec(9))
    taken.bind(JobBinding(server=_URL, account=_JAN, machine_id=_MACHINE))
    taken.src.mkdir()
    reported = _launched(root, 10)
    reported.mark_reported(1.0)
    (root / "repos").mkdir()
    (root / "11").mkdir()
    launcher = _Launcher(alive_units={f"knowop-job-{_TAG}-8"})
    rig = _host(tmp_path, launcher=launcher)
    rig.host.start(_MACHINE, _INSTANCE)
    try:
        assert sorted(rig.host._jobs) == [7, 8, 9]
        rig.host.step()
        _until(lambda: _phase(rig, 9) == "ready")
        assert taken.src in rig.repos.removed
        _cleaned(rig)
    finally:
        rig.host.stop()

    assert rig.server.reports[0] == (7, JobReport(state="completed", exit_code=0))
    assert sorted(rig.host._jobs) == [8, 9]
    assert "adopted by a new host on gx10" in alive.log.read_text()
    assert done.reported_at() is not None


@pytest.mark.parametrize("different", ["account", "machine", "server"])
@pytest.mark.parametrize("running", [False, True])
def test_recovery_leaves_another_identitys_jobs_untouched(
    tmp_path: Path,
    different: str,
    running: bool,
) -> None:
    first = _host(tmp_path)
    first.host.start(_MACHINE, _INSTANCE)
    if running:
        _running(first)
    else:
        _start(first)
    first.host.stop()
    before = (first.root / "7" / "log").read_bytes()
    server = _Server()
    server.machine_id = uuid.UUID(int=3) if different == "machine" else _MACHINE
    second = _host(
        tmp_path,
        server=server,
        launcher=first.launcher,
        account="ana@example.com" if different == "account" else _JAN,
        server_url="https://other.example.com" if different == "server" else _URL,
    )
    second.host.start(server.machine_id, _INSTANCE)
    try:
        second.host.step()
        assert second.repos.removed == []
        assert second.host._jobs == {}
        assert (first.root / "7" / "log").read_bytes() == before
        assert first.launcher.signals == []
    finally:
        second.host.stop()


@pytest.mark.parametrize("status", [None, 403, 404, 409, 410, 502])
def test_recovery_requires_server_authorization_before_touching_files(
    tmp_path: Path,
    status: int | None,
) -> None:
    first = _host(tmp_path)
    first.host.start(_MACHINE, _INSTANCE)
    _start(first)
    first.host.stop()
    server = _Server()
    server.report_errors = [ClientError("refused", status_code=status)] * 10
    second = _host(tmp_path, server=server)
    second.host.start(_MACHINE, _INSTANCE)
    try:
        assert second.repos.removed == []
        assert second.host._jobs == {}
    finally:
        second.host.stop()


@pytest.mark.parametrize("status", [403, 404, 410])
def test_refused_recovery_stops_its_persisted_live_scope(
    tmp_path: Path,
    status: int,
) -> None:
    _launched(tmp_path / "jobs", 7)
    unit = f"knowop-job-{_TAG}-7"
    launcher = _Launcher(alive_units={unit}, stubborn=True)
    server = _Server()
    server.report_errors = [ClientError("refused", status_code=status)] * 10
    rig = _host(tmp_path, launcher=launcher, server=server)
    rig.host.start(_MACHINE, _INSTANCE)
    try:
        rig.host.step()
        rig.clock.now += 11
        rig.host.step()
        assert not launcher.alive(unit)
        assert launcher.signals
        server.stop_answer = ClientError("revoked", status_code=410)
    finally:
        rig.host.stop()


@pytest.mark.parametrize("action", ["cancel", "revoke"])
def test_recovery_transport_failure_keeps_scope_accounted_and_stoppable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    _launched(tmp_path / "jobs", 7)
    unit = f"knowop-job-{_TAG}-7"
    launcher = _Launcher(alive_units={unit})
    rig = _host(tmp_path, launcher=launcher, budget_mb=1_024)
    report = rig.server.report_machine_job
    recovering: list[int] = []
    offline = threading.Event()
    offline.set()

    def answer(machine: uuid.UUID, job: int, value: JobReport) -> None:
        if isinstance(value, JobRecovery):
            recovering.append(job)
            if offline.is_set():
                raise ClientError("offline")
        report(machine, job, value)

    monkeypatch.setattr(rig.server, "report_machine_job", answer)
    rig.host.start(_MACHINE, _INSTANCE)
    try:
        second = _running(rig, 8)
        assert second.launch() is None
        assert launcher.alive(unit)
        offline.clear()
        rig.host.step()
        assert recovering.count(7) >= 2
        if action == "cancel":
            rig.host.handle(CancelWork(job_id=7))
            rig.host.step()
            assert (unit, signal.SIGTERM) in launcher.signals
            rig.host.step()
            _cleaned(rig)
            assert (7, "cancelled") in rig.server.states()
        else:
            rig.server.stop_answer = ClientError("revoked", status_code=410)
    finally:
        rig.host.stop()
    assert not launcher.alive(unit)


def test_a_recovery_the_server_did_not_answer_waits_out_the_retry_before_the_next(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adopted = _launched(tmp_path / "jobs", 7)
    rig = _host(
        tmp_path,
        launcher=_Launcher(alive_units={f"knowop-job-{_TAG}-7"}),
        retry_sec=60,
    )
    report = rig.server.report_machine_job
    recovering: list[int] = []

    def answer(machine: uuid.UUID, job: int, value: JobReport) -> None:
        if isinstance(value, JobRecovery):
            recovering.append(job)
            if len(recovering) == 1:
                raise ClientError("offline")
        report(machine, job, value)

    monkeypatch.setattr(rig.server, "report_machine_job", answer)
    rig.host.start(_MACHINE, _INSTANCE)
    try:
        rig.host.step()
        early = list(recovering)
        rig.clock.now += 60
        rig.host.step()
    finally:
        rig.host.stop()

    assert (early, recovering) == ([7], [7, 7])
    assert "adopted by a new host on gx10" in adopted.log.read_text()


def test_cleanup_waiting_for_preparation_does_not_delay_another_cancel(
    rig: _Rig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finished = _running(rig, 7)
    _running(rig, 9)
    root = tmp_path / "repos"
    root.mkdir()
    repos = Repos(root)
    monkeypatch.setattr(rig.host, "_repos", repos)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    fcntl.flock(fd, fcntl.LOCK_EX)
    finished.write_ended(Ended(state="completed", exit_code=0, reason=""))
    scheduler = threading.Thread(target=rig.host.step)
    scheduler.start()
    try:
        _until(lambda: (f"knowop-job-{_TAG}-7", signal.SIGKILL) in rig.launcher.signals)
        rig.host.handle(CancelWork(job_id=9))
        scheduler.join(timeout=1)
        assert not scheduler.is_alive()
        rig.host.step()
        assert (f"knowop-job-{_TAG}-9", signal.SIGTERM) in rig.launcher.signals
    finally:
        os.close(fd)
        scheduler.join(timeout=5)


def test_a_finished_scheduler_snapshot_does_not_repeat_cleanup(rig: _Rig) -> None:
    folder = _running(rig)
    snapshot = rig.host._jobs[7]
    folder.write_ended(Ended(state="completed", exit_code=0, reason=""))
    rig.host.step()
    _cleaned(rig)
    assert snapshot.cleanup_thread is not None
    snapshot.cleanup_thread.join(timeout=1)
    assert rig.repos.removed.count(folder.src) == 1

    rig.host._advance(snapshot)
    snapshot.cleanup_thread.join(timeout=1)

    assert rig.repos.removed.count(folder.src) == 1


def test_only_one_worktree_cleanup_runs_at_a_time(
    rig: _Rig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folders = [_running(rig, job_id) for job_id in (7, 8)]
    root = tmp_path / "repos"
    root.mkdir()
    repos = Repos(root)
    remove = repos.remove
    entered: list[Path] = []

    def counted(*, remote: str, dest: Path) -> None:
        entered.append(dest)
        remove(remote=remote, dest=dest)

    monkeypatch.setattr(repos, "remove", counted)
    monkeypatch.setattr(rig.host, "_repos", repos)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        for folder in folders:
            folder.write_ended(Ended(state="completed", exit_code=0, reason=""))
        rig.host.step()
        _until(lambda: bool(entered))
        rig.host.step()
        time.sleep(0.05)
        assert entered == [folders[0].src]
    finally:
        os.close(fd)
        _cleaned(rig, 7)
        rig.host.step()
        _cleaned(rig, 8)
    assert entered == [folder.src for folder in folders]


@pytest.mark.parametrize("cancel", [False, True])
def test_unexpected_preparation_failure_can_finish_and_be_cancelled(
    rig: _Rig,
    monkeypatch: pytest.MonkeyPatch,
    cancel: bool,
) -> None:
    def broken(self: _Repos, **kwargs: object) -> None:
        del self, kwargs
        raise RuntimeError("unexpected preparation failure")

    monkeypatch.setattr(_Repos, "checkout", broken)
    _start(rig)
    rig.host.step()
    worker = rig.host._jobs[7].fetch_thread
    assert worker is not None
    worker.join(timeout=1)
    if cancel:
        rig.host.handle(CancelWork(job_id=7))
    rig.host.step()
    _cleaned(rig)
    assert (7, "cancelled" if cancel else "failed") in rig.server.states()
    assert not rig.launcher.started


def test_the_sweep_deletes_folders_fourteen_days_after_their_end_was_reported(
    rig: _Rig,
) -> None:
    old = _launched(rig.root, 7)
    old.mark_reported(rig.clock.now - 14 * 86_400)
    young = _launched(rig.root, 8)
    young.mark_reported(rig.clock.now - 14 * 86_400 + 1)
    (rig.root / "repos").mkdir()

    rig.host.sweep()

    assert sorted(p.name for p in rig.root.iterdir()) == ["8", "repos"]


def test_the_sweep_runs_once_an_hour_as_the_jobs_are_moved_on(rig: _Rig) -> None:
    old = _launched(rig.root, 7)
    old.mark_reported(rig.clock.now - 15 * 86_400)
    rig.clock.now += 3_599
    rig.host.step()
    assert old.path.exists()

    rig.clock.now += 1
    rig.host.step()

    assert not old.path.exists()


def _read(rig: _Rig, **fields: object) -> ReadReply:
    read_id = uuid.uuid4()
    count = len(rig.server.replies)
    rig.host.handle(
        ReadWork.model_validate(
            {"read_id": read_id, "job_id": 7, "what": "log"} | fields,
        ),
    )
    _until(lambda: len(rig.server.replies) > count)
    found_id, reply = rig.server.replies[-1]
    assert found_id == read_id
    return reply


def test_a_read_returns_the_log_the_manifest_or_a_result_file(rig: _Rig) -> None:
    folder = _running(rig)
    _ = (folder.out / "a.txt").write_bytes(b"result")

    log = _read(rig).reply
    manifest = _read(rig, what="manifest").reply
    file = _read(rig, what="file", path="a.txt", offset=2).reply

    assert isinstance(log, Chunk)
    assert base64.b64decode(log.data_b64).startswith(b"knowop: accepted on gx10\n")
    assert isinstance(manifest, ResultManifest)
    assert [(e.path, e.size) for e in manifest.entries] == [("a.txt", 6)]
    assert isinstance(file, Chunk)
    assert base64.b64decode(file.data_b64) == b"sult"


def test_host_read_workers_remain_bounded_after_relay_timeouts(
    rig: _Rig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()
    entered: list[ReadWork] = []

    def blocked_read(work: ReadWork) -> ReadRefused:
        entered.append(work)
        release.wait(5)
        return ReadRefused(reason="test released")

    monkeypatch.setattr(rig.host, "_read", blocked_read)
    try:
        for _ in range(20):
            rig.host.handle(ReadWork(read_id=uuid.uuid4(), job_id=7, what="log"))
        time.sleep(0.05)
        assert len(entered) <= 4
    finally:
        release.set()


def test_results_over_this_machines_limit_are_refused_with_where_they_stay(
    tmp_path: Path,
) -> None:
    rig = _host(tmp_path, results_max=5)
    rig.host.start(_MACHINE, _INSTANCE)
    try:
        folder = _running(rig)
        _ = (folder.out / "a.txt").write_bytes(b"result")
        refused = _read(rig, what="manifest").reply
    finally:
        rig.host.stop()

    assert refused == ReadRefused(
        reason=f"the results are 6 bytes, over the 5 bytes gx10 sends back; they stay "
        f"in {folder.out} on gx10",
    )


def test_a_read_of_a_job_that_is_not_here_or_not_this_accounts_is_refused(
    rig: _Rig,
) -> None:
    other = JobFolder.create(rig.root / "8", _spec(8, owner="ana@example.com"))
    del other

    missing = _read(rig).reply
    foreign = _read(rig, job_id=8).reply

    assert missing == ReadRefused(reason="job 7 is not on gx10")
    assert foreign == ReadRefused(reason="job 8 is not on gx10")


def test_a_log_read_holds_for_new_bytes_while_the_job_runs(rig: _Rig) -> None:
    folder = _running(rig)
    size = folder.log.stat().st_size
    timer = threading.Timer(0.1, lambda: folder.note("more"))
    timer.start()

    held = _read(rig, offset=size, wait_sec=5).reply

    assert isinstance(held, Chunk)
    assert base64.b64decode(held.data_b64) == b"knowop: more\n"


def test_a_log_read_past_its_hold_returns_what_there_is(rig: _Rig) -> None:
    folder = _running(rig)
    size = folder.log.stat().st_size
    rig.clock.now += 1

    found = _read(rig, offset=size, wait_sec=0).reply

    assert isinstance(found, Chunk)
    assert (found.size, found.ended, found.data_b64) == (size, False, "")


def test_a_read_that_fails_on_the_disk_is_refused(rig: _Rig) -> None:
    folder = _running(rig)
    folder.log.unlink()
    folder.log.symlink_to(folder.path / "spec.json")

    refused = _read(rig).reply

    assert refused == ReadRefused(
        reason="job 7's files could not be read: Too many levels of symbolic links",
    )


def test_an_answer_the_server_did_not_take_is_sent_again_and_a_refused_one_not(
    rig: _Rig,
) -> None:
    _ = _running(rig)
    rig.server.reply_errors = [ClientError("down", status_code=503)]
    _ = _read(rig)
    rig.server.reply_errors = [ClientError("gone", status_code=404)]
    count = len(rig.server.replies)

    rig.host.handle(ReadWork(read_id=uuid.uuid4(), job_id=7, what="log"))
    time.sleep(0.05)

    assert len(rig.server.replies) == count


def test_an_answer_that_never_lands_is_given_up_after_three_tries(rig: _Rig) -> None:
    _ = _running(rig)
    rig.server.reply_errors = [ClientError("down", status_code=None)] * 3

    rig.host.handle(ReadWork(read_id=uuid.uuid4(), job_id=7, what="log"))
    _until(lambda: rig.server.reply_errors == [])
    time.sleep(0.02)

    assert rig.server.replies == []


def test_the_listener_hands_each_offer_to_the_host(tmp_path: Path) -> None:
    server = _Server()
    server.offers = [WorkOffer(work=StartWork(job=_spec()))]
    rig = _host(tmp_path, server=server)
    rig.host.start(_MACHINE, _INSTANCE)
    try:
        _until(lambda: rig.server.states() == [(7, "queued")])
    finally:
        rig.host.stop()


def test_a_failed_wait_is_tried_again(tmp_path: Path) -> None:
    server = _Server()
    server.offers = [
        ClientError("down", status_code=502),
        WorkOffer(work=StartWork(job=_spec())),
    ]
    rig = _host(tmp_path, server=server)
    rig.host.start(_MACHINE, _INSTANCE)
    try:
        _until(lambda: rig.server.states() == [(7, "queued")])
    finally:
        rig.host.stop()


def test_a_revoked_machine_stops_every_job_and_keeps_none(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    alive = _launched(root, 7)
    launcher = _Launcher(alive_units={f"knowop-job-{_TAG}-7"})
    server = _Server()
    server.offers = [ClientError("revoked", status_code=410)]
    rig = _host(tmp_path, server=server, launcher=launcher)

    rig.host.start(_MACHINE, _INSTANCE)
    _until(lambda: alive.reported_at() is not None)
    rig.host.stop()

    assert launcher.signals[0] == (f"knowop-job-{_TAG}-7", signal.SIGTERM)
    assert not alive.src.exists()
    assert "the machine was revoked; the job was stopped" in alive.log.read_text()
    assert rig.host._jobs == {}


@pytest.mark.parametrize("revoked_at", ["listener", "stop"])
def test_revocation_signals_running_jobs_before_waiting_for_cleanup(
    rig: _Rig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    revoked_at: str,
) -> None:
    finished = _running(rig, 7)
    running = [_running(rig, job_id) for job_id in (9, 10)]
    root = tmp_path / "repos"
    root.mkdir()
    monkeypatch.setattr(rig.host, "_repos", Repos(root))
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    fcntl.flock(fd, fcntl.LOCK_EX)
    stopper = threading.Thread(target=rig.host.stop)
    try:
        finished.write_ended(Ended(state="completed", exit_code=0, reason=""))
        rig.host.step()
        cleanup = rig.host._jobs[7].cleanup_thread
        assert cleanup is not None
        assert cleanup.is_alive()
        refused = ClientError("revoked", status_code=410)
        if revoked_at == "listener":
            rig.server.offers.append(refused)
        else:
            rig.server.stop_answer = refused
            stopper.start()

        _until(
            lambda: all(
                (f"knowop-job-{_TAG}-{job_id}", signal.SIGTERM) in rig.launcher.signals
                for job_id in (9, 10)
            ),
        )
        assert cleanup.is_alive()
        assert not rig.launcher.alive_units
    finally:
        os.close(fd)
        if stopper.ident is not None:
            stopper.join(timeout=5)
        _until(lambda: all(folder.reported_at() is not None for folder in running))
    assert not stopper.is_alive()


def test_a_stubborn_job_of_a_revoked_machine_is_killed_after_the_grace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(job_host, "_STOP_GRACE_SEC", 0.05)
    root = tmp_path / "jobs"
    alive = _launched(root, 7)
    unit = f"knowop-job-{_TAG}-7"
    launcher = _Launcher(alive_units={unit}, stubborn=True)
    server = _Server()
    server.stop_answer = ClientError("revoked", status_code=410)
    rig = _host(tmp_path, server=server, launcher=launcher)
    rig.host.start(_MACHINE, _INSTANCE)

    rig.host.stop()

    assert launcher.signals == [(unit, signal.SIGTERM), (unit, signal.SIGKILL)]
    assert alive.reported_at() is not None


def test_stopping_the_host_leaves_running_jobs_running(rig: _Rig) -> None:
    folder = _running(rig)
    rig.server.stop_answer = ClientError("down", status_code=None)

    rig.host.stop()

    assert rig.launcher.alive_units == {f"knowop-job-{_TAG}-7"}
    assert folder.reported_at() is None


def test_shutdown_stops_preparation_even_when_its_revocation_check_raises(
    rig: _Rig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = threading.Event()
    rig.repos.gate = gate
    _start(rig)
    rig.host.step()
    _until(lambda: bool(rig.repos.fetching))
    job = rig.host._jobs[7]
    worker = job.fetch_thread
    assert worker is not None
    wait = rig.server.wait_machine_work

    def broken(
        machine_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        wait_sec: float,
    ) -> WorkOffer:
        if wait_sec == 0:
            raise RuntimeError("unexpected revocation check failure")
        return wait(machine_id, instance=instance, wait_sec=wait_sec)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(rig.server, "wait_machine_work", broken)
            with pytest.raises(RuntimeError, match="unexpected revocation check"):
                rig.host.stop()
        assert job.preparation_cancel.is_set()
        assert not worker.is_alive()
    finally:
        gate.set()
        worker.join(timeout=5)


def test_a_host_that_was_never_started_has_no_machine_to_report_for(
    tmp_path: Path,
) -> None:
    rig = _host(tmp_path)

    with pytest.raises(RuntimeError, match="started"):
        rig.host.handle(CancelWork(job_id=7))
    with pytest.raises(RuntimeError, match="started"):
        rig.host.stop()


def test_default_schedule_and_retry_cadence(tmp_path: Path) -> None:
    server = _Server()
    host = JobHost(
        name="gx10",
        server_url=_URL,
        credential="key",
        client_for=lambda _: server,
        account=_JAN,
        root=tmp_path,
        repos=_Repos(),
        launcher=_Launcher(),
        gpus=(),
        idle_gpus=list,
        memory_budget_mb=1_024,
        results_max=1_000,
    )
    assert host._tick_sec == 1.0
    assert host._retry_sec == 5.0
    assert host._swept_at == 0.0


def test_the_scheduler_moves_the_jobs_on_at_every_tick_by_itself(
    tmp_path: Path,
) -> None:
    rig = _host(tmp_path, tick_sec=0.01)
    rig.host.start(_MACHINE, _INSTANCE)
    try:
        _ = _start(rig)
        _until(lambda: bool(rig.launcher.started))
    finally:
        rig.host.stop()

    assert rig.server.states() == [(7, "queued"), (7, "running")]


def test_a_cleanup_that_fails_is_logged_and_tried_again_at_the_next_step(
    rig: _Rig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    folder = _running(rig)
    rig.repos.removals_failing = 1
    folder.write_ended(Ended(state="completed", exit_code=0, reason=""))

    rig.host.step()
    _until(lambda: "Job 7 cleanup failed; will retry" in caplog.text)
    assert _phase(rig, 7) == "cleaning"
    rig.host.step()
    _cleaned(rig)

    assert rig.repos.removed == [folder.src, folder.src]
    assert not folder.src.exists()


class _Child:
    def __init__(self, code: int | None) -> None:
        self.code = code

    def poll(self) -> int | None:
        return self.code


def _no_systemctl(argv: Sequence[str]) -> tuple[int, str]:
    raise AssertionError(f"systemctl was asked: {argv}")


def test_the_systemd_launcher_starts_the_runner_in_a_capped_scope_of_its_own(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def popen(argv: list[str], **kwargs: object) -> _Child:
        seen.update(kwargs, argv=argv)
        return _Child(None)

    monkeypatch.setattr(job_host.subprocess, "Popen", popen)
    folder = JobFolder.create(tmp_path / "7", _spec())
    launcher = SystemdLauncher(run=_no_systemctl)

    launcher.start(folder, unit="knowop-job-ab-7", memory_mb=512, env={"A": "1"})

    argv = seen["argv"]
    assert isinstance(argv, list)
    assert argv[:10] == [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--unit",
        "knowop-job-ab-7",
        "-p",
        "MemoryMax=512M",
        "-p",
        "MemorySwapMax=0",
    ]
    assert argv[-3:] == [
        "-m",
        "knowop.shell.job_runner",
        str(folder.path),
    ]
    assert (seen["cwd"], seen["env"], seen["start_new_session"]) == (
        folder.path,
        {"A": "1"},
        True,
    )
    assert seen["stdin"] == subprocess.DEVNULL
    assert launcher.alive("knowop-job-ab-7")


def test_the_systemd_launcher_asks_systemd_about_a_scope_it_did_not_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def run(argv: Sequence[str]) -> tuple[int, str]:
        calls.append(list(argv))
        return (0 if argv[-1].endswith("-7.scope") else 3), ""

    def popen(*args: object, **kwargs: object) -> _Child:
        del args, kwargs
        return _Child(0)

    monkeypatch.setattr(job_host.subprocess, "Popen", popen)
    launcher = SystemdLauncher(run=run)
    folder = JobFolder.create(tmp_path / "9", _spec(9))
    launcher.start(folder, unit="knowop-job-ab-9", memory_mb=512, env={})

    found = [launcher.alive(unit) for unit in ("knowop-job-ab-7", "knowop-job-ab-8")]
    reaped = launcher.alive("knowop-job-ab-9")
    launcher.kill("knowop-job-ab-7", signal.SIGTERM)

    assert (found, reaped) == ([True, False], False)
    assert calls == [
        ["systemctl", "--user", "is-active", "--quiet", "knowop-job-ab-7.scope"],
        ["systemctl", "--user", "is-active", "--quiet", "knowop-job-ab-8.scope"],
        ["systemctl", "--user", "kill", "--signal=15", "knowop-job-ab-7.scope"],
    ]


def test_the_systemd_launcher_reads_a_timeout_from_systemds_result() -> None:
    results = {
        "knowop-job-ab-7.scope": (0, "timeout\n"),
        "knowop-job-ab-8.scope": (0, "success\n"),
        "knowop-job-ab-9.scope": (1, "timeout\n"),
    }
    calls: list[list[str]] = []

    def run(argv: Sequence[str]) -> tuple[int, str]:
        calls.append(list(argv))
        return results[argv[-1]]

    launcher = SystemdLauncher(run=run)

    found = [launcher.timed_out(f"knowop-job-ab-{job}") for job in (7, 8, 9)]

    assert found == [True, False, False]
    assert calls[0] == [
        "systemctl",
        "--user",
        "show",
        "--property=Result",
        "--value",
        "knowop-job-ab-7.scope",
    ]


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
