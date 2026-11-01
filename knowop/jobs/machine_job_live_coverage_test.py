"""Machine submission sends a small control model, never repository bytes."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from trackinizer.wire.wire_machine_job import (
    MachineJobSubmit,
    MachineJobView,
    ReadReply,
)

import pytest

from knowop.jobs import machine_job
from knowop.jobs.machine_job import Pushed, run_submit


@dataclass(slots=True, kw_only=True)
class _Server:
    body: MachineJobSubmit | None = None

    def machine_job_submit(self, body: MachineJobSubmit) -> MachineJobView:
        """Record the control request and return a queued job."""
        self.body = body
        return MachineJobView(
            id=7,
            machine=body.machine,
            state="queued",
            gpus=body.gpus,
            memory_mb=body.memory_mb,
            minutes=body.minutes,
            label=body.label,
            argv=body.argv,
            remote=body.remote,
            ref=body.ref,
            commit=body.commit,
            exit_code=None,
            reason="",
            created_at=datetime.now(UTC),
            queued_at=None,
            started_at=None,
            ended_at=None,
        )

    def machine_job_view(self, job_id: int) -> MachineJobView:
        """Reject unexpected polling."""
        raise AssertionError(f"Unexpected polling of job {job_id}.")

    def machine_job_log(
        self,
        job_id: int,
        *,
        offset: int,
        wait: float,
    ) -> ReadReply:
        """Reject unexpected log reads."""
        raise AssertionError(f"Unexpected log read for {job_id} at {offset=} {wait=}.")

    def machine_job_manifest(self, job_id: int) -> ReadReply:
        """Reject unexpected manifest reads."""
        raise AssertionError(f"Unexpected manifest read for job {job_id}.")

    def machine_job_file(self, job_id: int, path: str, *, offset: int) -> ReadReply:
        """Reject unexpected result-file reads."""
        raise AssertionError(
            f"Unexpected result read for {job_id}: {path} at {offset}.",
        )


def test_a_machine_submit_contains_control_fields_only_and_stays_under_two_kib(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def pushed_commit(repo: Path) -> Pushed:
        del repo
        return Pushed(
            remote="https://example.invalid/o/r.git",
            ref="refs/heads/main",
            commit="a" * 40,
            branch="main",
        )

    monkeypatch.setattr(machine_job, "pushed", pushed_commit)
    server = _Server()

    assert (
        run_submit(
            server,
            repo=tmp_path,
            machine="gx10",
            command=["python", "train.py"],
            gpus=0,
            memory_mb=512,
            minutes=1,
            label="test",
            sleep=lambda _: None,
            clock=lambda: 0,
        )
        == 7
    )

    body = server.body
    assert body is not None
    encoded = body.model_dump_json()
    assert len(encoded) < 2_048
    assert set(MachineJobSubmit.model_fields) == {
        "machine",
        "argv",
        "gpus",
        "memory_mb",
        "minutes",
        "label",
        "remote",
        "ref",
        "commit",
    }
    assert "payload" not in encoded


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
