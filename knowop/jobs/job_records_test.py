"""``knowop job status|cancel|secrets|secret`` against a scripted server."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import io
import sys

from trackinizer.client.errors import ClientError
from trackinizer.wire.wire_host_job import HostJobList, HostJobView
from trackinizer.wire.wire_job import JobList, JobView, SecretNames

import pytest

from knowop.jobs import job_records
from knowop.jobs.job import JOB_WORD, Job


if TYPE_CHECKING:
    from trackinizer.client.client import Client


_COMMIT = "a" * 40


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


def _host_job(state: str = "running", **fields: object) -> HostJobView:
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
            "remote": "git@github.com:o/r.git",
            "ref": "refs/heads/main",
            "commit": _COMMIT,
            "exit_code": None,
            "reason": "",
            "created_at": datetime(2026, 10, 23, 12, tzinfo=UTC),
            "queued_at": None,
            "started_at": None,
            "ended_at": None,
            **fields,
        },
    )


def _refusal(status: int) -> ClientError:
    return ClientError(f"-> {status}", status_code=status)


class _Server:
    """The server's job methods these verbs call, scripted, recording each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.view = _job("queued")
        self.jobs = [_job("running"), _job("completed", id=6, node=None)]
        self.names: list[str] = []
        self.host_jobs: list[HostJobView] = []
        self.gpu_answer = 200
        self.host_answer = 200

    def job_view(self, job_id: int) -> JobView:
        self.calls.append(("job", job_id))
        return self.view

    def job_list(self) -> JobList:
        self.calls.append(("jobs", None))
        if self.gpu_answer != 200:
            raise _refusal(self.gpu_answer)
        return JobList(jobs=self.jobs)

    def job_cancel(self, job_id: int) -> JobView:
        self.calls.append(("cancel", job_id))
        return _job("cancelled")

    def host_job_view(self, job_id: int) -> HostJobView:
        self.calls.append(("host_job", job_id))
        return self._host(job_id)

    def host_job_list(self) -> HostJobList:
        self.calls.append(("host_jobs", None))
        if self.host_answer != 200:
            raise _refusal(self.host_answer)
        return HostJobList(jobs=self.host_jobs)

    def host_job_cancel(self, job_id: int) -> HostJobView:
        self.calls.append(("host_cancel", job_id))
        return self._host(job_id).model_copy(update={"state": "cancelled"})

    def _host(self, job_id: int) -> HostJobView:
        if self.host_answer != 200:
            raise _refusal(self.host_answer)
        for each in self.host_jobs:
            if each.id == job_id:
                return each
        raise _refusal(404)

    def job_secret_names(self) -> SecretNames:
        return SecretNames(names=self.names)

    def job_put_secret(self, name: str, value: str) -> None:
        self.calls.append(("put_secret", (name, value)))

    def job_delete_secret(self, name: str) -> None:
        self.calls.append(("delete_secret", name))


@pytest.fixture
def server() -> _Server:
    return _Server()


def _run(server: _Server, *rest: str) -> None:
    Job.dispatch(JOB_WORD, list(rest), lambda: cast("Client", server))


def _no_client() -> Client:
    raise AssertionError("the command must not reach the server")


# -- Routing.


@pytest.mark.parametrize("rest", [["help"], ["cancel", "help"], ["secret", "-h"]])
def test_help_names_every_verb_without_running_anything(
    rest: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    Job.dispatch(JOB_WORD, list(rest), _no_client)

    out = capsys.readouterr().out
    for form in (
        "knowop job status [JOB]",
        "knowop job cancel JOB",
        "knowop job secrets",
        "knowop job secret NAME set < FILE",
        "knowop job secret NAME del",
    ):
        assert form in out


@pytest.mark.parametrize(
    "rest",
    [
        ["nonsense"],
        ["cancel"],
        ["cancel", "seven"],
        ["status", "7", "8"],
        ["secret", "HF_TOKEN"],
        ["secret", "HF_TOKEN", "set", "a-literal-value"],
        ["secret", "HF_TOKEN", "wipe"],
    ],
)
def test_a_malformed_command_is_refused_before_any_request(rest: list[str]) -> None:
    with pytest.raises(SystemExit) as err:
        Job.dispatch(JOB_WORD, list(rest), _no_client)

    assert err.value.code == 2


# -- Secrets: the value comes from stdin and nowhere else.


def test_a_secret_value_is_read_from_stdin_and_one_newline_is_stripped(
    server: _Server,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("hf_value\n"))

    _run(server, "secret", "HF_TOKEN", "set")

    assert server.calls == [("put_secret", ("HF_TOKEN", "hf_value"))]
    assert capsys.readouterr().out == "set: job secret HF_TOKEN\n"


@pytest.mark.parametrize(
    ("stdin", "value"),
    [("hf\r\n", "hf"), ("hf\n\n", "hf\n"), ("hf", "hf"), ("a b\n", "a b")],
)
def test_one_trailing_line_ending_is_stripped_from_a_secret_and_no_more(
    server: _Server,
    monkeypatch: pytest.MonkeyPatch,
    stdin: str,
    value: str,
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))

    _run(server, "secret", "WANDB_API_KEY", "set")

    assert server.calls == [("put_secret", ("WANDB_API_KEY", value))]


@pytest.mark.parametrize("stdin", ["", "\n", "\r\n"])
def test_an_empty_secret_is_refused(
    server: _Server,
    monkeypatch: pytest.MonkeyPatch,
    stdin: str,
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))

    with pytest.raises(ClientError, match=r"^stdin held no value for HF_TOKEN$"):
        _run(server, "secret", "HF_TOKEN", "set")

    assert server.calls == []


def test_a_secret_name_outside_the_two_known_is_refused_before_reading_stdin(
    server: _Server,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", None)

    with pytest.raises(
        ClientError,
        match=r"^a secret name must be one of HF_TOKEN, WANDB_API_KEY$",
    ):
        _run(server, "secret", "AWS_KEY", "set")


def test_secrets_lists_names_and_del_removes_one(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _run(server, "secrets")
    empty = capsys.readouterr().out
    server.names = ["HF_TOKEN", "WANDB_API_KEY"]
    _run(server, "secrets")
    listed = capsys.readouterr().out
    _run(server, "secret", "WANDB_API_KEY", "del")
    deleted = capsys.readouterr().out

    assert empty == "(no secrets)\n"
    assert listed == "HF_TOKEN\nWANDB_API_KEY\n"
    assert deleted == "deleted: job secret WANDB_API_KEY\n"
    assert server.calls == [("delete_secret", "WANDB_API_KEY")]


# -- status and cancel.


def test_status_lists_jobs_in_aligned_columns_with_the_command_cut(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _run(server, "status")

    assert capsys.readouterr().out == (
        "7  running    gpu-node-b  python x.py\n6  completed  -           python x.py\n"
    )


def test_a_long_command_is_cut_to_sixty_characters_in_the_listing(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.jobs = [_job("running", argv=["x" * 70])]

    _run(server, "status")

    assert capsys.readouterr().out == f"7  running  gpu-node-b  {'x' * 60}\n"


def test_status_of_one_job_shows_every_field(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.view = _job(
        "completed",
        started_at=datetime(2026, 10, 21, 12, 1, tzinfo=UTC),
        ended_at=datetime(2026, 10, 21, 12, 5, tzinfo=UTC),
        gpu_seconds=240,
        usd=0.2,
        usd_per_gpu_minute=0.05,
        reason="Stopped: $0.01 is left today.",
    )

    _run(server, "status", "7")

    assert capsys.readouterr().out == (
        "job:      7\n"
        "state:    completed\n"
        "node:     gpu-node-b\n"
        "gpus:     1 for up to 5 min\n"
        "label:    try\n"
        "command:  python x.py\n"
        f"commit:   {_COMMIT}\n"
        "created:  2026-10-21T12:00:00+00:00\n"
        "started:  2026-10-21T12:01:00+00:00\n"
        "ended:    2026-10-21T12:05:00+00:00\n"
        "rate:     usd 0.05 per gpu-min, before markup\n"
        "charged:  240 gpu-s, usd 0.2\n"
        "reason:   Stopped: $0.01 is left today.\n"
    )
    assert server.calls == [("host_job", 7), ("job", 7)]


def test_status_of_a_job_not_yet_started_shows_dashes(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.view = _job("requested", node=None, label="")

    _run(server, "status", "7")

    out = capsys.readouterr().out.splitlines()
    assert out[2:3] == ["node:     -"]
    assert out[4:5] == ["label:    -"]
    assert out[8:] == [
        "started:  -",
        "ended:    -",
        "rate:     -",
        "charged:  -, -",
        "reason:   -",
    ]


def test_status_with_no_jobs_says_so(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.jobs = []

    _run(server, "status")

    assert capsys.readouterr().out == "(no jobs)\n"


def test_cancel_reports_the_state_the_server_gave(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _run(server, "cancel", "7")

    assert server.calls == [("host_cancel", 7), ("cancel", 7)]
    assert capsys.readouterr().out == "job 7: cancelled\n"


# -- Jobs on the person's own hosts.


def test_status_lists_gpu_and_host_jobs_together_newest_first(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.host_jobs = [_host_job(), _host_job("completed", id=5)]

    _run(server, "status")

    assert capsys.readouterr().out == (
        "9  running    gx10        python train.py\n"
        "7  running    gpu-node-b  python x.py\n"
        "6  completed  -           python x.py\n"
        "5  completed  gx10        python train.py\n"
    )


@pytest.mark.parametrize(
    ("gpu", "host", "listed"),
    [
        (503, 200, ["9"]),
        (200, 404, ["7", "6"]),
        (200, 401, ["7", "6"]),
        (503, 503, []),
    ],
)
def test_status_lists_what_a_server_that_runs_one_kind_holds(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
    gpu: int,
    host: int,
    listed: list[str],
) -> None:
    server.host_jobs = [_host_job()]
    server.gpu_answer, server.host_answer = gpu, host

    _run(server, "status")

    out = capsys.readouterr().out
    assert [line.split()[0] for line in out.splitlines() if line[0].isdigit()] == listed
    assert (out == "(no jobs)\n") is (not listed)


def test_status_of_a_listing_that_fails_otherwise_says_so(server: _Server) -> None:
    server.host_answer = 500

    with pytest.raises(ClientError, match="500"):
        _run(server, "status")


def test_a_key_both_kinds_of_route_refuse_is_said_to_be_refused(
    server: _Server,
) -> None:
    server.gpu_answer = server.host_answer = 401

    with pytest.raises(ClientError, match="401"):
        _run(server, "status")


def test_a_box_credential_the_host_routes_refuse_reaches_its_gpu_job(
    server: _Server,
) -> None:
    # A campaign box's credential opens only the GPU job routes; elsewhere it is 401.
    server.host_answer = 401

    _run(server, "status", "7")
    _run(server, "cancel", "7")
    view = job_records.host_view(cast("Client", server), 7)

    assert view is None
    assert server.calls == [
        ("host_job", 7),
        ("job", 7),
        ("host_cancel", 7),
        ("cancel", 7),
        ("host_job", 7),
    ]


def test_status_of_a_host_job_shows_its_host_limits_and_commit(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.host_jobs = [
        _host_job(
            "failed",
            exit_code=3,
            reason="exit 3",
            started_at=datetime(2026, 10, 23, 12, 1, tzinfo=UTC),
            ended_at=datetime(2026, 10, 23, 12, 2, tzinfo=UTC),
            label="smoke",
        ),
    ]

    _run(server, "status", "9")

    assert capsys.readouterr().out == (
        "job:      9\n"
        "state:    failed\n"
        "host:     gx10\n"
        "limits:   0 GPUs, 2048 MiB, up to 5 min\n"
        "label:    smoke\n"
        "command:  python train.py\n"
        f"commit:   {_COMMIT} (refs/heads/main at git@github.com:o/r.git)\n"
        "created:  2026-10-23T12:00:00+00:00\n"
        "started:  2026-10-23T12:01:00+00:00\n"
        "ended:    2026-10-23T12:02:00+00:00\n"
        "exit:     3\n"
        "reason:   exit 3\n"
    )
    assert server.calls == [("host_job", 9)]


def test_status_of_a_host_job_not_yet_ended_shows_dashes(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.host_jobs = [_host_job("requested")]

    _run(server, "status", "9")

    out = capsys.readouterr().out.splitlines()
    assert out[4] == "label:    -"
    assert out[8:] == ["started:  -", "ended:    -", "exit:     -", "reason:   -"]


def test_cancel_of_a_host_job_goes_to_its_host(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.host_jobs = [_host_job()]

    _run(server, "cancel", "9")

    assert server.calls == [("host_cancel", 9)]
    assert capsys.readouterr().out == "job 9: cancelled\n"


def test_a_failure_other_than_not_found_is_not_taken_for_a_gpu_job(
    server: _Server,
) -> None:
    server.host_answer = 500

    with pytest.raises(ClientError, match="500"):
        _run(server, "cancel", "9")
    with pytest.raises(ClientError, match="500"):
        _ = job_records.host_view(cast("Client", server), 9)

    assert ("cancel", 9) not in server.calls


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
