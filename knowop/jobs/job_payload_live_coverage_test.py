"""Large job payloads travel to the node, not through server requests."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from trackinizer.client.client import Client
from trackinizer.wire.wire_job import (
    BOX_CREDENTIAL_PREFIX,
    GrantStatus,
    GrantTicket,
    ReadGrantBody,
    StartBody,
    SubmitBody,
)

import httpx2
import pytest

from knowop.jobs import (
    job as job_commands,
    job_test,
)


if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import BaseModel


@pytest.fixture
def world() -> job_test._Rig:
    return job_test._Rig()


def test_a_box_credential_sends_only_the_job_control_body() -> None:
    bearer = f"{BOX_CREDENTIAL_PREFIX}{UUID(int=101).hex}_{'1' * 43}"
    request_sizes: list[int] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["Authorization"] == f"Bearer {bearer}"
        request_sizes.append(len(request.content))
        return httpx2.Response(
            201,
            json=job_test._job().model_dump(mode="json"),
        )

    client = Client("https://jobs.test", api_key=bearer)
    client._http.close()
    client._http = httpx2.Client(
        base_url=client.base_url,
        headers={"Authorization": f"Bearer {bearer}"},
        transport=httpx2.MockTransport(answer),
    )
    with client:
        job = client.job_submit(
            SubmitBody(
                public_key=job_test._PUBLIC,
                gpus=1,
                minutes=10,
                label="box",
                argv=["python", "run.py"],
                commit=job_test._COMMIT,
            ),
        )

    assert job.id == 7
    assert len(request_sizes) == 1
    assert request_sizes[0] < 65_536


@pytest.mark.parametrize(
    "api_key",
    ["person-key", f"{BOX_CREDENTIAL_PREFIX}{UUID(int=101).hex}_{'1' * 43}"],
    ids=["person", "box"],
)
def test_large_bundle_and_results_never_enter_server_request_bodies(
    world: job_test._Rig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    api_key: str,
) -> None:
    _ = job_test._fake_source(monkeypatch, tmp_path, size=100_000)
    target = tmp_path / "results"
    responses: list[BaseModel] = [
        job_test._job("requested"),
        GrantStatus(state="ready", grant=job_test._grant()),
        job_test._job("starting"),
        job_test._job("queued"),
        job_test._job("queued"),
        GrantTicket(serial=9),
        GrantStatus(state="ready", grant=job_test._grant("ro")),
    ]
    requests: list[httpx2.Request] = []
    response_sizes: list[int] = []

    def results_dir(url: str, job: int, to: str | None) -> Path:
        del url, job, to
        return target

    def answer(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["Authorization"] == f"Bearer {api_key}"
        requests.append(request)
        response = httpx2.Response(200, json=responses.pop(0).model_dump(mode="json"))
        response_sizes.append(len(response.content))
        return response

    monkeypatch.setattr(job_commands, "results_dir", results_dir)
    with Client("https://jobs.test", api_key=api_key) as client:
        client._http.close()
        client._http = httpx2.Client(
            base_url=client.base_url,
            headers=dict(client._http.headers),
            transport=httpx2.MockTransport(answer),
        )
        rig = job_commands.Rig(
            client=client,
            runner=world.node,
            tools=job_test._TOOLS,
            sleep=world.time.sleep,
            clock=world.time.clock,
        )
        job_commands.Job.run_submit(
            rig,
            command=("true",),
            gpus=1,
            minutes=10,
            label="",
            follow=False,
            include_untracked=False,
        )
        world.node.files = {"large.bin": b"r" * 100_000}
        job_commands.Job.run_results(rig, 7, to=str(target))

    assert world.node.uploaded == [b"b" * 100_000]
    assert (target / "src.bundle").read_bytes() == b"b" * 100_000
    assert (target / "out" / "large.bin").read_bytes() == b"r" * 100_000
    assert len(requests) == 7
    assert responses == []
    assert max(len(request.content) for request in requests) < 65_536
    assert max(response_sizes) < 65_536
    assert (
        SubmitBody.model_validate_json(requests[0].content).commit == job_test._COMMIT
    )
    assert StartBody.model_validate_json(requests[2].content).bundle_bytes == 100_000
    assert (
        ReadGrantBody.model_validate_json(requests[5].content).public_key
        == job_test._PUBLIC
    )


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
