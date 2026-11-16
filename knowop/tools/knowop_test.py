"""Host actions use the client boundary, without a shell or server."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING
from uuid import UUID

from trackinizer.client.errors import ClientError
from trackinizer.wire.refs import SeqRef
from trackinizer.wire.wire_session_relay import Opened, ProcessSpec
from trackinizer.wire.wire_sessions import SessionStart, SessionStartResponse
from trackinizer.wire.wire_terminals import TerminalHost, TerminalHosts

import pytest

from knowop.hosts_test import _KNOWHOST, _KnowHostApi
from knowop.process.main_test import _attached
from knowop.tools import knowop


if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

    from trackinizer.types.inquiries import Inquiry
    from trackinizer.wire.refs import Ref
    from trackinizer.wire.wire_session_relay import AgentProcess, ViewerFrame
    from treekle import PlainTree


class _Client:
    def __init__(self) -> None:
        self.api = _KnowHostApi(knowhosts=[dict(_KNOWHOST)])
        self.answer = _attached()
        self.failure: ClientError | None = None
        self.frames = [Opened()]
        self.starts: list[SessionStart] = []
        self.started: list[ProcessSpec] = []
        self.resolved: list[SeqRef] = []
        self.stopped: list[UUID] = []
        self.stop_answer: AgentProcess | None = None

    def start_process(self, session_id: UUID, spec: ProcessSpec) -> AgentProcess:
        assert session_id in (UUID(int=11), UUID(int=15))
        self.started.append(spec)
        return self.answer

    @contextmanager
    def lines_stream(self, session_id: UUID) -> Generator[Iterator[ViewerFrame]]:
        assert session_id in (UUID(int=11), UUID(int=15))
        yield iter(self.frames)

    def resolve_id(self, ref: Ref) -> tuple[Inquiry.InquiryKind, UUID]:
        assert isinstance(ref, SeqRef)
        self.resolved.append(ref)
        return ref.kind, UUID(int=15)

    def stop_process(self, session_id: UUID) -> AgentProcess | None:
        self.stopped.append(session_id)
        return self.stop_answer

    def read_process(self, session_id: UUID) -> AgentProcess | None:
        assert session_id == UUID(int=15)
        return self.answer

    def get(self, path: str, *, timeout: float | None = None) -> PlainTree:
        del timeout
        if path.startswith("/api/addons/"):
            return self.api.get(path)
        assert path in (
            f"/api/inquiries/{UUID(int=11)}",
            f"/api/inquiries/{UUID(int=15)}",
        )
        return {"seq": 412}

    def post(self, path: str, **options: object) -> PlainTree:
        return self.api.post(path, **options)

    def terminal_hosts(self, *, owner: str | None = None) -> TerminalHosts:
        assert owner is None
        return TerminalHosts(
            hosts=[
                TerminalHost(
                    name="laptop",
                    status="online",
                    terminal_ready=True,
                    confirmed=True,
                ),
            ],
        )

    def session_start(self, body: SessionStart) -> SessionStartResponse:
        self.starts.append(body)
        if self.failure is not None:
            raise self.failure
        return SessionStartResponse(id=UUID(int=11), seq=0)


def _tools(client: _Client) -> knowop.KnowOpTools:
    return knowop.KnowOpTools(client, session_id=UUID(int=90))


@pytest.mark.asyncio
async def test_list_names_hosts_states_and_rental_availability() -> None:
    client = _Client()
    said = await _tools(client).act(action="host.list")
    assert "laptop" in said
    assert "knowhost-0b5e9a101c2d" in said
    assert "running" in said
    assert "rents KnowHosts" in said
    client.api.refuse = 404
    assert "rents no KnowHosts" in await _tools(client).act(action="host.list")
    client.api.refuse = None
    client.api.knowhosts = []
    assert "rents KnowHosts" in await _tools(client).act(action="host.list")


@pytest.mark.asyncio
async def test_start_sends_once_waits_150_seconds_and_names_online_host() -> None:
    client = _Client()
    assert "knowhost-0b5e9a101c2d is running" in await _tools(client).act(
        action="host.start",
    )
    assert client.api.calls == [
        ("POST", "/api/addons/knowhost/hosts", {"timeout": 150.0, "retry_attempts": 1}),
    ]


@pytest.mark.asyncio
async def test_pause_and_resume_resolve_names_and_say_when_memory_was_lost() -> None:
    client = _Client()
    tools = _tools(client)
    assert "is paused" in await tools.act(
        action="host.pause",
        name="knowhost-0b5e9a101c2d",
    )
    client.api.answer["memory_kept"] = False
    said = await tools.act(action="host.resume", name="knowhost-0b5e9a101c2d")
    assert "is running again" in said
    assert "processes were lost" in said
    assert client.api.calls[-1] == (
        "POST",
        "/api/addons/knowhost/hosts/0b5e9a10-1c2d-4e5f-8a9b-0c1d2e3f4a5b/resume",
        {"timeout": 150.0, "retry_attempts": 1},
    )


@pytest.mark.asyncio
async def test_launch_is_spawned_by_this_session_and_returns_number_not_event_seq() -> (
    None
):
    client = _Client()
    said = await _tools(client).act(
        action="session.launch",
        name="lead",
        time="20",
        model="sol",
        effort="high",
        budget=0.5,
    )
    assert said == "lead#412"
    [start] = client.starts
    assert start.spawned_by == UUID(int=90)
    assert start.cli == "sagent"
    assert start.cli_session_id is not None
    assert str(
        UUID(start.cli_session_id.removeprefix("chat:")),
    ) == start.cli_session_id.removeprefix("chat:")
    assert client.started == [
        ProcessSpec(
            args=(
                "--time",
                "0-00:20:00",
                "--name",
                "lead",
                "--model",
                "sol",
                "--model.thinking.effort",
                "high",
                "--max-budget-usd",
                "0.5",
            ),
        ),
    ]


@pytest.mark.asyncio
async def test_stop_resolves_the_session_number_and_stops_only_its_process() -> None:
    client = _Client()
    client.stop_answer = _attached("detached")
    assert "stopped lead#412" in await _tools(client).act(
        action="session.stop",
        name="lead#412",
    )
    assert client.resolved[0].seq == 412
    assert client.stopped == [UUID(int=15)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    ["host.start", "host.list", "host.pause", "host.resume"],
)
async def test_server_refusal_is_a_tool_answer(action: knowop.Action) -> None:
    client = _Client()
    client.api.refuse = 402
    assert (
        await _tools(client).act(
            action=action,
            name="knowhost-0b5e9a101c2d",
        )
        == "the server answered 402."
    )


@pytest.mark.asyncio
async def test_invalid_launch_opens_no_session() -> None:
    client = _Client()
    assert "no time limit" in await _tools(client).act(
        action="session.launch",
        name="lead",
        time="bad",
    )
    assert client.starts == []


@pytest.mark.asyncio
async def test_launch_returns_the_servers_refusal_sentence() -> None:
    client = _Client()
    client.failure = ClientError(
        'POST /api/sessions/start -> 403: {"detail": "This account cannot launch agents."}',
        status_code=403,
    )
    assert (
        await _tools(client).act(action="session.launch", time="20")
        == "the server answered 403: This account cannot launch agents."
    )
    assert client.started == []


@pytest.mark.asyncio
async def test_remote_flags_are_checked_before_opening_a_session() -> None:
    client = _Client()
    assert (
        await _tools(client).act(action="session.launch", name="--tools", time="20")
        == "a remote agent takes --name with a value"
    )
    assert client.starts == []


@pytest.mark.asyncio
async def test_stop_with_a_uuid_uses_it_without_an_inquiry_lookup() -> None:
    client = _Client()
    target = UUID(int=42)
    client.stop_answer = _attached("detached")
    assert (
        await _tools(client).act(action="session.stop", name=f"lead#{target}")
        == f"stopped lead#{target}"
    )
    assert client.stopped == [target]
    assert client.resolved == []


@pytest.mark.asyncio
async def test_launch_requires_a_time_limit_before_opening_a_session() -> None:
    client = _Client()
    assert "requires a time limit" in await _tools(client).act(action="session.launch")
    assert client.starts == client.started == []


@pytest.mark.asyncio
async def test_preflight_tool_has_no_caller_and_opens_no_session() -> None:
    client = _Client()
    tools = knowop.KnowOpTools(client, session_id=None)
    assert "requires a caller session" in await tools.act(
        action="session.launch",
        time="20",
    )
    assert client.starts == client.started == []


@pytest.mark.asyncio
async def test_launch_refuses_a_blank_label_before_opening_a_session() -> None:
    client = _Client()
    assert (
        await _tools(client).act(
            action="session.launch",
            name=" ",
            time="20",
        )
        == "the agent label must not be blank"
    )
    assert client.starts == client.started == []


@pytest.mark.asyncio
async def test_launch_defaults_only_the_label_and_optional_settings() -> None:
    client = _Client()
    assert await _tools(client).act(action="session.launch", time="3") == "Agent#412"
    assert client.started == [
        ProcessSpec(args=("--time", "0-00:03:00", "--name", "Agent")),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["starting", "attached", "detached"])
async def test_status_reads_one_named_sessions_process(state: str) -> None:
    client = _Client()
    client.answer = _attached(state)
    said = await _tools(client).act(action="session.status", name="lead#412")
    assert f"lead#412: {state}" in said
    if state == "detached":
        assert "it crashed" in said
    assert client.resolved == [SeqRef(kind="AgentSession", seq=412)]
    assert client.starts == client.started == []


@pytest.mark.asyncio
async def test_continue_restarts_the_same_session_with_a_new_time_limit() -> None:
    client = _Client()
    assert (
        await _tools(client).act(action="session.continue", name="lead#412", time="3")
        == "lead#412"
    )
    assert client.resolved == [SeqRef(kind="AgentSession", seq=412)]
    assert client.starts == []
    assert client.started == [
        ProcessSpec(args=("--time", "0-00:03:00", "--name", "lead")),
    ]


@pytest.mark.asyncio
async def test_continue_requires_a_new_time_limit_before_starting() -> None:
    client = _Client()
    assert "new time limit" in await _tools(client).act(
        action="session.continue",
        name="lead#412",
    )
    assert client.starts == client.started == []


@pytest.mark.asyncio
async def test_status_requires_a_named_session() -> None:
    client = _Client()
    assert "names no agent" in await _tools(client).act(action="session.status")
    assert client.starts == client.started == []


@pytest.mark.asyncio
async def test_stop_with_no_process_does_not_report_success() -> None:
    client = _Client()
    assert (
        await _tools(client).act(action="session.stop", name="lead#412")
        == "lead#412 has had no agent process"
    )


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
