"""Host actions use the client boundary, without a shell or server."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final
from uuid import UUID

from trackinizer.client.errors import ClientError
from trackinizer.wire.refs import SeqRef
from trackinizer.wire.wire_job import JobList, JobView
from trackinizer.wire.wire_session_relay import Opened, ProcessSpec
from trackinizer.wire.wire_sessions import SessionStart, SessionStartResponse
from trackinizer.wire.wire_terminals import TerminalHost, TerminalHosts

import pytest

from knowop.hosts_test import _KNOWHOST, _KnowHostApi
from knowop.process.main_test import _attached
from knowop.tools import knowop
from knowop.tools.records import RecordReads
from knowop.tools.records_test import _Server


if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

    from trackinizer.types.inquiries import Inquiry
    from trackinizer.wire.refs import Ref
    from trackinizer.wire.wire_job import JobState
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
        self.limits: list[PlainTree] | None = []
        self.limits_refusal = 0
        self.puts: list[tuple[str, object]] = []
        self.campaigns: list[PlainTree] = []
        self.campaigns_refusal = 0
        self.campaigns_asked: list[str] = []
        self.jobs: list[JobView] = []
        self.jobs_refusal = 0
        self.hosts = [
            TerminalHost(
                name="laptop",
                status="online",
                terminal_ready=True,
                confirmed=True,
            ),
        ]

    def job_list(self) -> JobList:
        if self.jobs_refusal:
            raise ClientError(
                f"GET /api/addons/jobs/jobs -> {self.jobs_refusal}",
                status_code=self.jobs_refusal,
            )
        return JobList(jobs=self.jobs)

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
        if path == "/api/addons/knowspend/limits":
            if self.limits is None or self.limits_refusal:
                status = self.limits_refusal or 404
                raise ClientError(f"GET {path} -> {status}", status_code=status)
            return {"account": "ada@x.test", "limits": self.limits}
        if path.startswith("/api/addons/knowspend/campaigns?"):
            self.campaigns_asked.append(path)
            if self.campaigns_refusal:
                status = self.campaigns_refusal
                raise ClientError(f"GET {path} -> {status}", status_code=status)
            return self.campaigns
        if path.startswith("/api/addons/"):
            return self.api.get(path)
        assert path in (
            f"/api/inquiries/{UUID(int=11)}",
            f"/api/inquiries/{UUID(int=15)}",
        )
        return {"seq": 412}

    def post(self, path: str, **options: object) -> PlainTree:
        return self.api.post(path, **options)

    def put(self, path: str, *, body: object = None) -> PlainTree:
        self.puts.append((path, body))
        return None

    def terminal_hosts(self, *, owner: str | None = None) -> TerminalHosts:
        assert owner is None
        return TerminalHosts(hosts=self.hosts)

    def session_start(self, body: SessionStart) -> SessionStartResponse:
        self.starts.append(body)
        if self.failure is not None:
            raise self.failure
        return SessionStartResponse(id=UUID(int=11), seq=0)


def _tools(client: _Client, server: _Server | None = None) -> knowop.KnowOpTools:
    return knowop.KnowOpTools(
        client,
        session_id=UUID(int=90),
        records=_reads(server or _Server()),
    )


def _reads(server: _Server) -> RecordReads:
    """Return record reads over ``server``, under the agent's key."""
    return RecordReads("https://records.test", api_key="agent-key", transport=server)


@pytest.mark.asyncio
async def test_list_names_hosts_states_and_rental_availability() -> None:
    client = _Client()
    assert await _tools(client).act(action="host.list") == (
        "laptop: online\nThis server rents KnowHosts.\nknowhost-0b5e9a101c2d: running"
    )
    client.api.refuse = 404
    assert await _tools(client).act(action="host.list") == (
        "laptop: online\nThis server rents no KnowHosts."
    )


@pytest.mark.asyncio
async def test_list_without_a_host_says_gpu_jobs_need_one_and_how_to_get_one() -> None:
    client = _Client()
    client.hosts = []
    client.api.knowhosts = []
    assert await _tools(client).act(action="host.list") == (
        "This server rents KnowHosts.\n"
        "No host: GPU jobs and commands need one. The person connects one from Hosts, "
        "or host.start rents one."
    )
    client.api.refuse = 404
    assert await _tools(client).act(action="host.list") == (
        "This server rents no KnowHosts.\n"
        "No host: GPU jobs and commands need one. The person connects one from Hosts."
    )


@pytest.mark.asyncio
async def test_start_sends_once_waits_150_seconds_and_names_online_host() -> None:
    client = _Client()
    client.api.answer["tools"] = False
    assert await _tools(client).act(action="host.start") == (
        "knowhost-0b5e9a101c2d is running: open its terminal from Hosts.\n"
        "Its knowop and trax are signed in as you.\n"
        "Its knowop serves terminals but no tools."
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
    assert said.splitlines()[1:] == [
        "It rebooted from its files: its processes were lost.",
    ]
    assert said.startswith("knowhost-0b5e9a101c2d is running again.")
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
    assert (
        await _tools(client).act(action="session.launch")
        == "session.launch requires a time limit"
    )
    assert client.starts == client.started == []


@pytest.mark.asyncio
async def test_preflight_tool_has_no_caller_and_opens_no_session() -> None:
    client = _Client()
    tools = knowop.KnowOpTools(client, session_id=None, records=_reads(_Server()))
    assert (
        await tools.act(action="session.launch", time="20")
        == "session.launch requires a caller session"
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
async def test_launch_defaults_the_label_and_the_most_per_campaign() -> None:
    client = _Client()
    assert await _tools(client).act(action="session.launch", time="3") == "Agent#412"
    assert client.started == [
        ProcessSpec(
            args=(
                "--time",
                "0-00:03:00",
                "--name",
                "Agent",
                "--max-budget-usd",
                "5",
            ),
        ),
    ]


_EACH_CAMPAIGN: Final[dict[str, PlainTree]] = {
    "micros": 20_500_000,
    "work": None,
    "ref": "campaign/*",
    "window": None,
    "stops": True,
    "model": "",
    "lane": None,
}

_RATE: Final[dict[str, PlainTree]] = {
    **_EACH_CAMPAIGN,
    "window": {"every": 1, "unit": "hour", "calendar": False},
}
_MODEL: Final[dict[str, PlainTree]] = {
    **_EACH_CAMPAIGN,
    "work": "model",
    "model": "sol",
}
_SET: Final[dict[str, PlainTree]] = {"micros": 30_000_000, "ref": "campaign/*"}

_WHERE: Final = "The person sets it as Each campaign budget under Budgets and alerts on the Usage page."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limits", "said"),
    [
        (
            [_EACH_CAMPAIGN],
            f"The campaign limit is $20.5, the person's setting. {_WHERE}",
        ),
        (None, f"The campaign limit is $5, this site's default. {_WHERE}"),
        # A rate on each campaign, the whole account's total, or one kind of work,
        # model or lane's is not the most per campaign.
        *(
            (
                [{**_EACH_CAMPAIGN, **other}],
                f"The campaign limit is $5, this site's default. {_WHERE}",
            )
            for other in (
                {"window": {"every": 1, "unit": "hour", "calendar": False}},
                {"ref": ""},
                {"work": "gpu"},
                {"work": "model", "model": "sol"},
                {"lane": "autonomy"},
                {"stops": False},
            )
        ),
    ],
)
async def test_budget_names_the_most_each_campaign_may_spend_and_where_it_is_set(
    limits: list[PlainTree] | None,
    said: str,
) -> None:
    client = _Client()
    client.limits = limits
    assert await _tools(client).act(action="session.budget") == said
    assert client.starts == client.started == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limits", "sent"),
    [
        pytest.param(
            [_RATE, _EACH_CAMPAIGN, _MODEL],
            [_RATE, _SET, _MODEL],
            id="replaces",
        ),
        pytest.param([_RATE, _MODEL], [_RATE, _MODEL, _SET], id="adds"),
    ],
)
async def test_set_budget_changes_only_the_campaign_limit(
    limits: list[PlainTree],
    sent: list[PlainTree],
) -> None:
    client = _Client()
    client.limits = limits
    assert (
        await _tools(client).act(action="session.set_budget", budget=30)
        == "The campaign limit is now $30."
    )
    assert client.puts == [
        (
            "/api/addons/knowspend/limits",
            {"account": "ada@x.test", "limits": sent},
        ),
    ]
    assert client.starts == client.started == []


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [None, -1])
async def test_set_budget_needs_an_amount_and_changes_nothing(
    budget: float | None,
) -> None:
    client = _Client()
    assert (
        await _tools(client).act(action="session.set_budget", budget=budget)
        == "session.set_budget requires a budget of $0 or more"
    )
    assert client.puts == []


@pytest.mark.asyncio
async def test_set_budget_on_a_server_without_usage_changes_nothing() -> None:
    client = _Client()
    client.limits = None
    assert (
        await _tools(client).act(action="session.set_budget", budget=30)
        == "This server keeps no campaign limit."
    )
    assert client.puts == []


@pytest.mark.asyncio
async def test_a_refused_limits_read_is_the_answer_and_starts_nothing() -> None:
    client = _Client()
    client.limits_refusal = 403
    assert (
        await _tools(client).act(action="session.launch", time="3")
        == "the server answered 403."
    )
    assert client.starts == client.started == []


def _job(state: JobState, *, campaign: str = "Issue#7") -> JobView:
    return JobView(
        id=1,
        state=state,
        node=None,
        gpus=1,
        minutes=5,
        label="",
        argv=["true"],
        commit="0" * 40,
        bundle_sha256=None,
        created_at=datetime(2026, 11, 17, tzinfo=UTC),
        started_at=None,
        ended_at=None,
        usd=None,
        reason="",
        upload_serial=None,
        campaign=campaign,
    )


def _spent(budget: int | None) -> list[PlainTree]:
    """Return usage's answer for Issue#7, as the route sends it."""
    spent: dict[str, PlainTree] = {
        "ref": "campaign/Issue#7",
        "spent": 2_500_000,
        "budget": budget,
        "last_hour": 400_000,
        "reaches_at": None,
        "alerts": [],
    }
    return [spent]


_NO_BUDGET: Final = (
    "campaign/Issue#7 spent $2.50, $0.40 of it in the last hour; it has no budget."
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("budget", "jobs", "said"),
    [
        pytest.param(
            20_000_000,
            [
                _job("running"),
                _job("queued"),
                _job("completed"),
                _job("running", campaign="Issue#8"),
                _job("running", campaign=""),
            ],
            "campaign/Issue#7 spent $2.50 of its $20.00 budget, $0.40 of it in the "
            "last hour. 2 of its GPU jobs still run, so the figure is not final.",
            id="running",
        ),
        pytest.param(
            None,
            [_job("starting"), _job("cancelled")],
            f"{_NO_BUDGET} 1 of its GPU jobs still runs, so the figure is not final.",
            id="one",
        ),
        pytest.param(
            None,
            [_job("failed"), _job("running", campaign="Issue#8")],
            f"{_NO_BUDGET} None of its GPU jobs runs.",
            id="settled",
        ),
    ],
)
async def test_budget_of_a_campaign_reads_its_spend_and_its_unfinished_gpu_jobs(
    budget: int | None,
    jobs: list[JobView],
    said: str,
) -> None:
    client = _Client()
    client.campaigns = _spent(budget)
    client.jobs = jobs
    assert await _tools(client).act(action="session.budget", name="Issue#7") == said
    assert client.campaigns_asked == [
        "/api/addons/knowspend/campaigns?ref=campaign%2FIssue%237",
    ]
    assert client.puts == client.starts == client.started == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("campaigns", "jobs", "said"),
    [
        (404, 0, "This server keeps no campaign spend."),
        (403, 0, "the server answered 403."),
        (0, 404, f"{_NO_BUDGET} This server runs no GPU jobs."),
        (0, 503, f"{_NO_BUDGET} This server runs no GPU jobs."),
        (0, 403, "the server answered 403."),
    ],
)
async def test_budget_of_a_campaign_says_what_a_server_without_it_or_a_refusal_says(
    campaigns: int,
    jobs: int,
    said: str,
) -> None:
    client = _Client()
    client.campaigns = _spent(None)
    client.campaigns_refusal = campaigns
    client.jobs_refusal = jobs
    assert await _tools(client).act(action="session.budget", name="Issue#7") == said


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["campaign/Issue#7", "Issue#0", "Issue#07", "lead#4"])
async def test_budget_of_anything_but_one_brief_reads_nothing(name: str) -> None:
    client = _Client()
    assert (
        await _tools(client).act(action="session.budget", name=name)
        == "session.budget names a campaign by its brief, Issue#N; "
        "or nothing, for the campaign limit."
    )
    assert client.campaigns_asked == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "name"),
    [("session.launch", "lead"), ("session.continue", "lead#412")],
)
async def test_a_confirmed_budget_over_the_limit_starts_with_that_budget(
    action: knowop.Action,
    name: str,
) -> None:
    client = _Client()
    client.limits = [_EACH_CAMPAIGN]
    tools = _tools(client)
    assert await tools.act(action=action, name=name, time="3") == "lead#412"
    assert client.started[-1].args[-2:] == ("--max-budget-usd", "20.5")
    assert await tools.act(action=action, name=name, time="3", budget=30) == "lead#412"
    assert client.started[-1].args[-2:] == ("--max-budget-usd", "30")


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["starting", "attached", "detached"])
async def test_status_reads_one_named_sessions_process(state: str) -> None:
    client = _Client()
    client.answer = _attached(state)
    said = await _tools(client).act(action="session.status", name="lead#412")
    assert said == f"lead#412: {state}" + (
        "; it crashed" if state == "detached" else ""
    )
    assert client.resolved == [SeqRef(kind="AgentSession", seq=412)]
    assert client.starts == client.started == []


@pytest.mark.asyncio
async def test_continue_restarts_the_same_session_with_a_new_time_limit() -> None:
    client = _Client()
    assert (
        await _tools(client).act(
            action="session.continue",
            name="lead#412",
            time="3",
            model="sol",
            effort="high",
        )
        == "lead#412"
    )
    assert client.resolved == [SeqRef(kind="AgentSession", seq=412)]
    assert client.starts == []
    assert client.started == [
        ProcessSpec(
            args=(
                "--time",
                "0-00:03:00",
                "--name",
                "lead",
                "--model",
                "sol",
                "--model.thinking.effort",
                "high",
                "--max-budget-usd",
                "5",
            ),
        ),
    ]


@pytest.mark.asyncio
async def test_continue_requires_a_new_time_limit_before_starting() -> None:
    client = _Client()
    assert (
        await _tools(client).act(action="session.continue", name="lead#412")
        == "continue requires a new time limit"
    )
    assert client.starts == client.started == []


@pytest.mark.asyncio
async def test_status_requires_a_named_session() -> None:
    client = _Client()
    assert "names no agent" in await _tools(client).act(action="session.status")
    assert client.starts == client.started == []


@dataclass(frozen=True, slots=True, kw_only=True)
class _Agent:
    label: str


@pytest.mark.asyncio
async def test_self_names_the_calling_agent_and_its_session() -> None:
    client = _Client()
    tools = knowop.KnowOpTools(
        client,
        session_id=UUID(int=11),
        records=_reads(_Server()),
    )
    said = await tools.act(action="session.self", context=_Agent(label="lead"))
    assert said == (
        f"lead#{UUID(int=11)}: you, the agent lead of session {UUID(int=11)}. "
        f"Pass it as --session lead#{UUID(int=11)} to knowop watch-pr."
    )
    assert (await tools.act(action="session.self")).startswith(
        f"Agent#{UUID(int=11)}: ",
    )
    assert client.starts == client.started == []


@pytest.mark.asyncio
async def test_self_without_a_session_says_so() -> None:
    tools = knowop.KnowOpTools(_Client(), session_id=None, records=_reads(_Server()))
    assert (
        await tools.act(action="session.self")
        == "session.self requires a caller session"
    )


@pytest.mark.asyncio
async def test_stop_with_no_process_does_not_report_success() -> None:
    client = _Client()
    assert (
        await _tools(client).act(action="session.stop", name="lead#412")
        == "lead#412 has had no agent process"
    )


@pytest.mark.asyncio
async def test_trax_reads_records_with_the_agents_own_key() -> None:
    """The command runs as trax reads it, over the reads' client, not KnowOp's."""
    client = _Client()
    server = _Server()

    said = await _tools(client, server).act(action="trax.read", command="trax version")

    assert said == "abc123\n"
    assert server.sent == [("GET", "/api/version", "Bearer agent-key")]
    assert client.starts == client.started == client.puts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("command", [None, "", "  "])
async def test_trax_requires_a_command_and_sends_nothing(command: str | None) -> None:
    """None leaves ``command`` out, as a model that names no command does."""
    server = _Server()
    tools = _tools(_Client(), server)

    said = await (
        tools.act(action="trax.read")
        if command is None
        else tools.act(action="trax.read", command=command)
    )

    assert said == "trax.read requires a command"
    assert (server.sent, server.closed) == ([], 0)


@pytest.mark.asyncio
async def test_trax_sends_no_write_and_says_so() -> None:
    """A write the agent asks KnowOp for is refused before it leaves, as text."""
    server = _Server()

    said = await _tools(_Client(), server).act(
        action="trax.read",
        command="issue 7 status to complete",
    )

    assert "is not available to this client" in said
    assert [method for method, _, _ in server.sent] == ["GET"]


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
