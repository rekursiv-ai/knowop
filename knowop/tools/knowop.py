"""The KnowOp tool: hosts, agent sessions, campaign scores and Trackinizer's records."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypedDict
from urllib.parse import urlencode
from uuid import UUID, uuid4

import asyncio
import re

from trackinizer.client.errors import ClientError, server_answer
from trackinizer.wire.refs import SeqRef
from trackinizer.wire.wire_job import TERMINAL_STATES
from trackinizer.wire.wire_science_chat import chat_session_id
from trackinizer.wire.wire_session_relay import ProcessSpec, remote_flags_refused
from trackinizer.wire.wire_sessions import SessionAgent, SessionStart
from treekle import from_plain

from knowop.hosts import (
    KnowHostClient,
    listed_knowhosts,
    named_knowhost,
    pause_knowhost,
    paused_line,
    resume_knowhost,
    resumed_lines,
    start_knowhost,
    started_lines,
)
from knowop.jobs.job_records import NOT_HERE
from knowop.process.start import ProcessClient, started_process


if TYPE_CHECKING:
    from trackinizer.types.inquiries import Inquiry
    from trackinizer.wire.refs import Ref
    from trackinizer.wire.wire_job import JobList
    from trackinizer.wire.wire_session_relay import AgentProcess
    from trackinizer.wire.wire_sessions import SessionStartResponse
    from trackinizer.wire.wire_terminals import TerminalHosts
    from treekle import PlainTree

    from knowop.tools.records import Records

KNOWOP_TOOL: Final = "KnowOp"
KNOWOP_DESCRIPTION: Final = (files(__package__) / "assets" / "knowop.md").read_text(
    encoding="utf-8",
)

_SITE_MOST_USD: Final = 5.0
"""The most each campaign may spend where the person set no Each campaign budget."""

type Action = Literal[
    "host.list",
    "host.start",
    "host.pause",
    "host.resume",
    "session.budget",
    "session.set_budget",
    "session.launch",
    "session.stop",
    "session.status",
    "session.continue",
    "session.self",
    "usage.today",
    "trax.read",
    "trax.write",
    "campaign.score",
]

type Verdict = Literal["works", "doesnt", "inconclusive"]  # codespell:ignore doesnt
"""A campaign iteration's keep-or-revert call, as the campaign view reads it."""


class Caller(Protocol):
    """The agent that calls the tool: an agent framework's context, such as sagent's."""

    @property
    def label(self) -> str:
        """Its label in its session, as ``LABEL#N`` names it; empty for the default."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class _Unnamed:
    """The caller of a tool that names none: the session's root agent."""

    label: str = ""


_UNNAMED: Final = _Unnamed()


class KnowOpClient(KnowHostClient, ProcessClient, Protocol):
    """The server calls used by KnowOp."""

    def terminal_hosts(self, *, owner: str | None = None) -> TerminalHosts:
        """List the person's hosts."""
        ...

    def session_start(self, body: SessionStart) -> SessionStartResponse:
        """Open a session."""
        ...

    def resolve_id(self, ref: Ref) -> tuple[Inquiry.InquiryKind, UUID]:
        """Resolve a session's number."""
        ...

    def stop_process(self, session_id: UUID) -> AgentProcess | None:
        """Stop a session's process."""
        ...

    def read_process(self, session_id: UUID) -> AgentProcess | None:
        """Read the session's latest process state."""
        ...

    def put(self, path: str, *, body: object = None) -> PlainTree:
        """Replace a server resource."""
        ...

    def job_list(self) -> JobList:
        """List the person's GPU jobs."""
        ...


class KnowOpTools:
    """Run the caller's hosts and sessions, score its campaigns, read and write records."""

    def __init__(
        self,
        client: KnowOpClient,
        *,
        session_id: UUID | None,
        records: Records,
    ) -> None:
        self._client = client
        self._session_id = session_id
        self._records = records

    async def act(
        self,
        *,
        action: Action,
        name: str = "",
        model: str = "",
        effort: str = "",
        budget: float | None = None,
        command: str = "",
        experiment: str = "",
        metric: str = "",
        value: float | None = None,
        verdict: Verdict | None = None,
        reason: str = "",
        context: Caller = _UNNAMED,
    ) -> str:
        """Operate on a host or agent session, read or write records, or score.

        Args:
          action: Operate hosts; launch, stop, inspect or continue a session; read
            records (trax.read) or change them (trax.write); or score a campaign
            iteration (campaign.score).
          name: The host name, launch label, required LABEL#N for session actions, or
            for ``session.budget`` a campaign's brief, Issue#N, to read its spend.
          model: Launch model; empty keeps the server's default.
          effort: Launch thinking effort; empty keeps the default.
          budget: Launch budget in USD; omitted takes the person's campaign limit.
          command: Required for trax.read and trax.write: a trax command, as typed
            after `trax`.
          experiment: Required for campaign.score: the iteration's Experiment,
            Experiment#N or N.
          metric: Required for campaign.score: the brief's metric, one lowercase
            name such as val_loss, the same on every iteration.
          value: Required for campaign.score: the iteration's score on ``metric``.
          verdict: For campaign.score after the baseline, required: ``works``,
            ``doesnt`` or ``inconclusive``; the baseline sends none.
          reason: For campaign.score after the baseline, required: one sentence of
            at most 200 characters saying why; the baseline sends none.
          context: The calling agent, whose label ``session.self`` answers with and
            trax.write records each change as; the agent framework passes it, and it
            is no argument of the model's.

        Returns:
          answer: The result or the server's refusal.

        """
        try:
            return await asyncio.to_thread(
                self._act,
                action=action,
                name=name,
                model=model,
                effort=effort,
                budget=budget,
                command=command,
                score=_Score(
                    experiment=experiment,
                    metric=metric,
                    value=value,
                    verdict=verdict,
                    reason=reason,
                ),
                label=context.label or "Agent",
            )
        except ClientError as error:
            return server_answer(error)
        except (ValueError, TimeoutError) as error:
            return str(error)

    def _act(
        self,
        *,
        action: Action,
        name: str,
        model: str,
        effort: str,
        budget: float | None,
        command: str,
        score: _Score,
        label: str,
    ) -> str:
        client = self._client
        match action:
            # Named by UUID: a process key reads no inquiry of its own session, only
            # of the sessions its agents spawned.
            case "session.self":
                if self._session_id is None:
                    return "session.self requires a caller session"
                named = f"{label}#{self._session_id}"
                return (
                    f"{named}: you, the agent {label} of session {self._session_id}. "
                    f"Pass it as --session {named} to knowop watch-pr."
                )
            case "host.list":
                registered = client.terminal_hosts().hosts
                rented = listed_knowhosts(client)
                return "\n".join(
                    [
                        *(f"{host.name}: {host.status}" for host in registered),
                        "This server rents no KnowHosts."
                        if rented is None
                        else "This server rents KnowHosts.",
                        *(f"{host['host']}: {host['state']}" for host in rented or ()),
                        *(
                            ()
                            if registered
                            else (
                                "No host: GPU jobs and commands need one. The person "
                                "connects one from Hosts"
                                + (
                                    "."
                                    if rented is None
                                    else ", or host.start rents one."
                                ),
                            )
                        ),
                    ],
                )
            case "host.start":
                return "\n".join(started_lines(start_knowhost(client)))
            case "host.pause":
                return paused_line(pause_knowhost(client, named_knowhost(client, name)))
            case "host.resume":
                return "\n".join(
                    resumed_lines(
                        resume_knowhost(client, named_knowhost(client, name)),
                    ),
                )
            case "session.budget" if name:
                return self._campaign_spend(name)
            case "session.budget":
                most, source = self._campaign_most()
                return (
                    f"The campaign limit is ${most:g}, {source}. The person sets it "
                    "as Each campaign budget under Budgets and alerts on the Usage page."
                )
            case "session.set_budget":
                return self._set_campaign_most(budget)
            case "session.stop" | "session.status" | "session.continue":
                target = SessionAgent.parse(name)
                session_id = (
                    client.resolve_id(SeqRef(kind="AgentSession", seq=target.session))[
                        1
                    ]
                    if isinstance(target.session, int)
                    else target.session
                )
                if action == "session.continue":
                    return self._started_session(
                        name=target.label,
                        model=model,
                        effort=effort,
                        budget=budget,
                        session_id=session_id,
                    )
                process = (
                    client.stop_process(session_id)
                    if action == "session.stop"
                    else client.read_process(session_id)
                )
                if process is None:
                    return f"{target} has had no agent process"
                if action == "session.stop":
                    return f"stopped {target}"
                return f"{target}: {process.state}" + (
                    f"; {process.reason}" if process.reason else ""
                )
            case "session.launch":
                if self._session_id is None:
                    return "session.launch requires a caller session"
                return self._started_session(
                    name=name,
                    model=model,
                    effort=effort,
                    budget=budget,
                )
            case "usage.today":
                return self._today()
            case "trax.read":
                if not command.strip():
                    return "trax.read requires a command"
                return self._records.read(command)
            case "trax.write":
                if not command.strip():
                    return "trax.write requires a command"
                if self._session_id is None:
                    return "trax.write requires a caller session"
                return self._records.write(
                    command,
                    actor=f"{label}#{self._session_id}",
                )
            case "campaign.score":
                return self._scored(score)

    # The server decides whether an Experiment is a baseline or a later iteration,
    # and refuses a stamp that does not fit; the tool sends what it was given.
    def _scored(self, score: _Score) -> str:
        """Stamp an iteration with ``score``; answer the server's line."""
        if not score.experiment:
            return "campaign.score requires an experiment"
        if not score.metric:
            return "campaign.score requires a metric"
        if score.value is None:
            return "campaign.score requires a value"
        named = re.fullmatch(r"(?:Experiment#)?([1-9][0-9]*)", score.experiment)
        if named is None:
            return "campaign.score takes its experiment as Experiment#N or N, N from 1."
        try:
            answer = self._client.post(
                f"/api/addons/campaignscores/iterations/{named[1]}",
                body={
                    "metric": score.metric,
                    "value": score.value,
                    "verdict": score.verdict,
                    "reason": score.reason,
                },
            )
        except ClientError as error:
            if error.status_code != 404:
                raise
            return "This server has no campaign scoring."
        return from_plain(answer, _Line)["line"]

    def _started_session(
        self,
        *,
        name: str,
        model: str,
        effort: str,
        budget: float | None,
        session_id: UUID | None = None,
    ) -> str:
        label = name or "Agent"
        if not label.strip():
            raise ValueError("the agent label must not be blank")
        args = ["--name", label]
        for flag, value in (("--model", model), ("--model.thinking.effort", effort)):
            if value:
                args.extend((flag, value))
        if budget is None:
            budget, _ = self._campaign_most()
        args.extend(("--max-budget-usd", f"{budget:g}"))
        built = tuple(args)
        if refused := remote_flags_refused(built):
            return refused
        if session_id is None:
            session_id = self._client.session_start(
                SessionStart(
                    cli="sagent",
                    cli_session_id=chat_session_id(uuid4()),
                    spawned_by=self._session_id,
                ),
            ).id
        _ = started_process(
            self._client,
            session_id=session_id,
            spec=ProcessSpec(args=built),
        )
        return f"{label}#{self._number(session_id)}"

    def _number(self, session_id: UUID) -> int:
        """Return the session's number, as ``AgentSession#N`` and ``LABEL#N`` name it."""
        return from_plain(
            self._client.get(f"/api/inquiries/{session_id}"),
            _Numbered,
        )["seq"]

    # The campaign limit is the person's Each campaign budget on the Usage page: their
    # total on each campaign, on every kind of work, model and lane. A server without
    # usage, or a person who set none, takes this site's default.
    def _campaign_most(self) -> tuple[float, str]:
        """Return the campaign limit, in USD, and where it is set."""
        held = self._limits()
        for limit in [] if held is None else held["limits"]:
            if _is_each_campaign(limit):
                return float(limit["micros"]) / 1_000_000, "the person's setting"
        return _SITE_MOST_USD, "this site's default"

    def _set_campaign_most(self, budget: float | None) -> str:
        """Set the campaign limit to ``budget`` USD, keeping every other limit."""
        if budget is None or budget < 0:
            return "session.set_budget requires a budget of $0 or more"
        held = self._limits()
        if held is None:
            return "This server keeps no campaign limit."
        limits: list[_Limit | _NewLimit] = list(held["limits"])
        new = _NewLimit(micros=round(budget * 1_000_000), ref="campaign/*")
        at = next(
            (at for at, each in enumerate(held["limits"]) if _is_each_campaign(each)),
            None,
        )
        if at is None:
            limits.append(new)
        else:
            limits[at] = new
        _ = self._client.put(
            "/api/addons/knowspend/limits",
            body={"account": held["account"], "limits": limits},
        )
        return f"The campaign limit is now ${budget:g}."

    # A job's spend is charged while it runs, so the sum grows until each job of the
    # campaign is in a state it never leaves.
    def _campaign_spend(self, name: str) -> str:
        """Return what campaign ``name`` (``Issue#7``) spent, and if that is final."""
        if re.fullmatch(r"Issue#[1-9][0-9]*", name) is None:
            return (
                "session.budget names a campaign by its brief, Issue#N; or nothing, "
                "for the campaign limit."
            )
        ref = f"campaign/{name}"
        try:
            [spend] = from_plain(
                self._client.get(
                    f"/api/addons/knowspend/campaigns?{urlencode({'ref': ref})}",
                ),
                list[_CampaignSpend],
            )
        except ClientError as error:
            if error.status_code != 404:
                raise
            return "This server keeps no campaign spend."
        spent, hour = spend["spent"] / 1_000_000, spend["last_hour"] / 1_000_000
        said = (
            f"{ref} spent ${spent:.2f}, ${hour:.2f} of it in the last hour; it has no "
            "budget."
            if spend["budget"] is None
            else f"{ref} spent ${spent:.2f} of its ${spend['budget'] / 1_000_000:.2f} "
            f"budget, ${hour:.2f} of it in the last hour."
        )
        return f"{said} {self._unfinished_jobs(name)}"

    def _unfinished_jobs(self, name: str) -> str:
        """Say how many of campaign ``name``'s GPU jobs have not ended."""
        try:
            jobs = self._client.job_list().jobs
        except ClientError as error:
            if error.status_code not in NOT_HERE:
                raise
            return "This server runs no GPU jobs."
        running = sum(
            each.campaign == name and each.state not in TERMINAL_STATES for each in jobs
        )
        if not running:
            return "None of its GPU jobs runs."
        return (
            f"{running} of its GPU jobs still {'runs' if running == 1 else 'run'}, so "
            "the figure is not final."
        )

    def _today(self) -> str:
        """Say what the person spent today, of their allowance, and the credit left."""
        try:
            today = from_plain(self._client.get("/api/addons/knowspend/today"), _Today)
        except ClientError as error:
            if error.status_code != 404:
                raise
            return "This server keeps no usage."
        limit = today["today_limit"]
        return (
            f"Today you spent ${today['today_spent'] / 1_000_000:.2f}, "
            f"${today['today_allowance_used'] / 1_000_000:.2f} of it from today's "
            f"${today['today_allowance'] / 1_000_000:.2f} allowance; "
            f"${today['credit_left'] / 1_000_000:.2f} of credit is left. "
            "Campaigns and other agent work may spend "
            f"${today['today_remaining'] / 1_000_000:.2f} more today; "
            f"${today['reserve'] / 1_000_000:.2f} is held back for Chat and KnowHosts. "
            + (
                "No daily limit is set."
                if limit is None
                else f"The daily limit is ${limit / 1_000_000:.2f}."
            )
            + " The allowance starts again at midnight UTC."
        )

    def _limits(self) -> _Limits | None:
        """Return the person's limits; ``None`` from a server without usage."""
        try:
            return from_plain(self._client.get("/api/addons/knowspend/limits"), _Limits)
        except ClientError as error:
            if error.status_code != 404:
                raise
            return None


def _is_each_campaign(limit: _Limit) -> bool:
    """Return whether ``limit`` is the person's campaign limit, as usage reads it."""
    return (
        limit["ref"] == "campaign/*"
        and limit["window"] is None
        and limit["work"] is None
        and not limit["model"]
        and limit["lane"] is None
        and limit["stops"]
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class _Score:
    """What ``campaign.score`` was given; an empty or ``None`` field was not."""

    experiment: str
    metric: str
    value: float | None
    verdict: Verdict | None
    reason: str


class _Line(TypedDict):
    line: str


class _Numbered(TypedDict):
    seq: int


class _CampaignSpend(TypedDict):
    spent: int
    budget: int | None
    last_hour: int


# The fields of knowspend's ``Today`` this tool reads. Its own type is not imported:
# the client ships alone as ``knowop``, which carries no knowspend
# (``import_boundary_test``).
class _Today(TypedDict):
    today_spent: int
    today_allowance: int
    today_allowance_used: int
    credit_left: int
    today_limit: int | None
    today_remaining: int
    reserve: int


class _Limit(TypedDict):
    micros: int
    ref: str
    window: dict[str, int | str | bool] | None
    work: str | None
    model: str
    lane: str | None
    stops: bool


class _NewLimit(TypedDict):
    micros: int
    ref: str


class _Limits(TypedDict):
    account: str
    limits: list[_Limit]
