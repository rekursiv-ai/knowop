"""The KnowOp tool for hosts and agent sessions."""

from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypedDict
from uuid import UUID, uuid4

import argparse
import asyncio

from trackinizer.client.errors import ClientError, server_answer
from trackinizer.wire.refs import SeqRef
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
from knowop.process.command import time_limit
from knowop.process.start import ProcessClient, start_args, started_process


if TYPE_CHECKING:
    from trackinizer.types.inquiries import Inquiry
    from trackinizer.wire.refs import Ref
    from trackinizer.wire.wire_session_relay import AgentProcess
    from trackinizer.wire.wire_sessions import SessionStartResponse
    from trackinizer.wire.wire_terminals import TerminalHosts

KNOWOP_TOOL: Final = "KnowOp"
KNOWOP_DESCRIPTION: Final = (files(__package__) / "assets" / "knowop.md").read_text(
    encoding="utf-8",
)

type Action = Literal[
    "host.list",
    "host.start",
    "host.pause",
    "host.resume",
    "session.launch",
    "session.stop",
    "session.status",
    "session.continue",
]


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


class KnowOpTools:
    """Operate on the caller's hosts and agent sessions."""

    def __init__(self, client: KnowOpClient, *, session_id: UUID | None) -> None:
        self._client = client
        self._session_id = session_id

    async def act(
        self,
        *,
        action: Action,
        name: str = "",
        time: str = "",
        model: str = "",
        effort: str = "",
        budget: float | None = None,
    ) -> str:
        """Operate on a host or launch or stop an agent session.

        Args:
          action: Operate hosts, or launch, stop, inspect or continue a session.
          name: The host name, launch label, or required LABEL#N for session actions.
          time: Required launch/continue Slurm time limit (20 means 20 minutes).
          model: Launch model; empty keeps the server's default.
          effort: Launch thinking effort; empty keeps the default.
          budget: Launch budget in USD; omitted keeps the default.

        Returns:
          answer: The result or the server's refusal.

        """
        try:
            return await asyncio.to_thread(
                self._act,
                action=action,
                name=name,
                time=time,
                model=model,
                effort=effort,
                budget=budget,
            )
        except ClientError as error:
            return server_answer(error)
        except (ValueError, argparse.ArgumentTypeError, TimeoutError) as error:
            return str(error)

    def _act(
        self,
        *,
        action: Action,
        name: str,
        time: str,
        model: str,
        effort: str,
        budget: float | None,
    ) -> str:
        client = self._client
        match action:
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
            case "session.stop" | "session.status" | "session.continue":
                target = SessionAgent.parse(name)
                if action == "session.continue" and not time:
                    return "continue requires a new time limit"
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
                        time=time,
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
                if not time:
                    return "session.launch requires a time limit"
                if self._session_id is None:
                    return "session.launch requires a caller session"
                return self._started_session(
                    name=name,
                    time=time,
                    model=model,
                    effort=effort,
                    budget=budget,
                )

    def _started_session(
        self,
        *,
        name: str,
        time: str,
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
        if budget is not None:
            args.extend(("--max-budget-usd", str(budget)))
        built = start_args(time_limit(time), args)
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
        number = from_plain(
            self._client.get(f"/api/inquiries/{session_id}"),
            _Numbered,
        )["seq"]
        return f"{label}#{number}"


class _Numbered(TypedDict):
    seq: int
