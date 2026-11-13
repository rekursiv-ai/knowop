"""The entry point of the agent process: one session's sagent, live in Trackinizer.

``knowop run`` runs sagent's own agent in this process, with sagent's screen at a
terminal or with none, and makes its session live: each event and the agent's state
go up the session's socket, its journal goes up as records, and what a person sends
from Chat comes down the socket into the runtime, as a key typed at the screen does.
A slash line runs on the one ``Commands`` the screen runs typed lines on, so the
terminal and Chat each show what the other ran.

::

    knowop run [SAGENT FLAGS]          sagent's screen at a terminal, else headless
    knowop run --headless              headless at a terminal
    knowop run --session ID            continue session ID here
    knowop run --on NAME [--cwd DIR]   start on host NAME through its daemon
    knowop run --login-host NAME       with --on, or alone: use host NAME's model login
    knowop run --stop ID               stop session ID's agent, wherever it runs
    knowop run --time TIME             end TIME after it starts, as a stop does (Slurm's --time)
    knowop run --serve URL             headless; reads one ServeSession from stdin

Every sagent flag works but those that pick a journal: knowop keeps the session's.
A start the server placed for a remote agent opens without sagent's own daemon, which
would otherwise start beside each agent and stay, and runs only the slash lines
``remote_commands.py`` lists.
A run prints the session's id alone as its first line on stdout, once the server
has its ``Hello``. A run started here has the server place its session in a workspace
and a canvas, by the canvas it was started from (``TRAX_CANVAS``) and its checkout,
and its tools read where in ``TRAX_WORKSPACE`` and ``TRAX_CANVAS``.

Exit codes: 0 after a stop, its time limit, a screen's quit or a ``/quit``; 1 for a start error, a
refused key, a revoked host or a session that is gone; 3 when another process holds
the session, continued it elsewhere, or holds its journal; 4 when headless and
without a socket for 10 minutes; 130 on Ctrl+C or SIGTERM.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast
from uuid import UUID, uuid4

import argparse
import asyncio
import contextlib
import logging
import os
import queue
import re
import sys
import threading
import time

from sagent.runtime import types
from sagent.session.jsonl import JournalInUseError
from sagent.session.types import AgentToAgentMessage
from sagent.tui.main import StartError, opened, parse_flags, run, signals
from sagent.tui.prompt import Panes
from sagent.tui.render import error_text
from sagent.tui.slash import Commands
from sagent.tui.types import CommandOutput
from trackinizer.client.client import Client
from trackinizer.client.conversation import PartReader
from trackinizer.client.errors import ClientError
from trackinizer.client.session_socket import (
    GAVE_UP,
    SessionSocket,
    SocketClient,
    SocketClosedError,
)
from trackinizer.client.session_upload import RecordAppender
from trackinizer.wire.wire_session_relay import (
    Command,
    Hello,
    ProcessLine,
    ProcessSpec,
    ServeSession,
)
from trackinizer.wire.wire_sessions import SessionStart
from trackinizer.wire.wire_terminals import (
    CLOSE_ENDED,
    CLOSE_HOST_REVOKED,
    CLOSE_REFUSED,
    CLOSE_TAKEN_OVER,
)

from knowop.agent.errors import (
    LossyConversionError,
    NotResumableError,
)
from knowop.agent.placement import place_run, placement_env
from knowop.agent.resume import session_journal
from knowop.hosts import host_name
from knowop.process.codec import input_message
from knowop.process.host_follower import HostFollower, HostLister
from knowop.process.publisher import Publisher
from knowop.process.remote_commands import REMOTE_AGENT
from knowop.process.server_relay import AgentMessenger, ServerRelay
from knowop.service import knowop_version


if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterator,
        Awaitable,
        Callable,
        Sequence,
    )

    from sagent.agent.agent import Agent
    from sagent.agent.team import Team
    from sagent.tui.main import OpenAgent
    from sagent.tui.slash import Allowed
    from trackinizer.trax.profile import Profile
    from trackinizer.wire.wire_session_relay import Input, ViewerFrame

    from knowop.agent.resume import JournalChoice


__all__ = ["ServeClient", "delivering", "first_done", "main", "serve"]


_GIVE_UP_SEC: Final = 600.0
"""How long a headless process runs without its socket before it ends itself."""

_STOPPED_SEC: Final = 30.0
"""How long a stopped process halts its turn and uploads before it ends."""

_ATTACHED_SEC: Final = 120.0
"""How long ``--on`` waits for the process it started to attach."""

_EXITS: Final = {
    CLOSE_ENDED: 0,
    CLOSE_REFUSED: 1,
    CLOSE_HOST_REVOKED: 1,
    CLOSE_TAKEN_OVER: 3,
    GAVE_UP: 4,
}
"""The exit code for each way the session's socket closes for good."""

_SLURM_TIME: Final = re.compile(r"(?:([0-9]+)-)?([0-9]+)(?::([0-9]+))?(?::([0-9]+))?")
"""Slurm's ``--time``: ``M``, ``M:S`` or ``H:M:S``; ``D-H``, ``D-H:M`` or ``D-H:M:S``."""

logger = logging.getLogger(__name__)


def main(argv: Sequence[str], *, profile: Profile | None) -> int:
    """Run one session's agent process until it ends.

    Args:
      argv: What follows ``knowop run``: sagent's flags, and knowop's own.
      profile: The saved server and key it reports to; ``None`` when a daemon
        started it, which hands it both.

    Returns:
      code: 0 after a stop or a quit; 1 for a start error, a refused key or a
        session that is gone; 3 when another process holds the session; 4 when it
        ran headless too long without its socket.

    """
    parser = argparse.ArgumentParser(
        prog="knowop run",
        description=(__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    flags, args = _parse_args(parser, argv)
    picked = _journal_flags(args)
    if picked:
        parser.error(f"knowop keeps the session's journal; drop {', '.join(picked)}")
    if flags.serve is not None:
        return _served(flags.serve, args, time_limit=flags.time_limit)
    if profile is None:
        sys.stderr.write("Error: knowop run needs a profile; run knowop login.\n")
        return 1
    if flags.login_host and flags.on is None:
        sys.stderr.write(
            "Error: no host answers another's model calls yet; --login-host needs --on.\n",
        )
        return 1
    with Client(profile.url, author=profile.author, api_key=profile.api_key) as client:
        if flags.stop is not None:
            return _stopped(client, flags.stop)
        try:
            session_id = (
                flags.session
                or client.session_start(
                    SessionStart(cli="sagent"),
                ).id
            )
        except ClientError as error:
            sys.stderr.write(f"Error: {error}\n")
            return 1
        if flags.on is not None:
            return _started_on(client, flags, args, session_id=session_id)
        # Told to this process's environment, which each tool the agent runs inherits.
        os.environ.update(
            placement_env(
                place_run(
                    client,
                    session_id=session_id,
                    environ=os.environ,
                    cwd=Path.cwd(),
                ),
            ),
        )
        return _signalled(
            args,
            session_id=session_id,
            process_id=uuid4(),
            client=client,
            tui=sys.stdin.isatty() and not flags.headless,
            host=host_name(),
            time_limit=flags.time_limit,
            remote=False,
        )


class Flags(Protocol):
    """knowop's own flags; every other word is sagent's."""

    headless: bool
    session: UUID | None
    on: str | None
    cwd: str
    login_host: str
    stop: UUID | None
    time_limit: timedelta | None
    serve: str | None


class ServeClient(
    PartReader,
    RecordAppender,
    HostLister,
    SocketClient,
    AgentMessenger,
    Protocol,
):
    """The ``Client`` an agent process holds: the calls it makes, its server and key."""


async def serve(
    args: Sequence[str],
    *,
    session_id: UUID,
    process_id: UUID,
    client: ServeClient,
    tui: bool,
    host: str,
    time_limit: timedelta | None = None,
    remote: bool = False,
) -> int:
    """Run the session's agent here, live, until it ends; return the exit code.

    Args:
      args: sagent's flags, none of which picks a journal.
      session_id: The session to serve.
      process_id: This process's id, which its ``Hello`` names.
      client: Talks to the server, with the key this process serves the session with.
      tui: Whether sagent's screen runs, for a person at this terminal.
      host: The host this runs on, as the server enrolled it.
      time_limit: How long after it starts it ends itself, as a stop ends it; ``None``
        for no limit.
      remote: Whether the server placed it for a remote agent, which starts and links
        no sagent daemon of its own and runs only the slash lines
        :data:`~knowop.process.remote_commands.REMOTE_AGENT` lists; else
        it is on a person's own host, with its daemon and every slash line.

    Returns:
      code: The exit code, as the module's docstring lists them.

    """
    try:
        choice = await asyncio.to_thread(_journal, client, session_id)
    except (ClientError, NotResumableError, LossyConversionError) as error:
        sys.stderr.write(f"Error: {error}\n")
        return 1
    flags, given = parse_flags([*args, "--session", str(choice.path)])
    hello = Hello(process_id=process_id, version=knowop_version(), host=host)
    try:
        async with SessionSocket(
            client,
            session_id=session_id,
            hello=hello,
            give_up_sec=None if tui else _GIVE_UP_SEC,
        ) as socket:
            sys.stdout.write(f"{session_id}\n")
            sys.stdout.flush()
            async with opened(
                flags,
                given=given,
                interactive=tui,
                use_daemon=not remote,
            ) as o:
                ended = await _live(
                    o,
                    socket=socket,
                    client=client,
                    session_id=session_id,
                    choice=choice,
                    host=host,
                    tui=tui,
                    history=flags.history,
                    time_limit=time_limit,
                    allowed=REMOTE_AGENT if remote else None,
                )
    except SocketClosedError as closed:
        sys.stderr.write(f"Error: {closed.reason}\n")
        return _EXITS[closed.code]
    except JournalInUseError as error:
        sys.stderr.write(f"Error: {error}\n")
        return 3
    except (StartError, ClientError) as error:
        sys.stderr.write(f"Error: {error}\n")
        return 1
    if ended.reason:
        sys.stderr.write(f"{ended.reason}\n")
    return ended.code


@contextlib.asynccontextmanager
async def delivering(
    inputs: AsyncIterator[Input],
    *,
    agent: Agent,
    team: Team,
    relay: ServerRelay,
    commands: Awaitable[Commands],
    publisher: Publisher,
) -> AsyncGenerator[asyncio.Future[Ended]]:
    """Deliver each input to the agent, as its kind says, while the block runs.

    A halt halts the turn at once. An agent's message from another session goes to
    the team at once, booked as its sender booked it. A person's line is pushed as a
    message from its author to the agent it names, and a slash line runs in
    ``commands``, each in its turn after the slash line before it.

    Args:
      inputs: The inputs, as they come down the socket.
      agent: The root agent.
      team: The session's agents, one of whom a line may name.
      relay: Takes an agent's message from another session.
      commands: The root agent's commands, which run each slash line and say what
        it did; awaited at the first slash line.
      publisher: Says what does not read, and sends the settings a slash line changed.

    Yields:
      quit_ran: Resolves once a ``/quit`` ran.

    """
    quit_ran = asyncio.Future[Ended](loop=asyncio.get_running_loop())
    lines: asyncio.Queue[AgentToAgentMessage | Command] = asyncio.Queue()
    tasks = [
        asyncio.create_task(
            _delivered(
                inputs,
                agent=agent,
                relay=relay,
                publisher=publisher,
                lines=lines,
            ),
        ),
        asyncio.create_task(
            _ran(
                lines,
                agent=agent,
                team=team,
                commands=commands,
                publisher=publisher,
                quit_ran=quit_ran,
            ),
        ),
    ]
    try:
        yield quit_ran
    finally:
        # Waited for: a slash line still running has ended before its caller closes
        # what it uses.
        for task in tasks:
            _ = task.cancel()
        _ = await asyncio.gather(*tasks, return_exceptions=True)


async def first_done[T](*awaited: Awaitable[T]) -> T:
    """Return what the first of ``awaited`` to finish returns; cancel the rest.

    Args:
      *awaited: What to wait on; of two that finish at once, the first named wins.

    Returns:
      result: What it returned.

    """
    tasks = [asyncio.ensure_future(each) for each in awaited]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            _ = task.cancel()
        _ = await asyncio.gather(*tasks, return_exceptions=True)
    return next(task for task in tasks if task in done).result()


@dataclass(frozen=True, slots=True, kw_only=True)
class Ended:
    """Why an agent process ends."""

    code: int
    """Its exit code."""

    reason: str = ""
    """What it says as it exits; empty for nothing."""

    stopped: bool = False
    """Whether a person stopped or quit it, or its time ran out, so it uploads what it
    has before it ends."""


def _parse_args(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
) -> tuple[Flags, list[str]]:
    """Register knowop's flags; return them and sagent's, which follow untouched."""
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run with no screen, at a terminal too.",
    )
    parser.add_argument("--session", type=UUID, help="The session to continue here.")
    parser.add_argument(
        "--on",
        nargs="?",
        const="",
        metavar="NAME",
        help="Start on host NAME through its daemon; with no name, the server's.",
    )
    parser.add_argument(
        "--cwd",
        default="~",
        metavar="DIR",
        help="With --on, the folder there to start in (default: %(default)s).",
    )
    parser.add_argument(
        "--login-host",
        default="",
        metavar="NAME",
        help="Use host NAME's model login.",
    )
    parser.add_argument(
        "--stop",
        type=UUID,
        metavar="ID",
        help="Stop session ID's agent, wherever it runs, and exit.",
    )
    parser.add_argument(
        "--time",
        type=_time_limit,
        metavar="TIME",
        dest="time_limit",
        help=(
            "End TIME after it starts, as a stop ends it; Slurm's --time: 120, "
            "2:00:00 or 1-0."
        ),
    )
    parser.add_argument(
        "--serve",
        metavar="URL",
        help="Serve the session a daemon hands on stdin, through the server at URL.",
    )
    parsed, rest = parser.parse_known_args(argv)
    flags = cast("Flags", parsed)
    if flags.stop is not None and (others := _other_flags(argv)):
        parser.error(f"--stop takes no other flag: {', '.join(others)}")
    return flags, rest


# Without days, ``M``, ``M:S`` or ``H:M:S``; with them, ``D-H``, ``D-H:M`` or
# ``D-H:M:S``.
def _time_limit(text: str) -> timedelta:
    """Read a time limit as Slurm's ``--time`` reads one, refusing one of no time."""
    matched = _SLURM_TIME.fullmatch(text)
    if matched is not None:
        days, *clock = matched.groups()
        given = [int(field) for field in clock if field is not None]
        # Without days, one or two fields are minutes first: ``M`` or ``M:S``.
        if days is None and len(given) < 3:
            given = [0, *given]
        hours, minutes, seconds = (*given, 0, 0)[:3]
        limit = timedelta(
            days=int(days or 0),
            hours=hours,
            minutes=minutes,
            seconds=seconds,
        )
        if limit:
            return limit
    raise argparse.ArgumentTypeError(
        f"{text!r} is no time limit, such as 120, 2:00:00 or 1-0",
    )


def _slurm_time(limit: timedelta) -> str:
    """Write ``limit`` as Slurm's ``D-HH:MM:SS``."""
    minutes, seconds = divmod(limit.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{limit.days}-{hours:02}:{minutes:02}:{seconds:02}"


# Read off the words, not the parsed values: a flag given at its default, such as
# ``--cwd ~``, parses as if it were not given.
def _other_flags(argv: Sequence[str]) -> list[str]:
    """Return each flag ``argv`` gives besides ``--stop``, by the name given."""
    names = (word.partition("=")[0] for word in argv)
    return [name for name in names if name.startswith("--") and name != "--stop"]


def _journal_flags(args: Sequence[str]) -> list[str]:
    """Return the flags of ``args`` that pick a journal, which knowop refuses."""
    flags, _ = parse_flags(args)
    named = {
        "--ephemeral": flags.ephemeral,
        "--resume": flags.resume is not None,
        "--continue": flags.continue_,
        "--resume-all": flags.resume_all,
        "--continue-all": flags.continue_all,
        "--hosted-session": flags.hosted_session is not None,
    }
    return [name for name, given in named.items() if given]


def _served(url: str, args: Sequence[str], *, time_limit: timedelta | None) -> int:
    """Serve the one session a daemon hands on stdin, with the key it hands."""
    try:
        handed = ServeSession.model_validate_json(sys.stdin.readline())
    except ValueError as error:
        sys.stderr.write(f"Error: stdin holds no session to serve: {error}\n")
        return 1
    with Client(url, api_key=handed.token) as client:
        return _signalled(
            args,
            session_id=handed.session_id,
            process_id=handed.process_id,
            client=client,
            tui=False,
            host=handed.spec.host,
            time_limit=time_limit,
            remote=handed.remote,
        )


def _signalled(
    args: Sequence[str],
    *,
    session_id: UUID,
    process_id: UUID,
    client: ServeClient,
    tui: bool,
    host: str,
    time_limit: timedelta | None,
    remote: bool,
) -> int:
    """Run :func:`serve`, answering Ctrl+C and SIGTERM as sagent does."""
    try:
        return asyncio.run(
            _within_signals(
                serve(
                    args,
                    session_id=session_id,
                    process_id=process_id,
                    client=client,
                    tui=tui,
                    host=host,
                    time_limit=time_limit,
                    remote=remote,
                ),
            ),
        )
    # ``signals`` answers the first one by cancelling the run, which ``asyncio.run``
    # raises again as ``CancelledError``.
    except (KeyboardInterrupt, asyncio.CancelledError):
        return 130


async def _within_signals(served: Awaitable[int]) -> int:
    """Await ``served`` with the first SIGINT or SIGTERM cancelling it."""
    with signals():
        return await served


def _journal(client: ServeClient, session_id: UUID) -> JournalChoice:
    """Return the journal this process writes the session in."""
    return session_journal(
        session_id,
        client.read_session_parts(session_id),
        client=client,
        lossy=True,
    )


async def _live(
    o: OpenAgent,
    *,
    socket: SessionSocket,
    client: ServeClient,
    session_id: UUID,
    choice: JournalChoice,
    host: str,
    tui: bool,
    history: Path | None,
    time_limit: timedelta | None,
    allowed: Allowed | None,
) -> Ended:
    """Run the open agent live until a stop, its time limit, a quit, or an answer that ends it."""
    publisher = Publisher(
        o.agent,
        client=client,
        session_id=session_id,
        part=choice.part,
        start=choice.start,
        send=socket.send,
    )
    o.agent.runtime.observers.append(publisher)
    # Chat is the user an agent's send to ``user`` reaches. At a terminal the screen
    # takes it over as it builds (``Views``), and prints it there.
    o.team.to_user = publisher.to_user
    commands = asyncio.Future[Commands]()
    build = _telling(
        partial(Commands, allowed=allowed),
        root=o.agent,
        publisher=publisher,
        built=commands,
    )
    if not tui:
        # No screen builds them, or prints what they say.
        _ = build(
            o.agent,
            catalog=o.catalog,
            team=o.team,
            say=_unsaid,
            screen=o.screen,
            panes=Panes(),
            theme=o.theme,
        )
    # The agents here reach another session's, on any host, through the server.
    relay = ServerRelay(
        client,
        session_id=session_id,
        team=o.team,
        root=o.agent.runtime.tool_context.label,
    )
    o.team.relay = relay
    try:
        async with (
            HostFollower(
                HostFollower.Config(),
                client=client,
                url=client.base_url,
                api_key=client.api_key,
                remote=o.remote,
                keep=o.links,
                own_host=host,
            ),
            relay,
            delivering(
                socket.inputs(),
                agent=o.agent,
                team=o.team,
                relay=relay,
                commands=commands,
                publisher=publisher,
            ) as quit_ran,
        ):
            screen = (
                [
                    _quit_at_screen(
                        run(
                            o.agent,
                            catalog=o.catalog,
                            team=o.team,
                            history=history,
                            show_thinking=o.screen.show_thinking,
                            theme=o.theme,
                            commands=build,
                        ),
                    ),
                ]
                if tui
                else []
            )
            ended = await first_done(
                _closed(socket),
                quit_ran,
                _upload_ended(publisher),
                *screen,
                *([] if time_limit is None else [_timed_out(time_limit)]),
            )
            if ended.stopped:
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(_STOPPED_SEC):
                        await _halted(o.agent)
                        await publisher.caught_up()
            return ended
    finally:
        await publisher.aclose()
        # Closed twice at a terminal, by the screen first, which a close allows.
        if commands.done():
            await commands.result().aclose()


# One ``Commands`` for the root agent, whoever sends the line: a slash line from Chat
# prints at the terminal and changes the screen that draws, and one typed at the
# terminal is told to Chat. The screen builds it, with its own printer and panes,
# once: its view of the root agent lasts as long as the screen (``Views``).
def _telling[**P](
    build: Callable[P, Commands],
    *,
    root: Agent,
    publisher: Publisher,
    built: asyncio.Future[Commands],
) -> Callable[P, Commands]:
    """Return ``build`` whose commands for ``root`` also tell Chat what they say."""

    def telling(*args: P.args, **kwargs: P.kwargs) -> Commands:
        commands = build(*args, **kwargs)
        if commands.agent is root:
            say = commands.say

            def both(output: CommandOutput) -> None:
                say(output)
                publisher.say(output)

            commands.say = both
            publisher.commands = commands
            built.set_result(commands)
        return commands

    return telling


def _unsaid(output: CommandOutput) -> None:
    """Print nowhere what a command said: headless, Chat alone is told."""
    del output


async def _closed(socket: SessionSocket) -> Ended:
    """Return why the process ends, once the socket closes for good."""
    code = await socket.closed
    return Ended(code=_EXITS[code], reason=socket.reason, stopped=code == CLOSE_ENDED)


async def _timed_out(time_limit: timedelta) -> Ended:
    """Return why the process ends, once its time limit is up: as a stop ends it."""
    loop = asyncio.get_running_loop()
    up = loop.create_future()
    timer = loop.call_later(time_limit.total_seconds(), up.set_result, None)
    try:
        await up
    finally:
        timer.cancel()
    return Ended(
        code=0,
        reason=f"ended at its time limit, {_slurm_time(time_limit)}",
        stopped=True,
    )


async def _upload_ended(publisher: Publisher) -> Ended:
    """Return why the process ends, once an upload's answer ends it."""
    code = await publisher.ended
    return Ended(code=code, reason=publisher.reason)


async def _quit_at_screen(screen: Awaitable[None]) -> Ended:
    """Return once a person quits sagent's screen."""
    await screen
    return Ended(code=0, stopped=True)


# As sagent's own ``clear`` waits for its event: the halt takes effect at the runtime
# loop's next drain, which publishes it, and what it wrote is journaled by then. Tool
# calls keep running, so waiting for the agent to idle could wait on them.
async def _halted(agent: Agent) -> None:
    """Halt the turn; return once the runtime has taken the halt."""
    taken = asyncio.Event()

    def observe(event: types.RuntimeEvent) -> None:
        if isinstance(event, types.Halt):
            taken.set()

    agent.runtime.observers.append(observe)
    try:
        agent.runtime.halt()
        _ = await taken.wait()
    finally:
        agent.runtime.observers.remove(observe)


# A halt does not wait its turn: a slash line such as ``/login`` runs until a person
# is done with it, and Stop in Chat halts the turn meanwhile, as Ctrl+C at the screen
# does past the line being run there.
async def _delivered(
    inputs: AsyncIterator[Input],
    *,
    agent: Agent,
    relay: ServerRelay,
    publisher: Publisher,
    lines: asyncio.Queue[AgentToAgentMessage | Command],
) -> None:
    """Halt at each halt and hand each agent's message on as it comes; queue the rest."""
    async for sent in inputs:
        try:
            message = input_message(sent)
        except ValueError as error:
            publisher.say(CommandOutput(text=error_text(error), tone="error"))
            continue
        if isinstance(message, types.Halt):
            agent.runtime.halt()
        elif isinstance(message, AgentToAgentMessage) and sent.booked is not None:
            relay.deliver(message, booked=sent.booked)
        else:
            lines.put_nowait(message)


async def _ran(
    lines: asyncio.Queue[AgentToAgentMessage | Command],
    *,
    agent: Agent,
    team: Team,
    commands: Awaitable[Commands],
    publisher: Publisher,
    quit_ran: asyncio.Future[Ended],
) -> None:
    """Run each line and slash line in turn; resolve ``quit_ran`` once a ``/quit`` runs."""
    while True:
        message = await lines.get()
        if not isinstance(message, Command):
            named = agent if message.recipient is None else team.get(message.recipient)
            if named is None:
                publisher.say(
                    CommandOutput(
                        text=f"No agent {message.recipient!r} here took this line.",
                        tone="error",
                    ),
                )
            else:
                named.runtime.push(message)
            continue
        # Shielded: cancelling this must not cancel what the screen has yet to build.
        ran = await asyncio.shield(commands)
        try:
            done = await ran(message.line)
        except Exception as error:
            logger.warning("command %r failed", message.line, exc_info=True)
            ran.say(CommandOutput(text=error_text(error), tone="error"))
            continue
        publisher.check_settings()
        if done:
            # As a stop: a remote agent's folder goes as it ends, its journal too.
            quit_ran.set_result(Ended(code=0, stopped=True))
            return


def _stopped(client: Client, session_id: UUID) -> int:
    """Stop ``session_id``'s agent, wherever it runs; say where, or why not."""
    try:
        process = client.stop_process(session_id)
    except ClientError as error:
        sys.stderr.write(f"Error: {error}\n")
        return 1
    if process is None:
        sys.stderr.write(f"Error: session {session_id} has had no agent process\n")
        return 1
    sys.stdout.write(f"stopped {session_id} on {process.host}\n")
    return 0


def _started_on(
    client: Client,
    flags: Flags,
    args: Sequence[str],
    *,
    session_id: UUID,
) -> int:
    """Start the session on the host ``--on`` names; return once it attaches or not."""
    frames: queue.Queue[ViewerFrame | ClientError] = queue.Queue()
    threading.Thread(
        target=_read_lines,
        args=(client, session_id, frames),
        name="lines",
        daemon=True,
    ).start()
    deadline = time.monotonic() + _ATTACHED_SEC
    try:
        # The stream opens before the start, so the line that says it attached is
        # not missed.
        _ = _next_frame(frames, deadline=deadline)
        # The time limit is the process's own flag, so it goes on the line it is
        # started with, and runs from its start there.
        limit = flags.time_limit
        timed = () if limit is None else ("--time", _slurm_time(limit))
        started = client.start_process(
            session_id,
            ProcessSpec(
                host=flags.on or "",
                login_host=flags.login_host,
                cwd=flags.cwd,
                args=(*timed, *args),
            ),
        )
        process = started
        while process.state != "detached" and not (
            process.state == "attached" and process.process_id == started.process_id
        ):
            frame = _next_frame(frames, deadline=deadline)
            if (
                isinstance(frame, ProcessLine)
                and frame.process.process_id == started.process_id
            ):
                process = frame.process
    except (ClientError, TimeoutError) as error:
        sys.stderr.write(f"Error: {error}\n")
        return 1
    if process.state == "detached":
        sys.stderr.write(f"Error: the process did not start: {process.reason}\n")
        return 1
    sys.stdout.write(f"{session_id}\n")
    return 0


def _read_lines(
    client: Client,
    session_id: UUID,
    frames: queue.Queue[ViewerFrame | ClientError],
) -> None:
    """Put each frame of the session's lines on ``frames``, then why they ended."""
    try:
        for frame in client.open_lines(session_id):
            frames.put(frame)
    except ClientError as error:
        frames.put(error)


def _next_frame(
    frames: queue.Queue[ViewerFrame | ClientError],
    *,
    deadline: float,
) -> ViewerFrame:
    """Return the next frame, before ``deadline`` on the monotonic clock."""
    try:
        frame = frames.get(timeout=max(0.0, deadline - time.monotonic()))
    except queue.Empty:
        raise TimeoutError(
            f"the process did not attach within {_ATTACHED_SEC:g} s",
        ) from None
    if isinstance(frame, ClientError):
        raise frame
    return frame
