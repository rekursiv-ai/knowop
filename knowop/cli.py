"""Log in to KnowOp, connect this computer, and run recorded agents and GPU jobs.

``knowop login`` signs in with a code a person approves in a browser and saves the
key as a trax profile. knowop records that profile as its own, and every verb uses
it, never trax's current profile, so knowop acts only on the server it logged in to.
``knowop --profile NAME VERB`` runs a verb with another saved profile instead.

At a terminal, ``knowop connect`` serves this computer from a user service (launchd's
or systemd's) that starts again at login; ``knowop logs -f`` follows it and ``knowop
disconnect`` stops it.

Examples:
  knowop connect          log in if needed, then connect this computer
  knowop login acme       log in to https://acme.knowop.ai, without connecting
  knowop run              sagent in this folder, live in Chat
  knowop run --headless -- --effort high
                          no screen; sagent's settings and actions go after --
  knowop run --on lab     sagent on your host lab, live in Chat
  knowop run -- login --provider oaisub
                          sagent's sign-in to a model provider, on this computer
  knowop job submit -- python train.py
                          run a job on a GPU node from this repository
  knowop job status       list your jobs
  knowop host             list your hosts, where campaigns may run, and KnowHosts
  knowop host start       rent a KnowHost that joins as one of your hosts
  knowop canvas           read the canvas a run sits on; --help shows how to change it
  knowop watch-pr add URL --session Agent#412
                          tell session 412's agent when the pull request's checks fail
  knowop session          the session whose agent runs this command
  knowop upload ~/.claude send finished Claude Code, Codex and sagent sessions
  knowop status           profile, server, account, hosts, their runs, services
  knowop logs -f          follow this computer's connect
  knowop disconnect       stop this computer's connect and its runs, everywhere
  knowop --profile staging status
                          any verb, with another saved profile

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, NoReturn, Protocol, TypedDict, cast

import argparse
import functools
import getpass
import io
import os
import re
import shlex
import signal
import sys
import time
import uuid

from trackinizer.client.client import Client, server_url
from trackinizer.client.errors import ClientError
from trackinizer.lib.userdirs import config_dir, state_dir
from trackinizer.trax.client_cache import close_clients
from trackinizer.trax.connection import connect, connect_flags
from trackinizer.trax.context import OUT_STREAM
from trackinizer.trax.grammar import parse_ref
from trackinizer.trax.profile import Profile, del_profile, read_profile
from trackinizer.trax.render import echo
from treekle import from_plain, json_dumps

from knowop import menubar, service
from knowop.connect import serve
from knowop.hosts import (
    HOST_WORD,
    Hosts,
    check_name,
    host_name,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import FrameType

    from trackinizer.runs import redact
    from trackinizer.trax import login
    from trackinizer.wire.wire_hosts import HostDetail

    from knowop import canvas, upload
    from knowop.connect import job_limits
    from knowop.jobs import cli as jobs_cli
    from knowop.process import (
        command as process_command,
        main as process_main,
    )
    from knowop.watch_pr import cli as watch_pr_cli
else:
    from wrapt import lazy_import

    # Only ``login``, ``connect`` and a first ``run`` reach the login, and only ``run``
    # and ``job`` the runners. Their imports cost ~100 ms or more that ``status`` would
    # pay at every start. Only ``canvas`` reaches trax's verbs, ~20 ms more.
    canvas = lazy_import("knowop.canvas")
    jobs_cli = lazy_import("knowop.jobs.cli")
    watch_pr_cli = lazy_import("knowop.watch_pr.cli")
    job_limits = lazy_import("knowop.connect.job_limits")
    login = lazy_import("trackinizer.trax.login")
    # The agent process ``run`` hands to, and its command line; each brings sagent.
    process_main = lazy_import("knowop.process.main")
    process_command = lazy_import("knowop.process.command")
    # ``upload`` alone reads transcripts; it brings every CLI's session reader.
    upload = lazy_import("knowop.upload")
    redact = lazy_import("trackinizer.runs.redact")


_DEFAULT_SERVER: Final = "https://alpha.knowop.ai"
"""The server ``login``, ``connect`` and a first ``run`` log in to when none is named."""

_CANVAS_WORD: Final = "canvas"
"""The verb trax's canvas command runs under, on knowop's login."""

_WATCH_PR_WORD: Final = "watch-pr"
"""The verb ``watch_pr/cli.py`` runs under, on knowop's login."""


def main(
    argv: Sequence[str] | None = None,
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Run ``knowop``; return the process exit code.

    Args:
      argv: The arguments after ``knowop``; ``None`` reads ``sys.argv``.
      sleep: Waits between a login's polls; a test replaces it.
      clock: Monotonic seconds, for a login code's lifetime; a test replaces it.

    Returns:
      exit_code: 0 on success, 1 when a verb fails, 130 on Ctrl-C. ``run`` and
        ``job`` exit as the runner they hand off to does.

    """
    # A login prints its code, then waits for a person to approve it. Block-buffered,
    # as stdout is whenever an agent or a pipe reads it, the code would stay unprinted
    # until the login ended, unapproved.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        named, args = _peeled(args)
        if not args:
            _next_step(named)
            return 0
        # Everything after ``run`` or ``job`` belongs to the runner, ``--help`` and
        # every other flag included; argparse would read them as knowop's own. Every
        # ``run``, ``host`` and ``canvas`` form is that verb's, its help pages too.
        if args[:1] in (["run"], [HOST_WORD], [_CANVAS_WORD], [_WATCH_PR_WORD]) or (
            args[:1] == ["job"] and args[1:] not in (["-h"], ["--help"])
        ):
            return _hand_off(args[0], args[1:], named=named, sleep=sleep, clock=clock)
        parser = argparse.ArgumentParser(
            prog="knowop",
            description=(__doc__ or "").strip(),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        _add_arguments(parser)
        flags = cast(Flags, parser.parse_args(args))
        _run(flags, named=named, sleep=sleep, clock=clock)
    except ClientError as error:
        echo(f"knowop: {error}", err=True)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        # ``connect`` signs in through trax's shared clients; close their sockets.
        close_clients()
    return 0


class Flags(Protocol):
    """Parsed command-line flags; each verb sets its own."""

    verb: str
    target: str
    name: str
    sandbox: bool
    no_exec: bool
    jobs: bool
    job_results_max: str
    foreground: bool
    follow: bool
    log_file: str
    json: bool
    all: bool
    lines: int | None


class _UploadFlags(Protocol):
    """The flags of ``upload``, whose ``--jobs`` is a count, unlike ``connect``'s."""

    paths: list[Path]
    public: bool
    server: str | None
    read_as: str | None
    no_subagents: bool
    jobs: int
    dry_run: bool


class _DownloadFlags(Protocol):
    """The flags of ``session``, and of ``session download`` when it is named."""

    session_verb: str | None
    ref: str | None
    output: Path | None


class _TopFlags(Protocol):
    """The flags read before the verb: ``connect_flags``'s, and what follows them."""

    profile: str | None
    host: str | None
    port: int | None
    rest: list[str]


# Read before the verb, as ``trax --profile`` is: a flag after the verb is the verb's
# own, and ``run`` and ``job`` hand theirs to the runner, which reads ``--profile`` too.
def _peeled(args: list[str]) -> tuple[str, list[str]]:
    """Return the profile named before the verb, or empty, and the rest of ``args``."""
    parser = argparse.ArgumentParser(prog="knowop", add_help=False)
    connect_flags(parser)
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    parsed, unknown = parser.parse_known_args(args)
    top = cast(_TopFlags, parsed)
    # knowop's own verbs reach the server a profile names, never one a flag rewrites.
    if top.host is not None or top.port is not None:
        raise ClientError(
            "knowop reads --profile alone before its verb; --host and --port are "
            "trax's",
        )
    return top.profile or "", [*unknown, *top.rest]


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the verbs and their flags on ``parser``."""
    parser.add_argument(
        "--version",
        action="version",
        version=f"knowop {service.knowop_version()}",
    )
    verbs = parser.add_subparsers(dest="verb", required=True, metavar="VERB")
    verb = verbs.add_parser(
        "login",
        help="log in with a code approved in a browser",
        description=(
            "Log in with a code approved in a browser; knowop uses this login "  # pragma: no mutate[diagnostic-text]
            "from then on."  # pragma: no mutate[diagnostic-text]
        ),
    )
    verb.add_argument(
        "target",
        nargs="?",
        default="",
        metavar="ORG|URL",
        help=f"ORG means https://ORG.knowop.ai; default {_DEFAULT_SERVER}",
    )
    logout = verbs.add_parser(
        "logout",
        help="log this computer out, revoke knowop's key and delete its profile",
        description=(
            "Log this computer's host out on the server, revoke knowop's key, "  # pragma: no mutate[diagnostic-text]
            "delete its profile and remove this computer's connect services. NAME "  # pragma: no mutate[diagnostic-text]
            "logs that host out alone, as the Hosts page does: its connect is "  # pragma: no mutate[diagnostic-text]
            "revoked, its runs stop, and the keys its logins made are revoked."  # pragma: no mutate[diagnostic-text]
        ),
    ).add_mutually_exclusive_group()
    logout.add_argument(
        "name",
        nargs="?",
        default="",
        help="a host of yours to log out, keeping this computer's login",
    )
    logout.add_argument(
        "--all",
        action="store_true",
        help="log every host of yours out first",
    )
    verbs.add_parser(
        "status",
        help="show knowop's profile, server, account, role, hosts and their runs",
        description=(
            "Show knowop's profile, server, account and role; each of your hosts, "  # pragma: no mutate[diagnostic-text]
            "online or not, with its knowop version and process while online, and "  # pragma: no mutate[diagnostic-text]
            "its live sessions; and this computer's connect services, with their flags."  # pragma: no mutate[diagnostic-text]
        ),
    ).add_argument(
        "--json",
        action="store_true",
        help="the same as one JSON object, for scripts",
    )
    verbs.add_parser(
        "menubar",
        help="show this computer's connects in the macOS menu bar",
        description=(
            "Show the knowop mark in the macOS menu bar, dimmed while a connect "  # pragma: no mutate[diagnostic-text]
            "service here is not online, with a menu of each server knowop is logged "  # pragma: no mutate[diagnostic-text]
            "in to and its hosts. knowop connect starts it at login beside its "  # pragma: no mutate[diagnostic-text]
            "connect services; Quit stops them and it until then."  # pragma: no mutate[diagnostic-text]
        ),
    )
    logs = verbs.add_parser(
        "logs",
        help="print or follow the log of a host's connect on this computer",
        description=(
            "Print the log the connect service of a host writes on this computer, "  # pragma: no mutate[diagnostic-text]
            "or follow it. A connect in the foreground prints to its terminal "  # pragma: no mutate[diagnostic-text]
            "instead, and a host connected elsewhere keeps its log there."  # pragma: no mutate[diagnostic-text]
        ),
    )
    logs.add_argument(
        "name",
        nargs="?",
        default="",
        help="the host; default this computer's short host name",
    )
    logs.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="keep printing what it writes, until Ctrl-C",
    )
    logs.add_argument(
        "-n",
        "--lines",
        type=int,
        metavar="N",
        help="print only its last N lines",
    )
    verbs.add_parser(
        "disconnect",
        help="disconnect a host of yours on the server, stopping its runs",
        description=(
            "Disconnect the host on the server: its connect is revoked, and every run "  # pragma: no mutate[diagnostic-text]
            "of yours that names it is stopped, here or anywhere. Its connect service "  # pragma: no mutate[diagnostic-text]
            "on this computer is removed, and it waits until the server shows the "  # pragma: no mutate[diagnostic-text]
            "host offline. knowop connect joins it again, with no login; knowop "  # pragma: no mutate[diagnostic-text]
            "logout also revokes this computer's key."  # pragma: no mutate[diagnostic-text]
        ),
    ).add_argument(
        "name",
        nargs="?",
        default="",
        help="the host; default this computer's short host name",
    )
    _add_connect_arguments(
        verbs.add_parser(
            "connect",
            help="log in if needed, then serve this computer's terminals and tools",
            description=(
                "Connect this computer and serve its terminals, and its tools to the "
                "server's assistant. At a terminal, join here, then serve it "
                "from a user service (launchd on macOS, systemd on Linux) that keeps "
                "running in the background and starts again at login; knowop logs -f "
                "follows it and knowop disconnect stops it. Elsewhere, and with "
                "--foreground, serve here until Ctrl-C. When knowop is not logged in "
                "to the server, log in first: the same approval enrolls and confirms "
                "this computer. An enrollment token piped to stdin joins instead, "
                "with no login, in the foreground."
            ),
        ),
    )
    # Only listed here: ``main`` hands every ``run`` form to the agent process.
    verbs.add_parser(
        "run",
        help="run sagent's agent here, or with --on on a host, live in Chat; "
        "sagent's arguments go after --",
    )
    # Only ``job -h`` reaches argparse: ``main`` hands every other form to the runner.
    verbs.add_parser(
        "job",
        help="run, list and cancel jobs on a GPU node, and keep their secrets",
        description=(
            "Run a job on a GPU node from this repository as knowop's account, read "  # pragma: no mutate[diagnostic-text]
            "its log and results, list and cancel jobs, and keep their secrets; its "  # pragma: no mutate[diagnostic-text]
            "arguments pass to the job runner unchanged."  # pragma: no mutate[diagnostic-text]
        ),
        usage="knowop job submit|logs|results|status|cancel|secrets|secret ...",
    )
    # Only listed here: ``main`` hands every ``host`` form to its command.
    verbs.add_parser(
        HOST_WORD,
        help="list, show, set and delete hosts; rent, pause, wake and end KnowHosts",
    )
    # Only listed here too: ``main`` hands every ``canvas`` form to trax's verb.
    verbs.add_parser(
        _CANVAS_WORD,
        help="read and change the canvas an agent run sits on, as trax canvas does",
    )
    # And every ``watch-pr`` form to its commands.
    verbs.add_parser(
        _WATCH_PR_WORD,
        help="watch GitHub pull requests for agent sessions, and tell their agents",
    )
    session = verbs.add_parser(
        "session",
        help="show the session whose agent runs this command",
        description=(
            "Show the session whose agent runs this command, as $TRAX_SESSION names "
            "it: its number and its UUID. knowop run sets it for its agent's tools; "
            "an agent whose tools run on another host asks its KnowOp tool's "
            "session.self instead."
        ),
    )
    download = session.add_subparsers(dest="session_verb", metavar="ACTION").add_parser(
        "download",
        help="write a session's every record as one JSON-lines file",
        description=(
            "Write a session's every record as one JSON-lines file, as Chat's ⋯ "
            "Download session does: a header naming its parts, then one record per "
            "line. To stdout, unless --output names a file."
        ),
    )
    download.add_argument(
        "ref",
        nargs="?",
        help="AgentSession#N or a UUID (default: the session $TRAX_SESSION names)",
    )
    download.add_argument("-o", "--output", type=Path, help="the file to write")
    _add_upload_arguments(
        verbs.add_parser(
            "upload",
            help="send finished Claude Code, Codex and sagent sessions to the server",
            description=(
                "Send finished Claude Code, Codex and sagent sessions to the server, "  # pragma: no mutate[diagnostic-text]
                "each as an AgentSession under the CLI's own id for it, started and "  # pragma: no mutate[diagnostic-text]
                "ended when it was. A folder is searched for session files; a "  # pragma: no mutate[diagnostic-text]
                "session's sub-agents go too, each linked to the session that spawned "  # pragma: no mutate[diagnostic-text]
                "it. Uploading a session again sends only what the server lacks, and "  # pragma: no mutate[diagnostic-text]
                "leaves it to the readers it has."  # pragma: no mutate[diagnostic-text]
            ),
        ),
    )


def _add_connect_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the flags of ``connect`` on ``parser``."""
    parser.add_argument(
        "target",
        nargs="?",
        default="",
        metavar="ORG|URL",
        help=(
            "as for login; default the server knowop is logged in to, else "
            f"{_DEFAULT_SERVER}"
        ),
    )
    parser.add_argument(
        "--name",
        default="",
        help="the host's name; default this computer's short host name",
    )
    parser.add_argument(
        "--no-exec",
        action="store_true",
        help="serve terminals only: run no tool and start no agent here for the server",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="record that this computer is disposable",
    )
    parser.add_argument(
        "--jobs",
        action="store_true",
        help=(
            "also run the jobs your account sends here with knowop job submit "
            "--host NAME, each in a fresh worktree of its pushed commit under a "
            "memory cap; needs systemd's user manager"
        ),
    )
    parser.add_argument(
        "--job-results-max",
        default="256M",
        metavar="SIZE",
        help=(
            "with --jobs: the most result bytes one job sends back (default "
            "%(default)s); larger results stay on this computer"
        ),
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="serve in this terminal until Ctrl-C, installing no service",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="once the service runs, follow its log until Ctrl-C",
    )
    # Only a service's definition names it: the service's own run keeps this log under
    # 5 MiB, and exits 0 on a refusal no restart can change, so it is not restarted.
    parser.add_argument("--log-file", default="", help=argparse.SUPPRESS)


def _add_upload_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the flags of ``upload`` on ``parser``."""
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        metavar="PATH",
        help="a session file, or a folder searched for them, such as ~/.claude",
    )
    audience = parser.add_mutually_exclusive_group()
    # Said to be told apart from ``--public`` in a script; nothing reads it.
    audience.add_argument(
        "--private",
        action="store_true",
        help="open a new session to you and those who oversee (default)",
    )
    audience.add_argument(
        "--public",
        action="store_true",
        help="open a new session to the organization",
    )
    parser.add_argument(
        "--server",
        metavar="ORG|URL",
        help="send to this server, as knowop login names one, with its saved login",
    )
    parser.add_argument(
        "--format",
        dest="read_as",
        choices=("claude", "codex", "sagent"),
        help="read every file as this format instead of from its content",
    )
    parser.add_argument(
        "--no-subagents",
        action="store_true",
        help="leave the sessions of sub-agents",
    )
    parser.add_argument(
        "--jobs",
        type=_count,
        default=min(8, os.cpu_count() or 1),
        metavar="N",
        help="sessions sent side by side (default: this computer's CPUs, at most 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list each session that would go up, and contact no server",
    )


def _count(text: str) -> int:
    """Return the positive number ``text`` spells, or refuse it as argparse does."""
    if not text.isdigit() or int(text) < 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number of 1 or more")
    return int(text)


def _run(
    flags: Flags,
    *,
    named: str,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> None:
    """Run the verb ``flags`` names; ``named`` is the profile ``--profile`` named."""
    match flags.verb:
        case "login":
            _log_in(flags.target, named=named, sleep=sleep, clock=clock)
        case "logout":
            _log_out(flags.name, every=flags.all, named=named)
        case "status":
            _status(as_json=flags.json, named=named)
        case "logs":
            _logs(
                flags.name,
                named=named,
                follow=flags.follow,
                lines=flags.lines,
                sleep=sleep,
            )
        case "disconnect":
            _disconnect(flags.name, named=named, sleep=sleep)
        case "menubar":
            if sys.platform != "darwin":
                raise ClientError("knowop menubar runs on macOS alone")
            services = service.here()
            assert isinstance(services, service.Services)
            menubar.run(stop=services.stop)
        case "upload":
            _upload(cast(_UploadFlags, flags), named=named)
        case "session" if cast(_DownloadFlags, flags).session_verb == "download":
            _download(cast(_DownloadFlags, flags), named=named)
        case "session":
            _session(named=named)
        case _:
            # ``connect``: ``main`` hands ``run`` and ``job`` off before parsing, and
            # argparse refuses any other word.
            _connect(flags, named=named, sleep=sleep, clock=clock)


def _session(*, named: str) -> None:
    """Print the session ``$TRAX_SESSION`` names: its number and its UUID."""
    session = _running_session()
    _, profile = _profile(named)
    with Client(profile.url, api_key=profile.api_key) as http:
        number = from_plain(http.get(f"/api/inquiries/{session}"), _Numbered)["seq"]
    echo(f"AgentSession#{number} {session}")


def _download(flags: _DownloadFlags, *, named: str) -> None:
    """Write a session's every record, as one file, to ``--output`` or stdout."""
    _, profile = _profile(named)
    with _client(profile) as http:
        if flags.ref is None:
            session = _running_session()
        else:
            try:
                ref = parse_ref(flags.ref)
            except ValueError as error:
                raise ClientError(
                    f"download takes AgentSession#N or a UUID: {error}",
                ) from error
            _, session = http.resolve_id(ref)
        lines = http.download_session(session)
        if flags.output is None:
            for line in lines:
                echo(line)
            return
        with flags.output.open("wb") as out:
            for line in lines:
                out.write(f"{line}\n".encode())
    echo(f"wrote {flags.output}")


def _running_session() -> uuid.UUID:
    """Return the session ``$TRAX_SESSION`` names, the one whose agent runs this."""
    text = os.environ.get("TRAX_SESSION") or ""
    if not text:
        # pragma: no mutate[diagnostic-text] start
        raise ClientError(
            "no agent's session runs this command ($TRAX_SESSION is unset); an agent "
            "whose tools run on another host asks its KnowOp tool's session.self",
        )
        # pragma: no mutate end
    return uuid.UUID(text)


class _Numbered(TypedDict):
    seq: int


def _client(profile: Profile) -> Client:
    """Return a client of the server ``profile`` logged in to, with its key."""
    return Client(profile.url, api_key=profile.api_key)


# A session that fails is said on its line, and the batch goes on; then the command
# fails. ``upload`` starts no login: a server knowop holds none for is refused.
def _upload(flags: _UploadFlags, *, named: str) -> None:
    """Send the sessions ``flags.paths`` name, one line each and a total; or list them."""
    if missing := [str(path) for path in flags.paths if not path.expanduser().exists()]:
        raise ClientError(f"no such file or folder: {', '.join(missing)}")
    sessions = upload.plan(
        [path.expanduser() for path in flags.paths],
        read_as=flags.read_as or "",
        subagents=not flags.no_subagents,
    )
    if not sessions:
        raise ClientError("no Claude Code, Codex or sagent session found there")
    if flags.dry_run:
        for session in sessions:
            echo(f"{session.format} {session.id}  {session.path}{_kin(session)}")
        echo(f"would upload {_sessions(len(sessions))}; --dry-run sent none")
        return
    profile = _login_to(flags.server, named=named)
    redactor = redact.redactor_from_environ(os.environ)
    echo(f"uploading {_sessions(len(sessions))} to {profile.url}")
    results = upload.upload(
        sessions,
        connect=functools.partial(Client, profile.url, api_key=profile.api_key),
        private=not flags.public,
        jobs=flags.jobs,
        redactor=redactor,
    )
    for result in results:
        echo(f"{result.session.format} {result.session.id}: {_outcome(result)}")
    failed = sum(1 for result in results if result.error)
    echo(
        f"uploaded {_sessions(len(results) - failed)}: "
        f"{sum(result.sent for result in results)} records sent, "
        f"{sum(result.held for result in results)} already there",
    )
    if failed:
        raise ClientError(f"{_sessions(failed)} failed")


def _login_to(server: str | None, *, named: str) -> Profile:
    """Return knowop's login, or its saved login for ``server``; refuse when it has none."""
    if server is None:
        return _profile(named)[1]
    url = _server(server).removesuffix("/")
    found = next((each for _, each in _logins(named) if each.url == url), None)
    if found is None:
        raise ClientError(f"not logged in to {url}; run knowop login {server}")
    return found


def _kin(session: upload.Session) -> str:
    """Return what ``session``'s line says of its sub-agents and its spawner."""
    said = f", {session.subagents} sub-agent{'' if session.subagents == 1 else 's'}"
    if session.linked:
        return f"{said}, spawned by {session.spawned_by}"
    if session.spawned_by:
        return f"{said}, spawned by {session.spawned_by}, whose file is not here"
    return said


def _outcome(result: upload.Uploaded) -> str:
    """Return what became of one session, as its line says it."""
    if result.error:
        return f"failed: {result.error}"
    refused = f", {result.refused} refused" if result.refused else ""
    return (
        f"AgentSession {result.agent_session} ({result.audience}), sent "
        f"{result.sent}, held {result.held}{refused}{_kin(result.session)}"
    )


def _sessions(count: int) -> str:
    """Return ``count`` sessions, in words."""
    return f"{count} session{'' if count == 1 else 's'}"


def _log_in(
    target: str = "",
    *,
    named: str,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> None:
    """Log in to the server ``target`` names, and make its profile knowop's."""
    result = login.login(
        _server(target),
        profile=named,
        client="knowop",
        sleep=sleep,
        clock=clock,
    )
    _remember(result.profile)
    echo(f"logged in: profile {result.profile} ({result.role})")


def _server(target: str) -> str:
    """Return the URL a login target names: none, an org, or a URL as given."""
    if not target:
        return _DEFAULT_SERVER
    if "://" in target:
        return target
    if re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", target) is None:
        raise ClientError(
            f"{target!r} is neither an org name, such as acme, nor a URL",
        )
    return f"https://{target}.knowop.ai"


# Logging this computer out also revokes knowop's key, deletes its profile and removes
# this computer's connect services of that server, none of which could start again
# without the login.
def _log_out(name: str, *, every: bool, named: str) -> None:
    """Log ``name`` out; else this computer, and with ``every`` all of yours, first."""
    profile_name, profile = _profile(named)
    services = service.here()
    with Client(profile.url, api_key=profile.api_key) as http:
        if name:
            check_name(name)
            _log_out_host(http, name)
            _remove_service(services, name, profile.url)
            return
        if (caller := _caller(http)) is not None:
            key = from_plain(caller.get("api_key_id"), uuid.UUID)
            email = from_plain(caller.get("email"), str)
            hosts = [host_name()]
            if every:
                # The host a login made this key for goes last: its logout revokes
                # the key the others' sign with.
                made_for = caller.get("host")
                hosts = sorted(
                    (m.name for m in http.list_hosts() if m.owner == email),
                    key=lambda host: host == made_for,
                )
            for host in hosts:
                # A host of another account by this computer's name, or none, is
                # not this one's: it reads as no host, and the key alone is revoked.
                try:
                    _log_out_host(http, host)
                except ClientError as error:
                    if error.status_code != 404:
                        raise
            # Logging a host out revokes the keys its logins made, this one among them.
            if _caller(http) is not None:
                http.post(f"/api/me/tokens/{key}/revoke")
    for each in [] if services is None else services.every():
        if each.server == profile.url:
            _remove_service(services, each.name, each.server)
    # ``del_profile`` refuses the profile trax has pinned, and a pin naming a deleted
    # profile would fail every later trax command. Unpinned, trax falls back to its
    # ``default`` profile.
    pin = config_dir() / "rekursiv-ai" / "trax" / "current"
    if pin.exists() and pin.read_text().strip() == profile_name:
        pin.unlink()
    del_profile(profile_name)
    _keep_logins([each for each, _ in _kept_logins()])
    revoked = "revoked its key" if caller is not None else "its key was revoked already"
    echo(f"logged out: {revoked}; deleted profile {profile_name}")
    if (left := next(iter(_kept_logins()), None)) is not None:
        echo(f"knowop now uses: profile {left[0]} ({left[1].url})")


def _log_out_host(http: Client, host: str) -> None:
    """Log ``host`` out on the server, and say what stopped."""
    out = http.logout_host(host)
    runs = ", ".join(run.run or str(run.session_id) for run in out.runs) or "none"
    echo(
        f"logged out: host {host}; runs stopped: {runs}; keys revoked: {out.keys}",
    )


def _remove_service(
    services: service.Services | None,
    name: str,
    server: str,
) -> None:
    """Remove the connect service of ``name`` on ``server`` from here, if it is here."""
    if services is not None and services.remove(name, server):
        echo(f"removed: the connect service of host {name} on this computer")


@dataclass(frozen=True, slots=True, kw_only=True)
class _Run:
    """A live session, as ``knowop status`` shows it."""

    run: str | None
    """Its name, as the server lists it; ``None`` for a run it names none."""

    cli: str
    cwd: str | None
    session_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class _Host:
    """One of the account's hosts, as ``knowop status`` shows it."""

    name: str
    status: str
    this_computer: bool
    knowop_version: str | None
    """While online, the knowop serving it, as its facts say; else ``None``."""

    pid: int | None
    started_at: str | None
    runs: list[_Run]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Login:
    """One of knowop's logins, as ``knowop status`` shows it."""

    profile: str
    server: str
    account: str | None
    """``None`` when the server refused the key."""

    role: str | None
    hosts: list[_Host] | None
    """``None`` when the key was refused, or the role may not list hosts."""


@dataclass(frozen=True, slots=True, kw_only=True)
class _Status:
    """What ``knowop status`` shows, and ``--json`` writes."""

    logins: list[_Login]
    """Each server knowop is logged in to, the one its verbs use first."""

    services: list[service.Connect]
    menubar: str | None
    """How the menu bar agent here is; ``None`` where none is installed."""


# A refused login is shown with the others, and then fails the command.
def _status(*, as_json: bool, named: str) -> None:
    """Print each of knowop's logins with its account's hosts, and services here."""
    _profile(named)
    logins = [_login(name, profile) for name, profile in _logins(named)]
    services = service.here()
    report = _Status(
        logins=logins,
        services=[] if services is None else services.every(),
        menubar=None if services is None else services.menu_bar(),
    )
    if as_json:
        echo(json_dumps(report, target=_Status, indent=2))
    else:
        _say_status(report, manager=None if services is None else services.manager)
    if refused := [each.server for each in logins if each.account is None]:
        raise ClientError(
            "not logged in: "
            + "; ".join(
                f"{url} refused the key, run knowop login {url}" for url in refused
            ),
        )


def _login(name: str, profile: Profile) -> _Login:
    """Return the login ``profile`` holds: its account, role and account's hosts."""
    with Client(profile.url, api_key=profile.api_key) as http:
        if (caller := _caller(http)) is None:
            return _Login(
                profile=name,
                server=profile.url,
                account=None,
                role=None,
                hosts=None,
            )
        email = from_plain(caller.get("email"), str)
        try:
            own = [m.name for m in http.list_hosts() if m.owner == email]
        except ClientError as error:
            if error.status_code != 403:
                raise
            own = None
        return _Login(
            profile=name,
            server=profile.url,
            account=email,
            role=from_plain(caller.get("role"), str),
            hosts=None if own is None else [_host(http, each) for each in own],
        )


def _host(http: Client, name: str) -> _Host:
    """Return one of the account's hosts, its facts while online, and its runs."""
    detail = http.get_host(name)
    online = detail.status == "online"
    return _Host(
        name=name,
        status=detail.status,
        this_computer=name == host_name(),
        knowop_version=(
            from_plain(detail.facts.get("knowop_version"), str, default=None)
            if online
            else None
        ),
        pid=_pid(detail) if online else None,
        started_at=(
            from_plain(detail.facts.get("started_at"), str, default=None)
            if online
            else None
        ),
        runs=[
            _Run(run=run.run, cli=run.cli, cwd=run.cwd, session_id=str(run.session_id))
            for run in http.host_runs(name)
        ],
    )


def _say_status(report: _Status, *, manager: str | None) -> None:
    """Print ``report`` as lines; ``manager`` names how a disabled service turns on."""
    for index, login in enumerate(report.logins):
        if index:
            echo("")
        _say_login(login)
    for each in report.services:
        flags = "".join(f" {flag}" for flag in each.flags)
        off = (
            "; turned off in Login Items: turn it on in System Settings > General > "
            "Login Items & Extensions"
            if manager == "launchd"
            else "; disabled: knowop connect enables it again"
        )
        echo(f"service: {each.name} {each.server}{flags}{off if each.disabled else ''}")
    if report.menubar is not None:
        echo(f"menubar: {report.menubar}")


def _say_login(login: _Login) -> None:
    """Print ``login``: its account, role and hosts, once the server accepts it."""
    echo(f"profile: {login.profile}")
    echo(f"server:  {login.server}")
    if login.account is None:
        return
    echo(f"account: {login.account}")
    echo(f"role:    {login.role}")
    if login.hosts is None:
        echo("host: listing hosts needs the writer role")
    elif not login.hosts:
        echo("host: none; run knowop connect")
    for host in login.hosts or []:
        version = host.knowop_version
        served = (f", knowop {version}" if version is not None else "") + (
            f", pid {host.pid}" if host.pid is not None else ""
        )
        live = len(host.runs)
        count = f", {live} run{'' if live == 1 else 's'} live" if live else ""
        mark = " (this computer)" if host.this_computer else ""
        echo(f"host: {host.name} {host.status}{served}{count}{mark}")
        for run in host.runs:
            echo(
                f"  run:   {run.run or run.session_id[:8]}  {run.cli}  {run.cwd or ''}",
            )


def _next_step(named: str) -> None:
    """Print the next setup step for this computer, without starting a login."""
    saved = _saved(named)
    if saved is None:
        echo("Not connected. Run:  knowop connect")
        return
    _, profile = saved
    with Client(profile.url, api_key=profile.api_key) as http:
        email = from_plain(_signed_in(http).get("email"), str)
        here = host_name()
        connected = any(
            m.name == here and m.owner == email and m.status == "online"
            for m in http.list_hosts()
        )
    if connected:
        echo(f"Connected as {here} to {profile.url}. Next:  knowop run")
    else:
        echo("Not connected. Run:  knowop connect")


# Its service here goes too, which would otherwise join the host again when it next
# starts.
def _disconnect(name: str, *, named: str, sleep: Callable[[float], None]) -> None:
    """Disconnect the host on the server, which stops its runs, and its service here."""
    host = name or host_name()
    check_name(host)
    _, profile = _profile(named)
    with Client(profile.url, api_key=profile.api_key) as http:
        for run in http.disconnect_host(host):
            echo(f"stopped: run {run.run or run.session_id} ({run.cli})")
        _remove_service(service.here(), host, profile.url)
        if _await(http, host, lambda d: d.status != "online", sleep=sleep) is None:
            raise ClientError(f"host {host} still reads online after 30 s")
    echo(
        f"disconnected: host {host}; its connect stops at its next heartbeat, "
        "and knowop connect joins it again",
    )


# At a terminal, or for a run (``for_run``), a user service serves it; elsewhere, or
# with ``--foreground``, this process does, until it is released.
def _connect(
    flags: Flags,
    *,
    named: str,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
    for_run: bool = False,
) -> None:
    """Log in unless knowop already is, to that server; then serve this computer."""
    name = flags.name or host_name()
    check_name(name)
    background = for_run or (
        sys.stdout.isatty() and not (flags.foreground or flags.log_file)
    )
    if background and os.geteuid() == 0:
        raise ClientError(
            "knowop connect would install a service of root, which serves root's "
            "shells; run it as your own user, or with --foreground",
        )
    saved = _saved(named)
    # Checked before any request, whichever way connect goes on: a login, a token
    # piped to a box, or a saved login, none of which crosses plain http.
    server = server_url(
        saved[1].url
        if saved is not None and not flags.target
        else _server(flags.target),
        "server",
    )
    # A saved login the server now refuses (its host was logged out elsewhere) is
    # no login: at a terminal connect logs in again, and a service's run says so.
    found = next(
        (each for each in _logins(named) if each[1].url == server),
        None,
    )
    if found is not None and _accepted(found[1]):
        profile, url, token = found[0], found[1].url, ""
        # The server connected to at a terminal is the one knowop's verbs then use; a
        # service's run at login leaves that as it was.
        if not flags.log_file:
            _remember(profile)
    elif flags.log_file:
        # A service has no person to approve a login, and no token piped to it.
        echo(f"knowop: not logged in to {server}; run knowop connect there", err=True)
        return
    # A box has no browser to approve a login in: the token piped to it joins alone.
    elif piped := serve.piped_token():
        _serve(
            name,
            server,
            profile="",  # pragma: no mutate[truthiness]
            token=piped,
            flags=flags,
            sleep=sleep,
        )
        return
    else:
        # The approval that logs in also enrolls and confirms this computer, and returns
        # its one-use token: one step for the person, and no second login.
        result = login.login(
            server,
            profile=named,
            host=name,
            client="knowop",
            sleep=sleep,
            clock=clock,
        )
        _remember(result.profile)
        echo(f"logged in: profile {result.profile} ({result.role})")
        profile, url = result.profile, read_profile(result.profile).url
        token = result.host_token
    if background and (services := service.here()) is not None:
        _connect_service(
            name,
            url,
            profile=profile,
            token=token,
            flags=flags,
            services=services,
            sleep=sleep,
        )
        return
    if background:
        echo(
            "knowop: no user service manager answers here (systemctl --user), so "
            "this terminal serves the host until Ctrl-C",
            err=True,
        )
    _serve(name, url, profile=profile, token=token, flags=flags, sleep=sleep)


# Joined here first, so what a service could only log (another account's host,
# --jobs with no memory cap, a refused login token) shows at this terminal.
def _connect_service(
    name: str,
    url: str,
    *,
    profile: str,
    token: str,
    flags: Flags,
    services: service.Services,
    sleep: Callable[[float], None],
) -> None:
    """Serve ``name`` from a user service, started here; then say how to reach it."""
    unit = service.Connect(
        name=name,
        server=url,
        flags=(
            *(["--sandbox"] if flags.sandbox else []),
            *(["--no-exec"] if flags.no_exec else []),
            *(
                ["--jobs", "--job-results-max", flags.job_results_max]
                if flags.jobs
                else []
            ),
        ),
    )
    with Client(url, api_key=read_profile(profile).api_key) as http:
        before = _detail(http, name)
        pid = None if before is None or before.status != "online" else _pid(before)
        ours = any((c.name, c.server) == (name, url) for c in services.every())
        if pid is not None and not ours:
            echo(
                f"connected already: host {name} at {url}, by knowop pid {pid}, "
                "which no service of this computer runs; stop that connect, then run "
                "knowop connect again to serve it from one",
            )
            return
        # Restarted when its definition or its knowop changed; else left running.
        current = (
            pid is not None
            and services.unchanged(unit)
            and before is not None
            and before.facts.get("knowop_version") == service.knowop_version()
        )
        if not current:
            _serve(
                name,
                url,
                profile=profile,
                token=token,
                flags=flags,
                sleep=sleep,
                join_only=True,
            )
            services.start(unit)

            # Not this process: the join here heartbeats once, and reads online.
            def started_elsewhere(d: HostDetail) -> bool:
                return d.status == "online" and _pid(d) not in {None, pid, os.getpid()}

            started = _await(http, name, started_elsewhere, sleep=sleep)
            if started is None:
                raise ClientError(
                    f"host {name} did not come online in 30 s; see knowop logs"
                    f"{_named(name)}",
                )
            pid = _pid(started)
    echo(
        f"connected{' already' if current else ''}: host {name} at {url} in the "
        f"background (pid {pid}); it starts again at login",
    )
    echo(f"log:  knowop logs -f{_named(name)}")
    echo(f"stop: knowop disconnect{_named(name)}")
    if (problem := services.show_menu_bar()) is not None:
        echo(
            f"knowop: no menu bar: {problem} in {sys.executable}; install PyObjC there "
            "(in a checkout, uv sync), then run knowop connect again",
            err=True,
        )
    if services.linger_off():
        echo(
            "keep it running after you log out: sudo loginctl enable-linger "
            f"{getpass.getuser()}",
        )
    if flags.follow:
        log = (
            state_dir()
            / "rekursiv-ai"
            / "knowop"
            / "connect"
            / f"{name}-{service.server_tag(url)}.log"
        )
        _print_log(log, lines=None, follow=True, sleep=sleep)


# A service manager stops a process with SIGTERM, and a closed ssh or tmux session sends
# SIGHUP; either must release the host as Ctrl-C does, or it reads online for three
# minutes after it is gone.
def _serve(
    name: str,
    server: str,
    *,
    profile: str,
    token: str,
    flags: Flags,
    sleep: Callable[[float], None],
    join_only: bool = False,
) -> None:
    """Serve ``name`` on ``server`` until it is released; ``profile`` signs in, if any."""
    before = {
        sig: signal.signal(sig, _interrupt) for sig in (signal.SIGTERM, signal.SIGHUP)
    }
    try:
        serve.connect_host(
            name,
            server=server,
            signed_in=(
                functools.partial(connect, argparse.Namespace(profile=profile))
                if profile
                else None
            ),
            sandbox=flags.sandbox,
            serve_tools=not flags.no_exec,
            # The host waits between heartbeats; a service's own run rotates its log.
            sleep=(
                functools.partial(_rotating, Path(flags.log_file), sleep)
                if flags.log_file
                else sleep
            ),
            token=token,
            jobs=flags.jobs,
            results_max=_size(flags.job_results_max),
            join_only=join_only,
        )
    except _ConnectInterrupted as stopped:
        echo(f"knowop: connect stopped by {stopped}", err=True)
        raise
    except ClientError as error:
        status = error.status_code
        # A service's own run exits 0 on a refusal no restart can change (a revoked
        # login, another account's host), so its manager starts nothing again;
        # 408 and 429 ask to try again later.
        if (
            flags.log_file
            and status is not None
            and 400 <= status < 500
            and status not in {408, 429}
        ):
            echo(f"knowop: {error}", err=True)
            echo("knowop: stopped; knowop connect starts it again", err=True)
            return
        raise
    finally:
        for sig, handler in before.items():
            signal.signal(sig, handler)


def _rotating(log: Path, sleep: Callable[[float], None], seconds: float) -> None:
    """Keep the service's log under its size, then wait ``seconds``."""
    service.rotate_log(log)
    sleep(seconds)


def _logs(
    name: str,
    *,
    named: str,
    follow: bool,
    lines: int | None,
    sleep: Callable[[float], None],
) -> None:
    """Print the log of ``name``'s connect service here; with ``follow``, follow it."""
    host = name or host_name()
    check_name(host)
    _, profile = _profile(named)
    log = (
        state_dir()
        / "rekursiv-ai"
        / "knowop"
        / "connect"
        / f"{host}-{service.server_tag(profile.url)}.log"
    )
    if not log.is_file():
        raise ClientError(
            f"no log of host {host} on this computer: its log is on the "
            "computer whose service connects it, and a connect in the foreground "
            "prints to its terminal",
        )
    _print_log(log, lines=lines, follow=follow, sleep=sleep)


# Bytes, as written: a follow that decoded each read could split a character in two.
def _print_log(
    log: Path,
    *,
    lines: int | None,
    follow: bool,
    sleep: Callable[[float], None],
) -> None:
    """Print ``log``, or its last ``lines``; with ``follow``, what it gains, until Ctrl-C."""
    sys.stdout.flush()
    out = sys.stdout.buffer
    with log.open("rb") as source:
        text = source.read()
        if lines is not None:
            kept = text.splitlines(keepends=True)
            text = b"".join(kept[max(len(kept) - lines, 0) :])
        while True:
            _ = out.write(text)
            out.flush()
            if not follow:
                return
            sleep(0.5)
            # The host empties a full log in place, once it has copied it aside.
            if os.fstat(source.fileno()).st_size < source.tell():
                source.seek(0)
            text = source.read()


def _size(text: str) -> int:
    """Return the bytes ``--job-results-max`` names, or refuse it plainly."""
    try:
        return job_limits.parse_size(text)
    except ValueError as error:
        raise ClientError(f"--job-results-max: {error}") from None


class _ConnectInterrupted(KeyboardInterrupt):
    """A signal to report once the connect has released the host."""


def _interrupt(signum: int, frame: FrameType | None) -> NoReturn:
    """End the connect as Ctrl-C does."""
    del frame
    raise _ConnectInterrupted(signal.Signals(signum).name)


def _hand_off(
    verb: str,
    args: list[str],
    *,
    named: str,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> int:
    """Run a verb ``main`` hands off as knowop's account; return its exit code."""
    if verb in (HOST_WORD, _CANVAS_WORD):
        # The login is read only when a form asks the server: help needs none.
        command = Hosts if verb == HOST_WORD else canvas.Canvas
        command.dispatch(
            # Neither command reads the verb it is run under (each has one name, and
            # ``del``s it), so a mutant that hands it none is this same call.
            verb,  # pragma: no mutate
            args,
            lambda: connect(argparse.Namespace(profile=_profile(named)[0])),
        )
        return 0
    if verb == "job":
        name, _ = _profile(named)
        return jobs_cli.main(["--profile", name, *args])
    if verb == _WATCH_PR_WORD:
        # Read only once the command asks the server: a help page needs no login.
        return watch_pr_cli.main(
            args,
            login=functools.partial(_profile, named),
            connect=_client,
        )
    # A refused command is refused here, before a login. A daemon's start is handed
    # the server's address and a key of its own on stdin; a help page and an action
    # of sagent's ask the server nothing. None reads a profile, and none logs in.
    if not process_command.logs_in(args):
        return process_main.main(args, profile=None)
    if _saved(named) is None:
        _log_in(named=named, sleep=sleep, clock=clock)
    _, profile = _profile(named)
    flags, _ = process_command.parse_run(process_command.run_parser(), args)
    if flags.stop is None and flags.on in {None, host_name()}:
        _connected_here(profile, named=named, sleep=sleep)
    # Sagent's own agent, live in Trackinizer: knowop's flags, then sagent's after
    # ``--``.
    return process_main.main(args, profile=profile)


# A run's agent on this computer needs its daemon: one with no screen is started
# through it, and a screen's opens this computer's terminal in Chat by it. Its output
# goes to stderr, since a run's first line on stdout is its session's id.
def _connected_here(
    profile: Profile,
    *,
    named: str,
    sleep: Callable[[float], None],
) -> None:
    """Connect this computer as ``knowop connect`` does, unless it is an online host."""
    here = host_name()
    try:
        with Client(profile.url, api_key=profile.api_key) as http:
            listed = http.list_hosts()
    except ClientError:
        return
    if any(m.name == here and m.status == "online" for m in listed):
        return
    services = service.here()
    if services is None:
        raise ClientError(
            f"this computer ({here}) is not connected, and no user service manager "
            "answers here to keep it connected; run knowop connect --foreground in "
            "another terminal, then run this again",
        )
    # Its own service's flags, if it has one, so a connect started here keeps them.
    installed = next(
        (c for c in services.every() if (c.name, c.server) == (here, profile.url)),
        None,
    )
    words = ["connect", *(() if installed is None else installed.flags)]
    echo(
        f"knowop: this computer ({here}) is not connected to {profile.url}; "
        f"connecting it first: knowop {shlex.join(words)}",
        err=True,
    )
    parser = argparse.ArgumentParser()
    _add_arguments(parser)
    said = OUT_STREAM.set(sys.stderr)
    try:
        _connect(
            cast(Flags, parser.parse_args(words)),
            named=named,
            sleep=sleep,
            # The run's login was just accepted, so this connect never logs in, and
            # never reads a login code's clock.
            clock=time.monotonic,  # pragma: no mutate
            for_run=True,
        )
    finally:
        OUT_STREAM.reset(said)


def _detail(http: Client, name: str) -> HostDetail | None:
    """Return the server's view of host ``name``; ``None`` before it is enrolled."""
    try:
        return http.get_host(name)
    except ClientError as error:
        if error.status_code != 404:
            raise
        return None


def _await(
    http: Client,
    name: str,
    until: Callable[[HostDetail], bool],
    *,
    sleep: Callable[[float], None],
) -> HostDetail | None:
    """Ask the server for ``name`` until ``until`` holds, for 30 s; ``None`` past it."""
    for _ in range(30):
        if until(detail := http.get_host(name)):
            return detail
        sleep(1.0)
    return None


def _pid(detail: HostDetail) -> int | None:
    """Return the process its facts say serves a host; ``None`` from an old knowop."""
    return from_plain(detail.facts.get("pid"), int, default=None)


def _named(name: str) -> str:
    """Return a space and ``name``, or nothing for the host verbs default to."""
    return "" if name == host_name() else f" {name}"


def _signed_in(http: Client) -> dict[str, object]:
    """Return the account the key belongs to, or say to log in when it is refused."""
    if (caller := _caller(http)) is None:
        raise ClientError(
            f"not logged in: {http.base_url} refused the key; run knowop login",
        )
    return caller


def _accepted(profile: Profile) -> bool:
    """Return whether the server still accepts ``profile``'s key."""
    with Client(profile.url, api_key=profile.api_key) as http:
        return _caller(http) is not None


def _caller(http: Client) -> dict[str, object] | None:
    """Return the account the key belongs to, or ``None`` when the server refuses it."""
    try:
        return from_plain(http.get("/api/me/profile"), dict[str, object])
    except ClientError as error:
        if error.status_code == 401:
            return None
        raise


def _profile(named: str) -> tuple[str, Profile]:
    """Return knowop's profile, or ``named``, or refuse when it is not logged in."""
    if (saved := _saved(named)) is None:
        flag = f" --profile {named}" if named else ""
        raise ClientError(f"not logged in; run knowop{flag} login")
    return saved


def _saved(named: str) -> tuple[str, Profile] | None:
    """Return the name and contents of knowop's profile, or of ``named``, if saved."""
    return next(iter(_logins(named)), None)


def _logins(named: str) -> list[tuple[str, Profile]]:
    """Return the login of the profile ``--profile`` named, else knowop's own logins."""
    return _found_logins([named]) if named else _kept_logins()


# Knowop keeps a login per server it connected to, one profile name a line: the first is
# the one every verb uses, and a connect service of another server signs in with its own.
def _kept_logins() -> list[tuple[str, Profile]]:
    """Return knowop's logins, the one its verbs use first; a deleted profile is none."""
    try:
        names = (
            (config_dir() / "rekursiv-ai" / "knowop" / "profile").read_text().split()
        )
    except FileNotFoundError:
        return []
    return _found_logins(names)


def _found_logins(names: Sequence[str]) -> list[tuple[str, Profile]]:
    """Return each of ``names`` once, with its saved profile; one not saved is left out."""
    logins: list[tuple[str, Profile]] = []
    for name in dict.fromkeys(names):
        try:
            logins.append((name, read_profile(name)))
        except ClientError:
            continue
    return logins


def _remember(profile: str) -> None:
    """Make ``profile`` the one every knowop verb uses, keeping knowop's other logins."""
    _keep_logins([profile, *(name for name, _ in _kept_logins() if name != profile)])


def _keep_logins(names: list[str]) -> None:
    """Record ``names`` as knowop's logins, the first in use; none removes the record."""
    path = config_dir() / "rekursiv-ai" / "knowop" / "profile"
    if not names:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{name}\n" for name in names))


# This module is imported, never run directly, so it carries no run guard and no
# shebang. ``__main__.py`` is the sole executable; run it via ``python -m``.
