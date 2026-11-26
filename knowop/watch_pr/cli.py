"""``knowop watch-pr``: watch GitHub pull requests for agent sessions.

A session watches a pull request with ``add``; ``serve``, on any host whose ``gh`` is
logged in, reads each watched pull request every minute, until stopped or nothing is
watched for 3 hours, and tells each watching session's agent when checks fail, it conflicts with its base, it merges, or a draft's
checks all pass. One agent fixes; ``fix`` names another. Chat shows the pull requests
over the session, as the pill.

SESSION names the watching session: ``LABEL#N`` (as AgentSend names its agent),
``AgentSession#N``, its number, or its UUID. With none, ``$TRAX_SESSION`` names it:
``knowop run`` sets it for the tools of its agent. An agent whose tools run on another
host asks its KnowOp tool's ``session.self``.

Examples:
  knowop watch-pr add https://github.com/acme/app/pull/7 --session Agent#412
  knowop watch-pr status
  knowop watch-pr fix https://github.com/acme/app/pull/7 --session Agent#415
  knowop watch-pr remove https://github.com/acme/app/pull/7 --session Agent#412
  knowop watch-pr serve --detach
  knowop watch-pr stop

"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Final, Protocol, cast
from uuid import UUID

import argparse
import logging
import os
import signal
import sys
import threading

from trackinizer.client.errors import ClientError
from trackinizer.trax.render import echo
from trackinizer.wire.refs import SeqRef
from trackinizer.wire.wire_sessions import SessionAgent
from treekle import from_plain

from knowop.hosts import host_name
from knowop.watch_pr.github import (
    GhError,
    GhUnansweredError,
    link,
    read_pull_request,
    run_gh,
)
from knowop.watch_pr.news import unread_reason
from knowop.watch_pr.rows import PullRequestRows, Stored, Watcher
from knowop.watch_pr.serve import (
    ServingError,
    detach,
    serve,
    serve_files,
    serving,
)
from knowop.watch_pr.watch import Watch, WatchClient


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import FrameType

    from trackinizer.trax.profile import Profile
    from trackinizer.types.inquiries import Inquiry
    from trackinizer.wire.refs import Ref

    from knowop.watch_pr.github import GhRunner
    from knowop.watch_pr.serve import ServeFiles


__all__ = ["WORD", "WatchPrClient", "main"]

WORD: Final = "watch-pr"
"""The verb ``knowop`` runs these commands under."""

_ACTOR: Final = "knowop-watch-pr"
"""Who the rows' writes are recorded as."""


class WatchPrClient(WatchClient, Protocol):
    """What the commands ask of the server: the watch's calls, and a session's id."""

    def resolve_id(self, ref: Ref) -> tuple[Inquiry.InquiryKind, UUID]:
        """Resolve a session's number."""
        ...


class _Flags(Protocol):
    command: str
    url: str
    one: str | None
    session: str
    detach: bool


def main(
    args: Sequence[str],
    *,
    login: Callable[[], tuple[str, Profile]],
    connect: Callable[[Profile], WatchPrClient],
    gh: GhRunner = run_gh,
) -> int:
    """Run one ``knowop watch-pr`` command; return its exit code.

    Args:
      args: The arguments after ``watch-pr``.
      login: Returns knowop's profile and its login, read once a command needs it.
      connect: Makes a client of the server a login names.
      gh: Runs ``gh``.

    Returns:
      code: 0 on success, 1 when the server or ``gh`` refuses, 2 for a refused
        command.

    """
    flags = cast(_Flags, _parser().parse_args(args))
    try:
        name, profile = login()
        host = host_name(profile.url)
        files = serve_files(profile.url)
        match flags.command:
            case "serve" if flags.detach:
                _detached(files=files, profile=name)
            case "serve":
                _serve(partial(connect, profile), files=files, host=host)
            case "stop":
                _stop(files)
            case _:
                _session_command(
                    flags,
                    client=connect(profile),
                    files=files,
                    gh=gh,
                    host=host,
                )
                if flags.command == "add":
                    _detached(files=files, profile=name)
    except (ClientError, GhError, ServingError, ValueError) as error:
        echo(f"knowop watch-pr: {error}", err=True)
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    """Return the parser of ``knowop watch-pr``'s commands."""
    parser = argparse.ArgumentParser(
        prog=f"knowop {WORD}",
        description=(__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, said in (
        ("add", "watch a pull request for a session; start this host's serve"),
        ("remove", "let a session go of a pull request, as Chat's x does"),
        ("fix", "make a watching session the one whose agent fixes it"),
    ):
        command = commands.add_parser(
            name,
            help=said,
            description=f"{said.capitalize()}.",
        )
        command.add_argument("url", help="https://github.com/OWNER/REPO/pull/N")
        _add_session(command)
    status = commands.add_parser(
        "status",
        help="list the watched pull requests, and whether this host serves",
        description=(
            "List each open pull request a session watches (or the one named, open "
            "or not): its checks, who watches it, who fixes it, and when a serve last "
            "read it; and whether a serve runs on this host."
        ),
    )
    status.add_argument("one", nargs="?", metavar="url", help="one pull request alone")
    commands.add_parser(
        "serve",
        help="read every watched pull request each minute, until stopped or idle",
        description=(
            "Read every pull request a session of yours watches through this host's "
            "gh, each minute; write what GitHub says to its row, and tell each "
            "watching agent the news. One serve runs per host; one on another host "
            "tells nothing twice. It ends once nothing is watched for 3 hours; the "
            "next add starts one again."
        ),
    ).add_argument(
        "--detach",
        action="store_true",
        help="run in the background, outliving this command; print its pid",
    )
    commands.add_parser("stop", help="stop this host's serve")
    return parser


def _add_session(parser: argparse.ArgumentParser) -> None:
    """Register ``--session`` on ``parser``."""
    parser.add_argument(
        "--session",
        default=os.environ.get("TRAX_SESSION", ""),
        metavar="SESSION",
        help="LABEL#N, AgentSession#N, N or a UUID; default $TRAX_SESSION",
    )


def _session_command(
    flags: _Flags,
    *,
    client: WatchPrClient,
    files: ServeFiles,
    gh: GhRunner,
    host: str,
) -> None:
    """Run ``add``, ``remove``, ``fix`` or ``status``."""
    rows = PullRequestRows(client=client, actor=_ACTOR)
    me = rows.profile()["email"]
    if flags.command == "status":
        one = flags.one
        _status(rows, me=me, url="" if one is None else link(one), files=files)
        return
    url = link(flags.url)
    watcher = _watcher(flags.session, client=client)
    match flags.command:
        case "add":
            try:
                read = read_pull_request(gh, url)
            # Not held as a serve holds it: the agent that ran add tries again.
            except GhUnansweredError as error:
                raise GhError(
                    f"{host} couldn't reach GitHub: {error}\nNothing is watched. Run "
                    "add again; if GitHub stays out of reach, add it from another host.",
                ) from error
            except GhError as error:
                raise GhError(
                    f"{unread_reason(error, host=host)}\nNothing is watched. Tell the "
                    "person; or add it from another host whose gh is logged in.",
                ) from error
            row = rows.made(url, me=me, title=f"{read.name} {read.title}")
            rows.watch(row, watcher=watcher)
            # Read now, not at the serve's next pass, so Chat shows it at once.
            Watch(client=client, host=host, gh=gh).read(_stored(rows, url, me=me))
            echo(f"watching {url} for session {watcher.session}")
            if read.state != "OPEN":
                echo(f"it is {read.state.lower()}: no news of it will come")
        case "remove":
            stored = _stored(rows, url, me=me)
            rows.dismiss(stored.row, session=watcher.session)
            echo(f"let go of {url} for session {watcher.session}")
        case _:
            stored = _stored(rows, url, me=me)
            if watcher.session not in {each.session for each in stored.watchers}:
                rows.watch(stored.row, watcher=watcher)
            _ = rows.write_pr(stored, partial(_fixed, session=watcher.session))
            echo(f"session {watcher.session} fixes {url}")


def _fixed(stored: Stored, *, session: UUID) -> tuple[dict[str, object], None]:
    """Return the row's ``app.knowop.pr`` with ``session`` its fixer."""
    return stored.pr | {"fixer": str(session)}, None


def _status(rows: PullRequestRows, *, me: str, url: str, files: ServeFiles) -> None:
    """Print each watched pull request, and whether a serve runs here."""
    # One named is shown merged or closed too, so an agent sees its add landed.
    candidates: list[Stored] = []
    if url:
        found = rows.find(url, me=me)
        if found is not None:
            candidates.append(found)
    else:
        candidates = rows.listed(me, status="active")
    listed = [each for each in candidates if each.watchers]
    for stored in listed:
        pr = stored.pr
        checks = from_plain(
            pr.get("checks"),
            dict[str, object],
            default=dict[str, object](),
        )
        failed = from_plain(checks.get("failed"), list[str], default=list[str]())
        echo(
            f"{stored.url}  {pr.get('state') or 'unread'}  "
            f"checks {checks.get('passed', 0)}/{checks.get('total', 0)}"
            + (f", {len(failed)} failed" if failed else "")
            + (f"  error: {pr['error']}" if pr.get("error") else ""),
        )
        fixer = from_plain(pr.get("fixer"), str, default="")
        for watcher in stored.watchers:
            role = " (fixes it)" if str(watcher.session) == fixer else ""
            echo(f"  watched by {watcher.label} of session {watcher.session}{role}")
        echo(
            f"  read {pr.get('read_at') or 'never'}"
            + (f" by {pr['read_by']}" if pr.get("read_by") else ""),
        )
    if not listed:
        echo(f"no session watches {url}" if url else "no open pull request is watched")
    pid = serving(files)
    echo(
        f"serve on this host: pid {pid}"
        if pid is not None
        else "serve on this host: none; knowop watch-pr serve --detach starts one",
    )


def _detached(*, files: ServeFiles, profile: str) -> None:
    """Start this host's serve in the background, or name the one running."""
    # ``-P``: under ``-m`` the working directory would come first on the path, and a
    # module left there would run in knowop's place.
    knowop = [sys.executable, "-P", "-m", (__package__ or "").rpartition(".")[0]]
    pid = detach([*knowop, "--profile", profile, WORD, "serve"], files=files)
    echo(f"serve on this host: pid {pid}; log {files.log}")


def _serve(
    client: Callable[[], WatchPrClient],
    *,
    files: ServeFiles,
    host: str,
) -> None:
    """Serve here until stopped by a signal."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # A line each request would bury the passes' own.
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    stop = threading.Event()
    before = {
        sig: signal.signal(sig, _stopping(stop))
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    }
    try:
        serve(client, files=files, host=host, stop=stop)
    finally:
        for sig, handler in before.items():
            signal.signal(sig, handler)


def _stopping(stop: threading.Event) -> Callable[[int, FrameType | None], None]:
    """Return a signal handler that ends the serve after its pass."""

    def handler(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        stop.set()

    return handler


def _stop(files: ServeFiles) -> None:
    """Stop this host's serve, if one runs."""
    pid = serving(files)
    if pid is None:
        echo("serve on this host: none")
        return
    os.kill(pid, signal.SIGTERM)
    echo(f"stopped the serve on this host (pid {pid})")


def _watcher(text: str, *, client: WatchPrClient) -> Watcher:
    """Return the session ``text`` names, and its agent's label."""
    if not text:
        raise ValueError(
            "name the watching session with --session LABEL#N; an agent asks its "
            "KnowOp tool's session.self",
        )
    if text.isdigit() or text.startswith("AgentSession#"):
        text = f"Agent#{text.removeprefix('AgentSession#')}"
    elif "#" not in text:
        text = f"Agent#{UUID(text)}"
    named = SessionAgent.parse(text)
    session = (
        client.resolve_id(SeqRef(kind="AgentSession", seq=named.session))[1]
        if isinstance(named.session, int)
        else named.session
    )
    return Watcher(session=session, label=named.label)


def _stored(rows: PullRequestRows, url: str, *, me: str) -> Stored:
    """Return the person's row of ``url``, or refuse when no session watched it."""
    stored = rows.find(url, me=me)
    if stored is None:
        raise ValueError(f"no session watches {url}; knowop watch-pr add watches it")
    return stored
