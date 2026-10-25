"""Log in to KnowOp, connect this computer, and run recorded agents and GPU jobs.

``knowop login`` signs in with a code a person approves in a browser and saves the
key as a trax profile. knowop records that profile as its own, and every verb uses
it, never trax's current profile, so knowop acts only on the server it logged in to.

Examples:
  knowop connect          log in if needed, then connect this computer
  knowop login acme       log in to https://acme.knowop.ai, without connecting
  knowop run claude       Claude Code in this folder, recorded, reachable from Chat
  knowop job submit -- python train.py
                          run a job on a GPU node from this repository
  knowop job status       list your jobs
  knowop machine          list the machines campaigns may run on
  knowop canvas           read the canvas a run sits on; --help shows how to change it
  knowop status           profile, server, account, role, machines and their runs
  knowop disconnect       stop this computer's runs and connect, everywhere

"""

from __future__ import annotations

from importlib import metadata
from typing import TYPE_CHECKING, Final, NoReturn, Protocol, cast

import argparse
import functools
import io
import re
import signal
import sys
import time
import uuid

from trackinizer.client.client import Client, server_url
from trackinizer.client.errors import ClientError
from trackinizer.lib.userdirs import config_dir
from trackinizer.trax.client_cache import close_clients
from trackinizer.trax.connection import connect
from trackinizer.trax.profile import Profile, del_profile, read_profile
from trackinizer.trax.render import echo
from treekle import from_plain

from knowop.machines import (
    MACHINE_WORD,
    Machines,
    check_name,
    host_name,
)
from knowop.shell import serve


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import FrameType

    from trackinizer.trax import login

    from knowop import canvas
    from knowop.agent import (
        cli as agent_cli,
        detach,
    )
    from knowop.jobs import cli as jobs_cli
    from knowop.shell import job_limits
else:
    from wrapt import lazy_import

    # Only ``login``, ``connect`` and a first ``run`` reach the login, and only ``run``
    # and ``job`` the runners. Their imports cost ~100 ms or more that ``status`` would
    # pay at every start. Only ``canvas`` reaches trax's verbs, ~20 ms more.
    agent_cli = lazy_import("knowop.agent.cli")
    canvas = lazy_import("knowop.canvas")
    detach = lazy_import("knowop.agent.detach")
    jobs_cli = lazy_import("knowop.jobs.cli")
    job_limits = lazy_import("knowop.shell.job_limits")
    login = lazy_import("trackinizer.trax.login")


_DEFAULT_SERVER: Final = "https://alpha.knowop.ai"
"""The server ``login``, ``connect`` and a first ``run`` log in to when none is named."""

_HOST_WORDS: Final = frozenset({"attach", "ls", "log", "send", "stop"})
"""The ``run`` words that reach a detached run on this computer instead of starting one."""

_CANVAS_WORD: Final = "canvas"
"""The verb trax's canvas command runs under, on knowop's login."""


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
        if not args:
            _next_step()
            return 0
        # Everything after ``run`` or ``job`` belongs to the runner, ``--help`` and
        # every other flag included; argparse would read them as knowop's own. Every
        # ``machine`` and ``canvas`` form is that verb's, its help pages too.
        if args[:1] in ([MACHINE_WORD], [_CANVAS_WORD]) or (
            args[:1] in (["run"], ["job"]) and args[1:] not in (["-h"], ["--help"])
        ):
            return _hand_off(args[0], args[1:], sleep=sleep, clock=clock)
        parser = argparse.ArgumentParser(
            prog="knowop",
            description=(__doc__ or "").strip(),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        _add_arguments(parser)
        flags = cast(Flags, parser.parse_args(args))
        _run(flags, sleep=sleep, clock=clock)
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


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the verbs and their flags on ``parser``."""
    parser.add_argument("--version", action="version", version=f"knowop {_version()}")
    verbs = parser.add_subparsers(dest="verb", required=True, metavar="VERB")
    verb = verbs.add_parser(
        "login",
        help="log in with a code approved in a browser",
        description=(
            "Log in with a code approved in a browser; knowop uses this login "
            "from then on."
        ),
    )
    verb.add_argument(
        "target",
        nargs="?",
        default="",
        metavar="ORG|URL",
        help=f"ORG means https://ORG.knowop.ai; default {_DEFAULT_SERVER}",
    )
    verbs.add_parser(
        "logout",
        help="revoke knowop's key on the server and delete its profile",
    )
    verbs.add_parser(
        "status",
        help="show knowop's profile, server, account, role, machines and their runs",
    )
    verbs.add_parser(
        "disconnect",
        help="stop this computer's runs, then disconnect its machine on the server",
        description=(
            "Stop every detached run on this computer, then disconnect the machine on "
            "the server: its connect is revoked, and every run of yours that names it "
            "is stopped, here or anywhere. knowop connect joins it again, with no "
            "login; knowop logout also revokes this computer's key."
        ),
    ).add_argument(
        "name",
        nargs="?",
        default="",
        help="the machine; default this computer's short host name",
    )
    _add_connect_arguments(
        verbs.add_parser(
            "connect",
            help="log in if needed, then serve this computer's terminals and commands",
            description=(
                "Connect this computer and serve its terminals, and the commands the "
                "server's assistant sends, until Ctrl-C. When knowop is not logged in "
                "to the server, log in first: the same approval enrolls and confirms "
                "this computer. An enrollment token piped to stdin joins instead, "
                "with no login."
            ),
        ),
    )
    verbs.add_parser(
        "run",
        help="run an agent here, recorded: claude, codex, attach NAME",
        description=(
            "Run Claude Code or Codex in this folder as knowop's account, recorded "
            "and reachable from Chat; its arguments pass to the runner unchanged. "
            "Logs in first when knowop is not logged in."
        ),
    ).add_argument(
        "args",
        nargs=argparse.REMAINDER,
        metavar="claude|codex|attach|ls|log|send|stop|resume ...",
    )
    verbs.add_parser(
        "job",
        help="run, list and cancel jobs on a GPU node, and keep their secrets",
        description=(
            "Run a job on a GPU node from this repository as knowop's account, read "
            "its log and results, list and cancel jobs, and keep their secrets; its "
            "arguments pass to the job runner unchanged."
        ),
    ).add_argument(
        "args",
        nargs=argparse.REMAINDER,
        metavar="submit|logs|results|status|cancel|secrets|secret ...",
    )
    # Only listed here: ``main`` hands every ``machine`` form to the registry.
    verbs.add_parser(
        MACHINE_WORD,
        help="list, show, set and delete the machines campaigns may run on",
    )
    # Only listed here too: ``main`` hands every ``canvas`` form to trax's verb.
    verbs.add_parser(
        _CANVAS_WORD,
        help="read and change the canvas an agent run sits on, as trax canvas does",
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
        help="the machine's name; default this computer's short host name",
    )
    parser.add_argument(
        "--no-exec",
        action="store_true",
        help=(
            "serve terminals only: refuse the commands the server's assistant would "
            "run here; each command it runs is logged in NAME.exec.log beside the "
            "machine credentials"
        ),
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="declare a disposable computer, where agents act without asking",
    )
    parser.add_argument(
        "--jobs",
        action="store_true",
        help=(
            "also run the jobs your account sends here with knowop job submit "
            "--machine NAME, each in a fresh worktree of its pushed commit under a "
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


def _run(
    flags: Flags,
    *,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> None:
    """Run the verb ``flags`` names."""
    match flags.verb:
        case "login":
            _log_in(flags.target, sleep=sleep, clock=clock)
        case "logout":
            _log_out()
        case "status":
            _status()
        case "disconnect":
            _disconnect(flags.name)
        case _:
            # ``connect``: ``main`` hands ``run`` and ``job`` off before parsing, and
            # argparse refuses any other word.
            _connect(flags, sleep=sleep, clock=clock)


def _log_in(
    target: str,
    *,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> None:
    """Log in to the server ``target`` names, and make its profile knowop's."""
    result = login.login(_server(target), client="knowop", sleep=sleep, clock=clock)
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


def _log_out() -> None:
    """Revoke knowop's key on its server, then delete its profile."""
    name, profile = _profile()
    with Client(profile.url, api_key=profile.api_key) as http:
        if (caller := _caller(http)) is not None:
            key = from_plain(caller.get("api_key_id"), uuid.UUID)
            http.post(f"/api/me/tokens/{key}/revoke")
    # ``del_profile`` refuses the profile trax has pinned, and a pin naming a deleted
    # profile would fail every later trax command. Unpinned, trax falls back to its
    # ``default`` profile.
    pin = config_dir() / "rekursiv-ai" / "trax" / "current"
    if pin.exists() and pin.read_text().strip() == name:
        pin.unlink()
    del_profile(name)
    (config_dir() / "rekursiv-ai" / "knowop" / "profile").unlink(missing_ok=True)
    revoked = "revoked its key" if caller is not None else "its key was revoked already"
    echo(f"logged out: {revoked}; deleted profile {name}")


def _status() -> None:
    """Print knowop's profile, server, account, role and the account's machines."""
    name, profile = _profile()
    with Client(profile.url, api_key=profile.api_key) as http:
        if (caller := _caller(http)) is None:
            raise ClientError(
                f"not logged in: {profile.url} refused the key; run knowop login",
            )
        email = from_plain(caller.get("email"), str)
        echo(f"profile: {name}")
        echo(f"server:  {profile.url}")
        echo(f"account: {email}")
        echo(f"role:    {from_plain(caller.get('role'), str)}")
        try:
            listed = http.list_machines()
        except ClientError as error:
            if error.status_code != 403:
                raise
            echo("machine: listing machines needs the writer role")
            return
        own = [machine for machine in listed if machine.owner == email]
        runs = {machine.name: http.machine_runs(machine.name) for machine in own}
    here = host_name()
    for machine in own:
        mark = " (this computer)" if machine.name == here else ""
        live = len(runs[machine.name])
        count = f", {live} run{'' if live == 1 else 's'} live" if live else ""
        echo(f"machine: {machine.name} {machine.status}{count}{mark}")
        for run in runs[machine.name]:
            echo(f"  run:   {run.run or run.session_id}  {run.cli}  {run.cwd or ''}")
    if not own:
        echo("machine: none; run knowop connect")
    # A run the server places on no machine of this account's still runs here.
    listed_runs = {run.run for each in runs.values() for run in each}
    for host in detach.live_hosts():
        if host.name not in listed_runs:
            echo(f"here:    {host.name}  {' '.join(host.command.split()[:1])}")


def _next_step() -> None:
    """Print the next setup step for this computer, without starting a login."""
    saved = _saved()
    if saved is None:
        echo("Not connected. Run:  knowop connect")
        return
    _, profile = saved
    with Client(profile.url, api_key=profile.api_key) as http:
        if (caller := _caller(http)) is None:
            raise ClientError(
                f"not logged in: {profile.url} refused the key; run knowop login",
            )
        email = from_plain(caller.get("email"), str)
        here = host_name()
        connected = any(
            m.name == here and m.owner == email and m.status == "online"
            for m in http.list_machines()
        )
    if connected:
        echo(f"Connected as {here} to {profile.url}. Next:  knowop run claude")
    else:
        echo("Not connected. Run:  knowop connect")


# This computer's runs first: once the server stops them, each is already ending,
# and a stop here would race its exit.
def _disconnect(name: str) -> None:
    """Stop this computer's runs when it is the machine, then disconnect the machine."""
    machine = name or host_name()
    check_name(machine)
    _, profile = _profile()
    if machine == host_name() and detach.stop_all() != 0:
        raise ClientError("a run here did not stop; see knowop run ls")
    with Client(profile.url, api_key=profile.api_key) as http:
        stopped = http.disconnect_machine(machine)
    for run in stopped:
        echo(f"stopped: run {run.run or run.session_id} ({run.cli})")
    echo(
        f"disconnected: machine {machine}; its connect stops at its next heartbeat, "
        "and knowop connect joins it again",
    )


def _connect(
    flags: Flags,
    *,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> None:
    """Log in unless knowop already is, to that server; then serve this computer."""
    name = flags.name or host_name()
    check_name(name)
    saved = _saved()
    server = (
        saved[1].url
        if saved is not None and not flags.target
        else _server(flags.target)
    )
    if saved is not None and saved[1].url == server.rstrip("/"):
        _serve(name, saved[1].url, profile=saved[0], token="", flags=flags)
        return
    # A box has no browser to approve a login in: the token piped to it joins alone,
    # and the server it names is checked as a login's would be.
    if piped := serve.piped_token():
        url = login.secure_url(server_url(server, "server"), what="server")
        _serve(name, url, profile="", token=piped, flags=flags)
        return
    # The approval that logs in also enrolls and confirms this computer, and returns its
    # one-use token: one step for the person, and no second login.
    result = login.login(
        server,
        machine=name,
        client="knowop",
        sleep=sleep,
        clock=clock,
    )
    _remember(result.profile)
    echo(f"logged in: profile {result.profile} ({result.role})")
    url = read_profile(result.profile).url
    _serve(name, url, profile=result.profile, token=result.machine_token, flags=flags)


# A service manager stops a process with SIGTERM, and a closed ssh or tmux session sends
# SIGHUP; either must release the machine as Ctrl-C does, or it reads online for three
# minutes after it is gone.
def _serve(
    name: str,
    server: str,
    *,
    profile: str,
    token: str,
    flags: Flags,
) -> None:
    """Serve ``name`` on ``server`` until it is released; ``profile`` signs in, if any."""
    before = {
        sig: signal.signal(sig, _interrupt) for sig in (signal.SIGTERM, signal.SIGHUP)
    }
    try:
        serve.connect_machine(
            name,
            server=server,
            signed_in=(
                functools.partial(connect, argparse.Namespace(profile=profile))
                if profile
                else None
            ),
            sandbox=flags.sandbox,
            exec_commands=not flags.no_exec,
            token=token,
            jobs=flags.jobs,
            results_max=_size(flags.job_results_max),
        )
    except KeyboardInterrupt:
        _say_runs_left()
        raise
    finally:
        for sig, handler in before.items():
            signal.signal(sig, handler)


# The machine reads offline once its connect lets go, but a detached run reaches the
# server with the account's key, not the machine's: it is not stopped with the connect.
def _say_runs_left() -> None:
    """Name the detached runs still live on this computer, and how to stop them."""
    hosts = detach.live_hosts()
    if hosts:
        names = ", ".join(host.name for host in hosts)
        echo(
            f"knowop: still reaching the server from here: {names}; stop them with "
            "knowop run stop --all, or everything with knowop disconnect",
            err=True,
        )


def _size(text: str) -> int:
    """Return the bytes ``--job-results-max`` names, or refuse it plainly."""
    try:
        return job_limits.parse_size(text)
    except ValueError as error:
        raise ClientError(f"--job-results-max: {error}") from None


def _interrupt(signum: int, frame: FrameType | None) -> NoReturn:
    """End the connect as Ctrl-C does."""
    del signum, frame
    raise KeyboardInterrupt


def _hand_off(
    verb: str,
    args: list[str],
    *,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> int:
    """Run a verb ``main`` hands off as knowop's account; return its exit code."""
    if verb in (MACHINE_WORD, _CANVAS_WORD):
        # The login is read only when a form asks the server: help needs none.
        (Machines if verb == MACHINE_WORD else canvas.Canvas).dispatch(
            verb,
            args,
            lambda: connect(argparse.Namespace(profile=_profile()[0])),
        )
        return 0
    if verb == "job":
        name, _ = _profile()
        return jobs_cli.main(["--profile", name, *args])
    if _starts_an_agent(args):
        if _saved() is None:
            _log_in("", sleep=sleep, clock=clock)
        name, profile = _profile()
        _say_whether_connected(profile)
    else:
        # A help page and the commands that reach a detached run on this computer ask
        # the server nothing, so they need no login.
        name = saved[0] if (saved := _saved()) is not None else ""
    # The runner reads a leading ``--profile``, and of two the last wins: one the caller
    # names first (``knowop run claude --profile other``) overrides knowop's.
    leading = ["--profile", name] if name else []
    match args[:1]:
        case ["codex"]:
            return agent_cli.codex_main([*leading, *args[1:]])
        case ["claude"]:
            return agent_cli.main([*leading, *args[1:]])
    # ``attach``, ``ls``, ``log``, ``send``, ``stop``, ``resume`` and the runner's own
    # flags are the claude runner's words.
    return agent_cli.main([*leading, *args])


def _starts_an_agent(args: Sequence[str]) -> bool:
    """Whether ``knowop run ARGS`` starts an agent, not a help page or a host command."""
    head = args[: args.index("--")] if "--" in args else args
    return not (head[:1] and head[0] in _HOST_WORDS) and not {"-h", "--help"} & set(
        head,
    )


# A run reaches Chat without this computer being connected; a connected one also opens
# its terminal there, which is how a person takes the run over from a browser.
def _say_whether_connected(profile: Profile) -> None:
    """Say to run ``knowop connect`` when this computer is not an online machine."""
    try:
        here = host_name()
        with Client(profile.url, api_key=profile.api_key) as http:
            listed = http.list_machines()
    except ClientError:
        return
    if not any(m.name == here and m.status == "online" for m in listed):
        echo(
            f"knowop: this computer ({here}) is not connected; run knowop connect "
            "in another terminal to open its terminal from Chat",
            err=True,
        )


def _caller(http: Client) -> dict[str, object] | None:
    """Return the account the key belongs to, or ``None`` when the server refuses it."""
    try:
        return from_plain(http.get("/api/me/profile"), dict[str, object])
    except ClientError as error:
        if error.status_code == 401:
            return None
        raise


def _profile() -> tuple[str, Profile]:
    """Return knowop's profile, or refuse when knowop is not logged in."""
    if (saved := _saved()) is None:
        raise ClientError("not logged in; run knowop login")
    return saved


def _saved() -> tuple[str, Profile] | None:
    """Return the name and contents of knowop's profile; ``None`` before a login."""
    try:
        name = (config_dir() / "rekursiv-ai" / "knowop" / "profile").read_text().strip()
        return name, read_profile(name)
    except (FileNotFoundError, ClientError):
        return None


def _remember(profile: str) -> None:
    """Make ``profile`` the one every knowop verb uses."""
    path = config_dir() / "rekursiv-ai" / "knowop" / "profile"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{profile}\n")


def _version() -> str:
    """Return the installed version; ``unknown`` from a source checkout."""
    try:
        return metadata.version("knowop")
    except metadata.PackageNotFoundError:
        return "unknown"


# This module is imported, never run directly, so it carries no run guard and no
# shebang. ``__main__.py`` is the sole executable; run it via ``python -m``.
