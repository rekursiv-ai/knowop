"""Log in to KnowOp, connect this computer, and run recorded agents and GPU jobs.

``knowop login`` signs in with a code a person approves in a browser and saves the
key as a trax profile. knowop records that profile as its own, and every verb uses
it, never trax's current profile, so knowop acts only on the server it logged in to.

At a terminal, ``knowop connect`` serves this computer from a user service (launchd's
or systemd's) that starts again at login; ``knowop logs -f`` follows it and ``knowop
disconnect`` stops it.

Examples:
  knowop connect          log in if needed, then connect this computer
  knowop login acme       log in to https://acme.knowop.ai, without connecting
  knowop run claude       Claude Code in this folder, recorded, reachable from Chat
  knowop run stop NAME    stop a run of yours, here or on another machine
  knowop job submit -- python train.py
                          run a job on a GPU node from this repository
  knowop job status       list your jobs
  knowop machine          list the machines campaigns may run on
  knowop canvas           read the canvas a run sits on; --help shows how to change it
  knowop status           profile, server, account, machines, their runs, services
  knowop logs -f          follow this computer's connect
  knowop disconnect       stop this computer's runs and connect, everywhere

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, NoReturn, Protocol, cast

import argparse
import functools
import getpass
import io
import os
import re
import signal
import sys
import time
import uuid

from trackinizer.client.client import Client, server_url
from trackinizer.client.errors import ClientError
from trackinizer.lib.userdirs import config_dir, state_dir
from trackinizer.trax.client_cache import close_clients
from trackinizer.trax.connection import connect
from trackinizer.trax.profile import Profile, del_profile, read_profile
from trackinizer.trax.render import echo
from treekle import from_plain, json_dumps

from knowop import menubar, service
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
    from trackinizer.wire.wire_machines import MachineDetail

    from knowop import canvas
    from knowop.agent import (
        cli as agent_cli,
        detach,
    )
    from knowop.jobs import cli as jobs_cli
    from knowop.shell import job_limits, machine_host
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
    machine_host = lazy_import("knowop.shell.machine_host")
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
    foreground: bool
    follow: bool
    log_file: str
    json: bool
    all: bool
    lines: int | None


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
    logout = verbs.add_parser(
        "logout",
        help="log this computer out, revoke knowop's key and delete its profile",
        description=(
            "Log this computer's machine out on the server, revoke knowop's key, "
            "delete its profile and remove this computer's connect services. NAME "
            "logs that machine out alone, as the Machines page does: its connect is "
            "revoked, its runs stop, and the keys its logins made are revoked."
        ),
    ).add_mutually_exclusive_group()
    logout.add_argument(
        "name",
        nargs="?",
        default="",
        help="a machine of yours to log out, keeping this computer's login",
    )
    logout.add_argument(
        "--all",
        action="store_true",
        help="log every machine of yours out first",
    )
    verbs.add_parser(
        "status",
        help="show knowop's profile, server, account, role, machines and their runs",
        description=(
            "Show knowop's profile, server, account and role; each of your machines, "
            "online or not, with its knowop version and process while online, and "
            "its live runs; and this computer's connect services, with their flags."
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
            "Show the knowop mark in the macOS menu bar, dimmed while a connect "
            "service here is not online, with a menu of each server knowop is logged "
            "in to and its machines. knowop connect starts it at login beside its "
            "connect services; Quit ends it until then."
        ),
    )
    logs = verbs.add_parser(
        "logs",
        help="print or follow the log of a machine's connect on this computer",
        description=(
            "Print the log the connect service of a machine writes on this computer, "
            "or follow it. A connect in the foreground prints to its terminal "
            "instead, and a machine connected elsewhere keeps its log there."
        ),
    )
    logs.add_argument(
        "name",
        nargs="?",
        default="",
        help="the machine; default this computer's short host name",
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
        help="stop this computer's runs, then disconnect its machine on the server",
        description=(
            "Stop every detached run on this computer, then disconnect the machine on "
            "the server: its connect is revoked, and every run of yours that names it "
            "is stopped, here or anywhere. Its connect service on this computer is "
            "removed, and it waits until the server shows the machine offline. "
            "knowop connect joins it again, with no login; knowop logout also "
            "revokes this computer's key."
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
                "server's assistant sends. At a terminal, join here, then serve it "
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
            _log_out(flags.name, every=flags.all)
        case "status":
            _status(as_json=flags.json)
        case "logs":
            _logs(flags.name, follow=flags.follow, lines=flags.lines, sleep=sleep)
        case "disconnect":
            _disconnect(flags.name, sleep=sleep)
        case "menubar":
            if sys.platform != "darwin":
                raise ClientError("knowop menubar runs on macOS alone")
            menubar.run()
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


# Logging this computer out also revokes knowop's key, deletes its profile and removes
# this computer's connect services of that server, none of which could start again
# without the login.
def _log_out(name: str, *, every: bool) -> None:
    """Log ``name`` out; else this computer, and with ``every`` all of yours, first."""
    profile_name, profile = _profile()
    services = service.here()
    with Client(profile.url, api_key=profile.api_key) as http:
        if name:
            check_name(name)
            _log_out_machine(http, name)
            _remove_service(services, name, profile.url)
            return
        if (caller := _caller(http)) is not None:
            key = from_plain(caller.get("api_key_id"), uuid.UUID)
            email = from_plain(caller.get("email"), str)
            machines = [host_name()]
            if every:
                # The machine a login made this key for goes last: its logout revokes
                # the key the others' sign with.
                made_for = caller.get("machine")
                machines = sorted(
                    (m.name for m in http.list_machines() if m.owner == email),
                    key=lambda machine: machine == made_for,
                )
            for machine in machines:
                # A machine of another account by this computer's name, or none, is
                # not this one's: it reads as no machine, and the key alone is revoked.
                try:
                    _log_out_machine(http, machine)
                except ClientError as error:
                    if error.status_code != 404:
                        raise
            # Logging a machine out revokes the keys its logins made, this one among them.
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
    _keep_logins([each for each, _ in _logins()])
    revoked = "revoked its key" if caller is not None else "its key was revoked already"
    echo(f"logged out: {revoked}; deleted profile {profile_name}")
    if (left := _saved()) is not None:
        echo(f"knowop now uses: profile {left[0]} ({left[1].url})")


def _log_out_machine(http: Client, machine: str) -> None:
    """Log ``machine`` out on the server, and say what stopped."""
    out = http.logout_machine(machine)
    runs = ", ".join(run.run or str(run.session_id) for run in out.runs) or "none"
    echo(
        f"logged out: machine {machine}; runs stopped: {runs}; keys revoked: {out.keys}",
    )


def _remove_service(
    services: service.Services | None,
    name: str,
    server: str,
) -> None:
    """Remove the connect service of ``name`` on ``server`` from here, if it is here."""
    if services is not None and services.remove(name, server):
        echo(f"removed: the connect service of machine {name} on this computer")


@dataclass(frozen=True, slots=True, kw_only=True)
class _Run:
    """A live run, as ``knowop status`` shows it."""

    run: str | None
    """Its detached host's name; ``None`` for a run in a terminal of its own."""

    cli: str
    cwd: str | None
    session_id: str | None
    """The session it drains; ``None`` for a run here the server places nowhere."""


@dataclass(frozen=True, slots=True, kw_only=True)
class _Machine:
    """One of the account's machines, as ``knowop status`` shows it."""

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
    machines: list[_Machine] | None
    """``None`` when the key was refused, or the role may not list machines."""


@dataclass(frozen=True, slots=True, kw_only=True)
class _Status:
    """What ``knowop status`` shows, and ``--json`` writes."""

    logins: list[_Login]
    """Each server knowop is logged in to, the one its verbs use first."""

    here: list[_Run]
    """Live detached runs here that name no machine of the accounts'."""

    services: list[service.Connect]
    menubar: str | None
    """How the menu bar agent here is; ``None`` where none is installed."""


# A refused login is shown with the others, and then fails the command.
def _status(*, as_json: bool) -> None:
    """Print each of knowop's logins with its account's machines, and services here."""
    _profile()
    logins = [_login(name, profile) for name, profile in _logins()]
    services = service.here()
    # A run the server places on no machine of these accounts' still runs here.
    listed = {run.run for each in logins for m in each.machines or [] for run in m.runs}
    report = _Status(
        logins=logins,
        here=[
            _Run(
                run=host.name,
                cli=" ".join(host.command.split()[:1]),
                cwd=host.cwd,
                session_id=None,
            )
            for host in detach.live_hosts()
            if any(each.machines is not None for each in logins)
            and host.name not in listed
        ],
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
    """Return the login ``profile`` holds: its account, role and account's machines."""
    with Client(profile.url, api_key=profile.api_key) as http:
        if (caller := _caller(http)) is None:
            return _Login(
                profile=name,
                server=profile.url,
                account=None,
                role=None,
                machines=None,
            )
        email = from_plain(caller.get("email"), str)
        try:
            own = [m.name for m in http.list_machines() if m.owner == email]
        except ClientError as error:
            if error.status_code != 403:
                raise
            own = None
        return _Login(
            profile=name,
            server=profile.url,
            account=email,
            role=from_plain(caller.get("role"), str),
            machines=None if own is None else [_machine(http, each) for each in own],
        )


def _machine(http: Client, name: str) -> _Machine:
    """Return one of the account's machines, its facts while online, and its runs."""
    detail = http.get_machine(name)
    online = detail.status == "online"
    return _Machine(
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
            for run in http.machine_runs(name)
        ],
    )


def _say_status(report: _Status, *, manager: str | None) -> None:
    """Print ``report`` as lines; ``manager`` names how a disabled service turns on."""
    for index, login in enumerate(report.logins):
        if index:
            echo("")
        _say_login(login)
    for run in report.here:
        echo(f"here:    {run.run}  {run.cli}")
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
    """Print ``login``: its account, role and machines, once the server accepts it."""
    echo(f"profile: {login.profile}")
    echo(f"server:  {login.server}")
    if login.account is None:
        return
    echo(f"account: {login.account}")
    echo(f"role:    {login.role}")
    if login.machines is None:
        echo("machine: listing machines needs the writer role")
    elif not login.machines:
        echo("machine: none; run knowop connect")
    for machine in login.machines or []:
        version = machine.knowop_version
        served = (f", knowop {version}" if version is not None else "") + (
            f", pid {machine.pid}" if machine.pid is not None else ""
        )
        live = len(machine.runs)
        count = f", {live} run{'' if live == 1 else 's'} live" if live else ""
        mark = " (this computer)" if machine.this_computer else ""
        echo(f"machine: {machine.name} {machine.status}{served}{count}{mark}")
        for run in machine.runs:
            short = (run.session_id or "")[:8]
            echo(f"  run:   {run.run or short}  {run.cli}  {run.cwd or ''}")


def _next_step() -> None:
    """Print the next setup step for this computer, without starting a login."""
    saved = _saved()
    if saved is None:
        echo("Not connected. Run:  knowop connect")
        return
    _, profile = saved
    with Client(profile.url, api_key=profile.api_key) as http:
        email = from_plain(_signed_in(http).get("email"), str)
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
# and a stop here would race its exit. Its service goes too, which would otherwise join
# the machine again when it next starts.
def _disconnect(name: str, *, sleep: Callable[[float], None]) -> None:
    """Stop this computer's runs when it is the machine, then disconnect the machine."""
    machine = name or host_name()
    check_name(machine)
    _, profile = _profile()
    if machine == host_name() and detach.stop_all() != 0:
        raise ClientError("a run here did not stop; see knowop run ls")
    with Client(profile.url, api_key=profile.api_key) as http:
        for run in http.disconnect_machine(machine):
            echo(f"stopped: run {run.run or run.session_id} ({run.cli})")
        _remove_service(service.here(), machine, profile.url)
        if _await(http, machine, lambda d: d.status != "online", sleep=sleep) is None:
            raise ClientError(f"machine {machine} still reads online after 30 s")
    echo(
        f"disconnected: machine {machine}; its connect stops at its next heartbeat, "
        "and knowop connect joins it again",
    )


# At a terminal a user service serves it; elsewhere, or with ``--foreground``, this
# process does, until it is released.
def _connect(
    flags: Flags,
    *,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> None:
    """Log in unless knowop already is, to that server; then serve this computer."""
    name = flags.name or host_name()
    check_name(name)
    background = sys.stdout.isatty() and not (flags.foreground or flags.log_file)
    if background and os.geteuid() == 0:
        raise ClientError(
            "knowop connect would install a service of root, which serves root's "
            "shells; run it as your own user, or with --foreground",
        )
    saved = _saved()
    server = (
        saved[1].url
        if saved is not None and not flags.target
        else _server(flags.target)
    )
    # A saved login the server now refuses (its machine was logged out elsewhere) is
    # no login: at a terminal connect logs in again, and a service's run says so.
    found = next(
        (each for each in _logins() if each[1].url == server.rstrip("/")),
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
    # A box has no browser to approve a login in: the token piped to it joins alone,
    # and the server it names is checked as a login's would be.
    elif piped := serve.piped_token():
        url = login.secure_url(server_url(server, "server"), what="server")
        _serve(name, url, profile="", token=piped, flags=flags, sleep=sleep)
        return
    else:
        # The approval that logs in also enrolls and confirms this computer, and returns
        # its one-use token: one step for the person, and no second login.
        result = login.login(
            server,
            machine=name,
            client="knowop",
            sleep=sleep,
            clock=clock,
        )
        _remember(result.profile)
        echo(f"logged in: profile {result.profile} ({result.role})")
        profile, url = result.profile, read_profile(result.profile).url
        token = result.machine_token
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
            "this terminal serves the machine until Ctrl-C",
            err=True,
        )
    _serve(name, url, profile=profile, token=token, flags=flags, sleep=sleep)


# Joined here first, so what a service could only log (another account's machine,
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
                f"connected already: machine {name} at {url}, by knowop pid {pid}, "
                "which no service of this computer runs; stop that connect, then run "
                "knowop connect again to serve it from one",
            )
            return
        # Restarted when its definition, its knowop or the CLIs it would find changed
        # (a host looks for them as it starts); else left running.
        current = (
            pid is not None
            and services.unchanged(unit)
            and before is not None
            and before.facts.get("knowop_version") == service.knowop_version()
            and before.facts.get("clis") == machine_host.host_facts()["clis"]
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
            started = _await(
                http,
                name,
                # Not this process: the join here heartbeats once, and reads online.
                lambda d: (
                    d.status == "online" and _pid(d) not in {None, pid, os.getpid()}
                ),
                sleep=sleep,
            )
            if started is None:
                raise ClientError(
                    f"machine {name} did not come online in 30 s; see knowop logs"
                    f"{_named(name)}",
                )
            pid = _pid(started)
    echo(
        f"connected{' already' if current else ''}: machine {name} at {url} in the "
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
# SIGHUP; either must release the machine as Ctrl-C does, or it reads online for three
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
    except ClientError as error:
        status = error.status_code or 0
        # A service's own run exits 0 on a refusal no restart can change (a revoked
        # login, another account's machine), so its manager starts nothing again;
        # 408 and 429 ask to try again later.
        if flags.log_file and 400 <= status < 500 and status not in {408, 429}:
            echo(f"knowop: {error}", err=True)
            echo("knowop: stopped; knowop connect starts it again", err=True)
            return
        raise
    except KeyboardInterrupt:
        _say_runs_left()
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
    follow: bool,
    lines: int | None,
    sleep: Callable[[float], None],
) -> None:
    """Print the log of ``name``'s connect service here; with ``follow``, follow it."""
    machine = name or host_name()
    check_name(machine)
    _, profile = _profile()
    log = (
        state_dir()
        / "rekursiv-ai"
        / "knowop"
        / "connect"
        / f"{machine}-{service.server_tag(profile.url)}.log"
    )
    if not log.is_file():
        raise ClientError(
            f"no log of machine {machine} on this computer: its log is on the "
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
    if (
        len(args) == 2
        and args[0] == "stop"
        and not args[1].startswith("-")
        and args[1] not in {host.name for host in detach.live_hosts()}
    ):
        _stop_run(args[1])
        return 0
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


# A run here stops through its own socket, as ever; any other live run of the account
# stops through the server, on whichever machine it says it runs.
def _stop_run(run: str) -> None:
    """Stop the live run of yours ``run`` names: its name, its session or its first 8."""
    _, profile = _profile()
    with Client(profile.url, api_key=profile.api_key) as http:
        email = from_plain(_signed_in(http).get("email"), str)
        found = [
            (machine.name, each)
            for machine in http.list_machines()
            if machine.owner == email
            for each in http.machine_runs(machine.name)
            if run in {each.run, str(each.session_id), str(each.session_id)[:8]}
        ]
        if len(found) > 1:
            listed = "".join(
                f"\n  {machine}  {each.run or '-'}  {each.session_id}  {each.cli}  "
                f"{each.cwd or ''}"
                for machine, each in found
            )
            raise ClientError(
                f"{run} names {len(found)} live runs; stop one by its session:{listed}",
            )
        if not found:
            raise ClientError(f"no live run {run} here or on your machines")
        [(machine, stopped)] = found
        http.stop_machine_run(machine, stopped.session_id)
    echo(
        f"stopped: run {stopped.run or stopped.session_id} ({stopped.cli}) on {machine}",
    )


def _detail(http: Client, name: str) -> MachineDetail | None:
    """Return the server's view of machine ``name``; ``None`` before it is enrolled."""
    try:
        return http.get_machine(name)
    except ClientError as error:
        if error.status_code != 404:
            raise
        return None


def _await(
    http: Client,
    name: str,
    until: Callable[[MachineDetail], bool],
    *,
    sleep: Callable[[float], None],
) -> MachineDetail | None:
    """Ask the server for ``name`` until ``until`` holds, for 30 s; ``None`` past it."""
    for _ in range(30):
        if until(detail := http.get_machine(name)):
            return detail
        sleep(1.0)
    return None


def _pid(detail: MachineDetail) -> int | None:
    """Return the process its facts say serves a machine; ``None`` from an old knowop."""
    return from_plain(detail.facts.get("pid"), int, default=None)


def _named(name: str) -> str:
    """Return a space and ``name``, or nothing for the machine verbs default to."""
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


def _profile() -> tuple[str, Profile]:
    """Return knowop's profile, or refuse when knowop is not logged in."""
    if (saved := _saved()) is None:
        raise ClientError("not logged in; run knowop login")
    return saved


def _saved() -> tuple[str, Profile] | None:
    """Return the name and contents of knowop's profile; ``None`` before a login."""
    return next(iter(_logins()), None)


# Knowop keeps a login per server it connected to, one profile name a line: the first is
# the one every verb uses, and a connect service of another server signs in with its own.
def _logins() -> list[tuple[str, Profile]]:
    """Return knowop's logins, the one its verbs use first; a deleted profile is none."""
    try:
        names = (
            (config_dir() / "rekursiv-ai" / "knowop" / "profile").read_text().split()
        )
    except FileNotFoundError:
        return []
    logins: list[tuple[str, Profile]] = []
    for name in dict.fromkeys(names):
        try:
            logins.append((name, read_profile(name)))
        except ClientError:
            continue
    return logins


def _remember(profile: str) -> None:
    """Make ``profile`` the one every knowop verb uses, keeping knowop's other logins."""
    _keep_logins([profile, *(name for name, _ in _logins() if name != profile)])


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
