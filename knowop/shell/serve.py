"""Connect this computer: get an enrollment token, join, and serve until revoked.

The enrollment token is the one the caller holds when it hands one in (a login it ran
already enrolled the machine), else comes from stdin when one is piped in (a box that
has no browser), else from the signed-in profile of the server, else from a login: the
code a person approves in a browser returns a token for this very machine, so
approving is the only step. The account signed in, by the profile's key or by the
login's approval, decides which machine this computer serves: a credential is used
only by the account it was saved for, which then asks for no token. A piped token is
an admin's for the machine and names no person, so its credential is the machine's
own.

Once joined the host serves terminals (``terminal_host.py``) and commands
(``exec_host.py``), in the foreground, until the machine is revoked, a newer connect
of it takes over, or Ctrl-C releases it.

With ``jobs`` it also runs the jobs its account submits with ``knowop job submit
--machine`` (``job_host.py``). That needs a memory cap systemd enforces, which is
proven before anything else, and a signed-in account: a computer joined by a piped
token belongs to no person and runs no one's jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TextIO
from urllib.parse import urlsplit

import functools
import select
import sys
import time

from trackinizer.client.client import Client
from trackinizer.client.errors import ClientError
from trackinizer.lib.userdirs import data_dir
from trackinizer.trax.connection import server_dir
from trackinizer.trax.context import env
from trackinizer.trax.profile import profile_for_server, read_profile
from trackinizer.trax.render import echo
from treekle import from_plain


if TYPE_CHECKING:
    from collections.abc import Callable

    from trackinizer.trax import login
    from trackinizer.wire import wire_machine_job
    from trackinizer.wire.wire_machine_host import JoinResponse
    from trackinizer.wire.wire_machines import Facts

    from knowop.shell import (
        exec_host,
        job_host,
        job_limits,
        job_repo,
        machine_host,
        terminal_direct_aiortc,
        terminal_host,
    )
    from knowop.shell.terminal_direct import PeerFactory
    from knowop.shell.terminal_host import TerminalHost
else:
    from wrapt import lazy_import

    # The host loop, the terminal and exec servers and the login poll bring pydantic,
    # websockets and the wire models; ``knowop status`` needs none of them.
    exec_host = lazy_import("knowop.shell.exec_host")
    job_host = lazy_import("knowop.shell.job_host")
    job_limits = lazy_import("knowop.shell.job_limits")
    job_repo = lazy_import("knowop.shell.job_repo")
    machine_host = lazy_import("knowop.shell.machine_host")
    terminal_direct_aiortc = lazy_import(
        "knowop.shell.terminal_direct_aiortc",
    )
    terminal_host = lazy_import("knowop.shell.terminal_host")
    login = lazy_import("trackinizer.trax.login")
    wire_machine_job = lazy_import("trackinizer.wire.wire_machine_job")


__all__ = ["connect_machine", "piped_token"]

_STDIN_WAIT_SEC: Final = 3.0
"""How long a piped stdin may stay silent before it is taken to hold no token."""

_PROFILE_PATH: Final = "/api/me/profile"
"""The signed-in account's profile, whose ``email`` names it."""


def connect_machine(
    name: str,
    *,
    server: str,
    signed_in: Callable[[], Client] | None,
    sandbox: bool,
    exec_commands: bool,
    sleep: Callable[[float], None] = time.sleep,
    token: str = "",
    jobs: bool = False,
    results_max: int = 256 * 2**20,
) -> None:
    """Join ``name`` on ``server`` and serve it until it is revoked or interrupted.

    Ctrl-C releases the machine, which then reads offline at once, and the interrupt
    goes on to the caller.

    Args:
      name: The machine's name, already checked against the name rules.
      server: The server's normalized URL, which keeps its own credential for ``name``.
      signed_in: Returns a client that holds a key for ``server``, or ``None`` when
        no saved profile does.
      sandbox: Whether this connect declares the computer a disposable sandbox.
      exec_commands: Whether the host runs the commands the assistant sends.
      sleep: Waits the given seconds between polls and heartbeats.
      token: A one-use enrollment token for ``name`` the caller already holds, as a
        login that enrolled the machine returns; used before any other.
      jobs: Whether the host runs the jobs its account submits to it.
      results_max: With ``jobs``, the most result bytes one job sends back.

    Raises:
      ClientError: The login was denied or expired, the token was refused, the
        server answered something a retry cannot change, or ``jobs`` was asked for
        where no memory cap holds or no account is signed in.

    """
    offer = _job_offer() if jobs else None
    direct = _direct_link()
    echo(
        f"serving: machine {name} at {server} "
        f"({_what_is_served(exec_commands, jobs=jobs)})",
    )
    account, read_token = _token_source(
        name,
        server=server,
        signed_in=signed_in,
        sleep=sleep,
        token=token,
    )
    jobs_for = None
    if offer is not None:
        if account is None:
            raise ClientError(
                "--jobs runs the jobs of a signed-in account; a computer joined by a "
                "piped token runs none",
            )
        jobs_for = functools.partial(
            _job_host,
            name,
            server,
            account=account,
            offer=offer,
            results_max=results_max,
        )
    machine_host.connect(
        name,
        server=server,
        client_for=lambda key: Client(server, api_key=key),
        read_token=read_token,
        sleep=sleep,
        terminals_for=lambda joined: _terminal_host(
            name,
            server,
            joined,
            direct=direct,
        ),
        exec_for=(
            (lambda joined: _exec_host(name, server, joined)) if exec_commands else None
        ),
        jobs_for=jobs_for,
        job_facts=None if offer is None else offer.facts,
        sandbox=sandbox,
        account=account,
    )


# A token is read from stdin only when it is piped in: a person at a terminal has a
# browser to approve a login in, and reading would wait on a keyboard.
def piped_token() -> str:
    """Return the first line piped to stdin, or empty when none is there to read.

    A pipe that stays silent for a few seconds holds no token.

    Returns:
      token: The line, stripped of blanks; empty when none was piped.

    """
    stdin = sys.stdin
    # No stdin at all (``<&-``), or a person at a terminal.
    if stdin is None or stdin.isatty():
        return ""
    return str(stdin.readline()).strip() if _speaks_soon(stdin) else ""


# A launcher may leave a pipe open that nothing writes to: an agent shell, a CI step,
# ``ssh`` without ``-t``. Waiting on it would hang before the login code is ever shown.
def _speaks_soon(stdin: TextIO) -> bool:
    """Return whether stdin has data, or its end, within a few seconds."""
    try:
        descriptor = stdin.fileno()
    except (OSError, ValueError):
        # An in-memory stream has nothing to wait for.
        return True
    poller = select.poll()
    poller.register(descriptor, select.POLLIN)
    # Data, or the end of the stream: a hang-up is as good as a line.
    return bool(poller.poll(_STDIN_WAIT_SEC * 1000))


def _terminal_host(
    name: str,
    url: str,
    joined: JoinResponse,
    *,
    direct: PeerFactory | None = None,
) -> TerminalHost:
    """Make the terminal server for a joined machine; its shells name the server."""
    # A URL with credentials in it would be sent to every ssh server by ``SendEnv LC_*``.
    trax_env = {} if "@" in urlsplit(url).netloc else {"LC_TRAX_SERVER": url}
    if (profile := profile_for_server(url)) is not None:
        trax_env["TRACKINIZER_PROFILE"] = profile
    return terminal_host.TerminalHost(
        name=name,
        server_url=url,
        credential=joined.credential,
        client_for=lambda key: Client(url, api_key=key),
        trax_env=trax_env,
        direct=direct,
    )


def _exec_host(name: str, url: str, joined: JoinResponse) -> exec_host.ExecHost:
    """Make the command server for a joined machine."""
    return exec_host.ExecHost(
        name=name,
        credential=joined.credential,
        client_for=lambda key: Client(url, api_key=key),
    )


def _direct_link() -> PeerFactory | None:
    """Return the peer factory for a direct terminal link, and say which way it is."""
    if env("TRAX_MACHINE_TERMINAL_DIRECT") == "0":
        echo("direct terminal link: off (TRAX_MACHINE_TERMINAL_DIRECT=0)")
        return None
    factory = terminal_direct_aiortc.aiortc_peer_factory()
    if factory is None:
        echo(
            "direct terminal link: off (aiortc is unavailable). "
            "Terminals still work through the relay.",
        )
    else:
        echo("direct terminal link: on")
    return factory


def _what_is_served(exec_commands: bool, *, jobs: bool) -> str:
    """Return the words that say what this host serves."""
    served = [
        "terminals",
        *(["commands"] if exec_commands else []),
        *(["jobs"] if jobs else []),
    ]
    if len(served) == 1:
        return served[0]
    return f"{', '.join(served[:-1])} and {served[-1]}"


@dataclass(frozen=True, slots=True, kw_only=True)
class _JobOffer:
    """What this computer offers jobs, as its cap probe and GPUs found it."""

    gpus: tuple[int, ...]
    memory_mb: int

    @property
    def facts(self) -> Facts:
        """The facts the host reports beside ``jobs: true``."""
        return {
            wire_machine_job.GPUS_FACT: len(self.gpus),
            wire_machine_job.JOBS_MEMORY_FACT: self.memory_mb,
        }


# A GPU that shares the CPU's memory is named, so its owner knows why a GPU job is
# refused here (see ``job_limits``).
def _job_offer() -> _JobOffer:
    """Prove the memory cap and find the GPUs a job may be given, or refuse ``--jobs``."""
    problem = job_limits.cap_problem()
    if problem is not None:
        raise ClientError(
            f"--jobs needs systemd's user manager to cap a job's memory, and {problem}",
        )
    found = job_limits.gpus()
    for gpu in found:
        if gpu.unified:
            echo(
                f"jobs: GPU {gpu.index} ({gpu.name}) shares the CPU's memory, which "
                "a memory cap does not bound, so no job is given it",
            )
    offer = _JobOffer(
        gpus=tuple(gpu.index for gpu in found if not gpu.unified),
        memory_mb=job_limits.memory_budget_mb(),
    )
    echo(
        f"jobs: on, up to {offer.memory_mb} MiB together, GPUs: "
        f"{','.join(map(str, offer.gpus)) or 'none'}",
    )
    return offer


def _job_host(
    name: str,
    url: str,
    joined: JoinResponse,
    *,
    account: str,
    offer: _JobOffer,
    results_max: int,
) -> job_host.JobHost:
    """Make the job host for a joined machine."""
    jobs = data_dir() / "rekursiv-ai" / "knowop" / "jobs"
    return job_host.JobHost(
        name=name,
        server_url=url,
        credential=joined.credential,
        client_for=lambda key: Client(url, api_key=key),
        account=account,
        root=jobs / server_dir(url),
        repos=job_repo.Repos(jobs / "repos"),
        launcher=job_host.SystemdLauncher(),
        gpus=offer.gpus,
        idle_gpus=job_limits.idle_gpus,
        memory_budget_mb=offer.memory_mb,
        results_max=results_max,
    )


def _token_source(
    name: str,
    *,
    server: str,
    signed_in: Callable[[], Client] | None,
    sleep: Callable[[float], None],
    token: str,
) -> tuple[str | None, Callable[[], str]]:
    """Return the account ``name`` is served for, and how to get the token joining it."""
    if token:
        # The caller's own login returned it, so it belongs to the account signed in.
        return (None if signed_in is None else _account_of(signed_in())), lambda: token
    if piped := piped_token():
        return None, lambda: piped
    return _signed_in_as(name, server=server, signed_in=signed_in, sleep=sleep)


# The server, asked with the key this computer holds, names the account, whether the
# key was saved or the login just made it. The enrollment waits until a join needs it:
# a credential already saved for the account needs no token, and an unused one would
# stay open on the server.
def _signed_in_as(
    name: str,
    *,
    server: str,
    signed_in: Callable[[], Client] | None,
    sleep: Callable[[float], None],
) -> tuple[str, Callable[[], str]]:
    """Return the signed-in account, logging in when no key is saved, and its token."""
    if signed_in is not None:
        client = signed_in()
        account = _account_of(client)
        return account, functools.partial(_enroll, client, name, account=account)
    result = login.login(server, machine=name, client="knowop", sleep=sleep)
    with Client(server, api_key=read_profile(result.profile).api_key) as client:
        return _account_of(client), lambda: result.machine_token


def _account_of(client: Client) -> str:
    """Return the email of the account ``client``'s key signs in as."""
    profile = from_plain(client.get(_PROFILE_PATH), dict[str, object])
    return from_plain(profile.get("email"), str)


def _enroll(client: Client, name: str, *, account: str) -> str:
    """Return a one-use token enrolling ``name`` for ``account``, or say who refused."""
    try:
        return client.enroll_machine(name).token
    except ClientError as err:
        if err.status_code != 403:
            raise
        # The server names no owner: a personal machine is seen by its owner alone.
        raise ClientError(
            f"machine {name} on {client.base_url} belongs to another account, not to "
            f"{account}, who is signed in; connect under another name with "
            f"`knowop connect --name NAME`, or ask an admin to free it with "
            f"`knowop machine {name} del`",
        ) from err
