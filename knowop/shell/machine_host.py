"""Host a machine: join it and keep it connected.

``knowop connect`` joins once with a one-use ``enr_`` token (read from stdin, never
argv, which any user on the host can read in a process listing, or returned by a
login) and saves the machine credential it receives, readable by this user only, under
the server it joined: one computer may be a machine of several servers. A personal
machine's credential is saved under its account too, so the account signed in decides
which machine a connect serves: another account's connect on the same computer never
finds it, and enrolls for itself, which the server refuses for a name another person
holds. It then heartbeats every 15 s until the server answers 410, which means the
machine was revoked (a disconnect, say): the host deletes the credential and stops,
and the next connect joins again. A credential saved by an earlier connect that the
server refuses before any heartbeat lands, revoked or unknown (401: the machine was
deleted since), is deleted the same way, and the connect joins again at once, as a
first one does: with the account's login, or the token on stdin. It joins again
once, never in a loop. Lost contact never stops it: a refused connection or a server
error is retried, sooner at first. Its first heartbeat takes the machine from any host
still holding it, so a restart starts serving at once; the host it replaced stops at
its next heartbeat. Ctrl-C releases the machine, which then reads offline at once.
When a heartbeat's answer says ``host_outdated``, one line says how to upgrade, and
the host keeps serving.

It uses the machine's own credential, never the profile's key, so it runs in the
calling process rather than in traxd.

A host serves terminals (``terminal_host.py``) and commands (``exec_host.py``) for as
long as it stays connected, and kills every shell and command when the machine is
revoked.

Each connect also records whether the person declared this computer a sandbox, a
disposable computer where agents may act freely (:func:`is_sandbox`). ``knowop run``
reads the record when it starts an agent.
"""

from __future__ import annotations

from contextlib import ExitStack, closing, suppress
from datetime import UTC, datetime
from functools import cache, partial
from importlib import metadata
from typing import TYPE_CHECKING, Final, Literal, Protocol

import getpass
import os
import platform
import shutil
import uuid

from trackinizer.client.errors import ClientError
from trackinizer.lib.userdirs import state_dir
from trackinizer.trax.connection import server_dir
from trackinizer.trax.render import echo
from trackinizer.wire.wire_machine_host import HEARTBEAT_SEC, JoinResponse
from trackinizer.wire.wire_machine_job import JOBS_FACT

from knowop.service import knowop_version


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from trackinizer.wire.wire_machine_host import Claim, HeartbeatResponse
    from trackinizer.wire.wire_machines import Facts, FactValue

    import pydantic
else:
    from wrapt import lazy_import

    # Only reading a corrupt credential file needs pydantic's error type.
    pydantic = lazy_import("pydantic")


__all__ = [
    "HOST_CLIS",
    "HostClient",
    "Serving",
    "connect",
    "host_facts",
    "is_sandbox",
    "set_sandbox",
]

HOST_CLIS: Final = ("claude", "codex", "sagent", "knowop", "gemini", "git", "uv")
"""The commands a host reports when it finds them on ``PATH``."""


class HostClient(Protocol):
    """The calls a host makes: the ``Client`` methods, and closing it."""

    def join_machine(
        self,
        name: str,
        *,
        token: str,
        instance: uuid.UUID,
        host_version: str,
        facts: Facts,
    ) -> JoinResponse:
        """Exchange the enrollment token for the machine credential."""
        ...

    def heartbeat_machine(
        self,
        machine_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        host_version: str,
        facts: Facts | None = None,
        claim: Claim = "keep",
    ) -> HeartbeatResponse:
        """Report that the host is alive, or that it lets go of the machine."""
        ...

    def close(self) -> None:
        """Release the connection."""
        ...


class Serving(Protocol):
    """Serves a joined machine's terminals, commands or jobs; started, stopped once."""

    def start(self, machine_id: uuid.UUID, instance: uuid.UUID) -> None:
        """Start listening for terminals to open."""
        ...

    def stop(self) -> None:
        """Stop listening and kill every shell."""
        ...


def connect(
    name: str,
    *,
    server: str,
    credential_client: Callable[[str], HostClient],
    read_token: Callable[[], str],
    sleep: Callable[[float], None],
    joined_terminals: Callable[[JoinResponse], Serving] | None = None,
    exec_for: Callable[[JoinResponse], Serving] | None = None,
    jobs_for: Callable[[JoinResponse], Serving] | None = None,
    job_facts: Mapping[str, FactValue] | None = None,
    sandbox: bool = False,
    account: str | None = None,
    join_only: bool = False,
) -> None:
    """Join ``name`` when no credential is saved, then heartbeat until it is revoked.

    A saved credential the server refuses before any heartbeat lands is replaced by
    one join. It also returns when a newer host took the machine over, and releases
    the machine on Ctrl-C before the interrupt goes on.

    Args:
      name: The machine's name, already checked against the name rules.
      server: The server's URL, which keeps its own credential for ``name``.
      credential_client: Returns a client that sends the given key; ``""`` sends none.
      read_token: Returns the enrollment token; called only to join.
      sleep: Waits the given seconds between heartbeats.
      joined_terminals: Makes the terminal server for the joined machine's credential;
        ``None`` serves none, whatever the server asks.
      exec_for: Makes the command server the same way; ``None`` serves none, and the
        facts do not say ``exec``.
      jobs_for: Makes the job host the same way; ``None`` runs no jobs, and the facts
        do not say ``jobs``.
      job_facts: What the job host offers jobs (its GPUs and memory), reported beside
        ``jobs: true`` when ``jobs_for`` is given.
      sandbox: Whether this connect declares the computer a sandbox. Every connect
        records its own answer once it holds a credential, so ``False`` withdraws
        an earlier declaration.
      account: The signed-in account a personal machine belongs to, whose own
        credential alone it uses; ``None`` for a machine an admin's token enrolls,
        whose credential belongs to the machine.
      join_only: Return once a credential is saved, joining only when none is,
        and serve nothing: the connect that serves it records the sandbox.

    Raises:
      ClientError: The token was refused, which after a refused saved credential
        names what joins the machine again; the credential became unknown to the
        server while serving; ``account`` is no email; or any other answer a retry
        cannot change.

    """
    path = _credential_path(name, server=server, account=account)
    instance = uuid.uuid4()
    version = _host_version()
    facts = host_facts(
        terminal=joined_terminals is not None,
        exec_commands=exec_for is not None,
        jobs=None if jobs_for is None else {} if job_facts is None else job_facts,
    )
    join = partial(
        _join,
        name,
        path=path,
        credential_client=credential_client,
        read_token=read_token,
        instance=instance,
        version=version,
        facts=facts,
    )
    host = partial(
        _host,
        name=name,
        server=server,
        credential_client=credential_client,
        instance=instance,
        version=version,
        sleep=sleep,
        makers=(joined_terminals, exec_for, jobs_for),
    )
    who = "" if account is None else f" as {account.lower()}"
    saved = _saved_credential(path)
    # Only a credential an earlier connect saved is joined again, and once: one this
    # connect joined was revoked since, and joining again would undo that, or loop.
    rejoin = saved is not None
    if saved is None:
        saved = join()
        echo(f"joined: machine {name}{who}")
    if join_only:
        return
    set_sandbox(on=sandbox)
    if sandbox:
        echo("sandbox: on (agents on this computer act without asking)")
    # The facts ride the first heartbeat only, and not after a join, which stored them.
    ending = host(saved, facts=facts if rejoin else None, rejoin=rejoin)
    if ending == "stale":
        _forget(path, refused=saved)
        saved = _rejoin(join, name=name, server=server, account=account)
        echo(f"rejoined: machine {name}{who}; the server refused its saved credential")
        ending = host(saved, facts=None, rejoin=False)
    if ending == "revoked":
        _forget(path, refused=saved)
        echo(f"revoked: machine {name}")


def host_facts(
    *,
    terminal: bool = False,
    exec_commands: bool = False,
    jobs: Mapping[str, FactValue] | None = None,
) -> Facts:
    """Return what this host reports about itself.

    Args:
      terminal: Whether this host serves terminals, and with them a ``files``
        terminal, which the facts then say.
      exec_commands: Whether this host runs commands the assistant sends, which the
        facts then say as ``exec``.
      jobs: What this host offers jobs, when it runs them; the facts then say
        ``jobs: true`` beside it.

    Returns:
      facts: Its OS and architecture, CPU count, memory in MB, the user it runs
        as, which of :data:`HOST_CLIS` are on ``PATH``, its process id, when it
        started and its knowop version, ``terminal: true`` and ``files: true`` when
        it serves terminals, ``exec: true`` when it runs commands, and ``jobs: true``
        with what it offers jobs when it runs them.

    """
    return {
        **({"terminal": True, "files": True} if terminal else {}),
        **({"exec": True} if exec_commands else {}),
        **({JOBS_FACT: True, **jobs} if jobs is not None else {}),
        "os": platform.system().lower(),
        "arch": platform.machine(),
        "cpus": os.cpu_count() or 0,
        "memory_mb": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2**20,
        "user": getpass.getuser(),
        "clis": [cli for cli in HOST_CLIS if shutil.which(cli) is not None],
        "pid": os.getpid(),
        "started_at": _started_at(),
        "knowop_version": knowop_version(),
    }


def is_sandbox() -> bool:
    """Return whether the last connect on this computer declared it a sandbox."""
    return (state_dir() / "rekursiv-ai" / "trax" / "machines" / ".sandbox").exists()


def set_sandbox(*, on: bool) -> None:
    """Record that this computer is a sandbox, or that it is not.

    The record belongs to the computer, not to a server or a machine name, so it sits
    in the directory that holds the per-server credential directories; the dot keeps
    its name from ever being a server's.

    Args:
      on: Whether agents on this computer may act freely.

    """
    path = state_dir() / "rekursiv-ai" / "trax" / "machines" / ".sandbox"
    if on:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.touch()
    else:
        path.unlink(missing_ok=True)


type _Ending = Literal["stale", "revoked", "taken over"]
"""Why a host stopped: its saved credential was refused, it was revoked, or replaced."""


def _host(
    joined: JoinResponse,
    *,
    name: str,
    server: str,
    credential_client: Callable[[str], HostClient],
    instance: uuid.UUID,
    version: str,
    facts: Facts | None,
    sleep: Callable[[float], None],
    makers: Sequence[Callable[[JoinResponse], Serving] | None],
    rejoin: bool,
) -> _Ending:
    """Serve with ``joined``'s credential until it ends; Ctrl-C releases the machine."""
    with (
        ExitStack() as serving,
        closing(credential_client(joined.credential)) as client,
    ):
        try:
            return _heartbeat(
                client,
                name=name,
                server=server,
                machine_id=joined.machine_id,
                instance=instance,
                version=version,
                facts=facts,
                sleep=sleep,
                serve=partial(
                    _serve,
                    serving,
                    makers,
                    joined=joined,
                    instance=instance,
                ),
                rejoin=rejoin,
            )
        except KeyboardInterrupt:
            # Once and best effort: a server out of reach leaves the machine to
            # read offline after 180 s of silence, as a crash does.
            with suppress(ClientError):
                client.heartbeat_machine(
                    joined.machine_id,
                    instance=instance,
                    host_version=version,
                    claim="release",
                )
            raise


# The first heartbeat that lands takes the machine, so a restart never waits out the
# host it replaced; every later one keeps it. The servers start only then: the server
# refuses a wait from an instance that does not hold the machine. A keep refused with
# 409 means a newer host took the machine, and this one stops rather than fight it for
# the machine. With ``rejoin``, a take refused with 410 or 401 means the saved
# credential is stale, which ``connect`` replaces; once serving, a 401 is an answer a
# retry cannot change, as for any credential.
def _heartbeat(
    client: HostClient,
    *,
    name: str,
    server: str,
    machine_id: uuid.UUID,
    instance: uuid.UUID,
    version: str,
    facts: Facts | None,
    sleep: Callable[[float], None],
    serve: Callable[[], None],
    rejoin: bool,
) -> _Ending:
    """Heartbeat until the machine is revoked or another host takes it over."""
    claim: Claim = "take"
    unsent = facts
    failures = 0
    told = False
    while True:
        try:
            beat = client.heartbeat_machine(
                machine_id,
                instance=instance,
                host_version=version,
                facts=unsent,
                claim=claim,
            )
        except ClientError as err:
            if rejoin and claim == "take" and err.status_code in {401, 410}:
                return "stale"
            if err.status_code == 410:
                return "revoked"
            if err.code == "another_host_connected":
                echo(f"taken over: machine {name} is served by a newer connect")
                return "taken over"
            failures += 1
            sleep(_retry_delay(err, failures=failures))
            continue
        if claim == "take":
            serve()
        # Said once: the server repeats it on every heartbeat, four times a minute.
        # Run at a terminal, the command also restarts a connect service, on the
        # version it fetched.
        if beat.host_outdated and not told:
            echo(
                f"knowop: this knowop is too old for {server}: "
                f"run uvx knowop@latest connect {server}",
                err=True,
            )
            told = True
        unsent = None
        claim = "keep"
        failures = 0
        sleep(HEARTBEAT_SEC)


def _serve(
    serving: ExitStack,
    makers: Sequence[Callable[[JoinResponse], Serving] | None],
    *,
    joined: JoinResponse,
    instance: uuid.UUID,
) -> None:
    """Start each server in order; ``serving`` stops those that started, in reverse."""
    for make in makers:
        if make is not None:
            served = make(joined)
            served.start(joined.machine_id, instance)
            serving.callback(served.stop)


# A transport failure carries no status. Anything in the 4xx range is an answer a
# retry cannot change. The exponent is capped: a host offline for hours counts past a
# thousand failures, and 2.0 ** 1024 raises OverflowError.
def _retry_delay(err: ClientError, *, failures: int) -> float:
    """Return how long to wait before the next try, or re-raise ``err``."""
    if err.status_code is not None and 400 <= err.status_code < 500:
        raise err
    echo(f"knowop: retrying after failure {failures}: {err}", err=True)
    return min(2.0 ** min(failures, 4), float(HEARTBEAT_SEC))


def _join(
    name: str,
    *,
    path: Path,
    credential_client: Callable[[str], HostClient],
    read_token: Callable[[], str],
    instance: uuid.UUID,
    version: str,
    facts: Facts,
) -> JoinResponse:
    """Join ``name`` with the token ``read_token`` returns, and save the credential."""
    token = read_token().strip()
    if not token:
        raise ClientError("connect reads the enrollment token from stdin; none came")
    with closing(credential_client("")) as joining:
        joined = joining.join_machine(
            name,
            token=token,
            instance=instance,
            host_version=version,
            facts=facts,
        )
    _save_credential(path, joined=joined)
    return joined


# A 401 is the account's key refused by the enrollment, or a token used or expired
# refused by the join; any other failure is said as the server said it.
def _rejoin(
    join: Callable[[], JoinResponse],
    *,
    name: str,
    server: str,
    account: str | None,
) -> JoinResponse:
    """Join ``name`` in place of its refused credential, or say what joins it."""
    try:
        return join()
    except ClientError as err:
        if err.status_code != 401:
            raise
        hint = (
            "pipe a new enrollment token to knowop connect"
            if account is None
            else "run knowop login, then knowop connect"
        )
        raise ClientError(
            f"machine {name} could not join {server} again; {hint}",
            status_code=401,
        ) from err


def _host_version() -> str:
    """Return the installed ``trackinizer`` version; empty from a source checkout."""
    try:
        return metadata.version("trackinizer")
    except metadata.PackageNotFoundError:
        return ""


# Cached: a join and every heartbeat after it name the same start.
@cache
def _started_at() -> str:
    """Return when this host first reported itself, in UTC, to the second."""
    return datetime.now(UTC).isoformat(timespec="seconds")


# An email is the account's whole identity; one that is not one, or that would leave
# the server's directory, is refused before anything is read or written.
def _credential_path(name: str, *, server: str, account: str | None) -> Path:
    """Return where ``name``'s credential from ``server`` is kept for ``account``."""
    machines = state_dir() / "rekursiv-ai" / "trax" / "machines" / server_dir(server)
    if account is None:
        return machines / f"{name}.credential"
    if "@" not in account or "/" in account or account.startswith("."):
        raise ClientError(f"{account!r} is not an account a credential can be kept for")
    return machines / account.lower() / f"{name}.credential"


def _saved_credential(path: Path) -> JoinResponse | None:
    """Return the credential saved at ``path``, or ``None`` when none is saved."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    try:
        return JoinResponse.model_validate_json(text)
    except pydantic.ValidationError:
        # Not chained: pydantic's message would quote the secret.
        raise ClientError(
            f"{path} does not hold a machine credential; remove it to join again",
        ) from None


# Created with mode 0600 in a 0700 directory before any byte lands, so the secret is
# never readable by another user; ``O_EXCL`` refuses to overwrite one saved meanwhile.
def _save_credential(path: Path, *, joined: JoinResponse) -> None:
    """Save the credential the join returned at ``path``, readable by this user only."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as file:
        file.write(joined.model_dump_json())
        file.flush()
        os.fsync(file.fileno())


# Another connect of the machine may have joined and saved its own credential here
# since; only the one the server refused is deleted.
def _forget(path: Path, *, refused: JoinResponse) -> None:
    """Delete the credential saved at ``path`` while it is still ``refused``."""
    if _saved_credential(path) == refused:
        path.unlink(missing_ok=True)
