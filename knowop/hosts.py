"""The ``knowop host`` command: your hosts, where campaigns may run, and KnowHosts.

The registry records the hosts campaigns may run on; listing and showing need the
writer role, and changing the registry needs the admin role. A KnowHost, a host the
server rents you, is one more host here: ``host start`` rents one, ``host NAME pause``
and ``resume`` stop and wake it, and ``host NAME del`` ends it, through the server's
KnowHost API (``/api/addons/knowhost``). ``knowop connect`` names this computer by
:func:`host_name`: the name :func:`keep_host_name` kept for the server, else
:func:`default_host_name`. It checks a name it is given with :func:`check_name`; a
hostname recorded elsewhere reads as a host name through :func:`short_name`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import (
    TYPE_CHECKING,
    Final,
    NotRequired,
    Protocol,
    TypedDict,
    cast,
    override,
)

import argparse
import getpass
import re
import sys

from trackinizer.client.errors import ClientError
from trackinizer.lib.posix import hostname
from trackinizer.lib.userdirs import config_dir
from trackinizer.trax.commands import Command, HelpPage, TraxArgumentParser
from trackinizer.trax.connection import server_dir
from trackinizer.trax.context import cwd
from trackinizer.trax.render import echo
from treekle import from_plain


if TYPE_CHECKING:
    from trackinizer.client.client import Client
    from trackinizer.wire import wire_hosts
    from trackinizer.wire.wire_hosts import Host
    from treekle import PlainTree
else:
    from wrapt import lazy_import

    # Only ``knowop host`` touches the host models, so their pydantic build stays
    # off the cold-start path of every other verb.
    wire_hosts = lazy_import("trackinizer.wire.wire_hosts")


__all__ = [
    "HOST_WORD",
    "Hosts",
    "KnowHost",
    "check_name",
    "default_host_name",
    "host_name",
    "keep_host_name",
    "listed_knowhosts",
    "named_knowhost",
    "pause_knowhost",
    "paused_line",
    "resume_knowhost",
    "resumed_lines",
    "short_name",
    "start_knowhost",
    "started_lines",
    "suggested_host_name",
    "taken_name_error",
]

HOST_WORD: Final = "host"
"""The verb that selects this command, in matching, help, usage and confirmations."""

PROGRAM: Final = f"knowop {HOST_WORD}"
"""The command's name, in help, usage and messages."""

_FORMS: Final = "[start | NAME [pause | resume | del | FIELD [to VALUE|-|@FILE] | label add|del LABEL]]"

_USAGE: Final = f"usage: {PROGRAM} {_FORMS}"

_FIELDS: Final = ("role", "how")

_PAUSE_NOTE: Final = (
    "A terminal open across a pause ends; jobs started with nohup or in tmux, and "
    "tool calls, carry on."
)
"""What a pause keeps and ends, said wherever a KnowHost pauses."""

_KNOWHOSTS: Final = "/api/addons/knowhost/hosts"

_WAIT_SEC: Final = 150.0
"""How long a start or resume is waited on: past the server's own 90 s deadline."""


class Hosts(Command):
    """List, show, set and delete hosts; rent, pause, wake and end KnowHosts."""

    names = (HOST_WORD,)

    help = HelpPage(
        usage=f"{PROGRAM} {_FORMS}",
        summary=(
            "Bare 'host' lists the registered hosts, then your KnowHosts; 'host NAME' "
            "shows one. Both need the writer role. Setting, labelling and deleting "
            "need the admin role. 'host start' rents a KnowHost, billed to your "
            "account, that joins as one of your hosts; it pauses after 30 minutes "
            "with no tool call and no terminal use, and costs nothing while paused. "
            "'knowop connect' connects this computer."
        ),
        arguments=(
            ("NAME", "lowercase letters, digits and '-', at most 63 characters"),
            ("role", "what the host is for; to '' clears it"),
            ("how", "one line telling an agent how to use the host; to '' clears"),
            ("VALUE", "text; '-' reads stdin, '@FILE' reads a file"),
            ("start", "rent a KnowHost; prints its host name once it is online"),
            ("pause", "pause your KnowHost NAME now; charging stops"),
            ("resume", "wake your KnowHost NAME, once it is online again"),
        ),
        examples=(
            f"{PROGRAM}                           list hosts",
            f"{PROGRAM} NAME                      show one host",
            f"{PROGRAM} NAME role                 print its role",
            f"{PROGRAM} NAME role to ROLE         set the role (creates)",
            f"{PROGRAM} NAME how to TEXT          set the how line",
            f"{PROGRAM} NAME how to -             how line from stdin",
            f"{PROGRAM} NAME how to @FILE         how line from a file",
            f"{PROGRAM} NAME label add LABEL      add a label",
            f"{PROGRAM} NAME label del LABEL      remove a label",
            f"{PROGRAM} NAME del                  unregister the host, or end a KnowHost",
            f"{PROGRAM} start                     rent a KnowHost",
            f"{PROGRAM} NAME pause                pause your KnowHost",
            f"{PROGRAM} NAME resume               wake your KnowHost",
        ),
        notes=(
            "A field left out of a set keeps its value; to '' clears one.",
            (
                "One trailing newline is stripped from stdin and file values; an "
                "empty one is refused."
            ),
            (
                "A last word of help, -h or --help shows this page: set such a value "
                "with @FILE; such a host name or label is out of reach here."
            ),
            _PAUSE_NOTE,
        ),
    )

    @classmethod
    @override
    def make_parser(cls) -> argparse.ArgumentParser:
        parser = TraxArgumentParser(
            prog=PROGRAM,
            description="List, show, set or delete the registered hosts.",
        )
        # REMAINDER, not "*": a value such as ``--verbose`` must reach ``run``
        # instead of failing as an unknown option.
        parser.add_argument("rest", nargs=argparse.REMAINDER, metavar="POS")
        return parser

    @classmethod
    @override
    def run(
        cls,
        verb: str,
        args: argparse.Namespace,
        client_factory: Callable[[], Client],
    ) -> None:
        del verb
        tokens = list(cast(Sequence[str], args.rest))
        words = [token.lower() for token in tokens]
        if not tokens:
            run_list(client_factory)
            return
        if words == ["start"]:
            for line in started_lines(start_knowhost(client_factory())):
                echo(line)
            return
        name = tokens[0]
        check_name(name)
        match words[1:]:
            case []:
                run_show(client_factory, name)
            case ["del"]:
                run_del(client_factory, name)
            case ["pause" | "resume" as action]:
                http = client_factory()
                knowhost = named_knowhost(http, name)
                if action == "pause":
                    paused = pause_knowhost(http, knowhost)
                    echo(paused_line(paused))
                else:
                    for line in resumed_lines(resume_knowhost(http, knowhost)):
                        echo(line)
            case [field] if field in _FIELDS:
                run_field(client_factory, name, field)
            case [field, "to", _] if field in _FIELDS:
                run_set(client_factory, name, field=field, source=tokens[3])
            case ["label", "add" | "del" as op, _]:
                run_label(client_factory, name, op=op, label=tokens[3])
            case _:
                raise ClientError(_USAGE)


def run_list(client_factory: Callable[[], Client]) -> None:
    """Print one line per host (name, role, labels, how), then one per KnowHost.

    Args:
      client_factory: Callable returning an authenticated trax Client.

    """
    http = client_factory()
    try:
        hosts = http.list_hosts()
    except ClientError as err:
        raise _refusal(err, role="writer") from err
    if not hosts:
        echo("(no hosts)")
    cells = [(m.name, m.role or "-", _labels(m) or "-", _cut(m.how)) for m in hosts]
    widths = [max((len(row[col]) for row in cells), default=0) for col in range(3)]
    for *head, how in cells:
        padded = (cell.ljust(width) for cell, width in zip(head, widths, strict=True))
        echo("  ".join((*padded, how)))
    rented = listed_knowhosts(http)
    if rented is not None and len(rented) > 0:
        echo("")
        echo("KnowHosts:")
    for each in rented or ():
        echo(f"{each['host']}  {each['state'].ljust(8)}  {each['id']}")


def run_show(client_factory: Callable[[], Client], name: str) -> None:
    """Print every field of one host.

    Args:
      client_factory: Callable returning an authenticated trax Client.
      name: Host name.

    """
    host = _get(client_factory, name)
    echo(f"host:       {host.name}")
    echo(f"role:       {host.role or '(none)'}")
    echo(f"labels:     {_labels(host) or '(none)'}")
    echo(f"how:        {_one_line(host.how) or '(none)'}")
    echo(f"status:     {host.status}")
    echo(f"updated by: {host.updated_by}")
    echo(f"updated:    {host.updated.isoformat()}")


def run_field(client_factory: Callable[[], Client], name: str, field: str) -> None:
    """Print one field's value exactly as stored.

    Args:
      client_factory: Callable returning an authenticated trax Client.
      name: Host name.
      field: ``role`` or ``how``.

    """
    host = _get(client_factory, name)
    echo(host.role if field == "role" else host.how)


def run_set(
    client_factory: Callable[[], Client],
    name: str,
    *,
    field: str,
    source: str,
) -> None:
    """Set ``field`` from ``source``, creating the host when it is new.

    Args:
      client_factory: Callable returning an authenticated trax Client.
      name: Host name.
      field: ``role`` or ``how``.
      source: Literal value, ``-``, or ``@FILE``.

    """
    value = _read_value(source)
    try:
        if field == "role":
            client_factory().put_host(name, role=value)
        else:
            client_factory().put_host(name, how=value)
    except ClientError as err:
        # No name: a set creates the host, so a 404 is a server without
        # the routes, not an absent host.
        raise _refusal(err, role="admin") from err
    echo(f"set: {HOST_WORD} {name} {field}")


def run_label(
    client_factory: Callable[[], Client],
    name: str,
    *,
    op: str,
    label: str,
) -> None:
    """Add or remove one label.

    Args:
      client_factory: Callable returning an authenticated trax Client.
      name: Host name.
      op: ``add`` or ``del``.
      label: The label.

    """
    try:
        if op == "add":
            client_factory().change_host_labels(name, add=[label])
        else:
            client_factory().change_host_labels(name, remove=[label])
    except ClientError as err:
        raise _refusal(err, role="admin", name=name) from err
    echo(f"set: {HOST_WORD} {name} label {op} {label}")


def run_del(client_factory: Callable[[], Client], name: str) -> None:
    """End a KnowHost of yours, VM and all; else unregister the host, touching nothing on it.

    Args:
      client_factory: Callable returning an authenticated trax Client.
      name: Host name.

    """
    http = client_factory()
    for each in listed_knowhosts(http) or ():
        if each["host"] == name:
            _end(http, each)
            return
    try:
        http.delete_host(name)
    except ClientError as err:
        raise _refusal(err, role="admin", name=name) from err
    echo(f"deleted: {HOST_WORD} {name}")


def host_name(server: str) -> str:
    """Return the name this computer is a host of ``server`` by, for this user.

    It is the name ``knowop connect --name`` last joined ``server`` under here
    (:func:`keep_host_name`), else :func:`default_host_name`. The command that
    connects this computer enrolls it under this name, an agent run says it runs on
    the host of that name, and every verb that means this computer defaults to it. A
    second person on a shared computer, whose hostname another person's host already
    holds, connects once under a name of their own and keeps it.

    Args:
      server: The server's URL.

    Returns:
      name: The host's name.

    Raises:
      ClientError: No name is kept and the hostname is reserved or still not a host
        name, or the kept one is not a host name.

    """
    kept = (
        config_dir()
        / "rekursiv-ai"
        / "knowop"
        / "hosts"
        / short_name(hostname.stable_hostname())
        / server_dir(server)
    )
    try:
        name = kept.read_text().strip()
    except FileNotFoundError:
        return default_host_name()
    check_name(name)
    return name


def keep_host_name(server: str, name: str) -> None:
    """Keep ``name`` as the name this computer is a host of ``server`` by, for this user.

    Args:
      server: The server's URL.
      name: A host name the user's account holds on ``server``.

    """
    # Per computer as well as per user: a home folder may be shared between computers
    # (a cluster's), each a host of its own, and two that read one name would each
    # take the host from the other.
    kept = (
        config_dir()
        / "rekursiv-ai"
        / "knowop"
        / "hosts"
        / short_name(hostname.stable_hostname())
        / server_dir(server)
    )
    kept.parent.mkdir(parents=True, exist_ok=True)
    kept.write_text(f"{name}\n")


def default_host_name() -> str:
    """Return this computer's short hostname as a host name.

    It is the :func:`~trackinizer.lib.posix.hostname.stable_hostname`, which a Mac keeps on
    every network: one named by the network would enroll as a new host on each.

    Returns:
      name: The hostname's :func:`short_name`.

    Raises:
      ClientError: The result is reserved or still not a host name.

    """
    name = short_name(hostname.stable_hostname())
    check_name(name)
    return name


def suggested_host_name(taken: str) -> str:
    """Return a name of this user's own on this computer, for ``taken`` held by another.

    Args:
      taken: A host name another person's host holds, such as ``lab``.

    Returns:
      name: ``taken`` and this user's login name, such as ``lab-ada``, cut to
        63 characters.

    """
    return f"{taken}-{short_name(getpass.getuser())}"[:63].rstrip("-")


def taken_name_error(
    name: str,
    *,
    server: str,
    account: str,
    status_code: int,
) -> ClientError:
    """Return the refusal of connecting this computer as ``name``, another's host.

    Args:
      name: The host name another account holds on ``server``.
      server: The server's URL.
      account: The account signed in, which the host is not.
      status_code: The server's refusal, kept so a connect service exits for good
        rather than restart into it.

    Returns:
      error: Says whose it is not, the command that connects under a name of the
        user's own, and how a name no longer used is freed.

    """
    # No owner is named: a writer's refusal names none, since a personal host is seen
    # by its owner alone.
    return ClientError(
        f"host {name} on {server} belongs to another account, not to {account}, who "
        "is signed in. Connect this computer under a name of your own, which knowop "
        f"then keeps for that server: knowop connect {server} --name "
        f"{suggested_host_name(name)}. An admin frees the name of a host no longer "
        f"used: knowop host {name} del",
        status_code=status_code,
    )


def short_name(full: str) -> str:
    """Return a hostname as the host name it makes, unchecked.

    Args:
      full: A hostname, as this computer or a record names it.

    Returns:
      name: Its first label, lowercased, each character a host name may not hold
        made ``-``, trimmed of ``-`` and cut to 63.

    """
    name = re.sub(r"[^a-z0-9-]", "-", full.split(".", maxsplit=1)[0].lower())
    return name.strip("-")[:63]


def check_name(name: str) -> None:
    """Refuse a reserved or malformed host name, before any request.

    Args:
      name: The name as typed or derived.

    Raises:
      ClientError: The name is reserved, or does not match the host name rules.

    """
    if name in wire_hosts.RESERVED_NAMES:
        raise ClientError(f"host name {name!r} is reserved")
    if not re.fullmatch(wire_hosts.NAME_PATTERN, name):
        raise ClientError(
            f"invalid host name {name!r}; expected {wire_hosts.NAME_PATTERN}",
        )


class KnowHost(TypedDict):
    """One KnowHost as the server answers it."""

    id: str
    host: str
    state: str
    memory_kept: NotRequired[bool | None]
    tools: NotRequired[bool | None]


class KnowHosts(TypedDict):
    hosts: list[KnowHost]


class KnowHostClient(Protocol):
    """The HTTP calls needed to rent and operate KnowHosts."""

    def get(self, path: str, *, timeout: float | None = None) -> PlainTree:
        """Read a server response."""
        ...

    def post(
        self,
        path: str,
        *,
        body: object = None,
        timeout: float | None = None,
        retry_attempts: int = 3,
    ) -> PlainTree:
        """Send a server request."""
        ...


def start_knowhost(http: KnowHostClient) -> KnowHost:
    """Rent a KnowHost once, waiting past the server's 90-second deadline.

    Args:
      http: The person's authenticated server client.

    Returns:
      host: The online KnowHost.

    """
    try:
        return from_plain(
            http.post(_KNOWHOSTS, timeout=_WAIT_SEC, retry_attempts=1),
            KnowHost,
        )
    except ClientError as error:
        if error.status_code == 404:
            raise ClientError(
                "this server rents no KnowHosts",
                status_code=404,
            ) from error
        raise


def listed_knowhosts(http: KnowHostClient) -> list[KnowHost] | None:
    """Return the caller's KnowHosts, or None when the server rents none.

    Args:
      http: The person's authenticated server client.

    Returns:
      hosts: The rentals, or None when the server has no rental service.

    """
    try:
        answer = http.get(_KNOWHOSTS)
    except ClientError as error:
        if error.status_code == 404:
            return None
        raise
    return from_plain(answer, KnowHosts)["hosts"]


def named_knowhost(http: KnowHostClient, name: str) -> KnowHost:
    """Return the caller's KnowHost named ``name``."""
    check_name(name)
    for each in listed_knowhosts(http) or ():
        if each["host"] == name:
            return each
    raise ClientError(f"you have no KnowHost {name}")


def pause_knowhost(http: KnowHostClient, knowhost: KnowHost) -> KnowHost:
    """Pause ``knowhost`` now; charging stops."""
    _ = http.post(f"{_KNOWHOSTS}/{knowhost['id']}/pause")
    return {**knowhost, "state": "paused"}


def resume_knowhost(http: KnowHostClient, knowhost: KnowHost) -> KnowHost:
    """Wake ``knowhost``; answer once it is online again.

    Args:
      http: The person's authenticated server client.
      knowhost: The rental to resume.

    Returns:
      host: The online KnowHost, including whether memory was retained.

    """
    return from_plain(
        http.post(
            f"{_KNOWHOSTS}/{knowhost['id']}/resume",
            timeout=_WAIT_SEC,
            retry_attempts=1,
        ),
        KnowHost,
    )


def started_lines(knowhost: KnowHost) -> list[str]:
    """Describe a KnowHost just rented, as the CLI and tool say it."""
    return [
        f"{knowhost['host']} is running: open its terminal from Hosts.",
        "Its knowop and trax are signed in as you.",
        *_tools_lines(knowhost),
    ]


def paused_line(knowhost: KnowHost) -> str:
    """Describe a KnowHost just paused."""
    return f"{knowhost['host']} is paused. {_PAUSE_NOTE}"


def resumed_lines(knowhost: KnowHost) -> list[str]:
    """Describe a KnowHost just woken, including lost memory and absent tools.

    Args:
      knowhost: The rental the server resumed.

    Returns:
      lines: The CLI and tool's explanation of the resume.

    """
    return [
        f"{knowhost['host']} is running again. {_PAUSE_NOTE}",
        *(
            ["It rebooted from its files: its processes were lost."]
            if knowhost.get("memory_kept") is False
            else []
        ),
        *_tools_lines(knowhost),
    ]


def _end(http: Client, knowhost: KnowHost) -> None:
    """End ``knowhost`` for good: its VM, and the host it joined as."""
    _ = http.delete(f"{_KNOWHOSTS}/{knowhost['id']}")
    echo(f"{knowhost['host']} is gone, and its VM with it.")


def _tools_lines(knowhost: KnowHost) -> list[str]:
    """Say when the KnowHost's host serves terminals but no tools."""
    return (
        ["Its knowop serves terminals but no tools."]
        if knowhost.get("tools") is False
        else []
    )


def _get(client_factory: Callable[[], Client], name: str) -> Host:
    """Fetch one host, mapping a failed request to the error the user sees."""
    try:
        return client_factory().get_host(name)
    except ClientError as err:
        raise _refusal(err, role="writer", name=name) from err


def _refusal(err: ClientError, *, role: str, name: str = "") -> ClientError:
    """Map a failed request to the error the user sees."""
    if err.status_code == 403:
        return ClientError(f"needs the {role} role")
    if err.status_code == 404 and name:
        return ClientError(f"no host {name}")
    return err


def _one_line(text: str) -> str:
    """Escape the line breaks of ``text`` so it fits on one line."""
    return text.replace("\r", "\\r").replace("\n", "\\n")


def _labels(host: Host) -> str:
    """Return the labels of ``host`` joined on one line."""
    return _one_line(", ".join(host.labels))


def _cut(how: str) -> str:
    """Return ``how`` on one line of at most 60 characters, ``-`` when empty."""
    one_line = _one_line(how)
    if not one_line:
        return "-"
    return one_line if len(one_line) <= 60 else f"{one_line[:57]}..."


def _read_value(source: str) -> str:
    """Resolve a value source; stdin and file text lose one trailing newline."""
    if source != "-" and not source.startswith("@"):
        return source
    try:
        text = _source_bytes(source).decode()
    except UnicodeDecodeError:
        raise ClientError("the value is not valid UTF-8") from None
    if text.endswith("\n"):
        text = text.removesuffix("\n").removesuffix("\r")
    if not text:
        raise ClientError("empty value; clear a field with to ''")
    return text


def _source_bytes(source: str) -> bytes:
    """Read the bytes of ``-`` (stdin) or ``@FILE``."""
    # Bytes, not text: text mode folds ``\r\n`` into ``\n``, which would hide the
    # pair that counts as one trailing newline.
    if source == "-":
        return cast(bytes, sys.stdin.buffer.read())
    path = source[1:]
    if not path:
        raise ClientError("@ value requires a path")
    try:
        # Resolved against the CALLER's directory: under the daemon the process cwd
        # belongs to whichever shell spawned it.
        return (cwd() / path).read_bytes()
    except OSError as err:
        raise ClientError(
            f"cannot read @{path}: {err.strerror or type(err).__name__}",
        ) from None
