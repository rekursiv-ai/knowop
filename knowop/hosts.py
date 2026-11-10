"""The ``knowop host`` command: your hosts, where campaigns may run, and KnowHosts.

The registry records the hosts campaigns may run on; listing and showing need the
writer role, and changing the registry needs the admin role. A KnowHost, a host the
server rents you, is one more host here: ``host start`` rents one, ``host NAME pause``
and ``resume`` stop and wake it, and ``host NAME del`` ends it, through the server's
KnowHost API (``/api/addons/knowhost``). ``knowop connect`` names this computer by
:func:`host_name` and checks a name it is given with :func:`check_name`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Final, NotRequired, TypedDict, cast, override

import argparse
import re
import socket
import sys

from trackinizer.client.errors import ClientError
from trackinizer.trax.commands import Command, HelpPage
from trackinizer.trax.context import cwd
from trackinizer.trax.render import echo
from treekle import from_plain


if TYPE_CHECKING:
    from trackinizer.client.client import Client
    from trackinizer.wire import wire_hosts
    from trackinizer.wire.wire_hosts import Host
else:
    from wrapt import lazy_import

    # Only ``knowop host`` touches the host models, so their pydantic build stays
    # off the cold-start path of every other verb.
    wire_hosts = lazy_import("trackinizer.wire.wire_hosts")


__all__ = ["HOST_WORD", "Hosts", "check_name", "host_name"]

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
            "need the admin role. 'host start' rents a _KnowHost, billed to your "
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
        parser = argparse.ArgumentParser(
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
            _start(client_factory())
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
                knowhost = _knowhost(http, name)
                (_pause if action == "pause" else _resume)(http, knowhost)
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
    rented = _listed(http)
    if rented:
        echo("")
        echo("KnowHosts:")
    for each in rented:
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
    for each in _listed(http):
        if each["host"] == name:
            _end(http, each)
            return
    try:
        http.delete_host(name)
    except ClientError as err:
        raise _refusal(err, role="admin", name=name) from err
    echo(f"deleted: {HOST_WORD} {name}")


def host_name() -> str:
    """Return this computer's short hostname as a host name.

    The command that connects this computer enrolls it under this name, and an agent
    run says it runs on the host of that name.

    Returns:
      name: The hostname's first label, lowercased, each character a host name
        may not hold made ``-``, trimmed of ``-`` and cut to 63.

    Raises:
      ClientError: The result is reserved or still not a host name.

    """
    name = re.sub(r"[^a-z0-9-]", "-", socket.gethostname().split(".")[0].lower())
    name = name.strip("-")[:63]
    check_name(name)
    return name


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


class _KnowHost(TypedDict):
    """One _KnowHost as the server answers it."""

    id: str
    host: str
    state: str
    memory_kept: NotRequired[bool | None]
    tools: NotRequired[bool | None]


class _KnowHosts(TypedDict):
    hosts: list[_KnowHost]


def _start(http: Client) -> None:
    """Start a _KnowHost, sent once: a retry would wait out the same deadline again."""
    try:
        knowhost = from_plain(
            http.post(_KNOWHOSTS, timeout=_WAIT_SEC, retry_attempts=1),
            _KnowHost,
        )
    except ClientError as error:
        if error.status_code == 404:
            raise ClientError(
                "this server rents no KnowHosts",
                status_code=404,
            ) from error
        raise
    echo(f"{knowhost['host']} is running: open its terminal from Hosts.")
    _say_tools(knowhost)


def _listed(http: Client) -> list[_KnowHost]:
    """Return the caller's KnowHosts; none from a server that rents none (404)."""
    try:
        answer = http.get(_KNOWHOSTS)
    except ClientError as error:
        if error.status_code == 404:
            return []
        raise
    return from_plain(answer, _KnowHosts)["hosts"]


def _knowhost(http: Client, name: str) -> _KnowHost:
    """Return the caller's KnowHost named ``name``."""
    for each in _listed(http):
        if each["host"] == name:
            return each
    raise ClientError(f"you have no KnowHost {name}")


def _pause(http: Client, knowhost: _KnowHost) -> None:
    """Pause ``knowhost`` now; charging stops."""
    _ = http.post(f"{_KNOWHOSTS}/{knowhost['id']}/pause")
    echo(f"{knowhost['host']} is paused. {_PAUSE_NOTE}")


def _resume(http: Client, knowhost: _KnowHost) -> None:
    """Wake ``knowhost``; answer once it is online again."""
    woken = from_plain(
        http.post(
            f"{_KNOWHOSTS}/{knowhost['id']}/resume",
            timeout=_WAIT_SEC,
            retry_attempts=1,
        ),
        _KnowHost,
    )
    echo(f"{woken['host']} is running again. {_PAUSE_NOTE}")
    if woken.get("memory_kept") is False:
        echo("It rebooted from its files: its processes were lost.")
    _say_tools(woken)


def _end(http: Client, knowhost: _KnowHost) -> None:
    """End ``knowhost`` for good: its VM, and the host it joined as."""
    _ = http.delete(f"{_KNOWHOSTS}/{knowhost['id']}")
    echo(f"{knowhost['host']} is gone, and its VM with it.")


def _say_tools(knowhost: _KnowHost) -> None:
    """Say when the KnowHost's host serves terminals but no tools."""
    if knowhost.get("tools") is False:
        echo("Its knowop serves terminals but no tools.")


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
