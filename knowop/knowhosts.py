"""The ``knowop host`` command: rent a KnowHost that joins as your machine.

A thin client of the server's KnowHost API (``/api/addons/knowhost``). ``start``
answers once the KnowHost's machine is online, as Machines then lists it; ``resume``
is the one way to wake a paused KnowHost. A KnowHost is named by its machine
(``box-...``) or its id.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Final, NotRequired, TypedDict, cast, override

import argparse

from trackinizer.client.errors import ClientError
from trackinizer.trax.commands import Command, HelpPage
from trackinizer.trax.render import echo
from treekle import from_plain


if TYPE_CHECKING:
    from trackinizer.client.client import Client


__all__ = ["KNOWHOST_WORD", "KnowHosts"]

KNOWHOST_WORD: Final = "host"
"""The verb that selects this command."""

PROGRAM: Final = f"knowop {KNOWHOST_WORD}"
"""The command's name, in help, usage and messages."""

_PATH: Final = "/api/addons/knowhost/hosts"

_FORMS: Final = "[start | list | pause NAME | resume NAME | rm NAME]"

_WAIT_SEC: Final = 150.0
"""How long a start or resume is waited on: past the server's own 90 s deadline."""


class _KnowHost(TypedDict):
    """One KnowHost as the server answers it."""

    id: str
    machine: str
    state: str
    memory_kept: NotRequired[bool | None]
    tools: NotRequired[bool | None]


_PAUSE_NOTE: Final = (
    "A terminal open across a pause ends; jobs started with nohup or in tmux, and "
    "tool calls, carry on."
)


class KnowHosts(Command):
    """Start, list, pause, resume and remove your rented KnowHosts."""

    names = (KNOWHOST_WORD,)

    help = HelpPage(
        usage=f"{PROGRAM} {_FORMS}",
        summary=(
            "Rent a KnowHost, billed to your account, that joins as one of your "
            "machines: its terminal opens from Machines. It pauses after 30 minutes "
            "with no tool call and no terminal use, and nothing is charged while it "
            "is paused; resume wakes it."
        ),
        arguments=(("NAME", "the KnowHost's machine name (box-...) or its id"),),
        examples=(
            f"{PROGRAM} start              start one; prints its machine name",
            f"{PROGRAM}                    list yours",
            f"{PROGRAM} pause box-1a2b3c   pause it now",
            f"{PROGRAM} resume box-1a2b3c  wake it, once it is online again",
            f"{PROGRAM} rm box-1a2b3c      end it for good, and its machine",
        ),
        notes=(_PAUSE_NOTE,),
    )

    @classmethod
    @override
    def make_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog=PROGRAM, description="Rent KnowHosts.")
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
        match list(cast(Sequence[str], args.rest)):
            case [] | ["list"]:
                _list(client_factory())
            case ["start"]:
                _start(client_factory())
            case ["pause" | "resume" | "rm" as action, name]:
                _act(client_factory(), action=action, name=name)
            case _:
                raise ClientError(f"usage: {PROGRAM} {_FORMS}")


def _start(http: Client) -> None:
    """Start a KnowHost, sent once: a retry would wait out the same deadline again."""
    try:
        knowhost = from_plain(
            http.post(_PATH, timeout=_WAIT_SEC, retry_attempts=1),
            _KnowHost,
        )
    except ClientError as error:
        raise _refusal(error) from error
    echo(f"{knowhost['machine']} is running: open its terminal from Machines.")
    _say_tools(knowhost)


def _list(http: Client) -> None:
    """Print one line per KnowHost: machine, state and id."""
    knowhosts = _listed(http)
    if not knowhosts:
        echo("(no KnowHosts)")
        return
    for knowhost in knowhosts:
        echo(f"{knowhost['machine']}  {knowhost['state'].ljust(8)}  {knowhost['id']}")


def _act(http: Client, *, action: str, name: str) -> None:
    """Pause, resume or remove the KnowHost ``name`` names."""
    host_id = _host_id(http, name)
    if action == "rm":
        _ = http.delete(f"{_PATH}/{host_id}")
        echo(f"{name} is gone, and its machine with it.")
        return
    if action == "pause":
        _ = http.post(f"{_PATH}/{host_id}/pause")
        echo(f"{name} is paused. {_PAUSE_NOTE}")
        return
    knowhost = from_plain(
        http.post(f"{_PATH}/{host_id}/resume", timeout=_WAIT_SEC, retry_attempts=1),
        _KnowHost,
    )
    echo(f"{knowhost['machine']} is running again. {_PAUSE_NOTE}")
    if knowhost.get("memory_kept") is False:
        echo("It rebooted from its files: its processes were lost.")
    _say_tools(knowhost)


def _host_id(http: Client, name: str) -> str:
    """Return the id of the KnowHost whose machine or id is ``name``."""
    for knowhost in _listed(http):
        if name in {knowhost["machine"], knowhost["id"]}:
            return knowhost["id"]
    raise ClientError(f"you have no KnowHost {name}")


def _listed(http: Client) -> list[_KnowHost]:
    """Return the caller's KnowHosts as the server lists them."""
    try:
        answer = http.get(_PATH)
    except ClientError as error:
        raise _refusal(error) from error
    return from_plain(answer, _Listed)["hosts"]


def _say_tools(knowhost: _KnowHost) -> None:
    """Say when the KnowHost's machine serves terminals but no tools."""
    if knowhost.get("tools") is False:
        echo("Its knowop serves terminals but no tools.")


def _refusal(error: ClientError) -> ClientError:
    """Say a server with no KnowHost rents none: it answers its routes 404."""
    if error.status_code == 404:
        return ClientError("this server rents no KnowHosts", status_code=404)
    return error


class _Listed(TypedDict):
    hosts: list[_KnowHost]
