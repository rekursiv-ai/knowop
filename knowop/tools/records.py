"""KnowOp's ``trax.read`` and ``trax.write`` actions: an agent's trax commands.

Each action takes a trax command line, as ``trax`` reads it: list and filter rows
(``issue status is active title re cache``), show one (``issue 7``) and its relations
(``issue 7 requires``), and the read verbs (``recent``, ``search``, ``next``,
``blocked``, ``graph``, ``board``, ``cost``, ``confidence``, ``authority``, ``id``,
``version``, ``help``); a write files, edits, labels, costs, links, unlinks, claims or
deletes a row (``issue title to x narrows issue 5``, ``issue 7 status to complete``,
``issue 7 del``). The command is the verb: an action per command would restate a part
of a grammar the trax CLI and its skill already teach.

Each runs in the agent's own process (``trax.invoke``) on a client of its own, which
calls only its action's routes and refuses any other request before sending it,
whether the agent holds a process key, which the server also holds to those routes,
or a person's own key at a terminal, which it does not: a read calls
:data:`~trackinizer.wire.wire_process_keys.RECORDS_ROUTES`, a write those and
:data:`~trackinizer.wire.wire_process_keys.RECORD_WRITE_ROUTES`. A write is its
agent's: the actor of every change is the agent the tool names, and a command naming
another (``--as``) is refused, as is a bulk edit's ``--makeitso``. A ``-`` or
``@path`` value reads nothing local. The verbs that act on this computer or on other
tools' ground are refused by name (:data:`RECORD_DISPATCHERS`). What the model reads of it is the Records section of
KnowOp's description.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING, Final

import functools
import shlex

from trackinizer.client.client import Client
from trackinizer.trax.cli import Help, parse_and_run
from trackinizer.trax.invoke import invoke
from trackinizer.trax.verbs import (
    Authority,
    Blocked,
    Board,
    Confidence,
    Cost,
    Graph,
    Id,
    Kind,
    Next,
    Recent,
    Search,
    Version,
)
from trackinizer.wire.wire_process_keys import (
    RECORD_WRITE_ROUTES,
    RECORDS_ROUTES,
)


if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from trackinizer.trax.commands import Command

    import httpx2


__all__ = ["RECORD_DISPATCHERS", "Records"]


RECORD_DISPATCHERS: Final[tuple[type[Command], ...]] = (
    Kind,
    Id,
    Recent,
    Search,
    Next,
    Blocked,
    Graph,
    Board,
    Cost,
    Confidence,
    Authority,
    Version,
    Help,
)
"""The trax dispatchers both actions run: every verb on records. ``send`` (AgentSend
does it), ``export`` (writes files here), ``canvas`` (the Canvas tool does it),
``doctor``, ``profile`` and ``env`` (this computer's setup, the org's variables) are
refused by name."""

_WIDTH: Final = 160
"""The table width trax renders at: wide enough for a row's title, narrow enough that
a list of them stays a list; sagent bounds the answer's length."""


class Records:
    """Run trax commands on records for one agent, with its server and key."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str,
        transport: httpx2.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._transport = transport

    def read(self, command: str) -> str:
        """Run one trax read command and answer with what it printed.

        Args:
          command: A trax command line, as typed after ``trax`` (a leading ``trax``
            is dropped), such as ``issue status is active title re cache``.

        Returns:
          output: What the command printed, then what it reported as an error; for
            a command a shell could not split, ``The command could not be read:``
            and why, with nothing sent.

        """
        return self._ran(command, routes=RECORDS_ROUTES.keys(), actor="")

    def write(self, command: str, *, actor: str) -> str:
        """Run one trax command that may change records, as ``actor``.

        Args:
          command: A trax command line, as typed after ``trax``, such as
            ``issue 7 status to complete``.
          actor: The agent each change is recorded as (``lead#<session>``).

        Returns:
          output: As :meth:`read` answers; for a command naming ``--as`` or
            ``--makeitso``, one line saying so, with nothing sent.

        """
        return self._ran(
            command,
            routes=RECORDS_ROUTES.keys() | RECORD_WRITE_ROUTES.keys(),
            actor=actor,
        )

    # The client opens when the command first asks for it, so help and a refused verb
    # open none, and a client that cannot be built is the command's error.
    def _ran(
        self,
        command: str,
        *,
        routes: Collection[tuple[str, str]],
        actor: str,
    ) -> str:
        """Run ``command`` on a client of its own, held to ``routes``, as ``actor``."""
        try:
            argv = shlex.split(command)
        except ValueError as error:
            return f"The command could not be read: {error}\n"
        if argv[:1] == ["trax"]:
            argv = argv[1:]
        if actor and (refused := _refused_flag(argv, actor=actor)):
            return refused
        with ExitStack() as opened:
            client = functools.cache(
                functools.partial(self._opened, opened, routes=routes, actor=actor),
            )
            ran = invoke(
                argv,
                env={},
                width=_WIDTH,
                local_input=None,
                run=functools.partial(
                    parse_and_run,
                    client_factory=client,
                    allowed=RECORD_DISPATCHERS,
                ),
            )
        return ran.stdout + ran.stderr

    def _opened(
        self,
        opened: ExitStack,
        *,
        routes: Collection[tuple[str, str]],
        actor: str,
    ) -> Client:
        """Open this agent's client, held to ``routes``; ``opened`` closes it."""
        return opened.enter_context(
            Client(
                self._base_url,
                author=actor,
                api_key=self._api_key,
                transport=self._transport,
                routes=routes,
            ),
        )


# Argparse takes any unambiguous prefix of a long option, so ``--make`` is
# ``--makeitso`` and ``--act`` is ``--actor``: matching the whole spelling alone would
# let either through.
def _refused_flag(argv: Sequence[str], *, actor: str) -> str:
    """Return why a write refuses ``argv``'s ``--as`` or ``--makeitso``; empty if not."""
    for word in argv:
        head = word.partition("=")[0]
        if len(head) > 2 and "--makeitso".startswith(head):
            return "trax.write takes no --makeitso: a bulk edit only previews.\n"
        if len(head) > 2 and ("--as".startswith(head) or "--actor".startswith(head)):
            return f"trax.write takes no --as: it writes as {actor}.\n"
    return ""
