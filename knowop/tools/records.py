"""KnowOp's ``trax.read`` action: an agent reads Trackinizer's records in trax's grammar.

The action takes a trax command line, as ``trax`` reads it: list and filter rows
(``issue status is active title re cache``), show one (``issue 7``) and its relations
(``issue 7 requires``), and the read verbs (``recent``, ``search``, ``next``,
``blocked``, ``graph``, ``board``, ``cost``, ``confidence``, ``authority``, ``id``,
``version``, ``help``). The command is the verb: an action per read command would
restate a part of a grammar the trax CLI and its skill already teach.

It runs in the agent's own process (``trax.invoke``) and reads only, under any key:
its client calls only :data:`~trackinizer.wire.wire_process_keys.RECORDS_ROUTES`
and refuses any other request before sending it, so a write never leaves, whether the
agent holds a process key, which the server also holds to that list, or a person's own
key at a terminal, which it does not. A ``-`` or ``@path`` value reads nothing local.
The verbs that act on this computer or on other tools' ground are refused by name
(:data:`READ_DISPATCHERS`). What the model reads of it is the Records section of
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
from trackinizer.wire.wire_process_keys import RECORDS_ROUTES


if TYPE_CHECKING:
    from collections.abc import Sequence

    from trackinizer.trax.commands import Command

    import httpx2


__all__ = ["READ_DISPATCHERS", "RecordReads"]


READ_DISPATCHERS: Final[tuple[type[Command], ...]] = (
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
"""The trax dispatchers the action runs: every reading verb. ``send`` (AgentSend does
it), ``export`` (writes files here), ``canvas`` (the Canvas tool does it), ``doctor``,
``profile`` and ``env`` (this computer's setup) are refused by name."""

_WIDTH: Final = 160
"""The table width trax renders at: wide enough for a row's title, narrow enough that
a list of them stays a list; sagent bounds the answer's length."""


class RecordReads:
    """Run trax read commands for one agent, with its server and key."""

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
        try:
            argv = shlex.split(command)
        except ValueError as error:
            return f"The command could not be read: {error}\n"
        if argv[:1] == ["trax"]:
            argv = argv[1:]
        return self._run(argv)

    # The client opens when the command first asks for it, so help and a refused verb
    # open none, and a client that cannot be built is the command's error.
    def _run(self, argv: Sequence[str]) -> str:
        """Run ``argv`` on a client of its own, closed when the command ends."""
        with ExitStack() as opened:
            client = functools.cache(functools.partial(self._opened, opened))
            ran = invoke(
                argv,
                env={},
                width=_WIDTH,
                local_input=None,
                run=functools.partial(
                    parse_and_run,
                    client_factory=client,
                    allowed=READ_DISPATCHERS,
                ),
            )
        return ran.stdout + ran.stderr

    def _opened(self, opened: ExitStack) -> Client:
        """Open this agent's client, held to the Records routes; ``opened`` closes it."""
        return opened.enter_context(
            Client(
                self._base_url,
                api_key=self._api_key,
                transport=self._transport,
                routes=RECORDS_ROUTES,
            ),
        )
