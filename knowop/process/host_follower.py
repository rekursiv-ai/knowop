"""Discover eligible hosts and connect a run to only the ones its tools use.

The server lists the hosts of the key's account, or those of an assistant's asker.
Listing needs no tool link. Selecting a host adds a native sagent link that stays
until the host becomes ineligible or the run ends. A host marked offline still
keeps an existing link so a pending call can survive a disconnect. An unchanged
listing does not redial a kept link. Losing access removes it and ends pending calls.

``set_links`` replaces the whole set, so the links sagent built itself, this machine's
and each saved host's, are passed in as ``keep`` and set again with every listing.
The host the agent runs on is left out of the listing: its local link reaches it.

The follower lists once, then holds a wait on the listing's version, which the server
moves with each write to the account's hosts, and lists again when it moves. The
first listing raises if it fails; a later failure keeps the last list through an
outage, asking again after ``retry_sec``, and ends the relayed links when the server
refuses.

Typical usage example:

  async with HostFollower(HostFollower.Config(), client=client, url=url,
                          api_key=key, remote=remote, keep=links, own_host=host):
      ...
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Final, Protocol, cast, override

import asyncio
import contextlib
import logging
import threading

from sagent.host.wire import MAX_LINE_BYTES
from sagent.journal.types import RemoteToolResult, ToolMessage
from sagent.tools.remote import Link, RemoteTool
from sagent.tools.types import parse
from sagent.types import AsyncResource
from trackinizer.client.errors import ClientError
from trackinizer.client.tool_bridge import dial_tools
from trackinizer.wire import wire_terminals


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from sagent.tools.remote import RemoteSession
    from sagent.tools.types import Stream, ToolContext
    from trackinizer.wire.wire_terminals import HostsVersion, TerminalHosts
    from treekle import PlainTree


__all__ = ["HostFollower", "HostLister"]


_LOG = logging.getLogger(__name__)

_REVOKED: Final = 401
"""What the server answers a key it no longer accepts."""


class HostLister(Protocol):
    """The ``Client`` methods that list an account's hosts and wait on their change."""

    def terminal_hosts(self, *, owner: str | None = None) -> TerminalHosts:
        """List the hosts of ``owner``, whom only the configured assistant names."""
        ...

    def hosts_version(
        self,
        *,
        since: int | None,
        wait_sec: float,
        owner: str | None = None,
    ) -> HostsVersion:
        """Hold for the version of ``owner``'s host list to move from ``since``."""
        ...


class HostFollower(AsyncResource):
    """Follow the hosts an account may use, and make them the session's links.

    Args:
      config: How long each wait is held, and the delay after a failed one.
      client: Lists the hosts.
      url: The server's base URL, which each host's tools are dialed through.
      api_key: The key sent to the server and shown nowhere.
      remote: The session whose links follow the list.
      asker: The email the configured assistant follows the hosts of; ``None`` for
        the key's own account.
      keep: The links set beside the listed hosts each time: those sagent built.
      own_host: The host the agent runs on, left out of the listing; empty for none.

    """

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Config:
        """How the follower waits."""

        wait_sec: float = 25.0
        """How long the server may hold each wait for the list to change; it holds
        none longer than ``wire_terminals.WAIT_MAX_SEC``."""

        retry_sec: float = 5.0
        """How long to wait before asking again after a wait or a listing failed."""

    def __init__(
        self,
        config: Config,
        *,
        client: HostLister,
        url: str,
        api_key: str,
        remote: RemoteSession,
        asker: str | None = None,
        keep: Sequence[Link] = (),
        own_host: str,
    ) -> None:
        self.config = config
        self._client = client
        self._url = url
        self._api_key = api_key
        self._asker = asker
        self._remote = remote
        self._keep = tuple(keep)
        self._own_host = own_host
        self._listed: dict[str, str] = {}
        self._active: set[str] = set()
        self._version: int | None = None
        """The version of the list the links were last set from; ``None`` for none."""

        self._task: asyncio.Task[None]
        """Follows the list once the first listing is in; a follower that failed to
        open is not closed, so none reads it before ``aopen`` sets it."""

    @override
    async def aopen(self) -> None:
        """List the hosts, then follow each change of them until closed.

        Raises:
          ClientError: The first listing failed, so there is no list to keep.

        """
        await self._list(since=self._version)
        self._task = asyncio.create_task(self._follow())

    @override
    async def aclose(self) -> None:
        """Stop following; the links stay as they were."""
        _ = self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

    def tool(self, native: RemoteTool) -> RemoteTool:
        """Bind a listing-aware tool to this run's host discovery.

        Args:
          native: The run's sagent tool, with its local tool schemas.

        Returns:
          tool: The same tool interface, with connection on first use.

        """
        return _HostedTool(
            remote=native.remote,
            local=native.local,
            display=native.display,
            hosts=self,
        )

    @property
    def unconnected(self) -> dict[str, str]:
        """List eligible hosts that do not already have an endpoint."""
        connected = self._remote.endpoints()
        return {
            name: status
            for name, status in self._listed.items()
            if name not in connected
        }

    async def activate(self, name: str) -> None:
        """Connect just the selected host, retaining it until it leaves the list.

        Args:
          name: A host to activate if it belongs to the account's list.

        """
        if name not in self._listed:
            return
        if name not in self._remote.endpoints():
            self._active.add(name)
            self._apply()
        endpoint = self._remote.endpoints()[name]
        try:
            async with asyncio.timeout(10.0):
                await endpoint.until(lambda remote: remote.tried)
        except TimeoutError as error:
            raise ValueError(
                f"The tool link to {name} did not connect in time.",
            ) from error

    # A server that cannot be reached leaves the links as they were. One that refuses
    # has ended the account's access, and each pipe's redial would be refused too, with
    # calls left waiting, so the relayed links end and those calls fail. Only the
    # server's own refusal says so: a 401 (the key is no longer good; an edge never
    # answers one) or a 403 naming the server's code. Any other, such as an edge's 403
    # in front of it, is an outage, and ending the links on it would end every call.
    async def _follow(self) -> None:
        """Wait for the list to change and list it again, until cancelled."""
        while True:
            try:
                await self._list(since=self._version)
            except ClientError as error:
                _LOG.warning("host list failed: %s", error)
                if (
                    error.status_code == _REVOKED
                    or error.code == wire_terminals.TERMINAL_FORBIDDEN
                ):
                    self._listed.clear()
                    self._active.clear()
                    self._apply()
                    # Listed again the moment access comes back, whatever the version.
                    self._version = None
                await asyncio.sleep(self.config.retry_sec)

    async def _list(self, *, since: int | None) -> None:
        """Hold until the list's version is not ``since``, then set the links from it."""
        changed = await _off_loop(
            partial(
                self._client.hosts_version,
                owner=self._asker,
                since=since,
                wait_sec=self.config.wait_sec,
            ),
        )
        if changed.version == since:
            return
        listed = await _off_loop(
            partial(self._client.terminal_hosts, owner=self._asker),
        )
        self._listed = {
            m.name: m.status
            for m in listed.hosts
            if m.status != "revoked"
            and m.confirmed
            and m.tools
            and m.name != self._own_host
        }
        self._active.intersection_update(self._listed)
        self._apply()
        # Kept only once listed: a change while it listed moves the version again, and
        # the next wait answers at once.
        self._version = changed.version

    def _apply(self) -> None:
        self._remote.set_links(
            [
                *self._keep,
                *(
                    Link(
                        dial=partial(
                            dial_tools,
                            self._url,
                            api_key=self._api_key,
                            host=name,
                            asker=self._asker,
                            limit=MAX_LINE_BYTES,
                        ),
                        name=name,
                        where=self._url,
                    )
                    for name in self._listed
                    if name in self._active
                ),
            ],
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _HostedTool(RemoteTool):
    hosts: HostFollower

    @override
    def reminder(self, context: ToolContext) -> str:
        native = super(_HostedTool, self).reminder(context)
        listed = ", ".join(
            f"{name} ({status}; on demand)"
            for name, status in self.hosts.unconnected.items()
        )
        return "; ".join(
            part
            for part in (native, f"remote hosts: {listed}" if listed else "")
            if part
        )

    @override
    async def _run(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
        streams: Mapping[str, Stream[object]],
    ) -> ToolMessage:
        asked = parse(arguments, RemoteTool.Arguments)
        if asked.endpoint:
            if asked.tool != "AgentSend":
                await self.hosts.activate(asked.endpoint)
            return await super(_HostedTool, self)._run(context, arguments, streams)
        listed = self.hosts.unconnected
        if not listed or asked.tool == "AgentSend":
            return await super(_HostedTool, self)._run(context, arguments, streams)
        available = "\n".join(
            f"{name}  {status}: opens on first use" for name, status in listed.items()
        )
        if not self.remote.endpoints():
            return ToolMessage(
                call_id="",
                state="done",
                stdout=(available,),
                result=RemoteToolResult(),
            )
        native = await super(_HostedTool, self)._run(context, arguments, streams)
        return replace(native, stdout=(*native.stdout, available))


# Not ``asyncio.to_thread``: the event loop's executor is joined as it closes, so a wait
# the server is still holding would keep the process from exiting for up to its hold.
async def _off_loop[T](call: Callable[[], T]) -> T:
    """Return what ``call`` returns, run on a daemon thread of its own."""
    loop = asyncio.get_running_loop()
    answer = cast(asyncio.Future[T], loop.create_future())

    def run() -> None:
        try:
            result = call()
        except Exception as error:  # noqa: BLE001 -- Raised again on the loop, by the awaiting caller.
            _settle(loop, partial(_fail, answer, error))
        else:
            _settle(loop, partial(_succeed, answer, result))

    threading.Thread(target=run, name="host-list", daemon=True).start()
    return await answer


def _settle(loop: asyncio.AbstractEventLoop, settle: Callable[[], None]) -> None:
    """Settle an answer on its loop; one that has closed meanwhile has no one waiting."""
    with contextlib.suppress(RuntimeError):
        _ = loop.call_soon_threadsafe(settle)


def _succeed[T](answer: asyncio.Future[T], result: T) -> None:
    """Give ``answer`` its result, unless its caller was cancelled."""
    if not answer.done():
        answer.set_result(result)


def _fail[T](answer: asyncio.Future[T], error: Exception) -> None:
    """Give ``answer`` its error, unless its caller was cancelled."""
    if not answer.done():
        answer.set_exception(error)
