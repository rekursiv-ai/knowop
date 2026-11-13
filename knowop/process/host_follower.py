"""Keep a sagent session linked to the connected hosts of an account.

The server lists the hosts of the key's own account, or, for the configured
assistant, of the person it answers as, and each confirmed one that serves tools and
is not revoked is a ``Link`` in the ``RemoteSession``: the hosts that account may use.
Whether one is up now is its link's to find out, so a host the listing shows offline
keeps its link, and a call running on it goes on. ``set_links`` keeps a host by name,
so a listing that changes nothing disturbs nothing: the host is not redialled and the
calls waiting on it go on. A host that leaves the list (removed, revoked, unconfirmed,
or no longer serving tools) is dropped by the next listing, which says ``Bye``.

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

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Final, Protocol, cast, override

import asyncio
import contextlib
import logging
import threading

from sagent.host.wire import MAX_LINE_BYTES
from sagent.tools.remote import Link
from sagent.types import AsyncResource
from trackinizer.client.errors import ClientError
from trackinizer.client.tool_bridge import dial_tools
from trackinizer.wire import wire_terminals


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from sagent.tools.remote import RemoteSession
    from trackinizer.wire.wire_terminals import HostsVersion, TerminalHosts


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
                    self._remote.set_links(self._keep)
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
        # Which hosts the account may use; whether each is up is its link's to find.
        self._remote.set_links(
            [
                *self._keep,
                *(
                    Link(
                        dial=partial(
                            dial_tools,
                            self._url,
                            api_key=self._api_key,
                            host=m.name,
                            asker=self._asker,
                            limit=MAX_LINE_BYTES,
                        ),
                        name=m.name,
                        where=self._url,
                    )
                    for m in listed.hosts
                    if m.status != "revoked"
                    and m.confirmed
                    and m.tools
                    and m.name != self._own_host
                ),
            ],
        )
        # Kept only once listed: a change while it listed moves the version again, and
        # the next wait answers at once.
        self._version = changed.version


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
