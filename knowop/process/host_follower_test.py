"""Tests for HostFollower: the hosts a person may use, followed as links.

Each host's tools answer over an in-process socket pair, and the follower's delays
are recorded, not slept, so no test opens a port or waits real time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING

import asyncio
import contextlib
import socket
import threading

from sagent.host.wire import (
    MAX_LINE_BYTES,
    Call,
    Describe,
    Described,
    Hello,
    Result,
    ToolSpec,
    Welcome,
    encoded,
    received,
)
from sagent.session.types import BashResult, ToolResult
from sagent.tools.bash import Bash
from sagent.tools.remote import Link, RemoteSession
from trackinizer.client.errors import ClientError
from trackinizer.wire.wire_terminals import (
    HostsVersion,
    TerminalHost,
    TerminalHosts,
)

import pytest

from knowop.process import host_follower
from knowop.process.host_follower import HostFollower


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Sequence

    from sagent.host.wire import Streams


_BASH = ToolSpec.from_tool(Bash())


@pytest.fixture(autouse=True)
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Keep each delay the follower sleeps, and sleep none of it."""
    # Not ``asyncio.sleep`` itself, as ``startup_clock`` patches it for the terminal
    # tests: sagent's links sleep between heartbeats, and theirs would spin.
    delays: list[float] = []
    monkeypatch.setattr(
        host_follower,
        "asyncio",
        SimpleNamespace(
            CancelledError=asyncio.CancelledError,
            Future=asyncio.Future,
            Task=asyncio.Task,
            create_task=asyncio.create_task,
            get_running_loop=asyncio.get_running_loop,
            sleep=_Sleep(delays=delays),
        ),
    )
    return delays


@dataclass(frozen=True, slots=True, kw_only=True)
class _Sleep:
    """Keep each delay asked for, and yield to the loop in its place."""

    delays: list[float]

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)
        await asyncio.sleep(0)


def _host(
    name: str,
    *,
    status: str = "online",
    tools: bool = True,
    confirmed: bool = True,
) -> TerminalHost:
    return TerminalHost.model_validate(
        {
            "name": name,
            "status": status,
            "terminal_ready": False,
            "confirmed": confirmed,
            "tools": tools,
        },
    )


@dataclass(slots=True, kw_only=True)
class _Client:
    """A trackinizer ``Client`` that lists what it is told, holds waits, or is down.

    The follower calls it from threads of its own, so what it changes is under a lock.
    """

    hosts: list[TerminalHost] = field(default_factory=list[TerminalHost])
    version: int = 7
    down: bool = False
    refusing: int | None = None
    refusing_code: str = ""
    """The code the refusal's body names; ``""`` for none, as an edge's own."""

    asked: list[str | None] = field(default_factory=list[str | None])
    """The ``owner`` of each listing."""

    waited: list[tuple[int | None, str | None]] = field(
        default_factory=list[tuple[int | None, str | None]],
    )
    """The ``since`` and ``owner`` of each wait, as it started."""

    held: list[bool] = field(default_factory=list[bool])
    """Whether each wait that answered was held to its end."""

    listed_by: list[int] = field(default_factory=list[int])
    """How many listings there had been as each wait started."""

    threads: list[tuple[str, bool]] = field(default_factory=list[tuple[str, bool]])
    """The name of the thread each call came on, and whether it is a daemon."""

    closed: bool = False
    changed: threading.Condition = field(default_factory=threading.Condition)

    def close(self) -> None:
        self.change(closed=True)

    def change(self, **changes: object) -> None:
        """Set what the server holds, and answer every wait held on the old version."""
        with self.changed:
            for name, value in changes.items():
                setattr(self, name, value)
            self.changed.notify_all()

    def bump(self, hosts: list[TerminalHost]) -> None:
        """Write the hosts, as a route does, and move the version on."""
        self.change(hosts=hosts, version=self.version + 1)

    def terminal_hosts(self, *, owner: str | None = None) -> TerminalHosts:
        with self.changed:
            self._came_on(threading.current_thread())
            self._refuse()
            self.asked.append(owner)
            self.changed.notify_all()
            return TerminalHosts(hosts=self.hosts)

    def hosts_version(
        self,
        *,
        since: int | None,
        wait_sec: float,
        owner: str | None = None,
    ) -> HostsVersion:
        with self.changed:
            self._came_on(threading.current_thread())
            self.waited.append((since, owner))
            self.listed_by.append(len(self.asked))
            self.changed.notify_all()
            self._refuse()
            moved = self.changed.wait_for(
                lambda: (
                    since != self.version
                    or self.down
                    or self.refusing is not None
                    or self.closed
                ),
                timeout=wait_sec,
            )
            self._refuse()
            self.held.append(not moved)
            return HostsVersion(version=self.version)

    async def until(self, holds: Callable[[_Client], bool]) -> None:
        """Return once ``holds`` is true of this client, within 5 seconds."""
        assert await asyncio.to_thread(self._until, holds), "the change never came"

    def _until(self, holds: Callable[[_Client], bool]) -> bool:
        with self.changed:
            return self.changed.wait_for(lambda: holds(self), timeout=5)

    def _came_on(self, thread: threading.Thread) -> None:
        self.threads.append((thread.name, thread.daemon))

    def _refuse(self) -> None:
        if self.down:
            raise ClientError("server unreachable")
        if self.refusing is not None:
            raise ClientError(
                "not allowed",
                status_code=self.refusing,
                code=self.refusing_code,
            )


@dataclass(slots=True, kw_only=True)
class _Daemons:
    """Fake tool hosts by host, each counting its connections and calls."""

    connects: dict[str, int] = field(default_factory=dict[str, int])
    calls: dict[str, list[Call]] = field(default_factory=dict[str, list[Call]])
    dialed: list[tuple[str, str, str, str | None]] = field(
        default_factory=list[tuple[str, str, str, str | None]],
    )
    answer: bool = False
    names: tuple[str, ...] = ()
    """The hosts that answer a dial."""

    serving: list[asyncio.Task[None]] = field(default_factory=list[asyncio.Task[None]])
    """Each dialed host's fake, serving its connection."""

    held: asyncio.Event = field(default_factory=asyncio.Event)
    sessions: list[str] = field(default_factory=list[str])
    """The session each hello named."""

    limits: list[int] = field(default_factory=list[int])
    """The longest line each dial's reader takes."""

    async def serve(
        self,
        host: str,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        hello = await received(reader)
        assert isinstance(hello, Hello)
        self.sessions.append(hello.session_id)
        self.connects[host] = self.connects.get(host, 0) + 1
        writer.write(
            encoded(
                Welcome(
                    name=host,
                    resumed=False,
                    calls=(),
                    tools={"Bash": _BASH.digest},
                ),
            ),
        )
        while (message := await received(reader)) is not None:
            if isinstance(message, Describe):
                writer.write(encoded(Described(id=message.id, spec=_BASH)))
            elif isinstance(message, Call):
                self.calls.setdefault(host, []).append(message)
                self.held.set()
                if self.answer:
                    record = ToolResult(
                        call_id="",
                        state="done",
                        stdout=(f"{host}\n",),
                        result=BashResult(command=("hostname",), exit_code=0),
                    )
                    writer.write(encoded(Result(id=message.id, record=record)))

    async def dial(
        self,
        url: str,
        *,
        api_key: str,
        host: str,
        limit: int,
        asker: str | None = None,
    ) -> Streams:
        """Reach ``host``'s fake over an in-process socket pair, as a dial would."""
        self.dialed.append((url, api_key, host, asker))
        self.limits.append(limit)
        if host not in self.names:
            raise ConnectionRefusedError(host)
        ours, theirs = socket.socketpair()
        served = await asyncio.open_connection(sock=theirs, limit=MAX_LINE_BYTES)
        self.serving.append(asyncio.create_task(self.serve(host, *served)))
        return await asyncio.open_connection(sock=ours, limit=MAX_LINE_BYTES)


@contextlib.asynccontextmanager
async def _daemons(
    monkeypatch: pytest.MonkeyPatch,
    *names: str,
    answer: bool = False,
) -> AsyncGenerator[_Daemons]:
    daemons = _Daemons(names=names, answer=answer)
    monkeypatch.setattr(host_follower, "dial_tools", daemons.dial)
    try:
        yield daemons
    finally:
        for serving in daemons.serving:
            _ = serving.cancel()
        _ = await asyncio.gather(*daemons.serving, return_exceptions=True)


@contextlib.asynccontextmanager
async def _follower(
    client: _Client,
    remote: RemoteSession,
    *,
    wait_sec: float = 3600,
    retry_sec: float = 3600,
    asker: str | None = "ada@x.y",
    keep: Sequence[Link] = (),
    own_host: str = "",
) -> AsyncGenerator[HostFollower]:
    follower = HostFollower(
        HostFollower.Config(wait_sec=wait_sec, retry_sec=retry_sec),
        client=client,
        url="https://relay.example",
        api_key="assistant-key",
        asker=asker,
        remote=remote,
        keep=keep,
        own_host=own_host,
    )
    try:
        async with follower:
            yield follower
    finally:
        # The waits it left held end now, so no thread outlives the test.
        client.change(closed=True)


async def _until(
    remote: RemoteSession,
    name: str,
    holds: Callable[[bool], bool] = lambda up: up,
) -> None:
    endpoint = remote.endpoints()[name]
    async with asyncio.timeout(5):
        await endpoint.until(lambda e: holds(e.up))


@pytest.mark.asyncio
async def test_it_follows_the_hosts_the_person_may_use() -> None:
    client = _Client(
        hosts=[
            _host("gx10"),
            _host("asleep", status="offline"),
            _host("gone", status="revoked"),
            _host("bare", tools=False),
            _host("fresh", confirmed=False),
            _host("laptop"),
        ],
    )
    async with RemoteSession([]) as remote, _follower(client, remote):
        # Whether a host is up is its link's to find; the listing says who may use it.
        assert list(remote.endpoints()) == ["gx10", "asleep", "laptop"]
        # Each is reached through the server it was listed by.
        assert {e.where for e in remote.endpoints().values()} == {
            "https://relay.example",
        }
        assert client.asked == ["ada@x.y"]

        client.bump([_host("laptop"), _host("mini")])
        await _listed(client, 2)

        assert list(remote.endpoints()) == ["laptop", "mini"]


@pytest.mark.asyncio
async def test_with_no_asker_it_follows_the_keys_own_account_and_dials_as_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client(hosts=[_host("gx10")])
    async with (
        _daemons(monkeypatch, "gx10") as daemons,
        RemoteSession([], retry_sec=0.01) as remote,
        _follower(client, remote, asker=None),
    ):
        await _until(remote, "gx10")

    assert (client.asked, client.waited[0]) == ([None], (None, None))
    assert daemons.dialed == [("https://relay.example", "assistant-key", "gx10", None)]


@pytest.mark.asyncio
async def test_sagents_own_links_are_kept_beside_the_hosts_and_through_a_refusal() -> (
    None
):
    # As this machine's own link, which this test never reaches.
    own = Link(dial=_never_answered, name="this-machine", where="local")
    client = _Client(hosts=[_host("gx10")])
    async with (
        RemoteSession([], retry_sec=3600) as remote,
        _follower(client, remote, keep=[own], retry_sec=0.01),
    ):
        assert list(remote.endpoints()) == [own.name, "gx10"]

        client.bump([_host("mini")])
        await _listed(client, 2)
        assert list(remote.endpoints()) == [own.name, "mini"]

        client.change(refusing=401)
        await _waiting(client, 3)
        assert list(remote.endpoints()) == [own.name]


@pytest.mark.asyncio
async def test_the_host_the_agent_runs_on_is_left_to_its_local_link() -> None:
    client = _Client(hosts=[_host("gx10"), _host("laptop")])
    async with (
        RemoteSession([]) as remote,
        _follower(client, remote, own_host="laptop"),
    ):
        assert list(remote.endpoints()) == ["gx10"]


@pytest.mark.asyncio
async def test_a_host_that_joins_mid_session_shows_up_without_a_poll() -> None:
    client = _Client(hosts=[_host("gx10")])
    async with RemoteSession([]) as remote, _follower(client, remote):
        await _waiting(client, 1)
        started = asyncio.get_running_loop().time()

        client.bump([_host("gx10"), _host("laptop")])
        await _listed(client, 2)

        assert list(remote.endpoints()) == ["gx10", "laptop"]
        # Its waits hold for an hour, and the host came in well under a second.
        assert asyncio.get_running_loop().time() - started < 1
        assert client.waited == [(None, "ada@x.y"), (7, "ada@x.y"), (8, "ada@x.y")]


@pytest.mark.asyncio
async def test_an_idle_follower_asks_once_per_wait_and_lists_nothing_more() -> None:
    client = _Client(hosts=[_host("gx10")])
    # Held for no time at all: each wait on an unchanged list ends at its hold's end.
    async with RemoteSession([]) as remote, _follower(client, remote, wait_sec=0):
        await client.until(lambda c: len(c.held) >= 4)

        assert client.asked == ["ada@x.y"]
        # The first asks for the version at once; each after it is held to its end.
        assert client.waited[:4] == [(None, "ada@x.y")] + [(7, "ada@x.y")] * 3
        assert client.held[1:4] == [True] * 3


@pytest.mark.asyncio
async def test_a_restarted_server_converges_on_its_own_version() -> None:
    client = _Client(hosts=[_host("gx10")])
    async with RemoteSession([]) as remote, _follower(client, remote):
        await _waiting(client, 1)

        # A restart forgets the versions: the new process holds one of its own.
        client.change(version=4_000, hosts=[_host("mini")])
        await _listed(client, 2)

        assert list(remote.endpoints()) == ["mini"]
        assert client.waited[-1] == (4_000, "ada@x.y")


@pytest.mark.asyncio
async def test_an_outage_keeps_the_links_and_asks_again_only_after_its_delay(
    slept: list[float],
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _Client(hosts=[_host("gx10")])
    async with RemoteSession([]) as remote, _follower(client, remote, retry_sec=0.01):
        await _waiting(client, 1)

        client.change(down=True)
        await _waiting(client, 4)

        assert list(remote.endpoints()) == ["gx10"]
        assert client.asked == ["ada@x.y"]
        # Three asks into the outage, each after the delay: not a tight loop.
        asked_again = len(client.waited) - 2
        assert asked_again >= 3
        assert slept[:asked_again] == [0.01] * asked_again
        # Each failure is said, with why, before its delay.
        said = [
            record.getMessage()
            for record in caplog.records
            if record.name == host_follower.__name__
        ]
        assert (
            said[:asked_again]
            == [
                "host list failed: server unreachable",
            ]
            * asked_again
        )
        client.change(down=False)
        client.bump([])
        await _listed(client, 2)
        assert list(remote.endpoints()) == []


@pytest.mark.parametrize(
    ("status", "code"),
    [(403, "terminal_forbidden"), (401, "")],
    ids=["forbidden", "revoked-key"],
)
@pytest.mark.asyncio
async def test_the_servers_refusal_ends_the_links_and_they_come_back_with_access(
    status: int,
    code: str,
) -> None:
    client = _Client(hosts=[_host("gx10")])
    async with RemoteSession([]) as remote, _follower(client, remote, retry_sec=0.01):
        await _waiting(client, 1)

        client.change(refusing=status, refusing_code=code)
        await _waiting(client, 2)
        assert list(remote.endpoints()) == []
        # The wait after the refused one names no version at all.
        assert client.waited[2] == (None, "ada@x.y")

        # Refused, it no longer holds a version, so its next wait answers at once.
        client.change(refusing=None)
        await _listed(client, 2)
        assert list(remote.endpoints()) == ["gx10"]


@pytest.mark.parametrize(
    ("status", "code"),
    [(503, ""), (403, ""), (403, "edge_says_no")],
)
@pytest.mark.asyncio
async def test_anything_but_the_servers_refusal_is_an_outage_and_keeps_the_links(
    status: int,
    code: str,
) -> None:
    """A 403 from the edge in front of the server must not end every running call."""
    client = _Client(hosts=[_host("gx10")])
    async with RemoteSession([]) as remote, _follower(client, remote, retry_sec=0.01):
        await _waiting(client, 1)

        client.change(refusing=status, refusing_code=code)
        await _waiting(client, 3)

        assert list(remote.endpoints()) == ["gx10"]


@pytest.mark.parametrize("failing", ["version", "listing"])
@pytest.mark.asyncio
async def test_a_first_listing_that_fails_raises(failing: str) -> None:
    client = _Client(hosts=[_host("gx10")])
    if failing == "version":
        client.down = True
    else:
        client.refusing = 503
    async with RemoteSession([]) as remote:
        with pytest.raises(ClientError):
            async with _follower(client, remote):
                pass
        assert list(remote.endpoints()) == []


@pytest.mark.asyncio
async def test_a_listing_with_no_change_disturbs_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client(hosts=[_host("gx10")])
    async with (
        _daemons(monkeypatch, "gx10") as daemons,
        RemoteSession([], retry_sec=0.01) as remote,
        _follower(client, remote),
    ):
        await _until(remote, "gx10")
        endpoint = remote.endpoints()["gx10"]
        endpoint.tools = {"Bash": _BASH.digest}
        running = asyncio.create_task(
            endpoint.call(
                agent="a",
                window="w",
                tool="Bash",
                arguments={"command": "sleep 99"},
            ),
        )
        async with asyncio.timeout(5):
            _ = await daemons.held.wait()

        client.bump([_host("gx10")])
        await _listed(client, 2)
        client.bump([_host("gx10")])
        await _listed(client, 3)

        assert remote.endpoints()["gx10"] is endpoint
        assert endpoint.up
        assert not running.done()
        assert daemons.connects == {"gx10": 1}
        assert daemons.dialed == [
            ("https://relay.example", "assistant-key", "gx10", "ada@x.y"),
        ]
        # The agent reads a host's results with sagent's own limit.
        assert daemons.limits == [MAX_LINE_BYTES]
        running.cancel()


@pytest.mark.asyncio
async def test_a_host_that_goes_is_removed_and_its_calls_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client(hosts=[_host("gx10"), _host("laptop")])
    async with (
        _daemons(monkeypatch, "gx10", "laptop", "mini") as daemons,
        RemoteSession([], retry_sec=0.01) as remote,
        _follower(client, remote),
    ):
        await _until(remote, "gx10")
        endpoint = remote.endpoints()["gx10"]
        endpoint.tools = {"Bash": _BASH.digest}
        waiting = asyncio.create_task(
            endpoint.call(
                agent="a",
                window="w",
                tool="Bash",
                arguments={"command": "sleep 99"},
            ),
        )
        async with asyncio.timeout(5):
            _ = await daemons.held.wait()

        client.bump([_host("laptop"), _host("mini")])
        await _listed(client, 2)

        assert list(remote.endpoints()) == ["laptop", "mini"]
        failure = await asyncio.wait_for(waiting, 5)
        assert "gx10 was removed from this session" in failure.text()
        await _until(remote, "mini")
        assert daemons.connects["mini"] == 1


@pytest.mark.asyncio
async def test_an_answer_whose_caller_left_is_dropped() -> None:
    loop = asyncio.get_running_loop()
    left, also_left = loop.create_future(), loop.create_future()
    _ = left.cancel()
    _ = also_left.cancel()

    host_follower._fail(left, ClientError("late"))
    host_follower._succeed(also_left, 7)

    assert left.cancelled()
    assert also_left.cancelled()


def test_an_answer_for_a_loop_that_closed_meanwhile_is_dropped() -> None:
    """A wait the server answers after the agent's loop closed raises nothing."""
    loop = asyncio.new_event_loop()
    loop.close()
    settled: list[bool] = []

    host_follower._settle(loop, lambda: settled.append(True))

    assert settled == []


@pytest.mark.asyncio
async def test_each_ask_runs_on_a_daemon_thread_named_for_it() -> None:
    """A wait the server still holds keeps no process from exiting."""
    client = _Client(hosts=[_host("gx10")])
    async with RemoteSession([]) as remote, _follower(client, remote):
        await _waiting(client, 1)
    # The version, the listing, and the wait on the version after it.
    assert client.threads == [("host-list", True)] * 3


async def _never_answered() -> Streams:
    """Dial a tool server that never answers."""
    _ = await asyncio.Event().wait()
    raise AssertionError("an unanswered dial returned")


async def _waiting(client: _Client, count: int) -> None:
    """Return once ``count`` waits have started past the first, the version's."""
    await client.until(lambda c: len(c.waited) >= count + 1)


# A wait starts only once the follower has set the links from what it last listed.
async def _listed(client: _Client, count: int) -> None:
    """Return once a wait has started after the ``count``-th listing was used."""
    await client.until(lambda c: count in c.listed_by)


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
