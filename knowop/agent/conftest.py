"""Fixtures and fakes the agent tests share.

Each test module keeps its own tests; what more than one reads -- a recording sink, a
scripted server, a fake CLI -- lives here, so no test module imports another.
"""

from __future__ import annotations

from collections import deque
from collections.abc import (
    AsyncIterator,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    Self,
    TextIO,
    TypedDict,
    cast,
    override,
)
from uuid import UUID, uuid4

import asyncio
import functools
import json
import os
import shutil
import sys
import threading
import time
import uuid

from trackinizer.client.client import Client
from trackinizer.client.errors import ClientError
from trackinizer.lib.agent.sessions import (
    claude,
    codex,
)
from trackinizer.lib.agent.sessions.tail import Tail
from trackinizer.lib.posix.host import HostSpec
from trackinizer.lib.posix.sockets import socket_address
from trackinizer.runs.asks import (
    Answer,
    Ask,
    Driver,
    DriverEvent,
)
from trackinizer.sagent.session.types import (
    AssistantMessage,
    EditResult,
    IncompleteRecord,
    SessionRecord,
    SystemMessage,
    UserMessage,
)
from trackinizer.types.session_records import SessionRecordRow
from trackinizer.wire.refs import UuidRef
from trackinizer.wire.wire_session_ir import (
    AppendRecordsRequest,
    AppendRecordsResponse,
    ManifestBody,
    PartBody,
    RecordBody,
    SlashCommandBody,
)
from trackinizer.wire.wire_sessions import (
    InboundDrainItem,
    SessionEnd,
    SessionEndResponse,
    SessionStart,
    SessionStartResponse,
)
from treekle import PlainTree, from_plain, immutable, json_dumps, json_loads

import httpx2
import pytest

from knowop.agent import (
    headless,
    session,
)
from knowop.agent.claude_stream import (
    ClaudeStreamDriver,
)
from knowop.agent.clicked_runs import (
    ClickedRuns,
)
from knowop.agent.codex_app import (
    CodexAppDriver,
)
from knowop.agent.detach import HeadlessSpec
from knowop.agent.headless import (
    HeadlessRun,
    HostState,
)
from knowop.agent.session import (
    RunClient,
    RunConfig,
    _drain_filesystem_loop,
    _Stats,
)
from knowop.agent.sink import (
    Sink,
)
from knowop.agent.tool_server import (
    ToolError,
)
from knowop.connect import daemon
from knowop.connect.conftest import _signal_process


if TYPE_CHECKING:
    from collections.abc import Coroutine, Iterable
    from datetime import datetime

    from trackinizer.client.conversation import PartReader, Segment
    from trackinizer.runs.adapters.custom_types import Adapter, Subagent
    from trackinizer.types.inquiries import Inquiry
    from trackinizer.types.streams import TraxRecord
    from trackinizer.wire.filters import Filter

    from knowop.agent.custom_types import Event
    from knowop.agent.slash import SlashCommand
    from knowop.connect.conftest import _Process

__all__ = [
    "ME",
    "_ANY_REQUEST",
    "_ASK_STEPS",
    "_BASH_INPUT",
    "_CANVAS",
    "_FAKE",
    "_INIT",
    "_PART",
    "_PLACED_SESSION",
    "_SCHEMA",
    "_SESSION",
    "_SMALL_PROXY_LIMIT",
    "_TESTDATA",
    "_THIS",
    "_THREAD",
    "_TURN",
    "_WAIT",
    "_WORKER_LOG",
    "_BatchClient",
    "_BlockingSink",
    "_ClaudeEvent",
    "_ClaudeFrame",
    "_Client",
    "_Event",
    "_FakeAdapter",
    "_FakeClient",
    "_FakeDriver",
    "_Frame",
    "_HeadlessRig",
    "_HeadlessWiring",
    "_NoServer",
    "_PartsClient",
    "_PlacedSink",
    "_PlacingClient",
    "_RecordingRelay",
    "_RecordingSink",
    "_RefusingClient",
    "_Rig",
    "_Run",
    "_ScriptedDriver",
    "_Server",
    "_ServerComesBackClient",
    "_SessionSink",
    "_Tree",
    "_TuiRelay",
    "_UploadOutageClient",
    "_WholeFileAdapter",
    "_Wiring",
    "_WiringRelay",
    "_ack",
    "_always_found",
    "_ask",
    "_asked",
    "_break",
    "_can_use_tool",
    "_claude_asked",
    "_claude_next",
    "_click",
    "_collect",
    "_command",
    "_direct_segments",
    "_drain_once",
    "_echo",
    "_headless_config",
    "_line_records",
    "_lines",
    "_next",
    "_real_pause",
    "_records",
    "_refuse",
    "_response",
    "_rest",
    "_runs",
    "_says",
    "_signal_app",
    "_spec",
    "_state",
    "_step",
    "_texts",
    "_uncut",
    "_wait_for_events",
    "_wire",
    "_wire_headless",
    "_wiring_drain",
]


@pytest.fixture
def local_session_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point both CLIs' session roots at temp dirs, and pin the codex version.

    Each adapter derives its own root, so isolating one leaves the other writing
    real rollouts into the operator's home, which its resume picker then offers
    and which makes a result depend on the host. The pinned version keeps a
    test from running the installed codex.
    """
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setattr(
        "knowop.agent.materialize._codex_cli_version",
        lambda: "0.999.0",
    )


_PART = Path("/sessions/a.jsonl")


_PROXY_LIMIT: Final = 8_000_000
"""The live proxy's request-body cap: Caddy's ``8MB`` counts decimal megabytes."""


_SMALL_PROXY_LIMIT: Final = 32_000
"""A scaled proxy bound for fast tests of oversized-record truncation."""


_WORKER_LOG: Final = (
    '\x1b[2K\r\x1b[36m[w3]\x1b[0m \x1b[2m22:45:01\x1b[0m\t{"step":120,"loss":0.25}'
    " \x1b[32m━━━━\x1b[0m\x1b[1A" + "\x00" * 28 + "\n"
)
"""One line of the worker logs a scientist read in bulk: a progress bar, then the
NUL padding of a preallocated log. Each control character takes six bytes on the
wire, which is how ~1 MB of output became an 11.39 MB record."""


class _BlockingSink(Sink):
    """A sink whose ``flush`` blocks until released, logging call spans.

    Records ``("op", "enter"/"exit")`` for each call so a test can prove the
    lock serializes overlapping calls from different threads: a serialized run
    never interleaves a second call's ``enter`` between the first call's
    ``enter`` and ``exit``.
    """

    def __init__(self, release: threading.Event) -> None:
        self._release = release
        self.entered = threading.Event()
        self.log: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    @property
    @override
    def session_id(self) -> UUID | None:
        return None

    @override
    def open(self) -> str | None:
        return None

    @override
    def set_cli_session_id(self, cli_session_id: str) -> None:
        del cli_session_id

    def _record(self, op: str, phase: str) -> None:
        with self._lock:
            self.log.append((op, phase))

    @override
    def restart(self, path: Path) -> None:
        del path
        self._record("restart", "enter")
        self._record("restart", "exit")

    @override
    def emit(self, adapter_name: str, event: Event) -> None:
        del adapter_name, event
        self._record("emit", "enter")
        self._record("emit", "exit")

    @override
    def emit_slash_command(self, command: SlashCommand, at: datetime) -> None:
        del command, at

    @override
    def emit_worktree_diff(self, records: Sequence[EditResult]) -> None:
        del records

    @override
    def flush(self) -> None:
        self._record("flush", "enter")
        self.entered.set()
        self._release.wait(5.0)
        self._record("flush", "exit")

    @override
    def pending(self) -> list[tuple[Path, RecordBody]]:
        return []

    @override
    def close(self) -> None:
        self._record("close", "enter")
        self._record("close", "exit")


class _Server:
    """Answers as Trackinizer does, behind a proxy with a byte limit."""

    def __init__(self, *, proxy_limit: int = _PROXY_LIMIT) -> None:
        self.session_id = uuid4()
        self.proxy_limit = proxy_limit
        self.uploads: list[bytes] = []
        self.down = False
        self.ended = False
        self.uploaded = threading.Event()

    def client(self) -> Client:
        """Return a real client whose requests reach this server and nothing else."""
        client = Client("http://server")
        client._http.close()  # noqa: SLF001 -- The client exposes no transport seam.
        client._http = httpx2.Client(  # noqa: SLF001 -- The client exposes no transport seam.
            base_url=client.base_url,
            transport=httpx2.MockTransport(self._answer),
        )
        return client

    def records(self) -> list[RecordBody]:
        """Return every record body uploaded, in order."""
        return [
            body
            for upload in self.uploads
            for body in AppendRecordsRequest.model_validate_json(upload).records
        ]

    def read_back(self) -> list[TraxRecord]:
        """Return every uploaded record as a server reader decodes it."""
        return [body.row(self.session_id, 0).record() for body in self.records()]

    def _answer(self, request: httpx2.Request) -> httpx2.Response:
        if self.down:
            return httpx2.Response(502, text="Bad Gateway")
        if len(request.content) > self.proxy_limit:
            return httpx2.Response(413, text="Request Entity Too Large")
        path = request.url.path
        if path.endswith("/records"):
            self.uploads.append(request.content)
            self.uploaded.set()
            return httpx2.Response(200, json={"written": 0, "skipped": 0})
        if path.endswith("/start"):
            return httpx2.Response(200, json={"id": str(self.session_id), "seq": 0})
        if path.endswith("/end"):
            self.ended = True
        return httpx2.Response(200, json={"id": str(self.session_id)})


def _uncut(record: TraxRecord) -> RecordBody:
    """Return the wire body ``record`` had before capture bounded records."""
    return RecordBody.of(
        SessionRecordRow.of(session_id=uuid4(), part=0, idx=0, record=record),
    )


_THIS: Final[Path] = Path(__file__).resolve()


type _Event = DriverEvent
_Tree = dict[str, PlainTree]
type _Frame = _Tree | str
type _Frames = Mapping[str, Sequence[_Frame]]


_FAKE: Final[Path] = _THIS.with_name("codex_app_fake.py")
_THREAD: Final = "thr_1"
_TURN: Final = "turn_1"


_REPLIES: Final[dict[str, _Tree]] = {
    "thread/start": {"thread": {"id": _THREAD}},
    "thread/resume": {"thread": {"id": _THREAD}},
    "turn/start": {"turn": {"id": _TURN}},
}


def _signal_app(processes: dict[int, _Process], pgid: int, signum: int) -> None:
    process = processes[pgid]
    if process.returncode is not None:
        raise ProcessLookupError
    if signum:
        _signal_process(process, signum)


def _amend() -> PlainTree:
    """Return a fresh amendment decision: ``to_plain`` writes a shared one as a ref."""
    return {"acceptWithExecpolicyAmendment": {"execpolicy_amendment": ["touch"]}}


def _command(request_id: int = 0, **extra: PlainTree) -> _Tree:
    return {
        "method": "item/commandExecution/requestApproval",
        "id": request_id,
        "params": {
            "kind": "command",
            "threadId": _THREAD,
            "turnId": _TURN,
            "itemId": "item_1",
            "startedAtMs": 1,
            "reason": "Outside the workspace",
            "command": "touch /x",
            "cwd": "/w",
            "availableDecisions": ["accept", _amend(), "cancel"],
            **extra,
        },
    }


class _Call(TypedDict):
    """A request sent by the driver to the app-server."""

    id: int
    method: str
    params: _Tree


class _Rig:
    """A driver wired to a scripted fake app-server, and the log of what it sent."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        replies: Mapping[str, PlainTree] | None = None,
        after: _Frames | None = None,
        on_response: _Frames | None = None,
        argv: tuple[str, ...] | None = None,
    ) -> None:
        self.log = tmp_path / "frames.jsonl"
        self.log.touch()
        script = tmp_path / "script.json"
        script.write_text(
            json_dumps(
                {
                    "log": str(self.log),
                    "replies": {**_REPLIES, **(replies or {})},
                    "after": after or {},
                    "on_response": on_response or {},
                },
            ),
        )
        self.cwd = tmp_path
        self.argv = argv or (sys.executable, str(_FAKE), str(script))
        self.driver = CodexAppDriver(argv=self.argv)

    async def start(
        self,
        *,
        resume: str = "",
        model: str = "",
        effort: str = "",
    ) -> str:
        return await self.driver.start(
            cwd=self.cwd,
            resume=resume,
            model=model,
            effort=effort,
        )

    def frames(self) -> list[_Tree]:
        return [
            from_plain(json_loads(line), _Tree)
            for line in self.log.read_text().splitlines()
        ]

    def called(self, method: str) -> list[_Call]:
        """Return the requests of ``method`` the driver sent, as whole frames."""
        return [
            from_plain(each, _Call)
            for each in self.frames()
            if each.get("method") == method
        ]

    def params(self, method: str) -> _Tree:
        """Return the params of the first request of ``method`` the driver sent."""
        return from_plain(self.called(method)[0]["params"], _Tree)

    def methods(self) -> list[str]:
        return [
            from_plain(each.get("method"), str, default="") for each in self.frames()
        ]

    def response(self, request_id: int) -> PlainTree:
        """Return the result the driver sent to the server request ``request_id``."""
        frame = next(
            each
            for each in self.frames()
            if each.get("id") == request_id and "method" not in each
        )
        return frame.get("result", frame.get("error"))

    def run[T](self, scenario: Coroutine[object, object, T]) -> T:
        """Run ``scenario`` to its end and reap the app-server, whatever it did."""
        return asyncio.run(_bounded(scenario, driver=self.driver))

    async def sent(self, frames: int) -> None:
        """Wait until the fake has logged ``frames`` frames."""
        while len(self.log.read_text().splitlines()) < frames:  # noqa: ASYNC110 -- The fake is another process; only its log file says what it got.
            # The fake is another process; only its log file says what it got.
            await asyncio.sleep(0.002)  # house-ignore[sleep] -- See above.


async def _bounded[T](
    scenario: Coroutine[object, object, T],
    *,
    driver: CodexAppDriver,
) -> T:
    """Run ``scenario`` for at most 10 s, then stop ``driver``, whatever it did."""
    try:
        return await asyncio.wait_for(scenario, 10)
    finally:
        await driver.stop()


async def _next(events: AsyncIterator[_Event]) -> _Event:
    return await asyncio.wait_for(anext(events), 5)


async def _ask(events: AsyncIterator[_Event]) -> Ask:
    """Return the next ask, skipping the state changes before it."""
    while True:
        event = await _next(events)
        if isinstance(event, Ask):
            return event


async def _collect(events: AsyncIterator[_Event]) -> list[_Event]:
    return [each async for each in events]


async def _asked(
    rig: _Rig,
    *,
    option: str,
    updated_input: dict[str, PlainTree] | None = None,
    stop_after: int = 5,
) -> tuple[Ask, list[_Event]]:
    """Run a turn that asks, answer its first ask, and return it with the events after."""
    await rig.start()
    events = rig.driver.events()
    await rig.driver.send_line("go")
    ask = await _ask(events)
    await rig.driver.answer(
        Answer(ask_id=ask.id, option=option, updated_input=updated_input, who="jp"),
    )
    await rig.sent(stop_after)
    await rig.driver.stop()
    return ask, await _collect(events)


type _ClaudeFrame = dict[str, PlainTree]


_BASH_INPUT: Final[PlainTree] = {
    "command": "touch ok.txt",
    "description": "Create empty file ok.txt",
}


def _can_use_tool(
    tool: str = "Bash",
    *,
    input: PlainTree = _BASH_INPUT,
    request_id: str = "r1",
    **fields: PlainTree,
) -> _ClaudeFrame:
    return {
        "type": "control_request",
        "request_id": request_id,
        "request": {
            "subtype": "can_use_tool",
            "tool_name": tool,
            "display_name": tool,
            "input": input,
            "tool_use_id": "toolu_1",
            **fields,
        },
    }


def _response(request_id: str, **body: PlainTree) -> _ClaudeFrame:
    return {
        "type": "control_response",
        "response": {"subtype": "success", "request_id": request_id, "response": body},
    }


_CLAUDE_FAKE: Final = """
import json, os, sys

script = json.load(open(os.environ["FAKE_SCRIPT"]))
log = open(os.environ["FAKE_LOG"], "a")
log.write(json.dumps({"argv": sys.argv[1:],
                      "env": os.environ.get("CLAUDE_CODE_EMIT_SESSION_STATE_EVENTS"),
                      "cwd": os.getcwd()}) + "\\n")
log.flush()


def matches(want, got):
    if isinstance(want, dict):
        return isinstance(got, dict) and all(
            key in got and matches(value, got[key]) for key, value in want.items()
        )
    return want == got


def answer(line):
    frame = json.loads(line)
    log.write(line)
    log.flush()
    for step in script:
        if matches(step["when"], frame):
            sys.stderr.write(step.get("stderr", ""))
            sys.stderr.flush()
            sys.stderr.buffer.write(bytes.fromhex(step.get("stderr_hex", "")))
            sys.stderr.flush()
            for emit in step["emit"]:
                text = emit if isinstance(emit, str) else json.dumps(emit)
                sys.stdout.write(text.replace("$request_id", frame.get("request_id", "")) + "\\n")
                sys.stdout.flush()
            sys.stdout.write(step.get("tail", ""))
            sys.stdout.flush()
            if "exit" in step:
                sys.exit(step["exit"])
            break


for line in sys.stdin:
    answer(line)
"""
"""A claude that answers from a script: each step names a frame to match and the frames
to print back, with ``$request_id`` filled in from the frame it matched."""


def _ack(**body: PlainTree) -> _ClaudeFrame:
    return _response("$request_id", **body)


def _step(
    when: _ClaudeFrame,
    *emit: _ClaudeFrame | str,
    **more: PlainTree,
) -> _ClaudeFrame:
    return {"when": when, "emit": list(emit), **more}


_INIT: Final = _step(
    {"request": {"subtype": "initialize"}},
    _ack(commands=[], session_state="idle"),
)
_ANY_REQUEST: Final = _step({"type": "control_request"}, _ack())


def _state(state: str) -> _ClaudeFrame:
    return {"type": "system", "subtype": "session_state_changed", "state": state}


def _says(text: str) -> _ClaudeFrame:
    return {"type": "user", "message": {"role": "user", "content": text}}


class _Run:
    """One driver over the scripted fake, and what the fake saw.

    Leaving the block stops the driver, so a test that fails half way leaves no fake
    claude behind.
    """

    def __init__(
        self,
        tmp_path: Path,
        *steps: _ClaudeFrame,
        permission_mode: str | None = None,
        request_timeout_sec: float = 2.0,
    ) -> None:
        self.tmp_path = tmp_path
        (tmp_path / "fake.py").write_text(_CLAUDE_FAKE)
        (tmp_path / "script.json").write_text(json.dumps(list(steps)))
        self.log = tmp_path / "log.jsonl"
        argv = (sys.executable, "-I", "-S", str(tmp_path / "fake.py"))
        env = {
            "FAKE_SCRIPT": str(tmp_path / "script.json"),
            "FAKE_LOG": str(self.log),
        }
        # A request nobody answers fails the test in seconds, not at the harness limit.
        self.driver = (
            ClaudeStreamDriver(
                argv=argv,
                env=env,
                request_timeout_sec=request_timeout_sec,
            )
            if permission_mode is None
            else ClaudeStreamDriver(
                argv=argv,
                env=env,
                permission_mode=permission_mode,
                request_timeout_sec=request_timeout_sec,
            )
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        del exc
        await self.driver.stop()

    async def start(
        self,
        *,
        resume: str = "",
        model: str = "",
        effort: str = "",
    ) -> str:
        return await self.driver.start(
            cwd=self.tmp_path,
            resume=resume,
            model=model,
            effort=effort,
        )

    def seen(self) -> list[_ClaudeFrame]:
        return [
            json_loads(line, dict[str, PlainTree])
            for line in self.log.read_text().splitlines()
        ]

    def sent(self) -> list[_ClaudeFrame]:
        """Return the frames the driver wrote, after the fake's own first line."""
        return self.seen()[1:]


type _ClaudeEvent = DriverEvent


async def _claude_next(events: AsyncIterator[_ClaudeEvent]) -> _ClaudeEvent:
    return await asyncio.wait_for(anext(events), 1)


async def _rest(driver: Driver) -> list[_ClaudeEvent]:
    async def drain() -> list[_ClaudeEvent]:
        return [each async for each in driver.events()]

    return await asyncio.wait_for(drain(), 2)


async def _claude_asked(run: _Run) -> AsyncIterator[_ClaudeEvent]:
    """Start the run, send "ask", and return its events with the ask taken."""
    _ = await run.start()
    events = run.driver.events()
    await run.driver.send_line("ask")
    _ = [await _claude_next(events), await _claude_next(events)]
    return events


_ASK_STEPS: Final = (
    _INIT,
    _step(_says("ask"), _state("requires_action"), _can_use_tool()),
    _step({"type": "control_response"}, _state("running"), _state("idle")),
    _step(
        {"request": {"subtype": "interrupt"}},
        {"type": "control_cancel_request", "request_id": "r1"},
        _ack(still_queued=[]),
    ),
    _ANY_REQUEST,
)


_WAIT: Final = 5.0


class _FakeDriver:
    """A scripted driver: it reports what it is pushed, and remembers what it is told."""

    def __init__(self, *, session: str = "s1", fail: Exception | None = None) -> None:
        self.session = session
        self.fail = fail
        self.started: tuple[Path, str, str, str] | None = None
        self.told: list[tuple[str, object]] = []
        self.stopped = False
        self.refuse: Exception | None = None
        self.hold_start: threading.Event | None = None
        self.starting = threading.Event()
        self.hang = False
        self.hanging = threading.Event()
        self.printed = ""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue = asyncio.Queue[DriverEvent | None]()

    def push(self, event: DriverEvent | None) -> None:
        """Report ``event`` from any thread; ``None`` ends the driver by itself."""
        loop = self._loop
        if loop is None:
            raise ValueError("Expected loop is not None.")
        loop.call_soon_threadsafe(self._queue.put_nowait, event)

    async def start(self, *, cwd: Path, resume: str, model: str, effort: str) -> str:
        self._loop = asyncio.get_running_loop()
        self.starting.set()
        if self.hold_start is not None:
            _ = await asyncio.to_thread(self.hold_start.wait, _WAIT)
        if self.fail is not None:
            raise self.fail
        self.started = (cwd, resume, model, effort)
        return self.session

    async def send_line(self, text: str) -> None:
        await self._tell("line", text)

    async def command(self, verb: str, args: str) -> str:
        await self._tell("command", (verb, args))
        return self.printed

    async def answer(self, answer: Answer) -> None:
        await self._tell("answer", answer)

    async def interrupt(self) -> None:
        await self._tell("interrupt", None)

    async def settings(self, *, model: str, effort: str) -> None:
        await self._tell("settings", (model, effort))

    async def stop(self) -> None:
        self.stopped = True
        await self._queue.put(None)

    async def events(self) -> AsyncIterator[DriverEvent]:
        while (event := await self._queue.get()) is not None:
            yield event

    async def _tell(self, what: str, detail: object) -> None:
        if self.refuse is not None:
            raise self.refuse
        if self.hang:
            self.hanging.set()
            _ = await asyncio.Event().wait()
        self.told.append((what, detail))


type _TuiMode = Literal["listens", "late", "silent", "crashes", "fails"]


class _FakeTui:
    """A takeover terminal: it listens, then runs until it is ended.

    ``late`` listens a moment after it starts, ``silent`` never listens, ``crashes``
    fails after it listens, and ``fails`` fails before it listens.
    """

    def __init__(
        self,
        *,
        listening: Callable[[], None],
        exits_with: int | None,
        mode: _TuiMode,
    ) -> None:
        self._listening = listening
        self._exits_with = exits_with
        self._mode = mode
        self.lines: list[str] = []
        self.ended = threading.Event()
        self.over = threading.Event()

    def run(self) -> int:
        if self._exits_with is not None:
            return self._exits_with
        if self._mode == "fails":
            raise OSError("out of ptys")
        if self._mode == "late":
            _ = self.ended.wait(0.03)
        if self._mode != "silent":
            self._listening()
        if self._mode == "crashes":
            raise RuntimeError("pty died")
        _ = self.ended.wait(_WAIT)
        self.over.set()
        return 0

    def terminate(self) -> None:
        self.ended.set()

    def submit(self, text: str) -> None:
        self.lines.append(text)

    def finish(self) -> None:
        """Quit the way a person does: the terminal ends by itself."""
        self.ended.set()


@dataclass(slots=True, kw_only=True)
class _HeadlessRig:
    """A headless run in a thread, with its fake drivers and terminal."""

    drivers: list[_FakeDriver] = field(default_factory=lambda: [_FakeDriver()])
    tui_exits: tuple[int | None, ...] = ()
    tui_mode: _TuiMode = "listens"
    no_tui: Exception | None = None
    resume: str = ""
    control: Path | None = None
    notify_takeover: bool = True
    records: list[SystemMessage] = field(default_factory=list[SystemMessage])
    sessions: list[str] = field(default_factory=list[str])
    states: list[HostState] = field(default_factory=list[HostState])
    taken: list[str] = field(default_factory=list[str])
    tuis: list[_FakeTui] = field(default_factory=list[_FakeTui])
    tui_for: list[str] = field(default_factory=list[str])
    status: list[int] = field(default_factory=list[int])
    started: threading.Event = field(default_factory=threading.Event)
    listening: threading.Event = field(default_factory=threading.Event)
    run: HeadlessRun = field(init=False)
    _thread: threading.Thread = field(init=False)
    _made: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.run = HeadlessRun(
            cli="codex",
            cwd=Path("/work"),
            resume=self.resume,
            model="m1",
            effort="high",
            make_driver=self._make_driver,
            make_tui=self._make_tui,
            record=self.records.append,
            on_session=self.sessions.append,
            on_listening=self.listening.set,
            on_state=self.states.append,
            on_takeover=self.taken.append if self.notify_takeover else None,
            control=self.control,
        )
        self._thread = threading.Thread(
            target=lambda: self.status.append(
                self.run.run(on_started=self.started.set),
            ),
            daemon=True,
        )

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.run.terminate()
        self._thread.join(_WAIT)
        if self._thread.is_alive():
            raise ValueError("the run did not end")

    def wait_started(self) -> None:
        if not self.started.wait(_WAIT):
            raise ValueError("the run did not start")

    def join(self) -> int:
        self._thread.join(_WAIT)
        if self._thread.is_alive():
            raise ValueError("the run did not end")
        return self.status[0]

    def _make_driver(self) -> Driver:
        driver = self.drivers[self._made]
        self._made += 1
        return cast("Driver", driver)

    def _make_tui(self, *, session_id: str, listening: Callable[[], None]) -> _FakeTui:
        if self.no_tui is not None:
            raise self.no_tui
        self.tui_for.append(session_id)
        nth = len(self.tuis)
        tui = _FakeTui(
            listening=listening,
            exits_with=self.tui_exits[nth] if nth < len(self.tui_exits) else None,
            mode=self.tui_mode,
        )
        self.tuis.append(tui)
        return tui


# Captured transcripts, kept beside the readers that parse them. trackinizer's wheel
# leaves them out, so the public export carries copies beside this module instead.
_TESTDATA: Final = _THIS.parent / "testdata" / "sessions"


# Fed a line at a time through :class:`Tail`, which is what capture does: the reader
# PULLS and the runner PUSHES, so driving the pull side directly would exercise a path
# the runner never takes.
def _records(name: str) -> tuple[list[TraxRecord], Mapping[str, PlainTree]]:
    """Return a corpus fixture's records and how its file spells its bytes."""
    records, encoding = _read(name)
    return list(records), encoding


@functools.cache
def _read(name: str) -> tuple[tuple[TraxRecord, ...], Mapping[str, PlainTree]]:
    """Read a corpus fixture as capture does; see :func:`_records`."""
    reader = Tail((codex if name.startswith("codex") else claude).normalize)
    out: list[TraxRecord] = []
    with (_TESTDATA / name).open(encoding="utf-8") as handle:
        for line in handle:
            out.extend(reader.feed(line))
    out.extend(reader.close())
    return tuple(out), immutable(reader.encoding)


class _FakeClient:
    """Serves one stored session.

    It has no method that stamps the CLI's id onto the row, so a resume that tried to
    would fail here: that absence is what the attach tests rely on.
    """

    def __init__(self, name: str, *, session_format: str) -> None:
        records, self.encoding = _records(name)
        self.records: Sequence[TraxRecord] = records
        self._format = session_format

    def read_session_parts(self, session_id: UUID) -> list[PartBody]:
        del session_id
        return [
            PartBody(
                part=0,
                name="s.jsonl",
                format=self._format,
                records=len(self.records),
                metadata=self.encoding,
                ir_id=uuid4(),
            ),
        ]

    def read_session_records(
        self,
        session_id: UUID,
        *,
        part: int = 0,
        **_: object,
    ) -> list[RecordBody]:
        return [
            RecordBody.of(
                SessionRecordRow.of(
                    session_id=session_id,
                    part=part,
                    idx=idx,
                    record=record,
                ),
            )
            for idx, record in enumerate(self.records)
        ]

    def get_inquiry(
        self,
        ref: object,
    ) -> tuple[Inquiry.InquiryKind, UUID, dict[str, PlainTree]]:
        """Return the session's detail view, which names its seq."""
        assert isinstance(ref, UuidRef)
        return "AgentSession", ref.uuid, {"self": {"seq": 7}}


class _PartsClient:
    """Serves a session of several parts, each one a file of its own format."""

    def __init__(
        self,
        parts: Mapping[int, tuple[str, Sequence[SessionRecord]]],
    ) -> None:
        self._parts = parts

    def read_session_parts(self, session_id: UUID) -> list[PartBody]:
        del session_id
        return [
            PartBody(
                part=number,
                name=f"{number}.jsonl",
                format=fmt,
                records=len(records),
            )
            for number, (fmt, records) in self._parts.items()
        ]

    def read_session_records(
        self,
        session_id: UUID,
        *,
        part: int = 0,
        **_: object,
    ) -> list[RecordBody]:
        return [
            RecordBody.of(
                SessionRecordRow.of(
                    session_id=session_id,
                    part=part,
                    idx=idx,
                    record=record,
                ),
            )
            for idx, record in enumerate(self._parts[part][1])
        ]

    def get_inquiry(
        self,
        ref: object,
    ) -> tuple[Inquiry.InquiryKind, UUID, dict[str, PlainTree]]:
        """Return the session's detail view, which names its seq."""
        assert isinstance(ref, UuidRef)
        return "AgentSession", ref.uuid, {"self": {"seq": 7}}


def _direct_segments(
    segments: list[Segment],
    client: PartReader,
    session_id: UUID,
) -> list[Segment]:
    del client, session_id
    return segments


_CANVAS = UUID(int=5)
_SESSION = UUID(int=1)


class _Client:
    def __init__(self, *, fail: bool = False) -> None:
        self.asked: list[tuple[str, object]] = []
        self._fail = fail

    def post(self, path: str, *, body: object = None) -> dict[str, str]:
        self.asked.append((path, body))
        if self._fail:
            raise ClientError("404 Not Found")
        return {"workspace_id": str(UUID(int=2))}


def _echo(arguments: Mapping[str, object]) -> str:
    return f"said {arguments['text']}"


def _refuse(arguments: Mapping[str, object]) -> str:
    del arguments
    raise ToolError("not that one")


def _break(arguments: Mapping[str, object]) -> str:
    del arguments
    raise RuntimeError("boom")


_SCHEMA: Final = {"type": "object", "properties": {"text": {"type": "string"}}}


ME: Final = "dan@x"


def _runs(
    cwd: Path,
    records: list[SystemMessage],
    *,
    env: Mapping[str, str] | None = None,
    shell_path: str = "/bin/sh",
    publish: Callable[[str, str], None] | None = None,
) -> ClickedRuns:
    """Return runs whose records go to ``records``, looking at output every 5 ms."""
    return ClickedRuns(
        record=records.append,
        cwd=cwd,
        env={"PATH": "/usr/bin:/bin", **(env or {})},
        shell_path=shell_path,
        publish=publish,
        publish_sec=0.005,
    )


def _click(
    run_id: str,
    command: str,
    *,
    stop: bool = False,
    by: str = ME,
) -> InboundDrainItem:
    """Return the ``run`` message a click on ``command`` drains as."""
    return InboundDrainItem(
        text=command,
        kind="run",
        payload={"run_id": run_id, "block": "line-1:0", "stop": stop},
        source=by,
    )


def _lines(runs: ClickedRuns) -> list[InboundDrainItem]:
    """Return the next lines due, waiting for them while a command runs."""
    deadline = time.monotonic() + 5.0
    while not (lines := runs.take()) and runs.running():
        if time.monotonic() > deadline:
            raise AssertionError("A clicked command did not end within 5 s.")
        # Clicked commands are other processes; only their output says they ran.
        time.sleep(0.001)  # house-ignore[sleep] -- See above.
    return lines


class _RecordingSink(Sink):
    """Collects emitted records for assertions.

    SUBCLASSES ``Sink`` rather than matching it structurally -- which is the
    Protocol's own rule, and load-bearing here: ``feed`` is concrete on the
    Protocol and reaches ``_readers``, so a structural stand-in gets the
    method without the state it needs.

    Positions are per-FILE, exactly as the real sinks derive them, so a test
    can assert each part numbers its records from 0.
    """

    def __init__(self) -> None:
        self.events: list[tuple[int, str, Event]] = []
        self.slash: list[tuple[SlashCommand, datetime]] = []
        self.diffs: list[list[EditResult]] = []
        self.closed = False
        self.flushes = 0
        self.cli_session_ids: list[str] = []
        self.changed = threading.Condition()
        """Notified as each event is recorded; the drain records from its thread."""
        self._next_idx: dict[Path, int] = {}

    @property
    @override
    def session_id(self) -> UUID | None:
        return None

    @override
    def open(self) -> str | None:
        return None

    @override
    def set_cli_session_id(self, cli_session_id: str) -> None:
        self.cli_session_ids.append(cli_session_id)

    @override
    def restart(self, path: Path) -> None:
        self._next_idx[path] = 0

    @override
    def pass_over(self, path: Path) -> None:
        self._next_idx[path] = self._next_idx.get(path, 0) + 1

    @override
    def emit(self, adapter_name: str, event: Event) -> None:
        idx = self._next_idx.get(event.path, 0)
        self._next_idx[event.path] = idx + 1
        with self.changed:
            self.events.append((idx, adapter_name, event))
            self.changed.notify_all()

    @override
    def emit_slash_command(self, command: SlashCommand, at: datetime) -> None:
        self.slash.append((command, at))

    @override
    def emit_worktree_diff(self, records: Sequence[EditResult]) -> None:
        self.diffs.append(list(records))

    @override
    def flush(self) -> None:
        self.flushes += 1

    @override
    def pending(self) -> list[tuple[Path, RecordBody]]:
        return []

    @override
    def close(self) -> None:
        self.closed = True


# Reads the record's own field rather than a projection, so a test failure names the
# turn that went missing rather than a search string.
#
# Records with no prose are SKIPPED, not asserted against: one native line legitimately
# produces several records -- a claude user line also emits the ``TurnContext`` that
# applies to it -- and the drain tests are about which turns were captured and in what
# order, not how many records a dialect spends saying so.
def _texts(sink: _RecordingSink) -> list[str]:
    """Return the prose of each captured record that carries any, in emit order."""
    out: list[str] = []
    for _idx, _name, event in sink.events:
        match event.record:
            case IncompleteRecord() as record:
                out.append(record.text)
            case UserMessage() | AssistantMessage() as record:
                out.append(record.content or "")
            case _:
                continue
    return out


# The fake dialect the file-drain tests use: it makes "which turns were captured, in
# what order" the only thing under test, with no provider grammar in the way -- so it
# states no opening ``TurnContext`` either, which a real dialect does and which would
# show up in every count here.
def _line_records(stream: TextIO) -> Iterator[SessionRecord]:
    """One record per line, carrying the line's own text."""
    for line in stream:
        yield UserMessage(content=line.rstrip("\n"))


# Mirrors gemini, which rewrites ONE JSON object in place: the runner re-feeds the whole
# body on each change and marks the chunk a restart, so each record lands back on the
# position it already held rather than the reader having to remember what it emitted.
def _document_records(stream: TextIO) -> Iterator[SessionRecord]:
    """Every message a whole document holds, re-read from its start."""
    obj = json_loads(stream.read(), dict[str, object])
    for text in from_plain(obj.get("messages"), list[str], default=list[str]()):
        yield UserMessage(content=text)


class _FakeAdapter:
    """Treats every ``*.jsonl`` line as one ``UserMessage`` record."""

    name: str = "fake"
    cli_binary: str = "fake"
    whole_file: bool = False
    parent_session_env: frozenset[str] = frozenset[str]()
    session_switch_commands: frozenset[str] = frozenset({"clear"})

    def __init__(self, root: Path) -> None:
        self._root = root
        self.subagents: dict[Path, Subagent] = {}

    def session_dirs(self) -> Iterable[Path]:
        return (self._root,)

    def matches_session_file(self, path: Path) -> bool:
        return path.suffix == ".jsonl"

    def session_scope(self) -> Path | None:
        return None

    def session_id_from_path(self, path: Path) -> str | None:
        del path
        return None

    def resumed_session(self, cli_args: tuple[str, ...]) -> str | None:
        del cli_args
        return None

    def subagent(self, path: Path) -> Subagent | None:
        """Return the sub-agent ``path`` names, as the run's CLI declares one."""
        return self.subagents.get(path)

    def reader(self) -> Tail[TraxRecord]:
        return Tail(_line_records)


class _WholeFileAdapter(_FakeAdapter):
    """A whole-file adapter: the runner must feed it the entire file body.

    Mirrors gemini, which rewrites one JSON object in place rather than
    appending lines.
    """

    name: str = "wholefile"
    cli_binary: str = "wholefile"
    whole_file: bool = True

    @override
    def matches_session_file(self, path: Path) -> bool:
        return path.suffix == ".json"

    @override
    def reader(self) -> Tail[TraxRecord]:
        return Tail(_document_records, whole_file=True)


def _always_found(
    cmd: str,
    mode: int = os.F_OK | os.X_OK,
    path: str | None = None,
) -> str:
    """Return a ``shutil.which`` that resolves anything: the binary is never exec'd."""
    del mode, path
    return f"/bin/{cmd}"


# Delivery is the handshake between a test's writes: a second write issued once the
# first's turn has been EMITTED cannot be folded into the same wake, which no fixed
# pause can promise. Returns rather than asserts on timeout so the caller's own
# assertion names what is missing.
def _wait_for_events(sink: _RecordingSink, count: int) -> None:
    """Block until ``sink`` holds ``count`` events, or give up after 3s."""
    with sink.changed:
        _ = sink.changed.wait_for(lambda: len(sink.events) >= count, timeout=3.0)


# The drain is wake-driven, so a test cannot call one scan and inspect the result: it
# starts the loop, writes, and waits for delivery. A ``write`` that handshakes on
# delivery mid-way passes the ``sink`` it will watch.
def _drain_once(
    adapter: Adapter,
    write: Callable[[], object],
    *,
    baseline: frozenset[Path] = frozenset(),
    expected: int = 1,
    sink: _RecordingSink | None = None,
    owner: session._SessionOwner | None = None,
) -> tuple[_Stats, _RecordingSink]:
    """Run the real drain, perform ``write``, and collect what it captured."""
    sink = sink or _RecordingSink()
    stats = _Stats()
    stop = threading.Event()
    armed = threading.Event()

    def _run() -> None:
        _drain_filesystem_loop(
            adapter,
            sink,
            stats,
            RunConfig(cli_name=adapter.name),
            stop,
            baseline=baseline,
            slash_queue=deque(),
            armed=armed,
            owner=owner,
        )

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    try:
        if not armed.wait(5.0):
            raise ValueError("session-log watch did not arm")
        write()
        _wait_for_events(sink, expected)
    finally:
        stop.set()
        worker.join(timeout=5.0)
        if worker.is_alive():
            raise ValueError("session drain did not stop")
    return stats, sink


class _RefusingClient:
    """A client whose every drain fails with ``error``, as the server refuses it."""

    def __init__(
        self,
        error: ClientError,
        *,
        stop_after: int = 0,
        stop: threading.Event | None = None,
    ) -> None:
        self._error = error
        self._stop_after = stop_after
        self._stop = stop
        self.drains = 0

    def drain_inbound(
        self,
        session_id: UUID,
        *,
        wait_sec: float = 0.0,
        **where: str | bool | None,
    ) -> list[InboundDrainItem]:
        del session_id, wait_sec, where
        self.drains += 1
        if self._stop is not None and self.drains >= self._stop_after:
            self._stop.set()
        raise self._error


class _NoServer:
    """A server the test keeps offline: every request the run makes fails it."""

    base_url = "http://no-server.test"

    def session_start(self, body: SessionStart) -> SessionStartResponse:
        del body
        raise AssertionError("session_start reached the offline server.")

    def append_records(
        self,
        session_id: UUID,
        *,
        name: str = "",
        manifest: ManifestBody | None = None,
        records: Sequence[RecordBody] = (),
        restart: bool = False,
        slash_commands: Sequence[SlashCommandBody] = (),
    ) -> AppendRecordsResponse:
        del session_id, name, manifest, records, restart, slash_commands
        raise AssertionError("append_records reached the offline server.")

    def session_end(
        self,
        session_id: UUID,
        body: SessionEnd | None = None,
    ) -> SessionEndResponse:
        del session_id, body
        raise AssertionError("session_end reached the offline server.")

    def post(self, path: str, *, body: object = None) -> PlainTree:
        del path, body
        raise AssertionError("post reached the offline server.")

    def get(
        self,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
    ) -> PlainTree:
        del path, params
        raise AssertionError("get reached the offline server.")

    def drain_inbound(
        self,
        session_id: UUID,
        *,
        wait_sec: float = 0.0,
        **where: str | bool | None,
    ) -> list[InboundDrainItem]:
        del session_id, wait_sec, where
        raise AssertionError("drain_inbound reached the offline server.")

    def read_session_parts(self, session_id: UUID) -> list[PartBody]:
        del session_id
        raise AssertionError("read_session_parts reached the offline server.")

    def read_session_records(
        self,
        session_id: UUID,
        *,
        part: int = 0,
        plaintext_only: bool = False,
    ) -> list[RecordBody]:
        del session_id, part, plaintext_only
        raise AssertionError("read_session_records reached the offline server.")

    def get_inquiry(
        self,
        ref: object,
    ) -> tuple[Inquiry.InquiryKind, UUID, dict[str, PlainTree]]:
        del ref
        raise AssertionError("get_inquiry reached the offline server.")

    def list_kind(
        self,
        kind: Inquiry.InquiryKind,
        *,
        filters: Sequence[Filter] = (),
    ) -> list[dict[str, PlainTree]]:
        del kind, filters
        raise AssertionError("list_kind reached the offline server.")

    def submit(self, kind: Inquiry.InquiryKind, body: Mapping[str, object]) -> UUID:
        del kind, body
        raise AssertionError("submit reached the offline server.")

    def add_edge(
        self,
        from_id: UUID,
        to_id: UUID,
        edge_kind: str,
        *,
        actor: str,
        note: str = "",
    ) -> object:
        del from_id, to_id, edge_kind, actor, note
        raise AssertionError("add_edge reached the offline server.")

    def transition_app(
        self,
        target_id: UUID,
        *,
        expected_from: dict[str, object] | None,
        to: dict[str, object] | None,
        actor: str,
    ) -> None:
        del target_id, expected_from, to, actor
        raise AssertionError("transition_app reached the offline server.")

    def transition_status(
        self,
        target_id: UUID,
        *,
        expected_from: str,
        to: str,
        actor: str,
    ) -> None:
        del target_id, expected_from, to, actor
        raise AssertionError("transition_status reached the offline server.")

    def patch(self, path: str, *, body: object = None) -> PlainTree:
        del path, body
        raise AssertionError("patch reached the offline server.")


@dataclass(slots=True, kw_only=True)
class _Wiring:
    """What ``_spawn_and_drain`` handed each collaborator."""

    argv: list[str] = field(default_factory=list[str])
    relay: dict[str, object] = field(default_factory=dict[str, object])
    relay_instance: object = None
    drain: dict[str, object] = field(default_factory=dict[str, object])
    inbound: list[object] = field(default_factory=list[object])
    daemons: list[bool] = field(default_factory=list[bool])
    thread_names: list[str] = field(default_factory=list[str])


class _WiringRelay:
    """A relay that records how it was built, plays ``output``, and exits ``rc``."""

    def __init__(
        self,
        argv: list[str],
        *,
        wiring: _Wiring,
        rc: int,
        output: bytes,
        **kwargs: object,
    ) -> None:
        wiring.argv = list(argv)
        wiring.relay = kwargs
        wiring.relay_instance = self
        self._rc = rc
        self._output = output
        self._on_output = kwargs["on_output"]

    def run(self) -> int:
        if callable(self._on_output) and self._output:
            _ = self._on_output(self._output)
        return self._rc


def _wiring_drain(
    wiring: _Wiring,
    *args: object,
    armed: threading.Event,
    **kwargs: object,
) -> None:
    """Record the drain's inputs, and that it runs as a daemon; release the spawn."""
    wiring.drain = {"args": args, **kwargs}
    wiring.daemons.append(threading.current_thread().daemon)
    wiring.thread_names.append(threading.current_thread().name)
    armed.set()


def _wiring_inbound(wiring: _Wiring, *args: object, **kwargs: object) -> None:
    """Record inbound delivery's inputs, and that it runs as a daemon."""
    wiring.inbound = [*args, kwargs]
    wiring.daemons.append(threading.current_thread().daemon)
    wiring.thread_names.append(threading.current_thread().name)


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config: RunConfig,
    adapter: Adapter,
    *,
    sink: Sink | None = None,
    stats: _Stats | None = None,
    rc: int = 0,
    output: bytes = b"",
    relay: type[_WiringRelay] = _WiringRelay,
    wiring: _Wiring | None = None,
) -> tuple[_Wiring, int]:
    """Run ``_spawn_and_drain`` against recording collaborators."""
    wiring = wiring or _Wiring()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setattr(
        session,
        "ThreadedRelay",
        partial(relay, wiring=wiring, rc=rc, output=output),
    )
    monkeypatch.setattr(
        session,
        "_drain_filesystem_loop",
        partial(_wiring_drain, wiring),
    )
    monkeypatch.setattr(session, "_inbound_poll_loop", partial(_wiring_inbound, wiring))
    monkeypatch.setattr(shutil, "which", _always_found)
    status = session._spawn_and_drain(  # noqa: SLF001 -- The run's wiring under test.
        replace(config, quiesce_seconds=0.0),
        adapter,
        sink or _RecordingSink(),
        stats or _Stats(),
    )
    return wiring, status


class _PlacedSink(_RecordingSink):
    """A sink whose server opened session ``_PLACED_SESSION``."""

    @property
    @override
    def session_id(self) -> UUID:
        return _PLACED_SESSION


_PLACED_SESSION = UUID("00000000-0000-0000-0000-0000000000a1")


class _PlacingClient(_NoServer):
    """A client whose server places a session, or refuses to; records what it was asked."""

    def __init__(self, *, refuse: bool = False) -> None:
        self.asked: list[tuple[str, object]] = []
        self._refuse = refuse

    @override
    def post(self, path: str, *, body: object = None) -> dict[str, str]:
        self.asked.append((path, body))
        if self._refuse:
            raise ClientError("404 Not Found")
        return {
            "workspace_id": str(UUID(int=7)),
            "canvas_id": str(UUID(int=8)),
        }


class _BatchClient:
    """Returns one batch of messages, then nothing (the queue is drained)."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.drains = 0

    def drain_inbound(
        self,
        session_id: UUID,
        *,
        wait_sec: float = 0.0,
        **where: str | bool | None,
    ) -> list[InboundDrainItem]:
        del where
        del session_id, wait_sec
        self.drains += 1
        if self.drains > 1:
            _real_pause(0.02)
            return []
        return [InboundDrainItem(text=text) for text in self._texts]


class _UploadOutageClient(_BatchClient):
    """Opens a session and serves its inbound queue, but every upload fails."""

    def __init__(self, texts: list[str]) -> None:
        super().__init__(texts)
        self.session = uuid.uuid4()
        self.drained_for: set[UUID] = set()

    def session_start(self, body: SessionStart) -> SessionStartResponse:
        return SessionStartResponse(id=self.session, seq=0, actor=body.actor)

    def append_records(
        self,
        session_id: UUID,
        **fields: object,
    ) -> AppendRecordsResponse:
        del session_id, fields
        raise RuntimeError("upload failed")

    def session_end(
        self,
        session_id: UUID,
        body: SessionEnd | None = None,
    ) -> SessionEndResponse:
        del body
        return SessionEndResponse(id=session_id)

    @override
    def drain_inbound(
        self,
        session_id: UUID,
        *,
        wait_sec: float = 0.0,
        **where: str | bool | None,
    ) -> list[InboundDrainItem]:
        del where
        self.drained_for.add(session_id)
        return super().drain_inbound(session_id, wait_sec=wait_sec)


class _ServerComesBackClient(_UploadOutageClient):
    """A server that refuses everything until ``up``, then accepts uploads too."""

    def __init__(self, texts: list[str]) -> None:
        super().__init__(texts)
        self.up = False

    @override
    def session_start(self, body: SessionStart) -> SessionStartResponse:
        if not self.up:
            raise RuntimeError("server down")
        return super().session_start(body)

    @override
    def append_records(
        self,
        session_id: UUID,
        **fields: object,
    ) -> AppendRecordsResponse:
        if not self.up:
            raise RuntimeError("server down")
        del fields
        return AppendRecordsResponse(part=0, written=1, skipped=0)


class _SessionSink:
    """A sink whose session is already open, so inbound has an id to use."""

    session_id = UUID("11111111-2222-3333-4444-555555555555")

    def __init__(self, *, account: str | None = None) -> None:
        self.kept_open = False
        self.account = account

    def keep_session_open(self) -> None:
        self.kept_open = True


class _RecordingRelay:
    """Records what would have been typed into the CLI."""

    def __init__(self) -> None:
        self.submitted: list[str] = []
        self.written: list[bytes] = []
        self.terminated = 0
        self.changed = threading.Condition()
        """Notified as each line is submitted; the inbound loop submits from its thread."""

    def submit(self, text: str) -> None:
        with self.changed:
            self.submitted.append(text)
            self.changed.notify_all()

    def wait_submitted(self, count: int) -> None:
        """Block until ``count`` lines are submitted, or give up after 3s."""
        with self.changed:
            _ = self.changed.wait_for(lambda: len(self.submitted) >= count, timeout=3.0)

    def write(self, data: bytes) -> None:
        self.written.append(data)

    def terminate(self) -> None:
        self.terminated += 1


def _real_pause(seconds: float) -> None:
    """Block without ``time.sleep``, so a sleep assertion stays meaningful."""
    threading.Event().wait(seconds)


def _spec(tmp_path: Path) -> HeadlessSpec:
    """Return the spec a ``--detach`` host is given, in ``tmp_path``."""
    return HeadlessSpec(
        address=socket_address(tmp_path / "host.sock"),
        scrollback=tmp_path / "scrollback.log",
        control=socket_address(tmp_path / "control.sock"),
    )


class _ScriptedDriver:
    """A driver that starts as told, reports what it was scripted to, and notes the rest."""

    def __init__(
        self,
        *,
        session: str = "s1",
        events: Sequence[DriverEvent] = (),
        fail: Exception | None = None,
    ) -> None:
        self.session = session
        self.script = events
        self.fail = fail
        self.started: tuple[Path, str, str, str] | None = None
        self.stopped = False
        self._ended: asyncio.Event | None = None

    async def start(self, *, cwd: Path, resume: str, model: str, effort: str) -> str:
        if self.fail is not None:
            raise self.fail
        self._ended = asyncio.Event()
        self.started = (cwd, resume, model, effort)
        return self.session

    async def send_line(self, text: str) -> None:
        del text

    async def command(self, verb: str, args: str) -> str:
        del verb, args
        return ""

    async def answer(self, answer: Answer) -> None:
        del answer

    async def interrupt(self) -> None:
        return

    async def settings(self, *, model: str, effort: str) -> None:
        del model, effort

    async def stop(self) -> None:
        self.stopped = True
        if self._ended is None:
            raise ValueError("Expected self._ended is not None.")
        self._ended.set()

    async def events(self) -> AsyncIterator[DriverEvent]:
        for event in self.script:
            yield event
        if self._ended is None:
            raise ValueError("Expected self._ended is not None.")
        _ = await self._ended.wait()


class _TuiRelay:
    """A takeover terminal: it listens at once and runs until it is ended."""

    def __init__(
        self,
        argv: list[str],
        *,
        wiring: _HeadlessWiring,
        **kwargs: object,
    ) -> None:
        self.argv = list(argv)
        self.kwargs = kwargs
        self.ended = threading.Event()
        self.over = False
        wiring.relays.append(self)

    def run(self) -> int:
        host = self.kwargs["host"]
        assert isinstance(host, HostSpec)
        if host.on_listening is None:
            raise ValueError("Expected host.on_listening is not None.")
        host.on_listening()
        _ = self.ended.wait(5.0)
        self.over = True
        return 0

    def terminate(self) -> None:
        self.ended.set()

    def submit(self, text: str) -> None:
        del text


@dataclass(slots=True, kw_only=True)
class _HeadlessWiring(_Wiring):
    """What a headless run handed each collaborator."""

    drivers: list[_ScriptedDriver] = field(default_factory=list[_ScriptedDriver])
    made: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    poll: list[object] = field(default_factory=list[object])
    relays: list[_TuiRelay] = field(default_factory=list[_TuiRelay])
    script: Callable[[HeadlessRun], None] | None = None
    polled: threading.Event = field(default_factory=threading.Event)
    stats: _Stats = field(default_factory=_Stats)
    taken: list[str] = field(default_factory=list[str])
    poll_thread: str = ""
    up: threading.Event = field(default_factory=threading.Event)
    after_ending: float = 0.0
    poll_done: threading.Event = field(default_factory=threading.Event)


def _hw_make_driver(
    wiring: _HeadlessWiring,
    config: RunConfig,
    adapter: Adapter,
    **kwargs: object,
) -> Driver:
    """Hand out the next scripted driver, noting what it was built from."""
    argv = cast("list[str]", kwargs["argv"])
    # Read now: the run takes the file away when it ends.
    settings = (
        Path(argv[argv.index("--settings") + 1]).read_text()
        if "--settings" in argv
        else ""
    )
    wiring.made.append(
        {"config": config, "adapter": adapter, "settings": settings, **kwargs},
    )
    return cast("Driver", wiring.drivers[len(wiring.made) - 1])


def _hw_poll(wiring: _HeadlessWiring, *args: object, **kwargs: object) -> None:
    """Note the poll loop's inputs, run the test's script against the run, end it."""
    wiring.poll = [*args, kwargs]
    wiring.daemons.append(threading.current_thread().daemon)
    wiring.poll_thread = threading.current_thread().name
    run = args[2]
    assert isinstance(run, HeadlessRun)
    wiring.polled.set()
    if wiring.script is not None:
        wiring.script(run)
    run.terminate()
    # Models the poll thread outliving the run, the window under test.
    time.sleep(wiring.after_ending)  # house-ignore[sleep] -- See above.
    wiring.poll_done.set()


# ``host`` says what hosts the run: a ``--detach`` host with a control socket, one with
# a terminal alone, or none. A ``clientless`` run has no server, so it is ended through
# its control socket.
def _wire_headless(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config: RunConfig,
    adapter: Adapter,
    drivers: Sequence[_ScriptedDriver] = (),
    *,
    wiring: _HeadlessWiring | None = None,
    script: Callable[[HeadlessRun], None] | None = None,
    sink: Sink | None = None,
    sandboxed: bool = False,
    client: RunClient | None = None,
    host: Literal["terminal", "plain", "none"] = "terminal",
    clientless: bool = False,
) -> tuple[_HeadlessWiring, int]:
    """Run ``_spawn_and_drain`` as a chat's headless run against scripted drivers."""
    wiring = wiring or _HeadlessWiring(drivers=list(drivers), script=script)
    spec = replace(
        _spec(tmp_path),
        on_takeover=wiring.taken.append,
        on_listening=(
            partial(_stop_soon, _spec(tmp_path).control)
            if clientless
            else wiring.up.set
        ),
    )
    hosts: dict[str, HostSpec | None] = {
        "terminal": spec,
        "plain": HostSpec(
            address=spec.address,
            scrollback=spec.scrollback,
            on_listening=wiring.up.set,
        ),
        "none": None,
    }
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setattr(session, "_make_driver", partial(_hw_make_driver, wiring))
    monkeypatch.setattr(
        session,
        "_drain_filesystem_loop",
        partial(_wiring_drain, wiring),
    )
    monkeypatch.setattr(session, "_inbound_poll_loop", partial(_hw_poll, wiring))
    monkeypatch.setattr(session, "ThreadedRelay", partial(_TuiRelay, wiring=wiring))
    daemon.set_sandbox(on=sandboxed)
    status = session._spawn_and_drain(  # noqa: SLF001 -- The run's wiring under test.
        replace(
            config,
            headless=True,
            client=None if clientless else client or _NoServer(),
            detached=hosts[host],
            run_name="chat-1",
            quiesce_seconds=0.0,
        ),
        adapter,
        sink or _RecordingSink(),
        wiring.stats,
    )
    return wiring, status


def _stop_soon(control: Path) -> None:
    """Take the run over a moment after it is up, then end it as ``knowop run stop``."""
    threading.Timer(0.02, _take_over_then_stop, args=(control,)).start()


def _take_over_then_stop(control: Path) -> None:
    _ = headless.request(control, "takeover")
    _ = headless.request(control, "stop")


def _headless_config(
    cli: str = "claude",
    *,
    actor: str | None = None,
    rooms: tuple[str, ...] = (),
    private: bool = False,
    cli_session_id: str | None = None,
    cli_args: tuple[str, ...] = (),
    compact_after_load: bool = False,
) -> RunConfig:
    """Return a chat run's config with the few fields these tests vary."""
    return RunConfig(
        cli_name=cli,
        actor=actor,
        rooms=rooms,
        private=private,
        cli_session_id=cli_session_id,
        cli_args=cli_args,
        compact_after_load=compact_after_load,
    )
