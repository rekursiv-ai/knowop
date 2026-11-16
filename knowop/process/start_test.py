"""The shared process start keeps the CLI's attach handshake and arguments."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, override
from uuid import UUID

import queue
import threading

from trackinizer.client.errors import ClientError
from trackinizer.wire.wire_session_relay import (
    AgentProcess,
    Opened,
    ProcessLine,
    ProcessSpec,
    ViewerFrame,
)

import pytest

from knowop.process import start
from knowop.process.command import parse_run, run_parser
from knowop.process.main_test import _SESSION, _attached, _Server


if TYPE_CHECKING:
    from collections.abc import Generator, Iterator


@pytest.mark.parametrize("timed", [[], ["--time", "1-2:03:04"]])
def test_arguments_keep_the_time_first_and_sagent_words_unchanged(
    timed: list[str],
) -> None:
    flags, args = parse_run(run_parser(), [*timed, "--", "--name", "lead"])
    expected = (
        ("--time", "1-02:03:04", "--name", "lead") if timed else ("--name", "lead")
    )
    assert start.start_args(flags.time_limit, args) == expected


@pytest.mark.parametrize("state", ["attached", "starting"])
def test_attaches_only_the_requested_process(state: str) -> None:
    client = _Server(version=1)
    client.answer = _attached(state)
    client.frames = [
        Opened(),
        Opened(),
        ProcessLine(process=_attached("detached", process=UUID(int=99))),
        ProcessLine(process=_attached()),
    ]
    assert (
        start.started_process(client, session_id=_SESSION, spec=ProcessSpec()).state
        == "attached"
    )
    assert client.started == [ProcessSpec()]
    assert client.reader == ("lines", True)


@pytest.mark.parametrize("state", ["starting", "detached"])
def test_detachment_reports_its_reason(state: str) -> None:
    client = _Server(version=1)
    client.answer = _attached(state)
    client.frames = [Opened(), ProcessLine(process=_attached("detached"))]
    with pytest.raises(ClientError, match="the process did not start: it crashed"):
        _ = start.started_process(client, session_id=_SESSION, spec=ProcessSpec())


def test_lines_refusal_opens_no_process() -> None:
    client = _Server(version=1)
    client.lines_refused = True
    with pytest.raises(ClientError, match="GET lines"):
        _ = start.started_process(client, session_id=_SESSION, spec=ProcessSpec())
    assert client.started == []


def test_attachment_timeout_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Server(version=1)
    client.answer = _attached("starting")
    client.frames = [Opened()]
    monkeypatch.setattr(start, "_ATTACHED_SEC", 0.001)
    with pytest.raises(TimeoutError, match=r"did not attach within 0\.001 s"):
        _ = start.started_process(client, session_id=_SESSION, spec=ProcessSpec())
    assert len(client.started) == 1


class _BlockingLines(_Server):
    def __init__(self) -> None:
        super().__init__(version=1)
        self.answer = _attached("starting")
        self.frames = [Opened(), ProcessLine(process=_attached())]
        self.start_failure: ClientError | None = None
        self.stream_closed = threading.Event()
        self.worker: threading.Thread | None = None

    @override
    def open_lines(self, session_id: UUID) -> Iterator[ViewerFrame]:
        del session_id
        self.worker = threading.current_thread()
        yield from self.frames
        _ = self.stream_closed.wait(0.05)

    @override
    def start_process(self, session_id: UUID, spec: ProcessSpec) -> AgentProcess:
        if self.start_failure is not None:
            raise self.start_failure
        return super().start_process(session_id, spec)

    @override
    @contextmanager
    def lines_stream(self, session_id: UUID) -> Generator[Iterator[ViewerFrame]]:
        try:
            yield self.open_lines(session_id)
        finally:
            self.stream_closed.set()

    def close_reader(self) -> None:
        self.stream_closed.set()
        if self.worker is not None:
            self.worker.join()


def test_attachment_closes_the_stream_and_joins_the_reader() -> None:
    client = _BlockingLines()
    try:
        assert (
            start.started_process(
                client,
                session_id=_SESSION,
                spec=ProcessSpec(),
            ).state
            == "attached"
        )
        assert client.stream_closed.is_set()
        assert client.worker is not None
        assert not client.worker.is_alive()
    finally:
        client.close_reader()


@pytest.mark.parametrize("failure", ["detached", "start refused", "timeout"])
def test_every_start_error_closes_the_stream_and_joins_the_reader(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _BlockingLines()
    if failure == "detached":
        client.answer = _attached("detached")
    elif failure == "start refused":
        client.start_failure = ClientError("start refused")
    else:
        client.frames = [Opened()]
        monkeypatch.setattr(start, "_ATTACHED_SEC", 0.001)
    try:
        with pytest.raises((ClientError, TimeoutError), match=r"did not|start refused"):
            _ = start.started_process(client, session_id=_SESSION, spec=ProcessSpec())
        assert client.stream_closed.is_set()
        assert client.worker is not None
        assert not client.worker.is_alive()
    finally:
        client.close_reader()


def test_reader_discards_frames_after_its_owner_stops() -> None:
    frames: queue.Queue[ViewerFrame | ClientError] = queue.Queue()
    stopped = threading.Event()
    stopped.set()
    start._read_lines(iter([Opened()]), frames, stopped)
    assert frames.empty()


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
