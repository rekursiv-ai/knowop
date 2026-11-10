"""Noticing a wake from a pause, by the wall clock leaping past the monotonic one."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import threading

from knowop.shell.wake import Listener, Wake


if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest


def _wake(wall: list[float], clock: list[float]) -> Wake:
    return Wake(wall=lambda: wall[0], clock=lambda: clock[0])


def test_clocks_that_run_together_or_drift_a_little_are_no_wake() -> None:
    wall, clock = [100.0], [5.0]
    wake = _wake(wall, clock)
    called: list[str] = []
    _ = wake.on_wake(lambda: called.append("woke"))

    wall[0] += 10.0
    clock[0] += 10.0
    together = wake.check()
    wall[0] += 0.9

    assert (together, wake.check(), called) == (False, False, [])


def test_a_leap_calls_each_callback_once_until_it_is_forgotten(
    caplog: pytest.LogCaptureFixture,
) -> None:
    wall, clock = [100.0], [5.0]
    wake = _wake(wall, clock)
    called: list[str] = []

    def broken() -> None:
        raise RuntimeError("no")

    _ = wake.on_wake(broken)
    forget = wake.on_wake(lambda: called.append("woke"))
    wall[0] += 30.0
    first = wake.check()
    again = wake.check()
    forget()
    wall[0] += 30.0
    after = wake.check()

    assert (first, again, after, called) == (True, False, True, ["woke"])
    assert "a wake's callback failed" in caplog.text


def test_the_watch_runs_on_its_own_thread_until_stopped() -> None:
    wall = [100.0]
    woke = threading.Event()
    wake = Wake(tick_sec=0.001, wall=lambda: wall[0], clock=lambda: 5.0)
    _ = wake.on_wake(woke.set)

    wake.start()
    wall[0] += 30.0
    seen = woke.wait(10)
    wake.stop()

    assert seen


def test_a_listener_started_anew_on_a_wake_tells_the_one_before_it_was_replaced() -> (
    None
):
    wall, clock = [100.0], [5.0]
    wake = _wake(wall, clock)
    stop = threading.Event()
    started: list[Callable[[], bool]] = []
    each = threading.Semaphore(0)

    def listen(current: Callable[[], bool]) -> None:
        started.append(current)
        each.release()

    Listener(name="test-listener", listen=listen, stop=stop).start(wake)
    assert each.acquire(timeout=5)
    wall[0] += 30.0
    _ = wake.check()
    assert each.acquire(timeout=5)
    stop.set()
    wall[0] += 30.0
    _ = wake.check()

    assert [current() for current in started] == [False, True]


def test_a_wake_while_the_first_listener_starts_starts_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wall, clock = [100.0], [5.0]
    wake = _wake(wall, clock)
    started: list[Callable[[], bool]] = []

    # Each listener runs as its thread starts, so the first one's wake comes as early
    # as a loaded machine can make it: before ``start`` has returned.
    class _AtOnce:
        def __init__(
            self,
            *,
            target: Callable[..., None],
            args: tuple[object, ...],
            **_: object,
        ) -> None:
            self._run = partial(target, *args)

        def start(self) -> None:
            self._run()

    monkeypatch.setattr(threading, "Thread", _AtOnce)

    def listen(current: Callable[[], bool]) -> None:
        started.append(current)
        if len(started) == 1:
            wall[0] += 30.0
            assert wake.check()

    Listener(name="test-listener", listen=listen, stop=threading.Event()).start(wake)

    assert [current() for current in started] == [False, True]


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
