"""Notice this computer waking from a pause, so its daemon speaks to the server at once.

A paused virtual machine, such as a cloud sandbox, stops its monotonic clock and keeps
no connection: on resume each timer and each read timeout goes on from where it
stopped, and each held request waits on a socket nothing answers any more. Left alone,
a daemon's next heartbeat waits out what its sleep had left, up to 15 s, and its held
waits what their read timeouts had left, up to 35 s. The wall clock, which the host
steps to the real time on resume, leaps ahead of the monotonic one, and that leap is
the signal: :class:`Wake` watches for it and tells whoever asked, at once.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Final

import logging
import threading
import time


if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = ["Listener", "Wake"]

_logger = logging.getLogger(__name__)

_LEAP_SEC: Final = 1.0
"""How far the wall clock must gain on the monotonic one to count as a wake: well past
any clock adjustment, and under any pause worth noticing."""


class Wake:
    """Watches the wall clock leap past the monotonic one, as it does after a pause.

    Args:
      tick_sec: How often the clocks are compared: at most this late to notice.
      wall: The wall clock, in seconds.
      clock: The monotonic clock, in seconds.

    """

    def __init__(
        self,
        *,
        tick_sec: float = 0.2,
        wall: Callable[[], float] = time.time,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._tick_sec = tick_sec
        self._wall = wall
        self._clock = clock
        self._offset = wall() - clock()
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[], None]] = []
        self._stop = threading.Event()

    def start(self) -> None:
        """Watch on a thread of its own until :meth:`stop`."""
        threading.Thread(name="wake", target=self._watch, daemon=True).start()

    def stop(self) -> None:
        """End the watch."""
        self._stop.set()

    def on_wake(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Call ``callback`` on the watching thread after each wake.

        Args:
          callback: What to call.

        Returns:
          forget: Stops calling it.

        """
        with self._lock:
            self._callbacks.append(callback)
        return partial(self._forget, callback)

    def check(self) -> bool:
        """Compare the clocks once; on a leap, call every callback.

        Returns:
          woke: Whether the wall clock leapt since the last check.

        """
        offset = self._wall() - self._clock()
        leapt = offset - self._offset > _LEAP_SEC
        self._offset = offset
        if not leapt:
            return False
        _logger.info("woke from a pause: the wall clock leapt ahead")
        with self._lock:
            callbacks = list(self._callbacks)
        for callback in callbacks:
            try:
                callback()
            except Exception:
                _logger.exception("a wake's callback failed")
        return True

    def _forget(self, callback: Callable[[], None]) -> None:
        with self._lock:
            self._callbacks.remove(callback)

    def _watch(self) -> None:
        while not self._stop.wait(self._tick_sec):
            _ = self.check()


class Listener:
    """One long-poll at a time, on a thread of its own, started anew on each wake.

    The wait a pause cut is left to end on its own, which may take its whole read
    timeout; the new one asks on a client of its own, as the old one's idle connections
    died with the pause too. Each is told whether it is still the current one, and ends
    once its wait returns when it is not.

    Args:
      name: Each thread's name.
      listen: Listens until the daemon stops, or the function it is given says it was
        replaced.
      stop: The daemon's stop: no listener starts once it is set.

    """

    def __init__(
        self,
        *,
        name: str,
        listen: Callable[[Callable[[], bool]], None],
        stop: threading.Event,
    ) -> None:
        self._name = name
        self._listen = listen
        self._stop = stop
        self._token = object()

    def start(self, wake: Wake | None) -> None:
        """Start the first listener, and a new one on each wake ``wake`` tells of."""
        # Told first: a wake the first listener's start races would otherwise be missed.
        if wake is not None:
            _ = wake.on_wake(self.restart)
        self.restart()

    def restart(self) -> None:
        """Start a listener that replaces the one before it."""
        if self._stop.is_set():
            return
        token = self._token = object()
        threading.Thread(
            name=self._name,
            target=self._listen,
            args=(partial(self._current, token),),
            daemon=True,
        ).start()

    def _current(self, token: object) -> bool:
        return self._token is token
