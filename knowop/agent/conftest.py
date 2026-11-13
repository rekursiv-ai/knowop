"""Fakes the agent tests share.

Each test module keeps its own tests; what more than one reads -- a placing client and
the ids it is asked about -- lives here, so no test module imports another.
"""

from __future__ import annotations

from uuid import UUID

from trackinizer.client.errors import ClientError


__all__ = ["_CANVAS", "_SESSION", "_Client"]


_CANVAS = UUID(int=5)
_SESSION = UUID(int=1)


class _Client:
    """A client whose placement answer names a workspace, or that refuses to place."""

    def __init__(self, *, fail: bool = False) -> None:
        self.asked: list[tuple[str, object]] = []
        self._fail = fail

    def post(self, path: str, *, body: object = None) -> dict[str, str]:
        self.asked.append((path, body))
        if self._fail:
            raise ClientError("404 Not Found")
        return {"workspace_id": str(UUID(int=2))}
