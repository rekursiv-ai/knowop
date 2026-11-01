#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P "$0" "$@"
Run Claude Code on this computer, recorded and reachable from Chat.

Executable entry point: dispatches to the sibling ``cli`` module's ``main``.
'''
# fmt: on

from __future__ import annotations

from knowop.agent.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
