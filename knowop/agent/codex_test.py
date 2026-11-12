"""Tests for the ``codex`` executable entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import runpy

import pytest


_CWD: Final[Path] = Path(__file__).resolve().parent


def test_running_the_script_exits_with_codex_mains_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "knowop.agent.cli.codex_main",
        lambda: 7,
    )

    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(_CWD / "codex.py"), run_name="__main__")

    assert raised.value.code == 7


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
