"""Tests for the runner's executables: polyglot dispatchers, run by module.

The conventions are pinned repo-wide by the ``cli`` skill; these assert only that
the two entry files follow them. A person runs ``knowop run``; these are for
unattended plumbing (``run_agent.sh``), which runs them as modules from a checkout and
keeps trax's connection rather than knowop's login.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import os
import runpy
import sys

import pytest

from knowop.agent import codex
from knowop.agent.cli import codex_main


_CWD: Final = Path(__file__).resolve().parent

_ENTRIES: Final = {
    "claude": (
        "__main__.py",
        "from knowop.agent.cli import main",
        "knowop.agent",
    ),
    "codex": (
        "codex.py",
        "from knowop.agent.cli import codex_main",
        "knowop.agent.codex",
    ),
}
"""Each CLI's entry file in this package, the one import it holds, and its module."""


@pytest.mark.parametrize("cli", sorted(_ENTRIES))
def test_each_entry_point_is_a_polyglot_executable_holding_no_logic(cli: str) -> None:
    """``#!/bin/sh`` plus the re-exec line pins the uv environment."""
    name, imported, _ = _ENTRIES[cli]
    text = (_CWD / name).read_text(encoding="utf-8")

    assert text.startswith("#!/bin/sh\n")
    assert "exec uv --quiet --project" in text
    assert "--frozen --no-sync" in text
    assert imported in text
    assert "argparse" not in text
    assert text.rstrip().endswith("# vim: ft=python")
    assert os.access(_CWD / name, os.X_OK)


def test_importing_the_codex_entry_runs_nothing_and_names_the_runners_main() -> None:
    assert codex.codex_main is codex_main


@pytest.mark.cli_python_subprocess
@pytest.mark.parametrize("cli", sorted(_ENTRIES))
def test_the_runner_runs_as_a_module(
    cli: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The entry file imports its runner and parses arguments without a process."""
    entry = _CWD / _ENTRIES[cli][0]
    monkeypatch.setattr(sys, "argv", [str(entry), "--help"])

    with pytest.raises(SystemExit) as exited:
        _ = runpy.run_path(str(entry), run_name="__main__")

    assert exited.value.code == 0
    assert capsys.readouterr().out.startswith(f"usage: knowop run {cli} ")


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
