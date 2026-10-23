"""The job runner's executable: a polyglot dispatcher over a library ``main``.

The conventions are pinned repo-wide by the ``cli`` skill; these assert only that
this entry point follows them, because a half-executable (a run guard with no
polyglot header) runs under whatever interpreter is on PATH rather than the pinned
uv environment. A person runs ``knowop job``; the module is for plumbing that keeps
trax's connection, such as the knowop job rehearsal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import os
import subprocess
import sys

import pytest


_CWD: Final = Path(__file__).resolve().parent


def test_entry_point_is_a_polyglot_executable() -> None:
    """``#!/bin/sh`` + the re-exec line is what pins the uv environment."""
    text = (_CWD / "__main__.py").read_text(encoding="utf-8")

    assert text.startswith("#!/bin/sh\n")
    assert "exec uv --quiet --project" in text
    assert "--frozen --no-sync" in text
    assert text.rstrip().endswith("# vim: ft=python")
    assert os.access(_CWD / "__main__.py", os.X_OK)


def test_entry_point_holds_no_logic() -> None:
    """The dispatcher imports ``main`` and threads its exit code; nothing else."""
    text = (_CWD / "__main__.py").read_text(encoding="utf-8")

    assert "from knowop.jobs.cli import main" in text
    assert "raise SystemExit(main())" in text
    assert "argparse" not in text


def test_the_library_module_carries_no_run_guard() -> None:
    """``cli.py`` is imported, never run: a bare guard would sidestep uv."""
    text = (_CWD / "cli.py").read_text(encoding="utf-8")

    assert '__name__ == "__main__"' not in text
    assert not text.startswith("#!")


@pytest.mark.cli_python_subprocess
def test_the_module_prints_the_usage() -> None:
    """The whole chain -- module, imports, argparse -- works."""
    # The checkout under test, not whichever one the shared venv installed. Counted
    # from the module's dotted name: the export republishes this tree shallower.
    root = _CWD.parents[__name__.count(".") - 1]
    completed = subprocess.run(
        [sys.executable, "-m", "knowop.jobs", "--help"],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
    )

    assert completed.stdout.startswith("Usage: knowop job submit")
    assert not completed.stderr


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
