"""The command a person types to reach a run: ``knowop run``."""

from __future__ import annotations

from typing import Final


__all__ = ["PROGRAM"]

PROGRAM: Final = "knowop run"
"""What the runner's messages and usage name: ``knowop run attach NAME`` reaches any
run, claude or codex, so one name serves both."""
