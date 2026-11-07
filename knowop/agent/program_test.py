"""Tests for ``knowop.agent.program``."""

from __future__ import annotations

from knowop.agent.program import PROGRAM


def test_the_program_is_the_command_a_person_types() -> None:
    assert PROGRAM == "knowop run"


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
