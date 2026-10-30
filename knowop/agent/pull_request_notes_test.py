"""Tests for noting what a session's shell calls did to pull requests."""

from __future__ import annotations

from trackinizer.sagent.session.types import ShellCommandResult, ToolCall

import pytest

from knowop.agent.pull_request_notes import Noted, RecordNotes


_URL = "https://github.com/acme/app/pull/7"


def test_a_claude_call_is_paired_with_its_result_by_id() -> None:
    notes = RecordNotes()
    call = ToolCall(call_id="c", name="Bash", arguments={"command": "gh pr create -f"})
    assert notes.note(call) is False
    assert notes.note(ShellCommandResult(call_id="c", stdout=f"{_URL}\n")) is True
    assert notes.take() == Noted(opened=(_URL,))
    assert notes.take() == Noted()


def test_a_codex_result_names_its_own_push() -> None:
    notes = RecordNotes()
    pushed = ShellCommandResult(call_id="x", command=("bash", "-lc", "git push origin"))
    assert notes.note(pushed) is True
    assert notes.take() == Noted(pushed=True)


@pytest.mark.parametrize(
    "result",
    [
        ShellCommandResult(call_id="x", command=("gh", "pr", "create"), exit_code=1),
        ShellCommandResult(call_id="x", command=("git", "push"), is_error=True),
        ShellCommandResult(call_id="x", command=("ls",), stdout=_URL),
        ShellCommandResult(call_id="x", stdout=_URL),
    ],
)
def test_a_failed_or_unrelated_call_notes_nothing(result: ShellCommandResult) -> None:
    notes = RecordNotes()
    assert notes.note(result) is False
    assert notes.take() == Noted()


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
