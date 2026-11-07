"""Tests for noting what a session's shell calls did to pull requests."""

from __future__ import annotations

from trackinizer.sagent.session.types import BashResult, SystemMessage, ToolCall

import pytest

from knowop.agent.pull_request_notes import Noted, RecordNotes


_URL = "https://github.com/acme/app/pull/7"


def test_a_claude_call_is_paired_with_its_result_by_id() -> None:
    notes = RecordNotes()
    call = ToolCall(call_id="c", name="Bash", arguments={"command": "gh pr create -f"})
    assert notes.note(call) is False
    assert notes.note(BashResult(call_id="c", stdout=f"{_URL}\n")) is True
    assert notes.take() == Noted(opened=(_URL,))
    assert notes.take() == Noted()


def test_a_command_spelled_as_a_list_is_read_from_its_words() -> None:
    notes = RecordNotes()
    call = ToolCall(
        call_id="c",
        name="exec",
        arguments={"argv": ["git", "push", "origin"], "options": {"cwd": "/w"}},
    )
    assert notes.note(call) is False
    assert notes.note(BashResult(call_id="c")) is True
    assert notes.take() == Noted(pushed=True)


def test_a_record_that_is_no_shell_call_notes_nothing() -> None:
    notes = RecordNotes()
    said = SystemMessage(
        timestamp="2026-01-01T00:00:00Z",
        subtype="x",
        content="git push",
    )
    assert notes.note(said) is False
    assert notes.take() == Noted()


def test_a_codex_result_names_its_own_push() -> None:
    notes = RecordNotes()
    pushed = BashResult(call_id="x", command=("bash", "-lc", "git push origin"))
    assert notes.note(pushed) is True
    assert notes.take() == Noted(pushed=True)


@pytest.mark.parametrize(
    "result",
    [
        BashResult(call_id="x", command=("gh", "pr", "create"), exit_code=1),
        BashResult(call_id="x", command=("git", "push"), is_error=True),
        BashResult(call_id="x", command=("ls",), stdout=_URL),
        BashResult(call_id="x", stdout=_URL),
    ],
)
def test_a_failed_or_unrelated_call_notes_nothing(result: BashResult) -> None:
    notes = RecordNotes()
    assert notes.note(result) is False
    assert notes.take() == Noted()


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
