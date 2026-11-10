"""Tests for the claude stream-json driver.

:class:`ClaudeWire` is the protocol half, tested on frames alone. The frames in these
tests are the ones claude 2.1.306 wrote in a live probe on 2026-10-21 (`--permission-mode
default`, haiku): a Bash ask with three suggestions, an ``AskUserQuestion``, an
``ExitPlanMode``, ``control_cancel_request`` after an interrupt, and the per-turn
``session_state_changed``. :class:`ClaudeStreamDriver` is the process half, tested
against a scripted fake that speaks the same lines.
"""

from __future__ import annotations

from asyncio.subprocess import Process
from collections.abc import Callable
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from functools import partial
from io import BytesIO, StringIO, TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast
from unittest.mock import AsyncMock, patch

import asyncio
import json
import os
import runpy
import sys
import uuid

from trackinizer.lib.posix.pipes import Piped
from trackinizer.runs.asks import (
    Answer,
    Ask,
    Compaction,
    Driver,
    Option,
    Resolved,
    StateChanged,
)
from treekle import PlainTree, from_plain

import pytest

from knowop.agent.claude_stream import (
    ClaudeStreamDriver,
    ClaudeWire,
    Reading,
)
from knowop.agent.conftest import (
    _ANY_REQUEST,
    _ASK_STEPS,
    _BASH_INPUT,
    _INIT,
    _ack,
    _can_use_tool,
    _claude_asked,
    _claude_next,
    _ClaudeEvent,
    _ClaudeFrame,
    _response,
    _rest,
    _Run,
    _says,
    _signal_app,
    _state,
    _step,
)
from knowop.connect.conftest import _process


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from knowop.connect.conftest import _Process


_CWD: Final = Path("/work")

_RULE: Final[PlainTree] = {
    "type": "addRules",
    "rules": [{"toolName": "Bash", "ruleContent": "touch ok.txt"}],
    "behavior": "allow",
    "destination": "localSettings",
}
_FOLDER: Final[PlainTree] = {
    "type": "addDirectories",
    "directories": ["/work"],
    "destination": "session",
}
_MODE: Final[PlainTree] = {
    "type": "setMode",
    "mode": "acceptEdits",
    "destination": "session",
}
_SUGGESTIONS: Final[PlainTree] = [_RULE, _FOLDER, _MODE]


_QUESTIONS: Final[PlainTree] = [
    {
        "question": "Do you prefer red or blue?",
        "header": "Color",
        "options": [
            {"label": "Red", "description": "Choose red"},
            {"label": "Blue", "description": "Choose blue"},
        ],
        "multiSelect": False,
    },
]


def _request(subtype: str, **fields: PlainTree) -> _ClaudeFrame:
    return {
        "type": "control_request",
        "request_id": "r2",
        "request": {"subtype": subtype, **fields},
    }


def _opened(reading: Reading) -> Ask:
    assert len(reading.events) == 1
    ask = reading.events[0]
    assert isinstance(ask, Ask)
    return ask


def _bash(
    wire: ClaudeWire,
    tool: str = "Bash",
    *,
    input: PlainTree = _BASH_INPUT,
    request_id: str = "r1",
    **fields: PlainTree,
) -> Ask:
    frame = _can_use_tool(tool, input=input, request_id=request_id, **fields)
    return _opened(wire.read(frame))


def _kinds(ask: Ask) -> list[tuple[str, str]]:
    return [(each.id, each.kind) for each in ask.options]


def test_a_bash_ask_becomes_a_permission_ask() -> None:
    wire = ClaudeWire(cwd=_CWD)
    ask = _bash(
        wire,
        description="Create empty file ok.txt",
        permission_suggestions=_SUGGESTIONS,
        blocked_path="/work/ok.txt",
    )
    assert ask == Ask(
        id="r1",
        provider="claude",
        kind="permission",
        title="Create empty file ok.txt",
        tool="Bash",
        tool_call_id="toolu_1",
        command="touch ok.txt",
        cwd="/work",
        input=from_plain(_BASH_INPUT, dict[str, PlainTree]),
        options=(
            Option(id="allow_once", label="Allow", kind="allow_once"),
            Option(
                id="allow_session",
                label="Allow access to /work for this session",
                kind="allow_session",
            ),
            Option(
                id="allow_session_2",
                label="Switch to acceptEdits mode for this session",
                kind="allow_session",
            ),
            Option(id="deny", label="Deny", kind="deny"),
            Option(id="deny_stop", label="Deny and stop", kind="deny_stop"),
        ),
    )


def test_the_title_beats_the_description_and_a_reason_loses_its_escapes() -> None:
    ask = _bash(
        ClaudeWire(cwd=_CWD),
        title="Claude wants to run touch",
        description="Create empty file ok.txt",
        decision_reason="\x1b[31mneeds approval\x1b[0m",
    )
    assert (ask.title, ask.reason) == ("Claude wants to run touch", "needs approval")


def test_only_session_suggestions_become_options() -> None:
    ask = _bash(
        ClaudeWire(cwd=_CWD),
        permission_suggestions=[
            {
                "type": "addRules",
                "rules": [{"toolName": "Bash", "ruleContent": "ls *"}],
                "behavior": "allow",
                "destination": "session",
            },
            {
                "type": "addRules",
                "rules": [{"toolName": "Read"}],
                "behavior": "allow",
                "destination": "session",
            },
            {"type": "setMode", "mode": "plan", "destination": "projectSettings"},
            {
                "type": "removeDirectories",
                "directories": ["/x"],
                "destination": "session",
            },
        ],
    )
    assert [each.label for each in ask.options] == [
        "Allow",
        "Allow Bash(ls *) for this session",
        "Allow Read for this session",
        "Deny",
        "Deny and stop",
    ]


def test_a_rule_is_not_offered_when_it_would_be_broader_than_the_ask() -> None:
    ask = _bash(
        ClaudeWire(cwd=_CWD),
        suppress_always_allow_rule=True,
        permission_suggestions=[
            {
                "type": "addRules",
                "rules": [{"toolName": "Bash"}],
                "behavior": "allow",
                "destination": "session",
            },
            {
                "type": "addDirectories",
                "directories": ["/work"],
                "destination": "session",
            },
        ],
    )
    assert _kinds(ask) == [
        ("allow_once", "allow_once"),
        ("allow_session", "allow_session"),
        ("deny", "deny"),
        ("deny_stop", "deny_stop"),
    ]
    assert ask.options[1].label == "Allow access to /work for this session"


def test_an_edit_ask_carries_its_diff() -> None:
    ask = _bash(
        ClaudeWire(cwd=_CWD),
        tool="Edit",
        input={
            "file_path": "/work/a.py",
            "old_string": "x = 1\n",
            "new_string": "x = 2\n",
            "replace_all": False,
        },
    )
    assert ask.diff == ("--- /work/a.py\n+++ /work/a.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n")
    assert ask.command == ""


def test_an_edit_without_strings_has_no_diff() -> None:
    ask = _bash(ClaudeWire(cwd=_CWD), tool="Edit", input={"file_path": "/work/a.py"})
    assert ask.diff == ""


def test_ask_user_question_becomes_a_question() -> None:
    ask = _bash(
        ClaudeWire(cwd=_CWD),
        tool="AskUserQuestion",
        input={"questions": _QUESTIONS},
        requires_user_interaction=True,
    )
    assert (ask.kind, ask.tool, ask.tool_call_id) == (
        "question",
        "AskUserQuestion",
        "toolu_1",
    )
    assert ask.questions == _QUESTIONS
    assert _kinds(ask) == [("allow_once", "allow_once"), ("deny", "deny")]
    assert [each.label for each in ask.options] == ["Submit answers", "Skip"]


def test_exit_plan_mode_becomes_a_plan() -> None:
    ask = _bash(
        ClaudeWire(cwd=_CWD),
        tool="ExitPlanMode",
        input={"plan": "# Plan\n1. Do it\n", "planFilePath": "~/.claude/plans/p.md"},
    )
    assert (ask.kind, ask.plan) == ("plan", "# Plan\n1. Do it\n")
    assert [each.label for each in ask.options] == ["Approve plan", "Keep planning"]


def test_a_tool_without_an_input_is_still_asked() -> None:
    wire = ClaudeWire(cwd=_CWD)
    frame = _can_use_tool("mcp__x__y")
    request = from_plain(frame["request"], dict[str, PlainTree])
    del request["input"]
    frame["request"] = request
    ask = _opened(wire.read(frame))
    assert (ask.tool, ask.input, ask.command) == ("mcp__x__y", {}, "")


def test_the_session_id_and_state_follow_the_cli() -> None:
    wire = ClaudeWire(cwd=_CWD)
    assert wire.read({"type": "system", "subtype": "init", "session_id": "s1"}) == (
        Reading()
    )
    assert wire.session_id == "s1"
    state = wire.read(
        {
            "type": "system",
            "subtype": "session_state_changed",
            "state": "requires_action",
        },
    )
    assert state.events == (StateChanged(state="requires_action"),)
    wire.read({"type": "system", "subtype": "init", "session_id": "s2"})
    assert wire.session_id == "s2"


# The shapes claude 2.1.319 writes: ``status`` ``compacting`` when a compaction begins and
# every 30 seconds while it runs, then ``status`` ``null`` with its result.
def _status(status: str | None, **result: str) -> _ClaudeFrame:
    return {
        "type": "system",
        "subtype": "status",
        "status": status,
        **result,
        "uuid": "0e7d",
        "session_id": "s1",
    }


def test_a_compaction_is_one_start_however_often_claude_repeats_it_then_its_end() -> (
    None
):
    wire = ClaudeWire(cwd=_CWD)
    assert wire.read(_status("compacting")).events == (Compaction(state="started"),)
    assert wire.read(_status("compacting")) == Reading()
    assert wire.read(_status(None, compact_result="success")).events == (
        Compaction(state="ended"),
    )
    assert wire.read(_status(None)) == Reading()


@pytest.mark.parametrize(
    ("result", "error"),
    [
        ({"compact_result": "failed", "compact_error": "aborted"}, "aborted"),
        ({"compact_result": "failed"}, "failed"),
    ],
)
def test_a_failed_compaction_ends_with_why(result: dict[str, str], error: str) -> None:
    wire = ClaudeWire(cwd=_CWD)
    _ = wire.read(_status("compacting"))
    assert wire.read(_status(None, **result)).events == (
        Compaction(state="ended", error=error),
    )


@pytest.mark.parametrize(
    "frame",
    [
        {"type": "assistant", "message": {"content": []}},
        _status(None),
        {"type": "result", "subtype": "success"},
        {"type": "system", "subtype": "session_state_changed", "state": "asleep"},
        {"type": "system", "subtype": "init"},
        {"type": "system", "subtype": "compact_boundary"},
        {"type": "control_cancel_request", "request_id": "nobody"},
        {"type": "control_request", "request": {"subtype": "can_use_tool"}},
        {"type": "mystery"},
    ],
)
def test_frames_that_open_nothing_are_ignored(frame: _ClaudeFrame) -> None:
    assert ClaudeWire(cwd=_CWD).read(frame) == Reading()


def test_a_replayed_request_opens_one_ask() -> None:
    wire = ClaudeWire(cwd=_CWD)
    _ = _bash(wire)
    assert wire.read(_can_use_tool()) == Reading()


def test_a_cancel_releases_the_ask() -> None:
    wire = ClaudeWire(cwd=_CWD)
    _ = _bash(wire)
    gone = wire.read({"type": "control_cancel_request", "request_id": "r1"})
    assert gone.events == (
        Resolved(ask_id="r1", option="", by="released", decision="released"),
    )
    assert wire.answer(Answer(ask_id="r1", option="allow_once")) is None


def test_allow_sends_the_input_back() -> None:
    wire = ClaudeWire(cwd=_CWD)
    _ = _bash(wire, permission_suggestions=_SUGGESTIONS)
    reply = wire.answer(Answer(ask_id="r1", option="allow_once", who="jp"))
    assert reply is not None
    assert reply.frame == _response(
        "r1",
        behavior="allow",
        updatedInput=_BASH_INPUT,
    )
    assert reply.resolved == Resolved(
        ask_id="r1",
        option="allow_once",
        by="chat",
        who="jp",
        decision="allow",
    )


def test_allow_for_the_session_echoes_the_suggestion() -> None:
    wire = ClaudeWire(cwd=_CWD)
    _ = _bash(wire, permission_suggestions=_SUGGESTIONS)
    reply = wire.answer(Answer(ask_id="r1", option="allow_session_2"))
    assert reply is not None
    assert reply.frame == _response(
        "r1",
        behavior="allow",
        updatedInput=_BASH_INPUT,
        updatedPermissions=[_MODE],
    )
    assert reply.resolved.option == "allow_session_2"


def test_an_older_answer_names_a_kind_not_an_id() -> None:
    wire = ClaudeWire(cwd=_CWD)
    _ = _bash(wire, permission_suggestions=_SUGGESTIONS)
    reply = wire.answer(Answer(ask_id="r1", option="allow_session"))
    assert reply is not None
    assert reply.frame == _response(
        "r1",
        behavior="allow",
        updatedInput=_BASH_INPUT,
        updatedPermissions=[_FOLDER],
    )


def test_updated_input_is_laid_over_the_tools_input() -> None:
    wire = ClaudeWire(cwd=_CWD)
    _ = _bash(wire)
    reply = wire.answer(
        Answer(ask_id="r1", option="allow_once", updated_input={"command": "touch b"}),
    )
    assert reply is not None
    assert reply.frame == _response(
        "r1",
        behavior="allow",
        updatedInput={"command": "touch b", "description": "Create empty file ok.txt"},
    )


def test_deny_sends_the_message_the_agent_reads() -> None:
    wire = ClaudeWire(cwd=_CWD)
    _ = _bash(wire)
    reply = wire.answer(Answer(ask_id="r1", option="deny", message="Not that one."))
    assert reply is not None
    assert reply.frame == _response("r1", behavior="deny", message="Not that one.")
    assert reply.resolved.decision == "deny"


def test_deny_without_a_message_says_who_declined() -> None:
    wire = ClaudeWire(cwd=_CWD)
    _ = _bash(wire)
    reply = wire.answer(Answer(ask_id="r1", option="deny"))
    assert reply is not None
    assert reply.frame == _response(
        "r1",
        behavior="deny",
        message="The person declined this request.",
    )


def test_deny_and_stop_denies_and_interrupts_in_one_frame() -> None:
    wire = ClaudeWire(cwd=_CWD)
    _ = _bash(wire)
    reply = wire.answer(Answer(ask_id="r1", option="deny_stop"))
    assert reply is not None
    assert reply.frame == _response(
        "r1",
        behavior="deny",
        message="The person declined this request and asked you to stop.",
        interrupt=True,
    )
    assert reply.resolved.decision == "deny"


def test_a_question_is_answered_in_updated_input() -> None:
    wire = ClaudeWire(cwd=_CWD)
    _ = _bash(wire, tool="AskUserQuestion", input={"questions": _QUESTIONS})
    answers: dict[str, PlainTree] = {"Do you prefer red or blue?": "Blue"}
    reply = wire.answer(
        Answer(ask_id="r1", option="allow_once", updated_input={"answers": answers}),
    )
    assert reply is not None
    assert reply.frame == _response(
        "r1",
        behavior="allow",
        updatedInput={"questions": _QUESTIONS, "answers": answers},
    )
    assert reply.resolved.decision == "answered"


def test_an_answer_keyed_by_header_is_keyed_by_question_text() -> None:
    wire = ClaudeWire(cwd=_CWD)
    _ = _bash(wire, tool="AskUserQuestion", input={"questions": _QUESTIONS})
    reply = wire.answer(
        Answer(
            ask_id="r1",
            option="allow_once",
            updated_input={"answers": {"Color": "Blue"}},
        ),
    )
    assert reply is not None
    assert reply.frame == _response(
        "r1",
        behavior="allow",
        updatedInput={
            "questions": _QUESTIONS,
            "answers": {"Do you prefer red or blue?": "Blue"},
        },
    )


def test_a_free_text_answer_survives_the_rekeying() -> None:
    wire = ClaudeWire(cwd=_CWD)
    _ = _bash(wire, tool="AskUserQuestion", input={"questions": _QUESTIONS})
    words = "discuss, do you find previous best practice on this?"
    reply = wire.answer(
        Answer(
            ask_id="r1",
            option="allow_once",
            updated_input={"answers": {"Color": words}},
        ),
    )
    assert reply is not None
    assert reply.frame == _response(
        "r1",
        behavior="allow",
        updatedInput={
            "questions": _QUESTIONS,
            "answers": {"Do you prefer red or blue?": words},
        },
    )


def test_every_question_of_an_ask_is_keyed_by_its_text() -> None:
    wire = ClaudeWire(cwd=_CWD)
    questions: list[PlainTree] = [
        {"question": "Which color?", "header": "Color", "options": []},
        {"question": "Which size?", "header": "Size", "id": "size_id", "options": []},
        {"question": "Which shape?", "header": "Shape", "options": []},
    ]
    _ = _bash(wire, tool="AskUserQuestion", input={"questions": questions})
    reply = wire.answer(
        Answer(
            ask_id="r1",
            option="allow_once",
            updated_input={
                "answers": {"Color": "Red", "size_id": "Big", "Which shape?": "Round"},
            },
        ),
    )
    assert reply is not None
    assert reply.frame == _response(
        "r1",
        behavior="allow",
        updatedInput={
            "questions": questions,
            "answers": {
                "Which color?": "Red",
                "Which size?": "Big",
                "Which shape?": "Round",
            },
        },
    )


def test_an_answer_naming_no_question_is_refused() -> None:
    wire = ClaudeWire(cwd=_CWD)
    _ = _bash(wire, tool="AskUserQuestion", input={"questions": _QUESTIONS})
    with pytest.raises(ValueError, match=r"^the ask has no question named 'Size'$"):
        _ = wire.answer(
            Answer(
                ask_id="r1",
                option="allow_once",
                updated_input={"answers": {"Size": "Big"}},
            ),
        )
    assert wire.answer(Answer(ask_id="r1", option="deny")) is not None


def test_a_question_cannot_be_submitted_without_answers() -> None:
    wire = ClaudeWire(cwd=_CWD)
    _ = _bash(wire, tool="AskUserQuestion", input={"questions": _QUESTIONS})
    with pytest.raises(
        ValueError,
        match=r"^a question is answered with updated_input\.answers$",
    ):
        _ = wire.answer(Answer(ask_id="r1", option="allow_once"))
    assert wire.answer(Answer(ask_id="r1", option="deny")) is not None


def test_a_skipped_question_is_a_denial_the_agent_can_read() -> None:
    wire = ClaudeWire(cwd=_CWD)
    _ = _bash(wire, tool="AskUserQuestion", input={"questions": _QUESTIONS})
    reply = wire.answer(Answer(ask_id="r1", option="deny"))
    assert reply is not None
    assert reply.frame == _response(
        "r1",
        behavior="deny",
        message="The person declined this request.",
    )


def test_plan_feedback_is_the_denial_message() -> None:
    wire = ClaudeWire(cwd=_CWD)
    _ = _bash(wire, tool="ExitPlanMode", input={"plan": "1. Do it"})
    reply = wire.answer(
        Answer(ask_id="r1", option="deny", message="Add a verify step."),
    )
    assert reply is not None
    assert reply.frame == _response(
        "r1",
        behavior="deny",
        message="Add a verify step.",
    )


def test_an_answer_names_an_option_the_ask_offers() -> None:
    wire = ClaudeWire(cwd=_CWD)
    _ = _bash(wire)
    with pytest.raises(ValueError, match="allow_session"):
        _ = wire.answer(Answer(ask_id="r1", option="allow_session"))
    assert wire.answer(Answer(ask_id="r1", option="allow_once")) is not None


def test_the_first_answer_wins() -> None:
    wire = ClaudeWire(cwd=_CWD)
    _ = _bash(wire)
    assert wire.answer(Answer(ask_id="r1", option="deny")) is not None
    assert wire.answer(Answer(ask_id="r1", option="allow_once")) is None
    assert wire.answer(Answer(ask_id="never", option="allow_once")) is None


def test_release_closes_every_open_ask_once() -> None:
    wire = ClaudeWire(cwd=_CWD)
    _ = _bash(wire, request_id="a")
    _ = _bash(wire, request_id="b")
    assert wire.release() == (
        Resolved(ask_id="a", option="", by="released", decision="released"),
        Resolved(ask_id="b", option="", by="released", decision="released"),
    )
    assert wire.release() == ()


def test_an_mcp_elicitation_is_an_ask_the_person_accepts_or_declines() -> None:
    wire = ClaudeWire(cwd=_CWD)
    ask = _opened(
        wire.read(
            _request(
                "elicitation",
                mcp_server_name="docs",
                title="Pick a canvas",
                message="Which canvas?",
                mode="url",
                url="https://example.test/pick",
                requested_schema={"type": "object"},
            ),
        ),
    )
    assert ask == Ask(
        id="r2",
        provider="claude",
        kind="elicitation",
        title="Pick a canvas",
        tool="docs",
        reason="Which canvas?",
        cwd="/work",
        input={
            "mode": "url",
            "url": "https://example.test/pick",
            "requested_schema": {"type": "object"},
        },
        options=(
            Option(id="allow_once", label="Accept", kind="allow_once"),
            Option(id="deny", label="Decline", kind="deny"),
        ),
    )
    accepted = wire.answer(
        Answer(ask_id="r2", option="allow_once", updated_input={"canvas": "w1"}),
    )
    assert accepted is not None
    assert accepted.frame == _response(
        "r2",
        action="accept",
        content={"canvas": "w1"},
    )


def test_a_declined_elicitation_declines() -> None:
    wire = ClaudeWire(cwd=_CWD)
    _ = wire.read(_request("elicitation", mcp_server_name="docs", message="Sign in"))
    reply = wire.answer(Answer(ask_id="r2", option="deny"))
    assert reply is not None
    assert reply.frame == _response("r2", action="decline")


def test_a_dialog_nobody_declared_is_cancelled_at_once() -> None:
    reading = ClaudeWire(cwd=_CWD).read(
        _request(
            "request_user_dialog",
            dialog_kind="refusal_fallback_prompt",
            payload={},
        ),
    )
    assert reading.events == ()
    assert reading.replies == (_response("r2", behavior="cancelled"),)


def _failure(request_id: str, error: str) -> _ClaudeFrame:
    return {
        "type": "control_response",
        "response": {"subtype": "error", "request_id": request_id, "error": error},
    }


def test_a_request_the_driver_cannot_serve_gets_an_error_not_silence() -> None:
    reading = ClaudeWire(cwd=_CWD).read(_request("hook_callback"))
    assert reading.replies == (
        _failure("r2", "unsupported control request: hook_callback"),
    )


def test_a_request_with_no_body_gets_an_error_too() -> None:
    reading = ClaudeWire(cwd=_CWD).read({"type": "control_request", "request_id": "r3"})
    assert reading.replies == (_failure("r3", "unsupported control request: "),)


def test_a_request_that_cannot_be_read_gets_an_error_and_opens_no_ask() -> None:
    wire = ClaudeWire(cwd=_CWD)
    reading = wire.read(_can_use_tool(input="not a mapping"))
    assert reading.events == ()
    (reply,) = reading.replies
    response = from_plain(reply["response"], dict[str, PlainTree])
    assert (response["subtype"], response["request_id"]) == ("error", "r1")
    assert str(response["error"]).startswith("unreadable can_use_tool request: ")
    assert wire.release() == ()


def test_a_question_without_questions_asks_none() -> None:
    ask = _bash(ClaudeWire(cwd=_CWD), tool="AskUserQuestion", input={})
    assert ask.questions == []


def test_a_wire_starts_on_the_session_it_is_given() -> None:
    assert ClaudeWire(cwd=_CWD).session_id == ""
    assert ClaudeWire(cwd=_CWD, session_id="s0").session_id == "s0"


def test_suggestions_are_worded_for_every_rule_and_folder() -> None:
    ask = _bash(
        ClaudeWire(cwd=_CWD),
        permission_suggestions=[
            {
                "type": "addRules",
                "rules": [
                    {"toolName": "Bash", "ruleContent": "ls *"},
                    {"toolName": "Read"},
                ],
                "behavior": "allow",
                "destination": "session",
            },
            {
                "type": "addDirectories",
                "directories": ["/a", "/b"],
                "destination": "session",
            },
        ],
    )
    assert [each.label for each in ask.options[1:3]] == [
        "Allow Bash(ls *), Read for this session",
        "Allow access to /a, /b for this session",
    ]


@pytest.mark.parametrize(
    "update",
    [
        {"type": "addRules", "behavior": "allow", "destination": "session"},
        {
            "type": "addRules",
            "rules": [],
            "behavior": "allow",
            "destination": "session",
        },
        {
            "type": "addRules",
            "rules": [{"toolName": "Bash"}],
            "behavior": "deny",
            "destination": "session",
        },
        {"type": "addDirectories", "destination": "session"},
        {"type": "addDirectories", "directories": [], "destination": "session"},
        {"type": "setMode", "mode": "bypassPermissions", "destination": "session"},
        {"type": "setMode", "destination": "session"},
        {
            "type": "replaceRules",
            "rules": [],
            "behavior": "allow",
            "destination": "session",
        },
        {"destination": "session"},
    ],
)
def test_a_suggestion_that_is_not_a_plain_allow_is_not_offered(
    update: _ClaudeFrame,
) -> None:
    ask = _bash(ClaudeWire(cwd=_CWD), permission_suggestions=[update])
    assert _kinds(ask) == [
        ("allow_once", "allow_once"),
        ("deny", "deny"),
        ("deny_stop", "deny_stop"),
    ]


def test_a_diff_needs_an_edit_and_both_strings() -> None:
    wire = ClaudeWire(cwd=_CWD)
    only_old = {"file_path": "/work/a.py", "old_string": "x\n"}
    only_new = {"file_path": "/work/a.py", "new_string": "x\n"}
    both = {"file_path": "/work/a.py", "old_string": "x\n", "new_string": "y\n"}
    assert _bash(wire, tool="Edit", input=only_old, request_id="a").diff == ""
    assert _bash(wire, tool="Edit", input=only_new, request_id="b").diff == ""
    assert _bash(wire, tool="Write", input=both, request_id="c").diff == ""
    assert _bash(wire, tool="Edit", input=both, request_id="d").diff != ""


@pytest.fixture(autouse=True)
def stream_process() -> Iterator[AsyncMock]:
    """Run the same scripted fake without launching an interpreter or signalling a PID."""
    processes: dict[int, _Process] = {}
    with ExitStack() as cleanup, pytest.MonkeyPatch.context() as boundary:
        start = AsyncMock(side_effect=partial(_spawn_stream, cleanup, processes))
        boundary.setattr(asyncio, "create_subprocess_exec", start)
        boundary.setattr(os, "killpg", partial(_signal_app, processes))
        yield start


async def _spawn_stream(
    cleanup: ExitStack,
    processes: dict[int, _Process],
    *argv: str,
    cwd: Path,
    env: Mapping[str, str],
    **options: object,
) -> Process:
    del options
    with (
        patch.object(sys, "stdin", StringIO()),
        patch.object(sys, "argv", list(argv[3:])),
        patch.object(os, "getcwd", return_value=str(cwd)),
        patch.dict(os.environ, env, clear=True),
    ):
        namespace = runpy.run_path(argv[3])
    cleanup.callback(cast(TextIOWrapper, namespace["log"]).close)
    answer = cast(Callable[[str], None], namespace["answer"])
    process = _process(code=None)
    processes[process.pid] = process
    process.stdin.write.side_effect = partial(_answer_stream, process, answer)
    return cast(Process, process)


def _answer_stream(
    process: _Process,
    answer: Callable[[str], None],
    data: bytes,
) -> None:
    if process.returncode is not None:
        raise BrokenPipeError
    output, error_bytes = StringIO(), BytesIO()
    status = None
    with TextIOWrapper(error_bytes, encoding="utf-8") as errors:
        with redirect_stdout(output), redirect_stderr(errors):
            try:
                for line in data.decode().splitlines(keepends=True):
                    answer(line)
            except SystemExit as ended:
                status = cast(int, ended.code)
        errors.flush()
        process.stderr.feed_data(error_bytes.getvalue())
    process.stdout.feed_data(output.getvalue().encode())
    if status is not None:
        process.returncode = status
        process.stdout.feed_eof()
        process.stderr.feed_eof()


def _expired(
    timeout: Callable[[float | None], asyncio.Timeout],
    delay: float | None,
) -> asyncio.Timeout:
    assert delay == 0.05
    return timeout(0)


def test_start_launches_claude_headless_over_stream_json(tmp_path: Path) -> None:
    async def go() -> tuple[_Run, str]:
        async with _Run(tmp_path, _INIT) as run:
            return run, await run.start(model="haiku", effort="low")

    run, session = asyncio.run(go())
    first = run.seen()[0]
    assert first["argv"] == [
        "-p",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
        "--thinking-display",
        "summarized",
        "--permission-prompt-tool",
        "stdio",
        "--model",
        "haiku",
        "--effort",
        "low",
        "--session-id",
        session,
    ]
    assert str(uuid.UUID(session)) == session
    assert run.driver.session_id == session
    assert first["env"] == "1"
    assert Path(str(first["cwd"])).samefile(tmp_path)
    assert [each["request"] for each in run.sent()] == [{"subtype": "initialize"}]


def test_a_resumed_session_keeps_its_id(tmp_path: Path) -> None:
    async def go() -> tuple[_Run, str]:
        async with _Run(tmp_path, _INIT, permission_mode="default") as run:
            return run, await run.start(resume="abc")

    run, session = asyncio.run(go())
    assert session == "abc"
    assert from_plain(run.seen()[0]["argv"], list[str])[10:] == [
        "--permission-mode",
        "default",
        "--resume",
        "abc",
    ]


def test_a_turn_reports_its_states(tmp_path: Path) -> None:
    async def go() -> list[_ClaudeEvent]:
        steps = (_INIT, _step(_says("hi"), _state("running"), _state("idle")))
        async with _Run(tmp_path, *steps) as run:
            _ = await run.start()
            events = run.driver.events()
            await run.driver.send_line("hi")
            return [await _claude_next(events), await _claude_next(events)]

    assert asyncio.run(go()) == [
        StateChanged(state="running"),
        StateChanged(state="idle"),
    ]


def test_an_ask_is_answered_from_chat(tmp_path: Path) -> None:
    async def go() -> tuple[_Run, list[_ClaudeEvent]]:
        async with _Run(tmp_path, *_ASK_STEPS) as run:
            events = await _claude_asked(run)
            await run.driver.answer(Answer(ask_id="r1", option="allow_once", who="jp"))
            return run, [await _claude_next(events) for _ in range(3)]

    run, seen = asyncio.run(go())
    assert seen == [
        Resolved(
            ask_id="r1",
            option="allow_once",
            by="chat",
            who="jp",
            decision="allow",
        ),
        StateChanged(state="running"),
        StateChanged(state="idle"),
    ]
    assert run.sent()[-1] == _response(
        "r1",
        behavior="allow",
        updatedInput=_BASH_INPUT,
    )


def test_the_ask_reaches_the_events_with_its_state(tmp_path: Path) -> None:
    async def go() -> list[_ClaudeEvent]:
        async with _Run(tmp_path, *_ASK_STEPS) as run:
            _ = await run.start()
            events = run.driver.events()
            await run.driver.send_line("ask")
            return [await _claude_next(events), await _claude_next(events)]

    state, ask = asyncio.run(go())
    assert state == StateChanged(state="requires_action")
    assert isinstance(ask, Ask)
    assert (ask.id, ask.kind, ask.command) == ("r1", "permission", "touch ok.txt")
    assert ask.cwd == str(tmp_path)


def test_an_interrupt_releases_the_ask_the_cli_withdraws(tmp_path: Path) -> None:
    async def go() -> tuple[_Run, _ClaudeEvent]:
        async with _Run(tmp_path, *_ASK_STEPS) as run:
            events = await _claude_asked(run)
            await run.driver.interrupt()
            return run, await _claude_next(events)

    run, released = asyncio.run(go())
    assert released == Resolved(
        ask_id="r1",
        option="",
        by="released",
        decision="released",
    )
    assert run.sent()[-1]["request"] == {"subtype": "interrupt"}


def test_a_late_answer_changes_nothing(tmp_path: Path) -> None:
    async def go() -> _Run:
        async with _Run(tmp_path, _INIT) as run:
            _ = await run.start()
            await run.driver.answer(Answer(ask_id="gone", option="allow_once"))
            return run

    assert len(asyncio.run(go()).sent()) == 1


def test_an_option_the_ask_lacks_is_refused_and_the_ask_stays_open(
    tmp_path: Path,
) -> None:
    async def go() -> list[_ClaudeEvent]:
        async with _Run(tmp_path, *_ASK_STEPS) as run:
            _ = await _claude_asked(run)
            with pytest.raises(ValueError, match="allow_session"):
                await run.driver.answer(Answer(ask_id="r1", option="allow_session"))
            await run.driver.stop()
            return await _rest(run.driver)

    assert asyncio.run(go()) == [
        Resolved(ask_id="r1", option="", by="released", decision="released"),
    ]


def test_a_cli_that_exits_releases_what_it_left_open(tmp_path: Path) -> None:
    async def go() -> list[_ClaudeEvent]:
        steps = (_INIT, _step(_says("die"), _can_use_tool(), exit=0))
        async with _Run(tmp_path, *steps) as run:
            _ = await run.start()
            await run.driver.send_line("die")
            return await _rest(run.driver)

    seen = asyncio.run(go())
    assert [type(each) for each in seen] == [Ask, Resolved]
    assert seen[1] == Resolved(
        ask_id="r1",
        option="",
        by="released",
        decision="released",
    )


def test_events_end_once_the_cli_is_gone(tmp_path: Path) -> None:
    async def go() -> tuple[list[_ClaudeEvent], list[_ClaudeEvent]]:
        async with _Run(tmp_path, _INIT) as run:
            _ = await run.start()
            await run.driver.stop()
            await run.driver.stop()
            return await _rest(run.driver), await _rest(run.driver)

    assert asyncio.run(go()) == ([], [])


def test_nothing_is_sent_to_a_cli_that_has_gone(tmp_path: Path) -> None:
    async def go() -> None:
        async with _Run(tmp_path, _INIT, _step(_says("die"), exit=0)) as run:
            _ = await run.start()
            await run.driver.send_line("die")
            _ = await _rest(run.driver)
            with pytest.raises(ConnectionError, match=r"^claude has ended$"):
                await run.driver.send_line("again")

    asyncio.run(go())


def test_a_failed_write_releases_the_ask_and_says_so(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def gone(self: Piped, data: bytes) -> bool:
        del self, data
        return False

    async def go() -> list[_ClaudeEvent]:
        async with _Run(tmp_path, *_ASK_STEPS) as run:
            events = await _claude_asked(run)
            monkeypatch.setattr(Piped, "write", gone)
            with pytest.raises(ConnectionError, match=r"^claude has ended$"):
                await run.driver.answer(Answer(ask_id="r1", option="allow_once"))
            monkeypatch.undo()
            return [await _claude_next(events)]

    assert asyncio.run(go()) == [
        Resolved(ask_id="r1", option="", by="released", decision="released"),
    ]


def test_a_line_that_is_not_a_frame_does_not_end_the_reader(tmp_path: Path) -> None:
    noise = _step(
        _says("noise"),
        "",
        "not json",
        "[1, 2]",
        {"type": 5},
        {"type": "control_response"},
        {
            "type": "control_response",
            "response": {"subtype": "success", "request_id": "nobody"},
        },
        {"state": "idle"},
        _state("idle"),
        stderr="warning\n",
    )

    async def go() -> list[_ClaudeEvent]:
        async with _Run(tmp_path, _INIT, noise) as run:
            _ = await run.start()
            events = run.driver.events()
            await run.driver.send_line("noise")
            return [await _claude_next(events)]

    assert asyncio.run(go()) == [StateChanged(state="idle")]


def test_a_last_frame_without_a_newline_is_still_read(tmp_path: Path) -> None:
    async def go() -> list[_ClaudeEvent]:
        bye = _step(_says("bye"), tail=json.dumps(_state("idle")), exit=0)
        async with _Run(tmp_path, _INIT, bye) as run:
            _ = await run.start()
            await run.driver.send_line("bye")
            return await _rest(run.driver)

    assert asyncio.run(go()) == [StateChanged(state="idle")]


def test_what_claude_wrote_to_stderr_explains_why_it_ended(tmp_path: Path) -> None:
    async def go() -> None:
        failing = _step(
            {"request": {"subtype": "initialize"}},
            stderr="not logged in\n",
            exit=1,
        )
        async with _Run(tmp_path, failing) as run:
            _ = await run.start()

    with pytest.raises(ConnectionError, match=r"^claude has ended: not logged in$"):
        asyncio.run(go())


def test_only_the_end_of_stderr_is_kept(tmp_path: Path) -> None:
    async def go() -> None:
        failing = _step(
            {"request": {"subtype": "initialize"}},
            stderr="x" * 600 + "tail\n",
            exit=1,
        )
        async with _Run(tmp_path, failing) as run:
            _ = await run.start()

    with pytest.raises(ConnectionError) as error:
        asyncio.run(go())
    assert str(error.value) == f"claude has ended: {'x' * 495}tail"


def test_a_pipe_that_still_accepts_bytes_is_not_written_to_after_the_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def open_pipe(self: Piped, data: bytes) -> bool:
        del self, data
        return True

    async def go() -> None:
        async with _Run(tmp_path, _INIT, _step(_says("die"), exit=0)) as run:
            _ = await run.start()
            await run.driver.send_line("die")
            _ = await _rest(run.driver)
            monkeypatch.setattr(Piped, "write", open_pipe)
            with pytest.raises(ConnectionError, match=r"^claude has ended$"):
                await run.driver.send_line("again")

    asyncio.run(go())


def test_a_request_the_driver_cannot_serve_is_answered_with_an_error(
    tmp_path: Path,
) -> None:
    async def go() -> _Run:
        serve = _step(
            _says("serve"),
            {
                "type": "control_request",
                "request_id": "u1",
                "request": {"subtype": "hook_callback"},
            },
        )
        seen = _step({"type": "control_response"}, _state("idle"))
        async with _Run(tmp_path, _INIT, serve, seen) as run:
            _ = await run.start()
            events = run.driver.events()
            await run.driver.send_line("serve")
            _ = await _claude_next(events)
            return run

    run = asyncio.run(go())
    assert run.sent()[-1] == {
        "type": "control_response",
        "response": {
            "subtype": "error",
            "request_id": "u1",
            "error": "unsupported control request: hook_callback",
        },
    }


def test_a_reply_that_cannot_be_written_does_not_end_the_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write = Piped.write

    async def refuse_replies(self: Piped, data: bytes) -> bool:
        return False if b"unsupported" in data else await write(self, data)

    async def go() -> list[_ClaudeEvent]:
        serve = _step(
            _says("serve"),
            {
                "type": "control_request",
                "request_id": "u1",
                "request": {"subtype": "hook_callback"},
            },
            _state("idle"),
        )
        async with _Run(tmp_path, _INIT, serve) as run:
            _ = await run.start()
            events = run.driver.events()
            monkeypatch.setattr(Piped, "write", refuse_replies)
            await run.driver.send_line("serve")
            return [await _claude_next(events)]

    assert asyncio.run(go()) == [StateChanged(state="idle")]


def test_stderr_that_is_not_text_is_kept_as_far_as_it_reads(tmp_path: Path) -> None:
    async def go() -> None:
        failing = _step(
            {"request": {"subtype": "initialize"}},
            stderr_hex="ff6f6b",
            exit=1,
        )
        async with _Run(tmp_path, failing) as run:
            _ = await run.start()

    with pytest.raises(ConnectionError) as error:
        asyncio.run(go())
    assert str(error.value) == "claude has ended: \ufffdok"


def test_a_refusal_without_words_still_says_it_was_refused(tmp_path: Path) -> None:
    refused = _step(
        {"request": {"subtype": "set_model"}},
        {
            "type": "control_response",
            "response": {"subtype": "error", "request_id": "$request_id"},
        },
    )

    async def go() -> None:
        async with _Run(tmp_path, _INIT, refused) as run:
            _ = await run.start()
            await run.driver.settings(model="nope", effort="")

    with pytest.raises(RuntimeError, match=r"^request refused$"):
        asyncio.run(go())


def test_a_slash_command_is_a_line(tmp_path: Path) -> None:
    async def go() -> tuple[_Run, list[str]]:
        async with _Run(
            tmp_path,
            _INIT,
            _step({"type": "user"}, _state("idle")),
        ) as run:
            _ = await run.start()
            events = run.driver.events()
            printed = [
                await run.driver.command("compact", ""),
                await run.driver.command("/goal", "ship it"),
            ]
            _ = [await _claude_next(events), await _claude_next(events)]
            return run, printed

    run, printed = asyncio.run(go())
    assert [each["message"] for each in run.sent()[1:]] == [
        {"role": "user", "content": "/compact"},
        {"role": "user", "content": "/goal ship it"},
    ]
    # Claude logs what it says of a command it takes as a line.
    assert printed == ["", ""]


# As claude 2.1.309 answered it on 2026-10-23.
_REMOTE_CONTROL: Final = _step(
    {"request": {"subtype": "remote_control"}},
    _ack(
        session_url="https://claude.ai/code/session_1",
        connect_url="https://claude.ai/code?environment=",
        environment_id="",
        bridge_epoch=1,
        bridge_session_id="cse_1",
    ),
)


def test_remote_control_is_a_request_whose_answer_is_the_link(tmp_path: Path) -> None:
    """Typed as a line, claude answers "/rc isn't available in this environment"."""

    async def go() -> tuple[_Run, list[str]]:
        async with _Run(tmp_path, _INIT, _REMOTE_CONTROL) as run:
            _ = await run.start()
            printed = [
                await run.driver.command("rc", ""),
                await run.driver.command("/remote-control", "my phone"),
            ]
            return run, printed

    run, printed = asyncio.run(go())
    assert [each["request"] for each in run.sent()[1:]] == [
        {"subtype": "remote_control", "enabled": True},
        {"subtype": "remote_control", "enabled": True, "name": "my phone"},
    ]
    link = "Remote Control is active: continue on your phone or at "
    assert printed == [f"{link}https://claude.ai/code/session_1"] * 2


def test_remote_control_claude_refuses_raises_what_it_said(tmp_path: Path) -> None:
    refused = _step(
        {"request": {"subtype": "remote_control"}},
        {
            "type": "control_response",
            "response": {
                "subtype": "error",
                "request_id": "$request_id",
                "error": "Remote Control cannot be enabled from inside a remote session",
            },
        },
    )

    async def go() -> None:
        async with _Run(tmp_path, _INIT, refused) as run:
            _ = await run.start()
            with pytest.raises(RuntimeError, match="inside a remote session"):
                _ = await run.driver.command("rc", "")

    asyncio.run(go())


def test_settings_change_the_model_and_the_effort(tmp_path: Path) -> None:
    async def go() -> _Run:
        async with _Run(tmp_path, _INIT, _ANY_REQUEST) as run:
            _ = await run.start()
            await run.driver.settings(model="sonnet", effort="")
            await run.driver.settings(model="", effort="low")
            await run.driver.settings(model="", effort="")
            return run

    assert [each["request"] for each in asyncio.run(go()).sent()[1:]] == [
        {"subtype": "set_model", "model": "sonnet"},
        {"subtype": "apply_flag_settings", "settings": {"effortLevel": "low"}},
    ]


def test_a_refused_request_raises_what_the_cli_said(tmp_path: Path) -> None:
    refused = _step(
        {"request": {"subtype": "set_model"}},
        {
            "type": "control_response",
            "response": {
                "subtype": "error",
                "request_id": "$request_id",
                "error": "no such model",
            },
        },
    )

    async def go() -> None:
        async with _Run(tmp_path, _INIT, refused) as run:
            _ = await run.start()
            with pytest.raises(RuntimeError, match="no such model"):
                await run.driver.settings(model="nope", effort="")

    asyncio.run(go())


def test_start_stops_the_cli_when_initialize_is_refused(tmp_path: Path) -> None:
    refused = _step(
        {"request": {"subtype": "initialize"}},
        {
            "type": "control_response",
            "response": {
                "subtype": "error",
                "request_id": "$request_id",
                "error": "boom",
            },
        },
    )

    async def go() -> list[_ClaudeEvent]:
        async with _Run(tmp_path, refused) as run:
            with pytest.raises(RuntimeError, match="boom"):
                _ = await run.start()
            return await _rest(run.driver)

    assert asyncio.run(go()) == []


def test_start_fails_when_the_cli_exits_before_it_answers(tmp_path: Path) -> None:
    async def go() -> None:
        gone = _step({"request": {"subtype": "initialize"}}, exit=1)
        async with _Run(tmp_path, gone) as run:
            _ = await run.start()

    with pytest.raises(ConnectionError, match=r"^claude has ended$"):
        asyncio.run(go())


def test_start_fails_when_the_cli_never_answers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asyncio, "timeout", partial(_expired, asyncio.timeout))

    async def go() -> None:
        async with _Run(tmp_path, request_timeout_sec=0.05) as run:
            _ = await run.start()

    with pytest.raises(TimeoutError):
        asyncio.run(go())


def test_start_fails_without_a_claude_binary(tmp_path: Path) -> None:
    driver = ClaudeStreamDriver(argv=("claude-binary-that-does-not-exist",))
    with pytest.raises(FileNotFoundError):
        _ = asyncio.run(
            driver.start(cwd=tmp_path, resume="", model="", effort=""),
        )


def test_a_driver_must_be_started_and_started_once(tmp_path: Path) -> None:
    async def go() -> None:
        async with _Run(tmp_path, _INIT) as run:
            with pytest.raises(RuntimeError, match=r"^claude driver not started$"):
                await run.driver.send_line("hi")
            with pytest.raises(RuntimeError, match=r"^claude driver not started$"):
                await run.driver.answer(Answer(ask_id="r1", option="deny"))
            _ = await run.start()
            with pytest.raises(RuntimeError, match=r"^claude driver already started$"):
                _ = await run.start()

    asyncio.run(go())


def test_the_session_id_follows_a_clear(tmp_path: Path) -> None:
    async def go() -> str:
        cleared = _step(
            _says("/clear"),
            {"type": "system", "subtype": "init", "session_id": "fresh"},
            _state("idle"),
        )
        async with _Run(tmp_path, _INIT, cleared) as run:
            _ = await run.start()
            events = run.driver.events()
            await run.driver.command("clear", "")
            _ = await _claude_next(events)
            return run.driver.session_id

    assert asyncio.run(go()) == "fresh"


def test_a_driver_has_no_session_before_it_starts() -> None:
    driver: Driver = ClaudeStreamDriver()
    assert isinstance(driver, ClaudeStreamDriver)
    assert driver.session_id == ""


def test_stopping_a_driver_that_never_started_does_nothing() -> None:
    driver = ClaudeStreamDriver(argv=("claude-binary-that-does-not-exist",))

    asyncio.run(driver.stop())

    assert driver.session_id == ""


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
