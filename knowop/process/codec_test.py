"""The codec: what an event is written as, and what an input becomes."""

from __future__ import annotations

from dataclasses import is_dataclass
from types import UnionType
from typing import (
    TYPE_CHECKING,
    TypeAliasType,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)
from uuid import UUID

from sagent.providers.openai.wire import user_content
from sagent.runtime.types import (
    AgentIdle,
    ChildEvent,
    Halt,
    ModelResponseError,
    ModelResponsePartial,
    RuntimeEvent,
)
from sagent.session.types import (
    AgentToAgentMessage,
    Attachment,
    UserMessage,
    stored_record,
)
from sagent.tui.render import error_text
from sagent.tui.types import CommandOutput
from trackinizer.wire.wire_session_relay import Command, Input
from trackinizer.wire.wire_sessions import CONTEXT_TRAILER
from treekle import PlainTree, immutable

import pytest

from knowop.process.codec import CLASSES, event_payload, input_message


if TYPE_CHECKING:
    from trackinizer.wire.json_types import JSON
    from trackinizer.wire.wire_session_relay import InputKind


def test_what_an_event_is_written_as_is_unchanged() -> None:
    assert event_payload(ModelResponsePartial(text="hi", record=2)) == {
        "py/object": "ModelResponsePartial",
        "text": "hi",
        "record": 2,
    }


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(event, id=type(event).__name__)
        for event in (
            ModelResponsePartial(text="hi"),
            AgentIdle(),
            AgentToAgentMessage(sender="dan@example.com", content="hi"),
            ChildEvent(label="Agent_0", inner=AgentIdle()),
            CommandOutput(text="Model: opus", tone="note", command="/model"),
        )
    ],
)
def test_an_event_reads_back_as_it_was_written(event: object) -> None:
    assert stored_record(event_payload(event), type(event), classes=CLASSES) == event


def test_an_exception_is_written_as_the_text_a_person_reads() -> None:
    payload = event_payload(ModelResponseError(exception=RuntimeError("boom")))

    assert payload == {
        "py/object": "ModelResponseError",
        "exception": "RuntimeError: boom",
    }
    assert payload["exception"] == error_text(RuntimeError("boom"))


def test_an_exception_inside_a_child_s_event_is_written_as_its_text_too() -> None:
    held = ChildEvent(
        label="Agent_0",
        inner=ModelResponseError(exception=RuntimeError("boom")),
    )

    assert event_payload(held) == {
        "py/object": "ChildEvent",
        "label": "Agent_0",
        "inner": {"py/object": "ModelResponseError", "exception": "RuntimeError: boom"},
    }


def test_every_class_an_event_can_hold_is_named() -> None:
    held = _held(RuntimeEvent)

    assert ChildEvent in held
    assert {
        kind.__name__ for kind in held if CLASSES.get(kind.__name__) is not kind
    } == set()


# ``event_payload`` rewords an exception a field holds itself. One inside a tuple or
# a mapping would reach treekle as it is, which cannot write it.
def test_no_event_holds_an_exception_inside_a_container() -> None:
    assert {
        f"{kind.__name__}.{name}"
        for kind in _held(RuntimeEvent)
        for name, hint in cast("dict[str, object]", get_type_hints(kind)).items()
        if _contains_exception(hint)
    } == set()
    assert _contains_exception(tuple[ValueError, ...])
    assert not _contains_exception(ValueError | None)


def test_a_person_s_line_becomes_a_message_named_by_its_author() -> None:
    said = UserMessage(
        content="hi",
        attachments=(Attachment(mime_descriptor="text/plain", data=b"a"),),
        extra={"author": "eve@example.com"},
    )

    assert input_message(
        _input("UserMessage", event_payload(said)),
    ) == AgentToAgentMessage(
        content="hi",
        attachments=said.attachments,
        sender="dan@example.com",
        extra={
            "author": "dan@example.com",
            "input": "00000000-0000-0000-0000-000000000007",
        },
    )


def test_a_line_keeps_the_files_it_names_for_whoever_draws_it() -> None:
    files: PlainTree = [{"name": "shot.png", "type": "image/png", "size": 1}]
    said = UserMessage(
        content="what is this?",
        attachments=(Attachment(mime_descriptor="image/png", data=b"a"),),
        extra={"files": files, "author": "eve@example.com", "input": "x"},
    )

    message = input_message(_input("UserMessage", event_payload(said)))

    assert isinstance(message, AgentToAgentMessage)
    assert message.attachments == said.attachments
    assert message.extra == {
        "author": "dan@example.com",
        "input": "00000000-0000-0000-0000-000000000007",
        "files": immutable(files),
    }


def test_a_line_names_the_agent_it_is_for() -> None:
    sent = _input("UserMessage", {"py/object": "UserMessage", "content": "status?"})

    message = input_message(sent.model_copy(update={"recipient": "lead"}))

    assert isinstance(message, AgentToAgentMessage)
    assert message.recipient == "lead"


def test_a_line_of_files_and_no_words_gives_the_model_their_list_and_the_picture() -> (
    None
):
    listed = "\n\nFiles attached to this message:\n- shot.png (image/png, 1 B)\n"
    picture = Attachment(mime_descriptor="image/png", data=b"a")
    said = UserMessage(content=listed, attachments=(picture,))

    message = input_message(_input("UserMessage", event_payload(said)))

    assert isinstance(message, AgentToAgentMessage)
    assert user_content(message) == (f"[from dan@example.com]: {listed}", (picture,))


def test_a_line_that_names_its_canvas_tells_the_agent_after_its_words() -> None:
    canvas = "c5286865-67b6-4bd8-ab51-e06e10c326c5"
    said = UserMessage(content="what is this?", extra={"canvas_id": canvas})

    assert input_message(
        _input("UserMessage", event_payload(said)),
    ) == AgentToAgentMessage(
        content=(
            f'what is this?{CONTEXT_TRAILER}{{"canvas_id": "{canvas}"}}\n'
            f"Read or change it with `knowop canvas {canvas}`; "
            "`knowop canvas --help` says how."
        ),
        sender="dan@example.com",
        extra={
            "author": "dan@example.com",
            "input": "00000000-0000-0000-0000-000000000007",
        },
    )


def test_a_line_of_no_words_that_names_its_canvas_tells_only_the_canvas() -> None:
    canvas = "c5286865-67b6-4bd8-ab51-e06e10c326c5"
    said = UserMessage(
        attachments=(Attachment(mime_descriptor="image/png", data=b"a"),),
        extra={"canvas_id": canvas},
    )

    message = input_message(_input("UserMessage", event_payload(said)))

    assert isinstance(message, AgentToAgentMessage)
    assert (message.content or "").startswith(
        f'{CONTEXT_TRAILER}{{"canvas_id": "{canvas}"}}\n',
    )


@pytest.mark.parametrize(
    "extra",
    [{}, {"canvas_id": ""}, {"canvas_id": "c5286865"}, {"canvas_id": 7}],
)
def test_a_line_naming_no_canvas_reads_as_it_was_typed(extra: dict[str, JSON]) -> None:
    said = UserMessage(content="hi", extra=extra)

    message = input_message(_input("UserMessage", event_payload(said)))

    assert isinstance(message, AgentToAgentMessage)
    assert message.content == "hi"


def test_a_halt_reads_as_a_halt() -> None:
    assert input_message(_input("Halt", event_payload(Halt()))) == Halt()


def test_a_slash_line_reads_as_a_command() -> None:
    assert input_message(_input("Command", {"line": "/model opus"})) == Command(
        line="/model opus",
    )


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        ("UserMessage", {"py/object": "Halt"}),
        ("UserMessage", {"py/object": "UserMessage", "sender": "eve@example.com"}),
        # Another record with words would pass for a line, under a sender of its own.
        ("UserMessage", {"py/object": "AgentToAgentMessage", "content": "hi"}),
        ("Halt", {"py/object": "UserMessage", "content": "hi"}),
        ("Command", {"line": ""}),
        # A line with no slash would be pushed as a user's message with no sender.
        ("Command", {"line": "model opus"}),
    ],
)
def test_a_payload_that_is_not_its_kind_does_not_read(
    kind: InputKind,
    payload: JSON,
) -> None:
    with pytest.raises(ValueError, match=r".+"):
        _ = input_message(_input(kind, payload))


def _input(kind: InputKind, payload: JSON) -> Input:
    """Return an input of ``kind`` that the server stamped as Dan's seventh."""
    return Input(id=UUID(int=7), author="dan@example.com", kind=kind, payload=payload)


def _held(hint: object) -> set[type]:
    """Return each dataclass a value of type ``hint`` can hold, however deep."""
    return {
        kind
        for kind in _named(hint, seen=set())
        if isinstance(kind, type) and is_dataclass(kind)
    }


# An alias may name itself, as a tree of plain data does, so each is walked once.
def _named(hint: object, *, seen: set[object]) -> set[object]:
    """Add to ``seen`` each alias and class ``hint`` names, however deep."""
    if isinstance(hint, (TypeAliasType, type)):
        if hint in seen:
            return seen
        seen.add(hint)
    if isinstance(hint, TypeAliasType):
        parts: tuple[object, ...] = (cast(object, hint.__value__),)
    elif isinstance(hint, type) and is_dataclass(hint):
        parts = tuple(cast("dict[str, object]", get_type_hints(hint)).values())
    else:
        parts = cast("tuple[object, ...]", get_args(hint))
    for part in parts:
        _ = _named(part, seen=seen)
    return seen


def _contains_exception(
    hint: object,
    *,
    contained: bool = False,
    seen: frozenset[object] = frozenset(),
) -> bool:
    """Whether ``hint`` names an exception inside a container, not as the value itself."""
    if isinstance(hint, TypeAliasType):
        return hint not in seen and _contains_exception(
            cast(object, hint.__value__),
            contained=contained,
            seen=seen | {hint},
        )
    if isinstance(hint, type):
        return contained and issubclass(hint, BaseException)
    inside = contained or get_origin(hint) is not UnionType
    return any(
        _contains_exception(part, contained=inside, seen=seen)
        for part in cast("tuple[object, ...]", get_args(hint))
    )


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
