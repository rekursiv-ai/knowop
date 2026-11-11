"""What crosses between an agent process and Trackinizer: events out, inputs in.

An event leaves as its class's name and its plain data. The server passes it on
unread, and a viewer reads it by that name. An input arrives as the kind its sender
posted and becomes what sagent's runtime takes, named by the author the server set.

::

    runtime event ──event_payload──▶ EventLine.payload ──▶ viewers
    Input         ──input_message──▶ runtime.push, runtime.halt or a slash line

Typical usage example:

  line = EventLine(kind=type(event).__name__, payload=event_payload(event))
  message = input_message(sent)        # An ``AgentToAgentMessage`` from ``sent.author``.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

from sagent.model import types as model_types
from sagent.runtime import types as runtime_types
from sagent.runtime.types import Halt
from sagent.session.types import (
    AgentToAgentMessage,
    UserMessage,
    record_classes,
    stored_plain,
    stored_record,
)
from sagent.tui.render import error_text
from sagent.tui.types import CommandOutput
from trackinizer.wire.wire_session_relay import Command


if TYPE_CHECKING:
    from collections.abc import Mapping

    from trackinizer.wire.json_types import JSON
    from trackinizer.wire.wire_session_relay import Input


__all__ = ["CLASSES", "event_payload", "input_message"]


CLASSES: Final[Mapping[str, type]] = MappingProxyType(
    {
        **{
            kind.__name__: kind
            for module in (model_types, runtime_types)
            for kind in cast("dict[str, object]", vars(module)).values()
            if isinstance(kind, type) and is_dataclass(kind)
        },
        **record_classes(),
        CommandOutput.__name__: CommandOutput,
    },
)
"""Every class an event or an input may name, by the name it is written under: the
session's records, each dataclass of sagent's runtime and model types, and what a
slash line says."""


def event_payload(event: object) -> JSON:
    """Return ``event`` as the plain data a viewer reads, each exception as its text."""
    return stored_plain(_worded(event), classes=CLASSES)


def input_message(sent: Input) -> AgentToAgentMessage | Halt | Command:
    """Return what the agent takes for ``sent``: a line as a message from its author.

    A line reaches the agent as sagent's own message between agents, so the model
    reads who sent it. Its sender is the author the server set, whatever the payload
    says.

    Args:
      sent: An input, as the server handed it down.

    Returns:
      message: A line, named by ``sent.author``; a halt; or a slash line.

    Raises:
      ValueError: The payload does not read as ``sent.kind``.

    """
    match sent.kind:
        case "UserMessage":
            said = stored_record(sent.payload, UserMessage)
            return AgentToAgentMessage(
                content=said.content,
                attachments=said.attachments,
                sender=sent.author,
                extra={"author": sent.author, "input": str(sent.id)},
            )
        case "Halt":
            return stored_record(sent.payload, Halt, classes=CLASSES)
        case "Command":
            return Command.model_validate(sent.payload)


# Treekle writes plain data, which an exception is not. An event holds the one that
# was raised; a viewer needs only what a person reads of it.
def _worded(value: object) -> object:
    """Return ``value`` with each exception its fields hold, however deep, as its text."""
    if isinstance(value, BaseException):
        return error_text(value)
    if is_dataclass(value) and not isinstance(value, type):
        changed: dict[str, object] = {}
        for field in fields(value):
            held = cast(object, getattr(value, field.name))
            worded = _worded(held)
            if worded is not held:
                changed[field.name] = worded
        if changed:
            return replace(value, **changed)
    return value
