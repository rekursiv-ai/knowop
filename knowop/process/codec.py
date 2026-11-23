"""What crosses between an agent process and Trackinizer: events out, inputs in.

An event leaves as its class's name and its plain data. The server passes it on
unread, and a viewer reads it by that name. An input arrives as the kind its sender
posted and becomes what sagent's runtime takes, named by the author the server set.

::

    runtime event ──event_payload──▶ EventLine.payload ──▶ viewers
    Input         ──input_message──▶ runtime.push, runtime.halt or a slash line

Typical usage example:

  line = EventLine(kind=type(event).__name__, payload=event_payload(event))
  message = input_message(sent)        # An ``AgentMessage`` from ``sent.author``.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast
from uuid import UUID

import json

from sagent.journal import types as session_types
from sagent.journal.types import (
    AgentMessage,
    UserMessage,
    stored_plain,
    stored_record,
)
from sagent.model import types as model_types
from sagent.runtime import types as runtime_types
from sagent.runtime.types import ModelHalt
from sagent.tui.render import error_text
from sagent.tui.types import CommandOutput
from trackinizer.wire.wire_session_relay import Command
from trackinizer.wire.wire_sessions import CONTEXT_TRAILER, LINE_CANVAS
from treekle import from_plain


if TYPE_CHECKING:
    from collections.abc import Mapping

    from trackinizer.wire.json_types import JSON
    from trackinizer.wire.wire_session_relay import Input


__all__ = ["CLASSES", "event_payload", "input_message", "told_canvas"]


CLASSES: Final[Mapping[str, type]] = MappingProxyType(
    {
        **{
            kind.__name__: kind
            for module in (model_types, session_types, runtime_types)
            for kind in cast("dict[str, object]", vars(module)).values()
            if isinstance(kind, type) and is_dataclass(kind)
        },
        CommandOutput.__name__: CommandOutput,
    },
)
"""Every class an event or an input may name, by the name it is written under: the
session's records and outcomes, each dataclass of sagent's runtime and model types,
and what a slash line says."""


def event_payload(event: object) -> JSON:
    """Return ``event`` as the plain data a viewer reads, each exception as its text."""
    return stored_plain(_worded(event), classes=CLASSES)


def input_message(sent: Input) -> AgentMessage | ModelHalt | Command:
    """Return what the agent takes for ``sent``: a line as a message from its author.

    A line reaches the agent as sagent's own message between agents, so the model
    reads who sent it. Its sender is the author the server set, whatever the payload
    says: a person's email, or ``label#number`` for another session's agent. Its
    recipient is the agent the input names; ``None``, for the root. A line whose
    ``extra`` names the canvas it was sent from (``canvas_id``, a UUID) tells the
    agent so after its words, behind ``CONTEXT_TRAILER``, with the tools and the
    command that act on it, and keeps it in the message's own ``extra``
    (:func:`told_canvas`); Chat names it only when it has not told this session's
    agent that canvas yet. A line's ``extra.files``, the name, kind and size of each
    file Chat attached to it, is kept in the message's, which Chat draws the stored
    line's files from without reading their bytes.

    Args:
      sent: An input, as the server handed it down.

    Returns:
      message: A line or a halt, named by ``sent.author``; or a slash line.

    Raises:
      ValueError: The payload does not read as ``sent.kind``, or a line names a
        sender of its own.

    """
    match sent.kind:
        case "UserMessage":
            line = stored_record(sent.payload, UserMessage)
            if line.sender is not None:
                raise ValueError("A line names no sender: the server says who sent it.")
            canvas = _canvas(line)
            files = line.extra.get("files")
            return AgentMessage(
                content=_told(line, canvas=canvas),
                attachments=line.attachments,
                sender=sent.author,
                recipient=sent.recipient or None,
                extra={
                    "author": sent.author,
                    "input": str(sent.id),
                    **({} if canvas is None else {"canvas_id": str(canvas)}),
                    **({} if files is None else {"files": files}),
                },
            )
        case "ModelHalt":
            return replace(
                stored_record(sent.payload, ModelHalt, classes=CLASSES),
                sender=sent.author,
            )
        case "Command":
            return Command.model_validate(sent.payload)


def told_canvas(message: AgentMessage) -> UUID | None:
    """Return the canvas a line from Chat named, as :func:`input_message` keeps it.

    Args:
      message: A message the agent was sent, a stored one included.

    Returns:
      canvas: The canvas; ``None`` for a message that named none.

    """
    named = message.extra.get("canvas_id")
    return None if named is None else UUID(from_plain(named, str))


# The canvas is a hint from the sender's page: one that is no UUID is left out, and
# the line goes as typed.
def _canvas(line: UserMessage) -> UUID | None:
    """Return the canvas ``line`` names, if it names one."""
    try:
        return UUID(from_plain(line.extra.get(LINE_CANVAS), str))
    except ValueError:
        return None


def _told(line: UserMessage, *, canvas: UUID | None) -> str | None:
    """Return ``line``'s words, then ``canvas`` and what acts on it, if it names one."""
    if canvas is None:
        return line.content
    context = json.dumps({"canvas_id": str(canvas)})
    # ``knowop``, not ``trax``: a host's ``trax`` may be a wrapper that wants a profile.
    how = (
        "Your Canvas tool (show, capture, mark and the rest) acts on it; with a "
        f"shell, `knowop canvas {canvas}` reads and changes it too, and `knowop canvas "
        "--help` says how. On the first line from the person that says what this "
        "chat is about, this one or a later one, `read` the canvas, and `rename` it "
        "while its name is `unnamed-canvas#N`."
    )
    return f"{line.content or ''}{CONTEXT_TRAILER}{context}\n{how}"


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
