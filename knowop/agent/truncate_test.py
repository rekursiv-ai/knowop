"""Bounding one captured record: what is cut, what is kept, and what says it was.

Each test builds a wire body the way capture does -- through
:class:`SessionRecordRow` -- and reads the result back the way the server's
readers do, so a cut that only looked right in memory cannot pass.
"""

from __future__ import annotations

from collections.abc import Mapping
from io import StringIO
from typing import TYPE_CHECKING, Final
from uuid import UUID

import base64
import json
import re

from trackinizer.runs.refusal import stand_in
from trackinizer.sagent.scripts.convert_session import claude
from trackinizer.sagent.session.types import (
    CONCRETE_TYPES,
    AgentToAgentMessage,
    Attachment,
    BashResult,
    ContextClear,
    IncompleteRecord,
    Thinking,
    ToolResult,
    UncategorizedRecord,
    UncategorizedToolResult,
    UserMessage,
)
from trackinizer.types.session_records import SessionRecordRow, search_text
from trackinizer.types.streams import Stderr, Stdin, Stdout
from trackinizer.wire.wire_session_ir import RecordBody
from treekle import immutable, json_dumps, json_loads, mutable

import httpx2
import pytest

from knowop.agent.truncate import (
    MAX_RECORD_BYTES,
    MAX_TOOL_RESULT_BYTES,
    NOTE,
    _cut_text,
    _named_media,
    _shrunk,
    bounded,
    bounded_payload,
    encoded_size,
    was_truncated,
    without_note,
)


if TYPE_CHECKING:
    from collections.abc import Iterator

    from trackinizer.sagent.session.types import SessionRecord
    from trackinizer.types.streams import TraxRecord
    from treekle import MutablePlainTree, PlainTree


# One cut string: what it kept, how much it says went, and what it kept after.
_CUT: Final = re.compile(
    r"(?P<head>.*)\n\[Trackinizer capture truncated: kept beginning/end; "
    r"omitted (?P<omitted>\d+) UTF-8 bytes\]\n(?P<tail>.*)",
    re.DOTALL,
)


_LOG: Final = (
    '\x1b[2m2026-09-17T22:45:01Z\x1b[0m \x1b[32mINFO\x1b[0m {"step": 120, '
    '"loss": 0.2531} ━━━━ 50%\n'
)
"""One line of a worker log: ANSI escapes, quotes and box drawing all widen on
the wire, which is how ~1 MB of output became 11.39 MB."""


def _shell(output: str, *, image: PlainTree = None) -> ToolResult[BashResult]:
    """Return a shell result holding ``output`` twice, as codex stores one."""
    extra: dict[str, PlainTree] = {"aggregated_output": output}
    if image is not None:
        extra["image"] = image
    return ToolResult(
        call_id="c",
        state="done",
        stdout=(output,),
        result=BashResult(
            command=("bash", "-lc", "find . -exec sed -n '1,180p' {} \\;"),
            exit_code=0,
        ),
        extra=immutable(extra),
    )


def _image(size: int) -> dict[str, PlainTree]:
    """Return a PNG image block of ``size`` bytes, as the messages API spells one."""
    data = base64.b64encode(bytes(range(256)) * (size // 256) + bytes(size % 256))
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data.decode()},
    }


def _image_read(image: dict[str, PlainTree]) -> str:
    """Return a claude session reading ``image``, every copy claude writes of it."""
    source = image["source"]
    assert isinstance(source, dict)
    lines = [
        {
            "type": "assistant",
            "uuid": "a",
            "sessionId": "s",
            "timestamp": "2026-10-24T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t",
                        "name": "Read",
                        "input": {"file_path": "/shot.png"},
                    },
                ],
            },
        },
        {
            "type": "user",
            "uuid": "b",
            "parentUuid": "a",
            "sessionId": "s",
            "timestamp": "2026-10-24T00:00:01Z",
            "message": {
                "role": "user",
                "content": [
                    {"tool_use_id": "t", "type": "tool_result", "content": [image]},
                ],
            },
            "toolUseResult": {
                "type": "image",
                "file": {"base64": source["data"], "type": "image/png"},
            },
        },
    ]
    return "".join(f"{json.dumps(line)}\n" for line in lines)


def _base64_sources(tree: MutablePlainTree) -> Iterator[str]:
    """Yield the data of every base64 media source in a JSON tree."""
    if isinstance(tree, list):
        for item in tree:
            yield from _base64_sources(item)
    if isinstance(tree, dict):
        if tree.get("type") == "base64" and isinstance(data := tree.get("data"), str):
            yield data
        for item in tree.values():
            yield from _base64_sources(item)


class TestAnEventPayload:
    """An event's plain data is cut as a record's strings are, to the bound given."""

    def test_one_that_fits_is_returned_as_it_is(self) -> None:
        payload = {"py/object": "ModelResponsePartial", "text": "hi"}

        assert bounded_payload(payload, max_bytes=1 << 10) is payload
        # As the client sends it: compact, non-ASCII kept raw.
        exact = len(json.dumps(payload, separators=(",", ":")).encode())
        assert bounded_payload(payload, max_bytes=exact) is payload

    def test_a_long_string_keeps_its_beginning_and_end_within_the_bound(self) -> None:
        text = _LOG * 2_000
        payload = {"py/object": "ModelResponsePartial", "text": text}

        cut = bounded_payload(payload, max_bytes=64 << 10)

        assert len(json_dumps(cut).encode()) <= 64 << 10
        assert cut["py/object"] == "ModelResponsePartial"
        kept = cut["text"]
        assert isinstance(kept, str)
        _assert_cut_from(text, kept)


class TestACutString:
    """One string cut: its beginning, the marker, its end, in the share it was given."""

    def test_it_takes_its_share_to_the_byte(self) -> None:
        """The marker's count has as many digits as the length, so nothing is left over."""
        assert len(_cut_text("a" * 20_000, cap=1_000, omitted=[]).encode()) == 1_000

    def test_one_its_share_s_length_is_kept_whole(self) -> None:
        assert _cut_text("a" * 1_000, cap=1_000, omitted=[]) == "a" * 1_000

    @pytest.mark.parametrize("unit", ["é", "€", "\U0001f600"])
    @pytest.mark.parametrize("offset", range(4))
    def test_a_cut_through_wide_characters_keeps_within_its_share(
        self,
        unit: str,
        offset: int,
    ) -> None:
        """Each end moves to a character's start away from the cut, never past it."""
        text = "a" * offset + unit * 2_000

        cut = _cut_text(text, cap=1_000, omitted=[])

        assert len(cut.encode()) <= 1_000
        _assert_cut_from(text, cut)

    def test_a_media_block_its_share_s_length_is_kept_and_one_byte_longer_named(
        self,
    ) -> None:
        image = _image(3_000)
        source = image["source"]
        assert isinstance(source, dict)
        data = source["data"]
        assert isinstance(data, str)

        block = mutable(image)

        assert _named_media(block, cap=len(data), omitted=[]) is None
        assert _named_media(block, cap=len(data) - 1, omitted=[]) is not None


class TestARecordThatFitsIsUntouched:
    """Small results stay byte-identical: nothing is re-encoded on the way."""

    def test_a_tool_result_under_its_budget_sends_the_same_bytes(self) -> None:
        body = _body(
            ToolResult(
                call_id="c",
                state="done",
                stdout=("é\x1b[0m" * 5_000,),
                result=BashResult(),
            ),
        )

        assert _wire(bounded(body)) == _wire(body)

    def test_a_message_between_the_two_budgets_is_not_cut(self) -> None:
        """Only a tool result gets the tight budget; prose gets the backstop."""
        body = _body(UserMessage(content="x" * MAX_TOOL_RESULT_BYTES))
        assert MAX_TOOL_RESULT_BYTES < encoded_size(body) < MAX_RECORD_BYTES

        assert _wire(bounded(body)) == _wire(body)


class TestEachBudgetIsExact:
    """A record its budget's size exactly is kept; one byte more is cut."""

    def test_a_body_within_the_tool_budget_is_not_read(self) -> None:
        """A kind this client cannot decode passes as it came, up to the byte."""
        unknown = _sized(
            RecordBody(idx=3, kind="Nope", payload={"x": ""}),
            MAX_TOOL_RESULT_BYTES,
        )

        assert bounded(unknown) is unknown
        with pytest.raises(ValueError, match="unknown session record kind 'Nope'"):
            _ = bounded(_sized(unknown, MAX_TOOL_RESULT_BYTES + 1))

    def test_a_message_its_budget_s_size_is_kept_and_one_byte_more_is_cut(self) -> None:
        words = _body(UserMessage(content="w" * (MAX_RECORD_BYTES // 3)))
        kept = _sized(words, MAX_RECORD_BYTES)

        assert bounded(kept) is kept
        cut = bounded(_sized(words, MAX_RECORD_BYTES + 1))
        assert encoded_size(cut) <= MAX_RECORD_BYTES
        assert was_truncated(_read(cut))


class TestAnOversizedToolResult:
    """The failing shape: one output, stored again in provider metadata."""

    def test_it_fits_its_budget(self) -> None:
        body = _body(_shell(_LOG * 4_000))
        assert encoded_size(body) > 2 * MAX_TOOL_RESULT_BYTES

        assert encoded_size(bounded(body)) <= MAX_TOOL_RESULT_BYTES

    def test_each_copy_keeps_its_beginning_and_end_around_the_marker(self) -> None:
        output = _LOG * 4_000
        record = _read(bounded(_body(_shell(output))))

        assert isinstance(record, ToolResult)
        assert isinstance(record.result, BashResult)
        copy = record.extra["aggregated_output"]
        assert isinstance(copy, str)
        for kept in (_stdout_text(record), copy):
            _assert_cut_from(output, kept)

    def test_the_note_counts_every_byte_left_out(self) -> None:
        output = _LOG * 4_000
        record = _read(bounded(_body(_shell(output))))

        assert isinstance(record, ToolResult)
        assert isinstance(record.result, BashResult)
        copy = record.extra["aggregated_output"]
        assert isinstance(copy, str)
        stated = sum(_omitted(kept) for kept in (_stdout_text(record), copy))
        assert record.extra[NOTE] == immutable({"omitted_bytes": stated})

    def test_search_text_is_derived_from_what_was_kept(self) -> None:
        cut = bounded(_body(_shell(_LOG * 4_000)))

        assert "Trackinizer capture truncated" in cut.text
        assert cut.text == search_text(_read(cut))

    def test_the_envelope_is_kept(self) -> None:
        body = _body(_shell(_LOG * 4_000))

        cut = bounded(body)

        assert cut.payload != body.payload
        assert cut.model_dump(exclude={"payload", "text"}) == body.model_dump(
            exclude={"payload", "text"},
        )


class TestTheCutKeepsTextValid:
    """Valid UTF-8 and valid JSON, whatever sits where the cut lands."""

    @pytest.mark.parametrize(
        "unit",
        [
            pytest.param("aé€😀", id="one-to-four-byte-characters"),
            pytest.param('\x00\x1b[31m\r\n\t\x7f"\\', id="control-characters"),
            pytest.param("\U0001f600", id="four-byte-only"),
        ],
    )
    def test_every_cut_reads_back_as_a_prefix_and_suffix(self, unit: str) -> None:
        output = unit * (MAX_TOOL_RESULT_BYTES // len(unit.encode()) * 2 + 1)
        cut = bounded(_body(_shell(output)))

        reread = RecordBody.model_validate_json(_wire(cut))
        record = _read(reread)

        assert isinstance(record, ToolResult)
        assert isinstance(record.result, BashResult)
        _assert_cut_from(output, _stdout_text(record))

    @pytest.mark.parametrize("offset", range(4))
    def test_no_cut_splits_a_character(self, offset: int) -> None:
        """Every alignment of a four-byte character against the cut."""
        output = "a" * offset + "\U0001f600" * (MAX_TOOL_RESULT_BYTES // 2)

        record = _read(bounded(_body(_shell(output))))

        assert isinstance(record, ToolResult)
        assert isinstance(record.result, BashResult)
        _assert_cut_from(output, _stdout_text(record))


class TestWhatIsNeverCut:
    def test_bytes_survive_whole(self) -> None:
        """An attachment is base64 of bytes: a cut would corrupt both."""
        data = bytes(range(256)) * 1_000
        record = ToolResult(
            call_id="c",
            state="done",
            stdout=(
                "x" * MAX_TOOL_RESULT_BYTES,
                Attachment(mime_descriptor="image/png", data=data),
            ),
            result=UncategorizedToolResult(),
        )

        cut = _read(bounded(_body(record)))

        assert isinstance(cut, ToolResult)
        assert cut.attachments[0].data == data
        assert was_truncated(cut)


class TestBytesTakeNoShareOfTheBudget:
    """What no cut shrinks is not what the budget bounds: the words beside it are."""

    def test_a_line_s_words_beside_a_large_picture_are_stored_whole(self) -> None:
        picture = Attachment(mime_descriptor="image/png", data=_bytes(1_400_000))
        words = "w" * (4 << 10)
        body = _body(
            AgentToAgentMessage(content=words, attachments=(picture,), sender="a@x.y"),
        )
        assert encoded_size(body) > MAX_RECORD_BYTES

        assert bounded(body) is body

    def test_a_tool_result_s_text_beside_its_picture_is_stored_whole(self) -> None:
        picture = Attachment(mime_descriptor="image/png", data=_bytes(300_000))
        result = ToolResult(
            call_id="c",
            state="done",
            stdout=("x" * (10 << 10), picture),
            result=UncategorizedToolResult(),
        )
        body = _body(result)
        assert encoded_size(body) > MAX_TOOL_RESULT_BYTES

        assert bounded(body) is body

    @pytest.mark.parametrize(
        "record",
        [
            pytest.param(
                Thinking(content="w" * (4 << 10), encrypted="s" * (1_100_000)),
                id="its-own-seal",
            ),
            pytest.param(
                ContextClear(
                    history=(
                        Thinking(content="why", encrypted="s" * 1_100_000),
                        UserMessage(content="w" * (4 << 10)),
                    ),
                ),
                id="a-seal-in-its-history",
            ),
        ],
    )
    def test_words_beside_a_seal_are_stored_whole(self, record: TraxRecord) -> None:
        body = _body(record)
        assert encoded_size(body) > MAX_RECORD_BYTES

        assert bounded(body) is body

    def test_a_line_s_words_before_a_large_text_file_are_kept(self) -> None:
        """A text file rides the line as prose, so the budget bounds it; the words lead."""
        words = "w" * (4 << 10)
        text = _LOG * 14_000
        record = _read(bounded(_body(UserMessage(content=f"{words}\n\n{text}"))))

        assert isinstance(record, UserMessage)
        assert record.content is not None
        assert record.content.startswith(words)
        assert was_truncated(record)

    def test_words_past_the_budget_are_still_cut_to_it_and_the_picture_kept(
        self,
    ) -> None:
        data = _bytes(300_000)
        picture = Attachment(mime_descriptor="image/png", data=data)
        words = _LOG * 10_000
        body = _body(UserMessage(content=words, attachments=(picture,)))

        cut = bounded(body)

        sent = len(json_dumps({"py/b64": base64.b64encode(data).decode()}).encode())
        assert encoded_size(cut) <= MAX_RECORD_BYTES + sent
        record = _read(cut)
        assert isinstance(record, UserMessage)
        assert record.attachments == (picture,)
        assert record.content is not None
        _assert_cut_from(words, record.content)


class TestAnImageIsKeptWholeOrNamed:
    """Half a base64 string is neither the image nor text, and the API refuses it."""

    def test_an_image_within_the_share_is_kept_whole(self) -> None:
        image = _image(30_000)
        body = _body(_shell(_LOG * 4_000, image=image))

        cut = bounded(body)

        assert encoded_size(cut) <= MAX_TOOL_RESULT_BYTES
        record = _read(cut)
        assert isinstance(record, ToolResult)
        assert isinstance(record.result, BashResult)
        assert record.extra["image"] == immutable(image)

    def test_an_image_over_the_share_becomes_text_naming_it(self) -> None:
        image = _image(MAX_TOOL_RESULT_BYTES)

        record = _read(bounded(_body(_shell("short", image=image))))

        assert isinstance(record, ToolResult)
        assert isinstance(record.result, BashResult)
        assert record.extra["image"] == immutable(
            {
                "type": "text",
                "text": "[Trackinizer capture omitted this image "
                "(image/png, 349528 bytes of base64)]",
            },
        )
        assert record.extra[NOTE] == immutable({"omitted_bytes": 349_528})

    def test_a_resumed_image_read_holds_no_broken_base64(self) -> None:
        """The incident: claude stores a read image three times, over the budget."""
        captured: list[SessionRecord] = []
        for record in claude.normalize(StringIO(_image_read(_image(119_209)))):
            cut = without_note(_read(bounded(_body(record))))
            assert not isinstance(cut, Stdin | Stdout | Stderr)
            captured.append(cut)
        resumed = StringIO()

        claude.denormalize(captured, resumed)

        for line in resumed.getvalue().splitlines():
            for data in _base64_sources(json_loads(line)):
                base64.b64decode(data, validate=True)
        assert "Trackinizer capture omitted this image" in resumed.getvalue()


class TestEveryKindHasABackstop:
    """The incident's class is one record over the limit, not one kind."""

    def test_a_message_over_the_backstop_fits_it(self) -> None:
        body = _body(UserMessage(content=_LOG * 10_000))

        cut = bounded(body)

        assert encoded_size(cut) <= MAX_RECORD_BYTES
        record = _read(cut)
        assert isinstance(record, UserMessage)
        assert record.content is not None
        _assert_cut_from(_LOG * 10_000, record.content)

    def test_a_nested_history_is_reached(self) -> None:
        body = _body(ContextClear(history=(UserMessage(content=_LOG * 10_000),)))

        assert encoded_size(bounded(body)) <= MAX_RECORD_BYTES

    def test_a_seal_in_a_nested_history_is_kept_whole(self) -> None:
        """A provider refuses a seal cut short, so a compaction's thought keeps it."""
        seal = base64.b64encode(b"s" * 450_000).decode()
        # Two long messages: the share each string keeps falls below the seal's length.
        history = (
            Thinking(content="why", encrypted=seal),
            UserMessage(content=_LOG * 10_000),
            UserMessage(content=_LOG * 10_000),
        )

        record = _read(bounded(_body(ContextClear(history=history))))

        assert isinstance(record, ContextClear)
        thought = record.history[0]
        assert isinstance(thought, Thinking)
        assert thought.encrypted == seal
        assert was_truncated(record)


class TestTheLossyNote:
    """A cut record says so wherever the record can hold it."""

    @pytest.mark.parametrize(
        "record",
        [
            pytest.param(_shell(_LOG * 4_000), id="extra"),
            pytest.param(
                UncategorizedRecord(kind="k", payload=immutable({"x": _LOG * 12_000})),
                id="payload",
            ),
            pytest.param(IncompleteRecord(text=_LOG * 10_000), id="marker-only"),
            pytest.param(Stdout(text=_LOG * 10_000), id="stream"),
        ],
    )
    def test_a_cut_record_reports_it(self, record: TraxRecord) -> None:
        assert not was_truncated(record)

        assert was_truncated(_read(bounded(_body(record))))

    def test_a_record_that_fits_reports_nothing(self) -> None:
        assert not was_truncated(_read(bounded(_body(_shell("short")))))

    def test_a_stand_in_for_a_refused_record_is_one_cut(self) -> None:
        """The server refused the record whole, so its place holds none of it."""
        refused = stand_in(_body(UserMessage(content="\x00")), reason="NUL byte")

        assert was_truncated(_read(refused))

    @pytest.mark.parametrize(
        "record",
        [
            pytest.param(_shell(_LOG * 4_000), id="extra"),
            pytest.param(
                UncategorizedRecord(kind="k", payload=immutable({"x": _LOG * 12_000})),
                id="payload",
            ),
        ],
    )
    def test_the_note_comes_off_and_the_marker_stays(self, record: TraxRecord) -> None:
        """A writer replays an uncategorized payload verbatim, note and all."""
        cut = _read(bounded(_body(record)))

        stored = json_dumps(
            without_note(cut),
            sort=False,
            default_concrete_types=CONCRETE_TYPES,
        )

        assert NOTE not in stored
        assert "Trackinizer capture truncated" in stored


class TestTheSharedLengthShrinks:
    """The length each string may keep, tried in turn until what is cut fits."""

    def test_a_cut_that_exactly_fits_is_taken(self) -> None:
        tried: list[int] = []

        def cut(cap: int) -> int:
            tried.append(cap)
            return cap

        assert _shrunk(4_096, cut=cut, size=lambda made: made) == 4_096
        assert tried == [4_096]

    def test_the_next_length_is_in_proportion_with_an_eighth_to_spare(self) -> None:
        """Twice over at 8192 bytes: half of it, less an eighth, is 3584."""
        tried: list[int] = []

        def cut(cap: int) -> int:
            tried.append(cap)
            return cap

        assert _shrunk(8_192, cut=cut, size=lambda made: 2 * made) == 3_584
        assert tried == [8_192, 3_584]


def test_the_size_is_what_the_http_client_sends() -> None:
    """Measured as ``httpx2`` encodes ``json=``: compact, non-ASCII kept raw."""
    body = _body(
        ToolResult(
            call_id="c",
            state="done",
            stdout=('é€😀\x00\x1b[31m"\\\n',),
            result=BashResult(),
            extra=immutable({"duration": 0.1, "nested": [{"a": None, "b": True}]}),
        ),
    )

    sent = httpx2.Request("POST", "http://server/", json=body.model_dump(mode="json"))

    assert encoded_size(body) == len(sent.content)


def _sized(body: RecordBody, size: int) -> RecordBody:
    """Return ``body`` padded in its payload's ``extra``, which is not searched, to ``size``."""
    payload = dict(body.payload)
    extra = payload.get("extra")
    padded = {**(dict(extra) if isinstance(extra, Mapping) else {}), "pad": ""}
    unpadded = encoded_size(
        body.model_copy(update={"payload": {**payload, "extra": padded}}),
    )
    padded["pad"] = "p" * (size - unpadded)
    return body.model_copy(update={"payload": {**payload, "extra": padded}})


def _bytes(size: int) -> bytes:
    """Return ``size`` bytes, as a picture's are to a cut: none of them text."""
    return bytes(range(256)) * (size // 256) + bytes(size % 256)


def _body(record: TraxRecord) -> RecordBody:
    """Return the wire body capture builds for ``record``, before any cut."""
    return RecordBody.of(
        SessionRecordRow.of(session_id=UUID(int=0), part=0, idx=3, record=record),
    )


def _read(body: RecordBody) -> TraxRecord:
    """Return the record a server reader decodes from ``body``."""
    return body.row(UUID(int=0), 0).record()


def _wire(body: RecordBody) -> bytes:
    """Return ``body`` as the HTTP client encodes it."""
    return httpx2.Request(
        "POST",
        "http://server/",
        json=body.model_dump(mode="json"),
    ).content


def _assert_cut_from(original: str, kept: str) -> None:
    """Fail unless ``kept`` is ``original``'s start and end around a true count."""
    match = _CUT.fullmatch(kept)
    assert match is not None, "no marker between a beginning and an end"
    head, tail = match["head"], match["tail"]
    assert head
    assert tail
    assert original.startswith(head)
    assert original.endswith(tail)
    omitted = len(original.encode()) - len(head.encode()) - len(tail.encode())
    assert int(match["omitted"]) == omitted
    assert kept.encode().decode() == kept


def _stdout_text(record: ToolResult[object, object, object]) -> str:
    """Return the single text item of ``record.stdout``."""
    [text] = [item for item in record.stdout if isinstance(item, str)]
    return text


def _omitted(kept: str) -> int:
    """Return the byte count the marker in ``kept`` states."""
    match = _CUT.fullmatch(kept)
    assert match is not None
    return int(match["omitted"])


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
