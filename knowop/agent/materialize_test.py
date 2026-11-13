"""Writing a stored sagent session back out as a journal this host continues."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import re
import stat

from trackinizer.sagent.scripts.convert_session import sagent
from trackinizer.sagent.session.jsonl import record_line
from trackinizer.sagent.session.types import (
    AssistantMessage,
    IncompleteRecord,
    SessionRecord,
    Thinking,
    UserMessage,
)

import pytest

from knowop.agent.errors import CiphertextDroppedError
from knowop.agent.materialize import materialize


if TYPE_CHECKING:
    from pathlib import Path


class TestTheJournal:
    """A journal in a folder of its own, holding the records it was given."""

    def test_the_journal_sits_in_a_new_folder_its_id_names(
        self,
        tmp_path: Path,
    ) -> None:
        """Sagent names a session by its journal's folder, and resumes by that name."""
        # Not made yet, as a session's journals folder is not before its first restore.
        folder = tmp_path / "journals" / "session"

        path = materialize([UserMessage(content="hi")], sealed=[None], folder=folder)

        assert path == folder / path.parent.name / "session.jsonl"
        assert UUID(hex=path.parent.name).hex == path.parent.name
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700

    def test_each_write_takes_a_folder_of_its_own(self, tmp_path: Path) -> None:
        first = materialize([UserMessage(content="a")], sealed=[], folder=tmp_path)
        second = materialize([UserMessage(content="b")], sealed=[], folder=tmp_path)

        assert first.parent != second.parent
        assert sorted(each.name for each in tmp_path.iterdir()) == sorted(
            [first.parent.name, second.parent.name],
        )

    def test_the_journal_is_the_records_it_was_given_line_for_line(
        self,
        tmp_path: Path,
    ) -> None:
        records = [
            UserMessage(content="what is 2+2?"),
            AssistantMessage(content="4"),
        ]

        path = materialize(records, sealed=[None, None], folder=tmp_path)

        assert path.read_bytes() == b"".join(map(record_line, records))
        with path.open(encoding="utf-8") as handle:
            assert list(sagent.normalize(handle)) == records


class TestCiphertext:
    """Sealed reasoning is spliced at materialization, or the replay refuses."""

    def test_ciphertext_is_spliced_back_into_its_record(self, tmp_path: Path) -> None:
        """The bytes live in another table; the journal needs them inline."""
        sealed = "c2VhbGVkLXJlYXNvbmluZw=="

        path = materialize(
            [
                UserMessage(content="a"),
                Thinking(encrypted=""),
                UserMessage(content="b"),
            ],
            sealed=[None, sealed, None],
            folder=tmp_path,
        )

        assert _read(path) == [
            UserMessage(content="a"),
            Thinking(encrypted=sealed),
            UserMessage(content="b"),
        ]

    def test_a_dropped_ciphertext_refuses_and_writes_nothing(
        self,
        tmp_path: Path,
    ) -> None:
        """Retention can drop the bytes; a hollow transcript fails at the provider.

        Raising names the cause. Writing an empty ``encrypted`` would fail inside
        the provider with nothing pointing back at retention, and a half-written
        journal would be a session this host appears to hold.
        """
        with pytest.raises(
            CiphertextDroppedError,
            match=re.escape(
                "record 1 is sealed reasoning whose ciphertext is no longer stored; "
                "the provider rejects a transcript with an empty 'encrypted' field, "
                "so this session cannot be resumed",
            ),
        ):
            _ = materialize(
                [UserMessage(content="a"), Thinking(encrypted="")],
                sealed=[None],
                folder=tmp_path,
            )

        assert list(tmp_path.iterdir()) == []

    def test_reasoning_never_sealed_or_readable_needs_no_ciphertext(
        self,
        tmp_path: Path,
    ) -> None:
        records = [
            Thinking(summary="summary", encrypted=None),
            Thinking(content="visible reasoning", encrypted=""),
        ]

        path = materialize(records, sealed=[None, None], folder=tmp_path)

        assert _read(path) == records


class TestAnUnparsedRecord:
    """A record read torn replays verbatim, but only as one line."""

    @pytest.mark.parametrize(
        "text",
        ['{"py/object": "UserMessage", "cont', '{"py/object": "Unknown"} \n'],
        ids=["torn", "one-value-and-its-newline"],
    )
    def test_a_line_replays_byte_for_byte(self, tmp_path: Path, text: str) -> None:
        path = materialize(
            [UserMessage(content="hi"), IncompleteRecord(text=text)],
            sealed=[],
            folder=tmp_path,
        )

        assert path.read_bytes() == (
            record_line(UserMessage(content="hi")) + text.encode()
        )

    @pytest.mark.parametrize(
        "text",
        [
            '{"type":"cost-state"}{"type":"atis-latch"}',
            '{"type":"cost-state"}\n{"type":"atis-latch"}',
        ],
        ids=["several-values", "several-lines"],
    )
    def test_a_record_that_cannot_be_one_line_is_dropped(
        self,
        tmp_path: Path,
        text: str,
    ) -> None:
        """A real capture stored three objects, no separator, under one record."""
        path = materialize(
            [UserMessage(content="hi"), IncompleteRecord(text=text)],
            sealed=[],
            folder=tmp_path,
        )

        assert path.read_bytes() == record_line(UserMessage(content="hi"))


def _read(path: Path) -> list[SessionRecord]:
    """Return the records a journal holds, as sagent reads them."""
    with path.open(encoding="utf-8") as handle:
        return list(sagent.normalize(handle))


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
