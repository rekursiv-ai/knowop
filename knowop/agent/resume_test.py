"""The journal an agent process continues a stored session in, on this host."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final
from uuid import UUID, uuid4

import re

from trackinizer.client.conversation import read_part
from trackinizer.lib.userdirs import state_dir
from trackinizer.sagent.scripts.convert_session import sagent
from trackinizer.sagent.session.jsonl import record_line
from trackinizer.sagent.session.types import (
    AssistantMessage,
    BashResult,
    SessionRecord,
    Thinking,
    ToolResult,
    TurnContext,
    UncategorizedRecord,
    UserMessage,
)
from trackinizer.types.session_records import SessionRecordRow
from trackinizer.wire.wire_session_ir import PartBody, RecordBody
from treekle import immutable

import pytest

from knowop.agent import resume
from knowop.agent.errors import (
    LossyConversionError,
    NotResumableError,
)
from knowop.agent.resume import (
    JournalChoice,
    session_journal,
)
from knowop.agent.truncate import NOTE


if TYPE_CHECKING:
    from pathlib import Path

    from trackinizer.client.conversation import PartReader


_SESSION_ID: Final = UUID(int=42)


@dataclass(slots=True, kw_only=True)
class _Stored:
    """A session's parts as the server stores them, each under the name it was sent."""

    parts: dict[int, tuple[str, str, list[SessionRecord]]] = field(
        default_factory=dict[int, tuple[str, str, list[SessionRecord]]],
    )
    asked: list[tuple[UUID, int, bool]] = field(
        default_factory=list[tuple[UUID, int, bool]],
    )
    """Each read of a part's records: the session, the part, and ``plaintext_only``."""

    def listed(self) -> list[PartBody]:
        return [
            PartBody(part=number, name=name, format=fmt, records=len(records))
            for number, (name, fmt, records) in self.parts.items()
        ]

    def read_session_parts(self, session_id: UUID) -> list[PartBody]:
        del session_id
        return self.listed()

    def read_session_records(
        self,
        session_id: UUID,
        *,
        part: int = 0,
        plaintext_only: bool = False,
    ) -> list[RecordBody]:
        self.asked.append((session_id, part, plaintext_only))
        return [
            RecordBody.of(
                SessionRecordRow.of(
                    session_id=session_id,
                    part=part,
                    idx=idx,
                    record=record,
                ),
            )
            for idx, record in enumerate(self.parts[part][2])
        ]


def _journal_here(folder: str, *records: SessionRecord) -> Path:
    """Write ``records`` as a journal this host holds in the session's ``folder``."""
    path = (
        state_dir()
        / "rekursiv-ai"
        / "knowop"
        / "journals"
        / str(_SESSION_ID)
        / folder
        / "session.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(b"".join(record_line(record) for record in records))
    return path


def _journal_lines(path: Path) -> list[SessionRecord]:
    with path.open(encoding="utf-8") as handle:
        return [r for r in sagent.normalize(handle) if not isinstance(r, TurnContext)]


_SAID: Final = (UserMessage(content="hi"), AssistantMessage(content="hello"))


class TestTheJournalAProcessWrites:
    """``session_journal``: the journal an agent process writes, and where it uploads."""

    def test_a_session_with_no_parts_gets_a_new_folder_and_an_empty_journal(
        self,
    ) -> None:
        choice = session_journal(_SESSION_ID, [], client=_Stored(), lossy=False)

        folder = choice.path.parent
        assert choice.path.name == "session.jsonl"
        assert (
            folder.parent
            == state_dir() / "rekursiv-ai" / "knowop" / "journals" / str(_SESSION_ID)
        )
        assert UUID(hex=folder.name).hex == folder.name
        assert choice == JournalChoice(
            path=choice.path,
            part=f"{folder.name}.jsonl",
            start=0,
            restored=False,
        )

    @pytest.mark.parametrize(
        "unstored",
        [(), (UserMessage(content="not yet stored"),)],
        ids=["all-stored", "one-not-yet-stored"],
    )
    def test_this_hosts_journal_of_the_last_part_is_continued_where_it_was_stored(
        self,
        unstored: tuple[SessionRecord, ...],
    ) -> None:
        folder = uuid4().hex
        path = _journal_here(folder, *_SAID, *unstored)
        stored = _Stored(parts={0: (f"{folder}.jsonl", "sagent", list(_SAID))})

        choice = session_journal(
            _SESSION_ID,
            stored.listed(),
            client=stored,
            lossy=False,
        )

        assert choice == JournalChoice(
            path=path,
            part=f"{folder}.jsonl",
            start=2,
            restored=False,
        )

    @pytest.mark.parametrize(
        "case",
        ["no-such-folder", "a-newer-part-elsewhere", "a-shorter-journal", "a-bad-name"],
    )
    def test_anything_else_is_restored_from_the_stored_parts(self, case: str) -> None:
        here = uuid4().hex
        parts: dict[int, tuple[str, str, list[SessionRecord]]] = {
            0: (f"{here}.jsonl", "sagent", [_SAID[0]]),
        }
        if case == "a-newer-part-elsewhere":
            _ = _journal_here(here, _SAID[0])
            parts[1] = (f"{uuid4().hex}.jsonl", "sagent", [_SAID[1]])
        elif case == "a-shorter-journal":
            _ = _journal_here(here)
        elif case == "a-bad-name":
            # Whole, where the name's ``..`` would reach: outside the session's folder.
            journals = state_dir() / "rekursiv-ai" / "knowop" / "journals"
            (journals / str(_SESSION_ID)).mkdir(parents=True)
            beside = journals / here
            beside.mkdir()
            (beside / "session.jsonl").write_bytes(record_line(_SAID[0]))
            parts = {0: (f"../{here}.jsonl", "sagent", [_SAID[0]])}
        stored = _Stored(parts=parts)

        choice = session_journal(
            _SESSION_ID,
            stored.listed(),
            client=stored,
            lossy=False,
        )

        assert choice.restored
        assert (
            choice.path.parent.parent
            == state_dir() / "rekursiv-ai" / "knowop" / "journals" / str(_SESSION_ID)
        )
        assert choice.path.parent.name != here
        assert choice.part == f"{choice.path.parent.name}.jsonl"
        assert choice.start == len(choice.path.read_bytes().splitlines())
        wanted = [r for _, _, records in parts.values() for r in records]
        assert [_text(r) for r in _journal_lines(choice.path)] == [
            _text(r) for r in wanted
        ]

    def test_a_restore_keeps_each_seal_and_drops_what_restarts_a_child(self) -> None:
        sealed = Thinking(content="why", encrypted="c2VhbGVk")
        child = UncategorizedRecord(
            kind="persistent_agent",
            payload=immutable({"label": "Agent.1", "standing": "child:Agent.1"}),
        )
        watcher = UncategorizedRecord(
            kind="watcher",
            payload=immutable({"label": "watcher", "standing": "watch:watcher"}),
        )
        stored = _Stored(
            parts={0: (f"{uuid4().hex}.jsonl", "sagent", [sealed, child, watcher])},
        )

        choice = session_journal(
            _SESSION_ID,
            stored.listed(),
            client=stored,
            lossy=False,
        )

        assert _journal_lines(choice.path) == [sealed]
        # With the ciphertext, which only a replay reads.
        assert stored.asked == [(_SESSION_ID, 0, False)]

    def test_a_part_whose_seals_do_not_line_up_is_refused(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A seal paired with the wrong record would replay reasoning out of place."""
        stored = _Stored(parts={0: (f"{uuid4().hex}.jsonl", "sagent", list(_SAID))})
        monkeypatch.setattr(resume, "read_part", _one_seal_short)

        with pytest.raises(ValueError, match="zip"):
            _ = session_journal(
                _SESSION_ID,
                stored.listed(),
                client=stored,
                lossy=False,
            )

    def test_a_part_of_another_format_is_refused(self) -> None:
        stored = _Stored(
            parts={
                0: (f"{uuid4().hex}.jsonl", "sagent", [_SAID[0]]),
                1: ("turn.jsonl", "assistant", [_SAID[1]]),
                2: ("c.jsonl", "claude", [_SAID[0]]),
            },
        )

        with pytest.raises(
            NotResumableError,
            match=_whole(
                f"session {_SESSION_ID} holds a part of assistant, claude, which no "
                "sagent journal continues; it stays readable",
            ),
        ):
            _ = session_journal(
                _SESSION_ID,
                stored.listed(),
                client=stored,
                lossy=False,
            )

    def test_a_cut_record_restores_only_when_a_loss_is_accepted(self) -> None:
        cut = ToolResult(
            call_id="c",
            state="done",
            result=BashResult(command=("ls",)),
            extra=immutable({NOTE: {"omitted_bytes": 1}}),
        )
        stored = _Stored(parts={0: (f"{uuid4().hex}.jsonl", "sagent", [cut])})

        with pytest.raises(
            LossyConversionError,
            match=_whole(
                f"1 record(s) of session {_SESSION_ID} were truncated at capture to "
                "fit an upload",
            ),
        ):
            _ = session_journal(
                _SESSION_ID,
                stored.listed(),
                client=stored,
                lossy=False,
            )
        choice = session_journal(
            _SESSION_ID,
            stored.listed(),
            client=stored,
            lossy=True,
        )

        [restored] = _journal_lines(choice.path)
        assert restored == ToolResult(
            call_id="c",
            state="done",
            result=BashResult(command=("ls",)),
        )


def _text(record: SessionRecord) -> str:
    assert isinstance(record, UserMessage | AssistantMessage)
    return record.content or ""


def _whole(message: str) -> str:
    """Return the pattern ``pytest.raises`` matches ``message`` by, and nothing more."""
    return f"^{re.escape(message)}$"


def _one_seal_short(
    client: PartReader,
    session_id: UUID,
    part: int,
    *,
    plaintext_only: bool,
) -> tuple[list[SessionRecord], list[str | None]]:
    """Read a part as ``read_part`` does, with its last seal missing."""
    records, sealed = read_part(client, session_id, part, plaintext_only=plaintext_only)
    return records, sealed[:-1]


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
