"""The journal an agent process continues a stored session in, on this host."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final
from uuid import UUID, uuid4

import re

from trackinizer.client.conversation import read_part
from trackinizer.lib.userdirs import state_dir
from trackinizer.sagent.journal.jsonl import record_line
from trackinizer.sagent.journal.types import (
    AssistantMessage,
    BashResult,
    ContextState,
    SessionContext,
    SessionRecord,
    SystemMessage,
    Thinking,
    TokenUsage,
    ToolCall,
    ToolMessage,
    UncategorizedRecord,
    UserMessage,
)
from trackinizer.sagent.scripts.convert_session import sagent
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
    from collections.abc import Mapping
    from pathlib import Path

    from trackinizer.client.conversation import PartReader
    from treekle import PlainTree


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
        return [
            r for r in sagent.normalize(handle) if not isinstance(r, SessionContext)
        ]


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

    def test_every_native_part_restores_as_stored_whichever_cli_wrote_it(
        self,
    ) -> None:
        """The records are provider-neutral: sagent reads its turns from them as stored."""
        claude: list[SessionRecord] = [
            UserMessage(content="read it"),
            Thinking(content="read", encrypted="c2ln", extra=_asked("a")),
            TokenUsage(info=immutable({"output_tokens": 1})),
            _call("1", "a"),
            UncategorizedRecord(kind="queue-operation", payload=immutable({})),
            _result("1"),
            ContextState(kind="queued_command", content="a report"),
        ]
        codex: list[SessionRecord] = [AssistantMessage(content="from codex")]
        stored = _Stored(
            parts={
                0: (f"{uuid4().hex}.jsonl", "sagent", list(_SAID)),
                1: ("c.jsonl", "claude", claude),
                2: ("rollout-1.jsonl", "codex", codex),
            },
        )

        choice = session_journal(
            _SESSION_ID,
            stored.listed(),
            client=stored,
            lossy=False,
        )

        assert _journal_lines(choice.path) == [*_SAID, *claude, *codex]

    def test_a_part_with_no_format_is_left_out(self) -> None:
        notes: list[SessionRecord] = [SystemMessage(content="/effort high")]
        stored = _Stored(
            parts={
                0: ("prompts-1", "", notes),
                1: (f"{uuid4().hex}.jsonl", "sagent", list(_SAID)),
            },
        )

        choice = session_journal(
            _SESSION_ID,
            stored.listed(),
            client=stored,
            lossy=False,
        )

        assert _journal_lines(choice.path) == list(_SAID)
        assert [part for _, part, _ in stored.asked] == [1]

    def test_a_session_of_only_parts_with_no_format_is_refused(self) -> None:
        notes: list[SessionRecord] = [SystemMessage(content="/effort high")]
        stored = _Stored(parts={0: ("prompts-1", "", notes)})

        with pytest.raises(
            NotResumableError,
            match=_whole(
                f"session {_SESSION_ID} holds no part with a native format, so there "
                "is no conversation to continue; it stays readable",
            ),
        ):
            _ = session_journal(
                _SESSION_ID,
                stored.listed(),
                client=stored,
                lossy=False,
            )

    def test_a_cut_record_restores_only_when_a_loss_is_accepted(self) -> None:
        cut = ToolMessage(
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
        assert restored == ToolMessage(
            call_id="c",
            state="done",
            result=BashResult(command=("ls",)),
        )


_PLAN: Final = "work@anthropic-sub/opus-5.5+fullcontext+highthink"
"""A model line of a subscription, which a host that holds keys alone does not offer."""

_SOL: Final = "openai/sol-6.2+fullcontext+highthink"


class TestAHostThatLacksTheSessionsSignIn:
    """A restore on a host that offers only some providers, as a remote agent's."""

    def test_any_host_by_default_restores_the_model_as_it_stood(self) -> None:
        choice, stood = _restored_on(_PLAN, cache=(_SOL,))

        assert stood == {"model": _PLAN, "cache": [_SOL], "standing": "model"}
        assert choice.unoffered == ""

    def test_a_model_the_host_lacks_falls_back_on_sol_unless_another_is_named(
        self,
    ) -> None:
        choice, stood = _restored_on(_PLAN, offered=("anthropic", "openai"))

        assert stood == {"model": "openai/sol-6.2", "cache": [], "standing": "model"}
        assert choice.unoffered == "anthropic-sub"

    def test_a_cached_model_the_host_lacks_is_forgotten_at_no_charge(self) -> None:
        """Its settings serve no switch here; left in, sagent refuses the whole resume."""
        choice, stood = _restored_on(
            _SOL,
            cache=(_PLAN, "anthropic/haiku-4.5"),
            offered=("anthropic", "openai"),
        )

        assert stood == {
            "model": _SOL,
            "cache": ["anthropic/haiku-4.5"],
            "standing": "model",
        }
        assert choice.unoffered == ""

    def test_a_model_the_host_lacks_stays_on_the_key_its_provider_extends(
        self,
    ) -> None:
        """``AnthropicSub`` is an ``Anthropic``: the same model, settings and account."""
        choice, stood = _restored_on(
            _PLAN,
            cache=(_SOL,),
            offered=("anthropic", "openai"),
            keyed=(("anthropic-sub", "anthropic"),),
            fallback="openai/sol-6.2",
        )

        assert stood == {
            "model": "work@anthropic/opus-5.5+fullcontext+highthink",
            "cache": [_SOL],
            "standing": "model",
        }
        assert choice.unoffered == "anthropic-sub"

    @pytest.mark.parametrize(
        ("cache", "model", "kept"),
        [
            ((_SOL,), _SOL, []),
            (("openai/sol-6.2 budget=auto",), "openai/sol-6.2 budget=auto", []),
            ((), "openai/sol-6.2", []),
            (
                ("openai/astra-6.2+highthink",),
                "openai/sol-6.2",
                ["openai/astra-6.2+highthink"],
            ),
        ],
        ids=["as its cache left it", "with its choices", "as named", "another kept"],
    )
    def test_with_no_key_of_the_vendor_s_it_falls_back_on_the_model_named(
        self,
        cache: tuple[str, ...],
        model: str,
        kept: list[str],
    ) -> None:
        choice, stood = _restored_on(
            _PLAN,
            cache=cache,
            offered=("openai",),
            fallback="openai/sol-6.2",
        )

        assert stood == {"model": model, "cache": kept, "standing": "model"}
        assert choice.unoffered == "anthropic-sub"

    def test_a_session_that_stood_no_model_restores_as_it_stood(self) -> None:
        stored = _Stored(parts={0: (f"{uuid4().hex}.jsonl", "sagent", list(_SAID))})

        choice = session_journal(
            _SESSION_ID,
            stored.listed(),
            client=stored,
            lossy=False,
            offered=frozenset({"openai"}),
        )

        assert (_journal_lines(choice.path), choice.unoffered) == (list(_SAID), "")

    def test_only_the_model_standing_last_is_asked_about(self) -> None:
        """Sagent resumes on the last; an earlier one it no longer uses is no refusal."""
        stored = _Stored(
            parts={
                0: (f"{uuid4().hex}.jsonl", "sagent", [_model(_PLAN), _model(_SOL)]),
            },
        )

        choice = session_journal(
            _SESSION_ID,
            stored.listed(),
            client=stored,
            lossy=False,
            offered=frozenset({"openai"}),
        )

        assert [_stood(r) for r in _journal_lines(choice.path)] == [
            {"model": _PLAN, "cache": [], "standing": "model"},
            {"model": _SOL, "cache": [], "standing": "model"},
        ]


def _model(line: str, *, cache: tuple[str, ...] = ()) -> UncategorizedRecord:
    """Return the record sagent stands its model on."""
    return UncategorizedRecord(
        kind="model",
        payload=immutable({"model": line, "cache": list(cache), "standing": "model"}),
    )


def _stood(record: SessionRecord) -> dict[str, PlainTree]:
    """Return a ``model`` record's payload, its cache a list as it was written."""
    assert isinstance(record, UncategorizedRecord)
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in record.payload.items()
    }


def _restored_on(
    line: str,
    *,
    cache: tuple[str, ...] = (),
    offered: tuple[str, ...] = (),
    keyed: tuple[tuple[str, str], ...] = (),
    fallback: str = "",
) -> tuple[JournalChoice, dict[str, PlainTree]]:
    """Restore a session standing on ``line`` where ``offered`` are; return its model."""
    stored = _Stored(
        parts={
            0: (
                f"{uuid4().hex}.jsonl",
                "sagent",
                [_SAID[0], _model(line, cache=cache)],
            ),
        },
    )
    # Named only when given, so a refusal is the default's.
    choice = session_journal(
        _SESSION_ID,
        stored.listed(),
        client=stored,
        lossy=False,
        offered=frozenset(offered),
        keyed=keyed,
        **({"fallback": fallback} if fallback else {}),
    )
    said, stood = _journal_lines(choice.path)
    assert said == _SAID[0]
    assert choice.start == 2
    return choice, _stood(stood)


def _text(record: SessionRecord) -> str:
    assert isinstance(record, UserMessage | AssistantMessage)
    return record.content or ""


def _asked(request: str) -> Mapping[str, PlainTree]:
    """Return the ``extra`` Claude's blocks of one reply share."""
    return immutable({"requestId": f"req_{request}"})


def _call(call_id: str, request: str) -> ToolCall:
    return ToolCall(
        call_id=call_id,
        name="Read",
        arguments=immutable({"file_path": call_id}),
        extra=_asked(request),
    )


def _result(call_id: str) -> ToolMessage:
    return ToolMessage(
        call_id=call_id,
        state="done",
        result=BashResult(command=("ls",)),
    )


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
