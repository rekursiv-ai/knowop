"""``knowop upload``'s plan and upload, against hand-written transcripts and a fake server."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, Final, override
from uuid import UUID, uuid4

import json
import os

from trackinizer.client.errors import ClientError
from trackinizer.lib import zstd_compat
from trackinizer.runs.redact import Redactor
from trackinizer.sagent.scripts.convert_session import claude
from trackinizer.sagent.scripts.convert_session.convert import die_with_parent
from trackinizer.sagent.scripts.convert_session.tail import Tail
from trackinizer.sagent.session.jsonl import read_record, record_line
from trackinizer.sagent.session.types import (
    AssistantMessage,
    UncategorizedRecord,
    UserMessage,
)
from trackinizer.wire.wire_session_ir import ManifestBody, PartBody, RecordBody
from trackinizer.wire.wire_sessions import (
    SessionEnd,
    SessionStart,
    SessionStartResponse,
)

import pytest

from knowop import upload
from knowop.upload import Session, Uploaded, plan


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path


_T0: Final = datetime(2026, 11, 1, 10, 0, tzinfo=UTC)
_MASKED: Final = "hunter2-hunter2"


@dataclass(slots=True, kw_only=True)
class _Held:
    """One AgentSession as the fake server holds it."""

    id: UUID
    start: SessionStart
    audience: str
    parts: dict[str, ManifestBody] = field(default_factory=dict[str, ManifestBody])
    records: dict[str, dict[int, RecordBody]] = field(
        default_factory=dict[str, dict[int, RecordBody]],
    )
    ends: list[SessionEnd] = field(default_factory=list[SessionEnd])


@dataclass(slots=True, kw_only=True)
class _Server:
    """Trackinizer as an upload meets it: a start re-opens the session it holds."""

    held: dict[str, _Held] = field(default_factory=dict[str, _Held])
    """By CLI id."""

    edges: list[tuple[UUID, UUID, str, str]] = field(
        default_factory=list[tuple[UUID, UUID, str, str]],
    )
    edge_actors: list[str] = field(default_factory=list[str])
    """Who each edge was added as."""

    calls: list[str] = field(default_factory=list[str])
    refusing: frozenset[int] = frozenset()
    """The positions of records the server cannot store."""

    refused_ids: frozenset[str] = frozenset()
    """The CLI ids of sessions whose start the server refuses."""

    breaking_ids: frozenset[str] = frozenset()
    """The CLI ids of sessions whose start fails with an error that says nothing."""

    predates_hiding: bool = False
    """Whether it opens every session to everyone, and names no audience in its answer."""

    opened: int = 0
    """How many clients were made; each is closed after its session."""

    closed: int = 0

    def connect(self) -> _Server:
        self.opened += 1
        return self

    def close(self) -> None:
        self.closed += 1

    def session_start(self, body: SessionStart) -> SessionStartResponse:
        self.calls.append("start")
        assert body.cli_session_id is not None
        if body.cli_session_id in self.refused_ids:
            raise ClientError("refused", status_code=403)
        if body.cli_session_id in self.breaking_ids:
            raise ValueError
        held = self.held.get(body.cli_session_id)
        if held is None:
            audience = (
                "private" if body.private and not self.predates_hiding else "open"
            )
            held = _Held(id=uuid4(), start=body, audience=audience)
            self.held[body.cli_session_id] = held
        elif body.private and not body.keep_audience:
            held.audience = "private"
        return SessionStartResponse(
            id=held.id,
            seq=0,
            audience=None if self.predates_hiding else held.audience,
        )

    def read_session_parts(self, session_id: UUID) -> list[PartBody]:
        self.calls.append("parts")
        held = self._by_id(session_id)
        return [
            PartBody(
                part=index,
                name=name,
                format=manifest.format,
                records=manifest.records,
                metadata=manifest.metadata,
                ir_id=manifest.ir_id,
            )
            for index, (name, manifest) in enumerate(held.parts.items())
        ]

    def append_records(
        self,
        session_id: UUID,
        *,
        name: str,
        manifest: ManifestBody,
        records: Sequence[RecordBody],
    ) -> object:
        self.calls.append("append")
        if any(
            r.idx in self.refusing and r.kind != "UncategorizedRecord" for r in records
        ):
            raise ClientError("cannot store a NUL byte", status_code=422)
        held = self._by_id(session_id)
        held.parts[name] = manifest
        stored = held.records.setdefault(name, {})
        for record in records:
            stored.setdefault(record.idx, record)
        return None

    def session_end(self, session_id: UUID, body: SessionEnd | None = None) -> object:
        self.calls.append("end")
        assert body is not None
        self._by_id(session_id).ends.append(body)
        return None

    def add_edge(
        self,
        from_id: UUID,
        to_id: UUID,
        edge_kind: str,
        *,
        actor: str,
        note: str = "",
    ) -> object:
        self.edge_actors.append(actor)
        self.calls.append("edge")
        self.edges.append((from_id, to_id, edge_kind, note))
        return None

    def session(self, cli_id: str) -> _Held:
        return self.held[cli_id]

    def _by_id(self, session_id: UUID) -> _Held:
        return next(held for held in self.held.values() if held.id == session_id)


def _sent(
    server: _Server,
    sessions: Sequence[Session],
    *,
    private: bool = True,
    redactor: Redactor | None = None,
) -> list[Uploaded]:
    return upload.upload(
        sessions,
        connect=server.connect,
        private=private,
        jobs=1,
        redactor=redactor,
    )


# Transcripts, a few lines each, written as each CLI writes them.


def _lines(path: Path, *records: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def _claude(path: Path, *, said: str = "hello", agent: str = "") -> Path:
    """Write a claude transcript: a prompt at ``_T0`` and a reply five minutes on."""
    who = {"agentId": agent} if agent else {}
    return _lines(
        path,
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": "s",
            "timestamp": "2026-11-01T10:00:00Z",
            "message": {"role": "user", "content": said},
            **who,
        },
        {
            "type": "assistant",
            "uuid": "u2",
            "parentUuid": "u1",
            "sessionId": "s",
            "timestamp": "2026-11-01T10:05:00Z",
            "message": {
                "role": "assistant",
                "model": "claude-x",
                "content": [{"type": "text", "text": "hi"}],
            },
            **who,
        },
    )


# ``workflow`` names the workflow that ran it, whose folder then holds both.
def _claude_agent(
    folder: Path,
    agent: str,
    meta: dict[str, str],
    *,
    workflow: str = "",
) -> Path:
    """Write sub-agent ``agent``'s transcript and meta in a session's ``subagents``."""
    within = folder / "subagents"
    if workflow:
        within = within / "workflows" / workflow
    path = _claude(within / f"agent-{agent}.jsonl", agent=agent)
    (within / f"agent-{agent}.meta.json").write_text(json.dumps(meta))
    return path


def _codex(folder: Path, thread: str, *, spawner: str = "") -> Path:
    """Write the rollout of codex thread ``thread``; ``spawner`` names its spawner's."""
    source = (
        {
            "subagent": {
                "thread_spawn": {
                    "parent_thread_id": spawner,
                    "agent_nickname": "Ada",
                    "agent_path": "/a",
                },
            },
        }
        if spawner
        else "cli"
    )
    return _lines(
        folder / f"rollout-2026-11-01T10-00-00-{thread}.jsonl",
        {
            "timestamp": "2026-11-01T10:00:00Z",
            "type": "session_meta",
            "payload": {
                "id": thread,
                "timestamp": "2026-11-01T10:00:00Z",
                "source": source,
            },
        },
        {
            "timestamp": "2026-11-01T10:01:00Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            },
        },
    )


def _zst(path: Path) -> Path:
    """Compress ``path`` in place, as codex does an older rollout: ``.jsonl.zst``."""
    packed = path.with_name(f"{path.name}.zst")
    _ = packed.write_bytes(zstd_compat.compress(path.read_bytes()))
    path.unlink()
    return packed


def _sagent(
    folder: Path,
    *,
    said: str = "hi",
    stood: dict[str, str] | None = None,
) -> Path:
    """Write a sagent journal in ``folder``; ``stood`` stands a child per label."""
    path = folder / "session.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    standing = [
        UncategorizedRecord(
            kind="persistent_agent",
            payload={"label": label, "journal": f"/elsewhere/{child}/session.jsonl"},
        )
        for label, child in (stood or {}).items()
    ]
    path.write_bytes(
        b"".join(
            record_line(record)
            for record in (
                UserMessage(content=said, timestamp="2026-11-01T10:00:00+00:00"),
                *standing,
                AssistantMessage(content="yo", timestamp="2026-11-01T10:02:00+00:00"),
            )
        ),
    )
    return path


_THREAD: Final = "01a1b7c9-2ac4-7a22-ae7d-a5a26dc8991f"
_SPAWNED: Final = "01a1b7c9-1227-7b63-b752-ac938abc3021"
_NESTED: Final = "01a1b7c9-4227-7bb0-8c66-565efc9662f6"
_SAGENT_CHILD: Final = "0f04a582-9c82-4509-8389-db6327d1a6e1"


# Each format alone.


def test_a_claude_session_goes_up_whole_under_its_id_where_it_happened(
    tmp_path: Path,
) -> None:
    path = _claude(tmp_path / "-w" / "s1.jsonl")
    server = _Server()

    [result] = _sent(server, plan([path]))

    held = server.session("s1")
    assert (held.start.cli, held.start.actor, held.start.started) == (
        "claude",
        "claude/s1",
        _T0,
    )
    assert [end.ended for end in held.ends] == [
        datetime(2026, 11, 1, 10, 5, tzinfo=UTC),
    ]
    assert sorted(held.records["s1.jsonl"]) == list(range(6))
    manifest = held.parts["s1.jsonl"]
    assert (manifest.format, manifest.records) == ("claude", 6)
    reader = Tail(claude.normalize)
    fed = [r for line in path.read_text().splitlines(True) for r in reader.feed(line)]
    assert len(fed) + len(reader.close()) == 6
    assert manifest.metadata == reader.encoding
    assert result == Uploaded(
        session=Session(format="claude", id="s1", path=path),
        agent_session=held.id,
        sent=6,
        audience="private",
    )
    assert server.calls == ["start", "parts", "append", "end"]
    assert server.closed == server.opened == 1


def test_a_codex_rollout_goes_up_under_its_thread_id(tmp_path: Path) -> None:
    path = _codex(tmp_path / "2026" / "11" / "01", _THREAD)
    server = _Server()

    [result] = _sent(server, plan([path]))

    held = server.session(_THREAD)
    assert (held.start.cli, result.sent) == ("codex", 3)
    assert held.parts[path.name].format == "codex"


def test_a_sagent_journal_is_a_part_named_for_its_folder(tmp_path: Path) -> None:
    path = _sagent(tmp_path / "a4458be87354")
    server = _Server()

    [result] = _sent(server, plan([path]))

    held = server.session("a4458be87354")
    assert (held.start.cli, result.sent) == ("sagent", 2)
    assert list(held.parts) == ["a4458be87354.jsonl"]
    assert held.parts["a4458be87354.jsonl"].format == "sagent"


def test_a_session_whose_records_state_no_time_leaves_its_times_to_the_server(
    tmp_path: Path,
) -> None:
    path = tmp_path / "a4458be87354" / "session.jsonl"
    path.parent.mkdir()
    path.write_bytes(record_line(UserMessage(content="hi")))
    server = _Server()

    [result] = _sent(server, plan([path]))

    held = server.session("a4458be87354")
    assert (held.start.started, [end.ended for end in held.ends]) == (None, [None])
    assert result.sent == 1


def test_each_record_is_masked_before_it_goes(tmp_path: Path) -> None:
    path = _sagent(tmp_path / "a4458be87354", said=f"the key is {_MASKED}")
    server = _Server()

    _ = _sent(server, plan([path]), redactor=Redactor({"KEY": _MASKED}))

    stored = server.session("a4458be87354").records["a4458be87354.jsonl"][0]
    assert _MASKED not in stored.model_dump_json()
    assert "[redacted:KEY]" in stored.text


def test_bytes_that_are_no_utf8_go_up_replaced(tmp_path: Path) -> None:
    path = tmp_path / "a4458be87354" / "session.jsonl"
    path.parent.mkdir()
    path.write_bytes(record_line(UserMessage(content="calf?")).replace(b"?", b"\xff"))
    server = _Server()

    [result] = _sent(server, plan([path], read_as="sagent"))

    stored = server.session("a4458be87354").records["a4458be87354.jsonl"][0]
    assert (result.error, stored.text) == ("", f"calf{chr(0xFFFD)}")


# A journal can be hundreds of megabytes, so only a line that names a standing record's
# kind is parsed; the standing records come anywhere in it.
def test_a_journals_standing_records_are_found_among_lines_never_parsed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "session.jsonl"
    records = (
        UserMessage(content="hi"),
        UncategorizedRecord(
            kind="persistent_agent",
            payload={"label": "scout", "journal": "/elsewhere/c1/session.jsonl"},
        ),
        UserMessage(content="watcher"),  # Names a kind, and is none.
        UncategorizedRecord(
            kind="watcher",
            payload={"journal": "/elsewhere/c2/session.jsonl"},
        ),
        UncategorizedRecord(kind="persistent_agent", payload={"label": "lost"}),
        AssistantMessage(content="yo"),
    )
    path.write_bytes(b"".join(record_line(record) for record in records))
    parsed: list[bytes] = []

    def read(line: bytes) -> object:
        parsed.append(line)
        return read_record(line)

    monkeypatch.setattr(upload, "read_record", read)

    assert upload._standing_labels(path) == {"c1": "scout", "c2": ""}
    assert parsed == [record_line(record) for record in records[1:5]]


# Sub-agents.


def test_claude_sub_agents_go_after_their_spawner_each_linked_to_it(
    tmp_path: Path,
) -> None:
    root = _claude(tmp_path / "-w" / "s1.jsonl")
    agent = _claude_agent(
        tmp_path / "-w" / "s1",
        "a1",
        {"agentType": "general-purpose", "description": "Dig", "toolUseId": "t1"},
    )
    nested = _claude_agent(tmp_path / "-w" / "s1", "a2", {"parentAgentId": "a1"})
    server = _Server()

    sessions = plan([tmp_path])
    results = _sent(server, sessions)

    assert [(s.id, s.spawned_by, s.linked, s.subagents) for s in sessions] == [
        ("s1", "", False, 1),
        ("s1/agent-a1", "s1", True, 1),
        ("s1/agent-a2", "s1/agent-a1", True, 0),
    ]
    assert [s.path for s in sessions] == [root, agent, nested]
    ids = {cli_id: held.id for cli_id, held in server.held.items()}
    assert server.edges == [
        (ids["s1/agent-a1"], ids["s1"], "produced_by", "general-purpose: Dig (t1)"),
        (ids["s1/agent-a2"], ids["s1/agent-a1"], "produced_by", ""),
    ]
    assert server.edge_actors == ["claude", "claude"]
    assert [r.error for r in results] == ["", "", ""]


def test_a_sub_agents_own_file_stands_for_the_session_at_its_root(
    tmp_path: Path,
) -> None:
    _ = _claude(tmp_path / "Downloads" / "-w" / "s1.jsonl")
    _ = _claude_agent(tmp_path / "Downloads" / "-w" / "s1", "a1", {})
    nested = _claude_agent(
        tmp_path / "Downloads" / "-w" / "s1",
        "a2",
        {"parentAgentId": "a1"},
    )

    sessions = plan([nested])

    assert [s.id for s in sessions] == ["s1", "s1/agent-a1", "s1/agent-a2"]


def test_a_sub_agent_whose_spawner_is_not_on_disk_goes_up_unlinked(
    tmp_path: Path,
) -> None:
    agent = _claude_agent(tmp_path / "-w" / "s1", "a1", {})
    server = _Server()

    sessions = plan([agent])
    [result] = _sent(server, sessions)

    assert sessions == [
        Session(format="claude", id="s1/agent-a1", path=agent, spawned_by="s1"),
    ]
    assert result.error == ""
    assert server.edges == []


# Measured on claude 2.1.312: a workflow's agents land in
# ``<session>/subagents/workflows/<wf>/``, beside the workflow's ``journal.jsonl``, and
# an agent a message continued has a second transcript, under the same id, among the
# session's own sub-agents.
def test_a_workflows_agents_go_up_linked_to_the_session_and_its_journal_does_not(
    tmp_path: Path,
) -> None:
    root = _claude(tmp_path / "-w" / "s1.jsonl")
    continued = _claude_agent(tmp_path / "-w" / "s1", "a1", {})
    ran = _claude_agent(
        tmp_path / "-w" / "s1",
        "a1",
        {"agentType": "workflow-subagent", "description": "audit"},
        workflow="wf_1",
    )
    journal = _lines(
        ran.with_name("journal.jsonl"),
        {"type": "launched"},
        {"type": "started", "agentId": "a1", "label": "audit"},
    )
    server = _Server()

    sessions = plan([tmp_path])
    results = _sent(server, sessions)

    assert [(s.id, s.spawned_by, s.linked, s.path) for s in sessions] == [
        ("s1", "", False, root),
        ("s1/agent-a1", "s1", True, continued),
        ("s1/wf_1/agent-a1", "s1", True, ran),
    ]
    assert sessions[0].subagents == 2
    ids = {cli_id: held.id for cli_id, held in server.held.items()}
    assert server.edges == [
        (ids["s1/agent-a1"], ids["s1"], "produced_by", ""),
        (ids["s1/wf_1/agent-a1"], ids["s1"], "produced_by", "workflow-subagent: audit"),
    ]
    assert [r.error for r in results] == ["", "", ""]
    assert plan([journal]) == []
    # Asked for all the same, a file that is no sub-agent's is named as a session's is.
    assert upload._Claude().placed(journal, "claude") == Session(
        format="claude",
        id="journal",
        path=journal,
    )
    assert [s.id for s in plan([ran])] == ["s1", "s1/agent-a1", "s1/wf_1/agent-a1"]


# Measured in one claude home: a fork carries its sub-agents' transcripts under their
# ids (32 ids), and three copies of one were alike for 539 lines and then were not.
def test_a_forks_copy_of_a_sub_agent_is_a_session_of_its_own(tmp_path: Path) -> None:
    for fork, said in (("s1", "one"), ("s2", "another")):
        _ = _claude(tmp_path / "-w" / f"{fork}.jsonl")
        _ = _claude(
            tmp_path / "-w" / fork / "subagents" / "agent-a1.jsonl",
            said=said,
            agent="a1",
        )
        _ = _claude_agent(tmp_path / "-w" / fork, "a2", {"parentAgentId": "a1"})
    server = _Server()

    sessions = plan([tmp_path])
    _ = _sent(server, sessions)

    assert [(s.id, s.spawned_by) for s in sessions] == [
        ("s1", ""),
        ("s1/agent-a1", "s1"),
        ("s1/agent-a2", "s1/agent-a1"),
        ("s2", ""),
        ("s2/agent-a1", "s2"),
        ("s2/agent-a2", "s2/agent-a1"),
    ]
    said = {
        cli_id: [r.text for r in held.records["agent-a1.jsonl"].values() if r.text]
        for cli_id, held in server.held.items()
        if cli_id.endswith("agent-a1")
    }
    assert said == {
        "s1/agent-a1": ["one", "hi"],
        "s2/agent-a1": ["another", "hi"],
    }


def test_a_codex_thread_it_spawned_goes_up_linked_from_another_day(
    tmp_path: Path,
) -> None:
    root = _codex(tmp_path / "2026" / "11" / "01", _THREAD)
    spawned = _codex(tmp_path / "2026" / "11" / "02", _SPAWNED, spawner=_THREAD)
    nested = _codex(tmp_path / "2026" / "11" / "02", _NESTED, spawner=_SPAWNED)
    other = _codex(
        tmp_path / "2026" / "11" / "02",
        "01a1b7c9-0000-7000-8000-0000000000aa",
    )
    server = _Server()

    sessions = plan([spawned])
    _ = _sent(server, sessions)

    assert [(s.path, s.spawned_by, s.note) for s in sessions] == [
        (root, "", ""),
        (spawned, _THREAD, "Ada (/a)"),
        (nested, _SPAWNED, "Ada (/a)"),
    ]
    assert [s.path for s in plan([tmp_path])] == [root, spawned, nested, other]
    ids = {cli_id: held.id for cli_id, held in server.held.items()}
    assert [(child, parent) for child, parent, _, _ in server.edges] == [
        (ids[_SPAWNED], ids[_THREAD]),
        (ids[_NESTED], ids[_SPAWNED]),
    ]


def test_a_codex_rollout_outside_dated_folders_finds_its_kin_beside_it(
    tmp_path: Path,
) -> None:
    root = _codex(tmp_path / "copy", _THREAD)
    spawned = _codex(tmp_path / "copy", _SPAWNED, spawner=_THREAD)
    _ = _codex(tmp_path / "copy" / "deeper", _NESTED, spawner=_SPAWNED)

    assert [s.path for s in plan([root])] == [root, spawned]


def test_a_codex_rollout_named_otherwise_still_finds_its_kin_by_its_first_line(
    tmp_path: Path,
) -> None:
    root = _codex(tmp_path / "copy", _THREAD)
    kid = _codex(tmp_path / "copy", _SPAWNED, spawner=_THREAD).rename(
        tmp_path / "copy" / "kid.jsonl",
    )
    lone = _codex(tmp_path / "other", _NESTED).rename(tmp_path / "other" / "lone.jsonl")

    sessions = plan([kid, lone])

    assert [(s.id, s.path, s.spawned_by, s.linked) for s in sessions] == [
        (_THREAD, root, "", False),
        (_SPAWNED, kid, _THREAD, True),
        ("lone", lone, "", False),
    ]


def test_codex_threads_naming_each_other_end_the_walk_to_the_root(
    tmp_path: Path,
) -> None:
    one = _codex(tmp_path, _THREAD, spawner=_SPAWNED)
    two = _codex(tmp_path, _SPAWNED, spawner=_THREAD)
    beside = _codex(tmp_path, _NESTED, spawner=_THREAD)

    assert sorted(upload._Codex().family(one)) == sorted([one, two, beside])
    # From a thread the two spawned, the walk up meets them and ends there too.
    assert sorted(upload._Codex().family(beside)) == sorted([one, two, beside])


def test_a_sagent_agent_goes_up_linked_with_the_label_its_spawner_stood(
    tmp_path: Path,
) -> None:
    root = _sagent(
        tmp_path / "a4458be87354",
        # A line that names a standing record's kind and is none stands no child.
        said="persistent_agent",
        stood={"scout": _SAGENT_CHILD, "critic": "critic"},
    )
    child = _sagent(tmp_path / "a4458be87354" / _SAGENT_CHILD)
    watcher = _sagent(tmp_path / "a4458be87354" / _SAGENT_CHILD / "watcher")
    labelled = _sagent(tmp_path / "a4458be87354" / "critic")
    server = _Server()

    sessions = plan([child])
    _ = _sent(server, sessions)

    assert [(s.path, s.id, s.spawned_by, s.note) for s in sessions] == [
        (root, "a4458be87354", "", ""),
        (child, f"a4458be87354/{_SAGENT_CHILD}", "a4458be87354", "scout"),
        (
            watcher,
            f"a4458be87354/{_SAGENT_CHILD}/watcher",
            f"a4458be87354/{_SAGENT_CHILD}",
            "",
        ),
        (labelled, "a4458be87354/critic", "a4458be87354", "critic"),
    ]
    held = server.session(f"a4458be87354/{_SAGENT_CHILD}")
    assert list(held.parts) == [f"{_SAGENT_CHILD}.jsonl"]
    assert len(server.edges) == 3


def test_sessions_beside_an_old_sagents_log_in_their_project_are_no_agents(
    tmp_path: Path,
) -> None:
    (tmp_path / "session.jsonl").write_text('{"kind": "meta"}\n')
    session = _sagent(tmp_path / "a4458be87354")
    (tmp_path / "a4458be87354" / "session.legacy.jsonl").write_text(
        '{"kind": "meta"}\n',
    )

    sessions = plan([tmp_path])

    assert [(s.id, s.linked) for s in sessions] == [
        ("a4458be87354", False),
        (tmp_path.name, False),
    ]
    assert sessions[0].path == session


def test_without_subagents_only_sessions_linked_to_none_go(tmp_path: Path) -> None:
    _ = _claude(tmp_path / "-w" / "s1.jsonl")
    _ = _claude_agent(tmp_path / "-w" / "s1", "a1", {})
    orphan = _claude_agent(tmp_path / "-w" / "s2", "b1", {})

    sessions = plan([tmp_path], subagents=False)

    assert [(s.id, s.subagents) for s in sessions] == [("s1", 0), ("s2/agent-b1", 0)]
    assert sessions[1].path == orphan


# Folders.


def test_a_mixed_folder_finds_each_formats_sessions_and_passes_over_the_rest(
    tmp_path: Path,
) -> None:
    _ = _claude(tmp_path / "claude" / "projects" / "-w" / "s1.jsonl")
    _ = _lines(
        tmp_path / "claude" / "history.jsonl",
        {"display": "hi", "sessionId": "s1", "timestamp": 1},
    )
    (tmp_path / "claude" / "settings.json").write_text('{"model": "opus"}')
    _ = _claude_agent(tmp_path / "claude" / "projects" / "-w" / "s1", "a1", {})
    _ = _codex(tmp_path / "codex" / "sessions" / "2026" / "11" / "01", _THREAD)
    _ = _sagent(tmp_path / "sagent" / "a4458be87354")

    sessions = plan([tmp_path, tmp_path / "claude"])

    assert sorted((s.format, s.id) for s in sessions) == [
        ("claude", "s1"),
        ("claude", "s1/agent-a1"),
        ("codex", _THREAD),
        ("sagent", "a4458be87354"),
    ]


def test_a_named_format_reads_a_file_the_sniff_misses(tmp_path: Path) -> None:
    path = _claude(tmp_path / "s1.jsonl")
    path.write_text("not json\n" * 64 + path.read_text())

    assert plan([path]) == []
    assert [s.format for s in plan([path], read_as="claude")] == ["claude"]


# Again, and part of the way.


def test_a_second_upload_sends_nothing_and_lands_on_the_same_sessions(
    tmp_path: Path,
) -> None:
    _ = _claude(tmp_path / "-w" / "s1.jsonl")
    _ = _claude_agent(tmp_path / "-w" / "s1", "a1", {})
    server = _Server()
    first = _sent(server, plan([tmp_path]))
    server.calls.clear()

    again = _sent(server, plan([tmp_path]))

    assert [r.agent_session for r in again] == [r.agent_session for r in first]
    assert [(r.sent, r.held) for r in again] == [(0, 6), (0, 6)]
    assert server.calls == ["start", "parts", "end", "start", "parts", "end", "edge"]
    assert len(server.held) == 2


def test_a_part_half_there_sends_the_rest_under_the_id_it_was_stored_with(
    tmp_path: Path,
) -> None:
    path = _claude(tmp_path / "-w" / "s1.jsonl")
    server = _Server()
    _ = _sent(server, plan([path]))
    held = server.session("s1")
    ir_id = held.parts["s1.jsonl"].ir_id
    held.parts["s1.jsonl"] = held.parts["s1.jsonl"].model_copy(update={"records": 3})
    for idx in range(3, 6):
        del held.records["s1.jsonl"][idx]

    [result] = _sent(server, plan([path]))

    assert (result.sent, result.held) == (3, 3)
    assert sorted(held.records["s1.jsonl"]) == list(range(6))
    assert held.parts["s1.jsonl"].ir_id == ir_id


def test_records_go_in_batches_of_the_wires_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(upload, "MAX_RECORD_BATCH", 4)
    _ = _claude(tmp_path / "-w" / "s1.jsonl")
    server = _Server()
    refusing = _Server(refusing=frozenset({1, 4}))

    [result] = _sent(server, plan([tmp_path]))
    [refused] = _sent(refusing, plan([tmp_path]))

    assert server.calls == ["start", "parts", "append", "append", "end"]
    assert sorted(server.session("s1").records["s1.jsonl"]) == list(range(6))
    assert (result.sent, result.refused) == (6, 0)
    # One refused in the first batch and one in the second, both counted.
    assert (refused.sent, refused.refused) == (6, 2)
    assert sorted(refusing.session("s1").records["s1.jsonl"]) == list(range(6))


# The audience.


def test_public_opens_a_new_session_to_everyone_and_leaves_a_held_one_as_it_is(
    tmp_path: Path,
) -> None:
    _ = _claude(tmp_path / "-w" / "s1.jsonl")
    _ = _claude_agent(tmp_path / "-w" / "s1", "a1", {})
    _ = _claude(tmp_path / "-w" / "s2.jsonl")
    server = _Server()
    _ = _sent(server, plan([tmp_path / "-w" / "s2.jsonl"]))

    results = _sent(server, plan([tmp_path]), private=False)

    assert [(r.session.id, r.audience) for r in results] == [
        ("s1", "open"),
        ("s1/agent-a1", "open"),
        ("s2", "private"),
    ]
    assert server.session("s1/agent-a1").start.private is False


def test_a_new_sub_agent_takes_its_spawners_audience(tmp_path: Path) -> None:
    _ = _claude(tmp_path / "-w" / "s1.jsonl")
    _ = _claude_agent(tmp_path / "-w" / "s1", "a1", {})
    server = _Server()
    _ = _sent(server, plan([tmp_path], subagents=False), private=True)

    results = _sent(server, plan([tmp_path]), private=False)

    assert [r.audience for r in results] == ["private", "private"]
    assert server.session("s1/agent-a1").start.private is True


def test_a_session_the_server_holds_keeps_its_audience_and_takes_what_it_lacks(
    tmp_path: Path,
) -> None:
    path = _claude(tmp_path / "-w" / "s1.jsonl")
    server = _Server()
    _ = _sent(server, plan([path]), private=False)
    held = server.session("s1")
    held.parts["s1.jsonl"] = held.parts["s1.jsonl"].model_copy(update={"records": 3})

    [result] = _sent(server, plan([path]), private=True)

    assert (result.error, result.audience, result.sent) == ("", "open", 3)
    assert held.audience == "open"


def test_a_sub_agent_the_server_holds_keeps_its_audience_beside_its_spawners(
    tmp_path: Path,
) -> None:
    _ = _claude(tmp_path / "-w" / "s1.jsonl")
    _ = _claude_agent(tmp_path / "-w" / "s1", "a1", {})
    server = _Server()
    _ = _sent(server, plan([tmp_path]), private=True)
    server.session("s1/agent-a1").audience = "open"

    results = _sent(server, plan([tmp_path]), private=False)

    assert [r.audience for r in results] == ["private", "open"]


def test_a_server_that_predates_hiding_takes_only_what_may_be_open(
    tmp_path: Path,
) -> None:
    path = _claude(tmp_path / "-w" / "s1.jsonl")
    server = _Server(predates_hiding=True)

    [hidden] = _sent(server, plan([path]))
    calls = list(server.calls)
    [shown] = _sent(server, plan([path]), private=False)

    assert hidden.error == "the server predates hiding a session; nothing was sent"
    assert calls == ["start", "end"]
    assert (shown.error, shown.audience, shown.sent > 0) == ("", "", True)


# Failures.


def test_a_session_that_fails_is_reported_and_the_batch_goes_on(
    tmp_path: Path,
) -> None:
    _ = _claude(tmp_path / "-w" / "s1.jsonl")
    _ = _claude_agent(tmp_path / "-w" / "s1", "a1", {})
    _ = _claude_agent(tmp_path / "-w" / "s1", "a2", {"parentAgentId": "a1"})
    unreadable = _claude(tmp_path / "-w" / "s2.jsonl")
    _ = _claude(tmp_path / "-w" / "s3.jsonl")
    _ = _claude(tmp_path / "-w" / "s4.jsonl")
    server = _Server(
        refused_ids=frozenset({"s1"}),
        breaking_ids=frozenset({"s4"}),
    )
    sessions = plan([tmp_path])
    unreadable.unlink()

    results = _sent(server, sessions)

    assert [(r.session.id, r.error) for r in results] == [
        ("s1", "refused"),
        ("s1/agent-a1", "not sent: the session that spawned it failed"),
        ("s1/agent-a2", "not sent: the session that spawned it failed"),
        ("s2", f"[Errno 2] No such file or directory: '{unreadable}'"),
        ("s3", ""),
        ("s4", "ValueError"),
    ]
    assert server.closed == server.opened == 4


def test_a_refused_record_goes_as_a_stand_in_and_is_counted(tmp_path: Path) -> None:
    _ = _claude(tmp_path / "-w" / "s1.jsonl")
    server = _Server(refusing=frozenset({3}))

    [result] = _sent(server, plan([tmp_path]))

    assert (result.sent, result.refused, result.error) == (6, 1, "")
    assert server.session("s1").records["s1.jsonl"][3].kind == "UncategorizedRecord"


class _Deferred:
    """A pool that runs a session only when the batch waits, and keeps which were out.

    Stands for ``_executor`` and for ``wait``.
    """

    def __init__(self) -> None:
        self.out: dict[Future[Uploaded], tuple[Session, Callable[[], Uploaded]]] = {}
        self.together: list[list[str]] = []
        """The ids of the sessions out at each submit, the one submitted among them."""

    def __call__(self, jobs: int) -> _Deferred:
        del jobs
        return self

    def submit(
        self,
        send: Callable[..., Uploaded],
        session: Session,
        **kwargs: object,
    ) -> Future[Uploaded]:
        future: Future[Uploaded] = Future()
        self.out[future] = (session, partial(send, session, **kwargs))
        self.together.append([each.id for each, _ in self.out.values()])
        return future

    def wait(
        self,
        running: object,
        return_when: str,
    ) -> tuple[set[Future[Uploaded]], set[Future[Uploaded]]]:
        del running, return_when
        future = next(iter(self.out))
        _, job = self.out.pop(future)
        future.set_result(job())
        return {future}, set()

    def shutdown(self, *, cancel_futures: bool = False) -> None:
        del cancel_futures


# Measured in one claude home: a session continued in another folder keeps its id, so
# two files carry it. The server holds no rule against two sessions under one id, so
# two starts of a new one at once would each open a session.
def test_sessions_under_one_id_go_one_after_another_into_one_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _claude(tmp_path / "-w" / "s1.jsonl")
    _ = _claude_agent(tmp_path / "-w" / "s1", "a1", {})
    _ = _claude(tmp_path / "-w" / "s2.jsonl")
    moved = _claude(tmp_path / "-moved" / "s1.jsonl")
    _ = _claude_agent(tmp_path / "-moved" / "s1", "a1", {})
    pool = _Deferred()
    monkeypatch.setattr(upload, "_executor", pool)
    monkeypatch.setattr(upload, "wait", pool.wait)
    server = _Server()

    sessions = plan([tmp_path])
    results = _sent(server, sessions)

    assert [(s.id, s.subagents) for s in sessions] == [
        ("s1", 2),
        ("s1/agent-a1", 0),
        ("s1/agent-a1", 0),
        ("s1", 0),
        ("s2", 0),
    ]
    assert sessions[0].path == moved
    assert [r.error for r in results] == [""] * 5
    assert [r.session for r in results] == sessions
    assert pool.together == [
        ["s1"],
        ["s1", "s2"],
        ["s2", "s1"],
        ["s2", "s1", "s1/agent-a1"],
        ["s1/agent-a1"],
    ]
    ids = {cli_id: held.id for cli_id, held in server.held.items()}
    assert sorted(ids) == ["s1", "s1/agent-a1", "s2"]
    # Each file's sub-agent is linked, the one that waited too.
    assert server.edges == [(ids["s1/agent-a1"], ids["s1"], "produced_by", "")] * 2
    count = results[0].sent
    assert count > 0
    assert [(r.sent, r.held) for r in results] == [
        (count, 0),
        (count, 0),
        (0, count),
        (0, count),
        (count, 0),
    ]


class _Pool(ThreadPoolExecutor):
    """One worker thread that keeps how it was shut down; stands for ``_executor``."""

    def __init__(self) -> None:
        super().__init__(max_workers=1)
        self.cancelled: list[bool] = []

    def __call__(self, jobs: int) -> _Pool:
        del jobs
        return self

    @override
    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self.cancelled.append(cancel_futures)
        super().shutdown(wait=wait, cancel_futures=cancel_futures)


def _interrupted(*args: object, **kwargs: object) -> object:
    """Stand for Ctrl-C, which reaches the process while it waits on its workers."""
    del args, kwargs
    raise KeyboardInterrupt


def test_an_interrupt_drops_the_sessions_still_waiting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _Pool()
    monkeypatch.setattr(upload, "_executor", pool)
    monkeypatch.setattr(upload, "wait", _interrupted)
    _ = _claude(tmp_path / "-w" / "s1.jsonl")
    server = _Server()

    with pytest.raises(KeyboardInterrupt):
        _ = _sent(server, plan([tmp_path]))

    assert pool.cancelled == [True]


def test_a_finished_batch_lets_its_workers_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _Pool()
    monkeypatch.setattr(upload, "_executor", pool)
    _ = _claude(tmp_path / "-w" / "s1.jsonl")

    _ = _sent(_Server(), plan([tmp_path]))

    assert pool.cancelled == [False]


# Workers.


def test_one_job_goes_up_on_a_thread_and_more_in_processes_that_die_with_this_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list[dict[str, object]] = []

    def processes(**kwargs: object) -> ProcessPoolExecutor:
        made.append(kwargs)
        return ProcessPoolExecutor(max_workers=1)

    monkeypatch.setattr(upload, "ProcessPoolExecutor", processes)
    one = upload._executor(1)
    more = upload._executor(3)

    assert isinstance(one, ThreadPoolExecutor)
    assert isinstance(more, ProcessPoolExecutor)
    assert made == [
        {
            "max_workers": 3,
            "initializer": die_with_parent,
            "initargs": (os.getpid(), 1.0),
        },
    ]
    one.shutdown()
    more.shutdown()


def test_no_more_workers_start_than_there_are_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[int] = []

    def executor(jobs: int) -> ThreadPoolExecutor:
        asked.append(jobs)
        return ThreadPoolExecutor(max_workers=1)

    monkeypatch.setattr(upload, "_executor", executor)
    _ = _claude(tmp_path / "-w" / "s1.jsonl")
    _ = _claude(tmp_path / "-w" / "s2.jsonl")
    server = _Server()

    nothing: list[Session] = []
    for sessions in (plan([tmp_path]), nothing):
        _ = upload.upload(
            sessions,
            connect=server.connect,
            private=True,
            jobs=8,
            redactor=None,
        )

    assert asked == [2, 1]


def test_a_compressed_codex_rollout_goes_up_as_the_rollout_it_was(
    tmp_path: Path,
) -> None:
    root = _codex(tmp_path / "2026" / "11" / "01", _THREAD)
    spawned = _zst(_codex(tmp_path / "2026" / "11" / "02", _SPAWNED, spawner=_THREAD))
    server = _Server()
    first = _sent(server, plan([tmp_path]))
    packed = _zst(root)

    again = _sent(server, plan([spawned]))

    assert [(r.session.id, r.session.path, r.held) for r in first] == [
        (_THREAD, root, 0),
        (_SPAWNED, spawned, 0),
    ]
    assert [(r.session.path, r.agent_session, r.sent) for r in again] == [
        (packed, first[0].agent_session, 0),
        (spawned, first[1].agent_session, 0),
    ]
    assert [r.held for r in again] == [r.sent for r in first]
    assert list(server.session(_SPAWNED).parts) == [spawned.name.removesuffix(".zst")]
    assert len(set(server.edges)) == 1


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
