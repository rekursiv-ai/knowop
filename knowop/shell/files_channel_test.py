"""The files channel: each request against a real folder, and the line framing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio
import os
import sys

from treekle import from_plain, json_dumps, json_loads

import pytest

from knowop.shell import files_channel
from knowop.shell.files_channel import (
    Failure,
    FilesChannel,
    Info,
    Read,
    answer,
)


if TYPE_CHECKING:
    from pathlib import Path


def _ask(op: str, path: str | Path, request_id: int = 7) -> Info | Read | Failure:
    """Answer one request line naming ``op`` and ``path``."""
    line = json_dumps({"id": request_id, "op": op, "path": str(path)})
    return answer(line.encode())


def test_a_folder_lists_its_folders_first_then_its_files_by_name(
    tmp_path: Path,
) -> None:
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "A.md").write_text("a")
    (tmp_path / "zeta").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / "to-zeta").symlink_to(tmp_path / "zeta")
    (tmp_path / "gone").symlink_to(tmp_path / "missing")

    info = _ask("info", tmp_path)

    assert isinstance(info, Info)
    assert (info.id, info.path, info.kind, info.truncated) == (
        7,
        str(tmp_path),
        "dir",
        False,
    )
    assert info.entries is not None
    assert [(each.name, each.kind, each.link) for each in info.entries] == [
        (".git", "dir", False),
        ("to-zeta", "dir", True),
        ("zeta", "dir", False),
        ("A.md", "file", False),
        ("b.txt", "file", False),
        ("gone", "other", True),
    ]


def test_a_file_is_described_without_entries(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")

    info = _ask("info", tmp_path / "a.py")

    assert isinstance(info, Info)
    assert (info.kind, info.size, info.entries, info.truncated) == (
        "file",
        6,
        None,
        None,
    )


def test_a_folder_lists_at_most_the_cap_and_says_more_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(files_channel, "MAX_ENTRIES", 2)
    for name in ("c", "a", "b"):
        (tmp_path / name).write_text(name)

    info = _ask("info", tmp_path)

    assert isinstance(info, Info)
    assert info.truncated is True
    assert [each.name for each in info.entries or []] == ["a", "b"]


def test_a_path_from_home_is_expanded_and_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "work").mkdir()

    home = _ask("info", "~")
    work = _ask("info", "~/work/../work/")

    assert isinstance(home, Info)
    assert isinstance(work, Info)
    assert (home.path, work.path) == (str(tmp_path), str(tmp_path / "work"))


def test_a_file_reads_as_text_with_a_tag_that_follows_its_changes(
    tmp_path: Path,
) -> None:
    file = tmp_path / "notes.md"
    file.write_bytes(b"# Notes\n\xff\n")

    first = _ask("read", file)
    os.utime(file, ns=(0, 1_000_000_000))
    second = _ask("read", file)

    assert isinstance(first, Read)
    assert isinstance(second, Read)
    assert (first.text, first.binary, first.too_large) == (
        "# Notes\n\N{REPLACEMENT CHARACTER}\n",
        False,
        False,
    )
    assert first.tag != second.tag
    assert second.mtime_ms == 1_000


def test_a_binary_or_too_large_file_reads_without_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(files_channel, "MAX_READ_BYTES", 4)
    (tmp_path / "image.png").write_bytes(b"\x89PN\x00")
    (tmp_path / "big.txt").write_text("hello")

    binary = _ask("read", tmp_path / "image.png")
    large = _ask("read", tmp_path / "big.txt")

    assert isinstance(binary, Read)
    assert isinstance(large, Read)
    assert (binary.text, binary.binary, binary.too_large) == (None, True, False)
    assert (large.text, large.binary, large.too_large, large.size) == (
        None,
        False,
        True,
        5,
    )


def test_a_folder_or_a_fifo_is_not_read(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "pipe")

    folder = _ask("read", tmp_path)
    fifo = _ask("read", tmp_path / "pipe")

    assert isinstance(folder, Failure)
    assert isinstance(fifo, Failure)
    assert (folder.error, fifo.error) == ("not_a_file", "not_a_file")


def test_a_missing_path_or_one_under_a_file_names_why(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")

    missing = _ask("info", tmp_path / "nope")
    under_a_file = _ask("read", tmp_path / "a.txt" / "b")

    assert isinstance(missing, Failure)
    assert isinstance(under_a_file, Failure)
    assert (missing.id, missing.error) == (7, "not_found")
    assert under_a_file.error == "not_a_dir"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any folder")
def test_a_folder_this_user_may_not_read_is_denied(tmp_path: Path) -> None:
    closed = tmp_path / "closed"
    closed.mkdir(mode=0o000)
    try:
        denied = _ask("info", closed)
    finally:
        closed.chmod(0o700)

    assert isinstance(denied, Failure)
    assert denied.error == "denied"


@pytest.mark.parametrize(
    ("line", "request_id"),
    [
        (b"not json", 0),
        (b'{"op": "info", "path": "/"}', 0),
        (b'{"id": 3, "op": "write", "path": "/"}', 3),
        (b'{"id": 4, "op": "info", "path": "relative/path"}', 4),
        (b'{"id": 5, "op": "info", "path": "/a\\u0000b"}', 5),
        (b'{"id": 6, "op": "info", "path": "~no-such-user-c8/work"}', 6),
        (None, 0),
    ],
)
def test_a_request_that_cannot_be_answered_is_a_bad_request(
    line: bytes | None,
    request_id: int,
) -> None:
    refused = answer(line)

    assert isinstance(refused, Failure)
    assert (refused.id, refused.error) == (request_id, "bad_request")


@pytest.mark.asyncio
async def test_requests_split_across_inputs_are_answered_in_order(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text("a")
    channel = FilesChannel()
    channel.set_winsize(24, 80)
    sent: asyncio.Queue[bytes] = asyncio.Queue()
    first = json_dumps({"id": 1, "op": "info", "path": str(tmp_path)}).encode()
    second = json_dumps({"id": 2, "op": "read", "path": str(tmp_path / "a.txt")})

    assert await channel.write(first[:5])
    assert await channel.write(first[5:] + b"\n\n" + second.encode())
    assert await channel.write(b"\n" + b"x" * 70_000)
    serving = asyncio.create_task(channel.serve(sent.put))
    output = b""
    while output.count(b"\n") < 3:
        output += await asyncio.wait_for(sent.get(), 5)
    serving.cancel()

    replies = [json_loads(line, dict[str, object]) for line in output.splitlines()]
    assert [(each["id"], each.get("error")) for each in replies] == [
        (1, None),
        (2, None),
        (0, "bad_request"),
    ]


@pytest.mark.asyncio
async def test_a_long_reply_is_sent_in_chunks_the_router_can_keep(
    tmp_path: Path,
) -> None:
    (tmp_path / "long.txt").write_text("x" * 200_000)
    channel = FilesChannel()
    sent: asyncio.Queue[bytes] = asyncio.Queue()
    chunks: list[bytes] = []
    line = json_dumps({"id": 1, "op": "read", "path": str(tmp_path / "long.txt")})

    assert await channel.write(line.encode() + b"\n")
    serving = asyncio.create_task(channel.serve(sent.put))
    while not chunks or not chunks[-1].endswith(b"\n"):
        chunks.append(await asyncio.wait_for(sent.get(), 5))
    serving.cancel()

    assert max(len(each) for each in chunks) <= 65_536
    reply = json_loads(b"".join(chunks), dict[str, object])
    assert reply["text"] == "x" * 200_000


@pytest.mark.asyncio
async def test_a_search_answers_with_the_lines_found_by_file_or_says_why_not(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text("one key\n")
    channel = FilesChannel()
    sent: asyncio.Queue[bytes] = asyncio.Queue()
    searches = [
        {"query": "key"},
        {"query": ""},
        {"query": "key", "path": str(tmp_path / "gone")},
        {"query": "key", "path": str(tmp_path / "a.txt")},
        {"query": "key", "path": "relative"},
        {"query": "(", "regex": True},
    ]
    serving = asyncio.create_task(channel.serve(sent.put))
    output = b""
    # One at a time: a newer search stops one still running.
    for at, search in enumerate(searches, start=1):
        line = json_dumps({"id": at, "op": "search", "path": str(tmp_path), **search})
        assert await channel.write(line.encode() + b"\n")
        while output.count(b"\n") < at:
            output += await asyncio.wait_for(sent.get(), 5)
    serving.cancel()

    replies = {
        reply["id"]: reply
        for reply in (
            json_loads(line, dict[str, object]) for line in output.splitlines()
        )
    }
    files = from_plain(replies[1]["files"], list[dict[str, object]])
    assert [
        (each["path"], len(from_plain(each["hits"], list[object]))) for each in files
    ] == [
        (str(tmp_path / "a.txt"), 1),
    ]
    assert (replies[2]["matches"], replies[2]["files"]) == (0, [])
    assert [replies[n].get("error") for n in (3, 4, 5, 6)] == [
        "not_found",
        "not_a_dir",
        "bad_request",
        "bad_request",
    ]


def test_a_symlink_loop_lists_as_other_and_reports_an_io_error(tmp_path: Path) -> None:
    loop = tmp_path / "cycle"
    loop.symlink_to(loop.name)
    info = _ask("info", tmp_path)
    refused = _ask("info", loop)

    assert isinstance(info, Info)
    assert [(entry.name, entry.kind, entry.link) for entry in info.entries or []] == [
        ("cycle", "other", True),
    ]
    assert isinstance(refused, Failure)
    assert (refused.id, refused.error) == (7, "io")


@pytest.mark.skipif(sys.platform == "darwin", reason="APFS refuses a name not in UTF-8")
def test_an_entry_whose_name_is_not_utf8_shows_a_replacement(tmp_path: Path) -> None:
    (tmp_path / os.fsdecode(b"bad\xff")).write_text("x")

    info = _ask("info", tmp_path)

    assert isinstance(info, Info)
    assert [each.name for each in info.entries or []] == [
        "bad\N{REPLACEMENT CHARACTER}",
    ]
    assert info.model_dump_json()


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
