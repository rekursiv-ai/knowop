"""A job's folder: its records, its log, and what a relayed read may reach in it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import base64
import hashlib
import os
import threading

from trackinizer.wire.wire_machine_job import (
    MAX_CHUNK_BYTES,
    Chunk,
    MachineJobSpec,
    ReadRefused,
    ResultManifest,
)

import pytest

from knowop.shell.job_folder import Ended, JobFolder, Launch


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


_SPEC = MachineJobSpec(
    id=7,
    owner="jan@example.com",
    argv=["python", "x.py"],
    remote="https://github.com/o/r.git",
    ref="refs/heads/main",
    commit="a" * 40,
    gpus=0,
    memory_mb=1_024,
    minutes=5,
    label="",
)
_MAX = 1_000


def _folder(tmp_path: Path) -> JobFolder:
    return JobFolder.create(tmp_path / "jobs" / "7", _SPEC)


def _data(chunk: Chunk | ReadRefused) -> bytes:
    assert isinstance(chunk, Chunk)
    return base64.b64decode(chunk.data_b64)


def test_a_new_folder_keeps_the_spec_and_only_its_user_reads_it(tmp_path: Path) -> None:
    folder = _folder(tmp_path)

    assert JobFolder(folder.path).spec() == _SPEC
    assert folder.out.is_dir()
    assert (
        folder.path.stat().st_mode & 0o777,
        folder.path.parent.stat().st_mode & 0o777,
    ) == (
        0o700,
        0o700,
    )
    with pytest.raises(FileExistsError):
        _ = JobFolder.create(folder.path, _SPEC)


def test_the_launch_and_the_end_are_recorded_whole_and_read_back(
    tmp_path: Path,
) -> None:
    folder = _folder(tmp_path)
    assert (folder.launch(), folder.ended(), folder.reported_at()) == (None, None, None)

    folder.write_launch(Launch(unit="knowop-job-ab-7", gpus=[2], memory_mb=1_024))
    folder.write_ended(Ended(state="failed", exit_code=3, reason="exit 3"))
    folder.mark_reported(1_700_000_000.5)

    assert folder.launch() == Launch(unit="knowop-job-ab-7", gpus=[2], memory_mb=1_024)
    assert folder.ended() == Ended(state="failed", exit_code=3, reason="exit 3")
    assert folder.reported_at() == 1_700_000_000.5
    assert sorted(p.name for p in folder.path.iterdir()) == [
        "ended",
        "exit.json",
        "launch.json",
        "out",
        "spec.json",
    ]


def test_the_log_is_read_from_an_offset_in_chunks_of_at_most_a_mebibyte(
    tmp_path: Path,
) -> None:
    folder = _folder(tmp_path)
    empty = folder.read_log(0)
    folder.note("fetching aaaaaaaaaaaa")
    with folder.log.open("ab") as log:
        _ = log.write(b"x" * (MAX_CHUNK_BYTES + 5))

    first = folder.read_log(0)
    rest = folder.read_log(MAX_CHUNK_BYTES)

    assert (empty.size, empty.ended, _data(empty)) == (0, False, b"")
    assert _data(first).startswith(b"knowop: fetching aaaaaaaaaaaa\n")
    assert len(_data(first)) == MAX_CHUNK_BYTES
    assert (first.offset, first.size) == (0, MAX_CHUNK_BYTES + 35)
    assert (rest.offset, len(_data(rest))) == (MAX_CHUNK_BYTES, 35)
    assert not rest.ended


def test_a_log_reads_ended_once_the_runner_or_the_host_says_so(tmp_path: Path) -> None:
    by_runner, by_host = _folder(tmp_path / "a"), _folder(tmp_path / "b")

    by_runner.write_ended(Ended(state="completed", exit_code=0, reason=""))
    by_host.mark_reported(1.0)

    assert by_runner.read_log(0).ended
    assert by_host.read_log(0).ended


def test_a_link_in_place_of_the_log_is_not_followed(tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    secret = tmp_path / "id_ed25519"
    _ = secret.write_text("PRIVATE")
    folder.log.symlink_to(secret)

    with pytest.raises(OSError, match="Too many levels of symbolic links"):
        _ = folder.read_log(0)


def _try_read(read: Callable[[], object], results: list[object]) -> None:
    try:
        results.append(read())
    except OSError as error:
        results.append(error)


@pytest.mark.parametrize("kind", ["log", "file", "manifest"])
def test_fifo_reads_refuse_without_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    folder = _folder(tmp_path)
    path = folder.log if kind == "log" else folder.out / "result"
    path.write_bytes(b"x")
    listed = folder._listing(results_max=_MAX, where="gx10")

    def listing(**kwargs: object) -> list[tuple[str, int]] | ReadRefused:
        del kwargs
        return listed

    monkeypatch.setattr(folder, "_listing", listing)
    path.unlink()
    os.mkfifo(path)
    reads: dict[str, Callable[[], object]] = {
        "log": lambda: folder.read_log(0),
        "file": lambda: folder.read_file("result", 0, results_max=_MAX, where="gx10"),
        "manifest": lambda: folder.manifest(results_max=_MAX, where="gx10"),
    }
    results: list[object] = []
    worker = threading.Thread(target=_try_read, args=(reads[kind], results))
    worker.start()
    worker.join(timeout=0.2)
    blocked = worker.is_alive()
    if blocked:
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
        os.close(fd)
        worker.join(timeout=1)

    assert not blocked, "a FIFO occupied the read worker"
    assert isinstance(results[0], OSError if kind == "log" else ReadRefused)


def _results(folder: JobFolder) -> None:
    (folder.out / "a" / "b").mkdir(parents=True)
    _ = (folder.out / "a" / "b" / "deep.txt").write_bytes(b"deep")
    _ = (folder.out / "a" / "sib.txt").write_bytes(b"sibling")
    _ = (folder.out / "top.bin").write_bytes(bytes(range(10)))


def test_the_manifest_lists_each_result_file_with_its_size_and_digest(
    tmp_path: Path,
) -> None:
    folder = _folder(tmp_path)
    _results(folder)

    manifest = folder.manifest(results_max=_MAX, where="gx10")

    assert isinstance(manifest, ResultManifest)
    assert [(e.path, e.size) for e in manifest.entries] == [
        ("a/b/deep.txt", 4),
        ("a/sib.txt", 7),
        ("top.bin", 10),
    ]
    assert manifest.entries[0].sha256 == hashlib.sha256(b"deep").hexdigest()
    assert manifest.total_bytes == 21


def test_links_fifos_and_names_that_cannot_travel_are_left_out(tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    secret = tmp_path / "id_ed25519"
    _ = secret.write_text("PRIVATE")
    (folder.out / "key").symlink_to(secret)
    (folder.out / "home").symlink_to(tmp_path, target_is_directory=True)
    os.mkfifo(folder.out / "pipe")
    _ = (folder.out / os.fsdecode(b"bad\xff")).write_bytes(b"x")
    _ = (folder.out / "two\nlines").write_bytes(b"x")
    _ = (folder.out / "ok.txt").write_bytes(b"ok")

    manifest = folder.manifest(results_max=_MAX, where="gx10")
    through_link = folder.read_file("key", 0, results_max=_MAX, where="gx10")
    into_home = folder.read_file("home/id_ed25519", 0, results_max=_MAX, where="gx10")
    escape = folder.read_file("../spec.json", 0, results_max=_MAX, where="gx10")

    assert isinstance(manifest, ResultManifest)
    assert [e.path for e in manifest.entries] == ["ok.txt"]
    assert [
        r.reason
        for r in (through_link, into_home, escape)
        if isinstance(r, ReadRefused)
    ] == [
        "'key' is not a result file of this job",
        "'home/id_ed25519' is not a result file of this job",
        "'../spec.json' is not a result file of this job",
    ]


def test_a_result_file_is_read_from_an_offset(tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    _results(folder)

    whole = folder.read_file("a/b/deep.txt", 0, results_max=_MAX, where="gx10")
    tail = folder.read_file("top.bin", 7, results_max=_MAX, where="gx10")

    assert _data(whole) == b"deep"
    assert (_data(tail), isinstance(tail, Chunk) and tail.size) == (
        bytes([7, 8, 9]),
        10,
    )


def test_a_result_replaced_by_a_link_after_it_was_listed_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = _folder(tmp_path)
    _results(folder)
    secret = tmp_path / "id_ed25519"
    _ = secret.write_text("PRIVATE")
    listed = folder._listing(results_max=_MAX, where="gx10")

    def swapped(*, results_max: int, where: str) -> object:
        del results_max, where
        (folder.out / "top.bin").unlink()
        (folder.out / "top.bin").symlink_to(secret)
        return listed

    monkeypatch.setattr(folder, "_listing", swapped)

    refused = folder.read_file("top.bin", 0, results_max=_MAX, where="gx10")

    assert refused == ReadRefused(reason="'top.bin' changed while it was read")


def test_a_manifest_whose_file_changed_while_it_was_listed_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = _folder(tmp_path)
    _results(folder)
    listed = folder._listing(results_max=_MAX, where="gx10")

    def gone(*, results_max: int, where: str) -> object:
        del results_max, where
        (folder.out / "top.bin").unlink()
        return listed

    monkeypatch.setattr(folder, "_listing", gone)

    assert folder.manifest(results_max=_MAX, where="gx10") == ReadRefused(
        reason="a result file changed while it was listed",
    )


@pytest.mark.parametrize(("limit", "sent"), [(21, True), (20, False)])
def test_results_over_the_machines_limit_are_refused_with_where_they_stay(
    tmp_path: Path,
    limit: int,
    sent: bool,
) -> None:
    folder = _folder(tmp_path)
    _results(folder)

    manifest = folder.manifest(results_max=limit, where="gx10")
    chunk = folder.read_file("top.bin", 0, results_max=limit, where="gx10")

    assert isinstance(manifest, ResultManifest) is sent
    assert isinstance(chunk, Chunk) is sent
    if not sent:
        reason = f"the results are 21 bytes, over the 20 bytes gx10 sends back; they stay in {folder.out} on gx10"
        assert manifest == ReadRefused(reason=reason)
        assert chunk == ReadRefused(reason=reason)


def test_results_of_more_files_than_a_manifest_holds_are_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "knowop.shell.job_folder.MAX_MANIFEST_ENTRIES",
        2,
    )
    folder = _folder(tmp_path)
    _results(folder)

    refused = folder.manifest(results_max=_MAX, where="gx10")

    assert refused == ReadRefused(
        reason=f"the results hold more than 2 files; they stay in {folder.out} on gx10",
    )


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
