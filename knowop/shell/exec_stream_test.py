"""The stream cutter and the folder report of a command's shell."""

from __future__ import annotations

import os

from trackinizer.wire.wire_exec import MAX_STREAM_BYTES

from knowop.shell.exec_stream import Cut, read_report, script


def _kept(chunks: list[bytes]) -> tuple[bytes, bool]:
    cut = Cut()
    for chunk in chunks:
        cut.feed(chunk)
    return cut.value()


def test_a_stream_within_the_limit_is_kept_whole_however_it_arrives() -> None:
    stream, cut = _kept([b"ab", b"", b"cd" * 1000, b"e"])

    assert stream == b"ab" + b"cd" * 1000 + b"e"
    assert not cut


def test_a_stream_exactly_at_the_limit_loses_nothing() -> None:
    stream, cut = _kept([b"x" * MAX_STREAM_BYTES])

    assert len(stream) == MAX_STREAM_BYTES
    assert not cut


def test_a_stream_past_the_limit_keeps_both_ends_and_says_how_much_went() -> None:
    chunks = [b"HEAD", b"m" * 3_000_000, b"TAIL"]

    stream, cut = _kept(chunks)

    assert cut
    assert len(stream) <= MAX_STREAM_BYTES
    assert stream.startswith(b"HEADmmm")
    assert stream.endswith(b"mmmTAIL")
    assert b"bytes cut" in stream


def test_a_stream_one_byte_past_the_limit_is_cut_and_still_fits() -> None:
    stream, cut = _kept([b"x" * (MAX_STREAM_BYTES + 1)])

    assert cut
    assert len(stream) <= MAX_STREAM_BYTES


def test_the_shell_script_reports_its_folder_as_it_exits() -> None:
    text = script("cd /usr", report_fd=7)

    assert text == "trap 'printf %s \"$PWD\" >&7' EXIT\ncd /usr"


def test_a_report_is_read_when_it_is_a_folder_the_wire_carries() -> None:
    read_end, write_end = os.pipe()
    try:
        _ = os.write(write_end, b"/home/ada/work")
        os.close(write_end)

        assert read_report(read_end, fallback="/start") == "/home/ada/work"
    finally:
        os.close(read_end)


def test_a_missing_or_unsound_report_falls_back_to_where_the_command_started() -> None:
    for written in (b"", b"relative", b"/with\nnewline"):
        read_end, write_end = os.pipe()
        try:
            _ = os.write(write_end, written)
            if written:
                os.close(write_end)
            assert read_report(read_end, fallback="/start") == "/start"
        finally:
            os.close(read_end)
            if not written:
                os.close(write_end)


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
