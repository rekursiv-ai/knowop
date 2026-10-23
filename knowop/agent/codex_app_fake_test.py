"""The ``codex app-server`` stand-in answers from its script, in this process."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import io
import json

import pytest

from knowop.agent.codex_app_fake import serve


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def test_it_answers_each_request_from_the_script_and_logs_every_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = tmp_path / "frames.log"
    sent: list[Mapping[str, object]] = [
        {"id": 1, "method": "initialize"},
        {"id": 2, "method": "unscripted"},
        {"id": 3, "method": "refused"},
        {"id": 4, "method": "unanswered"},
        {"id": 9, "result": {}},
        {"method": "notified"},
    ]
    script = _script(
        tmp_path,
        log=log,
        replies={
            "initialize": {"ok": True},
            "refused": {"$error": {"code": -1, "message": "no"}},
            "unanswered": {"$silent": True},
        },
        after={"initialize": [{"method": "ready"}, "not json"]},
        on_response={"9": [{"method": "thanked"}]},
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(_lines(sent)))

    serve(script)

    assert capsys.readouterr().out.splitlines() == [
        json.dumps({"id": 1, "result": {"ok": True}}),
        json.dumps({"method": "ready"}),
        "not json",
        json.dumps({"id": 2, "result": {}}),
        json.dumps({"id": 3, "error": {"code": -1, "message": "no"}}),
        json.dumps({"method": "thanked"}),
    ]
    assert log.read_text() == _lines(sent)
    assert Path(f"{log}.cwd").read_text() == str(Path.cwd())


def test_a_frame_can_write_to_stderr_and_end_the_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = _script(
        tmp_path,
        log=tmp_path / "frames.log",
        replies={},
        after={"crash": [{"$stderr": b"boom\n".hex()}, {"$exit": 3}]},
        on_response={},
    )
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(_lines([{"id": 1, "method": "crash"}])),
    )

    with pytest.raises(SystemExit) as ended:
        serve(script)

    assert ended.value.code == 3
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == (
        json.dumps({"id": 1, "result": {}}) + "\n",
        "boom\n",
    )


def _script(
    tmp_path: Path,
    *,
    log: Path,
    replies: dict[str, object],
    after: dict[str, object],
    on_response: dict[str, object],
) -> Path:
    """Write the fake's script and return its path."""
    script = tmp_path / "script.json"
    script.write_text(
        json.dumps(
            {
                "log": str(log),
                "replies": replies,
                "after": after,
                "on_response": on_response,
            },
        ),
    )
    return script


def _lines(frames: Sequence[Mapping[str, object]]) -> str:
    return "".join(json.dumps(frame) + "\n" for frame in frames)


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
