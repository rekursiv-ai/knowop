"""Tests for ``knowop.agent.custom_types``."""

from __future__ import annotations

from pathlib import Path

from trackinizer.types.streams import Stderr, Stdout

from knowop.agent.custom_types import Event


def test_an_event_s_kind_is_its_record_s_class_name() -> None:
    path = Path("rollout.jsonl")

    assert Event(record=Stdout(text="out\n"), path=path).kind == "Stdout"
    assert Event(record=Stderr(text="err\n"), path=path).kind == "Stderr"


def test_an_event_appends_unless_marked_as_a_restart() -> None:
    record = Stdout(text="line\n")
    path = Path("session.jsonl")

    assert Event(record=record, path=path).restart is False
    assert Event(record=record, path=path, restart=True).restart is True


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
