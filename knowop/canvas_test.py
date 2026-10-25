"""``knowop canvas``: trax's canvas verb, named as knowop's in its help and usage."""

from __future__ import annotations

from trackinizer.trax import verbs

import pytest

from knowop.canvas import Canvas


def test_the_help_names_knowop_canvas_and_never_trax_canvas() -> None:
    page = Canvas.help_text()

    assert page.startswith("Usage: knowop canvas CANVAS_UUID [show TYPE")
    assert "knowop canvas UUID show trax.subgraph --record RECORD_UUID" in page
    assert "`knowop canvas UUID` lists the boxes" in page
    assert "trax canvas" not in page


def test_the_grammar_is_traxs_own() -> None:
    argv = [
        "show",
        "trax.terminal",
        "--param",
        "machine=ada-mac",
        "--placement",
        "side",
    ]

    ours = Canvas.make_parser().parse_args(argv)

    assert ours == verbs.Canvas.make_parser().parse_args(argv)


def test_a_malformed_command_names_knowop_canvas_in_its_usage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        _ = Canvas.make_parser().parse_args(["--placement", "nowhere"])

    assert capsys.readouterr().err.startswith("usage: knowop canvas ")


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
