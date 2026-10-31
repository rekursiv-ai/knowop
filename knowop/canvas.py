"""The ``knowop canvas`` command: trax's canvas verb, on knowop's login.

The grammar, the requests and the output are trax's own
(:class:`~trackinizer.trax.verbs.Canvas`), and ``$TRAX_CANVAS`` names the canvas
when the command names none. An agent that ``knowop run`` started can then drive the
canvas it sits on from a computer where knowop is installed and trax is not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, override

from trackinizer.trax import verbs


if TYPE_CHECKING:
    import argparse


__all__ = ["PROGRAM", "Canvas"]

PROGRAM: Final = "knowop canvas"
"""The command's name, in help and usage."""


class Canvas(verbs.Canvas):
    """Read or operate the visual canvas paired with an agent."""

    @classmethod
    @override
    def make_parser(cls) -> argparse.ArgumentParser:
        parser = super().make_parser()
        parser.prog = PROGRAM
        return parser

    # Trax's page names the command it ships under; on a computer without trax, a
    # ``trax canvas`` example is one an agent cannot run.
    @classmethod
    @override
    def verb_help_text(cls, verb: str) -> str:
        return super().verb_help_text(verb).replace("trax canvas", PROGRAM)
