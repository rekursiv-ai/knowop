"""Run sagent's agent live in Chat, here or on a host, or one of sagent's actions here.

::

    knowop run [KNOWOP FLAGS] -- [SAGENT ARGUMENTS]

    knowop run                                  sagent here: its screen at a terminal
    knowop run --headless -- --effort high      no screen; sagent's setting after --
    knowop run --on lab --cwd ~/work -- --model sol
    knowop run --session ID                     continue session ID here
    knowop run --session AgentSession#412       the same, by its number (or 412)
    knowop run --stop ID                        stop session ID's agent, wherever it runs
    knowop run -- login --provider oaisub       sagent's sign-in, on this computer
    knowop run -- mcp list                      any of sagent's actions, on this computer
    knowop run -- --help                        sagent's own help

A session is named by its UUID or its number, as Chat's ⋯ menu shows both.

Before ``--``, only knowop's flags below; everything after it is sagent's, untouched:
its settings as flags, or an action, which runs here and opens no session. Every sagent
flag works but those that pick a journal: knowop keeps the session's. ``knowop
--profile NAME run ...`` runs with another saved profile. Each session opens in Chat,
as one started there does, and Chat names where its agent runs.
"""

from __future__ import annotations

from datetime import timedelta
from itertools import islice
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import re
import shlex
import sys

from sagent.tui.main import opened, parse_flags
from trackinizer.trax.grammar import parse_ref
from trackinizer.wire.refs import Ref, SeqRef


if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = ["Flags", "is_action", "logs_in", "parse_run", "run_parser", "sagent_command"]


_SLURM_TIME: Final = re.compile(r"(?:([0-9]+)-)?([0-9]+)(?::([0-9]+))?(?::([0-9]+))?")
"""Slurm's ``--time``: ``M``, ``M:S`` or ``H:M:S``; ``D-H``, ``D-H:M`` or ``D-H:M:S``."""

_HELP: Final = frozenset({"-h", "--help"})


class Flags(Protocol):
    """knowop's own flags, which come before ``--``."""

    headless: bool
    session: Ref | None
    on: str | None
    cwd: str
    login_host: str
    stop: Ref | None
    time_limit: timedelta | None
    serve: str | None


def run_parser() -> argparse.ArgumentParser:
    """Return ``knowop run``'s parser of knowop's own flags.

    Returns:
      parser: Its help is this module's usage, then each flag.

    """
    parser = argparse.ArgumentParser(
        prog="knowop run",
        description=(__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    _add_arguments(parser)
    return parser


def parse_run(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
) -> tuple[Flags, list[str]]:
    """Read ``knowop run ARGV``: knowop's flags, then sagent's words after ``--``.

    Both halves are checked here, before anything reaches a server: a word before
    ``--`` that is not knowop's is refused with the command that puts it where it
    goes; sagent's words are read by sagent's own parser; an action takes none of
    knowop's flags; a sagent flag that picks a journal is refused.

    Args:
      parser: :func:`run_parser`'s parser, which says each refusal.
      argv: What follows ``knowop run``.

    Returns:
      flags: knowop's flags.
      sagent: Every word after the first ``--``, as given.

    """
    split = argv.index("--") if "--" in argv else len(argv)
    knowop, sagent = list(argv[:split]), list(argv[split + 1 :])
    parsed, unknown = parser.parse_known_args(knowop)
    flags = cast("Flags", parsed)
    if flags.stop is not None and (others := _other_flags([*knowop, *sagent])):
        parser.error(f"--stop takes no other flag: {', '.join(others)}")
    if unknown:
        parser.error(_misplaced(knowop, unknown=unknown, sagent=sagent))
    if is_action(sagent):
        if given := _other_flags(knowop, but=""):
            parser.error(_not_for_an_action(given, action=sagent))
        return flags, sagent
    picked = _journal_flags(sagent)
    if picked:
        parser.error(f"knowop keeps the session's journal; drop {', '.join(picked)}")
    return flags, sagent


def is_action(sagent: Sequence[str]) -> bool:
    """Say whether sagent's words name an action or its help, not a session's settings.

    Args:
      sagent: The words after ``--``.

    Returns:
      action: Whether the first word is no flag, or a help flag is among the words.

    """
    return bool(sagent) and (not sagent[0].startswith("-") or bool(_HELP & {*sagent}))


def logs_in(argv: Sequence[str]) -> bool:
    """Say whether ``knowop run ARGV`` reaches the server with a saved profile.

    A refused command is refused here, before a login is asked for, and exits 2 as
    argparse exits.

    Args:
      argv: What follows ``knowop run``.

    Returns:
      logs: False for knowop's help, for a daemon's start, which the daemon hands a
        key, and for an action of sagent's, which runs here alone.

    """
    split = argv.index("--") if "--" in argv else len(argv)
    if _HELP & {*argv[:split]}:
        return False
    flags, sagent = parse_run(run_parser(), argv)
    return flags.serve is None and not is_action(sagent)


def sagent_command(sagent: Sequence[str]) -> list[str]:
    """Return the command that runs this install's sagent with ``sagent``'s words.

    The published package is named otherwise, so sagent's package is named from the
    module it is imported from; ``-P`` keeps a module in the folder from running in
    its place.

    Args:
      sagent: The words after ``--``.

    Returns:
      command: The interpreter, its flags, sagent's package and the words.

    """
    return [sys.executable, "-P", "-m", opened.__module__.rsplit(".", 2)[0], *sagent]


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register knowop's flags on ``parser``."""
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run with no screen, at a terminal too.",
    )
    parser.add_argument(
        "--session",
        type=_session,
        metavar="ID",
        help="The session to continue here: its UUID or number (AgentSession#N).",
    )
    parser.add_argument(
        "--on",
        nargs="?",
        const="",
        metavar="NAME",
        help="Start on host NAME through its daemon; with no name, the server's.",
    )
    parser.add_argument(
        "--cwd",
        default="~",
        metavar="DIR",
        help="With --on, the folder there to start in (default: %(default)s).",
    )
    parser.add_argument(
        "--login-host",
        default="",
        metavar="NAME",
        help="Use host NAME's model login.",
    )
    parser.add_argument(
        "--stop",
        type=_session,
        metavar="ID",
        help="Stop session ID's agent, wherever it runs, and exit.",
    )
    parser.add_argument(
        "--time",
        type=_time_limit,
        metavar="TIME",
        dest="time_limit",
        help=(
            "End TIME after it starts, as a stop ends it; Slurm's --time: 120, "
            "2:00:00 or 1-0."
        ),
    )
    parser.add_argument(
        "--serve",
        metavar="URL",
        help="Serve the session a daemon hands on stdin, through the server at URL.",
    )


def _session(text: str) -> Ref:
    """Read a session as its UUID or number: ``AgentSession#412``, or ``412`` alone."""
    try:
        ref = parse_ref(f"AgentSession#{text}" if text.isdigit() else text)
    except ValueError:
        ref = None
    if ref is None or (isinstance(ref, SeqRef) and ref.kind != "AgentSession"):
        raise argparse.ArgumentTypeError(
            f"{text!r} is no session: give its UUID or number (AgentSession#N)",
        )
    return ref


# Without days, ``M``, ``M:S`` or ``H:M:S``; with them, ``D-H``, ``D-H:M`` or
# ``D-H:M:S``.
def _time_limit(text: str) -> timedelta:
    """Read a time limit as Slurm's ``--time`` reads one, refusing one of no time."""
    matched = _SLURM_TIME.fullmatch(text)
    if matched is not None:
        days, *clock = matched.groups()
        given = [int(field) for field in clock if field is not None]
        # Without days, one or two fields are minutes first: ``M`` or ``M:S``.
        if days is None and len(given) < 3:
            given = [0, *given]
        hours, minutes, seconds = (*given, 0, 0)[:3]
        limit = timedelta(
            days=int(days or 0),
            hours=hours,
            minutes=minutes,
            seconds=seconds,
        )
        if limit:
            return limit
    raise argparse.ArgumentTypeError(
        f"{text!r} is no time limit, such as 120, 2:00:00 or 1-0",
    )


# Read off the words, not the parsed values: a flag given at its default, such as
# ``--cwd ~``, parses as if it were not given.
def _other_flags(words: Sequence[str], *, but: str = "--stop") -> list[str]:
    """Return each flag ``words`` give besides ``but``, by the name given."""
    names = (word.partition("=")[0] for word in words)
    return [name for name in names if name.startswith("--") and name != but]


# `argparse` leaves each word it does not know in order, a value after its flag, so
# what is left once they are taken out, in order, is what it read as knowop's.
def _misplaced(
    knowop: Sequence[str],
    *,
    unknown: Sequence[str],
    sagent: Sequence[str],
) -> str:
    """Say which words before ``--`` are not knowop's, and the command that places them."""
    kept: list[str] = []
    left = list(unknown)
    for word in knowop:
        if left and word == left[0]:
            del left[0]
        else:
            kept.append(word)
    profile: list[str] = []
    theirs: list[str] = []
    words = iter(unknown)
    for word in words:
        if word.partition("=")[0] == "--profile":
            profile.extend([word] if "=" in word else [word, *islice(words, 1)])
        else:
            theirs.append(word)
    said: list[str] = []
    if theirs:
        # An action is named by its first word; settings, by each flag.
        named = (
            [w.partition("=")[0] for w in theirs if w.startswith("-")]
            if theirs[0].startswith("-")
            else theirs[:1]
        )
        one = len(named) == 1
        said.append(
            f"{', '.join(named)} {'is' if one else 'are'} sagent's, so "
            f"{'it goes' if one else 'they go'} after --",
        )
    if profile:
        said.append("--profile is knowop's own, so it goes before run")
    after = [*theirs, *sagent]
    fixed = ["knowop", *profile, "run", *kept, *(["--", *after] if after else [])]
    return f"{'; '.join(said)}: {shlex.join(fixed)}"


def _not_for_an_action(given: Sequence[str], *, action: Sequence[str]) -> str:
    """Say why an action of sagent's takes knowop's ``given`` flags, which it does not."""
    named = (
        next(word for word in action if word in _HELP)
        if action[0].startswith("-")
        else action[0]
    )
    fixed = shlex.join(["knowop", "run", "--", *action])
    if "--on" in given:
        return (
            f"--on starts a session on a host, and sagent's {named} runs on this "
            f"computer alone; run it on that host: {fixed}"
        )
    one = len(given) == 1
    return (
        f"{', '.join(given)} {'is' if one else 'are'} for a session, and sagent's "
        f"{named} opens none: {fixed}"
    )


def _journal_flags(sagent: Sequence[str]) -> list[str]:
    """Return the flags of ``sagent`` that pick a journal, which knowop refuses."""
    flags, _ = parse_flags(sagent)
    named = {
        "--session": flags.session is not None,
        "--ephemeral": flags.ephemeral,
        "--resume": flags.resume is not None,
        "--continue": flags.continue_,
        "--resume-all": flags.resume_all,
        "--continue-all": flags.continue_all,
        "--hosted-session": flags.hosted_session is not None,
    }
    return [name for name, given in named.items() if given]
