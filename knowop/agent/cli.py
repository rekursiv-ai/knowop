"""The agent runner ``knowop run`` hands to: Claude Code or Codex here, recorded.

``knowop run claude [ARGS]`` calls :func:`main`, which runs Claude Code in the current
folder, and ``knowop run codex [ARGS]`` calls :func:`codex_main`, which runs Codex.
Each run is recorded as an AgentSession on the server of the profile it is given, and
a message sent to it from Chat reaches it. ``--detach`` hosts the run in the
background, and the host is reached by name:

    knowop run claude --detach --name fix-tests
    knowop run ls | attach NAME | log NAME [-f] | send NAME TEXT | stop NAME

``knowop run claude resume SESSION`` re-enters a stored session as claude, and
``knowop run codex resume SESSION`` as codex. Everything after ``--`` goes to the CLI
itself. The connection flags ``--profile``, ``--host`` and ``--port`` come first, as
they do for ``trax``: knowop puts its own profile there. Unattended plumbing that
keeps trax's connection instead (``run_agent.sh``) runs this package's
``__main__.py``, or ``codex.py`` for Codex.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import atexit
import functools
import sys
import uuid

from trackinizer.client.errors import ClientError
from trackinizer.trax.client_cache import close_clients
from trackinizer.trax.connection import connect, connect_flags
from trackinizer.trax.profile import unmatched_server_hint
from trackinizer.trax.render import echo

from knowop.agent import session
from knowop.agent.errors import (
    CiphertextDroppedError,
    LossyConversionError,
    NotResumableError,
)
from knowop.agent.program import PROGRAM


if TYPE_CHECKING:
    from collections.abc import Sequence

    from knowop.agent.resume import prepare_resume
else:
    from wrapt import lazy_import

    # Only a resume loads a stored conversation, and the loader brings the readers and
    # writers of every CLI's format.
    prepare_resume = lazy_import(
        "knowop.agent.resume",
        "prepare_resume",
    )


__all__ = ["codex_main", "main"]

# The connection flags that take a value, as ``trax`` peels them before its verb.
_VALUE_FLAGS: Final = frozenset({"--profile", "--host", "--port"})


def main(argv: Sequence[str] | None = None) -> int:
    """Run Claude Code, recorded (``knowop run claude``); return the exit code.

    Args:
      argv: Command-line arguments; ``None`` for ``sys.argv[1:]``.

    Returns:
      exit_code: The wrapped CLI's status; for ``--detach``, whether the host came
        up; 2 when a command is refused.

    """
    return _run("claude", sys.argv[1:] if argv is None else argv)


def codex_main(argv: Sequence[str] | None = None) -> int:
    """Run Codex, recorded (``knowop run codex``); return the exit code.

    Args:
      argv: Command-line arguments; ``None`` for ``sys.argv[1:]``.

    Returns:
      exit_code: The wrapped CLI's status; for ``--detach``, whether the host came
        up; 2 when a command is refused.

    """
    return _run("codex", sys.argv[1:] if argv is None else argv)


class _Connection(Protocol):
    """The connection flags, as ``connect_flags`` registers them."""

    profile: str | None


class _ResumeFlags(Protocol):
    """The flags the resume tail takes for itself."""

    session: uuid.UUID
    lossy: bool


def _run(cli: str, argv: Sequence[str]) -> int:
    """Run ``knowop run <cli>`` on ``argv``; return its exit code."""
    # Drain pooled sockets at exit; httpx2 warns if a Client is garbage collected with
    # any still open.
    atexit.register(close_clients)
    if (hint := unmatched_server_hint()) is not None:
        _ = sys.stderr.write(hint)
    try:
        connection, rest = _peel_connection(argv)
        if rest[:1] == ["resume"]:
            return _resume(cli, rest[1:], connection)
        return session.main(
            rest,
            default_cli=cli,
            client_factory=functools.partial(connect, connection),
            profile=cast(_Connection, connection).profile,
        )
    except ClientError as err:
        _ = sys.stderr.write(f"{PROGRAM} {cli}: {err}\n")
        return 2


# Only the leading run of connection flags, so a ``--profile`` after the first word is
# the runner's own or the CLI's, never this layer's.
def _peel_connection(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split the leading ``--profile``, ``--host`` and ``--port`` from ``argv``."""
    end = 0
    while end < len(argv) and argv[end].split("=", 1)[0] in _VALUE_FLAGS:
        end += 1 if "=" in argv[end] else 2
    parser = argparse.ArgumentParser(prog=PROGRAM, add_help=False)
    connect_flags(parser)
    return parser.parse_args(list(argv[:end])), list(argv[end:])


def _resume(cli: str, argv: Sequence[str], connection: argparse.Namespace) -> int:
    """Re-enter a stored session as ``cli``: ``SESSION [--lossy] [ARGS] [-- CLI ARGS]``."""
    boundary = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(
        prog=f"{PROGRAM} {cli} resume",
        description=f"Re-enter a stored session as {cli}.",
        epilog="Other arguments go to the runner; those after -- go to the CLI.",
    )
    _add_resume_arguments(parser)
    parsed, runner_args = parser.parse_known_args(list(argv[:boundary]))
    flags = cast(_ResumeFlags, parsed)
    client = connect(connection)
    try:
        written = prepare_resume(client, flags.session, cli, lossy=flags.lossy)
    except (NotResumableError, LossyConversionError) as err:
        raise ClientError(str(err)) from err
    except CiphertextDroppedError as err:
        raise ClientError(
            f"{err} (its records remain searchable; only the replay is lost)",
        ) from err
    echo(f"resuming {flags.session} as {cli} from {written.path}")
    return session.main(
        [*runner_args, *argv[boundary:]],
        default_cli=cli,
        client_factory=functools.partial(connect, connection),
        resume_path=written.path,
        cli_session_id=str(written.cli_session_id),
        profile=cast(_Connection, connection).profile,
        attach=written.attach,
        prefix_bytes=written.prefix_bytes,
    )


def _add_resume_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the resume tail's own flags on ``parser``."""
    parser.add_argument("session", type=uuid.UUID, help="The AgentSession to re-enter.")
    parser.add_argument(
        "--lossy",
        action="store_true",
        help=(
            "Accept a conversion that drops records the CLI cannot express, or a "
            "part whose records capture truncated."
        ),
    )
