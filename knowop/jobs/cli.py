"""The job runner ``knowop job`` hands to: run jobs on a GPU node from this computer.

``submit`` snapshots the repository you are in and uploads it to a GPU node over ssh;
``logs`` and ``results`` read the job back from the node. The server only brokers the
job: no code or result bytes pass through it. ``--profile``, ``--host`` and ``--port``
before the verb choose the server, as they do for ``trax``: knowop puts its own
profile there.

Examples:
  knowop job submit --minutes 30 --follow -- python train.py --steps 100
  knowop job logs 7 -f
  knowop job results 7 --to ~/results/run7

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import argparse
import functools
import sys

from trackinizer.client.errors import ClientError
from trackinizer.trax.client_cache import close_clients
from trackinizer.trax.connection import connect, connect_flags
from trackinizer.trax.profile import unmatched_server_hint

from knowop.jobs.job import JOB_WORD, PROGRAM, Job


if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = ["main"]

_VALUE_FLAGS: Final = frozenset({"--profile", "--host", "--port"})


def main(argv: Sequence[str] | None = None) -> int:
    """Run one ``knowop job`` command; return the process exit code.

    Args:
      argv: Command-line arguments; ``sys.argv[1:]`` when omitted.

    Returns:
      code: 0 on success, 2 when the command is refused. A job that ends badly
        exits 1 and an interrupted log exits 130, as the command decides.

    """
    connection, rest = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if (hint := unmatched_server_hint()) is not None:
            _ = sys.stderr.write(hint)
        Job.dispatch(JOB_WORD, rest, functools.partial(connect, connection))
    except ClientError as err:
        _ = sys.stderr.write(f"{PROGRAM}: {err}\n")
        return 2
    finally:
        close_clients()
    return 0


# A flag spelling after the verb is the command's own: ``submit -- CMD --port 9`` sends
# ``--port 9`` to the job.
def _parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    """Peel the connection flags in front of the verb; return them and the rest."""
    tokens = list(argv)
    # pragma: no mutate start -- ``None`` is as falsy as ``False``: no help flag either way.
    parser = argparse.ArgumentParser(prog=PROGRAM, add_help=False)
    # pragma: no mutate end
    connect_flags(parser)
    end = _prefix_end(tokens)
    flags, unknown = parser.parse_known_args(tokens[:end])
    return flags, [*unknown, *tokens[end:]]


def _prefix_end(tokens: Sequence[str]) -> int:
    """Return the index of the first token that is not a leading connection flag."""
    index = 0
    while index < len(tokens) and tokens[index].startswith("--"):
        # ``--flag=value`` carries its value; a bare value flag takes the next token.
        if "=" not in tokens[index] and tokens[index] in _VALUE_FLAGS:
            index += 1
        index += 1
    return min(index, len(tokens))
