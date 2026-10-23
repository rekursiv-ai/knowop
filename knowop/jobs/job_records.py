"""The ``knowop job`` verbs that read and change only what the server holds.

``status`` lists your jobs, on GPU nodes and on your own machines, or shows one,
``cancel`` stops one, and ``secrets`` and ``secret`` keep the values a job's requests
to Hugging Face and Weights & Biases carry. None of them reaches a node, so none needs
ssh. A GPU job and a machine job never share an id, so a verb given an id asks for the
machine job first and falls back to the GPU job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sys

from trackinizer.client.errors import ClientError
from trackinizer.trax.render import echo


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from trackinizer.client.client import Client
    from trackinizer.wire import wire_job
    from trackinizer.wire.wire_machine_job import MachineJobView
else:
    from wrapt import lazy_import

    # Only a secret's name check reads the job models, so their pydantic build stays
    # off the cold-start path of every other verb.
    wire_job = lazy_import("trackinizer.wire.wire_job")


__all__ = ["machine_view", "run_cancel", "run_secret", "run_secrets", "run_status"]

_NOT_HERE: Final = frozenset({404, 503})
"""The answers of a server that holds no such job of the kind asked, or runs none."""

_NO_MACHINE_JOB: Final = _NOT_HERE | {401}
"""Those, and the 401 a campaign box's credential gets on the machine-job routes,
which take only a person's key; a key refused everywhere fails on the GPU job's call."""


def machine_view(client: Client, job: int) -> MachineJobView | None:
    """Return the caller's machine job ``job``, or ``None`` when it is no machine job.

    Args:
      client: The server.
      job: A job id.

    Returns:
      view: The machine job; ``None`` when the server holds no machine job of the
        caller's by that id, or runs no machine jobs.

    """
    try:
        return client.machine_job_view(job)
    except ClientError as error:
        if error.status_code in _NO_MACHINE_JOB:
            return None
        raise


def run_status(client: Client, job: int | None) -> None:
    """List the caller's jobs, or show one.

    Args:
      client: The server.
      job: The job id; ``None`` lists every job.

    """
    if job is None:
        _list_jobs(client)
        return
    if (machine := machine_view(client, job)) is not None:
        _show_machine(machine)
        return
    view = client.job_view(job)
    charged = "-" if view.gpu_minutes is None else f"{view.gpu_minutes} gpu-min"
    money = "-" if view.usd is None else f"usd {view.usd}"
    _show(
        {
            "job": view.id,
            "state": view.state,
            "node": view.node or "-",
            "gpus": f"{view.gpus} for up to {view.minutes} min",
            "label": view.label or "-",
            "command": " ".join(view.argv),
            "commit": view.commit,
            "created": view.created_at.isoformat(),
            "started": view.started_at.isoformat() if view.started_at else "-",
            "ended": view.ended_at.isoformat() if view.ended_at else "-",
            "charged": f"{charged}, {money}",
            "reason": view.reason or "-",
        },
    )


def run_cancel(client: Client, job: int) -> None:
    """Ask the server to stop a job; a GPU job's minutes that already ran are charged.

    Args:
      client: The server.
      job: The job id: a machine job's, else a GPU job's.

    """
    try:
        state = client.machine_job_cancel(job).state
    except ClientError as error:
        if error.status_code not in _NO_MACHINE_JOB:
            raise
        state = client.job_cancel(job).state
    echo(f"job {job}: {state}")


def run_secrets(client: Client) -> None:
    """Print the names of the caller's job secrets."""
    names = client.job_secret_names().names
    if not names:
        echo("(no secrets)")
    for name in names:
        echo(name)


def run_secret(client: Client, name: str, *, op: str) -> None:
    """Set a job secret from stdin, or delete it.

    Args:
      client: The server.
      name: ``HF_TOKEN`` or ``WANDB_API_KEY``.
      op: ``set`` reads the value from stdin; ``del`` removes it.

    """
    if name not in wire_job.SECRET_NAMES:
        names = ", ".join(sorted(wire_job.SECRET_NAMES))
        raise ClientError(f"a secret name must be one of {names}")
    if op == "del":
        client.job_delete_secret(name)
        echo(f"deleted: job secret {name}")
        return
    value = sys.stdin.read().removesuffix("\n").removesuffix("\r")
    if not value:
        raise ClientError(f"stdin held no value for {name}")
    client.job_put_secret(name, value)
    echo(f"set: job secret {name}")


def _show_machine(view: MachineJobView) -> None:
    """Print one machine job, a field a line."""
    _show(
        {
            "job": view.id,
            "state": view.state,
            "machine": view.machine,
            "limits": f"{view.gpus} GPUs, {view.memory_mb} MiB, up to {view.minutes} min",
            "label": view.label or "-",
            "command": " ".join(view.argv),
            "commit": f"{view.commit} ({view.ref} at {view.remote})",
            "created": view.created_at.isoformat(),
            "started": view.started_at.isoformat() if view.started_at else "-",
            "ended": view.ended_at.isoformat() if view.ended_at else "-",
            "exit": "-" if view.exit_code is None else view.exit_code,
            "reason": view.reason or "-",
        },
    )


def _show(fields: dict[str, object]) -> None:
    """Print each field on a line of its own, the values lined up."""
    for name, value in fields.items():
        echo(f"{f'{name}:':<9} {value}")


def _list_jobs(client: Client) -> None:
    """Print one line per job, newest first: id, state, where, the command cut to 60."""
    rows = [
        (each.id, each.state, each.node or "-", each.argv)
        for each in _listed(lambda: client.job_list().jobs, not_here=_NOT_HERE)
    ] + [
        (each.id, each.state, each.machine, each.argv)
        for each in _listed(
            lambda: client.machine_job_list().jobs,
            not_here=_NO_MACHINE_JOB,
        )
    ]
    if not rows:
        echo("(no jobs)")
        return
    rows.sort(key=lambda row: row[0], reverse=True)
    heads = [(str(job), state, where) for job, state, where, _ in rows]
    widths = [max(len(head[col]) for head in heads) for col in range(3)]
    for head, (*_, argv) in zip(heads, rows, strict=True):
        padded = [cell.ljust(widths[col]) for col, cell in enumerate(head)]
        echo("  ".join((*padded, " ".join(argv)[:60])))


def _listed[T](
    listing: Callable[[], Sequence[T]],
    *,
    not_here: frozenset[int],
) -> Sequence[T]:
    """Return what ``listing`` lists; none when it answers one of ``not_here``."""
    try:
        return listing()
    except ClientError as error:
        if error.status_code in not_here:
            return ()
        raise
