"""The ``knowop job`` verbs that read and change only what the server holds.

``status`` lists your jobs or shows one, ``cancel`` stops one, and ``secrets`` and
``secret`` keep the values a job's requests to Hugging Face and Weights & Biases carry.
None of them reaches a node, so none needs ssh.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sys

from trackinizer.client.errors import ClientError
from trackinizer.trax.render import echo


if TYPE_CHECKING:
    from trackinizer.client.client import Client
    from trackinizer.wire import wire_job
else:
    from wrapt import lazy_import

    # Only a secret's name check reads the job models, so their pydantic build stays
    # off the cold-start path of every other verb.
    wire_job = lazy_import("trackinizer.wire.wire_job")


__all__ = ["run_cancel", "run_secret", "run_secrets", "run_status"]


def run_status(client: Client, job: int | None) -> None:
    """List the caller's jobs, or show one.

    Args:
      client: The server.
      job: The job id; ``None`` lists every job.

    """
    if job is None:
        _list_jobs(client)
        return
    view = client.job_view(job)
    charged = "-" if view.gpu_minutes is None else f"{view.gpu_minutes} gpu-min"
    money = "-" if view.usd is None else f"usd {view.usd}"
    fields = {
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
    }
    for name, value in fields.items():
        echo(f"{f'{name}:':<9} {value}")


def run_cancel(client: Client, job: int) -> None:
    """Ask the server to stop a job; minutes that already ran are charged."""
    echo(f"job {job}: {client.job_cancel(job).state}")


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


def _list_jobs(client: Client) -> None:
    """Print one line per job: id, state, node, then the command cut to 60."""
    jobs = client.job_list().jobs
    if not jobs:
        echo("(no jobs)")
        return
    rows = [
        ((str(each.id), each.state, each.node or "-"), " ".join(each.argv))
        for each in jobs
    ]
    widths = [max(len(head[col]) for head, _ in rows) for col in range(3)]
    for head, command in rows:
        padded = [cell.ljust(widths[col]) for col, cell in enumerate(head)]
        echo("  ".join((*padded, command[:60])))
