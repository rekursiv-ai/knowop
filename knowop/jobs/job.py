"""The ``knowop job`` commands: run a job on a GPU node, straight from this computer.

``submit`` snapshots the repository you are in, asks the server for a job and a
one-job ssh certificate, and uploads the code as a git bundle directly to the node.
``results`` and ``logs`` read the job back from the node the same way. The server
only brokers the job; no code or result bytes pass through it, and this computer
holds no long-lived key: every command makes a throwaway key and deletes it when it
ends. It needs ``ssh``, ``ssh-keygen`` and ``git`` on ``PATH``.

``status``, ``cancel``, ``secrets`` and ``secret`` only read and change what the
server holds; :mod:`job_records` runs them, with no ssh.

``submit --machine NAME`` runs the job on a computer of your own that serves jobs
(``knowop connect --jobs``) instead: :mod:`machine_job` names the commit you pushed,
the machine fetches it, and ``logs`` and ``results`` of such a job are read through
the server, with no ssh.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast, override

import argparse
import codecs
import io
import re
import shutil
import tempfile
import time

from trackinizer.client.errors import ClientError
from trackinizer.lib.userdirs import data_dir
from trackinizer.trax.commands import Command, HelpPage
from trackinizer.trax.connection import server_dir
from trackinizer.trax.context import cwd, out_stream
from trackinizer.trax.render import echo

from knowop.jobs import job_records


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence
    from typing import TextIO

    from _typeshed import ReadableBuffer
    from trackinizer.client.client import Client
    from trackinizer.wire import wire_job
    from trackinizer.wire.wire_job import Grant, GrantStatus

    import pydantic

    from knowop.jobs import (
        job_fetch,
        job_source,
        job_ssh,
        machine_job,
    )
    from knowop.jobs.job_source import BundleInfo, Snapshot
    from knowop.jobs.job_ssh import Gate, Runner, Tools
    from knowop.shell import job_limits
else:
    from wrapt import lazy_import

    # The pydantic models, ssh and git machinery load only when a ``job`` verb runs,
    # so they stay off the cold-start path of every other verb.
    pydantic = lazy_import("pydantic")
    wire_job = lazy_import("trackinizer.wire.wire_job")
    job_fetch = lazy_import("knowop.jobs.job_fetch")
    job_source = lazy_import("knowop.jobs.job_source")
    job_ssh = lazy_import("knowop.jobs.job_ssh")
    machine_job = lazy_import("knowop.jobs.machine_job")
    job_limits = lazy_import("knowop.shell.job_limits")


JOB_WORD: Final = "job"
"""The word for a job in messages, and the name the command is matched by."""

PROGRAM: Final = "knowop job"
"""The command's name, in help, usage and messages."""

_SUBMIT: Final = (
    f"{PROGRAM} submit [--machine NAME [--memory SIZE]] [--gpus N] [--minutes M] "
    "[--label TEXT] [--follow] [--include-untracked] -- CMD..."
)
_UNTRACKED_SHOWN: Final = 10
_MEMORY: Final = "32G"
"""A machine job's memory cap when ``--memory`` names none."""


class Rig:
    """What a flow talks to: the server, ssh, and the clock."""

    def __init__(
        self,
        *,
        client: Client,
        runner: Runner,
        tools: Tools,
        sleep: Callable[[float], None],
        clock: Callable[[], float],
    ) -> None:
        self.client = client
        self.runner = runner
        self.tools = tools
        self.sleep = sleep
        self.clock = clock


class Job(Command):
    """Run jobs on a GPU node, read their logs and results, and list and cancel them."""

    names = (JOB_WORD,)

    help = HelpPage(
        usage=_SUBMIT,
        summary=(
            "Runs CMD on a GPU node. 'submit' snapshots the repository you are in "
            "(uncommitted and untracked files included, ignored files left out), "
            "says which untracked files it took, uploads it straight to the node "
            "over ssh, and prints the job id. "
            "'results' downloads the job's output straight from the node. The "
            "server only brokers the job and signs a one-job certificate; no code "
            "or result bytes pass through it, and this computer keeps no key. "
            "With --machine NAME, CMD runs instead on a computer of yours that runs "
            "'knowop connect --jobs': it fetches the commit you pushed into a fresh "
            "worktree and runs CMD there under a memory cap, and its log and results "
            "come back through the server, which keeps none of them."
        ),
        arguments=(
            ("JOB", "a job id, as 'submit' printed it"),
            ("CMD...", "the command, run without a shell in the snapshot's root"),
            ("NAME", "HF_TOKEN or WANDB_API_KEY"),
        ),
        options=(
            (
                "--machine NAME",
                (
                    "with submit: run on your computer NAME, from the commit your "
                    "branch was pushed at; the worktree must be clean"
                ),
            ),
            (
                "--memory SIZE",
                f"with --machine: the job's memory cap, such as 8G (default {_MEMORY})",
            ),
            (
                "--gpus N",
                "GPUs for the job, 1 to 8; 0 too with --machine (default 1)",
            ),
            (
                "--minutes M",
                "wall-clock limit, 1 to 360; to 1440 with --machine (default 10)",
            ),
            ("--label TEXT", "a short name for the job"),
            ("--follow", "with submit: stream the log until the job ends"),
            (
                "--include-untracked",
                (
                    "with submit: also send untracked files named like secrets "
                    "(.env*, *.pem, id_*, *credentials*, .netrc, *.key), which are "
                    "refused unless .gitignore covers them"
                ),
            ),
            ("-f", "with logs: stream the log until the job ends"),
            ("--to DIR", "with results: download here, outside the repository"),
        ),
        examples=(
            f"{_SUBMIT}",
            f"{PROGRAM} logs JOB [-f]                 print the job's log",
            f"{PROGRAM} results JOB [--to DIR]        download the job's out/",
            f"{PROGRAM} status [JOB]                  list your jobs, or show one",
            f"{PROGRAM} cancel JOB                    stop a job",
            f"{PROGRAM} secrets                       list your secrets' names",
            f"{PROGRAM} secret NAME set < FILE        set a secret from stdin",
            f"{PROGRAM} secret NAME del               remove a secret",
        ),
        notes=(
            "Everything after the first '--' is the job's command, even '--help'.",
            (
                "A secret's value is read from stdin only. The job never sees it: "
                "the node adds it to requests to Hugging Face and Weights & Biases."
            ),
            "Minutes a cancelled job already ran are charged.",
            (
                "Results go to a directory of your own data folder unless --to names "
                "one; a directory inside the repository is refused. 'src.bundle', the "
                "exact code that ran, is kept beside them for a later pull request."
            ),
            (
                "--profile NAME, --host HOST and --port N go before the verb and choose "
                "the server, as they do for trax."
            ),
            "submit, logs and results need ssh, ssh-keygen and git on PATH.",
            (
                "A job on a machine writes its results to $KNOWOP_JOB_OUT; the "
                "machine sends back at most what its owner allows (256M unless "
                "'knowop connect --job-results-max' says otherwise)."
            ),
        ),
    )

    @classmethod
    @override
    def make_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog=PROGRAM,
            description=cls.__doc__,
            add_help=False,
        )
        # A verb sets only its own flags; the rest read as these, so ``run`` reads all.
        parser.set_defaults(
            gpus=1,
            minutes=10,
            label="",
            follow=False,
            include_untracked=False,
            machine="",
            memory=None,
            job=None,
            to=None,
            name="",
            op="",
        )
        verbs = parser.add_subparsers(dest="action", required=True, metavar="VERB")
        submit = verbs.add_parser("submit", add_help=False)
        submit.add_argument("--machine", default="")
        submit.add_argument("--memory", default=None)
        submit.add_argument("--gpus", type=int, default=1)
        submit.add_argument("--minutes", type=int, default=10)
        submit.add_argument("--label", default="")
        submit.add_argument("--follow", action="store_true")
        submit.add_argument("--include-untracked", action="store_true")
        logs = verbs.add_parser("logs", add_help=False)
        logs.add_argument("job", type=int)
        logs.add_argument("-f", "--follow", action="store_true")
        results = verbs.add_parser("results", add_help=False)
        results.add_argument("job", type=int)
        results.add_argument("--to", default=None)
        verbs.add_parser("status", add_help=False).add_argument(
            "job",
            nargs="?",
            type=int,
            default=None,
        )
        verbs.add_parser("cancel", add_help=False).add_argument("job", type=int)
        _ = verbs.add_parser("secrets", add_help=False)
        secret = verbs.add_parser("secret", add_help=False)
        secret.add_argument("name")
        secret.add_argument("op", choices=("set", "del"))
        return parser

    @classmethod
    @override
    def dispatch(
        cls,
        verb: str,
        rest: list[str],
        client_factory: Callable[[], Client],
    ) -> None:
        # The job's command is split off first: ``knowop job submit -- python x.py
        # --help`` must reach the job, and the base class would show help instead.
        head, command = _split(rest)
        if _asks_for_help(head):
            echo(cls.help_text(), nl=False)
            return
        args = cls.make_parser().parse_args(head)
        args.command = tuple(command)
        cls.run(verb, args, client_factory)

    @classmethod
    @override
    def run(
        cls,
        verb: str,
        args: argparse.Namespace,
        client_factory: Callable[[], Client],
    ) -> None:
        del verb
        flags = cast(_Flags, args)
        if flags.action != "submit" and flags.command:
            raise ClientError("only submit takes a command after --")
        if flags.action == "submit" and not flags.command:
            raise ClientError(f"submit needs a command after --: {_SUBMIT}")
        job = flags.job
        match flags.action:
            case "submit" if flags.machine:
                cls.run_machine_submit(
                    client_factory(),
                    machine=flags.machine,
                    command=flags.command,
                    gpus=flags.gpus,
                    memory=flags.memory or _MEMORY,
                    minutes=flags.minutes,
                    label=flags.label,
                    follow=flags.follow,
                    include_untracked=flags.include_untracked,
                )
            case "submit" if flags.memory is not None:
                raise ClientError(
                    "--memory caps a job on a machine: add --machine NAME",
                )
            case "submit":
                cls.run_submit(
                    _rig(client_factory),
                    command=flags.command,
                    gpus=flags.gpus,
                    minutes=flags.minutes,
                    label=flags.label,
                    follow=flags.follow,
                    include_untracked=flags.include_untracked,
                )
            case "status":
                job_records.run_status(client_factory(), job)
            case "secrets":
                job_records.run_secrets(client_factory())
            case "secret":
                job_records.run_secret(client_factory(), flags.name, op=flags.op)
            case "cancel" if isinstance(job, int):
                job_records.run_cancel(client_factory(), job)
            case "logs" if isinstance(job, int):
                client = client_factory()
                if job_records.machine_view(client, job) is not None:
                    cls.run_machine_logs(client, job, follow=flags.follow)
                else:
                    cls.run_logs(_rig(client_factory), job, follow=flags.follow)
            case _ if isinstance(job, int):
                client = client_factory()
                if (view := job_records.machine_view(client, job)) is not None:
                    machine_job.run_results(
                        client,
                        view,
                        results=results_dir(client.base_url, job, flags.to),
                    )
                else:
                    cls.run_results(_rig(client_factory), job, to=flags.to)

    @classmethod
    def run_machine_submit(
        cls,
        client: Client,
        *,
        machine: str,
        command: Sequence[str],
        gpus: int,
        memory: str,
        minutes: int,
        label: str,
        follow: bool,
        include_untracked: bool,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Queue a job on one of your machines, print its id once taken, maybe follow.

        Args:
          client: The server.
          machine: The machine's name.
          command: The job's command line.
          gpus: How many GPUs.
          memory: The memory cap, such as ``32G``.
          minutes: The wall-clock limit.
          label: A short name for the job.
          follow: Stream the log until the job ends.
          include_untracked: Refused: a machine runs the pushed commit, no snapshot.
          sleep: Waits between looks at the job.
          clock: Monotonic seconds.

        """
        if include_untracked:
            raise ClientError(
                "a job on a machine runs the commit you pushed, so "
                "--include-untracked has nothing to send",
            )
        try:
            memory_mb = job_limits.parse_size(memory) // 2**20
        except ValueError as error:
            raise ClientError(f"--memory: {error}") from None
        job = machine_job.run_submit(
            client,
            repo=cwd(),
            machine=machine,
            command=command,
            gpus=gpus,
            memory_mb=memory_mb,
            minutes=minutes,
            label=label,
            sleep=sleep,
            clock=clock,
        )
        echo(job)
        if follow:
            cls.run_machine_logs(client, job, follow=True)

    @classmethod
    def run_machine_logs(cls, client: Client, job: int, *, follow: bool) -> None:
        """Print a machine job's log; with ``follow``, until the job ends.

        Args:
          client: The server.
          job: The job's id.
          follow: Keep printing until the job ends.

        Raises:
          SystemExit: 130 on Ctrl-C; the job runs on.

        """
        try:
            machine_job.run_logs(
                client,
                job,
                follow=follow,
                sink=_TextSink(out_stream()),
            )
        except KeyboardInterrupt:
            raise SystemExit(130) from None

    @classmethod
    def run_submit(
        cls,
        rig: Rig,
        *,
        command: Sequence[str],
        gpus: int,
        minutes: int,
        label: str,
        follow: bool,
        include_untracked: bool,
    ) -> None:
        """Snapshot the repository, upload it to a node, and start the job.

        Prints the job id once the server says the job is queued or later.

        Args:
          rig: The server, ssh and the clock.
          command: The job's command line.
          gpus: How many GPUs.
          minutes: The wall-clock limit.
          label: A short name for the job.
          follow: Stream the log until the job ends.
          include_untracked: Send untracked files named like secrets, which the
            snapshot refuses otherwise.

        """
        with tempfile.TemporaryDirectory(prefix="job-") as scratch:
            snap = job_source.snapshot(
                cwd(),
                Path(scratch),
                allow_secret_names=include_untracked,
            )
            of_head = (
                "" if snap.head in {"", snap.commit} else f" (HEAD {snap.head[:12]})"
            )
            echo(
                f"snapshot: {snap.files} files at {snap.commit[:12]}{of_head}",
                err=True,
            )
            _say_untracked(snap)
            with job_ssh.one_time_key(rig.runner, tools=rig.tools) as key:
                body = _submit_body(
                    key.public_key,
                    command=command,
                    gpus=gpus,
                    minutes=minutes,
                    label=label,
                    commit=snap.commit,
                )
                job = rig.client.job_submit(body)
                echo(f"job {job.id}: requested, waiting for a node", err=True)
                try:
                    grant = _await_grant(
                        rig,
                        lambda wait: rig.client.job_grant(job.id, wait=wait),
                    )
                    gate = job_ssh.Gate(key, grant, runner=rig.runner, tools=rig.tools)
                    info = _upload(gate, snap, Path(scratch), grant)
                    results = results_dir(rig.client.base_url, job.id, None)
                    results.mkdir(parents=True, exist_ok=True)
                    _ = shutil.copyfile(info.path, results / "src.bundle")
                    _await_queued(rig, job.id, snap, info)
                except ClientError:
                    echo(
                        f"{PROGRAM}: job {job.id} was submitted; "
                        f"'{PROGRAM} cancel {job.id}' releases it",
                        err=True,
                    )
                    raise
        echo(job.id)
        if follow:
            cls.run_logs(rig, job.id, follow=True)

    @classmethod
    def run_logs(cls, rig: Rig, job: int, *, follow: bool) -> None:
        """Print a job's log; with ``follow``, until the job ends.

        Args:
          rig: The server, ssh and the clock.
          job: The job id.
          follow: Keep streaming until the job ends.

        """
        with _read_gate(rig, job) as gate:
            try:
                gate.logs(_TextSink(out_stream()), follow=follow)
            except KeyboardInterrupt:
                raise SystemExit(130) from None

    @classmethod
    def run_results(cls, rig: Rig, job: int, *, to: str | None) -> None:
        """Download a job's output into a local directory.

        Args:
          rig: The server, ssh and the clock.
          job: The job id.
          to: The directory to fill; ``None`` uses the per-server data directory.

        """
        results = results_dir(rig.client.base_url, job, to)
        view = rig.client.job_view(job)
        with _read_gate(rig, job) as gate:
            done = job_fetch.fetch(gate, gate.manifest(), results=results)
        (results / "job.json").write_text(f"{view.model_dump_json(indent=2)}\n")
        echo(
            f"results: {results / 'out'} "
            f"({done.downloaded} downloaded, {done.skipped} already held)",
        )


def results_dir(url: str, job: int, to: str | None) -> Path:
    """Return where a job's results go, refusing a ``--to`` inside the repository.

    Args:
      url: The server's URL, which names the default directory.
      job: The job id.
      to: The directory the caller asked for; ``None`` uses the per-server data
        directory.

    Returns:
      directory: ``to`` made absolute, or the job's directory under the data folder.

    Raises:
      ClientError: ``url`` names no host, or ``to`` is inside a git worktree.

    """
    if to is None:
        # Keyed by server: job 7 of another server is another job. The ``trax`` path
        # segment is the name this command had before it moved: results already on
        # disk resume from the same directory.
        server = server_dir(url)
        return data_dir() / "rekursiv-ai" / "trax" / "job" / server / str(job)
    target = (cwd() / Path(to).expanduser()).resolve()
    top = job_source.worktree_top(cwd())
    if top is not None and target.is_relative_to(top):
        raise ClientError(
            f"--to {to} is inside the repository at {top}; "
            "results are not kept in a worktree",
        )
    return target


class _Flags(Protocol):
    """Parsed ``knowop job`` flags; each verb sets only its own."""

    action: str
    command: tuple[str, ...]
    machine: str
    memory: str | None
    gpus: int
    minutes: int
    label: str
    follow: bool
    include_untracked: bool
    job: int | None
    to: str | None
    name: str
    op: str


# A colour (SGR), or a control character other than tab, newline and carriage return.
# ESC (0x1b) and the C1 range (0x80-0x9f, which holds a one-character CSI) are in it.
_TERMINAL_COMMAND: Final = re.compile(
    r"(\x1b\[[0-9;]*m)|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]",
)


class _TextSink(io.BytesIO):
    r"""A byte sink that writes UTF-8 text to a stream, whole characters at a time.

    The bytes are the job's own, so they may colour the text but never command the
    terminal: every other escape or control character is written as ``\xNN``. A
    colour cut in two by a read is shown that way too.
    """

    def __init__(self, stream: TextIO) -> None:
        super().__init__()
        self._stream = stream
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    @override
    def write(self, buffer: ReadableBuffer, /) -> int:
        data = bytes(buffer)
        text = _TERMINAL_COMMAND.sub(_shown, self._decoder.decode(data))
        _ = self._stream.write(text)
        self._stream.flush()
        return len(data)


def _shown(found: re.Match[str]) -> str:
    r"""Keep a colour; spell any other control character as ``\xNN``."""
    text = found.group(0)
    return text if found.start(1) != -1 else f"\\x{ord(text):02x}"


def _say_untracked(snap: Snapshot) -> None:
    """Name the untracked files the snapshot took, so a stray one is seen before it goes."""
    if not snap.untracked:
        return
    shown = ", ".join(
        job_source.plain_name(n) for n in snap.untracked[:_UNTRACKED_SHOWN]
    )
    more = len(snap.untracked) - _UNTRACKED_SHOWN
    names = f"{shown} and {more} more" if more > 0 else shown
    echo(f"untracked files sent ({len(snap.untracked)}): {names}", err=True)


def _split(rest: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split ``rest`` at its first ``--`` into the flags and the job's command."""
    if "--" not in rest:
        return list(rest), []
    at = rest.index("--")
    return list(rest[:at]), list(rest[at + 1 :])


# A last word of ``help`` is the usual trailing form, except as the value of a flag:
# ``--label help`` names a job.
def _asks_for_help(head: Sequence[str]) -> bool:
    """Whether the tokens before ``--`` ask for help rather than an action."""
    if not head or head[0] == "help" or {"--help", "-h"} & set(head):
        return True
    return head[-1] == "help" and head[-2] not in {"--label", "--to"}


def _rig(client_factory: Callable[[], Client]) -> Rig:
    """Build the real rig; a missing ssh fails before any job is asked for."""
    tools = job_ssh.Tools.find()
    return Rig(
        client=client_factory(),
        runner=job_ssh.SubprocessRunner(),
        tools=tools,
        sleep=time.sleep,
        clock=time.monotonic,
    )


def _submit_body(
    public_key: str,
    *,
    command: Sequence[str],
    gpus: int,
    minutes: int,
    label: str,
    commit: str,
) -> wire_job.SubmitBody:
    """Build the submit request, or raise an error naming each bad field."""
    try:
        return wire_job.SubmitBody.model_validate(
            {
                "public_key": public_key,
                "gpus": gpus,
                "minutes": minutes,
                "label": label,
                "argv": list(command),
                "commit": commit,
            },
        )
    except pydantic.ValidationError as err:
        problems = "; ".join(
            f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in err.errors()
        )
        raise ClientError(f"invalid job: {problems}") from None


# Exits 1 with the server's reason when it refuses. A server that answers before the
# wait it was given is asked again a second later, not in a tight loop.
def _await_grant(rig: Rig, poll: Callable[[float], GrantStatus]) -> Grant:
    """Long-poll for a certificate, for up to ten minutes."""
    deadline = rig.clock() + 10 * 60
    while True:
        asked = rig.clock()
        status = poll(25)
        if status.state == "refused":
            echo(f"refused: {status.reason or 'no reason given'}", err=True)
            raise SystemExit(1)
        if status.state == "ready" and status.grant is not None:
            return status.grant
        if rig.clock() >= deadline:
            raise ClientError("no certificate was granted within 10 minutes")
        if rig.clock() - asked < 1:
            rig.sleep(1)


@contextmanager
def _read_gate(rig: Rig, job: int) -> Generator[Gate]:
    """Get a read certificate for ``job`` and yield a gate that uses it."""
    with job_ssh.one_time_key(rig.runner, tools=rig.tools) as key:
        ticket = rig.client.job_read_grant(
            job,
            wire_job.ReadGrantBody(public_key=key.public_key),
        )
        grant = _await_grant(
            rig,
            lambda wait: rig.client.job_grant_serial(job, ticket.serial, wait=wait),
        )
        yield job_ssh.Gate(key, grant, runner=rig.runner, tools=rig.tools)


def _upload(
    gate: Gate,
    snap: Snapshot,
    scratch: Path,
    grant: Grant,
) -> BundleInfo:
    """Send the snapshot as a thin bundle, or a full one if the node rejects it."""
    if snap.shallow:
        echo(
            f"shallow clone: sending a full snapshot of HEAD {snap.head[:12]} "
            "as its own commit",
            err=True,
        )
    info = _bundle(snap, gate.bases(), scratch / "thin.bundle", grant)
    try:
        reply = gate.upload(info.path, sha256=info.sha256, commit=snap.commit)
    except job_ssh.GateError as err:
        if err.exit_code != 3 or info.base is None:
            raise
        echo("the node rejected the thin bundle; sending a full bundle", err=True)
        info = _bundle(snap, [], scratch / "full.bundle", grant)
        reply = gate.upload(info.path, sha256=info.sha256, commit=snap.commit)
    if (reply.commit, reply.bundle_sha256, reply.bytes) != (
        snap.commit,
        info.sha256,
        info.bytes,
    ):
        raise ClientError(
            "the node stored something other than what was sent; nothing was started",
        )
    return info


def _bundle(
    snap: Snapshot,
    bases: Sequence[str],
    out: Path,
    grant: Grant,
) -> BundleInfo:
    """Cut a bundle on the bases and refuse one over the node's limit."""
    info = job_source.bundle(snap, bases, out)
    how = "full" if info.base is None else f"on base {info.base[:12]}"
    echo(f"bundle: {info.bytes} bytes, {how}", err=True)
    if info.bytes > grant.max_bundle_bytes:
        raise ClientError(
            f"the bundle is {info.bytes} bytes; this node takes at most "
            f"{grant.max_bundle_bytes}",
        )
    return info


def _await_queued(rig: Rig, job: int, snap: Snapshot, info: BundleInfo) -> None:
    """Tell the server the code is up, then wait until the job is queued or later."""
    view = rig.client.job_start(
        job,
        wire_job.StartBody(
            commit=snap.commit,
            bundle_sha256=info.sha256,
            bundle_bytes=info.bytes,
        ),
    )
    deadline = rig.clock() + 5 * 60
    while view.state in {"requested", "placed", "granted", "starting"}:
        if rig.clock() >= deadline:
            raise ClientError(f"job {job} was not queued within 5 minutes")
        rig.sleep(2)
        view = rig.client.job_view(job)
    if view.state in wire_job.TERMINAL_STATES and view.state != "completed":
        echo(f"job {job} {view.state}: {view.reason or 'no reason given'}", err=True)
        raise SystemExit(1)
