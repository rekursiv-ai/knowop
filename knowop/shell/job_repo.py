"""Fetch a job's commit into a fresh worktree, by git alone, with the host's own access.

Each remote has one bare mirror under the root; a job fetches its branch into it, checks
that its commit is there, and gets a detached worktree of that commit. The host's user
fetches with their own credentials, as by hand; no code comes through the server.

The remote is the server's word, so git is held to it: only the protocols this host
allows (``https`` and ``ssh``) are spoken, which ``protocol.allow`` enforces in git
itself, so an ``ext::`` remote can never run a command and a ``file`` one can never copy
the host's own repositories. The remote goes after ``--``, no hook runs, and nothing
waits on a prompt.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Final

import fcntl
import hashlib
import os
import selectors
import shutil
import signal
import stat
import subprocess
import time
import uuid

from trackinizer.wire.wire_machine_job import remote_protocol

from knowop.jobs.job_source import clean_env
from knowop.jobs.job_ssh import require_tool
from knowop.shell.job_limits import run, scope_argv


if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    import threading


__all__ = ["FetchError", "Repos"]

_HOST_PROTOCOLS: Final = frozenset({"https", "ssh"})
MIRRORS_MAX_BYTES: Final = 4 * 2**30
PREPARATION_MEMORY_MB: Final = 256
PREPARATION_OUTPUT_BYTES: Final = 64 * 2**10


class FetchError(Exception):
    """The job's commit could not be fetched or checked out; the message says why."""


class Repos:
    """The host's mirrors of the remotes its jobs name.

    Args:
      root: The folder that holds one bare mirror per remote.
      protocols: The protocols git may speak; ``https`` and ``ssh`` unless a test
        serves a repository from a folder.
      timeout_sec: The longest one git command may take.

    """

    def __init__(
        self,
        root: Path,
        *,
        protocols: frozenset[str] = _HOST_PROTOCOLS,
        timeout_sec: float = 600.0,
    ) -> None:
        self._root = root
        self._protocols = protocols
        self._timeout_sec = timeout_sec
        self._cancel: threading.Event | None = None

    def checkout(
        self,
        *,
        remote: str,
        ref: str,
        commit: str,
        dest: Path,
        cancel: threading.Event | None = None,
    ) -> None:
        """Make ``dest`` a fresh worktree of ``commit``, fetched from ``remote``.

        Args:
          remote: The git remote.
          ref: The branch, ``refs/heads/NAME``, that holds the commit.
          commit: The commit to check out.
          dest: A folder that does not exist yet.
          cancel: Set to stop preparation, including a wait for another fetch.

        Raises:
          FetchError: The remote is not one this host fetches from, git failed or
            ran out of time, or the commit is not on the branch.

        """
        if remote_protocol(remote) not in self._protocols:
            allowed = " or ".join(sorted(self._protocols))
            raise FetchError(f"this machine fetches only over {allowed}: {remote}")
        mirror = self._mirror(remote)
        with self._exclusive(cancel):
            stage = self._root / ".prepare"
            shutil.rmtree(stage, ignore_errors=True)
            self._check_storage()
            try:
                self._prepare(stage, remote=remote, ref=ref, commit=commit)
                self._check_storage()
                if not mirror.exists():
                    stage.rename(mirror)
                else:
                    _promote(stage / "objects", mirror / "objects")
                _ = self._git(
                    mirror,
                    "worktree",
                    "add",
                    "--detach",
                    "--quiet",
                    str(dest),
                    commit,
                )
            finally:
                shutil.rmtree(stage, ignore_errors=True)

    def _prepare(self, stage: Path, *, remote: str, ref: str, commit: str) -> None:
        """Fetch into disposable storage; never damage objects of running worktrees."""
        _ = self._git(None, "init", "--bare", "--quiet", str(stage))
        _ = self._git(
            stage,
            "fetch",
            "--no-tags",
            "--quiet",
            "--",
            remote,
            f"+{ref}",
        )
        if not self._has(stage, commit):
            _ = self._git(
                stage,
                "fetch",
                "--no-tags",
                "--quiet",
                "--",
                remote,
                commit,
            )
        if not self._has(stage, commit):
            raise FetchError(f"{commit[:12]} is not on {ref} at {remote}")

    def remove(self, *, remote: str, dest: Path) -> None:
        """Delete a job's worktree, and the mirror's record of it.

        Args:
          remote: The remote its mirror is of.
          dest: The worktree.

        """
        with self._exclusive(None):
            shutil.rmtree(dest, ignore_errors=True)
            mirror = self._mirror(remote)
            if (mirror / "HEAD").exists():
                _ = self._git(mirror, "worktree", "prune")

    def _mirror(self, remote: str) -> Path:
        """Return the bare mirror of ``remote``."""
        return self._root / f"{hashlib.sha256(remote.encode()).hexdigest()[:16]}.git"

    @contextmanager
    def _exclusive(self, cancel: threading.Event | None) -> Generator[None]:
        """Serialize preparation across hosts sharing the mirror directory."""
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self._root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    raise FetchError("preparation cancelled")
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.05)
            self._cancel = cancel
            try:
                yield
            finally:
                self._cancel = None
        finally:
            os.close(fd)

    def _check_storage(self) -> None:
        """Reject retained plus staging bytes above the mirror budget."""
        size = 0
        for path in self._root.rglob("*"):
            try:
                info = path.stat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(info.st_mode):
                size += info.st_size
        if size > MIRRORS_MAX_BYTES:
            raise FetchError("mirror storage exceeds its byte budget")

    def _has(self, mirror: Path, commit: str) -> bool:
        """Whether the mirror holds ``commit``."""
        done = self._run(mirror, "cat-file", "-e", f"{commit}^{{commit}}")
        return done.returncode == 0

    def _git(self, mirror: Path | None, *args: str) -> str:
        """Run git, or raise the first line of its complaint."""
        done = self._run(mirror, *args)
        if done.returncode:
            reason = (done.stderr.strip().splitlines() or ["no reason given"])[0]
            raise FetchError(f"git {args[0]} failed: {reason}")
        return done.stdout

    def _run(self, mirror: Path | None, *args: str) -> subprocess.CompletedProcess[str]:
        """Run git on ``mirror`` with this host's protocols, no hooks and no prompts."""
        allowed = [
            arg
            for protocol in sorted(self._protocols)
            for arg in ("-c", f"protocol.{protocol}.allow=always")
        ]
        where = [] if mirror is None else ["--git-dir", str(mirror)]
        unit = f"knowop-prepare-{uuid.uuid4().hex}"
        argv = scope_argv(
            unit=unit,
            memory_mb=PREPARATION_MEMORY_MB,
            runtime_sec=self._timeout_sec,
            argv=[
                require_tool("git"),
                "-c",
                "protocol.allow=never",
                *allowed,
                "-c",
                "core.hooksPath=/dev/null",
                *where,
                *args,
            ],
        )
        try:
            with subprocess.Popen(  # noqa: S603 -- fixed argv; the remote follows ``--``.
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={
                    **clean_env(),
                    "GIT_ALLOW_PROTOCOL": ":".join(sorted(self._protocols)),
                    "GIT_TERMINAL_PROMPT": "0",
                    "SSH_ASKPASS_REQUIRE": "never",
                },
                # With no terminal of its own, ssh cannot ask for a passphrase on
                # ``connect``'s, and fails at once instead of waiting for nobody.
                start_new_session=True,
            ) as child:
                try:
                    return self._collect(child)
                finally:
                    _ = run(
                        [
                            "systemctl",
                            "--user",
                            "kill",
                            "--signal=SIGKILL",
                            f"{unit}.scope",
                        ],
                    )
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGKILL)
        except subprocess.TimeoutExpired as error:
            raise FetchError(
                f"git {args[0]} took more than {self._timeout_sec:g} s",
            ) from error

    def _collect(
        self,
        child: subprocess.Popen[bytes],
    ) -> subprocess.CompletedProcess[str]:
        """Drain pipes within a fixed byte budget while checking cancellation and disk."""
        if child.stdout is None or child.stderr is None:
            raise ValueError("Expected preparation output pipes.")
        stdout, stderr = child.stdout.fileno(), child.stderr.fileno()
        output = {stdout: bytearray(), stderr: bytearray()}
        deadline = time.monotonic() + self._timeout_sec
        with selectors.DefaultSelector() as selector:
            selector.register(stdout, selectors.EVENT_READ)
            selector.register(stderr, selectors.EVENT_READ)
            while selector.get_map() or child.poll() is None:
                if self._cancel is not None and self._cancel.is_set():
                    raise FetchError("preparation cancelled")
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(child.args, self._timeout_sec)
                self._check_storage()
                for key, _ in selector.select(timeout=0.05):
                    chunk = os.read(key.fd, 4096)
                    if not chunk:
                        selector.unregister(key.fd)
                    else:
                        output[key.fd].extend(chunk)
                        if sum(map(len, output.values())) > PREPARATION_OUTPUT_BYTES:
                            raise FetchError(
                                "preparation output exceeds its byte budget",
                            )
        self._check_storage()
        return subprocess.CompletedProcess(
            child.args,
            child.wait(),
            output[stdout].decode(errors="replace"),
            output[stderr].decode(errors="replace"),
        )


def _promote(source: Path, target: Path) -> None:
    """Move verified immutable Git objects without duplicating retained bytes."""
    for path in source.rglob("*"):
        if path.is_file():
            dest = target / path.relative_to(source)
            dest.parent.mkdir(exist_ok=True)
            if not dest.exists():
                path.rename(dest)
