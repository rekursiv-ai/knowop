"""Run ``knowop connect`` as a user service: launchd's on macOS, systemd's on Linux.

``knowop connect`` at a terminal installs one service per machine name and server,
which serves the machine in the background and starts again at login, as ``code
tunnel service install`` and ``cloudflared service install`` do. Each is a definition
file, ``ai.rekursiv.knowop.connect.NAME.TAG.plist`` in ``~/Library/LaunchAgents`` or
``knowop-connect-NAME-TAG.service`` in systemd's user folder, where ``TAG`` is
:func:`server_tag` of the server. It runs ``knowop connect`` with the flags the person
gave: this interpreter's ``-m knowop``, which ``uv tool upgrade knowop`` updates in
place, or ``uvx knowop@VERSION`` when this knowop runs from uv's cache.

The definition holds no secret: the host reads its saved machine credential, readable
by this user only, each time it starts. Its environment is ``PATH``, ``LANG`` and
``LC_*`` as the installing shell had them, so the CLIs a terminal runs (``claude``,
``git``, ``uv``) resolve as they did there (launchd's own ``PATH`` is
``/usr/bin:/bin:/usr/sbin:/sbin``), and the variables the per-user folders are read
from, so the host finds the login and the credential the shell would.

Both managers append the host's output to one log,
``state_dir()/rekursiv-ai/knowop/connect/NAME-TAG.log``, which the host keeps under
5 MiB (:func:`rotate_log`). A host that exits 0 is not started again, and one that
fails is, at most every 10 s (launchd) or 5 times in 5 minutes (systemd).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, TypedDict

import getpass
import hashlib
import os
import plistlib
import re
import shlex
import shutil
import sys
import time

from trackinizer.client.errors import ClientError
from trackinizer.lib.userdirs import (
    base_dir_variables,
    config_dir,
    data_dir,
    runtime_dir_variables,
    state_dir,
)
from treekle import from_plain

from knowop import menubar
from knowop.shell import job_limits


if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = [
    "Connect",
    "Services",
    "here",
    "knowop_version",
    "rotate_log",
    "server_tag",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Connect:
    """One machine's ``knowop connect``, as a service of this computer.

    Attributes:
      name: The machine's name.
      server: The server's URL, as knowop's login names it.
      flags: The connect flags it runs with, such as ``--sandbox``.
      disabled: Whether the person turned it off (macOS Login Items, ``systemctl
        --user disable``); read from the manager, never written.

    """

    name: str
    server: str
    flags: tuple[str, ...] = ()
    disabled: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class Services:
    """The ``knowop connect`` services of this computer, through its service manager.

    Attributes:
      manager: launchd on macOS; systemd's user manager on Linux.
      folder: Where that manager reads this user's service definitions.
      run: Runs ``launchctl``, ``systemctl`` and ``loginctl``.
      wait: Waits the given seconds between tries of a ``bootstrap``.
      menubar_problem: Says why the menu bar cannot start in this knowop's Python;
        ``None`` when it can. :func:`here` asks AppKit; others show no menu bar.
      set_icon: Gives a file the knowop icon, which Login Items shows for an agent
        that runs it; called only where the menu bar can start.

    """

    manager: Literal["launchd", "systemd"]
    folder: Path
    run: job_limits.Run
    wait: Callable[[float], None] = time.sleep
    menubar_problem: Callable[[], str | None] = lambda: (
        "AppKit not asked: here() asks it"
    )
    set_icon: Callable[[Path], None] = menubar.set_icon

    def start(self, connect: Connect) -> None:
        """Install ``connect``'s service and start it, or restart it when it runs.

        Args:
          connect: The machine, server and flags to serve.

        Raises:
          ClientError: A manager command failed.

        """
        path = self._path(connect)
        log = _log(connect)
        unchanged = self._write(path, self._definition(connect), log=log)
        if self.manager == "systemd":
            self._call("systemctl", "--user", "daemon-reload")
            self._call("systemctl", "--user", "enable", path.name)
            # Best effort: clears a crash loop's start limit, which refuses a restart.
            self.run(["systemctl", "--user", "reset-failed", path.name])
            self._call("systemctl", "--user", "restart", path.name)
            return
        fresh = self._launcher("knowop")
        self._load(path, unchanged=unchanged and fresh, again=("-k",))

    # An agent whose Python cannot import AppKit fails every 10 s, in a log no one
    # reads: this one is never installed, and the caller says why.
    def show_menu_bar(self) -> str | None:
        """Start the menu bar agent, one beside every connect agent, unless it runs.

        Returns:
          problem: Why the menu bar cannot start in this knowop's Python, which then
            removes any agent of it; ``None`` once it runs, and under systemd, which
            has none.

        Raises:
          ClientError: A manager command failed.

        """
        path = self.folder / f"{_MENUBAR}.plist"
        if self.manager != "launchd":
            return None
        if (problem := self.menubar_problem()) is not None:
            if path.is_file():
                self._unload(path)
            return problem
        log = state_dir() / "rekursiv-ai" / "knowop" / "menubar.log"
        launcher = (
            data_dir() / "rekursiv-ai" / "knowop" / "launchers" / "knowop menu bar"
        )
        agent = _agent(path.stem, [str(launcher), "menubar"], log)
        fresh = self._launcher(launcher.name)
        # Started again only when it ended: Quit, or a crash past its restarts.
        unchanged = self._write(path, agent, log=log)
        self._load(path, unchanged=unchanged and fresh, again=())
        return None

    def menu_bar(self) -> str | None:
        """Return how the menu bar agent is, as ``knowop status`` shows it.

        Returns:
          state: ``running``; stopped, as Quit leaves it; or failing, with its exit
            code and the last line of its log. ``None`` where none is installed.

        """
        if not (self.folder / f"{_MENUBAR}.plist").is_file():
            return None
        _, out = self.run(["launchctl", "print", f"gui/{os.getuid()}/{_MENUBAR}"])
        if re.search(r"^\s*state = running$", out, re.MULTILINE):
            return "running"
        code = re.search(r"last exit code = (\d+)", out)
        if code is None or code[1] == "0":
            return "stopped; knowop connect starts it"
        log = state_dir() / "rekursiv-ai" / "knowop" / "menubar.log"
        return f"failing, exit {code[1]}: {_last_line(log)}"

    def remove(self, name: str, server: str) -> bool:
        """Stop the service of ``name`` on ``server`` and delete its definition.

        The menu bar agent, and the launchers, go with the last connect agent here.

        Args:
          name: The machine's name.
          server: The server's URL.

        Returns:
          removed: Whether this computer had that service.

        Raises:
          ClientError: A manager command failed; the definition then stays.

        """
        path = self._path(Connect(name=name, server=server))
        if not path.is_file():
            return False
        if self.manager == "systemd":
            self._call("systemctl", "--user", "disable", "--now", path.name)
            path.unlink()
            self._call("systemctl", "--user", "daemon-reload")
            return True
        self._unload(path)
        if not any(self.folder.glob(_CONNECT_PLISTS)):
            if (agent := self.folder / f"{_MENUBAR}.plist").is_file():
                self._unload(agent)
            for each in ("knowop", "knowop menu bar"):
                launcher = data_dir() / "rekursiv-ai" / "knowop" / "launchers" / each
                launcher.unlink(missing_ok=True)
        return True

    def every(self) -> list[Connect]:
        """Return each connect service installed here, by definition file name.

        Returns:
          services: Each one's machine, server and flags, and whether the person
            turned it off.

        """
        launchd = self.manager == "launchd"
        paths = sorted(
            self.folder.glob(
                _CONNECT_PLISTS if launchd else "knowop-connect-*.service",
            ),
        )
        if not paths:
            return []
        disabled = self._disabled()
        found: list[Connect] = []
        for path in paths:
            if launchd:
                plist = from_plain(plistlib.loads(path.read_bytes()), _Plist)
                argv = plist["ProgramArguments"]
            else:
                argv = _exec_start(path.read_text())
            # As ``_definition`` writes it: SERVER FLAGS --name NAME --log-file LOG.
            rest = argv[argv.index("connect") + 1 :]
            found.append(
                Connect(
                    name=rest[-3],
                    server=rest[0],
                    flags=tuple(rest[1:-4]),
                    disabled=(path.stem if launchd else path.name) in disabled,
                ),
            )
        return found

    def unchanged(self, connect: Connect) -> bool:
        """Return whether ``connect``'s service is installed as :meth:`start` would.

        Args:
          connect: The machine, server and flags it serves.

        Returns:
          unchanged: Whether its definition, and under launchd its launcher, hold
            what :meth:`start` would write.

        """
        path = self._path(connect)
        same = path.is_file() and path.read_text() == self._definition(connect)
        if self.manager == "systemd":
            return same
        launcher = data_dir() / "rekursiv-ai" / "knowop" / "launchers" / "knowop"
        return (
            same
            and launcher.is_file()
            and launcher.read_text() == _script(self._program())
        )

    def linger_off(self) -> bool:
        """Return whether systemd stops this user's services when they log out.

        Returns:
          off: Whether ``loginctl`` says lingering is off; never under launchd, which
            runs an agent while its user is logged in.

        """
        if self.manager == "launchd":
            return False
        code, out = self.run(
            ["loginctl", "show-user", getpass.getuser(), "-p", "Linger"],
        )
        return code == 0 and out.strip() == "Linger=no"

    def _path(self, connect: Connect) -> Path:
        """Return where ``connect``'s definition is installed."""
        tag = server_tag(connect.server)
        if self.manager == "launchd":
            return (
                self.folder / f"ai.rekursiv.knowop.connect.{connect.name}.{tag}.plist"
            )
        return self.folder / f"knowop-connect-{connect.name}-{tag}.service"

    # A knowop that uvx runs lives in uv's cache, which ``uv cache clean`` or ``prune``
    # deletes; uvx fetches the same version again at the next start. Any other install
    # (``uv tool install``, a venv, a checkout) lasts, and its interpreter runs it.
    def _program(self) -> list[str]:
        """Return the command that runs this knowop, up to its verb."""
        code, cache = self.run(["uv", "cache", "dir"])
        uvx = shutil.which("uvx")
        if (
            code == 0
            and cache.strip()
            and uvx is not None
            and Path(sys.prefix).resolve().is_relative_to(Path(cache.strip()).resolve())
        ):
            return [uvx, f"knowop@{knowop_version()}"]
        # ``knowop`` once exported; the package that holds this module, in any form.
        return [sys.executable, "-m", __name__.rpartition(".")[0]]

    def _definition(self, connect: Connect) -> str:
        """Return the definition file that serves ``connect``."""
        log = _log(connect)
        launcher = data_dir() / "rekursiv-ai" / "knowop" / "launchers" / "knowop"
        argv = [
            *([str(launcher)] if self.manager == "launchd" else self._program()),
            "connect",
            connect.server,
            *connect.flags,
            "--name",
            connect.name,
            "--log-file",
            str(log),
        ]
        if self.manager == "launchd":
            return _agent(self._path(connect).stem, argv, log)
        env = _environment()
        # No sandboxing directive: each would bind every shell and agent the host
        # starts, and NoNewPrivileges survives execve, so sudo would fail in them.
        server = _unit_text(connect.server)
        return "\n".join(
            [
                "[Unit]",
                f"Description=knowop connect {connect.name} to {server}",
                "StartLimitIntervalSec=300",
                "StartLimitBurst=5",
                "",
                "[Service]",
                # ``$`` would expand a variable in ExecStart, though not in Environment.
                "ExecStart=" + " ".join(_unit_word(a).replace("$", "$$") for a in argv),
                *(f"Environment={_unit_word(f'{k}={v}')}" for k, v in env.items()),
                "Restart=on-failure",
                "RestartSec=5",
                # The host ends as Ctrl-C does on the SIGTERM that stops it: 130.
                "SuccessExitStatus=130",
                "UMask=0077",
                f"StandardOutput=append:{_unit_text(str(log))}",
                f"StandardError=append:{_unit_text(str(log))}",
                "",
                "[Install]",
                "WantedBy=default.target",
                "",
            ],
        )

    # The log is made, readable by this user only, before the manager would make it.
    def _write(self, path: Path, definition: str, *, log: Path) -> bool:
        """Install ``definition`` at ``path`` and make its ``log``; say if it was so."""
        log.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.close(os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600))
        return _put(path, definition, mode=0o644)

    # Login Items names a legacy agent by its program's file name, and shows that file's
    # icon: each agent runs knowop through a script of its own, named for what it is.
    def _launcher(self, name: str) -> bool:
        """Install the launcher ``name``, which runs this knowop; say if it was so."""
        path = data_dir() / "rekursiv-ai" / "knowop" / "launchers" / name
        unchanged = _put(path, _script(self._program()), mode=0o755)
        # Set again at each write, which drops it: it is the file's extended attributes.
        if self.menubar_problem() is None:
            self.set_icon(path)
        return unchanged

    def _load(self, path: Path, *, unchanged: bool, again: tuple[str, ...]) -> None:
        """Load ``path``'s agent; one loaded as written is kickstarted with ``again``."""
        domain = f"gui/{os.getuid()}"
        target = f"{domain}/{path.stem}"
        # Clears a record of the person turning it off, which refuses a bootstrap.
        self._call("launchctl", "enable", target)
        loaded = self.run(["launchctl", "print", target])[0] == 0
        if loaded and unchanged:
            self._call("launchctl", "kickstart", *again, target)
            return
        # A loaded agent keeps its old definition until it is booted out.
        if loaded:
            self._call("launchctl", "bootout", target)
        self._bootstrap(domain, path)

    def _unload(self, path: Path) -> None:
        """Stop ``path``'s agent where launchd has it, and delete its definition."""
        target = f"gui/{os.getuid()}/{path.stem}"
        if self.run(["launchctl", "print", target])[0] == 0:
            self._call("launchctl", "bootout", target)
        path.unlink()

    def _disabled(self) -> set[str]:
        """Return the labels (launchd) or units (systemd) the person turned off."""
        if self.manager == "launchd":
            _, out = self.run(["launchctl", "print-disabled", f"gui/{os.getuid()}"])
            # ``"LABEL" => disabled``; macOS before 14 says ``true``.
            return set(re.findall(r'"([^"]+)" => (?:disabled|true)\b', out))
        _, out = self.run(
            [
                "systemctl",
                "--user",
                "list-unit-files",
                "--no-legend",
                "knowop-connect-*",
            ],
        )
        return {
            words[0]
            for words in (line.split() for line in out.splitlines())
            if words[1:2] == ["disabled"]
        }

    # ``bootout`` returns before launchd lets go of the label, and a ``bootstrap`` at once
    # fails with 5 (an I/O error) until it has: the service is then left unloaded.
    def _bootstrap(self, domain: str, path: Path) -> None:
        """Load ``path`` into ``domain``, again for up to 5 s while launchd answers 5."""
        argv = ("launchctl", "bootstrap", domain, str(path))
        code, out = self.run(argv)
        for _ in range(19):
            if code != 5:
                break
            self.wait(0.25)
            code, out = self.run(argv)
        _refuse(argv, code, out)

    def _call(self, *argv: str) -> None:
        """Run ``argv``, or refuse naming it when it fails."""
        _refuse(argv, *self.run(argv))


def here(
    run: job_limits.Run = job_limits.run,
    *,
    logind: Path = Path("/run/user"),
) -> Services | None:
    """Return this computer's user service manager, or ``None`` where it has none.

    Args:
      run: Runs ``systemctl`` here, and the managers' commands later.
      logind: Where logind keeps each user's runtime folder.

    Returns:
      services: launchd's on macOS; systemd's on Linux where ``systemctl --user``
        reaches its user manager, which a container or a session without one does
        not; else ``None``.

    """
    if sys.platform == "darwin":
        return Services(
            manager="launchd",
            folder=Path.home() / "Library" / "LaunchAgents",  # noqa: TID251 -- vendor fixed path, not ours (AGENTS.md rule 3)  # house-ignore[xdg-literal] -- launchd reads a user's agents from this fixed folder (AGENTS.md rule 3).
            run=run,
            menubar_problem=menubar.problem,
            set_icon=menubar.set_icon,
        )
    # A login without PAM (sshd's ``UsePAM no``) sets no runtime folder, and
    # ``systemctl --user`` then finds no manager, though linger keeps one running.
    missing = {
        name: value
        for name, value in runtime_dir_variables(logind=logind).items()
        if os.environ.get(name) != value
    }
    if missing:
        run = partial(_with_variables, run, missing)
    if run(["systemctl", "--user", "show-environment"])[0] != 0:
        return None
    return Services(manager="systemd", folder=_unit_folder(run), run=run)


def server_tag(server: str) -> str:
    """Return the 8 hex digits that tell a server's services and logs from another's."""
    return hashlib.sha256(server.rstrip("/").encode()).hexdigest()[:8]


def knowop_version() -> str:
    """Return the installed knowop's version; ``unknown`` from a source checkout."""
    try:
        return metadata.version("knowop")
    except metadata.PackageNotFoundError:
        return "unknown"


# Copied and emptied in place, never renamed: the service manager holds the log open
# for appending, so its writes go on at the new end; a renamed log would keep them.
def rotate_log(log: Path) -> None:
    """Copy ``log`` to ``LOG.1`` and empty it once it reaches 5 MiB; keep one copy.

    Args:
      log: The service's log, which the host writes to through its manager.

    """
    try:
        size = log.stat().st_size
    except FileNotFoundError:
        return
    if size < 5 * 2**20:
        return
    kept = os.open(
        log.with_name(f"{log.name}.1"),
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    with log.open("rb") as source, os.fdopen(kept, "wb") as copy:
        shutil.copyfileobj(source, copy)
    os.truncate(log, 0)


_MENUBAR: Final = "ai.rekursiv.knowop.menubar"
"""The menu bar agent's label: one for every connect agent of this computer."""

_CONNECT_PLISTS: Final = "ai.rekursiv.knowop.connect.*.plist"
"""The connect agents' definitions, by file name."""


# The end alone: a crash loop writes its traceback every 10 s.
def _last_line(log: Path) -> str:
    """Return the last line ``log`` holds; empty when it holds none."""
    try:
        with log.open("rb") as file:
            file.seek(max(file.seek(0, os.SEEK_END) - 4096, 0))
            lines = file.read().decode(errors="replace").strip().splitlines()
    except FileNotFoundError:
        return ""
    return lines[-1] if lines else ""


def _log(connect: Connect) -> Path:
    """Return the log ``connect``'s service writes."""
    tag = server_tag(connect.server)
    return (
        state_dir() / "rekursiv-ai" / "knowop" / "connect" / f"{connect.name}-{tag}.log"
    )


# No TRAX_*, KNOWOP_* or key: the host reads knowop's own login, in the folders the
# installing shell's userdirs variables name.
def _environment() -> dict[str, str]:
    """Return a service's environment: the installing shell's ``PATH`` and locale."""
    shell = {
        name: value
        for name, value in os.environ.items()
        if name in {"PATH", "LANG"}
        or (name.startswith("LC_") and not name.startswith("LC_TRAX"))
    }
    return dict(sorted({**shell, **base_dir_variables()}.items()))


def _script(program: list[str]) -> str:
    """Return a launcher: it runs ``program`` with the arguments it is given."""
    return (
        "#!/bin/sh\n"
        "# knowop connect writes it: launchd runs knowop through it, under its name.\n"
        f'exec {shlex.join(program)} "$@"\n'
    )


# By a temporary file renamed over it, so a symlink there is replaced, never followed.
def _put(path: Path, text: str, *, mode: int) -> bool:
    """Write ``text`` to ``path``, readable as ``mode`` says; say if it held it already."""
    unchanged = path.is_file() and path.read_text() == text
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.tmp")
    staged.unlink(missing_ok=True)
    fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    # The umask would narrow it; nothing in it is secret.
    os.fchmod(fd, mode)
    with os.fdopen(fd, "w") as file:
        file.write(text)
    staged.replace(path)
    return unchanged


def _agent(label: str, argv: list[str], log: Path) -> str:
    """Return a launchd agent that runs ``argv`` at login, and again on a failure."""
    return plistlib.dumps(
        {
            "Label": label,
            "ProgramArguments": argv,
            "EnvironmentVariables": _environment(),
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},
            "ThrottleInterval": 10,
            "StandardOutPath": str(log),
            "StandardErrorPath": str(log),
        },
    ).decode()


class _Plist(TypedDict):
    """The part of a launchd definition read back."""

    ProgramArguments: list[str]


def _exec_start(unit: str) -> list[str]:
    """Return the argv a systemd unit's ``ExecStart`` line names, unescaped."""
    line = next(line for line in unit.splitlines() if line.startswith("ExecStart="))
    return [
        word.replace("$$", "$").replace("%%", "%")
        for word in shlex.split(line.removeprefix("ExecStart="))
    ]


def _unit_word(text: str) -> str:
    """Quote ``text`` as one word of a unit's value; ``%`` would name a specifier."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{_unit_text(escaped)}"'


def _unit_text(text: str) -> str:
    """Escape ``%``, which a unit file reads as a specifier."""
    return text.replace("%", "%%")


def _refuse(argv: Sequence[str], code: int, out: str) -> None:
    """Refuse a manager command that failed, naming it and what it said."""
    if code:
        said = f": {out.strip()}" if out.strip() else ""
        raise ClientError(f"{' '.join(argv)} failed with exit {code}{said}")


def _with_variables(
    run: job_limits.Run,
    variables: dict[str, str],
    argv: Sequence[str],
) -> tuple[int, str]:
    """Run ``argv`` with ``variables`` set, through ``env``."""
    return run(
        ["env", *(f"{name}={value}" for name, value in variables.items()), *argv],
    )


# The manager reads units from its own config folder, which a shell that moved its own
# (XDG_CONFIG_HOME) does not move: a unit written by the shell's would never load.
# systemd.unit(5) lists that folder first of the search path's ``.../systemd/user``.
def _unit_folder(run: job_limits.Run) -> Path:
    """Return the folder of the user's own units that this systemd reads, as it says."""
    _, out = run(["systemctl", "--user", "show", "--property=UnitPath", "--value"])
    for each in out.split():
        if Path(each).parts[-2:] == ("systemd", "user"):
            return Path(each)
    return (
        config_dir()
        / "systemd"
        / "user"  # house-ignore[userdirs-namespace] -- systemd's own folder of a user's units, a vendor's namespace (AGENTS.md rule 1).
    )
