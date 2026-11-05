"""Run knowop's launchd agents under KnowOp.app, so macOS asks in KnowOp's name.

macOS asks a person about their Desktop, Photos or Music in the name of the
*responsible* process: the one launchd started, whichever descendant touched them. A
connect agent's was a bare interpreter, so the dialogs said ``python3.14`` for what an
agent's shell did, and each new Python, a new hash to macOS, asked all over again.

The responsible process has to be the bundle's executable, and stay it: launchd follows
the process it started, and an ``exec`` of knowop's interpreter from there would make
that the interpreter again. So the executable is a copy of an interpreter that runs
:data:`_KEEP`, which starts the command it is given, passes on what launchd signals and
exits as it did. The command itself runs as it always has, under its own interpreter.
No compiler is needed: ``codesign``, which every Mac has, signs the copy ad hoc.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import hashlib
import logging
import plistlib
import shutil
import sys


if TYPE_CHECKING:
    from knowop.shell import job_limits


__all__ = ["command", "install", "probe"]


_logger: Final = logging.getLogger(__name__)

_IDENTIFIER: Final = "ai.rekursiv.knowop"
"""What macOS knows the app by; the settings it keeps are under this."""

_KEEP: Final = """\
import os
import signal
import sys

child = 0


def forward(number, _frame):
    if child:
        os.kill(child, number)


for number in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
    signal.signal(number, forward)
child = os.posix_spawn(sys.argv[1], sys.argv[1:], os.environ)
code = os.waitstatus_to_exitcode(os.waitpid(child, 0)[1])
sys.exit(128 - code if code < 0 else code)
"""
"""What the app runs: launchd restarts on an exit code, so it keeps the command's."""


def probe(app: Path) -> list[str]:
    """Return a command that succeeds when ``app`` can start.

    Args:
      app: The ``.app`` folder.

    Returns:
      argv: It fails for an app never installed, or whose Python has since gone.

    """
    return [str(app / "Contents" / "MacOS" / "KnowOp"), "-I", "-S", "-c", ""]


def command(app: Path) -> list[str]:
    """Return the command that runs what follows it under ``app``.

    Args:
      app: The ``.app`` folder.

    Returns:
      argv: To be followed by an absolute path and its arguments.

    """
    return [
        str(app / "Contents" / "MacOS" / "KnowOp"),
        "-I",
        "-S",
        str(app / "Contents" / "Resources" / "keep.py"),
    ]


# Rebuilt only for a new Python: the same bytes keep what the person answered macOS.
# Failure costs the name alone: the launcher runs knowop directly when the app won't.
def install(app: Path, run: job_limits.Run) -> None:
    """Build ``app`` from this Python, unless it is built already.

    Args:
      app: The ``.app`` folder to write.
      run: Runs ``codesign``.

    """
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    python = Path(sys.base_prefix) / "bin" / f"python{version}"
    if not python.is_file():
        _logger.warning("no KnowOp.app: %s is not a file", python)
        return
    build = hashlib.sha256(python.read_bytes() + _KEEP.encode()).hexdigest()
    info = plistlib.dumps(
        {
            "CFBundleDisplayName": "KnowOp",
            "CFBundleExecutable": "KnowOp",
            "CFBundleIdentifier": _IDENTIFIER,
            "CFBundleName": "KnowOp",
            "CFBundlePackageType": "APPL",
            "CFBundleShortVersionString": version,
            "KnowOpBuild": build,
            "LSUIElement": True,
        },
    )
    written = app / "Contents" / "Info.plist"
    if written.is_file() and written.read_bytes() == info:
        return
    staged = app.with_name(f".staged-{app.name}")
    shutil.rmtree(staged, ignore_errors=True)
    resources = staged / "Contents" / "Resources"
    executable = staged / "Contents" / "MacOS" / "KnowOp"
    resources.mkdir(parents=True)
    executable.parent.mkdir()
    # ``copyfile``: extended attributes would make ``codesign`` refuse the copy.
    shutil.copyfile(python, executable)
    executable.chmod(0o755)
    (staged / "Contents" / "Info.plist").write_bytes(info)
    (resources / "keep.py").write_text(_KEEP)
    # A symlink: Python reads the file beside the folder of its executable, where
    # ``codesign`` refuses a plain file, and this one points its copy at the library.
    (resources / "pyvenv.cfg").write_text(f"home = {python.parent}\n")
    (staged / "Contents" / "pyvenv.cfg").symlink_to("Resources/pyvenv.cfg")
    code, out = run(
        [
            "codesign",
            "--force",
            "--sign",
            "-",
            "--identifier",
            _IDENTIFIER,
            str(staged),
        ],
    )
    if code != 0:
        _logger.warning("no KnowOp.app: codesign said %s", out.strip())
        shutil.rmtree(staged, ignore_errors=True)
        return
    shutil.rmtree(app, ignore_errors=True)
    staged.replace(app)
