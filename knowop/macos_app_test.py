from __future__ import annotations

from typing import TYPE_CHECKING

import plistlib
import shutil
import signal
import subprocess
import sys

import pytest

from knowop import macos_app


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS's codesign")


def _codesign(argv: Sequence[str]) -> tuple[int, str]:
    """Run ``argv``, as knowop's own runner does."""
    done = subprocess.run(  # noqa: S603 -- codesign, on a folder this test wrote.
        argv,
        capture_output=True,
        text=True,
        check=False,
    )
    return done.returncode, done.stdout + done.stderr


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build one app for this module: the copy and the signature take a while."""
    built = tmp_path_factory.mktemp("app") / "KnowOp.app"
    macos_app.install(built, _codesign)
    return built


def test_macos_verifies_the_app_and_knows_it_as_knowop(app: Path) -> None:
    code, out = _codesign(["codesign", "--verify", "--strict", str(app)])
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())

    assert (code, out) == (0, "")
    assert (info["CFBundleName"], info["CFBundleIdentifier"]) == (
        "KnowOp",
        "ai.rekursiv.knowop",
    )


def test_a_built_app_starts(app: Path) -> None:
    assert _codesign(macos_app.probe(app)) == (0, "")


def test_the_app_runs_a_command_with_its_arguments_and_exit_code(app: Path) -> None:
    code, out = _codesign(
        [*macos_app.command(app), "/bin/sh", "-c", 'echo "[$1]"; exit 7', "-", "a b"],
    )

    assert (code, out) == (7, "[a b]\n")


def test_the_app_gives_a_signal_to_its_command_and_exits_as_it_did(app: Path) -> None:
    script = 'trap "exit 9" TERM; echo up; sleep 30 & wait'
    with subprocess.Popen(  # noqa: S603 -- the app just built, fixed words.
        [*macos_app.command(app), "/bin/sh", "-c", script],
        stdout=subprocess.PIPE,
        text=True,
    ) as kept:
        assert kept.stdout is not None
        assert kept.stdout.readline() == "up\n"
        kept.send_signal(signal.SIGTERM)

        assert kept.wait(timeout=5) == 9


def test_a_command_a_signal_killed_exits_as_a_shell_says(app: Path) -> None:
    code, _ = _codesign([*macos_app.command(app), "/bin/sh", "-c", "kill -9 $$"])

    assert code == 128 + signal.SIGKILL


def test_the_same_python_keeps_the_app_it_built(app: Path) -> None:
    executable = app / "Contents" / "MacOS" / "KnowOp"
    before = executable.stat().st_ino

    macos_app.install(app, _codesign)

    assert executable.stat().st_ino == before


def test_a_stale_app_is_replaced_whole(app: Path, tmp_path: Path) -> None:
    stale = tmp_path / "KnowOp.app"
    shutil.copytree(app, stale, symlinks=True)
    (stale / "Contents" / "Info.plist").write_bytes(b"old")
    (stale / "Contents" / "Resources" / "left-behind").write_text("old")

    macos_app.install(stale, _codesign)

    assert not (stale / "Contents" / "Resources" / "left-behind").exists()
    assert _codesign(["codesign", "--verify", "--strict", str(stale)]) == (0, "")


def test_an_app_codesign_refused_is_not_installed(tmp_path: Path) -> None:
    def refuse(argv: Sequence[str]) -> tuple[int, str]:
        del argv
        return 1, "no"

    macos_app.install(tmp_path / "KnowOp.app", refuse)

    assert list(tmp_path.iterdir()) == []


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
