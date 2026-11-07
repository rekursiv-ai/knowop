from __future__ import annotations

from typing import TYPE_CHECKING, cast

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


_DARWIN = sys.platform == "darwin"


def _codesign(argv: Sequence[str]) -> tuple[int, str]:
    """Run ``argv``, as knowop's own runner does."""
    done = subprocess.run(  # noqa: S603 -- codesign, on a folder this test wrote.
        argv,
        capture_output=True,
        text=True,
        check=False,
    )
    return done.returncode, done.stdout + done.stderr


def _sign(argv: Sequence[str]) -> tuple[int, str]:
    """Sign with codesign on macOS; elsewhere, where there is none, sign nothing."""
    if _DARWIN:
        return _codesign(argv)
    assert argv[:2] == ["codesign", "--force"], argv
    return 0, ""


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build one app for this module: the copy and the signature take a while."""
    built = tmp_path_factory.mktemp("app") / "KnowOp.app"
    macos_app.install(built, _sign)
    return built


def test_the_app_is_known_as_knowop_and_built_for_this_python(app: Path) -> None:
    info = cast(
        dict[str, str | bool],
        plistlib.loads((app / "Contents" / "Info.plist").read_bytes()),
    )

    assert (info["CFBundleName"], info["CFBundleIdentifier"]) == (
        "KnowOp",
        "ai.rekursiv.knowop",
    )
    assert info["CFBundleShortVersionString"] == (
        f"{sys.version_info.major}.{sys.version_info.minor}"
    )
    assert info["LSUIElement"] is True


@pytest.mark.skipif(not _DARWIN, reason="macOS's codesign")
def test_macos_verifies_the_app(app: Path) -> None:
    assert _codesign(["codesign", "--verify", "--strict", str(app)]) == (0, "")


def test_codesign_signs_the_staged_app_as_knowop(tmp_path: Path) -> None:
    signed: list[list[str]] = []

    def record(argv: Sequence[str]) -> tuple[int, str]:
        signed.append(list(argv))
        return _sign(argv)

    macos_app.install(tmp_path / "KnowOp.app", record)

    assert signed == [
        [
            "codesign",
            "--force",
            "--sign",
            "-",
            "--identifier",
            "ai.rekursiv.knowop",
            str(tmp_path / ".staged-KnowOp.app"),
        ],
    ]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["KnowOp.app"]


def test_no_app_is_built_without_this_pythons_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A Python with no ``bin/pythonX.Y`` has nothing to copy; the launcher runs bare."""
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "gone"))

    macos_app.install(tmp_path / "KnowOp.app", _sign)

    assert not (tmp_path / "KnowOp.app").exists()
    assert "is not a file" in caplog.text


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

    macos_app.install(app, _sign)

    assert executable.stat().st_ino == before


def test_a_stale_app_is_replaced_whole(app: Path, tmp_path: Path) -> None:
    stale = tmp_path / "KnowOp.app"
    shutil.copytree(app, stale, symlinks=True)
    (stale / "Contents" / "Info.plist").write_bytes(b"old")
    (stale / "Contents" / "Resources" / "left-behind").write_text("old")

    macos_app.install(stale, _sign)

    assert not (stale / "Contents" / "Resources" / "left-behind").exists()
    assert _codesign(macos_app.probe(stale)) == (0, "")


def test_an_app_codesign_refused_is_not_installed(tmp_path: Path) -> None:
    def refuse(argv: Sequence[str]) -> tuple[int, str]:
        del argv
        return 1, "no"

    macos_app.install(tmp_path / "KnowOp.app", refuse)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(not _DARWIN, reason="macOS's codesign")
def test_macos_verifies_a_replaced_app(app: Path, tmp_path: Path) -> None:
    stale = tmp_path / "KnowOp.app"
    shutil.copytree(app, stale, symlinks=True)
    (stale / "Contents" / "Info.plist").write_bytes(b"old")

    macos_app.install(stale, _codesign)

    assert _codesign(["codesign", "--verify", "--strict", str(stale)]) == (0, "")


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
