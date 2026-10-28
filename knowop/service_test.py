"""The connect services, against a recording stand-in for launchctl and systemctl."""

from __future__ import annotations

from functools import partial
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, override

import os
import plistlib
import shutil
import stat
import sys

from trackinizer.client.errors import ClientError
from trackinizer.lib.userdirs import config_dir, state_dir

import pytest

from knowop import service
from knowop.service import Connect, Services


if TYPE_CHECKING:
    from collections.abc import Sequence


_SERVER: Final = "https://acme.knowop.ai"
_TAG: Final = service.server_tag(_SERVER)
_LABEL: Final = f"ai.rekursiv.knowop.connect.ada-mac.{_TAG}"
_UNIT: Final = f"knowop-connect-ada-mac-{_TAG}.service"
_ADA: Final = Connect(name="ada-mac", server=_SERVER, flags=("--sandbox",))
_KEY: Final = "trax_" + "k" * 43
_MENUBAR: Final = "ai.rekursiv.knowop.menubar"


class _Manager:
    """Stands in for launchctl, systemctl, loginctl and uv: records each command.

    Each answers ``(0, "")`` unless its subcommand, read past an ``env`` that sets
    variables first, is scripted; ``uv`` is not on ``PATH`` (127) unless ``cache``
    scripts ``uv cache dir``. Its commands, and those for the menu bar agent, are kept
    apart from the connect agents'.
    """

    def __init__(self, **answers: tuple[int, str]) -> None:
        self.answers = answers
        self.calls: list[list[str]] = []
        self.uv: list[list[str]] = []
        self.menubar: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> tuple[int, str]:
        if argv[0] == "uv":
            self.uv.append(list(argv))
            return self.answers.get("cache", (127, ""))
        (self.menubar if any(_MENUBAR in word for word in argv) else self.calls).append(
            list(argv),
        )
        if argv[0] == "env":
            argv = [each for each in argv[1:] if "=" not in each]
        verb = argv[2] if argv[1] == "--user" else argv[1]
        return self.answers.get(verb.replace("-", "_"), (0, ""))


@pytest.fixture(autouse=True)
def _shell(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Give the installing shell a known environment, secrets among it."""
    for name in list(os.environ):
        if name.startswith(("LC_", "TRAX_", "KNOWOP_", "TRACKINIZER_")):
            monkeypatch.delenv(name)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    for name, value in _env(tmp_path).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("LC_TRAX_SERVER", _SERVER)
    monkeypatch.setenv("TRACKINIZER_TOKEN", _KEY)
    monkeypatch.setenv("KNOWOP_DEBUG", "1")
    monkeypatch.setenv("EDITOR", "vim")


def _env(tmp_path: Path) -> dict[str, str]:
    """Return what a service's environment copies of the installing shell's."""
    return {
        "HOME": str(tmp_path / "home"),
        "LANG": "en_US.UTF-8",
        "LC_CTYPE": "UTF-8",
        "PATH": "/opt/homebrew/bin:/usr/bin:/bin",
        # Where the shell finds knowop's login and the machine's credential.
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }


def _argv(log: Path) -> list[str]:
    """Return what ``_ADA``'s service runs: this interpreter, its knowop package."""
    return [
        sys.executable,
        "-m",
        service.__name__.rpartition(".")[0],
        "connect",
        _SERVER,
        "--sandbox",
        "--name",
        "ada-mac",
        "--log-file",
        str(log),
    ]


def test_a_launchd_agent_runs_this_knowop_with_its_flags_and_the_shells_paths(
    tmp_path: Path,
) -> None:
    Services(manager="launchd", folder=tmp_path / "agents", run=_Manager()).start(_ADA)

    path = tmp_path / "agents" / f"{_LABEL}.plist"
    log = state_dir() / "rekursiv-ai" / "knowop" / "connect" / f"ada-mac-{_TAG}.log"
    assert plistlib.loads(path.read_bytes()) == {
        "Label": _LABEL,
        "ProgramArguments": _argv(log),
        "EnvironmentVariables": _env(tmp_path),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 10,
        "StandardOutPath": str(log),
        "StandardErrorPath": str(log),
    }


def test_a_systemd_unit_restarts_on_failure_within_limits_and_appends_to_the_log(
    tmp_path: Path,
) -> None:
    Services(manager="systemd", folder=tmp_path / "agents", run=_Manager()).start(_ADA)

    unit = (tmp_path / "agents" / _UNIT).read_text().splitlines()
    log = state_dir() / "rekursiv-ai" / "knowop" / "connect" / f"ada-mac-{_TAG}.log"
    assert unit[:6] == [
        "[Unit]",
        f"Description=knowop connect ada-mac to {_SERVER}",
        "StartLimitIntervalSec=300",
        "StartLimitBurst=5",
        "",
        "[Service]",
    ]
    assert unit[7:] == [
        *(f'Environment="{name}={value}"' for name, value in _env(tmp_path).items()),
        "Restart=on-failure",
        "RestartSec=5",
        "SuccessExitStatus=130",
        "UMask=0077",
        f"StandardOutput=append:{log}",
        f"StandardError=append:{log}",
        "",
        "[Install]",
        "WantedBy=default.target",
    ]
    assert unit[6] == "ExecStart=" + " ".join(f'"{word}"' for word in _argv(log))
    assert Services(
        manager="systemd",
        folder=tmp_path / "agents",
        run=_Manager(),
    ).every() == [_ADA]


def test_a_shell_without_a_folder_variable_gives_the_service_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The host then reads the platform's default folder, as the shell did."""
    monkeypatch.delenv("XDG_CONFIG_HOME")
    monkeypatch.setenv("XDG_CACHE_HOME", "")

    Services(manager="launchd", folder=tmp_path / "agents", run=_Manager()).start(_ADA)

    path = tmp_path / "agents" / f"{_LABEL}.plist"
    env = plistlib.loads(path.read_bytes())["EnvironmentVariables"]
    assert {k: v for k, v in _env(tmp_path).items() if k in env} == env
    assert {"XDG_CONFIG_HOME", "XDG_CACHE_HOME"} & set(env) == set()


def _uvx_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Manager:
    """Make this knowop 0.1.6 run as uvx runs it, from uv's cache; return the runner."""
    cache = tmp_path / "uv-cache"
    monkeypatch.setattr(sys, "prefix", str(cache / "archive-v0" / "Xk3q"))
    monkeypatch.setattr(shutil, "which", _on_homebrews_path)
    installed = {"knowop": "0.1.6"}
    monkeypatch.setattr(metadata, "version", partial(_version_of, installed=installed))
    return _Manager(cache=(0, f"{cache}\n"))


def test_a_knowop_uvx_runs_from_its_cache_is_served_by_uvx_at_that_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``uv cache clean`` deletes the cached interpreter; uvx fetches the version again."""
    run = _uvx_run(monkeypatch, tmp_path)

    Services(manager="launchd", folder=tmp_path / "agents", run=run).start(_ADA)

    path = tmp_path / "agents" / f"{_LABEL}.plist"
    log = state_dir() / "rekursiv-ai" / "knowop" / "connect" / f"ada-mac-{_TAG}.log"
    assert plistlib.loads(path.read_bytes())["ProgramArguments"] == [
        "/opt/homebrew/bin/uvx",
        "knowop@0.1.6",
        *_argv(log)[3:],
    ]
    # Once for the connect agent, once for the menu bar's, which runs the same knowop.
    assert run.uv == [["uv", "cache", "dir"]] * 2


@pytest.mark.parametrize("manager", ["launchd", "systemd"])
def test_a_uvx_service_reads_back_as_the_same_connect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manager: Literal["launchd", "systemd"],
) -> None:
    services = Services(
        manager=manager,
        folder=tmp_path / "agents",
        run=_uvx_run(monkeypatch, tmp_path),
    )

    services.start(_ADA)

    assert services.every() == [_ADA]
    path = _installed(tmp_path, manager)
    assert "knowop@0.1.6" in path.read_text()
    assert sys.executable not in path.read_text()


@pytest.mark.parametrize(
    "uv",
    [(0, "/home/ada/.cache/uv\n"), (127, "")],
    ids=["uv tool install", "no uv"],
)
def test_a_knowop_installed_outside_uvs_cache_is_served_by_its_own_interpreter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    uv: tuple[int, str],
) -> None:
    # Uvx is there to run it; this interpreter's prefix is not in uv's cache.
    monkeypatch.setattr(shutil, "which", _on_homebrews_path)
    run = _Manager(cache=uv)

    Services(manager="launchd", folder=tmp_path / "agents", run=run).start(_ADA)

    path = tmp_path / "agents" / f"{_LABEL}.plist"
    log = state_dir() / "rekursiv-ai" / "knowop" / "connect" / f"ada-mac-{_TAG}.log"
    assert plistlib.loads(path.read_bytes())["ProgramArguments"] == _argv(log)


@pytest.mark.parametrize("manager", ["launchd", "systemd"])
def test_a_definition_holds_no_secret_and_no_trax_or_knowop_variable(
    tmp_path: Path,
    manager: Literal["launchd", "systemd"],
) -> None:
    Services(manager=manager, folder=tmp_path / "agents", run=_Manager()).start(_ADA)

    # Every definition: the menu bar agent's too.
    text = "".join(path.read_text() for path in (tmp_path / "agents").iterdir())
    assert [word for word in (_KEY, "KNOWOP_", "TRAX", "EDITOR") if word in text] == []


@pytest.mark.parametrize("manager", ["launchd", "systemd"])
def test_the_definition_is_world_readable_and_the_log_is_this_users_alone(
    tmp_path: Path,
    manager: Literal["launchd", "systemd"],
) -> None:
    before = os.umask(0o077)
    try:
        Services(manager=manager, folder=tmp_path / "agents", run=_Manager()).start(
            _ADA,
        )
    finally:
        os.umask(before)

    path = _installed(tmp_path, manager)
    log = state_dir() / "rekursiv-ai" / "knowop" / "connect" / f"ada-mac-{_TAG}.log"
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    assert stat.S_IMODE(log.parent.stat().st_mode) == 0o700


@pytest.mark.parametrize("manager", ["launchd", "systemd"])
def test_a_definition_replaces_a_symlink_in_its_place_without_following_it(
    tmp_path: Path,
    manager: Literal["launchd", "systemd"],
) -> None:
    services = Services(manager=manager, folder=tmp_path / "agents", run=_Manager())
    services.start(_ADA)
    path = _installed(tmp_path, manager)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_text("keep")
    path.unlink()
    path.symlink_to(elsewhere)

    services.start(_ADA)

    assert (path.is_symlink(), elsewhere.read_text()) == (False, "keep")
    assert services.unchanged(_ADA)
    assert [p.name for p in (tmp_path / "agents").iterdir() if p.suffix == ".tmp"] == []


def test_a_first_launchd_start_enables_then_bootstraps_the_agent(
    tmp_path: Path,
) -> None:
    launchctl = _Manager(print=(113, "Could not find service"))

    Services(manager="launchd", folder=tmp_path / "agents", run=launchctl).start(_ADA)

    target = f"gui/{os.getuid()}/{_LABEL}"
    assert launchctl.calls == [
        ["launchctl", "enable", target],
        ["launchctl", "print", target],
        [
            "launchctl",
            "bootstrap",
            f"gui/{os.getuid()}",
            str(tmp_path / "agents" / f"{_LABEL}.plist"),
        ],
    ]


def test_a_loaded_agent_restarts_in_place_when_unchanged_and_reloads_when_changed(
    tmp_path: Path,
) -> None:
    services = Services(manager="launchd", folder=tmp_path / "agents", run=_Manager())
    services.start(_ADA)
    target = f"gui/{os.getuid()}/{_LABEL}"
    plist = str(tmp_path / "agents" / f"{_LABEL}.plist")
    again = _Manager()
    changed = _Manager()

    Services(manager="launchd", folder=tmp_path / "agents", run=again).start(_ADA)
    Services(manager="launchd", folder=tmp_path / "agents", run=changed).start(
        Connect(name="ada-mac", server=_SERVER),
    )

    assert again.calls[1:] == [
        ["launchctl", "print", target],
        ["launchctl", "kickstart", "-k", target],
    ]
    assert changed.calls[1:] == [
        ["launchctl", "print", target],
        ["launchctl", "bootout", target],
        ["launchctl", "bootstrap", f"gui/{os.getuid()}", plist],
    ]


class _Busy(_Manager):
    """Answers ``bootstrap`` 5 its first ``times``, as launchd does while it still holds
    the label a ``bootout`` let go of.
    """

    def __init__(self, times: int) -> None:
        super().__init__()
        self.times = times

    @override
    def __call__(self, argv: Sequence[str]) -> tuple[int, str]:
        answer = super().__call__(argv)
        if argv[1] == "bootstrap" and self.times:
            self.times -= 1
            return 5, "Bootstrap failed: 5: Input/output error"
        return answer


@pytest.mark.parametrize(("busy", "tries"), [(2, 3), (25, 20)])
def test_a_reload_bootstraps_again_while_launchd_still_holds_the_old_agent(
    tmp_path: Path,
    busy: int,
    tries: int,
) -> None:
    Services(manager="launchd", folder=tmp_path / "agents", run=_Manager()).start(_ADA)
    manager = _Busy(busy)
    waited: list[float] = []
    reload = Services(
        manager="launchd",
        folder=tmp_path / "agents",
        run=manager,
        wait=waited.append,
    )

    if busy < tries:
        reload.start(Connect(name="ada-mac", server=_SERVER))
    else:
        with pytest.raises(ClientError, match="failed with exit 5: Bootstrap failed"):
            reload.start(Connect(name="ada-mac", server=_SERVER))

    assert [call[1] for call in manager.calls].count("bootstrap") == tries
    assert waited == [0.25] * (tries - 1)


def test_a_systemd_start_reloads_enables_clears_a_crash_loop_and_restarts(
    tmp_path: Path,
) -> None:
    systemctl = _Manager(reset_failed=(1, "Unit not failed"))

    Services(manager="systemd", folder=tmp_path / "agents", run=systemctl).start(_ADA)

    assert systemctl.calls == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", _UNIT],
        ["systemctl", "--user", "reset-failed", _UNIT],
        ["systemctl", "--user", "restart", _UNIT],
    ]


def test_a_command_that_fails_is_named_with_what_it_said(tmp_path: Path) -> None:
    launchctl = _Manager(bootstrap=(5, "Bootstrap failed: 5: Input/output error\n"))

    with pytest.raises(ClientError) as err:
        Services(
            manager="launchd",
            folder=tmp_path / "agents",
            run=launchctl,
            wait=lambda _: None,
        ).start(_ADA)

    assert str(err.value) == (
        f"launchctl bootstrap gui/{os.getuid()} "
        f"{tmp_path / 'agents' / f'{_LABEL}.plist'} failed with exit 5: "
        "Bootstrap failed: 5: Input/output error"
    )


def test_removing_a_loaded_agent_boots_it_out_and_deletes_its_definition(
    tmp_path: Path,
) -> None:
    Services(manager="launchd", folder=tmp_path / "agents", run=_Manager()).start(_ADA)
    launchctl = _Manager()

    assert Services(
        manager="launchd",
        folder=tmp_path / "agents",
        run=launchctl,
    ).remove("ada-mac", _SERVER)

    target = f"gui/{os.getuid()}/{_LABEL}"
    assert launchctl.calls == [
        ["launchctl", "print", target],
        ["launchctl", "bootout", target],
    ]
    assert list((tmp_path / "agents").iterdir()) == []


def test_removing_an_agent_launchd_no_longer_holds_deletes_it_alone(
    tmp_path: Path,
) -> None:
    Services(manager="launchd", folder=tmp_path / "agents", run=_Manager()).start(_ADA)
    launchctl = _Manager(print=(113, ""))

    assert Services(
        manager="launchd",
        folder=tmp_path / "agents",
        run=launchctl,
    ).remove("ada-mac", _SERVER)

    assert [call[1] for call in launchctl.calls] == ["print"]
    assert list((tmp_path / "agents").iterdir()) == []


def test_removing_a_unit_disables_and_stops_it_then_reloads_systemd(
    tmp_path: Path,
) -> None:
    Services(manager="systemd", folder=tmp_path / "agents", run=_Manager()).start(_ADA)
    systemctl = _Manager()

    assert Services(
        manager="systemd",
        folder=tmp_path / "agents",
        run=systemctl,
    ).remove("ada-mac", _SERVER)

    assert systemctl.calls == [
        ["systemctl", "--user", "disable", "--now", _UNIT],
        ["systemctl", "--user", "daemon-reload"],
    ]
    assert list((tmp_path / "agents").iterdir()) == []


@pytest.mark.parametrize("manager", ["launchd", "systemd"])
def test_removing_what_is_not_installed_runs_nothing(
    tmp_path: Path,
    manager: Literal["launchd", "systemd"],
) -> None:
    run = _Manager()

    assert not Services(manager=manager, folder=tmp_path / "agents", run=run).remove(
        "ada-mac",
        _SERVER,
    )

    assert run.calls == []


def test_a_launchd_start_runs_the_menu_bar_beside_it_and_leaves_one_running(
    tmp_path: Path,
) -> None:
    first = _Manager(print=(113, "Could not find service"))
    again = _Manager()

    Services(manager="launchd", folder=tmp_path / "agents", run=first).start(_ADA)
    Services(manager="launchd", folder=tmp_path / "agents", run=again).start(_ADA)

    path = tmp_path / "agents" / f"{_MENUBAR}.plist"
    log = state_dir() / "rekursiv-ai" / "knowop" / "menubar.log"
    package = service.__name__.rpartition(".")[0]
    assert plistlib.loads(path.read_bytes()) == {
        "Label": _MENUBAR,
        "ProgramArguments": [sys.executable, "-m", package, "menubar"],
        "EnvironmentVariables": _env(tmp_path),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 10,
        "StandardOutPath": str(log),
        "StandardErrorPath": str(log),
    }
    target = f"gui/{os.getuid()}/{_MENUBAR}"
    assert first.menubar == [
        ["launchctl", "enable", target],
        ["launchctl", "print", target],
        ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)],
    ]
    # Started if it ended (Quit), never restarted while it runs.
    assert again.menubar[-1] == ["launchctl", "kickstart", target]
    assert stat.S_IMODE(log.stat().st_mode) == 0o600


def test_the_menu_bar_goes_with_the_last_connect_agent(tmp_path: Path) -> None:
    gpu = Connect(name="gpu1", server="https://other.example")
    services = Services(manager="launchd", folder=tmp_path / "agents", run=_Manager())
    services.start(_ADA)
    services.start(gpu)
    first, last = _Manager(), _Manager()

    Services(manager="launchd", folder=tmp_path / "agents", run=first).remove(
        "ada-mac",
        _SERVER,
    )
    kept = sorted(path.name for path in (tmp_path / "agents").iterdir())
    Services(manager="launchd", folder=tmp_path / "agents", run=last).remove(
        "gpu1",
        gpu.server,
    )

    assert (first.menubar, kept[-1]) == ([], f"{_MENUBAR}.plist")
    target = f"gui/{os.getuid()}/{_MENUBAR}"
    assert last.menubar == [
        ["launchctl", "print", target],
        ["launchctl", "bootout", target],
    ]
    assert list((tmp_path / "agents").iterdir()) == []


def test_systemd_has_no_menu_bar(tmp_path: Path) -> None:
    run = _Manager()

    Services(manager="systemd", folder=tmp_path / "agents", run=run).start(_ADA)

    assert ([p.name for p in (tmp_path / "agents").iterdir()], run.menubar) == (
        [_UNIT],
        [],
    )


def _installed(tmp_path: Path, manager: Literal["launchd", "systemd"]) -> Path:
    """Return where ``_ADA``'s definition is installed under ``manager``."""
    return tmp_path / "agents" / (f"{_LABEL}.plist" if manager == "launchd" else _UNIT)


def test_a_definition_that_does_not_stop_stays(tmp_path: Path) -> None:
    Services(manager="launchd", folder=tmp_path / "agents", run=_Manager()).start(_ADA)

    with pytest.raises(ClientError, match="bootout"):
        Services(
            manager="launchd",
            folder=tmp_path / "agents",
            run=_Manager(bootout=(5, "")),
        ).remove(
            "ada-mac",
            _SERVER,
        )

    assert sorted(p.name for p in (tmp_path / "agents").iterdir()) == [
        f"{_LABEL}.plist",
        f"{_MENUBAR}.plist",
    ]


def test_each_installed_agent_reads_back_with_whether_login_items_turned_it_off(
    tmp_path: Path,
) -> None:
    gpu = Connect(name="gpu1", server="https://other.example", flags=("--jobs",))
    Services(manager="launchd", folder=tmp_path / "agents", run=_Manager()).start(_ADA)
    Services(manager="launchd", folder=tmp_path / "agents", run=_Manager()).start(gpu)
    launchctl = _Manager(
        print_disabled=(
            0,
            (
                "disabled services = {\n"
                '\t"com.apple.Siri" => disabled\n'
                f'\t"{_LABEL}" => disabled\n'
                f'\t"ai.rekursiv.knowop.connect.gpu1.{service.server_tag(gpu.server)}"'
                " => enabled\n}\n"
            ),
        ),
    )

    assert Services(
        manager="launchd",
        folder=tmp_path / "agents",
        run=launchctl,
    ).every() == [
        Connect(name="ada-mac", server=_SERVER, flags=("--sandbox",), disabled=True),
        gpu,
    ]
    assert launchctl.calls == [["launchctl", "print-disabled", f"gui/{os.getuid()}"]]


def test_an_older_macos_says_true_for_a_disabled_agent(tmp_path: Path) -> None:
    Services(manager="launchd", folder=tmp_path / "agents", run=_Manager()).start(_ADA)
    launchctl = _Manager(print_disabled=(0, f'\t"{_LABEL}" => true\n'))

    assert [
        c.disabled
        for c in Services(
            manager="launchd",
            folder=tmp_path / "agents",
            run=launchctl,
        ).every()
    ] == [
        True,
    ]


def test_each_installed_unit_reads_back_with_whether_it_is_disabled(
    tmp_path: Path,
) -> None:
    Services(manager="systemd", folder=tmp_path / "agents", run=_Manager()).start(_ADA)
    systemctl = _Manager(list_unit_files=(0, f"{_UNIT} disabled enabled\n"))

    assert Services(
        manager="systemd",
        folder=tmp_path / "agents",
        run=systemctl,
    ).every() == [
        Connect(name="ada-mac", server=_SERVER, flags=("--sandbox",), disabled=True),
    ]
    assert systemctl.calls == [
        [
            "systemctl",
            "--user",
            "list-unit-files",
            "--no-legend",
            "knowop-connect-*",
        ],
    ]


@pytest.mark.parametrize("manager", ["launchd", "systemd"])
def test_no_service_installed_asks_the_manager_nothing(
    tmp_path: Path,
    manager: Literal["launchd", "systemd"],
) -> None:
    run = _Manager()

    assert Services(manager=manager, folder=tmp_path / "agents", run=run).every() == []

    assert run.calls == []


def test_a_unit_keeps_every_character_systemd_would_otherwise_read_as_its_own(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    odd = 'https://x.example/a b%c$d"e\\"f'
    monkeypatch.setenv("LANG", "50%$HOME\\x")
    services = Services(manager="systemd", folder=tmp_path / "agents", run=_Manager())
    connect = Connect(name="ada-mac", server=odd, flags=("--job-results-max", "1G"))

    services.start(connect)

    assert services.every() == [connect]
    [unit] = (tmp_path / "agents").glob("*.service")
    lines = unit.read_text().splitlines()
    # Systemd reads C escapes in a quoted word, ``$`` only in ExecStart.
    assert 'Environment="LANG=50%%$HOME\\\\x"' in lines
    assert any(' "https://x.example/a b%%c$$d\\"e\\\\\\"f" ' in line for line in lines)


@pytest.mark.parametrize("manager", ["launchd", "systemd"])
def test_a_definition_is_unchanged_until_a_flag_or_the_shell_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manager: Literal["launchd", "systemd"],
) -> None:
    services = Services(manager=manager, folder=tmp_path / "agents", run=_Manager())
    assert not services.unchanged(_ADA)
    services.start(_ADA)

    assert services.unchanged(_ADA)
    assert not services.unchanged(Connect(name="ada-mac", server=_SERVER))
    monkeypatch.setenv("PATH", "/usr/bin")
    assert not services.unchanged(_ADA)


@pytest.mark.parametrize(
    ("answer", "off"),
    [((0, "Linger=no\n"), True), ((0, "Linger=yes\n"), False), ((1, ""), False)],
    ids=["off", "on", "no loginctl"],
)
def test_systemd_says_when_lingering_is_off(
    tmp_path: Path,
    answer: tuple[int, str],
    off: bool,
) -> None:
    loginctl = _Manager(show_user=answer)

    assert (
        Services(
            manager="systemd",
            folder=tmp_path / "agents",
            run=loginctl,
        ).linger_off()
        is off
    )
    assert loginctl.calls[0][:2] == ["loginctl", "show-user"]
    assert loginctl.calls[0][3:] == ["-p", "Linger"]


def test_launchd_keeps_an_agent_across_logouts_and_asks_nothing(
    tmp_path: Path,
) -> None:
    run = _Manager()

    assert not Services(
        manager="launchd",
        folder=tmp_path / "agents",
        run=run,
    ).linger_off()
    assert run.calls == []


def test_a_mac_has_launchd_with_the_users_launch_agents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    run = _Manager()

    assert service.here(run) == Services(
        manager="launchd",
        folder=tmp_path / "home" / "Library" / "LaunchAgents",
        run=run,
    )
    assert run.calls == []


@pytest.mark.parametrize(
    ("answer", "found"),
    [((0, "HOME=/home/ada\n"), True), ((1, "Failed to connect to bus"), False)],
    ids=["user manager", "none"],
)
def test_linux_has_systemd_only_where_its_user_manager_answers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    answer: tuple[int, str],
    found: bool,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    run = _Manager(show_environment=answer)

    services = service.here(run, logind=tmp_path)

    expected = Services(
        manager="systemd",
        folder=config_dir()
        / "systemd"
        / "user",  # house-ignore[userdirs-namespace] -- systemd's own folder of a user's units, a vendor's namespace (AGENTS.md rule 1).
        run=run,
    )
    assert services == (expected if found else None)


def test_a_login_without_a_runtime_dir_reaches_the_lingering_manager_through_logind(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    (tmp_path / str(os.getuid())).mkdir()
    run = _Manager()

    services = service.here(run, logind=tmp_path)
    assert services is not None
    services.run(["systemctl", "--user", "daemon-reload"])

    runtime = f"XDG_RUNTIME_DIR={tmp_path / str(os.getuid())}"
    assert services.manager == "systemd"
    assert run.calls == [
        ["env", runtime, "systemctl", "--user", "show-environment"],
        [
            "env",
            runtime,
            "systemctl",
            "--user",
            "show",
            "--property=UnitPath",
            "--value",
        ],
        ["env", runtime, "systemctl", "--user", "daemon-reload"],
    ]


@pytest.mark.parametrize(
    ("unit_path", "folder"),
    [
        (
            (
                0,
                (
                    "/srv/ada/.config/systemd/user.control /run/user/1000/systemd/transient"
                    " /srv/ada/.config/systemd/user /etc/systemd/user"
                    " /run/user/1000/systemd/user /usr/lib/systemd/user\n"
                ),
            ),
            "/srv/ada/.config/systemd/user",
        ),
        ((1, "Failed to connect to bus"), None),
    ],
    ids=["as systemd says", "it says nothing"],
)
def test_units_go_where_the_manager_reads_them_not_where_the_shell_moved_its_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    unit_path: tuple[int, str],
    folder: str | None,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    run = _Manager(show=unit_path)

    services = service.here(run, logind=tmp_path)

    assert services is not None
    assert (
        services.folder
        == (
            Path(folder)
            if folder is not None
            else config_dir()
            / "systemd"
            / "user"  # house-ignore[userdirs-namespace] -- systemd's own folder of a user's units, a vendor's namespace (AGENTS.md rule 1).
        )
    )


def test_a_servers_tag_ignores_a_trailing_slash_and_tells_servers_apart() -> None:
    assert service.server_tag(f"{_SERVER}/") == _TAG
    assert service.server_tag("https://other.example") != _TAG
    assert len(_TAG) == 8
    assert int(_TAG, 16) >= 0


@pytest.mark.parametrize(
    ("installed", "version"),
    [({"knowop": "0.1.6"}, "0.1.6"), ({}, "unknown")],
    ids=["installed", "source checkout"],
)
def test_the_version_is_the_installed_knowops_or_unknown(
    monkeypatch: pytest.MonkeyPatch,
    installed: dict[str, str],
    version: str,
) -> None:
    monkeypatch.setattr(metadata, "version", partial(_version_of, installed=installed))

    assert service.knowop_version() == version


def test_a_log_under_5_mib_is_left_alone(tmp_path: Path) -> None:
    log = tmp_path / "ada-mac.log"
    log.write_bytes(b"x" * (5 * 2**20 - 1))

    service.rotate_log(log)

    assert log.stat().st_size == 5 * 2**20 - 1
    assert not log.with_name("ada-mac.log.1").exists()


def test_a_full_log_is_copied_aside_and_emptied_where_its_writer_appends(
    tmp_path: Path,
) -> None:
    log = tmp_path / "ada-mac.log"
    before = os.open(log.with_name("ada-mac.log.1"), os.O_WRONLY | os.O_CREAT, 0o600)
    os.write(before, b"the copy before, longer than nothing")
    os.close(before)
    # As launchd and systemd open it: for appending, held across the rotation.
    writer = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(writer, b"x" * (5 * 2**20 - 6) + b"\nlast\n")

        service.rotate_log(log)
        os.write(writer, b"next\n")
    finally:
        os.close(writer)

    kept = log.with_name("ada-mac.log.1")
    assert kept.read_bytes().endswith(b"\nlast\n")
    assert kept.stat().st_size == 5 * 2**20
    assert stat.S_IMODE(kept.stat().st_mode) == 0o600
    assert log.read_bytes() == b"next\n"


def test_a_missing_log_is_not_rotated(tmp_path: Path) -> None:
    service.rotate_log(tmp_path / "gone.log")

    assert list(tmp_path.iterdir()) == []


def _on_homebrews_path(name: str) -> str:
    """Find every program in Homebrew's folder, as ``shutil.which`` would there."""
    return f"/opt/homebrew/bin/{name}"


def _version_of(name: str, *, installed: dict[str, str]) -> str:
    """Answer as ``importlib.metadata.version`` would with only ``installed`` present."""
    if name not in installed:
        raise metadata.PackageNotFoundError(name)
    return installed[name]


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
