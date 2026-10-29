"""The menu bar's view of ``knowop status --json``, and its status item over fake AppKit."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Final, Protocol, cast

import subprocess
import sys

from treekle import json_dumps

import pytest

from knowop import menubar
from knowop.menubar import Line, Report, view


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path


_ALPHA: Final = "https://alpha.knowop.ai"
_LAB: Final = "http://127.0.0.1:8802"


def _report(*, lab: str = "online", lab_service: bool = True) -> Report:
    """Return a status of two logins, each with a machine this computer serves."""
    return {
        "logins": [
            {
                "server": _ALPHA,
                "account": "ada@example.com",
                "machines": [
                    {"name": "ada-mac", "status": "online", "runs": [{}, {}]},
                    {"name": "gpu1", "status": "offline", "runs": [{}]},
                ],
            },
            {
                "server": _LAB,
                "account": "ada@example.com",
                "machines": [{"name": "lab-mac", "status": lab, "runs": []}],
            },
        ],
        "services": [
            {"name": "ada-mac", "server": _ALPHA, "disabled": False},
            {"name": "lab-mac", "server": _LAB, "disabled": not lab_service},
        ],
    }


def test_the_menu_shows_each_login_its_machines_and_a_line_opening_it() -> None:
    connected, lines = view(_report())

    assert connected
    assert lines == [
        Line(title="knowop: connected"),
        Line(title=""),
        Line(title=f"{_ALPHA}: ada@example.com"),
        Line(title="ada-mac: online, 2 runs live (this computer)"),
        Line(title="gpu1: offline, 1 run live"),
        Line(title=f"Open {_ALPHA}", url=_ALPHA),
        Line(title=""),
        Line(title=f"{_LAB}: ada@example.com"),
        Line(title="lab-mac: online (this computer)"),
        Line(title=f"Open {_LAB}", url=_LAB),
    ]


@pytest.mark.parametrize(
    ("lab", "lab_service", "connected"),
    [
        ("offline", True, False),
        ("never", True, False),
        ("offline", False, True),
    ],
    ids=["a service's machine offline", "never online", "that service turned off"],
)
def test_this_computer_reads_connected_while_every_service_on_serves_online(
    lab: str,
    lab_service: bool,
    connected: bool,
) -> None:
    assert view(_report(lab=lab, lab_service=lab_service))[0] is connected


def test_no_service_here_reads_not_connected() -> None:
    report = _report()
    report["services"] = []

    connected, lines = view(report)

    assert not connected
    assert lines[0] == Line(title="knowop: not connected")


def test_a_refused_login_says_to_log_in_again_and_lists_no_machine() -> None:
    report = _report()
    report["logins"][1] = {"server": _LAB, "account": None, "machines": None}

    connected, lines = view(report)

    assert not connected
    assert lines[-2:] == [
        Line(title=f"{_LAB}: it refused this login; run knowop login"),
        Line(title=f"Open {_LAB}", url=_LAB),
    ]


def test_no_report_says_to_connect() -> None:
    assert view(None) == (
        False,
        [Line(title="knowop: not logged in; run knowop connect")],
    )


@pytest.mark.parametrize(
    ("out", "read"),
    [(json_dumps(_report()), True), ("", False), ("{}", False)],
    ids=["a report", "none before a login", "not a report"],
)
def test_status_reads_what_this_knowop_prints_whatever_it_exits_with(
    monkeypatch: pytest.MonkeyPatch,
    out: str,
    read: bool,
) -> None:
    asked: list[Sequence[str]] = []

    def answer(argv: Sequence[str], **_: object) -> subprocess.CompletedProcess[str]:
        asked.append(argv)
        return subprocess.CompletedProcess(argv, 1, stdout=out, stderr="refused")

    monkeypatch.setattr(subprocess, "run", answer)

    assert menubar.status() == (_report() if read else None)
    package = menubar.__name__.rpartition(".")[0]
    assert asked == [[sys.executable, "-m", package, "status", "--json"]]


def test_status_reads_a_service_written_with_its_defaults_left_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``knowop status --json`` leaves out each field at its default: an enabled
    # service's ``disabled``.
    report = _report()
    enabled = [{"name": s["name"], "server": s["server"]} for s in report["services"]]
    out = json_dumps({**report, "services": enabled})

    def answer(argv: Sequence[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    monkeypatch.setattr(subprocess, "run", answer)

    assert view(menubar.status())[0]


class _Item:
    def __init__(self, title: str, action: str | None = None, key: str = "") -> None:
        self.title, self.action, self.key = title, action, key
        self.target: object = None
        self.url = ""
        self.enabled = True

    def setTarget_(self, target: object) -> None:  # noqa: N802 -- Mirrors AppKit.
        self.target = target

    def setRepresentedObject_(self, url: str) -> None:  # noqa: N802 -- Mirrors AppKit.
        self.url = url

    def representedObject(self) -> str:  # noqa: N802 -- Mirrors AppKit.
        return self.url

    def setEnabled_(self, enabled: bool) -> None:  # noqa: N802 -- Mirrors AppKit.
        self.enabled = enabled


def _bytes(data: bytes, length: int) -> bytes:
    """Stand in for ``NSData.dataWithBytes_length_``: the bytes themselves."""
    assert length == len(data)
    return data


def _url(text: str) -> str:
    """Stand in for ``NSURL.URLWithString_``: the text, marked as a URL."""
    return f"url:{text}"


class _Kit:
    """Stands in for AppKit: each class answers with itself, and records what it is told."""

    def __init__(self) -> None:
        self.policy = 0
        self.length = 0.0
        self.data = b""
        self.size = (0.0, 0.0)
        self.template = False
        self.image: object = None
        self.tip = ""
        self.disabled: bool | None = None
        self.menu: list[_Item] = []
        self.opened: list[str] = []
        self.iconed: list[tuple[object, str, int]] = []

    def namespace(self) -> SimpleNamespace:
        kit = self

        return SimpleNamespace(
            NSApplication=SimpleNamespace(sharedApplication=lambda: kit),
            NSData=SimpleNamespace(dataWithBytes_length_=_bytes),
            NSImage=SimpleNamespace(alloc=lambda: kit),
            NSMenu=SimpleNamespace(alloc=lambda: SimpleNamespace(init=lambda: kit)),
            NSMenuItem=SimpleNamespace(
                alloc=lambda: SimpleNamespace(
                    initWithTitle_action_keyEquivalent_=_Item,
                ),
                separatorItem=lambda: _Item("-"),
            ),
            NSStatusBar=SimpleNamespace(systemStatusBar=lambda: kit),
            NSURL=SimpleNamespace(URLWithString_=_url),
            NSWorkspace=SimpleNamespace(sharedWorkspace=lambda: kit),
        )

    def setActivationPolicy_(self, policy: int) -> bool:  # noqa: N802 -- Mirrors AppKit.
        self.policy = policy
        return True

    def statusItemWithLength_(self, length: float) -> _Kit:  # noqa: N802 -- Mirrors AppKit.
        self.length = length
        return self

    def button(self) -> _Kit:
        return self

    def initWithData_(self, data: bytes) -> _Kit:  # noqa: N802 -- Mirrors AppKit.
        self.data = data
        return self

    def setSize_(self, size: tuple[float, float]) -> None:  # noqa: N802 -- Mirrors AppKit.
        self.size = size

    def setTemplate_(self, template: bool) -> None:  # noqa: N802 -- Mirrors AppKit.
        self.template = template

    def setImage_(self, image: object) -> None:  # noqa: N802 -- Mirrors AppKit.
        self.image = image

    def setToolTip_(self, tip: str) -> None:  # noqa: N802 -- Mirrors AppKit.
        self.tip = tip

    def setAppearsDisabled_(self, disabled: bool) -> None:  # noqa: N802 -- Mirrors AppKit.
        self.disabled = disabled

    def setMenu_(self, menu: object) -> None:  # noqa: N802 -- Mirrors AppKit.
        assert menu is self

    def removeAllItems(self) -> None:  # noqa: N802 -- Mirrors AppKit.
        self.menu.clear()

    def addItem_(self, item: _Item) -> None:  # noqa: N802 -- Mirrors AppKit.
        self.menu.append(item)

    def openURL_(self, url: str) -> bool:  # noqa: N802 -- Mirrors AppKit.
        self.opened.append(url)
        return True

    def setIcon_forFile_options_(  # noqa: N802 -- Mirrors AppKit.
        self,
        image: object,
        path: str,
        options: int,
    ) -> bool:
        self.iconed.append((image, path, options))
        return True


class _Opener(Protocol):
    """The target of an ``open:`` line, as AppKit calls it."""

    def open_(self, sender: _Item, /) -> None: ...


class _Helper:
    """Stands in for PyObjC's AppHelper: runs a call at once, and keeps a later one."""

    def __init__(self) -> None:
        self.later: list[tuple[float, Callable[[], None]]] = []
        self.loops = 0

    def callAfter(  # noqa: N802 -- Mirrors AppHelper.
        self,
        call: Callable[[Report | None], None],
        answer: Report | None,
    ) -> None:
        call(answer)

    def callLater(self, delay: float, call: Callable[[], None]) -> None:  # noqa: N802 -- Mirrors AppHelper.
        self.later.append((delay, call))

    def runEventLoop(self) -> None:  # noqa: N802 -- Mirrors AppHelper.
        self.loops += 1


def test_the_status_item_shows_the_mark_the_menu_and_refreshes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit, helper = _Kit(), _Helper()
    monkeypatch.setattr(menubar, "_appkit", kit.namespace())
    monkeypatch.setattr(menubar, "_helper", helper)
    answers = [_report(), _report(lab="offline")]

    menubar.run(report=lambda: answers.pop(0), every=7.0, spawn=lambda work: work())

    assert (kit.policy, kit.length, kit.size, kit.template) == (
        1,
        -1.0,
        (18.0, 18.0),
        True,
    )
    assert kit.data.startswith(b"<svg")
    assert kit.image is kit
    assert (kit.disabled, kit.tip, helper.loops) == (False, "knowop: connected", 1)
    titles = [item.title for item in kit.menu]
    assert titles[:3] == ["knowop: connected", "-", f"{_ALPHA}: ada@example.com"]
    assert titles[-2:] == ["-", "Quit"]
    leave = kit.menu[-1]
    assert (leave.action, leave.key, leave.target) == ("terminate:", "q", kit)
    assert not kit.menu[0].enabled
    (opens,) = [item for item in kit.menu if item.url == _LAB]
    assert opens.action == "open:"
    cast("_Opener", opens.target).open_(opens)
    assert kit.opened == [f"url:{_LAB}"]

    ((delay, refresh),) = helper.later
    refresh()

    assert delay == 7.0
    assert (kit.disabled, kit.tip) == (True, "knowop: not connected")
    assert [item.title for item in kit.menu].count("Quit") == 1


@pytest.mark.parametrize("present", [True, False], ids=["AppKit", "none"])
def test_this_python_says_why_it_cannot_start_the_menu_bar(
    monkeypatch: pytest.MonkeyPatch,
    present: bool,
) -> None:
    # None in sys.modules fails an import, as a missing module does.
    monkeypatch.setitem(
        sys.modules,
        "AppKit",
        ModuleType("AppKit") if present else None,
    )

    said = menubar.problem()

    assert (said is None) is present
    assert present or "AppKit" in (said or "")


def test_a_file_gets_the_knowop_icon_on_a_dark_square(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    kit = _Kit()
    monkeypatch.setattr(menubar, "_appkit", kit.namespace())

    menubar.set_icon(tmp_path / "knowop")

    assert kit.iconed == [(kit, str(tmp_path / "knowop"), 0)]
    assert kit.data.startswith(b"<svg")
    assert b"<rect" in kit.data


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
