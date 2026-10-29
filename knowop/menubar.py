# ruff: noqa: N802 -- The Protocols below mirror AppKit's selector names.
"""Show knowop in the macOS menu bar: whether this computer is connected, and where.

On macOS, ``knowop connect`` at a terminal starts ``knowop menubar`` beside its connect
services, as a launchd agent of its own (:mod:`service`). It shows the knowop mark,
dimmed while a connect service of this computer is not online, and a menu with each
server knowop is logged in to: the account, its machines and their live runs, as
``knowop status`` shows them, and a line that opens the server's web app. Every 10 s it
reads ``knowop status --json`` from a process of its own, so a slow server never holds
the menu bar and a knowop upgraded since gives the next answer. Quit ends it until the
next login or ``knowop connect``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, NotRequired, Protocol, TypedDict, cast

import importlib
import subprocess
import sys
import threading

from treekle import json_loads
from wrapt import lazy_import


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


# macOS alone has them: importing them eagerly would make this module, and so its pure
# half, unimportable on Linux. Each is resolved at its first use.
_appkit = lazy_import("AppKit")
_helper = lazy_import("PyObjCTools.AppHelper")


__all__ = ["Line", "Report", "problem", "run", "set_icon", "status", "view"]


_MARK: Final = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <g fill="none" stroke="#000" stroke-width="8" stroke-linecap="square"
     stroke-linejoin="round">
    <path d="M9 10V54"/>
    <path d="M29 12 9 32 29 52H35a20 20 0 0 0 0-40Z"/>
  </g>
</svg>"""
"""The knowop mark, as the web app's favicon draws it, in black: a template image."""

_ICON: Final = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="14" fill="#0b0c0e"/>
  <g fill="none" stroke="#f7f8f8" stroke-width="6.5" stroke-linecap="square"
     stroke-linejoin="round" transform="translate(12.8 12.8) scale(0.6)">
    <path d="M9 10V54"/>
    <path d="M29 12 9 32 29 52H35a20 20 0 0 0 0-40Z"/>
  </g>
</svg>"""
"""The knowop icon, as the web app's draws it: the mark, light, on a dark square."""

_ACCESSORY: Final = 1
"""``NSApplicationActivationPolicyAccessory``: a menu bar item, no Dock icon."""

_VARIABLE_LENGTH: Final = -1.0
"""``NSVariableStatusItemLength``: as wide as the mark."""


class _Machine(TypedDict):
    name: str
    status: str
    runs: list[object]


class _Login(TypedDict):
    server: str
    account: str | None
    machines: list[_Machine] | None


class _Service(TypedDict):
    name: str
    server: str
    disabled: NotRequired[bool]
    """``service.Connect``'s, left out at its default, ``False``."""


class Report(TypedDict):
    """The part of ``knowop status --json`` the menu shows."""

    logins: list[_Login]
    services: list[_Service]


@dataclass(frozen=True, slots=True, kw_only=True)
class Line:
    """One line of the menu: what it says, and the page it opens.

    Attributes:
      title: Its text; empty for a separator.
      url: The page it opens; empty for a line that only says something.

    """

    title: str
    url: str = ""


def view(report: Report | None) -> tuple[bool, list[Line]]:
    """Return whether this computer is connected, and the menu's lines.

    Args:
      report: ``knowop status --json``; ``None`` when it gave none, as before a login.

    Returns:
      connected: Whether every connect service here that is on serves a machine
        its server shows online; never without one.
      lines: A heading; then each login: its server and account, its machines with
        their state and live runs, and a line that opens the server.

    """
    if report is None:
        return False, [Line(title="knowop: not logged in; run knowop connect")]
    online = {
        (login["server"], machine["name"])
        for login in report["logins"]
        for machine in login["machines"] or []
        if machine["status"] == "online"
    }
    served = {
        (s["server"], s["name"])
        for s in report["services"]
        if s.get("disabled") is not True
    }
    connected = bool(served) and served <= online
    lines: list[Line] = [
        Line(title=f"knowop: {'connected' if connected else 'not connected'}"),
    ]
    for login in report["logins"]:
        who = login["account"] or "it refused this login; run knowop login"
        lines += [Line(title=""), Line(title=f"{login['server']}: {who}")]
        for machine in login["machines"] or []:
            count = len(machine["runs"])
            runs = f", {count} run{'' if count == 1 else 's'} live" if count else ""
            here = (
                " (this computer)"
                if (login["server"], machine["name"]) in served
                else ""
            )
            lines.append(
                Line(title=f"{machine['name']}: {machine['status']}{runs}{here}"),
            )
        lines.append(Line(title=f"Open {login['server']}", url=login["server"]))
    return connected, lines


# Kept in the file's extended attributes, as Finder keeps a custom icon.
def set_icon(path: Path) -> None:
    """Give the file ``path`` the knowop icon, which Finder and Login Items show."""
    kit = cast("_AppKit", _appkit)
    icon = kit.NSImage.alloc().initWithData_(
        kit.NSData.dataWithBytes_length_(_ICON, len(_ICON)),
    )
    kit.NSWorkspace.sharedWorkspace().setIcon_forFile_options_(icon, str(path), 0)


# Imported, not looked up: a PyObjC install that is there but broken fails the same way.
def problem() -> str | None:
    """Return why the menu bar cannot start in this Python; ``None`` when it can.

    Returns:
      problem: What importing AppKit raised here, as in a checkout whose environment
        predates PyObjC; ``None`` where it imports.

    """
    try:
        importlib.import_module("AppKit")
    except ImportError as error:
        return str(error)
    return None


def status() -> Report | None:
    """Return what ``knowop status --json`` prints, from this knowop.

    Returns:
      report: The report; ``None`` when it printed none, as before a login.

    """
    done = subprocess.run(  # noqa: S603 -- this interpreter running this package, fixed words.
        # ``knowop`` once exported; the package that holds this module, in any form.
        [sys.executable, "-m", __name__.rpartition(".")[0], "status", "--json"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    # A login the server refused exits 1 after the report, which still says so.
    try:
        return json_loads(done.stdout, Report)
    except ValueError:
        return None


# Each refresh asks in a thread of its own and shows its answer on the main thread,
# which alone may touch AppKit.
def run(
    *,
    report: Callable[[], Report | None] = status,
    every: float = 10.0,
    spawn: Callable[[Callable[[], None]], None] = lambda work: threading.Thread(
        target=work,
        daemon=True,
    ).start(),
) -> None:
    """Show knowop's status item, refreshed every ``every`` s, until Quit.

    Args:
      report: Answers each refresh, off the main thread.
      every: Seconds from one answer shown to the next refresh.
      spawn: Runs a refresh off the main thread.

    """
    kit = cast("_AppKit", _appkit)
    helper = cast("_AppHelper", _helper)
    app = kit.NSApplication.sharedApplication()
    app.setActivationPolicy_(_ACCESSORY)
    button = (
        item := kit.NSStatusBar.systemStatusBar().statusItemWithLength_(
            _VARIABLE_LENGTH,
        )
    ).button()
    mark = kit.NSImage.alloc().initWithData_(
        kit.NSData.dataWithBytes_length_(_MARK, len(_MARK)),
    )
    mark.setSize_((18.0, 18.0))
    mark.setTemplate_(True)
    button.setImage_(mark)
    menu = kit.NSMenu.alloc().init()
    item.setMenu_(menu)
    opener = _Opener(kit)

    def show(answer: Report | None) -> None:
        connected, lines = view(answer)
        button.setAppearsDisabled_(not connected)
        button.setToolTip_(lines[0].title)
        menu.removeAllItems()
        for line in [*lines, Line(title="")]:
            menu.addItem_(_menu_item(kit, line, opener))
        leave = kit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Quit",
            "terminate:",
            "q",
        )
        leave.setTarget_(app)
        menu.addItem_(leave)
        helper.callLater(every, refresh)

    def refresh() -> None:
        spawn(lambda: helper.callAfter(show, report()))

    refresh()
    helper.runEventLoop()


def _menu_item(kit: _AppKit, line: Line, opener: _Opener) -> _MenuItem:
    """Return ``line`` as a menu item: a separator, a page it opens, or text alone."""
    if not line.title:
        return kit.NSMenuItem.separatorItem()
    item = kit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        line.title,
        "open:" if line.url else None,
        "",
    )
    if line.url:
        item.setTarget_(opener)
        item.setRepresentedObject_(line.url)
    else:
        item.setEnabled_(False)
    return item


# PyObjC hands AppKit's ``open:`` to the Python method ``open_``, for as long as the
# object lives; a menu item holds its target weakly, so ``run`` holds this one.
class _Opener:
    """The target of the menu's ``open:`` lines: it opens each one's page."""

    def __init__(self, kit: _AppKit) -> None:
        self.kit = kit

    def open_(self, sender: _MenuItem) -> None:
        """Open the page ``sender`` names."""
        url = self.kit.NSURL.URLWithString_(sender.representedObject())
        self.kit.NSWorkspace.sharedWorkspace().openURL_(url)


class _Image(Protocol):
    def setSize_(self, size: tuple[float, float]) -> None: ...
    def setTemplate_(self, template: bool) -> None: ...


class _Button(Protocol):
    def setImage_(self, image: _Image) -> None: ...
    def setToolTip_(self, tip: str) -> None: ...
    def setAppearsDisabled_(self, disabled: bool) -> None: ...


class _MenuItem(Protocol):
    def setTarget_(self, target: object) -> None: ...
    def setRepresentedObject_(self, value: str) -> None: ...
    def representedObject(self) -> str: ...
    def setEnabled_(self, enabled: bool) -> None: ...


class _Menu(Protocol):
    def removeAllItems(self) -> None: ...
    def addItem_(self, item: _MenuItem) -> None: ...


class _StatusItem(Protocol):
    def button(self) -> _Button: ...
    def setMenu_(self, menu: _Menu) -> None: ...


class _StatusBar(Protocol):
    def statusItemWithLength_(self, length: float) -> _StatusItem: ...


class _StatusBars(Protocol):
    def systemStatusBar(self) -> _StatusBar: ...


class _Application(Protocol):
    def setActivationPolicy_(self, policy: int) -> bool: ...


class _Applications(Protocol):
    def sharedApplication(self) -> _Application: ...


class _NewImage(Protocol):
    def initWithData_(self, data: object) -> _Image: ...


class _Images(Protocol):
    def alloc(self) -> _NewImage: ...


class _Data(Protocol):
    def dataWithBytes_length_(self, data: bytes, length: int) -> object: ...


class _NewMenu(Protocol):
    def init(self) -> _Menu: ...


class _Menus(Protocol):
    def alloc(self) -> _NewMenu: ...


class _NewMenuItem(Protocol):
    def initWithTitle_action_keyEquivalent_(
        self,
        title: str,
        action: str | None,
        key: str,
    ) -> _MenuItem: ...


class _MenuItems(Protocol):
    def alloc(self) -> _NewMenuItem: ...
    def separatorItem(self) -> _MenuItem: ...


class _URLs(Protocol):
    def URLWithString_(self, text: str) -> object: ...


class _Workspace(Protocol):
    def openURL_(self, url: object) -> bool: ...
    def setIcon_forFile_options_(
        self,
        image: _Image,
        path: str,
        options: int,
    ) -> bool: ...


class _Workspaces(Protocol):
    def sharedWorkspace(self) -> _Workspace: ...


class _AppKit(Protocol):
    """The part of PyObjC's ``AppKit`` this module calls, which ships no types."""

    NSApplication: _Applications
    NSData: _Data
    NSImage: _Images
    NSMenu: _Menus
    NSMenuItem: _MenuItems
    NSStatusBar: _StatusBars
    NSURL: _URLs
    NSWorkspace: _Workspaces


class _AppHelper(Protocol):
    """The part of PyObjC's ``PyObjCTools.AppHelper`` this module calls."""

    def callAfter(
        self,
        call: Callable[[Report | None], None],
        answer: Report | None,
        /,
    ) -> None: ...
    def callLater(self, delay: float, call: Callable[[], None], /) -> None: ...
    def runEventLoop(self) -> None: ...
