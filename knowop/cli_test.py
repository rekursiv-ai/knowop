"""The ``knowop`` command, against a scripted server and a clock that never waits.

The daemon, the agent process and the job runner are stood in for: each test
reads what knowop handed them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, override

import getpass
import io
import os
import select
import signal
import sys
import time
import uuid

from trackinizer.client.errors import ClientError
from trackinizer.lib.posix import hostname
from trackinizer.lib.userdirs import config_dir, state_dir
from trackinizer.trax.profile import (
    Profile,
    current_profile,
    del_profile,
    list_profiles,
    read_profile,
    save_profile,
    switch_profile,
)
from trackinizer.wire.wire_daemon import ENROLL_PATH
from trackinizer.wire.wire_device import (
    DEVICE_CODE_PATH,
    DEVICE_POLL_PATH,
    DeviceCode,
    DeviceToken,
    IssuedKey,
)
from treekle import json_loads

import httpx2
import pytest

from knowop import cli, menubar, service, upload
from knowop.connect import serve
from knowop.jobs import cli as jobs_cli
from knowop.process import main
from knowop.service import Connect, Services


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from trackinizer.client.client import Client


_KEY: Final = "trax_" + "k" * 43
_KEY_ID: Final = "00000000-0000-0000-0000-000000000007"
_EMAIL: Final = "ada@example.com"
_LOGIN_TOKEN: Final = "enr_" + "0" * 32 + "_" + "l" * 43
_ENROLL_TOKEN: Final = "enr_" + "0" * 32 + "_" + "e" * 43
_PROD: Final = Profile(url="https://prod.example", api_key="trax_" + "p" * 43)
_CANVAS: Final = "11111111-1111-4111-8111-1111111111c1"
_RECORD: Final = "22222222-2222-4222-8222-2222222222c2"
_VISUAL: Final = "33333333-3333-4333-8333-3333333333c3"


class _Server:
    """One scripted server behind every host: device login, the caller, hosts."""

    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self.decision: Literal["approved", "denied"] = "approved"
        self.revoked = False
        """Whether the key was revoked: every authenticated route answers 401."""
        self.refuse: dict[str, int] = {}
        """The status each listed path answers with, in place of its answer."""
        self.hosts = [
            _host("ada-mac", status="online", owner=_EMAIL),
            _host("gpu1", status="offline", owner=_EMAIL),
            _host("lab", status="online", owner="bob@example.com"),
        ]
        """What ``/api/hosts`` lists: the caller's two, and another person's."""
        self.host: object = None
        self.runs: dict[str, list[dict[str, object]]] = {}
        """Each host's live sessions, as its runs route lists them."""
        self.facts: dict[str, dict[str, object]] = {}
        """What each daemon last reported about itself."""
        self.disconnected: list[str] = []
        """Each host disconnected, in order."""
        self.logged_out: list[str] = []
        """Each host logged out, in order."""
        self.lingering: set[str] = set()
        """Hosts that still read online once disconnected."""
        self.key_host: str | None = None
        """The host a login made the key for, as the profile names it, since keys
        name one: that host's logout revokes the key."""

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if request.url.path == DEVICE_CODE_PATH:
            self.host = json_loads(request.content, dict[str, object])["host"]
            return httpx2.Response(201, content=_CODE.model_dump_json())
        if request.url.path == DEVICE_POLL_PATH:
            return httpx2.Response(200, content=self._poll().model_dump_json())
        if self.revoked or request.headers.get("authorization") != f"Bearer {_KEY}":
            return httpx2.Response(401, json={"detail": "Invalid or expired token"})
        if (status := self.refuse.get(request.url.path)) is not None:
            return httpx2.Response(status, json={"detail": "refused"})
        return self._answer((request.method, request.url.path))

    def netlocs(self) -> set[str]:
        """Return every host a request went to."""
        return {request.url.host for request in self.requests}

    def serves(self, name: str, *, pid: int, version: str = "unknown") -> None:
        """Show ``name`` online, served by knowop ``version`` in process ``pid``."""
        # By name, as the server lists them.
        self.hosts = sorted(
            [
                *(host for host in self.hosts if host["name"] != name),
                _host(name, status="online", owner=_EMAIL),
            ],
            key=lambda host: str(host["name"]),
        )
        self.facts[name] = {"pid": pid, "knowop_version": version}

    def server_status(self, name: str) -> str:
        """Return the status the server lists ``name`` with."""
        return next(str(m["status"]) for m in self.hosts if m["name"] == name)

    def _set_status(self, name: str, status: str) -> None:
        for host in self.hosts:
            if host["name"] == name:
                host["status"] = status

    def paths(self) -> list[str]:
        """Return every request's path, in order."""
        return [request.url.path for request in self.requests]

    def _poll(self) -> DeviceToken:
        if self.decision == "denied":
            return DeviceToken(status="denied")
        # An approval issues a fresh key, which no earlier revoke touched.
        self.revoked = False
        return DeviceToken(
            status="approved",
            issued=IssuedKey(
                api_key=_KEY,
                role="writer",
                email=_EMAIL,
                host_token="" if self.host is None else _LOGIN_TOKEN,
            ),
        )

    def _answer(self, route: tuple[str, str]) -> httpx2.Response:
        if route == ("GET", "/api/me/profile"):
            return httpx2.Response(
                200,
                json={
                    "email": _EMAIL,
                    "name": "Ada",
                    "role": "writer",
                    "api_key_id": _KEY_ID,
                    "host": self.key_host,
                },
            )
        if route == ("POST", f"/api/me/tokens/{_KEY_ID}/revoke"):
            self.revoked = True
            return httpx2.Response(200, json={"ok": True})
        if route == ("GET", "/api/hosts"):
            return httpx2.Response(200, json={"hosts": self.hosts})
        if route == ("GET", f"/api/canvases/{_CANVAS}"):
            return httpx2.Response(200, json={"id": _CANVAS, "revision": 3})
        if route == ("POST", f"/api/canvases/{_CANVAS}/operations"):
            shown = {"id": _VISUAL, "type": "trax.subgraph", "placement": "side"}
            return httpx2.Response(
                200,
                json={
                    "id": _CANVAS,
                    "revision": 4,
                    "visuals": [{**shown, "record_id": _RECORD}],
                },
            )
        for host in list(self.hosts):
            name = str(host["name"])
            if route == ("GET", f"/api/hosts/{name}"):
                detail = {
                    **host,
                    "daemon_version": "",
                    "facts": self.facts.get(name, {}),
                }
                return httpx2.Response(200, json=detail)
            if route == ("GET", f"/api/hosts/{name}/runs"):
                return httpx2.Response(200, json={"runs": self.runs.get(name, [])})
            if route == ("POST", f"/api/hosts/{name}/disconnect"):
                self.disconnected.append(name)
                if name not in self.lingering:
                    self._set_status(name, "revoked")
                return httpx2.Response(200, json={"runs": self.runs.pop(name, [])})
            if route == ("POST", f"/api/hosts/{name}/logout"):
                self.logged_out.append(name)
                self._set_status(name, "revoked")
                self.revoked = name == self.key_host
                keys = int(name == self.key_host)
                return httpx2.Response(
                    200,
                    json={"runs": self.runs.pop(name, []), "keys": keys},
                )
        if route == ("POST", ENROLL_PATH):
            return httpx2.Response(
                201,
                json={
                    "token": _ENROLL_TOKEN,
                    "expires_at": "2026-10-22T12:15:00Z",
                    "owner": _EMAIL,
                },
            )
        return httpx2.Response(404, json={"detail": "Not Found"})


_CODE: Final = DeviceCode(
    device_code="d" * 43,
    user_code="BCDF-GHJK",
    verification_uri="https://knowop.example/auth/device",
    verification_uri_complete="https://knowop.example/auth/device?user_code=BCDF-GHJK",
    expires_in=600,
    interval=5,
)


class _Time:
    """A monotonic clock that a sleep advances at once."""

    def __init__(self) -> None:
        self.now = 0.0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@dataclass(frozen=True, slots=True, kw_only=True)
class _Served:
    """One ``serve.connect_host`` call, as the daemon saw it."""

    name: str
    server: str
    token: str
    """The token the host joined with: the one handed in, else one it enrolled."""

    serve_tools: bool
    sandbox: bool
    jobs: bool = False
    results_max: int = 256 * 2**20
    join_only: bool = False
    signed_in_to: str = field(default="", compare=False)
    """The server of the login the host signs in with; empty for none. Left out of
    comparisons: the tests that compare whole calls ask what was served."""


class _Host:
    """Stands in for ``serve.connect_host``: records the call, takes a token."""

    def __init__(
        self,
        *,
        reads_token: bool = True,
        ending: BaseException | int | None = None,
        waits: int = 0,
    ) -> None:
        self.reads_token = reads_token
        """Whether it needs a token, as a host with no saved credential does."""
        self.ending = ending
        """What stops it: an exception it raises, or a signal it sends itself."""
        self.waits = waits
        """How many heartbeats it waits out before it ends."""
        self.connects: list[_Served] = []
        self.joined: Callable[[str], None] | None = None
        """Called with the name on a join here, as the server then shows it online."""

    def __call__(
        self,
        name: str,
        *,
        server: str,
        signed_in: Callable[[], Client] | None,
        sandbox: bool,
        serve_tools: bool,
        sleep: Callable[[float], None],
        token: str = "",
        jobs: bool = False,
        results_max: int = 0,
        join_only: bool = False,
    ) -> None:
        for _ in range(self.waits):
            sleep(15.0)
        if isinstance(self.ending, BaseException):
            raise self.ending
        if isinstance(self.ending, int):
            os.kill(os.getpid(), self.ending)
            # The signal's handler raises into this wait at once; 5 s only bounds a broken one.
            time.sleep(5)  # house-ignore[sleep] -- See above.
        if self.reads_token and not token and signed_in is not None:
            token = signed_in().enroll_host(name).token
        signed_in_to = ""
        if signed_in is not None:
            with signed_in() as http:
                signed_in_to = http.base_url.rstrip("/")
        self.connects.append(
            _Served(
                name=name,
                server=server,
                token=token if self.reads_token else "",
                serve_tools=serve_tools,
                sandbox=sandbox,
                jobs=jobs,
                results_max=results_max,
                join_only=join_only,
                signed_in_to=signed_in_to,
            ),
        )
        if join_only and self.joined is not None:
            self.joined(name)


class _Launchd:
    """Stands in for launchctl, and systemctl: a started service serves its host.

    Each start is a new process, on the server with this knowop's version unless
    ``version`` names another; a scripted answer replaces what a command does. This
    computer has no ``uv``, so the service runs this interpreter. The menu bar agent
    serves no host: its commands are kept apart, by verb.
    """

    def __init__(self, server: _Server, **answers: tuple[int, str]) -> None:
        self.server = server
        self.answers = answers
        self.calls: list[list[str]] = []
        self.loaded: set[str] = set()
        self.pid = 4242
        self.version = service.knowop_version()
        self.menubar: list[str] = []

    def __call__(self, argv: Sequence[str]) -> tuple[int, str]:
        if argv[0] == "uv":
            return 127, ""
        verb = argv[2] if argv[1] == "--user" else argv[1]
        if "ai.rekursiv.knowop.menubar" in argv[-1]:
            self.menubar.append(verb)
            return 0, ""
        self.calls.append(list(argv))
        if (answer := self.answers.get(verb.replace("-", "_"))) is not None:
            return answer
        label = Path(argv[-1]).stem if verb == "bootstrap" else argv[-1].split("/")[-1]
        if verb == "print":
            return (0, "") if label in self.loaded else (113, "")
        if verb == "bootout":
            self.loaded.discard(label)
        if verb in {"bootstrap", "kickstart"}:
            self.loaded.add(label)
            self._start(label.split(".")[-2])
        if verb == "restart":
            self._start(label.removeprefix("knowop-connect-").rsplit("-", 1)[0])
        return 0, ""

    def started(self) -> list[str]:
        """Return each command that started an agent: its verb."""
        return [call[1] for call in self.calls if call[1] in {"bootstrap", "kickstart"}]

    def _start(self, name: str) -> None:
        self.pid += 1
        self.server.serves(name, pid=self.pid, version=self.version)


class _Terminal(io.TextIOWrapper):
    """Stands for a terminal as stdout: a person runs knowop there."""

    def __init__(self) -> None:
        self.screen = io.BytesIO()
        super().__init__(self.screen, encoding="utf-8", line_buffering=True)

    @override
    def isatty(self) -> bool:
        return True

    def lines(self) -> list[str]:
        """Return what knowop printed, line by line."""
        self.flush()
        return self.screen.getvalue().decode().splitlines()


class _Runner:
    """Stands in for the job runner knowop hands off to: records argv, exits ``code``."""

    def __init__(self, *, code: int = 0) -> None:
        self.code = code
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str] | None = None) -> int:
        assert argv is not None, "knowop must hand the runner its arguments"
        self.calls.append(list(argv))
        return self.code


class _AgentProcess:
    """Stands in for the agent process ``run`` hands off to; exits ``code``."""

    def __init__(self, *, code: int = 0) -> None:
        self.code = code
        self.calls: list[tuple[list[str], str | None]] = []
        """Each hand-off: its arguments, and its profile's server or ``None``."""

    def __call__(self, argv: Sequence[str], *, profile: Profile | None) -> int:
        self.calls.append((list(argv), None if profile is None else profile.url))
        return self.code


@dataclass(frozen=True, slots=True, kw_only=True)
class _Runners:
    """The two runners ``run`` and ``job`` hand off to."""

    process: _AgentProcess
    job: _Runner

    def calls(self) -> list[tuple[str, list[str]]]:
        """Return every hand-off, each with the verb that made it."""
        return [
            *(("run", argv) for argv, _ in self.process.calls),
            *(("job", argv) for argv in self.job.calls),
        ]


@pytest.fixture(autouse=True)
def server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Server:
    """Answer every request from the scripted server, on a computer named Ada_Mac."""
    scripted = _Server()
    monkeypatch.setattr(httpx2, "HTTPTransport", _server_transport(scripted))
    for name in (
        "TRACKINIZER_PROFILE",
        "TRACKINIZER_URL",
        "TRACKINIZER_TOKEN",
        "TRAX_SERVER",
        "LC_TRAX_SERVER",
    ):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(hostname, "stable_hostname", lambda: "Ada_Mac.local")
    # A person at a terminal pipes no token; a test that does sets its own stdin.
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    # The real host serves until Ctrl-C, and the real runners start an agent or a job.
    monkeypatch.setattr(serve, "connect_host", _never_runs("connect_host"))
    monkeypatch.setattr(main, "main", _never_runs("the agent process"))
    monkeypatch.setattr(jobs_cli, "main", _never_runs("the job runner"))
    return scripted


@pytest.fixture
def launchd(server: _Server) -> _Launchd:
    return _Launchd(server)


@pytest.fixture(autouse=True)
def agents(
    launchd: _Launchd,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Services:
    """Stand in for launchctl and this user's agents folder: no real service runs."""
    # This Python can start the menu bar, wherever the tests run.
    stand_in = Services(
        manager="launchd",
        folder=tmp_path / "agents",
        run=launchd,
        menubar_problem=lambda: None,
        set_icon=lambda _path: None,
    )
    monkeypatch.setattr(service, "here", lambda: stand_in)
    return stand_in


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch) -> _Host:
    stand_in = _Host()
    monkeypatch.setattr(serve, "connect_host", stand_in)
    return stand_in


@pytest.fixture
def runners(monkeypatch: pytest.MonkeyPatch) -> _Runners:
    stand_ins = _Runners(process=_AgentProcess(), job=_Runner())
    monkeypatch.setattr(main, "main", stand_ins.process)
    monkeypatch.setattr(jobs_cli, "main", stand_ins.job)
    return stand_ins


def test_login_with_no_argument_signs_in_to_the_default_server(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login") == 0

    assert server.netlocs() == {"alpha.knowop.ai"}
    assert json_loads(server.requests[0].content, dict[str, object]) == {
        "client": "knowop",
        "hostname": "Ada_Mac",
        "host": None,
    }
    assert read_profile("alpha.knowop.ai") == Profile(
        url="https://alpha.knowop.ai",
        api_key=_KEY,
    )
    out = capsys.readouterr().out
    assert "BCDF-GHJK" in out
    assert "logged in: profile alpha.knowop.ai (writer)" in out
    assert _KEY not in out


@pytest.mark.parametrize(
    ("target", "url", "profile"),
    [
        ("acme", "https://acme.knowop.ai", "acme.knowop.ai"),
        ("a1-b2", "https://a1-b2.knowop.ai", "a1-b2.knowop.ai"),
        (
            "https://knowop.example.org",
            "https://knowop.example.org",
            "knowop.example.org",
        ),
        ("http://127.0.0.1:8790", "http://127.0.0.1:8790", "127.0.0.1-8790"),
    ],
)
def test_login_reads_an_org_as_its_knowop_server_and_a_url_as_given(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
    target: str,
    url: str,
    profile: str,
) -> None:
    assert _knowop("login", target) == 0

    assert str(server.requests[0].url) == f"{url}{DEVICE_CODE_PATH}"
    assert list_profiles() == [(profile, Profile(url=url, api_key=_KEY))]
    capsys.readouterr()
    assert _knowop("status") == 0
    assert f"profile: {profile}\n" in capsys.readouterr().out


def test_output_goes_out_line_by_line_so_a_reader_sees_the_code_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # What an agent's pipe gets: a block-buffered stream.
    piped = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    monkeypatch.setattr(sys, "stdout", piped)

    assert _knowop("status") == 1

    assert piped.line_buffering


@pytest.mark.parametrize(
    "target",
    ["Acme", "acme.knowop.ai", "acme corp", "acme-", "ftp://acme.example"],
)
def test_login_refuses_what_is_neither_an_org_nor_a_url(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
    target: str,
) -> None:
    assert _knowop("login", target) == 1

    assert server.requests == []
    assert list_profiles() == []
    err = capsys.readouterr().err
    assert err.startswith("knowop: ")
    assert err.count("\n") == 1


def test_login_refuses_plain_http_beyond_this_computer(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login", "http://knowop.example.org") == 1

    assert server.requests == []
    assert "plain http" in capsys.readouterr().err


def test_a_denied_login_leaves_knowop_logged_out(server: _Server) -> None:
    server.decision = "denied"

    assert _knowop("login", "acme") == 1

    assert list_profiles() == []
    assert _knowop("status") == 1


def test_login_leaves_traxs_current_profile_alone(
    capsys: pytest.CaptureFixture[str],
) -> None:
    save_profile("prod", _PROD)
    switch_profile("prod")

    assert _knowop("login", "acme") == 0

    assert current_profile() == "prod"
    capsys.readouterr()
    assert _knowop("status") == 0
    assert "server:  https://acme.knowop.ai\n" in capsys.readouterr().out


def test_knowop_records_its_profile_in_its_own_config_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = config_dir() / "rekursiv-ai" / "knowop" / "profile"
    opened: list[Path] = []
    read_text = Path.read_text

    # Path.read_text's own parameters: an installed knowop reads its version
    # metadata with an encoding.
    def recording(path: Path, encoding: str | None = None) -> str:
        opened.append(path)
        return read_text(path, encoding=encoding)

    assert _knowop("login", "acme") == 0
    monkeypatch.setattr(Path, "read_text", recording)
    assert _knowop("status") == 0

    # Names, not contents: macOS's case-insensitive filesystem opens the file under
    # any casing of its path, so only the names written and read tell them apart.
    files = (path for path in config_dir().rglob("*") if path.is_file())
    assert sorted(path.relative_to(config_dir()).as_posix() for path in files) == [
        "rekursiv-ai/knowop/profile",
        "rekursiv-ai/trax/current",
        "rekursiv-ai/trax/profiles/acme.knowop.ai",
    ]
    assert state in opened
    assert state.read_text() == "acme.knowop.ai\n"


def test_every_verb_reaches_knowops_server_while_trax_points_at_another(
    server: _Server,
    host: _Host,
    runners: _Runners,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_profile("prod", _PROD)
    switch_profile("prod")
    monkeypatch.setenv("TRACKINIZER_PROFILE", "prod")
    monkeypatch.setenv("TRACKINIZER_URL", _PROD.url)

    assert _knowop("login", "acme") == 0
    for argv in (
        ["status"],
        ["connect"],
        ["run"],
        ["job", "logs", "7"],
        ["host"],
        ["canvas", _CANVAS],
    ):
        assert _knowop(*argv) == 0, argv
    assert _knowop("logout") == 0

    assert server.netlocs() == {"acme.knowop.ai"}
    # The host enrolled this computer with knowop's key, not trax's.
    assert [(served.server, served.token) for served in host.connects] == [
        ("https://acme.knowop.ai", _ENROLL_TOKEN),
    ]
    assert runners.process.calls == [([], _ACME)]
    assert runners.job.calls == [["--profile", "acme.knowop.ai", "logs", "7"]]
    assert list_profiles() == [("prod", _PROD)]
    assert current_profile() == "prod"


def test_status_names_the_profile_server_account_role_and_the_accounts_hosts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("status") == 0

    assert capsys.readouterr().out.splitlines() == [
        "profile: acme.knowop.ai",
        "server:  https://acme.knowop.ai",
        "account: ada@example.com",
        "role:    writer",
        "host: ada-mac online (this computer)",
        "host: gpu1 offline",
    ]


def test_status_lists_each_hosts_runs(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.runs["gpu1"] = [_run("chat-1a2b3c4d")]
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("status") == 0

    assert capsys.readouterr().out.splitlines()[4:] == [
        "host: ada-mac online (this computer)",
        "host: gpu1 offline, 1 run live",
        "  run:   chat-1a2b3c4d  claude  ~/work",
    ]


def test_status_counts_a_hosts_runs_and_shows_one_with_no_folder(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.runs["gpu1"] = [_run("chat-1a2b3c4d"), {**_run("nightly"), "cwd": None}]
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("status") == 0

    assert capsys.readouterr().out.splitlines()[5:] == [
        "host: gpu1 offline, 2 runs live",
        "  run:   chat-1a2b3c4d  claude  ~/work",
        "  run:   nightly  claude  ",
    ]


def test_status_without_a_service_manager_lists_no_service_and_no_menu_bar(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(service, "here", lambda: None)
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("status") == 0
    assert _knowop("status", "--json") == 0

    out = capsys.readouterr().out
    text, printed = out.split("{", 1)
    assert text.splitlines()[-1] == "host: gpu1 offline"
    # Indented by two, for a person reading it too.
    assert printed.startswith('\n  "logins": [')
    report = json_loads("{" + printed, dict[str, object])
    assert (report["services"], report["menubar"]) == ([], None)


def test_status_without_a_login_exits_1_with_one_line(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("status") == 1

    assert capsys.readouterr() == ("", "knowop: not logged in; run knowop login\n")
    assert server.requests == []


def test_status_after_knowops_profile_is_deleted_says_not_logged_in(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Trax refuses to delete its current profile, so another one is current.
    save_profile("prod", _PROD)
    switch_profile("prod")
    assert _knowop("login", "acme") == 0
    del_profile("acme.knowop.ai")
    capsys.readouterr()

    assert _knowop("status") == 1

    assert capsys.readouterr().err == "knowop: not logged in; run knowop login\n"


def test_status_with_a_revoked_key_says_to_log_in_again(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login", "acme") == 0
    server.revoked = True
    capsys.readouterr()

    assert _knowop("status") == 1

    out, err = capsys.readouterr()
    assert out.splitlines() == ["profile: acme.knowop.ai", f"server:  {_ACME}"]
    assert err.count("\n") == 1
    assert err.startswith("knowop: not logged in:")
    assert "knowop login" in err


def test_status_says_when_the_account_has_no_host(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.hosts = server.hosts[2:]
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("status") == 0

    assert capsys.readouterr().out.splitlines()[-1] == (
        "host: none; run knowop connect"
    )


def test_status_of_a_viewer_says_hosts_need_the_writer_role(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.refuse["/api/hosts"] = 403
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("status") == 0

    assert capsys.readouterr().out.splitlines()[-1] == (
        "host: listing hosts needs the writer role"
    )


@pytest.mark.parametrize("path", ["/api/me/profile", "/api/hosts"])
def test_status_reports_a_refusal_it_cannot_explain(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
    path: str,
) -> None:
    server.refuse[path] = 404
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("status") == 1

    assert f"GET {path} -> 404" in capsys.readouterr().err


def test_logout_revokes_this_logins_key_then_removes_its_profile(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login", "acme") == 0
    # Trax had no profile, so the login made this one trax's current profile too.
    assert current_profile() == "acme.knowop.ai"

    assert _knowop("logout") == 0

    revokes = [r for r in server.requests if r.url.path.endswith("/revoke")]
    assert [r.url.path for r in revokes] == [f"/api/me/tokens/{_KEY_ID}/revoke"]
    assert revokes[0].headers["authorization"] == f"Bearer {_KEY}"
    assert list_profiles() == []
    assert current_profile() == "default"
    # Gone, not left naming a deleted profile that a later trax login would revive.
    assert not (config_dir() / "rekursiv-ai" / "knowop" / "profile").exists()
    assert "logged out" in capsys.readouterr().out
    assert _knowop("status") == 1


def test_logout_logs_this_computers_host_out_and_its_key_goes_with_it(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.key_host = "ada-mac"
    server.runs["ada-mac"] = [_run("chat-1a2b3c4d")]
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("logout") == 0

    assert server.logged_out == ["ada-mac"]
    # The host's logout revoked the key a login made for it, so nothing is left to revoke.
    assert not any(path.endswith("/revoke") for path in server.paths())
    assert list_profiles() == []
    assert capsys.readouterr().out.splitlines() == [
        "logged out: host ada-mac; runs stopped: chat-1a2b3c4d; keys revoked: 1",
        "logged out: revoked its key; deleted profile acme.knowop.ai",
    ]


def test_logout_of_a_host_the_server_does_not_know_as_ours_revokes_the_key_alone(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login", "acme") == 0
    server.refuse["/api/hosts/ada-mac/logout"] = 404
    capsys.readouterr()

    assert _knowop("logout") == 0

    assert server.logged_out == []
    assert list_profiles() == []
    assert capsys.readouterr().out.splitlines() == [
        "logged out: revoked its key; deleted profile acme.knowop.ai",
    ]


def test_logout_keeps_the_profile_when_the_host_logout_is_refused(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login", "acme") == 0
    server.refuse["/api/hosts/ada-mac/logout"] = 409

    assert _knowop("logout") == 1

    assert [name for name, _ in list_profiles()] == ["acme.knowop.ai"]
    assert "409" in capsys.readouterr().err


def test_logout_keeps_the_profile_when_the_key_cannot_be_revoked(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login", "acme") == 0
    server.refuse[f"/api/me/tokens/{_KEY_ID}/revoke"] = 403

    assert _knowop("logout") == 1

    assert [name for name, _ in list_profiles()] == ["acme.knowop.ai"]
    assert "403" in capsys.readouterr().err
    assert _knowop("status") == 0


def test_logout_of_a_key_revoked_elsewhere_still_removes_its_profile(
    server: _Server,
) -> None:
    assert _knowop("login", "acme") == 0
    server.revoked = True

    assert _knowop("logout") == 0

    assert not any(path.endswith("/revoke") for path in server.paths())
    assert list_profiles() == []


def test_logout_without_a_login_exits_1(server: _Server) -> None:
    assert _knowop("logout") == 1

    assert server.requests == []


def test_disconnect_disconnects_this_computer_and_names_the_runs_the_server_stopped(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.runs["ada-mac"] = [_run("chat-1a2b3c4d")]
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("disconnect") == 0

    assert server.disconnected == ["ada-mac"]
    assert capsys.readouterr().out.splitlines() == [
        "stopped: run chat-1a2b3c4d (claude)",
        (
            "disconnected: host ada-mac; its connect stops at its next heartbeat, and "
            "knowop connect joins it again"
        ),
    ]


def test_disconnect_of_another_host_disconnects_that_host(server: _Server) -> None:
    assert _knowop("login", "acme") == 0

    assert _knowop("disconnect", "gpu1") == 0

    assert server.disconnected == ["gpu1"]


def test_connect_without_a_login_enrolls_this_computer_in_the_same_approval(
    server: _Server,
    host: _Host,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("connect") == 0

    assert json_loads(server.requests[0].content, dict[str, object]) == {
        "client": "knowop",
        "hostname": "Ada_Mac",
        "host": "ada-mac",
    }
    assert host.connects == [
        _Served(
            name="ada-mac",
            server="https://alpha.knowop.ai",
            token=_LOGIN_TOKEN,
            serve_tools=True,
            sandbox=False,
        ),
    ]
    assert ENROLL_PATH not in server.paths()
    assert "logged in: profile alpha.knowop.ai (writer)\n" in capsys.readouterr().out
    assert _knowop("status") == 0
    assert "profile: alpha.knowop.ai\n" in capsys.readouterr().out


def test_connect_after_a_login_enrolls_the_named_host_with_knowops_key(
    server: _Server,
    host: _Host,
) -> None:
    assert _knowop("login", "acme") == 0

    assert _knowop("connect", "--name", "gpu1", "--sandbox") == 0

    assert host.connects == [
        _Served(
            name="gpu1",
            server="https://acme.knowop.ai",
            token=_ENROLL_TOKEN,
            serve_tools=True,
            sandbox=True,
        ),
    ]
    enrolls = [r for r in server.requests if r.url.path == ENROLL_PATH]
    assert [json_loads(r.content, dict[str, object]) for r in enrolls] == [
        {"name": "gpu1"},
    ]
    assert server.paths().count(DEVICE_CODE_PATH) == 1


def test_connect_logs_in_again_when_the_server_refuses_the_saved_login(
    server: _Server,
    host: _Host,
) -> None:
    assert _knowop("login", "acme") == 0
    # Logging this host out elsewhere revoked the key the saved login holds.
    server.revoked = True

    assert _knowop("connect", "--name", "gpu1") == 0

    assert server.paths().count(DEVICE_CODE_PATH) == 2
    assert [(served.name, served.token) for served in host.connects] == [
        ("gpu1", _LOGIN_TOKEN),
    ]


def test_connect_with_no_exec_serves_terminals_only(host: _Host) -> None:
    assert _knowop("login", "acme") == 0

    assert _knowop("connect", "--no-exec") == 0

    assert [served.serve_tools for served in host.connects] == [False]
    assert [served.jobs for served in host.connects] == [False]


def test_connect_with_jobs_runs_them_and_bounds_what_one_sends_back(
    host: _Host,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login", "acme") == 0

    assert _knowop("connect", "--jobs") == 0
    assert _knowop("connect", "--jobs", "--no-exec", "--job-results-max", "1G") == 0
    assert _knowop("connect", "--jobs", "--job-results-max", "lots") == 1

    assert [(s.jobs, s.serve_tools, s.results_max) for s in host.connects] == [
        (True, True, 256 * 2**20),
        (True, False, 2**30),
    ]
    assert (
        "knowop: --job-results-max: 'lots' is not a size such as 32G or 512M"
        in capsys.readouterr().err
    )


def test_connect_to_another_server_logs_in_there_enrolling_this_computer(
    server: _Server,
    host: _Host,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login", "acme") == 0

    assert _knowop("connect", "http://127.0.0.1:8790") == 0

    codes = [r for r in server.requests if r.url.path == DEVICE_CODE_PATH]
    assert [str(r.url) for r in codes] == [
        f"https://acme.knowop.ai{DEVICE_CODE_PATH}",
        f"http://127.0.0.1:8790{DEVICE_CODE_PATH}",
    ]
    assert json_loads(codes[1].content, dict[str, object])["host"] == "ada-mac"
    assert [(served.server, served.token) for served in host.connects] == [
        ("http://127.0.0.1:8790", _LOGIN_TOKEN),
    ]
    assert ENROLL_PATH not in server.paths()
    capsys.readouterr()
    assert _knowop("status") == 0
    assert "profile: 127.0.0.1-8790\n" in capsys.readouterr().out


@pytest.mark.parametrize("target", ["acme", "https://acme.knowop.ai/"])
def test_connect_to_the_server_knowop_is_logged_in_to_logs_in_once(
    server: _Server,
    host: _Host,
    target: str,
) -> None:
    assert _knowop("login", "acme") == 0

    assert _knowop("connect", target) == 0

    assert server.paths().count(DEVICE_CODE_PATH) == 1
    assert [(served.server, served.token) for served in host.connects] == [
        ("https://acme.knowop.ai", _ENROLL_TOKEN),
    ]


def test_connect_with_a_saved_credential_enrolls_nothing(
    server: _Server,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    joined = _Host(reads_token=False)
    monkeypatch.setattr(serve, "connect_host", joined)
    assert _knowop("login", "acme") == 0

    assert _knowop("connect") == 0

    assert [served.name for served in joined.connects] == ["ada-mac"]
    assert ENROLL_PATH not in server.paths()


@pytest.mark.parametrize(
    ("name", "said"),
    [
        ("Ada_Mac", "invalid host name 'Ada_Mac'; expected "),
        ("connect", "host name 'connect' is reserved"),
        ("a" * 64, f"invalid host name '{'a' * 64}'; expected "),
    ],
)
def test_connect_refuses_a_name_no_host_may_take_before_logging_in(
    server: _Server,
    host: _Host,
    capsys: pytest.CaptureFixture[str],
    name: str,
    said: str,
) -> None:
    assert _knowop("connect", "--name", name) == 1

    assert server.requests == []
    assert host.connects == []
    assert capsys.readouterr().err.startswith(f"knowop: {said}")


def test_ctrl_c_stops_connect_with_130(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve, "connect_host", _Host(ending=KeyboardInterrupt()))
    assert _knowop("login", "acme") == 0

    assert _knowop("connect") == 130


@pytest.mark.parametrize("ending", [signal.SIGTERM, signal.SIGHUP])
def test_a_stop_or_a_hangup_ends_connect_like_ctrl_c_and_is_then_forgotten(
    monkeypatch: pytest.MonkeyPatch,
    ending: signal.Signals,
) -> None:
    """``systemctl stop`` sends one, a closed ssh session the other: both release."""
    monkeypatch.setattr(serve, "connect_host", _Host(ending=ending))
    assert _knowop("login", "acme") == 0
    original = signal.signal(ending, _handler_before)
    try:
        assert _knowop("connect") == 130

        assert signal.getsignal(ending) is _handler_before
    finally:
        signal.signal(ending, original)


def test_connect_leaves_no_client_open_for_a_later_command(
    host: _Host,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _knowop("login", "acme") == 0
    assert _knowop("connect") == 0
    later = _Server()
    monkeypatch.setattr(httpx2, "HTTPTransport", _server_transport(later))

    assert _knowop("connect") == 0

    # It asks whether the saved login still stands, then enrolls.
    assert later.paths() == ["/api/me/profile", ENROLL_PATH]
    assert len(host.connects) == 2


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--headless", "--model", "opus"],
        ["--session", "00000000-0000-0000-0000-000000000009"],
        ["--on", "lab", "--", "--name", "nightly"],
        ["--", "--help"],
    ],
    ids=["bare", "sagents-flags", "a-session", "on-a-host", "help-after-the-dashes"],
)
def test_run_is_the_agent_process_with_knowops_profile(
    runners: _Runners,
    argv: list[str],
) -> None:
    assert _knowop("login", "acme") == 0

    assert _knowop("run", *argv) == 0

    assert runners.calls() == [("run", argv)]
    assert runners.process.calls == [(argv, _ACME)]


@pytest.mark.parametrize(
    "argv",
    [["--help"], ["-h"], ["--headless", "--help"], ["-h", "--", "x"]],
    ids=["help", "h", "after-a-flag", "before-the-dashes"],
)
def test_run_help_is_the_agent_processs_needs_no_login_and_asks_the_server_nothing(
    server: _Server,
    runners: _Runners,
    argv: list[str],
) -> None:
    assert _knowop("run", *argv) == 0

    assert runners.process.calls == [(argv, None)]
    assert server.requests == []
    assert list_profiles() == []


@pytest.mark.parametrize(("argv", "code"), [(["run"], 3), (["job", "logs", "7"], 4)])
def test_a_hand_off_exits_with_the_runners_status(
    runners: _Runners,
    argv: list[str],
    code: int,
) -> None:
    runners.process.code = 3
    runners.job.code = 4
    assert _knowop("login", "acme") == 0

    assert _knowop(*argv) == code


def test_job_refuses_without_a_login_before_any_request(
    server: _Server,
    runners: _Runners,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("job", "logs", "7") == 1

    assert server.requests == []
    assert runners.calls() == []
    assert capsys.readouterr().err == "knowop: not logged in; run knowop login\n"


@pytest.mark.parametrize(
    "argv",
    [
        ["submit", "--minutes", "30", "--", "python", "train.py", "--help"],
        ["logs", "7", "-f"],
        ["results", "7", "--to", "out"],
        ["status", "7"],
        ["cancel", "7"],
    ],
)
def test_job_hands_the_job_runner_knowops_profile_and_the_rest(
    runners: _Runners,
    argv: list[str],
) -> None:
    assert _knowop("login", "acme") == 0

    assert _knowop("job", *argv) == 0

    assert runners.calls() == [("job", ["--profile", "acme.knowop.ai", *argv])]


def test_host_lists_the_hosts_of_knowops_server(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("host") == 0

    assert capsys.readouterr().out.splitlines() == [
        "ada-mac  personal  -  -",
        "gpu1     personal  -  -",
        "lab      personal  -  -",
    ]
    # The KnowHosts it rents, if any, follow the registry's hosts.
    assert server.paths()[-2:] == ["/api/hosts", "/api/addons/knowhost/hosts"]


def test_host_refuses_without_a_login_before_any_request(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("host", "gpu1") == 1

    assert server.requests == []
    assert capsys.readouterr().err == "knowop: not logged in; run knowop login\n"


@pytest.mark.parametrize("argv", [["--help"], ["help"], ["gpu1", "role", "-h"]])
def test_host_help_needs_no_login_and_asks_the_server_nothing(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    assert _knowop("host", *argv) == 0

    assert capsys.readouterr().out.startswith("Usage: knowop host")
    assert server.requests == []


def test_canvas_help_names_its_own_verb_and_needs_no_login(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("canvas", "--help") == 0

    assert capsys.readouterr().out.startswith("Usage: knowop canvas CANVAS_UUID")
    assert server.requests == []


def test_canvas_shows_a_visual_on_the_runs_canvas_on_knowops_server(
    server: _Server,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With knowop alone, a run drives the canvas its ``TRAX_CANVAS`` names."""
    save_profile("prod", _PROD)
    switch_profile("prod")
    monkeypatch.setenv("TRACKINIZER_URL", _PROD.url)
    monkeypatch.setenv("TRAX_CANVAS", _CANVAS)
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    shown = ["show", "trax.subgraph", "--record", _RECORD, "--placement", "side"]
    assert _knowop("canvas", *shown) == 0

    assert server.netlocs() == {"acme.knowop.ai"}
    assert server.paths()[-2:] == [
        f"/api/canvases/{_CANVAS}",
        f"/api/canvases/{_CANVAS}/operations",
    ]
    assert json_loads(server.requests[-1].content, dict[str, object]) == {
        "revision": 3,
        "operation": {
            "kind": "show",
            "visual_type": "trax.subgraph",
            "placement": "side",
            "record_id": _RECORD,
        },
    }
    assert capsys.readouterr().out.splitlines() == [
        f"canvas {_CANVAS} revision 4 focused none",
        f"  trax.subgraph {_VISUAL} side record={_RECORD}",
    ]


def test_canvas_refuses_without_a_login_before_any_request(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("canvas", _CANVAS, "highlight", _RECORD) == 1

    assert server.requests == []
    assert capsys.readouterr().err == "knowop: not logged in; run knowop login\n"


@dataclass(slots=True, kw_only=True)
class _Uploads:
    """Stands for the upload itself: what it was asked, and a session it fails."""

    calls: list[tuple[list[str], str, bool, int, bool]] = field(
        default_factory=list[tuple[list[str], str, bool, int, bool]],
    )
    """Each call's session ids, server, privacy, jobs, and whether it masks."""

    failing: frozenset[str] = frozenset()
    keys: list[str | None] = field(default_factory=list[str | None])
    """The API key each call's client signs in with."""

    def __call__(
        self,
        sessions: Sequence[upload.Session],
        *,
        connect: Callable[[], Client],
        private: bool,
        jobs: int,
        redactor: object,
    ) -> list[upload.Uploaded]:
        with connect() as client:
            server = client.base_url
            self.keys.append(client.api_key)
        self.calls.append(
            (
                [f"{s.format} {s.id}" for s in sessions],
                server,
                private,
                jobs,
                redactor is not None,
            ),
        )
        return [
            upload.Uploaded(session=session, error="refused")
            if session.id in self.failing
            else upload.Uploaded(
                session=session,
                agent_session=uuid.UUID(int=index + 1),
                sent=7,
                held=index,
                refused=index,
                audience="private" if private else "open",
            )
            for index, session in enumerate(sessions)
        ]


@pytest.fixture
def uploads(monkeypatch: pytest.MonkeyPatch) -> _Uploads:
    stand_in = _Uploads()
    monkeypatch.setattr(upload, "upload", stand_in)
    return stand_in


def _transcripts(folder: Path) -> Path:
    """Write a claude session with a sub-agent, and one whose spawner is not here."""
    for name in (
        "s1.jsonl",
        "s1/subagents/agent-a1.jsonl",
        "s2/subagents/agent-b1.jsonl",
    ):
        path = folder / "-w" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"type": "user", "sessionId": "s1", "uuid": "u1"}\n')
    return folder


_JOBS: Final = min(8, os.cpu_count() or 1)
"""How many sessions ``upload`` sends side by side unless told."""

_CLAUDE: Final = ["claude s1", "claude s1/agent-a1", "claude s2/agent-b1"]
"""The sessions ``_transcripts`` writes, as the upload is handed them."""


def test_upload_sends_each_session_to_knowops_server_and_totals_them(
    uploads: _Uploads,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login", "acme") == 0
    folder = _transcripts(tmp_path / "copy")
    capsys.readouterr()

    assert _knowop("upload", str(folder)) == 0

    assert uploads.calls == [(_CLAUDE, _ACME, True, _JOBS, False)]
    assert uploads.keys == [_KEY]
    one, two, three = (uuid.UUID(int=n) for n in (1, 2, 3))
    assert capsys.readouterr().out.splitlines() == [
        f"uploading 3 sessions to {_ACME}",
        f"claude s1: AgentSession {one} (private), sent 7, held 0, 1 sub-agent",
        (
            f"claude s1/agent-a1: AgentSession {two} (private), sent 7, held 1, 1 refused, "
            "0 sub-agents, spawned by s1"
        ),
        (
            f"claude s2/agent-b1: AgentSession {three} (private), sent 7, held 2, 2 refused, "
            "0 sub-agents, spawned by s2, whose file is not here"
        ),
        "uploaded 3 sessions: 21 records sent, 3 already there",
    ]


def test_upload_flags_reach_the_upload(
    uploads: _Uploads,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAX_REDACT_NAMES", "SECRET")
    monkeypatch.setenv("SECRET", "hunter2-hunter2")
    assert _knowop("login", "acme") == 0
    folder = _transcripts(tmp_path)

    assert _knowop("upload", str(folder / "-w"), "--public", "--no-subagents") == 0
    assert _knowop("upload", str(folder), "--private", "--jobs", "1") == 0
    assert _knowop("upload", str(folder), "--format", "codex") == 0

    assert [call[0] for call in uploads.calls] == [
        ["claude s1", "claude s2/agent-b1"],
        _CLAUDE,
        ["codex agent-a1", "codex s1", "codex agent-b1"],
    ]
    assert [call[2:] for call in uploads.calls] == [
        (False, _JOBS, True),
        (True, 1, True),
        (True, _JOBS, True),
    ]


@pytest.mark.parametrize("jobs", ["0", "-1", "two"])
def test_upload_refuses_a_count_of_jobs_below_one(
    jobs: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        _ = _knowop("upload", str(tmp_path), "--jobs", jobs)

    assert f"argument --jobs: {jobs!r} is not a number of 1 or more" in (
        capsys.readouterr().err
    )


def test_upload_sends_one_at_a_time_where_the_cpus_cannot_be_counted(
    uploads: _Uploads,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert _knowop("login", "acme") == 0

    assert _knowop("upload", str(_transcripts(tmp_path))) == 0

    assert [call[3] for call in uploads.calls] == [1]


@pytest.mark.parametrize("read_as", ["claude", "codex", "sagent"])
def test_upload_reads_every_file_as_the_format_named(
    read_as: str,
    uploads: _Uploads,
    tmp_path: Path,
) -> None:
    assert _knowop("login", "acme") == 0
    folder = _transcripts(tmp_path)
    # A sagent session is its folder's ``session.jsonl``, whatever it is read as.
    (folder / "j1").mkdir()
    (folder / "j1" / "session.jsonl").write_text('{"kind": "meta"}\n')

    assert _knowop("upload", str(folder), "--format", read_as) == 0

    assert {name.split()[0] for name in uploads.calls[0][0]} == {read_as}


def test_upload_refuses_a_format_it_does_not_send(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        _ = _knowop("upload", str(tmp_path), "--format", "gemini")

    assert "argument --format: invalid choice: 'gemini'" in capsys.readouterr().err


def test_upload_help_names_each_flag_and_what_it_does(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        _ = _knowop("upload", "--help")

    said = " ".join(capsys.readouterr().out.split())
    for part in (
        "[--private | --public]",
        "[--server ORG|URL]",
        "[--format {claude,codex,sagent}]",
        "[--jobs N]",
        "PATH [PATH ...]",
        "--private open a new session to you and those who oversee (default)",
        "--public open a new session to the organization",
        "sends only what the server lacks, and leaves it to the readers it has.",
    ):
        assert part in said


def test_upload_to_another_server_uses_knowops_saved_login_for_it(
    uploads: _Uploads,
    tmp_path: Path,
) -> None:
    assert _knowop("login", "acme") == 0
    assert _knowop("login", "https://staging.example/") == 0
    folder = _transcripts(tmp_path)

    assert _knowop("upload", str(folder), "--server", "acme") == 0
    assert _knowop("upload", str(folder), "--server", "https://staging.example/") == 0

    assert [call[1] for call in uploads.calls] == [_ACME, "https://staging.example"]


def test_upload_with_a_profile_named_before_it_goes_to_that_login(
    uploads: _Uploads,
    tmp_path: Path,
) -> None:
    save_profile("staging", Profile(url="https://staging.example", api_key=_KEY))
    assert _knowop("login", "acme") == 0

    assert _knowop("--profile", "staging", "upload", str(_transcripts(tmp_path))) == 0

    assert [call[1] for call in uploads.calls] == ["https://staging.example"]


def test_upload_with_a_profile_named_looks_for_another_server_in_that_login_alone(
    uploads: _Uploads,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    save_profile("staging", Profile(url="https://staging.example", api_key=_KEY))
    assert _knowop("login", "acme") == 0
    capsys.readouterr()
    folder = str(_transcripts(tmp_path))

    assert _knowop("--profile", "staging", "upload", folder, "--server", "acme") == 1

    assert capsys.readouterr().err == (
        f"knowop: not logged in to {_ACME}; run knowop login acme\n"
    )
    assert uploads.calls == []


def test_upload_to_a_server_with_no_saved_login_says_to_log_in_and_starts_none(
    server: _Server,
    uploads: _Uploads,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login", "acme") == 0
    server.requests.clear()
    capsys.readouterr()

    assert _knowop("upload", str(_transcripts(tmp_path)), "--server", "lab") == 1

    assert capsys.readouterr().err == (
        "knowop: not logged in to https://lab.knowop.ai; run knowop login lab\n"
    )
    assert server.requests == []
    assert uploads.calls == []


def test_upload_without_a_login_refuses(uploads: _Uploads, tmp_path: Path) -> None:
    assert _knowop("upload", str(_transcripts(tmp_path))) == 1

    assert uploads.calls == []


def test_upload_says_each_failed_session_and_exits_1(
    uploads: _Uploads,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    uploads.failing = frozenset({"s1/agent-a1"})
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("upload", str(_transcripts(tmp_path))) == 1

    captured = capsys.readouterr()
    assert captured.out.splitlines()[2:] == [
        "claude s1/agent-a1: failed: refused",
        (
            f"claude s2/agent-b1: AgentSession {uuid.UUID(int=3)} (private), sent 7, held 2, "
            "2 refused, 0 sub-agents, spawned by s2, whose file is not here"
        ),
        "uploaded 2 sessions: 14 records sent, 2 already there",
    ]
    assert captured.err == "knowop: 1 session failed\n"


def test_upload_dry_run_lists_each_session_and_asks_no_server(
    server: _Server,
    uploads: _Uploads,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    folder = _transcripts(tmp_path)

    assert _knowop("upload", str(folder / "-w" / "s1.jsonl"), "--dry-run") == 0

    root = folder / "-w"
    assert capsys.readouterr().out.splitlines() == [
        f"claude s1  {root / 's1.jsonl'}, 1 sub-agent",
        (
            f"claude s1/agent-a1  {root / 's1/subagents/agent-a1.jsonl'}, 0 sub-agents, "
            "spawned by s1"
        ),
        "would upload 2 sessions; --dry-run sent none",
    ]
    assert server.requests == []
    assert uploads.calls == []


def test_upload_of_a_path_not_there_names_it(
    uploads: _Uploads,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("upload", str(tmp_path), str(tmp_path / "gone")) == 1
    assert _knowop("upload", str(tmp_path / "gone"), str(tmp_path / "lost")) == 1

    assert capsys.readouterr().err == (
        f"knowop: no such file or folder: {tmp_path / 'gone'}\n"
        f"knowop: no such file or folder: {tmp_path / 'gone'}, {tmp_path / 'lost'}\n"
    )
    assert uploads.calls == []


def test_upload_of_a_folder_holding_no_session_says_so(
    uploads: _Uploads,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "history.jsonl").write_text('{"display": "hi", "sessionId": "s"}\n')

    assert _knowop("upload", str(tmp_path)) == 1

    assert capsys.readouterr().err == (
        "knowop: no Claude Code, Codex or sagent session found there\n"
    )
    assert uploads.calls == []


@pytest.mark.parametrize("argv", [["--help"], ["help"], ["show", "-h"]])
def test_canvas_help_is_knowops_needs_no_login_and_asks_the_server_nothing(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    assert _knowop("canvas", *argv) == 0

    out = capsys.readouterr().out
    assert out.startswith("Usage: knowop canvas CANVAS_UUID")
    assert "trax canvas" not in out
    assert server.requests == []


def test_host_help_is_knowops_and_needs_no_login(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("host", "--help") == 0

    assert "knowop host [start | NAME [pause | resume | del" in capsys.readouterr().out
    assert server.requests == []


def test_knowops_help_lists_host_among_its_verbs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exited:
        _knowop("--help")

    assert exited.value.code == 0
    out = capsys.readouterr().out
    assert "    host " in out
    # The module's docstring is the page, its first line after the usage.
    assert out.splitlines()[2] == (
        "Log in to KnowOp, connect this computer, and run recorded agents and GPU jobs."
    )


def test_knowop_reads_its_verb_from_the_command_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["knowop", "status"])

    assert cli.main() == 1

    assert capsys.readouterr().err == "knowop: not logged in; run knowop login\n"


def test_knowops_help_lists_canvas_among_its_verbs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exited:
        _knowop("--help")

    assert exited.value.code == 0
    assert "    canvas " in capsys.readouterr().out


def test_a_first_run_logs_in_to_the_default_server_then_runs(
    server: _Server,
    runners: _Runners,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("run", "--headless") == 0

    assert server.netlocs() == {"alpha.knowop.ai"}
    assert json_loads(server.requests[0].content, dict[str, object]) == {
        "client": "knowop",
        "hostname": "Ada_Mac",
        "host": None,
    }
    assert runners.process.calls == [(["--headless"], "https://alpha.knowop.ai")]
    assert "logged in: profile alpha.knowop.ai (writer)" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("here", "said"),
    [("", True), ("offline", True), ("online", False)],
    ids=["unregistered", "offline", "online"],
)
def test_a_run_says_to_connect_when_this_computer_is_not_online(
    server: _Server,
    runners: _Runners,
    capsys: pytest.CaptureFixture[str],
    here: str,
    said: bool,
) -> None:
    others = [_host("gpu1", status="online", owner=_EMAIL)]
    mine = [_host("ada-mac", status=here, owner=_EMAIL)] if here else []
    server.hosts = [*others, *mine]
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("run") == 0

    hint = (
        "knowop: this computer (ada-mac) is not connected; run knowop connect in "
        "another terminal to open its terminal from Chat\n"
    )
    assert capsys.readouterr().err == (hint if said else "")
    assert runners.calls() == [("run", [])]


def test_a_run_says_nothing_of_connecting_when_the_hosts_cannot_be_listed(
    server: _Server,
    runners: _Runners,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login", "acme") == 0
    capsys.readouterr()
    server.refuse["/api/hosts"] = 403

    assert _knowop("run") == 0

    assert capsys.readouterr().err == ""
    assert runners.calls() == [("run", [])]


def test_connect_with_a_piped_token_joins_without_a_login(
    server: _Server,
    host: _Host,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(f"{_ENROLL_TOKEN}\n"))

    assert _knowop("connect", "acme", "--name", "box-7", "--sandbox") == 0

    assert server.requests == []
    assert host.connects == [
        _Served(
            name="box-7",
            server="https://acme.knowop.ai",
            token=_ENROLL_TOKEN,
            serve_tools=True,
            sandbox=True,
        ),
    ]
    # No login to sign in with: the token alone joins.
    assert [served.signed_in_to for served in host.connects] == [""]
    assert list_profiles() == []


def test_a_piped_token_is_never_sent_over_plain_http_beyond_this_computer(
    server: _Server,
    host: _Host,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(f"{_ENROLL_TOKEN}\n"))

    assert _knowop("connect", "http://knowop.example.org") == 1

    assert host.connects == []
    assert server.requests == []
    assert "plain http" in capsys.readouterr().err


def test_a_piped_token_is_left_to_the_host_when_knowop_is_logged_in_there(
    host: _Host,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _knowop("login", "acme") == 0
    monkeypatch.setattr(sys, "stdin", io.StringIO(f"{_LOGIN_TOKEN}\n"))

    assert _knowop("connect") == 0

    # The host reads stdin itself before it enrolls; knowop hands it no token.
    assert sys.stdin.read() == f"{_LOGIN_TOKEN}\n"
    assert [served.token for served in host.connects] == [_ENROLL_TOKEN]


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_job_help_is_knowops(
    runners: _Runners,
    capsys: pytest.CaptureFixture[str],
    flag: str,
) -> None:
    with pytest.raises(SystemExit) as exited:
        _knowop("job", flag)

    assert exited.value.code == 0
    assert capsys.readouterr().out.startswith("usage: knowop job ")
    assert runners.calls() == []


@pytest.mark.parametrize(
    ("installed", "printed"),
    [({"knowop": "0.1.0"}, "knowop 0.1.0\n"), ({}, "knowop unknown\n")],
    ids=["installed", "source checkout"],
)
def test_version_names_the_command_and_unknown_from_a_checkout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    installed: dict[str, str],
    printed: str,
) -> None:
    monkeypatch.setattr(
        metadata,
        "version",
        partial(_installed_version, installed=installed),
    )

    with pytest.raises(SystemExit) as exited:
        _knowop("--version")

    assert exited.value.code == 0
    assert capsys.readouterr().out == printed


def test_bare_knowop_gives_the_first_step_without_logging_in(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop() == 0

    assert capsys.readouterr() == ("Not connected. Run:  knowop connect\n", "")
    assert server.requests == []


def test_bare_knowop_names_this_connected_computer_and_the_next_step(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop() == 0

    assert capsys.readouterr() == (
        "Connected as ada-mac to https://acme.knowop.ai. Next:  knowop run\n",
        "",
    )


@pytest.mark.parametrize("here", ["offline", "never", "revoked", ""])
def test_bare_knowop_does_not_count_another_computers_connection(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
    here: str,
) -> None:
    server.hosts = [
        _host("gpu1", status="online", owner=_EMAIL),
        *([_host("ada-mac", status=here, owner=_EMAIL)] if here else []),
    ]
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop() == 0

    assert capsys.readouterr() == ("Not connected. Run:  knowop connect\n", "")


def test_bare_knowop_does_not_count_another_persons_computer(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.hosts = [_host("ada-mac", status="online", owner="bob@example.com")]
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop() == 0

    assert capsys.readouterr() == ("Not connected. Run:  knowop connect\n", "")


def test_bare_knowop_reports_a_revoked_login_instead_of_saying_connected(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login", "acme") == 0
    server.revoked = True
    capsys.readouterr()

    assert _knowop() == 1

    out, err = capsys.readouterr()
    assert out == ""
    assert "knowop login" in err


_ACME: Final = "https://acme.knowop.ai"
_CONNECTED: Final = (
    f"connected: host ada-mac at {_ACME} in the background (pid 4243); it starts "
    "again at login"
)


# One read of a terminal returns only the bytes that have crossed it so far: each line
# knowop prints is a write of its own, and on a loaded Linux runner a read can come back
# holding the first line alone.
def _screen_lines(fd: int, *, count: int) -> list[str]:
    """Return the lines printed on the terminal ``fd`` reads, once ``count`` show."""
    printed = b""
    deadline = time.monotonic() + 5.0
    while printed.count(b"\n") < count:
        left = deadline - time.monotonic()
        nothing: list[int] = []
        ready, _, _ = select.select([fd], nothing, nothing, max(left, 0))
        if not ready:
            raise AssertionError(f"{count} lines did not show: {printed!r}")
        printed += os.read(fd, 4096)
    return printed.decode().replace("\r\n", "\n").splitlines()


def test_connect_at_a_terminal_returns_once_its_service_serves_the_host(
    host: _Host,
    launchd: _Launchd,
    agents: Services,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stdout is a real terminal, and knowop comes back while its service serves."""
    assert _knowop("login", "acme") == 0
    primary, secondary = os.openpty()
    with (
        os.fdopen(primary, "rb", buffering=0) as screen,
        os.fdopen(secondary, "w", encoding="utf-8") as tty,
    ):
        monkeypatch.setattr(sys, "stdout", tty)

        assert _knowop("connect", "--sandbox") == 0

        printed = _screen_lines(screen.fileno(), count=3)
    assert printed == [
        _CONNECTED,
        "log:  knowop logs -f",
        "stop: knowop disconnect",
    ]
    # Joined here, so a refusal would have shown here; the service serves from then on.
    assert [(s.join_only, s.token, s.sandbox) for s in host.connects] == [
        (True, _ENROLL_TOKEN, True),
    ]
    assert launchd.started() == ["bootstrap"]
    assert agents.every() == [
        Connect(name="ada-mac", server=_ACME, flags=("--sandbox",)),
    ]
    # The menu bar shows it from then on.
    assert launchd.menubar[-1] == "bootstrap"


@pytest.mark.usefixtures("host", "launchd")
@pytest.mark.parametrize(
    ("given", "flags"),
    [
        ((), ()),
        (("--no-exec",), ("--no-exec",)),
        (("--sandbox", "--no-exec"), ("--sandbox", "--no-exec")),
        (("--jobs",), ("--jobs", "--job-results-max", "256M")),
        (
            ("--jobs", "--job-results-max", "1G"),
            ("--jobs", "--job-results-max", "1G"),
        ),
        (("--job-results-max", "1G"), ()),
    ],
    ids=["none", "no-exec", "sandbox no-exec", "jobs", "jobs max", "max alone"],
)
def test_the_service_serves_with_the_flags_connect_was_given(
    agents: Services,
    monkeypatch: pytest.MonkeyPatch,
    given: tuple[str, ...],
    flags: tuple[str, ...],
) -> None:
    """Each flag reaches the service's own run as given; a size alone asks for no jobs."""
    _ = _at_terminal(monkeypatch)
    assert _knowop("login", "acme") == 0

    assert _knowop("connect", *given) == 0

    assert agents.every() == [Connect(name="ada-mac", server=_ACME, flags=flags)]


@pytest.mark.parametrize(("platform", "code"), [("darwin", 0), ("linux", 1)])
def test_the_menu_bar_runs_on_macos_alone(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    platform: str,
    code: int,
) -> None:
    shown: list[str] = []
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(menubar, "run", lambda: shown.append("menu bar"))

    assert _knowop("menubar") == code

    assert shown == (["menu bar"] if code == 0 else [])
    assert capsys.readouterr().err == (
        "" if code == 0 else "knowop: knowop menubar runs on macOS alone\n"
    )


def test_connect_again_leaves_a_current_service_serving(
    host: _Host,
    launchd: _Launchd,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = _at_terminal(monkeypatch)
    assert _knowop("login", "acme") == 0
    assert _knowop("connect") == 0

    assert _knowop("connect") == 0

    assert launchd.started() == ["bootstrap"]
    assert len(host.connects) == 1
    assert terminal.lines()[-3] == _CONNECTED.replace(
        "connected:",
        "connected already:",
    )
    # A menu bar quit since starts again.
    assert launchd.menubar[-1] == "kickstart"


def test_connect_says_why_its_python_cannot_show_the_menu_bar_and_installs_none(
    host: _Host,
    launchd: _Launchd,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A checkout's environment synced before PyObjC was added, as in Dan's."""
    del host
    stand_in = Services(
        manager="launchd",
        folder=tmp_path / "agents",
        run=launchd,
        menubar_problem=lambda: "No module named 'AppKit'",
    )
    monkeypatch.setattr(service, "here", lambda: stand_in)
    terminal = _at_terminal(monkeypatch)
    assert _knowop("login", "acme") == 0

    assert _knowop("connect") == 0

    assert terminal.lines()[-3:] == [
        _CONNECTED,
        "log:  knowop logs -f",
        "stop: knowop disconnect",
    ]
    assert capsys.readouterr().err == (
        f"knowop: no menu bar: No module named 'AppKit' in {sys.executable}; install "
        "PyObjC there (in a checkout, uv sync), then run knowop connect again\n"
    )
    assert launchd.menubar == []


def test_status_says_how_the_menu_bar_is(
    agents: Services,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agents.start(Connect(name="ada-mac", server=_ACME))
    agents.show_menu_bar()
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("status") == 0

    # The stand-in launchd prints nothing of the agent: it has not run.
    assert capsys.readouterr().out.splitlines()[-1] == (
        "menubar: stopped; knowop connect starts it"
    )


def test_connect_with_other_flags_reloads_its_service(
    host: _Host,
    launchd: _Launchd,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = _at_terminal(monkeypatch)
    assert _knowop("login", "acme") == 0
    assert _knowop("connect") == 0

    assert _knowop("connect", "--no-exec") == 0

    assert [call[1] for call in launchd.calls[-3:]] == ["print", "bootout", "bootstrap"]
    assert [served.serve_tools for served in host.connects] == [True, False]
    assert terminal.lines()[-3] == _CONNECTED.replace("4243", "4244")


@pytest.mark.usefixtures("host")
def test_connect_restarts_a_service_whose_knowop_was_upgraded_since(
    server: _Server,
    launchd: _Launchd,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = _at_terminal(monkeypatch)
    assert _knowop("login", "acme") == 0
    assert _knowop("connect") == 0
    server.facts["ada-mac"]["knowop_version"] = "0.0.1"

    assert _knowop("connect") == 0

    assert launchd.started() == ["bootstrap", "kickstart"]
    assert terminal.lines()[-3] == _CONNECTED.replace("4243", "4244")


def test_connect_starts_nothing_where_another_connect_serves_the_host(
    server: _Server,
    host: _Host,
    launchd: _Launchd,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = _at_terminal(monkeypatch)
    server.serves("ada-mac", pid=777)
    assert _knowop("login", "acme") == 0

    assert _knowop("connect") == 0

    assert (host.connects, launchd.calls) == ([], [])
    assert terminal.lines()[-1] == (
        f"connected already: host ada-mac at {_ACME}, by knowop pid 777, which no "
        "service of this computer runs; stop that connect, then run knowop connect "
        "again to serve it from one"
    )


def test_connect_names_a_host_new_to_the_server_in_what_to_run_next(
    host: _Host,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = _at_terminal(monkeypatch)
    assert _knowop("login", "acme") == 0

    assert _knowop("connect", "--name", "newbox") == 0

    assert [served.name for served in host.connects] == ["newbox"]
    assert terminal.lines()[-2:] == [
        "log:  knowop logs -f newbox",
        "stop: knowop disconnect newbox",
    ]


def test_connect_as_root_at_a_terminal_installs_nothing_but_serves_in_the_foreground(
    server: _Server,
    host: _Host,
    launchd: _Launchd,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    terminal = _at_terminal(monkeypatch)
    monkeypatch.setattr(os, "geteuid", lambda: 0)

    assert _knowop("connect") == 1

    assert server.requests == []
    assert capsys.readouterr().err == (
        "knowop: knowop connect would install a service of root, which serves root's "
        "shells; run it as your own user, or with --foreground\n"
    )
    assert _knowop("connect", "--foreground") == 0
    assert [served.join_only for served in host.connects] == [False]
    assert launchd.calls == []
    assert "BCDF-GHJK" in "\n".join(terminal.lines())


@pytest.mark.parametrize(
    "argv",
    [["connect", "--foreground"], ["connect", "acme", "--name", "box-7"]],
    ids=["foreground", "piped token"],
)
def test_a_foreground_or_piped_connect_at_a_terminal_serves_here(
    host: _Host,
    launchd: _Launchd,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    terminal = _at_terminal(monkeypatch)
    if "--foreground" in argv:
        assert _knowop("login", "acme") == 0
    else:
        monkeypatch.setattr(sys, "stdin", io.StringIO(f"{_ENROLL_TOKEN}\n"))

    assert _knowop(*argv) == 0

    assert [(served.join_only, served.token) for served in host.connects] == [
        (False, _ENROLL_TOKEN),
    ]
    assert launchd.calls == []
    assert not any(line.startswith("connected") for line in terminal.lines())


def test_a_terminal_with_no_service_manager_serves_in_the_foreground_and_says_why(
    host: _Host,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _at_terminal(monkeypatch)
    monkeypatch.setattr(service, "here", lambda: None)
    assert _knowop("login", "acme") == 0

    assert _knowop("connect") == 0

    assert [served.join_only for served in host.connects] == [False]
    assert capsys.readouterr().err == (
        "knowop: no user service manager answers here (systemctl --user), so this "
        "terminal serves the host until Ctrl-C\n"
    )


@pytest.mark.usefixtures("host")
def test_a_service_that_never_comes_online_is_left_running_and_said(
    server: _Server,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _at_terminal(monkeypatch)
    silent = _Launchd(server, bootstrap=(0, ""))
    agents = Services(manager="launchd", folder=tmp_path / "agents", run=silent)
    monkeypatch.setattr(service, "here", lambda: agents)
    assert _knowop("login", "acme") == 0

    assert _knowop("connect") == 1

    assert capsys.readouterr().err == (
        "knowop: host ada-mac did not come online in 30 s; see knowop logs\n"
    )
    assert server.paths().count("/api/hosts/ada-mac") == 31
    assert len(agents.every()) == 1


def test_the_join_here_is_not_taken_for_the_service_serving(
    server: _Server,
    host: _Host,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _at_terminal(monkeypatch)
    # The join here heartbeats once, so the host reads online with this process.
    host.joined = lambda name: server.serves(name, pid=os.getpid())
    silent = _Launchd(server, bootstrap=(0, ""))
    agents = Services(manager="launchd", folder=tmp_path / "agents", run=silent)
    monkeypatch.setattr(service, "here", lambda: agents)
    assert _knowop("login", "acme") == 0

    assert _knowop("connect") == 1

    assert capsys.readouterr().err == (
        "knowop: host ada-mac did not come online in 30 s; see knowop logs\n"
    )


@pytest.mark.usefixtures("host")
def test_connect_at_a_terminal_says_when_the_server_will_not_show_the_host(
    server: _Server,
    launchd: _Launchd,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _at_terminal(monkeypatch)
    assert _knowop("login", "acme") == 0
    server.refuse["/api/hosts/ada-mac"] = 403

    assert _knowop("connect") == 1

    assert launchd.calls == []
    assert "GET /api/hosts/ada-mac -> 403" in capsys.readouterr().err


@pytest.mark.usefixtures("host")
def test_a_systemd_service_says_how_to_outlive_a_logout_when_lingering_is_off(
    server: _Server,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    terminal = _at_terminal(monkeypatch)
    systemctl = _Launchd(server, show_user=(0, "Linger=no\n"))
    units = Services(manager="systemd", folder=tmp_path / "units", run=systemctl)
    monkeypatch.setattr(service, "here", lambda: units)
    monkeypatch.setattr(getpass, "getuser", lambda: "ada")
    assert _knowop("login", "acme") == 0

    assert _knowop("connect") == 0

    assert [call[2] for call in systemctl.calls if call[0] == "systemctl"] == [
        "daemon-reload",
        "enable",
        "reset-failed",
        "restart",
    ]
    assert terminal.lines()[-1] == (
        "keep it running after you log out: sudo loginctl enable-linger ada"
    )


@pytest.mark.usefixtures("host")
def test_connect_follow_follows_the_services_log_until_ctrl_c(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = _at_terminal(monkeypatch)
    folder = state_dir() / "rekursiv-ai" / "knowop" / "connect"
    waits: list[float] = []

    def sleep(seconds: float) -> None:
        waits.append(seconds)
        if len(waits) > 1:
            raise KeyboardInterrupt
        with (folder / f"ada-mac-{service.server_tag(_ACME)}.log").open("a") as file:
            file.write("serving: host ada-mac\n")

    assert _knowop("login", "acme") == 0

    assert cli.main(["connect", "--follow"], sleep=sleep, clock=_Time().clock) == 130

    assert terminal.lines()[-1] == "serving: host ada-mac"
    assert waits == [0.5, 0.5]


def test_the_services_own_run_rotates_its_log_while_it_serves(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(serve, "connect_host", _Host(waits=1))
    log = tmp_path / "ada-mac.log"
    log.write_bytes(b"x" * 5 * 2**20)
    assert _knowop("login", "acme") == 0

    assert _knowop("connect", _ACME, "--name", "ada-mac", "--log-file", str(log)) == 0

    assert log.stat().st_size == 0
    assert log.with_name("ada-mac.log.1").stat().st_size == 5 * 2**20


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (400, 0),
        (401, 0),
        (403, 0),
        (404, 0),
        (410, 0),
        (499, 0),
        (408, 1),
        (429, 1),
        (500, 1),
        (None, 1),
    ],
)
def test_the_services_own_run_exits_0_on_a_refusal_no_restart_can_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    status: int | None,
    code: int,
) -> None:
    """Launchd and systemd start again what exits nonzero, and nothing that exits 0."""
    refused = ClientError("POST /heartbeat -> refused", status_code=status)
    monkeypatch.setattr(serve, "connect_host", _Host(ending=refused))
    assert _knowop("login", "acme") == 0
    log = str(tmp_path / "ada-mac.log")
    capsys.readouterr()

    assert _knowop("connect", _ACME, "--name", "ada-mac", "--log-file", log) == code
    # The service's run says why it stopped, or main does as it exits 1.
    assert capsys.readouterr().err == (
        "knowop: POST /heartbeat -> refused\n"
        + ("knowop: stopped; knowop connect starts it again\n" if code == 0 else "")
    )
    # Outside a service, every refusal still exits 1.
    assert _knowop("connect") == 1


def test_the_services_own_run_never_logs_in(
    server: _Server,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = str(tmp_path / "ada-mac.log")

    assert _knowop("connect", _ACME, "--name", "ada-mac", "--log-file", log) == 0

    assert server.requests == []
    assert capsys.readouterr().err == (
        f"knowop: not logged in to {_ACME}; run knowop connect there\n"
    )


def test_a_service_serves_with_its_own_servers_login_after_another_server_s(
    host: _Host,
    tmp_path: Path,
) -> None:
    assert _knowop("login", "acme") == 0
    assert _knowop("connect", "http://127.0.0.1:8790") == 0
    log = str(tmp_path / "ada-mac.log")

    assert _knowop("connect", _ACME, "--name", "ada-mac", "--log-file", log) == 0

    assert [(served.server, served.signed_in_to) for served in host.connects] == [
        ("http://127.0.0.1:8790", "http://127.0.0.1:8790"),
        (_ACME, _ACME),
    ]
    # Its run at login leaves the login knowop's verbs use as it was.
    assert _saved_logins() == ["127.0.0.1-8790", "acme.knowop.ai"]


def test_connect_at_a_terminal_to_a_server_knowop_is_logged_in_to_uses_that_login(
    server: _Server,
    host: _Host,
) -> None:
    assert _knowop("login", "acme") == 0
    assert _knowop("connect", "http://127.0.0.1:8790") == 0

    assert _knowop("connect", "acme") == 0

    assert server.paths().count(DEVICE_CODE_PATH) == 2
    assert [served.signed_in_to for served in host.connects][-1] == _ACME
    assert _saved_logins() == ["acme.knowop.ai", "127.0.0.1-8790"]


def test_status_shows_each_server_knowop_is_logged_in_to_the_one_in_use_first(
    host: _Host,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del host
    assert _knowop("login", "acme") == 0
    assert _knowop("connect", "http://127.0.0.1:8790") == 0
    capsys.readouterr()

    assert _knowop("status") == 0

    profiles = [
        line for line in capsys.readouterr().out.splitlines() if "profile:" in line
    ]
    assert profiles == ["profile: 127.0.0.1-8790", "profile: acme.knowop.ai"]


def test_status_shows_every_login_and_names_the_one_whose_key_was_refused(
    server: _Server,
    host: _Host,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del host
    assert _knowop("login", "acme") == 0
    assert _knowop("connect", "http://127.0.0.1:8790") == 0
    server.revoked = True
    capsys.readouterr()

    assert _knowop("status") == 1

    out, err = capsys.readouterr()
    assert out.splitlines() == [
        "profile: 127.0.0.1-8790",
        "server:  http://127.0.0.1:8790",
        "",
        "profile: acme.knowop.ai",
        "server:  https://acme.knowop.ai",
    ]
    assert err == (
        "knowop: not logged in: http://127.0.0.1:8790 refused the key, run knowop "
        "login http://127.0.0.1:8790; https://acme.knowop.ai refused the key, run "
        "knowop login https://acme.knowop.ai\n"
    )


def test_logout_keeps_knowops_login_to_another_server_and_says_it_is_in_use(
    host: _Host,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del host
    assert _knowop("login", "acme") == 0
    assert _knowop("connect", "http://127.0.0.1:8790") == 0
    capsys.readouterr()

    assert _knowop("logout") == 0

    assert [name for name, _ in list_profiles()] == ["acme.knowop.ai"]
    assert _saved_logins() == ["acme.knowop.ai"]
    assert capsys.readouterr().out.splitlines()[-1] == (
        "knowop now uses: profile acme.knowop.ai (https://acme.knowop.ai)"
    )


def test_a_profile_named_before_the_verb_is_the_login_that_verb_uses(
    server: _Server,
    runners: _Runners,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    save_profile("staging", Profile(url="https://staging.example", api_key=_KEY))
    save_profile("prod", _PROD)
    # Every connected shell sets it; knowop reads it never.
    monkeypatch.setenv("TRACKINIZER_PROFILE", "prod")
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("--profile", "staging", "status") == 0
    assert _knowop("--profile=staging", "run", "--headless") == 0
    # One named after the verb is the runner's own, which reads the last.
    assert _knowop("--profile", "staging", "job", "logs", "--profile", "other") == 0

    assert capsys.readouterr().out.splitlines()[:2] == [
        "profile: staging",
        "server:  https://staging.example",
    ]
    assert runners.process.calls == [(["--headless"], "https://staging.example")]
    assert runners.job.calls == [
        ["--profile", "staging", "logs", "--profile", "other"],
    ]
    assert server.netlocs() == {"acme.knowop.ai", "staging.example"}
    # Without the flag, knowop uses the login it remembered, as before.
    assert _saved_logins() == ["acme.knowop.ai"]


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["disconnect", "gpu1"],
        ["logout"],
        ["run"],
        ["host"],
        ["canvas", _CANVAS],
    ],
    ids=["next step", "disconnect", "logout", "run", "host", "canvas"],
)
@pytest.mark.usefixtures("runners")
def test_every_verb_asks_the_server_of_the_profile_named_before_it(
    server: _Server,
    argv: list[str],
) -> None:
    save_profile("staging", Profile(url="https://staging.example", api_key=_KEY))
    assert _knowop("login", "acme") == 0
    server.requests.clear()

    _ = _knowop("--profile", "staging", *argv)

    assert server.netlocs() == {"staging.example"}


def test_logs_reads_the_log_of_the_profile_named_before_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    save_profile("staging", Profile(url="https://staging.example", api_key=_KEY))
    assert _knowop("login", "acme") == 0
    _ = _connect_log("ada-mac")
    capsys.readouterr()

    assert _knowop("--profile", "staging", "logs") == 1

    assert capsys.readouterr().err.startswith(
        "knowop: no log of host ada-mac on this computer",
    )


def test_a_first_run_under_a_named_profile_logs_in_as_it(
    server: _Server,
    runners: _Runners,
) -> None:
    """A login knowop keeps already is not the named profile's."""
    assert _knowop("login", "acme") == 0

    assert _knowop("--profile", "fresh", "run") == 0

    assert [name for name, _ in list_profiles()] == ["acme.knowop.ai", "fresh"]
    assert server.paths().count(DEVICE_CODE_PATH) == 2
    assert runners.process.calls == [([], read_profile("fresh").url)]


def test_a_connect_under_a_named_profile_logs_in_and_saves_it_under_that_name(
    host: _Host,
) -> None:
    assert _knowop("--profile", "svc", "connect") == 0

    assert [name for name, _ in list_profiles()] == ["svc"]
    assert read_profile("svc").url == "https://alpha.knowop.ai"
    assert [(served.server, served.token) for served in host.connects] == [
        ("https://alpha.knowop.ai", _LOGIN_TOKEN),
    ]


def test_a_profile_that_is_not_saved_is_no_login(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("--profile", "nope", "status") == 1

    assert capsys.readouterr().err == (
        "knowop: not logged in; run knowop --profile nope login\n"
    )
    assert server.paths().count("/api/me/profile") == 0


def test_a_login_under_a_named_profile_saves_it_under_that_name_and_uses_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("--profile", "mine", "login", "acme") == 0

    assert [name for name, _ in list_profiles()] == ["mine"]
    assert read_profile("mine").url == _ACME
    assert _saved_logins() == ["mine"]
    assert "logged in: profile mine (writer)\n" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["--host", "--port"])
def test_traxs_other_connection_flags_are_refused_before_the_verb(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
    flag: str,
) -> None:
    assert _knowop(flag, "8765", "status") == 1

    assert capsys.readouterr().err == (
        "knowop: knowop reads --profile alone before its verb; --host and --port "
        "are trax's\n"
    )
    assert server.requests == []


def test_a_bad_flag_before_the_verb_is_refused_as_knowops(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exited:
        _knowop("--port", "x", "status")

    assert exited.value.code == 2
    assert capsys.readouterr().err.splitlines()[-1] == (
        "knowop: error: argument --port: invalid int value: 'x'"
    )
    assert server.requests == []


def test_run_serve_hands_the_agent_process_its_arguments_and_no_profile(
    server: _Server,
    runners: _Runners,
) -> None:
    runners.process.code = 4

    argv = ["--serve", "https://know.example", "--model", "opus"]
    assert _knowop("--profile", "staging", "run", *argv) == 4

    assert runners.process.calls == [(argv, None)]
    assert server.requests == []
    assert runners.job.calls == []


def test_connect_starts_no_agents_for_other_accounts(
    host: _Host,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A person's knowop serves their own computer; remote agents are no flag of it."""
    with pytest.raises(SystemExit) as exited:
        _knowop("connect", "--shared")

    assert exited.value.code == 2
    assert "unrecognized arguments: --shared" in capsys.readouterr().err
    assert host.connects == []


_CONNECT_HELP: Final = """\
usage: knowop connect [-h] [--name NAME] [--no-exec] [--sandbox] [--jobs] [--job-results-max SIZE] [--foreground] [--follow] [ORG|URL]

Connect this computer and serve its terminals, and its tools to the server's assistant. At a terminal, join here, then serve it from a user service (launchd on macOS, systemd on Linux) that keeps running in the background and starts again at login; knowop logs -f follows it and knowop disconnect stops it. Elsewhere, and with --foreground, serve here until Ctrl-C. When knowop is not logged in to
the server, log in first: the same approval enrolls and confirms this computer. An enrollment token piped to stdin joins instead, with no login, in the foreground.

positional arguments:
  ORG|URL               as for login; default the server knowop is logged in to, else https://alpha.knowop.ai

options:
  -h, --help            show this help message and exit
  --name NAME           the host's name; default this computer's short host name
  --no-exec             serve terminals only: run no tool and start no agent here for the server
  --sandbox             record that this computer is disposable
  --jobs                also run the jobs your account sends here with knowop job submit --host NAME, each in a fresh worktree of its pushed commit under a memory cap; needs systemd's user manager
  --job-results-max SIZE
                        with --jobs: the most result bytes one job sends back (default 256M); larger results stay on this computer
  --foreground          serve in this terminal until Ctrl-C, installing no service
  --follow              once the service runs, follow its log until Ctrl-C
"""
"""``knowop connect --help`` on a terminal wide enough to wrap none of it."""


def test_connect_help_names_each_flag_and_what_it_does(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("COLUMNS", "400")
    # Python 3.14's argparse colours its help where the environment asks it to.
    monkeypatch.setenv("PYTHON_COLORS", "0")

    with pytest.raises(SystemExit) as exited:
        _knowop("connect", "--help")

    assert exited.value.code == 0
    assert capsys.readouterr().out == _CONNECT_HELP


def _saved_logins() -> list[str]:
    """Return the profiles knowop records as its logins, the one in use first."""
    return (config_dir() / "rekursiv-ai" / "knowop" / "profile").read_text().split()


def _connect_log(name: str) -> Path:
    """Write the log a connect service of ``name`` on acme keeps here, and return it."""
    folder = state_dir() / "rekursiv-ai" / "knowop" / "connect"
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    log = folder / f"{name}-{service.server_tag(_ACME)}.log"
    log.write_text("serving: one\njoined: two\nretrying: three\n")
    return log


@pytest.mark.parametrize(
    ("argv", "printed"),
    [
        ([], "serving: one\njoined: two\nretrying: three\n"),
        (["-n", "2"], "joined: two\nretrying: three\n"),
        (["gpu1", "-n", "0"], ""),
    ],
)
def test_logs_prints_a_hosts_log_here_or_its_last_lines(
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    printed: str,
) -> None:
    _connect_log("ada-mac")
    _connect_log("gpu1")
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("logs", *argv) == 0

    assert capsys.readouterr().out == printed


def test_logs_follow_reads_on_from_the_start_of_a_rotated_log(
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = _connect_log("ada-mac")
    assert _knowop("login", "acme") == 0
    capsys.readouterr()
    waits: list[float] = []

    def sleep(seconds: float) -> None:
        waits.append(seconds)
        if len(waits) == 1:
            with log.open("a") as file:
                file.write("taken over: four\n")
        elif len(waits) == 2:
            # As the host rotates it: copied aside, then emptied where it is written.
            os.truncate(log, 0)
            with log.open("a") as file:
                file.write("five\n")
        else:
            raise KeyboardInterrupt

    assert cli.main(["logs", "-f"], sleep=sleep, clock=_Time().clock) == 130

    assert capsys.readouterr().out.splitlines()[-2:] == ["taken over: four", "five"]


def test_logs_of_a_host_connected_elsewhere_say_where_its_log_is(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("logs", "gpu1") == 1

    assert capsys.readouterr().err == (
        "knowop: no log of host gpu1 on this computer: its log is on the computer "
        "whose service connects it, and a connect in the foreground prints to its "
        "terminal\n"
    )


def test_status_shows_the_version_and_process_of_an_online_host_alone(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.facts["ada-mac"] = {"pid": 4242, "knowop_version": "0.1.6"}
    server.facts["gpu1"] = {"pid": 9, "knowop_version": "0.1.5"}
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("status") == 0

    assert capsys.readouterr().out.splitlines()[4:] == [
        "host: ada-mac online, knowop 0.1.6, pid 4242 (this computer)",
        "host: gpu1 offline",
    ]


def test_status_lists_this_computers_services_and_one_turned_off(
    launchd: _Launchd,
    agents: Services,
    capsys: pytest.CaptureFixture[str],
) -> None:
    flags = ("--sandbox", "--no-exec")
    agents.start(Connect(name="ada-mac", server=_ACME, flags=flags))
    agents.start(Connect(name="gpu1", server=_ACME))
    gpu1 = f"ai.rekursiv.knowop.connect.gpu1.{service.server_tag(_ACME)}"
    launchd.answers["print_disabled"] = (0, f'\t"{gpu1}" => disabled\n')
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("status") == 0

    assert capsys.readouterr().out.splitlines()[-2:] == [
        f"service: ada-mac {_ACME} --sandbox --no-exec",
        (
            f"service: gpu1 {_ACME}; turned off in Login Items: turn it on in System "
            "Settings > General > Login Items & Extensions"
        ),
    ]


def test_status_names_how_systemd_turns_a_disabled_service_on(
    server: _Server,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    unit = f"knowop-connect-ada-mac-{service.server_tag(_ACME)}.service"
    systemctl = _Launchd(server, list_unit_files=(0, f"{unit} disabled enabled\n"))
    units = Services(manager="systemd", folder=tmp_path / "units", run=systemctl)
    units.start(Connect(name="ada-mac", server=_ACME))
    monkeypatch.setattr(service, "here", lambda: units)
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("status") == 0

    assert capsys.readouterr().out.splitlines()[-1] == (
        f"service: ada-mac {_ACME}; disabled: knowop connect enables it again"
    )


def test_status_json_holds_each_login_and_the_services(
    server: _Server,
    agents: Services,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agents.start(Connect(name="ada-mac", server=_ACME, flags=("--sandbox",)))
    started = "2026-10-27T12:00:00+00:00"
    server.facts["ada-mac"] = {
        "pid": 4242,
        "knowop_version": "0.1.6",
        "started_at": started,
    }
    server.runs["gpu1"] = [_run("chat-1a2b3c4d")]
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("status", "--json") == 0

    assert json_loads(capsys.readouterr().out) == {
        "logins": [
            {
                "profile": "acme.knowop.ai",
                "server": _ACME,
                "account": _EMAIL,
                "role": "writer",
                "hosts": [
                    {
                        "name": "ada-mac",
                        "status": "online",
                        "this_computer": True,
                        "knowop_version": "0.1.6",
                        "pid": 4242,
                        "started_at": started,
                        "runs": [],
                    },
                    {
                        "name": "gpu1",
                        "status": "offline",
                        "this_computer": False,
                        "knowop_version": None,
                        "pid": None,
                        "started_at": None,
                        "runs": [
                            {
                                "run": "chat-1a2b3c4d",
                                "cli": "claude",
                                "cwd": "~/work",
                                "session_id": "11111111-1111-1111-1111-111111111111",
                            },
                        ],
                    },
                ],
            },
        ],
        "services": [
            {"name": "ada-mac", "server": _ACME, "flags": ["--sandbox"]},
        ],
        "menubar": None,
    }


def test_status_shows_a_run_without_a_name_by_its_short_session(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.runs["gpu1"] = [{**_run("x"), "run": None}]
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("status") == 0

    assert "  run:   11111111  claude  ~/work" in capsys.readouterr().out.splitlines()


def test_logout_of_one_host_keeps_this_computers_login(
    server: _Server,
    agents: Services,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agents.start(Connect(name="gpu1", server=_ACME))
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("logout", "gpu1") == 0

    assert server.logged_out == ["gpu1"]
    assert not any(path.endswith("/revoke") for path in server.paths())
    assert agents.every() == []
    assert [name for name, _ in list_profiles()] == ["acme.knowop.ai"]
    assert capsys.readouterr().out.splitlines() == [
        "logged out: host gpu1; runs stopped: none; keys revoked: 0",
        "removed: the connect service of host gpu1 on this computer",
    ]


def test_logout_of_a_host_that_is_not_yours_says_the_server_refused(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login", "acme") == 0
    server.refuse["/api/hosts/lab/logout"] = 404

    assert _knowop("logout", "lab") == 1

    assert "404" in capsys.readouterr().err
    assert [name for name, _ in list_profiles()] == ["acme.knowop.ai"]


def test_logout_all_logs_out_every_host_of_yours_then_this_computer(
    server: _Server,
    agents: Services,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agents.start(Connect(name="ada-mac", server=_ACME))
    agents.start(Connect(name="ada-mac", server="https://other.example"))
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("logout", "--all") == 0

    assert server.logged_out == ["ada-mac", "gpu1"]
    assert list_profiles() == []
    # Another server's login is not this one: its service stays.
    assert [each.server for each in agents.every()] == ["https://other.example"]
    assert capsys.readouterr().out.splitlines()[-2:] == [
        "removed: the connect service of host ada-mac on this computer",
        "logged out: revoked its key; deleted profile acme.knowop.ai",
    ]


def test_logout_all_logs_out_the_host_this_login_was_made_for_last(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.key_host = "ada-mac"
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("logout", "--all") == 0

    # Its logout revokes the key every logout before it signs with.
    assert server.logged_out == ["gpu1", "ada-mac"]
    assert list_profiles() == []
    assert capsys.readouterr().out.splitlines() == [
        "logged out: host gpu1; runs stopped: none; keys revoked: 0",
        "logged out: host ada-mac; runs stopped: none; keys revoked: 1",
        "logged out: revoked its key; deleted profile acme.knowop.ai",
    ]


def test_logout_refuses_a_name_and_all_at_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exited:
        _knowop("logout", "gpu1", "--all")

    assert exited.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_disconnect_removes_this_computers_service_once_the_server_lets_go(
    server: _Server,
    launchd: _Launchd,
    agents: Services,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agents.start(Connect(name="ada-mac", server=_ACME))
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("disconnect") == 0

    assert agents.every() == []
    assert launchd.calls[-1][1] == "bootout"
    assert server.server_status("ada-mac") == "revoked"
    assert capsys.readouterr().out.splitlines()[0] == (
        "removed: the connect service of host ada-mac on this computer"
    )


def test_disconnect_says_when_the_host_still_reads_online(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.lingering.add("gpu1")
    server.serves("gpu1", pid=9)
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("disconnect", "gpu1") == 1

    assert capsys.readouterr().err == (
        "knowop: host gpu1 still reads online after 30 s\n"
    )


def _knowop(*argv: str) -> int:
    """Run ``knowop`` with ``argv``, its login polling on a clock that never waits."""
    time = _Time()
    return cli.main(argv, sleep=time.sleep, clock=time.clock)


# Called in the test's body: pytest puts its own capture back on stdout once a test
# starts, after every fixture.
def _at_terminal(monkeypatch: pytest.MonkeyPatch) -> _Terminal:
    """Put stdout on a terminal, where a connect installs its service."""
    screen = _Terminal()
    monkeypatch.setattr(sys, "stdout", screen)
    return screen


def _handler_before(signum: int, frame: object) -> None:
    """Stand for the handler a signal had before ``connect``, to be found again after."""
    del signum, frame


def _never_runs(name: str) -> Callable[..., int]:
    """Return what stands in for ``name`` in a test that must not reach it."""

    def refuse(*args: object, **kwargs: object) -> int:
        raise AssertionError(f"a test ran {name} with {args} {kwargs}")

    return refuse


def _host(name: str, *, status: str, owner: str) -> dict[str, object]:
    return {
        "name": name,
        "role": "personal",
        "how": "",
        "labels": [],
        "updated_by": owner,
        "updated": "2026-10-22T12:00:00Z",
        "status": status,
        "last_heartbeat": None,
        "owner": owner,
    }


def _run(name: str) -> dict[str, object]:
    """Return a live claude run, as a host's runs route lists it."""
    return {
        "session_id": "11111111-1111-1111-1111-111111111111",
        "cli": "claude",
        "run": name,
        "cwd": "~/work",
        "state": "idle",
    }


def _server_transport(server: _Server) -> Callable[..., httpx2.MockTransport]:
    """Return what stands in for ``httpx2.HTTPTransport``: it answers from ``server``."""

    def make(**options: object) -> httpx2.MockTransport:
        del options
        return httpx2.MockTransport(server)

    return make


def _installed_version(name: str, *, installed: dict[str, str]) -> str:
    """Answer as ``importlib.metadata.version`` would with only ``installed`` present."""
    if name not in installed:
        raise metadata.PackageNotFoundError(name)
    return installed[name]


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
