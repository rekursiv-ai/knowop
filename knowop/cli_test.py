"""The ``knowop`` command, against a scripted server and a clock that never waits.

The machine host, the agent runners and the job runner are stood in for: each test
reads what knowop handed them.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import io
import os
import signal
import socket
import sys
import time

from trackinizer.lib.userdirs import config_dir
from trackinizer.trax.profile import (
    Profile,
    current_profile,
    del_profile,
    list_profiles,
    read_profile,
    save_profile,
    switch_profile,
)
from trackinizer.wire.wire_device import (
    DEVICE_CODE_PATH,
    DEVICE_POLL_PATH,
    DeviceCode,
    DeviceToken,
    IssuedKey,
)
from trackinizer.wire.wire_machine_host import ENROLL_PATH
from treekle import json_loads

import httpx2
import pytest

from knowop import cli
from knowop.agent import (
    cli as agent_cli,
    detach,
)
from knowop.agent.detach import HostRecord
from knowop.jobs import cli as jobs_cli
from knowop.shell import serve


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
    """One scripted server behind every host: device login, the caller, machines."""

    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self.decision: Literal["approved", "denied"] = "approved"
        self.revoked = False
        """Whether the key was revoked: every authenticated route answers 401."""
        self.refuse: dict[str, int] = {}
        """The status each listed path answers with, in place of its answer."""
        self.machines = [
            _machine("ada-mac", status="online", owner=_EMAIL),
            _machine("gpu1", status="offline", owner=_EMAIL),
            _machine("lab", status="online", owner="bob@example.com"),
        ]
        """What ``/api/machines`` lists: the caller's two, and another person's."""
        self.machine: object = None
        self.runs: dict[str, list[dict[str, object]]] = {}
        """Each machine's live runs, as its runs route lists them."""
        self.disconnected: list[str] = []
        """Each machine disconnected, in order."""
        self.logged_out: list[str] = []
        """Each machine logged out, in order."""
        self.key_names_machine = False
        """Whether the key names the machine, as one a login made since keys did: a
        machine's logout then revokes it."""

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if request.url.path == DEVICE_CODE_PATH:
            self.machine = json_loads(request.content, dict[str, object])["machine"]
            return httpx2.Response(201, content=_CODE.model_dump_json())
        if request.url.path == DEVICE_POLL_PATH:
            return httpx2.Response(200, content=self._poll().model_dump_json())
        if self.revoked or request.headers.get("authorization") != f"Bearer {_KEY}":
            return httpx2.Response(401, json={"detail": "Invalid or expired token"})
        if (status := self.refuse.get(request.url.path)) is not None:
            return httpx2.Response(status, json={"detail": "refused"})
        return self._answer((request.method, request.url.path))

    def hosts(self) -> set[str]:
        """Return every host a request went to."""
        return {request.url.host for request in self.requests}

    def paths(self) -> list[str]:
        """Return every request's path, in order."""
        return [request.url.path for request in self.requests]

    def _poll(self) -> DeviceToken:
        if self.decision == "denied":
            return DeviceToken(status="denied")
        return DeviceToken(
            status="approved",
            issued=IssuedKey(
                api_key=_KEY,
                role="writer",
                email=_EMAIL,
                machine_token="" if self.machine is None else _LOGIN_TOKEN,
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
                },
            )
        if route == ("POST", f"/api/me/tokens/{_KEY_ID}/revoke"):
            self.revoked = True
            return httpx2.Response(200, json={"ok": True})
        if route == ("GET", "/api/machines"):
            return httpx2.Response(200, json={"machines": self.machines})
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
        for name in (str(machine["name"]) for machine in self.machines):
            if route == ("GET", f"/api/machines/{name}/runs"):
                return httpx2.Response(200, json={"runs": self.runs.get(name, [])})
            if route == ("POST", f"/api/machines/{name}/disconnect"):
                self.disconnected.append(name)
                return httpx2.Response(200, json={"runs": self.runs.pop(name, [])})
            if route == ("POST", f"/api/machines/{name}/logout"):
                self.logged_out.append(name)
                self.revoked = self.key_names_machine
                keys = int(self.key_names_machine)
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
    """One ``serve.connect_machine`` call, as the machine host saw it."""

    name: str
    server: str
    token: str
    """The token the host joined with: the one handed in, else one it enrolled."""

    exec_commands: bool
    sandbox: bool
    jobs: bool = False
    results_max: int = 256 * 2**20


class _Host:
    """Stands in for ``serve.connect_machine``: records the call, takes a token."""

    def __init__(
        self,
        *,
        reads_token: bool = True,
        ending: BaseException | int | None = None,
    ) -> None:
        self.reads_token = reads_token
        """Whether it needs a token, as a host with no saved credential does."""
        self.ending = ending
        """What stops it: an exception it raises, or a signal it sends itself."""
        self.connects: list[_Served] = []

    def __call__(
        self,
        name: str,
        *,
        server: str,
        signed_in: Callable[[], Client] | None,
        sandbox: bool,
        exec_commands: bool,
        sleep: object = None,
        token: str = "",
        jobs: bool = False,
        results_max: int = 0,
    ) -> None:
        del sleep
        if isinstance(self.ending, BaseException):
            raise self.ending
        if isinstance(self.ending, int):
            os.kill(os.getpid(), self.ending)
            time.sleep(5)
        if self.reads_token and not token and signed_in is not None:
            token = signed_in().enroll_machine(name).token
        self.connects.append(
            _Served(
                name=name,
                server=server,
                token=token if self.reads_token else "",
                exec_commands=exec_commands,
                sandbox=sandbox,
                jobs=jobs,
                results_max=results_max,
            ),
        )


class _Runner:
    """Stands in for an entry point knowop hands off to: records argv, exits ``code``."""

    def __init__(self, *, code: int = 0) -> None:
        self.code = code
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str] | None = None) -> int:
        assert argv is not None, "knowop must hand the runner its arguments"
        self.calls.append(list(argv))
        return self.code


@dataclass(frozen=True, slots=True, kw_only=True)
class _Runners:
    """The three runners ``run`` and ``job`` hand off to."""

    claude: _Runner
    codex: _Runner
    job: _Runner

    def calls(self) -> list[tuple[str, list[str]]]:
        """Return every hand-off, each with the runner that took it."""
        return [
            *(("claude", argv) for argv in self.claude.calls),
            *(("codex", argv) for argv in self.codex.calls),
            *(("job", argv) for argv in self.job.calls),
        ]


@pytest.fixture(autouse=True)
def server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Server:
    """Answer every request from the scripted server, on a computer named Ada_Mac."""
    scripted = _Server()
    monkeypatch.setattr(httpx2, "HTTPTransport", _transport_of(scripted))
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
    monkeypatch.setattr(socket, "gethostname", lambda: "Ada_Mac.local")
    # A person at a terminal pipes no token; a test that does sets its own stdin.
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    # The real host serves until Ctrl-C, and the real runners start an agent or a job.
    monkeypatch.setattr(serve, "connect_machine", _never_runs("connect_machine"))
    monkeypatch.setattr(agent_cli, "main", _never_runs("the claude runner"))
    monkeypatch.setattr(agent_cli, "codex_main", _never_runs("the codex runner"))
    monkeypatch.setattr(jobs_cli, "main", _never_runs("the job runner"))
    return scripted


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch) -> _Host:
    stand_in = _Host()
    monkeypatch.setattr(serve, "connect_machine", stand_in)
    return stand_in


@pytest.fixture
def runners(monkeypatch: pytest.MonkeyPatch) -> _Runners:
    stand_ins = _Runners(claude=_Runner(), codex=_Runner(), job=_Runner())
    monkeypatch.setattr(agent_cli, "main", stand_ins.claude)
    monkeypatch.setattr(agent_cli, "codex_main", stand_ins.codex)
    monkeypatch.setattr(jobs_cli, "main", stand_ins.job)
    return stand_ins


def test_login_with_no_argument_signs_in_to_the_default_server(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login") == 0

    assert server.hosts() == {"alpha.knowop.ai"}
    assert json_loads(server.requests[0].content, dict[str, object]) == {
        "client": "knowop",
        "host": "Ada_Mac",
        "machine": None,
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
        ["run", "claude"],
        ["job", "logs", "7"],
        ["machine"],
        ["canvas", _CANVAS],
    ):
        assert _knowop(*argv) == 0, argv
    assert _knowop("logout") == 0

    assert server.hosts() == {"acme.knowop.ai"}
    # The host enrolled this computer with knowop's key, not trax's.
    assert [(served.server, served.token) for served in host.connects] == [
        ("https://acme.knowop.ai", _ENROLL_TOKEN),
    ]
    assert runners.calls() == [
        ("claude", ["--profile", "acme.knowop.ai"]),
        ("job", ["--profile", "acme.knowop.ai", "logs", "7"]),
    ]
    assert list_profiles() == [("prod", _PROD)]
    assert current_profile() == "prod"


def test_status_names_the_profile_server_account_role_and_the_accounts_machines(
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
        "machine: ada-mac online (this computer)",
        "machine: gpu1 offline",
    ]


def test_status_lists_each_machines_runs_and_the_runs_here_it_names_on_none(
    server: _Server,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.runs["gpu1"] = [_run("chat-1a2b3c4d")]
    monkeypatch.setattr(
        detach,
        "live_hosts",
        lambda: [_host("chat-1a2b3c4d"), _host("scratch")],
    )
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("status") == 0

    assert capsys.readouterr().out.splitlines()[4:] == [
        "machine: ada-mac online (this computer)",
        "machine: gpu1 offline, 1 run live",
        "  run:   chat-1a2b3c4d  claude  ~/work",
        "here:    scratch  claude",
    ]


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
    assert out == ""
    assert err.count("\n") == 1
    assert err.startswith("knowop: not logged in:")
    assert "knowop login" in err


def test_status_says_when_the_account_has_no_machine(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.machines = server.machines[2:]
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("status") == 0

    assert capsys.readouterr().out.splitlines()[-1] == (
        "machine: none; run knowop connect"
    )


def test_status_of_a_viewer_says_machines_need_the_writer_role(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.refuse["/api/machines"] = 403
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("status") == 0

    assert capsys.readouterr().out.splitlines()[-1] == (
        "machine: listing machines needs the writer role"
    )


@pytest.mark.parametrize("path", ["/api/me/profile", "/api/machines"])
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


def test_logout_logs_this_computers_machine_out_and_its_key_goes_with_it(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.key_names_machine = True
    server.runs["ada-mac"] = [_run("chat-1a2b3c4d")]
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("logout") == 0

    assert server.logged_out == ["ada-mac"]
    # The machine's logout revoked the key a login made for it, so nothing is left to revoke.
    assert not any(path.endswith("/revoke") for path in server.paths())
    assert list_profiles() == []
    assert capsys.readouterr().out.splitlines() == [
        "logged out: machine ada-mac; runs stopped: chat-1a2b3c4d; keys revoked: 1",
        "logged out: revoked its key; deleted profile acme.knowop.ai",
    ]


def test_logout_of_a_machine_the_server_does_not_know_as_ours_revokes_the_key_alone(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login", "acme") == 0
    server.refuse["/api/machines/ada-mac/logout"] = 404
    capsys.readouterr()

    assert _knowop("logout") == 0

    assert server.logged_out == []
    assert list_profiles() == []
    assert capsys.readouterr().out.splitlines() == [
        "logged out: revoked its key; deleted profile acme.knowop.ai",
    ]


def test_logout_keeps_the_profile_when_the_machine_logout_is_refused(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login", "acme") == 0
    server.refuse["/api/machines/ada-mac/logout"] = 409

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


def test_disconnect_stops_the_runs_here_before_it_disconnects_this_computer(
    server: _Server,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.runs["ada-mac"] = [_run("chat-1a2b3c4d")]
    assert _knowop("login", "acme") == 0
    asked: list[int] = []

    def stop_all() -> int:
        asked.append(len(server.requests))
        return 0

    monkeypatch.setattr(detach, "stop_all", stop_all)
    before = len(server.requests)
    capsys.readouterr()

    assert _knowop("disconnect") == 0

    assert (asked, server.disconnected) == ([before], ["ada-mac"])
    assert capsys.readouterr().out.splitlines() == [
        "stopped: run chat-1a2b3c4d (claude)",
        (
            "disconnected: machine ada-mac; its connect stops at its next heartbeat, and "
            "knowop connect joins it again"
        ),
    ]


def test_disconnect_of_another_machine_leaves_the_runs_here(
    server: _Server,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(detach, "stop_all", _never_runs("stop_all"))
    assert _knowop("login", "acme") == 0

    assert _knowop("disconnect", "gpu1") == 0

    assert server.disconnected == ["gpu1"]


def test_disconnect_stops_short_of_the_server_when_a_run_here_does_not_stop(
    server: _Server,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(detach, "stop_all", lambda: 1)
    assert _knowop("login", "acme") == 0

    assert _knowop("disconnect") == 1

    assert server.disconnected == []
    assert "a run here did not stop" in capsys.readouterr().err


def test_connect_without_a_login_enrolls_this_computer_in_the_same_approval(
    server: _Server,
    host: _Host,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("connect") == 0

    assert json_loads(server.requests[0].content, dict[str, object]) == {
        "client": "knowop",
        "host": "Ada_Mac",
        "machine": "ada-mac",
    }
    assert host.connects == [
        _Served(
            name="ada-mac",
            server="https://alpha.knowop.ai",
            token=_LOGIN_TOKEN,
            exec_commands=True,
            sandbox=False,
        ),
    ]
    assert ENROLL_PATH not in server.paths()
    assert "logged in: profile alpha.knowop.ai (writer)\n" in capsys.readouterr().out
    assert _knowop("status") == 0
    assert "profile: alpha.knowop.ai\n" in capsys.readouterr().out


def test_connect_after_a_login_enrolls_the_named_machine_with_knowops_key(
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
            exec_commands=True,
            sandbox=True,
        ),
    ]
    enrolls = [r for r in server.requests if r.url.path == ENROLL_PATH]
    assert [json_loads(r.content, dict[str, object]) for r in enrolls] == [
        {"name": "gpu1"},
    ]
    assert server.paths().count(DEVICE_CODE_PATH) == 1


def test_connect_with_no_exec_serves_terminals_only(host: _Host) -> None:
    assert _knowop("login", "acme") == 0

    assert _knowop("connect", "--no-exec") == 0

    assert [served.exec_commands for served in host.connects] == [False]
    assert [served.jobs for served in host.connects] == [False]


def test_connect_with_jobs_runs_them_and_bounds_what_one_sends_back(
    host: _Host,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login", "acme") == 0

    assert _knowop("connect", "--jobs") == 0
    assert _knowop("connect", "--jobs", "--no-exec", "--job-results-max", "1G") == 0
    assert _knowop("connect", "--jobs", "--job-results-max", "lots") == 1

    assert [(s.jobs, s.exec_commands, s.results_max) for s in host.connects] == [
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
    assert json_loads(codes[1].content, dict[str, object])["machine"] == "ada-mac"
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
    monkeypatch.setattr(serve, "connect_machine", joined)
    assert _knowop("login", "acme") == 0

    assert _knowop("connect") == 0

    assert [served.name for served in joined.connects] == ["ada-mac"]
    assert ENROLL_PATH not in server.paths()


@pytest.mark.parametrize(
    ("name", "said"),
    [
        ("Ada_Mac", "invalid machine name 'Ada_Mac'; expected "),
        ("connect", "machine name 'connect' is reserved"),
        ("a" * 64, f"invalid machine name '{'a' * 64}'; expected "),
    ],
)
def test_connect_refuses_a_name_no_machine_may_take_before_logging_in(
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
    monkeypatch.setattr(serve, "connect_machine", _Host(ending=KeyboardInterrupt()))
    assert _knowop("login", "acme") == 0

    assert _knowop("connect") == 130


def test_ctrl_c_names_the_runs_it_leaves_reaching_the_server(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The machine reads offline once its connect lets go; a detached run does not."""
    monkeypatch.setattr(serve, "connect_machine", _Host(ending=KeyboardInterrupt()))
    monkeypatch.setattr(detach, "live_hosts", lambda: [_host("chat-1a2b3c4d")])
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("connect") == 130

    assert capsys.readouterr().err == (
        "knowop: still reaching the server from here: chat-1a2b3c4d; stop them with "
        "knowop run stop --all, or everything with knowop disconnect\n"
    )


@pytest.mark.parametrize("ending", [signal.SIGTERM, signal.SIGHUP])
def test_a_stop_or_a_hangup_ends_connect_like_ctrl_c_and_is_then_forgotten(
    monkeypatch: pytest.MonkeyPatch,
    ending: signal.Signals,
) -> None:
    """``systemctl stop`` sends one, a closed ssh session the other: both release."""
    monkeypatch.setattr(serve, "connect_machine", _Host(ending=ending))
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
    monkeypatch.setattr(httpx2, "HTTPTransport", _transport_of(later))

    assert _knowop("connect") == 0

    assert later.paths() == [ENROLL_PATH]
    assert len(host.connects) == 2


@pytest.mark.parametrize(
    ("argv", "runner", "handed"),
    [
        (["claude", "--as", "ada", "--", "--model", "opus"], "claude", 1),
        (["codex", "--private"], "codex", 1),
        (["attach", "claude"], "claude", 0),
        (["resume", "00000000-0000-0000-0000-000000000009"], "claude", 0),
        (["--detach", "--name", "nightly"], "claude", 0),
    ],
)
def test_run_hands_the_agent_runner_knowops_profile_and_the_rest(
    runners: _Runners,
    argv: list[str],
    runner: str,
    handed: int,
) -> None:
    assert _knowop("login", "acme") == 0

    assert _knowop("run", *argv) == 0

    assert runners.calls() == [
        (runner, ["--profile", "acme.knowop.ai", *argv[handed:]]),
    ]


def test_run_help_is_knowops_and_a_runners_help_is_the_runners(
    runners: _Runners,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exited:
        _knowop("run", "--help")

    assert exited.value.code == 0
    assert capsys.readouterr().out.startswith("usage: knowop run ")
    assert runners.calls() == []
    assert _knowop("login", "acme") == 0
    assert _knowop("run", "claude", "--help") == 0
    assert runners.calls() == [("claude", ["--profile", "acme.knowop.ai", "--help"])]


@pytest.mark.parametrize("argv", [["run", "codex"], ["job", "logs", "7"]])
def test_a_hand_off_exits_with_the_runners_status(
    runners: _Runners,
    argv: list[str],
) -> None:
    runners.codex.code = 3
    runners.job.code = 3
    assert _knowop("login", "acme") == 0

    assert _knowop(*argv) == 3


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


def test_machine_lists_the_machines_of_knowops_server(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("machine") == 0

    assert capsys.readouterr().out.splitlines() == [
        "ada-mac  personal  -  -",
        "gpu1     personal  -  -",
        "lab      personal  -  -",
    ]
    assert server.paths()[-1] == "/api/machines"


def test_machine_refuses_without_a_login_before_any_request(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("machine", "gpu1") == 1

    assert server.requests == []
    assert capsys.readouterr().err == "knowop: not logged in; run knowop login\n"


@pytest.mark.parametrize("argv", [["--help"], ["help"], ["gpu1", "role", "-h"]])
def test_machine_help_needs_no_login_and_asks_the_server_nothing(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    assert _knowop("machine", *argv) == 0

    assert capsys.readouterr().out.startswith("Usage: knowop machine")
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

    assert server.hosts() == {"acme.knowop.ai"}
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
    assert _knowop("run", "claude", "--as", "ada") == 0

    assert server.hosts() == {"alpha.knowop.ai"}
    assert json_loads(server.requests[0].content, dict[str, object]) == {
        "client": "knowop",
        "host": "Ada_Mac",
        "machine": None,
    }
    assert runners.calls() == [
        ("claude", ["--profile", "alpha.knowop.ai", "--as", "ada"]),
    ]
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
    others = [_machine("gpu1", status="online", owner=_EMAIL)]
    mine = [_machine("ada-mac", status=here, owner=_EMAIL)] if here else []
    server.machines = [*others, *mine]
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop("run", "codex") == 0

    hint = (
        "knowop: this computer (ada-mac) is not connected; run knowop connect in "
        "another terminal to open its terminal from Chat\n"
    )
    assert capsys.readouterr().err == (hint if said else "")
    assert [runner for runner, _ in runners.calls()] == ["codex"]


def test_a_run_says_nothing_of_connecting_when_the_machines_cannot_be_listed(
    server: _Server,
    runners: _Runners,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _knowop("login", "acme") == 0
    capsys.readouterr()
    server.refuse["/api/machines"] = 403

    assert _knowop("run", "codex") == 0

    assert capsys.readouterr().err == ""
    assert [runner for runner, _ in runners.calls()] == ["codex"]


@pytest.mark.parametrize(
    "argv",
    [["attach", "nightly"], ["ls"], ["claude", "--help"], ["-h", "--", "x"]],
)
def test_help_and_a_host_command_need_no_login_and_ask_the_server_nothing(
    server: _Server,
    runners: _Runners,
    argv: list[str],
) -> None:
    assert _knowop("run", *argv) == 0

    assert server.requests == []
    assert runners.calls() == [("claude", argv[1:] if argv[0] == "claude" else argv)]


def test_a_host_command_still_names_knowops_profile_once_logged_in(
    server: _Server,
    runners: _Runners,
) -> None:
    assert _knowop("login", "acme") == 0
    asked = len(server.requests)

    assert _knowop("run", "stop", "nightly") == 0

    assert len(server.requests) == asked
    assert runners.calls() == [
        ("claude", ["--profile", "acme.knowop.ai", "stop", "nightly"]),
    ]


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
            exec_commands=True,
            sandbox=True,
        ),
    ]
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


def test_job_help_is_knowops(
    runners: _Runners,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exited:
        _knowop("job", "--help")

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
    monkeypatch.setattr(metadata, "version", partial(_version_of, installed=installed))

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
        "Connected as ada-mac to https://acme.knowop.ai. Next:  knowop run claude\n",
        "",
    )


@pytest.mark.parametrize("here", ["offline", "never", "revoked", ""])
def test_bare_knowop_does_not_count_another_computers_connection(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
    here: str,
) -> None:
    server.machines = [
        _machine("gpu1", status="online", owner=_EMAIL),
        *([_machine("ada-mac", status=here, owner=_EMAIL)] if here else []),
    ]
    assert _knowop("login", "acme") == 0
    capsys.readouterr()

    assert _knowop() == 0

    assert capsys.readouterr() == ("Not connected. Run:  knowop connect\n", "")


def test_bare_knowop_does_not_count_another_persons_computer(
    server: _Server,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.machines = [_machine("ada-mac", status="online", owner="bob@example.com")]
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


def _knowop(*argv: str) -> int:
    """Run ``knowop`` with ``argv``, its login polling on a clock that never waits."""
    time = _Time()
    return cli.main(argv, sleep=time.sleep, clock=time.clock)


def _handler_before(signum: int, frame: object) -> None:
    """Stand for the handler a signal had before ``connect``, to be found again after."""
    del signum, frame


def _never_runs(name: str) -> Callable[..., int]:
    """Return what stands in for ``name`` in a test that must not reach it."""

    def refuse(*args: object, **kwargs: object) -> int:
        raise AssertionError(f"a test ran {name} with {args} {kwargs}")

    return refuse


def _machine(name: str, *, status: str, owner: str) -> dict[str, object]:
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
    """Return a live claude run, as a machine's runs route lists it."""
    return {
        "session_id": "11111111-1111-1111-1111-111111111111",
        "cli": "claude",
        "run": name,
        "cwd": "~/work",
        "state": "idle",
    }


def _host(name: str) -> HostRecord:
    """Return the record of a detached claude run on this computer."""
    return HostRecord(
        name=name,
        command="claude --model opus",
        cwd="/work",
        pid=1,
        address="/run/host.sock",
        started="2026-10-24T12:00:00+00:00",
        state="running",
    )


def _transport_of(server: _Server) -> Callable[..., httpx2.MockTransport]:
    """Return what stands in for ``httpx2.HTTPTransport``: it answers from ``server``."""

    def make(**options: object) -> httpx2.MockTransport:
        del options
        return httpx2.MockTransport(server)

    return make


def _version_of(name: str, *, installed: dict[str, str]) -> str:
    """Answer as ``importlib.metadata.version`` would with only ``installed`` present."""
    if name not in installed:
        raise metadata.PackageNotFoundError(name)
    return installed[name]


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
