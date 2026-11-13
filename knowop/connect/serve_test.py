"""Connecting this computer: login, join, serve, and release, against a scripted server."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, cast
from unittest.mock import Mock

import io
import os
import select
import stat
import threading
import time
import uuid

from trackinizer.client.client import Client
from trackinizer.client.errors import ClientError
from trackinizer.lib.posix import hostname
from trackinizer.lib.userdirs import state_dir
from trackinizer.trax.connection import server_dir
from trackinizer.trax.profile import Profile, list_profiles, save_profile
from trackinizer.wire import wire_daemon, wire_terminals
from trackinizer.wire.wire_daemon import (
    HEARTBEAT_SEC,
    EnrollResponse,
    HeartbeatResponse,
    JoinResponse,
)
from trackinizer.wire.wire_device import (
    DEVICE_CODE_PATH,
    DEVICE_POLL_PATH,
    DeviceCode,
    DeviceToken,
    IssuedKey,
)
from trackinizer.wire.wire_host_job import HOST_WORK_PATH
from trackinizer.wire.wire_session_relay import HOST_SESSIONS_WAIT_PATH
from treekle import json_loads
from wrapt import lazy_import

import httpx2
import pytest

from knowop.connect import (
    daemon,
    job_limits,
    job_serving,
    serve,
    session_serving,
    terminal_direct_aiortc,
    terminal_serving,
    tool_serving,
)
from knowop.connect.terminal_direct_aiortc import AiortcPeer
from knowop.connect.wake import Wake


if TYPE_CHECKING:
    from collections.abc import Callable

    from trackinizer.mock_test_support import MockProtocol

    from knowop.connect.terminal_serving import TerminalServing


_SERVER: Final = "https://know.example"
_HOST_ID: Final = uuid.UUID(int=7)
_CREDENTIAL: Final = "trax_machine_" + "1" * 32 + "_" + "C" * 43
_REVOKED_CREDENTIAL: Final = "trax_machine_" + "2" * 32 + "_" + "R" * 43
_ENROLLMENT: Final = "enr_" + "0" * 32 + "_" + "e" * 43
_LOGIN_KEY: Final = "trax_" + "k" * 43
_PROFILE_KEY: Final = "trax_" + "p" * 43
_BOB_KEY: Final = "trax_" + "b" * 43
_HOST_PATH: Final = f"/api/hosts/{_HOST_ID}"
_ACCOUNTS: Final = {
    f"Bearer {_LOGIN_KEY}": "ada@example.com",
    f"Bearer {_PROFILE_KEY}": "ada@example.com",
    f"Bearer {_BOB_KEY}": "bob@example.com",
}
"""Whose account each key signs in as; ada owns the host, bob does not."""


class _Server:
    """A server for one host: login, enroll, join, heartbeat, terminals and tools."""

    def __init__(
        self,
        *,
        poll: DeviceToken | None = None,
        enroll_status: int = 201,
        revoked: str = "",
    ) -> None:
        self.poll = poll
        self.enroll_status = enroll_status
        """What an enrollment answers when ada asks: 201 enrolls, else a refusal."""
        self.revoked = revoked
        """The host credential whose heartbeats answer 410, as after a disconnect."""
        self.requests: list[tuple[str, str, str, dict[str, object]]] = []
        self.over = threading.Event()
        self.requested = threading.Condition()
        """Notified at each request; the daemon's threads make them."""

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json_loads(request.content, dict[str, object]) if request.content else {}
        method, path = request.method, request.url.path
        auth = request.headers.get("authorization", "")
        with self.requested:
            self.requests.append((method, path, auth, body))
            self.requested.notify_all()
        return self._answer(method, path, auth=auth)

    def _answer(
        self,
        method: str,
        path: str,
        *,
        auth: str,
    ) -> httpx2.Response:
        if path == "/api/me/profile":
            return httpx2.Response(
                200,
                json={"email": _ACCOUNTS[auth], "role": "writer"},
            )
        if path == wire_daemon.ENROLL_PATH and _ACCOUNTS[auth] != "ada@example.com":
            return httpx2.Response(
                403,
                json={"detail": "host belongs to another user"},
            )
        if path == wire_daemon.ENROLL_PATH and self.enroll_status != 201:
            return httpx2.Response(
                self.enroll_status,
                json={"detail": "enrollment is busy"},
            )
        if path == DEVICE_CODE_PATH:
            code = DeviceCode(
                device_code="d" * 43,
                user_code="BCDF-GHJK",
                verification_uri=f"{_SERVER}/auth/device",
                verification_uri_complete=f"{_SERVER}/auth/device?user_code=BCDF-GHJK",
                expires_in=600,
                interval=5,
            )
            return httpx2.Response(201, content=code.model_dump_json())
        if path == DEVICE_POLL_PATH:
            issued = IssuedKey.model_validate(
                {
                    "api_key": _LOGIN_KEY,
                    "role": "writer",
                    "email": "ada@example.com",
                    "host_token": _ENROLLMENT,
                },
            )
            token = self.poll or DeviceToken(status="approved", issued=issued)
            return httpx2.Response(200, content=token.model_dump_json())
        if path == wire_daemon.ENROLL_PATH:
            enrolled = EnrollResponse.model_validate(
                {
                    "token": _ENROLLMENT,
                    "expires_at": datetime(2030, 1, 1, tzinfo=UTC),
                    "owner": "ada@example.com",
                },
            )
            return httpx2.Response(200, content=enrolled.model_dump_json())
        if path == wire_daemon.JOIN_PATH:
            joined = JoinResponse(host_id=_HOST_ID, credential=_CREDENTIAL)
            return httpx2.Response(200, content=joined.model_dump_json())
        if path.endswith("/heartbeat") and auth == f"Bearer {self.revoked}":
            return httpx2.Response(410, json={"detail": "host_revoked"})
        if path.endswith("/heartbeat"):
            beat = HeartbeatResponse(server_time=datetime(2030, 1, 1, tzinfo=UTC))
            return httpx2.Response(200, content=beat.model_dump_json())
        if path.endswith("/tools/wait"):
            self.over.wait(0.02)
            return httpx2.Response(200, content=b"null")
        if path == wire_terminals.HOST_TERMINAL_WAIT_PATH.format(
            host_id=_HOST_ID,
        ):
            self.over.wait(0.02)
            return httpx2.Response(200, content=b"null")
        if path == HOST_WORK_PATH.format(host_id=_HOST_ID):
            self.over.wait(0.02)
            return httpx2.Response(200, json={"work": None})
        if path == HOST_SESSIONS_WAIT_PATH.format(host_id=_HOST_ID):
            self.over.wait(0.02)
            return httpx2.Response(200, content=b"null")
        raise AssertionError(f"unexpected request {method} {path}")

    def asked(self, *suffixes: str) -> Callable[[float], bool]:
        """Return a wait until every path ending in one of ``suffixes`` was requested."""
        return partial(self._asked, suffixes)

    def _asked(self, suffixes: tuple[str, ...], timeout: float) -> bool:
        with self.requested:
            return self.requested.wait_for(
                lambda: all(self.calls(suffix) for suffix in suffixes),
                timeout=timeout,
            )

    def calls(self, suffix: str) -> list[tuple[str, str, str, dict[str, object]]]:
        return [call for call in self.requests if call[1].endswith(suffix)]


def _never(timeout: float) -> bool:
    """Serve nothing, at once."""
    del timeout
    return False


class _Waits:
    """The waits between polls: the heartbeat's is Ctrl-C, once ``served`` holds."""

    def __init__(self, served: Callable[[float], bool] = bool) -> None:
        self.served = served
        """Waits up to its argument, in seconds, for the daemon to have served."""

    def __call__(self, seconds: float) -> None:
        if seconds != HEARTBEAT_SEC:
            return
        assert self.served(5), "the daemon never served"
        raise KeyboardInterrupt


class _NoToolServer:
    """Stands in for sagent's ``Server``: these tests cover connect, not what it serves."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    async def aopen(self) -> None:
        return

    async def aclose(self) -> None:
        return

    async def adopt(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


@dataclass(frozen=True, slots=True, kw_only=True)
class _Dirs:
    secrets: Path
    state: Path


@pytest.fixture(autouse=True)
def _no_sagent_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve tools the same with or without sagent's host, which an export may lack."""
    monkeypatch.setattr(
        tool_serving,
        "sagent_server",
        SimpleNamespace(HOST_COMMAND=(), DEFAULT_TOOLS=(), Server=_NoToolServer),
    )
    monkeypatch.setattr(tool_serving, "strings", SimpleNamespace(ServeDirs=_Dirs))
    monkeypatch.setattr(tool_serving, "tool_bridge", SimpleNamespace(bridged=None))


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the login shell and the terminal env off the developer's own setup."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("TRACKINIZER_PROFILE", raising=False)
    monkeypatch.setenv("TRAX_HOST_TERMINAL_DIRECT", "0")
    monkeypatch.setattr(hostname, "stable_hostname", lambda: "Dev_box.local")


def _stdin(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(text))


def _on(server: _Server, monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer every client's requests from ``server``."""

    def transport(**options: object) -> httpx2.MockTransport:
        del options
        return httpx2.MockTransport(server)

    monkeypatch.setattr(httpx2, "HTTPTransport", transport)


def _connect(
    waits: _Waits,
    *,
    name: str = "dev-box",
    signed_in: Callable[[], Client] | None = None,
    sandbox: bool = False,
    serve_tools: bool = True,
    token: str = "",
    jobs: bool = False,
) -> None:
    with pytest.raises(KeyboardInterrupt):
        serve.connect_host(
            name,
            server=_SERVER,
            signed_in=signed_in,
            sandbox=sandbox,
            serve_tools=serve_tools,
            sleep=waits,
            token=token,
            jobs=jobs,
        )


def _cap(
    monkeypatch: pytest.MonkeyPatch,
    *,
    problem: str | None = None,
    gpus: list[job_limits.Gpu] | None = None,
) -> None:
    """Answer the cap probe and the GPU reads without systemd or nvidia-smi."""
    monkeypatch.setattr(job_limits, "cap_problem", lambda: problem)
    monkeypatch.setattr(job_limits, "gpus", lambda: gpus or [])
    monkeypatch.setattr(job_limits, "memory_budget_mb", lambda: 61_440)


def test_a_computer_with_no_login_logs_in_joins_with_the_returned_token_and_serves(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server = _Server()
    _on(server, monkeypatch)
    _stdin(monkeypatch, "")

    _connect(_Waits(server.asked("/tools/wait", "/terminals/wait")))

    [code] = server.calls("/auth/device/code")
    assert code[3] == {"client": "knowop", "hostname": "Dev_box", "host": "dev-box"}
    [join] = server.calls("/api/hosts/join")
    assert join[3]["name"] == "dev-box"
    assert join[3]["token"] == _ENROLLMENT
    assert join[2] == ""
    facts = join[3]["facts"]
    assert isinstance(facts, dict)
    assert (facts["terminal"], facts["tools"]) == (True, True)
    assert server.calls("/tools/wait")
    assert server.calls("/terminals/wait")
    # The key the login returned is saved, so the next run needs no browser.
    assert [name for name, _ in list_profiles()] == ["know.example"]
    out = capsys.readouterr().out
    assert "BCDF-GHJK" in out
    assert "joined: host dev-box" in out
    assert "serving: host dev-box" in out


def test_a_computer_that_runs_jobs_offers_its_capped_memory_and_its_own_gpus(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server = _Server()
    _on(server, monkeypatch)
    _stdin(monkeypatch, "")
    _cap(
        monkeypatch,
        gpus=[
            job_limits.Gpu(index=0, uuid="GPU-a", name="NVIDIA GB10", used_mb=None),
            job_limits.Gpu(index=1, uuid="GPU-b", name="NVIDIA H100", used_mb=0),
        ],
    )

    _connect(_Waits(server.asked("/tools/wait", "/work")), jobs=True)

    [join] = server.calls("/api/hosts/join")
    facts = join[3]["facts"]
    assert isinstance(facts, dict)
    assert (facts["jobs"], facts["gpus"], facts["jobs_memory_mb"]) == (True, 1, 61_440)
    works = server.calls("/work")
    assert {call[2] for call in works} == {f"Bearer {_CREDENTIAL}"}
    out = capsys.readouterr().out
    assert (
        "jobs: GPU 0 (NVIDIA GB10) shares the CPU's memory, which a memory cap does "
        "not bound, so no job is given it"
    ) in out
    assert "jobs: on, up to 61440 MiB together, GPUs: 1" in out
    assert "(terminals, tools, jobs and agents)" in out


def test_jobs_where_no_memory_cap_holds_are_refused_before_anything_is_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _Server()
    _on(server, monkeypatch)
    _cap(monkeypatch, problem="systemd-run --user could not start a capped scope")

    with pytest.raises(ClientError, match="--jobs needs systemd's user manager"):
        serve.connect_host(
            "dev-box",
            server=_SERVER,
            signed_in=None,
            sandbox=False,
            serve_tools=True,
            sleep=_Waits(_never),
            jobs=True,
        )

    assert server.requests == []


def test_a_computer_joined_by_a_piped_token_runs_no_ones_jobs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server = _Server()
    _on(server, monkeypatch)
    _cap(monkeypatch)
    _stdin(monkeypatch, f"{_ENROLLMENT}\n")

    with pytest.raises(ClientError) as refused:
        serve.connect_host(
            "dev-box",
            server=_SERVER,
            signed_in=None,
            sandbox=False,
            serve_tools=False,
            # Refused before it serves: a heartbeat here is a failure, not a hang.
            sleep=_Waits(_never),
            jobs=True,
        )

    assert (str(refused.value), refused.value.status_code) == (
        (
            "--jobs runs the jobs of a signed-in account; a computer joined by a piped "
            "token runs none"
        ),
        None,
    )
    assert server.calls("/api/hosts/join") == []
    assert "(terminals and jobs)" in capsys.readouterr().out


def test_each_serving_and_the_daemon_get_one_wake_the_host_and_the_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _Server()
    _on(server, monkeypatch)
    _stdin(monkeypatch, "")
    _cap(monkeypatch)
    made = {
        "terminals": Mock(wraps=terminal_serving.TerminalServing),
        "tools": Mock(wraps=tool_serving.ToolServing),
        "sessions": Mock(wraps=session_serving.SessionServing),
        "jobs": Mock(wraps=job_serving.JobServing),
    }
    monkeypatch.setattr(terminal_serving, "TerminalServing", made["terminals"])
    monkeypatch.setattr(tool_serving, "ToolServing", made["tools"])
    monkeypatch.setattr(session_serving, "SessionServing", made["sessions"])
    monkeypatch.setattr(job_serving, "JobServing", made["jobs"])
    connected = Mock(wraps=daemon.connect)
    monkeypatch.setattr(daemon, "connect", connected)

    _connect(
        _Waits(server.asked("/tools/wait", "/sessions/wait", "/work")),
        signed_in=lambda: Client(_SERVER, api_key=_PROFILE_KEY),
        jobs=True,
    )

    wake = cast(object, connected.call_args.kwargs["wake"])
    assert isinstance(wake, Wake)
    assert all(
        made[each].call_args.kwargs["wake"] is wake
        for each in ("terminals", "tools", "sessions")
    )
    terminals = made["terminals"].call_args.kwargs
    assert (terminals["name"], terminals["server_url"]) == ("dev-box", _SERVER)
    tools = made["tools"].call_args
    assert (tools.args, tools.kwargs["name"]) == ((_SERVER,), "dev-box")
    jobs = made["jobs"].call_args.kwargs
    assert (jobs["name"], jobs["server_url"], jobs["account"]) == (
        "dev-box",
        _SERVER,
        "ada@example.com",
    )
    assert jobs["results_max"] == 256 * 2**20


def test_a_connect_that_runs_tools_here_starts_agents_here_too(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server = _Server()
    _on(server, monkeypatch)
    _stdin(monkeypatch, "")
    made = Mock(wraps=session_serving.SessionServing)
    monkeypatch.setattr(session_serving, "SessionServing", made)

    _connect(_Waits(server.asked("/sessions/wait")))

    [join] = server.calls("/api/hosts/join")
    facts = join[3]["facts"]
    assert isinstance(facts, dict)
    assert facts["sessions"] is True
    # A person's own connect runs their own agents: no remote agent service's.
    assert "remote_agents" not in facts
    assert {call[2] for call in server.calls("/sessions/wait")} == {
        f"Bearer {_CREDENTIAL}",
    }
    # The process is handed the server's address, never a profile.
    assert made.call_args.args == (_SERVER,)
    made_with = made.call_args.kwargs
    assert made_with["credential"] == _CREDENTIAL
    # Each start runs where a person's own host runs it, the serving's default.
    assert "place" not in made_with
    # A wait held across a pause is started anew on the wake, as the tools' is.
    assert isinstance(made_with["wake"], Wake)
    assert (
        f"serving: host dev-box at {_SERVER} (terminals, tools and agents)"
        in capsys.readouterr().out.splitlines()
    )


def test_ctrl_c_releases_the_host_before_the_interrupt_goes_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _Server()
    _on(server, monkeypatch)
    _stdin(monkeypatch, "")

    _connect(_Waits())

    beats = server.calls("/heartbeat")
    assert [beat[3].get("claim") for beat in beats] == ["take", "release"]
    assert {beat[1] for beat in beats} == {f"{_HOST_PATH}/heartbeat"}
    assert {beat[2] for beat in beats} == {f"Bearer {_CREDENTIAL}"}


def test_the_host_credential_is_saved_for_the_server_and_the_account_that_approved(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _on(_Server(), monkeypatch)
    _stdin(monkeypatch, "")

    _connect(_Waits())

    credential = (
        state_dir()
        / "rekursiv-ai"
        / "trax"
        / "hosts"
        / server_dir(_SERVER)
        / "ada@example.com"
        / "dev-box.credential"
    )
    saved = JoinResponse.model_validate_json(credential.read_text())
    assert (saved.host_id, saved.credential) == (_HOST_ID, _CREDENTIAL)
    assert "joined: host dev-box as ada@example.com" in capsys.readouterr().out


def test_another_account_on_this_computer_is_never_served_the_first_ones_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _on(_Server(), monkeypatch)
    _stdin(monkeypatch, "")
    _connect(_Waits())
    second = _Server()
    _on(second, monkeypatch)
    bob = Client(_SERVER, api_key=_BOB_KEY)

    with pytest.raises(ClientError) as err:
        serve.connect_host(
            "dev-box",
            server=_SERVER,
            signed_in=lambda: bob,
            sandbox=False,
            serve_tools=True,
            sleep=_Waits(_never),
        )

    assert str(err.value) == (
        f"host dev-box on {_SERVER} belongs to another account, not to "
        "bob@example.com, who is signed in; connect under another name with "
        "`knowop connect --name NAME`, or ask an admin to free it with "
        "`knowop host dev-box del`"
    )
    # A refusal a restart cannot change: a service exits for good on it.
    assert err.value.status_code == 403
    # Ada's credential, saved on this computer, never served bob.
    assert second.calls("/heartbeat") == []
    assert second.calls("/api/hosts/join") == []


def test_a_credential_saved_for_no_account_is_not_served_to_the_signed_in_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hosts = state_dir() / "rekursiv-ai" / "trax" / "hosts" / server_dir(_SERVER)
    hosts.mkdir(parents=True)
    stale = JoinResponse(host_id=uuid.UUID(int=9), credential=_CREDENTIAL)
    (hosts / "dev-box.credential").write_text(stale.model_dump_json())
    server = _Server()
    _on(server, monkeypatch)
    _stdin(monkeypatch, "")
    keyed = Client(_SERVER, api_key=_PROFILE_KEY)

    _connect(_Waits(), signed_in=lambda: keyed)

    [enroll] = server.calls("/api/hosts/enroll")
    assert enroll[2] == f"Bearer {_PROFILE_KEY}"
    assert server.calls("/api/hosts/join")
    assert (hosts / "ada@example.com" / "dev-box.credential").is_file()


def test_a_token_the_caller_holds_joins_before_stdin_a_profile_or_a_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A login that enrolled the host returns its token; nothing else is asked."""
    server = _Server()
    _on(server, monkeypatch)
    _stdin(monkeypatch, "enr_" + "0" * 32 + "_" + "p" * 43 + "\n")
    keyed = Client(_SERVER, api_key=_PROFILE_KEY)

    _connect(_Waits(), signed_in=lambda: keyed, token=_ENROLLMENT)

    assert server.calls("/auth/device/code") == []
    assert server.calls("/api/hosts/enroll") == []
    [join] = server.calls("/api/hosts/join")
    assert join[3]["token"] == _ENROLLMENT
    # The login that returned it signed in an account, whose host this is.
    hosts = state_dir() / "rekursiv-ai" / "trax" / "hosts" / server_dir(_SERVER)
    assert (hosts / "ada@example.com" / "dev-box.credential").is_file()


def test_an_enrollment_refused_for_another_reason_is_said_as_the_server_said_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a 403 names another owner; any other refusal is the server's own."""
    server = _Server(enroll_status=409)
    _on(server, monkeypatch)
    _stdin(monkeypatch, "")
    keyed = Client(_SERVER, api_key=_PROFILE_KEY)

    with pytest.raises(ClientError) as err:
        serve.connect_host(
            "dev-box",
            server=_SERVER,
            signed_in=lambda: keyed,
            sandbox=False,
            serve_tools=True,
            sleep=_Waits(_never),
        )

    assert err.value.status_code == 409
    assert "another account" not in str(err.value)
    assert server.calls("/api/hosts/join") == []


def test_a_join_only_connect_enrolls_joins_and_returns_serving_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """What the service would refuse shows at the terminal; the service serves."""
    server = _Server()
    _on(server, monkeypatch)
    _stdin(monkeypatch, "")
    keyed = Client(_SERVER, api_key=_PROFILE_KEY)

    serve.connect_host(
        "dev-box",
        server=_SERVER,
        signed_in=lambda: keyed,
        sandbox=False,
        serve_tools=True,
        sleep=_Waits(_never),
        join_only=True,
    )

    assert [call[1] for call in server.requests] == [
        "/api/me/profile",
        wire_daemon.ENROLL_PATH,
        wire_daemon.JOIN_PATH,
    ]
    hosts = state_dir() / "rekursiv-ai" / "trax" / "hosts" / server_dir(_SERVER)
    assert (hosts / "ada@example.com" / "dev-box.credential").is_file()
    assert capsys.readouterr().out == "joined: host dev-box as ada@example.com\n"


def test_a_saved_profile_enrolls_the_host_with_its_key_and_logs_nobody_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _Server()
    _on(server, monkeypatch)
    _stdin(monkeypatch, "")
    save_profile("work", Profile(url=_SERVER, api_key=_PROFILE_KEY))
    keyed = Client(_SERVER, api_key=_PROFILE_KEY)

    _connect(_Waits(), signed_in=lambda: keyed)

    [enroll] = server.calls("/api/hosts/enroll")
    assert enroll[2] == f"Bearer {_PROFILE_KEY}"
    assert enroll[3] == {"name": "dev-box"}
    assert server.calls("/auth/device/code") == []
    [join] = server.calls("/api/hosts/join")
    assert join[3]["token"] == _ENROLLMENT
    assert [name for name, _ in list_profiles()] == ["work"]


def test_an_enrollment_token_on_stdin_joins_without_a_login_or_a_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _Server()
    _on(server, monkeypatch)
    _stdin(monkeypatch, f"{_ENROLLMENT}\n")

    _connect(_Waits(), name="box-7")

    assert server.calls("/auth/device/code") == []
    assert server.calls("/api/hosts/enroll") == []
    [join] = server.calls("/api/hosts/join")
    assert (join[3]["name"], join[3]["token"]) == ("box-7", _ENROLLMENT)
    assert list_profiles() == []
    # An admin's token names the host, not a person: its credential is the host's.
    hosts = state_dir() / "rekursiv-ai" / "trax" / "hosts" / server_dir(_SERVER)
    assert (hosts / "box-7.credential").is_file()


def test_a_later_run_reuses_the_host_credential_and_asks_for_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _Server()
    _on(first, monkeypatch)
    _stdin(monkeypatch, "")
    _connect(_Waits())
    second = _Server()
    _on(second, monkeypatch)
    # The key the login saved, which ``knowop connect`` finds by the server.
    keyed = Client(_SERVER, api_key=_LOGIN_KEY)

    _connect(_Waits(), signed_in=lambda: keyed)

    assert second.calls("/api/hosts/join") == []
    assert second.calls("/api/hosts/enroll") == []
    assert second.calls("/auth/device/code") == []
    [take, release] = second.calls("/heartbeat")
    assert (take[3]["claim"], release[3]["claim"]) == ("take", "release")
    facts = take[3]["facts"]
    assert isinstance(facts, dict)
    assert facts["tools"] is True


def test_a_computer_disconnected_since_its_last_connect_joins_again_with_no_login(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The Hosts page disconnected it; the next connect enrolls with the key."""
    saved = (
        state_dir()
        / "rekursiv-ai"
        / "trax"
        / "hosts"
        / server_dir(_SERVER)
        / "ada@example.com"
        / "dev-box.credential"
    )
    saved.parent.mkdir(parents=True)
    stale = JoinResponse(host_id=_HOST_ID, credential=_REVOKED_CREDENTIAL)
    saved.write_text(stale.model_dump_json())
    server = _Server(revoked=_REVOKED_CREDENTIAL)
    _on(server, monkeypatch)
    _stdin(monkeypatch, "")
    keyed = Client(_SERVER, api_key=_PROFILE_KEY)

    _connect(_Waits(server.asked("/tools/wait")), signed_in=lambda: keyed)

    assert server.calls("/auth/device/code") == []
    [enroll] = server.calls("/api/hosts/enroll")
    assert enroll[2] == f"Bearer {_PROFILE_KEY}"
    [join] = server.calls("/api/hosts/join")
    assert join[3]["token"] == _ENROLLMENT
    assert [(beat[2], beat[3]["claim"]) for beat in server.calls("/heartbeat")] == [
        (f"Bearer {_REVOKED_CREDENTIAL}", "take"),
        (f"Bearer {_CREDENTIAL}", "take"),
        (f"Bearer {_CREDENTIAL}", "release"),
    ]
    assert JoinResponse.model_validate_json(saved.read_text()).credential == _CREDENTIAL
    assert (
        "rejoined: host dev-box as ada@example.com; the server refused its saved "
        "credential"
    ) in capsys.readouterr().out.splitlines()


def test_no_exec_serves_no_tools_and_does_not_say_it_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _Server()
    _on(server, monkeypatch)
    _stdin(monkeypatch, f"{_ENROLLMENT}\n")

    _connect(_Waits(server.asked("/terminals/wait")), serve_tools=False)

    [join] = server.calls("/api/hosts/join")
    facts = join[3]["facts"]
    assert isinstance(facts, dict)
    assert "exec" not in facts
    assert "tools" not in facts
    assert "sessions" not in facts
    assert facts["terminal"] is True
    assert server.calls("/tools/wait") == []
    assert server.calls("/sessions/wait") == []
    assert server.calls("/terminals/wait")


@pytest.mark.parametrize(
    ("seam", "module"),
    [
        ("sagent_server", "sagent.host.absent_from_this_install"),
        ("tool_bridge", "trackinizer.client.absent_from_this_install"),
    ],
)
def test_an_install_without_what_tools_need_serves_none_and_does_not_say_it_does(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    seam: str,
    module: str,
) -> None:
    server = _Server()
    _on(server, monkeypatch)
    _stdin(monkeypatch, f"{_ENROLLMENT}\n")
    monkeypatch.setattr(tool_serving, seam, lazy_import(module))

    _connect(_Waits(server.asked("/terminals/wait")))

    [join] = server.calls("/api/hosts/join")
    facts = join[3]["facts"]
    assert isinstance(facts, dict)
    assert "tools" not in facts
    assert facts["terminal"] is True
    assert server.calls("/tools/wait") == []
    captured = capsys.readouterr()
    # An agent process is this install's own, which needs no sagent host.
    assert "(terminals and agents)" in captured.out
    assert sum("tools: off" in line for line in captured.err.splitlines()) == 1


def test_the_tool_serving_asks_with_the_host_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _Server()
    _on(server, monkeypatch)
    _stdin(monkeypatch, f"{_ENROLLMENT}\n")

    _connect(_Waits(server.asked("/tools/wait")))

    assert server.calls("/tools/wait")[0][2] == f"Bearer {_CREDENTIAL}"
    tools = state_dir() / "rekursiv-ai" / "knowop" / "tools" / server_dir(_SERVER)
    tools /= "dev-box"
    assert stat.S_IMODE(tools.stat().st_mode) == 0o700


def test_the_sandbox_declaration_is_recorded_by_the_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _on(_Server(), monkeypatch)
    _stdin(monkeypatch, f"{_ENROLLMENT}\n")

    _connect(_Waits(), sandbox=True)

    sandbox = state_dir() / "rekursiv-ai" / "trax" / "hosts" / ".sandbox"
    assert sandbox.is_file()


def test_a_login_the_person_denies_ends_the_connect_with_nothing_saved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _Server(poll=DeviceToken(status="denied"))
    _on(server, monkeypatch)
    _stdin(monkeypatch, "")

    with pytest.raises(ClientError, match="denied"):
        serve.connect_host(
            "dev-box",
            server=_SERVER,
            signed_in=None,
            sandbox=False,
            serve_tools=True,
            sleep=_Waits(_never),
        )

    assert server.calls("/api/hosts/join") == []
    assert list_profiles() == []
    hosts = state_dir() / "rekursiv-ai" / "trax" / "hosts"
    assert not (hosts / server_dir(_SERVER) / "dev-box.credential").exists()


def test_the_terminal_server_names_the_server_and_the_profile_but_holds_no_key() -> (
    None
):
    save_profile("hosted", Profile(url=_SERVER, api_key=_PROFILE_KEY))
    joined = JoinResponse(host_id=_HOST_ID, credential=_CREDENTIAL)

    daemon = serve._terminal_serving("dev-box", _SERVER, joined)

    assert daemon.trax_env == {
        "LC_TRAX_SERVER": _SERVER,
        "TRACKINIZER_PROFILE": "hosted",
    }


class _Tty:
    """A stdin at a keyboard: reading it would wait for a person."""

    def isatty(self) -> bool:
        return True

    def readline(self) -> str:
        raise AssertionError("a terminal must not be read for a token")


def test_a_terminal_is_never_read_for_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", _Tty())

    assert serve.piped_token() == ""


def test_the_piped_token_loses_its_newline_and_blanks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stdin(monkeypatch, f"  {_ENROLLMENT}  \nignored\n")

    assert serve.piped_token() == _ENROLLMENT


def test_a_closed_stdin_holds_no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """``knowop connect <&-`` leaves ``sys.stdin`` as ``None``."""
    monkeypatch.setattr("sys.stdin", None)

    assert serve.piped_token() == ""


class _Poller(Mock):
    poll: MockProtocol[list[tuple[int, int]]]
    register: MockProtocol[None]


def test_a_pipe_nobody_writes_to_is_not_waited_on_for_a_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An agent shell or a CI step leaves an open pipe; the login must still start."""
    monkeypatch.setattr(serve, "_STDIN_WAIT_SEC", 0.05)
    poller = _Poller(wraps=select.poll())
    poller.poll.return_value = list[tuple[int, int]]()
    monkeypatch.setattr(select, "poll", lambda: poller)
    read_end, write_end = os.pipe()
    try:
        with os.fdopen(read_end) as pipe:
            monkeypatch.setattr("sys.stdin", pipe)
            started = time.monotonic()

            assert serve.piped_token() == ""

            assert time.monotonic() - started < 2
            poller.register.assert_called_once_with(read_end, select.POLLIN)
            poller.poll.assert_called_once_with(50)
    finally:
        os.close(write_end)


def test_a_token_written_to_a_pipe_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    poller = _Poller(wraps=select.poll())
    monkeypatch.setattr(select, "poll", lambda: poller)
    read_end, write_end = os.pipe()
    _ = os.write(write_end, f"{_ENROLLMENT}\n".encode())
    try:
        with os.fdopen(read_end) as pipe:
            monkeypatch.setattr("sys.stdin", pipe)

            assert serve.piped_token() == _ENROLLMENT
            poller.register.assert_called_once_with(read_end, select.POLLIN)
            poller.poll.assert_called_once_with(3_000)
    finally:
        os.close(write_end)


def test_a_pipe_closed_without_a_token_holds_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_end, write_end = os.pipe()
    os.close(write_end)
    with os.fdopen(read_end) as pipe:
        monkeypatch.setattr("sys.stdin", pipe)

        assert serve.piped_token() == ""


def test_a_server_url_with_credentials_is_never_exported_to_the_shell() -> None:
    joined = JoinResponse(host_id=_HOST_ID, credential=_CREDENTIAL)

    daemon = serve._terminal_serving("gx10", "https://u:pw@server.test", joined)

    assert "LC_TRAX_SERVER" not in daemon.trax_env
    assert "pw" not in repr(daemon.trax_env)


def _terminals_started(
    monkeypatch: pytest.MonkeyPatch,
    *,
    installed: bool,
    switch: str | None,
) -> TerminalServing:
    """Connect with the daemon loop stubbed; return the terminal server it was given."""
    if switch is None:
        monkeypatch.delenv("TRAX_HOST_TERMINAL_DIRECT", raising=False)
    else:
        monkeypatch.setenv("TRAX_HOST_TERMINAL_DIRECT", switch)
    monkeypatch.setattr(
        terminal_direct_aiortc,
        "aiortc_peer_factory",
        lambda: AiortcPeer if installed else None,
    )
    seen: dict[str, object] = {}

    def fake_connect(name: str, **kwargs: object) -> None:
        seen.update(kwargs, name=name)

    monkeypatch.setattr(daemon, "connect", fake_connect)
    # A piped token: the account is settled before the daemon loop, and a box logs no one in.
    _stdin(monkeypatch, f"{_ENROLLMENT}\n")
    serve.connect_host(
        "gx10",
        server=_SERVER,
        signed_in=None,
        sandbox=False,
        serve_tools=True,
    )
    joined_terminals = cast(
        "Callable[[JoinResponse], TerminalServing]",
        seen["joined_terminals"],
    )
    return joined_terminals(
        JoinResponse(host_id=_HOST_ID, credential=_CREDENTIAL),
    )


def test_the_daemon_offers_a_direct_link_when_the_library_is_installed_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    daemon = _terminals_started(monkeypatch, installed=True, switch=None)

    assert daemon.direct is AiortcPeer
    assert capsys.readouterr().out.splitlines()[0] == "direct terminal link: on"


def test_the_daemon_stays_on_the_relay_and_says_why_when_the_library_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    daemon = _terminals_started(monkeypatch, installed=False, switch=None)

    assert daemon.direct is None
    [said, *_] = capsys.readouterr().out.splitlines()
    assert said == (
        "direct terminal link: off (aiortc is unavailable). "
        "Terminals still work through the relay."
    )


def test_the_direct_link_can_be_switched_off_though_the_library_is_installed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    daemon = _terminals_started(monkeypatch, installed=True, switch="0")

    assert daemon.direct is None
    assert capsys.readouterr().out.splitlines()[0] == (
        "direct terminal link: off (TRAX_HOST_TERMINAL_DIRECT=0)"
    )


def test_each_server_is_made_for_this_host_with_the_one_wake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _Server()
    _on(server, monkeypatch)
    _stdin(monkeypatch, "")
    _cap(monkeypatch)
    monkeypatch.setattr(tool_serving, "can_serve", lambda: True)
    seen: dict[str, object] = {}
    made: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(daemon, "connect", partial(_noted, seen))
    for factory in ("_terminal_serving", "_tool_serving", "_job_serving"):
        monkeypatch.setattr(serve, factory, partial(_made, made, factory))
    keyed = Client(_SERVER, api_key=_PROFILE_KEY)

    serve.connect_host(
        "gx10",
        server=_SERVER,
        signed_in=lambda: keyed,
        sandbox=False,
        serve_tools=True,
        jobs=True,
        results_max=7,
    )
    joined = JoinResponse(host_id=_HOST_ID, credential=_CREDENTIAL)
    for each in ("joined_terminals", "tools_for", "jobs_for"):
        made_for = seen[each]
        assert callable(made_for)
        _ = made_for(joined)

    wake = seen["wake"]
    assert seen["name"] == "gx10"
    assert [
        (factory, args[0], kwargs.get("wake")) for factory, args, kwargs in made
    ] == [
        ("_terminal_serving", "gx10", wake),
        ("_tool_serving", "gx10", wake),
        ("_job_serving", "gx10", None),
    ]
    assert made[2][2]["results_max"] == 7


def test_a_join_only_connect_makes_its_terminals_with_no_direct_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _on(_Server(), monkeypatch)
    _stdin(monkeypatch, f"{_ENROLLMENT}\n")
    seen: dict[str, object] = {}
    made: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(daemon, "connect", partial(_noted, seen))
    monkeypatch.setattr(serve, "_terminal_serving", partial(_made, made, "terminals"))

    serve.connect_host(
        "gx10",
        server=_SERVER,
        signed_in=None,
        sandbox=False,
        serve_tools=False,
        join_only=True,
    )
    made_for = seen["joined_terminals"]
    assert callable(made_for)
    _ = made_for(JoinResponse(host_id=_HOST_ID, credential=_CREDENTIAL))

    # Here a direct link is a keyword a serving connect adds; a join-only one adds none.
    assert "direct" not in made[0][2]


def _noted(seen: dict[str, object], name: str, **kwargs: object) -> None:
    """Note what ``daemon.connect`` was given, serving nothing."""
    seen.update(kwargs, name=name)


def _made(
    made: list[tuple[str, tuple[object, ...], dict[str, object]]],
    factory: str,
    *args: object,
    **kwargs: object,
) -> None:
    """Note one server's making: which factory, and what it was given."""
    made.append((factory, args, kwargs))


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
