"""Connecting this computer: login, join, serve, and release, against a scripted server."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, cast
from unittest.mock import Mock

import base64
import io
import os
import select
import threading
import time
import uuid

from trackinizer.client.client import Client
from trackinizer.client.errors import ClientError
from trackinizer.lib.userdirs import state_dir
from trackinizer.trax.connection import server_dir
from trackinizer.trax.profile import Profile, list_profiles, save_profile
from trackinizer.wire import wire_machine_host, wire_terminals
from trackinizer.wire.wire_device import (
    DEVICE_CODE_PATH,
    DEVICE_POLL_PATH,
    DeviceCode,
    DeviceToken,
    IssuedKey,
)
from trackinizer.wire.wire_exec import ExecResult, ExecWait
from trackinizer.wire.wire_machine_host import (
    HEARTBEAT_SEC,
    EnrollResponse,
    HeartbeatResponse,
    JoinResponse,
)
from treekle import json_loads

import httpx2
import pytest

from knowop.shell import (
    machine_host,
    serve,
    terminal_direct_aiortc,
)
from knowop.shell.terminal_direct_aiortc import AiortcPeer


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from knowop.shell.terminal_host import TerminalHost


_SERVER: Final = "https://know.example"
_MACHINE_ID: Final = uuid.UUID(int=7)
_CREDENTIAL: Final = "trax_machine_" + "1" * 32 + "_" + "C" * 43
_ENROLLMENT: Final = "enr_" + "0" * 32 + "_" + "e" * 43
_LOGIN_KEY: Final = "trax_" + "k" * 43
_PROFILE_KEY: Final = "trax_" + "p" * 43
_BOB_KEY: Final = "trax_" + "b" * 43
_MACHINE_PATH: Final = f"/api/machines/{_MACHINE_ID}"
_ACCOUNTS: Final = {
    f"Bearer {_LOGIN_KEY}": "ada@example.com",
    f"Bearer {_PROFILE_KEY}": "ada@example.com",
    f"Bearer {_BOB_KEY}": "bob@example.com",
}
"""Whose account each key signs in as; ada owns the machine, bob does not."""


class _Server:
    """A server for one machine: login, enroll, join, heartbeat, terminals and exec."""

    def __init__(
        self,
        *,
        commands: list[ExecWait] | None = None,
        poll: DeviceToken | None = None,
        enroll_status: int = 201,
    ) -> None:
        self.poll = poll
        self.enroll_status = enroll_status
        """What an enrollment answers when ada asks: 201 enrolls, else a refusal."""
        self.requests: list[tuple[str, str, str, dict[str, object]]] = []
        self.commands = list(commands or [])
        self.results: list[ExecResult] = []
        self.result_posted = threading.Event()
        self.over = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json_loads(request.content, dict[str, object]) if request.content else {}
        method, path = request.method, request.url.path
        auth = request.headers.get("authorization", "")
        with self._lock:
            self.requests.append((method, path, auth, body))
        return self._answer(method, path, body, auth=auth)

    def _answer(
        self,
        method: str,
        path: str,
        body: dict[str, object],
        *,
        auth: str,
    ) -> httpx2.Response:
        if path == "/api/me/profile":
            return httpx2.Response(
                200,
                json={"email": _ACCOUNTS[auth], "role": "writer"},
            )
        if (
            path == wire_machine_host.ENROLL_PATH
            and _ACCOUNTS[auth] != "ada@example.com"
        ):
            return httpx2.Response(
                403,
                json={"detail": "machine belongs to another user"},
            )
        if path == wire_machine_host.ENROLL_PATH and self.enroll_status != 201:
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
                    "machine_token": _ENROLLMENT,
                },
            )
            token = self.poll or DeviceToken(status="approved", issued=issued)
            return httpx2.Response(200, content=token.model_dump_json())
        if path == wire_machine_host.ENROLL_PATH:
            enrolled = EnrollResponse.model_validate(
                {
                    "token": _ENROLLMENT,
                    "expires_at": datetime(2030, 1, 1, tzinfo=UTC),
                    "owner": "ada@example.com",
                },
            )
            return httpx2.Response(200, content=enrolled.model_dump_json())
        if path == wire_machine_host.JOIN_PATH:
            joined = JoinResponse(machine_id=_MACHINE_ID, credential=_CREDENTIAL)
            return httpx2.Response(200, content=joined.model_dump_json())
        if path.endswith("/heartbeat"):
            beat = HeartbeatResponse(server_time=datetime(2030, 1, 1, tzinfo=UTC))
            return httpx2.Response(200, content=beat.model_dump_json())
        if path.endswith("/exec/wait"):
            return self._next_command()
        if "/exec/" in path and path.endswith("/result"):
            self.results.append(ExecResult.model_validate(body))
            self.result_posted.set()
            return httpx2.Response(204)
        if path == wire_terminals.MACHINE_TERMINAL_WAIT_PATH.format(
            machine_id=_MACHINE_ID,
        ):
            self.over.wait(0.02)
            return httpx2.Response(200, content=b"null")
        raise AssertionError(f"unexpected request {method} {path}")

    def _next_command(self) -> httpx2.Response:
        with self._lock:
            command = self.commands.pop(0) if self.commands else None
        if command is None:
            self.over.wait(0.02)
            return httpx2.Response(200, content=b"null")
        return httpx2.Response(200, content=command.model_dump_json())

    def asked(self, *suffixes: str) -> Callable[[], bool]:
        """Return a check that every path ending in one of ``suffixes`` was requested."""
        return lambda: all(self.calls(suffix) for suffix in suffixes)

    def calls(self, suffix: str) -> list[tuple[str, str, str, dict[str, object]]]:
        return [call for call in self.requests if call[1].endswith(suffix)]


class _Waits:
    """The waits between polls: the heartbeat's is Ctrl-C, once ``served`` is true."""

    def __init__(self, served: Callable[[], bool] = lambda: True) -> None:
        self.served = served

    def __call__(self, seconds: float) -> None:
        if seconds != HEARTBEAT_SEC:
            return
        deadline = time.monotonic() + 5
        while not self.served():
            assert time.monotonic() < deadline, "the host never served"
            time.sleep(0.005)
        raise KeyboardInterrupt


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the login shell and the terminal env off the developer's own setup."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("TRACKINIZER_PROFILE", raising=False)
    monkeypatch.setenv("TRAX_MACHINE_TERMINAL_DIRECT", "0")
    monkeypatch.setattr("socket.gethostname", lambda: "Dev_box.local")


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
    exec_commands: bool = True,
    token: str = "",
) -> None:
    with pytest.raises(KeyboardInterrupt):
        serve.connect_machine(
            name,
            server=_SERVER,
            signed_in=signed_in,
            sandbox=sandbox,
            exec_commands=exec_commands,
            sleep=waits,
            token=token,
        )


def test_a_computer_with_no_login_logs_in_joins_with_the_returned_token_and_serves(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server = _Server()
    _on(server, monkeypatch)
    _stdin(monkeypatch, "")

    _connect(_Waits(server.asked("/exec/wait", "/terminals/wait")))

    [code] = server.calls("/auth/device/code")
    assert code[3] == {"client": "knowop", "host": "Dev_box", "machine": "dev-box"}
    [join] = server.calls("/api/machines/join")
    assert join[3]["name"] == "dev-box"
    assert join[3]["token"] == _ENROLLMENT
    assert join[2] == ""
    facts = join[3]["facts"]
    assert isinstance(facts, dict)
    assert (facts["terminal"], facts["exec"]) == (True, True)
    assert server.calls("/exec/wait")
    assert server.calls("/terminals/wait")
    # The key the login returned is saved, so the next run needs no browser.
    assert [name for name, _ in list_profiles()] == ["know.example"]
    out = capsys.readouterr().out
    assert "BCDF-GHJK" in out
    assert "joined: machine dev-box" in out
    assert "serving: machine dev-box" in out


def test_ctrl_c_releases_the_machine_before_the_interrupt_goes_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _Server()
    _on(server, monkeypatch)
    _stdin(monkeypatch, "")

    _connect(_Waits())

    beats = server.calls("/heartbeat")
    assert [beat[3].get("claim") for beat in beats] == ["take", "release"]
    assert {beat[1] for beat in beats} == {f"{_MACHINE_PATH}/heartbeat"}
    assert {beat[2] for beat in beats} == {f"Bearer {_CREDENTIAL}"}


def test_the_machine_credential_is_saved_for_the_server_and_the_account_that_approved(
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
        / "machines"
        / server_dir(_SERVER)
        / "ada@example.com"
        / "dev-box.credential"
    )
    saved = JoinResponse.model_validate_json(credential.read_text())
    assert (saved.machine_id, saved.credential) == (_MACHINE_ID, _CREDENTIAL)
    assert "joined: machine dev-box as ada@example.com" in capsys.readouterr().out


def test_another_account_on_this_computer_is_never_served_the_first_ones_machine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _on(_Server(), monkeypatch)
    _stdin(monkeypatch, "")
    _connect(_Waits())
    second = _Server()
    _on(second, monkeypatch)
    bob = Client(_SERVER, api_key=_BOB_KEY)

    with pytest.raises(ClientError) as err:
        serve.connect_machine(
            "dev-box",
            server=_SERVER,
            signed_in=lambda: bob,
            sandbox=False,
            exec_commands=True,
            sleep=_Waits(),
        )

    assert str(err.value) == (
        f"machine dev-box on {_SERVER} belongs to another account, not to "
        "bob@example.com, who is signed in; connect under another name with "
        "`knowop connect --name NAME`, or ask an admin to free it with "
        "`knowop machine dev-box del`"
    )
    # Ada's credential, saved on this computer, never served bob.
    assert second.calls("/heartbeat") == []
    assert second.calls("/api/machines/join") == []


def test_a_credential_saved_for_no_account_is_not_served_to_the_signed_in_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machines = state_dir() / "rekursiv-ai" / "trax" / "machines" / server_dir(_SERVER)
    machines.mkdir(parents=True)
    stale = JoinResponse(machine_id=uuid.UUID(int=9), credential=_CREDENTIAL)
    (machines / "dev-box.credential").write_text(stale.model_dump_json())
    server = _Server()
    _on(server, monkeypatch)
    _stdin(monkeypatch, "")
    keyed = Client(_SERVER, api_key=_PROFILE_KEY)

    _connect(_Waits(), signed_in=lambda: keyed)

    [enroll] = server.calls("/api/machines/enroll")
    assert enroll[2] == f"Bearer {_PROFILE_KEY}"
    assert server.calls("/api/machines/join")
    assert (machines / "ada@example.com" / "dev-box.credential").is_file()


def test_a_token_the_caller_holds_joins_before_stdin_a_profile_or_a_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A login that enrolled the machine returns its token; nothing else is asked."""
    server = _Server()
    _on(server, monkeypatch)
    _stdin(monkeypatch, "enr_" + "0" * 32 + "_" + "p" * 43 + "\n")
    keyed = Client(_SERVER, api_key=_PROFILE_KEY)

    _connect(_Waits(), signed_in=lambda: keyed, token=_ENROLLMENT)

    assert server.calls("/auth/device/code") == []
    assert server.calls("/api/machines/enroll") == []
    [join] = server.calls("/api/machines/join")
    assert join[3]["token"] == _ENROLLMENT
    # The login that returned it signed in an account, whose machine this is.
    machines = state_dir() / "rekursiv-ai" / "trax" / "machines" / server_dir(_SERVER)
    assert (machines / "ada@example.com" / "dev-box.credential").is_file()


def test_an_enrollment_refused_for_another_reason_is_said_as_the_server_said_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a 403 names another owner; any other refusal is the server's own."""
    server = _Server(enroll_status=409)
    _on(server, monkeypatch)
    _stdin(monkeypatch, "")
    keyed = Client(_SERVER, api_key=_PROFILE_KEY)

    with pytest.raises(ClientError) as err:
        serve.connect_machine(
            "dev-box",
            server=_SERVER,
            signed_in=lambda: keyed,
            sandbox=False,
            exec_commands=True,
            sleep=_Waits(),
        )

    assert err.value.status_code == 409
    assert "another account" not in str(err.value)
    assert server.calls("/api/machines/join") == []


def test_a_saved_profile_enrolls_the_machine_with_its_key_and_logs_nobody_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _Server()
    _on(server, monkeypatch)
    _stdin(monkeypatch, "")
    save_profile("work", Profile(url=_SERVER, api_key=_PROFILE_KEY))
    keyed = Client(_SERVER, api_key=_PROFILE_KEY)

    _connect(_Waits(), signed_in=lambda: keyed)

    [enroll] = server.calls("/api/machines/enroll")
    assert enroll[2] == f"Bearer {_PROFILE_KEY}"
    assert enroll[3] == {"name": "dev-box"}
    assert server.calls("/auth/device/code") == []
    [join] = server.calls("/api/machines/join")
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
    assert server.calls("/api/machines/enroll") == []
    [join] = server.calls("/api/machines/join")
    assert (join[3]["name"], join[3]["token"]) == ("box-7", _ENROLLMENT)
    assert list_profiles() == []
    # An admin's token names the machine, not a person: its credential is the machine's.
    machines = state_dir() / "rekursiv-ai" / "trax" / "machines" / server_dir(_SERVER)
    assert (machines / "box-7.credential").is_file()


def test_a_later_run_reuses_the_machine_credential_and_asks_for_nothing(
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

    assert second.calls("/api/machines/join") == []
    assert second.calls("/api/machines/enroll") == []
    assert second.calls("/auth/device/code") == []
    [take, release] = second.calls("/heartbeat")
    assert (take[3]["claim"], release[3]["claim"]) == ("take", "release")
    facts = take[3]["facts"]
    assert isinstance(facts, dict)
    assert facts["exec"] is True


def test_no_exec_serves_no_commands_and_does_not_say_it_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _Server()
    _on(server, monkeypatch)
    _stdin(monkeypatch, f"{_ENROLLMENT}\n")

    _connect(_Waits(server.asked("/terminals/wait")), exec_commands=False)

    [join] = server.calls("/api/machines/join")
    facts = join[3]["facts"]
    assert isinstance(facts, dict)
    assert "exec" not in facts
    assert facts["terminal"] is True
    assert server.calls("/exec/wait") == []
    assert server.calls("/terminals/wait")


def test_a_command_the_server_hands_over_is_run_and_its_result_posted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    command = ExecWait(
        exec_id=uuid.uuid4(),
        asker="ada@example.com",
        command="echo from the assistant",
        cwd=str(tmp_path),
        timeout_sec=10,
    )
    server = _Server(commands=[command])
    _on(server, monkeypatch)
    _stdin(monkeypatch, f"{_ENROLLMENT}\n")
    waits = _Waits(server.result_posted.is_set)

    _connect(waits)

    [result] = server.results
    assert base64.b64decode(result.stdout_b64) == b"from the assistant\n"
    [posted] = server.calls("/result")
    assert posted[1] == f"{_MACHINE_PATH}/exec/{command.exec_id}/result"
    assert posted[2] == f"Bearer {_CREDENTIAL}"
    log = state_dir() / "rekursiv-ai" / "trax" / "machines" / "dev-box.exec.log"
    assert "ada@example.com" in log.read_text()


def test_the_sandbox_declaration_is_recorded_by_the_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _on(_Server(), monkeypatch)
    _stdin(monkeypatch, f"{_ENROLLMENT}\n")

    _connect(_Waits(), sandbox=True)

    sandbox = state_dir() / "rekursiv-ai" / "trax" / "machines" / ".sandbox"
    assert sandbox.is_file()


def test_a_login_the_person_denies_ends_the_connect_with_nothing_saved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _Server(poll=DeviceToken(status="denied"))
    _on(server, monkeypatch)
    _stdin(monkeypatch, "")

    with pytest.raises(ClientError, match="denied"):
        serve.connect_machine(
            "dev-box",
            server=_SERVER,
            signed_in=None,
            sandbox=False,
            exec_commands=True,
            sleep=_Waits(),
        )

    assert server.calls("/api/machines/join") == []
    assert list_profiles() == []
    machines = state_dir() / "rekursiv-ai" / "trax" / "machines"
    assert not (machines / server_dir(_SERVER) / "dev-box.credential").exists()


def test_the_terminal_server_names_the_server_and_the_profile_but_holds_no_key() -> (
    None
):
    save_profile("hosted", Profile(url=_SERVER, api_key=_PROFILE_KEY))
    joined = JoinResponse(machine_id=_MACHINE_ID, credential=_CREDENTIAL)

    host = serve._terminal_host("dev-box", _SERVER, joined)

    assert host.trax_env == {
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


def test_a_pipe_nobody_writes_to_is_not_waited_on_for_a_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An agent shell or a CI step leaves an open pipe; the login must still start."""
    monkeypatch.setattr(serve, "_STDIN_WAIT_SEC", 0.05)
    read_end, write_end = os.pipe()
    try:
        with os.fdopen(read_end) as pipe:
            monkeypatch.setattr("sys.stdin", pipe)
            started = time.monotonic()

            assert serve.piped_token() == ""

            assert time.monotonic() - started < 2
    finally:
        os.close(write_end)


def test_a_token_written_to_a_pipe_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    poller = Mock(wraps=select.poll())
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
    joined = JoinResponse(machine_id=_MACHINE_ID, credential=_CREDENTIAL)

    host = serve._terminal_host("gx10", "https://u:pw@server.test", joined)

    assert "LC_TRAX_SERVER" not in host.trax_env
    assert "pw" not in repr(host.trax_env)


def _terminals_started(
    monkeypatch: pytest.MonkeyPatch,
    *,
    installed: bool,
    switch: str | None,
) -> TerminalHost:
    """Connect with the host loop stubbed; return the terminal server it was given."""
    if switch is None:
        monkeypatch.delenv("TRAX_MACHINE_TERMINAL_DIRECT", raising=False)
    else:
        monkeypatch.setenv("TRAX_MACHINE_TERMINAL_DIRECT", switch)
    monkeypatch.setattr(
        terminal_direct_aiortc,
        "aiortc_peer_factory",
        lambda: AiortcPeer if installed else None,
    )
    seen: dict[str, object] = {}

    def fake_connect(name: str, **kwargs: object) -> None:
        seen.update(kwargs, name=name)

    monkeypatch.setattr(machine_host, "connect", fake_connect)
    # A piped token: the account is settled before the host loop, and a box logs no one in.
    _stdin(monkeypatch, f"{_ENROLLMENT}\n")
    serve.connect_machine(
        "gx10",
        server=_SERVER,
        signed_in=None,
        sandbox=False,
        exec_commands=True,
    )
    terminals_for = cast(
        "Callable[[JoinResponse], TerminalHost]",
        seen["terminals_for"],
    )
    return terminals_for(JoinResponse(machine_id=_MACHINE_ID, credential=_CREDENTIAL))


def test_the_host_offers_a_direct_link_when_the_library_is_installed_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    host = _terminals_started(monkeypatch, installed=True, switch=None)

    assert host.direct is AiortcPeer
    assert capsys.readouterr().out.splitlines()[0] == "direct terminal link: on"


def test_the_host_stays_on_the_relay_and_says_why_when_the_library_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    host = _terminals_started(monkeypatch, installed=False, switch=None)

    assert host.direct is None
    [said, *_] = capsys.readouterr().out.splitlines()
    assert said == (
        "direct terminal link: off (aiortc is unavailable). "
        "Terminals still work through the relay."
    )


def test_the_direct_link_can_be_switched_off_though_the_library_is_installed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    host = _terminals_started(monkeypatch, installed=True, switch="0")

    assert host.direct is None
    assert capsys.readouterr().out.splitlines()[0] == (
        "direct terminal link: off (TRAX_MACHINE_TERMINAL_DIRECT=0)"
    )


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
