"""Tests for the machine host loop, against a scripted client."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import partial
from importlib import metadata
from typing import TYPE_CHECKING, Final

import getpass
import os
import platform
import shutil
import stat
import uuid

from trackinizer.client.errors import ClientError
from trackinizer.lib.userdirs import state_dir
from trackinizer.wire.wire_machine_host import (
    HEARTBEAT_SEC,
    HeartbeatResponse,
    JoinRequest,
    JoinResponse,
)

import pytest

from knowop.shell.machine_host import (
    HOST_CLIS,
    connect,
    host_facts,
    is_sandbox,
    set_sandbox,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from trackinizer.wire.wire_machine_host import Claim
    from trackinizer.wire.wire_machines import Facts


_MACHINE_ID: Final = uuid.UUID("11111111-1111-1111-1111-111111111111")
_CREDENTIAL: Final = "trax_machine_" + "1" * 32 + "_" + "C" * 43
_OTHER_CREDENTIAL: Final = "trax_machine_" + "2" * 32 + "_" + "D" * 43
"""A credential other than the one the scripted join returns."""
_TOKEN: Final = "enr_" + "0" * 32 + "_" + "S" * 43
_NOW: Final = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
_SERVER: Final = "https://trax.example"

_REVOKED: Final = ClientError("-> 410", status_code=410)
_UNKNOWN: Final = ClientError("-> 401", status_code=401)
_TAKEN: Final = ClientError("-> 409", status_code=409, code="another_host_connected")
_REFUSED: Final = ClientError("POST /x failed: connection refused")


class _Host:
    """A scripted server: each heartbeat takes the next outcome in turn.

    By default one heartbeat lands, and the next finds a newer connect took over,
    which ends the host and leaves its credential saved.
    """

    def __init__(
        self,
        *,
        beats: Sequence[ClientError | HeartbeatResponse | None] = (None, _TAKEN),
        join_error: ClientError | None = None,
    ) -> None:
        self.beats = list(beats)
        self.join_error = join_error
        self.calls: list[tuple[object, ...]] = []
        self.keys: list[str] = []
        self.closed = 0

    def __call__(self, key: str) -> _Host:
        self.keys.append(key)
        return self

    def join_machine(
        self,
        name: str,
        *,
        token: str,
        instance: uuid.UUID,
        host_version: str,
        facts: Facts,
    ) -> JoinResponse:
        self.calls.append(("join", name, token, instance, host_version, facts))
        if self.join_error is not None:
            raise self.join_error
        return JoinResponse(machine_id=_MACHINE_ID, credential=_CREDENTIAL)

    def heartbeat_machine(
        self,
        machine_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        host_version: str,
        facts: Facts | None = None,
        claim: Claim = "keep",
    ) -> HeartbeatResponse:
        self.calls.append(("beat", machine_id, instance, host_version, facts, claim))
        outcome = self.beats.pop(0)
        if isinstance(outcome, ClientError):
            raise outcome
        return HeartbeatResponse(server_time=_NOW) if outcome is None else outcome

    def close(self) -> None:
        self.closed += 1


# ``""`` is no account: the path then skips that directory, as ``connect`` does.
def _save(*, credential: str = _CREDENTIAL, account: str = "") -> None:
    """Save box-1's credential for ``account``, as a first ``connect`` would have."""
    machines = state_dir() / "rekursiv-ai" / "trax" / "machines" / "trax.example"
    path = machines / account / "box-1.credential"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        JoinResponse(machine_id=_MACHINE_ID, credential=credential).model_dump_json(),
    )


def _held(account: str = "") -> str | None:
    """Return the credential saved for box-1 and ``account``; ``None`` when none is."""
    machines = state_dir() / "rekursiv-ai" / "trax" / "machines" / "trax.example"
    path = machines / account / "box-1.credential"
    if not path.exists():
        return None
    return JoinResponse.model_validate_json(path.read_text()).credential


def _connect(host: _Host, *, stdin: str = f"{_TOKEN}\n") -> list[float]:
    """Connect box-1 with ``stdin`` as its first line, returning each wait."""
    slept: list[float] = []
    connect(
        "box-1",
        server=_SERVER,
        client_for=host,
        read_token=lambda: stdin,
        sleep=slept.append,
    )
    return slept


def _connect_as(name: str, host: _Host, *, sandbox: bool = False) -> list[float]:
    """Connect ``name`` with the token on stdin, declaring a sandbox if asked."""
    slept: list[float] = []
    connect(
        name,
        server=_SERVER,
        client_for=host,
        read_token=lambda: f"{_TOKEN}\n",
        sleep=slept.append,
        sandbox=sandbox,
    )
    return slept


def test_a_computer_is_no_sandbox_until_a_connect_declares_it() -> None:
    _save()

    _ = _connect_as("box-1", _Host())

    assert not is_sandbox()


def test_a_first_connect_with_sandbox_records_it_beside_the_credentials_and_says_so(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ = _connect_as("box-1", _Host(), sandbox=True)

    assert is_sandbox()
    machines = state_dir() / "rekursiv-ai" / "trax" / "machines"
    assert (machines / ".sandbox").is_file()
    assert (machines / "trax.example" / "box-1.credential").is_file()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "joined: machine box-1"
    assert lines[1].startswith("sandbox: on (")
    assert lines[2:] == ["taken over: machine box-1 is served by a newer connect"]


def test_a_saved_credential_connect_with_sandbox_records_it_too(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save()

    _ = _connect_as("box-1", _Host(), sandbox=True)

    assert is_sandbox()
    assert capsys.readouterr().out.count("sandbox: on") == 1


def test_a_connect_without_the_flag_clears_the_declaration_and_says_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save()
    set_sandbox(on=True)

    _ = _connect_as("box-1", _Host())

    assert not is_sandbox()
    assert "sandbox" not in capsys.readouterr().out


def test_the_declaration_belongs_to_the_computer_not_to_one_machine_name() -> None:
    _ = _connect_as("box-1", _Host(), sandbox=True)
    assert is_sandbox()

    _ = _connect_as("box-2", _Host())

    assert not is_sandbox()


def test_a_refused_join_leaves_the_declaration_as_it_was() -> None:
    refused = ClientError("-> 401", status_code=401)

    with pytest.raises(ClientError):
        _ = _connect_as("box-1", _Host(join_error=refused), sandbox=True)
    assert not is_sandbox()

    set_sandbox(on=True)
    with pytest.raises(ClientError):
        _ = _connect_as("box-1", _Host(join_error=refused))
    assert is_sandbox()


def test_clearing_a_declaration_that_was_never_made_creates_nothing() -> None:
    set_sandbox(on=False)

    assert not (state_dir() / "rekursiv-ai").exists()


def test_a_first_connect_joins_with_the_stdin_token_and_saves_the_credential(
    capsys: pytest.CaptureFixture[str],
) -> None:
    host = _Host(beats=[None, None, _TAKEN])

    slept = _connect(host)

    join, *beats = host.calls
    assert join[:3] == ("join", "box-1", _TOKEN)
    assert join[5] == host_facts()
    # The join stored the facts, so no heartbeat repeats them.
    assert [beat[4] for beat in beats] == [None, None, None]
    assert {beat[2] for beat in beats} == {join[3]}
    assert host.keys == ["", _CREDENTIAL]
    assert host.closed == 2
    assert slept == [HEARTBEAT_SEC, HEARTBEAT_SEC]
    saved = JoinResponse.model_validate_json(
        (
            state_dir()
            / "rekursiv-ai"
            / "trax"
            / "machines"
            / "trax.example"
            / "box-1.credential"
        ).read_text(),
    )
    assert (saved.machine_id, saved.credential) == (_MACHINE_ID, _CREDENTIAL)
    assert capsys.readouterr().out.splitlines() == [
        "joined: machine box-1",
        "taken over: machine box-1 is served by a newer connect",
    ]


def test_the_saved_credential_is_readable_by_this_user_only() -> None:
    _connect(_Host())

    path = (
        state_dir()
        / "rekursiv-ai"
        / "trax"
        / "machines"
        / "trax.example"
        / "box-1.credential"
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_a_saved_credential_skips_the_join_and_sends_facts_once() -> None:
    _save()
    host = _Host(beats=[None, None, _REVOKED])

    connect(
        "box-1",
        server=_SERVER,
        client_for=host,
        read_token=_no_token,
        sleep=lambda _: None,
    )

    assert [call[0] for call in host.calls] == ["beat"] * 3
    assert [call[4] for call in host.calls] == [host_facts(), None, None]
    assert host.keys == [_CREDENTIAL]


def test_a_personal_credential_is_kept_under_its_account(
    capsys: pytest.CaptureFixture[str],
) -> None:
    connect(
        "box-1",
        server=_SERVER,
        client_for=_Host(),
        read_token=lambda: _TOKEN,
        sleep=lambda _: None,
        account="Ada@X.y",
    )

    machines = state_dir() / "rekursiv-ai" / "trax" / "machines" / "trax.example"
    assert sorted(path.name for path in machines.rglob("*.credential")) == [
        "box-1.credential",
    ]
    assert (machines / "ada@x.y" / "box-1.credential").is_file()
    assert capsys.readouterr().out.splitlines()[0] == "joined: machine box-1 as ada@x.y"


def test_an_accounts_own_saved_credential_skips_the_join() -> None:
    connect(
        "box-1",
        server=_SERVER,
        client_for=_Host(),
        read_token=lambda: _TOKEN,
        sleep=lambda _: None,
        account="ada@x.y",
    )
    host = _Host()

    connect(
        "box-1",
        server=_SERVER,
        client_for=host,
        read_token=_no_token,
        sleep=lambda _: None,
        account="ada@x.y",
    )

    assert [call[0] for call in host.calls] == ["beat", "beat"]
    assert host.keys == [_CREDENTIAL]


@pytest.mark.parametrize("saved_by", ["ada@x.y", None], ids=["account", "no-account"])
def test_a_credential_saved_by_anyone_else_is_never_used(saved_by: str | None) -> None:
    connect(
        "box-1",
        server=_SERVER,
        client_for=_Host(),
        read_token=lambda: _TOKEN,
        sleep=lambda _: None,
        account=saved_by,
    )
    host = _Host()

    connect(
        "box-1",
        server=_SERVER,
        client_for=host,
        read_token=lambda: _TOKEN,
        sleep=lambda _: None,
        account="bob@x.y",
    )

    # Bob's own enrollment joins: the server, not a file on this computer, decides
    # whether bob may hold box-1.
    assert [call[0] for call in host.calls] == ["join", "beat", "beat"]
    machines = state_dir() / "rekursiv-ai" / "trax" / "machines" / "trax.example"
    assert (machines / "bob@x.y" / "box-1.credential").is_file()


@pytest.mark.parametrize("account", ["", "../ada@x.y", "ada/b@x.y", ".ada@x.y", "ada"])
def test_an_account_that_is_no_email_or_would_leave_its_directory_is_refused(
    account: str,
) -> None:
    host = _Host()

    with pytest.raises(ClientError, match="not an account"):
        connect(
            "box-1",
            server=_SERVER,
            client_for=host,
            read_token=_no_token,
            sleep=lambda _: None,
            account=account,
        )

    assert host.calls == []
    assert not (state_dir() / "rekursiv-ai").exists()


def test_each_process_heartbeats_as_its_own_instance() -> None:
    _save()
    first, second = _Host(), _Host()

    for host in (first, second):
        connect(
            "box-1",
            server=_SERVER,
            client_for=host,
            read_token=_no_token,
            sleep=lambda _: None,
        )

    assert first.calls[0][2] != second.calls[0][2]


def test_lost_contact_is_retried_sooner_at_first_and_never_ends_the_host(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save()
    down = ClientError("-> 503", status_code=503)
    host = _Host(
        beats=[_REFUSED, down, _REFUSED, _REFUSED, _REFUSED, None, _REFUSED, _REVOKED],
    )
    delays: list[float] = []

    connect(
        "box-1",
        server=_SERVER,
        client_for=host,
        read_token=_no_token,
        sleep=delays.append,
    )

    assert delays == [2.0, 4.0, 8.0, 15.0, 15.0, HEARTBEAT_SEC, 2.0]
    # Facts and the take ride the first heartbeat that lands, not the first one tried.
    assert [call[4] is not None for call in host.calls[:6]] == [True] * 6
    assert [call[5] for call in host.calls] == [*["take"] * 6, "keep", "keep"]
    assert host.calls[6][4] is None
    assert capsys.readouterr().err.count("retrying") == 6


def test_a_host_the_server_calls_outdated_says_how_to_upgrade_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save()
    outdated = HeartbeatResponse(server_time=_NOW, host_outdated=True)

    _connect(_Host(beats=[outdated, _REFUSED, outdated, outdated, _REVOKED]))

    line = f"knowop: this knowop is too old for {_SERVER}: run uvx knowop@latest"
    assert capsys.readouterr().err.splitlines().count(f"{line} connect {_SERVER}") == 1


def test_a_host_the_server_does_not_call_outdated_says_nothing_of_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save()

    _connect(_Host(beats=[None, None, _REVOKED]))

    assert "too old" not in capsys.readouterr().err


def test_hours_offline_never_overflow_the_backoff() -> None:
    _save()
    refused = ClientError("POST /x failed: connection refused")
    host = _Host(beats=[*[refused] * 1_100, None, _REVOKED])
    delays: list[float] = []

    connect(
        "box-1",
        server=_SERVER,
        client_for=host,
        read_token=_no_token,
        sleep=delays.append,
    )

    assert len(delays) == 1_101
    assert set(delays[4:]) == {float(HEARTBEAT_SEC)}


# A 401 before any heartbeat landed is a saved credential the server forgot, which
# the connect replaces (see the tests of joining again); once serving, it ends the host.
@pytest.mark.parametrize("status", [401, 404, 422])
def test_an_answer_a_retry_cannot_change_ends_the_host(status: int) -> None:
    _save()
    host = _Host(beats=[None, ClientError(f"-> {status}", status_code=status)])

    with pytest.raises(ClientError) as err:
        connect(
            "box-1",
            server=_SERVER,
            client_for=host,
            read_token=_no_token,
            sleep=lambda _: None,
        )

    assert err.value.status_code == status
    assert _held() == _CREDENTIAL


def test_an_empty_stdin_never_joins() -> None:
    host = _Host()

    with pytest.raises(ClientError, match="enrollment token from stdin"):
        _connect(host, stdin="  \n")

    assert host.calls == []
    assert not (
        state_dir()
        / "rekursiv-ai"
        / "trax"
        / "machines"
        / "trax.example"
        / "box-1.credential"
    ).exists()


def test_a_refused_join_saves_nothing() -> None:
    host = _Host(join_error=ClientError("-> 401", status_code=401))

    with pytest.raises(ClientError):
        _connect(host)

    assert not (
        state_dir()
        / "rekursiv-ai"
        / "trax"
        / "machines"
        / "trax.example"
        / "box-1.credential"
    ).exists()
    assert [call[0] for call in host.calls] == ["join"]


def test_a_corrupt_credential_file_is_named_without_quoting_it() -> None:
    path = (
        state_dir()
        / "rekursiv-ai"
        / "trax"
        / "machines"
        / "trax.example"
        / "box-1.credential"
    )
    path.parent.mkdir(parents=True)
    path.write_text(f'{{"credential": "{_CREDENTIAL}"}}')

    with pytest.raises(ClientError) as err:
        connect(
            "box-1",
            server=_SERVER,
            client_for=_Host(),
            read_token=_no_token,
            sleep=lambda _: None,
        )

    assert str(path) in str(err.value)
    assert _CREDENTIAL not in str(err.value)
    assert err.value.__cause__ is None


@pytest.mark.parametrize(
    ("refusal", "account", "said"),
    [
        (_REVOKED, "ada@x.y", "rejoined: machine box-1 as ada@x.y"),
        (_UNKNOWN, "ada@x.y", "rejoined: machine box-1 as ada@x.y"),
        (_REVOKED, "", "rejoined: machine box-1"),
    ],
    ids=["revoked", "unknown", "piped token"],
)
def test_a_saved_credential_refused_before_it_served_is_replaced_by_a_join(
    refusal: ClientError,
    account: str,
    said: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A disconnect, or a deleted machine, since the last connect: this one joins."""
    _save(credential=_OTHER_CREDENTIAL, account=account)
    host = _Host(beats=[refusal, None, _TAKEN])

    connect(
        "box-1",
        server=_SERVER,
        client_for=host,
        read_token=lambda: _TOKEN,
        sleep=lambda _: None,
        account=account or None,
    )

    assert [call[0] for call in host.calls] == ["beat", "join", "beat", "beat"]
    assert host.calls[1][2] == _TOKEN
    assert host.keys == [_OTHER_CREDENTIAL, "", _CREDENTIAL]
    # The join stored the facts, so the heartbeats after it do not repeat them.
    assert [call[4] for call in host.calls[2:]] == [None, None]
    assert [call[5] for call in host.calls[2:]] == ["take", "keep"]
    assert _held(account) == _CREDENTIAL
    assert capsys.readouterr().out.splitlines()[0] == (
        f"{said}; the server refused its saved credential"
    )


def _refuse_enrollment() -> str:
    raise ClientError("POST /api/machines/enroll -> 401", status_code=401)


@pytest.mark.parametrize(
    ("account", "read_token", "join_error", "asked", "hint"),
    [
        (
            "ada@x.y",
            lambda: _TOKEN,
            ClientError("-> 401", status_code=401),
            ["beat", "join"],
            "run knowop login, then knowop connect",
        ),
        (
            "ada@x.y",
            _refuse_enrollment,
            None,
            ["beat"],
            "run knowop login, then knowop connect",
        ),
        (
            "",
            lambda: _TOKEN,
            ClientError("-> 401", status_code=401),
            ["beat", "join"],
            "pipe a new enrollment token to knowop connect",
        ),
    ],
    ids=["join refused", "enrollment refused", "piped token refused"],
)
def test_a_refused_rejoin_says_what_to_run_and_leaves_no_credential(
    account: str,
    read_token: Callable[[], str],
    join_error: ClientError | None,
    asked: list[str],
    hint: str,
) -> None:
    _save(credential=_OTHER_CREDENTIAL, account=account)
    host = _Host(beats=[_REVOKED], join_error=join_error)

    with pytest.raises(ClientError) as err:
        connect(
            "box-1",
            server=_SERVER,
            client_for=host,
            read_token=read_token,
            sleep=lambda _: None,
            account=account or None,
        )

    assert str(err.value) == f"machine box-1 could not join {_SERVER} again; {hint}"
    assert err.value.status_code == 401
    assert _held(account) is None
    assert [call[0] for call in host.calls] == asked


def test_a_rejoin_that_fails_another_way_says_what_the_server_said() -> None:
    _save(credential=_OTHER_CREDENTIAL)
    down = ClientError("-> 503", status_code=503)

    with pytest.raises(ClientError) as err:
        _connect(_Host(beats=[_REVOKED], join_error=down))

    assert err.value is down
    assert _held() is None


def test_a_credential_refused_right_after_a_rejoin_ends_the_host_without_another(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save(credential=_OTHER_CREDENTIAL)
    host = _Host(beats=[_REVOKED, _REVOKED, None, _TAKEN])

    _connect(host)

    assert [call[0] for call in host.calls] == ["beat", "join", "beat"]
    assert _held() is None
    assert capsys.readouterr().out.splitlines()[1:] == ["revoked: machine box-1"]


def test_a_credential_revoked_right_after_a_first_join_is_forgotten_not_joined_again(
    capsys: pytest.CaptureFixture[str],
) -> None:
    host = _Host(beats=[_REVOKED, None, _TAKEN])

    _connect(host)

    assert [call[0] for call in host.calls] == ["join", "beat"]
    assert _held() is None
    assert capsys.readouterr().out == "joined: machine box-1\nrevoked: machine box-1\n"


def test_a_machine_revoked_while_it_serves_ends_the_host_and_forgets_the_credential(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save()
    host = _Host(beats=[None, _REVOKED])

    connect(
        "box-1",
        server=_SERVER,
        client_for=host,
        read_token=_no_token,
        sleep=lambda _: None,
    )

    assert [call[0] for call in host.calls] == ["beat", "beat"]
    assert _held() is None
    assert capsys.readouterr().out == "revoked: machine box-1\n"


def test_a_credential_another_connect_saved_meanwhile_outlives_the_revoke() -> None:
    _save()

    def another_connect_joins(seconds: float) -> None:
        del seconds
        _save(credential=_OTHER_CREDENTIAL)

    connect(
        "box-1",
        server=_SERVER,
        client_for=_Host(beats=[None, _REVOKED]),
        read_token=_no_token,
        sleep=another_connect_joins,
    )

    assert _held() == _OTHER_CREDENTIAL


def test_host_facts_fit_the_join_contract() -> None:
    facts = host_facts()

    assert set(facts) == {"os", "arch", "cpus", "memory_mb", "user", "clis"}
    cpus, memory_mb, clis = facts["cpus"], facts["memory_mb"], facts["clis"]
    assert isinstance(cpus, int)
    assert isinstance(memory_mb, int)
    assert isinstance(clis, list)
    assert (cpus > 0, memory_mb > 0, set(clis) <= set(HOST_CLIS)) == (True,) * 3
    joined = JoinRequest(
        name="box-1",
        token=_TOKEN,
        instance=uuid.uuid4(),
        host_version="",
        facts=facts,
    )
    assert joined.facts == facts


def test_host_facts_report_what_the_host_says_about_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sysconf = {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 2**20}

    def sysconf_of(name: str) -> int:
        return sysconf[name]

    def which(cli: str) -> str | None:
        return "/bin/" + cli if cli in {"git", "uv"} else None

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    monkeypatch.setattr(os, "sysconf", sysconf_of)
    monkeypatch.setattr(getpass, "getuser", lambda: "ada")
    monkeypatch.setattr(shutil, "which", which)

    assert host_facts() == {
        "os": "linux",
        "arch": "aarch64",
        "cpus": 0,
        "memory_mb": 4096,
        "user": "ada",
        "clis": ["git", "uv"],
    }
    assert host_facts(terminal=True)["terminal"] is True
    assert host_facts(exec_commands=True)["exec"] is True
    assert set(host_facts(terminal=True, exec_commands=True)) - set(host_facts()) == {
        "terminal",
        "exec",
    }


class _Terminals:
    """Records when a server starts and stops, relative to the heartbeats."""

    def __init__(
        self,
        calls: list[tuple[object, ...]],
        label: str = "terminals",
        *,
        start_error: Exception | None = None,
    ) -> None:
        self.calls = calls
        self.label = label
        self.start_error = start_error

    def start(self, machine_id: uuid.UUID, instance: uuid.UUID) -> None:
        self.calls.append((f"{self.label}-start", machine_id, instance))
        if self.start_error is not None:
            raise self.start_error

    def stop(self) -> None:
        self.calls.append((f"{self.label}-stop",))


def test_a_host_that_opted_in_reports_terminals_and_stops_them_before_it_returns() -> (
    None
):
    _save()
    host = _Host(beats=[None, _REVOKED])
    served: list[JoinResponse] = []

    def terminals_for(joined: JoinResponse) -> _Terminals:
        served.append(joined)
        return _Terminals(host.calls)

    connect(
        "box-1",
        server=_SERVER,
        client_for=host,
        read_token=_no_token,
        sleep=lambda _: None,
        terminals_for=terminals_for,
    )

    kinds = [call[0] for call in host.calls]
    assert kinds == ["beat", "terminals-start", "beat", "terminals-stop"]
    assert host.calls[0][4] == host_facts(terminal=True)
    assert host.calls[1][1] == _MACHINE_ID
    assert host.calls[1][2] == host.calls[0][2]
    assert [joined.credential for joined in served] == [_CREDENTIAL]


def test_no_server_starts_before_a_heartbeat_has_taken_the_machine() -> None:
    # The server refuses a wait from an instance that does not hold the machine, so a
    # server started first would be refused until the take lands.
    _save()
    host = _Host(beats=[_REFUSED, None, _REVOKED])

    connect(
        "box-1",
        server=_SERVER,
        client_for=host,
        read_token=_no_token,
        sleep=lambda _: None,
        terminals_for=lambda _joined: _Terminals(host.calls),
        exec_for=lambda _joined: _Terminals(host.calls, "exec"),
        jobs_for=lambda _joined: _Terminals(host.calls, "jobs"),
    )

    assert [call[0] for call in host.calls] == [
        "beat",
        "beat",
        "terminals-start",
        "exec-start",
        "jobs-start",
        "beat",
        "jobs-stop",
        "exec-stop",
        "terminals-stop",
    ]
    assert [call[5] for call in host.calls if call[0] == "beat"] == [
        "take",
        "take",
        "keep",
    ]


def test_a_host_that_serves_exec_reports_it_and_stops_it_before_the_terminals() -> None:
    _save()
    host = _Host(beats=[None, _REVOKED])

    connect(
        "box-1",
        server=_SERVER,
        client_for=host,
        read_token=_no_token,
        sleep=lambda _: None,
        terminals_for=lambda _joined: _Terminals(host.calls),
        exec_for=lambda _joined: _Terminals(host.calls, "exec"),
    )

    kinds = [call[0] for call in host.calls]
    assert kinds == [
        "beat",
        "terminals-start",
        "exec-start",
        "beat",
        "exec-stop",
        "terminals-stop",
    ]
    assert host.calls[0][4] == host_facts(terminal=True, exec_commands=True)
    assert host.calls[2][1:] == host.calls[1][1:]


def test_a_host_that_serves_exec_alone_reports_exec_and_no_terminal() -> None:
    _save()
    host = _Host()

    connect(
        "box-1",
        server=_SERVER,
        client_for=host,
        read_token=_no_token,
        sleep=lambda _: None,
        exec_for=lambda _joined: _Terminals(host.calls, "exec"),
    )

    assert host.calls[0][4] == {**host_facts(), "exec": True}
    assert "terminal" not in host_facts(exec_commands=True)


def test_a_host_that_runs_jobs_says_what_it_offers_and_stops_them_first() -> None:
    _save()
    host = _Host(beats=[None, _REVOKED])
    offered = {"gpus": 2, "jobs_memory_mb": 61_440}

    connect(
        "box-1",
        server=_SERVER,
        client_for=host,
        read_token=_no_token,
        sleep=lambda _: None,
        exec_for=lambda _joined: _Terminals(host.calls, "exec"),
        jobs_for=lambda _joined: _Terminals(host.calls, "jobs"),
        job_facts=offered,
    )

    assert [call[0] for call in host.calls] == [
        "beat",
        "exec-start",
        "jobs-start",
        "beat",
        "jobs-stop",
        "exec-stop",
    ]
    assert host.calls[0][4] == {
        **host_facts(exec_commands=True),
        "jobs": True,
        "gpus": 2,
        "jobs_memory_mb": 61_440,
    }


def test_a_host_that_runs_no_jobs_says_nothing_of_them() -> None:
    _save()
    host = _Host()

    connect(
        "box-1",
        server=_SERVER,
        client_for=host,
        read_token=_no_token,
        sleep=lambda _: None,
        job_facts={"gpus": 1},
    )

    sent = host.calls[0][4]
    assert isinstance(sent, dict)
    assert {"jobs", "gpus"}.isdisjoint(sent)
    assert host_facts(jobs={})["jobs"] is True


def test_an_exec_server_that_will_not_start_stops_the_terminals_already_running() -> (
    None
):
    _save()
    host = _Host(beats=[None])

    with pytest.raises(RuntimeError, match="no bash"):
        connect(
            "box-1",
            server=_SERVER,
            client_for=host,
            read_token=_no_token,
            sleep=lambda _: None,
            terminals_for=lambda _joined: _Terminals(host.calls),
            exec_for=lambda _joined: _Terminals(
                host.calls,
                "exec",
                start_error=RuntimeError("no bash"),
            ),
        )

    assert [call[0] for call in host.calls] == [
        "beat",
        "terminals-start",
        "exec-start",
        "terminals-stop",
    ]


def test_terminals_stop_when_the_host_ends_on_an_error_too() -> None:
    _save()
    host = _Host(beats=[None, ClientError("-> 403", status_code=403)])

    with pytest.raises(ClientError):
        connect(
            "box-1",
            server=_SERVER,
            client_for=host,
            read_token=_no_token,
            sleep=lambda _: None,
            terminals_for=lambda _joined: _Terminals(host.calls),
        )

    assert host.calls[-1] == ("terminals-stop",)


def test_a_host_that_did_not_opt_in_serves_no_terminals_and_says_so() -> None:
    _save()
    host = _Host()

    connect(
        "box-1",
        server=_SERVER,
        client_for=host,
        read_token=_no_token,
        sleep=lambda _: None,
    )

    assert "terminal" not in host_facts()
    assert [call[0] for call in host.calls] == ["beat", "beat"]
    assert host.calls[0][4] == host_facts()


def test_a_host_a_newer_connect_took_over_stops_its_terminals_and_returns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save()
    host = _Host(beats=[None, _TAKEN])
    slept: list[float] = []

    connect(
        "box-1",
        server=_SERVER,
        client_for=host,
        read_token=_no_token,
        sleep=slept.append,
        terminals_for=lambda _joined: _Terminals(host.calls),
    )

    kinds = [call[0] for call in host.calls]
    assert kinds == ["beat", "terminals-start", "beat", "terminals-stop"]
    assert slept == [HEARTBEAT_SEC]
    assert capsys.readouterr().out == (
        "taken over: machine box-1 is served by a newer connect\n"
    )


@pytest.mark.parametrize("answer", [None, _REFUSED])
def test_ctrl_c_releases_the_machine_once_then_stops_terminals_and_goes_on(
    answer: ClientError | None,
) -> None:
    _save()
    host = _Host(beats=[None, answer])

    def interrupt(seconds: float) -> None:
        del seconds
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        connect(
            "box-1",
            server=_SERVER,
            client_for=host,
            read_token=_no_token,
            sleep=interrupt,
            terminals_for=lambda _joined: _Terminals(host.calls),
        )

    beat, _, release, stop = host.calls
    assert (beat[5], release[5]) == ("take", "release")
    assert release[2] == beat[2]
    assert stop == ("terminals-stop",)
    assert host.closed == 1


def test_another_server_joins_the_same_name_with_its_own_credential() -> None:
    _save()
    other = _Host()

    connect(
        "box-1",
        server="https://Other.example:8443",
        client_for=other,
        read_token=lambda: _TOKEN,
        sleep=lambda _: None,
    )

    # The second server is joined: the first one's credential would be refused there.
    assert [call[0] for call in other.calls] == ["join", "beat", "beat"]
    machines = state_dir() / "rekursiv-ai" / "trax" / "machines"
    assert sorted(
        path.relative_to(machines).as_posix() for path in machines.rglob("*")
    ) == [
        "other.example_8443",
        "other.example_8443/box-1.credential",
        "trax.example",
        "trax.example/box-1.credential",
    ]


@pytest.mark.parametrize(
    ("installed", "reported"),
    [({"trackinizer": "9.9"}, "9.9"), ({}, "")],
    ids=["installed", "source checkout"],
)
def test_the_host_reports_the_installed_trackinizer_version_or_none(
    monkeypatch: pytest.MonkeyPatch,
    installed: dict[str, str],
    reported: str,
) -> None:
    monkeypatch.setattr(metadata, "version", partial(_version_of, installed=installed))
    _save()
    host = _Host()

    connect(
        "box-1",
        server=_SERVER,
        client_for=host,
        read_token=_no_token,
        sleep=lambda _: None,
    )

    assert [call[3] for call in host.calls] == [reported, reported]


def test_a_server_url_with_no_host_is_refused() -> None:
    with pytest.raises(ClientError, match="names no server host"):
        connect(
            "box-1",
            server="not a url",
            client_for=_Host(),
            read_token=_no_token,
            sleep=lambda _: None,
        )


def _no_token() -> str:
    raise AssertionError("a saved credential must not read a token")


def _version_of(name: str, *, installed: dict[str, str]) -> str:
    """Answer as ``importlib.metadata.version`` would with only ``installed`` present."""
    if name not in installed:
        raise metadata.PackageNotFoundError(name)
    return installed[name]


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
