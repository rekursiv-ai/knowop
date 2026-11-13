"""Tests for the daemon loop, against a scripted client."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import partial
from importlib import metadata
from typing import TYPE_CHECKING, Final

import getpass
import os
import platform
import stat
import uuid

from trackinizer.client.errors import ClientError
from trackinizer.lib.userdirs import state_dir
from trackinizer.wire.wire_daemon import (
    HEARTBEAT_SEC,
    HeartbeatResponse,
    JoinRequest,
    JoinResponse,
)
from trackinizer.wire.wire_session_relay import REMOTE_AGENTS_FACT, SESSIONS_FACT

import pytest

from knowop.connect import daemon
from knowop.connect.daemon import (
    connect,
    host_facts,
    is_sandbox,
    set_sandbox,
)
from knowop.connect.wake import Wake


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from trackinizer.wire.wire_daemon import Claim
    from trackinizer.wire.wire_hosts import Facts


_HOST_ID: Final = uuid.UUID("11111111-1111-1111-1111-111111111111")
_CREDENTIAL: Final = "trax_machine_" + "1" * 32 + "_" + "C" * 43
_OTHER_CREDENTIAL: Final = "trax_machine_" + "2" * 32 + "_" + "D" * 43
"""A credential other than the one the scripted join returns."""
_TOKEN: Final = "enr_" + "0" * 32 + "_" + "S" * 43
_NOW: Final = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
_SERVER: Final = "https://trax.example"

_REVOKED: Final = ClientError("-> 410", status_code=410)
_UNKNOWN: Final = ClientError("-> 401", status_code=401)
_TAKEN: Final = ClientError("-> 409", status_code=409, code="another_daemon_connected")
_REFUSED: Final = ClientError("POST /x failed: connection refused")


class _FakeServer:
    """A scripted server: each heartbeat takes the next outcome in turn.

    By default one heartbeat lands, and the next finds a newer connect took over,
    which ends the daemon and leaves its credential saved.
    """

    def __init__(
        self,
        *,
        beats: Sequence[ClientError | HeartbeatResponse | None] = (None, _TAKEN),
        join_error: ClientError | None = None,
    ) -> None:
        self.beats = list(beats)
        self.join_error = join_error
        self.calls: list[tuple[str | uuid.UUID | Facts | None, ...]] = []
        self.keys: list[str] = []
        self.closed = 0

    def __call__(self, key: str) -> _FakeServer:
        self.keys.append(key)
        return self

    def join_host(
        self,
        name: str,
        *,
        token: str,
        instance: uuid.UUID,
        daemon_version: str,
        facts: Facts,
    ) -> JoinResponse:
        self.calls.append(("join", name, token, instance, daemon_version, facts))
        if self.join_error is not None:
            raise self.join_error
        return JoinResponse(host_id=_HOST_ID, credential=_CREDENTIAL)

    def heartbeat_host(
        self,
        host_id: uuid.UUID,
        *,
        instance: uuid.UUID,
        daemon_version: str,
        facts: Facts | None = None,
        claim: Claim = "keep",
    ) -> HeartbeatResponse:
        self.calls.append(("beat", host_id, instance, daemon_version, facts, claim))
        outcome = self.beats.pop(0)
        if isinstance(outcome, ClientError):
            raise outcome
        return HeartbeatResponse(server_time=_NOW) if outcome is None else outcome

    def close(self) -> None:
        self.closed += 1


# ``""`` is no account: the path then skips that directory, as ``connect`` does.
def _save(*, credential: str = _CREDENTIAL, account: str = "") -> None:
    """Save box-1's credential for ``account``, as a first ``connect`` would have."""
    hosts = state_dir() / "rekursiv-ai" / "trax" / "hosts" / "trax.example"
    path = hosts / account / "box-1.credential"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        JoinResponse(host_id=_HOST_ID, credential=credential).model_dump_json(),
    )


def _held(account: str = "") -> str | None:
    """Return the credential saved for box-1 and ``account``; ``None`` when none is."""
    hosts = state_dir() / "rekursiv-ai" / "trax" / "hosts" / "trax.example"
    path = hosts / account / "box-1.credential"
    if not path.exists():
        return None
    return JoinResponse.model_validate_json(path.read_text()).credential


def _connect(fake: _FakeServer, *, stdin: str = f"{_TOKEN}\n") -> list[float]:
    """Connect box-1 with ``stdin`` as its first line, returning each wait."""
    slept: list[float] = []
    connect(
        "box-1",
        server=_SERVER,
        credential_client=fake,
        read_token=lambda: stdin,
        sleep=slept.append,
    )
    return slept


def _connect_as(name: str, fake: _FakeServer, *, sandbox: bool = False) -> list[float]:
    """Connect ``name`` with the token on stdin, declaring a sandbox if asked."""
    slept: list[float] = []
    connect(
        name,
        server=_SERVER,
        credential_client=fake,
        read_token=lambda: f"{_TOKEN}\n",
        sleep=slept.append,
        sandbox=sandbox,
    )
    return slept


def test_a_computer_is_no_sandbox_until_a_connect_declares_it() -> None:
    _save()

    _ = _connect_as("box-1", _FakeServer())

    assert not is_sandbox()


def test_a_first_connect_with_sandbox_records_it_beside_the_credentials_and_says_so(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ = _connect_as("box-1", _FakeServer(), sandbox=True)

    assert is_sandbox()
    hosts = state_dir() / "rekursiv-ai" / "trax" / "hosts"
    assert (hosts / ".sandbox").is_file()
    assert (hosts / "trax.example" / "box-1.credential").is_file()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "joined: host box-1"
    assert lines[1].startswith("sandbox: on (")
    assert lines[2:] == ["taken over: host box-1 is served by a newer connect"]


def test_a_heartbeat_taken_over_names_its_host_and_ends_as_taken_over() -> None:
    fake = _FakeServer(beats=[_TAKEN])
    instance = uuid.uuid4()

    ending = daemon._heartbeat(
        fake,
        name="box-1",
        server=_SERVER,
        host_id=_HOST_ID,
        instance=instance,
        version="1.2.3",
        facts=None,
        sleep=lambda _: None,
        serve=lambda: None,
        rejoin=False,
    )

    assert ending == "taken over"
    assert fake.calls == [("beat", _HOST_ID, instance, "1.2.3", None, "take")]


def test_a_saved_credential_connect_with_sandbox_records_it_too(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save()

    _ = _connect_as("box-1", _FakeServer(), sandbox=True)

    assert is_sandbox()
    assert capsys.readouterr().out.count("sandbox: on") == 1


def test_a_connect_without_the_flag_clears_the_declaration_and_says_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save()
    set_sandbox(on=True)

    _ = _connect_as("box-1", _FakeServer())

    assert not is_sandbox()
    assert "sandbox" not in capsys.readouterr().out


def test_the_declaration_belongs_to_the_computer_not_to_one_host_name() -> None:
    _ = _connect_as("box-1", _FakeServer(), sandbox=True)
    assert is_sandbox()

    _ = _connect_as("box-2", _FakeServer())

    assert not is_sandbox()


def test_a_refused_join_leaves_the_declaration_as_it_was() -> None:
    refused = ClientError("-> 401", status_code=401)

    with pytest.raises(ClientError):
        _ = _connect_as("box-1", _FakeServer(join_error=refused), sandbox=True)
    assert not is_sandbox()

    set_sandbox(on=True)
    with pytest.raises(ClientError):
        _ = _connect_as("box-1", _FakeServer(join_error=refused))
    assert is_sandbox()


def test_clearing_a_declaration_that_was_never_made_creates_nothing() -> None:
    set_sandbox(on=False)

    assert not (state_dir() / "rekursiv-ai").exists()


def test_a_first_connect_joins_with_the_stdin_token_and_saves_the_credential(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = _FakeServer(beats=[None, None, _TAKEN])

    slept = _connect(fake)

    join, *beats = fake.calls
    assert join[:3] == ("join", "box-1", _TOKEN)
    assert join[5] == host_facts()
    # The join stored the facts, so no heartbeat repeats them.
    assert [beat[4] for beat in beats] == [None, None, None]
    assert [beat[2] for beat in beats] == [join[3]] * 3
    assert fake.keys == ["", _CREDENTIAL]
    assert fake.closed == 2
    assert slept == [HEARTBEAT_SEC, HEARTBEAT_SEC]
    saved = JoinResponse.model_validate_json(
        (
            state_dir()
            / "rekursiv-ai"
            / "trax"
            / "hosts"
            / "trax.example"
            / "box-1.credential"
        ).read_text(),
    )
    assert (saved.host_id, saved.credential) == (_HOST_ID, _CREDENTIAL)
    assert capsys.readouterr().out.splitlines() == [
        "joined: host box-1",
        "taken over: host box-1 is served by a newer connect",
    ]


def test_the_saved_credential_is_readable_by_this_user_only() -> None:
    _connect(_FakeServer())

    path = (
        state_dir()
        / "rekursiv-ai"
        / "trax"
        / "hosts"
        / "trax.example"
        / "box-1.credential"
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_a_saved_credential_skips_the_join_and_sends_facts_once() -> None:
    _save()
    fake = _FakeServer(beats=[None, None, _REVOKED])

    connect(
        "box-1",
        server=_SERVER,
        credential_client=fake,
        read_token=_no_token,
        sleep=lambda _: None,
    )

    assert [call[0] for call in fake.calls] == ["beat"] * 3
    assert [call[4] for call in fake.calls] == [host_facts(), None, None]
    assert fake.keys == [_CREDENTIAL]


def test_a_personal_credential_is_kept_under_its_account(
    capsys: pytest.CaptureFixture[str],
) -> None:
    connect(
        "box-1",
        server=_SERVER,
        credential_client=_FakeServer(),
        read_token=lambda: _TOKEN,
        sleep=lambda _: None,
        account="Ada@X.y",
    )

    hosts = state_dir() / "rekursiv-ai" / "trax" / "hosts" / "trax.example"
    assert sorted(path.name for path in hosts.rglob("*.credential")) == [
        "box-1.credential",
    ]
    assert (hosts / "ada@x.y" / "box-1.credential").is_file()
    assert capsys.readouterr().out.splitlines()[0] == "joined: host box-1 as ada@x.y"


def test_an_accounts_own_saved_credential_skips_the_join() -> None:
    connect(
        "box-1",
        server=_SERVER,
        credential_client=_FakeServer(),
        read_token=lambda: _TOKEN,
        sleep=lambda _: None,
        account="ada@x.y",
    )
    fake = _FakeServer()

    connect(
        "box-1",
        server=_SERVER,
        credential_client=fake,
        read_token=_no_token,
        sleep=lambda _: None,
        account="ada@x.y",
    )

    assert [call[0] for call in fake.calls] == ["beat", "beat"]
    assert fake.keys == [_CREDENTIAL]


@pytest.mark.parametrize("saved_by", ["ada@x.y", None], ids=["account", "no-account"])
def test_a_credential_saved_by_anyone_else_is_never_used(saved_by: str | None) -> None:
    connect(
        "box-1",
        server=_SERVER,
        credential_client=_FakeServer(),
        read_token=lambda: _TOKEN,
        sleep=lambda _: None,
        account=saved_by,
    )
    fake = _FakeServer()

    connect(
        "box-1",
        server=_SERVER,
        credential_client=fake,
        read_token=lambda: _TOKEN,
        sleep=lambda _: None,
        account="bob@x.y",
    )

    # Bob's own enrollment joins: the server, not a file on this computer, decides
    # whether bob may hold box-1.
    assert [call[0] for call in fake.calls] == ["join", "beat", "beat"]
    hosts = state_dir() / "rekursiv-ai" / "trax" / "hosts" / "trax.example"
    assert (hosts / "bob@x.y" / "box-1.credential").is_file()


@pytest.mark.parametrize("account", ["", "../ada@x.y", "ada/b@x.y", ".ada@x.y", "ada"])
def test_an_account_that_is_no_email_or_would_leave_its_directory_is_refused(
    account: str,
) -> None:
    fake = _FakeServer()

    with pytest.raises(ClientError, match="not an account"):
        connect(
            "box-1",
            server=_SERVER,
            credential_client=fake,
            read_token=_no_token,
            sleep=lambda _: None,
            account=account,
        )

    assert fake.calls == []
    assert not (state_dir() / "rekursiv-ai").exists()


def test_each_process_heartbeats_as_its_own_instance() -> None:
    _save()
    first, second = _FakeServer(), _FakeServer()

    for fake in (first, second):
        connect(
            "box-1",
            server=_SERVER,
            credential_client=fake,
            read_token=_no_token,
            sleep=lambda _: None,
        )

    assert first.calls[0][2] != second.calls[0][2]


@pytest.mark.parametrize("answer", [None, ClientError("unreachable")])
def test_a_wake_from_a_pause_heartbeats_at_once_and_ends_with_the_daemon(
    answer: ClientError | None,
) -> None:
    _save()
    fake = _FakeServer(beats=(None, answer, _TAKEN))
    wall = [1_000.0]
    wake = Wake(wall=lambda: wall[0], clock=lambda: 0.0)
    slept: list[float] = []

    def paused(seconds: float) -> None:
        # The heartbeat's sleep spans a pause: the wall clock leaps, the monotonic not.
        slept.append(seconds)
        wall[0] += 30.0
        assert wake.check()

    connect(
        "box-1",
        server=_SERVER,
        credential_client=fake,
        read_token=_no_token,
        sleep=paused,
        wake=wake,
    )
    wall[0] += 30.0
    assert wake.check()

    assert [call[-1] for call in fake.calls] == ["take", "keep", "keep"]
    # Each heartbeat, the wake's too, names the same host, instance and version.
    assert {call[1:4] for call in fake.calls} == {fake.calls[0][1:4]}
    assert fake.calls[0][1] == _HOST_ID
    # The wake's heartbeat took the answer within the one pause; the next ended it.
    assert slept == [HEARTBEAT_SEC]


def test_lost_contact_is_retried_sooner_at_first_and_never_ends_the_daemon(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save()
    # 500 is the first status past the answers a retry cannot change.
    down = ClientError("-> 500", status_code=500)
    fake = _FakeServer(
        beats=[_REFUSED, down, _REFUSED, _REFUSED, _REFUSED, None, _REFUSED, _REVOKED],
    )
    delays: list[float] = []

    connect(
        "box-1",
        server=_SERVER,
        credential_client=fake,
        read_token=_no_token,
        sleep=delays.append,
    )

    assert delays == [2.0, 4.0, 8.0, 15.0, 15.0, HEARTBEAT_SEC, 2.0]
    # Facts and the take ride the first heartbeat that lands, not the first one tried.
    assert [call[4] is not None for call in fake.calls[:6]] == [True] * 6
    assert [call[5] for call in fake.calls] == [*["take"] * 6, "keep", "keep"]
    assert fake.calls[6][4] is None
    assert capsys.readouterr().err.count("retrying") == 6


def test_a_daemon_the_server_calls_outdated_says_how_to_upgrade_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save()
    outdated = HeartbeatResponse(server_time=_NOW, daemon_outdated=True)

    _connect(_FakeServer(beats=[outdated, _REFUSED, outdated, outdated, _REVOKED]))

    line = f"knowop: this knowop is too old for {_SERVER}: run uvx knowop@latest"
    assert capsys.readouterr().err.splitlines().count(f"{line} connect {_SERVER}") == 1


def test_a_daemon_the_server_does_not_call_outdated_says_nothing_of_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save()

    _connect(_FakeServer(beats=[None, None, _REVOKED]))

    assert "too old" not in capsys.readouterr().err


def test_hours_offline_never_overflow_the_backoff() -> None:
    _save()
    refused = ClientError("POST /x failed: connection refused")
    fake = _FakeServer(beats=[*[refused] * 1_100, None, _REVOKED])
    delays: list[float] = []

    connect(
        "box-1",
        server=_SERVER,
        credential_client=fake,
        read_token=_no_token,
        sleep=delays.append,
    )

    assert len(delays) == 1_101
    assert set(delays[4:]) == {float(HEARTBEAT_SEC)}


# A 401 before any heartbeat landed is a saved credential the server forgot, which
# the connect replaces (see the tests of joining again); once serving, it ends the daemon.
@pytest.mark.parametrize("status", [400, 401, 404, 422, 499])
def test_an_answer_a_retry_cannot_change_ends_the_daemon(status: int) -> None:
    _save()
    fake = _FakeServer(beats=[None, ClientError(f"-> {status}", status_code=status)])

    with pytest.raises(ClientError) as err:
        connect(
            "box-1",
            server=_SERVER,
            credential_client=fake,
            read_token=_no_token,
            sleep=lambda _: None,
        )

    assert err.value.status_code == status
    assert _held() == _CREDENTIAL


def test_an_empty_stdin_never_joins() -> None:
    fake = _FakeServer()

    with pytest.raises(ClientError, match="enrollment token from stdin"):
        _connect(fake, stdin="  \n")

    assert fake.calls == []
    assert not (
        state_dir()
        / "rekursiv-ai"
        / "trax"
        / "hosts"
        / "trax.example"
        / "box-1.credential"
    ).exists()


def test_a_refused_join_saves_nothing() -> None:
    fake = _FakeServer(join_error=ClientError("-> 401", status_code=401))

    with pytest.raises(ClientError):
        _connect(fake)

    assert not (
        state_dir()
        / "rekursiv-ai"
        / "trax"
        / "hosts"
        / "trax.example"
        / "box-1.credential"
    ).exists()
    assert [call[0] for call in fake.calls] == ["join"]


def test_a_corrupt_credential_file_is_named_without_quoting_it() -> None:
    path = (
        state_dir()
        / "rekursiv-ai"
        / "trax"
        / "hosts"
        / "trax.example"
        / "box-1.credential"
    )
    path.parent.mkdir(parents=True)
    path.write_text(f'{{"credential": "{_CREDENTIAL}"}}')

    with pytest.raises(ClientError) as err:
        connect(
            "box-1",
            server=_SERVER,
            credential_client=_FakeServer(),
            read_token=_no_token,
            sleep=lambda _: None,
        )

    assert str(path) in str(err.value)
    assert _CREDENTIAL not in str(err.value)
    assert err.value.__cause__ is None


@pytest.mark.parametrize(
    ("refusal", "account", "said"),
    [
        (_REVOKED, "ada@x.y", "rejoined: host box-1 as ada@x.y"),
        (_UNKNOWN, "ada@x.y", "rejoined: host box-1 as ada@x.y"),
        (_REVOKED, "", "rejoined: host box-1"),
    ],
    ids=["revoked", "unknown", "piped token"],
)
def test_a_saved_credential_refused_before_it_served_is_replaced_by_a_join(
    refusal: ClientError,
    account: str,
    said: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A disconnect, or a deleted host, since the last connect: this one joins."""
    _save(credential=_OTHER_CREDENTIAL, account=account)
    fake = _FakeServer(beats=[refusal, None, _TAKEN])

    connect(
        "box-1",
        server=_SERVER,
        credential_client=fake,
        read_token=lambda: _TOKEN,
        sleep=lambda _: None,
        account=account or None,
    )

    assert [call[0] for call in fake.calls] == ["beat", "join", "beat", "beat"]
    assert fake.calls[1][2] == _TOKEN
    assert fake.keys == [_OTHER_CREDENTIAL, "", _CREDENTIAL]
    # The join stored the facts, so the heartbeats after it do not repeat them.
    assert [call[4] for call in fake.calls[2:]] == [None, None]
    assert [call[5] for call in fake.calls[2:]] == ["take", "keep"]
    assert _held(account) == _CREDENTIAL
    assert capsys.readouterr().out.splitlines()[0] == (
        f"{said}; the server refused its saved credential"
    )


def _refuse_enrollment() -> str:
    raise ClientError("POST /api/hosts/enroll -> 401", status_code=401)


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
    fake = _FakeServer(beats=[_REVOKED], join_error=join_error)

    with pytest.raises(ClientError) as err:
        connect(
            "box-1",
            server=_SERVER,
            credential_client=fake,
            read_token=read_token,
            sleep=lambda _: None,
            account=account or None,
        )

    assert str(err.value) == f"host box-1 could not join {_SERVER} again; {hint}"
    assert err.value.status_code == 401
    assert _held(account) is None
    assert [call[0] for call in fake.calls] == asked


def test_a_rejoin_that_fails_another_way_says_what_the_server_said() -> None:
    _save(credential=_OTHER_CREDENTIAL)
    down = ClientError("-> 503", status_code=503)

    with pytest.raises(ClientError) as err:
        _connect(_FakeServer(beats=[_REVOKED], join_error=down))

    assert err.value is down
    assert _held() is None


def test_a_credential_refused_right_after_a_rejoin_ends_the_daemon_without_another(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save(credential=_OTHER_CREDENTIAL)
    fake = _FakeServer(beats=[_REVOKED, _REVOKED, None, _TAKEN])

    _connect(fake)

    assert [call[0] for call in fake.calls] == ["beat", "join", "beat"]
    assert _held() is None
    assert capsys.readouterr().out.splitlines()[1:] == ["revoked: host box-1"]


def test_a_credential_revoked_right_after_a_first_join_is_forgotten_not_joined_again(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = _FakeServer(beats=[_REVOKED, None, _TAKEN])

    _connect(fake)

    assert [call[0] for call in fake.calls] == ["join", "beat"]
    assert _held() is None
    assert capsys.readouterr().out == "joined: host box-1\nrevoked: host box-1\n"


def test_a_host_revoked_while_it_serves_ends_the_daemon_and_forgets_the_credential(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save()
    fake = _FakeServer(beats=[None, _REVOKED])

    connect(
        "box-1",
        server=_SERVER,
        credential_client=fake,
        read_token=_no_token,
        sleep=lambda _: None,
    )

    assert [call[0] for call in fake.calls] == ["beat", "beat"]
    assert _held() is None
    assert capsys.readouterr().out == "revoked: host box-1\n"


def test_a_credential_another_connect_saved_meanwhile_outlives_the_revoke() -> None:
    _save()

    def another_connect_joins(seconds: float) -> None:
        del seconds
        _save(credential=_OTHER_CREDENTIAL)

    connect(
        "box-1",
        server=_SERVER,
        credential_client=_FakeServer(beats=[None, _REVOKED]),
        read_token=_no_token,
        sleep=another_connect_joins,
    )

    assert _held() == _OTHER_CREDENTIAL


def test_daemon_facts_fit_the_join_contract() -> None:
    facts = host_facts()

    assert set(facts) == {
        "os",
        "arch",
        "cpus",
        "memory_mb",
        "user",
        "pid",
        "started_at",
        "knowop_version",
    }
    cpus, memory_mb = facts["cpus"], facts["memory_mb"]
    assert isinstance(cpus, int)
    assert isinstance(memory_mb, int)
    assert (cpus > 0, memory_mb > 0) == (True, True)
    joined = JoinRequest(
        name="box-1",
        token=_TOKEN,
        instance=uuid.uuid4(),
        daemon_version="",
        facts=facts,
    )
    assert joined.facts == facts


def test_daemon_facts_report_what_the_daemon_says_about_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sysconf = {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 2**20}

    def fake_sysconf(name: str) -> int:
        return sysconf[name]

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    monkeypatch.setattr(os, "sysconf", fake_sysconf)
    monkeypatch.setattr(getpass, "getuser", lambda: "ada")
    monkeypatch.setattr(os, "getpid", lambda: 4242)
    installed = {"knowop": "0.1.6"}
    monkeypatch.setattr(
        metadata,
        "version",
        partial(_installed_version, installed=installed),
    )

    facts = host_facts()

    assert facts == {
        "os": "linux",
        "arch": "aarch64",
        "cpus": 0,
        "memory_mb": 4096,
        "user": "ada",
        "pid": 4242,
        "started_at": facts["started_at"],
        "knowop_version": "0.1.6",
    }
    assert host_facts(terminal=True)["terminal"] is True
    assert "files" not in host_facts(terminal=True)
    assert host_facts(tools=True)["tools"] is True
    assert set(host_facts(terminal=True, tools=True)) - set(host_facts()) == {
        "terminal",
        "tools",
    }


def test_every_report_of_one_daemon_names_the_same_start_in_utc() -> None:
    """Read afresh: another test's report may have cached the start already."""
    daemon._started_at.cache_clear()
    started_at = host_facts()["started_at"]

    assert isinstance(started_at, str)
    started = datetime.fromisoformat(started_at)
    assert (started.tzinfo, started.microsecond) == (UTC, 0)
    assert started <= datetime.now(UTC)
    assert host_facts(terminal=True)["started_at"] == started_at


def test_a_join_only_connect_joins_saves_and_serves_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = _FakeServer(beats=[])

    connect(
        "box-1",
        server=_SERVER,
        credential_client=fake,
        read_token=lambda: _TOKEN,
        sleep=_never_waits,
        sandbox=True,
        join_only=True,
    )

    assert [call[0] for call in fake.calls] == ["join"]
    assert _held() == _CREDENTIAL
    # The connect that serves records the sandbox, so this one leaves it.
    assert not is_sandbox()
    assert capsys.readouterr().out == "joined: host box-1\n"


def test_a_join_only_connect_with_a_saved_credential_asks_nothing() -> None:
    _save()
    fake = _FakeServer(beats=[])

    connect(
        "box-1",
        server=_SERVER,
        credential_client=fake,
        read_token=_no_token,
        sleep=_never_waits,
        join_only=True,
    )

    assert (fake.calls, fake.keys) == ([], [])


class _Terminals:
    """Records when a server starts and stops, relative to the heartbeats."""

    def __init__(
        self,
        calls: list[tuple[str | uuid.UUID | Facts | None, ...]],
        label: str = "terminals",
        *,
        start_error: Exception | None = None,
    ) -> None:
        self.calls = calls
        self.label = label
        self.start_error = start_error

    def start(self, host_id: uuid.UUID, instance: uuid.UUID) -> None:
        self.calls.append((f"{self.label}-start", host_id, instance))
        if self.start_error is not None:
            raise self.start_error

    def stop(self) -> None:
        self.calls.append((f"{self.label}-stop",))


def test_a_daemon_that_opted_in_reports_terminals_and_stops_them_before_it_returns() -> (
    None
):
    _save()
    fake = _FakeServer(beats=[None, _REVOKED])
    served: list[JoinResponse] = []

    def joined_terminals(joined: JoinResponse) -> _Terminals:
        served.append(joined)
        return _Terminals(fake.calls)

    connect(
        "box-1",
        server=_SERVER,
        credential_client=fake,
        read_token=_no_token,
        sleep=lambda _: None,
        joined_terminals=joined_terminals,
    )

    kinds = [call[0] for call in fake.calls]
    assert kinds == ["beat", "terminals-start", "beat", "terminals-stop"]
    assert fake.calls[0][4] == host_facts(terminal=True)
    assert fake.calls[1][1] == _HOST_ID
    assert fake.calls[1][2] == fake.calls[0][2]
    assert [joined.credential for joined in served] == [_CREDENTIAL]


def test_no_server_starts_before_a_heartbeat_has_taken_the_host() -> None:
    # The server refuses a wait from an instance that does not hold the host, so a
    # server started first would be refused until the take lands.
    _save()
    fake = _FakeServer(beats=[_REFUSED, None, _REVOKED])

    connect(
        "box-1",
        server=_SERVER,
        credential_client=fake,
        read_token=_no_token,
        sleep=lambda _: None,
        joined_terminals=lambda _joined: _Terminals(fake.calls),
        tools_for=lambda _joined: _Terminals(fake.calls, "tools"),
        jobs_for=lambda _joined: _Terminals(fake.calls, "jobs"),
    )

    assert [call[0] for call in fake.calls] == [
        "beat",
        "beat",
        "terminals-start",
        "tools-start",
        "jobs-start",
        "beat",
        "jobs-stop",
        "tools-stop",
        "terminals-stop",
    ]
    assert [call[5] for call in fake.calls if call[0] == "beat"] == [
        "take",
        "take",
        "keep",
    ]


def test_a_daemon_that_serves_tools_reports_it_and_stops_it_before_the_terminals() -> (
    None
):
    _save()
    fake = _FakeServer(beats=[None, _REVOKED])

    connect(
        "box-1",
        server=_SERVER,
        credential_client=fake,
        read_token=_no_token,
        sleep=lambda _: None,
        joined_terminals=lambda _joined: _Terminals(fake.calls),
        tools_for=lambda _joined: _Terminals(fake.calls, "tools"),
    )

    kinds = [call[0] for call in fake.calls]
    assert kinds == [
        "beat",
        "terminals-start",
        "tools-start",
        "beat",
        "tools-stop",
        "terminals-stop",
    ]
    assert fake.calls[0][4] == host_facts(terminal=True, tools=True)
    assert fake.calls[2][1:] == fake.calls[1][1:]


def test_a_daemon_that_serves_tools_alone_reports_tools_and_no_terminal() -> None:
    _save()
    fake = _FakeServer()

    connect(
        "box-1",
        server=_SERVER,
        credential_client=fake,
        read_token=_no_token,
        sleep=lambda _: None,
        tools_for=lambda _joined: _Terminals(fake.calls, "tools"),
    )

    assert fake.calls[0][4] == {**host_facts(), "tools": True}
    assert "terminal" not in host_facts(tools=True)


def test_a_daemon_that_runs_jobs_says_what_it_offers_and_stops_them_first() -> None:
    _save()
    fake = _FakeServer(beats=[None, _REVOKED])
    offered = {"gpus": 2, "jobs_memory_mb": 61_440}

    connect(
        "box-1",
        server=_SERVER,
        credential_client=fake,
        read_token=_no_token,
        sleep=lambda _: None,
        tools_for=lambda _joined: _Terminals(fake.calls, "tools"),
        jobs_for=lambda _joined: _Terminals(fake.calls, "jobs"),
        job_facts=offered,
    )

    assert [call[0] for call in fake.calls] == [
        "beat",
        "tools-start",
        "jobs-start",
        "beat",
        "jobs-stop",
        "tools-stop",
    ]
    assert fake.calls[0][4] == {
        **host_facts(tools=True),
        "jobs": True,
        "gpus": 2,
        "jobs_memory_mb": 61_440,
    }


def test_a_daemon_that_starts_agents_says_so_and_stops_that_first() -> None:
    _save()
    fake = _FakeServer(beats=[None, _REVOKED])

    connect(
        "box-1",
        server=_SERVER,
        credential_client=fake,
        read_token=_no_token,
        sleep=lambda _: None,
        tools_for=lambda _joined: _Terminals(fake.calls, "tools"),
        sessions_for=lambda _joined: _Terminals(fake.calls, "sessions"),
    )

    assert [call[0] for call in fake.calls] == [
        "beat",
        "tools-start",
        "sessions-start",
        "beat",
        "sessions-stop",
        "tools-stop",
    ]
    assert fake.calls[0][4] == {**host_facts(tools=True), "sessions": True}
    assert fake.calls[2][1:] == fake.calls[1][1:]


@pytest.mark.parametrize(
    ("sessions", "remote_agents", "said"),
    [
        (True, True, {"sessions": True, "remote_agents": True}),
        (True, False, {"sessions": True}),
        # Nothing to confine where no agent is started.
        (False, True, {}),
    ],
)
def test_a_daemon_says_it_runs_remote_agents_only_beside_the_agents_it_confines(
    *,
    sessions: bool,
    remote_agents: bool,
    said: dict[str, bool],
) -> None:
    _save()
    fake = _FakeServer()

    connect(
        "box-1",
        server=_SERVER,
        credential_client=fake,
        read_token=_no_token,
        sleep=lambda _: None,
        sessions_for=(
            (lambda _joined: _Terminals(fake.calls, "sessions")) if sessions else None
        ),
        remote_agents=remote_agents,
    )

    assert fake.calls[0][4] == {**host_facts(), **said}


def test_the_session_facts_are_the_wires() -> None:
    assert host_facts(sessions=True)[SESSIONS_FACT] is True
    assert host_facts(sessions=True, remote_agents=True)[REMOTE_AGENTS_FACT] is True
    assert {SESSIONS_FACT, REMOTE_AGENTS_FACT}.isdisjoint(
        host_facts(tools=True, jobs={}),
    )


def test_a_daemon_that_runs_no_jobs_says_nothing_of_them() -> None:
    _save()
    fake = _FakeServer()

    connect(
        "box-1",
        server=_SERVER,
        credential_client=fake,
        read_token=_no_token,
        sleep=lambda _: None,
        job_facts={"gpus": 1},
    )

    sent = fake.calls[0][4]
    assert isinstance(sent, dict)
    assert {"jobs", "gpus"}.isdisjoint(sent)
    assert host_facts(jobs={})["jobs"] is True


def test_a_tool_server_that_will_not_start_stops_the_terminals_already_running() -> (
    None
):
    _save()
    fake = _FakeServer(beats=[None])

    with pytest.raises(RuntimeError, match="no bash"):
        connect(
            "box-1",
            server=_SERVER,
            credential_client=fake,
            read_token=_no_token,
            sleep=lambda _: None,
            joined_terminals=lambda _joined: _Terminals(fake.calls),
            tools_for=lambda _joined: _Terminals(
                fake.calls,
                "tools",
                start_error=RuntimeError("no bash"),
            ),
        )

    assert [call[0] for call in fake.calls] == [
        "beat",
        "terminals-start",
        "tools-start",
        "terminals-stop",
    ]


def test_terminals_stop_when_the_daemon_ends_on_an_error_too() -> None:
    _save()
    fake = _FakeServer(beats=[None, ClientError("-> 403", status_code=403)])

    with pytest.raises(ClientError):
        connect(
            "box-1",
            server=_SERVER,
            credential_client=fake,
            read_token=_no_token,
            sleep=lambda _: None,
            joined_terminals=lambda _joined: _Terminals(fake.calls),
        )

    assert fake.calls[-1] == ("terminals-stop",)


def test_a_daemon_that_did_not_opt_in_serves_no_terminals_and_says_so() -> None:
    _save()
    fake = _FakeServer()

    connect(
        "box-1",
        server=_SERVER,
        credential_client=fake,
        read_token=_no_token,
        sleep=lambda _: None,
    )

    assert "terminal" not in host_facts()
    assert [call[0] for call in fake.calls] == ["beat", "beat"]
    assert fake.calls[0][4] == host_facts()


def test_a_daemon_a_newer_connect_took_over_stops_its_terminals_and_returns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save()
    fake = _FakeServer(beats=[None, _TAKEN])
    slept: list[float] = []

    connect(
        "box-1",
        server=_SERVER,
        credential_client=fake,
        read_token=_no_token,
        sleep=slept.append,
        joined_terminals=lambda _joined: _Terminals(fake.calls),
    )

    kinds = [call[0] for call in fake.calls]
    assert kinds == ["beat", "terminals-start", "beat", "terminals-stop"]
    assert slept == [HEARTBEAT_SEC]
    assert capsys.readouterr().out == (
        "taken over: host box-1 is served by a newer connect\n"
    )


@pytest.mark.parametrize("answer", [None, _REFUSED])
def test_ctrl_c_releases_the_host_once_then_stops_terminals_and_goes_on(
    answer: ClientError | None,
) -> None:
    _save()
    fake = _FakeServer(beats=[None, answer])

    def interrupt(seconds: float) -> None:
        del seconds
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        connect(
            "box-1",
            server=_SERVER,
            credential_client=fake,
            read_token=_no_token,
            sleep=interrupt,
            joined_terminals=lambda _joined: _Terminals(fake.calls),
        )

    beat, _, release, stop = fake.calls
    # The same host, instance and version as every heartbeat, and no facts.
    assert beat[:2] == ("beat", _HOST_ID)
    assert release == (*beat[:4], None, "release")
    assert stop == ("terminals-stop",)
    assert fake.closed == 1


def test_another_server_joins_the_same_name_with_its_own_credential() -> None:
    _save()
    other = _FakeServer()

    connect(
        "box-1",
        server="https://Other.example:8443",
        credential_client=other,
        read_token=lambda: _TOKEN,
        sleep=lambda _: None,
    )

    # The second server is joined: the first one's credential would be refused there.
    assert [call[0] for call in other.calls] == ["join", "beat", "beat"]
    hosts = state_dir() / "rekursiv-ai" / "trax" / "hosts"
    assert sorted(path.relative_to(hosts).as_posix() for path in hosts.rglob("*")) == [
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
def test_the_daemon_reports_the_installed_trackinizer_version_or_none(
    monkeypatch: pytest.MonkeyPatch,
    installed: dict[str, str],
    reported: str,
) -> None:
    monkeypatch.setattr(
        metadata,
        "version",
        partial(_installed_version, installed=installed),
    )
    _save()
    fake = _FakeServer()

    connect(
        "box-1",
        server=_SERVER,
        credential_client=fake,
        read_token=_no_token,
        sleep=lambda _: None,
    )

    assert [call[3] for call in fake.calls] == [reported, reported]
    # A first connect's join names it too.
    joining = _FakeServer()
    _ = _connect_as("box-2", joining)
    assert [call[4 if call[0] == "join" else 3] for call in joining.calls] == [
        reported,
    ] * 3


def test_a_daemon_that_runs_jobs_with_nothing_to_offer_says_jobs_alone() -> None:
    _save()
    fake = _FakeServer()

    connect(
        "box-1",
        server=_SERVER,
        credential_client=fake,
        read_token=_no_token,
        sleep=lambda _: None,
        jobs_for=lambda _joined: _Terminals(fake.calls, "jobs"),
    )

    assert fake.calls[0][4] == {**host_facts(), "jobs": True}


def test_a_server_url_with_no_daemon_is_refused() -> None:
    with pytest.raises(ClientError, match="names no server host"):
        connect(
            "box-1",
            server="not a url",
            credential_client=_FakeServer(),
            read_token=_no_token,
            sleep=lambda _: None,
        )


def _no_token() -> str:
    raise AssertionError("a saved credential must not read a token")


def _never_waits(seconds: float) -> None:
    raise AssertionError(f"a join-only connect must not wait {seconds} s")


def _installed_version(name: str, *, installed: dict[str, str]) -> str:
    """Answer as ``importlib.metadata.version`` would with only ``installed`` present."""
    if name not in installed:
        raise metadata.PackageNotFoundError(name)
    return installed[name]


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
