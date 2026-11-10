"""Tests for ``knowop host``: list, show, set, label and delete; rent a KnowHost."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import io

from trackinizer.client.client import Client
from trackinizer.client.errors import ClientError
from trackinizer.wire.wire_hosts import (
    MAX_HOW_CHARS,
    NAME_PATTERN,
    RESERVED_NAMES,
    Host,
)
from treekle import PlainTree

import httpx2
import pytest

from knowop import hosts
from knowop.hosts import HOST_WORD, Hosts


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def _host(
    name: str,
    *,
    role: str = "",
    how: str = "",
    labels: Sequence[str] = (),
) -> Host:
    return Host(
        name=name,
        role=role,
        how=how,
        labels=list(labels),
        updated_by="alice",
        updated=datetime(2026, 10, 6, 12, 0, tzinfo=UTC),
    )


class _StubClient:
    """Records host calls; raises ``error`` from any call when set."""

    def __init__(
        self,
        hosts: Sequence[Host] = (),
        *,
        error: ClientError | None = None,
    ) -> None:
        self.hosts = list(hosts)
        self.error = error
        self.calls: list[tuple[object, ...]] = []

    def list_hosts(self) -> list[Host]:
        self.calls.append(("list",))
        if self.error is not None:
            raise self.error
        return self.hosts

    def get_host(self, name: str) -> Host:
        self.calls.append(("get", name))
        if self.error is not None:
            raise self.error
        return next(m for m in self.hosts if m.name == name)

    def put_host(
        self,
        name: str,
        *,
        role: str | None = None,
        how: str | None = None,
    ) -> None:
        self.calls.append(("put", name, role, how))
        if self.error is not None:
            raise self.error

    def change_host_labels(
        self,
        name: str,
        *,
        add: Sequence[str] = (),
        remove: Sequence[str] = (),
    ) -> None:
        self.calls.append(("labels", name, tuple(add), tuple(remove)))
        if self.error is not None:
            raise self.error

    def delete_host(self, name: str) -> None:
        self.calls.append(("delete", name))
        if self.error is not None:
            raise self.error

    def get(self, path: str) -> dict[str, list[object]]:
        """Answer the KnowHost listing: a server that rents none, as most here."""
        del path
        raise ClientError("-> 404", status_code=404)


def _run(argv: Sequence[str], client: _StubClient) -> None:
    Hosts.dispatch(HOST_WORD, list(argv), lambda: cast(Client, client))


def _refusal(argv: Sequence[str], client: _StubClient) -> ClientError:
    with pytest.raises(ClientError) as err:
        _run(argv, client=client)
    return err.value


def _stdin(monkeypatch: pytest.MonkeyPatch, data: bytes) -> None:
    monkeypatch.setattr(
        "sys.stdin",
        io.TextIOWrapper(io.BytesIO(data), encoding="utf-8"),
    )


def test_verb_word_is_one_constant() -> None:
    assert Hosts.names == (HOST_WORD,)
    assert HOST_WORD == "host"


def test_bare_host_lists_name_role_labels_and_how(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _StubClient(
        [
            _host("gpu-box", role="dev", how="ssh gpu-box", labels=["gpu", "a100"]),
            _host("cpu-1"),
        ],
    )
    _run([], client=client)
    assert capsys.readouterr().out.splitlines() == [
        "gpu-box  dev  gpu, a100  ssh gpu-box",
        "cpu-1    -    -          -",
    ]
    assert client.calls == [("list",)]


def test_list_cuts_how_to_one_short_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _StubClient(
        [
            _host("a", how="first\nsecond"),
            _host("b", how="x" * 200),
        ],
    )
    _run([], client=client)
    first, second = capsys.readouterr().out.splitlines()
    assert first == "a  -  -  first\\nsecond"
    assert second == f"b  -  -  {'x' * 57}..."


def test_list_and_show_keep_a_label_on_one_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _StubClient([_host("a", labels=["x\ny", "z\r"]), _host("b")])
    _run([], client=client)
    _run(["a"], client=client)
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == ["a  -  x\\ny, z\\r  -", f"b  -  {'-':<9}  -"]
    assert "labels:     x\\ny, z\\r" in lines


def test_bare_host_with_none_registered_says_so(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _run([], client=_StubClient())
    assert capsys.readouterr().out == "(no hosts)\n"


def test_show_prints_every_field(capsys: pytest.CaptureFixture[str]) -> None:
    client = _StubClient(
        [_host("gpu-box", role="dev", how="a\nb", labels=["gpu", "a100"])],
    )
    _run(["gpu-box"], client=client)
    assert capsys.readouterr().out.splitlines() == [
        "host:       gpu-box",
        "role:       dev",
        "labels:     gpu, a100",
        "how:        a\\nb",
        "status:     never",
        "updated by: alice",
        "updated:    2026-10-06T12:00:00+00:00",
    ]
    assert client.calls == [("get", "gpu-box")]


def test_show_of_a_bare_host_marks_what_is_unset(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _run(["cpu-1"], client=_StubClient([_host("cpu-1")]))
    out = capsys.readouterr().out.splitlines()
    assert out[1:4] == [
        "role:       (none)",
        "labels:     (none)",
        "how:        (none)",
    ]


def test_a_bare_field_prints_its_value_unchanged(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _StubClient([_host("gpu-box", role="dev", how="a\nb")])
    _run(["gpu-box", "ROLE"], client=client)
    _run(["gpu-box", "how"], client=client)
    assert capsys.readouterr().out == "dev\na\nb\n"
    assert client.calls == [("get", "gpu-box")] * 2


def test_set_role_creates_the_host(capsys: pytest.CaptureFixture[str]) -> None:
    client = _StubClient()
    _run(["gpu-box", "role", "TO", "dev"], client=client)
    assert client.calls == [("put", "gpu-box", "dev", None)]
    assert capsys.readouterr().out == "set: host gpu-box role\n"


def test_set_how_sends_only_how(capsys: pytest.CaptureFixture[str]) -> None:
    client = _StubClient()
    _run(["gpu-box", "how", "to", "ssh gpu-box; cd /work"], client=client)
    assert client.calls == [("put", "gpu-box", None, "ssh gpu-box; cd /work")]
    assert capsys.readouterr().out == "set: host gpu-box how\n"


def test_an_empty_literal_clears_a_field() -> None:
    client = _StubClient()
    _run(["gpu-box", "role", "to", ""], client=client)
    _run(["gpu-box", "how", "to", ""], client=client)
    assert client.calls == [
        ("put", "gpu-box", "", None),
        ("put", "gpu-box", None, ""),
    ]


def test_how_from_stdin_loses_exactly_one_trailing_newline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _StubClient()
    _stdin(monkeypatch, b"ssh gpu-box\r\n\n")
    _run(["gpu-box", "how", "to", "-"], client=client)
    assert client.calls == [("put", "gpu-box", None, "ssh gpu-box\r\n")]


def test_how_from_a_file_is_read_relative_to_the_caller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "how.txt").write_text("ssh gpu-box\n")
    monkeypatch.chdir(tmp_path)
    client = _StubClient()
    _run(["gpu-box", "how", "to", "@how.txt"], client=client)
    assert client.calls == [("put", "gpu-box", None, "ssh gpu-box")]


def test_an_empty_stdin_or_file_is_refused_instead_of_clearing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "empty.txt").write_text("")
    monkeypatch.chdir(tmp_path)
    client = _StubClient()
    _stdin(monkeypatch, b"")
    assert "empty value" in str(_refusal(["gpu-box", "how", "to", "-"], client=client))
    err = _refusal(["gpu-box", "how", "to", "@empty.txt"], client=client)
    assert "empty value" in str(err)
    assert client.calls == []


@pytest.mark.parametrize(
    ("source", "message"),
    [("@", "@ value requires a path"), ("@missing.txt", "cannot read @missing.txt")],
)
def test_an_unreadable_source_is_refused_before_any_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    message: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    client = _StubClient()
    assert message in str(_refusal(["gpu-box", "how", "to", source], client=client))
    assert client.calls == []


def test_non_utf8_input_is_refused_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _StubClient()
    _stdin(monkeypatch, b"\xff\xfe")
    err = _refusal(["gpu-box", "how", "to", "-"], client=client)
    assert str(err) == "the value is not valid UTF-8"
    assert client.calls == []


def test_label_add_and_del_send_one_label_each(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _StubClient()
    _run(["gpu-box", "label", "add", "gpu"], client=client)
    _run(["gpu-box", "LABEL", "DEL", "a100"], client=client)
    assert client.calls == [
        ("labels", "gpu-box", ("gpu",), ()),
        ("labels", "gpu-box", (), ("a100",)),
    ]
    assert capsys.readouterr().out.splitlines() == [
        "set: host gpu-box label add gpu",
        "set: host gpu-box label del a100",
    ]


def test_delete_unregisters_the_host(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _StubClient()
    _run(["gpu-box", "DEL"], client=client)
    assert client.calls == [("delete", "gpu-box")]
    assert capsys.readouterr().out == "deleted: host gpu-box\n"


@pytest.mark.parametrize("name", sorted(RESERVED_NAMES))
def test_a_reserved_name_is_refused_before_any_request(name: str) -> None:
    client = _StubClient()
    # Bare ``start`` rents a KnowHost: the name is reserved so no host can take it.
    for argv in (
        *([] if name == "start" else [[name]]),
        [name, "role"],
        [name, "role", "to", "dev"],
        [name, "label", "add", "gpu"],
        [name, "del"],
    ):
        assert str(_refusal(argv, client=client)) == (f"host name {name!r} is reserved")
    assert client.calls == []


@pytest.mark.parametrize(
    "name",
    ["GPU", "a_b", "a/b", "a b", "x" * 64, "gpu\n"],
    ids=["upper", "underscore", "slash", "space", "long", "newline"],
)
def test_a_malformed_name_is_refused_before_any_request(name: str) -> None:
    client = _StubClient()
    err = _refusal([name, "role", "to", "dev"], client=client)
    assert str(err) == f"invalid host name {name!r}; expected {NAME_PATTERN}"
    assert client.calls == []


def test_the_longest_name_is_accepted() -> None:
    client = _StubClient()
    _run(["a" * 63, "del"], client=client)
    assert client.calls == [("delete", "a" * 63)]


@pytest.mark.parametrize(
    "argv",
    [
        ["gpu-box", "colour"],
        ["gpu-box", "role", "to"],
        ["gpu-box", "role", "dev"],
        ["gpu-box", "role", "to", "a", "b"],
        ["gpu-box", "label"],
        ["gpu-box", "label", "add"],
        ["gpu-box", "label", "set", "gpu"],
        ["gpu-box", "label", "add", "gpu", "extra"],
        ["gpu-box", "del", "extra"],
        ["gpu-box", "labels"],
    ],
)
def test_a_malformed_command_is_refused_before_any_request(argv: list[str]) -> None:
    client = _StubClient()
    err = _refusal(argv, client=client)
    assert str(err).startswith("usage: knowop host")
    assert client.calls == []


@pytest.mark.parametrize(
    ("argv", "role"),
    [
        ([], "writer"),
        (["gpu-box"], "writer"),
        (["gpu-box", "role"], "writer"),
        (["gpu-box", "role", "to", "dev"], "admin"),
        (["gpu-box", "label", "add", "gpu"], "admin"),
        (["gpu-box", "del"], "admin"),
    ],
)
def test_forbidden_names_the_role_the_action_needs(
    argv: list[str],
    role: str,
) -> None:
    client = _StubClient(error=ClientError("GET ... -> 403", status_code=403))
    assert str(_refusal(argv, client=client)) == f"needs the {role} role"


@pytest.mark.parametrize(
    "argv",
    [
        ["gpu-box"],
        ["gpu-box", "how"],
        ["gpu-box", "label", "del", "gpu"],
        ["gpu-box", "del"],
    ],
)
def test_an_absent_host_is_named(argv: list[str]) -> None:
    client = _StubClient(error=ClientError("... -> 404", status_code=404))
    assert str(_refusal(argv, client=client)) == "no host gpu-box"


def test_a_set_never_reports_a_missing_host() -> None:
    # A set creates the host, so a 404 means the server lacks the route.
    client = _StubClient(error=ClientError("PUT /x -> 404: Not Found", status_code=404))
    assert str(_refusal(["gpu-box", "role", "to", "dev"], client=client)) == (
        "PUT /x -> 404: Not Found"
    )
    assert str(_refusal(["gpu-box", "how", "to", "ssh"], client=client)) == (
        "PUT /x -> 404: Not Found"
    )


def test_other_failures_pass_through() -> None:
    client = _StubClient(error=ClientError("PUT /x -> 503: down", status_code=503))
    assert str(_refusal(["gpu-box", "role", "to", "dev"], client=client)) == (
        "PUT /x -> 503: down"
    )
    client = _StubClient(error=ClientError("PUT /x failed: refused"))
    assert "refused" in str(_refusal(["gpu-box", "del"], client=client))


@pytest.mark.parametrize(
    "argv",
    [
        ["gpu-box", "role", "to", "Dev"],
        ["gpu-box", "how", "to", "x" * (MAX_HOW_CHARS + 1)],
    ],
    ids=["role", "how"],
)
def test_a_value_the_server_would_refuse_never_reaches_it(argv: list[str]) -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(204)

    with Client("https://server") as client:
        client._http.close()
        client._http = httpx2.Client(
            base_url=client.base_url,
            transport=httpx2.MockTransport(handler),
        )
        with pytest.raises(ClientError, match=r"^host 'gpu-box' "):
            Hosts.dispatch(HOST_WORD, argv, lambda: client)
    assert seen == []


def test_host_help_lists_every_form(capsys: pytest.CaptureFixture[str]) -> None:
    client = _StubClient()
    _run(["help"], client=client)
    out = capsys.readouterr().out
    for form in (
        f"knowop {HOST_WORD} NAME role to ROLE",
        f"knowop {HOST_WORD} NAME how to -",
        f"knowop {HOST_WORD} NAME how to @FILE",
        f"knowop {HOST_WORD} NAME label add LABEL",
        f"knowop {HOST_WORD} NAME label del LABEL",
        f"knowop {HOST_WORD} NAME del",
    ):
        assert form in out
    assert "'knowop connect' connects this computer." in out
    assert client.calls == []


@pytest.mark.parametrize("word", ["help", "-h", "--help"])
def test_a_trailing_help_word_shows_help_instead_of_setting_it(
    capsys: pytest.CaptureFixture[str],
    word: str,
) -> None:
    client = _StubClient()
    _run(["gpu-box", "how", "to", word], client=client)
    assert "Usage: knowop host" in capsys.readouterr().out
    assert client.calls == []


@pytest.mark.parametrize(
    "argv",
    [
        ["help"],
        ["gpu-box", "label", "add", "help"],
        ["gpu-box", "label", "del", "-h"],
        ["gpu-box", "label", "add", "--help"],
    ],
)
def test_a_trailing_help_word_is_never_a_name_or_label(
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    client = _StubClient([_host("help")])
    _run(argv, client=client)
    assert "Usage: knowop host" in capsys.readouterr().out
    assert client.calls == []


@pytest.mark.parametrize(
    "argv",
    [
        ["gx10", "connect"],
        ["gx10", "--sandbox"],
        ["gx10", "del", "--sandbox"],
        ["gx10", "role", "--sandbox"],
        ["gx10", "label", "add", "gpu", "--sandbox"],
    ],
)
def test_a_flag_left_on_another_form_is_refused_before_any_request(
    argv: list[str],
) -> None:
    client = _StubClient()

    assert str(_refusal(argv, client=client)).startswith("usage: knowop host")
    assert client.calls == []


def test_a_value_may_still_be_spelled_like_the_flag() -> None:
    client = _StubClient()

    _run(["gx10", "how", "to", "--sandbox"], client=client)

    assert client.calls == [("put", "gx10", None, "--sandbox")]


@pytest.mark.parametrize(
    ("hostname", "name"),
    [("gx10", "gx10"), ("Dev_Box.local", "dev-box"), ("-Lab_PC-.corp", "lab-pc")],
)
def test_host_name_is_the_short_hostname_as_a_host_name(
    monkeypatch: pytest.MonkeyPatch,
    hostname: str,
    name: str,
) -> None:
    """The rule a run reuses to say which host it is on, as connect names it."""
    monkeypatch.setattr("socket.gethostname", lambda: hostname)

    assert hosts.host_name() == name


def test_a_hostname_that_is_not_a_host_name_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("socket.gethostname", lambda: "init")

    with pytest.raises(ClientError, match="reserved"):
        _ = hosts.host_name()


def test_a_hostname_of_only_punctuation_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("socket.gethostname", lambda: "___")

    with pytest.raises(ClientError, match="invalid host name ''"):
        _ = hosts.host_name()


_KNOWHOSTS = "/api/addons/knowhost/hosts"
_ID = "0b5e9a10-1c2d-4e5f-8a9b-0c1d2e3f4a5b"
_KNOWHOST: dict[str, PlainTree] = {
    "id": _ID,
    "host": "knowhost-0b5e9a101c2d",
    "state": "running",
    "started_at": "2026-11-08T12:00:00Z",
    "memory_kept": None,
    "tools": True,
}


@dataclass(slots=True, kw_only=True)
class _KnowHostApi:
    """The server's KnowHost routes and host registry, as a test sets them.

    Every call is kept; ``refuse`` refuses the KnowHost routes alone.
    """

    knowhosts: list[dict[str, PlainTree]] = field(default_factory=list)
    registered: list[str] = field(default_factory=list)
    answer: dict[str, PlainTree] = field(default_factory=lambda: dict(_KNOWHOST))
    refuse: int | None = None
    calls: list[tuple[str, str, dict[str, object]]] = field(default_factory=list)

    def get(self, path: str, **options: object) -> PlainTree:
        return self._call("GET", path, options, {"hosts": list(self.knowhosts)})

    def post(self, path: str, **options: object) -> PlainTree:
        return self._call("POST", path, options, self.answer)

    def delete(self, path: str, **options: object) -> PlainTree:
        return self._call("DELETE", path, options, None)

    def list_hosts(self) -> list[SimpleNamespace]:
        self.calls.append(("REGISTRY", "list", {}))
        return [
            SimpleNamespace(name=name, role="", labels=[], how="")
            for name in self.registered
        ]

    def delete_host(self, name: str) -> None:
        self.calls.append(("REGISTRY", "delete", {"name": name}))

    def _call(
        self,
        method: str,
        path: str,
        options: dict[str, object],
        answer: PlainTree,
    ) -> PlainTree:
        self.calls.append((method, path, options))
        if self.refuse is not None:
            raise ClientError("refused", status_code=self.refuse)
        return answer


def _run_api(api: _KnowHostApi, *argv: str) -> None:
    Hosts.dispatch(HOST_WORD, list(argv), lambda: cast(Client, api))


def test_start_waits_past_the_servers_deadline_and_sends_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = _KnowHostApi()

    _run_api(api, "start")

    assert api.calls == [("POST", _KNOWHOSTS, {"timeout": 150.0, "retry_attempts": 1})]
    assert capsys.readouterr().out == (
        "knowhost-0b5e9a101c2d is running: open its terminal from Hosts.\n"
    )


def test_a_host_without_tools_is_said_to_serve_terminals_alone(
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = _KnowHostApi()
    api.answer["tools"] = False

    _run_api(api, "start")

    assert capsys.readouterr().out.splitlines()[-1] == (
        "Its knowop serves terminals but no tools."
    )


def test_the_listing_names_the_hosts_then_each_knowhost_by_state_and_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    paused = {**_KNOWHOST, "host": "knowhost-2", "id": "i2", "state": "paused"}
    api = _KnowHostApi(knowhosts=[dict(_KNOWHOST), paused], registered=["gx10"])

    _run_api(api)

    assert capsys.readouterr().out.splitlines() == [
        "gx10  -  -  -",
        "",
        "KnowHosts:",
        f"knowhost-0b5e9a101c2d  running   {_ID}",
        "knowhost-2  paused    i2",
    ]


def test_a_listing_on_a_server_that_rents_none_names_the_hosts_alone(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _run_api(_KnowHostApi(refuse=404))

    assert capsys.readouterr().out == "(no hosts)\n"


def test_pause_resume_and_del_reach_the_knowhost_by_its_host_name(
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = _KnowHostApi(knowhosts=[dict(_KNOWHOST)])
    name = "knowhost-0b5e9a101c2d"

    _run_api(api, name, "pause")
    _run_api(api, name, "resume")
    _run_api(api, name, "del")

    assert [(method, path) for method, path, _ in api.calls if method != "GET"] == [
        ("POST", f"{_KNOWHOSTS}/{_ID}/pause"),
        ("POST", f"{_KNOWHOSTS}/{_ID}/resume"),
        ("DELETE", f"{_KNOWHOSTS}/{_ID}"),
    ]
    assert api.calls[3][2] == {"timeout": 150.0, "retry_attempts": 1}
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith(f"{name} is paused. A terminal open across a pause ends")
    assert out[1].startswith(f"{name} is running again.")
    assert out[2] == f"{name} is gone, and its VM with it."


def test_del_of_a_host_that_is_no_knowhost_unregisters_it() -> None:
    api = _KnowHostApi(knowhosts=[dict(_KNOWHOST)])

    _run_api(api, "gx10", "del")

    assert ("REGISTRY", "delete", {"name": "gx10"}) in api.calls
    assert not [call for call in api.calls if call[0] == "DELETE"]


def test_a_resume_that_rebooted_says_processes_were_lost(
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = _KnowHostApi(knowhosts=[dict(_KNOWHOST)])
    api.answer["memory_kept"] = False

    _run_api(api, "knowhost-0b5e9a101c2d", "resume")

    assert "It rebooted from its files: its processes were lost." in (
        capsys.readouterr().out.splitlines()
    )


@pytest.mark.parametrize("refuse", [None, 404])
def test_a_knowhost_not_listed_is_named_as_missing(refuse: int | None) -> None:
    with pytest.raises(ClientError, match=r"\Ayou have no KnowHost knowhost-9\Z"):
        _run_api(_KnowHostApi(refuse=refuse), "knowhost-9", "pause")


def test_a_server_with_no_knowhost_rents_none() -> None:
    with pytest.raises(
        ClientError,
        match=r"\Athis server rents no KnowHosts\Z",
    ) as refused:
        _run_api(_KnowHostApi(refuse=404), "start")

    assert refused.value.status_code == 404


def test_another_refusal_is_the_servers_own() -> None:
    with pytest.raises(ClientError, match=r"\Arefused\Z") as refused:
        _run_api(_KnowHostApi(refuse=402), "start")
    with pytest.raises(ClientError, match=r"\Arefused\Z"):
        _run_api(_KnowHostApi(refuse=409))

    assert refused.value.status_code == 402


def test_start_is_no_host_name() -> None:
    with pytest.raises(ClientError, match=r"\Ahost name 'start' is reserved\Z"):
        _run_api(_KnowHostApi(), "start", "del")


def test_help_names_the_knowhost_forms_and_asks_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = _KnowHostApi()

    _run_api(api, "--help")

    out = capsys.readouterr().out
    assert "knowop host [start | NAME [pause | resume | del" in out
    assert "rent a KnowHost" in out
    assert api.calls == []


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
