"""``knowop host`` against a scripted KnowHost API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from trackinizer.client.errors import ClientError
from treekle import PlainTree

import pytest

from knowop.knowhosts import KNOWHOST_WORD, KnowHosts


if TYPE_CHECKING:
    from trackinizer.client.client import Client


_PATH = "/api/addons/knowhost/hosts"
_ID = "0b5e9a10-1c2d-4e5f-8a9b-0c1d2e3f4a5b"
_KNOWHOST: dict[str, PlainTree] = {
    "id": _ID,
    "machine": "box-0b5e9a101c2d",
    "state": "running",
    "started_at": "2026-11-08T12:00:00Z",
    "memory_kept": None,
    "tools": True,
}


@dataclass(slots=True, kw_only=True)
class _Api:
    """The server's KnowHost routes, as a test sets them; every call kept."""

    knowhosts: list[dict[str, PlainTree]] = field(default_factory=list)
    answer: dict[str, PlainTree] = field(default_factory=lambda: dict(_KNOWHOST))
    refuse: int | None = None
    calls: list[tuple[str, str, dict[str, object]]] = field(default_factory=list)

    def get(self, path: str, **options: object) -> PlainTree:
        return self._call("GET", path, options, {"hosts": list(self.knowhosts)})

    def post(self, path: str, **options: object) -> PlainTree:
        return self._call("POST", path, options, self.answer)

    def delete(self, path: str, **options: object) -> PlainTree:
        return self._call("DELETE", path, options, None)

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


def _run(api: _Api, *argv: str) -> None:
    KnowHosts.dispatch(KNOWHOST_WORD, list(argv), lambda: cast("Client", api))


def test_start_waits_past_the_servers_deadline_and_sends_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = _Api()

    _run(api, "start")

    assert api.calls == [("POST", _PATH, {"timeout": 150.0, "retry_attempts": 1})]
    assert capsys.readouterr().out == (
        "box-0b5e9a101c2d is running: open its terminal from Machines.\n"
    )


def test_a_machine_without_tools_is_said_to_serve_terminals_alone(
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = _Api()
    api.answer["tools"] = False

    _run(api, "start")

    assert capsys.readouterr().out.splitlines()[-1] == (
        "Its knowop serves terminals but no tools."
    )


@pytest.mark.parametrize("argv", [[], ["list"]])
def test_list_prints_each_knowhost_by_machine_state_and_id(
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    paused = {
        **_KNOWHOST,
        "machine": "box-2",
        "id": "i2",
        "state": "paused",
    }
    api = _Api(knowhosts=[dict(_KNOWHOST), paused])

    _run(api, *argv)

    assert capsys.readouterr().out.splitlines() == [
        f"box-0b5e9a101c2d  running   {_ID}",
        "box-2  paused    i2",
    ]


def test_list_of_none_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    _run(_Api(), "list")

    assert capsys.readouterr().out == "(no KnowHosts)\n"


@pytest.mark.parametrize("name", ["box-0b5e9a101c2d", _ID])
def test_pause_resume_and_rm_name_a_knowhost_by_machine_or_id(
    capsys: pytest.CaptureFixture[str],
    name: str,
) -> None:
    api = _Api(knowhosts=[dict(_KNOWHOST)])

    _run(api, "pause", name)
    _run(api, "resume", name)
    _run(api, "rm", name)

    assert [(method, path) for method, path, _ in api.calls if method != "GET"] == [
        ("POST", f"{_PATH}/{_ID}/pause"),
        ("POST", f"{_PATH}/{_ID}/resume"),
        ("DELETE", f"{_PATH}/{_ID}"),
    ]
    assert api.calls[3][2] == {"timeout": 150.0, "retry_attempts": 1}
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith(f"{name} is paused. A terminal open across a pause ends")
    assert out[1].startswith("box-0b5e9a101c2d is running again.")
    assert out[2] == f"{name} is gone, and its machine with it."


def test_a_resume_that_rebooted_says_processes_were_lost(
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = _Api(knowhosts=[dict(_KNOWHOST)])
    api.answer["memory_kept"] = False

    _run(api, "resume", _ID)

    assert "It rebooted from its files: its processes were lost." in (
        capsys.readouterr().out.splitlines()
    )


def test_a_knowhost_not_listed_is_named_as_missing() -> None:
    with pytest.raises(ClientError, match=r"\Ayou have no KnowHost box-9\Z"):
        _run(_Api(), "pause", "box-9")


@pytest.mark.parametrize("argv", [["start"], ["list"], ["pause", "box-1"]])
def test_a_server_with_no_knowhost_rents_none(argv: list[str]) -> None:
    with pytest.raises(
        ClientError,
        match=r"\Athis server rents no KnowHosts\Z",
    ) as refused:
        _run(_Api(refuse=404), *argv)

    assert refused.value.status_code == 404


def test_another_refusal_is_the_servers_own() -> None:
    with pytest.raises(ClientError, match=r"\Arefused\Z") as refused:
        _run(_Api(refuse=402), "start")
    with pytest.raises(ClientError, match=r"\Arefused\Z"):
        _run(_Api(refuse=409), "list")

    assert refused.value.status_code == 402


def test_an_unknown_form_says_how_to_use_it() -> None:
    with pytest.raises(ClientError, match=r"\Ausage: knowop host \[start"):
        _run(_Api(), "stop", "box-1", "now")


def test_help_names_every_form_and_asks_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = _Api()

    _run(api, "--help")

    out = capsys.readouterr().out
    assert "knowop host [start | list | pause NAME | resume NAME | rm NAME]" in out
    assert api.calls == []


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
