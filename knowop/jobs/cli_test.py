"""The ``knowop job`` runner: connection flags, the client, and the exit code."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import subprocess
import sys

from trackinizer.client.client import Client
from trackinizer.client.errors import ClientError
from trackinizer.trax import profile
from trackinizer.trax.client_cache import close_clients
from trackinizer.trax.profile import Profile

import pytest

from knowop.jobs.cli import main
from knowop.jobs.job import JOB_WORD, Job


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from collections.abc import Callable


# Counted from the module's dotted name, not a fixed hop: the export republishes this
# tree shallower.
_ROOT: Final = _CWD.parents[__name__.count(".") - 1]

_PROBE: Final = """\
import sys
sys.path.insert(0, sys.argv[1])
import knowop.jobs.cli
loaded = sorted(
    name.removeprefix("trackinizer.trax.")
    for name in sys.modules
    if name.startswith("trackinizer.trax.")
)
print(" ".join(loaded))
"""

_ALLOWED: Final = frozenset(
    {"client_cache", "commands", "connection", "context", "profile", "render"},
)
"""The trax modules an endpoint may load: no verb, no other endpoint's command."""


class _Seen:
    """What ``Job.dispatch`` was handed."""

    def __init__(self) -> None:
        self.verb = ""
        self.rest: list[str] = []
        self.factory: Callable[[], Client] | None = None


@pytest.fixture(autouse=True)
def no_terminal_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run as a shell that no trax terminal opened; a test that wants one sets it.

    A shell a terminal opened names its server, and with no profile for it every
    command first says how to add one, which the stderr tests here would read.
    """
    for name in ("TRAX_SERVER", "LC_TRAX_SERVER"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> _Seen:
    """Replace the command with one that records its arguments."""
    record = _Seen()

    def dispatch(
        cls: type[Job],
        verb: str,
        rest: list[str],
        client_factory: Callable[[], Client],
    ) -> None:
        del cls
        record.verb = verb
        record.rest = rest
        record.factory = client_factory

    monkeypatch.setattr(Job, "dispatch", classmethod(dispatch))
    return record


@pytest.mark.cli_python_subprocess
def test_importing_the_endpoint_loads_no_trax_verb() -> None:
    """``trax.cli`` and the verbs behind it move out in M1 and M2; this must not care."""
    done = subprocess.run(  # noqa: S603 -- fixed interpreter and probe.
        [sys.executable, "-c", _PROBE, str(_ROOT)],
        capture_output=True,
        text=True,
        check=True,
        cwd=_ROOT,
    )

    assert set(done.stdout.split()) <= _ALLOWED


def test_the_verb_and_its_arguments_reach_the_command(seen: _Seen) -> None:
    assert main(["submit", "--gpus", "2", "--", "python", "x.py"]) == 0

    assert seen.verb == JOB_WORD
    assert seen.rest == ["submit", "--gpus", "2", "--", "python", "x.py"]


def test_connection_flags_before_the_verb_choose_the_client(seen: _Seen) -> None:
    profile.save_profile("jan", Profile(url="http://jan-host:9100", author="jan"))

    assert main(["--profile", "jan", "status", "7"]) == 0

    assert seen.rest == ["status", "7"]
    assert seen.factory is not None
    client = seen.factory()
    try:
        assert (client.base_url, client.author) == ("http://jan-host:9100", "jan")
    finally:
        close_clients()


def test_host_and_port_flags_rewrite_the_profiles_address(seen: _Seen) -> None:
    profile.save_profile("jan", Profile(url="http://jan-host:9100", author="jan"))

    assert main(["--profile=jan", "--host", "gx10", "--port", "8765", "status"]) == 0

    assert seen.rest == ["status"]
    assert seen.factory is not None
    client = seen.factory()
    try:
        assert client.base_url == "http://gx10:8765"
    finally:
        close_clients()


@pytest.mark.parametrize(
    "tokens",
    [
        ["submit", "--", "python", "x.py", "--profile", "other", "--port", "9"],
        ["submit", "--label", "--host", "--", "true"],
        ["results", "7", "--to", "--profile"],
    ],
)
def test_a_flag_after_the_verb_belongs_to_the_command(
    seen: _Seen,
    tokens: list[str],
) -> None:
    assert main(tokens) == 0

    assert seen.rest == tokens


def test_a_leading_help_flag_reaches_the_command(seen: _Seen) -> None:
    assert main(["--profile", "jan", "--help"]) == 0

    assert seen.rest == ["--help"]


def test_help_prints_the_usage_without_a_server(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--help"]) == 0

    assert capsys.readouterr().out.startswith("Usage: knowop job submit")


def test_a_flag_that_lacks_its_value_is_refused_by_the_parser(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as err:
        main(["--profile"])

    assert err.value.code == 2
    assert capsys.readouterr().err.startswith("usage: knowop job")


def test_a_refused_command_names_the_program_and_exits_two(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def dispatch(
        cls: type[Job],
        verb: str,
        rest: list[str],
        client_factory: Callable[[], Client],
    ) -> None:
        del cls, verb, rest, client_factory
        raise ClientError("ssh was not found on PATH")

    monkeypatch.setattr(Job, "dispatch", classmethod(dispatch))

    assert main(["logs", "7"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "knowop job: ssh was not found on PATH\n"


def test_a_shell_from_a_server_without_a_profile_is_told_how_to_log_in(
    monkeypatch: pytest.MonkeyPatch,
    seen: _Seen,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LC_TRAX_SERVER", "https://remote.example")

    assert main(["status"]) == 0

    assert capsys.readouterr().err.startswith(
        "trax: this shell came from https://remote.example, which has no saved profile",
    )
    assert seen.rest == ["status"]


def test_the_shared_clients_are_closed_when_the_command_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile.save_profile("jan", Profile(url="http://jan-host:9100", author="jan"))
    closed: list[Client] = []
    real_close = Client.close

    def close(self: Client) -> None:
        closed.append(self)
        real_close(self)

    def dispatch(
        cls: type[Job],
        verb: str,
        rest: list[str],
        client_factory: Callable[[], Client],
    ) -> None:
        del cls, verb, rest
        _ = client_factory()

    monkeypatch.setattr(Client, "close", close)
    monkeypatch.setattr(Job, "dispatch", classmethod(dispatch))

    assert main(["--profile", "jan", "status"]) == 0

    assert [client.base_url for client in closed] == ["http://jan-host:9100"]


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
