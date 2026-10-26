from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import argparse
import ast
import os
import subprocess
import sys
import tempfile
import uuid

from trackinizer.client.client import Client
from trackinizer.client.errors import ClientError
from trackinizer.trax import profile
from trackinizer.trax.profile import Profile
from treekle import json_dumps, json_loads

import pytest

from knowop.agent import cli, detach, session
from knowop.agent.custom_types import HostRequest
from knowop.agent.errors import (
    CiphertextDroppedError,
    LossyConversionError,
    NotResumableError,
)
from knowop.agent.materialize import Materialized


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class _Entered:
    """What the runner was entered with, recorded by a stand-in for ``session.main``."""

    def __init__(self) -> None:
        self.argv: list[str] = []
        self.kwargs: dict[str, object] = {}
        self.client: Client | None = None

    def __call__(
        self,
        argv: Sequence[str],
        *,
        client_factory: Callable[[], Client] | None = None,
        **kwargs: object,
    ) -> int:
        self.argv = list(argv)
        self.kwargs = kwargs
        # The factory resolves the active profile only when invoked, as the sync
        # path does; ``--no-sync`` never calls it.
        assert client_factory is not None
        self.client = client_factory()
        return 7


@pytest.fixture(autouse=True)
def no_terminal_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run as a shell that no trax terminal opened; a test that wants one sets it.

    A shell a terminal opened names its server, and with no profile for it every
    run first says how to add one, which the stderr tests here would read.
    """
    for name in ("TRAX_SERVER", "LC_TRAX_SERVER"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def entered(monkeypatch: pytest.MonkeyPatch) -> _Entered:
    """Stand in for the runner, so a test sees the arguments it would run with."""
    recorded = _Entered()
    monkeypatch.setattr(session, "main", recorded)
    profile.save_profile("rprof", Profile(url="http://rhost:9100", author="bob"))
    profile.switch_profile("rprof")
    return recorded


def test_rai_claude_runs_claude_with_its_arguments_and_the_runners_status(
    entered: _Entered,
) -> None:
    assert cli.main(["--as", "bob", "--", "hi"]) == 7

    assert entered.argv == ["--as", "bob", "--", "hi"]
    assert entered.kwargs["default_cli"] == "claude"


def test_rai_codex_runs_codex(entered: _Entered) -> None:
    assert cli.codex_main(["--detach", "--name", "run-1"]) == 7

    assert entered.argv == ["--detach", "--name", "run-1"]
    assert entered.kwargs["default_cli"] == "codex"


def test_detach_has_no_pickle_import() -> None:
    tree = ast.parse(Path(detach.__file__).read_text())
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert "pickle" not in imports


def test_the_spawn_payload_is_plain_json(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[HostRequest] = []
    monkeypatch.setattr(subprocess, "Popen", partial(_read_spawn_payload, requests))
    with pytest.raises(InterruptedError, match="payload checked"):
        _ = cli.main(
            [
                "--profile",
                "elsewhere",
                "--host",
                "host.test",
                "--port",
                "9100",
                "--no-sync",
                "--detach",
                "--name",
                "json-host",
                "sh",
                "--room",
                "one",
                "--room",
                "two",
                "--",
                "echo",
                "héllo",
            ],
        )
    [request] = requests
    assert request["command"] == "sh echo 'héllo'"
    assert request["resume"] == ""
    assert request["run"]["connection"] == {
        "profile": "elsewhere",
        "host": "host.test",
        "port": 9100,
    }
    assert request["run"]["cli_argv"] == ["echo", "héllo"]
    assert request["run"]["flags"] == {
        "cli": "sh",
        "out": None,
        "verbose": False,
        "dry_run": False,
        "sync": False,
        "actor": None,
        "rooms": ["one", "two"],
        "private": False,
        "diff": False,
        "detach": True,
        "name": "json-host",
        "attach": None,
    }


def _read_spawn_payload(
    requests: list[HostRequest],
    argv: list[str],
    **kwargs: object,
) -> None:
    del kwargs
    with os.fdopen(os.dup(int(argv[-1])), "r") as source:
        text = source.read()
        # A jsonpickle tag is a key, so match its opening quote: bare ``py/``
        # also matches a state path under a tmp dir named like ``...hurkttpy/``.
        assert '"py/' not in text
        requests.append(json_loads(text, HostRequest))
    raise InterruptedError("payload checked")


def test_the_host_entry_reads_and_closes_its_private_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[HostRequest] = []
    monkeypatch.setattr(subprocess, "Popen", partial(_read_spawn_payload, requests))
    with pytest.raises(InterruptedError, match="payload checked"):
        _ = cli.main(["--no-sync", "--detach", "--name", "json-host", "sh"])
    [request] = requests
    read_fd, write_fd = os.pipe()
    request["ready"] = write_fd
    redirected: list[Path] = []
    monkeypatch.setattr(detach, "_redirect_stdio", redirected.append)
    ran: list[session.RunConfig] = []
    monkeypatch.setattr(session, "run", partial(_record_host_run, ran))
    with tempfile.TemporaryFile() as payload:
        _ = payload.write(json_dumps(request).encode())
        _ = payload.seek(0)
        inherited = os.dup(payload.fileno())
        try:
            assert cli.main(["--_host", str(inherited)]) == 7
            assert os.read(read_fd, 2) == b"ok"
        finally:
            os.close(read_fd)
        with pytest.raises(OSError, match="Bad file descriptor"):
            _ = os.fstat(inherited)
    [config] = ran
    assert config.cli_name == "sh"
    assert config.run_name == "json-host"
    assert config.client is None
    assert isinstance(config.host, detach.HeadlessSpec)
    assert str(config.host.address) == request["address"]
    assert str(config.host.control) == request["control"]
    assert config.host.resume == request["resume"]
    record = detach._read_record(Path(request["paths"]))
    assert record is not None
    assert (record.state, record.exit_code) == ("exited", 7)
    assert redirected == [Path(request["paths"]) / "host.log"]


def _record_host_run(ran: list[session.RunConfig], config: session.RunConfig) -> int:
    ran.append(config)
    assert config.host is not None
    assert config.host.on_listening is not None
    config.host.on_listening()
    return 7


@pytest.mark.parametrize("text", ["{broken", "{}", '{"paths": 12}'])
def test_an_invalid_host_payload_still_closes_its_descriptor(text: str) -> None:
    with tempfile.TemporaryFile() as payload:
        _ = payload.write(text.encode())
        _ = payload.seek(0)
        inherited = os.dup(payload.fileno())
        with pytest.raises((ValueError, TypeError)):
            _ = cli.main(["--_host", str(inherited)])
        with pytest.raises(OSError, match="Bad file descriptor"):
            _ = os.fstat(inherited)


def test_the_client_comes_from_the_active_profile(entered: _Entered) -> None:
    _ = cli.main(["claude"])

    resolved = entered.client
    assert isinstance(resolved, Client)
    try:
        assert resolved.base_url == "http://rhost:9100"
        assert resolved.author == "bob"
    finally:
        resolved.close()


def test_a_leading_profile_flag_picks_the_server_and_is_not_a_runner_argument(
    entered: _Entered,
) -> None:
    profile.save_profile("other", Profile(url="http://ohost:9200", author="amy"))

    _ = cli.main(["--profile", "other", "--no-sync"])

    assert entered.argv == ["--no-sync"]
    assert entered.kwargs["profile"] == "other"
    resolved = entered.client
    assert isinstance(resolved, Client)
    try:
        assert resolved.base_url == "http://ohost:9200"
    finally:
        resolved.close()


def test_a_flag_the_runner_owns_is_not_taken_for_a_connection_flag(
    entered: _Entered,
) -> None:
    _ = cli.main(["--no-sync", "--", "--profile", "x"])

    assert entered.argv == ["--no-sync", "--", "--profile", "x"]
    assert entered.kwargs["profile"] is None


def test_connection_flags_keep_equals_in_values_and_stop_at_the_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = Mock(
        return_value=argparse.ArgumentParser(prog="knowop run", add_help=False),
    )
    monkeypatch.setattr(argparse, "ArgumentParser", parser)
    tail = ["--no-sync", "--", "--profile=child"]

    flags, rest = cli._peel_connection(
        ["--profile=other=two", "--host", "example.test", "--port=9100", *tail],
    )

    assert flags == argparse.Namespace(
        profile="other=two",
        host="example.test",
        port=9100,
    )
    assert rest == tail
    parser.assert_called_once_with(prog="knowop run", add_help=False)


def test_a_host_command_reaches_the_host_commands_and_names_the_program_run() -> None:
    """The real runner answers ``attach`` for a host that is not there, as knowop run."""
    with pytest.raises(SystemExit) as stopped:
        _ = cli.codex_main(["attach", "ghost"])

    assert str(stopped.value) == (
        "knowop run: no host named 'ghost'; see knowop run ls --all"
    )


def _ran(argv: Sequence[str], **kwargs: object) -> int:
    """Stand in for the runner, which ran and ended well."""
    del argv, kwargs
    return 0


def _import_root() -> Path:
    """Return the directory ``loop`` imports from, so a child process finds this tree."""
    return Path(cli.__file__).resolve().parents[len(cli.__name__.split(".")) - 1]


def _child_env(tmp_path: Path, *, path: str) -> dict[str, str]:
    """Return an environment whose runs and profiles live in ``tmp_path``."""
    return {
        "PATH": path,
        "PYTHONPATH": str(_import_root()),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "HOME": str(tmp_path),
    }


@pytest.mark.cli_python_subprocess
def test_the_module_lists_its_hosts_as_a_process(tmp_path: Path) -> None:
    done = subprocess.run(  # noqa: S603 -- the current interpreter running this package.
        [sys.executable, "-m", cli.__package__ or "", "ls"],
        env=_child_env(tmp_path, path=os.defpath),
        capture_output=True,
        text=True,
        check=False,
    )

    assert (done.returncode, done.stdout) == (0, "NAME  STATE  PID  STARTED  COMMAND\n")


@pytest.mark.cli_python_subprocess
def test_knowop_run_reaches_the_runner_as_a_process_with_no_login(
    tmp_path: Path,
) -> None:
    """``knowop run ls`` asks no server, so it needs no login, and prints the hosts."""
    knowop = (cli.__package__ or "").rpartition(".")[0]

    done = subprocess.run(  # noqa: S603 -- the current interpreter running knowop.
        [sys.executable, "-m", knowop, "run", "ls"],
        env=_child_env(tmp_path, path=os.defpath),
        capture_output=True,
        text=True,
        check=False,
    )

    assert (done.returncode, done.stdout) == (0, "NAME  STATE  PID  STARTED  COMMAND\n")


def test_a_client_error_is_one_line_and_status_2(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def refuse(argv: Sequence[str], **kwargs: object) -> int:
        del argv, kwargs
        raise ClientError("server unreachable")

    monkeypatch.setattr(session, "main", refuse)

    assert cli.main([]) == 2
    assert capsys.readouterr().err == "knowop run claude: server unreachable\n"


def test_the_codex_program_names_itself_in_an_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def refuse(argv: Sequence[str], **kwargs: object) -> int:
        del argv, kwargs
        raise ClientError("server unreachable")

    monkeypatch.setattr(session, "main", refuse)

    assert cli.codex_main([]) == 2
    assert capsys.readouterr().err == "knowop run codex: server unreachable\n"


def test_a_shell_from_a_server_with_no_saved_profile_is_told_how_to_add_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A trax terminal names its server; with no profile for it, the run says so."""
    monkeypatch.setenv("LC_TRAX_SERVER", "https://elsewhere.example")
    monkeypatch.setattr(session, "main", _ran)

    assert cli.main([]) == 0
    assert capsys.readouterr().err.startswith(
        "trax: this shell came from https://elsewhere.example, which has no saved "
        "profile",
    )


class _Resumed:
    """The resume tail's collaborators: a loader that records, and the runner."""

    def __init__(self, tmp_path: Path, *, prefix_bytes: int = 0) -> None:
        self.loaded: list[tuple[uuid.UUID, str, bool]] = []
        self.entered: list[tuple[list[str], dict[str, object]]] = []
        self.written = Materialized(
            path=tmp_path / "session.jsonl",
            cli_session_id=uuid.uuid4(),
            attach=uuid.uuid4(),
            prefix_bytes=prefix_bytes,
        )

    def prepare(
        self,
        client: object,
        session_id: uuid.UUID,
        target: str,
        *,
        lossy: bool = False,
    ) -> Materialized:
        del client
        self.loaded.append((session_id, target, lossy))
        return self.written

    def enter(self, argv: Sequence[str], **kwargs: object) -> int:
        self.entered.append((list(argv), kwargs))
        return 0


@pytest.fixture
def resumed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Resumed:
    """Stand in for the conversation loader and the runner of a resume."""
    stand_in = _Resumed(tmp_path, prefix_bytes=4_096)
    monkeypatch.setattr(cli, "prepare_resume", stand_in.prepare)
    monkeypatch.setattr(session, "main", stand_in.enter)
    profile.save_profile("rprof", Profile(url="http://rhost:9100", author="bob"))
    profile.switch_profile("rprof")
    return stand_in


_SESSION = uuid.UUID("9f1c2e3a-0000-4000-8000-000000000001")


@pytest.mark.parametrize(
    "tail",
    [
        ["--as", "bob", "--", "--model", "opus"],
        ["--detach", "--name", "job"],
        ["--", "--lossy"],
    ],
)
def test_a_resume_preserves_the_runner_tail(
    tail: list[str],
    resumed: _Resumed,
) -> None:
    assert cli.main(["resume", str(_SESSION), *tail]) == 0

    [(argv, _kwargs)] = resumed.entered
    assert argv == tail


def test_a_resume_asks_for_its_own_cli_and_hands_the_runner_the_loaded_file(
    resumed: _Resumed,
) -> None:
    assert cli.codex_main(["resume", str(_SESSION)]) == 0

    assert resumed.loaded == [(_SESSION, "codex", False)]
    [(argv, kwargs)] = resumed.entered
    assert argv == []
    assert kwargs["attach"] == resumed.written.attach
    assert kwargs["prefix_bytes"] == 4_096
    assert kwargs["resume_path"] == resumed.written.path
    assert kwargs["cli_session_id"] == str(resumed.written.cli_session_id)
    assert kwargs["default_cli"] == "codex"


def test_the_resume_takes_its_own_lossy_flag_and_not_the_clis(
    resumed: _Resumed,
) -> None:
    _ = cli.main(["resume", str(_SESSION), "--lossy", "--as", "bob", "--", "--lossy"])

    assert resumed.loaded == [(_SESSION, "claude", True)]
    [(argv, _kwargs)] = resumed.entered
    assert argv == ["--as", "bob", "--", "--lossy"]


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (NotResumableError("no native part"), "no native part"),
        (LossyConversionError("drops 3 records"), "drops 3 records"),
        (
            CiphertextDroppedError("sealed"),
            "sealed (its records remain searchable; only the replay is lost)",
        ),
    ],
    ids=["not-resumable", "lossy", "ciphertext"],
)
def test_a_refused_resume_is_one_line_and_never_enters_the_runner(
    error: Exception,
    message: str,
    resumed: _Resumed,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def refuse(*args: object, **kwargs: object) -> Materialized:
        del args, kwargs
        raise error

    monkeypatch.setattr(cli, "prepare_resume", refuse)

    assert cli.main(["resume", str(_SESSION)]) == 2

    assert capsys.readouterr().err == f"knowop run claude: {message}\n"
    assert resumed.entered == []


def test_a_resume_names_what_it_loads(
    resumed: _Resumed,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ = cli.main(["resume", str(_SESSION)])

    assert capsys.readouterr().out == (
        f"resuming {_SESSION} as claude from {resumed.written.path}\n"
    )


def test_the_resume_flows_through_the_profile_it_is_given(
    resumed: _Resumed,
) -> None:
    profile.save_profile("other", Profile(url="http://ohost:9200", author="amy"))

    _ = cli.main(["--profile", "other", "resume", str(_SESSION)])

    [(_argv, kwargs)] = resumed.entered
    assert kwargs["profile"] == "other"
    assert kwargs["client_factory"] is not None


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
