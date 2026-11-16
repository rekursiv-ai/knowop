"""``knowop run``'s command line: knowop's flags, then sagent's words after ``--``."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING
from uuid import UUID

from trackinizer.wire.refs import SeqRef, UuidRef

import pytest

from knowop.process.command import (
    is_action,
    logs_in,
    parse_run,
    run_parser,
    sagent_command,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


_ID = "00000000-0000-0000-0000-000000000007"


def _refusal(argv: Sequence[str], capsys: pytest.CaptureFixture[str]) -> str:
    """Return what ``knowop run ARGV`` says as it refuses the command."""
    with pytest.raises(SystemExit) as refused:
        _ = parse_run(run_parser(), argv)
    assert refused.value.code == 2
    return capsys.readouterr().err.rpartition("knowop run: error: ")[2].strip()


def test_help_names_each_of_knowops_own_flags_with_what_it_takes_and_does(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLUMNS", "200")

    with pytest.raises(SystemExit) as exited:
        _ = parse_run(run_parser(), ["--help"])

    help_text = capsys.readouterr().out
    options = help_text[help_text.index("\noptions:\n") :].splitlines()[2:]
    assert exited.value.code == 0
    assert [" ".join(line.split()) for line in options] == [
        "-h, --help show this help message and exit",
        "--headless Run with no screen, through this computer's daemon, at a terminal too.",
        "--session ID The session to continue here: its UUID or number (AgentSession#N).",
        "--on [NAME] Start on host NAME through its daemon; with no name, the server's.",
        "--cwd DIR The folder to start in (default: ~ with --on, else this folder).",
        "--name NAME Open the chat's tab in Chat, named NAME (else it opens closed).",
        "--login-host NAME Use host NAME's model login.",
        "--stop ID Stop session ID's agent, wherever it runs, and exit.",
        "--time TIME End TIME after it starts, as a stop ends it; Slurm's --time: 120, 2:00:00 or 1-0.",
        "--serve URL Serve the session a daemon hands on stdin, through the server at URL.",
    ]


def test_on_with_no_name_names_the_servers_host_and_no_login_host() -> None:
    flags, _ = parse_run(run_parser(), ["--on"])

    assert (flags.on, flags.login_host) == ("", "")


@pytest.mark.parametrize(
    ("given", "ref"),
    [
        (_ID, UuidRef(uuid=UUID(_ID))),
        ("AgentSession#412", SeqRef(kind="AgentSession", seq=412)),
        ("agentsession#412", SeqRef(kind="AgentSession", seq=412)),
        ("412", SeqRef(kind="AgentSession", seq=412)),
    ],
)
def test_a_session_is_named_by_its_uuid_or_number(given: str, ref: object) -> None:
    for flag in ("--session", "--stop"):
        flags, _ = parse_run(run_parser(), [flag, given])
        assert (flags.session if flag == "--session" else flags.stop) == ref


@pytest.mark.parametrize("given", ["Issue#7", "knowop ui v2", "#412", ""])
def test_anything_else_is_no_session(
    given: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _refusal(["--session", given], capsys) == (
        f"argument --session: {given!r} is no session: give its UUID or number "
        "(AgentSession#N)"
    )


@pytest.mark.parametrize(
    ("argv", "sagent"),
    [
        ([], []),
        (["--headless", "--", "--effort", "high"], ["--effort", "high"]),
        (
            ["--", "--model", "sol", "--name", "a b"],
            ["--model", "sol", "--name", "a b"],
        ),
        (["--on", "lab", "--cwd", "~/w", "--"], []),
    ],
    ids=["bare", "a-setting", "settings-as-given", "nothing-after"],
)
def test_everything_after_the_first_separator_is_sagents_untouched(
    argv: list[str],
    sagent: list[str],
) -> None:
    flags, given = parse_run(run_parser(), argv)

    assert given == sagent
    assert flags.headless is ("--headless" in argv)


def test_knowops_own_flags_read_before_the_separator() -> None:
    flags, given = parse_run(
        run_parser(),
        ["--session", _ID, "--time", "90", "--on", "lab", "--login-host", "desk"],
    )
    named, _ = parse_run(run_parser(), ["--name", " paper review "])

    assert (flags.session, flags.time_limit) == (
        UuidRef(uuid=UUID(_ID)),
        timedelta(minutes=90),
    )
    # No --cwd: the run picks the folder, ~ on another host, this folder here.
    assert (flags.on, flags.login_host, flags.cwd, given) == ("lab", "desk", None, [])
    assert (flags.name, named.name) == (None, "paper review")


@pytest.mark.parametrize("given", ["", "   ", "x" * 121], ids=["empty", "blank", "121"])
def test_a_name_no_chat_could_have_is_refused(
    capsys: pytest.CaptureFixture[str],
    given: str,
) -> None:
    assert _refusal(["--name", given], capsys) == (
        "argument --name: A canvas name is 1 to 120 characters."
    )


def test_a_name_of_120_characters_is_a_chats() -> None:
    flags, _ = parse_run(run_parser(), ["--name", "x" * 120])

    assert flags.name == "x" * 120


@pytest.mark.parametrize(
    ("argv", "said"),
    [
        (
            ["--headless", "--effort", "high"],
            (
                "--effort is sagent's, so it goes after --: "
                "knowop run --headless -- --effort high"
            ),
        ),
        (
            ["--headless", "--effort", "high", "--profile", "alpha-staging.knowop.ai"],
            (
                "--effort is sagent's, so it goes after --; --profile is knowop's own, so "
                "it goes before run: knowop --profile alpha-staging.knowop.ai run "
                "--headless -- --effort high"
            ),
        ),
        (
            ["--profile=lab", "--effort", "high", "--on", "vm"],
            (
                "--effort is sagent's, so it goes after --; --profile is knowop's own, "
                "so it goes before run: knowop --profile=lab run --on vm -- --effort high"
            ),
        ),
        (
            ["--model", "sol", "--effort=max", "--", "--name", "a b"],
            (
                "--model, --effort are sagent's, so they go after --: "
                "knowop run -- --model sol --effort=max --name 'a b'"
            ),
        ),
        (
            ["login", "--provider", "oaisub"],
            (
                "login is sagent's, so it goes after --: "
                "knowop run -- login --provider oaisub"
            ),
        ),
        (
            ["--profile", "lab", "--effort", "high", "--headless"],
            (
                "--effort is sagent's, so it goes after --; --profile is knowop's own, so "
                "it goes before run: knowop --profile lab run --headless -- --effort high"
            ),
        ),
        (
            ["--headless", "--profile"],
            (
                "--profile is knowop's own, so it goes before run: "
                "knowop --profile run --headless"
            ),
        ),
        (
            ["--sess", _ID],
            f"--sess is sagent's, so it goes after --: knowop run -- --sess {_ID}",
        ),
    ],
    ids=[
        "dans",
        "dans-with-profile",
        "profile-alone",
        "two-and-more",
        "action",
        "profile-first-then-knowops",
        "profile-with-no-value",
        "abbrev",
    ],
)
def test_a_word_before_the_separator_that_is_not_knowops_is_refused_with_the_fix(
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    said: str,
) -> None:
    assert _refusal(argv, capsys) == said


@pytest.mark.parametrize(
    ("argv", "action"),
    [
        (["--", "login", "--provider", "oaisub"], True),
        (["--", "mcp", "list"], True),
        (["--", "--help"], True),
        (["--", "--effort", "high", "-h"], True),
        (["--", "--effort", "high"], False),
        (["--"], False),
        ([], False),
    ],
    ids=["login", "mcp", "help", "help-later", "a-setting", "nothing", "bare"],
)
def test_an_action_is_a_first_word_that_is_no_flag_or_a_help_after_the_separator(
    argv: list[str],
    action: bool,
) -> None:
    _, given = parse_run(run_parser(), argv)

    assert is_action(given) is action


@pytest.mark.parametrize(
    ("argv", "said"),
    [
        (
            ["--on", "lab", "--", "login", "--provider", "oaisub"],
            (
                "--on starts a session on a host, and sagent's login runs on this "
                "computer alone; run it on that host: knowop run -- login --provider oaisub"
            ),
        ),
        (
            ["--headless", "--time", "90", "--", "mcp", "list"],
            (
                "--headless, --time are for a session, and sagent's mcp opens none: "
                "knowop run -- mcp list"
            ),
        ),
        (
            ["--session", _ID, "--", "--help"],
            (
                "--session is for a session, and sagent's --help opens none: "
                "knowop run -- --help"
            ),
        ),
    ],
    ids=["on", "session-flags", "help"],
)
def test_an_action_takes_none_of_knowops_flags(
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    said: str,
) -> None:
    assert _refusal(argv, capsys) == said


@pytest.mark.parametrize(
    "picked",
    [
        ["--resume"],
        ["--continue"],
        ["--resume-all"],
        ["--continue-all"],
        ["--ephemeral"],
        ["--hosted-session", "x/session.jsonl"],
        ["--session", "x/session.jsonl"],
    ],
)
def test_a_sagent_flag_that_picks_a_journal_is_refused(
    capsys: pytest.CaptureFixture[str],
    picked: list[str],
) -> None:
    assert _refusal(["--", *picked], capsys) == (
        f"knowop keeps the session's journal; drop {picked[0]}"
    )


def test_a_time_limit_slurm_would_not_read_is_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _refusal(["--headless", "--time", "2h"], capsys) == (
        "argument --time: '2h' is no time limit, such as 120, 2:00:00 or 1-0"
    )


def test_sagents_own_parser_checks_its_half(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as refused:
        _ = parse_run(run_parser(), ["--", "--effort"])

    assert refused.value.code == 2
    assert "--effort needs a value" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("given", "named"),
    [
        (["--on", "vm"], "--on"),
        (["--headless"], "--headless"),
        (["--cwd", "~"], "--cwd"),
        (["--", "--model", "opus"], "--model"),
        (["--sto", "x"], "--sto"),
    ],
    ids=["on", "headless", "cwd-at-its-default", "sagents", "no-abbreviation"],
)
def test_stop_takes_no_other_word(
    capsys: pytest.CaptureFixture[str],
    given: list[str],
    named: str,
) -> None:
    assert _refusal(["--stop", _ID, *given], capsys) == (
        f"--stop takes no other flag: {named}"
    )


def test_a_daemons_start_names_its_server_and_time_before_the_separator() -> None:
    flags, given = parse_run(
        run_parser(),
        ["--serve", "https://box.example", "--time", "120", "--", "--model", "opus"],
    )

    assert (flags.serve, flags.time_limit) == (
        "https://box.example",
        timedelta(hours=2),
    )
    assert given == ["--model", "opus"]


@pytest.mark.parametrize(
    ("argv", "logs"),
    [
        ([], True),
        (["--headless", "--", "--effort", "high"], True),
        (["--on", "lab"], True),
        (["--stop", _ID], True),
        (["--serve", "https://box.example"], False),
        (["--", "login", "--provider", "oaisub"], False),
        (["--", "--help"], False),
    ],
    ids=["bare", "headless", "on", "stop", "serve", "login", "sagents-help"],
)
def test_only_a_command_that_reaches_the_server_with_a_profile_logs_in(
    argv: list[str],
    logs: bool,
) -> None:
    assert logs_in(argv) is logs


def test_a_refused_command_is_refused_before_it_logs_in(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as refused:
        _ = logs_in(["--headless", "--effort", "high"])

    assert refused.value.code == 2
    assert "knowop run --headless -- --effort high" in capsys.readouterr().err


def test_knowops_help_needs_no_login_and_is_knowop_runs_usage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert logs_in(["--headless", "--help", "--", "--effort", "high"]) is False
    with pytest.raises(SystemExit) as exited:
        _ = parse_run(run_parser(), ["--help"])

    assert exited.value.code == 0
    assert "knowop run [KNOWOP FLAGS] -- [SAGENT ARGUMENTS]" in capsys.readouterr().out


def test_sagents_command_runs_this_installs_sagent_with_the_words_given() -> None:
    command = sagent_command(["login", "--provider", "oaisub"])

    assert command[1:] == ["-P", "-m", "sagent", "login", "--provider", "oaisub"]


def test_the_parser_is_knowop_runs_and_takes_no_abbreviation() -> None:
    parser = run_parser()

    assert (parser.prog, parser.allow_abbrev) == ("knowop run", False)


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
