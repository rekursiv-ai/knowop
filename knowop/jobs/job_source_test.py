"""Exercise snapshot and bundle orchestration at the Git process boundary."""

from __future__ import annotations

from subprocess import CompletedProcess
from typing import TYPE_CHECKING
from unittest.mock import Mock, call

import hashlib
import os
import platform
import shutil
import subprocess

from trackinizer.client.errors import ClientError

import pytest

from knowop.jobs import job_source
from knowop.jobs.conftest import _done
from knowop.jobs.job_source import Git, Snapshot


if TYPE_CHECKING:
    from pathlib import Path


_HEAD = "a" * 40
_TREE = "b" * 40
_OLD_TREE = "c" * 40
_COMMIT = "d" * 40
_NO_SUCH_COMMIT = "f" * 40

# APFS refuses to create a file whose name is not UTF-8 (``EILSEQ``), so a name a
# Linux checkout can hold never reaches a snapshot on macOS.
_NAMES_MAY_BE_BYTES = pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="APFS refuses a file name that is not UTF-8",
)

_SECRET_NAMES = [
    ".env",
    ".env.local",
    ".envrc",
    "sub/.ENV",
    "id_rsa",
    "keys/id_ed25519",
    "ID_RSA",
    "server.pem",
    "a/b/tls.key",
    "TLS.KEY",
    "aws_credentials.json",
    "credentials",
    "gcp-Credentials.json",
    ".netrc",
]


@pytest.fixture
def git(monkeypatch: pytest.MonkeyPatch) -> Mock:
    run = Mock(return_value=_done())
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(shutil, "which", Mock(return_value="/tools/git"))
    monkeypatch.setattr(
        os,
        "environ",
        {
            "PATH": "/tools",
            "KEEP": "value",
            "GIT_AUTHOR_NAME": "caller author",
            "GIT_AUTHOR_EMAIL": "author@example.com",
            "GIT_COMMITTER_NAME": "caller committer",
            "GIT_COMMITTER_EMAIL": "committer@example.com",
        },
    )
    return run


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "index").write_bytes(b"user index\x00\xff")
    return repo


@pytest.fixture
def scratch(tmp_path: Path) -> Path:
    private = tmp_path / "private"
    (private / "objects" / "info").mkdir(parents=True)
    return private


@pytest.fixture
def replies(repo: Path, git: Mock) -> dict[str, CompletedProcess[bytes]]:
    answers = {
        "paths": _done(
            os.fsencode(f"{repo}\n{repo / '.git'}\n{repo / '.git' / 'index'}\nfalse\n"),
        ),
        "init": _done(),
        "untracked": _done(b"sub/c.txt\0new.py\0"),
        "add": _done(),
        "tree": _done(f" {_TREE}\n".encode()),
        "head": _done(f"{_HEAD}\n{_OLD_TREE}\n".encode()),
        "commit": _done(f" {_COMMIT}\n".encode()),
        "files": _done(b".gitignore\0a.txt\0new.py\0sub/b.txt\0sub/c.txt\0"),
    }
    git.side_effect = answers.values()
    return answers


@pytest.mark.usefixtures("replies")
def test_a_snapshot_holds_the_worktree_untracked_files_included_ignored_excluded(
    repo: Path,
    scratch: Path,
    git: Mock,
) -> None:
    snap = job_source.snapshot(repo, scratch)

    assert snap == Snapshot(
        commit=_COMMIT,
        private_git_dir=scratch,
        files=5,
        untracked=("new.py", "sub/c.txt"),
        head=_HEAD,
        shallow=False,
    )
    assert (scratch / "objects" / "info" / "alternates").read_text() == (
        f"{repo / '.git' / 'objects'}\n"
    )
    env = dict(os.environ)
    private_env = {
        **env,
        "GIT_AUTHOR_NAME": "knowop job",
        "GIT_AUTHOR_EMAIL": "knowop-job@localhost",
        "GIT_COMMITTER_NAME": "knowop job",
        "GIT_COMMITTER_EMAIL": "knowop-job@localhost",
        "GIT_INDEX_FILE": str(scratch / "index"),
        "GIT_OBJECT_DIRECTORY": str(scratch / "objects"),
    }
    options = {"input": b"", "capture_output": True, "check": False}
    assert git.call_args_list == [
        call(
            [
                "/tools/git",
                "rev-parse",
                "--path-format=absolute",
                "--show-toplevel",
                "--git-common-dir",
                "--git-path",
                "index",
                "--is-shallow-repository",
            ],
            cwd=repo,
            env=env,
            **options,
        ),
        call(
            ["/tools/git", "init", "--bare", str(scratch)],
            cwd=repo,
            env=env,
            **options,
        ),
        call(
            ["/tools/git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=repo,
            env=private_env,
            **options,
        ),
        call(
            ["/tools/git", "-c", "core.splitIndex=false", "add", "-A"],
            cwd=repo,
            env=private_env,
            **options,
        ),
        call(
            ["/tools/git", "-c", "core.splitIndex=false", "write-tree"],
            cwd=repo,
            env=private_env,
            **options,
        ),
        call(
            ["/tools/git", "rev-parse", "HEAD", "HEAD^{tree}"],
            cwd=repo,
            env=private_env,
            **options,
        ),
        call(
            [
                "/tools/git",
                "commit-tree",
                _TREE,
                "-p",
                _HEAD,
                "-m",
                f"knowop job snapshot of {_HEAD}",
            ],
            cwd=repo,
            env=private_env,
            **options,
        ),
        call(
            ["/tools/git", "ls-tree", "-r", "-z", _TREE],
            cwd=repo,
            env=private_env,
            **options,
        ),
    ]


@pytest.mark.usefixtures("replies")
def test_a_snapshot_leaves_the_users_repository_untouched(
    repo: Path,
    scratch: Path,
) -> None:
    before = (repo / ".git" / "index").read_bytes()
    snap = job_source.snapshot(repo, scratch)
    assert (repo / ".git" / "index").read_bytes() == before
    assert sorted(path.name for path in (repo / ".git").iterdir()) == ["index"]
    assert snap.private_git_dir != repo / ".git"
    assert snap.commit == _COMMIT


@pytest.mark.usefixtures("replies")
def test_a_split_index_gets_no_new_shared_index_in_the_users_repository(
    repo: Path,
    scratch: Path,
    git: Mock,
) -> None:
    (repo / ".git" / "sharedindex.abc").write_bytes(b"shared index")
    job_source.snapshot(repo, scratch)
    assert (repo / ".git" / "sharedindex.abc").read_bytes() == b"shared index"
    assert (scratch / "index").read_bytes() == b"user index\x00\xff"
    assert [each.args[0] for each in git.call_args_list[3:5]] == [
        ["/tools/git", "-c", "core.splitIndex=false", "add", "-A"],
        ["/tools/git", "-c", "core.splitIndex=false", "write-tree"],
    ]


def test_a_clean_tree_reuses_head_and_writes_no_commit(
    repo: Path,
    scratch: Path,
    git: Mock,
    replies: dict[str, CompletedProcess[bytes]],
) -> None:
    replies["head"].stdout = f"{_HEAD}\n{_TREE}\n".encode()
    replies["untracked"].stdout = b""
    replies["files"].stdout = b".gitignore\0a.txt\0sub/b.txt\0"
    git.side_effect = [value for name, value in replies.items() if name != "commit"]
    snap = job_source.snapshot(repo, scratch)
    assert snap == Snapshot(commit=_HEAD, private_git_dir=scratch, files=3, head=_HEAD)
    assert [each.args[0][1] for each in git.call_args_list] == [
        "rev-parse",
        "init",
        "ls-files",
        "-c",
        "-c",
        "rev-parse",
        "ls-tree",
    ]


@pytest.mark.parametrize("text", ["two\n", "two changed\n"], ids=["clean", "dirty"])
def test_a_shallow_clones_snapshot_is_a_commit_with_no_parent(
    repo: Path,
    scratch: Path,
    git: Mock,
    replies: dict[str, CompletedProcess[bytes]],
    text: str,
) -> None:
    replies["paths"].stdout = os.fsencode(
        f"{repo}\n{repo / '.git'}\n{repo / '.git' / 'index'}\ntrue\n",
    )
    head_tree = {"two\n": _TREE, "two changed\n": _OLD_TREE}[text]
    replies["head"].stdout = f"{_HEAD}\n{head_tree}\n".encode()
    snap = job_source.snapshot(repo, scratch)
    assert (snap.commit, snap.head, snap.shallow) == (_COMMIT, _HEAD, True)
    assert git.call_args_list[6].args[0] == [
        "/tools/git",
        "commit-tree",
        _TREE,
        "-m",
        f"knowop job snapshot of {_HEAD}",
    ]


def test_a_snapshot_works_from_a_subdirectory_and_a_linked_worktree(
    repo: Path,
    scratch: Path,
    git: Mock,
    replies: dict[str, CompletedProcess[bytes]],
    tmp_path: Path,
) -> None:
    linked = tmp_path / "linked"
    index = repo / ".git" / "worktrees" / "linked" / "index"
    index.parent.mkdir(parents=True)
    index.write_bytes(b"linked index")
    replies["paths"].stdout = os.fsencode(
        f"{linked}\n{repo / '.git'}\n{index}\nfalse\n",
    )
    replies["untracked"].stdout = b"linked.txt\0"
    snap = job_source.snapshot(linked / "sub", scratch)
    assert snap.untracked == ("linked.txt",)
    assert (scratch / "index").read_bytes() == b"linked index"
    assert (
        scratch / "objects" / "info" / "alternates"
    ).read_text() == f"{repo / '.git' / 'objects'}\n"
    assert [each.kwargs["cwd"] for each in git.call_args_list] == [
        linked / "sub",
    ] * 2 + [linked] * 6


def test_a_repository_with_no_commit_snapshots_as_a_root_commit(
    repo: Path,
    scratch: Path,
    git: Mock,
    replies: dict[str, CompletedProcess[bytes]],
) -> None:
    replies["paths"].stdout = os.fsencode(
        f"{repo}\n{repo / '.git'}\n{repo / '.git' / 'missing-index'}\nfalse\n",
    )
    replies["head"].returncode = 128
    replies["head"].stdout = b"HEAD\n"
    replies["untracked"].stdout = b"a.txt\0"
    replies["files"].stdout = b"a.txt\0"
    snap = job_source.snapshot(repo, scratch)
    assert snap == Snapshot(
        commit=_COMMIT,
        private_git_dir=scratch,
        files=1,
        untracked=("a.txt",),
    )
    assert not (scratch / "index").exists()
    assert git.call_args_list[6].args[0] == [
        "/tools/git",
        "commit-tree",
        _TREE,
        "-m",
        "knowop job snapshot of an empty repository",
    ]


def test_a_directory_outside_any_repository_is_refused(
    repo: Path,
    scratch: Path,
    git: Mock,
) -> None:
    git.return_value = _done(code=128, stderr=b"fatal: not a git repository\n")
    with pytest.raises(
        ClientError,
        match=r"^git rev-parse failed: fatal: not a git repository$",
    ):
        job_source.snapshot(repo, scratch)
    assert git.call_count == 1
    assert not (scratch / "objects" / "info" / "alternates").exists()


def test_the_worktree_top_is_found_from_a_subdirectory_and_absent_outside(
    repo: Path,
    git: Mock,
) -> None:
    git.side_effect = [
        _done(os.fsencode(f" {repo / '..' / 'repo'}\n")),
        _done(code=128, stderr=b"outside\n"),
    ]
    assert job_source.worktree_top(repo / "sub") == repo.resolve()
    assert job_source.worktree_top(repo.parent) is None
    assert [each.args[0] for each in git.call_args_list] == [
        ["/tools/git", "rev-parse", "--show-toplevel"],
    ] * 2
    assert [each.kwargs["cwd"] for each in git.call_args_list] == [
        repo / "sub",
        repo.parent,
    ]


@pytest.mark.usefixtures("replies")
def test_the_snapshot_commit_is_named_for_its_parent_and_authored_by_the_tool(
    repo: Path,
    scratch: Path,
    git: Mock,
) -> None:
    snap = job_source.snapshot(repo, scratch)
    commit = git.call_args_list[6]
    assert commit.args[0] == [
        "/tools/git",
        "commit-tree",
        _TREE,
        "-p",
        _HEAD,
        "-m",
        f"knowop job snapshot of {_HEAD}",
    ]
    assert commit.kwargs["env"] == {
        "PATH": "/tools",
        "KEEP": "value",
        "PYTEST_CURRENT_TEST": os.environ["PYTEST_CURRENT_TEST"],
        "GIT_AUTHOR_NAME": "knowop job",
        "GIT_AUTHOR_EMAIL": "knowop-job@localhost",
        "GIT_COMMITTER_NAME": "knowop job",
        "GIT_COMMITTER_EMAIL": "knowop-job@localhost",
        "GIT_INDEX_FILE": str(scratch / "index"),
        "GIT_OBJECT_DIRECTORY": str(scratch / "objects"),
    }
    assert snap.commit == _COMMIT


def test_the_first_snapshot_of_an_empty_repository_says_so(
    repo: Path,
    scratch: Path,
    git: Mock,
    replies: dict[str, CompletedProcess[bytes]],
) -> None:
    replies["head"].returncode = 128
    job_source.snapshot(repo, scratch)
    assert git.call_args_list[6].args[0] == [
        "/tools/git",
        "commit-tree",
        _TREE,
        "-m",
        "knowop job snapshot of an empty repository",
    ]


def test_a_checkout_whose_path_holds_quotes_colons_and_backslashes_snapshots(
    repo: Path,
    scratch: Path,
    git: Mock,
    replies: dict[str, CompletedProcess[bytes]],
    tmp_path: Path,
) -> None:
    odd = tmp_path / 'we"ird:dir\\x'
    replies["paths"].stdout = os.fsencode(
        f"{odd}\n{repo / '.git'}\n{repo / '.git' / 'index'}\nfalse\n",
    )
    snap = job_source.snapshot(odd, scratch)
    assert snap.head == _HEAD
    assert [each.kwargs["cwd"] for each in git.call_args_list] == [odd] * 8


@pytest.mark.usefixtures("replies")
def test_the_private_index_starts_as_a_copy_of_the_users_when_there_is_one(
    repo: Path,
    scratch: Path,
) -> None:
    job_source.snapshot(repo, scratch)
    assert (scratch / "index").read_bytes() == (repo / ".git" / "index").read_bytes()


@_NAMES_MAY_BE_BYTES
def test_a_filename_that_is_not_text_is_counted_and_kept(
    repo: Path,
    scratch: Path,
    replies: dict[str, CompletedProcess[bytes]],
) -> None:
    replies["files"].stdout += b"bad\xff.txt\0"
    assert job_source.snapshot(repo, scratch).files == 6


def test_a_snapshot_lists_the_untracked_files_it_added_and_not_the_ignored_ones(
    repo: Path,
    scratch: Path,
    replies: dict[str, CompletedProcess[bytes]],
) -> None:
    replies["untracked"].stdout = b"sub/deep/d.txt\0\0sub/c.txt\0new.py\0"
    assert job_source.snapshot(repo, scratch).untracked == (
        "new.py",
        "sub/c.txt",
        "sub/deep/d.txt",
    )


def test_a_snapshot_of_a_clean_tree_adds_no_untracked_file(
    repo: Path,
    scratch: Path,
    replies: dict[str, CompletedProcess[bytes]],
) -> None:
    replies["untracked"].stdout = b""
    assert job_source.snapshot(repo, scratch).untracked == ()


@pytest.mark.usefixtures("replies")
def test_the_untracked_files_are_listed_from_a_subdirectory_by_their_path_from_the_top(
    repo: Path,
    scratch: Path,
    git: Mock,
) -> None:
    snap = job_source.snapshot(repo / "sub", scratch)
    assert snap.untracked == ("new.py", "sub/c.txt")
    assert [each.kwargs["cwd"] for each in git.call_args_list] == [repo / "sub"] * 2 + [
        repo,
    ] * 6


@_NAMES_MAY_BE_BYTES
def test_an_untracked_file_whose_name_is_not_text_is_listed(
    repo: Path,
    scratch: Path,
    replies: dict[str, CompletedProcess[bytes]],
) -> None:
    replies["untracked"].stdout = b"bad\xff.txt\0"
    assert os.fsdecode(b"bad\xff.txt") in job_source.snapshot(repo, scratch).untracked


def test_an_untracked_file_whose_name_holds_a_newline_is_listed(
    repo: Path,
    scratch: Path,
    replies: dict[str, CompletedProcess[bytes]],
) -> None:
    replies["untracked"].stdout = b"two\nlines.txt\0"
    assert job_source.snapshot(repo, scratch).untracked == ("two\nlines.txt",)


@pytest.mark.parametrize("name", _SECRET_NAMES)
def test_an_untracked_file_that_looks_like_a_secret_is_refused_by_name(
    repo: Path,
    scratch: Path,
    git: Mock,
    replies: dict[str, CompletedProcess[bytes]],
    name: str,
) -> None:
    replies["untracked"].stdout = os.fsencode(f"new.py\0{name}\0")
    with pytest.raises(ClientError) as refused:
        job_source.snapshot(repo, scratch)
    assert str(refused.value) == (
        f"refusing to upload untracked files that look like secrets: {name}. "
        "Add them to .gitignore, or pass --include-untracked to send them."
    )
    assert git.call_count == 3


@pytest.mark.parametrize("name", _SECRET_NAMES)
def test_a_secret_named_file_goes_when_the_caller_says_to_include_it(
    repo: Path,
    scratch: Path,
    git: Mock,
    replies: dict[str, CompletedProcess[bytes]],
    name: str,
) -> None:
    replies["untracked"].stdout = os.fsencode(f"new.py\0{name}\0")
    snap = job_source.snapshot(repo, scratch, allow_secret_names=True)
    assert name in snap.untracked
    assert git.call_args_list[3].args[0] == [
        "/tools/git",
        "-c",
        "core.splitIndex=false",
        "add",
        "-A",
    ]


@pytest.mark.parametrize("name", _SECRET_NAMES)
def test_a_secret_named_file_that_gitignore_covers_is_neither_sent_nor_refused(
    repo: Path,
    scratch: Path,
    git: Mock,
    replies: dict[str, CompletedProcess[bytes]],
    name: str,
) -> None:
    replies["untracked"].stdout = b"new.py\0sub/c.txt\0"
    snap = job_source.snapshot(repo, scratch)
    assert name not in snap.untracked
    assert git.call_args_list[2].args[0] == [
        "/tools/git",
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    ]


def test_a_tracked_file_with_a_secret_name_is_the_users_choice_and_goes(
    repo: Path,
    scratch: Path,
    git: Mock,
    replies: dict[str, CompletedProcess[bytes]],
) -> None:
    replies["untracked"].stdout = b""
    replies["files"].stdout = b".env.example\0"
    snap = job_source.snapshot(repo, scratch)
    assert snap.untracked == ()
    assert snap.files == 1
    assert git.call_args_list[3].args[0] == [
        "/tools/git",
        "-c",
        "core.splitIndex=false",
        "add",
        "-A",
    ]


@pytest.mark.parametrize(
    "name",
    [
        "envelope.py",
        "keys.py",
        "identity.py",
        "monkey.txt",
        "id.txt",
        "a.key.md",
        "pem",
    ],
)
def test_an_untracked_file_that_only_resembles_a_secret_name_goes(
    repo: Path,
    scratch: Path,
    replies: dict[str, CompletedProcess[bytes]],
    name: str,
) -> None:
    replies["untracked"].stdout = os.fsencode(f"{name}\0")
    assert name in job_source.snapshot(repo, scratch).untracked


def test_the_refusal_names_every_secret_file_up_to_ten_and_counts_the_rest(
    repo: Path,
    scratch: Path,
    replies: dict[str, CompletedProcess[bytes]],
) -> None:
    replies["untracked"].stdout = "".join(
        f"k{number:02d}.pem\0" for number in range(13)
    ).encode()
    with pytest.raises(ClientError) as refused:
        job_source.snapshot(repo, scratch)
    names = ", ".join(f"k{number:02d}.pem" for number in range(10))
    assert str(refused.value) == (
        "refusing to upload untracked files that look like secrets: "
        f"{names} and 3 more. "
        "Add them to .gitignore, or pass --include-untracked to send them."
    )


def test_a_refusal_shows_a_name_with_a_control_character_as_an_escape(
    repo: Path,
    scratch: Path,
    replies: dict[str, CompletedProcess[bytes]],
) -> None:
    replies["untracked"].stdout = b"x\x1b[2J.pem\0"
    with pytest.raises(ClientError) as refused:
        job_source.snapshot(repo, scratch)
    assert "\x1b" not in str(refused.value)
    assert "x\\x1b[2J.pem" in str(refused.value)


def test_a_refused_snapshot_writes_nothing_to_the_users_repository(
    repo: Path,
    scratch: Path,
    git: Mock,
    replies: dict[str, CompletedProcess[bytes]],
) -> None:
    replies["untracked"].stdout = b".env\0"
    before = (repo / ".git" / "index").read_bytes()
    with pytest.raises(ClientError):
        job_source.snapshot(repo, scratch)
    assert (repo / ".git" / "index").read_bytes() == before
    assert sorted(path.name for path in (repo / ".git").iterdir()) == ["index"]
    assert git.call_count == 3


def test_plain_name_escapes_what_a_terminal_would_act_on() -> None:
    assert job_source.plain_name("a/b.txt") == "a/b.txt"
    assert job_source.plain_name("a\x1bb\nc") == "a\\x1bb\\nc"


def test_variables_that_pick_a_repository_do_not_redirect_git(
    repo: Path,
    scratch: Path,
    git: Mock,
    replies: dict[str, CompletedProcess[bytes]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_NAMESPACE",
        "GIT_PREFIX",
    ):
        monkeypatch.setenv(name, str(repo.parent / "nowhere"))
    out = repo.parent / "out.bundle"
    out.write_bytes(b"bundle\x00\xff")
    git.side_effect = [
        *replies.values(),
        _done(),
        _done(),
        _done(os.fsencode(f"{repo}\n")),
    ]
    snap = job_source.snapshot(repo, scratch)
    info = job_source.bundle(snap, [], out)
    assert info.bytes > 0
    assert job_source.worktree_top(repo) == repo.resolve()
    expected = {
        "PATH": "/tools",
        "KEEP": "value",
        "PYTEST_CURRENT_TEST": os.environ["PYTEST_CURRENT_TEST"],
        "GIT_AUTHOR_NAME": "caller author",
        "GIT_AUTHOR_EMAIL": "author@example.com",
        "GIT_COMMITTER_NAME": "caller committer",
        "GIT_COMMITTER_EMAIL": "committer@example.com",
    }
    assert [each.kwargs["env"] for each in git.call_args_list[:2]] == [expected] * 2
    assert [each.kwargs["env"] for each in git.call_args_list[8:]] == [expected] * 3
    assert git.call_args_list[3].kwargs["env"] == {
        **expected,
        "GIT_AUTHOR_NAME": "knowop job",
        "GIT_AUTHOR_EMAIL": "knowop-job@localhost",
        "GIT_COMMITTER_NAME": "knowop job",
        "GIT_COMMITTER_EMAIL": "knowop-job@localhost",
        "GIT_INDEX_FILE": str(scratch / "index"),
        "GIT_OBJECT_DIRECTORY": str(scratch / "objects"),
    }


def test_a_bundle_on_a_known_base_is_thin_and_verifies_where_the_base_is_a_ref(
    scratch: Path,
    tmp_path: Path,
    git: Mock,
) -> None:
    snap = Snapshot(commit=_COMMIT, private_git_dir=scratch, files=5)
    thin, full = tmp_path / "thin.bundle", tmp_path / "full.bundle"
    thin.write_bytes(b"thin")
    full.write_bytes(b"full history")
    git.side_effect = [
        _done(),
        _done(f"{_HEAD} commit 123\n".encode()),
        _done(),
        _done(),
        _done(),
    ]
    cut = job_source.bundle(snap, [_HEAD], thin)
    whole = job_source.bundle(snap, [], full)
    assert (cut.base, whole.base) == (_HEAD, None)
    assert cut.bytes < whole.bytes
    assert cut.path == thin
    options = {
        "cwd": scratch,
        "env": dict(os.environ),
        "capture_output": True,
        "check": False,
    }
    assert git.call_args_list == [
        call(
            ["/tools/git", "update-ref", f"refs/trax/gpu/{_COMMIT}", _COMMIT],
            input=b"",
            **options,
        ),
        call(
            ["/tools/git", "cat-file", "--batch-check"],
            input=f"{_HEAD}^{{commit}}\n".encode(),
            **options,
        ),
        call(
            [
                "/tools/git",
                "bundle",
                "create",
                str(thin),
                f"refs/trax/gpu/{_COMMIT}",
                f"^{_HEAD}",
            ],
            input=b"",
            **options,
        ),
        call(
            ["/tools/git", "update-ref", f"refs/trax/gpu/{_COMMIT}", _COMMIT],
            input=b"",
            **options,
        ),
        call(
            ["/tools/git", "bundle", "create", str(full), f"refs/trax/gpu/{_COMMIT}"],
            input=b"",
            **options,
        ),
    ]


def test_a_full_bundle_verifies_in_an_empty_repository(
    scratch: Path,
    tmp_path: Path,
    git: Mock,
) -> None:
    out = tmp_path / "full.bundle"
    out.write_bytes(b"complete history")
    info = job_source.bundle(
        Snapshot(commit=_COMMIT, private_git_dir=scratch, files=5),
        [],
        out,
    )
    assert info.base is None
    assert info.bytes == len(b"complete history")
    assert [each.args[0] for each in git.call_args_list] == [
        ["/tools/git", "update-ref", f"refs/trax/gpu/{_COMMIT}", _COMMIT],
        ["/tools/git", "bundle", "create", str(out), f"refs/trax/gpu/{_COMMIT}"],
    ]


def test_a_bundle_reports_the_size_and_digest_of_its_file(
    scratch: Path,
    tmp_path: Path,
    git: Mock,
) -> None:
    out = tmp_path / "out.bundle"
    data = b"bundle\x00\xff\n"
    out.write_bytes(data)
    info = job_source.bundle(
        Snapshot(commit=_COMMIT, private_git_dir=scratch, files=5),
        [],
        out,
    )
    assert info.path == out
    assert info.bytes == len(data)
    assert info.sha256 == hashlib.sha256(data).hexdigest()
    assert git.call_count == 2


def test_a_base_the_repository_lacks_or_that_is_not_a_sha_is_dropped(
    scratch: Path,
    tmp_path: Path,
    git: Mock,
) -> None:
    out = tmp_path / "out.bundle"
    out.write_bytes(b"bundle")
    git.side_effect = [
        _done(),
        _done(f"{_NO_SUCH_COMMIT}^{{commit}} missing\n".encode()),
        _done(),
    ]
    info = job_source.bundle(
        Snapshot(commit=_COMMIT, private_git_dir=scratch, files=5),
        [_NO_SUCH_COMMIT, "not-a-sha", "--all"],
        out,
    )
    assert info.base is None
    assert (
        git.call_args_list[1].kwargs["input"]
        == f"{_NO_SUCH_COMMIT}^{{commit}}\n".encode()
    )
    assert git.call_args_list[2].args[0] == [
        "/tools/git",
        "bundle",
        "create",
        str(out),
        f"refs/trax/gpu/{_COMMIT}",
    ]


def test_the_newest_known_base_is_the_one_a_bundle_is_cut_on(
    scratch: Path,
    tmp_path: Path,
    git: Mock,
) -> None:
    snap = Snapshot(commit=_COMMIT, private_git_dir=scratch, files=5)
    out = tmp_path / "out.bundle"
    out.write_bytes(b"bundle")
    first, second = _HEAD, "e" * 40
    git.side_effect = [
        _done(),
        _done(f"{second} commit 123\n{first} commit 123\n".encode()),
        _done(),
        _done(),
        _done(f"{first} commit 123\n{second} commit 123\n".encode()),
        _done(),
        _done(),
        _done(f"{_NO_SUCH_COMMIT}^{{commit}} missing\n{first} commit 123\n".encode()),
        _done(),
    ]
    newest = job_source.bundle(snap, [second, first], out)
    oldest = job_source.bundle(snap, [first, second], out)
    skipped = job_source.bundle(snap, [_NO_SUCH_COMMIT, first], out)
    assert (newest.base, oldest.base, skipped.base) == (second, first, first)
    assert [each.args[0] for each in git.call_args_list[2::3]] == [
        [
            "/tools/git",
            "bundle",
            "create",
            str(out),
            f"refs/trax/gpu/{_COMMIT}",
            f"^{second}",
            f"^{first}",
        ],
        [
            "/tools/git",
            "bundle",
            "create",
            str(out),
            f"refs/trax/gpu/{_COMMIT}",
            f"^{first}",
            f"^{second}",
        ],
        [
            "/tools/git",
            "bundle",
            "create",
            str(out),
            f"refs/trax/gpu/{_COMMIT}",
            f"^{first}",
        ],
    ]


def test_a_snapshot_that_is_itself_a_base_is_bundled_on_its_parent(
    scratch: Path,
    tmp_path: Path,
    git: Mock,
) -> None:
    out = tmp_path / "out.bundle"
    out.write_bytes(b"bundle")
    git.side_effect = [
        _done(),
        _done(f"{_COMMIT} commit 123\n".encode()),
        _done(f"{_HEAD}\n".encode()),
        _done(),
    ]
    info = job_source.bundle(
        Snapshot(commit=_COMMIT, private_git_dir=scratch, files=5),
        [_COMMIT],
        out,
    )
    assert info.base == _HEAD
    assert git.call_args_list[2].args[0] == ["/tools/git", "rev-parse", f"{_COMMIT}^@"]
    assert git.call_args_list[3].args[0] == [
        "/tools/git",
        "bundle",
        "create",
        str(out),
        f"refs/trax/gpu/{_COMMIT}",
        f"^{_HEAD}",
    ]


def test_a_base_that_descends_from_the_snapshot_falls_back_to_a_full_bundle(
    scratch: Path,
    tmp_path: Path,
    git: Mock,
) -> None:
    out = tmp_path / "out.bundle"
    out.write_bytes(b"bundle")
    git.side_effect = [
        _done(),
        _done(f"{_HEAD} commit 123\n".encode()),
        _done(code=128, stderr=b"Refusing to create empty bundle\n"),
        _done(),
    ]
    info = job_source.bundle(
        Snapshot(commit=_COMMIT, private_git_dir=scratch, files=5),
        [_HEAD],
        out,
    )
    assert info.base is None
    assert info.bytes > 0
    assert [each.args[0] for each in git.call_args_list[2:]] == [
        [
            "/tools/git",
            "bundle",
            "create",
            str(out),
            f"refs/trax/gpu/{_COMMIT}",
            f"^{_HEAD}",
        ],
        ["/tools/git", "bundle", "create", str(out), f"refs/trax/gpu/{_COMMIT}"],
    ]


def test_a_bundle_that_cannot_be_written_is_one_plain_error(
    scratch: Path,
    tmp_path: Path,
    git: Mock,
) -> None:
    out = tmp_path / "no-such-dir" / "out.bundle"
    git.side_effect = [
        _done(),
        _done(code=128, stderr=b"fatal: cannot create bundle\n"),
    ]
    with pytest.raises(
        ClientError,
        match=r"^git bundle failed: fatal: cannot create bundle$",
    ):
        job_source.bundle(
            Snapshot(commit=_COMMIT, private_git_dir=scratch, files=5),
            [],
            out,
        )
    assert git.call_count == 2


@pytest.mark.parametrize("held", ["nothing", "head", "the snapshot"])
def test_a_shallow_clones_snapshot_is_bundled_whole_and_unpacks_in_an_empty_repo(
    scratch: Path,
    tmp_path: Path,
    git: Mock,
    held: str,
) -> None:
    out = tmp_path / "out.bundle"
    out.write_bytes(b"whole shallow snapshot")
    bases = {"nothing": [], "head": [_HEAD], "the snapshot": [_COMMIT]}[held]
    info = job_source.bundle(
        Snapshot(commit=_COMMIT, private_git_dir=scratch, files=5, shallow=True),
        bases,
        out,
    )
    assert info.base is None
    assert [each.args[0] for each in git.call_args_list] == [
        ["/tools/git", "update-ref", f"refs/trax/gpu/{_COMMIT}", _COMMIT],
        ["/tools/git", "bundle", "create", str(out), f"refs/trax/gpu/{_COMMIT}"],
    ]


def test_git_preserves_argv_environment_input_and_nontext_output(
    repo: Path,
    git: Mock,
) -> None:
    git.return_value = _done(b"bad\xff.txt\n")
    run = Git(cwd=repo, env={"KEEP": "value"})
    assert run("cat-file", "--batch-check", stdin="bad\udcff.txt\n") == os.fsdecode(
        b"bad\xff.txt\n",
    )
    git.assert_called_once_with(
        ["/tools/git", "cat-file", "--batch-check"],
        cwd=repo,
        env={"KEEP": "value"},
        input=b"bad\xff.txt\n",
        capture_output=True,
        check=False,
    )


def test_missing_git_is_a_plain_tool_error(
    repo: Path,
    git: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", Mock(return_value=None))
    assert job_source.worktree_top(repo) is None
    with pytest.raises(ClientError, match="git was not found on PATH"):
        Git(cwd=repo, env={})("status")
    git.assert_not_called()


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
