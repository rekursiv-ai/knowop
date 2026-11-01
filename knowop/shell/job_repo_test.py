"""Fetching a job's commit into a fresh worktree: what git may reach, and what it may run."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from http import server
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast, override

import hashlib
import os
import selectors
import shlex
import shutil
import ssl
import subprocess
import threading
import time

import pytest

from knowop.shell import job_repo
from knowop.shell.job_limits import run
from knowop.shell.job_repo import FetchError, Repos


if TYPE_CHECKING:
    from collections.abc import Sequence
    from selectors import SelectorKey


_FILE = frozenset({"file"})


@dataclass(kw_only=True, slots=True)
class _Remote:
    url: str
    refs: dict[str, str] = field(default_factory=dict)
    commits: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    trees: dict[str, dict[str, bytes]] = field(default_factory=dict)

    def commit(self, files: dict[str, bytes], *, ref: str = "refs/heads/main") -> str:
        tree = hashlib.sha1(
            repr(sorted(files.items())).encode(),
            usedforsecurity=False,
        ).hexdigest()
        parent = self.refs.get(ref)
        commit = hashlib.sha1(
            f"{tree} {parent} {len(self.commits)}".encode(),
            usedforsecurity=False,
        ).hexdigest()
        self.trees[tree] = dict(files)
        self.commits[commit] = (tree, parent)
        self.refs[ref] = commit
        return commit

    def history(self, commit: str) -> list[str]:
        found: list[str] = []
        while commit:
            found.append(commit)
            parent = self.commits[commit][1]
            if parent is None:
                break
            commit = parent
        return found


class _FakeGit:
    def __init__(self, remote: _Remote) -> None:
        self.remote = remote
        self.calls: list[tuple[str, ...]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def run(
            repos: Repos,
            mirror: Path | None,
            *args: str,
        ) -> subprocess.CompletedProcess[str]:
            return self.run(repos, mirror, *args)

        monkeypatch.setattr(Repos, "_run", run)

    def run(
        self,
        repos: Repos,
        mirror: Path | None,
        *args: str,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        match args:
            case ["init", "--bare", "--quiet", path] if mirror is None:
                root = Path(path)
                (root / "objects").mkdir(parents=True)
                _ = (root / "HEAD").write_text("ref: refs/heads/main\n")
                return _completed(args, repos=repos)
            case ["fetch", "--no-tags", "--quiet", "--", url, want] if mirror:
                if url != self.remote.url:
                    return _completed(
                        args,
                        repos=repos,
                        stderr="fatal: not our ref\n",
                        returncode=128,
                    )
                if want.startswith("+"):
                    ref = want[1:]
                    commit = self.remote.refs.get(ref)
                    if commit is None:
                        return _completed(
                            args,
                            repos=repos,
                            stderr=f"fatal: couldn't find remote ref {ref}\n",
                            returncode=128,
                        )
                    commits = self.remote.history(commit)
                elif want in self.remote.trees:
                    _store(
                        mirror,
                        want,
                        "tree",
                        b"".join(self.remote.trees[want].values()),
                    )
                    return _completed(args, repos=repos)
                elif want in self.remote.commits:
                    commits = self.remote.history(want)
                else:
                    return _completed(
                        args,
                        repos=repos,
                        stderr="fatal: not our ref\n",
                        returncode=128,
                    )
                for commit in commits:
                    tree, _parent = self.remote.commits[commit]
                    files = self.remote.trees[tree]
                    _store(mirror, commit, "commit", tree.encode())
                    _store(mirror, tree, "tree", b"".join(files.values()))
                return _completed(args, repos=repos)
            case ["cat-file", "-e", spec] if mirror and spec.endswith("^{commit}"):
                object_id = spec.removesuffix("^{commit}")
                is_commit = _stored(mirror, object_id) == "commit"
                return _completed(args, repos=repos, returncode=0 if is_commit else 128)
            case ["worktree", "add", "--detach", "--quiet", dest, commit] if mirror:
                if _stored(mirror, commit) != "commit":
                    return _completed(
                        args,
                        repos=repos,
                        stderr="fatal: invalid reference\n",
                        returncode=128,
                    )
                tree, _parent = self.remote.commits[commit]
                worktree = Path(dest)
                worktree.mkdir(parents=True)
                for name, content in self.remote.trees[tree].items():
                    _ = (worktree / name).write_bytes(content)
                record = mirror / "worktrees" / worktree.name
                record.mkdir(parents=True)
                _ = (record / "gitdir").write_text(f"{worktree / '.git'}\n")
                _ = (record / "HEAD").write_text(f"{commit}\n")
                _ = (worktree / ".git").write_text(f"gitdir: {record}\n")
                return _completed(args, repos=repos)
            case ["worktree", "prune"] if mirror:
                for record in (mirror / "worktrees").glob("*"):
                    if not Path((record / "gitdir").read_text().strip()).exists():
                        for child in record.iterdir():
                            child.unlink()
                        record.rmdir()
                return _completed(args, repos=repos)
            case _:
                raise AssertionError(f"git {args} is not modeled")

    def head(self, worktree: Path) -> str:
        gitdir = Path((worktree / ".git").read_text().removeprefix("gitdir: ").strip())
        return (gitdir / "HEAD").read_text().strip()

    def worktrees(self, mirror: Path) -> list[Path]:
        return sorted(
            Path((record / "gitdir").read_text().strip()).parent
            for record in (mirror / "worktrees").glob("*")
        )


def _store(git_dir: Path, object_id: str, kind: str, body: bytes) -> None:
    path = git_dir / "objects" / object_id[:2] / object_id[2:]
    path.parent.mkdir(exist_ok=True)
    _ = path.write_bytes(kind.encode() + b"\0" + body)


def _stored(git_dir: Path, object_id: str) -> str | None:
    path = git_dir / "objects" / object_id[:2] / object_id[2:]
    return path.read_bytes().split(b"\0")[0].decode() if path.is_file() else None


def _completed(
    args: list[str] | tuple[str, ...],
    *,
    repos: Repos | None = None,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    if repos is not None:
        repos._check_storage()
    return subprocess.CompletedProcess(args, returncode, stdout, stderr)


class _Exited:
    def __init__(self, argv: list[str]) -> None:
        self.args = argv
        self.pid = 1

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        del exc

    def poll(self) -> int:
        return 0


def _capture_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    remote_config: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    spawned: list[tuple[list[str], dict[str, object]]] = []

    def popen(argv: list[str], **kwargs: object) -> _Exited:
        spawned.append((argv, kwargs))
        return _Exited(argv)

    def require_tool(name: str) -> str:
        assert name == "git"
        return "/usr/bin/git"

    def collect(
        self: Repos,
        child: subprocess.Popen[bytes],
    ) -> subprocess.CompletedProcess[str]:
        del self
        return _completed(cast("list[str]", child.args))

    def kill_scope(argv: Sequence[str]) -> tuple[int, str]:
        assert argv[0] == "systemctl"
        return 0, ""

    monkeypatch.setattr(job_repo, "require_tool", require_tool)
    monkeypatch.setattr(job_repo, "scope_argv", _unscoped)
    monkeypatch.setattr(job_repo.subprocess, "Popen", popen)
    monkeypatch.setattr(Repos, "_collect", collect)
    monkeypatch.setattr(job_repo, "run", kill_scope)
    if remote_config is not None:
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", remote_config)
    Repos(tmp_path / "repos")._run(None, "version")
    argv, kwargs = spawned[0]
    return argv, cast("dict[str, str]", kwargs["env"])


def _unscoped(
    *,
    unit: str,
    memory_mb: int,
    argv: Sequence[str],
    runtime_sec: float | None = None,
) -> list[str]:
    assert unit.startswith("knowop-prepare-")
    assert memory_mb == 256
    assert runtime_sec is not None
    return list(argv)


@pytest.fixture(autouse=True)
def unscoped_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(job_repo, "scope_argv", _unscoped)


@pytest.fixture
def remote() -> _Remote:
    remote = _Remote(url="file:///srv/remote.git")
    _ = remote.commit({"train.py": b"print(1)\n"})
    _ = remote.commit({"train.py": b"print(2)\n"})
    return remote


@pytest.fixture
def fake_git(remote: _Remote, monkeypatch: pytest.MonkeyPatch) -> _FakeGit:
    fake = _FakeGit(remote)
    fake.install(monkeypatch)
    return fake


def _url(remote: _Remote) -> str:
    return remote.url


def test_a_job_gets_a_detached_worktree_of_its_commit(
    tmp_path: Path,
    remote: _Remote,
    fake_git: _FakeGit,
) -> None:
    repos = Repos(tmp_path / "repos", protocols=_FILE)
    first = remote.history(remote.refs["refs/heads/main"])[1]

    repos.checkout(
        remote=_url(remote),
        ref="refs/heads/main",
        commit=first,
        dest=tmp_path / "jobs" / "7" / "src",
    )

    src = tmp_path / "jobs" / "7" / "src"
    assert (src / "train.py").read_text() == "print(1)\n"
    assert fake_git.head(src) == first
    assert [p.suffix for p in (tmp_path / "repos").iterdir()] == [".git"]


def test_a_second_job_from_the_same_remote_shares_its_mirror(
    tmp_path: Path,
    remote: _Remote,
    fake_git: _FakeGit,
) -> None:
    repos = Repos(tmp_path / "repos", protocols=_FILE)
    tip = remote.refs["refs/heads/main"]

    for name in ("7", "8"):
        repos.checkout(
            remote=_url(remote),
            ref="refs/heads/main",
            commit=tip,
            dest=tmp_path / name,
        )

    assert len(list((tmp_path / "repos").iterdir())) == 1
    assert (tmp_path / "8" / "train.py").read_text() == "print(2)\n"
    assert (
        fake_git.calls.count(
            ("worktree", "add", "--detach", "--quiet", str(tmp_path / "7"), tip),
        )
        == 1
    )


def test_failed_preparation_preserves_active_worktrees_and_later_promotes_new_objects(
    tmp_path: Path,
    remote: _Remote,
    fake_git: _FakeGit,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repos"
    repos = Repos(root, protocols=_FILE)
    first = remote.refs["refs/heads/main"]
    repos.checkout(
        remote=_url(remote),
        ref="refs/heads/main",
        commit=first,
        dest=tmp_path / "first",
    )
    payload = os.urandom(32 * 1024)
    payload_file = tmp_path / "payload"
    _ = payload_file.write_bytes(payload)
    new = remote.commit({"train.py": b"print(2)\n", "payload": payload})
    retained_bytes = sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    byte_budget = retained_bytes + 16 * 1024
    with monkeypatch.context() as patch:
        patch.setattr(job_repo, "MIRRORS_MAX_BYTES", byte_budget)
        with pytest.raises(FetchError, match="mirror storage"):
            repos.checkout(
                remote=_url(remote),
                ref="refs/heads/main",
                commit=new,
                dest=tmp_path / "second",
            )
    assert not (root / ".prepare").exists()
    assert sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) == (
        retained_bytes
    )
    assert fake_git.head(tmp_path / "first") == first
    repos.checkout(
        remote=_url(remote),
        ref="refs/heads/main",
        commit=new,
        dest=tmp_path / "second",
    )
    assert (tmp_path / "second" / "payload").read_bytes() == payload_file.read_bytes()
    assert fake_git.head(tmp_path / "first") == first


def test_a_commit_that_is_not_on_the_remote_is_refused(
    tmp_path: Path,
    remote: _Remote,
    fake_git: _FakeGit,
) -> None:
    repos = Repos(tmp_path / "repos", protocols=_FILE)

    with pytest.raises(FetchError, match=r"git fetch failed|is not on"):
        repos.checkout(
            remote=_url(remote),
            ref="refs/heads/main",
            commit="f" * 40,
            dest=tmp_path / "src",
        )
    assert not (tmp_path / "src").exists()
    assert fake_git.calls[-1][0] == "fetch"


def test_a_branch_that_is_not_on_the_remote_says_what_git_said(
    tmp_path: Path,
    remote: _Remote,
    fake_git: _FakeGit,
) -> None:
    repos = Repos(tmp_path / "repos", protocols=_FILE)

    with pytest.raises(FetchError, match="git fetch failed: fatal: couldn't find"):
        repos.checkout(
            remote=_url(remote),
            ref="refs/heads/gone",
            commit=remote.refs["refs/heads/main"],
            dest=tmp_path / "src",
        )
    assert fake_git.calls[-1][0] == "fetch"


@pytest.mark.parametrize(
    ("url", "protocols"),
    [
        ("file:///srv/repo.git", frozenset({"https", "ssh"})),
        ("https://github.com/o/r.git", _FILE),
        ("ext::sh -c touch% /tmp/knowop-ext", frozenset({"https", "ssh", "file"})),
        ("/srv/repo.git", frozenset({"https", "ssh", "file"})),
        ("-u/tmp/x", frozenset({"https", "ssh", "file"})),
    ],
)
def test_a_remote_this_host_does_not_fetch_over_is_refused_before_git_runs(
    tmp_path: Path,
    url: str,
    protocols: frozenset[str],
) -> None:
    repos = Repos(tmp_path / "repos", protocols=protocols)

    with pytest.raises(FetchError, match="this machine fetches only over"):
        repos.checkout(
            remote=url,
            ref="refs/heads/main",
            commit="a" * 40,
            dest=tmp_path / "s",
        )
    assert not (tmp_path / "repos").exists()


def test_the_default_host_fetches_over_https_and_ssh_only(tmp_path: Path) -> None:
    with pytest.raises(FetchError, match="fetches only over https or ssh"):
        Repos(tmp_path).checkout(
            remote="file:///srv/repo.git",
            ref="refs/heads/main",
            commit="a" * 40,
            dest=tmp_path / "s",
        )


def test_git_is_configured_to_refuse_every_non_host_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, _kwargs = _capture_run(monkeypatch, tmp_path)

    assert argv[argv.index("-c") + 1] == "protocol.allow=never"
    assert "protocol.file.allow=always" not in argv


def test_git_runs_no_mirror_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, _kwargs = _capture_run(monkeypatch, tmp_path)

    assert argv[argv.index("-c") + 1] == "protocol.allow=never"
    assert "core.hooksPath=/dev/null" in argv


@pytest.mark.parametrize("protocol", ["file", "ext"])
def test_hostile_transport_configuration_cannot_expand_git_protocols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protocol: str,
) -> None:
    config = tmp_path / "hostile.gitconfig"
    config.write_text(f'[protocol "{protocol}"]\nallow = always\n')
    monkeypatch.setenv("GIT_DIR", "/elsewhere/.git")
    argv, env = _capture_run(monkeypatch, tmp_path, remote_config=str(config))

    assert "protocol.allow=never" in argv
    assert f"protocol.{protocol}.allow=always" not in argv
    assert env["GIT_ALLOW_PROTOCOL"] == "https:ssh"
    assert env["GIT_CONFIG_GLOBAL"] == str(config)
    assert "GIT_DIR" not in env


def test_removing_a_worktree_deletes_it_and_the_mirrors_record(
    tmp_path: Path,
    remote: _Remote,
    fake_git: _FakeGit,
) -> None:
    repos = Repos(tmp_path / "repos", protocols=_FILE)
    tip = remote.refs["refs/heads/main"]
    repos.checkout(
        remote=_url(remote),
        ref="refs/heads/main",
        commit=tip,
        dest=tmp_path / "a",
    )
    mirror = next((tmp_path / "repos").iterdir())

    repos.remove(remote=_url(remote), dest=tmp_path / "a")
    repos.remove(remote="https://github.com/never/fetched.git", dest=tmp_path / "x")

    assert not (tmp_path / "a").exists()
    assert fake_git.worktrees(mirror) == []


def test_git_uses_the_memory_and_time_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, _env = _capture_run(monkeypatch, tmp_path)

    assert argv == [
        "/usr/bin/git",
        "-c",
        "protocol.allow=never",
        "-c",
        "protocol.https.allow=always",
        "-c",
        "protocol.ssh.allow=always",
        "-c",
        "core.hooksPath=/dev/null",
        "version",
    ]


class _PipedChild:
    args = ("git", "version")

    def __init__(self, stdout: int, stderr: int) -> None:
        self.stdout = os.fdopen(stdout, "rb", buffering=0)
        self.stderr = os.fdopen(stderr, "rb", buffering=0)

    def poll(self) -> int:
        return 0

    def wait(self) -> int:
        return 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        del exc
        self.stdout.close()
        self.stderr.close()


def test_preparation_output_is_bounded_before_launch() -> None:
    stdout, writer = os.pipe()
    stderr, stderr_writer = os.pipe()
    os.close(stderr_writer)
    child = _PipedChild(stdout, stderr)
    payload = b"x" * (job_repo.PREPARATION_OUTPUT_BYTES + 1)

    def write_output() -> None:
        try:
            view = memoryview(payload)
            while view:
                view = view[os.write(writer, view) :]
        except BrokenPipeError:
            pass
        finally:
            os.close(writer)

    writer_thread = threading.Thread(target=write_output)
    writer_thread.start()
    with child, pytest.raises(FetchError, match="output exceeds its byte budget"):
        Repos(Path("unused"))._collect(cast("subprocess.Popen[bytes]", child))
    writer_thread.join()


def test_preparation_refuses_a_full_mirror_budget(
    tmp_path: Path,
    remote: _Remote,
    fake_git: _FakeGit,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repos"
    root.mkdir()
    (root / "retained").write_bytes(b"x" * 65_536)
    monkeypatch.setattr(job_repo, "MIRRORS_MAX_BYTES", 32_768, raising=False)
    repos = Repos(root, protocols=_FILE)

    with pytest.raises(FetchError, match="mirror storage"):
        repos.checkout(
            remote=_url(remote),
            ref="refs/heads/main",
            commit=remote.refs["refs/heads/main"],
            dest=tmp_path / "src",
        )
    assert not fake_git.calls


def test_storage_scan_tolerates_a_git_file_renamed_between_checks(
    tmp_path: Path,
    remote: _Remote,
    fake_git: _FakeGit,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repos"
    root.mkdir()
    temporary = root / "temporary.pack"
    temporary.write_bytes(b"pack")
    original = Path.stat
    calls: list[Path] = []

    def changing(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if path == temporary:
            calls.append(path)
            if len(calls) == 2:
                path.rename(root / "finished.pack")
        return original(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", changing)
    Repos(root, protocols=_FILE).checkout(
        remote=_url(remote),
        ref="refs/heads/main",
        commit=remote.refs["refs/heads/main"],
        dest=tmp_path / "src",
    )
    assert len(calls) == 2
    assert fake_git.calls
    assert (tmp_path / "src" / "train.py").read_text() == "print(2)\n"


def test_git_uses_the_users_configured_credential_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "gitconfig"
    _ = config.write_text("[credential]\nhelper = /tmp/credential-helper\n")
    _argv, env = _capture_run(monkeypatch, tmp_path, remote_config=str(config))

    assert env["GIT_CONFIG_GLOBAL"] == str(config)
    assert env["GIT_ALLOW_PROTOCOL"] == "https:ssh"


def test_git_that_runs_out_of_time_is_said_plainly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def too_slow(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise subprocess.TimeoutExpired(cmd="git", timeout=5)

    monkeypatch.setattr(Repos, "_collect", too_slow)
    repos = Repos(tmp_path / "repos", protocols=_FILE, timeout_sec=5)

    with pytest.raises(FetchError, match="git init took more than 5 s"):
        repos.checkout(
            remote="file:///srv/repo.git",
            ref="refs/heads/main",
            commit="a" * 40,
            dest=tmp_path / "s",
        )


def test_git_still_running_at_its_deadline_is_killed_and_said_plainly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A git that is still running when the deadline passes, on every platform: a real
    # ``git init`` may already have exited by the time the scope's kill returns.
    git = tmp_path / "git"
    _ = git.write_text("#!/bin/sh\nexec sleep 60\n")
    git.chmod(0o700)

    def tool(name: str) -> str:
        assert name == "git"
        return str(git)

    monkeypatch.setattr(job_repo, "require_tool", tool)
    repos = Repos(tmp_path / "repos", protocols=_FILE, timeout_sec=0)
    started = time.monotonic()

    with pytest.raises(FetchError, match="git init took more than 0 s"):
        repos.checkout(
            remote="file:///srv/repo.git",
            ref="refs/heads/main",
            commit="a" * 40,
            dest=tmp_path / "s",
        )

    # Leaving ``Popen`` waits for the child, so only the kill ends it this soon.
    assert time.monotonic() - started < 30


def test_an_object_the_fetches_bring_that_is_not_a_commit_is_refused(
    tmp_path: Path,
    remote: _Remote,
    fake_git: _FakeGit,
) -> None:
    repos = Repos(tmp_path / "repos", protocols=_FILE)
    tree, _parent = remote.commits[remote.refs["refs/heads/main"]]

    with pytest.raises(FetchError, match=f"{tree[:12]} is not on refs/heads/main"):
        repos.checkout(
            remote=_url(remote),
            ref="refs/heads/main",
            commit=tree,
            dest=tmp_path / "src",
        )
    assert not (tmp_path / "src").exists()
    assert fake_git.calls[-1] == ("cat-file", "-e", f"{tree}^{{commit}}")


def test_a_preparation_cancelled_before_it_holds_the_mirrors_runs_no_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_git(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("git ran after the cancel")

    monkeypatch.setattr(Repos, "_run", no_git)
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(FetchError, match="preparation cancelled"):
        Repos(tmp_path / "repos", protocols=_FILE).checkout(
            remote="file:///srv/repo.git",
            ref="refs/heads/main",
            commit="a" * 40,
            dest=tmp_path / "s",
            cancel=cancel,
        )


class _SetOnceHeld(threading.Event):
    """A cancel that is clear when the mirrors' lock is taken, and set from then on."""

    def __init__(self) -> None:
        super().__init__()
        self.asked = 0

    @override
    def is_set(self) -> bool:
        self.asked += 1
        return self.asked > 1


def test_a_preparation_cancelled_while_git_runs_stops_it(tmp_path: Path) -> None:
    cancel = _SetOnceHeld()

    with pytest.raises(FetchError, match="preparation cancelled"):
        Repos(tmp_path / "repos", protocols=_FILE).checkout(
            remote="file:///srv/repo.git",
            ref="refs/heads/main",
            commit="a" * 40,
            dest=tmp_path / "s",
            cancel=cancel,
        )
    assert cancel.asked == 2
    assert not (tmp_path / "repos" / ".prepare").exists()


class _Pipeless:
    """A child started without output pipes."""

    args = ("git",)
    stdout: None = None
    stderr: None = None


def test_a_git_child_without_output_pipes_is_refused(tmp_path: Path) -> None:
    repos = Repos(tmp_path / "repos", protocols=_FILE)

    with pytest.raises(ValueError, match="preparation output pipes"):
        _ = repos._collect(cast("subprocess.Popen[bytes]", _Pipeless()))


def _real_git(root: Path, *args: str) -> str:
    code, stdout = run(["git", "-C", str(root), *args])
    assert code == 0, (args, stdout)
    return stdout.strip()


@pytest.fixture(scope="module")
def real_remote_seed(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str, str]:
    root = tmp_path_factory.mktemp("git-remote-seed")
    work = root / "work"
    bare = root / "remote.git"
    work.mkdir()
    _real_git(work, "init", "--quiet", "--initial-branch=main")
    _real_git(work, "config", "user.name", "Test")
    _real_git(work, "config", "user.email", "test@example.com")
    _ = (work / "train.py").write_text("print(2)\n")
    _real_git(work, "add", "train.py")
    _real_git(work, "commit", "--quiet", "-m", "initial")
    commit = _real_git(work, "rev-parse", "HEAD")
    _real_git(root, "clone", "--quiet", "--bare", str(work), str(bare))
    return bare, commit, _real_git(work, "rev-parse", "HEAD^{tree}")


@pytest.fixture
def real_remote(
    real_remote_seed: tuple[Path, str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, str, str]:
    templates = tmp_path / "templates"
    templates.mkdir()
    monkeypatch.setenv("GIT_TEMPLATE_DIR", str(templates))
    seed, commit, tree = real_remote_seed
    bare = tmp_path / "remote.git"
    shutil.copytree(seed, bare)
    return bare, commit, tree


def _ignore_scope_kill(argv: Sequence[str]) -> tuple[int, str]:
    assert argv[:2] == ["systemctl", "--user"]
    return 0, ""


class _PromptSelector(selectors.DefaultSelector):
    """Keep real pipe readiness while shortening the supervision polling interval."""

    @override
    def select(self, timeout: float | None = None) -> list[tuple[SelectorKey, int]]:
        return super().select(None if timeout is None else min(timeout, 0.001))


def _real_repos(monkeypatch: pytest.MonkeyPatch, root: Path) -> Repos:
    monkeypatch.setattr(job_repo, "scope_argv", _unscoped)
    monkeypatch.setattr(job_repo.selectors, "DefaultSelector", _PromptSelector)
    monkeypatch.setattr(job_repo, "run", _ignore_scope_kill)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    config = root / "global.gitconfig"
    config.write_text("[user]\nname = Test\nemail = test@example.com\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    return Repos(root / "repos", protocols=_FILE)


@pytest.mark.cli_git
def test_real_git_fetches_the_requested_commit_into_a_bare_mirror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_remote: tuple[Path, str, str],
) -> None:
    bare, commit, _tree = real_remote
    repos = _real_repos(monkeypatch, tmp_path)
    stage = tmp_path / "stage.git"

    repos._prepare(stage, remote=f"file://{bare}", ref="refs/heads/main", commit=commit)

    assert _real_git(stage, "rev-parse", "--is-bare-repository") == "true"
    assert _real_git(stage, "cat-file", "-t", commit) == "commit"


@pytest.mark.cli_git
def test_real_git_adds_a_detached_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_remote: tuple[Path, str, str],
) -> None:
    bare, commit, _tree = real_remote
    dest = tmp_path / "job" / "src"

    _real_repos(monkeypatch, tmp_path).checkout(
        remote=f"file://{bare}",
        ref="refs/heads/main",
        commit=commit,
        dest=dest,
    )

    assert (dest / "train.py").read_text() == "print(2)\n"
    assert _real_git(dest, "rev-parse", "HEAD") == commit
    assert _real_git(dest, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"
    assert len(list((tmp_path / "repos").glob("*.git"))) == 1


@pytest.mark.cli_git
def test_real_git_rejects_a_tree_as_the_requested_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_remote: tuple[Path, str, str],
) -> None:
    bare, _commit, tree = real_remote

    with pytest.raises(FetchError, match=f"{tree[:12]} is not on refs/heads/main"):
        _real_repos(monkeypatch, tmp_path).checkout(
            remote=f"file://{bare}",
            ref="refs/heads/main",
            commit=tree,
            dest=tmp_path / "src",
        )
    assert not (tmp_path / "src").exists()


@pytest.mark.cli_git
def test_real_git_prunes_a_removed_worktree_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_remote: tuple[Path, str, str],
) -> None:
    bare, commit, _tree = real_remote
    repos = _real_repos(monkeypatch, tmp_path)
    dest = tmp_path / "src"
    repos.checkout(
        remote=f"file://{bare}",
        ref="refs/heads/main",
        commit=commit,
        dest=dest,
    )
    mirror = next((tmp_path / "repos").glob("*.git"))
    assert _real_git(mirror, "worktree", "list", "--porcelain").count("worktree ") == 2

    repos.remove(remote=f"file://{bare}", dest=dest)

    assert not dest.exists()
    assert _real_git(mirror, "worktree", "list", "--porcelain").count("worktree ") == 1


class _AuthenticatedRepo(server.SimpleHTTPRequestHandler):
    @override
    def do_GET(self) -> None:
        if self.headers.get("Authorization") != "Basic dGVzdDp0b2tlbg==":
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="test"')
            self.end_headers()
            return
        super().do_GET()

    @override
    def log_message(self, format: str, *args: object) -> None:
        del format, args


@pytest.mark.cli_git
@pytest.mark.network_localhost
def test_real_git_uses_the_credential_helper_over_local_https(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_remote: tuple[Path, str, str],
) -> None:
    bare, commit, _tree = real_remote
    _real_git(bare, "update-server-info")
    certificate, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    code, output = run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "ec",
            "-pkeyopt",
            "ec_paramgen_curve:P-256",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(certificate),
            "-days",
            "1",
            "-subj",
            "/CN=127.0.0.1",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
        ],
    )
    assert code == 0, output
    helper = tmp_path / "credential-helper"
    helper_called = tmp_path / "helper-called"
    helper.write_text(
        "#!/bin/sh\n"
        f'echo "$1" >> {shlex.quote(str(helper_called))}\n'
        'if [ "$1" = get ]; then echo username=test; echo password=token; fi\n',
    )
    helper.chmod(0o700)
    ca_directory = tmp_path / "ca"
    ca_directory.mkdir()
    config = tmp_path / "global.gitconfig"
    config.write_text(
        "[user]\nname = Test\nemail = test@example.com\n"
        f"[credential]\nhelper = {helper}\n"
        f"[http]\nsslCAInfo = {certificate}\nsslCAPath = {ca_directory}\n",
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setattr(job_repo, "run", _ignore_scope_kill)
    monkeypatch.setattr(job_repo.selectors, "DefaultSelector", _PromptSelector)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    with server.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        partial(_AuthenticatedRepo, directory=str(bare.parent)),
    ) as httpd:
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
        worker = threading.Thread(
            target=partial(httpd.serve_forever, poll_interval=0.005),
        )
        worker.start()
        try:
            dest = tmp_path / "src"
            Repos(tmp_path / "repos").checkout(
                remote=f"https://127.0.0.1:{httpd.server_port}/{bare.name}",
                ref="refs/heads/main",
                commit=commit,
                dest=dest,
            )
        finally:
            httpd.shutdown()
            worker.join(timeout=5)

    assert "get" in helper_called.read_text().splitlines()
    assert (tmp_path / "src" / "train.py").read_text() == "print(2)\n"


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
