"""Fetching a job's commit into a fresh worktree: what git may reach, and what it may run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast
from unittest.mock import Mock

import hashlib
import os
import signal
import subprocess
import threading
import time

import pytest

from knowop.shell import job_repo
from knowop.shell.conftest import _SetOnceHeld, _unscoped
from knowop.shell.job_repo import FetchError, Repos


if TYPE_CHECKING:
    from collections.abc import Sequence


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


def _capture_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    remote_config: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    spawned: list[tuple[list[str], dict[str, object]]] = []

    def popen(argv: list[str], **kwargs: object) -> _PipedChild:
        spawned.append((argv, kwargs))
        return _pipe_reply(argv, stdout=b"version\xff\n", stderr=b"notice\n")

    def require_tool(name: str) -> str:
        assert name == "git"
        return "/usr/bin/git"

    def kill_scope(argv: Sequence[str]) -> tuple[int, str]:
        assert argv[:4] == ["systemctl", "--user", "kill", "--signal=SIGKILL"]
        assert argv[4].startswith("knowop-prepare-")
        assert argv[4].endswith(".scope")
        return 0, ""

    monkeypatch.setattr(job_repo, "require_tool", require_tool)
    monkeypatch.setattr(job_repo, "scope_argv", _unscoped)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(job_repo, "run", kill_scope)
    if remote_config is not None:
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", remote_config)
    done = Repos(tmp_path / "repos")._run(None, "version")
    argv, kwargs = spawned[0]
    assert (cast(list[str], done.args), done.returncode, done.stdout, done.stderr) == (
        argv,
        0,
        "version\ufffd\n",
        "notice\n",
    )
    assert kwargs.keys() == {"stdin", "stdout", "stderr", "env", "start_new_session"}
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["stdout"] == subprocess.PIPE
    assert kwargs["stderr"] == subprocess.PIPE
    assert kwargs["start_new_session"] is True
    env = cast("dict[str, str]", kwargs["env"])
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["SSH_ASKPASS_REQUIRE"] == "never"
    return argv, env


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
    def __init__(self, stdout: int, stderr: int) -> None:
        self.stdout = os.fdopen(stdout, "rb", buffering=0)
        self.stderr = os.fdopen(stderr, "rb", buffering=0)
        self.args: list[str] = ["git", "version"]
        self.pid = 7
        self.returncode: int | None = 0

    def poll(self) -> int | None:
        return self.returncode

    def wait(self) -> int:
        return 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        del exc
        self.stdout.close()
        self.stderr.close()


def _pipe_reply(
    argv: list[str],
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
) -> _PipedChild:
    out, out_writer = os.pipe()
    err, err_writer = os.pipe()
    _ = os.write(out_writer, stdout)
    _ = os.write(err_writer, stderr)
    os.close(out_writer)
    os.close(err_writer)
    child = _PipedChild(out, err)
    child.args = argv
    return child


class _GitProcess(Mock):
    return_value: _PipedChild


@pytest.fixture
def git_process(monkeypatch: pytest.MonkeyPatch) -> _GitProcess:
    """Replace only Git's OS process and scope signals, keeping pipe supervision real."""
    started = _GitProcess(return_value=_pipe_reply(["git"]))
    monkeypatch.setattr(subprocess, "Popen", started)
    monkeypatch.setattr(job_repo, "require_tool", Mock(return_value="/usr/bin/git"))
    monkeypatch.setattr(job_repo, "run", Mock(return_value=(0, "")))
    monkeypatch.setattr(os, "killpg", Mock())
    return started


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
    git_process: _GitProcess,
) -> None:
    del git_process

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
    git_process: _GitProcess,
) -> None:
    child = git_process.return_value
    child.returncode = None
    signals = Mock()
    monkeypatch.setattr(os, "killpg", signals)
    repos = Repos(tmp_path / "repos", protocols=_FILE, timeout_sec=0)
    started = time.monotonic()

    with pytest.raises(FetchError, match="git init took more than 0 s"):
        repos.checkout(
            remote="file:///srv/repo.git",
            ref="refs/heads/main",
            commit="a" * 40,
            dest=tmp_path / "s",
        )

    assert time.monotonic() - started < 30
    signals.assert_called_once_with(child.pid, signal.SIGKILL)
    assert child.stdout.closed
    assert child.stderr.closed


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


def test_a_preparation_cancelled_while_git_runs_stops_it(
    tmp_path: Path,
    git_process: _GitProcess,
) -> None:
    del git_process
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


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
