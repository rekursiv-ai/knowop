"""Fetching a job's commit into a fresh worktree: what git may reach, and what it may run."""

from __future__ import annotations

from functools import partial
from http import server
from pathlib import Path
from typing import TYPE_CHECKING, cast, override

import os
import shlex
import ssl
import subprocess
import sys
import threading
import time

import pytest

from knowop.shell import job_repo
from knowop.shell.job_limits import run, scope_argv
from knowop.shell.job_repo import FetchError, Repos


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


_FILE = frozenset({"file"})


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


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(  # noqa: S603 -- fixed git argv on a tmp_path repo.
        ["git", "-C", str(repo), *args],  # noqa: S607 -- git is the tool under test.
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout.strip()


@pytest.fixture(autouse=True)
def _git_identity(tmp_path: Path) -> Iterator[None]:
    config = tmp_path / "gitconfig"
    _ = config.write_text("[user]\n\tname = t\n\temail = t@example.com\n")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("GIT_CONFIG_GLOBAL", str(config))
        patch.setenv("GIT_CONFIG_NOSYSTEM", "1")
        patch.delenv("GIT_DIR", raising=False)
        patch.setattr(job_repo, "scope_argv", _unscoped)
        yield


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    """Return a bare repository whose ``main`` is two commits deep, as pushed."""
    work = tmp_path / "work"
    work.mkdir()
    _ = _git(work, "init", "--quiet", "--initial-branch=main")
    _ = (work / "train.py").write_text("print(1)\n")
    _ = _git(work, "add", "-A")
    _ = _git(work, "commit", "--quiet", "-m", "one")
    _ = (work / "train.py").write_text("print(2)\n")
    _ = _git(work, "commit", "--quiet", "-am", "two")
    bare = tmp_path / "remote.git"
    _ = _git(tmp_path, "clone", "--quiet", "--bare", str(work), str(bare))
    return bare


def _url(remote: Path) -> str:
    return f"file://{remote}"


def test_a_job_gets_a_detached_worktree_of_its_commit(
    tmp_path: Path,
    remote: Path,
) -> None:
    repos = Repos(tmp_path / "repos", protocols=_FILE)
    first = _git(remote, "rev-parse", "main~1")

    repos.checkout(
        remote=_url(remote),
        ref="refs/heads/main",
        commit=first,
        dest=tmp_path / "jobs" / "7" / "src",
    )

    src = tmp_path / "jobs" / "7" / "src"
    assert (src / "train.py").read_text() == "print(1)\n"
    assert _git(src, "rev-parse", "HEAD") == first
    assert _git(src, "branch", "--show-current") == ""
    assert [p.suffix for p in (tmp_path / "repos").iterdir()] == [".git"]


def test_a_second_job_from_the_same_remote_shares_its_mirror(
    tmp_path: Path,
    remote: Path,
) -> None:
    repos = Repos(tmp_path / "repos", protocols=_FILE)
    tip = _git(remote, "rev-parse", "main")

    for name in ("7", "8"):
        repos.checkout(
            remote=_url(remote),
            ref="refs/heads/main",
            commit=tip,
            dest=tmp_path / name,
        )

    assert len(list((tmp_path / "repos").iterdir())) == 1
    assert (tmp_path / "8" / "train.py").read_text() == "print(2)\n"


def test_failed_preparation_preserves_active_worktrees_and_later_promotes_new_objects(
    tmp_path: Path,
    remote: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repos"
    repos = Repos(root, protocols=_FILE)
    first = _git(remote, "rev-parse", "main")
    repos.checkout(
        remote=_url(remote),
        ref="refs/heads/main",
        commit=first,
        dest=tmp_path / "first",
    )
    work = tmp_path / "work"
    (work / "payload").write_bytes(os.urandom(512 * 1024))
    _git(work, "add", "payload")
    _git(work, "commit", "--quiet", "-m", "payload")
    _git(work, "push", "--quiet", str(remote), "main")
    new = _git(remote, "rev-parse", "main")
    with monkeypatch.context() as patch:
        patch.setattr(job_repo, "MIRRORS_MAX_BYTES", 128 * 1024)
        with pytest.raises(FetchError, match="mirror storage"):
            repos.checkout(
                remote=_url(remote),
                ref="refs/heads/main",
                commit=new,
                dest=tmp_path / "second",
            )
    assert not (root / ".prepare").exists()
    assert sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) < 128 * 1024
    assert _git(tmp_path / "first", "rev-parse", "HEAD") == first
    repos.checkout(
        remote=_url(remote),
        ref="refs/heads/main",
        commit=new,
        dest=tmp_path / "second",
    )
    assert (tmp_path / "second" / "payload").read_bytes() == (
        work / "payload"
    ).read_bytes()
    assert _git(tmp_path / "first", "rev-parse", "HEAD") == first


def test_a_commit_that_is_not_on_the_remote_is_refused(
    tmp_path: Path,
    remote: Path,
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


def test_a_branch_that_is_not_on_the_remote_says_what_git_said(
    tmp_path: Path,
    remote: Path,
) -> None:
    repos = Repos(tmp_path / "repos", protocols=_FILE)

    with pytest.raises(FetchError, match="git fetch failed: fatal: couldn't find"):
        repos.checkout(
            remote=_url(remote),
            ref="refs/heads/gone",
            commit=_git(remote, "rev-parse", "main"),
            dest=tmp_path / "src",
        )


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


def test_git_itself_refuses_every_protocol_the_host_does_not_allow(
    tmp_path: Path,
    remote: Path,
) -> None:
    # Allowed by the parser, then refused by git: the parser and git agree on ``file``.
    repos = Repos(tmp_path / "repos", protocols=_FILE)
    tip = _git(remote, "rev-parse", "main")
    repos.checkout(
        remote=_url(remote),
        ref="refs/heads/main",
        commit=tip,
        dest=tmp_path / "a",
    )
    mirror = next((tmp_path / "repos").iterdir())

    refused = subprocess.run(  # noqa: S603 -- the host's own argv, replayed.
        [  # noqa: S607 -- git is the tool under test.
            "git",
            "-c",
            "protocol.allow=never",
            "-c",
            "protocol.https.allow=always",
            "--git-dir",
            str(mirror),
            "fetch",
            "--",
            _url(remote),
            "+refs/heads/main",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert refused.returncode != 0
    assert "transport 'file' not allowed" in refused.stderr


def test_no_hook_of_the_mirror_runs_on_checkout(tmp_path: Path, remote: Path) -> None:
    repos = Repos(tmp_path / "repos", protocols=_FILE)
    tip = _git(remote, "rev-parse", "main")
    repos.checkout(
        remote=_url(remote),
        ref="refs/heads/main",
        commit=tip,
        dest=tmp_path / "a",
    )
    mirror = next((tmp_path / "repos").iterdir())
    hook = mirror / "hooks" / "post-checkout"
    _ = hook.write_text(f"#!/bin/sh\ntouch {tmp_path / 'hooked'}\n")
    hook.chmod(0o755)

    repos.checkout(
        remote=_url(remote),
        ref="refs/heads/main",
        commit=tip,
        dest=tmp_path / "b",
    )

    assert (tmp_path / "b" / "train.py").exists()
    assert not (tmp_path / "hooked").exists()


@pytest.mark.parametrize("protocol", ["file", "ext"])
def test_hostile_global_transport_rewrites_cannot_fetch_local_code(
    tmp_path: Path,
    remote: Path,
    monkeypatch: pytest.MonkeyPatch,
    protocol: str,
) -> None:
    tip = _git(remote, "rev-parse", "main")
    target = _url(remote) if protocol == "file" else f"ext::git upload-pack {remote}"
    config = tmp_path / "hostile.gitconfig"
    config.write_text(
        f'[protocol "{protocol}"]\nallow = always\n'
        f'[url "{target}"]\ninsteadOf = https://127.0.0.1:1/repo\n',
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    repos = Repos(tmp_path / "repos", timeout_sec=5)

    with pytest.raises(FetchError):
        repos.checkout(
            remote="https://127.0.0.1:1/repo",
            ref="refs/heads/main",
            commit=tip,
            dest=tmp_path / "src",
        )

    assert not (tmp_path / "src").exists()


def test_removing_a_worktree_deletes_it_and_the_mirrors_record(
    tmp_path: Path,
    remote: Path,
) -> None:
    repos = Repos(tmp_path / "repos", protocols=_FILE)
    tip = _git(remote, "rev-parse", "main")
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
    listed = _git(tmp_path, "--git-dir", str(mirror), "worktree", "list", "--porcelain")
    assert str(tmp_path / "a") not in listed


@pytest.mark.parametrize("code", ["print('x' * 131072)", "x = bytearray(512 * 2**20)"])
def test_preparation_output_and_memory_are_bounded_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code: str,
) -> None:
    git = tmp_path / "git"
    git.write_text(
        f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -c {shlex.quote(code)}\n",
    )
    git.chmod(0o700)

    def tool(name: str) -> str:
        assert name == "git"
        return str(git)

    monkeypatch.setattr(job_repo, "require_tool", tool)
    if "bytearray" in code:
        monkeypatch.setattr(job_repo, "scope_argv", scope_argv)
    repos = Repos(tmp_path / "repos")

    with pytest.raises(FetchError):
        repos._git(None, "version")


def test_preparation_refuses_a_full_mirror_budget(
    tmp_path: Path,
    remote: Path,
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
            commit=_git(remote, "rev-parse", "main"),
            dest=tmp_path / "src",
        )


def test_storage_scan_tolerates_a_git_file_renamed_between_checks(
    tmp_path: Path,
    remote: Path,
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
        commit=_git(remote, "rev-parse", "main"),
        dest=tmp_path / "src",
    )
    assert (tmp_path / "src" / "train.py").read_text() == "print(2)\n"


class _AuthenticatedRepo(server.SimpleHTTPRequestHandler):
    """Serve a dumb Git repository only after the test helper supplies credentials."""

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


@pytest.mark.network_localhost
def test_https_preparation_uses_the_users_configured_credential_helper(
    tmp_path: Path,
    remote: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    certificate, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    code, _ = run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
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
    assert code == 0
    monkeypatch.setenv("GIT_SSL_CAINFO", str(certificate))
    _git(remote, "update-server-info")
    tip = _git(remote, "rev-parse", "main")
    helper = tmp_path / "helper"
    helper.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$1\" >> {shlex.quote(str(tmp_path / 'helper-called'))}\n"
        "if [ \"$1\" = get ]; then printf 'username=test\\npassword=token\\n'; fi\n",
    )
    helper.chmod(0o700)
    with (tmp_path / "gitconfig").open("a") as config:
        config.write(
            f"[credential]\nhelper = {helper}\n[http]\nsslCAInfo = {certificate}\n",
        )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    with server.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        partial(_AuthenticatedRepo, directory=str(tmp_path)),
    ) as httpd:
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
        worker = threading.Thread(target=httpd.serve_forever)
        worker.start()
        try:
            Repos(tmp_path / "repos").checkout(
                remote=f"https://127.0.0.1:{httpd.server_port}/remote.git",
                ref="refs/heads/main",
                commit=tip,
                dest=tmp_path / "src",
            )
        finally:
            httpd.shutdown()
            worker.join(timeout=5)
    assert "get" in (tmp_path / "helper-called").read_text().splitlines()
    assert (tmp_path / "src" / "train.py").read_text() == "print(2)\n"


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
    remote: Path,
) -> None:
    repos = Repos(tmp_path / "repos", protocols=_FILE)
    tree = _git(remote, "rev-parse", "main^{tree}")

    with pytest.raises(FetchError, match=f"{tree[:12]} is not on refs/heads/main"):
        repos.checkout(
            remote=_url(remote),
            ref="refs/heads/main",
            commit=tree,
            dest=tmp_path / "src",
        )
    assert not (tmp_path / "src").exists()


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


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
