"""The client's shipped modules import nothing a public install lacks.

The client ships alone, as the public ``knowop`` package; the project around it stays
private. A shipped module may import the standard library, the client itself, and
what a ``trackinizer`` install provides (trackinizer, the ``lib`` modules it carries,
and its dependencies), and the ``lib`` modules knowop's export vendors itself. The
agent process's folder, ``process/``, may also import all of sagent. Any other import
fails on a public install, or ships private code. Every export withholds a module
with ``private`` in its path, so none of these is allowed from any package.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import ast
import sys

import pytest


if TYPE_CHECKING:
    from collections.abc import Iterator


_CWD: Final[Path] = Path(__file__).resolve().parent

# The package this test lives in; the monorepo and the export name it differently.
_PACKAGE: Final = __name__.rpartition(".")[0]

_ALLOWED: Final = (
    _PACKAGE,
    "trackinizer",
    # The modules trackinizer's export carries: the session records a stored
    # session restores from, their readers, the pty plumbing the hosts use, and the
    # per-user folders.
    "trackinizer.sagent.session",
    "trackinizer.sagent.scripts.convert_session",
    # Sagent's tool server, from the sagent package knowop depends on: the tool host
    # serves sagent's remote tools through it (connect/tool_serving.py).
    "sagent.host",
    "trackinizer.lib.posix",
    "trackinizer.lib.userdirs",
    # Vendored by knowop's own export (copy.barista.toml), beside the package.
    "knowop.lib.files",
    "treekle",
    # Dependencies of trackinizer, so installed with it.
    "httpx2",
    "pydantic",
    "websockets",
    "wrapt",
    # Knowop's dependency for the direct terminal link.
    "aiortc",
    # Knowop's macOS dependency for the menu bar (pyobjc-framework-Cocoa).
    "AppKit",
    "PyObjCTools",
    # Type checkers' own stubs, named under ``TYPE_CHECKING`` and never imported.
    "_typeshed",
)

# The agent process runs sagent itself, in-process, so ``process/`` alone may import
# all of the sagent package knowop depends on. Anywhere else the export reads a sagent
# name as trackinizer's copy of the session records, which carries no runtime, so such
# an import would fail on a public install.
_PROCESS_ALLOWED: Final = ("sagent",)


def _shipped() -> list[Path]:
    """Return the modules the wheel ships: no tests, conftests or test support."""
    return sorted(
        path
        for path in _CWD.rglob("*.py")
        if not path.name.endswith(("_test.py", "_parity.py"))
        and path.name != "conftest.py"
        and not {".export", "testing"} & set(path.relative_to(_CWD).parts)
        and all("private" not in part for part in path.relative_to(_CWD).parts)
    )


@pytest.mark.parametrize(
    "path",
    [pytest.param(path, id=path.name) for path in _shipped()],
)
def test_a_shipped_module_imports_only_the_client_trackinizer_and_the_stdlib(
    path: Path,
) -> None:
    also = _PROCESS_ALLOWED if path.relative_to(_CWD).parts[0] == "process" else ()

    assert [name for name in _imports(path) if not _allowed(name, also=also)] == []


def test_all_of_sagent_is_allowed_in_the_agent_process_alone() -> None:
    runtime = f"{_PROCESS_ALLOWED[0]}.runtime.types"

    assert _CWD / "process" / "codec.py" in _shipped()
    assert runtime in set(_imports(_CWD / "process" / "codec.py"))
    assert not _allowed(runtime)
    assert _allowed(runtime, also=_PROCESS_ALLOWED)


def test_a_module_an_export_withholds_is_never_allowed() -> None:
    assert _allowed("trackinizer.trax.render")
    assert not _allowed("trackinizer.server.api.private.routers")
    assert not _allowed(f"{_PACKAGE}.private")


def test_the_scan_excludes_tests_parity_and_every_private_path_segment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "cli.py",
        "agent/private/check.py",
        "shell/private_checks/check.py",
        "private_check.py",
        "cli_test.py",
        "cli_parity.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    monkeypatch.setattr(f"{__name__}._CWD", tmp_path)

    assert _shipped() == [tmp_path / "cli.py"]


def test_the_scan_reads_the_cli_and_its_lazy_imports() -> None:
    assert _CWD / "cli.py" in _shipped()
    assert any(
        name.endswith("trackinizer.trax.login") for name in _imports(_CWD / "cli.py")
    )


def _allowed(name: str, *, also: tuple[str, ...] = ()) -> bool:
    """Whether a shipped module may import ``name``; ``also`` is what its folder adds."""
    # The exports leave out ``**/*private*``, so such a module is in no public install.
    if any("private" in part for part in name.split(".")):
        return False
    return name.split(".", maxsplit=1)[0] in sys.stdlib_module_names or any(
        name == prefix or name.startswith(f"{prefix}.") for prefix in (*_ALLOWED, *also)
    )


def _imports(path: Path) -> Iterator[str]:
    """Yield every module ``path`` imports, ``lazy_import`` targets included."""
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module is not None
        ):
            yield node.module
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "lazy_import"
            and node.args
            and isinstance(first := node.args[0], ast.Constant)
            and isinstance(first.value, str)
        ):
            yield first.value


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
