"""The client's shipped modules import nothing a public install lacks.

The client ships alone, as the public ``knowop`` package; the project around it stays
private. A shipped module may import the standard library, the client itself, and
what a ``trackinizer`` install provides: trackinizer, the ``lib`` modules it carries,
and its dependencies. Any other import fails on a public install, or ships private
code.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import ast
import sys

import pytest


if TYPE_CHECKING:
    from collections.abc import Iterator


_CWD: Final = Path(__file__).resolve().parent

# The package this test lives in; the monorepo and the export name it differently.
_PACKAGE: Final = __name__.rpartition(".")[0]

_ALLOWED: Final = (
    _PACKAGE,
    "trackinizer",
    # The lib modules trackinizer's export carries: the session records the agent
    # runner writes, the pty plumbing the hosts use, and the per-user folders.
    "trackinizer.lib.agent",
    "trackinizer.lib.posix",
    "trackinizer.lib.userdirs",
    "treekle",
    # Dependencies of trackinizer, so installed with it.
    "httpx2",
    "pydantic",
    "websockets",
    "wrapt",
    # The direct terminal link's optional extra, imported only where it is installed.
    "aiortc",
    # Type checkers' own stubs, named under ``TYPE_CHECKING`` and never imported.
    "_typeshed",
)


def _shipped() -> list[Path]:
    """Return the modules the wheel ships: no tests, conftests or test support."""
    return sorted(
        path
        for path in _CWD.rglob("*.py")
        if not path.name.endswith("_test.py")
        and path.name != "conftest.py"
        and not {".export", "testing"} & set(path.relative_to(_CWD).parts)
    )


@pytest.mark.parametrize("path", _shipped(), ids=lambda path: path.name)
def test_a_shipped_module_imports_only_the_client_trackinizer_and_the_stdlib(
    path: Path,
) -> None:
    assert [name for name in _imports(path) if not _allowed(name)] == []


def test_the_scan_reads_the_cli_and_its_lazy_imports() -> None:
    assert _CWD / "cli.py" in _shipped()
    assert any(
        name.endswith("trackinizer.trax.login") for name in _imports(_CWD / "cli.py")
    )


def _allowed(name: str) -> bool:
    """Whether a shipped module may import ``name``."""
    return name.split(".", maxsplit=1)[0] in sys.stdlib_module_names or any(
        name == prefix or name.startswith(f"{prefix}.") for prefix in _ALLOWED
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
