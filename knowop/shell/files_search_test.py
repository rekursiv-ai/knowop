"""Folder search, through ripgrep where it is installed and through Python without it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import shutil

from treekle import json_dumps

import pytest

from knowop.lib.files.grep import Query, Row
from knowop.shell import files_search
from knowop.shell.files_search import Search, SearchError


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(params=["rg", "python"])
def engine(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Search with ripgrep, or with it hidden from ``PATH``."""
    name = str(request.param)
    if name == "rg" and shutil.which("rg") is None:
        pytest.skip("ripgrep is not installed")
    if name == "python":
        monkeypatch.setattr(shutil, "which", _nowhere)
    return name


def _nowhere(name: str) -> None:
    """Find no executable, as on a machine without ripgrep."""
    del name


def _tree(root: Path) -> None:
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("def main():\n    return Key(1)\n")
    (root / "src" / "keys.ts").write_text("const keyring = 1;\n")
    (root / "notes.md").write_text("A key, a KEY and a monkey\n")
    (root / "logo.png").write_bytes(b"\x89PNG\x00 key")


def _found(search: Search) -> dict[str, list[tuple[int, str, list[tuple[int, int]]]]]:
    """Each file's hits, by path under the root, as line, text and ranges."""
    found = search.run()
    return {
        each.path.removeprefix(search.root + "/"): [
            (hit.line, hit.text, hit.ranges) for hit in each.hits
        ]
        for each in found.files
    }


def test_text_matches_any_case_by_default_and_skips_binary_files(
    tmp_path: Path,
    engine: str,
) -> None:
    _tree(tmp_path)

    found = _found(Search(root=str(tmp_path), query="key"))

    assert Search(root=str(tmp_path), query="key").run().engine == engine
    assert found == {
        "notes.md": [(1, "A key, a KEY and a monkey", [(2, 5), (9, 12), (22, 25)])],
        "src/keys.ts": [(1, "const keyring = 1;", [(6, 9)])],
        "src/main.py": [(2, "    return Key(1)", [(11, 14)])],
    }


def test_case_word_and_regex_narrow_the_matches(tmp_path: Path, engine: str) -> None:
    del engine
    _tree(tmp_path)
    root = str(tmp_path)

    cased = _found(Search(root=root, query="KEY", case=True))
    words = _found(Search(root=root, query="key", word=True))
    regex = _found(Search(root=root, query=r"Key\(\d\)", regex=True))
    plain = _found(Search(root=root, query=r"Key\(\d\)"))

    assert cased == {"notes.md": [(1, "A key, a KEY and a monkey", [(9, 12)])]}
    assert list(words) == ["notes.md", "src/main.py"]
    assert regex == {"src/main.py": [(2, "    return Key(1)", [(11, 17)])]}
    assert plain == {}


def test_globs_to_include_and_exclude_are_comma_separated(
    tmp_path: Path,
    engine: str,
) -> None:
    del engine
    _tree(tmp_path)
    root = str(tmp_path)

    included = _found(Search(root=root, query="key", include="*.py, *.md"))
    excluded = _found(Search(root=root, query="key", exclude="src/**"))

    assert list(included) == ["notes.md", "src/main.py"]
    assert list(excluded) == ["notes.md"]


def test_ranges_count_utf16_units_as_a_browser_indexes_a_string(
    tmp_path: Path,
    engine: str,
) -> None:
    del engine
    (tmp_path / "a.txt").write_text("😀 héllo wörld\n")

    found = _found(Search(root=str(tmp_path), query="wörld"))

    # The emoji is two UTF-16 units and four UTF-8 bytes; é is one and two.
    assert found == {"a.txt": [(1, "😀 héllo wörld", [(9, 14)])]}


def test_a_long_line_is_cut_around_its_first_match(tmp_path: Path, engine: str) -> None:
    del engine
    (tmp_path / "min.js").write_text("x" * 900 + "needle" + "y" * 900 + "\n")

    [[(line, text, ranges)]] = _found(
        Search(root=str(tmp_path), query="needle"),
    ).values()

    assert (line, len(text)) == (1, 400)
    assert text[ranges[0][0] : ranges[0][1]] == "needle"


def test_past_the_cap_the_result_says_it_is_truncated(
    tmp_path: Path,
    engine: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del engine
    monkeypatch.setattr(files_search, "MAX_HITS", 2)
    (tmp_path / "a.txt").write_text("key\nkey\nkey\n")

    found = Search(root=str(tmp_path), query="key").run()

    assert (found.matches, found.truncated) == (2, True)


def test_a_stopped_search_returns_what_it_found_as_truncated(tmp_path: Path) -> None:
    if shutil.which("rg") is None:
        pytest.skip("ripgrep is not installed")
    (tmp_path / "a.txt").write_text("key\n")
    search = Search(root=str(tmp_path), query="key")

    search.stop()
    found = search.run()

    assert (found.matches, found.truncated) == (0, True)


def test_an_invalid_regex_says_why(tmp_path: Path, engine: str) -> None:
    del engine
    (tmp_path / "a.txt").write_text("key\n")

    with pytest.raises(SearchError):
        Search(root=str(tmp_path), query="(unclosed", regex=True).run()


@pytest.mark.parametrize(
    "line",
    [
        b"not json",
        b'{"data": {}}',
        json_dumps(
            {
                "data": {
                    "path": {"bytes": "YQ=="},
                    "lines": {"text": "key"},
                    "line_number": 1,
                    "submatches": [],
                },
            },
        ).encode(),
    ],
)
def test_malformed_or_non_utf8_ripgrep_matches_are_ignored(line: bytes) -> None:
    assert files_search._rg_hit(line) is None


def _rows_with_metadata(root: Path, query: Query) -> list[Row]:
    assert query.output_mode == "content"
    prefix = f"{root}/a.txt:1:"
    return [
        Row(text="--"),
        Row(text=prefix + "key", prefix_end=len(prefix), column=len(prefix)),
    ]


def test_python_search_ignores_rows_that_locate_no_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", _nowhere)
    monkeypatch.setattr(files_search, "grep_rows", _rows_with_metadata)

    assert _found(Search(root=str(tmp_path), query="key")) == {
        "a.txt": [(1, "key", [(0, 3)])],
    }


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
