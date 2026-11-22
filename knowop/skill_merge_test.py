"""Tests for ``skill_merge.py``: skill folders merged into one, as links."""

from __future__ import annotations

from typing import TYPE_CHECKING

from knowop.skill_merge import merged


if TYPE_CHECKING:
    from pathlib import Path


def _skill(folder: Path, name: str) -> Path:
    (folder / name).mkdir(parents=True)
    (folder / name / "SKILL.md").write_text(f"---\nname: {name}\n---\n")
    return folder / name


def test_each_skill_is_linked_the_first_folder_s_winning_a_name(tmp_path: Path) -> None:
    cloud, repo = tmp_path / "cloud", tmp_path / "repo"
    into = tmp_path / "home" / ".local" / "skills"
    shared, mine, tdd = (
        _skill(cloud, "shared"),
        _skill(cloud, "mine"),
        _skill(repo, "tdd"),
    )
    _ = _skill(repo, "shared")
    (repo / "README.md").write_text("Not a skill.\n")
    (repo / "notes").mkdir()
    (repo / "lower").mkdir()
    (repo / "lower" / "skill.md").write_text("Not named as sagent opens it.\n")

    links = merged([cloud, repo], into=into)

    assert links == [into / "mine", into / "shared", into / "tdd"]
    assert [each.readlink() for each in links] == [
        mine.resolve(),
        shared.resolve(),
        tdd.resolve(),
    ]
    assert sorted(each.name for each in into.iterdir()) == ["mine", "shared", "tdd"]


def test_an_entry_already_there_stays(tmp_path: Path) -> None:
    into = tmp_path / "into"
    own = _skill(into, "tdd")
    _ = _skill(tmp_path / "repo", "tdd")
    (into / "gone").symlink_to(tmp_path / "nowhere")
    _ = _skill(tmp_path / "repo", "gone")

    assert merged([tmp_path / "repo"], into=into) == []
    assert not own.is_symlink()
    assert (into / "gone").readlink() == tmp_path / "nowhere"


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
