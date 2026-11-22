"""Merge skill folders into one, as links, for an agent that loads one folder.

sagent loads the skills of the folder it starts in and its user's folder alone. An
agent given several skill folders has them merged into its user's folder: a link per
skill (a folder holding ``SKILL.md``), each to the skill itself, so the merged folder
works wherever it is read. Of two skills of one name, the first folder's is linked; an
entry already there stays.

A remote agent's service names its folders in ``KNOWOP_SKILL_DIRS``, which its process
merges at each start (``process/main.py``).

Typical usage example:

  merged([Path("cloud/skills"), Path(".sagent/skills")], into=user_skills)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


__all__ = ["SKILL_DIRS", "merged"]


SKILL_DIRS: Final = "KNOWOP_SKILL_DIRS"
"""The variable that names an agent's skill folders, joined by ``os.pathsep``."""


def merged(sources: Iterable[Path], *, into: Path) -> list[Path]:
    """Link each skill of ``sources`` into ``into``; return the links made.

    Args:
      sources: The skill folders, the first's winning a name.
      into: The folder an agent loads, made, with each above it, when absent.

    Returns:
      links: Those made, in order; none for a name ``into`` already holds.

    """
    into.mkdir(parents=True, exist_ok=True)
    links: list[Path] = []
    for source in sources:
        for skill in sorted(source.iterdir()):
            link = into / skill.name
            # By its exact name, as sagent opens it, on a filesystem that folds case too.
            held = {each.name for each in skill.glob("*")}
            if "SKILL.md" in held and not (link.is_symlink() or link.exists()):
                link.symlink_to(skill.resolve())
                links.append(link)
    return links
