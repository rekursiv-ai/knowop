"""Every slash line sagent's registry holds is classed for a remote agent, run or refused.

The registry is harvested as a remote agent's commands hold it, its watcher's too, so a
setting sagent adds later fails here until it is listed as run or as refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import fnmatch

from sagent.agent.agent import Agent
from sagent.agent.team import Team
from sagent.model.scripted import ScriptedProvider
from sagent.providers.catalog import ModelCatalog
from sagent.settings.harvest import harvest
from sagent.tools.read import Read
from sagent.tui.prompt import Panes
from sagent.tui.slash import Commands

import pytest

from knowop.process.remote_commands import REMOTE_AGENT


if TYPE_CHECKING:
    from pathlib import Path

    from sagent.tui.types import CommandOutput


_REFUSED: Final = frozenset(
    {
        # A sign-in, or the order of the provider's sign-ins, which it writes down.
        "login",
        "login.precedence",
        "model.provider.account",
        # Whether the agent takes part in messaging across sessions.
        "peer-messaging",
        # A terminal's theme, saved as a file of the machine's.
        "screen.theme",
        "screen.theme.*",
        # MCP servers, this machine's tool server, other machines' servers.
        "mcp.*",
        "host",
        "host.*",
        "remote.*",
    },
)
"""What a remote agent refuses, by the reason it may not run: everything else."""


@pytest.fixture(autouse=True)
def working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


@dataclass(slots=True, kw_only=True)
class _Screen:
    show_thinking: bool = False


def _said(output: CommandOutput) -> None:
    del output


async def _registry() -> list[str]:
    """Return every name a remote agent's commands harvest, and its watcher's."""
    catalog = ModelCatalog(factories={"scripted": ScriptedProvider})
    agent = Agent(model=catalog["scripted"]["scripted"], tools=[Read()])
    async with Team(catalog=catalog) as team:
        _ = team.join(agent)
        commands = Commands(
            agent,
            catalog=catalog,
            say=_said,
            screen=_Screen(),
            panes=Panes(),
            team=team,
        )
        _ = await commands("/send parent watch")
        watcher = await commands.child_commands(team, "watcher")
        names = [
            ".".join(s.name)
            for each in (commands, watcher)
            for s in harvest(each.owners())
        ]
        await commands.aclose()
    return list(dict.fromkeys(names))


@pytest.mark.asyncio
async def test_every_line_of_the_registry_is_run_or_refused_never_both() -> None:
    unclassed = [
        name
        for name in await _registry()
        if REMOTE_AGENT.allows(tuple(name.split("."))) == _refused(name)
    ]
    assert unclassed == []


@pytest.mark.asyncio
async def test_every_pattern_names_a_line_the_registry_holds() -> None:
    registry = await _registry()
    stale = [
        pattern
        for pattern in REMOTE_AGENT.names | _REFUSED
        if not fnmatch.filter(registry, pattern)
    ]
    assert stale == []


def _refused(name: str) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in _REFUSED)


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
