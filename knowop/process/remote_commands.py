"""The slash lines a remote agent runs: what acts on its own conversation, and its view.

A remote agent runs on a machine of the service's, under the service's account, for
whoever the server admits. A slash line from Chat runs on sagent's ``Commands``,
whose registry also holds what a person runs on their own host: MCP servers, the
host's tool server, other machines' servers, sign-ins. On a remote agent none of those
may run: a line must not start a program, read or write a file of the machine's, open
a connection to an address it names, or change which tools, servers, hosts, sign-ins
or credentials the agent has. So a remote agent runs only the lines listed here, and
every other one is refused: a setting added to sagent later is refused until it is
listed. An agent on a person's own host runs every one.

Typical usage example:

  Commands(agent, ..., allowed=REMOTE_AGENT)
  /compact          # Runs.
  /mcp.list         # "/mcp.list is not available for a remote agent"
"""

from __future__ import annotations

from typing import Final

from sagent.tui.slash import Allowed


__all__ = ["REMOTE_AGENT"]


REMOTE_AGENT: Final = Allowed(
    names=frozenset(
        {
            # The conversation: end it, compact or clear it, halt or cancel its work,
            # message its agents, queue a line, list its calls.
            "quit",
            "exit",
            "help",
            "clear",
            "compact",
            "recompact",
            "halt",
            "kill",
            "send",
            "focus",
            "defer",
            "tasks",
            "watcher",
            # The model, among the providers the agent was started with
            # (``--allow-providers``), and its choices.
            "model",
            "model.provider",
            "model.context",
            "model.thinking",
            "model.thinking.*",
            "model.service_tier",
            "model.cache_ttl_sec",
            "model.manage_context_server_side",
            # How its own calls wait, and how they show.
            "runtime.detach",
            "runtime.retry",
            "tool.*.display.*",
            "screen.thinking",
        },
    ),
    refusal="is not available for a remote agent",
)
"""What a remote agent's slash lines may run, each a pattern of dotted names."""
