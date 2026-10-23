---
name: knowop-run
description: ALWAYS invoke this skill before running Claude Code or Codex recorded with knowop run, detaching a run, attaching to one, sending it a message, stopping it, or resuming a stored session. Do not start or reach an agent run, or tell a person which command reaches it, from memory -- invoke this skill first.
---

# knowop run -- recorded agent runs

`knowop run claude` runs Claude Code in the current folder and `knowop run codex`
runs Codex. Each run is recorded as an AgentSession on the server knowop is logged
in to, and a message sent to it from Chat reaches it. A first run logs in when
knowop has no login. `trax run` moved out of trax.

`knowop run claude --help` is the source of truth for the flags; if this skill
disagrees, trust the help.

## Run

```bash
knowop run claude                              # Claude Code here, recorded
knowop run codex                               # Codex here, recorded
knowop run claude --as scientist --room lab -- --model opus
knowop run claude -- "fix the failing test"    # after -- everything goes to the CLI
```

- `--as NAME` names the session. Other agents and the web app address it as
  `@NAME`. It defaults to `$AGENTNAME`, else `Agent`, and the server makes it
  unique among live sessions (`scientist#2`).
- `--room ROOM` (repeatable) joins a room; address the session as
  `@NAME:ROOM`.
- `--private` starts the session hidden: only its owner, staff and admins read
  it. A server that will not hide it stops the run before the CLI starts.
- `--diff` shows the session's live `git diff` on its page.
- `--no-sync` or `--out PATH` records to a local JSONL file and uses no
  network. `--dry-run` replays existing session files and spawns nothing.
- `knowop run gemini` and `knowop run sh -- CMD` still wrap gemini and any
  command; `sh` records the command's input and output as one terminal.
- A run that starts an agent says so when this computer is not connected, so
  Chat cannot open its terminal: run `knowop connect` in another terminal.

A missing `claude` or `codex` binary stops the run and prints the install
command.

## Detach, then reach it by name

```bash
knowop run claude --detach --name lead --as lead
knowop run ls                    # live hosts; --all adds exited and lost ones
knowop run attach lead           # drive it here; Ctrl-\ detaches
knowop run log lead --follow     # scrollback, then live output
knowop run send lead "status?"   # type a line in and press Enter
knowop run stop lead             # stop it; waits until it has exited
```

- A detached run is hosted in a background process no terminal owns, so closing
  the shell or dropping ssh leaves it running. It records and takes messages
  exactly as a foreground run does.
- `--name` defaults to the `--as` name, else the CLI. A name belongs to one
  live host at a time.
- A host can be attached only on the computer it runs on. `ls`, `attach`,
  `log`, `send` and `stop` reach claude and codex hosts alike, and need no login.
- A Chat run has no terminal until someone attaches. Attaching takes it over;
  Chat gets it back with Reattach in the chat.
- To tell a person how to reach a Chat run, give `knowop run attach NAME` with
  the name Chat shows, on the computer Chat names.

## Resume a stored session

```bash
trax agentsession 42                            # its id, among the row's fields
knowop run claude resume SESSION_UUID           # re-enter it as claude
knowop run codex resume SESSION_UUID --lossy    # resume as codex
```

The target CLI is chosen here, not by whatever captured the session, so a
codex-captured session resumes as claude. The run writes into the session it
resumes as a new part. `--lossy` accepts a conversion that drops records the CLI
cannot express; without it such a resume is refused. `trax agentsession N run`
moved out of trax.

## What is recorded

- Every turn, tool call and result of the CLI's own session log, as records of
  one AgentSession, synced as they happen. The row and its transcript are read
  with `trax agentsession N` and `trax agentsession N records`.
- Secrets named in `$TRAX_REDACT_NAMES` are replaced by `[redacted:NAME]`
  before anything leaves the process.
- The permission, question and trust prompts of a synced run reach Chat as
  cards, and Chat's answers come back.
- Messages sent to the session (`trax send @NAME ...`, or Chat) are typed into
  the CLI once it is ready for them.
- A run placed on a station gets `$TRAX_WORKSPACE`, `$TRAX_WORKSTREAM` and
  `$TRAX_ACTOR`, and is told its canvas is the station's; the canvas and the
  rows are `trax`'s to read and write (see the trax skill).
- A sandbox computer (`knowop connect --sandbox`) starts agents with
  no permission prompts, unless the person passed the CLI's own permission
  flags.

## Rules

- Run an agent under `knowop run claude` or `knowop run codex`, not bare `claude`
  or `codex`, when the work should be recorded or reachable from Chat.
- Do not put a secret on the command line; the CLI's own arguments are recorded
  in the host list.
