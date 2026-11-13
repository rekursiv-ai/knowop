---
name: knowop-run
description: ALWAYS invoke this skill before running sagent with knowop run, starting it on a host, or continuing a stored session. Do not start an agent run, or tell a person which command reaches it, from memory -- invoke this skill first.
---

# knowop run -- sagent, live in Chat

`knowop run` runs sagent's own agent in this process, with sagent's screen at a
terminal and headless elsewhere. Its session is live in Chat: each event goes up as
it happens, its journal is stored as records, and a line sent from Chat reaches the
agent as one typed at its screen. A first run logs in when knowop has no login.

`knowop run --help` is the source of truth for the flags; if this skill disagrees,
trust the help.

## Run

```bash
knowop run                          # sagent here, live in Chat
knowop run --headless               # the same, with no screen
knowop run --session ID             # continue session ID here
knowop run --on lab --cwd ~/work    # start it on your host lab, through its daemon
knowop run --stop ID                # stop session ID's agent, wherever it runs
knowop run --time 2:00:00           # end 2 h after it starts, as a stop does
knowop run --model opus             # any other flag is sagent's own
```

- The first line a run prints on stdout is its session's id.
- Every sagent flag works but those that pick a journal (`--resume`, `--continue`,
  `--ephemeral` and their kin): knowop keeps the session's journal.
- `--on NAME` asks the host's daemon (`knowop connect` there) to start the agent,
  waits for it to attach, prints the session's id and exits. With no name, the
  server's own host runs it.
- A run says so when this computer is not connected; a connected computer also
  opens its terminal in Chat.
- Exit codes: 0 after a stop or a quit; 1 for a start error, a refused key or a
  session that is gone; 3 when another process holds the session; 4 when a headless
  run went 10 minutes without its socket; 130 on Ctrl-C.

## Reach it

- Chat sends it lines, and its Stop halts the turn; a slash line such as
  `/model opus` does what it does at sagent's screen.
- `knowop run --stop ID` stops the agent of session ID, on whichever host it
  runs, and prints `stopped ID on HOST`.
- `knowop run --time TIME` ends the agent TIME after it starts, as a stop does: it
  halts the turn and uploads its journal. TIME is Slurm's `--time`: `120`,
  `2:00:00` or `1-0`. With `--on`, the process on that host keeps the time.
- Its agents are `LABEL#N`: an agent's label (the root's is `--name`, `Agent` by
  default), then `N`, the session's number or the id the run printed. Another
  session's agent sends one with `AgentSend(to="lead#412", ...)`, on any host; a
  shell sends the same with `trax send lead#412 TEXT`. A message shows who sent it,
  as `worker#415`, and is answered by AgentSend to that name. No list of these
  names is shown: an agent learns one by being told it, or by a message from it.
  A message that reaches no one comes back as `[message refused] ...`.
- Over HTTP, with a key: `POST /api/sessions/ID/inputs` sends a line,
  `DELETE /api/sessions/ID/process` stops the agent, and
  `GET /api/sessions/ID/lines` follows it.
- `knowop status` lists each host's live sessions.

## Continue a stored session

`knowop run --session ID` continues it here: in this host's own journal of its
newest part when this host holds it whole, else in a new journal restored from the
stored parts. A part another agent than sagent wrote is not continued; it stays
readable with `trax agentsession N records`.

## Rules

- Run an agent under `knowop run`, not bare `sagent`, when the work should be
  recorded or reachable from Chat.
- Do not put a secret on the command line; the agent's arguments are recorded.
