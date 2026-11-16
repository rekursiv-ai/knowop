---
name: knowop-run
description: ALWAYS invoke this skill before running sagent with knowop run, starting an agent headless, on a host or from another agent, or continuing a stored session. Do not start an agent run, or tell a person which command reaches it, from memory -- invoke this skill first.
---

# knowop run -- sagent, live in Chat

`knowop run` runs sagent's own agent in this process, with sagent's screen at a
terminal and headless elsewhere. Its session is live in Chat: each event goes up as
it happens, its journal is stored as records, and a line sent from Chat reaches the
agent as one typed at its screen. A first run logs in when knowop has no login.
Every session opens in Chat the same way, however it was started; Chat names where
its agent runs.

`knowop run --help` is the source of truth for the flags; if this skill disagrees,
trust the help.

## Run

`knowop run [KNOWOP FLAGS] -- [SAGENT ARGUMENTS]`: knowop's flags first, then
everything of sagent's after `--`, untouched.

```bash
knowop run                          # sagent here with its screen, live in Chat
knowop run --headless               # no screen: this computer's daemon runs it
knowop run --session ID             # continue session ID here
knowop run --session AgentSession#412   # the same, by its number (or 412)
knowop run --on lab --cwd ~/work    # start it on your host lab, through its daemon
knowop run --stop ID                # stop session ID's agent, wherever it runs
knowop run --time 2:00:00           # end 2 h after it starts, as a stop does
knowop run -- --model opus          # sagent's settings go after --
knowop run -- login --provider oaisub   # sagent's sign-in, on this computer
knowop run -- --help                # sagent's own flags and actions
```

- The first line a run prints on stdout is its session's id.
- A sagent flag before `--` is refused, and the error shows the command with it
  after `--`; `--profile` goes before `run` (`knowop --profile NAME run ...`).
  A refused or mistyped command opens no session.
- An action after `--` (`login`, `mcp ...`, `--help`) runs on this computer as
  sagent's own command and opens no session; it takes none of knowop's flags. A
  sign-in made here is used by every later `knowop run` here.
- Every sagent flag works but those that pick a journal (`--resume`, `--continue`,
  `--ephemeral`, `--session` and their kin): knowop keeps the session's journal.
- Only a screen runs in the `knowop run` process. A run with none (`--headless`,
  or no terminal: an agent's tool call, a script, cron) asks this computer's
  daemon to start the agent in this folder (`--cwd` names another), waits for it
  to attach, prints the session's id and exits; the agent runs on until stopped.
- `--on NAME` does the same through host NAME's daemon (`knowop connect` there),
  in `~` unless `--cwd` names a folder. With no name, the server's own host runs it.
- A run on this computer connects it first when it is not, as `knowop connect`
  does, saying so and where its log is on stderr.
- Exit codes: 0 after a stop or a quit, and once a daemon's start attaches; 1 for a
  start error, a refused key or a session that is gone; 2 for a refused command; 3
  when another process holds the session; 4 when a daemon's start went 10 minutes
  without its socket; 130 on Ctrl-C. An action exits as sagent's command does.

## Start an agent from an agent

```bash
knowop run --headless               # on this computer, in this folder
knowop status                       # the hosts; this computer's says "(this computer)"
knowop run --on HOST --cwd ~/work   # on another host
```

- Each returns within seconds with the new session's id; the host's daemon runs
  the agent until it is stopped.
- Leave the model and effort at knowop's defaults; pass a sagent setting after
  `--` only when the person asked for it.
- Name the new agent `Agent#N` (below) to message it; stop it with
  `knowop run --stop ID`.

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

## The canvas

A line you send from Chat names the canvas it came from, once, after its words. Another person's line in a chat
you share names none: the agent acts for you, on your canvas. Every agent of the
run then acts on that canvas with its `Canvas` tool, with or without a shell, on
your own host and on Our cloud alike. Its `action` is the verb `knowop canvas`
takes, and the rest of its arguments are what the verb needs:

- `read` lists the canvas's page, visuals and red boxes, each with its id.
- `show` opens a `visual` (`trax.record`, `trax.subgraph`, `trax.timeline`,
  `trax.files`, `trax.terminal` and the rest) on a `record`, with its `params`;
  `hide` closes one. `chat` is a visual too: hide it to have the person look.
- `navigate` moves the page to a `route`, such as `#/pulls`.
- `highlight` points at records; `mark` and `unmark` draw and take away a red box;
  `capture` sees the canvas as the person does, as a picture.
- `set` changes how the app looks for the person: font, text sizes, theme, skin,
  Chat's side and the rest its schema lists, which the server gives. The tool sends
  no other setting: never the folder an agent starts in, nor auto-fix or
  auto-merge. For an agent a daemon or Our cloud started, the server refuses those
  to its key too. An agent you run at a terminal holds your own key, which the
  server lets set them, as `knowop canvas autofix to true` does: there only the
  tool, not the server, keeps `set` to its schema.

A record is `Issue#123` or its UUID. Until a line has named a canvas, the tool says
there is none. With a shell, `knowop canvas` does the same (`knowop canvas --help`).
The agent's key reaches the canvas its session's Chat is on, and no other.

## Continue a stored session

`--session` and `--stop` take a session's UUID or its number (`AgentSession#412`,
or `412`); Chat's ⋯ menu copies either (Copy session #, Copy UUID).

`knowop run --session ID` continues it here: in this host's own journal of its
newest part when this host holds it whole, else in a new journal restored from the
stored parts. A part another agent than sagent wrote is not continued; it stays
readable with `trax agentsession N records`.

## Rules

- Run an agent under `knowop run`, not bare `sagent`, when the work should be
  recorded or reachable from Chat.
- Do not put a secret on the command line; the agent's arguments are recorded.
