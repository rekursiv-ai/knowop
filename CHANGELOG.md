# Changelog

All notable knowop changes are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Fixed

- `knowop job submit` works from a shallow clone (`git clone --depth 1`). It
  failed after the node was granted, with `git bundle failed: error: Could not
  read <parent>`, because the clone lacks the history below its `HEAD`. Its
  snapshot is now a commit of its own with no parent, sent as a full bundle;
  submit says so, and the `snapshot:` line names the `HEAD` it was taken of.

## 0.1.5 - 2026-10-26

### Changed

- Requires trackinizer 0.1.12 and treekle 0.3.3. The session records the agent
  runner captures are `trackinizer.sagent.session.types` classes, no longer
  `trackinizer.lib.agent.types.sessions`, and are stored in treekle 0.3.3's form.
- Agent messages and host records are written with treekle's typed
  serialization, and a loaded record is rebuilt through its constructor, so its
  validation runs.

### Fixed

- `knowop connect` joins again after a disconnect. A saved machine credential the
  server refuses at the first heartbeat, revoked (a Disconnect on the Machines
  page, `knowop disconnect`) or unknown (the machine was deleted), is deleted, and
  the connect joins once more, with knowop's login or the piped token, and prints
  `rejoined: machine NAME`. Before, every later connect printed `revoked` and
  exited until the credential file was removed by hand. A refused join says what to
  run: `knowop login`, then `knowop connect`, or a new enrollment token. A revoke
  while serving still stops the connect, and now deletes its credential.

## 0.1.4 - 2026-10-25

### Added

- `knowop canvas` reads and changes a canvas with the grammar of `trax canvas`,
  on knowop's login. `$TRAX_CANVAS` names the canvas when the command names none,
  so an agent `knowop run` started drives the person's canvas on a computer with
  knowop alone.
- `knowop connect` prints `knowop: this knowop is too old for SERVER: run uvx
  knowop@latest connect SERVER` once, when the server's heartbeat answer says this
  host is older than it supports, and keeps serving.

### Changed

- A run's canvas note names `knowop canvas` where it named `trax canvas`. It says
  to open a terminal the person asks for in their canvas (`trax.terminal`), not in
  a desktop terminal app.
- A run attached to a chat (`--attach SESSION`) gets `TRAX_CANVAS`,
  `TRAX_WORKSPACE` and the canvas note for the canvas its chat sits on, read from
  the server's session list. It warns and runs unplaced on a server without that
  list.
- Requires pydantic 2.12, with which trackinizer's `Client` ignores a field a newer
  server added to an answer, so `knowop connect` keeps serving terminals when the
  server adds one to a terminal wait.

### Fixed

- `knowop status` and `knowop disconnect` no longer fail with `AttributeError`:
  0.1.3 called `Client.machine_runs` and `Client.disconnect_machine`, which
  trackinizer 0.1.10 lacks. Requires `trackinizer>=0.1.11`.

## 0.1.3 - 2026-10-25

### Added

- `knowop` with no verb prints the next step, from the saved login and this
  computer's live connection: `knowop connect` when it is not connected, else
  `knowop run claude`. A refused key says to run `knowop login`.
- `knowop disconnect [NAME]` stops this computer's runs, then disconnects NAME on
  the server: its connect is revoked and every run of yours that names it stops.
  `knowop run stop --all` stops every detached run on this computer.

### Changed

- A `knowop run` whose chat refuses its drain for good (a 403, a 404, or one of
  `RUN_ENDING_REFUSALS`, now with `provider_is_cli`) stops with one line naming the
  server's reason, where it used to retry unseen for ever. A failure it retries
  says what failed.
- `knowop status` lists the runs live on each of your machines, and the runs here
  that name no machine of yours.

### Fixed

- A reconnect with a saved login starts its terminal, command and job listeners
  only after its first heartbeat takes the machine, so the first wait no longer
  fails with 403 "stale host instance".
- Capture keeps a base64 image block whole, or replaces it with a text block naming
  its size, so a resumed Claude session no longer fails with "invalid base64 data".

## 0.1.2 - 2026-10-23

### Changed

- **Breaking:** use Canvas APIs and `TRAX_CANVAS` for chat placement, and
  Workspace APIs and `TRAX_WORKSPACE` for macro efforts. Requires
  `trackinizer>=0.1.10` and a server with the matching migration.
- Version 0.1.1 does not include the Canvas/Workspace rename.

## 0.1.1 - 2026-10-23

## 0.1.0 - 2026-10-23

### Added

- `knowop login [ORG|URL]`: device login, approved in a browser. No argument
  means `https://alpha.knowop.ai`; `ORG` means `https://ORG.knowop.ai`.
- `knowop connect [ORG|URL] [--name N] [--no-exec] [--sandbox]`: logs in when
  needed, enrolling and confirming this computer in the same approval, then
  serves its terminal, and the assistant's commands unless `--no-exec`.
- `knowop status`, `knowop logout`.
- `knowop run claude|codex`: Claude Code or Codex in this folder, recorded and
  reachable from Chat; `knowop run attach|ls|log|send|stop NAME` reaches a
  detached run, and `knowop run claude|codex resume SESSION` re-enters a stored
  one.
- `knowop job`: runs a job on a GPU node from this repository, reads its log and
  results, lists and cancels jobs, and keeps the secrets a job may use.
- `knowop machine`: the machines campaigns may run on.
- The `direct` extra: the direct terminal link between the browser and this
  computer.
