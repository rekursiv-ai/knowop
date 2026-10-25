# Changelog

All notable knowop changes are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
