# Changelog

All notable knowop changes are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
