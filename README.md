# knowop🔌

[![PyPI version](https://img.shields.io/pypi/v/knowop.svg)](https://pypi.org/project/knowop/)
[![CI](https://github.com/rekursiv-ai/knowop/actions/workflows/package-validation.yml/badge.svg?branch=main)](https://github.com/rekursiv-ai/knowop/actions/workflows/package-validation.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/discord/1530237005311639592?logo=discord&logoColor=white&label=Discord&color=5865F2)](https://discord.gg/2GZFPPvCqn)

One command to log in to KnowOp, connect your computer, and run your agents and GPU
jobs where Chat can reach them.

## Quick Start

```bash
# Mac:
#   # Required for quick install.
#   brew install uv

# Ubuntu/Debian:
#   # Required for quick install.
#   sudo apt-get install -y curl
#   curl -LsSf https://astral.sh/uv/install.sh | sh

uvx knowop connect
```

`uvx knowop connect` runs knowop without installing it. To keep the command for
every day, install it:

```bash
uv tool install knowop
# Alternatively: python -m pip install knowop
```

## Use

```bash
knowop connect          # this computer: log in, enroll, serve its terminal and commands
knowop run claude       # Claude Code in this folder, recorded, reachable from Chat
knowop run codex        # the same for Codex
knowop job submit -- python train.py   # a job on a GPU node, from this repository
knowop job submit --machine lab-box -- python train.py   # on your computer lab-box
knowop status           # server, account, your machines and the runs live on each
knowop disconnect       # stop this computer's runs, then disconnect it on the server
```

A run reaches Chat with your account's key, not through `knowop connect`: it keeps
running after Ctrl-C ends the connect, which then names the runs it leaves. The web
app shows a machine connected while one of its runs is live. `knowop run stop --all`
stops the runs on this computer; `knowop disconnect` also revokes its connect and
stops every run of yours that names it, wherever it runs.

A job can also run on a computer of your own: start it with `knowop connect --jobs`
there, push your branch, and `knowop job submit --machine NAME -- CMD` from the
repository. That computer fetches the pushed commit into a fresh worktree and runs CMD
under a memory cap (a `systemd-run --user` scope), only for your account; its log and
results come back through the server, which keeps none of them.

A job runs with the full authority of the Unix user running `knowop connect --jobs`,
including access to that user's files, credentials and inherited environment. The
worktree and memory cap do not restrict that authority. GPU selection through
`CUDA_VISIBLE_DEVICES` is cooperative: submitted code can change it. Run untrusted
code under a separate Unix identity or external isolation that enforces the access
restrictions you need.

`knowop connect` prints a code; approve it in your browser, on any device. That one
approval logs you in, enrolls this computer and confirms it. Until you press Ctrl-C,
the computer's terminal opens in the web app, and it runs the commands the server's
assistant sends; `--no-exec` serves the terminal only. `knowop connect acme` does the
same on `https://acme.knowop.ai`. A computer with no browser, such as a box, pipes an
enrollment token to `knowop connect` instead, and joins with no login.

Plain `knowop login` logs in to `https://alpha.knowop.ai`; `knowop login acme` to
`https://acme.knowop.ai`, and `knowop login URL` to any server. Use it on a computer
you will not connect, where only your agents work. A later `knowop connect` there
enrolls the computer with that login's key, and it waits until you confirm it in the
web app. A first `knowop run` logs in too.

## Commands

| Command | Does |
|---|---|
| `knowop login [ORG\|URL]` | Logs in with a code approved in a browser and saves the key. |
| `knowop connect [ORG\|URL] [--name N] [--no-exec] [--sandbox] [--jobs [--job-results-max SIZE]]` | Logs in first unless knowop is logged in to that server or a token is piped in, enrolling and confirming this computer in the same approval; then serves its terminal, and the assistant's commands unless `--no-exec`, until Ctrl-C. No server means knowop's own, else `https://alpha.knowop.ai`. `--sandbox` declares a disposable computer where agents act without asking. `--jobs` also runs your account's `knowop job submit --machine` jobs, each under a memory cap; it needs systemd's user manager. |
| `knowop status` | Prints knowop's profile, the server, your account and role, your computers with whether each is online and the runs live on each, and the runs here that name no machine of yours. Exits 1 when not logged in. |
| `knowop disconnect [NAME]` | Stops this computer's runs when NAME is this computer (the default), then disconnects NAME on the server: its connect is revoked and every run of yours that names it stops. `knowop connect` joins it again with no login. |
| `knowop logout` | Logs this computer's machine out on the server (its connect revoked, its runs stopped, the keys its logins made revoked), revokes knowop's key if that left it, and deletes the saved login. |
| `knowop run claude\|codex [ARGS]` | Runs Claude Code or Codex in this folder, recorded and reachable from Chat. Says to run `knowop connect` when this computer is not connected. |
| `knowop run attach\|ls\|log\|send\|stop NAME` | Reaches a run started with `--detach`. `knowop run stop --all` stops every one on this computer. |
| `knowop run claude\|codex resume SESSION` | Re-enters a stored session. |
| `knowop job submit\|logs\|results ...` | Runs a job on a GPU node from this repository, or with `--machine NAME` on a computer of yours, and reads its log and results. |
| `knowop job status [JOB]\|cancel JOB\|secrets\|secret NAME set\|del` | Lists your jobs or shows one, stops one, and keeps the secrets a job may use. |
| `knowop machine [NAME ...]` | Lists, shows, sets, labels and deletes the machines campaigns may run on. |
| `knowop canvas [CANVAS_UUID] [show TYPE\|navigate ROUTE\|highlight UUID,...\|hide ID\|KEY to VALUE ...]` | Reads and changes a canvas with the grammar of `trax canvas`. In a run that `knowop run` placed on a canvas, `$TRAX_CANVAS` names it, so an agent drives the person's canvas with knowop alone. |
| `knowop --version` | Prints the version; `unknown` from a checkout. |

Every command takes `--help`; `knowop run claude --help` and `knowop job submit
--help` show the runner's own.

knowop keeps one login and uses it for every command. It saves the key as a `trax`
profile, and `trax`'s own current profile stays as it was: knowop never acts on a
server it did not log in to.

## Direct terminal link

Every install includes aiortc for a direct link between the browser and this
computer. Terminals use the server relay when a direct link is unavailable.

## Layout

| Package | Holds |
|---|---|
| `cli.py` | The `knowop` command. |
| `shell/` | The machine host `knowop connect` runs: terminals, the assistant's commands, and with `--jobs` your jobs (`job_host.py`). |
| `agent/` | The agent runner `knowop run` hands to. Its `__main__.py` and `codex.py` run it by module, on trax's connection, for unattended plumbing. |
| `jobs/` | The job runner `knowop job` hands to. |
| `machines.py` | The machine registry `knowop machine` reads and writes. |
| `canvas.py` | `knowop canvas`: trax's canvas command, named as knowop's. |
| `skills/` | Each package's instructions for agents. |

knowop calls the server through trackinizer's `Client`, as trax does, and never runs
`trax`. `trax machine` and `trax job` say they moved out of trax.

## Development

```bash
uv sync --all-groups
uv run pytest
```

## See also

Sibling projects in the [rekursiv-ai](https://github.com/rekursiv-ai) family:

- [trackinizer](https://github.com/rekursiv-ai/trackinizer) — Centralized agent database for tracking inquiries, work, and the evidence behind conclusions; knowop's `trax` and its client come from it.
- [sagent](https://github.com/rekursiv-ai/sagent) — The self-mutating multi-provider coding-agent CLI and typed Python library.
- [wesearch](https://github.com/rekursiv-ai/wesearch) — Web search, resilient page fetch, and scholarly-paper lookup without a browser stack.
