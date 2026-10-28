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
knowop connect          # this computer: log in, enroll, serve it in the background
knowop logs -f          # follow this computer's connect
knowop run claude       # Claude Code in this folder, recorded, reachable from Chat
knowop run codex        # the same for Codex
knowop run stop NAME    # stop a run of yours, here or on another of your machines
knowop job submit -- python train.py   # a job on a GPU node, from this repository
knowop job submit --machine lab-box -- python train.py   # on your computer lab-box
knowop status           # server, account, your machines, their runs, this computer's services
knowop disconnect       # stop this computer's runs and connect, then disconnect it
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
approval logs you in, enrolls this computer and confirms it. The computer's terminal
then opens in the web app, its files can be browsed there (read-only, as you), and
it runs the commands the server's assistant sends; `--no-exec` serves the terminal
and files only. `knowop connect acme` does the same on `https://acme.knowop.ai`. A
computer with no browser, such as a box, pipes an
enrollment token to `knowop connect` instead, and joins with no login.

At a terminal, `knowop connect` joins in the foreground, so a refusal shows there,
then keeps serving from a user service: a launchd agent in `~/Library/LaunchAgents`
on macOS, a `systemd --user` unit on Linux. It runs in the background, starts again
at login and after a crash, and returns once the server shows the computer online.
`knowop logs -f` follows its log, `knowop status` lists it with its flags, and
`knowop disconnect` stops and removes it. Running `knowop connect` again restarts it
only when its flags, its environment, its knowop or the CLIs it finds changed. The
service runs the knowop that installed it: an installed one in place, so after `uv
tool upgrade knowop` run `knowop connect` again, and one `uvx` ran as `uvx
knowop@VERSION`, which `uv cache clean` cannot take away; `uvx knowop@latest
connect` moves it to the latest. It keeps your shell's `PATH`, `LANG`, `LC_*` and
the variables your per-user folders are read from (`HOME`, `XDG_*_HOME`), so it
finds your CLIs and your login as your shell does. It holds no secret: it reads the
machine credential saved for your user each time it starts. It is never installed
as root. Over ssh on Linux, `sudo
loginctl enable-linger $USER` keeps it running after you log out; knowop says so.
Anywhere else, such as a pipe, a box or a service of your own, and with
`--foreground`, `knowop connect` serves in the foreground until Ctrl-C.

Plain `knowop login` logs in to `https://alpha.knowop.ai`; `knowop login acme` to
`https://acme.knowop.ai`, and `knowop login URL` to any server. Use it on a computer
you will not connect, where only your agents work. A later `knowop connect` there
enrolls the computer with that login's key, and it waits until you confirm it in the
web app. A first `knowop run` logs in too.

## In Docker

A container has no service manager: `knowop connect --foreground` serves it, and
Docker starts it again. A volume on `/root` keeps its login, the machine credential
and uv's cache, so a new container joins again with no login:

```bash
docker run -d --name knowop --restart unless-stopped -v knowop:/root \
  ghcr.io/astral-sh/uv:python3.12-bookworm-slim \
  uvx knowop connect --foreground --name my-box
docker logs -f knowop   # approve the code it prints, then Ctrl-C
```

Its terminals open a shell in the container, as root, and its agents and jobs run
there: start from an image that has the tools they need (`git`, `claude`, `codex`).
Docker starts it again after a disconnect from the web app; `docker stop knowop`
keeps it stopped.

## As a systemd system service

`knowop connect` at a terminal installs a user service, which stops when you log
out unless lingering is on. To serve a Linux server with no one logged in, run it
from a system unit as your own user instead. Connect once in the foreground, which
logs in and enrolls this computer, then install the unit:

```bash
uv tool install knowop
knowop connect --foreground   # approve the code; once it is serving, Ctrl-C
sudo tee /etc/systemd/system/knowop-connect.service > /dev/null << EOF
[Unit]
Description=knowop connect
Wants=network-online.target
After=network-online.target

[Service]
User=$USER
Environment=PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=$HOME/.local/bin/knowop connect --foreground
Restart=on-failure
RestartSec=5
# knowop ends as Ctrl-C does on the SIGTERM that stops it.
SuccessExitStatus=130

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now knowop-connect
journalctl -u knowop-connect -f
```

`PATH` names where your agents' CLIs are. `knowop status` shows the machine, not the
unit, and `knowop logs` reads only a user service's log: use `journalctl`. `sudo
systemctl disable --now knowop-connect` stops it.

## Commands

| Command | Does |
|---|---|
| `knowop login [ORG\|URL]` | Logs in with a code approved in a browser and saves the key. |
| `knowop connect [ORG\|URL] [--name N] [--no-exec] [--sandbox] [--jobs [--job-results-max SIZE]] [--foreground] [--follow]` | Logs in first unless knowop is logged in to that server or a token is piped in, enrolling and confirming this computer in the same approval; then serves its terminal, and the assistant's commands unless `--no-exec`. At a terminal it serves from a user service that starts again at login, and `--follow` then follows its log; elsewhere, and with `--foreground`, until Ctrl-C. No server means knowop's own, else `https://alpha.knowop.ai`. `--sandbox` declares a disposable computer where agents act without asking. `--jobs` also runs your account's `knowop job submit --machine` jobs, each under a memory cap; it needs systemd's user manager. |
| `knowop logs [NAME] [-f] [-n N]` | Prints the log of NAME's connect service on this computer (default this computer's name), its last N lines, or follows it until Ctrl-C. A machine connected from another computer keeps its log there. |
| `knowop status [--json]` | Prints, for each server knowop is logged in to (the one its commands use first), the profile, the server, your account and role, your computers with whether each is online, the knowop version and process serving each one online, and the runs live on each; the runs here that name no machine of yours; and this computer's connect services with their flags, and whether each is turned off (macOS Login Items). `--json` prints the same as one JSON object, one entry per login. Exits 1 when not logged in, or when a server refuses its login's key. |
| `knowop disconnect [NAME]` | Stops this computer's runs when NAME is this computer (the default), then disconnects NAME on the server: its connect is revoked and every run of yours that names it stops. Removes NAME's connect service on this computer, and waits until the server shows NAME offline. `knowop connect` joins it again with no login. |
| `knowop logout [NAME\|--all]` | Logs this computer's machine out on the server (its connect revoked, its runs stopped, the keys its logins made revoked), revokes knowop's key if that left it, removes this computer's connect services of that server, and deletes the saved login; knowop then uses its login to another server, if it has one, and says which. `--all` logs every machine of yours out first. NAME logs that one machine out alone, as the Machines page does, and keeps this computer's login. |
| `knowop run claude\|codex [ARGS]` | Runs Claude Code or Codex in this folder, recorded and reachable from Chat. Says to run `knowop connect` when this computer is not connected. |
| `knowop run attach\|ls\|log\|send\|stop NAME` | Reaches a run started with `--detach`. `knowop run stop --all` stops every one on this computer. `knowop run stop RUN` stops any live run of yours another machine runs, by its name or session (its first 8 characters will do); a RUN that names several lists them. |
| `knowop run claude\|codex resume SESSION` | Re-enters a stored session. |
| `knowop job submit\|logs\|results ...` | Runs a job on a GPU node from this repository, or with `--machine NAME` on a computer of yours, and reads its log and results. |
| `knowop job status [JOB]\|cancel JOB\|secrets\|secret NAME set\|del` | Lists your jobs or shows one, stops one, and keeps the secrets a job may use. |
| `knowop machine [NAME ...]` | Lists, shows, sets, labels and deletes the machines campaigns may run on. |
| `knowop canvas [CANVAS_UUID] [show TYPE\|navigate ROUTE\|highlight UUID,...\|hide ID\|KEY to VALUE ...]` | Reads and changes a canvas with the grammar of `trax canvas`. In a run that `knowop run` placed on a canvas, `$TRAX_CANVAS` names it, so an agent drives the person's canvas with knowop alone. |
| `knowop --version` | Prints the version; `unknown` from a checkout. |

Every command takes `--help`; `knowop run claude --help` and `knowop job submit
--help` show the runner's own.

knowop keeps a login per server it logs in or connects to, and every command uses
the last one; `knowop status` shows each. A connect service signs in with its own
server's login, so this computer can serve `alpha` and another server at once. Each
key is saved as a `trax` profile, and `trax`'s own current profile stays as it was:
knowop never acts on a server it did not log in to.

## Direct terminal link

Every install includes aiortc for a direct link between the browser and this
computer. Terminals use the server relay when a direct link is unavailable.

## Layout

| Package | Holds |
|---|---|
| `cli.py` | The `knowop` command. |
| `service.py` | The user service `knowop connect` installs: launchd's or systemd's. |
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
