---
name: knowop-connect
description: ALWAYS invoke this skill before connecting a computer with knowop connect, serving or refusing the commands a server's agent may run on it, declaring a sandbox, or reading and changing the registry of machines. Do not tell a person how to connect a computer, or run a machine form, from memory -- invoke this skill first.
---

# knowop connect -- connect this computer

`knowop connect` connects the computer you are on to a server. It serves the
computer's shell to the person's browser (a phone will do) and runs the commands
the server's agent sends for that person. The registry of machines is `trax
machine`'s.

`knowop connect --help` is the source of truth for the flags; if this skill
disagrees, trust the help.

## Connect

```bash
knowop connect                                # log in if needed, join, serve
knowop connect https://trackinizer.example --name gx10
knowop connect --sandbox                      # this computer is disposable
knowop connect --no-exec                      # terminals only
knowop connect --jobs                         # also run your knowop job --machine jobs
knowop connect --name box-7 < enrollment-token   # join with a token, no login
```

- It runs in the foreground until the machine is revoked, a newer `knowop connect`
  of the same machine takes over, or Ctrl-C. Ctrl-C, and the termination signal a
  service manager sends, show the machine offline at once. Keep it in tmux or as a
  user service.
- With no knowop login for the server it prints a link and a code. The person
  approves the code in a browser, and approving also allows the machine. The key
  it receives is saved as knowop's profile, and `knowop run` and `knowop job` use
  the same login.
- Logged in already, it enrolls a new machine with the login's key; the person
  then allows the machine on the web app's Machines page.
- An enrollment token (`enr_...`) on standard input joins without a login. Put it
  there, never on the command line.
- Later runs reuse the login and the machine's credential and ask for nothing. A
  credential is used only by the account it was saved for.
- `ORG` means `https://ORG.knowop.ai`; a URL names any server. The default is the
  server knowop is logged in to, else the one `knowop connect --help` names. A
  server over plain `http` must be this computer: any other is refused, because the
  key and the server's commands would cross the network in the clear.
  `--name NAME` overrides the machine name, which is otherwise the computer's
  short hostname.
- One computer can be a machine of several servers: run one `knowop connect` per
  server.

## What the server may do here

- **Terminals.** The person's browser opens a login shell on this computer. Closing
  the terminal ends the shell's process group; `tmux` and `nohup` jobs keep running.
  Every install includes aiortc for the direct link.
  `TRAX_MACHINE_TERMINAL_DIRECT=0` turns it off; terminals use the server relay.
- **Commands.** The server's agent may run `bash -lc COMMAND` as you, in a process
  group of its own, with at most 600 seconds, standard input from the request, and
  each output stream cut at 1 MiB from the middle. A command killed at its timeout
  exits 124. Each command is appended to `NAME.exec.log` in the machines folder of
  your user state, readable only by you, with who asked for it.
  `--no-exec` serves terminals only; the server sees no exec fact and refuses.
  The command inherits this host's environment less `TRACKINIZER_URL`,
  `TRACKINIZER_TOKEN`, `TRACKINIZER_PROFILE` and every `TRAX_*` and `LC_TRAX_*`
  name, so the server's agent never reads a key from it. The log line is written
  when the command ends, and names the variables the request set; a command that
  takes the host down with it leaves none.
- **Jobs.** `--jobs` also runs the jobs your account sends with `knowop job submit
  --machine NAME`, and no one else's. Each fetches your pushed commit into a fresh
  worktree and runs under a memory cap in a `systemd-run --user` scope with no swap,
  so `--jobs` refuses a computer without systemd's user manager. A GPU that shares
  the CPU's memory (GB10) is given to no job, since no cap bounds what CUDA takes
  there. Jobs keep running when `knowop connect` stops; the next one adopts them.
  `--job-results-max SIZE` (default 256M) caps the results one job sends back.
- **Sandbox.** `--sandbox` declares the computer disposable, so agents started
  there by `knowop run` ask for no permissions. Every run records its answer, so a
  run without the flag withdraws it. Never declare a computer that holds work or
  keys you would mind losing.

## The registry of machines

```bash
knowop machine                                # name, role, labels, how
knowop machine gx10                           # every field and the status
knowop machine gx10 role                      # print a field as stored
knowop machine gx10 role to dev               # set it (creates the machine)
knowop machine gx10 how to @how.txt           # a literal, - for stdin, or @FILE
knowop machine gx10 label add gpu             # add or remove a label
knowop machine gx10 del                       # unregister; the computer is untouched
```

- Listing and showing need the writer role; setting, labelling and deleting need
  the admin role.
- A name is lowercase letters, digits and `-`, at most 63 characters, and not one of
  `enroll join init import check connect leave top`.
- A role is `[a-z][a-z0-9-]{0,31}`, a `how` line is at most 2,000 characters, and `to ''`
  clears a field. From stdin or a file one trailing newline is stripped and an empty
  value is refused.
- Read a machine's `how` before using it. The registry only records machines; it never
  reaches one.

## Rules

- Do not put a secret on the command line.
- `trax machine` moved out of trax: it exits 2 and says so. Connect with
  `knowop connect`, and read the registry with `knowop machine`.
