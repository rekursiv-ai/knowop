---
name: knowop-job
description: ALWAYS invoke this skill to run a command on a GPU node from the repository you are in, or to read, stop or collect such a job -- knowop job submit, logs, results, status, cancel, secrets. Do not ssh to a node or copy code to it by hand -- invoke this skill first.
argument-hint: "[submit|logs|results|status|cancel|secrets|secret] ..."
user-invocable: true
tools: Bash, Read
---

# knowop job -- GPU jobs from your computer

`knowop job submit [--gpus N] [--minutes M] [--label TEXT] [--follow]
[--include-untracked] -- CMD...` runs CMD on a GPU node from the repository you
are in, and prints the job id. Everything after the first `--` is the job's
command, even `--help`. `--gpus` is 1 to 8 (default 1) and `--minutes` is 1 to
360 (default 10).

Core rule: `knowop job submit --help` is the source of truth. If this skill
disagrees with it, trust the help.

## What submit sends

- It snapshots the worktree without touching it: uncommitted and untracked
  files are included, ignored files are left out, and it says which untracked
  files it took.
- An untracked `.env*`, `*.pem`, `id_*`, `*credentials*`, `.netrc` or `*.key` is
  refused unless `.gitignore` covers it or `--include-untracked` is given.
- The code goes straight from this computer to the node over ssh, cut on top of
  the commits the node already holds. No code or result bytes pass through the
  server, and no key is kept on this computer: each command makes a throwaway
  key and deletes it when it ends.
- It needs `ssh`, `ssh-keygen` and `git` on `PATH`, as do `logs` and `results`.

## Follow a job

| Command | Does |
|---|---|
| `knowop job status [JOB]` | Lists your jobs, or shows one in full. |
| `knowop job logs JOB [-f]` | Prints the job's log; `-f` streams it until the job ends. |
| `knowop job results JOB [--to DIR]` | Downloads the job's `out/` from the node. |
| `knowop job cancel JOB` | Stops a job; minutes that already ran are charged. |

`results` fills `out/` and writes `job.json`, in your data folder unless `--to`
names a directory. It keeps `src.bundle`, the exact code that ran, beside them
for a later pull request. Files already held with the same size and digest are
skipped, so an interrupted download resumes when you run it again. A `--to`
inside the repository is refused.

`submit --follow` streams the log after printing the id. Exit codes: 0 on
success, 1 when the server refuses the job or it ends badly, 2 for a refused
command, 130 when you interrupt a log.

## Secrets

Set a job's Hugging Face or Weights & Biases token with the value on stdin:

```
knowop job secrets
knowop job secret HF_TOKEN set < FILE
knowop job secret WANDB_API_KEY del
```

The name is `HF_TOKEN` or `WANDB_API_KEY`. The job never holds the value: the
node adds it to the job's requests to Hugging Face and Weights & Biases. Never
put a secret on a command line, and never echo one into a row or your output.

## Choose the server

`knowop job` uses knowop's login, and refuses without one: run `knowop login`.
`trax job` moved out of trax: it exits 2 and says so.
