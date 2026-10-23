# Agent instructions

Each of knowop's packages keeps its instructions for agents in
`skills/<package>/SKILL.md`: `shell/` for `knowop connect`, `agent/` for `knowop
run`, and `jobs/` for `knowop job`.

`knowop skill install` will place them in an agent's skill folder. Until then they are
read where they are. They still name the commands as they were before the move into
knowop.

A skill documents what the command does, in the words of its `--help`. When a
command changes, its skill changes in the same pull request.
