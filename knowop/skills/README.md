# Agent instructions

`using-knowop/SKILL.md` is what an agent running inside knowop is told it can do: Chat's
lines, the person's canvas, its pull requests and Auto-fix, and filing results to trax.
`knowop run` gives its body to every claude and codex it starts as their standing
instructions (`--append-system-prompt`, codex's `developer_instructions`), and types it
again after a compaction, or on the first line to a CLI with no such flag. Those words
live there alone: change what an agent in a run is told there. Its commands are
`knowop canvas`, never `trax canvas`, since a computer may have knowop alone.

Each of knowop's packages keeps its instructions for an agent that drives it in
`skills/<package>/SKILL.md`: `shell/` for `knowop connect`, `agent/` for `knowop
run`, and `jobs/` for `knowop job`.

`knowop skill install` will place them in an agent's skill folder. Until then they are
read where they are. They still name the commands as they were before the move into
knowop.

A skill documents what the command does, in the words of its `--help`. When a
command changes, its skill changes in the same pull request.
