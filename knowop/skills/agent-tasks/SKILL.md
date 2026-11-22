---
name: agent-tasks
description: "ALWAYS invoke this skill when you start, plan, hand off or finish a task in KnowOp, a task an agent spawned you for included: work under the issue your brief names or your spawner's, or search Trackinizer for related issues, file the agreed plan, say which agent and session wrote it, link pull requests and evidence, mark what is done. Do not leave a task's state only in Chat."
---

# A task's issue in KnowOp

KnowOp keeps each task's state in its Trackinizer issue, not in Chat: whoever
resumes the task, you after a restart or another agent, starts from the issue.

Every unit of work is tied to an issue: a task, a sub-agent's job, a pull request,
an experiment. The agent that plans a task files its issue and names it in each
agent's brief and each pull request. An agent spawned to help works under the
issue its brief names, or its spawner's, and writes what it did there, or in a
sub-issue that `narrows` it, before it reports.

1. Start: before anything else, find the task's issue: the one your brief names.
   Spawned with none, take your spawner's: `id` of the session `session.self`
   names shows the issue it is `produced_by`. With neither, search Trackinizer
   for issues related to the task:
   `search "..." --kind issue`, and titles. Read each related issue's description,
   its edges, and the sessions that produced it. Work under the task's issue when
   one fits, or file a sub-issue that `narrows` it for a part of your own; file a
   new one only when none fits, linked under the issue it serves (`narrows`).
2. Plan: once the person agrees a plan, write it into the issue's description:
   the goal, the steps, and how each is checked.
3. Work: link each result's evidence to the issue as it lands, and each pull
   request you open, as a code change whose description is its link:
   `issue M produced codechange sha to FULL_HEAD_SHA title to "PR #7: TITLE"
   description to PR_URL`. Its body names the issue, as `Issue#M` (`pr`).
4. Checkpoints: mark finished steps done in the description, and add the latest
   status (`report-status`).
5. Open questions: write each one you ask the person, or cannot settle, under
   "Open questions" in the description. When the person answers, write their answer,
   in their words, under it, and what it changed.
6. Finish: write what was verified into the issue's `validation`. Close it only
   when the person asks, or when closing is plainly the task.

## Say who wrote it

Every issue you file or take on names the agent and session behind it, so a
reader can find them. Your `KnowOp` tool's `session.self` gives your `LABEL#UUID`;
`id UUID` gives your session's row, `AgentSession#N`.

- `owner`: your `LABEL#UUID`.
- Edge: `issue M produced_by agentsession N`.
- Description: a first line `Written by LABEL#UUID (AgentSession#N) for PERSON`.

## Another agent on it

An issue's `owner` and its `produced_by` sessions name the agents on it. When one
is still working (its session `active`, its last record, in `agentsession N
records`, under an hour old), message it with `AgentSend` (to `LABEL#UUID`, or
`LABEL#N`) to coordinate, or to ask what the issue leaves out, before you change
what it works on. Idle an hour or more, or ended: send nothing, and work from the
issue and its session's records.

## Writing

Write with your `KnowOp` tool's `trax.write` action (the `trax` skill gives its
commands), or with `trax` in your own shell when you have one. Keeping the task's
issue current is part of the task: do it unasked, and say what you changed, citing
each row.
