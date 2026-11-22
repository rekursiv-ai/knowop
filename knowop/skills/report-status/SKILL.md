---
name: report-status
description: "ALWAYS invoke this skill at a checkpoint of work that spans many steps: a plan step done, a pull request opened, pushed, failing or merged, a sub-agent's report, a long job's end, or before you stop to wait for the person. Do not make the person ask what is done and what is left."
---

# Report status

At each checkpoint, tell the person where the work stands, then go on.

## Checkpoints

- A step of your plan is done.
- A pull request is opened, pushed, failing, or merged.
- A sub-agent reports, or a long job ends.
- You are about to stop and wait for the person.

## The status

Up to four short lines; leave out a line with nothing in it:

```text
Done: what finished since the last status, with its link or evidence
Next: the step you start now
PRs: each pull request of this work, its link and state (open, ready, failing, merged)
Waiting on: a person's answer, a job or an agent, and since when
```

- Mark the step done in your `Plan` checklist in the same turn.
- List every pull request of the work until it merges or closes, not only the
  newest.
- A task with a Trackinizer issue gets the same status there (`agent-tasks`), so
  whoever resumes the task starts from it.
- Nothing changed since the last status: write none.
