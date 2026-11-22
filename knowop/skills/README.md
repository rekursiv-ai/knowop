# Agent instructions

knowop's own skills. An agent `knowop run` starts on a person's host loads the skills
of the folder it starts in, as any sagent does, so it finds these where that folder
links them.

| Skill | Folder | What it does |
|---|---|---|
| `knowop-run` | `agent/` | `knowop run`: start an agent here, headless, on a host or from another agent |
| `knowop-job` | `jobs/` | `knowop job`: run a job on a host |
| `knowop-connect` | `shell/` | `knowop connect`: connect this computer as a host |
| `pr` | `pr/` | A pull request's routine, and `knowop watch-pr` |
| `agent-tasks` | `agent-tasks/` | Keeps a task's state in its Trackinizer issue |
| `report-status` | `report-status/` | Tells the person where the work stands at each checkpoint |

`pr`, `agent-tasks` and `report-status` are also the skills of KnowOp's agents on
its cloud, which link to them here. This folder is exported on its own, so nothing
here links outside it.

A skill documents what its command does, in the words of its `--help`. When a
command changes, its skill changes in the same pull request.
