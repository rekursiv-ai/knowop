Operate on the person's KnowOp server: hosts and campaign lead sessions.

## Host

- `host.list`: list the person's hosts, each KnowHost's running/paused
  state, and whether this server rents KnowHosts.
- `host.start`: rent a KnowHost; answer its name once online. Ask the
  person first: billed per running minute; pauses after 30 idle minutes.
  Sends once, waits up to 150 seconds; never retry an uncertain start.
- `host.pause`: pause KnowHost `name`; charging stops.
- `host.resume`: wake KnowHost `name`; says when memory was lost. Resume
  a paused host before using it through RemoteTool.

## Session

- `session.budget`: the person's campaign limit, in USD: their Each campaign
  budget on the Usage page, else this site's default.
- `session.set_budget`: set the person's campaign limit to `budget` (USD),
  only when the person asks; every other limit stays.
- `session.launch`: start a campaign lead on our cloud, spawned by this
  session. `name` is its label; `time` is required, a Slurm time limit
  (20 = 20 minutes). `model`, `effort`, and `budget` (USD) are optional;
  `budget` defaults to the person's campaign limit (`session.budget`).
  Returns `LABEL#N` once attached; use that name with AgentSend.
- `session.status`: inspect session `name` (`LABEL#N`, required); answer
  its process state (starting, attached, detached) and reason. The server
  permits only sessions spawned below this session.
- `session.stop`: stop the process of session `name` (`LABEL#N`).
- `session.continue`: restart session `name` (`LABEL#N`, required) on the
  SAME session and journal, with a new `time` limit (required). Optional
  `model`, `effort`, and `budget` work as for launch. Returns `LABEL#N`
  once attached. Messaging remains AgentSend to that name.

Server refusals are returned as text. No delete-host action.
