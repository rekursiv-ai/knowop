"""Tools an agent is given when it starts: plain functions over Trackinizer's client.

``Canvas`` (``canvas.CanvasTools.act``) is the agent's hands on the canvas its
session's Chat is on, and ``KnowOp`` (``knowop.KnowOpTools.act``) its hosts, the
agent sessions it launches and Trackinizer's records; each is a noun with an
``action`` that names its verb, as sagent's own are. KnowOp's ``trax.read`` and
``trax.write`` actions (``records.Records``) each take a whole trax command, since
trax's grammar is already how agents read and write records and an action per command
would restate a part of it.

A tool is a typed async function whose signature is what a model passes, with a
description in ``assets/`` that the model reads; none names an agent framework, so an
adapter makes each a tool of any (the agent process makes them sagent's,
``process/main.py``).
"""

from __future__ import annotations
