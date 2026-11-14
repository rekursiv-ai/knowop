"""Tools an agent is given when it starts: plain functions over Trackinizer's client.

Each tool is a noun with an ``action`` that names its verb, as sagent's own are:
``Canvas`` (``canvas.CanvasTools.act``) is the agent's hands on the canvas its
session's Chat is on. A tool is a typed async function whose signature is what a model
passes, with a description in ``assets/`` that the model reads; none names an agent
framework, so an adapter makes each a tool of any (the agent process makes them
sagent's, ``process/main.py``).
"""

from __future__ import annotations
