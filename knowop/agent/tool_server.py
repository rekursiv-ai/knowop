"""Serve a run's own tools to the agent it runs, over MCP.

claude and codex take a tool from outside themselves over MCP alone, so the run serves
its tools at ``http://127.0.0.1:PORT/mcp`` in MCP's Streamable HTTP: each POST is one
JSON-RPC message, and each request is answered with one JSON body, as the transport
lets a server that streams nothing do.

The key is a bearer token, new each run, which only the CLI the run starts is given,
in its environment as :data:`BEARER_ENV`: never on a command line, which any user of
the computer may read with ``ps``. A request without it is refused, and so is one that
carries an ``Origin``, as the transport asks of a local server: a web page may post to
``127.0.0.1`` too.

It holds until sagent serves MCP to a CLI's agent itself, as it already serves
sagent's own tools to claude.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Final, NotRequired, TypedDict, override

import json
import logging
import secrets
import threading

from treekle import ReadError, from_plain, json_loads


if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = ["BEARER_ENV", "PROTOCOL", "AgentTool", "ToolError", "ToolServer"]

_logger = logging.getLogger(__name__)

PROTOCOL: Final = "2025-06-18"
"""The MCP revision a server answers with when the client names none."""

BEARER_ENV: Final = "KNOWOP_TOOLS_TOKEN"
"""The environment variable that hands the CLI the server's bearer token."""

_BAD_PARAMS: Final = -32_602


class ToolError(Exception):
    """A call a tool refused; its message is what the agent reads back."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentTool:
    """One tool the agent may call.

    Attributes:
      name: What the agent calls it by.
      description: What the agent reads about when and how to call it.
      schema: The JSON Schema of its arguments.
      call: Runs it on the arguments the agent passed; returns what the agent reads,
        or raises :class:`ToolError` with what went wrong.

    """

    name: str
    description: str
    schema: Mapping[str, object]
    call: Callable[[Mapping[str, object]], str]


class ToolServer:
    """``tools`` at a URL of their own, served on a thread of its own.

    Args:
      tools: What the agent may call.

    """

    def __init__(self, tools: Sequence[AgentTool]) -> None:
        """Bind a free port on ``127.0.0.1``; nothing is served until :meth:`start`."""
        self._http = _Http(tools)
        self._serving: threading.Thread | None = None

    @property
    def url(self) -> str:
        """Return where the CLI reaches the tools; it holds no secret."""
        host, port = self._http.server_address[:2]
        return f"http://{host!s}:{port}/mcp"

    @property
    def token(self) -> str:
        """Return the bearer token a request must carry, for the CLI's environment."""
        return self._http.token

    def start(self) -> None:
        """Serve until :meth:`close`, on a thread the process's exit ends."""
        self._serving = threading.Thread(
            name="agent tools",
            target=self._http.serve_forever,
            daemon=True,
        )
        self._serving.start()

    def close(self) -> None:
        """Stop serving and free the port."""
        # ``shutdown`` waits on a ``serve_forever`` that never began.
        if self._serving is not None:
            self._http.shutdown()
        self._http.server_close()


class _Http(ThreadingHTTPServer):
    """The listener, with what its handlers answer from."""

    daemon_threads = True

    def __init__(self, tools: Sequence[AgentTool]) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.tools = {tool.name: tool for tool in tools}
        self.token = secrets.token_hex(16)


class _Handler(BaseHTTPRequestHandler):
    """One request: a JSON-RPC message posted to ``/mcp`` with the bearer token."""

    def do_POST(self) -> None:
        """Answer one JSON-RPC message."""
        http = self._admitted()
        if http is None:
            return
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            message = json_loads(body, _Message)
        except ValueError as error:
            self._reply(HTTPStatus.BAD_REQUEST, _error(None, -32_700, str(error)))
            return
        if "id" not in message or "method" not in message:
            # A notification, or the client's answer to a request this never makes.
            self._reply(HTTPStatus.ACCEPTED, None)
            return
        self._reply(HTTPStatus.OK, _answer(http, message))

    def do_GET(self) -> None:
        """Refuse the stream a client may open: this server sends nothing unasked."""
        if self._admitted() is not None:
            self._reply(HTTPStatus.METHOD_NOT_ALLOWED, None)

    @override
    def log_message(self, format: str, *args: object) -> None:
        # ``http.server`` writes each request to stderr, which is the person's terminal
        # under a CLI the run hosts.
        del format, args

    def _admitted(self) -> _Http | None:
        """Return the server when the request may be answered; else refuse it."""
        http = self.server
        assert isinstance(http, _Http)
        if "Origin" in self.headers:
            self._reply(HTTPStatus.FORBIDDEN, None)
            return None
        if self.path != "/mcp":
            self._reply(HTTPStatus.NOT_FOUND, None)
            return None
        # As bytes: ``compare_digest`` refuses a non-ASCII str, and http.server reads a
        # header as Latin-1.
        given = self.headers.get("Authorization", "").encode("latin-1")
        if not secrets.compare_digest(given, f"Bearer {http.token}".encode()):
            self._reply(HTTPStatus.UNAUTHORIZED, None)
            return None
        return http

    def _reply(self, status: HTTPStatus, body: Mapping[str, object] | None) -> None:
        """Send ``body`` as JSON with ``status``; no body for ``None``."""
        # Plain JSON, as an MCP client reads it: treekle would tag a schema two tools
        # share as a back-reference, which no client resolves.
        data = b"" if body is None else json.dumps(body).encode()
        self.send_response(status)
        if status == HTTPStatus.METHOD_NOT_ALLOWED:
            self.send_header("Allow", "POST")
        if body is not None:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        _ = self.wfile.write(data)


class _Message(TypedDict):
    id: NotRequired[int | str]
    method: NotRequired[str]
    params: NotRequired[dict[str, object]]


class _Initialize(TypedDict):
    protocolVersion: NotRequired[str]


class _Call(TypedDict):
    name: str
    arguments: NotRequired[dict[str, object]]


def _answer(http: _Http, message: _Message) -> dict[str, object]:
    """Return the JSON-RPC response to a request."""
    ident, params = message.get("id"), message.get("params")
    match message.get("method"):
        case "initialize":
            asked = from_plain(params, _Initialize, default=_Initialize())
            result: dict[str, object] = {
                "protocolVersion": asked.get("protocolVersion") or PROTOCOL,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "knowop", "version": "1"},
            }
        case "ping":
            result = {}
        case "tools/list":
            result = {
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": dict(tool.schema),
                    }
                    for tool in http.tools.values()
                ],
            }
        case "tools/call":
            try:
                call = from_plain(params, _Call)
            except ReadError as error:
                return _error(ident, _BAD_PARAMS, str(error))
            tool = http.tools.get(call["name"])
            if tool is None:
                return _error(ident, _BAD_PARAMS, f"Unknown tool: {call['name']}")
            result = _called(tool, call.get("arguments", {}))
        case method:
            return _error(ident, -32_601, f"Method not found: {method}")
    return {"jsonrpc": "2.0", "id": ident, "result": result}


def _called(tool: AgentTool, arguments: Mapping[str, object]) -> dict[str, object]:
    """Run ``tool``; a refusal or a failure is a result the agent reads, as MCP says."""
    try:
        text, failed = tool.call(arguments), False
    except ToolError as error:
        text, failed = str(error), True
    except Exception as error:
        _logger.warning("knowop run: tool %s failed", tool.name, exc_info=True)
        text, failed = f"The run failed at {tool.name}: {error}", True
    return {"content": [{"type": "text", "text": text}], "isError": failed}


def _error(ident: int | str | None, code: int, text: str) -> dict[str, object]:
    """Return a JSON-RPC error response."""
    return {"jsonrpc": "2.0", "id": ident, "error": {"code": code, "message": text}}
