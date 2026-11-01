"""Tests for the MCP server a run offers its agent's CLI its tools on.

Each test serves on ``127.0.0.1`` and posts JSON-RPC as claude and codex do.
"""

from __future__ import annotations

from http.client import HTTPConnection
from typing import TYPE_CHECKING, override
from urllib.parse import urlsplit

import json
import socket
import time

from treekle import json_loads

import pytest

from knowop.agent.tool_server import (
    PROTOCOL,
    AgentTool,
    ToolError,
    ToolServer,
)


if TYPE_CHECKING:
    from collections.abc import Buffer, Iterator, Mapping


def _echo(arguments: Mapping[str, object]) -> str:
    return f"said {arguments['text']}"


def _refuse(arguments: Mapping[str, object]) -> str:
    del arguments
    raise ToolError("not that one")


def _break(arguments: Mapping[str, object]) -> str:
    del arguments
    raise RuntimeError("boom")


_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}}


@pytest.fixture
def server() -> Iterator[ToolServer]:
    served = ToolServer(
        [
            AgentTool(name="echo", description="Say it.", schema=_SCHEMA, call=_echo),
            AgentTool(name="refuse", description="No.", schema=_SCHEMA, call=_refuse),
            AgentTool(name="break", description="Fail.", schema=_SCHEMA, call=_break),
        ],
    )
    served.start()
    yield served
    served.close()


def _post(
    server: ToolServer,
    body: object,
    *,
    path: str = "/mcp",
    headers: Mapping[str, str] | None = None,
) -> tuple[int, dict[str, object] | None]:
    """Post ``body`` as a CLI does; return the status and the JSON answer, if any."""
    status, data, _ = _send(
        server,
        "POST",
        path=path,
        body=body if isinstance(body, bytes) else json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **(headers or {}),
        },
    )
    return status, json_loads(data, dict[str, object]) if data else None


def _send(
    server: ToolServer,
    method: str,
    *,
    path: str = "/mcp",
    body: bytes = b"",
    headers: Mapping[str, str] | None = None,
) -> tuple[int, bytes, str | None]:
    """Send one request with the token, unless ``headers`` replace it."""
    parts = urlsplit(server.url)
    connection = HTTPConnection(parts.hostname or "", parts.port, timeout=5)
    try:
        connection.request(
            method,
            path,
            body=body,
            headers={"Authorization": f"Bearer {server.token}", **(headers or {})},
        )
        response = connection.getresponse()
        return response.status, response.read(), response.getheader("Allow")
    finally:
        connection.close()


def _rpc(server: ToolServer, method: str, params: object = None) -> dict[str, object]:
    status, answer = _post(
        server,
        {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}},
    )
    assert status == 200
    assert answer is not None
    return answer


def test_initialize_offers_tools_at_the_version_the_client_asked(
    server: ToolServer,
) -> None:
    answer = _rpc(server, "initialize", {"protocolVersion": "2025-03-26"})
    assert answer["result"] == {
        "protocolVersion": "2025-03-26",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "knowop", "version": "1"},
    }
    unnamed = _rpc(server, "initialize")["result"]
    assert isinstance(unnamed, dict)
    assert unnamed["protocolVersion"] == PROTOCOL
    initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    assert _post(server, initialized) == (202, None)


def test_the_tools_are_listed_with_their_schemas(server: ToolServer) -> None:
    listed = _rpc(server, "tools/list")["result"]
    assert listed == {
        "tools": [
            {"name": "echo", "description": "Say it.", "inputSchema": _SCHEMA},
            {"name": "refuse", "description": "No.", "inputSchema": _SCHEMA},
            {"name": "break", "description": "Fail.", "inputSchema": _SCHEMA},
        ],
    }


@pytest.mark.parametrize(
    ("name", "text", "failed"),
    [
        ("echo", "said hi", False),
        ("refuse", "not that one", True),
        ("break", "The run failed at break: boom", True),
    ],
)
def test_a_call_answers_its_text_and_a_failure_is_a_result(
    server: ToolServer,
    name: str,
    text: str,
    failed: bool,
) -> None:
    answer = _rpc(server, "tools/call", {"name": name, "arguments": {"text": "hi"}})
    assert answer["result"] == {
        "content": [{"type": "text", "text": text}],
        "isError": failed,
    }


@pytest.mark.parametrize(
    ("method", "params", "code"),
    [
        ("tools/call", {"name": "nope"}, -32_602),
        ("tools/call", {"arguments": {}}, -32_602),
        ("resources/list", {}, -32_601),
    ],
)
def test_an_unknown_tool_or_method_is_a_json_rpc_error(
    server: ToolServer,
    method: str,
    params: dict[str, object],
    code: int,
) -> None:
    answer = _rpc(server, method, params)
    error = answer["error"]
    assert isinstance(error, dict)
    assert error["code"] == code
    assert answer["id"] == 1


def test_ping_and_bad_json(server: ToolServer) -> None:
    assert _rpc(server, "ping")["result"] == {}
    assert _post(server, b"{not json") == (
        400,
        {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32_700,
                "message": "Expecting property name enclosed in double quotes: "
                "line 1 column 2 (char 1)",
            },
        },
    )


@pytest.mark.parametrize(
    "body",
    [
        b"[]",
        b"null",
        b"1",
        b'"ping"',
        b'{"id": [], "method": "ping"}',
        b'{"id": 1, "method": 7}',
        b'{"id": 1, "method": "ping", "params": []}',
    ],
)
def test_a_message_with_the_wrong_shape_is_a_parse_error(
    server: ToolServer,
    body: bytes,
) -> None:
    status, answer = _post(server, body)
    assert status == 400
    assert answer is not None
    error = answer["error"]
    assert isinstance(error, dict)
    message = error["message"]
    assert isinstance(message, str)
    assert message not in ("", "None")
    assert answer == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": -32_700, "message": message},
    }


def test_a_request_without_content_length_reads_no_body(server: ToolServer) -> None:
    parts = urlsplit(server.url)
    connection = HTTPConnection(parts.hostname or "", parts.port, timeout=1.0)
    try:
        connection.putrequest("POST", "/mcp")
        connection.putheader("Authorization", f"Bearer {server.token}")
        connection.endheaders()
        connection.send(b" ")
        response = connection.getresponse()
        assert response.status == 400
        assert json_loads(response.read(), dict[str, object]) == {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32_700,
                "message": "Expecting value: line 1 column 1 (char 0)",
            },
        }
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("path", "headers", "status"),
    [
        ("/mcp", {}, 200),
        ("/wrong", {}, 404),
        ("/mcp", {"Origin": "https://evil.test"}, 403),
        ("/mcp", {"Authorization": ""}, 401),
        ("/mcp", {"Authorization": "Bearer wrong"}, 401),
        # Latin-1 on the wire, as http.server reads a header.
        ("/mcp", {"Authorization": "Bearer \u00e9"}, 401),
    ],
)
def test_only_the_token_and_no_web_page_reach_the_tools(
    server: ToolServer,
    path: str,
    headers: dict[str, str],
    status: int,
) -> None:
    ping = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    assert _post(server, ping, path=path, headers=headers)[0] == status
    assert _send(server, "GET", path=path, headers=headers)[0] == (
        405 if status == 200 else status
    )


def test_the_url_holds_no_secret(server: ToolServer) -> None:
    assert len(server.token) == 32
    assert server.token not in server.url
    assert urlsplit(server.url).path == "/mcp"
    other = ToolServer([])
    other.close()
    assert other.token != server.token


def test_a_stream_is_refused(server: ToolServer) -> None:
    assert _send(server, "GET") == (405, b"", "POST")


class _GoneAfterTheHeaders(socket.socket):
    """A client that closes after response headers, as for an empty body."""

    sent: bytes = b""

    @override
    def sendall(self, data: Buffer, flags: int = 0, /) -> None:
        del flags
        if b"\r\n\r\n" in self.sent:
            raise BrokenPipeError(32, "Broken pipe")
        self.sent += bytes(data)


def test_an_empty_answer_ends_with_its_headers(server: ToolServer) -> None:
    ours, theirs = socket.socketpair()
    with ours, _GoneAfterTheHeaders(fileno=theirs.detach()) as client:
        ours.sendall(b"GET /mcp HTTP/1.1\r\nHost: x\r\n\r\n")
        server._http.finish_request(client, ("127.0.0.1", 1))
    assert client.sent.startswith(b"HTTP/1.0 401 Unauthorized\r\n")


def test_a_request_writes_nothing_to_the_persons_terminal(
    server: ToolServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ = _rpc(server, "ping")
    assert _send(server, "POST", path="/mcpx")[0] == 404
    assert capsys.readouterr().err == ""


def test_a_server_never_started_closes() -> None:
    ToolServer([]).close()


def test_close_returns_before_the_next_server_poll() -> None:
    server = ToolServer([])
    server.start()
    parts = urlsplit(server.url)
    connection = HTTPConnection(parts.hostname or "", parts.port, timeout=1.0)
    try:
        connection.request("GET", "/mcp")
        response = connection.getresponse()
        assert response.status == 401
        _ = response.read()
    finally:
        connection.close()

    time.sleep(0.02)
    started = time.monotonic()
    server.close()
    elapsed = time.monotonic() - started

    thread = server._serving
    assert thread is not None
    thread.join(timeout=1.0)
    assert elapsed < 0.1, f"close waited {elapsed:.3f}s for serve_forever"
    assert not thread.is_alive()


def test_close_stops_the_server_thread() -> None:
    server = ToolServer([])
    server.start()
    thread = server._serving
    assert thread is not None

    server.close()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert server._http.socket.fileno() == -1


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
