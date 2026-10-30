"""Tests for the MCP server a run offers its agent's CLI its tools on.

Each test serves on ``127.0.0.1`` and posts JSON-RPC as claude and codex do.
"""

from __future__ import annotations

from http.client import HTTPConnection
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import json

from treekle import json_loads

import pytest

from knowop.agent.tool_server import (
    PROTOCOL,
    AgentTool,
    ToolError,
    ToolServer,
)


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


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
    status, answer = _post(server, b"{not json")
    assert status == 400
    assert answer is not None
    error = answer["error"]
    assert isinstance(error, dict)
    assert error["code"] == -32_700


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


def test_a_request_writes_nothing_to_the_persons_terminal(
    server: ToolServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ = _rpc(server, "ping")
    assert _send(server, "POST", path="/mcpx")[0] == 404
    assert capsys.readouterr().err == ""


def test_a_server_never_started_closes() -> None:
    ToolServer([]).close()


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
