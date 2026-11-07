"""Tests for the MCP server a run offers its agent's CLI its tools on.

The real HTTP handler reads request bytes from an in-memory socket. Listening,
thread shutdown and a disconnected peer are checked by the hand-run parity script.
"""

from __future__ import annotations

from functools import partial
from http.client import HTTPResponse
from io import BytesIO
from typing import TYPE_CHECKING, NotRequired, TypedDict
from unittest.mock import Mock
from urllib.parse import urlsplit

import json
import socket
import threading

from treekle import PlainTree, json_loads

import pytest

from knowop.agent import tool_server
from knowop.agent.conftest import _SCHEMA, _break, _echo, _refuse
from knowop.agent.tool_server import (
    PROTOCOL,
    AgentTool,
    ToolServer,
)


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


class _Error(TypedDict):
    """A JSON-RPC failure with its diagnostic."""

    code: int
    message: str


class _Response(TypedDict):
    """The JSON-RPC envelope returned by the tool server."""

    jsonrpc: str
    id: int | str | None
    result: NotRequired[dict[str, PlainTree]]
    error: NotRequired[_Error]


@pytest.fixture(autouse=True)
def no_listener(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the real HTTP server's constructor off the listening socket boundary."""
    monkeypatch.setattr(tool_server._Http, "server_bind", Mock())
    monkeypatch.setattr(tool_server._Http, "server_activate", Mock())


@pytest.fixture
def server() -> Iterator[ToolServer]:
    served = ToolServer(
        [
            AgentTool(name="echo", description="Say it.", schema=_SCHEMA, call=_echo),
            AgentTool(name="refuse", description="No.", schema=_SCHEMA, call=_refuse),
            AgentTool(name="break", description="Fail.", schema=_SCHEMA, call=_break),
        ],
    )
    yield served
    served.close()


def _post(
    server: ToolServer,
    body: object,
    *,
    path: str = "/mcp",
    headers: Mapping[str, str] | None = None,
) -> tuple[int, _Response | None]:
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
    return status, json_loads(data, _Response) if data else None


def _send(
    server: ToolServer,
    method: str,
    *,
    path: str = "/mcp",
    body: bytes = b"",
    headers: Mapping[str, str] | None = None,
    content_length: bool = True,
) -> tuple[int, bytes, str | None]:
    """Pass wire bytes through the real handler and parse its HTTP response."""
    fields = {"Host": "127.0.0.1", "Authorization": f"Bearer {server.token}"}
    if content_length:
        fields["Content-Length"] = str(len(body))
    fields.update(headers or {})
    raw = (
        f"{method} {path} HTTP/1.1\r\n"
        + "".join(f"{key}: {value}\r\n" for key, value in fields.items())
        + "\r\n"
    ).encode("latin-1") + body
    peer = Mock(
        spec=socket.socket,
        makefile=Mock(return_value=BytesIO(_exchange(server, raw))),
    )
    with HTTPResponse(peer) as response:
        response.begin()
        return response.status, response.read(), response.getheader("Allow")


def _exchange(server: ToolServer, raw: bytes, *, headers_only: bool = False) -> bytes:
    """Run one handler against a socket's input and collect exactly its output."""
    sent = bytearray()
    peer = Mock(
        spec=socket.socket,
        makefile=Mock(return_value=BytesIO(raw)),
        sendall=Mock(
            side_effect=partial(_write_response, sent, headers_only=headers_only),
        ),
    )
    server._http.finish_request(peer, ("127.0.0.1", 1))
    return bytes(sent)


def _write_response(sent: bytearray, data: bytes, *, headers_only: bool) -> None:
    if headers_only and b"\r\n\r\n" in sent:
        raise BrokenPipeError(32, "Broken pipe")
    sent.extend(data)


def _rpc(server: ToolServer, method: str, params: object = None) -> _Response:
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
    assert "result" in answer
    assert answer["result"] == {
        "protocolVersion": "2025-03-26",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "knowop", "version": "1"},
    }
    unnamed = _rpc(server, "initialize")
    assert "result" in unnamed
    assert unnamed["result"]["protocolVersion"] == PROTOCOL
    initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    assert _post(server, initialized) == (202, None)


def test_the_tools_are_listed_with_their_schemas(server: ToolServer) -> None:
    listed = _rpc(server, "tools/list")
    assert "result" in listed
    assert listed["result"] == {
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
    assert "result" in answer
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
    assert "error" in answer
    assert answer["error"]["code"] == code
    assert answer["id"] == 1


def test_ping_and_bad_json(server: ToolServer) -> None:
    answer = _rpc(server, "ping")
    assert "result" in answer
    assert answer["result"] == {}
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
    assert "error" in answer
    message = answer["error"]["message"]
    assert message not in ("", "None")
    assert answer == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": -32_700, "message": message},
    }


def test_a_request_without_content_length_reads_no_body(server: ToolServer) -> None:
    status, body, _ = _send(server, "POST", body=b" ", content_length=False)
    assert status == 400
    assert json_loads(body, _Response) == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {
            "code": -32_700,
            "message": "Expecting value: line 1 column 1 (char 0)",
        },
    }


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
    assert server._http.server_address == ("127.0.0.1", 0)
    other = ToolServer([])
    other.close()
    assert other.token != server.token


def test_a_stream_is_refused(server: ToolServer) -> None:
    assert _send(server, "GET") == (405, b"", "POST")


def test_an_empty_answer_ends_with_its_headers(server: ToolServer) -> None:
    sent = _exchange(
        server,
        b"GET /mcp HTTP/1.1\r\nHost: x\r\n\r\n",
        headers_only=True,
    )
    assert sent.startswith(b"HTTP/1.0 401 Unauthorized\r\n")


def test_a_request_writes_nothing_to_the_persons_terminal(
    server: ToolServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ = _rpc(server, "ping")
    assert _send(server, "POST", path="/mcpx")[0] == 404
    assert capsys.readouterr().err == ""


def test_a_server_never_started_closes() -> None:
    ToolServer([]).close()


def test_close_stops_the_server_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    server = ToolServer([])
    start = Mock()
    worker = Mock(spec=threading.Thread, start=start)
    spawn = Mock(return_value=worker)
    shutdown = Mock()
    monkeypatch.setattr(threading, "Thread", spawn)
    monkeypatch.setattr(server._http, "shutdown", shutdown)

    server.start()
    spawn.assert_called_once_with(
        name="agent tools",
        target=server._http.serve_forever,
        kwargs={"poll_interval": 0.01},
        daemon=True,
    )
    start.assert_called_once_with()
    assert server._serving is worker

    server.close()

    shutdown.assert_called_once_with()
    assert server._http.socket.fileno() == -1


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
