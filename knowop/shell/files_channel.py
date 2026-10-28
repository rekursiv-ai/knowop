"""Answer the file requests a ``files`` terminal carries: describe a path, read a file.

The owner's browser browses this machine's files over a terminal of kind ``files``
(``wire_terminals``), relayed like a shell or sent direct. It writes requests as the
terminal's input and reads the replies from its output: UTF-8 JSON, one object per
line. Requests are answered one at a time, in the order they came, as the user who
started ``knowop connect``, but for a search, which runs beside them until a newer
search stops it. Nothing is written, so a reply lost on the way is safe to ask for
again.

A request is ``{"id": n, "op": "info" | "read" | "search", "path": p}``, where
``n >= 1`` and ``p`` is absolute or starts with ``~``; a search adds ``query``,
``case``, ``word``, ``regex``, ``include`` and ``exclude`` (``files_search.py``). The
replies:

- ``info``: :class:`Info`, the path's kind, size, mtime and ``tag``; a folder's also
  lists its entries, folders first, up to :data:`MAX_ENTRIES`.
- ``read``: :class:`Read`, the file's text, or none when it is binary or larger than
  :data:`MAX_READ_BYTES`.
- ``search``: :class:`Found`, the matching lines under ``p`` by file.
- Any failure: :class:`Failure`, with ``id`` 0 when the line named none.

A ``tag`` changes whenever the size or mtime does, so a later write may say which
version it replaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import asyncio
import os
import stat

from pydantic import BaseModel, ConfigDict, ValidationError

from knowop.shell.files_search import Search, Searched, SearchError


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


__all__ = [
    "MAX_ENTRIES",
    "MAX_READ_BYTES",
    "Entry",
    "Failure",
    "FilesChannel",
    "Found",
    "Info",
    "Read",
    "answer",
]

MAX_ENTRIES: Final = 2_000
"""The most entries a folder's ``info`` lists; ``truncated`` says when it has more."""

MAX_READ_BYTES: Final = 2 * 2**20
"""The largest file ``read`` sends; a larger one is ``too_large``, with no text."""

type Kind = Literal["dir", "file", "other"]
type Code = Literal[
    "not_found",
    "denied",
    "not_a_dir",
    "not_a_file",
    "bad_request",
    "io",
]


class Entry(BaseModel):
    """One entry of a folder; a symlink's kind is its target's."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: Kind
    link: bool
    size: int
    mtime_ms: int


class Info(BaseModel):
    """What a path is; a folder's entries too."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int
    path: str
    kind: Kind
    size: int
    mtime_ms: int
    tag: str
    entries: list[Entry] | None = None
    truncated: bool | None = None


class Read(BaseModel):
    """A file's text, or why it has none."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int
    path: str
    size: int
    mtime_ms: int
    tag: str
    text: str | None
    binary: bool
    too_large: bool


class Failure(BaseModel):
    """A request that could not be answered."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int
    error: Code
    message: str


class Found(Searched):
    """A search's matching lines under ``path``, by file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int
    path: str


class FilesChannel:
    """The ``Shell`` a ``files`` terminal's router writes input to.

    Input is split into lines here, and :meth:`serve` answers them.
    """

    def __init__(self) -> None:
        self._partial = bytearray()
        self._lines: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._sending = asyncio.Lock()
        self._search: Search | None = None
        self._searching: set[asyncio.Task[None]] = set()

    async def write(self, data: bytes) -> bool:
        """Take input; each complete line is a request for :meth:`serve`.

        Args:
          data: The input, which may end mid-line.

        Returns:
          taken: Always true: no request is refused here.

        """
        self._partial.extend(data)
        *lines, rest = self._partial.split(b"\n")
        for line in lines:
            if line.strip():
                self._lines.put_nowait(bytes(line))
        self._partial = bytearray(rest)
        # The server bounds a frame, not a line: a line that never ends is refused once,
        # rather than held.
        if len(self._partial) > 65_536:
            self._partial.clear()
            self._lines.put_nowait(None)
        return True

    def set_winsize(self, rows: int, cols: int) -> None:
        """Ignore a resize: there is no pty."""
        del rows, cols

    async def serve(self, send: Callable[[bytes], Awaitable[None]]) -> None:
        """Answer each request in order, and each search beside them, until cancelled.

        Args:
          send: Sends output to the browser, by whichever route it takes.

        """
        try:
            while True:
                request = _parsed(await self._lines.get())
                if isinstance(request, _Request) and request.op == "search":
                    self._start_search(request, send)
                    continue
                if isinstance(request, _Request):
                    request = await asyncio.to_thread(_answer, request)
                await self._reply(request, send)
        finally:
            if self._search is not None:
                self._search.stop()

    def _start_search(
        self,
        request: _Request,
        send: Callable[[bytes], Awaitable[None]],
    ) -> None:
        """Stop the search running, if any, and start ``request``'s beside the others."""
        if self._search is not None:
            self._search.stop()
        root = _absolute(request.path) or request.path
        self._search = Search(
            root=root,
            query=request.query,
            case=request.case,
            word=request.word,
            regex=request.regex,
            include=request.include,
            exclude=request.exclude,
        )
        task = asyncio.create_task(self._searched(request, self._search, send))
        # The loop holds a task weakly; this set keeps a running search alive.
        self._searching.add(task)
        task.add_done_callback(self._searching.discard)

    async def _searched(
        self,
        request: _Request,
        search: Search,
        send: Callable[[bytes], Awaitable[None]],
    ) -> None:
        """Run one search and send what it found."""
        await self._reply(await asyncio.to_thread(_search, request, search), send)

    async def _reply(
        self,
        reply: Info | Read | Found | Failure,
        send: Callable[[bytes], Awaitable[None]],
    ) -> None:
        """Send one reply as one line; a search's and another's never interleave."""
        data = reply.model_dump_json(exclude_none=True).encode() + b"\n"
        # The router keeps the last ring's worth of output to resume from, and a chunk
        # longer than that would lose its own start.
        step = 65_536
        async with self._sending:
            for at in range(0, len(data), step):
                await send(data[at : at + step])


def answer(line: bytes | None) -> Info | Read | Failure:
    """Answer one request line other than a search.

    Args:
      line: The request, or ``None`` for one too long to hold.

    Returns:
      reply: What the request asked for, or why it could not be answered.

    """
    request = _parsed(line)
    return request if isinstance(request, Failure) else _answer(request)


def _parsed(line: bytes | None) -> _Request | Failure:
    """Read one request line, or say why it is none."""
    if line is None:
        return Failure(id=0, error="bad_request", message="request line too long")
    try:
        return _Request.model_validate_json(line)
    except ValidationError:
        return Failure(id=0, error="bad_request", message="not a request")


def _answer(request: _Request) -> Info | Read | Failure:
    """Answer an ``info`` or ``read``; any other op is a bad request."""
    path = _absolute(request.path)
    if path is None:
        return Failure(
            id=request.id,
            error="bad_request",
            message="a path is absolute or starts with ~",
        )
    try:
        if request.op == "info":
            return _info(request.id, path)
        if request.op == "read":
            return _read(request.id, path)
    except OSError as error:
        return Failure(id=request.id, error=_code(error), message=str(error))
    except ValueError as error:
        # A NUL or a lone surrogate, which no filesystem path holds.
        return Failure(id=request.id, error="bad_request", message=str(error))
    return Failure(id=request.id, error="bad_request", message=f"no op {request.op}")


class _Request(BaseModel):
    """One request line; an unknown ``op`` is answered, not refused."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int
    op: str
    path: str
    query: str = ""
    case: bool = False
    word: bool = False
    regex: bool = False
    include: str = ""
    exclude: str = ""


def _search(request: _Request, search: Search) -> Found | Failure:
    """Search the folder ``request`` names; an empty query finds nothing."""
    if not Path(search.root).is_absolute():
        return Failure(
            id=request.id,
            error="bad_request",
            message="a path is absolute or starts with ~",
        )
    try:
        if not Path(search.root).is_dir():
            # The stat names a missing folder; ``is_dir`` alone cannot.
            _ = Path(search.root).stat()
            return Failure(
                id=request.id,
                error="not_a_dir",
                message=f"{search.root} is not a folder",
            )
        found = (
            search.run()
            if request.query
            else Searched(engine="rg", matches=0, truncated=False, files=[])
        )
    except OSError as error:
        return Failure(id=request.id, error=_code(error), message=str(error))
    except SearchError as error:
        return Failure(id=request.id, error="bad_request", message=str(error))
    return Found(
        id=request.id,
        path=search.root,
        engine=found.engine,
        matches=found.matches,
        truncated=found.truncated,
        files=found.files,
    )


def _absolute(path: str) -> str | None:
    """Return ``path`` normalized, with ``~`` expanded; ``None`` when it is relative."""
    try:
        expanded = Path(path).expanduser()
    except RuntimeError:
        # ``~name`` names a user this machine does not have: no home to resolve against.
        return None
    return os.path.normpath(expanded) if expanded.is_absolute() else None


def _info(request_id: int, path: str) -> Info:
    """Describe ``path``, following a symlink; list it when it is a folder."""
    status = Path(path).stat()
    kind = _kind(status.st_mode)
    entries, truncated = _entries(path) if kind == "dir" else (None, None)
    return Info(
        id=request_id,
        path=path,
        kind=kind,
        size=status.st_size,
        mtime_ms=status.st_mtime_ns // 1_000_000,
        tag=_tag(status),
        entries=entries,
        truncated=truncated,
    )


def _entries(folder: str) -> tuple[list[Entry], bool]:
    """Return a folder's first entries, folders first, and whether more exist."""
    with os.scandir(folder) as scan:
        found = sorted(scan, key=lambda each: (not _is_dir(each), each.name.casefold()))
    # Only the kept entries are stat'ed: a folder of a million files lists in the time
    # its names take to sort.
    return [_entry(each) for each in found[:MAX_ENTRIES]], len(found) > MAX_ENTRIES


def _is_dir(entry: os.DirEntry[str]) -> bool:
    """Return whether an entry is a folder or a link to one."""
    try:
        return entry.is_dir()
    except OSError:
        return False


def _entry(entry: os.DirEntry[str]) -> Entry:
    """Describe one entry; a broken link is ``other``."""
    # A name that is not UTF-8 cannot travel as JSON text; it shows with U+FFFD.
    name = os.fsencode(entry.name).decode(errors="replace")
    try:
        status = entry.stat()
    except OSError:
        return Entry(
            name=name,
            kind="other",
            link=entry.is_symlink(),
            size=0,
            mtime_ms=0,
        )
    return Entry(
        name=name,
        kind=_kind(status.st_mode),
        link=entry.is_symlink(),
        size=status.st_size,
        mtime_ms=status.st_mtime_ns // 1_000_000,
    )


def _read(request_id: int, path: str) -> Read | Failure:
    """Read a regular file's text, unless it is binary or too large."""
    # Non-blocking, so a FIFO put where a file was opens without waiting for a writer;
    # the check below then refuses it.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    status = os.fstat(fd)
    if not stat.S_ISREG(status.st_mode):
        os.close(fd)
        return Failure(
            id=request_id,
            error="not_a_file",
            message=f"{path} is not a file",
        )
    with os.fdopen(fd, "rb") as file:
        data = file.read(MAX_READ_BYTES + 1)
    too_large = len(data) > MAX_READ_BYTES
    # A NUL byte in the first 8 KiB marks a binary file, as git decides.
    binary = b"\x00" in data[:8_192]
    return Read(
        id=request_id,
        path=path,
        size=status.st_size,
        mtime_ms=status.st_mtime_ns // 1_000_000,
        tag=_tag(status),
        text=None if too_large or binary else data.decode(errors="replace"),
        binary=binary,
        too_large=too_large,
    )


def _kind(mode: int) -> Kind:
    """Return a mode's kind."""
    if stat.S_ISDIR(mode):
        return "dir"
    return "file" if stat.S_ISREG(mode) else "other"


def _tag(status: os.stat_result) -> str:
    """Return a version tag that changes with the size or mtime."""
    return f"{status.st_size}:{status.st_mtime_ns}"


def _code(error: OSError) -> Code:
    """Return the failure code of a filesystem error."""
    if isinstance(error, FileNotFoundError):
        return "not_found"
    if isinstance(error, PermissionError):
        return "denied"
    if isinstance(error, NotADirectoryError):
        return "not_a_dir"
    return "io"
