"""Search a folder's files for the explorer, as VS Code's Search view does.

ripgrep runs when it is installed, as sagent's Grep does, streamed: it is stopped once
:data:`MAX_HITS` lines have matched, at the deadline, or when a newer search replaces
it, so a query of one letter over a home folder costs no more than the lines shown.
Without it, ``knowop.lib.files.grep`` searches in Python, as Grep falls back to. Either
skips ``.git``, what ``.gitignore`` names, and binary files, and searches hidden files.

Each matching line comes back with the ranges of its matches as UTF-16 offsets, the
indices a browser's strings use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import re
import shutil
import subprocess
import threading

from pydantic import BaseModel, ConfigDict, ValidationError

from knowop.lib.files.grep import GrepError, Query, grep_rows


if TYPE_CHECKING:
    from collections.abc import Iterable


__all__ = ["MAX_HITS", "FileHits", "Hit", "Search", "SearchError", "Searched"]

MAX_HITS: Final = 2_000
"""The most matching lines a search returns; ``truncated`` says when there were more."""

type Engine = Literal["rg", "python"]


class SearchError(Exception):
    """A search that cannot run, such as an invalid regex; the message says why."""


class Hit(BaseModel):
    """One matching line: its number, its text and where its matches are."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    line: int
    text: str
    ranges: list[tuple[int, int]]


class FileHits(BaseModel):
    """One file's matching lines."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    hits: list[Hit]


class Searched(BaseModel):
    """What a search found, by file in path order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    engine: Engine
    matches: int
    """Every match, two on one line counting as two; ``MAX_HITS`` caps lines."""

    truncated: bool
    files: list[FileHits]


@dataclass(frozen=True, slots=True, kw_only=True)
class Search:
    """One search of the files under ``root``, which :meth:`stop` may end early.

    Attributes:
      root: The folder searched, absolute.
      query: The text, or the regex with ``regex``.
      case: Match case; otherwise case is ignored.
      word: Match whole words only.
      regex: Read ``query`` as a regex; otherwise as plain text.
      include: Comma-separated globs of the files to search; empty searches all.
      exclude: Comma-separated globs of the files to skip.
      deadline_sec: How long ripgrep may run before what it found is returned.

    """

    root: str
    query: str
    case: bool = False
    word: bool = False
    regex: bool = False
    include: str = ""
    exclude: str = ""
    deadline_sec: float = 20.0
    _stopped: threading.Event = field(default_factory=threading.Event, init=False)

    def stop(self) -> None:
        """End the search early: it returns what it found so far, as truncated."""
        self._stopped.set()

    def run(self) -> Searched:
        """Search, blocking; ripgrep when it is on ``PATH``, else Python.

        Returns:
          searched: The matching lines by file, at most :data:`MAX_HITS` of them.

        Raises:
          SearchError: The regex is invalid, or the folder cannot be searched.

        """
        rg = shutil.which("rg")
        if rg is None:
            return self._python()
        return self._rg(rg)

    def _rg(self, rg: str) -> Searched:
        """Stream ripgrep's matches until the cap, the deadline or a stop."""
        # ``--no-messages``: a home folder's unreadable files would fill stderr, which is
        # read only at the end, and stall ripgrep until the deadline.
        argv = [rg, "--json", "--hidden", "--no-messages", "--glob", "!.git"]
        argv += ["--case-sensitive" if self.case else "--ignore-case"]
        argv += ["--word-regexp"] * self.word + ["--fixed-strings"] * (not self.regex)
        for glob in _globs(self.include):
            argv += ["--glob", glob]
        for glob in _globs(self.exclude):
            argv += ["--glob", f"!{glob}"]
        # Run in the folder: ripgrep anchors a glob with a slash (`src/**`) at its working
        # directory, where VS Code's fields mean the folder searched.
        process = subprocess.Popen(  # noqa: S603 -- A fixed ripgrep argv, no shell.
            [*argv, "--", self.query, "."],
            cwd=self.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Killing ripgrep ends its output, which ends the read below.
        threading.Thread(
            target=_kill_when,
            args=(process, self._stopped, self.deadline_sec),
            daemon=True,
        ).start()
        hits: dict[str, list[Hit]] = {}
        count = 0
        truncated = False
        try:
            for line in process.stdout or ():
                if count >= MAX_HITS or self._stopped.is_set():
                    truncated = True
                    break
                if not line.startswith(b'{"type":"match"'):
                    continue
                found = _rg_hit(line)
                if found is not None:
                    path = self.root + found[0].removeprefix(".")
                    hits.setdefault(path, []).append(found[1])
                    count += 1
        finally:
            process.kill()
            stderr = process.stderr.read() if process.stderr is not None else b""
            code = process.wait()
        # 2 is an error: an invalid regex, or a folder ripgrep cannot read at all. One
        # unreadable file below the folder is also 2, with what was found kept.
        if code == 2 and not hits and not truncated:
            raise SearchError(stderr.decode(errors="replace").strip() or "rg failed")
        return _searched("rg", hits, truncated=truncated or code < 0)

    def _python(self) -> Searched:
        """Search with ``knowop.lib.files.grep``'s Python backend."""
        pattern = self.query if self.regex else re.escape(self.query)
        if self.word:
            pattern = rf"\b(?:{pattern})\b"
        try:
            rows = grep_rows(
                Path(self.root),
                Query(
                    pattern=pattern,
                    glob=_brace(_globs(self.include)),
                    exclude=_brace(_globs(self.exclude)),
                    output_mode="content",
                    case_insensitive=not self.case,
                ),
            )
            compiled = re.compile(pattern, 0 if self.case else re.IGNORECASE)
        except (GrepError, re.error) as error:
            raise SearchError(str(error)) from error
        hits: dict[str, list[Hit]] = {}
        for row in rows[:MAX_HITS]:
            if row.column < 0:
                continue
            name, _, number = row.text[: row.prefix_end - 1].rpartition(":")
            text = row.text[row.prefix_end :]
            spans = [m.span() for m in compiled.finditer(text) if m.end() > m.start()]
            hits.setdefault(name, []).append(_hit(int(number), text, spans))
        return _searched("python", hits, truncated=len(rows) > MAX_HITS)


class _Text(BaseModel):
    """A path or a line as ripgrep writes it: ``text``, or base64 ``bytes``."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    text: str | None = None


class _Submatch(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    start: int
    end: int


class _Match(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    path: _Text
    lines: _Text
    line_number: int
    submatches: list[_Submatch]


class _MatchEvent(BaseModel):
    """One ``--json`` line of ``type: match``."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    data: _Match


def _rg_hit(line: bytes) -> tuple[str, Hit] | None:
    """Return the file and :class:`Hit` of one ``match`` line; ``None`` for one not UTF-8."""
    try:
        match = _MatchEvent.model_validate_json(line).data
    except ValidationError:
        return None
    if match.path.text is None or match.lines.text is None:
        return None
    # The offsets count bytes of the line as UTF-8.
    raw = match.lines.text.encode()
    spans = [
        (
            len(raw[: each.start].decode(errors="replace")),
            len(raw[: each.end].decode(errors="replace")),
        )
        for each in match.submatches
    ]
    return match.path.text, _hit(match.line_number, match.lines.text, spans)


def _hit(number: int, text: str, spans: list[tuple[int, int]]) -> Hit:
    """Cut a line to 400 characters around its first match; ranges in UTF-16 units."""
    text = text.rstrip("\r\n")
    start = 0
    if len(text) > 400:
        first = spans[0][0] if spans else 0
        start = max(0, min(first - 100, len(text) - 400))
        text = text[start : start + 400]
    kept = [
        (max(0, a - start), min(len(text), b - start))
        for a, b in spans
        if b > start and a < start + len(text)
    ]
    return Hit(
        line=number,
        text=text,
        ranges=[(_utf16(text, a), _utf16(text, b)) for a, b in kept],
    )


def _utf16(text: str, index: int) -> int:
    """Return the UTF-16 offset of the character at ``index``."""
    return len(text[:index].encode("utf-16-le")) // 2


def _searched(
    engine: Engine,
    hits: dict[str, list[Hit]],
    *,
    truncated: bool,
) -> Searched:
    """Gather hits by file, in path order."""
    return Searched(
        engine=engine,
        matches=sum(len(hit.ranges) for each in hits.values() for hit in each),
        truncated=truncated,
        files=[
            FileHits(path=path, hits=found)
            for path, found in sorted(hits.items(), key=lambda item: item[0].split("/"))
        ],
    )


def _globs(text: str) -> list[str]:
    """Split a comma-separated list of globs, as VS Code's fields take them."""
    return [glob.strip() for glob in text.split(",") if glob.strip()]


def _brace(globs: Iterable[str]) -> str:
    """Join globs into one brace set, the one glob ``Query`` takes."""
    joined = list(globs)
    if len(joined) < 2:
        return "".join(joined)
    return "{" + ",".join(joined) + "}"


def _kill_when(
    process: subprocess.Popen[bytes],
    stopped: threading.Event,
    deadline_sec: float,
) -> None:
    """Kill ``process`` once the search is stopped or its deadline passes."""
    _ = stopped.wait(deadline_sec)
    # A process already reaped is not signalled again (``Popen`` knows its status).
    process.kill()
