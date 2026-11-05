"""Fixtures the agent tests share."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def local_session_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point both CLIs' session roots at temp dirs, and pin the codex version.

    Each adapter derives its own root, so isolating one leaves the other writing
    real rollouts into the operator's home, which its resume picker then offers
    and which makes a result depend on the machine. The pinned version keeps a
    test from running the installed codex.
    """
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setattr(
        "knowop.agent.materialize._codex_cli_version",
        lambda: "0.999.0",
    )
