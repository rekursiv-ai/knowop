"""The trax context a host withholds from the terminals and commands it starts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from knowop.shell.trax_context import inherited_trax_context


if TYPE_CHECKING:
    import pytest


def test_the_routing_variables_and_every_trax_name_are_withheld(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "TRACKINIZER_URL",
        "TRACKINIZER_TOKEN",
        "TRACKINIZER_PROFILE",
        "TRAX_CANVAS",
        "LC_TRAX_SERVER",
        "PATH_FROM_THE_USER",
    ):
        monkeypatch.setenv(name, "x")

    withheld = inherited_trax_context()

    assert {
        "TRACKINIZER_URL",
        "TRACKINIZER_TOKEN",
        "TRACKINIZER_PROFILE",
        "TRAX_CANVAS",
        "LC_TRAX_SERVER",
    } <= withheld
    assert "PATH_FROM_THE_USER" not in withheld
    assert "HOME" not in withheld


def test_a_name_that_is_unset_is_not_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRACKINIZER_TOKEN", raising=False)

    assert "TRACKINIZER_TOKEN" not in inherited_trax_context()


if __name__ == "__main__":
    from knowop.lib.testing.runner import test_main

    test_main(__file__)
